"""STENCIL shared job glue — PG1 load, the `stencil|` splits (subjects + seeds + items), the MMLU
adaptation-subject roster, the bounded forgetting battery, and the adapt-and-eval driver (attach an
arm → train with a checkpoint trajectory → read task-gain + composite forgetting at each checkpoint).

Every measured surface reuses verified code (`inherit_pg1`, `relearn`, `text_eval`, `stencil_core`,
`_specter_shared`, `ration_core`); the NEW logic is entirely in `stencil_core`. Heavy imports inside
functions so the module py_compiles on the control plane. No try/except hiding (CLAUDE.md).

Journal amendments A-STENCIL-1..7 govern the choices here:
  * adaptation task = an MMLU SUBJECT (relearn QA); task-gain = held-out subject accuracy (mmlu_blankimg
    + item_filter). forgetting battery = { broad-MMLU · VQAv2 · COCO-CIDEr · wikitext-ppl } (A-1).
  * subject roster is data-driven + deterministic: subjects with ≥ MIN_TEST rows, split by name-hash
    into discovery / sealed-confirmation; a disjoint fixed set is the broad-MMLU forgetting probe.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from infra.aws_pg.jobs._regrain_shared import (  # noqa: F401
    S3_BUCKET,
    fetch,
    install_pg,
    pinned_pg1_sha,
)

D = 2048
N_LAYERS = 18
SEED = 1337

# ---- adaptation-subject roster (frozen selection rule) ------------------------------------------
MIN_TEST = 180                     # a subject enters the roster only if it has ≥ this many test rows
N_ADAPT = 8                        # adaptation subjects (split ~half discovery, half sealed-confirm)
TRAIN_FRAC_SALT = "stencil_train"  # per-subject train/eval-half split salt

# ---- bounded forgetting battery (sizes frozen on discovery; small for fleet economy, A-6) -------
N_VQA_BATTERY = 200                # VQAv2 items (generation-scored)
N_COCO_BATTERY = 64                # COCO caption items (CIDEr)
N_PPL_BATTERY = 120                # wikitext windows
BROAD_MMLU_SUBJECTS = 6            # held-out subjects for the broad-MMLU forgetting probe
BROAD_MMLU_PER_SUBJECT = 24        # items per broad subject
BATTERY_COMPONENTS = ("mmlu_broad", "vqav2", "coco_cider", "wikitext_ppl")

# ---- default training recipe (A-STENCIL-9: gentle lr — aggressive lr catastrophically destabilizes
# PG1 [composite forget 64-80]; gentle lr cleanly learns [+14pt gain]). Overridable per job. --------
RECIPE = {"lr": 5e-5, "epochs": 12, "batch_size": 16, "warmup_ratio": 0.1, "micro_batch": 8}
CHECKPOINT_FRACS = (0.25, 0.5, 0.75, 1.0)   # eval task-gain + forgetting at these fractions of training
# A-STENCIL-9: the forgetting composite is the MULTIMODAL collateral (capabilities orthogonal to text-
# MMLU adaptation); broad-MMLU transfers + text-ppl improves under MMLU fine-tuning, masking collateral.
MULTIMODAL_COMPONENTS = ("vqav2", "coco_cider")


# ===========================================================================
# Model + splits.
# ===========================================================================
def load_model(workdir="/tmp/stencil"):
    """PG1 pt-224, eager, bf16; the 18 decoder layers; the pinned sha + FUNC base_ppl anchor."""
    from infra.pg import inherit_pg1 as IP
    sha, base_ppl = pinned_pg1_sha(workdir)
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    assert len(layers) == N_LAYERS, f"expected 18 PG1 layers, got {len(layers)}"
    return model, proc, layers, sha, base_ppl


def snapshot_base(model):
    """A CPU deep-copy of the model state_dict — the clean reference every cell resets to (taken
    ONCE before any attach). ~6GB CPU; negligible on the GPU box."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def reset_model(model, snapshot):
    """Return the model to its snapshot: detach every adapter (restores the frozen base Linears),
    load the snapshot state_dict (undoes full-FT), freeze all params, eval mode. The fleet-wide
    between-cell reset — no re-load, exact restoration."""
    from infra.pg import stencil_core as SC
    SC.detach_adapters(model)
    missing, unexpected = model.load_state_dict(snapshot, strict=True)
    assert not missing and not unexpected, f"reset load not clean: missing={missing} unexpected={unexpected}"
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def _hash_bucket(s, salt, mod=2):
    return int(hashlib.sha1(f"{salt}|{s}".encode()).hexdigest(), 16) % mod


def mmlu_by_subject():
    """{subject: {'dev': [5-shot rows], 'test': [rows]}} from cais/mmlu 'all' (sorted subjects)."""
    from collections import defaultdict
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all")
    dev, test = defaultdict(list), defaultdict(list)
    for r in ds["dev"]:
        dev[r["subject"]].append(r)
    for r in ds["test"]:
        test[r["subject"]].append(r)
    return {s: {"dev": dev[s], "test": test[s]} for s in sorted(test)}


def build_roster(mmlu=None):
    """Deterministic subject roster: eligible subjects = those with ≥ MIN_TEST test rows and ≥5 dev
    shots, ordered LARGEST-first (then name) so the roster is robust to the exact threshold. Adaptation
    subjects = the N_ADAPT largest; split by name-hash into discovery / sealed-confirm. broad-MMLU
    probe = the next BROAD_MMLU_SUBJECTS eligible, DISJOINT from adaptation. Fail-loud if the pool is
    too small (never silently shrink the design)."""
    mmlu = mmlu or mmlu_by_subject()
    eligible = sorted([s for s in mmlu if len(mmlu[s]["test"]) >= MIN_TEST and len(mmlu[s]["dev"]) >= 5],
                      key=lambda s: (-len(mmlu[s]["test"]), s))
    assert len(eligible) >= N_ADAPT + BROAD_MMLU_SUBJECTS, \
        f"only {len(eligible)} eligible MMLU subjects (need {N_ADAPT + BROAD_MMLU_SUBJECTS})"
    adapt = eligible[:N_ADAPT]
    discovery = [s for s in adapt if _hash_bucket(s, "stencil_subject") == 0]
    confirm = [s for s in adapt if _hash_bucket(s, "stencil_subject") == 1]
    # guarantee both non-empty (rebalance deterministically if a bucket is empty)
    if not discovery or not confirm:
        discovery, confirm = adapt[::2], adapt[1::2]
    broad = [s for s in eligible if s not in adapt][:BROAD_MMLU_SUBJECTS]
    # the broad-MMLU forgetting-probe allowlist: first BROAD_MMLU_PER_SUBJECT item hashes per broad
    # subject (a fixed, bounded held-out probe — restricts the forward without touching adapted subjects)
    from infra.pg.relearn import qa_item_hash
    broad_hashes = sorted({qa_item_hash(r["question"])
                           for s in broad for r in mmlu[s]["test"][:BROAD_MMLU_PER_SUBJECT]})
    return {"adapt": adapt, "discovery": discovery, "confirm": confirm, "broad_mmlu": broad,
            "broad_hashes": broad_hashes, "n_eligible": len(eligible),
            "test_counts": {s: len(mmlu[s]["test"]) for s in adapt + broad}}


def subject_train_eval(subject_rows, salt=TRAIN_FRAC_SALT):
    """Split ONE subject's test rows into (train_rows, eval_rows) by item-hash bucket (even=train,
    odd=eval) — deduped, disjoint, deterministic. train_rows fine-tune the adapter; eval_rows measure
    task-gain (never trained on). Returns (train, eval, eval_hashes)."""
    from infra.pg.relearn import qa_item_hash
    train, evl, ev_hashes = [], [], []
    for r in subject_rows:
        h = qa_item_hash(r["question"])
        if int(hashlib.sha1(f"{salt}|{h}".encode()).hexdigest(), 16) % 2 == 0:
            train.append(r)
        else:
            evl.append(r)
            ev_hashes.append(h)
    assert len(train) >= 40 and len(evl) >= 40, f"subject too small: train {len(train)}, eval {len(evl)}"
    return train, evl, ev_hashes


def splittable(subject_rows, salt=TRAIN_FRAC_SALT, minhalf=40):
    """True if a subject's rows split into non-degenerate train/eval halves. Some MMLU subjects
    (moral_scenarios/moral_disputes) share a fixed 64-char question prefix → qa_item_hash collides for
    every row → both the split AND the item_filter degenerate. Such subjects are skipped."""
    from infra.pg.relearn import qa_item_hash
    tr = ev = 0
    for r in subject_rows:
        h = qa_item_hash(r["question"])
        if int(hashlib.sha1(f"{salt}|{h}".encode()).hexdigest(), 16) % 2 == 0:
            tr += 1
        else:
            ev += 1
    # also require the item hashes to be DISTINCT enough that the eval filter can separate items
    n_distinct = len({qa_item_hash(r["question"]) for r in subject_rows})
    return tr >= minhalf and ev >= minhalf and n_distinct >= minhalf


def train_examples(subject_rows, eval_hashes):
    """relearn.format_qa examples from a subject's train rows, deduped against the eval-half hashes
    (contamination firewall). Returns (examples, dedup_audit)."""
    from infra.pg import relearn as RL
    audit = RL.dedup_against_eval(subject_rows, eval_hashes)
    return [RL.format_qa(r["question"], r["choices"], r["answer"]) for r in audit["kept"]], audit


# ===========================================================================
# Task-gain (held-out subject accuracy) + the bounded forgetting battery.
# ===========================================================================
def _subject_item_filter(eval_hashes):
    ev = set(eval_hashes)
    return lambda h: h in ev


def subject_accuracy(model, proc, device, eval_hashes, n_per_subject=None):
    """Held-out accuracy on ONE subject's eval-half items (mmlu_blankimg + item_filter to the eval
    hashes) → {acc, per_item, per_item_hash}. The task-gain read (adapted − base on the same items)."""
    from infra.pg import text_eval as TE
    r = TE.mmlu_blankimg(model, proc, device, item_filter=_subject_item_filter(eval_hashes),
                         n_per_subject=n_per_subject)
    return {"acc": r["acc"], "per_item": r["per_item"], "per_item_hash": r["per_item_hash"], "n": r["n"]}


def broad_mmlu_accuracy(model, proc, device, broad_hashes):
    """Broad-MMLU forgetting probe: accuracy over the fixed held-out allowlist (the broad subjects'
    bounded item hashes) via mmlu_blankimg + item_filter — forwards ONLY the ~BROAD*PER items, not
    all 57 subjects. Per-item aligned across base and every arm (the pairing contract)."""
    from infra.pg import text_eval as TE
    allow = set(broad_hashes)
    r = TE.mmlu_blankimg(model, proc, device, item_filter=lambda h: h in allow)
    return {"acc": r["acc"], "per_item": r["per_item"], "per_item_hash": r["per_item_hash"], "n": r["n"]}


def battery_probe(model, proc, device, bctx):
    """One forgetting-battery read → {mmlu_broad, vqav2, coco_cider, wikitext_ppl}. Reused for the
    base reference and every adapted cell (identical items/order across arms — the pairing contract).
    `bctx` = the battery context from `make_battery_ctx` (tasks, rows, broad_hashes, sizes)."""
    from infra.aws_pg.jobs._specter_shared import safe_ppl_gate
    from infra.pg import inherit_pg1 as IP
    out = {}
    out["mmlu_broad"] = broad_mmlu_accuracy(model, proc, device, bctx["broad_hashes"])["acc"]
    out["vqav2"] = IP.eval_pg1(model, proc, bctx["vqa_task"], bctx["vqa_rows"][:bctx["n_vqa"]], device)["score"]
    out["coco_cider"] = IP.eval_pg1(model, proc, bctx["coco_task"], bctx["coco_rows"][:bctx["n_coco"]], device)["score"]
    out["wikitext_ppl"] = safe_ppl_gate(model, proc, device, n_seq=bctx["n_ppl"])["ppl"]
    return out


def make_battery_ctx(proc, roster, n_vqa=N_VQA_BATTERY, n_coco=N_COCO_BATTERY, n_ppl=N_PPL_BATTERY):
    """Assemble the frozen battery context (tasks + bounded rows + broad allowlist + sizes) ONCE per
    job so every base/cell read is on identical items."""
    from infra.pg import inherit_pg1 as IP
    vqa_task, coco_task = IP.pg1_tasks()
    vqa_rows, _ = IP.load_items(vqa_task, n_vqa)
    coco_rows, _ = IP.load_items(coco_task, n_coco)
    return {"vqa_task": vqa_task, "vqa_rows": vqa_rows, "coco_task": coco_task, "coco_rows": coco_rows,
            "broad_hashes": roster["broad_hashes"], "n_vqa": n_vqa, "n_coco": n_coco, "n_ppl": n_ppl}


# ===========================================================================
# The adapt-and-eval driver — attach an arm, train with a checkpoint trajectory, read the frontier.
# ===========================================================================
def adapt_and_eval(model, proc, device, *, arm, banks, rank, alpha, subject_rows,
                   eval_hashes, bctx, base_battery, base_subj_acc, weights,
                   recipe=None, seed=SEED, checkpoint_fracs=CHECKPOINT_FRACS):
    """Attach `arm` at `rank`, fine-tune on the subject's train examples, and at each checkpoint read
    task-gain (subject acc − base) + composite forgetting (battery vs base). Returns
    {arm, rank, trajectory:[{step, task_gain, forgetting, per_component, subj_acc, battery}], ...}.

    `banks` = {layer: layer_bank} (ignored by ref arms). `bctx` = battery context (make_battery_ctx).
    `base_battery`/`base_subj_acc` are the frozen base references (measured ONCE before adaptation)."""
    from infra.pg import stencil_core as SC
    recipe = recipe or RECIPE
    examples, dedup = train_examples(subject_rows, eval_hashes)

    attach = SC.attach_arm(model, arm, banks, rank, alpha, seed=seed)

    n_batches = max(1, (len(examples) + recipe["batch_size"] - 1) // recipe["batch_size"])
    total = recipe["epochs"] * n_batches
    cps = sorted(set(max(1, int(round(f * total))) for f in checkpoint_fracs))

    def _eval(m, step):
        sa = subject_accuracy(m, proc, device, eval_hashes)
        batt = battery_probe(m, proc, device, bctx)
        comp = SC.composite_forgetting(base_battery, batt, weights)
        return {"task_gain": sa["acc"] - base_subj_acc, "subj_acc": sa["acc"],
                "forgetting": comp["composite"], "per_component": comp["per_component"], "battery": batt}

    stats = SC.train_with_trajectory(
        model, proc, device, examples, lr=recipe["lr"], epochs=recipe["epochs"],
        batch_size=recipe["batch_size"], warmup_ratio=recipe["warmup_ratio"], seed=seed,
        max_steps=total, micro_batch=recipe.get("micro_batch"), checkpoint_steps=cps, eval_fn=_eval)
    return {"arm": arm, "rank": rank, "alpha": alpha,
            "attach": {k: attach[k] for k in attach if k not in ("names", "basis_ids")},
            "basis_ids": attach.get("basis_ids"), "degenerate": attach.get("degenerate", []),
            "dedup": dedup, "n_examples": len(examples), "total_steps": total,
            "trajectory": stats["trajectory"], "train_stats": {k: stats[k] for k in
            ("steps", "final_loss", "mean_loss_last_epoch", "tokens_seen")}}


# ===========================================================================
# Freeze mechanics (STENCIL's own tag + spec sha256).
# ===========================================================================
_JOBS_DIR = pathlib.Path(__file__).resolve().parent
SPEC_FROZEN = _JOBS_DIR / "stencil_spec_frozen.md"
FREEZE_MANIFEST = _JOBS_DIR / "stencil_freeze.json"


def assert_frozen(arm):
    """Driver half of freeze-or-refuse (the _specter_shared.assert_frozen contract, STENCIL tags)."""
    assert FREEZE_MANIFEST.exists(), f"no freeze manifest — arm '{arm}' refuses to launch"
    man = json.loads(FREEZE_MANIFEST.read_text())
    assert arm in man, f"arm '{arm}' has no freeze entry — refuses to launch (no retroactive tags)"
    assert SPEC_FROZEN.exists(), "frozen spec copy missing from the packaged tree"
    h = hashlib.sha256(SPEC_FROZEN.read_bytes()).hexdigest()
    assert h == man[arm]["spec_sha256"], (
        f"spec hash mismatch for arm '{arm}': tree={h[:12]} manifest={man[arm]['spec_sha256'][:12]}")
    return man[arm]
