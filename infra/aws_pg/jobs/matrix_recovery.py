"""MENDELEEV N8 — recovery job: REC / fair-scratch / healthy-trained / REH / permuted-label for one
(row, recipe, arm), with the GB, SUB and IM sub-modes. Contract `infra/pg/MATRIX_CONTRACT.md` §7; spec
§1.2 (Recovery, GB, SUB, IM), §5, §6.3, §6.6, §6.8, §6.13. Pure helpers: `infra/pg/matrix_recovery_core.py`.
Verdicts are the lead's, computed OFFLINE (`scripts/matrix_verdicts.py`) from what this job banks — nothing
here decides; the only in-job thresholds are the GB trigger (O_lost ≥ 0.3, spec §1.2) and the D18 gate.

  python -m infra.aws_pg.jobs.matrix_recovery --row 2 --recipe r512 --arm rec --doses 250,1000,4000,16000 \
      --behaviour-metrics <local path | s3://…/out/metrics.json> --out DIR [--gb] [--sub --unrelated-adapter P]
      [--smoke] [--adam-offload]
  python -m infra.aws_pg.jobs.matrix_recovery --row 2 --im all|mmlu_letter,vqav2,… --behaviour-metrics … --out DIR
  python -m infra.aws_pg.jobs.matrix_recovery --row 7a --recipe r512 --arm rec --behaviour-metrics … --out DIR
  python -m infra.aws_pg.jobs.matrix_recovery --dryrun

Unlearning rows (7a/7b, spec §6.10): train items = matrix_items.wmdp_items("train") (bio + cyber, QA format,
relearn.ladder_subset order), doses {250, 500, 1000}, the same recipe; per dose S_W (0-shot letter on the
WMDP eval split — text_eval.wmdp_choice_logits semantics through the Forward), K_W (0-shot option-text
cloze on the same items), P_W (ridge-logistic probe per probe layer, trained on the current model's
WMDP-train residuals, tested on WMDP-eval, label-shuffle null beside) and MMLU S/K (the retained skill).
Dose-0 identity covers S/K (mandatory) and S_W/K_W when the behaviour cell banks them (else "ABSENT").
REH/GB/SUB/IM and the permuted control are not defined on these rows (spec §3.2, §6.13) — refused.

Order of a run: manifest → dose-0 readouts of the DAMAGED model (S 5-shot, K, O on both halves) → the D18
identity gate against the behaviour cell's `after` vectors (aligned by per_item_hash; ≥ 99 % per item and
|Δacc| ≤ 0.01, else IdentityMismatch before any training) → per dose: fresh adapter (LoRA, alpha = r,
seed 0) or fresh copy (full-FT: the pre-dose snapshot restored), train 2 epochs over the dose (batch 16,
5 % warmup, answer-token CE, per-step curve), save the weights, evaluate S/K on both halves (held-out
subjects split out) and O, bank the S_rec / K_rec / O_lost inputs, [SUB], [GB at the first dose with
O_lost ≥ 0.3], write `metrics.json` (`kind: "recovery"`, incremental: a crash after 4k keeps 250/1k/4k).

Arms: rec = the damaged model; scratch = the damaged shell with its language decoder re-initialised at the
healthy decoder's per-tensor scale (the FAIRSCRATCH rule); healthy = the healthy before-model (teaching
ceiling); reh = 50/50 MMLU + VQAv2-train per step (dose n = n MMLU items + n VQAv2-train items, so the MMLU
exposure matches the plain arm and the step count doubles); permuted = relearn.shuffle_labels on the 4k
MMLU items (the H-2 recipe). Training forward for MMLU pools is TEXT-ONLY (the W17 / C14 recipe); S and K
are READ through the row's shell (blank image on VLM rows). VQAv2 pools use the processor's suffix path.

Seams (modules owned by other agents; imported lazily; ASSERTED against the real interface, never assumed):
  matrix_models.load_pair(row, device, which=("before","after")) -> ONE MEMBER DICT PER SIDE,
      {"before": member, "after": member}; member = {"fwd","model","tok","proc","repo","which","shell",
      "info","local"} (`matrix_models.assert_member`). It is NOT an object with attributes, and it never
      was: until 2026-09-03 this job asserted nine attributes on it (.before .after .tokenizer .processor
      .shell .forward .lm .manifest .o_tasks) and called it with the DEFAULT `which`, so the first real
      recovery cell would have put a model on the GPU and then died. `_bind_pair` is now the ONLY place
      this job reads the loader's return, and `_stub_context` goes through the same real `load_pair`.
      Context.forward(model) = `matrix_models.forward_for(member, model)` — the member's shell/tokenizer/
      device over a model the loader never returned (the fair-scratch shell, a graft-back donor copy);
      Context.lm(model) = `matrix_forward.lm_anatomy(model)["lm"]` (the adapter scope); o_tasks =
      `matrix_models.o_tasks(row)`; the model manifest is built by `_model_manifest` (see it for what
      B1/B3/B4 this job computes and what it carries by reference).
  The O readout, per shell — `o_record(shell, fwd, col, rows)`, the ONE place this job builds an O
      record. It re-runs the BEHAVIOUR CELL'S OWN CODE, because the D18 identity gate pairs the two per
      item at dose 0 and demands ≥ 99 % agreement: same prompts, same images, same distractor draw, or
      the gate is comparing two instruments instead of two models.
        molmo2     `vlm_skill_eval.evaluate(col, fwd, rows, prompts)` over `vlm_skill_eval.molmo2_prompts()`
                   through a `vlm_skill_eval.Molmo2SkillForward` (its `_encode` binds the item's image).
        paligemma  `matrix_behaviour.cand_column(fwd, task, rows, hashes)` through a `PaliGemmaForward`.
      FIXED 2026-09-03 (SEAMS audit): this job called `vlm_skill_eval.readout(task, call, processor,
      rows, batch_size, device, generate=...)`, a function that does not exist and never did — the
      module's dispatcher is `evaluate(col, fwd, rows, prompts)` and its candidate columns are named
      `<task>_cand`. Nothing caught it because the dry-run supplied its own O readout and never touched
      the module. `_stub_context` now goes through the real `o_record` too.
      `call` is a model or a broca_sweep.PatchedLM (GB); a fresh Forward is built over the AFTER
      member's shell and its `.model` swapped to `call` — the construction `matrix_models.forward_for`
      documents. GENERATION READOUTS ARE NOT RE-RUN HERE (see `O_VERDICT_TASK`).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import time

from infra.pg import matrix_recovery_core as RC

HALVES = ("discovery", "confirmation")
IM_TRAIN_TASKS = ("mmlu_letter", "vqav2", "triviaqa_cloze", "arith")   # the tasks with a train source; refcoco measured only
IM_RECIPE = "r512"
UNLEARN_DOSES = (250, 500, 1000)   # spec §5: the WMDP ladder on rows 7 (items, D31)
GB_BATCH = 4                      # graft evaluations at the identity-gate batch (spec §11.5, D22)
# Spec §1.2 (v2.3): for EVERY verdict that consumes O — H_repair … H_capacity, the graft-back rule and
# the interference matrix — O := the VQAv2 CANDIDATE-LIKELIHOOD accuracy, exact match reported beside it
# and never scored (D4: exact match collapses under answer-format drift; CARREL watched a vision score
# go 0.31 → 0.00 that way). The banked key for that readout is `vqav2_cand` on BOTH shells
# (`matrix_behaviour.O_VERDICT_KEY`, asserted in the dry-run). It was `"vqav2"` here until 2026-09-03,
# a key no behaviour record has ever held: PaliGemma banks `vqav2_em` / `vqav2_cand` and Molmo2 banks
# `vqav2` as its SOFT-ACCURACY generation readout, so the GB trigger read a KeyError and the identity
# gate silently stamped "ABSENT in behaviour cell".
O_VERDICT_TASK = "vqav2_cand"
# Which O columns the ladder re-measures per dose: the candidate columns ONLY. The generation readouts
# (RefCOCO, VQAv2/TextVQA exact match or soft accuracy) stay behaviour-cell columns, measured once per
# row — 200 greedy generations per task per model per DOSE per arm is not affordable in a ladder, no
# verdict reads them, and the spec's "reported beside" obligation is discharged where they are measured.
O_COL_SUFFIX = "_cand"
SMOKE_O_ITEMS = 24                # ≥ 20: matrix_core.preflight_match's floor for the O identity check
REPO = pathlib.Path(__file__).resolve().parents[3]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _git_sha() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else os.environ.get("GIT_SHA", "unavailable (no .git on the box)")


def _take(pool: list, n: int) -> list:
    assert n <= len(pool), f"dose {n} exceeds the pool ({len(pool)} items)"
    return pool[:n]


# ============================================================================================ context
class Context:
    """Everything the runner touches. `_real_context` binds weights + data, `_stub_context` the dry-run.
    row, arm, recipe (RC.recipe dict), recipes_sha, device, smoke, n_boot, micro_batch, seed, adam_offload,
    before, after, tok, proc (None on text rows), shell, lm(model), forward(model) — all bound by
    `_bind_pair` from the loader's member dicts —
    batch_size, eval_items{half}, dev_shots, held_out,
    train_rows(n), vqa_examples(n) | None, encode_text(ex), encode_vlm(ex), im_examples(task), o_tasks,
    o_readout(col, call, batch_size), arith (items, exemplars),
    tq (items, exemplars), behaviour, model_manifest, item_manifest, remap_before_sd(sd), probe(fwd).

    `o_tasks` holds O COLUMN names (`vqav2_cand`, `textvqa_cand`) — the keys the behaviour cell banks
    and the keys the verdicts read — not the task names `matrix_models.o_tasks(row)` returns."""

    def forward_at(self, model):
        fwd = self.forward(model)
        fwd.batch_size = self.batch_size
        return fwd


def o_columns(row: str) -> list:
    """The O COLUMNS this job re-measures for `row`: the candidate-likelihood column of each task in the
    registry's O instrument. `[]` (a statement, not a gap) where the row has no per-item vision O."""
    from infra.pg import matrix_models as MM
    return [t + O_COL_SUFFIX for t in MM.o_tasks(row)]


def o_task_of(col: str) -> str:
    assert col.endswith(O_COL_SUFFIX), f"{col} is not an O candidate column"
    return col[:-len(O_COL_SUFFIX)]


# Spec §12.11: "a record without real-inference samples is incomplete and its numbers are not quotable."
# NO O producer banks `extra.samples` (2026-09-03 audit) — not `matrix_behaviour._cand_record`, not the
# three generation records of `matrix_behaviour._o_column`, not `vlm_skill_eval.{refcoco,vqa,vqa_cand}` —
# so every O number in the program, the own-skill loss four recovery hypotheses consume included, is
# currently unquotable. The fix is one line inside each of those builders (each already holds the prompts,
# the per-item predictions, the candidates and their scores, the golds and the per-item correctness that
# `format_readouts._samples` wants); none of them is this job's to edit. Until they do it, every O record
# this job banks carries the stamp below, so the gap travels WITH the number instead of being invisible,
# and `scripts/matrix_verdicts.py` refuses to present such a number as quotable.
O_SAMPLES_MISSING = ("NOT QUOTABLE (spec §12.11): this readout's producer banks no `extra.samples`, so no "
                     "real-inference sample stands behind the number. Fix at the producer — "
                     "matrix_behaviour._cand_record / _o_column, vlm_skill_eval.{refcoco,vqa,vqa_cand} — "
                     "with format_readouts._samples(prompts, per_item, pred, golds, scores, options=…, "
                     "decoded=… on the generation readouts).")


def o_record(shell: str, fwd, col: str, rows: list, hashes=None) -> dict:
    """ONE O record for `col` through the SAME code the behaviour cell used for this shell (D18 pairs
    the two per item at dose 0 — see the module docstring). `fwd` is the shell's Forward with `.model`
    already pointing at the model or PatchedLM being scored.

    The §12.11 stamp is attached HERE and nowhere else in this job: this is the one place an O record is
    built, and the stamp is a statement about the record, not a second implementation of the readout. It
    disappears by itself the day a producer banks samples."""
    task = o_task_of(col)
    if shell == "molmo2":
        from infra.pg import vlm_skill_eval as V
        rec = V.evaluate(col, fwd, rows, V.molmo2_prompts())
    else:
        assert shell == "paligemma", f"no O readout for shell {shell!r} (rows with an O column are PaliGemma or Molmo2)"
        from infra.aws_pg.jobs import matrix_behaviour as MB
        rec = MB.cand_column(fwd, task, rows, hashes)
    extra = rec.setdefault("extra", {})
    if not extra.get("samples"):
        extra["samples_missing_12_11"] = O_SAMPLES_MISSING
    return rec


def _load_behaviour(path: str, out: pathlib.Path) -> dict:
    if path.startswith("s3://"):
        dst = out / "behaviour_metrics.json"
        subprocess.run(["aws", "s3", "cp", path, str(dst), "--region", "us-east-1", "--only-show-errors"], check=True)
        path = str(dst)
    p = pathlib.Path(path)
    assert p.exists(), f"behaviour metrics not found: {path}"
    beh = json.loads(p.read_text())
    assert beh.get("kind", "behaviour") == "behaviour" and "cols" in beh, "not a behaviour-cell metrics.json"
    for col in ("S", "K"):
        for model in ("before", "after"):
            for half in HALVES:
                assert half in beh["cols"][col][model], f"behaviour cell lacks cols.{col}.{model}.{half}"
    return beh


def _align_ref(ref: dict, cur_hashes: list) -> dict:
    """The behaviour cell's record restricted + reordered to the current hash sequence. Unique hashes
    → by hash (the launcher contract); duplicated hashes (the sha1(question[:64]) collision trap) → by
    position (equal length, or the smoke stride: one item per subject). Hash equality asserted."""
    rh = list(ref["per_item_hash"])
    cur = list(cur_hashes)
    if len(set(rh)) == len(rh):
        pos = {h: i for i, h in enumerate(rh)}
        missing = [h for h in cur if h not in pos]
        assert not missing, f"{len(missing)} current items absent from the behaviour cell (first {missing[:3]})"
        idx = [pos[h] for h in cur]
    else:
        assert len(rh) % len(cur) == 0, f"reference n={len(rh)} not positionally alignable to n={len(cur)}"
        idx = list(range(0, len(rh), len(rh) // len(cur)))
    sub = {"per_item": [ref["per_item"][i] for i in idx], "per_item_hash": [rh[i] for i in idx]}
    assert sub["per_item_hash"] == cur, "aligned reference hashes differ from the current readout's"
    return sub


def _beh_record(beh: dict, col: str, which: str):
    """The behaviour cell's WMDP record for `col` (`S_W` / `K_W`), banked either as a bare record or as
    {name: record} with one entry; None when the cell does not carry it."""
    node = beh["cols"].get(col, {}).get(which)
    if node is None:
        return None
    if "per_item_hash" in node:
        return node
    assert len(node) == 1, f"behaviour cols.{col}.{which} has {len(node)} entries; expected one WMDP eval record"
    return next(iter(node.values()))


def _beh_o(beh: dict, which: str) -> dict:
    """The behaviour cell's O records for one member, {column name: record}.

    O is HALF-INDEPENDENT, so `matrix_behaviour` banks it as `cols.O.<member>.all` — one more level
    than S/K's `cols.<col>.<member>.<half>`. This job read `cols.O.<member>` directly and asked whether
    `"vqav2"` was a key of it; the only key there is `"all"`, so EVERY O identity check would have
    stamped "ABSENT in behaviour cell" and passed — a gate that cannot fire (2026-09-03 SEAMS audit).
    `{}` when the cell banked no O at all, which is a real state the caller states out loud."""
    node = beh["cols"].get("O", {}).get(which)
    if not node:
        return {}
    assert "all" in node, f"behaviour cols.O.{which} has keys {sorted(node)}; O is banked under 'all'"
    return node["all"]


def identity_gate(ctx: Context, ev0: dict, which: str = "after") -> dict:
    """D18: dose-0 S/K per half (and O per task / S_W, K_W when the behaviour cell carries them) vs the
    behaviour cell's `after` vectors (`before` for the healthy-trained substrate). Raises
    RC.IdentityMismatch — never caught."""
    gate = {}
    for col in ("S", "K"):
        for half in HALVES:
            ref = _align_ref(ctx.behaviour["cols"][col][which][half], ev0[col][half]["per_item_hash"])
            gate[f"{col}:{half}"] = RC.identity_check(ref, ev0[col][half])
    if ctx.unlearn:
        for col in ("S_W", "K_W"):
            ref = _beh_record(ctx.behaviour, col, which)
            gate[f"{col}:eval"] = (RC.identity_check(_align_ref(ref, ev0[col]["eval"]["per_item_hash"]), ev0[col]["eval"])
                                   if ref is not None else "ABSENT in behaviour cell — not checked (stated)")
    o_after = _beh_o(ctx.behaviour, which)
    for col in ctx.o_tasks:
        if col in o_after:
            gate[f"O:{col}"] = RC.identity_check(_align_ref(o_after[col], ev0["O"][col]["per_item_hash"]), ev0["O"][col])
        else:
            gate[f"O:{col}"] = f"ABSENT in behaviour cell (has {sorted(o_after)}) — not checked (stated)"
    return gate


# ============================================================================================ readouts
def evaluate(ctx: Context, model, fwd, with_o: bool = True) -> dict:
    """S (5-shot) and K on both halves with the held-out / trained-subject split, plus O per task."""
    from infra.pg import format_readouts as FR
    out = {"S": {}, "K": {}, "held_out": {}}
    held = set(ctx.held_out)
    for half in HALVES:
        items = ctx.eval_items[half]
        out["S"][half] = FR.S(fwd, items, 5, dev_shots=ctx.dev_shots)
        out["K"][half] = FR.K(fwd, items, 5, dev_shots=ctx.dev_shots)
        trained = {it["subject"] for it in items} - held
        out["held_out"][half] = {col: {"held_out": RC.subset_by_subject(items, out[col][half], held),
                                       "trained_subjects": RC.subset_by_subject(items, out[col][half], trained)}
                                 for col in ("S", "K")}
    if with_o:
        out["O"] = {col: ctx.o_readout(col, model, ctx.batch_size) for col in ctx.o_tasks}
    if ctx.unlearn:
        out.update(wmdp_readouts(ctx, fwd))
    return out


def _wmdp_harvest(fwd, items: list, layers: list) -> dict:
    """Final-token residuals of the 0-shot S_W prompt (probe_readout.harvest with the WMDP prompt)."""
    import torch
    from infra.pg.text_eval import _q_block
    prompts = [_q_block(it["question"], it["choices"]) for it in items]
    X = {L: [] for L in layers}
    step = max(fwd.batch_size * 8, 8)
    for s in range(0, len(prompts), step):
        res = fwd.final_residuals(prompts[s:s + step], layers)
        for L in layers:
            X[L].append(res[L])
    return {"X": {L: torch.cat(v, 0) for L, v in X.items()},
            "y": torch.tensor([int(it["answer"]) for it in items], dtype=torch.long), "n": len(items)}


def wmdp_readouts(ctx: Context, fwd) -> dict:
    """Rows 7 (spec §6.10): S_W = 0-shot letter on the WMDP eval split (text_eval.wmdp_choice_logits
    semantics: _q_block(question, choices) → ' A'..' D'); K_W = 0-shot option-text cloze on the same items
    (_q_block_cloze(question), per-token mean, 4-way argmax); P_W = ridge-logistic probe per probe layer
    (⌊n/3⌋, ⌊2n/3⌋) trained on THIS model's WMDP-train residuals, tested on WMDP-eval, label-shuffle null
    beside (probe_readout estimator). Records carry per_item_hash for offline pairing."""
    from infra.pg import format_readouts as FR
    from infra.pg import probe_readout as PR
    from infra.pg.text_eval import _q_block, _q_block_cloze
    items = ctx.wmdp_eval
    hashes, golds = [it["hash"] for it in items], [it["answer"] for it in items]
    s_rec = FR._letter_readout(fwd, "S_W", [_q_block(it["question"], it["choices"]) for it in items],
                               fwd.letter_ids(), golds, hashes, {"shots": 0})
    k_rec = FR._cloze_readout(fwd, "K_W", [_q_block_cloze(it["question"]) for it in items],
                              [it["choices"] for it in items], golds, hashes, {"shots": 0})
    layers = PR.probe_layers(fwd.n_layers)
    tr, ev = _wmdp_harvest(fwd, ctx.wmdp_probe_train, layers), _wmdp_harvest(fwd, items, layers)
    pw = {"layers": layers, "decisive_layer": layers[0], "n_train": tr["n"], "n_eval": ev["n"], "per_layer": {}}
    for L in layers:
        probe = PR.fit_probe(tr["X"][L], tr["y"])
        rec = PR.evaluate(probe, ev["X"][L], ev["y"])
        sh = PR.shuffle_null(tr["X"][L], tr["y"], ev["X"][L], ev["y"])
        pw["per_layer"][str(L)] = {**rec, "per_item_hash": hashes, "shuffle_acc": sh["acc"],
                                   "n_iter": probe["n_iter"], "floor": 0.25}
    return {"S_W": {"eval": s_rec}, "K_W": {"eval": k_rec}, "P_W": pw}


def evaluate_im(ctx: Context, model, fwd) -> dict:
    """All IM-measured tasks: S, K (both halves), O per task, A, TQ."""
    from infra.pg import format_readouts as FR
    out = evaluate(ctx, model, fwd)
    out["A"] = FR.A(fwd, ctx.arith[0], ctx.arith[1])
    out["TQ"] = FR.TQ(fwd, ctx.tq[0], ctx.tq[1])
    return out


def fractions(ctx: Context, dose0: dict, ev: dict) -> dict:
    """S_rec / K_rec inputs (behaviour-cell before, dose-0 after, this dose) and O_lost vs the damaged model."""
    out = {"S_rec": {}, "K_rec": {}, "O_lost": {}}
    for col in ("S", "K"):
        for half in HALVES:
            ref_b = _align_ref(ctx.behaviour["cols"][col]["before"][half], ev[col][half]["per_item_hash"])
            out[f"{col}_rec"][half] = RC.rec_fraction(ref_b["per_item"], dose0[col][half]["per_item"],
                                                      ev[col][half]["per_item"], ctx.n_boot)
    for col in ctx.o_tasks:
        out["O_lost"][col] = RC.lost_fraction(dose0["O"][col]["per_item"], ev["O"][col]["per_item"],
                                              ev["O"][col]["floor"], ctx.n_boot)
    if ctx.unlearn:                       # S_W / K_W recovered vs the PARENT (behaviour `before`) when banked
        for col in ("S_W", "K_W"):
            ref = _beh_record(ctx.behaviour, col, "before")
            cur = ev[col]["eval"]
            out[f"{col}_rec"] = (RC.rec_fraction(_align_ref(ref, cur["per_item_hash"])["per_item"],
                                                 dose0[col]["eval"]["per_item"], cur["per_item"], ctx.n_boot)
                                 if ref is not None else "ABSENT (parent record not in the behaviour cell)")
        out["P_W_acc"] = {L: r["acc"] for L, r in ev["P_W"]["per_layer"].items()}
    return out


# ============================================================================================ training
def _letter_pool(rows: list) -> list:
    from infra.pg import relearn as R
    out = []
    for r in rows:
        e = R.format_qa(r["question"], r["choices"], r["answer"])
        out.append({"prompt": e["prompt"], "target_text": e["target_text"], "hash": e["hash"]})
    return out


def build_pools(ctx: Context, dose: int):
    """(pools, encoders, per_pool) for one dose of the arm's data."""
    from infra.pg import relearn as R
    rows = ctx.train_rows(dose)
    if ctx.arm == "permuted":
        rows = R.shuffle_labels(rows, seed=1337)
    pools, encoders = [_letter_pool(rows)], [ctx.encode_text]
    if ctx.arm == "reh":
        pools.append(ctx.vqa_examples(dose))
        encoders.append(ctx.encode_vlm)
        return pools, encoders, ctx.recipe["batch"] // 2
    return pools, encoders, ctx.recipe["batch"]


def train_dose(ctx: Context, model, lm, dose, pools, encoders, per_pool, curve_path) -> dict:
    """Fresh adapter (LoRA) or the full-FT scope on the current weights; the frozen-recipe trainer."""
    import torch
    rcp = ctx.recipe
    if rcp["key"] == "full":
        params = [p for _, p in ctx.ft_named]
        info = {"scope": "full-FT: decoder + norms (embed/lm_head excluded — the LoRA scope + 1-D tensors)",
                "trainable_params": int(sum(p.numel() for p in params))}
    else:
        info = RC.attach_lora(lm, rcp["rank"], ctx.seed)
        params = [p for p in model.parameters() if p.requires_grad]
    opt = RC.make_optimizer(params, rcp, ctx.adam_offload)
    curve: list = []
    t0 = time.time()
    tr = RC.train_answer_ce(model, pools, encoders, opt, rcp=rcp, per_pool=per_pool, seed=ctx.seed,
                            micro_batch=ctx.micro_batch, loss_log=curve,
                            max_steps=RC.SMOKE_STEPS if ctx.smoke else None)
    state = opt.state_gb() if hasattr(opt, "state_gb") else None
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    curves = json.loads(curve_path.read_text()) if curve_path.exists() else {}
    curves[str(dose)] = curve
    RC.atomic_write_json(curve_path, curves)
    return {**tr, "attach": {k: v for k, v in info.items() if k != "names"}, "opt_state": state,
            "ce_every_10": curve[9::10], "train_s": time.time() - t0, "dose_items": dose}


def save_dose_weights(ctx: Context, lm, tag, out: pathlib.Path) -> str:
    """LoRA: A/B per module (+ alpha/rank so ΔW is recoverable offline); full-FT: the trained tensors (bf16)."""
    import torch
    d = out / "adapters" / str(tag)
    d.mkdir(parents=True, exist_ok=True)
    if ctx.recipe["key"] == "full":
        obj = {"kind": "full-FT", "state": {n: p.detach().cpu() for n, p in ctx.ft_named}}
    else:
        obj = {"kind": "lora", "A_B": RC.lora_state(lm), "rank": ctx.recipe["rank"], "alpha": ctx.recipe["alpha"]}
    obj.update({"row": ctx.row, "arm": ctx.arm, "recipe": ctx.recipe, "tag": str(tag), "seed": ctx.seed,
                "im_task": str(tag)[3:] if str(tag).startswith("im_") else None})
    p = d / "weights.pt"
    torch.save(obj, p)
    return str(p)


def deltas_from_file(path: str) -> dict:
    """ΔW = (alpha / r_eff)·A·B per module from a saved LoRA file (r_eff = A.shape[1] = min(r, out, in),
    exactly relearn.make_adapter's scale). The SUB unrelated-adapter null (an IM adapter of another task)."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    assert obj["kind"] == "lora", f"{path}: unrelated adapter must be a LoRA file, got {obj['kind']}"
    sd, out = obj["A_B"], {}
    for k in sd:
        if k.endswith(".A"):
            name = k[:-2]
            A, B = sd[k].float(), sd[name + ".B"].float()
            out[name] = (float(obj["alpha"]) / A.shape[1]) * (A @ B)
    assert out, f"{path}: no adapter tensors"
    return {"deltas": out, "task": obj.get("im_task"), "row": obj.get("row"), "path": path}


def install_deltas(ctx: Context) -> dict:
    """ΔW_inst per adapter-target matrix (after − before, fp32 CPU) — computed BEFORE any adapter wraps the
    Linears (the names are the pristine module names) and before full-FT touches the weights."""
    from infra.pg import relearn as R
    lm_a = ctx.lm(ctx.after)
    sd_a, sd_b = lm_a.state_dict(), ctx.remap_before_sd(ctx.lm(ctx.before).state_dict())
    out = {}
    for name in R.adapter_target_names(lm_a):
        k = name + ".weight"
        assert k in sd_b, f"install delta: {k} missing from the healthy LM (remap?)"
        assert sd_a[k].shape == sd_b[k].shape, (k, sd_a[k].shape, sd_b[k].shape)
        out[name] = sd_a[k].detach().float().cpu() - sd_b[k].detach().float().cpu()
    return out


def sub_block(ctx: Context, deltas_rec: dict, subspaces: dict, k: int, k_requested: int, unrelated: dict) -> dict:
    """SUB statistics on the device (QR + projections), the Haar draws being the exact numpy sequence of
    matrix_core.subspace_overlap; the unrelated-adapter null over the SAME matrices (asserted)."""
    dev = ctx.device
    assert set(unrelated["deltas"]) == set(deltas_rec), "unrelated adapter covers different matrices"
    st = RC.sub_statistics({n: d.to(dev) for n, d in deltas_rec.items()}, {n: u.to(dev) for n, u in subspaces.items()},
                           k, ctx.n_null, 1337, {n: d.to(dev) for n, d in unrelated["deltas"].items()})
    st.update({"k_requested": k_requested, "unrelated_adapter": {kk: v for kk, v in unrelated.items() if kk != "deltas"}})
    return st


# ============================================================================================ GB
def graft_back(ctx: Context, recipient, donor, dose, o_rec: dict, o_dam: dict) -> dict:
    """Spec §1.2 GB: donor = the pre-recovery DAMAGED model, recipient = the recovered model; the donor's
    layer-L residual replaces the recipient's for every sweep layer (stride 2 + last); O (candidate
    likelihood) per layer; ≥ 5 mismatched-item cuts on O for T_O (D12). Banks O_restored inputs only."""
    import torch
    from infra.aws_pg.jobs.broca_sweep import PatchedLM
    donor.to(ctx.device)
    fr, fd = ctx.forward(recipient), ctx.forward(donor)
    assert fr.n_layers == fd.n_layers
    sweep, nulls = RC.sweep_layers(fr.n_layers, 2), RC.null_layers(fr.n_layers)
    res = {"dose": dose, "task": O_VERDICT_TASK, "O_recovered": o_rec["acc"], "O_damaged": o_dam["acc"],
           "n_layers": fr.n_layers, "sweep_layers": sweep, "null_layers": nulls, "batch": GB_BATCH,
           "layers": {}, "null": {}}
    lost = o_dam["acc"] - o_rec["acc"] > 1e-9          # guaranteed by the trigger; False only when GB is forced
    if not lost:
        res["note"] = "O not lost by recovery (GB forced below its trigger): O_restored undefined, banked as None"
    for kind, layers, mismatch in (("layers", sweep, False), ("null", nulls, True)):
        for L in layers:
            call = PatchedLM(donor, recipient, fd.layers, fr.layers, L, mismatch=mismatch)
            rec = ctx.o_readout(O_VERDICT_TASK, call, GB_BATCH)
            r = RC.o_restored(rec["acc"], o_rec["acc"], o_dam["acc"]) if lost else None
            # §12.11: GB dropped `rec["extra"]`, so the whole graft-back column — the readout that
            # decides whether a lost skill is unread or gone — reached `samples.json` as a bare number.
            # `o_readout` already ran real inference and already banked the samples; only the samples
            # block is carried up (the per-item vectors are right here beside it).
            sm = (rec.get("extra") or {}).get("samples")
            assert sm, f"GB L={L}: the O readout banked no extra.samples (§12.11)"
            res[kind][str(L)] = {"O_graft": rec["acc"], "O_restored": r,
                                 "per_item": rec["per_item"], "per_item_hash": rec["per_item_hash"],
                                 "extra": {"col": f"GB-{O_VERDICT_TASK}", "acc": rec["acc"], "n": rec.get("n"),
                                           "floor": rec.get("floor"), "samples": sm}}
            print(f"[REC][GB] {'null ' if mismatch else ''}L={L} O_graft={rec['acc']:.3f} O_restored={r}", flush=True)
    donor.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# ============================================================================================ runners
def _write(out: pathlib.Path, m: dict) -> None:
    RC.atomic_write_json(out / "metrics.json", m)
    _write_samples(str(out), m, "recovery")                 # spec §12.11: every eval carries real data


def run_recovery(ctx: Context, out: pathlib.Path, doses: list, a, m: dict, after_dose=None) -> dict:
    import torch
    from infra.pg.util_probe import log_event
    ev_path = out / "events.jsonl"
    dam = ctx.after
    fwd_dam = ctx.forward(dam)
    ctx.probe(fwd_dam)
    fwd_dam.batch_size = ctx.batch_size
    log_event(ev_path, stage="dose0", batch=ctx.batch_size)
    ev0 = evaluate(ctx, dam, fwd_dam)
    m["dose0"] = ev0
    m["identity"] = identity_gate(ctx, ev0)
    _write(out, m)
    print("[REC] dose-0 identity gate:", {k: (v["agree"] if isinstance(v, dict) else v) for k, v in m["identity"].items()}, flush=True)

    donor = copy.deepcopy(dam).to("cpu") if a.gb else None          # pristine damaged donor (pre-training)
    subspaces = unrelated = None
    if a.sub:
        inst = install_deltas(ctx)
        k_req = ctx.recipe["rank"] or 512
        k = min(k_req, min(d.shape[0] for d in inst.values()))
        subspaces = {n: RC.install_subspace(d.to(ctx.device), k).cpu() for n, d in inst.items()}
        unrelated = deltas_from_file(a.unrelated_adapter)
        m["sub_setup"] = {"k": k, "k_requested": k_req, "n_matrices": len(inst), "n_null": ctx.n_null,
                          "install_delta_fro": {n: float(d.norm()) for n, d in inst.items()},
                          "unrelated_adapter": {kk: v for kk, v in unrelated.items() if kk != "deltas"}}
        del inst

    model = ctx.before if ctx.arm == "healthy" else dam
    lm = ctx.lm(model)
    if ctx.arm == "scratch":
        m["reinit"] = RC.reinit_decoder(lm, ctx.remap_before_sd(ctx.lm(ctx.before).state_dict()), ctx.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    snap = snap_by_name = None
    if ctx.recipe["key"] == "full":
        ctx.ft_named = RC.full_ft_params(lm)
        for _, p in ctx.ft_named:
            p.requires_grad_(True)
        snap = RC.snapshot([p for _, p in ctx.ft_named])
        snap_by_name = {n: s for (n, _), s in zip(ctx.ft_named, snap)}
        ctx.ft_snapshot = snap
    fwd = ctx.forward_at(model)
    if ctx.arm in ("scratch", "healthy"):
        m["substrate_dose0"] = evaluate(ctx, model, fwd)
        if ctx.arm == "healthy":                       # the ceiling's substrate is the behaviour cell's `before`
            m["identity_substrate"] = identity_gate(ctx, m["substrate_dose0"], which="before")
    m["doses"] = {}
    _write(out, m)

    for dose in doses:
        log_event(ev_path, stage=f"train:{dose}")
        pools, encoders, per_pool = build_pools(ctx, dose)
        tr = train_dose(ctx, model, lm, dose, pools, encoders, per_pool, out / "train_curve.json")
        path = save_dose_weights(ctx, lm, dose, out)
        log_event(ev_path, stage=f"eval:{dose}")
        ev = evaluate(ctx, model, fwd)
        rec = {"train": tr, "weights": path, **ev, "fractions": fractions(ctx, ev0, ev)}
        if a.sub:
            deltas = RC.lora_deltas(lm) if ctx.recipe["key"] != "full" else RC.full_ft_deltas(lm, snap_by_name)
            rec["sub"] = sub_block(ctx, deltas, subspaces, m["sub_setup"]["k"], m["sub_setup"]["k_requested"], unrelated)
            del deltas
        m["doses"][str(dose)] = rec
        _write(out, m)
        s_rec = {h: rec["fractions"]["S_rec"][h]["acc"] for h in HALVES}
        print(f"[REC] dose {dose}: S={s_rec} K={ {h: rec['fractions']['K_rec'][h]['acc'] for h in HALVES} } "
              f"O_lost={ {t: rec['fractions']['O_lost'][t]['lost'] for t in ctx.o_tasks} } ce_final={tr['final_loss']:.3f}", flush=True)
        if a.gb and "gb" not in m and O_VERDICT_TASK in ctx.o_tasks:
            lost = rec["fractions"]["O_lost"][O_VERDICT_TASK]["lost"]
            # nan (VOID-low-margin) never triggers; a negative trigger forces GB at the first dose (dry-run)
            if a.gb_trigger < 0 or (lost == lost and lost >= a.gb_trigger):
                log_event(ev_path, stage=f"gb:{dose}")
                m["gb"] = graft_back(ctx, model, donor, dose, ev["O"][O_VERDICT_TASK], ev0["O"][O_VERDICT_TASK])
                _write(out, m)
        if ctx.recipe["key"] == "full":
            RC.restore([p for _, p in ctx.ft_named], snap)
        else:
            RC.detach_lora(lm)
        if after_dose is not None:
            after_dose(dose)
    if a.gb and "gb" not in m:
        m["gb"] = {"status": f"not triggered (O_lost < {a.gb_trigger} at every dose)"}
    m["complete"] = True
    _write(out, m)
    return m


def run_im(ctx: Context, out: pathlib.Path, tasks: list, m: dict) -> dict:
    """IM (spec §1.2, §6.8): per training task one r512 run at 4k (or the task's train size) on the DAMAGED
    model; every measured task before (once — detach restores the base modules exactly) and after each."""
    from infra.pg.util_probe import log_event
    ev_path = out / "events.jsonl"
    assert ctx.recipe["key"] == IM_RECIPE and ctx.arm == "rec", "IM is one r512 run per task on the damaged model"
    for t in tasks:
        assert t in IM_TRAIN_TASKS, f"{t} is not an IM training task {IM_TRAIN_TASKS}"
    examples = {t: ctx.im_examples(t) for t in tasks}            # data first: a missing source fails before GPU work
    model = ctx.after
    lm = ctx.lm(model)
    fwd = ctx.forward(model)
    ctx.probe(fwd)
    fwd.batch_size = ctx.batch_size
    for p in model.parameters():
        p.requires_grad_(False)
    log_event(ev_path, stage="im:before")
    before = evaluate_im(ctx, model, fwd)
    m["dose0"] = before
    m["identity"] = identity_gate(ctx, before)
    m["im"] = {"tasks": tasks, "recipe": IM_RECIPE, "dose_cap": RC.IM_DOSE, "per_task": {}}
    _write(out, m)
    for task in tasks:
        d = out / "im" / task
        d.mkdir(parents=True, exist_ok=True)
        ex = examples[task]
        dose = len(ex)
        log_event(ev_path, stage=f"im:{task}")
        enc = ctx.encode_vlm if task in RC.IM_VLM_TASKS else ctx.encode_text
        tr = train_dose(ctx, model, lm, dose, [ex], [enc], ctx.recipe["batch"], d / "train_curve.json")
        path = save_dose_weights(ctx, lm, f"im_{task}", d)
        after = evaluate_im(ctx, model, fwd)
        RC.detach_lora(lm)
        rec = {"task": task, "dose": dose, "train": tr, "weights": path, "before": before, "after": after,
               "example_hashes_sha256": _sha([e["hash"] for e in ex])}
        RC.atomic_write_json(d / "metrics.json", {"kind": "recovery-im", "row": ctx.row, **rec})
        m["im"]["per_task"][task] = {k: v for k, v in rec.items() if k != "before"}
        _write(out, m)
        print(f"[REC][IM] {task}: dose={dose} ce_final={tr['final_loss']:.3f} S_after="
              f"{ {h: after['S'][h]['acc'] for h in HALVES} }", flush=True)
    m["complete"] = True
    _write(out, m)
    return m


# ============================================================================================ contexts
def _base_context(a, device: str) -> Context:
    ctx = Context()
    recipes, ctx.recipes_sha = RC.load_recipes(a.recipes)
    ctx.recipe = RC.recipe(recipes, a.recipe)
    ctx.row, ctx.arm, ctx.device, ctx.smoke = a.row, a.arm, device, bool(a.smoke)
    ctx.n_boot = 10 if a.smoke else a.n_boot
    ctx.n_null = 20 if a.smoke else 200
    ctx.micro_batch, ctx.seed, ctx.adam_offload, ctx.batch_size = a.micro_batch, a.seed, bool(a.adam_offload), 4
    ctx.im_dose = RC.IM_DOSE
    ctx.unlearn = bool(a.unlearn)
    ctx.remap_before_sd = lambda sd: sd
    return ctx


def _smoke_items(items: list, smoke: bool) -> list:
    return items[::5] if smoke else items           # one item per subject (halves are 5 rows per subject)


def _triviaqa_train(load_dataset, n: int, eval_hashes, seed: int) -> list:
    from infra.pg import matrix_items as MI
    rows = load_dataset(MI.TRIVIAQA_REPO, MI.TRIVIAQA_CONFIG, split="train")
    perm = list(range(len(rows)))
    random.Random(seed).shuffle(perm)
    out = []
    for i in perm:
        r = rows[i]
        q, gold = str(r["question"]).strip(), str(r["answer"]["value"]).strip()
        h = MI.item_hash(MI.TRIVIAQA_SUBJECT, q)     # same key the eval set was built with
        if not q or not gold or h in eval_hashes:
            continue
        out.append({"subject": MI.TRIVIAQA_SUBJECT, "question": q, "answer": gold, "hash": h})
        if len(out) == n:
            break
    assert len(out) == n, f"TriviaQA train: only {len(out)} usable rows for n={n}"
    return out


def _arith_train(n: int) -> list:
    """Fresh draws of the arith template family (seed SEED + 2), deduped against the eval + exemplar hashes
    (relearn.dedup_against_eval semantics); the drop count is banked in the manifest."""
    from infra.pg import matrix_items as MI
    ev = {it["hash"] for it in MI.arith_items()} | {e["hash"] for e in MI.arith_exemplars()}
    raw = MI._arith_build(MI.SEED + 2, 2 * n)
    rows = [r for r in raw if r["hash"] not in ev][:n]
    assert len(rows) == n, f"arith IM train: only {len(rows)} eval-disjoint rows for n={n}"
    rows[0] = {**rows[0], "_n_dropped_eval_collisions": len(raw) - len([r for r in raw if r["hash"] not in ev])}
    return rows


def _bind_items(ctx: Context, a, load_dataset) -> None:
    """Item sets shared by the real and stub contexts (the loader is injected)."""
    from infra.pg import matrix_items as MI
    ctx.eval_items = {h: _smoke_items(MI.mmlu_eval(h, load_dataset), a.smoke) for h in HALVES}
    ctx.dev_shots = lambda subj, k: MI.mmlu_dev_shots(subj, k, load_dataset)
    ctx.held_out = MI.held_out_subjects()
    ex = {it["hash"] for h in HALVES for it in MI.mmlu_eval(h, load_dataset)}
    if ctx.unlearn:
        from infra.pg.relearn import ladder_subset
        pool = MI.wmdp_items("train", load_dataset)
        ctx.wmdp_probe_train = pool[:64] if a.smoke else pool
        ctx.wmdp_eval = MI.wmdp_items("eval", load_dataset)
        assert not ({r["hash"] for r in pool} & {it["hash"] for it in ctx.wmdp_eval}), "WMDP train/eval hashes overlap"
        if a.smoke:
            ctx.wmdp_eval = ctx.wmdp_eval[:SMOKE_O_ITEMS]
        ctx.train_rows = lambda n: ladder_subset(pool, n, salt=f"mdl-wmdp-{a.seed}")
    else:
        pool = MI.mmlu_train(None, a.seed, ex, load_dataset)
        ctx.train_rows = lambda n: _take(pool, n)
    ctx.arith = (MI.arith_items(), MI.arith_exemplars())
    tq = MI.triviaqa_items(load_dataset)
    ctx.tq = (tq["items"], tq["exemplars"])
    tq_hashes = {it["hash"] for it in tq["items"]} | {e["hash"] for e in tq["exemplars"]}

    def im_examples(task: str) -> list:
        n = ctx.im_dose
        if task in ("mmlu_letter", "mmlu_cloze"):
            return RC.im_text_examples(task, ctx.train_rows(n))
        if task == "arith":
            return RC.im_text_examples(task, _arith_train(n))
        if task == "triviaqa_cloze":
            return RC.im_text_examples(task, _triviaqa_train(load_dataset, n, tq_hashes, a.seed))
        if task == "vqav2":
            assert ctx.vqa_examples is not None, "vqav2 IM needs a VLM row"
            return ctx.vqa_examples(n)
        assert task != "refcoco", ("refcoco IM training set: no frozen source — lmms-lab/RefCOCO ships "
                                   "val/test/testA/testB only (spec §5 gap); pass --im without refcoco")
        raise ValueError(task)
    ctx.im_examples = im_examples
    ctx.item_manifest = {"eval": {h: {"n": len(v), "sha256": _sha([it["hash"] for it in v])} for h, v in ctx.eval_items.items()},
                         "held_out_subjects": ctx.held_out,
                         "train_pool": {"n": len(pool), "sha256": _sha([r["hash"] for r in pool]), "seed": a.seed,
                                        "source": "wmdp_items('train')" if ctx.unlearn else "mmlu_train"},
                         "wmdp_eval": ({"n": len(ctx.wmdp_eval), "sha256": _sha([it["hash"] for it in ctx.wmdp_eval])}
                                       if ctx.unlearn else None),
                         "arith": {"n": len(ctx.arith[0]), "sha256": _sha([it["hash"] for it in ctx.arith[0]])},
                         "triviaqa": {"n": len(ctx.tq[0]), "sha256": _sha([it["hash"] for it in ctx.tq[0]])}}


def _bind_pair(ctx: Context, pair: dict) -> tuple:
    """Bind ONE `matrix_models.load_pair(row, device, which=("before","after"))` return onto the Context.

    THE ONLY place this job reads the loader's return, so the real context and the dry-run bind through
    exactly the same code and a shape change breaks both. Every member goes through the loader's own
    `assert_member`, which is the contract (§1): a member is a DICT — {"fwd","model","tok","proc","repo",
    "which","shell","info","local"} — never a Forward and never an object with attributes.

    Readouts are read through the AFTER member's shell/tokenizer/processor, which is what this job always
    assumed with its single `pair.tokenizer`; a model that IS one of the two members gets that member's
    own shell, and a DERIVED model (the fair-scratch shell, a graft-back donor copy) gets the after
    member's. `_model_manifest` banks the two tokenizers' B1 comparison so a reader can see the
    assumption rather than inherit it."""
    from infra.pg import matrix_forward as MF
    from infra.pg import matrix_models as MM
    before = MM.assert_member(pair["before"], "before")
    after = MM.assert_member(pair["after"], "after")
    assert before["shell"] == after["shell"], f"pair shells differ: {before['shell']} vs {after['shell']}"
    ctx.before, ctx.after, ctx.tok, ctx.proc = before["model"], after["model"], after["tok"], after["proc"]
    ctx.shell = after["shell"]
    ctx.lm = lambda model: MF.lm_anatomy(model)["lm"]

    def forward(model):
        for mem in (after, before):
            if model is mem["model"]:
                return MM.forward_for(mem)

        return MM.forward_for(after, model)

    ctx.forward = forward
    if ctx.shell == "molmo2":
        from infra.pg.family_resolvers import qwen3_to_molmo2
        ctx.remap_before_sd = qwen3_to_molmo2
    return before, after


def _model_manifest(row: str, before: dict, after: dict, behaviour: dict, behaviour_source: str) -> dict:
    """This job's model provenance. B1 (tokenizer identity / shared prefix) and B3 (config, safetensors
    index and tokenizer-file hashes of the pinned snapshots) are COMPUTED HERE — both are cheap and both
    are about what THIS run loaded. B4 (the per-layer ‖ΔW‖/‖W‖ of the text decoder, embed/head, frozen_lm)
    is carried BY REFERENCE from the behaviour cell of the same row.

    Why B4 by reference and not `matrix_models.binds(...)`: reproducing it needs `snapshot_lm` to write
    the whole LM decoder to scratch and then a full-precision tensor-by-tensor compare — model-sized disk
    and minutes — to land, value for value, on the numbers that cell already banked from the SAME repos at
    the SAME pinned revisions. A row has one behaviour cell and many recovery cells, so recomputing turns
    a per-row cost into a per-cell one and buys nothing. What this job owes instead is that the pair it
    loaded IS that pair, and the D18 identity gate establishes that far more sharply than a weight norm:
    per item, ≥ 99 % agreement with the behaviour cell's `after` readout vectors, before any training."""
    from infra.pg import matrix_models as MM
    b = (behaviour or {}).get("binds")
    binds = ({"source": behaviour_source, "status": "the behaviour cell banks no `binds` record"} if not b else
             {"source": behaviour_source, "frozen_lm": b.get("frozen_lm"), "n_layers": b.get("n_layers"),
              "family": b.get("family"), "record_sha256": _sha(b)})
    return {"members": {m["which"]: {k: m[k] for k in ("which", "repo", "shell", "local", "info")}
                        for m in (before, after)},
            "repos": {r: MM.repo_hashes(r, MM.hf_files(r)) for r in MM.row_repos(row)},        # B3
            "known": {r: MM.KNOWN[r] for r in MM.row_repos(row)},
            "tokenizer": MM.shared_prefix(before["tok"], after["tok"]),                        # B1
            "binds": binds,                                                                    # B4, by reference
            "binds_note": "B4 is the behaviour cell's; B1/B3 computed here (see _model_manifest)"}


def _real_context(a, out: pathlib.Path) -> Context:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                   # FIRST, before transformers/datasets (contract)
    import torch
    from datasets import load_dataset
    from infra.pg import matrix_items as MI
    from infra.pg import matrix_models as MM            # agent MODELS (in progress): the pair loader
    from infra.pg.util_probe import choose_batch
    assert torch.cuda.is_available(), "the recovery job runs on the GPU box only"
    ctx = _base_context(a, "cuda")
    ctx.o_tasks = o_columns(a.row)                  # the row's O COLUMNS, from the registry (one place)
    if ctx.o_tasks:
        # Seam check BEFORE any weights reach the GPU: the names this job is about to call must exist
        # in the modules that own them. `hasattr` on the ONE dispatcher, not a docstring's word for it.
        from infra.aws_pg.jobs import matrix_behaviour as MB
        from infra.pg import vlm_skill_eval as V
        for mod, name in ((V, "evaluate"), (V, "molmo2_prompts"), (V, "Molmo2SkillForward"), (MB, "cand_column")):
            assert hasattr(mod, name), f"row {a.row} has O columns {ctx.o_tasks} but {mod.__name__} has no `{name}`"
        assert O_VERDICT_TASK == MB.O_VERDICT_KEY, (O_VERDICT_TASK, MB.O_VERDICT_KEY)
        assert O_VERDICT_TASK in ctx.o_tasks, (O_VERDICT_TASK, ctx.o_tasks)
    ctx.behaviour = _load_behaviour(a.behaviour_metrics, out)
    before, after = _bind_pair(ctx, MM.load_pair(a.row, "cuda", which=("before", "after")))
    ctx.model_manifest = _model_manifest(a.row, before, after, ctx.behaviour, a.behaviour_metrics)
    ctx.encode_text = lambda ex: RC.encode_text(ctx.tok, ex, "cuda")
    _bind_items(ctx, a, load_dataset)
    ctx.vqa_examples = None
    if ctx.proc is not None:
        max_n = max(int(x) for x in a.doses.split(",")) if a.doses else RC.IM_DOSE
        n_vqa = RC.SMOKE_DOSE if a.smoke else max(max_n, RC.IM_DOSE)
        if ctx.arm == "reh" or a.im:
            vt = MI.vqav2_train(n_vqa, a.seed, load_dataset)
            ds = load_dataset(vt["dataset"], data_files={"train": vt["data_files"]}, split="train",
                              verification_mode="no_checks").select(vt["indices"])
            ex_all = [{"prompt": f"answer en {ds[i]['question']}", "target_text": str(ds[i]["multiple_choice_answer"]),
                       "hash": h, "row": i} for i, h in enumerate(vt["hashes"])]
            ctx.vqa_examples = lambda n: _take(ex_all, n)
            ctx.encode_vlm = lambda ex: RC.encode_vlm(ctx.proc, [{**e, "image": ds[e["row"]]["image"]} for e in ex], "cuda")
            ctx.item_manifest["vqav2_train"] = {"n": len(ex_all), "sha256": _sha(vt["hashes"]), "seed": a.seed}
    if ctx.o_tasks:
        from infra.pg import vlm_skill_eval as V         # N14
        vis = MI.vision_index_lists(load_dataset)
        o_rows, o_hashes = {}, {}
        for col in ctx.o_tasks:
            task = o_task_of(col)
            idx = vis[task]["indices"][:SMOKE_O_ITEMS] if a.smoke else vis[task]["indices"]
            # cached 200-row subset when the box has one (verified row by row), else the full split
            o_rows[col] = MI.vision_rows(task, load_dataset)[:len(idx)]
            o_hashes[col] = list(vis[task]["hashes"][:len(idx)])
            assert [r["_index"] for r in o_rows[col]] == list(idx), \
                f"O:{col}: the loaded rows are not the index list this run banked"
            ctx.item_manifest[f"O:{col}"] = {"n": len(idx), "task": task, "sha256": _sha(idx),
                                             "hashes_sha256": _sha(o_hashes[col]),
                                             "hash_path": vis[task]["hash_path"]}

        def o_forward(call):
            """The O shell over `call` (a model or a broca_sweep.PatchedLM). Anatomy comes from the
            AFTER member's real module — a PatchedLM is a bare callable with no layers to inspect — and
            `.model` is then swapped to the thing being scored, the construction `forward_for`
            documents. Molmo2 needs `Molmo2SkillForward`, NOT the plain shell: the N14 prompts carry
            their own `<|image|>` placeholder, which `Molmo2Forward._encode` would prepend a second
            copy of, and the item's real image is bound on `.image` rather than passed per call."""
            if ctx.shell == "molmo2":
                f = V.Molmo2SkillForward(ctx.after, ctx.proc, "cuda")
            else:
                f = ctx.forward(ctx.after)
            f.model = call
            return f

        def o_readout(col, call, batch_size):
            f = o_forward(call)
            f.batch_size = batch_size
            return o_record(ctx.shell, f, col, o_rows[col], o_hashes[col])
        ctx.o_readout = o_readout

    def probe(fwd):
        from infra.pg import format_readouts as FR
        items = ctx.eval_items["discovery"][:32]
        prompts = [FR.s_prompt(it, ctx.dev_shots(it["subject"], 5)) for it in items]
        res = choose_batch(fwd, prompts, start=4, max_bs=64)
        ctx.batch_size = int(res["batch"])
        ctx.probe_result = res
    ctx.probe = probe
    return ctx


def _manifest(ctx: Context, a, argv) -> dict:
    import torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "transformers": getattr(sys.modules.get("transformers"), "__version__", None)}
    return {"row": ctx.row, "arm": ctx.arm, "recipe": ctx.recipe, "recipes_file": str(a.recipes),
            "recipes_sha256": ctx.recipes_sha, "doses": a.doses, "im": a.im, "seed": ctx.seed, "smoke": ctx.smoke,
            "n_boot": ctx.n_boot, "n_null": ctx.n_null, "micro_batch": ctx.micro_batch, "adam_offload": ctx.adam_offload,
            "gb": bool(a.gb), "sub": bool(a.sub), "unrelated_adapter": a.unrelated_adapter,
            "behaviour_metrics": a.behaviour_metrics, "git_sha": _git_sha(), "env": env,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "items": ctx.item_manifest, "models": ctx.model_manifest, "argv": list(argv or sys.argv[1:]),
            "identity_min_agree": RC.IDENTITY_MIN_AGREE, "gb_trigger_O_lost": a.gb_trigger}


# ============================================================================================ CLI
def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--row", default=None)
    ap.add_argument("--recipe", default="r512", choices=sorted(RC.RECIPE_KEYS))
    ap.add_argument("--arm", default="rec", choices=RC.ARMS)
    ap.add_argument("--doses", default=None, help="items per dose, comma list (default 250,1000,4000,16000; permuted: 4000)")
    ap.add_argument("--behaviour-metrics", default=None, help="behaviour cell metrics.json: local path or s3://")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gb", action="store_true", help="graft-back at the first dose with O_lost ≥ 0.3 (arm rec)")
    ap.add_argument("--sub", action="store_true", help="install-subspace overlap per dose (arm rec; needs --unrelated-adapter)")
    ap.add_argument("--unrelated-adapter", default=None, help="a saved IM LoRA file from ANOTHER task (the D9 null)")
    ap.add_argument("--im", default=None, help="IM training tasks: 'all' or a comma list of " + ",".join(IM_TRAIN_TASKS))
    ap.add_argument("--smoke", action="store_true", help="1 item/subject, 24 O items, one 32-item dose, 4 steps, n_boot 10")
    ap.add_argument("--adam-offload", action="store_true", help="full-FT: fp32 Adam state on the host")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipes", default=str(REPO / "configs" / "matrix_recipes.json"))
    ap.add_argument("--gb-trigger", type=float, default=RC.GB_TRIGGER_O_LOST,
                    help="O_lost that triggers GB (spec 0.3); negative = force at the first dose (dry-run only)")
    ap.add_argument("--dryrun", action="store_true")
    a = ap.parse_args(argv)
    if a.dryrun:
        return a
    assert a.row and a.out and a.behaviour_metrics, "--row, --out and --behaviour-metrics are required"
    assert pathlib.Path(a.recipes).exists(), f"recipes file missing: {a.recipes} (the launcher must ship configs/)"
    a.unlearn = str(a.row).startswith("7")
    if a.im:
        a.im = list(IM_TRAIN_TASKS) if a.im == "all" else a.im.split(",")
        assert a.recipe == IM_RECIPE and a.arm == "rec" and not (a.gb or a.sub), "--im is one r512 run per task on the damaged model"
    if a.doses is None:
        a.doses = ("4000" if a.arm == "permuted" else
                   ",".join(str(d) for d in (UNLEARN_DOSES if a.unlearn else (250, 1000, 4000, 16000))))
    if a.unlearn:
        assert a.arm in ("rec", "scratch", "healthy") and not (a.gb or a.sub or a.im), \
            "rows 7: REC (+ fair-scratch / healthy twins) only — REH/GB/SUB/IM and the permuted control are not defined there (spec §3.2, §6.13)"
    if a.sub:
        assert a.unrelated_adapter, "--sub needs --unrelated-adapter (the binding D9 null; Haar alone cannot fail)"
    if a.gb or a.sub:
        assert a.arm == "rec", "GB and SUB are defined on the recovered DAMAGED model (arm rec)"
    if a.adam_offload:
        assert a.recipe == "full", "--adam-offload applies to the full-FT recipe only"
    return a


MODULE = "infra.aws_pg.jobs.matrix_recovery"


def main(argv=None):
    # `python -m <job>` calls main() with argv=None, so the pin re-exec must resolve the real argv
    # before handing it to the child — `subprocess.run([py, "-m", mod, *None])` raises
    # "TypeError: Value after * must be an iterable, not NoneType". matrix_behaviour normalises on its
    # first line (`argv = list(sys.argv[1:] if argv is None else argv)`); these two did not, so my own
    # pin fix crashed the 3b recovery pack on its next launch.
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _parse(argv)
    if a.dryrun:
        _dryrun()
        return
    # §8b transformers pin. `matrix_behaviour` has always done this; these two jobs never did, so a
    # Molmo2 row's remote modeling code ran against the AMI's transformers 5.9.0 instead of its declared
    # 4.57.1. It fails as `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS[self.rope_type]` inside
    # `modeling_molmo2.py` — an error that names neither transformers nor the pin. The 2026-09-03
    # confound audit flagged this exact gap as item 30 ("matrix_graft.py / matrix_recovery.py never call
    # it"); it cost the 3b recovery packs two launches on 2026-09-04 before the prediction was matched to
    # the traceback. A row with no pin returns None here and falls straight through.
    from infra.pg.matrix_models import ensure_transformers_pin
    if ensure_transformers_pin(a.row, MODULE, argv) is not None:
        return
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = _real_context(a, out)
    from infra.pg.util_probe import DmonSampler
    sampler = DmonSampler(out / "events.jsonl").start()
    m = {"kind": "recovery", "row": a.row, "arm": ctx.arm, "recipe": ctx.recipe,
         "mode": {"gb": bool(a.gb), "sub": bool(a.sub), "im": a.im, "smoke": bool(a.smoke)}, "complete": False}
    m["manifest"] = _manifest(ctx, a, argv)
    _write(out, m)
    t0 = time.time()
    if a.im:
        run_im(ctx, out, a.im, m)
    else:
        run_recovery(ctx, out, RC.dose_schedule([int(x) for x in a.doses.split(",")], ctx.arm, a.smoke), a, m)
    m["wall_s"] = time.time() - t0
    _write(out, m)
    sampler.stop()
    print(f"RECOVERY_OK row={a.row} recipe={a.recipe} arm={ctx.arm} im={a.im} wall={m['wall_s']:.0f}s", flush=True)


# ============================================================================================ dry-run
def _stub_load_pair(row: str, d: int) -> dict:
    """The REAL `matrix_models.load_pair`, run weight-free: ONLY the model / tokenizer / processor
    construction is patched out (the same technique `matrix_models`'s own dry-run uses), so the dry-run
    exercises the loader's real branches and, above all, its real RETURN SHAPE.

    This is the rule the two 2026-09-03 crashes were paid for: a stub that builds the job's context by
    hand cannot see a loader whose shape it does not match. `_stub_context` used to construct the Context
    directly, so nine attributes this job asserted on `load_pair`'s return — .before .after .tokenizer
    .processor .shell .forward .lm .manifest .o_tasks, none of which the loader ever returned — sat green
    under `--dryrun`; matrix_graft's stub returned a Forward where the loader returns a member dict and
    three GPU packs died in `attach`. A stub must exercise the real loader."""
    from infra.pg import matrix_models as MM
    from infra.pg.matrix_forward import _StubTok, stub_text_model
    r = MM.ROWS[row] if row in MM.ROWS else None
    assert r is not None, f"the dry-run must name a REAL row of the registry, got {row!r}"
    assert r["shell"] == "text" and not r["adapter"] and not r["gated_on"] and not r["install"], \
        f"row {row}: the weight-free stub covers plain text rows (no shell, adapter, gate or install path)"
    tok, seeds = _StubTok(), {}
    saved = {k: getattr(MM, k) for k in ("_load_model", "_tokenizer", "_processor")}

    def load_model(repo, device, attn_impl, remote=False, device_map=None):
        seeds.setdefault(repo, len(seeds) + 1)              # before → seed 1, after → seed 2 (as before)
        return stub_text_model(n_layers=6, d=d, seed=seeds[repo]), "/local/" + repo

    def processor(local, remote=False):
        raise AssertionError("the weight-free stub covers text shells only")

    MM._load_model, MM._tokenizer, MM._processor = load_model, (lambda local, remote=False: tok), processor
    try:
        return MM.load_pair(row, "cpu", which=("before", "after"))
    finally:
        for k, v in saved.items():
            setattr(MM, k, v)


def _stub_context(a, tmp: pathlib.Path, d: int = 128) -> Context:
    """Weight-free, but through the REAL loader AND the REAL O readout: `_stub_load_pair` runs
    `matrix_models.load_pair` with the weights patched out, `_bind_pair` binds its members exactly as
    `_real_context` does, and `o_readout` goes through `o_record(...)` into
    `vlm_skill_eval.evaluate` over `vlm_skill_eval.molmo2_prompts()` — the real dispatcher, the real
    prompts, the real candidate draw, the real record shape. On top of that: matrix_items' synthetic
    loader and a second text pool standing in for VQAv2-train (the REH mixer / vqav2 IM path). Eval
    halves restricted to 4 subjects (20 items).

    What is patched, and only what genuinely needs weights: the FORWARD (a `TextForward` over the stub
    pair, `.image` ignored — a real Molmo run uses `Molmo2SkillForward` and a vision tower) and the
    ROWS (`vlm_skill_eval._stub_rows`, synthetic PIL images). So a green dry-run proves the seam
    exists, is called correctly and returns a bankable record; it does NOT prove any vision number.

    The predecessor of this readout was the defect: it was a text cloze that mimicked a record and
    never imported `vlm_skill_eval`, so `V.readout(...)` — a function that has never existed — sat
    green in every dry-run until a real row was queued."""
    from infra.pg import matrix_items as MI
    from infra.pg.matrix_forward import TextForward
    ctx = _base_context(a, "cpu")
    ctx.im_dose = 24
    _bind_pair(ctx, _stub_load_pair(ctx.row, d))
    tok = ctx.tok
    ctx.encode_text = lambda ex: RC.encode_text(tok, ex, "cpu")
    ctx.encode_vlm = ctx.encode_text                       # stub "VQA" pool = letter items through the text path
    ctx.probe = lambda fwd: None
    ld = MI.fake_load_dataset
    a_items = argparse.Namespace(smoke=False, seed=a.seed)
    _bind_items(ctx, a_items, ld)
    subjects = ("abstract_algebra", "anatomy", "astronomy", "world_religions")
    ctx.eval_items = {h: [it for it in v if it["subject"] in subjects] for h, v in ctx.eval_items.items()}
    ctx.held_out = ["astronomy", "world_religions"] + [s for s in MI.held_out_subjects() if s not in subjects]
    pool_tail = ctx.train_rows(60 if ctx.unlearn else 300)[::-1]   # disjoint from the small doses' prefix
    ctx.vqa_examples = lambda n: _letter_pool(pool_tail[:n])
    ctx.tq = (ctx.tq[0][:24], ctx.tq[1])
    ctx.arith = (ctx.arith[0][:24], ctx.arith[1])
    # The O columns are the REAL ones (`o_columns` off the registry would give [] for this text row, so
    # they are named here) and they run through the REAL readout — see the docstring for what is
    # patched. SMOKE_O_ITEMS rows because `matrix_core.preflight_match` asserts n >= 20.
    from infra.pg import vlm_skill_eval as V
    ctx.o_tasks = [O_VERDICT_TASK, "textvqa_cand"]
    o_rows = {c: V._stub_rows(o_task_of(c), SMOKE_O_ITEMS) for c in ctx.o_tasks}

    def o_readout(col, call, batch_size):
        fwd = TextForward(ctx.after, tok, "cpu")
        fwd.model = call                                    # a bare model or a PatchedLM
        fwd.batch_size = batch_size
        fwd.image = None                                    # the attribute `vqa_cand` binds; TextForward ignores it
        return o_record("molmo2", fwd, col, o_rows[col])
    ctx.o_readout = o_readout
    ctx.model_manifest = {"stub": True, "d": d}   # `_model_manifest` is exercised separately in _dryrun
    return ctx


def _stub_behaviour(ctx: Context) -> dict:
    """The behaviour cell this stub pair would have banked, in `matrix_behaviour`'s OWN layout:
    cols.{S,K}.{member}.{half} for the half-dependent columns and cols.O.{member}.**all** for O, which
    is half-independent. The "all" level was missing here until 2026-09-03, so the stub agreed with the
    job's (also wrong) reader and the O identity gate could not fire in either."""
    ev_b = evaluate(ctx, ctx.before, ctx.forward_at(ctx.before))
    ev_a = evaluate(ctx, ctx.after, ctx.forward_at(ctx.after))
    cols = {col: {"before": ev_b[col], "after": ev_a[col]} for col in ("S", "K")}
    cols["O"] = {"before": {"all": ev_b["O"]}, "after": {"all": ev_a["O"]}}
    for col in ("S_W", "K_W"):
        if col in ev_a:                                  # unlearning rows: banked as bare records
            cols[col] = {"before": ev_b[col]["eval"], "after": ev_a[col]["eval"]}
    return {"kind": "behaviour", "row": "stub", "cols": cols}


def _dryrun():
    import tempfile
    import torch
    from infra.pg import matrix_core as MC
    from infra.pg import relearn as R
    torch.manual_seed(0)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mdl_rec_"))
    # a REAL row id: the stub now goes through the real `load_pair` (row 5 = gemma-3-4b pt→it, text shell)
    args = lambda *extra: _parse(["--row", "5", "--out", str(tmp / "o"), "--behaviour-metrics", str(tmp / "beh.json"), *extra])

    # -- the LOADER SEAM, weight-free through the REAL `matrix_models.load_pair` (the defect this closes:
    #    the job asserted nine attributes on a return that is one member DICT per side, and the old
    #    hand-built stub could not see it). Shape, binding, o_tasks derivation and the model manifest.
    from infra.pg import matrix_models as MM
    from infra.pg.matrix_forward import Forward as _RealForward
    pair5 = _stub_load_pair("5", 128)
    assert set(pair5) == {"before", "after"}
    for w in ("before", "after"):
        assert set(pair5[w]) == set(MM.MEMBER_KEYS) and MM.assert_member(pair5[w], w) is pair5[w]
    assert pair5["before"]["repo"] == MM.ROWS["5"]["before"] and pair5["after"]["repo"] == MM.ROWS["5"]["after"]
    _c = Context()
    _b, _af = _bind_pair(_c, pair5)
    assert _c.before is pair5["before"]["model"] and _c.after is pair5["after"]["model"]
    assert _c.tok is pair5["after"]["tok"] and _c.proc is None and _c.shell == "text"
    assert _c.lm(_c.after) is _c.after.model                       # lm_anatomy's LM module = the adapter scope
    f_own, f_new = _c.forward(_c.before), _c.forward(copy.deepcopy(_c.after))
    assert isinstance(f_own, _RealForward) and f_own.model is _c.before and f_own.device == "cpu"
    assert f_new.model is not _c.after and f_new.shell == "text"   # a DERIVED model (scratch shell / GB donor)
    assert MM.o_tasks("5") == [] and MM.o_tasks("7a") == [] and MM.o_tasks("2") == ["vqav2", "textvqa"]
    assert MM.o_tasks("1b-pi0") == [] and MM.o_tasks("3c-Act") == [] and MM.o_tasks("3b") == ["vqav2", "textvqa"]
    # the O COLUMNS this job measures are the registry's tasks' CANDIDATE columns, and the verdict key
    # is the one `matrix_behaviour` actually banks — the two ends of the seam, named once each
    from infra.aws_pg.jobs import matrix_behaviour as _MB
    assert o_columns("2") == ["vqav2_cand", "textvqa_cand"] == o_columns("3b") and o_columns("5") == []
    assert O_VERDICT_TASK == _MB.O_VERDICT_KEY == "vqav2_cand"
    assert O_VERDICT_TASK in _MB.O_KEYS["paligemma"] and O_VERDICT_TASK in _MB.O_KEYS["molmo2"]
    assert "vqav2" not in _MB.O_KEYS["paligemma"], "the OLD O_VERDICT_TASK named no PaliGemma key at all"
    assert set(o_columns("2")) <= set(_MB.O_KEYS["paligemma"]) and set(o_columns("3b")) <= set(_MB.O_KEYS["molmo2"])
    assert o_task_of("vqav2_cand") == "vqav2" and MC._raises(o_task_of, "vqav2")
    _hf, _rh = MM.hf_files, MM.repo_hashes                          # manifest: B1/B3 here, B4 by reference
    MM.hf_files = lambda repo, revision=None: "/nonexistent"
    MM.repo_hashes = lambda repo, local: {"repo": repo, "sha": MM.KNOWN[repo]["sha"], "stub": True}
    try:
        man5 = _model_manifest("5", _b, _af, {"binds": {"frozen_lm": False, "n_layers": 6}}, "s3://beh/metrics.json")
        man_nb = _model_manifest("5", _b, _af, {}, "s3://beh/metrics.json")
    finally:
        MM.hf_files, MM.repo_hashes = _hf, _rh
    assert set(man5["members"]) == {"before", "after"} and set(man5["repos"]) == set(MM.row_repos("5"))
    assert man5["tokenizer"]["identical"] and man5["binds"]["frozen_lm"] is False
    assert len(man5["binds"]["record_sha256"]) == 64 and man_nb["binds"]["status"].startswith("the behaviour cell")

    # -- context + behaviour cell of the stub pair; identity PASS on the same model, FAIL on a perturbed cell
    a = args("--recipe", "r64", "--doses", "8,16")
    ctx = _stub_context(a, tmp)
    beh = _stub_behaviour(ctx)
    ctx.behaviour = beh
    ev0 = evaluate(ctx, ctx.after, ctx.forward_at(ctx.after))
    gate = identity_gate(ctx, ev0)
    assert all(v["agree"] == 1.0 for k, v in gate.items() if k[0] in "SK"), gate
    assert set(ctx.o_tasks) == {"vqav2_cand", "textvqa_cand"}
    for c in ctx.o_tasks:                              # every O column gated, none stamped ABSENT
        assert gate[f"O:{c}"]["agree"] == 1.0 and gate[f"O:{c}"]["n"] == SMOKE_O_ITEMS, gate[f"O:{c}"]
    assert set(_beh_o(beh, "after")) == set(ctx.o_tasks)     # the behaviour layout the real job banks
    assert MC._raises(_beh_o, {"cols": {"O": {"after": {"vqav2_cand": {}}}}}, "after")   # missing the 'all' level
    bad = copy.deepcopy(beh)
    v = bad["cols"]["S"]["after"]["discovery"]["per_item"]
    for i in range(0, len(v), 4):
        v[i] = 1 - v[i]
    ctx.behaviour = bad
    raised = False
    try:
        identity_gate(ctx, ev0)
    except RC.IdentityMismatch as e:
        raised = True
        print("  identity FAIL raised as designed:", str(e)[:80])
    assert raised, "perturbed behaviour cell must fail the identity gate"
    # hash alignment: a reordered reference aligns back; the smoke stride path aligns 1-per-subject
    ref = beh["cols"]["K"]["after"]["confirmation"]
    rev = {"per_item": ref["per_item"][::-1], "per_item_hash": ref["per_item_hash"][::-1]}
    assert _align_ref(rev, ref["per_item_hash"])["per_item"] == ref["per_item"]
    dup = {"per_item": ref["per_item"], "per_item_hash": [ref["per_item_hash"][0]] * len(ref["per_item_hash"])}
    st = _align_ref(dup, dup["per_item_hash"][::5])
    assert st["per_item"] == ref["per_item"][::5]
    ctx.behaviour = beh

    # -- the general trainer == relearn.train_adapter_qa step for step (loss curve) on one letter pool
    rows = ctx.train_rows(24)
    ex = _letter_pool(rows)
    m1, m2 = copy.deepcopy(ctx.after), copy.deepcopy(ctx.after)
    for p in list(m1.parameters()) + list(m2.parameters()):
        p.requires_grad_(False)
    RC.attach_lora(m1.model, 16, 0)
    log_ref: list = []
    R.train_adapter_qa(m1, ctx.tok, "cpu", [R.format_qa(r["question"], r["choices"], r["answer"]) for r in rows],
                       lr=2e-4, epochs=2, batch_size=8, warmup_ratio=0.05, seed=0, micro_batch=4,
                       weight_decay=0.0, loss_log=log_ref)
    RC.attach_lora(m2.model, 16, 0)
    rcp = {**ctx.recipe, "batch": 8, "rank": 16, "alpha": 16.0}
    opt = RC.make_optimizer([p for p in m2.parameters() if p.requires_grad], rcp, False)
    log_new: list = []
    RC.train_answer_ce(m2, [ex], [ctx.encode_text], opt, rcp=rcp, per_pool=8, seed=0, micro_batch=4, loss_log=log_new)
    assert len(log_ref) == len(log_new) == 6 and max(abs(x - y) for x, y in zip(log_ref, log_new)) < 1e-5, (log_ref, log_new)
    print(f"  trainer equivalence: {len(log_ref)} steps, max|Δce| = {max(abs(x - y) for x, y in zip(log_ref, log_new)):.2e}")

    # -- an "IM adapter of another task" for the SUB null (arith, text path) → saved file → deltas
    im_dir = tmp / "im"
    im_dir.mkdir()
    ctx_im = _stub_context(args("--im", "arith"), tmp)
    ctx_im.behaviour = beh
    m_im = {"kind": "recovery", "complete": False}
    run_im(ctx_im, im_dir, ["arith"], m_im)
    assert m_im["complete"] and (im_dir / "im" / "arith" / "metrics.json").exists()
    im_rec = m_im["im"]["per_task"]["arith"]
    assert set(im_rec["after"]) >= {"S", "K", "O", "A", "TQ"} and im_rec["dose"] == 24 and im_rec["train"]["steps"] == 4
    unrelated = im_dir / "im" / "arith" / "adapters" / "im_arith" / "weights.pt"
    assert unrelated.exists()
    # detach restored the base modules exactly: the damaged model still reproduces its dose-0 readout
    again = evaluate(ctx_im, ctx_im.after, ctx_im.forward_at(ctx_im.after), with_o=False)
    assert again["S"]["discovery"]["per_item"] == m_im["dose0"]["S"]["discovery"]["per_item"]
    dl = deltas_from_file(str(unrelated))
    assert len(dl["deltas"]) == 6 and all(v.shape == (128, 128) for v in dl["deltas"].values()) and dl["task"] == "arith"
    raised = False
    try:
        ctx_im.im_examples("refcoco")
    except AssertionError as e:
        raised = "no frozen source" in str(e)
    assert raised, "refcoco IM train set must fail loud (no frozen source)"
    assert len(ctx_im.im_examples("vqav2")) == 24 and len(ctx_im.im_examples("triviaqa_cloze")) == 24

    # -- rec r64, two doses (incremental), GB forced, SUB with the unrelated null
    a = args("--recipe", "r64", "--doses", "8,16", "--gb", "--sub", "--unrelated-adapter", str(unrelated), "--gb-trigger", "-1")
    ctx = _stub_context(a, tmp)
    ctx.behaviour = beh
    out = tmp / "rec"
    out.mkdir()
    seen = {}

    def after_dose(dose):
        m_disk = json.loads((out / "metrics.json").read_text())
        seen[dose] = (sorted(m_disk["doses"]), m_disk["complete"])
    m = {"kind": "recovery", "complete": False}
    run_recovery(ctx, out, RC.dose_schedule([8, 16], "rec", False), a, m, after_dose=after_dose)
    assert seen[8] == (["8"], False) and seen[16] == (["16", "8"], False) and m["complete"]
    for dose in ("8", "16"):
        r = m["doses"][dose]
        assert r["train"]["steps"] == 2 * -(-int(dose) // 16) and pathlib.Path(r["weights"]).exists()
        assert set(r["fractions"]["S_rec"]) == set(HALVES) and "ratio_status" in r["fractions"]["S_rec"]["discovery"]
        assert "lost" in r["fractions"]["O_lost"]["vqav2_cand"] and r["held_out"]["discovery"]["S"]["held_out"]["n"] == 10
        sub = r["sub"]
        assert sub["k"] == 64 and len(sub["per_layer"]) == 6 and set(sub["per_matrix"]) == set(dl["deltas"])
        for L, s in sub["per_layer"].items():
            assert 0 <= s["overlap"] <= 1 and 0 <= s["overlap_unrelated"] <= 1 and 0 <= s["haar_p95"] <= 1, (L, s)
    curves = json.loads((out / "train_curve.json").read_text())
    assert sorted(curves) == ["16", "8"] and len(curves["16"]) == 2
    gb = m["gb"]
    assert gb["dose"] == 8 and gb["sweep_layers"] == [0, 2, 4, 5] and len(gb["null"]) == 5
    assert all(set(v) >= {"O_graft", "O_restored", "per_item"} for v in list(gb["layers"].values()) + list(gb["null"].values()))
    assert gb["task"] == O_VERDICT_TASK
    assert gb["O_recovered"] == m["doses"]["8"]["O"][O_VERDICT_TASK]["acc"] and gb["O_damaged"] == m["dose0"]["O"][O_VERDICT_TASK]["acc"]
    assert m["identity"][f"O:{O_VERDICT_TASK}"]["agree"] == 1.0
    # the record came from the REAL readout: `vqa_cand`'s own column name, floor and 4-way scores
    o0 = m["dose0"]["O"][O_VERDICT_TASK]
    assert o0["col"] == O_VERDICT_TASK and o0["floor"] == 0.25 and o0["n"] == SMOKE_O_ITEMS
    assert all(len(x) == 4 for x in o0["per_item_scores"]) and len(o0["extra"]["options"][0]) == 4
    assert o0["extra"]["prompt_example"].startswith("<|image|><|im_start|>user\nvqa2: ")
    # §12.11, one branch or the other and never silence: the producer banks real samples, or this job
    # stamps the record NOT QUOTABLE. Today it is the second; the assert flips to the first for free.
    assert bool(o0["extra"].get("samples")) != bool(o0["extra"].get("samples_missing_12_11")), sorted(o0["extra"])
    print("  O §12.11:", "real inference samples banked by the producer" if o0["extra"].get("samples") else
          "NO producer samples — every O record stamped `samples_missing_12_11`, so no O number is quotable")
    # the damaged model is back to its base modules after the last detach
    assert not any(RC._is_adapter(mod) for _, mod in ctx.after.named_modules())
    man = _manifest(ctx, a, ["--dryrun"])
    assert len(man["recipes_sha256"]) == 64 and man["items"]["train_pool"]["n"] > 24 and man["sub"]
    print("  rec r64 ×2 doses + GB + SUB: OK")

    # -- every other arm × recipe path on the smoke ladder (one 32-item dose, 4 steps)
    for arm, recipe, extra in (("scratch", "full", ["--adam-offload"]), ("healthy", "r64", []),
                               ("reh", "r512", []), ("permuted", "r512", []), ("rec", "full", [])):
        a = args("--arm", arm, "--recipe", recipe, "--smoke", *extra)
        ctx = _stub_context(a, tmp)
        ctx.behaviour = beh
        out = tmp / f"{arm}_{recipe}"
        out.mkdir()
        m = {"kind": "recovery", "complete": False}
        run_recovery(ctx, out, RC.dose_schedule([int(x) for x in a.doses.split(",")], arm, True), a, m)
        r = m["doses"]["32"]
        assert m["complete"] and r["train"]["steps"] == RC.SMOKE_STEPS and pathlib.Path(r["weights"]).exists()
        if arm in ("scratch", "healthy"):
            assert "substrate_dose0" in m
        if arm == "healthy":
            assert all(v["agree"] == 1.0 for k, v in m["identity_substrate"].items() if isinstance(v, dict))
        if arm == "scratch":
            assert m["reinit"]["n_random"] == 6 and r["train"]["opt_state"]["where"] == "host"
        if arm == "reh":
            assert r["train"]["recipe"]["batch_size"] == 16 and r["train"]["n_examples"] == 64
        if recipe == "full":
            w = torch.load(r["weights"], map_location="cpu", weights_only=False)
            assert w["kind"] == "full-FT" and len(w["state"]) == len(ctx.ft_named)
            # training moved the weights; the per-dose restore put the substrate back on its snapshot
            assert any(not torch.equal(w["state"][n], s) for (n, _), s in zip(ctx.ft_named, ctx.ft_snapshot))
            assert all(torch.equal(p.detach(), s) for (_, p), s in zip(ctx.ft_named, ctx.ft_snapshot))
        print(f"  arm {arm} × {recipe}: OK (ce {r['train']['final_loss']:.3f})")

    # -- unlearning row (7a): WMDP ladder, S_W / K_W / P_W + retained MMLU S/K, identity incl. S_W/K_W
    a = args("--row", "7a", "--doses", "16,32")
    assert a.unlearn
    ctx = _stub_context(a, tmp)
    beh7 = _stub_behaviour(ctx)
    ctx.behaviour = beh7
    out = tmp / "unlearn"
    out.mkdir()
    m = {"kind": "recovery", "complete": False}
    run_recovery(ctx, out, RC.dose_schedule([16, 32], "rec", False), a, m)
    assert m["complete"] and m["identity"]["S_W:eval"]["agree"] == 1.0 and m["identity"]["K_W:eval"]["agree"] == 1.0
    r = m["doses"]["32"]
    assert r["S_W"]["eval"]["n"] == 40 and r["K_W"]["eval"]["col"] == "K_W" and r["P_W"]["layers"] == [2, 4]
    assert all({"acc", "per_item", "shuffle_acc"} <= set(v) for v in r["P_W"]["per_layer"].values())
    assert r["fractions"]["S_W_rec"]["ratio_status"] and set(r["fractions"]["P_W_acc"]) == {"2", "4"}
    assert r["train"]["dose_items"] == 32 and r["train"]["steps"] == 4 and "S" in r["held_out"]["discovery"]
    assert m["manifest"]["items"]["wmdp_eval"]["n"] == 40 if "manifest" in m else True
    ctx.behaviour = {"kind": "behaviour", "cols": {k: v for k, v in beh7["cols"].items() if k in ("S", "K")}}
    ev7 = evaluate(ctx, ctx.after, ctx.forward_at(ctx.after))
    g7 = identity_gate(ctx, ev7)
    assert g7["S_W:eval"].startswith("ABSENT") and g7["S:discovery"]["agree"] == 1.0
    assert isinstance(fractions(ctx, ev7, ev7)["K_W_rec"], str)
    print("  row 7a (WMDP ladder, S_W/K_W/P_W, identity present+absent): OK")

    # -- CLI validation
    a7 = _parse(["--row", "7a", "--out", "x", "--behaviour-metrics", "b"])
    assert a7.unlearn and a7.doses == "250,500,1000"
    for bad_argv in (["--row", "2", "--out", "x", "--behaviour-metrics", "b", "--sub"],
                     ["--row", "2", "--out", "x", "--behaviour-metrics", "b", "--gb", "--arm", "reh"],
                     ["--row", "2", "--out", "x", "--behaviour-metrics", "b", "--im", "all", "--recipe", "r64"],
                     ["--row", "7a", "--out", "x", "--behaviour-metrics", "b", "--gb"],
                     ["--row", "7a", "--out", "x", "--behaviour-metrics", "b", "--arm", "permuted"]):
        try:
            _parse(bad_argv)
            raise RuntimeError("bad argv accepted: " + " ".join(bad_argv))
        except AssertionError:
            pass
    a = _parse(["--row", "2", "--out", "x", "--behaviour-metrics", "b", "--im", "all"])
    assert a.im == list(IM_TRAIN_TASKS) and a.doses == "250,1000,4000,16000"
    assert _parse(["--row", "2", "--out", "x", "--behaviour-metrics", "b", "--arm", "permuted"]).doses == "4000"
    print("PASS matrix_recovery dry-run (identity pass+fail; trainer == relearn; rec+GB+SUB; scratch/healthy/reh/permuted; "
          "full-FT ±offload; IM; row 7a WMDP ladder)")


def _write_samples(out_dir, metrics, kind):
    """Spec §12.11: one samples.json per cell, covering every eval this job ran (the readout records it
    banked already carry `extra.samples`)."""
    import json as _json, os as _os
    from infra.pg.format_readouts import collect_samples
    sm = collect_samples(metrics)
    tmp = _os.path.join(out_dir, "samples.json.tmp")
    with open(tmp, "w") as f:
        _json.dump({"row": metrics.get("row"), "kind": kind, "smoke": metrics.get("smoke"),
                    "cols": sm}, f, indent=1, default=float)
    _os.replace(tmp, _os.path.join(out_dir, "samples.json"))
    return len(sm)


if __name__ == "__main__":
    main()


