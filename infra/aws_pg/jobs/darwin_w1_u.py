"""DARWIN W1-U — geometric replay, the retention arm. GPU. MISSION-CENTRAL (§4/W1-U).

Fine-tune on a recipe known to damage the base, under four update rules, and ask which one keeps
the base's language ability:

    naive                  task loss only
    anchor(lam)            + core-ACTIVATION drift penalty              <- the treatment
    ewc_lite(lam)          + core-WEIGHT-coordinate penalty             <- the named control
    replay(frac)           + interleaved base-distribution batches      <- the incumbent

REFUTER: the anchor's retention is no better than EWC-lite at matched task score — then the
effect is generic regularisation, not anything about the core.

W1-U is a §0.2.7 optimizer-mechanics arm, so the optimizer is EXACT fp32 Adam
(`darwin_train.ExactAdam`, unit-tested against torch AdamW to 1e-6) and `check_optimizer` is
called before a single step. The trainable scope (q_proj + o_proj, ~151M params) is identical
across every arm, so the arms differ ONLY in the update rule.

Readouts per checkpoint, for every arm:
  * task loss on a held-out slice of the fine-tune task
  * RETENTION: wikitext blank-image perplexity — the same harness G2 anchored, so the retention
    number is commensurable with every other ppl in this program
  * the §4/W1-U forgetting gauge: per-cell core-projected weight-drift energy

  pg run infra.aws_pg.jobs.darwin_w1_u --args "--arm W1-U" \
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

# The arm's POSITIVE CONTROL, frozen here. W1-U measures how much of the base's language ability
# each update rule KEEPS while learning a damaging task. If the naive arm does not actually damage
# the base, there is nothing to keep: every rule ties at zero and the arm reports a comparison it
# never made. The 8-step smoke showed exactly that shape (naive retention_loss = -0.009), which is
# the failure mode feedback_signals_that_cannot_fail names.
#
# Rule: naive's final retention loss must exceed DAMAGE_MIN_REL x base ppl, or the arm records
# VOID-no-damage and makes no anchor-vs-control claim at all.
DAMAGE_MIN_REL = 0.05

SPEC_EXPECTATIONS = {
    "W1-U": ("anchor's retention <= EWC-lite control at matched task score",
             "a core-activation anchor retains more LM ability than EWC-lite and replay at "
             "matched task score"),
    "W1-K": ("recovery <= twin recovery after FT, or battery harm above CI at all amplitudes",
             "an in-core rank-1 stamp is recoverable after a full fine-tune at an amplitude that "
             "does not harm the battery"),
    "W1-T": ("dominated by LoRA at every point of the data ladder",
             "highway-projected training matches or beats LoRA-r32 at matched optimizer memory "
             "in the low-data regime"),
}
# W1-T is NOT in this module. Its named incumbents are LoRA-r32 and GaLore, and neither is
# implemented here; running a "highway_restricted vs naive" pair and calling it W1-T would be a
# Register-B contrast with no incumbent, which §3's matched-incumbent rule forbids. W1-T stays
# unexecuted and visible as such in the coverage table until its incumbents exist.


def build_batches(processor, rows, device, max_len=128):
    """Tokenise the task's (prompt, target) pairs into supervised batches of size 1.

    Batch 1 throughout: padding would contaminate both the loss and the anchor's per-token
    activations, and there is no throughput argument at this scale that outweighs that.
    """
    import torch
    out = []
    for r in rows:
        text = r["prompt"] + r["target"]
        enc = processor.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 4:
            continue
        n_prompt = processor.tokenizer(r["prompt"], return_tensors="pt",
                                       truncation=True, max_length=max_len).input_ids.shape[1]
        labels = ids.clone()
        labels[:, :min(n_prompt, ids.shape[1] - 1)] = -100      # supervise the target only
        out.append({"input_ids": ids, "labels": labels})
    assert out, "no usable training batches"
    return out


def text_task_rows(n_train, n_eval, seed=1337):
    """The damage recipe, as a TEXT task on a distribution far from wikitext.

    §4/W1-U names the refcoco-seg recipe. That recipe needs the segmentation-token vocabulary and
    an image pipeline, and its damage mechanism — driving the decoder hard onto a narrow output
    grammar — is reproduced here by a code-completion task, which is equally far from wikitext and
    equally narrow in grammar. The SUBSTITUTION IS RECORDED on every row (`recipe_substitution`),
    because it changes what the arm generalises to: this measures retention under a
    narrow-grammar text recipe, not under refcoco-seg specifically.
    """
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    rows = []
    for r in ds:
        c = str(r["content"])
        if len(c) < 400:
            continue
        rows.append({"prompt": c[:200], "target": c[200:400]})
        if len(rows) >= n_train + n_eval:
            break
    assert len(rows) >= n_train + n_eval, f"only {len(rows)} rows available"
    return rows[:n_train], rows[n_train:n_train + n_eval]


def wikitext_rows(n, seed=1337):
    """Base-distribution rows for the replay incumbent: plain wikitext, self-supervised."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    rows = []
    for t in ds["text"]:
        t = str(t).strip()
        if len(t) < 400:
            continue
        rows.append({"prompt": t[:1], "target": t[1:400]})   # supervise essentially the whole span
        if len(rows) >= n:
            break
    assert len(rows) >= max(1, n // 4), f"wikitext replay pool too small: {len(rows)}"
    return rows


def eval_task_loss(model, batches):
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            out = model(input_ids=b["input_ids"], labels=b["labels"])
            tot += float(out.loss.item())
            n += 1
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="W1-U", choices=("W1-U", "W1-T", "W1-K"))
    ap.add_argument("--amplitudes", default="0.001,0.01,0.1,1.0")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--replay-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--allow-no-damage", action="store_true",
                    help="run the treatment arms even if the damage positive control fails "
                         "(smoke only; the rows are then not a retention comparison)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        layers, a.steps, a.eval_every = layers[:4], 8, 4
        a.n_train, a.n_eval, a.n_seq = 24, 8, 4

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers,
                                                       _load_full_pg1, _ppl_blankimg)

    lams = T.lam_ladder()
    D.check_dose_grid(lams, arm=a.arm, min_doses=2 if a.smoke else 4)
    # §0.2.7 — refuse an approximate optimizer BEFORE any step, not after the run.
    optinfo = D.check_optimizer(a.arm, "darwin_train.ExactAdam", offload_exact_fp32=False)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="GRAD", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "GRAD",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "layers": layers,
                "trainable_types": list(T.TRAINABLE_TYPES), "steps": a.steps, "lr": a.lr,
                "lam_ladder": lams, "n_train": a.n_train, "n_eval": a.n_eval,
                "n_seq_ppl": a.n_seq, "replay_frac": a.replay_frac, "optimizer": optinfo,
                "expected_ref": ref, "started_utc": utc, "seed": a.seed,
                "recipe_substitution": ("§4/W1-U names refcoco-seg; this run uses a narrow-grammar "
                                        "code-completion recipe with the same damage mechanism. "
                                        "Recorded on every row: the arm measures retention under "
                                        "THIS recipe, not under refcoco-seg specifically."),
                "trainable_scope": ("q_proj+o_proj over the listed layers, identical across arms; "
                                    "the arms differ only in the update rule")}
    D.write_json(out, "manifest.json", manifest)

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, a.seed)
    # The replay incumbent needs BASE-DISTRIBUTION batches to interleave. Built from the same
    # wikitext the retention readout scores, which is what "available-data replay" means here:
    # the practitioner keeps some of the pretraining-like data and mixes it back in.
    replay_rows = wikitext_rows(a.n_train // 2, a.seed)

    def fresh_model():
        m = _load_full_pg1(base_sha, "cuda")
        m.train()
        for p in m.parameters():
            p.requires_grad_(False)
        mods = T.trainable_modules(_decoder_layers(m), layers=layers)
        for mod in mods.values():
            mod.weight.requires_grad_(True)
        return m, mods

    banks = {}
    # Four update rules, one of them dosed on the frozen lambda ladder. Every entry here is
    # IMPLEMENTED in the loop below; an arm that appears in this list without a corresponding
    # branch would silently run as `naive` under another name, which is a fabricated control.
    # naive runs FIRST: it carries the damage gate, and a treatment measured against a recipe
    # that does no damage is not worth the GPU time.
    if a.arm == "W1-K":
        # W1-K rebuilt to actually TRAIN (defect D21). The earlier version added a fine-tune's
        # delta arithmetically and measured recovery on (W - W0 - dW), which is identically the
        # stamp: recovery came out 1.0 for every bank including all three Haar twins. Only
        # gradient steps can move a stamp, so recovery is only a question after real training.
        amps = [float(x) for x in a.amplitudes.split(",") if x.strip()]
        D.check_dose_grid(amps, arm="W1-K", min_doses=2 if a.smoke else 4)
        arms = [(f"stamp_{bid}", amp) for bid in ("core32", "anti32", "haar32_s0",
                                                  "complement32") for amp in amps]
    elif a.arm == "W1-U":
        arms = ([("naive", None)]
                + [("anchor", lam) for lam in lams]
                + [("ewc_lite", lam) for lam in lams]
                + [("replay", a.replay_frac)])
    else:
        # W1-T: highway-restricted training against its now-IMPLEMENTED incumbent (LoRA-r32) and
        # the unrestricted full-Adam reference, at matched step budget. Optimizer-state bytes are
        # recorded per arm so "matched optimizer memory" is an audited number, not a claim.
        arms = [("naive", None), ("highway_restricted", None), ("lora_r32", float(a.lora_rank))]

    results = []
    for arm_id, dose in arms:
        model, mods = fresh_model()
        if not banks:
            banks = {k: W.load_bank(bank_dir, k[0], k[1], "core32") for k in mods}
        train_b = build_batches(processor, train_rows, "cuda")
        eval_b = build_batches(processor, eval_rows, "cuda")
        replay_b = (build_batches(processor, replay_rows, "cuda")
                    if arm_id == "replay" else None)
        lora = (T.LoRAAdapter(mods, rank=a.lora_rank, alpha=a.lora_alpha, device="cuda",
                              seed=H.cell_seed(a.seed, "lora")) if arm_id == "lora_r32" else None)
        if lora is not None:
            for mod in mods.values():
                mod.weight.requires_grad_(False)          # only the adapter trains
            opt = T.ExactAdam(lora.parameters(), lr=a.lr)
        else:
            opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        projector = (T.SubspaceProjector({k: W.load_bank(bank_dir, k[0], k[1], "anti32")
                                          for k in mods}, "cuda")
                     if arm_id == "highway_restricted" else None)
        ewc = T.EwcLiteCore(mods, banks, "cuda", dose) if arm_id == "ewc_lite" else None
        anchor = T.CoreActivationAnchor(mods, banks, "cuda", dose) if arm_id == "anchor" else None

        # W1-K: plant the rank-1 stamp BEFORE training, and keep the planted directions so the
        # post-training residual can be projected back onto them.
        stamp = None
        if arm_id.startswith("stamp_"):
            bid = arm_id[len("stamp_"):]
            stamp = {}
            with torch.no_grad():
                for key, mod in mods.items():
                    U = W.load_bank(bank_dir, key[0], key[1], bid)
                    rg = np.random.default_rng(H.cell_seed(a.seed, bid, key[0], key[1]))
                    z = rg.standard_normal(U.shape[1]); z /= np.linalg.norm(z)
                    u = U @ z
                    v = rg.standard_normal(mod.weight.shape[0]); v /= np.linalg.norm(v)
                    stamp[key] = (u, v, mod.weight.detach().to(torch.float64).cpu().numpy().copy())
                    mod.weight.data.add_(torch.as_tensor(dose * np.outer(v, u),
                                                         dtype=mod.weight.dtype,
                                                         device=mod.weight.device))
        base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        rng = np.random.default_rng(H.cell_seed(a.seed, arm_id, str(dose)))
        ctx = anchor if anchor is not None else (lora if lora is not None else _NullCtx())
        with ctx:
            for step in range(a.steps):
                # The replay incumbent draws a base-distribution batch with probability
                # `replay_frac`; every other arm always draws the task. Same step count, same
                # optimizer budget — replay spends part of it on the base instead of the task,
                # which is exactly the trade the incumbent makes.
                use_replay = replay_b is not None and float(rng.random()) < float(dose)
                pool = replay_b if use_replay else train_b
                b = pool[int(rng.integers(0, len(pool)))]
                opt.zero_grad()
                if anchor is not None:
                    with torch.no_grad():
                        anchor.record_reference()
                        model(input_ids=b["input_ids"])
                    anchor.record_current()
                loss = model(input_ids=b["input_ids"], labels=b["labels"]).loss
                if anchor is not None:
                    loss = loss + anchor.penalty()
                if ewc is not None:
                    loss = loss + ewc.penalty()
                loss.backward()
                if projector is not None:
                    projector.apply(mods)
                opt.step()

                if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
                    ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
                    tl = eval_task_loss(model, eval_b)
                    gauge = T.core_energy_gauge(mods, banks)
                    fl = {"arm_id": arm_id, "dose": dose, "step": step + 1,
                          "base_ppl": base_ppl, "retention_loss": ppl - base_ppl,
                          "optimizer": "ExactAdam fp32",
                          "optimizer_state_bytes": opt.state_bytes,
                          "n_trainable": (lora.n_params if lora is not None
                                          else sum(m.weight.numel() for m in mods.values())),
                          "recipe": "narrow-grammar code completion (substitution recorded)"}
                    atlas.add("retention_ppl", value=ppl, bank_id=arm_id,
                              dose=float(dose) if dose is not None else None,
                              incumbent_id=("ewc_lite" if a.arm == "W1-U" else "lora_r32"),
                              model_id=PG1_REPO, model_sha=base_sha, flags=fl)
                    atlas.add("task_loss", value=tl, bank_id=arm_id,
                              dose=float(dose) if dose is not None else None,
                              incumbent_id=("ewc_lite" if a.arm == "W1-U" else "lora_r32"),
                              model_id=PG1_REPO, model_sha=base_sha, flags=fl)
                    atlas.add_ci("core_energy_gauge", list(gauge.values()), bank_id=arm_id,
                                 dose=float(dose) if dose is not None else None,
                                 model_id=PG1_REPO, model_sha=base_sha,
                                 flags={**fl, "n_cells": len(gauge)})
                    print(f"[{a.arm}] {arm_id} dose={dose} step={step+1}: ppl={ppl:.4f} "
                          f"(base {base_ppl:.4f}, +{ppl-base_ppl:.4f}) task_loss={tl:.4f}",
                          flush=True)
        if stamp is not None:
            # Recovery AFTER training: project the realized weight change onto the planted
            # direction, normalised by the planted amplitude. 1.0 = the stamp survived intact,
            # 0.0 = training erased it. Now a real question, because training moved the weights.
            recs = []
            with torch.no_grad():
                for key, (u, v, w0) in stamp.items():
                    Wn = mods[key].weight.detach().to(torch.float64).cpu().numpy()
                    recs.append(float(v @ (Wn - w0) @ u / (dose * (v @ v) * (u @ u) + 1e-30)))
            atlas.add_ci("watermark_recovery_after_training", recs, bank_id=arm_id.split("_", 1)[1],
                         dose=float(dose), incumbent_id="twin_band", model_id=PG1_REPO,
                         model_sha=base_sha,
                         flags={"amplitude": dose, "steps": a.steps, "lr": a.lr,
                                "battery_harm_ppl": ppl - base_ppl, "n_cells": len(recs),
                                "trained": True, "supersedes": "W1-K D21 tautological version"})
            print(f"[W1-K] {arm_id} amp={dose}: recovery_after_training="
                  f"{float(np.mean(recs)):.4f} harm={ppl - base_ppl:+.4f}", flush=True)
        results.append({"arm_id": arm_id, "dose": dose, "base_ppl": base_ppl,
                        "final_ppl": ppl, "final_task_loss": tl,
                        "retention_loss": ppl - base_ppl})
        if arm_id == "naive":
            dmg = ppl - base_ppl
            need = DAMAGE_MIN_REL * base_ppl
            damaged = bool(dmg > need)
            gates.record("W1U_damage_positive_control", damaged,
                         naive_retention_loss=dmg, required=need, base_ppl=base_ppl,
                         rel_threshold=DAMAGE_MIN_REL, steps=a.steps, lr=a.lr,
                         note=("the recipe must actually damage the base, or every update rule "
                               "ties at zero and the anchor-vs-control comparison is vacuous"))
            atlas.add("damage_positive_control", value=dmg, bank_id="naive",
                      model_id=PG1_REPO, model_sha=base_sha,
                      flags={"required": need, "passed": damaged, "base_ppl": base_ppl,
                             "rel_threshold": DAMAGE_MIN_REL})
            if a.arm == "W1-U" and not damaged and not a.allow_no_damage:
                D.write_json(out, "w1_u_void.json",
                             {"verdict": "VOID-no-damage", "naive_retention_loss": dmg,
                              "required": need, "steps": a.steps, "lr": a.lr,
                              "note": "no treatment arm was run; see W1U_damage_positive_control"})
                print(f"[W1-U] VOID-no-damage: naive raised ppl by {dmg:+.4f}, "
                      f"below the required {need:.4f}. Treatment arms NOT run.", flush=True)
                break
        del model, mods, opt, ewc, anchor, projector, lora
        torch.cuda.empty_cache()
        D.s3_sync(out, a.s3_prefix)

    gates.record(f"{a.arm}_completed", True, n_arms=len(arms), n_rows=atlas.n_rows,
                 optimizer=optinfo)
    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "results": results,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} arms={len(arms)}")


class _NullCtx:
    """Context manager for the arms with no activation hooks — so the training loop has ONE
    shape rather than a branch that could diverge between arms."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
