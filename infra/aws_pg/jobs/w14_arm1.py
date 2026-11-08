"""W14 ARM 1 — subspace privilege: does install need the LEARNED ΔW subspace, or just rank?

The EQ1 remainder. EQ1's only randomness control was PEFT-random init (A=0, unconstrained
search) — it conflates "wrong subspace" with "unconstrained search." Arm 1 adds two genuinely
NEW rank-confined inits at matched rank r, trained with the SAME frozen recipe (only the initial
subspace differs):

  topk   — A,B from the rank-r SVD of ΔW = W_after − W_before (= the no-retrain edit at init).
  inspan — a random rotation of r directions WITHIN ΔW's top-R_eff span (NEW).
  oospan — the same, in the ORTHOGONAL COMPLEMENT of ΔW's span (NEW; the matched-rank null EQ1 lacked).

inspan/oospan are matched-norm to topk (the construction gate asserts residual<1e-5, Gram−I<1e-5,
matched-norm±1e-3 per matrix, HALT on fail). H_subspace-privileged: topk ≫ inspan ≈ oospan ≈ FLOOR.
H_rank-sufficient: topk ≈ inspan ≈ oospan > FLOOR.

ONE (kind, rank, seed) cell per run (matches EQ1's per-cell design). Q1 onset bracket = topk runs
across ranks×seeds; Q2 crucial experiment = {topk, inspan, oospan} at r=r*. Primary readout =
RefCOCO acc@0.5 R_cap + <loc>-emission (EQ1 spine, verbatim); blank-image MMLU is an exploratory
cross-capability check. Distillation = KL(teacher‖student) on the install corpus, VERBATIM from EQ1.

FROZEN-RECIPE NOTE: lr is LOCKED at 1e-5 / 1500 steps — EQ1's confirmatory-ladder operating point
(eq1_r*_topk_s* run JSONs; NOT the eq1_trained_adapter.py:67 script default 2e-5). See LR comment below.
The recipe is frozen identically across the three kinds (that is what makes Q2 a clean subspace
test); only the initial subspace differs. Fail-loud; no try/except.

    pg run infra.aws_pg.jobs.w14_arm1 --type p5.4xlarge,g6e.8xlarge \
        --args="--kind inspan --rank 1024 --seed 0 --steps 1500 --eval-n 1975 \
                --controls-s3 s3://pg-lineage-769063704696/results/w14_arm23_controls/out/legA_controls.json" \
        --s3-prefix results/w14_arm1_q2_inspan_r1024_s0 --budget-min 240
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.e4_capability import _eval_rows, _generate, SEED
from infra.aws_pg.jobs.e4_compress_curve import _decoder_keys
from infra.aws_pg.jobs.eq1_trained_adapter import (
    _adapter_module, _img_hash, _install_items, _refcoco_emission, _score, _split_audit,
    INSTALL_CORPORA, WARMUP_RATIO,
)
from infra.pg import w14_localize as WL

LR = 1e-5            # LOCKED frozen recipe = EQ1's CONFIRMATORY-LADDER lr (verified: eq1_r512_topk_s0/s1
                     # run JSONs both record lr=1e-05; prereg §A1.5 "do not retune"). This is NOT the
                     # eq1_trained_adapter.py:67 script DEFAULT (2e-5, overridden at run time). 2e-5 would
                     # retune the recipe and break Q2's clean-subspace-test premise.
STEPS = 1500
TRAIN_BATCH = 4


def _cpu_real_deltas(dkeys, pg_key="PG"):
    """Load the REAL per-matrix ΔW = W_after − W_base on CPU (no GPU, no PaliGemma vision tower):
    AFTER = PG2's language_model (the PT teacher decoder), BASE = the gemma-2-2b decoder swapped
    into PG2's LM key/shape interface (= the student before any adapter). Same construction as
    diff_harvest.build_before_after, CPU-only, so the feasibility gate costs no GPU minute. Yields
    (key, dW_fp64) per decoder matrix in `dkeys`. fp64 weight-diff math (HARD RULE)."""
    import torch
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
    from infra.pg.contrast_a1 import L0_GEMMA_REPO, _lm_swap_state_dict
    from infra.pg.entities import ENTITIES, resolve_submodule_paths
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES[pg_key].repo_id, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager")
    paths = resolve_submodule_paths(pg)
    after_sd = pg.get_submodule(paths["language_model"]).state_dict()                 # PT teacher
    g2 = AutoModelForCausalLM.from_pretrained(
        L0_GEMMA_REPO, revision=ENTITIES["L0"].revision, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager")
    base_sd, _ = _lm_swap_state_dict(after_sd, g2.model.state_dict())                  # gemma-2-2b base
    for key in dkeys:
        yield key, (after_sd[key].detach().double() - base_sd[key].detach().double())


def _preflight_feasibility(dkeys, rank, kind, pg_key="PG"):
    """CPU pre-flight HALT gate (prereg §A1.4), BEFORE any model/GPU load: assert the rank-`rank`
    `kind` construction is feasible on the REAL ΔW for EVERY targeted matrix, then HALT fail-loud
    naming the binding matrix — an impossible (rank,kind) never burns a GPU minute. No try/except.

    PER-KIND (gates only the kind THIS run constructs; each Q2 arm = a separate run):
      topk   — the rank-k SVD truncation, ALWAYS constructable (uses min(k,rank) components) → skip.
      inspan — needs k ≤ R_eff (draw k from ΔW's R_eff-dim energy span).
      oospan — needs k ≤ out_d−R_eff AND k ≤ in_d−R_eff (draw k from the orthogonal complement).
    At the install rank the high-eff-rank attention ΔW (kv_proj, out_d=1024, R_eff>512) can host
    in-span but NOT a matched-rank out-of-span — a DISCLOSED geometric limitation of the oospan arm,
    not a reason to block the decisive top-k vs in-span cut (§A1.3). So an oospan run HALTs while the
    topk/inspan runs proceed."""
    if kind == "topk":
        print(f"[W14-A1] CPU pre-flight: kind=topk (rank-k SVD truncation, always constructable) — gate skipped", flush=True)
        return
    n = 0
    for key, dW in _cpu_real_deltas(dkeys, pg_key):
        feas = WL.subspace_feasible(dW, rank, kind)
        assert feas["feasible"], (
            f"CPU pre-flight INFEASIBLE: r*={rank} kind={kind} on {key} — "
            f"R_eff={feas['R_eff']} max_rank_inspan={feas['max_rank_inspan']} "
            f"max_rank_oospan={feas['max_rank_oospan']} ({feas['reason']}). "
            f"escalate (lower r* / disclose the geometric obstruction); do NOT silently cap.")
        n += 1
    print(f"[W14-A1] CPU pre-flight PASSED: r*={rank} kind={kind} feasible on all {n} matrices", flush=True)


def _attach_subspace_adapters(lm, lm_after, dkeys, rank, kind, device, gen):
    """Replace each target decoder Linear with a rank-r adapter whose init sits in the chosen
    subspace (topk/inspan/oospan), via the tested w14_localize.subspace_AB + construction_audit.
    Returns (n_params, audit_summary). HALTs if ANY matrix fails its construction gate."""
    import torch
    RankKAdapter = _adapter_module()
    after_sd = lm_after.state_dict()
    n_params, audits, worst = 0, [], {"A_subspace_residual": 0.0, "B_subspace_residual": 0.0,
                                      "matched_norm_abs_err": 0.0}
    for key in dkeys:
        modpath = key[:-len(".weight")]
        lin = lm.get_submodule(modpath)
        base_W = lin.weight.detach()
        dW = (after_sd[key].detach().cpu().double() - base_W.detach().cpu().double())  # ΔW, CPU fp64
        A, B = WL.subspace_AB(dW, rank, kind, gen)                          # fp64 (out,k),(k,in)
        aud = WL.construction_audit(A, B, dW, rank, kind)
        if not aud["ok"]:
            raise RuntimeError(f"construction gate FAILED on {key}: {aud}")
        worst["A_subspace_residual"] = max(worst["A_subspace_residual"], aud["A_subspace_residual"])
        worst["B_subspace_residual"] = max(worst["B_subspace_residual"], aud["B_subspace_residual"])
        worst["matched_norm_abs_err"] = max(worst["matched_norm_abs_err"], aud["matched_norm_abs_err"])
        audits.append({"key": key, **{k: aud[k] for k in ("R_eff", "A_subspace_residual",
                                                          "B_subspace_residual", "matched_norm_abs_err")}})
        adapter = RankKAdapter(lin, A.to(device).to(torch.bfloat16), B.to(device).to(torch.bfloat16), 1.0)
        parent, child = modpath.rsplit(".", 1)
        setattr(lm.get_submodule(parent), child, adapter)
        n_params += A.numel() + B.numel()
    return n_params, {"n_matrices": len(audits), "worst": worst, "per_matrix": audits}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg-variant", choices=["pt", "mix"], default="pt",
                    help="base→pt (W14) or base→mix (second-ΔW generality gate)")
    ap.add_argument("--kind", choices=["topk", "inspan", "oospan"], required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--corpus", choices=list(INSTALL_CORPORA), default="refcoco")
    ap.add_argument("--corpus-n", type=int, default=8000)
    ap.add_argument("--eval-n", type=int, default=1975)
    ap.add_argument("--eval-tasks", default="refcoco")
    ap.add_argument("--train-batch", type=int, default=TRAIN_BATCH)
    ap.add_argument("--controls-s3", default=None, help="legA_controls.json for the blank-image MMLU cross-check")
    ap.add_argument("--mmlu-cross-check", action="store_true", help="also score blank-image MMLU (exploratory)")
    ap.add_argument("--shuffle-teacher", action="store_true", help="label-permuted teacher (null control)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    device = "cuda"
    assert torch.cuda.is_available(), "W14 Arm 1 needs a GPU (trained adapter)."
    if a.smoke:
        a.corpus_n, a.steps, a.eval_n = 64, 20, 32
    torch.manual_seed(a.seed)
    eval_tasks = tuple(t.strip() for t in a.eval_tasks.split(","))
    print(f"[W14-A1] kind={a.kind} rank={a.rank} seed={a.seed} steps={a.steps} lr={a.lr} "
          f"corpus={a.corpus} eval_n={a.eval_n} smoke={a.smoke} gpu={torch.cuda.get_device_name(0)}", flush=True)

    pg_key = "PG_mix" if a.pg_variant == "mix" else "PG"
    build_pair_name = "base-mix" if a.pg_variant == "mix" else "base-pt"
    print(f"[W14-A1] pg_variant={a.pg_variant} (pg_key={pg_key}, pair={build_pair_name})", flush=True)
    dkeys = _decoder_keys()
    # FINDING #1: CPU pre-flight feasibility gate (per-kind) BEFORE any model/GPU load (HALT on
    # infeasible (r*,kind); topk skips — always constructable).
    _preflight_feasibility(dkeys, a.rank, a.kind, pg_key)

    # data first (EQ1 order: split-audit fail-fast before the model build)
    corpus = INSTALL_CORPORA[a.corpus]
    eval_rows = {t: _eval_rows(B.TASKS[t], a.eval_n, SEED) for t in eval_tasks}
    eval_hashes = {_img_hash(B.TASKS[t].image_fn(r)) for t in eval_tasks for r in eval_rows[t]}
    install_items, install_hashes = _install_items(B, a.corpus_n, SEED + a.seed, eval_hashes, corpus)
    _split_audit(B, install_hashes, eval_rows, out)

    pg_after, pg_before, proc, paths, _it, gate = H.build_pair(build_pair_name, device)
    lm = pg_before.get_submodule(paths["language_model"])         # student decoder host (base)
    lm_after = pg_after.get_submodule(paths["language_model"])     # teacher decoder (adapted)

    # FLOOR + CEILING for R_cap (EQ1 spine)
    anchors = {}
    for t in eval_tasks:
        mf, ef, _ = _score(pg_before, proc, B.TASKS[t], eval_rows[t], device)
        mc, ec, _ = _score(pg_after, proc, B.TASKS[t], eval_rows[t], device)
        anchors[t] = {"floor": mf, "ceiling": mc, "floor_emit": ef, "ceiling_emit": ec}
        print(f"[W14-A1] {t} FLOOR={mf:.4f}(emit={ef}) CEILING={mc:.4f}(emit={ec})", flush=True)

    # attach the rank-confined adapter of the chosen subspace + run the construction gate
    for p in pg_before.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device="cpu").manual_seed(SEED + a.seed)
    n_adapter, audit = _attach_subspace_adapters(lm, lm_after, dkeys, a.rank, a.kind, device, gen)
    (out / "construction_audit.json").write_text(json.dumps(
        {"kind": a.kind, "rank": a.rank, **audit}, indent=2, default=str))
    print(f"[W14-A1] construction gate PASSED: {audit['n_matrices']} matrices, "
          f"worst A_resid={audit['worst']['A_subspace_residual']:.2e} "
          f"matched_norm_err={audit['worst']['matched_norm_abs_err']:.2e}", flush=True)
    trainable = [p for p in pg_before.parameters() if p.requires_grad]
    assert int(sum(p.numel() for p in trainable)) == n_adapter, "freeze leak: trainable != adapter params"

    # init-eval (for topk this IS the no-retrain edit; for in/out-span it is the matched-norm start)
    init = {}
    for t in eval_tasks:
        m0, e0, _ = _score(pg_before, proc, B.TASKS[t], eval_rows[t], device)
        d = anchors[t]["ceiling"] - anchors[t]["floor"]
        init[t] = {"metric": m0, "emit": e0, "R_cap": ((m0 - anchors[t]["floor"]) / d if d > 0 else float("nan"))}
        print(f"[W14-A1] {t} INIT metric={m0:.4f} emit={e0} R_cap={init[t]['R_cap']:.3f}", flush=True)

    # distillation: KL(teacher || student) on suffix tokens (EQ1 recipe, verbatim)
    if a.shuffle_teacher:
        rng = np.random.default_rng(SEED + a.seed)
        perm = rng.permutation(len(install_items))
        for it, j in zip(install_items, perm):
            it["row_target"] = install_items[j]["row"]

    def collate(items):
        imgs = [B.TASKS[it["t"]].image_fn(it["row"]).convert("RGB") for it in items]
        prompts = [B.TASKS[it["t"]].prompt_fn(it["row"]) for it in items]
        tgts = [it.get("row_target", it["row"]) for it in items]
        suffixes = [B.TASKS[it["t"]].target_fn(tr) for it, tr in zip(items, tgts)]
        enc = proc(text=prompts, images=imgs, suffix=suffixes, return_tensors="pt", padding="longest")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        return enc

    loader = DataLoader(install_items, batch_size=a.train_batch, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(WARMUP_RATIO * a.steps),
                          num_training_steps=a.steps)
    pg_before.train()
    step, losses, done = 0, [], False
    while not done:
        for enc in loader:
            enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
            ii, am, pv, labels = enc["input_ids"], enc["attention_mask"], enc["pixel_values"], enc["labels"]
            with torch.no_grad():
                tl = pg_after(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
            sl = pg_before(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
            mask = (labels[:, 1:] != -100)
            tp = torch.log_softmax(tl[mask].float(), dim=-1)
            sp = torch.log_softmax(sl[mask].float(), dim=-1)
            loss = (tp.exp() * (tp - sp)).sum(-1).mean()
            assert torch.isfinite(loss), f"non-finite loss step {step}: {loss.item()} (diverged — reported)"
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item())); step += 1
            if step % 25 == 0 or step == 1:
                print(f"[W14-A1] step {step}/{a.steps} kl_loss={losses[-1]:.4f}", flush=True)
            if step >= a.steps:
                done = True
                break

    # trained eval: RefCOCO R_cap + emission (primary)
    pg_before.eval()
    trained = {}
    for t in eval_tasks:
        m1, e1, s1 = _score(pg_before, proc, B.TASKS[t], eval_rows[t], device)
        d = anchors[t]["ceiling"] - anchors[t]["floor"]
        trained[t] = {"metric": m1, "emit": e1,
                      "R_cap": ((m1 - anchors[t]["floor"]) / d if d > 0 else float("nan")), "samples": s1}
        print(f"[W14-A1] {t} TRAINED metric={m1:.4f} emit={e1} R_cap={trained[t]['R_cap']:.3f}", flush=True)

    # blank-image MMLU cross-capability check (exploratory; install↔forget map)
    mmlu_cross = None
    if a.mmlu_cross_check:
        from infra.pg import text_eval as TE
        bi = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=(2 if a.smoke else None), gray=128)
        mmlu_cross = {"acc_pct": bi["acc_pct"], "n": bi["n"]}
        print(f"[W14-A1] blank-image MMLU cross-check = {bi['acc_pct']:.2f} (exploratory)", flush=True)

    result = {
        "experiment": "w14_arm1_subspace", "status": "confirmatory",
        "config": {"kind": a.kind, "rank": a.rank, "seed": a.seed, "steps": a.steps, "lr": a.lr,
                   "pg_variant": a.pg_variant, "corpus": a.corpus, "corpus_n": len(install_items),
                   "eval_n": a.eval_n, "eval_tasks": list(eval_tasks), "n_adapter_params": n_adapter,
                   "shuffle_teacher": a.shuffle_teacher},
        "provenance": {"teacher": "google/paligemma2-3b-pt-224", "student_decoder": "google/gemma-2-2b",
                       "swap_gate": gate, "frozen_except": "rank-r decoder adapter",
                       "recipe_note": "lr LOCKED 1e-5 / 1500 steps — EQ1's confirmatory-ladder lr (run JSONs), NOT the eq1_trained_adapter.py:67 default 2e-5 (§A1.3/§A1.5)"},
        "construction_gate": {"passed": True, "worst": audit["worst"]},
        "anchors": anchors, "init_eval": init, "trained_eval": trained, "mmlu_cross_check": mmlu_cross,
        "train_loss": {"start": (losses[0] if losses else None),
                       "end_mean10": (float(np.mean(losses[-10:])) if losses else None)},
    }
    (out / "arm1_subspace.json").write_text(json.dumps(result, indent=2, default=str))
    print("W14_ARM1_OK", json.dumps({t: {"trained_R_cap": round(trained[t]["R_cap"], 3),
                                         "emit": trained[t]["emit"]} for t in eval_tasks}))


if __name__ == "__main__":
    main()
