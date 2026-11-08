"""M-Q stage 2 — fine-tuning under a gauge: THE causal Adam-basis test (standing W1-Q). Wave 1.

Stage 1 proved PG1 has an exact residual-stream gauge: fold every RMSNorm gain to 1, rotate the
stream, and the logits are unchanged to the arithmetic floor. Stage 2 spends that gauge.

    THE QUESTION. The sink is a set of COORDINATES. Adam's second moment is per-COORDINATE. So
    "the core is a gradient sink" may be a statement about the model or a statement about the
    basis Adam happens to be counting in. A rotation changes the basis and NOT the function, which
    makes it the one intervention that separates the two.

    REFUTER (pre-logged): the Hadamard arm moves the sink no more than the unrotated arm.

FIVE ARMS, ONE RECIPE, THREE SEEDS EACH:

    unrotated   fold only, no rotation      — the control that isolates the ROTATION. The fold
                                              itself rewrites every reader, so an unfolded
                                              baseline would confound gauge with reassociation.
    hadamard    fold + Hadamard             — the sharpest scatter: every coordinate spread over
                                              all others uniformly.
    align       fold + align-to-consensus   — the opposite extreme: the sink IS the first 32
                                              coordinates.
    haar_s0/s1  fold + Haar draw            — the matched-random gauge. "Any rotation does this"
                                              and "THIS rotation does this" are different claims,
                                              and only the Haar arms can tell them apart. They are
                                              also §3's random reference basis for a coordinate
                                              claim.

WHAT IS MEASURED, AND WHY IT IS TRANSPORTED. The bank U lives in the ORIGINAL stream basis. Both
operations move it, and both must be applied or the arm would measure a different subspace in each
gauge and call the difference an effect:

    fold      a reader becomes W diag(g), so a row-space direction pushes forward as u -> diag(g) u
    rotate    a reader becomes W R^T, so it pushes forward again as u -> R u

so the transported bank is orth(R diag(g) U), recomputed per gauge, per layer.

SCOPE IS q_proj ONLY, and that is not a shortcut: q_proj's input IS the gauged residual stream,
while o_proj reads the attention output, which no residual-stream rotation touches. Training both
would average a gauged cell with an ungauged one. ERLANGEN independently found H_repelled scoped
to q_proj.

THE STATISTIC IS THE SINK ITSELF: gradient share into the subspace, motion share out of it, and
their ratio — TENANT's "16x gradient in, 0.2x motion out" object, measured under each gauge.

§0.2.7 plus amendment A1: optimizer-mechanics arm, so ExactAdam and no approximation. fp32
throughout, TF32 off — stage 1 established that TF32 alone breaks a 1e-4 logit gate.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import mercator_arch as AR
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_gauge as G
from infra.pg import mercator_program as P

ARM = "M-Q"
BANK = "core32"
# The control battery for the MOTION statistic. The gauge arms are the treatment axis; these are
# the subspace axis, and both are needed — "the sink moved more under Hadamard" is only a claim
# about the sink if a random subspace did not move more too.
MOTION_BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"


def orth(A):
    """An orthonormal basis for the column space of A, sized to A's rank as given."""
    Q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    return Q[:, :A.shape[1]]


def consensus_subspace(bases, k: int):
    """Top-k eigenvectors of sum_L U_L U_L^T — the one stream-level address a single gauge can act
    on.

    A residual-stream rotation is ONE matrix shared by every layer, so "align the sink to the first
    k axes" is only well posed for a subspace defined on the stream, not per layer. This is that
    subspace, and its per-layer agreement is reported alongside so a weak consensus cannot pass as
    a strong one.
    """
    d = bases[0].shape[0]
    S = np.zeros((d, d), dtype=np.float64)
    for U in bases:
        S += U @ U.T
    w, V = np.linalg.eigh(S)
    C = V[:, ::-1][:, :k]
    agree = [float((np.linalg.norm(C.T @ U, "fro") ** 2) / U.shape[1]) for U in bases]
    return C, {"eigenvalue_top": float(w[-1]), "eigenvalue_kth": float(w[-k]),
               "energy_captured": float(w[::-1][:k].sum() / w.sum()),
               "per_layer_agreement_mean": float(np.mean(agree)),
               "per_layer_agreement_min": float(np.min(agree)),
               "note": ("agreement is ||C^T U||_F^2 / k, i.e. the fraction of each layer's bank "
                        "that the consensus captures; 1.0 would mean every layer shares one "
                        "subspace exactly")}


def main():
    ap = argparse.ArgumentParser(description="M-Q stage 2 — fine-tuning under a gauge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gauges", default="unrotated,hadamard,align,haar_s0,haar_s1")
    ap.add_argument("--seeds", default="1337,1338,1339")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--n-gate-seq", type=int, default=4)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gauges = [g.strip() for g in a.gauges.split(",") if g.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    assert len(seeds) >= 3, f"the arm compares gauges at matched seeds; got {seeds}"
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_q_stage2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _decoder_layers,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows
    from infra.aws_pg.jobs.mercator_mq import (_layers, _reference_logits, _untie_lm_head,
                                               fold_model, rotate_model, scale_gauge_model)

    assert torch.cuda.is_available(), "no CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)
    gate_texts = ["The capital of France is Paris, and the capital of Italy is",
                  "In 1969 the first humans landed on the",
                  "Water boils at one hundred degrees",
                  "The quick brown fox jumps over the lazy"][: a.n_gate_seq]
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seeds[0])
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    layers_idx = list(range(H.N_LAYERS))
    raw_banks = {L: {b: WT.load_bank(bank_dir, "q_proj", L, b) for b in MOTION_BANKS}
                 for L in layers_idx}

    def load():
        """A fresh fp32 PG1 with lm_head untied. fp32 because the gauge gate's tolerance is 1e-4
        on logits and bf16's own rounding is larger than that."""
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            WT.PG1_REPO, revision=base_sha, torch_dtype=torch.float32, device_map="cuda",
            attn_implementation="eager")
        _untie_lm_head(m)
        return m

    # The ungauged reference logits, the input-layernorm gains the fold will consume, and the
    # ARITHMETIC NOISE FLOOR. Stage 1 established that the spec's absolute 1e-4 logit tolerance is
    # below what the surgery pipeline itself costs in fp32, so the binding comparison is against a
    # measured floor: a CONSTANT-scale gauge is mathematically the identity on the function but
    # forces exactly the same reassociation the real fold does, so whatever it moves the logits by
    # is the price of the pipeline rather than the price of the gauge.
    model = load()
    model.eval()
    ref_logits = _reference_logits(model, processor, "cuda", gate_texts, blank)
    _lm, layers = _layers(model)
    gains = {L: G.gemma_effective_gain(
        layers[L].input_layernorm.weight.detach().cpu().numpy()) for L in layers_idx}
    n_scaled = scale_gauge_model(model, G.REASSOCIATION_PROBE_SCALE)
    floor = G.function_preservation(
        ref_logits, _reference_logits(model, processor, "cuda", gate_texts, blank))
    assert floor["max_abs_diff"] > 0, (
        "the constant-scale gauge moved the logits by exactly zero, so there is no measured floor "
        "and the spec's absolute tolerance is the correct gate; re-run without the floor branch")
    at.add("gauge_reassociation_noise_floor", value=float(floor["max_abs_diff"]), bank_id=None,
           flags={**floor, "what": "constant-scale gauge", "n_norms": n_scaled,
                  "probe_scale": G.REASSOCIATION_PROBE_SCALE})
    print(f"[M-Q2] reassociation noise floor = {floor['max_abs_diff']:.3e} "
          f"({n_scaled} norms)", flush=True)
    del model, _lm, layers
    torch.cuda.empty_cache()

    # The banks, pushed forward through the FOLD (the rotation is applied per gauge below).
    folded_banks = {L: {b: orth(gains[L][:, None] * raw_banks[L][b]) for b in MOTION_BANKS}
                    for L in layers_idx}
    d = folded_banks[0][BANK].shape[0]
    consensus, cons_info = consensus_subspace([folded_banks[L][BANK] for L in layers_idx],
                                              folded_banks[0][BANK].shape[1])
    at.add("gauge_consensus_energy", value=cons_info["energy_captured"], bank_id=BANK,
           flags=cons_info)
    print(f"[M-Q2] consensus core: energy {cons_info['energy_captured']:.4f}, per-layer agreement "
          f"mean {cons_info['per_layer_agreement_mean']:.4f}", flush=True)

    def rotation_for(name):
        if name == "unrotated":
            return None
        if name == "hadamard":
            return G.hadamard_rotation(d)
        if name == "align":
            return G.align_rotation(consensus, seed=seeds[0])
        assert name.startswith("haar_s"), f"unknown gauge {name!r}"
        return AR.random_rotation(d, seed=int(name.split("_s")[1]))

    results, rows = {}, []
    for gname in gauges:
        model = load()
        model.eval()
        fold_info = fold_model(model)
        R = rotation_for(gname)
        rot_info = rotate_model(model, R) if R is not None else {"per_group": {}, "reader": 0,
                                                                 "writer": 0}
        fp = G.function_preservation(ref_logits,
                                     _reference_logits(model, processor, "cuda", gate_texts,
                                                       blank))
        # THE GATE IS PER GAUGE. A gauge whose function is not preserved is not a gauge, and a
        # fine-tune under it would be a fine-tune of a different model wearing this arm's label.
        fp_gate = G.assert_function_preserved(fp, what=f"gauge:{gname}",
                                              noise_floor=floor["max_abs_diff"])
        at.add("gauge_logit_maxabs", value=float(fp["max_abs_diff"]), bank_id=None,
               flags={**fp_gate, "gauge": gname, "rotation_applied": R is not None,
                      "n_norms_folded": fold_info["n_norms_folded"],
                      "ratio_to_noise_floor": fp["max_abs_diff"] / floor["max_abs_diff"],
                      **rot_info})
        print(f"[M-Q2] {gname}: fold ok ({fold_info['n_norms_folded']} norms), "
              f"logit max|d| = {fp['max_abs_diff']:.3e}", flush=True)

        # The banks in THIS gauge's coordinates, ON THE GPU. The gradient share is accumulated
        # every step over 18 cells x 8 banks; done on the host in fp64 that is ~40 GFLOP of CPU
        # work per step and would dominate the arm's runtime, so the projections run on device in
        # fp32 and only the resulting SCALARS are accumulated in Python floats.
        U_np = {L: {b: (folded_banks[L][b] if R is None else orth(R @ folded_banks[L][b]))
                    for b in MOTION_BANKS} for L in layers_idx}
        U = {L: {b: torch.as_tensor(U_np[L][b], device="cuda", dtype=torch.float32)
                 for b in MOTION_BANKS} for L in layers_idx}
        for p in model.parameters():
            p.requires_grad_(False)
        mods = T.trainable_modules(_decoder_layers(model), types=("q_proj",), layers=layers_idx)
        for mod in mods.values():
            mod.weight.requires_grad_(True)
        pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
        train_b = build_batches(processor, train_rows, "cuda")
        eval_b = build_batches(processor, eval_rows, "cuda")
        base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])

        for seed in seeds:
            for k, m in mods.items():
                m.weight.data.copy_(pristine[k])
            opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
            rng = np.random.default_rng(H.cell_seed(seed, ARM, gname, "stage2"))
            grad_in = {b: 0.0 for b in MOTION_BANKS}
            grad_tot = 0.0
            model.train()
            for _step in range(a.steps):
                batch = train_b[int(rng.integers(0, len(train_b)))]
                opt.zero_grad()
                model(input_ids=batch["input_ids"], labels=batch["labels"]).loss.backward()
                # GRADIENT SHARE, accumulated over the run. This is the "in" half of the sink and
                # it has to be read from the gradient itself — the weights only carry the "out".
                with torch.no_grad():
                    for (t, L), m in mods.items():
                        g = m.weight.grad.to(torch.float32)
                        grad_tot += float((g * g).sum())
                        for b in MOTION_BANKS:
                            grad_in[b] += float((g @ U[L][b]).pow(2).sum())
                opt.step()
            model.eval()
            ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
            task = eval_task_loss(model, eval_b)
            move_in, move_tot = {b: 0.0 for b in MOTION_BANKS}, 0.0
            with torch.no_grad():
                for (t, L), m in mods.items():
                    dW = m.weight.detach().to(torch.float32) - pristine[(t, L)]
                    move_tot += float((dW * dW).sum())
                    for b in MOTION_BANKS:
                        move_in[b] += float((dW @ U[L][b]).pow(2).sum())
            assert move_tot > 0 and grad_tot > 0, f"{gname}/s{seed}: nothing moved"
            for b in MOTION_BANKS:
                gs, ms = grad_in[b] / grad_tot, move_in[b] / move_tot
                # THE SINK ITSELF: share of motion out over share of gradient in. Below 1 means
                # the optimizer delivers less motion into the subspace than the gradient asked
                # for — TENANT's suppression, expressed as a single gauge-comparable ratio.
                row = {"gauge": gname, "seed": seed, "bank": b, "grad_share": gs,
                       "motion_share": ms, "delivery_ratio": ms / gs,
                       "ppl": ppl, "task_loss": task, "base_ppl": base_ppl,
                       "total_motion": float(np.sqrt(move_tot))}
                rows.append(row)
                at.add("gauge_delivery_ratio", value=row["delivery_ratio"], bank_id=b,
                       model_id=WT.PG1_REPO, model_sha=base_sha,
                       flags={**row, "gauge_rotation": gname, "steps": a.steps, "lr": a.lr,
                              "optimizer": optinfo,
                              "definition": ("(||dW U||^2/||dW||^2) / (sum_t ||g_t U||^2 / "
                                             "sum_t ||g_t||^2); < 1 is suppression")})
                at.add("gauge_motion_share", value=ms, bank_id=b, model_id=WT.PG1_REPO,
                       model_sha=base_sha, flags={**row, "gauge_rotation": gname})
                at.add("gauge_grad_share", value=gs, bank_id=b, model_id=WT.PG1_REPO,
                       model_sha=base_sha, flags={**row, "gauge_rotation": gname})
            at.add("gauge_ppl", value=ppl, bank_id=None, corpus_id=PPL_CORPUS,
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"gauge": gname, "seed": seed, "base_ppl": base_ppl})
            at.add("gauge_task_loss", value=task, bank_id=None, corpus_id=TASK_CORPUS,
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"gauge": gname, "seed": seed})
            core = [r for r in rows if r["gauge"] == gname and r["seed"] == seed
                    and r["bank"] == BANK][0]
            print(f"[M-Q2] {gname:10s} s{seed}: core grad_share={core['grad_share']:.5f} "
                  f"motion_share={core['motion_share']:.5f} "
                  f"delivery={core['delivery_ratio']:.4f} ppl={ppl:.3f}", flush=True)
        results[gname] = {"fold": fold_info, "rotation": rot_info, "gate": fp_gate,
                          "base_ppl": base_ppl}
        del model, mods, pristine, train_b, eval_b, U, U_np
        torch.cuda.empty_cache()

    # ---- the refuter, adjudicated ------------------------------------------------------------
    at.check_controls_per_statistic(["gauge_delivery_ratio", "gauge_motion_share",
                                     "gauge_grad_share"])
    at.check_readout_corpus(["gauge_ppl", "gauge_task_loss"])

    def core_series(gname, key):
        return [r[key] for r in rows if r["gauge"] == gname and r["bank"] == BANK]

    summary = {g: {k: float(np.mean(core_series(g, k)))
                   for k in ("grad_share", "motion_share", "delivery_ratio", "ppl", "task_loss")}
               for g in gauges}
    for g in gauges:
        summary[g]["motion_share_seeds"] = core_series(g, "motion_share")
    haar = [g for g in gauges if g.startswith("haar_s")]
    haar_motion = [v for g in haar for v in core_series(g, "motion_share")]
    moved_more = None
    if "hadamard" in summary and "unrotated" in summary:
        moved_more = bool(summary["hadamard"]["motion_share"]
                          > summary["unrotated"]["motion_share"])
        # THE REFUTER, and it is one-sided by design only because the spec's refuter is: "Hadamard
        # moves the sink NO MORE than unrotated". The Haar band is recorded next to it so that
        # "more" can be read against what ANY rotation does, which is the difference between a
        # claim about Hadamard and a claim about rotating at all.
        gates.record("mq2_hadamard_moves_sink_more_than_unrotated", moved_more,
                     hadamard=summary["hadamard"]["motion_share"],
                     unrotated=summary["unrotated"]["motion_share"],
                     haar_band=[float(np.min(haar_motion)), float(np.max(haar_motion))]
                     if haar_motion else None,
                     outside_haar_band=(bool(summary["hadamard"]["motion_share"]
                                             > max(haar_motion)) if haar_motion else None),
                     note=("refuter fires when this is False; outside_haar_band False means the "
                           "effect is 'a rotation', not 'the Hadamard rotation'"))
        s = M.surprise(summary["hadamard"]["motion_share"],
                       summary["unrotated"]["motion_share"],
                       core_series("unrotated", "motion_share"))
        sur.add(arm=ARM, expected_ref=expected_ref, cell="hadamard_vs_unrotated",
                statistic="gauge_motion_share/core32", s=s)
    if "align" in summary and "unrotated" in summary:
        gates.record("mq2_align_concentrates_sink", summary["align"]["motion_share"]
                     > summary["unrotated"]["motion_share"],
                     align=summary["align"]["motion_share"],
                     unrotated=summary["unrotated"]["motion_share"],
                     note=("align makes the sink axis-aligned; if the address is an Adam-basis "
                           "property the three gauges should order predictably"))
    # The function is identical in every gauge, so any outcome difference is the optimizer's doing.
    ppls = [summary[g]["ppl"] for g in gauges]
    gates.record("mq2_outcomes_differ_across_gauges",
                 bool(max(ppls) - min(ppls) > 0), ppl_by_gauge={g: summary[g]["ppl"]
                                                                for g in gauges},
                 note=("the gauges are function-identical before training, so a spread here is "
                       "attributable to the optimizer's basis dependence and nothing else"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_stage2_result.json",
                 {"arm": ARM, "stage": "2 (fine-tuning under a gauge)",
                  "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "gauges": gauges, "seeds": seeds, "steps": a.steps, "lr": a.lr,
                  "trainable": "q_proj x 18 layers", "optimizer": optinfo,
                  "consensus": cons_info, "noise_floor": floor,
                  "by_gauge": results, "summary": summary,
                  "rows": rows,
                  "hadamard_moves_sink_more_than_unrotated": moved_more,
                  "transport": ("banks pushed forward as orth(R diag(g) U): the fold makes a "
                                "reader W diag(g) and the rotation makes it W R^T")})
    print("MQ2_OK " + json.dumps({"rows": info["n_rows"],
                                  "motion_share": {g: summary[g]["motion_share"] for g in gauges},
                                  "hadamard_more": moved_more}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
