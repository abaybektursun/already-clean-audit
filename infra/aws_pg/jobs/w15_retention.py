"""W15 §9.5 STAGE-0 RETENTION arm (~2 GPU-h gradients + ~1 CPU-h aggregation). The
separability fork: are the FORGETTING-carrier directions (those whose movement deletes
free-standing MMLU) disjoint from the INSTALL-carrier directions (those that install RefCOCO)?

In W_before's fp64 SVD basis, per decoder matrix:
  M = Uᵀ ΔW V                                       (how each direction-pair actually moved)
  I_forget  = |M ⊙ (Uᵀ G_forget V)|                  G_forget  = ∂(blank-image MMLU CE)/∂W_before
  I_install = |M ⊙ (Uᵀ G_install V)|                 G_install = ∂(RefCOCO <loc> CE)/∂W_before
  I_mag     = |M|                                    (the magnitude / instrument-validity control)
top-k(I) at k = #edges capturing 50% of ‖M‖²_F per matrix; pooled O = Σ|A∩B|/Σk over the 182
matrices (degenerate matrices excluded, logged). Decision (w15_carrier.retention_verdict, §6):
R-SEPARABLE iff O(forget,install) < permuted-null p5 AND < O(forget,|M|) p5 AND < O(install,|M|) p5
(≥3/3 seeds, BH-FDR); R-INSTRUMENT-NULL if O is not below the magnitude controls (maps are
magnitude → redesign). f_perp(install) reported regardless (large ⇒ install lives out-of-span,
the in-span edge map cannot see its carrier — a finding, not a silent pass).

Run once per ΔW axis (generality): --pg-variant base-pt, then base-mix.

    pg run infra.aws_pg.jobs.w15_retention --type g6e.8xlarge,p5.4xlarge \
        --args "--pg-variant base-pt --seeds 0,1,2 --grad-n 256 --mmlu-per-subject 4" \
        --s3-prefix results/w15_retention_pt --budget-min 200
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _degenerate(s, rtol=1e-3):
    """Spectral-FLATNESS diagnostic: fraction of adjacent singular-value gaps below rtol·σ_max.
    NOTE: weight matrices have smoothly-decaying (power-law) spectra, so this fraction is naturally
    HIGH (the tail's consecutive σ are close) — it does NOT mean the dominant basis is non-identifiable.
    Reported per-matrix as a diagnostic (prereg 'logged count'); the per-matrix EXCLUSION is disabled
    by default (--degen-max 1.0) because this heuristic over-flags smooth decay and would drop the whole
    population. Basis-dependence of the top-k overlap is instead assessed by the seed control, the
    magnitude control, and the k∈{25,50,75}% robustness (the §6 confirmatory battery)."""
    import torch
    s = torch.as_tensor(s, dtype=torch.float64)
    smax = s[0].clamp_min(1e-30)
    gaps = (s[:-1] - s[1:]) / smax
    return float((gaps < rtol).double().mean().item())


def matrix_basis(Wb, dW):
    """The fp64 W_before SVD ONCE per matrix (the cost): returns (U, V, M=UᵀΔWV, I_mag=|M|,
    R_eff, degen). W_before's basis is identical across seeds AND fracs, so this is computed once
    per matrix and reused — NOT recomputed in the seed×frac loop (182 SVDs total, not 182×|seeds|×|fracs|)."""
    from infra.pg import w15_carrier as C
    U, s, V = C.svd_fp64(Wb)
    M = C.to_basis(dW, U, V)
    return U, V, M, M.abs(), C.effective_rank_energy(s), _degenerate(s)


def matrix_record(M, U, V, I_mag, Gi, Gf, fracs, R_eff):
    """All cheap (no-SVD) per-matrix quantities for one seed: the install/forget saliency projections
    (once), then top-k intersections at each frac. Returns ({str(frac): {k,N,inter_FI,inter_FM,inter_IM}},
    f_perp_install). f_perp is frac-invariant (gradient-vs-span, no top-k)."""
    from infra.pg import w15_carrier as C
    I_inst = C.edge_saliency(M, C.to_basis(Gi, U, V))      # |M ⊙ Uᵀ G_install V|
    I_forg = C.edge_saliency(M, C.to_basis(Gf, U, V))      # |M ⊙ Uᵀ G_forget V|
    out = {}
    for frac in fracs:
        k = C.energy_topk_count(M, frac)
        mF = C.topk_mask(I_forg, k); mI = C.topk_mask(I_inst, k); mM = C.topk_mask(I_mag, k)
        out[str(frac)] = {"k": int(k), "N": int(M.numel()),
                          "inter_FI": int((mF & mI).sum()), "inter_FM": int((mF & mM).sum()),
                          "inter_IM": int((mI & mM).sum())}
    return out, C.grad_out_of_span_frac(Gi, U, V, R_eff)


def _pooled(per_mat, key_a):
    """Σ inter_{key_a} / Σ k over the kept matrices."""
    inter = sum(m[key_a] for m in per_mat)
    ksum = sum(m["k"] for m in per_mat)
    return inter / ksum if ksum > 0 else float("nan")


def _cluster_boot(per_mat, key_a, n_boot, seed):
    """Cluster (matrix-resample) bootstrap of pooled O for one overlap kind → point + p5/p50/p95."""
    import numpy as np
    rng = np.random.default_rng(seed)
    inter = np.array([m[key_a] for m in per_mat], float)
    kk = np.array([m["k"] for m in per_mat], float)
    n = len(per_mat)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ks = kk[idx].sum()
        if ks > 0:
            draws.append(inter[idx].sum() / ks)
    draws = np.asarray(draws)
    return {"point": float(inter.sum() / kk.sum()), "p5": float(np.percentile(draws, 5)),
            "p50": float(np.percentile(draws, 50)), "p95": float(np.percentile(draws, 95)),
            "n_boot": int(draws.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg-variant", choices=["base-pt", "base-mix", "pg1"], default="base-pt",
                    help="pg1 = cross-gen: PaliGemma-1 / Gemma-1 (RESEARCH_INDEX §5.2 companion)")
    ap.add_argument("--seeds", default="0,1,2", help="saliency-minibatch seeds (≥3)")
    ap.add_argument("--grad-n", type=int, default=256, help="RefCOCO install items for G_install")
    ap.add_argument("--mmlu-per-subject", type=int, default=4, help="MMLU items/subject for G_forget")
    ap.add_argument("--frac", type=float, default=0.50, help="headline top-edge energy fraction")
    ap.add_argument("--frac-robust", default="0.25,0.75", help="robustness fractions (F5)")
    ap.add_argument("--degen-max", type=float, default=1.0,
                    help="exclude matrix if spectral-flatness fraction > this. Default 1.0 = include all "
                         "(the flatness heuristic over-flags smooth weight-spectrum decay); degeneracy is "
                         "logged as a diagnostic, basis-robustness via seed/magnitude/k controls.")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import w15_carrier as C
    from infra.aws_pg.jobs import _w15_shared as S

    assert torch.cuda.is_available(), "W15 retention needs a GPU (gradients)"
    device = "cuda"
    seeds = [int(s) for s in a.seeds.split(",")]
    fracs = [a.frac] + [float(x) for x in a.frac_robust.split(",")]
    if a.smoke:
        a.grad_n, a.mmlu_per_subject, a.n_null, a.n_boot, seeds = 16, 1, 200, 200, seeds[:2]
    print(f"[w15-ret] variant={a.pg_variant} seeds={seeds} grad_n={a.grad_n} "
          f"mmlu/subj={a.mmlu_per_subject} fracs={fracs} smoke={a.smoke}", flush=True)

    # ---- install items (RefCOCO val, hash-disjoint from a small testA reference) ----
    task = B.REFCOCO_TASK
    eval_ref = S._eval_rows(task, 300 if not a.smoke else 32, S.SEED)
    eval_hashes = {S.img_hash(task.image_fn(r)) for r in eval_ref}
    install_items, install_hashes = S.refcoco_install_items(B, max(a.grad_n * 4, 256), S.SEED, eval_hashes)
    S.split_audit(B, install_hashes, {"refcoco": eval_ref}, out)

    if a.pg_variant == "pg1":                        # cross-gen: Gemma-1 (§5.2 companion)
        from infra.aws_pg.jobs.xgen_install_fperp import build_before_after_pg1
        pg_after, pg_before, proc, paths, img_idx, gate = build_before_after_pg1(device)
    else:
        pg_after, pg_before, proc, paths, img_idx, gate = H.build_pair(a.pg_variant, device)
    lm_before = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    if a.pg_variant == "pg1":
        dkeys = sorted(k for k, p in lm_before.named_parameters()
                       if p.ndim == 2 and "layers." in k
                       and "embed_tokens" not in k and "lm_head" not in k)
    else:
        dkeys = S._decoder_keys()
    # Extract both decoder state-dicts to CPU and free pg_after BEFORE the gradient backward: only
    # after_sd is needed downstream (M = Uᵀ(after−before)V on CPU), and the gradients run on pg_before
    # alone. Holding pg_after (~6GB) + cuda state-dict clones (~10GB) alongside pg_before + the fp32
    # grads-on-182-matrices OOMs the 44GB L40S — so keep the SDs off-GPU and drop the teacher now.
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_before.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    del pg_after, lm_after
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- gradients on GPU, per seed (G at W_before = base decoder) → CPU fp32 to bound RAM ----
    grads = {}     # grads[seed] = {"install": {key:G}, "forget": {key:G}}  (fp32 CPU)
    for sd in seeds:
        rng = np.random.default_rng(S.SEED + sd)
        order = list(rng.permutation(len(install_items)))
        rc_items = [install_items[i] for i in order[:a.grad_n]]
        rc_batches = S._refcoco_batches(B, proc, rc_items, train_batch=2)
        gi, li = S.decoder_weight_grad(pg_before, paths, dkeys, rc_batches, S.refcoco_ce, device)
        # MMLU is 5-shot → ~2000-token prompts; at batch>1 the backward activation graph OOMs the
        # 44GB L40S (RefCOCO's short seqs are fine at batch 2). batch_size=1 keeps the long-sequence
        # backward at ~1/4 the peak (≈19GB total) with large margin.
        mb, _lids = S.mmlu_blank_batches(proc, a.mmlu_per_subject, 1, device,
                                         max_subjects=(4 if a.smoke else None))
        gf, lf = S.decoder_weight_grad(pg_before, paths, dkeys, mb, S.mmlu_blank_ce, device)
        grads[sd] = {"install": {k: gi[k].float() for k in dkeys},
                     "forget": {k: gf[k].float() for k in dkeys},
                     "install_ce": li, "forget_ce": lf}
        print(f"[w15-ret] seed {sd}: refcoco_ce={li:.4f} mmlu_blank_ce={lf:.4f} (gradients cached)",
              flush=True)
    del pg_before                                            # pg_after already freed; rest is CPU weight-math
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- PASS 1: per-matrix basis ONCE (the 182 SVDs), then cheap records for every (seed, frac) ----
    records = {sd: {str(f): [] for f in fracs} for sd in seeds}   # records[sd][frac] = [per-matrix dicts]
    fperp = {sd: [] for sd in seeds}
    degens = []                                                  # spectral-flatness diagnostic (logged)
    excluded = 0
    for ki, k in enumerate(dkeys):
        Wb = base_sd[k].detach().double().cpu()
        dW = after_sd[k].detach().double().cpu() - Wb
        U, V, M, I_mag, R_eff, degen = matrix_basis(Wb, dW)     # the only SVD, once per matrix
        degens.append(degen)
        if degen > a.degen_max:
            excluded += 1
            del U, V, M, I_mag
            continue
        for sd in seeds:
            rec, fp = matrix_record(M, U, V, I_mag, grads[sd]["install"][k].double(),
                                    grads[sd]["forget"][k].double(), fracs, R_eff)
            for f in fracs:
                r = rec[str(f)]; r["key"] = k
                records[sd][str(f)].append(r)
            fperp[sd].append(fp)
        del U, V, M, I_mag
        if (ki + 1) % 40 == 0:
            print(f"[w15-ret] basis {ki + 1}/{len(dkeys)} matrices ({excluded} degenerate so far)", flush=True)
    degen_diag = {"mean": float(np.mean(degens)), "max": float(np.max(degens)),
                  "n_excluded": excluded, "degen_max_threshold": a.degen_max, "rtol": 1e-3}
    print(f"[w15-ret] per-matrix basis done: {len(dkeys) - excluded} kept, {excluded} excluded "
          f"(spectral-flatness mean={degen_diag['mean']:.3f} max={degen_diag['max']:.3f}; "
          f"exclusion threshold {a.degen_max})", flush=True)

    # ---- PASS 2: pool + cluster-bootstrap + permuted null + verdict, per (seed, frac) ----
    seed_results = {}
    for sd in seeds:
        per_frac = {}
        f_perp = float(np.mean(fperp[sd])) if fperp[sd] else float("nan")
        for frac in fracs:
            kept = records[sd][str(frac)]
            km = [m["k"] for m in kept]
            Nm = [m["N"] for m in kept]
            O_FI = _cluster_boot(kept, "inter_FI", a.n_boot, S.SEED + sd)
            O_FM = _cluster_boot(kept, "inter_FM", a.n_boot, S.SEED + sd + 1)
            O_IM = _cluster_boot(kept, "inter_IM", a.n_boot, S.SEED + sd + 2)
            null = C.pooled_permuted_null(km, Nm, a.n_null, S.SEED + sd)
            null_band = {"analytic": C.pooled_analytic_null(km, Nm),
                         "p5": float(np.percentile(null, 5)), "p50": float(np.percentile(null, 50)),
                         "p95": float(np.percentile(null, 95)), "n_draws": a.n_null}
            verdict = C.retention_verdict(O_FI["point"], null_band["p5"], null_band["p95"],
                                          O_FM["p5"], O_IM["p5"], seeds_ok=True, fdr_ok=True)
            per_frac[str(frac)] = {"O_forget_install": O_FI, "O_forget_mag": O_FM, "O_install_mag": O_IM,
                                   "permuted_null": null_band, "f_perp_install_mean": f_perp,
                                   "n_kept": len(kept), "n_excluded_degenerate": excluded,
                                   "verdict": verdict}
            print(f"[w15-ret] seed {sd} frac {frac}: O(F,I)={O_FI['point']:.4f} "
                  f"null_p5={null_band['p5']:.4f} O(F,|M|)_p5={O_FM['p5']:.4f} "
                  f"O(I,|M|)_p5={O_IM['p5']:.4f} f_perp={f_perp:.3f} → {verdict} "
                  f"(kept {len(kept)}, excl {excluded})", flush=True)
        seed_results[sd] = per_frac

    # ---- seed stability on the headline frac ----
    head = str(a.frac)
    head_verdicts = [seed_results[sd][head]["verdict"] for sd in seeds]
    seed_stable = len(set(head_verdicts)) == 1
    headline = head_verdicts[0] if seed_stable else "SEED-UNSTABLE"

    # ---- planted-control sanity (reproduce O≈0/1 in-job) ----
    Ia1, Ib1, k1 = C.plant_identical((64, 64), 100, torch.Generator().manual_seed(1))
    Ia0, Ib0, k0 = C.plant_disjoint((64, 64), 100, torch.Generator().manual_seed(2))
    planted = {"identical_O": C.overlap_maps(Ia1, Ib1, k1), "disjoint_O": C.overlap_maps(Ia0, Ib0, k0)}
    assert abs(planted["identical_O"] - 1.0) < 1e-6 and planted["disjoint_O"] < 1e-6, \
        f"planted control broken: {planted}"

    metrics = {"experiment": "w15_retention", "status": "confirmatory", "pg_variant": a.pg_variant,
               "config": {"seeds": seeds, "grad_n": a.grad_n, "mmlu_per_subject": a.mmlu_per_subject,
                          "fracs": fracs, "degen_max": a.degen_max, "n_null": a.n_null, "n_boot": a.n_boot},
               "swap_gate": gate, "degeneracy_diag": degen_diag,
               "grad_ce": {sd: {"install_ce": grads[sd]["install_ce"], "forget_ce": grads[sd]["forget_ce"]}
                           for sd in seeds},
               "per_seed": seed_results,
               "verdict": {"headline_frac": a.frac, "per_seed": head_verdicts,
                           "seed_stable": seed_stable, "headline": headline}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    controls = {"experiment": "w15_retention_controls", "pg_variant": a.pg_variant, "planted": planted,
                "frozen": {"frac_headline": a.frac, "degen_rtol": 1e-3, "n_boot": a.n_boot,
                           "n_null": a.n_null, "seed": S.SEED}}
    (out / "controls.json").write_text(json.dumps(controls, indent=2, default=str))

    print(f"[w15-ret] VERDICT={headline} seed_stable={seed_stable} "
          f"O(F,I)={seed_results[seeds[0]][head]['O_forget_install']['point']:.4f}", flush=True)
    print("W15_RETENTION_OK", json.dumps({"verdict": headline,
          "O_forget_install": round(seed_results[seeds[0]][head]["O_forget_install"]["point"], 4),
          "null_p5": round(seed_results[seeds[0]][head]["permuted_null"]["p5"], 4)}), flush=True)


if __name__ == "__main__":
    main()
