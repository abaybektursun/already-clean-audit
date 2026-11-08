"""XGEN V2 — out-of-span carrier (weight-only, CPU). Prereg §9.4.

Does ΔW₁ = W(PaliGemma-1 LM) − W(gemma-2b) sit OUT-OF-SPAN of the base weight's dominant
subspace — the structural signature (W15 label "out-of-span") that made cheap in-span reuse
fail on PG2? Read on the RECTANGULAR MLP matrices only (square attention has f_perp ≡ 0).

For every 18×7 decoder matrix:
  * f_perp = gain_routing.decompose(w_base, w_pg1).f_perp           (out-of-span fraction)
  * the dimension-counting matched-random null per shape (null_band, 3 seeds)
  * the row-shuffle control (≥30 draws): a structured ΔW₁ stays OFF the null; shuffling
    rows collapses f_perp toward it (swap_back.row_shuffled_after).
Writes results/xgen/carrier.json. Verdict (§6) is applied at unblind; this job is the
weight-only DESCRIPTIVE corroboration. fp64 throughout. No GPU. Fail-loud; no try/except.

    pg run infra.aws_pg.jobs.xgen_carrier --type r7i.4xlarge --no-weights \
        --s3-prefix results/xgen_carrier --budget-min 60
"""
from __future__ import annotations

import argparse
import json
import pathlib

NULL_DRAWS = 2000
NULL_SEEDS = (1337, 2024, 7)            # 3 null draws (prereg §6: 3/3 weight-decomposition seeds)
SHUFFLE_DRAWS = 50
TOP_K = 8                               # only used for C_route reporting; f_perp is top_k-agnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-revision", default="main")
    ap.add_argument("--shuffle-seed", type=int, default=1337)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                   # FIRST — gated metadata 401s otherwise

    import numpy as np
    import torch

    from infra.pg import gain_routing as GR
    from infra.pg import swap_back as SB
    from infra.pg.gain_routing import fine_type, matched_decoder_weight_matrices, null_band

    # --- load base Gemma-1 + PaliGemma-1 LM decoder (CPU fp32; reuses e0a extractors) ---
    base, base_dec, pg1_dec, meta = SB.load_xgen_decoders(device="cpu", base_revision=args.base_revision)
    matched = matched_decoder_weight_matrices(pg1_dec, base_dec)    # {key: (w_base, w_pg1)}
    assert len(matched) == 18 * 7, f"expected 18×7=126 matched matrices, got {len(matched)}"
    print(f"[XGEN-V2] {meta['base_repo']} vs {meta['pg1_repo']}: {len(matched)} matched matrices")

    # --- recon-identity gate ONCE (fp64), then check_recon=False in the per-matrix loop ---
    k0 = sorted(matched)[0]
    recon0 = GR.decompose(*matched[k0], check_recon=True)["recon_residual"]
    assert recon0 < 1e-10, f"fp64 recon identity broken on {k0}: residual={recon0}"
    print(f"[XGEN-V2] recon gate OK on {k0}: residual={recon0:.2e}")

    # --- per-matrix f_perp + null + shuffle ---
    null_cache: dict = {}
    per_matrix = []
    for k in sorted(matched):
        wb, wa = matched[k]
        d = GR.decompose(wb, wa, top_k=TOP_K, check_recon=False)
        shape = tuple(d["shape"])
        ftype = fine_type(k)
        is_mlp = ftype in ("gate_proj", "up_proj", "down_proj")

        # null + shuffle are computed ONLY for the MLP matrices (the §6 V2 read is MLP-only;
        # attention is excluded — square o/q have f_perp≡0, and k/v are not part of the verdict).
        if is_mlp and shape not in null_cache:
            bands = [null_band(shape, NULL_DRAWS, s, TOP_K) for s in NULL_SEEDS]
            null_cache[shape] = {
                "analytic_f_perp": bands[0]["analytic_f_perp"],
                "p95_per_seed": [b["f_perp"]["p95.0"] for b in bands],
                "p95_mean": float(np.mean([b["f_perp"]["p95.0"] for b in bands])),
                "rank_r": bands[0]["rank_r"],
            }
        null = null_cache.get(shape) if is_mlp else None

        # row-shuffle control (MLP only). Precompute SVD(w_before) ONCE and reuse across all draws:
        # decompose(wb, ·) re-SVDs wb every draw → 2700 redundant big fp64 SVDs (hours). f_perp here
        # is identical to decompose's: out_of_span/total with in_span = ‖Uᵀ ΔW_perm V‖² (frozen math).
        shuf = None
        if is_mlp:
            Uw, _sw, _Vhw, Vw, _rw = GR._svd_fp64(wb)
            dW = wa.detach().to(torch.float64) - wb.detach().to(torch.float64)
            g = torch.Generator().manual_seed(args.shuffle_seed)
            sh = []
            for _ in range(SHUFFLE_DRAWS):
                dWp = dW[torch.randperm(dW.shape[0], generator=g)]
                M = Uw.transpose(0, 1) @ dWp @ Vw
                total = float((dWp ** 2).sum())
                in_span = float((M ** 2).sum())
                sh.append((total - in_span) / total if total > 0 else float("nan"))
            shuf = {"mean": float(np.mean(sh)), "std": float(np.std(sh)), "n": len(sh)}

        per_matrix.append({
            "key": k, "type": ftype, "shape": list(shape), "is_mlp": is_mlp,
            "is_square_full_rank": d["is_square_full_rank"],
            "f_perp": d["f_perp"], "E_gain": d["E_gain"],
            "gain_frac": d["gain_frac"], "route_frac": d["route_frac"],
            "in_span_frac": d["in_span_frac"], "C_route": d.get("C_route"),
            "n_degenerate_pairs": d["n_degenerate_pairs"], "degen_route_frac": d["degen_route_frac"],
            "null": null, "shuffle": shuf,
        })

    # --- MLP-only summary: median f_perp + cluster(=per-shape) bootstrap CI + enrichment ---
    mlp = [r for r in per_matrix if r["is_mlp"]]
    mlp_fp = np.array([r["f_perp"] for r in mlp], dtype=float)
    rng = np.random.default_rng(1337)
    boots = np.array([np.median(mlp_fp[rng.integers(0, mlp_fp.size, mlp_fp.size)]) for _ in range(2000)])
    median_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    # §6 V2 rule is "median MLP f_perp lower-CI > null p95 for 3/3 seeds" — so test the lower-CI
    # against the per-seed median null p95, all three seeds (NOT the mean-of-seeds).
    mlp_null_p95_per_seed = [float(np.median([r["null"]["p95_per_seed"][s] for r in mlp]))
                             for s in range(len(NULL_SEEDS))]
    mlp_null_p95 = float(np.median([r["null"]["p95_mean"] for r in mlp]))   # reporting only
    mlp_shuffle_mean = float(np.mean([r["shuffle"]["mean"] for r in mlp]))
    enrichment = float(np.median([r["f_perp"] / r["null"]["analytic_f_perp"] for r in mlp
                                  if r["null"]["analytic_f_perp"] > 0]))

    result = {
        "torch": torch.__version__, "meta": meta,
        "n_matched": len(matched), "n_mlp": len(mlp), "recon_gate_residual": recon0,
        "null_draws": NULL_DRAWS, "null_seeds": list(NULL_SEEDS), "shuffle_draws": SHUFFLE_DRAWS,
        "mlp_summary": {
            "median_f_perp": float(np.median(mlp_fp)),
            "median_f_perp_ci95": median_ci,
            "null_p95_median": mlp_null_p95,
            "null_p95_median_per_seed": mlp_null_p95_per_seed,
            "shuffle_f_perp_mean": mlp_shuffle_mean,
            "enrichment_median": enrichment,
            # DESCRIPTIVE prefiguring of the §6 verdict (binding verdict APPLIED at unblind under
            # BH-FDR): the 3/3-seeds form — median MLP f_perp lower-CI clears EVERY seed's null p95.
            "descriptive_out_of_span": bool(all(median_ci[0] > p for p in mlp_null_p95_per_seed)),
        },
        "per_matrix": per_matrix,
        "firewall": "labels-only vs PG2 (W15 'out-of-span'); PG1 threshold on PG1 null",
        "label_discipline": "DESCRIPTIVE corroboration (exploratory) — no causal weight alone",
    }
    (out / "carrier.json").write_text(json.dumps(result, indent=2, default=str))
    s = result["mlp_summary"]
    print(f"[XGEN-V2] MLP median f_perp={s['median_f_perp']:.4f} CI={s['median_f_perp_ci95']} "
          f"null_p95={s['null_p95_median']:.4f} shuffle={s['shuffle_f_perp_mean']:.4f} "
          f"enrich={s['enrichment_median']:.3f} → descriptive_out_of_span={s['descriptive_out_of_span']}")


if __name__ == "__main__":
    main()
