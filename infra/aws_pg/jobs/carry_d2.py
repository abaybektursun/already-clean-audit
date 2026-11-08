"""CARRY Arm D2 (CPU) — intruder-dimension census (S6) across the 24 FTs × 7 matrix types ×
18 layers, on the RESIDUAL-FACING side (comwrite_core.RESIDUAL_SIDE): a 2048² eigendecomp of
each matrix's residual-side Gram (never a full SVD of a 16384×2048 matrix). Outer loop over
the 126 (L,t) pairs, inner over 24 FTs; each ΔW deleted after use (disk).

An intruder at threshold ε = a post-FT top singular vector whose max |cos| to EVERY base
singular vector is < ε. R-D2 adjudicates the median intruder count per FT across the grid.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_d2 --type r7i.8xlarge,r7i.4xlarge \
      --no-weights --s3-prefix results/carry_d2 --budget-min 360
"""
import argparse
import json
import os
import pathlib

NULL_PAIRS = [(0, "q_proj"), (9, "q_proj"), (17, "q_proj"), (0, "down_proj"),
              (9, "down_proj"), (17, "down_proj"), (4, "gate_proj"), (13, "v_proj")]


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import comwrite_core as CW
    from infra.pg.gauge_common import N_LAYERS, PROJ

    freeze = CS.assert_frozen("d2")
    CS.install_pg()
    rep = {"object": "carry_d2_v1", "freeze": freeze,
           "gpu_seconds": {"total": 0.0, "note": "CPU job"}}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/carry_d2")
    (wd / "scratch").mkdir(parents=True, exist_ok=True)
    anat = CS.load_roster(str(wd))
    n_ft = len(anat["roster"])
    rep["eps_grid"] = list(CC.RD2_EPS_GRID)

    def base_eigvecs(L, t):
        f = CS.fetch(f"{CS.COMWRITE_ANATOMY}/scratch/base_pg1__{t}__L{L}.npy",
                     wd / "scratch" / f"base_{t}_L{L}.npy")
        W = np.load(f).astype(np.float64)
        G = (W.T @ W) if CW.residual_side(t) == "input" else (W @ W.T)
        _wv, V = np.linalg.eigh(G)
        return W, V[:, ::-1]                      # (2048, 2048) desc

    def ft_top10(W_base, L, t, ridx):
        f = CS.fetch(f"{CS.COMWRITE_ANATOMY}/scratch/{t}__L{L}__{ridx}.npy",
                     wd / "scratch" / f"dW_{t}_L{L}_{ridx}.npy")
        dW = np.load(f).astype(np.float64)
        os.remove(f)
        W = W_base + dW
        G = (W.T @ W) if CW.residual_side(t) == "input" else (W @ W.T)
        return CW.u_wr(G, CC.RD2_TOPK)           # (2048, 10)

    with tm.stage("census"):
        # per-FT accumulated counts across all 126 pairs; per-(ft,L) breakdown for F-D1
        per_ft = [{f"{e:.1f}": 0 for e in CC.RD2_EPS_GRID} for _ in range(n_ft)]
        breakdown = {}                            # f"{ridx}/L{L}/{t}" -> counts
        null_rows = {}
        pair_i = 0
        for L in range(N_LAYERS):
            for t in PROJ:
                W_base, U_base = base_eigvecs(L, t)
                for ridx in range(n_ft):
                    U_ft = ft_top10(W_base, L, t, ridx)
                    counts, maxcos = CC.intruder_counts(U_ft, U_base)
                    for e in CC.RD2_EPS_GRID:
                        per_ft[ridx][f"{e:.1f}"] += counts[f"{e:.1f}"]
                    breakdown[f"{ridx}/L{L}/{t}"] = {
                        "counts": counts, "min_maxcos": float(min(maxcos)),
                        "median_maxcos": float(np.median(maxcos))}
                    if ridx == 0 and (L, t) in NULL_PAIRS:
                        B_sh = CC.entry_shuffle_null(U_base, CC.RD2_TOPK,
                                                     seed=100 + NULL_PAIRS.index((L, t)))
                        ncounts, _ = CC.intruder_counts(U_ft, B_sh)
                        null_rows[f"L{L}/{t}"] = ncounts
                del U_base, W_base
                pair_i += 1
                if pair_i % 7 == 0:
                    print(f"[carry-d2] {pair_i}/126 pairs done", flush=True)
                    rep["progress_pairs"] = pair_i
                    _w()
        rep["per_ft_counts"] = per_ft
        rep["breakdown"] = breakdown
        rep["null_rows"] = null_rows
    _w()

    rep["rd2"] = CC.adjudicate_rd2(per_ft)
    if rep["rd2"]["verdict"] == "H_intruders-present":
        # name the max-intruder (ft, L, t) at the tightest ε for the successor cell
        worst = max(breakdown, key=lambda k: breakdown[k]["counts"]["0.9"])
        rep["next"] = {"max_intruder_cell": worst,
                       "counts": breakdown[worst]["counts"],
                       "note": "optional λ-scaling GPU cell — a successor, not run here"}
    CS.assert_gpu_seconds(rep)
    print("CARRY_D2_OK", json.dumps({"verdict": rep["rd2"]["verdict"],
                                     "median_counts": rep["rd2"]["median_counts"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
