"""FAIRSCRATCH UNBLIND aggregator (control-plane; pure arithmetic over banked cells).

Applies the FROZEN §6 procedure mechanically to the banked `w17_cell` rows; no GPU,
no model. Two modes:

  --mode preflight  : §6.0 gate. pg@16k/r512 (3 seeds, confirmation) seed-mean S_recover
                      must reproduce the W17 banked value within W17's δ_eq@16k. HALT on
                      fail (harness drift; no scratch number is trustworthy until this passes).
  --mode full       : the unblind. Reads pg@16k, scr* (matched budget, max_steps None),
                      scr=CE (matched train-CE, max_steps>0) confirmation arms;
                      recomputes δ_eq on this study's control cells (§6.3); the m=2 primary
                      family {ΔS_A>0, ΔS_B>0} via paired_arm_diff_s + BH-FDR; the §6.4
                      verdict map; fills the §11 grid.

Arms share the cell_id shape (decoder/arm/rank/n/seed/split) so scr* and scr=CE are
told apart by `train.recipe.max_steps`. Fail-loud; recompute, never transcribe.

    python -m infra.aws_pg.jobs.fairscratch_unblind --out results/fairscratch \
        --controls <w17_controls.json> --cells-dir <dir with **/cell_*.json> \
        --w17-metrics results/w17/metrics.json --recipe <fairscratch_recipe.json> --mode full
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEEDS = (1337, 1338, 1339)


# ===========================================================================
# PURE §6.4 verdict map — the §9.0 weight-free dry-run exercises this.
# ===========================================================================
def fairscratch_verdict(dS_A_point, dS_A_sig, dS_B_point, deq,
                        recipe_admissible, preflight_ok, gap_stable=True) -> dict:
    """The §6.4 truth table, mechanical. dS_A = S_pg − S_scr* (matched budget);
    dS_B = S_pg − S_scr=CE (matched train-CE). deq = recomputed δ_eq@16k."""
    if not preflight_ok:
        return {"verdict": "HALT", "basis": "§6.0 preflight failed — harness drift"}
    if not recipe_admissible:
        return {"verdict": "INDETERMINATE-untrainable",
                "basis": "§6.1 found no admissible recipe — bounds construction cost from below"}
    if not gap_stable:
        return {"verdict": "non-result",
                "basis": "per-cell seed spread exceeds δ_eq (§6.3) — contrast voided"}
    if dS_A_point > deq and dS_A_sig:
        if dS_B_point > deq:
            return {"verdict": "H_construction",
                    "basis": "matched-budget gap survives the fair recipe AND matched-train-CE "
                             "gap persists — pretrained init is a reachability/generalization advantage"}
        return {"verdict": "H_access",
                "basis": "matched-budget gap survives but matched-train-CE closes it — "
                         "the edge is optimization speed, not generalization"}
    if dS_A_point <= deq:
        return {"verdict": "H_recipe-artifact",
                "basis": "matched-budget gap collapses under the fair recipe — W17's magnitude "
                         "was the crippled-recipe (M2) confound"}
    return {"verdict": "indeterminate",
            "basis": "ΔS_A point exceeds δ_eq but is not BH-significant — inconclusive"}


# ===========================================================================
# Cell selection + per-item assembly.
# ===========================================================================
def _cap_conf(rows, decoder, n=16000):
    return [r for r in rows if r["decoder"] == decoder and r["arm"] == "cap"
            and r["n"] == n and r["rank"] == 512 and r["eval_split"] == "confirmation"]


def _arms(rows):
    """pg / scr* (matched budget, max_steps None) / scr=CE (matched train-CE, max_steps>0)."""
    pg = _cap_conf(rows, "pg")
    rnd = _cap_conf(rows, "random")
    scr_star = [r for r in rnd if r["train"]["recipe"].get("max_steps") in (None, -1)]
    scr_ce = [r for r in rnd if (r["train"]["recipe"].get("max_steps") or -1) > 0]
    return pg, scr_star, scr_ce


def _per_item(cells, b_conf):
    arrs = []
    for r in cells:
        pi = r["per_item"]
        assert pi is not None and len(pi) == len(b_conf), \
            f"{r['cell_id']}: confirmation per_item misaligned ({len(pi) if pi else None}/{len(b_conf)})"
        arrs.append(pi)
    return arrs


def _load_cells(cells_dir):
    rows = [json.loads(p.read_text()) for p in sorted(pathlib.Path(cells_dir).glob("**/cell_*.json"))]
    assert rows, f"no cell_*.json under {cells_dir}"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--w17-metrics", required=True, help="results/w17/metrics.json @ 0e71d06 (reproduction target)")
    ap.add_argument("--recipe", default=None, help="fairscratch_recipe.json (FROZEN flag for verdict)")
    ap.add_argument("--mode", choices=["preflight", "full"], default="full")
    ap.add_argument("--n-boot", type=int, default=2001)
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import w16_probes as W
    from infra.pg import w17_verdict as V

    C = json.loads(pathlib.Path(args.controls).read_text())
    assert C["CONTROLS_OK"] is True
    hashes = C["item_hashes"]
    b_conf = W.mask_by(C["BEFORE"]["per_item"], hashes, W.is_confirmation)
    f_conf = W.mask_by(C["FLOOR"]["per_item"], hashes, W.is_confirmation)

    W17M = json.loads(pathlib.Path(args.w17_metrics).read_text())
    w17_target = float(W17M["cell_S"]["cap_16k"]["S_seed_mean"])   # banked W17 pg seed-mean S
    w17_deq16k = float(W17M["delta_eq"]["16k"])                    # W17 reproduction tolerance

    rows = _load_cells(args.cells_dir)
    pg, scr_star, scr_ce = _arms(rows)
    assert pg, "no pg@16k confirmation cells — cannot run preflight or unblind"

    pg_arrs = _per_item(pg, b_conf)
    pg_seed_S = V.per_seed_s(pg_arrs, b_conf, f_conf)
    pg_S_mean = float(np.mean(pg_seed_S))
    preflight_ok = bool(abs(pg_S_mean - w17_target) <= w17_deq16k)

    pf = {"pg_S_seed_mean": pg_S_mean, "pg_per_seed_S": pg_seed_S, "n_pg_cells": len(pg),
          "w17_target": w17_target, "w17_deq16k": w17_deq16k,
          "abs_diff": abs(pg_S_mean - w17_target), "preflight_ok": preflight_ok}
    print(f"[FS-UNB] preflight: pg S={pg_S_mean:.4f} vs W17 {w17_target:.4f} "
          f"(|Δ|={pf['abs_diff']:.4f} ≤ δ_eq {w17_deq16k:.4f}? {preflight_ok})")

    if args.mode == "preflight":
        (out / "preflight.json").write_text(json.dumps(pf, indent=1, default=str))
        if not preflight_ok:
            raise RuntimeError(f"§6.0 PREFLIGHT FAILED: pg S={pg_S_mean:.4f} not within "
                               f"δ_eq {w17_deq16k:.4f} of W17 {w17_target:.4f} — HALT (harness drift)")
        print("[FS-UNB] PREFLIGHT PASS")
        return

    # ---- full unblind ----
    assert scr_star, "no scr* (matched-budget random) confirmation cells"
    scr_star_arrs = _per_item(scr_star, b_conf)
    scr_star_seed_S = V.per_seed_s(scr_star_arrs, b_conf, f_conf)

    # δ_eq recomputed on THIS study's control cells (§6.3): pg + scr* (each ≥2 runs)
    deq = V.delta_eq([pg_arrs, scr_star_arrs], b_conf, f_conf, n_boot=args.n_boot, pct=95.0)
    deq16k = float(deq["delta_eq"])

    # Contrast A — matched budget: ΔS_A = S_pg − S_scr*
    dS_A = V.paired_arm_diff_s(pg_arrs, scr_star_arrs, b_conf, f_conf, n_boot=args.n_boot, seed=1337)

    # Contrast B — matched train-CE: ΔS_B = S_pg − S_scr=CE (if the arm ran)
    dS_B = None
    scr_ce_seed_S = None
    if scr_ce:
        scr_ce_arrs = _per_item(scr_ce, b_conf)
        scr_ce_seed_S = V.per_seed_s(scr_ce_arrs, b_conf, f_conf)
        dS_B = V.paired_arm_diff_s(pg_arrs, scr_ce_arrs, b_conf, f_conf, n_boot=args.n_boot, seed=1337)

    # m=2 primary family BH-FDR
    pvals = {"dS_A_matched_budget": dS_A["p_one_sided"]}
    if dS_B is not None:
        pvals["dS_B_matched_ce"] = dS_B["p_one_sided"]
    bh = W.bh_fdr(pvals, q=args.q)
    dS_A_sig = bool(bh["dS_A_matched_budget"]["significant"])

    # seed-stability (§6.3): each decision cell's spread must sit inside δ_eq
    def spread(s):
        return float(max(s) - min(s))
    spreads = {"pg": spread(pg_seed_S), "scr_star": spread(scr_star_seed_S)}
    if scr_ce_seed_S is not None:
        spreads["scr_ce"] = spread(scr_ce_seed_S)
    gap_stable = bool(spreads["pg"] <= deq16k and spreads["scr_star"] <= deq16k)

    recipe_admissible = True
    recipe_status = "assumed (no recipe file)"
    if args.recipe:
        rj = json.loads(pathlib.Path(args.recipe).read_text())
        recipe_admissible = bool(rj.get("FROZEN") and rj.get("chosen") is not None)
        recipe_status = rj.get("status")

    verdict = fairscratch_verdict(
        dS_A_point=dS_A["point"], dS_A_sig=dS_A_sig,
        dS_B_point=(dS_B["point"] if dS_B else None),
        deq=deq16k, recipe_admissible=recipe_admissible,
        preflight_ok=preflight_ok, gap_stable=gap_stable)

    grid = {
        "preflight_pg_reproduces_w17": preflight_ok,
        "recipe_admissible": recipe_admissible,
        "dS_A_matched_budget": {"point": dS_A["point"], "ci95": dS_A["ci95"],
                                "gt_deq_and_bh_sig": bool(dS_A["point"] > deq16k and dS_A_sig)},
        "dS_B_matched_ce": ({"point": dS_B["point"], "ci95": dS_B["ci95"],
                             "gt_deq": bool(dS_B["point"] > deq16k)} if dS_B else None),
        "scr_star_S_seed_mean": float(np.mean(scr_star_seed_S)),
        "scr_ce_S_seed_mean": (float(np.mean(scr_ce_seed_S)) if scr_ce_seed_S else None),
        "pg_S_seed_mean": pg_S_mean,
        "overall_verdict": verdict["verdict"],
    }

    result = {
        "controls": args.controls, "cells_dir": args.cells_dir, "n_cells": len(rows),
        "mode": "full", "preflight": pf,
        "delta_eq_16k_recomputed": deq16k, "delta_eq_detail": deq,
        "n_arms": {"pg": len(pg), "scr_star": len(scr_star), "scr_ce": len(scr_ce)},
        "per_seed_S": {"pg": pg_seed_S, "scr_star": scr_star_seed_S, "scr_ce": scr_ce_seed_S},
        "spreads": spreads, "gap_stable": gap_stable,
        "contrast_A_matched_budget": dS_A,
        "contrast_B_matched_ce": dS_B,
        "bh_fdr": bh, "recipe_status": recipe_status,
        "VERDICT": verdict, "prediction_grid": grid,
        "w17_metrics_ref": args.w17_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[FS-UNB] δ_eq@16k={deq16k:.4f}  ΔS_A={dS_A['point']:.4f} "
          f"CI[{dS_A['ci95'][0]:.4f},{dS_A['ci95'][1]:.4f}] sig={dS_A_sig}")
    if dS_B:
        print(f"[FS-UNB] ΔS_B={dS_B['point']:.4f} CI[{dS_B['ci95'][0]:.4f},{dS_B['ci95'][1]:.4f}]")
    print(f"[FS-UNB] VERDICT: {verdict['verdict']} — {verdict['basis']}")


if __name__ == "__main__":
    main()
