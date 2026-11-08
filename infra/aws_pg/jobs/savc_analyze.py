"""SAV-C — pooled WITHIN-FAMILY analysis of the SAV-B relearnability gauge (control-plane, no GPU).

Does L9 representation-overlap predict relearnability S(16k) WITHIN a fine-tune family (overlap>cut), or
only BETWEEN the 3 coarse regimes SAV-B's critic scoped it to? Reads the 13 FROZEN SAV-B baseline rows
(savb_metrics.json) + the new units (unit_*.json from savb_unit) + the n5c cloud (cached a64/rhoL/frob),
pools them, runs the FROZEN §6 analysis (full-pool replication + within-family ΔLOO-R²/Spearman vs the
permutation null + weight-space baselines), and applies the §6 verdict. Reuses infra/pg/savb.py verbatim.
FAIL LOUD: no try/except, no synthetic fallback.

Spec: studies/lineage/paligemma1-pi0/research/2026-06-24_savc_prereg.md (§6/§13).

    python -m infra.aws_pg.jobs.savc_analyze --out results/savc \
        --savb-metrics results/savb/savb_metrics.json --units-dir /tmp/savc_units \
        --n5c-cloud studies/lineage/paligemma1-pi0/research/2026-06-02_n5c_cloud.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

WITHIN_CUT = 0.62   # frozen on SAV-B data: excludes pi0 (0.505), pi05 (0.488), refcoco-seg (0.615)
GAP_HI = 0.78       # min new-unit overlap < this => the 0.62-0.82 gap is filled (not saturated)
SCICAP = "paligemma-3b-ft-scicap-224"   # the extreme-frob leave-out (sensitivity, §7)


def _new_unit_row(r, cloud_by_task):
    """Extract one new unit's row in the SAV-B savb_metrics schema (mirrors savb_analyze.py:48-59)."""
    task = r["unit"]
    cm = cloud_by_task[task]                          # KeyError = not in the n5c cloud — fail loud
    row = {"unit": task, "kind": "n5c_new",
           "S_16k": r["outcomes"]["16000"]["S_recover"],
           "S_1k": r["outcomes"]["1000"]["S_recover"],
           **r["predictors"],                         # l2_base, lmc_barrier, overlap
           "a64": cm["A"]["64"]["a_pooled_full"],
           "rhoL": cm["rhoL_enrichment"]["64"],
           "frob_cached": cm["frob_dW_total"]}
    for k in ("overlap", "S_16k", "S_1k", "l2_base", "lmc_barrier"):
        assert k in row and row[k] is not None, f"new unit {task} missing {k}"
    assert 0.0 <= row["overlap"] <= 1.0, f"{task} overlap out of [0,1]: {row['overlap']}"
    return row


def _analyze(pool, SBV, np, n_boot, n_perm, cut=WITHIN_CUT):
    """Full-pool + within-family (overlap>cut) Spearman + ΔLOO-R²(overlap | l2,lmc), on S16 (primary) and
    S1 (secondary). Returns the analysis dict."""
    def block(rows, label):
        y16 = np.array([u["S_16k"] for u in rows], float)
        y1 = np.array([u["S_1k"] for u in rows], float)
        ov = np.array([u["overlap"] for u in rows], float)
        Xb = np.column_stack([[u["l2_base"] for u in rows], [u["lmc_barrier"] for u in rows]])
        res = {"n": len(rows), "units": [u["unit"] for u in rows],
               "overlap_min": float(ov.min()), "overlap_max": float(ov.max())}
        for yname, y in (("S_16k", y16), ("S_1k", y1)):
            res[yname] = {
                "spearman_overlap": SBV.boot_spearman(ov, y, n_boot=n_boot, seed=1337),
                "loo_overlap": SBV.delta_loo_perm(Xb, ov, y, n_perm=n_perm, seed=1337)}
        return res
    out = {"full_pool": block(pool, "full")}
    wf = [u for u in pool if u["overlap"] > cut]
    out["within_family"] = block(wf, "within_F")
    wf_no_scicap = [u for u in wf if u["unit"] != SCICAP]
    if len(wf_no_scicap) >= 4 and len(wf_no_scicap) < len(wf):
        out["within_family_no_scicap"] = block(wf_no_scicap, "within_F_no_scicap")
    out["within_cut"] = cut
    return out


def _verdict(analysis, new_rows, n_new):
    """§6 frozen naming rule."""
    if n_new < 3:
        return {"named": "INSTRUMENT_BROKEN", "basis": f"only {n_new} new units completed (<3)"}
    fp = analysis["full_pool"]["S_16k"]["loo_overlap"]
    full_replicates = bool(fp["delta"] > 0 and fp["p_perm"] < 0.10)
    if not full_replicates:
        return {"named": "H_no-gauge", "full_replicates": False,
                "basis": (f"the full-pool (13+{n_new}) replication FAILS — overlap adds no LOO power over "
                          f"l2+lmc on S16 (ΔLOO-R²={fp['delta']:+.3f}, perm p={fp['p_perm']:.4f}); SAV-B's "
                          f"between-regime result does not survive the larger n")}
    new_ov = [r["overlap"] for r in new_rows]
    min_new = float(min(new_ov))
    gap_filled = bool(min_new < GAP_HI)
    wf = analysis["within_family"]["S_16k"]
    wf_loo, wf_rho = wf["loo_overlap"], wf["spearman_overlap"]
    wf_resolves = bool(wf_rho["ci95"][0] > 0 and wf_loo["delta"] > 0 and wf_loo["p_perm"] < 0.10)
    detail = {"full_replicates": True, "fp_delta": fp["delta"], "fp_p": fp["p_perm"],
              "min_new_overlap": min_new, "gap_filled": gap_filled,
              "wf_n": analysis["within_family"]["n"], "wf_rho": wf_rho["rho"],
              "wf_rho_ci": wf_rho["ci95"], "wf_delta": wf_loo["delta"], "wf_p": wf_loo["p_perm"]}
    if not gap_filled:
        return {"named": "H_saturated", **detail,
                "basis": (f"the new units do NOT fill the 0.62-0.82 gap (min new overlap {min_new:.3f} ≥ "
                          f"{GAP_HI}) — near-task FT is representationally saturated; within-family "
                          f"resolution is untestable on this family. The gauge stays between-regime-only "
                          f"by default (full-pool replicates: ΔLOO-R²={fp['delta']:+.3f}, p={fp['p_perm']:.4f}).")}
    if wf_resolves:
        return {"named": "H_within-resolution", **detail,
                "basis": (f"overlap predicts S16 WITHIN the family (overlap>{WITHIN_CUT}, n={detail['wf_n']}): "
                          f"ρ={wf_rho['rho']:+.3f} CI[{wf_rho['ci95'][0]:+.3f},{wf_rho['ci95'][1]:+.3f}], "
                          f"ΔLOO-R²={wf_loo['delta']:+.3f}, perm p={wf_loo['p_perm']:.4f} — the gauge is a "
                          f"usable within-family reuse-triage screen (predictive ∧ novel ∧ practical)")}
    return {"named": "H_between-regime-only", **detail,
            "basis": (f"the new units fill the gap (min overlap {min_new:.3f}) BUT within-family overlap does "
                      f"NOT predict S16 (ρ={wf_rho['rho']:+.3f} CI[{wf_rho['ci95'][0]:+.3f},"
                      f"{wf_rho['ci95'][1]:+.3f}], ΔLOO-R²={wf_loo['delta']:+.3f}, p={wf_loo['p_perm']:.4f}) — "
                      f"the gauge is a coarse between-regime separator, not a within-family triage tool; "
                      f"SAV-B's scoped reading confirmed, the practical-triage reading falsified")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--savb-metrics", required=True, help="the 13 frozen SAV-B baseline rows")
    ap.add_argument("--units-dir", required=True, help="dir of new unit_*.json from savb_unit")
    ap.add_argument("--n5c-cloud", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import savb as SBV

    baseline = json.loads(pathlib.Path(a.savb_metrics).read_text())["units"]
    assert len(baseline) == 13, f"expected 13 frozen SAV-B baseline rows, found {len(baseline)}"
    cloud = json.loads(pathlib.Path(a.n5c_cloud).read_text())
    cloud_by_task = {m["repo_id"].split("/")[-1]: m for m in cloud["members"]}

    new_rows = []
    for p in sorted(pathlib.Path(a.units_dir).glob("unit_*.json")):
        new_rows.append(_new_unit_row(json.loads(p.read_text()), cloud_by_task))
    n_new = len(new_rows)
    print(f"[savc] {len(baseline)} baseline + {n_new} new units", flush=True)
    for r in new_rows:
        print(f"[savc]   {r['unit']}: overlap={r['overlap']:.3f} S16={r['S_16k']:+.3f} "
              f"S1={r['S_1k']:+.3f} frob={r['frob_cached']:.1f}", flush=True)

    pool = baseline + new_rows
    analysis = _analyze(pool, SBV, np, a.n_boot, a.n_perm)
    v = _verdict(analysis, new_rows, n_new)

    rep = {"object": "savc_within_family_v1", "n_baseline": len(baseline), "n_new": n_new,
           "n_pool": len(pool), "within_cut": WITHIN_CUT, "gap_hi": GAP_HI,
           "new_units": new_rows, "pool": pool, "analysis": analysis, "verdict": v,
           "frozen_rule": ("H_within-resolution iff within-F ρ CI>0 ∧ ΔLOO-R²>0 ∧ perm p<0.10 (with full-pool "
                           "replicating); H_saturated iff min new overlap≥0.78; H_between-regime-only iff gap "
                           "filled but within-F null; H_no-gauge iff full-pool fails")}
    (out / "savc_metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    fp = analysis["full_pool"]["S_16k"]["loo_overlap"]
    wf = analysis["within_family"]["S_16k"]
    print("=" * 78, flush=True)
    print(f"[savc] full-pool ΔLOO-R²={fp['delta']:+.3f} p={fp['p_perm']:.4f} | "
          f"within-F(n={analysis['within_family']['n']}) ρ={wf['spearman_overlap']['rho']:+.3f} "
          f"ΔLOO-R²={wf['loo_overlap']['delta']:+.3f} p={wf['loo_overlap']['p_perm']:.4f}", flush=True)
    print(f"[savc] VERDICT: {v['named']} — {v['basis']}", flush=True)
    print(f"SAVC_OK verdict={v['named']} n_new={n_new}", flush=True)


if __name__ == "__main__":
    main()
