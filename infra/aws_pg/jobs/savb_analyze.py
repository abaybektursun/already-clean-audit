"""SAV-B aggregate + FROZEN analysis (control-plane; ONE outcome–predictor join).

Inputs: 10 N5C unit rows + chain_predictors.json + SAV-A frozen confirmation
cells (chain outcomes, seed-mean) + cached n5c_cloud.json (a64/rhoL descriptive
family; chain: pi0 from pi0_member, PG1/pi05 median-imputed — disclosed).

Frozen plan (prereg 2026-06-12 §2): primary outcome S(16k), secondary S(1k);
Spearman + bootstrap-over-units CI per predictor; LOO-R² {L2, LMC} vs +overlap;
SUPPORTED iff ΔLOO-R² > 0 AND permutation p < 0.10. DISCOVERY-grade.

    python -m infra.aws_pg.jobs.savb_analyze --out results/savb \
        --units-dir /tmp/savb_units --chain-predictors /tmp/savb_units/chain_predictors.json \
        --sav-cells-dir /tmp/sav_cells --n5c-cloud /tmp/n5c_cloud.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

CHAIN = ("PG1", "pi0", "pi05")
NS = (1000, 16000)
SEEDS = (1337, 1338, 1339)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units-dir", required=True)
    ap.add_argument("--chain-predictors", required=True)
    ap.add_argument("--sav-cells-dir", required=True)
    ap.add_argument("--n5c-cloud", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import savb as SBV

    cloud = json.loads(pathlib.Path(args.n5c_cloud).read_text())
    cached = {m["repo_id"].split("/")[-1]: m for m in cloud["members"]}
    cached_chain = {"pi0": cloud["pi0_member"]}

    units = []
    for p in sorted(pathlib.Path(args.units_dir).glob("unit_*.json")):
        r = json.loads(p.read_text())
        task = r["unit"]
        cm = cached[task]                       # KeyError = unit not in cloud — fail loud
        units.append({"unit": task, "kind": "n5c",
                      "S_16k": r["outcomes"]["16000"]["S_recover"],
                      "S_1k": r["outcomes"]["1000"]["S_recover"],
                      "ce_16k": r["outcomes"]["16000"]["train"]["final_loss"],
                      **r["predictors"],
                      "a64": cm["A"]["64"]["a_pooled_full"],
                      "rhoL": cm["rhoL_enrichment"]["64"],
                      "frob_cached": cm["frob_dW_total"]})
    assert len(units) == 10, f"expected 10 N5C unit rows, found {len(units)}"

    cp = json.loads(pathlib.Path(args.chain_predictors).read_text())["rows"]
    cells = {}
    for p in sorted(pathlib.Path(args.sav_cells_dir).rglob("cell_*.json")):
        r = json.loads(p.read_text())
        cells[r["cell_id"]] = r

    def chain_outcome(ck, n):
        ss = []
        for s in SEEDS:
            cid = f"{ck}_cap_r512_n{n}_s{s}_confirmation"
            assert cid in cells, f"missing SAV-A frozen cell {cid}"
            ss.append(cells[cid]["eval"]["S_recover"])
        return float(np.mean(ss)), [float(v) for v in ss]

    n5c_a64 = [u["a64"] for u in units]
    n5c_rho = [u["rhoL"] for u in units]
    med_a64, med_rho = float(np.median(n5c_a64)), float(np.median(n5c_rho))
    for ck in CHAIN:
        s16, s16_seeds = chain_outcome(ck, 16000)
        s1, s1_seeds = chain_outcome(ck, 1000)
        cm = cached_chain.get(ck)
        units.append({"unit": ck, "kind": "chain", "S_16k": s16, "S_1k": s1,
                      "S_16k_seeds": s16_seeds, "S_1k_seeds": s1_seeds,
                      **cp[ck]["predictors"],
                      "a64": cm["A"]["64"]["a_pooled_full"] if cm else med_a64,
                      "rhoL": cm["rhoL_enrichment"]["64"] if cm else med_rho,
                      "a64_imputed": cm is None, "rhoL_imputed": cm is None})
    assert len(units) == 13

    y16 = np.array([u["S_16k"] for u in units])
    y1 = np.array([u["S_1k"] for u in units])
    P = {k: np.array([u[k] for u in units])
         for k in ("l2_base", "lmc_barrier", "overlap", "a64", "rhoL")}

    analysis = {}
    for yname, y in (("S_16k_primary", y16), ("S_1k_secondary", y1)):
        sp = {k: SBV.boot_spearman(v, y, n_boot=args.n_boot, seed=1337)
              for k, v in P.items()}
        Xb = np.column_stack([P["l2_base"], P["lmc_barrier"]])
        loo = SBV.delta_loo_perm(Xb, P["overlap"], y, n_perm=args.n_perm, seed=1337)
        cached_fam = SBV.delta_loo_perm(Xb, P["a64"], y, n_perm=args.n_perm, seed=1337)
        analysis[yname] = {"spearman": sp, "loo_overlap": loo,
                           "loo_a64_descriptive": cached_fam}
        print(f"[SAVB-AGG] {yname}: " + "  ".join(
            f"ρ({k})={sp[k]['rho']:+.3f}" for k in P))
        print(f"[SAVB-AGG] {yname}: LOO base={loo['r2_base']:+.3f} "
              f"+overlap={loo['r2_aug']:+.3f} Δ={loo['delta']:+.3f} p={loo['p_perm']:.4f}")

    loo = analysis["S_16k_primary"]["loo_overlap"]
    supported = bool(loo["delta"] > 0 and loo["p_perm"] < 0.10)
    verdict = ("H_repr-predicts-latent SUPPORTED (DISCOVERY): overlap adds LOO power "
               f"over L2+LMC on S(16k) — ΔLOO-R²={loo['delta']:+.3f}, perm p={loo['p_perm']:.4f}"
               if supported else
               "H_repr-predicts-latent KILLED (DISCOVERY-grade test): overlap adds no "
               f"LOO power over L2+LMC on S(16k) — ΔLOO-R²={loo['delta']:+.3f}, "
               f"perm p={loo['p_perm']:.4f}")
    res = {"units": units, "analysis": analysis, "VERDICT": verdict,
           "frozen_rule": "SUPPORTED iff ΔLOO-R²>0 AND perm p<0.10 on S_16k (prereg §2)",
           "n_units": len(units),
           "inputs": {"units_dir": args.units_dir, "chain_predictors": args.chain_predictors,
                      "sav_cells_dir": args.sav_cells_dir, "n5c_cloud": args.n5c_cloud}}
    (out / "savb_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[SAVB-AGG] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
