"""N6 stage 2 (spec section 2.3), the record beside `n6 final`: reads the final pairs that `n6 final` wrote from L1's banked
column norms and energies, and writes the balance numbers per quantity, the count per half, the overlap with the stage-1 pairs,
and the registered failing input of the stage-2 balance gate (dry run A14: the dated poles' column norms planted as exp(-0.2 age))
evaluated on the real L1 arrays, where it must stop with "L2 does not launch".

    PYTHONPATH=. python3 -m infra.pg.n6 final --pairs studies/next/n6/pairs.json --l1 studies/next/n6/banks/L1-02 \
        --out studies/next/n6/pairs_L2.json
    PYTHONPATH=. python3 studies/next/n6/stage2_check.py [bank dir] [pairs_L2.json] [out]

The file name pairs_L2.json is the one the L2 job fetches from results/next_n6/inputs_L2/ (n6_edit.INPUT_SHA["L2"]).
"""
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import n6  # noqa: E402

HERE = pathlib.Path("studies/next/n6")
BANK = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "banks/L1-02"
FINAL = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "pairs_L2.json"
OUT = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "stage2_check.json"


def main():
    s1 = json.loads((HERE / "pairs.json").read_text())
    fin = json.loads(FINAL.read_text())
    cn, en = np.load(BANK / "colnorms.npy"), np.load(BANK / "energies.npy")
    assert cn.shape == (32, 11008) and en.shape == (32, 11008, 2), (cn.shape, en.shape)
    keys = ("ln_max_abs", "ln_density", "ln_colnorm", "ln_E_plus")
    b = fin["balance"]
    bal = {k: {m: b[k][m] for m in ("rho_pair", "smd", "pooled_within_layer", "mean_abs_diff")} for k in keys}
    un = fin["failing_input_unmatched"]
    poles = n6.load_dated()
    cn2 = cn.copy()
    for p in poles:
        cn2[p["layer"], p["neuron"]] = math.exp(-0.2 * p["age"])
    cov2 = n6.covariates(poles, {"colnorms": cn2, "energies": en})          # final_pairs' two stop conditions, on the planted input
    scan2 = n6.caliper_scan(poles, cov2, n6.GRID_INST, fixed=s1["caliper_scan"]["chosen"])
    n2 = None if scan2["chosen"] is None else len(n6.match(poles, scan2["chosen"], cov2))
    stopped = scan2["chosen"] is None or n2 < n6.MIN_PAIRS_L2
    assert stopped, f"the registered failing input did not stop stage 2: {scan2['chosen']}, {n2} pairs"
    msg = ("no instance grid point gives a balanced matched set: L2 does not launch" if scan2["chosen"] is None else
           f"{n2} pairs, fewer than {n6.MIN_PAIRS_L2}: L2 does not launch")
    planted_best = sorted(scan2["grid"], key=lambda r: -r["n_pairs"])[0]
    cov = n6.covariates(poles, {"colnorms": cn, "energies": en})
    lay = np.array([p["layer"] for p in poles])
    age = np.array([p["age"] for p in poles], float)
    s1_set = {(p["early"], p["late"]) for p in s1["pairs"]}
    fin_set = {(p["early"], p["late"]) for p in fin["pairs"]}
    out = {"what": "N6 stage 2 record (spec section 2.3): the final pairs from L1's banked arrays, balance, halves, failing input",
           "inputs": {"pairs_L2.json": n6.sha256_file(FINAL), "colnorms.npy": n6.sha256_file(BANK / "colnorms.npy"),
                      "energies.npy": n6.sha256_file(BANK / "energies.npy"), "pairs.json (stage 1)": n6.sha256_file(HERE / "pairs.json")},
           "chosen_calipers": fin["caliper_scan"]["chosen"],
           "n_pairs": fin["n_pairs"], "n_freeze": fin["n_freeze"], "n_test": fin["n_test"],
           "n_layers": len({p["layer"] for p in fin["pairs"]}), "min_pairs_L2": n6.MIN_PAIRS_L2,
           "at_least_40": fin["n_pairs"] >= n6.MIN_PAIRS_L2, "balanced": n6.balanced(b),
           "bounds": {"rho_pair": n6.RHO_MAX, "smd": n6.SMD_MAX, "rho_pair_on": ["ln_max_abs", "ln_density"], "smd_on": list(keys)},
           "balance": bal,
           "unmatched_all_600_within_layer_spearman": {k: n6.stratified_spearman(age, cov[k], lay) for k in keys},
           "failing_input_unmatched_pairs": {"n_pairs": un["n_pairs"], "balanced": un["balanced"],
                                             **{k: {m: un["balance"][k][m] for m in ("rho_pair", "smd")} for k in keys}},
           "failing_input_planted_age_colnorm": {"rule": "column norm of every dated pole set to exp(-0.2 age) (dry run A14), real E+",
                                                 "stopped": stopped, "message": msg,
                                                 "balanced_grid_points": sum(r["balanced"] for r in scan2["grid"]),
                                                 "largest_grid_point": planted_best},
           "overlap_with_stage1": {"pairs_kept": len(s1_set & fin_set), "stage1_pairs": len(s1_set)},
           "grid_points_balanced": sum(r["balanced"] for r in fin["caliper_scan"]["grid"]),
           "grid_points": len(fin["caliper_scan"]["grid"]),
           "age_gap_mean": float(np.mean([p["age_late"] - p["age_early"] for p in fin["pairs"]])),
           "n_concordant_adj": {"all": sum(p["concordant_adj"] for p in fin["pairs"]),
                                "test": sum(p["concordant_adj"] for p in fin["pairs"] if p["half"] == "test")}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("chosen_calipers", "n_pairs", "n_freeze", "n_test", "at_least_40", "balanced")}))
    for k in keys:
        print(k, "rho_pair %+.3f smd %+.3f" % (bal[k]["rho_pair"], bal[k]["smd"]),
              "| unmatched rho_pair %+.3f smd %+.3f" % (out["failing_input_unmatched_pairs"][k]["rho_pair"],
                                                        out["failing_input_unmatched_pairs"][k]["smd"]))
    print("planted failing input:", msg)


if __name__ == "__main__":
    main()
