"""Reader of M2, phase 2: field by field, verdict_m2.json and thresholds.json (DISCOVERY) against the reader's own
numbers (read/r2_from_ranks.json, r6_gates.json, r7_floor.json). Run after the phase-1 commit (ce7683df8).
Output: read/r8_compare.json (largest absolute difference per field).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r2_from_ranks import wilson

M2 = Path(__file__).resolve().parents[1]


def main():
    V = json.loads((M2 / "verdict_m2.json").read_text())
    T = json.loads((M2 / "thresholds.json").read_text())
    R = json.loads((M2 / "read/r2_from_ranks.json").read_text())
    G = json.loads((M2 / "read/r6_gates.json").read_text())
    F = json.loads((M2 / "read/r7_floor.json").read_text())
    rows = []

    def prop(src, name, mine):
        p, lo, hi, h = src
        if "wilson" not in mine:
            mine = dict(mine, wilson=wilson(mine["hits"], mine["n"]))
        d = max(abs(p - mine["p"]), abs(lo - mine["wilson"][0]), abs(hi - mine["wilson"][1]))
        rows.append({"field": name, "job": [h, p, lo, hi], "reader": [mine["hits"], mine["p"]] + mine["wilson"],
                     "hits_equal": h == mine["hits"], "max_abs_diff": d})

    def pair(src, name, mine_m, mine_ci, lab=None, mine_lab=None):
        d = max(abs(src["diff"] - mine_m), abs(src["lo"] - mine_ci[0]), abs(src["hi"] - mine_ci[1]))
        rows.append({"field": name, "job": [src["diff"], src["lo"], src["hi"]], "reader": [mine_m] + mine_ci,
                     "max_abs_diff": d, "label_job": lab, "label_reader": mine_lab})

    for half, src in (("test", V["test"]), ("freeze", T["discovery"])):
        H = R[half]
        for f in ("pre_A", "pre_B", "pre_ratio", "pre_maxabs", "pre_density", "pre_colnorm", "birth_A", "birth_B",
                  "end_A", "end_B", "end_opposite_A", "end_opposite_B"):
            prop(src[f], f"{half}.{f}", H[f])
        prop(src["pb_pre_B"], f"{half}.pb_pre_B", {"hits": H["twin_B"]["twin_hits"], "p": H["twin_B"]["twin_p"],
                                                   "wilson": H["twin_B"]["twin_wilson"]})
        prop(src["pb_pre_A"], f"{half}.pb_pre_A", {"hits": H["twin_A"]["twin_hits"], "p": H["twin_A"]["twin_p"],
                                                   "wilson": H["twin_A"]["twin_wilson"]})
        pair(src["pair_A_B"], f"{half}.pair_A_B", H["H_AB"]["diff_mean"], H["H_AB"]["diff_ci"])
        pair(src["pair_real_pb_B"], f"{half}.pair_real_pb_B", H["twin_B"]["diff_mean"], H["twin_B"]["diff_ci"])
        pair(src["pair_real_pb_A"], f"{half}.pair_real_pb_A", H["twin_A"]["diff_mean"], H["twin_A"]["diff_ci"])
        for s in ("A", "B"):
            fj = src[f"floor_{s}"]
            rows.append({"field": f"{half}.floor_{s}", "job": [fj["q975"], fj["mean"], fj["max"], fj["all_pairs_mean"]],
                         "reader": [F[half][s]["p97.5"], H[f"floor_{s}"]["mean"], H[f"floor_{s}"]["max"],
                                    H[f"floor_{s}"]["matched_hit_share"]],
                         "max_abs_diff_q975": abs(fj["q975"] - F[half][s]["p97.5"]),
                         "max_abs_diff_mean": abs(fj["mean"] - H[f"floor_{s}"]["mean"])})
    for s in ("A", "B"):
        rows.append({"field": f"bar_{s}", "job": V["bars"][s], "reader": F["bar"][s],
                     "max_abs_diff": abs(V["bars"][s] - F["bar"][s])})
    L = R["labels"]
    for k in ("H_B", "H_A"):
        rows.append({"field": f"label.{k}", "job": V["labels"][k]["label"], "reader": L[k]["label"],
                     "equal": V["labels"][k]["label"] == L[k]["label"]})
    for k in ("H_AB", "H_sel_B", "H_sel_A"):
        rows.append({"field": f"label.{k}", "job": V["labels"][k]["label"], "reader": L[k],
                     "equal": V["labels"][k]["label"] == L[k]})
    rows.append({"field": "gates", "job": V["gates"], "reader": {"problems": G["problems"], "n_gates": G["n_gates"],
                                                                  "complete": G["complete"]}})
    rows.append({"field": "cross_device_gex.points_over", "job": V["cross_device_gex"]["points_over"],
                 "reader": sorted([int(x[1:3]) for x in G["cross_device_gex_above_tol"]], reverse=True)})
    rows.append({"field": "degenerate_poles", "job": {t: p["poles"] for t, p in V["degenerate_poles"]["points"].items() if p["n"]},
                 "reader": {t: p["degenerate"] for t, p in G["points"].items() if p["degenerate"]}})
    rows.append({"field": "thresholds.sha256_ranks_freeze", "job": T["sha256_ranks_freeze"],
                 "reader": R["sha256"]["freeze"], "equal": T["sha256_ranks_freeze"] == R["sha256"]["freeze"]})
    rows.append({"field": "thresholds.sha256_job_metrics", "job": T["sha256_job_metrics"], "reader": G["sha256"],
                 "equal": T["sha256_job_metrics"] == G["sha256"]})
    rows.append({"field": "headline", "job": V["headline"],
                 "reader": f"H_B {L['H_B']['label']}; H_sel_B {L['H_sel_B']}; H_A {L['H_A']['label']}; H_AB {L['H_AB']}"})
    (M2 / "read/r8_compare.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
