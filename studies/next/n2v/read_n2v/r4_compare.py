"""N2V reader, step 4: every number of the registered verdict (verdict_n2v.json, test half) against mine (r2_verdict.json).

    python -I r4_compare.py --mine r2_verdict.json --registered <scratchpad>/verdict_n2v.json --out r4_compare.json
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--mine", required=True)
ap.add_argument("--registered", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = json.loads(pathlib.Path(a.mine).read_text())
v = json.loads(pathlib.Path(a.registered).read_text())
mt, vt = m["halves"]["test"], v["test"]
rows = []


def add(name, mine, reg):
    rows.append({"field": name, "mine": mine, "registered": reg,
                 "abs_diff": None if isinstance(mine, (str, bool)) else abs(mine - reg),
                 "equal": mine == reg})


for ro in ("window", "fd", "own", "window_jvp"):
    o, r = mt[ro], vt[ro]
    val = r["validation"]
    add(f"{ro}.rho", o["rho"], val["rho"])
    add(f"{ro}.lo", o["seed0"]["lo"], val["lo"])
    add(f"{ro}.hi", o["seed0"]["hi"], val["hi"])
    add(f"{ro}.shuffle_q975", o["seed0"]["shuffle_q975"], val["shuffle_q975"])
    add(f"{ro}.slope_through_origin", o["slope_through_origin"], val["slope_through_origin"])
    add(f"{ro}.sign_agreement", o["sign_agreement"], val["sign_agreement"])
    add(f"{ro}.median_abs_meas", o["median_abs_meas"], val["median_abs_meas"])
    add(f"{ro}.median_abs_pred", o["median_abs_pred"], val["median_abs_pred"])
    add(f"{ro}.failing_input_rho", o["failing_input_rho"], val["failing_input"]["rho"])
    add(f"{ro}.failing_input_pass", o["failing_input_pass"], val["failing_input"]["pass"])
    add(f"{ro}.pass", o["pass_rule"], val["pass"])
    add(f"{ro}.above_ceiling", o["above_ceiling"], val["above_ceiling"])
    add(f"{ro}.room.median_rho (file order)", o["room_file_order"]["median_rho"], r["room"]["median_rho"])
    add(f"{ro}.room.q05 (file order)", o["room_file_order"]["q05"], r["room"]["q05"])
    add(f"{ro}.room.pass", o["room"]["pass"], r["room"]["pass"])
    add(f"{ro}.room.median_abs_noise", o["room"]["median_abs_noise"], r["room"]["median_abs_noise"])
    if ro == "fd":
        add("fd.max_rel_dev", o["fd_copy_max_rel_dev"], val["max_rel_dev"])
        add("fd.copy (above_ceiling)", o["fd_copy"], val["above_ceiling"])
    add(f"{ro}.label_test", m["labels_test"][ro], v["labels_test"][ro])
    add(f"{ro}.label_both", m["labels_both"][ro], v["labels"][ro])
add("room_failing_input.median_rho (file order)", mt["window"]["room_failing_input_file_order"]["median_rho"],
    vt["room_failing_input"]["median_rho"])
add("room_failing_input.pass", mt["window"]["room_failing_input"]["pass"], vt["room_failing_input"]["pass"])
add("bf16_vs_jvp.spearman", mt["bf16_vs_jvp_prediction"]["spearman"], vt["bf16_vs_jvp_prediction"]["spearman"])
add("bf16_vs_jvp.median_rel_diff", mt["bf16_vs_jvp_prediction"]["median_rel_diff"], vt["bf16_vs_jvp_prediction"]["median_rel_diff"])
add("probe_median_abs", mt["probe_median_abs_carrier"], vt["probe_median_abs"])
add("probe_own_median_abs", mt["probe_own_median_abs_carrier"], vt["probe_own_median_abs"])
add("branch", m["branch"], v["decision"]["branch"])
num = [x for x in rows if x["abs_diff"] is not None]
out = {"n_fields": len(rows), "n_equal": sum(x["equal"] for x in rows),
       "max_abs_diff_numeric": max(x["abs_diff"] for x in num),
       "not_equal": [x for x in rows if not x["equal"]], "rows": rows}
pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("n_fields", "n_equal", "max_abs_diff_numeric")}, indent=1))
print("not bitwise equal:", [(x["field"], x["abs_diff"]) for x in out["not_equal"]])
