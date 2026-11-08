"""Phase 2: field by field, verdict_n12.json (test half) and thresholds.json -> discovery (freeze half) against this
reader's phase-1 numbers (r2_gval.json, r3_summary.json). Point values should agree exactly; bootstrap bounds differ
by the seed (job 0, reader 7)."""
import json, pathlib

here = pathlib.Path(__file__).resolve().parent
ver = json.load(open(here.parent / "verdict_n12.json"))
thr = json.load(open(here.parent / "thresholds.json"))
S = json.load(open(here / "r3_summary.json"))
G = json.load(open(here / "r2_gval.json"))

def mean3(r):
    return [r["mean"], r["lo"], r["hi"]]

rows = []
def add(half, field, job, mine):
    rows.append({"half": half, "field": field, "job": job, "reader": mine,
                 "point_diff": None if job is None or mine is None else abs(job[0] - mine[0])})

for half, J in (("test", ver["test"]), ("freeze", thr["discovery"])):
    M = S[half]
    add(half, "n real/pb/dt", [J["n"]["real"], J["n"]["pb"], J["n"]["dt"]],
        [M["real"]["n"], M["pb"]["n"], M["dt"]["n"]])
    for jf, kind, mf in (("F_real", "real", "F"), ("move_real", "real", "movement"), ("dc_real", "real", "c_own"),
                         ("do_real", "real", "c_oth"), ("drop_real", "real", "drop"), ("drop_pb", "pb", "drop"),
                         ("drop_dt", "dt", "drop"), ("F_pb", "pb", "F"), ("F_dt", "dt", "F"),
                         ("u_S_real", "real", "u_S"), ("u_P_real", "real", "u_P"), ("u_Pp_real", "real", "u_Pp"),
                         ("u_gain_real", "real", "u_gain"), ("rho_S_gain_real", "real", "rho_S_gain"),
                         ("rho_S_Bpre_real", "real", "rho_S_Bpre"), ("rho_S_gain_pb", "pb", "rho_S_gain")):
        if jf in J:
            add(half, jf, mean3(J[jf]), mean3(M[kind][mf]))
    for kind in ("real", "pb", "dt"):
        for jf, mf in (("pre", "hit_pre"), ("birth", "hit_birth"), ("own_only", "hit_rc"), ("others_only", "hit_ro")):
            w = M[kind][mf]
            add(half, f"hits {kind} {jf} (Wilson)", J["hits"][kind][jf], [w["p"], w["lo"], w["hi"]])
    for jf, mf in (("pair_F_real_pb", "F_real_minus_pb"), ("pair_F_real_dt", "F_real_minus_dt"),
                   ("pair_drop_real_pb", "drop_real_minus_pb"), ("pair_drop_real_dt", "drop_real_minus_dt"),
                   ("pair_u_pb_real", "u_S_pb_minus_real"), ("pair_u_dt_real", "u_S_dt_minus_real"),
                   ("pair_hit_real_dt", "hit_real_minus_dt"), ("pair_hit_dt_pb", "hit_dt_minus_pb"),
                   ("pair_hit_real_pb", "hit_real_minus_pb")):
        if jf in J:
            j = J[jf]
            m = M["pairs"][mf]
            add(half, jf, [j["diff"], j["lo"], j["hi"], j["n"], j["x_only"], j["y_only"]],
                [m["mean"], m["lo"], m["hi"], m["n"], m["x_only"], m["y_only"]])
    fl = J["floor_u"]
    add(half, "floor_u q025 (matched floor)", [fl["q025"], fl["mean"]],
        [M["real"]["floor_matched_job_rule"]["q025"], M["real"]["floor_matched_job_rule"]["mean"]])
    v = J["validation"] if "validation" in J else thr["validation_freeze"]
    g = G[half]
    add(half, "G-VAL rho, lo, hi, shuffle q975, slope", [v["rho"], v["lo"], v["hi"], v["shuffle_q975"],
                                                       v["slope_through_origin"]],
        [g["rho"], g["boot_lo"], g["boot_hi"], g["shuffle_q975"], g["slope_through_origin"]])
    add(half, "G-VAL pass", [float(v["pass"])], [float(g["pass_rule"])])
    if "bar_median" in J:
        add(half, "bar_median real/pb/dt", [J["bar_median"]["real"], J["bar_median"]["pb"], J["bar_median"]["dt"]],
            [M["real"]["median_bar_pre"], M["pb"]["median_bar_pre"], M["dt"]["median_bar_pre"]])
    if "dt_end_hits" in J:
        c = M["dt"]["dt_control_T15"]
        add(half, "dt_end_hits", J["dt_end_hits"], [c["precision_at_10"][k] for k in ("p", "lo", "hi")])
        add(half, "dt_end_opposite_hits", J["dt_end_opposite_hits"],
            [c["opposite_pole_precision_at_10"][k] for k in ("p", "lo", "hi")])
res = ver["u_S_resolution"]
for half in ("freeze", "test"):
    add(half, "u_S resolution n_within_noise", [res[half]["n_within_noise"]],
        [S[half]["real"]["u_S_resolution_exact_ties"]])
labels_job = {k: v["label"] for k, v in ver["labels"].items()}
labels_mine = S["labels"]
out = {"rows": rows, "labels_job": labels_job, "headline_job": ver["headline"],
       "headline_reader": labels_mine["headline"],
       "max_point_diff": max(r["point_diff"] for r in rows if r["point_diff"] is not None),
       "rows_point_diff_gt_1e-9": [r for r in rows if r["point_diff"] is not None and r["point_diff"] > 1e-9]}
(here / "r8_compare.json").write_text(json.dumps(out, indent=1))
for r in rows:
    f = lambda x: "[" + ", ".join(f"{y:.6g}" for y in x) + "]"
    print(f"{r['half']:6s} {r['field']:42s} job {f(r['job'])}  reader {f(r['reader'])}  d={r['point_diff']:.2e}")
print(out["headline_job"] == out["headline_reader"], out["max_point_diff"])
print([(r["half"], r["field"], r["point_diff"]) for r in out["rows_point_diff_gt_1e-9"]])
