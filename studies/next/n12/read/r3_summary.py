"""Summaries and labels from r1_units.json, by the rules of N12_SPEC.md sections 3.3-3.6 and 5 as written (the
reader's implementation, phase 1, before the verdict file or the verdict code was opened). Bootstrap: percentile,
10,000 resamples over units (paired: over births), seed 7. Matched floor: 1,000 replicates, seed 7."""
import json, pathlib
import numpy as np

here = pathlib.Path(__file__).resolve().parent
V = json.load(open(here / "r1_units.json"))["units"]
gval = json.load(open(here / "r2_gval.json"))
SEED = 7


def boot(x, seed=SEED, nb=10000):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (nb, x.size))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def wilson(h, n, z=1.959964):
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"hits": int(h), "n": int(n), "p": float(p), "lo": float(max(0, c - w)), "hi": float(min(1, c + w))}


def lab(r, names):
    return names[0] if r["lo"] > 0 else (names[2] if r["hi"] < 0 else names[1])


SEL = ("ABOVE_SELECTION", "AT_SELECTION", "BELOW_SELECTION")


def units(half, kind):
    return {v["birth"]: v for v in V.values() if v["kind"] == kind and (half == "both" or v["half"] == half)}


def paired(half, a, b, f):
    A, Bk = units(half, a), units(half, b)
    ids = sorted(set(A) & set(Bk))
    d = [float(f(A[i])) - float(f(Bk[i])) for i in ids]
    r = boot(d)
    r["x_only"] = int(sum(x > 0 for x in d))
    r["y_only"] = int(sum(x < 0 for x in d))
    return r


def floor(us, seed=SEED, reps=1000):
    us = np.array(us)
    rng = np.random.default_rng(seed)
    v = np.array([us[np.arange(len(us)), rng.integers(0, us.shape[1], len(us))].mean() for _ in range(reps)])
    return {"q025": float(np.quantile(v, 0.025)), "mean": float(v.mean()), "reps": reps}


out = {}
for half in ("freeze", "test", "both"):
    o = {}
    for kind in ("real", "pb", "dt"):
        U = list(units(half, kind).values())
        k = {"n": len(U)}
        k["F"] = boot([u["F"] for u in U])
        k["F_sign"] = {"pos": sum(u["F"] > 0 for u in U), "zero": sum(u["F"] == 0 for u in U),
                       "neg": sum(u["F"] < 0 for u in U)}
        k["movement"] = boot([u["move"] for u in U])
        k["c_own"] = boot([u["c_own"] for u in U])
        k["c_oth"] = boot([u["c_oth"] for u in U])
        k["drop"] = boot([u["drop"] for u in U])
        for h in ("hit_pre", "hit_birth", "hit_rc", "hit_ro"):
            k[h] = wilson(sum(u[h] for u in U), len(U))
        k["u_S"] = boot([u["u_S"] for u in U])
        k["u_P"] = boot([u["u_P"] for u in U])
        k["u_Pp"] = boot([u["u_Pp"] for u in U])
        k["u_gain"] = boot([u["u_gain"] for u in U])
        rr = [u["rho_S_gain"] for u in U if u["rho_S_gain"] is not None]
        k["rho_S_gain"] = boot(rr)
        rb = [u["rho_S_Bpre"] for u in U if u["rho_S_Bpre"] is not None]
        k["rho_S_Bpre"] = boot(rb)
        k["median_bar_pre"] = float(np.median([u["bar_pre"] for u in U]))
        k["median_B_pre"] = float(np.median([u["B_pre"] for u in U]))
        k["median_B_birth"] = float(np.median([u["B_birth"] for u in U]))
        k["median_r0"] = float(np.median([u["r0"] for u in U]))
        k["r0_eq_1"] = sum(u["r0"] == 1 for u in U)
        k["share_B_pre_lt_0.999"] = float(np.mean([u["B_pre"] < 0.999 for u in U]))
        k["share_B_birth_ge_0.999"] = float(np.mean([u["B_birth"] >= 0.999 for u in U]))
        k["share_r_o_ge_r0"] = float(np.mean([u["r_o"] >= u["r0"] for u in U]))
        k["share_pole_birth_above_all_C_pre"] = float(np.mean([u["pole_birth_above_allC_pre"] for u in U]))
        k["median_Cmin_B_pre"] = float(np.median([u["Cmin_B_pre"] for u in U]))
        k["median_Cmax_B_pre"] = float(np.median([u["Cmax_B_pre"] for u in U]))
        k["ties_at_20"] = sum(u["tie20"] for u in U)
        k["n_cand_lt_20"] = sum(u["n_cand"] < 20 for u in U)
        k["mean_F_minus_move"] = float(np.mean([u["F"] - u["move"] for u in U]))
        if kind == "dt" and half != "both":
            ends = [u for u in U if "end_rank" in u]
            k["dt_control_T15"] = {"precision_at_10": wilson(sum(u["end_rank"] <= 10 for u in ends), len(ends)),
                                   "opposite_pole_precision_at_10":
                                       wilson(sum(u["end_rank_opp"] <= 10 for u in ends), len(ends))}
            k["dt_control_T15"]["pass"] = bool(k["dt_control_T15"]["precision_at_10"]["p"] >= 0.9 and
                                               k["dt_control_T15"]["opposite_pole_precision_at_10"]["p"] < 0.9)
            k["d_equals_d_real"] = sum(u["d"] == u["d_real"] for u in U)
        if kind == "real":
            k["u_S_resolution_exact_ties"] = sum(u["S_gap_min"] <= 0.0 for u in U)
            # matched floor: (a) the matched pole compared with all 20 of C (this reader, first pass);
            # (b) a matched pole that is itself in C compared with the other 19 (the job's rule, seen in phase 1 as
            # the only per-unit difference, 9 units)
            ua = [u["u_matched"] for u in U]
            ub = []
            for u in U:
                row = []
                for j, x in enumerate(u["u_matched"]):
                    row.append((x * 20 - 1) / 19 if j in u["matched_in_C"] else x)
                ub.append(row)
            k["floor_matched_all20"] = floor(ua)
            k["floor_matched_job_rule"] = floor(ub)
        o[kind] = k
    o["pairs"] = {
        "drop_real_minus_pb": paired(half, "real", "pb", lambda u: u["drop"]),
        "drop_real_minus_dt": paired(half, "real", "dt", lambda u: u["drop"]),
        "drop_dt_minus_pb": paired(half, "dt", "pb", lambda u: u["drop"]),
        "hit_real_minus_dt": paired(half, "real", "dt", lambda u: u["hit_pre"]),
        "hit_dt_minus_pb": paired(half, "dt", "pb", lambda u: u["hit_pre"]),
        "hit_real_minus_pb": paired(half, "real", "pb", lambda u: u["hit_pre"]),
        "u_S_pb_minus_real": paired(half, "pb", "real", lambda u: u["u_S"]),
        "u_S_dt_minus_real": paired(half, "dt", "real", lambda u: u["u_S"]),
        "F_real_minus_pb": paired(half, "real", "pb", lambda u: u["F"]),
        "F_real_minus_dt": paired(half, "real", "dt", lambda u: u["F"]),
        "bar_real_minus_pb": paired(half, "real", "pb", lambda u: u["bar_pre"]),
        "bar_real_minus_dt": paired(half, "real", "dt", lambda u: u["bar_pre"]),
    }
    out[half] = o

# ---- labels, the spec's section 5 rules applied to this reader's test-half numbers
T, Fz = out["test"], out["freeze"]
lbl = {}
lbl["movement_ok"] = {h: bool(out[h]["real"]["movement"]["lo"] > 0) for h in ("freeze", "test")}
lbl["usable_real_test"] = T["real"]["n"]
f = T["real"]["F"]
lbl["H_cf"] = "OWN_RISE" if f["lo"] > 0 else ("OTHERS_FALL" if f["hi"] < 0 else "MIXED")
lbl["H_away_sel_pb"] = lab(T["pairs"]["drop_real_minus_pb"], SEL)
lbl["H_away_sel_dt"] = lab(T["pairs"]["drop_real_minus_dt"], SEL)
gv = {h: bool(gval[h]["pass_rule"]) for h in ("freeze", "test")}
lbl["G_VAL_pass"] = gv
for rule in ("all20", "job_rule"):
    bar = min(0.5, Fz["real"][f"floor_matched_{rule}"]["q025"], T["real"][f"floor_matched_{rule}"]["q025"])
    us, ug = T["real"]["u_S"], T["real"]["u_gain"]
    if us["hi"] < ug["lo"]:
        s = "ABOVE_CEILING"
    else:
        s = "SUPPORTED" if us["hi"] < bar else ("REFUTED" if us["mean"] >= bar else "UNDECIDED")
    lbl[f"H_signal_exploratory_{rule}"] = {"label_if_G_VAL_had_passed": s, "bar": bar}
lbl["H_signal"] = "DOES NOT COUNT (G-VAL fails on both halves)" if not all(gv.values()) else "see exploratory"
lbl["H_sel_signal_pb_exploratory"] = lab(T["pairs"]["u_S_pb_minus_real"], SEL)
lbl["H_sel_signal_dt_exploratory"] = lab(T["pairs"]["u_S_dt_minus_real"], SEL)
rb = T["real"]["rho_S_gain"]
lbl["H_bridge_exploratory"] = "FORETELLS" if rb["lo"] > 0 else ("CONTRARY" if rb["hi"] < 0 else "UNDECIDED")
lbl["H_bridge"] = "DOES NOT COUNT (G-VAL)" if not all(gv.values()) else lbl["H_bridge_exploratory"]
lbl["H_sel_dt"] = lab(T["pairs"]["hit_real_minus_dt"], SEL)
lbl["H_dt_pb"] = lab(T["pairs"]["hit_dt_minus_pb"], ("HIGHER", "SAME", "LOWER"))
ctrl = T["dt"]["dt_control_T15"]["pass"]
if not ctrl:
    lbl["H_dt"] = "DOES NOT COUNT"
elif lbl["H_sel_dt"] in ("AT_SELECTION", "ABOVE_SELECTION") and lbl["H_dt_pb"] == "LOWER":
    lbl["H_dt"] = "SUPPORTED"
elif lbl["H_sel_dt"] == "BELOW_SELECTION":
    lbl["H_dt"] = "REFUTED"
else:
    lbl["H_dt"] = "UNDECIDED"
lbl["H_0_real_minus_twin_contrasts"] = {k: lab(T["pairs"][k], SEL) for k in
                                        ("drop_real_minus_pb", "drop_real_minus_dt", "hit_real_minus_dt",
                                         "hit_real_minus_pb")}
lbl["headline"] = (f"H_cf {lbl['H_cf']}; H_away_sel_pb {lbl['H_away_sel_pb']}; H_signal DOES NOT COUNT; "
                   f"H_bridge DOES NOT COUNT; H_dt {lbl['H_dt']}") if not all(gv.values()) else None
out["labels"] = lbl
(here / "r3_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(lbl, indent=1))
