"""N2V reader, step 3: the verdict quantities with my own code (numpy only; nothing imported from infra/).

    python -I r2_verdict.py --recount r1_recount.json --thresholds <worktree>/studies/next/n2v/thresholds.json --out r2_verdict.json

Inputs: the per-pair recount from losses/tNN.npz (r1_recount.json; pred_bf16 is N12's banked abl[i][1]) and the
freeze half's labels from thresholds.json (freeze 2). Per half and readout (SPEC section 3.4):
  window      pred_bf16        against dL
  fd          0.1 x pred_bf16  against dL_fd
  own         pred_jvp_own     against dL_own
  window_jvp  pred_jvp         against dL
the Spearman over pairs (average ranks for ties); the registered interval: a percentile bootstrap over units, 1,000
resamples, from one default_rng(0) stream that first draws the 1,000 shuffles (n12.validation's order); the shuffle's
97.5th percentile; the sign agreement; the slope through the origin; seeds 1 and 2; a basic (reverse-percentile)
interval and a 10,000-resample percentile interval as second methods; G-ROOM (1,000 replicates, seed 0, measurement =
prediction + each pair's probe with a random sign) and its failing input (N12's bf16 measured change of each unit's
carrier as the floor); the ceiling (rho > 0.999) with the inspection; the fd copy test; the labels, the both-halves
rule and the branch.
"""
import argparse
import json
import pathlib

import numpy as np

READ = {"window": ("pred_bf16_n12", "dL", 1.0), "fd": ("pred_bf16_n12", "dL_fd", 0.1),
        "own": ("pred_jvp_own", "dL_own", 1.0), "window_jvp": ("pred_jvp", "dL", 1.0)}
NOISE = {"window": "probe", "fd": "probe", "own": "probe_own", "window_jvp": "probe"}
RHO_MIN, CEIL, ROOM_MIN, INSPECT, FD_COPY = 0.7, 0.999, 0.9, 1e-3, 1e-6
ORDER = [14, 13, 12, 11, 10, 9, 8, 7, 5]     # the job's point order, which is the records' file order


def avg_rank(x):
    """Average ranks, 0-based (ties share the mean of their positions)."""
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    new = np.r_[True, xs[1:] != xs[:-1]]
    start = np.flatnonzero(new)
    cnt = np.diff(np.r_[start, x.size])
    r = np.empty(x.size)
    r[o] = (start + (cnt - 1) / 2.0)[np.cumsum(new) - 1]
    return r


def rho(x, y):
    a, b = avg_rank(x), avg_rank(y)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return None if den == 0 else float((a * b).sum() / den)


def stats(units, pr, me, seed=0, n_boot=1000, n_shuf=1000):
    """The registered procedure's draws, in its order: shuffles first, then the unit bootstrap, from one stream."""
    rng = np.random.default_rng(seed)
    shuf = np.array([rho(rng.permutation(pr), me) for _ in range(n_shuf)])
    uu = np.unique(units)
    pos = {u: np.flatnonzero(units == u) for u in uu}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(uu, uu.size)
        idx = np.concatenate([pos[u] for u in pick])
        v = rho(pr[idx], me[idx])
        if v is not None:
            bs.append(v)
    bs = np.array(bs)
    return shuf, bs


def readout(pairs, ro, seeds=(0, 1, 2)):
    pk, mk, sc = READ[ro]
    units = np.array([p["uidx"] for p in pairs])
    pr = np.array([sc * p[pk] for p in pairs], np.float64)
    me = np.array([p[mk] for p in pairs], np.float64)
    r = rho(pr, me)
    out = {"n_pairs": len(pairs), "n_units": int(np.unique(units).size), "rho": r}
    for s in seeds:
        shuf, bs = stats(units, pr, me, seed=s)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out[f"seed{s}"] = {"lo": float(lo), "hi": float(hi), "shuffle_q975": float(np.quantile(shuf, 0.975)),
                           "basic_lo": float(2 * r - hi), "basic_hi": float(2 * r - lo)}
    _, bs10 = stats(units, pr, me, seed=0, n_boot=10000)
    out["seed0_10000"] = {"lo": float(np.quantile(bs10, 0.025)), "hi": float(np.quantile(bs10, 0.975))}
    out["sign_agreement"] = float(np.mean(np.sign(pr) == np.sign(me)))
    out["slope_through_origin"] = float((pr * me).sum() / (pr * pr).sum())
    out["median_abs_meas"], out["median_abs_pred"] = float(np.median(np.abs(me))), float(np.median(np.abs(pr)))
    out["failing_input_rho"] = rho(-pr, me)
    out["failing_input_pass"] = bool(out["failing_input_rho"] >= RHO_MIN)
    rel = np.abs(me - pr) / np.abs(pr)
    out["above_ceiling"] = bool(r > CEIL)
    out["inspection"] = {"n_equal": int((me == pr).sum()), "median_rel_dev": float(np.median(rel)),
                         "cleared": bool((me != pr).all() and np.median(rel) >= INSPECT)}
    if ro == "fd":
        out["fd_copy_max_rel_dev"] = float(rel.max())
        out["fd_copy"] = bool(rel.max() < FD_COPY)
    # G-ROOM, the registered draw (rng.choice of the two signs per pair, one stream, seed 0)
    nz = np.array([p[NOISE[ro]] for p in pairs], np.float64)
    out["room"] = room(pr, nz, 0)
    # the same draw with the pairs in the records' file order (point order T_14 ... T_5, units ascending, rank), the
    # order in which the registered code walks them; only the assignment of random signs to pairs changes
    fo = np.array(sorted(range(len(pairs)), key=lambda i: pairs[i]["file_order"]))
    out["room_file_order"] = room(pr[fo], nz[fo], 0)
    out["room_seed1"] = room(pr, nz, 1)
    car = {p["uidx"]: abs(p["meas_bf16_n12"]) for p in pairs if p["rank"] == 0}
    out["room_failing_input"] = room(pr, np.array([car[p["uidx"]] for p in pairs]), 0)
    out["room_failing_input_file_order"] = room(pr[fo], np.array([car[p["uidx"]] for p in pairs])[fo], 0)
    s0 = out["seed0"]
    out["pass_rule"] = bool(r >= RHO_MIN and s0["lo"] > s0["shuffle_q975"])
    out["label"] = label(out, ro)
    return out


def room(pr, nz, seed, reps=1000):
    rng = np.random.default_rng(seed)
    v = np.array([rho(pr, pr + rng.choice([-1.0, 1.0], pr.size) * nz) for _ in range(reps)])
    return {"median_rho": float(np.median(v)), "q05": float(np.quantile(v, 0.05)), "min": float(v.min()),
            "pass": bool(np.median(v) >= ROOM_MIN), "median_abs_noise": float(np.median(np.abs(nz)))}


def label(o, ro):
    if not o["room"]["pass"]:
        return "DOES NOT COUNT: no room"
    if o["failing_input_pass"]:
        return "DOES NOT COUNT: failing input passed"
    if ro == "fd":
        if o["fd_copy"]:
            return "DOES NOT COUNT: above ceiling"
    elif o["above_ceiling"] and not o["inspection"]["cleared"]:
        return "DOES NOT COUNT: above ceiling"
    return "PASS" if o["pass_rule"] else "FAIL"


def both(a, b):
    if "DOES NOT COUNT" in a or "DOES NOT COUNT" in b:
        return a if "DOES NOT COUNT" in a else b
    return "PASS" if a == b == "PASS" else "FAIL"


def branch(win, fd):
    if win.startswith("DOES NOT COUNT"):
        return "D"
    if win == "PASS":
        return "A"
    return "B" if fd == "PASS" else "C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recount", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rc = json.loads(pathlib.Path(a.recount).read_text())
    th = json.loads(pathlib.Path(a.thresholds).read_text())
    out = {"halves": {}}
    for h in ("freeze", "test"):
        ps = sorted([p for p in rc["pairs"] if p["half"] == h], key=lambda p: (p["unit"], p["rank"]))
        ids = sorted({p["unit"] for p in ps})
        for p in ps:
            p["uidx"] = ids.index(p["unit"])
            p["file_order"] = (ORDER.index(p["t"]), p["unit"], p["rank"])
        blk = {"n_pairs": len(ps), "n_units": len(ids)}
        for ro in READ:
            blk[ro] = readout(ps, ro)
        a16 = np.array([p["pred_bf16_n12"] for p in ps])
        aj = np.array([p["pred_jvp"] for p in ps])
        blk["bf16_vs_jvp_prediction"] = {"spearman": rho(a16, aj),
                                         "median_rel_diff": float(np.median(np.abs(a16 - aj) / np.abs(aj)))}
        blk["probe_median_abs_carrier"] = float(np.median([abs(p["probe"]) for p in ps if p["rank"] == 0]))
        blk["probe_own_median_abs_carrier"] = float(np.median([abs(p["probe_own"]) for p in ps if p["rank"] == 0]))
        out["halves"][h] = blk
    lt = {ro: out["halves"]["test"][ro]["label"] for ro in READ}
    lf_mine = {ro: out["halves"]["freeze"][ro]["label"] for ro in READ}
    lf = th["labels_freeze"]
    lab = {ro: both(lf[ro], lt[ro]) for ro in READ}
    t = out["halves"]["test"]
    refuse = []
    if t["n_units"] < 50:
        refuse.append("fewer than 50 test units")
    if t["window"]["room_failing_input"]["pass"]:
        refuse.append("G-ROOM's failing input passed on the test half")
    out["labels_freeze_thresholds"] = lf
    out["labels_freeze_mine"] = lf_mine
    out["labels_test"] = lt
    out["labels_both"] = lab
    out["refusals"] = refuse
    out["branch"] = "D" if refuse else branch(lab["window"], lab["fd"])
    # the freeze half against thresholds.json
    cmpf = {}
    for ro in READ:
        mv, tv = out["halves"]["freeze"][ro], th["freeze"][ro]
        cmpf[ro] = {"rho": [mv["rho"], tv["validation"]["rho"]], "lo": [mv["seed0"]["lo"], tv["validation"]["lo"]],
                    "hi": [mv["seed0"]["hi"], tv["validation"]["hi"]],
                    "shuffle_q975": [mv["seed0"]["shuffle_q975"], tv["validation"]["shuffle_q975"]],
                    "slope": [mv["slope_through_origin"], tv["validation"]["slope_through_origin"]],
                    "sign": [mv["sign_agreement"], tv["validation"]["sign_agreement"]],
                    "room_median": [mv["room"]["median_rho"], tv["room"]["median_rho"]],
                    "room_median_file_order": [mv["room_file_order"]["median_rho"], tv["room"]["median_rho"]]}
    cmpf["room_failing_input"] = [out["halves"]["freeze"]["window"]["room_failing_input"]["median_rho"],
                                  th["freeze"]["room_failing_input"]["median_rho"]]
    cmpf["room_failing_input_file_order"] = [out["halves"]["freeze"]["window"]["room_failing_input_file_order"]["median_rho"],
                                             th["freeze"]["room_failing_input"]["median_rho"]]
    cmpf["bf16_vs_jvp"] = [out["halves"]["freeze"]["bf16_vs_jvp_prediction"], th["freeze"]["bf16_vs_jvp_prediction"]]
    out["freeze_mine_vs_thresholds"] = cmpf
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("labels_freeze_thresholds", "labels_freeze_mine", "labels_test",
                                          "labels_both", "refusals", "branch")}, indent=1))


if __name__ == "__main__":
    main()
