"""N2V reader, step 5 (branch A): N12's gradient statistics re-read with my own code from N12's banked records, with the
fp32 window dL (my recount from losses/tNN.npz) in place of N12's bf16 measured column. A re-reading of known values:
N12's test half was read on 2026-11-06 and these values are printed in studies/next/n12/READ_n12.md.

    python -I r5_reread.py --repo <worktree> --recount r1_recount.json --registered <scratchpad>/verdict_n2v.json --out r5_reread.json

N12's rules (N12_SPEC.md section 5 step 7, 3.4, 3.5), recomputed here without infra/:
  G-VAL on N12's records with the fp32 measurement: Spearman, unit bootstrap (1,000, one default_rng(0) stream after
    1,000 shuffles), shuffle q975; the freeze half in N12's file order (as n2v.reread passes it), the test half with
    real units sorted by birth (as n12.half_summary does).
  H_signal: u_S of each real test unit (share of its 20 competitors with S >= the carrier's S), recomputed from the
    record's comp[*].S and pull.S; mean with a 10,000-resample bootstrap (seed 0); bar = min(0.5, N12's freeze floor
    q025 from N12's thresholds.json, the test floor q025: 1,000 replicates, seed 0, one matched pole per unit from
    pull.u_matched); SUPPORTED if hi < bar, REFUTED if mean >= bar, else UNDECIDED; ABOVE_CEILING if hi < u_gain's lo.
  H_sel_signal_pb / _dt: paired u_S(twin) - u_S(real) over births with that twin; 10,000 resamples, seed 0; ABOVE if
    lo > 0, BELOW if hi < 0, else AT; NOT_RUN below 50 pairs.
  H_bridge: mean over real units of Spearman(S_q, gain_q) over the competitors (recomputed from comp[*].S and gain);
    FORETELLS if lo > 0, CONTRARY if hi < 0, else UNDECIDED.
"""
import argparse
import json
import pathlib

import numpy as np


def read_jsonl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def avg_rank(x):
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
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    a, b = avg_rank(x), avg_rank(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def gval(recs):
    pairs = [(i, a[1], a[2]) for i, r in enumerate(recs) for a in r["abl"]]
    ui = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs], np.float64)
    me = np.array([p[2] for p in pairs], np.float64)
    r = rho(pr, me)
    rng = np.random.default_rng(0)
    shuf = np.array([rho(rng.permutation(pr), me) for _ in range(1000)])
    uu = np.unique(ui)
    bs = []
    for _ in range(1000):
        pick = rng.choice(uu, uu.size)
        idx = np.concatenate([np.flatnonzero(ui == j) for j in pick])
        v = rho(pr[idx], me[idx])
        if v is not None:
            bs.append(v)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    q = float(np.quantile(shuf, 0.975))
    return {"n_pairs": len(pairs), "n_units": int(uu.size), "rho": r, "lo": float(lo), "hi": float(hi),
            "shuffle_q975": q, "slope_through_origin": float((pr * me).sum() / (pr * pr).sum()),
            "pass": bool(r >= 0.7 and lo > q), "above_ceiling": bool(r > 0.999), "failing_input_rho": rho(-pr, me)}


def boot(x, n=10000, seed=0):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n)])
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def u_s(rec):
    s0 = rec["pull"]["S"]
    return float(np.mean([c["S"] >= s0 for c in rec["comp"]]))


def sign_label(st, names):
    return names[0] if st["lo"] > 0 else (names[2] if st["hi"] < 0 else names[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--recount", required=True)
    ap.add_argument("--registered", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = pathlib.Path(a.repo)
    nb = repo / "studies/next/n12/banks/score-02/n12"
    th12 = json.loads((repo / "studies/next/n12/thresholds.json").read_text())
    fp = {(p["unit"], p["pole"]): p["dL"] for p in json.loads(pathlib.Path(a.recount).read_text())["pairs"]}
    out = {}
    recs = {}
    for h in ("freeze", "test"):
        rs = read_jsonl(nb / f"n12_{h}.jsonl")
        for r in rs:
            if r["kind"] == "real":
                r["abl_bf16"] = r["abl"]
                r["abl"] = [[x[0], x[1], fp[(r["unit"], x[0])]] for x in r["abl"]]
        recs[h] = rs
    fz_real = [r for r in recs["freeze"] if r["kind"] == "real" and "pull" in r]
    R = {r["birth"]: r for r in recs["test"] if r["kind"] == "real"}
    T = {r["birth"]: r for r in recs["test"] if r["kind"] == "pb"}
    D = {r["birth"]: r for r in recs["test"] if r["kind"] == "dt"}
    gr = [i for i in sorted(R) if "pull" in R[i]]
    out["G_VAL_fp32_on_N12_records"] = {"freeze_N12_file_order": gval(fz_real), "test_sorted_by_birth": gval([R[i] for i in gr])}
    # u_S recomputed from comp S against the records
    allr = [r for h in recs for r in recs[h] if "pull" in r]
    d_us = [abs(u_s(r) - r["pull"]["u_S"]) for r in allr]
    out["u_S_recount_vs_records"] = {"n": len(allr), "max_abs_diff": float(max(d_us)), "n_equal": int(sum(x == 0 for x in d_us))}
    us = [u_s(R[i]) for i in gr]
    st = boot(us)
    um = np.array([R[i]["pull"]["u_matched"] for i in gr], np.float64)
    rng = np.random.default_rng(0)
    nu, k = um.shape
    fl = np.array([um[np.arange(nu), rng.integers(0, k, nu)].mean() for _ in range(1000)])
    floor_test = float(np.quantile(fl, 0.025))
    bar = min(0.5, th12["floor_u_freeze"]["q025"], floor_test)
    ug = boot([R[i]["pull"]["u_gain"] for i in gr])
    lab = "SUPPORTED" if st["hi"] < bar else ("REFUTED" if st["mean"] >= bar else "UNDECIDED")
    if st["hi"] < ug["lo"]:
        lab = "ABOVE_CEILING"
    out["H_signal"] = {"label": lab, **st, "bar": bar, "floor_freeze_q025": th12["floor_u_freeze"]["q025"],
                       "floor_test_q025": floor_test, "u_gain": ug}
    sel = ("ABOVE_SELECTION", "AT_SELECTION", "BELOW_SELECTION")
    for name, X in (("pb", T), ("dt", D)):
        ids = [i for i in gr if i in X and "pull" in X[i]]
        d = np.array([u_s(X[i]) - u_s(R[i]) for i in ids])
        rng = np.random.default_rng(0)
        m = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(10000)])
        lo, hi = np.quantile(m, [0.025, 0.975])
        stp = {"n": int(d.size), "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
               "x_only": int((d > 0).sum()), "y_only": int((d < 0).sum())}
        out[f"H_sel_signal_{name}"] = {"label": "NOT_RUN" if d.size < 50 else sign_label(stp, sel), **stp}
    rs_ = []
    d_rho = []
    for i in gr:
        v = rho([c["S"] for c in R[i]["comp"]], R[i]["gain"])
        if R[i]["pull"]["rho_S_gain"] is not None and v is not None:
            d_rho.append(abs(v - R[i]["pull"]["rho_S_gain"]))
        if v is not None:
            rs_.append(v)
    sb = boot(rs_)
    out["H_bridge"] = {"label": sign_label(sb, ("FORETELLS", "UNDECIDED", "CONTRARY")), **sb,
                       "n_undefined": len(gr) - len(rs_), "rho_S_gain_recount_vs_records_max_abs": float(max(d_rho))}
    # the registered re-reading, for comparison
    v = json.loads(pathlib.Path(a.registered).read_text())["n12_reread"]
    cmp = {}
    for k in ("H_signal", "H_sel_signal_pb", "H_sel_signal_dt", "H_bridge"):
        mine, reg = out[k], v["labels"][k]
        cmp[k] = {"label": [mine["label"], reg["label"]],
                  **{f: [mine[f], reg[f], abs(mine[f] - reg[f])] for f in ("mean", "diff", "lo", "hi", "bar") if f in reg}}
    for h, key in (("freeze_N12_file_order", "validation_freeze"), ("test_sorted_by_birth", "validation_test")):
        mine, reg = out["G_VAL_fp32_on_N12_records"][h], v[key]
        cmp[key] = {f: [mine[f], reg[f], abs(mine[f] - reg[f])] for f in ("rho", "lo", "hi", "shuffle_q975", "slope_through_origin")}
    out["registered_reread_vs_mine"] = cmp
    out["registered_reread_headline"] = v["headline"]
    # N12's printed exploratory values (READ_n12.md section 4 and 5; N12_SPEC.md section 12), typed here from the text
    out["N12_printed"] = {"H_signal": "REFUTED, mean u_S 0.5268 [0.451, 0.600], bar 0.4225 (job's floor)",
                          "H_sel_signal_pb": "AT, paired u_pb - u_real -0.0386 (READ_n12 s4), -0.039 [-0.134, +0.057] (N12_SPEC s12)",
                          "H_sel_signal_dt": "AT, -0.0201 (READ_n12 s4), -0.020 [-0.124, +0.084] (N12_SPEC s12)",
                          "H_bridge": "UNDECIDED, rho(S, gain) -0.01259 [-0.0718, +0.0459] (job's seed, READ_n12 s4)"}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:5000])


if __name__ == "__main__":
    main()
