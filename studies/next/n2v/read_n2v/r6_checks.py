"""N2V reader, step 6: checks (a)-(h) on the recounted pairs (r1_recount.json) and the raw rows (losses/tNN.npz).

    python -I r6_checks.py --recount r1_recount.json --s3 <dir holding score-02/n2v> --repo <worktree> --out r6_checks.json

Readouts as in SPEC section 3.4: window (pred_bf16 vs dL), fd (0.1 pred_bf16 vs dL_fd), own (pred_jvp_own vs dL_own),
window_jvp (pred_jvp vs dL). Halves as registered (freeze, test). Own code, numpy only.
"""
import argparse
import collections
import json
import pathlib

import numpy as np

READ = {"window": ("pred_bf16_n12", "dL", 1.0), "fd": ("pred_bf16_n12", "dL_fd", 0.1),
        "own": ("pred_jvp_own", "dL_own", 1.0), "window_jvp": ("pred_jvp", "dL", 1.0)}


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


def pearson(x, y):
    return float(np.corrcoef(np.asarray(x, np.float64), np.asarray(y, np.float64))[0, 1])


def kendall4(p, m):
    c = d = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            s = np.sign(p[i] - p[j]) * np.sign(m[i] - m[j])
            c += s > 0
            d += s < 0
    return (c - d) / (len(p) * (len(p) - 1) / 2)


def boot_units(vals, n=10000, seed=0):
    v = np.asarray(vals, np.float64)
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)])
    return {"mean": float(v.mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)), "n": int(v.size)}


def arrays(ps, ro):
    pk, mk, sc = READ[ro]
    return np.array([sc * p[pk] for p in ps]), np.array([p[mk] for p in ps])


def by_unit(ps):
    u = collections.OrderedDict()
    for i, p in enumerate(ps):
        u.setdefault(p["unit"], []).append(i)
    return u


def check_a(ps, ro, rng):
    pr, me = arrays(ps, ro)
    U = by_unit(ps)
    tau_s, tau_a, rs_s, top_abs, top_s = [], [], [], [], []
    for u, ix in U.items():
        p, m = pr[ix], me[ix]
        tau_s.append(kendall4(p, m))
        tau_a.append(kendall4(np.abs(p), np.abs(m)))
        r = rho(p, m)
        rs_s.append(np.nan if r is None else r)
        top_abs.append(int(np.argmax(np.abs(p)) == np.argmax(np.abs(m))))
        top_s.append(int(np.argmax(p) == np.argmax(m)))
    cen_p, cen_m, wr_p, wr_m, sc_p, sc_m = (np.empty(len(ps)) for _ in range(6))
    for u, ix in U.items():
        cen_p[ix], cen_m[ix] = pr[ix] - pr[ix].mean(), me[ix] - me[ix].mean()
        wr_p[ix], wr_m[ix] = avg_rank(pr[ix]), avg_rank(me[ix])
        s = np.abs(pr[ix]).max()
        sc_p[ix], sc_m[ix] = pr[ix] / s, me[ix] / s
    # within-unit permutation of the measurement: keeps every unit's set of values, destroys the within-unit order
    perm_full, perm_cen, perm_top = [], [], []
    for _ in range(1000):
        mp = me.copy()
        for u, ix in U.items():
            mp[ix] = me[rng.permutation(ix)]
        perm_full.append(rho(pr, mp))
        c = mp.copy()
        for u, ix in U.items():
            c[ix] = mp[ix] - mp[ix].mean()
        perm_cen.append(rho(cen_p, c))
        perm_top.append(np.mean([np.argmax(np.abs(pr[ix])) == np.argmax(np.abs(mp[ix])) for ix in U.values()]))
    um = np.empty(len(ps))
    for u, ix in U.items():
        um[ix] = me[ix].mean()
    return {"n_units": len(U),
            "kendall_signed": {**boot_units(tau_s), "share_tau_1": float(np.mean(np.array(tau_s) == 1)),
                               "median": float(np.median(tau_s))},
            "kendall_abs": {**boot_units(tau_a), "share_tau_1": float(np.mean(np.array(tau_a) == 1)),
                            "median": float(np.median(tau_a))},
            "spearman4_signed_mean": float(np.nanmean(rs_s)),
            "top_abs_pole_match": {**boot_units(top_abs), "chance": 0.25,
                                   "within_unit_perm_q975": float(np.quantile(perm_top, 0.975))},
            "top_signed_pole_match": boot_units(top_s),
            "rho_pairs": rho(pr, me),
            "rho_per_unit_centered": rho(cen_p, cen_m),
            "rho_within_unit_ranks": rho(wr_p, wr_m),
            "rho_scaled_by_unit_max_abs_pred": rho(sc_p, sc_m),
            "rho_pred_vs_unit_mean_meas": rho(pr, um),
            "within_unit_perm": {"rho_pairs_median": float(np.median(perm_full)),
                                 "rho_pairs_q025_q975": [float(np.quantile(perm_full, 0.025)), float(np.quantile(perm_full, 0.975))],
                                 "rho_centered_median": float(np.median(perm_cen)),
                                 "rho_centered_q975": float(np.quantile(perm_cen, 0.975))}}


def check_b(ps):
    out = {}
    for thr in (1e-3, 1e-4):
        sel = [p for p in ps if abs(p["pred_bf16_n12"]) < thr]
        out[f"abs_pred_lt_{thr:g}"] = {
            "n": len(sel),
            "pairs": [{"unit": p["unit"], "pole": p["pole"], "half": p["half"], "rank": p["rank"],
                       "pred_bf16": p["pred_bf16_n12"], "pred_jvp": p["pred_jvp"], "dL": p["dL"], "dL_fd": p["dL_fd"],
                       "probe": p["probe"], "abs_dL_over_abs_probe": abs(p["dL"]) / abs(p["probe"]) if p["probe"] else None,
                       "meas_bf16": p["meas_bf16_n12"]} for p in sorted(sel, key=lambda q: abs(q["pred_bf16_n12"]))][:40],
            "median_abs_dL": float(np.median([abs(p["dL"]) for p in sel])) if sel else None,
            "median_abs_probe": float(np.median([abs(p["probe"]) for p in sel])) if sel else None,
            "median_abs_dL_minus_pred_jvp": float(np.median([abs(p["dL"] - p["pred_jvp"]) for p in sel])) if sel else None,
            "sign_agreement_dL_pred_bf16": float(np.mean([np.sign(p["dL"]) == np.sign(p["pred_bf16_n12"]) for p in sel])) if sel else None,
            "sign_agreement_dL_pred_jvp": float(np.mean([np.sign(p["dL"]) == np.sign(p["pred_jvp"]) for p in sel])) if sel else None,
            "share_abs_dL_gt_10x_abs_probe": float(np.mean([abs(p["dL"]) > 10 * abs(p["probe"]) for p in sel])) if sel else None}
    allp = np.array([abs(p["probe"]) for p in ps if p["rank"] == 0])
    out["probe_all_units"] = {"median_abs": float(np.median(allp)), "max_abs": float(allp.max()), "n": int(allp.size)}
    own = np.array([abs(p["probe_own"]) for p in ps if p["rank"] == 0])
    out["probe_own_all_units"] = {"median_abs": float(np.median(own)), "max_abs": float(own.max())}
    res = np.array([abs(p["dL"] - p["pred_jvp"]) for p in ps])
    pu = np.array([abs(p["probe"]) for p in ps])
    out["residual_vs_probe"] = {"median_abs_dL_minus_pred_jvp": float(np.median(res)), "median_abs_probe": float(np.median(pu)),
                                "share_residual_gt_10x_probe": float(np.mean(res > 10 * pu))}
    return out


def deciles(ps, key_x, f):
    x = np.array([abs(p[key_x]) for p in ps])
    edges = np.quantile(x, np.linspace(0, 1, 11))
    out = []
    for k in range(10):
        sel = [p for p, v in zip(ps, x) if (edges[k] <= v <= edges[k + 1] if k == 9 else edges[k] <= v < edges[k + 1])]
        vals = np.array([f(p) for p in sel])
        out.append({"abs_pred_range": [float(edges[k]), float(edges[k + 1])], "n": len(sel),
                    "q10": float(np.quantile(vals, 0.1)), "median": float(np.median(vals)), "q90": float(np.quantile(vals, 0.9))})
    return out


def check_c(ps):
    out = {}
    out["dL_fd_over_0.1_dL_by_abs_pred_decile"] = deciles(ps, "pred_bf16_n12", lambda p: p["dL_fd"] / (0.1 * p["dL"]) if p["dL"] else np.nan)
    out["dL_over_pred_bf16_by_abs_pred_decile"] = deciles(ps, "pred_bf16_n12", lambda p: p["dL"] / p["pred_bf16_n12"])
    out["dL_over_pred_jvp_by_abs_pred_decile"] = deciles(ps, "pred_jvp", lambda p: p["dL"] / p["pred_jvp"])
    out["dL_fd_over_0.1pred_jvp_by_abs_pred_decile"] = deciles(ps, "pred_jvp", lambda p: p["dL_fd"] / (0.1 * p["pred_jvp"]))
    pr = np.array([p["pred_bf16_n12"] for p in ps])
    pj = np.array([p["pred_jvp"] for p in ps])
    me = np.array([p["dL"] for p in ps])
    fd = np.array([p["dL_fd"] for p in ps])
    contrib = pr * me
    o = np.argsort(-pr * pr)
    out["slope_window"] = float(contrib.sum() / (pr * pr).sum())
    out["slope_window_jvp"] = float((pj * me).sum() / (pj * pj).sum())
    out["slope_fd_over_0.1"] = float((0.1 * pr * fd).sum() / ((0.1 * pr) ** 2).sum())
    out["share_of_sum_pred_sq_top5"] = float((pr[o[:5]] ** 2).sum() / (pr * pr).sum())
    out["slope_window_without_top_k"] = {k: float(contrib[o[k:]].sum() / (pr[o[k:]] ** 2).sum()) for k in (1, 3, 5, 10)}
    out["top10_by_abs_pred"] = [{"unit": ps[i]["unit"], "pole": ps[i]["pole"], "rank": ps[i]["rank"], "t": ps[i]["t"],
                                 "pred_bf16": float(pr[i]), "pred_jvp": float(pj[i]), "dL": float(me[i]),
                                 "dL_fd": float(fd[i]), "dL_over_pred_bf16": float(me[i] / pr[i]),
                                 "dL_fd_over_0.1pred_bf16": float(fd[i] / (0.1 * pr[i]))} for i in o[:10]]
    # first and second order from the two rows: dL = a + b, dL_fd = 0.1 a + 0.01 b (a quadratic through 0)
    a = (fd - 0.01 * me) / 0.09
    out["two_point_first_order_a_vs_pred_jvp"] = {"spearman": rho(a, pj), "slope": float((pj * a).sum() / (pj * pj).sum()),
                                                  "median_abs_rel": float(np.median(np.abs(a - pj) / np.abs(pj)))}
    out["two_point_first_order_a_vs_pred_bf16"] = {"spearman": rho(a, pr), "slope": float((pr * a).sum() / (pr * pr).sum()),
                                                   "median_abs_rel": float(np.median(np.abs(a - pr) / np.abs(pr)))}
    b = me - a
    out["second_order_share_median_abs_b_over_abs_dL"] = float(np.median(np.abs(b) / np.abs(me)))
    return out


def check_d(ps):
    out = {}
    for ro in READ:
        pr, me = arrays(ps, ro)
        U = by_unit(ps)
        loo = []
        for u, ix in U.items():
            keep = np.setdiff1d(np.arange(len(ps)), ix)
            loo.append((rho(pr[keep], me[keep]), u))
        lmin, lmax = min(loo), max(loo)
        d = {"rho": rho(pr, me), "pearson": pearson(pr, me),
             "pearson_log_abs": pearson(np.log10(np.abs(pr)), np.log10(np.abs(me) + 1e-12)),
             "leave_one_unit_out": {"min": lmin[0], "min_unit": lmin[1], "max": lmax[0], "max_unit": lmax[1]}}
        for name, sel in (("carriers_rank0", [i for i, p in enumerate(ps) if p["rank"] == 0]),
                          ("competitors_rank1_3", [i for i, p in enumerate(ps) if p["rank"] > 0])):
            d[name] = {"n": len(sel), "rho": rho(pr[sel], me[sel]),
                       "sign_agreement": float(np.mean(np.sign(pr[sel]) == np.sign(me[sel])))}
        bt = {}
        for t in sorted({p["t"] for p in ps}):
            sel = [i for i, p in enumerate(ps) if p["t"] == t]
            bt[t] = {"n_pairs": len(sel), "n_units": len({ps[i]["unit"] for i in sel}), "rho": rho(pr[sel], me[sel])}
        d["by_point"] = bt
        bl = {}
        for g in ((0, 7), (8, 15), (16, 23), (24, 31)):
            sel = [i for i, p in enumerate(ps) if g[0] <= p["layer"] <= g[1]]
            bl[f"L{g[0]:02d}-{g[1]:02d}"] = {"n_pairs": len(sel), "n_units": len({ps[i]["unit"] for i in sel}),
                                             "rho": rho(pr[sel], me[sel]) if len(sel) >= 3 else None}
        d["by_layer_group"] = bl
        out[ro] = d
    return out


def check_e(ps):
    a = np.array([p["pred_bf16_n12"] for p in ps])
    b = np.array([p["pred_jvp"] for p in ps])
    m = np.array([p["dL"] for p in ps])
    rel = np.abs(a - b) / np.abs(b)
    o = np.argsort(-rel)
    e16, ej = np.abs(m - a), np.abs(m - b)
    fd = np.array([p["dL_fd"] for p in ps])
    top = o[:len(ps) // 10]
    f16, fj = np.abs(fd - 0.1 * a), np.abs(fd - 0.1 * b)
    extra = {"fd_rho_with_0.1pred_bf16": rho(0.1 * a, fd), "fd_rho_with_0.1pred_jvp": rho(0.1 * b, fd),
             "share_jvp_closer_to_dL_fd_all": float(np.mean(fj < f16)),
             "share_jvp_closer_to_dL_fd_top10pct_disagree": float(np.mean((fj < f16)[top])),
             "share_jvp_closer_to_dL_top10pct_disagree": float(np.mean((ej < e16)[top])),
             "n_top10pct": int(top.size)}
    return {**extra, "spearman_bf16_jvp": rho(a, b), "median_rel_diff": float(np.median(rel)),
            "share_rel_diff_gt_0.1": float(np.mean(rel > 0.1)), "share_rel_diff_gt_0.5": float(np.mean(rel > 0.5)),
            "share_jvp_closer_to_dL": float(np.mean(ej < e16)),
            "spearman_relbf16jvp_vs_rel_err_bf16": rho(rel, e16 / np.abs(m)),
            "spearman_absdiff_vs_abs_err_bf16": rho(np.abs(a - b), e16),
            "median_abs_err_bf16_top10pct_disagree": float(np.median((e16 / np.abs(m))[o[:len(ps) // 10]])),
            "median_abs_err_bf16_rest": float(np.median((e16 / np.abs(m))[o[len(ps) // 10:]])),
            "top10": [{"unit": ps[i]["unit"], "pole": ps[i]["pole"], "rank": ps[i]["rank"], "pred_bf16": float(a[i]),
                       "pred_jvp": float(b[i]), "dL": float(m[i]), "rel_diff_bf16_jvp": float(rel[i]),
                       "abs_err_bf16": float(e16[i]), "abs_err_jvp": float(ej[i])} for i in o[:10]]}


def rho_boot_units(ps, ro, n=2000, seed=0):
    pr, me = arrays(ps, ro)
    U = list(by_unit(ps).values())
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(U), len(U))
        ix = np.concatenate([U[k] for k in pick])
        out.append(rho(pr[ix], me[ix]))
    return np.array(out)


def check_f(H):
    out = {}
    for ro in READ:
        bf, bt = rho_boot_units(H["freeze"], ro, seed=0), rho_boot_units(H["test"], ro, seed=1)
        d = bf - bt
        pf, mf = arrays(H["freeze"], ro)
        pt, mt = arrays(H["test"], ro)
        out[ro] = {"rho_freeze": rho(pf, mf), "rho_test": rho(pt, mt), "diff_freeze_minus_test": rho(pf, mf) - rho(pt, mt),
                   "diff_interval_2000_boot": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                   "slope_freeze": float((pf * mf).sum() / (pf * pf).sum()), "slope_test": float((pt * mt).sum() / (pt * pt).sum())}
    return out


def check_g(H):
    out = {}
    for h, ps in H.items():
        pr = np.array([p["pred_bf16_n12"] for p in ps])
        pj = np.array([p["pred_jvp"] for p in ps])
        pjo = np.array([p["pred_jvp_own"] for p in ps])
        m16 = np.array([p["meas_bf16_n12"] for p in ps])
        rng = np.random.default_rng(0)
        out[h] = {"window_with_N12_bf16_measurement_rho": rho(pr, m16),
                  "window_with_shuffled_dL_rho_median": float(np.median([rho(pr, rng.permutation(np.array([p["dL"] for p in ps]))) for _ in range(200)])),
                  "window_if_dL_were_pred_jvp_exactly_rho": rho(pr, pj),
                  "window_jvp_if_dL_were_pred_jvp_exactly_rho": rho(pj, pj),
                  "own_if_dL_own_were_pred_jvp_own_exactly_rho": rho(pjo, pjo),
                  "fd_if_dL_fd_were_0.1pred_exactly_max_rel_dev": 0.0}
    out["note"] = ("A measurement equal to the fp32 forward-mode prediction would give the window readout rho = Spearman(pred_bf16, "
                   "pred_jvp), below 0.999, so N12's ceiling could not fire there even on a copy of the fp32 prediction; on own and "
                   "window_jvp a copy gives 1.0 > 0.999 and the inspection (n_equal > 0) would not clear it.")
    return out


def check_h_npz(s3, H, units):
    """Rows equal to their clean row at every position, and zero rows equal to the clean row through i0 (G-CAUSAL's
    failing input; metrics.json counts them per point as n_equal)."""
    plan = {}
    for ps in H.values():
        for p in ps:
            plan.setdefault(p["t"], set()).add(p["unit"])
    eq_all = collections.Counter()
    eq_all_list = []
    zero_eq_through_i0 = {}
    for t, us in sorted(plan.items()):
        z = np.load(s3 / "losses" / f"t{t:02d}.npz")
        n = 0
        for k in z.keys():
            u, w, kind, q = k.split("|")
            if kind in ("clean", "jvp"):
                continue
            w = int(w)
            i0 = min(1 + e % 2048 for e in units[u]["tok"] if e // 2048 == w and 1 + e % 2048 <= 2047)
            lc, lr = z[f"{u}|{w}|clean|None"], z[k]
            if kind == "zero" and np.array_equal(lc[:i0 + 1], lr[:i0 + 1]):
                n += 1
            if np.array_equal(lc, lr):
                eq_all[f"{kind}|{q}"] += 1
                eq_all_list.append({"t": t, "unit": u, "window": w, "kind": kind, "q": int(q)})
        zero_eq_through_i0[t] = n
    return {"rows_equal_to_clean_everywhere": dict(eq_all), "list": eq_all_list,
            "zero_rows_equal_through_i0_by_point": zero_eq_through_i0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recount", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rc = json.loads(pathlib.Path(a.recount).read_text())
    H = {}
    for h in ("freeze", "test"):
        ps = sorted([p for p in rc["pairs"] if p["half"] == h], key=lambda p: (p["unit"], p["rank"]))
        for p in ps:
            p["layer"] = int(p["pole"][1:].split("N")[0])
        H[h] = ps
    out = {"a_within_unit": {}, "b_smallest_predictions": {}, "c_linearity": {}, "d_leverage": {}, "e_bf16_vs_jvp": {}}
    for h, ps in H.items():
        rng = np.random.default_rng(0)
        out["a_within_unit"][h] = {ro: check_a(ps, ro, rng) for ro in READ}
        out["b_smallest_predictions"][h] = check_b(ps)
        out["c_linearity"][h] = check_c(ps)
        out["d_leverage"][h] = check_d(ps)
        out["e_bf16_vs_jvp"][h] = check_e(ps)
    out["f_freeze_vs_test"] = check_f(H)
    out["g_room"] = check_g(H)
    s3 = pathlib.Path(a.s3) / "score-02" / "n2v"
    units = {u["unit"]: u for u in json.loads((pathlib.Path(a.repo) / "studies/next/n12/units.json").read_text())["units"]}
    out["h_rows_equal_everywhere"] = check_h_npz(s3, H, units)
    # dL exactly 0
    out["h_pairs_with_dL_exactly_0"] = [{"unit": p["unit"], "pole": p["pole"], "half": p["half"], "dL": p["dL"],
                                          "pred_bf16": p["pred_bf16_n12"]} for ps in H.values() for p in ps if p["dL"] == 0]
    # the shared (window, layer) at one point: the job's h capture keyed by (window, layer) keeps the last unit's indices
    sh = []
    for it in rc["shared_window_layer_at_point"]["items"]:
        ix = {}
        for u in it["units"]:
            ix[u] = sorted(1 + e % 2048 for e in units[u]["tok"] if e // 2048 == it["window"] and 1 + e % 2048 <= 2047)
        sh.append({**it, "indices": ix, "same_indices": len({tuple(v) for v in ix.values()}) == 1})
    out["h_shared_window_layer_indices"] = sh
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("written", a.out)


if __name__ == "__main__":
    main()
