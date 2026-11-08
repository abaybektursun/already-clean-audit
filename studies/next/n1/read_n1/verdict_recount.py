"""N1 independent read, step 3 and step 5: the verdict quantities and the confound checks, from this read's own
recount (units_recount.json, written by recount.py from stats.npz), with code written for this read (numpy and scipy
only; nothing imported from infra.pg). The section 5 label rules of N1_SPEC.md are applied here by hand.

    python3 -I verdict_recount.py --units-recount studies/next/n1/read_n1/units_recount.json \
        --thresholds studies/next/n1/thresholds.json --n12-ranks-test studies/next/n12/banks/score-02/n12/ranks_test.jsonl \
        --n12-ranks-freeze studies/next/n12/banks/score-02/n12/ranks_freeze.jsonl --onsets studies/next/n1/onsets/onsets.json \
        --out studies/next/n1/read_n1/verdict_recount.json
"""
import argparse
import collections
import json
import pathlib

import numpy as np
from scipy import stats as st

B = 10000


def boot_loop(x, seed, n_boot=B):
    """The registered procedure, rewritten: one rng.integers(0, n, n) draw per resample, mean, linear quantiles."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = np.empty(n_boot)
    for i in range(n_boot):
        m[i] = x[rng.integers(0, x.size, x.size)].mean()
    return m


def pct(m):
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)


def bca(x, m):
    x = np.asarray(x, np.float64)
    th = x.mean()
    z0 = st.norm.ppf(np.mean(m < th))
    jk = np.array([np.delete(x, i).mean() for i in range(x.size)])
    d = jk.mean() - jk
    acc = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)
    out = []
    for al in (0.025, 0.975):
        za = st.norm.ppf(al)
        out.append(float(np.quantile(m, st.norm.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za))))))
    return out[0], out[1], float(z0), float(acc)


def methods(x):
    x = np.asarray(x, np.float64)
    n, mu, sd = x.size, float(x.mean()), float(x.std(ddof=1))
    res = {"n": n, "mean": mu, "sd": sd, "se": sd / np.sqrt(n)}
    m0 = boot_loop(x, 0)
    res["pct_seed0"] = pct(m0)
    res["pct_seed1"] = pct(boot_loop(x, 1))
    res["pct_seed2"] = pct(boot_loop(x, 2))
    rng = np.random.default_rng(0)
    mv = x[rng.integers(0, n, (100000, n))].mean(1)
    res["pct_vectorized_100k_seed0"] = pct(mv)
    lo, hi = res["pct_seed0"]
    res["basic_seed0"] = (2 * mu - hi, 2 * mu - lo)
    b = bca(x, m0)
    res["bca_seed0"], res["bca_z0_acc"] = (b[0], b[1]), (b[2], b[3])
    t = st.t.ppf(0.975, n - 1)
    res["student_t"] = (mu - t * sd / np.sqrt(n), mu + t * sd / np.sqrt(n))
    res["n_pos"], res["n_neg"], res["n_zero"] = int((x > 0).sum()), int((x < 0).sum()), int((x == 0).sum())
    res["sign_test_p"] = float(st.binomtest(res["n_pos"], res["n_pos"] + res["n_neg"], 0.5).pvalue)
    res["wilcoxon_p"] = float(st.wilcoxon(x).pvalue)
    res["median"] = float(np.median(x))
    mm = np.median(x[rng.integers(0, n, (10000, n))], axis=1)
    res["median_pct_boot"] = pct(mm)
    tr = np.sort(x)[int(0.1 * n): n - int(0.1 * n)]
    res["trimmed10_mean"] = float(tr.mean())
    # leave-one-out: the largest upper bound when any single birth is dropped (registered percentile, seed 0)
    his = []
    for i in range(n):
        his.append(pct(boot_loop(np.delete(x, i), 0, 2000))[1])
    res["drop_one_hi_max_2000boot"] = float(max(his))
    res["drop_one_argmax"] = int(np.argmax(his))
    return res


def label(lo, hi):
    return "ABOVE_SELECTION" if lo > 0 else ("BELOW_SELECTION" if hi < 0 else "AT_SELECTION")


def main():
    ap = argparse.ArgumentParser()
    for a in ("units-recount", "thresholds", "n12-ranks-test", "n12-ranks-freeze", "onsets", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    U = json.loads(pathlib.Path(a.units_recount).read_text())
    th = json.loads(pathlib.Path(a.thresholds).read_text())
    out = {}
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        real = [u for u in H if u["kind"] == "real"]                 # units.json order (the records' order)
        tw = {kind: {u["birth"]: u for u in H if u["kind"] == kind} for kind in ("dt", "pb")}
        rb = {u["birth"]: u for u in real}
        o = {"n_real": len(real), "n_dt": len(tw["dt"]), "n_pb": len(tw["pb"])}
        for kind in ("dt", "pb"):
            ids = sorted(b for b in rb if b in tw[kind])
            o[f"n_pairs_{kind}"] = len(ids)
            o[f"pairs_same_k_{kind}"] = all(rb[b]["k"] == tw[kind][b]["k"] for b in ids)
            o[f"pairs_same_layer_{kind}"] = all(rb[b]["layer"] == tw[kind][b]["layer"] for b in ids)
            o[f"pairs_same_neuron_{kind}"] = sum(rb[b]["neuron"] == tw[kind][b]["neuron"] for b in ids)
            for s in ("in", "out", "gate", "up"):
                d = np.array([rb[b][s]["e_window"] - tw[kind][b][s]["e_window"] for b in ids])
                if s in ("in", "out") or kind == "dt":
                    o[f"pair_{s}_real_{kind}"] = methods(d) if (half == "test" and kind == "dt" and s in ("in", "out")) else \
                        {"n": len(d), "mean": float(d.mean()), "pct_seed0": pct(boot_loop(d, 0)),
                         "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}
                dk = np.array([rb[b][s]["e_k"] - tw[kind][b][s]["e_k"] for b in ids])
                o[f"pair_k_{s}_real_{kind}"] = {"n": len(dk), "mean": float(dk.mean()), "pct_seed0": pct(boot_loop(dk, 0)),
                                                "n_pos": int((dk > 0).sum()), "n_neg": int((dk < 0).sum())}
            # decomposition of the paired difference into the window part and the reference part
            if kind == "dt":
                for s in ("in", "out"):
                    wr = np.array([rb[b][s]["e_window"] + rb[b][s]["ref_mean"] for b in ids])
                    wt = np.array([tw[kind][b][s]["e_window"] + tw[kind][b][s]["ref_mean"] for b in ids])
                    rr = np.array([rb[b][s]["ref_mean"] for b in ids])
                    rt = np.array([tw[kind][b][s]["ref_mean"] for b in ids])
                    o[f"decomp_{s}"] = {"window_mean_z_real_minus_dt": {"mean": float((wr - wt).mean()), "pct_seed0": pct(boot_loop(wr - wt, 0))},
                                        "reference_mean_z_real_minus_dt": {"mean": float((rr - rt).mean()), "pct_seed0": pct(boot_loop(rr - rt, 0))},
                                        "window_mean_z_real": float(wr.mean()), "window_mean_z_dt": float(wt.mean()),
                                        "reference_mean_z_real": float(rr.mean()), "reference_mean_z_dt": float(rt.mean())}
                    # paired profile real - dt per bracket offset
                    o[f"paired_profile_{s}"] = {}
                    for dd in (-2, -1, 0, 1):
                        v = np.array([rb[b][s]["e_by_bracket"][str(rb[b]["k"] + dd)] - tw[kind][b][s]["e_by_bracket"][str(rb[b]["k"] + dd)]
                                      for b in ids if str(rb[b]["k"] + dd) in rb[b][s]["e_by_bracket"]])
                        o[f"paired_profile_{s}"][str(dd)] = {"n": len(v), "mean": float(v.mean()), "pct_seed0": pct(boot_loop(v, 0))}
        for s in ("in", "out", "gate", "up"):
            x = np.array([u[s]["e_window"] for u in real])
            o[f"real_{s}"] = {"n": len(x), "mean": float(x.mean()), "pct_seed0": pct(boot_loop(x, 0))}
            for kind in ("dt", "pb"):
                y = np.array([u[s]["e_window"] for u in tw[kind].values()])
                o[f"{kind}_{s}"] = {"n": len(y), "mean": float(y.mean()), "pct_seed0": pct(boot_loop(y, 0))}
            ek = np.array([u[s]["e_k"] for u in real])
            o[f"real_e_k_{s}"] = {"n": len(ek), "mean": float(ek.mean()), "pct_seed0": pct(boot_loop(ek, 0))}
            o[f"profile_{s}"] = {}
            for dd in (-2, -1, 0, 1):
                v = np.array([u[s]["e_by_bracket"][str(u["k"] + dd)] for u in real if str(u["k"] + dd) in u[s]["e_by_bracket"]])
                o[f"profile_{s}"][str(dd)] = {"n": len(v), "mean": float(v.mean()), "pct_seed0": pct(boot_loop(v, 0))}
        for s in ("in", "out"):
            K = min(len(u["matched_e_window"][s]) for u in real)
            me = np.array([u["matched_e_window"][s][:K] for u in real])
            rng = np.random.default_rng(0)
            reps = np.array([me[np.arange(len(real)), rng.integers(0, K, len(real))].mean() for _ in range(1000)])
            o[f"floor_{s}"] = {"K": K, "mean": float(reps.mean()), "q025": float(np.quantile(reps, 0.025)),
                               "q975": float(np.quantile(reps, 0.975))}
            o[f"ceiling_{s}"] = float(np.median([u["ceiling"][s] for u in real]))
        out[half] = o

    # section 5 labels by hand, test half
    T = out["test"]
    lab = {}
    for s in ("in", "out"):
        p = T[f"pair_{s}_real_dt"]
        lo, hi = p["pct_seed0"]
        sel = "NOT_RUN" if p["n"] < 50 else label(lo, hi)
        bar = max(th[f"floor_{s}_freeze"]["q975"], T[f"floor_{s}"]["q975"])
        rl = T[f"real_{s}"]["pct_seed0"][0]
        fl = "ABOVE_FLOOR" if rl > bar else "NOT_ABOVE_FLOOR"
        if rl > T[f"ceiling_{s}"]:
            sel = "ABOVE_CEILING"
        lab[s] = {"sel": sel, "floor": fl, "bar": bar, "real_lo": rl, "ceiling": T[f"ceiling_{s}"],
                  "labels_by_method": {m: label(*p[m]) for m in ("pct_seed0", "pct_seed1", "pct_seed2",
                                                                  "pct_vectorized_100k_seed0", "basic_seed0", "bca_seed0", "student_t")}}
    a_in, a_out = lab["in"]["sel"] == "ABOVE_SELECTION", lab["out"]["sel"] == "ABOVE_SELECTION"
    hyp = "TOGETHER" if a_in and a_out else "ROWS_FIRST" if a_in else "COLUMN_FIRST" if a_out else \
        ("NEITHER (H_0: both at selection)" if lab["in"]["sel"] == lab["out"]["sel"] == "AT_SELECTION" else "NEITHER")
    reasons = []
    if T["n_real"] < 50:
        reasons.append("fewer than 50 real")
    if any(lab[s]["sel"] in ("NOT_RUN", "ABOVE_CEILING") for s in ("in", "out")):
        reasons.append("NOT_RUN or ABOVE_CEILING")
    out["labels"] = lab
    out["hypothesis"] = hyp
    out["headline"] = ("DOES NOT COUNT: " + "; ".join(reasons)) if reasons else \
        f"{hyp}: in {lab['in']['sel']} ({lab['in']['floor']}), out {lab['out']['sel']} ({lab['out']['floor']})"

    # ---------------------------------------------------------------- confounds
    C = {}
    # (a) e against the birth bracket k, within real and within dt; the paired difference by k
    for half in ("test", "freeze", "both"):
        H = [u for u in U.values() if half == "both" or u["half"] == half]
        for kind in ("real", "dt"):
            for s in ("in", "out"):
                ks = np.array([u["k"] for u in H if u["kind"] == kind], float)
                ev = np.array([u[s]["e_window"] for u in H if u["kind"] == kind])
                lr = st.linregress(ks, ev)
                C[f"a_slope_{s}_{kind}_{half}"] = {"n": len(ks), "slope_per_k": float(lr.slope), "se": float(lr.stderr),
                                                   "p": float(lr.pvalue), "spearman": float(st.spearmanr(ks, ev).statistic)}
                C[f"a_bins_{s}_{kind}_{half}"] = {f"k{lo_}-{hi_}": {"n": int(((ks >= lo_) & (ks <= hi_)).sum()),
                                                                     "mean": float(ev[(ks >= lo_) & (ks <= hi_)].mean())}
                                                  for lo_, hi_ in ((6, 11), (12, 12), (13, 13), (14, 14), (15, 15))
                                                  if ((ks >= lo_) & (ks <= hi_)).sum()}
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        rb = {u["birth"]: u for u in H if u["kind"] == "real"}
        td = {u["birth"]: u for u in H if u["kind"] == "dt"}
        ids = sorted(b for b in rb if b in td)
        ks = np.array([rb[b]["k"] for b in ids], float)
        for s in ("in", "out"):
            d = np.array([rb[b][s]["e_window"] - td[b][s]["e_window"] for b in ids])
            lr = st.linregress(ks, d)
            C[f"a_pairdiff_vs_k_{s}_{half}"] = {"slope": float(lr.slope), "se": float(lr.stderr), "p": float(lr.pvalue),
                                               "bins": {f"k{lo_}-{hi_}": {"n": int(((ks >= lo_) & (ks <= hi_)).sum()),
                                                                          "mean": float(d[(ks >= lo_) & (ks <= hi_)].mean())}
                                                        for lo_, hi_ in ((6, 11), (12, 12), (13, 13), (14, 15))}}
        C[f"a_k_counts_{half}"] = dict(sorted(collections.Counter(int(x) for x in ks).items()))
    # (b) layer and activation size: carriers against their dt twins
    rk = {}
    for p in (a.n12_ranks_test, a.n12_ranks_freeze):
        for line in pathlib.Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["role"] == "pre":
                rk[r["unit"]] = r
    on = {r["unit"]: r for r in json.loads(pathlib.Path(a.onsets).read_text())["units"]}
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        rb = {u["birth"]: u for u in H if u["kind"] == "real"}
        td = {u["birth"]: u for u in H if u["kind"] == "dt"}
        ids = sorted(b for b in rb if b in td)
        row = {"same_layer": sum(rb[b]["layer"] == td[b]["layer"] for b in ids), "n": len(ids),
               "same_k": sum(rb[b]["k"] == td[b]["k"] for b in ids)}
        for nm, f in (("maxabs_pre_log", lambda u: np.log(rk[u["unit"]]["value"]["maxabs"])),
                      ("ratio_pre", lambda u: rk[u["unit"]]["value"]["ratio"]),
                      ("r_pre_onsets", lambda u: on[u["unit"]]["r"]["pre"]),
                      ("r_birth_onsets", lambda u: on[u["unit"]]["r"]["birth"]),
                      ("rise_k_onsets", lambda u: on[u["unit"]]["rise_k"]),
                      ("log_norm_in_prev_k", lambda u: np.log(u["norm_prev_k"]["in"])),
                      ("log_norm_out_prev_k", lambda u: np.log(u["norm_prev_k"]["out"])),
                      ("ref_mean_z_in", lambda u: u["in"]["ref_mean"]), ("ref_mean_z_out", lambda u: u["out"]["ref_mean"])):
            xr = np.array([f(rb[b]) for b in ids])
            xt = np.array([f(td[b]) for b in ids])
            row[nm] = {"real_median": float(np.median(xr)), "dt_median": float(np.median(xt)),
                       "paired_mean_diff": float((xr - xt).mean()), "paired_pct_seed0": pct(boot_loop(xr - xt, 0)),
                       "wilcoxon_p": float(st.wilcoxon(xr - xt).pvalue) if np.any(xr != xt) else None,
                       "ks_p": float(st.ks_2samp(xr, xt).pvalue)}
        # does the activation-size difference explain the paired e_in difference?
        dmx = np.array([np.log(rk[rb[b]["unit"]]["value"]["maxabs"]) - np.log(rk[td[b]["unit"]]["value"]["maxabs"]) for b in ids])
        drp = np.array([on[rb[b]["unit"]]["r"]["pre"] - on[td[b]["unit"]]["r"]["pre"] for b in ids])
        for s in ("in", "out"):
            d = np.array([rb[b][s]["e_window"] - td[b][s]["e_window"] for b in ids])
            X = np.c_[np.ones(len(ids)), dmx, drp]
            beta, *_ = np.linalg.lstsq(X, d, rcond=None)
            row[f"regress_pairdiff_{s}_on_dlogmaxabs_drpre"] = {"intercept": float(beta[0]), "b_dlogmaxabs": float(beta[1]),
                                                                "b_drpre": float(beta[2]),
                                                                "corr_dlogmaxabs": float(np.corrcoef(dmx, d)[0, 1]),
                                                                "corr_drpre": float(np.corrcoef(drp, d)[0, 1])}
        C[f"b_{half}"] = row
    # (c) the two halves: the test label's probability under the freeze-half estimate; the halves' difference
    cc = {}
    for s in ("in", "out"):
        F, Tt = out["freeze"][f"pair_{s}_real_dt"], out["test"][f"pair_{s}_real_dt"]
        Hf = [u for u in U.values() if u["half"] == "freeze"]
        rbf = {u["birth"]: u for u in Hf if u["kind"] == "real"}
        tdf = {u["birth"]: u for u in Hf if u["kind"] == "dt"}
        df = np.array([rbf[b][s]["e_window"] - tdf[b][s]["e_window"] for b in sorted(b for b in rbf if b in tdf)])
        Ht = [u for u in U.values() if u["half"] == "test"]
        rbt = {u["birth"]: u for u in Ht if u["kind"] == "real"}
        tdt = {u["birth"]: u for u in Ht if u["kind"] == "dt"}
        dtv = np.array([rbt[b][s]["e_window"] - tdt[b][s]["e_window"] for b in sorted(b for b in rbt if b in tdt)])
        se_f, se_t = df.std(ddof=1) / np.sqrt(df.size), dtv.std(ddof=1) / np.sqrt(dtv.size)
        mu_f = df.mean()
        z = (dtv.mean() - mu_f) / np.sqrt(se_f ** 2 + se_t ** 2)
        # P(BELOW at the test half's n) if the true mean were the freeze estimate; normal, with the test SE
        p_below_norm = float(st.norm.cdf((-1.96 * se_t - mu_f) / se_t))
        # the same by resampling: draw the test half's n differences from the freeze half's, label by its percentile
        rng = np.random.default_rng(0)
        draws = df[rng.integers(0, df.size, (20000, dtv.size))]
        mu_d = draws.mean(1)
        se_d = draws.std(1, ddof=1) / np.sqrt(dtv.size)
        p_below_rs = float(np.mean(mu_d + 1.96 * se_d < 0))
        p_above_rs = float(np.mean(mu_d - 1.96 * se_d > 0))
        pooled = np.r_[df, dtv]
        cc[s] = {"freeze_mean": float(mu_f), "freeze_se": float(se_f), "test_mean": float(dtv.mean()), "test_se": float(se_t),
                 "test_minus_freeze": float(dtv.mean() - mu_f), "z": float(z), "p_two_sided": float(2 * st.norm.sf(abs(z))),
                 "welch_t_p": float(st.ttest_ind(dtv, df, equal_var=False).pvalue),
                 "p_test_BELOW_if_true_mean_is_freeze_estimate_normal": p_below_norm,
                 "p_test_BELOW_resampling_freeze_diffs": p_below_rs, "p_test_ABOVE_resampling_freeze_diffs": p_above_rs,
                 "pooled_both_halves": {"n": int(pooled.size), "mean": float(pooled.mean()), "pct_seed0": pct(boot_loop(pooled, 0)),
                                        "label": label(*pct(boot_loop(pooled, 0)))}}
    C["c_halves"] = cc
    # (d) room: the paired mean needed for ABOVE or BELOW at the test SE; real mean against floor bar and ceiling
    C["d_room"] = {s: {"test_se": T[f"pair_{s}_real_dt"]["se"],
                       "min_mean_for_ABOVE_approx": 1.96 * T[f"pair_{s}_real_dt"]["se"],
                       "real_mean": T[f"real_{s}"]["mean"], "real_pct": T[f"real_{s}"]["pct_seed0"],
                       "floor_bar": lab[s]["bar"], "ceiling": T[f"ceiling_{s}"],
                       "gap_real_lo_to_bar": lab[s]["real_lo"] - lab[s]["bar"],
                       "gap_ceiling_to_real_lo": T[f"ceiling_{s}"] - lab[s]["real_lo"],
                       "max_unit_e_real": max(u[s]["e_window"] for u in U.values() if u["half"] == "test" and u["kind"] == "real"),
                       "share_real_units_above_ceiling": float(np.mean([u[s]["e_window"] > u["ceiling"][s] for u in U.values()
                                                                         if u["half"] == "test" and u["kind"] == "real"]))}
                   for s in ("in", "out")}
    # (e) dependence: births sharing a carrier neuron or a twin neuron; a bootstrap clustered by carrier neuron
    ee = {}
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        rb = {u["birth"]: u for u in H if u["kind"] == "real"}
        td = {u["birth"]: u for u in H if u["kind"] == "dt"}
        ids = sorted(b for b in rb if b in td)
        car = [(rb[b]["layer"], rb[b]["neuron"]) for b in ids]
        twn = [(td[b]["layer"], td[b]["neuron"]) for b in ids]
        allneur = collections.Counter(car + twn)
        ee[half] = {"n_pairs": len(ids), "distinct_carrier_neurons": len(set(car)), "distinct_twin_neurons": len(set(twn)),
                    "twin_neuron_is_some_carrier_in_half": sum(t in set(car) for t in twn),
                    "neurons_used_more_than_once_in_pairs": sum(v > 1 for v in allneur.values())}
        if half == "test":
            for s in ("in", "out"):
                d = {b: rb[b][s]["e_window"] - td[b][s]["e_window"] for b in ids}
                groups = collections.defaultdict(list)
                for b in ids:
                    groups[(rb[b]["layer"], rb[b]["neuron"])].append(d[b])
                gl = list(groups.values())
                rng = np.random.default_rng(0)
                ms = []
                for _ in range(B):
                    pick = rng.integers(0, len(gl), len(gl))
                    v = np.concatenate([gl[i] for i in pick])
                    ms.append(v.mean())
                ee[half][f"cluster_boot_{s}"] = {"n_clusters": len(gl), "pct": pct(np.array(ms))}
            # the largest single contributions to the input-side paired mean
            d_in = sorted(((rb[b]["in"]["e_window"] - td[b]["in"]["e_window"], b) for b in ids))
            ee[half]["most_negative_in"] = [(round(x, 4), b, rb[b]["k"]) for x, b in d_in[:5]]
            ee[half]["most_positive_in"] = [(round(x, 4), b, rb[b]["k"]) for x, b in d_in[-5:]]
    C["e_dependence"] = ee
    out["confounds"] = C
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(out["headline"])
    for s in ("in", "out"):
        p = T[f"pair_{s}_real_dt"]
        print(s, {k: p[k] for k in ("n", "mean", "pct_seed0", "pct_seed1", "pct_seed2", "pct_vectorized_100k_seed0",
                                    "basic_seed0", "bca_seed0", "student_t", "n_pos", "n_neg", "sign_test_p", "wilcoxon_p",
                                    "median", "median_pct_boot", "trimmed10_mean", "drop_one_hi_max_2000boot")})
        print(" labels by method", lab[s]["labels_by_method"])


if __name__ == "__main__":
    main()
