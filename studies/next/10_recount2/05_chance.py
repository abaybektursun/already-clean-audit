"""Chance reference for the dates and for b: what dates arise when a cluster is replaced by a random group of
the same pole's non-echo T_15 exemplars, of the same size, dated with the same rule.

Written before the run (2026-11-05):
- Primary reference R1t ("size and top rank"): per pole (sample 0, m >= 2), the cluster labels of the pole's
  non-echo ranks are shuffled, except that the pole's top non-echo rank keeps the main cluster's label. Every
  random group has its real cluster's size; main's replacement holds the top exemplar, which is what makes a
  cluster main (largest max|a| at T_15, SPEC line 291). Main of the random partition is chosen by the same rule.
- Sensitivity R1 ("size only"): all labels shuffled; main = the group that receives the top rank.
- Sensitivity R2 ("size, top rank and rank composition"): as R1t, but labels are shuffled only within rank
  bins of five (ranks 0-4, 5-9, ..., 45-49), so each group keeps its count per bin.
- 2,000 random partitions per pole (one partition of every pole = one replicate of b). Seed 20261105.
- b reference value = mean of the replicate b's; interval = 2.5th and 97.5th percentiles of the replicate b's.
  Paired gap = mean over poles of (real indicator - the pole's reference probability), percentile bootstrap
  over poles, 10,000 resamples.
- Per cluster: p_late = P(random date >= real date), p_early = P(random date <= real date), from the
  cluster's own column of the partitions (never born counts as latest). Threshold alpha = 0.05 one-sided in
  each direction. Expected count at chance = sum over clusters of P_ref(p(B) <= 0.05) for B drawn from the
  same reference, which is exact for these discrete dates (it is below 0.05 per cluster when dates tie).
- Calibration input (must give counts near the expected count, and b near the reference): one extra random
  partition per pole from R1t, used in place of the real clusters.
- Positive control (must give counts far above the expected count): every non-main cluster replaced by the
  group of its size made of the pole's latest-dated single exemplars (late control) or earliest-dated single
  exemplars (early control), top rank excluded; ties broken by rank.
Outputs: out/05_chance.json, out/05_per_cluster.csv.
"""
import json, os
import numpy as np
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
A = C.load_arrays()
REPS, ALPHA, SEED = 2000, 0.05, 20261105
rng = np.random.default_rng(SEED)
mine = [json.loads(l) for l in open(C.BIG + "/births.jsonl")]
poles = {}
for l in mine:
    if l["sample"] == 0:
        poles.setdefault(l["pole"], []).append(l)
P = [p for p in sorted(poles) if len(poles[p]) >= 2]


def shuffled_labels(lab, ne, mode, reps):
    n = len(lab)
    u = rng.random((reps, n))
    if mode == "R1":
        keys = u
    elif mode == "R1t":
        keys = u.copy(); keys[:, 0] = -1.0
    elif mode == "R2":
        keys = (ne // 5)[None, :] + u; keys[:, 0] = -1.0
    perm = np.argsort(keys, axis=1)
    return lab[perm]                                   # [reps, n]


def date_partitions(i, ne, newlab, m):
    """Births [reps, m] of the groups defined by newlab; plus the index of the group holding the top rank."""
    reps = newlab.shape[0]
    order = np.argsort(newlab, axis=1, kind="stable")
    counts = np.bincount(newlab[0], minlength=m)
    starts = np.concatenate([[0], np.cumsum(counts)])
    B = np.zeros((reps, m), dtype=np.int64)
    for j in range(m):
        groups = ne[order[:, starts[j]:starts[j + 1]]]
        B[:, j] = C.births_matrix(A["a_plus"][:, i, :], A["q999"][:, i], groups)
    return B, newlab[:, 0]


def single_exemplar_dates(i, ne):
    g = ne[:, None]
    return C.births_matrix(A["a_plus"][:, i, :], A["q999"][:, i], g)


res = {"design": __doc__, "n_poles": len(P), "reps": REPS, "alpha": ALPHA, "seed": SEED}
per_cluster = []
summ = {}
for mode in ["R1t", "R1", "R2"]:
    real_ind, ref_prob, rep_ind = [], [], []
    real_strict, rep_strict = [], []
    real_auc, rep_auc = [], []   # tie-neutral: share of the pole's other clusters born after main, ties as 1/2
    cal_ind = []
    pc = []   # per cluster: (pole, c, size, is_main, real, p_late, p_early, exp_late, exp_early, cal_late, cal_early, ctrl...)
    for p in P:
        cl = poles[p]
        i = A["row"][p]
        m = len(cl)
        ne = np.array(sorted(r for c in cl for r in c["ranks"]))
        lab = np.empty(len(ne), dtype=np.int64)
        pos = {r: k for k, r in enumerate(ne)}
        for j, c in enumerate(cl):
            for r in c["ranks"]:
                lab[pos[r]] = j
        jm = [j for j, c in enumerate(cl) if c["is_main"]][0]
        assert lab[0] == jm, (p, "main does not hold the top non-echo rank")
        real = np.array([99 if c["birth_pct"] is None else c["birth_pct"] for c in cl])
        real_ind.append(real[jm] == real.min())
        real_strict.append(real[jm] == real.min() and (real == real.min()).sum() == 1)
        newlab = shuffled_labels(lab, ne, mode, REPS + 1)
        B, top = date_partitions(i, ne, newlab, m)
        Bmain = B[np.arange(REPS + 1), top]
        ind = Bmain == B.min(axis=1)
        rep_ind.append(ind[:REPS]); ref_prob.append(ind[:REPS].mean()); cal_ind.append(ind[REPS])
        strict = ind & ((B == B.min(axis=1, keepdims=True)).sum(axis=1) == 1)
        rep_strict.append(strict[:REPS])
        oth = np.array([j for j in range(m) if j != jm])
        real_auc.append(np.mean((real[oth] > real[jm]) + 0.5 * (real[oth] == real[jm])))
        rr = np.arange(REPS)
        others = np.array([[j for j in range(m) if j != top[r]] for r in rr])
        Bo = B[rr[:, None], others]
        rep_auc.append(np.mean((Bo > Bmain[:REPS, None]) + 0.5 * (Bo == Bmain[:REPS, None]), axis=1))
        Bref, Bcal = B[:REPS], B[REPS]
        if mode == "R1t":
            sd = single_exemplar_dates(i, ne)                    # date of each single exemplar
            rest = np.arange(1, len(ne))                         # top rank excluded
            late_order = rest[np.lexsort((ne[rest], -sd[rest]))]  # latest first, then by rank
            early_order = rest[np.lexsort((ne[rest], sd[rest]))]
        for j, c in enumerate(cl):
            col = np.sort(Bref[:, j])
            def p_late(x):  # P(random >= x)
                return 1.0 - np.searchsorted(col, x, side="left") / REPS
            def p_early(x):  # P(random <= x)
                return np.searchsorted(col, x, side="right") / REPS
            vals, cnt = np.unique(col, return_counts=True)
            exp_late = float(sum(cn for v, cn in zip(vals, cnt) if p_late(v) <= ALPHA) / REPS)
            exp_early = float(sum(cn for v, cn in zip(vals, cnt) if p_early(v) <= ALPHA) / REPS)
            row = {"pole": p, "c": c["c"], "size": len(c["ranks"]), "m": m, "is_main": c["is_main"],
                   "real": int(real[j]), "p_late": float(p_late(real[j])), "p_early": float(p_early(real[j])),
                   "exp_late": exp_late, "exp_early": exp_early,
                   "cal_late": float(p_late(Bcal[j])), "cal_early": float(p_early(Bcal[j])),
                   "ref_median": float(np.median(col)), "ref_mean": float(col.mean())}
            if mode == "R1t" and not c["is_main"]:
                n = len(c["ranks"])
                for name, order in [("ctrl_late", late_order), ("ctrl_early", early_order)]:
                    if n <= len(order):
                        g = ne[order[:n]][None, :]
                        bd = int(C.births_matrix(A["a_plus"][:, i, :], A["q999"][:, i], g)[0])
                        row[name + "_date"] = bd
                        row[name + "_p"] = float(p_late(bd) if name == "ctrl_late" else p_early(bd))
            pc.append(row)
    real_ind = np.array(real_ind, float); ref_prob = np.array(ref_prob); rep_b = np.array(rep_ind).mean(axis=0)
    cal_ind = np.array(cal_ind, float)
    diff = real_ind - ref_prob
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    rng_s = np.random.default_rng(SEED + 1)   # separate stream, so the draws above do not move
    diff_a = np.array(real_auc) - np.array(rep_auc).mean(axis=1)
    rng_a = np.random.default_rng(SEED + 2)
    boot_a = np.array([diff_a[rng_a.integers(0, len(diff_a), len(diff_a))].mean() for _ in range(10000)])
    rep_auc_b = np.array(rep_auc).mean(axis=0)
    diff_s = np.array(real_strict, float) - np.array(rep_strict).mean(axis=1)
    boot_s = np.array([diff_s[rng_s.integers(0, len(diff_s), len(diff_s))].mean() for _ in range(10000)])
    nm = [r for r in pc if not r["is_main"]]
    s = {"b_real": float(real_ind.mean()),
         "b_ref_mean": float(rep_b.mean()),
         "b_ref_ci95_over_replicates": [float(np.percentile(rep_b, 2.5)), float(np.percentile(rep_b, 97.5))],
         "share_replicates_b_ref_ge_b_real": float((rep_b >= real_ind.mean()).mean()),
         "paired_gap_b_minus_ref": float(diff.mean()),
         "paired_gap_ci95_bootstrap_poles": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
         "calibration_b": float(cal_ind.mean()),
         "b_strict_real": float(np.mean(real_strict)),
         "b_strict_ref_mean": float(np.array(rep_strict).mean()),
         "b_strict_paired_gap": float(diff_s.mean()),
         "auc_real": float(np.mean(real_auc)), "auc_ref_mean": float(rep_auc_b.mean()),
         "auc_ref_ci95_over_replicates": [float(np.percentile(rep_auc_b, 2.5)), float(np.percentile(rep_auc_b, 97.5))],
         "auc_paired_gap": float(diff_a.mean()),
         "auc_paired_gap_ci95": [float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5))],
         "b_strict_paired_gap_ci95": [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
         "b_strict_ref_ci95_over_replicates": [float(np.percentile(np.array(rep_strict).mean(axis=0), 2.5)),
                                               float(np.percentile(np.array(rep_strict).mean(axis=0), 97.5))],
         "n_clusters": len(pc), "n_main": len(pc) - len(nm), "n_nonmain": len(nm)}
    for name, rows in [("all", pc), ("main", [r for r in pc if r["is_main"]]), ("nonmain", nm)]:
        s[name] = {"n": len(rows),
                   "late_beat": sum(r["p_late"] <= ALPHA for r in rows),
                   "late_expected": round(sum(r["exp_late"] for r in rows), 1),
                   "early_beat": sum(r["p_early"] <= ALPHA for r in rows),
                   "early_expected": round(sum(r["exp_early"] for r in rows), 1),
                   "calibration_late_beat": sum(r["cal_late"] <= ALPHA for r in rows),
                   "calibration_early_beat": sum(r["cal_early"] <= ALPHA for r in rows),
                   "mean_real_date": float(np.mean([r["real"] for r in rows])),
                   "mean_ref_date": float(np.mean([r["ref_mean"] for r in rows])),
                   "n_real_equal_ref_median": sum(r["real"] == r["ref_median"] for r in rows),
                   "naive_expected_alpha_times_n": round(ALPHA * len(rows), 1)}
    if mode == "R1t":
        for name in ["ctrl_late", "ctrl_early"]:
            rr = [r for r in nm if name + "_p" in r]
            s[name] = {"n": len(rr), "beat": sum(r[name + "_p"] <= ALPHA for r in rr),
                       "expected": round(sum(r["exp_late" if name == "ctrl_late" else "exp_early"] for r in rr), 1)}
        per_cluster = pc
    summ[mode] = s
    print(mode, json.dumps(s, indent=1))
res["summary"] = summ
json.dump(res, open(OUT + "/05_chance.json", "w"), indent=1)
keys = ["pole", "c", "size", "m", "is_main", "real", "p_late", "p_early", "exp_late", "exp_early", "ref_median",
        "ref_mean", "ctrl_late_date", "ctrl_late_p", "ctrl_early_date", "ctrl_early_p"]
with open(OUT + "/05_per_cluster_R1t.csv", "w") as f:
    f.write(",".join(keys) + "\n")
    for r in per_cluster:
        f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
