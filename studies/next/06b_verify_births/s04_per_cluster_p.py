"""Step 3. Per late-born job (status late_primary, 1,238), the one-sided p of its birth against the 300 random
same-size groups of its own pole (set A; the group in the same slot of each draw is a uniform random subset of
that size from the pole's non-echo top-50 positions):
    p = (1 + #{draws with random birth >= real birth}) / (1 + 300).
With 300 draws, p < 0.05 is the same as "born later than more than 95% of the random groups" (k <= 14 of 300).
Chance level: (i) 0.05 x n; (ii) the selection-matched count, taking each draw d of set A in turn as the
pseudo-real grouping, applying the same selection (late_pct, not T_1, B3 raw agreement, B4 match) to its groups,
and scoring each selected group against the other 299 draws of its slot. (iii) the same with set B as pseudo-real
and all 300 draws of A as reference (no reuse).
Outputs: out/s04_per_cluster_p.json (per-cluster p), out/s04_per_cluster_p.txt."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
R = np.load(OUT / "random_groups.npz", allow_pickle=False)
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]}
cl = list(R["cluster"])
real_bp = np.array([banked[c]["birth_pct"] for c in cl])
status = np.array([banked[c]["status"] for c in cl])
bpA, bpB = R["A_bp"], R["B_bp"]
D = bpA.shape[0]

ge = (bpA >= real_bp[None, :]).sum(0)               # [2454]
lt = (bpA < real_bp[None, :]).sum(0)
p = (1 + ge) / (1 + D)
prim = status == "late_primary"
late_any = np.isin(status, ["formed_before_step_1000", "readings_disagree", "unmatched_in_sample_1", "late_primary"])
n_prim = int(prim.sum())
pass_ = prim & (p < 0.05)
mid = ((bpA > real_bp).sum(0) + 0.5 * (bpA == real_bp).sum(0)) / D      # mid-p (ties count half), tie sensitivity
later_than_median = prim & (real_bp > np.median(bpA, axis=0))

# (ii) leave-one-draw-out, set A as pseudo-real
def selected(prefix, d):
    return (R[f"{prefix}_late_pct"][d] & (R[f"{prefix}_bp"][d] != 1) & R[f"{prefix}_late_raw"][d] & R[f"{prefix}_matched"][d])

def selected_pct(prefix, d):
    return R[f"{prefix}_late_pct"][d]

loo = {"B3B4": [], "pct_only": [], "B3B4_midp": []}
loo_med, loo_med_fail = [], []
for d in range(D):
    others = np.delete(bpA, d, axis=0)
    pd_ = (1 + (others >= bpA[d][None, :]).sum(0)) / (1 + D - 1)
    md_ = ((others > bpA[d][None, :]).sum(0) + 0.5 * (others == bpA[d][None, :]).sum(0)) / (D - 1)
    for name, sel in (("B3B4", selected("A", d)), ("pct_only", selected_pct("A", d))):
        loo[name].append((int(sel.sum()), int((sel & (pd_ < 0.05)).sum())))
    sel = selected("A", d)
    loo["B3B4_midp"].append((int(sel.sum()), int((sel & (md_ < 0.05)).sum())))
    later = bpA[d] > np.median(others, axis=0)
    loo_med.append(float(later[sel].mean()))
    fsel = sel & (pd_ >= 0.05)
    loo_med_fail.append(float(later[fsel].mean()))
crossB = {"B3B4": [], "pct_only": []}
for d in range(D):
    pd_ = (1 + (bpA >= bpB[d][None, :]).sum(0)) / (1 + D)
    for name, sel in (("B3B4", selected("B", d)), ("pct_only", selected_pct("B", d))):
        crossB[name].append((int(sel.sum()), int((sel & (pd_ < 0.05)).sum())))

def summ(x):
    x = np.array(x, float)
    share = x[:, 1].sum() / x[:, 0].sum()
    return {"mean_selected": x[:, 0].mean(), "mean_passing": x[:, 1].mean(), "pass_share": share,
            "expected_among_real_n": share * n_prim, "passing_ci95": [float(np.percentile(x[:, 1], 2.5)), float(np.percentile(x[:, 1], 97.5))]}

pB = (1 + (bpB >= real_bp[None, :]).sum(0)) / (1 + D)
pAB = (1 + (np.vstack([bpA, bpB]) >= real_bp[None, :]).sum(0)) / (1 + 2 * D)
res = {"n_pass_vs_setB": int((prim & (pB < 0.05)).sum()), "n_pass_vs_setA_and_B_600": int((prim & (pAB < 0.05)).sum()),
       "n_late_primary": n_prim, "n_pass_p_lt_0.05": int(pass_.sum()), "share_pass": float(pass_.sum() / n_prim),
       "n_pass_midp_lt_0.05": int((prim & (mid < 0.05)).sum()),
       "n_pass_strict_gt95pct": int((prim & (lt / D > 0.95)).sum()),
       "share_later_than_median_random": float(later_than_median.sum() / n_prim),
       "chance_share_later_than_median_B3B4": float(np.mean(loo_med)),
       "chance_share_later_than_median_B3B4_ci95": [float(np.percentile(loo_med, 2.5)), float(np.percentile(loo_med, 97.5))],
       "share_later_than_median_failing_only": float((later_than_median & (p >= 0.05)).sum() / (prim & (p >= 0.05)).sum()),
       "chance_share_later_than_median_failing_only": float(np.mean(loo_med_fail)),
       "chance_share_later_than_median_failing_only_ci95": [float(np.percentile(loo_med_fail, 2.5)), float(np.percentile(loo_med_fail, 97.5))],
       "late_pct_any_n": int(late_any.sum()), "late_pct_any_pass": int((late_any & (p < 0.05)).sum()),
       "chance_naive_0.05n": 0.05 * n_prim,
       "chance_loo_A": {k: summ(v) for k, v in loo.items()},
       "chance_B_vs_A": {k: summ(v) for k, v in crossB.items()},
       "pass_by_birth_point": {f"T_{t}": [int((prim & (real_bp == t)).sum()), int((pass_ & (real_bp == t)).sum())] for t in range(16)},
       "per_cluster": {c: {"birth_pct": int(real_bp[j]), "p": float(p[j]), "n_random_ge": int(ge[j]), "status": str(status[j])}
                       for j, c in enumerate(cl)}}
json.dump(res, open(OUT / "s04_per_cluster_p.json", "w"), indent=1, default=float)
L = [f"late_primary jobs: {n_prim}",
     f"p < 0.05 (= born later than > 95% of 300 random same-size groups; ties count against): {pass_.sum()} ({pass_.sum()/n_prim:.3f})",
     f"same count against set B instead of A: {res['n_pass_vs_setB']}; against A and B together (600 draws): {res['n_pass_vs_setA_and_B_600']}",
     f"mid-p version (ties count half): {res['n_pass_midp_lt_0.05']}; strict share-earlier > 0.95: {res['n_pass_strict_gt95pct']}",
     f"share born later than the median random group: {res['share_later_than_median_random']:.3f}",
     f"chance level of that share (leave-one-draw-out, same selection B3+B4): {res['chance_share_later_than_median_B3B4']:.3f} {res['chance_share_later_than_median_B3B4_ci95']}",
     f"failing jobs only: share later than the median random group {res['share_later_than_median_failing_only']:.3f}; chance level {res['chance_share_later_than_median_failing_only']:.3f} {res['chance_share_later_than_median_failing_only_ci95']}",
     f"all late-born by the percentile reading (n {res['late_pct_any_n']}): {res['late_pct_any_pass']} pass",
     f"chance (i) 0.05 x {n_prim} = {0.05*n_prim:.1f}",
     f"chance (ii) leave-one-draw-out in A, same selection (B3+B4): {res['chance_loo_A']['B3B4']}",
     f"chance (ii) leave-one-draw-out in A, percentile late only: {res['chance_loo_A']['pct_only']}",
     f"chance (ii) for the mid-p version, B3+B4: {res['chance_loo_A']['B3B4_midp']}",
     f"chance (iii) set B pseudo-real vs A, B3+B4: {res['chance_B_vs_A']['B3B4']}",
     f"chance (iii) set B pseudo-real vs A, percentile late only: {res['chance_B_vs_A']['pct_only']}",
     f"late_primary [n, passing] by birth point: {res['pass_by_birth_point']}"]
(OUT / "s04_per_cluster_p.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
