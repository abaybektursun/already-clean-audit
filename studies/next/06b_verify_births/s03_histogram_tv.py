"""Step 2b. Birth histogram (percentile reading) of the 2,454 real non-echo clusters against random groups,
total-variation distance over the 16 points, and noise floors. Also late-born counts before and after B3/B4.
Inputs: out/q2_banked.json, out/random_groups.npz. Outputs: out/s03_histogram_tv.json, out/s03_histogram_tv.txt."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
R = np.load(OUT / "random_groups.npz", allow_pickle=False)
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]}
cl = list(R["cluster"])
real_bp = np.array([banked[c]["birth_pct"] for c in cl])
real_main = np.array([banked[c]["is_main"] for c in cl])
real_status = np.array([banked[c]["status"] for c in cl])
T, N = 16, len(cl)
STEPS = (0, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 35000, 60000, 100000, 180000, 320000, 560000, 1000000, 1413814)


def counts(b, sel=None):
    b = b if sel is None else b[sel]
    return np.bincount(b, minlength=T)


def tv(p, q):
    return 0.5 * np.abs(p / p.sum() - q / q.sum()).sum()


hr = counts(real_bp).astype(float)
hA = np.array([counts(d) for d in R["A_bp"]], float)        # [300, 16]
hB = np.array([counts(d) for d in R["B_bp"]], float)
poolA, poolB = hA.sum(0), hB.sum(0)
tv_real = tv(hr, poolA)
tv_real_B = tv(hr, poolB)
loo = np.array([tv(hA[d], poolA - hA[d]) for d in range(len(hA))])      # one draw vs the other 299 (set A)
pair = np.array([tv(hA[d], hB[d]) for d in range(len(hA))])              # two single independent draws
tv_real_single = np.array([tv(hr, hA[d]) for d in range(len(hA))])       # real vs each single draw
tv_AB = tv(poolA, poolB)
p_tv = (1 + (loo >= tv_real).sum()) / (1 + len(loo))

# main / non-main split, counts per point: real vs mean over set A
mainA = np.array([counts(R["A_bp"][d], R["A_main"][d]) for d in range(len(hA))], float)
nonA = np.array([counts(R["A_bp"][d], ~R["A_main"][d]) for d in range(len(hA))], float)

# late-born counts
lateA = R["A_late_pct"].sum(1)
primA = (R["A_late_pct"] & (R["A_bp"] != 1) & R["A_late_raw"] & R["A_matched"]).sum(1)
b3A = (R["A_late_pct"] & (R["A_bp"] != 1) & R["A_late_raw"]).sum(1)
real_late = int(np.isin(real_status, ["formed_before_step_1000", "readings_disagree", "unmatched_in_sample_1", "late_primary"]).sum())
real_b3 = int(np.isin(real_status, ["unmatched_in_sample_1", "late_primary"]).sum())
real_prim = int((real_status == "late_primary").sum())
q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

late_T14_T15_real = int(counts(real_bp, ~real_main)[14:].sum())
late_T14_T15_A = nonA[:, 14:].sum(1)

res = {"n_clusters": N, "n_draws": len(hA),
       "real_counts": hr.tolist(), "random_mean_counts_A": (poolA / len(hA)).tolist(),
       "real_share": (hr / hr.sum()).tolist(), "random_share_A": (poolA / poolA.sum()).tolist(),
       "TV_real_vs_pooledA": tv_real, "TV_real_vs_pooledB": tv_real_B,
       "TV_one_draw_vs_other_299_A": {"mean": loo.mean(), "median": float(np.median(loo)), "p95": float(np.percentile(loo, 95)), "max": loo.max()},
       "p_TV_real_vs_loo": p_tv,
       "TV_single_draw_A_vs_single_draw_B": {"mean": pair.mean(), "median": float(np.median(pair)), "p95": float(np.percentile(pair, 95))},
       "TV_real_vs_single_draw_A": {"mean": tv_real_single.mean(), "median": float(np.median(tv_real_single)), "min": tv_real_single.min()},
       "TV_pooledA_vs_pooledB": tv_AB,
       "main_real": counts(real_bp, real_main).tolist(), "main_random_mean": mainA.mean(0).tolist(),
       "non_main_real": counts(real_bp, ~real_main).tolist(), "non_main_random_mean": nonA.mean(0).tolist(),
       "late_pct_real": real_late, "late_pct_random": {"mean": lateA.mean(), "ci95": q(lateA)},
       "late_B3_real": real_b3, "late_B3_random": {"mean": b3A.mean(), "ci95": q(b3A)},
       "late_primary_real": real_prim, "late_primary_random_B3_B4": {"mean": primA.mean(), "ci95": q(primA)},
       "non_main_births_T14_T15_real": late_T14_T15_real,
       "non_main_births_T14_T15_random": {"mean": late_T14_T15_A.mean(), "ci95": q(late_T14_T15_A)}}
json.dump(res, open(OUT / "s03_histogram_tv.json", "w"), indent=1, default=float)

L = ["point | step | real count | real share | random mean count | random share | main real | main random | non-main real | non-main random"]
for t in range(T):
    L.append(f"T_{t} | {STEPS[t]} | {int(hr[t])} | {hr[t]/hr.sum():.3f} | {poolA[t]/len(hA):.1f} | {poolA[t]/poolA.sum():.3f} | "
             f"{res['main_real'][t]} | {res['main_random_mean'][t]:.1f} | {res['non_main_real'][t]} | {res['non_main_random_mean'][t]:.1f}")
L += [f"TV(real, pooled set A, 300 draws) = {tv_real:.4f}; vs pooled set B = {tv_real_B:.4f}",
      f"TV(one draw, other 299 of A): mean {loo.mean():.4f}, median {np.median(loo):.4f}, 95th {np.percentile(loo,95):.4f}, max {loo.max():.4f}; p(real) = {p_tv:.4f}",
      f"TV(single draw A_d, single draw B_d): mean {pair.mean():.4f}, 95th {np.percentile(pair,95):.4f}",
      f"TV(real, single draw A_d): mean {tv_real_single.mean():.4f}, min {tv_real_single.min():.4f}",
      f"TV(pooled A, pooled B) = {tv_AB:.4f}",
      f"late-born (percentile reading, before B3/B4): real {real_late}, random mean {lateA.mean():.1f} {q(lateA)}",
      f"late-born after B3 (and not T_1): real {real_b3}, random mean {b3A.mean():.1f} {q(b3A)}",
      f"late_primary (B3 and B4): real {real_prim}, random mean {primA.mean():.1f} {q(primA)}",
      f"non-main births at T_14+T_15: real {late_T14_T15_real}, random mean {late_T14_T15_A.mean():.1f} {q(late_T14_T15_A)}"]
(OUT / "s03_histogram_tv.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
