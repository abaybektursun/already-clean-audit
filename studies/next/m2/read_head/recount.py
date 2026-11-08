"""Head's recount of the M2 verdict from the job's per-record ranks (own code; numpy only)."""
import json, math, numpy as np, collections
def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, c-h, c+h
def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]
for half in ("test", "freeze"):
    R = load(f"studies/next/m2/banks/score-06/m2/ranks_{half}.jsonl")
    kinds = collections.Counter((r["kind"], r["role"]) for r in R)
    print(f"\n== {half}: {len(R)} records; kinds x roles:", dict(kinds))
    pre = {r["unit"]: r for r in R if r["kind"] == "real" and r["role"] == "pre"}
    birth = {r["unit"]: r for r in R if r["kind"] == "real" and r["role"] == "birth"}
    end = {r["unit"]: r for r in R if r["kind"] == "real" and r["role"] == "end"}
    twins_kind = [k for k, _ in kinds if k != "real"]
    tw = {r["unit"]: r for r in R if r["kind"] != "real" and r["role"] == "pre"}
    n = len(pre); print("usable births (pre records):", n, "| birth records", len(birth), "| end records", len(end), "| twin pre records", len(tw), "kinds", twins_kind)
    for score in ("B", "A", "ratio"):
        hits = sum(1 for r in pre.values() if r["rank"][score] <= 10)
        p, lo, hi = wilson(hits, n)
        print(f"  pre {score}: hits {hits}/{n} = {p:.4f} [{lo:.4f}, {hi:.4f}]")
    for score in ("B", "A"):
        hb = sum(1 for r in birth.values() if r["rank"][score] <= 10); he = sum(1 for r in end.values() if r["rank"][score] <= 10)
        print(f"  birth {score}: {hb}/{len(birth)} = {hb/len(birth):.4f} | end {score}: {he}/{len(end)} = {he/len(end):.4f}")
    # paired A - B over births (my own bootstrap seed)
    units = sorted(pre); hA = np.array([pre[u]["rank"]["A"] <= 10 for u in units], float); hB = np.array([pre[u]["rank"]["B"] <= 10 for u in units], float)
    rng = np.random.default_rng(12345); d = hA - hB
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    print(f"  pair A-B: diff {d.mean():+.4f} [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}]  A-only {int(((hA==1)&(hB==0)).sum())} B-only {int(((hA==0)&(hB==1)).sum())}")
    # twins: match each twin to its birth
    # twin records carry "birth" = the real birth unit they belong to
    tw_by_birth = {r["birth"]: r for r in tw.values()}
    paired = [u for u in units if u in tw_by_birth]
    for score in ("B", "A"):
        hr = np.array([pre[u]["rank"][score] <= 10 for u in paired], float); ht = np.array([tw_by_birth[u]["rank"][score] <= 10 for u in paired], float)
        d = hr - ht; m = len(paired); bs = np.array([d[rng.integers(0, m, m)].mean() for _ in range(10000)])
        print(f"  twins {score}: n {m}, twin hits {int(ht.sum())}/{m} = {ht.mean():.4f}; real-twin diff {d.mean():+.4f} [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}]  real-only {int(((hr==1)&(ht==0)).sum())} twin-only {int(((hr==0)&(ht==1)).sum())}")
    # bar check: floor fields not in ranks; chance
    print("  chance 10/22016 =", round(10/22016, 6))
    # ranks distribution for B at pre
    rb = sorted(pre[u]["rank"]["B"] for u in units)
    print("  B pre ranks: median", rb[len(rb)//2], "p25", rb[len(rb)//4], "p75", rb[3*len(rb)//4], "max", rb[-1], "| rank 1 count", sum(1 for x in rb if x == 1))
    ra = sorted(pre[u]["rank"]["A"] for u in units)
    print("  A pre ranks: median", ra[len(ra)//2], "| rank==22016 (no overlap) count", sum(1 for x in ra if x >= 22016), "| A value 0 count", sum(1 for u in units if pre[u]["value"]["A"] == 0))
