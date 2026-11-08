"""Three raw examples for the report, and how much of the real dates the chance reference predicts.

Examples: (1) a non-main cluster dated later than chance under R1t; (2) a non-main cluster at chance;
(3) no recount mismatch exists, so the third is a cluster whose date moves with the median convention.
For each: exemplar positions, per-checkpoint a+ of every exemplar, the median, q999, presence, the recount
and banked dates, and the R1t reference date histogram from a fresh draw of 2,000 groups (seed 7).
Outputs: out/06_examples.json, out/06_predictability.json.
"""
import json, os, csv
import numpy as np
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
A = C.load_arrays()
pc = list(csv.DictReader(open(OUT + "/05_per_cluster_R1t.csv")))
for r in pc:
    for k in ["size", "m", "real", "c"]:
        r[k] = int(r[k])
    for k in ["p_late", "p_early", "ref_median", "ref_mean", "exp_late"]:
        r[k] = float(r[k])
    r["is_main"] = r["is_main"] == "True"


def ranks_avg(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    xs = np.asarray(x)[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


real = np.array([r["real"] for r in pc], float); refm = np.array([r["ref_mean"] for r in pc])
pred = {"n": len(pc), "pearson_real_vs_ref_mean": float(np.corrcoef(real, refm)[0, 1]),
        "spearman_real_vs_ref_mean": float(np.corrcoef(ranks_avg(real), ranks_avg(refm))[0, 1]),
        "share_real_equal_ref_median": float(np.mean([r["real"] == r["ref_median"] for r in pc])),
        "share_within_chance_both_sides": float(np.mean([(r["p_late"] > 0.05) and (r["p_early"] > 0.05) for r in pc])),
        "mean_abs_real_minus_ref_mean": float(np.mean(np.abs(real - refm))),
        "sd_real_dates": float(real.std())}
json.dump(pred, open(OUT + "/06_predictability.json", "w"), indent=1)
print(json.dumps(pred, indent=1))

births = {(l["pole"], l["c"]): l for l in map(json.loads, open(C.BIG + "/births.jsonl")) if l["sample"] == 0}
banked = {}
for l in open(C.BANKED):
    r = json.loads(l)
    banked[(r["pole"], r["c"])] = r["birth_pct"]
names = {}
for r in C.load_clusters():
    if r["sample"] == 0:
        for c in r["clusters"]:
            names[(r["pole"], c["c"])] = c["name"]
ctx15 = {}
for l in open(C.BANK + "/w3/grid/1/out/t15_step1413814/samples/contexts.jsonl"):
    r = json.loads(l)
    ctx15[r["pole"]] = r["topk"]

pick = {}
cand = [r for r in pc if not r["is_main"] and r["p_late"] <= 0.05 and r["size"] <= 5]
cand.sort(key=lambda r: (r["p_late"], r["size"], r["pole"]))
pick["beyond_chance_late"] = cand[0]
cand = [r for r in pc if not r["is_main"] and r["real"] == r["ref_median"] and 0.3 <= r["p_late"] <= 0.7
        and 3 <= r["size"] <= 5]
cand.sort(key=lambda r: (abs(r["p_late"] - 0.5), r["pole"]))
pick["at_chance"] = cand[0]
cand = [l for l in births.values() if l["birth_pct_lower"] != l["birth_pct"] and l["size"] == 4 and not l["is_main"]]
cand.sort(key=lambda l: (l["pole"], l["c"]))
pick["median_convention_sensitive"] = {"pole": cand[0]["pole"], "c": cand[0]["c"]}

rng = np.random.default_rng(7)
ex = {}
for name, r in pick.items():
    l = births[(r["pole"], r["c"])]
    i = A["row"][l["pole"]]
    R = l["ranks"]
    allc = [b for b in births.values() if b["pole"] == l["pole"]]
    ne = np.array(sorted(x for b in allc for x in b["ranks"]))
    top = ne[0]
    pool = ne[1:] if not l["is_main"] else ne[1:]
    n = len(R)
    if l["is_main"]:
        groups = np.array([np.concatenate([[top], rng.choice(pool, n - 1, replace=False)]) for _ in range(2000)])
    else:
        groups = np.array([rng.choice(pool, n, replace=False) for _ in range(2000)])
    bd = C.births_matrix(A["a_plus"][:, i, :], A["q999"][:, i], groups)
    vals, cnt = np.unique(bd, return_counts=True)
    ex[name] = {"pole": l["pole"], "c": l["c"], "name": names[(l["pole"], l["c"])], "is_main": l["is_main"],
                "m": len(allc), "size": n, "ranks": R,
                "positions_T15": [{"rank": k, "doc": ctx15[l["pole"]][k]["doc"], "pos": ctx15[l["pole"]][k]["pos"],
                                   "tok": ctx15[l["pole"]][k]["tok"], "token": ctx15[l["pole"]][k]["token"],
                                   "before_tail": ctx15[l["pole"]][k]["before"][-80:]} for k in R],
                "a_plus_by_t": [[round(float(A["a_plus"][t, i, k]), 4) for k in R] for t in range(16)],
                "median_by_t": [round(x, 4) for x in l["med"]],
                "q999_by_t": [round(x, 4) for x in l["q999"]],
                "present_pct": l["present_pct"],
                "birth_recount": l["birth_pct"], "birth_banked": banked[(l["pole"], l["c"])],
                "birth_lower_median": l["birth_pct_lower"], "birth_upper_median": l["birth_pct_upper"],
                "ref_hist_fresh_draw": {int(v): int(c) for v, c in zip(vals, cnt)},
                "p_late_fresh": float(np.mean(bd >= l["birth_pct"])),
                "p_early_fresh": float(np.mean(bd <= l["birth_pct"])),
                "other_clusters_of_pole": [{"c": b["c"], "size": b["size"], "is_main": b["is_main"],
                                            "birth": b["birth_pct"]} for b in sorted(allc, key=lambda b: b["c"])]}
    if name != "median_convention_sensitive":
        ex[name]["p_late_05"] = r["p_late"]; ex[name]["p_early_05"] = r["p_early"]
json.dump(ex, open(OUT + "/06_examples.json", "w"), indent=1)
for k, v in ex.items():
    print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("a_plus_by_t",)}, ensure_ascii=False)[:2500])
    for t in range(16):
        print("  t", t, v["a_plus_by_t"][t], "med", v["median_by_t"][t], "q999", v["q999_by_t"][t], v["present_pct"][t])
