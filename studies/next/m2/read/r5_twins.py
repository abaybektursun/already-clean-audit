"""Reader of M2, phase 1, step 5: why do the twins lead their layers more often than the real carriers?

The twin differs from the birth in two registered ways (M2_SPEC.md 3.3): its group is not a coherent cluster, and its
date is typical for its pole rather than late. A hit under B needs the unit's own percentile v to beat the other
21,999+ poles' own percentiles on the same positions. So the paired difference can come from the unit's own
percentile (own side) or from the competitors' percentiles (competitor side). This script splits the two from the
banked records, and measures lateness for the twins with the same arrays the births were dated with.

Inputs: banks/score-06/m2/ranks_*.jsonl (all pre-role records), read/r3_A.json (own A recompute and competition
counts from the banked top-50 arrays), arrays.npz, clusters.jsonl, judgments.jsonl (W4 judge, sample 0), births.json,
and the grid's T_15 contexts (S3 results/neurons/w3/grid/1/out/t15_step1413814/samples/contexts.jsonl, path given as
argv[1]; sha256 recorded) for the current token at each exemplar position. Output: read/r5_twins.json.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r1_births import ARR, CLU, date, aplus
from r2_from_ranks import boot, wilson

ROOT = Path(__file__).resolve().parents[4]
M2 = ROOT / "studies/next/m2"


def late_proxy(A, Q, row, sign, ne, size, k, rng, R=2000):
    """share of R uniform random same-size groups of the pole's non-echo ranks dated at k or later (never = latest)."""
    ne = np.array(ne)
    c = 0
    for _ in range(R):
        g = np.sort(rng.choice(ne, size, replace=False))
        kk, _m = date(A, Q, row, g, sign)
        c += (kk is None) or (kk >= k)
    return c / R


def main():
    d = json.loads((M2 / "births.json").read_text())
    births = {b["id"]: b for b in d["births"]}
    rec = {}
    for h in ("freeze", "test"):
        for l in (M2 / f"banks/score-06/m2/ranks_{h}.jsonl").open():
            r = json.loads(l)
            rec[(r["birth"], r["kind"], r["role"])] = r
    A3 = json.loads((M2 / "read/r3_A.json").read_text())["units"]
    z = np.load(ARR, allow_pickle=False)
    A, Q = z["a"], z["q999"]
    clus = [json.loads(l) for l in CLU.open()]
    s0 = {c["pole"]: c for c in clus if c["sample"] == 0}

    ctxp = Path(sys.argv[1])
    ctx = {}
    for line in ctxp.open():
        c = json.loads(line)
        ctx[c["pole"]] = c["topk"]
    J = [json.loads(l) for l in (CLU.parent / "judgments.jsonl").open()]
    jacc = {}
    for j in J:
        if j["sample"] == 0 and not j["control"]:
            jacc.setdefault(j["cluster"], []).append(int(j["correct"]))

    def tokens(pole, ranks, toks):
        tk = ctx[pole]
        assert [tk[r]["tok"] for r in ranks] == toks, pole
        return [tk[r]["token"] for r in ranks]

    def ne_of(p):
        return sorted(r for x in s0[p]["clusters"] if not x["echo"] for r in x["exemplar_ranks"])

    rows = []
    rng = np.random.default_rng(7)
    for bid, b in births.items():
        if (bid, "pb", "pre") not in rec:
            continue
        row = {"birth": bid, "half": b["half"], "k": b["k"], "size": b["size"], "layer": b["layer"]}
        for kind in ("real", "pb"):
            r = rec[(bid, kind, "pre")]
            vals = [x[1] for x in r["top10"]["B"]]
            names = [x[0] for x in r["top10"]["B"]]
            others = [v for n, v in zip(names, vals) if n != r["carrier"]]
            u = A3[f"{bid}|{kind}|pre"]
            src = b if kind == "real" else b["pb"]
            row[kind] = {"pole": r["carrier"], "B": r["value"]["B"], "rank_B": r["rank"]["B"],
                         "hit_B": int(r["rank"]["B"] <= 10), "hit_A": int(r["rank"]["A"] <= 10),
                         "ratio": r["value"]["ratio"], "A": r["value"]["A"],
                         "B10_listed": vals[9], "best_other_B": others[0],
                         "n_other_poles_A_ge1": u["n_other_poles_A_ge1"],
                         "n_other_pole_slots_per_pos": u["n_other_pole_slots_on_pos"] / b["size"],
                         "share_pos_in_other_top50": u["share_pos_in_other_top50"],
                         "share_pos_in_own_top50": u["share_pos_in_own_top50"],
                         "distinct_token_share": len(set(tokens(src["pole"], src["ranks"], src["tok"]))) / b["size"],
                         "late_proxy": late_proxy(A, Q, src["row"], 0 if src["pole"].endswith("+") else 1,
                                                  ne_of(src["pole"]), b["size"], b["k"], rng)}
        js = jacc.get(f"{b['pole']}:c{b['c']}", [])
        row["judge_acc_real"] = float(np.mean(js)) if js else None
        row["judge_n_real"] = len(js)
        rows.append(row)

    def summ(sel, name):
        R = [x for x in rows if sel(x)]
        n = len(R)
        if n == 0:
            return {"n": 0}
        g = lambda kind, f: np.array([x[kind][f] for x in R], float)
        o = {"n": n}
        for f in ("B", "ratio", "B10_listed", "best_other_B", "n_other_poles_A_ge1", "n_other_pole_slots_per_pos",
                  "share_pos_in_other_top50", "share_pos_in_own_top50", "late_proxy", "A",
                  "distinct_token_share"):
            r_, t_ = g("real", f), g("pb", f)
            o[f] = {"real_median": float(np.median(r_)), "twin_median": float(np.median(t_)),
                    "share_twin_gt_real": float(np.mean(t_ > r_)), "share_twin_lt_real": float(np.mean(t_ < r_))}
        hr, ht = g("real", "hit_B"), g("pb", "hit_B")
        o["hits_B"] = {"real": int(hr.sum()), "twin": int(ht.sum())}
        # counterfactuals: own percentile of one unit against the other's competitor bar (10th listed value)
        Br, Bt, Cr, Ct = g("real", "B"), g("pb", "B"), g("real", "B10_listed"), g("pb", "B10_listed")
        o["counterfactual_hits"] = {
            "real_own_vs_real_bar": int((Br >= Cr).sum()), "twin_own_vs_twin_bar": int((Bt >= Ct).sum()),
            "twin_own_vs_real_bar": int((Bt >= Cr).sum()), "real_own_vs_twin_bar": int((Br >= Ct).sum())}
        o["diff_own_side"] = boot((Br >= Cr).astype(float) - (Bt >= Cr).astype(float))
        o["diff_competitor_side"] = boot((Br >= Cr).astype(float) - (Br >= Ct).astype(float))
        return o

    out = {"n_pairs": len(rows)}
    for h in ("freeze", "test", "both"):
        sel = (lambda x: True) if h == "both" else (lambda x, h=h: x["half"] == h)
        out[h] = summ(sel, h)
    # hit rate against own B, pooled over real and twin units, in bins of 1 - B (log scale)
    pooled = [(x[k]["B"], x[k]["hit_B"], k) for x in rows for k in ("real", "pb")]
    edges = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.01]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cell = {}
        for k in ("real", "pb"):
            s = [h for B, h, kk in pooled if kk == k and lo <= 1 - B < hi]
            cell[k] = [int(sum(s)), len(s)]
        bins.append({"one_minus_B": [lo, hi], "real_hits_n": cell["real"], "twin_hits_n": cell["pb"]})
    out["hit_rate_by_own_B_bin"] = bins
    # coherence proxy 1: share of distinct current tokens at the unit's positions (1/size = one token everywhere)
    cb = []
    for lo, hi in ((0, 0.34), (0.34, 0.67), (0.67, 0.99), (0.99, 1.01)):
        cell = {}
        for k in ("real", "pb"):
            s_ = [x[k]["hit_B"] for x in rows if lo <= x[k]["distinct_token_share"] < hi]
            cell[k] = [int(sum(s_)), len(s_)]
        cb.append({"distinct_token_share": [lo, hi], "real_hits_n": cell["real"], "twin_hits_n": cell["pb"]})
    out["hit_rate_by_distinct_token_share"] = cb
    # coherence proxy 2 (real births only): the W4 judge's accuracy on the cluster's label against decoys
    jr = [(x["judge_acc_real"], x["real"]["hit_B"]) for x in rows if x["judge_acc_real"] is not None]
    if jr:
        qs = np.quantile([a for a, _ in jr], [1 / 3, 2 / 3])
        tb = []
        for lo, hi in ((-0.01, qs[0]), (qs[0], qs[1]), (qs[1], 1.01)):
            s_ = [h for a, h in jr if lo < a <= hi]
            tb.append({"judge_acc": [float(lo), float(hi)], "real_hits_n": [int(sum(s_)), len(s_)]})
        out["real_hit_rate_by_judge_accuracy_tercile"] = tb
        out["n_real_with_judge"] = len(jr)
    # coherence proxy 3: other poles holding the positions in their top-50 at T_(k-1) (from r3)
    ob = []
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 1e9)):
        cell = {}
        for k in ("real", "pb"):
            s_ = [x[k]["hit_B"] for x in rows if lo <= x[k]["n_other_pole_slots_per_pos"] < hi]
            cell[k] = [int(sum(s_)), len(s_)]
        ob.append({"other_pole_slots_per_position": [lo, hi], "real_hits_n": cell["real"], "twin_hits_n": cell["pb"]})
    out["hit_rate_by_other_pole_slots_per_position"] = ob
    out["contexts_sha256"] = hashlib.sha256(ctxp.read_bytes()).hexdigest()
    # lateness: twins split at the real births' own late_proxy level
    lp_real = np.array([x["real"]["late_proxy"] for x in rows])
    out["late_proxy_real_quantiles"] = [float(v) for v in np.quantile(lp_real, [0.1, 0.25, 0.5, 0.75, 0.9])]
    lp_tw = np.array([x["pb"]["late_proxy"] for x in rows])
    out["late_proxy_twin_quantiles"] = [float(v) for v in np.quantile(lp_tw, [0.1, 0.25, 0.5, 0.75, 0.9])]
    thr = float(np.median(lp_real))
    for name, sel in (("twins_late_like_real", lp_tw <= thr), ("twins_not_late", lp_tw > thr)):
        T = [x for x, s in zip(rows, sel) if s]
        if T:
            ht = sum(x["pb"]["hit_B"] for x in T)
            hr = sum(x["real"]["hit_B"] for x in T)
            out[name] = {"threshold_real_median_late_proxy": thr, "n": len(T), "twin_hits_B": ht,
                         "twin_p": ht / len(T), "twin_wilson": wilson(ht, len(T)),
                         "real_hits_B_same_births": hr,
                         "twin_B_median": float(np.median([x["pb"]["B"] for x in T]))}
    (M2 / "read/r5_twins.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
