"""Written AFTER reading 06b_VERIFY_BIRTHS.md and 06_CRITIC.md Part 0 (2026-11-05), for the comparison table only.
It changes no number of sections 1-10. It re-slices the per-cluster test onto the earlier computations' subsets:
(a) all 2,428 clusters of the 574 poles, (b) the clusters late-born by the percentile reading, (c) the banked
late_primary clusters (status in q2_clusters.jsonl). Reference R1 (all labels shuffled, the earlier computations'
design) with a fresh seed, and R1t from 05_per_cluster_R1t.csv. Two p conventions: fraction of draws >= real (mine),
and (1 + count) / (1 + draws) < 0.05 (the verifier's). Chance expectation: unselected (sum of per-cluster
P(p <= 0.05)) and selection-matched for (b) (pass share among late-born random groups times the real count).
Output: out/07_like_for_like.json.
"""
import json, os, csv
import numpy as np
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
A = C.load_arrays()
REPS, ALPHA = 2000, 0.05
rng = np.random.default_rng(7001)
status = {}
for l in open(C.BANKED):
    r = json.loads(l)
    status[(r["pole"], r["c"])] = r["status"]
mine = [json.loads(l) for l in open(C.BIG + "/births.jsonl")]
poles = {}
for l in mine:
    if l["sample"] == 0:
        poles.setdefault(l["pole"], []).append(l)
P = [p for p in sorted(poles) if len(poles[p]) >= 2]
rows = []
sel_pass, sel_n = 0.0, 0
for p in P:
    cl = poles[p]; i = A["row"][p]; m = len(cl)
    ne = np.array(sorted(r for c in cl for r in c["ranks"]))
    pos = {r: k for k, r in enumerate(ne)}
    lab = np.empty(len(ne), dtype=np.int64)
    for j, c in enumerate(cl):
        for r in c["ranks"]:
            lab[pos[r]] = j
    newlab = lab[np.argsort(rng.random((REPS, len(ne))), axis=1)]
    order = np.argsort(newlab, axis=1, kind="stable")
    counts = np.bincount(lab, minlength=m); starts = np.concatenate([[0], np.cumsum(counts)])
    B = np.zeros((REPS, m), dtype=np.int64)
    for j in range(m):
        B[:, j] = C.births_matrix(A["a_plus"][:, i, :], A["q999"][:, i], ne[order[:, starts[j]:starts[j + 1]]])
    real = np.array([c["birth_pct"] for c in cl])
    cols = [np.sort(B[:, j]) for j in range(m)]
    def pl(j, x):
        return 1.0 - np.searchsorted(cols[j], x, side="left") / REPS
    # selection-matched chance for "late-born by the percentile reading"
    late_rand = B > B.min(axis=1, keepdims=True)
    for j in range(m):
        xs = B[late_rand[:, j], j]
        if len(xs):
            ge = 1.0 - np.searchsorted(cols[j], xs, side="left") / REPS
            sel_pass += float((ge <= ALPHA).sum()); sel_n += len(xs)
    for j, c in enumerate(cl):
        cnt_ge = int((B[:, j] >= real[j]).sum())
        vals, cn = np.unique(cols[j], return_counts=True)
        rows.append({"pole": p, "c": c["c"], "late_pct": bool(real[j] > real.min()),
                     "status": status[(p, c["c"])], "p_late": float(pl(j, real[j])),
                     "p_late_plus1": (1 + cnt_ge) / (1 + REPS),
                     "exp_late": float(sum(n for v, n in zip(vals, cn) if pl(j, v) <= ALPHA) / REPS)})
r1t = {(r["pole"], int(r["c"])): float(r["p_late"]) for r in csv.DictReader(open(OUT + "/05_per_cluster_R1t.csv"))}
r1t_exp = {(r["pole"], int(r["c"])): float(r["exp_late"]) for r in csv.DictReader(open(OUT + "/05_per_cluster_R1t.csv"))}
res = {"seed": 7001, "reps": REPS, "selection_matched_pass_share_late_born": sel_pass / sel_n}
for name, f in [("all", lambda r: True), ("late_born_pct", lambda r: r["late_pct"]),
                ("late_primary", lambda r: r["status"] == "late_primary")]:
    rr = [r for r in rows if f(r)]
    res[name] = {"n": len(rr),
                 "R1_beat_frac_le_005": sum(r["p_late"] <= ALPHA for r in rr),
                 "R1_beat_plus1_lt_005": sum(r["p_late_plus1"] < ALPHA for r in rr),
                 "R1_expected_unselected": round(sum(r["exp_late"] for r in rr), 1),
                 "R1t_beat_frac_le_005": sum(r1t[(r["pole"], r["c"])] <= ALPHA for r in rr),
                 "R1t_expected_unselected": round(sum(r1t_exp[(r["pole"], r["c"])] for r in rr), 1)}
res["late_born_pct"]["R1_expected_selection_matched"] = round(res["selection_matched_pass_share_late_born"]
                                                              * res["late_born_pct"]["n"], 1)
json.dump(res, open(OUT + "/07_like_for_like.json", "w"), indent=1)
print(json.dumps(res, indent=1))
