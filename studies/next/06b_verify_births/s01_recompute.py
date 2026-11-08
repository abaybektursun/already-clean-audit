"""Step 1. Recompute every cluster's presence, birth, main, late flags and status from the banked activations,
(i) with an implementation written here and (ii) with the repo's own cluster_table / birth_order_test, and compare
both with the banked rows in q2_clusters.jsonl and with metrics.json q2.birth_order and q2.births.
Outputs: out/s01_recompute.json, out/s01_recompute.txt."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/abaybektursun/projects/interp_research")
import infra.pg.neurons_verdict as V   # the code under audit, used only for the cross-check (ii)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
z = np.load(OUT / "inputs.npz", allow_pickle=False)
poles, A, Q999, MAX = list(z["poles"]), z["a"], z["q999"], z["max_abs"]
row = {p: i for i, p in enumerate(poles)}
clusters = json.load(open(OUT / "clusters.json"))
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json"))}
metrics = json.load(open(BANK / "w5/verdict/metrics.json"))["q2"]
T = 16


def mag_of(pole):
    s = 1.0 if pole.endswith("+") else -1.0
    return np.maximum(s * A[:, row[pole], :], 0.0)          # a+ = max(s*a, 0), [16, 50]


def first_birth(p):
    for t in range(T):
        if p[t] and (t == T - 1 or p[t + 1]):
            return t
    return None


def mine(pole):
    """Independent implementation of SPEC_D1 §2.4 / §6.3 statuses."""
    mag, i = mag_of(pole), row[pole]
    others = [set(c["exemplar_ranks"]) for c in clusters[pole]["1"] if not c["echo"]]
    out = []
    for c in clusters[pole]["0"]:
        rk = c["exemplar_ranks"]
        med = np.median(mag[:, rk], axis=1)
        pct = (med > 0) & (med >= Q999[:, i])
        raw = (med > 0) & (med >= 0.3 * MAX[:, i])
        out.append(dict(cluster=f"{pole}:c{c['c']}", c=c["c"], echo=c["echo"], med=med, birth_pct=first_birth(pct),
                        birth_raw=first_birth(raw), peak_end=float(mag[15, rk].max()),
                        matched=any(2 * len(set(rk) & o) > len(rk) for o in others), is_main=False))
    jobs = [r for r in out if not r["echo"]]
    if not jobs:
        return out
    top = max(r["peak_end"] for r in jobs)
    for r in jobs:
        r["is_main"] = r["c"] == min(x["c"] for x in jobs if x["peak_end"] == top)
    bp = min((r["birth_pct"] for r in jobs if r["birth_pct"] is not None), default=None)
    br = min((r["birth_raw"] for r in jobs if r["birth_raw"] is not None), default=None)
    for r in jobs:
        r["late_pct"] = r["birth_pct"] is not None and r["birth_pct"] > bp
        r["late_raw"] = r["birth_raw"] is not None and r["birth_raw"] > br
        r["status"] = ("never_born" if r["birth_pct"] is None else "earliest" if not r["late_pct"] else
                       "formed_before_step_1000" if r["birth_pct"] == 1 else "readings_disagree" if not r["late_raw"]
                       else "unmatched_in_sample_1" if not r["matched"] else "late_primary")
    return out


FIELDS = ("birth_pct", "birth_raw", "is_main", "late_pct", "late_raw", "status", "matched")
mism = {"mine": {f: 0 for f in FIELDS}, "repo": {f: 0 for f in FIELDS}}
med_diff = {"mine": 0.0, "repo": 0.0}
n_rows, n_jobs = 0, 0
order_mine, order_repo = [], []
hist = {"main": np.zeros(T, int), "non_main": np.zeros(T, int)}
per_cluster = {}
for pole in sorted(clusters):
    i = row[pole]
    rep = V.cluster_table(pole, clusters[pole]["0"], clusters[pole]["1"], V.signed_magnitude(A[:, i, :], 0 if pole.endswith("+") else 1),
                          list(Q999[:, i]), list(MAX[:, i]))
    for src, rows_ in (("mine", mine(pole)), ("repo", rep)):
        for r in rows_:
            b = banked[r["cluster"]]
            med_diff[src] = max(med_diff[src], float(np.abs(np.asarray(r["med"]) - np.asarray(b["med"])).max()))
            if r["echo"]:
                assert b["echo"]
                continue
            for f in FIELDS:
                mism[src][f] += r[f] != b[f]
        jobs = [r for r in rows_ if not r["echo"]]
        main = [r for r in jobs if r["is_main"]]
        if main and main[0]["birth_pct"] is not None:
            born = [r for r in jobs if r["birth_pct"] is not None]
            if len(born) >= 2:
                (order_mine if src == "mine" else order_repo).append(([r["birth_pct"] for r in born], born.index(main[0])))
    for r in mine(pole):
        n_rows += 1
        if not r["echo"]:
            n_jobs += 1
            hist["main" if r["is_main"] else "non_main"][r["birth_pct"]] += 1
            per_cluster[r["cluster"]] = {k: r[k] for k in ("birth_pct", "birth_raw", "is_main", "status")}

# Broken-input check: the comparison must report mismatches when the rule is wrong. Two wrong rules:
# (a) birth = first t present (no t+1 requirement); (b) the threshold read one grid point late (q999 of t+1).
broken = {"no_next_point_rule": 0, "q999_shifted_one_point": 0}
for pole in sorted(clusters):
    i, mag = row[pole], mag_of(pole)
    q_shift = np.r_[Q999[1:, i], Q999[-1, i]]
    for c in clusters[pole]["0"]:
        if c["echo"]:
            continue
        med = np.median(mag[:, c["exemplar_ranks"]], axis=1)
        pct = (med > 0) & (med >= Q999[:, i])
        b_a = next((t for t in range(T) if pct[t]), None)
        b_b = first_birth((med > 0) & (med >= q_shift))
        bb = banked[f"{pole}:c{c['c']}"]["birth_pct"]
        broken["no_next_point_rule"] += b_a != bb
        broken["q999_shifted_one_point"] += b_b != bb

hits = [b[m] == min(b) for b, m in order_mine]
b_mine = float(np.mean(hits))
bo = V.birth_order_test(order_repo, V.N_PERM, 0)
mb = metrics["birth_order"]
res = {"broken_input_birth_pct_mismatches": broken, "n_rows": n_rows, "n_non_echo": n_jobs, "mismatches_vs_banked": mism, "max_abs_med_diff_vs_banked": med_diff,
       "b_mine": b_mine, "n_poles_b_mine": len(order_mine), "birth_order_repo_recomputed": bo,
       "birth_order_metrics_json": mb,
       "births_hist_recomputed": {k: v.tolist() for k, v in hist.items()},
       "births_hist_metrics_json": {k: metrics["births"][k]["counts"] for k in ("main", "non_main")},
       "status_counts_recomputed": {s: sum(v["status"] == s for v in per_cluster.values()) for s in
                                    ("never_born", "earliest", "formed_before_step_1000", "readings_disagree",
                                     "unmatched_in_sample_1", "late_primary")},
       "status_counts_metrics_json": metrics["status_counts"]}
json.dump(res, open(OUT / "s01_recompute.json", "w"), indent=1)
lines = [f"rows {n_rows}, non-echo {n_jobs}",
         f"mismatches vs q2_clusters.jsonl (non-echo rows), independent code: {mism['mine']}",
         f"mismatches vs q2_clusters.jsonl (non-echo rows), repo cluster_table: {mism['repo']}",
         f"max |med - banked med|: {med_diff}",
         f"broken-input check, birth_pct mismatches under deliberately wrong rules: {broken}",
         f"b independent: {b_mine:.6f} on {len(order_mine)} poles",
         f"b repo birth_order_test recomputed: b {bo['b']:.6f} ci {bo['ci95']} n {bo['n_poles']} perm mean {bo['null_mean']:.6f} perm 97.5 {bo['null_97_5']:.6f}",
         f"b metrics.json: b {mb['b']:.6f} ci {mb['ci95']} n {mb['n_poles']} perm mean {mb['null_mean']:.6f} perm 97.5 {mb['null_97_5']:.6f}",
         f"births hist recomputed == metrics.json: {res['births_hist_recomputed'] == res['births_hist_metrics_json']}",
         f"status counts recomputed: {res['status_counts_recomputed']}",
         f"status counts metrics.json: {res['status_counts_metrics_json']}"]
(OUT / "s01_recompute.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
