"""Cause of the score-01 stop (2026-11-06): t14_G_EX failed (p99 0.2563 against the cross-device 0.0444).

Laptop, banked arrays only. Rebuilds the 477 (grid row, T_15 rank) entries the job compared at T_14 (`m2.plan`,
the same dedup as `m2_score.ex_entries`) and compares two banks on them:
  1b: the banked T_14 exemplar_a (RTX PRO 5000 Blackwell), the bank G-EX reads;
  1c: the T_14 repeat on an L40S (s3 results/neurons/w3/grid/1c/out/t14_step1000000/poles/exemplar_a.npz, the device
      check's second side).
If 1c against 1b on these entries gives the job's numbers to every digit, the job's L40S forward equals the banked
L40S repeat there, and the failure is the cross-device difference on this subset, not a wrong activation.

    PYTHONPATH=. python studies/next/m2/diag/score01_t14_gex.py <1c exemplar_a.npz> <out.json>   (from the worktree root)
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np  # noqa: E402

from infra.pg import m2  # noqa: E402

BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid")
JOB = {"n": 477, "n_poles": 62, "median_rel": 0.0052419216385804785, "p99_rel": 0.2562705341026918,
       "max_rel": 1.952766309595231, "exact_share": 0.32914046121593293}   # banks/score-01/metrics.json, t14_G_EX


def entries_at(births, t):
    pl = m2.plan(births)
    units = {u["unit"]: u for u in pl["units"]}
    seen = set()
    for uid, _role in pl["points"][t]["roles"]:
        u = units[uid]
        for r in u["ranks"]:
            seen.add((u["row"], r))
    return sorted(seen)


def main():
    c_path, out = sys.argv[1], sys.argv[2]
    births = json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"]
    ent = entries_at(births, 14)
    rows = np.array([e[0] for e in ent])
    cols = np.array([e[1] for e in ent])
    b = np.load(BANK / m2.POINTS[14][1] / "poles" / "exemplar_a.npz")["a"]
    c = np.load(c_path)["a"]
    g = m2.values_gate(c[rows, cols], b[rows, cols], m2.TOL["cross"], 1, len(set(rows.tolist())))
    same = {k: (g[k] == JOB[k]) for k in JOB}
    allg = m2.values_gate(c.ravel(), b.ravel(), m2.TOL["cross"])
    r = m2.rel_diff(c[rows, cols], b[rows, cols])
    den = np.maximum(np.abs(c[rows, cols]), np.abs(b[rows, cols]))
    rall = m2.rel_diff(c.ravel(), b.ravel())
    dall = np.maximum(np.abs(c.ravel()), np.abs(b.ravel()))
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    by_size = {}
    for lo, hi in [(0, 0.01), (0.01, 0.1), (0.1, np.inf)]:
        m_s, m_a = (den >= lo) & (den < hi), (dall >= lo) & (dall < hi)
        by_size[f"[{lo}, {hi})"] = {
            "subset_n": int(m_s.sum()), "subset_p99": float(np.quantile(r[m_s], 0.99)) if m_s.any() else None,
            "all_n": int(m_a.sum()), "all_p99": float(np.quantile(rall[m_a], 0.99)) if m_a.any() else None}
    res = {"what": __doc__.split("\n")[0], "entries": len(ent),
           "1c_vs_1b_on_job_entries": g, "equal_to_job_digit_for_digit": same,
           "1c_vs_1b_all_32000": allg,
           "abs_a_quantiles_10_25_50_75_90": {"job_entries": np.quantile(den, q).tolist(), "all_32000": np.quantile(dall, q).tolist()},
           "p99_by_size_of_max_abs_pair": by_size}
    # The same subsets at the other points, from the Blackwell or L40S bank alone (no repeat exists there): the share
    # of G-EX entries with |a_bank| below 0.01, where the relative difference at T_14 reached p99 1.86.
    small = {}
    for t in range(5, 16):
        e = entries_at(births, t)
        a = np.abs(np.load(BANK / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"][[x[0] for x in e], [x[1] for x in e]])
        small[t] = {"bank_device": m2.POINTS[t][4], "entries": len(e), "share_below_0.01": float((a < 0.01).mean()),
                    "share_below_0.1": float((a < 0.1).mean()), "median_abs": float(np.median(a))}
    res["small_share_by_point"] = small
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"entries": len(ent), "subset": {k: g[k] for k in JOB}, "equal": same,
                      "all": {k: allg[k] for k in ("n", "median_rel", "p99_rel", "max_rel", "exact_share")}}, indent=1))


if __name__ == "__main__":
    main()
