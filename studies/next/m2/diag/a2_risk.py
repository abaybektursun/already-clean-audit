"""Amendment A2 (2026-11-06), exploratory, not a rule: how likely the A2 G-EX gate is to stop score-03 at the
Blackwell points. The model of a1_risk.py restricted to the entries A2 gates: each G-EX entry of point t with
|a_bank| >= m2.EX_ABS_MIN draws its statistic from the T_14 device pair's entries with |a_bank| >= EX_ABS_MIN in the
same |a_bank| bin; 2,000 draws per point; the share of draws whose p99 exceeds m2.TOL["ex_cross"], per point and
draw by draw over T_7 ... T_13. T_14 also gets the pair's actual value on the plan's entries (the job reproduced the
repeat to every digit at T_14 in score-01 and score-02). Laptop, banked arrays only; it decides nothing.
A second, empirical model ("plan_pool"): the pool is the statistic of the plan's own gated entries at T_14 (the repeat
against the bank; entries of the same kind as the other points', pre-birth and birth-point exemplars), resampled to
each point's gated count. The bin model understated T_14 in every version (A1, its revision, A2), so both are shown.

    PYTHONPATH=. python studies/next/m2/diag/a2_risk.py <repeat exemplar_a.npz> <out.json>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np  # noqa: E402

from infra.pg import m2  # noqa: E402

BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid")
EDGES = np.array([0.1, 0.3, 1, 3, np.inf])


def main():
    rep_path, out = sys.argv[1], sys.argv[2]
    assert m2.sha256_file(rep_path) == m2.REPEAT_T14["sha256"]
    tol = m2.TOL["ex_cross"]
    ex = {t: np.load(BANK / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"].astype(np.float64) for t in range(7, 15)}
    rep = np.load(rep_path)["a"].astype(np.float64)
    ab = np.abs(ex[14].ravel())
    keep = ab >= tol["abs_min"]
    s = m2.mixed_diff(rep.ravel(), ex[14].ravel())[keep]
    bins = np.digitize(ab[keep], EDGES) - 1
    pools = [s[bins == i] for i in range(len(EDGES) - 1)]
    res = {"what": __doc__.split("\n")[0], "tol": tol, "pool": [{"bin": [float(EDGES[i]), float(EDGES[i + 1])],
           "n": int(p.size), "p99": float(np.quantile(p, 0.99))} for i, p in enumerate(pools)], "points": {}}
    births = json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"]
    pl = m2.plan(births)
    um = {u["unit"]: u for u in pl["units"]}
    e14 = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][14]["roles"] for r in um[uid]["ranks"]})
    r14, c14 = np.array([x[0] for x in e14]), np.array([x[1] for x in e14])
    k14 = np.abs(ex[14][r14, c14]) >= tol["abs_min"]
    plan_pool = m2.mixed_diff(rep[r14, c14], ex[14][r14, c14])[k14]
    res["plan_pool"] = {"n": int(plan_pool.size), "above_tol": int((plan_pool > tol["p99"]).sum()),
                        "share_above_tol": float((plan_pool > tol["p99"]).mean()),
                        "pair_share_above_tol": float((s > tol["p99"]).mean())}
    rng = np.random.default_rng(0)
    pass_draws = np.ones(2000, bool)
    pass_draws_plan = np.ones(2000, bool)
    for t in range(7, 15):
        e = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][t]["roles"] for r in um[uid]["ranks"]})
        rr, cc = np.array([x[0] for x in e]), np.array([x[1] for x in e])
        a = np.abs(ex[t][rr, cc])
        g = a >= tol["abs_min"]
        bb = np.digitize(a[g], EDGES) - 1
        p99 = np.array([np.quantile(np.concatenate([rng.choice(pools[i], int((bb == i).sum())) for i in
                                                    range(len(EDGES) - 1)]), 0.99) for _ in range(2000)])
        q99 = np.array([np.quantile(rng.choice(plan_pool, int(g.sum())), 0.99) for _ in range(2000)])
        row = {"entries": len(e), "gated": int(g.sum()), "below": int((~g).sum()),
               "p99_model_median": float(np.median(p99)), "p99_model_q90": float(np.quantile(p99, 0.9)),
               "share_above_tol": float((p99 > tol["p99"]).mean()),
               "plan_pool_p99_median": float(np.median(q99)), "plan_pool_share_above_tol": float((q99 > tol["p99"]).mean())}
        if t == 14:
            row["p99_actual_repeat"] = m2.values_gate(rep[rr, cc], ex[14][rr, cc], tol)["p99"]
        else:
            pass_draws &= p99 <= tol["p99"]
            pass_draws_plan &= q99 <= tol["p99"]
        res["points"][t] = row
    res["share_draws_pass_all_T7_to_T13"] = float(pass_draws.mean())
    res["plan_pool_share_draws_pass_all_T7_to_T13"] = float(pass_draws_plan.mean())
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
