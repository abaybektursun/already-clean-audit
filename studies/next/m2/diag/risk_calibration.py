"""Calibration of the two risk models on the one out-of-sample value banked: score-02's T_13 G-EX under the
freeze-1c statistic (all 717 entries, p99 0.05433, studies/next/m2/banks/score-02/metrics.json). Bin model: each entry
draws d from the T_14 device pair in its |a_bank| bin (a1_risk.py). Plan pool: each entry draws d from the plan's own
477 entries at T_14 (repeat against bank). Share of 2,000 draws with p99 >= the observed value. Laptop, banked only.

    PYTHONPATH=. python studies/next/m2/diag/risk_calibration.py <repeat exemplar_a.npz> <out.json>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np  # noqa: E402

from infra.pg import m2  # noqa: E402

BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid")
EDGES = np.array([0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, np.inf])


def main():
    rep_path, out = sys.argv[1], sys.argv[2]
    assert m2.sha256_file(rep_path) == m2.REPEAT_T14["sha256"]
    obs = json.loads(pathlib.Path("studies/next/m2/banks/score-02/metrics.json").read_text())["gates"]["t13_G_EX"]
    ex = {t: np.load(BANK / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"].astype(np.float64) for t in (13, 14)}
    rep = np.load(rep_path)["a"].astype(np.float64)
    d = m2.mixed_diff(rep.ravel(), ex[14].ravel())
    b = np.digitize(np.abs(ex[14].ravel()), EDGES) - 1
    pools = [d[b == i] for i in range(len(EDGES) - 1)]
    pl = m2.plan(json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"])
    um = {u["unit"]: u for u in pl["units"]}

    def ent(t):
        e = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][t]["roles"] for r in um[uid]["ranks"]})
        return np.array([x[0] for x in e]), np.array([x[1] for x in e])
    r14, c14 = ent(14)
    plan_pool = m2.mixed_diff(rep[r14, c14], ex[14][r14, c14])
    r13, c13 = ent(13)
    bb = np.digitize(np.abs(ex[13][r13, c13]), EDGES) - 1
    rng = np.random.default_rng(0)
    pb = np.array([np.quantile(np.concatenate([rng.choice(pools[i], int((bb == i).sum())) for i in
                                               range(len(EDGES) - 1)]), 0.99) for _ in range(2000)])
    pp = np.array([np.quantile(rng.choice(plan_pool, r13.size), 0.99) for _ in range(2000)])
    res = {"what": __doc__.split("\n")[0], "observed_t13_p99": obs["p99"], "n_t13": int(r13.size),
           "bin_model": {"p99_median": float(np.median(pb)), "share_ge_observed": float((pb >= obs["p99"]).mean())},
           "plan_pool": {"n_pool": int(plan_pool.size), "p99_median": float(np.median(pp)),
                         "share_ge_observed": float((pp >= obs["p99"]).mean())}}
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
