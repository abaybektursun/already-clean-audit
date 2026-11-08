"""score-03 stopped at t13_G_EX under A2 (591 gated entries, p99 0.05191 against 0.038718). Laptop, banked arrays and
the job's metrics.json only; no rank file, raw example or vector of score-03 is opened.

(1) How the two A2 risk models placed the observed value: share of 2,000 draws of T_13's gated entries with p99 at or
above it (bin model: the T_14 pair's gated entries by |a_bank| bin; plan pool: the plan's own gated entries at T_14).
(2) Which grid poles hold the farthest entries named by the jobs (rows 595 and 581) and the four T_14 entries above the
A2 tolerance (rows 307, 581, 581, 595), and their share of T_13's gated entries.
(3) Depth: the pair's statistic by layer group, the layer mix of the gated entries at T_13 and T_14, and the layers of
the four T_14 entries above the tolerance.

    PYTHONPATH=. python studies/next/m2/diag/score03_t13_gex.py <repeat exemplar_a.npz> <out.json>
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
    g = json.loads(pathlib.Path("studies/next/m2/banks/score-03/metrics.json").read_text())["gates"]["t13_G_EX"]
    tol = m2.TOL["ex_cross"]
    ex = {t: np.load(BANK / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"].astype(np.float64) for t in (13, 14)}
    rep = np.load(rep_path)["a"].astype(np.float64)
    ab = np.abs(ex[14].ravel())
    keep = ab >= tol["abs_min"]
    d = m2.mixed_diff(rep.ravel(), ex[14].ravel())[keep]
    b = np.digitize(ab[keep], EDGES) - 1
    pools = [d[b == i] for i in range(len(EDGES) - 1)]
    pl = m2.plan(json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"])
    um = {u["unit"]: u for u in pl["units"]}

    def ent(t):
        e = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][t]["roles"] for r in um[uid]["ranks"]})
        return np.array([x[0] for x in e]), np.array([x[1] for x in e])
    r14, c14 = ent(14)
    k14 = np.abs(ex[14][r14, c14]) >= tol["abs_min"]
    plan_pool = m2.mixed_diff(rep[r14, c14], ex[14][r14, c14])[k14]
    r13, c13 = ent(13)
    a13 = np.abs(ex[13][r13, c13])
    k13 = a13 >= tol["abs_min"]
    bb = np.digitize(a13[k13], EDGES) - 1
    rng = np.random.default_rng(0)
    pb = np.array([np.quantile(np.concatenate([rng.choice(pools[i], int((bb == i).sum())) for i in
                                               range(len(EDGES) - 1)]), 0.99) for _ in range(2000)])
    pp = np.array([np.quantile(rng.choice(plan_pool, int(k13.sum())), 0.99) for _ in range(2000)])
    n = int(k13.sum())
    res = {"what": __doc__.split("\n")[0], "observed": {k: g[k] for k in ("n", "n_gated", "median", "p99", "max",
                                                                           "exact_share", "below_abs_min", "farthest")},
           "tol": tol, "gated_entries_above_tol_needed_for_a_stop": int(n - np.floor(0.99 * (n - 1)) - 1),
           "bin_model": {"p99_median": float(np.median(pb)), "share_ge_observed": float((pb >= g["p99"]).mean())},
           "plan_pool": {"p99_median": float(np.median(pp)), "share_ge_observed": float((pp >= g["p99"]).mean())}}
    poles = [str(p) for p in np.load(BANK.parent.parent / "next/recount2/arrays.npz")["poles"]]
    units_of = {}
    for u in pl["units"]:
        units_of.setdefault(u["row"], []).append([u["unit"], u["kind"], u["roles"]])
    res["rows"] = {}
    for row in (595, 581, 307):
        m13 = (r13 == row) & k13
        res["rows"][row] = {"pole": poles[row], "units": units_of.get(row, []),
                            "t13_gated_entries": int(m13.sum()), "t13_abs_bank": np.round(a13[m13], 4).tolist()}
    # (3) depth: the pair's statistic by layer group (gated entries), and the layer mix of the gated entries the gate read
    lay = np.array([m2.parse_pole(p)[0] for p in poles])
    lay_all = np.repeat(lay, ex[14].shape[1])[keep]
    groups = [(0, 7), (8, 15), (16, 23), (24, 31)]
    res["pair_by_layer_group"] = {f"L{lo:02d}-L{hi:02d}": {"n": int(((lay_all >= lo) & (lay_all <= hi)).sum()),
                                  "p99": float(np.quantile(d[(lay_all >= lo) & (lay_all <= hi)], 0.99)),
                                  "share_above_tol": float((d[(lay_all >= lo) & (lay_all <= hi)] > tol["p99"]).mean())}
                                  for lo, hi in groups}
    for t, (rr, kk) in ((13, (r13, k13)), (14, (r14, k14))):
        lt = lay[rr][kk]
        res[f"t{t}_gated_layer_mix"] = {f"L{lo:02d}-L{hi:02d}": int(((lt >= lo) & (lt <= hi)).sum()) for lo, hi in groups}
    lt14 = lay[r14][k14]
    res["t14_plan_above_tol_layers"] = sorted(lt14[plan_pool > tol["p99"]].tolist())
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
