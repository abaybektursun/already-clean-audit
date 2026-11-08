"""Amendment A1 (2026-11-05), exploratory, not a rule: how likely G-EX is to stop score-02 at the other Blackwell points.

Laptop, banked arrays only. The only cross-device pair banked is T_14 (Blackwell bank, L40S repeat, m2.REPEAT_T14).
At T_14 the job's L40S forward reproduced the repeat digit for digit (score-01), so the plan's 477 entries there give
p99 0.0403 under A1, against the tolerance 0.0423: a pass known in advance. At T_7 ... T_13 no repeat exists. The model:
the statistic of an entry depends on |a_bank| only; each G-EX entry of point t draws its statistic from the T_14 pair's
32,000 entries in the same |a_bank| bin; 2,000 draws per point give the distribution of the p99 and the share above the
tolerance. A crude model (one checkpoint's device noise reused at the others); it decides nothing.
Revision of A1 (head, 2026-11-05 20:50 CST): the cross-device tolerance is m2.TOL["cross"] (twice the pair's p99 over
|a_bank| < 0.5, 0.044273); each point also reports the share above the freeze-1b tolerance (2 * m2.CTRL_P99, 0.042318),
and the share of draws passing at all seven points T_7 ... T_13 is counted draw by draw (the draws are independent
across points, so it matches the product of the per-point shares within sampling error).
Optional: topk layer files of T_14 (S3 1b/out/t14_step1000000/topk/layer_LL.npz) give the |a| of the banked top-50
entries G-TOPK compares at the plan's captured positions.

    PYTHONPATH=. python studies/next/m2/diag/a1_risk.py <repeat exemplar_a.npz> <out.json> [topk layer_LL.npz ...]
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
    ex = {t: np.load(BANK / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"].astype(np.float64) for t in range(5, 16)}
    rep = np.load(rep_path)["a"].astype(np.float64)
    s = m2.mixed_diff(rep.ravel(), ex[14].ravel())
    ab = np.abs(ex[14].ravel())
    bins = np.digitize(ab, EDGES) - 1
    tol = m2.TOL["cross"]["p99"]
    res = {"what": __doc__.split("\n")[0], "tol_cross": tol, "by_abs_bank_bin": [], "points": {}}
    for i in range(len(EDGES) - 1):
        m = bins == i
        res["by_abs_bank_bin"].append({"bin": [float(EDGES[i]), float(EDGES[i + 1])], "n": int(m.sum()),
                                       "median": float(np.median(s[m])), "p99": float(np.quantile(s[m], 0.99)),
                                       "max": float(s[m].max()),
                                       "abs_diff_p99": float(np.quantile(np.abs(rep.ravel() - ex[14].ravel())[m], 0.99))})
    births = json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"]
    pl = m2.plan(births)
    um = {u["unit"]: u for u in pl["units"]}
    rng = np.random.default_rng(0)
    pools = [s[bins == i] for i in range(len(EDGES) - 1)]
    p_pass_all = 1.0
    tol_1b = 2 * m2.CTRL_P99
    pass_draws = np.ones(2000, bool)
    for t in range(7, 15):
        e = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][t]["roles"] for r in um[uid]["ranks"]})
        rr, cc = np.array([x[0] for x in e]), np.array([x[1] for x in e])
        bb = np.digitize(np.abs(ex[t][rr, cc]), EDGES) - 1
        p99 = np.array([np.quantile(np.concatenate([rng.choice(pools[i], int((bb == i).sum())) for i in
                                                    range(len(EDGES) - 1)]), 0.99) for _ in range(2000)])
        row = {"entries": len(e), "share_abs_below_0.1": float((np.abs(ex[t][rr, cc]) < 0.1).mean()),
               "p99_model_median": float(np.median(p99)), "p99_model_q90": float(np.quantile(p99, 0.9)),
               "share_above_tol": float((p99 > tol).mean()),
               "share_above_tol_freeze_1b": float((p99 > tol_1b).mean())}
        if t == 14:
            row["p99_actual_repeat"] = m2.values_gate(rep[rr, cc], ex[14][rr, cc], m2.TOL["cross"])["p99"]
        else:
            p_pass_all *= 1 - row["share_above_tol"]
            pass_draws &= p99 <= tol
        res["points"][t] = row
    res["model_pass_all_T7_to_T13"] = p_pass_all
    res["share_draws_pass_all_T7_to_T13"] = float(pass_draws.mean())
    topk = {}
    for f in sys.argv[3:]:
        lyr = int(pathlib.Path(f).stem.split("_")[1])
        z = np.load(f)
        caps = sorted({e for w in pl["points"][14]["need"].values() for lv, _, e in w if lv == lyr})
        v = np.abs(z["a"][np.isin(z["tok"], caps)])
        topk[lyr] = {"captured_positions": len(caps), "topk_entries": int(v.size),
                     "abs_a_min_q01_q50": np.quantile(v, [0, 0.01, 0.5]).tolist(), "share_below_0.1": float((v < 0.1).mean())}
    res["g_topk_t14_layers"] = topk
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"points": {t: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                                 for t, r in res["points"].items()}, "pass_all_T7_T13": round(p_pass_all, 3),
                      "draws_pass_all_T7_T13": res["share_draws_pass_all_T7_to_T13"], "tol_cross": tol,
                      "topk": topk}, indent=1))


if __name__ == "__main__":
    main()
