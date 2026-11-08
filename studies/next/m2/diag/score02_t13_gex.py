"""score-02 stopped at t13_G_EX (p99 0.05433 against 0.044273). Laptop, banked arrays and the job's metrics.json only;
no rank file, raw example or vector of score-02 is opened.

Questions: (1) does the gate's record look like the device difference or like a wrong forward? (2) how far is the
observed p99 inside the risk model of A1 (each entry's d drawn from the T_14 device pair in its |a_bank| bin)?
(3) how do the 717 entries at T_13 compare with the 477 at T_14 in |a_bank|?

    PYTHONPATH=. python studies/next/m2/diag/score02_t13_gex.py <repeat exemplar_a.npz> <out.json>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np  # noqa: E402

from infra.pg import m2  # noqa: E402

BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid")
EDGES = np.array([0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, np.inf])
JOB = pathlib.Path("studies/next/m2/banks/score-02/metrics.json")


def main():
    rep_path, out = sys.argv[1], sys.argv[2]
    assert m2.sha256_file(rep_path) == m2.REPEAT_T14["sha256"]
    jm = json.loads(JOB.read_text())
    ex = {t: np.load(BANK / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"].astype(np.float64) for t in (13, 14)}
    rep = np.load(rep_path)["a"].astype(np.float64)
    d_pair = m2.mixed_diff(rep.ravel(), ex[14].ravel())
    ad_pair = np.abs(rep.ravel() - ex[14].ravel())
    b_pair = np.digitize(np.abs(ex[14].ravel()), EDGES) - 1
    births = json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"]
    pl = m2.plan(births)
    um = {u["unit"]: u for u in pl["units"]}
    res = {"what": __doc__.split("\n")[0], "tol_cross": m2.TOL["cross"]["p99"], "job": {}, "points": {}}
    for t in (14, 13):
        g = jm["gates"][f"t{t:02d}_G_EX"]
        res["job"][t] = {k: g[k] for k in ("pass", "n", "n_poles", "median", "p99", "max", "exact_share")}
        res["job"][t]["failing_input"] = {k: g["failing_input"][k] for k in ("pass", "median", "p99")}
        if "farthest" in g:
            f = g["farthest"]
            res["job"][t]["farthest"] = {**f, "abs_diff": abs(f["a_new"] - f["a_bank"]),
                                         "d": float(m2.mixed_diff(f["a_new"], f["a_bank"]))}
        e = sorted({(um[uid]["row"], r) for uid, _ in pl["points"][t]["roles"] for r in um[uid]["ranks"]})
        rr, cc = np.array([x[0] for x in e]), np.array([x[1] for x in e])
        ab = np.abs(ex[t][rr, cc])
        bb = np.digitize(ab, EDGES) - 1
        res["points"][t] = {"entries": len(e), "median_abs_bank": float(np.median(ab)),
                            "bin_counts": {f"{EDGES[i]}-{EDGES[i + 1]}": int((bb == i).sum()) for i in range(len(EDGES) - 1)},
                            "share_below_0.1": float((ab < 0.1).mean()),
                            "entries_above_tol_needed_for_a_stop": int(len(e) - np.floor(0.99 * (len(e) - 1)) - 1)}
        rng = np.random.default_rng(0)
        pools = [d_pair[b_pair == i] for i in range(len(EDGES) - 1)]
        p99 = np.array([np.quantile(np.concatenate([rng.choice(pools[i], int((bb == i).sum())) for i in
                                                    range(len(EDGES) - 1)]), 0.99) for _ in range(2000)])
        obs = res["job"][t]["p99"]
        res["points"][t]["model"] = {"p99_median": float(np.median(p99)), "p99_q90": float(np.quantile(p99, 0.9)),
                                     "p99_q99": float(np.quantile(p99, 0.99)),
                                     "share_at_or_above_observed": float((p99 >= obs).mean()),
                                     "share_above_tol": float((p99 > m2.TOL["cross"]["p99"]).mean())}
    i = 3   # the bin 0.03-0.1 holds the farthest entry at T_13 (|a_bank| 0.031)
    m = b_pair == i
    res["pair_bin_0.03_0.1"] = {"n": int(m.sum()), "abs_diff_p99": float(np.quantile(ad_pair[m], 0.99)),
                                "abs_diff_max": float(ad_pair[m].max()), "d_p99": float(np.quantile(d_pair[m], 0.99)),
                                "d_max": float(d_pair[m].max())}
    res["pair_all"] = {"median_d": float(np.median(d_pair)), "exact_share": float((rep.ravel() == ex[14].ravel()).mean())}
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
