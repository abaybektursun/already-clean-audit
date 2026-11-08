"""N1 independent read, step 5 (e): how much the input-side label rests on single births, and whether the carriers'
larger activation size at T_(k-1) (N12's ranks, value.maxabs at the 'pre' role) accounts for the paired difference.

    python3 -I fragility.py --units-recount studies/next/n1/read_n1/units_recount.json \
        --n12-ranks-test studies/next/n12/banks/score-02/n12/ranks_test.jsonl \
        --n12-ranks-freeze studies/next/n12/banks/score-02/n12/ranks_freeze.jsonl --out studies/next/n1/read_n1/fragility.json
"""
import argparse
import json
import pathlib

import numpy as np
from scipy import stats as st


def boot_hi_lo(x, seed=0, n_boot=10000):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    for a in ("units-recount", "n12-ranks-test", "n12-ranks-freeze", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    U = json.loads(pathlib.Path(a.units_recount).read_text())
    mx = {}
    for p in (a.n12_ranks_test, a.n12_ranks_freeze):
        for line in pathlib.Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["role"] == "pre":
                mx[r["unit"]] = r["value"]["maxabs"]
    out = {}
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        rb = {u["birth"]: u for u in H if u["kind"] == "real"}
        td = {u["birth"]: u for u in H if u["kind"] == "dt"}
        ids = sorted(b for b in rb if b in td)
        o = {}
        for s in ("in", "out"):
            d = np.array([rb[b][s]["e_window"] - td[b][s]["e_window"] for b in ids])
            if half == "test" and s == "in":
                # drop one birth at a time: the registered percentile interval (10,000, seed 0) and the t interval
                rows = []
                for i, b in enumerate(ids):
                    x = np.delete(d, i)
                    lo, hi = boot_hi_lo(x)
                    t = st.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
                    rows.append({"birth": b, "k": rb[b]["k"], "d": float(d[i]), "hi_pct": hi, "hi_t": float(x.mean() + t)})
                flips = [r for r in rows if r["hi_pct"] >= 0]
                o["drop_one_in"] = {"n_births_whose_removal_flips_BELOW_to_AT_pct": len(flips),
                                    "n_flip_t": sum(r["hi_t"] >= 0 for r in rows), "flips": flips,
                                    "hi_pct_range": [min(r["hi_pct"] for r in rows), max(r["hi_pct"] for r in rows)]}
                # drop the k most negative / most positive pairs
                o["drop_extremes_in"] = {}
                order = np.argsort(d)
                for kk in (1, 2, 3):
                    x = np.delete(d, order[:kk])
                    o["drop_extremes_in"][f"drop_{kk}_most_negative"] = {"mean": float(x.mean()), "pct": boot_hi_lo(x)}
                    x = np.delete(d, order[-kk:])
                    o["drop_extremes_in"][f"drop_{kk}_most_positive"] = {"mean": float(x.mean()), "pct": boot_hi_lo(x)}
            # adjust the paired difference for the activation-size difference (log maxabs at T_(k-1))
            dm = np.array([np.log(mx[rb[b]["unit"]]) - np.log(mx[td[b]["unit"]]) for b in ids])
            X = np.c_[np.ones(len(ids)), dm]
            beta = np.linalg.lstsq(X, d, rcond=None)[0]
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(10000):
                j = rng.integers(0, len(ids), len(ids))
                bs.append(np.linalg.lstsq(X[j], d[j], rcond=None)[0])
            bs = np.array(bs)
            o[f"adjusted_for_dlogmaxabs_{s}"] = {"intercept": float(beta[0]), "intercept_pct": np.quantile(bs[:, 0], [0.025, 0.975]).tolist(),
                                                 "slope": float(beta[1]), "slope_pct": np.quantile(bs[:, 1], [0.025, 0.975]).tolist(),
                                                 "mean_dlogmaxabs": float(dm.mean())}
            # pairs with the twin's activation size close to the carrier's (|d log maxabs| below the median)
            close = np.abs(dm) <= np.median(np.abs(dm))
            o[f"close_maxabs_pairs_{s}"] = {"n": int(close.sum()), "mean": float(d[close].mean()), "pct": boot_hi_lo(d[close]),
                                            "far_n": int((~close).sum()), "far_mean": float(d[~close].mean())}
        # e against log maxabs within each kind (does a larger activation go with a smaller excess?)
        for kind in ("real", "dt", "pb"):
            us = [u for u in H if u["kind"] == kind]
            x = np.array([np.log(mx[u["unit"]]) for u in us])
            for s in ("in", "out"):
                y = np.array([u[s]["e_window"] for u in us])
                lr = st.linregress(x, y)
                o[f"e_{s}_vs_logmaxabs_{kind}"] = {"n": len(us), "slope": float(lr.slope), "se": float(lr.stderr),
                                                   "p": float(lr.pvalue), "spearman": float(st.spearmanr(x, y).statistic)}
        out[half] = o
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
