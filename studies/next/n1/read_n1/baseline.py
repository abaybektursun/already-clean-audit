"""N1 independent read, step 5 (e): where carriers and twins sit against the whole layer and against the
activation-matched neurons. For every unit, e over its window for every neuron of its layer with the unit's k (the
layer's mean and median e: what an average neuron of that layer shows with the same window and reference), and per
birth the carrier minus the mean of its activation-matched neurons. Own code, from stats.npz.

    python3 -I baseline.py --stats S3/stats.npz --units-recount studies/next/n1/read_n1/units_recount.json \
        --out studies/next/n1/read_n1/baseline.json
"""
import argparse
import json
import pathlib

import numpy as np

MAD_K = 1.4826


def med_last(a):
    s = np.sort(a, axis=-1)
    n = s.shape[-1]
    return s[..., n // 2] if n % 2 else 0.5 * (s[..., n // 2 - 1] + s[..., n // 2])


def boot(x, seed=0, n_boot=10000):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    return {"n": int(x.size), "mean": float(x.mean()), "pct": np.quantile(m, [0.025, 0.975]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    for a in ("stats", "units-recount", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    S = np.load(a.stats)
    norms, diff = S["norms"], S["diff"]
    p = norms[:-1]
    R = {"in": np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2) / np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2),
         "out": diff[..., 2] / p[..., 2]}
    Z = {}
    for s, r in R.items():
        x = np.log(r)
        m = med_last(x)[..., None]
        Z[s] = (x - m) / (MAD_K * med_last(np.abs(x - m))[..., None])
    U = json.loads(pathlib.Path(a.units_recount).read_text())
    cache = {}

    def layer_e(s, l, k):
        key = (s, l, k)
        if key not in cache:
            zz = Z[s][:, l, :]
            e = zz[[k - 3, k - 2, k - 1]].mean(0) - zz[[i - 1 for i in range(2, k - 2)]].mean(0)
            cache[key] = (float(e.mean()), float(np.median(e)), float(np.quantile(e, 0.25)), float(np.quantile(e, 0.75)))
        return cache[key]

    out = {}
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        o = {}
        for s in ("in", "out"):
            for kind in ("real", "dt", "pb"):
                us = [u for u in H if u["kind"] == kind]
                lm = np.array([layer_e(s, u["layer"], u["k"])[0] for u in us])
                lmed = np.array([layer_e(s, u["layer"], u["k"])[1] for u in us])
                ev = np.array([u[s]["e_window"] for u in us])
                o[f"{kind}_{s}"] = {"e": boot(ev), "layer_mean_e_same_k": float(lm.mean()), "layer_median_e_same_k": float(lmed.mean()),
                                    "e_minus_layer_mean": boot(ev - lm), "e_minus_layer_median": boot(ev - lmed)}
            real = [u for u in H if u["kind"] == "real"]
            dm = np.array([u[s]["e_window"] - np.mean(u["matched_e_window"][s]) for u in real])
            o[f"real_minus_matched_mean_{s}"] = boot(dm)
            o[f"matched_mean_{s}"] = float(np.mean([np.mean(u["matched_e_window"][s]) for u in real]))
            o[f"matched_minus_layer_mean_{s}"] = boot([np.mean(u["matched_e_window"][s]) - layer_e(s, u["layer"], u["k"])[0] for u in real])
        out[half] = o
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
