"""N1 independent read: the unit of z and of the paired gap in natural units. z is log r in units of 1.4826 x the
median absolute deviation of log r over the layer's 11,008 neurons in that bracket. This script reports that scale
per bracket (median, min, max over the 32 layers) and the test half's paired real - dt gap in mean log r over the
window (no reference subtracted), as a relative change.

    python3 -I scale.py --stats S3/stats.npz --units-recount studies/next/n1/read_n1/units_recount.json \
        --out studies/next/n1/read_n1/scale.json
"""
import argparse
import json
import pathlib

import numpy as np


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
    out = {}
    for s, r in R.items():
        x = np.log(r)
        m = med_last(x)
        sc = 1.4826 * med_last(np.abs(x - m[..., None]))           # [15, 32]
        out[f"scale_{s}"] = {str(j + 1): {"median_r": float(np.median(np.exp(m[j]))), "z_unit_in_log_r_median": float(np.median(sc[j])),
                                          "min": float(sc[j].min()), "max": float(sc[j].max())} for j in range(15)}
    U = json.loads(pathlib.Path(a.units_recount).read_text())
    for half in ("test", "freeze"):
        H = [u for u in U.values() if u["half"] == half]
        rb = {u["birth"]: u for u in H if u["kind"] == "real"}
        td = {u["birth"]: u for u in H if u["kind"] == "dt"}
        ids = sorted(b for b in rb if b in td)
        for s in ("in", "out"):
            d = np.array([rb[b]["r_window_mean_log"][s] - td[b]["r_window_mean_log"][s] for b in ids])
            bt = boot(d)
            out[f"{half}_pair_mean_log_r_window_{s}"] = {**bt, "as_relative": [float(np.exp(v) - 1) for v in [bt["mean"]] + bt["pct"]]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("scale")}, indent=1))
    for s in ("in", "out"):
        print(s, {j: (round(v["median_r"], 3), round(v["z_unit_in_log_r_median"], 4)) for j, v in out[f"scale_{s}"].items()})


if __name__ == "__main__":
    main()
