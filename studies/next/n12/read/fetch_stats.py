"""Fetch q999 and max_abs (and, for named pairs, hist and hist_edges) of the banked grid stats files by byte-range
reads (S3 read-only, profile fuelos, us-east-1). Output: one .npz per (point, layer) in --out (scratchpad, not
committed). Usage: python3 fetch_stats.py --units units.json --out DIR [--hist t:layer ...]"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "m2" / "read"))
import s3range  # noqa: E402  (byte-range reader of single npz members, REUSED unchanged)

DIRS = {5: "1/out/t05_step8000", 6: "1/out/t06_step12000", 7: "1b/out/t07_step20000", 8: "1b/out/t08_step35000",
        9: "1b/out/t09_step60000", 10: "1b/out/t10_step100000", 11: "1b/out/t11_step180000",
        12: "1b/out/t12_step320000", 13: "1b/out/t13_step560000", 14: "1b/out/t14_step1000000",
        15: "1/out/t15_step1413814"}


def key(t, layer):
    return f"results/neurons/w3/grid/{DIRS[t]}/stats/layer_{layer:02d}.npz"


ap = argparse.ArgumentParser()
ap.add_argument("--units", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hist", nargs="*", default=[])
ap.add_argument("--extra", nargs="*", default=[], help="t:layer pairs to fetch q999/max_abs for")
a = ap.parse_args()
out = pathlib.Path(a.out)
units = json.load(open(a.units))["units"]
pairs = sorted({(x["roles"]["pre"], x["layer"]) for x in units})
pairs += [tuple(int(v) for v in p.split(":")) for p in a.extra]
for t, layer in pairs:
    f = out / f"t{t:02d}_L{layer:02d}.npz"
    if not f.exists():
        m = s3range.members(key(t, layer), ["q999", "max_abs"])
        np.savez(f, **m)
        print("fetched", f.name, {k: v.shape for k, v in m.items()}, flush=True)
for p in a.hist:
    t, layer = (int(v) for v in p.split(":"))
    f = out / f"hist_t{t:02d}_L{layer:02d}.npz"
    if not f.exists():
        m = s3range.members(key(t, layer), ["hist", "hist_edges", "q999", "n_pole_tokens"])
        np.savez(f, **m)
        print("fetched", f.name, {k: v.shape for k, v in m.items()}, flush=True)
