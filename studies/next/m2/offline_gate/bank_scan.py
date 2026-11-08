"""Offline gate, amendment A4 (M2_SPEC.md §3.8 "A4"): the banked layer files of the grid harvest at the 11 points M2
scores (T_5 ... T_15), all 32 layers, scanned for every pole with no token of its own sign (max_abs 0 or an empty
own-sign histogram, n_pole_tokens 0) and for the other banked facts the scoring path assumes (members present, array
shapes, the histogram edges, q999 finite and positive on every other pole).

Reads only the small members of stats/layer_LL.npz (max_abs, hist_edges, n_pole_tokens, q999, density) and the .npy
headers of the large ones (stats hist; topk tok and a), by byte-range requests (aws s3api get-object --range); no full
file is downloaded. Positive control of the scan itself: its max_abs and q999 of the 640 grid poles equal the laptop
bank (arrays.npz, sha256 registered in m2.BANK_SHA) at every point; failing input: the same compared with the
neighbouring point's laptop values. Facts only; the checks are made by the dry run (m2_dryrun --scan) and printed here.

    PYTHONPATH=. python studies/next/m2/offline_gate/bank_scan.py --bank <BANK/neurons> --births studies/next/m2/births.json \
        --out studies/next/m2/offline_gate/bank_scan.json
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np  # noqa: E402

from infra.pg import m2  # noqa: E402

BUCKET, ROOT = "pg-lineage-769063704696", "results/neurons/w3/grid"
AWS = ["--profile", "fuelos", "--region", "us-east-1"]
FULL = ("max_abs", "hist_edges", "n_pole_tokens", "q999", "density")     # read whole (small)
HEAD = {"stats": ("hist",), "topk": ("tok", "a")}                          # .npy header only (large)
NEED = {"stats": ("hist", "hist_edges", "q999", "max_abs", "density", "n_pole_tokens"), "topk": ("tok", "a")}
POINTS_SCORED = list(range(5, 16))


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of one S3 object. Spans are fetched by ranged GETs and cached; a read outside them
    fetches 64 KB from the read position."""

    def __init__(self, key, size):
        self.key, self.size, self.pos, self.cache, self.calls = key, size, 0, [], 0

    def fetch(self, lo, hi):
        hi = min(hi, self.size)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "part")
            subprocess.run(["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", self.key, "--range",
                            f"bytes={lo}-{hi - 1}", *AWS, f], capture_output=True, check=True)
            data = pathlib.Path(f).read_bytes()
        assert len(data) == hi - lo, (self.key, lo, hi, len(data))
        self.cache.append((lo, data))
        self.calls += 1

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for lo, data in self.cache:
            if lo <= self.pos and self.pos + n <= lo + len(data):
                b[:n] = data[self.pos - lo:self.pos - lo + n]
                self.pos += n
                return n
        self.fetch(self.pos, self.pos + max(n, 1 << 16))
        return self.readinto(b)


def npy_header(z, name):
    with z.open(name + ".npy") as f:
        head = io.BytesIO(f.read(512))
    ver = np.lib.format.read_magic(head)
    shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else
                             np.lib.format.read_array_header_2_0)(head)
    return {"shape": list(shape), "dtype": str(dtype)}


def open_zip(key, size, full, head):
    rf = RangeFile(key, size)
    rf.fetch(max(0, size - (1 << 16)), size)                  # end of central directory and the directory
    z = zipfile.ZipFile(rf)
    info = {i.filename[:-4]: i for i in z.infolist()}
    spans = sorted((info[n].header_offset, info[n].header_offset + 30 + len(info[n].filename) + 256 +
                    (info[n].compress_size if n in full else 4096)) for n in full + head if n in info)
    merged = []
    for lo, hi in spans:                                       # one GET per run of adjacent members
        if merged and lo - merged[-1][1] < (1 << 16):
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    for lo, hi in merged:
        rf.fetch(lo, hi)
    return z, rf, sorted(info)


def scan_stats(t, lyr, size):
    key = f"{ROOT}/{m2.POINTS[t][1]}/stats/layer_{lyr:02d}.npz"
    z, rf, members = open_zip(key, size, FULL, HEAD["stats"])
    arr = {n: np.load(io.BytesIO(z.read(n + ".npy"))) for n in FULL}
    out = {"key": key, "members": members, "headers": {n: npy_header(z, n) for n in HEAD["stats"]},
           "shapes": {n: {"shape": list(arr[n].shape), "dtype": str(arr[n].dtype)} for n in FULL},
           "edges_equal": bool(np.array_equal(arr["hist_edges"], (10.0 ** np.linspace(
               m2.HIST_LO, m2.HIST_HI, m2.HIST_B + 1)).astype(np.float32)))}
    n, mx, q, den = arr["n_pole_tokens"], arr["max_abs"], arr["q999"], arr["density"]
    empty, dead = n == 0, ~(mx > 0)
    deg = empty | dead
    out["degenerate"] = [{"pole": m2.pole_id(lyr, int(i), int(s)), "n_pole_tokens": int(n[i, s]),
                          "max_abs": float(mx[i, s]), "q999": None if np.isnan(q[i, s]) else float(q[i, s]),
                          "density": int(den[i, s])} for i, s in np.argwhere(deg)]
    out["n_empty"], out["n_dead"], out["n_degenerate"] = int(empty.sum()), int(dead.sum()), int(deg.sum())
    ok = ~deg
    out["n_other"] = int(ok.sum())
    out["other_q999_not_finite_positive"] = int((~(np.isfinite(q[ok]) & (q[ok] > 0))).sum())
    out["other_max_abs_not_finite"] = int((~np.isfinite(mx[ok])).sum())
    out["calls"] = rf.calls
    return out, {"max_abs": mx, "q999": q}


def scan_topk(t, lyr, size):
    key = f"{ROOT}/{m2.POINTS[t][1]}/topk/layer_{lyr:02d}.npz"
    z, rf, members = open_zip(key, size, (), HEAD["topk"])
    return {"key": key, "members": members, "headers": {n: npy_header(z, n) for n in HEAD["topk"]}, "calls": rf.calls}


def sizes(t, kind):
    out = subprocess.run(["aws", "s3", "ls", f"s3://{BUCKET}/{ROOT}/{m2.POINTS[t][1]}/{kind}/", *AWS],
                         capture_output=True, text=True, check=True).stdout
    return {int(ln.split()[3][6:8]): int(ln.split()[2]) for ln in out.splitlines() if ln.split()[3].startswith("layer_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--births", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = np.load(pathlib.Path(a.bank) / m2.BANK_FILES["arrays"])
    assert m2.sha256_file(pathlib.Path(a.bank) / m2.BANK_FILES["arrays"]) == m2.BANK_SHA["arrays"]
    grid = [m2.parse_pole(str(p)) for p in B["poles"]]
    sz = {(t, kind): sizes(t, kind) for t in POINTS_SCORED for kind in ("stats", "topk")}
    for (t, kind), s in sz.items():
        assert sorted(s) == list(range(m2.N_LAYERS)), (t, kind, sorted(s))
    res, vals = {}, {}
    with cf.ThreadPoolExecutor(16) as ex:
        fut = {}
        for t in POINTS_SCORED:
            for lyr in range(m2.N_LAYERS):
                fut[ex.submit(scan_stats, t, lyr, sz[(t, "stats")][lyr])] = (t, lyr, "stats")
                fut[ex.submit(scan_topk, t, lyr, sz[(t, "topk")][lyr])] = (t, lyr, "topk")
        for f in cf.as_completed(fut):
            t, lyr, kind = fut[f]
            r = f.result()
            if kind == "stats":
                res.setdefault((t, lyr), {})["stats"], vals[(t, lyr)] = r
            else:
                res.setdefault((t, lyr), {})["topk"] = r
            print(t, lyr, kind, "degenerate", r[0]["n_degenerate"] if kind == "stats" else "-", flush=True)
    # the scan's own positive control: the 640 grid poles against the laptop bank, and against the neighbour point
    ctrl = {}
    for t in POINTS_SCORED:
        nb = t + 1 if t < 15 else t - 1
        row = {}
        for nm in ("max_abs", "q999"):
            got = np.array([float(vals[(t, lyr)][nm][n, s]) for lyr, n, s in grid])
            want, fail = B[nm][t].astype(np.float32).astype(np.float64), B[nm][nb].astype(np.float32).astype(np.float64)
            row[nm] = {"n": len(grid), "equal": int((got == want).sum()), "max_abs_diff": float(np.abs(got - want).max()),
                       "equal_to_neighbour": int((got == fail).sum()), "neighbour": nb}
        ctrl[t] = row
    pl = m2.plan(json.loads(pathlib.Path(a.births).read_text())["births"])
    um = {u["unit"]: u for u in pl["units"]}
    points = {}
    for t in POINTS_SCORED:
        scored = sorted({lyr for w in pl["points"][t]["need"].values() for lyr, _, _ in w}) if t in pl["points"] else []
        deg_all = [d["pole"] for lyr in range(m2.N_LAYERS) for d in res[(t, lyr)]["stats"]["degenerate"]]
        deg_scored = [d["pole"] for lyr in scored for d in res[(t, lyr)]["stats"]["degenerate"]]
        own = [[uid, role, um[uid]["pole"]] for uid, role in (pl["points"][t]["roles"] if t in pl["points"] else [])
               if um[uid]["pole"] in set(deg_all)]
        points[t] = {"scored_layers": scored, "n_degenerate_all_layers": len(deg_all), "degenerate_all_layers": deg_all,
                     "n_degenerate_scored_layers": len(deg_scored), "degenerate_scored_layers": deg_scored,
                     "units_whose_own_pole_is_degenerate": own,
                     "min_other_poles_scored_layer": min((res[(t, lyr)]["stats"]["n_other"] for lyr in scored),
                                                         default=None)}
    out = {"what": __doc__.split("\n\n")[0], "bucket": BUCKET, "root": ROOT, "points": points, "control": ctrl,
           "files": {f"t{t:02d}_L{lyr:02d}": res[(t, lyr)] for t in POINTS_SCORED for lyr in range(m2.N_LAYERS)},
           "n_calls": sum(r["stats"]["calls"] + r["topk"]["calls"] for r in res.values())}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print("point | degenerate (all 32 layers) | in the scored layers | ids")
    for t, p in points.items():
        print(f"T_{t} | {p['n_degenerate_all_layers']} | {p['n_degenerate_scored_layers']} | "
              f"{', '.join(p['degenerate_all_layers'])} | own-pole units {p['units_whose_own_pole_is_degenerate']}")
    print("control", json.dumps(ctrl))
    print("BANK_SCAN_DONE", out["n_calls"], "range or list calls")


if __name__ == "__main__":
    main()
