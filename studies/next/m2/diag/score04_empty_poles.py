"""score-04 crashed at T_10 in m2.percentile ("a pole with no token of its sign"): which poles of the layers the plan
scores at T_5 ... T_15 have no token of their own sign in the banked histogram (stats/ n_pole_tokens == 0), and which
have max_abs == 0 (m2.matched asserts max_abs > 0 over the whole layer). Reads only the members n_pole_tokens.npy and
max_abs.npy of each banked stats/layer_LL.npz on S3, by byte-range requests (aws s3api get-object --range); no full
stats file is downloaded. Banked harvest arrays only; nothing of score-04 beyond metrics.json and job.log is opened.

    PYTHONPATH=. python studies/next/m2/diag/score04_empty_poles.py <out.json>
"""
import io
import json
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


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of one S3 object; every read is one ranged GET."""

    def __init__(self, key):
        self.key, self.pos = key, 0
        out = subprocess.run(["aws", "s3api", "head-object", "--bucket", BUCKET, "--key", key, *AWS,
                              "--query", "ContentLength", "--output", "text"], capture_output=True, text=True, check=True)
        self.size = int(out.stdout.strip())
        self.calls = 0

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
        with tempfile.NamedTemporaryFile() as f:
            subprocess.run(["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", self.key, "--range",
                            f"bytes={self.pos}-{self.pos + n - 1}", *AWS, f.name], capture_output=True, check=True)
            data = pathlib.Path(f.name).read_bytes()
        assert len(data) == n, (len(data), n)
        b[:n] = data
        self.pos += n
        self.calls += 1
        return n


def members(key):
    rf = RangeFile(key)
    with zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)) as z:
        out = {nm: np.load(io.BytesIO(z.read(nm + ".npy"))) for nm in ("n_pole_tokens", "max_abs")}
    return out, rf.calls


def main():
    out = pathlib.Path(sys.argv[1])
    pl = m2.plan(json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"])
    res = {"what": __doc__.split("\n")[0], "points": {}}
    for t in sorted(pl["points"]):
        if t > 10:
            continue
        layers = sorted({lyr for w in pl["points"][t]["need"].values() for lyr, _, _ in w})
        first = pl["points"][t]["roles"][0]
        row = {"layers": layers, "first_scored_unit": first, "empty": {}, "dead": {}}
        for lyr in layers:
            z, calls = members(f"{ROOT}/{m2.POINTS[t][1]}/stats/layer_{lyr:02d}.npz")
            n, mx = z["n_pole_tokens"], z["max_abs"]
            assert n.shape == (m2.N_NEURONS, 2) and mx.shape == (m2.N_NEURONS, 2), (n.shape, mx.shape)
            e = np.argwhere(n == 0)
            dd = np.argwhere(mx <= 0)
            if e.size:
                row["empty"][lyr] = [m2.pole_id(lyr, int(i), int(s)) for i, s in e]
            if dd.size:
                row["dead"][lyr] = [m2.pole_id(lyr, int(i), int(s)) for i, s in dd]
            print(t, lyr, "empty", len(e), "dead", len(dd), "calls", calls, flush=True)
        res["points"][t] = row
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({t: {"empty": {k: len(v) for k, v in r["empty"].items()}, "dead": {k: len(v) for k, v in r["dead"].items()},
                          "first": r["first_scored_unit"]} for t, r in res["points"].items()}, indent=1))


if __name__ == "__main__":
    main()
