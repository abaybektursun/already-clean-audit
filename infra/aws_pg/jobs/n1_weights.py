"""N1 job (studies/next/07_PICKS/N1_SPEC.md section 9): the per-neuron weight statistics of the 16 grid checkpoints, on
one CPU instance (no GPU). One checkpoint on disk at a time; the previous checkpoint's MLP tensors held in memory (bf16).

Per checkpoint T_j, j = 0..15:
  1. Download (m2_score.download), G-REV; G-SHAPE: the 96 MLP tensors exist with gate/up [11,008, 4,096] and down
     [4,096, 11,008] (failing input: the shapes of the transposed down tensor).
  2. Per layer, fp64 from the bf16 tensors (exact): the norm of every neuron's gate row, up row and down column
     (norms [16, 32, 11,008, 3]); for j >= 1, the norm of each vector's change since T_(j-1) and the dot product of the
     two (diff, dot [15, 32, 11,008, 3]).
  3. K-B where the banks hold a column norm of this checkpoint (N12's and M2's records at T_5..T_15, N6's whole T_15
     array): |ours - bank| / bank <= n1.KB_TOL (the fp32 rounding bound of a 4,096-term norm in any order); failing
     input: our column norms of T_(j-1) against the bank's T_j values. REPORTED, not gated (the head's decision of
     2026-11-06 19:25Z: no control pair exists for an L40S fp32 path against a CPU fp64 path, the two-path rule); its
     values and its failing input's are written to metrics.json -> reports -> tNN_K_B. The fp32 norm of the same column on this CPU
     (torch, the bank's operation) is reported beside it. TRI (reported, not a gate: its failing input has no power,
     the offline gate showed it): for every banked neuron with values at T_(j-1) and T_j, our relative column change
     is at least the banked relative norm change (the triangle inequality) less 2 n1.KB_TOL.
  4. PC (positive control) for j >= 2: in each layer, n1.PC_N seeded neurons get a planted change of relative size
     equal to the layer's 99th percentile of r in this bracket (input rows and column separately, a seeded random
     direction); the share of planted neurons whose z lands above the layer's 95th percentile must be at least
     n1.PC_MIN on both sides; failing input: the same with a planted size 0.
  5. At T_15: I-ID, the tensors read a second time from disk: norms bitwise equal and the change of T_15 against itself
     exactly 0 for every neuron; failing input: the change of T_14 against T_15 (must be non-zero).
After the last checkpoint: stats.npz (norms, diff, dot), the unit records n1_<half>.jsonl (n1.records), the raw examples
samples/raw_examples_<half>.jsonl (the example births of registry["examples"], n1.EXAMPLES; A1, 2026-11-07: weights-01
read them from units.json, which carries only "units" and "n", and stopped with KeyError). out/n1/metrics.json is
rewritten after every phase and carries no excess, no z and no label.

    python -m infra.aws_pg.jobs.n1_weights --inputs <uri> --budget-min <min> --out <dir>

Fail-loud: no try/except, no fallback, no retry.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np

from infra.aws_pg.jobs import m2_score as M
from infra.aws_pg.jobs.neurons_harvest import fetch
from infra.pg import m2, n1

UNITS_SHA = "ada35f38cc4179f4630a14c331359fa584d0eb079e529a2ac27490e0dfd83ed9"   # studies/next/n1/units.json
KB_SHA = "401c6134fc9bd5bc068c364026b2843382033ff3c0b3b33d6acbf1f0072fe988"      # studies/next/n1/kb_bank.json
N6_SHA = "dff570e3d32059eeb5edb6036fa4abf40a60a6dfeffecc92cf9e8036ffe5beb1"   # n6 banks/L1-02/colnorms.npy (LFS oid)


class Run(M.Run):
    def save(self, phase: str):
        self.rec["phase"], self.rec["updated_utc"] = phase, M.utc()
        self.rec["elapsed_s"] = round(time.time() - self.t0, 1)
        M.dump(self.out / "n1" / "metrics.json", self.rec)
        print(f"[n1] {phase}", flush=True)


def names(layer: int) -> tuple:
    p = f"model.layers.{layer}.mlp."
    return p + "gate_proj.weight", p + "up_proj.weight", p + "down_proj.weight"


def read_mlp(local: pathlib.Path, n_layers: int) -> dict:
    """{layer: (gate, up, down)} bf16 torch tensors from the snapshot's safetensors (every MLP tensor, any shard)."""
    from safetensors.torch import safe_open
    idx = json.loads((local / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for lyr in range(n_layers):
        ts = []
        for nm in names(lyr):
            with safe_open(str(local / idx[nm]), framework="pt") as f:
                ts.append(f.get_tensor(nm).clone())
        out[lyr] = tuple(ts)
    return out


def shape_gate(mlp: dict, N: int, H: int) -> dict:
    sh = {lyr: [list(t.shape) for t in v] for lyr, v in mlp.items()}
    want = [[N, H], [N, H], [H, N]]
    bad = [lyr for lyr, s in sh.items() if s != want]
    tr = [[N, H], [N, H], [N, H]]
    return {"pass": not bad and all(str(t.dtype) == "torch.bfloat16" for v in mlp.values() for t in v), "bad_layers": bad,
            "n_layers": len(sh), "failing_input": {"what": "the down tensor transposed", "pass": tr == want}}


def layer_stats(cur: tuple, prev: tuple | None):
    """fp64: norms [N, 3] of gate rows, up rows, down columns; with prev, diff [N, 3] and dot [N, 3]."""
    import torch
    g, u, d = (t.to(torch.float64) for t in cur)
    norms = torch.stack([g.norm(dim=1), u.norm(dim=1), d.norm(dim=0)], 1)
    if prev is None:
        return norms.numpy(), None, None
    pg, pu, pd = (t.to(torch.float64) for t in prev)
    diff = torch.stack([(g - pg).norm(dim=1), (u - pu).norm(dim=1), (d - pd).norm(dim=0)], 1)
    dot = torch.stack([(g * pg).sum(1), (u * pu).sum(1), (d * pd).sum(0)], 1)
    return norms.numpy(), diff.numpy(), dot.numpy()


def fp32_colnorm(d) -> np.ndarray:
    """The bank's operation, down_proj.weight.float().norm(dim=0), here on the CPU (reported beside K-B)."""
    return d.float().norm(dim=0).numpy().astype(np.float64)


def planted(cur: tuple, prev: tuple, neurons: np.ndarray, c_in: float, c_out: float, seed: int) -> np.ndarray:
    """r of the planted neurons [P, 2] (in, out): change += c * ||vector(T_(j-1))|| * a seeded unit direction."""
    import torch
    rng = np.random.default_rng(seed)
    g, u, d = (t.to(torch.float64) for t in cur)
    pg, pu, pd = (t.to(torch.float64) for t in prev)
    out = []
    for n in neurons:
        vin = torch.cat([g[n] - pg[n], u[n] - pu[n]])
        pin = torch.cat([pg[n], pu[n]])
        e = torch.from_numpy(rng.normal(size=vin.numel()))
        vin = vin + c_in * pin.norm() * e / e.norm()
        vout = d[:, n] - pd[:, n]
        e = torch.from_numpy(rng.normal(size=vout.numel()))
        vout = vout + c_out * pd[:, n].norm() * e / e.norm()
        out.append([float(vin.norm() / pin.norm()), float(vout.norm() / pd[:, n].norm())])
    return np.array(out)


def run(a, reg: dict, out: pathlib.Path):
    import torch
    torch.set_num_threads(os.cpu_count())
    R = Run(out, a.budget_min * 60)
    v = {"torch": torch.__version__.split("+")[0]}
    R.gate("G_SAME", {"pass": v == reg["versions"], "versions": v, "registered": reg["versions"],
                      "failing_input": {"pass": {"torch": "0"} == reg["versions"]}})
    paths = {"units": fetch(f"{a.inputs.rstrip('/')}/units.json", "inputs/units.json"),
             "kb": fetch(f"{a.inputs.rstrip('/')}/kb_bank.json", "inputs/kb_bank.json"),
             "n6": fetch(f"{a.inputs.rstrip('/')}/n6_colnorms_t15.npy", "inputs/n6_colnorms_t15.npy")}
    got = {k: m2.sha256_file(p) for k, p in paths.items()}
    want = {"units": reg["units_sha"], "kb": reg["kb_sha"], "n6": reg["n6_sha"]}
    R.gate("INPUTS", {"pass": got == want, "mismatch": sorted(k for k in want if got[k] != want[k]),
                      "failing_input": {"pass": got["units"] == "0" * 64}})
    uj = json.loads(paths["units"].read_text())
    kb = json.loads(paths["kb"].read_text())["values"]          # [layer, neuron, t, colnorm]
    n6 = np.load(paths["n6"]).astype(np.float64)
    L, N, H = reg["n_layers"], reg["n_neurons"], reg["hidden"]
    T = len(reg["points"])
    norms = np.zeros((T, L, N, 3))
    diff = np.zeros((T - 1, L, N, 3))
    dot = np.zeros((T - 1, L, N, 3))
    prev, pc = None, {}
    for t in range(T):
        pt = {"t": t}
        R.rec["points"][str(t)] = pt
        t0 = time.time()
        local, pt["download"] = M.download(reg["points"][t][2], reg["points"][t][3])
        R.gate(f"t{t:02d}_G_REV", pt["download"].pop("G_REV"))
        cur = read_mlp(local, L)
        R.gate(f"t{t:02d}_G_SHAPE", shape_gate(cur, N, H))
        fp32 = np.zeros((L, N))
        for lyr in range(L):
            nm, df, dt = layer_stats(cur[lyr], prev[lyr] if prev is not None else None)
            norms[t, lyr] = nm
            fp32[lyr] = fp32_colnorm(cur[lyr][2])
            if df is not None:
                diff[t - 1, lyr], dot[t - 1, lyr] = df, dt
        bank = [x for x in kb if x[2] == t]
        if t == T - 1:
            bank = bank + [[lyr, n, t, float(n6[lyr, n])] for lyr in range(L) for n in range(N)]
        if bank:
            ours = np.array([norms[t, x[0], x[1], 2] for x in bank])
            bv = np.array([x[3] for x in bank])
            rel = np.abs(ours - bv) / bv
            g = {"pass": bool(rel.max() <= n1.KB_TOL), "n": len(bank), "max_rel": float(rel.max()),
                 "n_exact_fp32": int((np.array([fp32[x[0], x[1]] for x in bank]) == bv.astype(np.float32)).sum()),
                 "max_rel_fp32_cpu": float((np.abs(np.array([fp32[x[0], x[1]] for x in bank]) - bv) / bv).max()),
                 "tol": n1.KB_TOL}
            if t >= 1:
                fo = np.array([norms[t - 1, x[0], x[1], 2] for x in bank])
                fr = np.abs(fo - bv) / bv
                g["failing_input"] = {"what": "our column norms of T_(t-1) against the bank's T_t values",
                                      "pass": bool(fr.max() <= n1.KB_TOL), "max_rel": float(fr.max())}
            R.report(f"t{t:02d}_K_B", g)
        pairs = [(x, y) for x in kb if x[2] == t - 1 for y in kb if y[2] == t and y[:2] == x[:2]]
        if pairs:
            ro = np.array([diff[t - 1, x[0], x[1], 2] / norms[t - 1, x[0], x[1], 2] for x, _ in pairs])
            bn = np.array([abs(y[3] - x[3]) / x[3] for x, y in pairs])
            R.report(f"t{t:02d}_TRI", {"holds": bool((ro >= bn - 2 * n1.KB_TOL).all()), "n": len(pairs),
                                       "min_margin": float((ro - bn).min()),
                                       "rolled_holds": bool((ro >= np.roll(bn, 1) - 2 * n1.KB_TOL).all())})
        if t >= 2:
            rr = n1.side_r(norms[t - 1:t + 1], diff[t - 1:t])
            share, share0 = {"in": [], "out": []}, {"in": [], "out": []}
            for lyr in range(L):
                ns = np.random.default_rng([n1.PC_SEED, t, lyr]).choice(N, n1.PC_N, replace=False)
                c = {s: float(np.quantile(rr[s][0, lyr], 0.99)) for s in ("in", "out")}
                for s, sh in ((c, share), ({"in": 0.0, "out": 0.0}, share0)):
                    rp = planted(cur[lyr], prev[lyr], ns, s["in"], s["out"], seed=t * 1000 + lyr)
                    for i, side in enumerate(("in", "out")):
                        x = np.log(rr[side][0, lyr])
                        med = np.median(x)
                        mad = n1.MAD_K * np.median(np.abs(x - med))
                        z = (np.log(rp[:, i]) - med) / mad
                        sh[side].append(float(np.mean(z > np.quantile((x - med) / mad, 0.95))))
            pc[t] = {"in": float(np.mean(share["in"])), "out": float(np.mean(share["out"])),
                     "in_size0": float(np.mean(share0["in"])), "out_size0": float(np.mean(share0["out"]))}
            R.gate(f"t{t:02d}_PC", {"pass": min(pc[t]["in"], pc[t]["out"]) >= n1.PC_MIN, **pc[t], "min": n1.PC_MIN,
                                    "failing_input": {"what": "planted size 0",
                                                      "pass": min(pc[t]["in_size0"], pc[t]["out_size0"]) >= n1.PC_MIN}})
        if t == T - 1:
            again = read_mlp(local, L)
            same, zero, fz = True, True, True
            for lyr in range(L):
                a2, d2, _ = layer_stats(again[lyr], cur[lyr])
                same &= bool(np.array_equal(a2, norms[t, lyr]))
                zero &= bool((d2 == 0).all())
                fz &= bool((diff[t - 1, lyr] == 0).all())
            R.gate("I_ID", {"pass": same and zero, "norms_bitwise": same, "self_change_zero": zero,
                            "failing_input": {"what": "the change of T_14 against T_15", "pass": fz}})
        pt["seconds"] = round(time.time() - t0, 1)
        prev = cur
        shutil.rmtree(pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + m2.REPO.replace("/", "--")))
        R.save(f"done T_{t}")
        assert time.time() - R.t0 < 0.95 * R.budget_s, "over 0.95 of the budget"
    np.savez_compressed(out / "n1" / "stats.npz", norms=norms, diff=diff, dot=dot)
    recs = n1.records(uj["units"], norms, diff, dot)
    for h in ("freeze", "test"):
        with open(out / "n1" / f"n1_{h}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs if r["half"] == h)
    (out / "n1" / "samples").mkdir(exist_ok=True)
    ex = {r["unit"]: r for r in recs}
    for h in ("freeze", "test"):
        with open(out / "n1" / "samples" / f"raw_examples_{h}.jsonl", "w") as f:
            for b in reg["examples"][h]:          # A1: the registry's example births (units.json has none)
                f.write(json.dumps({"birth": b, **{k: ex[b + s] for k, s in (("real", ""), ("dt", ":dt"), ("pb", ":pb"))
                                                   if b + s in ex}}) + "\n")
    R.rec["n_records"] = {h: sum(r["half"] == h for r in recs) for h in ("freeze", "test")}
    R.rec["complete"] = True
    R.save("done")


def registry(path=None) -> dict:
    if path is not None:
        r = json.loads(pathlib.Path(path).read_text())
        r["points"] = {int(k): v for k, v in r["points"].items()}
        return r
    assert UNITS_SHA and KB_SHA, "units.json and kb_bank.json are registered at freeze 1"
    return {"versions": {"torch": M.VERSIONS["torch"].split("+")[0]}, "units_sha": UNITS_SHA, "kb_sha": KB_SHA,
            "n6_sha": N6_SHA, "points": {p[0]: list(p) for p in m2.POINTS}, "n_layers": 32, "n_neurons": 11008,
            "hidden": 4096, "examples": n1.EXAMPLES}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True, help="S3 prefix or directory: units.json, kb_bank.json, n6_colnorms_t15.npy")
    ap.add_argument("--budget-min", type=float, required=True)
    ap.add_argument("--registry", default=None, help="dry run only: a planted registry JSON")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = pathlib.Path(a.out)
    (out / "n1").mkdir(parents=True, exist_ok=True)
    if a.registry is None:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    run(a, registry(a.registry), out)
    print("N1_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
