"""N1 offline gate: the inputs of K-B and both sides of every gate whose quantity the banks hold (the M2 A5 method).

The banks hold one weight statistic: the down-projection column norm of the neuron each record scores,
`value.colnorm` in M2's ranks (score-06) and N12's ranks (score-02), both `down_proj.weight.float().norm(dim=0)` on an
L40S, and N6's whole T_15 array (L1-02 `colnorms.npy`, the same operation). This script
  - writes studies/next/n1/kb_bank.json: one value per (layer, neuron, point), from N12's and M2's records;
  - copies N6's array to studies/next/n1/inputs/n6_colnorms_t15.npy (git LFS, the same object);
  - measures the control pairs on the bank's path (three L40S runs: M2 against N12 at every shared point, each against
    N6 at T_15): bitwise or not;
  - measures K-B's failing input offline (the neighbouring point's value of the same neuron against the tolerance);
  - and the room of the frozen labels.

    PYTHONPATH=. python studies/next/n1/offline_gate/gate_sides.py --out studies/next/n1/offline_gate/gate_sides.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import m2, n1

ROOT = pathlib.Path(__file__).resolve().parents[4]
N6_PATH = "studies/next/n6/banks/L1-02/colnorms.npy"


def recs(p):
    return m2.read_jsonl(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    srcs = {"m2": [ROOT / f"studies/next/m2/banks/score-06/m2/ranks_{h}.jsonl" for h in ("freeze", "test")],
            "n12": [ROOT / f"studies/next/n12/banks/score-02/n12/ranks_{h}.jsonl" for h in ("freeze", "test")]}
    vals = {}
    for name, ps in srcs.items():
        d = {}
        for p in ps:
            for r in recs(p):
                lyr, n, _ = m2.parse_pole(r["carrier"])
                d.setdefault((lyr, n, r["t"]), set()).add(r["value"]["colnorm"])
        assert all(len(v) == 1 for v in d.values()), f"{name}: a neuron with two column norms at one point"
        vals[name] = {k: next(iter(v)) for k, v in d.items()}
    common = sorted(set(vals["m2"]) & set(vals["n12"]))
    diff_mn = [abs(vals["m2"][k] - vals["n12"][k]) for k in common]
    blob = subprocess.run(["git", "show", f"n6:{N6_PATH}"], capture_output=True, check=True, cwd=ROOT).stdout
    raw = subprocess.run(["git", "lfs", "smudge"], input=blob, capture_output=True, check=True, cwd=ROOT).stdout
    dst = ROOT / "studies/next/n1/inputs/n6_colnorms_t15.npy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    n6 = np.load(dst)
    assert n6.shape == (32, 11008) and n6.dtype == np.float32
    vs6 = {name: [abs(float(np.float32(v)) - float(n6[k[0], k[1]])) for k, v in vals[name].items() if k[2] == 15]
           for name in vals}
    allv = {**vals["n12"], **vals["m2"]}
    kb = [[k[0], k[1], k[2], allv[k]] for k in sorted(allv)]
    (ROOT / "studies/next/n1/kb_bank.json").write_text(json.dumps(
        {"what": "down-projection column norms banked by M2 (score-06) and N12 (score-02) ranks records: "
                 "[layer, neuron, point, value]; equal where both hold one", "values": kb}))
    by = {}
    for (lyr, n, t), v in allv.items():
        by.setdefault((lyr, n), {})[t] = v
    adj = []
    for tv in by.values():
        ts = sorted(tv)
        adj += [abs(tv[b] - tv[x]) / tv[x] for x, b in zip(ts, ts[1:]) if b == x + 1]
    adj = np.array(adj)
    points = sorted({k[2] for k in allv})
    U = json.loads((ROOT / "studies/next/n1/units.json").read_text())["units"]
    res = {
        "kb_bank": {"n_values": len(kb), "points": points, "n_per_point": {t: sum(k[2] == t for k in allv) for t in points},
                    "sha256": m2.sha256_file(ROOT / "studies/next/n1/kb_bank.json")},
        "n6_t15": {"shape": list(n6.shape), "sha256": m2.sha256_file(dst), "lfs_oid_registered": m2.sha256_file(dst) ==
                   "dff570e3d32059eeb5edb6036fa4abf40a60a6dfeffecc92cf9e8036ffe5beb1"},
        "control_pairs_bank_path": {
            "m2_vs_n12": {"n": len(common), "max_abs": float(max(diff_mn)), "n_bitwise": int(sum(x == 0 for x in diff_mn))},
            "m2_vs_n6_t15": {"n": len(vs6["m2"]), "max_abs": float(max(vs6["m2"])), "n_bitwise": int(sum(x == 0 for x in vs6["m2"]))},
            "n12_vs_n6_t15": {"n": len(vs6["n12"]), "max_abs": float(max(vs6["n12"])),
                              "n_bitwise": int(sum(x == 0 for x in vs6["n12"]))}},
        "K_B": {"tol": n1.KB_TOL, "tol_reason": "the fp32 rounding bound of the bank's 4,096-term norm in any order",
                "failing_input_adjacent_point": {"n": int(adj.size), "min_rel": float(adj.min()),
                                                 "q01": float(np.quantile(adj, 0.01)), "median": float(np.median(adj)),
                                                 "n_within_tol": int((adj <= n1.KB_TOL).sum())},
                "pass_side": "not banked: no CPU fp64 norm of these columns exists; the bound holds for any summation "
                             "order, so a path difference cannot exceed it"},
        "units": {"n": len(U), "real_with_matched": sum(len(u.get("matched", [])) >= 90 for u in U if u["kind"] == "real"),
                  "matched_min": min(len(u["matched"]) for u in U if u["kind"] == "real")},
        "room": n1.room_check(),
        "windows": {"k_min": min(u["k"] for u in U), "reference_of_k_min": n1.reference(min(u["k"] for u in U)),
                    "window_of_k_min": n1.window(min(u["k"] for u in U))},
    }
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("N1_GATE_SIDES_OK")


if __name__ == "__main__":
    main()
