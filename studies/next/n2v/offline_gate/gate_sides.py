"""N2V offline gate: both sides of every gate whose quantity the banks hold, on N12's banked records (the M2 A5 method).

Read: studies/next/n12/banks/score-02/n12/n12_{freeze,test}.jsonl (N12's 612 ablated pairs: the banked bf16 prediction
-h G and the banked bf16 measured dL), studies/next/n12/thresholds.json, studies/next/n12/units.json. Nothing of N2V
exists yet, so every pass side that needs an fp32 measurement is planted and said so.

    PYTHONPATH=. python studies/next/n2v/offline_gate/gate_sides.py --out studies/next/n2v/offline_gate/gate_sides.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from infra.pg import m2, n12, n2v

ROOT = pathlib.Path(__file__).resolve().parents[4]
NB = ROOT / "studies/next/n12/banks/score-02"
TH = ROOT / "studies/next/n12/thresholds.json"


def pairs_of(h: str, plant=None, seed: int = 1) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for r in m2.read_jsonl(NB / "n12" / f"n12_{h}.jsonl"):
        if r["kind"] != "real":
            continue
        for i, a in enumerate(r["abl"]):
            p = {"unit": r["unit"], "pole": a[0], "rank": i, "pred_bf16": a[1], "meas_bf16": a[2], "dL": a[2],
                 "probe": 0.0, "probe_own": 0.0}
            if plant is not None:
                p.update(plant(a[1], rng))
            out.append(p)
    return out


def floor_for_room(ps: list, readout: str) -> float:
    """The largest per-unit floor (the same for every unit) at which G-ROOM still passes: bisection in log10."""
    lo, hi = -8.0, 0.0
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = n2v.room(ps, readout, noise={p["unit"]: 10 ** mid for p in ps}, reps=200)["pass"]
        lo, hi = (mid, hi) if ok else (lo, mid)
    return float(10 ** lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"inputs": {"n12_freeze": m2.sha256_file(NB / "n12/n12_freeze.jsonl"),
                      "n12_test": m2.sha256_file(NB / "n12/n12_test.jsonl"),
                      "n12_thresholds": m2.sha256_file(TH), "units": m2.sha256_file(ROOT / "studies/next/n12/units.json")}}
    U = json.loads((ROOT / "studies/next/n12/units.json").read_text())["units"]
    recs = m2.read_jsonl(NB / "n12/n12_freeze.jsonl") + m2.read_jsonl(NB / "n12/n12_test.jsonl")
    pl = n2v.plan(U, recs)
    c = n2v.counts(pl)
    res["plan"] = {"points": {str(t): x for t, x in c.items()}, "unit_windows": sum(x["unit_windows"] for x in c.values()),
                   "units": sum(x["units"] for x in c.values()),
                   "pairs": {h: 4 * sum(1 for r in recs if r["kind"] == "real" and r["half"] == h) for h in ("freeze", "test")}}
    # K-A: the registered failing input (the poles rolled inside each unit) has room against the tolerance
    ka = pl[n2v.KNOWN_T][:n2v.K_A_N]
    by = {r["unit"]: r for r in recs}
    k0 = json.loads((NB / "n12/metrics.json").read_text())["gates"]["K0"]["units"]     # the units N12's K0 ran on
    gaps = []
    for it in ka:
        d = [x[2] for x in by[it["unit"]]["abl"]]
        gaps += [abs(d[j] - d[(j + 1) % 4]) for j in range(4)]
    res["K_A"] = {"units": [it["unit"] for it in ka], "equal_to_N12_K0_units": [it["unit"] for it in ka] == k0,
                  "tol": n2v.K_A_TOL, "failing_input_min_abs_diff": float(min(gaps)),
                  "failing_input_fails": bool(min(gaps) > n2v.K_A_TOL),
                  "pass_side": "not banked: no second bf16 run of N12's ablation rows exists; N12's G-ABL-ID (a clean row in "
                               "an ablation batch equal to the forward batch's value, 0.0 at 153 rows) is the evidence "
                               "that a row's loss does not depend on its batch companions"}
    # the re-reading needs no gradient run: every unit of both halves carries its pull
    nopull = [r["unit"] for r in recs if "pull" not in r]
    th = json.loads(TH.read_text())
    res["reread_inputs"] = {"records": len(recs), "records_without_pull": len(nopull),
                            "real_with_u_matched": sum("u_matched" in r.get("pull", {}) for r in recs if r["kind"] == "real"),
                            "pb_dt_with_u_S": sum("u_S" in r.get("pull", {}) for r in recs if r["kind"] != "real"),
                            "floor_u_freeze_q025": th["floor_u_freeze"]["q025"]}
    res["halves"] = {}
    for h in ("freeze", "test"):
        real = pairs_of(h)
        v = n2v.validation(real, "window")
        planted = pairs_of(h, lambda pr, g: {"dL": pr * (1 + 0.05 * g.normal()) + 1e-4 * g.normal(),
                                             "dL_fd": 0.1 * pr * (1 + 0.01 * g.normal()) + 1e-4 * g.normal(),
                                             "probe": 1e-4, "probe_own": 1e-5})
        vp = n2v.validation(planted, "window")
        vf = n2v.validation(planted, "fd")
        blk = {
            "G_VAL_on_N12_bf16_measurement": {"rho": v["rho"], "lo": v["lo"], "shuffle_q975": v["shuffle_q975"],
                                              "pass": v["pass"], "failing_input_rho": v["failing_input"]["rho"]},
            "G_VAL_planted_fp32_5pct": {"rho": vp["rho"], "pass": vp["pass"], "above_ceiling": vp["above_ceiling"],
                                        "inspection": n2v.inspect(planted, "window"),
                                        "failing_input_rho": vp["failing_input"]["rho"],
                                        "failing_input_pass": vp["failing_input"]["pass"]},
            "G_VAL_planted_copy": {"inspection": n2v.inspect([dict(p, dL=p["pred_bf16"]) for p in planted], "window")},
            "G_FD_planted": {"rho": vf["rho"], "pass": vf["pass"], "above_ceiling": vf["above_ceiling"],
                             "max_rel_dev": vf["max_rel_dev"], "failing_input_pass": vf["failing_input"]["pass"]},
            "G_FD_planted_copy": {"above_ceiling": n2v.validation([dict(p, dL_fd=0.1 * p["pred_bf16"]) for p in planted],
                                                                  "fd")["above_ceiling"]},
            "G_ROOM_failing_input_bf16": n2v.room(real, "window", noise=n2v.bf16_noise(real)),
            "G_ROOM_planted_1e-4": n2v.room(planted, "window"),
            "G_ROOM_fd_planted_1e-4": n2v.room(planted, "fd"),
            "largest_floor_with_room_window": floor_for_room(real, "window"),
            "largest_floor_with_room_fd": floor_for_room(real, "fd"),
            "abs_pred_quantiles_10_50_90": np.quantile([abs(p["pred_bf16"]) for p in real], [.1, .5, .9]).tolist(),
        }
        res["halves"][h] = blk
    P = {h: pairs_of(h, lambda pr, g: {"dL": pr * (1 + 0.05 * g.normal()) + 1e-4 * g.normal()}) for h in ("freeze", "test")}
    rr = n2v.reread(NB, TH, P["freeze"], P["test"], ceiling_cleared=True)
    rr_bf = n2v.reread(NB, TH, pairs_of("freeze"), pairs_of("test"))
    rr_nc = n2v.reread(NB, TH, P["freeze"], P["test"], ceiling_cleared=False)
    res["reread"] = {"planted_fp32_cleared": {"headline": rr["headline"], "labels": rr["labels"]},
                     "planted_fp32_not_cleared": {"headline": rr_nc["headline"]},
                     "N12_bf16_measurement": {"headline": rr_bf["headline"]},
                     "n12_ceiling_restored": n12.VAL_RHO_CEIL == 0.999}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("plan", "K_A", "reread_inputs")}, indent=1))
    for h, b in res["halves"].items():
        print(h, json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in list(v)[:4]}) for k, v in b.items()}))
    print(json.dumps(res["reread"], indent=1))
    print("N2V_GATE_SIDES_OK")


if __name__ == "__main__":
    main()
