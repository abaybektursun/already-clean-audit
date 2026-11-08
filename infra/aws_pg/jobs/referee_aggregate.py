"""REFEREE MDE aggregate — the frozen unblind rules applied once (journal FREEZE block).
Order is mandatory: P4 motion audits and P5 divergence verdicts are computed BEFORE any
endpoint is read; VOID cells are excluded with disclosure. Then P2 continuity, then the
per-axis contrasts and the §1 mechanical adjudication + §3 branch decision.

    python -m infra.aws_pg.jobs.referee_aggregate --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
SEEDS = (0, 1, 2)
ARMS = {"baseline": "baseline", "incore16": "incore16", "leverage16": "leverage16"}
N_BOOT = 10000
BOOT_SEED = 20260806
TARGET_PERSTEP = 0.00517875318702413      # frozen P3 target (journal)
DOSE_BAND = (0.5, 2.0)                    # delivered-dose tolerance on realized/target


def _pull(rel, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{rel}", str(dest),
                    "--region", "us-east-1"], check=True, capture_output=True)
    return json.loads(dest.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import referee_hook as RH

    rep = {"unblind_order": "P4 -> P5 -> P2 -> contrasts -> adjudication", "cells": {},
           "void": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    C = {}
    for s in SEEDS:
        for arm in ARMS:
            C[(arm, s)] = _pull(f"referee_mde_s{s}/out/{arm}/metrics.json",
                                out / f"in/{arm}_s{s}.json")

    # ---- P4: realized-motion audit (BEFORE endpoints) ---------------------------------------
    for s in SEEDS:
        for arm in ("incore16", "leverage16"):
            r = C[(arm, s)]
            aud = r["motion_audit"][-1]["audit"]
            insub = [aud[k]["in_sub"] for k in sorted(aud)]
            realized = float(np.median(np.sqrt(insub))) / r["n_steps"]
            lam = r["lam"]
            ratio = realized / (lam * TARGET_PERSTEP)
            ok = DOSE_BAND[0] <= ratio <= DOSE_BAND[1]
            rep["cells"][f"{arm}_s{s}"] = {"p4_realized_over_target": ratio,
                                           "p4_delivered": ok}
            if not ok:
                rep["void"].append({"cell": f"{arm}_s{s}",
                                    "reason": f"P4 dose not delivered (ratio {ratio:.3f})"})
    _w()

    # ---- P5: divergence verdicts (vs the seed-matched baseline) -----------------------------
    for s in SEEDS:
        ref = C[("baseline", s)]["final_loss_mean5"]
        for arm in ("incore16", "leverage16"):
            fl = C[(arm, s)]["final_loss_mean5"]
            v = RH.divergence_verdict(fl, ref, retried=C[(arm, s)]["eta_scale"] < 1.0)
            cell = rep["cells"].setdefault(f"{arm}_s{s}", {})
            cell["p5_final_loss"] = fl
            cell["p5_ref"] = ref
            cell["p5_verdict"] = v
            if v != "ok":
                rep["void"].append({"cell": f"{arm}_s{s}", "reason": f"P5 {v}"})
    voided = {v["cell"] for v in rep["void"]}
    rep["n_void"] = len(voided)
    _w()

    # ---- P2: baseline s0 loss-trace continuity vs banked tenant_x4/adamw_s0 -----------------
    tx = _pull("tenant_x4/out/adamw_s0/metrics.json", out / "in/tenant_adamw_s0.json")
    ref_losses = [r2["shares"] is not None for r2 in tx["grad_shares"]]  # presence only
    b0 = C[("baseline", 0)]
    # the recipes share code path/order; compare final losses (bit-level equality is not
    # expected across the two jobs' logging differences — band ±5% frozen here)
    p2 = abs(b0["final_loss_mean5"] - tx["final_loss_mean5"]) / tx["final_loss_mean5"]
    rep["p2_continuity"] = {"referee_baseline_s0": b0["final_loss_mean5"],
                            "banked_tenant_adamw_s0": tx["final_loss_mean5"],
                            "rel_diff": p2, "pass": bool(p2 <= 0.05)}
    _w()

    # ---- contrasts per axis (frozen rules) ---------------------------------------------------
    def paired(a_arm, b_arm, s, key):
        A = np.asarray(C[(a_arm, s)]["evals"]["end"][key], float)
        B = np.asarray(C[(b_arm, s)]["evals"]["end"][key], float)
        m = ~(np.isnan(A) | np.isnan(B))
        d = A[m] - B[m]
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
        boots = d[idx].mean(axis=1)
        return {"d": float(d.mean()), "lo95": float(np.percentile(boots, 2.5)),
                "hi95": float(np.percentile(boots, 97.5)), "n": int(d.size)}

    axes = {}
    live = [s for s in SEEDS if f"incore16_s{s}" not in voided]
    rep["live_seeds"] = live
    for key, axis in (("task_per_item", "task"), ("battery_per_item", "battery")):
        axes[axis] = {}
        for ctrl in ("baseline", "leverage16"):
            per_seed = [paired("incore16", ctrl, s, key) for s in live
                        if f"{ctrl}_s{s}" not in voided or ctrl == "baseline"]
            sep = (len(per_seed) > 0
                   and all(r["hi95"] < 0 or r["lo95"] > 0 for r in per_seed))
            axes[axis][f"incore_vs_{ctrl}"] = {"per_seed": per_seed,
                                               "separates_all_seeds": bool(sep)}
    ppl = {arm: [C[(arm, s)]["evals"]["end"]["ppl"] for s in SEEDS
                 if f"{arm}_s{s}" not in voided or arm == "baseline"]
           for arm in ARMS}
    axes["ppl"] = {"values": ppl}
    for ctrl in ("baseline", "leverage16"):
        di = np.mean(ppl["incore16"]) - np.mean(ppl[ctrl])
        sd = np.std(ppl["incore16"] + ppl[ctrl], ddof=1)
        axes["ppl"][f"incore_vs_{ctrl}"] = {
            "d_mean": float(di), "pooled_sd": float(sd),
            "separates": bool(abs(di) > 2.0 * sd * np.sqrt(2.0 / max(len(live), 1)))}
    rep["axes"] = axes
    _w()

    # ---- §1 mechanical adjudication + §3 branch ---------------------------------------------
    # RF-E1 fix: the adjudication is GATED on dose delivery — if every dose cell is VOID the
    # §3 branches (which presuppose delivered doses) may not fire; the MDE outcome is
    # VOID-instrument, terminal and clean per §9.
    dose_cells = [f"{arm}_s{s}" for arm in ("incore16", "leverage16") for s in SEEDS]
    if all(c in voided for c in dose_cells):
        rep["adjudication"] = {
            "BRANCH": "VOID-instrument: no dose cell delivered its dose (P4) and/or "
                      "survived P5 — the section-3 branches may not fire; terminal per "
                      "section 9",
            "task_separates_both": None, "collateral_separates_both": None,
            "generic_pattern": None}
        _w()
        print("REFEREE_AGGREGATE_OK", json.dumps(
            {"branch": "VOID-instrument", "n_void": rep["n_void"],
             "p2": rep["p2_continuity"]["pass"]}), flush=True)
        return
    task_sep = axes["task"]["incore_vs_baseline"]["separates_all_seeds"] and \
        axes["task"]["incore_vs_leverage16"]["separates_all_seeds"]
    coll_sep = (axes["battery"]["incore_vs_baseline"]["separates_all_seeds"]
                and axes["battery"]["incore_vs_leverage16"]["separates_all_seeds"]) or \
        (axes["ppl"]["incore_vs_baseline"]["separates"]
         and axes["ppl"]["incore_vs_leverage16"]["separates"])
    generic = (not task_sep and not coll_sep
               and (axes["task"]["incore_vs_baseline"]["separates_all_seeds"]
                    or axes["ppl"]["incore_vs_baseline"]["separates"])
               and not axes["task"]["incore_vs_leverage16"]["separates_all_seeds"])
    if task_sep:
        branch = "H_signal-flavored separation -> FULL LADDER"
    elif coll_sep:
        branch = "H_guardian-flavored separation -> FULL LADDER"
    elif generic:
        branch = "H_generic -> ladder on the control arm"
    else:
        branch = "H_noise-neutral at scope -> lower doses CANCELLED (early close)"
    rep["adjudication"] = {"task_separates_both": bool(task_sep),
                          "collateral_separates_both": bool(coll_sep),
                          "generic_pattern": bool(generic), "BRANCH": branch}
    _w()
    print("REFEREE_AGGREGATE_OK", json.dumps({"branch": branch, "n_void": rep["n_void"],
                                              "p2": rep["p2_continuity"]["pass"]}),
          flush=True)


if __name__ == "__main__":
    main()
