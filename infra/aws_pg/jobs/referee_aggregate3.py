"""REFEREE aggregate v3 — POST-CRITIC, full-grid unblind (supersedes v1/v2, kept as
snapshots). Implements every fold: P4 horizon-matched to the seed-baseline's own final-audit
out-of-core motion (F1/RF-E2), reported under BOTH bands with authorship labels (F4);
P4 = N/A for P5-failed cells (F8); the P5 retry chain finalized from the levretry cells
(F3); sign-conditioned §1 adjudication (F7/RF-E5); dose-response monotonicity; task-2
replication; loss-trace P2 (F15); SAP F-referee m=5 BH.

    python -m infra.aws_pg.jobs.referee_aggregate3 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
SEEDS = (0, 1, 2)
N_BOOT = 10000
BOOT_SEED = 20260806
BAND_FROZEN = (0.7, 1.4)          # the only pre-data numeric band (P3)
BAND_POST = (0.5, 2.0)            # post-unblind-authored (RF-E4), labeled
CELLS = {  # name -> (prefix template, arm-tag inside)
    "baseline": "referee_mde_s{s}/out/baseline",
    "incore16": "referee_mde_s{s}/out/incore16",
    "leverage16": "referee_mde_s{s}/out/leverage16",
    "incore1": "referee_ladder_L1/out/incore1_s{s}",
    "incore4": "referee_ladder_L2/out/incore4_s{s}",
    "haar16": "referee_ladder_L3/out/haar16_s{s}",
    "levretry": "referee_ladder_L4/out/levretry_s{s}",
    "t2base": "referee_ladder_T2a/out/t2base_s{s}",
    "t2incore": "referee_ladder_T2b/out/t2incore_s{s}",
    "t2lev": "referee_ladder_T2c/out/t2lev_s{s}",
    "t2incore_r": "referee_ladder_T2r/out/t2incore_r_s{s}",
    "t2lev_r": "referee_ladder_T2r/out/t2lev_r_s{s}"}
ONLY_SEEDS = {"t2incore_r": (0,), "t2lev_r": (0, 1)}   # the retry chain's actual cells
LAM = {"incore16": 16, "leverage16": 16, "incore1": 1, "incore4": 4, "haar16": 16,
       "levretry": 16, "t2incore": 16, "t2lev": 16, "t2incore_r": 16, "t2lev_r": 16}


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

    rep = {"version": "post-critic v3", "cells": {}, "void": [], "p5_chain": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    C = {}
    for name, tpl in CELLS.items():
        for s in ONLY_SEEDS.get(name, SEEDS):
            C[(name, s)] = _pull(f"{tpl.format(s=s)}/metrics.json",
                                 out / f"in/{name}_s{s}.json")

    def seeds_of(name):
        return ONLY_SEEDS.get(name, SEEDS)

    def base_of(name):
        return "t2base" if name.startswith("t2") else "baseline"

    def oocore_target(base_name, s):
        r = C[(base_name, s)]
        aud = r["motion_audit"][-1]["audit"]
        oo = [aud[k]["tot"] - aud[k]["in_core"] for k in sorted(aud)]
        return float(np.median(np.sqrt(np.maximum(oo, 0)))) / r["n_steps"]

    def realized(name, s):
        r = C[(name, s)]
        aud = r["motion_audit"][-1]["audit"]
        key = "in_sub" if "in_sub" in aud[sorted(aud)[0]] else "in_core"
        v = [aud[k][key] for k in sorted(aud)]
        return float(np.median(np.sqrt(v))) / r["n_steps"]

    # ---- P5 chain first (F8: P4 meaningless post-divergence) --------------------------------
    for name in [n for n in CELLS if n not in ("baseline", "t2base")]:
        for s in seeds_of(name):
            ref = C[(base_of(name), s)]["final_loss_mean5"]
            fl = C[(name, s)]["final_loss_mean5"]
            retried = C[(name, s)]["eta_scale"] < 1.0
            v = RH.divergence_verdict(fl, ref, retried=retried)
            rep["p5_chain"][f"{name}_s{s}"] = {"final": fl, "ref": ref, "retried": retried,
                                               "verdict": v}
            if v == "VOID":
                rep["void"].append({"cell": f"{name}_s{s}", "reason": "P5 VOID (post-retry)"})
            elif v == "retry_halved":
                rep["void"].append({"cell": f"{name}_s{s}",
                                    "reason": "P5 first-offence (superseded by the levretry "
                                              "chain)" if name == "leverage16"
                                    else "P5 retry pending"})
    voided = {v["cell"] for v in rep["void"]}
    _w()

    # ---- P4 horizon-matched (F1), N/A for P5-failed -----------------------------------------
    for name in [n for n in CELLS if n in LAM]:
        for s in seeds_of(name):
            cell = f"{name}_s{s}"
            if cell in voided:
                rep["cells"][cell] = {"p4": "N/A (P5-failed; post-divergence weights)"}
                continue
            t = oocore_target(base_of(name), s)
            ratio = realized(name, s) / (LAM[name] * t)
            rep["cells"][cell] = {
                "p4_ratio_horizon_matched": ratio,
                "delivered_frozen_band": bool(BAND_FROZEN[0] <= ratio <= BAND_FROZEN[1]),
                "delivered_post_band": bool(BAND_POST[0] <= ratio <= BAND_POST[1])}
            if not (BAND_POST[0] <= ratio <= BAND_POST[1]):
                rep["void"].append({"cell": cell, "reason": f"P4 undelivered ({ratio:.3f})"})
    voided = {v["cell"] for v in rep["void"]}
    rep["band_disclosure"] = ("frozen band [0.7,1.4] is the P3 band (pre-data); [0.5,2.0] "
                              "was post-unblind-authored (RF-E4) — both reported; "
                              "adjudication uses the post band WITH this label")
    _w()

    # ---- contrasts ---------------------------------------------------------------------------
    def paired(a_name, b_name, s, key):
        A = np.asarray(C[(a_name, s)]["evals"]["end"][key], float)
        B = np.asarray(C[(b_name, s)]["evals"]["end"][key], float)
        m = ~(np.isnan(A) | np.isnan(B))
        d = A[m] - B[m]
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
        boots = d[idx].mean(axis=1)
        return {"d": float(d.mean()), "lo95": float(np.percentile(boots, 2.5)),
                "hi95": float(np.percentile(boots, 97.5))}

    def live(name):
        return [s for s in seeds_of(name) if f"{name}_s{s}" not in voided]

    con = {}
    for name in ("incore1", "incore4", "incore16", "leverage16", "levretry", "haar16",
                 "t2incore", "t2lev", "t2incore_r", "t2lev_r"):
        b = base_of(name)
        con[name] = {"live_seeds": live(name)}
        for key, axis in (("task_per_item", "task"), ("battery_per_item", "battery")):
            con[name][axis] = [paired(name, b, s, key) for s in live(name)]
        con[name]["ppl"] = [C[(name, s)]["evals"]["end"]["ppl"] for s in live(name)]
        con[name]["ppl_base"] = [C[(b, s)]["evals"]["end"]["ppl"] for s in live(name)]
    rep["contrasts"] = con
    _w()

    # ---- §1 adjudication, sign-conditioned (F7) ---------------------------------------------
    def sep_all(rows, sign):
        if not rows:
            return False
        return all((r["lo95"] > 0) if sign > 0 else (r["hi95"] < 0) for r in rows)

    inc = con["incore16"]
    lev_alive = con["levretry"] if live("levretry") else con["leverage16"]
    task_improves = sep_all(inc["task"], +1)
    task_degrades = sep_all(inc["task"], -1)
    coll_degrades = sep_all(inc["battery"], -1) or \
        (len(inc["ppl"]) > 0 and min(inc["ppl"]) > max(inc["ppl_base"]) + 5.0)
    dose_task = [float(np.mean([r["d"] for r in con[n]["task"]]) if con[n]["task"] else
                       np.nan) for n in ("incore1", "incore4", "incore16")]
    dose_ppl = [float(np.mean(con[n]["ppl"]) if con[n]["ppl"] else np.nan)
                for n in ("incore1", "incore4", "incore16")]
    monotone_ppl = all(x == x for x in dose_ppl) and dose_ppl[0] <= dose_ppl[1] <= dose_ppl[2]
    monotone_task = all(x == x for x in dose_task) and \
        (dose_task[0] >= dose_task[1] >= dose_task[2] or
         dose_task[0] <= dose_task[1] <= dose_task[2])
    rep["dose_response"] = {"task_d_by_lambda": dose_task, "ppl_by_lambda": dose_ppl,
                            "monotone_task": bool(monotone_task),
                            "monotone_ppl": bool(monotone_ppl)}
    rep["adjudication"] = {
        "task_improves_beyond_base": bool(task_improves),
        "task_degrades": bool(task_degrades),
        "collateral_degrades": bool(coll_degrades),
        "incore_less_damaged_than_matched_dose_controls": None,   # filled below
        "reading": None}
    # is in-core damage SMALLER than controls at matched nominal dose (guardian requires WORSE)?
    less = []
    for ctrl in ("leverage16", "levretry", "haar16"):
        for s in set(live("incore16")) & set(live(ctrl)):
            d = paired("incore16", ctrl, s, "task_per_item")
            less.append(d["lo95"] > 0)
    rep["adjudication"]["incore_less_damaged_than_matched_dose_controls"] = \
        (all(less) if less else None)
    if task_degrades and coll_degrades and (all(less) if less else False):
        reading = ("MIXED (reported mixed per §6): raw-step dosing at delivered ~16-27x "
                   "damages ALL subspaces; the in-core direction is the LEAST damaging "
                   "place to put raw motion — the OPPOSITE of H_guardian's core-specific "
                   "harm; H_signal absent (no improvement anywhere); H_noise-neutral false "
                   "at this dose (effects everywhere); H_generic false (subspaces separate). "
                   "Scope: delivered dose far above the designed regime.")
    else:
        reading = "see flags; mixed/partial — reported as measured"
    rep["adjudication"]["reading"] = reading
    _w()

    # ---- motion-vs-damage census (post-hoc, labeled; the Result-3 artifact) ----------------
    census = []
    for name in ("incore1", "incore4", "incore16", "levretry", "t2incore", "t2incore_r",
                 "t2lev", "t2lev_r"):
        for s in live(name):
            cellrec = rep["cells"].get(f"{name}_s{s}", {})
            ratio = cellrec.get("p4_ratio_horizon_matched")
            if not isinstance(ratio, float):
                continue
            b = base_of(name)
            census.append({
                "cell": f"{name}_s{s}", "task": C[(name, s)]["task"],
                "subspace": ("core" if "incore" in name else "leverage"),
                "motion_units": ratio * LAM[name],
                "ppl": C[(name, s)]["evals"]["end"]["ppl"],
                "ppl_base": C[(b, s)]["evals"]["end"]["ppl"]})
    rep["motion_damage_census"] = {
        "rows": sorted(census, key=lambda r_: r_["motion_units"]),
        "note": "post-hoc, labeled: is collateral a function of realized motion alone, or "
                "does the subspace matter at matched motion? Pairs at comparable motion "
                "adjudicate; n is small and the two tasks are reported separately."}
    _w()

    # ---- SAP F-referee m=5 ------------------------------------------------------------------
    members = []
    for nm, axis, key in (("task_incore_vs_lev", "task", "task_per_item"),
                          ("battery_incore_vs_lev", "battery", "battery_per_item")):
        ps = []
        for s in set(live("incore16")) & set(live("leverage16") + live("levretry")):
            ctrl = "levretry" if s in live("levretry") else "leverage16"
            st = paired("incore16", ctrl, s, key)
            d = st["d"]
            rng = np.random.default_rng(BOOT_SEED)
            A = np.asarray(C[("incore16", s)]["evals"]["end"][key], float)
            B = np.asarray(C[(ctrl, s)]["evals"]["end"][key], float)
            m = ~(np.isnan(A) | np.isnan(B))
            dd = A[m] - B[m]
            idx = rng.integers(0, dd.size, size=(N_BOOT, dd.size))
            boots = dd[idx].mean(axis=1)
            ps.append(float((1 + (boots <= 0).sum()) / (N_BOOT + 1)))
        members.append({"name": nm, "p": max(ps) if ps else float("nan")})
    ppl_i = con["incore16"]["ppl"]
    ppl_l = (con["levretry"]["ppl"] or con["leverage16"]["ppl"])
    members.append({"name": "ppl_incore_vs_lev",
                    "p": float("nan"), "census": {"incore": ppl_i, "lev": ppl_l},
                    "note": "seed-level census (n small; VOID-adjusted)"})
    members.append({"name": "incore_dose_trend",
                    "p": float("nan"), "census": rep["dose_response"]})
    t2 = {"task": [r for r in con["t2incore"]["task"]],
          "vs_lev": [paired("t2incore", "t2lev", s, "task_per_item")
                     for s in set(live("t2incore")) & set(live("t2lev"))]}
    members.append({"name": "task2_replication", "p": float("nan"), "census": t2})
    pv = [m_ for m_ in members if m_["p"] == m_["p"]]
    srt = sorted((m_["p"], m_["name"]) for m_ in pv)
    k_star = max([i for i, (p, _) in enumerate(srt, 1) if p <= 0.05 * i / 5], default=0)
    rep["families"] = {"F-referee": {"members": members, "m": 5,
                                     "m_pvalued": len(pv),
                                     "bh": {m_["name"]: m_["name"] in
                                            {n for _, n in srt[:k_star]} for m_ in pv}}}
    _w()
    print("REFEREE_AGG3_OK", json.dumps({"n_void": len(rep["void"]),
                                         "reading": reading[:80]}), flush=True)


if __name__ == "__main__":
    main()
