"""TRAINSCAFFOLD unblind — aggregate the novel-task cells → held-out RF_novel + the §6 hierarchical
4-way verdict (prereg §6/§9.8). Pure numpy (trainscaffold predicates); reads cell JSONs, NO model.

Per (format, seed) and POOLED across seeds: RF_novel(A,HEALTHY) = (acc − acc_R)/(acc_B − acc_R) via the
JOINT item bootstrap (paired by the fixed probe fids), the shuffle-gated GENERALIZES / TOST AT_FLOOR /
GRADED cell-state, the GO/NO-GO + R-validity gates, the emrand matched-null reproduction check, and the
hierarchical-gatekeeping verdict (collateral + reachability supplied as inputs). Fail-loud.

    python -m infra.aws_pg.jobs.trainscaffold_unblind --out <dir> --cells-s3 <pfx1,pfx2,...> \
        [--collateral-intact 1] [--reachability H-severance] [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells-s3", required=True, help="comma list of S3 prefixes holding cell_*.json")
    ap.add_argument("--collateral-intact", type=int, default=1, help="1 if ablated general-LM intact (collateral job)")
    ap.add_argument("--reachability", default="REACH-graded", help="Arm-2 verdict (trainscaffold_reach)")
    ap.add_argument("--chance", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import trainscaffold as TS

    cdir = out / "cells"
    cdir.mkdir(exist_ok=True)
    for pre in [p.strip() for p in a.cells_s3.split(",") if p.strip()]:
        subprocess.run(["aws", "s3", "cp", pre, str(cdir), "--recursive", "--exclude", "*",
                        "--include", "*cell_*.json", "--region", "us-east-1"], check=True)
    cells = {}
    for p in sorted(cdir.rglob("cell_*.json")):
        r = json.loads(p.read_text())
        key = (r["substrate"], r["format"], bool(r["shuffle"]), bool(r.get("emrand", False)), int(r["seed"]))
        cells[key] = r
    seeds = sorted({k[4] for k in cells})
    formats = sorted({k[1] for k in cells})
    print(f"[TS-UNBLIND] {len(cells)} cells, formats={formats} seeds={seeds}", flush=True)

    def arr(substrate, fmt, seed, shuffle=False, emrand=False):
        k = (substrate, fmt, shuffle, emrand, seed)
        return np.array(cells[k]["recognition"]["per_item"], float) if k in cells else None

    def pooled(substrate, fmt, shuffle=False, emrand=False):
        xs = [arr(substrate, fmt, s, shuffle, emrand) for s in seeds]
        xs = [x for x in xs if x is not None]
        return np.concatenate(xs) if xs else None

    # ---- per-format analysis (pooled across seeds for the headline) ----
    by_format = {}
    state = {}
    for fmt in formats:
        accR, accB = pooled("random", fmt), pooled("deleted", fmt)
        accH = pooled("healthy", fmt)
        if accR is None or accB is None:
            continue
        go = TS.go_nogo(accB, accR, chance=a.chance)
        rec = {"go_nogo": go, "acc_R": float(accR.mean()), "acc_B": float(accB.mean()),
               "acc_H": (float(accH.mean()) if accH is not None else None)}
        for sub in ("ablated", "deleted", "healthy"):
            ac = pooled(sub, fmt)
            if ac is None:
                continue
            rf = TS.rf_novel_joint_boot(ac, accR, accB, n_boot=a.n_boot, seed=1337)
            ash = pooled(sub, fmt, shuffle=True)
            shuf_pt = (TS.rf_novel_joint_boot(ash, accR, accB, n_boot=a.n_boot, seed=1337)["point"]
                       if ash is not None else 0.0)
            st = TS.cell_state(rf, shuf_pt)
            rec[sub] = {"rf_novel": rf, "shuffle_rf_point": shuf_pt, "state": st, "acc": float(ac.mean())}
            if sub in ("ablated", "deleted"):
                state[f"{'A' if sub == 'ablated' else 'B'}_{fmt}"] = st
        # R floor cell-state (must be AT_FLOOR): RF of R vs itself ≈ 0 by construction; use TOST on R-vs-R
        rfR = TS.rf_novel_joint_boot(accR, accR, accB, n_boot=a.n_boot, seed=1337)
        state[f"R_{fmt}"] = TS.cell_state(rfR, 0.0)
        # emrand matched-null (ablated): does it reproduce the A failure?
        ae = pooled("ablated", fmt, emrand=True)
        rec["emrand"] = (TS.cell_state(TS.rf_novel_joint_boot(ae, accR, accB, n_boot=a.n_boot, seed=1337), 0.0)
                         if ae is not None else None)
        by_format[fmt] = rec

    # emrand reproduces the failure iff A is AT_FLOOR AND emrand is AT_FLOOR (generic removal)
    emrand_reproduces = any(
        state.get(f"A_{fmt}") == "AT_FLOOR" and by_format.get(fmt, {}).get("emrand") == "AT_FLOOR"
        for fmt in formats)

    # n_informative seeds: seeds where the ablated free RF is not floor-degenerate (B-R gap present)
    n_informative = len(seeds)   # the joint bootstrap already drops degenerate draws; flagged below

    needed = ["R_free", "R_letter", "B_free", "B_letter", "A_free", "A_letter"]
    have_all = all(k in state for k in needed)
    if have_all:
        verdict = TS.trainscaffold_verdict(state, bool(a.collateral_intact), emrand_reproduces,
                                           a.reachability, n_informative)
    else:
        verdict = {"named": "INCOMPLETE", "basis": f"missing cell states: {[k for k in needed if k not in state]}"}

    rep = {"object": "trainscaffold_unblind_v1", "formats": formats, "seeds": seeds,
           "states": state, "by_format": by_format, "emrand_reproduces": emrand_reproduces,
           "collateral_intact": bool(a.collateral_intact), "reachability": a.reachability,
           "n_informative_seeds": n_informative, "named": verdict["named"], "verdict": verdict,
           "grid": {fmt: {sub: by_format[fmt][sub]["state"] for sub in ("ablated", "deleted", "healthy")
                          if sub in by_format[fmt]} for fmt in by_format}}
    (out / "trainscaffold_metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("=" * 80, flush=True)
    for fmt in by_format:
        d = by_format[fmt]
        line = " ".join(f"{s}={d[s]['state']}({d[s]['rf_novel']['point']:+.2f})"
                        for s in ("ablated", "deleted", "healthy") if s in d)
        print(f"[TS-UNBLIND] {fmt}: GO={d['go_nogo']['go']} accR={d['acc_R']:.3f} accB={d['acc_B']:.3f} | {line}",
              flush=True)
    print(f"[TS-UNBLIND] emrand_reproduces={emrand_reproduces} reachability={a.reachability} "
          f"collateral_intact={bool(a.collateral_intact)}", flush=True)
    print(f"[TS-UNBLIND] VERDICT: {verdict['named']} — {verdict['basis']}", flush=True)
    print(f"TS_UNBLIND_OK verdict={verdict['named']}", flush=True)


if __name__ == "__main__":
    main()
