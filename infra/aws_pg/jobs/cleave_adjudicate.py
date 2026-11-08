"""CLEAVE adjudication (CPU, pure-python) — apply the frozen §6 verdict order ONCE, on confirmation.

`cleave_protect --phase confirm` measures E_protect / specificity / the ordering null and stops; the
§6 branch order was implemented and branch-tested in `cleave_core.adjudicate_cleave` but never invoked
from the job. This step closes that gap **without touching any measurement**: every input below is
read from an artifact that already exists, and `adjudicate_cleave` is a pure function of them (no
torch, no numpy, no model), so applying it here is still "once, on confirmation".

Inputs (each a metrics.json already on S3):
  --conf    the sealed confirmation      → E_protect, specificity, ordering null, frontiers, naturals
  --mech    the mechanism census         → mechanism_confirms (the unifying prediction's measured leg)
  --target  the targeting axis, optional → targeting_wins (absent ⇒ False, and recorded as absent)
  --banks   the merged Q banks           → resolution_ok at the rank the map arms actually used

The two range guards are recomputed here from the CONFIRMATION's own curves — not inherited from
discovery — because §6's VOID branch asks whether *this* data has removable collateral:
  forget_range_ok    := natural forgetting > δ_forget
  removable_range_ok := (natural − min forgetting anywhere on the NULL's α-curve) > δ_range
`null_saturates` is the saturation guard: the null already removes ~all forgetting at no task cost.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_adjudicate \
      --args "--conf results/cleave_conf/out/metrics.json --mech results/cleave_mech/out/metrics.json \
              --banks results/cleave_banks2/out/metrics.json" \
      --type r7i.xlarge --no-weights --s3-prefix results/cleave_verdict --budget-min 20
"""
import argparse
import json
import os
import pathlib

DELTA_FORGET = 0.01     # the natural update must disturb pretraining at all (spec §6 ceiling)
DELTA_RANGE = 0.005     # ...and some of that disturbance must be REMOVABLE, or there is nothing to protect


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--mech", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--banks", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.pg import cleave_core as CL
    CLS.install_pg()

    def load(key):
        if not key:
            return None
        return json.loads(pathlib.Path(CLS.fetch_s3(key, f"/tmp/cleave/{key.replace('/', '_')}")).read_text())

    conf, mech, target, banks = load(a.conf), load(a.mech), load(a.target), load(a.banks)
    rep = {"object": "cleave_verdict_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"},
           "sources": {"conf": a.conf, "mech": a.mech, "target": a.target, "banks": a.banks}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- range guards, recomputed from the CONFIRMATION's own curves -----------------------
    F_full = conf.get("natural_forgetting_mean")
    assert F_full is not None, "confirmation has no natural_forgetting_mean — cannot judge range"
    null_fr = (conf.get("frontiers") or {}).get(CL.NULL_POLICY)
    assert null_fr, f"confirmation has no '{CL.NULL_POLICY}' frontier — the null is the range reference"
    off_min = min(q["forgetting"] for q in null_fr)
    removable = F_full - off_min
    # the null's task cost at its best-forgetting point (for the saturation guard)
    best = min(null_fr, key=lambda q: q["forgetting"])
    rep["range"] = {"natural_forgetting": F_full, "min_off_forgetting": off_min,
                    "removable": removable, "delta_forget": DELTA_FORGET, "delta_range": DELTA_RANGE,
                    "forget_range_ok": bool(F_full > DELTA_FORGET),
                    "removable_range_ok": bool(removable > DELTA_RANGE),
                    "null_best_point": best}
    # saturation: the null removes essentially ALL the forgetting AND keeps the task
    null_saturates = bool(off_min < 0.1 * F_full and best.get("retention", 0) > 0.9)
    rep["null_saturates"] = null_saturates

    # ---- resolution: was the rank the map arms used actually certified? --------------------
    r_star = ((conf.get("frozen") or {}).get("r_star")
              or (conf.get("r_star_by_support") or {}).get("r_star"))
    sup = (conf.get("support_by_rank") or {}).get(str(r_star)) or (conf.get("support_by_rank") or {}).get(r_star)
    n_sup = (sup or {}).get("n_layers", 0)
    resolution_ok = bool(r_star and n_sup > 0)
    rep["resolution"] = {"r_star": r_star, "n_support_layers": n_sup, "resolution_ok": resolution_ok,
                         "note": ("map arms ran only on the certified support at r*; resolution_ok is "
                                  "False only if NO layer certifies that rank")}

    # ---- wiring: every confirmation cell's own install-vs-trained delta --------------------
    nat = [n for n in (conf.get("naturals") or []) if "skipped" not in n]
    wire = [max(abs(n.get("wiring_delta_gain", 9)), abs(n.get("wiring_delta_forget", 9))) for n in nat]
    wiring_ok = bool(nat and max(wire) <= 0.02)
    rep["wiring"] = {"n_cells": len(nat), "max_delta": max(wire) if wire else None,
                     "wiring_ok": wiring_ok}

    # ---- the mechanism + targeting legs ----------------------------------------------------
    mechanism_confirms = bool((mech or {}).get("mechanism_confirms"))
    targeting_wins = bool((target or {}).get("targeting_wins"))
    rep["legs"] = {"mechanism_confirms": mechanism_confirms,
                   "mechanism_rank_used": (mech or {}).get("mechanism_rank_used"),
                   "targeting_wins": targeting_wins,
                   "targeting_axis_present": target is not None,
                   "note": ("targeting_wins=False when the TARGET axis is absent — recorded as absent, "
                            "not asserted as a negative result")}

    # ---- H_map-hurts descriptor (reported, never a verdict — §6 licenses five branches) ----
    at = conf.get("forget_at_rho") or {}
    map_hurts = None
    if at.get(CL.MAP_POLICY) and at.get(CL.NULL_POLICY):
        import statistics as st
        mv = [v for v in at[CL.MAP_POLICY].values() if v is not None]
        nv = [v for v in at[CL.NULL_POLICY].values() if v is not None]
        if mv and nv:
            map_hurts = bool(st.fmean(mv) > F_full and st.fmean(nv) > F_full)
    rep["map_hurts_task"] = map_hurts

    # ---- THE VERDICT (frozen §6 order, applied once) ---------------------------------------
    verdict = CL.adjudicate_cleave(
        E_protect=conf.get("E_protect") or {"ci_lo": None, "insufficient": True},
        specificity=conf.get("specificity") or {"ci_lo": None, "insufficient": True},
        mechanism_confirms=mechanism_confirms,
        wiring_ok=wiring_ok, resolution_ok=resolution_ok,
        forget_range_ok=rep["range"]["forget_range_ok"],
        removable_range_ok=rep["range"]["removable_range_ok"],
        null_saturates=null_saturates, targeting_wins=targeting_wins,
        map_hurts_task=map_hurts, name="cleave")
    rep["verdict"] = verdict
    rep["green"] = True
    _w()
    print("CLEAVE_VERDICT", json.dumps({"verdict": verdict["verdict"], "reason": verdict["reason"],
                                        "range": rep["range"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
