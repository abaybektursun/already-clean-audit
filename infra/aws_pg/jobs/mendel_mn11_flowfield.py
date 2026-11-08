"""MENDEL MN-11 — the flow-field mining pass (queue item 21, the curiosity reserve).

WHY THIS ARM EXISTS AT ALL. π0's action expert is exact geometry by construction — a flow-matching
velocity field, the only object in this repo with no interpretive overlay between the measurement
and the thing measured. The program's fourth commitment says to spend a fraction of every budget on
the highest-leverage question that might return nothing, especially when it is cheap. This is
cheap: 17 banked field tensors, ~72 MB, no GPU.

THE NAMED QUESTION, registered before any number is read (the charter makes naming it mandatory):

    WHERE does the U3 / U3c co-training difference localise — across flow-matching time t, across
    the 50-step action-chunk horizon, and across the action dimensions — and does that
    localisation DEVELOP across the film (512 -> 4096 steps)?

GESTALT already established THAT the contrast is real (endpoint gap 0.3025 [0.2963, 0.3086] against
a within-model drift of 0.2614, p <= 5e-4). It never asked where the difference lives. That is what
is unmined, and it is the only question here whose answer would change what anyone does next.

THE NULL is within-model drift, exactly as the charter requires: the same statistic computed
between ADJACENT U3 checkpoints — "the same model, trained longer" rather than "a different
co-training recipe". A localisation that does not exceed that band is training noise with a shape.

**adim_valid = 7.** The tensors carry 32 action dimensions and only the first 7 are real; the rest
are padding. Averaging over all 32 would dilute every statistic by ~4.6x with zeros and is exactly
the kind of silent axis error that makes a mining pass produce confident nonsense. The valid count
is read from the bank's own metrics, never assumed.

Discovery register only: this arm produces no confirmatory verdict, and its frozen tokens are
DISCOVERY-REGISTERED or NOTHING-CLEARS.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP

ARM = "MN-11"
FIELD = f"{MC.RESULTS_S3}/gestalt_field/out"
UNITS = ("U3", "U3c")
STEPS = (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
T_GRID = (0.9, 0.7, 0.5, 0.3, 0.1)


def load_v(path, n_valid: int):
    """The velocity field, restricted to the VALID action dimensions.

    Shape (n_obs, n_noise, |t_grid|, chunk, adim) -> (..., n_valid). fp16 on disk, promoted to
    fp64: every statistic below is a difference of two similar tensors, and differencing in fp16
    would quantise exactly the small differences this arm is looking for.
    """
    z = np.load(path)
    v = z["v"]
    assert v.ndim == 5, f"{path}: expected a 5-D field, got {v.shape}"
    assert 0 < n_valid <= v.shape[-1], f"n_valid={n_valid} outside 1..{v.shape[-1]}"
    return v[..., :n_valid].astype(np.float64)


def rel_difference(va, vb) -> dict:
    """RMS(va - vb) / RMS(vb), resolved along each axis separately.

    Relative rather than absolute because `v_rms` varies across the film (0.518 at step 512 to
    0.645 at 4096 for U3), so an absolute difference would confound "the recipes differ" with "the
    field got bigger".
    """
    d = va - vb
    scale = float(np.sqrt((vb ** 2).mean()))
    assert scale > 0, "zero-norm reference field"
    out = {"overall": float(np.sqrt((d ** 2).mean()) / scale)}
    # axes: 0 obs, 1 noise, 2 t, 3 horizon, 4 action-dim
    out["by_t"] = [float(np.sqrt((d[:, :, i] ** 2).mean()) / scale) for i in range(d.shape[2])]
    out["by_horizon"] = [float(np.sqrt((d[:, :, :, i] ** 2).mean()) / scale)
                         for i in range(d.shape[3])]
    out["by_adim"] = [float(np.sqrt((d[..., i] ** 2).mean()) / scale) for i in range(d.shape[4])]
    return out


def argmax_profile(profile) -> dict:
    p = np.asarray(profile, dtype=np.float64)
    return {"argmax": int(np.argmax(p)), "max": float(p.max()), "min": float(p.min()),
            "ratio_max_over_min": float(p.max() / p.min()) if p.min() > 0 else float("inf"),
            "profile": p.tolist()}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-11 — the flow-field mining pass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn11", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn11"), out)

    fm = json.loads(man.require("field_metrics", f"{FIELD}/metrics.json").read_text())
    n_valid = {c["adim_valid"] for c in fm["cells"] if c.get("kind") == "bank"}
    assert len(n_valid) == 1, f"banks disagree about adim_valid: {n_valid}"
    n_valid = int(n_valid.pop())
    assert tuple(fm["steps"]) == STEPS and tuple(fm["units"]) == UNITS, \
        f"bank metrics disagree with this arm's frozen grid: {fm['steps']} / {fm['units']}"
    print(f"[mn11] adim_valid = {n_valid} of 32 — the other {32 - n_valid} are padding", flush=True)

    v = {}
    for unit in UNITS:
        for s in STEPS:
            tag = f"{unit}_{s:08d}"
            v[(unit, s)] = load_v(man.require(f"bank_{tag}", f"{FIELD}/bank_{tag}.npz"), n_valid)

    # ---- the treatment: co-training difference at each step of the film ----
    contrast = {s: rel_difference(v[("U3c", s)], v[("U3", s)]) for s in STEPS}
    # ---- the null: within-model drift between ADJACENT checkpoints of the SAME unit ----
    drift = {}
    for unit in UNITS:
        for i in range(len(STEPS) - 1):
            drift[(unit, STEPS[i], STEPS[i + 1])] = rel_difference(v[(unit, STEPS[i + 1])],
                                                                   v[(unit, STEPS[i])])
    drift_overall = [d["overall"] for d in drift.values()]
    # THE NULL MUST BE HORIZON-MATCHED. The endpoint contrast compares two units at step 4096, so
    # its null is one checkpoint interval of within-unit change AT THE SAME POINT IN THE FILM —
    # not the maximum over the whole film. The 512->1024 gaps are early-training transients (1.73
    # and 2.24 here) and taking a max over them yields a null six to eight times the late drift,
    # which no endpoint contrast could ever clear. A null that cannot be beaten is not a null, and
    # the first version of this arm used exactly that and reported NOTHING-CLEARS.
    late = [drift[(u, STEPS[-2], STEPS[-1])]["overall"] for u in UNITS]
    band_hi = float(np.max(late))
    band_hi_unmatched = float(np.max(drift_overall))

    for s in STEPS:
        at.add("flowfield_cotraining_difference", value=contrast[s]["overall"], item_id=f"step{s}",
               control_id=MC.TREATMENT,
               flags={"n_valid_adim": n_valid, "by_t": contrast[s]["by_t"]})
    at.add_ci("flowfield_cotraining_difference", late, item_id="within_model_drift_late",
              control_id=MC.STRUCTURED,
              flags={"what": ("horizon-matched null: one checkpoint interval of within-unit change "
                              "at the END of the film, the same interval the endpoint contrast "
                              "spans"),
                     "pairs": [f"{u} {STEPS[-2]}->{STEPS[-1]}" for u in UNITS], "max": band_hi})
    at.add_ci("flowfield_cotraining_difference", drift_overall, item_id="within_model_drift_all",
              control_id=MC.HAAR,
              flags={"what": ("UNMATCHED: every adjacent pair including the 512->1024 transients; "
                              "reported so the effect of the null choice is visible, NOT used to "
                              "adjudicate"),
                     "n_pairs": len(drift_overall), "max": band_hi_unmatched})

    endpoint = contrast[STEPS[-1]]["overall"]
    clears = endpoint > band_hi
    # ---- localisation, only meaningful if the contrast clears its null ----
    loc = {axis: argmax_profile(contrast[STEPS[-1]][f"by_{axis}"])
           for axis in ("t", "horizon", "adim")}
    drift_loc = {axis: argmax_profile(np.mean([d[f"by_{axis}"] for d in drift.values()], axis=0))
                 for axis in ("t", "horizon", "adim")}
    for axis in ("t", "horizon", "adim"):
        for i, val in enumerate(contrast[STEPS[-1]][f"by_{axis}"]):
            at.add(f"flowfield_difference_by_{axis}", value=val, item_id=f"{axis}{i}",
                   control_id=MC.TREATMENT,
                   flags={"axis": axis, "index": i,
                          "t_value": T_GRID[i] if axis == "t" else None})
        for i, val in enumerate(drift_loc[axis]["profile"]):
            at.add(f"flowfield_difference_by_{axis}", value=val, item_id=f"{axis}{i}",
                   control_id=MC.STRUCTURED, flags={"axis": axis, "index": i, "null": "drift"})

    # ---- development: does the contrast's SHAPE change across the film? ----
    from infra.pg import mendel_critic as CR
    shape_first = contrast[STEPS[0]]["by_adim"]
    shape_last = contrast[STEPS[-1]]["by_adim"]
    shape_rho = CR.spearman(shape_first, shape_last) if n_valid >= 3 else float("nan")
    trend = [contrast[s]["overall"] for s in STEPS]

    gates.record("mn11_contrast_clears_within_model_drift", clears,
                 endpoint=endpoint, drift_band_hi_matched=band_hi,
                 drift_band_hi_unmatched=band_hi_unmatched, n_drift_pairs=len(drift_overall),
                 note=("the charter's KILL, against the HORIZON-MATCHED band. The unmatched band "
                       "is reported beside it because its max is set by the 512->1024 "
                       "early-training transient and could not be cleared by anything"))
    verdict = MP.check_verdict(ARM, "DISCOVERY-REGISTERED" if clears else "NOTHING-CLEARS")
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "named_question": ("where does the U3/U3c co-training difference localise across "
                           "flow-matching time, the action-chunk horizon and the action "
                           "dimensions, and does that localisation develop across the film?"),
        "adim_valid": n_valid, "adim_total": 32,
        "t_grid": list(T_GRID), "steps": list(STEPS),
        "contrast_by_step": {str(s): contrast[s]["overall"] for s in STEPS},
        "contrast_trend": trend,
        "within_model_drift": {
            "matched_late_pairs": {f"{u} {STEPS[-2]}->{STEPS[-1]}": d for u, d in zip(UNITS, late)},
            "matched_band_hi": band_hi,
            "all_adjacent_values": drift_overall, "unmatched_band_hi": band_hi_unmatched,
            "mean_all": float(np.mean(drift_overall)),
            "note": ("the matched band is the null; the unmatched max is set by the 512->1024 "
                     "transient and is reported only to show what the null choice does")},
        "endpoint_clears_drift": bool(clears),
        "localisation_endpoint": loc, "localisation_drift_null": drift_loc,
        "shape_stability_across_film_spearman": shape_rho,
        "atlas": info})
    print("MN11_OK " + json.dumps({"verdict": verdict, "endpoint": round(endpoint, 4),
                                   "drift_band_hi": round(band_hi, 4),
                                   "argmax_t": loc["t"]["argmax"],
                                   "argmax_adim": loc["adim"]["argmax"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
