"""MENDEL MN-16 — v-seeding with the computable dose (queue item 22), the OFFLINE half.

WHAT THE CHARTER SAYS, AND WHY IT NEEDS CORRECTING. Item 22: "seeding v̂ on a capability's
coordinates ... raises the coordinate's effective denominator, pushing its drift-vs-jitter
crossover out by a calculable number of steps — protection per unit of seeded moment becomes a
theory-backed quantity."

**The crossover does not move.** Under SINKFORM Result 6 the crossover is T* = τ·VarX / ρ², and
seeding v̂ to a value v_s scales the drift ρ by 1/√v_s and the jitter VARIANCE by 1/v_s, so

    T*  =  (σ²/v_s) / (μ²/v_s)  =  σ²/μ²  =  1/r²        — INVARIANT in v_s.

A uniform seed scales drift and jitter equally and buys nothing at the crossover. Whatever
protection v-seeding gives is therefore not a shifted crossover; it is a TRANSIENT, and the
quantity to pre-register is the transient's length.

THE CORRECT DOSE→HORIZON LAW. With v̂ seeded to v_0 and a stationary second moment E[g²] = s²,

    v̂_t = β₂^t v_0 + (1 − β₂^t) s²
    suppression(t) = √(s² / v̂_t) = 1 / √(1 + β₂^t (D − 1)),      D = v_0 / s²   (the DOSE)

Suppression starts at 1/√D and relaxes to 1. Taking the half-power point (suppression = 1/√2, i.e.
β₂^T (D−1) = 1):

    **T_p = ln(D − 1) / (−ln β₂)**

Verified against simulation to a ratio of 1.000–1.006 across D ∈ [4, 10⁴] and β₂ ∈ {0.99, 0.999}.

THE PRACTICAL CONSEQUENCE, which is the pre-registerable prediction: **the horizon grows only
LOGARITHMICALLY in the dose.** At β₂ = 0.999, a 10× larger seed buys 2.09× then 1.50× then 1.33×
more horizon. You cannot buy much protection by seeding harder — but a modest dose already covers a
realistic fine-tune (D = 100 gives ~4600 steps), so the method is not useless, it is
range-limited in a specific and checkable way.

THE MUTANT this check must reject: horizon LINEAR in dose, T = c·D. At β₂ = 0.999 the law predicts
2196 → 4593 for D 10 → 100 (2.1×) where the mutant predicts 10×. Trivially separable, which is what
makes the resolution gate pass.

WHAT REMAINS (the GPU half, declared not skipped): the charter's incumbents are EWC-lite AND replay
at matched cost, and its second KILL clause is "realized horizons miss predicted ones". Both need
training runs — three methods × dose rungs × seeds — and a protection benchmark that does not yet
exist in this repo. The offline gate below is what the house requires BEFORE that spend, and it
changes what the runs should measure: a decaying suppression window, not a moved crossover.

Pure math. Runs on the control plane.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mendel_sinkform as SF

ARM = "MN-16"
DOSE_RUNGS = (4, 10, 100, 1000, 10000)
BETA2_GRID = (0.99, 0.999, 0.9999)
HALF_POWER = 1.0 / math.sqrt(2.0)
TOL = 0.02                  # |simulated/predicted − 1| allowed
MIN_SEPARATION = 3.0


def horizon_law(dose: float, beta2: float) -> float:
    """T_p = ln(D − 1) / (−ln β₂): the step at which the seeded suppression has relaxed to 1/√2."""
    assert dose > 1.0, f"a dose of {dose} is not a seed (D must exceed 1)"
    assert 0.0 < beta2 < 1.0, f"beta2 {beta2} outside (0, 1)"
    return math.log(dose - 1.0) / (-math.log(beta2))


MUTANT_ANCHOR = 10.0


def horizon_mutant(dose: float, beta2: float) -> float:
    """THE MUTANT: horizon LINEAR in dose, CALIBRATED TO AGREE WITH THE LAW at D = 10.

    Anchoring it matters. A mutant that is wrong everywhere is easy to reject and proves little;
    one that matches the law at a reference dose and then diverges is the error someone would
    actually make — fit the protection at one rung, assume it scales proportionally, and predict
    10x the horizon for 10x the seed where the truth is 2.09x.
    """
    return horizon_law(MUTANT_ANCHOR, beta2) * (dose / MUTANT_ANCHOR)


def simulate_horizon(dose: float, beta2: float, *, max_steps: int = 2_000_000) -> int:
    """The step at which suppression first reaches 1/√2, by direct recursion on v̂."""
    s2, v, t = 1.0, float(dose), 0
    while 1.0 / math.sqrt(v / s2) < HALF_POWER:
        v = beta2 * v + (1.0 - beta2) * s2
        t += 1
        assert t < max_steps, f"suppression never relaxed at D={dose}, beta2={beta2}"
    return t


def crossover_is_invariant(*, mu: float, sigma: float, dose: float) -> dict:
    """T* under a seeded denominator, computed the way Result 6 computes it.

    Seeding scales the drift by 1/sqrt(v_s) and the jitter VARIANCE by 1/v_s, so the ratio that
    defines the crossover is unchanged. Computed rather than asserted, because this is the step
    that corrects the charter.
    """
    s2 = mu * mu + sigma * sigma
    v_s = dose * s2
    rho_seeded = mu / math.sqrt(v_s)
    var_seeded = sigma * sigma / v_s
    t_star_seeded = var_seeded / (rho_seeded ** 2)
    return {"t_star_unseeded": SF.t_star(mu / sigma), "t_star_seeded": t_star_seeded,
            "ratio": t_star_seeded / SF.t_star(mu / sigma), "dose": dose}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-16 — the computable v-seeding dose")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn16", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)

    # ---- 1. the crossover is invariant: the charter's mechanism, corrected ----
    inv = [crossover_is_invariant(mu=0.3, sigma=1.0, dose=D) for D in DOSE_RUNGS]
    invariant = all(abs(x["ratio"] - 1.0) < 1e-12 for x in inv)
    for x in inv:
        at.add("vseed_tstar_ratio_under_seeding", value=x["ratio"], item_id=f"D={x['dose']:g}",
               control_id=MC.TREATMENT,
               flags={"t_star_unseeded": x["t_star_unseeded"],
                      "t_star_seeded": x["t_star_seeded"],
                      "why": "seeding scales drift by 1/sqrt(v_s) and jitter variance by 1/v_s"})
    gates.record("mn16_crossover_is_invariant_under_seeding", invariant,
                 ratios=[x["ratio"] for x in inv],
                 note=("CORRECTS the charter: item 22 says seeding pushes the crossover out. It "
                       "does not — a uniform seed scales drift and jitter equally. The protection "
                       "is a transient, and the horizon law below is what should be registered"))

    # ---- 2. the horizon law, against simulation, with a mutant and a resolution gate ----
    rows, worst = [], 0.0
    for b2 in BETA2_GRID:
        for D in DOSE_RUNGS:
            pred = horizon_law(D, b2)
            sim = simulate_horizon(D, b2)
            mut = horizon_mutant(D, b2)
            err = abs(sim / pred - 1.0)
            worst = max(worst, err)
            gate = SF.resolution_gate(true_value=pred, mutant_value=mut,
                                      uncertainty=max(1.0, pred * TOL),
                                      min_separation=MIN_SEPARATION)
            rows.append({"beta2": b2, "dose": D, "predicted": pred, "simulated": sim,
                         "mutant": mut, "rel_error": err,
                         "resolution": gate["can_resolve"],
                         "separation_sigma": gate["separation_sigma"]})
            at.add("vseed_protection_horizon", value=float(sim), item_id=f"b2={b2}/D={D}",
                   control_id=MC.TREATMENT,
                   flags={"predicted": pred, "rel_error": err, "beta2": b2, "dose": D})
            at.add("vseed_protection_horizon", value=mut, item_id=f"b2={b2}/D={D}",
                   control_id=MC.SHUFFLE,
                   flags={"what": "MUTANT: horizon linear in dose", "beta2": b2, "dose": D})
    law_holds = worst <= TOL
    # The mutant is ANCHORED at D = MUTANT_ANCHOR, so at that rung the two laws are identical by
    # construction and separation is 0. That is not a failure of the check — it is the finding: a
    # single dose rung CANNOT distinguish a logarithmic horizon from a linear one, because any
    # one-point calibration fits both. Separability is therefore judged on the rungs away from the
    # anchor, and the anchor's non-separability is recorded as a design requirement for the GPU
    # half: it needs a dose LADDER, not a dose.
    off_anchor = [r for r in rows if r["dose"] != MUTANT_ANCHOR]
    anchor = [r for r in rows if r["dose"] == MUTANT_ANCHOR]
    resolvable = all(r["resolution"] for r in off_anchor)
    gates.require("mn16_horizon_law_matches_simulation", law_holds,
                  worst_relative_error=worst, tol=TOL, n_cells=len(rows))
    gates.record("mn16_law_separable_from_its_mutant_off_anchor", resolvable,
                 min_separation_off_anchor=min(r["separation_sigma"] for r in off_anchor),
                 anchor_separation=max(r["separation_sigma"] for r in anchor),
                 anchor_dose=MUTANT_ANCHOR,
                 note=("the mutant is calibrated to the law at D=%g, so the two are identical "
                       "there. A single dose rung cannot distinguish a logarithmic horizon from a "
                       "linear one — the GPU half needs a LADDER" % MUTANT_ANCHOR))

    # ---- 3. the pre-registered horizons, per dose rung ----
    prereg = {f"beta2={b2}": {f"D={D}": horizon_law(D, b2) for D in DOSE_RUNGS}
              for b2 in BETA2_GRID}
    log_growth = {f"beta2={b2}": [horizon_law(DOSE_RUNGS[i + 1], b2) / horizon_law(DOSE_RUNGS[i], b2)
                                  for i in range(len(DOSE_RUNGS) - 1)] for b2 in BETA2_GRID}
    at.add("vseed_horizon_growth_per_decade", value=float(np.mean(log_growth["beta2=0.999"][1:])),
           control_id=MC.TREATMENT,
           flags={"what": "mean horizon multiplier per 10x dose, beyond D=10",
                  "by_beta2": log_growth})

    verdict = MP.check_verdict(ARM, "BLOCKED-WITH-UNLOCK")
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "half_completed": "offline — the computable dose and its pre-registered horizons",
        "half_remaining": {
            "what": "the GPU comparison against EWC-lite AND replay at matched cost",
            "why_blocked": ("three training methods and a protection benchmark that do not exist "
                            "in this repo; the charter's second KILL clause ('realized horizons "
                            "miss predicted ones') needs those runs"),
            "what_it_should_now_measure": ("a DECAYING suppression window of length "
                                           "ln(D-1)/(-ln beta2), not a moved crossover")},
        "charter_correction": {
            "charter_says": "seeding pushes the drift-vs-jitter crossover out",
            "measured": "T* is INVARIANT under a uniform seed (ratio 1.0 at every dose rung)",
            "because": "seeding scales drift by 1/sqrt(v_s) and jitter variance by 1/v_s",
            "t_star_ratios": [x["ratio"] for x in inv]},
        "horizon_law": "T_p = ln(D - 1) / (-ln beta2)",
        "law_vs_simulation": {"worst_relative_error": worst, "tol": TOL, "holds": law_holds},
        "design_requirement_for_the_gpu_half": (
            "a single dose rung cannot distinguish the logarithmic law from a linear one — any "
            "one-point calibration fits both. The runs need a dose LADDER spanning at least a "
            "decade; %g-vs-%g separates them by %.0fx in predicted horizon"
            % (MUTANT_ANCHOR, 100.0, horizon_mutant(100.0, 0.999) / horizon_law(100.0, 0.999))),
        "preregistered_horizons": prereg,
        "horizon_growth_per_10x_dose": log_growth,
        "headline": ("the protection horizon grows only LOGARITHMICALLY in the dose; at "
                     "beta2=0.999 a 10x larger seed buys 2.09x then 1.50x then 1.33x more "
                     "horizon, so protection cannot be bought by seeding harder — but D=100 "
                     "already covers ~4600 steps, which is a realistic fine-tune"),
        "cells": rows, "atlas": at.to_parquet()})
    print("MN16_OK " + json.dumps({"verdict": verdict, "law_holds": law_holds,
                                   "worst_rel_error": round(worst, 5),
                                   "crossover_invariant": invariant}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
