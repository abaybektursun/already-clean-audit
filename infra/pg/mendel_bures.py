"""MENDEL MN-10 — the Bures repair (queue item 20A).

WHAT BROKE. GESTALT's shape-diffing payoff went VOID at L9 on an INSTRUMENT failure, not on
science: the frozen cost

    cost(i, j) = ||mu_i - mu_j||^2 + Bures^2(Sigma_i, Sigma_j)

is heavy-tailed on massive-activation streams. The banked symptom is stark — split-half cost width
3.2e6 against a median of 478, and 16/16 cells band-inverted. Both thresholds are QUANTILES of that
distribution (tau_move = p97.5 of the split-half self-match costs, tau_chance = p2.5 of the
shuffled costs), so a tail that spans four orders of magnitude drags tau_move above tau_chance and
the "moved" band swallows the "died" band. Every rule-formal verdict downstream is then vacuous.

THE DIAGNOSIS THAT PICKS THE REPAIR. The cost is a sum of SQUARED quantities on a stream whose
components differ enormously in scale, so a handful of massive-activation components dominate. Three
repairs are possible in principle and they are NOT equivalent:

  A. `whitened`  — divide each component's mean-offset and covariance by its own scale before
     comparing, so the cost measures SHAPE difference rather than size difference. Changes the
     cost's ordering, so it changes the Hungarian assignment: a genuinely different instrument.

  B. `log1p`     — the identical cost, squashed monotonically. It preserves the order of every
     INDIVIDUAL cost entry, so it makes a heavy-tailed band readable. It does NOT, however,
     preserve the Hungarian assignment: Hungarian minimises a SUM, and a monotone map on entries
     does not preserve the order of sums. Counterexample, verified in the dry-run:
     C = [[1, 5], [5, 10]] assigns anti-diagonally (10 < 11) and log1p(C) assigns diagonally
     (3.091 < 3.584). So log1p is a genuine change of instrument too — a smaller one than
     whitening, but not free, and this module claimed otherwise until the banked atlases said
     so.

  C. `relative`  — keep the raw cost and replace the p97.5 threshold with a MULTIPLE OF THE
     MEDIAN, i.e. stop using a tail quantile as a threshold at all.

THE GATE each must pass, stated so a variant cannot pass by being uninformative:

  1. ORDERING — tau_move < tau_chance. This is the failure that VOIDed L9.
  2. TAIL — p97.5 / p50 of the split-half distribution <= WIDTH_RATIO_BAR. A band whose upper
     quantile is thousands of times its median is not a band.
  3. SEPARATION — tau_chance / tau_move >= SEPARATION_BAR, so the "moved" band has somewhere to
     live. A variant that collapses every cost to a constant would pass 1 and 2 and is useless;
     this is what stops it.

KILL (MN-10): no variant passes all three -> the shape-diffing instrument class is dead at these
sites, and MN-13 and MN-18 lose their instrument.

Pure numpy. The reused pieces (`bures2_in_span`, `cost_matrix`, `hungarian`, `shuffle_chance_costs`)
come from `gestalt_corr` verbatim so the repaired instrument is the same object with a different
cost, not a reimplementation.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import math

from infra.pg import gestalt_corr as GC

WIDTH_RATIO_BAR = 10.0        # p97.5 / p50 of the split-half band
SEPARATION_BAR = 1.5          # tau_chance / tau_move
VARIANTS = ("original", "log1p", "whitened", "relative")
RELATIVE_TAU_MULT = 3.0       # tau_move = RELATIVE_TAU_MULT * median, for the `relative` variant


def component_scale(S_i, mu_i) -> float:
    """One component's overall scale: sqrt(tr Sigma + ||mu||^2).

    Both terms because the frozen cost sums a mean-offset term and a covariance term, so a
    whitening that normalised only one of them would leave the other free to dominate — which is
    the same failure in half the expression.
    """
    import numpy as np
    s = float((np.asarray(S_i, dtype=np.float64) ** 2).sum())
    m = float((np.asarray(mu_i, dtype=np.float64) ** 2).sum())
    tot = s + m
    assert tot >= 0, "negative component scale"
    return math.sqrt(tot) if tot > 0 else 1.0


def whiten(atlas_np: dict) -> dict:
    """Per-component scale normalisation: every component becomes unit scale.

    This is variant A and it is a real change of instrument — after whitening the cost cannot see
    that one component is a thousand times larger than another, which is exactly the property that
    was dominating it, and also a property one might have wanted. Stated rather than hidden.
    """
    import numpy as np
    mu = np.asarray(atlas_np["mu"], dtype=np.float64).copy()
    S = np.asarray(atlas_np["S"], dtype=np.float64).copy()
    for i in range(mu.shape[0]):
        c = component_scale(S[i], mu[i])
        mu[i] /= c
        S[i] /= c
    return {"mu": mu, "S": S, "weight": atlas_np["weight"]}


def cost_matrix_variant(na: dict, nb: dict, variant: str):
    """The (K_A, K_B) cost under one repair variant.

    `original` and `relative` share a matrix — `relative` differs only in how the threshold is
    taken from it, which is the point of separating the two.
    """
    import numpy as np
    assert variant in VARIANTS, f"unknown variant {variant!r} (expected one of {VARIANTS})"
    if variant == "whitened":
        return GC.cost_matrix(whiten(na), whiten(nb))
    C = GC.cost_matrix(na, nb)
    return np.log1p(C) if variant == "log1p" else C


def band_statistics(split_half_costs, shuffled_costs, variant: str) -> dict:
    """tau_move, tau_chance and the three gate readings for one variant.

    `split_half_costs` are the self-match costs of an atlas half A against half B of the SAME model
    and site — the identity band, i.e. what "no change" looks like for this instrument.
    """
    import numpy as np
    sh = np.asarray(split_half_costs, dtype=np.float64).ravel()
    ch = np.asarray(shuffled_costs, dtype=np.float64).ravel()
    assert sh.size >= 4 and ch.size >= 50, f"degenerate pools: {sh.size} split-half, {ch.size} chance"
    med = float(np.median(sh))
    med_chance = float(np.median(ch))
    p975 = float(np.percentile(sh, GC.TAU_MOVE_PCTL))
    tau_move = RELATIVE_TAU_MULT * med if variant == "relative" else p975
    tau_chance = float(np.percentile(ch, GC.TAU_CHANCE_PCTL))
    width_ratio = p975 / med if med > 0 else float("inf")
    separation = tau_chance / tau_move if tau_move > 0 else float("inf")
    ordering_ok = tau_move < tau_chance
    tail_ok = width_ratio <= WIDTH_RATIO_BAR
    sep_ok = separation >= SEPARATION_BAR
    # The MEDIAN separation, reported beside the quantile gate. The gate is deliberately strict
    # (non-overlapping tails); the medians answer the different and prior question of whether an
    # atlas half matches its own other half BETTER THAN CHANCE AT ALL. A variant can fail the gate
    # while still carrying signal, and declaring an instrument class dead without checking that
    # would be over-reading a strict threshold.
    median_separation = med_chance / med if med > 0 else float("inf")
    return {"variant": variant, "median": med, "p97.5": p975,
            "median_chance": med_chance, "median_separation": median_separation,
            "median_separates": bool(median_separation > 1.0),
            "tau_move": tau_move, "tau_chance": tau_chance,
            "width_ratio": width_ratio, "separation": separation,
            "ordering_ok": bool(ordering_ok), "tail_ok": bool(tail_ok),
            "separation_ok": bool(sep_ok),
            "passes": bool(ordering_ok and tail_ok and sep_ok),
            "n_split_half": int(sh.size), "n_chance": int(ch.size),
            "bars": {"width_ratio": WIDTH_RATIO_BAR, "separation": SEPARATION_BAR}}


def assignment_is_unchanged(C_a, C_b) -> bool:
    """Do two cost matrices induce the SAME Hungarian assignment?

    Reported, never assumed. A monotone transform of the entries does NOT guarantee this, because
    Hungarian minimises a sum (see the module docstring's counterexample), so whether a variant
    changed any decision is an empirical question about the data it ran on.
    """
    import numpy as np
    _ra, ca, _ = GC.hungarian(np.asarray(C_a, dtype=np.float64))
    _rb, cb, _ = GC.hungarian(np.asarray(C_b, dtype=np.float64))
    return bool(np.array_equal(ca, cb))


def heavy_tail_stream(*, K: int, d: int, r: int, seed: int, tail_sigma: float = 3.0) -> dict:
    """A synthetic atlas with MASSIVE-ACTIVATION scale heterogeneity — the §9 dry-run substrate.

    Component scales are lognormal with a fat sigma, so a few components are orders of magnitude
    larger than the rest, reproducing the regime the banked cost failed in. `tail_sigma = 0`
    gives a homogeneous stream, so the same generator produces both the broken and the healthy
    case and a test can show the gate discriminating.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(0.0, tail_sigma, K))
    mu = rng.standard_normal((K, d)) * scale[:, None]
    S = rng.standard_normal((K, d, r)) * scale[:, None, None]
    return {"mu": mu, "S": S, "weight": np.full(K, 1.0 / K)}


def perturb(atlas: dict, *, seed: int, rel: float) -> dict:
    """A scale-proportional perturbation — the stand-in for 'the same atlas, measured twice'."""
    import numpy as np
    rng = np.random.default_rng(seed)
    mu = np.asarray(atlas["mu"], dtype=np.float64)
    S = np.asarray(atlas["S"], dtype=np.float64)
    return {"mu": mu + rel * rng.standard_normal(mu.shape) * np.abs(mu).mean(axis=1, keepdims=True),
            "S": S + rel * rng.standard_normal(S.shape) * np.abs(S).mean(axis=(1, 2))[:, None, None],
            "weight": atlas["weight"]}
