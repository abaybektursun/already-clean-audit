"""M-4 — the erosion-curve instrument (queue item 4; queue item A1 rides inside).

Two things happen in this arm, in this order, and the order is the design:

1. THE MARK-VS-FUNCTION CONFOUND ARM, first and gating. W1-L measured that stamps placed in the
   sink erode far less than stamps placed in the highway. There are two readings. Either the
   SUBSPACE protects the stamp (the interesting one), or sink stamps were simply SMALLER — a mark
   rather than a function — and small perturbations survive everywhere. The confound arm places
   stamps at matched, EDIT-SCALE amplitudes in every subspace and asks whether the difference
   survives. Its REFUTER: 'large stamps still die everywhere'.

2. THE EROSION CURVE, only if the confound arm leaves something to explain: decay rate vs subspace
   write-traffic across >=6 subspaces x >=3 durations x 2 recipes, then one fit.

A1 (motion-matched controls) rides inside as `matched_motion_partner`: REFEREE's successor asks
whether the effect is about the subspace or about how much the weights MOVED, so every treatment
is paired with a control at matched REALIZED motion, not matched dose.

Everything here is pure numpy on weight deltas. The training that produces those deltas is the
job's business; this module is the estimator, so it dry-runs weight-free.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations


# §4: ">=6 subspaces x >=3 durations x 2 recipes". Encoded so a shrunken sweep fails loud.
MIN_SUBSPACES = 6
MIN_DURATIONS = 3
MIN_RECIPES = 2
# A1's matching tolerance, named so the job that decides WHETHER a match exists and the instrument
# that asserts it reads the same number rather than two copies of 0.05.
MOTION_MATCH_TOL = 0.05


def check_sweep(subspaces, durations, recipes) -> dict:
    """§4's sweep shape, enforced rather than trusted."""
    s, d, r = list(subspaces), list(durations), list(recipes)
    assert len(set(s)) >= MIN_SUBSPACES, f"{len(set(s))} subspaces, §4 requires >= {MIN_SUBSPACES}"
    assert len(set(d)) >= MIN_DURATIONS, f"{len(set(d))} durations, §4 requires >= {MIN_DURATIONS}"
    assert len(set(r)) >= MIN_RECIPES, f"{len(set(r))} recipes, §4 requires >= {MIN_RECIPES}"
    return {"n_subspaces": len(set(s)), "n_durations": len(set(d)), "n_recipes": len(set(r)),
            "n_cells": len(set(s)) * len(set(d)) * len(set(r))}


def stamp(U, amplitude: float, seed: int, *, out_dim: int):
    """A rank-1 stamp of Frobenius norm `amplitude` living entirely inside span(U).

    Returns (Delta, direction) where Delta is (out_dim x n) and `direction` is Delta/‖Delta‖_F, kept
    so retention can be measured as a projection onto the SAME object that was written.

    Placing the stamp in the row space of U (i.e. Delta = a uᵀ with u in span(U)) makes "inside the
    subspace" a statement about the matrix's INPUT coordinates, which is the axis every bank in
    this program lives on.
    """
    import numpy as np
    Uc = np.asarray(U, dtype=np.float64)
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(Uc.shape[1])
    u = Uc @ c
    u /= np.linalg.norm(u)
    a = rng.standard_normal(int(out_dim))
    a /= np.linalg.norm(a)
    D = amplitude * np.outer(a, u)
    f = float(np.linalg.norm(D))
    assert abs(f - amplitude) < 1e-9 * max(1.0, amplitude), f"stamp norm {f} != {amplitude}"
    leak = float(np.linalg.norm(D - (D @ Uc) @ Uc.T))
    assert leak < 1e-9 * f, f"stamp leaked outside span(U): {leak:.3e}"
    return D, D / f


def retained_fraction(delta_train, direction) -> float:
    """How much of the stamp survives training: <Delta_train, stamp_direction>_F.

    The stamp direction is unit-norm, so this is the signed component of the realized training
    delta along the stamp, in the stamp's own units. Divided by the stamp's amplitude by the caller
    it becomes a fraction in [0, 1] for pure erosion — but it is NOT clipped here: a value above 1
    (training reinforced the stamp) or below 0 (training reversed it) is a real observation and
    clipping it would hide the two most interesting outcomes.
    """
    import numpy as np
    A = np.asarray(delta_train, dtype=np.float64)
    D = np.asarray(direction, dtype=np.float64)
    assert A.shape == D.shape, f"shape mismatch {A.shape} vs {D.shape}"
    assert abs(float(np.linalg.norm(D)) - 1.0) < 1e-8, "direction is not unit-norm"
    return float((A * D).sum())


def write_traffic(delta_train, U) -> float:
    """The fraction of training's write energy that lands inside span(U) — the x-axis of the
    erosion curve. E[traffic] = k/n for a matrix that writes isotropically."""
    from infra.pg import humboldt_core as H
    return H.bank_energy_fraction(delta_train, U)


def decay_constant(durations, retained) -> dict:
    """Fit retained(t) = exp(-lambda t) by least squares on log-retention.

    Non-positive retentions cannot be logged; rather than dropping them (which would silently
    bias the fit toward the surviving stamps) the fit is refused and the caller records the cell
    as fully eroded, which is what a non-positive retention means.
    """
    import numpy as np
    t = np.asarray(durations, dtype=np.float64)
    y = np.asarray(retained, dtype=np.float64)
    assert t.shape == y.shape and t.size >= MIN_DURATIONS, \
        f"need >= {MIN_DURATIONS} paired durations, got {t.shape}"
    assert (t > 0).all(), "durations must be positive"
    assert (y > 0).all(), (
        "non-positive retention in the ladder — the stamp is fully eroded (or reversed) at some "
        "duration, and an exponential fit would silently drop that point. Record the cell as "
        "fully_eroded instead of fitting it")
    A = np.stack([t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    lam = float(-coef[0])
    pred = A @ coef
    ss_res = float(((np.log(y) - pred) ** 2).sum())
    ss_tot = float(((np.log(y) - np.log(y).mean()) ** 2).sum())
    return {"lambda": lam, "intercept": float(coef[1]),
            "r2_log": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "half_life": float(np.log(2.0) / lam) if lam > 0 else float("inf"),
            "n_durations": int(t.size)}


def fit_decay_vs_traffic(traffic, lambdas, *, n_boot: int = 2000, seed: int = 1337) -> dict:
    """THE deliverable curve: does one relation fit decay rate against write traffic?

    Fits log(lambda) = a + b*log(traffic) — a power law, because both axes are positive rates and
    the house's damage law is already a power law (REFEREE's motion^1.42), so a power form is the
    comparable one. Returns the slope with a bootstrap CI, the fit quality, AND the Spearman
    monotonicity, because the REFUTER is written against monotonicity ('no monotone
    traffic-erosion relation'), not against a particular functional form.
    """
    import numpy as np
    x = np.asarray(traffic, dtype=np.float64)
    y = np.asarray(lambdas, dtype=np.float64)
    assert x.shape == y.shape and x.size >= MIN_SUBSPACES, \
        f"need >= {MIN_SUBSPACES} subspace points, got {x.shape}"
    assert (x > 0).all() and (y > 0).all(), "power-law fit needs positive traffic and rates"
    lx, ly = np.log(x), np.log(y)

    def _slope(i):
        A = np.stack([lx[i], np.ones(i.size)], axis=1)
        c, *_ = np.linalg.lstsq(A, ly[i], rcond=None)
        return float(c[0]), float(c[1])

    b, a = _slope(np.arange(x.size))
    rng = np.random.default_rng(seed)
    boot = np.array([_slope(rng.integers(0, x.size, x.size))[0] for _ in range(n_boot)])
    pred = a + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {"slope": b, "intercept": a,
            "slope_ci_lo": float(np.percentile(boot, 2.5)),
            "slope_ci_hi": float(np.percentile(boot, 97.5)),
            "r2_log": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "spearman": _spearman(x, y), "n_points": int(x.size), "n_boot": int(n_boot)}


def _spearman(x, y) -> float:
    import numpy as np
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    assert d > 0, "degenerate ranks"
    return float((ra * rb).sum() / d)


def amplitude_ladder(edit_scale: float, n: int = 4) -> list:
    """The confound arm's dose grid: >=4 log-spaced amplitudes bracketing the EDIT scale.

    'Edit scale' is the amplitude a real weight edit uses, recomputed at preflight from the W1-L
    artifact — never transcribed. The ladder spans a decade below to a decade above it, so a
    'stamps die everywhere at edit scale' result is visible as a ladder position rather than as a
    single point.
    """
    import numpy as np
    assert edit_scale > 0, "edit scale must be positive"
    assert n >= 4, f"§3 requires >= 4 doses, got {n}"
    return list(np.logspace(np.log10(edit_scale / 10.0), np.log10(edit_scale * 10.0), n))


def matched_motion_partner(candidates, target_motion: float, *,
                           tol: float = MOTION_MATCH_TOL) -> dict:
    """A1, riding inside: pick the control cell whose REALIZED motion matches the treatment's.

    REFEREE's successor question is whether an effect belongs to the subspace or to how far the
    weights actually moved. Matching on dose does not answer it — the same dose produces different
    motion in different subspaces, which is the whole point of the gradient sink. So the control is
    chosen on realized ‖ΔW‖_F, within `tol` relative.

    `candidates` maps control-cell id -> realized motion. Fails loud when nothing matches: an
    unmatched comparison must be recorded as unmatched, not quietly run against the nearest cell.
    """
    assert target_motion > 0, "target motion must be positive"
    scored = {k: abs(v - target_motion) / target_motion for k, v in candidates.items()}
    assert scored, "no candidate control cells"
    best = min(scored, key=scored.get)
    ok = scored[best] <= tol
    assert ok, (
        f"no motion-matched control within {tol:.0%} of {target_motion:.4g}; closest is {best!r} at "
        f"{scored[best]:.1%}. A1 requires MATCHED REALIZED MOTION — widen the control ladder or "
        f"record the cell as motion-unmatched, never substitute the nearest one")
    return {"partner": best, "target_motion": float(target_motion),
            "partner_motion": float(candidates[best]), "rel_error": float(scored[best]),
            "tol": float(tol)}
