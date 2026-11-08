"""GAUGE-DIST (P6) — the edit-distance profile: the locality cost of changing function.

Question (§9). "Holistic / un-editable" (subtraction fails, replacement works; no cheap local
repair) has never been measured as a *profile*: how does the minimal function-changing weight edit
scale with the edit's allowed support (locality)? For a support class S and edit type T, the
threshold ε*(S,T) is the MINIMAL relative-Frobenius-norm edit confined to S that pushes the function
readout past δ_fun. The QEC frame predicts a *locality premium* (a localized edit either does nothing
or leaves the code space, so it costs more than a distributed one); a generic-smooth-landscape frame
predicts none beyond parameter-count scaling.

This module holds ONLY the pure, weight-free search machinery, so the WHOLE search is testable on a
TOY QUADRATIC model with a known closed-form ε* (see infra/pg/tests/test_gauge_dist.py). Everything
here operates on caller-supplied callables and boolean support masks:

  * `function_readout(edit) -> ΔL`   — the caller's function-change readout (mean next-token NLL shift
                                        on the frozen stream for gemma-2b; VQA soft-accuracy for PG1).
  * `grad_readout(edit)     -> ∇ΔL`  — the caller's ΔL-gradient (autograd on the real model; H·x on
                                        the toy). Only the ADVERSARIAL edit type needs it.
  * a boolean `mask` over the flat in-scope parameter vector — the edit's allowed support S.
  * `base_norm` = ‖W_base restricted to S‖_F — the relative→absolute scale (so ε is dimensionless per
                                                support and comparable across supports).

An "edit" is a flat fp64 vector over the in-scope parameters; the mask zeroes off-support coordinates.
The bisection walks a FROZEN log-spaced norm grid; the premium statistic π(S)=ε*_adv(S)/ε*_adv(global)
is compared against the count-matched RANDOM-edit ratio band (the parameter-counting reference) plus an
analytic √(n_S/n_global) overlay; the search-artifact gate is a budget-doubling control on the smallest
support. House style: pure, fp64, heavy imports (numpy) INSIDE functions, NO try/except (a wrong
assumption RAISES), JSON-able returns (infra/pg/gauge_cs.py, infra/pg/relaw.py).
"""
from __future__ import annotations

# The four support classes (§9.2), frozen. 'global' is the premium denominator, never counted as a
# localized detection; the other three are the LOCALIZED supports (increasing locality left→right is
# irrelevant — what matters is localized-vs-global).
LOCALIZED_SUPPORTS = ("single_matrix", "single_layer", "band3")
SUPPORT_CLASSES = LOCALIZED_SUPPORTS + ("global",)
EDIT_TYPES = ("random", "adversarial")


# ---------------------------------------------------------------------------
# masks + the frozen norm grid.
# ---------------------------------------------------------------------------
def mask_from_ranges(total, ranges):
    """Boolean support mask of length `total` with True on the given (start, stop) coordinate ranges
    (the flat in-scope parameter layout the job builds from the decoder matrices). RAISES on an
    out-of-bounds or empty range (never silently clips a support). fp-free (bool)."""
    import numpy as np
    m = np.zeros(int(total), dtype=bool)
    for lo, hi in ranges:
        assert 0 <= lo < hi <= total, f"range ({lo},{hi}) out of bounds for total {total}"
        m[lo:hi] = True
    assert m.any(), "empty support mask"
    return m


def frozen_norm_grid(lo=1e-4, hi=1e0, n=64):
    """The frozen log-spaced relative-Frobenius-norm grid the bisection walks (§9.2 'frozen norm
    grid', §9.3 'grid spans 4 decades'). Ascending, fp64. Bisection over this grid costs
    ≤ ⌈log2(n)⌉ interior evaluations (n=64 ⇒ 6, the §9.2 '≤6 bisection forwards')."""
    import numpy as np
    assert 0 < lo < hi, "need 0 < lo < hi"
    return np.logspace(np.log10(lo), np.log10(hi), int(n))


# ---------------------------------------------------------------------------
# edit directions (the two edit types T).
# ---------------------------------------------------------------------------
def random_support_direction(mask, seed):
    """A unit-ℓ2 isotropic random direction confined to `mask` (zeros off-support) — the 'random
    direction' edit type. fp64. The edit at relative norm ε is ε·base_norm·direction."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    n_on = int(m.sum())
    assert n_on > 0, "empty support"
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_on)
    d = np.zeros(m.size, dtype=np.float64)
    d[m] = z / np.linalg.norm(z)
    return d


def adversarial_support_direction(grad_readout, mask, seed, n_steps, probe_norm, base_norm):
    """The 'adversarial' edit type: the unit direction ON `mask` that (locally) MAXIMIZES the function
    change, by FROZEN-BUDGET gradient ascent. Power-iteration on the support-restricted curvature:
    start from a random support direction, then repeat `n_steps` times

        d ← normalize( mask ⊙ grad_readout(edit) ),   edit = probe_norm·base_norm·d,

    i.e. the ΔL-gradient is evaluated at a FIXED small probe radius (the 'frozen budget' — a consistent
    scale so the ascent is not confounded by where it samples curvature) and re-confined to the support
    each step. For a quadratic ΔL(x)=½ xᵀH x this is power-iteration on the support-restricted P_S H P_S
    and converges to its top eigenvector — the max-curvature localized edit (the cheapest adversarial
    ε*). `n_steps` IS the search budget the §9.2 budget-doubling control doubles. Returns the final
    unit direction (fp64). A degenerate flat step (zero gradient) stops the ascent early and keeps the
    current direction (disclosed by the caller), never a silent guess."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    radius = float(probe_norm) * float(base_norm)
    d = random_support_direction(mask, seed)
    for _ in range(int(n_steps)):
        g = np.asarray(grad_readout(radius * d), dtype=np.float64)
        assert g.shape == d.shape, f"grad shape {g.shape} != edit shape {d.shape}"
        g = g * m                                   # re-confine to the support
        nrm = np.linalg.norm(g)
        if nrm == 0.0:
            break                                   # flat direction; keep current (degenerate)
        d = g / nrm
    return d


# ---------------------------------------------------------------------------
# the bisection edit-search harness (ε*).
# ---------------------------------------------------------------------------
def readout_at_norm(function_readout, direction, base_norm):
    """The one-argument closure ε ↦ ΔL(edit) for a FIXED unit direction, where the edit is scaled to
    RELATIVE Frobenius norm ε: edit = ε·base_norm·direction. This is exactly what the bisection walks.
    `base_norm` = ‖W_base restricted to the support‖_F converts the dimensionless ε to the absolute edit
    the readout consumes.

    Type-agnostic in `direction`: the scaling `ε·base_norm·direction` is a plain scalar multiply, so
    `direction` may be a numpy vector (the toy test) OR a torch tensor on GPU (the real job's
    model-sized global edit) — whatever `function_readout` consumes. No np.asarray here, so a giant
    global edit is never forced through fp64 numpy."""
    bn = float(base_norm)

    def _f(eps):
        return float(function_readout(float(eps) * bn * direction))
    return _f


def bisect_threshold(f, grid, delta):
    """ε*(direction): the MINIMAL relative norm on the frozen ascending `grid` with ΔL(ε*) ≥ δ_fun, by
    BINARY SEARCH over grid indices. ΔL is assumed monotone non-decreasing in ε for a fixed direction
    (true for the quadratic and any fixed ray of a curvature-dominated landscape). Costs 2 bracket
    evaluations + ≤⌈log2(len grid)⌉ interior evaluations of `f`.

    Returns a JSON-able dict:
      eps_star  — the grid ε at the found index (an UPPER bound on the true threshold, resolved to the
                  grid spacing);
      idx       — its grid index;
      n_eval    — number of `f` calls (the audited forward count);
      floored   — grid[0] already ≥ δ (ε* is only bounded ABOVE; the support is easier than the grid
                  resolves);
      saturated — even grid[-1] < δ (SUPPORT-UNRESOLVED, §9.3: δ_fun never reached in-grid).
    fp64."""
    import numpy as np
    g = np.asarray(grid, dtype=np.float64)
    assert g.ndim == 1 and g.size >= 2, "grid must be a 1-D array of ≥2 points"
    assert np.all(np.diff(g) > 0), "grid must be strictly ascending"
    n = g.size
    dl = float(delta)
    n_eval = 0
    lo_val = f(float(g[0])); n_eval += 1
    if lo_val >= dl:
        return {"eps_star": float(g[0]), "idx": 0, "n_eval": n_eval,
                "floored": True, "saturated": False}
    hi_val = f(float(g[-1])); n_eval += 1
    if hi_val < dl:
        return {"eps_star": float(g[-1]), "idx": int(n - 1), "n_eval": n_eval,
                "floored": False, "saturated": True}
    # invariant: f(g[lo]) < δ ≤ f(g[hi]); shrink to the smallest hi with f(g[hi]) ≥ δ.
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(float(g[mid])) >= dl:
            hi = mid
        else:
            lo = mid
        n_eval += 1
    return {"eps_star": float(g[hi]), "idx": int(hi), "n_eval": n_eval,
            "floored": False, "saturated": False}


def eps_star_for_direction(function_readout, direction, base_norm, grid, delta):
    """The per-(support, type, seed) ATOM the job loops over: scale a fixed unit `direction` to relative
    norm and bisect for its function-changing threshold ε*. Composes readout_at_norm + bisect_threshold.
    Returns the bisect_threshold dict."""
    f = readout_at_norm(function_readout, direction, base_norm)
    return bisect_threshold(f, grid, delta)


# ---------------------------------------------------------------------------
# δ_fun calibration (pre-real, from the readout's own seed-noise band).
# ---------------------------------------------------------------------------
def calibrate_delta_fun(seed_noise_shifts, k=3.0):
    """δ_fun — the minimal ΔL counted as a genuine function change (§9.2, calibrated PRE-REAL so it is
    not tuned to the edit results). `seed_noise_shifts` = |ΔL| from RE-SEEDING the readout with NO edit
    (stream subsampling / dropout seeds): the readout's own noise floor. δ_fun = mean + k·std (k frozen
    at 3). RAISES with <2 samples (a band needs ≥2 draws)."""
    import numpy as np
    x = np.abs(np.asarray(seed_noise_shifts, dtype=np.float64))
    assert x.size >= 2, "need ≥2 seed-noise samples to set δ_fun"
    return float(x.mean() + float(k) * x.std())


# ---------------------------------------------------------------------------
# the premium statistic + the count-matched random-edit reference.
# ---------------------------------------------------------------------------
def premium_ratio(eps_star_support, eps_star_global):
    """π(S) = ε*_adv(S)/ε*_adv(global) — the locality-premium statistic (§9.2). π>1 ⇒ changing the
    function inside the localized support S costs MORE relative norm than a global edit: a locality
    premium. RAISES on a non-positive global threshold (the denominator must be resolved)."""
    assert float(eps_star_global) > 0, "global ε* must be positive (unresolved denominator)"
    return float(eps_star_support) / float(eps_star_global)


def random_ratio_band(eps_rand_support_seeds, eps_rand_global_seeds):
    """The COUNT-MATCHED null: the SAME ratio π computed for RANDOM edits, whose scaling with support
    size is the parameter-counting reference (§9.2). Given the per-seed random ε*(S) and ε*(global),
    forms the seed-MATCHED ratios (so support-size scaling — not seed noise — drives the ratio) and
    returns {mean, lo, hi, ratios, n_seed}. [lo,hi]=[min,max] over seeds is the reference band the
    premium must exceed. RAISES on shape mismatch or a zero global draw."""
    import numpy as np
    s = np.asarray(eps_rand_support_seeds, dtype=np.float64)
    g = np.asarray(eps_rand_global_seeds, dtype=np.float64)
    assert s.shape == g.shape and s.ndim == 1 and s.size >= 1, "need matched per-seed random ε* arrays"
    assert (g > 0).all(), "zero global random ε* (unresolved denominator)"
    r = s / g
    return {"mean": float(r.mean()), "lo": float(r.min()), "hi": float(r.max()),
            "ratios": r.tolist(), "n_seed": int(r.size)}


def analytic_count_overlay(n_support, n_global):
    """The parameter-counting OVERLAY √(n_S/n_global) (§9.2, REPORTED not tested): the naive random-edit
    ratio if the minimal function-changing norm scaled purely with the parameter count in support. Drawn
    on the profile figure as the reference the count-matched null is expected to track; the empirical
    band (random_ratio_band) is the actual adjudicator. RAISES on an empty global support."""
    import numpy as np
    assert int(n_global) > 0, "empty global support"
    return float(np.sqrt(float(n_support) / float(n_global)))


# ---------------------------------------------------------------------------
# the search-artifact gate (budget-doubling) + seed consistency.
# ---------------------------------------------------------------------------
def budget_doubling_ok(pi_single_budget, pi_double_budget, seed_band):
    """§9.2 H_search-artifact gate, run on the SMALLEST support only: does the premium ratio move by
    LESS than the 3-seed band when the adversarial step budget DOUBLES? If doubling the attack shifts π
    by more than the seed band, the premium was optimizer difficulty at small support (our attack), not
    landscape structure. Returns True ⇒ the premium SURVIVES the control."""
    return bool(abs(float(pi_double_budget) - float(pi_single_budget)) < float(seed_band))


def seed_consistent(pi_seeds, null_hi):
    """3-seed consistency (§9.2 'π(S) … 3-seed consistent'): the premium π(S) exceeds the random band's
    upper edge for EVERY seed, not just the mean — guards a premium carried by one lucky seed."""
    import numpy as np
    p = np.asarray(pi_seeds, dtype=np.float64)
    assert p.size >= 1, "no per-seed premiums"
    return bool((p > float(null_hi)).all())


# ---------------------------------------------------------------------------
# the frozen Detection rule (§9.2).
# ---------------------------------------------------------------------------
def detection_premium(pi_by_support, null_hi_by_support, seed_consistent_by_support, budget_ok):
    """§9.2 frozen Detection (locality premium). TRUE iff ALL hold:
      * π(S) exceeds the count-matched random-edit ratio band (π(S) > null_hi[S]) at ≥2 LOCALIZED
        supports (single_matrix/single_layer/band3 — 'global' is the denominator, never counted);
      * that excess is 3-seed consistent at each such support (seed_consistent_by_support[S] True);
      * AND the premium survives the budget-doubling control (budget_ok True).
    Pure predicate over already-computed per-support scalars. Returns {detected, n_premium_supports,
    premium_supports}."""
    prem = [s for s in LOCALIZED_SUPPORTS
            if s in pi_by_support
            and float(pi_by_support[s]) > float(null_hi_by_support[s])
            and bool(seed_consistent_by_support.get(s, False))]
    detected = bool(len(prem) >= 2 and bool(budget_ok))
    return {"detected": detected, "n_premium_supports": len(prem), "premium_supports": prem}
