"""MENDEL G2 — the SINKFORM adversarial audit kit (queue item 13).

SINKFORM's own two house rules are adopted here, because they are what makes an audit an audit:

  1. EVERY COMPUTATIONAL PROOF SHIPS A MUTANT — a deliberately wrong variant of the same formula
     that the check must REJECT. A check that only ever confirms is not a check.
  2. A NUMERICAL CHECK MUST BE SHOWN ABLE TO RESOLVE ITS TARGET before it is trusted. The mutant
     rule catches wrong formulas; the resolution gate catches blind instruments — four SINKFORM
     checks were measuring Monte Carlo noise or float64 saturation and would have passed or failed
     for reasons unrelated to the claim.

The two audited steps:

  T1  DOMINATED CONVERGENCE (Result 5). As β₂ → 1 with β₁ FIXED, E[m̂/√v̂] → μ/√(μ²+σ²).
      The adversarial reading: m̂ does NOT converge — with β₁ fixed its effective window stays
      finite, so it keeps fluctuating forever. The theorem survives only because v̂ becomes
      DETERMINISTIC, which lets E[m̂/√v̂] factor into E[m̂]/√E[g²]. So the sharp test is at LARGE
      β₁, where m̂'s fluctuation is largest and a limit that secretly needed m̂ → μ would fail.

  T2  THE β₁-CANCELLATION (Result 6). τ·Var X = σ²/s², independent of β₁, hence T* = 1/r².
      The adversarial reading: EWMA smoothing suppresses per-step variance by (1−β₁)/(1+β₁) and
      introduces autocorrelation over (1+β₁)/(1−β₁) steps. The claim is that these cancel EXACTLY
      in the variance of the SUM. That is a statement about the long-run variance (the spectral
      density at zero), and it is exact only asymptotically in T — so the REGIME OF VALIDITY is
      part of the claim and is measured here rather than assumed.

Pure numpy/sympy on small objects. Runs on the control plane (the math-studio scope note).

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# T1 — the two-timescale limit
# ---------------------------------------------------------------------------
def adam_drift_simulation(*, mu: float, sigma: float, beta1: float, beta2: float,
                          n_steps: int, n_burn: int, seed: int) -> dict:
    """E[m̂/√v̂] for ONE stationary scalar coordinate under Adam's bias-corrected moments.

    Gradients are i.i.d. Normal(mu, sigma²) — the stationary regime Result 5 assumes, stated as an
    assumption rather than smuggled in. `n_burn` steps are discarded so the estimate is of the
    STATIONARY mean and not of the bias-correction transient.
    """
    import numpy as np
    assert 0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0, f"bad betas {beta1}, {beta2}"
    assert sigma > 0 and n_steps > n_burn >= 0, "degenerate simulation request"
    rng = np.random.default_rng(seed)
    g = rng.normal(mu, sigma, n_steps)
    m = v = 0.0
    vals = np.empty(n_steps)
    for t in range(n_steps):
        m = beta1 * m + (1 - beta1) * g[t]
        v = beta2 * v + (1 - beta2) * g[t] * g[t]
        mh = m / (1 - beta1 ** (t + 1))
        vh = v / (1 - beta2 ** (t + 1))
        vals[t] = mh / math.sqrt(vh)
    tail = vals[n_burn:]
    # THE NAIVE SEM IS A LIE HERE. X_t is heavily autocorrelated — at β₂ = 0.9999 the v̂ window is
    # ~20 000 steps, so 400 000 samples carry only ~20 independent ones and std/sqrt(n) understates
    # the uncertainty by an order of magnitude. Reporting it unadjusted is precisely the blind
    # instrument the resolution gate exists to catch, so the effective sample size is measured from
    # the sample's own integrated autocorrelation time rather than assumed.
    tau_int = integrated_autocorr_time(tail)
    n_eff = tail.size / max(1.0, 2.0 * tau_int)
    return {"mean": float(tail.mean()),
            "sem_naive": float(tail.std(ddof=1) / math.sqrt(tail.size)),
            "sem": float(tail.std(ddof=1) / math.sqrt(n_eff)),
            "tau_int": float(tau_int), "n_eff": float(n_eff),
            "n": int(tail.size), "beta1": beta1, "beta2": beta2,
            "n_eff_numerator": (1 + beta1) / (1 - beta1),
            "n_eff_denominator": (1 + beta2) / (1 - beta2)}


def integrated_autocorr_time(x, *, c: float = 6.0, max_lag: int = 200000) -> float:
    """τ_int = 1/2 + Σ_{h≥1} ρ_h, with Sokal's automatic windowing (stop at h ≥ c·τ_int).

    Without windowing the tail of the sum is pure noise and τ_int wanders; Sokal's rule is the
    standard fix and is what makes the effective sample size a number rather than an opinion.
    """
    import numpy as np
    a = np.asarray(x, dtype=np.float64)
    a = a - a.mean()
    n = a.size
    assert n > 8, "integrated autocorrelation time needs a non-trivial sample"
    f = np.fft.rfft(a, 2 * n)
    acf = np.fft.irfft(f * np.conjugate(f), 2 * n)[:n].real
    if acf[0] <= 0:
        return 0.5
    acf /= acf[0]
    total = 0.5
    for h in range(1, min(n, max_lag)):
        total += acf[h]
        if h >= c * total:
            break
    return float(max(0.5, total))


def drift_limit(mu: float, sigma: float) -> float:
    """The theorem's limit: μ/√(μ²+σ²)."""
    return mu / math.sqrt(mu * mu + sigma * sigma)


def drift_limit_mutant(mu: float, sigma: float) -> float:
    """THE MUTANT for T1: μ/(μ²+σ²) — the same expression with the square root dropped.

    Chosen because it is the error a careless derivation actually makes (forgetting that v̂ tracks
    E[g²] and the update divides by its SQUARE ROOT), and because it is close enough to the true
    limit at some (μ, σ) to be a real test rather than a straw man.
    """
    return mu / (mu * mu + sigma * sigma)


# ---------------------------------------------------------------------------
# T2 — the β₁ cancellation
# ---------------------------------------------------------------------------
def tau_var_x_closed_form(sigma: float, s: float, beta1: float) -> dict:
    """τ · Var X from Result 6's two factors, kept SEPARATE so the cancellation is visible.

        Var X = σ²(1−β₁) / ((1+β₁) s²)          per-step variance, suppressed by smoothing
        τ     = (1+β₁)/(1−β₁)                    integrated autocorrelation time, inflated by it
        τ·VarX = σ²/s²                           the (1±β₁) factors cancel exactly
    """
    var_x = sigma * sigma * (1 - beta1) / ((1 + beta1) * s * s)
    tau = (1 + beta1) / (1 - beta1)
    return {"var_x": var_x, "tau": tau, "product": tau * var_x,
            "closed_form": sigma * sigma / (s * s), "beta1": beta1}


def tau_var_x_mutant(sigma: float, s: float, beta1: float) -> float:
    """THE MUTANT for T2: the same product with the smoothing suppression DROPPED from Var X.

    τ · (σ²/s²) = σ²(1+β₁)/((1−β₁)s²), which diverges as β₁ → 1. This is exactly the error of
    counting the autocorrelation inflation while forgetting the variance suppression, and it makes
    T* depend on β₁ — so if the check cannot separate the two, it cannot see the theorem at all.
    """
    return (1 + beta1) / (1 - beta1) * sigma * sigma / (s * s)


def accumulation_simulation(*, mu: float, sigma: float, beta1: float, beta2: float, T: int,
                            n_trials: int, seed: int) -> dict:
    """Var(Σ_{t=1..T} X_t) / T measured directly, where X_t = m̂_t/√v̂_t.

    This is the quantity Result 6 claims equals τ·Var X = σ²/s² independent of β₁, so measuring it
    across a β₁ sweep is the whole test. Simulated rather than derived a second time: an
    independent derivation could share the original's mistake, a simulation cannot.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    g = rng.normal(mu, sigma, (n_trials, T))
    m = np.zeros(n_trials)
    v = np.zeros(n_trials)
    total = np.zeros(n_trials)
    for t in range(T):
        m = beta1 * m + (1 - beta1) * g[:, t]
        v = beta2 * v + (1 - beta2) * g[:, t] ** 2
        total += (m / (1 - beta1 ** (t + 1))) / np.sqrt(v / (1 - beta2 ** (t + 1)))
    return {"var_over_T": float(total.var(ddof=1) / T), "mean_over_T": float(total.mean() / T),
            "T": int(T), "n_trials": int(n_trials), "beta1": beta1, "beta2": beta2}


def t_star(r: float) -> float:
    """The crossover horizon T* = 1/r², free of β₁, β₂, η."""
    assert r > 0, "T* is undefined at zero consistency"
    return 1.0 / (r * r)


def symbolic_cancellation() -> dict:
    """The cancellation, checked in a CAS rather than by hand.

    sympy is asked to simplify τ·Var X and confirm the result is free of β₁, and separately to
    confirm the mutant is NOT free of β₁ — the mutant half is what proves the check has teeth.
    """
    import sympy as sp
    b1, sig, s = sp.symbols("beta1 sigma s", positive=True)
    var_x = sig ** 2 * (1 - b1) / ((1 + b1) * s ** 2)
    tau = (1 + b1) / (1 - b1)
    prod = sp.simplify(tau * var_x)
    mutant = sp.simplify(tau * sig ** 2 / s ** 2)
    return {"product": str(prod), "expected": str(sp.simplify(sig ** 2 / s ** 2)),
            "product_free_of_beta1": bool(sp.simplify(sp.diff(prod, b1)) == 0),
            "mutant": str(mutant),
            "mutant_free_of_beta1": bool(sp.simplify(sp.diff(mutant, b1)) == 0),
            "product_equals_expected": bool(sp.simplify(prod - sig ** 2 / s ** 2) == 0)}


# ---------------------------------------------------------------------------
# Resolution gates
# ---------------------------------------------------------------------------
def resolution_gate(*, true_value: float, mutant_value: float, uncertainty: float,
                    min_separation: float = 3.0) -> dict:
    """Can this measurement TELL THE TWO APART?

    A check whose uncertainty exceeds the gap between the theorem and its mutant is measuring noise,
    and its agreement with the theorem is not evidence. Reported before the verdict, never after.
    """
    gap = abs(float(true_value) - float(mutant_value))
    sep = gap / uncertainty if uncertainty > 0 else float("inf")
    return {"true_value": float(true_value), "mutant_value": float(mutant_value), "gap": gap,
            "uncertainty": float(uncertainty), "separation_sigma": sep,
            "min_separation": float(min_separation), "can_resolve": bool(sep >= min_separation)}


# ---------------------------------------------------------------------------
# The T*-map: recovering r = mu/sigma from a FINITE-T consistency measurement
# ---------------------------------------------------------------------------
def consistency_floor(n_steps: int) -> float:
    """The value `cons` takes when the true consistency is ZERO: 1/sqrt(T).

    With mu = 0 the numerator ‖Σ_t g_t‖ is a random walk of length ~sigma·sqrt(T·d) while the
    denominator Σ_t ‖g_t‖ grows like T·sigma·sqrt(d), so their ratio floors at 1/sqrt(T) rather
    than at 0. At MENDEL's banked T = 24 that floor is 0.204 — and the sink is the LOW-r set, so
    the floor sits exactly where the measurement matters.
    """
    assert n_steps >= 1, "consistency_floor needs at least one step"
    return 1.0 / math.sqrt(n_steps)


def r_from_consistency(cons, n_steps: int):
    """Invert cons -> r = mu/sigma, correcting for the finite-T floor.

        cons² ≈ (T r² + 1) / (T (1 + r²))     =>     r² = (T cons² − 1) / (T (1 − cons²))

    The naive inversion r = cons/sqrt(1−cons²) ignores the floor and reports r ≈ 0.207 for a
    coordinate whose true r is 0 — a T*-map built on it would call every deep-sink coordinate
    T* ≈ 25 when the truth is T* -> infinity. Verified against simulation across r ∈ [0, 3] at
    T = 24 and T = 200 (tests/test_mendel_sinkform.py).

    Coordinates at or below the floor are returned as r = 0 (their true r is unresolvable, not
    zero) and flagged by `at_floor` so the caller can report how much of the map is censored.
    """
    import numpy as np
    c = np.asarray(cons, dtype=np.float64)
    assert n_steps >= 2, "the inversion needs at least two steps"
    assert ((c >= 0) & (c <= 1)).all(), "consistency outside [0, 1]"
    c2 = c * c
    num = n_steps * c2 - 1.0
    den = n_steps * (1.0 - c2)
    r2 = np.where(den > 0, np.divide(num, den, out=np.zeros_like(c), where=den > 0), np.inf)
    at_floor = num <= 0
    return np.sqrt(np.maximum(r2, 0.0)), at_floor


def t_star_map(cons, n_steps: int):
    """T*_j = 1/r_j², the horizon at which coordinate j's drift overtakes its own jitter.

    Censored coordinates (r unresolvable at this T) get T* = inf, which is the honest reading:
    the measurement says only "T* is larger than this batch can see".
    """
    import numpy as np
    r, at_floor = r_from_consistency(cons, n_steps)
    with np.errstate(divide="ignore"):
        ts = np.where(r > 0, 1.0 / (r * r), np.inf)
    return ts, r, at_floor


def membership_at_horizon(r, T: float):
    """SINKFORM's membership law: coordinate j is in the sink at horizon T iff r_j < T^(−1/2).

    Note for the address readout: the SET depends on T, but the RANKING by r does not, so an
    address built as "the k smallest r" is horizon-free. That is worth stating rather than
    discovering — it means the address test cannot be tuned by choosing T.
    """
    import numpy as np
    assert T > 0, "horizon must be positive"
    return np.asarray(r, dtype=np.float64) < T ** -0.5
