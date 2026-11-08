"""MENDEL — the critic-pass statistics (G1, G2).

A critic pass is not a re-read of a report; it is an INDEPENDENT re-derivation of the deciding
statistic from the raw rows, by a code path that does not import the original. AMUNDSEN's own
review re-derived 11/11 numbers by a second path and said so plainly — but a self-audit shares the
author's assumptions, and the assumptions are where verdicts break. So the functions here are
written from the statistic's DEFINITION, not from the original implementation.

Everything is pure numpy on small arrays and is exercised by the dry-run gate.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import itertools
import math


def average_rank(values):
    """Ranks with TIES AVERAGED — the only correct ranking for Spearman when ties exist.

    This is not pedantry. AM-3's magnitude covariate is |log s| over
    s ∈ {0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}, where |log 0.5| = |log 2| = 0.6931 is an exact tie. A
    rank built from argsort(argsort(·)) breaks that tie arbitrarily and returns ρ = 0.9643 where
    the correct tie-corrected value is 0.9370 — which is exactly the number AMUNDSEN banked. The
    first version of this critic pass got it wrong and the artifact caught the critic.
    """
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and a.size > 0, f"average_rank expects a non-empty 1-D vector, got {a.shape}"
    order = np.argsort(a, kind="stable")
    r = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y) -> float:
    """Spearman's ρ via tie-corrected average ranks."""
    rx, ry = average_rank(x), average_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    assert den > 0, "Spearman is undefined when either variable is constant"
    return float((rx @ ry) / den)


def spearman_exact_p(x, y, *, max_n: int = 8) -> dict:
    """Two-sided exact permutation p for Spearman's ρ.

    Exact rather than asymptotic because these ladders have n = 7: a t-approximation at n = 7 is
    not a p-value, it is a decoration. Refuses to enumerate above `max_n` instead of silently
    running 9! or worse — the first draft of this pass hung on a 14-row arm.
    """
    n = len(x)
    assert n == len(y), f"length mismatch {n} vs {len(y)}"
    assert n <= max_n, (
        f"exact enumeration refused at n={n} (max {max_n}); {n}! permutations is not a computation")
    obs = spearman(x, y)
    draws = [spearman(x, [y[i] for i in p]) for p in itertools.permutations(range(n))]
    hits = sum(1 for v in draws if abs(v) >= abs(obs) - 1e-12)
    return {"rho": obs, "p_exact": hits / len(draws), "n": n, "n_permutations": len(draws)}


def paired_bootstrap(values, *, n_boot: int = 20000, seed: int = 1337) -> dict:
    """Mean + percentile CI of a per-unit vector (windows, seeds, ...)."""
    import numpy as np
    v = np.asarray(values, dtype=np.float64).ravel()
    assert v.size > 0 and np.isfinite(v).all(), "paired_bootstrap on an empty or non-finite sample"
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return {"mean": float(v.mean()), "ci_lo": float(np.percentile(b, 2.5)),
            "ci_hi": float(np.percentile(b, 97.5)), "n": int(v.size), "n_boot": int(n_boot)}


def ratio_bootstrap(num, den, *, n_boot: int = 20000, seed: int = 1337) -> dict:
    """Ratio of means with a paired percentile CI — resampling the SAME unit indices on both sides,
    because the two arms were measured on the same windows and an unpaired resample would inflate
    the interval with variation the design already removed."""
    import numpy as np
    a = np.asarray(num, dtype=np.float64).ravel()
    b = np.asarray(den, dtype=np.float64).ravel()
    assert a.shape == b.shape, f"paired ratio needs matched vectors, got {a.shape} vs {b.shape}"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n_boot, a.size))
    r = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return {"ratio": float(a.mean() / b.mean()), "ci_lo": float(np.percentile(r, 2.5)),
            "ci_hi": float(np.percentile(r, 97.5)), "n": int(a.size)}


def threshold_fragility(per_unit_ratios, threshold: float, *, n_boot: int = 20000,
                        seed: int = 1337) -> dict:
    """How often would a resample of the UNITS fail to clear a frozen threshold?

    A verdict that rests on a point estimate crossing a bar by 0.0035 is not the same object as one
    that clears it comfortably, and the difference is invisible in the point estimate. This reports
    the probability that the bar is NOT cleared under a bootstrap over the units — the honest
    statement of how much of the verdict is the data and how much is the draw.
    """
    import numpy as np
    v = np.asarray(per_unit_ratios, dtype=np.float64).ravel()
    assert v.size > 0, "threshold_fragility on an empty sample"
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return {"point_estimate": float(v.mean()), "threshold": float(threshold),
            "margin": float(v.mean() - threshold),
            "p_fails_threshold": float((b <= threshold).mean()),
            "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
            "n_units": int(v.size), "n_boot": int(n_boot),
            "n_distinct_resample_means": int(len(set(np.round(b, 12).tolist())))}


def loglog_slope(x, y, *, window: int) -> float:
    """|d log y / d log x| over the LAST `window` points."""
    import numpy as np
    lx = np.log(np.asarray(x, dtype=np.float64)[-window:])
    ly = np.log(np.asarray(y, dtype=np.float64)[-window:])
    assert lx.size == ly.size >= 2, f"slope needs >=2 points, got {lx.size}"
    return abs(float(np.polyfit(lx, ly, 1)[0]))


def slope_window_sweep(x, y, windows=(3, 5, 8, 10)) -> dict:
    """The same slope at several windows.

    A plateau test compares |slope| to a bar. If the artifact does not state the WINDOW, the test
    is not reproducible, and the honest way to show that is to print the whole sweep and let the
    reader see where the bar is crossed — rather than picking the window that agrees with the
    banked verdict.
    """
    n = len(x)
    return {f"last{w}": loglog_slope(x, y, window=w) for w in windows if w <= n}


def perplexity_from_window_nll(nll, ntok):
    """Per-window perplexity exp(nll/ntok).

    The estimator matters and is worth stating: a CORPUS perplexity is exp(Σnll/Σntok), which is
    not the mean of per-window perplexities. AM-1's headline contrast is per-window paired (that is
    what its CIs are over) while its positive control is quoted on the corpus aggregate; both are
    legitimate, but they are different numbers and a reproduction must use the right one for each.
    """
    import numpy as np
    a = np.asarray(nll, dtype=np.float64)
    b = np.asarray(ntok, dtype=np.float64)
    assert a.shape == b.shape, f"nll/ntok mismatch {a.shape} vs {b.shape}"
    assert (b > 0).all(), "a window with zero tokens has no perplexity"
    return np.exp(a / b)


def corpus_perplexity(nll, ntok) -> float:
    import numpy as np
    a = np.asarray(nll, dtype=np.float64)
    b = np.asarray(ntok, dtype=np.float64)
    return float(np.exp(a.sum() / b.sum()))


def reproduction_verdict(ours: float, banked: float, tol: float) -> dict:
    """One re-derivation's outcome. `tol` is absolute and must be stated by the caller."""
    d = abs(float(ours) - float(banked))
    return {"ours": float(ours), "banked": float(banked), "abs_diff": d, "tol": float(tol),
            "reproduced": bool(d <= tol)}
