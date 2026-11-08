"""QUORUM core — the necessary-rank estimator, the injected positive control, the bimodality test,
the necessary-rank gap statistic, and `adjudicate_quorum` (spec §6 frozen verdict order).

The measured quantity is n* — the smallest removal rank at which an installed capability's task-gain
retention first falls below ρ_break. The genuinely NEW logic:
  1. `necessary_rank(retentions, ranks, rho_break, ...)` — isotonic-smoothed first-crossing with
     right-censoring to d (never a silent "n* = max ladder").
  2. `inject_rank1_delta` — the positive control: a rank-1 update along an energy eigenvector, whose
     necessary-rank is analytically 1 (the ceiling that makes H_redundant falsifiable).
  3. `bimodality_test` — a 2-split separation statistic vs a unimodal-Gaussian null, for the
     H_capability-dependent adjudication.
  4. `delta_nstar` + `paired_delta_ci` — the Δn* statistic and its paired CI.
  5. `adjudicate_quorum` — the §6 branch order, applied ONCE on confirmation.

Everything reuses verified surfaces (`cleave_core.off_bases/mechanism_census/retention/paired_boot`,
`comwrite_surgery.build_decoder`). Heavy imports inside functions so the module py_compiles on the
control plane (no torch on the laptop). No try/except hiding — a wrong assumption RAISES (CLAUDE.md).
"""
from __future__ import annotations

from infra.pg.gauge_common import N_LAYERS, PROJ  # noqa: F401

# The removal-rank ladder. Reaches d so the energy arm is never censored (A-QUORUM-4), AND starts at 1
# so the positive control (a rank-1 capability) can resolve n*=1 (A-QUORUM-5 — CLEAVE's PROTECT_LADDER
# started at 5, which would give the rank-1 injection n*=5 > 2 and spuriously fail the pos-control gate,
# VOIDing the whole primary). The low end {1,2,3} is where compact storage lives; the search-rank is 20.
D_MODEL = 2048
RANK_LADDER = (1, 2, 3, 5, 10, 20, 50, 150, 350, 679, 1024, 1536, 2048)

# The search-rank marker (CLEAVE R4: the task gradient is 26× enriched in the top-20 energy subspace).
SEARCH_RANK = 20

# capability admission — a capability enters the necessary-rank readout only if it learns (retention
# is a ratio; a near-zero denominator is noise). CLEAVE's screen, same threshold.
MIN_GAIN_FULL = 0.08

POLICIES = ("energy", "random", "susceptibility", "susceptibility_shuffle")
ENERGY = "energy"
NULL = "random"
SEED = 1337


# ===========================================================================
# 1. necessary-rank — isotonic-smoothed first-crossing with right-censoring.
# ===========================================================================
def _isotonic_nonincreasing(x, y):
    """Pool-Adjacent-Violators onto a NON-INCREASING fit (retention can only fall as more of the
    update is removed; sampling noise makes the raw curve wiggle). Weighted-mean pooling, O(n).
    Returns the fitted y at the same x. Pure numpy."""
    import numpy as np
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    # PAVA for a monotone NON-increasing sequence: negate → non-decreasing PAVA → negate back.
    v = -y
    w = np.ones(n)
    # stacks of (value, weight, count)
    vals, wts, cnts = [], [], []
    for i in range(n):
        cv, cw, cc = v[i], w[i], 1
        while vals and vals[-1] >= cv:              # violation → pool
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            cv = (pv * pw + cv * cw) / (pw + cw)
            cw = pw + cw
            cc = pc + cc
        vals.append(cv); wts.append(cw); cnts.append(cc)
    out = np.empty(n)
    j = 0
    for cv, cc in zip(vals, cnts):
        out[j:j + cc] = cv
        j += cc
    return -out


def necessary_rank(retentions, ranks, rho_break, d=D_MODEL, smooth=True):
    """n* = the smallest rank r at which the (isotonic-smoothed) retention first drops below
    `rho_break`. Linear interpolation between ladder points gives a continuous n* (the crossing rank),
    not just a ladder rung. **Right-censored to d** if the curve never crosses — recorded, never a
    silent "n* = max ladder" (spec §6 ceiling; `feedback_one_sided_thresholds`).

    Returns {n_star, censored, crossed_at_rank, retention_fit, rho_break}. Fail-loud on ragged input.
    """
    import numpy as np
    r = np.asarray(ranks, dtype=np.float64)
    y = np.asarray(retentions, dtype=np.float64)
    assert r.shape == y.shape and r.ndim == 1 and r.size >= 2, "need ≥2 aligned (rank, retention) points"
    order = np.argsort(r)
    r, y = r[order], y[order]
    fit = _isotonic_nonincreasing(r, y) if smooth else y.copy()
    below = fit < rho_break
    if not below.any():
        return {"n_star": float(d), "censored": True, "crossed_at_rank": None,
                "retention_fit": fit.tolist(), "ranks": r.tolist(), "rho_break": float(rho_break)}
    k = int(np.argmax(below))                       # first index below ρ_break
    if k == 0:
        n_star = float(r[0])                        # already below at the smallest rank
    else:
        # linear crossing between r[k-1] (≥ρ) and r[k] (<ρ)
        y0, y1 = fit[k - 1], fit[k]
        r0, r1 = r[k - 1], r[k]
        frac = (y0 - rho_break) / (y0 - y1) if (y0 - y1) > 1e-12 else 0.0
        n_star = float(r0 + frac * (r1 - r0))
    return {"n_star": n_star, "censored": False, "crossed_at_rank": float(r[k]),
            "retention_fit": fit.tolist(), "ranks": r.tolist(), "rho_break": float(rho_break)}


def rho_break_from_null(null_retentions, null_ranks, d=D_MODEL):
    """ρ_break = the retention at the KNEE of the random-basis retention curve (spec §6, the
    basis-agnostic reference). Knee = the point of maximum perpendicular distance from the chord on
    the (log rank, retention) box — the hyperparameter-free elbow, read off the NULL, never the
    treatment. Returns {rho_break, knee_rank}."""
    import numpy as np
    r = np.log(np.asarray(null_ranks, dtype=np.float64))
    y = _isotonic_nonincreasing(np.asarray(null_ranks, float), np.asarray(null_retentions, float))
    assert r.size >= 3, "need ≥3 points for a knee"
    o = np.argsort(r)
    r, y = r[o], y[o]
    xr, yr = float(r[-1] - r[0]), float(y[-1] - y[0])
    assert xr > 0, "degenerate rank axis"
    if abs(yr) < 1e-12:                             # flat null → break at its (constant) level
        return {"rho_break": float(np.median(y)), "knee_rank": float(np.exp(r[len(r) // 2]))}
    xn = (r - r[0]) / xr
    yn = (y - y[0]) / yr
    dist = np.abs(yn - xn) / np.sqrt(2.0)
    k = int(np.argmax(dist))
    return {"rho_break": float(y[k]), "knee_rank": float(np.exp(r[k]))}


# ===========================================================================
# 2. injected rank-1 positive control — analytic n* = 1.
# ===========================================================================
def inject_rank1_delta(U_energy, layer, mtype, scale, d=D_MODEL):
    """A rank-1 update ΔW = scale · u₀ u₀ᵀ on ONE (layer, mtype), where u₀ = the top energy eigenvector
    (residual-facing side). It amplifies the residual's highest-variance component, so it produces a
    measurable behavioural change; and because it lies ENTIRELY in the top-1 energy direction,
    `off_bases("energy", r≥1)` annihilates it ⇒ retention cliffs to 0 at r=1 ⇒ **n* = 1 analytically**.

    This is the ceiling that makes H_redundant falsifiable: if the instrument reports n* ≫ 1 for THIS,
    it cannot resolve compact storage and the whole necessary-rank readout VOIDs (spec §5).

    Returns the ΔW matrix (out_d, in_d) for the target — shaped per `comwrite_core.residual_side`:
    output-side (o/down) ⇒ the residual-facing side is the OUTPUT (d rows); input-side ⇒ INPUT (d cols).
    """
    import numpy as np

    from infra.pg import comwrite_core as CW
    u = np.asarray(U_energy, dtype=np.float64)[:, 0]      # top energy direction (d,)
    u = u / (np.linalg.norm(u) + 1e-30)
    if CW.residual_side(mtype) == "output":
        # residual-facing = output (rows). A square rank-1 on the output side: u (uᵀ·anything). Use
        # a d×d block u uᵀ so the whole update lives in span(u) on the output side.
        return float(scale) * np.outer(u, u)
    return float(scale) * np.outer(u, u)                  # input side is also d×d for q/k/v (d×d)


# ===========================================================================
# 3. bimodality (H_capability-dependent).
# ===========================================================================
def _best_2split_ratio(sorted_x):
    """The largest between-over-total variance ratio over all gap-splits of a sorted 1-D sample —
    a 1-D k=2 separation statistic (Fisher's). ∈ [0,1]; ~1 = two tight far-apart clusters, ~0 =
    one blob. Deterministic, O(n)."""
    import numpy as np
    x = np.asarray(sorted_x, dtype=np.float64)
    n = x.size
    tot = float(((x - x.mean()) ** 2).sum())
    if tot < 1e-30:
        return 0.0
    csum = np.cumsum(x)
    best = 0.0
    for k in range(1, n):                              # split after index k-1
        n1, n2 = k, n - k
        m1 = csum[k - 1] / n1
        m2 = (csum[-1] - csum[k - 1]) / n2
        between = n1 * (m1 - x.mean()) ** 2 + n2 * (m2 - x.mean()) ** 2
        best = max(best, between / tot)
    return float(best)


def bimodality_test(values, n_boot=2000, seed=SEED):
    """Is the sample bimodal? Statistic = the best 2-split between/total variance ratio; p-value =
    bootstrap against a UNIMODAL Gaussian null (same mean/sd/n). Small p ⇒ the two-cluster split is
    better-separated than a unimodal sample of the same spread would produce ⇒ bimodal. Clean, small-n
    honest, and the right null for "compact vs redundant are two modes" (not a uniformity test).

    Returns {statistic, p_value, n, split_point, low_mode_n, high_mode_n, modes_hint}."""
    import numpy as np
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    assert n >= 4, "bimodality test needs ≥4 values"
    stat = _best_2split_ratio(x)
    mu, sd = float(x.mean()), float(x.std(ddof=1) + 1e-30)
    rng = np.random.default_rng(int(seed))
    boots = np.array([_best_2split_ratio(np.sort(rng.normal(mu, sd, n))) for _ in range(n_boot)])
    p = float((boots >= stat).mean())
    # the split that achieved the statistic (for the compact/redundant member counts)
    best_k, best_val = 1, -1.0
    csum = np.cumsum(x)
    for k in range(1, n):
        m1 = csum[k - 1] / k
        m2 = (csum[-1] - csum[k - 1]) / (n - k)
        between = k * (m1 - x.mean()) ** 2 + (n - k) * (m2 - x.mean()) ** 2
        if between > best_val:
            best_val, best_k = between, k
    return {"statistic": stat, "p_value": p, "n": int(n),
            "split_value": float((x[best_k - 1] + x[best_k]) / 2) if 0 < best_k < n else None,
            "low_mode_n": int(best_k), "high_mode_n": int(n - best_k),
            "modes_hint": "bimodal" if p < 0.05 else "unimodal"}


# ===========================================================================
# 4. the Δn* gap statistic.
# ===========================================================================
def delta_nstar(nstar_random, nstar_energy):
    """Δn* = n*_random − n*_energy for one capability. Positive ⇒ energy-removal breaks the capability
    EARLIER than random-removal ⇒ storage is compact (aligned with the search's energy directions).
    ≈0 ⇒ energy is no more necessary than a random subspace ⇒ redundant storage."""
    return float(nstar_random) - float(nstar_energy)


def redundancy_ratio(nstar_energy, search_rank=SEARCH_RANK):
    """ρ_red = n*_energy / search-rank. ≈1 ⇒ storage follows search (compact); ≫1 ⇒ redundant."""
    assert search_rank > 0
    return float(nstar_energy) / float(search_rank)


def paired_delta_ci(deltas, seed=SEED, n_boot=10000, min_n=3):
    """Paired bootstrap median of Δn* across capabilities + 95% CI. Small-n-safe (the confirmation
    bank is a handful of capabilities, not 10+ items) — mirrors `cleave_core.paired_boot`. Returns
    E=None + insufficient when < min_n survive."""
    import numpy as np
    d = np.asarray(deltas, dtype=np.float64)
    assert d.ndim == 1, "deltas must be 1-D"
    if d.size < min_n:
        return {"median": None, "ci_lo": None, "ci_hi": None, "n": int(d.size), "insufficient": True}
    rng = np.random.default_rng(int(seed))
    boots = np.median(d[rng.integers(0, d.size, size=(n_boot, d.size))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"median": float(np.median(d)), "ci_lo": float(lo), "ci_hi": float(hi),
            "mean": float(d.mean()), "n": int(d.size), "small_n": bool(d.size < 10),
            "insufficient": False}


# ===========================================================================
# 5. adjudicate_quorum — the spec §6 frozen verdict order (applied ONCE on confirmation).
# ===========================================================================
def adjudicate_quorum(*, delta_ci, nstar_energy_median, search_rank,
                      pos_control_nstar, pos_control_fired,
                      null_band_hi, dip_p, n_compact, n_redundant,
                      any_learner, shuffle_matches_random, name="quorum"):
    """Frozen §6 branch order. Every guard two-sided (`feedback_one_sided_thresholds`); no
    post-unblind rule swap.

      delta_ci            : paired Δn* dict (median + CI). H_compact needs ci_lo > null_band_hi.
      nstar_energy_median : median n*_energy over confirmation capabilities.
      search_rank         : the gradient search-rank (~20, CLEAVE R4).
      pos_control_nstar   : the injected rank-1 capability's measured n* (MUST be ≈1).
      pos_control_fired   : n*_pos ≤ 2 (the instrument can resolve compact storage).
      null_band_hi        : the upper edge of the Δn*-under-random-vs-random null band.
      dip_p               : Hartigan dip p-value on the per-capability Δn* (H_capability-dependent).
      n_compact/n_redundant : counts of capabilities on each side of the mode split at FDR.
      any_learner         : ≥1 confirmation capability cleared MIN_GAIN_FULL.
      shuffle_matches_random : the shuffle control collapsed to n*_random (structure is load-bearing).
    Returns {verdict, reason, ...inputs}.
    """
    r = {"name": name, "delta_ci": delta_ci, "nstar_energy_median": nstar_energy_median,
         "search_rank": search_rank, "pos_control_nstar": pos_control_nstar,
         "pos_control_fired": bool(pos_control_fired), "null_band_hi": null_band_hi,
         "dip_p": dip_p, "n_compact": n_compact, "n_redundant": n_redundant,
         "any_learner": bool(any_learner), "shuffle_matches_random": bool(shuffle_matches_random)}

    # 1. VOID — the instrument cannot see compactness, or nothing learned. A "redundant" verdict from a
    #    blind instrument is the trap the positive control exists to spring (spec §5/§6).
    if (not pos_control_fired) or (not any_learner):
        r["verdict"] = "VOID"
        r["reason"] = ("the positive control did not fire (the injected rank-1 capability's n\\* is not "
                       f"≈1: {pos_control_nstar}) and/or no confirmation capability cleared "
                       "MIN_GAIN_FULL — the necessary-rank readout cannot resolve compact storage, so a "
                       "'redundant' verdict would be an artifact of a blind instrument. Reported, not "
                       "worked around (spec §6 VOID).")
        return r

    ci_lo = (delta_ci or {}).get("ci_lo")
    insuff = (delta_ci or {}).get("insufficient")
    # 2. INSUFFICIENT — the paired Δn* was not estimable (too few surviving cells). Withhold, do not
    #    read "did not clear" as "redundant" (the CLEAVE E17 lesson).
    if insuff or ci_lo is None:
        r["verdict"] = "INSUFFICIENT-primary"
        r["reason"] = (f"Δn\\* was not estimable (n={(delta_ci or {}).get('n')} paired capabilities) — "
                       "the primary is UNDECIDED. Missing evidence, not evidence of redundancy; report "
                       "the point estimates and the axes that landed.")
        return r

    # 3. H_capability-dependent — the per-capability Δn* is genuinely bimodal.
    if dip_p is not None and dip_p < 0.05 and n_compact >= 2 and n_redundant >= 2:
        r["verdict"] = "H_capability-dependent"
        r["reason"] = ("the per-capability Δn\\* is bimodal (dip p<0.05) with ≥2 compact and ≥2 "
                       "redundant members → n\\* is capability-dependent; subspace methods work for the "
                       "compact ones, and predicting membership (Arm 3) is the next program.")
        return r

    # 4. H_compact — energy-removal breaks capabilities far earlier than random, above the null band.
    if ci_lo > (null_band_hi if null_band_hi is not None else 0.0):
        r["verdict"] = "H_compact"
        r["reason"] = ("median Δn\\* clears the null band (energy-removal breaks the capability EARLIER "
                       "than a matched random subspace) → storage follows the search: capabilities live "
                       "in the energy directions the gradient used. Subspace methods can work.")
        return r

    # 5. H_redundant — Δn* brackets 0 AND n*_energy sits far above the search-rank.
    redundant = (delta_ci["ci_lo"] <= 0 <= delta_ci["ci_hi"]
                 and nstar_energy_median is not None and nstar_energy_median > 3.0 * search_rank)
    if redundant:
        r["verdict"] = "H_redundant"
        r["reason"] = ("Δn\\* brackets 0 (energy is no more necessary than a random subspace of the same "
                       f"rank) and n\\*_energy ({nstar_energy_median:.0f}) ≫ search-rank ({search_rank}) "
                       "— with the positive control firing, this is DIFFUSE storage, not a blind "
                       "instrument. No subspace method (install/protect/remove) can work: the "
                       "subspace-surgery leg is closed by a law (the hypothesis-to-kill, KILLED by "
                       "surviving its own null).")
        return r

    # 6. neither cleanly — Δn* small but n*_energy near the search-rank (borderline compact).
    r["verdict"] = "H_compact-weak"
    r["reason"] = ("Δn\\* did not clear the null band, but n\\*_energy is not far above the search-rank "
                   f"({nstar_energy_median} vs {search_rank}) — storage is near the search directions "
                   "but the energy-vs-random gap is not resolved at this power; not redundant, not "
                   "decisively compact. Report as borderline; more capabilities would sharpen it.")
    return r
