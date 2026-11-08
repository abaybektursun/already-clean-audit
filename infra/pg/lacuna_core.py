"""LACUNA pure core — every estimand, null band, gate and label in the program, weight-free.

Spec: studies/lineage/paligemma1-pi0/research/2026-07-27_lacuna_program_spec.md
This module holds ONLY pure functions of banked per-item scores / matrices, so the whole §6
decision surface is unit-testable on the control plane before a single GPU minute
(`feedback_offline_gate_before_gpu`). Every heavy import lives inside a function.

Design commitments carried from the house lessons:
  * every floor has a CEILING and every null has a SATURATION guard + a VOID branch
    (`feedback_one_sided_thresholds`) — see `power_gate`, `theta_null`, `label_row`;
  * status flags are computed LAST, mechanically, from numbers already banked
    (`feedback_signals_that_cannot_fail`);
  * the item-paired bootstrap resamples ONE index vector per draw and applies it to every row,
    so all contrasts are paired by construction (spec §4.3).

No try/except (CLAUDE.md): a shape/alignment error RAISES with a trace.
"""
from __future__ import annotations

# S bands (spec §6.1) — pre-set conventions, disclosed as such, NOT imported thresholds.
S_RETAINED = 0.5          # lower95(S) ≥ 0.5 → `retained` (majority of ceiling)
THETA_NULL_MAX = 0.5      # θ_null ≥ this → the null band swallows the retained band → VOID
POWER_MIN_GAP = 0.15      # G3: CEILING − FLOOR ≥ 15pp (score-scale equivalent per task)
N_BOOT = 10_000


# ===========================================================================
# Discovery / confirmation split (spec §4.2) — full-content hash parity.
# ===========================================================================
def parity_split(hashes):
    """(discovery_mask, confirmation_mask) from FULL per-item content hashes.

    Reuses the house rule `w16_probes.is_discovery` = int(h,16) % 2 == 0 on the FULL hash
    (`projector_ablation._content_hash` = md5(prompt ⊕ 32×32 image bytes)). The truncated
    `q[:64]` key collides (W11 lesson) — never key on a prefix."""
    from infra.pg.w16_probes import is_confirmation, is_discovery
    assert len(set(hashes)) == len(hashes), \
        f"item hashes are not unique ({len(hashes) - len(set(hashes))} collisions) — split would be biased"
    disc = [is_discovery(h) for h in hashes]
    conf = [is_confirmation(h) for h in hashes]
    assert all(d ^ c for d, c in zip(disc, conf)), "discovery/confirmation is not a partition"
    return disc, conf


def mask_of(half, disc, conf):
    """half ∈ {discovery, confirmation, full} → a boolean mask list."""
    assert half in ("discovery", "confirmation", "full"), half
    if half == "discovery":
        return list(disc)
    if half == "confirmation":
        return list(conf)
    return [True] * len(disc)


# ===========================================================================
# The S estimand + item-paired bootstrap (spec §2.1, §7).
# ===========================================================================
def s_point(cfg_mean, floor_mean, ceiling_mean):
    """S = (score − FLOOR)/(CEILING − FLOOR), clamped [0,1] (the house estimand, verbatim
    `battery_tasks.bootstrap_S_ci`'s `_S`). nan when the band is degenerate — the caller's
    power gate is what turns that into a VOID, never a silent 0."""
    denom = ceiling_mean - floor_mean
    if denom <= 0:
        return float("nan")
    return float(min(1.0, max(0.0, (cfg_mean - floor_mean) / denom)))


def boot_S_paired(cfg_items, floor_items, ceiling_items, n_boot=N_BOOT, seed=0):
    """Item-PAIRED bootstrap CI for S. One resampled index vector per draw, applied to all three
    per-item arrays (they are position-aligned on the same items — spec §4.3), so the numerator
    and the band move together and the CI is not inflated by independent resampling.

    Pre-freeze amendment vs the ledger's `battery_tasks.bootstrap_S_ci`: identical estimand and
    clamp, PAIRED resampling over items instead of independent resampling over per-seed scores.
    The ledger's estimator is a per-seed resampler; LACUNA's rows are single-pass per-item evals
    on shared items, so pairing is both available and required by §4.3. Banked as such.
    """
    import numpy as np
    c = np.asarray(cfg_items, dtype=float)
    f = np.asarray(floor_items, dtype=float)
    k = np.asarray(ceiling_items, dtype=float)
    assert c.shape == f.shape == k.shape and c.ndim == 1, \
        f"paired S arrays misaligned: {c.shape} {f.shape} {k.shape}"
    n = c.size
    assert n > 1, "need >1 item to bootstrap"
    rng = np.random.default_rng(seed)
    point = s_point(c.mean(), f.mean(), k.mean())
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = s_point(c[idx].mean(), f[idx].mean(), k[idx].mean())
    valid = draws[~np.isnan(draws)]
    assert valid.size > 0.5 * n_boot, \
        f"S bootstrap degenerate: {valid.size}/{n_boot} draws had CEILING ≤ FLOOR"
    return {"S_point": point,
            "S_lo95": float(np.percentile(valid, 2.5)),
            "S_hi95": float(np.percentile(valid, 97.5)),
            "n_boot_valid": int(valid.size), "n_items": int(n), "seed": int(seed),
            # S is clamped to [0,1]; a row ABOVE the ceiling saturates at 1.0. The raw excess is
            # banked so "matches the ceiling" and "beats the ceiling" stay distinguishable.
            "raw_excess_over_ceiling": float(c.mean() - k.mean()),
            "raw_mean": float(c.mean()), "raw_floor": float(f.mean()), "raw_ceiling": float(k.mean())}


def boot_mean_ci(items, n_boot=N_BOOT, seed=0):
    """Plain item-bootstrap mean CI (raw score reporting; every number gets a CI, §7)."""
    import numpy as np
    x = np.asarray(items, dtype=float)
    assert x.ndim == 1 and x.size > 1, f"bad score array {x.shape}"
    rng = np.random.default_rng(seed)
    d = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    return {"mean": float(x.mean()), "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)), "n": int(x.size)}


def boot_diff_ci(a_items, b_items, n_boot=N_BOOT, seed=0):
    """Paired item-bootstrap CI + one-sided p for mean(a) − mean(b) > 0.

    `p_le0` = P̂(bootstrapped difference ≤ 0), the achieved-significance level for the registered
    positive direction. Both tails are banked so a NEGATIVE landing is reported, never hidden."""
    import numpy as np
    a = np.asarray(a_items, dtype=float)
    b = np.asarray(b_items, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, f"paired diff arrays misaligned {a.shape} {b.shape}"
    rng = np.random.default_rng(seed)
    n = a.size
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = a[idx].mean() - b[idx].mean()
    return {"diff": float(a.mean() - b.mean()),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_le0": float(((d <= 0).sum() + 1) / (n_boot + 1)),
            "p_ge0": float(((d >= 0).sum() + 1) / (n_boot + 1)), "n": int(n)}


def delta_perception(real_items, gray_items, n_boot=N_BOOT, seed=0):
    """Δ_perception = acc(real) − acc(gray), paired per item (spec §2.1). The discriminator RFP
    did not have: it says whether image information FLOWS, independent of how much task score
    survives."""
    out = boot_diff_ci(real_items, gray_items, n_boot=n_boot, seed=seed)
    out["alive"] = bool(out["lo95"] > 0)
    return out


# ===========================================================================
# The null band (spec §6.1) — R6 split-half against itself, on control data only.
# ===========================================================================
def theta_null(floor_items, ceiling_items, n_boot=N_BOOT, seed=20260727):
    """θ_null = upper95 of the S a FLOOR row scores against ITSELF.

    Construction (spec §6.1 "R6 split-half against itself"): each draw seeds a fresh random
    half/half partition of the floor's items; half A plays the row under test, half B plays the
    FLOOR, the CEILING is resampled on the same items. The 97.5th percentile of the resulting S
    distribution is the band above which a row is not floor-like.

    Ceiling / saturation guard (`feedback_one_sided_thresholds`): a θ_null ≥ THETA_NULL_MAX means
    the null band has swallowed the `retained` band, so `consumed` could not fail — `saturated`
    is returned True and the caller must VOID the labels rather than report them.
    """
    import numpy as np
    f = np.asarray(floor_items, dtype=float)
    k = np.asarray(ceiling_items, dtype=float)
    assert f.shape == k.shape and f.ndim == 1, f"null-band arrays misaligned {f.shape} {k.shape}"
    n = f.size
    assert n >= 8, f"need ≥8 floor items to split-half the null band (got {n})"
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        perm = rng.permutation(n)
        a_idx, b_idx = perm[: n // 2], perm[n // 2:]
        s = s_point(f[a_idx].mean(), f[b_idx].mean(), k[perm].mean())
        if not np.isnan(s):
            draws.append(s)
    draws = np.asarray(draws)
    assert draws.size > 0.5 * n_boot, "null band degenerate (CEILING ≤ FLOOR in most draws)"
    theta = float(np.percentile(draws, 97.5))
    return {"theta_null": theta, "median": float(np.median(draws)),
            "saturated": bool(theta >= THETA_NULL_MAX),
            "note": ("NULL-BAND SATURATED — θ_null ≥ %.2f swallows the retained band; labels VOID"
                     % THETA_NULL_MAX) if theta >= THETA_NULL_MAX else "null band usable",
            "n_boot_valid": int(draws.size), "seed": int(seed)}


# ===========================================================================
# Gates (spec §6.1) — order fixed, flags computed LAST.
# ===========================================================================
def power_gate(floor_items, ceiling_items, min_gap=POWER_MIN_GAP, n_boot=2000, seed=7):
    """G3: the task carries confirmatory weight iff CEILING − FLOOR ≥ min_gap AND the gap's
    lower-CI clears 0. Both directions matter: too small a gap = no power (N1's "null too weak"),
    and a NEGATIVE gap = the floor beat the ceiling = instrument bug."""
    d = boot_diff_ci(ceiling_items, floor_items, n_boot=n_boot, seed=seed)
    ok = bool(d["diff"] >= min_gap and d["lo95"] > 0)
    return {"gap": d["diff"], "gap_lo95": d["lo95"], "gap_hi95": d["hi95"],
            "min_gap": float(min_gap), "passes": ok,
            "inverted": bool(d["hi95"] < 0),
            "note": ("power OK" if ok else
                     ("INVERTED — floor above ceiling (instrument bug)" if d["hi95"] < 0
                      else "under-powered: no confirmatory weight for this task"))}


def label_row(s_ci, theta, dperc=None):
    """The per-(row, task) label, computed mechanically AFTER G1–G4 pass (spec §6.1).

      consumed  : S ≤ θ_null (the row lands INSIDE the band a floor row occupies)
                  AND upper95(S) < 0.5 (retention is excluded — the power guard)
      retained  : lower95(S) ≥ 0.5
      partial   : otherwise — ALWAYS reported with S + CI, never as a bare flag
      perception-alive: lower95(Δ_perception) > 0 — an ORTHOGONAL label, reported alongside.

    PRE-FREEZE AMENDMENT-5 (found by the G0 dry-run, spec §6.1's rule as written could not fire).
    The spec's rule was `consumed := upper95(S) < θ_null`. θ_null is the 97.5th percentile of the
    S a floor row scores against itself, so it is *smaller than* the sampling half-width of S; a
    genuinely-at-floor row therefore has upper95(S) > θ_null about half the time and would be
    labelled `partial`. The replayed known event the spec itself mandates (§6.1-G0: "a synthetic
    'policy ≈ null' row must emit `consumed`") FAILED on the dry-run — the defect is in the rule,
    not the data.

    The corrected rule keeps the same two objects and adds nothing new: the LABEL is a
    band-membership statement about the point estimate (is the row inside the floor's band?), and
    the CONFIRMATORY TEST is unchanged — the one-sided `p_above_band` on the bootstrap draws is
    what enters the BH family, so nothing about the T1/T2 evidence standard is loosened. The extra
    `upper95(S) < 0.5` clause is a guard in the OTHER direction: a row whose CI cannot exclude
    retention is never called `consumed`, it is called `partial` and flagged `underpowered`.
    """
    assert not (s_ci["S_point"] <= theta and s_ci["S_lo95"] >= S_RETAINED), \
        "label logic contradiction: a row cannot be both inside the null band and retained"
    inside_band = s_ci["S_point"] <= theta
    excludes_retention = s_ci["S_hi95"] < S_RETAINED
    if inside_band and excludes_retention:
        label = "consumed"
    elif s_ci["S_lo95"] >= S_RETAINED:
        label = "retained"
    else:
        label = "partial"
    out = {"label": label, "S": s_ci["S_point"], "S_lo95": s_ci["S_lo95"],
           "S_hi95": s_ci["S_hi95"], "theta_null": float(theta),
           "inside_null_band": bool(inside_band),
           "underpowered": bool(inside_band and not excludes_retention)}
    # CEILING-EXCEEDED disclosure. S clamps to [0, 1], so a row that BEATS the CEILING is reported
    # as S = 1.0 and is indistinguishable from a row that merely matches it — the saturation guard
    # in the upper direction. `exceeds_ceiling` + the raw excess make that visible instead of
    # silently flattening it. (Not a rule change: S is unmodified, this is a flag beside it.)
    if s_ci.get("raw_excess_over_ceiling") is not None:
        out["exceeds_ceiling"] = bool(s_ci["raw_excess_over_ceiling"] > 0)
        out["raw_excess_over_ceiling"] = s_ci["raw_excess_over_ceiling"]
    if dperc is not None:
        out["perception_alive"] = bool(dperc["alive"])
        out["delta_perception"] = dperc["diff"]
        out["delta_perception_lo95"] = dperc["lo95"]
    return out


def scope_cap_g5(policy_labels, policy_dperc):
    """G5 saturation guard (spec §6.1). If BOTH policies sit inside the null band on EVERY task
    AND both Δ_perception lower-CIs ≤ 0, the verdict language is capped — the H_uniform vs
    H_shell residual is DISCLOSED, not adjudicated.

    `policy_labels`: {policy: {task: label_str}}; `policy_dperc`: {policy: lo95 float}.
    """
    all_floor = all(lab == "consumed" for p in policy_labels.values() for lab in p.values())
    no_perc = all(lo <= 0 for lo in policy_dperc.values())
    fired = bool(all_floor and no_perc and policy_labels and policy_dperc)
    return {"g5_fired": fired,
            "all_tasks_in_null_band": bool(all_floor),
            "no_perception_anywhere": bool(no_perc),
            "capped_verdict": ("no vision-language function is readable through the PG1 shell for "
                               "either policy" if fired else None),
            "note": ("H_uniform vs H_shell NOT adjudicated — inherited RFP limit, pre-committed cap"
                     if fired else "G5 not fired — the labels stand as computed")}


# ===========================================================================
# Multiplicity (spec §7): Benjamini–Hochberg over the frozen confirmatory family.
# ===========================================================================
def bh_reject(pvals, q=0.05):
    """Benjamini–Hochberg step-up. `pvals` = {test_id: p or None}. A test with p=None is VOID
    (gate-failed) and is EXCLUDED from the family — m counts only the tests actually run, and the
    excluded ids are returned so the report can never silently shrink the family.
    """
    import numpy as np
    live = {k: float(v) for k, v in pvals.items() if v is not None}
    void = sorted(k for k, v in pvals.items() if v is None)
    m = len(live)
    if m == 0:
        return {"m": 0, "q": q, "rejected": [], "void": void, "crit_p": None, "ranked": []}
    ids = sorted(live, key=lambda k: live[k])
    ps = np.array([live[i] for i in ids])
    thresh = q * np.arange(1, m + 1) / m
    below = np.nonzero(ps <= thresh)[0]
    kmax = int(below.max()) if below.size else -1
    rejected = ids[: kmax + 1] if kmax >= 0 else []
    return {"m": m, "q": float(q), "rejected": rejected, "void": void,
            "crit_p": float(ps[kmax]) if kmax >= 0 else None,
            "ranked": [{"id": i, "p": float(live[i]), "bh_thresh": float(t)}
                       for i, t in zip(ids, thresh)]}


def p_above_band(s_draws, theta):
    """One-sided achieved-significance for "S is ABOVE the null band": P̂(S_boot ≤ θ_null)."""
    import numpy as np
    d = np.asarray(s_draws, dtype=float)
    d = d[~np.isnan(d)]
    return float(((d <= theta).sum() + 1) / (d.size + 1))


def boot_S_draws(cfg_items, floor_items, ceiling_items, n_boot=N_BOOT, seed=0):
    """The raw S bootstrap draws (for `p_above_band`), same resampling as `boot_S_paired`."""
    import numpy as np
    c = np.asarray(cfg_items, float); f = np.asarray(floor_items, float)
    k = np.asarray(ceiling_items, float)
    assert c.shape == f.shape == k.shape
    rng = np.random.default_rng(seed)
    n = c.size
    return np.array([s_point(c[i].mean(), f[i].mean(), k[i].mean())
                     for i in (rng.integers(0, n, n) for _ in range(n_boot))])


# ===========================================================================
# Arm 2 — weight/geometry census statistics (spec §2.2, §6.2).
# ===========================================================================
def stable_rank(mat):
    """‖M‖²_F / σ₁² — the W1-analogue rank statistic (fp64).

    σ₁ comes from an eigendecomposition of the SMALLER Gram, not a full SVD: only the largest
    singular value is needed, and for the LM's 16384×2048 MLP matrices the full factorisation is
    ~20× the cost of the 2048² eigensolve. Verified identical to `svd(...)[0]` to 1e-8."""
    import numpy as np
    M = np.asarray(mat, dtype=np.float64)
    fro2 = float((M ** 2).sum())
    assert fro2 > 0, "zero matrix has no stable rank"
    G = (M.T @ M) if M.shape[1] <= M.shape[0] else (M @ M.T)
    s1sq = float(max(np.linalg.eigvalsh(G).max(), 1e-300))
    return fro2 / s1sq


_SRANK_BAND_CACHE = {}


def srank_random_band(shape, n_draws=32, seed=0):
    """Frobenius-matched random-ΔW band for `stable_rank`. Norm-matching does not change stable
    rank (it is scale-invariant), so the matched null is SHAPE-only — stated explicitly so the
    band is not mistaken for a norm-dependent calibration.

    Being shape-only, the band is also CACHED per (shape, n_draws, seed): the LM grid has ~4
    distinct shapes across 126 matrices, so recomputing it per matrix was 30× redundant work."""
    import numpy as np
    key = (tuple(shape), int(n_draws), int(seed))
    if key in _SRANK_BAND_CACHE:
        return _SRANK_BAND_CACHE[key]
    rng = np.random.default_rng(seed)
    v = np.asarray([stable_rank(rng.standard_normal(shape)) for _ in range(n_draws)])
    band = {"mean": float(v.mean()), "p2_5": float(np.percentile(v, 2.5)),
            "p97_5": float(np.percentile(v, 97.5)), "n_draws": int(n_draws),
            "shape": list(shape), "cached": True,
            "note": "shape-only band; stable rank is scale-invariant so Frobenius matching is a no-op"}
    _SRANK_BAND_CACHE[key] = band
    return band


def overlap_vs_rotation_null(basis_a, basis_b, n_draw=64, seed=0):
    """Top-k principal-subspace overlap vs the analytic matched-random ROTATION null (w12 infra,
    REUSED: `w8_weight_activation.subspace_overlap` + `empirical_random_subspace_overlap`).
    Preserved iff overlap > the null's p95; rewritten iff ≤ it."""
    import numpy as np
    from infra.pg.w8_weight_activation import (
        empirical_random_subspace_overlap, subspace_overlap,
    )
    A = np.asarray(basis_a, dtype=np.float64)
    B = np.asarray(basis_b, dtype=np.float64)
    ov = subspace_overlap(A, B)
    null = empirical_random_subspace_overlap(A.shape[0], A.shape[1], n_draw=n_draw, seed=seed)
    return {"overlap": float(ov), "null_p95": null["empirical_p95"],
            "null_mean": null["empirical_mean"], "analytic_floor": null["analytic_floor_k_over_d"],
            "above_null": bool(ov > null["empirical_p95"]), "k": int(A.shape[1]),
            "d": int(A.shape[0])}


def preservation_P(ov, floor, ceiling):
    """2d normalized geometry-preservation P = (ov − FLOOR)/(CEILING − FLOOR) (spec §2.2).
    VOID handling is the caller's: the band's own lower-CI must clear 0 (§6.2)."""
    denom = ceiling - floor
    if denom <= 0:
        return float("nan")
    return float((ov - floor) / denom)


def band_label_P(p_point, p_lo=None, p_hi=None):
    """§6.2 bands: preserved P ≥ 0.75 / rewritten P < 0.25 / partial otherwise (pre-set
    convention, labelled as such). Uses the point estimate with the CI reported beside it.

    RANGE GUARD (added 2026-07-28 after the 2d-v1 instrument failure): P is only defined inside
    [0, 1]. A P above 1 means the measured quantity EXCEEDED its own ceiling, i.e. the ceiling is
    not a ceiling — v1 of the 2d job reported `preserved` at P = 4.6 because the label only ever
    checked the lower side. `VOID-INVALID` is the branch that makes that observable instead of
    silently flattering the result (`feedback_one_sided_thresholds`)."""
    if p_point != p_point:      # nan
        return "VOID"
    if p_point > 1.0 + 1e-9:
        return "VOID-INVALID (statistic above its own ceiling — the band is not a band)"
    if p_point < -1e-9:
        return "VOID-INVALID (statistic below its own floor — the floor is not a floor)"
    if p_point >= 0.75:
        return "preserved"
    if p_point < 0.25:
        return "rewritten"
    return "partial"


# ===========================================================================
# Arm 3 — savings (spec §2.3, §6.3).
# ===========================================================================
def savings_ci(unit_scores, ref_scores, n_boot=N_BOOT, seed=0):
    """savings(U) = mean pc_success(U) − mean pc_success(U3) at matched budget, bootstrap over the
    (task × episode × seed) units. Unpaired resampling: the two arms are separate rollouts."""
    import numpy as np
    a = np.asarray(unit_scores, dtype=float)
    b = np.asarray(ref_scores, dtype=float)
    assert a.ndim == b.ndim == 1 and a.size > 1 and b.size > 1
    rng = np.random.default_rng(seed)
    d = np.array([a[rng.integers(0, a.size, a.size)].mean() - b[rng.integers(0, b.size, b.size)].mean()
                  for _ in range(n_boot)])
    return {"savings": float(a.mean() - b.mean()), "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)),
            "p_le0": float(((d <= 0).sum() + 1) / (n_boot + 1)),
            "n_unit": int(a.size), "n_ref": int(b.size)}


def range_guard(unit_means, sat_hi=0.8, n_units_required=2):
    """§6.3 range guard: saturation (all units ≥ sat_hi) OR collapse (all units ≈ 0) both leave the
    contrast unmeasurable. Both directions are named so the guard cannot only fail one way."""
    vals = list(unit_means.values())
    assert len(vals) >= n_units_required, f"range guard needs ≥{n_units_required} units"
    saturated = all(v >= sat_hi for v in vals)
    collapsed = all(v <= 0.02 for v in vals)
    return {"saturated": bool(saturated), "collapsed": bool(collapsed),
            "void_range": bool(saturated or collapsed), "unit_means": dict(unit_means),
            "note": ("VOID-range: all units saturated" if saturated else
                     "VOID-range: every unit at zero (no dynamic range)" if collapsed else
                     "range OK")}
