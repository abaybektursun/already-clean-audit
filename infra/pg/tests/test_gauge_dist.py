"""Weight-free CPU dry-run gate for GAUGE-DIST (P6) — §12.1 P6 suite.

NO model, NO GPU — a TOY QUADRATIC readout only. The whole ε* search is exercised against a landscape
with a KNOWN closed-form threshold, so the bisection is CHECKABLE:

    ΔL(edit) = ½ editᵀ H edit,   H PSD.
For a fixed unit direction d on a support (‖d‖₂=1), edit = ε·base_norm·d ⇒
    ΔL(ε) = ½ ε² base_norm² (dᵀH d).
Setting ΔL = δ gives the analytic threshold
    ε*(d) = sqrt(2δ / (dᵀH d)) / base_norm,
monotone-decreasing in the direction's curvature c=dᵀH d. The adversarial (max-curvature) direction is
the top eigenvector of the support-restricted P_S H P_S, so ε*_adv ≤ ε*_rand; a support whose max
curvature is SMALLER than global's yields ε*(S) > ε*(global), i.e. a locality premium π>1.

Proves: the frozen grid; random/adversarial direction construction; bisection finds the analytic ε*
(to the grid resolution) and flags floored/saturated; adversarial power-iteration reaches the top
eigenvector; the premium ratio, count-matched random band, analytic overlay, budget-doubling gate,
seed consistency, and the frozen Detection predicate. Run remotely:
    python -m pytest infra/pg/tests/test_gauge_dist.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_dist as GD


# ---------------------------------------------------------------------------
# a toy quadratic readout with a KNOWN closed-form ε*.
# ---------------------------------------------------------------------------
def _psd(d, seed, floor=0.1):
    """A random PSD H = A Aᵀ/d + floor·I over d coords (strictly positive-definite)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T / d + floor * np.eye(d)


def _readout(H):
    """ΔL(edit) = ½ editᵀ H edit."""
    def f(edit):
        x = np.asarray(edit, dtype=np.float64)
        return 0.5 * float(x @ H @ x)
    return f


def _grad(H):
    """∇ΔL(edit) = H edit."""
    def g(edit):
        return H @ np.asarray(edit, dtype=np.float64)
    return g


def _analytic_eps(H, direction, base_norm, delta):
    d = np.asarray(direction, dtype=np.float64)
    c = float(d @ H @ d)                       # curvature along the direction
    return np.sqrt(2.0 * delta / c) / base_norm


# ---------------------------------------------------------------------------
# frozen grid + masks.
# ---------------------------------------------------------------------------
def test_frozen_grid_spans_decades_ascending():
    g = GD.frozen_norm_grid(1e-4, 1e0, 64)
    assert g.size == 64
    assert np.all(np.diff(g) > 0)                      # strictly ascending
    assert g[0] == pytest.approx(1e-4) and g[-1] == pytest.approx(1e0)
    assert np.log10(g[-1] / g[0]) == pytest.approx(4.0)   # spans 4 decades (§9.3)


def test_mask_from_ranges_and_bounds():
    m = GD.mask_from_ranges(10, [(0, 3), (7, 10)])
    assert m.tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    with pytest.raises(AssertionError):
        GD.mask_from_ranges(10, [(5, 12)])            # out of bounds → RAISES


def test_random_direction_is_unit_on_support():
    m = GD.mask_from_ranges(20, [(4, 12)])
    d = GD.random_support_direction(m, seed=0)
    assert d.shape == (20,)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert not d[~m].any()                             # zero off-support


# ---------------------------------------------------------------------------
# THE core check: bisection recovers the analytic ε* for a fixed direction.
# ---------------------------------------------------------------------------
def test_bisection_finds_analytic_eps_star():
    d_dim, base_norm, delta = 24, 3.0, 0.5
    H = _psd(d_dim, seed=1)
    mask = np.ones(d_dim, dtype=bool)                  # global support
    direction = GD.random_support_direction(mask, seed=7)
    eps_true = _analytic_eps(H, direction, base_norm, delta)
    grid = GD.frozen_norm_grid(1e-3, 1e1, 128)
    res = GD.eps_star_for_direction(_readout(H), direction, base_norm, grid, delta)
    assert not res["floored"] and not res["saturated"]
    # ε* is the SMALLEST grid point ≥ the analytic threshold: within one grid step ABOVE it, and the
    # grid point just below is < the threshold.
    assert res["eps_star"] >= eps_true
    step = grid[res["idx"]] / grid[res["idx"] - 1]     # multiplicative grid spacing
    assert res["eps_star"] <= eps_true * step * 1.0001
    assert grid[res["idx"] - 1] < eps_true
    # the readout at ε* clears δ; just below it does not (bisection is correct on a monotone ray).
    f = GD.readout_at_norm(_readout(H), direction, base_norm)
    assert f(res["eps_star"]) >= delta
    assert f(grid[res["idx"] - 1]) < delta
    # forward budget is bounded: 2 bracket + ≤⌈log2(128)⌉=7 interior.
    assert res["n_eval"] <= 2 + 7


def test_bisection_floored_and_saturated_flags():
    d_dim, base_norm = 12, 1.0
    H = _psd(d_dim, seed=2)
    direction = GD.random_support_direction(np.ones(d_dim, bool), seed=3)
    f = GD.readout_at_norm(_readout(H), direction, base_norm)
    grid = GD.frozen_norm_grid(1e-4, 1e0, 32)
    # δ below the smallest grid point's ΔL ⇒ floored.
    lo = GD.bisect_threshold(f, grid, delta=f(grid[0]) * 0.5)
    assert lo["floored"] and lo["idx"] == 0
    # δ above the largest grid point's ΔL ⇒ saturated (SUPPORT-UNRESOLVED).
    hi = GD.bisect_threshold(f, grid, delta=f(grid[-1]) * 2.0)
    assert hi["saturated"] and hi["idx"] == grid.size - 1


# ---------------------------------------------------------------------------
# adversarial power-iteration reaches the top eigenvector (max curvature).
# ---------------------------------------------------------------------------
def test_adversarial_direction_maximizes_curvature():
    d_dim = 30
    H = _psd(d_dim, seed=4)
    mask = np.ones(d_dim, dtype=bool)                  # global support: P_S H P_S = H
    # power iteration converges at rate (λ2/λ1)^steps; the MP edge gap is ~1% for a random PSD, so
    # 60 steps left a 1.2% curvature gap. 300 steps drives it well below the 5e-3 convergence bar
    # (a frozen-budget attack in the real job need only exceed random — see the ≥ assertion below).
    d_adv = GD.adversarial_support_direction(_grad(H), mask, seed=0, n_steps=300,
                                             probe_norm=1e-2, base_norm=1.0)
    assert np.linalg.norm(d_adv) == pytest.approx(1.0)
    c_adv = float(d_adv @ H @ d_adv)
    lam_max = float(np.linalg.eigvalsh(H)[-1])
    assert c_adv == pytest.approx(lam_max, rel=5e-3)   # converged to the top eigenvector
    # and its curvature dominates a random direction's ⇒ ε*_adv ≤ ε*_rand.
    d_rand = GD.random_support_direction(mask, seed=9)
    assert c_adv >= float(d_rand @ H @ d_rand)


def test_adversarial_confined_to_support():
    d_dim = 24
    H = _psd(d_dim, seed=5)
    mask = GD.mask_from_ranges(d_dim, [(0, 8)])        # single-block support
    d_adv = GD.adversarial_support_direction(_grad(H), mask, seed=1, n_steps=50,
                                             probe_norm=1e-2, base_norm=1.0)
    assert not d_adv[~mask].any()                      # stays on the support
    # converges to the top eigenvector of the support-restricted block H[S,S].
    Hs = H[np.ix_(mask, mask)]
    lam_block = float(np.linalg.eigvalsh(Hs)[-1])
    assert float(d_adv @ H @ d_adv) == pytest.approx(lam_block, rel=1e-3)


# ---------------------------------------------------------------------------
# premium ratio, count-matched null, analytic overlay.
# ---------------------------------------------------------------------------
def test_premium_ratio_and_count_matched_band():
    # a planted locality premium: global's max curvature is BIG (a spread top-eigenvector), so any
    # localized block captures less curvature ⇒ ε*(block) > ε*(global) ⇒ π>1.
    d_dim, base_norm, delta = 40, 1.0, 0.5
    v = np.ones(d_dim) / np.sqrt(d_dim)                # spread direction
    H = 20.0 * np.outer(v, v) + 0.5 * np.eye(d_dim)    # big curvature only along the SPREAD v
    grid = GD.frozen_norm_grid(1e-3, 1e2, 256)
    g_mask = np.ones(d_dim, dtype=bool)
    s_mask = GD.mask_from_ranges(d_dim, [(0, 5)])      # a small localized block

    def _adv_eps(mask, seed):
        d = GD.adversarial_support_direction(_grad(H), mask, seed, n_steps=80,
                                             probe_norm=1e-2, base_norm=base_norm)
        return GD.eps_star_for_direction(_readout(H), d, base_norm, grid, delta)["eps_star"]

    eps_g = _adv_eps(g_mask, 0)
    eps_s = _adv_eps(s_mask, 0)
    pi = GD.premium_ratio(eps_s, eps_g)
    assert pi > 1.0                                    # localized edit costs more → locality premium

    # count-matched RANDOM-edit band across 3 seeds (the parameter-counting reference).
    def _rand_eps(mask, seed):
        d = GD.random_support_direction(mask, seed)
        return GD.eps_star_for_direction(_readout(H), d, base_norm, grid, delta)["eps_star"]
    r_s = [_rand_eps(s_mask, k) for k in (11, 12, 13)]
    r_g = [_rand_eps(g_mask, k) for k in (11, 12, 13)]
    band = GD.random_ratio_band(r_s, r_g)
    assert band["n_seed"] == 3 and band["lo"] <= band["mean"] <= band["hi"]


def test_analytic_count_overlay():
    assert GD.analytic_count_overlay(4, 16) == pytest.approx(0.5)      # √(4/16)
    assert GD.analytic_count_overlay(16, 16) == pytest.approx(1.0)
    with pytest.raises(AssertionError):
        GD.analytic_count_overlay(4, 0)


# ---------------------------------------------------------------------------
# δ_fun calibration.
# ---------------------------------------------------------------------------
def test_calibrate_delta_fun_is_mean_plus_k_std():
    x = np.array([0.01, -0.02, 0.015, -0.008])
    d = GD.calibrate_delta_fun(x, k=3.0)
    assert d == pytest.approx(np.abs(x).mean() + 3.0 * np.abs(x).std())
    with pytest.raises(AssertionError):
        GD.calibrate_delta_fun([0.01])                 # need ≥2 samples


# ---------------------------------------------------------------------------
# budget-doubling gate + seed consistency.
# ---------------------------------------------------------------------------
def test_budget_doubling_gate():
    # a stable premium moves < the seed band when the budget doubles ⇒ survives.
    assert GD.budget_doubling_ok(1.80, 1.83, seed_band=0.10) is True
    # a search-artifact premium moves MORE than the band ⇒ fails.
    assert GD.budget_doubling_ok(1.80, 1.50, seed_band=0.10) is False


def test_seed_consistency():
    assert GD.seed_consistent([1.4, 1.5, 1.6], null_hi=1.3) is True
    assert GD.seed_consistent([1.4, 1.2, 1.6], null_hi=1.3) is False   # one seed dips under


# ---------------------------------------------------------------------------
# the frozen Detection predicate.
# ---------------------------------------------------------------------------
def test_detection_premium_truth_table():
    null_hi = {"single_matrix": 0.9, "single_layer": 0.9, "band3": 0.9}
    consist = {"single_matrix": True, "single_layer": True, "band3": True}
    # ≥2 localized supports over the band, 3-seed consistent, budget survives → detected.
    pi = {"single_matrix": 1.5, "single_layer": 1.4, "band3": 0.8}
    assert GD.detection_premium(pi, null_hi, consist, budget_ok=True)["detected"] is True
    # only 1 support over the band → not detected.
    pi1 = {"single_matrix": 1.5, "single_layer": 0.85, "band3": 0.8}
    assert GD.detection_premium(pi1, null_hi, consist, budget_ok=True)["detected"] is False
    # 2 supports over the band but the budget-doubling control FAILS → not detected.
    assert GD.detection_premium(pi, null_hi, consist, budget_ok=False)["detected"] is False
    # 2 over the band but one is not seed-consistent → drops below 2 → not detected.
    consist2 = {"single_matrix": True, "single_layer": False, "band3": True}
    assert GD.detection_premium(pi, null_hi, consist2, budget_ok=True)["detected"] is False
    # 'global' is never counted even if huge.
    pig = {"single_matrix": 1.5, "single_layer": 1.4, "band3": 0.8, "global": 99.0}
    out = GD.detection_premium(pig, null_hi, consist, budget_ok=True)
    assert "global" not in out["premium_supports"]
