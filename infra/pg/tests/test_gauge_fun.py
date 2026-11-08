"""Weight-free CPU dry-run gate for GAUGE-FUN / P5 (spec §12.1 P5 suite).

NO model, NO datasets, NO GPU — tiny planted tensors only. Proves the PURE metric MATH + the range
gate + the menu→verdict wiring BEFORE any GPU minute:
  * sym_kl        — identity=0, symmetric, monotone, ≥0;
  * linear_cka/d_rep — self=1/0, orthogonal- + isotropic-scale-invariant, independent<1;
  * fisher_weighted_distance / fisher_diag — F≡1 ⇒ ‖Δ‖, zero-Δ=0, monotone, mean-square Fisher,
    fail-loud on shape/negativity;
  * principal_angle_distance / frozen_ntk_rank — same-grad=0, orthogonal≈1, monotone, midpoint;
  * range_gate — flat family FIRES, spread family does NOT, fail-loud on zero noise;
  * run_menu on a SYNTHETIC pool — planted signal RECOVERED (H_function-metric, the true metric's
    law_verdict True) and a null pool FLAT (H_deflation-deepens), reusing savb.delta_loo_perm +
    relaw.{bh_fdr,loo_sign_stable,law_verdict} verbatim; range-bound short-circuit issues no metric.

Run (on a remote `pg` CPU instance — never locally; torch/numpy are large libs):
    python -m pytest infra/pg/tests/test_gauge_fun.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_fun as GF
from infra.pg import relaw as RL


# ---------------------------------------------------------------------------
# D_KL — symmetrized output KL.
# ---------------------------------------------------------------------------
def test_sym_kl_identical_is_zero():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((7, 11))
    assert GF.sym_kl(L, L) == pytest.approx(0.0, abs=1e-12)


def test_sym_kl_symmetric_and_nonneg():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((5, 9))
    Q = rng.standard_normal((5, 9))
    a = GF.sym_kl(P, Q)
    b = GF.sym_kl(Q, P)
    assert a == pytest.approx(b, rel=1e-12)          # symmetric in its arguments
    assert a > 0.0                                    # distinct distributions ⇒ strictly positive


def test_sym_kl_monotone_in_divergence():
    base = np.zeros((1, 4))
    near = np.array([[0.2, 0.0, 0.0, 0.0]])
    far = np.array([[6.0, 0.0, 0.0, 0.0]])            # a near-one-hot, very far from uniform
    assert GF.sym_kl(base, near) < GF.sym_kl(base, far)


# ---------------------------------------------------------------------------
# D_rep — 1 − linear CKA.
# ---------------------------------------------------------------------------
def test_d_rep_self_is_zero():
    X = np.random.default_rng(2).standard_normal((30, 8))
    assert GF.d_rep(X, X) == pytest.approx(0.0, abs=1e-10)
    assert GF.linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


def test_linear_cka_orthogonal_and_scale_invariant():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 6))
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))   # orthogonal reparam of features
    assert GF.linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-10)
    assert GF.linear_cka(X, 5.0 * X) == pytest.approx(1.0, abs=1e-10)   # isotropic scale
    # a rigid shift (adds a constant row) is removed by the centring ⇒ still identical.
    assert GF.linear_cka(X, X + 3.0) == pytest.approx(1.0, abs=1e-10)


def test_linear_cka_independent_below_one():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 5))
    Y = rng.standard_normal((60, 5))
    c = GF.linear_cka(X, Y)
    assert 0.0 <= c < 0.9 and GF.d_rep(X, Y) > 0.1


# ---------------------------------------------------------------------------
# D_fisher — diagonal-Fisher-weighted parameter distance.
# ---------------------------------------------------------------------------
def test_fisher_distance_unit_fisher_is_l2():
    d = np.array([3.0, -4.0, 0.0])
    F = np.ones(3)
    assert GF.fisher_weighted_distance(F, d) == pytest.approx(5.0, rel=1e-12)   # ‖(3,4,0)‖


def test_fisher_distance_zero_delta_and_monotone():
    F = np.array([1.0, 2.0, 0.5])
    assert GF.fisher_weighted_distance(F, np.zeros(3)) == pytest.approx(0.0, abs=1e-12)
    small = GF.fisher_weighted_distance(F, np.array([0.1, 0.1, 0.1]))
    big = GF.fisher_weighted_distance(F, np.array([1.0, 1.0, 1.0]))
    assert big > small


def test_fisher_diag_is_mean_square():
    G = np.array([[1.0, 2.0], [3.0, 4.0]])            # mean of squares per column
    assert np.allclose(GF.fisher_diag(G), [(1 + 9) / 2, (4 + 16) / 2])


def test_fisher_distance_fail_loud():
    with pytest.raises(AssertionError):
        GF.fisher_weighted_distance(np.ones(3), np.ones(4))     # shape mismatch
    with pytest.raises(AssertionError):
        GF.fisher_weighted_distance(np.array([-1.0, 1.0]), np.ones(2))   # negative Fisher


# ---------------------------------------------------------------------------
# D_ntk — principal-angle distance between top-r gradient subspaces.
# ---------------------------------------------------------------------------
def test_d_ntk_same_grads_is_zero():
    G = np.random.default_rng(5).standard_normal((50, 12))
    assert GF.principal_angle_distance(G, G, r=3) == pytest.approx(0.0, abs=1e-9)


def test_d_ntk_orthogonal_subspaces_near_one():
    # base grads live in the first 4 coords, unit grads in a DISJOINT block of 4 ⇒ top-r subspaces
    # are orthogonal ⇒ overlap≈0 ⇒ distance≈1.
    rng = np.random.default_rng(6)
    N, p, r = 80, 16, 4
    Gb = np.zeros((N, p)); Gb[:, 0:4] = rng.standard_normal((N, 4))
    Gu = np.zeros((N, p)); Gu[:, 8:12] = rng.standard_normal((N, 4))
    assert GF.principal_angle_distance(Gb, Gu, r) > 0.95


def test_d_ntk_monotone_in_rotation():
    rng = np.random.default_rng(7)
    N, p, r = 100, 10, 3
    Gb = rng.standard_normal((N, p))
    near = Gb + 0.02 * rng.standard_normal((N, p))
    far = Gb + 2.0 * rng.standard_normal((N, p))
    assert GF.principal_angle_distance(Gb, near, r) < GF.principal_angle_distance(Gb, far, r)


def test_d_ntk_rank_out_of_range_raises():
    with pytest.raises(AssertionError):
        GF.principal_angle_distance(np.ones((4, 3)), np.ones((4, 3)), r=9)


def test_frozen_ntk_rank_is_grid_midpoint():
    assert GF.frozen_ntk_rank([8, 16, 32, 64, 128, 256, 512]) == 64
    assert GF.frozen_ntk_rank([1, 2, 3]) == 2


# ---------------------------------------------------------------------------
# range gate — G spread vs the LOO noise band.
# ---------------------------------------------------------------------------
def test_range_gate_flat_family_fires():
    rng = np.random.default_rng(8)
    G = 0.5 + 0.01 * rng.standard_normal(20)          # nearly constant G
    sig = 0.10 * np.ones(20)                            # noise ≫ spread ⇒ SNR ≪ 1
    g = GF.range_gate(G, sig)
    assert g["range_bound"] is True and g["snr"] < 1.0


def test_range_gate_spread_family_does_not_fire():
    G = np.linspace(0.0, 1.0, 20)                       # large spread
    sig = 0.01 * np.ones(20)                            # tiny noise ⇒ SNR ≫ 1
    g = GF.range_gate(G, sig)
    assert g["range_bound"] is False and g["snr"] > 1.0


def test_range_gate_zero_noise_raises():
    with pytest.raises(AssertionError):
        GF.range_gate(np.linspace(0, 1, 5), np.zeros(5))


# ---------------------------------------------------------------------------
# run_menu — the estimand end-to-end on a SYNTHETIC pool (reuses the RELAW instrument).
# ---------------------------------------------------------------------------
def _pool_with_signal(n=24, seed=0):
    """One menu metric (D_KL) truly predicts G beyond a noise R_dist baseline; the rest are noise."""
    rng = np.random.default_rng(seed)
    d_kl = rng.uniform(0.4, 1.0, n)
    G = RL.planted_g(d_kl, slope=2.0, noise_sd=0.05, seed=seed + 1)
    R_dist = rng.normal(0, 1, n)                        # independent of G
    metrics = {"D_KL": d_kl,
               "D_rep": rng.normal(0, 1, n),
               "D_fisher": rng.normal(0, 1, n),
               "D_ntk": rng.normal(0, 1, n)}
    g_sigmas = 0.01 * np.ones(n)                        # tiny noise ⇒ range gate passes
    return metrics, R_dist, G, g_sigmas


def test_run_menu_recovers_planted_metric():
    metrics, R_dist, G, g_sig = _pool_with_signal()
    res = GF.run_menu(metrics, R_dist, G, g_sig, q=0.10, n_perm=500, seed=1337)
    assert res["metrics_issued"] is True
    assert res["range_gate"]["range_bound"] is False
    assert res["verdict"] == "H_function-metric"
    assert res["verdicts"]["D_KL"] is True             # the true metric clears the RELAW rule
    assert res["per_metric"]["D_KL"]["delta_loo_r2"] > 0.5
    assert res["per_metric"]["D_KL"]["p_perm"] < 0.05


def test_run_menu_null_pool_deflation_deepens():
    # G driven ENTIRELY by R_dist; all menu metrics independent noise ⇒ none adds LOO power.
    rng = np.random.default_rng(42)
    n = 24
    R_dist = rng.normal(0, 1, n)
    G = 3.0 * R_dist + rng.normal(0, 0.05, n)           # R_dist already explains G (real spread)
    metrics = {m: rng.normal(0, 1, n) for m in GF.MENU}
    g_sig = 0.01 * np.ones(n)                            # spread ≫ noise ⇒ gate does NOT fire
    res = GF.run_menu(metrics, R_dist, G, g_sig, q=0.10, n_perm=500, seed=1337)
    assert res["metrics_issued"] is True
    assert res["range_gate"]["range_bound"] is False
    assert res["verdict"] == "H_deflation-deepens"
    assert not any(res["verdicts"].values())


def test_run_menu_range_bound_short_circuits():
    # flat G family ⇒ the range gate fires FIRST and no metric verdict is issued.
    rng = np.random.default_rng(3)
    n = 20
    G = 0.5 + 0.01 * rng.standard_normal(n)
    R_dist = rng.normal(0, 1, n)
    metrics = {"D_KL": rng.normal(0, 1, n)}
    g_sig = 0.10 * np.ones(n)
    res = GF.run_menu(metrics, R_dist, G, g_sig, q=0.10, n_perm=200, seed=1337)
    assert res["verdict"] == "H_range-bound"
    assert res["metrics_issued"] is False
    assert "per_metric" not in res


def test_run_menu_rejects_off_menu_metric():
    metrics, R_dist, G, g_sig = _pool_with_signal()
    metrics = dict(metrics)
    metrics["D_made_up"] = np.zeros(G.size)             # not in the frozen menu
    with pytest.raises(AssertionError):
        GF.run_menu(metrics, R_dist, G, g_sig, n_perm=50)
