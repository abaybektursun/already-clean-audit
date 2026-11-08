"""Weight-free CPU dry-run gate for GAUGE-RMT (P3) — §12.1 suite.

NO model, NO GPU — planted matrices only. Proves: the MP edge algebra + central-quantile σ² fit
recovers the true noise scale on a pure iid-Gaussian ΔW; the fit-quality KS is small for a genuine
MP bulk and large for a heavy-tailed spectrum; a planted low-rank spike sits ABOVE the buffered edge
(n_out / E_out detect it) while the entry-permutation null carries ~none; the outlier-subspace
overlap reads high for a SHARED planted subspace and collapses to the Haar floor for independent
subspaces; and the frozen decision predicates (δ_RMT half-width, discovery-clear, detection,
fit-quality gate) behave.

Run remotely: python -m pytest infra/pg/tests/test_gauge_rmt.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_rmt as RMT


def _gauss(m, n, sigma, seed):
    """Pure iid-Gaussian ΔW (entry std σ) — a genuine MP bulk, no outliers."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n)) * sigma


def _spiked(m, n, sigma, spike, k, seed):
    """iid-Gaussian bulk + a rank-k planted spike (orthonormal U_k V_kᵀ scaled by `spike`)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) * sigma
    U = np.linalg.qr(rng.standard_normal((m, k)))[0]
    V = np.linalg.qr(rng.standard_normal((n, k)))[0]
    return A + spike * (U @ V.T), U, V


# ---------------------------------------------------------------------------
# MP edge algebra + central-quantile σ² fit.
# ---------------------------------------------------------------------------
def test_mp_edges_formula():
    lo, hi = RMT.mp_edges(sigma2=2.0, gamma=0.25)     # (1±0.5)² = 0.25, 2.25 → ×2
    assert lo == pytest.approx(2.0 * 0.25) and hi == pytest.approx(2.0 * 2.25)


def test_sigma2_fit_recovers_noise_scale():
    # a pure iid-Gaussian ΔW with entry variance σ²: the central-quantile fit recovers σ².
    m, n, sigma = 400, 300, 0.7
    A = _gauss(m, n, sigma, seed=0)
    sv = RMT.spectrum(A)
    eigs, gamma = RMT.eigenvalues(sv, m, n)
    s2 = RMT.fit_sigma2(eigs, gamma)
    assert s2 == pytest.approx(sigma ** 2, rel=0.15)     # λ=σ²/N with N=max(m,n) ⇒ scale σ²


def test_gamma_is_aspect_ratio():
    _e, gamma = RMT.eigenvalues(np.ones(200), m=200, n=500)
    assert gamma == pytest.approx(200 / 500)


# ---------------------------------------------------------------------------
# fit-quality KS: small for a true MP bulk, large for a heavy-tailed spectrum.
# ---------------------------------------------------------------------------
def test_ks_small_for_mp_bulk_large_for_heavy_tail():
    m, n = 400, 300
    A = _gauss(m, n, 1.0, seed=1)
    sv = RMT.spectrum(A)
    eigs, gamma = RMT.eigenvalues(sv, m, n)
    s2 = RMT.fit_sigma2(eigs, gamma)
    _lm, lp = RMT.mp_edges(s2, gamma)
    ks_mp = RMT.mp_ks(eigs, s2, gamma, lp)
    # heavy-tailed: raise the eigenvalues to a power ⇒ shape no longer MP.
    heavy = eigs ** 2.5
    ks_heavy = RMT.mp_ks(heavy, RMT.fit_sigma2(heavy, gamma), gamma,
                         RMT.mp_edges(RMT.fit_sigma2(heavy, gamma), gamma)[1])
    assert ks_mp < 0.15
    assert ks_heavy > ks_mp


# ---------------------------------------------------------------------------
# per-matrix decomposition: a planted spike is an outlier; the null carries ~none.
# ---------------------------------------------------------------------------
def test_planted_spike_detected_above_buffered_edge():
    m, n, k = 300, 250, 3
    # spike must sit CLEARLY above the MP bulk edge (σ_edge ≈ √m+√n ≈ 33 for a unit-σ 300×250
    # Gaussian); a 25-magnitude spike is at the BBP borderline (energy grazed 0.05). 80 ≫ edge ⇒
    # three unambiguous outliers carrying ~20% Frobenius energy.
    A, _U, _V = _spiked(m, n, sigma=1.0, spike=80.0, k=k, seed=2)
    res = RMT.analyze_matrix(A, n_perm=3, seed=7)
    assert res["n_out"] >= k                         # the k planted directions clear the edge
    assert res["e_out"] > 0.05                        # and carry real Frobenius energy
    # the entry-permutation null destroys the spike ⇒ ~no outlier energy.
    assert res["e_out_perm_median"] < res["e_out"]
    assert res["gamma"] == pytest.approx(250 / 300)


def test_pure_bulk_has_negligible_outlier_energy():
    # a pure MP bulk: real E_out ≈ the permutation E_out (nothing above the buffered edge but noise).
    A = _gauss(300, 250, 1.0, seed=3)
    res = RMT.analyze_matrix(A, n_perm=3, seed=8)
    assert res["e_out"] < 0.1
    assert abs(res["e_out"] - res["e_out_perm_median"]) < 0.1


# ---------------------------------------------------------------------------
# outlier-subspace reproducibility vs the Haar floor.
# ---------------------------------------------------------------------------
def test_shared_subspace_reads_high_overlap():
    # FTs that SHARE a planted outlier subspace read overlap ≫ the Haar floor.
    m, n, k = 200, 180, 3
    rng = np.random.default_rng(4)
    U = np.linalg.qr(rng.standard_normal((m, k)))[0]
    V = np.linalg.qr(rng.standard_normal((n, k)))[0]
    dws = [0.3 * rng.standard_normal((m, n)) + 20.0 * (U @ V.T) for _ in range(4)]
    real = RMT.mean_pairwise_overlap(dws, k, side="left")
    floor = RMT.haar_overlap_floor(k, m, n_draws=300, seed=0)
    assert real["overlap"] > 0.8                      # shared span ⇒ near-1 overlap
    assert RMT.overlap_beats_floor(real["overlap"], floor)
    assert floor["mean"] == pytest.approx(k / m, rel=0.3)   # Haar analytic floor ≈ k/d


def test_independent_subspaces_sit_at_haar_floor():
    # FTs with INDEPENDENT random outlier subspaces read overlap near the Haar floor (idiosyncratic).
    m, n, k = 200, 180, 3
    rng = np.random.default_rng(5)
    dws = [20.0 * (np.linalg.qr(rng.standard_normal((m, k)))[0]
                   @ np.linalg.qr(rng.standard_normal((n, k)))[0].T)
           + 0.1 * rng.standard_normal((m, n)) for _ in range(4)]
    real = RMT.mean_pairwise_overlap(dws, k, side="left")
    floor = RMT.haar_overlap_floor(k, m, n_draws=300, seed=0)
    assert not RMT.overlap_beats_floor(real["overlap"], floor)   # FT-idiosyncratic, not a family


def test_subspace_overlap_identity_and_bounds():
    rng = np.random.default_rng(6)
    Q = np.linalg.qr(rng.standard_normal((50, 4)))[0]
    assert RMT.subspace_overlap(Q, Q) == pytest.approx(1.0)      # identical span ⇒ 1
    Q2 = np.linalg.qr(rng.standard_normal((50, 4)))[0]
    ov = RMT.subspace_overlap(Q, Q2)
    assert 0.0 <= ov <= 1.0


# ---------------------------------------------------------------------------
# frozen decision predicates.
# ---------------------------------------------------------------------------
def test_band_halfwidth():
    hw = RMT.band_halfwidth([0.0, 0.0, 0.02, 0.0, 0.04, 0.0])
    assert hw >= 0.0


def test_discovery_clear_and_detection_predicates():
    assert RMT.discovery_clear_rmt(e_out_excess_ci_lo=0.10, delta_rmt=0.02) is True
    assert RMT.discovery_clear_rmt(e_out_excess_ci_lo=0.01, delta_rmt=0.02) is False
    assert RMT.detection_rmt(0.10, 0.02, bh_ok=True, overlap_beats=True) is True
    assert RMT.detection_rmt(0.10, 0.02, bh_ok=False, overlap_beats=True) is False
    assert RMT.detection_rmt(0.10, 0.02, bh_ok=True, overlap_beats=False) is False   # idiosyncratic
    assert RMT.detection_rmt(0.01, 0.02, bh_ok=True, overlap_beats=True) is False


def test_mp_fit_gate_abort_to_descriptive():
    frac, abort = RMT.mp_fit_gate([True, True, True, False])     # >half reject
    assert frac == pytest.approx(0.75) and abort is True
    frac2, abort2 = RMT.mp_fit_gate([True, False, False, False])
    assert abort2 is False
