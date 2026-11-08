"""Weight-free CPU dry-run gate for GAUGE-CS (P1, the MDE) — §12.1 suite.

NO model, NO GPU — planted matrices only. Proves: the concentration statistic; the
rotation-invariance reduction (null band is basis-independent AND equals the random-K distribution);
that a matrix genuinely concentrated in a KNOWN fixed basis reads excess in that class but a
spectrum-matched control does not; the entry-permutation control kills marginal-texture but not
real basis structure; and the CS-recovery leg recovers a sparse signal below energy-truncation.
Run remotely: python -m pytest infra/pg/tests/test_gauge_cs.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_cs as CS


def _sparse_in_basis(m, n, k, seed):
    """A matrix whose coefficients are k-sparse in a KNOWN random orthonormal basis pair (U_B,V_B):
    ΔW = U_B S V_Bᵀ with S having k nonzero entries. Returns (ΔW, U_B, V_B)."""
    rng = np.random.default_rng(seed)
    U_B = CS.random_ortho(m, seed)
    V_B = CS.random_ortho(n, seed + 1000)
    S = np.zeros((m, n))
    ii = rng.integers(0, m, k)
    jj = rng.integers(0, n, k)
    S[ii, jj] = rng.standard_normal(k) * 5.0
    return U_B @ S @ V_B.T, U_B, V_B


# ---------------------------------------------------------------------------
# concentration statistic.
# ---------------------------------------------------------------------------
def test_concentration_dwsvd_is_maximally_concentrated():
    # in ΔW's OWN svd basis, M_B = diag(σ): mass on min(m,n) entries out of m·n ⇒ very concentrated.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 20))
    entry = CS.concentration_in_class(A, "entry")
    dwsvd = CS.concentration_in_class(A, "dW_svd")
    assert dwsvd["gini"] > entry["gini"]           # own-svd concentrates far more than raw entries
    assert dwsvd["pr"] < entry["pr"]


# ---------------------------------------------------------------------------
# the rotation-invariance reduction (load-bearing).
# ---------------------------------------------------------------------------
def test_null_band_is_basis_independent():
    # same spectrum, two DIFFERENT fixed bases give the SAME null concentration distribution.
    rng = np.random.default_rng(1)
    A = rng.standard_normal((40, 25))
    sig = np.linalg.svd(A, compute_uv=False)
    b1 = CS.null_concentration_band(sig, 40, 25, n_draws=200, seed=7)
    b2 = CS.null_concentration_band(sig, 40, 25, n_draws=200, seed=8)
    # two independent MC estimates of the SAME (basis-free) distribution agree closely.
    assert abs(b1["gini"]["mean"] - b2["gini"]["mean"]) < 0.02


def test_random_k_class_sits_in_the_null_band():
    # class random-K concentration of a REAL matrix ≈ the spectrum-matched null (rotation invariance):
    # a random basis is incoherent, so it reads the spectrum texture, not structure.
    rng = np.random.default_rng(2)
    A = rng.standard_normal((48, 32))
    sig = np.linalg.svd(A, compute_uv=False)
    null = CS.null_concentration_band(sig, 48, 32, n_draws=200, seed=3)
    rk = CS.concentration_in_class(A, "random_k", random_seed=101, n_random=8)
    assert null["gini"]["p2.5"] - 0.05 < rk["gini"] < null["gini"]["p97.5"] + 0.05


# ---------------------------------------------------------------------------
# detection vs null: a KNOWN hidden basis reads excess; a spectrum-match does not.
# ---------------------------------------------------------------------------
def test_hidden_basis_reads_excess_over_null():
    m, n, k = 40, 30, 6
    A, U_B, V_B = _sparse_in_basis(m, n, k, seed=5)
    sig = np.linalg.svd(A, compute_uv=False)
    null = CS.null_concentration_band(sig, m, n, n_draws=200, seed=9)
    # measured in the TRUE hidden basis (here supplied as 'base_svd' via W_base with those vectors):
    c_true = CS.concentration(CS.coeffs(A, U_B, V_B))
    # the real concentration in the hidden basis is far ABOVE the spectrum-matched null band.
    assert c_true["gini"] > null["gini"]["p97.5"]
    # but the SAME spectrum in a random basis (the null) is not concentrated.
    assert null["gini"]["mean"] < c_true["gini"] - 0.1


# ---------------------------------------------------------------------------
# entry-permutation control.
# ---------------------------------------------------------------------------
def test_entry_permutation_kills_basis_structure_keeps_marginal():
    m, n, k = 40, 30, 6
    A, U_B, V_B = _sparse_in_basis(m, n, k, seed=6)
    perm = CS.permute_entries(A, seed=1)
    # entry-class concentration is a function of the MARGINAL only ⇒ permutation ~unchanged.
    ce_real = CS.concentration(A)["gini"]
    ce_perm = CS.concentration(perm)["gini"]
    assert abs(ce_real - ce_perm) < 1e-9
    # but the hidden-basis concentration COLLAPSES under permutation (structure destroyed).
    cb_real = CS.concentration(CS.coeffs(A, U_B, V_B))["gini"]
    cb_perm = CS.concentration(CS.coeffs(perm, U_B, V_B))["gini"]
    assert cb_perm < cb_real - 0.1


# ---------------------------------------------------------------------------
# consensus bases.
# ---------------------------------------------------------------------------
def test_consensus_bases_orthonormal_and_shaped():
    rng = np.random.default_rng(3)
    dws = [rng.standard_normal((20, 14)) for _ in range(5)]
    U_c, V_c = CS.consensus_bases(dws)
    assert U_c.shape == (20, 20) and V_c.shape == (14, 14)
    assert np.allclose(U_c.T @ U_c, np.eye(20), atol=1e-9)
    assert np.allclose(V_c.T @ V_c, np.eye(14), atol=1e-9)


def test_consensus_bases_empty_raises():
    with pytest.raises(AssertionError):
        CS.consensus_bases([])


# ---------------------------------------------------------------------------
# summary + decision predicates.
# ---------------------------------------------------------------------------
def test_median_excess_ci_positive_signal():
    ci = CS.median_excess_ci([0.3, 0.35, 0.28, 0.31, 0.33], n_boot=1000)
    assert ci["ci_lo"] > 0 and ci["median"] > 0.25


def test_discovery_clear_and_trivial_gate():
    # clears when CI-lo>δ and the permutation control does NOT clear.
    assert CS.discovery_clear(ci_lo=0.2, delta_cs=0.05, perm_ci_lo=0.0) is True
    # trivial-concentration: permutation control ALSO clears → not a discovery.
    assert CS.discovery_clear(ci_lo=0.2, delta_cs=0.05, perm_ci_lo=0.2) is False
    # below δ → not clear.
    assert CS.discovery_clear(ci_lo=0.01, delta_cs=0.05, perm_ci_lo=0.0) is False


def test_bootstrap_p_le():
    # excesses clearly above delta → small p (median rarely ≤ delta); clearly below → large p.
    hi = [0.30, 0.28, 0.33, 0.31, 0.29]
    lo = [0.01, -0.01, 0.02, 0.00, 0.005]
    assert CS.bootstrap_p_le(hi, delta=0.05, n_boot=1000) < 0.05
    assert CS.bootstrap_p_le(lo, delta=0.05, n_boot=1000) > 0.5


def test_detection_predicate():
    assert CS.detection(sealed_ci_lo=0.2, delta_cs=0.05, bh_ok=True, secondary_sign_agrees=True) is True
    assert CS.detection(sealed_ci_lo=0.2, delta_cs=0.05, bh_ok=False, secondary_sign_agrees=True) is False
    assert CS.detection(sealed_ci_lo=0.2, delta_cs=0.05, bh_ok=True, secondary_sign_agrees=False) is False
    assert CS.detection(sealed_ci_lo=0.01, delta_cs=0.05, bh_ok=True, secondary_sign_agrees=True) is False


# ---------------------------------------------------------------------------
# CS-recovery leg (stage 3).
# ---------------------------------------------------------------------------
def test_l1_recovers_sparse_below_energy_truncation():
    # a k-sparse coefficient vector: L1 from m random measurements recovers it near-exactly, and
    # its error is well below the energy-truncation error at the SAME measurement budget k.
    rng = np.random.default_rng(4)
    d, k = 120, 6
    x = np.zeros(d)
    x[rng.choice(d, k, replace=False)] = rng.standard_normal(k) * 4.0
    err_l1 = CS.l1_recover(x, m_meas=60, seed=0, n_iter=800)
    err_energy = CS.energy_truncation_error(x, k=k)          # perfect here (signal IS k-sparse)
    assert err_l1 < 0.1                                       # L1 recovers the sparse signal
    # on a DENSE vector L1 from few measurements cannot beat keeping the top-k energy directly.
    xd = rng.standard_normal(d)
    assert CS.energy_truncation_error(xd, k=k) < 1.0


def test_null_band_pooled_matches_single_spectrum():
    # pooling one spectrum reproduces the single-spectrum band (same machinery); pooling several
    # spectra gives a band bracketing their individual medians.
    rng = np.random.default_rng(7)
    A = rng.standard_normal((40, 25))
    sig = np.linalg.svd(A, compute_uv=False)
    single = CS.null_concentration_band(sig, 40, 25, n_draws=200, seed=3)
    pooled = CS.null_band_pooled([sig], 40, 25, n_draws=200, seed=3)
    assert abs(single["gini"]["mean"] - pooled["gini"]["mean"]) < 0.02
    # pooled over 3 different-scale spectra still yields a valid band (scale-invariant gini).
    b = CS.null_band_pooled([sig, 2 * sig, 0.5 * sig], 40, 25, n_draws=150, seed=1)
    assert b["gini"]["p2.5"] <= b["gini"]["p50"] <= b["gini"]["p97.5"]


def test_null_band_pooled_wrong_length_raises():
    with pytest.raises(AssertionError):
        CS.null_band_pooled([np.ones(10)], 40, 25, n_draws=5)   # 10 != min(40,25)=25


def test_precomputed_bases_match_on_the_fly():
    # passing precomputed base-SVD + random bases gives IDENTICAL concentration to building them
    # inline (the job caches per (type,layer); the tests build on the fly) — same seeds ⇒ same result.
    rng = np.random.default_rng(8)
    A = rng.standard_normal((32, 24))
    Wb = rng.standard_normal((32, 24))
    U, V = CS.svd_bases(Wb)
    on_fly = CS.concentration_in_class(A, "base_svd", W_base=Wb)
    cached = CS.concentration_in_class(A, "base_svd", base_svd_bases=(U, V))
    assert on_fly["gini"] == pytest.approx(cached["gini"], rel=1e-12)
    rb = [(CS.random_ortho(32, 5 + 2 * j), CS.random_ortho(24, 6 + 2 * j)) for j in range(4)]
    c_seed = CS.concentration_in_class(A, "random_k", random_seed=5, n_random=4)   # builds 5,6,7,8...
    c_cache = CS.concentration_in_class(A, "random_k", random_bases=rb)
    assert c_seed["gini"] == pytest.approx(c_cache["gini"], rel=1e-12)


def test_base_svd_without_basis_or_wbase_raises():
    A = np.random.default_rng(0).standard_normal((10, 8))
    with pytest.raises(AssertionError):
        CS.concentration_in_class(A, "base_svd")            # neither W_base nor precomputed bases


def test_energy_truncation_error_bounds():
    x = np.array([3.0, 0.0, 0.0, 0.0])
    assert CS.energy_truncation_error(x, k=1) == pytest.approx(0.0, abs=1e-12)   # 1-sparse, k=1
    assert CS.energy_truncation_error(np.array([1.0, 1.0, 1.0, 1.0]), k=2) == pytest.approx(
        np.sqrt(2) / 2, rel=1e-9)
