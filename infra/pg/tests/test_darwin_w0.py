"""Weight-free unit tests for the DARWIN Wave-0 instruments (§9 step 2).

Run: PYTHONPATH=. python3 infra/pg/tests/test_darwin_w0.py

Every statistic is checked against a case whose answer is known independently of the
implementation — a closed form, a brute-force enumeration, or a construction with a forced
answer. A test that only re-executes the implementation proves nothing.
"""
import itertools
import math

import numpy as np

from infra.pg import darwin_w0 as W
from infra.pg import humboldt_core as H


# ---------------------------------------------------------------------------
# Coordinate sets.
# ---------------------------------------------------------------------------
def test_threshold_uses_median_so_a_heavy_tail_cannot_hide_itself():
    v = np.concatenate([np.ones(1000), np.full(10, 100.0)])       # median 1, mean ~2
    r = W.threshold_coords(v, 5.0)
    assert r["n_selected"] == 10 and r["median"] == 1.0 and r["threshold"] == 5.0
    assert sorted(r["coords"]) == list(range(1000, 1010))
    # The same vector scored against its MEAN would still catch these, so make the failure mode
    # explicit: 400 large values drag the mean past them, but not the median.
    v2 = np.concatenate([np.ones(1000), np.full(400, 20.0)])
    assert W.threshold_coords(v2, 5.0)["n_selected"] == 400
    assert float(v2.mean()) * 5.0 > 20.0                          # a mean rule would select ZERO


def test_top_k_is_deterministic_under_ties():
    v = np.array([1.0, 3.0, 3.0, 3.0, 0.5])
    assert W.top_k_coords(v, 3) == [1, 2, 3]                      # ties broken by index, ascending
    assert W.top_k_coords(v, 1) == [1]


def test_jaccard_endpoints():
    assert W.jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert W.jaccard([1, 2], [3, 4]) == 0.0
    assert W.jaccard([1, 2, 3, 4], [3, 4, 5, 6]) == 2 / 6
    assert math.isnan(W.jaccard([], []))


def test_log_hypergeom_matches_brute_force_enumeration():
    """The tail is checked against an exact rational enumeration, not against itself."""
    N, K, n = 20, 7, 6
    def exact_sf(k):
        tot = math.comb(N, n)
        return sum(math.comb(K, i) * math.comb(N - K, n - i)
                   for i in range(k, min(K, n) + 1)) / tot
    for k in range(0, min(K, n) + 1):
        got = math.exp(W.log_hypergeom_sf(k, N, K, n))
        assert abs(got - exact_sf(k)) < 1e-12, (k, got, exact_sf(k))
    assert abs(math.exp(W.log_hypergeom_sf(0, N, K, n)) - 1.0) < 1e-12   # P[X>=0] = 1


def test_log_hypergeom_survives_the_deep_tail():
    """A 2048-dim space with a 60/60 overlap of 55 is the regime W0-1 actually reports."""
    lp = W.log_hypergeom_sf(55, 2048, 60, 60)
    assert lp < -100 and math.isfinite(lp)                        # underflows exp(), not lgamma


def test_overlap_stats_enrichment_is_one_for_independent_lists():
    N = 4096
    rng = np.random.default_rng(0)
    a = rng.choice(N, 200, replace=False)
    b = rng.choice(N, 200, replace=False)
    s = W.overlap_stats(a, b, N)
    assert abs(s["expected_intersection"] - 200 * 200 / N) < 1e-9
    assert 0.2 < s["enrichment"] < 3.0                            # independent draws sit near 1
    perfect = W.overlap_stats(a, a, N)
    assert perfect["jaccard"] == 1.0 and perfect["enrichment"] > 10


# ---------------------------------------------------------------------------
# W0-2 cross-layer agreement.
# ---------------------------------------------------------------------------
def test_shared_fraction_is_normalised_by_the_smaller_list():
    assert W.shared_coordinate_fraction([1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]) == 1.0
    assert W.jaccard([1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]) == 3 / 8   # why Jaccard is wrong here
    assert W.shared_coordinate_fraction([1, 2], [3, 4]) == 0.0
    assert math.isnan(W.shared_coordinate_fraction([], [1]))


def test_subspace_agreement_endpoints():
    U = H.haar_subspace(128, 8, 0)
    same = W.subspace_agreement(U, U)
    assert abs(same["mean_cos2"] - 1.0) < 1e-12
    orth = W.subspace_agreement(U, H.complement_subspace(U, 1))
    assert orth["mean_cos2"] < 1e-12
    assert abs(orth["min_angle_rad"] - np.pi / 2) < 1e-7
    # Independent Haar subspaces sit at k/n, the reference the arm compares against.
    ind = W.subspace_agreement(H.haar_subspace(512, 16, 2), H.haar_subspace(512, 16, 3))
    assert abs(ind["mean_cos2"] - 16 / 512) < 0.02


def test_w1r_routing_has_both_branches():
    assert W.w1r_routing(0.80)["variant"] == "global"
    assert W.w1r_routing(0.50)["variant"] == "global"             # boundary is inclusive
    assert W.w1r_routing(0.49)["variant"] == "per_layer"
    assert W.w1r_routing(0.0)["variant"] == "per_layer"


# ---------------------------------------------------------------------------
# W0-0 bank identity.
# ---------------------------------------------------------------------------
def test_bank_identity_tolerates_sign_but_reports_a_real_difference():
    U = H.haar_subspace(64, 8, 5)
    flipped = U * np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    r = W.bank_identity(flipped, U)
    assert r["max_abs_diff_sign_aligned"] < 1e-15                 # sign is not a difference
    assert r["max_principal_angle_rad"] < 1e-7
    assert not r["bit_identical"]                                 # recorded, and correctly false
    # A genuinely different subspace must be caught by the GATED number.
    other = H.haar_subspace(64, 8, 6)
    assert W.bank_identity(other, U)["max_principal_angle_rad"] > 0.5


def test_bank_identity_passes_the_subspace_gate_under_within_subspace_rotation():
    """The reason the gate is on angles, not elements: a rotated basis of the SAME subspace is an
    equally correct derivation, and LAPACK may return either."""
    U = H.haar_subspace(64, 8, 7)
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((8, 8)))
    r = W.bank_identity(U @ Q, U)
    # SUBSPACE_ANGLE_TOL, not something tighter: cos θ = 1 - O(eps) and arccos loses half the
    # mantissa there, so two bit-for-bit-equivalent subspaces still report ~1.5e-8 rad. A 1e-10
    # gate would fail for float reasons alone. `mean_cos2` is the numerically stable companion.
    assert r["max_principal_angle_rad"] < W.SUBSPACE_ANGLE_TOL
    assert 1.0 - r["mean_cos2"] < 1e-14
    assert r["max_abs_diff_sign_aligned"] > 0.1                   # ...while elements differ a lot


def test_subspace_angle_tolerance_is_above_the_arccos_noise_floor():
    """The tolerance is a property of arccos, so pin it: identical subspaces must clear it with
    room to spare, and a genuinely different one must not come close."""
    worst = 0.0
    for s in range(12):
        U = H.haar_subspace(256, 32, s)
        worst = max(worst, W.bank_identity(U.copy(), U)["max_principal_angle_rad"])
    assert worst < W.SUBSPACE_ANGLE_TOL / 10, worst              # >= 10x headroom on a true match
    U = H.haar_subspace(256, 32, 0)
    assert W.bank_identity(H.haar_subspace(256, 32, 1), U)["max_principal_angle_rad"] > 1.0


def test_bank_identity_bit_identical_on_an_exact_copy():
    U = H.haar_subspace(32, 4, 9)
    assert W.bank_identity(U.copy(), U)["bit_identical"]


# ---------------------------------------------------------------------------
# W0-7 OOD score.
# ---------------------------------------------------------------------------
def test_band_deviation_is_zero_inside_and_distance_outside():
    d = W.band_deviation([0.0, 0.5, 1.0, 2.0, -1.0], lo=0.0, hi=1.0)
    assert list(d) == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_roc_auc_endpoints_and_tie_handling():
    assert abs(W.roc_auc([1, 2, 3], [-1, -2, -3])["auc"] - 1.0) < 1e-12       # perfect
    assert abs(W.roc_auc([-1, -2, -3], [1, 2, 3])["auc"] - 0.0) < 1e-12       # perfectly inverted
    tied = W.roc_auc([1, 1, 1], [1, 1, 1])
    assert abs(tied["auc"] - 0.5) < 1e-12 and tied["tie_fraction"] > 0.8      # all ties ⇒ 0.5
    # Against an independent implementation: AUC = P(pos > neg) + 0.5 P(pos == neg).
    rng = np.random.default_rng(1)
    p, n = rng.normal(0.5, 1, 200), rng.normal(0.0, 1, 300)
    brute = np.mean([(1.0 if x > y else 0.5 if x == y else 0.0)
                     for x, y in itertools.product(p, n)])
    assert abs(W.roc_auc(p, n)["auc"] - brute) < 1e-12


# ---------------------------------------------------------------------------
# W0-8 compression frontier.
# ---------------------------------------------------------------------------
def test_energy_captured_endpoints():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((16, 64))
    full = np.eye(64)
    assert abs(W.energy_captured(A, full) - 1.0) < 1e-12          # a complete basis captures all
    U = H.haar_subspace(64, 8, 11)
    e = W.energy_captured(A, U)
    assert 0.0 < e < 1.0 and abs(e - 8 / 64) < 0.12               # Haar reference is k/n
    # A basis aligned to the delta's own top singular directions must beat a random one.
    V = np.linalg.svd(A, full_matrices=False)[2].T[:, :8]
    assert W.energy_captured(A, V) > e


def test_rank_ladder_is_the_spec_ladder():
    assert W.rank_ladder() == [4, 8, 16, 32, 64, 128, 256]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
