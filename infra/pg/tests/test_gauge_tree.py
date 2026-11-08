"""Weight-free CPU dry-run gate for GAUGE-TREE (P4) — §12.1 suite ("CCC + triple statistic +
null-cloud generator").

NO model, NO GPU — planted distance matrices / Grams only. Proves: the Gram→distance construction
identities; average-linkage cophenetic distances + CCC (=1 on a perfect ultrametric, <1 on a
random cloud); the triple statistic (=0 on an ultrametric, >0 off it); the matched-cloud dimension
recovery + null generator; the label-permutation null centred below a real tree; the δ_UM /
detection decision machinery; the largest-cluster testable/UNTESTABLE rule; and the chordal-distance
read-family helper. Run remotely: python -m pytest infra/pg/tests/test_gauge_tree.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_tree as TREE


def _gram_from_points(X):
    """Raw inner-product Gram of row-vector points (Gᵢⱼ=⟨xᵢ,xⱼ⟩) — the object the streaming job builds."""
    X = np.asarray(X, dtype=np.float64)
    return X @ X.T


# ---------------------------------------------------------------------------
# Gram → per-FT-normalised Frobenius distance.
# ---------------------------------------------------------------------------
def test_gram_to_distance_identities():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 10))
    D = TREE.gram_to_distance(_gram_from_points(X))
    assert np.allclose(np.diag(D), 0.0)                      # self-distance 0
    assert np.allclose(D, D.T)                               # symmetric
    # matches the direct normalised-ΔW Frobenius distance.
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Ddirect = np.sqrt(((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(-1))
    assert np.allclose(D, Ddirect, atol=1e-9)


def test_gram_to_distance_orthogonal_is_sqrt2_and_parallel_is_zero():
    # two orthogonal unit ΔW ⇒ d=sqrt(2); a scaled copy (cos=1) ⇒ d=0 (scale-invariant by design).
    G = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [2.0, 0.0, 4.0]])   # row0 ⟂ row1; row2 = 2·row0
    D = TREE.gram_to_distance(G)
    assert D[0, 1] == pytest.approx(np.sqrt(2.0), rel=1e-9)
    assert D[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_gram_to_distance_zero_norm_raises():
    with pytest.raises(AssertionError):
        TREE.gram_to_distance(np.array([[0.0, 0.0], [0.0, 1.0]]))


# ---------------------------------------------------------------------------
# average-linkage cophenetic + CCC.
# ---------------------------------------------------------------------------
def test_cophenetic_is_ultrametric_fixed_point_and_ccc_one():
    # the cophenetic matrix of ANY D is ULTRAMETRIC ⇒ average-linkage reproduces it ⇒ CCC(U)=1.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((9, 5))
    D = TREE.gram_to_distance(_gram_from_points(X))
    U = TREE.cophenetic_matrix(D)
    assert np.allclose(U, TREE.cophenetic_matrix(U), atol=1e-9)   # ultrametric fixed point
    assert TREE.ccc(U) == pytest.approx(1.0, abs=1e-9)


def test_ccc_below_one_for_generic_cloud():
    rng = np.random.default_rng(2)
    D = TREE.gram_to_distance(_gram_from_points(rng.standard_normal((12, 8))))
    c = TREE.ccc(D)
    assert -1.0 <= c <= 1.0 and c < 0.999                    # a real cloud is not a perfect tree


def test_cophenetic_matches_hand_computed_upgma():
    # 3 leaves: closest pair merges first at their distance; the third joins at the average.
    D = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 5.0], [4.0, 5.0, 0.0]])
    U = TREE.cophenetic_matrix(D)
    assert U[0, 1] == pytest.approx(1.0)                     # closest pair height
    assert U[0, 2] == pytest.approx(4.5) and U[1, 2] == pytest.approx(4.5)   # (4+5)/2 average-linkage


# ---------------------------------------------------------------------------
# triple statistic.
# ---------------------------------------------------------------------------
def test_triple_statistic_zero_on_ultrametric():
    rng = np.random.default_rng(3)
    U = TREE.cophenetic_matrix(TREE.gram_to_distance(_gram_from_points(rng.standard_normal((8, 6)))))
    assert TREE.triple_statistic(U) == pytest.approx(0.0, abs=1e-9)


def test_triple_statistic_positive_off_tree():
    rng = np.random.default_rng(4)
    D = TREE.gram_to_distance(_gram_from_points(rng.standard_normal((10, 7))))
    assert TREE.triple_statistic(D) > 1e-3                   # a generic cloud violates ultrametricity


# ---------------------------------------------------------------------------
# matched-Gaussian-cloud null.
# ---------------------------------------------------------------------------
def test_matched_cloud_dim_recovers_scale():
    # points drawn in dimension p ⇒ the matched dimension is in the right ballpark (CV-based).
    rng = np.random.default_rng(5)
    for p in (20, 80):
        X = rng.standard_normal((24, p))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        est = TREE.matched_cloud_dim(D)
        assert 0.4 * p <= est <= 2.5 * p                     # generous CV-inversion band


def test_matched_cloud_null_shape_and_bounds():
    rng = np.random.default_rng(6)
    D = TREE.gram_to_distance(_gram_from_points(rng.standard_normal((14, 9))))
    null = TREE.matched_cloud_null(D, n_draws=120, seed=7)
    assert null.shape == (120,) and np.all(null <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# label-permutation (Mantel) null — a real tree beats its own relabelling.
# ---------------------------------------------------------------------------
def test_label_permutation_null_below_real_tree():
    rng = np.random.default_rng(7)
    # a strongly tree-structured family: near-ultrametric distances.
    U = TREE.cophenetic_matrix(TREE.gram_to_distance(_gram_from_points(rng.standard_normal((12, 6)))))
    real = TREE.ccc(U)                                       # = 1.0 for the ultrametric
    perm = TREE.label_permutation_null(U, n_draws=200, seed=3)
    assert real > np.percentile(perm, 95)                    # relabelling destroys the correspondence


# ---------------------------------------------------------------------------
# δ_UM + detection machinery.
# ---------------------------------------------------------------------------
def test_ccc_with_nulls_record_keys_and_flags():
    rng = np.random.default_rng(8)
    D = TREE.gram_to_distance(_gram_from_points(rng.standard_normal((16, 10))))
    res = TREE.ccc_with_nulls(D, n_null=100, seed=1)
    for k in ("ccc", "triple_statistic", "cloud_delta_um", "perm_delta_um",
              "beats_cloud", "beats_perm", "matched_dim"):
        assert k in res
    assert res["beats_cloud"] == (res["ccc"] > res["cloud_delta_um"])


def test_delta_um_is_95th_percentile():
    x = np.linspace(0.0, 1.0, 101)
    assert TREE.delta_um(x) == pytest.approx(0.95, abs=1e-9)


def test_detection_requires_both_nulls_and_within_cluster():
    win = {"ccc": 0.9, "beats_cloud": True, "beats_perm": True}
    assert TREE.beats_both(win) is True
    assert TREE.detection(win, within_cluster_excess=True) is True
    assert TREE.detection(win, within_cluster_excess=False) is False    # cluster-trivial
    assert TREE.detection(win, within_cluster_excess=None) is False     # UNTESTABLE ⇒ uncertified
    lose = {"ccc": 0.1, "beats_cloud": True, "beats_perm": False}
    assert TREE.detection(lose, within_cluster_excess=True) is False    # must beat BOTH nulls


# ---------------------------------------------------------------------------
# largest-cluster control (testable / UNTESTABLE frozen rule).
# ---------------------------------------------------------------------------
def test_largest_cluster_testable_and_untestable():
    big = TREE.largest_cluster(["near"] * 10 + ["far"] * 3, min_size=8)
    assert big["label"] == "near" and big["size"] == 10 and big["testable"] is True
    assert big["indices"] == list(range(10))
    small = TREE.largest_cluster(["a", "a", "a", "b", "b"], min_size=8)
    assert small["testable"] is False                        # UNTESTABLE, not skipped


def test_largest_cluster_empty_raises():
    with pytest.raises(AssertionError):
        TREE.largest_cluster([])


# ---------------------------------------------------------------------------
# read-family chordal distance (exploratory rows).
# ---------------------------------------------------------------------------
def test_chordal_distance_identities():
    rng = np.random.default_rng(9)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 3)))        # 10×3 orthonormal basis
    assert TREE.chordal_distance(Q, Q) == pytest.approx(0.0, abs=1e-9)   # same subspace ⇒ 0
    # an orthogonal-complement basis is maximally far: d² = k.
    Qc, _ = np.linalg.qr(rng.standard_normal((10, 3)))
    Qc = Qc - Q @ (Q.T @ Qc)
    Qc, _ = np.linalg.qr(Qc)
    assert TREE.chordal_distance(Q, Qc[:, :3]) == pytest.approx(np.sqrt(3.0), abs=1e-6)


def test_chordal_distance_matrix_symmetric_zero_diag():
    rng = np.random.default_rng(10)
    bases = [np.linalg.qr(rng.standard_normal((8, 2)))[0] for _ in range(4)]
    D = TREE.chordal_distance_matrix(bases)
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0.0)
