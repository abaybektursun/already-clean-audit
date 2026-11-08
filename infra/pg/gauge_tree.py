"""GAUGE-TREE (P4) — ultrametricity / hierarchical geometry of the adaptation family.

Question (§7). Spin-glass theory predicts frustrated systems organise their states in ULTRAMETRIC
(tree-like) families, not in a basis. Is the 24-FT PG1 ΔW family tree-organised beyond matched-
geometry nulls? The distance object is the per-FT-normalised Frobenius distance between fine-tunes:

    d(i,j) = ‖ ΔW_i/‖ΔW_i‖_F  −  ΔW_j/‖ΔW_j‖_F ‖_F   =   sqrt( 2 (1 − cos_ij) ),
    cos_ij = ⟨ΔW_i, ΔW_j⟩ / (‖ΔW_i‖_F ‖ΔW_j‖_F).

So the WHOLE distance matrix is a function of the 24×24 raw inner-product Gram G (G_ij = ⟨ΔW_i,ΔW_j⟩,
G_ii = ‖ΔW_i‖²_F) — exactly the "cached per-pair inner products" the streaming job accumulates one
matrix at a time (pooled = all 126 matrices; per-matrix-type = that type's matrices). `gram_to_distance`
turns that Gram into D; everything downstream is pure geometry.

Pure, weight-free, fp64, NO try/except (house style: gauge_cs.py, invariant_core.py). Heavy imports
inside functions; JSON-able returns.

DESIGN NOTES (frozen at authoring, §7.2).
* PRIMARY statistic = cophenetic correlation coefficient (CCC) under AVERAGE-LINKAGE (UPGMA):
  Pearson corr between the condensed pairwise distances and the condensed cophenetic distances
  (the merge height at which two leaves first join). Implemented in ~40 lines of pure numpy —
  scipy.cluster.hierarchy is NOT used anywhere in this repo (grepped at Gate-0), so we keep the
  cluster core self-contained rather than add the dependency.
* CCC is affine-invariant in D (average-linkage merge heights are convex means of pairwise
  distances ⇒ an affine map of D induces the same affine map of the cophenetic matrix ⇒ Pearson
  corr is unchanged). Consequence: the matched-Gaussian-cloud null only needs to match the SHAPE
  (coefficient of variation) of the distance distribution — set by the cloud DIMENSION — not the
  mean/variance, which an affine rescale (the spec's literal "scale to match mean/variance") maps
  onto the observed values without touching CCC. `matched_cloud_dim` picks that dimension.
* SECONDARY (sign-agreement, "robust"): the triple statistic = median over all triples of
  (d_max − d_mid)/d_min. Exactly 0 on a perfect ultrametric (largest two triangle sides equal);
  larger ⇒ less tree-like.
* Two nulls, ≥100 draws each: (i) matched-Gaussian-cloud — n iid Gaussian points in the matched
  dimension, its OWN CCC (calibrates "UPGMA imposes a tree on anything"); (ii) label-permutation —
  Mantel-style, corr between a leaf-relabelled D and the ORIGINAL cophenetic matrix (destroys the
  D↔tree correspondence, preserves both marginals).
* Frozen rule: δ_UM = 95th percentile of a null CCC distribution. Detection-grade = real CCC > δ_UM
  under BOTH nulls AND the excess persists within the largest task-type cluster (else H_flat-family
  or H_task-cluster-trivial). n=24 single-shot; disclosed exploratory-strong at best.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# distance construction from the cached inner-product Gram.
# ---------------------------------------------------------------------------
def gram_to_distance(G):
    """Per-FT-normalised Frobenius distance matrix from the raw inner-product Gram G (n×n, symmetric,
    G_ij=⟨ΔW_i,ΔW_j⟩, G_ii=‖ΔW_i‖²_F). D_ij = sqrt(2(1−cos_ij)), cos_ij = G_ij/sqrt(G_ii G_jj).
    Cosines are clipped to [−1,1] (fp roundoff only) before the sqrt; the diagonal is forced to 0.
    RAISES on a non-positive diagonal (a zero/degenerate ΔW that slipped the FT-scale gate). fp64."""
    import numpy as np
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    assert G.shape == (n, n), f"Gram must be square, got {G.shape}"
    d = np.diag(G).copy()
    assert (d > 0).all(), "non-positive Gram diagonal (zero-norm ΔW — degenerate FT)"
    norms = np.sqrt(d)
    C = G / np.outer(norms, norms)
    C = np.clip(C, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)          # symmetrise away fp asymmetry


# ---------------------------------------------------------------------------
# condensed (upper-triangle) helpers.
# ---------------------------------------------------------------------------
def _condensed(M):
    """The strict upper-triangle of a symmetric matrix as a 1-D vector (the n(n−1)/2 pair values)."""
    import numpy as np
    M = np.asarray(M, dtype=np.float64)
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def _pearson(a, b):
    """Pearson correlation of two 1-D vectors. Returns 0.0 if either is constant (an undefined corr
    is reported as 'no linear correspondence', the conservative null value — degenerate only on a
    trivial tree, disclosed by the caller)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# average-linkage (UPGMA) cophenetic distances + CCC — pure numpy, O(n³), n=24 tiny.
# ---------------------------------------------------------------------------
def cophenetic_matrix(D):
    """Average-linkage (UPGMA) cophenetic distance matrix of a symmetric distance matrix D. Repeatedly
    merges the two closest active clusters at height = their current inter-cluster distance, records
    that height as the cophenetic distance between EVERY leaf-pair spanning the merge, then updates
    distances by the size-weighted average d(new,c) = (n_a d(a,c) + n_b d(b,c))/(n_a+n_b). Returns an
    n×n symmetric matrix (0 diagonal). Pure numpy; no scipy. fp64."""
    import numpy as np
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    assert D.shape == (n, n) and n >= 2, f"need a square distance matrix n≥2, got {D.shape}"
    coph = np.zeros((n, n), dtype=np.float64)
    members = {i: [i] for i in range(n)}        # cluster-id -> leaf list
    size = {i: 1 for i in range(n)}
    active = set(range(n))
    dist = {}                                    # {(min,max): distance} over ACTIVE cluster ids
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(D[i, j])
    nxt = n
    while len(active) > 1:
        # closest active pair.
        best, bd = None, float("inf")
        for (x, y), dv in dist.items():
            if x in active and y in active and dv < bd:
                bd, best = dv, (x, y)
        a, b = best
        for u in members[a]:                     # stamp the merge height on all crossing leaf-pairs
            for v in members[b]:
                coph[u, v] = coph[v, u] = bd
        c = nxt
        nxt += 1
        members[c] = members[a] + members[b]
        size[c] = size[a] + size[b]
        active.discard(a)
        active.discard(b)
        na, nb = size[a], size[b]
        for k in active:
            kd = (min(a, k), max(a, k))
            ld = (min(b, k), max(b, k))
            dist[(min(c, k), max(c, k))] = (na * dist[kd] + nb * dist[ld]) / (na + nb)
        active.add(c)
    return coph


def ccc(D):
    """Cophenetic correlation coefficient (PRIMARY, §7.2): Pearson corr between the condensed pairwise
    distances of D and the condensed average-linkage cophenetic distances. In [−1,1]; →1 ⇒ the family
    is well described by a tree. fp64."""
    return _pearson(_condensed(D), _condensed(cophenetic_matrix(D)))


def triple_statistic(D):
    """SECONDARY ultrametricity index (§7.2): median over ALL triples (i<j<k) of (d_max−d_mid)/d_min,
    the three triangle sides sorted. Exactly 0 on a perfect ultrametric (largest two sides equal);
    larger ⇒ farther from tree-like. RAISES if any triple's smallest side is 0 (coincident leaves —
    a degenerate distance matrix). fp64."""
    import numpy as np
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = np.sort([D[i, j], D[i, k], D[j, k]])
                assert s[0] > 0, "triple with a zero side (coincident FTs)"
                vals.append((s[2] - s[1]) / s[0])
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# matched-Gaussian-cloud null.
# ---------------------------------------------------------------------------
def matched_cloud_dim(D):
    """The iid-Gaussian-cloud DIMENSION whose pairwise-distance shape matches D. For n points ~N(0,I_p),
    the distance coefficient of variation CV=std/mean ≈ 1/sqrt(2p); inverting, p ≈ 1/(2 CV²). Since CCC
    is affine-invariant in D, dimension (which fixes CV, the only non-affine feature) is the sole
    load-bearing match; the spec's mean/variance match is an affine rescale that leaves CCC untouched.
    Clamped to ≥2. fp64."""
    import numpy as np
    v = _condensed(D)
    mu = float(v.mean())
    sd = float(v.std())
    assert mu > 0, "degenerate distance matrix (zero mean pair distance)"
    cv = sd / mu
    if cv <= 0:
        return 2
    return int(max(2, round(1.0 / (2.0 * cv * cv))))


def matched_cloud_null(D, n_draws=200, seed=1337, dim=None):
    """Matched-Gaussian-cloud null CCC distribution (§7.2 null i). Each draw: n iid N(0,I_p) points
    (p = matched_cloud_dim(D) unless `dim` overrides), its own Euclidean distance matrix, its own CCC.
    Calibrates the 'UPGMA imposes a tree on anything' floor. Returns a numpy array of `n_draws` CCCs."""
    import numpy as np
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    p = int(dim) if dim is not None else matched_cloud_dim(D)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    for t in range(n_draws):
        X = rng.standard_normal((n, p))
        diff = X[:, None, :] - X[None, :, :]
        Dn = np.sqrt((diff ** 2).sum(-1))
        out[t] = ccc(Dn)
    return out


def label_permutation_null(D, n_draws=200, seed=1337):
    """Label-permutation (Mantel-style) null CCC distribution (§7.2 null ii). Cophenetic matrix is
    computed ONCE from D; each draw relabels the leaves of D and correlates the relabelled condensed
    distances against the ORIGINAL condensed cophenetic distances — destroying the D↔tree
    correspondence while preserving both matrices' marginal distributions. The standard (weak) test:
    the matched cloud is the decisive null, this one guards a pathological D↔coph mismatch. Returns a
    numpy array of `n_draws` CCCs."""
    import numpy as np
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    coph0 = _condensed(cophenetic_matrix(D))
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    for t in range(n_draws):
        pi = rng.permutation(n)
        out[t] = _pearson(_condensed(D[np.ix_(pi, pi)]), coph0)
    return out


# ---------------------------------------------------------------------------
# null summary + the frozen decision machinery (§7.2).
# ---------------------------------------------------------------------------
def _null_stat(samples, pct=95.0):
    """mean/std/percentiles(2.5/50/95/97.5) + raw samples of a null CCC array (JSON-able)."""
    import numpy as np
    a = np.asarray(samples, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size),
            "p2.5": float(np.percentile(a, 2.5)), "p50": float(np.percentile(a, 50.0)),
            "p95": float(np.percentile(a, 95.0)), "p97.5": float(np.percentile(a, 97.5)),
            "delta_um": float(np.percentile(a, pct)), "samples": a.tolist()}


def delta_um(null_cccs, pct=95.0):
    """δ_UM = the `pct`-th percentile (default 95th, frozen) of a null CCC distribution (§7.2)."""
    import numpy as np
    return float(np.percentile(np.asarray(null_cccs, dtype=np.float64), pct))


def ccc_with_nulls(D, n_null=200, seed=1337, dim=None):
    """The full per-distance-matrix read: real CCC + triple statistic + BOTH null CCC distributions
    (matched-cloud, label-permutation) + their δ_UM cuts + the two `beats_*` booleans. Reused for the
    pooled matrix, each per-matrix-type matrix, and the within-cluster sub-matrix. JSON-able."""
    real = ccc(D)
    cloud = matched_cloud_null(D, n_null, seed, dim)
    perm = label_permutation_null(D, n_null, seed + 1)
    cs, ps = _null_stat(cloud), _null_stat(perm)
    return {"ccc": real, "triple_statistic": triple_statistic(D), "n": int(D.shape[0]),
            "matched_dim": matched_cloud_dim(D) if dim is None else int(dim),
            "cloud_null": cs, "perm_null": ps,
            "cloud_delta_um": cs["delta_um"], "perm_delta_um": ps["delta_um"],
            "beats_cloud": bool(real > cs["delta_um"]), "beats_perm": bool(real > ps["delta_um"])}


def beats_both(res):
    """Whether a `ccc_with_nulls` record clears δ_UM under BOTH nulls (the excess-tree-likeness gate)."""
    return bool(res["beats_cloud"] and res["beats_perm"])


def detection(full_res, within_cluster_excess):
    """§7.2 Detection-grade predicate: real CCC > δ_UM under BOTH nulls AND the excess persists within
    the largest task-type cluster. `within_cluster_excess` is True/False from the sub-matrix re-test,
    or None if the control is UNTESTABLE (largest cluster < 8) — in which case detection CANNOT be
    certified (the H_task-cluster-trivial branch is unadjudicated). Pure boolean."""
    return bool(beats_both(full_res) and within_cluster_excess is True)


def largest_cluster(labels, min_size=8):
    """Largest task-type cluster from per-FT labels (aligned to the distance-matrix index order). The
    within-cluster control (§7.2 / §7.3) is TESTABLE only if it has ≥`min_size` members; otherwise the
    control is reported UNTESTABLE (frozen rule — not skipped silently). Returns {label,size,indices,
    testable}. RAISES on empty labels."""
    from collections import Counter
    labels = list(labels)
    assert labels, "empty label list"
    label, size = Counter(labels).most_common(1)[0]
    indices = [i for i, l in enumerate(labels) if l == label]
    return {"label": label, "size": int(size), "indices": indices,
            "testable": bool(size >= min_size), "min_size": int(min_size)}


# ---------------------------------------------------------------------------
# read-family (S_action / S_vl) OPTIONAL rows — principal-angle chordal distances (§7.2).
# ---------------------------------------------------------------------------
def chordal_distance(Q1, Q2):
    """Chordal (projection Frobenius) principal-angle distance between two subspaces given as
    orthonormal bases Q1,Q2 (d×k, same k). d² = Σ sin²θ_i = k − ‖Q1ᵀQ2‖²_F. RAISES on a rank/shape
    mismatch. fp64. (Used only for the exploratory S_action/S_vl draw-family rows.)"""
    import numpy as np
    Q1 = np.asarray(Q1, dtype=np.float64)
    Q2 = np.asarray(Q2, dtype=np.float64)
    assert Q1.shape == Q2.shape, f"subspace shape mismatch {Q1.shape} vs {Q2.shape}"
    k = Q1.shape[1]
    fro2 = float(((Q1.T @ Q2) ** 2).sum())
    return float(np.sqrt(max(0.0, k - fro2)))


def chordal_distance_matrix(bases):
    """Pairwise chordal-distance matrix of a list of orthonormal subspace bases (the banked S_action /
    S_vl re-estimation draws). Symmetric, 0 diagonal. Feeds `ccc_with_nulls` for the exploratory
    read-family rows."""
    import numpy as np
    b = list(bases)
    m = len(b)
    D = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = chordal_distance(b[i], b[j])
    return D
