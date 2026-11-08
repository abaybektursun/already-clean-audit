"""W8 — Simple weight + activation cross-checks (spec §7 W8, post-freeze amendment
2026-05-27). EXPLORATORY / DISCOVERY.

Scientific hook (the W8 cross-check): W1 says the LM weight change is full-rank +
uniform across all 26 decoder layers (eff-rank ~1440), yet the Delta-Crosscoder says
emergent=0 (re-weighting, no new feature directions). **How does a full-rank ΔW produce
NO new directions?** The reconciling hypothesis: a full-rank ΔW can be a rotation /
rescaling WITHIN the base weight's existing row/column (output) space — it re-weights
existing directions rather than writing into new ones. eff-rank alone is blind to this:
a matrix can be full-rank and still have all its energy inside an existing subspace.

This module provides the SIMPLEST direct tests, pure-CPU where possible:

  1. WEIGHT-SPACE (CPU, cached weights only):
     * full SVD spectrum of ΔW = (W_after - W_before) per LM decoder matrix
       (q/k/v/o/gate/up/down, sampled layers) — flat (full-rank) vs concentrated.
     * ΔW magnitude (entry-value) histograms per matrix type.
     * principal-angle / SUBSPACE-OVERLAP between the top-k OUTPUT-direction subspace of
       ΔW and the top-k output-direction subspace of W_before. HIGH overlap = the full-rank
       ΔW writes into the SAME output directions the base layer already uses (re-weighting),
       reconciling full-rank-ΔW with crosscoder emergent=0.
     * NULL: the same overlap computed against a random Gaussian ΔW (matched shape) — the
       N3-style random-init floor. A random ΔW's subspace overlaps W_before only at the
       chance level set by dimensions; real ΔW must beat that to claim "stays in base space".

  2. ACTIVATION-SPACE (the direct test; needs a tiny activation harvest, done in the
     Modal wrapper — this module only holds the pure math):
     * Grassmann / principal-subspace overlap between the top-k PCA subspace of BEFORE
       (decoder-reset PG2) activations and AFTER (PG2) activations at a layer, on a shared
       probe set. HIGH overlap = the rewrite re-weights within the existing activation
       subspace (no new directions) — corroborating crosscoder emergent=0.
     * NULL: overlap vs a random k-dim subspace of R^d (the chance floor).

Convention (Linear layer): W has shape [out_features, in_features], y = x @ W.T. The
directions a layer WRITES into the residual stream are the LEFT singular vectors U of
W (W = U S V^T, U columns in R^out). "Output-direction subspace" below always means the
span of the leading left singular vectors. For ΔW this answers: does the weight update add
NEW output directions, or only re-mix existing ones?

Subspace-overlap metric (used for both weight and activation):
  Given orthonormal bases A (d×k) and B (d×k) of two k-dim subspaces, the mean squared
  cosine of the principal angles is  ||A^T B||_F^2 / k  ∈ [0, 1].
    = 1  → subspaces coincide (every principal angle 0).
    = 0  → subspaces orthogonal (every principal angle 90°).
  This equals the average projection energy of one basis onto the other; for a random
  k-dim subspace of R^d the expectation is k/d (the chance floor we report as the null).

Design rules (CLAUDE.md + spec §7):
  * No try/except — fail loud with full trace.
  * Pure functions, JSON-serialisable returns. NO Modal, NO I/O, NO global state.
  * The weight functions are CPU-only (numpy/torch CPU). The activation function takes
    plain tensors/arrays (the Modal wrapper harvests them on GPU and passes them here).
"""
from __future__ import annotations

from typing import Any


# ===========================================================================
# Subspace-overlap primitive (shared by weight + activation tests).
# ===========================================================================

def subspace_overlap(basis_a, basis_b) -> float:
    """Mean squared principal-angle cosine between two subspaces.

    basis_a: (d, ka) ORTHONORMAL columns spanning subspace A.
    basis_b: (d, kb) ORTHONORMAL columns spanning subspace B.

    Returns ||A^T B||_F^2 / min(ka, kb) ∈ [0, 1]. Normalised by the SMALLER subspace
    dimension so the metric is 1 iff the smaller subspace is contained in the larger.
    """
    import numpy as np

    A = np.asarray(basis_a, dtype=np.float64)
    B = np.asarray(basis_b, dtype=np.float64)
    assert A.ndim == 2 and B.ndim == 2, f"bases must be 2-D, got {A.shape}, {B.shape}"
    assert A.shape[0] == B.shape[0], f"basis ambient-dim mismatch: {A.shape} vs {B.shape}"
    k = min(A.shape[1], B.shape[1])
    M = A.T @ B                                  # (ka, kb)
    return float((M ** 2).sum() / k)


def random_subspace_overlap_floor(d: int, k: int) -> float:
    """Expected subspace_overlap of two INDEPENDENT random k-dim subspaces of R^d.

    For Haar-random k-subspaces, E[||A^T B||_F^2] = k^2 / d, so the normalised metric
    (divide by k) has expectation k/d. This is the chance floor: any real overlap must
    beat k/d to mean "the subspaces genuinely share directions".
    """
    assert d > 0 and k > 0, f"d,k must be positive (d={d}, k={k})"
    return float(k) / float(d)


def empirical_random_subspace_overlap(d: int, k: int, n_draw: int, seed: int) -> dict:
    """Monte-Carlo the random-subspace overlap floor (mean ± std over n_draw pairs).

    Confirms the analytic k/d floor and gives an empirical null band for the figure.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draw):
        A = np.linalg.qr(rng.standard_normal((d, k)))[0][:, :k]
        B = np.linalg.qr(rng.standard_normal((d, k)))[0][:, :k]
        vals.append(subspace_overlap(A, B))
    vals = np.asarray(vals)
    return {
        "d": int(d), "k": int(k), "n_draw": int(n_draw),
        "analytic_floor_k_over_d": float(k) / float(d),
        "empirical_mean": float(vals.mean()),
        "empirical_std": float(vals.std()),
        "empirical_p95": float(np.percentile(vals, 95)),
    }


# ===========================================================================
# WEIGHT-SPACE cross-check (CPU). All on a single ΔW = W_after - W_before.
# ===========================================================================

def _svd_via_gram(mat):
    """Economy SVD (U, S, Vt) of `mat` (out×in) via the SMALLER Gram matrix — much faster
    on CPU than np.linalg.svd for very tall/wide matrices (e.g. Gemma2 MLP 9216×2304).

    For these LM matrices in=2304 (small) while out∈{1024,2048,2304,9216}. We always
    eigendecompose the in×in Gram W^T W (2304×2304), then U = W V / S. Returns
    (U (out×r), S (r,), Vt (r×in)) with r = #positive singular values, descending.
    Mathematically identical to np.linalg.svd(full_matrices=False) up to sign/degeneracy.
    """
    import numpy as np
    W = np.asarray(mat, dtype=np.float64)
    out_d, in_d = W.shape
    # eigendecompose the smaller side's Gram matrix.
    if in_d <= out_d:
        G = W.T @ W                                   # (in, in)
        w, V = np.linalg.eigh(G)                       # ascending
        order = np.argsort(w)[::-1]
        w = np.clip(w[order], 0.0, None)
        V = V[:, order]                                # (in, r0)
        S = np.sqrt(w)
        pos = S > (S[0] * 1e-12 if S.size and S[0] > 0 else 0.0)
        S = S[pos]; V = V[:, pos]
        U = (W @ V) / S[None, :]                        # (out, r)
        return U, S, V.T
    else:
        G = W @ W.T                                    # (out, out) — only if out<in (none here)
        w, U = np.linalg.eigh(G)
        order = np.argsort(w)[::-1]
        w = np.clip(w[order], 0.0, None)
        U = U[:, order]
        S = np.sqrt(w)
        pos = S > (S[0] * 1e-12 if S.size and S[0] > 0 else 0.0)
        S = S[pos]; U = U[:, pos]
        Vt = (U.T @ W) / S[:, None]
        return U, S, Vt


def _left_singular_basis(mat, k: int):
    """Top-k LEFT singular vectors (output directions) of `mat` (out×in), orthonormal."""
    import numpy as np
    U, S, _ = _svd_via_gram(mat)
    k = min(k, U.shape[1])
    return U[:, :k], S


def delta_spectrum(w_before, w_after, hist_bins: int = 60) -> dict:
    """Full SVD spectrum of ΔW = w_after - w_before, plus magnitude histogram + the
    SVD participation-ratio eff-rank (same definition as W1's weight_diff).

    w_before / w_after: 2-D arrays/tensors of identical [out, in] shape.

    Returns (all JSON-able):
      singular_values        : full σ list (descending)
      sv_normalised          : σ / σ_max  (spectrum shape: flat→all near plateau,
                               concentrated→fast decay)
      participation_ratio    : (Σσ)^2 / Σσ^2  ∈ [1, min(out,in)]  (=W1 eff-rank)
      spectral_concentration : σ_1^2 / Σσ_i^2  (energy in the top direction; low→spread)
      energy_top1/top10/top50_frac : cumulative energy fractions (spectrum shape)
      delta_value_hist       : histogram of ΔW entry values (counts + bin edges)
      frob_norm              : ||ΔW||_F
      rel_frob               : ||ΔW||_F / ||W_before||_F
    """
    import numpy as np

    Wb = np.asarray(w_before, dtype=np.float64)
    Wa = np.asarray(w_after, dtype=np.float64)
    assert Wb.shape == Wa.shape and Wb.ndim == 2, f"need matching 2-D, got {Wb.shape},{Wa.shape}"
    D = Wa - Wb

    _, sv, _ = _svd_via_gram(D)                       # fast singular values via in×in Gram
    sv = sv[sv > 0]
    energy = (sv ** 2)
    total_e = float(energy.sum())
    cum = np.cumsum(energy)

    def _frac(n):
        n = min(n, sv.size)
        return float(cum[n - 1] / total_e) if total_e > 0 and n > 0 else 0.0

    pr = (float(sv.sum()) ** 2 / total_e) if total_e > 0 else 0.0
    frob = float(np.linalg.norm(D))
    base_frob = float(np.linalg.norm(Wb))

    counts, edges = np.histogram(D.ravel(), bins=hist_bins)

    return {
        "shape": list(D.shape),
        "n_singular_values": int(sv.size),
        "max_rank": int(min(D.shape)),
        "singular_values": sv.tolist(),
        "sv_normalised": (sv / sv[0]).tolist() if sv.size else [],
        "participation_ratio": pr,
        "participation_ratio_frac_of_maxrank": pr / float(min(D.shape)),
        "spectral_concentration_top1": float(energy[0] / total_e) if total_e > 0 else 0.0,
        "energy_top1_frac": _frac(1),
        "energy_top10_frac": _frac(10),
        "energy_top50_frac": _frac(50),
        "frob_norm": frob,
        "rel_frob": (frob / base_frob) if base_frob > 0 else frob,
        "delta_value_hist": {"counts": counts.tolist(), "bin_edges": edges.tolist()},
    }


def delta_vs_base_subspace_overlap(
    w_before, w_after, k: int, n_random: int = 8, seed: int = 0,
) -> dict:
    """THE reconciling test (weight space): does ΔW's top-k OUTPUT-direction subspace
    lie inside W_before's top-k output-direction subspace?

    Computes:
      overlap_real  = subspace_overlap(U_k(ΔW), U_k(W_before))   ∈ [0,1]
      overlap_null  = same, but ΔW replaced by a random Gaussian matrix scaled to the
                      same Frobenius norm (matched shape) — the N3 random-init floor;
                      averaged over n_random draws (mean ± std).
      chance_floor  = k / out_features  (Haar random-subspace expectation).

    HIGH overlap_real (≫ overlap_null and ≫ chance_floor) → the full-rank ΔW re-uses the
    base layer's existing output directions (rotation/rescale WITHIN the base column
    space) → reconciles "full-rank ΔW" with crosscoder "emergent=0 / re-weighting".
    LOW overlap_real → ΔW writes into NEW output directions → tension with the crosscoder
    (report it honestly, that is a real finding).

    Also reports overlap_real_full_vs_full: top-k(ΔW) vs the FULL leading-k of W_before,
    and a sweep over several k so the result is not a single-k artefact.
    """
    import numpy as np

    Wb = np.asarray(w_before, dtype=np.float64)
    Wa = np.asarray(w_after, dtype=np.float64)
    assert Wb.shape == Wa.shape and Wb.ndim == 2
    D = Wa - Wb
    out_dim = D.shape[0]

    Ub_full, _ = _left_singular_basis(Wb, min(out_dim, max(k, 50)))

    def _overlap_at(kk: int, delta_mat) -> float:
        Ud, _ = _left_singular_basis(delta_mat, kk)
        Ub = Ub_full[:, :kk]
        return subspace_overlap(Ud, Ub)

    # real overlap at the requested k.
    overlap_real = _overlap_at(k, D)

    # null: random Gaussian matrices, scaled to ||D||_F, matched shape.
    rng = np.random.default_rng(seed)
    frob_D = float(np.linalg.norm(D))
    null_vals = []
    for _ in range(n_random):
        R = rng.standard_normal(D.shape)
        R = R * (frob_D / float(np.linalg.norm(R)))     # match Frobenius norm
        null_vals.append(_overlap_at(k, R))
    null_vals = np.asarray(null_vals)

    # k-sweep (real only) — robustness of the headline to the chosen k.
    k_sweep = {}
    for kk in sorted({max(1, k // 4), max(1, k // 2), k, min(out_dim, 2 * k)}):
        if kk <= out_dim:
            k_sweep[str(kk)] = _overlap_at(kk, D)

    return {
        "k": int(k),
        "out_dim": int(out_dim),
        "in_dim": int(D.shape[1]),
        "overlap_real": overlap_real,
        "overlap_null_random_dW_mean": float(null_vals.mean()),
        "overlap_null_random_dW_std": float(null_vals.std()),
        "overlap_null_random_dW_max": float(null_vals.max()),
        "chance_floor_k_over_out": float(k) / float(out_dim),
        "excess_over_null": overlap_real - float(null_vals.mean()),
        "excess_over_chance": overlap_real - float(k) / float(out_dim),
        "k_sweep_overlap_real": k_sweep,
        "interpretation": (
            "ΔW stays in base output-subspace (re-weighting; reconciles full-rank + emergent=0)"
            if overlap_real > max(float(null_vals.mean()), float(k) / float(out_dim)) * 1.5
            else "ΔW overlaps base only near the null → moves into NEW directions (tension w/ crosscoder)"
        ),
    }


def delta_energy_in_base_subspace(
    w_before, w_after, ranks=(8, 32, 64, 128, 256), n_random: int = 8, seed: int = 0,
) -> dict:
    """The FAITHFUL reconciliation test (weight space), complementary to the strict
    top-k-vs-top-k overlap above.

    Question: what FRACTION of ΔW's output energy lands inside the rank-r PRINCIPAL
    output subspace of W_before (its top-r left singular vectors)? I.e. how much of the
    weight update writes into the directions the base layer ALREADY uses most.

      frac_energy(r) = ||P_r^{W0} · ΔW||_F^2 / ||ΔW||_F^2 ,   P_r^{W0} = U_r U_r^T

    where U_r are the top-r left singular vectors of W_before. Swept over r so we see the
    full curve, not a single cut. This does NOT require ΔW's *dominant* directions to be
    W0's dominant directions (the strict overlap test); it asks whether ΔW lives inside the
    span W0 spans — the correct phrasing of "rotation/rescale within the base column space".

    NULL: for a random ΔW (matched shape/Frobenius), E[frac_energy(r)] = r / out_features
    exactly (isotropic energy → fraction = subspace dim / ambient dim). We also Monte-Carlo
    it. Real frac_energy(r) ABOVE r/out → ΔW preferentially reuses W0's dominant output
    directions (re-weighting); AT r/out → ΔW is isotropic w.r.t. W0's basis (writes
    everywhere W0 could, no preference); the cumulative curve reaching ~1 well before r=out
    means ΔW is confined to a low-dim slice of W0's output span.
    """
    import numpy as np

    Wb = np.asarray(w_before, dtype=np.float64)
    Wa = np.asarray(w_after, dtype=np.float64)
    assert Wb.shape == Wa.shape and Wb.ndim == 2
    D = Wa - Wb
    out_dim = D.shape[0]
    frob2_D = float((D ** 2).sum())

    # full left-singular basis of W_before (output directions, ordered by energy) — fast Gram route.
    Ub, _, _ = _svd_via_gram(Wb)                        # (out, min(out,in))

    valid_ranks = [r for r in ranks if r <= Ub.shape[1]]
    rmax = max(valid_ranks)

    def _cum_frac_curve(mat):
        """Cumulative fraction of `mat`'s energy in W0's top-r output span, for all
        valid ranks at once: project ONCE onto Ub[:, :rmax], take cumulative row energy."""
        proj = (Ub[:, :rmax].T @ mat)                   # (rmax, in)
        f2 = float((mat ** 2).sum())
        if f2 <= 0:
            return {str(r): 0.0 for r in valid_ranks}
        row_e = np.cumsum((proj ** 2).sum(axis=1))      # cumulative energy by rank
        return {str(r): float(row_e[r - 1] / f2) for r in valid_ranks}

    real_curve = _cum_frac_curve(D)

    # null: random Gaussian matrices (scale-free fraction → no Frobenius match needed);
    # ONE matrix per draw, all ranks from a single projection.
    rng = np.random.default_rng(seed)
    null_draws = [_cum_frac_curve(rng.standard_normal(D.shape)) for _ in range(n_random)]
    null_curve_mean, null_curve_std, chance = {}, {}, {}
    for r in valid_ranks:
        vals = np.asarray([d[str(r)] for d in null_draws])
        null_curve_mean[str(r)] = float(vals.mean())
        null_curve_std[str(r)] = float(vals.std())
        chance[str(r)] = float(r) / float(out_dim)

    # headline at a mid rank (first available ≥ out_dim//8, else the largest).
    target = next((r for r in valid_ranks if r >= max(8, out_dim // 8)), rmax)
    real_h = real_curve[str(target)]
    chance_h = chance[str(target)]
    return {
        "out_dim": int(out_dim), "in_dim": int(D.shape[1]),
        "ranks": valid_ranks,
        "frac_energy_in_base_topr_real": real_curve,
        "frac_energy_in_base_topr_null_mean": null_curve_mean,
        "frac_energy_in_base_topr_null_std": null_curve_std,
        "chance_frac_r_over_out": chance,
        "headline_rank": int(target),
        "headline_real": real_h,
        "headline_chance": chance_h,
        "headline_excess_over_chance": real_h - chance_h,
        "interpretation": (
            f"at rank {target}: {real_h:.1%} of ΔW energy is in W0's top-{target} output "
            f"directions vs {chance_h:.1%} chance → "
            + ("ΔW PREFERENTIALLY reuses W0's dominant directions (re-weighting)"
               if real_h > chance_h * 1.5 else
               "ΔW is ~isotropic in W0's basis (no strong preference for W0's dominant dirs)")),
    }


# ===========================================================================
# ACTIVATION-SPACE cross-check (pure math; harvested tensors passed in).
# ===========================================================================

def _pca_full(acts):
    """Full principal-component basis of row-vectors `acts` (N×d), mean-centred, via the
    d×d covariance eigendecomposition (FAST: eigh on d×d, not SVD of the tall N×d — for
    N≫d this avoids forming the N×d left factor and is the standard efficient PCA).

    Returns (V, eigvals_descending) where V is (d, d) orthonormal with columns = PCs
    ordered by descending variance. Top-k subspace basis is simply V[:, :k]."""
    import numpy as np
    X = np.asarray(acts, dtype=np.float64)
    assert X.ndim == 2, f"acts must be (N,d), got {X.shape}"
    N = X.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(N - 1, 1)                 # (d, d) symmetric PSD
    w, V = np.linalg.eigh(cov)                         # ascending eigvals
    order = np.argsort(w)[::-1]                        # descending
    return V[:, order], w[order]


def _pca_basis(acts, k: int):
    """Top-k PC basis (orthonormal, d×k) + top-k explained-variance ratio."""
    import numpy as np
    V, evals = _pca_full(acts)
    k = min(k, V.shape[1])
    tot = float(evals.sum())
    evr = float(evals[:k].sum() / tot) if tot > 0 else 0.0
    return V[:, :k], evr


def activation_subspace_overlap(
    acts_before, acts_after, k: int, n_random: int = 8, seed: int = 0,
) -> dict:
    """THE direct test: do AFTER (PG2) activations span the SAME k-dim subspace as BEFORE
    (decoder-reset base) activations, on a shared (paired) probe set at one layer?

    acts_before / acts_after: (N, d) row-aligned activation matrices (token i is the same
    token in both, harvested from the paired before/after forward).

    Returns:
      overlap_real            = subspace_overlap(PCA_k(after), PCA_k(before))
      overlap_null_random     = overlap of PCA_k(after) vs a RANDOM k-subspace of R^d
                                (mean ± std over n_random) — the chance floor.
      chance_floor_k_over_d   = k / d (analytic Haar floor)
      evr_before/after        = top-k explained-variance ratio of each (subspace richness)

    HIGH overlap_real (≫ null) → the rewrite re-weights WITHIN the existing activation
    subspace; no new activation directions → corroborates crosscoder emergent=0 and
    reconciles the full-rank ΔW. LOW → the rewrite moves activations into new directions
    (tension; report honestly).
    """
    import numpy as np

    Xb = np.asarray(acts_before, dtype=np.float64)
    Xa = np.asarray(acts_after, dtype=np.float64)
    assert Xb.shape == Xa.shape and Xb.ndim == 2, f"need matching (N,d): {Xb.shape},{Xa.shape}"
    d = Xb.shape[1]

    # Compute the FULL PC basis ONCE for each (fast eigh on d×d); slice for the k-sweep.
    Vb, evb = _pca_full(Xb)
    Va, eva = _pca_full(Xa)
    totb, tota = float(evb.sum()), float(eva.sum())

    def _evr(ev, tot, kk):
        return float(ev[:kk].sum() / tot) if tot > 0 else 0.0

    overlap_real = subspace_overlap(Va[:, :k], Vb[:, :k])
    evr_b, evr_a = _evr(evb, totb, k), _evr(eva, tota, k)

    # null: PCA_k(after) vs random k-subspaces.
    rng = np.random.default_rng(seed)
    null_vals = []
    for _ in range(n_random):
        R = np.linalg.qr(rng.standard_normal((d, k)))[0][:, :k]
        null_vals.append(subspace_overlap(Va[:, :k], R))
    null_vals = np.asarray(null_vals)

    # k-sweep so the headline is not a single-k artefact (just slices of the full bases).
    k_sweep = {}
    for kk in sorted({max(1, k // 4), max(1, k // 2), k, min(d, 2 * k)}):
        if kk <= d:
            k_sweep[str(kk)] = subspace_overlap(Va[:, :kk], Vb[:, :kk])

    return {
        "k": int(k),
        "d_model": int(d),
        "n_tokens": int(Xb.shape[0]),
        "overlap_real": overlap_real,
        "overlap_null_random_mean": float(null_vals.mean()),
        "overlap_null_random_std": float(null_vals.std()),
        "overlap_null_random_p95": float(np.percentile(null_vals, 95)),
        "chance_floor_k_over_d": float(k) / float(d),
        "evr_before_topk": evr_b,
        "evr_after_topk": evr_a,
        "excess_over_null": overlap_real - float(null_vals.mean()),
        "k_sweep_overlap_real": k_sweep,
        "interpretation": (
            "AFTER spans the SAME subspace as BEFORE (re-weighting within existing "
            "activation subspace; reconciles full-rank ΔW + emergent=0)"
            if overlap_real > max(float(null_vals.mean()), float(k) / float(d)) * 1.5
            else "AFTER moves into NEW activation directions (tension with crosscoder emergent=0)"
        ),
    }


# Optional convenience: mean per-dimension range/variance shift (the [V19-P5]/W3 audit
# rendered as a number) — image vs text positions, before vs after. Pure stats.
def activation_norm_stats(acts, is_image_mask) -> dict:
    """Per-token L2 norm distribution split by image vs text position. Renders the W3
    img/txt norm-ratio shift as a number (not a swap; both halves measured the same way)."""
    import numpy as np
    X = np.asarray(acts, dtype=np.float64)
    m = np.asarray(is_image_mask, dtype=bool)
    norms = np.linalg.norm(X, axis=1)
    img, txt = norms[m], norms[~m]

    def _s(a):
        return {"n": int(a.size), "mean": float(a.mean()) if a.size else None,
                "median": float(np.median(a)) if a.size else None,
                "std": float(a.std()) if a.size else None} if a.size else {"n": 0}
    img_mean = float(img.mean()) if img.size else None
    txt_mean = float(txt.mean()) if txt.size else None
    return {
        "image_token_norm": _s(img),
        "text_token_norm": _s(txt),
        "img_over_txt_norm_ratio": (img_mean / txt_mean) if (img_mean and txt_mean) else None,
    }
