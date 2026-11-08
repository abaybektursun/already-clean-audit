"""GAUGE-CS (P1, the MDE) — basis-coherence scan + compressed-sensing recovery core.

Question (§4): is ΔW concentrated (sparse/compressible) in some basis our energy-coherent
instruments never looked at? The program's sparsity reads all live in bases COHERENT with energy
(ΔW's own SVD; W13's dictionary). Compressed sensing's theorem: sparse structure is recoverable
only when the measurement basis is INCOHERENT with the sparsity basis. So we express each ΔW in
five candidate basis classes and ask whether the coefficient concentration exceeds a
spectrum-matched rotation null.

Pure, weight-free, fp64, no try/except (house style: relaw.py, reuse_estimators.py). Heavy imports
inside functions.

THE ROTATION-INVARIANCE REDUCTION (load-bearing, makes the null cheap and basis-independent).
Coefficients in a FIXED orthonormal basis B are  M_B = U_Bᵀ ΔW V_B. The spectrum-matched null
replaces ΔW's singular VECTORS by Haar-random ones at the SAME spectrum σ:
  ΔW_null = Q_L diag(σ) Q_Rᵀ,  Q_L,Q_R Haar frames.
Then  M_B = (U_Bᵀ Q_L) diag(σ) (Q_Rᵀ V_B). Because U_B, V_B are fixed orthonormal and Q_L, Q_R
are Haar, (U_Bᵀ Q_L) and (V_Bᵀ Q_R) are THEMSELVES Haar frames (invariance of Haar under a fixed
orthogonal map). ⇒ the distribution of the null coefficient concentration depends ONLY on (σ, m, n),
NEVER on B. So ONE null band per (matrix) serves EVERY fixed-basis class, and — the same argument —
the concentration of the REAL ΔW measured in a UNIFORMLY RANDOM basis (class random-K) is
DISTRIBUTED EXACTLY as this null. That is why random-K calibrates the incoherent floor: any excess
over it is basis-structure, not spectrum texture.
"""
from __future__ import annotations

from infra.pg import gauge_common as GC

# The candidate basis classes (§4.2), frozen. 'dW_svd' is the reference (concentration maximal by
# construction — reported, never tested); 'random_k' calibrates the incoherent floor (= the null,
# by the rotation-invariance reduction above).
BASIS_CLASSES = ("entry", "base_svd", "dW_svd", "random_k", "consensus")


# ---------------------------------------------------------------------------
# concentration statistic (Gini primary, participation ratio secondary).
# ---------------------------------------------------------------------------
def concentration(M):
    """Concentration of a coefficient matrix M: Gini (primary) + participation ratio (secondary) of
    the SQUARED coefficient magnitudes, flattened. High Gini / low PR ⇒ concentrated. Returns
    {'gini':…, 'pr':…}. fp64."""
    import numpy as np
    a = (np.asarray(M, dtype=np.float64) ** 2).ravel()
    return {"gini": GC.gini(a), "pr": GC.participation_ratio(a)}


# ---------------------------------------------------------------------------
# candidate bases + coefficient projection.
# ---------------------------------------------------------------------------
def coeffs(dW, U_B, V_B):
    """M_B = U_Bᵀ ΔW V_B — ΔW's coefficients in the (left,right) basis pair. fp64. For the entry
    class pass identities; callers usually build the class-specific basis then call this."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    return np.asarray(U_B, np.float64).T @ A @ np.asarray(V_B, np.float64)


def svd_bases(W):
    """(U, V) — left (m×m via full) and right (n×n) singular bases of W, fp64. Used for the
    'base_svd' class (W = W_base) and, applied to ΔW itself, the 'dW_svd' reference class. Full
    orthonormal factors so the two-sided projection is exact (no truncation)."""
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    U, _s, Vh = np.linalg.svd(A, full_matrices=True)
    return U, Vh.T


def random_ortho(d, seed):
    """A Haar-random d×d orthogonal matrix (QR of a Gaussian, sign-fixed on R's diagonal so the
    result is genuinely Haar). One of the K seeded incoherent probes (class 'random_k')."""
    import numpy as np
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(Z)
    Q *= np.sign(np.diag(R))            # remove QR's sign ambiguity → Haar
    return Q


def consensus_bases(dW_list):
    """The family consensus bases (class 'consensus', §4.2 class 5), estimated on the DISCOVERY FTs
    only (leakage control §4.3). Left basis = eigenvectors of Σ_i ΔW_iΔW_iᵀ/‖ΔW_i‖²_F (m×m); right
    basis = eigenvectors of Σ_i ΔW_iᵀΔW_i/‖ΔW_i‖²_F (n×n) — the right one is exactly
    invariant_core._consensus_gram's matrix. Eigenvectors returned in DESCENDING eigenvalue order
    (the family's most-written coordinates first). Each ΔW is unit-Frobenius normalised so every FT
    contributes equally (invariant_core convention)."""
    import numpy as np
    assert dW_list, "empty ΔW list for consensus bases"
    m, n = np.asarray(dW_list[0]).shape
    GL = np.zeros((m, m), dtype=np.float64)
    GR = np.zeros((n, n), dtype=np.float64)
    for dW in dW_list:
        A = np.asarray(dW, dtype=np.float64)
        assert A.shape == (m, n), f"consensus: shape drift {A.shape} vs {(m, n)}"
        fro2 = float((A ** 2).sum())
        assert fro2 > 0, "zero ΔW in consensus set (degenerate FT slipped the gate)"
        GL += (A @ A.T) / fro2
        GR += (A.T @ A) / fro2
    # eigvalsh ascending → reverse to descending.
    _wl, UL = np.linalg.eigh(GL)
    _wr, VR = np.linalg.eigh(GR)
    return UL[:, ::-1], VR[:, ::-1]


def concentration_in_class(dW, cls, *, W_base=None, base_svd_bases=None, consensus=None,
                           random_bases=None, random_seed=None, n_random=8):
    """Concentration of one ΔW in one basis class. Dispatches the class-specific basis then projects.

    PERFORMANCE: base_svd and random_k bases are FT-INDEPENDENT (W_base and the seeded Haar frames
    do not vary across fine-tunes of the same matrix). Callers streaming many FTs of one (type,layer)
    MUST precompute them ONCE and pass `base_svd_bases=(U,V)` / `random_bases=[(U,V),...]` — otherwise
    a full 2048² SVD + K QRs are redundantly recomputed per FT (~12× at the 24-FT family). When the
    precomputed bases are absent the function falls back to building them (used by the weight-free
    tests on tiny matrices).
      entry     — identity bases (M_B = ΔW itself).
      base_svd  — singular bases of W_base (requires W_base).
      dW_svd    — ΔW's own singular bases (reference; concentration maximal — M_B = diag(σ)).
      random_k  — the MEAN concentration over n_random seeded Haar orthogonal basis pairs.
      consensus — the family consensus bases (requires the precomputed (U_cons,V_cons)).
    Returns {'gini','pr'} (for random_k, the mean over the K probes)."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    m, n = A.shape
    if cls == "entry":
        return concentration(A)
    if cls == "base_svd":
        U_B, V_B = base_svd_bases if base_svd_bases is not None else svd_bases(
            W_base if W_base is not None else _need("base_svd", "W_base"))
        return concentration(coeffs(A, U_B, V_B))
    if cls == "dW_svd":
        U_B, V_B = svd_bases(A)
        return concentration(coeffs(A, U_B, V_B))
    if cls == "consensus":
        assert consensus is not None, "consensus needs precomputed (U_cons,V_cons)"
        U_B, V_B = consensus
        return concentration(coeffs(A, U_B, V_B))
    if cls == "random_k":
        if random_bases is None:
            assert random_seed is not None, "random_k needs precomputed random_bases or a base seed"
            random_bases = [(random_ortho(m, random_seed + 2 * j),
                             random_ortho(n, random_seed + 2 * j + 1)) for j in range(n_random)]
        gs, ps = [], []
        for U_B, V_B in random_bases:
            c = concentration(coeffs(A, U_B, V_B))
            gs.append(c["gini"])
            ps.append(c["pr"])
        return {"gini": float(np.mean(gs)), "pr": float(np.mean(ps))}
    raise ValueError(f"unknown basis class {cls!r}")


def _need(cls, arg):
    raise AssertionError(f"{cls} needs {arg} (or a precomputed basis)")


# ---------------------------------------------------------------------------
# spectrum-matched rotation null (basis-independent, by the reduction above).
# ---------------------------------------------------------------------------
def null_concentration_band(spectrum, m, n, n_draws=32, seed=1337,
                            pcts=(2.5, 50, 97.5)):
    """The spectrum-matched rotation-null band for the coefficient concentration of an m×n matrix
    with singular values `spectrum` (length r=min(m,n)). Per draw: M = L diag(σ) Rᵀ with Haar
    frames L(m×r), R(n×r); concentration of M. BASIS-INDEPENDENT (the module docstring reduction) —
    so this one band is the null for every fixed-basis class AND the expected distribution of the
    random-K class. Returns percentiles + mean/std + raw samples for both gini and pr."""
    import numpy as np
    sig = np.asarray(spectrum, dtype=np.float64)
    r = sig.shape[0]
    assert r == min(m, n), f"spectrum length {r} != min(m,n)={min(m, n)}"
    rng = np.random.default_rng(seed)

    def _haar_frame(d):
        Z = rng.standard_normal((d, r))
        Q, _ = np.linalg.qr(Z)          # d×r, orthonormal columns (Haar frame)
        return Q

    gvals = np.empty(n_draws, dtype=np.float64)
    pvals = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        L = _haar_frame(m)
        R = _haar_frame(n)
        M = (L * sig[None, :]) @ R.T
        c = concentration(M)
        gvals[i], pvals[i] = c["gini"], c["pr"]

    def _stat(v):
        return {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size),
                **{f"p{p}": float(np.percentile(v, p)) for p in pcts},
                "samples": v.tolist()}
    return {"gini": _stat(gvals), "pr": _stat(pvals)}


def null_band_pooled(spectra, m, n, n_draws=48, seed=1337, pcts=(2.5, 50, 97.5)):
    """Per-CELL spectrum-matched null band — the §4.3 form (the band is built PER (matrix-type,
    class), NOT per FT). Each of `n_draws` draws uses one discovery FT's REAL spectrum, round-robin
    over `spectra`, with fresh Haar frames L(m×r), R(n×r); concentration of M = L diag(σ) Rᵀ. The
    band therefore reflects the cell's actual spectral distribution at a cost of `n_draws` QRs TOTAL
    (not n_draws × n_FT — the earlier per-FT null was ~n_FT× slower and finer-grained than the spec
    asks). Basis-independent (rotation invariance) ⇒ one band per cell serves every fixed class.
    Same {gini, pr} band format as null_concentration_band."""
    import numpy as np
    specs = [np.asarray(s, dtype=np.float64) for s in spectra]
    assert specs, "no spectra for the pooled null band"
    r = min(m, n)
    for s in specs:
        assert s.shape[0] == r, f"spectrum length {s.shape[0]} != min(m,n)={r}"
    rng = np.random.default_rng(seed)
    gvals = np.empty(n_draws, dtype=np.float64)
    pvals = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        sig = specs[i % len(specs)]
        L, _ = np.linalg.qr(rng.standard_normal((m, r)))
        R, _ = np.linalg.qr(rng.standard_normal((n, r)))
        M = (L * sig[None, :]) @ R.T
        c = concentration(M)
        gvals[i], pvals[i] = c["gini"], c["pr"]

    def _stat(v):
        return {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size),
                **{f"p{p}": float(np.percentile(v, p)) for p in pcts}, "samples": v.tolist()}
    return {"gini": _stat(gvals), "pr": _stat(pvals)}


def permute_entries(dW, seed):
    """The entry-permutation control (§4.1 H_trivial-concentration gate): a copy of ΔW with its
    entries permuted — SAME marginal entry distribution, ALL cross-entry (basis) structure
    destroyed. If the concentration excess persists here, it is heavy-tail texture, not structure."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    rng = np.random.default_rng(seed)
    flat = A.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(A.shape)


# ---------------------------------------------------------------------------
# family-level summary + frozen decision rules (§4.2 summary, §4.4).
# ---------------------------------------------------------------------------
def excess(real_c, null_median):
    """One cell's concentration excess = real − null-band median (the §4.2 per-FT excess ĉ_B term
    before the family median/CI)."""
    return float(real_c) - float(null_median)


def median_excess_ci(excesses, n_boot=2000, seed=1337):
    """Family-median excess ĉ_B + bootstrap-over-FTs 95% CI (§4.2 summary). `excesses` = per-FT
    [C_B − null-median] on the discovery (or sealed) half. Returns {median, ci_lo, ci_hi}."""
    import numpy as np
    e = np.asarray(excesses, dtype=np.float64)
    assert e.size >= 1, "no excesses to summarise"
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(e, e.size, replace=True)) for _ in range(n_boot)])
    return {"median": float(np.median(e)),
            "ci_lo": float(np.percentile(meds, 2.5)),
            "ci_hi": float(np.percentile(meds, 97.5))}


def bootstrap_p_le(excesses, delta, n_boot=2000, seed=1337):
    """One-sided bootstrap p that the family-median excess does NOT exceed `delta` (i.e. the fraction
    of bootstrap-over-FT medians ≤ delta). Small p ⇒ the median excess is confidently above δ_CS.
    Feeds the BH-FDR step over the ≤3 promoted cells at the sealed-confirmation stage (§4.4)."""
    import numpy as np
    e = np.asarray(excesses, dtype=np.float64)
    assert e.size >= 1, "no excesses for bootstrap p"
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(e, e.size, replace=True)) for _ in range(n_boot)])
    return float((np.sum(meds <= delta) + 1) / (n_boot + 1))


def discovery_clear(ci_lo, delta_cs, perm_ci_lo):
    """§4.4 Discovery-clear: the cell's median-excess CI-lo > δ_CS AND the entry-permutation control
    for that cell does NOT itself clear (perm_ci_lo ≤ δ_CS). If the permutation control also clears,
    the cell is H_trivial-concentration (heavy-tail texture), not a discovery. Pure predicate."""
    return bool(ci_lo > delta_cs and not (perm_ci_lo > delta_cs))


def detection(sealed_ci_lo, delta_cs, bh_ok, secondary_sign_agrees):
    """§4.4 Detection: a discovery-cleared cell whose SEALED-half median-excess CI-lo > δ_CS, under
    BH (q=0.10) over the ≤3 promoted cells, with the secondary statistic (PR) agreeing in sign.
    Pure boolean predicate (the BH reject decision is computed by relaw.bh_fdr upstream)."""
    return bool(sealed_ci_lo > delta_cs and bh_ok and secondary_sign_agrees)


# ---------------------------------------------------------------------------
# Stage 3 — CS-recovery leg (promotion-gated curiosity slice, §4.3).
# ---------------------------------------------------------------------------
def energy_truncation_error(x, k):
    """Relative ℓ2 error of the top-k energy (magnitude) truncation of coefficient vector x — the
    ENERGY-RANKED reconstruction GAUGE-CS's recovery leg must beat to call a basis load-bearing."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64).ravel()
    keep = np.argsort(np.abs(x))[::-1][:k]
    xr = np.zeros_like(x)
    xr[keep] = x[keep]
    denom = np.linalg.norm(x)
    assert denom > 0, "zero coefficient vector"
    return float(np.linalg.norm(x - xr) / denom)


def l1_recover(x, m_meas, seed, n_iter=400):
    """Basis-pursuit-denoise recovery of coefficient vector x from m_meas random Gaussian
    measurements y = Φx (Φ: m_meas×len(x), iid N(0,1/m)). FISTA on min ½‖Φz−y‖² + λ‖z‖₁ with a
    small λ (noiseless CS regime). Per-matrix FACTORED scope (one matrix's coefficient vector; NO
    global vec-ΔW solve — §4.3 / pre-mortem iii). Returns the relative ℓ2 reconstruction error
    ‖x−ẑ‖/‖x‖. Simplicity over speed: this leg only runs if a cell CONFIRMS (author predicts it
    never does — G1)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64).ravel()
    d = x.size
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((m_meas, d)) / np.sqrt(m_meas)
    y = Phi @ x
    # Lipschitz constant of the smooth part = largest eigenvalue of ΦᵀΦ.
    L = float(np.linalg.norm(Phi, 2) ** 2)
    lam = 1e-3 * float(np.abs(Phi.T @ y).max())        # small λ ⇒ near basis-pursuit
    z = np.zeros(d)
    w = z.copy()
    t = 1.0
    for _ in range(n_iter):
        grad = Phi.T @ (Phi @ w - y)
        zk = w - grad / L
        thr = lam / L
        zk = np.sign(zk) * np.maximum(np.abs(zk) - thr, 0.0)   # soft-threshold (ℓ1 prox)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        w = zk + ((t - 1.0) / t_next) * (zk - z)
        z, t = zk, t_next
    denom = np.linalg.norm(x)
    assert denom > 0, "zero coefficient vector"
    return float(np.linalg.norm(x - z) / denom)
