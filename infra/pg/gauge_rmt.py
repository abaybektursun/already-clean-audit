"""GAUGE-RMT (P3) — bulk/outlier decomposition of the ΔW singular spectrum.

Question (§6). "Full-rank" (W1) conflates two objects: a Marchenko–Pastur (MP) *bulk* — what a sum
of many quasi-random updates generically produces — and spectral *outliers*, directions carrying
signal above the noise pile. Random-matrix theory separates them. Which part of the full-rank
verdict is pile, which is signal?

Per matrix we compare the empirical spectral density (ESD) of the singular values against
  (a) an MP bulk fit with aspect-ratio-correct edge  λ± = σ²(1±√γ)², γ = min(m,n)/max(m,n);
      σ² is fit on the CENTRAL quantile range of the eigenvalues (outlier-robust), and
  (b) the DECISIVE null: the ESD of the ENTRY-PERMUTED ΔW (destroys cross-entry structure, keeps
      the marginal entry distribution), ≥3 permutations — this calibrates the edge-fluctuation
      buffer and the E_out null band, and the MP fit is only scaffolding (§6.3).

Variables (§6.2):
  * n_out = count of singular values whose eigenvalue λ_i = σ_i²/N exceeds the fitted bulk edge
    plus an edge-fluctuation BUFFER calibrated on the permutation null;
  * E_out = fraction of ‖ΔW‖²_F carried by those above-edge singular values;
  * reproducibility = mean pairwise principal-angle overlap of the top-n_out left/right outlier
    subspaces across discovery FTs, vs the Haar overlap floor (reuse_estimators.haar_null, REUSED).

Frozen rules (§6.2):
  * δ_RMT = permutation E_out band half-width (discovery, pre-real).
  * Detection = E_out excess CI-lo > δ_RMT on discovery AND sealed halves (BH over ≤3 promoted
    matrix-type rows) AND the outlier-subspace overlap beats the Haar floor — else "outliers exist
    but are FT-idiosyncratic" (its own labeled finding, not a family detection).
  * Fit-quality gate: if the MP fit rejects on >half the discovery cells (real bulk KS beyond the
    permutation-calibrated bound), report H_non-MP characterization and do NOT interpret the
    edge/outlier machinery (frozen abort-to-descriptive).

Pure, weight-free, fp64, no try/except (house style: gauge_cs.py, reuse_estimators.py). Heavy
imports inside functions. JSON-able returns.

REUSE (spec §14): the ESD is extracted with `latablate.singular_spectrum` (the LATABLATE object is
retired but this pure function is explicitly reuse-allowed); the entry-permutation control and the
family-median bootstrap CI are `gauge_cs.permute_entries` / `gauge_cs.median_excess_ci`; the Haar
overlap floor is `reuse_estimators.haar_null` with a flat rank-k spectrum (the ρ_L estimand of a
random k-frame against a fixed k-frame IS the mean squared principal-angle cosine).

THE MP–ESD MAP (load-bearing, kept simple). For an m×n matrix put N=max(m,n), p=min(m,n),
γ=p/N ∈ (0,1]. The p eigenvalues of the p×p Gram (1/N)·(wider ΔW)(wider ΔW)ᵀ are λ_i = σ_i²/N. For
entries iid with variance σ² these follow the MP law on [σ²(1−√γ)², σ²(1+√γ)²]. σ² is fit by
matching the empirical central-quantile MEAN of the λ_i to the MP central-quantile mean at σ²=1
(MP scales linearly in σ², so one grid integral gives the constant) — robust because the outliers
sit ABOVE the upper quantile and never enter the fit. The permutation null uses the IDENTICAL
pipeline, so any bias in the σ² estimator is common-mode and cancels in the E_out excess.
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# MP density / edges / central-quantile fit (pure numpy, aspect-ratio-correct).
# ---------------------------------------------------------------------------
def mp_edges(sigma2, gamma):
    """MP bulk edges λ± = σ²(1±√γ)² (γ = aspect ratio min/max ∈ (0,1]). Returns (λ−, λ+)."""
    sg = math.sqrt(gamma)
    return sigma2 * (1.0 - sg) ** 2, sigma2 * (1.0 + sg) ** 2


def mp_density(lam, sigma2, gamma):
    """MP density f(λ) = √((λ+−λ)(λ−λ−)) / (2π σ² γ λ) on (λ−, λ+), else 0 (γ ≤ 1: no atom at 0).
    Vectorised over the grid `lam`. Pure numpy."""
    import numpy as np
    lam = np.asarray(lam, dtype=np.float64)
    lo, hi = mp_edges(sigma2, gamma)
    d = np.zeros_like(lam)
    m = (lam > lo) & (lam < hi)
    d[m] = np.sqrt((hi - lam[m]) * (lam[m] - lo)) / (2.0 * np.pi * sigma2 * gamma * lam[m])
    return d


def _mp_grid(gamma, sigma2, npts=4000):
    """(λ grid, density, CDF) for the MP law at (γ, σ²). CDF by cumulative-trapezoid, normalised."""
    import numpy as np
    lo, hi = mp_edges(sigma2, gamma)
    lam = np.linspace(max(lo, 0.0), hi, npts)
    f = mp_density(lam, sigma2, gamma)
    cdf = np.cumsum(f)
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    return lam, f, cdf


def _mp_central_mean(gamma, q_lo, q_hi, npts=4000):
    """The mean of λ over the central quantile window [q_lo, q_hi] of the σ²=1 MP law (a constant
    c(γ, q_lo, q_hi); the fit uses σ² = empirical-central-mean / c because MP scales linearly)."""
    import numpy as np
    lam, f, cdf = _mp_grid(gamma, 1.0, npts)
    lo_i = int(np.searchsorted(cdf, q_lo))
    hi_i = int(np.searchsorted(cdf, q_hi))
    seg_l, seg_f = lam[lo_i:hi_i], f[lo_i:hi_i]
    s = float(seg_f.sum())
    assert s > 0, "empty MP central window (γ or quantiles degenerate)"
    return float((seg_l * seg_f).sum() / s)


def fit_sigma2(eigs, gamma, q_lo=0.25, q_hi=0.75):
    """Fit the MP noise scale σ² on the CENTRAL quantile range of the eigenvalues. σ² =
    (empirical mean of λ_i in [q_lo,q_hi]) / (MP central-quantile mean at σ²=1). Outlier-robust:
    above-edge outliers live above q_hi and never enter the fit. fp64."""
    import numpy as np
    e = np.sort(np.asarray(eigs, dtype=np.float64))
    lo, hi = np.quantile(e, q_lo), np.quantile(e, q_hi)
    central = e[(e >= lo) & (e <= hi)]
    assert central.size > 0, "empty central quantile window"
    return float(central.mean() / _mp_central_mean(gamma, q_lo, q_hi))


def mp_ks(eigs, sigma2, gamma, edge):
    """KS distance between the empirical CDF of the BULK eigenvalues (λ_i ≤ edge) and the fitted MP
    CDF. The §6.2 fit-quality statistic; large ⇒ the MP reference class is wrong (heavy-tailed /
    structured spectrum). fp64."""
    import numpy as np
    lam, f, cdf = _mp_grid(gamma, sigma2, npts=4000)
    bulk = np.sort(np.asarray(eigs, dtype=np.float64))
    bulk = bulk[bulk <= edge]
    if bulk.size == 0:
        return 0.0
    Femp = np.arange(1, bulk.size + 1, dtype=np.float64) / bulk.size
    Fmp = np.interp(bulk, lam, cdf)
    return float(np.max(np.abs(Femp - Fmp)))


# ---------------------------------------------------------------------------
# per-matrix bulk/outlier decomposition (real + entry-permutation null).
# ---------------------------------------------------------------------------
def spectrum(dW):
    """The singular values of ΔW (length p=min(m,n)), fp64. Thin wrapper over the reuse-allowed
    latablate.singular_spectrum (ESD extraction, spec §14)."""
    import numpy as np
    from infra.pg import latablate as LA
    return np.asarray(LA.singular_spectrum(dW)["singular_values"], dtype=np.float64)


def eigenvalues(sv, m, n):
    """The Gram eigenvalues λ_i = σ_i²/N (N=max(m,n)) and the aspect ratio γ=min/max. These are the
    quantities the MP law describes (see the module map)."""
    import numpy as np
    s = np.asarray(sv, dtype=np.float64)
    N, p = max(m, n), min(m, n)
    assert s.size == p, f"spectrum length {s.size} != min(m,n)={p}"
    return (s ** 2) / N, p / N


def _e_out(sv, edge_eig, N):
    """(n_out, E_out) above an eigenvalue threshold: n_out = #{σ_i²/N > edge}, E_out = fraction of
    ‖ΔW‖²_F carried by those singular values."""
    import numpy as np
    s = np.asarray(sv, dtype=np.float64)
    mask = (s ** 2 / N) > edge_eig
    total = float((s ** 2).sum())
    e = float((s[mask] ** 2).sum() / total) if total > 0 else 0.0
    return int(mask.sum()), e


def analyze_matrix(dW, n_perm=3, seed=1337, q_lo=0.25, q_hi=0.75, buf_pct=50.0):
    """Full per-matrix RMT decomposition of one ΔW.

    Fits the MP bulk (σ², edges), runs ≥3 entry-permutation draws (each re-fit with the IDENTICAL
    pipeline) to calibrate the edge-fluctuation BUFFER (the `buf_pct` percentile of the permutation
    top-eigenvalue overshoot above its own fitted edge) and the permutation E_out null band, then
    counts real outliers above (λ+ + buffer). The permutation is the decisive control; the MP fit is
    scaffolding whose validity is gated by KS (real bulk KS vs the max permutation KS).

    Returns a JSON-able dict:
      shape, gamma, sigma2, lambda_minus, lambda_plus, buffer, edge (=λ+ + buffer);
      n_out, e_out            — REAL, at the buffered edge;
      e_out_perm, n_out_perm  — per-permutation, at each permutation's OWN buffered edge (null);
      e_out_perm_median;
      ks, ks_perm, ks_bound (=max perm KS), mp_reject (=ks > ks_bound).
    """
    import numpy as np
    from infra.pg import gauge_cs as CS
    A = np.asarray(dW, dtype=np.float64)
    m, n = A.shape
    N = max(m, n)

    sv = spectrum(A)
    eigs, gamma = eigenvalues(sv, m, n)
    sigma2 = fit_sigma2(eigs, gamma, q_lo, q_hi)
    lm, lp = mp_edges(sigma2, gamma)

    # entry-permutation null: re-fit each draw's own edge; record top-eig overshoot + bulk KS.
    perm = []
    overshoots, ks_perm = [], []
    for j in range(n_perm):
        Ap = CS.permute_entries(A, seed + 1 + j)
        svp = spectrum(Ap)
        eigp, _ = eigenvalues(svp, m, n)
        s2p = fit_sigma2(eigp, gamma, q_lo, q_hi)
        _lmp, lpp = mp_edges(s2p, gamma)
        overshoots.append(float(eigp.max() - lpp))
        ks_perm.append(mp_ks(eigp, s2p, gamma, lpp))
        perm.append((svp, lpp))

    buffer = max(0.0, float(np.percentile(overshoots, buf_pct)))
    edge = lp + buffer
    n_out, e_out = _e_out(sv, edge, N)

    e_out_perm, n_out_perm = [], []
    for svp, lpp in perm:
        no, eo = _e_out(svp, lpp + buffer, N)
        n_out_perm.append(no)
        e_out_perm.append(eo)

    ks_real = mp_ks(eigs, sigma2, gamma, lp)
    ks_bound = float(max(ks_perm)) if ks_perm else float("inf")
    return {"shape": [int(m), int(n)], "gamma": float(gamma), "sigma2": float(sigma2),
            "lambda_minus": float(lm), "lambda_plus": float(lp), "buffer": float(buffer),
            "edge": float(edge), "n_out": n_out, "e_out": e_out,
            "e_out_perm": [float(x) for x in e_out_perm], "n_out_perm": n_out_perm,
            "e_out_perm_median": float(np.median(e_out_perm)),
            "ks": float(ks_real), "ks_perm": [float(x) for x in ks_perm],
            "ks_bound": ks_bound, "mp_reject": bool(ks_real > ks_bound)}


# ---------------------------------------------------------------------------
# outlier-subspace reproducibility vs the Haar overlap floor (reuse_estimators).
# ---------------------------------------------------------------------------
def outlier_subspaces(dW, k):
    """(U_k, V_k) — the top-k LEFT (m×k) and RIGHT (n×k) singular vectors of ΔW, orthonormal
    columns. The 'outlier subspaces' whose cross-FT reproducibility is the §6.2 statistic."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    U, _s, Vh = np.linalg.svd(A, full_matrices=False)
    k = int(min(k, U.shape[1]))
    return U[:, :k], Vh[:k, :].T


def subspace_overlap(Q1, Q2):
    """Mean squared cosine of the principal angles between two orthonormal frames Q1,Q2 (same
    ambient dim, same k columns) = ‖Q1ᵀQ2‖²_F / k ∈ [0,1]. 1 = identical span, ≈k/d = random."""
    import numpy as np
    A = np.asarray(Q1, dtype=np.float64)
    B = np.asarray(Q2, dtype=np.float64)
    k = A.shape[1]
    assert k == B.shape[1] and k > 0, "overlap needs equal, non-empty column counts"
    return float(np.sum((A.T @ B) ** 2)) / k


def mean_pairwise_overlap(dW_list, k, side="left"):
    """Mean over all i<j FT pairs of the top-k outlier-subspace overlap (mean squared principal-angle
    cosine). side ∈ {'left','right'}. Returns {'overlap', 'n_pairs', 'k', 'side'}. A high value ⇒ the
    outlier directions are a REPRODUCIBLE family feature, not FT-idiosyncratic."""
    import numpy as np
    idx = 0 if side == "left" else 1
    subs = [outlier_subspaces(A, k)[idx] for A in dW_list]
    ov, npairs = [], 0
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            ov.append(subspace_overlap(subs[i], subs[j]))
            npairs += 1
    assert npairs > 0, "need ≥2 FTs for a pairwise overlap"
    return {"overlap": float(np.mean(ov)), "n_pairs": npairs, "k": int(k), "side": side}


def haar_overlap_floor(k, d, n_draws=1000, seed=1337):
    """The Haar overlap floor for two INDEPENDENT random k-frames in ℝ^d — REUSED from
    reuse_estimators.haar_null via its ρ_L estimand. A flat rank-k spectrum (ΔW₂ = a k-dim
    orthogonal projector) makes ρ_L = ‖(random k-frame)ᵀ(fixed k-frame)‖²/k, i.e. EXACTLY the mean
    squared principal-angle cosine between a random k-frame and a fixed one. Analytic mean = k/d.
    Returns the haar_null dict (mean/std/percentiles incl. p95.0 + raw samples)."""
    import numpy as np
    from infra.pg.reuse_estimators import haar_null
    spec = np.zeros(d, dtype=np.float64)          # rank-k projector spectrum: k ones, d−k zeros
    spec[:k] = 1.0
    return haar_null(spec, None, k, d, d, "rho_L", n_draws=n_draws, seed=seed)


def overlap_beats_floor(real_overlap, haar_floor):
    """§6.2 reproducibility predicate: the measured mean pairwise overlap clears the Haar floor's
    upper tail (p95). If False the finding is 'outliers exist but are FT-idiosyncratic'."""
    return bool(real_overlap > haar_floor["pcts"]["p95.0"])


# ---------------------------------------------------------------------------
# family summary + frozen decision predicates (§6.2).
# ---------------------------------------------------------------------------
def band_halfwidth(samples, lo_pct=2.5, hi_pct=97.5):
    """Half-width (p{hi}−p{lo})/2 of a pooled permutation E_out sample — the δ_RMT primitive
    (pooled over discovery cells, pre-real, by the caller)."""
    import numpy as np
    a = np.asarray(samples, dtype=np.float64)
    assert a.size >= 1, "empty permutation pool for δ_RMT"
    return float((np.percentile(a, hi_pct) - np.percentile(a, lo_pct)) / 2.0)


def discovery_clear_rmt(e_out_excess_ci_lo, delta_rmt):
    """§6.2 discovery-clear: the family-median E_out-excess CI-lower exceeds δ_RMT (the permutation
    band half-width). Pure predicate."""
    return bool(e_out_excess_ci_lo > delta_rmt)


def detection_rmt(sealed_ci_lo, delta_rmt, bh_ok, overlap_beats):
    """§6.2 Detection: a discovery-cleared matrix-type row whose SEALED-half E_out-excess CI-lo >
    δ_RMT, surviving BH (q=0.10) over the ≤3 promoted rows, AND whose outlier subspaces beat the
    Haar floor. If the overlap fails, the row is 'outliers exist but FT-idiosyncratic', NOT a family
    detection. Pure boolean predicate."""
    return bool(sealed_ci_lo > delta_rmt and bh_ok and overlap_beats)


def mp_fit_gate(mp_reject_flags):
    """§6.2 fit-quality gate. Fraction of discovery cells whose MP fit rejects (real bulk KS beyond
    the permutation-calibrated bound); abort-to-descriptive (H_non-MP) when >half reject. Returns
    (frac_reject, abort_to_descriptive)."""
    r = [bool(x) for x in mp_reject_flags]
    assert r, "no discovery cells for the fit-quality gate"
    frac = sum(r) / len(r)
    return float(frac), bool(frac > 0.5)
