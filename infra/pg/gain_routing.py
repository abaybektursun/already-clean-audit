"""W12 — Gain vs Routing decomposition of a weight delta (prereg §3, fp64, instrument-free).

ΔW = W_after − W_before, put in W_before's SVD basis (W_before = U Σ Vᵀ); M = Uᵀ ΔW V.
Orthonormal U,V ⇒ exact energy split  ‖ΔW‖²_F = ‖M‖² + ‖ΔW_⊥‖²:

  gain          = ‖diag(M)‖²       rescale each EXISTING pairing v_i → u_i
  in-span route = ‖offdiag(M)‖²    cross-couple v_j → u_i (i≠j) among EXISTING directions
  out-of-span   = ‖ΔW‖² − ‖M‖²     energy outside W_before's span (intruder dimensions)

Metrics vs the §5 matched-random null (raw fractions are dimension-dominated — a random ΔW is
~1−1/r "routing" for free — so we ALWAYS report enrichment over the null, never the raw fraction):

  E_gain  = (gain / in_span) / (1/r) = (gain / in_span) · r      gain-enrichment over the 1/r null
  C_route = (top-k offdiag energy / offdiag energy)
            / (k(k−1)/(r(r−1)))                                  routing concentration over uniform
  f_perp  = out_of_span / ‖ΔW‖²                                  out-of-span fraction
            (meaningful only when U or V is incomplete — rectangular; ≡0 for square full-rank, §3)

fp64 MANDATE (repo spectral-roundoff trap — modal_b200_ogbench_gotchas): every SVD / projection /
energy sum runs in float64. The gain (diagonal) energy is small vs the total and is exactly where
fp32 roundoff would masquerade as signal; the §9-1c reconstruction-identity assert guards it.

Pure functions, plain JSON-able dicts, no try/except (fail loud). Heavy imports live inside the
functions so this module py_compiles on the control plane without torch installed.
"""
from __future__ import annotations


# Gemma2 decoder weight-matrix projection types (the §2 unit of analysis: 26 layers × 7).
PROJECTION_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fine_type(key: str) -> str:
    """Map a decoder state_dict key to its projection type (or 'norm'/'other')."""
    for t in PROJECTION_TYPES:
        if t in key:
            return t
    if "norm" in key:
        return "norm"
    return "other"


def matched_decoder_weight_matrices(pg_lm_sd, l0_sd):
    """Return {key: (w_before, w_after)} for the 26×7 decoder WEIGHT MATRICES (q/k/v/o/gate/up/
    down, ndim==2), aligned by key between base gemma-2-2b (l0=g2.model) and PG2 language_model
    (pg). Norms (1-D) + embeddings/lm_head excluded — the §2 unit of analysis. Fail-loud on a
    shape mismatch or a missing counterpart (architecture regression)."""
    out = {}
    for k in sorted(set(pg_lm_sd) & set(l0_sd)):
        if fine_type(k) not in PROJECTION_TYPES:
            continue
        wb, wa = l0_sd[k], pg_lm_sd[k]
        assert wb.ndim == 2 and wb.shape == wa.shape, \
            f"bad decoder matrix {k}: before {tuple(wb.shape)} after {tuple(wa.shape)}"
        out[k] = (wb, wa)
    return out


def _svd_fp64(w):
    """fp64 thin SVD of a 2-D matrix. Returns U (m×r), s (r), Vh (r×n), V (n×r), r=min(m,n)."""
    import torch
    assert w.ndim == 2, f"gain_routing needs a 2-D matrix, got shape {tuple(w.shape)}"
    U, s, Vh = torch.linalg.svd(w.detach().to(torch.float64), full_matrices=False)
    return U, s, Vh, Vh.transpose(0, 1), s.shape[0]


def decompose(w_before, w_after, top_k=None, degen_rtol=1e-3, check_recon=True):
    """Gain/routing/out-of-span energy split of ΔW = w_after − w_before in w_before's SVD basis.

    All math in fp64. Returns a JSON-able dict (energies, fractions, E_gain, optional C_route,
    f_perp, reconstruction residual, degeneracy flag). No null calibration here — that is §5
    (matched_random_delta + the controls job). C_route only when top_k is given.

    check_recon=True computes the §9-1c fp64 reconstruction identity (forms the n×n projector
    V Vᵀ — O(n³), expensive for the wide MLP matrices). Set False in the per-matrix unblind loop
    (the identity is validated once in preflight; recomputing it on every matrix is redundant).
    """
    import torch

    wb = w_before.detach().to(torch.float64)
    wa = w_after.detach().to(torch.float64)
    assert wb.shape == wa.shape, f"shape mismatch: before {tuple(wb.shape)} after {tuple(wa.shape)}"
    m, n = wb.shape
    dW = wa - wb
    total = (dW ** 2).sum().item()                              # ‖ΔW‖²_F

    U, s, Vh, V, r = _svd_fp64(wb)
    M = U.transpose(0, 1) @ dW @ V                              # r×r, exact (orthonormal U,V)
    gain = (torch.diagonal(M) ** 2).sum().item()
    in_span = (M ** 2).sum().item()                            # = ‖projection of ΔW onto span‖²
    route = in_span - gain                                     # ‖offdiag(M)‖²
    out_of_span = total - in_span

    # §9-1c reconstruction identity (NOT a no-op): U M Vᵀ must equal the INDEPENDENTLY computed
    # in-span projection (U Uᵀ) ΔW (V Vᵀ) to fp64 roundoff. If M had any fp32 intermediate the two
    # would differ by ~1e-7, blowing the 1e-10 gate — so this guards the fp64 mandate. It does NOT
    # test the energy partition (the §5 planted controls do that).
    denom = total ** 0.5
    if check_recon:
        recon = U @ M @ Vh
        dW_par = (U @ U.transpose(0, 1)) @ dW @ (V @ Vh)            # n×n projector — O(n³), expensive
        recon_residual = (recon - dW_par).norm().item() / denom if denom > 0 else 0.0
        recon_vs_full = (recon - dW).norm().item() / denom if denom > 0 else 0.0  # ≈0 ⇔ U,V complete
    else:
        recon_residual = recon_vs_full = None

    e_gain = (gain / in_span) * r if in_span > 0 else float("nan")  # E[gain/in_span|random]=1/r
    f_perp = out_of_span / total if total > 0 else float("nan")

    # degeneracy: near-equal singular values make in-block U,V rotation non-identifiable, so the
    # gain/offdiag split WITHIN a degenerate block is basis-dependent (can fake routing). Flag the
    # adjacent degenerate pairs AND bound the routing energy they could explain (degen_route_frac =
    # the off-diagonal energy at those adjacent pairs / route); if tiny, the routing signal cannot be
    # a degeneracy artifact (§5 subspace-level reporting, made quantitative).
    s_list = s.tolist()
    smax = s_list[0] if (r and s_list[0] > 0) else 1.0
    degen_idx = [i for i in range(r - 1) if (s_list[i] - s_list[i + 1]) / smax < degen_rtol]
    degen_off = sum(M[i, i + 1].item() ** 2 + M[i + 1, i].item() ** 2 for i in degen_idx)
    degen_route_frac = (degen_off / route) if route > 0 else 0.0

    out = {
        "shape": [int(m), int(n)], "rank_r": int(r), "is_square_full_rank": bool(m == n),
        "frob_dW_sq": total, "frob_dW": denom,
        "gain_energy": gain, "route_energy": route,
        "in_span_energy": in_span, "out_of_span_energy": out_of_span,
        "gain_frac": gain / total if total > 0 else float("nan"),
        "route_frac": route / total if total > 0 else float("nan"),
        "in_span_frac": in_span / total if total > 0 else float("nan"),
        "f_perp": f_perp, "E_gain": e_gain,
        "recon_residual": recon_residual, "recon_vs_full": recon_vs_full,
        "n_degenerate_pairs": len(degen_idx), "degen_route_frac": degen_route_frac,
        "degen_rtol": degen_rtol,
    }

    # C_route = concentration of OFF-DIAGONAL (routing) energy in the entries that TOUCH the top-k
    # leading directions (row<k OR col<k) — captures the top-k×top-k block AND cross-block routing
    # into/out of the leading subspace (a top-k×top-k-only metric would miss cross-block routing) —
    # normalised by the uniform random expectation. → 1 under the null; ≫1 ⇒ routing concentrated
    # in the leading directions (the Si/Jin top-subspace structure).
    if top_k is not None and route > 0 and r >= 2:
        k = int(min(top_k, r))
        band = (M[:k, :].pow(2).sum() + M[k:, :k].pow(2).sum()).item()      # entries with row<k OR col<k
        band_off = band - torch.diagonal(M)[:k].pow(2).sum().item()         # minus the k in-band diagonal
        n_band_off = 2 * k * r - k * k - k
        uniform = n_band_off / (r * (r - 1))
        out["C_route"] = (band_off / route) / uniform if uniform > 0 else float("nan")
        out["C_route_top_k"] = k

    return out


# ===========================================================================
# §5 controls — the matched-random null + the synthetic positive-control planters.
# ===========================================================================

def matched_random_delta(shape, frob_norm, generator, dtype=None):
    """§5 N3 null draw: a Gaussian ΔW matched in SHAPE and Frobenius norm to a real ΔW. Its
    expected raw split is gain ≈ 1/r, routing ≈ 1−1/r — raw 'routing-dominant' is the null by
    dimension-counting. Caller draws ≥30 for the percentile band (the controls job, not here)."""
    import torch
    m, n = shape
    z = torch.randn(m, n, generator=generator, dtype=torch.float64)
    z = z * (frob_norm / z.norm().clamp_min(1e-30))
    return z if dtype is None else z.to(dtype)


def plant_gain(w_before, generator):
    """§5 positive control (a): pure-gain ΔW = U diag(g) Vᵀ → decompose must give gain_frac ≈ 1."""
    import torch
    U, s, Vh, V, r = _svd_fp64(w_before)
    g = torch.randn(r, generator=generator, dtype=torch.float64)
    return U @ torch.diag(g) @ Vh


def plant_routing(w_before, generator):
    """§5 positive control (b): pure-routing ΔW = U R Vᵀ, R off-diagonal only → route_frac ≈ 1."""
    import torch
    U, s, Vh, V, r = _svd_fp64(w_before)
    R = torch.randn(r, r, generator=generator, dtype=torch.float64)
    R.fill_diagonal_(0.0)
    return U @ R @ Vh


def plant_out_of_span(w_before, generator):
    """§5 positive control (c): pure-out-of-span ΔW orthogonal to W_before's span → f_perp ≈ 1.
    Requires a RECTANGULAR w_before (square full-rank ⇒ complete U,V ⇒ no orthogonal complement).
    z − (U Uᵀ) z (V Vᵀ) has Uᵀ(·)V = 0 by construction, so all its energy is out-of-span."""
    import torch
    U, s, Vh, V, r = _svd_fp64(w_before)
    m, n = w_before.shape
    assert m != n, "out-of-span control needs a rectangular matrix (square full-rank ⇒ f_perp≡0)"
    z = torch.randn(m, n, generator=generator, dtype=torch.float64)
    return z - (U @ U.transpose(0, 1)) @ z @ (V @ Vh)


def null_band(shape, n_draws, seed, top_k, rank_k=None, pcts=(50.0, 95.0, 99.0, 99.5)):
    """§5 N3 matched-random-matrix null BAND for a given matrix shape, computed ANALYTICALLY.

    Rotation invariance: for a random ΔW, the split (gain/route/out-of-span) of its projection onto
    W_before's SVD basis depends ONLY on the shape, not on the (random) basis — U,V random orthonormal
    applied to an iid-Gaussian ΔW leaves M = Uᵀ ΔW V iid Gaussian. So we draw the r×r projection M and
    the out-of-span energy DIRECTLY — no per-draw SVD, no m×n matrix (the full-SVD-per-draw path was
    the controls bottleneck). σ=1 (the fractions E_gain/C_route/f_perp are scale-invariant). Returns
    percentiles + raw samples (for empirical-p FDR). The controls null_sane gate cross-checks the
    result against the analytic expectations and against the empirically-computed shapes.

    rank_k=None → full-rank Gaussian ΔW (norm-matched null, primary): M ~ N(0,1)^{r×r}; out-of-span
    ~ χ²(m·n − r²). rank_k=K → a rank-K random ΔW (§7 robustness variant): M = A'B' with A'(r×K),
    B'(K×r) iid Gaussian; ‖ΔW‖² = trace(AᵀA · BBᵀ) via the in-span + complement Gram pieces.

    Analytic expectations the band must straddle: E_gain → 1, C_route → 1,
    f_perp → 1 − r²/(m·n) (raw 'routing/out-of-span dominant' is the dimension-counting null)."""
    import numpy as np
    m, n = shape
    r = min(m, n)
    k = int(min(top_k, r))
    n_band_off = 2 * k * r - k * k - k
    uniform = n_band_off / (r * (r - 1)) if r >= 2 else float("nan")
    oos_df = m * n - r * r
    rng = np.random.default_rng(seed)
    egain, croute, fperp = [], [], []
    for _ in range(n_draws):
        if rank_k is None:
            M = rng.standard_normal((r, r))
            oos = float(rng.chisquare(oos_df)) if oos_df > 0 else 0.0
        else:
            kk = int(min(rank_k, m, n))
            Ap = rng.standard_normal((r, kk)); Bp = rng.standard_normal((kk, r))
            M = Ap @ Bp
            AtA = Ap.T @ Ap
            BBt = Bp @ Bp.T
            if m > r:
                App = rng.standard_normal((m - r, kk)); AtA = AtA + App.T @ App
            if n > r:
                Bpp = rng.standard_normal((kk, n - r)); BBt = BBt + Bpp @ Bpp.T
            oos = max(float(np.trace(AtA @ BBt)) - float((M ** 2).sum()), 0.0)
        gain = float((np.diag(M) ** 2).sum())
        in_span = float((M ** 2).sum())
        route = in_span - gain
        total = in_span + oos
        egain.append((gain / in_span) * r if in_span > 0 else float("nan"))
        fperp.append(oos / total if total > 0 else float("nan"))
        if route > 0 and r >= 2:
            band = float((M[:k, :] ** 2).sum() + (M[k:, :k] ** 2).sum())
            band_off = band - float((np.diag(M)[:k] ** 2).sum())
            croute.append((band_off / route) / uniform if uniform > 0 else float("nan"))

    def _pcts(xs):
        a = np.asarray(xs, dtype=float)
        return {f"p{p}": float(np.percentile(a, p)) for p in pcts} | {
            "mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    return {"shape": [int(m), int(n)], "rank_r": int(r), "top_k": k,
            "n_draws": int(n_draws), "seed": int(seed), "rank_k": (int(rank_k) if rank_k else None),
            "E_gain": _pcts(egain), "C_route": _pcts(croute) if croute else None,
            "f_perp": _pcts(fperp),
            "analytic_E_gain": 1.0, "analytic_C_route": 1.0,
            "analytic_f_perp": 1.0 - (r * r) / (m * n),
            "samples": {"E_gain": egain, "C_route": croute, "f_perp": fperp}}


# ===========================================================================
# §6B secondary spectra + §6C causal reconstruction.
# ===========================================================================

def spectra(w_before, w_after, top=64):
    """§6B cross-check: per matched-index Δσ_i = σ_i^after − σ_i^before, same-index left/right
    cosines |⟨u_i^b,u_i^a⟩| / |⟨v_i^b,v_i^a⟩| (the Jin/Si rotation signature), and principal-angle
    cosines between the top-`top` left subspaces. fp64. (If H_routing: small Δσ + low same-index
    cosine in the top block — values stable, vectors rotate.)"""
    import torch
    Ub, sb, Vhb, _, rb = _svd_fp64(w_before)
    Ua, sa, Vha, _, ra = _svd_fp64(w_after)
    k = int(min(top, rb, ra))
    dsigma = (sa[:k] - sb[:k]).tolist()
    cos_u = [abs(torch.dot(Ub[:, i], Ua[:, i]).item()) for i in range(k)]
    cos_v = [abs(torch.dot(Vhb[i, :], Vha[i, :]).item()) for i in range(k)]
    pa_u = torch.linalg.svdvals(Ub[:, :k].transpose(0, 1) @ Ua[:, :k]).clamp(max=1.0)
    return {"top_k": k,
            "dsigma": dsigma, "sigma_before_top": sb[:k].tolist(), "sigma_after_top": sa[:k].tolist(),
            "cos_u_same_index": cos_u, "cos_v_same_index": cos_v,
            "principal_cos_u": pa_u.tolist(),
            "mean_cos_u_topk": sum(cos_u) / k, "mean_cos_v_topk": sum(cos_v) / k}


def reconstruct(w_before, w_after, mode):
    """§6C causal reconstruction of one matrix (fp64; caller casts to the model dtype):
      'gain'    → W_before + U diag(M) Vᵀ      (freeze routing — keep only rescaled pairings)
      'routing' → W_before + U offdiag(M) Vᵀ   (freeze gain — keep only the cross-couplings)
      'in_span' → W_before + U M Vᵀ            (drop only the out-of-span energy)
      'full'    → W_after                      (the real installed weights)
    gain+routing reconstruct the in-span part of W_after; they omit the out-of-span energy (which
    is neither) — the §6C question is which IN-SPAN component alone carries the capability."""
    import torch
    if mode == "full":
        return w_after.detach().clone()
    U, s, Vh, V, r = _svd_fp64(w_before)
    wb = w_before.detach().to(torch.float64)
    M = U.transpose(0, 1) @ (w_after.detach().to(torch.float64) - wb) @ V
    if mode == "gain":
        comp = U @ torch.diag(torch.diagonal(M)) @ Vh
    elif mode == "routing":
        comp = U @ (M - torch.diag(torch.diagonal(M))) @ Vh
    elif mode == "in_span":
        comp = U @ M @ Vh
    else:
        raise ValueError(f"unknown reconstruct mode {mode!r}")
    return wb + comp
