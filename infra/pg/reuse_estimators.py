"""E1 — the reuse-estimator instrument (pi-trajectory spec §E1 / plan §3.2-3.3).

Measures, SIGNED, whether FT#2's update ΔW₂ reuses the directions FT#1's update ΔW₁ built.
All three reuse estimands live in ΔW₁'s SVD basis (NOT W_b's — that is the Form arm, §3.1, a
DIFFERENT frame; never narrate one edit in both). fp64 throughout (repo spectral-roundoff trap);
no try/except (fail loud); pure JSON-able returns; heavy imports inside functions.

Estimands (per-tensor, d_out ≠ d_in; U_r ∈ ℝ^{d_out×r} top-r LEFT of ΔW₁, V_r ∈ ℝ^{d_in×r} right):
  rho_L = ‖U_rᵀ ΔW₂‖²/‖ΔW₂‖²   does FT#2 write into the OUTPUT directions FT#1 built  (null r/d_out)
  rho_R = ‖ΔW₂ V_r‖²/‖ΔW₂‖²    does FT#2 read from the INPUT directions FT#1 used     (null r/d_in)
  rho_J = ‖U_rᵀ ΔW₂ V_r‖²/‖ΔW₂‖²  same OPERATOR subspace (two-sided)                 (null (r/d)² iso)
  a     = Σσ_i(ΔW₁)·sign(u_iᵀ ΔW₂ v_i) / Σσ_i   the SIGNED paired bilinear ∈ [−1,1]   (null its own band)
          extend ⇒ a>0, overwrite-undo ⇒ a<0. SVD-sign-invariant (joint flip uᵢ,vᵢ→−uᵢ,−vᵢ leaves the
          paired sign fixed; the one-vector sign⟨ΔW₂,uᵢ⟩ is NOT invariant — rejected, plan B1).

THE NULL (N1, spectrum-preserving Haar; §5).  Hold ΔW₂'s spectrum fixed, replace ΔW₁'s top-r frame by
a uniformly-random orthonormal r-frame (a random r-subspace).  KEY (rotation invariance): the energy
ΔW₂ projects onto a random r-subspace depends ONLY on ΔW₂'s singular VALUES and the ambient dim — not
on ΔW₂'s vectors.  So we draw the reduced r×ρ Gram pieces directly (A~N(0,1)^{r×ρ} = the random frame's
projection onto ΔW₂'s singular basis; E~N(0,1)^{r×(d−ρ)} = its complement) — O(r²ρ)/draw, NOT a full
d-dim Haar (which would be O(d³) and blow the budget on the 16384-wide MLP tensors).  The G-isotropic
calibration gate asserts the MC means collapse to the closed forms r/d_out, r/d_in, (r/d)².
"""
from __future__ import annotations

# Gemma-1 decoder weight-matrix projection types (the §3.2 unit of analysis: 18 layers × 7).
# Same 7 types as Gemma-2; the COUNT differs (18 vs 26) — E2b asserts the layer count explicitly.
PROJECTION_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


TAU_GAP = 1e-2          # both-sided singular-gap threshold (identifiability, plan §3.1-C13 / §E2b-6)


def forced_overlap_floor(r: int, d: int) -> float:
    """The UNAVOIDABLE projection-energy fraction: two generic r-subspaces of ℝ^d are forced to share
    ≥ max(0, 2r−d) dimensions, so even a disjoint ΔW₂ reads ρ ≥ max(0,2r−d)/d. Reported alongside
    every ρ; claims are restricted to r ≪ d where this floor ≈ 0 (plan §3.2 small-r regime, C2a)."""
    return max(0, 2 * r - d) / d


def both_sided_gaps(s):
    """g_i = min((σ_{i−1}−σ_i)/σ_i, (σ_i−σ_{i+1})/σ_i) per index — the §3.1-C13 in-block identifiability
    gap (a near-degenerate block makes the paired sign of `a` basis-dependent). Endpoints → +inf."""
    s = [float(x) for x in s]
    n = len(s)
    out = []
    for i in range(n):
        left = (s[i - 1] - s[i]) / s[i] if (i > 0 and s[i] > 0) else float("inf")
        right = (s[i] - s[i + 1]) / s[i] if (i < n - 1 and s[i] > 0) else float("inf")
        out.append(min(left, right))
    return out


def signed_a_from_svd(U, s, Vh, dW2, r):
    """The paired bilinear `a` from PRECOMPUTED ΔW₁ SVD factors (no re-SVD) — so a caller can INJECT a
    sign gauge directly and test invariance (G-sign-inv). a = Σσ_i·sign((U_rᵀ ΔW₂ V_r)_ii)/Σσ_i, top-r.
    Invariant under the JOINT flip (u_i,v_i)→(−u_i,−v_i) (the diagonal sign is unchanged); the rejected
    one-vector form would NOT be (it sees only u_i's flip)."""
    import torch
    k = int(min(r, s.shape[0]))
    Ur = U[:, :k]
    Vr = Vh[:k, :].transpose(0, 1)
    M = Ur.transpose(0, 1) @ dW2.detach().to(torch.float64) @ Vr
    ssum = s[:k].sum().item()
    return (s[:k] * torch.sign(torch.diagonal(M))).sum().item() / ssum if ssum > 0 else float("nan")


# ---------------------------------------------------------------------------
# Real (deterministic) statistics — measured in ΔW₁'s top-r SVD basis.
# ---------------------------------------------------------------------------

def reuse_stats(dW1, dW2, r):
    """All four reuse estimands of ΔW₂ in ΔW₁'s top-r basis, plus the INDEPENDENT perp check.

    fp64. One SVD of ΔW₁. ρ_⊥ is computed independently as ‖(I−U_rU_rᵀ)ΔW₂‖²/‖ΔW₂‖² (NOT 1−ρ_L), so
    |ρ_L+ρ_⊥−1| is a real fp64-identity test (G-recon), not a tautology. No d×d projector is formed —
    ρ_⊥ = ‖ΔW₂ − U_r(U_rᵀΔW₂)‖² (O(d_out·d_in·r)). Returns deterministic scalars (no CI — §9 aggregation:
    per-tensor ρ/a are point estimates judged vs the N1 band, the bootstrap is for per-layer pooling)."""
    import torch
    from infra.pg.gain_routing import _svd_fp64        # ONE SVD source (sign gauge pinned, §E1)

    A = dW1.detach().to(torch.float64)
    B = dW2.detach().to(torch.float64)
    assert A.shape == B.shape, f"ΔW₁/ΔW₂ shape mismatch: {tuple(A.shape)} vs {tuple(B.shape)}"
    d_out, d_in = A.shape
    U, s, Vh, V, rank = _svd_fp64(A)
    k = int(min(r, rank))
    Ur = U[:, :k]                                       # d_out × k
    Vr = V[:, :k]                                       # d_in × k
    s_top = s[:k]

    total = (B ** 2).sum().item()                       # ‖ΔW₂‖²_F
    UtB = Ur.transpose(0, 1) @ B                         # k × d_in
    BVr = B @ Vr                                         # d_out × k
    rho_L = (UtB ** 2).sum().item() / total if total > 0 else float("nan")
    rho_R = (BVr ** 2).sum().item() / total if total > 0 else float("nan")
    M = UtB @ Vr                                         # k × k  (= U_rᵀ ΔW₂ V_r)
    rho_J = (M ** 2).sum().item() / total if total > 0 else float("nan")

    # INDEPENDENT perp (left): residual after removing the U_r component, never 1−ρ_L.
    perp = B - Ur @ UtB                                  # d_out × d_in, energy outside U_r's span
    rho_perp_L = (perp ** 2).sum().item() / total if total > 0 else float("nan")
    recon_resid = abs(rho_L + rho_perp_L - 1.0)

    # signed paired bilinear — sign of the i-th DIAGONAL of M, weighted by ΔW₁'s σ_i.
    diagM = torch.diagonal(M)
    ssum = s_top.sum().item()
    a = (s_top * torch.sign(diagM)).sum().item() / ssum if ssum > 0 else float("nan")

    # TWO distinct degeneracy gates (plan §3.3 / spec §E2b-6):
    #  (1) ρ_L/ρ_J truncation-boundary gate — a non-unique top-r subspace (σ_r ≈ σ_{r+1}) makes the
    #      projector basis-dependent. Relative gap (σ_r−σ_{r+1})/σ_max.
    #  (2) `a` IN-BLOCK gate — the per-direction paired sign breaks inside a near-degenerate block;
    #      both-sided g_i < τ_gap flags it. Report the fraction of the top-r flagged.
    gap = ((s[k - 1] - s[k]) / s[0]).item() if (rank > k and s[0] > 0) else float("inf")
    inblock = both_sided_gaps(s_top.tolist())
    inblock_frac_flagged = sum(1 for g in inblock if g < TAU_GAP) / k if k > 0 else 0.0

    return {
        "r": int(k), "shape": [int(d_out), int(d_in)],
        "rho_L": rho_L, "rho_R": rho_R, "rho_J": rho_J, "rho_perp_L": rho_perp_L,
        "recon_resid": recon_resid, "a": a,
        "forced_floor_L": forced_overlap_floor(k, d_out),
        "forced_floor_R": forced_overlap_floor(k, d_in),
        "trunc_gap_rel": gap, "dw1_inblock_frac_flagged": inblock_frac_flagged, "tau_gap": TAU_GAP,
        "dw1_sigma_top": s_top.tolist(),
    }


def rho_L(dW1, dW2, r):
    """ρ_L + its INDEPENDENT perp (§E1 G-recon). Thin wrapper over reuse_stats."""
    st = reuse_stats(dW1, dW2, r)
    return {"rho_L": st["rho_L"], "rho_perp_L": st["rho_perp_L"], "recon_resid": st["recon_resid"]}


def rho_R(dW1, dW2, r):
    return reuse_stats(dW1, dW2, r)["rho_R"]


def rho_J(dW1, dW2, r):
    return reuse_stats(dW1, dW2, r)["rho_J"]


def signed_a(dW1, dW2, r):
    """The signed paired bilinear a ∈ [−1,1]. Thin wrapper over reuse_stats (SVD-sign-invariant)."""
    return reuse_stats(dW1, dW2, r)["a"]


# ---------------------------------------------------------------------------
# N1 — spectrum-preserving Haar null, drawn ANALYTICALLY (rotation invariance).
# ---------------------------------------------------------------------------

def haar_null(dw2_spectrum, dw1_sigma_top, r, d_out, d_in, estimand,
              n_draws=1000, seed=1337, pcts=(0.5, 2.5, 5.0, 50.0, 95.0, 97.5, 99.5)):
    """Spectrum-preserving Haar null for one estimand at rank r (§5 N1).

    Inputs are SPECTRA, not matrices — the rotation-invariance reduction means the null needs only
    ΔW₂'s singular values (`dw2_spectrum`, length ρ=min(d_out,d_in)) and, for the `a`-null, ΔW₁'s top-r
    singular values (`dw1_sigma_top`, the fixed weights). The caller (E2b/calibration) SVDs once and
    passes the spectra, so we never re-SVD per estimand×r.

    Per draw we materialise the random r-frame's projection onto ΔW₂'s singular basis as a Gaussian
    block A (r×ρ) and its ambient complement E (r×(d−ρ)); the frame's Gram is GᵀG = AAᵀ + EEᵀ (r×r):
      rho_L/rho_R : projection energy  trace((GᵀG)⁻¹ A diag(σ²) Aᵀ) / Σσ²   (ambient d = d_out / d_in)
      rho_J       : ‖C_L diag(σ) C_Rᵀ‖² / Σσ²,  C = (GᵀG)^{-1/2} A   (INDEPENDENT left/right frames)
      a           : Σ σ₁ᵢ sign( (C_L diag(σ) C_Rᵀ)ᵢᵢ ) / Σσ₁ᵢ   (the polar factor (GᵀG)^{-1/2}A is a
                    genuine Haar Stiefel frame with correct random column signs — no QR sign-fix footgun)

    Returns percentiles + mean/std + the raw samples (for empirical-p FDR) + the analytic expectation
    the G-isotropic gate cross-checks against (r/d_out, r/d_in, (r/d)², 0)."""
    import numpy as np

    sig = np.asarray(dw2_spectrum, dtype=np.float64)
    sig2 = sig ** 2
    rho = sig.shape[0]
    assert rho == min(d_out, d_in), f"spectrum length {rho} != min(d_out,d_in)={min(d_out, d_in)}"
    sumsig2 = float(sig2.sum())
    k = int(min(r, rho))
    rng = np.random.default_rng(seed)
    s1 = np.asarray(dw1_sigma_top, dtype=np.float64)[:k] if estimand == "a" else None

    # BATCHED over draws (stacked numpy linalg) — the per-draw Python loop was the dominant cost at the
    # wide-MLP shapes (16384-row ⇒ a 64×14336 complement E per draw × 1000 × 2016 calls). Batch size is
    # bounded so the largest stacked E (nb×k×c) stays ≈0.5 GB. Identical math + distribution to the loop.
    cmax = max(d_out, d_in) - rho
    BATCH = max(1, min(256, n_draws, 64_000_000 // (k * (max(cmax, rho) + 1))))

    def _gram_batch(amb, nb):
        """nb random frames in ambient `amb`: A (nb×k×ρ) onto ΔW₂'s singular basis + complement
        E (nb×k×(amb−ρ)); returns A and stacked GᵀG = AAᵀ+EEᵀ (nb×k×k)."""
        A = rng.standard_normal((nb, k, rho))
        G = A @ A.transpose(0, 2, 1)
        c = amb - rho
        if c > 0:
            E = rng.standard_normal((nb, k, c))
            G = G + E @ E.transpose(0, 2, 1)
        return A, G

    def _polar_batch(A, G):
        """C = (GᵀG)^{-1/2} A per draw (the polar/symmetric-orthonormalisation Haar factor), stacked."""
        w, Q = np.linalg.eigh(G)                        # (nb,k), (nb,k,k)
        Ginvsqrt = (Q * (1.0 / np.sqrt(w))[:, None, :]) @ Q.transpose(0, 2, 1)
        return Ginvsqrt @ A                             # (nb,k,ρ)

    vals = np.empty(n_draws, dtype=np.float64)
    done = 0
    while done < n_draws:
        nb = min(BATCH, n_draws - done)
        if estimand in ("rho_L", "rho_R"):
            amb = d_out if estimand == "rho_L" else d_in
            A, G = _gram_batch(amb, nb)
            HHt = (A * sig2[None, None, :]) @ A.transpose(0, 2, 1)   # (nb,k,k) = A diag(σ²) Aᵀ
            v = np.trace(np.linalg.solve(G, HHt), axis1=1, axis2=2) / sumsig2
        elif estimand == "rho_J":
            AL, GL = _gram_batch(d_out, nb)
            AR, GR = _gram_batch(d_in, nb)
            CL, CR = _polar_batch(AL, GL), _polar_batch(AR, GR)
            P = (CL * sig[None, None, :]) @ CR.transpose(0, 2, 1)    # (nb,k,k) = C_L diag(σ) C_Rᵀ
            v = (P ** 2).sum(axis=(1, 2)) / sumsig2
        elif estimand == "a":
            AL, GL = _gram_batch(d_out, nb)
            AR, GR = _gram_batch(d_in, nb)
            CL, CR = _polar_batch(AL, GL), _polar_batch(AR, GR)
            diag = (CL * sig[None, None, :] * CR).sum(axis=2)        # (nb,k) = diag(C_L diag(σ) C_Rᵀ)
            v = (s1[None, :] * np.sign(diag)).sum(axis=1) / s1.sum()
        else:
            raise ValueError(f"unknown estimand {estimand!r}")
        vals[done:done + nb] = v
        done += nb

    analytic = {"rho_L": k / d_out, "rho_R": k / d_in,
                "rho_J": (k / d_out) * (k / d_in), "a": 0.0}[estimand]
    return {
        "estimand": estimand, "r": k, "shape": [int(d_out), int(d_in)],
        "n_draws": int(n_draws), "seed": int(seed),
        "mean": float(vals.mean()), "std": float(vals.std()),
        "pcts": {f"p{p}": float(np.percentile(vals, p)) for p in pcts},
        "analytic": analytic, "samples": vals.tolist(),
    }


def haar_null_for(dW1, dW2, r, estimand, n_draws=1000, seed=1337):
    """Convenience wrapper: SVD ΔW₁/ΔW₂ then call haar_null. (E2b prefers the spectra form to cache
    the SVDs across estimand×r; calibration uses this for readability on small matrices.)"""
    import torch
    from infra.pg.gain_routing import _svd_fp64
    sig2 = torch.linalg.svdvals(dW2.detach().to(torch.float64)).tolist()
    U, s, Vh, V, rank = _svd_fp64(dW1.detach().to(torch.float64))
    d_out, d_in = dW1.shape
    return haar_null(sig2, s[:r].tolist(), r, int(d_out), int(d_in), estimand, n_draws, seed)


# ---------------------------------------------------------------------------
# N3 — calibration ladder (positive control; subspace alignment swept, spectrum fixed).
# ---------------------------------------------------------------------------

def reuse_ladder(dW1, r, alpha, seed, ref_spectrum=None):
    """Build ΔW₂(α) whose top-r LEFT energy aligns with ΔW₁'s top-r left span by a controlled amount α,
    with ‖ΔW₂(α)‖ and its FULL singular spectrum held FIXED at every α (plan §5 N3, C19).

    carrier(α) = [α·U_rU_rᵀ + (1−α)·(I−U_rU_rᵀ)] · B   for one fixed random B (seed) — the in-span and
    orthogonal carriers are the two complementary halves of a SINGLE matrix, so only the alignment moves.
    Then SVD-replace-Σ: ΔW₂(α) = Ũ diag(σ_ref) Ṽᵀ, σ_ref a fixed reference (ΔW₁'s spectrum by default).
    ⇒ ‖ΔW₂(α)‖ and the spectrum are α-invariant; ONLY ρ_L can move with α (G-ladder asserts monotone ↑)."""
    import torch
    from infra.pg.gain_routing import _svd_fp64

    A = dW1.detach().to(torch.float64)
    d_out, d_in = A.shape
    U, s, Vh, V, rank = _svd_fp64(A)
    k = int(min(r, rank))
    Ur = U[:, :k]
    g = torch.Generator().manual_seed(int(seed))
    B = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    UrUrtB = Ur @ (Ur.transpose(0, 1) @ B)                      # in-span (left) part of B
    carrier = alpha * UrUrtB + (1.0 - alpha) * (B - UrUrtB)     # α·in-span + (1−α)·orthogonal
    Uc, sc, Vhc, Vc, rc = _svd_fp64(carrier)
    ref = s.clone() if ref_spectrum is None else torch.as_tensor(ref_spectrum, dtype=torch.float64)
    m = int(min(rc, ref.shape[0]))
    return Uc[:, :m] @ torch.diag(ref[:m]) @ Vhc[:m, :]         # spectrum-fixed, alignment ∝ α


# ---------------------------------------------------------------------------
# §4 decision waterfall — the pre-registered bins (frozen in E1, applied in E2b).
# ---------------------------------------------------------------------------

def classify_reuse(rho_L_val, a_val, haar_L, haar_a, forced_floor_L, a_halfwidth, margin_k=1.0):
    """Bin one (tensor, r) into extend / overwrite-undo / build_disjoint / partial_reuse (spec §E2b-10).

    Bins are EXHAUSTIVE + EXCLUSIVE (waterfall):
      build_disjoint : ρ_L ≤ Haar-p95  AND  ρ_L ≤ forced-floor                 (no reuse above chance)
      elif undo      : a-upper-bound < 0   (and ρ_L clears Haar-p95)            (FT#2 reverses FT#1)
      elif extend    : a-lower-bound > 0   AND  (ρ_L−mean)/std > margin_k       (FT#2 reinforces FT#1)
      else           : partial_reuse                                           (incl. alternating a≈0)

    G-power: where E1's signed-a half-width > 0.5 at this r, `a` is too blunt — the signed call is
    suppressed and only ρ_L-vs-Haar is reported (`a_resolvable=False`). haar_L carries p95/mean/std;
    haar_a carries the a-band {p2.5,p97.5}; a_halfwidth is the E1 G-power half-width at this r."""
    p95 = haar_L["pcts"]["p95.0"]
    mean, std = haar_L["mean"], haar_L["std"]
    a_lb, a_ub = haar_a["pcts"]["p2.5"], haar_a["pcts"]["p97.5"]
    rho_clears = rho_L_val > p95
    a_resolvable = a_halfwidth <= 0.5
    margin = (rho_L_val - mean) / std if std > 0 else float("inf")

    # Waterfall (exhaustive + exclusive). disjoint ⟺ ρ_L does NOT clear the Haar-p95 null (no reuse above
    # chance) — matching plan §4 "ρ_L ≤ shared-substrate level ∧ a≈null" (when ρ_L is at the null, a is too).
    # forced_floor_L is an ANNOTATION, NOT a trigger: at small r it is 0, so requiring ρ_L ≤ forced_floor
    # would make build_disjoint unreachable for any real ΔW₂ (critic CRITICAL, confirmed by G-cuts).
    if not rho_clears:
        b = "build_disjoint"
    elif a_resolvable and (a_val < a_lb) and (a_val < 0):
        b = "overwrite_undo"
    elif a_resolvable and (a_val > a_ub) and (a_val > 0) and (margin > margin_k):
        b = "extend"
    else:
        b = "partial_reuse"
    return {"bin": b, "a_resolvable": bool(a_resolvable), "rho_clears_haar": bool(rho_clears),
            "rho_margin_sigma": float(margin), "a_lb": float(a_lb), "a_ub": float(a_ub),
            "a_halfwidth": float(a_halfwidth), "forced_floor_L": float(forced_floor_L),
            "rho_exceeds_forced_floor": bool(rho_L_val > forced_floor_L)}
