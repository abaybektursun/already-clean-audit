"""E1 — synthetic calibration battery for the reuse-estimator instrument (spec §E1).

Proves, on synthetic matrices with KNOWN answers, that reuse_estimators is (a) numerically correct AND
(b) powerful enough to read the signed trichotomy at the small r the plan mandates — a HALT precondition
before any real ΔW is computed. Each item is a pre-registered gate {threshold, observed, pass}. fp64; no
try/except; pinned SVD source (gain_routing._svd_fp64) for BOTH constructing the synthetic items and
testing them, so the per-column sign gauge cannot differ between build and test.

Shapes span Gemma-1 geometry incl. the real MQA o-like transpose orientation 2048×256 (so a d_out≠d_in
transpose bug is caught on a shape that maps to a real tensor) and a strongly non-square shape.
"""
from __future__ import annotations

import math


def _svd(w):
    from infra.pg.gain_routing import _svd_fp64
    return _svd_fp64(w)                                 # U(m×r), s(r), Vh(r×n), V(n×r), r


def _randmat(shape, seed):
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape[0], shape[1], generator=g, dtype=torch.float64)


def _rank_r_with_spectrum(shape, r, spectrum, seed):
    """A rank-r matrix U_r diag(spectrum) V_rᵀ with random orthonormal U_r,V_r (fixed seed)."""
    import torch
    m, n = shape
    g = torch.Generator().manual_seed(int(seed))
    Uf, _ = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64))
    Vf, _ = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=torch.float64))
    s = torch.as_tensor(spectrum, dtype=torch.float64)[:r]
    return Uf @ torch.diag(s) @ Vf.transpose(0, 1)


# ===========================================================================
# The pre-registered gates (the both-sided-gap helper is canonical in reuse_estimators).
# ===========================================================================

def g_recon(seed):
    """ρ_L + ρ_⊥(independent) = 1 to fp64 across ALL orientations (square / wide / tall / non-square)."""
    from infra.pg import reuse_estimators as re
    shapes = [(512, 512), (256, 2048), (2048, 256), (512, 4096)]
    worst = 0.0
    for i, sh in enumerate(shapes):
        dW1, dW2 = _randmat(sh, seed + i), _randmat(sh, seed + 100 + i)
        for r in (8, 32, 64):
            worst = max(worst, re.reuse_stats(dW1, dW2, r)["recon_resid"])
    return {"threshold": 1e-10, "observed": worst, "pass": worst < 1e-10,
            "note": "max|ρ_L+ρ_⊥−1| over 4 shapes × 3 r; ρ_⊥ computed independently"}


def g_xcheck(seed):
    """Cross-check ρ_L against the EXISTING directed projector-energy code (different implementation):
    delta_energy_in_base_subspace(ΔW₁, ΔW₁+ΔW₂) frac_energy(r) ≡ ‖U_r(ΔW₁)ᵀ ΔW₂‖²/‖ΔW₂‖² = ρ_L."""
    from infra.pg import reuse_estimators as re
    from infra.pg.w8_weight_activation import delta_energy_in_base_subspace
    sh = (2048, 256)
    dW1, dW2 = _randmat(sh, seed), _randmat(sh, seed + 7)
    de = delta_energy_in_base_subspace(dW1.numpy(), (dW1 + dW2).numpy(), ranks=(8, 32, 64), n_random=2)
    worst = 0.0
    for r in (8, 32, 64):
        mine = re.reuse_stats(dW1, dW2, r)["rho_L"]
        worst = max(worst, abs(mine - de["frac_energy_in_base_topr_real"][str(r)]))
    return {"threshold": 1e-10, "observed": worst, "pass": worst < 1e-10,
            "note": "|ρ_L(reuse_estimators) − frac_energy(w8_weight_activation)| (Gram-SVD vs thin-SVD)"}


def g_ladder(seed):
    """ρ_L(α) monotone ↑ with swing ≥0.5; ‖ΔW₂(α)‖ and top-r σ constant (spectrum-replace holds energy)."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (2048, 256), 32                              # TALL: U_r^⊥ has room → ρ_L(0)=0 reachable
    dW1 = _randmat(sh, seed)
    rho = min(sh)
    ref = torch.tensor([math.exp(-3.0 * i / r) for i in range(rho)], dtype=torch.float64)  # steep → top-r heavy
    alphas = [0.0, 0.25, 0.5, 0.75, 0.95]
    rhoLs, norms, topsig = [], [], []
    for a in alphas:
        dW2 = re.reuse_ladder(dW1, r, a, seed + 1, ref_spectrum=ref)
        rhoLs.append(re.reuse_stats(dW1, dW2, r)["rho_L"])
        norms.append(dW2.norm().item())
        topsig.append(torch.linalg.svdvals(dW2)[:r].tolist())
    mono = all(rhoLs[i + 1] >= rhoLs[i] - 1e-9 for i in range(len(rhoLs) - 1))
    swing = rhoLs[-1] - rhoLs[0]
    norm_const = max(norms) - min(norms)
    sig_const = max(max(abs(topsig[i][j] - topsig[0][j]) for j in range(r)) for i in range(len(topsig)))
    ok = mono and swing >= 0.5 and norm_const < 1e-8 and sig_const < 1e-8
    return {"threshold": "mono ∧ swing≥0.5 ∧ |Δ‖·‖|<1e-8 ∧ |Δσ|<1e-8",
            "observed": {"rho_L": rhoLs, "swing": swing, "norm_spread": norm_const, "sigma_spread": sig_const},
            "pass": bool(ok), "note": "ρ_L tracks alignment not energy — N1 energy confound controlled"}


def g_sign_inv(seed):
    """INJECT a sign gauge directly into `a` (via signed_a_from_svd — NO re-SVD, which would silently
    recover the canonical gauge and make the test a tautology). Two checks, on a square AND the MQA
    (256,2048) orientation: (1) the JOINT flip (u_i,v_i)→(−u_i,−v_i) leaves a unchanged <1e-12 — the
    rejected one-vector form sign⟨ΔW₂,u_i⟩ would NOT (it sees only u_i's flip); (2) an ALL-LEFT flip
    NEGATES a (a→−a), proving `a` responds to relative sign (it IS the paired bilinear, not gauge-blind).
    dW2 = dW1 + small noise so a0 is strongly positive → the negation is an unambiguous change."""
    from infra.pg import reuse_estimators as re
    import torch
    out, ok = {}, True
    for sh in [(512, 512), (256, 2048)]:
        dW1 = _randmat(sh, seed)
        dW2 = dW1 + 0.1 * _randmat(sh, seed + 3)                    # a0 ≈ +1 (signs aligned with dW1)
        U, s, Vh, V, rank = _svd(dW1)
        r = 32
        a0 = re.signed_a_from_svd(U, s, Vh, dW2, r)
        g = torch.Generator().manual_seed(seed + 9)
        worst_joint = 0.0
        for _ in range(100):
            flip = torch.randint(0, 2, (rank,), generator=g, dtype=torch.float64) * 2 - 1
            aj = re.signed_a_from_svd(U * flip, s, Vh * flip[:, None], dW2, r)   # JOINT flip
            worst_joint = max(worst_joint, abs(aj - a0))
        a_left = re.signed_a_from_svd(-U, s, Vh, dW2, r)            # ALL-LEFT flip → a → −a
        negates = abs(a_left + a0) < 1e-12 and abs(a0) > 0.05
        out[f"{sh[0]}x{sh[1]}"] = {"a0": a0, "worst_joint_dev": worst_joint, "a_left": a_left,
                                   "negates": bool(negates)}
        ok = ok and worst_joint < 1e-12 and negates
    return {"threshold": "joint-flip dev<1e-12 (gauge-invariant) ∧ all-left flip negates a (paired-bilinear)",
            "observed": out, "pass": bool(ok),
            "note": "distinguishes the paired bilinear from the rejected one-vector form (NOT a tautology)"}


def g_altern(seed, n_draws):
    """Alternating ΔW₂=Σ(−1)ⁱσ_i u_i v_iᵀ: ρ_L≥0.95 (lives in span) BUT a within its N1 band → partial/mixed,
    never disjoint. The trap a high-reuse-yet-a≈0 case that a sign-blind ρ_L alone would misread."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (1024, 256), 64
    s_in = torch.linspace(2.0, 1.0, r, dtype=torch.float64)         # rank-r dW1 (top-r captures all energy)
    dW1 = _rank_r_with_spectrum(sh, r, s_in, seed)
    U, s, Vh, V, rank = _svd(dW1)
    signs = torch.tensor([(-1.0) ** i for i in range(r)], dtype=torch.float64)
    dW2 = (U[:, :r] * (s[:r] * signs)) @ Vh[:r, :]                  # Σ(−1)ⁱσ_i u_i v_iᵀ
    st = re.reuse_stats(dW1, dW2, r)
    band = re.haar_null(torch.linalg.svdvals(dW2).tolist(), s[:r].tolist(), r, sh[0], sh[1], "a", n_draws, seed)
    a_in_band = band["pcts"]["p2.5"] <= st["a"] <= band["pcts"]["p97.5"]
    ok = st["rho_L"] >= 0.95 and a_in_band
    return {"threshold": "ρ_L≥0.95 ∧ a∈N1[p2.5,p97.5]",
            "observed": {"rho_L": st["rho_L"], "a": st["a"], "a_band": [band["pcts"]["p2.5"], band["pcts"]["p97.5"]]},
            "pass": bool(ok), "note": "alternating reuse → high ρ_L with a≈0 (partial/mixed, not disjoint)"}


def g_adversarial(seed, n_draws):
    """Spectrum-matched-orthogonal ΔW₂ (ΔW₁'s spectrum, but left/right vecs ⊥ ΔW₁'s top-r): ρ_L WITHIN the
    N1 band, not above p97.5. A disjoint edit carrying ΔW₁'s spectrum must NOT fake reuse."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (2048, 256), 32
    dW1 = _randmat(sh, seed)
    U, s, Vh, V, rank = _svd(dW1)
    m, n = sh
    g = torch.Generator().manual_seed(seed + 5)
    # left frame ⊥ ALL of U (hence ⊥ top-r): orthonormalise random off U's span
    Z = torch.randn(m, rank, generator=g, dtype=torch.float64)
    Z = Z - U @ (U.transpose(0, 1) @ Z)
    Uo, _ = torch.linalg.qr(Z)                                      # ⊥ U
    Zr = torch.randn(n, rank, generator=g, dtype=torch.float64)
    Vo, _ = torch.linalg.qr(Zr)
    dW2 = (Uo[:, :rank] * s) @ Vo[:, :rank].transpose(0, 1)         # same spectrum, LEFT-orthogonal subspace
    st = re.reuse_stats(dW1, dW2, r)
    sig2 = torch.linalg.svdvals(dW2).tolist()
    bL = re.haar_null(sig2, s[:r].tolist(), r, m, n, "rho_L", n_draws, seed)
    bJ = re.haar_null(sig2, s[:r].tolist(), r, m, n, "rho_J", n_draws, seed)
    # left ⊥ U ⇒ both ρ_L and ρ_J read at the null by construction (U_rᵀΔW₂=0). ρ_R is NOT controlled on
    # this tall (2048×256) shape — V is full-rank, no right-orthogonal room — so it is reported, not gated.
    ok = st["rho_L"] <= bL["pcts"]["p97.5"] and st["rho_J"] <= bJ["pcts"]["p97.5"]
    return {"threshold": "ρ_L ≤ N1-p97.5 ∧ ρ_J ≤ N1-p97.5",
            "observed": {"rho_L": st["rho_L"], "rho_J": st["rho_J"], "rho_R_uncontrolled": st["rho_R"],
                         "p97.5_L": bL["pcts"]["p97.5"], "p97.5_J": bJ["pcts"]["p97.5"]},
            "pass": bool(ok), "note": "spectrum-matched LEFT-orthogonal edit reads at the null (ρ_L,ρ_J), not above"}


def g_degeneracy(seed):
    """Flat-spectrum ΔW₁: ≥99% indices flag the both-sided gap g_i < τ_gap → identifiability gate fires."""
    import torch
    sh, r = (512, 512), 64
    tau_gap = 1e-2
    from infra.pg.reuse_estimators import both_sided_gaps
    flat = _rank_r_with_spectrum(sh, r, torch.ones(r, dtype=torch.float64), seed)  # all σ equal
    U, s, Vh, V, rank = _svd(flat)
    gaps = both_sided_gaps(s[:r].tolist())
    frac_flagged = sum(1 for gv in gaps if gv < tau_gap) / r
    return {"threshold": "≥0.99 indices flagged (g_i<τ_gap=1e-2)", "observed": {"frac_flagged": frac_flagged},
            "pass": frac_flagged >= 0.99, "note": "near-flat spectrum trips the in-block degeneracy gate"}


def g_identity(seed):
    """ΔW₂=ΔW₁ (rank-r) → ρ_L=ρ_R=ρ_J=1, a=+1 within 1e-10 (self-reuse = full extend anchor)."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (1024, 256), 64
    dW1 = _rank_r_with_spectrum(sh, r, torch.linspace(3.0, 1.0, r, dtype=torch.float64), seed)
    st = re.reuse_stats(dW1, dW1, r)
    worst = max(abs(st["rho_L"] - 1), abs(st["rho_R"] - 1), abs(st["rho_J"] - 1), abs(st["a"] - 1))
    # off-diagonal cross-check (catches a LEFT/RIGHT projector transposition that self-identity cannot):
    # a matrix sharing ΔW₁'s RIGHT span but with LEFT span ⊥ ΔW₁'s → ρ_L≈0 (correct left projector) but
    # ρ_R≈1. A transposed ρ_L (using the right projector) would read ≈1 here.
    U, s, Vh, V, rank = _svd(dW1)
    g = torch.Generator().manual_seed(seed + 41)
    Z = torch.randn(sh[0], r, generator=g, dtype=torch.float64)
    Z = Z - U[:, :r] @ (U[:, :r].transpose(0, 1) @ Z)
    Uperp, _ = torch.linalg.qr(Z)
    dW1_perp = (Uperp[:, :r] * s[:r]) @ Vh[:r, :]              # left ⊥ ΔW₁'s, right = ΔW₁'s
    cross = re.reuse_stats(dW1, dW1_perp, r)
    cross_ok = cross["rho_L"] < 1e-6 and cross["rho_R"] > 0.99
    return {"threshold": "ρ=a=1 (<1e-10) ∧ cross ρ_L≈0,ρ_R≈1 (left projector, not transposed)",
            "observed": {"rho_L": st["rho_L"], "rho_R": st["rho_R"], "rho_J": st["rho_J"], "a": st["a"],
                         "cross_rho_L": cross["rho_L"], "cross_rho_R": cross["rho_R"]},
            "pass": bool(worst < 1e-10 and cross_ok),
            "note": "ΔW₂=ΔW₁ reads as full extend; off-diagonal cross-check pins ρ_L as the LEFT projector"}


def g_forced_floor(seed, n_draws):
    """Orthogonal ΔW₂ reads ABOVE max(0,2r−d)/d at large r=⌈0.6·min⌉, WITHIN N1 band at small r — the
    empirical justification of the small-r regime (forced-overlap floor bites only at large r)."""
    from infra.pg import reuse_estimators as re
    import torch
    sh = (512, 512)                                       # square → 2r−d>0 reachable
    d = min(sh)
    dW1 = _randmat(sh, seed)
    dW2 = _randmat(sh, seed + 11)                         # generic (≈isotropic) disjoint edit
    sig2 = torch.linalg.svdvals(dW2).tolist()
    U, s, Vh, V, rank = _svd(dW1)
    small = {}
    for r in (8, 16, 32, 64):
        st = re.reuse_stats(dW1, dW2, r)
        band = re.haar_null(sig2, s[:r].tolist(), r, sh[0], sh[1], "rho_L", n_draws, seed)
        small[r] = {"rho_L": st["rho_L"], "floor": re.forced_overlap_floor(r, d),
                    "in_band": band["pcts"]["p0.5"] <= st["rho_L"] <= band["pcts"]["p99.5"]}
    rL = math.ceil(0.6 * d)
    st_large = re.reuse_stats(dW1, dW2, rL)
    floor_large = re.forced_overlap_floor(rL, d)
    ok = all(small[r]["in_band"] for r in small) and st_large["rho_L"] > floor_large > 0
    return {"threshold": "small-r ρ_L∈N1 ∧ large-r ρ_L>floor>0",
            "observed": {"small_r": small, "large_r": {"r": rL, "rho_L": st_large["rho_L"], "floor": floor_large}},
            "pass": bool(ok), "note": "forced-overlap floor justifies restricting claims to r≪d"}


def g_isotropic(seed, n_draws):
    """≥1000 Haar draws: MC means → r/d_out, r/d_in, (r/d)² within 3·SE; per-estimand Q_L/Q_R wiring
    verified on a NON-SQUARE shape (a left/right swap would push ρ_L's mean to r/d_in — caught here)."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (256, 2048), 32                               # d_out=256 ≠ d_in=2048 → wiring is observable
    dW1 = _randmat(sh, seed)
    dW2 = _randmat(sh, seed + 13)
    sig2 = torch.linalg.svdvals(dW2).tolist()
    U, s, Vh, V, rank = _svd(dW1)
    res, ok = {}, True
    for est, analytic in (("rho_L", r / sh[0]), ("rho_R", r / sh[1]), ("rho_J", (r / sh[0]) * (r / sh[1]))):
        b = re.haar_null(sig2, s[:r].tolist(), r, sh[0], sh[1], est, n_draws, seed)
        se = b["std"] / math.sqrt(b["n_draws"])
        within = abs(b["mean"] - analytic) <= 3 * se
        res[est] = {"mean": b["mean"], "analytic": analytic, "se": se, "within_3se": within}
        ok = ok and within
    return {"threshold": "|MC mean − closed form| ≤ 3·SE for ρ_L,ρ_R,ρ_J", "observed": res,
            "pass": bool(ok), "note": "null reduces to the isotropic closed forms; L/R/J plumbing correct"}


def g_power(seed, n_draws):
    """Report the N1 signed-a 95% half-width at r∈{8,16,32,64}; pre-register: half-width >0.5 at r ⇒ the
    signed extend/undo call at that r is reported ρ_L-only / inconclusive (the resolution gate)."""
    from infra.pg import reuse_estimators as re
    import torch
    sh = (2048, 2048)                                     # square q/o geometry
    dW1 = _randmat(sh, seed)
    dW2 = _randmat(sh, seed + 17)
    sig2 = torch.linalg.svdvals(dW2).tolist()
    U, s, Vh, V, rank = _svd(dW1)
    hw = {}
    for r in (8, 16, 32, 64):
        b = re.haar_null(sig2, s[:r].tolist(), r, sh[0], sh[1], "a", n_draws, seed)
        hw[r] = (b["pcts"]["p97.5"] - b["pcts"]["p2.5"]) / 2.0
    mono = all(hw[a] >= hw[b] - 1e-9 for a, b in ((8, 16), (16, 32), (32, 64)))  # tighter as r grows
    resolvable = {r: hw[r] <= 0.5 for r in hw}
    finite = all(v == v for v in hw.values())
    # ABSOLUTE bounds (critic): mono+non-NaN alone would pass a degenerate hw≡0; pre-register that small r
    # is unresolvable (hw[8]>0.5) and large r resolvable (hw[64]<0.5) — matching the MC-verified ~0.748/0.248.
    bounds_ok = hw[8] > 0.5 and hw[64] < 0.5
    return {"threshold": "monotone↓ ∧ finite ∧ hw[8]>0.5 (small-r unresolvable) ∧ hw[64]<0.5 (large-r resolvable)",
            "observed": {"half_width": hw, "resolvable_a": resolvable},
            "pass": bool(mono and finite and bounds_ok),
            "note": "signed-trichotomy resolution pinned before it is read (G-power); abs bounds, not just mono"}


def g_cuts(seed, n_draws):
    """The §4 decision waterfall bins KNOWN ground-truth correctly: extend (+dW1), undo (−dW1), disjoint
    (exactly orthogonal, ρ_L≈0), AND a NON-degenerate disjoint (random independent ΔW₂, ρ_L≈r/d>0 but
    <p95) — the last exercises the fixed `ρ_L ≤ p95` rule with ρ_L>0, which the broken forced-floor AND
    could never reach. Freezes the DECISION RULE, not only the estimators."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (2048, 256), 32
    dW1 = _randmat(sh, seed)
    U, s, Vh, V, rank = _svd(dW1)
    m, n = sh
    g = torch.Generator().manual_seed(seed + 21)
    noise = 0.05 * _randmat(sh, seed + 23)
    Z = torch.randn(m, rank, generator=g, dtype=torch.float64); Z = Z - U @ (U.transpose(0, 1) @ Z)
    Uo, _ = torch.linalg.qr(Z); Vo, _ = torch.linalg.qr(torch.randn(n, rank, generator=g, dtype=torch.float64))
    cases = {
        "extend": (dW1 + noise, "extend"),
        "overwrite_undo": (-dW1 + noise, "overwrite_undo"),
        "build_disjoint": ((Uo[:, :rank] * s) @ Vo[:, :rank].transpose(0, 1), "build_disjoint"),
        "build_disjoint_noisy": (_randmat(sh, seed + 31), "build_disjoint"),   # ρ_L ~ r/d_out > 0, < p95
    }
    out, ok = {}, True
    for name, (dW2, expect) in cases.items():
        st = re.reuse_stats(dW1, dW2, r)
        hL = re.haar_null(torch.linalg.svdvals(dW2).tolist(), s[:r].tolist(), r, m, n, "rho_L", n_draws, seed)
        hA = re.haar_null(torch.linalg.svdvals(dW2).tolist(), s[:r].tolist(), r, m, n, "a", n_draws, seed)
        a_hw = (hA["pcts"]["p97.5"] - hA["pcts"]["p2.5"]) / 2.0
        c = re.classify_reuse(st["rho_L"], st["a"], hL, hA, st["forced_floor_L"], a_hw)
        out[name] = {"got": c["bin"], "expect": expect, "rho_L": st["rho_L"], "a": st["a"],
                     "a_resolvable": c["a_resolvable"]}
        ok = ok and (c["bin"] == expect)
    return {"threshold": "classify_reuse bins extend/undo/disjoint(+noisy) correctly",
            "observed": out, "pass": bool(ok), "note": "the §4 cuts are frozen (incl. a non-degenerate disjoint)"}


def g_mlp(seed, n_draws):
    """The 16384-wide MLP shape (gate/up_proj geometry) — the ONLY gate exercising the large-complement
    Gram path (E ~ r×14336) that E2b will hit. recon identity + isotropic null mean → r/d_out: catches a
    numerical blow-up in GtG = AAᵀ+EEᵀ when E has far more columns than A."""
    from infra.pg import reuse_estimators as re
    import torch
    sh, r = (16384, 2048), 32
    dW1, dW2 = _randmat(sh, seed), _randmat(sh, seed + 2)
    st = re.reuse_stats(dW1, dW2, r)
    sig2 = torch.linalg.svdvals(dW2.to(torch.float64)).tolist()
    U, s, Vh, V, rank = _svd(dW1)
    b = re.haar_null(sig2, s[:r].tolist(), r, sh[0], sh[1], "rho_L", n_draws, seed)
    se = b["std"] / math.sqrt(b["n_draws"])
    within = abs(b["mean"] - r / sh[0]) <= 3 * se
    ok = st["recon_resid"] < 1e-10 and within
    return {"threshold": "recon<1e-10 ∧ isotropic mean→r/d_out (3·SE) on 16384×2048",
            "observed": {"recon_resid": st["recon_resid"], "null_mean": b["mean"],
                         "analytic": r / sh[0], "se": se, "within_3se": within},
            "pass": bool(ok), "note": "large-complement Gram path (E r×14336) numerically sound"}


def run_calibration(seed=1337, n_draws=2000):
    """Run all 13 gates; return {gates, all_pass}. n_draws≥1000 for the Haar nulls (G-power/G-isotropic)."""
    gates = {
        "G-recon": g_recon(seed),
        "G-xcheck": g_xcheck(seed),
        "G-ladder": g_ladder(seed),
        "G-sign-inv": g_sign_inv(seed),
        "G-altern": g_altern(seed, n_draws),
        "G-adversarial": g_adversarial(seed, n_draws),
        "G-degeneracy": g_degeneracy(seed),
        "G-identity": g_identity(seed),
        "G-forced-floor": g_forced_floor(seed, n_draws),
        "G-isotropic": g_isotropic(seed, n_draws),
        "G-mlp": g_mlp(seed, n_draws),
        "G-power": g_power(seed, n_draws),
        "G-cuts": g_cuts(seed, n_draws),
    }
    return {"gates": gates, "all_pass": all(v["pass"] for v in gates.values()),
            "seed": seed, "n_draws": n_draws}
