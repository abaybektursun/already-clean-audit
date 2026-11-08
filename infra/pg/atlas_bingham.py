"""ATLAS T7 — the max-entropy (matrix Bingham) ensemble on Gr(k, d) at fixed expected removed
energy, and the dispersion curve it predicts from a site's spectrum alone.

The registered family: p(V) ∝ exp(θ · tr(P_V Σ)) — the maximum-entropy distribution on the
Grassmannian under an expected-removed-energy constraint; θ is the Lagrange multiplier the energy
target pins. Everything is rotation-equivariant, so we work in Σ's eigenbasis with the DIAGONAL
spectrum Λ; the predicted dispersion curve is a functional of the spectrum alone (T7's claim, and
R-N's testable content: does the max-ent ensemble reproduce the dispersion of the house
subspace-iteration/bisection sampler at matched E — fit-verified on L0/L14, PREDICTED for L8
before L8's curve is sampled).

Sampler: Hoff-style Gibbs over columns; the column conditional is a vector Bingham on the null
space of the other columns, sampled EXACTLY by rejection from an angular central Gaussian
envelope (Kent–Ganeiber–Mardia). Two structural tricks keep d = 2048 cheap without approximation:

  1. The ACG envelope draw in the constrained subspace S = null(V₋ⱼ) is the projection of a
     full-space diagonal Gaussian: g ~ N(0, Ω), z = P g / ‖P g‖ is exactly ACG(PΩP) on S's sphere.
  2. The envelope density needs (PΩP)⁺z, solved by Woodbury with Ω diagonal and a (k−1)×(k−1)
     system — never a dense complement basis, never a large eigendecomposition.

The acceptance bound is the exact 1-D maximization of e^s(1 − 2s/b)^{ν/2} over the reachable
range of s = zᵀAz (A ⪯ 0 after the shift), so acceptance is rigorous for ANY b > 0; b is tuned on
the full-space spectrum only for efficiency. The D-suite checks the k = 1 sampler against
small-d spherical quadrature and an independent importance-sampling estimate (two methods, one
distribution), and the Gibbs chain's marginal energy against the k = 1 exact case.

No try/except hiding; a sampler that cannot reach its target energy RAISES.
"""
import math

import torch

from infra.pg.bispec_core import energy_dispersion, removed_energy_fraction

MAX_REJECT = 200_000


# ---------------------------------------------------------------------------
# Vector Bingham on the sphere (diagonal A), exact rejection sampler.
# ---------------------------------------------------------------------------
def _envelope_b(a, nu):
    """Kent-style ACG tuning: b solves Σ 1/(b − 2aᵢ) = 1 over the (shifted, ≤0) spectrum a.
    Monotone-decreasing in b — bisection. `nu` is kept for signature clarity; the BRACKET must be
    sized by len(a) (the sum has len(a) terms — bracketing by the subspace dimension fails for
    constrained calls with k > ~10, caught in the first D-suite round)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    lo, hi = 1e-9, float(a.shape[0]) + 2.0 * float(-a.min()) + 10.0

    def f(b):
        return float((1.0 / (b - 2.0 * a)).sum()) - 1.0

    assert f(lo) > 0 > f(hi), "envelope root not bracketed"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _log_bound(a_min, b, nu):
    """log max over s ∈ [a_min, 0] of e^s (1 − 2s/b)^(ν/2) — the exact rejection constant."""
    s_star = (b - nu) / 2.0
    cands = [a_min, 0.0] + ([s_star] if a_min <= s_star <= 0.0 else [])
    return max(s + 0.5 * nu * math.log1p(-2.0 * s / b) for s in cands)


def sample_vector_bingham(a, gen, Q_perp=None, b=None):
    """One EXACT draw of x ∝ exp(Σ aᵢ xᵢ²) on the unit sphere, optionally constrained to
    null(Q_perp) (Q_perp d×m orthonormal). `a` is the diagonal (shifted so max(a) = 0).

    Envelope: full-space diagonal Gaussian g ~ N(0, Ω), Ω = (I − 2A/b)⁻¹, projected onto the
    constraint and normalized — exactly ACG(PΩP). Accept with the exact 1-D bound. RAISES after
    MAX_REJECT proposals (a distribution this concentrated needs a better b, not silence)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    d = a.shape[0]
    assert float(a.max()) <= 1e-12, "shift the spectrum so max(a) = 0 before sampling"
    nu = d - (0 if Q_perp is None else Q_perp.shape[1])
    if b is None:
        b = _envelope_b(a, nu)
    om_inv = 1.0 - 2.0 * a / b                      # Ω⁻¹ diagonal, ≥ 1
    log_M = _log_bound(float(a.min()), b, nu)
    for _ in range(MAX_REJECT):
        g = torch.randn(d, generator=gen, dtype=torch.float64) / om_inv.sqrt()
        if Q_perp is not None:
            g = g - Q_perp @ (Q_perp.transpose(0, 1) @ g)
        z = g / g.norm().clamp_min(1e-300)
        s = float((a * z ** 2).sum())
        # envelope density exponent: q(z) = zᵀ(PΩP)⁺z via Woodbury (diagonal Ω, small system)
        if Q_perp is None:
            q = float((z ** 2 * om_inv).sum())
        else:
            oz = om_inv * z
            QtOz = Q_perp.transpose(0, 1) @ oz
            S = Q_perp.transpose(0, 1) @ (om_inv.unsqueeze(1) * Q_perp)
            alpha = torch.linalg.solve(S, -QtOz)
            w = om_inv * (z + Q_perp @ alpha)
            q = float(z @ w)
        log_ratio = s + 0.5 * nu * math.log(q) - log_M
        assert log_ratio <= 1e-8, f"rejection bound violated ({log_ratio}) — envelope bug"
        u = torch.rand(1, generator=gen, dtype=torch.float64)
        if float(u.log()) < log_ratio:
            return z
    raise RuntimeError(f"vector-Bingham rejection failed after {MAX_REJECT} proposals (b={b})")


# ---------------------------------------------------------------------------
# Matrix Bingham on Gr(k, d): Metropolis-within-Gibbs over columns.
#
# The Kent rejection constant e^((b−ν)/2)·(ν/b)^(ν/2) EXPLODES at strong tilt in high subspace
# dimension (measured: log M ≈ 42 at θ·λ ∈ [−40, 0], ν = 37 — acceptance ~e⁻⁴², the sampler
# starves; third D-suite round). A column update does not need an independent exact draw — any
# kernel leaving the column conditional invariant makes the Gibbs sweep exact — so each column
# takes a short MH chain: ACG independence proposals (well-matched body) + geodesic random-walk
# moves at two step sizes (symmetric on the sphere, robust at any concentration). At θ = 0 the
# ACG proposal is uniform and q̃ ≡ 1, so every proposal accepts and the sweep is exactly Haar.
# ---------------------------------------------------------------------------
def _q_tilde(z, om_inv, Q_perp):
    """zᵀ(PΩP)⁺z via Woodbury (Ω diagonal; small (k−1)-system)."""
    if Q_perp is None or Q_perp.shape[1] == 0:
        return float((z ** 2 * om_inv).sum())
    oz = om_inv * z
    QtOz = Q_perp.transpose(0, 1) @ oz
    S = Q_perp.transpose(0, 1) @ (om_inv.unsqueeze(1) * Q_perp)
    alpha = torch.linalg.solve(S, -QtOz)
    return float(z @ (om_inv * (z + Q_perp @ alpha)))


def _mh_column(z, a, gen, Q_perp, om_inv, nu, n_indep=4, geo_angles=(0.3, 0.08, 0.3, 0.08)):
    """MH chain on the column conditional p(z) ∝ exp(zᵀAz) on sphere(S), S = null(Q_perp)."""
    import math as _m
    s = float((a * z ** 2).sum())
    q = _q_tilde(z, om_inv, Q_perp)
    d = a.shape[0]
    for _ in range(n_indep):
        g = torch.randn(d, generator=gen, dtype=torch.float64) / om_inv.sqrt()
        if Q_perp is not None:
            g = g - Q_perp @ (Q_perp.transpose(0, 1) @ g)
        z2 = g / g.norm().clamp_min(1e-300)
        s2 = float((a * z2 ** 2).sum())
        q2 = _q_tilde(z2, om_inv, Q_perp)
        la = (s2 - s) + 0.5 * nu * (_m.log(q2) - _m.log(q))
        if float(torch.rand(1, generator=gen).log()) < la:
            z, s, q = z2, s2, q2
    for ang in geo_angles:
        g = torch.randn(d, generator=gen, dtype=torch.float64)
        if Q_perp is not None:
            g = g - Q_perp @ (Q_perp.transpose(0, 1) @ g)
        u = g - (z @ g) * z
        un = u.norm()
        if float(un) < 1e-12:
            continue
        u = u / un
        z2 = z * _m.cos(ang) + u * _m.sin(ang)
        z2 = z2 / z2.norm()
        s2 = float((a * z2 ** 2).sum())
        if float(torch.rand(1, generator=gen).log()) < (s2 - s):
            z, s = z2, s2
            q = _q_tilde(z, om_inv, Q_perp)
    return z


def sample_grassmann_bingham(spectrum, theta, k, seed, n_sweeps=30, V0=None):
    """One frame V (d×k, orthonormal, in the EIGENBASIS coordinates) with stationary law
    p(V) ∝ exp(θ Σᵢ λᵢ ‖Vᵢ·‖²), after `n_sweeps` Metropolis-within-Gibbs sweeps from a Haar
    start. Deterministic in `seed`."""
    lam = torch.as_tensor(spectrum, dtype=torch.float64)
    d = lam.shape[0]
    a_full = theta * lam
    a = a_full - a_full.max()                        # shift: max 0 (density invariant)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if V0 is None:
        V, _ = torch.linalg.qr(torch.randn(d, k, generator=gen, dtype=torch.float64))
    else:
        V = V0.clone()
    b = _envelope_b(a, d - k + 1)
    om_inv = 1.0 - 2.0 * a / b
    nu = d - k + 1
    for _ in range(n_sweeps):
        for j in range(k):
            others = torch.cat([V[:, :j], V[:, j + 1:]], dim=1)
            V[:, j] = _mh_column(V[:, j].clone(), a, gen, others, om_inv, nu)
    return V


def energy_of_theta(spectrum, theta, k, n_draws=4, seed=11, n_sweeps=20):
    """MC mean removed-energy fraction of the Bingham ensemble at tilt θ."""
    lam = torch.as_tensor(spectrum, dtype=torch.float64)
    tr = float(lam.sum())
    es = []
    for s in range(n_draws):
        V = sample_grassmann_bingham(lam, theta, k, seed=seed + 1000 * s, n_sweeps=n_sweeps)
        es.append(float((lam * (V ** 2).sum(1)).sum() / tr))
    return sum(es) / len(es), es


def fit_theta_for_energy(spectrum, k, target_E, theta_hi=None, tol=0.015, seed=11):
    """Monotone bisection on θ for E(θ) = target. θ = 0 is Haar; θ → ∞ → top-k. Returns
    (theta, E_real, saturated): near the E_max ceiling the required tilt grows steeply, and a
    rung the MH chain cannot certify within the growth budget comes back `saturated=True` — the
    CALLER must flag and count that rung, never score it silently (one-sided-threshold rule)."""
    lam = torch.as_tensor(spectrum, dtype=torch.float64)
    E_haar = float(k * lam.mean() / lam.sum())
    assert target_E > E_haar - 0.005, f"target {target_E} below Haar {E_haar}: negative-tilt regime, out of scope"
    lo = 0.0
    hi = theta_hi if theta_hi is not None else 4.0 / max(float(lam.max()), 1e-12)
    E_hi, _ = energy_of_theta(lam, hi, k, seed=seed)
    grow = 0
    while E_hi < target_E and grow < 14:
        hi *= 4.0
        E_hi, _ = energy_of_theta(lam, hi, k, seed=seed)
        grow += 1
    if E_hi < target_E:
        return hi, E_hi, True
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        E_mid, _ = energy_of_theta(lam, mid, k, seed=seed)
        if abs(E_mid - target_E) < tol:
            return mid, E_mid, False
        if E_mid < target_E:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    E_mid, _ = energy_of_theta(lam, mid, k, seed=seed)
    return mid, E_mid, abs(E_mid - target_E) > 3 * tol


def bingham_dispersion_curve(spectrum, k, E_over_Emax_grid, n_draws=8, seed=200, n_sweeps=30):
    """The T7 prediction: energy-dispersion (mean pairwise 1 − alignment of removed-energy
    operators, the A7 metric) of the max-ent ensemble per E/E_max rung — from the spectrum alone.

    Frames are sampled in the eigenbasis; `energy_dispersion` needs Σ in the same coordinates,
    which is diag(spectrum) — exact, no basis needed. Returns per-rung dict rows.
    """
    lam = torch.as_tensor(spectrum, dtype=torch.float64)
    d = lam.shape[0]
    Sigma_diag = torch.diag(lam)
    ev_desc = torch.sort(lam, descending=True).values
    E_max = float(ev_desc[:k].sum() / lam.sum())
    rows = []
    for r in E_over_Emax_grid:
        target = r * E_max
        theta, E_real, saturated = fit_theta_for_energy(lam, k, target, seed=seed)
        frames = [sample_grassmann_bingham(lam, theta, k, seed=seed + 7000 + 1000 * s,
                                           n_sweeps=n_sweeps) for s in range(n_draws)]
        disp = energy_dispersion(frames, Sigma_diag)
        realized = [removed_energy_fraction(V, Sigma_diag) for V in frames]
        rows.append({"E_over_Emax": r, "target_E": target, "theta": theta,
                     "mc_energy_at_theta": E_real, "realized_E": realized,
                     "target_missed": bool(saturated),
                     "energy_mean_distance": disp["mean_distance"],
                     "energy_mean_alignment": disp["mean_alignment"], "n_draws": n_draws})
    return {"k": k, "d": d, "E_max_analytic": E_max, "curve": rows,
            "n_rungs_missed": sum(1 for r_ in rows if r_["target_missed"])}


# ---------------------------------------------------------------------------
# D-suite reference implementations (independent of the sampler above).
# ---------------------------------------------------------------------------
def bingham_moments_smalld_quadrature(a, n_grid=400):
    """E[xᵢ²] for the d = 3 vector Bingham by direct spherical quadrature — the analytic-side
    anchor for the k = 1 moment check (exact to quadrature error, no sampling)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    assert a.shape[0] == 3
    th = torch.linspace(0, math.pi, n_grid, dtype=torch.float64)
    ph = torch.linspace(0, 2 * math.pi, 2 * n_grid, dtype=torch.float64)
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    x = torch.stack([TH.sin() * PH.cos(), TH.sin() * PH.sin(), TH.cos()])
    w = TH.sin()
    dens = torch.exp(a[0] * x[0] ** 2 + a[1] * x[1] ** 2 + a[2] * x[2] ** 2) * w
    Z = dens.sum()
    return [float((dens * x[i] ** 2).sum() / Z) for i in range(3)]


def bingham_moments_importance(a, n=200_000, seed=3):
    """E[xᵢ²] by importance sampling with an ACG proposal and EXACT weights — the second,
    independent estimator for the k = 1 moment check (any-d)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    d = a.shape[0]
    a = a - a.max()
    b = _envelope_b(a, d)
    om_inv = 1.0 - 2.0 * a / b
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    g = torch.randn(n, d, generator=gen, dtype=torch.float64) / om_inv.sqrt()
    z = g / g.norm(dim=1, keepdim=True)
    s = (a * z ** 2).sum(1)
    q = (z ** 2 * om_inv).sum(1)
    logw = s + 0.5 * d * q.log()
    w = (logw - logw.max()).exp()
    w = w / w.sum()
    return [(float((w * z[:, i] ** 2).sum())) for i in range(d)]
