"""SPECTER core — the susceptibility operator itself, measured matrix-free.

The object (spec §2/§3.1): the pullback Fisher/Gauss–Newton operator at a site,

    F̄ = E_batch[ Jᵀ H_out J ],   J = ∂logits/∂x_site,   H_out = diag(p) − p pᵀ  (model p),

restricted to sequence-mode probes δ_t = env_ω(t)·v — one d×d PSD operator H_ω per envelope
ω ∈ {dc, sin1, sin4, white}. Everything here is either weight-free math (Lanczos, envelopes,
spike stability, closed-form functionals, adjudicators — D-suite-testable on synthetic
operators) or a thin differentiation shim over a caller-supplied forward (FD-checked at Gate-0).

Frozen conventions (spec §3.1):
  * per-token-energy normalization: H_ω v := (1/T)·E_ωᵀ (JᵀH_out J) E_ω v per sequence, then
    mean over sequences — envelopes are unit mean-square, so ‖E_ω v‖² = T·‖v‖² and the Rayleigh
    quotient of H_ω is damage-per-unit-per-token-energy (the behavioural matching convention).
  * H_white uses FIXED seeded Rademacher sign-sets (two; agreement is a stability row); Lanczos
    on one sign-set is exact for that realized operator (disclosed randomized proxy).
  * Lanczos vectors fp32, FULL reorthogonalization, over a bf16 forward; fp64 tridiagonal solve.
  * loss weights: H_out applies at loss-carrying logit positions, scaled by the loss's own
    per-position weight (mean-CE ⇒ 1/N) — so F̄ is exactly the GGN of the scalar loss.
  * differentiation: torch.func (jvp/vjp) primary; manual double-backprop fallback
    (`ggn_vp(..., path="manual")`) is the frozen alternative (§0.5-1). Gate-0's FD check
    certifies whichever path is active.

No try/except hiding — a wrong assumption RAISES (CLAUDE.md).
"""
import contextlib
import math

import torch

from infra.pg.bispec_core import principal_angles
from infra.pg.inherit_core import _split_hidden

# ---------------------------------------------------------------------------
# Frozen constants (spec §3, §6).
# ---------------------------------------------------------------------------
MODES = ("dc", "sin1", "sin4", "white")
WHITE_SIGN_SEEDS = (211, 212)          # the two fixed Rademacher sign-sets
LANCZOS_M = 72                         # iterations per operator estimate (top-64 Ritz reported)
LANCZOS_TOPK = 64
SPIKE_OVERLAP_MIN = 0.70               # R-O1 split-half
SPIKE_RAYLEIGH_BAND = (0.5, 2.0)       # R-O1 held-out (two-sided by construction)
EIGENGAP_REL = 0.10                    # frozen eigengap clustering: split when gap > 10% of scale
PROMOTE_K = 8                          # Arm S promotion count (frozen)
N_TRACE_PROBES = 8                     # Hutchinson probes per (sub-band × mode)
N_TR2_PROBES = 4                       # probes for tr(H²) (participation ratio)
FD_REL_TOL = 1e-2                      # Gate-0 wiring check
CENSUS_N_PROBES = 16                   # kernel/bulk census draws per family


# ===========================================================================
# The streaming Fisher product at the logits: H_out u = w ⊙ (p⊙u − p(pᵀu)).
# ===========================================================================
def fisher_logits_product(logits, u, loss_w, chunk=65536):
    """(diag(p) − ppᵀ)u per position, scaled by the per-position loss weight, streaming over the
    vocab axis so no full-vocab fp32 temporary beyond one chunk exists.

    logits (T,V) any float dtype; u (T,V); loss_w (T,) with Σ over loss positions = 1 for a
    mean-CE loss (zeros elsewhere). Returns (T,V) fp32. Positions with w=0 return 0 rows —
    they are skipped entirely.
    """
    T, V = logits.shape
    w = loss_w.to(logits.device, torch.float32)
    live = w > 0
    out = torch.zeros(T, V, dtype=torch.float32, device=logits.device)
    if not bool(live.any()):
        return out
    lg = logits[live].float()
    uu = u[live].float()
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    s = torch.zeros(lg.shape[0], 1, dtype=torch.float32, device=lg.device)
    for c0 in range(0, V, chunk):
        p = (lg[:, c0:c0 + chunk] - lse).exp()
        s += (p * uu[:, c0:c0 + chunk]).sum(-1, keepdim=True)
    rows = torch.empty_like(uu)
    for c0 in range(0, V, chunk):
        p = (lg[:, c0:c0 + chunk] - lse).exp()
        rows[:, c0:c0 + chunk] = p * (uu[:, c0:c0 + chunk] - s)
    out[live] = rows * w[live].unsqueeze(1)
    return out


def fisher_quad_from_tangent(logits, u, loss_w, chunk=65536):
    """uᵀH_out u summed over weighted positions — the quadratic form from ONE jvp (no vjp).
    Returns a python float (fp64 accumulation): Σ_t w_t·(Σ_v p u² − (Σ_v p u)²)."""
    T, V = logits.shape
    w = loss_w.to(logits.device, torch.float32)
    live = w > 0
    if not bool(live.any()):
        return 0.0
    lg = logits[live].float()
    uu = u[live].float()
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    s1 = torch.zeros(lg.shape[0], dtype=torch.float64, device=lg.device)
    s2 = torch.zeros(lg.shape[0], dtype=torch.float64, device=lg.device)
    for c0 in range(0, V, chunk):
        p = (lg[:, c0:c0 + chunk] - lse).exp().double()
        uc = uu[:, c0:c0 + chunk].double()
        s1 += (p * uc).sum(-1)
        s2 += (p * uc * uc).sum(-1)
    return float((w[live].double() * (s2 - s1 ** 2)).sum())


# ===========================================================================
# Injection shims — a delta field enters the forward at a named surface.
# The holder pattern keeps the field a FUNCTIONAL INPUT (torch.func-composable).
# ===========================================================================
def make_residual_injector(layers, L):
    """(install_ctx, holder): pre-hook on layers[L] adding holder['delta'] (T,d) to every batch
    row of the residual ENTERING the layer (the house site convention; all positions)."""
    holder = {}

    def hook(_m, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        d = holder["delta"]
        assert d.shape[-2:] == hs.shape[-2:], f"delta {tuple(d.shape)} vs hidden {tuple(hs.shape)}"
        return rebuild(hs + d.to(hs.dtype).unsqueeze(0) if d.dim() == 2 else hs + d.to(hs.dtype))

    @contextlib.contextmanager
    def install():
        h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
        yield
        h.remove()
    return install, holder


def make_proj_injector(layer, attr):
    """(install_ctx, holder): forward hook on layers[L].self_attn.<attr> adding holder['delta']
    (T,width) to the projection output — the channel-census surface."""
    holder = {}

    def hook(_m, _a, out):
        d = holder["delta"]
        assert d.shape[-1] == out.shape[-1]
        return out + d.to(out.dtype).unsqueeze(0) if d.dim() == 2 else out + d.to(out.dtype)

    @contextlib.contextmanager
    def install():
        h = getattr(layer.self_attn, attr).register_forward_hook(hook)
        yield
        h.remove()
    return install, holder


def make_weight_injector(module):
    """(install_ctx, holder): forward hook on a Linear adding y_t @ holder['delta'].T — the exact
    output field of a weight perturbation W → W + ΔW (Arm K's direct weight-GGN surface). The
    holder key is 'delta' so ggn_quadform/ggn_vp_item work unchanged with ΔW as the probe."""
    holder = {}

    def hook(_m, args, out):
        dW = holder["delta"]
        return out + (args[0].to(dW.dtype) @ dW.transpose(0, 1)).to(out.dtype)

    @contextlib.contextmanager
    def install():
        h = module.register_forward_hook(hook)
        yield
        h.remove()
    return install, holder


# ===========================================================================
# GGN-vector products and quadratic forms over ONE sequence.
#
# `item` contract (built by the driver, one per sequence; B=1 forwards — no padding, positions
# are arange):  forward_fn() -> logits (1,T_log,V) with the injector installed and the holder
# holding the CURRENT delta; loss_w (T_log,) fp32 summing to 1 over loss positions; T (probe
# length = positions the delta field covers).
# ===========================================================================
def _jvp_logits(forward_fn, holder, delta, path):
    """(logits, u = J·delta) at the CLEAN point (primal field = 0) via double-backprop (vjp of
    vjp) — the frozen manual fallback (§0.5-1). The torch.func path lives inline in
    ggn_quadform/ggn_vp_item (forward-mode duals, no reverse graph)."""
    assert path == "manual"
    d0 = torch.zeros_like(delta).requires_grad_(True)
    holder["delta"] = d0
    with torch.enable_grad():
        logits = forward_fn()
        w = torch.zeros_like(logits, requires_grad=True)
        (vjp,) = torch.autograd.grad((logits * w).sum(), d0, create_graph=True)
        (u,) = torch.autograd.grad((vjp * delta.detach()).sum(), w, retain_graph=False)
    return logits.detach(), u.detach()


def ggn_quadform(item, delta, path="func"):
    """δᵀ(JᵀH_out J)δ for one sequence — ONE jvp, no vjp. `delta` (T,d_surface) fp32."""
    if path == "func":
        import torch.func as F
        holder = item["holder"]

        def f(d):
            holder["delta"] = d
            return item["forward_fn"]()
        with torch.no_grad():
            logits, u = F.jvp(f, (torch.zeros_like(delta),), (delta,))
        return fisher_quad_from_tangent(logits[0], u[0], item["loss_w"])
    logits, u = _jvp_logits(item["forward_fn"], item["holder"], delta, "manual")
    return fisher_quad_from_tangent(logits[0], u[0], item["loss_w"])


def ggn_vp_item(item, delta, path="func"):
    """(JᵀH_out J)δ for one sequence: jvp → streaming Fisher → vjp. Returns (T,d) fp32.
    The vjp runs through a graph-resident fp32 cast of the logits, so the cotangent enters in
    fp32 and autograd handles the bf16 boundary."""
    holder = item["holder"]
    if path == "func":
        import torch.func as F

        def f(d):
            holder["delta"] = d
            return item["forward_fn"]()
        with torch.no_grad():
            logits, u = F.jvp(f, (torch.zeros_like(delta),), (delta,))
        m = fisher_logits_product(logits[0], u[0], item["loss_w"]).unsqueeze(0)
        d0 = torch.zeros_like(delta).requires_grad_(True)
        holder["delta"] = d0
        with torch.enable_grad():
            logits2 = item["forward_fn"]().float()
            (w,) = torch.autograd.grad(logits2, d0, grad_outputs=m)
        return w.detach().float()
    # manual: one graph, three backprops
    d0 = torch.zeros_like(delta).requires_grad_(True)
    holder["delta"] = d0
    with torch.enable_grad():
        logits = item["forward_fn"]()
        wdum = torch.zeros_like(logits, requires_grad=True)
        (vjp,) = torch.autograd.grad((logits * wdum).sum(), d0, create_graph=True)
        (u,) = torch.autograd.grad((vjp * delta.detach()).sum(), wdum, retain_graph=True)
        m = fisher_logits_product(logits[0].detach(), u[0].detach(), item["loss_w"]).unsqueeze(0)
        (w,) = torch.autograd.grad(logits, d0, grad_outputs=m.to(logits.dtype), retain_graph=False)
    return w.detach().float()


def hessian_vp_item(item, delta, path="manual"):
    """Full-Hessian–vector product of the SCALAR loss wrt the site field (gap measurement only).
    loss = Σ_t w_t·CE_t recomputed from logits + labels carried by the item (`labels` (T_log,)
    long, −100 = ignore; weights = loss_w)."""
    holder = item["holder"]
    d0 = torch.zeros_like(delta).requires_grad_(True)
    holder["delta"] = d0
    with torch.enable_grad():
        logits = item["forward_fn"]()[0]
        lab = item["labels"].to(logits.device)
        live = lab != -100
        lse = torch.logsumexp(logits[live].float(), dim=-1)
        nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
        loss = (nll * item["loss_w"].to(logits.device)[live]).sum()
        (g,) = torch.autograd.grad(loss, d0, create_graph=True)
        (hv,) = torch.autograd.grad((g * delta.detach()).sum(), d0)
    return hv.detach().float()


# ===========================================================================
# Sequence-mode envelope maps E_ω (and their transpose contraction).
# ===========================================================================
def env_coeffs(mode, T, sign_seed=None, device="cpu"):
    """c (T,) fp32, unit mean-square. dc/sin1/sin4 from the frozen parent convention
    (bispec_core.envelope, T0=512); white = a FIXED seeded Rademacher sign-set."""
    from infra.pg.bispec_core import envelope
    pos = torch.arange(T)
    if mode == "dc":
        c = envelope("dc", pos)
    elif mode == "sin1":
        c = envelope("sin1", pos, freq=1.0)
    elif mode == "sin4":
        c = envelope("sin4", pos, freq=4.0)
    elif mode == "white":
        assert sign_seed is not None, "white envelope needs its frozen sign-set seed"
        gen = torch.Generator(device="cpu").manual_seed(int(sign_seed))
        c = (torch.randint(0, 2, (T,), generator=gen, dtype=torch.int64) * 2 - 1).double()
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return c.float().to(device)


def mode_matvec(items, v, mode, sign_seed=None, path="func"):
    """H_ω v = mean over items of (1/T)·E_ωᵀ (JᵀH_out J) (E_ω v). v (d,) fp32 on device."""
    acc = torch.zeros_like(v, dtype=torch.float64)
    for it in items:
        T = it["T"]
        c = env_coeffs(mode, T, sign_seed=sign_seed, device=v.device)
        delta = c.unsqueeze(1) * v.unsqueeze(0)                      # (T,d)
        w = ggn_vp_item(it, delta, path=path)                        # (T,d)
        acc += (c.unsqueeze(1) * w).sum(0).double() / T
    return (acc / len(items)).float()


def mode_quadform(items, v, mode, sign_seed=None, path="func"):
    """vᵀH_ω v — jvp-only (cheap): mean over items of quadform(E_ω v)/T."""
    tot = 0.0
    for it in items:
        c = env_coeffs(mode, it["T"], sign_seed=sign_seed, device=v.device)
        tot += ggn_quadform(it, c.unsqueeze(1) * v.unsqueeze(0), path=path) / it["T"]
    return tot / len(items)


def field_quadform(items, fields, path="func"):
    """Quadform of an arbitrary per-item probe FIELD (radial family, key-coherent census, weight
    pullback): `fields[i]` (T_i, d_surface) fp32, per-token-normalized by the CALLER. Returns
    mean over items of q(field)/T (the same per-token convention as mode_quadform)."""
    tot = 0.0
    for it, f in zip(items, fields):
        tot += ggn_quadform(it, f, path=path) / it["T"]
    return tot / len(items)


# ===========================================================================
# Matrix-free Lanczos, fp32 vectors, FULL reorthogonalization, fp64 tridiagonal.
# ===========================================================================
def lanczos(matvec, d, m=LANCZOS_M, seed=0, device="cpu"):
    """Returns dict(evals desc fp64 (m,), evecs (d,m) fp32 columns matching evals, resid (m,)
    fp64 Ritz residual estimates |β_m·s_last|, alphas, betas). Deterministic in `seed`."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    q = torch.randn(d, generator=gen, dtype=torch.float32).to(device)
    q = q / q.norm()
    Q = torch.zeros(d, m, dtype=torch.float32, device=device)
    alphas, betas = [], []
    beta = 0.0
    q_prev = torch.zeros_like(q)
    n_done = 0
    for j in range(m):
        Q[:, j] = q
        w = matvec(q)
        assert torch.isfinite(w).all(), f"non-finite matvec at Lanczos step {j}"
        alpha = float(q.double() @ w.double())
        w = w - alpha * q - beta * q_prev
        # full reorthogonalization (twice — the classical "twice is enough")
        for _ in range(2):
            w = w - Q[:, :j + 1] @ (Q[:, :j + 1].transpose(0, 1) @ w)
        alphas.append(alpha)
        n_done = j + 1
        beta = float(w.norm())
        if beta < 1e-12:
            break
        betas.append(beta)
        q_prev = q
        q = w / beta
    Tm = torch.zeros(n_done, n_done, dtype=torch.float64)
    for i, a in enumerate(alphas[:n_done]):
        Tm[i, i] = a
    for i, b in enumerate(betas[:n_done - 1]):
        Tm[i, i + 1] = b
        Tm[i + 1, i] = b
    evals, S = torch.linalg.eigh(Tm)
    idx = torch.argsort(evals, descending=True)
    evals, S = evals[idx], S[:, idx]
    beta_last = betas[n_done - 1] if len(betas) >= n_done else 0.0
    resid = (beta_last * S[-1, :].abs()).double()
    evecs = (Q[:, :n_done].double() @ S.to(Q.device)).float()
    evecs = evecs / evecs.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return {"evals": evals, "evecs": evecs, "resid": resid,
            "alphas": alphas, "betas": betas, "m": n_done}


# ===========================================================================
# Spike stability: eigengap clustering + cluster-span matching (R-O1's pure half).
# ===========================================================================
def eigengap_clusters(evals, k=LANCZOS_TOPK, rel=EIGENGAP_REL):
    """Group the top-k Ritz values into clusters split at relative gaps > `rel` of the local
    scale. Near-degenerate pairs stay together so their SPANS are compared, never their
    vectors (the flapping-vector repair). Returns list of (lo, hi) index ranges."""
    ev = torch.as_tensor(evals, dtype=torch.float64)[:k]
    out, lo = [], 0
    for i in range(len(ev) - 1):
        scale = max(abs(float(ev[i])), 1e-300)
        if (float(ev[i]) - float(ev[i + 1])) / scale > rel:
            out.append((lo, i + 1))
            lo = i + 1
    out.append((lo, len(ev)))
    return out


def match_cluster_overlaps(res_a, res_b, k=LANCZOS_TOPK):
    """Split-half comparison: cluster A's spans vs the ALIGNED span in B (same index range —
    clusters are matched by spectral position). Overlap = cos of the LARGEST principal angle
    between cluster spans of equal rank (min of the two ranks columns used). Returns per-cluster
    rows [{idx_range, eval_mean_a, eval_mean_b, overlap}]."""
    cl = eigengap_clusters(res_a["evals"], k=k)
    rows = []
    for lo, hi in cl:
        Ua = res_a["evecs"][:, lo:hi].double()
        Ub = res_b["evecs"][:, lo:hi].double()
        th = principal_angles(Ua, Ub)
        rows.append({"lo": lo, "hi": hi,
                     "eval_a": float(res_a["evals"][lo:hi].mean()),
                     "eval_b": float(res_b["evals"][lo:hi].mean()),
                     "overlap": float(torch.cos(th[-1]))})
    return rows


def stable_spikes(res_a1, res_a2, rayleigh_b_fn, evals_full=None, k=LANCZOS_TOPK,
                  overlap_min=SPIKE_OVERLAP_MIN, band=SPIKE_RAYLEIGH_BAND):
    """R-O1's pure evaluation. `rayleigh_b_fn(u) -> float` measures the held-out Rayleigh
    quotient (driver closure over batch B). A cluster is stable iff split-half overlap ≥ 0.7;
    within a stable cluster, each eigvec (from the A1 estimate) must land its held-out ratio
    q_B/λ_A1 inside `band`. Returns (spikes list, cluster rows)."""
    rows = match_cluster_overlaps(res_a1, res_a2, k=k)
    spikes = []
    for r in rows:
        if r["overlap"] < overlap_min:
            continue
        for i in range(r["lo"], r["hi"]):
            lam = float(res_a1["evals"][i])
            if lam <= 0:
                continue
            qb = float(rayleigh_b_fn(res_a1["evecs"][:, i]))
            ratio = qb / lam
            if band[0] <= ratio <= band[1]:
                spikes.append({"idx": i, "eval_a1": lam, "eval_a2": float(res_a2["evals"][i]),
                               "rayleigh_b": qb, "ratio_b": ratio,
                               "cluster": [r["lo"], r["hi"]], "overlap": r["overlap"]})
    return spikes, rows


def participation_ratio(trace, trace_sq):
    """PR = (tr H)²/tr(H²) — d_eff of a measured spectrum (spec T4 lifted to the operator)."""
    assert trace_sq > 0, "degenerate operator: tr(H²) ≤ 0"
    return float(trace) ** 2 / float(trace_sq)


# ===========================================================================
# Spectral completion — the frozen q̂(δ) functional M4 uses (zero fitted directions).
# ===========================================================================
def spectral_completion_quadform(delta_hat, spikes_U, spikes_lam, band_bulk, basis, sb_slices):
    """q̂(δ̂) = Σ_i λ_i⟨δ̂,u_i⟩² + Σ_b bulk_b·‖P_b δ̂_⊥spike‖², δ̂ unit. `band_bulk[b]` is the
    per-direction bulk mean of sub-band b AFTER removing the spike share:
    (tr_b − Σ_{spikes in b} λ_i·mass_i,b)/k_b, floored at 0 by the caller.

    This is the closed-form completion of a top-k + band-trace measurement — every ingredient
    is measured, none fitted (M4's contract)."""
    v = torch.as_tensor(delta_hat, dtype=torch.float64).reshape(-1)
    v = v / v.norm().clamp_min(1e-300)
    q = 0.0
    resid = v.clone()
    if spikes_U is not None and spikes_U.numel():
        U = torch.as_tensor(spikes_U, dtype=torch.float64)
        coef = U.transpose(0, 1) @ v
        q += float((torch.as_tensor(spikes_lam, dtype=torch.float64) * coef ** 2).sum())
        resid = v - U @ coef
    B = torch.as_tensor(basis, dtype=torch.float64)
    coefs = B.transpose(0, 1) @ resid
    for b, (lo, hi) in enumerate(sb_slices):
        q += float(band_bulk[b]) * float((coefs[lo:hi] ** 2).sum())
    return q


def band_bulk_from_traces(sb_traces, sb_slices, spikes_U, spikes_lam, basis):
    """Per-direction bulk means after removing each spike's in-band share (floored at 0)."""
    tr = [float(t) for t in sb_traces]
    if spikes_U is not None and spikes_U.numel():
        B = torch.as_tensor(basis, dtype=torch.float64)
        U = torch.as_tensor(spikes_U, dtype=torch.float64)
        C = B.transpose(0, 1) @ U                                # (d, n_spike) coefs
        for b, (lo, hi) in enumerate(sb_slices):
            mass = (C[lo:hi] ** 2).sum(0)
            tr[b] -= float((torch.as_tensor(spikes_lam, dtype=torch.float64) * mass).sum())
    return [max(t, 0.0) / max(hi - lo, 1) for t, (lo, hi) in zip(tr, sb_slices)]


# ===========================================================================
# The two frozen damage predictors (M0 energy-only, M4 operator) — 1-D isotone links only.
# ===========================================================================
def fit_link(q_values, damages):
    """The single fitted object per model: a monotone scalar link g. PAVA on (log10 q, damage)."""
    import numpy as np

    from infra.pg.atlas_fit import pava_fit
    x = np.log10(np.maximum(np.asarray(q_values, float), 1e-30))
    bx, by = pava_fit(x, np.asarray(damages, float))
    return {"iso_x": bx.tolist(), "iso_y": by.tolist(), "n": len(x)}


def predict_link(link, q_values):
    import numpy as np

    from infra.pg.atlas_fit import pava_predict
    x = np.log10(np.maximum(np.asarray(q_values, float), 1e-30))
    return pava_predict(np.asarray(link["iso_x"]), np.asarray(link["iso_y"]), x)


# ===========================================================================
# Arm K — K-FAC pullback + stiffness quantiles.
# ===========================================================================
def kfac_stiffness(dW, Sigma_y, spikes_U, spikes_lam, band_bulk, basis, sb_slices):
    """Q(ΔW) = tr(ΔW Σ_y ΔWᵀ F̂_z) with F̂_z the spectral completion. Computed in the
    eigenbasis of Σ_y-independent form: Q = Σ over an orthonormal factorization of ΔWΣ_yΔWᵀ.
    Direct: eigendecompose M = ΔW Σ_y ΔWᵀ (d_out×d_out, PSD, low effective rank) and sum
    μ_j·q̂(w_j)."""
    M = dW.double() @ Sigma_y.double() @ dW.double().transpose(0, 1)
    M = 0.5 * (M + M.transpose(0, 1))
    mu, W = torch.linalg.eigh(M)
    q = 0.0
    for j in range(len(mu)):
        if float(mu[j]) <= 1e-18 * float(mu.max()):
            continue
        q += float(mu[j]) * spectral_completion_quadform(W[:, j], spikes_U, spikes_lam,
                                                         band_bulk, basis, sb_slices)
    return q


def stiffness_quantile(values, reference):
    """Mean quantile of `values` inside the `reference` distribution (Haar band) ∈ [0,1]."""
    import numpy as np
    ref = np.sort(np.asarray(reference, float))
    vals = np.asarray(values, float)
    qs = np.searchsorted(ref, vals, side="right") / len(ref)
    return float(qs.mean()), [float(x) for x in qs]


# ===========================================================================
# Arm H — fixed-energy projection (the ascent loop lives in the driver).
# ===========================================================================
def project_pertoken_energy(delta, eps_abs):
    """Renormalize each token row to ‖δ_t‖ = eps_abs exactly (the fixed-energy manifold)."""
    n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return delta * (float(eps_abs) / n)


def project_coherent(delta_v, eps_abs):
    """Coherent family: one d-vector at norm eps_abs."""
    return delta_v * (float(eps_abs) / delta_v.norm().clamp_min(1e-12))


# ===========================================================================
# Range guard + the frozen d_eff convention.
# ===========================================================================
def range_guard(response, floor, mult=3.0):
    """True iff `response` clears `mult`× the noise floor. BINDING: `floor` must be measured on
    the SAME construction family as `response` (the ATLAS R-T1 lesson, §5 control row) — the
    D-suite replays the misfire where a mismatched floor voids a live family."""
    assert floor >= 0
    return float(response) >= mult * float(floor)


def d_eff_log_damage(dppl_values):
    """The frozen d_eff convention (ATLAS R12): 2/CV² on log(1+Δppl), never on raw Δppl.
    Raw-damage d_eff of a heavy-tailed family reads ≈1 regardless of its true dimension —
    the R12 event; the log convention is what the banked corrected values use."""
    import math as _m

    from infra.pg.atlas_core import d_eff_from_damages
    return d_eff_from_damages([_m.log1p(max(v, -0.99)) for v in dppl_values])


# ===========================================================================
# §6 adjudicators — every rule carries a ceiling or VOID and a constructible failure branch.
# Status flags computed LAST by drivers (REGRAIN E17). Tallies never restated in prose.
# ===========================================================================
def adjudicate_ro1(*, n_stable, converged, restart_used):
    """R-O1. FAILS H_spikes (reports H_field) when zero clusters survive split-half + held-out —
    the metric is item-conditional and the pre-specified per-item mini-battery runs."""
    if not converged:
        return {"verdict": "VOID-nonconverged", "restart_used": bool(restart_used),
                "reading": "reported at reduced k with residuals; the spectrum estimate itself "
                           "remains a deliverable"}
    if n_stable >= 1:
        return {"verdict": "H_spikes", "n_stable": int(n_stable)}
    return {"verdict": "H_field", "n_stable": 0,
            "reading": "no stable average spikes — the per-item mini-battery (power iteration, "
                       "16 items, top-1 each) is promoted from curiosity to named successor"}


def adjudicate_ro2_key(*, mean_rayleigh, bulk_mean):
    """R-O2 key-coherent slice. ≥0.1×bulk is a fail-loud INSTRUMENT HALT (the theorem is exact)."""
    assert bulk_mean > 0, "degenerate bulk reference"
    r = mean_rayleigh / bulk_mean
    if r <= 1e-3:
        return {"verdict": "H_theorem-verified", "ratio": r}
    if r < 0.1:
        return {"verdict": "H_measured-leak", "ratio": r,
                "reading": "quantified leak; RoPE counter-rotation precision and V-channel bleed "
                           "are the named suspects, both measurable from the same census"}
    return {"verdict": "HALT-instrument", "ratio": r,
            "reading": "the post-RoPE coherent-K gauge is exact; a ≥0.1×bulk reading is wiring — "
                       "fail loud, fix, re-run"}


def adjudicate_ro2_radial(*, mean_rayleigh, bulk_mean):
    """R-O2 radial family. ≥0.25×bulk = H_radial-active-quadratic — pre-registered as the most
    interesting landing (quadratic and finite-amplitude regimes disagree about the gauge)."""
    assert bulk_mean > 0, "degenerate bulk reference"
    r = mean_rayleigh / bulk_mean
    if r <= 0.05:
        return {"verdict": "H_gauge-confirmed", "ratio": r}
    if r < 0.25:
        return {"verdict": "H_skip-share-measured", "ratio": r,
                "reading": "feeds M4's radial production as a MEASURED constant"}
    return {"verdict": "H_radial-active-quadratic", "ratio": r,
            "reading": "scale-inconsistency between quadratic and finite-amplitude regimes — the "
                       "successor's opening question"}


def adjudicate_rc1(*, tau_ppl, p_ppl, tau_vqa, p_vqa, bh_pass_ppl, bh_pass_vqa,
                   regressor_span_ratio, u_dc_stable):
    """R-C1 (the MDE). Guard FIRST: span < 10× ⇒ VOID-no-range (the alternative regressor
    adjudicates instead — the caller re-enters with that regressor's numbers and
    `regressor_span_ratio` recomputed)."""
    if regressor_span_ratio < 10.0:
        return {"verdict": "VOID-no-range", "span_ratio": regressor_span_ratio,
                "reading": "the draw family is near-orthogonal to the spike axis — itself a "
                           "measured alignment fact; the top-8 spike-subspace energy regressor "
                           "adjudicates instead"}
    named = (tau_ppl >= 0.6 and p_ppl <= 0.05 and bh_pass_ppl) or \
            (tau_vqa >= 0.6 and p_vqa <= 0.05 and bh_pass_vqa)
    if named:
        return {"verdict": "H_carrier-named", "tau_ppl": tau_ppl, "tau_vqa": tau_vqa,
                "reading": "the 110× spread has an identity; the program's #1 unnamed object closes"}
    if u_dc_stable and tau_ppl < 0.33 and tau_vqa < 0.33:
        return {"verdict": "H_spike-not-carrier", "tau_ppl": tau_ppl, "tau_vqa": tau_vqa,
                "reading": "the spread is off-average or non-quadratic — weight routes to Arm H "
                           "and the H_field mini-battery; the spike stays real, scoped"}
    return {"verdict": "H_partial", "tau_ppl": tau_ppl, "tau_vqa": tau_vqa,
            "reading": "both legs reported; the alternative regressor's reading banked for the "
                       "successor"}


def adjudicate_rc2(*, band1_white_ratio, band4_reversed_ratio):
    """R-C2. band1: tr(H_white P_b1)/k_b1 ≥ 30× median coherent-Haar band1 quad AND band4
    reversed ≥ 3× ⇒ the S3 reversal is quadratic-order fact."""
    b1 = band1_white_ratio >= 30.0
    b4 = band4_reversed_ratio >= 3.0
    if b1 and b4:
        return {"verdict": "H_reversal-quadratic", "b1_ratio": band1_white_ratio,
                "b4_ratio": band4_reversed_ratio}
    if b1 != b4:
        return {"verdict": "H_one-side", "b1": bool(b1), "b4": bool(b4),
                "reading": "the failing side names where non-quadratic structure lives"}
    return {"verdict": "H_reversal-nonquadratic",
            "reading": "a clean, large statement bounding every quadratic account of the "
                       "sequence axis"}


def adjudicate_rc3(*, R_l0, ratio_l0, R_l14, ratio_l14, tol=0.4):
    """R-C3, the sharpest single test. Both within ±0.4 ⇒ dial explained at quadratic order and
    R(L8) is journaled as a registered prediction for the pre-approved L8 battery."""
    hit0 = abs(R_l0 - ratio_l0) <= tol
    hit14 = abs(R_l14 - ratio_l14) <= tol
    if hit0 and hit14:
        return {"verdict": "H_dial-quadratic", "miss": {}}
    misses = {k: v for k, v in
              [("L0", abs(R_l0 - ratio_l0)), ("L14", abs(R_l14 - ratio_l14))]
              if v > tol}
    return {"verdict": "H_dial-partial" if (hit0 or hit14) else "H_dial-miss", "miss": misses,
            "reading": "the miss pattern localizes the non-quadratic transport component by site"}


def adjudicate_rc4(overlaps, haar_p95s):
    """R-C4 census. Identification iff ≥ 3× matched-rank Haar p95. Zero identifications is the
    pre-registered exciting branch (new structure, ships with its constructive handle)."""
    idents = {k: (v, v / haar_p95s[k]) for k, v in overlaps.items()
              if haar_p95s[k] > 0 and v >= 3.0 * haar_p95s[k]}
    if idents:
        return {"verdict": "H_identified", "identifications": idents}
    return {"verdict": "H_new-structure", "identifications": {},
            "reading": "the stable spike set is unaligned with every named object — the eigvec "
                       "bank ships as the successor's starting material"}


def adjudicate_rk1(*, kfac_rho, qbar, haar_p05, haar_p95, quartile):
    """R-K1 CORE⟂SPIKE, gated on K-FAC fidelity ρ ≥ 0.8 (below ⇒ pre-specified de-scope to the
    direct-subsample scope, not a halt)."""
    scope = "kfac" if kfac_rho >= 0.8 else "direct-subsample-only"
    if qbar >= haar_p95 and quartile >= 0.75:
        v = ("H_core-stiff", "curvature explains conservation; protect-the-stiff-set becomes a "
                             "testable transfer rule")
    elif qbar <= haar_p05 and quartile <= 0.25:
        v = ("H_core-sloppy", "fine-tuning preserves the FREE directions — flat-minima-flavoured "
                              "inversion; gradient starvation is the next mechanism candidate")
    else:
        v = ("H_core-generic", "conservation is not curvature-positioned — the strongest license "
                               "yet for the data-side (visibility/starvation) account")
    return {"verdict": v[0], "reading": v[1], "scope": scope, "kfac_rho": kfac_rho,
            "qbar": qbar, "haar_band": [haar_p05, haar_p95]}


def adjudicate_rs1(*, damage_cilo_above_null, rotated_inside_band, void=False):
    """R-S1 per promoted spike: behaviourally real iff damage clears the Bingham em-null band
    (CI-lo > 0) AND the rotated control sits inside it. Below-band ⇒ spectral-only, still a
    deliverable."""
    if void:
        return {"verdict": "VOID", "reading": "saturation guard fired on the pair"}
    if damage_cilo_above_null and rotated_inside_band:
        return {"verdict": "H_behaviourally-real"}
    if damage_cilo_above_null and not rotated_inside_band:
        return {"verdict": "H_not-direction-specific",
                "reading": "any matched-energy vector at this construction does it — the spike "
                           "is not the carrier of its own cell's damage"}
    return {"verdict": "H_spectral-only",
            "reading": "honestly scoped; the quadratic atlas marks it"}


def adjudicate_rh1(*, G, Q_ratio, improved_factor):
    """R-H1 hunt. VOID-flat iff the optimizer improved < 1.2× over init on the proxy."""
    if improved_factor < 1.2:
        return {"verdict": "VOID-flat", "improved": improved_factor,
                "reading": "the landscape is locally flat at this energy — a tolerance statement"}
    if G <= 2.0:
        return {"verdict": "H_concentration-certified", "G": G,
                "reading": "an empirical robustness-radius certificate — a positive deliverable "
                           "no perturbation study has issued"}
    if 0.5 <= Q_ratio <= 2.0:
        return {"verdict": "H_quadratic-worst-case", "G": G, "Q_ratio": Q_ratio}
    if Q_ratio > 2.0:
        return {"verdict": "H_boundary-dominated", "G": G, "Q_ratio": Q_ratio,
                "reading": "witness direction banked — the non-quadratic exceptional set becomes "
                           "constructive"}
    return {"verdict": "H_subquadratic-found", "G": G, "Q_ratio": Q_ratio,
            "reading": "the found direction under-performs its own quadratic prediction — "
                       "curvature over-states damage off-axis; reported as measured"}


def adjudicate_rm1(*, drho, drho_cilo, drho_cihi, rho_m4, rho_m0, n_void_backups):
    """R-M1 primary (per readout; BH upstream). All five branches, incl. the author's death
    branch written as a full positive result."""
    if n_void_backups > 10:
        return {"verdict": "VOID-battery",
                "response": "one labeled half-amplitude re-freeze (the single pre-specified "
                            "re-freeze in the program)"}
    if drho_cilo > 0 and rho_m4 >= 0.5:
        return {"verdict": "H_operator-predicts", "drho": drho, "rho_m4": rho_m4, "rho_m0": rho_m0}
    if rho_m0 >= 0.5 and drho_cihi < 0.1:
        return {"verdict": "H_energy-sufficient-again", "drho": drho, "rho_m4": rho_m4,
                "rho_m0": rho_m0,
                "reading": "scale separation established at sealed grade — damage at behavioural "
                           "amplitudes is not set by the exactly-measured local quadratic form; "
                           "the strongest possible motivation for the boundary program"}
    if rho_m4 >= 0.5 and rho_m0 >= 0.5:
        return {"verdict": "H_partial", "drho": drho, "rho_m4": rho_m4, "rho_m0": rho_m0,
                "reading": "the per-descriptor residual decomposition maps which grammar axes the "
                           "operator wins on — the successor's chart"}
    return {"verdict": "H_neither", "drho": drho, "rho_m4": rho_m4, "rho_m0": rho_m0,
            "reading": "anomaly set v2, doubly characterized — the best-instrumented open object "
                       "the lineage has produced"}


def adjudicate_rb1_collapse(curves, layer_range=(4, 17), tol=0.05):
    """R-B1 bulk. `curves` = {layer: [dispersion per rung]} on one shared rung grid. Collapse iff
    pairwise max|Δ| ≤ 0.05 across layers 4–17; the peel-off boundary (first violating layer,
    scanning DOWN from 17) reported wherever it lands."""
    import numpy as np
    Ls = sorted(L for L in curves if layer_range[0] <= L <= layer_range[1])
    assert len(Ls) >= 2, "collapse needs at least two layers in range"
    M = {}
    for i, La in enumerate(Ls):
        for Lb in Ls[i + 1:]:
            M[(La, Lb)] = float(np.max(np.abs(np.asarray(curves[La]) - np.asarray(curves[Lb]))))
    collapse = all(v <= tol for v in M.values())
    # peel-off boundary: the HIGHEST layer whose curve violates the collapse against the
    # top-layer reference (the registered prediction is peel ≤ 3 — early layers peel, the
    # rest collapse); None if every layer matches
    ref = curves[max(Ls)]
    viol = [L for L in sorted(curves)
            if float(np.max(np.abs(np.asarray(curves[L]) - np.asarray(ref)))) > tol]
    peel = max(viol) if viol else None
    return {"verdict": "H_collapse" if collapse else "H_no-collapse",
            "pairwise_max": {f"{a}-{b}": v for (a, b), v in M.items()},
            "worst_pair": max(M.values()), "peel_off_layer": peel}


# ===========================================================================
# Finite-difference wiring check (Gate-0's decisive assert, pure recipe).
# ===========================================================================
def fd_directional_check(item, delta, path="func", h_rel=1e-3):
    """Compares the GGN-vp inner product ⟨δ, Gδ⟩ against the central finite difference of the
    LOSS GRADIENT along δ contracted with δ — for CE at the clean point these agree up to the
    (measured) GGN−Hessian gap plus O(h²); the D-suite verifies the identity exactly on a toy
    model where the map is linear (gap = 0), and Gate-0 uses the FISHER-EXACT identity instead:

        ⟨δ, Gδ⟩  ==  d²/dα² [ KL(p(0) ‖ p(αδ)) ]|₀  ≈  (KL(+h) + KL(−h))/h²

    because the Fisher is the KL Hessian in the perturbation — this form involves NO label and
    NO Hessian gap, so it checks the wiring of J and H_out jointly (symmetric difference kills
    the odd cubic term; KL(0)=KL'(0)=0). Both sides carry the same loss weights, so no extra
    normalization enters. Returns (quad_ggn, quad_fd, rel_err)."""
    q = ggn_quadform(item, delta, path=path)
    holder, fwd = item["holder"], item["forward_fn"]

    def _logp(scale):
        holder["delta"] = delta * scale
        with torch.no_grad():
            lg = fwd()[0].float()
        return lg - torch.logsumexp(lg, dim=-1, keepdim=True)

    h = float(h_rel)
    lp0 = _logp(0.0)
    w = item["loss_w"].to(lp0.device)
    live = w > 0

    def _kl(lp_pert, chunk=65536):
        a0 = lp0[live].double()
        a1 = lp_pert[live].double()
        tot = torch.zeros(a0.shape[0], dtype=torch.float64, device=a0.device)
        for c0 in range(0, a0.shape[1], chunk):
            p0 = a0[:, c0:c0 + chunk].exp()
            tot += (p0 * (a0[:, c0:c0 + chunk] - a1[:, c0:c0 + chunk])).sum(-1)
        return float((w[live].double() * tot).sum())

    kl_p = _kl(_logp(+h))
    kl_m = _kl(_logp(-h))
    q_fd = (kl_p + kl_m) / (h * h)
    rel = abs(q - q_fd) / max(abs(q_fd), 1e-30)
    return q, q_fd, rel
