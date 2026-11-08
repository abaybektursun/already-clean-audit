"""GESTALT splat instrument — mixture-of-low-rank-Gaussians fitter, moment-matched twin,
the FROZEN reconstruction rule, and the splat-swap hook. Spec: 2026-08-04_gestalt_program_spec.md
(Arm S estimand, FR-4/FR-8 in the journal).

THE FROZEN RECONSTRUCTION RULE (spec §2 Arm S; do not change without a journalled re-freeze):

    x_hat = sum_c gamma_c(x) * [ mu_c + U_c U_c^T (x - mu_c) ]

where gamma_c(x) is the posterior responsibility of component c under the fitted mixture and
U_c is the component's ORTHONORMAL rank-r basis. r = d makes x_hat == x exactly (the built-in
identity check the swap hook's P3 leans on).

FITTER (FR-4): Mixture of PPCA (Tipping & Bishop 1999), full-batch EM, component c =
(pi_c, mu_c, U_c diag(lam_c) U_c^T + sigma2_c I). The per-component M-step is the closed-form
PPCA solution from the responsibility-weighted second moment, realized with a truncated
randomized SVD of the sqrt(gamma)-weighted centered data (deterministic given the fit seed).
Divergence (non-finite or LL drop > LL_DECREASE_TOL) -> ONE retry (sub-seed+1, init floor x10);
still bad -> the (K, source, seed) cell is VOID. A component collapsing below MIN_WEIGHT is
RECORDED in the atlas, never resampled (spec intercurrent rule).

TWIN (spec §1.1): same mean, same covariance (fp64 streaming moments of the REAL stream), same
sample count, fresh RNG per seed; sampled x = m + z @ C_sym_sqrt. "The twin explains everything"
is the program's honest exit, so the twin generator's determinism is itself gated (P2).

All torch, device-agnostic (CPU in the offline gate, CUDA in jobs). Heavy imports inside
functions so the module py_compiles on the control plane. No try/except (CLAUDE.md): a shape or
numerics error RAISES.
"""
from __future__ import annotations

# ---- frozen constants (changing any is a re-freeze; journal it) ---------------------------
K_GRID = (1, 2, 4, 8, 16, 32, 64)
EM_SEEDS = (0, 1)
EM_MAX_ITERS = 60
# D11 amendment (pre-real-data; offline gate re-run required): stopping/divergence are
# PER-POINT NATS, scale-free in d. The earlier |LL|-relative tolerance starved EM at d=2048
# (per-point |LL| ~ 2.4e3 nats made the stop threshold exceed the entire K=1->2 split gain).
# D14 amendment (gate-driven, pre-real-data): per-point LL contains d*log(sigma2), so a ~0.1%
# sigma2 jitter from the randomized-SVD M-step swings LL by whole nats/pt at d=2048 — LL cannot
# gate convergence/divergence for an approximate-M-step EM. Divergence := NON-FINITE only
# (retry -> VOID); convergence := PARAMETER MOVEMENT (scale-free), LL banked as diagnostics.
EM_STOP_MEAN_SHIFT_REL = 1e-3     # max component-mean shift / cloud RMS radius, below this...
EM_STOP_SIGMA_REL = 1e-3          # ...and max relative sigma2 change below this...
EM_STOP_PATIENCE = 2              # ...for this many CONSECUTIVE iterations => converged
MIN_WEIGHT = 1e-4            # component weight below this = collapsed (recorded)
SIGMA2_MIN_FRAC = 1e-6       # sigma2 floor as a fraction of mean data energy
N_FIT = 100_000              # fit subsample per (site, source); frozen (FR-8 note)
SVD_OVERSAMPLE = 8
SVD_NITER = 4


# ===========================================================================
# Moments + twin.
# ===========================================================================
def stream_moments(X):
    """fp64 (mean (d,), cov (d,d)) of a (N, d) tensor. Full-batch is fine at N~1e6, d=2048
    (the fp64 Gram is 33 MB); chunked matmul keeps peak memory flat."""
    import torch
    X = X.to(torch.float64)
    n, d = X.shape
    assert n > d, f"need N > d for a full-rank covariance estimate, got {n} x {d}"
    m = X.mean(0)
    G = torch.zeros(d, d, dtype=torch.float64, device=X.device)
    for s in range(0, n, 65536):
        c = X[s:s + 65536]
        G += c.T @ c
    C = G / n - torch.outer(m, m)
    assert torch.isfinite(C).all(), "non-finite covariance"
    return m, C


def cov_sym_sqrt(C):
    """Symmetric PSD square root via fp64 eigh, negative eigenvalues clamped to 0."""
    import torch
    w, V = torch.linalg.eigh(C)
    w = torch.clamp(w, min=0.0)
    return (V * w.sqrt().unsqueeze(0)) @ V.T


def twin_sample(mean, cov_sqrt, n, seed, device="cpu", chunk=65536):
    """N(mean, cov) sample of n rows, deterministic in `seed` (fresh RNG per seed, spec §1.1).
    Returns fp32 (n, d). Generated on CPU generator for cross-device determinism, moved after."""
    import torch
    d = mean.shape[0]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.empty(n, d, dtype=torch.float32, device=device)
    m32 = mean.to(torch.float32).to(device)
    L32 = cov_sqrt.to(torch.float32).to(device)
    for s in range(0, n, chunk):
        z = torch.randn(min(chunk, n - s), d, generator=g, dtype=torch.float32).to(device)
        out[s:s + z.shape[0]] = m32 + z @ L32
    return out


def twin_spec(mean, cov, n, seed):
    """The banked twin specification: moment hashes + RNG seed (spec §4 'twin specifications')."""
    import hashlib
    return {"mean_sha": hashlib.sha256(mean.cpu().numpy().tobytes()).hexdigest()[:16],
            "cov_sha": hashlib.sha256(cov.cpu().numpy().tobytes()).hexdigest()[:16],
            "n": int(n), "seed": int(seed)}


def fit_subsample(X, seed, n_fit=N_FIT):
    """The frozen uniform fit subsample (FR-8): deterministic in `seed`, without replacement."""
    import torch
    n = X.shape[0]
    if n <= n_fit:
        return X
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 990_001)
    idx = torch.randperm(n, generator=g)[:n_fit]
    return X[idx.to(X.device)]


# ===========================================================================
# MPPCA EM.
# ===========================================================================
def _log_resp(X, atlas_t, chunk=65536):
    """(log responsibilities (N, K), total log-likelihood scalar) under the mixture, fp32,
    Woodbury/determinant-lemma so cost is O(N d r K) and no d x d solve appears.

      C_c = U diag(lam) U^T + s2 I
      logdet C_c = (d - r) log s2 + sum_i log(lam_i + s2)
      q_c(x) = ||y||^2/s2 - sum_i (lam_i/(s2 (lam_i+s2))) (y^T u_i)^2 ,  y = x - mu_c
    """
    import math

    import torch
    log_pi, mu, U, lam, s2 = (atlas_t[k] for k in ("log_pi", "mu", "U", "lam", "sigma2"))
    K, d, r = U.shape
    n = X.shape[0]
    logdet = (d - r) * torch.log(s2) + torch.log(lam + s2.unsqueeze(1)).sum(1)     # (K,)
    scale = lam / (s2.unsqueeze(1) * (lam + s2.unsqueeze(1)))                       # (K, r)
    const = -0.5 * d * math.log(2 * math.pi)
    U_flat = U.permute(0, 2, 1).reshape(K * r, d)                                   # (K*r, d)
    mproj = torch.einsum("kd,kdr->kr", mu, U).unsqueeze(0)                          # (1, K, r)
    out = torch.empty(n, K, dtype=torch.float32, device=X.device)
    for s in range(0, n, chunk):
        xc = X[s:s + chunk]                                                         # (b, d)
        # ||y||^2 = ||x||^2 - 2 x.mu + ||mu||^2 ; projections via one (b,d)@(d,K*r) matmul
        x2 = (xc * xc).sum(1, keepdim=True)                                         # (b, 1)
        xm = xc @ mu.T                                                              # (b, K)
        m2 = (mu * mu).sum(1).unsqueeze(0)                                          # (1, K)
        y2 = x2 - 2 * xm + m2                                                       # (b, K)
        proj = (xc @ U_flat.T).reshape(xc.shape[0], K, r)                           # (b, K, r)
        yproj = proj - mproj                                                        # (b, K, r)
        q = y2 / s2.unsqueeze(0) - (scale.unsqueeze(0) * yproj * yproj).sum(-1)     # (b, K)
        out[s:s + xc.shape[0]] = const - 0.5 * (logdet.unsqueeze(0) + q) + log_pi.unsqueeze(0)
    ll = torch.logsumexp(out, dim=1)
    # fp64 accumulation: an fp32 sum's pairwise error (~eps*sqrt(N)*|ll|) reaches ~0.03 nats/pt
    # at d=2048, swamping the per-point stopping tolerances (D12)
    total_ll = float(ll.double().sum().item())
    log_gamma = out - ll.unsqueeze(1)
    return log_gamma, total_ll


def _init_atlas(X, K, r, seed):
    """kmeans++-style seeding of means on a subsample; U from global PCA directions; flat pi;
    sigma2 from residual energy. Deterministic in `seed`."""
    import torch
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    sub = X[torch.randperm(n, generator=g)[: min(n, 20_000)].to(X.device)].float()
    Xc = sub - sub.mean(0)
    torch.manual_seed(int(seed))
    _, S, V = torch.svd_lowrank(Xc, q=r + SVD_OVERSAMPLE, niter=SVD_NITER)
    U0 = V[:, :r]                                                          # (d, r) shared init
    # kmeans++ seeding in the TOP-PCA coordinates (D15): full-d distances are noise-chi2
    # dominated at high d (within-cluster ~ d*sigma^2 vs structure O(1) dims), so raw-space
    # seeding lands multiple components in one mode and hard high-d responsibilities never
    # escape. The top-(r+SVD_OVERSAMPLE) coords are structure-dominated; fp64 for overflow
    # safety (the D3 lesson).
    Z = (Xc @ V).to(torch.float64)                                         # (n_sub, r+os)
    seed_idx = [int(torch.randint(sub.shape[0], (1,), generator=g).item())]
    d2 = ((Z - Z[seed_idx[0]]) ** 2).sum(1)
    for _ in range(1, K):
        p = (d2 / d2.sum()).cpu()
        assert bool(torch.isfinite(p).all()), "kmeans++ seeding probabilities non-finite"
        nxt = int(torch.multinomial(p, 1, generator=g).item())
        seed_idx.append(nxt)
        d2 = torch.minimum(d2, ((Z - Z[nxt]) ** 2).sum(1))
    # D16: initial means are LLOYD CENTROIDS (10 iterations in Z-space), never raw points — a
    # raw seed carries its own noise vector, and in the small-sigma2 metric the cross-term
    # <x_noise, seed_noise>/sigma2 has std ~ sqrt(d) nats (~45 at d=2048), swamping the cluster
    # signal so the first E-step assigns by noise-alignment lottery (read directly off the
    # gestalt_p1diag trajectory: cross-cluster seeds at +/-3.9 collapsed to ~0 in ONE M-step).
    # Centroid averaging shrinks that noise by 1/sqrt(members).
    cent = Z[torch.tensor(seed_idx, device=Z.device)]                      # (K, r+os) fp64
    assign = None
    for _ in range(10):
        d2k = ((Z.unsqueeze(1) - cent.unsqueeze(0)) ** 2).sum(-1)          # (n_sub, K)
        assign = d2k.argmin(1)
        for c in range(K):
            m_c = assign == c
            if bool(m_c.any()):
                cent[c] = Z[m_c].mean(0)
    mu_list = []
    for c in range(K):
        m_c = assign == c
        # empty Lloyd cluster: keep the raw seed point (recorded via the collapsed-component
        # bookkeeping downstream, never resampled — the frozen intercurrent rule)
        mu_list.append(sub[m_c].mean(0) if bool(m_c.any()) else sub[seed_idx[c]])
    mu = torch.stack(mu_list)                                              # (K, d)
    U = U0.unsqueeze(0).expand(K, d, r).contiguous()
    ev = (S[:r] ** 2) / sub.shape[0]
    resid = ((Xc ** 2).sum() / sub.shape[0] - ev.sum()) / max(d - r, 1)
    s2 = torch.full((K,), float(torch.clamp(resid, min=1e-8)), device=X.device)
    lam = torch.clamp(ev - s2[0], min=0.0).unsqueeze(0).expand(K, r).contiguous()
    log_pi = torch.full((K,), -float(torch.log(torch.tensor(float(K)))), device=X.device)
    return {"log_pi": log_pi, "mu": mu, "U": U, "lam": lam.clone(), "sigma2": s2}


def _m_step(X, log_gamma, r, seed, floor):
    """Closed-form MPPCA M-step from sqrt(gamma)-weighted centered data, truncated randomized
    SVD (deterministic: torch.manual_seed(seed) before the sketch)."""
    import torch
    n, d = X.shape
    K = log_gamma.shape[1]
    gamma = log_gamma.exp()                                                # (N, K)
    nk = gamma.sum(0)                                                      # (K,)
    log_pi = torch.log(torch.clamp(nk / n, min=1e-12))
    mu = (gamma.T @ X) / torch.clamp(nk, min=1e-9).unsqueeze(1)            # (K, d)
    U_new, lam_new, s2_new = [], [], []
    torch.manual_seed(int(seed))
    for c in range(K):
        w = gamma[:, c]
        Y = (X - mu[c]) * w.sqrt().unsqueeze(1)                            # (N, d)
        _, S, V = torch.svd_lowrank(Y, q=min(r + SVD_OVERSAMPLE, d), niter=SVD_NITER)
        nc = torch.clamp(nk[c], min=1e-9)
        ev = (S[:r] ** 2) / nc                                             # top-r eigenvalues
        tr = (Y * Y).sum() / nc
        s2 = torch.clamp((tr - ev.sum()) / max(d - r, 1), min=floor)
        U_new.append(V[:, :r])
        lam_new.append(torch.clamp(ev - s2, min=0.0))
        s2_new.append(s2)
    return {"log_pi": log_pi, "mu": mu, "U": torch.stack(U_new),
            "lam": torch.stack(lam_new), "sigma2": torch.stack(s2_new)}


def fit_mppca(X, K, r, seed, max_iters=EM_MAX_ITERS, _retry=False):
    """Fit a K-component rank-r MPPCA to (N, d) fp32 `X`. Returns the atlas dict:
    {log_pi, mu, U, lam, sigma2 (torch, on X.device), K, r, seed, n_fit, ll_trace, n_iters,
     collapsed, void, void_reason, retry_used}. VOID per the frozen divergence rule."""
    import torch
    assert X.dtype == torch.float32 and X.ndim == 2, f"X must be fp32 (N,d), got {X.dtype} {X.shape}"
    if not _retry:
        # Non-finite INPUT is a contract violation and RAISES (fail-loud); the VOID branch is
        # reserved for EM divergence on valid data (the frozen intercurrent rule, journal FR-4/D3).
        assert bool(torch.isfinite(X).all()), "fit_mppca: input contains non-finite values"
    n, d = X.shape
    assert 1 <= r <= d and 1 <= K, f"bad K={K} r={r} for d={d}"
    floor = SIGMA2_MIN_FRAC * float((X * X).mean().item()) * (10.0 if _retry else 1.0)
    atlas = _init_atlas(X, K, r, seed + (1 if _retry else 0))
    ll_trace = []
    import math
    rms_radius = float((X * X).sum(1).mean().sqrt().item())
    calm = 0
    for it in range(max_iters):
        log_gamma, total_ll = _log_resp(X, atlas)
        if not math.isfinite(total_ll):
            if not _retry:
                return fit_mppca(X, K, r, seed, max_iters=max_iters, _retry=True)
            return {"K": K, "r": r, "seed": seed, "void": True, "retry_used": True,
                    "void_reason": f"EM divergence (non-finite LL) at iter {it}",
                    "ll_trace": ll_trace}
        ll_trace.append(total_ll)
        prev_mu, prev_s2 = atlas["mu"], atlas["sigma2"]
        atlas = _m_step(X, log_gamma, r, seed, floor)
        mu_shift = float((atlas["mu"] - prev_mu).norm(dim=1).max().item()) / max(rms_radius, 1e-30)
        s2_rel = float(((atlas["sigma2"] - prev_s2).abs()
                        / prev_s2.clamp(min=1e-30)).max().item())
        calm = calm + 1 if (mu_shift < EM_STOP_MEAN_SHIFT_REL
                            and s2_rel < EM_STOP_SIGMA_REL) else 0
        if calm >= EM_STOP_PATIENCE:
            break
    collapsed = [int(c) for c in range(K) if float(atlas["log_pi"][c].exp()) < MIN_WEIGHT]
    atlas.update({"K": K, "r": r, "seed": int(seed), "n_fit": int(n), "d": int(d),
                  "ll_trace": ll_trace, "n_iters": len(ll_trace), "collapsed": collapsed,
                  "void": False, "void_reason": None, "retry_used": bool(_retry)})
    return atlas


def atlas_to_cpu(atlas):
    """Detach the atlas tensors to CPU fp32 (for banking)."""
    import torch
    out = dict(atlas)
    for k in ("log_pi", "mu", "U", "lam", "sigma2"):
        out[k] = atlas[k].detach().to("cpu", torch.float32)
    return out


def atlas_sha(atlas):
    """Content hash over the atlas parameters (banked beside every fit)."""
    import hashlib
    h = hashlib.sha256()
    for k in ("log_pi", "mu", "U", "lam", "sigma2"):
        h.update(atlas[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ===========================================================================
# The FROZEN reconstruction rule + swap hook.
# ===========================================================================
def reconstruct(X, atlas, chunk=65536):
    """x_hat = sum_c gamma_c(x) [mu_c + U_c U_c^T (x - mu_c)] — the frozen rule, verbatim.
    fp32 in, fp32 out, same shape."""
    import torch
    assert not atlas.get("void"), "reconstruct() on a VOID atlas"
    mu, U = atlas["mu"], atlas["U"]
    K, d, r = U.shape
    U_flat = U.permute(0, 2, 1).reshape(K * r, d)                          # (K*r, d)
    mproj = torch.einsum("kd,kdr->kr", mu, U).unsqueeze(0)                 # (1, K, r)
    out = torch.empty_like(X)
    for s in range(0, X.shape[0], chunk):
        xc = X[s:s + chunk]
        b = xc.shape[0]
        log_gamma, _ = _log_resp(xc, atlas)
        gamma = log_gamma.exp()                                            # (b, K)
        yproj = (xc @ U_flat.T).reshape(b, K, r) - mproj                   # (b, K, r)
        # x_hat = (gamma @ mu) + reshape(gamma * yproj) @ U_flat — never a (b, K, d) tensor
        out[s:s + b] = gamma @ mu + (gamma.unsqueeze(-1) * yproj).reshape(b, K * r) @ U_flat
    return out


class SplatSwapHook:
    """Forward hook on ONE decoder layer replacing the OUTPUT residual with its frozen-rule
    mixture reconstruction at ALL positions (prefill + decode): h <- x_hat(h). fp32 compute,
    cast back to the stream dtype (the AblateHook B2 lesson). Context-managed. A residual-dim
    mismatch RAISES; no try/except."""

    def __init__(self, layer_module, atlas):
        assert not atlas.get("void"), "SplatSwapHook on a VOID atlas"
        self.layer = layer_module
        self.atlas = atlas
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output            # (b, T, d)
        d_atlas = self.atlas["mu"].shape[1]
        if hs.shape[-1] != d_atlas:
            raise RuntimeError(f"SplatSwapHook: atlas d={d_atlas} != residual d={hs.shape[-1]}")
        flat = hs.reshape(-1, hs.shape[-1]).float()
        new = reconstruct(flat, self.atlas).to(hs.dtype).reshape(hs.shape)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


# ===========================================================================
# P1 plant generators + arrangement statistics (offline gate; GPU-scale P1 reuses them).
# ===========================================================================
def _embed_plane(low, d, seed):
    """Embed (n, k) low-dim points into R^d via a random orthonormal frame + isotropic noise-free
    lift. Deterministic."""
    import torch
    n, k = low.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 77)
    A = torch.randn(d, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return low @ Q[:, :k].T


def make_circle(n, d, seed, radius=6.0, noise=0.15):
    import math

    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    th = torch.rand(n, generator=g) * 2 * math.pi
    low = torch.stack([radius * th.cos(), radius * th.sin()], 1)
    x = _embed_plane(low, d, seed) + noise * torch.randn(n, d, generator=g)
    return x.float()


def make_two_cluster(n, d, seed, sep=8.0, noise=1.0):
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    lab = (torch.arange(n) % 2).float()
    low = torch.stack([(lab * 2 - 1) * sep / 2, torch.zeros(n)], 1)
    x = _embed_plane(low, d, seed) + noise * torch.randn(n, d, generator=g)
    return x.float(), lab


def make_curved_manifold(n, d, seed, curv=0.35, extent=6.0, noise=0.15):
    """A curved 2-manifold (paraboloid patch): (s, t) -> (s, t, curv*(s^2+t^2)) in a 3-frame."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    st = (torch.rand(n, 2, generator=g) - 0.5) * 2 * extent
    z = curv * (st ** 2).sum(1, keepdim=True)
    low = torch.cat([st, z], 1)
    x = _embed_plane(low, d, seed) + noise * torch.randn(n, d, generator=g)
    return x.float()


def _live_mu(atlas):
    """Component means with weight >= MIN_WEIGHT. Dead components stay RECORDED in the atlas
    (the frozen intercurrent rule) but carry garbage means (near-zero responsibility mass), so
    arrangement STATISTICS exclude them (D15)."""
    import torch
    w = atlas["log_pi"].exp().cpu()
    mu = atlas["mu"].cpu()
    live = w >= MIN_WEIGHT
    assert int(live.sum()) >= 2, f"only {int(live.sum())} live components — atlas degenerate"
    return mu[live]


def circle_stat(atlas, X_ref):
    """Radial coefficient-of-variation of LIVE component means in the data's top-2 PCA plane.
    Small = means arranged on a circle. Pure geometry; the P1 rule compares real vs twin."""
    import torch
    mu = _live_mu(atlas)
    Xc = X_ref.cpu() - X_ref.cpu().mean(0)
    torch.manual_seed(0)
    _, _, V = torch.svd_lowrank(Xc, q=8, niter=4)
    P = V[:, :2]
    proj = (mu - X_ref.cpu().mean(0)) @ P
    radii = proj.norm(dim=1)
    assert radii.numel() >= 4, "circle_stat needs >= 4 components"
    return float((radii.std() / radii.mean().clamp(min=1e-9)).item())


def ll_per_point(atlas, X):
    """Mixture log-likelihood per point (fp32) — the K-comparison statistic for P1's
    bimodality discriminator (a genuinely bimodal cloud gains far more from K=1->2 than its
    Gaussian twin does; posterior purity does NOT discriminate — a K=2 fit on the twin splits
    the elongated Gaussian along the separation axis and scores high purity spuriously)."""
    _lg, total = _log_resp(X, atlas)
    return total / X.shape[0]


def cluster_purity(atlas, X, labels):
    """Fraction of points whose argmax-responsibility component is label-pure under the best
    component->label map (K=2 fit on the two-cluster plant)."""
    import torch
    log_gamma, _ = _log_resp(X, atlas)
    a = log_gamma.argmax(1)
    best = 0.0
    for c0 in (0.0, 1.0):
        ok = ((a.cpu() == 0) & (labels == c0)) | ((a.cpu() == 1) & (labels == (1 - c0)))
        best = max(best, float(ok.float().mean().item()))
    return best


def manifold_dist_stat(atlas, curv, seed, d):
    """Mean distance of LIVE component means to the TRUE planted manifold, computed in the
    planted 3-frame (re-derived from the same seed's embedding). Small = means on the sheet."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 77)
    A = torch.randn(d, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    mu = _live_mu(atlas)
    mu3 = mu @ Q[:, :3]                                      # coordinates in the planted frame
    off_frame = (mu - mu3 @ Q[:, :3].T).norm(dim=1)
    resid = (mu3[:, 2] - curv * (mu3[:, 0] ** 2 + mu3[:, 1] ** 2)).abs()
    return float((resid + off_frame).mean().item())
