"""PRAGNANZ pure core — the resample ladder, transport, optimizer projector, DOME estimator,
and the mean-cancellation decomposition. Spec: 2026-08-05_pragnanz_program_spec.md.
All torch, device-agnostic, weight-free-testable; heavy imports inside functions; no try/except.

THE LADDER'S FROZEN SAMPLING CONTRACT (P2 depends on it): per-token randomness is drawn from a
generator seeded ONLY by (ladder_seed, token_index) BEFORE any assignment is applied, so the
permutation enters exclusively through the assignment vector — r2 with the identity permutation
is BIT-IDENTICAL to r1.

  r1 resample_from_assignments(X, atlas, assign)            — arrangement PRESERVED
  r2 resample_from_assignments(X, atlas, assign[perm])      — arrangement DESTROYED, marginals
     exact (a permutation preserves the component histogram exactly)
  r3 gestalt_splat.twin_sample                               — the moment floor (REUSED)

TRANSPORT (Q2, frozen rule): x' = x + (mu_dst[pi(c)] - mu_src[c]) — pure component-mean
transport along the GESTALT correspondence assignment pi; the matched-random arm applies a
random permutation's displacement RESCALED per-token to the real transport's norm (equal
displacement, wrong direction).

OPTIMIZER PROJECTOR (M4/M5, frozen): g <- (I - Q Q^T) g per named matrix, applied to the
gradient BEFORE the optimizer state update; the random arm uses per-matrix rank-matched Haar
bases with per-matrix seeds (FULCRUM control, non-negotiable).

DOME ESTIMATOR (M2/M4 comparator; claim-level read journaled 2026-08-05, arXiv 2507.03545):
H_t = [g_j - mean_j g_j]; Y' = (1/B) H (H^T U); Y = ((t-1)/t) U Lambda + (1/(t B)) Y';
U,R = QR(Y); Lambda = |diag(R)|. Filter: g~ = (I - U U^T) g_mean.
"""
from __future__ import annotations

LADDER_RUNGS = ("r0", "r1", "r2", "r3")
HALF_K_SENSITIVITY = True


# ===========================================================================
# The ladder samplers.
# ===========================================================================
def hard_assignments(X, atlas, chunk=65536):
    """Argmax-responsibility component per token (the arrangement object the ladder permutes)."""
    import torch
    from infra.pg.gestalt_splat import _log_resp
    out = []
    for s in range(0, X.shape[0], chunk):
        lg, _ = _log_resp(X[s:s + chunk], atlas)
        out.append(lg.argmax(1))
    return torch.cat(out)


def _token_noise(n, r, d, ladder_seed, device):
    """(z (n, r), eps (n, d)) drawn from a generator seeded by ladder_seed ONLY — never by the
    assignment — so permuting assignments cannot change the noise (the P2 contract)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(ladder_seed))
    z = torch.randn(n, r, generator=g, dtype=torch.float32).to(device)
    eps = torch.randn(n, d, generator=g, dtype=torch.float32).to(device)
    return z, eps


def resample_from_assignments(atlas, assign, ladder_seed):
    """Redraw every token from the component named by `assign`:
        x' = mu_c + U_c (sqrt(lam_c) * z) + sqrt(sigma2_c) * eps
    Identical (ladder_seed, token) noise regardless of `assign` (P2)."""
    import torch
    mu, U, lam, s2 = atlas["mu"], atlas["U"], atlas["lam"], atlas["sigma2"]
    K, d, r = U.shape
    n = assign.shape[0]
    z, eps = _token_noise(n, r, d, ladder_seed, mu.device)
    a = assign.to(mu.device)
    scaled = z * lam.clamp(min=0.0).sqrt()[a]                       # (n, r)
    inplane = torch.einsum("nr,ndr->nd", scaled, U[a])              # (n, d) via gathered bases
    return mu[a] + inplane + s2.clamp(min=0.0).sqrt()[a].unsqueeze(1) * eps


def permute_assignments(assign, perm_seed):
    """A seeded uniform permutation of the assignment VECTOR across tokens — the component
    histogram (mixture marginal) is preserved EXACTLY; which token carries which component
    (the arrangement) is destroyed."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(perm_seed))
    perm = torch.randperm(assign.shape[0], generator=g)
    return assign[perm.to(assign.device)]


# ===========================================================================
# Q2 transport.
# ===========================================================================
def transport_shift(atlas_src, atlas_dst, assign, corr_cols):
    """Per-token shift for the frozen mean-transport rule: shift_i = mu_dst[pi(c_i)] - mu_src[c_i].
    `corr_cols` = the GESTALT correspondence assignment (row c of src -> corr_cols[c] of dst)."""
    import torch
    pi = torch.as_tensor(corr_cols, dtype=torch.long, device=atlas_src["mu"].device)
    a = assign.to(atlas_src["mu"].device)
    return atlas_dst["mu"][pi[a]] - atlas_src["mu"][a]


def random_transport_shift(real_shift, seed):
    """The matched-random arm: a random direction per token, RESCALED to the real per-token
    displacement norm (equal displacement, wrong direction)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    v = torch.randn(real_shift.shape, generator=g, dtype=torch.float32).to(real_shift.device)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return v * real_shift.norm(dim=1, keepdim=True)


# ===========================================================================
# M4/M5 optimizer projector.
# ===========================================================================
def project_out_grad(grad, Q):
    """(I - Q Q^T) grad for a (out, in) matrix gradient, Q (dim, k) orthonormal on the OUT axis
    when Q.shape[0] == grad.shape[0], else on the IN axis. fp32 compute, cast back."""
    import torch
    G = grad.float()
    if Q.shape[0] == G.shape[0]:
        R = G - Q @ (Q.T @ G)
    elif Q.shape[0] == G.shape[1]:
        R = G - (G @ Q) @ Q.T
    else:
        raise RuntimeError(f"projector dim {Q.shape[0]} matches neither axis of {tuple(G.shape)}")
    return R.to(grad.dtype)


def haar_basis(dim, k, seed):
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(dim, k, generator=g)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def project_into_grad(grad, Q):
    """Q Q^T grad (axis-matched like project_out_grad) — the M5 'placed-basis' constraint:
    with a zero-init factor, projecting every gradient INTO span(Q) keeps the factor there."""
    return grad - project_out_grad(grad, Q)


class GradProjector:
    """Per-named-matrix gradient projector applied AFTER backward, BEFORE optimizer.step()
    (call .apply(model) between them). mode 'out' removes span(Q) from the gradient (M4);
    mode 'into' keeps ONLY span(Q) (M5 placement arms)."""

    def __init__(self, bases, mode="out"):        # {param_name: Q tensor or None}
        assert mode in ("out", "into")
        self.bases, self.mode = bases, mode

    def apply(self, model):
        import torch
        fn = project_out_grad if self.mode == "out" else project_into_grad
        n_hit = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                Q = self.bases.get(name)
                if Q is None or p.grad is None:
                    continue
                p.grad.copy_(fn(p.grad, Q.to(p.grad.device)))
                n_hit += 1
        assert n_hit > 0, "GradProjector matched no gradients — name map wrong"
        return n_hit


def random_bases_like(core_bases, seed0):
    """Rank-matched Haar per matrix, per-matrix seeds (seed0 + index) — the FULCRUM control."""
    out = {}
    for i, (name, Q) in enumerate(sorted(core_bases.items())):
        out[name] = None if Q is None else haar_basis(Q.shape[0], Q.shape[1], seed0 + i)
    return out


# ===========================================================================
# DOME estimator (per the journaled claim-level read).
# ===========================================================================
class DomeEstimator:
    """Streaming randomized power method on the batch-centered per-sample gradient covariance."""

    def __init__(self, dim, k, seed=0, device=None):
        """device=None keeps state on CPU (the M2-validated path). A CUDA device keeps U/Lam
        resident there — same math, no per-update round-trip (M4 online-arm cost fix)."""
        import torch
        self._dev = device if device is not None else "cpu"
        self.U = haar_basis(dim, k, seed).to(self._dev)
        self.Lam = torch.zeros(k, device=self._dev)
        self.t = 0

    def update(self, G_samples):
        """G_samples (B, dim): per-sample gradients for one step."""
        import torch
        B = G_samples.shape[0]
        H = (G_samples - G_samples.mean(0, keepdim=True)).T.float()        # (dim, B)
        Yp = (H @ (H.T @ self.U.to(H.device))) / B                         # (dim, k)
        self.t += 1
        Y = ((self.t - 1) / self.t) * self.U.to(H.device) * self.Lam.to(H.device).unsqueeze(0) \
            + Yp / self.t
        Q, R = torch.linalg.qr(Y)
        self.U = Q.to(self._dev)
        self.Lam = torch.diagonal(R).abs().to(self._dev)
        return self

    def nuisance_fraction(self, g):
        import torch
        gf = g.float().reshape(-1)
        U = self.U.to(gf.device)
        return float((U.T @ gf).norm() / gf.norm().clamp(min=1e-30))

    def filter(self, g):
        import torch
        gf = g.float().reshape(-1)
        U = self.U.to(gf.device)
        return (gf - U @ (U.T @ gf)).reshape(g.shape).to(g.dtype)


# ===========================================================================
# M1 mean-cancellation decomposition.
# ===========================================================================
def mean_cancel_ratio(V):
    """V (T, k): per-step IN-SUBSPACE gradient component vectors. Returns
    mean2 = ||mean_t V||^2 and var = mean_t ||V_t - mean||^2, and their ratio.
    H_mean-cancel: ratio(in-core) << ratio(matched-random) — the ratio-of-ratios is the M1
    statistic (run-level CIs by the caller)."""
    import torch
    V = V.float()
    m = V.mean(0)
    mean2 = float((m * m).sum())
    var = float(((V - m) ** 2).sum(1).mean())
    return {"mean2": mean2, "var": var,
            "ratio": mean2 / var if var > 0 else float("inf"), "T": int(V.shape[0])}
