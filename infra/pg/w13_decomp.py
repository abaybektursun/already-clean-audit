"""W13 — ΔW parameter-decomposition arms (the two confirmatory instruments).

Two arms, both fp64, CPU-only (no torch.cuda — Tier-1/2 run on the r-family Standard quota):

  * LOW-RANK arm  — full fp64 `torch.linalg.svd` truncation (Eckart-Young), NOT randomized
    `svd_lowrank` (the randomized routine cannot reconstruct the eff-rank tail the question needs;
    prereg §3.1). N(target) = smallest rank whose captured σ² energy ≥ target.
  * SPARSE-DICT arm — overcomplete dictionary learning on ΔW rows: min ‖X − A Dᵀ‖²_F + λ‖A‖₁
    (prereg §2.3, §3.2). Hand-rolled ISTA (sparse-code) + MOD (dict update); no sklearn dep on
    the pg image. Active-atom count = #atoms used by ≥1 row.

This module is the *instrument*; the synthetic positive controls that gate it live in the
preflight job (infra/aws_pg/jobs/w13_preflight.py). The behavioral readout (`N_behav` via the
GraftLM no-retrain graft) is a separate GPU step (prereg §3 Tier-3) — these arms produce the
candidate component sets it grafts.

Design rules (CLAUDE.md): no try/except — fail loud with full trace; plain torch/JSON-able
returns; all math fp64 on CPU.
"""
from __future__ import annotations

import torch

torch.manual_seed(0)  # module-level determinism floor; every routine also takes an explicit seed


# ----------------------------------------------------------------------------- fp64 SVD identity
def svd_fp64(w):
    """fp64 thin SVD of a 2-D matrix (mirrors infra.pg.gain_routing._svd_fp64). CPU."""
    assert w.ndim == 2, f"need a 2-D matrix, got {tuple(w.shape)}"
    U, s, Vh = torch.linalg.svd(w.detach().to(torch.float64), full_matrices=False)
    return U, s, Vh


def recon_identity(w):
    """§9-1c reconstruction identity: ‖U Σ Vᵀ − W‖_F / ‖W‖_F. fp64 → must be < 1e-10."""
    U, s, Vh = svd_fp64(w)
    recon = (U * s) @ Vh
    return (recon - w.to(torch.float64)).norm().item() / w.to(torch.float64).norm().clamp_min(1e-30).item()


# --------------------------------------------------------------------------------- low-rank arm
def lowrank_n(w, energy_target):
    """N for the low-rank arm = smallest rank r whose cumulative σ² energy fraction ≥ target.

    Returns (n, energy_curve) where energy_curve[r-1] = fraction captured by the top-r triplets.
    """
    assert 0.0 < energy_target <= 1.0
    _, s, _ = svd_fp64(w)
    e = (s ** 2)
    cum = torch.cumsum(e, 0) / e.sum().clamp_min(1e-30)
    n = int((cum < energy_target).sum().item()) + 1   # first index reaching the target (1-based)
    n = min(n, s.shape[0])
    return n, cum.tolist()


# ------------------------------------------------------------------------------- sparse-dict arm
def _soft_threshold(z, thr):
    return torch.sign(z) * torch.clamp(z.abs() - thr, min=0.0)


def lam_max(X, D):
    """The smallest λ that drives all codes to zero (data-driven L1 scale): max|Xᵀd_k| over (i,k).
    Setting λ = lam_frac · lam_max makes the penalty scale-free (no hand-tuned absolute λ)."""
    return (X.to(torch.float64) @ D.to(torch.float64).transpose(0, 1)).abs().max().item()


def _debias(X, A, D, rel_tol=1e-3):
    """Per-row least-squares refit on the L1-selected support → removes L1 shrinkage bias so the
    reconstruction is tight (the support stays sparse; only the active coefficients are corrected).
    Without this the L1 estimate is biased and G4's real-vs-shuffle residual gap collapses."""
    X = X.to(torch.float64); D = D.to(torch.float64)
    A = A.clone()
    row_thr = rel_tol * A.abs().max(dim=1, keepdim=True).values.clamp_min(1e-30)
    for i in range(X.shape[0]):
        supp = (A[i].abs() > row_thr[i]).nonzero(as_tuple=True)[0]
        A[i] = 0.0
        if supp.numel() > 0:
            c = torch.linalg.lstsq(D[supp].transpose(0, 1), X[i]).solution   # x_i ≈ D[supp]ᵀ c
            A[i, supp] = c
    return A


def sparse_code(X, D, lam, n_iter, debias=False):
    """Sparse-code rows of X against a FIXED dictionary D (atoms = rows of D, unit-norm).

    min_A 0.5‖X − A Dᵀ‖²_F + λ‖A‖₁ via FISTA (Nesterov momentum → O(1/k²)). X:(R,d) D:(K,d) → A:(R,K).
    Step η = 1/L, L = spectral norm of (D Dᵀ). `lam` is the ABSOLUTE threshold scale (caller sets it
    via lam_frac·lam_max). debias=True LS-refits on the final support. fp64.
    """
    X = X.to(torch.float64); D = D.to(torch.float64)
    R, K = X.shape[0], D.shape[0]
    L = torch.linalg.matrix_norm(D @ D.transpose(0, 1), ord=2).item()
    eta = 1.0 / max(L, 1e-12)
    A = torch.zeros(R, K, dtype=torch.float64)
    Y = A.clone(); t = 1.0
    for _ in range(n_iter):
        grad = (Y @ D - X) @ D.transpose(0, 1)        # (R,K)
        A_next = _soft_threshold(Y - eta * grad, eta * lam)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Y = A_next + ((t - 1.0) / t_next) * (A_next - A)
        A, t = A_next, t_next
    return _debias(X, A, D) if debias else A


def dict_learn(X, n_atoms, lam_frac, n_outer, n_iter, seed):
    """Overcomplete dictionary learning on rows of X (MOD: alternate FISTA sparse-code + LS dict).

    Returns (D, A): D:(n_atoms,d) unit-norm atoms, A:(R,n_atoms) sparse codes (debiased at the end).
    `lam_frac` is the fraction of the data-driven lam_max (scale-free). Atoms init from random rows.
    """
    X = X.to(torch.float64)
    R, d = X.shape
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(R, generator=g)[:n_atoms]
    D = X[idx].clone()
    if D.shape[0] < n_atoms:                          # pad with Gaussian atoms if R < n_atoms
        extra = torch.randn(n_atoms - D.shape[0], d, generator=g, dtype=torch.float64)
        D = torch.cat([D, extra], 0)
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    A = None
    for _ in range(n_outer):
        lam = lam_frac * lam_max(X, D)
        A = sparse_code(X, D, lam, n_iter)
        D = torch.linalg.pinv(A) @ X                  # MOD dict update (least squares)
        norms = D.norm(dim=1, keepdim=True)
        dead = (norms < 1e-9).squeeze(1)              # re-seed dead atoms from worst-fit rows
        if dead.any():
            resid = (X - A @ D).norm(dim=1)
            worst = torch.topk(resid, int(dead.sum().item())).indices
            D[dead] = X[worst]
            norms = D.norm(dim=1, keepdim=True)
        D = D / norms.clamp_min(1e-12)
    A = sparse_code(X, D, lam_frac * lam_max(X, D), n_iter, debias=True)
    return D, A


def active_atoms(A, rel_tol=1e-3):
    """#atoms used by ≥1 row = #columns of A whose ℓ2 norm exceeds rel_tol × the max column norm."""
    col = A.to(torch.float64).norm(dim=0)
    thr = rel_tol * col.max().clamp_min(1e-30)
    return int((col > thr).sum().item())


def row_sparsity(A, rel_tol=1e-3):
    """Median per-row count of active atoms (|a| > rel_tol × that row's max |a|)."""
    A = A.to(torch.float64).abs()
    thr = rel_tol * A.max(dim=1, keepdim=True).values.clamp_min(1e-30)
    return float((A > thr).sum(dim=1).double().median().item())


def recon_residual(X, A, D):
    """Relative Frobenius reconstruction residual ‖X − A Dᵀ‖_F / ‖X‖_F."""
    X = X.to(torch.float64)
    return (X - A.to(torch.float64) @ D.to(torch.float64)).norm().item() / X.norm().clamp_min(1e-30).item()


# ----------------------------------------------------------- synthetic structure planters (§5)
def plant_lowrank(m, n, r0, seed):
    """Rank-r0 matrix with a sharp spectral gap (the easy regime, pos-control #1a)."""
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(m, r0, generator=g, dtype=torch.float64)
    V = torch.randn(n, r0, generator=g, dtype=torch.float64)
    return U @ V.transpose(0, 1)


def plant_sparse_overcomplete(n_rows, d, n_atoms0, s0, seed):
    """ΔW that is FULL-RANK in the SVD basis yet s0-sparse over a random overcomplete dictionary
    (pos-control #2, the adversarial tail regime the verdict turns on). Each row = sum of s0
    randomly-chosen unit atoms (Gaussian → rotated out of axis alignment). With n_atoms0 > d and
    n_rows > d the atoms span R^d ⇒ SVD-rank = d (full), but each row is s0-sparse in the dict.

    Returns (X, D0, A0): X:(n_rows,d), D0:(n_atoms0,d) unit atoms, A0:(n_rows,n_atoms0) s0-sparse.
    """
    assert n_atoms0 > d and n_rows > d, "need overcomplete dict + enough rows for full SVD rank"
    g = torch.Generator().manual_seed(seed)
    D0 = torch.randn(n_atoms0, d, generator=g, dtype=torch.float64)
    D0 = D0 / D0.norm(dim=1, keepdim=True).clamp_min(1e-12)
    A0 = torch.zeros(n_rows, n_atoms0, dtype=torch.float64)
    for i in range(n_rows):
        atoms = torch.randperm(n_atoms0, generator=g)[:s0]
        A0[i, atoms] = torch.randn(s0, generator=g, dtype=torch.float64)
    return A0 @ D0, D0, A0


def atom_recovery(D_learned, D0, cos_thr=0.9):
    """Fraction of planted atoms (rows of D0) matched by some learned atom at |cosine| ≥ cos_thr.

    The #2 HALT criterion: the method must DETECT a sparse basis that IS present, else a null
    N≈eff-rank is method impotence, not evidence (prereg §5 pos-control #2).
    """
    Dl = D_learned.to(torch.float64); Dl = Dl / Dl.norm(dim=1, keepdim=True).clamp_min(1e-12)
    D0n = D0.to(torch.float64); D0n = D0n / D0n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (D0n @ Dl.transpose(0, 1)).abs()            # (K0, K) — planted × learned
    best = cos.max(dim=1).values
    return float((best >= cos_thr).double().mean().item())


def shuffle_entries(w, seed):
    """Permute all entries of W within the matrix (prereg §5 shuffle test — structure must die)."""
    g = torch.Generator().manual_seed(seed)
    flat = w.to(torch.float64).flatten()
    return flat[torch.randperm(flat.numel(), generator=g)].reshape(w.shape)
