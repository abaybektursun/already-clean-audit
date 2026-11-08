"""TENANT pure core — the four NEW instruments (spec §11), numpy-only so every one is
weight-free-testable on the laptop mirror AND the CPU gate box before any GPU spend.

1. leverage sampler   — random k-dim basis matching a target row-leverage profile,
                        Haar-conditional on it (the Wing-P confound-killer arm, P1).
2. confinement check  — input-span leakage of a trained adapter factor (P3).
3. q/k product-space  — cancellation index of paired steps in W_q^T W_k (X-paired).
4. lag-autocorrelation— take-and-revert signature of per-step in-core components (X-orbit).

All fail-loud; no try/except."""
from __future__ import annotations

import numpy as np


# ===========================================================================
# 1. Leverage-matched subspace sampler (P1)
# ===========================================================================
def leverage_profile(Q):
    """Row leverages of an orthonormal basis Q (d, k): l_i = ||Q[i, :]||^2, sum = k.
    atol 1e-5: banked bases are stored fp32 (Q^T Q deviates ~1e-7 at d=2048); the profile is
    computed on the re-orthonormalized copy so the target still sums to k exactly."""
    Q = np.asarray(Q, dtype=np.float64)
    QtQ = Q.T @ Q
    assert np.allclose(QtQ, np.eye(Q.shape[1]), atol=1e-5), "Q not orthonormal"
    Q, R = np.linalg.qr(Q)
    Q = Q * np.sign(np.diagonal(R))[None, :]
    return (Q ** 2).sum(axis=1)


def leverage_matched_basis(l_target, k, seed, n_iter=300, tol=0.02):
    """A random orthonormal (d, k) basis whose leverage profile matches `l_target`,
    Haar-random conditional on it. Alternating projection: scale rows toward the target
    leverage, re-orthonormalize (QR), repeat until max|l - l_target| <= tol.

    Direction content beyond the profile comes only from the seeded Gaussian init, so two
    draws share nothing but leverage (unit-tested: draw-vs-draw overlap == draw-vs-core
    overlap in distribution). Fail-loud if not converged."""
    l_target = np.asarray(l_target, dtype=np.float64)
    d = l_target.shape[0]
    assert abs(l_target.sum() - k) < 1e-6, f"target leverages must sum to k ({l_target.sum()})"
    assert (l_target >= 0).all() and (l_target <= 1 + 1e-12).all()
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, k)) * np.sqrt(np.maximum(l_target, 1e-12))[:, None]
    Q, _ = np.linalg.qr(G)
    for _ in range(n_iter):
        l = (Q ** 2).sum(axis=1)
        err = float(np.abs(l - l_target).max())
        if err <= tol:
            return Q, err
        scale = np.sqrt((l_target + 1e-12) / (l + 1e-12))
        Q, R = np.linalg.qr(Q * scale[:, None])
        Q = Q * np.sign(np.diagonal(R))[None, :]
    raise RuntimeError(f"leverage sampler did not converge: max|l-l*| = {err:.4f} > {tol}")


def subspace_overlap(A, B):
    """Mean squared projection of A's columns into span(B) (both orthonormal)."""
    return float(((np.asarray(B).T @ np.asarray(A)) ** 2).sum(axis=0).mean())


# ===========================================================================
# 2. Confinement check (P3)
# ===========================================================================
def confinement_frac(B, Q):
    """Leakage of adapter factor B (r, d) outside span(Q) (d, k): ||B(I-QQ^T)||^2 / ||B||^2.
    0 = perfectly confined. Returns 0.0 for an all-zero B (vacuously confined, flagged)."""
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    tot = float((B ** 2).sum())
    if tot == 0.0:
        return 0.0
    resid = B - (B @ Q) @ Q.T
    return float((resid ** 2).sum() / tot)


# ===========================================================================
# 3. q/k product-space cancellation (X-paired)
# ===========================================================================
def qk_product_cancellation(dWq, dWk, Wq0, Wk0, Qq, Qk):
    """Cancellation index of one step in the attention read product A = W_q^T W_k,
    restricted to the paired core block (Qq on q's input axis, Qk on k's input axis).

      term_q = (dWq)^T Wk0   restricted:  Qq^T term_q Qk
      term_k = Wq0^T (dWk)   restricted likewise
      combined = term_q + term_k (first-order dA)

    index = ||combined|| / (||term_q|| + ||term_k||)  in [0, 1]:
      ~1 -> no functional cancellation; ~0 -> paired steps cancel in the product."""
    tq = Qq.T @ (np.asarray(dWq, np.float64).T @ np.asarray(Wk0, np.float64)) @ Qk
    tk = Qq.T @ (np.asarray(Wq0, np.float64).T @ np.asarray(dWk, np.float64)) @ Qk
    nq, nk = float(np.linalg.norm(tq)), float(np.linalg.norm(tk))
    denom = nq + nk
    if denom == 0.0:
        return {"index": float("nan"), "norm_q": 0.0, "norm_k": 0.0}
    return {"index": float(np.linalg.norm(tq + tk) / denom), "norm_q": nq, "norm_k": nk}


# ===========================================================================
# 4. Lag-autocorrelation of in-core step components (X-orbit)
# ===========================================================================
def lag_autocorr(V, max_lag):
    """V (T, k): per-step in-core components. Returns rho[l] for l = 1..max_lag with the
    time-mean removed:  rho_l = sum_t <v~_t, v~_{t+l}> / sum_t ||v~_t||^2.
    Take-and-revert (alternating +v/-v) -> rho_1 ~ -1; iid steps -> ~0."""
    V = np.asarray(V, dtype=np.float64)
    T = V.shape[0]
    assert T > max_lag + 2, f"need T > max_lag+2 ({T} vs {max_lag})"
    Vc = V - V.mean(axis=0, keepdims=True)
    denom = float((Vc ** 2).sum())
    assert denom > 0, "degenerate (constant) step series"
    return [float((Vc[:-l] * Vc[l:]).sum() / denom) for l in range(1, max_lag + 1)]
