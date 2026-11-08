"""SAV-B pure core — predictors + frozen analysis machinery (prereg 2026-06-12).

Predictors (per unit, vs the Gemma-1 base):
  L2_base   ‖W_unit − W_base‖_F over 2-D decoder mats (fp32 accumulate).
  LMC_barrier  acc(midpoint shell) − ½[acc(base) + acc(unit)] on the MMLU grid bucket.
  overlap   mean cos² principal angles between top-16 right singular subspaces of
            layer-9 hidden states (64 fixed wikitext-2 prompts), unit vs base.

Analysis (frozen pre-outcome): Spearman + bootstrap-over-units CI; LOO linear R²
of baselines {L2, LMC} vs +overlap; permutation p for ΔLOO-R² (units permuted).
Standardization is per-LOO-fold (train-fold mean/std) — no leakage.
"""
from __future__ import annotations

import hashlib

import numpy as np

WIKI_N_PROMPTS = 64
WIKI_MIN_CHARS = 200
WIKI_MAX_TOKENS = 96
OVERLAP_LAYER = 9          # of 18 decoder layers; hidden_states index 9 = after layer 9
OVERLAP_K = 16


# ---------------- predictors ----------------

def wiki_prompts(lines, n=WIKI_N_PROMPTS, min_chars=WIKI_MIN_CHARS):
    """First n stripped lines with ≥min_chars from the (static) wikitext-2 test split."""
    keep = [ln.strip() for ln in lines if len(ln.strip()) >= min_chars]
    assert len(keep) >= n, f"only {len(keep)} prompts ≥{min_chars} chars"
    out = keep[:n]
    h = hashlib.sha1("\n".join(out).encode()).hexdigest()[:12]
    return out, h


def topk_right_basis(X, k=OVERLAP_K):
    """X [T, d] fp64 → top-k right singular vectors, rows [k, d]."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vh = np.linalg.svd(X, full_matrices=False)
    return Vh[:k]


def subspace_overlap(Vb, Vu):
    """Mean cos² principal angles = ‖Vb Vuᵀ‖_F² / k  (1 = identical span, →0 orthogonal)."""
    Vb, Vu = np.asarray(Vb, np.float64), np.asarray(Vu, np.float64)
    assert Vb.shape == Vu.shape
    return float(np.linalg.norm(Vb @ Vu.T, "fro") ** 2 / Vb.shape[0])


def l2_to_base(dec_u, dec_b):
    """‖W_u − W_b‖_F over shared 2-D keys (fp32). Key sets must match on the 2-D scope."""
    import torch
    k_u = {k for k, v in dec_u.items() if v.ndim == 2}
    k_b = {k for k, v in dec_b.items() if v.ndim == 2}
    assert k_u == k_b, f"2-D key mismatch: {sorted(k_u ^ k_b)[:4]}"
    tot = 0.0
    for k in sorted(k_u):
        d = dec_u[k].detach().float() - dec_b[k].detach().float()
        tot += float((d * d).sum())
    return float(np.sqrt(tot))


def midpoint_decoder(dec_u, dec_b):
    """(W_u + W_b)/2 in fp32, cast back to unit dtype; norms averaged too (full sd midpoint)."""
    assert set(dec_u) == set(dec_b), "decoder key sets differ"
    return {k: ((dec_u[k].detach().float() + dec_b[k].detach().float()) / 2).to(dec_u[k].dtype)
            for k in dec_u}


def lmc_barrier(acc_mid, acc_base, acc_unit):
    return float(acc_mid - 0.5 * (acc_base + acc_unit))


# ---------------- frozen analysis ----------------

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    assert den > 0, "constant input to spearman"
    return float((rx * ry).sum() / den)


def boot_spearman(x, y, n_boot=2000, seed=1337):
    """Bootstrap-over-units CI for spearman; constant-resample draws dropped + counted."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    draws, dropped = [], 0
    while len(draws) < n_boot:
        idx = rng.integers(0, x.size, x.size)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            dropped += 1
            assert dropped < 50 * n_boot, "degenerate bootstrap"
            continue
        draws.append(spearman(x[idx], y[idx]))
    d = np.asarray(draws)
    return {"rho": spearman(x, y), "ci95": [float(np.percentile(d, 2.5)),
                                            float(np.percentile(d, 97.5))],
            "n_dropped": dropped}


def _loo_pred(X, y):
    """LOO predictions of an intercept+linear model; per-fold standardization (no leakage)."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    n = y.size
    assert X.shape[0] == n and n >= X.shape[1] + 3, f"too few units: {n} for p={X.shape[1]}"
    preds = np.empty(n)
    for i in range(n):
        tr = np.ones(n, bool)
        tr[i] = False
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        assert (sd > 0).all(), "constant predictor column in a LOO fold"
        Z = (X[tr] - mu) / sd
        A = np.column_stack([np.ones(tr.sum()), Z])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        zi = (X[i] - mu) / sd
        preds[i] = w[0] + zi @ w[1:]
    return preds


def loo_r2(X, y):
    y = np.asarray(y, np.float64)
    preds = _loo_pred(X, y)
    sst = float(((y - y.mean()) ** 2).sum())
    assert sst > 0, "constant outcome"
    return float(1.0 - ((y - preds) ** 2).sum() / sst)


def delta_loo_perm(X_base, x_add, y, n_perm=2000, seed=1337):
    """Observed ΔLOO-R²(+x_add) and permutation p (x_add rows shuffled; +1-smoothed)."""
    X_base = np.asarray(X_base, np.float64)
    x_add = np.asarray(x_add, np.float64).reshape(-1, 1)
    y = np.asarray(y, np.float64)
    r2_b = loo_r2(X_base, y)
    r2_a = loo_r2(np.column_stack([X_base, x_add]), y)
    obs = r2_a - r2_b
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        xp = x_add[rng.permutation(y.size)]
        d = loo_r2(np.column_stack([X_base, xp]), y) - r2_b
        hits += int(d >= obs)
    return {"r2_base": r2_b, "r2_aug": r2_a, "delta": float(obs),
            "p_perm": float((1 + hits) / (1 + n_perm)), "n_perm": n_perm}
