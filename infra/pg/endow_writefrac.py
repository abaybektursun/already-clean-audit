"""ENDOW-J write-fraction battery (endow spec §6, the J3 cross-prediction) — did the robot hop
write into each named span? Weight-free unit-tested (infra/pg/tests/test_endow_writefrac.py).

Generalizes INHERIT's R6 weight-side instrument (inherit_rollout_op: ‖ΔW·U‖_F/‖ΔW‖_F vs a
matched-random subspace, no rollout) from the single U_unt-qk cell to a BATTERY of span objects
(U_unt-qk, S_vl, S_lm, S_action, matched-random) under one layer-bootstrap rank-ordering rule.

J3 rule (frozen): the conserved primitive U_unt is SPECIAL iff its per-layer write fraction is
STRICTLY below (non-overlapping bootstrap CIs) every comparison span's — the robot hop avoids
writing into U_unt MORE than into every matched named span. Killer (the pre-registered boring
branch): comparable avoidance everywhere (overlapping CIs) → low-write is generic fine-tuning
behavior, and every 'conserved' claim in the program gets a correction pass.

Design rules (CLAUDE.md): NO try/except hiding; heavy imports inside functions; simplest math.
"""
from __future__ import annotations


def write_fraction(dW, U):
    """‖ΔW·U‖_F / ‖ΔW‖_F — the fraction of the weight-change energy landing in the INPUT span
    span(U). ΔW (out, in); U (in, k) orthonormal. 0 = the hop avoids the span; 1 = it writes only
    there. The exact R6 formula (inherit_rollout_op)."""
    import torch
    D = torch.as_tensor(dW, dtype=torch.float64)
    Ub = torch.as_tensor(U, dtype=torch.float64)
    assert D.ndim == 2 and Ub.ndim == 2 and D.shape[1] == Ub.shape[0], \
        f"ΔW{tuple(D.shape)} · U{tuple(Ub.shape)} shape mismatch"
    denom = float(D.norm())
    assert denom > 0, "zero ΔW — undefined write fraction"
    return float((D @ Ub).norm()) / denom


def matched_random_fractions(dW, k, n_draws, seed):
    """Write fractions of ΔW into n_draws random k-subspaces of the INPUT space (the matched-random
    comparison span; chance ≈ √(k/in_dim))."""
    from infra.pg.projector_ablation import random_basis
    import torch
    in_d = dW.shape[1]
    return [write_fraction(dW, random_basis(in_d, k, seed * 100003 + i, "cpu", torch.float64))
            for i in range(n_draws)]


def per_layer_fractions(deltas_by_layer, U):
    """{L → write_fraction(ΔW_L, U)} across the decoder layers for one span object U."""
    return {int(L): write_fraction(dW, U) for L, dW in deltas_by_layer.items()}


def layer_bootstrap(per_layer, n_boot=2000, seed=1337):
    """Cluster-by-layer bootstrap of the mean write fraction (layers resampled with replacement —
    the §7 layer-bootstrap). Returns {mean, ci_lo, ci_hi, n_layers}."""
    import numpy as np
    v = np.asarray(list(per_layer.values()), dtype=float)
    assert v.size >= 2, "need ≥2 layers to bootstrap"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "n_layers": int(v.size)}


def specialness(unt_boot, comparison_boots):
    """The frozen J3 rule. unt_boot + each comparison_boots[name] are layer_bootstrap dicts.
    SPECIAL iff U_unt's write fraction is strictly below every comparison span's (unt.ci_hi <
    comparison.ci_lo — non-overlapping, U_unt writes less). Returns {special, beaten:{name→bool},
    verdict}. Not-special → the generic-low-write branch (invariant-core conservation re-read)."""
    beaten = {name: bool(unt_boot["ci_hi"] < cb["ci_lo"]) for name, cb in comparison_boots.items()}
    special = bool(beaten) and all(beaten.values())
    return {"special": special, "beaten": beaten,
            "verdict": "conservation_special" if special else "low_write_generic"}
