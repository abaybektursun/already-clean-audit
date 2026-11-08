"""SHARP pure core — frozen verdict maps + items+seeds stats (prereg 2026-06-13).

Stats reuse `w16_probes._boot_draws` (shared-index item bootstrap) with the W18
cross_test seed-resample pattern (child rng, seed 424242 stream separate from the
item-index stream).
"""
from __future__ import annotations

import numpy as np

from infra.pg.w16_probes import _boot_draws


def s_vs_threshold(per_seed, base, floor, thr=0.5, n_boot=2000, seed=1337):
    """P(S >= thr) one-sided via items+seeds bootstrap. per_seed = list of per-item
    0/1 arrays (one per seed), base/floor = controls per-item arrays, same index."""
    per_seed = [np.asarray(p, float) for p in per_seed]
    b, f = np.asarray(base, float), np.asarray(floor, float)
    k = len(per_seed)
    assert k >= 2 and all(p.size == b.size == f.size for p in per_seed)
    rng = np.random.default_rng(424242)

    def stat(idx):
        den = b[idx].mean() - f[idx].mean()
        if den <= 0:
            return None
        picks = rng.integers(0, k, k)
        a = np.mean([per_seed[j][idx].mean() for j in picks])
        return (a - f[idx].mean()) / den

    draws, dropped = _boot_draws(stat, b.size, n_boot, seed)
    assert dropped < n_boot // 2, f"{dropped}/{n_boot} denominator-dropped — scale HALT"
    point = float((np.mean([p.mean() for p in per_seed]) - f.mean()) / (b.mean() - f.mean()))
    p = float((1 + (draws < thr).sum()) / (1 + draws.size))
    return {"point_S": point, "ci95": [float(np.percentile(draws, 2.5)),
                                       float(np.percentile(draws, 97.5))],
            "p_one_sided": p, "thr": thr, "n_draws_dropped": int(dropped),
            "per_seed_S": [float((q.mean() - f.mean()) / (b.mean() - f.mean()))
                           for q in per_seed]}


def paired_diff_s(per_seed_a, per_seed_b, base, floor, n_boot=2000, seed=1337):
    """P(S_a − S_b > 0) one-sided, paired on the shared item index; independent
    seed-resample per arm (child rng)."""
    A = [np.asarray(p, float) for p in per_seed_a]
    B = [np.asarray(p, float) for p in per_seed_b]
    b, f = np.asarray(base, float), np.asarray(floor, float)
    ka, kb = len(A), len(B)
    rng = np.random.default_rng(424242)

    def stat(idx):
        den = b[idx].mean() - f[idx].mean()
        if den <= 0:
            return None
        pa = rng.integers(0, ka, ka)
        pb = rng.integers(0, kb, kb)
        a = np.mean([A[j][idx].mean() for j in pa])
        c = np.mean([B[j][idx].mean() for j in pb])
        return (a - c) / den

    draws, dropped = _boot_draws(stat, b.size, n_boot, seed)
    assert dropped < n_boot // 2, f"{dropped}/{n_boot} denominator-dropped — scale HALT"
    den_full = b.mean() - f.mean()
    point = float((np.mean([p.mean() for p in A]) - np.mean([q.mean() for q in B])) / den_full)
    p = float((1 + (draws <= 0).sum()) / (1 + draws.size))
    return {"point_delta": point, "ci95": [float(np.percentile(draws, 2.5)),
                                           float(np.percentile(draws, 97.5))],
            "p_one_sided": p, "n_draws_dropped": int(dropped)}


def gate_bracket(r128_sig, r256_sig):
    """SHARP-1 mechanical bracket (prereg). Grades: lower edge of (64,128] is
    DISCOVERY-anchored; upper edge of (256,512] is confirmation-anchored."""
    if r128_sig and r256_sig:
        return "gate ∈ (64, 128] (lower edge DISCOVERY-anchored at r64)"
    if r256_sig and not r128_sig:
        return "gate ∈ (128, 256]"
    if not r128_sig and not r256_sig:
        return "gate ∈ (256, 512] (upper edge confirmation-anchored at r512)"
    return ("MONOTONICITY HALT — r128 significant but r256 not; "
            "harness-indicted, no bracket emitted")


def pi0_verdict(sav64_sig, trend_sig, s64_point):
    """SHARP-2 mechanical verdict map (prereg)."""
    if sav64_sig and trend_sig:
        return ("SLOWER-NOT-TERMINAL: savings exist and grow with radius — "
                "consumed-to-floor RE-SCOPED (re-priced, not destroyed) for ≥64k")
    if trend_sig and not sav64_sig:
        return ("LEAK-ONLY (consumed STANDS): S rises with data but is "
                "indistinguishable from scratch-training at matched n")
    if sav64_sig and not trend_sig:
        return ("STATIC-SAVINGS (ambiguous): small latent remnant, not growing; "
                "16k phrasing kept, remnant noted")
    if s64_point < 0.25:
        return ("TERMINAL-CONFIRMED (at stated power): no significant savings or "
                "trend at 64k and the asymptote sits below 0.25")
    return "INDETERMINATE: drift without significance (point ≥ 0.25, nothing sig)"
