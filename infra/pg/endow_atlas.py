"""ENDOW-F storage-form atlas axes (endow spec §5) — the two-kinds-law instrument.

Pure math over per-object RF sweeps, weight-free unit-tested (infra/pg/tests/test_endow_atlas.py).
Three mechanical, frozen axes + the threshold-free law predicate:

  * COMPACTNESS k_90 — the smallest rank at which the ablation effect E(k)=max(0,1−RF(k)) reaches
    90% of its grid-max value, WITH the saturation rule: if E is still rising at the grid max by
    more than the null band's width, the object is 'distributed (k_90 undefined at grid)' — the
    guard that stops an estimator's rank cutoff from faking compactness.
  * CANONICITY c — mean pairwise principal-subspace overlap across ≥3 estimation draws at k_can,
    normalized against the chance floor k/d as (o − k/d)/(1 − k/d) (so a distributed object with
    chance overlap reads c≈0, and a canonical object reads c≈1).
  * DEPTH-SPREAD PR — participation ratio of the per-layer effects (descriptive; no verdict rides
    on it).

Two-kinds law (H_two_kinds) holds iff BOTH: every primitive's k_90 < every capability-read's k_90
(a distributed/undefined read counts as larger than any defined k_90), and every primitive's c >
every capability-read's c. Any violating pair is a NAMED counterexample. The comparison is over
labels/orderings, never cross-object effect sizes (home readouts are heterogeneous — §5.5).

Design rules (CLAUDE.md): NO try/except hiding; heavy imports inside functions; simplest math.
"""
from __future__ import annotations

from infra.pg.ctrliface_core import principal_angle_overlap


# ===========================================================================
# Compactness — E(k) and the saturation-guarded k_90.
# ===========================================================================
def effect_curve(rf_by_k):
    """E(k) = max(0, 1 − RF(k)) over the frozen k-grid. rf_by_k: {k:int → RF:float}."""
    assert len(rf_by_k) >= 2, "need ≥2 grid points for a compactness curve"
    return {int(k): max(0.0, 1.0 - float(rf)) for k, rf in rf_by_k.items()}


def k90(rf_by_k, null_width):
    """Saturation-guarded compactness. Returns {k90:int|None, saturated:bool, e_max, e_target,
    rise_at_max}. Distributed (k90=None) iff E is still rising at grid-max by more than null_width
    — the estimator-shape guard (§5.2). Else k90 = the smallest grid k with E(k) ≥ 0.9·E(k_max)."""
    E = effect_curve(rf_by_k)
    grid = sorted(E)
    k_max = grid[-1]
    e_max = E[k_max]
    rise = e_max - E[grid[-2]]                                   # rise over the last grid step
    if rise > float(null_width):
        return {"k90": None, "saturated": True, "e_max": e_max,
                "e_target": 0.9 * e_max, "rise_at_max": rise}
    target = 0.9 * e_max
    reached = [k for k in grid if E[k] >= target]
    assert reached, "no grid point reaches 0.9·E_max despite non-rising tail — degenerate curve"
    return {"k90": int(min(reached)), "saturated": False, "e_max": e_max,
            "e_target": target, "rise_at_max": rise}


# ===========================================================================
# Canonicity — normalized mean pairwise draw-overlap at k_can.
# ===========================================================================
def pairwise_overlap_mean(draws):
    """Mean principal-subspace overlap over all unordered pairs of estimation draws. `draws` is a
    list of ≥2 orthonormal bases (d, k_can) — the caller orthonormalizes and fixes k_can."""
    assert len(draws) >= 2, "canonicity needs ≥2 draws"
    ks = {int(d.shape[1]) for d in draws}
    assert len(ks) == 1, f"draws must share k_can, got {ks}"
    ovs = []
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            o, _ = principal_angle_overlap(draws[i], draws[j])
            ovs.append(o)
    return float(sum(ovs) / len(ovs))


def normalized_canonicity(mean_overlap, k, d):
    """(o − k/d)/(1 − k/d) — mean pairwise overlap referenced to the chance floor k/d and the
    ceiling 1. c≈0 for chance-overlapping (distributed) draws, c≈1 for a canonical basis."""
    floor = float(k) / float(d)
    assert 0.0 <= floor < 1.0, f"degenerate k/d floor {floor}"
    return (float(mean_overlap) - floor) / (1.0 - floor)


def canonicity(draws, d):
    """End-to-end normalized canonicity c from the estimation draws (k_can = draws' shared rank)."""
    k = int(draws[0].shape[1])
    return normalized_canonicity(pairwise_overlap_mean(draws), k, d)


# ===========================================================================
# Depth-spread — participation ratio (descriptive).
# ===========================================================================
def participation_ratio(effect_by_layer):
    """PR = (Σ e_L)² / Σ e_L² over per-layer effects e_L = max(0, 1 − RF(L)). 1 = one layer,
    n = uniform over n layers. Descriptive only."""
    import numpy as np
    e = np.asarray([max(0.0, float(x)) for x in effect_by_layer.values()], dtype=float)
    s2 = float((e ** 2).sum())
    assert s2 > 0, "all-zero per-layer effects — no depth spread defined"
    return float(e.sum() ** 2 / s2)


# ===========================================================================
# The frozen, threshold-free two-kinds law.
# ===========================================================================
def _k90_val(x):
    return float("inf") if x is None else float(x)


def two_kinds_law(primitives, reads):
    """primitives / reads: {name → {'k90': int|None, 'c': float}}. Returns the verdict dict.
    H_two_kinds iff every primitive is strictly more compact (smaller k90) AND strictly more
    canonical (larger c) than every read. A distributed/undefined k90 counts as +∞ (so a
    distributed READ is consistent with the law; a distributed PRIMITIVE violates it)."""
    assert primitives and reads, "law needs ≥1 primitive and ≥1 read"
    counter = []
    for pn, pv in primitives.items():
        for rn, rv in reads.items():
            if not _k90_val(pv["k90"]) < _k90_val(rv["k90"]):
                counter.append({"primitive": pn, "read": rn, "axis": "compactness",
                                "p_k90": pv["k90"], "r_k90": rv["k90"]})
            if not float(pv["c"]) > float(rv["c"]):
                counter.append({"primitive": pn, "read": rn, "axis": "canonicity",
                                "p_c": pv["c"], "r_c": rv["c"]})
    holds = not counter
    return {"holds": holds, "counterexamples": counter,
            "label": "H_two_kinds" if holds else "H_continuum_or_counterexample"}
