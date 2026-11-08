"""DARWIN Wave 0 — the free/CPU instruments (§4 Wave 0).

Pure functions with numpy only, so every one of them unit-tests on the control plane and the
weight-free dry-run gate (§9 step 2) is a real gate rather than a formality.

Frozen decision rules live here as module constants. Two of them are amendments recorded in the
journal on 2026-08-12, both forced by what the HUMBOLDT artifacts actually contain:

  OUTLIER_MULT = 5.0    §4/W0-1's own criterion, applied to BOTH sides of the overlap so the two
                        coordinate lists are built by the same rule rather than by two rules
                        chosen separately (which would make the Jaccard uninterpretable).
  LEVERAGE_MULT = 5.0   the bank-side threshold. §4/W0-1 froze the activation-side rule and left
                        "top-leverage coordinate lists" unquantified; using the SAME functional
                        form is the choice that keeps the two sets commensurable. The FULL
                        leverage vector is banked regardless, so any other threshold is
                        re-derivable from the artifact without re-running the arm.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import math

from infra.pg import humboldt_core as H

OUTLIER_MULT = 5.0
LEVERAGE_MULT = 5.0
FIXED_TOP_K = (32, 64, 128)        # fixed-size lists, recorded alongside the threshold rule
W1R_SHARED_FRACTION_GATE = 0.5     # §4/W1-R: mean shared-coordinate fraction >= 0.5 -> global R

# Tolerance for "these two derivations found the SAME subspace" (the W0-0 / G1 gate).
# Set by float arithmetic, not by taste: cos θ = 1 - O(2^-52) for a true match, and arccos loses
# half the mantissa there, so two bit-equivalent subspaces still report ~1.5e-8 rad. A tighter
# gate would fail for float reasons alone — a gate that fires on correct inputs is worse than no
# gate. Measured headroom: a true match clears this by >10x, a genuine mismatch exceeds 1.0 rad.
SUBSPACE_ANGLE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Coordinate sets.
# ---------------------------------------------------------------------------
def threshold_coords(values, mult: float):
    """Coordinates whose value exceeds `mult` x the median of the vector.

    The median (not the mean) is the reference because the object being detected is a heavy right
    tail: a handful of massive coordinates drags a mean up until it can no longer flag them, which
    would make the criterion quietly weaker exactly where the phenomenon is strongest.
    """
    import numpy as np
    v = np.asarray(values, dtype=np.float64).ravel()
    assert v.size > 0, "threshold_coords on an empty vector"
    assert np.isfinite(v).all(), "threshold_coords saw non-finite values"
    med = float(np.median(v))
    thr = mult * med
    idx = np.flatnonzero(v > thr)
    return {"coords": idx.astype(np.int64).tolist(), "threshold": thr, "median": med,
            "n_selected": int(idx.size), "n_total": int(v.size),
            "fraction_selected": float(idx.size / v.size), "mult": float(mult)}


def top_k_coords(values, k: int):
    """The k largest coordinates, ties broken by index so the list is deterministic."""
    import numpy as np
    v = np.asarray(values, dtype=np.float64).ravel()
    assert 0 < k <= v.size, f"k={k} outside 1..{v.size}"
    order = np.lexsort((np.arange(v.size), -v))
    return order[:k].astype(np.int64).tolist()


def jaccard(a, b) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def log_hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """log P[X >= k] for X ~ Hypergeometric(N population, K successes, n draws).

    Implemented with lgamma rather than scipy: the pg env AMI is not guaranteed to carry scipy,
    and a missing import discovered on the instance costs a launch. Summing the upper tail in log
    space keeps it exact where the p-values matter (deep tails).
    """
    assert 0 <= K <= N and 0 <= n <= N, f"bad hypergeometric params N={N} K={K} n={n}"
    assert 0 <= k <= n, f"k={k} outside 0..n={n}"
    lg = math.lgamma

    def lchoose(a, b):
        if b < 0 or b > a:
            return -math.inf
        return lg(a + 1) - lg(b + 1) - lg(a - b + 1)

    denom = lchoose(N, n)
    hi = min(K, n)
    terms = []
    for i in range(k, hi + 1):
        t = lchoose(K, i) + lchoose(N - K, n - i) - denom
        if t > -math.inf:
            terms.append(t)
    if not terms:
        return -math.inf
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def overlap_stats(a, b, n_total: int) -> dict:
    """The W0-1 overlap readout between two coordinate lists in a space of `n_total` dims."""
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    inter = len(sa & sb)
    logp = (log_hypergeom_sf(inter, n_total, len(sa), len(sb))
            if (sa and sb) else float("nan"))
    expected = len(sa) * len(sb) / n_total if n_total else float("nan")
    return {"n_a": len(sa), "n_b": len(sb), "n_intersection": inter,
            "jaccard": jaccard(sa, sb), "expected_intersection": expected,
            "enrichment": (inter / expected) if expected else float("nan"),
            "log_p_hypergeom": logp,
            "p_hypergeom": math.exp(logp) if logp > -700 else 0.0,
            "n_total": int(n_total)}


# ---------------------------------------------------------------------------
# W0-2 — cross-layer agreement.
# ---------------------------------------------------------------------------
def shared_coordinate_fraction(coords_a, coords_b) -> float:
    """|A ∩ B| / min(|A|, |B|) — the fraction of the SMALLER list the two share.

    Jaccard is the wrong statistic for §4/W1-R's gate: it penalises a size mismatch between two
    lists, and the question W1-R asks is whether one layer's spiky coordinates are reused by
    another, not whether the two layers are equally spiky.
    """
    sa, sb = set(int(x) for x in coords_a), set(int(x) for x in coords_b)
    if not sa or not sb:
        return float("nan")
    return len(sa & sb) / min(len(sa), len(sb))


def subspace_agreement(U, V) -> dict:
    """Principal-angle summary between two banks of the same ambient space.

    `mean_cos2` = (1/k)Σcos²θ is the Grassmann overlap: 1 for identical subspaces, k'/n for
    independent ones. Recorded next to the raw angles so a reader is never forced to re-derive it.
    """
    import numpy as np
    ang = H.principal_angles(U, V)
    c2 = np.cos(ang) ** 2
    return {"angles_rad": ang.astype(float).tolist(),
            "mean_cos2": float(c2.mean()), "max_cos2": float(c2.max()),
            "min_angle_rad": float(ang.min()), "max_angle_rad": float(ang.max()),
            "k": int(ang.size)}


def w1r_routing(mean_shared_fraction: float) -> dict:
    """§4/W1-R's mechanical branch: >= 0.5 -> one global reader/writer; else per-layer.

    Returned as a recorded decision with its input, so the choice is auditable from the artifact
    and cannot be re-read after unblinding (feedback_no_post_unblind_rule_swap).
    """
    v = float(mean_shared_fraction)
    assert v == v, "w1r_routing on a NaN shared fraction"
    return {"gate": "W1R_routing", "mean_shared_fraction": v,
            "threshold": W1R_SHARED_FRACTION_GATE,
            "variant": "global" if v >= W1R_SHARED_FRACTION_GATE else "per_layer"}


# ---------------------------------------------------------------------------
# W0-0 — bank re-derivation identity.
# ---------------------------------------------------------------------------
def sign_align(U_new, U_ref):
    """Flip the sign of each column of U_new to match U_ref.

    A singular/eigen vector is defined only up to sign, so a raw elementwise diff between two
    correct derivations can be 2.0. Aligning signs first makes the elementwise number mean what a
    reader assumes it means.
    """
    import numpy as np
    A = np.asarray(U_new, dtype=np.float64)
    B = np.asarray(U_ref, dtype=np.float64)
    assert A.shape == B.shape, f"shape mismatch {A.shape} vs {B.shape}"
    s = np.sign((A * B).sum(axis=0))
    s[s == 0] = 1.0
    return A * s


def bank_identity(U_new, U_ref) -> dict:
    """Compare a re-derived bank against the banked one.

    Two numbers, and only the second is a gate:
      * `max_abs_diff_sign_aligned` — the elementwise agreement. RECORDED. Bit-identity of the
        BASIS is not a well-posed requirement: within a degenerate (or near-degenerate) singular
        subspace any orthogonal rotation is an equally correct answer, and LAPACK's choice depends
        on the BLAS thread count. A gate on this number would fail for reasons that have nothing
        to do with whether the bank is the right object.
      * `max_principal_angle_rad` — the SUBSPACE agreement. GATED. This is the property the spec's
        G1 is actually asserting: that the re-derivation lands on the same subspace.
    """
    import numpy as np
    A = sign_align(U_new, U_ref)
    B = np.asarray(U_ref, dtype=np.float64)
    ang = H.principal_angles(A, B)
    return {"max_abs_diff_sign_aligned": float(np.abs(A - B).max()),
            "max_principal_angle_rad": float(ang.max()),
            "mean_cos2": float((np.cos(ang) ** 2).mean()),
            "bit_identical": bool(np.array_equal(np.asarray(U_new), B)),
            "k": int(B.shape[1]), "n": int(B.shape[0])}


# ---------------------------------------------------------------------------
# W0-7 — band-deviation OOD score and its ROC.
# ---------------------------------------------------------------------------
def band_deviation(series, lo: float, hi: float):
    """Per-token deviation from a normal band: 0 inside [lo, hi], else the signed distance out.

    Absolute distance (not squared) so the score's units stay the units of e_U, and a reader can
    read a threshold off the ROC in the same units the atlas records.
    """
    import numpy as np
    x = np.asarray(series, dtype=np.float64).ravel()
    assert hi >= lo, f"inverted band [{lo}, {hi}]"
    return np.maximum(0.0, np.maximum(lo - x, x - hi))


def roc_auc(scores_pos, scores_neg) -> dict:
    """AUC via the Mann-Whitney U identity, with ties counted as half.

    Exact rather than trapezoidal: the per-token score has heavy ties at 0 (every in-band token),
    and a trapezoidal AUC over a coarse threshold grid would misreport precisely that regime.
    """
    import numpy as np
    p = np.asarray(scores_pos, dtype=np.float64).ravel()
    n = np.asarray(scores_neg, dtype=np.float64).ravel()
    assert p.size > 0 and n.size > 0, "roc_auc needs both classes"
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    # average ranks within tie groups
    s = allv[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    auc = (ranks[:p.size].sum() - p.size * (p.size + 1) / 2.0) / (p.size * n.size)
    return {"auc": float(auc), "n_pos": int(p.size), "n_neg": int(n.size),
            "tie_fraction": float(1.0 - np.unique(allv).size / allv.size)}


# ---------------------------------------------------------------------------
# W0-8 — delta-compression frontier.
# ---------------------------------------------------------------------------
def energy_captured(delta, U) -> float:
    """‖ΔW U‖²_F / ‖ΔW‖²_F — the fraction of a fine-tune's write energy a basis captures."""
    import numpy as np
    A = np.asarray(delta, dtype=np.float64)
    num = float(np.linalg.norm(A @ np.asarray(U, dtype=np.float64)) ** 2)
    den = float(np.linalg.norm(A) ** 2)
    assert den > 0, "energy_captured on a zero delta"
    return num / den


def rank_ladder(max_rank: int = 256, min_rank: int = 4):
    """The §4/W0-8 highway rank ladder 4..256, log-spaced by powers of two."""
    r, out = min_rank, []
    while r <= max_rank:
        out.append(r)
        r *= 2
    return out
