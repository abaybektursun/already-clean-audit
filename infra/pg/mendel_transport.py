"""MENDEL — the transport / co-location instrument (NEW, §13 ledger).

Serves MN-1 (the MDE), MN-2, MN-3 and MN-13. All four ask ONE question in two shapes:

  ADDRESS shape (MN-1)      — is the child's sink ADDRESS where the parent's sink address was,
                              beyond where the parent's activation TRAFFIC alone would put it?
  SUBSPACE shape (MN-2/13)  — does a hop's faculty-relevant WRITE land inside the sink subspace,
                              beyond what an energy-matched subspace of the same size would catch?

Both are differences against a STRUCTURED null, never against haar alone (§5 standing rule), and
both carry the haar band and a label-shuffle beside the treatment.

Why the traffic null is inside the statistic rather than beside it. The elegant-unification alarm
in §8 is about exactly this arm: "the sink organizes the lineage" is true-sounding and would be
produced by a completely inert sink, because a sink address and an activation-traffic address are
both concentrated on the same few high-energy coordinates. Subtracting the traffic-predicted
overlap per cell — the same cell, the same k, the same scorer — is what makes the remainder
attributable to transport rather than to shared energy. H_inert is the null this instrument is
built to be able to report.

Pure numpy + the reused DARWIN/MERCATOR selectors. No torch, no S3, no model: every function here
is exercised by the weight-free dry-run gate (tests/test_mendel_transport.py).

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from infra.pg import darwin_w0 as W0
from infra.pg import humboldt_core as H
from infra.pg import mercator_predict as PR


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
def bank_address(U, k: int):
    """The measured address of a bank: top-k input coordinates by leverage. REUSED verbatim from
    the M-8 scorer so the prediction and the ground truth are read by ONE implementation (§13)."""
    return PR.measured_address(U, k)


def score_address(score, k: int):
    """The address a per-coordinate score predicts: top-k by score. REUSED from M-8."""
    return PR.predict_address(score, k)


def haar_address(n_total: int, k: int, seed: int):
    """The chance floor: a uniformly random k-subset of the n_total coordinates.

    Drawn as a permutation prefix rather than by thresholding a random score, so the draw is
    exactly uniform over k-subsets — `random_baseline` + top-k would also be uniform, but only
    because its scores are continuous, and depending on that is a footgun if the scorer ever
    changes to something with ties.
    """
    import numpy as np
    assert 0 < k <= n_total, f"k={k} outside 1..{n_total}"
    return np.random.default_rng(seed).permutation(n_total)[:k].astype(np.int64).tolist()


# ---------------------------------------------------------------------------
# The hop side: where a SINGLE hop declined to write
# ---------------------------------------------------------------------------
def column_write_energy(dW):
    """Per-input-coordinate write energy of one hop: e_j = ‖ΔW[:, j]‖² / ‖ΔW‖²_F, summing to 1.

    THIS, not a bottom-k eigenbasis, is the single-hop analogue of the house `core32`
    construction, and the reason is arithmetic rather than taste.

    `core32` is the bottom-32 eigenspace of a CONSENSUS Gram Σ_i ΔW_iᵀΔW_i over a whole fine-tune
    cloud. Applied to ONE hop the same recipe breaks on this architecture's MQA matrices:
    k_proj / v_proj have ΔW of shape 256×2048, so ΔWᵀΔW has rank ≤ 256 and a 1792-dimensional NULL
    SPACE, and the "bottom 32 eigenvectors" are an essentially arbitrary basis drawn from it. On a
    ΔW with a planted quiet coordinate set the eigenbasis readout recovers that set at Jaccard
    ≈ 0.03–0.10 (chance is 0.008) while column energy recovers it at 1.000 — measured, in
    tests/test_mendel_transport.py, not asserted.

    Where the eigenbasis IS well posed — q/o_proj (2048×2048) and gate/up_proj (16384×2048) — the
    two readouts agree exactly (both 1.000 on the same construction). So column energy is not a
    weaker substitute: it is correct everywhere the eigenbasis is correct, and correct where the
    eigenbasis is not. It is also a plain sum over output rows, which is what makes the split-half
    stability gate (A7) meaningful — a row-subsampled eigenbasis of a square ΔW is rank-deficient
    (1024 < 2048) and could not check itself.

    The PARENT side keeps the banked `core32` leverage unchanged — the parent has a real cloud, so
    its consensus bank is well posed, and keeping it is what makes MN-1's numbers commensurable
    with M-8's 0.4160 reference.
    """
    import numpy as np
    D = np.asarray(dW, dtype=np.float64)
    assert D.ndim == 2, f"dW must be 2-D (out, in), got {D.shape}"
    tot = float((D ** 2).sum())
    assert tot > 0, "zero ΔW — this hop wrote nothing, its sink address is undefined"
    return (D ** 2).sum(axis=0) / tot


def hop_sink_address(dW, k: int):
    """The hop's sink address: the k input coordinates with the SMALLEST write energy.

    Note the direction. The parent's address is the TOP-k by `core32` leverage (coordinates most
    aligned with the low-motion consensus subspace); the hop's is the BOTTOM-k by write energy
    (coordinates this hop moved least). Both name "the sink", from the two sides; getting the sign
    wrong here would invert the whole arm, so the two selectors are written next to each other.
    """
    import numpy as np
    e = np.asarray(column_write_energy(dW), dtype=np.float64)
    return W0.top_k_coords(-e, k)


def split_half_stability(dW, k: int) -> dict:
    """The instrument's self-consistency on this cell (freeze amendment A7).

    ΔW's output rows are split by index parity; each half yields its own write-energy profile and
    its own address. Their Jaccard says whether this cell's address is a property of the hop or of
    the particular rows that happened to be measured. Parity rather than a random split so the
    halves are deterministic and interleaved — a contiguous split would confound the halves with
    head index on the attention matrices.
    """
    import numpy as np
    D = np.asarray(dW, dtype=np.float64)
    a, b = D[0::2, :], D[1::2, :]
    assert a.shape[0] >= 1 and b.shape[0] >= 1, f"cannot split {D.shape[0]} rows in half"
    A, B = hop_sink_address(a, k), hop_sink_address(b, k)
    return {"self_jaccard": float(W0.jaccard(A, B)), "n_rows": int(D.shape[0]),
            "k": int(k), "n_total": int(D.shape[1])}


# ---------------------------------------------------------------------------
# MN-1 — the address shape
# ---------------------------------------------------------------------------
def transport_cell(*, U_parent, dW_hop, traffic_score, k: int, seed: int, n_haar: int = 8) -> dict:
    """One (type, layer) cell of one hop: observed transport vs the traffic-predicted null.

      observed = Jaccard( top-k leverage of the PARENT's core32 , bottom-k write energy of ΔW )
      traffic  = Jaccard( top-k of the PARENT's activation RMS   , bottom-k write energy of ΔW )
      delta    = observed - traffic

    `traffic_score` is the PARENT's per-coordinate activation RMS on the frozen batch — the same
    incumbent M-8's REFUTER is written against, and the one that BEAT the mechanism predictor
    (0.4160 vs 0.3068 mean Jaccard against the banked core32). It predicts where this hop declined
    to write using nothing but where the parent's activations are large, so `delta` is the part of
    the hop's sink that the parent's SINK explains and the parent's TRAFFIC does not.

    The haar band is n_haar independent random addresses scored the same way: the chance floor,
    reported beside the structured null and never in place of it.
    """
    import numpy as np
    Up = np.asarray(U_parent, dtype=np.float64)
    D = np.asarray(dW_hop, dtype=np.float64)
    tr = np.asarray(traffic_score, dtype=np.float64).ravel()
    assert Up.ndim == 2 and D.ndim == 2, f"expected 2-D bank / ΔW, got {Up.shape} / {D.shape}"
    assert Up.shape[0] == D.shape[1] == tr.size, (
        f"parent bank {Up.shape}, ΔW {D.shape} and traffic {tr.shape} disagree about the "
        f"coordinate axis — all three must be indexed by the SAME input dimension")
    n = int(D.shape[1])
    assert 0 < k <= n, f"k={k} outside 1..{n}"

    hop = hop_sink_address(D, k)
    obs = W0.overlap_stats(bank_address(Up, k), hop, n)
    tra = W0.overlap_stats(score_address(tr, k), hop, n)
    haar = [W0.overlap_stats(haar_address(n, k, seed * 1_000_003 + i), hop, n)["jaccard"]
            for i in range(n_haar)]
    return {"observed": obs, "traffic": tra, "hop_address": hop,
            "haar_jaccard": [float(x) for x in haar],
            "haar_mean": float(np.mean(haar)),
            "haar_lo": float(np.min(haar)), "haar_hi": float(np.max(haar)),
            "delta": float(obs["jaccard"] - tra["jaccard"]),
            "stability": split_half_stability(D, k),
            "k": int(k), "n_total": n, "seed": int(seed), "n_haar": int(n_haar)}


def shuffle_transport(cells: list, seed: int) -> list:
    """§5's shuffle test for MN-1: re-pair each cell's PARENT address with another cell's CHILD.

    The permutation is a derangement over cells, so no cell keeps its own child; a plain shuffle
    leaves fixed points with probability ~1-1/e over the whole draw and those fixed points are
    real transport leaking into the null. `cells` is a list of {"parent": [...], "child": [...],
    "n_total": int} dicts; the returned Jaccards are the shuffled-label distribution the treatment
    is read against.
    """
    import numpy as np
    m = len(cells)
    assert m >= 2, "the shuffle test needs at least two cells to re-pair"
    n_totals = {c["n_total"] for c in cells}
    assert len(n_totals) == 1, (
        f"shuffle_transport over cells with different coordinate-space sizes {sorted(n_totals)} — "
        f"a Jaccard between addresses in different spaces is not a comparable statistic")
    rng = np.random.default_rng(seed)
    perm = np.arange(m)
    while True:
        rng.shuffle(perm)
        if not (perm == np.arange(m)).any():
            break
    return [W0.jaccard(cells[i]["parent"], cells[int(perm[i])]["child"]) for i in range(m)]


# ---------------------------------------------------------------------------
# MN-2 / MN-13 — the subspace shape
# ---------------------------------------------------------------------------
def colocation_cell(dW, *, U_sink, U_energy, U_haar_list) -> dict:
    """One cell of the co-location question: where did this hop's write energy land?

    Returns the write fraction into the sink subspace, into the energy-matched control subspace,
    and the haar band — all through ENDOW's `write_fraction` (‖ΔW·U‖_F/‖ΔW‖_F), REUSED verbatim so
    MENDEL's fractions are on the same scale as INHERIT's R6 and ENDOW's J3 numbers (§13).

    The energy-matched subspace is the one that matters. A sink subspace is, by construction, the
    low-motion end of the consensus Gram; a random subspace of the same dimension catches far less
    activation energy, so 'the write avoids the sink more than it avoids a random subspace' is a
    statement about energy, not about the sink. `U_energy` removes that reading.
    """
    import numpy as np
    from infra.pg.endow_writefrac import write_fraction
    D = np.asarray(dW, dtype=np.float64)
    assert D.ndim == 2, f"dW must be 2-D (out, in), got {D.shape}"
    for name, U in (("U_sink", U_sink), ("U_energy", U_energy)):
        assert np.asarray(U).shape[0] == D.shape[1], (
            f"{name} first axis {np.asarray(U).shape[0]} != dW input dim {D.shape[1]}")
    haar = [float(write_fraction(D, U)) for U in U_haar_list]
    assert haar, "colocation_cell needs a non-empty haar band"
    return {"sink": float(write_fraction(D, U_sink)),
            "energy": float(write_fraction(D, U_energy)),
            "haar": haar, "haar_mean": float(np.mean(haar)),
            "haar_lo": float(np.min(haar)), "haar_hi": float(np.max(haar)),
            "delta_vs_energy": float(write_fraction(D, U_sink) - write_fraction(D, U_energy)),
            "k": int(np.asarray(U_sink).shape[1]), "n_total": int(D.shape[1])}


def planted_colocation(n: int, k: int, out_dim: int, seed: int, *, strength: float = 1.0):
    """§5's POSITIVE CONTROL for the co-location instrument, as data rather than as a promise.

    Builds (dW, U_sink) where a known `strength` fraction of dW's energy is planted INSIDE
    span(U_sink). The instrument must recover that fraction; if it cannot, the arm is
    VOID-INSTRUMENT and says so in the verdict expression (the B1/X2 lesson, §5).

    `strength` is the fraction of ENERGY (squared norm) planted in the span, so the returned
    write_fraction — which is a ratio of norms, not of squared norms — should recover sqrt(strength).
    Stating the two scales here is deliberate: conflating them is how a positive control silently
    passes at the wrong level.
    """
    import numpy as np
    assert 0 < k < n, f"k={k} outside 1..{n - 1}"
    assert 0.0 <= strength <= 1.0, f"strength {strength} outside [0,1]"
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    U = np.ascontiguousarray(Q[:, :k])                 # the planted span
    C = np.ascontiguousarray(Q[:, k:])                 # its complement
    A = rng.standard_normal((out_dim, k))
    B = rng.standard_normal((out_dim, n - k))
    A *= np.sqrt(strength) / np.linalg.norm(A)
    B *= np.sqrt(1.0 - strength) / np.linalg.norm(B)
    return (A @ U.T + B @ C.T), U


# ---------------------------------------------------------------------------
# Verdict expressions (§6 frozen rules, one function each)
# ---------------------------------------------------------------------------
def mn1_verdict(per_hop: dict, *, k_hops: int = 2) -> dict:
    """MN-1's frozen rule, verbatim from §2:

      INHERITED-BEYOND-TRAFFIC iff observed exceeds traffic-predicted with CI clear of 0 at
      >= 2 independent hops; TRAFFIC-ONLY iff no hop clears; else AMBIGUOUS.

    `per_hop` maps hop_id -> `mendel_core.paired_delta_ci` result on (observed, traffic).
    """
    from infra.pg.mendel_core import clears_at_k_hops
    c = clears_at_k_hops(per_hop, k=k_hops)
    if c["clears"]:
        v = "INHERITED-BEYOND-TRAFFIC"
    elif c["n_clearing"] == 0:
        v = "TRAFFIC-ONLY"
    else:
        v = "AMBIGUOUS"
    return {"verdict": v, **c}


def mn2_verdict(per_faculty_hop: dict, *, k_faculties: int = 2, k_hops: int = 2) -> dict:
    """MN-2/MN-13's frozen rule, verbatim from §2:

      SINK-COUPLED iff the sink fraction exceeds the energy-matched band on >= 2 faculties at
      >= 2 hops; ENERGY-ONLY iff never; else AMBIGUOUS.

    `per_faculty_hop` maps (faculty, hop) -> `mendel_core.exceeds_band` result. The rule needs BOTH
    counts, so a single faculty firing at four hops is AMBIGUOUS rather than SINK-COUPLED — which
    is the point of writing the rule as two counts instead of one.
    """
    above = [key for key, r in per_faculty_hop.items() if r["above_band"]]
    faculties = sorted({str(f) for f, _ in above})
    hops = sorted({str(h) for _, h in above})
    if len(faculties) >= k_faculties and len(hops) >= k_hops:
        v = "SINK-COUPLED"
    elif not above:
        v = "ENERGY-ONLY"
    else:
        v = "AMBIGUOUS"
    return {"verdict": v, "n_above": len(above),
            "faculties_above": faculties, "hops_above": hops,
            "k_faculties_required": int(k_faculties), "k_hops_required": int(k_hops),
            "n_cells": len(per_faculty_hop)}


def instrument_is_void(planted_recovered: float, planted_expected: float, tol: float = 0.10) -> dict:
    """§5's VOID-INSTRUMENT test, consumed inside the verdict expression rather than beside it.

    The planted positive control must be recovered to `tol` in write-fraction units. A failure here
    does not invalidate the run — it CHANGES THE VERDICT to VOID-INSTRUMENT, which is a reportable
    outcome. Returning the evidence rather than asserting is deliberate: the caller must be able to
    bank the failed control alongside the token.
    """
    err = abs(float(planted_recovered) - float(planted_expected))
    return {"recovered": float(planted_recovered), "expected": float(planted_expected),
            "abs_error": err, "tol": float(tol), "void": bool(err > tol)}


# Re-exported so an arm needs one import for the whole instrument.
leverage = H.leverage
jaccard = W0.jaccard
overlap_stats = W0.overlap_stats
