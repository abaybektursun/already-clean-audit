"""M-12 — the sink x induction-heads bridge (queue item 12).

Hypothesis-to-kill: induction heads are principal READERS of the sink — the route on our road bed.
The instrument has three parts, and only the first two are new; part (c) reuses conserved-core's
probe corpora verbatim, because those probes ARE the waived pos-vs-induction discriminator this arm
is chartered to resolve.

(a) IDENTIFY. The standard repeated-random-token method: build a sequence of random tokens and
    concatenate it with itself, then score each head by how much it attends from a token in the
    second copy back to the token that FOLLOWED that token's previous occurrence. That target
    offset is exactly −L+1 positions. The threshold is frozen before the measurement and the arm
    reports the whole SWEEP as well (§7 threats: 'induction-head identification threshold games').

(b) RESTRICT. For each head, how much of its READ lives in the sink versus in each twin. A head's
    read of the residual stream is the row space of its own q (and k) slice, so the statistic is
    the fraction of that slice's energy captured by a bank:

        e(head, U) = ‖W_head U‖²_F / ‖W_head‖²_F        with E[e] = k/n for a Haar bank

    The REFUTER is written against this: 'induction heads' read overlap with sink within twin
    band'. A twin band, not zero — the null is the twin battery, not "no overlap".

(c) DISCRIMINATE. conserved_core_probes.copy_corpus (induction-pure) and .position_corpus
    (position-pure), each with its shuffle-label positive control, under core ablation vs the twin
    battery.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from infra.pg import conserved_core_probes as CP   # REUSED verbatim (§11 reuse ledger)

# The frozen identification threshold. The standard method's convention: a head counts as an
# induction head when its prefix-matching score exceeds this. Frozen HERE, before any measurement,
# and the arm additionally reports the score distribution and a sweep so the choice is auditable.
INDUCTION_THRESHOLD = 0.20
THRESHOLD_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40)


def repeated_random_tokens(vocab_size: int, seq_len: int, n_seq: int, seed: int,
                           *, forbid=()):
    """`n_seq` sequences of the form [R, R] where R is `seq_len` uniform random token ids.

    Sampling is UNIFORM over the vocabulary rather than over natural text: the point of the method
    is that the second copy is unpredictable from anything except the first copy, so any
    frequency-weighted sampler would let ordinary bigram statistics score as induction.

    `forbid` removes special ids (BOS/EOS/PAD) which would otherwise appear mid-sequence and let a
    head score by attending to a delimiter.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    bad = set(int(x) for x in forbid)
    pool = np.array([i for i in range(vocab_size) if i not in bad], dtype=np.int64)
    assert pool.size > seq_len, f"vocabulary too small after forbidding {sorted(bad)}"
    R = rng.choice(pool, size=(n_seq, seq_len), replace=True)
    return np.concatenate([R, R], axis=1)


def prefix_matching_score(attn, seq_len: int, *, offset: int = 0):
    """Per-head prefix-matching score on one [R, R] sequence.

    `attn` is (n_heads, T, T) with T = offset + 2*seq_len; `offset` is the number of leading tokens
    that are not part of the repeat (a BOS, say). For a query at position i in the SECOND copy, the
    induction target is position i − seq_len + 1: the token that followed the previous occurrence.

    The first query of the second copy is skipped: its target would be the position before the
    sequence starts, and including it would put a structurally-zero term into every head's mean.
    """
    import numpy as np
    a = np.asarray(attn, dtype=np.float64)
    assert a.ndim == 3 and a.shape[1] == a.shape[2], f"attn shape {a.shape}, expected (H, T, T)"
    T = a.shape[1]
    # The repeat block must FIT in the sequence, not BE the sequence: PaliGemma's processor wraps
    # text as [image tokens .. <bos> .. text .. trailing "\n"], so there is at least one token
    # after the block. Requiring equality asserted that the wrapper does not exist.
    assert T >= offset + 2 * seq_len, (
        f"T={T} is too short for a {seq_len}+{seq_len} repeat block at offset {offset}")
    qs = np.arange(offset + seq_len + 1, offset + 2 * seq_len)
    ts = qs - seq_len + 1
    assert (ts >= offset).all(), "induction target fell outside the first copy"
    return a[:, qs, ts].mean(axis=1)


def induction_heads(scores, threshold: float = INDUCTION_THRESHOLD) -> dict:
    """Head list at the frozen threshold, plus the full sweep (§7 threat mitigation).

    `scores` is (n_layers, n_heads) or a flat per-head vector. Returns the selected (layer, head)
    pairs, the count at every swept threshold, and the score distribution — so a reader can see
    whether the result depends on the cut.
    """
    import numpy as np
    s = np.asarray(scores, dtype=np.float64)
    flat = s.reshape(-1)
    idx = np.argwhere(s >= threshold)
    return {"threshold": float(threshold),
            "heads": [tuple(int(x) for x in row) for row in idx],
            "n_heads_selected": int(idx.shape[0]), "n_heads_total": int(flat.size),
            "sweep": {float(t): int((s >= t).sum()) for t in THRESHOLD_SWEEP},
            "score_mean": float(flat.mean()), "score_max": float(flat.max()),
            "score_p95": float(np.percentile(flat, 95))}


def head_slice(W, head: int, head_dim: int):
    """The rows of a projection matrix belonging to one head.

    W is (n_heads*head_dim, d_model) for q_proj/k_proj/v_proj. Under MQA/GQA the k/v matrices have
    FEWER heads than q, so the caller must pass the head index in the matrix's OWN head numbering;
    this function refuses an index the matrix cannot have rather than silently wrapping.
    """
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    n_heads = A.shape[0] // head_dim
    assert A.shape[0] % head_dim == 0, f"rows {A.shape[0]} not divisible by head_dim {head_dim}"
    assert 0 <= head < n_heads, f"head {head} out of range for a matrix with {n_heads} heads"
    return A[head * head_dim:(head + 1) * head_dim, :]


def head_read_energy(W_head, U) -> float:
    """e = ‖W_head U‖²_F / ‖W_head‖²_F — the fraction of a head's read energy inside subspace U.

    For a Haar k-subspace of ℝⁿ this has expectation k/n, which is the band the REFUTER is read
    against. Reuses HUMBOLDT's energy convention so the number is comparable to every other
    bank-energy number in the program.
    """
    from infra.pg import humboldt_core as H
    return H.bank_energy_fraction(W_head, U)


def head_battery(W, U_by_bank: dict, *, head: int, head_dim: int) -> dict:
    """One head x the whole control battery (§3): the treatment and every twin in one dict."""
    A = head_slice(W, head, head_dim)
    return {bid: head_read_energy(A, U) for bid, U in U_by_bank.items()}


# ---------------------------------------------------------------------------
# (b) two-clock overlay
# ---------------------------------------------------------------------------
def onset_step(steps, values, *, frac: float = 0.5) -> dict:
    """The step at which a monotone-ish curve first reaches `frac` of its own total rise.

    Used for BOTH clocks — the induction bump and the crystallization curve — so the two are read
    by ONE rule and the overlay is not an artifact of two different definitions of "when".

    Reports the interpolated crossing as well as the first observed step past it, because the
    checkpoint grid is coarse and the difference between the two is the honest resolution of the
    comparison.
    """
    import numpy as np
    s = np.asarray(steps, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    assert s.shape == v.shape and s.size >= 2, f"need >=2 paired points, got {s.shape}"
    assert (np.diff(s) > 0).all(), "steps must be strictly increasing"
    lo, hi = float(v[0]), float(v.max())
    assert hi > lo, "curve has no rise — onset is undefined and must not be reported as step 0"
    target = lo + frac * (hi - lo)
    j = int(np.argmax(v >= target))
    assert v[j] >= target
    if j == 0:
        crossing = float(s[0])
    else:
        w = (target - v[j - 1]) / (v[j] - v[j - 1])
        crossing = float(s[j - 1] + w * (s[j] - s[j - 1]))
    return {"onset_step": crossing, "first_step_at_or_past": float(s[j]),
            "target_value": float(target), "frac": float(frac),
            "v_start": lo, "v_max": hi, "grid_resolution": float(s[j] - s[max(j - 1, 0)])}


def two_clock_overlay(induction, crystallization) -> dict:
    """The two clocks, read by one rule, plus their separation in steps and in grid units.

    `induction` and `crystallization` are each (steps, values). The separation is reported with the
    coarser of the two grid resolutions attached, so "the clocks coincide" cannot be claimed at a
    resolution the checkpoint grid does not have.
    """
    a = onset_step(*induction)
    b = onset_step(*crystallization)
    res = max(a["grid_resolution"], b["grid_resolution"])
    d = b["onset_step"] - a["onset_step"]
    return {"induction": a, "crystallization": b, "separation_steps": float(d),
            "grid_resolution": float(res),
            "separation_in_grid_units": float(d / res) if res > 0 else float("inf")}


# ---------------------------------------------------------------------------
# (c) the pos-vs-induction discriminator — conserved-core's probes, unforked.
# ---------------------------------------------------------------------------
def discriminator_corpora(n_items: int, *, seed_copy: int = 20260619, seed_pos: int = 20260620):
    """The two synthetic corpora, each induction-pure or position-pure BY CONSTRUCTION.

    copy_corpus is repeat-3 copy with the repeated block placed so no positional shortcut answers
    it; position_corpus is explicit-position retrieval, which an induction head cannot shortcut.
    Both come from conserved_core_probes unchanged — this arm is chartered to RESOLVE that
    program's waived discriminator, so forking its corpora would answer a different question.
    """
    return {"induction": CP.copy_corpus(n_items, seed=seed_copy),
            "position": CP.position_corpus(n_items, seed=seed_pos)}


def positive_control(labels, seed: int):
    """The construct-validity control that CAN fail: shuffle the labels and re-score.

    A probe whose score survives label shuffling is reading something other than the labels, and
    the arm must report that instead of the contrast. Reuses conserved_core_probes.shuffle_labels.
    """
    return CP.shuffle_labels(labels, seed)
