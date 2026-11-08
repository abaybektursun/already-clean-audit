"""DARWIN W1-N / W1-N4 — quantization instruments (core-exact quantization, CEQ).

Source: post-freeze ADDENDUM `2026-08-12_darwin_addendum_ceq.md`. The frozen spec is NOT edited;
every row this module feeds carries `source = ADDENDUM` via the arm registry.

THE BINDING CLAIM REGISTER (from the addendum's dry-run, `09_ceq_dryruns/RESULTS.md`, re-run in
this worktree 8/8 PASS). These are constraints on what the arm may RECORD, not advice:

  MAY record   per-matrix core action is EXACT under CEQ routing (T1/T3)
  MAY record   with protected writer rows, the core lane after ONE layer is exact (T4a)
  MUST NOT     end-to-end core-lane purity. It is first-order in bulk noise (T4b measured only
               1.48x across two layers). No statistic in this module computes it and no row may
               assert it.
  MUST record  vault-reuse error as PREDICTED (leak law err = ‖ΔW·U‖) vs REALIZED, per fine-tune.

Design note on the routing. The addendum writes y = A(Uᵀx) + Q(W·U⊥)(U⊥ᵀx). Materialising U⊥
(n x (n-k)) per matrix would cost as much memory as W itself, so the equivalent ambient form is
used for the per-matrix arm:

    y = A (Uᵀx) + Q(W_bulk) (x - U (Uᵀx)),      W_bulk = W - A Uᵀ,   A = W U

The trailing projection is what makes this equivalent rather than merely similar: W_bulk U = 0
holds exactly, but Q(W_bulk) U does NOT, so feeding the raw x would let quantised bulk re-inject
core-direction content and destroy both the exactness claim (T1) and the T7 property that the
bulk never sees the spiky core activations. The rotated-block arm (e) materialises the rotation
once globally instead, and T3 proves the two agree bit-level.

FAIL LOUD. Pure numpy; heavy imports inside functions.
"""
from __future__ import annotations

import math

# Frozen quantizer settings for the arm (bulk quantizer identical across arms, per the addendum).
DEFAULT_BITS = 4
GROUP_SIZE = 128          # per-output-channel groups; the standard INT4 weight-quant recipe


# ---------------------------------------------------------------------------
# Quantizers. Deliberately simple and exactly specified: the arm compares ROUTING schemes at a
# matched bit budget, so the quantizer must be identical across arms and auditable.
# ---------------------------------------------------------------------------
def quantize_uniform(W, bits: int = DEFAULT_BITS, group_size: int = GROUP_SIZE):
    """Symmetric per-row, per-group absmax integer quantization; returns the dequantized matrix.

    Symmetric (not asymmetric) so a zero row stays zero and W_bulk's exact zeros on the core
    columns are preserved by the quantizer rather than smeared by a zero-point.
    """
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    m, n = A.shape
    qmax = 2 ** (bits - 1) - 1
    out = np.empty_like(A)
    for s in range(0, n, group_size):
        e = min(s + group_size, n)
        blk = A[:, s:e]
        scale = np.abs(blk).max(axis=1, keepdims=True) / qmax
        scale[scale == 0] = 1.0
        out[:, s:e] = np.round(blk / scale).clip(-qmax, qmax) * scale
    return out


def quantize_with_fp16_columns(W, keep_cols, bits: int = DEFAULT_BITS,
                               group_size: int = GROUP_SIZE):
    """Arm b, the honest incumbent: uniform INT4 everywhere EXCEPT a kept column list held at
    fp16. `keep_cols` is the outlier-column list (or the core-column list for arm c)."""
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    keep = np.asarray(sorted(set(int(c) for c in keep_cols)), dtype=np.int64)
    assert keep.size == 0 or (keep.min() >= 0 and keep.max() < A.shape[1]), "column out of range"
    Q = quantize_uniform(A, bits, group_size)
    Q[:, keep] = np.asarray(A[:, keep], dtype=np.float16).astype(np.float64)
    return Q


def fp16_parameter_count(*, m: int, n: int, k: int, scheme: str) -> int:
    """Number of values held at fp16, for the matched-bit-budget rule (§3 / addendum).

    arms b, c: a column list of width k              -> m*k
    arms d, e: the vault A = W U (m x k) AND U (n x k) -> m*k + n*k
    The vault costs an EXTRA n*k because the basis must be stored too. The addendum's rule is
    that arms b-e get identical total fp16 parameter counts, so the honest comparison gives the
    column-list arms a wider list, not the same k. `matched_k_for_vault` computes it.
    """
    if scheme in ("column_list",):
        return m * k
    if scheme in ("vault",):
        return m * k + n * k
    raise AssertionError(f"unknown scheme {scheme}")


def matched_k_for_vault(*, m: int, n: int, k_vault: int) -> int:
    """The column-list width that spends the SAME fp16 budget as a rank-k vault.

    Without this the comparison silently favours CEQ: a k=32 vault stores m*32 + n*32 values
    while a k=32 column list stores only m*32. Arm b must be given the larger list.
    """
    budget = fp16_parameter_count(m=m, n=n, k=k_vault, scheme="vault")
    return int(budget // m)


# ---------------------------------------------------------------------------
# CEQ.
# ---------------------------------------------------------------------------
def vault_split(W, U):
    """(A, W_bulk) with A = W U and W_bulk = W - A Uᵀ. W_bulk U = 0 holds to float precision."""
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    Ub = np.asarray(U, dtype=np.float64)
    assert A.shape[1] == Ub.shape[0], f"shape mismatch W {A.shape} vs U {Ub.shape}"
    Amat = A @ Ub
    bulk = A - Amat @ Ub.T
    resid = float(np.abs(bulk @ Ub).max())
    assert resid < 1e-9, f"bulk did not annihilate the core: max|W_bulk U| = {resid:.3e}"
    return Amat, bulk


def ceq_apply(A_vault, Q_bulk, U, x):
    """y = A(Uᵀx) + Q_bulk (x - U(Uᵀx)). `x` is (..., n); returns (..., m).

    The trailing projection is load-bearing — see the module docstring.
    """
    import numpy as np
    Xa = np.asarray(x, dtype=np.float64)
    Ub = np.asarray(U, dtype=np.float64)
    c = Xa @ Ub                                   # (..., k) core coordinates
    x_perp = Xa - c @ Ub.T
    return c @ np.asarray(A_vault, dtype=np.float64).T + x_perp @ np.asarray(Q_bulk).T


def core_action_error(W, U, A_vault, Q_bulk) -> float:
    """‖Ŵ U − W U‖_F / ‖W U‖_F, where Ŵ is CEQ's effective operator.

    This is the statistic the arm MAY assert exactness on (T1). Applying CEQ to the columns of U
    itself: the perp branch contributes Q_bulk(U − U) = 0, so the result is A_vault exactly.
    """
    import numpy as np
    Wm = np.asarray(W, dtype=np.float64)
    Ub = np.asarray(U, dtype=np.float64)
    got = ceq_apply(A_vault, Q_bulk, Ub, Ub.T).T        # apply to each column of U
    want = Wm @ Ub
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# ---------------------------------------------------------------------------
# The leak law (T6) — the family-vault claim's predictor.
# ---------------------------------------------------------------------------
def leak_norm(delta_W, U) -> float:
    """‖ΔW · U‖_F — how much a fine-tune's update touches the vaulted subspace."""
    import numpy as np
    return float(np.linalg.norm(np.asarray(delta_W, dtype=np.float64)
                                @ np.asarray(U, dtype=np.float64)))


def vault_reuse_error(W_ft, U, A_base) -> float:
    """‖W_ft U − A_base‖_F: the realized core-action error when a fine-tune reuses the BASE
    vault unchanged. T6 proves this equals the leak norm exactly, so the two are recorded side by
    side and their ratio is the arm's prediction check."""
    import numpy as np
    return float(np.linalg.norm(np.asarray(W_ft, dtype=np.float64)
                                @ np.asarray(U, dtype=np.float64)
                                - np.asarray(A_base, dtype=np.float64)))


def leak_law_check(W_base, W_ft, U) -> dict:
    """Predicted (leak norm) vs realized vault-reuse error, and their ratio.

    The addendum's graduation clause turns on this ratio being within 2x, so it is computed here
    once and recorded, never re-derived in prose.
    """
    import numpy as np
    A_base = np.asarray(W_base, dtype=np.float64) @ np.asarray(U, dtype=np.float64)
    dW = np.asarray(W_ft, dtype=np.float64) - np.asarray(W_base, dtype=np.float64)
    pred = leak_norm(dW, U)
    real = vault_reuse_error(W_ft, U, A_base)
    return {"predicted_leak_norm": pred, "realized_vault_error": real,
            "ratio": (real / pred) if pred > 0 else float("inf"),
            "within_2x": bool(pred > 0 and 0.5 <= real / pred <= 2.0),
            "abs_deviation": abs(real - pred)}


# ---------------------------------------------------------------------------
# Rotated-block form (arm e) — licensed only when W0-2's frozen threshold says so.
# ---------------------------------------------------------------------------
def union_basis(bases, tol: float = 1e-8):
    """An orthonormal basis of the span of several per-layer cores (T5's union construction).

    Rank is `shared + Σ private`, which for disjoint cores is Σ ranks — the worst case the arm
    must budget for, so the realized rank is recorded on every row.

    SVD, not QR. Un-pivoted QR is NOT rank-revealing: on a stack of overlapping orthonormal
    blocks the near-zero R diagonals land in the MIDDLE of the sequence, and every later column
    of Q is then built by projecting off the arbitrary unit vectors QR inserted at those
    positions. Keeping columns by |diag(R)| therefore returns a basis that does not contain the
    inputs — measured here at residual 0.51 for a case whose true residual is 0. The left
    singular vectors with non-negligible singular value are the column space, with no ordering
    assumption to get wrong.
    """
    import numpy as np
    M = np.concatenate([np.asarray(B, dtype=np.float64) for B in bases], axis=1)
    Uu, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(1.0, float(s.max()))
    out = np.ascontiguousarray(Uu[:, keep])
    resid = max((float(np.linalg.norm(np.asarray(B, dtype=np.float64)
                                      - out @ (out.T @ np.asarray(B, dtype=np.float64))))
                 for B in bases), default=0.0)
    assert resid < 1e-8, f"union basis does not contain its inputs: residual {resid:.3e}"
    return out


def rotated_block_quantize(W, U, bits: int = DEFAULT_BITS, group_size: int = GROUP_SIZE):
    """Arm e: rotate into [U | U⊥], keep the first k columns fp16, quantize the rest.

    Returns the effective dequantized operator in the ORIGINAL basis, so the caller can compare
    it against arms a-d without another change of coordinates. T3 proves this agrees with the
    projector-routed form; `test_darwin_ceq` re-checks that agreement rather than assuming it.
    """
    import numpy as np
    Wm = np.asarray(W, dtype=np.float64)
    Ub = np.asarray(U, dtype=np.float64)
    n, k = Ub.shape
    # Complete U to a full orthonormal basis of R^n.
    Z = np.eye(n) - Ub @ Ub.T
    Qz, _ = np.linalg.qr(Z)
    Uperp = np.ascontiguousarray(Qz[:, :n - k])
    R = np.concatenate([Ub, Uperp], axis=1)                   # n x n, orthogonal
    Wr = Wm @ R                                               # rotated columns
    Wr[:, k:] = quantize_uniform(Wr[:, k:], bits, group_size)  # bulk only
    return Wr @ R.T


# ---------------------------------------------------------------------------
# Activation / KV variant (the T7 mechanism).
# ---------------------------------------------------------------------------
def quantize_activations(X, bits: int = 8, keep_dims=()):
    """Per-token absmax activation quantization, optionally holding `keep_dims` at fp16.

    T7's claim is that a routed bulk never sees the spiky core content, so the honest comparison
    is: quantize the AMBIENT activation vs quantize only its complement component.
    """
    import numpy as np
    A = np.asarray(X, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(A).max(axis=-1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    out = np.round(A / scale).clip(-qmax, qmax) * scale
    keep = np.asarray(sorted(set(int(d) for d in keep_dims)), dtype=np.int64)
    if keep.size:
        out[..., keep] = np.asarray(A[..., keep], dtype=np.float16).astype(np.float64)
    return out


def routed_activation_error(X, U, bits: int = 8) -> dict:
    """T7's readout: ambient-quantized activation error vs routed (core exact, bulk quantized).

    Returns both errors and their ratio. Recorded, never asserted as an end-to-end claim.
    """
    import numpy as np
    A = np.asarray(X, dtype=np.float64)
    Ub = np.asarray(U, dtype=np.float64)
    amb = quantize_activations(A, bits) - A
    c = A @ Ub
    perp = A - c @ Ub.T
    routed = (quantize_activations(perp, bits) - perp)        # core carried exactly
    ea, er = float(np.linalg.norm(amb)), float(np.linalg.norm(routed))
    return {"ambient_error": ea, "routed_error": er,
            "ratio": (ea / er) if er > 0 else float("inf"), "bits": bits}


# ---------------------------------------------------------------------------
# §5 / addendum graduation clause, evaluated mechanically.
# ---------------------------------------------------------------------------
def ceq_graduates(*, ppl_ceq: float, ppl_ceq_ci_lo: float, ppl_ceq_ci_hi: float,
                  ppl_incumbent_b: float, leak_within_2x: bool) -> dict:
    """Addendum's frozen clause: CEQ graduates iff arm d or e beats arm b on ppl at matched bits
    by more than the bootstrap CI half-width, OR ties within CI while arm f's realized
    degradation matches the leak-law prediction within 2x.

    Perplexity is lower-is-better; that is stated here rather than assumed by the caller.
    """
    hw = (ppl_ceq_ci_hi - ppl_ceq_ci_lo) / 2.0
    assert hw >= 0, f"inverted CI [{ppl_ceq_ci_lo}, {ppl_ceq_ci_hi}]"
    delta = ppl_incumbent_b - ppl_ceq                 # positive = CEQ better (lower ppl)
    beats = bool(delta > hw)
    ties = bool(abs(delta) <= hw)
    return {"beats_incumbent_b": beats, "ties_within_ci": ties,
            "leak_law_within_2x": bool(leak_within_2x),
            "graduates": bool(beats or (ties and leak_within_2x)),
            "path": ("beats_b" if beats else
                     ("tie_plus_leak_law" if (ties and leak_within_2x) else "neither")),
            "signed_delta_ppl": delta, "ci_halfwidth": hw,
            "note": "lower ppl is better; delta = incumbent_b - ceq"}


def storage_bits(*, m: int, n: int, k: int, scheme: str, bits: int = DEFAULT_BITS,
                 n_layers_sharing_rotation: int = 1) -> dict:
    """Exact storage accounting per matrix, in bits, plus bits-per-original-weight.

    The arms do NOT store the same things, and glossing that is how a quantization comparison
    flatters itself. Enumerated explicitly:

      uniform        every weight at `bits`.
      column_list    k columns at fp16 (m*k values), the remaining m*(n-k) at `bits`.
                     Arms b and c.
      vault          A = W U at fp16 (m*k) AND the basis U at fp16 (n*k), PLUS the full m*n bulk
                     at `bits` — W_bulk is dense, so routing does not shrink it. Arm d.
      rotated_block  the rotated matrix has k fp16 columns and m*(n-k) quantized, and the n*n
                     rotation is SHARED across every matrix that uses it, so its cost is divided
                     by `n_layers_sharing_rotation`. Arm e. This amortisation is the rotated
                     form's actual advantage over arm d and is recorded, not assumed.

    `bits_per_weight` is normalised by the m*n weights the matrix originally had, so arms are
    directly comparable and a scheme that stores MORE than fp16 shows up as > 16.
    """
    assert m > 0 and n > 0 and 0 <= k <= n, f"bad shape m={m} n={n} k={k}"
    assert n_layers_sharing_rotation >= 1
    if scheme == "uniform":
        fp16_vals, quant_vals, shared = 0, m * n, 0.0
    elif scheme == "column_list":
        fp16_vals, quant_vals, shared = m * k, m * (n - k), 0.0
    elif scheme == "vault":
        fp16_vals, quant_vals, shared = m * k + n * k, m * n, 0.0
    elif scheme == "rotated_block":
        fp16_vals, quant_vals = m * k, m * (n - k)
        shared = (n * n * 16) / n_layers_sharing_rotation
    else:
        raise AssertionError(f"unknown scheme {scheme}")
    total = fp16_vals * 16 + quant_vals * bits + shared
    return {"scheme": scheme, "fp16_values": fp16_vals, "quantized_values": quant_vals,
            "shared_rotation_bits": shared, "total_bits": total,
            "bits_per_weight": total / (m * n) if m * n else math.nan,
            "bits": bits, "k": k, "m": m, "n": n,
            "n_layers_sharing_rotation": n_layers_sharing_rotation}
