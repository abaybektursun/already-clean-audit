"""M-6 — the rate-of-change checkpoint format (queue item 6).

The format follows from two measured facts, and its whole value is that it prices itself against
them rather than against intuition:

  * a global rank-32 basis captures a large share of each layer's q_proj sink, and a UNION-k basis
    captures more at a few percent overhead — so the STABLE BLOCK can be shipped ONCE per family;
  * r=8 highway coefficients already carry most of a fine-tune's task recovery — so what a
    fine-tune actually has to ship is a small MOTION FILE, not a full delta.

Three parts:
    stable block   U_union, shipped once per family, priced by the leak law
    vault          the exact action of the base weights on the stable block (darwin_ceq.vault_split)
    motion file    per-fine-tune coefficients of ΔW in the highway basis, at rank r

Every quantization/vault primitive is DARWIN's `darwin_ceq`, reused unforked — this module is the
container format, the round-trip gates, and the byte accounting.

THE ROUND-TRIP GATES ARE THE POINT. A compression format that is never checked against the tensor
it claims to reconstruct is a story about bytes. `roundtrip` returns the reconstruction error and
`assert_roundtrip` refuses a tolerance that was chosen after seeing it.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from infra.pg import darwin_ceq as CEQ    # REUSED verbatim (§11 reuse ledger)


def union_stable_block(bases, tol: float = 1e-8):
    """The per-family stable block: an orthonormal basis of the UNION of the per-layer cores.

    Reuses darwin_ceq.union_basis so the block in the format is the same object the CEQ arms
    measured, not a re-derivation that might differ in rank or ordering.
    """
    U = CEQ.union_basis(bases, tol=tol)
    assert U.ndim == 2 and U.shape[1] >= 1, f"degenerate union block {U.shape}"
    return U


def encode(W_base, W_ft, U_block, r: int, *, bits: int = 4) -> dict:
    """One fine-tune, encoded.

    Returns the three parts plus the numbers a reader needs to price them. The BASE side (vault +
    quantized bulk) is family-level and identical across fine-tunes; it is returned here so a
    single-fine-tune round trip is self-contained, and `bytes_breakdown` charges it once.
    """
    import numpy as np
    Wb = np.asarray(W_base, dtype=np.float64)
    Wf = np.asarray(W_ft, dtype=np.float64)
    U = np.asarray(U_block, dtype=np.float64)
    assert Wb.shape == Wf.shape, f"shape mismatch {Wb.shape} vs {Wf.shape}"
    assert U.shape[0] == Wb.shape[1], f"block ambient {U.shape[0]} != input dim {Wb.shape[1]}"
    assert 1 <= r <= Wb.shape[1], f"motion rank {r} out of range"

    A_vault, Q_bulk = CEQ.vault_split(Wb, U)
    D = Wf - Wb
    # Motion file: rank-r truncated SVD of the delta. The highway basis is DATA-DERIVED per
    # fine-tune rather than fixed, because the delta's own leading directions ARE what "highway
    # motion" names; a fixed basis would price a different object.
    Um, s, Vt = np.linalg.svd(D, full_matrices=False)
    motion = {"U": Um[:, :r] * s[:r], "Vt": Vt[:r], "r": int(r)}
    return {"stable_block": U, "vault": A_vault, "bulk": Q_bulk, "motion": motion,
            "bits": int(bits), "shape": tuple(Wb.shape),
            "delta_energy_captured": float((s[:r] ** 2).sum() / (s ** 2).sum())}


def decode(enc: dict):
    """Reconstruct W_ft from the encoded parts: (vault ⊕ quantized bulk) + rank-r motion."""
    import numpy as np
    U = enc["stable_block"]
    base_hat = enc["vault"] @ U.T + enc["bulk"]
    return base_hat + enc["motion"]["U"] @ enc["motion"]["Vt"]


def roundtrip(W_base, W_ft, U_block, r: int, *, bits: int = 4) -> dict:
    """Encode → decode → error, with the two error sources separated.

    Separating them matters: the BASE error is the format's quantization of the frozen backbone and
    is paid once per family, while the DELTA error is the rank-r truncation and is paid per
    fine-tune. A single blended number would let a bad truncation hide behind a good vault.
    """
    import numpy as np
    enc = encode(W_base, W_ft, U_block, r, bits=bits)
    W_hat = decode(enc)
    Wf = np.asarray(W_ft, dtype=np.float64)
    Wb = np.asarray(W_base, dtype=np.float64)
    base_hat = enc["vault"] @ enc["stable_block"].T + enc["bulk"]
    return {"rel_error_total": float(np.linalg.norm(W_hat - Wf) / np.linalg.norm(Wf)),
            "rel_error_base": float(np.linalg.norm(base_hat - Wb) / np.linalg.norm(Wb)),
            "rel_error_delta": float(
                np.linalg.norm((W_hat - base_hat) - (Wf - Wb)) / np.linalg.norm(Wf - Wb)),
            "delta_energy_captured": enc["delta_energy_captured"],
            "core_action_error": CEQ.core_action_error(Wb, enc["stable_block"], enc["vault"],
                                                       enc["bulk"]),
            "r": int(r), "bits": int(bits)}


def assert_roundtrip(rt: dict, *, tol_total: float, tol_core: float):
    """The §4 round-trip equivalence gate. Tolerances are arguments, so they are set in the job's
    frozen config and appear in the artifact — never chosen after reading the error."""
    assert rt["rel_error_total"] <= tol_total, (
        f"round-trip relative error {rt['rel_error_total']:.4e} exceeds the frozen tolerance "
        f"{tol_total:.4e} — the format does not reconstruct what it claims to store")
    assert rt["core_action_error"] <= tol_core, (
        f"core action error {rt['core_action_error']:.4e} exceeds {tol_core:.4e} — the vault is "
        f"not exact on the stable block, which is the one property the block is for")
    return {"gate": "roundtrip", "passed": True, **rt}


def bytes_breakdown(*, m: int, n: int, k_block: int, r: int, n_finetunes: int,
                    bits: int = 4, n_matrices_sharing_block: int = 1) -> dict:
    """What the format costs, charged honestly: the family-level parts ONCE, the per-fine-tune
    motion file `n_finetunes` times.

    Per matrix the base side is DARWIN's `vault` scheme exactly (A = W_base U at fp16, the basis U
    at fp16, the dense bulk at `bits`) — reused rather than re-derived so the format's price and the
    CEQ arms' price are the same accounting.

    THE STABLE BLOCK IS THE ONE THING THIS FORMAT AMORTISES, and it is the format's whole claim, so
    it is charged explicitly: the basis costs n*k*16 bits and is divided by the number of matrices
    that share it. With `n_matrices_sharing_block=1` the format is charged the full basis per
    matrix, which is the conservative accounting and the default — a shared-cost claim has to be
    asked for, not assumed.

    TWO INCUMBENTS, AND THE MATCHED ONE IS THE ONE THAT ADJUDICATES (defect M-D40). Both ship the
    identical rank-r motion file, so they differ from the format only in how the BACKBONE is
    stored:

      fp16      m*n at fp16 — an UNQUANTIZED backbone. Against this the format is ~3x cheaper at
                one fine-tune before the stable block does any work at all, because the comparison
                is really 4-bit-vs-fp16 quantization wearing the format's name. A gate on this
                ratio cannot fail and this number is reported for context only.
      matched   m*n at the SAME `bits` — a plain uniformly-quantized backbone. The format is
                strictly MORE expensive than this (it adds A and the shared basis on top of an
                identical quantized bulk), so the only way it can win is on CAPABILITY at matched
                bytes, which is the claim the vault actually makes: an exact action on the stable
                block that uniform quantization does not have. `capability_vs_bytes` therefore
                refuses to build a curve without the incumbent's capability MEASURED.
    """
    assert n_matrices_sharing_block >= 1, "a block cannot be shared by fewer than one matrix"
    v = CEQ.storage_bits(m=m, n=n, k=k_block, scheme="vault", bits=bits)
    basis_bits = n * k_block * 16
    family_bits = v["total_bits"] - basis_bits + basis_bits / n_matrices_sharing_block
    motion_bits = (m * r + r * n) * 16          # fp16 factors, the house convention for deltas
    total = family_bits + n_finetunes * motion_bits
    inc_fp16 = m * n * 16 + n_finetunes * motion_bits
    inc_matched = CEQ.storage_bits(m=m, n=n, k=0, scheme="uniform",
                                   bits=bits)["total_bits"] + n_finetunes * motion_bits
    return {"family_bits": float(family_bits), "basis_bits": int(basis_bits),
            "vault_scheme_bits": float(v["total_bits"]),
            "motion_bits_per_ft": int(motion_bits),
            "total_bits": float(total),
            "incumbent_fp16_total_bits": float(inc_fp16),
            "incumbent_matched_total_bits": float(inc_matched),
            "bits_ratio_vs_fp16": float(total / inc_fp16),
            "bits_ratio_vs_matched": float(total / inc_matched),
            "bits_per_weight": float(total / (m * n * max(n_finetunes, 1))),
            "n_finetunes": int(n_finetunes), "k_block": int(k_block), "r": int(r),
            "n_matrices_sharing_block": int(n_matrices_sharing_block)}


def affordable_incumbent_rank(r: int, *, m: int, n: int, k_block: int, n_finetunes: int,
                              bits: int = 4, n_matrices_sharing_block: int = 1) -> int:
    """The rank the matched-bits incumbent can buy for what the format spends at rank `r`.

    THIS IS WHAT MAKES THE COMPARISON TWO-SIDED. The format is strictly more expensive than the
    matched-bits incumbent at equal rank — 2.6% to 11% on PG1's shapes — so a gate asking "is the
    format cheaper?" can never pass, exactly as the old fp16 gate could never fail. Neither
    measures anything.

    The comparison that can land either way is at EQUAL BYTES: the format's surplus buys the
    incumbent extra motion rank, so the incumbent is run at the higher rank it can actually afford
    and the two are compared there. The format wins only if its vault beats the plain quantized
    backbone by more than the extra rank the same bytes would have bought.

    Returns the largest rank whose incumbent cost does not exceed the format's cost at `r`, and
    never less than `r` itself (the incumbent can always afford at least the format's own rank,
    since it is cheaper there).
    """
    fmt = bytes_breakdown(m=m, n=n, k_block=k_block, r=r, n_finetunes=n_finetunes, bits=bits,
                          n_matrices_sharing_block=n_matrices_sharing_block)["total_bits"]
    per_rank = n_finetunes * (m + n) * 16                 # one rank of motion, every fine-tune
    base = CEQ.storage_bits(m=m, n=n, k=0, scheme="uniform", bits=bits)["total_bits"]
    r_prime = int((fmt - base) // per_rank)
    assert r_prime >= r, (
        f"the matched-bits incumbent affords rank {r_prime} < the format's {r}, which contradicts "
        f"its being the cheaper scheme at equal rank — check the accounting, not the data")
    return min(r_prime, n)


def capability_vs_bytes(rank_ladder, capability_by_rank: dict,
                        incumbent_capability_by_rank: dict, *, m: int, n: int, k_block: int,
                        n_finetunes: int, bits: int = 4, n_matrices_sharing_block: int = 1,
                        higher_is_better: bool = True) -> list:
    """The M-C3 deliverable curve: measured capability against shipped bytes, one row per rank.

    BOTH capabilities are MEASURED, never modelled here — and the incumbent's is measured at the
    BYTE-MATCHED rank `affordable_incumbent_rank` returns, not at the format's rank. Keying it by
    the format's rank would compare two schemes at different prices; keying it by the affordable
    rank is the comparison that can land either way (M-D40). `incumbent_capability_by_rank` is
    therefore indexed by the incumbent's OWN ranks.

    `higher_is_better` names the readout's polarity, because the two arms that use this disagree:
    M-C3 reads a task SCORE and M-6 reads a task LOSS. `capability_gain` is signed so that positive
    always means the format is better, whichever way the readout points.
    """
    rows = []
    for r in rank_ladder:
        assert r in capability_by_rank, f"rank {r} has no measured format capability"
        rp = affordable_incumbent_rank(int(r), m=m, n=n, k_block=k_block,
                                       n_finetunes=n_finetunes, bits=bits,
                                       n_matrices_sharing_block=n_matrices_sharing_block)
        assert rp in incumbent_capability_by_rank, (
            f"the incumbent affords rank {rp} for what the format spends at rank {r}, and no "
            f"capability was measured there — measure the incumbent at the ranks "
            f"`affordable_incumbent_rank` names, or the comparison is not at equal bytes")
        b = bytes_breakdown(m=m, n=n, k_block=k_block, r=int(r), n_finetunes=n_finetunes,
                            bits=bits, n_matrices_sharing_block=n_matrices_sharing_block)
        inc_bits = bytes_breakdown(m=m, n=n, k_block=k_block, r=rp, n_finetunes=n_finetunes,
                                   bits=bits, n_matrices_sharing_block=n_matrices_sharing_block
                                   )["incumbent_matched_total_bits"]
        one_rank = n_finetunes * (m + n) * 16
        assert 0 <= b["total_bits"] - inc_bits < one_rank, (
            f"rank {r}: the byte-matched incumbent should sit just under the format — within one "
            f"rank of motion and never above it — but the gap is "
            f"{b['total_bits'] - inc_bits:.0f} bits against a rank costing {one_rank}")
        d = float(capability_by_rank[r]) - float(incumbent_capability_by_rank[rp])
        rows.append({"r": int(r), "incumbent_r": int(rp),
                     "capability": float(capability_by_rank[r]),
                     "incumbent_capability": float(incumbent_capability_by_rank[rp]),
                     "capability_gain": d if higher_is_better else -d,
                     "higher_is_better": bool(higher_is_better),
                     "total_bits": b["total_bits"],
                     "incumbent_total_bits": float(inc_bits),
                     "incumbent_fp16_total_bits": b["incumbent_fp16_total_bits"],
                     "bits_ratio": float(b["total_bits"] / inc_bits),
                     "bits_ratio_vs_fp16": b["bits_ratio_vs_fp16"]})
    return rows


def pareto_wins(curve) -> list:
    """The ranks where the format beats the byte-matched incumbent.

    THE BYTE SIDE IS SETTLED BY CONSTRUCTION AND IS NOT RE-TESTED HERE. `affordable_incumbent_rank`
    floors, so the incumbent is handed the largest rank that costs NO MORE than the format — its
    bytes land just under the format's, never over, and `capability_vs_bytes` asserts that. Gating
    on `bits_ratio <= 1.0` on top of that would fail at every rank by the rounding remainder alone,
    which is the same can't-pass defect one step further down (M-D40). Any rounding slack is given
    to the INCUMBENT, so a format win here is the conservative direction.

    What decides is capability. An empty list is the refuter firing, not a missing number.
    """
    return [row["r"] for row in curve if row["capability_gain"] > 0.0]
