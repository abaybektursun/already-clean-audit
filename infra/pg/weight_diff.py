"""W1 — Weight-difference pipeline (spec §7 W1, controls N1/N3/P2).

Given two state_dicts of matching architecture, per-parameter tensor:
  (a) relative Frobenius norm  ||ΔW||_F / ||W_base||_F
  (b) effective rank of ΔW for 2-D matrices via the SVD participation ratio
      (Σ σ_i)² / Σ σ_i²  (∈ [1, min(m,n)]; high ≈ full-rank, low ≈ low-rank)

Aggregated per layer/module-type.

Design rules:
  * No try/except — fail loud with full trace.
  * Returns plain Python dicts/lists — JSON-serialisable.
  * No GPU required; all math is on CPU tensors.
"""
from __future__ import annotations

import math
from typing import Any


def _classify_key(key: str) -> str:
    """Coarse module-type bucket for aggregation."""
    key_lower = key.lower()
    if "embed" in key_lower:
        return "embedding"
    if "lm_head" in key_lower:
        return "lm_head"
    if "self_attn" in key_lower or "attention" in key_lower:
        return "attention"
    if "mlp" in key_lower or "feed_forward" in key_lower:
        return "mlp"
    if "norm" in key_lower or "layernorm" in key_lower:
        return "norm"
    if "projector" in key_lower or "connector" in key_lower:
        return "projector"
    if "vision" in key_lower:
        return "vision_other"
    return "other"


def _layer_index(key: str) -> int:
    """Extract the first integer from the key (layer index), or -1."""
    import re
    m = re.search(r"\d+", key)
    return int(m.group()) if m else -1


def _participation_ratio(singular_values) -> float:
    """SVD participation ratio = (Σσ_i)² / Σσ_i²  ∈ [1, rank].

    For a rank-1 matrix σ = [s, 0, …] → PR = 1.
    For uniform σ → PR = rank (maximum).
    Requires singular_values to be a 1-D tensor.
    """
    import torch
    sv = singular_values.float()
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return 0.0
    numerator = sv.sum().item() ** 2
    denominator = (sv ** 2).sum().item()
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_weight_diff(
    state_dict_base: dict,
    state_dict_other: dict,
    assert_zero: bool = False,
) -> dict[str, Any]:
    """Compute per-tensor ΔW metrics.

    Parameters
    ----------
    state_dict_base:  the "base" / "before" model's state_dict
    state_dict_other: the "other" / "after" model's state_dict
    assert_zero:      if True, assert every |ΔW||_F == 0 (N1 identity control)

    Returns
    -------
    dict with keys:
        'per_tensor':  list of per-key dicts with rel_frob and (opt) eff_rank
        'by_module_type': aggregated stats per module-type bucket
        'summary':     scalar summary stats
    """
    import torch

    keys_base  = set(state_dict_base.keys())
    keys_other = set(state_dict_other.keys())
    assert keys_base == keys_other, (
        f"State-dict key mismatch: only-in-base={keys_base - keys_other!r}, "
        f"only-in-other={keys_other - keys_base!r}"
    )

    per_tensor: list[dict] = []
    by_module: dict[str, list] = {}

    for key in sorted(state_dict_base.keys()):
        w_base  = state_dict_base[key].float().cpu()
        w_other = state_dict_other[key].float().cpu()

        assert w_base.shape == w_other.shape, (
            f"Shape mismatch for {key!r}: base={w_base.shape} other={w_other.shape}"
        )

        delta = w_other - w_base

        norm_delta = delta.norm(p="fro").item()
        norm_base  = w_base.norm(p="fro").item()

        if norm_base == 0.0:
            # Zero-init base (e.g. projector bias at init).  Use absolute norm.
            rel_frob = norm_delta  # raw absolute; flag it
            zero_base = True
        else:
            rel_frob = norm_delta / norm_base
            zero_base = False

        if assert_zero:
            assert norm_delta == 0.0, (
                f"N1 FAIL: {key!r} has ||ΔW||_F = {norm_delta:.6e} (expected 0)."
            )

        record: dict[str, Any] = {
            "key":       key,
            "shape":     list(w_base.shape),
            "module_type": _classify_key(key),
            "layer_idx": _layer_index(key),
            "norm_base":  norm_base,
            "norm_delta": norm_delta,
            "rel_frob":   rel_frob,
            "zero_base":  zero_base,
        }

        # Effective rank only for 2-D weight matrices (linear layers, attention projections).
        if delta.ndim == 2:
            sv = torch.linalg.svdvals(delta)
            record["eff_rank"]  = _participation_ratio(sv)
            record["max_rank"]  = min(delta.shape)
            record["sv_top5"]   = sv[:5].tolist()
        else:
            record["eff_rank"] = None
            record["max_rank"] = None

        per_tensor.append(record)

        mtype = record["module_type"]
        by_module.setdefault(mtype, []).append(record)

    # --- aggregate by module type ------------------------------------------
    def _agg(records: list[dict]) -> dict:
        rel_frobs = [r["rel_frob"] for r in records]
        ranks     = [r["eff_rank"] for r in records if r["eff_rank"] is not None]
        n = len(rel_frobs)
        return {
            "n_tensors":       n,
            "rel_frob_mean":   sum(rel_frobs) / n,
            "rel_frob_max":    max(rel_frobs),
            "rel_frob_median": sorted(rel_frobs)[n // 2],
            "eff_rank_mean":   sum(ranks) / len(ranks) if ranks else None,
            "eff_rank_max":    max(ranks) if ranks else None,
        }

    by_module_type = {mtype: _agg(recs) for mtype, recs in by_module.items()}

    # --- scalar summary -------------------------------------------------------
    all_rel = [r["rel_frob"] for r in per_tensor]
    all_rank = [r["eff_rank"] for r in per_tensor if r["eff_rank"] is not None]
    n_total  = len(per_tensor)

    summary = {
        "n_tensors_total":  n_total,
        "rel_frob_mean":    sum(all_rel) / n_total,
        "rel_frob_max":     max(all_rel),
        "rel_frob_median":  sorted(all_rel)[n_total // 2],
        "eff_rank_mean":    sum(all_rank) / len(all_rank) if all_rank else None,
        "eff_rank_max":     max(all_rank) if all_rank else None,
        "n_zero_base_keys": sum(1 for r in per_tensor if r["zero_base"]),
    }

    return {
        "per_tensor":    per_tensor,
        "by_module_type": by_module_type,
        "summary":       summary,
    }
