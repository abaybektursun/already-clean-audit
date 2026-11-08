"""Pure contracts for the PG2 image-token-position versus prompt-position discriminator."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .behavioral_common import BehavioralGateError, paired_bootstrap_lower


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise BehavioralGateError("localization quantile requires values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def select_localization_amplitude(
    rows: list[dict[str, Any]],
    ladder: list[float],
    *,
    median_kl_min: float,
    median_kl_max: float,
    p90_kl_max: float,
    per_token_fraction_max: float,
) -> dict[str, Any]:
    """Select the smallest full-readable amplitude safe in every location."""
    values: dict[tuple[float, str], list[float]] = defaultdict(list)
    per_token: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (float(row["amplitude"]), str(row["arm"]))
        values[key].append(float(row["kl_nats"]))
        per_token[key].append(float(row["per_token_activation_rms_fraction"]))
    arms = ("full_sensitive", "image_sensitive", "prompt_sensitive")
    diagnostics: dict[str, Any] = {}
    selected = None
    for amplitude in ladder:
        by_arm: dict[str, Any] = {}
        complete = True
        for arm in arms:
            key = (float(amplitude), arm)
            if not values[key]:
                complete = False
                by_arm[arm] = {"status": "SKIPPED_UNSAFE_OR_MISSING"}
                continue
            by_arm[arm] = {
                "status": "EVALUATED",
                "median_kl": _quantile(values[key], 0.5),
                "p90_kl": _quantile(values[key], 0.9),
                "max_per_token_activation_rms_fraction": max(per_token[key]),
            }
        full = by_arm["full_sensitive"]
        usable = complete and (
            median_kl_min <= full["median_kl"] <= median_kl_max
            and all(row["p90_kl"] <= p90_kl_max for row in by_arm.values())
            and all(
                row["max_per_token_activation_rms_fraction"] <= per_token_fraction_max
                for row in by_arm.values()
            )
        )
        diagnostics[str(amplitude)] = {
            "arms": by_arm,
            "all_location_cells_evaluated": complete,
            "usable": usable,
        }
        if selected is None and usable:
            selected = float(amplitude)
    return {
        "status": "GREEN" if selected is not None else "VOID_NO_COMMON_SAFE_AMPLITUDE",
        "selected_amplitude": selected,
        "diagnostics": diagnostics,
    }


def prefix_location_masks(
    inputs: Mapping[str, Any], *, image_token_index: int, expected_image_tokens: int
) -> dict[str, Any]:
    """Partition the non-padding, non-suffix prefix without relying on token_type_ids."""
    attention = inputs["attention_mask"].bool()
    suffix = inputs["labels"].ne(-100)
    prefix = attention & ~suffix
    image = prefix & inputs["input_ids"].eq(int(image_token_index))
    prompt = prefix & ~image
    if bool((image & prompt).any()) or not bool(torch_equal(image | prompt, prefix)):
        raise BehavioralGateError("image/prompt masks do not exactly partition the prefix")
    if bool((image & suffix).any()) or bool((prompt & suffix).any()):
        raise BehavioralGateError("localization mask leaked into the teacher-forced suffix")
    image_counts = image.sum(dim=1)
    prompt_counts = prompt.sum(dim=1)
    if not bool((image_counts == int(expected_image_tokens)).all()):
        raise BehavioralGateError(f"unexpected image-token count: {image_counts.tolist()}")
    if bool((prompt_counts <= 0).any()):
        raise BehavioralGateError("an item has no prompt/BOS prefix positions")
    return {"full_prefix": prefix, "image_tokens": image, "prompt_tokens": prompt}


def torch_equal(left: Any, right: Any) -> bool:
    import torch

    return bool(torch.equal(left, right))


def equal_total_energy_rms(
    activation_rms: Any, *, reference_mask: Any, location_mask: Any
) -> Any:
    """Scale per-token RMS so each location receives the full-prefix total L2 energy."""
    reference_counts = reference_mask.bool().sum(dim=1).float()
    location_counts = location_mask.bool().sum(dim=1).float()
    if bool((reference_counts <= 0).any()) or bool((location_counts <= 0).any()):
        raise BehavioralGateError("equal-total-energy scaling requires nonempty masks")
    if bool((location_counts > reference_counts).any()):
        raise BehavioralGateError("location mask is not contained in its reference prefix")
    return activation_rms.float() * torch_sqrt(reference_counts / location_counts)


def torch_sqrt(values: Any) -> Any:
    import torch

    return torch.sqrt(values)


def total_energy_audit(
    field: Any,
    *,
    location_mask: Any,
    reference_mask: Any,
    activation_rms: Any,
    amplitude: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Audit total BF16 field energy against the common full-prefix budget per item."""
    width = int(field.shape[-1])
    observed = field.float().square().sum(dim=(1, 2))
    reference_counts = reference_mask.bool().sum(dim=1).float()
    expected = activation_rms.float().square() * float(amplitude) ** 2 * width * reference_counts
    inactive = field.float().norm(dim=-1)[~location_mask.bool()]
    inactive_max = float(inactive.max()) if inactive.numel() else 0.0
    relative = (observed - expected).abs() / expected.clamp_min(1e-30)
    maximum = float(relative.max())
    if maximum > relative_tolerance or inactive_max != 0.0:
        raise BehavioralGateError(
            f"total energy mismatch: relative={maximum} inactive_max={inactive_max}"
        )
    return {
        "status": "GREEN",
        "max_relative_total_energy_error": maximum,
        "relative_tolerance": relative_tolerance,
        "inactive_max_norm": inactive_max,
        "reference_tokens": [int(value) for value in reference_counts.tolist()],
        "location_tokens": [int(value) for value in location_mask.bool().sum(dim=1).tolist()],
    }


def _paired_difference(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    if set(left) != set(right) or not left:
        raise BehavioralGateError("localization cells require identical nonempty item IDs")
    return {item: float(left[item]) - float(right[item]) for item in sorted(left)}


def _contrast_gate(
    differences: Mapping[str, float], *, mde: float, seed: int, draws: int, lower_quantile: float
) -> dict[str, Any]:
    ids = sorted(differences)
    values = [float(differences[item]) for item in ids]
    zeros = [0.0] * len(values)
    effect = math.fsum(values) / len(values)
    positive_lower = paired_bootstrap_lower(
        values, zeros, seed=seed, draws=draws, quantile=lower_quantile
    )
    negative_lower = paired_bootstrap_lower(
        [-value for value in values], zeros, seed=seed + 1, draws=draws, quantile=lower_quantile
    )
    return {
        "mean_effect_nats": effect,
        "effect_over_mde": effect / mde,
        "positive_familywise_lower": positive_lower,
        "negative_familywise_lower": negative_lower,
        "positive_green": effect >= mde and positive_lower > 0,
        "negative_green": effect <= -mde and negative_lower > 0,
    }


def adjudicate_localization(
    cells: Mapping[str, Mapping[str, float]],
    *,
    frozen_mde: float,
    seed: int,
    draws: int,
    lower_quantile: float,
) -> dict[str, Any]:
    """Adjudicate position localization without interpreting image slots as image semantics."""
    required = {
        "full_sensitive",
        "full_rotated",
        "full_white",
        "image_sensitive",
        "image_rotated",
        "prompt_sensitive",
        "prompt_rotated",
    }
    if set(cells) != required:
        raise BehavioralGateError(f"localization cell mismatch: {sorted(set(cells) ^ required)}")
    item_sets = {frozenset(values) for values in cells.values()}
    if len(item_sets) != 1:
        raise BehavioralGateError("localization cells have different item IDs")
    advantages = {
        "full_rotated": _paired_difference(cells["full_sensitive"], cells["full_rotated"]),
        "full_white": _paired_difference(cells["full_sensitive"], cells["full_white"]),
        "image": _paired_difference(cells["image_sensitive"], cells["image_rotated"]),
        "prompt": _paired_difference(cells["prompt_sensitive"], cells["prompt_rotated"]),
    }
    image_minus_prompt = _paired_difference(advantages["image"], advantages["prompt"])
    gates = {
        "full_rotated": _contrast_gate(
            advantages["full_rotated"], mde=frozen_mde, seed=seed, draws=draws,
            lower_quantile=lower_quantile,
        ),
        "full_white": _contrast_gate(
            advantages["full_white"], mde=frozen_mde, seed=seed + 5, draws=draws,
            lower_quantile=lower_quantile,
        ),
        "image": _contrast_gate(
            advantages["image"], mde=frozen_mde, seed=seed + 10, draws=draws,
            lower_quantile=lower_quantile,
        ),
        "prompt": _contrast_gate(
            advantages["prompt"], mde=frozen_mde, seed=seed + 20, draws=draws,
            lower_quantile=lower_quantile,
        ),
        "image_minus_prompt": _contrast_gate(
            image_minus_prompt, mde=frozen_mde, seed=seed + 30, draws=draws,
            lower_quantile=lower_quantile,
        ),
    }
    full_replication = gates["full_rotated"]["positive_green"] and gates["full_white"]["positive_green"]
    if not full_replication:
        status = "VOID_NO_FULL_PREFIX_REPLICATION"
    elif gates["image"]["positive_green"] and gates["image_minus_prompt"]["positive_green"]:
        status = "IMAGE_TOKEN_POSITION_DOMINANT"
    elif gates["prompt"]["positive_green"] and gates["image_minus_prompt"]["negative_green"]:
        status = "PROMPT_POSITION_DOMINANT"
    elif gates["image"]["positive_green"] and gates["prompt"]["positive_green"]:
        status = "MIXED_POSITION_RANGE"
    else:
        status = "DISTRIBUTED_OR_UNRESOLVED_POSITION_RANGE"
    return {
        "status": status,
        "items": len(next(iter(item_sets))),
        "frozen_mde_nats": frozen_mde,
        "gates": gates,
        "full_prefix_replication_green": full_replication,
        "semantic_claim_licensed": False,
    }
