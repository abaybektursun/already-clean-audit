"""Pure field, energy, and adjudication contracts for the paired behavioral MDE."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class BehavioralGateError(RuntimeError):
    """Raised when a behavioral perturbation crosses its frozen contract."""


def select_target_stratified(
    rows: Iterable[dict[str, Any]], *, task_id: str, per_target: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task_id") == task_id and row.get("image") is not None:
            grouped[str(row["target"])].append(row)
    if not grouped:
        raise BehavioralGateError(f"no rows for {task_id}")
    selected: list[dict[str, Any]] = []
    for target, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: str(row["item_id"]))
        if len(ordered) < per_target:
            raise BehavioralGateError(f"underfilled target stratum {target}: {len(ordered)}")
        selected.extend(ordered[:per_target])
    selected.sort(key=lambda row: (str(row["target"]), str(row["item_id"])))
    if len({row["item_id"] for row in selected}) != len(selected):
        raise BehavioralGateError("selected panel contains duplicate item IDs")
    return selected


def deterministic_seed(namespace: str, lineage: str, item_id: str, construction: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{lineage}:{item_id}:{construction}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _orthogonalize(values: Any, direction: Any) -> Any:
    import torch

    unit = direction / direction.norm().clamp_min(torch.finfo(torch.float32).eps)
    projected = values - torch.sum(values * unit, dim=-1, keepdim=True) * unit
    norms = projected.norm(dim=-1, keepdim=True)
    if bool((norms <= torch.finfo(torch.float32).eps).any()):
        raise BehavioralGateError("random control collapsed during orthogonalization")
    return projected / norms


def make_field(
    *,
    construction: str,
    sensitive_direction: Any,
    active_mask: Any,
    activation_rms: Any,
    amplitude: float,
    namespace: str,
    lineage: str,
    item_ids: Sequence[str],
) -> Any:
    """Create fields with exact equal L2 norm at every affected token."""
    import torch

    if amplitude < 0:
        raise BehavioralGateError("amplitude must be nonnegative")
    if construction not in {"sensitive_coherent", "rotated_coherent", "white"}:
        raise BehavioralGateError(f"unknown construction: {construction}")
    if active_mask.ndim != 2 or len(item_ids) != active_mask.shape[0]:
        raise BehavioralGateError("active mask or item ID batch mismatch")
    direction = sensitive_direction.float()
    if direction.ndim != 1 or not bool(torch.isfinite(direction).all()) or float(direction.norm()) == 0:
        raise BehavioralGateError("sensitive direction is zero or nonfinite")
    direction = direction / direction.norm()
    batch, tokens = active_mask.shape
    width = direction.numel()
    target_norms = activation_rms.float().reshape(batch, 1, 1) * float(amplitude) * math.sqrt(width)
    fields = []
    for index, item_id in enumerate(item_ids):
        if construction == "sensitive_coherent":
            vectors = direction.expand(tokens, width)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(namespace, lineage, item_id, construction))
            count = 1 if construction == "rotated_coherent" else tokens
            random_values = torch.randn(count, width, generator=generator, dtype=torch.float32)
            vectors = _orthogonalize(random_values, direction.cpu())
            if construction == "rotated_coherent":
                vectors = vectors.expand(tokens, width)
        field = vectors.to(active_mask.device) * target_norms[index]
        field = field * active_mask[index].reshape(tokens, 1)
        fields.append(field)
    return torch.stack(fields).to(dtype=sensitive_direction.dtype)


def energy_audit(
    field: Any,
    active_mask: Any,
    activation_rms: Any,
    amplitude: float,
    *,
    relative_tolerance: float = 2e-5,
) -> dict[str, Any]:
    import torch

    width = field.shape[-1]
    norms = field.float().norm(dim=-1)
    expected = activation_rms.float().reshape(-1, 1) * float(amplitude) * math.sqrt(width)
    active = active_mask.bool()
    if not bool(active.any()):
        raise BehavioralGateError("energy audit has no active tokens")
    max_abs = float((norms[active] - expected.expand_as(norms)[active]).abs().max())
    scale = float(expected.expand_as(norms)[active].abs().max())
    inactive_max = float(norms[~active].max()) if bool((~active).any()) else 0.0
    relative = max_abs / max(scale, 1e-30)
    status = "GREEN" if relative <= relative_tolerance and inactive_max == 0 else "RED"
    if status != "GREEN":
        raise BehavioralGateError(
            f"per-token energy mismatch: relative={relative} inactive_max={inactive_max}"
        )
    return {
        "status": status,
        "active_tokens": int(active.sum()),
        "max_relative_norm_error": relative,
        "relative_tolerance": relative_tolerance,
        "injection_dtype": str(field.dtype),
        "inactive_max_norm": inactive_max,
    }


def select_amplitude(
    rows: Sequence[dict[str, float]],
    ladder: Sequence[float],
    *,
    median_kl_min: float,
    median_kl_max: float,
    p90_kl_max: float,
) -> dict[str, Any]:
    by_amplitude: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_amplitude[float(row["amplitude"])].append(float(row["kl_nats"]))
    diagnostics: dict[str, Any] = {}
    chosen = None
    for amplitude in ladder:
        values = sorted(by_amplitude.get(float(amplitude), []))
        if not values:
            raise BehavioralGateError(f"amplitude ladder is incomplete: {amplitude}")
        median = _quantile(values, 0.5)
        p90 = _quantile(values, 0.9)
        usable = median_kl_min <= median <= median_kl_max and p90 <= p90_kl_max
        diagnostics[str(amplitude)] = {"median_kl": median, "p90_kl": p90, "usable": usable}
        if chosen is None and usable:
            chosen = float(amplitude)
    return {
        "status": "GREEN" if chosen is not None else "VOID_NO_READABLE_AMPLITUDE",
        "selected_amplitude": chosen,
        "diagnostics": diagnostics,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise BehavioralGateError("quantile requires values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sign_flip_mde(
    differences: Sequence[float], *, seed: int, draws: int, quantile: float, minimum: float
) -> dict[str, float]:
    if not differences:
        raise BehavioralGateError("sign-flip MDE requires differences")
    rng = random.Random(seed)
    magnitudes = []
    values = [float(value) for value in differences]
    for _ in range(draws):
        magnitudes.append(abs(sum(value * (1 if rng.getrandbits(1) else -1) for value in values) / len(values)))
    band = _quantile(magnitudes, quantile)
    return {"sign_flip_band_nats": band, "minimum_nats": minimum, "frozen_mde_nats": max(band, minimum)}


def paired_bootstrap_lower(
    left: Sequence[float], right: Sequence[float], *, seed: int, draws: int, quantile: float
) -> float:
    if len(left) != len(right) or not left:
        raise BehavioralGateError("paired bootstrap requires equal nonempty samples")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    return _quantile(means, quantile)


def adjudicate_range(
    sensitive: Mapping[str, float],
    rotated: Mapping[str, float],
    white: Mapping[str, float],
    *,
    frozen_mde: float,
    seed: int,
    draws: int,
    lower_quantile: float,
) -> dict[str, Any]:
    if not (set(sensitive) == set(rotated) == set(white)):
        raise BehavioralGateError("confirmation constructions have different item IDs")
    ids = sorted(sensitive)
    comparisons: dict[str, Any] = {}
    for offset, (name, null) in enumerate((("rotated", rotated), ("white", white))):
        left = [sensitive[item] for item in ids]
        right = [null[item] for item in ids]
        effect = sum(a - b for a, b in zip(left, right)) / len(ids)
        lower = paired_bootstrap_lower(
            left, right, seed=seed + offset, draws=draws, quantile=lower_quantile
        )
        comparisons[name] = {
            "mean_effect_nats": effect,
            "effect_over_mde": effect / frozen_mde,
            "bootstrap_familywise_lower": lower,
            "green": effect >= frozen_mde and lower > 0,
        }
    green = all(row["green"] for row in comparisons.values())
    return {
        "status": "GREEN_DIRECTIONAL_RANGE" if green else "NO_RANGE",
        "items": len(ids),
        "frozen_mde_nats": frozen_mde,
        "comparisons": comparisons,
    }
