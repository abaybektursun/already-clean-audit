"""Pure numerical construction and adjudication for the PG2 scalar operator cell."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def categorical_fisher_apply(probabilities: Any, tangent: Any) -> Any:
    """Apply ``diag(p)-p p^T`` along the final dimension."""
    centered = tangent - (probabilities * tangent).sum(dim=-1, keepdim=True)
    return probabilities * centered


def categorical_fisher_rayleigh(probabilities: Any, tangent: Any) -> Any:
    return (tangent * categorical_fisher_apply(probabilities, tangent)).sum(dim=-1)


def deterministic_directions(
    sensitive: Any, *, count: int, namespace: str
) -> list[Any]:
    import torch

    if count < 1:
        raise ValueError("direction count must be positive")
    first = sensitive.detach().cpu().float()
    first = first / first.norm().clamp_min(torch.finfo(torch.float32).eps)
    directions = [first]
    for index in range(1, count):
        digest = hashlib.sha256(f"{namespace}:{index}".encode()).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big"))
        candidate = torch.randn(first.shape, generator=generator, dtype=torch.float32)
        for prior in directions:
            candidate = candidate - torch.dot(candidate, prior) * prior
        norm = candidate.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= torch.finfo(torch.float32).eps:
            raise ValueError(f"direction {index} collapsed")
        directions.append(candidate / norm)
    return directions


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1e-30)


def adjudicate_scalar_operator(cells: Sequence[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    by_direction: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        by_direction[int(row["direction_index"])].append(row)
    required_directions = int(gates["directions_required"])
    required_eps = int(gates["usable_values_required_per_direction"])
    usable = {
        direction: sorted(
            (row for row in rows if bool(row["usable"])), key=lambda row: float(row["epsilon"])
        )
        for direction, rows in by_direction.items()
    }
    usable_counts = {str(direction): len(rows) for direction, rows in sorted(usable.items())}
    if len(by_direction) != required_directions or any(
        len(usable.get(index, [])) < required_eps for index in range(required_directions)
    ):
        return {
            "status": "REDESIGN_NUMERICAL_FLOOR_OR_SATURATION",
            "directions": len(by_direction),
            "usable_epsilon_counts": usable_counts,
        }

    accepted = [row for rows in usable.values() for row in rows]
    kl_errors = [
        _relative(row["symmetric_direct_kl_curvature"], row["vjp_rayleigh"])
        for row in accepted
    ]
    route_errors = [
        _relative(row["finite_difference_fisher_rayleigh"], row["vjp_rayleigh"])
        for row in accepted
    ]
    slopes = []
    jvp_changes = []
    for rows in usable.values():
        for left, right in zip(rows, rows[1:]):
            slopes.append(
                math.log(
                    float(right["symmetric_kl_nats"])
                    / float(left["symmetric_kl_nats"])
                )
                / math.log(float(right["epsilon"]) / float(left["epsilon"]))
            )
            if right.get("jvp_relative_change_from_previous") is not None:
                jvp_changes.append(float(right["jvp_relative_change_from_previous"]))
    diagnostics = {
        "directions": len(by_direction),
        "accepted_cells": len(accepted),
        "usable_epsilon_counts": usable_counts,
        "kl_q_relative_error": {
            "median": _quantile(kl_errors, 0.5),
            "p90": _quantile(kl_errors, 0.9),
        },
        "fisher_vjp_relative_error": {
            "median": _quantile(route_errors, 0.5),
            "p90": _quantile(route_errors, 0.9),
        },
        "epsilon": {
            "max_adjacent_jvp_relative_change": max(jvp_changes),
            "kl_slope_median_abs_error_from_two": _quantile(
                [abs(value - 2.0) for value in slopes], 0.5
            ),
            "kl_slope_p90_abs_error_from_two": _quantile(
                [abs(value - 2.0) for value in slopes], 0.9
            ),
        },
        "minimum_vjp_rayleigh": min(float(row["vjp_rayleigh"]) for row in accepted),
    }
    if (
        diagnostics["kl_q_relative_error"]["median"]
        > float(gates["kl_q_median_relative_error_max"])
        or diagnostics["kl_q_relative_error"]["p90"]
        > float(gates["kl_q_p90_relative_error_max"])
    ):
        status = "REDESIGN_KL_QUADRATIC_IDENTITY"
    elif (
        diagnostics["fisher_vjp_relative_error"]["median"]
        > float(gates["fisher_vjp_median_relative_error_max"])
        or diagnostics["fisher_vjp_relative_error"]["p90"]
        > float(gates["fisher_vjp_p90_relative_error_max"])
    ):
        status = "REDESIGN_FISHER_VJP_ROUTE"
    elif (
        diagnostics["epsilon"]["max_adjacent_jvp_relative_change"]
        > float(gates["adjacent_epsilon_jvp_relative_change_max"])
        or diagnostics["epsilon"]["kl_slope_median_abs_error_from_two"]
        > float(gates["kl_loglog_slope_median_abs_error_from_two_max"])
        or diagnostics["epsilon"]["kl_slope_p90_abs_error_from_two"]
        > float(gates["kl_loglog_slope_p90_abs_error_from_two_max"])
    ):
        status = "REDESIGN_EPSILON_STABILITY"
    else:
        positive_scale = max(
            _quantile([abs(float(row["vjp_rayleigh"])) for row in accepted], 0.5), 1e-30
        )
        psd_floor = -float(gates["psd_relative_negative_tolerance"]) * positive_scale
        diagnostics["psd_floor"] = psd_floor
        status = (
            "SCALAR_ROUTE_GREEN"
            if diagnostics["minimum_vjp_rayleigh"] >= psd_floor
            else "REDESIGN_PSD"
        )
    return {"status": status, **diagnostics}
