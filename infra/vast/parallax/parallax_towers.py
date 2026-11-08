"""Strict tower-key boundary checks for planted and real-model graft gates."""

from __future__ import annotations

from collections.abc import Mapping


VISION_PREFIXES = ("vision_tower.", "model.vision_tower.")
FORBIDDEN_FIXED_MARKERS = ("embed_tokens", "lm_head", "multi_modal_projector", "projector")


def key_role(key: str) -> str:
    if key.startswith(VISION_PREFIXES):
        return "vision"
    if any(marker in key for marker in FORBIDDEN_FIXED_MARKERS):
        return "fixed_interface"
    if "language_model" in key or key.startswith("model.language_model."):
        return "language"
    return "unmapped"


def replacement_plan(
    host_shapes: Mapping[str, tuple[int, ...]],
    donor_shapes: Mapping[str, tuple[int, ...]],
    *,
    role: str,
) -> dict[str, str]:
    if role not in {"vision", "language"}:
        raise ValueError("role must be vision or language")
    plan: dict[str, str] = {}
    for key, host_shape in host_shapes.items():
        if key_role(key) != role:
            continue
        donor_shape = donor_shapes.get(key)
        if donor_shape is None:
            plan[key] = "missing_donor"
        elif tuple(donor_shape) != tuple(host_shape):
            plan[key] = "shape_mismatch"
        else:
            plan[key] = "replace"
    return plan


def assert_fixed_interface_untouched(plan: Mapping[str, str]) -> None:
    offenders = [key for key, action in plan.items() if action == "replace" and key_role(key) == "fixed_interface"]
    if offenders:
        raise ValueError(f"replacement crosses fixed interface: {offenders}")
