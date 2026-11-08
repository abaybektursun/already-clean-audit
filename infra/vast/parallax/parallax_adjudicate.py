"""Frozen branch adjudicators with planted-fixture-friendly inputs."""

from __future__ import annotations


def graft_world(
    *, native_effect: float, graft_effect: float, repaired_effect: float, mde: float
) -> str:
    if mde <= 0:
        raise ValueError("MDE must be positive")
    native = native_effect >= mde
    graft = graft_effect >= mde
    repaired = repaired_effect >= mde
    if not native:
        return "VOID_NO_NATIVE_RANGE"
    if graft:
        return "PORTABLE"
    if repaired:
        return "REMAPPABLE"
    return "CO_ADAPTED_OR_BRIDGE_DISABLED"


def operator_world(*, scalar_green: bool, replay_green: bool, behavioral_range: bool) -> str:
    if not behavioral_range:
        return "VOID_NO_DIRECTIONAL_RANGE"
    if not scalar_green:
        return "REDESIGN"
    if not replay_green:
        return "SITE_LOCAL_VOID"
    return "VALID"
