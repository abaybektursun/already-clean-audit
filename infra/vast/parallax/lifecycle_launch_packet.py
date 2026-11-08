"""Lifecycle-v2 extension for future PARALLAX big-run launch packets.

The v1 validator remains byte-identical because historical frozen transfer
contracts bind it. Every new research lifecycle spec must use this validator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .launch_packet import LaunchPacketError, validate_launch_packet


LIFECYCLE_GREEN_PATHS = (
    "weight_free_plants",
    "continuous_mirror",
    "detached_watchdog",
    "provider_ttl_guard",
    "bootstrap_orphan_guard",
    "durability_before_destroy",
    "stop_preserves_disk",
    "live_rehearsal",
)


def validate_lifecycle_launch_packet(
    packet_path: Path,
    *,
    repo_root: Path | None = None,
    verify_evidence: bool = True,
) -> dict[str, Any]:
    payload = validate_launch_packet(packet_path, repo_root=repo_root, verify_evidence=verify_evidence)
    lifecycle = payload.get("lifecycle", {})
    errors: list[str] = []
    for name in LIFECYCLE_GREEN_PATHS:
        row = lifecycle.get(name)
        if not isinstance(row, dict) or row.get("status") != "GREEN":
            errors.append(f"lifecycle.{name}.status must be GREEN")
    expected = {
        "provider": "vast.ai",
        "launcher": "infra.vast.parallax.guarded_launch",
        "direct_vast_create_forbidden": True,
        "h100_api_key_scope": "per_instance_only",
        "local_controller_required_after_arm": False,
    }
    for key, value in expected.items():
        if lifecycle.get(key) != value:
            errors.append(f"lifecycle.{key} must be {value!r}")
    try:
        if float(lifecycle.get("maximum_mirror_interval_seconds")) > 5.0:
            errors.append("lifecycle maximum mirror interval must be <= 5 seconds")
        if float(lifecycle.get("maximum_idle_gpu_seconds_after_durable_result")) > 5.0:
            errors.append("lifecycle durable-result teardown target must be <= 5 seconds")
        if float(lifecycle.get("maximum_bootstrap_orphan_seconds")) > 120.0:
            errors.append("lifecycle bootstrap orphan guard must be <= 120 seconds")
    except (TypeError, ValueError):
        errors.append("lifecycle timing bounds must be numeric")
    if errors:
        raise LaunchPacketError("lifecycle launch refused:\n- " + "\n- ".join(errors))
    return payload
