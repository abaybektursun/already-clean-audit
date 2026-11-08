"""Strict validator for PARALLAX big-run launch packets."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


class LaunchPacketError(ValueError):
    """Raised when a launch packet does not authorize compute."""


GREEN_STATUS_PATHS = (
    "range.positive_control",
    "range.null_control",
    "range.floor_ceiling",
    "range.treatment_contrast",
    "mechanics.weight_free_tests",
    "mechanics.planted_fixtures",
    "mechanics.identity_reconstruction",
    "mechanics.real_forward_smoke",
    "mechanics.deterministic_seed_replay",
    "mechanics.interruption_resume",
    "one_cell_rehearsal.training_or_intervention",
    "one_cell_rehearsal.evaluation",
    "one_cell_rehearsal.adjudication",
    "one_cell_rehearsal.figure_stub",
    "one_cell_rehearsal.upload_checksum",
    "one_cell_rehearsal.remote_restore",
    "numerics.update_floor",
    "numerics.finite_nan_overflow",
    "analysis.confirmation_unread",
    "analysis.adjudicator_all_branches",
    "analysis.exclusion_rules_executable",
)


def canonical_sha256(payload: dict[str, Any]) -> str:
    normalized = copy.deepcopy(payload)
    normalized.setdefault("review", {}).pop("packet_sha256", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise LaunchPacketError(f"missing required field: {dotted}")
        current = current[part]
    return current


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(child, dotted)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _resolve_evidence(packet_path: Path, value: str, repo_root: Path | None) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    bases = [packet_path.parent]
    if repo_root is not None:
        bases.insert(0, repo_root)
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (bases[0] / candidate).resolve()


def validate_launch_packet(
    packet_path: Path,
    *,
    repo_root: Path | None = None,
    verify_evidence: bool = True,
) -> dict[str, Any]:
    payload = yaml.safe_load(packet_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LaunchPacketError("packet must be a YAML mapping")

    errors: list[str] = []
    for dotted, value in _walk(payload):
        if isinstance(value, str) and (
            value == "REQUIRED" or value.startswith("REQUIRED_") or "RESOLVE_AT" in value
        ):
            errors.append(f"unresolved sentinel at {dotted}: {value}")

    for dotted in GREEN_STATUS_PATHS:
        try:
            section = _get(payload, dotted)
        except LaunchPacketError as error:
            errors.append(str(error))
            continue
        if not isinstance(section, dict) or section.get("status") != "GREEN":
            errors.append(f"{dotted}.status must be GREEN")

    if payload.get("go") is not True:
        errors.append("go must be true")
    if payload.get("review", {}).get("decision") != "GO":
        errors.append("review.decision must be GO")
    if payload.get("confounds", {}).get("unresolved_count") != 0:
        errors.append("confounds.unresolved_count must be 0")

    trigger = payload.get("big_run_trigger", {})
    expected_triggers: set[str] = set()
    try:
        if float(trigger.get("expected_h100_hours", 0)) > 6:
            expected_triggers.add("h100_hours_gt_6")
        if float(trigger.get("expected_output_gb", 0)) > 100:
            expected_triggers.add("output_gb_gt_100")
        if int(trigger.get("expected_cells", 0)) > 8:
            expected_triggers.add("cells_gt_8")
    except (TypeError, ValueError):
        errors.append("big_run_trigger expected values must be numeric")
    declared = set(trigger.get("triggered_by") or [])
    if declared != expected_triggers:
        errors.append(
            "big_run_trigger.triggered_by mismatch: "
            f"expected={sorted(expected_triggers)} declared={sorted(declared)}"
        )

    projection = payload.get("resources", {}).get("projection", {})
    try:
        if float(projection.get("disk_fraction_after_stage")) >= 0.80:
            errors.append("projected disk fraction must be < 0.80")
    except (TypeError, ValueError):
        errors.append("resources.projection.disk_fraction_after_stage must be numeric")

    if verify_evidence:
        for dotted, value in _walk(payload):
            if dotted.endswith(".evidence") or dotted.endswith("_artifact_path"):
                if not isinstance(value, str):
                    errors.append(f"evidence path must be a string: {dotted}")
                    continue
                evidence = _resolve_evidence(packet_path, value, repo_root)
                if not evidence.exists():
                    errors.append(f"missing evidence for {dotted}: {value}")

    supplied_sha = payload.get("review", {}).get("packet_sha256")
    actual_sha = canonical_sha256(payload)
    if supplied_sha != actual_sha:
        errors.append(f"packet SHA mismatch: expected {actual_sha}, found {supplied_sha}")

    if errors:
        raise LaunchPacketError("launch refused:\n- " + "\n- ".join(errors))
    return payload
