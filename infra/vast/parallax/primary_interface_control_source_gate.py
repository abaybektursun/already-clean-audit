"""Prove exact source closure for the primary PG1 interface H100 cell."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import sha256_file
from .primary_interface_control_job import (
    CONFIG_PATH,
    EXTERNAL_PROCESSOR_S3_VERSION_ID,
    EXTERNAL_PROCESSOR_TERMINAL_PATH,
    EXTERNAL_PROCESSOR_TERMINAL_SHA256,
    LINEAGE,
    RUN_ID,
    SOURCE_RUN_ID,
    SOURCE_CONTRACT_PATH,
)
from .training_common import TrainingGateError


BUNDLE_RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)
FREEZE_PATH = Path("studies/lineage/cross-lineage/config/parallax_execution_freeze_v111.json")


def validate_closure(
    *,
    contract: dict[str, Any],
    bundle_entries: list[dict[str, Any]],
    overlays: dict[str, str],
    source_contract_sha256: str,
) -> dict[str, Any]:
    bundled = {str(row.get("path", "")): str(row.get("sha256", "")) for row in bundle_entries}
    if len(bundled) != len(bundle_entries):
        raise TrainingGateError("primary source bundle paths are duplicated")
    artifact_sources: dict[str, str] = {}
    required_overlays = {SOURCE_CONTRACT_PATH}
    for label, row in contract["artifacts"].items():
        path = str(row["path"])
        digest = str(row["sha256"])
        if bundled.get(path) == digest:
            artifact_sources[label] = "base_bundle"
        elif path in bundled:
            raise TrainingGateError(f"primary source bundle has a stale conflicting path: {label}")
        elif overlays.get(path) == digest:
            artifact_sources[label] = "exact_overlay"
            required_overlays.add(path)
        elif path in overlays:
            raise TrainingGateError(f"primary source overlay has a hash mismatch: {label}")
        else:
            raise TrainingGateError(f"primary source closure is missing an artifact: {label}")
    if overlays.get(SOURCE_CONTRACT_PATH) != source_contract_sha256:
        raise TrainingGateError("primary source contract is not an exact overlay")
    if set(overlays) != required_overlays:
        raise TrainingGateError("primary overlay set is not the exact bundle delta")
    return {
        "status": "GREEN_EXACT_PRIMARY_BUNDLE_OVERLAY_SOURCE_CLOSURE",
        "contract_artifacts": len(artifact_sources),
        "base_bundle_artifacts": sum(value == "base_bundle" for value in artifact_sources.values()),
        "overlay_artifacts": sum(value == "exact_overlay" for value in artifact_sources.values()),
        "overlay_objects_including_contract": len(overlays),
        "artifact_sources": artifact_sources,
    }


def expected_overlay_identities(root: Path, contract: dict[str, Any]) -> dict[str, str]:
    receipt = json.loads((root / BUNDLE_RECEIPT).read_text(encoding="utf-8"))
    entries = receipt.get("entries")
    if (
        receipt.get("status") != "GREEN"
        or receipt.get("confirmation_image_included") is not False
        or receipt.get("confirmation_manifest_included") is not False
        or not isinstance(entries, list)
        or len(entries) != 155
    ):
        raise TrainingGateError("primary source bundle receipt is not exact/no-confirmation")
    bundled = {str(row["path"]): str(row["sha256"]) for row in entries}
    overlays: dict[str, str] = {}
    for row in contract["artifacts"].values():
        path = str(row["path"])
        digest = str(row["sha256"])
        if bundled.get(path) == digest:
            continue
        if path in bundled:
            raise TrainingGateError(f"primary source cannot replace stale bundled path: {path}")
        overlays[path] = digest
    overlays[SOURCE_CONTRACT_PATH] = sha256_file(root / SOURCE_CONTRACT_PATH)
    return overlays


def validate_local(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / SOURCE_CONTRACT_PATH
    if not path.is_file():
        raise TrainingGateError("primary interface source contract is absent")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "FROZEN"
        or contract.get("stage") != "training_interface_control"
        or contract.get("run_id") != SOURCE_RUN_ID
        or contract.get("lineage") != LINEAGE
        or contract.get("role")
        != "primary_within_lineage_discovery_independent_of_supporting_pg2"
        or contract.get("confirmation_items_read") != 0
        or contract.get("paid_authorization") is not False
        or contract.get("destination", {}).get("persistent_system_of_record") != "amazon_s3"
        or contract.get("destination", {}).get("vast_persistent_storage") != "FORBIDDEN"
        or contract.get("lineage_boundary", {}).get("supporting_pg2_dependency")
        != "none_within_lineage"
        or contract.get("lineage_boundary", {}).get("single_pg1_pg2_causal_chain") is not False
    ):
        raise TrainingGateError("primary interface source contract escaped its boundary")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 26:
        raise TrainingGateError("primary interface source artifact inventory changed")
    for label, row in artifacts.items():
        target = root / str(row.get("path", ""))
        if not target.is_file() or sha256_file(target) != row.get("sha256"):
            raise TrainingGateError(f"primary interface source hash mismatch: {label}")
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if (
        config.get("status") != "frozen_design_before_tokenizer_preflight"
        or config.get("paid_authorization") is not False
        or [row.get("id") for row in config.get("lineage_order", [])] != [LINEAGE]
        or config.get("primary_within_lineage", {}).get("supporting_pg2_dependency")
        != "none_within_lineage"
        or config.get("population", {}).get("confirmation_reads") != 0
        or config.get("storage", {}).get("durable_provider") != "amazon_s3"
        or config.get("storage", {}).get("vast_persistent_volumes") != "forbidden"
    ):
        raise TrainingGateError("primary interface config escaped its order/storage boundary")
    terminal_path = root / EXTERNAL_PROCESSOR_TERMINAL_PATH
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    if (
        sha256_file(terminal_path) != EXTERNAL_PROCESSOR_TERMINAL_SHA256
        or terminal.get("status") != "GREEN_JOB_TERMINAL"
        or payload.get("status") != "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT"
        or payload.get("h100_instances_created") != 0
        or payload.get("model_weight_urls") != 0
        or payload.get("model_loads") != 0
        or payload.get("gpu_operations") != 0
        or payload.get("confirmation_items_read") != 0
        or contract.get("external_processor_gate", {}).get("terminal_s3_version_id")
        != EXTERNAL_PROCESSOR_S3_VERSION_ID
    ):
        raise TrainingGateError("primary external processor evidence is not exact green")
    freeze = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_POST_PRIMARY_PG1_PROCESSOR_GREEN_ALL_PAID_ACTION_OFF"
        or any(freeze.get("authorization", {}).values())
        or freeze.get("processor_gate", {}).get("h100_instances_created") != 0
        or freeze.get("processor_gate", {}).get("terminal", {}).get("s3_version_id")
        != EXTERNAL_PROCESSOR_S3_VERSION_ID
    ):
        raise TrainingGateError("post-processor predecessor freeze is not exact/all-paid-off")
    overlays = expected_overlay_identities(root, contract)
    receipt = json.loads((root / BUNDLE_RECEIPT).read_text(encoding="utf-8"))
    closure = validate_closure(
        contract=contract,
        bundle_entries=receipt["entries"],
        overlays=overlays,
        source_contract_sha256=sha256_file(path),
    )
    return {
        "status": "GREEN_EXACT_PRIMARY_PG1_INTERFACE_SOURCE",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "lineage": LINEAGE,
        "role": contract["role"],
        "verified_artifacts": len(artifacts),
        "overlay_objects": len(overlays),
        "overlay_identities": overlays,
        "source_contract_sha256": sha256_file(path),
        "source_closure": closure,
        "external_processor_s3_version_id": EXTERNAL_PROCESSOR_S3_VERSION_ID,
        "h100_instances_created_by_processor_gate": 0,
        "confirmation_items_read": 0,
        "paid_authorization": False,
        "durable_provider": "amazon_s3",
        "vast_persistent_storage": "FORBIDDEN",
    }
