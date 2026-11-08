"""Fail closed on the exact PG2 interface-control source and lineage boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import sha256_file
from .ephemeral_interface_control_job import (
    CONFIG_PATH,
    LINEAGE,
    OVERLAY_IDENTITIES,
    RUN_ID,
    SOURCE_CONTRACT_PATH,
    SOURCE_CONTRACT_SHA256,
)
from .training_common import TrainingGateError


BUNDLE_RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)


def validate_closure(
    *,
    contract: dict[str, Any],
    bundle_entries: list[dict[str, Any]],
    overlays: dict[str, str],
) -> dict[str, Any]:
    bundled = {str(row.get("path", "")): str(row.get("sha256", "")) for row in bundle_entries}
    if len(bundled) != len(bundle_entries):
        raise TrainingGateError("interface source bundle paths are duplicated")
    artifact_sources = {}
    required_overlay_paths = {SOURCE_CONTRACT_PATH}
    for label, row in contract["artifacts"].items():
        path = str(row["path"])
        digest = str(row["sha256"])
        if bundled.get(path) == digest:
            artifact_sources[label] = "base_bundle"
        elif overlays.get(path) == digest:
            artifact_sources[label] = "exact_overlay"
            required_overlay_paths.add(path)
        elif path in bundled:
            raise TrainingGateError(f"interface source closure has a hash mismatch: {label}")
        elif path in overlays:
            raise TrainingGateError(f"interface source overlay has a hash mismatch: {label}")
        else:
            raise TrainingGateError(f"interface source closure is missing an artifact: {label}")
    if overlays.get(SOURCE_CONTRACT_PATH) != SOURCE_CONTRACT_SHA256:
        raise TrainingGateError("interface source contract is not an exact overlay")
    if set(overlays) != required_overlay_paths:
        raise TrainingGateError("interface overlay set is not the exact source-closure delta")
    return {
        "status": "GREEN_EXACT_BUNDLE_OVERLAY_SOURCE_CLOSURE",
        "contract_artifacts": len(artifact_sources),
        "base_bundle_artifacts": sum(value == "base_bundle" for value in artifact_sources.values()),
        "overlay_artifacts": sum(value == "exact_overlay" for value in artifact_sources.values()),
        "overlay_objects_including_contract": len(overlays),
        "artifact_sources": artifact_sources,
    }


def validate_local(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / SOURCE_CONTRACT_PATH
    if not path.is_file() or sha256_file(path) != SOURCE_CONTRACT_SHA256:
        raise TrainingGateError("interface-control source contract is absent or hash-drifted")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "FROZEN"
        or contract.get("stage") != "training_interface_control"
        or contract.get("run_id") != RUN_ID
        or contract.get("lineage") != LINEAGE
        or contract.get("role") != "supporting_boundary_first"
        or contract.get("confirmation_items_read") != 0
        or contract.get("paid_authorization") is not False
        or contract.get("destination", {}).get("persistent_system_of_record") != "amazon_s3"
        or contract.get("destination", {}).get("vast_persistent_storage") != "FORBIDDEN"
    ):
        raise TrainingGateError("interface-control source contract escaped its boundary")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 21:
        raise TrainingGateError("interface-control source artifact inventory changed")
    for label, row in artifacts.items():
        target = root / str(row.get("path", ""))
        if not target.is_file() or sha256_file(target) != row.get("sha256"):
            raise TrainingGateError(f"interface-control source hash mismatch: {label}")
    observed_overlays = {
        relative: sha256_file(root / relative) for relative in sorted(OVERLAY_IDENTITIES)
    }
    if observed_overlays != OVERLAY_IDENTITIES:
        raise TrainingGateError("interface-control overlay identities changed")
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if (
        config.get("status") != "frozen_design_before_tokenizer_preflight"
        or config.get("paid_authorization") is not False
        or config.get("lineage_order", [])[0].get("id") != "pg2"
        or config.get("staged_stop_rule", {}).get("pg1_requires_separate_freeze_after_pg2_green")
        is not True
        or config.get("population", {}).get("confirmation_reads") != 0
        or config.get("storage")
        != {
            "durable_provider": "amazon_s3",
            "aws_profile": "fuelos",
            "region": "us-east-1",
            "vast_persistent_volumes": "forbidden",
            "vast_instance_scratch": "exact_disposable_request_only",
            "laptop_model_data_container_archive_bytes": "forbidden",
        }
    ):
        raise TrainingGateError("interface-control config escaped its frozen order/storage scope")
    receipt = json.loads((root / BUNDLE_RECEIPT).read_text(encoding="utf-8"))
    closure = validate_closure(
        contract=contract,
        bundle_entries=receipt.get("entries", []),
        overlays=OVERLAY_IDENTITIES,
    )
    freeze_row = artifacts.get("post_v2_void_freeze", {})
    freeze = json.loads((root / str(freeze_row["path"])).read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "FROZEN_POST_INTERFACE_V2_PRE_SCIENCE_VOID_ALL_PAID_ACTION_OFF"
        or any(freeze.get("authorization", {}).values())
        or "automatic_paid_retry"
        not in freeze.get("forbidden", [])
        or freeze.get("state", {}).get("interface_control") != "UNANSWERED"
    ):
        raise TrainingGateError("interface-control predecessor freeze is not exact/all-paid-off")
    return {
        "status": "GREEN_EXACT_PG2_INTERFACE_CONTROL_SOURCE",
        "run_id": RUN_ID,
        "lineage": "pg2",
        "role": "supporting_boundary_first",
        "verified_artifacts": len(artifacts),
        "overlay_objects": len(OVERLAY_IDENTITIES),
        "source_closure": closure,
        "confirmation_items_read": 0,
        "paid_authorization": False,
        "durable_provider": "amazon_s3",
        "vast_persistent_storage": "FORBIDDEN",
    }
