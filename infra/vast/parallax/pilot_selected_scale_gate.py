"""Fail-closed audit for the one PG1 pilot permitted by optimizer selection."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


BASE_CONFIG = Path(
    "studies/lineage/cross-lineage/config/parallax_training_pilot_cell_v1.yaml"
)
SELECTED_CONFIG = Path(
    "studies/lineage/cross-lineage/config/parallax_training_pilot_cell_v2.yaml"
)
DELTA = Path(
    "studies/lineage/cross-lineage/config/parallax_training_pilot_selected_scale_delta_v1.json"
)
SOURCE_CONTRACT = Path(
    "studies/lineage/cross-lineage/config/parallax_training_pilot_source_contract_v3.json"
)
OPTIMIZER_ADJUDICATION = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_optimizer_scale_s3_v3_adjudication.json"
)

BASE_CONFIG_SHA256 = "d6987a5b82f1f2aa1c54681acbe2e8184f7298a7907356ac13288ff892eadd6c"
SELECTED_CONFIG_SHA256 = "690a4401b086cd280aca6f7d78a2e4dda82635a899cfd85aef4dd596fb34530f"
DELTA_SHA256 = "836e002f25a9606f88cdda1a10bd00d2239aba835937a0af95af4e8f742108dd"
SOURCE_CONTRACT_SHA256 = "5099d2bc91065e84fcc3ce708d860b74453c99ec67e293f8dbd6db0c39cc92de"
OPTIMIZER_ADJUDICATION_SHA256 = (
    "1c991caf81b084ce88ff5fdcbc5a80cc1bc6fdb63308006c8b658f9e59cd2880"
)
RUN_ID = "parallax-pg1-selected-scale-pilot-s3-v1"


class SelectedScaleGateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_file(root: Path, relative: Path, expected: str, *, label: str) -> Path:
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected:
        raise SelectedScaleGateError(f"{label} is absent or hash-drifted")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(row) for key, row in value.items()}
    if isinstance(value, list):
        return [_jsonable(row) for row in value]
    return value


def _leaf_diffs(left: Any, right: Any, pointer: str = "") -> dict[str, list[Any]]:
    left = _jsonable(left)
    right = _jsonable(right)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return {pointer or "/": [left, right]}
        result: dict[str, list[Any]] = {}
        for key in sorted(left):
            result.update(_leaf_diffs(left[key], right[key], f"{pointer}/{key}"))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {pointer or "/": [left, right]}
        result = {}
        for index, (before, after) in enumerate(zip(left, right)):
            result.update(_leaf_diffs(before, after, f"{pointer}/{index}"))
        return result
    return {} if left == right else {pointer or "/": [left, right]}


def validate_local(root: Path) -> dict[str, Any]:
    root = root.resolve()
    base_path = _exact_file(root, BASE_CONFIG, BASE_CONFIG_SHA256, label="base pilot config")
    selected_path = _exact_file(
        root, SELECTED_CONFIG, SELECTED_CONFIG_SHA256, label="selected pilot config"
    )
    delta_path = _exact_file(root, DELTA, DELTA_SHA256, label="selected-scale delta")
    source_path = _exact_file(
        root, SOURCE_CONTRACT, SOURCE_CONTRACT_SHA256, label="pilot source contract"
    )
    adjudication_path = _exact_file(
        root,
        OPTIMIZER_ADJUDICATION,
        OPTIMIZER_ADJUDICATION_SHA256,
        label="optimizer adjudication",
    )

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    selected = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    expected_diffs = {
        "/authored": ["2026-07-16", "2026-07-18"],
        "/freeze_id": [
            "parallax-training-pilot-cell-v1",
            "parallax-training-pilot-cell-v2-selected-scale",
        ],
        "/optimizer/learning_rate": [0.001, 0.00001],
        "/provider/forbidden": [
            ["modal", "aws", "legacy_mlops"],
            ["modal", "aws_gpu", "legacy_mlops", "vast_persistent_storage"],
        ],
        "/required_outputs/9": [
            "upload_delete_restore_receipt",
            "S3_upload_version_checksum_and_controller_HEAD_receipt",
        ],
    }
    observed_diffs = _leaf_diffs(base, selected)
    if observed_diffs != expected_diffs:
        raise SelectedScaleGateError("selected pilot contains an unregistered configuration change")
    if (
        selected.get("lineage")
        != {
            "id": "pg1_pi",
            "role": "primary",
            "checkpoint_role": "fused",
            "promotion_eligible": False,
        }
        or selected.get("optimizer", {}).get("learning_rate") != 0.00001
        or selected.get("panel", {}).get("confirmation_reads") != 0
        or selected.get("claim_boundary") != "PILOT_MECHANICAL_NOT_PROMOTION_ELIGIBLE"
    ):
        raise SelectedScaleGateError("selected pilot escapes its primary nonpromotion boundary")

    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if (
        adjudication.get("status") != "GREEN_SELECTED_SCALE_MECHANICAL_NONPROMOTION"
        or adjudication.get("adjudication", {}).get("verdict") != "GREEN_SELECTED_SCALE"
        or adjudication.get("adjudication", {}).get("selected_learning_rate") != 0.00001
        or adjudication.get("adjudication", {}).get("scientific_claim") is not False
    ):
        raise SelectedScaleGateError("optimizer adjudication does not select LR 1e-5")

    delta = json.loads(delta_path.read_text(encoding="utf-8"))
    if (
        delta.get("status") != "FROZEN_SELECTED_SCALE_DELTA"
        or delta.get("base_config", {}).get("sha256") != BASE_CONFIG_SHA256
        or delta.get("selected_config", {}).get("sha256") != SELECTED_CONFIG_SHA256
        or delta.get("selection_evidence", {}).get("sha256")
        != OPTIMIZER_ADJUDICATION_SHA256
        or delta.get("selection_evidence", {}).get("selected_learning_rate") != 0.00001
        or delta.get("observed_complete_config_diff_count") != 5
        or delta.get("observed_scientific_diff_count") != 1
        or delta.get("all_other_fields_identical") is not True
    ):
        raise SelectedScaleGateError("selected-scale delta receipt is not exact")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    destination = source.get("destination", {})
    if (
        source.get("status") != "FROZEN"
        or source.get("stage") != "training_artifact_rehearsal"
        or source.get("provider") != "vast.ai"
        or source.get("gpu") != "H100"
        or destination.get("persistent_system_of_record") != "amazon_s3"
        or destination.get("archive_key")
        != f"parallax/results/{RUN_ID}/artifacts.tar"
        or destination.get("terminal_key")
        != f"parallax/results/{RUN_ID}/terminal.json"
        or destination.get("vast_persistent_storage") != "FORBIDDEN"
        or source.get("firewalls", {}).get("scientific_runtime_network_access_after_cache_gate")
        is not False
        or source.get("firewalls", {}).get("post_science_S3_transport_only") is not True
    ):
        raise SelectedScaleGateError("pilot source contract is not exact and S3-only")
    for name, row in source.get("artifacts", {}).items():
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise SelectedScaleGateError(f"source artifact path is unsafe: {name}")
        _exact_file(root, relative, str(row.get("sha256", "")), label=f"source artifact {name}")

    return {
        "status": "GREEN_EXACT_SELECTED_SCALE_PILOT",
        "run_id": RUN_ID,
        "lineage": "pg1_pi",
        "role": "primary",
        "learning_rate": 0.00001,
        "complete_config_diffs": observed_diffs,
        "scientific_diff_count": 1,
        "source_artifact_count": len(source["artifacts"]),
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "confirmation_reads": 0,
        "promotion_eligible": False,
        "durable_provider": "amazon_s3",
        "vast_persistent_storage": "FORBIDDEN",
        "paid_authorization": False,
    }
