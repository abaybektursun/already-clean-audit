"""Post-authority exact-body hydration gate for PG1 native production.

Preflight never calls this module.  Production calls it after one-shot owner
admission and before any ML import or ``SCIENCE_STARTED`` transition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .pg1_execution_capsule import CapsuleError, safe_relative, validate_capsule


def _model_bodies(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    contract = capsule["contract"]
    paths = contract["canonical_paths"]
    science = contract["science_inputs"]
    manifest_path = Path(paths["model_manifest"])
    if sha256_file(manifest_path) != science["model"]["manifest_sha256"]:
        raise CapsuleError("model hydration manifest bytes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("status") != "FROZEN" or not isinstance(files, list) or not files:
        raise CapsuleError("model hydration manifest is not exact frozen metadata")
    snapshot = (
        Path(paths["cache_root"])
        / ("models--" + science["model"]["repo"].replace("/", "--"))
        / "snapshots"
        / science["model"]["revision"]
    )
    receipts: list[dict[str, Any]] = []
    names: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            raise CapsuleError("model hydration row is not an object")
        relative = safe_relative(row.get("path"), label="model hydration file")
        target = snapshot / relative
        expected_bytes = row.get("bytes")
        expected_sha256 = row.get("sha256")
        if (
            not target.is_file()
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or target.stat().st_size != expected_bytes
            or not isinstance(expected_sha256, str)
            or sha256_file(target) != expected_sha256
        ):
            raise CapsuleError(f"model/tokenizer body mismatch: {relative}")
        names.add(str(relative))
        receipts.append(
            {"path": str(target), "bytes": expected_bytes, "sha256": expected_sha256}
        )
    required_tokenizer_assets = {
        "preprocessor_config.json", "tokenizer.json", "tokenizer.model"
    }
    if not required_tokenizer_assets.issubset(names):
        raise CapsuleError("frozen tokenizer assets are incomplete")
    return receipts


def _task_bodies(capsule: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    from .training_common import TrainingGateError
    from .training_model import select_split_stratified

    contract = capsule["contract"]
    paths = contract["canonical_paths"]
    science = contract["science_inputs"]
    manifest_path = Path(paths["discovery_manifest"])
    if sha256_file(manifest_path) != science["discovery_manifest"]["sha256"]:
        raise CapsuleError("discovery task manifest bytes changed")
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("split") not in {"train", "discovery_eval"}
                or not isinstance(row.get("item_id"), str)
                or not row["item_id"]
                or not isinstance(row.get("task_id"), str)
                or not row["task_id"]
                or not isinstance(row.get("target"), str)
                or not row["target"]
            ):
                raise CapsuleError(f"discovery task row is not exact at line {line_number}")
            image = row.get("image")
            if image is None:
                if "image_sha256" in row or "image_bytes" in row:
                    raise CapsuleError(
                        f"image-less task row has extra image identity at line {line_number}"
                    )
            elif isinstance(image, str) and image:
                safe_relative(image, label="discovery image")
                if (
                    not isinstance(row.get("image_sha256"), str)
                    or len(row["image_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in row["image_sha256"]
                    )
                    or not isinstance(row.get("image_bytes"), int)
                    or row["image_bytes"] <= 0
                ):
                    raise CapsuleError(
                        f"visual task row image identity changed at line {line_number}"
                    )
            else:
                raise CapsuleError(f"task row image semantics changed at line {line_number}")
            rows.append(row)
    config_path = Path(paths["primary_config"])
    if sha256_file(config_path) != science["primary_config"]["sha256"]:
        raise CapsuleError("capsule-bound task selection config bytes changed")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    population = config.get("population", {})
    task_id = str(population.get("source_task_id"))
    task_rows = [row for row in rows if row["task_id"] == task_id]
    try:
        selected = [
            *select_split_stratified(
                task_rows,
                task_id=task_id,
                split=str(population.get("train_split")),
                per_target=int(population.get("train_per_semantic_target")),
            ),
            *select_split_stratified(
                task_rows,
                task_id=task_id,
                split=str(population.get("discovery_split")),
                per_target=int(population.get("discovery_per_semantic_target")),
            ),
        ]
    except (TrainingGateError, TypeError, ValueError) as error:
        raise CapsuleError("capsule-bound exact task selection failed") from error
    if len({row["item_id"] for row in selected}) != len(selected):
        raise CapsuleError("capsule-bound selected item identity is duplicated")
    receipts: list[dict[str, Any]] = []
    task_root = Path(paths["task_root"])
    for row in selected:
        if row.get("image") is None:
            receipts.append(
                {
                    "item_id": row["item_id"],
                    "task_id": row["task_id"],
                    "split": row["split"],
                    "body_semantics": "IMAGELESS_ITEM_MANIFEST_HASH_VERIFIED",
                    "image": None,
                }
            )
            continue
        relative = safe_relative(row["image"], label="selected discovery image")
        target = task_root / relative
        if (
            not target.is_file()
            or target.stat().st_size != row["image_bytes"]
            or sha256_file(target) != row["image_sha256"]
        ):
            raise CapsuleError(f"selected task image body mismatch: {row['item_id']}")
        receipts.append(
            {
                "item_id": row["item_id"],
                "task_id": row["task_id"],
                "split": row["split"],
                "body_semantics": "SELECTED_VISUAL_IMAGE_BODY_VERIFIED",
                "path": str(target),
                "bytes": row["image_bytes"],
                "sha256": row["image_sha256"],
            }
        )
    return [str(row["item_id"]) for row in selected], receipts


def verify_runtime_hydration(*, capsule: dict[str, Any]) -> dict[str, Any]:
    validate_capsule(capsule)
    evidence_path = Path(capsule["contract"]["canonical_paths"]["hydration_evidence"])
    if evidence_path.exists():
        raise CapsuleError("refusing to reuse runtime hydration evidence")
    tokenizer_evidence = Path(
        capsule["contract"]["canonical_paths"]["tokenizer_evidence"]
    )
    if not tokenizer_evidence.is_file():
        raise CapsuleError("processor tokenizer evidence is absent")
    tokenizer = json.loads(tokenizer_evidence.read_text(encoding="utf-8"))
    if (
        tokenizer.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or tokenizer.get("lineage") != capsule["contract"]["science_inputs"]["lineage"]
        or tokenizer.get("model_loads") != 0
        or tokenizer.get("gpu_operations") != 0
        or tokenizer.get("confirmation_items_read") != 0
    ):
        raise CapsuleError("processor tokenizer evidence changed before hydration")
    model_files = _model_bodies(capsule)
    item_ids, task_images = _task_bodies(capsule)
    evidence = {
        "schema": "pg1-native-runtime-hydration-evidence/v1",
        "status": "GREEN_EXACT_RUNTIME_BODIES_VERIFIED_BEFORE_SCIENCE",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "model_files": model_files,
        "tokenizer_asset_count": sum(
            row["path"].endswith(("tokenizer.json", "tokenizer.model", "preprocessor_config.json"))
            for row in model_files
        ),
        "selected_item_ids": item_ids,
        "task_images": task_images,
        "visual_body_count": sum(
            row["body_semantics"] == "SELECTED_VISUAL_IMAGE_BODY_VERIFIED"
            for row in task_images
        ),
        "imageless_item_count": sum(
            row["body_semantics"] == "IMAGELESS_ITEM_MANIFEST_HASH_VERIFIED"
            for row in task_images
        ),
        "tokenizer_evidence": {
            "path": str(tokenizer_evidence),
            "sha256": sha256_file(tokenizer_evidence),
        },
        "science_started": False,
        "confirmation_items_read": 0,
    }
    atomic_json(evidence_path, evidence)
    return evidence
