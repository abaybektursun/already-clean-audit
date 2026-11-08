"""Run one frozen fused-base range cell and publish its archive to S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:  # Staged as standalone siblings on the worker.
    import ephemeral_optimizer_scale_job as base
    import s3_post_upload as upload
except ModuleNotFoundError:  # Imported through the repository package in planted tests.
    from . import ephemeral_optimizer_scale_job as base
    from . import s3_post_upload as upload


RUN_ID = "parallax-pg1-selected-scale-pilot-s3-v1"
PG2_RUN_ID = "parallax-pg2-seed17-range-s3-v1"
PG2_V2_RUN_ID = "parallax-pg2-seed17-range-s3-v2"
SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/parallax_training_pilot_source_contract_v3.json"
)
SOURCE_CONTRACT_SHA256 = "5099d2bc91065e84fcc3ce708d860b74453c99ec67e293f8dbd6db0c39cc92de"
CONFIG_PATH = "studies/lineage/cross-lineage/config/parallax_training_pilot_cell_v2.yaml"
CONFIG_SHA256 = "690a4401b086cd280aca6f7d78a2e4dda82635a899cfd85aef4dd596fb34530f"
FREEZE_PATH = "studies/lineage/cross-lineage/config/parallax_execution_freeze_v81.json"
FREEZE_SHA256 = "a8b35663b89b837d1a813adb3292ab72c374c1757c19cf90506b743b7c6248c9"
DELTA_PATH = (
    "studies/lineage/cross-lineage/config/parallax_training_pilot_selected_scale_delta_v1.json"
)
DELTA_SHA256 = "836e002f25a9606f88cdda1a10bd00d2239aba835937a0af95af4e8f742108dd"
OVERLAY_IDENTITIES = {
    SOURCE_CONTRACT_PATH: SOURCE_CONTRACT_SHA256,
    CONFIG_PATH: CONFIG_SHA256,
    FREEZE_PATH: FREEZE_SHA256,
    DELTA_PATH: DELTA_SHA256,
}
PG2_SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_range_source_contract_pg2_seed17_v1.json"
)
PG2_SOURCE_CONTRACT_SHA256 = "ca418820257f07c1fa6b0f58e7f497f4535011536f5993fc959c21e4281772e4"
PG2_V2_SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_range_source_contract_pg2_seed17_v2.json"
)
PG2_V2_SOURCE_CONTRACT_SHA256 = "b48d08aade0654860b754a55d38357042f0461434ea75870b53e7d60f02d492e"
PG2_CONFIG_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_range_screen_pg2_seed17_v1.yaml"
)
PG2_CONFIG_SHA256 = "435a57522066854486302617a0b3d5eb2d2d68bf597d57f9627cb0d02d1775b4"
PG2_FREEZE_PATH = "studies/lineage/cross-lineage/config/parallax_execution_freeze_v94.json"
PG2_FREEZE_SHA256 = "b06a673be72a86680a5e03347b36f2aa52c5658ea3cc6e2ba8ae51688f77e2f7"
PG2_V2_FREEZE_PATH = "studies/lineage/cross-lineage/config/parallax_execution_freeze_v96.json"
PG2_V2_FREEZE_SHA256 = "a53d585ed9ff43c7c8d01cce6d91f5c52c6f1282684ae48b4bd1bd9b00ca7d0a"
PG2_PARITY_PATH = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_seed17_range_parity_v1.json"
)
PG2_PARITY_SHA256 = "e57c57ebc90456f8ac7cb5b8eb829e4a1c09e78c2e20ad6e477bb1be1e2b127f"
PG2_OPTIMIZER_PATH = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_optimizer_scale_s3_v3_adjudication.json"
)
PG2_OPTIMIZER_SHA256 = "97e4e17140acd7d9b4b7ac4dd96ff5f51888ce561c70dd5fe987270d89d16a1d"
PG2_MODEL_MANIFEST_PATH = (
    "studies/lineage/cross-lineage/config/parallax_pg2_h100_s3_hydration_job_v1.json"
)
PG2_MODEL_MANIFEST_SHA256 = "709b84e2b0276a3afcd9c9265e6b1f8e54d6f24f845d0b696cfe2285160e893a"
PILOT_CELL_PATH = "infra/vast/parallax/training_pilot_cell.py"
PILOT_CELL_SHA256 = "63fe9c3c732bb131e8da5a3da36d6cdfed1650443f932130d04169ac8d59860b"
PILOT_PREFLIGHT_PATH = "infra/vast/parallax/training_pilot_preflight.py"
PILOT_PREFLIGHT_SHA256 = "e4ef0a4aac0c5252c444ef2d6ffc22cf97f4813fbbcb9d3835840b885a717f34"
PG2_OVERLAY_IDENTITIES = {
    PG2_SOURCE_CONTRACT_PATH: PG2_SOURCE_CONTRACT_SHA256,
    PG2_CONFIG_PATH: PG2_CONFIG_SHA256,
    PG2_FREEZE_PATH: PG2_FREEZE_SHA256,
    PG2_PARITY_PATH: PG2_PARITY_SHA256,
    PG2_OPTIMIZER_PATH: PG2_OPTIMIZER_SHA256,
    PG2_MODEL_MANIFEST_PATH: PG2_MODEL_MANIFEST_SHA256,
    PILOT_CELL_PATH: PILOT_CELL_SHA256,
    PILOT_PREFLIGHT_PATH: PILOT_PREFLIGHT_SHA256,
}
PG2_V2_OVERLAY_IDENTITIES = {
    PG2_V2_SOURCE_CONTRACT_PATH: PG2_V2_SOURCE_CONTRACT_SHA256,
    PG2_CONFIG_PATH: PG2_CONFIG_SHA256,
    PG2_V2_FREEZE_PATH: PG2_V2_FREEZE_SHA256,
    PG2_PARITY_PATH: PG2_PARITY_SHA256,
    PG2_OPTIMIZER_PATH: PG2_OPTIMIZER_SHA256,
    PG2_MODEL_MANIFEST_PATH: PG2_MODEL_MANIFEST_SHA256,
    PILOT_CELL_PATH: PILOT_CELL_SHA256,
    PILOT_PREFLIGHT_PATH: PILOT_PREFLIGHT_SHA256,
}
PG2_REPLACED_BUNDLE_IDENTITIES = {
    PILOT_CELL_PATH: "f76cdb4973dfe628accf28b1b2181600cbe13bd75ddfb226fc580d1cb4a3d927",
    PILOT_PREFLIGHT_PATH: "66a1ea1153d42613d0d99fa3fa4c332ba875951b54bd7f8a28f3cbb33fbc47d8",
}
PG2_MODEL_NAMES = {
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
PG2_MODEL_BYTES = 6_099_504_596
RUN_PROFILES = {
    RUN_ID: {
        "source_contract_path": SOURCE_CONTRACT_PATH,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "config_path": CONFIG_PATH,
        "overlay_identities": OVERLAY_IDENTITIES,
        "replacement_identities": {},
        "source_artifacts": 35,
        "model_files": base.EXPECTED_MODEL_FILES,
        "model_names": base.EXPECTED_MODEL_NAMES,
        "model_bytes": base.EXPECTED_MODEL_BYTES,
        "lineage": "pg1_pi",
        "role": "primary",
        "lineage_firewall": "primary_lineage_only",
        "scratch_gb": 80,
        "scientific_config_diff_count": 1,
    },
    PG2_RUN_ID: {
        "source_contract_path": PG2_SOURCE_CONTRACT_PATH,
        "source_contract_sha256": PG2_SOURCE_CONTRACT_SHA256,
        "config_path": PG2_CONFIG_PATH,
        "overlay_identities": PG2_OVERLAY_IDENTITIES,
        "replacement_identities": PG2_REPLACED_BUNDLE_IDENTITIES,
        "source_artifacts": 24,
        "model_files": 9,
        "model_names": PG2_MODEL_NAMES,
        "model_bytes": PG2_MODEL_BYTES,
        "lineage": "pg2",
        "role": "supporting",
        "lineage_firewall": "supporting_lineage_only",
        "scratch_gb": 80,
        "scientific_config_diff_count": 0,
    },
    PG2_V2_RUN_ID: {
        "source_contract_path": PG2_V2_SOURCE_CONTRACT_PATH,
        "source_contract_sha256": PG2_V2_SOURCE_CONTRACT_SHA256,
        "config_path": PG2_CONFIG_PATH,
        "overlay_identities": PG2_V2_OVERLAY_IDENTITIES,
        "replacement_identities": PG2_REPLACED_BUNDLE_IDENTITIES,
        "source_artifacts": 24,
        "model_files": 9,
        "model_names": PG2_MODEL_NAMES,
        "model_bytes": PG2_MODEL_BYTES,
        "lineage": "pg2",
        "role": "supporting",
        "lineage_firewall": "supporting_lineage_only",
        "scratch_gb": 80,
        "scientific_config_diff_count": 0,
    },
}
EXPECTED_SNAPSHOTS = [0, 1, 2, 4, 8, 16]


class PilotJobError(RuntimeError):
    pass


def _absolute(value: object, *, prefix: str, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not str(path).startswith(prefix):
        raise PilotJobError(f"{label} escapes {prefix}")
    return path


def _validate_bundle(bundle: dict[str, Any]) -> None:
    base._validate_object(bundle, kind="bundle")
    if (
        int(bundle.get("bytes", 0)) != base.EXPECTED_BUNDLE_BYTES
        or bundle.get("sha256") != base.EXPECTED_BUNDLE_SHA256
    ):
        raise PilotJobError("pilot source bundle identity changed")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != 155:
        raise PilotJobError("pilot source bundle receipt must bind 155 files")
    paths: set[str] = set()
    for row in entries:
        relative = PurePosixPath(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PilotJobError("pilot bundle receipt contains an unsafe path")
        if str(relative) in paths:
            raise PilotJobError("pilot bundle receipt contains a duplicate path")
        paths.add(str(relative))
        base._digest(row.get("sha256"), label="pilot bundle entry digest")
        if not isinstance(row.get("bytes"), int) or int(row["bytes"]) < 0:
            raise PilotJobError("pilot bundle receipt contains invalid bytes")
    task_prefix = base.EXPECTED_TASK_ROOT + "/"
    if (
        task_prefix + "manifests/items.jsonl" in paths
        or task_prefix + "manifests/confirmation.sha256" in paths
        or any(path.startswith(task_prefix) and "/confirmation_eval/" in path for path in paths)
    ):
        raise PilotJobError("pilot bundle contains a confirmation-capable artifact")
    discovery = [
        path
        for path in paths
        if path.startswith(task_prefix + "images/") and "/discovery_eval/" in path
    ]
    train = [
        path
        for path in paths
        if path.startswith(task_prefix + "images/") and "/train/" in path
    ]
    if len(discovery) != 24 or len(train) != 96:
        raise PilotJobError("pilot task image inventory changed")


def _validate_overlays(
    rows: object, *, identities: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(identities):
        raise PilotJobError("range cell must bind every exact source overlay")
    observed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PilotJobError("pilot overlay is not an object")
        base._validate_object(row, kind="overlay")
        path = str(row.get("path", ""))
        relative = PurePosixPath(path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PilotJobError("pilot overlay destination is unsafe")
        if path in observed:
            raise PilotJobError("pilot overlay destination is duplicated")
        observed[path] = str(row["sha256"])
    if observed != identities:
        raise PilotJobError("range-cell overlay identities changed")
    return rows


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    run_id = str(contract.get("run_id", ""))
    profile = RUN_PROFILES.get(run_id)
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "FROZEN_H100_SELECTED_PILOT_JOB"
        or contract.get("stage") != "training_artifact_rehearsal"
        or contract.get("provider") != "vast.ai"
        or profile is None
    ):
        raise PilotJobError("range-cell job contract is not exact frozen schema v1")
    instance_id = contract.get("instance_id")
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise PilotJobError("pilot job instance ID is absent")
    if contract.get("storage") != {
        "durable_provider": "amazon_s3",
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": profile["scratch_gb"],
        "aws_credentials_on_worker": False,
    }:
        raise PilotJobError("range-cell storage contract is not exact S3-only scratch")

    model = contract.get("model")
    if not isinstance(model, list) or len(model) != profile["model_files"]:
        raise PilotJobError("range cell must bind every model file")
    names: set[str] = set()
    for row in model:
        base._validate_object(row, kind="model")
        relative = PurePosixPath(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise PilotJobError("pilot model filename escapes the snapshot root")
        names.add(str(relative))
    if (
        names != profile["model_names"]
        or sum(int(row["bytes"]) for row in model) != profile["model_bytes"]
    ):
        raise PilotJobError("range-cell model inventory changed")
    _validate_bundle(contract.get("bundle", {}))
    _validate_overlays(
        contract.get("overlays"), identities=profile["overlay_identities"]
    )

    paths = contract.get("paths", {})
    _absolute(paths.get("runtime_root"), prefix="/workspace/parallax", label="runtime root")
    _absolute(paths.get("source_root"), prefix="/workspace/parallax/", label="source root")
    _absolute(paths.get("bundle_path"), prefix="/workspace/parallax/", label="bundle path")
    _absolute(paths.get("snapshot_root"), prefix="/root/.cache/huggingface/hub/", label="snapshot root")
    _absolute(paths.get("preflight_output"), prefix="/workspace/parallax/", label="preflight output")
    _absolute(paths.get("pilot_output_dir"), prefix="/workspace/parallax/", label="pilot output")
    _absolute(paths.get("figure_output"), prefix="/workspace/parallax/", label="figure output")
    _absolute(paths.get("manifest_output"), prefix="/workspace/parallax/", label="manifest output")
    _absolute(paths.get("archive_output"), prefix="/workspace/parallax/", label="archive output")
    _absolute(paths.get("final_output"), prefix="/workspace/parallax/", label="final output")

    if contract.get("source_contract_sha256") != profile["source_contract_sha256"]:
        raise PilotJobError("range-cell source contract hash changed")
    if contract.get("bounds") != {
        "parallel_downloads": 3,
        "download_timeout_seconds": 600,
        "preflight_timeout_seconds": 180,
        "pilot_timeout_seconds": 600,
        "figure_timeout_seconds": 30,
        "artifact_upload_timeout_seconds": 300,
        "minimum_free_bytes": 50_000_000_000,
    }:
        raise PilotJobError("pilot execution bounds changed")
    if contract.get("firewall") != {
        profile["lineage_firewall"]: True,
        "confirmation_manifest_present": False,
        "confirmation_images_present": False,
        "train_items": 96,
        "discovery_items": 24,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
        "automatic_retry": False,
    }:
        raise PilotJobError("range-cell data/science firewall changed")
    return contract


def _download_overlays(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    profile = RUN_PROFILES[str(contract["run_id"])]
    replacements = profile["replacement_identities"]
    records = []
    for row in sorted(contract["overlays"], key=lambda value: str(value["path"])):
        destination = source / str(row["path"])
        old_digest = replacements.get(str(row["path"]))
        if old_digest is None:
            if destination.exists():
                raise PilotJobError(f"overlay would replace a bundled file: {row['path']}")
            records.append(base._download(row, destination))
            continue
        if not destination.is_file() or base.sha256_file(destination) != old_digest:
            raise PilotJobError(f"registered bundled replacement base changed: {row['path']}")
        replacement = destination.with_name(f".{destination.name}.registered-overlay")
        record = base._download(row, replacement)
        if base.sha256_file(destination) != old_digest:
            replacement.unlink(missing_ok=True)
            raise PilotJobError(f"bundled replacement changed during download: {row['path']}")
        os.replace(replacement, destination)
        record["path"] = str(destination)
        record["registered_replacement"] = True
        record["replaced_sha256"] = old_digest
        records.append(record)
    return {
        "status": "GREEN_EXACT_VERSIONED_OVERLAYS",
        "objects": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "records": records,
    }


def _verify_source(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    profile = RUN_PROFILES[str(contract["run_id"])]
    source_contract_path = source / str(profile["source_contract_path"])
    if base.sha256_file(source_contract_path) != profile["source_contract_sha256"]:
        raise PilotJobError("selected range-cell source contract is not exact")
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    if (
        source_contract.get("status") != "FROZEN"
        or source_contract.get("stage") != "training_artifact_rehearsal"
        or source_contract.get("destination", {}).get("persistent_system_of_record")
        != "amazon_s3"
        or source_contract.get("destination", {}).get("vast_persistent_storage")
        != "FORBIDDEN"
        or source_contract.get("destination", {}).get("archive_key")
        != f"parallax/results/{contract['run_id']}/artifacts.tar"
    ):
        raise PilotJobError("selected range source contract escapes its S3-only boundary")
    verified = 0
    for name, row in source_contract.get("artifacts", {}).items():
        relative = PurePosixPath(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PilotJobError(f"selected source artifact path is unsafe: {name}")
        path = source / str(relative)
        if not path.is_file() or base.sha256_file(path) != row.get("sha256"):
            raise PilotJobError(f"selected source artifact hash mismatch: {name}")
        verified += 1
    if verified != profile["source_artifacts"]:
        raise PilotJobError("selected source artifact count changed")
    return {
        "status": "GREEN_EXACT_SELECTED_PILOT_SOURCE",
        "source_contract_sha256": profile["source_contract_sha256"],
        "verified_artifacts": verified,
        "learning_rate": 0.00001,
        "scientific_config_diff_count": profile["scientific_config_diff_count"],
    }


def _run_science(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    profile = RUN_PROFILES[str(contract["run_id"])]
    cache_root = Path(contract["paths"]["snapshot_root"]).parents[2]
    task_root = source / base.EXPECTED_TASK_ROOT
    source_contract = source / str(profile["source_contract_path"])
    environment = base._runtime_environment(source)
    preflight_output = Path(contract["paths"]["preflight_output"])
    pilot_output = Path(contract["paths"]["pilot_output_dir"])
    figure_output = Path(contract["paths"]["figure_output"])

    preflight_command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_pilot_preflight",
        "--root",
        str(source),
        "--source-contract",
        str(source_contract),
        "--cache-root",
        str(cache_root),
        "--task-root",
        str(task_root),
        "--instance-id",
        str(contract["instance_id"]),
        "--output",
        str(preflight_output),
    ]
    preflight_run = base._run_command(
        preflight_command,
        cwd=source,
        environment=environment,
        timeout=int(contract["bounds"]["preflight_timeout_seconds"]),
        label="selected-pilot H100 preflight",
    )
    preflight = json.loads(preflight_output.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "GREEN"
        or int(preflight.get("science_items_evaluated", -1)) != 0
        or int(preflight.get("selected_image_count", -1)) != 120
        or int(preflight.get("confirmation_items_read", -1)) != 0
    ):
        raise PilotJobError("selected-pilot preflight is not green/no-science")

    pilot_command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_pilot_cell",
        "--root",
        str(source),
        "--source-contract",
        str(source_contract),
        "--matrix",
        str(source / "studies/lineage/cross-lineage/config/parallax_matrix.yaml"),
        "--config",
        str(source / str(profile["config_path"])),
        "--cache-root",
        str(cache_root),
        "--task-root",
        str(task_root),
        "--instance-id",
        str(contract["instance_id"]),
        "--output-dir",
        str(pilot_output),
    ]
    pilot_run = base._run_command(
        pilot_command,
        cwd=source,
        environment=environment,
        timeout=int(contract["bounds"]["pilot_timeout_seconds"]),
        label=f"selected-scale {profile['lineage']} range cell",
    )
    pilot_json = pilot_output / "pilot.json"
    pilot = json.loads(pilot_json.read_text(encoding="utf-8"))
    if (
        pilot.get("status") != "GREEN_MECHANICAL_EXECUTION"
        or pilot.get("lineage") != profile["lineage"]
        or pilot.get("role") != profile["role"]
        or pilot.get("label") != "PILOT_MECHANICAL_NOT_PROMOTION_ELIGIBLE"
        or int(pilot.get("confirmation_items_read", -1)) != 0
        or pilot.get("frozen_parameter_bytes_exact") is not True
    ):
        raise PilotJobError("selected-scale pilot escaped its mechanical boundary")
    if [int(row["step"]) for row in pilot.get("snapshots", [])] != EXPECTED_SNAPSHOTS:
        raise PilotJobError("selected-scale pilot snapshot schedule changed")

    figure_command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_pilot_figure",
        "--result",
        str(pilot_json),
        "--output",
        str(figure_output),
    ]
    figure_run = base._run_command(
        figure_command,
        cwd=source,
        environment=environment,
        timeout=int(contract["bounds"]["figure_timeout_seconds"]),
        label="selected-pilot deterministic figure",
    )
    return {
        "preflight": preflight,
        "preflight_sha256": base.sha256_file(preflight_output),
        "preflight_run": preflight_run,
        "pilot": pilot,
        "pilot_sha256": base.sha256_file(pilot_json),
        "pilot_run": pilot_run,
        "figure_sha256": base.sha256_file(figure_output),
        "figure_run": figure_run,
        "runtime_versions_observed": base._package_versions(),
    }


def _artifact_rows(contract: dict[str, Any], execution: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = Path(contract["paths"]["preflight_output"])
    pilot_dir = Path(contract["paths"]["pilot_output_dir"])
    figure = Path(contract["paths"]["figure_output"])
    pilot = execution["pilot"]
    rows = [
        {"archive_path": "preflight.json", "path": preflight},
        {"archive_path": "pilot/pilot.json", "path": pilot_dir / "pilot.json"},
        {"archive_path": "pilot/pilot.svg", "path": figure},
    ]
    checkpoint_steps = []
    for snapshot in pilot["snapshots"]:
        step = int(snapshot["step"])
        checkpoint_steps.append(step)
        checkpoint = Path(str(snapshot["checkpoint"]["path"]))
        if checkpoint.parent != pilot_dir or checkpoint.name != f"step-{step:06d}.pt":
            raise PilotJobError("pilot checkpoint path escaped the exact output directory")
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != int(snapshot["checkpoint"]["bytes"])
            or base.sha256_file(checkpoint) != snapshot["checkpoint"]["sha256"]
        ):
            raise PilotJobError(f"pilot checkpoint proof mismatch at step {step}")
        rows.append({"archive_path": f"pilot/{checkpoint.name}", "path": checkpoint})
    if checkpoint_steps != EXPECTED_SNAPSHOTS:
        raise PilotJobError("pilot archive does not contain the exact six checkpoints")
    manifest_rows = []
    for row in sorted(rows, key=lambda value: value["archive_path"]):
        path = row["path"]
        if not path.is_file():
            raise PilotJobError(f"pilot artifact is absent: {row['archive_path']}")
        manifest_rows.append(
            {
                "path": row["archive_path"],
                "bytes": path.stat().st_size,
                "sha256": base.sha256_file(path),
            }
        )
    return manifest_rows


def _build_archive(contract: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(contract["paths"]["manifest_output"])
    archive_path = Path(contract["paths"]["archive_output"])
    if manifest_path.exists() or archive_path.exists():
        raise PilotJobError("refusing to overwrite a pilot manifest or archive")
    rows = _artifact_rows(contract, execution)
    manifest = {
        "schema_version": 1,
        "status": "GREEN_EXACT_PILOT_ARTIFACT_MANIFEST",
        "run_id": contract["run_id"],
        "created_utc": execution["pilot"]["created_utc"],
        "entries": rows,
        "entry_count": len(rows),
        "checkpoint_steps": EXPECTED_SNAPSHOTS,
        "checkpoint_count": 6,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
    }
    base.atomic_json(manifest_path, manifest)
    manifest_sha256 = base.sha256_file(manifest_path)
    source_by_archive = {
        "preflight.json": Path(contract["paths"]["preflight_output"]),
        "pilot/pilot.json": Path(contract["paths"]["pilot_output_dir"]) / "pilot.json",
        "pilot/pilot.svg": Path(contract["paths"]["figure_output"]),
        **{
            f"pilot/step-{step:06d}.pt": Path(contract["paths"]["pilot_output_dir"])
            / f"step-{step:06d}.pt"
            for step in EXPECTED_SNAPSHOTS
        },
        "artifact-manifest.json": manifest_path,
    }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for archive_name in sorted(source_by_archive):
            source = source_by_archive[archive_name]
            info = tarfile.TarInfo(archive_name)
            info.size = source.stat().st_size
            info.mtime = 0
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    size = archive_path.stat().st_size
    digest = base.sha256_file(archive_path)
    return {
        "path": archive_path,
        "bytes": size,
        "sha256": digest,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "deterministic_tar_metadata": True,
    }


def run(input_contract_path: Path, artifact_post_path: Path) -> dict[str, Any]:
    raw = json.loads(input_contract_path.read_text(encoding="utf-8"))
    contract = validate_contract(raw)
    run_id = str(contract["run_id"])
    post_contract = upload.validate_contract(
        json.loads(artifact_post_path.read_text(encoding="utf-8"))
    )
    if post_contract["run_id"] != run_id:
        raise PilotJobError("artifact POST run identity changed")
    final_output = Path(contract["paths"]["final_output"])
    if final_output.exists():
        raise PilotJobError("refusing to overwrite final pilot job result")
    free = shutil.disk_usage(final_output.parent).free
    if free < int(contract["bounds"]["minimum_free_bytes"]):
        raise PilotJobError("worker scratch is below the frozen free-space floor")

    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    bundle_download = base.download_source_bundle(contract)
    source_extract = base.extract_and_verify_bundle(contract)
    overlays = _download_overlays(contract)
    source = _verify_source(contract)
    hydration = base.hydrate_model(contract)

    input_contract_path.unlink()
    for row in raw.get("model", []):
        row["url"] = ""
    raw.get("bundle", {})["url"] = ""
    for row in raw.get("overlays", []):
        row["url"] = ""
    del raw

    execution = _run_science(contract)
    archive = _build_archive(contract, execution)
    with archive["path"].open("rb") as handle:
        artifact_proof = upload.post_stream(
            contract=post_contract,
            stream=handle,
            size=int(archive["bytes"]),
            sha256_hex=str(archive["sha256"]),
            filename="artifacts.tar",
            timeout_seconds=int(contract["bounds"]["artifact_upload_timeout_seconds"]),
        )
    artifact_post_path.unlink()
    post_contract.clear()
    archive["path"].unlink()

    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_SELECTED_PILOT_ARCHIVE_DURABLE",
        "run_id": run_id,
        "stage": "training_artifact_rehearsal",
        "pilot": execution.pop("pilot"),
        "artifact_proof": artifact_proof,
        "artifact_manifest": archive["manifest"],
        "artifact_manifest_sha256": archive["manifest_sha256"],
        "execution_provenance": {
            "started_utc": started_utc,
            "wall_seconds_including_hydration_and_upload": time.monotonic() - started,
            "durable_provider": "amazon_s3",
            "vast_storage": "ephemeral_instance_scratch_only",
            "aws_credentials_on_worker": False,
            "presigned_input_contract_deleted_before_model_import": True,
            "presigned_artifact_contract_deleted_after_upload": True,
            "local_archive_deleted_after_S3_version_returned": True,
            "bundle_download": bundle_download,
            "source_extract": source_extract,
            "overlays": overlays,
            "source": source,
            "hydration": hydration,
            "archive": {
                "bytes": archive["bytes"],
                "sha256": archive["sha256"],
                "deterministic_tar_metadata": True,
            },
            **execution,
            "automatic_retry": False,
            "promotion_eligible": False,
            "confirmation_items_read": 0,
        },
    }
    base.atomic_json(final_output, packet)
    if final_output.stat().st_size > 1024 * 1024:
        raise PilotJobError("pilot terminal payload exceeds the 1 MiB bound")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--artifact-post", type=Path, required=True)
    arguments = parser.parse_args()
    packet = run(arguments.input_contract.resolve(), arguments.artifact_post.resolve())
    print(
        json.dumps(
            {
                "status": packet["status"],
                "verdict": packet["pilot"]["mechanical_adjudication"]["verdict"],
                "artifact_version_id": packet["artifact_proof"]["version_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
