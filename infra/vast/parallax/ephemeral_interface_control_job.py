"""Hydrate processor first, gate A/B/C, then run the PG2 interface control.

This worker-side job has no AWS credentials.  Every input and the artifact POST
arrive as version-bound, narrowly scoped presigned contracts.  Weight shards
are structurally unreachable until the processor-only confound gate passes.
"""

from __future__ import annotations

import concurrent.futures
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
    import interface_control_s3_post_upload as upload
except ModuleNotFoundError:  # Imported through the repository in planted tests.
    from . import ephemeral_optimizer_scale_job as base
    from . import interface_control_s3_post_upload as upload


RUN_ID = "parallax-pg2-interface-control-s3-v3"
LINEAGE = "pg2"
SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_interface_control_source_contract_pg2_v3.json"
)
SOURCE_CONTRACT_SHA256 = "c521bbf98a6eafc421cb1f1484d65c960fca80a53d23d9a185c8c449e73c8293"
CONFIG_PATH = (
    "studies/lineage/cross-lineage/config/parallax_training_interface_control_v1.yaml"
)
MATRIX_PATH = "studies/lineage/cross-lineage/config/parallax_matrix.yaml"
TASK_ROOT = "studies/lineage/cross-lineage/data/parallax_tasks_v1"
MODEL_FILES = 9
MODEL_BYTES = 6_099_504_596
MODEL_NAMES = {
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
WEIGHT_NAMES = {
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
}
OVERLAY_IDENTITIES = {
    "infra/vast/parallax/training_interface_control.py": (
        "81ec1e5b7a6fa50db0a8243c21f03bb9c6a9830a6b25532de763abf8d5497da6"
    ),
    "infra/vast/parallax/training_interface_tokenizer_preflight.py": (
        "1f08fd2b43ddf9933abba8650c3582c837136fc7c92fd9abececb42ffea9be31"
    ),
    "infra/vast/parallax/training_interface_control_cell.py": (
        "2bdfb7bef9137eccf51211b5708d2bef9f98fe3d2577593479ab2c24f284647e"
    ),
    CONFIG_PATH: "4ca5c1dfded233339cf894fe3ab66a1fb581ad1989a5f3a96da827625c3d4d61",
    SOURCE_CONTRACT_PATH: SOURCE_CONTRACT_SHA256,
    "studies/lineage/cross-lineage/config/parallax_pg2_h100_s3_hydration_job_v1.json": (
        "709b84e2b0276a3afcd9c9265e6b1f8e54d6f24f845d0b696cfe2285160e893a"
    ),
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_optimizer_scale_s3_v3_adjudication.json": (
        "97e4e17140acd7d9b4b7ac4dd96ff5f51888ce561c70dd5fe987270d89d16a1d"
    ),
    "studies/lineage/cross-lineage/research/2026-07-18_parallax_interface_control_spec.md": (
        "ae7b8cedfc7dec15782c797db154bd11c7bad1f2e2ccc916788017c26d419c03"
    ),
    "studies/lineage/cross-lineage/config/parallax_execution_freeze_v102.json": (
        "0a58d8fe12b184768217744f7a02754cc85c3d6e10c0c4850a6ea53a1269b5b8"
    ),
}
EXPECTED_SNAPSHOTS = [0, 1, 2, 4, 8, 16]
EXPECTED_CODEBOOKS = ["cycle_0", "cycle_1", "cycle_2"]


class InterfaceJobError(RuntimeError):
    pass


def _absolute(value: object, *, prefix: str, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not str(path).startswith(prefix):
        raise InterfaceJobError(f"{label} escapes {prefix}")
    return path


def _validate_bundle(bundle: dict[str, Any]) -> None:
    base._validate_object(bundle, kind="bundle")
    if (
        int(bundle.get("bytes", 0)) != base.EXPECTED_BUNDLE_BYTES
        or bundle.get("sha256") != base.EXPECTED_BUNDLE_SHA256
        or not isinstance(bundle.get("entries"), list)
        or len(bundle["entries"]) != 155
    ):
        raise InterfaceJobError("interface source bundle identity changed")
    paths = {str(row.get("path", "")) for row in bundle["entries"]}
    task_prefix = TASK_ROOT + "/"
    if (
        task_prefix + "manifests/items.jsonl" in paths
        or task_prefix + "manifests/confirmation.sha256" in paths
        or any(path.startswith(task_prefix) and "/confirmation_eval/" in path for path in paths)
    ):
        raise InterfaceJobError("interface bundle contains confirmation-capable data")
    if len([path for path in paths if path.startswith(task_prefix + "images/") and "/train/" in path]) != 96:
        raise InterfaceJobError("interface bundle train-image population changed")
    if len([path for path in paths if path.startswith(task_prefix + "images/") and "/discovery_eval/" in path]) != 24:
        raise InterfaceJobError("interface bundle discovery-image population changed")


def _validate_overlays(rows: object) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(OVERLAY_IDENTITIES):
        raise InterfaceJobError("interface job must bind every exact source overlay")
    observed = {}
    for row in rows:
        base._validate_object(row, kind="overlay")
        path = str(row.get("path", ""))
        relative = PurePosixPath(path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or path in observed:
            raise InterfaceJobError("interface overlay path is unsafe or duplicated")
        observed[path] = str(row["sha256"])
    if observed != OVERLAY_IDENTITIES:
        raise InterfaceJobError("interface overlay identity set changed")
    return rows


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "FROZEN_H100_INTERFACE_CONTROL_JOB"
        or contract.get("stage") != "training_interface_control"
        or contract.get("provider") != "vast.ai"
        or contract.get("run_id") != RUN_ID
        or contract.get("lineage") != LINEAGE
    ):
        raise InterfaceJobError("interface job contract is not exact PG2 schema v1")
    if not isinstance(contract.get("instance_id"), int) or int(contract["instance_id"]) <= 0:
        raise InterfaceJobError("interface job instance ID is absent")
    if contract.get("storage") != {
        "durable_provider": "amazon_s3",
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": 80,
        "aws_credentials_on_worker": False,
    }:
        raise InterfaceJobError("interface job storage contract changed")
    model = contract.get("model")
    if not isinstance(model, list) or len(model) != MODEL_FILES:
        raise InterfaceJobError("interface job must bind nine PG2 model objects")
    for row in model:
        base._validate_object(row, kind="model")
    if (
        {str(row.get("path", "")) for row in model} != MODEL_NAMES
        or sum(int(row["bytes"]) for row in model) != MODEL_BYTES
    ):
        raise InterfaceJobError("interface PG2 model inventory changed")
    _validate_bundle(contract.get("bundle", {}))
    _validate_overlays(contract.get("overlays"))
    paths = contract.get("paths", {})
    for name in (
        "source_root",
        "bundle_path",
        "tokenizer_preflight_output",
        "control_output_dir",
        "manifest_output",
        "archive_output",
        "phase_output",
        "final_output",
    ):
        _absolute(paths.get(name), prefix="/workspace/parallax/", label=name)
    _absolute(paths.get("snapshot_root"), prefix="/root/.cache/huggingface/hub/", label="snapshot")
    if contract.get("source_contract_sha256") != SOURCE_CONTRACT_SHA256:
        raise InterfaceJobError("interface source contract hash changed")
    if contract.get("bounds") != {
        "parallel_downloads": 3,
        "processor_download_timeout_seconds": 180,
        "weight_download_timeout_seconds": 600,
        "tokenizer_preflight_timeout_seconds": 120,
        "control_timeout_seconds": 900,
        "artifact_upload_timeout_seconds": 600,
        "minimum_free_bytes": 50_000_000_000,
        "maximum_terminal_bytes": 4 * 1024 * 1024,
    }:
        raise InterfaceJobError("interface execution bounds changed")
    if contract.get("firewall") != {
        "processor_gate_before_weight_hydration": True,
        "model_loads_before_processor_gate": 0,
        "weight_objects_before_processor_gate": 0,
        "train_items": 96,
        "discovery_items": 24,
        "confirmation_manifest_present": False,
        "confirmation_images_present": False,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
        "automatic_retry": False,
    }:
        raise InterfaceJobError("interface science/data firewall changed")
    return contract


def _phase(contract: dict[str, Any], phase: str, **updates: Any) -> None:
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": phase,
        "science_started": phase in {"SCIENCE_STARTED", "SCIENCE_FINISHED", "ARTIFACT_DURABLE"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **updates,
    }
    base.atomic_json(Path(contract["paths"]["phase_output"]), payload)


def _download_overlays(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    records = []
    for row in sorted(contract["overlays"], key=lambda value: str(value["path"])):
        destination = source / str(row["path"])
        if destination.exists():
            raise InterfaceJobError(f"interface overlay would replace a bundled file: {row['path']}")
        records.append(base._download(row, destination))
    return {
        "status": "GREEN_EXACT_INTERFACE_OVERLAYS",
        "objects": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "records": records,
    }


def _verify_source(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    path = source / SOURCE_CONTRACT_PATH
    if not path.is_file() or base.sha256_file(path) != SOURCE_CONTRACT_SHA256:
        raise InterfaceJobError("interface source contract is absent or changed")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if (
        frozen.get("status") != "FROZEN"
        or frozen.get("stage") != "training_interface_control"
        or frozen.get("run_id") != RUN_ID
        or frozen.get("lineage") != LINEAGE
        or frozen.get("confirmation_items_read") != 0
        or frozen.get("destination", {}).get("persistent_system_of_record") != "amazon_s3"
        or frozen.get("destination", {}).get("vast_persistent_storage") != "FORBIDDEN"
    ):
        raise InterfaceJobError("interface source contract escaped its boundary")
    verified = 0
    for label, row in frozen["artifacts"].items():
        target = source / str(row["path"])
        if not target.is_file() or base.sha256_file(target) != row["sha256"]:
            raise InterfaceJobError(f"interface source artifact mismatch: {label}")
        verified += 1
    return {
        "status": "GREEN_EXACT_INTERFACE_SOURCE",
        "verified_artifacts": verified,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
    }


def _hydrate_rows(
    contract: dict[str, Any], rows: list[dict[str, Any]], *, timeout_seconds: int
) -> list[dict[str, Any]]:
    partial = Path(contract["paths"]["snapshot_root"] + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=int(contract["bounds"]["parallel_downloads"]))
    futures = [pool.submit(base._download, row, partial / str(row["path"])) for row in rows]
    try:
        done, pending = concurrent.futures.wait(futures, timeout=timeout_seconds)
        if pending:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise InterfaceJobError("interface hydration exceeded its frozen timeout")
        records = [future.result() for future in futures]
        pool.shutdown(wait=True)
        return records
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise


def _hydrate_processor(contract: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(contract["paths"]["snapshot_root"])
    partial = Path(str(snapshot) + ".partial")
    if snapshot.exists() or partial.exists():
        raise InterfaceJobError("interface snapshot destination already exists")
    rows = [row for row in contract["model"] if row["path"] not in WEIGHT_NAMES]
    started = time.monotonic()
    records = _hydrate_rows(
        contract,
        rows,
        timeout_seconds=int(contract["bounds"]["processor_download_timeout_seconds"]),
    )
    return {
        "status": "GREEN_PROCESSOR_FILES_ONLY",
        "elapsed_seconds": time.monotonic() - started,
        "verified_objects": len(records),
        "verified_bytes": sum(int(row["bytes"]) for row in records),
        "weight_objects_downloaded": 0,
        "records": records,
    }


def _hydrate_weights(contract: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(contract["paths"]["snapshot_root"])
    partial = Path(str(snapshot) + ".partial")
    if snapshot.exists() or not partial.is_dir():
        raise InterfaceJobError("processor-gated partial snapshot is absent")
    rows = [row for row in contract["model"] if row["path"] in WEIGHT_NAMES]
    started = time.monotonic()
    records = _hydrate_rows(
        contract,
        rows,
        timeout_seconds=int(contract["bounds"]["weight_download_timeout_seconds"]),
    )
    os.sync()
    os.replace(partial, snapshot)
    return {
        "status": "GREEN_WEIGHTS_AFTER_PROCESSOR_GATE",
        "elapsed_seconds": time.monotonic() - started,
        "verified_objects": len(records),
        "verified_bytes": sum(int(row["bytes"]) for row in records),
        "records": records,
    }


def _runtime_environment(source: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(source),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _run_tokenizer_preflight(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(contract["paths"]["source_root"])
    output = Path(contract["paths"]["tokenizer_preflight_output"])
    command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_interface_tokenizer_preflight",
        "--lineage",
        LINEAGE,
        "--config",
        str(source / CONFIG_PATH),
        "--snapshot",
        contract["paths"]["snapshot_root"] + ".partial",
        "--task-root",
        str(source / TASK_ROOT),
        "--output",
        str(output),
    ]
    run = base._run_command(
        command,
        cwd=source,
        environment=_runtime_environment(source),
        timeout=int(contract["bounds"]["tokenizer_preflight_timeout_seconds"]),
        label="processor-only A/B/C confound gate",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if (
        result.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or result.get("lineage") != LINEAGE
        or result.get("model_loads") != 0
        or result.get("gpu_operations") != 0
        or result.get("confirmation_items_read") != 0
        or len(set(result.get("candidate_suffix_tokens", {}).values())) != 1
    ):
        raise InterfaceJobError("processor-only A/B/C gate is not exact green")
    return result, run


def _clear_presigned_inputs(contract_path: Path, raw: dict[str, Any]) -> None:
    contract_path.unlink(missing_ok=True)
    for row in raw.get("model", []):
        row["url"] = ""
    raw.get("bundle", {})["url"] = ""
    for row in raw.get("overlays", []):
        row["url"] = ""


def _run_control(
    contract: dict[str, Any], tokenizer_result: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(contract["paths"]["source_root"])
    output_dir = Path(contract["paths"]["control_output_dir"])
    cache_root = Path(contract["paths"]["snapshot_root"]).parents[2]
    command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_interface_control_cell",
        "--root",
        str(source),
        "--source-contract",
        str(source / SOURCE_CONTRACT_PATH),
        "--matrix",
        str(source / MATRIX_PATH),
        "--config",
        str(source / CONFIG_PATH),
        "--tokenizer-preflight",
        str(Path(contract["paths"]["tokenizer_preflight_output"])),
        "--cache-root",
        str(cache_root),
        "--task-root",
        str(source / TASK_ROOT),
        "--lineage",
        LINEAGE,
        "--instance-id",
        str(contract["instance_id"]),
        "--output-dir",
        str(output_dir),
    ]
    run = base._run_command(
        command,
        cwd=source,
        environment=_runtime_environment(source),
        timeout=int(contract["bounds"]["control_timeout_seconds"]),
        label="PG2 three-codebook interface control",
    )
    result_path = output_dir / "interface-control.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "GREEN_MECHANICAL_EXECUTION"
        or result.get("lineage") != LINEAGE
        or result.get("label") != "INTERFACE_CONTROL_PROVISIONAL_NOT_PROMOTION_ELIGIBLE"
        or result.get("confirmation_items_read") != 0
        or result.get("promotion_eligible") is not False
        or result.get("fused_base_reset_exact") is not True
        or [row.get("codebook_id") for row in result.get("codebooks", [])] != EXPECTED_CODEBOOKS
        or any(
            [int(snapshot["step"]) for snapshot in row.get("snapshots", [])] != EXPECTED_SNAPSHOTS
            for row in result.get("codebooks", [])
        )
        or result.get("tokenizer_preflight", {}).get("candidate_suffix_tokens")
        != tokenizer_result["candidate_suffix_tokens"]
    ):
        raise InterfaceJobError("PG2 interface-control output escaped its frozen boundary")
    return result, run


def _artifact_rows(contract: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = Path(contract["paths"]["tokenizer_preflight_output"])
    output_dir = Path(contract["paths"]["control_output_dir"])
    rows = [
        {"archive_path": "tokenizer-preflight.json", "path": preflight},
        {"archive_path": "interface/interface-control.json", "path": output_dir / "interface-control.json"},
    ]
    for cell in control["codebooks"]:
        codebook_id = str(cell["codebook_id"])
        for snapshot in cell["snapshots"]:
            step = int(snapshot["step"])
            checkpoint = Path(str(snapshot["checkpoint"]["path"]))
            expected = output_dir / codebook_id / f"step-{step:06d}.pt"
            if checkpoint != expected or not checkpoint.is_file():
                raise InterfaceJobError("interface checkpoint escaped its exact cell directory")
            if (
                checkpoint.stat().st_size != int(snapshot["checkpoint"]["bytes"])
                or base.sha256_file(checkpoint) != snapshot["checkpoint"]["sha256"]
            ):
                raise InterfaceJobError("interface checkpoint proof mismatch")
            rows.append(
                {"archive_path": f"interface/{codebook_id}/{checkpoint.name}", "path": checkpoint}
            )
    if len(rows) != 20:
        raise InterfaceJobError("interface artifact inventory is not exact 20 files")
    return rows


def _build_archive(contract: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(contract["paths"]["manifest_output"])
    archive_path = Path(contract["paths"]["archive_output"])
    if manifest_path.exists() or archive_path.exists():
        raise InterfaceJobError("refusing to overwrite interface manifest/archive")
    sources = _artifact_rows(contract, control)
    entries = [
        {
            "path": row["archive_path"],
            "bytes": row["path"].stat().st_size,
            "sha256": base.sha256_file(row["path"]),
        }
        for row in sorted(sources, key=lambda value: value["archive_path"])
    ]
    manifest = {
        "schema_version": 1,
        "status": "GREEN_EXACT_INTERFACE_CONTROL_ARTIFACT_MANIFEST",
        "run_id": RUN_ID,
        "created_utc": control["created_utc"],
        "entries": entries,
        "entry_count": len(entries),
        "checkpoint_count": 18,
        "codebooks": EXPECTED_CODEBOOKS,
        "checkpoint_steps_per_codebook": EXPECTED_SNAPSHOTS,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
    }
    base.atomic_json(manifest_path, manifest)
    sources.append({"archive_path": "artifact-manifest.json", "path": manifest_path})
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in sorted(sources, key=lambda value: value["archive_path"]):
            source = row["path"]
            info = tarfile.TarInfo(row["archive_path"])
            info.size = source.stat().st_size
            info.mtime = 0
            info.mode = 0o600
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "path": archive_path,
        "bytes": archive_path.stat().st_size,
        "sha256": base.sha256_file(archive_path),
        "manifest": manifest,
        "manifest_sha256": base.sha256_file(manifest_path),
        "deterministic_tar_metadata": True,
    }


def run(contract_path: Path, artifact_post_path: Path) -> dict[str, Any]:
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = validate_contract(raw)
    post_contract = upload.validate_contract(json.loads(artifact_post_path.read_text(encoding="utf-8")))
    if post_contract["run_id"] != RUN_ID:
        raise InterfaceJobError("interface artifact POST identity changed")
    final_output = Path(contract["paths"]["final_output"])
    if final_output.exists():
        raise InterfaceJobError("refusing to overwrite interface job result")
    if shutil.disk_usage(final_output.parent).free < int(contract["bounds"]["minimum_free_bytes"]):
        raise InterfaceJobError("interface worker scratch is below the frozen floor")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    _phase(contract, "SOURCE_STAGING")
    bundle_download = base.download_source_bundle(contract)
    source_extract = base.extract_and_verify_bundle(contract)
    overlays = _download_overlays(contract)
    source = _verify_source(contract)
    _phase(contract, "PROCESSOR_HYDRATION", weight_objects_downloaded=0, model_loads=0)
    processor_hydration = _hydrate_processor(contract)
    _phase(contract, "TOKENIZER_PREFLIGHT", weight_objects_downloaded=0, model_loads=0)
    try:
        tokenizer_result, tokenizer_run = _run_tokenizer_preflight(contract)
    except Exception as error:
        _clear_presigned_inputs(contract_path, raw)
        artifact_post_path.unlink(missing_ok=True)
        post_contract.clear()
        _phase(
            contract,
            "TOKENIZER_PREFLIGHT_RED",
            weight_objects_downloaded=0,
            model_loads=0,
            error_type=type(error).__name__,
        )
        packet = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "RED_TOKENIZER_PREFLIGHT_BEFORE_WEIGHT_HYDRATION",
            "run_id": RUN_ID,
            "stage": "training_interface_control",
            "lineage": LINEAGE,
            "error": {"type": type(error).__name__, "message": str(error)[:16_384]},
            "processor_hydration": processor_hydration,
            "weight_objects_downloaded": 0,
            "weight_bytes_downloaded": 0,
            "model_loads": 0,
            "science_started": False,
            "confirmation_items_read": 0,
            "automatic_retry": False,
        }
        base.atomic_json(final_output, packet)
        return packet
    _phase(contract, "TOKENIZER_PREFLIGHT_GREEN", weight_objects_downloaded=0, model_loads=0)
    weight_hydration = _hydrate_weights(contract)
    _clear_presigned_inputs(contract_path, raw)
    _phase(contract, "SCIENCE_STARTED", weight_objects_downloaded=len(WEIGHT_NAMES))
    control, control_run = _run_control(contract, tokenizer_result)
    _phase(contract, "SCIENCE_FINISHED", verdict=control["adjudication"]["verdict"])
    archive = _build_archive(contract, control)
    with archive["path"].open("rb") as handle:
        artifact_proof = upload.post_stream(
            contract=post_contract,
            stream=handle,
            size=int(archive["bytes"]),
            sha256_hex=str(archive["sha256"]),
            filename="artifacts.tar",
            timeout_seconds=int(contract["bounds"]["artifact_upload_timeout_seconds"]),
        )
    artifact_post_path.unlink(missing_ok=True)
    post_contract.clear()
    archive["path"].unlink()
    _phase(contract, "ARTIFACT_DURABLE", artifact_version_id=artifact_proof["version_id"])
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_INTERFACE_CONTROL_ARCHIVE_DURABLE",
        "run_id": RUN_ID,
        "stage": "training_interface_control",
        "lineage": LINEAGE,
        "control": control,
        "artifact_proof": artifact_proof,
        "artifact_manifest": archive["manifest"],
        "artifact_manifest_sha256": archive["manifest_sha256"],
        "execution_provenance": {
            "started_utc": started_utc,
            "wall_seconds_including_hydration_and_upload": time.monotonic() - started,
            "durable_provider": "amazon_s3",
            "vast_storage": "ephemeral_instance_scratch_only",
            "aws_credentials_on_worker": False,
            "processor_gate_before_weight_hydration": True,
            "presigned_input_contract_deleted_before_model_import": True,
            "local_archive_deleted_after_S3_version_returned": True,
            "bundle_download": bundle_download,
            "source_extract": source_extract,
            "overlays": overlays,
            "source": source,
            "processor_hydration": processor_hydration,
            "tokenizer_preflight": tokenizer_result,
            "tokenizer_preflight_run": tokenizer_run,
            "weight_hydration": weight_hydration,
            "control_run": control_run,
            "archive": {
                "bytes": archive["bytes"],
                "sha256": archive["sha256"],
                "deterministic_tar_metadata": True,
            },
            "automatic_retry": False,
            "promotion_eligible": False,
            "confirmation_items_read": 0,
        },
    }
    base.atomic_json(final_output, packet)
    if final_output.stat().st_size > int(contract["bounds"]["maximum_terminal_bytes"]):
        raise InterfaceJobError("interface terminal payload exceeds the frozen 4 MiB bound")
    return packet


def main() -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--artifact-post", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.input_contract.resolve(), args.artifact_post.resolve())
    print(
        json.dumps(
            {
                "status": packet["status"],
                "verdict": packet.get("control", {}).get("adjudication", {}).get("verdict"),
                "artifact_version_id": packet.get("artifact_proof", {}).get("version_id"),
            },
            sort_keys=True,
        )
    )
    return 0 if packet["status"] == "GREEN_INTERFACE_CONTROL_ARCHIVE_DURABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
