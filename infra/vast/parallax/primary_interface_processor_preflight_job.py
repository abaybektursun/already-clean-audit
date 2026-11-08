"""Run the exact PG1 interface processor gate without weights or GPU access.

The disposable control instance receives only immutable, version-bound S3 GET
URLs for the small PG1 processor inventory and the already frozen discovery
bundle.  It never receives a model-weight URL or AWS credentials.  The caller
sets ``CUDA_VISIBLE_DEVICES`` to the empty string, and this job records zero
model loads, zero GPU operations, and zero science evaluations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:  # Staged as a standalone sibling on the disposable control instance.
    import ephemeral_optimizer_scale_job as base
except ModuleNotFoundError:  # Imported through the repository in planted tests.
    from . import ephemeral_optimizer_scale_job as base


RUN_ID = "parallax-pg1-primary-processor-preflight-v1"
LINEAGE = "pg1_pi"
CONFIG_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_interface_control_pg1_primary_v1.yaml"
)
TASK_ROOT = "studies/lineage/cross-lineage/data/parallax_tasks_v1"
PROCESSOR_NAMES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
PROCESSOR_BYTES = 21_856_089
WEIGHT_NAMES = {
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
}
OVERLAY_IDENTITIES = {
    "infra/vast/parallax/training_interface_control.py": (
        "81ec1e5b7a6fa50db0a8243c21f03bb9c6a9830a6b25532de763abf8d5497da6"
    ),
    "infra/vast/parallax/training_interface_tokenizer_preflight.py": (
        "1f08fd2b43ddf9933abba8650c3582c837136fc7c92fd9abececb42ffea9be31"
    ),
    CONFIG_PATH: "fa5d8ccefeacfbb57fcd223b3b4486d4adef80388ae06d48452221d738e53078",
}


class PrimaryProcessorPreflightError(RuntimeError):
    pass


def _absolute(value: object, *, prefix: str, label: str) -> Path:
    path = Path(str(value))
    root = Path(prefix.rstrip("/"))
    try:
        path.relative_to(root)
    except ValueError:
        raise PrimaryProcessorPreflightError(f"{label} escapes {prefix}")
    if not path.is_absolute():
        raise PrimaryProcessorPreflightError(f"{label} is not absolute")
    return path


def _validate_processor(rows: object) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(PROCESSOR_NAMES):
        raise PrimaryProcessorPreflightError("processor inventory is not exact")
    names: set[str] = set()
    for row in rows:
        base._validate_object(row, kind="processor")
        name = str(row.get("path", ""))
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise PrimaryProcessorPreflightError("processor filename escapes snapshot root")
        if name in names:
            raise PrimaryProcessorPreflightError("processor inventory has a duplicate filename")
        names.add(name)
    if names != PROCESSOR_NAMES or sum(int(row["bytes"]) for row in rows) != PROCESSOR_BYTES:
        raise PrimaryProcessorPreflightError("PG1 processor identity changed")
    if names & WEIGHT_NAMES or any("safetensors" in name for name in names):
        raise PrimaryProcessorPreflightError("a model-weight object entered processor preflight")
    return rows


def _validate_bundle(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise PrimaryProcessorPreflightError("source bundle is absent")
    base._validate_object(bundle, kind="bundle")
    if (
        int(bundle.get("bytes", 0)) != base.EXPECTED_BUNDLE_BYTES
        or bundle.get("sha256") != base.EXPECTED_BUNDLE_SHA256
        or not isinstance(bundle.get("entries"), list)
        or len(bundle["entries"]) != 155
    ):
        raise PrimaryProcessorPreflightError("source bundle identity changed")
    paths = {str(row.get("path", "")) for row in bundle["entries"]}
    task_prefix = TASK_ROOT + "/"
    if len(paths) != 155:
        raise PrimaryProcessorPreflightError("source bundle paths are duplicated")
    if (
        task_prefix + "manifests/items.jsonl" in paths
        or task_prefix + "manifests/confirmation.sha256" in paths
        or any(path.startswith(task_prefix) and "/confirmation_eval/" in path for path in paths)
    ):
        raise PrimaryProcessorPreflightError("confirmation-capable data entered preflight")
    discovery = [
        path for path in paths
        if path.startswith(task_prefix + "images/") and "/discovery_eval/" in path
    ]
    if len(discovery) != 24:
        raise PrimaryProcessorPreflightError("discovery anchor population changed")
    return bundle


def _validate_overlays(rows: object) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(OVERLAY_IDENTITIES):
        raise PrimaryProcessorPreflightError("processor preflight overlays are not exact")
    observed: dict[str, str] = {}
    staged_paths: set[str] = set()
    for row in rows:
        if set(row) != {"path", "sha256", "bytes", "staged_path"}:
            raise PrimaryProcessorPreflightError("processor preflight overlay schema changed")
        path = str(row["path"])
        relative = PurePosixPath(path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PrimaryProcessorPreflightError("processor preflight overlay path is unsafe")
        staged = _absolute(row["staged_path"], prefix="/run/parallax/", label="overlay")
        if str(staged) in staged_paths or int(row.get("bytes", 0)) <= 0:
            raise PrimaryProcessorPreflightError("processor preflight overlay is duplicated or empty")
        staged_paths.add(str(staged))
        observed[path] = str(row["sha256"])
    if observed != OVERLAY_IDENTITIES or len(observed) != len(rows):
        raise PrimaryProcessorPreflightError("processor preflight overlay identities changed")
    return rows


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if set(contract) != {
        "schema_version", "status", "stage", "provider", "run_id", "lineage",
        "instance_id", "storage", "processor", "bundle", "overlays", "paths",
        "bounds", "firewall",
    }:
        raise PrimaryProcessorPreflightError("processor preflight contract schema changed")
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "FROZEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT_JOB"
        or contract.get("stage") != "primary_pg1_interface_processor_preflight"
        or contract.get("provider") != "vast.ai"
        or contract.get("run_id") != RUN_ID
        or contract.get("lineage") != LINEAGE
        or not isinstance(contract.get("instance_id"), int)
        or int(contract["instance_id"]) <= 0
    ):
        raise PrimaryProcessorPreflightError("processor preflight identity changed")
    if contract.get("storage") != {
        "durable_provider": "amazon_s3",
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": 30,
        "aws_credentials_on_worker": False,
    }:
        raise PrimaryProcessorPreflightError("processor preflight storage boundary changed")
    _validate_processor(contract.get("processor"))
    _validate_bundle(contract.get("bundle"))
    _validate_overlays(contract.get("overlays"))
    paths = contract.get("paths", {})
    if set(paths) != {
        "runtime_root", "source_root", "bundle_path", "snapshot_root",
        "preflight_output", "final_output",
    }:
        raise PrimaryProcessorPreflightError("processor preflight path schema changed")
    for name in paths:
        _absolute(paths[name], prefix="/workspace/parallax/", label=name)
    if contract.get("bounds") != {
        "parallel_downloads": 4,
        "download_timeout_seconds": 180,
        "preflight_timeout_seconds": 120,
        "minimum_free_bytes": 5_000_000_000,
    }:
        raise PrimaryProcessorPreflightError("processor preflight bounds changed")
    if contract.get("firewall") != {
        "h100_creation_reachable": False,
        "model_weight_urls": 0,
        "model_loads": 0,
        "gpu_operations": 0,
        "science_items_evaluated": 0,
        "discovery_items_processor_only": 3,
        "confirmation_items_read": 0,
        "automatic_retry": False,
    }:
        raise PrimaryProcessorPreflightError("processor preflight firewall changed")
    return contract


def _install_overlays(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    installed = []
    for row in sorted(contract["overlays"], key=lambda value: str(value["path"])):
        staged = Path(str(row["staged_path"]))
        if (
            not staged.is_file()
            or staged.stat().st_size != int(row["bytes"])
            or base.sha256_file(staged) != row["sha256"]
        ):
            raise PrimaryProcessorPreflightError(f"staged overlay mismatch: {row['path']}")
        destination = source / str(row["path"])
        if destination.exists():
            raise PrimaryProcessorPreflightError(f"overlay would replace bundled input: {row['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        shutil.copyfile(staged, temporary)
        if base.sha256_file(temporary) != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise PrimaryProcessorPreflightError(f"installed overlay mismatch: {row['path']}")
        os.replace(temporary, destination)
        staged.unlink()
        installed.append({"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]})
    return {"status": "GREEN_EXACT_PROCESSOR_PREFLIGHT_OVERLAYS", "files": installed}


def _hydrate_processor(contract: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(contract["paths"]["snapshot_root"])
    partial = snapshot.with_name(snapshot.name + ".partial")
    if snapshot.exists() or partial.exists():
        raise PrimaryProcessorPreflightError("processor snapshot destination already exists")
    partial.mkdir(parents=True)
    started = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(contract["bounds"]["parallel_downloads"])
    )
    try:
        futures = [
            pool.submit(base._download, row, partial / str(row["path"]))
            for row in contract["processor"]
        ]
        _done, pending = concurrent.futures.wait(
            futures, timeout=int(contract["bounds"]["download_timeout_seconds"])
        )
        if pending:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise PrimaryProcessorPreflightError("processor hydration exceeded frozen timeout")
        records = [future.result() for future in futures]
        pool.shutdown(wait=True)
        os.replace(partial, snapshot)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {
        "status": "GREEN_EXACT_PG1_PROCESSOR_ONLY",
        "verified_objects": len(records),
        "verified_bytes": sum(int(row["bytes"]) for row in records),
        "elapsed_seconds": time.monotonic() - started,
        "weight_objects_downloaded": 0,
        "weight_bytes_downloaded": 0,
        "records": records,
    }


def _clear_presigned_inputs(contract_path: Path, raw: dict[str, Any]) -> None:
    contract_path.unlink(missing_ok=True)
    for row in raw.get("processor", []):
        row["url"] = ""
    raw.get("bundle", {})["url"] = ""


def _run_processor_gate(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(contract["paths"]["source_root"])
    output = Path(contract["paths"]["preflight_output"])
    command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_interface_tokenizer_preflight",
        "--lineage", LINEAGE,
        "--config", str(source / CONFIG_PATH),
        "--snapshot", str(contract["paths"]["snapshot_root"]),
        "--task-root", str(source / TASK_ROOT),
        "--output", str(output),
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": str(source),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    run = base._run_command(
        command,
        cwd=source,
        environment=environment,
        timeout=int(contract["bounds"]["preflight_timeout_seconds"]),
        label="primary PG1 processor-only interface gate",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if (
        result.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or result.get("lineage") != LINEAGE
        or result.get("model_loads") != 0
        or result.get("gpu_operations") != 0
        or result.get("science_items_evaluated") != 0
        or result.get("discovery_items_processor_only") != 3
        or result.get("confirmation_items_read") != 0
        or len(set(result.get("candidate_suffix_tokens", {}).values())) != 1
    ):
        raise PrimaryProcessorPreflightError("PG1 processor-only gate is not exact green")
    return result, run


def run(contract_path: Path) -> dict[str, Any]:
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = validate_contract(raw)
    output = Path(contract["paths"]["final_output"])
    if output.exists():
        raise PrimaryProcessorPreflightError("refusing to overwrite processor preflight result")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < int(contract["bounds"]["minimum_free_bytes"]):
        raise PrimaryProcessorPreflightError("disposable scratch is below the frozen free-space floor")
    started = time.monotonic()
    try:
        bundle_download = base.download_source_bundle(contract)
        source_extract = base.extract_and_verify_bundle(contract)
        overlays = _install_overlays(contract)
        processor = _hydrate_processor(contract)
        _clear_presigned_inputs(contract_path, raw)
        preflight, command = _run_processor_gate(contract)
        packet = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT",
            "run_id": RUN_ID,
            "lineage": LINEAGE,
            "preflight": preflight,
            "bundle": bundle_download,
            "source": source_extract,
            "overlays": overlays,
            "processor": processor,
            "command": command,
            "h100_instances_created": 0,
            "model_weight_urls": 0,
            "model_loads": 0,
            "gpu_operations": 0,
            "science_items_evaluated": 0,
            "confirmation_items_read": 0,
            "automatic_retry": False,
            "wall_seconds": time.monotonic() - started,
        }
    except BaseException as error:
        _clear_presigned_inputs(contract_path, raw)
        packet = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "RED_PRIMARY_PG1_PROCESSOR_PREFLIGHT",
            "run_id": RUN_ID,
            "lineage": LINEAGE,
            "error": {"type": type(error).__name__, "message": str(error)[:16_384]},
            "h100_instances_created": 0,
            "model_weight_urls": 0,
            "model_loads": 0,
            "gpu_operations": 0,
            "science_items_evaluated": 0,
            "confirmation_items_read": 0,
            "automatic_retry": False,
            "wall_seconds": time.monotonic() - started,
        }
    base.atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract.resolve())
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
