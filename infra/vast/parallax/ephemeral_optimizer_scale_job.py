"""Hydrate, preflight, and run the frozen PG1 optimizer microgate on one H100.

This file is staged directly into a disposable Vast worker.  It deliberately
uses only the Python standard library until the exact source bundle and model
snapshot have been downloaded from immutable S3 versions and verified.  AWS
credentials are never present on the worker; the controller supplies narrowly
scoped presigned GET URLs in a 0600 one-shot contract which this process deletes
before importing the scientific runtime.

The source bundle predates the train-only optimizer gate and contains discovery
images needed by the later pilot.  The frozen H100 preflight verifies that
bundle, after which this wrapper physically removes every discovery image before
any model is loaded.  The optimizer runner itself remains byte-identical.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
EXPECTED_MODEL_BYTES = 11_715_866_017
EXPECTED_MODEL_FILES = 12
EXPECTED_MODEL_NAMES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
EXPECTED_BUNDLE_BYTES = 12_626_256
EXPECTED_BUNDLE_SHA256 = "8b2f16138376e992816f669fb63ab9ff18d6085da3d009aa9931f6e0a834c517"
EXPECTED_SOURCE_CONTRACT_SHA256 = "fe693a6f9ed9542170a5cba8e539308875f09e1143e33375032136da94c212c9"
EXPECTED_SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_optimizer_scale_source_contract_v1.json"
)
EXPECTED_MATRIX_PATH = "studies/lineage/cross-lineage/config/parallax_matrix.yaml"
EXPECTED_CONFIG_PATH = (
    "studies/lineage/cross-lineage/config/parallax_training_optimizer_scale_gate_v1.yaml"
)
EXPECTED_TASK_ROOT = "studies/lineage/cross-lineage/data/parallax_tasks_v1"


class OptimizerJobError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _absolute_path(value: object, *, prefix: str, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not str(path).startswith(prefix):
        raise OptimizerJobError(f"{label} escapes {prefix}")
    return path


def _digest(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OptimizerJobError(f"{label} is not a lowercase SHA-256")
    return digest


def validate_get_url(url: str, row: dict[str, Any]) -> str:
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    try:
        expiry = int(query.get("X-Amz-Expires", ["0"])[0])
    except ValueError as error:
        raise OptimizerJobError("input URL expiry is malformed") from error
    if (
        parsed.scheme != "https"
        or parsed.netloc != f"{BUCKET}.s3.{REGION}.amazonaws.com"
        or unquote(parsed.path) != "/" + str(row["key"])
        or query.get("versionId") != [str(row["version_id"])]
        or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
        or query.get("X-Amz-SignedHeaders") != ["host"]
        or not 60 <= expiry <= 3600
        or len(query.get("X-Amz-Signature", [""])[0]) != 64
    ):
        raise OptimizerJobError("input URL is not exact, version-bound, and presigned")
    return url


def _validate_object(row: dict[str, Any], *, kind: str) -> dict[str, Any]:
    digest = _digest(row.get("sha256"), label=f"{kind} digest")
    key = str(row.get("key", ""))
    version_id = str(row.get("version_id", ""))
    if (
        row.get("bucket") != BUCKET
        or row.get("region") != REGION
        or not key.startswith(f"parallax/input-staging/sha256/{digest}/")
        or ".." in key.split("/")
        or not version_id
        or int(row.get("bytes", 0)) <= 0
    ):
        raise OptimizerJobError(f"{kind} source is not an exact immutable S3 object")
    validate_get_url(str(row.get("url", "")), row)
    return row


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN_H100_OPTIMIZER_SCALE_JOB":
        raise OptimizerJobError("optimizer job contract is not frozen schema v1")
    if contract.get("stage") != "training_optimizer_scale_microgate" or contract.get("provider") != "vast.ai":
        raise OptimizerJobError("optimizer job escapes its registered stage/provider")
    instance_id = contract.get("instance_id")
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise OptimizerJobError("optimizer job instance ID is absent")
    storage = contract.get("storage")
    if storage != {
        "durable_provider": "amazon_s3",
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "aws_credentials_on_worker": False,
    }:
        raise OptimizerJobError("optimizer storage contract is not S3-only")
    model = contract.get("model")
    if not isinstance(model, list) or len(model) != EXPECTED_MODEL_FILES:
        raise OptimizerJobError("optimizer job must bind all twelve PG1 model files")
    names: set[str] = set()
    for row in model:
        _validate_object(row, kind="model")
        name = str(row.get("path", ""))
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise OptimizerJobError("model filename escapes the exact snapshot root")
        names.add(name)
    if names != EXPECTED_MODEL_NAMES or sum(int(row["bytes"]) for row in model) != EXPECTED_MODEL_BYTES:
        raise OptimizerJobError("optimizer PG1 model inventory changed")
    bundle = _validate_object(contract.get("bundle", {}), kind="bundle")
    if int(bundle["bytes"]) != EXPECTED_BUNDLE_BYTES or bundle["sha256"] != EXPECTED_BUNDLE_SHA256:
        raise OptimizerJobError("optimizer source bundle identity changed")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != 155:
        raise OptimizerJobError("optimizer source bundle receipt must bind 155 files")
    entry_paths: set[str] = set()
    for row in entries:
        path = PurePosixPath(str(row.get("path", "")))
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise OptimizerJobError("bundle receipt contains an unsafe path")
        if str(path) in entry_paths:
            raise OptimizerJobError("bundle receipt contains a duplicate path")
        entry_paths.add(str(path))
        _digest(row.get("sha256"), label="bundle entry digest")
        if not isinstance(row.get("bytes"), int) or int(row["bytes"]) < 0:
            raise OptimizerJobError("bundle receipt contains invalid bytes")
    required = {
        EXPECTED_SOURCE_CONTRACT_PATH,
        EXPECTED_MATRIX_PATH,
        EXPECTED_CONFIG_PATH,
    }
    if not required.issubset(entry_paths):
        raise OptimizerJobError("bundle receipt omits a required frozen input")
    task_prefix = EXPECTED_TASK_ROOT + "/"
    forbidden_task_paths = {
        task_prefix + "manifests/items.jsonl",
        task_prefix + "manifests/confirmation.sha256",
    }
    if forbidden_task_paths.intersection(entry_paths) or any(
        path.startswith(task_prefix) and "/confirmation_eval/" in path
        for path in entry_paths
    ):
        raise OptimizerJobError("bundle receipt contains a confirmation-capable artifact")
    discovery_images = [
        path for path in entry_paths
        if path.startswith(task_prefix + "images/") and "/discovery_eval/" in path
    ]
    train_images = [
        path for path in entry_paths
        if path.startswith(task_prefix + "images/") and "/train/" in path
    ]
    if len(discovery_images) != 24 or len(train_images) != 96:
        raise OptimizerJobError("bundle task image inventory changed")
    paths = contract.get("paths", {})
    _absolute_path(paths.get("runtime_root"), prefix="/workspace/parallax", label="runtime root")
    _absolute_path(paths.get("source_root"), prefix="/workspace/parallax/", label="source root")
    _absolute_path(paths.get("bundle_path"), prefix="/workspace/parallax/", label="bundle path")
    _absolute_path(paths.get("snapshot_root"), prefix="/root/.cache/huggingface/hub/", label="snapshot root")
    _absolute_path(paths.get("preflight_output"), prefix="/workspace/parallax/", label="preflight output")
    _absolute_path(paths.get("optimizer_output"), prefix="/workspace/parallax/", label="optimizer output")
    _absolute_path(paths.get("final_output"), prefix="/workspace/parallax/", label="final output")
    if contract.get("source_contract_sha256") != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise OptimizerJobError("optimizer source contract hash changed")
    bounds = contract.get("bounds", {})
    if bounds != {
        "parallel_downloads": 3,
        "download_timeout_seconds": 600,
        "preflight_timeout_seconds": 180,
        "optimizer_timeout_seconds": 600,
        "minimum_free_bytes": 50_000_000_000,
    }:
        raise OptimizerJobError("optimizer execution bounds changed")
    firewall = contract.get("firewall")
    if firewall != {
        "confirmation_manifest_present": False,
        "confirmation_images_present": False,
        "discovery_images_removed_before_model_load": True,
        "discovery_images_opened_by_optimizer": 0,
        "discovery_items_evaluated": 0,
        "promotion_eligible": False,
        "automatic_retry": False,
    }:
        raise OptimizerJobError("optimizer data firewall changed")
    return contract


def _download(row: dict[str, Any], destination: Path) -> dict[str, Any]:
    started = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or temporary.exists():
        raise OptimizerJobError(f"refusing to overwrite input: {destination.name}")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(str(row["url"]), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    observed = digest.hexdigest()
    if size != int(row["bytes"]) or observed != row["sha256"]:
        temporary.unlink(missing_ok=True)
        raise OptimizerJobError(f"input hash/size mismatch: {destination.name}")
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": size,
        "sha256": observed,
        "version_id": row["version_id"],
        "elapsed_seconds": time.monotonic() - started,
    }


def download_source_bundle(contract: dict[str, Any]) -> dict[str, Any]:
    bundle_path = Path(contract["paths"]["bundle_path"])
    if bundle_path.exists():
        raise OptimizerJobError("source bundle destination already exists")
    record = _download(contract["bundle"], bundle_path)
    return {
        "status": "GREEN_EXACT_VERSIONED_SOURCE_BUNDLE",
        "verified_objects": 1,
        "verified_bytes": record["bytes"],
        "record": record,
    }


def hydrate_model(contract: dict[str, Any]) -> dict[str, Any]:
    paths = contract["paths"]
    snapshot = Path(paths["snapshot_root"])
    if snapshot.exists():
        raise OptimizerJobError("model snapshot destination already exists")
    snapshot_partial = snapshot.with_name(snapshot.name + ".partial")
    if snapshot_partial.exists():
        shutil.rmtree(snapshot_partial)
    snapshot_partial.mkdir(parents=True)
    jobs = [(row, snapshot_partial / row["path"]) for row in contract["model"]]
    started = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(contract["bounds"]["parallel_downloads"])
    )
    try:
        futures = [pool.submit(_download, row, destination) for row, destination in jobs]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=int(contract["bounds"]["download_timeout_seconds"]),
        )
        if pending:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise OptimizerJobError("input hydration exceeded the frozen timeout")
        records = [future.result() for future in futures]
        pool.shutdown(wait=True)
        os.sync()
        os.replace(snapshot_partial, snapshot)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(snapshot_partial, ignore_errors=True)
        raise
    return {
        "status": "GREEN_EXACT_VERSIONED_MODEL",
        "elapsed_seconds": time.monotonic() - started,
        "parallel_downloads": contract["bounds"]["parallel_downloads"],
        "verified_objects": len(records),
        "verified_bytes": sum(int(row["bytes"]) for row in records),
        "model_files": EXPECTED_MODEL_FILES,
        "model_bytes": EXPECTED_MODEL_BYTES,
        "records": records,
    }


def validate_tar_members(members: list[tarfile.TarInfo], entries: list[dict[str, Any]]) -> None:
    expected = {str(row["path"]): int(row["bytes"]) for row in entries}
    observed: dict[str, int] = {}
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise OptimizerJobError("source archive contains path traversal")
        if member.isdir():
            continue
        if not member.isfile() or member.issym() or member.islnk():
            raise OptimizerJobError("source archive contains a non-regular entry")
        if str(path) in observed:
            raise OptimizerJobError("source archive contains a duplicate file")
        observed[str(path)] = int(member.size)
    if observed != expected:
        raise OptimizerJobError("source archive inventory differs from the frozen receipt")


def extract_and_verify_bundle(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    bundle_path = Path(contract["paths"]["bundle_path"])
    if source.exists():
        raise OptimizerJobError("source root already exists")
    partial = source.with_name(source.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    started = time.monotonic()
    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = archive.getmembers()
            validate_tar_members(members, contract["bundle"]["entries"])
            archive.extractall(partial, members=members, filter="data")
        expected = {str(row["path"]): row for row in contract["bundle"]["entries"]}
        for relative, row in expected.items():
            path = partial / relative
            if (
                not path.is_file()
                or path.stat().st_size != int(row["bytes"])
                or sha256_file(path) != row["sha256"]
            ):
                raise OptimizerJobError(f"extracted source mismatch: {relative}")
        if sha256_file(partial / EXPECTED_SOURCE_CONTRACT_PATH) != EXPECTED_SOURCE_CONTRACT_SHA256:
            raise OptimizerJobError("extracted optimizer source contract mismatch")
        os.sync()
        os.replace(partial, source)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    bundle_path.unlink()
    return {
        "status": "GREEN_EXACT_SOURCE",
        "entries": len(contract["bundle"]["entries"]),
        "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "elapsed_seconds": time.monotonic() - started,
        "bundle_deleted_after_verification": True,
    }


def _run_command(
    command: list[str], *, cwd: Path, environment: dict[str, str], timeout: int, label: str
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-16_384:]
        raise OptimizerJobError(f"{label} failed with {completed.returncode}: {tail}")
    return {
        "returncode": int(completed.returncode),
        "elapsed_seconds": time.monotonic() - started,
        "stdout_tail": completed.stdout[-4096:],
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


def _remove_discovery_images(task_root: Path) -> dict[str, Any]:
    roots = sorted(path for path in task_root.glob("images/*/discovery_eval") if path.is_dir())
    files = sorted(path for root in roots for path in root.rglob("*") if path.is_file())
    count = len(files)
    total = sum(path.stat().st_size for path in files)
    for root in roots:
        shutil.rmtree(root)
    remaining = list(task_root.glob("images/*/discovery_eval/**/*"))
    if remaining:
        raise OptimizerJobError("discovery image deletion was incomplete")
    return {
        "status": "GREEN_DISCOVERY_IMAGES_ABSENT_BEFORE_MODEL_LOAD",
        "deleted_files": count,
        "deleted_bytes": total,
        "remaining_files": 0,
        "discovery_images_opened_by_optimizer": 0,
        "discovery_items_evaluated": 0,
    }


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "safetensors", "Pillow", "PyYAML")
    return {name: importlib.metadata.version(name) for name in names}


def execute(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    cache_root = Path(contract["paths"]["snapshot_root"]).parents[2]
    task_root = source / EXPECTED_TASK_ROOT
    environment = _runtime_environment(source)
    preflight_output = Path(contract["paths"]["preflight_output"])
    optimizer_output = Path(contract["paths"]["optimizer_output"])
    source_contract = source / EXPECTED_SOURCE_CONTRACT_PATH
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
    preflight_run = _run_command(
        preflight_command,
        cwd=source,
        environment=environment,
        timeout=int(contract["bounds"]["preflight_timeout_seconds"]),
        label="H100 preflight",
    )
    preflight = json.loads(preflight_output.read_text(encoding="utf-8"))
    if preflight.get("status") != "GREEN" or int(preflight.get("science_items_evaluated", -1)) != 0:
        raise OptimizerJobError("H100 preflight result is not green/no-science")
    firewall = _remove_discovery_images(task_root)
    optimizer_command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_optimizer_scale_gate",
        "--root",
        str(source),
        "--source-contract",
        str(source_contract),
        "--matrix",
        str(source / EXPECTED_MATRIX_PATH),
        "--config",
        str(source / EXPECTED_CONFIG_PATH),
        "--cache-root",
        str(cache_root),
        "--task-root",
        str(task_root),
        "--instance-id",
        str(contract["instance_id"]),
        "--output",
        str(optimizer_output),
    ]
    optimizer_run = _run_command(
        optimizer_command,
        cwd=source,
        environment=environment,
        timeout=int(contract["bounds"]["optimizer_timeout_seconds"]),
        label="optimizer-scale microgate",
    )
    optimizer = json.loads(optimizer_output.read_text(encoding="utf-8"))
    if (
        optimizer.get("status") != "GREEN_MECHANICAL_EXECUTION"
        or optimizer.get("probe_split") != "train_only"
        or int(optimizer.get("discovery_items_read", -1)) != 0
        or int(optimizer.get("confirmation_items_read", -1)) != 0
    ):
        raise OptimizerJobError("optimizer result escapes its train-only mechanical boundary")
    return {
        "preflight": {
            "status": preflight["status"],
            "sha256": sha256_file(preflight_output),
            "runtime_versions": preflight.get("runtime_versions"),
            "cache_gate_sha256": preflight.get("cache_gate_sha256"),
            "run": preflight_run,
        },
        "firewall": firewall,
        "optimizer": optimizer,
        "optimizer_sha256": sha256_file(optimizer_output),
        "optimizer_run": optimizer_run,
        "runtime_versions_observed": _package_versions(),
    }


def run(contract_path: Path) -> dict[str, Any]:
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = validate_contract(raw)
    final_output = Path(contract["paths"]["final_output"])
    if final_output.exists():
        raise OptimizerJobError("refusing to overwrite final optimizer job result")
    free = shutil.disk_usage(final_output.parent).free
    if free < int(contract["bounds"]["minimum_free_bytes"]):
        raise OptimizerJobError("worker scratch is below the frozen free-space floor")
    started = time.monotonic()
    started_utc = utc_now()
    bundle_download = download_source_bundle(contract)
    source = extract_and_verify_bundle(contract)
    # Fail on any source/archive incompatibility before paying the network and
    # wall-time cost of the 11.7 GB model hydration.
    hydration = hydrate_model(contract)
    # Presigned input URLs are no longer needed.  Remove the only persisted copy
    # before any scientific package or model is imported.
    contract_path.unlink()
    for row in raw.get("model", []):
        row["url"] = ""
    raw.get("bundle", {})["url"] = ""
    del raw
    execution = execute(contract)
    optimizer = execution.pop("optimizer")
    packet = dict(optimizer)
    packet["execution_provenance"] = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "started_utc": started_utc,
        "wall_seconds_including_hydration": time.monotonic() - started,
        "durable_input_provider": "amazon_s3",
        "vast_storage": "ephemeral_instance_scratch_only",
        "aws_credentials_on_worker": False,
        "presigned_input_contract_deleted_before_model_import": True,
        "bundle_download": bundle_download,
        "hydration": hydration,
        "source": source,
        **execution,
        "automatic_retry": False,
        "promotion_eligible": False,
    }
    atomic_json(final_output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.contract.resolve())
    print(
        json.dumps(
            {
                "status": packet["status"],
                "verdict": packet["adjudication"]["verdict"],
                "selected_learning_rate": packet["adjudication"]["selected_learning_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
