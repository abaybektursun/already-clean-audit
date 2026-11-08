"""Run the frozen supporting-lineage optimizer microgate on disposable Vast scratch.

The worker receives only version-bound presigned S3 GETs.  It verifies the
existing PG1 source bundle first, applies six small content-addressed PG2
overlays, proves the exact prior PG2 mechanical compatibility gate, and only
then hydrates the 6.1 GB checkpoint.  Discovery images are physically removed
before the first model load.  AWS credentials and persistent Vast storage are
never used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

try:  # Package import in tests; direct sibling import on the staged worker.
    from .ephemeral_optimizer_scale_job import (
        EXPECTED_BUNDLE_BYTES,
        EXPECTED_BUNDLE_SHA256,
        EXPECTED_TASK_ROOT,
        OptimizerJobError,
        _absolute_path,
        _download,
        _package_versions,
        _remove_discovery_images,
        _run_command,
        _runtime_environment,
        _validate_object,
        atomic_json,
        download_source_bundle,
        extract_and_verify_bundle,
        sha256_file,
        utc_now,
    )
except ImportError:  # pragma: no cover - exercised by the actual staged script.
    from ephemeral_optimizer_scale_job import (  # type: ignore[no-redef]
        EXPECTED_BUNDLE_BYTES,
        EXPECTED_BUNDLE_SHA256,
        EXPECTED_TASK_ROOT,
        OptimizerJobError,
        _absolute_path,
        _download,
        _package_versions,
        _remove_discovery_images,
        _run_command,
        _runtime_environment,
        _validate_object,
        atomic_json,
        download_source_bundle,
        extract_and_verify_bundle,
        sha256_file,
        utc_now,
    )


EXPECTED_REPO = "google/paligemma2-3b-pt-224"
EXPECTED_REVISION = "96eeb174da13ca1a2b247e4d0867436296c36420"
EXPECTED_MODEL_BYTES = 6_099_504_596
EXPECTED_MODEL_FILES = 9
EXPECTED_MODEL_NAMES = {
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
EXPECTED_SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_optimizer_scale_source_contract_pg2_v1.json"
)
EXPECTED_SOURCE_CONTRACT_SHA256 = (
    "9a16e2c22aaca9d4752678025534cf32de0ee15ad827e0276abee7e41d9ec2eb"
)
EXPECTED_CONFIG_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_optimizer_scale_gate_pg2_v1.yaml"
)
EXPECTED_MATRIX_PATH = "studies/lineage/cross-lineage/config/parallax_matrix.yaml"
EXPECTED_HYDRATION_PATH = (
    "studies/lineage/cross-lineage/config/parallax_pg2_h100_s3_hydration_job_v1.json"
)
EXPECTED_PRIOR_GATE_PATH = (
    "studies/lineage/cross-lineage/results/parallax_training_real_gate_v1/gate.json"
)
EXPECTED_PRIOR_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/parallax_training_real_gate_source_contract_v2.json"
)
EXPECTED_OVERLAYS = {
    "infra/vast/parallax/training_optimizer_scale_gate_pg2.py": (
        "502f85d3f87f615ba026910d9b3b56a1ae9330a67fc5731c9676564fbc8eacb0"
    ),
    EXPECTED_CONFIG_PATH: (
        "5556c0c031b6ab628ad43246943073911330f4a8262db77f7a329096056fb585"
    ),
    EXPECTED_SOURCE_CONTRACT_PATH: EXPECTED_SOURCE_CONTRACT_SHA256,
    EXPECTED_HYDRATION_PATH: (
        "709b84e2b0276a3afcd9c9265e6b1f8e54d6f24f845d0b696cfe2285160e893a"
    ),
    EXPECTED_PRIOR_GATE_PATH: (
        "41c3220bc1e056521b1fc6e7325ecb26a981d681c39b65cb831d8e6e3e508a79"
    ),
    EXPECTED_PRIOR_CONTRACT_PATH: (
        "039410c0d95499c80934af285caab768383c07a361e884e7b838871dc5ebcb66"
    ),
}
EXPECTED_BOUNDS = {
    "parallel_downloads": 3,
    "download_timeout_seconds": 600,
    "optimizer_timeout_seconds": 600,
    "minimum_free_bytes": 50_000_000_000,
}
EXPECTED_FIREWALL = {
    "confirmation_manifest_present": False,
    "confirmation_images_present": False,
    "discovery_images_removed_before_model_load": True,
    "discovery_images_opened_by_optimizer": 0,
    "discovery_items_evaluated": 0,
    "prior_pg2_dynamic_gate_verified": True,
    "promotion_eligible": False,
    "automatic_retry": False,
}


def _safe_relative(value: object, *, label: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise OptimizerJobError(f"unsafe {label} path")
    return str(path)


def _validate_bundle_receipt(bundle: dict[str, Any]) -> set[str]:
    if (
        int(bundle.get("bytes", 0)) != EXPECTED_BUNDLE_BYTES
        or bundle.get("sha256") != EXPECTED_BUNDLE_SHA256
    ):
        raise OptimizerJobError("PG2 optimizer base bundle identity changed")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != 155:
        raise OptimizerJobError("PG2 optimizer base bundle receipt changed")
    paths: set[str] = set()
    for row in entries:
        path = _safe_relative(row.get("path"), label="bundle receipt")
        if path in paths:
            raise OptimizerJobError("base bundle receipt contains a duplicate path")
        paths.add(path)
        digest = str(row.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise OptimizerJobError("base bundle receipt contains an invalid digest")
        if not isinstance(row.get("bytes"), int) or int(row["bytes"]) < 0:
            raise OptimizerJobError("base bundle receipt contains invalid bytes")
    task_prefix = EXPECTED_TASK_ROOT + "/"
    if {
        task_prefix + "manifests/items.jsonl",
        task_prefix + "manifests/confirmation.sha256",
    }.intersection(paths):
        raise OptimizerJobError("base bundle contains a confirmation-capable manifest")
    if any("/confirmation_eval/" in path for path in paths):
        raise OptimizerJobError("base bundle contains confirmation images")
    if len([path for path in paths if "/discovery_eval/" in path]) != 24:
        raise OptimizerJobError("base bundle discovery image inventory changed")
    if len([path for path in paths if "/train/" in path]) != 96:
        raise OptimizerJobError("base bundle train image inventory changed")
    return paths


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "stage",
        "provider",
        "lineage",
        "instance_id",
        "storage",
        "model",
        "bundle",
        "overlays",
        "source_contract_sha256",
        "paths",
        "bounds",
        "firewall",
    }
    if set(contract) != expected_top or contract.get("schema_version") != 1:
        raise OptimizerJobError("PG2 optimizer job contract schema changed")
    if (
        contract.get("status") != "FROZEN_H100_OPTIMIZER_SCALE_PG2_JOB"
        or contract.get("stage") != "training_optimizer_scale_microgate"
        or contract.get("provider") != "vast.ai"
        or contract.get("lineage")
        != {
            "id": "pg2",
            "role": "supporting",
            "repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
        }
    ):
        raise OptimizerJobError("PG2 optimizer job escapes its supporting-lineage scope")
    instance_id = contract.get("instance_id")
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise OptimizerJobError("PG2 optimizer job instance ID is absent")
    if contract.get("storage") != {
        "durable_provider": "amazon_s3",
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "aws_credentials_on_worker": False,
    }:
        raise OptimizerJobError("PG2 optimizer storage contract is not S3-only")

    model = contract.get("model")
    if not isinstance(model, list) or len(model) != EXPECTED_MODEL_FILES:
        raise OptimizerJobError("PG2 optimizer must bind all nine model files")
    names = set()
    for row in model:
        _validate_object(row, kind="PG2 model")
        name = _safe_relative(row.get("path"), label="model")
        if len(PurePosixPath(name).parts) != 1:
            raise OptimizerJobError("PG2 model filename escapes the snapshot root")
        names.add(name)
    if names != EXPECTED_MODEL_NAMES or sum(int(row["bytes"]) for row in model) != EXPECTED_MODEL_BYTES:
        raise OptimizerJobError("PG2 optimizer model inventory changed")

    bundle = _validate_object(contract.get("bundle", {}), kind="base bundle")
    base_paths = _validate_bundle_receipt(bundle)
    overlays = contract.get("overlays")
    if not isinstance(overlays, list) or len(overlays) != len(EXPECTED_OVERLAYS):
        raise OptimizerJobError("PG2 optimizer must bind exactly six overlays")
    observed_overlays: dict[str, str] = {}
    for row in overlays:
        _validate_object(row, kind="PG2 overlay")
        path = _safe_relative(row.get("path"), label="overlay")
        if path in observed_overlays or path in base_paths:
            raise OptimizerJobError("PG2 overlay duplicates another source path")
        observed_overlays[path] = str(row["sha256"])
    if observed_overlays != EXPECTED_OVERLAYS:
        raise OptimizerJobError("PG2 optimizer overlay identities changed")

    paths = contract.get("paths", {})
    for label, prefix in (
        ("runtime_root", "/workspace/parallax"),
        ("source_root", "/workspace/parallax/"),
        ("bundle_path", "/workspace/parallax/"),
        ("overlay_root", "/workspace/parallax/"),
        ("snapshot_root", "/root/.cache/huggingface/hub/"),
        ("preflight_output", "/workspace/parallax/"),
        ("optimizer_output", "/workspace/parallax/"),
        ("final_output", "/workspace/parallax/"),
    ):
        _absolute_path(paths.get(label), prefix=prefix, label=label)
    expected_snapshot = (
        "/root/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/"
        f"snapshots/{EXPECTED_REVISION}"
    )
    if paths.get("snapshot_root") != expected_snapshot:
        raise OptimizerJobError("PG2 optimizer snapshot destination changed")
    if contract.get("source_contract_sha256") != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise OptimizerJobError("PG2 optimizer source contract hash changed")
    if contract.get("bounds") != EXPECTED_BOUNDS:
        raise OptimizerJobError("PG2 optimizer execution bounds changed")
    if contract.get("firewall") != EXPECTED_FIREWALL:
        raise OptimizerJobError("PG2 optimizer data firewall changed")
    return contract


def download_and_apply_overlays(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    staging = Path(contract["paths"]["overlay_root"])
    if staging.exists():
        raise OptimizerJobError("PG2 overlay staging destination already exists")
    staging.mkdir(parents=True)
    records = []
    try:
        for row in contract["overlays"]:
            relative = _safe_relative(row["path"], label="overlay")
            record = _download(row, staging / relative)
            destination = source / relative
            if destination.exists():
                raise OptimizerJobError(f"PG2 overlay would overwrite base source: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)
            records.append({**record, "path": relative})
        shutil.rmtree(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if sha256_file(source / EXPECTED_SOURCE_CONTRACT_PATH) != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise OptimizerJobError("applied PG2 source contract identity changed")
    return {
        "status": "GREEN_EXACT_PG2_OVERLAYS",
        "verified_objects": len(records),
        "verified_bytes": sum(int(row["bytes"]) for row in records),
        "records": records,
    }


def _manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("path", "bytes", "sha256", "bucket", "region", "key", "version_id")
    return [{name: row[name] for name in fields} for row in sorted(rows, key=lambda item: item["path"])]


def verify_source_and_prerequisite(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    source_contract_path = source / EXPECTED_SOURCE_CONTRACT_PATH
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    if (
        source_contract.get("status") != "FROZEN"
        or source_contract.get("cache_scope") != "pg2_supporting_optimizer_scale_only"
    ):
        raise OptimizerJobError("PG2 optimizer source contract is not exact/frozen")
    for name, row in source_contract["artifacts"].items():
        if sha256_file(source / row["path"]) != row["sha256"]:
            raise OptimizerJobError(f"PG2 source artifact drift: {name}")
    checkpoint = source_contract.get("checkpoint", {})
    hydration_row = checkpoint.get("input_manifest", {})
    if (
        checkpoint.get("repo") != EXPECTED_REPO
        or checkpoint.get("revision") != EXPECTED_REVISION
        or hydration_row.get("path") != EXPECTED_HYDRATION_PATH
        or sha256_file(source / hydration_row["path"]) != hydration_row.get("sha256")
    ):
        raise OptimizerJobError("PG2 checkpoint/source binding changed")
    hydration = json.loads((source / EXPECTED_HYDRATION_PATH).read_text(encoding="utf-8"))
    if (
        hydration.get("status") != "FROZEN_INPUT_MANIFEST"
        or hydration.get("repo") != EXPECTED_REPO
        or hydration.get("revision") != EXPECTED_REVISION
        or hydration.get("expected_file_count") != EXPECTED_MODEL_FILES
        or hydration.get("expected_total_bytes") != EXPECTED_MODEL_BYTES
        or _manifest_rows(hydration.get("files", [])) != _manifest_rows(contract["model"])
    ):
        raise OptimizerJobError("PG2 job objects differ from the staged checkpoint manifest")

    prerequisite = source_contract.get("mechanical_prerequisite", {})
    result_row = prerequisite.get("result", {})
    prior_contract_row = prerequisite.get("source_contract", {})
    if (
        result_row.get("path") != EXPECTED_PRIOR_GATE_PATH
        or prior_contract_row.get("path") != EXPECTED_PRIOR_CONTRACT_PATH
        or sha256_file(source / result_row["path"]) != result_row.get("sha256")
        or sha256_file(source / prior_contract_row["path"]) != prior_contract_row.get("sha256")
        or prerequisite.get("scientific_claim") is not False
    ):
        raise OptimizerJobError("prior PG2 mechanical prerequisite binding changed")
    prior_contract = json.loads((source / EXPECTED_PRIOR_CONTRACT_PATH).read_text(encoding="utf-8"))
    current_to_prior = {
        "training_model": "training_model",
        "training_common": "training_common",
        "runtime_common": "g1_common",
        "entities": "entities",
        "tasks": "parallax_tasks",
        "requirements": "requirements",
        "matrix": "matrix",
        "discovery_manifest": "discovery_manifest",
    }
    for current_name, prior_name in current_to_prior.items():
        if source_contract["artifacts"][current_name]["sha256"] != prior_contract["artifacts"][prior_name]["sha256"]:
            raise OptimizerJobError(f"PG2 prerequisite code/data drift: {current_name}")
    prior = json.loads((source / EXPECTED_PRIOR_GATE_PATH).read_text(encoding="utf-8"))
    pg2 = prior.get("lineages", {}).get("pg2", {})
    update = pg2.get("update", {})
    report = pg2.get("trainable_contract", {})
    runtime = prior.get("runtime", {})
    gradient = float(update.get("gradient_norm", float("nan")))
    loss = float(update.get("loss", float("nan")))
    if (
        prior.get("status") != "GREEN"
        or pg2.get("checkpoint") != {"repo": EXPECTED_REPO, "revision": EXPECTED_REVISION}
        or pg2.get("zero_delta_logits_bit_exact") is not True
        or pg2.get("attention") != ["eager", "eager", "eager"]
        or not math.isfinite(gradient)
        or gradient <= 0
        or not math.isfinite(loss)
        or report.get("status") != "GREEN"
        or report.get("lora_parameter_count") != 5_191_680
        or report.get("trainable_parameter_count") != 7_848_192
        or report.get("trainable_dtypes") != ["torch.float32"]
        or any("vision_tower" in name for name in report.get("trainable_names", []))
        or runtime.get("gpu") != "NVIDIA H100 NVL"
        or runtime.get("torch") != "2.13.0+cu130"
        or runtime.get("transformers") != "5.5.4"
        or runtime.get("peft") != "0.19.1"
    ):
        raise OptimizerJobError("prior exact PG2 dynamic compatibility gate is not green")
    return {
        "status": "GREEN_SOURCE_AND_PRIOR_PG2_DYNAMIC_GATE",
        "source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "prior_gate_sha256": result_row["sha256"],
        "prior_gradient_norm": gradient,
        "prior_zero_delta_logits_bit_exact": True,
        "same_training_code_and_task_hashes": True,
        "scientific_claim": False,
    }


def hydrate_model(contract: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(contract["paths"]["snapshot_root"])
    if snapshot.exists():
        raise OptimizerJobError("PG2 model snapshot destination already exists")
    partial = snapshot.with_name(snapshot.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    jobs = [(row, partial / row["path"]) for row in contract["model"]]
    started = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=contract["bounds"]["parallel_downloads"])
    try:
        futures = [pool.submit(_download, row, destination) for row, destination in jobs]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=contract["bounds"]["download_timeout_seconds"],
        )
        if pending:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise OptimizerJobError("PG2 model hydration exceeded the frozen timeout")
        records = [future.result() for future in futures]
        pool.shutdown(wait=True)
        os.sync()
        os.replace(partial, snapshot)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {
        "status": "GREEN_EXACT_VERSIONED_PG2_MODEL",
        "elapsed_seconds": time.monotonic() - started,
        "parallel_downloads": contract["bounds"]["parallel_downloads"],
        "verified_objects": len(records),
        "verified_bytes": sum(int(row["bytes"]) for row in records),
        "model_files": EXPECTED_MODEL_FILES,
        "model_bytes": EXPECTED_MODEL_BYTES,
        "records": records,
    }


def _required_versions(source: Path) -> dict[str, str]:
    required = {}
    for line in (source / "infra/vast/parallax/requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        if name in {"torch", "transformers", "peft", "safetensors", "pillow", "pyyaml"}:
            required[{"pillow": "Pillow", "pyyaml": "PyYAML"}.get(name, name)] = version
    return required


def runtime_preflight(contract: dict[str, Any], source_proof: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise OptimizerJobError("PG2 optimizer destination is not an H100")
    source = Path(contract["paths"]["source_root"])
    observed_versions = _package_versions()
    if observed_versions != _required_versions(source):
        raise OptimizerJobError("PG2 optimizer runtime package versions changed")
    task_root = source / EXPECTED_TASK_ROOT
    forbidden = [
        task_root / "manifests/items.jsonl",
        task_root / "manifests/confirmation.sha256",
    ]
    if any(path.exists() for path in forbidden) or list(task_root.glob("images/*/confirmation_eval/**/*")):
        raise OptimizerJobError("confirmation-capable task payload is present")
    train = sorted(task_root.glob("images/*/train/*.png"))
    discovery = sorted(task_root.glob("images/*/discovery_eval/*.png"))
    if len(train) != 96 or len(discovery) != 24:
        raise OptimizerJobError("PG2 optimizer task image inventory changed")
    cache_root = Path(contract["paths"]["snapshot_root"]).parents[2]
    encoded = json.dumps(
        _manifest_rows(contract["model"]), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    content_sha256 = hashlib.sha256(encoded).hexdigest()
    atomic_json(
        cache_root / ".parallax_cache_gate.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN",
            "destination_instance_id": contract["instance_id"],
            "content_manifest_sha256": content_sha256,
            "network_cache_mutation_after_this_marker": "FORBIDDEN",
            "scope": "pg2_supporting_optimizer_scale_only",
            "repository": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
        },
    )
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_NO_SCIENCE_PG2_OPTIMIZER_PREFLIGHT",
        "provider": "vast.ai",
        "instance_id": contract["instance_id"],
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "runtime_versions": observed_versions,
        "verified_model_files": EXPECTED_MODEL_FILES,
        "verified_model_bytes": EXPECTED_MODEL_BYTES,
        "train_images_present": len(train),
        "discovery_images_present_before_physical_removal": len(discovery),
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "cache_gate_sha256": content_sha256,
        "source_proof": source_proof,
    }
    atomic_json(Path(contract["paths"]["preflight_output"]), packet)
    return packet


def execute(contract: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    source = Path(contract["paths"]["source_root"])
    cache_root = Path(contract["paths"]["snapshot_root"]).parents[2]
    task_root = source / EXPECTED_TASK_ROOT
    firewall = _remove_discovery_images(task_root)
    environment = _runtime_environment(source)
    output = Path(contract["paths"]["optimizer_output"])
    command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_optimizer_scale_gate_pg2",
        "--root",
        str(source),
        "--source-contract",
        str(source / EXPECTED_SOURCE_CONTRACT_PATH),
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
        str(output),
    ]
    run_record = _run_command(
        command,
        cwd=source,
        environment=environment,
        timeout=contract["bounds"]["optimizer_timeout_seconds"],
        label="PG2 optimizer-scale microgate",
    )
    optimizer = json.loads(output.read_text(encoding="utf-8"))
    if (
        optimizer.get("status") != "GREEN_MECHANICAL_EXECUTION"
        or optimizer.get("lineage") != "pg2"
        or optimizer.get("role") != "supporting"
        or optimizer.get("probe_split") != "train_only"
        or optimizer.get("discovery_items_read") != 0
        or optimizer.get("confirmation_items_read") != 0
        or optimizer.get("adjudication", {}).get("promotion_eligible") is not False
    ):
        raise OptimizerJobError("PG2 optimizer result escapes its supporting mechanical boundary")
    return {
        "optimizer": optimizer,
        "optimizer_sha256": sha256_file(output),
        "optimizer_run": run_record,
        "preflight": preflight,
        "firewall": firewall,
        "runtime_versions_observed": _package_versions(),
    }


def run(contract_path: Path) -> dict[str, Any]:
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = validate_contract(raw)
    final_output = Path(contract["paths"]["final_output"])
    if final_output.exists():
        raise OptimizerJobError("refusing to overwrite final PG2 optimizer result")
    if shutil.disk_usage(final_output.parent).free < contract["bounds"]["minimum_free_bytes"]:
        raise OptimizerJobError("worker scratch is below the frozen free-space floor")
    started = time.monotonic()
    started_utc = utc_now()
    bundle_download = download_source_bundle(contract)
    base_source = extract_and_verify_bundle(contract)
    overlays = download_and_apply_overlays(contract)
    source_proof = verify_source_and_prerequisite(contract)
    hydration = hydrate_model(contract)
    contract_path.unlink()
    for row in raw.get("model", []):
        row["url"] = ""
    raw.get("bundle", {})["url"] = ""
    for row in raw.get("overlays", []):
        row["url"] = ""
    del raw
    preflight = runtime_preflight(contract, source_proof)
    execution = execute(contract, preflight)
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
        "base_source": base_source,
        "overlays": overlays,
        "hydration": hydration,
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
