"""Launch the S3-durable PG2 seed-17 range cell on the cheapest eligible H100."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .ephemeral_optimizer_scale_launch import (
    RUNTIME_IMAGE,
    RUNTIME_TEMPLATE,
    _guardian_contract,
    require_machine_quarantine,
    select_cheapest_h100_offer,
    wait_h100_ssh,
)
from .ephemeral_pilot_job import (
    PG2_MODEL_BYTES,
    PG2_MODEL_NAMES,
    PG2_V2_OVERLAY_IDENTITIES as OVERLAY_IDENTITIES,
    PG2_V2_RUN_ID as RUN_ID,
    PG2_V2_SOURCE_CONTRACT_SHA256 as SOURCE_CONTRACT_SHA256,
    validate_contract as validate_job_contract,
)
from .ephemeral_pilot_worker import (
    EXPECTED_COMMAND,
    validate_contract as validate_worker_contract,
)
from .ephemeral_s3_launch import (
    BUCKET,
    REGION,
    _aws_credentials,
    _aws_json,
    _claim,
    _confirm_account_zero,
    _create_arguments,
    _finalize_lifecycle,
    _require_guardian_live,
    _stage,
    _stage_atomic,
    _wait_remote_file,
    _wait_remote_json_exact,
    _wait_result,
    require_s3_safety,
    select_cheapest_offer,
)
from .ephemeral_pg2_optimizer_scale_launch import (
    _record_launch_state,
    _worker_assignment,
    _worker_assignment_ack,
)
from .guarded_launch import _create_instance, _run, _ssh_base, wait_ssh
from .pg2_range_source_gate import validate_local as validate_range_source
from .s3_presign_post import presign_artifact_post
from .s3_presign_put import presign_get, presign_result_put


class PilotLaunchError(RuntimeError):
    pass


def _digest(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PilotLaunchError(f"{label} is not a lowercase SHA-256")
    return digest


def _resolve_bound(root: Path, row: dict[str, Any], *, label: str) -> Path:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PilotLaunchError(f"unsafe {label} path")
    path = (root / relative).resolve()
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise PilotLaunchError(f"{label} hash mismatch")
    return path


def _validate_input(row: dict[str, Any], *, label: str) -> dict[str, Any]:
    digest = _digest(row.get("sha256"), label=f"{label} digest")
    key = str(row.get("key", ""))
    if (
        row.get("bucket") != BUCKET
        or row.get("region") != REGION
        or not key.startswith(f"parallax/input-staging/sha256/{digest}/")
        or ".." in key.split("/")
        or not str(row.get("version_id", ""))
        or not isinstance(row.get("bytes"), int)
        or int(row["bytes"]) <= 0
    ):
        raise PilotLaunchError(f"{label} is not an exact versioned S3 input")
    return row


def _model_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_bound(root, spec["inputs"]["model_manifest"], label="model manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if (
        manifest.get("status") != "FROZEN_INPUT_MANIFEST"
        or manifest.get("scope") != "s3_worker_cache_hydration_only"
        or manifest.get("lineage") != "pg2"
        or manifest.get("role") != "supporting_fused_base"
        or manifest.get("repo") != "google/paligemma2-3b-pt-224"
        or manifest.get("revision") != "96eeb174da13ca1a2b247e4d0867436296c36420"
        or manifest.get("expected_total_bytes") != PG2_MODEL_BYTES
        or manifest.get("worker_snapshot_root")
        != (
            "/root/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/"
            "snapshots/96eeb174da13ca1a2b247e4d0867436296c36420"
        )
        or manifest.get("confirmation_items_read") != 0
        or manifest.get("science_items_evaluated") != 0
        or manifest.get("model_loads") != 0
        or manifest.get("gpu_work") != 0
        or not isinstance(rows, list)
        or len(rows) != 9
    ):
        raise PilotLaunchError("PG2 model input manifest is not exact")
    for index, row in enumerate(rows):
        _validate_input(row, label=f"model {index}")
    if (
        {str(row.get("path", "")) for row in rows} != PG2_MODEL_NAMES
        or sum(int(row["bytes"]) for row in rows) != PG2_MODEL_BYTES
    ):
        raise PilotLaunchError("PG2 model input inventory changed")
    return rows


def _overlay_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_bound(root, spec["inputs"]["overlay_receipt"], label="overlay receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    rows = receipt.get("objects")
    if (
        receipt.get("status") != "GREEN_PG2_SEED17_RANGE_OVERLAYS_S3"
        or receipt.get("run_id") != RUN_ID
        or receipt.get("lineage") != "pg2"
        or receipt.get("role") != "supporting"
        or receipt.get("aws_profile") != "fuelos"
        or receipt.get("region") != REGION
        or receipt.get("bucket") != BUCKET
        or receipt.get("object_count") != 8
        or not isinstance(rows, list)
        or len(rows) != 8
    ):
        raise PilotLaunchError("pilot overlay receipt is not exact")
    observed = {}
    for index, row in enumerate(rows):
        _validate_input(row, label=f"overlay {index}")
        observed[str(row.get("path", ""))] = str(row["sha256"])
        if row.get("controller_head_exact_match") is not True:
            raise PilotLaunchError("pilot overlay was not independently HEAD verified")
    if observed != OVERLAY_IDENTITIES:
        raise PilotLaunchError("pilot overlay identity set changed")
    return rows


def _bundle(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _validate_input(spec["inputs"]["bundle"], label="source bundle")
    receipt_path = _resolve_bound(root, spec["inputs"]["bundle_receipt"], label="bundle receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("bundle_sha256") != row["sha256"]
        or receipt.get("bundle_bytes") != row["bytes"]
        or not isinstance(receipt.get("entries"), list)
        or len(receipt["entries"]) != 155
        or receipt.get("confirmation_image_included") is not False
        or receipt.get("confirmation_manifest_included") is not False
    ):
        raise PilotLaunchError("source bundle receipt is not exact/no-confirmation")
    return row, receipt


def validate_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "created_utc",
        "program",
        "status",
        "scope",
        "run_id",
        "provider",
        "automatic_paid_retry",
        "authorization",
        "storage",
        "ssh_private_key",
        "scoped_lifecycle_api_key_file",
        "worker_image",
        "worker_registry_auth",
        "guardian",
        "worker",
        "job",
        "inputs",
        "artifact",
        "result",
        "one_shot_authorization",
        "output_receipt",
        "bound_artifacts",
        "implementation",
        "invariants",
        "promotion_boundary",
    }
    if set(spec) != expected_top or spec.get("schema_version") != 1:
        raise PilotLaunchError("pilot launch spec schema changed")
    if (
        spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_H100"
        or spec.get("program") != "parallax"
        or spec.get("scope") != "pg2_supporting_seed17_nonpromotion_range_screen"
        or spec.get("provider") != "vast.ai"
        or spec.get("run_id") != RUN_ID
        or spec.get("one_shot_authorization", {}).get("id") != RUN_ID
        or spec.get("automatic_paid_retry") is not False
    ):
        raise PilotLaunchError("PG2 range launch escapes its one-shot supporting scope")
    if spec.get("authorization") != {
        "paid_pilot": True,
        "h100": True,
        "model_reads": True,
        "train_data_reads": True,
        "discovery_evaluation": True,
        "training": True,
        "confirmation": False,
        "promotion": False,
        "PG2": True,
        "broad_run": False,
        "science_claim": False,
    }:
        raise PilotLaunchError("pilot authorization changed")
    if spec.get("storage") != {
        "durable_provider": "amazon_s3",
        "aws_profile": "fuelos",
        "region": REGION,
        "bucket": BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": 80,
    }:
        raise PilotLaunchError("pilot storage boundary changed")
    if spec.get("worker_image") != RUNTIME_IMAGE:
        raise PilotLaunchError("pilot runtime image changed")
    worker = spec.get("worker", {})
    if (
        worker.get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time"
        or worker.get("gpu_kind") != "single_h100_80gb_or_larger"
        or worker.get("launch_mode") != "managed_template_private_runtime"
        or worker.get("image") != RUNTIME_IMAGE
        or worker.get("template_hash") != RUNTIME_TEMPLATE
        or int(worker.get("disk_gb", 0)) != 80
        or int(worker.get("minimum_gpu_ram_mb", 0)) != 80000
        or int(worker.get("hard_destroy_seconds", 0)) > 1200
    ):
        raise PilotLaunchError("pilot H100 construction changed")
    job = spec.get("job", {})
    if (
        job.get("stage") != "training_artifact_rehearsal"
        or job.get("maximum_seconds") != 900
        or job.get("command") != EXPECTED_COMMAND
        or job.get("environment") != {}
        or job.get("bounds")
        != {
            "parallel_downloads": 3,
            "download_timeout_seconds": 600,
            "preflight_timeout_seconds": 180,
            "pilot_timeout_seconds": 600,
            "figure_timeout_seconds": 30,
            "artifact_upload_timeout_seconds": 300,
            "minimum_free_bytes": 50_000_000_000,
        }
    ):
        raise PilotLaunchError("pilot job or execution bounds changed")
    if spec.get("artifact") != {
        "key": f"parallax/results/{RUN_ID}/artifacts.tar",
        "minimum_bytes": 100_000_000,
        "maximum_bytes": 1_100_000_000,
        "post_policy_seconds": 1800,
        "required_http_status": 201,
        "required_version_id": True,
        "required_checksum": "SHA256",
        "required_encryption": "AES256",
    }:
        raise PilotLaunchError("pilot artifact boundary changed")
    if spec.get("result") != {
        "key": f"parallax/results/{RUN_ID}/terminal.json",
        "maximum_bytes": 1024 * 1024,
        "put_url_seconds": 1800,
        "wait_seconds": 1100,
        "required_http_status": 200,
        "required_version_id": True,
        "required_encryption": "AES256",
        "archive_durable_action": "destroy",
        "archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
    }:
        raise PilotLaunchError("pilot terminal/lifecycle boundary changed")
    for label, row in spec.get("bound_artifacts", {}).items():
        _resolve_bound(root, row, label=f"bound artifact {label}")
    for label, row in spec.get("implementation", {}).items():
        _resolve_bound(root, row, label=f"implementation {label}")
    validate_range_source(root)
    _model_rows(root, spec)
    _overlay_rows(root, spec)
    _bundle(root, spec)
    require_machine_quarantine(root, spec)
    if not isinstance(spec.get("invariants"), list) or set(spec["invariants"]) != {
        "supporting_pg2_only_not_primary_chain",
        "PG2_LR_1e-5_selected_independently_not_imported_from_PG1",
        "same_scientific_task_split_items_seed_adapter_steps_snapshots_thresholds_as_PG1_seed17",
        "raw_cross_generation_objective_comparison_forbidden",
        "confirmation_reads_zero",
        "nonpromotion",
        "no_automatic_retry",
        "cheapest_current_eligible_H100_all_regions",
        "no_Vast_persistent_storage",
        "artifact_archive_durable_before_destroy",
        "archive_not_durable_stops_GPU_and_preserves_scratch",
        "controller_HEAD_exact_archive_and_terminal_versions",
        "guardian_acknowledges_exact_worker_ID_before_H100_SSH",
        "known_instance_cleanup_and_account_zero_each_require_three_observations",
        "exact_80_GB_disposable_scratch_no_blanket_allowance",
    }:
        raise PilotLaunchError("pilot invariants changed")
    return spec


def validate_execution_freeze(
    root: Path, spec_path: Path, freeze_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "FROZEN_ONE_SHOT_PG2_SEED17_RANGE_AFTER_NO_CREATE_GREEN"
        or freeze.get("run_id") != RUN_ID
        or freeze.get("automatic_paid_retry") is not False
        or freeze.get("authorization")
        != {
            "create_vast_instances": True,
            "create_vast_volumes": False,
            "h100": True,
            "model_reads": True,
            "train_data_reads": True,
            "discovery_evaluation": True,
            "training": True,
            "confirmation": False,
            "promotion": False,
            "PG2": True,
            "broad_run": False,
            "automatic_retry": False,
        }
        or freeze.get("launch_spec", {}).get("path")
        != str(spec_path.resolve().relative_to(root.resolve()))
        or freeze.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
    ):
        raise PilotLaunchError("PG2 range paid execution freeze is absent or not exact")
    no_create_path = _resolve_bound(
        root, freeze.get("no_create_rehearsal", {}), label="paid no-create rehearsal"
    )
    no_create = json.loads(no_create_path.read_text(encoding="utf-8"))
    if (
        no_create.get("status") != "GREEN_PILOT_NO_CREATE_REHEARSAL"
        or no_create.get("run_id") != RUN_ID
        or no_create.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
        or no_create.get("mutation")
        != {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "s3_objects_written": 0,
            "files_downloaded": 0,
            "gpu_operations": 0,
        }
        or no_create.get("paid_authorization") is not False
    ):
        raise PilotLaunchError("pilot paid freeze lacks an exact zero-mutation rehearsal")
    planted_path = _resolve_bound(root, freeze.get("planted_tests", {}), label="paid plants")
    planted = json.loads(planted_path.read_text(encoding="utf-8"))
    if (
        planted.get("status") != "GREEN"
        or planted.get("exit_code") != 0
        or planted.get("paid_action") is not False
        or planted.get("confirmation_items_read") != 0
    ):
        raise PilotLaunchError("pilot paid freeze lacks exact green plants")
    return freeze


def _require_input_heads(
    model: list[dict[str, Any]], bundle: dict[str, Any], overlays: list[dict[str, Any]]
) -> dict[str, Any]:
    verified = []
    for row in [*model, bundle, *overlays]:
        head = _aws_json(
            "s3api",
            "head-object",
            "--bucket",
            BUCKET,
            "--key",
            row["key"],
            "--version-id",
            row["version_id"],
            "--checksum-mode",
            "ENABLED",
        )
        expected_checksum = base64.b64encode(bytes.fromhex(row["sha256"])).decode("ascii")
        if (
            int(head.get("ContentLength", -1)) != int(row["bytes"])
            or head.get("VersionId") != row["version_id"]
            or head.get("ServerSideEncryption") != "AES256"
            or head.get("ChecksumSHA256") != expected_checksum
            or head.get("Metadata", {}).get("sha256") != row["sha256"]
        ):
            raise PilotLaunchError(f"immutable S3 input HEAD failed: {row['key']}")
        verified.append(
            {
                "key": row["key"],
                "version_id": row["version_id"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "server_side_encryption": "AES256",
            }
        )
    return {
        "status": "GREEN_EXACT_IMMUTABLE_INPUTS",
        "objects": len(verified),
        "bytes": sum(int(row["bytes"]) for row in verified),
        "verified": verified,
    }


def _require_result_absent(key: str) -> None:
    listing = _aws_json(
        "s3api",
        "list-object-versions",
        "--bucket",
        BUCKET,
        "--prefix",
        key,
        "--max-keys",
        "2",
    )
    matches = [
        row
        for group in ("Versions", "DeleteMarkers")
        for row in listing.get(group, [])
        if row.get("Key") == key
    ]
    if matches:
        raise PilotLaunchError(f"one-shot result key already exists: {key}")


def _signed(row: dict[str, Any], credentials: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        **row,
        "url": presign_get(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            bucket=BUCKET,
            key=row["key"],
            version_id=row["version_id"],
            region=REGION,
            now=now,
            expires_seconds=1800,
        ),
    }


def _job_contract(
    spec: dict[str, Any],
    *,
    instance_id: int,
    model: list[dict[str, Any]],
    bundle: dict[str, Any],
    bundle_receipt: dict[str, Any],
    overlays: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_SELECTED_PILOT_JOB",
        "stage": "training_artifact_rehearsal",
        "provider": "vast.ai",
        "run_id": RUN_ID,
        "instance_id": instance_id,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "ephemeral_scratch_gb": 80,
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": {**bundle, "entries": bundle_receipt["entries"]},
        "overlays": overlays,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/"
                "snapshots/96eeb174da13ca1a2b247e4d0867436296c36420"
            ),
            "preflight_output": "/workspace/parallax/preflight.json",
            "pilot_output_dir": (
                "/workspace/parallax/source/studies/lineage/cross-lineage/results/"
                "parallax_training_range_screen_pg2_seed17_v2"
            ),
            "figure_output": (
                "/workspace/parallax/source/studies/lineage/cross-lineage/results/"
                "parallax_training_range_screen_pg2_seed17_v2/pilot.svg"
            ),
            "manifest_output": "/workspace/parallax/artifact-manifest.json",
            "archive_output": "/workspace/parallax/artifacts.tar",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": spec["job"]["bounds"],
        "firewall": {
            "supporting_lineage_only": True,
            "confirmation_manifest_present": False,
            "confirmation_images_present": False,
            "train_items": 96,
            "discovery_items": 24,
            "confirmation_items_read": 0,
            "promotion_eligible": False,
            "automatic_retry": False,
        },
    }


def _worker_contract(spec: dict[str, Any], *, instance_id: int, result_url: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_SELECTED_PILOT_WORKER",
        "run_id": RUN_ID,
        "stage": "training_artifact_rehearsal",
        "instance_id": instance_id,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": EXPECTED_COMMAND,
        "maximum_job_seconds": 900,
        "maximum_result_bytes": 1024 * 1024,
        "result_key": spec["result"]["key"],
        "result_put_url": result_url,
        "archive_durable_action": "destroy",
        "archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
        "automatic_retry": False,
        "environment": {},
    }


def _pilot_worker_start_command() -> str:
    return (
        "set -eu; mkdir -p /workspace/parallax; "
        "nohup python3 /opt/parallax/ephemeral_pilot_worker.py --contract "
        "/run/parallax/worker-contract.json >/workspace/parallax/worker.log 2>&1 </dev/null & "
        "worker_pid=$!; kill -0 $worker_pid 2>/dev/null || test -s /workspace/parallax/terminal.json"
    )


def _artifact_head(proof: dict[str, Any]) -> dict[str, Any]:
    expected_checksum = base64.b64encode(bytes.fromhex(str(proof.get("sha256", "")))).decode("ascii")
    head = _aws_json(
        "s3api",
        "head-object",
        "--bucket",
        BUCKET,
        "--key",
        f"parallax/results/{RUN_ID}/artifacts.tar",
        "--version-id",
        str(proof.get("version_id", "")),
        "--checksum-mode",
        "ENABLED",
    )
    if (
        int(head.get("ContentLength", -1)) != int(proof.get("bytes", -2))
        or head.get("VersionId") != proof.get("version_id")
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("ChecksumSHA256") != expected_checksum
        or head.get("Metadata", {}).get("sha256") != proof.get("sha256")
    ):
        raise PilotLaunchError("controller artifact HEAD did not match the worker proof")
    return {
        "bucket": BUCKET,
        "key": f"parallax/results/{RUN_ID}/artifacts.tar",
        "version_id": proof["version_id"],
        "bytes": proof["bytes"],
        "sha256": proof["sha256"],
        "checksum_sha256_base64": expected_checksum,
        "server_side_encryption": "AES256",
        "controller_head_exact_match": True,
    }


def _launch_state_path(root: Path, spec: dict[str, Any]) -> Path:
    output = Path(str(spec.get("output_receipt", {}).get("path", "")))
    if output.is_absolute() or not output.parts or ".." in output.parts:
        raise PilotLaunchError("PG2 range launch-state path is unsafe")
    return root / output.parent / "launch-state.json"


def run(root: Path, spec_path: Path, freeze_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    validate_execution_freeze(root, spec_path, freeze_path, spec)
    quarantine = require_machine_quarantine(root, spec)
    initial = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    s3_safety = require_s3_safety()
    model_rows = _model_rows(root, spec)
    overlay_rows = _overlay_rows(root, spec)
    bundle_row, bundle_receipt = _bundle(root, spec)
    input_proof = _require_input_heads(model_rows, bundle_row, overlay_rows)
    _require_result_absent(spec["artifact"]["key"])
    _require_result_absent(spec["result"]["key"])
    _claim(root, spec)
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise PilotLaunchError("SSH or scoped lifecycle key is absent")

    guardian_id: int | None = None
    worker_id: int | None = None
    guardian_market: dict[str, Any] | None = None
    worker_market: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    terminal_body: bytes | None = None
    terminal_s3_proof: dict[str, Any] | None = None
    artifact_s3_proof: dict[str, Any] | None = None
    job_released = False
    cleanup: dict[str, Any] | None = None
    launch_state_path = _launch_state_path(root, spec)
    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "ACTIVE",
        "phase": "CLAIMED_BEFORE_PAID_MUTATION",
        "guardian_instance_id": None,
        "worker_instance_id": None,
        "guardian_offer": None,
        "worker_offer": None,
        "guardian_exact_worker_ID_acknowledged": False,
        "job_released": False,
        "s3_durable": False,
        "failure": None,
        "cleanup": None,
        "automatic_retry": False,
    }

    def record(phase: str, **updates: Any) -> None:
        state.update(updates)
        state["phase"] = phase
        _record_launch_state(launch_state_path, state)

    record("CLAIMED_BEFORE_PAID_MUTATION")
    started = time.monotonic()
    try:
        guardian_market = select_cheapest_offer(role="guardian", contract=spec["guardian"])
        record("GUARDIAN_SELECTED", guardian_offer=guardian_market["selected"])
        guardian_create = _create_instance(
            int(guardian_market["selected"]["id"]),
            _create_arguments(role="guardian", spec=spec, run_id=RUN_ID),
        )
        guardian_id = int(guardian_create["new_contract"])
        record("GUARDIAN_CREATED", guardian_instance_id=guardian_id)
        guardian_row = wait_ssh(
            guardian_id,
            key_path,
            timeout_seconds=int(spec["guardian"]["bootstrap_seconds"]),
        )
        guardian_ssh = _ssh_base(guardian_row, key_path)
        record("GUARDIAN_SSH_REACHED")
        guardian_contract = _guardian_contract(spec, run_id=RUN_ID, instance_id=guardian_id)
        _stage(
            guardian_ssh,
            remote="/opt/parallax/ephemeral_guardian.py",
            content=(root / "infra/vast/parallax/ephemeral_guardian.py").read_bytes(),
            mode="0700",
        )
        _stage(
            guardian_ssh,
            remote=guardian_contract["scoped_api_key_path"],
            content=scoped_key_path.read_bytes(),
        )
        _stage(
            guardian_ssh,
            remote=guardian_contract["self_api_key_path"],
            content=str(guardian_create["instance_api_key"]).encode("utf-8"),
        )
        _stage(
            guardian_ssh,
            remote="/run/parallax/guardian-contract.json",
            content=json.dumps(guardian_contract, sort_keys=True).encode("utf-8"),
        )
        start_guardian = _run(
            [
                *guardian_ssh,
                "nohup python3 /opt/parallax/ephemeral_guardian.py --contract "
                "/run/parallax/guardian-contract.json >/opt/parallax/guardian.log 2>&1 </dev/null &",
            ],
            check=False,
            timeout_seconds=30,
        )
        if start_guardian.returncode != 0:
            raise PilotLaunchError("pilot guardian process did not start")
        _wait_remote_file(guardian_ssh, guardian_contract["armed_path"])
        _require_guardian_live(guardian_id)
        record("GUARDIAN_ARMED_AND_LIVE")

        worker_market = select_cheapest_h100_offer(
            spec["worker"],
            quarantined_machine_ids=set(quarantine["active_machine_ids"]),
            excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
        )
        record("H100_SELECTED", worker_offer=worker_market["selected"])
        worker_create = _create_instance(
            int(worker_market["selected"]["id"]),
            _create_arguments(role="worker", spec=spec, run_id=RUN_ID),
        )
        worker_id = int(worker_create["new_contract"])
        record("H100_CREATED", worker_instance_id=worker_id)
        assignment = _worker_assignment(run_id=RUN_ID, worker_id=worker_id)
        _stage_atomic(
            guardian_ssh,
            remote=guardian_contract["worker_assignment_path"],
            content=json.dumps(assignment, sort_keys=True).encode("utf-8"),
        )
        expected_ack = _worker_assignment_ack(run_id=RUN_ID, worker_id=worker_id)
        _wait_remote_json_exact(
            guardian_ssh,
            guardian_contract["worker_assignment_ack_path"],
            expected_ack,
        )
        _require_guardian_live(guardian_id)
        record(
            "GUARDIAN_EXACT_WORKER_ID_ACKNOWLEDGED",
            guardian_exact_worker_ID_acknowledged=True,
        )
        worker_row = wait_h100_ssh(
            worker_id,
            key_path,
            bootstrap_seconds=int(spec["worker"]["bootstrap_seconds"]),
            running_ssh_deadline_seconds=int(spec["worker"]["running_ssh_deadline_seconds"]),
            health_check=lambda: _require_guardian_live(guardian_id),
        )
        _require_guardian_live(guardian_id)
        worker_ssh = _ssh_base(worker_row, key_path)
        record("H100_SSH_REACHED")
        credentials = _aws_credentials()
        now = datetime.now(timezone.utc)
        model = [_signed(row, credentials, now) for row in model_rows]
        bundle = _signed(bundle_row, credentials, now)
        overlays = [_signed(row, credentials, now) for row in overlay_rows]
        input_contract = _job_contract(
            spec,
            instance_id=worker_id,
            model=model,
            bundle=bundle,
            bundle_receipt=bundle_receipt,
            overlays=overlays,
        )
        artifact_post = presign_artifact_post(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            run_id=RUN_ID,
            key=spec["artifact"]["key"],
            minimum_bytes=int(spec["artifact"]["minimum_bytes"]),
            maximum_bytes=int(spec["artifact"]["maximum_bytes"]),
            now=now,
            expires_seconds=int(spec["artifact"]["post_policy_seconds"]),
        )
        result_url = presign_result_put(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            bucket=BUCKET,
            key=spec["result"]["key"],
            region=REGION,
            now=now,
            expires_seconds=int(spec["result"]["put_url_seconds"]),
        )
        credentials.clear()
        worker_contract = _worker_contract(spec, instance_id=worker_id, result_url=result_url)
        validate_job_contract(input_contract)
        validate_worker_contract(worker_contract)

        record("STAGING_WORKER_CONTROL_FILES")
        staged = {
            "/opt/parallax/ephemeral_s3_worker.py": "infra/vast/parallax/ephemeral_s3_worker.py",
            "/opt/parallax/ephemeral_optimizer_scale_job.py": (
                "infra/vast/parallax/ephemeral_optimizer_scale_job.py"
            ),
            "/opt/parallax/s3_post_upload.py": "infra/vast/parallax/s3_post_upload.py",
            "/opt/parallax/ephemeral_pilot_job.py": "infra/vast/parallax/ephemeral_pilot_job.py",
            "/opt/parallax/ephemeral_pilot_worker.py": (
                "infra/vast/parallax/ephemeral_pilot_worker.py"
            ),
        }
        for remote, relative in staged.items():
            _stage(worker_ssh, remote=remote, content=(root / relative).read_bytes(), mode="0700")
        _stage(
            worker_ssh,
            remote="/run/parallax/pilot-input-contract.json",
            content=json.dumps(input_contract, sort_keys=True).encode("utf-8"),
        )
        _stage(
            worker_ssh,
            remote="/run/parallax/pilot-artifact-post.json",
            content=json.dumps(artifact_post, sort_keys=True).encode("utf-8"),
        )
        _stage(
            worker_ssh,
            remote=worker_contract["instance_api_key_path"],
            content=str(worker_create["instance_api_key"]).encode("utf-8"),
        )
        _stage(
            worker_ssh,
            remote="/run/parallax/worker-contract.json",
            content=json.dumps(worker_contract, sort_keys=True).encode("utf-8"),
        )
        model.clear()
        bundle.clear()
        overlays.clear()
        input_contract.clear()
        artifact_post.clear()
        worker_contract.clear()
        job_released = True
        record("JOB_RELEASED", job_released=True)
        start_worker = _run(
            [*worker_ssh, _pilot_worker_start_command()],
            check=False,
            timeout_seconds=30,
        )
        if start_worker.returncode != 0:
            raise PilotLaunchError("pilot worker process did not start")
        terminal, terminal_body, terminal_s3_proof = _wait_result(
            key=spec["result"]["key"],
            run_id=RUN_ID,
            instance_id=worker_id,
            seconds=int(spec["result"]["wait_seconds"]),
        )
        worker_artifact = (
            terminal.get("job", {}).get("result", {}).get("payload", {}).get("artifact_proof")
        )
        if isinstance(worker_artifact, dict):
            artifact_s3_proof = _artifact_head(worker_artifact)
        if artifact_s3_proof is not None and terminal_s3_proof is not None:
            record("S3_DURABLE", s3_durable=True)
    except Exception as error:
        state["failure"] = {
            "type": type(error).__name__,
            "message": str(error)[:500],
            "phase": state["phase"],
        }
        state["status"] = "FAILED_BEFORE_OR_DURING_JOB"
        _record_launch_state(launch_state_path, state)
        raise
    finally:
        try:
            cleanup = _finalize_lifecycle(
                worker_id=worker_id,
                guardian_id=guardian_id,
                job_released=job_released,
                s3_durable=(
                    artifact_s3_proof is not None and terminal_s3_proof is not None
                ),
            )
        except Exception as cleanup_error:
            state["status"] = "CLEANUP_CONFIRMATION_FAILED"
            state["cleanup"] = {
                "error_type": type(cleanup_error).__name__,
                "message": str(cleanup_error)[:500],
            }
            _record_launch_state(launch_state_path, state)
            raise
        else:
            state["cleanup"] = cleanup
            state["status"] = (
                "CLEANED_AFTER_FAILURE"
                if state["failure"]
                else "CLEANED_AFTER_S3_DURABILITY"
            )
            state["phase"] = "TERMINAL_CLEANUP_CONFIRMED"
            _record_launch_state(launch_state_path, state)

    if terminal is None or terminal_body is None or terminal_s3_proof is None:
        raise PilotLaunchError("pilot terminal result is absent")
    mechanical_green = (
        terminal.get("status") == "GREEN_JOB_TERMINAL" and artifact_s3_proof is not None
    )
    final = _confirm_account_zero(seconds=45)
    job_payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_SELECTED_PILOT_EXECUTED" if mechanical_green else "RED_SELECTED_PILOT_EXECUTION",
        "run_id": RUN_ID,
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "scope": "PG2_SUPPORTING_SEED17_RANGE_MECHANICAL_NONPROMOTION",
        "initial_account": initial,
        "s3_safety": s3_safety,
        "input_proof": input_proof,
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_instance_id": guardian_id,
        "worker_instance_id": worker_id,
        "terminal": terminal,
        "pilot_adjudication": job_payload.get("pilot", {}).get("mechanical_adjudication"),
        "execution_provenance": job_payload.get("execution_provenance"),
        "artifact_s3_proof": artifact_s3_proof,
        "terminal_s3_proof": terminal_s3_proof,
        "cleanup": cleanup,
        "launch_state": {
            "path": launch_state_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(launch_state_path),
        },
        "terminal_account": final,
        "wall_seconds": time.monotonic() - started,
        "automatic_retry": False,
        "promotion_boundary": "ONE_SUPPORTING_PG2_RANGE_SCREEN_NO_PROMOTION_PRIMARY_CHAIN_OR_BROAD_RUN",
    }
    output_relative = Path(str(spec["output_receipt"].get("path", "")))
    if output_relative.is_absolute() or not output_relative.parts or ".." in output_relative.parts:
        raise PilotLaunchError("pilot output receipt path is unsafe")
    output = root / output_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, packet)
    (output.parent / "terminal.json").write_bytes(terminal_body)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    arguments = parser.parse_args()
    packet = run(arguments.root.resolve(), arguments.spec.resolve(), arguments.freeze.resolve())
    print(json.dumps({"run_id": RUN_ID, "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"] == "GREEN_SELECTED_PILOT_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
