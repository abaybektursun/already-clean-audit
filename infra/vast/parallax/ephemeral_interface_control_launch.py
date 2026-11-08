"""Launch one S3-durable PG2 interface control on the cheapest eligible H100."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .ephemeral_interface_control_job import (
    MODEL_BYTES,
    MODEL_NAMES,
    OVERLAY_IDENTITIES,
    RUN_ID,
    SOURCE_CONTRACT_SHA256,
    validate_contract as validate_job_contract,
)
from .ephemeral_interface_control_worker import (
    EXPECTED_COMMAND,
    MAXIMUM_RESULT_BYTES,
    validate_contract as validate_worker_contract,
)
from .ephemeral_optimizer_scale_launch import (
    RUNTIME_IMAGE,
    RUNTIME_TEMPLATE,
    _guardian_contract,
    require_machine_quarantine,
    select_cheapest_h100_offer,
    wait_h100_ssh,
)
from .ephemeral_pg2_optimizer_scale_launch import (
    _record_launch_state,
    _worker_assignment,
    _worker_assignment_ack,
)
from .ephemeral_pilot_launch import (
    _bundle,
    _model_rows,
    _require_input_heads,
    _require_result_absent,
    _resolve_bound,
    _signed,
    _validate_input,
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
    _require_worker_can_still_publish,
    _stage,
    _stage_atomic,
    _wait_remote_file,
    _wait_remote_json_exact,
    EphemeralLaunchError,
    require_s3_safety,
    select_cheapest_offer,
)
from .guarded_launch import _create_instance, _run, _ssh_base, wait_ssh
from .interface_control_source_gate import validate_local as validate_interface_source
from .interface_control_s3_post_upload import validate_contract as validate_post_contract
from .interface_control_s3_presign_post import presign_artifact_post
from .s3_presign_put import presign_result_put


class InterfaceLaunchError(RuntimeError):
    pass


def _overlay_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_bound(root, spec["inputs"]["overlay_receipt"], label="overlay receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    rows = receipt.get("objects")
    if (
        receipt.get("status") != "GREEN_PG2_INTERFACE_CONTROL_OVERLAYS_S3"
        or receipt.get("run_id") != RUN_ID
        or receipt.get("lineage") != "pg2"
        or receipt.get("role") != "supporting_boundary_first"
        or receipt.get("aws_profile") != "fuelos"
        or receipt.get("region") != REGION
        or receipt.get("bucket") != BUCKET
        or receipt.get("object_count") != len(OVERLAY_IDENTITIES)
        or not isinstance(rows, list)
        or len(rows) != len(OVERLAY_IDENTITIES)
    ):
        raise InterfaceLaunchError("interface overlay receipt is not exact")
    observed = {}
    for index, row in enumerate(rows):
        _validate_input(row, label=f"interface overlay {index}")
        observed[str(row.get("path", ""))] = str(row["sha256"])
        if row.get("controller_head_exact_match") is not True:
            raise InterfaceLaunchError("interface overlay lacks an exact controller HEAD")
    if observed != OVERLAY_IDENTITIES:
        raise InterfaceLaunchError("interface overlay identities changed")
    return rows


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
        raise InterfaceLaunchError("interface launch spec schema changed")
    if (
        spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_H100"
        or spec.get("program") != "parallax"
        or spec.get("scope") != "pg2_supporting_interface_control_nonpromotion"
        or spec.get("provider") != "vast.ai"
        or spec.get("run_id") != RUN_ID
        or spec.get("one_shot_authorization", {}).get("id") != RUN_ID
        or spec.get("automatic_paid_retry") is not False
    ):
        raise InterfaceLaunchError("interface launch escaped its exact one-shot scope")
    if spec.get("authorization") != {
        "paid_control": True,
        "h100": True,
        "processor_reads_before_weights": True,
        "model_weight_reads_after_processor_gate": True,
        "train_data_reads": True,
        "discovery_evaluation": True,
        "training": True,
        "confirmation": False,
        "promotion": False,
        "PG2": True,
        "PG1": False,
        "broad_run": False,
        "science_claim": False,
    }:
        raise InterfaceLaunchError("interface paid authorization changed")
    if spec.get("storage") != {
        "durable_provider": "amazon_s3",
        "aws_profile": "fuelos",
        "region": REGION,
        "bucket": BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": 80,
    }:
        raise InterfaceLaunchError("interface storage boundary changed")
    if spec.get("worker_image") != RUNTIME_IMAGE:
        raise InterfaceLaunchError("interface runtime image changed")
    worker = spec.get("worker", {})
    if (
        worker.get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time"
        or worker.get("gpu_kind") != "single_h100_80gb_or_larger"
        or worker.get("launch_mode") != "managed_template_private_runtime"
        or worker.get("image") != RUNTIME_IMAGE
        or worker.get("template_hash") != RUNTIME_TEMPLATE
        or worker.get("disk_gb") != 80
        or worker.get("minimum_gpu_ram_mb") != 80000
        or worker.get("hard_destroy_seconds") != 1500
    ):
        raise InterfaceLaunchError("interface H100 construction changed")
    job = spec.get("job", {})
    if (
        job.get("stage") != "training_interface_control"
        or job.get("maximum_seconds") != 1200
        or job.get("command") != EXPECTED_COMMAND
        or job.get("environment") != {}
        or job.get("bounds")
        != {
            "parallel_downloads": 3,
            "processor_download_timeout_seconds": 180,
            "weight_download_timeout_seconds": 600,
            "tokenizer_preflight_timeout_seconds": 120,
            "control_timeout_seconds": 900,
            "artifact_upload_timeout_seconds": 600,
            "minimum_free_bytes": 50_000_000_000,
            "maximum_terminal_bytes": MAXIMUM_RESULT_BYTES,
        }
    ):
        raise InterfaceLaunchError("interface job or execution bounds changed")
    if spec.get("artifact") != {
        "key": f"parallax/results/{RUN_ID}/artifacts.tar",
        "minimum_bytes": 1_200_000_000,
        "maximum_bytes": 2_100_000_000,
        "post_policy_seconds": 2700,
        "required_http_status": 201,
        "required_version_id": True,
        "required_checksum": "SHA256",
        "required_encryption": "AES256",
    }:
        raise InterfaceLaunchError("interface artifact boundary changed")
    if spec.get("result") != {
        "key": f"parallax/results/{RUN_ID}/terminal.json",
        "maximum_bytes": MAXIMUM_RESULT_BYTES,
        "put_url_seconds": 2700,
        "wait_seconds": 1400,
        "required_http_status": 200,
        "required_version_id": True,
        "required_encryption": "AES256",
        "archive_durable_action": "destroy",
        "pre_science_terminal_durable_action": "destroy",
        "science_started_archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
    }:
        raise InterfaceLaunchError("interface terminal/lifecycle boundary changed")
    for label, row in spec.get("bound_artifacts", {}).items():
        _resolve_bound(root, row, label=f"bound artifact {label}")
    for label, row in spec.get("implementation", {}).items():
        _resolve_bound(root, row, label=f"implementation {label}")
    validate_interface_source(root)
    model = _model_rows(root, spec)
    if {str(row["path"]) for row in model} != MODEL_NAMES or sum(
        int(row["bytes"]) for row in model
    ) != MODEL_BYTES:
        raise InterfaceLaunchError("interface PG2 model identity changed")
    _overlay_rows(root, spec)
    _bundle(root, spec)
    require_machine_quarantine(root, spec)
    if set(spec.get("invariants", [])) != {
        "supporting_PG2_first_only_not_primary_chain",
        "PG1_requires_separate_post_PG2_green_freeze",
        "processor_gate_precedes_weight_hydration_and_model_load",
        "three_cyclic_codebooks_counterbalance_letter_prior",
        "per_example_suffix_loss",
        "exact_registered_seed17_items_schedule_adapter_steps_and_LR",
        "confirmation_reads_zero",
        "nonpromotion",
        "no_automatic_retry",
        "cheapest_current_eligible_H100_all_regions",
        "no_Vast_persistent_storage",
        "exact_80_GB_disposable_scratch_no_blanket_allowance",
        "pre_science_terminal_durable_then_destroy",
        "artifact_archive_durable_before_destroy",
        "post_science_archive_nondurable_stops_and_preserves_scratch",
        "controller_HEAD_exact_archive_and_terminal_versions",
        "guardian_acknowledges_exact_worker_ID_before_H100_SSH",
        "known_instance_cleanup_and_account_zero_each_require_three_observations",
    }:
        raise InterfaceLaunchError("interface invariants changed")
    return spec


def validate_execution_freeze(
    root: Path, spec_path: Path, freeze_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status")
        != "FROZEN_ONE_SHOT_PG2_INTERFACE_CONTROL_AFTER_NO_CREATE_GREEN"
        or freeze.get("run_id") != RUN_ID
        or freeze.get("automatic_paid_retry") is not False
        or freeze.get("authorization")
        != {
            "create_vast_instances": True,
            "create_vast_volumes": False,
            "h100": True,
            "processor_reads_before_weights": True,
            "model_weight_reads_after_processor_gate": True,
            "train_data_reads": True,
            "discovery_evaluation": True,
            "training": True,
            "confirmation": False,
            "promotion": False,
            "PG2": True,
            "PG1": False,
            "broad_run": False,
            "automatic_retry": False,
        }
        or freeze.get("launch_spec", {}).get("path")
        != str(spec_path.resolve().relative_to(root.resolve()))
        or freeze.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
    ):
        raise InterfaceLaunchError("interface paid execution freeze is absent or not exact")
    rehearsal_path = _resolve_bound(
        root, freeze.get("no_create_rehearsal", {}), label="interface no-create rehearsal"
    )
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if (
        rehearsal.get("status") != "GREEN_INTERFACE_CONTROL_NO_CREATE_REHEARSAL"
        or rehearsal.get("run_id") != RUN_ID
        or rehearsal.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
        or rehearsal.get("mutation")
        != {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "s3_objects_written": 0,
            "files_downloaded": 0,
            "gpu_operations": 0,
        }
        or rehearsal.get("paid_authorization") is not False
    ):
        raise InterfaceLaunchError("interface freeze lacks an exact zero-mutation rehearsal")
    plants_path = _resolve_bound(root, freeze.get("planted_tests", {}), label="interface plants")
    plants = json.loads(plants_path.read_text(encoding="utf-8"))
    if (
        plants.get("status") != "GREEN"
        or plants.get("exit_code") != 0
        or plants.get("paid_action") is not False
        or plants.get("confirmation_items_read") != 0
    ):
        raise InterfaceLaunchError("interface freeze lacks exact green planted tests")
    return freeze


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
        "status": "FROZEN_H100_INTERFACE_CONTROL_JOB",
        "stage": "training_interface_control",
        "provider": "vast.ai",
        "run_id": RUN_ID,
        "lineage": "pg2",
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
            "tokenizer_preflight_output": "/workspace/parallax/tokenizer-preflight.json",
            "control_output_dir": (
                "/workspace/parallax/source/studies/lineage/cross-lineage/results/"
                "parallax_training_interface_control_pg2_v3"
            ),
            "manifest_output": "/workspace/parallax/artifact-manifest.json",
            "archive_output": "/workspace/parallax/artifacts.tar",
            "phase_output": "/workspace/parallax/phase.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": spec["job"]["bounds"],
        "firewall": {
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
        },
    }


def _worker_contract(spec: dict[str, Any], *, instance_id: int, result_url: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_SELECTED_INTERFACE_WORKER",
        "run_id": RUN_ID,
        "stage": "training_interface_control",
        "instance_id": instance_id,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "phase_path": "/workspace/parallax/phase.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": EXPECTED_COMMAND,
        "maximum_job_seconds": 1200,
        "maximum_result_bytes": MAXIMUM_RESULT_BYTES,
        "result_key": spec["result"]["key"],
        "result_put_url": result_url,
        "archive_durable_action": "destroy",
        "pre_science_terminal_durable_action": "destroy",
        "science_started_archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
        "automatic_retry": False,
        "environment": {},
    }


def _worker_start_command() -> str:
    return (
        "set -eu; mkdir -p /workspace/parallax; "
        "nohup python3 /opt/parallax/ephemeral_interface_control_worker.py --contract "
        "/run/parallax/worker-contract.json >/workspace/parallax/worker.log 2>&1 </dev/null & "
        "worker_pid=$!; kill -0 $worker_pid 2>/dev/null || test -s /workspace/parallax/terminal.json"
    )


def _artifact_head(proof: dict[str, Any]) -> dict[str, Any]:
    try:
        expected_checksum = base64.b64encode(
            bytes.fromhex(str(proof.get("sha256", "")))
        ).decode("ascii")
    except ValueError as error:
        raise InterfaceLaunchError("worker artifact proof has malformed SHA-256") from error
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
        not 1_200_000_000 <= int(proof.get("bytes", 0)) <= 2_100_000_000
        or int(head.get("ContentLength", -1)) != int(proof.get("bytes", -2))
        or head.get("VersionId") != proof.get("version_id")
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("ChecksumSHA256") != expected_checksum
        or head.get("Metadata", {}).get("sha256") != proof.get("sha256")
    ):
        raise InterfaceLaunchError("controller artifact HEAD did not match worker proof")
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
        raise InterfaceLaunchError("interface launch-state path is unsafe")
    return root / output.parent / "launch-state.json"


def _wait_interface_result(
    *, key: str, run_id: str, instance_id: int, seconds: int, maximum_bytes: int
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if maximum_bytes != MAXIMUM_RESULT_BYTES:
        raise InterfaceLaunchError("interface terminal byte bound changed")
    deadline = time.monotonic() + seconds
    head: dict[str, Any] | None = None
    consecutive_absences = 0
    while time.monotonic() < deadline:
        try:
            candidate = _aws_json(
                "s3api",
                "head-object",
                "--bucket",
                BUCKET,
                "--key",
                key,
                "--checksum-mode",
                "ENABLED",
            )
        except EphemeralLaunchError:
            candidate = None
        if isinstance(candidate, dict) and candidate.get("VersionId"):
            head = candidate
            break
        if _require_worker_can_still_publish(instance_id):
            consecutive_absences += 1
            if consecutive_absences >= 3:
                raise InterfaceLaunchError("interface worker disappeared before S3 durability")
        else:
            consecutive_absences = 0
        time.sleep(2)
    if head is None:
        raise InterfaceLaunchError("versioned interface terminal did not appear before deadline")
    version_id = str(head["VersionId"])
    with tempfile.TemporaryDirectory(prefix="parallax-interface-terminal-") as temporary:
        path = Path(temporary) / "terminal.json"
        response = _aws_json(
            "s3api",
            "get-object",
            "--bucket",
            BUCKET,
            "--key",
            key,
            "--version-id",
            version_id,
            "--checksum-mode",
            "ENABLED",
            str(path),
            timeout_seconds=60,
        )
        body = path.read_bytes()
    if len(body) != int(head.get("ContentLength", -1)) or len(body) > maximum_bytes:
        raise InterfaceLaunchError("interface terminal length is not exact or exceeds 4 MiB")
    terminal = json.loads(body)
    if terminal.get("run_id") != run_id or terminal.get("instance_id") != instance_id:
        raise InterfaceLaunchError("interface terminal identity does not match the worker")
    if head.get("ServerSideEncryption") != "AES256" or response.get("ServerSideEncryption") != "AES256":
        raise InterfaceLaunchError("interface terminal is not AES256 encrypted")
    proof = {
        "bucket": BUCKET,
        "key": key,
        "version_id": version_id,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "server_side_encryption": "AES256",
        "checksum_sha256_base64": head.get("ChecksumSHA256") or response.get("ChecksumSHA256"),
    }
    return terminal, body, proof


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
        raise InterfaceLaunchError("SSH or scoped lifecycle key is absent")

    guardian_id: int | None = None
    worker_id: int | None = None
    guardian_market: dict[str, Any] | None = None
    worker_market: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    terminal_body: bytes | None = None
    terminal_s3_proof: dict[str, Any] | None = None
    artifact_s3_proof: dict[str, Any] | None = None
    pre_science_terminal_durable = False
    lifecycle_durable = False
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
        guardian_start = _run(
            [
                *guardian_ssh,
                "nohup python3 /opt/parallax/ephemeral_guardian.py --contract "
                "/run/parallax/guardian-contract.json >/opt/parallax/guardian.log 2>&1 </dev/null &",
            ],
            check=False,
            timeout_seconds=30,
        )
        if guardian_start.returncode != 0:
            raise InterfaceLaunchError("interface guardian did not start")
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
        _wait_remote_json_exact(
            guardian_ssh,
            guardian_contract["worker_assignment_ack_path"],
            _worker_assignment_ack(run_id=RUN_ID, worker_id=worker_id),
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
        validate_post_contract(artifact_post)
        validate_worker_contract(worker_contract)
        record("STAGING_WORKER_CONTROL_FILES")
        staged = {
            "/opt/parallax/ephemeral_s3_worker.py": "infra/vast/parallax/ephemeral_s3_worker.py",
            "/opt/parallax/ephemeral_optimizer_scale_job.py": (
                "infra/vast/parallax/ephemeral_optimizer_scale_job.py"
            ),
            "/opt/parallax/interface_control_s3_post_upload.py": (
                "infra/vast/parallax/interface_control_s3_post_upload.py"
            ),
            "/opt/parallax/ephemeral_interface_control_job.py": (
                "infra/vast/parallax/ephemeral_interface_control_job.py"
            ),
            "/opt/parallax/ephemeral_interface_control_worker.py": (
                "infra/vast/parallax/ephemeral_interface_control_worker.py"
            ),
        }
        for remote, relative in staged.items():
            _stage(worker_ssh, remote=remote, content=(root / relative).read_bytes(), mode="0700")
        _stage(
            worker_ssh,
            remote="/run/parallax/interface-input-contract.json",
            content=json.dumps(input_contract, sort_keys=True).encode("utf-8"),
        )
        _stage(
            worker_ssh,
            remote="/run/parallax/interface-artifact-post.json",
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
        worker_start = _run([*worker_ssh, _worker_start_command()], check=False, timeout_seconds=30)
        if worker_start.returncode != 0:
            raise InterfaceLaunchError("interface worker did not start")
        terminal, terminal_body, terminal_s3_proof = _wait_interface_result(
            key=spec["result"]["key"],
            run_id=RUN_ID,
            instance_id=worker_id,
            seconds=int(spec["result"]["wait_seconds"]),
            maximum_bytes=int(spec["result"]["maximum_bytes"]),
        )
        worker_artifact = (
            terminal.get("job", {}).get("result", {}).get("payload", {}).get("artifact_proof")
        )
        if isinstance(worker_artifact, dict):
            artifact_s3_proof = _artifact_head(worker_artifact)
        pre_science_terminal_durable = bool(
            terminal_s3_proof
            and terminal.get("durability_contract", {}).get("pre_science") is True
            and terminal.get("durability_contract", {}).get("post_terminal_action") == "destroy"
        )
        lifecycle_durable = terminal_s3_proof is not None and (
            artifact_s3_proof is not None or pre_science_terminal_durable
        )
        if lifecycle_durable:
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
                s3_durable=lifecycle_durable,
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
                "CLEANED_AFTER_FAILURE" if state["failure"] else "CLEANED_AFTER_S3_DURABILITY"
            )
            state["phase"] = "TERMINAL_CLEANUP_CONFIRMED"
            _record_launch_state(launch_state_path, state)

    if terminal is None or terminal_body is None or terminal_s3_proof is None:
        raise InterfaceLaunchError("interface terminal result is absent")
    final = _confirm_account_zero(seconds=45)
    job_payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    mechanically_complete = (
        terminal.get("status") == "GREEN_JOB_TERMINAL" and artifact_s3_proof is not None
    )
    preflight_red = terminal.get("status") == "RED_PREFLIGHT_TERMINAL"
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "GREEN_INTERFACE_CONTROL_EXECUTED"
            if mechanically_complete
            else "RED_INTERFACE_CONTROL_PREFLIGHT"
            if preflight_red
            else "RED_INTERFACE_CONTROL_EXECUTION"
        ),
        "run_id": RUN_ID,
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "scope": "PG2_SUPPORTING_INTERFACE_CONTROL_PROVISIONAL_NONPROMOTION",
        "initial_account": initial,
        "s3_safety": s3_safety,
        "input_proof": input_proof,
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_instance_id": guardian_id,
        "worker_instance_id": worker_id,
        "terminal": terminal,
        "interface_adjudication": job_payload.get("control", {}).get("adjudication"),
        "execution_provenance": job_payload.get("execution_provenance"),
        "artifact_s3_proof": artifact_s3_proof,
        "terminal_s3_proof": terminal_s3_proof,
        "pre_science_terminal_durable": pre_science_terminal_durable,
        "cleanup": cleanup,
        "launch_state": {
            "path": launch_state_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(launch_state_path),
        },
        "terminal_account": final,
        "wall_seconds": time.monotonic() - started,
        "automatic_retry": False,
        "promotion_boundary": (
            "PG2_SUPPORTING_INTERFACE_CONTROL_ONLY_PG1_REQUIRES_SEPARATE_POST_GREEN_FREEZE"
        ),
    }
    output_relative = Path(str(spec["output_receipt"].get("path", "")))
    if output_relative.is_absolute() or not output_relative.parts or ".." in output_relative.parts:
        raise InterfaceLaunchError("interface output receipt path is unsafe")
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
    args = parser.parse_args()
    packet = run(args.root.resolve(), args.spec.resolve(), args.freeze.resolve())
    print(json.dumps({"run_id": RUN_ID, "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"] in {
        "GREEN_INTERFACE_CONTROL_EXECUTED",
        "RED_INTERFACE_CONTROL_PREFLIGHT",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
