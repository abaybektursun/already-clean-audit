"""Launch exactly one S3-durable PG1 optimizer-scale microgate on Vast H100.

The cheap guardian is live on a different machine before the H100 is created.
The H100 is selected from the current all-region market by literal quoted cost,
hydrates immutable S3 versions into disposable scratch, and publishes its small
terminal packet to versioned S3 before self-destroying.  This launcher contains
no persistent Vast volume operation and no automatic retry path.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .ephemeral_s3_launch import (
    AWS_PROFILE,
    BUCKET,
    REGION,
    EphemeralLaunchError,
    _aws_credentials,
    _aws_json,
    _claim,
    _confirm_account_zero,
    _create_arguments,
    _finalize_lifecycle,
    _require_guardian_live,
    _require_initial_zero,
    _stage,
    _stage_atomic,
    _wait_remote_file,
    _wait_remote_json_exact,
    _wait_result,
    _worker_start_command,
    require_s3_safety,
    select_cheapest_offer,
)
from .guarded_launch import (
    _create_instance,
    _rows,
    _run,
    _show_instance,
    _ssh_base,
    _vast,
    wait_ssh,
)
from .lifecycle_supervisor import LifecycleError
from .s3_presign_put import presign_get, presign_result_put


RUNTIME_IMAGE = (
    "769063704696.dkr.ecr.us-east-1.amazonaws.com/parallax/runtime-minimal@"
    "sha256:f7407a831951f4c744a9afecf59d248bfadd31d44d012f200bc5055b6d4d4ed6"
)
RUNTIME_TEMPLATE = "a49dfc59f765f8c5b9c1e0c3a318a486"
PUBLIC_OFFER_FIELDS = (
    "id",
    "machine_id",
    "gpu_name",
    "num_gpus",
    "gpu_ram",
    "cpu_ram",
    "cpu_cores_effective",
    "reliability",
    "dph_total",
    "storage_cost",
    "disk_space",
    "disk_bw",
    "duration",
    "geolocation",
    "verification",
    "cuda_max_good",
    "inet_down",
    "inet_up",
    "static_ip",
    "direct_port_count",
)
H100_NAMES = {"H100", "H100 PCIE", "H100 SXM", "H100 NVL"}


class OptimizerLaunchError(EphemeralLaunchError):
    pass


def _digest(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OptimizerLaunchError(f"{label} is not a lowercase SHA-256")
    return digest


def _resolve_bound(root: Path, row: dict[str, Any], *, label: str) -> Path:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise OptimizerLaunchError(f"unsafe {label} path")
    path = (root / relative).resolve()
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise OptimizerLaunchError(f"{label} hash mismatch")
    return path


def require_machine_quarantine(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen, mode-specific machine eligibility registry or refuse."""
    binding = spec.get("bound_artifacts", {}).get("machine_quarantine")
    if not isinstance(binding, dict):
        raise OptimizerLaunchError("optimizer launch has no bound machine quarantine registry")
    path = _resolve_bound(root, binding, label="machine_quarantine")
    registry = json.loads(path.read_text(encoding="utf-8"))
    construction = str(spec.get("worker", {}).get("launch_mode", ""))
    expected_top = {
        "schema_version",
        "created_utc",
        "program",
        "status",
        "scope",
        "launch_construction",
        "runtime_image",
        "template_hash",
        "policy",
        "entries",
        "active_machine_ids_for_launch_construction",
        "rehabilitated_machine_ids_for_launch_construction",
        "adjudication",
        "promotion_boundary",
    }
    if (
        set(registry) != expected_top
        or registry.get("schema_version") != 1
        or registry.get("program") != "parallax"
        or registry.get("status") != "FROZEN"
        or registry.get("scope") != "h100_machine_access_eligibility"
        or registry.get("launch_construction") != construction
        or registry.get("runtime_image") != RUNTIME_IMAGE
        or registry.get("template_hash") != RUNTIME_TEMPLATE
    ):
        raise OptimizerLaunchError("machine quarantine registry is not exact for this launch")
    if registry.get("policy") != {
        "missing_registry": "REFUSE_LAUNCH",
        "hash_drift": "REFUSE_LAUNCH",
        "active_matching_entry": "EXCLUDE_MACHINE_BEFORE_PRICE_ORDERING",
        "raw_mode_failure_alone": "DO_NOT_EXCLUDE_FROM_MANAGED_MODE",
        "rehabilitation": "REQUIRES_EXACT_MANAGED_MODE_GREEN_EVIDENCE",
        "unknown_machine": "ELIGIBLE_SUBJECT_TO_ALL_OTHER_FROZEN_FILTERS",
    }:
        raise OptimizerLaunchError("machine quarantine policy changed")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise OptimizerLaunchError("machine quarantine registry has no evidence entries")
    machine_ids: set[int] = set()
    active: set[int] = set()
    rehabilitated: set[int] = set()
    for entry in entries:
        if set(entry) != {"machine_id", "status", "applies_to", "reason", "evidence"}:
            raise OptimizerLaunchError("machine quarantine entry schema changed")
        machine_id = int(entry["machine_id"])
        if machine_id <= 0 or machine_id in machine_ids:
            raise OptimizerLaunchError("machine quarantine contains an invalid or duplicate id")
        machine_ids.add(machine_id)
        status = entry.get("status")
        applies_to = entry.get("applies_to")
        if status not in {"ACTIVE", "REHABILITATED"} or applies_to != [construction]:
            raise OptimizerLaunchError("machine quarantine entry has an invalid mode/status")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise OptimizerLaunchError("machine quarantine entry lacks evidence")
        for index, row in enumerate(evidence):
            _resolve_bound(root, row, label=f"machine_quarantine_{machine_id}_evidence_{index}")
        (active if status == "ACTIVE" else rehabilitated).add(machine_id)
    if active != set(registry.get("active_machine_ids_for_launch_construction", [])):
        raise OptimizerLaunchError("machine quarantine active-id index disagrees with entries")
    if rehabilitated != set(
        registry.get("rehabilitated_machine_ids_for_launch_construction", [])
    ):
        raise OptimizerLaunchError("machine quarantine rehabilitation index disagrees with entries")
    if active & rehabilitated:
        raise OptimizerLaunchError("machine cannot be active and rehabilitated simultaneously")
    return {
        "status": "GREEN_FROZEN_MODE_SPECIFIC_REGISTRY",
        "path": str(binding["path"]),
        "sha256": str(binding["sha256"]),
        "launch_construction": construction,
        "active_machine_ids": sorted(active),
        "rehabilitated_machine_ids": sorted(rehabilitated),
        "entries": len(entries),
    }


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
        raise OptimizerLaunchError(f"{label} is not an exact versioned input")
    return row


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
        "result",
        "one_shot_authorization",
        "output_receipt",
        "bound_artifacts",
        "implementation",
        "invariants",
        "promotion_boundary",
    }
    if set(spec) != expected_top or spec.get("schema_version") != 1:
        raise OptimizerLaunchError("optimizer launch spec schema changed")
    if (
        spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_H100"
        or spec.get("scope") != "pg1_optimizer_scale_microgate"
        or spec.get("provider") != "vast.ai"
        or spec.get("program") != "parallax"
    ):
        raise OptimizerLaunchError("optimizer launch spec escapes its frozen scope")
    run_id = str(spec.get("run_id", ""))
    if run_id != spec.get("one_shot_authorization", {}).get("id"):
        raise OptimizerLaunchError("optimizer one-shot claim is not bound to run id")
    if spec.get("automatic_paid_retry") is not False:
        raise OptimizerLaunchError("optimizer launch must forbid automatic retry")
    if spec.get("authorization") != {
        "paid_microgate": True,
        "h100": True,
        "model_reads": True,
        "train_data_reads": True,
        "training": True,
        "discovery_evaluation": False,
        "confirmation": False,
        "promotion": False,
        "science_claim": False,
    }:
        raise OptimizerLaunchError("optimizer authorization changed")
    if spec.get("storage") != {
        "durable_provider": "amazon_s3",
        "aws_profile": AWS_PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "maximum_ephemeral_scratch_gb": 1000,
    }:
        raise OptimizerLaunchError("optimizer storage contract is not exact S3-only")
    if spec.get("worker_image") != RUNTIME_IMAGE:
        raise OptimizerLaunchError("optimizer runtime image changed")
    expected_auth = {
        "type": "aws_ecr_ephemeral",
        "profile": AWS_PROFILE,
        "region": REGION,
        "registry": "769063704696.dkr.ecr.us-east-1.amazonaws.com",
        "username": "AWS",
        "credential_source": "aws_ecr_get_login_password",
        "persist_secret": False,
        "maximum_token_age_seconds": 43200,
    }
    if spec.get("worker_registry_auth") != expected_auth:
        raise OptimizerLaunchError("optimizer registry authorization changed")
    guardian = spec.get("guardian", {})
    if (
        guardian.get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time"
        or guardian.get("gpu_kind") != "non_h100_mechanical_guard"
        or guardian.get("disk_gb") != 10
        or not 0 < float(guardian.get("price_ceiling_dph", 0)) <= 0.12
        or int(guardian.get("worker_hard_runtime_seconds", 0)) != 960
    ):
        raise OptimizerLaunchError("optimizer guardian contract changed")
    worker = spec.get("worker", {})
    if (
        worker.get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time"
        or worker.get("gpu_kind") != "single_h100_80gb_or_larger"
        or worker.get("launch_mode") != "managed_template_private_runtime"
        or worker.get("image") != RUNTIME_IMAGE
        or worker.get("template_hash") != RUNTIME_TEMPLATE
        or worker.get("disk_gb") != 80
        or worker.get("minimum_gpu_ram_mb") != 80_000
        or worker.get("minimum_cpu_ram_mb") != 64_000
        or worker.get("minimum_cpu_cores_effective") != 8
        or worker.get("minimum_disk_bw_mb_s") != 500
        or worker.get("minimum_cuda") != 13.0
        or worker.get("minimum_inet_down_mbps") != 500
        or not 0 < float(worker.get("price_ceiling_dph", 0)) <= 2.25
        or not 10 <= int(worker.get("running_ssh_deadline_seconds", 0)) <= 90
        or int(worker.get("running_ssh_deadline_seconds", 0))
        >= int(worker.get("bootstrap_seconds", 0))
        or int(worker.get("hard_destroy_seconds", 0)) != 1200
    ):
        raise OptimizerLaunchError("optimizer H100 contract changed")
    job = spec.get("job", {})
    if job != {
        "stage": "training_optimizer_scale_microgate",
        "maximum_seconds": 900,
        "command": [
            "python3",
            "/opt/parallax/ephemeral_optimizer_scale_job.py",
            "--contract",
            "/run/parallax/optimizer-job-contract.json",
        ],
        "environment": {},
        "bounds": {
            "parallel_downloads": 3,
            "download_timeout_seconds": 600,
            "preflight_timeout_seconds": 180,
            "optimizer_timeout_seconds": 600,
            "minimum_free_bytes": 50_000_000_000,
        },
    }:
        raise OptimizerLaunchError("optimizer job command/bounds changed")
    inputs = spec.get("inputs", {})
    model = inputs.get("model")
    if not isinstance(model, list) or len(model) != 12:
        raise OptimizerLaunchError("optimizer launch must bind all twelve model objects")
    for row in model:
        _validate_input(row, label="model input")
    if sum(int(row["bytes"]) for row in model) != 11_715_866_017:
        raise OptimizerLaunchError("optimizer model byte total changed")
    _validate_input(inputs.get("bundle", {}), label="source bundle")
    if (
        inputs["bundle"].get("bytes") != 12_626_256
        or inputs["bundle"].get("sha256")
        != "8b2f16138376e992816f669fb63ab9ff18d6085da3d009aa9931f6e0a834c517"
    ):
        raise OptimizerLaunchError("optimizer source bundle identity changed")
    receipt_path = _resolve_bound(root, inputs["bundle_receipt"], label="bundle receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "GREEN"
        or receipt.get("bundle_sha256") != inputs["bundle"]["sha256"]
        or receipt.get("bundle_bytes") != inputs["bundle"]["bytes"]
        or receipt.get("confirmation_image_included") is not False
        or receipt.get("confirmation_manifest_included") is not False
        or len(receipt.get("entries", [])) != 155
    ):
        raise OptimizerLaunchError("optimizer source bundle receipt is not exact/green")
    result = spec.get("result", {})
    if result != {
        "key": f"parallax/results/{run_id}/terminal.json",
        "maximum_bytes": 1024 * 1024,
        "put_url_seconds": 1800,
        "wait_seconds": 1100,
        "required_http_status": 200,
        "required_version_id": True,
        "required_encryption": "AES256",
        "post_upload_action": "destroy",
        "upload_failure_action": "stop",
    }:
        raise OptimizerLaunchError("optimizer terminal-result contract changed")
    for label, row in spec.get("bound_artifacts", {}).items():
        _resolve_bound(root, row, label=label)
    implementation_paths = {
        "launcher": "infra/vast/parallax/ephemeral_optimizer_scale_launch.py",
        "job": "infra/vast/parallax/ephemeral_optimizer_scale_job.py",
        "worker": "infra/vast/parallax/ephemeral_s3_worker.py",
        "guardian": "infra/vast/parallax/ephemeral_guardian.py",
        "presigner": "infra/vast/parallax/s3_presign_put.py",
        "transport_launcher": "infra/vast/parallax/ephemeral_s3_launch.py",
        "vast_launcher": "infra/vast/parallax/guarded_launch.py",
        "lifecycle_supervisor": "infra/vast/parallax/lifecycle_supervisor.py",
        "artifacts": "infra/vast/parallax/artifacts.py",
        "no_create_rehearsal": "infra/vast/parallax/optimizer_scale_no_create_rehearsal.py",
    }
    for label, relative in implementation_paths.items():
        row = spec.get("implementation", {}).get(label, {})
        if row.get("path") != relative or sha256_file(root / relative) != row.get("sha256"):
            raise OptimizerLaunchError(f"optimizer {label} implementation hash mismatch")
    require_machine_quarantine(root, spec)
    return spec


def validate_execution_freeze(
    root: Path, spec_path: Path, freeze_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    """Require a separate paid freeze that binds this spec and its no-create rehearsal."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "created_utc",
        "program",
        "status",
        "scope",
        "run_id",
        "automatic_paid_retry",
        "authorization",
        "one_shot",
        "bound_artifacts",
        "stop_on",
        "next_authorized_action",
        "promotion_boundary",
    }
    run_id = str(spec["run_id"])
    if (
        set(freeze) != expected_top
        or freeze.get("schema_version") != 1
        or freeze.get("program") != "parallax"
        or freeze.get("status") != "FROZEN_ONE_SHOT_PAID_OPTIMIZER_SCALE_AFTER_NO_CREATE_GREEN"
        or freeze.get("scope") != run_id
        or freeze.get("run_id") != run_id
        or freeze.get("automatic_paid_retry") is not False
    ):
        raise OptimizerLaunchError("paid execution freeze is absent or not exact for this run")
    if freeze.get("authorization") != {
        "create_vast_instances": "ONE_GUARDIAN_THEN_ONE_H100",
        "create_vast_volumes": False,
        "h100": "ONE_ONLY",
        "gpu_work": "EXACT_FROZEN_OPTIMIZER_SCALE_MICROGATE",
        "model_reads": "PG1_FUSED_ONLY",
        "data_reads": "TRAIN_ONLY",
        "discovery_evaluation": False,
        "confirmation": False,
        "promotion": False,
        "science_claim": False,
    }:
        raise OptimizerLaunchError("paid execution authorization changed")
    if freeze.get("one_shot") != {
        "run_id": run_id,
        "claim_path": spec["one_shot_authorization"]["claim_path"],
        "terminal_s3_key": spec["result"]["key"],
        "claim_must_be_absent": True,
        "terminal_key_must_be_absent": True,
        "create_failure_consumes_authorization": True,
    }:
        raise OptimizerLaunchError("paid freeze one-shot identity changed")
    if freeze.get("stop_on") != [
        "ANY_HASH_OR_SCHEMA_DRIFT",
        "ANY_NONZERO_VAST_ACCOUNT_BEFORE_CLAIM",
        "ANY_EXISTING_CLAIM_OR_TERMINAL_S3_VERSION",
        "NO_ELIGIBLE_H100_AT_OR_BELOW_PRICE_CEILING",
        "GUARDIAN_NOT_ARMED_OR_NOT_LIVE",
        "WORKER_CREATE_OR_RUNNING_SSH_FAILURE",
        "ANY_PREFLIGHT_OR_CONFOUND_GATE_FAILURE",
        "RESULT_UPLOAD_OR_EXACT_VERSION_READBACK_FAILURE",
        "ANY_TERMINAL_NONZERO_VAST_ACCOUNT",
    ]:
        raise OptimizerLaunchError("paid execution stop policy changed")
    if (
        freeze.get("next_authorized_action")
        != "CREATE_ONE_GUARDIAN_THEN_ONE_LITERAL_CHEAPEST_ELIGIBLE_ALL_REGION_H100_FOR_EXACT_MICROGATE"
        or freeze.get("promotion_boundary")
        != "ONE_PAID_MECHANICAL_TRAIN_ONLY_MICROGATE_NO_AUTOMATIC_RETRY_OR_SCIENTIFIC_CLAIM"
    ):
        raise OptimizerLaunchError("paid execution boundary changed")
    bindings = freeze.get("bound_artifacts", {})
    if set(bindings) != {
        "launch_spec",
        "no_create_rehearsal",
        "planted_tests",
        "s3_execution_amendment",
        "prior_no_paid_freeze",
    }:
        raise OptimizerLaunchError("paid freeze bindings changed")
    for label, row in bindings.items():
        _resolve_bound(root, row, label=f"paid_freeze_{label}")
    expected_spec_path = spec_path.resolve().relative_to(root.resolve()).as_posix()
    if (
        bindings["launch_spec"].get("path") != expected_spec_path
        or bindings["launch_spec"].get("sha256") != sha256_file(spec_path)
    ):
        raise OptimizerLaunchError("paid freeze does not bind the exact launch spec")
    rehearsal_path = root / str(bindings["no_create_rehearsal"]["path"])
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    _require_green_no_create_receipt(
        rehearsal,
        run_id=run_id,
        spec_sha256=sha256_file(spec_path),
        result_key=str(spec["result"]["key"]),
    )
    planted_path = root / str(bindings["planted_tests"]["path"])
    planted = json.loads(planted_path.read_text(encoding="utf-8"))
    _require_green_planted_tests(root, planted)
    prior_path = root / str(bindings["prior_no_paid_freeze"]["path"])
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("status") != "FROZEN_POST_V2_PRECREATE_VOID_NO_PAID_ACTION":
        raise OptimizerLaunchError("paid freeze does not descend from the exact no-paid boundary")
    amendment_path = root / str(bindings["s3_execution_amendment"]["path"])
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("status") != "FROZEN_NONSCIENTIFIC_EXECUTION_AMENDMENT":
        raise OptimizerLaunchError("paid freeze does not bind the S3-only execution amendment")
    return freeze


def _require_green_planted_tests(root: Path, planted: dict[str, Any]) -> None:
    expected_implementations = {
        "launcher": "infra/vast/parallax/ephemeral_optimizer_scale_launch.py",
        "no_create_rehearsal": "infra/vast/parallax/optimizer_scale_no_create_rehearsal.py",
        "transport_launcher": "infra/vast/parallax/ephemeral_s3_launch.py",
        "tests": "infra/vast/parallax/tests/test_ephemeral_optimizer_scale.py",
    }
    if (
        planted.get("status") != "GREEN_69_PASSED"
        or planted.get("passed") != 69
        or planted.get("failed") != 0
        or planted.get("h100_created") is not False
        or planted.get("model_or_data_downloaded") is not False
        or planted.get("paid_action") is not False
        or planted.get("automatic_retry") is not False
    ):
        raise OptimizerLaunchError("paid freeze planted tests are not exact and green")
    implementations = planted.get("bound_implementation", {})
    if set(implementations) != set(expected_implementations):
        raise OptimizerLaunchError("paid freeze planted implementation set changed")
    for label, relative in expected_implementations.items():
        row = implementations[label]
        if row.get("path") != relative or row.get("sha256") != sha256_file(root / relative):
            raise OptimizerLaunchError("paid freeze planted implementation hash mismatch")


def _require_green_no_create_receipt(
    rehearsal: dict[str, Any], *, run_id: str, spec_sha256: str, result_key: str
) -> None:
    """Prove the rehearsal crossed every construction seam with zero mutation."""
    zero_account = {
        "active_instances": 0,
        "active_volumes": 0,
        "instance_ids": [],
        "volume_ids": [],
    }
    zero_mutation = {
        "instances_created": 0,
        "volumes_created": 0,
        "claims_created": 0,
        "S3_objects_written": 0,
        "model_or_data_downloaded": False,
        "GPU_work": False,
    }
    if (
        rehearsal.get("status") != "GREEN_FULL_NO_CREATE_REHEARSAL"
        or rehearsal.get("run_id") != run_id
        or rehearsal.get("launch_spec", {}).get("sha256") != spec_sha256
        or rehearsal.get("initial_account") != zero_account
        or rehearsal.get("terminal_account") != zero_account
        or rehearsal.get("mutation") != zero_mutation
        or rehearsal.get("claim", {}).get("status") != "GREEN_ABSENT"
        or rehearsal.get("result_key")
        != {
            "key": result_key,
            "status": "GREEN_ZERO_VERSIONS_AND_DELETE_MARKERS",
        }
        or rehearsal.get("contract_validation")
        != {
            "guardian": "GREEN_EXACT",
            "optimizer_job_with_exact_presigned_versioned_inputs": "GREEN_EXACT",
            "terminal_worker_with_exact_presigned_S3_result": "GREEN_EXACT",
            "secrets_or_urls_persisted": False,
        }
        or rehearsal.get("guardian_construction", {}).get("status")
        != "GREEN_EXACT_CREATE_ARGUMENTS"
        or rehearsal.get("worker_construction", {}).get("status")
        != "GREEN_EXACT_CREATE_ARGUMENTS"
        or rehearsal.get("guardian_construction", {}).get("secrets_persisted") is not False
        or rehearsal.get("worker_construction", {}).get("secrets_persisted") is not False
        or rehearsal.get("automatic_retry") is not False
        or rehearsal.get("next_boundary")
        != "A_SEPARATE_PAID_EXECUTION_FREEZE_MUST_BIND_THIS_EXACT_RECEIPT_AND_SPEC"
        or rehearsal.get("promotion_boundary")
        != "NO_CREATE_REHEARSAL_ONLY_NO_PAID_OR_SCIENTIFIC_AUTHORIZATION"
    ):
        raise OptimizerLaunchError("paid freeze no-create rehearsal is not exact and green")


def _sanitize_offer(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in PUBLIC_OFFER_FIELDS if field in row}


def select_cheapest_h100_offer(
    contract: dict[str, Any], *, quarantined_machine_ids: set[int], excluded_machine_ids: set[int]
) -> dict[str, Any]:
    all_excluded = quarantined_machine_ids | excluded_machine_ids
    disk = int(contract["disk_gb"])
    query = (
        "gpu_name in [H100_SXM,H100_NVL,H100_PCIE,H100] num_gpus=1 gpu_ram>=80 "
        f"reliability>={float(contract['minimum_reliability']):g} disk_space>={disk} "
        "duration>=1 rentable=true rented=false static_ip=true direct_port_count>=2 "
        f"cuda_max_good>={float(contract['minimum_cuda']):g} "
        f"inet_down>={float(contract['minimum_inet_down_mbps']):g}"
    )
    rows = _rows(
        _vast(
            "search",
            "offers",
            query,
            "--limit",
            "100",
            "--storage",
            str(disk),
            "--order",
            "dph_total",
        )
    )
    rows = [
        row
        for row in rows
        if str(row.get("gpu_name", "")).replace("_", " ").upper() in H100_NAMES
        and int(row.get("num_gpus", 0)) == 1
        and float(row.get("gpu_ram", 0)) >= float(contract["minimum_gpu_ram_mb"])
        and float(row.get("cpu_ram", 0)) >= float(contract["minimum_cpu_ram_mb"])
        and float(row.get("cpu_cores_effective", 0))
        >= float(contract["minimum_cpu_cores_effective"])
        and float(row.get("reliability", 0)) >= float(contract["minimum_reliability"])
        and float(row.get("disk_space", 0)) >= disk
        and float(row.get("disk_bw", 0)) >= float(contract["minimum_disk_bw_mb_s"])
        and float(row.get("cuda_max_good", 0)) >= float(contract["minimum_cuda"])
        and float(row.get("inet_down", 0)) >= float(contract["minimum_inet_down_mbps"])
        and row.get("static_ip") is True
        and int(row.get("direct_port_count", 0)) >= 2
        and int(row.get("machine_id", -1)) not in all_excluded
    ]
    rows.sort(
        key=lambda row: (
            float(row.get("dph_total", float("inf"))),
            -float(row.get("reliability", 0)),
            -float(row.get("inet_down", 0)),
        )
    )
    if not rows:
        raise OptimizerLaunchError("no eligible all-region H100 offer exists")
    selected = rows[0]
    if float(selected.get("dph_total", float("inf"))) > float(contract["price_ceiling_dph"]):
        raise OptimizerLaunchError("literal-cheapest eligible H100 exceeds the frozen price ceiling")
    return {
        "query": query,
        "candidate_count": len(rows),
        "quarantined_machine_ids": sorted(quarantined_machine_ids),
        "other_excluded_machine_ids": sorted(excluded_machine_ids),
        "excluded_machine_ids": sorted(all_excluded),
        "quarantine_applied_before_price_ordering": True,
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "selected": _sanitize_offer(selected),
    }


def wait_h100_ssh(
    instance_id: int,
    key: Path,
    *,
    bootstrap_seconds: int,
    running_ssh_deadline_seconds: int,
    poll_seconds: float = 2.0,
    health_check=lambda: None,
) -> dict[str, Any]:
    """Bound both provider loading and paid time after the first `running` state."""
    bootstrap_deadline = time.monotonic() + bootstrap_seconds
    running_deadline: float | None = None
    terminal = {"exited", "unknown", "offline"}
    observed = False
    attempts = 0
    last_error = "instance never entered running state"
    while time.monotonic() < bootstrap_deadline:
        health_check()
        row = _show_instance(instance_id)
        if row is not None:
            observed = True
        elif observed:
            raise LifecycleError(f"instance {instance_id} disappeared during SSH bootstrap")
        status = (row or {}).get("actual_status")
        if status in terminal:
            raise LifecycleError(f"instance {instance_id} entered terminal bootstrap state {status}")
        if status == "running":
            if running_deadline is None:
                running_deadline = time.monotonic() + running_ssh_deadline_seconds
            remaining = running_deadline - time.monotonic()
            if remaining <= 0:
                raise LifecycleError(
                    f"instance {instance_id} rejected SSH for the frozen post-running deadline; "
                    f"seconds={running_ssh_deadline_seconds}; attempts={attempts}; "
                    f"last_error={last_error}"
                )
            try:
                probe = _run(
                    [*_ssh_base(row or {}, key), "true"],
                    check=False,
                    timeout_seconds=max(0.1, min(10.0, remaining)),
                )
            except subprocess.TimeoutExpired:
                attempts += 1
                last_error = "SSH probe timed out inside post-running deadline"
                continue
            attempts += 1
            if probe.returncode == 0:
                return row or {}
            lines = probe.stderr.decode("utf-8", errors="replace").strip().splitlines()
            last_error = str(lines[-1] if lines else "empty SSH stderr")[:300]
            if time.monotonic() >= running_deadline:
                raise LifecycleError(
                    f"instance {instance_id} rejected SSH for the frozen post-running deadline; "
                    f"seconds={running_ssh_deadline_seconds}; attempts={attempts}; "
                    f"last_error={last_error}"
                )
        time.sleep(poll_seconds)
    raise LifecycleError(
        f"instance {instance_id} never accepted SSH before provider bootstrap timeout; "
        f"attempts={attempts}; last_error={last_error}"
    )


def _require_input_heads(spec: dict[str, Any]) -> dict[str, Any]:
    rows = [*spec["inputs"]["model"], spec["inputs"]["bundle"]]
    verified = []
    for row in rows:
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
            raise OptimizerLaunchError(f"immutable S3 input HEAD failed: {row['key']}")
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
        raise OptimizerLaunchError("one-shot terminal result key already exists")


def _guardian_contract(spec: dict[str, Any], *, run_id: str, instance_id: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_GUARDIAN",
        "run_id": run_id,
        "worker_label": f"parallax-worker-{run_id}",
        "self_instance_id": instance_id,
        "scoped_api_key_path": "/run/parallax/scoped-lifecycle-key",
        "self_api_key_path": "/run/parallax/self-instance-key",
        "receipt_path": "/opt/parallax/guardian-receipt.json",
        "armed_path": "/opt/parallax/guardian-armed.json",
        "worker_assignment_path": "/run/parallax/worker-assignment.json",
        "worker_assignment_ack_path": "/opt/parallax/guardian-assignment-ack.json",
        "discovery_seconds": int(spec["guardian"]["discovery_seconds"]),
        "provider_loading_seconds": int(spec["guardian"]["provider_loading_seconds"]),
        "hard_runtime_seconds": int(spec["guardian"]["worker_hard_runtime_seconds"]),
        "poll_seconds": 1,
        "consecutive_absent_observations": 10,
    }


def _worker_assignment(*, run_id: str, worker_id: int) -> dict[str, Any]:
    if worker_id <= 0 or not run_id:
        raise OptimizerLaunchError("guardian worker assignment identity is invalid")
    return {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": run_id,
        "worker_label": f"parallax-worker-{run_id}",
        "worker_id": worker_id,
    }


def _worker_assignment_ack(*, run_id: str, worker_id: int) -> dict[str, Any]:
    return {
        **_worker_assignment(run_id=run_id, worker_id=worker_id),
        "status": "GUARDIAN_WORKER_ASSIGNMENT_ACKNOWLEDGED",
    }


def _presigned_inputs(spec: dict[str, Any], credentials: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    now = datetime.now(timezone.utc)

    def signed(row: dict[str, Any]) -> dict[str, Any]:
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

    return [signed(row) for row in spec["inputs"]["model"]], signed(spec["inputs"]["bundle"])


def _job_contract(
    root: Path,
    spec: dict[str, Any],
    *,
    instance_id: int,
    model: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = _resolve_bound(root, spec["inputs"]["bundle_receipt"], label="bundle receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_OPTIMIZER_SCALE_JOB",
        "stage": "training_optimizer_scale_microgate",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": {**bundle, "entries": receipt["entries"]},
        "source_contract_sha256": receipt["source_contract_sha256"],
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/parallax-training-optimizer-scale-v1.tar.gz",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
                "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
            ),
            "preflight_output": "/workspace/parallax/preflight.json",
            "optimizer_output": "/workspace/parallax/optimizer-result.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": spec["job"]["bounds"],
        "firewall": {
            "confirmation_manifest_present": False,
            "confirmation_images_present": False,
            "discovery_images_removed_before_model_load": True,
            "discovery_images_opened_by_optimizer": 0,
            "discovery_items_evaluated": 0,
            "promotion_eligible": False,
            "automatic_retry": False,
        },
    }


def _worker_contract(
    spec: dict[str, Any], *, run_id: str, instance_id: int, result_url: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "run_id": run_id,
        "stage": spec["job"]["stage"],
        "instance_id": instance_id,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": spec["job"]["command"],
        "maximum_job_seconds": int(spec["job"]["maximum_seconds"]),
        "maximum_result_bytes": int(spec["result"]["maximum_bytes"]),
        "result_key": spec["result"]["key"],
        "result_put_url": result_url,
        "post_upload_action": "destroy",
        "upload_failure_action": "stop",
        "environment": {},
    }


def run(root: Path, spec_path: Path, freeze_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    validate_execution_freeze(root, spec_path, freeze_path, spec)
    quarantine = require_machine_quarantine(root, spec)
    initial = _require_initial_zero()
    s3_safety = require_s3_safety()
    inputs = _require_input_heads(spec)
    _require_result_absent(spec["result"]["key"])
    _claim(root, spec)
    run_id = str(spec["run_id"])
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise OptimizerLaunchError("SSH or scoped lifecycle key is absent")
    guardian_id: int | None = None
    worker_id: int | None = None
    guardian_market: dict[str, Any] | None = None
    worker_market: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    terminal_body: bytes | None = None
    s3_proof: dict[str, Any] | None = None
    job_released = False
    started = time.monotonic()
    cleanup: dict[str, Any] | None = None
    try:
        guardian_market = select_cheapest_offer(role="guardian", contract=spec["guardian"])
        guardian_create = _create_instance(
            int(guardian_market["selected"]["id"]),
            _create_arguments(role="guardian", spec=spec, run_id=run_id),
        )
        guardian_id = int(guardian_create["new_contract"])
        guardian_row = wait_ssh(
            guardian_id,
            key_path,
            timeout_seconds=int(spec["guardian"]["bootstrap_seconds"]),
        )
        guardian_ssh = _ssh_base(guardian_row, key_path)
        guardian_contract = _guardian_contract(spec, run_id=run_id, instance_id=guardian_id)
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
        started_guardian = _run(
            [
                *guardian_ssh,
                "nohup python3 /opt/parallax/ephemeral_guardian.py --contract "
                "/run/parallax/guardian-contract.json >/opt/parallax/guardian.log 2>&1 </dev/null &",
            ],
            check=False,
            timeout_seconds=30,
        )
        if started_guardian.returncode != 0:
            raise OptimizerLaunchError("guardian process did not start")
        _wait_remote_file(guardian_ssh, guardian_contract["armed_path"])
        _require_guardian_live(guardian_id)

        worker_market = select_cheapest_h100_offer(
            spec["worker"],
            quarantined_machine_ids=set(quarantine["active_machine_ids"]),
            excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
        )
        worker_create = _create_instance(
            int(worker_market["selected"]["id"]),
            _create_arguments(role="worker", spec=spec, run_id=run_id),
        )
        worker_id = int(worker_create["new_contract"])
        _stage_atomic(
            guardian_ssh,
            remote=guardian_contract["worker_assignment_path"],
            content=json.dumps(
                _worker_assignment(run_id=run_id, worker_id=worker_id),
                sort_keys=True,
            ).encode("utf-8"),
        )
        _wait_remote_json_exact(
            guardian_ssh,
            guardian_contract["worker_assignment_ack_path"],
            _worker_assignment_ack(run_id=run_id, worker_id=worker_id),
        )
        _require_guardian_live(guardian_id)
        worker_row = wait_h100_ssh(
            worker_id,
            key_path,
            bootstrap_seconds=int(spec["worker"]["bootstrap_seconds"]),
            running_ssh_deadline_seconds=int(
                spec["worker"]["running_ssh_deadline_seconds"]
            ),
            health_check=lambda: _require_guardian_live(guardian_id),
        )
        _require_guardian_live(guardian_id)
        worker_ssh = _ssh_base(worker_row, key_path)
        credentials = _aws_credentials()
        model, bundle = _presigned_inputs(spec, credentials)
        job_contract = _job_contract(
            root,
            spec,
            instance_id=worker_id,
            model=model,
            bundle=bundle,
        )
        result_url = presign_result_put(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            bucket=BUCKET,
            key=spec["result"]["key"],
            region=REGION,
            now=datetime.now(timezone.utc),
            expires_seconds=int(spec["result"]["put_url_seconds"]),
        )
        credentials.clear()
        worker_contract = _worker_contract(
            spec,
            run_id=run_id,
            instance_id=worker_id,
            result_url=result_url,
        )
        _stage(
            worker_ssh,
            remote="/opt/parallax/ephemeral_s3_worker.py",
            content=(root / "infra/vast/parallax/ephemeral_s3_worker.py").read_bytes(),
            mode="0700",
        )
        _stage(
            worker_ssh,
            remote="/opt/parallax/ephemeral_optimizer_scale_job.py",
            content=(root / "infra/vast/parallax/ephemeral_optimizer_scale_job.py").read_bytes(),
            mode="0700",
        )
        _stage(
            worker_ssh,
            remote="/run/parallax/optimizer-job-contract.json",
            content=json.dumps(job_contract, sort_keys=True).encode("utf-8"),
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
        job_contract.clear()
        job_released = True
        started_worker = _run(
            [*worker_ssh, _worker_start_command()],
            check=False,
            timeout_seconds=30,
        )
        if started_worker.returncode != 0:
            raise OptimizerLaunchError("optimizer worker process did not start")
        terminal, terminal_body, s3_proof = _wait_result(
            key=spec["result"]["key"],
            run_id=run_id,
            instance_id=worker_id,
            seconds=int(spec["result"]["wait_seconds"]),
        )
    finally:
        cleanup = _finalize_lifecycle(
            worker_id=worker_id,
            guardian_id=guardian_id,
            job_released=job_released,
            s3_durable=s3_proof is not None,
        )
    final = _confirm_account_zero(seconds=45)
    if terminal is None or terminal_body is None or s3_proof is None:
        raise OptimizerLaunchError("optimizer terminal S3 result is absent")
    job_result = terminal.get("job", {}).get("result", {}).get("payload", {})
    mechanical_green = terminal.get("status") == "GREEN_JOB_TERMINAL"
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_OPTIMIZER_MICROGATE_EXECUTED" if mechanical_green else "RED_OPTIMIZER_MICROGATE_EXECUTION",
        "run_id": run_id,
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "scope": "PG1_TRAIN_ONLY_MECHANICAL_NONPROMOTION",
        "initial_account": initial,
        "s3_safety": s3_safety,
        "input_proof": inputs,
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_instance_id": guardian_id,
        "worker_instance_id": worker_id,
        "terminal": terminal,
        "optimizer_adjudication": job_result.get("adjudication"),
        "execution_provenance": job_result.get("execution_provenance"),
        "s3_proof": s3_proof,
        "cleanup": cleanup,
        "terminal_account": final,
        "wall_seconds": time.monotonic() - started,
        "automatic_retry": False,
        "promotion_boundary": "MECHANICAL_SCALE_SELECTION_ONLY_NO_SCIENTIFIC_CLAIM_OR_TASK_PROMOTION",
    }
    output_relative = Path(str(spec["output_receipt"].get("path", "")))
    if output_relative.is_absolute() or not output_relative.parts or ".." in output_relative.parts:
        raise OptimizerLaunchError("optimizer receipt path is unsafe")
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
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
