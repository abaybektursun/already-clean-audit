"""Launch one S3-durable supporting-lineage optimizer microgate on Vast H100.

The launcher arms an independent cheap guardian before creating the literal
cheapest currently eligible H100 across all regions.  Every input is an exact
versioned S3 object; Vast disk is disposable scratch only.  The worker uploads
and verifies its terminal packet before self-destruction.  There is no retry
path and no persistent Vast volume operation.
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
from .ephemeral_optimizer_scale_launch import (
    RUNTIME_IMAGE,
    RUNTIME_TEMPLATE,
    _guardian_contract,
    _worker_contract,
    require_machine_quarantine,
    select_cheapest_h100_offer,
    wait_h100_ssh,
)
from .ephemeral_pg2_optimizer_scale_job import (
    EXPECTED_BOUNDS,
    EXPECTED_BUNDLE_BYTES,
    EXPECTED_BUNDLE_SHA256,
    EXPECTED_FIREWALL,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILES,
    EXPECTED_MODEL_NAMES,
    EXPECTED_OVERLAYS,
    EXPECTED_REPO,
    EXPECTED_REVISION,
    EXPECTED_SOURCE_CONTRACT_SHA256,
)
from .ephemeral_s3_launch import (
    AWS_PROFILE,
    BUCKET,
    REGION,
    EphemeralLaunchError,
    _account_zero,
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
from .guarded_launch import _create_instance, _run, _ssh_base, wait_ssh
from .s3_presign_put import presign_get, presign_result_put


class PG2OptimizerLaunchError(EphemeralLaunchError):
    pass


EXPECTED_AUTHORIZATION = {
    "paid_microgate": True,
    "h100": True,
    "model_reads": True,
    "train_data_reads": True,
    "training": True,
    "discovery_evaluation": False,
    "confirmation": False,
    "promotion": False,
    "science_claim": False,
}
EXPECTED_STORAGE = {
    "durable_provider": "amazon_s3",
    "aws_profile": AWS_PROFILE,
    "region": REGION,
    "bucket": BUCKET,
    "vast_persistent_volumes": "FORBIDDEN",
    "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
    "maximum_ephemeral_scratch_gb": 1000,
}
EXPECTED_JOB = {
    "stage": "training_optimizer_scale_microgate",
    "maximum_seconds": 720,
    "command": [
        "python3",
        "/opt/parallax/ephemeral_pg2_optimizer_scale_job.py",
        "--contract",
        "/run/parallax/pg2-optimizer-job-contract.json",
    ],
    "environment": {},
    "bounds": EXPECTED_BOUNDS,
}
EXPECTED_RUN_ID = "parallax-pg2-optimizer-scale-s3-v3"
EXPECTED_INVARIANTS = [
    "supporting_PG2_is_not_part_of_the_primary_PG1_to_pi_causal_chain",
    "same_candidate_LR_ladder_is_recalibrated_not_transferred_from_PG1",
    "raw_cross_generation_objectives_are_not_compared",
    "launcher_requires_a_separate_paid_execution_freeze",
    "paid_freeze_binds_exact_green_plants_and_no_create_rehearsal",
    "guardian_armed_on_a_different_machine_before_H100_create",
    "guardian_acknowledges_exact_worker_ID_before_H100_SSH_wait",
    "assigned_guardian_survives_unobservable_label_until_bounded_provider_timeout",
    "machine_quarantine_applied_before_price_ordering",
    "60_second_post_running_SSH_access_deadline",
    "literal_cheapest_eligible_all_region_H100_requeried_at_mutation_time",
    "all_small_source_and_prior_dynamic_evidence_verified_before_6.1GB_hydration",
    "all_16_inputs_are_exact_versioned_checksum_verified_S3_objects",
    "no_AWS_credentials_on_Vast",
    "no_persistent_Vast_volume_or_gateway",
    "24_discovery_images_removed_before_model_load",
    "zero_discovery_or_confirmation_evaluation",
    "train_only_six_image_repeated_probe",
    "terminal_JSON_atomic_then_versioned_S3_HTTP_200_plus_version_id_then_self_DESTROY",
    "S3_failure_means_STOP_and_bounded_scratch_preservation",
    "pre_release_and_post_durability_destroy_are_unconditional_idempotent_requests",
    "released_nondurable_stop_is_an_unconditional_billing_ending_request",
    "known_instance_cleanup_requires_three_confirmed_absences",
    "terminal_account_zero_requires_three_consecutive_observations",
    "840_second_guardian_worker_hard_runtime",
    "no_automatic_paid_retry",
    "terminal_zero_instances_zero_volumes",
]


def _digest(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PG2OptimizerLaunchError(f"{label} is not a lowercase SHA-256")
    return digest


def _resolve_bound(root: Path, row: dict[str, Any], *, label: str) -> Path:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PG2OptimizerLaunchError(f"unsafe {label} path")
    path = (root / relative).resolve()
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise PG2OptimizerLaunchError(f"{label} hash mismatch")
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
        raise PG2OptimizerLaunchError(f"{label} is not an exact versioned S3 input")
    return row


def _validate_prior_pg2_evidence(root: Path, spec: dict[str, Any]) -> None:
    bindings = spec["bound_artifacts"]
    result = json.loads(_resolve_bound(root, bindings["prior_pg2_dynamic_gate"], label="prior gate").read_text(encoding="utf-8"))
    contract = json.loads(_resolve_bound(root, bindings["prior_pg2_dynamic_source_contract"], label="prior gate contract").read_text(encoding="utf-8"))
    pg2 = result.get("lineages", {}).get("pg2", {})
    if (
        result.get("status") != "GREEN"
        or pg2.get("checkpoint") != {"repo": EXPECTED_REPO, "revision": EXPECTED_REVISION}
        or pg2.get("zero_delta_logits_bit_exact") is not True
        or float(pg2.get("update", {}).get("gradient_norm", 0)) <= 0
        or contract.get("status") != "FROZEN"
        or contract.get("artifacts", {}).get("training_model", {}).get("sha256")
        != spec["bound_artifacts"]["source_contract_training_model"]["sha256"]
    ):
        raise PG2OptimizerLaunchError("prior exact PG2 dynamic gate is not a valid prerequisite")


def _validate_post_incident_lifecycle_evidence(root: Path, spec: dict[str, Any]) -> None:
    bindings = spec["bound_artifacts"]
    incident = json.loads(
        _resolve_bound(root, bindings["v1_void_incident"], label="PG2 v1 void incident").read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        _resolve_bound(
            root, bindings["lifecycle_proof_receipt"], label="live lifecycle proof"
        ).read_text(encoding="utf-8")
    )
    terminal_path = _resolve_bound(
        root, bindings["lifecycle_proof_terminal"], label="live lifecycle terminal"
    )
    lifecycle_freeze = json.loads(
        _resolve_bound(
            root, bindings["post_lifecycle_freeze"], label="post-lifecycle freeze"
        ).read_text(encoding="utf-8")
    )
    tests = json.loads(
        _resolve_bound(
            root, bindings["lifecycle_hardening_tests"], label="lifecycle hardening tests"
        ).read_text(encoding="utf-8")
    )
    v2_incident = json.loads(
        _resolve_bound(root, bindings["v2_void_incident"], label="PG2 v2 void incident").read_text(
            encoding="utf-8"
        )
    )
    post_v2_freeze = json.loads(
        _resolve_bound(root, bindings["post_v2_freeze"], label="post-v2 freeze").read_text(
            encoding="utf-8"
        )
    )
    stage_tests = json.loads(
        _resolve_bound(root, bindings["stage_retry_tests"], label="stage retry tests").read_text(
            encoding="utf-8"
        )
    )
    cleanup = receipt.get("cleanup", {}).get("confirmation", {})
    account = receipt.get("terminal_account", {})
    proof = receipt.get("s3_proof", {})
    if (
        incident.get("status") != "VOID_PRE_SSH_GUARDIAN_ABSENT_NO_SCIENCE"
        or incident.get("failure", {}).get("model_loads") != 0
        or incident.get("failure", {}).get("training_updates") != 0
        or receipt.get("status") != "GREEN_CHEAP_LIVE_ASSIGNMENT_CLEANUP_AND_S3_PROOF"
        or receipt.get("guardian_acknowledgement", {}).get("worker_id")
        != receipt.get("worker_instance_id")
        or cleanup.get("status") != "GREEN_KNOWN_INSTANCES_CONFIRMED_ABSENT"
        or cleanup.get("consecutive_absences") != 3
        or account.get("status") != "GREEN_ACCOUNT_ZERO_CONFIRMED"
        or account.get("consecutive_observations") != 3
        or account.get("active_instances") != 0
        or account.get("active_volumes") != 0
        or proof.get("status") != "GREEN_EXACT_VERSIONED_S3_READBACK"
        or proof.get("sha256") != sha256_file(terminal_path)
        or lifecycle_freeze.get("status")
        != "FROZEN_POST_CHEAP_LIFECYCLE_GREEN_ALL_PAID_ACTION_OFF"
        or any(lifecycle_freeze.get("authorization", {}).values())
        or tests.get("status") != "GREEN_LIFECYCLE_HARDENING_TESTS"
        or int(tests.get("tests_passed", 0)) < 328
        or tests.get("tests_failed") != 0
        or tests.get("paid_action") is not False
        or v2_incident.get("status")
        != "VOID_PRE_H100_GUARDIAN_SSH_STAGE_FAILURE_NO_SCIENCE"
        or v2_incident.get("execution_boundary", {}).get("H100_created") is not False
        or v2_incident.get("execution_boundary", {}).get("model_loads") != 0
        or v2_incident.get("execution_boundary", {}).get("training_updates") != 0
        or post_v2_freeze.get("status")
        != "FROZEN_POST_PG2_V2_PRE_H100_VOID_ALL_PAID_ACTION_OFF"
        or any(post_v2_freeze.get("authorization", {}).values())
        or stage_tests.get("status") != "GREEN_PG2_V3_STAGE_RETRY_TESTS"
        or int(stage_tests.get("tests_passed", 0)) < 334
        or stage_tests.get("tests_failed") != 0
        or stage_tests.get("paid_action") is not False
    ):
        raise PG2OptimizerLaunchError("post-incident lifecycle evidence is not exact and green")


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
        raise PG2OptimizerLaunchError("PG2 optimizer launch spec schema changed")
    if (
        spec.get("program") != "parallax"
        or spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_H100"
        or spec.get("scope") != "pg2_supporting_optimizer_scale_microgate"
        or spec.get("provider") != "vast.ai"
        or spec.get("automatic_paid_retry") is not False
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer launch escapes its frozen scope")
    run_id = str(spec.get("run_id", ""))
    if (
        run_id != EXPECTED_RUN_ID
        or spec.get("one_shot_authorization")
        != {
            "id": EXPECTED_RUN_ID,
            "claim_path": (
                "studies/lineage/cross-lineage/results/readiness/claims/"
                "parallax_pg2_optimizer_scale_s3_v3.json"
            ),
        }
        or spec.get("output_receipt")
        != {
            "path": (
                "studies/lineage/cross-lineage/results/"
                "parallax_training_optimizer_scale_gate_pg2_v3/receipt.json"
            )
        }
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer one-shot identity changed")
    if spec.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PG2OptimizerLaunchError("PG2 optimizer authorization changed")
    if spec.get("storage") != EXPECTED_STORAGE:
        raise PG2OptimizerLaunchError("PG2 optimizer storage is not exact AWS-only durability")
    if spec.get("worker_image") != RUNTIME_IMAGE:
        raise PG2OptimizerLaunchError("PG2 optimizer runtime image changed")
    if spec.get("worker_registry_auth") != {
        "type": "aws_ecr_ephemeral",
        "profile": AWS_PROFILE,
        "region": REGION,
        "registry": "769063704696.dkr.ecr.us-east-1.amazonaws.com",
        "username": "AWS",
        "credential_source": "aws_ecr_get_login_password",
        "persist_secret": False,
        "maximum_token_age_seconds": 43_200,
    }:
        raise PG2OptimizerLaunchError("PG2 optimizer registry authorization changed")
    guardian = spec.get("guardian", {})
    if (
        guardian.get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time"
        or guardian.get("gpu_kind") != "non_h100_mechanical_guard"
        or guardian.get("disk_gb") != 10
        or not 0 < float(guardian.get("price_ceiling_dph", 0)) <= 0.12
        or guardian.get("worker_hard_runtime_seconds") != 840
        or guardian.get("hard_destroy_seconds") != 1800
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer guardian contract changed")
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
        or worker.get("running_ssh_deadline_seconds") != 60
        or worker.get("hard_destroy_seconds") != 1080
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer H100 contract changed")
    if spec.get("job") != EXPECTED_JOB:
        raise PG2OptimizerLaunchError("PG2 optimizer job command or bounds changed")

    inputs = spec.get("inputs", {})
    model = inputs.get("model")
    if not isinstance(model, list) or len(model) != EXPECTED_MODEL_FILES:
        raise PG2OptimizerLaunchError("PG2 optimizer launch must bind all nine model objects")
    names = set()
    for row in model:
        _validate_input(row, label="PG2 model input")
        names.add(str(row.get("path")))
    if names != EXPECTED_MODEL_NAMES or sum(int(row["bytes"]) for row in model) != EXPECTED_MODEL_BYTES:
        raise PG2OptimizerLaunchError("PG2 optimizer model inventory changed")
    bundle = _validate_input(inputs.get("bundle", {}), label="base source bundle")
    if bundle.get("bytes") != EXPECTED_BUNDLE_BYTES or bundle.get("sha256") != EXPECTED_BUNDLE_SHA256:
        raise PG2OptimizerLaunchError("PG2 optimizer base source bundle changed")
    bundle_receipt = json.loads(_resolve_bound(root, inputs.get("bundle_receipt", {}), label="base bundle receipt").read_text(encoding="utf-8"))
    if (
        bundle_receipt.get("status") != "GREEN"
        or bundle_receipt.get("bundle_sha256") != EXPECTED_BUNDLE_SHA256
        or bundle_receipt.get("bundle_bytes") != EXPECTED_BUNDLE_BYTES
        or len(bundle_receipt.get("entries", [])) != 155
        or bundle_receipt.get("confirmation_image_included") is not False
        or bundle_receipt.get("confirmation_manifest_included") is not False
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer base bundle receipt changed")
    overlays = inputs.get("overlays")
    if not isinstance(overlays, list) or len(overlays) != len(EXPECTED_OVERLAYS):
        raise PG2OptimizerLaunchError("PG2 optimizer launch must bind six overlays")
    observed = {}
    for row in overlays:
        _validate_input(row, label="PG2 overlay")
        observed[str(row.get("path"))] = str(row.get("sha256"))
    if observed != EXPECTED_OVERLAYS:
        raise PG2OptimizerLaunchError("PG2 optimizer overlay identities changed")
    overlay_receipt = json.loads(_resolve_bound(root, inputs.get("overlay_receipt", {}), label="PG2 overlay receipt").read_text(encoding="utf-8"))
    if (
        overlay_receipt.get("status") != "GREEN_PG2_OPTIMIZER_OVERLAYS_S3"
        or overlay_receipt.get("object_count") != 6
        or [
            {key: row[key] for key in ("path", "bytes", "sha256", "bucket", "region", "key", "version_id")}
            for row in overlay_receipt.get("objects", [])
        ]
        != [
            {key: row[key] for key in ("path", "bytes", "sha256", "bucket", "region", "key", "version_id")}
            for row in overlays
        ]
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer overlay receipt disagrees with launch inputs")

    result = spec.get("result", {})
    if result != {
        "key": f"parallax/results/{run_id}/terminal.json",
        "maximum_bytes": 1024 * 1024,
        "put_url_seconds": 1200,
        "wait_seconds": 900,
        "required_http_status": 200,
        "required_version_id": True,
        "required_encryption": "AES256",
        "post_upload_action": "destroy",
        "upload_failure_action": "stop",
    }:
        raise PG2OptimizerLaunchError("PG2 optimizer terminal result contract changed")
    if spec.get("invariants") != EXPECTED_INVARIANTS:
        raise PG2OptimizerLaunchError("PG2 optimizer invariants changed")
    if spec.get("promotion_boundary") != "PG2_SUPPORTING_MECHANICAL_SCALE_ONLY_NO_SCIENTIFIC_CLAIM_PROMOTION_OR_PRIMARY_CHAIN_INFERENCE":
        raise PG2OptimizerLaunchError("PG2 optimizer promotion boundary changed")

    required_bindings = {
        "research_spec",
        "design_audit",
        "matrix",
        "dependency_audit",
        "storage_policy",
        "program_s3_amendment",
        "source_contract",
        "source_contract_training_model",
        "optimizer_config",
        "checkpoint_hydration_manifest",
        "checkpoint_stage_adjudication",
        "overlay_receipt",
        "prior_pg2_dynamic_gate",
        "prior_pg2_dynamic_source_contract",
        "base_bundle_receipt",
        "runtime_registry",
        "runtime_h100_linkage",
        "managed_template",
        "scoped_lifecycle_authority",
        "machine_quarantine",
        "planted_tests",
        "readiness",
        "v1_void_incident",
        "lifecycle_proof_receipt",
        "lifecycle_proof_terminal",
        "lifecycle_hardening_tests",
        "post_lifecycle_freeze",
        "v2_void_incident",
        "post_v2_freeze",
        "stage_retry_tests",
    }
    if set(spec.get("bound_artifacts", {})) != required_bindings:
        raise PG2OptimizerLaunchError("PG2 optimizer bound artifact set changed")
    for label, row in spec["bound_artifacts"].items():
        _resolve_bound(root, row, label=label)
    if spec["bound_artifacts"]["source_contract"]["sha256"] != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise PG2OptimizerLaunchError("PG2 optimizer source contract binding changed")
    _validate_prior_pg2_evidence(root, spec)
    _validate_post_incident_lifecycle_evidence(root, spec)

    expected_implementation = {
        "launcher": "infra/vast/parallax/ephemeral_pg2_optimizer_scale_launch.py",
        "job": "infra/vast/parallax/ephemeral_pg2_optimizer_scale_job.py",
        "base_job_helper": "infra/vast/parallax/ephemeral_optimizer_scale_job.py",
        "worker": "infra/vast/parallax/ephemeral_s3_worker.py",
        "guardian": "infra/vast/parallax/ephemeral_guardian.py",
        "presigner": "infra/vast/parallax/s3_presign_put.py",
        "transport_launcher": "infra/vast/parallax/ephemeral_s3_launch.py",
        "vast_launcher": "infra/vast/parallax/guarded_launch.py",
        "lifecycle_supervisor": "infra/vast/parallax/lifecycle_supervisor.py",
        "artifacts": "infra/vast/parallax/artifacts.py",
        "no_create_rehearsal": "infra/vast/parallax/pg2_optimizer_scale_no_create_rehearsal.py",
        "overlay_stage": "infra/vast/parallax/pg2_optimizer_overlay_stage.py",
    }
    if set(spec.get("implementation", {})) != set(expected_implementation):
        raise PG2OptimizerLaunchError("PG2 optimizer implementation set changed")
    for label, relative in expected_implementation.items():
        row = spec["implementation"][label]
        if row.get("path") != relative or sha256_file(root / relative) != row.get("sha256"):
            raise PG2OptimizerLaunchError(f"PG2 optimizer {label} implementation hash mismatch")
    require_machine_quarantine(root, spec)
    return spec


def _require_green_planted_tests(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_bound(root, row, label="paid freeze planted tests")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "GREEN_PG2_OPTIMIZER_PLANTED_TESTS"
        or int(receipt.get("tests_passed", 0)) < 17
        or receipt.get("tests_failed") != 0
        or receipt.get("paid_action") is not False
        or receipt.get("vast_resources_created") != 0
        or receipt.get("gpu_seconds") != 0
        or receipt.get("source_contract_sha256") != EXPECTED_SOURCE_CONTRACT_SHA256
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer planted tests are not exact and green")
    return receipt


def _require_green_no_create_receipt(
    receipt: dict[str, Any], *, run_id: str, spec_sha256: str, result_key: str
) -> None:
    def account_is_zero(row: object) -> bool:
        return isinstance(row, dict) and (
            row.get("active_instances") == 0
            and row.get("active_volumes") == 0
            and row.get("instance_ids", []) == []
            and row.get("volume_ids", []) == []
        )

    if (
        receipt.get("status") != "GREEN_FULL_PG2_NO_CREATE_REHEARSAL"
        or receipt.get("run_id") != run_id
        or receipt.get("launch_spec", {}).get("sha256") != spec_sha256
        or not account_is_zero(receipt.get("initial_account"))
        or not account_is_zero(receipt.get("terminal_account"))
        or receipt.get("result_key", {}).get("key") != result_key
        or receipt.get("mutation")
        != {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "S3_objects_written": 0,
            "model_or_data_downloaded": False,
            "GPU_work": False,
        }
        or receipt.get("automatic_retry") is not False
    ):
        raise PG2OptimizerLaunchError("PG2 optimizer no-create rehearsal is not exact and green")


def validate_execution_freeze(
    root: Path, spec_path: Path, freeze_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
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
        or freeze.get("status")
        != "FROZEN_ONE_SHOT_PAID_PG2_OPTIMIZER_SCALE_AFTER_NO_CREATE_GREEN"
        or freeze.get("scope") != run_id
        or freeze.get("run_id") != run_id
        or freeze.get("automatic_paid_retry") is not False
    ):
        raise PG2OptimizerLaunchError("PG2 paid execution freeze is absent or inexact")
    if freeze.get("authorization") != {
        "create_vast_instances": "ONE_GUARDIAN_THEN_ONE_H100",
        "create_vast_volumes": False,
        "h100": "ONE_ONLY",
        "gpu_work": "EXACT_FROZEN_PG2_OPTIMIZER_SCALE_MICROGATE",
        "model_reads": "PG2_FUSED_ONLY",
        "data_reads": "TRAIN_ONLY",
        "discovery_evaluation": False,
        "confirmation": False,
        "promotion": False,
        "science_claim": False,
    }:
        raise PG2OptimizerLaunchError("PG2 paid execution authorization changed")
    if freeze.get("one_shot") != {
        "run_id": run_id,
        "claim_path": spec["one_shot_authorization"]["claim_path"],
        "terminal_s3_key": spec["result"]["key"],
        "claim_must_be_absent": True,
        "terminal_key_must_be_absent": True,
        "create_failure_consumes_authorization": True,
    }:
        raise PG2OptimizerLaunchError("PG2 paid one-shot identity changed")
    if freeze.get("stop_on") != [
        "ANY_HASH_OR_SCHEMA_DRIFT",
        "ANY_NONZERO_VAST_ACCOUNT_BEFORE_CLAIM",
        "ANY_EXISTING_CLAIM_OR_TERMINAL_S3_VERSION",
        "NO_ELIGIBLE_H100_AT_OR_BELOW_PRICE_CEILING",
        "GUARDIAN_NOT_ARMED_LIVE_OR_EXACT_WORKER_ID_ACKNOWLEDGED",
        "WORKER_CREATE_OR_RUNNING_SSH_FAILURE",
        "ANY_PREFLIGHT_OR_CONFOUND_GATE_FAILURE",
        "RESULT_UPLOAD_OR_EXACT_VERSION_READBACK_FAILURE",
        "ANY_DESTROY_STOP_OR_THREE_OBSERVATION_ACCOUNT_ZERO_FAILURE",
    ]:
        raise PG2OptimizerLaunchError("PG2 paid stop policy changed")
    if (
        freeze.get("next_authorized_action")
        != "CREATE_ONE_GUARDIAN_THEN_ONE_LITERAL_CHEAPEST_ELIGIBLE_ALL_REGION_H100_FOR_EXACT_PG2_MICROGATE"
        or freeze.get("promotion_boundary")
        != "ONE_PAID_SUPPORTING_MECHANICAL_TRAIN_ONLY_MICROGATE_NO_RETRY_SCIENTIFIC_CLAIM_OR_PRIMARY_CHAIN_INFERENCE"
    ):
        raise PG2OptimizerLaunchError("PG2 paid execution boundary changed")
    bindings = freeze.get("bound_artifacts", {})
    if set(bindings) != {
        "launch_spec",
        "no_create_rehearsal",
        "planted_tests",
        "program_s3_amendment",
        "post_lifecycle_freeze",
        "lifecycle_proof_receipt",
    }:
        raise PG2OptimizerLaunchError("PG2 paid freeze binding set changed")
    for label, row in bindings.items():
        _resolve_bound(root, row, label=f"PG2 paid freeze {label}")
    expected_spec_path = spec_path.resolve().relative_to(root.resolve()).as_posix()
    if (
        bindings["launch_spec"].get("path") != expected_spec_path
        or bindings["launch_spec"].get("sha256") != sha256_file(spec_path)
    ):
        raise PG2OptimizerLaunchError("PG2 paid freeze does not bind the exact launch spec")
    rehearsal = json.loads((root / bindings["no_create_rehearsal"]["path"]).read_text(encoding="utf-8"))
    _require_green_no_create_receipt(
        rehearsal,
        run_id=run_id,
        spec_sha256=sha256_file(spec_path),
        result_key=spec["result"]["key"],
    )
    _require_green_planted_tests(root, bindings["planted_tests"])
    prior = json.loads((root / bindings["post_lifecycle_freeze"]["path"]).read_text(encoding="utf-8"))
    if any(prior.get("authorization", {}).values()):
        raise PG2OptimizerLaunchError("prior execution freeze is not zero-paid")
    proof = json.loads((root / bindings["lifecycle_proof_receipt"]["path"]).read_text(encoding="utf-8"))
    if proof.get("status") != "GREEN_CHEAP_LIVE_ASSIGNMENT_CLEANUP_AND_S3_PROOF":
        raise PG2OptimizerLaunchError("paid freeze lacks the exact live lifecycle proof")
    return freeze


def _require_input_heads(spec: dict[str, Any]) -> dict[str, Any]:
    rows = [*spec["inputs"]["model"], spec["inputs"]["bundle"], *spec["inputs"]["overlays"]]
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
            raise PG2OptimizerLaunchError(f"immutable PG2 S3 input HEAD failed: {row['key']}")
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
        "status": "GREEN_EXACT_IMMUTABLE_PG2_INPUTS",
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
        raise PG2OptimizerLaunchError("PG2 one-shot terminal result key already exists")


def _presigned_inputs(
    spec: dict[str, Any], credentials: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
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
                expires_seconds=1200,
            ),
        }

    return (
        [signed(row) for row in spec["inputs"]["model"]],
        signed(spec["inputs"]["bundle"]),
        [signed(row) for row in spec["inputs"]["overlays"]],
    )


def _job_contract(
    root: Path,
    spec: dict[str, Any],
    *,
    instance_id: int,
    model: list[dict[str, Any]],
    bundle: dict[str, Any],
    overlays: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = json.loads(_resolve_bound(root, spec["inputs"]["bundle_receipt"], label="base bundle receipt").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_OPTIMIZER_SCALE_PG2_JOB",
        "stage": "training_optimizer_scale_microgate",
        "provider": "vast.ai",
        "lineage": {
            "id": "pg2",
            "role": "supporting",
            "repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
        },
        "instance_id": instance_id,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": {**bundle, "entries": receipt["entries"]},
        "overlays": overlays,
        "source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/parallax-training-optimizer-scale-v1.tar.gz",
            "overlay_root": "/workspace/parallax/input/overlays",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/"
                f"snapshots/{EXPECTED_REVISION}"
            ),
            "preflight_output": "/workspace/parallax/preflight.json",
            "optimizer_output": "/workspace/parallax/optimizer-result.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": spec["job"]["bounds"],
        "firewall": EXPECTED_FIREWALL,
    }


def _worker_assignment(*, run_id: str, worker_id: int) -> dict[str, Any]:
    if worker_id <= 0 or not run_id:
        raise PG2OptimizerLaunchError("PG2 guardian worker assignment identity is invalid")
    return {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": run_id,
        "worker_label": f"parallax-worker-{run_id}",
        "worker_id": worker_id,
    }


def _worker_assignment_ack(*, run_id: str, worker_id: int) -> dict[str, Any]:
    assignment = _worker_assignment(run_id=run_id, worker_id=worker_id)
    return {
        **assignment,
        "status": "GUARDIAN_WORKER_ASSIGNMENT_ACKNOWLEDGED",
    }


def _launch_state_path(root: Path, spec: dict[str, Any]) -> Path:
    output = Path(str(spec.get("output_receipt", {}).get("path", "")))
    if output.is_absolute() or not output.parts or ".." in output.parts:
        raise PG2OptimizerLaunchError("PG2 optimizer launch-state path is unsafe")
    return root / output.parent / "launch-state.json"


def _record_launch_state(path: Path, state: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_id",
        "status",
        "phase",
        "guardian_instance_id",
        "worker_instance_id",
        "guardian_offer",
        "worker_offer",
        "guardian_exact_worker_ID_acknowledged",
        "job_released",
        "s3_durable",
        "failure",
        "cleanup",
        "automatic_retry",
    }
    if set(state) != expected or state.get("automatic_retry") is not False:
        raise PG2OptimizerLaunchError("PG2 optimizer launch state is malformed")
    atomic_json(
        path,
        {
            **state,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def run(root: Path, spec_path: Path, freeze_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    validate_execution_freeze(root, spec_path, freeze_path, spec)
    quarantine = require_machine_quarantine(root, spec)
    initial = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    s3_safety = require_s3_safety()
    inputs = _require_input_heads(spec)
    _require_result_absent(spec["result"]["key"])
    _claim(root, spec)
    run_id = str(spec["run_id"])
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise PG2OptimizerLaunchError("SSH or scoped lifecycle key is absent")
    guardian_id: int | None = None
    worker_id: int | None = None
    guardian_market: dict[str, Any] | None = None
    worker_market: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    terminal_body: bytes | None = None
    s3_proof: dict[str, Any] | None = None
    job_released = False
    cleanup: dict[str, Any] | None = None
    launch_state_path = _launch_state_path(root, spec)
    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
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
            _create_arguments(role="guardian", spec=spec, run_id=run_id),
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
        guardian_contract = _guardian_contract(spec, run_id=run_id, instance_id=guardian_id)
        record("STAGING_GUARDIAN_SOURCE")
        _stage(
            guardian_ssh,
            remote="/opt/parallax/ephemeral_guardian.py",
            content=(root / "infra/vast/parallax/ephemeral_guardian.py").read_bytes(),
            mode="0700",
        )
        record("STAGING_GUARDIAN_SCOPED_KEY")
        _stage(
            guardian_ssh,
            remote=guardian_contract["scoped_api_key_path"],
            content=scoped_key_path.read_bytes(),
        )
        record("STAGING_GUARDIAN_SELF_KEY")
        _stage(
            guardian_ssh,
            remote=guardian_contract["self_api_key_path"],
            content=str(guardian_create["instance_api_key"]).encode("utf-8"),
        )
        record("STAGING_GUARDIAN_CONTRACT")
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
            raise PG2OptimizerLaunchError("PG2 guardian process did not start")
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
            _create_arguments(role="worker", spec=spec, run_id=run_id),
        )
        worker_id = int(worker_create["new_contract"])
        record("H100_CREATED", worker_instance_id=worker_id)
        worker_assignment = _worker_assignment(run_id=run_id, worker_id=worker_id)
        _stage_atomic(
            guardian_ssh,
            remote=guardian_contract["worker_assignment_path"],
            content=json.dumps(worker_assignment, sort_keys=True).encode("utf-8"),
        )
        assignment_ack = _worker_assignment_ack(run_id=run_id, worker_id=worker_id)
        _wait_remote_json_exact(
            guardian_ssh,
            guardian_contract["worker_assignment_ack_path"],
            assignment_ack,
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
        model, bundle, overlays = _presigned_inputs(spec, credentials)
        job_contract = _job_contract(
            root,
            spec,
            instance_id=worker_id,
            model=model,
            bundle=bundle,
            overlays=overlays,
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
        record("STAGING_WORKER_CONTROL_FILES")
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
            remote="/opt/parallax/ephemeral_pg2_optimizer_scale_job.py",
            content=(root / "infra/vast/parallax/ephemeral_pg2_optimizer_scale_job.py").read_bytes(),
            mode="0700",
        )
        _stage(
            worker_ssh,
            remote="/run/parallax/pg2-optimizer-job-contract.json",
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
        overlays.clear()
        job_contract.clear()
        job_released = True
        record("JOB_RELEASED", job_released=True)
        started_worker = _run(
            [*worker_ssh, _worker_start_command()],
            check=False,
            timeout_seconds=30,
        )
        if started_worker.returncode != 0:
            raise PG2OptimizerLaunchError("PG2 optimizer worker process did not start")
        terminal, terminal_body, s3_proof = _wait_result(
            key=spec["result"]["key"],
            run_id=run_id,
            instance_id=worker_id,
            seconds=int(spec["result"]["wait_seconds"]),
        )
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
                s3_durable=s3_proof is not None,
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
            state["status"] = "CLEANED_AFTER_FAILURE" if state["failure"] else "CLEANED_AFTER_S3_DURABILITY"
            state["phase"] = "TERMINAL_CLEANUP_CONFIRMED"
            _record_launch_state(launch_state_path, state)
    final = _confirm_account_zero(seconds=45)
    if terminal is None or terminal_body is None or s3_proof is None:
        raise PG2OptimizerLaunchError("PG2 optimizer terminal S3 result is absent")
    job_result = terminal.get("job", {}).get("result", {}).get("payload", {})
    mechanical_green = terminal.get("status") == "GREEN_JOB_TERMINAL"
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "GREEN_PG2_OPTIMIZER_MICROGATE_EXECUTED"
            if mechanical_green
            else "RED_PG2_OPTIMIZER_MICROGATE_EXECUTION"
        ),
        "run_id": run_id,
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "scope": "PG2_SUPPORTING_TRAIN_ONLY_MECHANICAL_NONPROMOTION",
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
        "launch_state": {
            "path": launch_state_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(launch_state_path),
        },
        "terminal_account": final,
        "wall_seconds": time.monotonic() - started,
        "automatic_retry": False,
        "promotion_boundary": "PG2_SUPPORTING_MECHANICAL_SCALE_ONLY_NO_SCIENTIFIC_CLAIM_OR_PRIMARY_CHAIN_INFERENCE",
    }
    output_relative = Path(str(spec["output_receipt"].get("path", "")))
    if output_relative.is_absolute() or not output_relative.parts or ".." in output_relative.parts:
        raise PG2OptimizerLaunchError("PG2 optimizer receipt path is unsafe")
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
