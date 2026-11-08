"""Prove the post-incident worker-assignment and cleanup path on two cheap Vast instances.

This gate creates no H100, reads no model or task byte, and uses no Vast volume.  A cheap
guardian acknowledges the exact cheap-worker instance ID, then the controller exercises the
same unconditional pre-release cleanup used by science launches and requires repeated absence
plus whole-account-zero observations.  The small terminal proof is copied to versioned S3.
There is no retry path.
"""

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
from .ephemeral_guardian import validate_contract as validate_guardian_contract
from .ephemeral_optimizer_scale_launch import (
    _guardian_contract,
    _worker_assignment,
    _worker_assignment_ack,
)
from .ephemeral_s3_launch import (
    AWS_PROFILE,
    BUCKET,
    REGION,
    EphemeralLaunchError,
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
    require_s3_safety,
    select_cheapest_offer,
)
from .guarded_launch import _create_instance, _run, _ssh_base, wait_ssh


RUN_ID = "parallax-lifecycle-assignment-rehearsal-v1"
STOCK_IMAGE = (
    "vastai/base-image@"
    "sha256:a264d0c242bebbdaa6f50b34fb81c7e06cb21d8487a202f27cce22794a2e68e4"
)


class AssignmentRehearsalError(EphemeralLaunchError):
    pass


def _resolve_bound(root: Path, row: dict[str, Any], *, label: str) -> Path:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise AssignmentRehearsalError(f"unsafe {label} path")
    path = (root / relative).resolve()
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise AssignmentRehearsalError(f"{label} hash mismatch")
    return path


def _role_contract(role: str) -> dict[str, Any]:
    common = {
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "image": STOCK_IMAGE,
        "disk_gb": 10,
        "minimum_reliability": 0.98,
        "minimum_inet_down_mbps": 100,
        "minimum_cuda": 0,
        "price_ceiling_dph": 0.12,
        "bootstrap_seconds": 180,
        "hard_destroy_seconds": 300,
    }
    if role == "worker":
        return {
            **common,
            "gpu_kind": "non_h100_mechanical_worker",
            "launch_mode": "stock_direct_ssh",
        }
    if role == "guardian":
        return {
            **common,
            "gpu_kind": "non_h100_mechanical_guard",
            "discovery_seconds": 90,
            "provider_loading_seconds": 60,
            "worker_hard_runtime_seconds": 120,
        }
    raise ValueError(role)


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
        "guardian",
        "worker",
        "result",
        "one_shot_authorization",
        "output",
        "bound_artifacts",
        "implementation",
        "invariants",
        "promotion_boundary",
    }
    if set(spec) != expected_top or spec.get("schema_version") != 1:
        raise AssignmentRehearsalError("assignment rehearsal schema changed")
    if (
        spec.get("program") != "parallax"
        or spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_CHEAP_LIVE_REHEARSAL"
        or spec.get("scope") != "post_incident_assignment_and_cleanup_only"
        or spec.get("run_id") != RUN_ID
        or spec.get("provider") != "vast.ai"
        or spec.get("automatic_paid_retry") is not False
    ):
        raise AssignmentRehearsalError("assignment rehearsal escapes its frozen scope")
    if spec.get("authorization") != {
        "paid_rehearsal": True,
        "vast_instances": "TWO_CHEAP_NON_H100_ONLY",
        "h100": False,
        "model_or_data_reads": False,
        "training_or_inference": False,
        "science_claim": False,
    }:
        raise AssignmentRehearsalError("assignment rehearsal authorization changed")
    if spec.get("storage") != {
        "durable_provider": "amazon_s3",
        "aws_profile": AWS_PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "maximum_ephemeral_scratch_gb_per_instance": 10,
    }:
        raise AssignmentRehearsalError("assignment rehearsal storage is not exact AWS-only")
    if spec.get("guardian") != _role_contract("guardian"):
        raise AssignmentRehearsalError("cheap guardian contract changed")
    if spec.get("worker") != _role_contract("worker"):
        raise AssignmentRehearsalError("cheap worker contract changed")
    if spec.get("result") != {
        "bucket": BUCKET,
        "region": REGION,
        "key": f"parallax/control-plane/{RUN_ID}/terminal.json",
        "maximum_bytes": 262_144,
        "required_version_id": True,
        "required_encryption": "AES256",
    }:
        raise AssignmentRehearsalError("assignment rehearsal S3 result changed")
    if spec.get("one_shot_authorization") != {
        "id": RUN_ID,
        "claim_path": (
            "studies/lineage/cross-lineage/results/readiness/claims/"
            "parallax_lifecycle_assignment_rehearsal_v1.json"
        ),
    }:
        raise AssignmentRehearsalError("assignment rehearsal one-shot identity changed")
    if spec.get("output") != {
        "terminal_path": (
            "studies/lineage/cross-lineage/results/readiness/"
            "parallax_lifecycle_assignment_rehearsal_v1/terminal.json"
        ),
        "receipt_path": (
            "studies/lineage/cross-lineage/results/readiness/"
            "parallax_lifecycle_assignment_rehearsal_v1/receipt.json"
        ),
    }:
        raise AssignmentRehearsalError("assignment rehearsal output paths changed")
    expected_bindings = {"incident", "post_incident_freeze", "hardening_tests"}
    if set(spec.get("bound_artifacts", {})) != expected_bindings:
        raise AssignmentRehearsalError("assignment rehearsal evidence set changed")
    for label, row in spec["bound_artifacts"].items():
        _resolve_bound(root, row, label=label)
    expected_implementation = {
        "launcher": "infra/vast/parallax/lifecycle_assignment_rehearsal.py",
        "guardian": "infra/vast/parallax/ephemeral_guardian.py",
        "transport": "infra/vast/parallax/ephemeral_s3_launch.py",
        "optimizer_lifecycle": "infra/vast/parallax/ephemeral_optimizer_scale_launch.py",
        "vast_launcher": "infra/vast/parallax/guarded_launch.py",
        "artifacts": "infra/vast/parallax/artifacts.py",
    }
    if set(spec.get("implementation", {})) != set(expected_implementation):
        raise AssignmentRehearsalError("assignment rehearsal implementation set changed")
    for label, relative in expected_implementation.items():
        row = spec["implementation"][label]
        if row.get("path") != relative or sha256_file(root / relative) != row.get("sha256"):
            raise AssignmentRehearsalError(f"assignment rehearsal {label} hash mismatch")
    if spec.get("invariants") != [
        "guardian_live_before_worker_create",
        "exact_worker_ID_assignment_published_atomically",
        "guardian_process_acknowledges_exact_worker_ID_before_controller_proceeds",
        "worker_and_guardian_use_different_machines",
        "both_instances_are_non_H100",
        "pre_release_destroy_is_unconditional_and_reissued_until_confirmed_absent",
        "three_consecutive_targeted_absences_and_three_account_zero_observations",
        "provider_native_300_second_destroy_deadman_on_both_instances",
        "no_Vast_volume_gateway_model_data_training_or_inference",
        "small_terminal_is_exact_versioned_S3_then_committed_to_Git",
        "no_automatic_paid_retry",
    ]:
        raise AssignmentRehearsalError("assignment rehearsal invariants changed")
    if spec.get("promotion_boundary") != (
        "LIFECYCLE_MECHANICAL_PROOF_ONLY_NO_H100_SCIENCE_OR_FRESH_PG2_AUTHORIZATION"
    ):
        raise AssignmentRehearsalError("assignment rehearsal promotion boundary changed")
    return spec


def _result_absent(key: str) -> None:
    payload = _aws_json(
        "s3api",
        "list-object-versions",
        "--bucket",
        BUCKET,
        "--prefix",
        key,
        "--max-items",
        "10",
    )
    versions = [row for row in payload.get("Versions", []) if row.get("Key") == key]
    markers = [row for row in payload.get("DeleteMarkers", []) if row.get("Key") == key]
    if versions or markers:
        raise AssignmentRehearsalError("assignment rehearsal S3 key is not fresh")


def _claim_absent(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = root / spec["one_shot_authorization"]["claim_path"]
    if path.exists():
        raise AssignmentRehearsalError("assignment rehearsal one-shot claim is consumed")
    return {"status": "GREEN_CLAIM_ABSENT", "path": str(path.relative_to(root))}


def _construction(spec: dict[str, Any], *, role: str) -> dict[str, Any]:
    arguments = _create_arguments(role=role, spec=spec, run_id=RUN_ID)
    rendered = " ".join(arguments)
    if (
        "--volume" in rendered
        or "--link-volume" in rendered
        or "--create-volume" in rendered
        or "--disk 10" not in rendered
        or "--ssh" not in arguments
        or "--direct" not in arguments
        or STOCK_IMAGE not in arguments
    ):
        raise AssignmentRehearsalError(f"{role} construction escaped cheap ephemeral stock mode")
    return {"role": role, "arguments": arguments, "persistent_storage": False}


def no_create(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    initial = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    s3_safety = require_s3_safety()
    claim = _claim_absent(root, spec)
    _result_absent(spec["result"]["key"])
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key.is_file():
        raise AssignmentRehearsalError("SSH or scoped lifecycle key is absent")
    guardian_market = select_cheapest_offer(role="guardian", contract=spec["guardian"])
    worker_market = select_cheapest_offer(
        role="worker",
        contract=spec["worker"],
        excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
    )
    guardian_contract = _guardian_contract(spec, run_id=RUN_ID, instance_id=1)
    validate_guardian_contract(guardian_contract)
    assignment = _worker_assignment(run_id=RUN_ID, worker_id=2)
    acknowledgement = _worker_assignment_ack(run_id=RUN_ID, worker_id=2)
    terminal = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_LIFECYCLE_ASSIGNMENT_NO_CREATE",
        "run_id": RUN_ID,
        "scope": "exact_cheap_live_construction_without_claim_or_create",
        "launch_spec": {
            "path": str(spec_path.resolve().relative_to(root.resolve())),
            "sha256": sha256_file(spec_path),
        },
        "initial_account": initial,
        "terminal_account": terminal,
        "s3_safety": s3_safety,
        "claim": claim,
        "result_key": {"key": spec["result"]["key"], "versions": 0, "delete_markers": 0},
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_construction": _construction(spec, role="guardian"),
        "worker_construction": _construction(spec, role="worker"),
        "guardian_contract": guardian_contract,
        "synthetic_assignment": assignment,
        "expected_acknowledgement": acknowledgement,
        "mutation": {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "S3_objects_written": 0,
            "model_or_data_downloaded": False,
            "GPU_work": False,
        },
        "automatic_retry": False,
        "next_boundary": "SEPARATE_ONE_SHOT_CHEAP_LIVE_EXECUTION_FREEZE_REQUIRED",
    }


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
    if (
        set(freeze) != expected_top
        or freeze.get("schema_version") != 1
        or freeze.get("program") != "parallax"
        or freeze.get("status") != "FROZEN_ONE_SHOT_CHEAP_LIFECYCLE_REHEARSAL"
        or freeze.get("scope") != RUN_ID
        or freeze.get("run_id") != RUN_ID
        or freeze.get("automatic_paid_retry") is not False
        or freeze.get("authorization")
        != {
            "create_vast_instances": "ONE_CHEAP_GUARDIAN_THEN_ONE_CHEAP_NON_H100_WORKER",
            "create_vast_volumes": False,
            "h100": False,
            "model_or_data_reads": False,
            "training_or_inference": False,
            "maximum_quoted_dph_each": 0.12,
            "maximum_provider_deadman_seconds_each": 300,
            "maximum_rate_times_deadman_envelope_usd": 0.02,
        }
    ):
        raise AssignmentRehearsalError("cheap live execution freeze is absent or inexact")
    bindings = freeze.get("bound_artifacts", {})
    if set(bindings) != {"launch_spec", "no_create", "hardening_tests", "incident"}:
        raise AssignmentRehearsalError("cheap live freeze evidence set changed")
    if _resolve_bound(root, bindings["launch_spec"], label="freeze launch spec") != spec_path:
        raise AssignmentRehearsalError("cheap live freeze binds another launch spec")
    no_create_path = _resolve_bound(root, bindings["no_create"], label="freeze no-create")
    no_create_receipt = json.loads(no_create_path.read_text(encoding="utf-8"))
    expected_zero = {
        "active_instances": 0,
        "active_volumes": 0,
        "instance_ids": [],
        "volume_ids": [],
        "status": "GREEN_ACCOUNT_ZERO_CONFIRMED",
        "consecutive_observations": 3,
    }
    if (
        no_create_receipt.get("status") != "GREEN_LIFECYCLE_ASSIGNMENT_NO_CREATE"
        or no_create_receipt.get("run_id") != RUN_ID
        or no_create_receipt.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
        or no_create_receipt.get("initial_account") != expected_zero
        or no_create_receipt.get("terminal_account") != expected_zero
        or no_create_receipt.get("result_key")
        != {"key": spec["result"]["key"], "versions": 0, "delete_markers": 0}
        or no_create_receipt.get("guardian_market", {}).get("selected", {}).get("gpu_name", "").upper().find("H100")
        != -1
        or no_create_receipt.get("worker_market", {}).get("selected", {}).get("gpu_name", "").upper().find("H100")
        != -1
        or no_create_receipt.get("guardian_market", {}).get("selected", {}).get("machine_id")
        == no_create_receipt.get("worker_market", {}).get("selected", {}).get("machine_id")
        or no_create_receipt.get("guardian_construction", {}).get("persistent_storage") is not False
        or no_create_receipt.get("worker_construction", {}).get("persistent_storage") is not False
        or no_create_receipt.get("mutation")
        != {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "S3_objects_written": 0,
            "model_or_data_downloaded": False,
            "GPU_work": False,
        }
        or no_create_receipt.get("automatic_retry") is not False
    ):
        raise AssignmentRehearsalError("cheap live no-create receipt is not exact and green")
    tests_path = _resolve_bound(root, bindings["hardening_tests"], label="freeze tests")
    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    if (
        tests.get("status") != "GREEN_LIFECYCLE_HARDENING_TESTS"
        or tests.get("tests_failed") != 0
        or int(tests.get("tests_passed", 0)) < 318
        or tests.get("paid_action") is not False
    ):
        raise AssignmentRehearsalError("cheap live freeze tests are not exact and green")
    _resolve_bound(root, bindings["incident"], label="freeze incident")
    if freeze.get("one_shot") != {
        "run_id": RUN_ID,
        "claim_path": spec["one_shot_authorization"]["claim_path"],
        "terminal_s3_key": spec["result"]["key"],
        "claim_must_be_absent": True,
        "terminal_key_must_be_absent": True,
        "create_failure_consumes_authorization": True,
    }:
        raise AssignmentRehearsalError("cheap live freeze one-shot identity changed")
    if freeze.get("stop_on") != [
        "ANY_HASH_OR_SCHEMA_DRIFT",
        "ANY_NONZERO_VAST_ACCOUNT",
        "ANY_EXISTING_CLAIM_OR_TERMINAL_S3_VERSION",
        "NO_TWO_ELIGIBLE_NON_H100_OFFERS_AT_OR_BELOW_PRICE_CEILING_ON_DIFFERENT_MACHINES",
        "GUARDIAN_NOT_ARMED_LIVE_OR_EXACTLY_ACKNOWLEDGED",
        "ANY_DESTROY_STOP_OR_ACCOUNT_ZERO_CONFIRMATION_FAILURE",
        "ANY_S3_VERSION_CHECKSUM_ENCRYPTION_OR_READBACK_FAILURE",
    ]:
        raise AssignmentRehearsalError("cheap live freeze stop policy changed")
    if freeze.get("next_authorized_action") != (
        "CREATE_ONE_CHEAP_NON_H100_GUARDIAN_THEN_ONE_CHEAP_NON_H100_WORKER_ONLY"
    ):
        raise AssignmentRehearsalError("cheap live freeze next action changed")
    if freeze.get("promotion_boundary") != spec["promotion_boundary"]:
        raise AssignmentRehearsalError("cheap live freeze promotion boundary changed")
    return freeze


def _publish_terminal(root: Path, spec: dict[str, Any], terminal: dict[str, Any]) -> dict[str, Any]:
    relative = Path(spec["output"]["terminal_path"])
    path = (root / relative).resolve()
    atomic_json(path, terminal)
    digest = sha256_file(path)
    body = path.read_bytes()
    if len(body) > int(spec["result"]["maximum_bytes"]):
        raise AssignmentRehearsalError("cheap live terminal exceeds frozen byte bound")
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    put = _aws_json(
        "s3api",
        "put-object",
        "--bucket",
        BUCKET,
        "--key",
        spec["result"]["key"],
        "--body",
        str(path),
        "--server-side-encryption",
        "AES256",
        "--checksum-sha256",
        checksum,
        "--metadata",
        f"sha256={digest}",
    )
    version_id = str(put.get("VersionId", ""))
    if not version_id:
        raise AssignmentRehearsalError("cheap live terminal PUT returned no version ID")
    head = _aws_json(
        "s3api",
        "head-object",
        "--bucket",
        BUCKET,
        "--key",
        spec["result"]["key"],
        "--version-id",
        version_id,
        "--checksum-mode",
        "ENABLED",
    )
    with tempfile.TemporaryDirectory(prefix="parallax-lifecycle-small-") as temporary:
        readback = Path(temporary) / "terminal.json"
        get = _aws_json(
            "s3api",
            "get-object",
            "--bucket",
            BUCKET,
            "--key",
            spec["result"]["key"],
            "--version-id",
            version_id,
            "--checksum-mode",
            "ENABLED",
            str(readback),
        )
        readback_digest = sha256_file(readback)
    if (
        head.get("VersionId") != version_id
        or head.get("ContentLength") != len(body)
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("ChecksumSHA256") != checksum
        or head.get("Metadata", {}).get("sha256") != digest
        or get.get("VersionId") != version_id
        or readback_digest != digest
    ):
        raise AssignmentRehearsalError("cheap live terminal exact S3 readback failed")
    return {
        "status": "GREEN_EXACT_VERSIONED_S3_READBACK",
        "bucket": BUCKET,
        "region": REGION,
        "key": spec["result"]["key"],
        "version_id": version_id,
        "bytes": len(body),
        "sha256": digest,
        "checksum_sha256_base64": checksum,
        "server_side_encryption": "AES256",
    }


def execute(root: Path, spec_path: Path, freeze_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    validate_execution_freeze(root, spec_path, freeze_path, spec)
    initial = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    s3_safety = require_s3_safety()
    _claim_absent(root, spec)
    _result_absent(spec["result"]["key"])
    _claim(root, spec)
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise AssignmentRehearsalError("SSH or scoped lifecycle key is absent")
    guardian_id: int | None = None
    worker_id: int | None = None
    guardian_market: dict[str, Any] | None = None
    worker_market: dict[str, Any] | None = None
    acknowledgement: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        guardian_market = select_cheapest_offer(role="guardian", contract=spec["guardian"])
        guardian_create = _create_instance(
            int(guardian_market["selected"]["id"]),
            _create_arguments(role="guardian", spec=spec, run_id=RUN_ID),
        )
        guardian_id = int(guardian_create["new_contract"])
        guardian_row = wait_ssh(
            guardian_id,
            key_path,
            timeout_seconds=int(spec["guardian"]["bootstrap_seconds"]),
        )
        guardian_ssh = _ssh_base(guardian_row, key_path)
        guardian_contract = _guardian_contract(spec, run_id=RUN_ID, instance_id=guardian_id)
        validate_guardian_contract(guardian_contract)
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
            raise AssignmentRehearsalError("cheap live guardian process did not start")
        _wait_remote_file(guardian_ssh, guardian_contract["armed_path"])
        _require_guardian_live(guardian_id)
        worker_market = select_cheapest_offer(
            role="worker",
            contract=spec["worker"],
            excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
        )
        worker_create = _create_instance(
            int(worker_market["selected"]["id"]),
            _create_arguments(role="worker", spec=spec, run_id=RUN_ID),
        )
        worker_id = int(worker_create["new_contract"])
        assignment = _worker_assignment(run_id=RUN_ID, worker_id=worker_id)
        _stage_atomic(
            guardian_ssh,
            remote=guardian_contract["worker_assignment_path"],
            content=json.dumps(assignment, sort_keys=True).encode("utf-8"),
        )
        acknowledgement = _wait_remote_json_exact(
            guardian_ssh,
            guardian_contract["worker_assignment_ack_path"],
            _worker_assignment_ack(run_id=RUN_ID, worker_id=worker_id),
        )
        _require_guardian_live(guardian_id)
    finally:
        cleanup = _finalize_lifecycle(
            worker_id=worker_id,
            guardian_id=guardian_id,
            job_released=False,
            s3_durable=False,
        )
    final = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    terminal = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_CHEAP_LIVE_ASSIGNMENT_AND_CLEANUP_PROOF",
        "run_id": RUN_ID,
        "scope": "PRE_SSH_LIFECYCLE_ONLY_NO_SCIENCE",
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "storage": spec["storage"],
        "initial_account": initial,
        "s3_safety": s3_safety,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_instance_id": guardian_id,
        "worker_instance_id": worker_id,
        "guardian_acknowledgement": acknowledgement,
        "worker_SSH_reached": False,
        "model_or_data_reads": 0,
        "training_or_inference": False,
        "cleanup": cleanup,
        "terminal_account": final,
        "wall_seconds": time.monotonic() - started,
        "maximum_rate_times_deadman_envelope_usd": 0.02,
        "automatic_retry": False,
        "promotion_boundary": spec["promotion_boundary"],
    }
    s3_proof = _publish_terminal(root, spec, terminal)
    receipt = {
        **terminal,
        "status": "GREEN_CHEAP_LIVE_ASSIGNMENT_CLEANUP_AND_S3_PROOF",
        "s3_proof": s3_proof,
    }
    receipt_path = (root / spec["output"]["receipt_path"]).resolve()
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--mode", choices=("no-create", "execute"), required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "no-create":
        if args.output is None:
            raise AssignmentRehearsalError("no-create mode requires --output")
        packet = no_create(root, args.spec.resolve())
        atomic_json(args.output.resolve(), packet)
    else:
        if args.freeze is None:
            raise AssignmentRehearsalError("execute mode requires --freeze")
        packet = execute(root, args.spec.resolve(), args.freeze.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
