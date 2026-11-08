"""Traverse the exact PG2 optimizer launch path without claiming or creating."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ephemeral_pg2_optimizer_scale_job as optimizer_job
from . import ephemeral_pg2_optimizer_scale_launch as launch
from . import ephemeral_s3_worker as s3_worker
from .artifacts import atomic_json, sha256_file
from .optimizer_scale_no_create_rehearsal import (
    NoCreateRehearsalError,
    _claim_absent,
    _sanitized_create_arguments,
)


def run(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = launch.validate_spec(root, spec_path)
    claim = _claim_absent(root, spec)
    initial = launch._confirm_account_zero(seconds=10, poll_seconds=0.2)
    s3_safety = launch.require_s3_safety()
    inputs = launch._require_input_heads(spec)
    launch._require_result_absent(spec["result"]["key"])
    quarantine = launch.require_machine_quarantine(root, spec)

    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise NoCreateRehearsalError("SSH or scoped lifecycle key is absent")

    guardian_market = launch.select_cheapest_offer(role="guardian", contract=spec["guardian"])
    guardian_arguments = launch._create_arguments(
        role="guardian", spec=spec, run_id=str(spec["run_id"])
    )
    guardian_construction = _sanitized_create_arguments(
        guardian_arguments, role="guardian", spec=spec
    )
    guardian_arguments.clear()

    worker_market = launch.select_cheapest_h100_offer(
        spec["worker"],
        quarantined_machine_ids=set(quarantine["active_machine_ids"]),
        excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
    )
    worker_arguments = launch._create_arguments(
        role="worker", spec=spec, run_id=str(spec["run_id"])
    )
    worker_construction = _sanitized_create_arguments(
        worker_arguments, role="worker", spec=spec
    )
    worker_arguments.clear()

    synthetic_instance_id = 1
    guardian_contract = launch._guardian_contract(
        spec,
        run_id=str(spec["run_id"]),
        instance_id=synthetic_instance_id,
    )
    if (
        guardian_contract.get("status") != "FROZEN_GUARDIAN"
        or guardian_contract.get("hard_runtime_seconds")
        != spec["guardian"]["worker_hard_runtime_seconds"]
        or guardian_contract.get("worker_assignment_path")
        != "/run/parallax/worker-assignment.json"
        or guardian_contract.get("worker_assignment_ack_path")
        != "/opt/parallax/guardian-assignment-ack.json"
    ):
        raise NoCreateRehearsalError("PG2 guardian contract construction changed")

    credentials: dict[str, Any] = {}
    model: list[dict[str, Any]] = []
    bundle: dict[str, Any] = {}
    overlays: list[dict[str, Any]] = []
    job_contract: dict[str, Any] = {}
    worker_contract: dict[str, Any] = {}
    result_url = ""
    try:
        credentials = launch._aws_credentials()
        model, bundle, overlays = launch._presigned_inputs(spec, credentials)
        job_contract = launch._job_contract(
            root,
            spec,
            instance_id=synthetic_instance_id,
            model=model,
            bundle=bundle,
            overlays=overlays,
        )
        optimizer_job.validate_contract(job_contract)
        result_url = launch.presign_result_put(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            bucket=launch.BUCKET,
            key=spec["result"]["key"],
            region=launch.REGION,
            now=datetime.now(timezone.utc),
            expires_seconds=int(spec["result"]["put_url_seconds"]),
        )
        worker_contract = launch._worker_contract(
            spec,
            run_id=str(spec["run_id"]),
            instance_id=synthetic_instance_id,
            result_url=result_url,
        )
        s3_worker.validate_contract(worker_contract)
    finally:
        credentials.clear()
        model.clear()
        bundle.clear()
        overlays.clear()
        job_contract.clear()
        worker_contract.clear()
        result_url = ""

    terminal = launch._confirm_account_zero(seconds=10, poll_seconds=0.2)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_FULL_PG2_NO_CREATE_REHEARSAL",
        "run_id": spec["run_id"],
        "scope": "pg2_supporting_optimizer_exact_launch_construction_without_claim_or_create",
        "launch_spec": {
            "path": spec_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(spec_path),
        },
        "initial_account": initial,
        "terminal_account": terminal,
        "s3_safety": s3_safety,
        "input_proof": inputs,
        "claim": claim,
        "result_key": {
            "key": spec["result"]["key"],
            "status": "GREEN_ZERO_VERSIONS_AND_DELETE_MARKERS",
        },
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_construction": guardian_construction,
        "worker_construction": worker_construction,
        "contract_validation": {
            "guardian": "GREEN_EXACT",
            "PG2_optimizer_job_with_16_exact_presigned_versioned_inputs": "GREEN_EXACT",
            "terminal_worker_with_exact_presigned_S3_result": "GREEN_EXACT",
            "secrets_or_urls_persisted": False,
        },
        "mutation": {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "S3_objects_written": 0,
            "model_or_data_downloaded": False,
            "GPU_work": False,
        },
        "automatic_retry": False,
        "next_boundary": "A_SEPARATE_PAID_PG2_EXECUTION_FREEZE_MUST_BIND_THIS_EXACT_RECEIPT_AND_SPEC",
        "promotion_boundary": "PG2_NO_CREATE_REHEARSAL_ONLY_NO_PAID_SCIENTIFIC_OR_PRIMARY_CHAIN_AUTHORIZATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root.resolve(), args.spec.resolve())
    atomic_json(args.output.resolve(), packet)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "run_id": packet["run_id"],
                "h100": packet["worker_market"]["selected"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
