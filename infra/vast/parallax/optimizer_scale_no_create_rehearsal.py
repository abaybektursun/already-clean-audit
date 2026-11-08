"""Exercise the complete optimizer launch construction without claiming or creating.

This read-only gate traverses exact spec validation, the machine quarantine,
live all-region market selection, both Vast create-argument builders, immutable
S3 HEADs, ECR authentication construction, presigned input/result contracts,
and the actual worker/job validators.  Secrets and URLs are never persisted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ephemeral_optimizer_scale_job as optimizer_job
from . import ephemeral_optimizer_scale_launch as launch
from . import ephemeral_s3_worker as s3_worker
from .artifacts import atomic_json, sha256_file


class NoCreateRehearsalError(RuntimeError):
    pass


def _claim_absent(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(spec["one_shot_authorization"].get("claim_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise NoCreateRehearsalError("one-shot claim path is unsafe")
    path = (root / relative).resolve()
    if path.exists():
        raise NoCreateRehearsalError("candidate one-shot claim already exists")
    return {"status": "GREEN_ABSENT", "path": relative.as_posix()}


def _sanitized_create_arguments(
    arguments: list[str], *, role: str, spec: dict[str, Any]
) -> dict[str, Any]:
    if any("volume" in str(value).lower() for value in arguments):
        raise NoCreateRehearsalError("Vast volume argument escaped into S3-only launch")
    sanitized = list(arguments)
    login_present = "--login" in sanitized
    if role == "guardian":
        if login_present or sanitized[:2] != ["--image", str(spec["guardian"]["image"])]:
            raise NoCreateRehearsalError("guardian create construction changed")
    else:
        if not login_present:
            raise NoCreateRehearsalError("worker create construction lacks ephemeral ECR login")
        index = sanitized.index("--login")
        if index + 1 >= len(sanitized):
            raise NoCreateRehearsalError("worker login argument is malformed")
        login = str(sanitized[index + 1])
        expected_registry = str(spec["worker_registry_auth"]["registry"])
        if "-u AWS" not in login or "-p " not in login or expected_registry not in login:
            raise NoCreateRehearsalError("worker ECR login does not match the exact registry")
        sanitized[index + 1] = "<EPHEMERAL_ECR_LOGIN_REDACTED>"
        login = ""
        if sanitized[:4] != [
            "--template_hash",
            str(spec["worker"]["template_hash"]),
            "--image",
            str(spec["worker"]["image"]),
        ]:
            raise NoCreateRehearsalError("worker managed-template create construction changed")
    if "--onstart-cmd" not in sanitized or "--disk" not in sanitized:
        raise NoCreateRehearsalError("provider-native worker deadman or scratch request is absent")
    return {
        "status": "GREEN_EXACT_CREATE_ARGUMENTS",
        "role": role,
        "arguments": sanitized,
        "login_present": login_present,
        "secrets_persisted": False,
    }


def run(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = launch.validate_spec(root, spec_path)
    claim = _claim_absent(root, spec)
    initial = launch._require_initial_zero()
    s3_safety = launch.require_s3_safety()
    inputs = launch._require_input_heads(spec)
    launch._require_result_absent(spec["result"]["key"])
    quarantine = launch.require_machine_quarantine(root, spec)

    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise NoCreateRehearsalError("SSH or scoped lifecycle key is absent")

    guardian_market = launch.select_cheapest_offer(
        role="guardian", contract=spec["guardian"]
    )
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
    ):
        raise NoCreateRehearsalError("guardian contract construction changed")

    credentials: dict[str, Any] = {}
    model: list[dict[str, Any]] = []
    bundle: dict[str, Any] = {}
    job_contract: dict[str, Any] = {}
    worker_contract: dict[str, Any] = {}
    result_url = ""
    try:
        credentials = launch._aws_credentials()
        model, bundle = launch._presigned_inputs(spec, credentials)
        job_contract = launch._job_contract(
            root,
            spec,
            instance_id=synthetic_instance_id,
            model=model,
            bundle=bundle,
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
        job_contract.clear()
        worker_contract.clear()
        result_url = ""

    terminal = launch._account_zero()
    if terminal["active_instances"] or terminal["active_volumes"]:
        raise NoCreateRehearsalError("no-create rehearsal did not preserve a zero Vast account")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_FULL_NO_CREATE_REHEARSAL",
        "run_id": spec["run_id"],
        "scope": "optimizer_scale_exact_launch_construction_without_claim_or_create",
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
            "optimizer_job_with_exact_presigned_versioned_inputs": "GREEN_EXACT",
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
        "next_boundary": "A_SEPARATE_PAID_EXECUTION_FREEZE_MUST_BIND_THIS_EXACT_RECEIPT_AND_SPEC",
        "promotion_boundary": "NO_CREATE_REHEARSAL_ONLY_NO_PAID_OR_SCIENTIFIC_AUTHORIZATION",
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
