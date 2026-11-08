"""Exercise the entire PG2 seed-17 range launch construction with zero mutation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .ephemeral_pilot_job import PG2_V2_RUN_ID as RUN_ID, validate_contract as validate_job_contract
from .ephemeral_pilot_launch import (
    _bundle,
    _job_contract,
    _model_rows,
    _overlay_rows,
    _require_input_heads,
    _require_result_absent,
    _signed,
    _worker_contract,
    require_machine_quarantine,
    select_cheapest_h100_offer,
    validate_spec,
)
from .ephemeral_pilot_worker import validate_contract as validate_worker_contract
from .ephemeral_s3_launch import (
    _aws_credentials,
    _confirm_account_zero,
    _create_arguments,
    require_s3_safety,
    select_cheapest_offer,
)
from .pg2_range_source_gate import validate_local as validate_range_source
from .s3_post_upload import validate_contract as validate_post_contract
from .s3_presign_post import presign_artifact_post
from .s3_presign_put import presign_result_put


class PilotNoCreateError(RuntimeError):
    pass


def _claim_absent(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(spec["one_shot_authorization"].get("claim_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PilotNoCreateError("pilot one-shot claim path is unsafe")
    if (root / relative).exists():
        raise PilotNoCreateError("pilot one-shot claim already exists")
    return {"path": str(relative), "status": "GREEN_ABSENT"}


def _sanitized_construction(arguments: list[str], *, role: str) -> dict[str, Any]:
    forbidden = {"--volume", "--volume-label", "--volume_mount", "--mount"}
    if forbidden.intersection(arguments):
        raise PilotNoCreateError(f"{role} construction contains a persistent storage argument")
    sanitized = list(arguments)
    if "--login" in sanitized:
        index = sanitized.index("--login") + 1
        if index >= len(sanitized) or not sanitized[index]:
            raise PilotNoCreateError("worker registry login construction is absent")
        sanitized[index] = "<ephemeral-redacted>"
    return {
        "status": "GREEN_EXACT_CREATE_ARGUMENTS",
        "role": role,
        "arguments": sanitized,
        "secrets_persisted": False,
        "persistent_storage_arguments": 0,
    }


def run(root: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    output = output.resolve()
    if output.exists():
        raise PilotNoCreateError("pilot no-create receipt already exists")
    spec = validate_spec(root, spec_path)
    initial = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    safety = require_s3_safety()
    selected = validate_range_source(root)
    claim = _claim_absent(root, spec)
    _require_result_absent(spec["artifact"]["key"])
    _require_result_absent(spec["result"]["key"])
    result_keys = {
        "artifact": {
            "key": spec["artifact"]["key"],
            "status": "GREEN_ZERO_VERSIONS_AND_DELETE_MARKERS",
        },
        "terminal": {
            "key": spec["result"]["key"],
            "status": "GREEN_ZERO_VERSIONS_AND_DELETE_MARKERS",
        },
    }
    quarantine = require_machine_quarantine(root, spec)
    model_rows = _model_rows(root, spec)
    overlay_rows = _overlay_rows(root, spec)
    bundle_row, bundle_receipt = _bundle(root, spec)
    input_proof = _require_input_heads(model_rows, bundle_row, overlay_rows)

    guardian_market = select_cheapest_offer(role="guardian", contract=spec["guardian"])
    worker_market = select_cheapest_h100_offer(
        spec["worker"],
        quarantined_machine_ids=set(quarantine["active_machine_ids"]),
        excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
    )
    guardian_construction = _sanitized_construction(
        _create_arguments(role="guardian", spec=spec, run_id=RUN_ID), role="guardian"
    )
    worker_construction = _sanitized_construction(
        _create_arguments(role="worker", spec=spec, run_id=RUN_ID), role="worker"
    )

    credentials = _aws_credentials()
    now = datetime.now(timezone.utc)
    model = [_signed(row, credentials, now) for row in model_rows]
    bundle = _signed(bundle_row, credentials, now)
    overlays = [_signed(row, credentials, now) for row in overlay_rows]
    job_contract = _job_contract(
        spec,
        instance_id=1,
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
        bucket="pg-lineage-769063704696",
        key=spec["result"]["key"],
        region="us-east-1",
        now=now,
        expires_seconds=int(spec["result"]["put_url_seconds"]),
    )
    worker_contract = _worker_contract(spec, instance_id=1, result_url=result_url)
    validate_job_contract(job_contract)
    validate_post_contract(artifact_post)
    validate_worker_contract(worker_contract)
    credentials.clear()
    model.clear()
    bundle.clear()
    overlays.clear()
    job_contract.clear()
    artifact_post.clear()
    worker_contract.clear()

    terminal = _confirm_account_zero(seconds=10, poll_seconds=0.2)
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_PILOT_NO_CREATE_REHEARSAL",
        "run_id": RUN_ID,
        "scope": "FULL_CONSTRUCTION_ZERO_MUTATION_NO_PAID_COMPUTE",
        "launch_spec": {
            "path": str(spec_path.relative_to(root)),
            "sha256": sha256_file(spec_path),
        },
        "range_source_gate": selected,
        "initial_account": initial,
        "terminal_account": terminal,
        "s3_safety": safety,
        "claim": claim,
        "result_keys": result_keys,
        "input_proof": input_proof,
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_construction": guardian_construction,
        "worker_construction": worker_construction,
        "contract_validation": {
            "range_job_with_18_exact_versioned_inputs": "GREEN_EXACT",
            "artifact_POST_exact_key_size_encryption_checksum": "GREEN_EXACT",
            "terminal_worker_exact_lifecycle": "GREEN_EXACT",
            "secrets_or_urls_persisted": False,
        },
        "mutation": {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "s3_objects_written": 0,
            "files_downloaded": 0,
            "gpu_operations": 0,
        },
        "automatic_retry": False,
        "paid_authorization": False,
        "next_boundary": "A_SEPARATE_EXECUTION_FREEZE_MUST_BIND_THIS_EXACT_RECEIPT_AND_GREEN_PLANTS",
        "promotion_boundary": "PG2_SUPPORTING_NO_CREATE_ONLY_NO_PAID_SCIENCE_PRIMARY_CHAIN_OR_PROMOTION_AUTHORIZATION",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = run(arguments.root, arguments.spec, arguments.output)
    print(json.dumps({"run_id": RUN_ID, "status": packet["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
