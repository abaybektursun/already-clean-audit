"""Rehearse the complete primary PG1 H100 construction with zero mutation."""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from . import primary_interface_control_launch as launch
from .interface_control_s3_post_upload import validate_contract as validate_post_contract
from .interface_control_s3_presign_post import presign_artifact_post
from .pilot_no_create_rehearsal import _sanitized_construction
from .primary_interface_control_job import (
    RUN_ID,
    SOURCE_CONTRACT_PATH,
    legacy_compatible_source_contract,
    validate_contract as validate_job_contract,
)
from .primary_interface_control_source_gate import validate_local as validate_primary_source
from .primary_interface_control_worker import validate_contract as validate_worker_contract
from .s3_presign_put import presign_result_put


class PrimaryInterfaceNoCreateError(RuntimeError):
    pass


def _claim_absent(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(spec["one_shot_authorization"].get("claim_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PrimaryInterfaceNoCreateError("primary interface one-shot claim path is unsafe")
    if (root / relative).exists():
        raise PrimaryInterfaceNoCreateError("primary interface one-shot claim already exists")
    return {"path": str(relative), "status": "GREEN_ABSENT"}


def run(root: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    output = output.resolve()
    if output.exists():
        raise PrimaryInterfaceNoCreateError("primary no-create receipt already exists")
    spec = launch.validate_spec(root, spec_path)
    local_outputs = launch.require_local_outputs_absent(root, spec)
    initial = launch.legacy._confirm_account_zero(seconds=10, poll_seconds=0.2)
    safety = launch.legacy.require_s3_safety()
    source = validate_primary_source(root)
    source_contract = json.loads((root / SOURCE_CONTRACT_PATH).read_text(encoding="utf-8"))
    compatible_source_contract = legacy_compatible_source_contract(source_contract)
    claim = _claim_absent(root, spec)
    launch.legacy._require_result_absent(spec["artifact"]["key"])
    launch.legacy._require_result_absent(spec["result"]["key"])
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
    quarantine = launch.legacy.require_machine_quarantine(root, spec)
    model_rows = launch._model_rows(root, spec)
    overlay_rows = launch._overlay_rows(root, spec)
    bundle_row, bundle_receipt = launch.legacy._bundle(root, spec)
    input_proof = launch.legacy._require_input_heads(model_rows, bundle_row, overlay_rows)
    guardian_market = launch.legacy.select_cheapest_offer(
        role="guardian", contract=spec["guardian"]
    )
    worker_market = launch.legacy.select_cheapest_h100_offer(
        spec["worker"],
        quarantined_machine_ids=set(quarantine["active_machine_ids"]),
        excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
    )
    guardian_arguments = launch.legacy._create_arguments(
        role="guardian", spec=spec, run_id=RUN_ID
    )
    worker_arguments = launch.legacy._create_arguments(role="worker", spec=spec, run_id=RUN_ID)
    if "--volume" in guardian_arguments or "--volume" in worker_arguments:
        raise PrimaryInterfaceNoCreateError("Vast persistent storage entered construction")
    guardian_construction = _sanitized_construction(guardian_arguments, role="guardian")
    worker_construction = _sanitized_construction(worker_arguments, role="worker")
    credentials = launch.legacy._aws_credentials()
    now = datetime.now(timezone.utc)
    model = [launch.legacy._signed(row, credentials, now) for row in model_rows]
    bundle = launch.legacy._signed(bundle_row, credentials, now)
    overlays = [launch.legacy._signed(row, credentials, now) for row in overlay_rows]
    job_contract = launch._job_contract(
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
        bucket=launch.legacy.BUCKET,
        key=spec["result"]["key"],
        region=launch.legacy.REGION,
        now=now,
        expires_seconds=int(spec["result"]["put_url_seconds"]),
    )
    worker_contract = launch._worker_contract(spec, instance_id=1, result_url=result_url)
    validate_job_contract(job_contract)
    validate_post_contract(artifact_post)
    validate_worker_contract(worker_contract)
    transport_source = inspect.getsource(launch.legacy.run)
    adapter_source = inspect.getsource(launch._patched_legacy)
    if (
        transport_source.count("_create_instance(") != 2
        or transport_source.index("_wait_remote_json_exact(")
        > transport_source.index("wait_h100_ssh(")
        or transport_source.index("_wait_interface_result(")
        > transport_source.index("_finalize_lifecycle(")
        or "/opt/parallax/primary_interface_control_job.py" not in adapter_source
        or "/opt/parallax/primary_interface_control_worker.py" not in adapter_source
    ):
        raise PrimaryInterfaceNoCreateError("primary durability or exact-ID ordering changed")
    credentials.clear()
    model.clear()
    bundle.clear()
    overlays.clear()
    job_contract.clear()
    artifact_post.clear()
    worker_contract.clear()
    terminal = launch.legacy._confirm_account_zero(seconds=10, poll_seconds=0.2)
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": launch.NO_CREATE_STATUS,
        "run_id": RUN_ID,
        "scope": "FULL_PRIMARY_PG1_INTERFACE_CONSTRUCTION_ZERO_MUTATION_NO_PAID_COMPUTE",
        "launch_spec": {
            "path": str(spec_path.relative_to(root)),
            "sha256": sha256_file(spec_path),
        },
        "source_gate": source,
        "initial_account": initial,
        "terminal_account": terminal,
        "s3_safety": safety,
        "claim": claim,
        "local_outputs": local_outputs,
        "result_keys": result_keys,
        "input_proof": input_proof,
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "guardian_construction": guardian_construction,
        "worker_construction": worker_construction,
        "contract_validation": {
            "primary_config_to_legacy_interface_config_alias": (
                "GREEN_UNIQUE_EXACT_IN_MEMORY_ONLY"
                if compatible_source_contract["artifacts"]["interface_config"]
                == compatible_source_contract["artifacts"]["primary_config"]
                else "RED"
            ),
            "external_processor_gate": "GREEN_EXACT_REPLAY_NO_H100_PROCESSOR_CALLS",
            "versioned_inputs": len(model_rows) + 1 + len(overlay_rows),
            "model_objects": len(model_rows),
            "overlay_objects": len(overlay_rows),
            "bundle_objects": 1,
            "artifact_POST_exact_key_size_encryption_checksum": "GREEN_EXACT",
            "phase_aware_terminal_worker": "GREEN_EXACT",
            "primary_entrypoints_staged": "GREEN_EXACT",
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
        "next_boundary": "SEPARATE_EXECUTION_FREEZE_MUST_BIND_THIS_RECEIPT_AND_GREEN_PLANTS",
        "promotion_boundary": launch.PRIMARY_PROMOTION_BOUNDARY,
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root, args.spec, args.output)
    print(json.dumps({"run_id": RUN_ID, "status": packet["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
