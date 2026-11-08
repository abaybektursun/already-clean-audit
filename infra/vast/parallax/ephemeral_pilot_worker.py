"""Supervise one pilot, publish its terminal, then destroy or stop safely."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Standalone siblings on the Vast worker.
    import ephemeral_s3_worker as base
except ModuleNotFoundError:  # Repository package in planted tests.
    from . import ephemeral_s3_worker as base


RUN_ID = "parallax-pg1-selected-scale-pilot-s3-v1"
PG2_RUN_ID = "parallax-pg2-seed17-range-s3-v1"
PG2_V2_RUN_ID = "parallax-pg2-seed17-range-s3-v2"
EXPECTED_COMMAND = [
    "python3",
    "/opt/parallax/ephemeral_pilot_job.py",
    "--input-contract",
    "/run/parallax/pilot-input-contract.json",
    "--artifact-post",
    "/run/parallax/pilot-artifact-post.json",
]
RUN_PROFILES = {
    RUN_ID: {"lineage": "pg1_pi", "command": EXPECTED_COMMAND},
    PG2_RUN_ID: {"lineage": "pg2", "command": EXPECTED_COMMAND},
    PG2_V2_RUN_ID: {"lineage": "pg2", "command": EXPECTED_COMMAND},
}


class PilotWorkerError(RuntimeError):
    pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    run_id = str(contract.get("run_id", ""))
    profile = RUN_PROFILES.get(run_id)
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "FROZEN_SELECTED_PILOT_WORKER"
        or profile is None
        or contract.get("stage") != "training_artifact_rehearsal"
    ):
        raise PilotWorkerError("pilot worker contract is not exact frozen schema v1")
    instance_id = contract.get("instance_id")
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise PilotWorkerError("pilot worker instance id is absent")
    root = Path(str(contract.get("runtime_root", "")))
    if root != Path("/workspace/parallax"):
        raise PilotWorkerError("pilot worker runtime root changed")
    paths = {
        name: Path(str(contract.get(name, "")))
        for name in (
            "job_output_path",
            "job_log_path",
            "terminal_path",
            "instance_api_key_path",
        )
    }
    if any(not path.is_absolute() for path in paths.values()):
        raise PilotWorkerError("pilot worker paths must be absolute")
    if any(
        not _within(path, root)
        for name, path in paths.items()
        if name != "instance_api_key_path"
    ):
        raise PilotWorkerError("pilot job artifact path escapes disposable scratch")
    if paths["instance_api_key_path"] != Path("/run/parallax/instance-api-key"):
        raise PilotWorkerError("pilot lifecycle key path changed")
    if contract.get("command") != profile["command"]:
        raise PilotWorkerError("pilot worker command changed")
    if contract.get("maximum_job_seconds") != 900:
        raise PilotWorkerError("pilot worker timeout changed")
    if contract.get("maximum_result_bytes") != 1024 * 1024:
        raise PilotWorkerError("pilot terminal payload bound changed")
    result_key = str(contract.get("result_key", ""))
    if result_key != f"parallax/results/{run_id}/terminal.json":
        raise PilotWorkerError("pilot terminal key changed")
    base.validate_result_url(str(contract.get("result_put_url", "")), result_key=result_key)
    if (
        contract.get("archive_durable_action") != "destroy"
        or contract.get("archive_not_durable_action") != "stop"
        or contract.get("terminal_upload_failure_action") != "stop"
        or contract.get("automatic_retry") is not False
    ):
        raise PilotWorkerError("pilot worker lifecycle policy changed")
    return contract


def _archive_proof(result: dict[str, Any], *, run_id: str) -> dict[str, Any] | None:
    if result.get("present") is not True or not isinstance(result.get("payload"), dict):
        return None
    payload = result["payload"]
    proof = payload.get("artifact_proof")
    if (
        payload.get("status") != "GREEN_SELECTED_PILOT_ARCHIVE_DURABLE"
        or payload.get("run_id") != run_id
        or not isinstance(proof, dict)
        or proof.get("key") != f"parallax/results/{run_id}/artifacts.tar"
        or proof.get("server_side_encryption") != "AES256"
        or proof.get("http_status") != 201
        or not isinstance(proof.get("version_id"), str)
        or not proof.get("version_id")
        or not isinstance(proof.get("bytes"), int)
        or not 100_000_000 <= int(proof["bytes"]) <= 1_100_000_000
        or not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("sha256", "")))
        or not isinstance(proof.get("checksum_sha256_base64"), str)
        or not proof.get("checksum_sha256_base64")
    ):
        return None
    return proof


def run(contract_path: Path) -> dict[str, Any]:
    contract = validate_contract(json.loads(contract_path.read_text(encoding="utf-8")))
    run_id = str(contract["run_id"])
    output = Path(contract["job_output_path"])
    log = Path(contract["job_log_path"])
    terminal = Path(contract["terminal_path"])
    if output.exists() or terminal.exists():
        raise PilotWorkerError("refusing to overwrite a prior pilot or terminal result")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    timed_out = False
    with log.open("wb") as handle:
        try:
            completed = subprocess.run(
                contract["command"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=int(contract["maximum_job_seconds"]),
                check=False,
                env={**os.environ, **contract.get("environment", {})},
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
    result_error = None
    try:
        result = base._load_job_result(output, int(contract["maximum_result_bytes"]))
    except (OSError, ValueError, json.JSONDecodeError, base.WorkerError) as error:
        result = {"present": output.is_file()}
        result_error = f"{type(error).__name__}: {error}"
    archive_proof = _archive_proof(result, run_id=run_id)
    archive_durable = archive_proof is not None
    green = returncode == 0 and archive_durable and result_error is None
    lifecycle_action = "destroy" if archive_durable else "stop"
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "provider": "vast.ai",
        "instance_id": contract["instance_id"],
        "stage": contract["stage"],
        "status": "GREEN_JOB_TERMINAL" if green else "RED_JOB_TERMINAL",
        "started_utc": started_utc,
        "wall_seconds": time.monotonic() - started,
        "job": {
            "returncode": returncode,
            "timed_out": timed_out,
            "log_tail": base._tail(log),
            "result": result,
            "result_error": result_error,
        },
        "durability_contract": {
            "durable_provider": "amazon_s3",
            "bucket": "pg-lineage-769063704696",
            "region": "us-east-1",
            "archive_key": f"parallax/results/{run_id}/artifacts.tar",
            "terminal_key": contract["result_key"],
            "archive_durable": archive_durable,
            "archive_proof": archive_proof,
            "post_terminal_action": lifecycle_action,
            "terminal_upload_failure_action": "stop",
            "vast_persistent_storage": "FORBIDDEN",
        },
    }
    base.atomic_json(terminal, packet)
    try:
        base.put_terminal(url=contract["result_put_url"], terminal_path=terminal)
    except BaseException:
        base.instance_action(
            instance_id=int(contract["instance_id"]),
            api_key_path=Path(contract["instance_api_key_path"]),
            action="stop",
        )
        raise
    base.instance_action(
        instance_id=int(contract["instance_id"]),
        api_key_path=Path(contract["instance_api_key_path"]),
        action=lifecycle_action,
    )
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    arguments = parser.parse_args()
    packet = run(arguments.contract.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"] == "GREEN_JOB_TERMINAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
