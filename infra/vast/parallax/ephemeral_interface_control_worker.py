"""Supervise one PG2 interface control and enforce phase-aware teardown."""

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

try:  # Staged as a standalone sibling on the worker.
    import ephemeral_s3_worker as base
except ModuleNotFoundError:
    from . import ephemeral_s3_worker as base


RUN_ID = "parallax-pg2-interface-control-s3-v3"
EXPECTED_COMMAND = [
    "python3",
    "/opt/parallax/ephemeral_interface_control_job.py",
    "--input-contract",
    "/run/parallax/interface-input-contract.json",
    "--artifact-post",
    "/run/parallax/interface-artifact-post.json",
]
MAXIMUM_RESULT_BYTES = 4 * 1024 * 1024


class InterfaceWorkerError(RuntimeError):
    pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "FROZEN_SELECTED_INTERFACE_WORKER"
        or contract.get("run_id") != RUN_ID
        or contract.get("stage") != "training_interface_control"
        or not isinstance(contract.get("instance_id"), int)
        or int(contract["instance_id"]) <= 0
    ):
        raise InterfaceWorkerError("interface worker contract is not exact schema v1")
    root = Path(str(contract.get("runtime_root", "")))
    if root != Path("/workspace/parallax"):
        raise InterfaceWorkerError("interface worker runtime root changed")
    names = (
        "job_output_path",
        "job_log_path",
        "terminal_path",
        "phase_path",
        "instance_api_key_path",
    )
    paths = {name: Path(str(contract.get(name, ""))) for name in names}
    if any(not path.is_absolute() for path in paths.values()):
        raise InterfaceWorkerError("interface worker paths must be absolute")
    if any(
        not _within(path, root)
        for name, path in paths.items()
        if name != "instance_api_key_path"
    ):
        raise InterfaceWorkerError("interface worker artifact path escapes scratch")
    if paths["instance_api_key_path"] != Path("/run/parallax/instance-api-key"):
        raise InterfaceWorkerError("interface worker lifecycle key path changed")
    if contract.get("command") != EXPECTED_COMMAND:
        raise InterfaceWorkerError("interface worker command changed")
    if contract.get("maximum_job_seconds") != 1200:
        raise InterfaceWorkerError("interface worker timeout changed")
    if contract.get("maximum_result_bytes") != MAXIMUM_RESULT_BYTES:
        raise InterfaceWorkerError("interface worker result bound changed")
    result_key = str(contract.get("result_key", ""))
    if result_key != f"parallax/results/{RUN_ID}/terminal.json":
        raise InterfaceWorkerError("interface terminal key changed")
    base.validate_result_url(str(contract.get("result_put_url", "")), result_key=result_key)
    if (
        contract.get("archive_durable_action") != "destroy"
        or contract.get("pre_science_terminal_durable_action") != "destroy"
        or contract.get("science_started_archive_not_durable_action") != "stop"
        or contract.get("terminal_upload_failure_action") != "stop"
        or contract.get("automatic_retry") is not False
    ):
        raise InterfaceWorkerError("interface phase-aware lifecycle policy changed")
    return contract


def _archive_proof(result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("present") is not True or not isinstance(result.get("payload"), dict):
        return None
    payload = result["payload"]
    proof = payload.get("artifact_proof")
    if (
        payload.get("status") != "GREEN_INTERFACE_CONTROL_ARCHIVE_DURABLE"
        or payload.get("run_id") != RUN_ID
        or not isinstance(proof, dict)
        or proof.get("key") != f"parallax/results/{RUN_ID}/artifacts.tar"
        or proof.get("server_side_encryption") != "AES256"
        or proof.get("http_status") != 201
        or not isinstance(proof.get("version_id"), str)
        or not proof["version_id"]
        or not isinstance(proof.get("bytes"), int)
        or not 1_200_000_000 <= int(proof["bytes"]) <= 2_100_000_000
        or not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("sha256", "")))
        or not isinstance(proof.get("checksum_sha256_base64"), str)
        or not proof["checksum_sha256_base64"]
    ):
        return None
    return proof


def _phase(path: Path) -> dict[str, Any]:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"phase": "ABSENT", "science_started": False}
    if row.get("run_id") != RUN_ID or not isinstance(row.get("science_started"), bool):
        raise InterfaceWorkerError("interface phase marker is malformed")
    return row


def run(contract_path: Path) -> dict[str, Any]:
    contract = validate_contract(json.loads(contract_path.read_text(encoding="utf-8")))
    output = Path(contract["job_output_path"])
    log = Path(contract["job_log_path"])
    terminal = Path(contract["terminal_path"])
    phase_path = Path(contract["phase_path"])
    if output.exists() or terminal.exists():
        raise InterfaceWorkerError("refusing to overwrite interface result/terminal")
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
    phase = _phase(phase_path)
    archive_proof = _archive_proof(result)
    archive_durable = archive_proof is not None
    pre_science_red = (
        result.get("present") is True
        and isinstance(result.get("payload"), dict)
        and result["payload"].get("status") == "RED_TOKENIZER_PREFLIGHT_BEFORE_WEIGHT_HYDRATION"
        and result["payload"].get("science_started") is False
        and phase.get("science_started") is False
    )
    green = returncode == 0 and archive_durable and result_error is None
    pre_science = phase.get("science_started") is False
    if archive_durable or pre_science:
        lifecycle_action = "destroy"
    else:
        lifecycle_action = "stop"
    status = (
        "GREEN_JOB_TERMINAL"
        if green
        else "RED_PREFLIGHT_TERMINAL"
        if pre_science_red
        else "RED_JOB_TERMINAL"
    )
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "provider": "vast.ai",
        "instance_id": contract["instance_id"],
        "stage": contract["stage"],
        "status": status,
        "started_utc": started_utc,
        "wall_seconds": time.monotonic() - started,
        "phase": phase,
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
            "archive_key": f"parallax/results/{RUN_ID}/artifacts.tar",
            "terminal_key": contract["result_key"],
            "archive_durable": archive_durable,
            "archive_proof": archive_proof,
            "pre_science": pre_science,
            "pre_science_report_complete": pre_science_red,
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
    args = parser.parse_args()
    packet = run(args.contract.resolve())
    print(json.dumps({"run_id": RUN_ID, "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"] == "GREEN_JOB_TERMINAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
