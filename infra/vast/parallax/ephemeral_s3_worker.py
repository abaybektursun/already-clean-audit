"""Run one bounded job, publish its terminal JSON to S3, then leave Vast.

This module is intentionally dependency-free.  It runs inside a disposable
Vast instance after the controller has staged a 0600 contract and the
per-instance lifecycle key.  S3 durability is established by a successful
encrypted PUT before the instance asks Vast to destroy it.  If publication
fails, the instance asks Vast to stop instead so GPU billing ends without
discarding the only result copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


class WorkerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_result_url(url: str, *, result_key: str) -> None:
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    expected_path = "/" + "/".join(result_key.split("/"))
    if (
        parsed.scheme != "https"
        or parsed.netloc != "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
        or unquote(parsed.path) != expected_path
        or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
        or query.get("X-Amz-SignedHeaders") != ["host;x-amz-server-side-encryption"]
        or len(query.get("X-Amz-Signature", [])) != 1
        or not query.get("X-Amz-Expires")
    ):
        raise WorkerError("terminal PUT URL violates the exact S3 result contract")


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN":
        raise WorkerError("worker contract is not frozen schema v1")
    run_id = str(contract.get("run_id", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,79}", run_id):
        raise WorkerError("run id violates the frozen grammar")
    instance_id = contract.get("instance_id")
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise WorkerError("instance id is absent")
    root = Path(str(contract.get("runtime_root", "")))
    if not root.is_absolute():
        raise WorkerError("runtime root must be absolute")
    paths = {
        name: Path(str(contract.get(name, "")))
        for name in ("job_output_path", "job_log_path", "terminal_path", "instance_api_key_path")
    }
    if any(not path.is_absolute() for path in paths.values()):
        raise WorkerError("worker paths must be absolute")
    if any(not _within(path, root) for name, path in paths.items() if name != "instance_api_key_path"):
        raise WorkerError("job artifact path escapes the runtime root")
    if not str(paths["instance_api_key_path"]).startswith("/run/parallax/"):
        raise WorkerError("lifecycle key path escapes /run/parallax")
    command = contract.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise WorkerError("job command must be a nonempty string vector")
    timeout = contract.get("maximum_job_seconds")
    if not isinstance(timeout, int) or not 1 <= timeout <= 1800:
        raise WorkerError("job timeout escapes the frozen bound")
    maximum_result_bytes = contract.get("maximum_result_bytes")
    if not isinstance(maximum_result_bytes, int) or not 1024 <= maximum_result_bytes <= 64 * 1024 * 1024:
        raise WorkerError("terminal result size bound is invalid")
    result_key = str(contract.get("result_key", ""))
    if not result_key.startswith("parallax/results/") or not result_key.endswith("/terminal.json"):
        raise WorkerError("terminal result key is outside the PARALLAX result prefix")
    validate_result_url(str(contract.get("result_put_url", "")), result_key=result_key)
    if contract.get("post_upload_action") != "destroy" or contract.get("upload_failure_action") != "stop":
        raise WorkerError("terminal lifecycle actions are not fail-safe")
    return contract


def _tail(path: Path, maximum_bytes: int = 32768) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > maximum_bytes:
            handle.seek(-maximum_bytes, os.SEEK_END)
        return handle.read(maximum_bytes).decode("utf-8", errors="replace")


def _load_job_result(path: Path, maximum_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise WorkerError(f"job result size {size} escapes the frozen bound")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkerError("job result is not a JSON object")
    return {
        "present": True,
        "bytes": size,
        "sha256": sha256_file(path),
        "payload": payload,
    }


def put_terminal(*, url: str, terminal_path: Path, timeout_seconds: int = 60) -> dict[str, Any]:
    body = terminal_path.read_bytes()
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Content-Type": "application/json",
            "x-amz-server-side-encryption": "AES256",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            version_id = response.headers.get("x-amz-version-id")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        raise WorkerError(f"terminal S3 PUT failed: {type(error).__name__}") from error
    if status != 200 or not version_id:
        raise WorkerError(f"terminal S3 PUT lacked HTTP 200/version id: {status}")
    return {"http_status": status, "version_id": version_id, "bytes": len(body)}


def instance_action(
    *, instance_id: int, api_key_path: Path, action: str, attempts: int = 10
) -> None:
    if action not in {"stop", "destroy"} or not 1 <= attempts <= 20:
        raise ValueError("invalid Vast lifecycle action")
    api_key = api_key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise WorkerError("per-instance lifecycle key is absent")
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
    body = json.dumps({"state": "stopped"}).encode("utf-8") if action == "stop" else None
    method = "PUT" if action == "stop" else "DELETE"
    last_error = "not attempted"
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if 200 <= int(response.status) < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = type(error).__name__
        if attempt < attempts:
            time.sleep(2)
    raise WorkerError(f"Vast {action} failed after {attempts} attempts: {last_error}")


def run(contract_path: Path) -> dict[str, Any]:
    contract = validate_contract(json.loads(contract_path.read_text(encoding="utf-8")))
    output = Path(contract["job_output_path"])
    log = Path(contract["job_log_path"])
    terminal = Path(contract["terminal_path"])
    if output.exists() or terminal.exists():
        raise WorkerError("refusing to overwrite a prior job or terminal result")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = utc_now()
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
        result = _load_job_result(output, int(contract["maximum_result_bytes"]))
    except (OSError, ValueError, json.JSONDecodeError, WorkerError) as error:
        result = {"present": output.is_file()}
        result_error = f"{type(error).__name__}: {error}"
    green = returncode == 0 and result.get("present") is True and result_error is None
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "run_id": contract["run_id"],
        "provider": "vast.ai",
        "instance_id": contract["instance_id"],
        "stage": contract["stage"],
        "status": "GREEN_JOB_TERMINAL" if green else "RED_JOB_TERMINAL",
        "started_utc": started_utc,
        "wall_seconds": time.monotonic() - started,
        "job": {
            "returncode": returncode,
            "timed_out": timed_out,
            "log_tail": _tail(log),
            "result": result,
            "result_error": result_error,
        },
        "durability_contract": {
            "bucket": "pg-lineage-769063704696",
            "region": "us-east-1",
            "key": contract["result_key"],
            "required_encryption": "AES256",
            "post_upload_action": "destroy",
            "upload_failure_action": "stop",
        },
    }
    atomic_json(terminal, packet)
    try:
        put_terminal(url=contract["result_put_url"], terminal_path=terminal)
    except BaseException:
        instance_action(
            instance_id=int(contract["instance_id"]),
            api_key_path=Path(contract["instance_api_key_path"]),
            action="stop",
        )
        raise
    instance_action(
        instance_id=int(contract["instance_id"]),
        api_key_path=Path(contract["instance_api_key_path"]),
        action="destroy",
    )
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.contract.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
