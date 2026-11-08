#!/usr/local/bin/python3
"""Independent PG1 STOP-only lease deadman.

This absolute image script is armed before bootstrap.  It observes the fixed
S3 lease, bootstrap heartbeat, guardian terminal receipt, and local failure
pointer.  Once STOP is required it never returns to watching and never exits
unverified: fixed Vast CLI STOP is retried with capped backoff, the established
Vast REST PUT STOP channel is added after the action deadline, and provider
show rows remain the only STOP proof.  This module has no DESTROY authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


AWS_EXECUTABLE = Path("/usr/local/bin/aws")
STOP_CONTEXT_RUNTIME_PATH = Path("/run/parallax/pg1-native-stop-context.json")
VAST_API_KEY_PATH = Path("/run/secrets/pg1-native-vast-api-key")
WORKSPACE_ROOT = Path("/workspace/parallax")
FAILURE_POINTER_PATH = WORKSPACE_ROOT / "failure-stop-pointer.json"
HEARTBEAT_PATH = WORKSPACE_ROOT / "pg1-native-deadman-heartbeat.json"
RESULT_PATH = WORKSPACE_ROOT / "pg1-native-deadman-result.json"
AUTHORIZATION_BUCKET = "pg-lineage-769063704696"
AUTHORIZATION_PREFIX = "parallax/authorizations/pg1-native-v3"
EXPECTED_BUCKET_OWNER = "769063704696"
PROVIDER_TIMEOUT_SECONDS = 30
POLL_SECONDS = 5
MAX_BACKOFF_SECONDS = 30
PREARM_SECONDS = 30
HEARTBEAT_STALE_SECONDS = 30
ALTERNATE_STOP_AFTER_SECONDS = 120
ABSENCE_OBSERVATIONS = 3
API_BASE = "https://console.vast.ai/api/v0"


class DeadmanError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fixed_env() -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "AWS_PROFILE": "fuelos",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": "/run/parallax/aws/config",
        "AWS_SHARED_CREDENTIALS_FILE": "/run/parallax/aws/credentials",
        "AWS_EC2_METADATA_DISABLED": "true",
        "PYTHONPATH": "/opt/parallax/image/source",
    }


def _aws_prefix() -> list[str]:
    return [str(AWS_EXECUTABLE), "--profile", "fuelos", "--region", "us-east-1"]


def _load_context() -> dict[str, Any]:
    path = STOP_CONTEXT_RUNTIME_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o400
    ):
        raise DeadmanError("deadman fixed stop context is absent")
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeadmanError("deadman stop context is unreadable") from error
    contract = context.get("contract") if isinstance(context, dict) else None
    if (
        not isinstance(contract, dict)
        or context.get("content_address")
        != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(_canonical_bytes(contract)).hexdigest(),
        }
        or contract.get("issuer_account") != EXPECTED_BUCKET_OWNER
        or not isinstance(contract.get("instance_id"), int)
        or isinstance(contract.get("instance_id"), bool)
        or contract["instance_id"] <= 0
        or not isinstance(contract.get("capsule_sha256"), str)
        or len(contract["capsule_sha256"]) != 64
        or not isinstance(contract.get("attempt_run_id"), str)
        or not contract["attempt_run_id"]
    ):
        raise DeadmanError("deadman stop context identity changed")
    return context


def lease_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/deadman-leases/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def bootstrap_heartbeat_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/deadman-heartbeats/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def guardian_receipt_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/stop-guardian-receipts/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def deadman_receipt_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/deadman-receipts/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def _metadata(context: dict[str, Any], kind: str) -> dict[str, str]:
    contract = context["contract"]
    return {
        "capsule-sha256": contract["capsule_sha256"],
        "attempt-run-id": contract["attempt_run_id"],
        "instance-id": str(contract["instance_id"]),
        "object-kind": kind,
    }


def _get_command(key: str, target: Path) -> list[str]:
    return [
        *_aws_prefix(), "s3api", "get-object",
        "--bucket", AUTHORIZATION_BUCKET,
        "--key", key,
        "--checksum-mode", "ENABLED",
        "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        str(target),
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROVIDER_TIMEOUT_SECONDS,
            env=_fixed_env(),
        )
    except subprocess.TimeoutExpired as error:
        raise DeadmanError("deadman fixed command timed out") from error


def _probe(
    context: dict[str, Any], *, key: str, kind: str
) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix="pg1-deadman-probe-") as directory:
        target = Path(directory) / "body.json"
        completed = _run(_get_command(key, target))
        if completed.returncode != 0:
            return None
        try:
            response = json.loads(completed.stdout or "")
            data = target.read_bytes()
            value = json.loads(data)
        except (OSError, json.JSONDecodeError) as error:
            raise DeadmanError("deadman observed malformed S3 evidence") from error
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    if (
        not isinstance(response, dict)
        or not isinstance(value, dict)
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or response.get("ContentLength") != len(data)
        or response.get("ContentType") != "application/json"
        or response.get("Metadata") != _metadata(context, kind)
    ):
        raise DeadmanError("deadman S3 evidence identity changed")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _heartbeat(
    context: dict[str, Any], *, status: str, attempts: int, errors: int
) -> None:
    _atomic_json(
        HEARTBEAT_PATH,
        {
            "schema": "pg1-native-external-deadman-heartbeat/v1",
            "status": status,
            "capsule_sha256": context["contract"]["capsule_sha256"],
            "attempt_run_id": context["contract"]["attempt_run_id"],
            "instance_id": context["contract"]["instance_id"],
            "stop_attempts": attempts,
            "errors": errors,
            "wall_time_unix": time.time(),
            "destroy_authority": False,
        },
    )


def _local_failure_required(context: dict[str, Any]) -> bool:
    if not FAILURE_POINTER_PATH.is_file():
        return False
    try:
        value = json.loads(FAILURE_POINTER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeadmanError("external deadman local lifecycle is unreadable") from error
    contract = context["contract"]
    if (
        not isinstance(value, dict)
        or value.get("schema") != "pg1-native-local-lifecycle/v1"
        or value.get("version") != 1
        or value.get("capsule_sha256") != contract["capsule_sha256"]
        or value.get("attempt_run_id") != contract["attempt_run_id"]
        or value.get("instance_id") != contract["instance_id"]
        or value.get("state") not in {
            "ARMED", "RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED",
            "TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE",
        }
        or isinstance(value.get("sequence"), bool)
        or not isinstance(value.get("sequence"), int)
        or value["sequence"] < 1
    ):
        raise DeadmanError("external deadman local lifecycle identity changed")
    return value["state"] == "FAILURE_STOP_REQUIRED"


def _validate_lease(context: dict[str, Any], lease: dict[str, Any]) -> dict[str, Any]:
    contract = context["contract"]
    if (
        lease.get("schema") != "pg1-native-external-deadman-lease/v1"
        or lease.get("status") not in {
            "ARMED", "RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED",
            "TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE",
        }
        or lease.get("version") != 1
        or isinstance(lease.get("sequence"), bool)
        or not isinstance(lease.get("sequence"), int)
        or lease["sequence"] < 1
        or lease.get("capsule_sha256") != contract["capsule_sha256"]
        or lease.get("attempt_run_id") != contract["attempt_run_id"]
        or lease.get("instance_id") != contract["instance_id"]
        or lease.get("stop_context_sha256") != context["content_address"]["digest"]
        or not isinstance(lease.get("issued_unix"), (int, float))
        or not isinstance(lease.get("hard_stop_unix"), (int, float))
        or float(lease["hard_stop_unix"]) <= float(lease["issued_unix"])
        or lease.get("heartbeat_stale_seconds") != HEARTBEAT_STALE_SECONDS
        or lease.get("destroy_authority") is not False
    ):
        raise DeadmanError("external deadman lease identity changed")
    return lease


def _rest_stop(instance_id: int, api_key: str) -> dict[str, Any]:
    data = json.dumps({"state": "stopped"}).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE}/instances/{instance_id}/",
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
            status = int(response.status)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
        raise DeadmanError("alternate Vast REST STOP failed") from error
    if not 200 <= status < 300:
        raise DeadmanError("alternate Vast REST STOP returned non-2xx")
    return {"schema": "pg1-vast-rest-stop-exit/v1", "http_status": status}


def _put_receipt(
    context: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    data = _canonical_bytes(payload)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="pg1-deadman-receipt-") as directory:
        body = Path(directory) / "receipt.json"
        body.write_bytes(data)
        command = [
            *_aws_prefix(), "s3api", "put-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", deadman_receipt_key(context),
            "--body", str(body),
            "--if-none-match", "*",
            "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum,
            "--content-length", str(len(data)),
            "--content-type", "application/json",
            "--server-side-encryption", "AES256",
            "--metadata", json.dumps(
                _metadata(context, "external-deadman-terminal-receipt"),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        ]
        completed = _run(command)
        if completed.returncode != 0:
            existing = _probe(
                context,
                key=deadman_receipt_key(context),
                kind="external-deadman-terminal-receipt",
            )
            if existing != payload:
                raise DeadmanError("external deadman receipt replay changed")
            replay = True
        else:
            try:
                response = json.loads(completed.stdout or "")
            except json.JSONDecodeError as error:
                raise DeadmanError("external deadman receipt response malformed") from error
            if (
                not isinstance(response, dict)
                or response.get("ChecksumSHA256") != checksum
                or response.get("ServerSideEncryption") != "AES256"
                or not response.get("VersionId")
                or not response.get("ETag")
            ):
                raise DeadmanError("external deadman receipt upload unverified")
            replay = False
    result = {**payload, "receipt_durable": True, "receipt_replayed_exact": replay}
    _atomic_json(RESULT_PATH, result)
    _transition_stopped_preserved(context, result)
    return result


def _transition_stopped_preserved(
    context: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    if (
        receipt.get("receipt_durable") is not True
        or receipt.get("terminal_state") != "stopped"
    ):
        raise DeadmanError("deadman stopped lifecycle requires durable STOP proof")
    try:
        prior = json.loads(FAILURE_POINTER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeadmanError("deadman stopped lifecycle prior state unreadable") from error
    if prior.get("state") == "TERMINAL_STOPPED_PRESERVED":
        return prior
    if prior.get("state") not in {
        "ARMED", "RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED"
    }:
        raise DeadmanError("deadman cannot regress local terminal lifecycle")
    value = {
        "schema": "pg1-native-local-lifecycle/v1",
        "version": 1,
        "state": "TERMINAL_STOPPED_PRESERVED",
        "sequence": prior["sequence"] + 1,
        "capsule_sha256": context["contract"]["capsule_sha256"],
        "attempt_run_id": context["contract"]["attempt_run_id"],
        "instance_id": context["contract"]["instance_id"],
        "stop_context_sha256": context["content_address"]["digest"],
        "previous_sha256": hashlib.sha256(_canonical_bytes(prior)).hexdigest(),
        "reason": "PROVIDER_CONFIRMED_NONBILLABLE_STOP_SCRATCH_PRESERVED",
        "wall_time_unix": time.time(),
        "terminal_receipt_sha256": hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest(),
    }
    _atomic_json(FAILURE_POINTER_PATH, value)
    return value


def _terminal_payload(
    context: dict[str, Any], *, terminal_state: str, attempts: int, errors: int
) -> dict[str, Any]:
    if terminal_state != "stopped":
        raise DeadmanError("external deadman cannot infer DESTROY terminal state")
    return {
        "schema": "pg1-native-external-deadman-terminal/v1",
        "status": "EXTERNAL_DEADMAN_STOP_VERIFIED",
        "capsule_sha256": context["contract"]["capsule_sha256"],
        "attempt_run_id": context["contract"]["attempt_run_id"],
        "instance_id": context["contract"]["instance_id"],
        "terminal_state": terminal_state,
        "stop_attempts": attempts,
        "errors": errors,
        "destroy_authority": False,
    }


def _sleep(errors: int) -> None:
    time.sleep(min(float(MAX_BACKOFF_SECONDS), float(POLL_SECONDS) * (2 ** min(errors, 3))))


def run_deadman(context: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    action_started: float | None = None
    stop_required = False
    stop_attempts = 0
    errors = 0
    consecutive_absences = 0
    _heartbeat(context, status="ARMED_BEFORE_BOOTSTRAP", attempts=0, errors=0)
    while True:
        status = "STOP_RETRY_FAIL_LOUD" if stop_required else "WATCHING_FIXED_S3_LEASE"
        _heartbeat(context, status=status, attempts=stop_attempts, errors=errors)
        if not stop_required:
            try:
                terminal = _probe(
                    context,
                    key=guardian_receipt_key(context),
                    kind="stop-guardian-terminal-receipt",
                )
                if (
                    terminal is not None
                    and terminal.get("terminal_state") == "stopped"
                    and terminal.get("receipt_durable") is True
                ):
                    return _put_receipt(
                        context,
                        _terminal_payload(
                            context,
                            terminal_state=str(terminal["terminal_state"]),
                            attempts=stop_attempts,
                            errors=errors,
                        ),
                    )
                lease = _probe(
                    context, key=lease_key(context), kind="external-deadman-lease"
                )
                heartbeat = _probe(
                    context,
                    key=bootstrap_heartbeat_key(context),
                    kind="bootstrap-supervisor-heartbeat",
                )
                now = time.time()
                if _local_failure_required(context):
                    stop_required = True
                elif lease is None and time.monotonic() - started >= PREARM_SECONDS:
                    stop_required = True
                elif lease is not None:
                    lease = _validate_lease(context, lease)
                    if (
                        lease["status"] == "FAILURE_STOP_REQUIRED"
                        or now >= float(lease["hard_stop_unix"])
                    ):
                        stop_required = True
                    elif heartbeat is None:
                        if now - float(lease["issued_unix"]) > HEARTBEAT_STALE_SECONDS:
                            stop_required = True
                    elif (
                        heartbeat.get("schema")
                        != "pg1-native-bootstrap-supervisor-heartbeat/v1"
                        or heartbeat.get("capsule_sha256")
                        != context["contract"]["capsule_sha256"]
                        or heartbeat.get("attempt_run_id")
                        != context["contract"]["attempt_run_id"]
                        or heartbeat.get("instance_id")
                        != context["contract"]["instance_id"]
                        or heartbeat.get("destroy_authority") is not False
                        or not isinstance(heartbeat.get("wall_time_unix"), (int, float))
                        or now - float(heartbeat["wall_time_unix"])
                        > HEARTBEAT_STALE_SECONDS
                    ):
                        stop_required = True
            except Exception:
                errors += 1
                if time.monotonic() - started >= PREARM_SECONDS:
                    stop_required = True
            if not stop_required:
                _sleep(errors)
                continue
            action_started = time.monotonic()

        provider: Any | None = None
        api_key: str | None = None
        try:
            provider = importlib.import_module("infra.vast.parallax.pg1_native_provider")
            api_key = provider.load_vast_key(context)
            observed = provider.observe_vast_instance(context, api_key)
            if observed is None:
                consecutive_absences += 1
            else:
                consecutive_absences = 0
                if observed["state"] in {"stopped", "exited", "offline"}:
                    payload = _terminal_payload(
                        context,
                        terminal_state="stopped",
                        attempts=stop_attempts,
                        errors=errors,
                    )
                    while True:
                        try:
                            return _put_receipt(context, payload)
                        except Exception:
                            errors += 1
                            _heartbeat(
                                context,
                                status="TERMINAL_RECEIPT_RETRY_FAIL_LOUD",
                                attempts=stop_attempts,
                                errors=errors,
                            )
                            _sleep(errors)
        except Exception:
            errors += 1
        if provider is not None and api_key is not None:
            try:
                stop_attempts += 1
                provider.run_vast_mutation(
                    provider.vast_stop_command(
                        context["contract"]["instance_id"], api_key
                    )
                )
            except Exception:
                errors += 1
            if (
                action_started is not None
                and time.monotonic() - action_started >= ALTERNATE_STOP_AFTER_SECONDS
            ):
                try:
                    _rest_stop(context["contract"]["instance_id"], api_key)
                except Exception:
                    errors += 1
        _sleep(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--deadman", action="store_true")
    parser.add_argument("--stop-context", type=Path)
    args = parser.parse_args(argv)
    if not args.deadman or args.stop_context != STOP_CONTEXT_RUNTIME_PATH:
        raise DeadmanError("external deadman argv changed")
    result = run_deadman(_load_context())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
