"""Persistent tracked consumer for the canonical PG1 stop marker.

Provider/key failures are retryable observations, never reasons to destroy
scratch.  This process has STOP authority only.  If the backend independently
destroys after its complete durability gate, this guardian may observe that
fact only after three successful canonical listings omit the bound id.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .pg1_execution_capsule import (
    AUTHORIZATION_BUCKET,
    AUTHORIZATION_PREFIX,
    CAPSULE_RUNTIME_PATH,
    EXPECTED_BUCKET_OWNER,
    GUARDIAN_POLL_SECONDS,
    GUARDIAN_TIMEOUT_SECONDS,
    PROVIDER_TIMEOUT_SECONDS,
    STOP_CONTEXT_RUNTIME_PATH,
    CapsuleError,
    canonical_bytes,
    load_capsule,
)
from .pg1_native_provider import (
    FIXED_ENV,
    aws_prefix,
    load_stop_context,
    load_vast_key,
    observe_vast_instance,
    run_json,
    run_vast_mutation,
    vast_stop_command,
)
from .pg1_native_vast_adapter import TERMINAL_STOP_STATES
from . import pg1_native_lifecycle as lifecycle


WORKSPACE_ROOT = Path("/workspace/parallax")
FAILURE_POINTER_PATH = WORKSPACE_ROOT / "failure-stop-pointer.json"
LOCK_PATH = WORKSPACE_ROOT / "pg1-native-guardian.lock"
PENDING_RECEIPT_PATH = WORKSPACE_ROOT / "pg1-native-guardian-terminal.pending.json"
RESULT_PATH = WORKSPACE_ROOT / "pg1-native-guardian-result.json"
HEARTBEAT_PATH = WORKSPACE_ROOT / "pg1-native-guardian-heartbeat.json"
TERMINAL_ABSENCE_OBSERVATIONS = 3
ACTION_TIMEOUT_SECONDS = 300
MAX_BACKOFF_SECONDS = 30


def marker_key(capsule_sha256: str, attempt_run_id: str) -> str:
    return f"{AUTHORIZATION_PREFIX}/stop-guardian/{capsule_sha256}/{attempt_run_id}.json"


def receipt_key(capsule_sha256: str, attempt_run_id: str) -> str:
    return (
        f"{AUTHORIZATION_PREFIX}/stop-guardian-receipts/{capsule_sha256}/"
        f"{attempt_run_id}.json"
    )


def marker_command(context: dict[str, Any], target: Path) -> list[str]:
    contract = context["contract"]
    return [
        *aws_prefix(), "s3api", "get-object",
        "--bucket", AUTHORIZATION_BUCKET,
        "--key", marker_key(contract["capsule_sha256"], contract["attempt_run_id"]),
        "--checksum-mode", "ENABLED",
        "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        str(target),
    ]


def _expected_metadata(context: dict[str, Any], *, kind: str) -> dict[str, str]:
    return {
        "capsule-sha256": context["contract"]["capsule_sha256"],
        "attempt-run-id": context["contract"]["attempt_run_id"],
        "instance-id": str(context["contract"]["instance_id"]),
        "object-kind": kind,
    }


def _verify_s3_download(
    response: Any,
    data: bytes,
    context: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    if (
        not isinstance(response, dict)
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("VersionId")
        or not response.get("ETag")
        or response.get("ContentLength") != len(data)
        or response.get("ContentType") != "application/json"
        or response.get("Metadata") != _expected_metadata(context, kind=kind)
    ):
        raise CapsuleError(f"guardian {kind} S3 identity changed")
    return response


def _probe_marker(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    with tempfile.TemporaryDirectory(prefix="pg1-guardian-poll-") as directory:
        target = Path(directory) / "marker.json"
        try:
            completed = subprocess.run(
                marker_command(context, target),
                check=False,
                capture_output=True,
                text=True,
                timeout=PROVIDER_TIMEOUT_SECONDS,
                env=FIXED_ENV,
            )
        except subprocess.TimeoutExpired as error:
            raise CapsuleError("guardian marker poll timed out") from error
        if completed.returncode != 0:
            return None
        try:
            response = json.loads(completed.stdout or "")
            data = target.read_bytes()
            marker = json.loads(data)
        except (json.JSONDecodeError, OSError) as error:
            raise CapsuleError("guardian marker response is invalid") from error
    _verify_s3_download(
        response, data, context, kind="stop-guardian-request"
    )
    if not isinstance(marker, dict):
        raise CapsuleError("guardian marker body is not an object")
    return marker, response


def validate_marker(marker: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    contract = context["contract"]
    if (
        set(marker) != {
            "schema", "capsule_sha256", "attempt_run_id", "stop_context_sha256",
            "instance_id", "offer_id", "machine_id", "container_id", "failure",
            "failure_pointer", "failure_pointer_sha256", "artifact_manifest",
            "scratch_preserved",
        }
        or marker.get("schema") != "pg1-native-stop-guardian-request/v3"
        or marker.get("capsule_sha256") != contract["capsule_sha256"]
        or marker.get("attempt_run_id") != contract["attempt_run_id"]
        or marker.get("stop_context_sha256") != context["content_address"]["digest"]
        or marker.get("instance_id") != contract["instance_id"]
        or marker.get("offer_id") != contract["offer_id"]
        or marker.get("machine_id") != contract["machine_id"]
        or marker.get("container_id") != contract["container_id"]
        or marker.get("failure_pointer") != str(FAILURE_POINTER_PATH)
        or marker.get("artifact_manifest") != str(WORKSPACE_ROOT / "artifacts.manifest.json")
        or marker.get("scratch_preserved") is not True
    ):
        raise CapsuleError("guardian marker attempt or instance identity changed")
    pointer_digest = marker.get("failure_pointer_sha256")
    if pointer_digest is not None and (
        not isinstance(pointer_digest, str)
        or len(pointer_digest) != 64
        or any(character not in "0123456789abcdef" for character in pointer_digest)
    ):
        raise CapsuleError("guardian failure pointer digest changed")
    return marker


def _synthetic_marker(context: dict[str, Any], failure: str) -> dict[str, Any]:
    contract = context["contract"]
    return {
        "schema": "pg1-native-stop-guardian-request/v3",
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"],
        "stop_context_sha256": context["content_address"]["digest"],
        "instance_id": contract["instance_id"],
        "offer_id": contract["offer_id"],
        "machine_id": contract["machine_id"],
        "container_id": contract["container_id"],
        "failure": failure,
        "failure_pointer": str(FAILURE_POINTER_PATH),
        "failure_pointer_sha256": None,
        "artifact_manifest": str(WORKSPACE_ROOT / "artifacts.manifest.json"),
        "scratch_preserved": True,
    }


def _preserve_pointer(marker: dict[str, Any], context: dict[str, Any]) -> Path:
    for _attempt in range(8):
        prior = lifecycle.read(FAILURE_POINTER_PATH, context)
        if prior is not None and prior["state"] == "FAILURE_STOP_REQUIRED":
            return FAILURE_POINTER_PATH
        if prior is not None and prior["state"] in lifecycle.TERMINAL_STATES:
            raise CapsuleError("guardian cannot regress terminal lifecycle")
        try:
            lifecycle.transition(
                FAILURE_POINTER_PATH,
                context,
                "FAILURE_STOP_REQUIRED",
                reason=str(marker["failure"]),
                expected_sequence=0 if prior is None else int(prior["sequence"]),
            )
            return FAILURE_POINTER_PATH
        except lifecycle.LifecycleCASFailure as error:
            if "sequence mismatch" not in str(error):
                raise CapsuleError(str(error)) from error
    raise CapsuleError("guardian lifecycle CAS retry budget exhausted")
    return FAILURE_POINTER_PATH


def _probe_local_pointer(context: dict[str, Any]) -> dict[str, Any] | None:
    """Only FAILURE_STOP_REQUIRED is an immediate local STOP trigger."""

    if not FAILURE_POINTER_PATH.is_file():
        return None
    try:
        pointer = lifecycle.read(FAILURE_POINTER_PATH, context)
    except lifecycle.LifecycleCASFailure as error:
        raise CapsuleError(str(error)) from error
    if pointer is None:
        return None
    if pointer["state"] != "FAILURE_STOP_REQUIRED":
        return None
    marker = _synthetic_marker(
        context, f"LOCAL_LIFECYCLE:{str(pointer.get('reason', 'UNKNOWN'))}"
    )
    marker["failure_pointer_sha256"] = sha256_file(FAILURE_POINTER_PATH)
    return marker


def _heartbeat(
    context: dict[str, Any], *, status: str, events: int, stop_attempts: int
) -> None:
    atomic_json(
        HEARTBEAT_PATH,
        {
            "schema": "pg1-native-local-guardian-heartbeat/v1",
            "status": status,
            "capsule_sha256": context["contract"]["capsule_sha256"],
            "attempt_run_id": context["contract"]["attempt_run_id"],
            "instance_id": context["contract"]["instance_id"],
            "events": events,
            "stop_attempts": stop_attempts,
            "wall_time_unix": time.time(),
            "destroy_authority": False,
        },
    )


def _claim(context: dict[str, Any]) -> bool:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(context["content_address"]["digest"] + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _completion(context: dict[str, Any]) -> dict[str, Any] | None:
    if not RESULT_PATH.is_file():
        return None
    value = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "pg1-native-guardian-result/v2"
        or value.get("capsule_sha256") != context["contract"]["capsule_sha256"]
        or value.get("attempt_run_id") != context["contract"]["attempt_run_id"]
        or value.get("instance_id") != context["contract"]["instance_id"]
        or value.get("terminal_state") != "stopped"
        or value.get("consecutive_terminal_observations") != 3
    ):
        raise CapsuleError("guardian completion identity changed")
    return value


def _receipt_put_command(
    context: dict[str, Any], body: Path, data: bytes
) -> list[str]:
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    contract = context["contract"]
    return [
        *aws_prefix(), "s3api", "put-object",
        "--bucket", AUTHORIZATION_BUCKET,
        "--key", receipt_key(contract["capsule_sha256"], contract["attempt_run_id"]),
        "--body", str(body),
        "--if-none-match", "*",
        "--checksum-algorithm", "SHA256",
        "--checksum-sha256", checksum,
        "--content-length", str(len(data)),
        "--content-type", "application/json",
        "--server-side-encryption", "AES256",
        "--metadata", json.dumps(
            _expected_metadata(context, kind="stop-guardian-terminal-receipt"),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
    ]


def _receipt_get_command(context: dict[str, Any], target: Path) -> list[str]:
    contract = context["contract"]
    return [
        *aws_prefix(), "s3api", "get-object",
        "--bucket", AUTHORIZATION_BUCKET,
        "--key", receipt_key(contract["capsule_sha256"], contract["attempt_run_id"]),
        "--checksum-mode", "ENABLED",
        "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        str(target),
    ]


def _persist_terminal(
    payload: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    data = canonical_bytes(payload)
    PENDING_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{PENDING_RECEIPT_PATH.name}.", dir=PENDING_RECEIPT_PATH.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, PENDING_RECEIPT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    replay = False
    try:
        response = run_json(_receipt_put_command(context, PENDING_RECEIPT_PATH, data))
        if (
            not isinstance(response, dict)
            or response.get("ChecksumSHA256") != checksum
            or response.get("ServerSideEncryption") != "AES256"
            or not response.get("VersionId")
            or not response.get("ETag")
        ):
            raise CapsuleError("guardian terminal receipt upload evidence changed")
    except Exception:
        with tempfile.TemporaryDirectory(prefix="pg1-guardian-receipt-replay-") as directory:
            target = Path(directory) / "receipt.json"
            response = run_json(_receipt_get_command(context, target))
            existing = target.read_bytes()
        if existing != data:
            raise CapsuleError("guardian terminal receipt replay body changed")
        _verify_s3_download(
            response,
            existing,
            context,
            kind="stop-guardian-terminal-receipt",
        )
        replay = True
    result = {
        **payload,
        "schema": "pg1-native-guardian-result/v2",
        "receipt_key": receipt_key(
            context["contract"]["capsule_sha256"],
            context["contract"]["attempt_run_id"],
        ),
        "receipt_version_id": str(response["VersionId"]),
        "receipt_etag": str(response["ETag"]),
        "receipt_checksum_sha256": checksum,
        "receipt_replayed_exact": replay,
        "receipt_durable": True,
    }
    atomic_json(RESULT_PATH, result)
    _transition_stopped_preserved(context, result)
    return result


def _transition_stopped_preserved(
    context: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    if (
        receipt.get("receipt_durable") is not True
        or receipt.get("terminal_state") != "stopped"
    ):
        raise CapsuleError("stopped lifecycle requires a durable STOP receipt")
    for _attempt in range(8):
        prior = lifecycle.read(FAILURE_POINTER_PATH, context)
        if prior is not None and prior["state"] == "TERMINAL_STOPPED_PRESERVED":
            return prior
        try:
            return lifecycle.transition(
                FAILURE_POINTER_PATH,
                context,
                "TERMINAL_STOPPED_PRESERVED",
                reason="PROVIDER_CONFIRMED_NONBILLABLE_STOP_SCRATCH_PRESERVED",
                expected_sequence=0 if prior is None else int(prior["sequence"]),
                terminal_receipt=receipt,
            )
        except lifecycle.LifecycleCASFailure as error:
            if "sequence mismatch" not in str(error):
                raise CapsuleError(str(error)) from error
    raise CapsuleError("guardian stopped lifecycle CAS retry budget exhausted")


def _terminal_payload(
    marker: dict[str, Any],
    context: dict[str, Any],
    *,
    status: str,
    terminal_state: str,
    events: list[dict[str, Any]],
    consecutive_absences: int,
    consecutive_terminal_observations: int,
) -> dict[str, Any]:
    if terminal_state != "stopped":
        raise CapsuleError("STOP-only guardian cannot infer DESTROY completion")
    return {
        "schema": "pg1-native-guardian-terminal-payload/v2",
        "status": status,
        "capsule_sha256": context["contract"]["capsule_sha256"],
        "attempt_run_id": context["contract"]["attempt_run_id"],
        "instance_id": context["contract"]["instance_id"],
        "marker_sha256": hashlib.sha256(canonical_bytes(marker)).hexdigest(),
        "terminal_state": terminal_state,
        "consecutive_absences": consecutive_absences,
        "consecutive_terminal_observations": consecutive_terminal_observations,
        "events": events,
    }


def _sleep_bounded(deadline: float, failures: int) -> None:
    remaining = max(0.0, deadline - time.monotonic())
    delay = min(
        float(MAX_BACKOFF_SECONDS),
        float(GUARDIAN_POLL_SECONDS) * (2 ** min(failures, 3)),
        remaining,
    )
    if delay > 0:
        time.sleep(delay)


def _act(
    marker: dict[str, Any], context: dict[str, Any], *, deadline: float
) -> dict[str, Any]:
    _preserve_pointer(marker, context)
    prior = _completion(context)
    if prior is not None:
        return prior
    if not _claim(context):
        while time.monotonic() < deadline:
            prior = _completion(context)
            if prior is not None:
                return prior
            _sleep_bounded(deadline, 0)
        # A dead owner cannot strand lifecycle action forever.  The exact
        # context-bound result still decides replay after takeover.
        LOCK_PATH.unlink(missing_ok=True)
        if not _claim(context):
            raise CapsuleError("guardian race takeover failed closed")

    instance_id = context["contract"]["instance_id"]
    events: list[dict[str, Any]] = []
    failures = 0
    stop_attempts = 0
    consecutive_absences = 0
    consecutive_terminal = 0
    pending_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        _heartbeat(
            context,
            status="STOP_ACTION_RETRYING",
            events=len(events),
            stop_attempts=stop_attempts,
        )
        if pending_payload is not None:
            try:
                return _persist_terminal(pending_payload, context)
            except Exception as error:
                failures += 1
                events.append(
                    {
                        "action": "terminal_receipt",
                        "status": "retry",
                        "error": type(error).__name__,
                    }
                )
                _sleep_bounded(deadline, failures)
                continue
        try:
            api_key = load_vast_key(context)
            observed = observe_vast_instance(context, api_key)
            failures = 0
            if observed is None:
                consecutive_terminal = 0
                consecutive_absences += 1
                events.append(
                    {
                        "action": "show",
                        "status": "absent",
                        "consecutive_absences": consecutive_absences,
                    }
                )
                # Absence is not teardown proof.  Only the backend's separate
                # durability gate may establish a valid destroy completion.
                # This STOP-only consumer therefore remains fail-loud and
                # keeps retrying instead of manufacturing a terminal state.
            else:
                consecutive_absences = 0
                events.append(
                    {"action": "show", "status": "observed", "state": observed["state"]}
                )
                if observed["state"] in TERMINAL_STOP_STATES:
                    consecutive_terminal += 1
                    if consecutive_terminal >= 3:
                        pending_payload = _terminal_payload(
                            marker,
                            context,
                            status="GUARDIAN_STOP_VERIFIED_TERMINAL",
                            terminal_state="stopped",
                            events=events,
                            consecutive_absences=0,
                            consecutive_terminal_observations=consecutive_terminal,
                        )
                        continue
                else:
                    consecutive_terminal = 0
            stop_attempts += 1
            mutation = run_vast_mutation(vast_stop_command(instance_id, api_key))
            events.append(
                {
                    "action": "stop",
                    "attempt": stop_attempts,
                    "status": "exit_zero_polling_for_state",
                    "exit_code": mutation["exit_code"],
                }
            )
        except Exception as error:
            failures += 1
            events.append(
                {
                    "action": "provider_retry",
                    "status": "retry",
                    "error": type(error).__name__,
                }
            )
        _sleep_bounded(deadline, failures)

    _heartbeat(
        context,
        status="STOP_UNCONFIRMED_RESTART_REQUIRED",
        events=len(events),
        stop_attempts=stop_attempts,
    )
    LOCK_PATH.unlink(missing_ok=True)
    raise CapsuleError("guardian STOP remains unconfirmed; restart and retry required")


def guard(*, capsule: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if context["contract"]["capsule_sha256"] != capsule["content_address"]["digest"]:
        raise CapsuleError("guardian capsule/context identity changed")
    prior = _completion(context)
    if prior is not None:
        return prior
    watch_deadline = time.monotonic() + GUARDIAN_TIMEOUT_SECONDS
    events: list[dict[str, Any]] = []
    consecutive_absences = 0
    consecutive_terminal = 0
    while time.monotonic() < watch_deadline:
        _heartbeat(
            context,
            status="WATCHING_LOCAL_POINTER_AND_S3_MARKER",
            events=len(events),
            stop_attempts=0,
        )
        try:
            local = _probe_local_pointer(context)
            if local is not None:
                return _act(
                    local,
                    context,
                    deadline=time.monotonic() + ACTION_TIMEOUT_SECONDS,
                )
        except Exception as error:
            events.append(
                {"action": "local_pointer", "status": "retry", "error": type(error).__name__}
            )
        try:
            probe = _probe_marker(context)
            if probe is not None:
                marker, _response = probe
                return _act(
                    validate_marker(marker, context),
                    context,
                    deadline=min(
                        watch_deadline + ACTION_TIMEOUT_SECONDS,
                        time.monotonic() + ACTION_TIMEOUT_SECONDS,
                    ),
                )
        except Exception as error:
            events.append(
                {"action": "marker_poll", "status": "retry", "error": type(error).__name__}
            )
        try:
            api_key = load_vast_key(context)
            observed = observe_vast_instance(context, api_key)
            if observed is None:
                consecutive_terminal = 0
                consecutive_absences += 1
                if consecutive_absences >= TERMINAL_ABSENCE_OBSERVATIONS:
                    events.append(
                        {
                            "action": "show",
                            "status": "absence_is_not_terminal_without_destroy_gate",
                            "consecutive_absences": consecutive_absences,
                        }
                    )
            else:
                consecutive_absences = 0
                if observed["state"] in TERMINAL_STOP_STATES:
                    consecutive_terminal += 1
                    if consecutive_terminal >= 3:
                        marker = _synthetic_marker(context, "INSTANCE_ALREADY_STOPPED")
                        _preserve_pointer(marker, context)
                        return _persist_terminal(
                            _terminal_payload(
                                marker,
                                context,
                                status="GUARDIAN_STOP_VERIFIED_TERMINAL",
                                terminal_state="stopped",
                                events=events,
                                consecutive_absences=0,
                                consecutive_terminal_observations=consecutive_terminal,
                            ),
                            context,
                        )
                else:
                    consecutive_terminal = 0
        except Exception as error:
            consecutive_absences = 0
            consecutive_terminal = 0
            events.append(
                {"action": "provider_watch", "status": "retry", "error": type(error).__name__}
            )
        _sleep_bounded(watch_deadline, len(events))

    marker = _synthetic_marker(context, "GUARDIAN_TIMEOUT_FAIL_CLOSED")
    return _act(
        marker,
        context,
        deadline=time.monotonic() + ACTION_TIMEOUT_SECONDS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--stop-context", type=Path)
    args = parser.parse_args()
    if (
        not args.guard
        or args.capsule != CAPSULE_RUNTIME_PATH
        or args.stop_context != STOP_CONTEXT_RUNTIME_PATH
    ):
        raise CapsuleError("guardian production argv changed")
    capsule = load_capsule(args.capsule)
    context = load_stop_context(args.stop_context, capsule=capsule)
    result = guard(capsule=capsule, context=context)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
