#!/usr/local/bin/python3
"""Restartable outside-worker PG1 launch-state and terminal convergence monitor."""

from __future__ import annotations

import argparse
import base64
import hashlib
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

try:
    from . import pg1_native_launch_state as state_module
    from . import pg1_native_mechanical_rebase as mechanical_rebase
    from .pg1_native_vast_adapter import (
        VastAdapterError, discover_created_by_label, instance_state, select_instance,
    )
except ImportError:
    from infra.vast.parallax import pg1_native_launch_state as state_module
    from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase
    from infra.vast.parallax.pg1_native_vast_adapter import (
        VastAdapterError, discover_created_by_label, instance_state, select_instance,
    )


AWS = Path("/opt/parallax/control/bin/aws")
VAST = Path("/opt/parallax/control/bin/vastai")
CONTROL_ROOT = Path("/opt/parallax/control/source")
AWS_CONFIG = Path("/run/parallax/control-plane/aws/config")
AWS_CREDENTIALS = Path("/run/parallax/control-plane/aws/credentials")
VAST_KEY = Path("/run/secrets/pg1-native-control-plane-vast-api-key")
ARM_ACK = Path("/run/parallax/control-plane/pg1-deadman-arm-ack.json")
BOUND_ACK = Path("/run/parallax/control-plane/pg1-deadman-bound-ack.json")
STATE_RECEIPT = Path("/run/parallax/control-plane/pg1-canonical-state-receipt.json")
TRANSACTION = Path("/run/parallax/control-plane/pg1-worker-launch-transaction.json")
BUCKET = "pg-lineage-769063704696"
PREFIX = "parallax/authorizations/pg1-native-v3"
OWNER = "769063704696"
ISSUER = "PG1_CANONICAL_EXTERNAL_CONTROL_DEADMAN_V1"
API_BASE = "https://console.vast.ai/api/v0"
POLL_SECONDS = 2
MAX_BACKOFF_SECONDS = 30
TIMEOUT_SECONDS = 30
TERMINAL_OBSERVATIONS = 3
MAX_MUTATIONS_PER_CYCLE = 2
MECHANICAL_REST: Path | None = None


class ControlDeadmanError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o400); os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def fixed_env(*, api_key: str | None = None) -> dict[str, str]:
    value = {
        "PATH": "/opt/parallax/control/bin:/usr/bin:/bin", "HOME": "/nonexistent",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "AWS_PROFILE": "pg1-control-deadman", "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_CONFIG_FILE": str(AWS_CONFIG),
        "AWS_SHARED_CREDENTIALS_FILE": str(AWS_CREDENTIALS),
        "AWS_EC2_METADATA_DISABLED": "true", "PYTHONPATH": "/opt/parallax/control/source",
    }
    if api_key is not None:
        value["VAST_API_KEY"] = api_key
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise ControlDeadmanError(f"fixed control-plane file changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlDeadmanError(f"control-plane JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ControlDeadmanError("control-plane JSON must be an object")
    return value


def _identity(capsule: dict[str, Any]) -> tuple[str, str]:
    contract = capsule.get("contract") if isinstance(capsule, dict) else None
    if (
        not isinstance(contract, dict)
        or capsule.get("content_address")
        != {"algorithm": "sha256", "digest": digest(contract)}
    ):
        raise ControlDeadmanError("control deadman capsule address changed")
    return (str(capsule["content_address"]["digest"]),
            str(contract["attempt_identity"]["attempt_run_id"]))


def _api_key() -> str:
    if VAST_KEY.is_symlink() or not VAST_KEY.is_file() or stat.S_IMODE(VAST_KEY.stat().st_mode) != 0o400:
        raise ControlDeadmanError("control-plane Vast key changed")
    value = VAST_KEY.read_text(encoding="utf-8").strip()
    if not value:
        raise ControlDeadmanError("control-plane Vast key is empty")
    return value


def _run(command: list[str], *, api_key: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, env=fixed_env(api_key=api_key),
            cwd=str(CONTROL_ROOT), close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ControlDeadmanError("control-plane command failed or timed out") from error


def aws_prefix() -> list[str]:
    return [str(AWS), "--profile", "pg1-control-deadman", "--region", "us-east-1"]


def _key(packet_sha: str, kind: str, suffix: str = "latest.json") -> str:
    return f"{PREFIX}/control-deadman/{packet_sha}/{kind}/{suffix}"


def heartbeat_key(capsule: dict[str, Any]) -> str:
    capsule_sha, attempt = _identity(capsule)
    return f"{PREFIX}/deadman-heartbeats/{capsule_sha}/{attempt}.json"


def _explicit_absence(result: subprocess.CompletedProcess[str]) -> bool:
    text = f"{result.stdout}\n{result.stderr}".lower()
    return any(token in text for token in ("nosuchkey", "not found", "404"))


def _read_object(key: str) -> dict[str, Any]:
    head = _run([
        *aws_prefix(), "s3api", "head-object", "--bucket", BUCKET, "--key", key,
        "--expected-bucket-owner", OWNER,
    ])
    if head.returncode != 0:
        return {"status": "ABSENT" if _explicit_absence(head) else "UNKNOWN"}
    try:
        head_value = json.loads(head.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "UNKNOWN"}
    version_id = head_value.get("VersionId")
    if not version_id:
        return {"status": "UNKNOWN"}
    with tempfile.TemporaryDirectory(prefix="pg1-control-read-") as directory:
        target = Path(directory) / "value.json"
        result = _run([
            *aws_prefix(), "s3api", "get-object", "--bucket", BUCKET, "--key", key,
            "--version-id", str(version_id), "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", OWNER, str(target),
        ])
        if result.returncode != 0:
            return {"status": "ABSENT" if _explicit_absence(result) else "UNKNOWN"}
        try:
            response = json.loads(result.stdout or "{}")
            data = target.read_bytes(); value = json.loads(data)
        except (OSError, json.JSONDecodeError):
            return {"status": "UNKNOWN"}
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    if (
        not isinstance(value, dict) or response.get("ChecksumSHA256") != checksum
        or response.get("VersionId") != version_id
        or response.get("ETag") != head_value.get("ETag")
        or response.get("ServerSideEncryption") != "AES256"
        or head_value.get("ServerSideEncryption") != "AES256"
        or response.get("ContentLength") != len(data)
        or head_value.get("ContentLength") != len(data)
        or response.get("ContentType") != "application/json"
        or head_value.get("ContentType") != "application/json"
        or response.get("Metadata") != head_value.get("Metadata")
    ):
        return {"status": "CONTRADICTORY"}
    return {
        "status": "PRESENT", "key": key, "body": value,
        "etag": str(response["ETag"]),
        "version_id": str(version_id), "checksum_sha256": checksum,
        "metadata": response["Metadata"], "raw_bytes": data,
        "raw_sha256": hashlib.sha256(data).hexdigest(),
    }


def _verify_exact_storage(
    observed: dict[str, Any], *, key: str, payload: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
    """Prove one exact canonical object body and its complete storage identity."""

    raw = canonical_bytes(payload)
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if (
        observed.get("status") != "PRESENT"
        or observed.get("key") != key
        or observed.get("body") != payload
        or observed.get("raw_bytes") != raw
        or observed.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
        or observed.get("checksum_sha256") != checksum
        or observed.get("metadata") != metadata
        or not all(
            isinstance(observed.get(field), str) and bool(observed[field])
            for field in ("version_id", "etag")
        )
    ):
        raise ControlDeadmanError("exact canonical storage proof changed")
    return observed


def _put(
    *, key: str, payload: dict[str, Any], metadata: dict[str, str],
    prior_etag: str | None = None,
) -> dict[str, Any]:
    data = canonical_bytes(payload)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    condition = ["--if-none-match", "*"] if prior_etag is None else ["--if-match", prior_etag]
    with tempfile.TemporaryDirectory(prefix="pg1-control-put-") as directory:
        body = Path(directory) / "value.json"; body.write_bytes(data)
        result = _run([
            *aws_prefix(), "s3api", "put-object", "--bucket", BUCKET, "--key", key,
            "--body", str(body), *condition, "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum, "--content-length", str(len(data)),
            "--content-type", "application/json", "--server-side-encryption", "AES256",
            "--metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            "--expected-bucket-owner", OWNER,
        ])
    if result.returncode != 0:
        existing = _read_object(key)
        try:
            _verify_exact_storage(
                existing, key=key, payload=payload, metadata=metadata
            )
        except ControlDeadmanError as error:
            raise ControlDeadmanError(
                "control deadman CAS conflict is not exact replay"
            ) from error
        return {"etag": existing["etag"], "version_id": existing["version_id"],
                "checksum_sha256": existing["checksum_sha256"], "replayed_exact": True}
    try:
        response = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise ControlDeadmanError("control deadman durable write malformed") from error
    if (
        response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("ETag") or not response.get("VersionId")
    ):
        existing = _read_object(key)
        try:
            _verify_exact_storage(
                existing, key=key, payload=payload, metadata=metadata
            )
        except ControlDeadmanError as error:
            raise ControlDeadmanError(
                "control deadman durable write unverified"
            ) from error
        return {"etag": existing["etag"], "version_id": existing["version_id"],
                "checksum_sha256": existing["checksum_sha256"],
                "recovered_after_ambiguous_put": True}
    return {"etag": str(response["ETag"]), "version_id": str(response["VersionId"]),
            "checksum_sha256": checksum, "replayed_exact": False}


def _state_metadata(state: dict[str, Any]) -> dict[str, str]:
    return {
        "object-kind": "canonical-launch-state", "packet-sha256": state["packet_sha256"],
        "capsule-sha256": state["capsule_sha256"],
        "attempt-run-id": state["attempt_run_id"], "state-sha256": state["self_sha256"],
        "state-sequence": str(state["sequence"]),
        "raw-body-sha256": hashlib.sha256(canonical_bytes(state)).hexdigest(),
    }


def _verify_stored_state(observed: dict[str, Any]) -> dict[str, Any]:
    state = observed.get("body")
    try:
        state_module.validate(state)
    except state_module.LaunchStateError as error:
        raise ControlDeadmanError("stored canonical state invalid") from error
    _verify_exact_storage(
        observed, key=state_module.state_key(state["packet_sha256"]),
        payload=state, metadata=_state_metadata(state),
    )
    return state


def _load_state(
    capsule: dict[str, Any], transaction_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = _load_json(transaction_path)
    try:
        state_module.validate(cached)
    except state_module.LaunchStateError as error:
        raise ControlDeadmanError(str(error)) from error
    capsule_sha, attempt = _identity(capsule)
    observed = _read_object(state_module.state_key(cached["packet_sha256"]))
    if observed.get("status") != "PRESENT":
        raise ControlDeadmanError("canonical launch state unavailable")
    state = _verify_stored_state(observed)
    if (
        state["capsule_sha256"] != capsule_sha or state["attempt_run_id"] != attempt
        or not state_module.proves_descendant(state, cached)
    ):
        raise ControlDeadmanError("canonical launch state identity or ancestry changed")
    _atomic(transaction_path, state)
    return state, observed


def _publish_state(
    state: dict[str, Any], storage: dict[str, Any], phase: str, *,
    science_state: str | None = None, terminal_receipt: dict[str, Any] | None = None,
    **fields: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = state_module.transition(
        state, phase, science_state=science_state,
        terminal_receipt=terminal_receipt, **fields,
    )
    if not state_module.proves_direct_transition(candidate, state):
        raise ControlDeadmanError("canonical state candidate is not one exact legal edge")
    try:
        receipt = _put(
            key=state_module.state_key(state["packet_sha256"]), payload=candidate,
            metadata=_state_metadata(candidate), prior_etag=storage["etag"],
        )
    except Exception as error:
        observed = _read_object(state_module.state_key(state["packet_sha256"]))
        try:
            winner = _verify_stored_state(observed)
            exact_child = (
                winner == candidate
                and state_module.proves_direct_transition(winner, state)
            )
        except (TypeError, ControlDeadmanError, state_module.LaunchStateError):
            exact_child = False
        if (
            observed.get("status") != "PRESENT" or not exact_child
        ):
            raise ControlDeadmanError(
                "canonical state ambiguous reread is not the exact attempted child"
            ) from error
        receipt = {
            "etag": observed["etag"], "version_id": observed["version_id"],
            "checksum_sha256": observed["checksum_sha256"],
            "recovered_after_ambiguous_put": True,
        }
    _atomic(TRANSACTION, candidate)
    _atomic(STATE_RECEIPT, {
        "schema": "pg1-native-canonical-state-cas-receipt/v1",
        "state_sha256": candidate["self_sha256"], "sequence": candidate["sequence"],
        **receipt,
    })
    raw = canonical_bytes(candidate)
    return candidate, {
        **receipt, "status": "PRESENT",
        "key": state_module.state_key(candidate["packet_sha256"]),
        "body": candidate,
        "metadata": _state_metadata(candidate), "raw_bytes": raw,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def arm(capsule: dict[str, Any], transaction_path: Path) -> dict[str, Any]:
    state, _storage = _load_state(capsule, transaction_path)
    if state["phase"] != "PRELAUNCH_CLEANUP_INTENT":
        return {"status": "RESTART_RESUMED_CANONICAL_STATE",
                "packet_sha256": state["packet_sha256"],
                "transaction_sha256": digest(state), "destroy_authority": False}
    payload = {
        "schema": "pg1-native-prelaunch-cleanup-lease/v3",
        "status": "PRELAUNCH_CLEANUP_ARMED", "packet_sha256": state["packet_sha256"],
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"], "offer_id": state["offer_id"],
        "launch_label": state["launch_label"], "transaction_sha256": digest(state),
        "canonical_state_sequence": state["sequence"], "instance_id": None,
        "science_state": state_module.SCIENCE_DISABLED, "destroy_authority": False,
    }
    receipt = _put(
        key=_key(state["packet_sha256"], "prelaunch-arm"), payload=payload,
        metadata={"packet-sha256": state["packet_sha256"],
                  "capsule-sha256": state["capsule_sha256"],
                  "attempt-run-id": state["attempt_run_id"],
                  "object-kind": "prelaunch-cleanup-arm"},
    )
    ack = {**payload, **receipt}; _atomic(ARM_ACK, ack)
    return ack


def _bind_ack(state: dict[str, Any]) -> dict[str, Any]:
    if (
        state["phase"] not in {
            "INSTANCE_CONTEXT_BOUND", "HANDOFF_BOUND_DEADMAN_ACKED",
            "START_COMMIT_INTENT", "SCIENCE_RELEASED", "SCIENCE_RELEASE_UNKNOWN",
            "START_COMMIT_PROVEN_ABSENT",
        }
        or not isinstance(state.get("instance_id"), int)
        or not isinstance(state.get("machine_id"), int)
        or not state.get("container_id") or not state.get("stop_context_sha256")
        or not isinstance(state.get("authorization_sha256"), str)
        or len(state["authorization_sha256"]) != 64
        or not isinstance(state.get("authorization_receipt_sha256"), str)
        or len(state["authorization_receipt_sha256"]) != 64
        or not isinstance(state.get("mount_inventory_sha256"), str)
        or len(state["mount_inventory_sha256"]) != 64
    ):
        raise ControlDeadmanError("instance-bound canonical state changed")
    ack_nonce = hashlib.sha256(canonical_bytes({
        "scope": "PG1_INSTANCE_BOUND_DEADMAN_ACK_NONCE_V1",
        "packet_sha256": state["packet_sha256"],
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"],
        "instance_id": state["instance_id"],
        "stop_context_sha256": state["stop_context_sha256"],
    })).hexdigest()
    ack_id = hashlib.sha256(
        f"PG1_BOUND_ACK_V3:{ack_nonce}".encode()
    ).hexdigest()
    payload = {
        "schema": "pg1-native-instance-bound-deadman-ack/v3",
        "status": "INSTANCE_BOUND_DEADMAN_ARMED", "packet_sha256": state["packet_sha256"],
        "issuer": ISSUER,
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"], "instance_id": state["instance_id"],
        "offer_id": state["offer_id"], "machine_id": state["machine_id"],
        "container_id": state["container_id"],
        "stop_context_sha256": state["stop_context_sha256"],
        "authorization_body_sha256": state["authorization_sha256"],
        "authorization_receipt_sha256": state["authorization_receipt_sha256"],
        "mount_inventory_sha256": state["mount_inventory_sha256"],
        "ack_id": ack_id, "ack_nonce": ack_nonce,
        "replay_envelope": {
            "schema": "pg1-native-bound-ack-exact-replay/v1",
            "ack_id": ack_id, "one_shot": True, "exact_replay_only": True,
        },
        "destroy_authority": False,
    }
    receipt = _put(
        key=_key(state["packet_sha256"], "bound-arm"), payload=payload,
        metadata={"packet-sha256": state["packet_sha256"],
                  "capsule-sha256": state["capsule_sha256"],
                  "attempt-run-id": state["attempt_run_id"],
                  "instance-id": str(state["instance_id"]),
                  "object-kind": "instance-bound-deadman-arm"},
    )
    ack = {
        **payload, "ack_body_sha256": digest(payload),
        "etag": receipt["etag"], "version_id": receipt["version_id"],
        "checksum_sha256": receipt["checksum_sha256"],
    }
    _atomic(BOUND_ACK, ack)
    return ack


def vast_show_command(instance_id: int | None = None) -> list[str]:
    return [str(VAST), "show", "instances" if instance_id is None else "instance",
            *([] if instance_id is None else [str(instance_id)]), "--raw"]


def vast_stop_command(instance_id: int) -> list[str]:
    return [str(VAST), "stop", "instance", str(instance_id), "--raw"]


def vast_destroy_command(instance_id: int) -> list[str]:
    return [str(VAST), "destroy", "instance", str(instance_id), "--raw"]


def _rest_mutation(instance_id: int, api_key: str, *, action: str) -> dict[str, Any]:
    if action not in {"stop", "destroy"}:
        raise ControlDeadmanError("unsupported external REST mutation")
    if MECHANICAL_REST is not None:
        result = _run(
            [str(MECHANICAL_REST), "rest", action, str(instance_id)], api_key=api_key
        )
        if result.returncode != 0:
            raise ControlDeadmanError("deterministic mechanical REST mutation failed")
        try:
            value = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as error:
            raise ControlDeadmanError("deterministic mechanical REST response malformed") from error
        if value != {"channel": "REST", "action": action, "http_status": 200}:
            raise ControlDeadmanError("deterministic mechanical REST response changed")
        return value
    data = json.dumps({"state": "stopped"}).encode() if action == "stop" else None
    request = urllib.request.Request(
        f"{API_BASE}/instances/{instance_id}/", data=data,
        method="PUT" if action == "stop" else "DELETE",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = int(response.status)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
        raise ControlDeadmanError(f"external REST {action} failed") from error
    if not 200 <= status < 300:
        raise ControlDeadmanError(f"external REST {action} returned non-2xx")
    return {"channel": "REST", "action": action, "http_status": status}


def _persist_event(
    state: dict[str, Any], *, kind: str, sequence: int, payload: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "schema": "pg1-native-control-deadman-event/v2", "kind": kind,
        "sequence": sequence, "packet_sha256": state["packet_sha256"],
        "canonical_state_sha256": state["self_sha256"],
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"], "instance_id": state.get("instance_id"),
        "science_state": state["science_state"], "payload": payload,
        "wall_time_unix": time.time(),
    }
    receipt = _put(
        key=_key(state["packet_sha256"], kind, f"{sequence:020d}.json"), payload=value,
        metadata={"packet-sha256": state["packet_sha256"],
                  "attempt-run-id": state["attempt_run_id"],
                  "object-kind": f"control-deadman-{kind}"},
    )
    return {**value, **receipt}


def _observe(instance_id: int, api_key: str) -> tuple[str | None, dict[str, Any] | None]:
    shown = _run(vast_show_command(instance_id), api_key=api_key)
    if shown.returncode != 0:
        raise ControlDeadmanError("external canonical show returned nonzero")
    try:
        row = select_instance(json.loads(shown.stdout), instance_id)
    except (json.JSONDecodeError, VastAdapterError) as error:
        raise ControlDeadmanError("external canonical show malformed") from error
    return (None, None) if row is None else (instance_state(row), row)


def _discover(state: dict[str, Any], api_key: str) -> int | None:
    shown = _run(vast_show_command(), api_key=api_key)
    if shown.returncode != 0:
        raise ControlDeadmanError("prelaunch orphan discovery returned nonzero")
    try:
        found = discover_created_by_label(json.loads(shown.stdout), launch_label=state["launch_label"])
    except (json.JSONDecodeError, VastAdapterError) as error:
        raise ControlDeadmanError("prelaunch orphan discovery malformed or ambiguous") from error
    return None if found is None else int(found["instance_id"])


def _start_commit_status(state: dict[str, Any]) -> dict[str, Any]:
    observed = _read_object(state_module.start_commit_key(state["packet_sha256"]))
    if observed.get("status") != "PRESENT":
        return observed
    body = observed["body"]
    contract = body.get("contract") if isinstance(body, dict) else None
    if (
        body.get("schema") != "pg1-native-canonical-start-commit/v1"
        or body.get("status") != "CANONICAL_START_COMMITTED"
        or not isinstance(contract, dict)
        or body.get("content_address") != {"algorithm": "sha256", "digest": digest(contract)}
        or contract.get("packet_sha256") != state["packet_sha256"]
        or contract.get("capsule_sha256") != state["capsule_sha256"]
        or contract.get("attempt_run_id") != state["attempt_run_id"]
        or contract.get("instance_id") != state.get("instance_id")
        or contract.get("science_released") is not True
        or contract.get("canonical_state_sha256") not in (
            [state["self_sha256"], *state["ancestor_state_sha256"]]
        )
        or observed.get("metadata") != {
            "object-kind": "canonical-start-commit",
            "packet-sha256": state["packet_sha256"],
            "capsule-sha256": state["capsule_sha256"],
            "attempt-run-id": state["attempt_run_id"],
            "canonical-state-sha256": contract.get("canonical_state_sha256"),
        }
    ):
        return {"status": "CONTRADICTORY"}
    return {**observed, "start_commit_sha256": body["content_address"]["digest"]}


def _validate_destroy_request(
    request: dict[str, Any], state: dict[str, Any], start: dict[str, Any],
) -> str:
    gate = request.get("destroy_gate")
    state_is_destroying = state["phase"] == "CLEANUP_DESTROYING"
    parent_sequence = state["sequence"] - 1 if state_is_destroying else state["sequence"]
    parent_sha256 = (
        state["previous_state_sha256"] if state_is_destroying else state["self_sha256"]
    )
    common = (
        state_is_destroying
        and request.get("schema") == "pg1-native-external-destroy-request/v2"
        and request.get("status") == "DURABLE_EXTERNAL_DESTROY_REQUEST"
        and request.get("capsule_sha256") == state["capsule_sha256"]
        and request.get("attempt_run_id") == state["attempt_run_id"]
        and request.get("instance_id") == state.get("instance_id")
        and gate in {"COMPLETE_ARTIFACT_DURABILITY", "PRE_SCIENCE_ZERO_BODY_DURABILITY"}
        and request.get("teardown_authorized") is True
        and isinstance(request.get("evidence_sha256"), str)
        and len(request["evidence_sha256"]) == 64
        and state.get("destroy_request") == {
            "sha256": digest(request), "storage": "CANONICAL_STATE_RAW_BODY",
        }
    )
    if not common:
        raise ControlDeadmanError("external destroy request state sequence or digest changed")
    if gate == "PRE_SCIENCE_ZERO_BODY_DURABILITY":
        evidence = request.get("pre_science_zero_body_evidence")
        expected_evidence = {
            "schema": "pg1-native-pre-science-zero-body-evidence/v1",
            "status": "VERIFIED_NO_SCIENCE_BODY_OR_OUTPUT_CREATED",
            "archive_body_reads": 0, "model_body_reads": 0, "data_body_reads": 0,
            "confirmation_items_read": 0, "science_output_files": 0,
            "science_checkpoints": 0, "science_released": False,
        }
        cleanup_reference = request.get("cleanup_receipt")
        if (
            state["science_state"] != state_module.SCIENCE_DISABLED
            or request.get("science_released") is not False
            or request.get("start_commit_classification") != "START_COMMIT_PROVEN_ABSENT"
            or start.get("status") != "ABSENT"
            or request.get("parent_launch_state_sequence") != parent_sequence
            or request.get("parent_launch_state_sha256") != parent_sha256
            or request.get("consecutive_absences") != 3
            or evidence != expected_evidence
            or request.get("evidence_sha256") != digest(expected_evidence)
            or not isinstance(cleanup_reference, dict)
            or set(cleanup_reference) != {
                "key", "sha256", "version_id", "etag", "checksum_sha256",
            }
            or cleanup_reference.get("key")
            != f"{PREFIX}/gateway-cleanup/{state['packet_sha256']}.json"
        ):
            raise ControlDeadmanError("stale or contradictory pre-science destroy request")
        cleanup = _read_object(cleanup_reference["key"])
        try:
            cleanup_body = _validate_cleanup_observation(cleanup, state)
        except (TypeError, ControlDeadmanError) as error:
            raise ControlDeadmanError("pre-science cleanup receipt changed") from error
        if (
            cleanup_reference != {
                "key": cleanup_reference["key"], "sha256": digest(cleanup_body),
                "version_id": cleanup.get("version_id"), "etag": cleanup.get("etag"),
                "checksum_sha256": cleanup.get("checksum_sha256"),
            }
            or cleanup_body.get("status")
            != "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED"
            or cleanup_body.get("instance_id") != state["instance_id"]
            or request.get("parent_launch_state_raw_sha256")
            != state.get("previous_raw_body_sha256")
            or any(
                request.get(field) != value
                for field, value in cleanup_body.items()
                if field not in {"schema", "status"}
            )
        ):
            raise ControlDeadmanError("pre-science cleanup receipt changed")
    elif (
        state["science_state"] != state_module.SCIENCE_RELEASED
        or start.get("status") != "PRESENT"
        or request.get("start_commit_sha256") != start.get("start_commit_sha256")
        or request.get("launch_state_sequence") != parent_sequence
        or request.get("launch_state_sha256") != parent_sha256
    ):
        raise ControlDeadmanError("complete-artifact destroy lacks canonical start commit")
    return str(gate)


def _cleanup_key(state: dict[str, Any]) -> str:
    return f"{PREFIX}/gateway-cleanup/{state['packet_sha256']}.json"


def _cleanup_metadata(state: dict[str, Any], instance_id: int) -> dict[str, str]:
    return {
        "packet-sha256": state["packet_sha256"],
        "instance-id": str(instance_id),
        "object-kind": "gateway-orphan-cleanup",
    }


def _validate_cleanup_observation(
    observed: dict[str, Any], state: dict[str, Any],
) -> dict[str, Any]:
    receipt = observed.get("body")
    expected_fields = {
        "schema", "status", "packet_sha256", "instance_id", "failure",
        "stop_attempts", "errors", "consecutive_terminal_observations",
        "science_released", "science_state", "destroy_gate",
        "start_commit_classification",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_fields
        or receipt.get("schema") != "pg1-native-gateway-orphan-cleanup/v2"
        or receipt.get("packet_sha256") != state["packet_sha256"]
        or isinstance(receipt.get("instance_id"), bool)
        or not isinstance(receipt.get("instance_id"), int)
        or receipt["instance_id"] <= 0
        or not isinstance(receipt.get("failure"), str)
        or not receipt["failure"]
        or any(
            isinstance(receipt.get(field), bool)
            or not isinstance(receipt.get(field), int)
            or receipt[field] < 0
            for field in (
                "stop_attempts", "errors", "consecutive_terminal_observations"
            )
        )
        or receipt.get("science_state") not in state_module.SCIENCE_STATES
        or receipt.get("science_released")
        is not (receipt.get("science_state") == state_module.SCIENCE_RELEASED)
        or receipt.get("status") not in {
            "RELEASED_WORKER_STOPPED_SCRATCH_PRESERVED",
            "ORPHAN_STOPPED_SCIENCE_NEVER_RELEASED",
            "RELEASED_WORKER_STOP_UNCONFIRMED_SCRATCH_PRESERVED",
            "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED",
        }
    ):
        raise ControlDeadmanError("immutable cleanup receipt identity changed")
    _verify_exact_storage(
        observed, key=_cleanup_key(state), payload=receipt,
        metadata=_cleanup_metadata(state, receipt["instance_id"]),
    )
    if receipt["status"] == "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED" and (
        receipt["science_state"] != state_module.SCIENCE_DISABLED
        or receipt["science_released"] is not False
        or receipt["destroy_gate"] != "PRE_SCIENCE_ZERO_BODY_DURABILITY"
        or receipt["start_commit_classification"] != "START_COMMIT_PROVEN_ABSENT"
        or receipt["stop_attempts"] != 6
        or receipt["consecutive_terminal_observations"] >= TERMINAL_OBSERVATIONS
    ):
        raise ControlDeadmanError("pre-science cleanup receipt gate changed")
    return receipt


def _build_pre_science_destroy_request(
    receipt: dict[str, Any], cleanup_storage: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    if (
        state["science_state"] != state_module.SCIENCE_DISABLED
        or state.get("instance_id") is not None
        and receipt["instance_id"] != state.get("instance_id")
        or receipt["status"]
        != "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED"
    ):
        raise ControlDeadmanError("cleanup receipt cannot authorize this state")
    zero_body_evidence = {
        "schema": "pg1-native-pre-science-zero-body-evidence/v1",
        "status": "VERIFIED_NO_SCIENCE_BODY_OR_OUTPUT_CREATED",
        "archive_body_reads": 0, "model_body_reads": 0, "data_body_reads": 0,
        "confirmation_items_read": 0, "science_output_files": 0,
        "science_checkpoints": 0, "science_released": False,
    }
    return {
        **receipt,
        "schema": "pg1-native-external-destroy-request/v2",
        "status": "DURABLE_EXTERNAL_DESTROY_REQUEST",
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"],
        "teardown_authorized": True,
        "evidence_sha256": digest(zero_body_evidence),
        "pre_science_zero_body_evidence": zero_body_evidence,
        "cleanup_receipt": {
            "key": _cleanup_key(state), "sha256": digest(receipt),
            "version_id": cleanup_storage["version_id"],
            "etag": cleanup_storage["etag"],
            "checksum_sha256": cleanup_storage["checksum_sha256"],
        },
        "parent_launch_state_sequence": state["sequence"],
        "parent_launch_state_sha256": state["self_sha256"],
        "parent_launch_state_raw_sha256": hashlib.sha256(
            canonical_bytes(state)
        ).hexdigest(),
        "start_commit_sha256": state.get("start_commit_sha256"),
        "terminal_state": None,
        "consecutive_absences": TERMINAL_OBSERVATIONS,
    }


def _recover_cleanup_intent(
    state: dict[str, Any], storage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently adopt one exact gateway receipt into durable destroy intent."""

    cleanup_storage = _read_object(_cleanup_key(state))
    if cleanup_storage.get("status") == "ABSENT":
        return state, storage
    if cleanup_storage.get("status") != "PRESENT":
        raise ControlDeadmanError("cleanup recovery receipt existence is inconclusive")
    receipt = _validate_cleanup_observation(cleanup_storage, state)
    if receipt["status"] != "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED":
        return state, storage
    if state["phase"] == "CLEANUP_DESTROYING":
        start = _start_commit_status(state)
        _validate_destroy_request(state.get("destroy_request_body"), state, start)
        return state, storage
    if (
        state["phase"] in state_module.TERMINAL_PHASES
        or state["science_state"] != state_module.SCIENCE_DISABLED
        or "CLEANUP_DESTROYING"
        not in state_module.TRANSITIONS.get(state["phase"], frozenset())
    ):
        return state, storage
    request = _build_pre_science_destroy_request(receipt, cleanup_storage, state)
    return _publish_state(
        state, storage, "CLEANUP_DESTROYING",
        science_state=state_module.SCIENCE_DISABLED,
        instance_id=receipt["instance_id"],
        start_commit_classification="START_COMMIT_PROVEN_ABSENT",
        destroy_request={
            "sha256": digest(request), "storage": "CANONICAL_STATE_RAW_BODY",
        },
        destroy_request_body=request,
        destroy_gate="PRE_SCIENCE_ZERO_BODY_DURABILITY",
    )


def _effective_science(state: dict[str, Any], start: dict[str, Any]) -> str:
    if start.get("status") == "PRESENT":
        return state_module.SCIENCE_RELEASED
    if start.get("status") in {"UNKNOWN", "CONTRADICTORY"}:
        return state_module.SCIENCE_RELEASE_UNKNOWN
    if state["science_state"] != state_module.SCIENCE_DISABLED:
        return state_module.SCIENCE_RELEASE_UNKNOWN
    return state_module.SCIENCE_DISABLED


def _action_required(
    capsule: dict[str, Any], state: dict[str, Any], start: dict[str, Any],
) -> tuple[str | None, str | None]:
    if state["phase"] == "CLEANUP_STOPPING":
        return "stop", None
    effective = _effective_science(state, start)
    if state["phase"] == "CLEANUP_DESTROYING":
        try:
            gate = _validate_destroy_request(
                state.get("destroy_request_body"), state, start,
            )
            return "destroy", gate
        except (TypeError, ControlDeadmanError):
            # A contradictory canonical body fails closed to preservation.
            return "stop", None
    if effective == state_module.SCIENCE_RELEASE_UNKNOWN:
        return "stop", None
    if (
        effective == state_module.SCIENCE_DISABLED
        and time.time() - float(state.get("wall_time_unix", 0)) > 30
        and state["phase"] not in {"PRELAUNCH_CLEANUP_INTENT", "CREATE_INTENT_ARMED"}
    ):
        return "stop", None
    lease = _read_object(heartbeat_key(capsule))
    if effective == state_module.SCIENCE_RELEASED and (
        lease.get("status") != "PRESENT"
        or not isinstance(lease["body"].get("wall_time_unix"), (int, float))
        or time.time() - float(lease["body"]["wall_time_unix"]) > 30
    ):
        return "stop", None
    return None, None


def _validate_terminal_receipt(
    payload: dict[str, Any], state: dict[str, Any], *, action: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema", "status", "action", "packet_sha256", "capsule_sha256",
        "attempt_run_id", "instance_id", "parent_state_sha256",
        "parent_state_sequence", "terminal_state", "consecutive_observations",
        "attempts", "errors", "destroy_gate", "science_state",
        "external_control_plane", "receipt_durable",
    }
    expected_status = (
        "TERMINAL_DESTROYED_DURABLE" if action == "destroy"
        else "TERMINAL_STOPPED_PRESERVED"
    )
    state_is_terminal = state["phase"] in state_module.TERMINAL_PHASES
    parent_sha256 = (
        state["previous_state_sha256"] if state_is_terminal else state["self_sha256"]
    )
    parent_sequence = state["sequence"] - 1 if state_is_terminal else state["sequence"]
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload.get("schema") != "pg1-native-external-control-terminal/v3"
        or payload.get("status") != expected_status
        or payload.get("action") != action
        or payload.get("packet_sha256") != state["packet_sha256"]
        or payload.get("capsule_sha256") != state["capsule_sha256"]
        or payload.get("attempt_run_id") != state["attempt_run_id"]
        or payload.get("instance_id") != state.get("instance_id")
        or payload.get("parent_state_sha256") != parent_sha256
        or payload.get("parent_state_sequence") != parent_sequence
        or payload.get("terminal_state") != ("destroyed" if action == "destroy" else "stopped")
        or payload.get("receipt_durable") is not True
        or payload.get("consecutive_observations") != 3
        or payload.get("science_state") != state["science_state"]
        or payload.get("external_control_plane") is not True
        or isinstance(payload.get("attempts"), bool)
        or not isinstance(payload.get("attempts"), int)
        or payload["attempts"] < 0
        or isinstance(payload.get("errors"), bool)
        or not isinstance(payload.get("errors"), int)
        or payload["errors"] < 0
        or payload.get("destroy_gate") != (
            state.get("destroy_gate") if action == "destroy" else None
        )
    ):
        raise ControlDeadmanError("immutable terminal receipt identity changed")
    return payload


def _terminal_metadata(state: dict[str, Any]) -> dict[str, str]:
    return {
        "packet-sha256": state["packet_sha256"],
        "instance-id": str(state["instance_id"]),
        "object-kind": "immutable-external-terminal-receipt",
    }


def _terminal_reference(
    key: str, payload: dict[str, Any], observed: dict[str, Any], *, action: str,
) -> dict[str, Any]:
    return {
        "key": key, "sha256": digest(payload),
        "version_id": observed.get("version_id"), "etag": observed.get("etag"),
        "checksum_sha256": observed.get("checksum_sha256"), "action": action,
    }


def _validate_terminal_observation(
    observed: dict[str, Any], state: dict[str, Any], *, action: str,
) -> dict[str, Any]:
    payload = observed.get("body")
    if not isinstance(payload, dict):
        raise ControlDeadmanError("immutable terminal receipt body changed")
    _validate_terminal_receipt(payload, state, action=action)
    _verify_exact_storage(
        observed,
        key=state_module.terminal_receipt_key(state["packet_sha256"], action),
        payload=payload, metadata=_terminal_metadata(state),
    )
    return payload


def _validate_terminal_state_reread(
    observed: dict[str, Any], expected: dict[str, Any], prior: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if observed.get("status") != "PRESENT" or not isinstance(observed.get("body"), dict):
        raise ControlDeadmanError("terminal canonical state reread unavailable")
    winner = observed["body"]
    try:
        winner = _verify_stored_state(observed)
    except ControlDeadmanError as error:
        raise ControlDeadmanError("terminal canonical state reread invalid") from error
    if (
        winner != expected
        or not state_module.proves_descendant(winner, prior)
        or not state_module.proves_direct_transition(winner, prior)
    ):
        raise ControlDeadmanError("terminal canonical state identity or ancestry changed")
    return winner, observed


def _terminal_receipt(
    state: dict[str, Any], storage: dict[str, Any], *, action: str,
    attempts: int, errors: int, observations: int, destroy_gate: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    key = state_module.terminal_receipt_key(state["packet_sha256"], action)
    existing = _read_object(key)
    if existing.get("status") == "PRESENT":
        payload = _validate_terminal_observation(existing, state, action=action)
        receipt = existing
    elif existing.get("status") == "ABSENT":
        payload = {
            "schema": "pg1-native-external-control-terminal/v3",
            "status": (
                "TERMINAL_DESTROYED_DURABLE" if action == "destroy"
                else "TERMINAL_STOPPED_PRESERVED"
            ),
            "action": action, "packet_sha256": state["packet_sha256"],
            "capsule_sha256": state["capsule_sha256"],
            "attempt_run_id": state["attempt_run_id"], "instance_id": state["instance_id"],
            "parent_state_sha256": state["self_sha256"],
            "parent_state_sequence": state["sequence"],
            "terminal_state": "destroyed" if action == "destroy" else "stopped",
            "consecutive_observations": observations, "attempts": attempts,
            "errors": errors, "destroy_gate": destroy_gate,
            "science_state": state["science_state"],
            "external_control_plane": True, "receipt_durable": True,
        }
        stored = _put(
            key=key, payload=payload,
            metadata=_terminal_metadata(state),
        )
        receipt = _read_object(key)
        _validate_terminal_observation(receipt, state, action=action)
        if any(
            stored.get(field) != receipt.get(field)
            for field in ("etag", "version_id", "checksum_sha256")
        ):
            raise ControlDeadmanError("terminal receipt write/reread identity changed")
    else:
        raise ControlDeadmanError("terminal receipt existence is inconclusive")
    reference = _terminal_reference(key, payload, receipt, action=action)
    terminal_phase = (
        "TERMINAL_DESTROYED_DURABLE" if action == "destroy"
        else "TERMINAL_STOPPED_PRESERVED"
    )
    terminal_state, terminal_storage = _publish_state(
        state, storage, terminal_phase, terminal_receipt=reference,
    )
    return payload, terminal_state, terminal_storage


def _converge_terminal_orphan(
    state: dict[str, Any], storage: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    if state["phase"] in state_module.TERMINAL_PHASES:
        reference = state.get("terminal_receipt")
        if not isinstance(reference, dict):
            raise ControlDeadmanError("terminal canonical state receipt reference changed")
        observed = _read_object(reference["key"])
        action = "destroy" if state["phase"] == "TERMINAL_DESTROYED_DURABLE" else "stop"
        payload = _validate_terminal_observation(observed, state, action=action)
        if (
            _terminal_reference(
                reference["key"], payload, observed, action=action,
            ) != reference
        ):
            raise ControlDeadmanError("terminal canonical state receipt reference changed")
        return payload, state, storage
    if state["phase"] not in {"CLEANUP_STOPPING", "CLEANUP_DESTROYING"}:
        return None
    action = "destroy" if state["phase"] == "CLEANUP_DESTROYING" else "stop"
    observed = _read_object(state_module.terminal_receipt_key(state["packet_sha256"], action))
    if observed.get("status") != "PRESENT":
        return None
    payload = _validate_terminal_observation(observed, state, action=action)
    reference = _terminal_reference(
        state_module.terminal_receipt_key(state["packet_sha256"], action),
        payload, observed, action=action,
    )
    phase = "TERMINAL_DESTROYED_DURABLE" if action == "destroy" else "TERMINAL_STOPPED_PRESERVED"
    terminal, terminal_storage = _publish_state(
        state, storage, phase, terminal_receipt=reference,
    )
    return payload, terminal, terminal_storage


def supervise_action(
    capsule: dict[str, Any], state: dict[str, Any], storage: dict[str, Any], *,
    action: str, destroy_gate: str | None,
) -> dict[str, Any]:
    target_phase = "CLEANUP_DESTROYING" if action == "destroy" else "CLEANUP_STOPPING"
    if action == "destroy" and state["phase"] != "CLEANUP_DESTROYING":
        raise ControlDeadmanError("destroy mutation requires canonical destroying intent")
    if state["phase"] != target_phase:
        state, storage = _publish_state(
            state, storage, target_phase, destroy_gate=destroy_gate,
        )
    converged = _converge_terminal_orphan(state, storage)
    if converged is not None:
        return converged[0]
    instance_id = int(state["instance_id"])
    attempts = 0; errors = 0; observations = 0
    while True:
        try:
            key = _api_key()
            observed_state, _row = _observe(instance_id, key)
            if action == "destroy":
                observations = observations + 1 if observed_state is None else 0
            else:
                observations = observations + 1 if observed_state in {"stopped", "exited", "offline"} else 0
            _persist_event(
                state, kind="heartbeat", sequence=time.time_ns(),
                payload={"action": action, "state": observed_state,
                         "observations": observations, "attempts": attempts, "errors": errors},
            )
            if observations >= TERMINAL_OBSERVATIONS:
                payload, _terminal, _terminal_storage = _terminal_receipt(
                    state, storage, action=action, attempts=attempts, errors=errors,
                    observations=observations, destroy_gate=destroy_gate,
                )
                return payload
            for channel in range(MAX_MUTATIONS_PER_CYCLE):
                attempts += 1
                _persist_event(
                    state, kind="attempt", sequence=time.time_ns(),
                    payload={"action": action, "channel": "CLI" if channel == 0 else "REST",
                             "intent_durable_before_mutation": True},
                )
                try:
                    if channel == 0:
                        command = vast_destroy_command(instance_id) if action == "destroy" else vast_stop_command(instance_id)
                        result = _run(command, api_key=key)
                        if result.returncode != 0:
                            raise ControlDeadmanError(f"external CLI {action} returned nonzero")
                    else:
                        _rest_mutation(instance_id, key, action=action)
                except Exception as error:
                    errors += 1
                    try:
                        _persist_event(
                            state, kind="error", sequence=time.time_ns(),
                            payload={"action": action, "channel": "CLI" if channel == 0 else "REST",
                                     "error": type(error).__name__, "process_continues": True},
                        )
                    except Exception:
                        pass
        except Exception as error:
            errors += 1
            try:
                _persist_event(
                    state, kind="error", sequence=time.time_ns(),
                    payload={"action": action, "error": type(error).__name__,
                             "process_continues": True},
                )
            except Exception:
                pass
        time.sleep(min(MAX_BACKOFF_SECONDS, POLL_SECONDS * (2 ** min(errors, 4))))


def run(capsule: dict[str, Any], transaction_path: Path) -> dict[str, Any]:
    errors = 0
    while True:
        try:
            arm(capsule, transaction_path)
            break
        except Exception:
            errors += 1
            time.sleep(min(MAX_BACKOFF_SECONDS, POLL_SECONDS * (2 ** min(errors, 4))))
    bound_state_sha: str | None = None
    while True:
        try:
            state, storage = _load_state(capsule, transaction_path)
            state, storage = _recover_cleanup_intent(state, storage)
            converged = _converge_terminal_orphan(state, storage)
            if converged is not None:
                return converged[0]
            api_key = _api_key()
            if not isinstance(state.get("instance_id"), int):
                orphan = _discover(state, api_key)
                if orphan is not None:
                    state, storage = _publish_state(
                        state, storage, "CLEANUP_STOPPING", instance_id=orphan,
                        science_state=state_module.SCIENCE_DISABLED,
                    )
                    return supervise_action(
                        capsule, state, storage, action="stop", destroy_gate=None
                    )
                time.sleep(POLL_SECONDS); continue
            if (
                state["phase"] in {
                    "INSTANCE_CONTEXT_BOUND", "HANDOFF_BOUND_DEADMAN_ACKED",
                    "START_COMMIT_INTENT", "SCIENCE_RELEASED", "SCIENCE_RELEASE_UNKNOWN",
                    "START_COMMIT_PROVEN_ABSENT",
                }
                and bound_state_sha is None
            ):
                _bind_ack(state); bound_state_sha = state["self_sha256"]
            start = _start_commit_status(state)
            effective = _effective_science(state, start)
            if effective != state["science_state"] and state["phase"] == "START_COMMIT_INTENT":
                phase = "SCIENCE_RELEASED" if effective == state_module.SCIENCE_RELEASED else "SCIENCE_RELEASE_UNKNOWN"
                state, storage = _publish_state(
                    state, storage, phase, science_state=effective,
                    start_commit_sha256=start.get("start_commit_sha256"),
                )
            action, gate = _action_required(capsule, state, start)
            if action is not None:
                return supervise_action(
                    capsule, state, storage, action=action, destroy_gate=gate
                )
            _persist_event(
                state, kind="heartbeat", sequence=time.time_ns(),
                payload={"status": "WATCHING", "errors": errors,
                         "bound_ack": bound_state_sha is not None},
            )
            errors = 0
        except Exception as error:
            errors += 1
            try:
                state, _storage = _load_state(capsule, transaction_path)
                _persist_event(
                    state, kind="error", sequence=time.time_ns(),
                    payload={"error": type(error).__name__, "process_continues": True},
                )
            except Exception:
                pass
        time.sleep(min(MAX_BACKOFF_SECONDS, POLL_SECONDS * (2 ** min(errors, 4))))


def configure_from_source_location() -> dict[str, Any] | None:
    """Derive any mechanical root from these resolved copied source bytes."""

    global AWS, VAST, CONTROL_ROOT, AWS_CONFIG, AWS_CREDENTIALS, VAST_KEY
    global ARM_ACK, BOUND_ACK
    global STATE_RECEIPT, TRANSACTION, MECHANICAL_REST, POLL_SECONDS, MAX_BACKOFF_SECONDS
    discovered = mechanical_rebase.control_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    contract, _contract_path = discovered
    root = contract["root"]
    def at(path: str) -> Path:
        return mechanical_rebase.rebased(root, Path(path))
    AWS = at("/opt/parallax/control/bin/aws"); VAST = at("/opt/parallax/control/bin/vastai")
    CONTROL_ROOT = at("/opt/parallax/control/source")
    AWS_CONFIG = at("/run/parallax/control-plane/aws/config")
    AWS_CREDENTIALS = at("/run/parallax/control-plane/aws/credentials")
    VAST_KEY = at("/run/secrets/pg1-native-control-plane-vast-api-key")
    ARM_ACK = at("/run/parallax/control-plane/pg1-deadman-arm-ack.json")
    BOUND_ACK = at("/run/parallax/control-plane/pg1-deadman-bound-ack.json")
    STATE_RECEIPT = at("/run/parallax/control-plane/pg1-canonical-state-receipt.json")
    TRANSACTION = at("/run/parallax/control-plane/pg1-worker-launch-transaction.json")
    MECHANICAL_REST = VAST
    POLL_SECONDS = 0.01; MAX_BACKOFF_SECONDS = 0.05
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--control-deadman", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--launch-transaction", type=Path)
    args = parser.parse_args(argv)
    configure_from_source_location()
    if not args.control_deadman or args.capsule is None or args.launch_transaction is None:
        raise ControlDeadmanError("control-plane deadman argv changed")
    print(json.dumps(run(_load_json(args.capsule), args.launch_transaction), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
