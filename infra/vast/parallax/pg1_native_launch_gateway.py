#!/usr/local/bin/python3
"""Orphan-safe PG1 control-plane create, bind, deadman, and start gateway."""

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
from pathlib import Path
from typing import Any

try:
    from . import pg1_native_launch_packet as packet_module
    from . import pg1_native_launch_authority as authority_module
    from . import pg1_native_launch_state as state_module
    from . import pg1_native_mechanical_rebase as mechanical_rebase
    from .pg1_native_vast_adapter import (
        VastAdapterError,
        discover_created_by_label,
        parse_create_response,
        select_instance,
        validate_created_binding,
    )
except ImportError:  # direct fixed-path executable
    from infra.vast.parallax import pg1_native_launch_packet as packet_module
    from infra.vast.parallax import pg1_native_launch_authority as authority_module
    from infra.vast.parallax import pg1_native_launch_state as state_module
    from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase
    from infra.vast.parallax.pg1_native_vast_adapter import (
        VastAdapterError,
        discover_created_by_label,
        parse_create_response,
        select_instance,
        validate_created_binding,
    )


CONTROL_ROOT = Path("/opt/parallax/control/source")
CONTROL_VAST = Path("/opt/parallax/control/bin/vastai")
CONTROL_AWS = Path("/opt/parallax/control/bin/aws")
CONTROL_DEADMAN = CONTROL_ROOT / "infra/vast/parallax/pg1_native_control_deadman.py"
CONTROL_VAST_KEY = Path("/run/secrets/pg1-native-control-plane-vast-api-key")
CONTROL_AWS_CONFIG = Path("/run/parallax/control-plane/aws/config")
CONTROL_AWS_CREDENTIALS = Path("/run/parallax/control-plane/aws/credentials")
CONTROL_RUNTIME_RECEIPT = Path("/run/parallax/control-plane/runtime-image-digest")
ARM_ACK = Path("/run/parallax/control-plane/pg1-deadman-arm-ack.json")
BOUND_ACK = Path("/run/parallax/control-plane/pg1-deadman-bound-ack.json")
TRANSACTION = packet_module.TRANSACTION_PATH
CAPSULE_PATH = packet_module.CAPSULE_PATH
PACKET_PATH = packet_module.PACKET_PATH
START_LEASE_REMOTE = packet_module.START_LEASE_PATH
AUTHORIZATION_BUCKET = "pg-lineage-769063704696"
AUTHORIZATION_PREFIX = "parallax/authorizations/pg1-native-v3"
EXPECTED_BUCKET_OWNER = "769063704696"
CONTROL_DEADMAN_ISSUER = "PG1_CANONICAL_EXTERNAL_CONTROL_DEADMAN_V1"
WORKER_CWD = Path("/opt/parallax/image/source")
COMMAND_TIMEOUT_SECONDS = 60
ARM_TIMEOUT_SECONDS = 60
BIND_TIMEOUT_SECONDS = 120
TERMINAL_OBSERVATIONS = 3
STATE_RECEIPT = Path("/run/parallax/control-plane/pg1-canonical-state-receipt.json")
START_COMMIT_PROBES = 3
MECHANICAL_DEADMAN_ARGV: list[str] | None = None
MECHANICAL_ROOT_CONTRACT: Path | None = None
MECHANICAL_WORKER_ROOT: Path | None = None


class LaunchGatewayError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixed(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise LaunchGatewayError(f"fixed gateway input changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchGatewayError(f"gateway JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise LaunchGatewayError("gateway input must be an object")
    return value


def _capsule_identity(capsule: dict[str, Any]) -> tuple[str, str]:
    contract = capsule.get("contract") if isinstance(capsule, dict) else None
    if (
        not isinstance(contract, dict)
        or capsule.get("content_address")
        != {"algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(contract)).hexdigest()}
    ):
        raise LaunchGatewayError("gateway capsule address changed")
    return (str(capsule["content_address"]["digest"]),
            str(contract["attempt_identity"]["attempt_run_id"]))


def authority_storage_digest(
    authorization: dict[str, Any], storage: dict[str, Any],
) -> str:
    return hashlib.sha256(canonical_bytes({
        "authorization_sha256": authorization["content_address"]["digest"],
        "version_id": storage["version_id"], "etag": storage["etag"],
        "checksum_sha256": storage["checksum_sha256"],
        "metadata": storage["metadata"],
    })).hexdigest()


def prebound_mount_inventory_digest(
    capsule: dict[str, Any], packet: dict[str, Any], context: dict[str, Any],
    instance_key_sha256: str,
) -> str:
    inputs = packet["contract"]["frozen_inputs"]
    inventory = {
        "/run/parallax/pg1-execution-capsule.json": capsule["content_address"]["digest"],
        "/run/parallax/pg1-native-stop-context.json": context["content_address"]["digest"],
        "/run/parallax/aws/config": inputs["control_mounts"]["aws_config_sha256"],
        "/run/parallax/aws/credentials": inputs["control_mounts"]["aws_credentials_sha256"],
        "/run/secrets/pg1-native-vast-api-key": instance_key_sha256,
        "/run/parallax/runtime-image-digest": inputs["runtime_image"]["receipt_sha256"],
        "/run/parallax/pg1-native-entry-contract.json": hashlib.sha256(
            canonical_bytes(production_entry_contract(capsule))
        ).hexdigest(),
        str(packet_module.WORKER_PACKET_PATH): packet["content_address"]["digest"],
    }
    return hashlib.sha256(canonical_bytes(inventory)).hexdigest()


def _secure(path: Path, *, mode: int, sha256: str | None = None) -> bytes:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != mode:
        raise LaunchGatewayError(f"fixed control-plane material changed: {path}")
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise LaunchGatewayError(f"fixed control-plane provenance changed: {path}")
    return data


def fixed_env(packet: dict[str, Any] | None = None) -> dict[str, str]:
    value = {
        "PATH": "/opt/parallax/control/bin:/usr/bin:/bin", "HOME": "/nonexistent",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(CONTROL_ROOT),
        "AWS_PROFILE": "pg1-control-deadman", "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_CONFIG_FILE": str(CONTROL_AWS_CONFIG),
        "AWS_SHARED_CREDENTIALS_FILE": str(CONTROL_AWS_CREDENTIALS),
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    if packet is not None:
        expected = packet["contract"]["frozen_inputs"]["control_mounts"][
            "control_vast_key_sha256"
        ]
        value["VAST_API_KEY"] = _secure(
            CONTROL_VAST_KEY, mode=0o400, sha256=expected
        ).decode("utf-8").strip()
    return value


def _run(command: list[str], *, packet: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT_SECONDS, cwd=str(CONTROL_ROOT), env=fixed_env(packet),
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LaunchGatewayError("fixed control-plane command failed or timed out") from error


def _atomic(path: Path, value: dict[str, Any], *, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, mode); os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_state(state: dict[str, Any], receipt: dict[str, Any]) -> None:
    state_module.validate(state)
    _atomic(TRANSACTION, state)
    _atomic(
        STATE_RECEIPT,
        {"schema": "pg1-native-canonical-state-cas-receipt/v1",
         "state_sha256": state["self_sha256"], "sequence": state["sequence"],
         **receipt},
    )


def validate_launch_authorization(
    authorization: dict[str, Any], storage: dict[str, Any],
    packet: dict[str, Any], capsule: dict[str, Any],
) -> dict[str, Any]:
    if MECHANICAL_ROOT_CONTRACT is None:
        authority_module.validate_authorization(authorization, capsule, packet)
        expected_metadata = authority_module.metadata(authorization)
    else:
        authority_module.validate_mechanical_authorization(
            authorization, capsule, packet
        )
        expected_metadata = {
            "object-kind": "mechanical-launch-authorization",
            "issuer": authority_module.MECHANICAL_ISSUER,
            "packet-sha256": authorization["contract"]["packet_sha256"],
            "capsule-sha256": authorization["contract"]["capsule_sha256"],
            "attempt-run-id": authorization["contract"]["attempt_run_id"],
        }
    if (
        storage.get("status") != "PRESENT"
        or storage.get("body") != authorization
        or storage.get("raw_bytes") != canonical_bytes(authorization)
        or not storage.get("version_id") or not storage.get("etag")
        or storage.get("metadata") != expected_metadata
    ):
        raise LaunchGatewayError("canonical fetched launch authorization changed")
    return authorization


def _aws_put(
    *, key: str, value: dict[str, Any], metadata: dict[str, str],
    packet: dict[str, Any], prior_etag: str | None = None,
) -> dict[str, Any]:
    data = canonical_bytes(value)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    condition = ["--if-none-match", "*"] if prior_etag is None else ["--if-match", prior_etag]
    with tempfile.TemporaryDirectory(prefix="pg1-gateway-put-") as directory:
        body = Path(directory) / "value.json"; body.write_bytes(data)
        command = [
            str(CONTROL_AWS), "--profile", "pg1-control-deadman", "--region", "us-east-1",
            "s3api", "put-object", "--bucket", AUTHORIZATION_BUCKET, "--key", key,
            "--body", str(body), *condition, "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum, "--content-length", str(len(data)),
            "--content-type", "application/json", "--server-side-encryption", "AES256",
            "--metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        ]
        completed = _run(command, packet=packet)
    if completed.returncode != 0:
        raise LaunchGatewayError("canonical conditional publication failed")
    try:
        response = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as error:
        raise LaunchGatewayError("canonical publication response malformed") from error
    if (
        response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("ETag") or not response.get("VersionId")
    ):
        raise LaunchGatewayError("canonical publication durability unverified")
    return {"etag": str(response["ETag"]), "version_id": str(response["VersionId"]),
            "checksum_sha256": checksum, "server_side_encryption": "AES256"}


def _explicit_absence(completed: subprocess.CompletedProcess[str]) -> bool:
    text = f"{completed.stdout}\n{completed.stderr}".lower()
    return any(token in text for token in ("nosuchkey", "not found", "404"))


def _read_object(
    *, key: str, packet: dict[str, Any], expected_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    head_command = [
        str(CONTROL_AWS), "--profile", "pg1-control-deadman", "--region", "us-east-1",
        "s3api", "head-object", "--bucket", AUTHORIZATION_BUCKET, "--key", key,
        "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
    ]
    head_result = _run(head_command, packet=packet)
    if head_result.returncode != 0:
        return {"status": "ABSENT" if _explicit_absence(head_result) else "UNKNOWN"}
    try:
        head = json.loads(head_result.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "UNKNOWN"}
    version_id = head.get("VersionId")
    if not version_id:
        return {"status": "UNKNOWN"}
    with tempfile.TemporaryDirectory(prefix="pg1-gateway-read-") as directory:
        target = Path(directory) / "object.json"
        get_command = [
            str(CONTROL_AWS), "--profile", "pg1-control-deadman", "--region", "us-east-1",
            "s3api", "get-object", "--bucket", AUTHORIZATION_BUCKET, "--key", key,
            "--version-id", str(version_id), "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER, str(target),
        ]
        get_result = _run(get_command, packet=packet)
        if get_result.returncode != 0:
            return {"status": "ABSENT" if _explicit_absence(get_result) else "UNKNOWN"}
        try:
            response = json.loads(get_result.stdout or "{}")
            data = target.read_bytes(); body = json.loads(data)
        except (OSError, json.JSONDecodeError):
            return {"status": "UNKNOWN"}
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    if (
        not isinstance(body, dict)
        or response.get("VersionId") != version_id
        or response.get("ETag") != head.get("ETag")
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or head.get("ServerSideEncryption") != "AES256"
        or response.get("ContentLength") != len(data)
        or head.get("ContentLength") != len(data)
        or response.get("ContentType") != "application/json"
        or head.get("ContentType") != "application/json"
        or response.get("Metadata") != head.get("Metadata")
        or expected_metadata is not None and head.get("Metadata") != expected_metadata
    ):
        return {"status": "CONTRADICTORY"}
    return {
        "status": "PRESENT", "body": body, "etag": str(head["ETag"]),
        "version_id": str(version_id), "checksum_sha256": checksum,
        "metadata": head["Metadata"], "raw_bytes": data,
        "raw_sha256": hashlib.sha256(data).hexdigest(),
    }


def _put_immutable_verified(
    *, key: str, value: dict[str, Any], metadata: dict[str, str],
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Conditionally create once, then accept only an exact durable reread."""

    stored: dict[str, Any] | None
    try:
        stored = _aws_put(
            key=key, value=value, metadata=metadata, packet=packet,
        )
    except LaunchGatewayError:
        stored = None
    observed = _read_object(
        key=key, packet=packet, expected_metadata=metadata,
    )
    if (
        observed.get("status") != "PRESENT"
        or observed.get("body") != value
        or observed.get("raw_bytes") != canonical_bytes(value)
        or observed.get("raw_sha256")
        != hashlib.sha256(canonical_bytes(value)).hexdigest()
        or observed.get("metadata") != metadata
        or not all(
            isinstance(observed.get(field), str) and bool(observed[field])
            for field in ("version_id", "etag", "checksum_sha256")
        )
        or stored is not None and any(
            stored.get(field) != observed.get(field)
            for field in ("version_id", "etag", "checksum_sha256")
        )
    ):
        raise LaunchGatewayError("immutable canonical publication reread changed")
    return observed


def fetch_launch_authorization(
    capsule: dict[str, Any], packet: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = authority_module.authorization_key(packet["content_address"]["digest"])
    observed = _read_object(key=key, packet=packet)
    if observed.get("status") != "PRESENT":
        raise LaunchGatewayError("canonical launch authorization is unavailable")
    body = observed["body"]
    validate_launch_authorization(body, observed, packet, capsule)
    if body["contract"]["issued_key"] != key:
        raise LaunchGatewayError("canonical launch authorization key changed")
    return body, observed


def _state_metadata(state: dict[str, Any]) -> dict[str, str]:
    return {
        "object-kind": "canonical-launch-state",
        "packet-sha256": state["packet_sha256"],
        "capsule-sha256": state["capsule_sha256"],
        "attempt-run-id": state["attempt_run_id"],
        "state-sha256": state["self_sha256"],
        "state-sequence": str(state["sequence"]),
        "raw-body-sha256": hashlib.sha256(canonical_bytes(state)).hexdigest(),
    }


def _verify_exact_state_observation(
    observed: dict[str, Any], expected: dict[str, Any], *,
    packet: dict[str, Any], prior: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = canonical_bytes(expected)
    winner = observed.get("body")
    try:
        state_module.validate(winner, packet=packet)
        legal = (
            winner["sequence"] == 1 and prior is None
            or prior is not None and state_module.proves_direct_transition(winner, prior)
        )
    except (TypeError, state_module.LaunchStateError):
        legal = False
    if (
        observed.get("status") != "PRESENT"
        or winner != expected
        or observed.get("raw_bytes") != raw
        or observed.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
        or observed.get("metadata") != _state_metadata(expected)
        or not legal
        or not all(
            isinstance(observed.get(field), str) and bool(observed[field])
            for field in ("version_id", "etag", "checksum_sha256")
        )
    ):
        raise LaunchGatewayError("canonical launch state raw observation changed")
    return observed


def _verify_stored_state_observation(
    observed: dict[str, Any], *, packet: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the exact canonical bytes and storage identity of a current state."""

    state = observed.get("body")
    try:
        state_module.validate(state, packet=packet)
    except (TypeError, state_module.LaunchStateError) as error:
        raise LaunchGatewayError("stored canonical launch state invalid") from error
    raw = canonical_bytes(state)
    if (
        observed.get("status") != "PRESENT"
        or observed.get("raw_bytes") != raw
        or observed.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
        or observed.get("metadata") != _state_metadata(state)
        or not all(
            isinstance(observed.get(field), str) and bool(observed[field])
            for field in ("version_id", "etag", "checksum_sha256")
        )
    ):
        raise LaunchGatewayError("stored canonical launch state raw identity changed")
    return state, observed


def _publish_state(
    packet: dict[str, Any], state: dict[str, Any], *, prior_etag: str | None,
    prior_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_module.validate(state, packet=packet)
    metadata = _state_metadata(state)
    try:
        receipt = _aws_put(
            key=state_module.state_key(state["packet_sha256"]), value=state,
            metadata=metadata, packet=packet, prior_etag=prior_etag,
        )
    except Exception as error:
        observed = _read_object(
            key=state_module.state_key(state["packet_sha256"]), packet=packet,
        )
        try:
            _verify_exact_state_observation(
                observed, state, packet=packet, prior=prior_state,
            )
        except LaunchGatewayError:
            raise LaunchGatewayError("canonical launch state CAS inconclusive") from error
        metadata = _state_metadata(state)
        receipt = {
            "etag": observed["etag"], "version_id": observed["version_id"],
            "checksum_sha256": observed["checksum_sha256"],
            "server_side_encryption": "AES256", "recovered_after_ambiguous_put": True,
        }
    _cache_state(state, receipt)
    return state, receipt


def _advance_state(
    packet: dict[str, Any], state: dict[str, Any], state_receipt: dict[str, Any],
    phase: str, *, science_state: str | None = None,
    **fields: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = state_module.transition(
        state, phase, science_state=science_state, **fields
    )
    return _publish_state(
        packet, candidate, prior_etag=state_receipt["etag"], prior_state=state,
    )


def consume_launch_authorization(
    authorization: dict[str, Any], storage: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    contract = authorization["contract"]
    marker = {
        "schema": "pg1-native-launch-authorization-consumption/v1",
        "status": "CONSUMED_EXACTLY_ONCE_BEFORE_CREATE",
        "authorization_sha256": authorization["content_address"]["digest"],
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"], "offer_id": contract["offer_id"],
        "authorization_version_id": storage["version_id"],
        "authorization_etag": storage["etag"],
        "authorization_checksum_sha256": storage["checksum_sha256"],
        "authorization_metadata_sha256": hashlib.sha256(
            canonical_bytes(storage["metadata"])
        ).hexdigest(),
        "one_shot_consumed": True, "science_released": False,
    }
    return _aws_put(
        key=authority_module.consumption_key(marker["authorization_sha256"]), value=marker,
        metadata={"packet-sha256": marker["packet_sha256"],
                  "capsule-sha256": marker["capsule_sha256"],
                  "attempt-run-id": marker["attempt_run_id"],
                  "object-kind": "launch-authorization-consumption"}, packet=packet,
    )


def control_deadman_argv() -> list[str]:
    if MECHANICAL_DEADMAN_ARGV is not None:
        return list(MECHANICAL_DEADMAN_ARGV)
    return [str(CONTROL_DEADMAN), "--control-deadman", "--capsule", str(CAPSULE_PATH),
            "--launch-transaction", str(TRANSACTION)]


BOUND_ACK_PAYLOAD_FIELDS = frozenset({
    "schema", "status", "packet_sha256", "issuer", "capsule_sha256",
    "attempt_run_id", "instance_id", "offer_id", "machine_id", "container_id",
    "stop_context_sha256", "authorization_body_sha256",
    "authorization_receipt_sha256", "mount_inventory_sha256", "ack_id",
    "ack_nonce", "replay_envelope", "destroy_authority",
})
BOUND_ACK_FIELDS = BOUND_ACK_PAYLOAD_FIELDS | {
    "ack_body_sha256", "etag", "version_id", "checksum_sha256",
}


def validate_bound_ack(
    value: dict[str, Any], transaction: dict[str, Any], packet: dict[str, Any],
) -> dict[str, Any]:
    payload = {key: value[key] for key in BOUND_ACK_PAYLOAD_FIELDS} if (
        isinstance(value, dict) and set(value) == BOUND_ACK_FIELDS
    ) else {}
    replay = value.get("replay_envelope") if isinstance(value, dict) else None
    expected_nonce = hashlib.sha256(canonical_bytes({
        "scope": "PG1_INSTANCE_BOUND_DEADMAN_ACK_NONCE_V1",
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": transaction.get("capsule_sha256"),
        "attempt_run_id": transaction.get("attempt_run_id"),
        "instance_id": transaction.get("instance_id"),
        "stop_context_sha256": transaction.get("stop_context_sha256"),
    })).hexdigest()
    expected_id = hashlib.sha256(
        f"PG1_BOUND_ACK_V3:{expected_nonce}".encode()
    ).hexdigest()
    expected_checksum = base64.b64encode(
        hashlib.sha256(canonical_bytes(payload)).digest()
    ).decode("ascii") if payload else None
    if (
        not payload or value.get("schema") != "pg1-native-instance-bound-deadman-ack/v3"
        or value.get("status") != "INSTANCE_BOUND_DEADMAN_ARMED"
        or value.get("issuer") != CONTROL_DEADMAN_ISSUER
        or value.get("packet_sha256") != packet["content_address"]["digest"]
        or any(value.get(field) != transaction.get(field) for field in (
            "capsule_sha256", "attempt_run_id", "instance_id", "offer_id",
            "machine_id", "container_id", "stop_context_sha256",
            "mount_inventory_sha256",
        ))
        or value.get("authorization_body_sha256")
        != transaction.get("authorization_sha256")
        or value.get("authorization_receipt_sha256")
        != transaction.get("authorization_receipt_sha256")
        or value.get("ack_body_sha256") != hashlib.sha256(
            canonical_bytes(payload)
        ).hexdigest()
        or value.get("ack_id") != expected_id
        or value.get("ack_nonce") != expected_nonce
        or replay != {
            "schema": "pg1-native-bound-ack-exact-replay/v1",
            "ack_id": value.get("ack_id"), "one_shot": True,
            "exact_replay_only": True,
        }
        or value.get("destroy_authority") is not False
        or value.get("checksum_sha256") != expected_checksum
        or not all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in ("etag", "version_id", "checksum_sha256")
        )
    ):
        raise LaunchGatewayError("instance-bound deadman ACK exact envelope changed")
    return value


def _wait_ack(path: Path, packet: dict[str, Any], *, status: str) -> dict[str, Any]:
    deadline = time.monotonic() + ARM_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if path.is_file():
            value = _load_fixed(path)
            transaction = _load_fixed(TRANSACTION)
            common = (
                value.get("status") == status
                and value.get("packet_sha256") == packet["content_address"]["digest"]
                and value.get("destroy_authority") is False
                and value.get("etag") and value.get("version_id")
            )
            prelaunch = (
                status == "PRELAUNCH_CLEANUP_ARMED"
                and value.get("transaction_sha256")
                == hashlib.sha256(canonical_bytes(transaction)).hexdigest()
            )
            bound = (
                status == "INSTANCE_BOUND_DEADMAN_ARMED"
                and validate_bound_ack(value, transaction, packet) is value
            )
            if common and (prelaunch or bound):
                return value
        time.sleep(0.25)
    raise LaunchGatewayError(f"external deadman {status} deadline expired")


def source_materializer_onstart(
    packet: dict[str, Any], capsule: dict[str, Any]
) -> str:
    materializer = packet["contract"]["onstart_materializer"]
    argv = [
        packet["content_address"]["digest"] if value == "${PACKET_SHA256}" else value
        for value in materializer["argv"]
    ]
    argv.extend(("--maximum-seconds", str(materializer["maximum_prestart_seconds"])))
    relative = Path("infra/vast/parallax/pg1_native_onstart_materializer.py")
    source = CONTROL_ROOT / relative
    expected = capsule["contract"]["runtime_wiring"][str(relative)]
    data = _secure(source, mode=0o755, sha256=expected)
    encoded = base64.b64encode(data).decode("ascii")
    environment = [f"{key}={value}" for key, value in materializer["environment"].items()]
    import shlex
    installer = (
        "import base64,os,sys;"
        "p=sys.argv[1];d=base64.b64decode(sys.argv[2],validate=True);"
        "os.makedirs(os.path.dirname(p),mode=0o700,exist_ok=True);"
        "t=p+'.'+str(os.getpid());f=open(t,'xb');f.write(d);f.flush();"
        "os.fsync(f.fileno());f.close();os.chmod(t,0o700);os.replace(t,p);"
        "c=sys.argv[3];e=sys.argv[4];"
        "[(os.makedirs(os.path.dirname(c),mode=0o700,exist_ok=True),"
        "(q:=c+'.'+str(os.getpid())),(g:=open(q,'xb')) ,"
        "g.write(base64.b64decode(e,validate=True)),g.flush(),os.fsync(g.fileno()),"
        "g.close(),os.chmod(q,0o400),os.replace(q,c)) for _ in (0,) if c!='-'];"
        "os.execve(p,[p,*sys.argv[5:]],"
        "{'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/nonexistent',"
        "'LANG':'C.UTF-8','LC_ALL':'C.UTF-8'})"
    )
    target = packet_module.ONSTART_MATERIALIZER
    python = Path("/usr/local/bin/python3")
    contract_target = "-"
    contract_encoded = "-"
    if MECHANICAL_ROOT_CONTRACT is not None and MECHANICAL_WORKER_ROOT is not None:
        target = MECHANICAL_WORKER_ROOT / target.relative_to("/")
        python = MECHANICAL_WORKER_ROOT / python.relative_to("/")
        contract_target = str(
            MECHANICAL_WORKER_ROOT
            / "run/parallax/pg1-native-mechanical-contract.json"
        )
        contract_encoded = base64.b64encode(
            _secure(MECHANICAL_ROOT_CONTRACT, mode=0o400)
        ).decode("ascii")
    command = [str(python), "-c", installer, str(target), encoded,
               contract_target, contract_encoded, *argv[1:]]
    return f"cd / && exec {shlex.join(['/usr/bin/env', '-i', *environment, *command])}"


def worker_create_command(packet: dict[str, Any], capsule: dict[str, Any]) -> list[str]:
    packet_module.validate_packet(packet, capsule)
    inputs = packet["contract"]["frozen_inputs"]
    return [
        str(CONTROL_VAST), "create", "instance", str(packet["contract"]["offer_id"]),
        "--image", inputs["worker_image"], "--disk", str(inputs["worker_disk_gb"]),
        "--label", packet["contract"]["launch_label"], "--ssh", "--direct",
        "--onstart-cmd", source_materializer_onstart(packet, capsule), "--raw",
    ]


def _show_command(instance_id: int | None = None) -> list[str]:
    return [str(CONTROL_VAST), "show", "instances" if instance_id is None else "instance",
            *( [] if instance_id is None else [str(instance_id)]), "--raw"]


def _stop_command(instance_id: int) -> list[str]:
    return [str(CONTROL_VAST), "stop", "instance", str(instance_id), "--raw"]


def _copy_command(instance_id: int, source: Path, target: Path) -> list[str]:
    return [str(CONTROL_VAST), "copy", str(source), f"{instance_id}:{target}", "--raw"]


def production_entry_contract(capsule: dict[str, Any]) -> dict[str, Any]:
    digest, attempt = _capsule_identity(capsule)
    argv = capsule["contract"]["entrypoints"]["production_argv"]
    if MECHANICAL_WORKER_ROOT is not None:
        expected_argv = [
            str(MECHANICAL_WORKER_ROOT / item.lstrip("/"))
            if item.startswith("/") else item for item in argv
        ]
        entry_path = MECHANICAL_WORKER_ROOT / "run/parallax/pg1-native-entry-contract.json"
        bootstrap_root = MECHANICAL_WORKER_ROOT / "opt/parallax/future-freeze/source"
        bootstrap_argv = [
            str(MECHANICAL_WORKER_ROOT / item.lstrip("/"))
            if item.startswith("/") else item
            for item in capsule["contract"]["entrypoints"]["bootstrap_argv"]
        ]
        bootstrap_argv[bootstrap_argv.index("--entry-contract") + 1] = str(entry_path)
        worker_argv = [
            str(bootstrap_root / "infra/vast/parallax/pg1_native_worker.py"),
            "--mechanical-no-model-proof", "--capsule",
            str(MECHANICAL_WORKER_ROOT / "run/parallax/pg1-execution-capsule.json"),
            "--staged-root", str(bootstrap_root),
        ]
        return {
            "schema": "pg1-native-entry-dispatch-contract/v2",
            "kind": "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2",
            "capsule_sha256": digest, "attempt_run_id": attempt,
            "argv_sha256": hashlib.sha256(canonical_bytes(expected_argv)).hexdigest(),
            "argv": expected_argv, "rebase_root": str(MECHANICAL_WORKER_ROOT),
            "image_root": str(MECHANICAL_WORKER_ROOT / "opt/parallax/image/source"),
            "bootstrap_root": str(bootstrap_root),
            "runtime_wiring": capsule["contract"]["runtime_wiring"],
            "bootstrap_argv": bootstrap_argv,
            "receipt_path": str(
                MECHANICAL_WORKER_ROOT / "run/parallax/exact-bootstrap-receipt.json"
            ),
            "worker_argv": worker_argv,
            "worker_receipt_path": str(
                MECHANICAL_WORKER_ROOT / "run/parallax/exact-worker-receipt.json"
            ),
            "dispatch_scope": "EXACT_REBASED_BYTES_ONLY",
            "production_authority": False, "may_mint_authorization": False,
        }
    return {
        "schema": "pg1-native-entry-dispatch-contract/v1",
        "kind": "PRODUCTION_AUTHORITY_BOUND", "capsule_sha256": digest,
        "attempt_run_id": attempt,
        "argv_sha256": hashlib.sha256(canonical_bytes(argv)).hexdigest(),
        "bootstrap_argv_sha256": hashlib.sha256(
            canonical_bytes(capsule["contract"]["entrypoints"]["bootstrap_argv"])
        ).hexdigest(),
        "rebase_root": None, "dispatch_scope": "PRODUCTION_ENTRY_ONLY",
        "may_mint_authorization": False,
    }


def _publish_stop_context(
    capsule: dict[str, Any], packet: dict[str, Any], binding: dict[str, Any],
    instance_key_sha256: str,
) -> dict[str, Any]:
    capsule_sha, attempt = _capsule_identity(capsule)
    inputs = packet["contract"]["frozen_inputs"]
    receipts = inputs["worker_runtime_receipts"]
    context_key = f"{AUTHORIZATION_PREFIX}/deployment-context/{capsule_sha}/{attempt}.json"
    runtime_receipts = {
        "python": {"path": "/usr/local/bin/python3", "sha256": receipts["python_sha256"], "mode": "0755"},
        "aws": {"path": "/usr/local/bin/aws", "sha256": receipts["aws_sha256"], "mode": "0755"},
        "vast": {"path": "/usr/local/bin/vastai", "sha256": receipts["vast_sha256"], "mode": "0755"},
        "aws_config": {"path": "/run/parallax/aws/config",
                       "sha256": inputs["control_mounts"]["aws_config_sha256"], "mode": "0400"},
        "aws_credentials": {"path": "/run/parallax/aws/credentials",
                            "sha256": inputs["control_mounts"]["aws_credentials_sha256"], "mode": "0400"},
        "vast_api_key": {"path": "/run/secrets/pg1-native-vast-api-key",
                         "sha256": instance_key_sha256, "mode": "0400"},
        "runtime_image": {"path": "/run/parallax/runtime-image-digest",
                          "sha256": inputs["runtime_image"]["receipt_sha256"], "mode": "0400"},
    }
    contract = {
        "schema": "pg1-native-production-stop-context/v1",
        "issuer": "PG1_PRODUCTION_DEPLOYMENT_CONTEXT_ISSUER_V1",
        "issuer_account": EXPECTED_BUCKET_OWNER, "capsule_sha256": capsule_sha,
        "attempt_run_id": attempt, "instance_id": binding["id"],
        "offer_id": packet["contract"]["offer_id"], "machine_id": binding["machine_id"],
        "container_id": binding["container_id"],
        "runtime_image": inputs["runtime_image"]["image"],
        "runtime_receipts": runtime_receipts, "context_key": context_key,
    }
    bare = {"schema_version": 1,
            "status": "ISSUED_FOR_ONE_FUTURE_PRODUCTION_INSTANCE",
            "content_address": {"algorithm": "sha256",
                                "digest": hashlib.sha256(canonical_bytes(contract)).hexdigest()},
            "contract": contract}
    issuance = _aws_put(
        key=context_key, value=bare,
        metadata={"capsule-sha256": capsule_sha, "attempt-run-id": attempt,
                  "instance-id": str(binding["id"]), "object-kind": "deployment-context"},
        packet=packet,
    )
    return {**bare, "issuance": {
        "schema": "pg1-native-stop-context-issuance/v1", "key": context_key,
        **issuance,
    }}


def _write_bytes(path: Path, data: bytes, *, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, mode); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_handoff(
    directory: Path, capsule: dict[str, Any], packet: dict[str, Any],
    context: dict[str, Any], instance_key: str, bound_ack: dict[str, Any],
) -> tuple[dict[Path, Path], dict[str, Any]]:
    inputs = packet["contract"]["frozen_inputs"]
    authority_binding = {
        "schema": "pg1-native-worker-authority-binding/v1",
        "packet_sha256": bound_ack["packet_sha256"],
        "capsule_sha256": bound_ack["capsule_sha256"],
        "attempt_run_id": bound_ack["attempt_run_id"],
        "authorization_body_sha256": bound_ack["authorization_body_sha256"],
        "authorization_receipt_sha256": bound_ack["authorization_receipt_sha256"],
        "mount_inventory_sha256": bound_ack["mount_inventory_sha256"],
    }
    sources: dict[Path, tuple[bytes, str]] = {
        Path("/run/parallax/pg1-execution-capsule.json"): (canonical_bytes(capsule), "0400"),
        Path("/run/parallax/pg1-native-stop-context.json"): (canonical_bytes(context), "0400"),
        Path("/run/parallax/aws/config"): (_secure(CONTROL_AWS_CONFIG, mode=0o400,
            sha256=inputs["control_mounts"]["aws_config_sha256"]), "0400"),
        Path("/run/parallax/aws/credentials"): (_secure(CONTROL_AWS_CREDENTIALS, mode=0o400,
            sha256=inputs["control_mounts"]["aws_credentials_sha256"]), "0400"),
        Path("/run/secrets/pg1-native-vast-api-key"): ((instance_key + "\n").encode(), "0400"),
        Path("/run/parallax/runtime-image-digest"): (_secure(CONTROL_RUNTIME_RECEIPT, mode=0o400,
            sha256=inputs["runtime_image"]["receipt_sha256"]), "0400"),
        Path("/run/parallax/pg1-native-entry-contract.json"): (
            canonical_bytes(production_entry_contract(capsule)), "0400"),
        Path("/run/parallax/pg1-external-deadman-arm.json"): (
            canonical_bytes(bound_ack), "0400"),
        Path("/run/parallax/pg1-launch-authority-binding.json"): (
            canonical_bytes(authority_binding), "0400"),
        packet_module.WORKER_PACKET_PATH: (canonical_bytes(packet), "0400"),
    }
    copies: dict[Path, Path] = {}
    mounts: dict[str, Any] = {}
    for index, (target, (data, mode)) in enumerate(sources.items()):
        source = directory / f"handoff-{index}.bin"
        _write_bytes(source, data, mode=int(mode, 8))
        copies[target] = source
        mounts[str(target)] = {"mode": mode, "sha256": hashlib.sha256(data).hexdigest(),
                               "bytes": len(data)}
    layout = {
        "schema": "pg1-native-bound-handoff-layout/v1",
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "instance_id": context["contract"]["instance_id"],
        "source_bundle": inputs["source_bundle"],
        "source_runtime_sha256": packet["contract"]["source_runtime_sha256"],
        "mounts": mounts, "science_released": False,
    }
    layout_source = directory / "handoff-layout.json"; _atomic(layout_source, layout)
    copies[Path("/run/parallax/pg1-native-handoff-layout.json")] = layout_source
    return copies, layout


def _copy_all(instance_id: int, copies: dict[Path, Path], packet: dict[str, Any]) -> None:
    for target, source in copies.items():
        completed = _run(_copy_command(instance_id, source, target), packet=packet)
        if completed.returncode != 0:
            raise LaunchGatewayError("bound handoff copy failed")


def start_commit_body(
    capsule: dict[str, Any], packet: dict[str, Any], state: dict[str, Any],
    context: dict[str, Any], layout: dict[str, Any], bound_ack: dict[str, Any],
) -> dict[str, Any]:
    if state.get("phase") != "START_COMMIT_INTENT":
        raise LaunchGatewayError("start commit requires canonical intent state")
    contract = {
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": packet["contract"]["attempt_run_id"],
        "instance_id": context["contract"]["instance_id"],
        "canonical_state_sequence": state["sequence"],
        "canonical_state_sha256": state["self_sha256"],
        "stop_context_sha256": context["content_address"]["digest"],
        "deadman_ack_sha256": hashlib.sha256(canonical_bytes(bound_ack)).hexdigest(),
        "handoff_layout_sha256": hashlib.sha256(canonical_bytes(layout)).hexdigest(),
        "start_commit_key": state_module.start_commit_key(
            packet["content_address"]["digest"]
        ),
        "science_released": True,
    }
    return {
        "schema": "pg1-native-canonical-start-commit/v1",
        "status": "CANONICAL_START_COMMITTED",
        "content_address": {"algorithm": "sha256", "digest": hashlib.sha256(
            canonical_bytes(contract)
        ).hexdigest()},
        "contract": contract,
    }


def start_commit_metadata(body: dict[str, Any]) -> dict[str, str]:
    contract = body["contract"]
    return {
        "object-kind": "canonical-start-commit",
        "packet-sha256": contract["packet_sha256"],
        "capsule-sha256": contract["capsule_sha256"],
        "attempt-run-id": contract["attempt_run_id"],
        "canonical-state-sha256": contract["canonical_state_sha256"],
    }


def validate_start_commit(
    body: dict[str, Any], *, expected: dict[str, Any]
) -> dict[str, Any]:
    if body != expected:
        raise LaunchGatewayError("canonical start commit content changed")
    return body


def classify_start_commit(
    packet: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    absences = 0
    for attempt in range(START_COMMIT_PROBES):
        observed = _read_object(
            key=expected["contract"]["start_commit_key"], packet=packet,
            expected_metadata=start_commit_metadata(expected),
        )
        if observed.get("status") == "PRESENT":
            try:
                validate_start_commit(observed["body"], expected=expected)
            except LaunchGatewayError:
                return {"classification": "SCIENCE_RELEASE_UNKNOWN", "observed": observed}
            return {"classification": "SCIENCE_RELEASED", "observed": observed}
        if observed.get("status") == "ABSENT":
            absences += 1
        else:
            return {"classification": "SCIENCE_RELEASE_UNKNOWN", "observed": observed}
        if attempt + 1 < START_COMMIT_PROBES:
            time.sleep(attempt + 1)
    if absences == START_COMMIT_PROBES:
        return {"classification": "START_COMMIT_PROVEN_ABSENT",
                "consecutive_absences": absences}
    return {"classification": "SCIENCE_RELEASE_UNKNOWN"}


def _discover(packet: dict[str, Any]) -> int | None:
    completed = _run(_show_command(), packet=packet)
    if completed.returncode != 0:
        return None
    try:
        found = discover_created_by_label(
            json.loads(completed.stdout), launch_label=packet["contract"]["launch_label"]
        )
    except (json.JSONDecodeError, VastAdapterError) as error:
        raise LaunchGatewayError("orphan label discovery ambiguous or malformed") from error
    return None if found is None else int(found["instance_id"])


def _cleanup_key(packet: dict[str, Any]) -> str:
    return (
        f"{AUTHORIZATION_PREFIX}/gateway-cleanup/"
        f"{packet['content_address']['digest']}.json"
    )


def _cleanup_metadata(receipt: dict[str, Any]) -> dict[str, str]:
    return {
        "packet-sha256": receipt["packet_sha256"],
        "instance-id": str(receipt["instance_id"]),
        "object-kind": "gateway-orphan-cleanup",
    }


def _validate_cleanup_receipt(
    receipt: dict[str, Any], storage: dict[str, Any], packet: dict[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema", "status", "packet_sha256", "instance_id", "failure",
        "stop_attempts", "errors", "consecutive_terminal_observations",
        "science_released", "science_state", "destroy_gate",
        "start_commit_classification",
    }
    raw = canonical_bytes(receipt)
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != "pg1-native-gateway-orphan-cleanup/v2"
        or receipt.get("packet_sha256") != packet["content_address"]["digest"]
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
        or storage.get("status") != "PRESENT"
        or storage.get("body") != receipt
        or storage.get("raw_bytes") != raw
        or storage.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
        or storage.get("metadata") != _cleanup_metadata(receipt)
        or not all(
            isinstance(storage.get(field), str) and bool(storage[field])
            for field in ("version_id", "etag", "checksum_sha256")
        )
    ):
        raise LaunchGatewayError("immutable cleanup receipt identity changed")
    if receipt["status"] == "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED":
        if (
            receipt["science_state"] != state_module.SCIENCE_DISABLED
            or receipt["science_released"] is not False
            or receipt["destroy_gate"] != "PRE_SCIENCE_ZERO_BODY_DURABILITY"
            or receipt["start_commit_classification"]
            != "START_COMMIT_PROVEN_ABSENT"
            or receipt["consecutive_terminal_observations"]
            == TERMINAL_OBSERVATIONS
        ):
            raise LaunchGatewayError("pre-science cleanup receipt gate changed")
    return receipt


def _build_pre_science_destroy_request(
    receipt: dict[str, Any], cleanup_storage: dict[str, Any],
    state: dict[str, Any], packet: dict[str, Any],
) -> dict[str, Any]:
    if (
        state["science_state"] != state_module.SCIENCE_DISABLED
        or state.get("instance_id") is not None
        and receipt["instance_id"] != state.get("instance_id")
        or receipt["status"]
        != "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED"
    ):
        raise LaunchGatewayError("cleanup receipt cannot authorize this state")
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
        "capsule_sha256": packet["contract"]["capsule_sha256"],
        "attempt_run_id": packet["contract"]["attempt_run_id"],
        "teardown_authorized": True,
        "evidence_sha256": hashlib.sha256(
            canonical_bytes(zero_body_evidence)
        ).hexdigest(),
        "pre_science_zero_body_evidence": zero_body_evidence,
        "cleanup_receipt": {
            "key": _cleanup_key(packet),
            "sha256": hashlib.sha256(canonical_bytes(receipt)).hexdigest(),
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
        "consecutive_absences": START_COMMIT_PROBES,
    }


def _cas_cleanup_destroying(
    packet: dict[str, Any], state: dict[str, Any], storage: dict[str, Any],
    receipt: dict[str, Any], cleanup_storage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Converge on the one exact state edge that contains destroy authority."""

    for _attempt in range(state_module.MAX_ANCESTORS):
        if state["phase"] in state_module.TERMINAL_PHASES:
            return state, storage
        if state["phase"] == "CLEANUP_DESTROYING":
            request = state.get("destroy_request_body")
            if (
                not isinstance(request, dict)
                or state.get("destroy_request") != {
                    "sha256": hashlib.sha256(canonical_bytes(request)).hexdigest(),
                    "storage": "CANONICAL_STATE_RAW_BODY",
                }
            ):
                raise LaunchGatewayError("canonical destroy authority body changed")
            return state, storage
        request = _build_pre_science_destroy_request(
            receipt, cleanup_storage, state, packet
        )
        candidate = state_module.transition(
            state, "CLEANUP_DESTROYING",
            science_state=state_module.SCIENCE_DISABLED,
            instance_id=receipt["instance_id"],
            start_commit_classification="START_COMMIT_PROVEN_ABSENT",
            destroy_request={
                "sha256": hashlib.sha256(canonical_bytes(request)).hexdigest(),
                "storage": "CANONICAL_STATE_RAW_BODY",
            },
            destroy_request_body=request,
            destroy_gate="PRE_SCIENCE_ZERO_BODY_DURABILITY",
        )
        try:
            return _publish_state(
                packet, candidate, prior_etag=storage["etag"], prior_state=state,
            )
        except LaunchGatewayError as conflict:
            observed = _read_object(
                key=state_module.state_key(state["packet_sha256"]), packet=packet,
            )
            winner, winner_storage = _verify_stored_state_observation(
                observed, packet=packet
            )
            if not state_module.proves_descendant(winner, state):
                raise LaunchGatewayError(
                    "cleanup recovery CAS winner is not a canonical descendant"
                ) from conflict
            state, storage = winner, winner_storage
    raise LaunchGatewayError("cleanup recovery exhausted canonical ancestry")


def _start_recovery_deadman(
    packet: dict[str, Any], state: dict[str, Any], storage: dict[str, Any],
) -> None:
    _cache_state(state, storage)
    subprocess.Popen(
        control_deadman_argv(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, cwd=str(CONTROL_ROOT), env=fixed_env(packet),
        start_new_session=True, close_fds=True, text=True,
    )


def recover_pre_science_cleanup(
    packet: dict[str, Any],
) -> dict[str, Any] | None:
    """Production restart entry: recover cleanup before authority consumption."""

    observed = _read_object(
        key=state_module.state_key(packet["content_address"]["digest"]), packet=packet,
    )
    if observed.get("status") == "ABSENT":
        return None
    state, storage = _verify_stored_state_observation(observed, packet=packet)
    cleanup_storage = _read_object(key=_cleanup_key(packet), packet=packet)
    if cleanup_storage.get("status") == "PRESENT":
        receipt = _validate_cleanup_receipt(
            cleanup_storage["body"], cleanup_storage, packet
        )
        if receipt["status"] == "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED":
            state, storage = _cas_cleanup_destroying(
                packet, state, storage, receipt, cleanup_storage
            )
    elif cleanup_storage.get("status") not in {"ABSENT"}:
        raise LaunchGatewayError("cleanup recovery receipt existence is inconclusive")
    _start_recovery_deadman(packet, state, storage)
    return {
        "status": "RESTART_RESUMED_CANONICAL_STATE",
        "packet_sha256": state["packet_sha256"],
        "canonical_state_sha256": state["self_sha256"],
        "canonical_state_sequence": state["sequence"],
        "canonical_phase": state["phase"],
        "science_released": state["science_state"] == state_module.SCIENCE_RELEASED,
    }


def _cleanup(
    instance_id: int, packet: dict[str, Any], *, failure: str,
    science_released: bool = False, science_state: str | None = None,
    canonical_state: dict[str, Any] | None = None,
    start_commit_classification: str | None = None,
) -> dict[str, Any]:
    effective_science = science_state or (
        state_module.SCIENCE_RELEASED if science_released
        else state_module.SCIENCE_DISABLED
    )
    released_or_unknown = effective_science != state_module.SCIENCE_DISABLED
    observations = 0
    attempts = 0
    errors = 0
    while attempts < 6 and observations < TERMINAL_OBSERVATIONS:
        attempts += 1
        try:
            _run(_stop_command(instance_id), packet=packet)
            shown = _run(_show_command(instance_id), packet=packet)
            row = select_instance(json.loads(shown.stdout), instance_id)
            state = None if row is None else str(row.get("actual_status") or row.get("cur_state", "")).lower()
            observations = observations + 1 if state in {"stopped", "exited", "offline"} else 0
        except Exception:
            errors += 1; observations = 0
        time.sleep(min(5, attempts))
    if observations == TERMINAL_OBSERVATIONS:
        status = (
            "RELEASED_WORKER_STOPPED_SCRATCH_PRESERVED"
            if released_or_unknown else "ORPHAN_STOPPED_SCIENCE_NEVER_RELEASED"
        )
    else:
        status = (
            "RELEASED_WORKER_STOP_UNCONFIRMED_SCRATCH_PRESERVED"
            if released_or_unknown else "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED"
        )
    receipt = {
        "schema": "pg1-native-gateway-orphan-cleanup/v2", "status": status,
        "packet_sha256": packet["content_address"]["digest"],
        "instance_id": instance_id, "failure": failure,
        "stop_attempts": attempts, "errors": errors,
        "consecutive_terminal_observations": observations,
        "science_released": effective_science == state_module.SCIENCE_RELEASED,
        "science_state": effective_science,
        "destroy_gate": (
            None if released_or_unknown else "PRE_SCIENCE_ZERO_BODY_DURABILITY"
        ),
        "start_commit_classification": start_commit_classification,
    }
    cleanup_key = _cleanup_key(packet)
    cleanup_storage = _put_immutable_verified(
        key=cleanup_key,
        value=receipt,
        metadata=_cleanup_metadata(receipt),
        packet=packet,
    )
    if (
        observations != TERMINAL_OBSERVATIONS
        and effective_science == state_module.SCIENCE_DISABLED
        and canonical_state is not None
        and start_commit_classification == "START_COMMIT_PROVEN_ABSENT"
    ):
        request = _build_pre_science_destroy_request(
            receipt, cleanup_storage, canonical_state, packet
        )
        receipt["destroy_request_body"] = request
        receipt["cleanup_storage"] = cleanup_storage
    return receipt


def launch(capsule: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    packet = packet_module.validate_packet(packet, capsule)
    recovered = recover_pre_science_cleanup(packet)
    if recovered is not None:
        return recovered
    authorization, authorization_storage = fetch_launch_authorization(capsule, packet)
    consume_launch_authorization(authorization, authorization_storage, packet)
    created_instance: int | None = None
    science_state = state_module.SCIENCE_DISABLED
    start_classification: str | None = None
    expected_start: dict[str, Any] | None = None
    state, state_receipt = _publish_state(
        packet,
        state_module.initial(
            packet, authorization_sha256=authorization["content_address"]["digest"],
            authorization_receipt_sha256=authority_storage_digest(
                authorization, authorization_storage
            ),
        ),
        prior_etag=None,
    )
    subprocess.Popen(
        control_deadman_argv(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, cwd=str(CONTROL_ROOT), env=fixed_env(packet),
        start_new_session=True, close_fds=True, text=True,
    )
    try:
        _wait_ack(ARM_ACK, packet, status="PRELAUNCH_CLEANUP_ARMED")
        state, state_receipt = _advance_state(
            packet, state, state_receipt, "CREATE_INTENT_ARMED"
        )
        completed = _run(worker_create_command(packet, capsule), packet=packet)
        if completed.returncode != 0:
            raise LaunchGatewayError("worker create returned nonzero")
        try:
            create = parse_create_response(
                json.loads(completed.stdout), offer_id=packet["contract"]["offer_id"]
            )
        except (json.JSONDecodeError, VastAdapterError) as error:
            raise LaunchGatewayError("worker create response malformed or unbound") from error
        created_instance = int(create["instance_id"])
        instance_key = str(create["instance_api_key"])
        instance_key_sha = hashlib.sha256((instance_key + "\n").encode()).hexdigest()
        state, state_receipt = _advance_state(
            packet, state, state_receipt, "WORKER_ID_BOUND_SCIENCE_DISABLED",
            instance_id=created_instance, instance_api_key_sha256=instance_key_sha,
        )
        shown = _run(_show_command(created_instance), packet=packet)
        try:
            binding = validate_created_binding(
                json.loads(shown.stdout), instance_id=created_instance,
                launch_label=packet["contract"]["launch_label"],
            )
        except (json.JSONDecodeError, VastAdapterError) as error:
            raise LaunchGatewayError("created worker binding row changed") from error
        context = _publish_stop_context(
            capsule, packet, binding, instance_key_sha
        )
        state, state_receipt = _advance_state(
            packet, state, state_receipt, "INSTANCE_CONTEXT_BOUND",
            machine_id=binding["machine_id"], container_id=binding["container_id"],
            stop_context_sha256=context["content_address"]["digest"],
            mount_inventory_sha256=prebound_mount_inventory_digest(
                capsule, packet, context, instance_key_sha
            ),
        )
        bound_ack = _wait_ack(
            BOUND_ACK, packet, status="INSTANCE_BOUND_DEADMAN_ARMED"
        )
        with tempfile.TemporaryDirectory(prefix="pg1-bound-handoff-") as directory:
            copies, layout = _materialize_handoff(
                Path(directory), capsule, packet, context, instance_key, bound_ack
            )
            _copy_all(created_instance, copies, packet)
            layout_sha = hashlib.sha256(canonical_bytes(layout)).hexdigest()
            state, state_receipt = _advance_state(
                packet, state, state_receipt, "HANDOFF_BOUND_DEADMAN_ACKED",
                handoff_layout_sha256=layout_sha,
            )
            state, state_receipt = _advance_state(
                packet, state, state_receipt, "START_COMMIT_INTENT",
            )
            expected_start = start_commit_body(
                capsule, packet, state, context, layout, bound_ack
            )
            start_intent = {
                "schema": "pg1-native-bound-start-intent/v1",
                "status": "SCIENCE_DISABLED_WAITING_FOR_CANONICAL_START_COMMIT",
                "packet_sha256": packet["content_address"]["digest"],
                "capsule_sha256": capsule["content_address"]["digest"],
                "attempt_run_id": packet["contract"]["attempt_run_id"],
                "instance_id": created_instance,
                "stop_context_sha256": context["content_address"]["digest"],
                "deadman_ack_sha256": hashlib.sha256(canonical_bytes(bound_ack)).hexdigest(),
                "handoff_layout_sha256": layout_sha,
                "canonical_state_sequence": state["sequence"],
                "canonical_state_sha256": state["self_sha256"],
                "start_commit_key": expected_start["contract"]["start_commit_key"],
                "start_commit_sha256": expected_start["content_address"]["digest"],
                "not_after_unix": time.time() + packet["contract"]["frozen_inputs"][
                    "maximum_prestart_seconds"
                ],
                "science_released": False,
            }
            start_source = Path(directory) / "start-intent.json"
            _atomic(start_source, start_intent)
            copied = _run(_copy_command(created_instance, start_source, START_LEASE_REMOTE),
                          packet=packet)
            if copied.returncode != 0:
                raise LaunchGatewayError("science-disabled start intent handoff failed")
            commit_put_verified = False
            try:
                _aws_put(
                    key=expected_start["contract"]["start_commit_key"],
                    value=expected_start, metadata=start_commit_metadata(expected_start),
                    packet=packet,
                )
                commit_put_verified = True
            except Exception:
                pass
            classification = classify_start_commit(packet, expected_start)
            start_classification = str(classification["classification"])
            if commit_put_verified and start_classification != "SCIENCE_RELEASED":
                start_classification = "SCIENCE_RELEASE_UNKNOWN"
            if start_classification == "SCIENCE_RELEASED":
                science_state = state_module.SCIENCE_RELEASED
                state, state_receipt = _advance_state(
                    packet, state, state_receipt, "SCIENCE_RELEASED",
                    science_state=science_state,
                    start_commit_sha256=expected_start["content_address"]["digest"],
                )
            elif start_classification == "START_COMMIT_PROVEN_ABSENT":
                state, state_receipt = _advance_state(
                    packet, state, state_receipt, "START_COMMIT_PROVEN_ABSENT"
                )
                raise LaunchGatewayError("canonical start commit proven absent")
            else:
                science_state = state_module.SCIENCE_RELEASE_UNKNOWN
                state, state_receipt = _advance_state(
                    packet, state, state_receipt, "SCIENCE_RELEASE_UNKNOWN",
                    science_state=science_state,
                )
                raise LaunchGatewayError("canonical start release is unknown")
        return {"status": "WORKER_BOUND_DEADMAN_ACKED_START_RELEASED",
                "instance_id": created_instance,
                "packet_sha256": packet["content_address"]["digest"],
                "canonical_state_sha256": state["self_sha256"],
                "start_commit_sha256": expected_start["content_address"]["digest"],
                "science_released": True}
    except BaseException as error:
        if expected_start is not None and science_state == state_module.SCIENCE_DISABLED:
            classification = classify_start_commit(packet, expected_start)
            start_classification = str(classification["classification"])
            if start_classification == "SCIENCE_RELEASED":
                science_state = state_module.SCIENCE_RELEASED
            elif start_classification != "START_COMMIT_PROVEN_ABSENT":
                science_state = state_module.SCIENCE_RELEASE_UNKNOWN
        elif start_classification is None:
            absences = 0
            for attempt in range(START_COMMIT_PROBES):
                observed = _read_object(
                    key=state_module.start_commit_key(packet["content_address"]["digest"]),
                    packet=packet,
                )
                if observed.get("status") == "ABSENT":
                    absences += 1
                else:
                    science_state = state_module.SCIENCE_RELEASE_UNKNOWN
                    break
                if attempt + 1 < START_COMMIT_PROBES:
                    time.sleep(attempt + 1)
            start_classification = (
                "START_COMMIT_PROVEN_ABSENT" if absences == START_COMMIT_PROBES
                else "SCIENCE_RELEASE_UNKNOWN"
            )
        if created_instance is None:
            created_instance = _discover(packet)
        if created_instance is not None:
            cleanup = _cleanup(
                created_instance, packet, failure=type(error).__name__,
                science_state=science_state, canonical_state=state,
                start_commit_classification=start_classification,
            )
            destroy_body = cleanup.get("destroy_request_body")
            if (
                isinstance(destroy_body, dict)
                and state["science_state"] == state_module.SCIENCE_DISABLED
            ):
                state, state_receipt = _cas_cleanup_destroying(
                    packet, state, state_receipt,
                    cleanup["cleanup_storage"]["body"],
                    cleanup["cleanup_storage"],
                )
            elif state["phase"] not in {
                "CLEANUP_STOPPING", "CLEANUP_DESTROYING", *state_module.TERMINAL_PHASES
            }:
                state, state_receipt = _advance_state(
                    packet, state, state_receipt, "CLEANUP_STOPPING",
                    science_state=science_state, instance_id=created_instance,
                    start_commit_classification=start_classification,
                )
        raise


def configure_from_source_location() -> dict[str, Any] | None:
    """Derive any mechanical root from these resolved copied source bytes."""

    global CONTROL_ROOT, CONTROL_VAST, CONTROL_AWS, CONTROL_DEADMAN
    global CONTROL_VAST_KEY, CONTROL_AWS_CONFIG, CONTROL_AWS_CREDENTIALS
    global CONTROL_RUNTIME_RECEIPT, ARM_ACK, BOUND_ACK, TRANSACTION, CAPSULE_PATH
    global PACKET_PATH, STATE_RECEIPT, MECHANICAL_DEADMAN_ARGV
    global MECHANICAL_ROOT_CONTRACT, MECHANICAL_WORKER_ROOT
    discovered = mechanical_rebase.control_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    contract, contract_path = discovered
    root = contract["root"]
    def at(path: str) -> Path:
        return mechanical_rebase.rebased(root, Path(path))
    CONTROL_ROOT = at("/opt/parallax/control/source")
    CONTROL_VAST = at("/opt/parallax/control/bin/vastai")
    CONTROL_AWS = at("/opt/parallax/control/bin/aws")
    CONTROL_DEADMAN = CONTROL_ROOT / "infra/vast/parallax/pg1_native_control_deadman.py"
    CONTROL_VAST_KEY = at("/run/secrets/pg1-native-control-plane-vast-api-key")
    CONTROL_AWS_CONFIG = at("/run/parallax/control-plane/aws/config")
    CONTROL_AWS_CREDENTIALS = at("/run/parallax/control-plane/aws/credentials")
    CONTROL_RUNTIME_RECEIPT = at("/run/parallax/control-plane/runtime-image-digest")
    ARM_ACK = at("/run/parallax/control-plane/pg1-deadman-arm-ack.json")
    BOUND_ACK = at("/run/parallax/control-plane/pg1-deadman-bound-ack.json")
    TRANSACTION = at("/run/parallax/control-plane/pg1-worker-launch-transaction.json")
    CAPSULE_PATH = at("/run/parallax/control-plane/pg1-execution-capsule.json")
    PACKET_PATH = at("/run/parallax/control-plane/pg1-future-launch-packet.json")
    STATE_RECEIPT = at("/run/parallax/control-plane/pg1-canonical-state-receipt.json")
    MECHANICAL_DEADMAN_ARGV = [
        str(CONTROL_DEADMAN), "--control-deadman", "--capsule", str(CAPSULE_PATH),
        "--launch-transaction", str(TRANSACTION),
    ]
    MECHANICAL_ROOT_CONTRACT = contract_path.resolve()
    MECHANICAL_WORKER_ROOT = root / ".pg1-fake-services/worker"
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--packet", type=Path)
    args = parser.parse_args(argv)
    configure_from_source_location()
    if (
        not args.launch or args.capsule != CAPSULE_PATH or args.packet != PACKET_PATH
    ):
        raise LaunchGatewayError("control-plane gateway argv changed")
    result = launch(_load_fixed(args.capsule), _load_fixed(args.packet))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
