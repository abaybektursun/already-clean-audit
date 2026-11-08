#!/usr/local/bin/python3
"""Tracked fixed-layout provisioner and actual PG1 future entrypoint.

The future instance image invokes this module.  It creates the fixed directory
layout, verifies mounted secrets/context/receipts, stages the capsule-bound
source inventory from the immutable image root, writes an exact layout receipt,
and starts only the capsule's fixed bootstrap argv.  It has no provider API,
market, price, rehearsal, or caller-selected path interface.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from . import pg1_native_lifecycle as lifecycle
    from . import pg1_native_mechanical_rebase as mechanical_rebase
except ImportError:  # direct absolute production script
    import pg1_native_lifecycle as lifecycle  # type: ignore[no-redef]
    import pg1_native_mechanical_rebase as mechanical_rebase  # type: ignore[no-redef]


PYTHON_EXECUTABLE = Path("/usr/local/bin/python3")
AWS_EXECUTABLE = Path("/usr/local/bin/aws")
CAPSULE_RUNTIME_PATH = Path("/run/parallax/pg1-execution-capsule.json")
STOP_CONTEXT_RUNTIME_PATH = Path("/run/parallax/pg1-native-stop-context.json")
ENTRY_CONTRACT_RUNTIME_PATH = Path("/run/parallax/pg1-native-entry-contract.json")
START_LEASE_RUNTIME_PATH = Path("/run/parallax/pg1-native-start-lease.json")
HANDOFF_LAYOUT_RUNTIME_PATH = Path("/run/parallax/pg1-native-handoff-layout.json")
PACKET_RUNTIME_PATH = Path("/run/parallax/pg1-future-launch-packet.json")
AUTHORITY_BINDING_RUNTIME_PATH = Path(
    "/run/parallax/pg1-launch-authority-binding.json"
)
INSTALLED_SOURCE_RUNTIME_PATH = Path("/run/parallax/pg1-native-installed-source-layout.json")
IMMUTABLE_IMAGE_SOURCE_ROOT = Path("/opt/parallax/image/source")
BOOTSTRAP_SOURCE_ROOT = Path("/opt/parallax/future-freeze/source")
LAYOUT_RECEIPT_PATH = Path("/run/parallax/pg1-native-deployment-layout.json")
LIFECYCLE_PATH = Path("/workspace/parallax/failure-stop-pointer.json")
GUARDIAN_RESULT_PATH = Path("/workspace/parallax/pg1-native-guardian-result.json")
EXTERNAL_ARM_RECEIPT_PATH = Path("/run/parallax/pg1-external-deadman-arm.json")
LEASE_CAS_RECEIPT_PATH = Path("/run/parallax/pg1-lifecycle-lease-cas.json")
GUARDIAN_IMAGE_PATH = (
    IMMUTABLE_IMAGE_SOURCE_ROOT / "infra/vast/parallax/pg1_native_guardian.py"
)
AUTHORIZATION_BUCKET = "pg-lineage-769063704696"
AUTHORIZATION_PREFIX = "parallax/authorizations/pg1-native-v3"
EXPECTED_BUCKET_OWNER = "769063704696"
PROVIDER_TIMEOUT_SECONDS = 30
GUARDIAN_TIMEOUT_SECONDS = 10_800
BOOTSTRAP_TIMEOUT_SECONDS = 10_800 + 600
FIXED_DIRECTORIES = {
    Path("/opt/parallax/future-freeze"): 0o700,
    Path("/run/parallax"): 0o700,
    Path("/run/parallax/aws"): 0o700,
    Path("/run/secrets"): 0o700,
    Path("/workspace/parallax"): 0o700,
}
FIXED_MOUNTS = {
    CAPSULE_RUNTIME_PATH: 0o400,
    STOP_CONTEXT_RUNTIME_PATH: 0o400,
    Path("/run/parallax/aws/config"): 0o400,
    Path("/run/parallax/aws/credentials"): 0o400,
    Path("/run/secrets/pg1-native-vast-api-key"): 0o400,
    Path("/run/parallax/runtime-image-digest"): 0o400,
    EXTERNAL_ARM_RECEIPT_PATH: 0o400,
    ENTRY_CONTRACT_RUNTIME_PATH: 0o400,
    START_LEASE_RUNTIME_PATH: 0o400,
    HANDOFF_LAYOUT_RUNTIME_PATH: 0o400,
    PACKET_RUNTIME_PATH: 0o400,
    INSTALLED_SOURCE_RUNTIME_PATH: 0o400,
}


class DeploymentError(RuntimeError):
    pass


def _fixed_env(*, source_root: Path = IMMUTABLE_IMAGE_SOURCE_ROOT) -> dict[str, str]:
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
        "PYTHONPATH": str(source_root),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_json_path(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o400); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DeploymentError("deployment inventory contains an unsafe path")
    return path


def load_capsule_minimal() -> dict[str, Any]:
    if (
        CAPSULE_RUNTIME_PATH.is_symlink()
        or not CAPSULE_RUNTIME_PATH.is_file()
        or stat.S_IMODE(CAPSULE_RUNTIME_PATH.stat().st_mode) != 0o400
    ):
        raise DeploymentError("fixed deployment capsule mount is absent")
    try:
        capsule = json.loads(CAPSULE_RUNTIME_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError("fixed deployment capsule is unreadable") from error
    contract = capsule.get("contract") if isinstance(capsule, dict) else None
    if (
        not isinstance(contract, dict)
        or capsule.get("content_address")
        != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(_canonical_bytes(contract)).hexdigest(),
        }
    ):
        raise DeploymentError("deployment capsule content address changed")
    return capsule


def load_context_minimal() -> dict[str, Any]:
    path = STOP_CONTEXT_RUNTIME_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o400
    ):
        raise DeploymentError("fixed deployment stop context is absent")
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError("fixed deployment stop context is unreadable") from error
    contract = context.get("contract") if isinstance(context, dict) else None
    if (
        not isinstance(contract, dict)
        or context.get("content_address")
        != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(_canonical_bytes(contract)).hexdigest(),
        }
        or contract.get("issuer_account") != EXPECTED_BUCKET_OWNER
        or isinstance(contract.get("instance_id"), bool)
        or not isinstance(contract.get("instance_id"), int)
        or contract["instance_id"] <= 0
        or not isinstance(contract.get("capsule_sha256"), str)
        or len(contract["capsule_sha256"]) != 64
        or not isinstance(contract.get("attempt_run_id"), str)
        or not contract["attempt_run_id"]
    ):
        raise DeploymentError("fixed deployment stop context identity changed")
    return context


def _read_lifecycle(context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return lifecycle.read(LIFECYCLE_PATH, context)
    except lifecycle.LifecycleCASFailure as error:
        raise DeploymentError(str(error)) from error


def _transition_lifecycle(
    context: dict[str, Any],
    state: str,
    *,
    reason: str,
    terminal_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically advance the attempt-bound monotonic lifecycle state."""

    for _attempt in range(8):
        prior = _read_lifecycle(context)
        expected = 0 if prior is None else int(prior["sequence"])
        try:
            return lifecycle.transition(
                LIFECYCLE_PATH,
                context,
                state,
                reason=reason,
                expected_sequence=expected,
                terminal_receipt=terminal_receipt,
            )
        except lifecycle.LifecycleCASFailure as error:
            if "sequence mismatch" not in str(error):
                raise DeploymentError(str(error)) from error
    raise DeploymentError("local lifecycle CAS retry budget exhausted")


def bootstrap_command(capsule: dict[str, Any]) -> list[str]:
    expected = [
        str(BOOTSTRAP_SOURCE_ROOT / "infra/vast/parallax/pg1_native_bootstrap.py"),
        "--deploy", "--capsule", str(CAPSULE_RUNTIME_PATH),
        "--stop-context", str(STOP_CONTEXT_RUNTIME_PATH),
        "--entry-contract", str(ENTRY_CONTRACT_RUNTIME_PATH),
    ]
    if capsule["contract"]["entrypoints"].get("bootstrap_argv") != expected:
        raise DeploymentError("capsule bootstrap argv changed")
    return expected


def _verify_external_deadman_arm(
    capsule: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    if (
        EXTERNAL_ARM_RECEIPT_PATH.is_symlink()
        or not EXTERNAL_ARM_RECEIPT_PATH.is_file()
        or stat.S_IMODE(EXTERNAL_ARM_RECEIPT_PATH.stat().st_mode) != 0o400
    ):
        raise DeploymentError("external control deadman arm receipt is absent")
    try:
        value = json.loads(EXTERNAL_ARM_RECEIPT_PATH.read_text(encoding="utf-8"))
        packet = json.loads(PACKET_RUNTIME_PATH.read_text(encoding="utf-8"))
        authority = json.loads(
            AUTHORITY_BINDING_RUNTIME_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError("external control deadman arm receipt unreadable") from error
    attempt = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    payload_fields = {
        "schema", "status", "packet_sha256", "issuer", "capsule_sha256",
        "attempt_run_id", "instance_id", "offer_id", "machine_id", "container_id",
        "stop_context_sha256", "authorization_body_sha256",
        "authorization_receipt_sha256", "mount_inventory_sha256", "ack_id",
        "ack_nonce", "replay_envelope", "destroy_authority",
    }
    fields = payload_fields | {
        "ack_body_sha256", "etag", "version_id", "checksum_sha256",
    }
    payload = {key: value[key] for key in payload_fields} if (
        isinstance(value, dict) and set(value) == fields
    ) else {}
    packet_contract = packet.get("contract") if isinstance(packet, dict) else None
    expected_nonce = hashlib.sha256(_canonical_bytes({
        "scope": "PG1_INSTANCE_BOUND_DEADMAN_ACK_NONCE_V1",
        "packet_sha256": packet.get("content_address", {}).get("digest"),
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": attempt,
        "instance_id": context["contract"]["instance_id"],
        "stop_context_sha256": context["content_address"]["digest"],
    })).hexdigest()
    expected_id = hashlib.sha256(
        f"PG1_BOUND_ACK_V3:{expected_nonce}".encode()
    ).hexdigest()
    expected_checksum = base64.b64encode(
        hashlib.sha256(_canonical_bytes(payload)).digest()
    ).decode("ascii") if payload else None
    if (
        not payload
        or value.get("schema") != "pg1-native-instance-bound-deadman-ack/v3"
        or value.get("status") != "INSTANCE_BOUND_DEADMAN_ARMED"
        or value.get("issuer") != "PG1_CANONICAL_EXTERNAL_CONTROL_DEADMAN_V1"
        or value.get("capsule_sha256") != capsule["content_address"]["digest"]
        or value.get("attempt_run_id") != attempt
        or value.get("instance_id") != context["contract"]["instance_id"]
        or value.get("offer_id") != context["contract"]["offer_id"]
        or value.get("machine_id") != context["contract"]["machine_id"]
        or value.get("container_id") != context["contract"]["container_id"]
        or value.get("stop_context_sha256") != context["content_address"]["digest"]
        or not isinstance(packet_contract, dict)
        or packet.get("content_address") != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(_canonical_bytes(packet_contract)).hexdigest(),
        }
        or value.get("packet_sha256") != packet["content_address"]["digest"]
        or authority != {
            "schema": "pg1-native-worker-authority-binding/v1",
            "packet_sha256": value.get("packet_sha256"),
            "capsule_sha256": value.get("capsule_sha256"),
            "attempt_run_id": value.get("attempt_run_id"),
            "authorization_body_sha256": value.get("authorization_body_sha256"),
            "authorization_receipt_sha256": value.get("authorization_receipt_sha256"),
            "mount_inventory_sha256": value.get("mount_inventory_sha256"),
        }
        or value.get("ack_nonce") != expected_nonce or value.get("ack_id") != expected_id
        or value.get("replay_envelope") != {
            "schema": "pg1-native-bound-ack-exact-replay/v1",
            "ack_id": expected_id, "one_shot": True, "exact_replay_only": True,
        }
        or value.get("ack_body_sha256") != hashlib.sha256(
            _canonical_bytes(payload)
        ).hexdigest()
        or value.get("checksum_sha256") != expected_checksum
        or not all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in ("etag", "version_id", "checksum_sha256")
        )
        or value.get("destroy_authority") is not False
        or context["contract"]["capsule_sha256"] != value["capsule_sha256"]
    ):
        raise DeploymentError("external control deadman arm identity changed")
    return value


def _lease_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/external-deadman-leases/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def _read_durable_lease(
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = context["contract"]
    with tempfile.TemporaryDirectory(prefix="pg1-deadman-lease-reconcile-") as directory:
        target = Path(directory) / "lease.json"
        command = [
            str(AWS_EXECUTABLE), "--profile", "fuelos", "--region", "us-east-1",
            "s3api", "get-object", "--bucket", AUTHORIZATION_BUCKET,
            "--key", _lease_key(context), "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER, str(target),
        ]
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True,
                timeout=PROVIDER_TIMEOUT_SECONDS, cwd=str(IMMUTABLE_IMAGE_SOURCE_ROOT),
                env=_fixed_env(),
            )
        except subprocess.TimeoutExpired as error:
            raise DeploymentError("durable lease reconciliation timed out") from error
        if completed.returncode != 0:
            raise DeploymentError("durable lease reconciliation read failed")
        try:
            response = json.loads(completed.stdout or "")
            data = target.read_bytes()
            payload = json.loads(data)
        except (OSError, json.JSONDecodeError) as error:
            raise DeploymentError("durable lease reconciliation response malformed") from error
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    metadata = response.get("Metadata") if isinstance(response, dict) else None
    if (
        not isinstance(payload, dict)
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("VersionId")
        or not response.get("ETag")
        or metadata
        != {
            "capsule-sha256": contract["capsule_sha256"],
            "attempt-run-id": contract["attempt_run_id"],
            "instance-id": str(contract["instance_id"]),
            "object-kind": "external-control-deadman-lease",
        }
    ):
        raise DeploymentError("durable lease reconciliation identity changed")
    return payload, response


def _reconcile_lease_conflict(
    context: dict[str, Any],
    local_lifecycle: dict[str, Any],
    durable_payload: dict[str, Any],
) -> dict[str, Any]:
    contract = context["contract"]
    durable_lifecycle = durable_payload.get("local_lifecycle")
    if (
        durable_payload.get("schema") != "pg1-native-external-deadman-lease/v2"
        or durable_payload.get("capsule_sha256") != contract["capsule_sha256"]
        or durable_payload.get("attempt_run_id") != contract["attempt_run_id"]
        or durable_payload.get("instance_id") != contract["instance_id"]
        or durable_payload.get("stop_context_sha256")
        != context["content_address"]["digest"]
        or isinstance(durable_payload.get("sequence"), bool)
        or not isinstance(durable_payload.get("sequence"), int)
        or durable_payload["sequence"] < 2
        or not isinstance(durable_lifecycle, dict)
    ):
        raise DeploymentError("durable lease conflict identity changed")
    try:
        lifecycle.validate(durable_lifecycle, context)
        return lifecycle.reconcile(local_lifecycle, durable_lifecycle)
    except lifecycle.LifecycleCASFailure as error:
        raise DeploymentError(str(error)) from error


def _publish_lease(
    context: dict[str, Any], local_lifecycle: dict[str, Any]
) -> dict[str, Any]:
    contract = context["contract"]
    lock_path = LEASE_CAS_RECEIPT_PATH.with_name(".pg1-lifecycle-lease-cas.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    try:
        if LEASE_CAS_RECEIPT_PATH.is_file():
            prior = json.loads(LEASE_CAS_RECEIPT_PATH.read_text(encoding="utf-8"))
            prior_etag = str(prior["etag"])
            lease_sequence = int(prior["lease_sequence"]) + 1
            if int(local_lifecycle["sequence"]) < int(prior["local_sequence"]):
                raise DeploymentError("durable lease update would regress local lifecycle")
        else:
            arm = json.loads(EXTERNAL_ARM_RECEIPT_PATH.read_text(encoding="utf-8"))
            prior_etag = str(arm["etag"])
            lease_sequence = 2
        reconciled_conflict = False
        for _cas_attempt in range(3):
            now = time.time()
            payload = {
                "schema": "pg1-native-external-deadman-lease/v2",
                "status": local_lifecycle["state"],
                "version": local_lifecycle["version"],
                "sequence": lease_sequence,
                "local_sequence": local_lifecycle["sequence"],
                "capsule_sha256": contract["capsule_sha256"],
                "attempt_run_id": contract["attempt_run_id"],
                "instance_id": contract["instance_id"],
                "stop_context_sha256": context["content_address"]["digest"],
                "local_lifecycle_sha256": local_lifecycle["self_sha256"],
                "local_lifecycle": local_lifecycle,
                "issued_unix": now,
                "hard_stop_unix": now + GUARDIAN_TIMEOUT_SECONDS,
                "heartbeat_stale_seconds": 30,
                "destroy_authority": False,
            }
            data = _canonical_bytes(payload)
            checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
            with tempfile.TemporaryDirectory(prefix="pg1-deadman-lease-") as directory:
                body = Path(directory) / "lease.json"
                body.write_bytes(data)
                command = [
                    str(AWS_EXECUTABLE), "--profile", "fuelos", "--region", "us-east-1",
                    "s3api", "put-object", "--bucket", AUTHORIZATION_BUCKET,
                    "--key", _lease_key(context), "--body", str(body),
                    "--if-match", prior_etag,
                    "--checksum-algorithm", "SHA256", "--checksum-sha256", checksum,
                    "--content-length", str(len(data)), "--content-type", "application/json",
                    "--server-side-encryption", "AES256", "--metadata", json.dumps(
                        {"capsule-sha256": contract["capsule_sha256"],
                         "attempt-run-id": contract["attempt_run_id"],
                         "instance-id": str(contract["instance_id"]),
                         "object-kind": "external-control-deadman-lease"},
                        sort_keys=True, separators=(",", ":")
                    ), "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
                ]
                try:
                    completed = subprocess.run(
                        command, check=False, capture_output=True, text=True,
                        timeout=PROVIDER_TIMEOUT_SECONDS,
                        cwd=str(IMMUTABLE_IMAGE_SOURCE_ROOT), env=_fixed_env(),
                    )
                except subprocess.TimeoutExpired as error:
                    raise DeploymentError("external deadman lease upload timed out") from error
            if completed.returncode != 0:
                reconciled_conflict = True
                durable_payload, durable_response = _read_durable_lease(context)
                winner = _reconcile_lease_conflict(
                    context, local_lifecycle, durable_payload
                )
                durable_lifecycle = durable_payload["local_lifecycle"]
                durable_receipt = {
                    "schema": "pg1-native-durable-lease-cas-receipt/v1",
                    "lease_sequence": int(durable_payload["sequence"]),
                    "local_sequence": int(durable_lifecycle["sequence"]),
                    "local_lifecycle_sha256": durable_lifecycle["self_sha256"],
                    "etag": str(durable_response["ETag"]),
                    "version_id": str(durable_response["VersionId"]),
                }
                _atomic_json_path(LEASE_CAS_RECEIPT_PATH, durable_receipt)
                if winner["self_sha256"] == durable_lifecycle["self_sha256"]:
                    try:
                        lifecycle.install_reconciled(
                            LIFECYCLE_PATH,
                            context,
                            expected_local=local_lifecycle,
                            durable=durable_lifecycle,
                        )
                    except lifecycle.LifecycleCASFailure as error:
                        raise DeploymentError(str(error)) from error
                    return {
                        **durable_payload,
                        "s3_version_id": str(durable_response["VersionId"]),
                        "etag": str(durable_response["ETag"]),
                        "reconciled_conflict": True,
                    }
                prior_etag = str(durable_response["ETag"])
                lease_sequence = int(durable_payload["sequence"]) + 1
                continue
            try:
                response = json.loads(completed.stdout or "")
            except json.JSONDecodeError as error:
                raise DeploymentError("external deadman lease response malformed") from error
            if (
                not isinstance(response, dict)
                or response.get("ChecksumSHA256") != checksum
                or response.get("ServerSideEncryption") != "AES256"
                or not response.get("VersionId")
                or not response.get("ETag")
            ):
                raise DeploymentError("external deadman lease upload unverified")
            receipt = {
                "schema": "pg1-native-durable-lease-cas-receipt/v1",
                "lease_sequence": lease_sequence,
                "local_sequence": int(local_lifecycle["sequence"]),
                "local_lifecycle_sha256": local_lifecycle["self_sha256"],
                "etag": str(response["ETag"]),
                "version_id": str(response["VersionId"]),
            }
            _atomic_json_path(LEASE_CAS_RECEIPT_PATH, receipt)
            return {
                **payload, "s3_version_id": str(response["VersionId"]),
                "etag": str(response["ETag"]),
                "reconciled_conflict": reconciled_conflict,
            }
        raise DeploymentError("external deadman conditional lease CAS retries exhausted")
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _fallback_bootstrap_command() -> list[str]:
    """Still enter pointer-first bootstrap when source layout validation fails."""

    return [
        str(
            (BOOTSTRAP_SOURCE_ROOT if BOOTSTRAP_SOURCE_ROOT.exists() else IMMUTABLE_IMAGE_SOURCE_ROOT)
            / "infra/vast/parallax/pg1_native_bootstrap.py"
        ),
        "--deploy", "--capsule", str(CAPSULE_RUNTIME_PATH),
        "--stop-context", str(STOP_CONTEXT_RUNTIME_PATH),
        "--entry-contract", str(ENTRY_CONTRACT_RUNTIME_PATH),
    ]


def _ensure_layout() -> None:
    for path, mode in FIXED_DIRECTORIES.items():
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        os.chmod(path, mode)
        if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != mode:
            raise DeploymentError(f"fixed deployment directory changed: {path}")
    for path, mode in FIXED_MOUNTS.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != mode
        ):
            raise DeploymentError(f"fixed deployment mount changed: {path}")


def _verify_context_receipts(context: dict[str, Any]) -> None:
    contract = context.get("contract") if isinstance(context, dict) else None
    receipts = contract.get("runtime_receipts") if isinstance(contract, dict) else None
    expected = {
        "python": Path("/usr/local/bin/python3"),
        "aws": Path("/usr/local/bin/aws"),
        "vast": Path("/usr/local/bin/vastai"),
        "aws_config": Path("/run/parallax/aws/config"),
        "aws_credentials": Path("/run/parallax/aws/credentials"),
        "vast_api_key": Path("/run/secrets/pg1-native-vast-api-key"),
        "runtime_image": Path("/run/parallax/runtime-image-digest"),
    }
    if not isinstance(receipts, dict) or set(receipts) != set(expected):
        raise DeploymentError("deployment receipt inventory changed")
    for label, path in expected.items():
        receipt = receipts[label]
        if (
            not isinstance(receipt, dict)
            or receipt.get("path") != str(path)
            or receipt.get("mode") not in {"0755", "0400"}
            or path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != int(receipt["mode"], 8)
            or _sha256(path) != receipt.get("sha256")
        ):
            raise DeploymentError(f"deployment {label} provenance changed")


def _inventory(capsule: dict[str, Any]) -> dict[Path, str]:
    values: dict[Path, str] = {}
    for row in capsule["contract"]["artifacts"].values():
        values[_safe_relative(row["path"])] = str(row["sha256"])
    for path, digest in capsule["contract"]["runtime_wiring"].items():
        values[_safe_relative(path)] = str(digest)
    return values


def stage_exact_runtime(capsule: dict[str, Any]) -> dict[str, Any]:
    if BOOTSTRAP_SOURCE_ROOT.exists():
        raise DeploymentError("refusing to merge fixed bootstrap source root")
    if IMMUTABLE_IMAGE_SOURCE_ROOT.is_symlink() or not IMMUTABLE_IMAGE_SOURCE_ROOT.is_dir():
        raise DeploymentError("immutable image source root is absent")
    rows: list[dict[str, Any]] = []
    try:
        for relative, expected_sha256 in sorted(
            _inventory(capsule).items(), key=lambda item: str(item[0])
        ):
            source = IMMUTABLE_IMAGE_SOURCE_ROOT / relative
            target = BOOTSTRAP_SOURCE_ROOT / relative
            if source.is_symlink() or not source.is_file() or _sha256(source) != expected_sha256:
                raise DeploymentError(f"immutable deployment source changed: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            target_mode = (
                0o755
                if str(relative) == "infra/vast/parallax/pg1_native_bootstrap.py"
                else 0o400
            )
            os.chmod(target, target_mode)
            if _sha256(target) != expected_sha256:
                raise DeploymentError(f"staged deployment source changed: {relative}")
            rows.append(
                {
                    "path": str(relative),
                    "sha256": expected_sha256,
                    "bytes": target.stat().st_size,
                    "mode": f"{target_mode:04o}",
                }
            )
    except Exception:
        shutil.rmtree(BOOTSTRAP_SOURCE_ROOT, ignore_errors=True)
        raise
    return {
        "schema": "pg1-native-deployment-layout-receipt/v1",
        "status": "FIXED_LAYOUT_AND_EXACT_RUNTIME_STAGED",
        "capsule_sha256": capsule["content_address"]["digest"],
        "immutable_source_root": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
        "bootstrap_source_root": str(BOOTSTRAP_SOURCE_ROOT),
        "directories": {str(path): f"{mode:04o}" for path, mode in FIXED_DIRECTORIES.items()},
        "mounts": {str(path): f"{mode:04o}" for path, mode in FIXED_MOUNTS.items()},
        "files": rows,
        "file_count": len(rows),
        "bootstrap_argv": bootstrap_command(capsule),
        "provider_calls": 0,
    }


def _write_receipt(receipt: dict[str, Any]) -> None:
    if LAYOUT_RECEIPT_PATH.exists():
        raise DeploymentError("refusing to overwrite deployment layout receipt")
    data = _canonical_bytes(receipt)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{LAYOUT_RECEIPT_PATH.name}.", dir=LAYOUT_RECEIPT_PATH.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, LAYOUT_RECEIPT_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def validate_layout_receipt(capsule: dict[str, Any]) -> dict[str, Any]:
    """Recheck the exact provisioned layout before bootstrap imports runtime owners."""

    if (
        LAYOUT_RECEIPT_PATH.is_symlink()
        or not LAYOUT_RECEIPT_PATH.is_file()
        or stat.S_IMODE(LAYOUT_RECEIPT_PATH.stat().st_mode) != 0o400
    ):
        raise DeploymentError("fixed deployment layout receipt is absent")
    try:
        receipt = json.loads(LAYOUT_RECEIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError("fixed deployment layout receipt is unreadable") from error
    expected_paths = {
        str(path): digest for path, digest in _inventory(capsule).items()
    }
    rows = receipt.get("files") if isinstance(receipt, dict) else None
    observed = {
        str(row.get("path")): str(row.get("sha256"))
        for row in rows
        if isinstance(row, dict)
    } if isinstance(rows, list) else {}
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema", "status", "capsule_sha256", "immutable_source_root",
            "bootstrap_source_root", "directories", "mounts", "files",
            "file_count", "bootstrap_argv", "provider_calls",
        }
        or receipt.get("schema") != "pg1-native-deployment-layout-receipt/v1"
        or receipt.get("status") != "FIXED_LAYOUT_AND_EXACT_RUNTIME_STAGED"
        or receipt.get("capsule_sha256") != capsule["content_address"]["digest"]
        or receipt.get("immutable_source_root") != str(IMMUTABLE_IMAGE_SOURCE_ROOT)
        or receipt.get("bootstrap_source_root") != str(BOOTSTRAP_SOURCE_ROOT)
        or receipt.get("directories")
        != {str(path): f"{mode:04o}" for path, mode in FIXED_DIRECTORIES.items()}
        or receipt.get("mounts")
        != {str(path): f"{mode:04o}" for path, mode in FIXED_MOUNTS.items()}
        or receipt.get("file_count") != len(expected_paths)
        or observed != expected_paths
        or receipt.get("bootstrap_argv") != bootstrap_command(capsule)
        or receipt.get("provider_calls") != 0
    ):
        raise DeploymentError("deployment layout receipt identity changed")
    for row in rows:
        if (
            set(row) != {"path", "sha256", "bytes", "mode"}
            or row.get("mode")
            != (
                "0755"
                if row.get("path") == "infra/vast/parallax/pg1_native_bootstrap.py"
                else "0400"
            )
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
        ):
            raise DeploymentError("deployment staged file receipt changed")
        target = BOOTSTRAP_SOURCE_ROOT / _safe_relative(row["path"])
        if (
            target.is_symlink()
            or not target.is_file()
            or stat.S_IMODE(target.stat().st_mode) != int(row["mode"], 8)
            or target.stat().st_size != row["bytes"]
            or _sha256(target) != row["sha256"]
        ):
            raise DeploymentError("deployment staged runtime verification changed")
    return receipt


def _verify_absolute_entries(capsule: dict[str, Any]) -> None:
    wiring = capsule["contract"]["runtime_wiring"]
    expected = {
        IMMUTABLE_IMAGE_SOURCE_ROOT / "infra/vast/parallax/pg1_native_deployment.py":
            wiring["infra/vast/parallax/pg1_native_deployment.py"],
    }
    for path, digest in expected.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o755
            or _sha256(path) != digest
        ):
            raise DeploymentError(f"absolute executable identity changed: {path}")


def _read_terminal_result(context: dict[str, Any]) -> dict[str, Any] | None:
    contract = context["contract"]
    for path in (GUARDIAN_RESULT_PATH,):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DeploymentError("supervised terminal result is unreadable") from error
        if (
            not isinstance(value, dict)
            or value.get("capsule_sha256") != contract["capsule_sha256"]
            or value.get("attempt_run_id") != contract["attempt_run_id"]
            or value.get("instance_id") != contract["instance_id"]
            or value.get("receipt_durable") is not True
            or value.get("terminal_state") != "stopped"
            or value.get("consecutive_terminal_observations") != 3
        ):
            raise DeploymentError("supervised terminal result identity changed")
        return value
    return None


def _start_failure_bootstrap() -> subprocess.Popen[str]:
    return subprocess.Popen(
        _fallback_bootstrap_command(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(IMMUTABLE_IMAGE_SOURCE_ROOT),
        env=_fixed_env(
            source_root=(
                BOOTSTRAP_SOURCE_ROOT
                if (
                    BOOTSTRAP_SOURCE_ROOT
                    / "infra/vast/parallax/pg1_native_bootstrap.py"
                ).is_file()
                else IMMUTABLE_IMAGE_SOURCE_ROOT
            )
        ),
        start_new_session=True,
        close_fds=True,
        text=True,
    )


def _supervise_local_terminal(
    context: dict[str, Any], process: subprocess.Popen[str] | None = None
) -> dict[str, Any]:
    """Local guardian fallback; the outside-worker deadman remains independent."""

    current = process
    failure_bootstrap: subprocess.Popen[str] | None = None
    while True:
        result = _read_terminal_result(context)
        if result is not None:
            return result
        if current is None or current.poll() is not None:
            if failure_bootstrap is None or failure_bootstrap.poll() is not None:
                try:
                    failure_bootstrap = _start_failure_bootstrap()
                    current = failure_bootstrap
                except BaseException:
                    current = None
        time.sleep(1)


def provision_and_launch(
    *,
    capsule: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> int:
    capsule = load_capsule_minimal() if capsule is None else capsule
    context = load_context_minimal() if context is None else context
    _ensure_layout()
    _verify_external_deadman_arm(capsule, context)
    _verify_absolute_entries(capsule)
    _verify_context_receipts(context)
    receipt = stage_exact_runtime(capsule)
    _write_receipt(receipt)
    running = _transition_lifecycle(
        context, "RUNNING_HEARTBEAT", reason="FIXED_LAYOUT_VERIFIED_BOOTSTRAP_STARTING"
    )
    _publish_lease(context, running)
    process = subprocess.Popen(
        bootstrap_command(capsule),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(IMMUTABLE_IMAGE_SOURCE_ROOT),
        env=_fixed_env(source_root=BOOTSTRAP_SOURCE_ROOT),
        close_fds=True,
        text=True,
    )
    while process.poll() is None:
        time.sleep(1)
    if process.returncode != 0:
        raise DeploymentError("supervised bootstrap exited nonzero")
    terminal = _supervise_local_terminal(context, process)
    stopped = _transition_lifecycle(
        context,
        "TERMINAL_STOPPED_PRESERVED",
        reason="PROVIDER_CONFIRMED_NONBILLABLE_STOP_SCRATCH_PRESERVED",
        terminal_receipt=terminal,
    )
    _publish_lease(context, stopped)
    return 0


def _load_entry_contract(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise DeploymentError("entry dispatch contract is absent or mutable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError("entry dispatch contract is unreadable") from error
    if (
        not isinstance(value, dict)
        or (value.get("schema"), value.get("kind")) not in {
            ("pg1-native-entry-dispatch-contract/v1", "PRODUCTION_AUTHORITY_BOUND"),
            ("pg1-native-entry-dispatch-contract/v1", "MECHANICAL_NO_AUTHORITY"),
            ("pg1-native-entry-dispatch-contract/v2",
             "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"),
        }
        or value.get("may_mint_authorization") is not False
    ):
        raise DeploymentError("entry dispatch contract schema/kind changed")
    return value


def _mechanical_entry_from_source() -> tuple[dict[str, Any], Path] | None:
    discovered = mechanical_rebase.worker_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    _root_contract, _root_contract_path, worker = discovered
    entry_path = worker / "run/parallax/pg1-native-entry-contract.json"
    contract = _load_entry_contract(entry_path)
    if (
        contract.get("kind") == "PRODUCTION_AUTHORITY_BOUND"
        or Path(str(contract.get("rebase_root", ""))).resolve() != worker
        or Path(str(contract.get("image_root", ""))).resolve()
        != worker / "opt/parallax/image/source"
        or Path(str(contract.get("bootstrap_root", ""))).resolve()
        != worker / "opt/parallax/future-freeze/source"
    ):
        raise DeploymentError("copied deployment source contract changed")
    return contract, entry_path.resolve()


def _mechanical_dispatch(
    contract: dict[str, Any], invoked_argv: list[str]
) -> int:
    global EXTERNAL_ARM_RECEIPT_PATH, PACKET_RUNTIME_PATH
    global AUTHORITY_BINDING_RUNTIME_PATH
    legacy_fields = {
        "schema", "kind", "capsule_sha256", "attempt_run_id", "argv_sha256",
        "argv", "rebase_root", "image_root", "bootstrap_root", "runtime_wiring",
        "bootstrap_argv", "receipt_path", "dispatch_scope",
        "production_authority", "may_mint_authorization",
    }
    worker_fields = {*legacy_fields, "worker_argv", "worker_receipt_path"}
    if (
        (contract.get("schema"), contract.get("kind")) not in {
            ("pg1-native-entry-dispatch-contract/v1", "MECHANICAL_NO_AUTHORITY"),
            ("pg1-native-entry-dispatch-contract/v2",
             "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"),
        }
        or contract.get("dispatch_scope") != "EXACT_REBASED_BYTES_ONLY"
        or contract.get("production_authority") is not False
        or (
            (contract.get("schema"), contract.get("kind"))
            == ("pg1-native-entry-dispatch-contract/v1", "MECHANICAL_NO_AUTHORITY")
            and set(contract) != legacy_fields
        )
        or (
            (contract.get("schema"), contract.get("kind"))
            == ("pg1-native-entry-dispatch-contract/v2",
                "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2")
            and set(contract) != worker_fields
        )
    ):
        raise DeploymentError("mechanical dispatch cannot enter production scope")
    if contract.get("argv") != invoked_argv or contract.get("argv_sha256") != hashlib.sha256(
        _canonical_bytes(invoked_argv)
    ).hexdigest():
        raise DeploymentError("mechanical provisioner argv changed")
    rebase_root = Path(str(contract["rebase_root"])).resolve()
    image_root = Path(str(contract["image_root"])).resolve()
    bootstrap_root = Path(str(contract["bootstrap_root"])).resolve()
    source = Path(__file__).resolve()
    if (
        rebase_root not in image_root.parents
        or rebase_root not in bootstrap_root.parents
        or source != image_root / "infra/vast/parallax/pg1_native_deployment.py"
        or not isinstance(contract.get("runtime_wiring"), dict)
    ):
        raise DeploymentError("mechanical rebase escaped its contract root")
    before: dict[str, str] = {}
    EXTERNAL_ARM_RECEIPT_PATH = rebase_root / "run/parallax/pg1-external-deadman-arm.json"
    PACKET_RUNTIME_PATH = rebase_root / "run/parallax/pg1-future-launch-packet.json"
    AUTHORITY_BINDING_RUNTIME_PATH = (
        rebase_root / "run/parallax/pg1-launch-authority-binding.json"
    )
    capsule_path = rebase_root / "run/parallax/pg1-execution-capsule.json"
    context_path = rebase_root / "run/parallax/pg1-native-stop-context.json"
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if contract["schema"] == "pg1-native-entry-dispatch-contract/v2":
        # The causal worker path consumes the exact receipt created upstream.
        _verify_external_deadman_arm(capsule, context)
    else:
        # The legacy no-worker rehearsal is explicitly mechanical and must not
        # masquerade a fabricated fixture as production deadman authority.
        mechanical_arm = json.loads(
            EXTERNAL_ARM_RECEIPT_PATH.read_text(encoding="utf-8")
        )
        if mechanical_arm != {
            "schema": "pg1-native-mechanical-fixed-mount/v1",
            "production_path": "/run/parallax/pg1-external-deadman-arm.json",
        }:
            raise DeploymentError("legacy mechanical arm marker changed")
    for relative_text, expected in contract["runtime_wiring"].items():
        relative = _safe_relative(relative_text)
        image = image_root / relative
        staged = bootstrap_root / relative
        if image.is_symlink() or not image.is_file() or _sha256(image) != expected:
            raise DeploymentError(f"mechanical exact image byte changed: {relative}")
        if not staged.exists():
            staged.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with image.open("rb") as reader, staged.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
                writer.flush(); os.fsync(writer.fileno())
            os.chmod(staged, stat.S_IMODE(image.stat().st_mode))
        if staged.is_symlink() or not staged.is_file() or _sha256(staged) != expected:
            raise DeploymentError(f"mechanical exact bootstrap byte changed: {relative}")
        before[relative_text] = expected
    completed = subprocess.run(
        list(contract["bootstrap_argv"]), check=False, capture_output=True, text=True,
        timeout=30, cwd=str(bootstrap_root),
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
             "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "PYTHONPATH": str(bootstrap_root)},
    )
    if completed.returncode != 0:
        raise DeploymentError("exact copied bootstrap bytes exited nonzero")
    for relative_text, expected in before.items():
        if _sha256(image_root / _safe_relative(relative_text)) != expected:
            raise DeploymentError("mechanical image bytes changed after dispatch")
        if _sha256(bootstrap_root / _safe_relative(relative_text)) != expected:
            raise DeploymentError("mechanical bootstrap bytes changed after dispatch")
    receipt_path = Path(str(contract["receipt_path"])).resolve()
    if rebase_root not in receipt_path.parents or not receipt_path.is_file():
        raise DeploymentError("exact bootstrap execution receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "EXACT_COPIED_BOOTSTRAP_BYTES_EXECUTED"
        or receipt.get("production_authority") is not False
        or receipt.get("provider_calls") != 0
    ):
        raise DeploymentError("exact bootstrap execution receipt changed")
    if contract["kind"] == "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2" and (
        receipt.get("worker_status") != "EXACT_COPIED_WORKER_BYTES_EXECUTED"
        or not isinstance(receipt.get("worker_sha256"), str)
        or len(receipt["worker_sha256"]) != 64
        or not isinstance(receipt.get("worker_receipt_sha256"), str)
        or len(receipt["worker_receipt_sha256"]) != 64
    ):
        raise DeploymentError("exact worker execution receipt changed")
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--stop-context", type=Path)
    parser.add_argument("--entry-contract", type=Path)
    args = parser.parse_args(argv)
    if args.entry_contract is None:
        raise DeploymentError("entry dispatch contract path is required")
    discovered = _mechanical_entry_from_source()
    actual_argv = [str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
    if discovered is not None:
        entry_contract, expected_entry = discovered
        if args.entry_contract.resolve() != expected_entry:
            raise DeploymentError("mechanical entry contract path is not fixed")
        return _mechanical_dispatch(entry_contract, actual_argv)
    if args.entry_contract != ENTRY_CONTRACT_RUNTIME_PATH:
        raise DeploymentError("production entry contract path is not fixed")
    entry_contract = _load_entry_contract(args.entry_contract)
    if entry_contract["kind"] != "PRODUCTION_AUTHORITY_BOUND":
        raise DeploymentError("mechanical contract cannot select canonical source bytes")
    context = load_context_minimal()
    capsule = load_capsule_minimal()
    expected_argv = capsule["contract"]["entrypoints"]["production_argv"]
    if (
        args.entry_contract != ENTRY_CONTRACT_RUNTIME_PATH
        or entry_contract.get("kind") != "PRODUCTION_AUTHORITY_BOUND"
        or entry_contract.get("dispatch_scope") != "PRODUCTION_ENTRY_ONLY"
        or entry_contract.get("capsule_sha256") != capsule["content_address"]["digest"]
        or entry_contract.get("attempt_run_id")
        != capsule["contract"]["attempt_identity"]["attempt_run_id"]
        or entry_contract.get("argv_sha256")
        != hashlib.sha256(_canonical_bytes(expected_argv)).hexdigest()
        or entry_contract.get("bootstrap_argv_sha256")
        != hashlib.sha256(
            _canonical_bytes(capsule["contract"]["entrypoints"]["bootstrap_argv"])
        ).hexdigest()
        or expected_argv != actual_argv
    ):
        raise DeploymentError("production entry dispatch contract changed")
    _verify_external_deadman_arm(capsule, context)
    lifecycle = _transition_lifecycle(
        context, "ARMED", reason="OUTER_ABSOLUTE_ENTRY_ARMED_BEFORE_CLI"
    )
    try:
        _verify_absolute_entries(capsule)
        _publish_lease(context, lifecycle)
        if (
            not args.launch
            or args.capsule != CAPSULE_RUNTIME_PATH
            or args.stop_context != STOP_CONTEXT_RUNTIME_PATH
        ):
            raise DeploymentError("production deployment argv changed")
        return provision_and_launch(capsule=capsule, context=context)
    except BaseException as error:
        failed = _transition_lifecycle(
            context, "FAILURE_STOP_REQUIRED", reason=type(error).__name__
        )
        try:
            _publish_lease(context, failed)
        except BaseException:
            pass
        try:
            _start_failure_bootstrap()
        except BaseException:
            pass
        # The local lifecycle state makes STOP immediate even during S3 loss.
        # Keep the independent deadman alive until a durable terminal receipt.
        terminal = _supervise_local_terminal(context)
        stopped = _transition_lifecycle(
            context,
            "TERMINAL_STOPPED_PRESERVED",
            reason="PROVIDER_CONFIRMED_NONBILLABLE_STOP_SCRATCH_PRESERVED",
            terminal_receipt=terminal,
        )
        _publish_lease(context, stopped)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
