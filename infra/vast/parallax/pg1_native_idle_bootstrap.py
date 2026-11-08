#!/usr/local/bin/python3
"""Science-disabled PG1 worker prestart gate and fixed-layout handoff verifier."""

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
    from . import pg1_native_mechanical_rebase as mechanical_rebase
except ImportError:
    from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase


POINTER = Path("/run/parallax/pg1-idle-bootstrap-pointer.json")
START_LEASE = Path("/run/parallax/pg1-native-start-lease.json")
CAPSULE = Path("/run/parallax/pg1-execution-capsule.json")
STOP_CONTEXT = Path("/run/parallax/pg1-native-stop-context.json")
ENTRY_CONTRACT = Path("/run/parallax/pg1-native-entry-contract.json")
EXTERNAL_ARM = Path("/run/parallax/pg1-external-deadman-arm.json")
AWS_CONFIG = Path("/run/parallax/aws/config")
AWS_CREDENTIALS = Path("/run/parallax/aws/credentials")
VAST_KEY = Path("/run/secrets/pg1-native-vast-api-key")
RUNTIME_RECEIPT = Path("/run/parallax/runtime-image-digest")
LAYOUT_RECEIPT = Path("/run/parallax/pg1-native-handoff-layout.json")
PACKET = Path("/run/parallax/pg1-future-launch-packet.json")
INSTALLED_SOURCE = Path("/run/parallax/pg1-native-installed-source-layout.json")
AUTHORITY_BINDING = Path("/run/parallax/pg1-launch-authority-binding.json")
AWS = Path("/usr/local/bin/aws")
SOURCE_ROOT = Path("/opt/parallax/image/source")
PRODUCTION_ENTRY = SOURCE_ROOT / "infra/vast/parallax/pg1_native_deployment.py"
POLL_SECONDS = 1
BUCKET = "pg-lineage-769063704696"
OWNER = "769063704696"
MECHANICAL_WORKER_ROOT: Path | None = None
MECHANICAL_CANONICAL_MOUNTS: dict[Path, str] = {}


class IdleBootstrapError(RuntimeError):
    pass


class StartCommitPending(IdleBootstrapError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o400); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pointer_first(raw_argv: list[str]) -> None:
    _atomic(
        POINTER,
        {"schema": "pg1-native-idle-prestart-pointer/v1",
         "status": "SCIENCE_DISABLED_WAITING_FOR_BOUND_START_LEASE",
         "argv_sha256": hashlib.sha256(canonical_bytes(raw_argv)).hexdigest(),
         "science_released": False, "wall_time_unix": time.time()},
    )


def _fixed(path: Path, mode: int) -> bytes:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != mode:
        raise IdleBootstrapError(f"bound handoff file changed: {path}")
    return path.read_bytes()


def _json(path: Path, mode: int = 0o400) -> dict[str, Any]:
    try:
        value = json.loads(_fixed(path, mode))
    except json.JSONDecodeError as error:
        raise IdleBootstrapError(f"bound handoff JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise IdleBootstrapError("bound handoff JSON must be an object")
    return value


def _aws_env() -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "AWS_PROFILE": "fuelos",
        "AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": str(AWS_CONFIG),
        "AWS_SHARED_CREDENTIALS_FILE": str(AWS_CREDENTIALS),
        "AWS_EC2_METADATA_DISABLED": "true", "PYTHONPATH": str(SOURCE_ROOT),
    }


def _start_commit(intent: dict[str, Any]) -> dict[str, Any]:
    key = intent.get("start_commit_key")
    if not isinstance(key, str) or not key:
        raise IdleBootstrapError("canonical start commit key changed")
    head = subprocess.run(
        [str(AWS), "--profile", "fuelos", "--region", "us-east-1",
         "s3api", "head-object", "--bucket", BUCKET, "--key", key,
         "--expected-bucket-owner", OWNER],
        check=False, capture_output=True, text=True, timeout=30,
        cwd=str(SOURCE_ROOT), env=_aws_env(), close_fds=True,
    )
    if head.returncode != 0:
        raise StartCommitPending("canonical start commit not yet visible")
    try:
        head_value = json.loads(head.stdout or "{}")
    except json.JSONDecodeError as error:
        raise IdleBootstrapError("canonical start HEAD malformed") from error
    version_id = head_value.get("VersionId")
    with tempfile.TemporaryDirectory(prefix="pg1-idle-start-") as directory:
        target = Path(directory) / "commit.json"
        result = subprocess.run(
            [str(AWS), "--profile", "fuelos", "--region", "us-east-1",
             "s3api", "get-object", "--bucket", BUCKET, "--key", key,
             "--version-id", str(version_id), "--checksum-mode", "ENABLED",
             "--expected-bucket-owner", OWNER, str(target)],
            check=False, capture_output=True, text=True, timeout=30,
            cwd=str(SOURCE_ROOT), env=_aws_env(), close_fds=True,
        )
        if result.returncode != 0:
            raise StartCommitPending("canonical start version not yet visible")
        try:
            response = json.loads(result.stdout or "{}")
            data = target.read_bytes(); body = json.loads(data)
        except (OSError, json.JSONDecodeError) as error:
            raise IdleBootstrapError("canonical start GET malformed") from error
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    contract = body.get("contract") if isinstance(body, dict) else None
    expected_metadata = {
        "object-kind": "canonical-start-commit",
        "packet-sha256": intent["packet_sha256"],
        "capsule-sha256": intent["capsule_sha256"],
        "attempt-run-id": intent["attempt_run_id"],
        "canonical-state-sha256": intent["canonical_state_sha256"],
    }
    if (
        not version_id or body.get("schema") != "pg1-native-canonical-start-commit/v1"
        or body.get("status") != "CANONICAL_START_COMMITTED"
        or not isinstance(contract, dict)
        or body.get("content_address") != {
            "algorithm": "sha256", "digest": hashlib.sha256(canonical_bytes(contract)).hexdigest()
        }
        or body["content_address"]["digest"] != intent["start_commit_sha256"]
        or contract != {
            "packet_sha256": intent["packet_sha256"],
            "capsule_sha256": intent["capsule_sha256"],
            "attempt_run_id": intent["attempt_run_id"],
            "instance_id": intent["instance_id"],
            "canonical_state_sequence": intent["canonical_state_sequence"],
            "canonical_state_sha256": intent["canonical_state_sha256"],
            "stop_context_sha256": intent["stop_context_sha256"],
            "deadman_ack_sha256": intent["deadman_ack_sha256"],
            "handoff_layout_sha256": intent["handoff_layout_sha256"],
            "start_commit_key": intent["start_commit_key"],
            "science_released": True,
        }
        or response.get("VersionId") != version_id
        or response.get("ETag") != head_value.get("ETag")
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or head_value.get("ServerSideEncryption") != "AES256"
        or response.get("ContentLength") != len(data)
        or head_value.get("ContentLength") != len(data)
        or response.get("ContentType") != "application/json"
        or head_value.get("ContentType") != "application/json"
        or response.get("Metadata") != expected_metadata
        or head_value.get("Metadata") != expected_metadata
    ):
        raise IdleBootstrapError("canonical start commit provenance changed")
    return body


def validate_and_command(*, packet_sha256: str, attempt_run_id: str) -> list[str]:
    lease = _json(START_LEASE)
    capsule = _json(CAPSULE)
    context = _json(STOP_CONTEXT)
    layout = _json(LAYOUT_RECEIPT)
    arm = _json(EXTERNAL_ARM)
    entry = _json(ENTRY_CONTRACT)
    packet = _json(PACKET)
    installed_source = _json(INSTALLED_SOURCE)
    authority_binding = _json(AUTHORITY_BINDING)
    capsule_contract = capsule.get("contract") if isinstance(capsule, dict) else None
    capsule_digest = hashlib.sha256(canonical_bytes(capsule_contract)).hexdigest()
    packet_contract = packet.get("contract") if isinstance(packet, dict) else None
    expected_ack_nonce = hashlib.sha256(canonical_bytes({
        "scope": "PG1_INSTANCE_BOUND_DEADMAN_ACK_NONCE_V1",
        "packet_sha256": packet_sha256,
        "capsule_sha256": capsule_digest,
        "attempt_run_id": attempt_run_id,
        "instance_id": context.get("contract", {}).get("instance_id"),
        "stop_context_sha256": context.get("content_address", {}).get("digest"),
    })).hexdigest()
    expected_ack_id = hashlib.sha256(
        f"PG1_BOUND_ACK_V3:{expected_ack_nonce}".encode()
    ).hexdigest()
    frozen = packet_contract.get("frozen_inputs", {}) if isinstance(
        packet_contract, dict
    ) else {}
    expected_mount_inventory = {
        "/run/parallax/pg1-execution-capsule.json": capsule_digest,
        "/run/parallax/pg1-native-stop-context.json": context.get(
            "content_address", {}
        ).get("digest"),
        "/run/parallax/aws/config": frozen.get("control_mounts", {}).get(
            "aws_config_sha256"
        ),
        "/run/parallax/aws/credentials": frozen.get("control_mounts", {}).get(
            "aws_credentials_sha256"
        ),
        "/run/secrets/pg1-native-vast-api-key": context.get("contract", {}).get(
            "runtime_receipts", {}
        ).get("vast_api_key", {}).get("sha256"),
        "/run/parallax/runtime-image-digest": frozen.get("runtime_image", {}).get(
            "receipt_sha256"
        ),
        "/run/parallax/pg1-native-entry-contract.json": hashlib.sha256(
            canonical_bytes(entry)
        ).hexdigest(),
        "/run/parallax/pg1-future-launch-packet.json": packet_sha256,
    }
    expected_mount_inventory_sha256 = hashlib.sha256(
        canonical_bytes(expected_mount_inventory)
    ).hexdigest()
    if (
        not isinstance(capsule_contract, dict)
        or capsule.get("content_address")
        != {"algorithm": "sha256", "digest": capsule_digest}
        or capsule_contract.get("attempt_identity", {}).get("attempt_run_id")
        != attempt_run_id
        or lease.get("schema") != "pg1-native-bound-start-intent/v1"
        or lease.get("status") != "SCIENCE_DISABLED_WAITING_FOR_CANONICAL_START_COMMIT"
        or lease.get("packet_sha256") != packet_sha256
        or lease.get("capsule_sha256") != capsule_digest
        or lease.get("attempt_run_id") != attempt_run_id
        or lease.get("science_released") is not False
        or not isinstance(lease.get("not_after_unix"), (int, float))
        or time.time() >= float(lease["not_after_unix"])
        or lease.get("deadman_ack_sha256") != hashlib.sha256(canonical_bytes(arm)).hexdigest()
        or context.get("content_address", {}).get("digest")
        != lease.get("stop_context_sha256")
        or layout.get("schema") != "pg1-native-bound-handoff-layout/v1"
        or layout.get("packet_sha256") != packet_sha256
        or layout.get("instance_id") != context.get("contract", {}).get("instance_id")
        or entry.get("kind") != (
            "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"
            if MECHANICAL_WORKER_ROOT is not None else "PRODUCTION_AUTHORITY_BOUND"
        )
        or not isinstance(packet_contract, dict)
        or packet.get("content_address") != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(packet_contract)).hexdigest(),
        }
        or packet.get("content_address", {}).get("digest") != packet_sha256
        or installed_source.get("schema") != "pg1-native-installed-source-layout/v1"
        or installed_source.get("packet_sha256") != packet_sha256
        or installed_source.get("source_bundle_sha256")
        != packet.get("contract", {}).get("frozen_inputs", {}).get("source_bundle", {}).get("sha256")
        or installed_source.get("members")
        != packet.get("contract", {}).get("frozen_inputs", {}).get("source_bundle", {}).get("members")
        or installed_source.get("science_released") is not False
        or arm.get("schema") != "pg1-native-instance-bound-deadman-ack/v3"
        or arm.get("status") != "INSTANCE_BOUND_DEADMAN_ARMED"
        or arm.get("issuer") != "PG1_CANONICAL_EXTERNAL_CONTROL_DEADMAN_V1"
        or arm.get("capsule_sha256") != capsule_digest
        or arm.get("attempt_run_id") != attempt_run_id
        or arm.get("instance_id") != context.get("contract", {}).get("instance_id")
        or arm.get("offer_id") != context.get("contract", {}).get("offer_id")
        or arm.get("machine_id") != context.get("contract", {}).get("machine_id")
        or arm.get("container_id") != context.get("contract", {}).get("container_id")
        or arm.get("stop_context_sha256") != context.get("content_address", {}).get("digest")
        or arm.get("packet_sha256") != packet_sha256
        or arm.get("mount_inventory_sha256")
        != expected_mount_inventory_sha256
        or arm.get("ack_nonce") != expected_ack_nonce
        or arm.get("ack_id") != expected_ack_id
        or set(arm) != {
            "schema", "status", "packet_sha256", "issuer", "capsule_sha256",
            "attempt_run_id", "instance_id", "offer_id", "machine_id",
            "container_id", "stop_context_sha256", "authorization_body_sha256",
            "authorization_receipt_sha256", "mount_inventory_sha256", "ack_id",
            "ack_nonce", "replay_envelope", "destroy_authority",
            "ack_body_sha256", "etag", "version_id", "checksum_sha256",
        }
        or authority_binding != {
            "schema": "pg1-native-worker-authority-binding/v1",
            "packet_sha256": arm.get("packet_sha256"),
            "capsule_sha256": arm.get("capsule_sha256"),
            "attempt_run_id": arm.get("attempt_run_id"),
            "authorization_body_sha256": arm.get("authorization_body_sha256"),
            "authorization_receipt_sha256": arm.get("authorization_receipt_sha256"),
            "mount_inventory_sha256": arm.get("mount_inventory_sha256"),
        }
        or arm.get("ack_body_sha256") != hashlib.sha256(canonical_bytes({
            key: arm[key] for key in arm
            if key not in {"ack_body_sha256", "etag", "version_id", "checksum_sha256"}
        })).hexdigest()
        or arm.get("checksum_sha256") != base64.b64encode(hashlib.sha256(
            canonical_bytes({
                key: arm[key] for key in arm
                if key not in {
                    "ack_body_sha256", "etag", "version_id", "checksum_sha256"
                }
            })
        ).digest()).decode("ascii")
        or arm.get("replay_envelope") != {
            "schema": "pg1-native-bound-ack-exact-replay/v1",
            "ack_id": arm.get("ack_id"), "one_shot": True,
            "exact_replay_only": True,
        }
        or not all(
            isinstance(arm.get(field), str) and bool(arm[field])
            for field in ("ack_id", "ack_nonce", "etag", "version_id", "checksum_sha256")
        )
        or arm.get("destroy_authority") is not False
    ):
        raise IdleBootstrapError("bound start lease or handoff identity changed")
    _start_commit(lease)
    expected = {
        CAPSULE: (0o400, hashlib.sha256(canonical_bytes(capsule)).hexdigest()),
        STOP_CONTEXT: (0o400, hashlib.sha256(canonical_bytes(context)).hexdigest()),
        AWS_CONFIG: (0o400, context["contract"]["runtime_receipts"]["aws_config"]["sha256"]),
        AWS_CREDENTIALS: (0o400, context["contract"]["runtime_receipts"]["aws_credentials"]["sha256"]),
        VAST_KEY: (0o400, context["contract"]["runtime_receipts"]["vast_api_key"]["sha256"]),
        RUNTIME_RECEIPT: (0o400, context["contract"]["runtime_receipts"]["runtime_image"]["sha256"]),
        ENTRY_CONTRACT: (0o400, hashlib.sha256(canonical_bytes(entry)).hexdigest()),
        EXTERNAL_ARM: (0o400, hashlib.sha256(canonical_bytes(arm)).hexdigest()),
        PACKET: (0o400, hashlib.sha256(canonical_bytes(packet)).hexdigest()),
        AUTHORITY_BINDING: (
            0o400, hashlib.sha256(canonical_bytes(authority_binding)).hexdigest()
        ),
    }
    inventory = layout.get("mounts")
    if not isinstance(inventory, dict):
        raise IdleBootstrapError("bound mount inventory absent")
    for path, (mode, digest) in expected.items():
        data = _fixed(path, mode)
        canonical_name = MECHANICAL_CANONICAL_MOUNTS.get(path, str(path))
        if hashlib.sha256(data).hexdigest() != digest or inventory.get(canonical_name) != {
            "mode": f"{mode:04o}", "sha256": digest, "bytes": len(data)
        }:
            raise IdleBootstrapError(f"bound mount provenance changed: {path}")
    source_hashes = layout.get("source_runtime_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise IdleBootstrapError("bound source runtime inventory absent")
    for relative_text, expected_sha256 in source_hashes.items():
        relative = Path(str(relative_text))
        if relative.is_absolute() or ".." in relative.parts:
            raise IdleBootstrapError("bound source runtime path escaped root")
        source = SOURCE_ROOT / relative
        if (
            source.is_symlink() or not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256
        ):
            raise IdleBootstrapError(f"bound source runtime changed: {relative}")
    production_argv = capsule_contract.get("entrypoints", {}).get("production_argv")
    if not isinstance(production_argv, list) or production_argv[0] != str(PRODUCTION_ENTRY):
        raise IdleBootstrapError("production entry argv changed")
    selected_argv = production_argv
    selected_source = SOURCE_ROOT
    selected_path = "/usr/local/bin:/usr/bin:/bin"
    if MECHANICAL_WORKER_ROOT is not None:
        rebased = [
            str(MECHANICAL_WORKER_ROOT / item.lstrip("/"))
            if item.startswith("/") else item for item in production_argv
        ]
        if (
            entry.get("schema") != "pg1-native-entry-dispatch-contract/v2"
            or entry.get("capsule_sha256") != capsule_digest
            or entry.get("attempt_run_id") != attempt_run_id
            or entry.get("argv") != rebased
            or entry.get("argv_sha256")
            != hashlib.sha256(canonical_bytes(rebased)).hexdigest()
            or entry.get("rebase_root") != str(MECHANICAL_WORKER_ROOT)
            or entry.get("image_root") != str(SOURCE_ROOT)
            or entry.get("runtime_wiring") != capsule_contract.get("runtime_wiring")
            or entry.get("dispatch_scope") != "EXACT_REBASED_BYTES_ONLY"
            or entry.get("production_authority") is not False
            or entry.get("may_mint_authorization") is not False
        ):
            raise IdleBootstrapError("mechanical production entry identity changed")
        selected_argv = rebased
        selected_path = (
            f"{MECHANICAL_WORKER_ROOT / 'usr/local/bin'}:/usr/bin:/bin"
        )
    return [
        "/usr/bin/env", "-i", f"PATH={selected_path}", "HOME=/nonexistent",
        "LANG=C.UTF-8", "LC_ALL=C.UTF-8", f"PYTHONPATH={selected_source}", *selected_argv,
    ]


def run(*, packet_sha256: str, attempt_run_id: str, maximum_seconds: int) -> int:
    deadline = time.monotonic() + maximum_seconds
    while time.monotonic() < deadline:
        if START_LEASE.is_file():
            try:
                command = validate_and_command(
                    packet_sha256=packet_sha256, attempt_run_id=attempt_run_id
                )
            except StartCommitPending:
                time.sleep(POLL_SECONDS)
                continue
            except IdleBootstrapError:
                # The remote handoff copier may expose a path before the exact
                # immutable bytes/mode are complete.  Stay science-disabled and
                # retry; a finalized invalid lease still fails at the deadline.
                time.sleep(POLL_SECONDS)
                continue
            completed = subprocess.run(
                command, check=False, cwd=str(SOURCE_ROOT), env={}, close_fds=True
            )
            if completed.returncode != 0:
                raise IdleBootstrapError("production entry returned nonzero")
            return 0
        time.sleep(POLL_SECONDS)
    raise IdleBootstrapError("bound start lease deadline expired with science disabled")


def configure_from_source_location() -> dict[str, Any] | None:
    """Derive any mechanical worker root from these installed source bytes."""

    global POINTER, START_LEASE, CAPSULE, STOP_CONTEXT, ENTRY_CONTRACT, EXTERNAL_ARM
    global AWS_CONFIG, AWS_CREDENTIALS, VAST_KEY, RUNTIME_RECEIPT, LAYOUT_RECEIPT
    global PACKET, INSTALLED_SOURCE, AUTHORITY_BINDING, AWS, SOURCE_ROOT
    global MECHANICAL_WORKER_ROOT
    global POLL_SECONDS, MECHANICAL_CANONICAL_MOUNTS
    discovered = mechanical_rebase.worker_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    contract, _contract_path, worker = discovered
    def at(path: str) -> Path:
        return worker / Path(path).relative_to("/")
    POINTER = at("/run/parallax/pg1-idle-bootstrap-pointer.json")
    START_LEASE = at("/run/parallax/pg1-native-start-lease.json")
    CAPSULE = at("/run/parallax/pg1-execution-capsule.json")
    STOP_CONTEXT = at("/run/parallax/pg1-native-stop-context.json")
    ENTRY_CONTRACT = at("/run/parallax/pg1-native-entry-contract.json")
    EXTERNAL_ARM = at("/run/parallax/pg1-external-deadman-arm.json")
    AWS_CONFIG = at("/run/parallax/aws/config")
    AWS_CREDENTIALS = at("/run/parallax/aws/credentials")
    VAST_KEY = at("/run/secrets/pg1-native-vast-api-key")
    RUNTIME_RECEIPT = at("/run/parallax/runtime-image-digest")
    LAYOUT_RECEIPT = at("/run/parallax/pg1-native-handoff-layout.json")
    PACKET = at("/run/parallax/pg1-future-launch-packet.json")
    INSTALLED_SOURCE = at("/run/parallax/pg1-native-installed-source-layout.json")
    AUTHORITY_BINDING = at("/run/parallax/pg1-launch-authority-binding.json")
    AWS = at("/usr/local/bin/aws")
    SOURCE_ROOT = at("/opt/parallax/image/source")
    MECHANICAL_WORKER_ROOT = worker
    MECHANICAL_CANONICAL_MOUNTS = {
        at(name): name for name in (
            "/run/parallax/pg1-execution-capsule.json",
            "/run/parallax/pg1-native-stop-context.json",
            "/run/parallax/aws/config", "/run/parallax/aws/credentials",
            "/run/secrets/pg1-native-vast-api-key",
            "/run/parallax/runtime-image-digest",
            "/run/parallax/pg1-native-entry-contract.json",
            "/run/parallax/pg1-external-deadman-arm.json",
            "/run/parallax/pg1-future-launch-packet.json",
            "/run/parallax/pg1-launch-authority-binding.json",
        )
    }
    POLL_SECONDS = 0.01
    return contract


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else __import__("sys").argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--idle-prestart", action="store_true")
    parser.add_argument("--packet-sha256")
    parser.add_argument("--attempt-run-id")
    parser.add_argument("--maximum-seconds", type=int, default=300)
    args = parser.parse_args(raw)
    configure_from_source_location()
    pointer_first(raw)
    if (
        not args.idle_prestart or not args.packet_sha256 or len(args.packet_sha256) != 64
        or not args.attempt_run_id or not 30 <= args.maximum_seconds <= 900
    ):
        raise IdleBootstrapError("idle bootstrap argv changed after pointer")
    return run(packet_sha256=args.packet_sha256,
               attempt_run_id=args.attempt_run_id,
               maximum_seconds=args.maximum_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
