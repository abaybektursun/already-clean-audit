#!/usr/local/bin/python3
"""Compile a future PG1 native launch packet without minting authority.

The compiler is deliberately provider-free.  It binds one caller-supplied
integer offer, one already-addressed capsule attempt, and one immutable input
manifest into a content-addressed transaction contract.  The emitted
authority row is a request for the canonical production issuer, never an
authorization or a provider identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "pg1-native-future-launch-packet/v2"
STATUS = "COMPILED_REQUEST_NOT_AUTHORITY"
INPUT_SCHEMA = "pg1-native-future-launch-inputs/v1"
PROTECTED_BODY_ARTIFACT_LABELS = frozenset({"discovery_manifest", "discovery_seal"})
ISSUER = "PG1_PRODUCTION_S3_CONDITIONAL_STORE_V1"
CONTROL_ROOT = Path("/opt/parallax/control/source")
GATEWAY = CONTROL_ROOT / "infra/vast/parallax/pg1_native_launch_gateway.py"
IDLE_BOOTSTRAP = Path(
    "/opt/parallax/image/source/infra/vast/parallax/pg1_native_idle_bootstrap.py"
)
ONSTART_MATERIALIZER = Path(
    "/run/parallax/pg1-native-onstart-materializer.py"
)
CAPSULE_PATH = Path("/run/parallax/control-plane/pg1-execution-capsule.json")
PACKET_PATH = Path("/run/parallax/control-plane/pg1-future-launch-packet.json")
TRANSACTION_PATH = Path(
    "/run/parallax/control-plane/pg1-worker-launch-transaction.json"
)
START_LEASE_PATH = Path("/run/parallax/pg1-native-start-lease.json")
WORKER_PACKET_PATH = Path("/run/parallax/pg1-future-launch-packet.json")


class LaunchPacketError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def address(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "content_address": {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        },
        "contract": contract,
    }


def _fixed_json(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise LaunchPacketError(f"fixed compiler input changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchPacketError(f"compiler input unreadable: {path}") from error
    if not isinstance(value, dict) or (schema is not None and value.get("schema") != schema):
        raise LaunchPacketError("compiler input schema changed")
    return value


def _capsule_identity(capsule: dict[str, Any]) -> tuple[str, str]:
    contract = capsule.get("contract") if isinstance(capsule, dict) else None
    if (
        not isinstance(contract, dict)
        or capsule.get("content_address")
        != {"algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(contract)).hexdigest()}
    ):
        raise LaunchPacketError("capsule content address changed")
    attempt = contract.get("attempt_identity")
    if not isinstance(attempt, dict) or not isinstance(attempt.get("attempt_run_id"), str):
        raise LaunchPacketError("capsule attempt identity changed")
    return str(capsule["content_address"]["digest"]), str(attempt["attempt_run_id"])


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise LaunchPacketError(f"{label} digest changed")
    return text


def validate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    if (
        set(inputs)
        != {"schema", "worker_image", "worker_disk_gb", "source_bundle",
            "runtime_image", "control_mounts", "control_executables",
            "worker_runtime_receipts",
            "maximum_prestart_seconds"}
        or inputs.get("schema") != INPUT_SCHEMA
        or not isinstance(inputs.get("worker_image"), str)
        or "@sha256:" not in inputs["worker_image"]
        or isinstance(inputs.get("worker_disk_gb"), bool)
        or not isinstance(inputs.get("worker_disk_gb"), int)
        or not 1 <= inputs["worker_disk_gb"] <= 4096
        or not isinstance(inputs.get("maximum_prestart_seconds"), int)
        or not 30 <= inputs["maximum_prestart_seconds"] <= 900
    ):
        raise LaunchPacketError("future launch input envelope changed")
    source = inputs.get("source_bundle")
    runtime = inputs.get("runtime_image")
    mounts = inputs.get("control_mounts")
    control_executables = inputs.get("control_executables")
    worker_receipts = inputs.get("worker_runtime_receipts")
    if (
        not isinstance(source, dict)
        or set(source) != {"bucket", "key", "version_id", "sha256", "bytes", "members"}
        or not all(isinstance(source.get(field), str) and source[field]
                   for field in ("bucket", "key", "version_id"))
        or _digest(source.get("sha256"), "source bundle") != source["sha256"]
        or not isinstance(source.get("bytes"), int)
        or source["bytes"] <= 0
        or not isinstance(source.get("members"), dict)
        or not source["members"]
        or not isinstance(runtime, dict)
        or set(runtime) != {"image", "receipt_sha256"}
        or runtime.get("image") != inputs["worker_image"]
        or _digest(runtime.get("receipt_sha256"), "runtime receipt")
        != runtime["receipt_sha256"]
        or not isinstance(mounts, dict)
        or set(mounts)
        != {"aws_config_sha256", "aws_credentials_sha256",
            "control_vast_key_sha256", "ssh_private_key_sha256"}
        or not isinstance(worker_receipts, dict)
        or set(worker_receipts) != {"python_sha256", "aws_sha256", "vast_sha256"}
        or not isinstance(control_executables, dict)
        or set(control_executables) != {"aws_sha256", "vastai_sha256"}
    ):
        raise LaunchPacketError("future launch frozen input changed")
    for label, digest_value in mounts.items():
        _digest(digest_value, label)
    for label, digest_value in worker_receipts.items():
        _digest(digest_value, label)
    for label, digest_value in control_executables.items():
        _digest(digest_value, label)
    for relative_text, member in source["members"].items():
        relative = Path(str(relative_text))
        if (
            not isinstance(relative_text, str) or relative.is_absolute()
            or not relative.parts or ".." in relative.parts
            or not isinstance(member, dict)
            or set(member) != {"sha256", "bytes", "mode"}
            or _digest(member.get("sha256"), f"source member {relative_text}")
            != member["sha256"]
            or not isinstance(member.get("bytes"), int) or member["bytes"] < 0
            or (member["bytes"] == 0 and relative_text != "infra/__init__.py")
            or member.get("mode") not in {"0644", "0755"}
        ):
            raise LaunchPacketError("future launch source member inventory changed")
    return inputs


def compile_packet(
    capsule: dict[str, Any], inputs: dict[str, Any], *, offer_id: int,
    attempt_run_id: str,
) -> dict[str, Any]:
    capsule_sha, capsule_attempt = _capsule_identity(capsule)
    inputs = validate_inputs(inputs)
    if (
        isinstance(offer_id, bool) or not isinstance(offer_id, int) or offer_id <= 0
        or not attempt_run_id or attempt_run_id != capsule_attempt
    ):
        raise LaunchPacketError("offer or parameterized attempt changed")
    input_sha = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    runtime_hashes = capsule["contract"].get("runtime_wiring")
    artifact_rows = capsule["contract"].get("artifacts")
    artifact_hashes = (
        {str(row["path"]): str(row["sha256"])
         for label, row in artifact_rows.items()
         if label not in PROTECTED_BODY_ARTIFACT_LABELS}
        if isinstance(artifact_rows, dict)
        and all(
            isinstance(row, dict) and isinstance(row.get("path"), str)
            and isinstance(row.get("sha256"), str)
            for row in artifact_rows.values()
        ) else None
    )
    source_hashes = (
        {**artifact_hashes, **runtime_hashes}
        if isinstance(runtime_hashes, dict) and isinstance(artifact_hashes, dict)
        else None
    )
    if (
        not isinstance(runtime_hashes, dict) or not runtime_hashes
        or not isinstance(artifact_hashes, dict) or not artifact_hashes
        or not isinstance(source_hashes, dict) or not source_hashes
        or any(
            path in runtime_hashes and runtime_hashes[path] != expected
            for path, expected in artifact_hashes.items()
        )
        or inputs["worker_image"]
        != capsule["contract"].get("production_runtime", {}).get("image")
        or {name: row["sha256"] for name, row in inputs["source_bundle"]["members"].items()}
        != source_hashes
    ):
        raise LaunchPacketError("capsule runtime wiring changed")
    label = f"pg1-native-{capsule_sha[:16]}-{hashlib.sha256(attempt_run_id.encode()).hexdigest()[:12]}"
    gateway_argv = [
        str(GATEWAY), "--launch", "--capsule", str(CAPSULE_PATH),
        "--packet", str(PACKET_PATH),
    ]
    contract = {
        "kind": "PRODUCTION_CONTROL_PLANE_LAUNCH_REQUEST",
        "capsule_sha256": capsule_sha,
        "attempt_run_id": attempt_run_id,
        "offer_id": offer_id,
        "launch_label": label,
        "input_manifest_sha256": input_sha,
        "frozen_inputs": inputs,
        "source_runtime_sha256": source_hashes,
        "gateway_argv": gateway_argv,
        "prelaunch_cleanup": {
            "schema": "pg1-native-prelaunch-cleanup-intent/v1",
            "status": "MUST_ARM_BEFORE_CREATE",
            "transaction_path": str(TRANSACTION_PATH),
            "discover_by": {"label": label, "offer_id": offer_id},
            "science_released": False,
            "no_science_destroy_gate": True,
        },
        "authority_request": {
            "schema": "pg1-native-launch-authority-request/v1",
            "scope": "PG1_NATIVE_ONE_SHOT_LAUNCH",
            "issuer": ISSUER,
            "one_shot": True,
            "offer_id": offer_id,
            "capsule_sha256": capsule_sha,
            "attempt_run_id": attempt_run_id,
            "may_mint_authorization": False,
        },
        "stop_context_template": {
            "schema": "pg1-native-production-stop-context/v1",
            "issuer": "PG1_PRODUCTION_DEPLOYMENT_CONTEXT_ISSUER_V1",
            "instance_id": "BIND_FROM_TRACKED_CREATE_RESPONSE",
            "offer_id": offer_id,
            "machine_id": "BIND_FROM_CANONICAL_SHOW_ROW",
            "container_id": "C.<instance_id>",
            "capsule_sha256": capsule_sha,
            "attempt_run_id": attempt_run_id,
        },
        "mount_inventory": {
            "/run/parallax/pg1-execution-capsule.json": "0400",
            "/run/parallax/pg1-native-stop-context.json": "0400",
            "/run/parallax/aws/config": "0400",
            "/run/parallax/aws/credentials": "0400",
            "/run/secrets/pg1-native-vast-api-key": "0400",
            "/run/parallax/runtime-image-digest": "0400",
            "/run/parallax/pg1-native-entry-contract.json": "0400",
            "/run/parallax/pg1-external-deadman-arm.json": "0400",
            str(START_LEASE_PATH): "0400",
        },
        "onstart_materializer": {
            "argv": [str(ONSTART_MATERIALIZER), "--materialize", "--packet-sha256",
                     "${PACKET_SHA256}", "--attempt-run-id", attempt_run_id],
            "cwd": "/opt/parallax/image/source",
            "environment": {
                "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "PYTHONPATH": "/opt/parallax/image/source",
                "PG1_SCIENCE_DISABLED": "1",
            },
            "waits_for": str(WORKER_PACKET_PATH),
            "installs_then_invokes": str(IDLE_BOOTSTRAP),
            "maximum_prestart_seconds": inputs["maximum_prestart_seconds"],
        },
        "exact_commands": {
            "create": ["/opt/parallax/control/bin/vastai", "create", "instance",
                       str(offer_id), "--image", inputs["worker_image"], "--disk",
                       str(inputs["worker_disk_gb"]), "--label", label,
                       "--ssh", "--direct", "--onstart-cmd", "<SOURCE_MATERIALIZER>", "--raw"],
            "bind": [str(GATEWAY), "<POST_CREATE_BIND_FROM_CANONICAL_RESPONSE>"],
            "deadman": ["/opt/parallax/control/source/infra/vast/parallax/pg1_native_control_deadman.py",
                        "--control-deadman", "--capsule", str(CAPSULE_PATH),
                        "--launch-transaction", str(TRANSACTION_PATH)],
            "publish_start_lease_last": [str(GATEWAY), "<PUBLISH_BOUND_START_LEASE_LAST>"],
        },
    }
    return address(contract)


def validate_packet(packet: dict[str, Any], capsule: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(packet, dict)
        or set(packet) != {"schema", "status", "content_address", "contract"}
        or packet.get("schema") != SCHEMA
        or packet.get("status") != STATUS
        or not isinstance(packet.get("contract"), dict)
    ):
        raise LaunchPacketError("launch packet envelope changed")
    contract = packet["contract"]
    expected = hashlib.sha256(canonical_bytes(contract)).hexdigest()
    capsule_sha, attempt = _capsule_identity(capsule)
    if (
        packet.get("content_address") != {"algorithm": "sha256", "digest": expected}
        or contract.get("capsule_sha256") != capsule_sha
        or contract.get("attempt_run_id") != attempt
        or contract.get("onstart_materializer", {}).get("argv", [None] * 4)[3]
        != "${PACKET_SHA256}"
        or contract.get("authority_request", {}).get("may_mint_authorization") is not False
        or contract.get("authority_request", {}).get("issuer") != ISSUER
        or contract.get("prelaunch_cleanup", {}).get("science_released") is not False
    ):
        raise LaunchPacketError("launch packet identity changed")
    validate_inputs(contract.get("frozen_inputs"))
    rebuilt = compile_packet(
        capsule, contract["frozen_inputs"], offer_id=contract.get("offer_id"),
        attempt_run_id=attempt,
    )
    if packet != rebuilt:
        raise LaunchPacketError("launch packet differs from canonical compiler output")
    return packet


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--attempt-run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        not args.compile or args.capsule is None or args.inputs is None
        or args.offer_id is None or not args.attempt_run_id or args.output is None
    ):
        raise LaunchPacketError("packet compiler argv changed")
    packet = compile_packet(
        _fixed_json(args.capsule), _fixed_json(args.inputs, schema=INPUT_SCHEMA),
        offer_id=args.offer_id, attempt_run_id=args.attempt_run_id,
    )
    validate_packet(packet, _fixed_json(args.capsule))
    _atomic(args.output, packet)
    print(json.dumps({"status": STATUS,
                      "packet_sha256": packet["content_address"]["digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
