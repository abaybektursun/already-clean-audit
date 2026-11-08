#!/usr/local/bin/python3
"""Issue one immutable canonical PG1 launch authorization through fixed AWS."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

try:
    from . import pg1_native_launch_packet as packet_module
    from . import pg1_native_mechanical_rebase as mechanical_rebase
except ImportError:
    from infra.vast.parallax import pg1_native_launch_packet as packet_module
    from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase


SCHEMA = "pg1-native-canonical-launch-authorization/v1"
STATUS = "ISSUED_IMMUTABLE_BY_CANONICAL_PRODUCTION_AUTHORITY"
MECHANICAL_SCHEMA = "pg1-native-mechanical-launch-authorization/v1"
MECHANICAL_STATUS = "OFFLINE_MECHANICAL_NO_PRODUCTION_AUTHORITY"
MECHANICAL_ISSUER = "PG1_TRACKED_MECHANICAL_ISSUER_V1"
ISSUER = packet_module.ISSUER
BUCKET = "pg-lineage-769063704696"
PREFIX = "parallax/authorizations/pg1-native-v3"
OWNER = "769063704696"
AWS = Path("/opt/parallax/control/bin/aws")
AWS_CONFIG = Path("/run/parallax/control-plane/aws/config")
AWS_CREDENTIALS = Path("/run/parallax/control-plane/aws/credentials")
CAPSULE = packet_module.CAPSULE_PATH
PACKET = packet_module.PACKET_PATH
RECEIPT = Path("/run/parallax/control-plane/pg1-launch-authority-receipt.json")
TIMEOUT_SECONDS = 60
CONTROL_SOURCE_ROOT = Path("/opt/parallax/control/source")


class LaunchAuthorityError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def authorization_key(packet_sha256: str) -> str:
    return f"{PREFIX}/launch-authorizations/{packet_sha256}.json"


def consumption_key(authorization_sha256: str) -> str:
    return f"{PREFIX}/launch-consumptions/{authorization_sha256}.json"


def metadata(body: dict[str, Any]) -> dict[str, str]:
    contract = body["contract"]
    return {
        "object-kind": "canonical-launch-authorization",
        "issuer": ISSUER,
        "packet-sha256": contract["packet_sha256"],
        "capsule-sha256": contract["capsule_sha256"],
        "attempt-run-id": contract["attempt_run_id"],
        "input-manifest-sha256": contract["input_manifest_sha256"],
    }


def build_authorization(
    capsule: dict[str, Any], packet: dict[str, Any], *, one_shot_nonce: str,
) -> dict[str, Any]:
    packet_module.validate_packet(packet, capsule)
    if (
        not isinstance(one_shot_nonce, str) or len(one_shot_nonce) != 64
        or any(character not in "0123456789abcdef" for character in one_shot_nonce)
    ):
        raise LaunchAuthorityError("one-shot authority nonce changed")
    packet_contract = packet["contract"]
    inputs = packet_contract["frozen_inputs"]
    contract = {
        "scope": "PG1_NATIVE_ONE_SHOT_LAUNCH",
        "issuer": ISSUER, "issuer_account": OWNER, "one_shot": True,
        "one_shot_nonce": one_shot_nonce,
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": packet_contract["attempt_run_id"],
        "offer_id": packet_contract["offer_id"],
        "launch_label": packet_contract["launch_label"],
        "source_bundle_sha256": inputs["source_bundle"]["sha256"],
        "source_bundle_version_id": inputs["source_bundle"]["version_id"],
        "runtime_image": inputs["runtime_image"]["image"],
        "runtime_receipt_sha256": inputs["runtime_image"]["receipt_sha256"],
        "input_manifest_sha256": packet_contract["input_manifest_sha256"],
        "source_runtime_manifest_sha256": digest(
            packet_contract["source_runtime_sha256"]
        ),
        "issued_key": authorization_key(packet["content_address"]["digest"]),
    }
    return {
        "schema": SCHEMA, "status": STATUS,
        "content_address": {"algorithm": "sha256", "digest": digest(contract)},
        "contract": contract,
    }


def validate_authorization(
    body: dict[str, Any], capsule: dict[str, Any], packet: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(body, dict)
        or set(body) != {"schema", "status", "content_address", "contract"}
        or body.get("schema") != SCHEMA or body.get("status") != STATUS
        or not isinstance(body.get("contract"), dict)
        or body.get("content_address") != {
            "algorithm": "sha256", "digest": digest(body["contract"])
        }
    ):
        raise LaunchAuthorityError("canonical launch authorization envelope changed")
    nonce = body["contract"].get("one_shot_nonce")
    expected = build_authorization(capsule, packet, one_shot_nonce=nonce)
    if body != expected:
        raise LaunchAuthorityError("canonical launch authorization content changed")
    return body


def build_mechanical_authorization(
    capsule: dict[str, Any], packet: dict[str, Any], *, one_shot_nonce: str,
) -> dict[str, Any]:
    packet_module.validate_packet(packet, capsule)
    if len(one_shot_nonce) != 64 or any(c not in "0123456789abcdef" for c in one_shot_nonce):
        raise LaunchAuthorityError("mechanical one-shot nonce changed")
    inputs = packet["contract"]["frozen_inputs"]
    contract = {
        "scope": "PG1_MECHANICAL_NO_AUTHORITY_LAUNCH",
        "issuer": MECHANICAL_ISSUER, "production_authority": False,
        "may_mint_production_authority": False, "provider_calls": 0,
        "one_shot": True, "one_shot_nonce": one_shot_nonce,
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": packet["contract"]["attempt_run_id"],
        "offer_id": packet["contract"]["offer_id"],
        "launch_label": packet["contract"]["launch_label"],
        "input_manifest_sha256": packet["contract"]["input_manifest_sha256"],
        "source_bundle_sha256": inputs["source_bundle"]["sha256"],
        "issued_key": authorization_key(packet["content_address"]["digest"]),
    }
    return {
        "schema": MECHANICAL_SCHEMA, "status": MECHANICAL_STATUS,
        "content_address": {"algorithm": "sha256", "digest": digest(contract)},
        "contract": contract,
    }


def validate_mechanical_authorization(
    body: dict[str, Any], capsule: dict[str, Any], packet: dict[str, Any],
) -> dict[str, Any]:
    contract = body.get("contract") if isinstance(body, dict) else None
    nonce = contract.get("one_shot_nonce") if isinstance(contract, dict) else None
    if (
        not isinstance(nonce, str)
        or body != build_mechanical_authorization(
            capsule, packet, one_shot_nonce=nonce
        )
    ):
        raise LaunchAuthorityError("mechanical launch authorization changed")
    return body


def fixed_env() -> dict[str, str]:
    return {
        "PATH": "/opt/parallax/control/bin:/usr/bin:/bin", "HOME": "/nonexistent",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "AWS_PROFILE": "pg1-launch-authority", "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_CONFIG_FILE": str(AWS_CONFIG),
        "AWS_SHARED_CREDENTIALS_FILE": str(AWS_CREDENTIALS),
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def _fixed_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise LaunchAuthorityError(f"fixed authority input changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchAuthorityError(f"fixed authority input unreadable: {path}") from error
    if not isinstance(value, dict):
        raise LaunchAuthorityError("fixed authority input must be an object")
    return value


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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, cwd=str(CONTROL_SOURCE_ROOT),
            env=fixed_env(), close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LaunchAuthorityError("fixed authority AWS command failed") from error


def issue(
    capsule: dict[str, Any], packet: dict[str, Any], *, one_shot_nonce: str,
    receipt_path: Path | None = None,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    receipt_path = RECEIPT if receipt_path is None else receipt_path
    body = build_authorization(capsule, packet, one_shot_nonce=one_shot_nonce)
    data = canonical_bytes(body)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="pg1-authority-issue-") as directory:
        source = Path(directory) / "authorization.json"; source.write_bytes(data)
        command = [
            str(AWS), "--profile", "pg1-launch-authority", "--region", "us-east-1",
            "s3api", "put-object", "--bucket", BUCKET,
            "--key", body["contract"]["issued_key"], "--body", str(source),
            "--if-none-match", "*", "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum, "--content-length", str(len(data)),
            "--content-type", "application/json", "--server-side-encryption", "AES256",
            "--metadata", json.dumps(metadata(body), sort_keys=True, separators=(",", ":")),
            "--expected-bucket-owner", OWNER,
        ]
        completed = runner(command)
    if completed.returncode != 0:
        raise LaunchAuthorityError("canonical launch authority conditional issue failed")
    try:
        response = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as error:
        raise LaunchAuthorityError("canonical launch authority response malformed") from error
    if (
        response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("VersionId") or not response.get("ETag")
    ):
        raise LaunchAuthorityError("canonical launch authority durability unverified")
    receipt = {
        "schema": "pg1-native-launch-authority-issuance-receipt/v1",
        "status": "CANONICAL_AUTHORITY_BODY_IMMUTABLE",
        "authorization_sha256": body["content_address"]["digest"],
        "key": body["contract"]["issued_key"], "version_id": str(response["VersionId"]),
        "etag": str(response["ETag"]), "checksum_sha256": checksum,
        "server_side_encryption": "AES256", "metadata": metadata(body),
    }
    _atomic(receipt_path, receipt)
    return receipt


def issue_mechanical(
    capsule: dict[str, Any], packet: dict[str, Any], *, one_shot_nonce: str,
    receipt_path: Path | None = None,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    receipt_path = RECEIPT if receipt_path is None else receipt_path
    body = build_mechanical_authorization(
        capsule, packet, one_shot_nonce=one_shot_nonce
    )
    data = canonical_bytes(body)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    mechanical_metadata = {
        "object-kind": "mechanical-launch-authorization",
        "issuer": MECHANICAL_ISSUER,
        "packet-sha256": body["contract"]["packet_sha256"],
        "capsule-sha256": body["contract"]["capsule_sha256"],
        "attempt-run-id": body["contract"]["attempt_run_id"],
    }
    with tempfile.TemporaryDirectory(prefix="pg1-mechanical-authority-") as directory:
        source = Path(directory) / "authorization.json"; source.write_bytes(data)
        completed = runner([
            str(AWS), "--profile", "pg1-launch-authority", "--region", "us-east-1",
            "s3api", "put-object", "--bucket", BUCKET,
            "--key", body["contract"]["issued_key"], "--body", str(source),
            "--if-none-match", "*", "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum, "--content-length", str(len(data)),
            "--content-type", "application/json", "--server-side-encryption", "AES256",
            "--metadata", json.dumps(mechanical_metadata, sort_keys=True, separators=(",", ":")),
            "--expected-bucket-owner", OWNER,
        ])
    if completed.returncode != 0:
        raise LaunchAuthorityError("mechanical authorization conditional issue failed")
    response = json.loads(completed.stdout or "{}")
    if (
        response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("VersionId") or not response.get("ETag")
    ):
        raise LaunchAuthorityError("mechanical authorization durability unverified")
    receipt = {
        "schema": "pg1-native-mechanical-authority-receipt/v1",
        "status": "MECHANICAL_BODY_IMMUTABLE_NO_PRODUCTION_AUTHORITY",
        "authorization_sha256": body["content_address"]["digest"],
        "key": body["contract"]["issued_key"], "version_id": str(response["VersionId"]),
        "etag": str(response["ETag"]), "checksum_sha256": checksum,
        "server_side_encryption": "AES256", "metadata": mechanical_metadata,
        "production_authority": False, "provider_calls": 0,
    }
    _atomic(receipt_path, receipt)
    return receipt


def configure_from_source_location() -> dict[str, Any] | None:
    """Derive any mechanical root from these resolved copied source bytes."""

    global AWS, AWS_CONFIG, AWS_CREDENTIALS, CAPSULE, PACKET, RECEIPT, CONTROL_SOURCE_ROOT
    discovered = mechanical_rebase.control_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    contract, _contract_path = discovered
    root = contract["root"]
    AWS = mechanical_rebase.rebased(root, Path("/opt/parallax/control/bin/aws"))
    AWS_CONFIG = mechanical_rebase.rebased(root, Path("/run/parallax/control-plane/aws/config"))
    AWS_CREDENTIALS = mechanical_rebase.rebased(
        root, Path("/run/parallax/control-plane/aws/credentials")
    )
    CAPSULE = mechanical_rebase.rebased(
        root, Path("/run/parallax/control-plane/pg1-execution-capsule.json")
    )
    PACKET = mechanical_rebase.rebased(
        root, Path("/run/parallax/control-plane/pg1-future-launch-packet.json")
    )
    RECEIPT = mechanical_rebase.rebased(
        root, Path("/run/parallax/control-plane/pg1-launch-authority-receipt.json")
    )
    CONTROL_SOURCE_ROOT = mechanical_rebase.rebased(root, Path("/opt/parallax/control/source"))
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--issue", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--one-shot-nonce")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    mechanical = configure_from_source_location()
    if (
        not args.issue or args.capsule != CAPSULE or args.packet != PACKET
        or args.receipt != RECEIPT or not args.one_shot_nonce
    ):
        raise LaunchAuthorityError("production authority issuer argv changed")
    issuer = issue_mechanical if mechanical is not None else issue
    print(json.dumps(issuer(
        _fixed_json(args.capsule), _fixed_json(args.packet),
        one_shot_nonce=args.one_shot_nonce, receipt_path=args.receipt,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
