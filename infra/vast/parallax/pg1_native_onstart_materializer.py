#!/usr/local/bin/python3
"""Pinned-image onstart source materializer for a future PG1 worker."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any


POINTER = Path("/run/parallax/pg1-onstart-materializer-pointer.json")
PACKET = Path("/run/parallax/pg1-future-launch-packet.json")
AWS_CONFIG = Path("/run/parallax/aws/config")
AWS_CREDENTIALS = Path("/run/parallax/aws/credentials")
AWS = Path("/usr/local/bin/aws")
SOURCE_ROOT = Path("/opt/parallax/image/source")
IDLE_RELATIVE = Path("infra/vast/parallax/pg1_native_idle_bootstrap.py")
INSTALLED_RECEIPT = Path("/run/parallax/pg1-native-installed-source-layout.json")
POLL_SECONDS = 1
MECHANICAL_WORKER_ROOT: Path | None = None


class OnstartMaterializerError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def configure_from_source_location() -> dict[str, Any] | None:
    """Derive a mechanical worker root from this resolved installed executable."""

    global POINTER, PACKET, AWS_CONFIG, AWS_CREDENTIALS, AWS, SOURCE_ROOT
    global INSTALLED_RECEIPT, POLL_SECONDS, MECHANICAL_WORKER_ROOT
    resolved = Path(__file__).resolve()
    token = "/run/parallax/"
    if token not in str(resolved):
        raise OnstartMaterializerError("materializer source is outside canonical layout")
    prefix = str(resolved).split(token, 1)[0]
    worker = Path(prefix or "/").resolve()
    if worker == Path("/"):
        return None
    contract_path = worker / "run/parallax/pg1-native-mechanical-contract.json"
    if (
        contract_path.is_symlink() or not contract_path.is_file()
        or stat.S_IMODE(contract_path.stat().st_mode) != 0o400
    ):
        raise OnstartMaterializerError("offline root contract is absent or mutable")
    try:
        value = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OnstartMaterializerError("offline root contract unreadable") from error
    body = value.get("contract") if isinstance(value, dict) else None
    if (
        not isinstance(body, dict)
        or value.get("schema") != "pg1-native-exact-chain-mechanical-contract/v1"
        or value.get("status") != "OFFLINE_DETERMINISTIC_FAKE_SERVICES_ONLY"
        or value.get("content_address") != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(body)).hexdigest(),
        }
        or body.get("kind") != "MECHANICAL_NO_AUTHORITY"
        or body.get("production_authority") is not False
        or body.get("provider_calls") != 0
        or body.get("fake_services") != "TRACKED_DETERMINISTIC_EXECUTABLES"
        or body.get("network_scope") != "LOOPBACK_OR_NONE"
        or body.get("may_mint_production_authority") is not False
    ):
        raise OnstartMaterializerError("offline root contract changed")
    root = Path(str(body.get("root_rebase")))
    if not root.is_absolute() or root == Path("/") or root.is_symlink() or not root.is_dir():
        raise OnstartMaterializerError("offline root changed")
    if root.resolve() / ".pg1-fake-services/worker" != worker:
        raise OnstartMaterializerError("derived worker root and contract root differ")
    def at(path: str) -> Path:
        return worker / Path(path).relative_to("/")
    POINTER = at("/run/parallax/pg1-onstart-materializer-pointer.json")
    PACKET = at("/run/parallax/pg1-future-launch-packet.json")
    AWS_CONFIG = at("/run/parallax/aws/config")
    AWS_CREDENTIALS = at("/run/parallax/aws/credentials")
    AWS = at("/usr/local/bin/aws")
    SOURCE_ROOT = at("/opt/parallax/image/source")
    INSTALLED_RECEIPT = at("/run/parallax/pg1-native-installed-source-layout.json")
    POLL_SECONDS = 0.01
    MECHANICAL_WORKER_ROOT = worker
    return value


def _atomic(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, mode); os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def pointer_first(raw_argv: list[str]) -> None:
    _atomic(
        POINTER,
        canonical_bytes({
            "schema": "pg1-native-onstart-source-pointer/v1",
            "status": "SCIENCE_DISABLED_WAITING_FOR_EXACT_SOURCE",
            "argv_sha256": hashlib.sha256(canonical_bytes(raw_argv)).hexdigest(),
            "science_released": False, "wall_time_unix": time.time(),
        }),
        mode=0o400,
    )


def _fixed(path: Path, mode: int = 0o400) -> bytes:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != mode:
        raise OnstartMaterializerError(f"fixed onstart material changed: {path}")
    return path.read_bytes()


def _packet(packet_sha256: str, attempt_run_id: str) -> dict[str, Any]:
    try:
        packet = json.loads(_fixed(PACKET))
    except json.JSONDecodeError as error:
        raise OnstartMaterializerError("worker launch packet unreadable") from error
    contract = packet.get("contract") if isinstance(packet, dict) else None
    if (
        not isinstance(contract, dict)
        or packet.get("content_address") != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        }
        or packet["content_address"]["digest"] != packet_sha256
        or contract.get("attempt_run_id") != attempt_run_id
        or contract.get("prelaunch_cleanup", {}).get("science_released") is not False
    ):
        raise OnstartMaterializerError("worker launch packet identity changed")
    return packet


def _aws_env() -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "AWS_PROFILE": "fuelos",
        "AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": str(AWS_CONFIG),
        "AWS_SHARED_CREDENTIALS_FILE": str(AWS_CREDENTIALS),
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def _download_source(packet: dict[str, Any], target: Path) -> bytes:
    source = packet["contract"]["frozen_inputs"]["source_bundle"]
    command = [
        str(AWS), "--profile", "fuelos", "--region", "us-east-1",
        "s3api", "get-object", "--bucket", source["bucket"], "--key", source["key"],
        "--version-id", source["version_id"], "--checksum-mode", "ENABLED", str(target),
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=120,
            cwd="/", env=_aws_env(), close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OnstartMaterializerError("source bundle download failed") from error
    if completed.returncode != 0:
        raise OnstartMaterializerError("source bundle download returned nonzero")
    data = target.read_bytes()
    try:
        response = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as error:
        raise OnstartMaterializerError("source bundle response malformed") from error
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    if (
        response.get("VersionId") != source["version_id"]
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or response.get("ContentLength") != len(data)
        or len(data) != source["bytes"]
        or hashlib.sha256(data).hexdigest() != source["sha256"]
    ):
        raise OnstartMaterializerError("source bundle provenance changed")
    return data


def _safe_relative(text: str) -> Path:
    value = Path(text)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise OnstartMaterializerError("source bundle member escaped fixed root")
    return value


def install_source(packet: dict[str, Any], archive: Path) -> dict[str, Any]:
    contract = packet["contract"]
    source = contract["frozen_inputs"]["source_bundle"]
    members = source.get("members")
    runtime = contract.get("source_runtime_sha256")
    if (
        not isinstance(members, dict) or not members
        or not isinstance(runtime, dict) or set(members) != set(runtime)
    ):
        raise OnstartMaterializerError("source member inventory changed")
    installed: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, "r:*") as handle:
        rows = handle.getmembers()
        file_rows = [row for row in rows if row.isfile()]
        if any(
            row.issym() or row.islnk() or row.isdev()
            or row.name.startswith("/") or ".." in Path(row.name).parts
            for row in rows
        ) or {row.name for row in file_rows} != set(members):
            raise OnstartMaterializerError("source archive member set changed")
        for row in file_rows:
            relative = _safe_relative(row.name)
            expected = members[row.name]
            stream = handle.extractfile(row)
            if stream is None:
                raise OnstartMaterializerError("source archive member unreadable")
            data = stream.read()
            mode = stat.S_IMODE(row.mode)
            if (
                not isinstance(expected, dict)
                or expected != {
                    "sha256": runtime[row.name], "bytes": len(data),
                    "mode": f"{mode:04o}",
                }
                or hashlib.sha256(data).hexdigest() != runtime[row.name]
                or mode not in {0o644, 0o755}
            ):
                raise OnstartMaterializerError(f"source member provenance changed: {row.name}")
            target = SOURCE_ROOT / relative
            _atomic(target, data, mode=mode)
            if (
                target.is_symlink() or stat.S_IMODE(target.stat().st_mode) != mode
                or hashlib.sha256(target.read_bytes()).hexdigest() != runtime[row.name]
            ):
                raise OnstartMaterializerError(
                    f"installed source member changed: {row.name}"
                )
            installed[row.name] = expected
    return {
        "schema": "pg1-native-installed-source-layout/v1",
        "packet_sha256": packet["content_address"]["digest"],
        "source_bundle_sha256": source["sha256"], "members": installed,
        "science_released": False,
    }


def run(*, packet_sha256: str, attempt_run_id: str, maximum_seconds: int) -> int:
    deadline = time.monotonic() + maximum_seconds
    while time.monotonic() < deadline:
        if PACKET.is_file() and AWS_CONFIG.is_file() and AWS_CREDENTIALS.is_file():
            packet = _packet(packet_sha256, attempt_run_id)
            with tempfile.TemporaryDirectory(prefix="pg1-source-materialize-") as directory:
                archive = Path(directory) / "source.tar"
                _download_source(packet, archive)
                receipt = install_source(packet, archive)
            _atomic(INSTALLED_RECEIPT, canonical_bytes(receipt), mode=0o400)
            idle = SOURCE_ROOT / IDLE_RELATIVE
            expected_idle = packet["contract"]["source_runtime_sha256"][str(IDLE_RELATIVE)]
            if (
                idle.is_symlink() or not idle.is_file()
                or stat.S_IMODE(idle.stat().st_mode) != 0o755
                or hashlib.sha256(idle.read_bytes()).hexdigest() != expected_idle
            ):
                raise OnstartMaterializerError("installed idle bootstrap changed")
            command = [str(idle), "--idle-prestart", "--packet-sha256", packet_sha256,
                       "--attempt-run-id", attempt_run_id, "--maximum-seconds",
                       str(maximum_seconds)]
            completed = subprocess.run(
                command, check=False, cwd=str(SOURCE_ROOT), env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
                    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                    "PYTHONPATH": str(SOURCE_ROOT),
                }, close_fds=True,
            )
            if completed.returncode != 0:
                raise OnstartMaterializerError("installed idle bootstrap returned nonzero")
            return 0
        time.sleep(POLL_SECONDS)
    raise OnstartMaterializerError("source materialization deadline expired")


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else __import__("sys").argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--packet-sha256")
    parser.add_argument("--attempt-run-id")
    parser.add_argument("--maximum-seconds", type=int, default=300)
    args = parser.parse_args(raw)
    configure_from_source_location()
    pointer_first(raw)
    if (
        not args.materialize or not isinstance(args.packet_sha256, str)
        or len(args.packet_sha256) != 64 or not args.attempt_run_id
        or not 30 <= args.maximum_seconds <= 900
    ):
        raise OnstartMaterializerError("onstart materializer argv changed after pointer")
    return run(packet_sha256=args.packet_sha256,
               attempt_run_id=args.attempt_run_id,
               maximum_seconds=args.maximum_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
