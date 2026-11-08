#!/usr/bin/env python3
"""Tracked deterministic file-backed AWS CLI subset for offline PG1 proof only."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".pg1-fake-services"
        if candidate.is_dir():
            return candidate
    raise SystemExit("tracked fake AWS service root absent")


ROOT = _root()
OBJECTS = ROOT / "aws-objects"
LOCK = ROOT / "aws.lock"
CONTROL = ROOT / "control.json"


def _arg(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit(f"fake AWS missing {name}") from error


def _object_dir(bucket: str, key: str) -> Path:
    identity = hashlib.sha256(f"{bucket}\n{key}".encode()).hexdigest()
    return OBJECTS / identity


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _fault(operation: str, key: str, *, after_side_effect: bool = False) -> str | None:
    control = _json(CONTROL, {"faults": []})
    for row in control.get("faults", []):
        if (
            row.get("operation") == operation
            and str(row.get("key_contains", "")) in key
            and int(row.get("remaining", 0)) > 0
            and bool(row.get("after_side_effect", False)) == after_side_effect
        ):
            row["remaining"] = int(row["remaining"]) - 1
            CONTROL.write_text(json.dumps(control, sort_keys=True), encoding="utf-8")
            return str(row.get("mode"))
    return None


def _fail(mode: str | None) -> None:
    if mode == "absent":
        print("An error occurred (NoSuchKey) when calling operation: Not Found 404", file=sys.stderr)
        raise SystemExit(254)
    if mode in {"unavailable", "success_then_timeout"}:
        print("deterministic fake service unavailable", file=sys.stderr)
        raise SystemExit(253)


def _put(bucket: str, key: str) -> None:
    _fail(_fault("put-object", key))
    directory = _object_dir(bucket, key); directory.mkdir(parents=True, exist_ok=True)
    latest_path = directory / "latest.json"
    latest = _json(latest_path, None)
    if "--if-none-match" in sys.argv and latest is not None:
        print("PreconditionFailed", file=sys.stderr); raise SystemExit(255)
    if "--if-match" in sys.argv and (
        latest is None or latest.get("ETag") != _arg("--if-match")
    ):
        print("PreconditionFailed", file=sys.stderr); raise SystemExit(255)
    source = Path(_arg("--body")); data = source.read_bytes()
    sha = hashlib.sha256(data).digest()
    checksum = base64.b64encode(sha).decode("ascii")
    if _arg("--checksum-sha256") != checksum or int(_arg("--content-length")) != len(data):
        print("BadDigest", file=sys.stderr); raise SystemExit(255)
    counter_path = ROOT / "aws-counter"
    counter = int(counter_path.read_text() if counter_path.exists() else "0") + 1
    counter_path.write_text(str(counter), encoding="utf-8")
    version = f"fake-version-{counter:08d}"
    etag = f'"{hashlib.md5(data, usedforsecurity=False).hexdigest()}"'
    body_path = directory / f"{version}.body"; shutil.copyfile(source, body_path)
    record = {
        "Bucket": bucket, "Key": key, "VersionId": version, "ETag": etag,
        "ChecksumSHA256": checksum, "ServerSideEncryption": _arg("--server-side-encryption"),
        "ContentLength": len(data), "ContentType": _arg("--content-type"),
        "Metadata": json.loads(_arg("--metadata")), "body": str(body_path),
    }
    (directory / f"{version}.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    _fail(_fault("put-object", key, after_side_effect=True))
    print(json.dumps({field: record[field] for field in (
        "VersionId", "ETag", "ChecksumSHA256", "ServerSideEncryption"
    )}, sort_keys=True))


def _record(bucket: str, key: str) -> dict[str, Any]:
    directory = _object_dir(bucket, key)
    if "--version-id" in sys.argv:
        path = directory / f"{_arg('--version-id')}.json"
    else:
        path = directory / "latest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail("absent"); raise AssertionError


def _head(bucket: str, key: str) -> None:
    _fail(_fault("head-object", key))
    record = _record(bucket, key)
    output = {field: record[field] for field in (
        "VersionId", "ETag", "ServerSideEncryption", "ContentLength",
        "ContentType", "Metadata",
    )}
    mode = _fault("head-object", key, after_side_effect=True)
    if mode == "contradictory":
        output["Metadata"] = {"object-kind": "contradictory"}
    else:
        _fail(mode)
    print(json.dumps(output, sort_keys=True))


def _get(bucket: str, key: str) -> None:
    _fail(_fault("get-object", key))
    record = _record(bucket, key)
    target = Path(sys.argv[-1]); target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record["body"], target)
    output = {field: record[field] for field in (
        "VersionId", "ETag", "ChecksumSHA256", "ServerSideEncryption",
        "ContentLength", "ContentType", "Metadata",
    )}
    mode = _fault("get-object", key, after_side_effect=True)
    if mode == "contradictory":
        output["ChecksumSHA256"] = "contradictory"
    else:
        _fail(mode)
    print(json.dumps(output, sort_keys=True))


def main() -> int:
    if "s3api" not in sys.argv:
        raise SystemExit("fake AWS accepts s3api only")
    operation = sys.argv[sys.argv.index("s3api") + 1]
    bucket = _arg("--bucket"); key = _arg("--key")
    LOCK.touch(exist_ok=True)
    with LOCK.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if operation == "put-object": _put(bucket, key)
        elif operation == "head-object": _head(bucket, key)
        elif operation == "get-object": _get(bucket, key)
        else: raise SystemExit(f"fake AWS operation rejected: {operation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
