"""Mirror one exact volume object to S3 through an ephemeral presigned PUT URL."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit


class MirrorError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mirror_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise MirrorError("object mirror spec is not frozen")
    if spec.get("scope") != "pg1_s3_object_mirror_probe_only":
        raise MirrorError("object mirror scope is not calibration-only")
    if spec.get("provider") != "vast.ai" or spec.get("object_store") != "amazon_s3":
        raise MirrorError("object mirror provider contract changed")
    if spec.get("aws_profile_control_plane_only") != "fuelos" or spec.get("region") != "us-east-1":
        raise MirrorError("object mirror AWS boundary changed")
    source = spec.get("source", {})
    source_path = PurePosixPath(str(source.get("path", "")))
    if not source_path.is_relative_to(PurePosixPath("/volume/parallax/cache/")):
        raise MirrorError("source is outside the exact model cache")
    if int(source.get("bytes", 0)) <= 0 or int(source.get("bytes", 0)) > 25_000_000:
        raise MirrorError("probe source must be one small real cache object")
    digest = str(source.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MirrorError("source SHA-256 is malformed")
    destination = spec.get("destination", {})
    expected_key = f"parallax/input-staging/sha256/{digest}/{source_path.name}"
    if (
        destination.get("bucket") != "pg-lineage-769063704696"
        or destination.get("key") != expected_key
        or destination.get("server_side_encryption") != "AES256"
        or destination.get("checksum_sha256_base64")
        != base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    ):
        raise MirrorError("destination is not the exact content-addressed object")
    if spec.get("require_destination_absent_before_upload") is not True:
        raise MirrorError("probe must begin with an absent destination object")
    durable_result = PurePosixPath(str(spec.get("durable_result", "")))
    if not durable_result.is_relative_to(PurePosixPath("/volume/parallax/object-mirror/")):
        raise MirrorError("durable result escapes the object-mirror namespace")
    if any(int(spec.get(field, -1)) != 0 for field in (
        "confirmation_items_read", "science_items_evaluated", "model_loads", "gpu_work"
    )):
        raise MirrorError("object mirror cannot read science or use a GPU")
    return spec


def validate_upload_url(url: str, spec: dict[str, Any]) -> tuple[str, str]:
    parsed = urlsplit(url.strip())
    destination = spec["destination"]
    expected_host = f"{destination['bucket']}.s3.{spec['region']}.amazonaws.com"
    expected_path = "/" + destination["key"]
    query = parse_qs(parsed.query)
    signed_headers = query.get("X-Amz-SignedHeaders", [])
    required_headers = (
        "host;x-amz-checksum-sha256;x-amz-meta-sha256;x-amz-server-side-encryption"
    )
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected_host
        or parsed.path != expected_path
        or parsed.fragment
        or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
        or signed_headers != [required_headers]
        or len(query.get("X-Amz-Signature", [""])[0]) != 64
        or int(query.get("X-Amz-Expires", ["0"])[0]) > 3600
    ):
        raise MirrorError("ephemeral PUT URL violates the frozen destination contract")
    return expected_host, parsed.path + "?" + parsed.query


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(spec: dict[str, Any], *, url_file: Path) -> dict[str, Any]:
    validate_mirror_spec(spec)
    source = Path(spec["source"]["path"])
    if not source.is_file() or source.stat().st_size != int(spec["source"]["bytes"]):
        raise MirrorError("source size is absent or changed")
    source_sha256 = sha256_file(source)
    if source_sha256 != spec["source"]["sha256"]:
        raise MirrorError("source SHA-256 changed")
    url = url_file.read_text(encoding="utf-8").strip()
    try:
        host, target = validate_upload_url(url, spec)
    finally:
        url_file.unlink(missing_ok=True)
    checksum = spec["destination"]["checksum_sha256_base64"]
    headers = {
        "Content-Length": str(source.stat().st_size),
        "x-amz-checksum-sha256": checksum,
        "x-amz-meta-sha256": source_sha256,
        "x-amz-server-side-encryption": "AES256",
    }
    started = time.monotonic()
    connection = http.client.HTTPSConnection(host, timeout=int(spec["upload_timeout_seconds"]))
    try:
        connection.putrequest("PUT", target, skip_host=True)
        connection.putheader("Host", host)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read(1_000_001)
        if response.status != 200 or len(body) > 1_000_000:
            raise MirrorError(
                f"S3 PUT failed with HTTP {response.status}; body={body[:500]!r}"
            )
        etag = response.getheader("ETag")
        checksum_response = response.getheader("x-amz-checksum-sha256")
    finally:
        connection.close()
    elapsed = time.monotonic() - started
    return {
        "schema_version": 1,
        "status": "GREEN_UPLOADED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": spec["run_id"],
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "destination": {
            "bucket": spec["destination"]["bucket"],
            "key": spec["destination"]["key"],
            "region": spec["region"],
            "server_side_encryption": "AES256",
            "checksum_sha256_base64": checksum,
            "etag": etag,
            "response_checksum_sha256": checksum_response,
        },
        "elapsed_seconds": elapsed,
        "bytes_per_second": source.stat().st_size / elapsed,
        "url_persisted": False,
        "aws_credentials_present_on_gateway": False,
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "model_loads": 0,
        "gpu_work": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    try:
        packet = run(spec, url_file=args.url_file)
    except Exception as error:
        args.url_file.unlink(missing_ok=True)
        packet = {
            "schema_version": 1,
            "status": "RED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": spec.get("run_id"),
            "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            "url_persisted": False,
            "aws_credentials_present_on_gateway": False,
            "confirmation_items_read": 0,
            "science_items_evaluated": 0,
            "model_loads": 0,
            "gpu_work": 0,
        }
    atomic_json(args.output, packet)
    return 0 if packet["status"] == "GREEN_UPLOADED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
