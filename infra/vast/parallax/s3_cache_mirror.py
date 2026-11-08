"""Mirror an exact frozen cache manifest from a Vast volume to versioned S3.

The controller proves every destination either absent or byte/hash exact before
launch and supplies single-use presigned PUT URLs only for absent objects.  This
worker deletes the URL file before touching source bytes, re-hashes every source,
and publishes each object with an S3 atomic PUT.  There are no automatic retries.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit


class CacheMirrorError(RuntimeError):
    pass


PG1_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
PG1_ROOT = PurePosixPath(
    "/volume/parallax/cache/huggingface/"
    "models--google--paligemma-3b-pt-224/snapshots/" + PG1_REVISION
)
BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
PREFIX = "parallax/input-staging/sha256/"
FULL_BYTES = 11_715_866_017
FULL_FILES = 12
MAX_SINGLE_PUT_BYTES = 5_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_key(row: dict[str, Any]) -> str:
    return f"{PREFIX}{row['sha256']}/{PurePosixPath(str(row['path'])).name}"


def expected_checksum(row: dict[str, Any]) -> str:
    return base64.b64encode(bytes.fromhex(str(row["sha256"]))).decode("ascii")


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise CacheMirrorError("cache mirror spec is not frozen")
    if (
        spec.get("scope") != "pg1_s3_cache_mirror"
        or spec.get("provider") != "vast.ai"
        or spec.get("object_store") != "amazon_s3"
        or spec.get("aws_profile_control_plane_only") != "fuelos"
        or spec.get("region") != REGION
        or spec.get("bucket") != BUCKET
    ):
        raise CacheMirrorError("cache mirror provider boundary changed")
    mode = spec.get("mode")
    if mode not in {"rehearsal", "full"}:
        raise CacheMirrorError("cache mirror mode must be rehearsal or full")
    if PurePosixPath(str(spec.get("source_root", ""))) != PG1_ROOT:
        raise CacheMirrorError("cache mirror source root changed")
    if spec.get("destination_prefix") != PREFIX:
        raise CacheMirrorError("cache mirror destination prefix changed")
    if spec.get("existing_object_policy") != "reuse_only_if_exact_versioned_head":
        raise CacheMirrorError("cache mirror existing-object policy changed")
    if spec.get("atomic_publication") != "s3_single_put_per_content_addressed_object":
        raise CacheMirrorError("cache mirror atomic-publication contract changed")
    if spec.get("automatic_retry") is not False:
        raise CacheMirrorError("cache mirror cannot retry automatically")
    parallel = int(spec.get("parallel_streams", 0))
    if not 1 <= parallel <= 3:
        raise CacheMirrorError("cache mirror parallelism must be in [1, 3]")
    timeout = int(spec.get("per_file_timeout_seconds", 0))
    if not 60 <= timeout <= 1800:
        raise CacheMirrorError("cache mirror per-file timeout is unsafe")
    rows = spec.get("files")
    if not isinstance(rows, list) or not rows:
        raise CacheMirrorError("cache mirror requires files")
    paths: set[str] = set()
    keys: set[str] = set()
    total = 0
    absent = 0
    existing = 0
    for row in rows:
        relative = PurePosixPath(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        size = int(row.get("bytes", -1))
        key = str(row.get("destination_key", ""))
        checksum = str(row.get("checksum_sha256_base64", ""))
        initial = row.get("expected_initial_state")
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise CacheMirrorError("cache mirror file path is unsafe")
        if relative.as_posix() in paths or key in keys:
            raise CacheMirrorError("cache mirror paths and keys must be unique")
        if size <= 0 or size > MAX_SINGLE_PUT_BYTES:
            raise CacheMirrorError("cache mirror file exceeds single-PUT boundary")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CacheMirrorError("cache mirror digest is malformed")
        if key != expected_key(row) or checksum != expected_checksum(row):
            raise CacheMirrorError("cache mirror content address is incorrect")
        if initial == "absent":
            if row.get("existing_version_id") is not None:
                raise CacheMirrorError("absent object cannot bind a version")
            absent += 1
        elif initial == "exact_existing":
            if not str(row.get("existing_version_id", "")):
                raise CacheMirrorError("existing object lacks immutable version")
            existing += 1
        else:
            raise CacheMirrorError("cache mirror initial state is invalid")
        paths.add(relative.as_posix())
        keys.add(key)
        total += size
    if int(spec.get("expected_total_bytes", -1)) != total:
        raise CacheMirrorError("cache mirror byte total disagrees with files")
    if total > int(spec.get("maximum_total_bytes", 0)):
        raise CacheMirrorError("cache mirror exceeds byte ceiling")
    if absent < 1:
        raise CacheMirrorError("cache mirror must publish at least one absent object")
    if mode == "rehearsal":
        if not 2 <= len(rows) <= 9 or total > 25_000_000 or existing < 1:
            raise CacheMirrorError("rehearsal must test upload and exact reuse under 25 MB")
    elif len(rows) != FULL_FILES or total != FULL_BYTES:
        raise CacheMirrorError("full mirror must bind the exact 12-file PG1 snapshot")
    durable = PurePosixPath(str(spec.get("durable_result", "")))
    if not durable.is_relative_to(PurePosixPath("/volume/parallax/cache-mirror/")):
        raise CacheMirrorError("cache mirror result escapes its durable namespace")
    if any(int(spec.get(field, -1)) != 0 for field in (
        "confirmation_items_read", "science_items_evaluated", "model_loads", "gpu_work"
    )):
        raise CacheMirrorError("cache mirror cannot read science or use a GPU")
    return spec


def validate_put_url(url: str, row: dict[str, Any]) -> tuple[str, str]:
    parsed = urlsplit(url.strip())
    host = f"{BUCKET}.s3.{REGION}.amazonaws.com"
    query = parse_qs(parsed.query)
    required_headers = (
        "host;x-amz-checksum-sha256;x-amz-meta-sha256;x-amz-server-side-encryption"
    )
    if (
        parsed.scheme != "https"
        or parsed.netloc != host
        or parsed.path != "/" + row["destination_key"]
        or parsed.fragment
        or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
        or query.get("X-Amz-SignedHeaders") != [required_headers]
        or len(query.get("X-Amz-Signature", [""])[0]) != 64
        or int(query.get("X-Amz-Expires", ["0"])[0]) > 3600
    ):
        raise CacheMirrorError("cache mirror PUT URL violates its object binding")
    return host, parsed.path + "?" + parsed.query


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _upload(row: dict[str, Any], *, root: Path, url: str, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    path = root / str(row["path"])
    resolved = path.resolve(strict=True)
    repository_root = root.parent.parent.resolve(strict=True)
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise CacheMirrorError("source escapes the frozen repository cache")
    if resolved.stat().st_size != int(row["bytes"]):
        raise CacheMirrorError("source size changed")
    actual = sha256_file(resolved)
    if actual != row["sha256"]:
        raise CacheMirrorError("source hash changed")
    if row["expected_initial_state"] == "exact_existing":
        return {
            "path": row["path"], "bytes": row["bytes"], "sha256": actual,
            "destination_key": row["destination_key"],
            "version_id": row["existing_version_id"],
            "status": "REUSED_EXACT", "elapsed_seconds": time.monotonic() - started,
        }
    if not url:
        raise CacheMirrorError("absent object lacks its ephemeral PUT URL")
    host, target = validate_put_url(url, row)
    headers = {
        "Content-Length": str(row["bytes"]),
        "x-amz-checksum-sha256": row["checksum_sha256_base64"],
        "x-amz-meta-sha256": row["sha256"],
        "x-amz-server-side-encryption": "AES256",
    }
    connection = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        connection.putrequest("PUT", target, skip_host=True)
        connection.putheader("Host", host)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read(1_000_001)
        if response.status != 200 or len(body) > 1_000_000:
            raise CacheMirrorError(f"S3 PUT failed with HTTP {response.status}")
        response_checksum = response.getheader("x-amz-checksum-sha256")
        if response_checksum != row["checksum_sha256_base64"]:
            raise CacheMirrorError("S3 response checksum disagrees")
        etag = response.getheader("ETag")
        version_id = response.getheader("x-amz-version-id")
        if not version_id:
            raise CacheMirrorError("versioned bucket returned no version ID")
    finally:
        connection.close()
    elapsed = time.monotonic() - started
    return {
        "path": row["path"], "bytes": row["bytes"], "sha256": actual,
        "destination_key": row["destination_key"], "version_id": version_id,
        "etag": etag, "response_checksum_sha256": response_checksum,
        "status": "GREEN_UPLOADED", "elapsed_seconds": elapsed,
        "bytes_per_second": int(row["bytes"]) / elapsed,
    }


def run(spec: dict[str, Any], *, url_file: Path) -> dict[str, Any]:
    validate_spec(spec)
    raw_urls = json.loads(url_file.read_text(encoding="utf-8"))
    url_file.unlink(missing_ok=True)
    if not isinstance(raw_urls, dict):
        raise CacheMirrorError("URL secret must be a key-to-URL object")
    expected_url_keys = {
        row["destination_key"] for row in spec["files"]
        if row["expected_initial_state"] == "absent"
    }
    if set(raw_urls) != expected_url_keys:
        raise CacheMirrorError("ephemeral URL set disagrees with absent objects")
    started = time.monotonic()
    root = Path(spec["source_root"])
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(spec["parallel_streams"])) as pool:
        futures = {
            pool.submit(
                _upload, row, root=root,
                url=str(raw_urls.get(row["destination_key"], "")),
                timeout=int(spec["per_file_timeout_seconds"]),
            ): row
            for row in spec["files"]
        }
        raw_urls.clear()
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({
                    "path": str(row["path"]), "type": type(error).__name__,
                    "message": str(error)[:500],
                })
    results.sort(key=lambda row: str(row["path"]))
    elapsed = time.monotonic() - started
    uploaded_bytes = sum(
        int(row["bytes"]) for row in results if row["status"] == "GREEN_UPLOADED"
    )
    return {
        "schema_version": 1,
        "status": "GREEN_MIRRORED" if not failures else "RED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "files": results,
        "failures": failures,
        "file_count": len(results),
        "expected_file_count": len(spec["files"]),
        "expected_total_bytes": int(spec["expected_total_bytes"]),
        "uploaded_bytes": uploaded_bytes,
        "reused_bytes": sum(
            int(row["bytes"]) for row in results if row["status"] == "REUSED_EXACT"
        ),
        "elapsed_seconds": elapsed,
        "aggregate_uploaded_bytes_per_second": uploaded_bytes / elapsed if elapsed else 0,
        "url_file_deleted_before_source_reads": True,
        "url_persisted": False,
        "aws_credentials_present_on_gateway": False,
        "automatic_retry": False,
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "model_loads": 0,
        "gpu_work": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
            "schema_version": 1, "status": "RED", "run_id": spec.get("run_id"),
            "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            "url_persisted": False, "aws_credentials_present_on_gateway": False,
            "automatic_retry": False, "confirmation_items_read": 0,
            "science_items_evaluated": 0, "model_loads": 0, "gpu_work": 0,
        }
    atomic_json(args.output, packet)
    return 0 if packet["status"] == "GREEN_MIRRORED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
