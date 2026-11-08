"""Hash a frozen, minimal model-cache subset on the Vast persistent volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class CacheAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise CacheAuditError("cache audit spec is not frozen")
    if spec.get("provider") != "vast.ai" or spec.get("scope") != "pg1_model_cache_integrity":
        raise CacheAuditError("cache audit scope is not the PG1 Vast preflight")
    audit_id = str(spec.get("audit_id", ""))
    if not audit_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in audit_id):
        raise CacheAuditError("cache audit id is unsafe")
    cache_root = PurePosixPath(str(spec.get("cache_root", "")))
    if not cache_root.is_absolute() or ".." in cache_root.parts:
        raise CacheAuditError("cache_root must be absolute and safe")
    files = spec.get("files")
    if not isinstance(files, list) or not files:
        raise CacheAuditError("cache audit requires a nonempty file list")
    paths: set[str] = set()
    expected_total = 0
    for row in files:
        relative = PurePosixPath(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        size = int(row.get("bytes", -1))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise CacheAuditError("cache audit file path must be relative and safe")
        if relative.as_posix() in paths:
            raise CacheAuditError("cache audit file paths must be unique")
        if size < 0 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CacheAuditError("cache audit file binding is malformed")
        paths.add(relative.as_posix())
        expected_total += size
    if int(spec.get("expected_total_bytes", -1)) != expected_total:
        raise CacheAuditError("cache audit total-byte binding disagrees with files")
    if expected_total > int(spec.get("maximum_audit_bytes", 0)):
        raise CacheAuditError("cache audit exceeds its frozen byte ceiling")
    return spec


def audit(spec: dict[str, Any]) -> dict[str, Any]:
    validate_audit_spec(spec)
    started = time.monotonic()
    cache_root = Path(spec["cache_root"])
    cache_root.resolve(strict=True)
    # Hugging Face snapshot entries are normally symlinks into the repository's
    # sibling blobs/ directory. Permit that exact repository root, not the rest
    # of the shared cache volume.
    repository_root = cache_root.parent.parent.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    verified_bytes = 0
    for expected in spec["files"]:
        relative = Path(expected["path"])
        path = cache_root / relative
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(repository_root):
                raise CacheAuditError("resolved path escapes the model repository")
            stat = resolved.stat()
            if not resolved.is_file():
                raise CacheAuditError("not a regular file")
            actual_digest = _sha256(resolved)
            actual_size = stat.st_size
            match = actual_size == int(expected["bytes"]) and actual_digest == expected["sha256"]
            if not match:
                errors.append(relative.as_posix())
            else:
                verified_bytes += actual_size
            rows.append(
                {
                    "path": relative.as_posix(),
                    "bytes": actual_size,
                    "sha256": actual_digest,
                    "expected_bytes": int(expected["bytes"]),
                    "expected_sha256": expected["sha256"],
                    "match": match,
                }
            )
        except (OSError, CacheAuditError) as error:
            errors.append(relative.as_posix())
            rows.append({"path": relative.as_posix(), "match": False, "error": str(error)[:300]})
    return {
        "schema_version": 1,
        "status": "GREEN" if not errors else "RED",
        "audit_id": spec["audit_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "files": rows,
        "file_count": len(rows),
        "verified_bytes": verified_bytes,
        "expected_total_bytes": int(spec["expected_total_bytes"]),
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_created": False,
        "model_loaded": False,
        "network_reads": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        result = audit(spec)
        exit_code = 0 if result["status"] == "GREEN" else 2
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "RED",
            "audit_id": "unresolved",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {"type": type(error).__name__, "message": str(error)[:500]},
            "verified_bytes": 0,
            "gpu_created": False,
            "model_loaded": False,
            "network_reads": 0,
        }
        exit_code = 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
