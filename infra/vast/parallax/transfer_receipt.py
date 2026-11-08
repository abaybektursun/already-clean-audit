"""Build and verify exact small-input transfers from durable Vast storage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .artifacts import atomic_json, iter_files, sha256_file


REQUIRED_TRANSFER_SCOPE = {
    "runtime_bundle",
    "source_snapshot",
    "readiness_manifests",
    "task_data",
}


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"transfer input is outside root: {path}") from error


def build_manifest(root: Path, scoped_paths: dict[str, Iterable[Path]]) -> dict[str, object]:
    if set(scoped_paths) != REQUIRED_TRANSFER_SCOPE:
        raise ValueError(f"transfer scope must be exactly {sorted(REQUIRED_TRANSFER_SCOPE)}")
    entries: list[dict[str, object]] = []
    for scope in sorted(scoped_paths):
        paths = list(scoped_paths[scope])
        if not paths:
            raise ValueError(f"empty transfer scope: {scope}")
        for path in iter_files(paths):
            entries.append({
                "scope": scope,
                "path": _relative(root, path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    entries.sort(key=lambda row: (str(row["scope"]), str(row["path"])))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "payload_scope": sorted(REQUIRED_TRANSFER_SCOPE),
        "entry_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "entries": entries,
    }


def verify_inbound(
    manifest_path: Path,
    restored_root: Path,
    *,
    destination_instance_id: int,
    cache_contract_manifest_sha256: str,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "GREEN":
        raise ValueError("inbound transfer manifest is not GREEN")
    if set(manifest.get("payload_scope", [])) != REQUIRED_TRANSFER_SCOPE:
        raise ValueError("inbound transfer manifest has incomplete or extra scope")
    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError("inbound transfer manifest has no entries")
    errors: list[str] = []
    for row in entries:
        path = restored_root / row["path"]
        if not path.is_file():
            errors.append(f"missing: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            errors.append(f"size: {row['path']}")
        elif sha256_file(path) != row["sha256"]:
            errors.append(f"sha256: {row['path']}")
    if errors:
        raise IOError(f"inbound transfer verification failed: {errors[:10]}")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "provider": "vast.ai",
        "direction": "durable_to_h100_scratch",
        "transport": "direct_ssh_scp_with_precreated_ephemeral_account_key",
        "destination_instance_id": destination_instance_id,
        "payload_scope": sorted(REQUIRED_TRANSFER_SCOPE),
        "source_manifest_sha256": sha256_file(manifest_path),
        "cache_contract_manifest_sha256": cache_contract_manifest_sha256,
        "verified_entry_count": len(entries),
        "verified_bytes": sum(int(row["bytes"]) for row in entries),
    }


def _scopes(values: list[str], root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--scope must have NAME=PATH form")
        name, raw_path = value.split("=", 1)
        result.setdefault(name, []).append(root / raw_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--scope", action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--restored-root", type=Path, required=True)
    verify.add_argument("--destination-instance-id", type=int, required=True)
    verify.add_argument("--cache-contract-manifest-sha256", required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build_manifest(args.root, _scopes(args.scope, args.root))
    else:
        report = verify_inbound(
            args.manifest,
            args.restored_root,
            destination_instance_id=args.destination_instance_id,
            cache_contract_manifest_sha256=args.cache_contract_manifest_sha256,
        )
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
