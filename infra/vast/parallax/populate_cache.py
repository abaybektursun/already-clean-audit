"""Populate and verify the frozen model cache on a cheap Vast gateway."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

from .artifacts import atomic_json, sha256_file


ACK = "POPULATE_DURABLE_CACHE_ON_GATEWAY"


def populate(inventory_path: Path, cache_root: Path, *, acknowledge: str) -> dict[str, object]:
    if acknowledge != ACK:
        raise PermissionError(f"refusing cache mutation without --acknowledge {ACK}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "GREEN":
        raise ValueError("download inventory is not GREEN")
    rows: list[dict[str, object]] = []
    for repo in inventory["repositories"]:
        allow = [row["path"] for row in repo["files"]]
        snapshot = Path(
            snapshot_download(
                repo_id=repo["repo"],
                revision=repo["revision"],
                cache_dir=cache_root,
                allow_patterns=allow,
            )
        )
        files = []
        for expected in repo["files"]:
            path = snapshot / expected["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(expected["bytes"]):
                raise IOError(f"size mismatch: {path}")
            files.append(
                {
                    "path": expected["path"],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        rows.append(
            {
                "repo": repo["repo"],
                "revision": repo["revision"],
                "snapshot": str(snapshot),
                "files": files,
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "provider": "vast.ai",
        "location": "persistent_volume",
        "repository_count": len(rows),
        "verified_bytes": sum(file["bytes"] for row in rows for file in row["files"]),
        "content_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "repositories": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    report = populate(args.inventory, args.cache_root, acknowledge=args.acknowledge)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "verified_bytes": report["verified_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
