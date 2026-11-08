"""Hydrate exact model bytes on H100 scratch before the cache gate.

This is a no-science staging action. It is deliberately separate from the
preflight: once preflight writes the cache-gate marker, this module refuses all
network mutation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from .artifacts import atomic_json, sha256_file
from .cache_gate import CACHE_GATE_MARKER, HYDRATION_ORIGIN


ACK = "HYDRATE_H100_CACHE_BEFORE_GATE"


def hydrate(
    manifest_path: Path,
    cache_root: Path,
    *,
    destination_instance_id: int,
    acknowledge: str,
    downloader: Callable[..., str] = snapshot_download,
) -> dict[str, object]:
    if acknowledge != ACK:
        raise PermissionError(f"refusing H100 cache mutation without --acknowledge {ACK}")
    if destination_instance_id <= 0:
        raise ValueError("destination_instance_id must be a positive Vast instance ID")
    gate = cache_root / CACHE_GATE_MARKER
    if gate.exists():
        raise PermissionError(f"cache gate already exists; network mutation is forbidden: {gate}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("status") == "GREEN"
        and manifest.get("provider") == "vast.ai"
        and manifest.get("content_manifest_sha256")
    ):
        raise ValueError("durable model-cache manifest is not a GREEN Vast contract")

    verified_bytes = 0
    verified_files = 0
    cache_root.mkdir(parents=True, exist_ok=True)
    for repo in manifest.get("repositories", []):
        allow_patterns = [row["path"] for row in repo["files"]]
        snapshot = Path(
            downloader(
                repo_id=repo["repo"],
                revision=repo["revision"],
                cache_dir=cache_root,
                allow_patterns=allow_patterns,
            )
        )
        for expected in repo["files"]:
            path = snapshot / expected["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(expected["bytes"]):
                raise IOError(f"size mismatch: {repo['repo']}:{expected['path']}")
            if sha256_file(path) != expected["sha256"]:
                raise IOError(f"sha256 mismatch: {repo['repo']}:{expected['path']}")
            verified_bytes += path.stat().st_size
            verified_files += 1

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "mode": "NO_SCIENCE_CACHE_HYDRATION",
        "provider": "vast.ai",
        "origin": HYDRATION_ORIGIN,
        "destination_instance_id": destination_instance_id,
        "content_manifest_sha256": manifest["content_manifest_sha256"],
        "repository_count": len(manifest.get("repositories", [])),
        "verified_file_count": verified_files,
        "verified_bytes": verified_bytes,
        "cache_gate_present_during_hydration": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--destination-instance-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    report = hydrate(
        args.manifest,
        args.cache_root,
        destination_instance_id=args.destination_instance_id,
        acknowledge=args.acknowledge,
    )
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "verified_bytes": report["verified_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
