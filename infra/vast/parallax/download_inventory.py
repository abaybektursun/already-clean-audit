"""Resolve the frozen model download and conservative scratch projection off-GPU."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .artifacts import atomic_json
from .parallax_entities import frozen_entities


def _required(filename: str) -> bool:
    return filename.endswith((".safetensors", ".json", ".model")) and not filename.endswith(".gguf")


def build(matrix: Path, *, scratch_gb: float = 500.0) -> dict[str, Any]:
    api = HfApi()
    repos: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity in frozen_entities(matrix):
        identity = (entity.repo, entity.revision)
        if identity in seen:
            continue
        seen.add(identity)
        info = api.model_info(
            entity.repo, revision=entity.revision, files_metadata=True, timeout=30
        )
        files = [
            {"path": sibling.rfilename, "bytes": int(sibling.size or 0)}
            for sibling in info.siblings
            if _required(sibling.rfilename)
        ]
        if not files or not any(row["path"].endswith(".safetensors") for row in files):
            raise ValueError(f"no safetensor inventory for {entity.repo}@{entity.revision}")
        repos.append(
            {
                "repo": entity.repo,
                "revision": entity.revision,
                "files": files,
                "required_bytes": sum(row["bytes"] for row in files),
            }
        )

    cache_bytes = sum(row["required_bytes"] for row in repos)
    gib = 1024**3
    # Conservative before measurement: count the whole durable cache against
    # local scratch, add 25 GiB runtime and 100 GiB first-six-hour artifacts,
    # then apply the frozen 30% contingency rule.
    projected_peak_bytes = int((cache_bytes + 25 * gib + 100 * gib) * 1.30)
    scratch_bytes = int(scratch_gb * gib)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN" if projected_peak_bytes < 0.80 * scratch_bytes else "RED",
        "mode": "OFF_GPU_METADATA_ONLY",
        "repositories": repos,
        "unique_repository_count": len(repos),
        "required_cache_bytes": cache_bytes,
        "projection": {
            "scratch_gb": scratch_gb,
            "runtime_allowance_gib": 25,
            "first_six_hour_artifact_allowance_gib": 100,
            "contingency_multiplier": 1.30,
            "projected_peak_bytes": projected_peak_bytes,
            "projected_disk_fraction": projected_peak_bytes / scratch_bytes,
            "required_max_fraction": 0.80,
            "note": "Conservatively counts the persistent model cache against disposable scratch even though the cache remains on the mounted Vast volume.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--scratch-gb", type=float, default=500.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.matrix, scratch_gb=args.scratch_gb)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "required_cache_bytes": report["required_cache_bytes"]}))
    return 0 if report["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
