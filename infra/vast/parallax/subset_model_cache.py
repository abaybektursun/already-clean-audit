"""Seal a minimal model-cache manifest for an explicitly named repository set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .artifacts import atomic_json, sha256_file


def run(source: Path, output: Path, repos: list[str]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    indexed = {row["repo"]: row for row in payload["repositories"]}
    if set(repos) != set(indexed) & set(repos) or len(repos) != len(set(repos)):
        raise ValueError("requested cache repositories are missing or duplicated")
    selected = [indexed[repo] for repo in repos]
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    packet = {
        "schema_version": 1,
        "status": "GREEN",
        "provider": "vast.ai",
        "mode": "NO_SCIENCE_CACHE_SUBSET",
        "source_manifest_sha256": sha256_file(source),
        "content_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "repositories": selected,
        "repository_count": len(selected),
        "verified_file_count": sum(len(row["files"]) for row in selected),
        "verified_bytes": sum(file["bytes"] for row in selected for file in row["files"]),
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", action="append", required=True)
    args = parser.parse_args()
    packet = run(args.source.resolve(), args.output.resolve(), args.repo)
    print(json.dumps({"status": packet["status"], "verified_bytes": packet["verified_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
