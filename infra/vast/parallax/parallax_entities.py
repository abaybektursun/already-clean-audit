"""Frozen model entities and cache contracts for PARALLAX preflight.

This module deliberately performs no network access.  The H100 preflight may
inspect an already populated durable cache, but it may never download weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelEntity:
    lineage: str
    role: str
    repo: str
    revision: str

    @property
    def cache_slug(self) -> str:
        return "models--" + self.repo.replace("/", "--")

    def snapshot_path(self, cache_root: Path) -> Path:
        return cache_root / self.cache_slug / "snapshots" / self.revision


def frozen_entities(matrix_path: Path) -> list[ModelEntity]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "frozen":
        raise ValueError("model entities require a frozen matrix")
    entities: list[ModelEntity] = []
    for lineage, row in matrix["lineages"].items():
        for role in ("vision_parent_candidate", "language_parent", "fused"):
            ref = row[role]
            entities.append(ModelEntity(lineage, role, ref["repo"], ref["revision"]))
        for index, ref in enumerate(row.get("later", [])):
            entities.append(ModelEntity(lineage, f"later_{index}", ref["repo"], ref["revision"]))
    for entity in entities:
        if len(entity.revision) != 40 or any(c not in "0123456789abcdef" for c in entity.revision):
            raise ValueError(f"checkpoint revision is not a full SHA: {entity.repo}")
    return entities


def cache_inventory(matrix_path: Path, cache_root: Path) -> dict[str, Any]:
    rows = []
    for entity in frozen_entities(matrix_path):
        snapshot = entity.snapshot_path(cache_root)
        files = sorted(path for path in snapshot.rglob("*") if path.is_file()) if snapshot.is_dir() else []
        rows.append(
            {
                **asdict(entity),
                "snapshot": str(snapshot),
                "present": snapshot.is_dir(),
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "has_config": (snapshot / "config.json").is_file(),
            }
        )
    return {
        "entities": rows,
        "all_present": all(row["present"] and row["has_config"] for row in rows),
        "total_bytes": sum(row["bytes"] for row in rows),
    }
