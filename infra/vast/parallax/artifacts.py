"""Content-addressed, atomically written artifact manifests.

The durable-store transport is deliberately separate from this module. This
module proves what bytes were produced and whether restored bytes are exact;
the transport may be direct SSH or a future object store. PARALLAX forbids the
Vast copy helper after a false-success rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(path)


@dataclass
class ArtifactManifest:
    root: Path
    stage_id: str
    entries: list[dict[str, object]] = field(default_factory=list)
    schema_version: int = 1

    def add(self, *paths: Path) -> None:
        root = self.root.resolve()
        for path in iter_files(paths):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"artifact is outside manifest root: {resolved}") from error
            stat = resolved.stat()
            self.entries.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(resolved),
                    "bytes": stat.st_size,
                }
            )
        self.entries.sort(key=lambda item: str(item["path"]))

    @property
    def total_bytes(self) -> int:
        return sum(int(item["bytes"]) for item in self.entries)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "root_name": self.root.name,
            "entry_count": len(self.entries),
            "total_bytes": self.total_bytes,
            "entries": self.entries,
        }

    def write(self, path: Path) -> None:
        if not self.entries:
            raise ValueError("refusing to write an empty artifact manifest")
        atomic_json(path, self.payload())

    @staticmethod
    def verify(manifest_path: Path, restored_root: Path) -> list[str]:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors: list[str] = []
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            return ["manifest has no entries"]
        for entry in entries:
            relative = Path(str(entry["path"]))
            target = restored_root / relative
            if not target.is_file():
                errors.append(f"missing: {relative.as_posix()}")
                continue
            actual_size = target.stat().st_size
            if actual_size != int(entry["bytes"]):
                errors.append(
                    f"size mismatch: {relative.as_posix()} expected={entry['bytes']} actual={actual_size}"
                )
                continue
            actual_hash = sha256_file(target)
            if actual_hash != entry["sha256"]:
                errors.append(f"sha256 mismatch: {relative.as_posix()}")
        return errors
