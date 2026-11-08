"""Read the frozen task bank without leaking confirmation items."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


DISCOVERY_SPLITS = frozenset({"train", "discovery_eval"})


class ConfirmationFirewallError(PermissionError):
    pass


class FrozenTaskBank:
    def __init__(self, manifest: Path, confirmation_seal: Path):
        self.manifest = manifest
        self.confirmation_seal = confirmation_seal

    def iter_discovery(self, *, task_id: str | None = None) -> Iterator[dict[str, Any]]:
        with self.manifest.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item["split"] not in DISCOVERY_SPLITS:
                    continue
                if task_id is None or item["task_id"] == task_id:
                    yield item


class DiscoveryTaskBank:
    """Read a physically separated discovery manifest and reject mixed splits."""

    def __init__(self, manifest: Path):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.manifest = manifest

    def iter_rows(self, *, task_id: str | None = None) -> Iterator[dict[str, Any]]:
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                item = json.loads(line)
                if item.get("split") not in DISCOVERY_SPLITS:
                    raise ConfirmationFirewallError(
                        f"discovery-only manifest contains sealed split at line {line_number}"
                    )
                if task_id is None or item.get("task_id") == task_id:
                    yield item

    def iter_confirmation(
        self, *, arm_freeze: Path | None, task_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        if arm_freeze is None or not arm_freeze.is_file():
            raise ConfirmationFirewallError("confirmation read requires an existing arm-freeze file")
        if not self.confirmation_seal.is_file():
            raise ConfirmationFirewallError("confirmation seal is missing")
        seal = self.confirmation_seal.read_text(encoding="utf-8").strip()
        frozen = json.loads(arm_freeze.read_text(encoding="utf-8"))
        if frozen.get("confirmation_seal_sha256") != hashlib.sha256(
            self.confirmation_seal.read_bytes()
        ).hexdigest():
            raise ConfirmationFirewallError("arm freeze does not bind the current confirmation seal")
        if not seal:
            raise ConfirmationFirewallError("confirmation seal is empty")
        with self.manifest.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item["split"] != "confirmation_eval":
                    continue
                if task_id is None or item["task_id"] == task_id:
                    yield item
