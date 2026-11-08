"""Aggregate-audit a rescue panel and prove no overlap with consumed v1 confirmation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .audit_localization_panel import audit as audit_panel


def _rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "manifests/items.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def audit(rescue_root: Path, consumed_v1_root: Path) -> dict[str, Any]:
    panel = audit_panel(rescue_root)
    rescue = _rows(rescue_root)
    consumed = [row for row in _rows(consumed_v1_root) if row["split"] == "confirmation_eval"]
    rescue_ids = {str(row["item_id"]) for row in rescue}
    rescue_images = {str(row["image_sha256"]) for row in rescue}
    rescue_sources = {
        json.dumps(row["source_identity"], sort_keys=True, separators=(",", ":"))
        for row in rescue
    }
    overlaps = {
        "item_ids": len(rescue_ids & {str(row["item_id"]) for row in consumed}),
        "image_hashes": len(rescue_images & {str(row["image_sha256"]) for row in consumed}),
        "source_identities": len(
            rescue_sources
            & {
                json.dumps(row["source_identity"], sort_keys=True, separators=(",", ":"))
                for row in consumed
            }
        ),
    }
    errors = list(panel["errors"])
    errors.extend(f"consumed_v1_overlap:{name}={count}" for name, count in overlaps.items() if count)
    return {
        "schema_version": 1,
        "status": "GREEN" if not errors else "RED",
        "errors": errors,
        "split_counts": panel["split_counts"],
        "target_balance": panel["target_balance"],
        "unique_item_ids": panel["unique_item_ids"],
        "unique_image_hashes": panel["unique_image_hashes"],
        "cross_split_image_hashes": panel["cross_split_image_hashes"],
        "cross_split_source_identities": panel["cross_split_source_identities"],
        "consumed_v1_confirmation_rows_parsed": len(consumed),
        "rescue_metadata_rows_parsed": len(rescue),
        "consumed_v1_overlap": overlaps,
        "items_jsonl_sha256": panel["items_jsonl_sha256"],
        "content_manifest_sha256": panel["content_manifest_sha256"],
        "consumed_v1_items_jsonl_sha256": sha256_file(
            consumed_v1_root / "manifests/items.jsonl"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", type=Path, required=True)
    parser.add_argument("--consumed-v1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.rescue_root, args.consumed_v1_root)
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
