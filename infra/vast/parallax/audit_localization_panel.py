"""Audit the PG2 localization panel without printing item-level confirmation metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file


def audit(root: Path) -> dict[str, Any]:
    item_path = root / "manifests/items.jsonl"
    content_path = root / "manifests/content.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    counts: Counter[str] = Counter()
    targets: dict[str, Counter[str]] = defaultdict(Counter)
    image_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    item_ids: set[str] = set()
    for line in item_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        split = str(row["split"])
        counts[split] += 1
        targets[split][str(row["target"])] += 1
        if row["item_id"] in item_ids:
            errors.append("duplicate item ID")
        item_ids.add(row["item_id"])
        image_path = root / row["image"]
        if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
            errors.append(f"missing or corrupt image: {row['item_id']}")
        image_splits[row["image_sha256"]].add(split)
        source_key = json.dumps(row["source_identity"], sort_keys=True, separators=(",", ":"))
        source_splits[source_key].add(split)
        scene = {obj["label"]: obj for obj in row["scene"]}
        answer = scene[row["target"]]
        reference = scene[row["reference"]]
        relation = row["relation"]
        valid = {
            "directly left of": answer["x"] < reference["x"] and answer["y"] == reference["y"],
            "directly right of": answer["x"] > reference["x"] and answer["y"] == reference["y"],
            "directly above": answer["y"] < reference["y"] and answer["x"] == reference["x"],
            "directly below": answer["y"] > reference["y"] and answer["x"] == reference["x"],
        }[relation]
        if not valid:
            errors.append(f"false spatial target: {row['item_id']}")
    expected_counts = {"discovery_eval": 16, "confirmation_eval": 32}
    if dict(counts) != expected_counts:
        errors.append(f"split counts differ: {dict(counts)}")
    expected_targets = {
        "discovery_eval": Counter({candidate: 4 for candidate in "ABCD"}),
        "confirmation_eval": Counter({candidate: 8 for candidate in "ABCD"}),
    }
    for split in expected_counts:
        if targets[split] != expected_targets[split]:
            errors.append(f"target imbalance: {split}={dict(targets[split])}")
    cross_images = sum(len(splits) > 1 for splits in image_splits.values())
    cross_sources = sum(len(splits) > 1 for splits in source_splits.values())
    if cross_images:
        errors.append(f"{cross_images} image hashes cross split")
    if cross_sources:
        errors.append(f"{cross_sources} source identities cross split")
    if sha256_file(item_path) != content.get("items_jsonl_sha256"):
        errors.append("items manifest hash mismatch")
    return {
        "schema_version": 1,
        "status": "GREEN" if not errors else "RED",
        "errors": errors,
        "total_items": sum(counts.values()),
        "split_counts": dict(sorted(counts.items())),
        "target_balance": {
            split: dict(sorted(values.items())) for split, values in sorted(targets.items())
        },
        "unique_item_ids": len(item_ids),
        "unique_image_hashes": len(image_splits),
        "cross_split_image_hashes": cross_images,
        "cross_split_source_identities": cross_sources,
        "items_jsonl_sha256": sha256_file(item_path),
        "content_manifest_sha256": sha256_file(content_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
