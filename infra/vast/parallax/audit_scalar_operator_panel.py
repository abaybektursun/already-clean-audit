"""Aggregate-audit the scalar-operator panel against all prior localization panels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file


def _rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "manifests/items.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def audit(root: Path, prior_roots: list[Path]) -> dict[str, Any]:
    rows = _rows(root)
    errors: list[str] = []
    if len(rows) != 8 or Counter(row["split"] for row in rows) != Counter({"confirmation_eval": 8}):
        errors.append("operator panel split count differs")
    targets = Counter(str(row["target"]) for row in rows)
    if targets != Counter({candidate: 2 for candidate in "ABCD"}):
        errors.append(f"operator panel target imbalance: {dict(targets)}")
    item_ids = {str(row["item_id"]) for row in rows}
    image_hashes = {str(row["image_sha256"]) for row in rows}
    sources = {
        json.dumps(row["source_identity"], sort_keys=True, separators=(",", ":")) for row in rows
    }
    if len(item_ids) != 8 or len(image_hashes) != 8 or len(sources) != 8:
        errors.append("operator panel contains duplicate item, image, or source identity")
    for row in rows:
        image_path = root / row["image"]
        if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
            errors.append("operator panel image missing or corrupt")
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
            errors.append("operator panel contains a false spatial target")

    prior_counts = []
    overlap = {"item_ids": 0, "image_hashes": 0, "source_identities": 0}
    for prior_root in prior_roots:
        prior = _rows(prior_root)
        prior_counts.append(len(prior))
        overlap["item_ids"] += len(item_ids & {str(row["item_id"]) for row in prior})
        overlap["image_hashes"] += len(
            image_hashes & {str(row["image_sha256"]) for row in prior}
        )
        overlap["source_identities"] += len(
            sources
            & {
                json.dumps(row["source_identity"], sort_keys=True, separators=(",", ":"))
                for row in prior
            }
        )
    errors.extend(f"prior_panel_overlap:{name}={value}" for name, value in overlap.items() if value)
    item_path = root / "manifests/items.jsonl"
    content_path = root / "manifests/content.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    if sha256_file(item_path) != content.get("items_jsonl_sha256"):
        errors.append("operator items manifest hash mismatch")
    return {
        "schema_version": 1,
        "status": "GREEN" if not errors else "RED",
        "errors": errors,
        "split_counts": {"confirmation_eval": len(rows)},
        "target_balance": dict(sorted(targets.items())),
        "unique_item_ids": len(item_ids),
        "unique_image_hashes": len(image_hashes),
        "unique_source_identities": len(sources),
        "prior_panel_metadata_rows_parsed": prior_counts,
        "prior_panel_overlap": overlap,
        "items_jsonl_sha256": sha256_file(item_path),
        "content_manifest_sha256": sha256_file(content_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.prior_root)
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
