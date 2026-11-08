"""Confound and split-firewall audit for a materialized PARALLAX task bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file


def load_items(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(root: Path) -> dict[str, Any]:
    item_path = root / "manifests/items.jsonl"
    content_path = root / "manifests/content.json"
    items = load_items(item_path)
    content = json.loads(content_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    ids = [row["item_id"] for row in items]
    if len(ids) != len(set(ids)):
        errors.append("duplicate item IDs")
    if sha256_file(item_path) != content.get("items_jsonl_sha256"):
        errors.append("items manifest SHA does not match content manifest")

    image_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    semantic_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    target_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    task_split_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in items:
        split = row["split"]
        task = row["task_id"]
        task_split_counts[task][split] += 1
        target_counts[task][split][str(row["target"])] += 1
        if row.get("image"):
            image_path = root / row["image"]
            if not image_path.is_file():
                errors.append(f"missing image: {row['image']}")
            elif sha256_file(image_path) != row.get("image_sha256"):
                errors.append(f"image SHA mismatch: {row['image']}")
            image_splits[str(row["image_sha256"])].add(split)
        source_key = json.dumps(row["source_identity"], sort_keys=True, separators=(",", ":"))
        source_splits[source_key].add(split)
        if "nonce" in row:
            semantic_splits[(task, str(row["nonce"]))].add(split)
        if task.startswith("iface_"):
            semantic = str(row.get("semantic_answer")) + ":" + row["prompt"].split("?")[0]
            semantic_splits[(task, semantic)].add(split)

        if task == "rel_a_spatial_selection":
            scene = {obj["label"]: obj for obj in row["scene"]}
            answer = scene[row["target"]]
            reference = scene[row["reference"]]
            relation = row["relation"]
            truth = {
                "directly left of": answer["x"] < reference["x"] and answer["y"] == reference["y"],
                "directly right of": answer["x"] > reference["x"] and answer["y"] == reference["y"],
                "directly above": answer["y"] < reference["y"] and answer["x"] == reference["x"],
                "directly below": answer["y"] > reference["y"] and answer["x"] == reference["x"],
            }[relation]
            if not truth:
                errors.append(f"false registered spatial target: {row['item_id']}")
        if task == "rel_b_relation_comparison":
            scene = {obj["label"]: obj for obj in row["scene"]}
            first, second = scene[row["first"]], scene[row["second"]]
            relation = row["relation"]
            truth = {
                "left of": first["x"] < second["x"],
                "right of": first["x"] > second["x"],
                "above": first["y"] < second["y"],
                "below": first["y"] > second["y"],
            }[relation]
            expected = "yes" if truth else "no"
            if row["target"] != expected:
                errors.append(f"false registered relation target: {row['item_id']}")

    cross_split_images = sorted(key for key, splits in image_splits.items() if len(splits) > 1)
    if cross_split_images:
        errors.append(f"{len(cross_split_images)} image hashes cross split firewalls")
    cross_split_sources = sorted(key for key, splits in source_splits.items() if len(splits) > 1)
    if cross_split_sources:
        errors.append(f"{len(cross_split_sources)} source identities cross split firewalls")
    cross_split_semantics = sorted(key for key, splits in semantic_splits.items() if len(splits) > 1)
    if cross_split_semantics:
        errors.append(f"{len(cross_split_semantics)} nonce/interface semantics cross split firewalls")

    balance: dict[str, dict[str, dict[str, int]]] = {}
    for task, split_rows in sorted(target_counts.items()):
        balance[task] = {}
        for split, counts in sorted(split_rows.items()):
            balance[task][split] = dict(sorted(counts.items()))
            values = list(counts.values())
            if task in {"rel_b_relation_comparison", "iface_b_nonce_output_codes"} and max(values) - min(values) > 1:
                errors.append(f"binary target imbalance: {task}/{split}={dict(counts)}")
            if task.startswith("vr_") and max(values) != min(values):
                errors.append(f"recognition target imbalance: {task}/{split}={dict(counts)}")

    expected_tasks = {
        "vr_a_beans_balanced",
        "vr_b_fashion_category",
        "rel_a_spatial_selection",
        "rel_b_relation_comparison",
        "new_a_nonce_fact_binding",
        "new_b_image_nonce_binding",
        "iface_a_permuted_abcd",
        "iface_b_nonce_output_codes",
    }
    if set(task_split_counts) != expected_tasks:
        errors.append(f"task set mismatch: {sorted(task_split_counts)}")
    for task, counts in task_split_counts.items():
        if set(counts) != {"train", "discovery_eval", "confirmation_eval"}:
            errors.append(f"split set mismatch: {task}={dict(counts)}")

    return {
        "schema_version": 1,
        "status": "GREEN" if not errors else "RED",
        "errors": errors,
        "total_items": len(items),
        "task_split_counts": {task: dict(sorted(counts.items())) for task, counts in sorted(task_split_counts.items())},
        "target_balance": balance,
        "cross_split_image_hashes": len(cross_split_images),
        "cross_split_source_identities": len(cross_split_sources),
        "cross_split_nonce_or_interface_semantics": len(cross_split_semantics),
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
    print(json.dumps({"output": str(args.output), "status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
