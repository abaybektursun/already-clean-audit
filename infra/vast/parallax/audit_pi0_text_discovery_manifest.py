"""Independently recompute a frozen P2 text panel without opening image bodies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class P2TextDiscoveryAuditError(RuntimeError):
    """Raised when independent recomputation contradicts the frozen manifest."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _project(row: Mapping[str, Any], expected_candidates: list[str]) -> dict[str, Any]:
    return {
        "item_id": str(row["item_id"]),
        "task_id": str(row["task_id"]),
        "split": str(row["split"]),
        "prompt": str(row["prompt"]),
        "target": str(row["target"]),
        "candidates": expected_candidates,
        "source_identity": row["source_identity"],
        "image": str(row["image"]),
        "manifest_image_sha256": str(row["image_sha256"]),
        "manifest_image_bytes": int(row["image_bytes"]),
    }


def audit(
    *, source_path: Path, contract_path: Path, frozen_path: Path, execution_freeze_path: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_freeze_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    source_contract = contract["source"]
    selection = contract["selection"]
    expected_source_sha = str(source_contract["manifest_sha256"])
    observed_source_sha = _sha(source_path)
    if observed_source_sha != expected_source_sha:
        errors.append("source_manifest_sha256")
    if execution.get("status") != "FROZEN_ONE_SHOT_LOCAL_P2_DISCOVERY_MANIFEST_AUDIT":
        errors.append("execution_freeze_status")
    authorization = execution.get("authorization", {})
    if authorization.get("local_discovery_manifest_audit") is not True:
        errors.append("execution_freeze_audit_authorization")
    for field in (
        "paid_action",
        "AWS_CPU_create",
        "create_vast_instance",
        "create_vast_volume",
        "H100",
        "image_body_read",
        "model_load",
        "simulator_start",
        "training",
        "confirmation_items_read",
        "promotion",
        "automatic_retry",
    ):
        if authorization.get(field) is not False:
            errors.append(f"execution_freeze_{field}")
    if execution.get("consumed") is not False or execution.get("reusable") is not False:
        errors.append("execution_freeze_one_shot")

    allowed_splits = {str(value) for value in source_contract["allowed_splits_in_physical_manifest"]}
    expected_labels = {str(value) for value in selection["expected_label_strata"]}
    expected_candidates = [str(value) for value in selection["expected_candidate_set"]]
    task_id = str(selection["task_id"])
    split = str(source_contract["selected_split"])
    minima: dict[str, dict[str, Any]] = {}
    physical_splits: set[str] = set()
    source_rows = 0
    eligible_rows = 0
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            row_split = str(row.get("split"))
            physical_splits.add(row_split)
            if row_split not in allowed_splits:
                errors.append(f"forbidden_split_line_{line_number}")
                continue
            if str(row.get("task_id")) != task_id or row_split != split:
                continue
            eligible_rows += 1
            label = str(row.get("target"))
            if label not in expected_labels:
                errors.append(f"unexpected_label_{label}")
                continue
            if [str(value) for value in row.get("candidates") or []] != expected_candidates:
                errors.append(f"candidate_drift_{row.get('item_id')}")
                continue
            if not row.get("image") or not row.get("image_sha256"):
                errors.append(f"image_identity_{row.get('item_id')}")
                continue
            item_id = str(row["item_id"])
            if label not in minima or item_id < str(minima[label]["item_id"]):
                minima[label] = dict(row)
    if set(minima) != expected_labels:
        errors.append("label_coverage")

    recomputed_core = [_project(minima[label], expected_candidates) for label in sorted(minima)]
    recomputed_core.sort(key=lambda row: row["item_id"])
    recomputed_items: list[dict[str, Any]] = []
    for index, row in enumerate(recomputed_core):
        donor = recomputed_core[(index + 1) % len(recomputed_core)]
        output = dict(row)
        output["shuffled_image_donor"] = {
            "item_id": donor["item_id"],
            "target": donor["target"],
            "image": donor["image"],
            "manifest_image_sha256": donor["manifest_image_sha256"],
            "manifest_image_bytes": donor["manifest_image_bytes"],
        }
        recomputed_items.append(output)

    observed_items = frozen.get("items", [])
    if observed_items != recomputed_items:
        errors.append("items_exact")
    item_ids = [row["item_id"] for row in recomputed_items]
    donor_map = {
        row["item_id"]: row["shuffled_image_donor"]["item_id"] for row in recomputed_items
    }
    hashes = {
        "item_ids_sha256": _canonical(item_ids),
        "donor_map_sha256": _canonical(donor_map),
        "items_payload_sha256": _canonical(recomputed_items),
    }
    for name, value in hashes.items():
        if frozen.get("selection", {}).get(name) != value:
            errors.append(name)
    if len(set(donor_map.values())) != len(recomputed_items):
        errors.append("donor_bijection")
    for row in recomputed_items:
        donor = row["shuffled_image_donor"]
        if donor["item_id"] == row["item_id"]:
            errors.append(f"same_item_donor_{row['item_id']}")
        if donor["target"] == row["target"]:
            errors.append(f"same_target_donor_{row['item_id']}")
        if donor["manifest_image_sha256"] == row["manifest_image_sha256"]:
            errors.append(f"same_image_donor_{row['item_id']}")
    if frozen.get("firewall", {}).get("confirmation_items_read") != 0:
        errors.append("frozen_confirmation_counter")
    if frozen.get("firewall", {}).get("image_bodies_opened_on_laptop") != 0:
        errors.append("frozen_image_body_counter")

    return {
        "schema_version": 1,
        "status": "GREEN_INDEPENDENT_EXACT" if not errors else "RED",
        "errors": sorted(set(errors)),
        "bindings": {
            "source_manifest_sha256": observed_source_sha,
            "selection_contract_sha256": _sha(contract_path),
            "frozen_manifest_sha256": _sha(frozen_path),
            "execution_freeze_sha256": _sha(execution_freeze_path),
        },
        "recomputation": {
            "source_rows_parsed": source_rows,
            "eligible_discovery_rows": eligible_rows,
            "selected_items": len(recomputed_items),
            "labels": sorted(minima),
            **hashes,
            "donor_bijection": len(set(donor_map.values())) == len(recomputed_items),
            "exact_items_match": observed_items == recomputed_items,
        },
        "firewall": {
            "physical_source_splits": sorted(physical_splits),
            "confirmation_values_emitted": 0,
            "confirmation_items_read": 0,
            "image_bodies_opened": 0,
            "endpoint_outputs_read": 0,
            "model_loads": 0,
            "GPU_work": False,
            "paid_compute": False,
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.with_name(path.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite independent audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        source_path=args.source,
        contract_path=args.contract,
        frozen_path=args.frozen,
        execution_freeze_path=args.execution_freeze,
    )
    _atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "errors": payload["errors"]}))
    return 0 if payload["status"] == "GREEN_INDEPENDENT_EXACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
