"""Freeze the deterministic discovery-only P2 text/VQA panel without image-body access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class P2TextDiscoveryFreezeError(RuntimeError):
    """Raised when the discovery selection or its firewall differs from the freeze."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_hex_sha256(value: object, *, field: str) -> str:
    rendered = str(value)
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise P2TextDiscoveryFreezeError(f"{field} is not a lowercase SHA-256")
    return rendered


def _require_safe_relative_path(value: object, *, field: str) -> str:
    rendered = str(value)
    path = Path(rendered)
    if not rendered or path.is_absolute() or ".." in path.parts:
        raise P2TextDiscoveryFreezeError(f"{field} is not a safe relative path: {rendered!r}")
    return rendered


def load_rows(manifest: Path, *, expected_sha256: str, allowed_splits: set[str]) -> list[dict[str, Any]]:
    if sha256_file(manifest) != _require_hex_sha256(expected_sha256, field="manifest_sha256"):
        raise P2TextDiscoveryFreezeError("discovery manifest SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split"))
            if split not in allowed_splits:
                raise P2TextDiscoveryFreezeError(
                    f"physical discovery manifest contains forbidden split {split!r} at line {line_number}"
                )
            rows.append(row)
    if not rows:
        raise P2TextDiscoveryFreezeError("discovery manifest is empty")
    return rows


def select_panel(rows: Iterable[Mapping[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = contract["source"]
    selection = contract["selection"]
    task_id = str(selection["task_id"])
    selected_split = str(source["selected_split"])
    expected_labels = [str(value) for value in selection["expected_label_strata"]]
    expected_candidates = [str(value) for value in selection["expected_candidate_set"]]
    per_label = int(selection["items_per_label_stratum"])
    if per_label != 1:
        raise P2TextDiscoveryFreezeError("v1 P2 mechanical panel requires exactly one item per label")
    if len(expected_labels) != len(set(expected_labels)):
        raise P2TextDiscoveryFreezeError("expected label strata contain duplicates")
    if set(expected_labels) != set(expected_candidates):
        raise P2TextDiscoveryFreezeError("label strata and candidate set differ")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("task_id")) != task_id or str(row.get("split")) != selected_split:
            continue
        target = str(row.get("target"))
        if target not in set(expected_labels):
            raise P2TextDiscoveryFreezeError(f"unexpected target label in frozen task: {target!r}")
        if not row.get("image") or not row.get("image_sha256"):
            raise P2TextDiscoveryFreezeError(f"eligible row lacks image identity: {row.get('item_id')}")
        candidates = [str(value) for value in row.get("candidates") or []]
        if candidates != expected_candidates:
            raise P2TextDiscoveryFreezeError(f"candidate-set drift: {row.get('item_id')}")
        if target not in candidates:
            raise P2TextDiscoveryFreezeError(f"target absent from candidates: {row.get('item_id')}")
        grouped[target].append(row)

    if set(grouped) != set(expected_labels):
        raise P2TextDiscoveryFreezeError(
            f"label strata mismatch: observed={sorted(grouped)} expected={sorted(expected_labels)}"
        )

    selected: list[dict[str, Any]] = []
    for target in sorted(expected_labels):
        ordered = sorted(grouped[target], key=lambda row: str(row["item_id"]))
        row = ordered[0]
        selected.append(
            {
                "item_id": _require_hex_sha256(row["item_id"], field="item_id"),
                "task_id": task_id,
                "split": selected_split,
                "prompt": str(row["prompt"]),
                "target": target,
                "candidates": expected_candidates,
                "source_identity": row["source_identity"],
                "image": _require_safe_relative_path(row["image"], field="image"),
                "manifest_image_sha256": _require_hex_sha256(
                    row["image_sha256"], field="manifest_image_sha256"
                ),
                "manifest_image_bytes": int(row["image_bytes"]),
            }
        )
    selected.sort(key=lambda row: row["item_id"])
    if len(selected) != int(selection["expected_items"]):
        raise P2TextDiscoveryFreezeError("selected panel size differs from freeze")
    if len({row["item_id"] for row in selected}) != len(selected):
        raise P2TextDiscoveryFreezeError("selected item IDs are not unique")
    prompts = {row["prompt"] for row in selected}
    candidate_sets = {tuple(row["candidates"]) for row in selected}
    if len(prompts) != 1 or len(candidate_sets) != 1:
        raise P2TextDiscoveryFreezeError("frozen correct/shuffled control needs common prompt and candidates")
    return selected


def attach_donors(panel: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(panel, key=lambda row: str(row["item_id"]))
    if len(ordered) < 2:
        raise P2TextDiscoveryFreezeError("shuffled-image control needs at least two selected items")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        donor = ordered[(index + 1) % len(ordered)]
        if donor["item_id"] == row["item_id"]:
            raise P2TextDiscoveryFreezeError("donor reused the recipient item")
        if donor["target"] == row["target"]:
            raise P2TextDiscoveryFreezeError("donor reused the recipient target label")
        if donor["manifest_image_sha256"] == row["manifest_image_sha256"]:
            raise P2TextDiscoveryFreezeError("donor reused the recipient image bytes")
        if donor["task_id"] != row["task_id"]:
            raise P2TextDiscoveryFreezeError("donor crossed the frozen task stratum")
        if donor["prompt"] != row["prompt"] or donor["candidates"] != row["candidates"]:
            raise P2TextDiscoveryFreezeError("donor would change prompt or candidate tokens")
        output = dict(row)
        output["shuffled_image_donor"] = {
            "item_id": donor["item_id"],
            "target": donor["target"],
            "image": donor["image"],
            "manifest_image_sha256": donor["manifest_image_sha256"],
            "manifest_image_bytes": donor["manifest_image_bytes"],
        }
        result.append(output)
    return result


def freeze_manifest(
    *, rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], bindings: Mapping[str, str]
) -> dict[str, Any]:
    selected = select_panel(rows, contract)
    items = attach_donors(selected)
    item_ids = [row["item_id"] for row in items]
    donor_map = {row["item_id"]: row["shuffled_image_donor"]["item_id"] for row in items}
    return {
        "schema_version": 1,
        "status": "GREEN_P2_TEXT_SELECTION_FROZEN_EXACT_CLOUD_IMAGE_VERSIONS_PENDING",
        "stage": "pi0_endpoint_p2_text_vqa_discovery_manifest",
        "claim_boundary": "SELECTION_AND_SEMANTIC_IMAGE_IDENTITIES_ONLY_NO_MODEL_OR_ENDPOINT_RESULT",
        "bindings": dict(sorted(bindings.items())),
        "selection": {
            "task_id": contract["selection"]["task_id"],
            "split": contract["source"]["selected_split"],
            "label_strata": sorted(str(value) for value in contract["selection"]["expected_label_strata"]),
            "items_per_label_stratum": 1,
            "items": len(items),
            "item_ids_sha256": canonical_sha256(item_ids),
            "donor_map_sha256": canonical_sha256(donor_map),
            "items_payload_sha256": canonical_sha256(items),
        },
        "items": items,
        "firewall": {
            "physical_source_splits": sorted(set(str(row["split"]) for row in rows)),
            "selected_split": contract["source"]["selected_split"],
            "confirmation_values_emitted": 0,
            "confirmation_items_read": 0,
            "endpoint_outputs_read": 0,
            "image_bodies_opened_on_laptop": 0,
            "model_loads": 0,
            "GPU_work": False,
            "paid_compute": False,
        },
        "next_required": [
            "reproduce_the_ten_selected_images_in_the_exact_Linux_cloud_package",
            "stage_each_exact_image_body_to_an_immutable_versioned_S3_key",
            "bind_exact_S3_VersionIds_and_cloud_body_SHA256s_before_P2_H100_use",
        ],
    }


def validate_execution_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    authorization = payload.get("authorization", {})
    if payload.get("status") != "FROZEN_ONE_SHOT_LOCAL_P2_DISCOVERY_MANIFEST_READ":
        raise P2TextDiscoveryFreezeError("execution freeze does not authorize this one-shot read")
    if authorization.get("local_discovery_manifest_read") is not True:
        raise P2TextDiscoveryFreezeError("local discovery manifest read is not authorized")
    forbidden_true = [
        "paid_action",
        "AWS_CPU_create",
        "create_vast_instance",
        "create_vast_volume",
        "H100",
        "model_load",
        "simulator_start",
        "training",
        "confirmation_items_read",
        "promotion",
        "automatic_retry",
    ]
    if any(authorization.get(field) is not False for field in forbidden_true):
        raise P2TextDiscoveryFreezeError("execution freeze accidentally authorizes broader work")
    if payload.get("consumed") is not False or payload.get("reusable") is not False:
        raise P2TextDiscoveryFreezeError("execution freeze is not fresh and one-shot")
    return payload


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = (root / args.contract).resolve() if not args.contract.is_absolute() else args.contract
    freeze_path = (
        (root / args.execution_freeze).resolve()
        if not args.execution_freeze.is_absolute()
        else args.execution_freeze
    )
    output_path = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output_path.exists() or output_path.with_name(output_path.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite P2 discovery manifest: {output_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    execution_freeze = validate_execution_freeze(freeze_path)
    source = contract["source"]
    manifest_path = (root / source["manifest_path"]).resolve()
    rows = load_rows(
        manifest_path,
        expected_sha256=source["manifest_sha256"],
        allowed_splits={str(value) for value in source["allowed_splits_in_physical_manifest"]},
    )
    bindings = {
        "selection_contract_sha256": sha256_file(contract_path),
        "execution_freeze_sha256": sha256_file(freeze_path),
        "source_discovery_manifest_sha256": sha256_file(manifest_path),
        "source_discovery_seal_sha256": sha256_file(root / source["discovery_seal_path"]),
        "source_content_manifest_sha256": sha256_file(root / source["content_manifest_path"]),
        "task_contract_sha256": sha256_file(root / source["task_contract_path"]),
        "predecessor_execution_freeze_sha256": str(
            execution_freeze["predecessor_execution_freeze"]["sha256"]
        ),
    }
    if bindings["source_discovery_seal_sha256"] != source["discovery_seal_sha256"]:
        raise P2TextDiscoveryFreezeError("discovery seal SHA-256 mismatch")
    if bindings["source_content_manifest_sha256"] != source["content_manifest_sha256"]:
        raise P2TextDiscoveryFreezeError("content manifest SHA-256 mismatch")
    if bindings["task_contract_sha256"] != source["task_contract_sha256"]:
        raise P2TextDiscoveryFreezeError("task contract SHA-256 mismatch")
    payload = freeze_manifest(rows=rows, contract=contract, bindings=bindings)
    _atomic_json(output_path, payload)
    print(json.dumps({"output": str(output_path), "status": payload["status"], "items": len(payload["items"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
