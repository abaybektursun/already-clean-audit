"""Freeze the PG2 scalar-operator confirmation without item-level panel access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_common import BehavioralGateError
from .g1_common import utc_now


SOURCE_FILES = (
    "infra/vast/parallax/artifacts.py",
    "infra/vast/parallax/g1_common.py",
    "infra/vast/parallax/parallax_entities.py",
    "infra/vast/parallax/parallax_tasks.py",
    "infra/vast/parallax/materialize_tasks.py",
    "infra/vast/parallax/materialize_scalar_operator_panel.py",
    "infra/vast/parallax/audit_scalar_operator_panel.py",
    "infra/vast/parallax/scalar_operator.py",
    "infra/vast/parallax/scalar_operator_model.py",
    "infra/vast/parallax/scalar_operator_freeze.py",
    "infra/vast/parallax/scalar_operator_confirm.py",
    "infra/vast/parallax/behavioral_model.py",
    "infra/vast/parallax/behavioral_common.py",
)


def freeze(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    task_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scalar operator freeze: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_operator_panel_materialization":
        raise BehavioralGateError("scalar operator design is not frozen")
    rescue_row = config["prerequisites"]["localization_rescue"]
    rescue_path = root / rescue_row["path"]
    if sha256_file(rescue_path) != rescue_row["sha256"]:
        raise BehavioralGateError("scalar operator rescue prerequisite hash differs")
    rescue = json.loads(rescue_path.read_text(encoding="utf-8"))
    if rescue["verdict"]["status"] != rescue_row["required_verdict"]:
        raise BehavioralGateError("scalar operator rescue prerequisite verdict differs")
    direction_row = config["prerequisites"]["direction"]
    direction_path = root / direction_row["path"]
    if sha256_file(direction_path) != direction_row["file_sha256"]:
        raise BehavioralGateError("scalar operator direction prerequisite hash differs")
    audit_path = task_root / "manifests/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "GREEN" or audit.get("prior_panel_overlap") != {
        "image_hashes": 0,
        "item_ids": 0,
        "source_identities": 0,
    }:
        raise BehavioralGateError("scalar operator panel audit is not green and fresh")
    expected_items = int(config["panel"]["confirmation_items"])
    if audit.get("split_counts") != {"confirmation_eval": expected_items}:
        raise BehavioralGateError("scalar operator panel count differs")
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "FROZEN_FOR_SCALAR_OPERATOR_CONFIRMATION",
        "stage": "pg2_full_prefix_scalar_operator_confirmation",
        "confirmation_metadata_rows_parsed_by_frozen_aggregate_audit": expected_items,
        "confirmation_metadata_rows_parsed_by_scientific_runners_before_freeze": 0,
        "confirmation_unique_items_evaluated_before_freeze": 0,
        "bindings": {
            "config_sha256": sha256_file(config_path),
            "matrix_sha256": sha256_file(matrix),
            "panel_content_sha256": sha256_file(task_root / "manifests/content.json"),
            "panel_audit_sha256": sha256_file(audit_path),
            "direction_file_sha256": sha256_file(direction_path),
            "rescue_confirmation_sha256": sha256_file(rescue_path),
            "source_sha256": {path: sha256_file(root / path) for path in SOURCE_FILES},
        },
        "confirmation_seal_sha256": sha256_file(task_root / "manifests/confirmation.sha256"),
        "directions": config["directions"],
        "epsilon": config["epsilon"],
        "gates": config["gates"],
        "verdicts": config["verdicts"],
        "claim_boundary": "pg2_layer0_real_image_suffix_kl_coherent_full_prefix_scalar_only",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        root=args.root,
        matrix=args.matrix,
        config_path=args.config,
        task_root=args.task_root,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
