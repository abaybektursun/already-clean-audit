"""Freeze the fixed-0.001 PG2 localization rescue without item-level panel access."""

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
    "infra/vast/parallax/materialize_localization_panel.py",
    "infra/vast/parallax/audit_localization_panel.py",
    "infra/vast/parallax/audit_localization_rescue_panel.py",
    "infra/vast/parallax/localization_rescue_confirm.py",
    "infra/vast/parallax/localization_rescue_freeze.py",
    "infra/vast/parallax/behavioral_localization.py",
    "infra/vast/parallax/behavioral_localization_model.py",
    "infra/vast/parallax/behavioral_model.py",
    "infra/vast/parallax/behavioral_common.py",
)


def _prerequisite(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = root / str(row["path"])
    observed = sha256_file(path)
    if observed != row["sha256"]:
        raise BehavioralGateError(f"rescue prerequisite hash mismatch: {path}")
    return {"path": str(row["path"]), "sha256": observed}


def freeze(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    task_root: Path,
    direction_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite localization rescue freeze: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_new_panel_materialization":
        raise BehavioralGateError("localization rescue design is not frozen")
    amplitude = float(config["energy"]["fixed_reference_amplitude"])
    mde = float(config["mde"]["fixed_nats"])
    if amplitude != 0.001 or mde != 0.021652735769748688:
        raise BehavioralGateError("localization rescue amplitude or MDE differs from freeze")
    if sha256_file(direction_path) != config["direction"]["file_sha256"]:
        raise BehavioralGateError("localization rescue direction file mismatch")

    prerequisites = {
        name: _prerequisite(root, row)
        for name, row in (
            ("behavioral_confirmation", config["prerequisites"]["behavioral_confirmation"]),
            ("localization_calibration", config["prerequisites"]["localization_calibration"]),
            ("localization_void", config["prerequisites"]["localization_void"]),
        )
    }
    behavioral = json.loads(
        (root / prerequisites["behavioral_confirmation"]["path"]).read_text(encoding="utf-8")
    )
    if behavioral["lineages"]["pg2"]["verdict"]["status"] != "GREEN_DIRECTIONAL_RANGE":
        raise BehavioralGateError("prior PG2 behavioral confirmation is not green")
    calibration = json.loads(
        (root / prerequisites["localization_calibration"]["path"]).read_text(encoding="utf-8")
    )
    rung = calibration["amplitude"]["diagnostics"].get("0.001", {})
    if not rung.get("usable"):
        raise BehavioralGateError("0.001 was not safe and usable in localization discovery")
    prior_void = json.loads(
        (root / prerequisites["localization_void"]["path"]).read_text(encoding="utf-8")
    )
    if prior_void["verdict"]["status"] != "VOID_NO_FULL_PREFIX_REPLICATION":
        raise BehavioralGateError("prior localization result is not the registered VOID")
    if float(prior_void["selected_amplitude"]) != 0.0003:
        raise BehavioralGateError("prior localization VOID amplitude differs")

    audit_path = task_root / "manifests/audit.json"
    content_path = task_root / "manifests/content.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_confirmation = int(config["new_panel"]["confirmation_items"])
    if audit.get("status") != "GREEN":
        raise BehavioralGateError("localization rescue panel audit is not green")
    if int(audit["split_counts"]["confirmation_eval"]) != expected_confirmation:
        raise BehavioralGateError("localization rescue confirmation count differs")

    source_hashes = {path: sha256_file(root / path) for path in SOURCE_FILES}
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "FROZEN_FOR_LOCALIZATION_RESCUE_CONFIRMATION",
        "stage": "pg2_prefix_position_localization_rescue_confirmation",
        "confirmation_metadata_rows_parsed_by_frozen_aggregate_audit": expected_confirmation,
        "confirmation_metadata_rows_parsed_by_scientific_runners_before_freeze": 0,
        "confirmation_unique_items_evaluated_before_freeze": 0,
        "bindings": {
            "config_sha256": sha256_file(config_path),
            "matrix_sha256": sha256_file(matrix),
            "panel_content_sha256": sha256_file(content_path),
            "panel_audit_sha256": sha256_file(audit_path),
            "direction_file_sha256": sha256_file(direction_path),
            "prerequisites": prerequisites,
            "source_sha256": source_hashes,
        },
        "confirmation_seal_sha256": sha256_file(task_root / "manifests/confirmation.sha256"),
        "frozen_arms": list(config["cells"]["arms"]),
        "selected_amplitude": amplitude,
        "frozen_mde_nats": mde,
        "branch_worlds": list(config["adjudication"]["worlds"]),
        "claim_boundary": "token_position_localization_only_not_image_content_semantics",
        "second_rescue": "FORBIDDEN",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        root=args.root,
        matrix=args.matrix,
        config_path=args.config,
        task_root=args.task_root,
        direction_path=args.direction,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
