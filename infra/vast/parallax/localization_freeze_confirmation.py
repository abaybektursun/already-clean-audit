"""Create the content-addressed PG2 localization confirmation freeze without item access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    "infra/vast/parallax/localization_calibrate.py",
    "infra/vast/parallax/localization_confirm.py",
    "infra/vast/parallax/localization_freeze_confirmation.py",
    "infra/vast/parallax/behavioral_localization.py",
    "infra/vast/parallax/behavioral_localization_model.py",
    "infra/vast/parallax/behavioral_model.py",
    "infra/vast/parallax/behavioral_common.py",
)


def freeze(
    *,
    root: Path,
    config_path: Path,
    calibration_dir: Path,
    task_root: Path,
    direction_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite localization arm freeze: {output}")
    calibration_path = calibration_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "GREEN_CALIBRATED":
        raise BehavioralGateError("localization calibration is not green")
    if calibration.get("confirmation_metadata_rows_parsed_by_calibration") != 0:
        raise BehavioralGateError("localization calibration parsed confirmation metadata")
    if calibration.get("confirmation_unique_items_evaluated_by_calibration") != 0:
        raise BehavioralGateError("localization calibration evaluated confirmation items")
    if calibration["bindings"]["config_sha256"] != sha256_file(config_path):
        raise BehavioralGateError("localization calibration config binding mismatch")
    selected = calibration["amplitude"]["selected_amplitude"]
    mde = calibration["intrinsic_mde"]["frozen_mde_nats"]
    if selected is None or mde is None:
        raise BehavioralGateError("localization calibration did not freeze amplitude and MDE")
    source_hashes = {path: sha256_file(root / path) for path in SOURCE_FILES}
    panel_audit = json.loads((task_root / "manifests/audit.json").read_text(encoding="utf-8"))
    aggregate_confirmation_rows = int(panel_audit["split_counts"]["confirmation_eval"])
    packet: dict[str, object] = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "FROZEN_FOR_LOCALIZATION_CONFIRMATION",
        "stage": "pg2_prefix_position_localization_confirmation",
        "confirmation_metadata_rows_parsed_by_frozen_aggregate_audit": aggregate_confirmation_rows,
        "confirmation_metadata_rows_parsed_by_scientific_runners_before_freeze": 0,
        "confirmation_unique_items_evaluated_before_freeze": 0,
        "bindings": {
            "config_sha256": sha256_file(config_path),
            "calibration_sha256": sha256_file(calibration_path),
            "calibration_manifest_sha256": sha256_file(calibration_dir / "manifest.json"),
            "panel_content_sha256": sha256_file(task_root / "manifests/content.json"),
            "panel_audit_sha256": sha256_file(task_root / "manifests/audit.json"),
            "direction_file_sha256": sha256_file(direction_path),
            "source_sha256": source_hashes,
        },
        "confirmation_seal_sha256": sha256_file(task_root / "manifests/confirmation.sha256"),
        "frozen_arms": [
            "full_sensitive",
            "full_rotated",
            "full_white",
            "image_sensitive",
            "image_rotated",
            "prompt_sensitive",
            "prompt_rotated",
        ],
        "selected_amplitude": float(selected),
        "frozen_mde_nats": float(mde),
        "branch_worlds": [
            "VOID_NO_FULL_PREFIX_REPLICATION",
            "IMAGE_TOKEN_POSITION_DOMINANT",
            "PROMPT_POSITION_DOMINANT",
            "MIXED_POSITION_RANGE",
            "DISTRIBUTED_OR_UNRESOLVED_POSITION_RANGE",
        ],
        "claim_boundary": "token_position_localization_only_not_image_content_semantics",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        root=args.root,
        config_path=args.config,
        calibration_dir=args.calibration_dir,
        task_root=args.task_root,
        direction_path=args.direction,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
