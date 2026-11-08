"""Content-hash calibrated behavioral arms before sealed confirmation may be read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .behavioral_common import BehavioralGateError
from .g1_common import utc_now


def build(
    *, config: Path, calibration_dir: Path, task_root: Path, runner: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation freeze: {output}")
    packet_path = calibration_dir / "calibration.json"
    manifest_path = calibration_dir / "manifest.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("status") != "GREEN_CALIBRATED" or packet.get("confirmation_items_read") != 0:
        raise BehavioralGateError("calibration does not license confirmation freeze")
    if packet["bindings"]["config_sha256"] != sha256_file(config):
        raise BehavioralGateError("calibration/config binding mismatch")
    seal = task_root / "manifests/confirmation.sha256"
    if sha256_file(seal) != packet["bindings"]["confirmation_seal_file_sha256"]:
        raise BehavioralGateError("calibration/confirmation seal-file binding mismatch")
    lineages = {}
    for lineage in ("pg1_pi", "pg2"):
        row = packet["lineages"][lineage]
        direction = calibration_dir / row["direction"]["path"]
        if sha256_file(direction) != row["direction"]["sha256"]:
            raise BehavioralGateError(f"calibration direction mismatch: {lineage}")
        lineages[lineage] = {
            "role": row["role"],
            "checkpoint": row["checkpoint"],
            "selected_amplitude": row["amplitude"]["selected_amplitude"],
            "frozen_mde_nats": row["intrinsic_mde"]["frozen_mde_nats"],
            "direction_file": direction.name,
            "direction_sha256": sha256_file(direction),
            "direction_tensor_sha256": row["direction"]["tensor_sha256"],
        }
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "FROZEN_FOR_CONFIRMATION",
        "provider": "vast.ai",
        "confirmation_items_read_before_freeze": 0,
        # FrozenTaskBank checks this exact key before yielding any sealed row.
        "confirmation_seal_sha256": sha256_file(seal),
        "confirmation_content_digest": seal.read_text(encoding="utf-8").strip(),
        "bindings": {
            "config_sha256": sha256_file(config),
            "calibration_sha256": sha256_file(packet_path),
            "calibration_manifest_sha256": sha256_file(manifest_path),
            "confirmation_runner_sha256": sha256_file(runner),
        },
        "lineage_firewall": {
            "primary": "pg1_pi",
            "supporting": "pg2",
            "raw_cross_gemma_comparison": False,
            "independent_amplitudes_and_mdes": True,
        },
        "lineages": lineages,
    }
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        config=args.config,
        calibration_dir=args.calibration_dir,
        task_root=args.task_root,
        runner=args.runner,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
