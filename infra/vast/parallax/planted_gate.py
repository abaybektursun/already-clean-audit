"""Run production-level planted gates and bind the result to source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .parallax_adjudicate import graft_world, operator_world
from .parallax_entities import frozen_entities
from .parallax_operator import bilinear_symmetry_error, central_directional_derivative, linearity_error
from .parallax_tasks import ConfirmationFirewallError, FrozenTaskBank
from .parallax_towers import key_role, replacement_plan
from .preflight import MAX_PREFLIGHT_MINUTES


SOURCES = (
    "parallax_entities.py",
    "parallax_tasks.py",
    "parallax_operator.py",
    "parallax_towers.py",
    "parallax_adjudicate.py",
    "preflight.py",
)


def run(matrix: Path) -> dict[str, object]:
    checks: dict[str, object] = {}
    entities = frozen_entities(matrix)
    checks["entity_count"] = len(entities)
    checks["both_fusion_jumps"] = {
        "pg1_pi": {row.role for row in entities if row.lineage == "pg1_pi"} >= {"vision_parent_candidate", "language_parent", "fused", "later_0"},
        "pg2": {row.role for row in entities if row.lineage == "pg2"} >= {"vision_parent_candidate", "language_parent", "fused", "later_0"},
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "items.jsonl"
        manifest.write_text(
            json.dumps({"split": "confirmation_eval", "task_id": "t", "item_id": "sealed"}) + "\n",
            encoding="utf-8",
        )
        seal = root / "confirmation.sha256"
        seal.write_text("sealed\n", encoding="utf-8")
        try:
            list(FrozenTaskBank(manifest, seal).iter_confirmation(arm_freeze=None))
        except ConfirmationFirewallError:
            checks["confirmation_refusal"] = True
        else:
            checks["confirmation_refusal"] = False

    apply = lambda vector: [2.0 * vector[0] + vector[1], vector[0] + 3.0 * vector[1]]
    checks["symmetry_error"] = bilinear_symmetry_error(apply, [1.0, 2.0], [-2.0, 0.5])
    checks["linearity_error"] = linearity_error(apply, [1.0, 2.0], [-2.0, 0.5], 0.7)
    checks["central_difference"] = central_directional_derivative(
        lambda x: x[0] ** 2 + 3.0 * x[1] ** 2, [2.0, -1.0], [0.5, 2.0], 1e-5
    )
    shapes = {"vision_tower.w": (2, 2), "multi_modal_projector.w": (2, 2)}
    checks["tower_plan"] = replacement_plan(shapes, shapes, role="vision")
    checks["projector_fixed"] = key_role("multi_modal_projector.w") == "fixed_interface"
    checks["graft_branches"] = sorted({
        graft_world(native_effect=2, graft_effect=2, repaired_effect=0, mde=1),
        graft_world(native_effect=2, graft_effect=0, repaired_effect=2, mde=1),
        graft_world(native_effect=2, graft_effect=0, repaired_effect=0, mde=1),
        graft_world(native_effect=0, graft_effect=0, repaired_effect=0, mde=1),
    })
    checks["operator_branches"] = sorted({
        operator_world(scalar_green=True, replay_green=True, behavioral_range=True),
        operator_world(scalar_green=False, replay_green=True, behavioral_range=True),
        operator_world(scalar_green=True, replay_green=False, behavioral_range=True),
        operator_world(scalar_green=True, replay_green=True, behavioral_range=False),
    })
    checks["preflight_max_minutes"] = MAX_PREFLIGHT_MINUTES

    green = (
        all(checks["both_fusion_jumps"].values())
        and checks["confirmation_refusal"] is True
        and float(checks["symmetry_error"]) < 1e-12
        and float(checks["linearity_error"]) < 1e-12
        and abs(float(checks["central_difference"]) + 10.0) < 1e-7
        and checks["tower_plan"] == {"vision_tower.w": "replace"}
        and checks["projector_fixed"] is True
        and len(checks["graft_branches"]) == 4
        and len(checks["operator_branches"]) == 4
        and checks["preflight_max_minutes"] == 30
    )
    source_root = Path(__file__).parent
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN" if green else "RED",
        "checks": checks,
        "source_sha256": {name: sha256_file(source_root / name) for name in SOURCES},
        "matrix_sha256": sha256_file(matrix),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.matrix)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
