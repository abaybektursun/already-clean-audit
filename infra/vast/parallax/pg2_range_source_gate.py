"""Fail-closed source/parity gate for the one PG2 seed-17 range screen."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import sha256_file
from .training_common import TrainingGateError


PG1_CONFIG = Path(
    "studies/lineage/cross-lineage/config/parallax_training_pilot_cell_v2.yaml"
)
PG2_CONFIG = Path(
    "studies/lineage/cross-lineage/config/parallax_training_range_screen_pg2_seed17_v1.yaml"
)
PG2_OPTIMIZER = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_optimizer_scale_s3_v3_adjudication.json"
)
DEPENDENCY_AUDIT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_next_range_dependency_audit_v1.json"
)
POST_OPTIMIZER_FREEZE = Path(
    "studies/lineage/cross-lineage/config/parallax_execution_freeze_v94.json"
)

PG1_CONFIG_SHA256 = "690a4401b086cd280aca6f7d78a2e4dda82635a899cfd85aef4dd596fb34530f"
PG2_CONFIG_SHA256 = "435a57522066854486302617a0b3d5eb2d2d68bf597d57f9627cb0d02d1775b4"
PG2_OPTIMIZER_SHA256 = "97e4e17140acd7d9b4b7ac4dd96ff5f51888ce561c70dd5fe987270d89d16a1d"
DEPENDENCY_AUDIT_SHA256 = "90cf36b5150fd244180548b4522dacc5a692fc9e6e2aa9c59e2a593316ce3842"
POST_OPTIMIZER_FREEZE_SHA256 = "b06a673be72a86680a5e03347b36f2aa52c5658ea3cc6e2ba8ae51688f77e2f7"


def _exact(root: Path, relative: Path, digest: str, *, label: str) -> Path:
    path = root / relative
    if not path.is_file() or sha256_file(path) != digest:
        raise TrainingGateError(f"PG2 range {label} is absent or hash-drifted")
    return path


def validate_config_pair(
    primary: dict[str, Any], supporting: dict[str, Any], optimizer: dict[str, Any]
) -> dict[str, Any]:
    if primary.get("lineage") != {
        "id": "pg1_pi",
        "role": "primary",
        "checkpoint_role": "fused",
        "promotion_eligible": False,
    }:
        raise TrainingGateError("PG1 range reference is not exact")
    if supporting.get("lineage") != {
        "id": "pg2",
        "role": "supporting",
        "checkpoint_role": "fused",
        "promotion_eligible": False,
    }:
        raise TrainingGateError("PG2 range lineage firewall changed")
    for field in (
        "panel",
        "adapter",
        "runtime",
        "mechanical_adjudication",
        "required_outputs",
    ):
        if supporting.get(field) != primary.get(field):
            raise TrainingGateError(f"PG2 range scientific parity changed: {field}")
    primary_optimizer = copy.deepcopy(primary.get("optimizer", {}))
    supporting_optimizer = copy.deepcopy(supporting.get("optimizer", {}))
    scale_origin = supporting_optimizer.pop("scale_origin", None)
    if supporting_optimizer != primary_optimizer:
        raise TrainingGateError("PG2 range optimizer schedule differs from PG1 seed 17")
    if scale_origin != "independently_selected_within_pg2_not_imported_from_pg1":
        raise TrainingGateError("PG2 range optimizer origin is not independent")
    if (
        optimizer.get("status") != "GREEN_PG2_SELECTED_SCALE_MECHANICAL_NONPROMOTION"
        or optimizer.get("optimizer_adjudication", {}).get("verdict")
        != "GREEN_SELECTED_SCALE"
        or optimizer.get("optimizer_adjudication", {}).get("selected_learning_rate")
        != supporting_optimizer.get("learning_rate")
        or optimizer.get("lineage_firewall", {}).get(
            "PG1_selected_learning_rate_imported_as_PG2_answer"
        )
        is not False
    ):
        raise TrainingGateError("PG2 range does not bind its independent optimizer answer")
    if supporting.get("cross_lineage_firewall") != {
        "primary_causal_chain": "SigLIP_plus_Gemma1_to_PaliGemma1_to_pi0",
        "PG2_role": "supporting_replication_and_boundary_only",
        "import_PG1_selected_learning_rate": "forbidden",
        "compare_raw_objectives_across_generations": "forbidden",
        "allowed_cross_lineage_output": "normalized_range_verdicts_only",
        "single_cross_model_causal_chain_claimed": False,
    }:
        raise TrainingGateError("PG2 range cross-lineage firewall changed")
    if (
        supporting.get("panel", {}).get("confirmation_reads") != 0
        or supporting.get("claim_boundary")
        != "PG2_SUPPORTING_PROVISIONAL_RANGE_NOT_PROMOTION_ELIGIBLE"
    ):
        raise TrainingGateError("PG2 range escapes its nonpromotion boundary")
    return {
        "status": "GREEN_EXACT_PG2_SEED17_RANGE_PARITY",
        "same_fields": [
            "panel",
            "adapter",
            "optimizer_without_scale_origin",
            "runtime",
            "mechanical_adjudication",
            "required_outputs",
        ],
        "separate_fields": [
            "lineage_and_role",
            "checkpoint_and_revision_via_matrix",
            "independent_optimizer_execution_and_evidence",
            "claim_boundary",
        ],
        "learning_rate": supporting_optimizer["learning_rate"],
        "seed": supporting_optimizer["seed"],
        "promotion_eligible": False,
        "confirmation_reads": 0,
    }


def validate_local(root: Path) -> dict[str, Any]:
    root = root.resolve()
    primary_path = _exact(root, PG1_CONFIG, PG1_CONFIG_SHA256, label="PG1 config")
    supporting_path = _exact(root, PG2_CONFIG, PG2_CONFIG_SHA256, label="PG2 config")
    optimizer_path = _exact(
        root, PG2_OPTIMIZER, PG2_OPTIMIZER_SHA256, label="optimizer adjudication"
    )
    dependency_path = _exact(
        root, DEPENDENCY_AUDIT, DEPENDENCY_AUDIT_SHA256, label="dependency audit"
    )
    freeze_path = _exact(
        root,
        POST_OPTIMIZER_FREEZE,
        POST_OPTIMIZER_FREEZE_SHA256,
        label="post-optimizer freeze",
    )
    result = validate_config_pair(
        yaml.safe_load(primary_path.read_text(encoding="utf-8")),
        yaml.safe_load(supporting_path.read_text(encoding="utf-8")),
        json.loads(optimizer_path.read_text(encoding="utf-8")),
    )
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    if (
        dependency.get("status") != "GREEN_READ_ONLY_DEPENDENCY_AUDIT"
        or "run_one_PG2_seed_17_nonpromotion_range_screen_only_if_a_scale_passes"
        not in dependency.get("selected_sequence", [])
    ):
        raise TrainingGateError("PG2 range dependency order changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_POST_PG2_OPTIMIZER_GREEN_ALL_PAID_ACTION_OFF"
        or any(freeze.get("authorization", {}).values())
        or freeze.get("state", {}).get("PG2_optimizer_scale")
        != "ANSWERED_LR_1E_5_INDEPENDENTLY"
        or freeze.get("state", {}).get("PG2_seed_17_range") != "UNANSWERED"
    ):
        raise TrainingGateError("PG2 range post-optimizer state is not exact/all-paid-off")
    return {
        **result,
        "lineage": "pg2",
        "role": "supporting",
        "task_id": "vr_a_beans_balanced",
        "train_per_target": 32,
        "discovery_per_target": 8,
        "steps": 16,
        "snapshots": [0, 1, 2, 4, 8, 16],
        "paid_authorization": False,
    }
