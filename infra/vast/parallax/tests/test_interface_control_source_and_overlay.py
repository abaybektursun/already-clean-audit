from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infra.vast.parallax import interface_control_overlay_stage as stage
from infra.vast.parallax import interface_control_source_gate as source_gate
from infra.vast.parallax.training_common import TrainingGateError


ROOT = Path(__file__).resolve().parents[4]


def test_source_gate_binds_pg2_first_s3_only_no_confirmation() -> None:
    result = source_gate.validate_local(ROOT)
    assert result["status"] == "GREEN_EXACT_PG2_INTERFACE_CONTROL_SOURCE"
    assert result["run_id"] == "parallax-pg2-interface-control-s3-v3"
    assert result["verified_artifacts"] == 21
    assert result["overlay_objects"] == 9
    assert result["confirmation_items_read"] == 0
    assert result["vast_persistent_storage"] == "FORBIDDEN"
    assert result["source_closure"]["status"] == "GREEN_EXACT_BUNDLE_OVERLAY_SOURCE_CLOSURE"
    assert result["source_closure"]["base_bundle_artifacts"] == 13
    assert result["source_closure"]["overlay_artifacts"] == 8


def test_overlay_inventory_is_seven_tiny_text_objects() -> None:
    source_gate.validate_local(ROOT)
    rows = [stage._object(ROOT, Path(path)) for path in sorted(stage.OVERLAY_IDENTITIES)]
    assert len(rows) == len({row["key"] for row in rows}) == 9
    assert sum(int(row["bytes"]) for row in rows) < 256 * 1024
    assert {str(row["path"]): str(row["sha256"]) for row in rows} == stage.OVERLAY_IDENTITIES


def test_source_closure_plants_missing_and_mismatched_artifacts() -> None:
    contract = json.loads((ROOT / source_gate.SOURCE_CONTRACT_PATH).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / source_gate.BUNDLE_RECEIPT).read_text(encoding="utf-8"))
    exact = copy.deepcopy(source_gate.OVERLAY_IDENTITIES)
    result = source_gate.validate_closure(
        contract=contract,
        bundle_entries=receipt["entries"],
        overlays=exact,
    )
    assert result["overlay_objects_including_contract"] == 9

    missing = copy.deepcopy(exact)
    missing.pop(contract["artifacts"]["model_cache_manifest"]["path"])
    with pytest.raises(TrainingGateError, match="missing an artifact"):
        source_gate.validate_closure(
            contract=contract,
            bundle_entries=receipt["entries"],
            overlays=missing,
        )

    mismatched = copy.deepcopy(exact)
    mismatched[contract["artifacts"]["optimizer_adjudication"]["path"]] = "0" * 64
    with pytest.raises(TrainingGateError, match="overlay has a hash mismatch"):
        source_gate.validate_closure(
            contract=contract,
            bundle_entries=receipt["entries"],
            overlays=mismatched,
        )
