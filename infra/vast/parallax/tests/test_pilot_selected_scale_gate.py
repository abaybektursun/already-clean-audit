from __future__ import annotations

import copy
from pathlib import Path

import pytest

from infra.vast.parallax import pilot_selected_scale_gate as gate


ROOT = Path(__file__).resolve().parents[4]


def test_consumed_selected_scale_gate_refuses_shared_range_code_drift() -> None:
    with pytest.raises(gate.SelectedScaleGateError, match="source artifact"):
        gate.validate_local(ROOT)


def test_config_diff_gate_detects_an_extra_scientific_mutation() -> None:
    before = {"optimizer": {"learning_rate": 0.001, "steps": 16}}
    after = copy.deepcopy(before)
    after["optimizer"]["learning_rate"] = 0.00001
    after["optimizer"]["steps"] = 32
    assert gate._leaf_diffs(before, after) == {
        "/optimizer/learning_rate": [0.001, 0.00001],
        "/optimizer/steps": [16, 32],
    }


def test_source_contract_hash_drift_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(gate.SelectedScaleGateError, match="base pilot config"):
        gate.validate_local(tmp_path)
