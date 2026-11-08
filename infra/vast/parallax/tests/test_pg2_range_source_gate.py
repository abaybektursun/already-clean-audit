from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from infra.vast.parallax import pg2_range_source_gate as gate
from infra.vast.parallax.training_common import TrainingGateError
from infra.vast.parallax.training_pilot_preflight import _repository_from_manifest


ROOT = Path(__file__).resolve().parents[4]


def _rows():
    primary = yaml.safe_load((ROOT / gate.PG1_CONFIG).read_text(encoding="utf-8"))
    supporting = yaml.safe_load((ROOT / gate.PG2_CONFIG).read_text(encoding="utf-8"))
    optimizer = json.loads((ROOT / gate.PG2_OPTIMIZER).read_text(encoding="utf-8"))
    return primary, supporting, optimizer


def test_pg2_seed17_range_is_exact_scientific_parity_with_independent_scale() -> None:
    result = gate.validate_local(ROOT)
    assert result["status"] == "GREEN_EXACT_PG2_SEED17_RANGE_PARITY"
    assert result["lineage"] == "pg2"
    assert result["role"] == "supporting"
    assert result["learning_rate"] == 1e-5
    assert result["seed"] == 17
    assert result["promotion_eligible"] is False
    assert result["confirmation_reads"] == 0
    assert result["paid_authorization"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["panel"].update(task_id="different"),
        lambda row: row["adapter"].update(rank=8),
        lambda row: row["optimizer"].update(seed=29),
        lambda row: row["optimizer"].update(steps=8),
        lambda row: row["runtime"].update(attention="sdpa"),
        lambda row: row["mechanical_adjudication"].update(minimum_objective_drop=0.0),
    ],
)
def test_pg2_seed17_range_refuses_scientific_parity_drift(mutation) -> None:
    primary, supporting, optimizer = _rows()
    mutation(supporting)
    with pytest.raises(TrainingGateError, match="parity|schedule"):
        gate.validate_config_pair(primary, supporting, optimizer)


def test_pg2_seed17_range_refuses_imported_optimizer_or_primary_role() -> None:
    primary, supporting, optimizer = _rows()
    supporting["optimizer"]["scale_origin"] = "copied_from_pg1"
    with pytest.raises(TrainingGateError, match="origin"):
        gate.validate_config_pair(primary, supporting, optimizer)
    primary, supporting, optimizer = _rows()
    supporting["lineage"]["role"] = "primary"
    with pytest.raises(TrainingGateError, match="lineage firewall"):
        gate.validate_config_pair(primary, supporting, optimizer)


def test_pg2_exact_s3_hydration_manifest_normalizes_to_repository() -> None:
    manifest = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/"
            "parallax_pg2_h100_s3_hydration_job_v1.json"
        ).read_text(encoding="utf-8")
    )
    repository, excluded = _repository_from_manifest(
        manifest,
        repo="google/paligemma2-3b-pt-224",
        revision="96eeb174da13ca1a2b247e4d0867436296c36420",
    )
    assert len(repository["files"]) == 9
    assert sum(row["bytes"] for row in repository["files"]) == 6_099_504_596
    assert excluded == []
