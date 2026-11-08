from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from infra.vast.parallax import training_interface_control_cell as legacy_cell
from infra.vast.parallax import training_interface_control_primary_cell as primary
from infra.vast.parallax.training_common import TrainingGateError
from infra.vast.parallax.training_interface_control import CODEBOOKS, CODES, SEMANTIC_LABELS


ROOT = Path(__file__).resolve().parents[4]
CONFIG = (
    ROOT
    / "studies/lineage/cross-lineage/config/parallax_training_interface_control_pg1_primary_v1.yaml"
)


def _matrix(mapping: dict[str, str], value: float) -> dict[str, dict[str, float]]:
    matrix = {semantic: {code: 0.0 for code in CODES} for semantic in SEMANTIC_LABELS}
    for semantic in SEMANTIC_LABELS:
        matrix[semantic][mapping[semantic]] = value
    return matrix


def _rows(mapping: dict[str, str], matrix: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for semantic in SEMANTIC_LABELS:
        for index in range(4):
            for code in CODES:
                rows.append(
                    {
                        "item_id": f"{semantic}-{index}",
                        "target": mapping[semantic],
                        "candidate": code,
                        "suffix_tokens": 2,
                        "log_likelihood_sum": matrix[semantic][code],
                    }
                )
    return rows


def _packet(values: dict[str, float] | None = None) -> dict:
    values = values or {codebook_id: 1.0 for codebook_id in CODEBOOKS}
    codebooks = []
    for codebook_id, mapping in CODEBOOKS.items():
        base_rows = _rows(mapping, _matrix(mapping, 0.0))
        final_rows = _rows(mapping, _matrix(mapping, values[codebook_id]))
        codebooks.append(
            {
                "codebook_id": codebook_id,
                "mapping": mapping,
                "snapshots": [
                    {"step": 0, "evaluation": {"candidate_rows": base_rows}},
                    {"step": 16, "evaluation": {"candidate_rows": final_rows}},
                ],
            }
        )
    return {
        "lineage": "pg1_pi",
        "status": "GREEN_MECHANICAL_EXECUTION",
        "adjudication": {"verdict": "LEGACY_DIAGNOSTIC"},
        "codebooks": codebooks,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
    }


def _fast_adjudicate(packet: dict) -> dict:
    return primary.adjudicate_primary_packet(
        packet, bootstrap_replicates=200, permutation_replicates=199
    )


def test_primary_config_is_independent_nonpromotional_and_paid_off() -> None:
    config = primary._validate_primary_config(CONFIG)
    assert config["lineage_order"][0]["id"] == "pg1_pi"
    assert config["primary_within_lineage"]["supporting_pg2_dependency"] == "none_within_lineage"
    assert config["primary_within_lineage"]["paired_or_shared_range_claim"] == "forbidden"
    assert config["paid_authorization"] is False


def test_primary_config_rejects_a_hidden_pg2_dependency_or_paid_authority(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["primary_within_lineage"]["supporting_pg2_dependency"] = "pg2_green"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(TrainingGateError, match="escaped"):
        primary._validate_primary_config(path)

    config["primary_within_lineage"]["supporting_pg2_dependency"] = "none_within_lineage"
    config["paid_authorization"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(TrainingGateError, match="escaped"):
        primary._validate_primary_config(path)


def test_primary_interaction_is_the_verdict_and_probability_is_diagnostic() -> None:
    result = _fast_adjudicate(_packet())
    assert result["status"] == "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION"
    assert result["adjudication"]["verdict"] == "GREEN_PROVISIONAL_IMAGE_CONDITIONED_RANGE"
    assert result["diagnostic_probability_adjudication"]["verdict"] == "LEGACY_DIAGNOSTIC"
    assert result["supporting_pg2_dependency"] == "none_within_lineage"
    assert result["paired_or_shared_range_claim"] is False
    assert result["promotion_eligible"] is False


def test_primary_positive_average_cannot_hide_a_bad_codebook() -> None:
    result = _fast_adjudicate(_packet({"cycle_0": -1.0, "cycle_1": 2.0, "cycle_2": 2.0}))
    assert result["adjudication"]["mapping_marginalized_interaction_gain"] > 0
    assert result["adjudication"]["verdict"] == "RED_CODEBOOK_NONUNIFORM"


def test_primary_packet_refuses_confirmation_or_missing_endpoints() -> None:
    confirmation = _packet()
    confirmation["confirmation_items_read"] = 1
    with pytest.raises(TrainingGateError, match="confirmation"):
        _fast_adjudicate(confirmation)

    missing = _packet()
    missing["codebooks"][0]["snapshots"][-1]["step"] = 8
    with pytest.raises(TrainingGateError, match="endpoints"):
        _fast_adjudicate(missing)


def test_primary_adapter_restores_historical_allowed_contract(monkeypatch, tmp_path: Path) -> None:
    original = dict(legacy_cell.ALLOWED["pg1_pi"])

    def fake_run(**kwargs):
        assert kwargs["lineage"] == "pg1_pi"
        assert legacy_cell.ALLOWED["pg1_pi"]["role"] == primary.ROLE
        kwargs["output_dir"].mkdir()
        return _packet()

    monkeypatch.setattr(legacy_cell, "run", fake_run)
    monkeypatch.setattr(
        primary,
        "adjudicate_codebooks",
        lambda _rows, **_kwargs: {"metric_id": "token_prior_invariant_association_v1", "verdict": "RED_TEST"},
    )
    result = primary.run(
        root=ROOT,
        source_contract=tmp_path / "source.json",
        matrix=tmp_path / "matrix.yaml",
        config_path=CONFIG,
        tokenizer_preflight=tmp_path / "preflight.json",
        cache_root=tmp_path / "cache",
        task_root=tmp_path / "tasks",
        instance_id=1,
        output_dir=tmp_path / "output",
    )
    assert result["adjudication"]["verdict"] == "RED_TEST"
    assert legacy_cell.ALLOWED["pg1_pi"] == original
    assert (tmp_path / "output/interface-control.json").is_file()


def test_primary_adapter_restores_historical_contract_on_failure(monkeypatch, tmp_path: Path) -> None:
    original = dict(legacy_cell.ALLOWED["pg1_pi"])

    def fail(**_kwargs):
        assert legacy_cell.ALLOWED["pg1_pi"]["role"] == primary.ROLE
        raise RuntimeError("planted failure")

    monkeypatch.setattr(legacy_cell, "run", fail)
    with pytest.raises(RuntimeError, match="planted"):
        primary.run(
            root=ROOT,
            source_contract=tmp_path / "source.json",
            matrix=tmp_path / "matrix.yaml",
            config_path=CONFIG,
            tokenizer_preflight=tmp_path / "preflight.json",
            cache_root=tmp_path / "cache",
            task_root=tmp_path / "tasks",
            instance_id=1,
            output_dir=tmp_path / "output",
        )
    assert legacy_cell.ALLOWED["pg1_pi"] == original
