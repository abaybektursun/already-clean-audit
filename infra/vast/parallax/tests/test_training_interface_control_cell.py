from __future__ import annotations

import copy

import pytest
import yaml

from infra.vast.parallax.training_interface_control import SEMANTIC_LABELS
from infra.vast.parallax.training_interface_control_cell import _aggregate_adjudication


def _config() -> dict:
    return yaml.safe_load(
        """
provisional_mechanical_adjudication:
  minimum_objective_drop_per_passing_codebook: 0.01
  minimum_mapping_marginalized_probability_gain: 0.01
  minimum_codebooks_with_positive_probability_gain: 2
  every_semantic_mapping_marginalized_gain_strictly_positive: true
  floor_probability: 0.02
  ceiling_probability: 0.98
  all_values_finite: true
  promotion_use: forbidden
"""
    )


def _cell(gain: float = 0.06, semantic_gains: dict[str, float] | None = None) -> dict:
    semantic_gains = semantic_gains or {semantic: gain for semantic in SEMANTIC_LABELS}
    initial_by = {
        semantic: {"mean_target_probability": 0.3} for semantic in SEMANTIC_LABELS
    }
    final_by = {
        semantic: {"mean_target_probability": 0.3 + semantic_gains[semantic]}
        for semantic in SEMANTIC_LABELS
    }
    return {
        "snapshots": [
            {
                "objective": 1.0,
                "semantic_summary_sum": {
                    "aggregate": {"mean_target_probability": 0.3},
                    "by_semantic_target": initial_by,
                },
            },
            {
                "objective": 0.7,
                "semantic_summary_sum": {
                    "aggregate": {"mean_target_probability": 0.3 + gain},
                    "by_semantic_target": final_by,
                },
            },
        ],
        "adjudication": {
            "objective_drop": 0.3,
            "registered_control_probability_gain": gain,
            "positive_heldout_gain": gain > 0,
        },
    }


def test_aggregate_gate_requires_mapping_robust_class_uniform_range() -> None:
    green = _aggregate_adjudication([_cell(), _cell(), _cell()], _config())
    assert green["verdict"] == "GREEN_PROVISIONAL_INTERFACE_CONTROL"
    assert green["positive_gain_codebooks"] == 3
    assert green["mapping_marginalized_probability_gain"] == pytest.approx(0.06)

    fragile = _aggregate_adjudication([_cell(), _cell(-0.01), _cell(-0.01)], _config())
    assert fragile["verdict"] == "RED_MAPPING_FRAGILE"

    nonuniform_cell = _cell(
        gain=0.02,
        semantic_gains={
            "angular_leaf_spot": 0.05,
            "bean_rust": 0.03,
            "healthy": -0.02,
        },
    )
    nonuniform = _aggregate_adjudication(
        [copy.deepcopy(nonuniform_cell) for _ in range(3)], _config()
    )
    assert nonuniform["verdict"] == "RED_SEMANTIC_NONUNIFORM"
