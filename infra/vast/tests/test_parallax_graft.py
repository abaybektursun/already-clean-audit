from __future__ import annotations

import math

import pytest
import torch

from infra.vast.parallax.graft_common import (
    GraftGateError,
    assert_vision_state_compatible,
    candidate_rows,
    discovery_range,
    item_scores,
    select_discovery_panel,
    shuffled_patch_features,
    suffix_log_likelihoods,
    wrong_image_map,
)


def _row(task: str, target: str, index: int) -> dict[str, object]:
    return {
        "item_id": f"{task}-{target}-{index:02d}",
        "task_id": task,
        "split": "discovery_eval",
        "prompt": "choose",
        "target": target,
        "candidates": ["a", "b"],
        "image": f"images/{task}-{target}-{index}.png",
        "image_sha256": f"pixels-{task}-{target}-{index}",
    }


def test_panel_selection_is_frozen_within_target_strata() -> None:
    rows = [_row("task", target, index) for target in ("a", "b") for index in range(4)]
    selected = select_discovery_panel(reversed(rows), {"task": 2})
    assert [row["item_id"] for row in selected] == [
        "task-a-00",
        "task-a-01",
        "task-b-00",
        "task-b-01",
    ]


def test_panel_selection_refuses_underfilled_stratum() -> None:
    with pytest.raises(GraftGateError, match="insufficient"):
        select_discovery_panel([_row("task", "a", 0)], {"task": 2})


def test_wrong_image_control_is_a_pixel_derangement() -> None:
    rows = [_row("task", "a", index) for index in range(3)]
    mapping = wrong_image_map(rows)
    assert set(mapping) == {row["item_id"] for row in rows}
    assert all(mapping[row["item_id"]] != row["image"] for row in rows)


def test_candidate_expansion_requires_registered_target() -> None:
    row = _row("task", "a", 0)
    assert len(candidate_rows([row])) == 2
    row["target"] = "missing"
    with pytest.raises(GraftGateError, match="target absent"):
        candidate_rows([row])


def test_frozen_graft_runner_makes_paligemma_image_marker_explicit() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "infra/vast/parallax/graft_discovery.py"
    ).read_text(encoding="utf-8")
    assert 'row["prompt"] = "<image>" + row["prompt"]' in source


def test_suffix_likelihood_uses_only_processor_labels() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 2, 1]])
    logits[0, 1, 2] = 2.0
    logits[0, 2, 1] = 1.0
    sums, means, counts = suffix_log_likelihoods(logits, labels)
    expected = torch.log_softmax(logits[0, 1], dim=-1)[2] + torch.log_softmax(logits[0, 2], dim=-1)[1]
    assert sums == pytest.approx([float(expected)])
    assert means == pytest.approx([float(expected) / 2])
    assert counts == [2]


def test_item_score_is_candidate_softmax_not_cross_lineage_logit_comparison() -> None:
    rows = [
        {
            "item_id": "one",
            "task_id": "task",
            "candidate": candidate,
            "target": "a",
            "log_likelihood_sum": score,
        }
        for candidate, score in (("a", -1.0), ("b", -2.0))
    ]
    result = item_scores(rows)[0]
    assert result["prediction"] == "a"
    assert result["correct"] == 1
    assert result["target_probability"] == pytest.approx(1 / (1 + math.exp(-1)))


def test_discovery_range_requires_both_null_controls() -> None:
    def cells(values: list[float]) -> list[dict[str, object]]:
        return [
            {"item_id": str(index), "task_id": "task", "target_probability": value}
            for index, value in enumerate(values)
        ]

    green = discovery_range(
        cells([0.9] * 12), cells([0.2] * 12), cells([0.3] * 12), minimum_raw_effect=0.02, seed=1
    )
    assert green["status"] == "GREEN_RANGE"
    void = discovery_range(
        cells([0.9] * 12), cells([0.2] * 12), cells([0.89] * 12), minimum_raw_effect=0.02, seed=1
    )
    assert void["status"] == "VOID_NO_NATIVE_RANGE"


def test_vision_state_contract_refuses_even_one_extra_key() -> None:
    host = {"weight": torch.zeros(2, 2)}
    assert assert_vision_state_compatible(host, {"weight": torch.ones(2, 2)})["keys"] == 1
    with pytest.raises(GraftGateError, match="extra=1"):
        assert_vision_state_compatible(host, {"weight": torch.ones(2, 2), "bias": torch.ones(2)})


def test_shuffled_feature_control_reverses_only_patch_axis_and_counts_calls() -> None:
    projector = torch.nn.Identity()
    hidden = torch.arange(12).reshape(1, 4, 3)
    with shuffled_patch_features(projector) as counts:
        observed = projector(hidden)
    assert torch.equal(observed, hidden[:, [3, 2, 1, 0], :])
    assert counts == {"calls": 1, "tokens": 4}


def test_gateway_rehearsal_cannot_read_scientific_task_bank() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "infra/vast/parallax/graft_gateway_rehearsal.py").read_text(
        encoding="utf-8"
    )
    assert "FrozenTaskBank" not in source
    assert "iter_discovery" not in source
    assert '"discovery_items_read": 0' in source
    assert '"confirmation_items_read": 0' in source
