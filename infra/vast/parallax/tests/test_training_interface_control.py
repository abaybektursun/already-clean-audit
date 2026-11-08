from __future__ import annotations

import math

import pytest
import torch

from infra.vast.parallax.training_common import TrainingGateError
from infra.vast.parallax.training_common import causal_suffix_loss
from infra.vast.parallax.training_interface_control import CODEBOOKS
from infra.vast.parallax.training_interface_control import CONTROL_TASK_ID
from infra.vast.parallax.training_interface_control import SOURCE_PROMPT
from infra.vast.parallax.training_interface_control import SOURCE_TASK_ID
from infra.vast.parallax.training_interface_control import audit_equal_suffix_counts
from infra.vast.parallax.training_interface_control import causal_suffix_loss_per_example
from infra.vast.parallax.training_interface_control import remap_rows
from infra.vast.parallax.training_interface_control import semantic_summary
from infra.vast.parallax.training_interface_control import one_update_per_example
from infra.vast.parallax.training_interface_control import validate_codebooks


def _source_row(target: str, item_id: str = "item-1") -> dict:
    return {
        "item_id": item_id,
        "task_id": SOURCE_TASK_ID,
        "prompt": SOURCE_PROMPT,
        "target": target,
        "candidates": ["angular_leaf_spot", "bean_rust", "healthy"],
        "image": "image.png",
        "split": "train",
    }


def test_three_codebooks_are_a_complete_latin_square() -> None:
    result = validate_codebooks()
    assert result["status"] == "GREEN_LATIN_SQUARE_CODEBOOKS"
    assert result["each_semantic_uses_each_code_once"] is True
    assert {mapping["healthy"] for mapping in CODEBOOKS.values()} == {"A", "B", "C"}


def test_row_remap_changes_only_the_output_interface() -> None:
    source = _source_row("bean_rust")
    remapped = remap_rows([source], codebook_id="cycle_1")[0]
    assert source["target"] == "bean_rust"
    assert remapped["target"] == "C"
    assert remapped["semantic_target"] == "bean_rust"
    assert remapped["task_id"] == CONTROL_TASK_ID
    assert remapped["candidates"] == ["A", "B", "C"]
    assert remapped["image"] == source["image"]
    assert "A=healthy; B=angular_leaf_spot; C=bean_rust" in remapped["prompt"]


def test_row_remap_refuses_source_prompt_or_candidate_drift() -> None:
    row = _source_row("healthy")
    row["prompt"] = "changed"
    with pytest.raises(TrainingGateError, match="frozen bean interface"):
        remap_rows([row], codebook_id="cycle_0")


def test_per_example_loss_removes_suffix_length_weighting() -> None:
    # Example 0 has one active token with CE=1. Example 1 has three active
    # tokens with CE=3. Token pooling gives 2.5; example pooling gives 2.0.
    logits = torch.zeros(2, 4, 2)
    logits[0, 0, 1] = math.log(math.exp(1.0) - 1.0)
    logits[1, :3, 1] = math.log(math.exp(3.0) - 1.0)
    labels = torch.tensor([[-100, 0, -100, -100], [-100, 0, 0, 0]])
    token_pooled = float(causal_suffix_loss(logits, labels))
    example_pooled = float(causal_suffix_loss_per_example(logits, labels))
    assert token_pooled == pytest.approx(2.5, abs=1e-6)
    assert example_pooled == pytest.approx(2.0, abs=1e-6)


def test_per_example_loss_refuses_an_unlabeled_example() -> None:
    logits = torch.zeros(2, 3, 4)
    labels = torch.tensor([[-100, 0, -100], [-100, -100, -100]])
    with pytest.raises(TrainingGateError, match="every interface-control example"):
        causal_suffix_loss_per_example(logits, labels)


def test_example_balanced_update_changes_trainable_parameters() -> None:
    from types import SimpleNamespace

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 3, 4))

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=self.logits.expand(input_ids.shape[0], -1, -1))

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.logits.detach().clone()
    batch = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2]]),
    }
    result = one_update_per_example(model=model, optimizer=optimizer, batch=batch)
    assert result["gradient_norm"] > 0
    assert result["loss_reduction"].startswith("mean_of_per_example")
    assert not torch.equal(before, model.logits)


def test_equal_suffix_audit_proves_sum_mean_ranking_identity() -> None:
    rows = []
    for item_id in ("one", "two"):
        for index, candidate in enumerate(("A", "B", "C")):
            mean = -1.0 - index
            rows.append(
                {
                    "item_id": item_id,
                    "candidate": candidate,
                    "suffix_tokens": 2,
                    "log_likelihood_mean": mean,
                    "log_likelihood_sum": 2 * mean,
                }
            )
    result = audit_equal_suffix_counts(rows)
    assert result == {
        "status": "GREEN_EQUAL_SUFFIX_COUNTS",
        "candidate_suffix_tokens": {"A": 2, "B": 2, "C": 2},
        "equal_count": 2,
        "items_with_sum_mean_ranking_identity": 2,
    }
    rows[-1]["suffix_tokens"] = 3
    with pytest.raises(TrainingGateError, match="unstable|unequal"):
        audit_equal_suffix_counts(rows)


def test_semantic_summary_does_not_confuse_code_with_class() -> None:
    transformed = [
        remap_rows([_source_row(label, f"item-{index}")], codebook_id="cycle_2")[0]
        for index, label in enumerate(("angular_leaf_spot", "bean_rust", "healthy"))
    ]
    items = [
        {
            "item_id": row["item_id"],
            "correct": int(index != 1),
            "target_probability": 0.2 + index * 0.1,
        }
        for index, row in enumerate(transformed)
    ]
    result = semantic_summary(items, transformed)
    assert result["aggregate"]["accuracy"] == pytest.approx(2 / 3)
    assert result["by_semantic_target"]["bean_rust"]["accuracy"] == 0.0
    assert result["by_semantic_target"]["healthy"]["mean_target_probability"] == 0.4
