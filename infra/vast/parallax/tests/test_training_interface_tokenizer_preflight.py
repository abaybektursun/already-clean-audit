from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from infra.vast.parallax.training_common import TrainingGateError
from infra.vast.parallax.training_interface_control import SOURCE_PROMPT, SOURCE_TASK_ID
from infra.vast.parallax.training_interface_tokenizer_preflight import audit_processor


class FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<image>"
        return 9


class FakeProcessor:
    image_token_id = 9
    tokenizer = FakeTokenizer()

    def __init__(self, counts: dict[str, int] | None = None):
        self.counts = counts or {"A": 2, "B": 2, "C": 2}

    def __call__(self, *, images, text, suffix, padding, return_tensors):
        assert len(images) == len(text) == len(suffix) == 1
        candidate = suffix[0]
        count = self.counts[candidate]
        length = 40 + count
        input_ids = torch.zeros(1, length, dtype=torch.long)
        input_ids[:, :16] = self.image_token_id
        labels = torch.full((1, length), -100, dtype=torch.long)
        base_id = {"A": 20, "B": 30, "C": 40}[candidate]
        labels[:, -count:] = torch.arange(base_id, base_id + count)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 224, 224),
            "labels": labels,
            "token_type_ids": torch.zeros_like(input_ids),
        }


def _rows(task_root: Path) -> list[dict]:
    result = []
    for index, target in enumerate(("angular_leaf_spot", "bean_rust", "healthy")):
        relative = f"{target}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(task_root / relative)
        result.append(
            {
                "item_id": f"item-{index}",
                "task_id": SOURCE_TASK_ID,
                "split": "discovery_eval",
                "prompt": SOURCE_PROMPT,
                "target": target,
                "candidates": ["angular_leaf_spot", "bean_rust", "healthy"],
                "image": relative,
            }
        )
    return result


def test_processor_gate_exercises_every_codebook_candidate_and_semantic(tmp_path: Path) -> None:
    result = audit_processor(
        processor=FakeProcessor(), source_rows=_rows(tmp_path), task_root=tmp_path
    )
    assert result["status"] == "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
    assert result["candidate_suffix_tokens"] == {"A": 2, "B": 2, "C": 2}
    assert result["processor_calls"] == 27
    assert result["model_loads"] == result["gpu_operations"] == 0
    assert result["confirmation_items_read"] == 0


def test_processor_gate_fails_before_model_load_on_unequal_suffixes(tmp_path: Path) -> None:
    processor = FakeProcessor({"A": 2, "B": 3, "C": 2})
    with pytest.raises(TrainingGateError, match="suffix token counts are unequal"):
        audit_processor(processor=processor, source_rows=_rows(tmp_path), task_root=tmp_path)
