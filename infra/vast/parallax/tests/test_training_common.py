from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from infra.vast.parallax.training_common import (
    TrainingGateError,
    assert_trainable_contract,
    atomic_training_checkpoint,
    causal_suffix_loss,
    deterministic_batches,
    mechanical_adjudicate,
    named_parameter_sha256,
    optimizer_state_sha256,
    restore_training_checkpoint,
    schedule_sha256,
    train_steps,
)
from infra.vast.parallax.training_planted_gate import run as run_planted_gate


class ToyLoRASequence(nn.Module):
    """Frozen identity features plus a zero-delta low-rank token writer."""

    def __init__(self, vocab: int = 4, rank: int = 4):
        super().__init__()
        self.vocab = vocab
        self.base = nn.Linear(vocab, vocab, bias=False)
        self.lora_A = nn.Linear(vocab, rank, bias=False)
        self.lora_B = nn.Linear(rank, vocab, bias=False)
        with torch.no_grad():
            self.base.weight.zero_()
            nn.init.orthogonal_(self.lora_A.weight)
            self.lora_B.weight.zero_()
        self.base.weight.requires_grad_(False)

    def forward(self, input_ids, use_cache=False, return_dict=True, **_kwargs):
        features = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab).float()
        logits = self.base(features) + self.lora_B(self.lora_A(features))
        return SimpleNamespace(logits=logits)


def rows(*, shuffled: bool = False, repeats: int = 24) -> list[dict[str, torch.Tensor]]:
    result = []
    true_targets = [(token + 1) % 4 for token in range(4)] * repeats
    targets = list(true_targets)
    if shuffled:
        # Every input token sees every target equally often.  This is a strict
        # planted no-signal control, not a lucky finite random permutation.
        targets = [(index // 4) % 4 for index in range(len(targets))]
    for index, target in enumerate(targets):
        token = index % 4
        result.append(
            {
                "input_ids": torch.tensor([token, 0], dtype=torch.long),
                "labels": torch.tensor([-100, target], dtype=torch.long),
            }
        )
    return result


def mean_loss(model: nn.Module, data: list[dict[str, torch.Tensor]]) -> float:
    with torch.inference_mode():
        values = []
        for row in data:
            output = model(input_ids=row["input_ids"].unsqueeze(0), use_cache=False, return_dict=True)
            values.append(float(causal_suffix_loss(output.logits, row["labels"].unsqueeze(0))))
    return sum(values) / len(values)


def make(seed: int = 17):
    torch.manual_seed(seed)
    model = ToyLoRASequence()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.08)
    return model, optimizer


def test_masked_suffix_loss_refuses_all_masked_and_bad_shapes():
    logits = torch.zeros(2, 3, 8)
    with pytest.raises(TrainingGateError, match="all target tokens"):
        causal_suffix_loss(logits, torch.full((2, 3), -100))
    with pytest.raises(TrainingGateError, match="align"):
        causal_suffix_loss(logits, torch.zeros(2, 2, dtype=torch.long))


def test_zero_delta_identity_and_trainable_allowlist():
    model, _ = make()
    inputs = torch.tensor([[0, 0], [3, 0]])
    with torch.inference_mode():
        expected = model.base(torch.nn.functional.one_hot(inputs, num_classes=4).float())
        observed = model(input_ids=inputs).logits
    assert torch.equal(expected, observed)
    report = assert_trainable_contract(
        model,
        allowed_fullmatch_patterns=[r"lora_[AB]\.weight"],
        forbidden_substrings=["vision"],
    )
    assert report["lora_parameter_count"] == 32
    model.base.weight.requires_grad_(True)
    with pytest.raises(TrainingGateError, match="escaped allowlist"):
        assert_trainable_contract(model, allowed_fullmatch_patterns=[r"lora_[AB]\.weight"])


def test_learnable_plant_moves_objective_and_preserves_frozen_bytes():
    model, optimizer = make()
    data = rows()
    schedule = deterministic_batches(item_count=len(data), batch_size=16, steps=30, seed=17)
    initial = mean_loss(model, data)
    frozen_before = named_parameter_sha256(model, trainable=False)
    metrics = train_steps(
        model=model,
        optimizer=optimizer,
        rows=data,
        schedule=schedule,
        start_step=0,
        stop_step=30,
        device="cpu",
    )
    final = mean_loss(model, data)
    assert final < initial - 1.0
    assert metrics[-1]["loss"] < metrics[0]["loss"]
    assert named_parameter_sha256(model, trainable=False) == frozen_before


def test_shuffled_labels_do_not_fake_true_heldout_gain():
    true_data = rows()
    shuffled_data = rows(shuffled=True)
    schedule = deterministic_batches(item_count=len(true_data), batch_size=16, steps=35, seed=29)
    true_model, true_optimizer = make(29)
    shuffle_model, shuffle_optimizer = make(29)
    baseline = mean_loss(true_model, true_data)
    train_steps(model=true_model, optimizer=true_optimizer, rows=true_data, schedule=schedule,
                start_step=0, stop_step=35, device="cpu")
    train_steps(model=shuffle_model, optimizer=shuffle_optimizer, rows=shuffled_data, schedule=schedule,
                start_step=0, stop_step=35, device="cpu")
    true_gain = baseline - mean_loss(true_model, true_data)
    shuffle_gain = baseline - mean_loss(shuffle_model, true_data)
    assert true_gain > 1.0
    assert shuffle_gain < 0.25


def test_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path):
    data = rows()
    schedule = deterministic_batches(item_count=len(data), batch_size=16, steps=24, seed=41)
    digest = schedule_sha256(schedule)

    full, full_opt = make(41)
    train_steps(model=full, optimizer=full_opt, rows=data, schedule=schedule,
                start_step=0, stop_step=24, device="cpu")

    interrupted, interrupted_opt = make(41)
    train_steps(model=interrupted, optimizer=interrupted_opt, rows=data, schedule=schedule,
                start_step=0, stop_step=11, device="cpu")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_hash = atomic_training_checkpoint(
        checkpoint,
        model=interrupted,
        optimizer=interrupted_opt,
        step=11,
        schedule_sha256=digest,
        metadata={"fixture": "resume"},
    )

    resumed, resumed_opt = make(41)
    restored = restore_training_checkpoint(
        checkpoint, model=resumed, optimizer=resumed_opt, expected_schedule_sha256=digest
    )
    assert restored["step"] == 11
    assert restored["checkpoint_sha256"] == checkpoint_hash
    assert restored["metadata"] == {"fixture": "resume"}
    train_steps(model=resumed, optimizer=resumed_opt, rows=data, schedule=schedule,
                start_step=11, stop_step=24, device="cpu")

    assert named_parameter_sha256(resumed, trainable=True) == named_parameter_sha256(full, trainable=True)
    assert optimizer_state_sha256(resumed_opt) == optimizer_state_sha256(full_opt)


def test_schedule_is_replayable_and_seed_sensitive():
    left = deterministic_batches(item_count=19, batch_size=4, steps=12, seed=7)
    assert left == deterministic_batches(item_count=19, batch_size=4, steps=12, seed=7)
    assert left != deterministic_batches(item_count=19, batch_size=4, steps=12, seed=8)
    assert len({index for batch in left[:4] for index in batch}) == 16


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"finite": False}, "ABORT_NONFINITE"),
        ({"at_floor": True}, "VOID_READOUT_FLOOR"),
        ({"at_ceiling": True}, "VOID_READOUT_CEILING"),
        ({"shuffled_gain": 0.3}, "VOID_SHUFFLE_CONTROL"),
        ({"objective_drop": 0.01}, "RED_OBJECTIVE_NOT_MOVING"),
        ({"heldout_gain": 0.01}, "NO_HELDOUT_RANGE"),
        ({}, "GREEN_MECHANICAL"),
    ],
)
def test_every_mechanical_adjudicator_branch(overrides, expected):
    values = dict(
        finite=True,
        objective_drop=0.8,
        heldout_gain=0.7,
        shuffled_gain=0.0,
        at_floor=False,
        at_ceiling=False,
        minimum_objective_drop=0.2,
        minimum_heldout_gain=0.2,
        maximum_shuffle_gain=0.1,
    )
    values.update(overrides)
    assert mechanical_adjudicate(**values) == expected


def test_checkpoint_rejects_wrong_schedule(tmp_path: Path):
    model, optimizer = make()
    checkpoint = tmp_path / "checkpoint.pt"
    atomic_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=0,
        schedule_sha256="a" * 64,
    )
    with pytest.raises(TrainingGateError, match="schedule hash"):
        restore_training_checkpoint(
            checkpoint, model=model, optimizer=optimizer, expected_schedule_sha256="b" * 64
        )


def test_banked_planted_gate_packet_is_green(tmp_path: Path):
    output = tmp_path / "gate.json"
    packet = run_planted_gate(output)
    assert output.is_file()
    assert packet["status"] == "GREEN"
    assert packet["determinism"]["exact_resume"] is True
    assert packet["frozen_parameter_bytes"]["exact"] is True
