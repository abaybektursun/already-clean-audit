"""Run and bank the weight-free PARALLAX training plants."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .artifacts import atomic_json
from .training_common import (
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


def _fixture_rows(*, shuffled: bool = False, repeats: int = 24) -> list[dict[str, Any]]:
    import torch

    targets = [(token + 1) % 4 for token in range(4)] * repeats
    if shuffled:
        targets = [(index // 4) % 4 for index in range(len(targets))]
    return [
        {
            "input_ids": torch.tensor([index % 4, 0], dtype=torch.long),
            "labels": torch.tensor([-100, target], dtype=torch.long),
        }
        for index, target in enumerate(targets)
    ]


def _make(seed: int):
    import torch
    from torch import nn

    class ToyLoRASequence(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(4, 4, bias=False)
            self.lora_A = nn.Linear(4, 4, bias=False)
            self.lora_B = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.base.weight.zero_()
                nn.init.orthogonal_(self.lora_A.weight)
                self.lora_B.weight.zero_()
            self.base.weight.requires_grad_(False)

        def forward(self, input_ids, use_cache=False, return_dict=True, **_kwargs):
            features = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            return SimpleNamespace(
                logits=self.base(features) + self.lora_B(self.lora_A(features))
            )

    torch.manual_seed(seed)
    model = ToyLoRASequence()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.08)
    return model, optimizer


def _mean_loss(model: Any, rows: list[dict[str, Any]]) -> float:
    import torch

    with torch.inference_mode():
        values = []
        for row in rows:
            output = model(input_ids=row["input_ids"].unsqueeze(0), use_cache=False, return_dict=True)
            values.append(float(causal_suffix_loss(output.logits, row["labels"].unsqueeze(0))))
    return sum(values) / len(values)


def run(output: Path) -> dict[str, Any]:
    import torch

    data = _fixture_rows()
    shuffled = _fixture_rows(shuffled=True)
    schedule = deterministic_batches(item_count=len(data), batch_size=16, steps=35, seed=29)
    schedule_hash = schedule_sha256(schedule)

    model, optimizer = _make(29)
    trainable = assert_trainable_contract(
        model,
        allowed_fullmatch_patterns=[r"lora_[AB]\.weight"],
        forbidden_substrings=["vision"],
    )
    input_ids = torch.tensor([[0, 0], [3, 0]])
    with torch.inference_mode():
        base = model.base(torch.nn.functional.one_hot(input_ids, num_classes=4).float())
        zero_delta_identity = bool(torch.equal(base, model(input_ids=input_ids).logits))
    if not zero_delta_identity:
        raise TrainingGateError("planted zero-delta LoRA is not an identity")

    baseline = _mean_loss(model, data)
    frozen_before = named_parameter_sha256(model, trainable=False)
    metrics = train_steps(
        model=model,
        optimizer=optimizer,
        rows=data,
        schedule=schedule,
        start_step=0,
        stop_step=35,
        device="cpu",
    )
    final = _mean_loss(model, data)
    frozen_after = named_parameter_sha256(model, trainable=False)
    if final >= baseline - 1.0 or frozen_before != frozen_after:
        raise TrainingGateError("learnable plant or frozen-byte plant failed")

    shuffle_model, shuffle_optimizer = _make(29)
    train_steps(
        model=shuffle_model,
        optimizer=shuffle_optimizer,
        rows=shuffled,
        schedule=schedule,
        start_step=0,
        stop_step=35,
        device="cpu",
    )
    shuffle_gain = baseline - _mean_loss(shuffle_model, data)
    true_gain = baseline - final
    if shuffle_gain >= 0.25 or true_gain <= 1.0:
        raise TrainingGateError("label-shuffle plant crossed the held-out gain band")

    resume_schedule = deterministic_batches(item_count=len(data), batch_size=16, steps=24, seed=41)
    resume_hash = schedule_sha256(resume_schedule)
    full, full_optimizer = _make(41)
    train_steps(model=full, optimizer=full_optimizer, rows=data, schedule=resume_schedule,
                start_step=0, stop_step=24, device="cpu")
    interrupted, interrupted_optimizer = _make(41)
    train_steps(model=interrupted, optimizer=interrupted_optimizer, rows=data,
                schedule=resume_schedule, start_step=0, stop_step=11, device="cpu")
    with tempfile.TemporaryDirectory(prefix="parallax-training-plant-") as temporary:
        checkpoint = Path(temporary) / "checkpoint.pt"
        checkpoint_hash = atomic_training_checkpoint(
            checkpoint,
            model=interrupted,
            optimizer=interrupted_optimizer,
            step=11,
            schedule_sha256=resume_hash,
            metadata={"fixture": "resume"},
        )
        resumed, resumed_optimizer = _make(41)
        restored = restore_training_checkpoint(
            checkpoint,
            model=resumed,
            optimizer=resumed_optimizer,
            expected_schedule_sha256=resume_hash,
        )
        train_steps(model=resumed, optimizer=resumed_optimizer, rows=data,
                    schedule=resume_schedule, start_step=11, stop_step=24, device="cpu")
    resumed_parameters = named_parameter_sha256(resumed, trainable=True)
    full_parameters = named_parameter_sha256(full, trainable=True)
    resumed_optimizer_hash = optimizer_state_sha256(resumed_optimizer)
    full_optimizer_hash = optimizer_state_sha256(full_optimizer)
    exact_resume = resumed_parameters == full_parameters and resumed_optimizer_hash == full_optimizer_hash
    if not exact_resume:
        raise TrainingGateError("interrupted plant did not reproduce uninterrupted training")

    negative_gates = {}
    try:
        causal_suffix_loss(torch.zeros(2, 3, 4), torch.full((2, 3), -100))
    except TrainingGateError:
        negative_gates["all_masked_refused"] = True
    escaped, _ = _make(17)
    escaped.base.weight.requires_grad_(True)
    try:
        assert_trainable_contract(escaped, allowed_fullmatch_patterns=[r"lora_[AB]\.weight"])
    except TrainingGateError:
        negative_gates["allowlist_escape_refused"] = True
    if set(negative_gates) != {"all_masked_refused", "allowlist_escape_refused"}:
        raise TrainingGateError("a negative planted gate did not fire")

    base_adjudication = dict(
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
    branches = {}
    fixtures = {
        "ABORT_NONFINITE": {"finite": False},
        "VOID_READOUT_FLOOR": {"at_floor": True},
        "VOID_READOUT_CEILING": {"at_ceiling": True},
        "VOID_SHUFFLE_CONTROL": {"shuffled_gain": 0.3},
        "RED_OBJECTIVE_NOT_MOVING": {"objective_drop": 0.01},
        "NO_HELDOUT_RANGE": {"heldout_gain": 0.01},
        "GREEN_MECHANICAL": {},
    }
    for expected, changes in fixtures.items():
        values = dict(base_adjudication)
        values.update(changes)
        observed = mechanical_adjudicate(**values)
        branches[expected] = observed
        if observed != expected:
            raise TrainingGateError(f"adjudicator plant expected {expected}, observed {observed}")

    packet = {
        "schema_version": 1,
        "stage": "training_harness_weight_free_gates",
        "status": "GREEN",
        "device": "cpu",
        "torch": torch.__version__,
        "zero_delta_identity": zero_delta_identity,
        "trainable_contract": trainable,
        "learnable_plant": {
            "initial_loss": baseline,
            "final_loss": final,
            "objective_drop": true_gain,
            "first_step_loss": metrics[0]["loss"],
            "last_step_loss": metrics[-1]["loss"],
        },
        "label_shuffle_plant": {
            "true_heldout_gain": true_gain,
            "shuffle_heldout_gain": shuffle_gain,
            "maximum_allowed_shuffle_gain": 0.25,
        },
        "frozen_parameter_bytes": {
            "before_sha256": frozen_before,
            "after_sha256": frozen_after,
            "exact": frozen_before == frozen_after,
        },
        "determinism": {
            "schedule_sha256": schedule_hash,
            "resume_schedule_sha256": resume_hash,
            "checkpoint_sha256": checkpoint_hash,
            "restored_checkpoint_sha256": restored["checkpoint_sha256"],
            "uninterrupted_trainable_sha256": full_parameters,
            "resumed_trainable_sha256": resumed_parameters,
            "uninterrupted_optimizer_sha256": full_optimizer_hash,
            "resumed_optimizer_sha256": resumed_optimizer_hash,
            "exact_resume": exact_resume,
        },
        "negative_gates": negative_gates,
        "adjudicator_branches": branches,
        "claim_boundary": "Mechanical planted evidence only; no model, task-range, or scientific claim.",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.output.resolve())
    print(packet["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
