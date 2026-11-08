"""Fail-closed training mechanics for the PARALLAX Vast runtime.

This module has no provider, launcher, model-repository, or storage assumptions.
It is intentionally small enough to exercise with planted CPU fixtures before a
real PaliGemma checkpoint or paid H100 is touched.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import random
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class TrainingGateError(RuntimeError):
    """A mechanical training invariant failed; no scientific result is licensed."""


def causal_suffix_loss(logits: Any, labels: Any) -> Any:
    """Compute next-token loss only at non-masked suffix labels."""
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2:
        raise TrainingGateError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise TrainingGateError("labels do not align with logits")
    if logits.shape[1] < 2:
        raise TrainingGateError("causal suffix loss requires at least two sequence positions")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    active = shifted_labels.ne(-100)
    if not bool(active.any()):
        raise TrainingGateError("all target tokens are masked")
    if bool(((shifted_labels < 0) & shifted_labels.ne(-100)).any()):
        raise TrainingGateError("labels contain an unsupported negative token ID")
    if bool((shifted_labels[active] >= logits.shape[-1]).any()):
        raise TrainingGateError("a target token is outside the vocabulary")
    loss = F.cross_entropy(shifted_logits[active], shifted_labels[active], reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise TrainingGateError("suffix loss is nonfinite")
    return loss


def deterministic_batches(
    *, item_count: int, batch_size: int, steps: int, seed: int
) -> list[list[int]]:
    """Return a replayable epoch-shuffled batch schedule without global RNG use."""
    import torch

    if item_count <= 0 or batch_size <= 0 or steps < 0:
        raise TrainingGateError("item_count and batch_size must be positive; steps must be nonnegative")
    batches: list[list[int]] = []
    epoch = 0
    while len(batches) < steps:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + epoch * 1_000_003)
        order = torch.randperm(item_count, generator=generator).tolist()
        for start in range(0, item_count, batch_size):
            batch = order[start : start + batch_size]
            if len(batch) == batch_size:
                batches.append(batch)
                if len(batches) == steps:
                    break
        epoch += 1
        if item_count < batch_size:
            raise TrainingGateError("item_count is smaller than batch_size; no complete batch exists")
    return batches


def _tensor_bytes(tensor: Any) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(dtype=__import__("torch").uint8).numpy().tobytes()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def named_parameter_sha256(model: Any, *, trainable: bool | None) -> str:
    """Hash names, dtypes, shapes, and every parameter byte in a selected scope."""
    digest = hashlib.sha256()
    selected = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda row: row[0]):
        if trainable is not None and bool(parameter.requires_grad) != trainable:
            continue
        selected += 1
        header = f"{name}\0{parameter.dtype}\0{tuple(parameter.shape)}\0".encode()
        digest.update(header)
        digest.update(_tensor_bytes(parameter))
    if selected == 0:
        raise TrainingGateError("parameter hash selected no parameters")
    return digest.hexdigest()


def assert_trainable_contract(
    model: Any,
    *,
    allowed_fullmatch_patterns: Sequence[str],
    forbidden_substrings: Sequence[str] = (),
    require_lora: bool = True,
) -> dict[str, Any]:
    """Require every trainable parameter to match an explicit allowlist."""
    compiled = [re.compile(pattern) for pattern in allowed_fullmatch_patterns]
    names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not names:
        raise TrainingGateError("model has no trainable parameters")
    escaped = [name for name in names if not any(pattern.fullmatch(name) for pattern in compiled)]
    if escaped:
        raise TrainingGateError(f"trainable parameters escaped allowlist: {escaped[:8]}")
    forbidden = [name for name in names if any(value in name for value in forbidden_substrings)]
    if forbidden:
        raise TrainingGateError(f"forbidden parameters are trainable: {forbidden[:8]}")
    lora_names = [name for name in names if "lora_" in name]
    if require_lora and not lora_names:
        raise TrainingGateError("no trainable LoRA parameters were found")
    trainable = sum(parameter.numel() for _, parameter in model.named_parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "status": "GREEN",
        "trainable_names": names,
        "trainable_parameter_count": int(trainable),
        "total_parameter_count": int(total),
        "trainable_fraction": float(trainable / total),
        "lora_parameter_count": int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and "lora_" in name
            )
        ),
    }


def trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: Any, state: Mapping[str, Any]) -> None:
    import torch

    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(current) != set(state):
        missing = sorted(set(current) - set(state))
        extra = sorted(set(state) - set(current))
        raise TrainingGateError(f"trainable checkpoint key mismatch: missing={missing[:8]} extra={extra[:8]}")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise TrainingGateError(f"trainable checkpoint shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda"):
        if not torch.cuda.is_available():
            raise TrainingGateError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(list(state["torch_cuda"]))


def atomic_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    schedule_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Atomically save only trainable tensors plus optimizer and RNG state."""
    import torch

    if step < 0 or len(schedule_sha256) != 64:
        raise TrainingGateError("checkpoint step or schedule hash is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "step": int(step),
        "schedule_sha256": schedule_sha256,
        "trainable_state": trainable_state(model),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "metadata": dict(metadata or {}),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def restore_training_checkpoint(
    path: Path, *, model: Any, optimizer: Any, expected_schedule_sha256: str
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise TrainingGateError("unsupported training checkpoint schema")
    if payload.get("schedule_sha256") != expected_schedule_sha256:
        raise TrainingGateError("training schedule hash mismatch")
    restore_trainable_state(model, payload["trainable_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_state(payload["rng_state"])
    return {
        "step": int(payload["step"]),
        "schedule_sha256": str(payload["schedule_sha256"]),
        "metadata": dict(payload.get("metadata", {})),
        "checkpoint_sha256": _sha256_file(path),
    }


def schedule_sha256(schedule: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for batch in schedule:
        digest.update(",".join(str(int(index)) for index in batch).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def optimizer_state_sha256(optimizer: Any) -> str:
    buffer = io.BytesIO()
    __import__("torch").save(optimizer.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def train_steps(
    *,
    model: Any,
    optimizer: Any,
    rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Sequence[int]],
    start_step: int,
    stop_step: int,
    device: str,
) -> list[dict[str, float | int]]:
    """Run an exact interval of a precomputed schedule using the audited loss."""
    import torch

    if not (0 <= start_step <= stop_step <= len(schedule)):
        raise TrainingGateError("invalid training interval")
    model.train()
    metrics: list[dict[str, float | int]] = []
    for step in range(start_step, stop_step):
        indices = schedule[step]
        keys = set(rows[indices[0]])
        if any(set(rows[index]) != keys for index in indices):
            raise TrainingGateError("batch rows have inconsistent tensor keys")
        batch = {
            key: torch.stack([rows[index][key] for index in indices]).to(device)
            for key in sorted(keys)
        }
        if "labels" not in batch:
            raise TrainingGateError("training batch has no labels")
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss(output.logits, batch["labels"])
        loss.backward()
        grad_sq = 0.0
        trainable_with_grad = 0
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise TrainingGateError(f"nonfinite gradient at step {step}")
            grad_sq += float(parameter.grad.detach().float().square().sum())
            trainable_with_grad += 1
        grad_norm = math.sqrt(grad_sq)
        if trainable_with_grad == 0 or not math.isfinite(grad_norm) or grad_norm == 0:
            raise TrainingGateError(f"zero or nonfinite trainable gradient at step {step}")
        before = named_parameter_sha256(model, trainable=True)
        optimizer.step()
        after = named_parameter_sha256(model, trainable=True)
        if before == after:
            raise TrainingGateError(f"optimizer produced no resolved parameter update at step {step}")
        metrics.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "gradient_norm": grad_norm,
                "trainable_tensors_with_grad": trainable_with_grad,
            }
        )
    return metrics


def mechanical_adjudicate(
    *,
    finite: bool,
    objective_drop: float,
    heldout_gain: float,
    shuffled_gain: float,
    at_floor: bool,
    at_ceiling: bool,
    minimum_objective_drop: float,
    minimum_heldout_gain: float,
    maximum_shuffle_gain: float,
) -> str:
    """Mechanical harness branches only; this is not the G3 promotion rule."""
    if not finite:
        return "ABORT_NONFINITE"
    if at_floor:
        return "VOID_READOUT_FLOOR"
    if at_ceiling:
        return "VOID_READOUT_CEILING"
    if shuffled_gain > maximum_shuffle_gain:
        return "VOID_SHUFFLE_CONTROL"
    if objective_drop < minimum_objective_drop:
        return "RED_OBJECTIVE_NOT_MOVING"
    if heldout_gain < minimum_heldout_gain:
        return "NO_HELDOUT_RANGE"
    return "GREEN_MECHANICAL"
