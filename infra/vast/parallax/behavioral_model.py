"""Shared PaliGemma activation-intervention mechanics for behavioral calibration and confirmation."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .behavioral_common import BehavioralGateError, energy_audit, make_field


def to_device(batch: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def decoder_layer(model: Any, index: int) -> Any:
    layers = model.model.language_model.layers
    if index < 0 or index >= len(layers):
        raise BehavioralGateError(f"language layer out of range: {index}/{len(layers)}")
    return layers[index]


def _hidden(output: Any) -> Any:
    if hasattr(output, "ndim") and output.ndim == 3:
        return output
    if isinstance(output, tuple) and output and hasattr(output[0], "ndim") and output[0].ndim == 3:
        return output[0]
    raise BehavioralGateError(f"unexpected decoder-layer output: {type(output).__name__}")


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if hasattr(output, "ndim") and output.ndim == 3:
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise BehavioralGateError(f"cannot replace decoder-layer output: {type(output).__name__}")


@contextmanager
def site_hook(layer: Any, *, mode: str, field: Any | None = None) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"calls": 0, "hidden": None}

    def hook(_module: Any, _args: tuple[Any, ...], output: Any):
        hidden = _hidden(output)
        state["calls"] += 1
        if mode == "observe":
            state["hidden"] = hidden.detach()
            return None
        if mode == "cut_grad":
            replacement = hidden.detach().requires_grad_(True)
            replacement.retain_grad()
            state["hidden"] = replacement
            return _replace_hidden(output, replacement)
        if mode == "add":
            if field is None or tuple(field.shape) != tuple(hidden.shape):
                raise BehavioralGateError("intervention field shape mismatch")
            replacement = hidden + field.to(device=hidden.device, dtype=hidden.dtype)
            state["hidden"] = hidden.detach()
            return _replace_hidden(output, replacement)
        raise BehavioralGateError(f"unknown hook mode: {mode}")

    handle = layer.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def processor_batch(processor: Any, rows: Sequence[dict[str, Any]], task_root: Path) -> dict[str, Any]:
    from PIL import Image

    images = []
    prompts = []
    targets = []
    for row in rows:
        with Image.open(task_root / row["image"]) as image:
            images.append(image.convert("RGB").copy())
        prompt = str(row["prompt"])
        if prompt.startswith("<image>"):
            raise BehavioralGateError("task prompt already contains a PaliGemma image marker")
        prompts.append("<image>" + prompt)
        targets.append(str(row["target"]))
    batch = processor(
        images=images,
        text=prompts,
        suffix=targets,
        padding=True,
        return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "labels", "token_type_ids"}
    if not required.issubset(batch):
        raise BehavioralGateError(f"processor suffix contract missing: {sorted(required - set(batch))}")
    if batch["labels"].shape != batch["input_ids"].shape:
        raise BehavioralGateError("processor labels do not align with input IDs")
    suffix_mask = batch["labels"].ne(-100)
    if not bool(suffix_mask.any(dim=1).all()):
        raise BehavioralGateError("processor produced an item without suffix labels")
    if bool((suffix_mask & batch["attention_mask"].eq(0)).any()):
        raise BehavioralGateError("processor labeled a padding position")
    return dict(batch)


def suffix_logits(logits: Any, labels: Any) -> list[Any]:
    shifted = logits[:, :-1, :].float()
    mask = labels[:, 1:].ne(-100)
    result = [shifted[index][mask[index]] for index in range(shifted.shape[0])]
    if any(value.shape[0] == 0 for value in result):
        raise BehavioralGateError("suffix processor produced no labeled token")
    return result


def affected_prefix_mask(inputs: dict[str, Any]) -> Any:
    """Select image+prompt positions and exclude teacher-forced target tokens."""
    attention = inputs["attention_mask"].bool()
    suffix = inputs["labels"].ne(-100)
    affected = attention & ~suffix
    if not bool(affected.any(dim=1).all()):
        raise BehavioralGateError("an item has no affected image/prompt prefix positions")
    if bool((affected & suffix).any()):
        raise BehavioralGateError("teacher-forced suffix leaked into the intervention mask")
    return affected


def suffix_metrics(base: Sequence[Any], perturbed: Sequence[Any], labels: Any) -> list[dict[str, float | int]]:
    import torch

    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    results = []
    for index, (base_logits, perturbed_logits) in enumerate(zip(base, perturbed)):
        targets = shifted_labels[index][mask[index]]
        if base_logits.shape != perturbed_logits.shape or base_logits.shape[0] != targets.shape[0]:
            raise BehavioralGateError("suffix metric shape mismatch")
        base_logp = torch.log_softmax(base_logits.float(), dim=-1)
        perturbed_logp = torch.log_softmax(perturbed_logits.float(), dim=-1)
        probabilities = base_logp.exp()
        kl = torch.sum(probabilities * (base_logp - perturbed_logp), dim=-1).mean()
        base_nll = -base_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
        perturbed_nll = -perturbed_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
        values = (kl, base_nll, perturbed_nll)
        if not all(bool(torch.isfinite(value)) for value in values):
            raise BehavioralGateError("nonfinite suffix metric")
        results.append(
            {
                "suffix_tokens": int(targets.numel()),
                "kl_nats": float(kl),
                "base_target_nll": float(base_nll),
                "perturbed_target_nll": float(perturbed_nll),
                "damage_nats": float(perturbed_nll - base_nll),
            }
        )
    return results


def activation_rms(hidden: Any, active_mask: Any) -> Any:
    import torch

    mask = active_mask.float().unsqueeze(-1)
    denominator = active_mask.sum(dim=-1).float().clamp_min(1) * hidden.shape[-1]
    values = torch.sum(hidden.float().square() * mask, dim=(1, 2)) / denominator
    result = torch.sqrt(values)
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise BehavioralGateError("activation RMS is zero or nonfinite")
    return result


def fixed_parameter_versions(model: Any) -> dict[str, tuple[int, int, int]]:
    return {
        name: (id(parameter), int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in model.named_parameters()
    }


def assert_parameter_versions(model: Any, expected: dict[str, tuple[int, int, int]]) -> None:
    observed = fixed_parameter_versions(model)
    if observed != expected:
        changed = sorted(key for key in set(expected) | set(observed) if expected.get(key) != observed.get(key))
        raise BehavioralGateError(f"activation intervention changed parameters: {changed[:8]}")


def identity_intervention(model: Any, inputs: dict[str, Any], layer: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        baseline = model(**inputs, use_cache=False, return_dict=True)
    base_rows = [value.detach().cpu() for value in suffix_logits(baseline.logits, inputs["labels"])]
    del baseline
    with site_hook(layer, mode="observe") as observed:
        with torch.inference_mode():
            observed_output = model(**inputs, use_cache=False, return_dict=True)
    observed_rows = [value.detach().cpu() for value in suffix_logits(observed_output.logits, inputs["labels"])]
    del observed_output
    exact = observed["calls"] == 1 and all(torch.equal(a, b) for a, b in zip(base_rows, observed_rows))
    if not exact:
        raise BehavioralGateError("observational site hook changed suffix logits or did not fire once")
    zero = torch.zeros_like(observed["hidden"])
    with site_hook(layer, mode="add", field=zero) as added:
        with torch.inference_mode():
            added_output = model(**inputs, use_cache=False, return_dict=True)
    added_rows = [value.detach().cpu() for value in suffix_logits(added_output.logits, inputs["labels"])]
    del added_output
    zero_exact = added["calls"] == 1 and all(torch.equal(a, b) for a, b in zip(base_rows, added_rows))
    if not zero_exact:
        raise BehavioralGateError("zero intervention changed suffix logits or did not fire once")
    digest = hashlib.sha256()
    for row in base_rows:
        digest.update(row[:, :256].float().contiguous().numpy().tobytes())
    return {
        "status": "GREEN",
        "observational_exact": True,
        "zero_add_exact": True,
        "hook_calls": {"observe": observed["calls"], "zero_add": added["calls"]},
        "anchor_sha256": digest.hexdigest(),
    }


def discovery_direction(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    task_root: Path,
    layer: Any,
    batch_size: int,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    total = None
    diagnostics = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        inputs = to_device(processor_batch(processor, batch_rows, task_root))
        with site_hook(layer, mode="cut_grad") as state:
            output = model(**inputs, use_cache=False, return_dict=True)
        if state["calls"] != 1:
            raise BehavioralGateError(f"gradient hook fired {state['calls']} times")
        logits = output.logits[:, :-1, :].float()
        labels = inputs["labels"][:, 1:]
        mask = labels.ne(-100)
        loss = F.cross_entropy(logits[mask], labels[mask], reduction="mean")
        loss.backward()
        gradient = state["hidden"].grad
        if gradient is None:
            raise BehavioralGateError("site gradient was not retained")
        active = affected_prefix_mask(inputs).float().unsqueeze(-1)
        batch_direction = torch.sum(gradient.float() * active, dim=(0, 1)).detach().cpu()
        total = batch_direction if total is None else total + batch_direction
        diagnostics.append(
            {
                "items": len(batch_rows),
                "target_nll": float(loss.detach()),
                "gradient_norm": float(batch_direction.norm()),
                "hook_calls": state["calls"],
            }
        )
        del inputs, output, logits, labels, mask, loss, gradient
    if total is None or not bool(torch.isfinite(total).all()) or float(total.norm()) == 0:
        raise BehavioralGateError("discovery direction is zero or nonfinite")
    return total / total.norm(), diagnostics


def evaluate_constructions(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    task_root: Path,
    layer: Any,
    sensitive_direction: Any,
    lineage: str,
    namespace: str,
    amplitudes: Sequence[float],
    constructions: Sequence[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    output_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    hook_calls = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        item_ids = [str(row["item_id"]) for row in batch_rows]
        inputs = to_device(processor_batch(processor, batch_rows, task_root))
        active = affected_prefix_mask(inputs).float()
        with site_hook(layer, mode="observe") as observed:
            with torch.inference_mode():
                baseline = model(**inputs, use_cache=False, return_dict=True)
        if observed["calls"] != 1:
            raise BehavioralGateError(f"baseline site hook fired {observed['calls']} times")
        hook_calls += observed["calls"]
        rms = activation_rms(observed["hidden"], active)
        base_rows = [value.detach() for value in suffix_logits(baseline.logits, inputs["labels"])]
        del baseline
        for construction in constructions:
            for amplitude in amplitudes:
                field = make_field(
                    construction=construction,
                    sensitive_direction=sensitive_direction,
                    active_mask=active,
                    activation_rms=rms,
                    amplitude=float(amplitude),
                    namespace=namespace,
                    lineage=lineage,
                    item_ids=item_ids,
                )
                # This is the field actually added to the BF16 residual stream.  Audit
                # after the cast so equal-energy controls cannot diverge through dtype.
                field = field.to(device=observed["hidden"].device, dtype=observed["hidden"].dtype)
                audit = energy_audit(field, active, rms, float(amplitude), relative_tolerance=5e-3)
                energy_rows.append({"construction": construction, "amplitude": float(amplitude), **audit})
                with site_hook(layer, mode="add", field=field) as perturbed_state:
                    with torch.inference_mode():
                        perturbed = model(**inputs, use_cache=False, return_dict=True)
                if perturbed_state["calls"] != 1:
                    raise BehavioralGateError(
                        f"perturbation site hook fired {perturbed_state['calls']} times"
                    )
                hook_calls += perturbed_state["calls"]
                perturbed_rows = suffix_logits(perturbed.logits, inputs["labels"])
                metrics = suffix_metrics(base_rows, perturbed_rows, inputs["labels"])
                for item, metric, item_rms in zip(batch_rows, metrics, rms):
                    output_rows.append(
                        {
                            "item_id": str(item["item_id"]),
                            "target": str(item["target"]),
                            "construction": construction,
                            "amplitude": float(amplitude),
                            "activation_rms": float(item_rms),
                            **metric,
                        }
                    )
                del field, perturbed, perturbed_rows
        del inputs, active, rms, base_rows
    return output_rows, {
        "status": "GREEN",
        "total_hook_calls": hook_calls,
        "energy_audits": energy_rows,
    }
