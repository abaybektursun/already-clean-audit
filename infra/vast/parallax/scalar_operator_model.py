"""Model-side finite-difference-JVP/Fisher/VJP mechanics for one PG2 scalar cell."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Sequence

from .behavioral_common import BehavioralGateError, energy_audit, make_field
from .behavioral_model import (
    activation_rms,
    affected_prefix_mask,
    processor_batch,
    site_hook,
    suffix_logits,
    to_device,
)
from .scalar_operator import categorical_fisher_apply, categorical_fisher_rayleigh


def _symmetric_metrics(base: Any, plus: Any, minus: Any, epsilon: float) -> dict[str, Any]:
    import torch

    base_logp = torch.log_softmax(base.float(), dim=-1)
    probabilities = base_logp.exp()
    plus_logp = torch.log_softmax(plus.float(), dim=-1)
    minus_logp = torch.log_softmax(minus.float(), dim=-1)
    kl_plus = torch.sum(probabilities * (base_logp - plus_logp), dim=-1).mean()
    kl_minus = torch.sum(probabilities * (base_logp - minus_logp), dim=-1).mean()
    symmetric_kl = kl_plus + kl_minus
    tangent = (plus.float() - minus.float()) / (2.0 * float(epsilon))
    fisher = categorical_fisher_rayleigh(probabilities, tangent).mean()
    fisher_vector = categorical_fisher_apply(probabilities, tangent).detach()
    values = (kl_plus, kl_minus, symmetric_kl, fisher)
    if not all(bool(torch.isfinite(value)) for value in values) or not bool(torch.isfinite(tangent).all()):
        raise BehavioralGateError("nonfinite scalar operator metric")
    return {
        "kl_plus": float(kl_plus),
        "kl_minus": float(kl_minus),
        "symmetric_kl": float(symmetric_kl),
        "direct_curvature": float(symmetric_kl / float(epsilon) ** 2),
        "fisher_rayleigh": float(fisher),
        "tangent": tangent.detach(),
        "fisher_vector": fisher_vector,
    }


def evaluate_scalar_operator(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    task_root: Path,
    layer: Any,
    directions: Sequence[Any],
    epsilons: Sequence[float],
    namespace: str,
    batch_size: int,
    expected_image_tokens: int,
    per_token_fraction_max: float,
    kl_min: float,
    kl_max: float,
    energy_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(abs(float(epsilon)) > per_token_fraction_max for epsilon in epsilons):
        raise BehavioralGateError("scalar operator epsilon exceeds per-token safety ceiling")
    aggregate: dict[tuple[int, float], dict[str, Any]] = {}
    energy_rows: list[dict[str, Any]] = []
    hook_calls = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        item_ids = [str(row["item_id"]) for row in batch_rows]
        inputs = to_device(processor_batch(processor, batch_rows, task_root))
        active = affected_prefix_mask(inputs).float()
        image_counts = (
            inputs["input_ids"].eq(int(model.config.image_token_index)) & active.bool()
        ).sum(dim=1)
        if not bool(image_counts.eq(int(expected_image_tokens)).all()):
            raise BehavioralGateError(
                f"scalar operator image-token count differs: {image_counts.tolist()}"
            )
        with site_hook(layer, mode="observe") as observed:
            with torch.inference_mode():
                baseline = model(**inputs, use_cache=False, return_dict=True)
        if observed["calls"] != 1:
            raise BehavioralGateError("scalar baseline hook did not fire once")
        hook_calls += 1
        base_rows = [value.detach() for value in suffix_logits(baseline.logits, inputs["labels"])]
        rms = activation_rms(observed["hidden"], active)
        del baseline

        for direction_index, direction in enumerate(directions):
            basis = make_field(
                construction="sensitive_coherent",
                sensitive_direction=direction,
                active_mask=active,
                activation_rms=rms,
                amplitude=1.0,
                namespace=namespace,
                lineage="pg2",
                item_ids=item_ids,
            ).to(device=observed["hidden"].device, dtype=torch.float32)
            previous_tangents: list[Any] | None = None
            for epsilon in epsilons:
                plus_field = (basis * float(epsilon)).to(dtype=observed["hidden"].dtype)
                minus_field = (-basis * float(epsilon)).to(dtype=observed["hidden"].dtype)
                plus_audit = energy_audit(
                    plus_field, active, rms, float(epsilon), relative_tolerance=energy_tolerance
                )
                minus_audit = energy_audit(
                    minus_field, active, rms, float(epsilon), relative_tolerance=energy_tolerance
                )
                energy_rows.append(
                    {
                        "batch_start": start,
                        "direction_index": direction_index,
                        "epsilon": float(epsilon),
                        "plus": plus_audit,
                        "minus": minus_audit,
                    }
                )
                with site_hook(layer, mode="add", field=plus_field) as plus_state:
                    with torch.inference_mode():
                        plus_output = model(**inputs, use_cache=False, return_dict=True)
                with site_hook(layer, mode="add", field=minus_field) as minus_state:
                    with torch.inference_mode():
                        minus_output = model(**inputs, use_cache=False, return_dict=True)
                if plus_state["calls"] != 1 or minus_state["calls"] != 1:
                    raise BehavioralGateError("scalar central-difference hook count differs")
                hook_calls += 2
                plus_rows = suffix_logits(plus_output.logits, inputs["labels"])
                minus_rows = suffix_logits(minus_output.logits, inputs["labels"])
                item_metrics = [
                    _symmetric_metrics(base, plus, minus, float(epsilon))
                    for base, plus, minus in zip(base_rows, plus_rows, minus_rows)
                ]
                del plus_output, minus_output, plus_rows, minus_rows

                effective_basis = (plus_field.float() - minus_field.float()) / (
                    2.0 * float(epsilon)
                )
                with torch.enable_grad():
                    with site_hook(layer, mode="cut_grad") as vjp_state:
                        vjp_output = model(**inputs, use_cache=False, return_dict=True)
                    if vjp_state["calls"] != 1:
                        raise BehavioralGateError("scalar VJP hook did not fire once")
                    differentiable_rows = suffix_logits(vjp_output.logits, inputs["labels"])
                    objective = sum(
                        (logits.float() * metric["fisher_vector"]).sum()
                        / logits.shape[0]
                        for logits, metric in zip(differentiable_rows, item_metrics)
                    ) / len(item_metrics)
                    objective.backward()
                    gradient = vjp_state["hidden"].grad
                    if gradient is None:
                        raise BehavioralGateError("scalar VJP cut gradient is absent")
                    vjp_rayleigh = float(torch.sum(gradient.float() * effective_basis))
                hook_calls += 1
                del vjp_output, differentiable_rows, objective, gradient

                changes = []
                if previous_tangents is not None:
                    for current, previous in zip(
                        [metric["tangent"] for metric in item_metrics], previous_tangents
                    ):
                        numerator = torch.linalg.vector_norm(current - previous)
                        denominator = max(
                            float(torch.linalg.vector_norm(current)),
                            float(torch.linalg.vector_norm(previous)),
                            1e-30,
                        )
                        changes.append(float(numerator) / denominator)
                previous_tangents = [metric["tangent"] for metric in item_metrics]
                symmetric_values = [metric["symmetric_kl"] for metric in item_metrics]
                key = (direction_index, float(epsilon))
                bucket = aggregate.setdefault(
                    key,
                    {
                        "direction_index": direction_index,
                        "direction_kind": "sensitive" if direction_index == 0 else "orthogonal_control",
                        "epsilon": float(epsilon),
                        "item_metrics": [],
                        "vjp_rayleigh_batches": [],
                        "jvp_changes": [],
                    },
                )
                bucket["item_metrics"].extend(
                    {
                        "item_id": item_id,
                        "kl_plus_nats": metric["kl_plus"],
                        "kl_minus_nats": metric["kl_minus"],
                        "symmetric_kl_nats": metric["symmetric_kl"],
                        "symmetric_direct_kl_curvature": metric["direct_curvature"],
                        "finite_difference_fisher_rayleigh": metric["fisher_rayleigh"],
                    }
                    for item_id, metric in zip(item_ids, item_metrics)
                )
                bucket["vjp_rayleigh_batches"].append(vjp_rayleigh)
                bucket["jvp_changes"].extend(changes)
                del plus_field, minus_field, effective_basis, item_metrics, symmetric_values
            del basis, previous_tangents
        del inputs, active, image_counts, base_rows, rms

    cells = []
    for key in sorted(aggregate):
        bucket = aggregate[key]
        item_metrics = bucket.pop("item_metrics")
        vjp_batches = bucket.pop("vjp_rayleigh_batches")
        changes = bucket.pop("jvp_changes")
        symmetric = [float(row["symmetric_kl_nats"]) for row in item_metrics]
        median_kl = statistics.median(symmetric)
        cells.append(
            {
                **bucket,
                "items": len(item_metrics),
                "symmetric_kl_nats": sum(symmetric) / len(symmetric),
                "median_symmetric_kl_nats": median_kl,
                "symmetric_direct_kl_curvature": sum(
                    float(row["symmetric_direct_kl_curvature"]) for row in item_metrics
                )
                / len(item_metrics),
                "finite_difference_fisher_rayleigh": sum(
                    float(row["finite_difference_fisher_rayleigh"]) for row in item_metrics
                )
                / len(item_metrics),
                "vjp_rayleigh": sum(vjp_batches) / len(vjp_batches),
                "jvp_relative_change_from_previous": max(changes) if changes else None,
                "usable": kl_min <= median_kl <= kl_max,
                "item_metrics": item_metrics,
            }
        )
    return cells, {"status": "GREEN", "hook_calls": hook_calls, "energy_audits": energy_rows}
