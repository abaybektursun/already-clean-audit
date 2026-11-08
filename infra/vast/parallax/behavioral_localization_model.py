"""Model-side mechanics for the PG2 prefix-position localization assay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .behavioral_common import BehavioralGateError, energy_audit, make_field
from .behavioral_localization import (
    equal_total_energy_rms,
    prefix_location_masks,
    total_energy_audit,
)
from .behavioral_model import (
    activation_rms,
    processor_batch,
    site_hook,
    suffix_logits,
    suffix_metrics,
    to_device,
)


ARM_CONTRACT = {
    "full_sensitive": ("full_prefix", "sensitive_coherent"),
    "full_rotated": ("full_prefix", "rotated_coherent"),
    "full_white": ("full_prefix", "white"),
    "image_sensitive": ("image_tokens", "sensitive_coherent"),
    "image_rotated": ("image_tokens", "rotated_coherent"),
    "prompt_sensitive": ("prompt_tokens", "sensitive_coherent"),
    "prompt_rotated": ("prompt_tokens", "rotated_coherent"),
}


def evaluate_localization(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    task_root: Path,
    layer: Any,
    sensitive_direction: Any,
    namespace: str,
    amplitudes: Sequence[float],
    arms: Sequence[str],
    batch_size: int,
    expected_image_tokens: int,
    per_token_fraction_max: float,
    total_energy_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    unknown = set(arms) - set(ARM_CONTRACT)
    if unknown:
        raise BehavioralGateError(f"unknown localization arms: {sorted(unknown)}")
    output_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    hook_calls = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        item_ids = [str(row["item_id"]) for row in batch_rows]
        inputs = to_device(processor_batch(processor, batch_rows, task_root))
        masks = prefix_location_masks(
            inputs,
            image_token_index=int(model.config.image_token_index),
            expected_image_tokens=expected_image_tokens,
        )
        with site_hook(layer, mode="observe") as observed:
            with torch.inference_mode():
                baseline = model(**inputs, use_cache=False, return_dict=True)
        if observed["calls"] != 1:
            raise BehavioralGateError("localization baseline hook did not fire exactly once")
        hook_calls += observed["calls"]
        base_rows = [value.detach() for value in suffix_logits(baseline.logits, inputs["labels"])]
        base_rms = activation_rms(observed["hidden"], masks["full_prefix"].float())
        del baseline
        for arm in arms:
            location, construction = ARM_CONTRACT[arm]
            location_mask = masks[location].float()
            scaled_rms = equal_total_energy_rms(
                base_rms,
                reference_mask=masks["full_prefix"],
                location_mask=location_mask,
            )
            scale = torch.sqrt(
                masks["full_prefix"].sum(dim=1).float() / location_mask.sum(dim=1).float()
            )
            for amplitude in amplitudes:
                per_token_fractions = scale * float(amplitude)
                if bool((per_token_fractions > per_token_fraction_max).any()):
                    audits.append(
                        {
                            "arm": arm,
                            "amplitude": float(amplitude),
                            "status": "SKIPPED_UNSAFE_PER_TOKEN_AMPLITUDE",
                            "max_per_token_activation_rms_fraction": float(
                                per_token_fractions.max()
                            ),
                            "ceiling": per_token_fraction_max,
                            "intervention_executed": False,
                        }
                    )
                    continue
                field = make_field(
                    construction=construction,
                    sensitive_direction=sensitive_direction,
                    active_mask=location_mask,
                    activation_rms=scaled_rms,
                    amplitude=float(amplitude),
                    namespace=namespace,
                    lineage="pg2",
                    item_ids=item_ids,
                ).to(device=observed["hidden"].device, dtype=observed["hidden"].dtype)
                per_token_audit = energy_audit(
                    field,
                    location_mask,
                    scaled_rms,
                    float(amplitude),
                    relative_tolerance=total_energy_tolerance,
                )
                total_audit = total_energy_audit(
                    field,
                    location_mask=location_mask,
                    reference_mask=masks["full_prefix"],
                    activation_rms=base_rms,
                    amplitude=float(amplitude),
                    relative_tolerance=total_energy_tolerance,
                )
                audits.append(
                    {
                        "arm": arm,
                        "amplitude": float(amplitude),
                        "per_token": per_token_audit,
                        "total": total_audit,
                    }
                )
                with site_hook(layer, mode="add", field=field) as perturbed_state:
                    with torch.inference_mode():
                        perturbed = model(**inputs, use_cache=False, return_dict=True)
                if perturbed_state["calls"] != 1:
                    raise BehavioralGateError("localization perturbation hook did not fire once")
                hook_calls += perturbed_state["calls"]
                metrics = suffix_metrics(
                    base_rows, suffix_logits(perturbed.logits, inputs["labels"]), inputs["labels"]
                )
                for item, metric, item_rms, fraction in zip(
                    batch_rows, metrics, base_rms, per_token_fractions
                ):
                    output_rows.append(
                        {
                            "item_id": str(item["item_id"]),
                            "target": str(item["target"]),
                            "arm": arm,
                            "location": location,
                            "construction": construction,
                            "amplitude": float(amplitude),
                            "per_token_activation_rms_fraction": float(fraction),
                            "activation_rms": float(item_rms),
                            **metric,
                        }
                    )
                del field, perturbed, metrics
        del inputs, masks, base_rows, base_rms
    return output_rows, {"status": "GREEN", "hook_calls": hook_calls, "energy_audits": audits}
