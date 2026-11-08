"""Real PaliGemma training/evaluation mechanics shared by H100 gates and cells."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .behavioral_model import processor_batch, suffix_logits, to_device
from .graft_common import candidate_rows, item_scores, suffix_log_likelihoods, summarize_items
from .training_common import TrainingGateError, assert_trainable_contract, causal_suffix_loss


def select_split_stratified(
    rows: Iterable[dict[str, Any]], *, task_id: str, split: str, per_target: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task_id") == task_id and row.get("split") == split:
            grouped[str(row["target"])].append(row)
    if not grouped:
        raise TrainingGateError(f"no {split} rows for {task_id}")
    selected: list[dict[str, Any]] = []
    for target, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: str(row["item_id"]))
        if len(ordered) < per_target:
            raise TrainingGateError(f"underfilled {split} target {target}")
        selected.extend(ordered[:per_target])
    selected.sort(key=lambda row: (str(row["target"]), str(row["item_id"])))
    if len({str(row["item_id"]) for row in selected}) != len(selected):
        raise TrainingGateError("training panel contains duplicate item IDs")
    return selected


def _attention_configs(model: Any) -> list[Any]:
    return [
        value
        for value in (model.config, model.config.text_config, model.config.vision_config)
        if hasattr(value, "_attn_implementation")
    ]


def attach_free_lora_connector(model: Any, adapter: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Freeze the base, attach LM-only LoRA, and expose the native connector in fp32."""
    from peft import LoraConfig, TaskType, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    leaves = "|".join(str(value) for value in adapter["target_modules"])
    target_regex = rf".*language_model.*\.({leaves})$"
    config = LoraConfig(
        r=int(adapter["rank"]),
        lora_alpha=int(adapter["alpha"]),
        lora_dropout=float(adapter["dropout"]),
        bias=str(adapter["bias"]),
        target_modules=target_regex,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config, autocast_adapter_dtype=True)
    connector_names = []
    for name, parameter in model.named_parameters():
        if "multi_modal_projector" in name:
            parameter.requires_grad_(True)
            parameter.data = parameter.data.float()
            connector_names.append(name)
        elif parameter.requires_grad:
            parameter.data = parameter.data.float()
    if not connector_names:
        raise TrainingGateError("no connector parameters matched after LoRA attachment")
    leaves_regex = "|".join(str(value) for value in adapter["target_modules"])
    report = assert_trainable_contract(
        model,
        allowed_fullmatch_patterns=[
            rf".*language_model.*\.({leaves_regex})\.lora_[AB]\.default\.weight",
            r".*multi_modal_projector\..*",
        ],
        forbidden_substrings=["vision_tower"],
        require_lora=True,
    )
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad})
    if dtypes != ["torch.float32"]:
        raise TrainingGateError(f"trainable parameters are not all fp32: {dtypes}")
    report.update(
        {
            "target_regex": target_regex,
            "connector_names": connector_names,
            "trainable_dtypes": dtypes,
        }
    )
    return model, report


def suffix_anchor(model: Any, batch: Mapping[str, Any]) -> list[Any]:
    import torch

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True)
    return [value.detach().cpu() for value in suffix_logits(output.logits, batch["labels"])]


def one_update(
    *, model: Any, optimizer: Any, batch: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss(output.logits, batch["labels"])
    loss.backward()
    grad_sq = 0.0
    tensors_with_grad = 0
    missing_grad = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_grad.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise TrainingGateError(f"nonfinite trainable gradient: {name}")
        grad_sq += float(parameter.grad.detach().float().square().sum())
        tensors_with_grad += 1
    gradient_norm = math.sqrt(grad_sq)
    if tensors_with_grad == 0 or not math.isfinite(gradient_norm) or gradient_norm == 0:
        raise TrainingGateError("one-update gate has zero or nonfinite gradient")
    # LoRA A can legitimately have zero/None gradient at zero-B initialization,
    # but the connector and LoRA B must produce a resolved update.
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "trainable_tensors_with_grad": tensors_with_grad,
        "trainable_tensors_without_grad": missing_grad,
    }


def _rng_snapshot() -> dict[str, Any]:
    import torch

    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return bool(
        torch.equal(left["cpu"], right["cpu"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))
    )


def evaluate_candidates(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    task_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    expanded = candidate_rows(rows)
    for row in expanded:
        if row["prompt"].startswith("<image>"):
            raise TrainingGateError("task prompt already contains an image marker")
        row["prompt"] = "<image>" + row["prompt"]
    image_cache = {}

    def image(relative: str):
        if relative not in image_cache:
            with Image.open(task_root / relative) as handle:
                image_cache[relative] = handle.convert("RGB").copy()
        return image_cache[relative]

    before_rng = _rng_snapshot()
    scored = []
    model.eval()
    for start in range(0, len(expanded), batch_size):
        batch_rows = expanded[start : start + batch_size]
        inputs = processor(
            images=[image(row["image"]) for row in batch_rows],
            text=[row["prompt"] for row in batch_rows],
            suffix=[row["candidate"] for row in batch_rows],
            padding=True,
            return_tensors="pt",
        )
        required = {"input_ids", "attention_mask", "pixel_values", "labels", "token_type_ids"}
        if not required.issubset(inputs):
            raise TrainingGateError(f"candidate processor contract missing: {sorted(required - set(inputs))}")
        device_inputs = to_device(dict(inputs))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**device_inputs, use_cache=False, return_dict=True)
        sums, means, counts = suffix_log_likelihoods(output.logits, device_inputs["labels"])
        for row, total, mean, count in zip(batch_rows, sums, means, counts):
            if not math.isfinite(total) or not math.isfinite(mean):
                raise TrainingGateError("candidate evaluation is nonfinite")
            scored.append(
                {
                    "item_id": row["item_id"],
                    "task_id": row["task_id"],
                    "candidate": row["candidate"],
                    "target": row["target"],
                    "log_likelihood_sum": total,
                    "log_likelihood_mean": mean,
                    "suffix_tokens": int(count),
                }
            )
        del inputs, device_inputs, output
    after_rng = _rng_snapshot()
    if not _rng_equal(before_rng, after_rng):
        raise TrainingGateError("post-checkpoint evaluation changed RNG state")
    items = item_scores(scored)
    return {
        "status": "GREEN",
        "rng_isolated": True,
        "candidate_rows": scored,
        "items": items,
        "summary": summarize_items(items),
    }


def real_training_batch(processor: Any, rows: Sequence[dict[str, Any]], task_root: Path) -> dict[str, Any]:
    batch = processor_batch(processor, rows, task_root)
    return to_device(batch)

