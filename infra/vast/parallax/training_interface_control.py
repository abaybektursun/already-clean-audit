"""Pure mechanics for the PARALLAX equal-token range interface control.

The control counterbalances three semantic bean classes over three cyclic
A/B/C codebooks.  It also uses a loss that gives every example equal weight,
regardless of the number of processor-labeled suffix tokens.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .training_common import TrainingGateError


SOURCE_TASK_ID = "vr_a_beans_balanced"
CONTROL_TASK_ID = "vr_a_beans_balanced_interface_abc_v1"
SOURCE_PROMPT = (
    "Identify the condition of the bean leaf. Answer with one label: "
    "angular_leaf_spot, bean_rust, healthy."
)
SEMANTIC_LABELS = ("angular_leaf_spot", "bean_rust", "healthy")
CODES = ("A", "B", "C")
CODEBOOKS: dict[str, dict[str, str]] = {
    "cycle_0": {
        "angular_leaf_spot": "A",
        "bean_rust": "B",
        "healthy": "C",
    },
    "cycle_1": {
        "angular_leaf_spot": "B",
        "bean_rust": "C",
        "healthy": "A",
    },
    "cycle_2": {
        "angular_leaf_spot": "C",
        "bean_rust": "A",
        "healthy": "B",
    },
}


def validate_codebooks(codebooks: Mapping[str, Mapping[str, str]] = CODEBOOKS) -> dict[str, Any]:
    if tuple(codebooks) != ("cycle_0", "cycle_1", "cycle_2"):
        raise TrainingGateError("interface codebook order changed")
    expected_semantics = set(SEMANTIC_LABELS)
    expected_codes = set(CODES)
    for codebook_id, mapping in codebooks.items():
        if set(mapping) != expected_semantics or set(mapping.values()) != expected_codes:
            raise TrainingGateError(f"interface codebook is not a bijection: {codebook_id}")
    for semantic in SEMANTIC_LABELS:
        if {mapping[semantic] for mapping in codebooks.values()} != expected_codes:
            raise TrainingGateError(f"semantic class is not counterbalanced: {semantic}")
    return {
        "status": "GREEN_LATIN_SQUARE_CODEBOOKS",
        "codebook_ids": list(codebooks),
        "semantic_labels": list(SEMANTIC_LABELS),
        "codes": list(CODES),
        "each_semantic_uses_each_code_once": True,
    }


def control_prompt(mapping: Mapping[str, str]) -> str:
    inverse = {code: semantic for semantic, code in mapping.items()}
    if set(inverse) != set(CODES):
        raise TrainingGateError("control prompt received a non-bijective codebook")
    assignments = "; ".join(f"{code}={inverse[code]}" for code in CODES)
    return (
        "Identify the condition of the bean leaf. "
        f"Use this codebook: {assignments}. "
        "Answer with exactly one code: A, B, or C."
    )


def remap_rows(
    rows: Sequence[Mapping[str, Any]], *, codebook_id: str
) -> list[dict[str, Any]]:
    validate_codebooks()
    mapping = CODEBOOKS.get(codebook_id)
    if mapping is None:
        raise TrainingGateError(f"unknown interface codebook: {codebook_id}")
    result = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        item_id = str(row.get("item_id", ""))
        semantic = str(row.get("target", ""))
        if (
            row.get("task_id") != SOURCE_TASK_ID
            or row.get("prompt") != SOURCE_PROMPT
            or tuple(row.get("candidates", ())) != SEMANTIC_LABELS
            or semantic not in SEMANTIC_LABELS
            or not item_id
        ):
            raise TrainingGateError(f"source row escaped the frozen bean interface: {item_id}")
        if item_id in seen:
            raise TrainingGateError(f"duplicate source item in interface control: {item_id}")
        seen.add(item_id)
        row.update(
            {
                "task_id": CONTROL_TASK_ID,
                "source_task_id": SOURCE_TASK_ID,
                "source_prompt": SOURCE_PROMPT,
                "prompt": control_prompt(mapping),
                "semantic_target": semantic,
                "semantic_candidates": list(SEMANTIC_LABELS),
                "target": mapping[semantic],
                "candidates": list(CODES),
                "codebook_id": codebook_id,
            }
        )
        result.append(row)
    if not result:
        raise TrainingGateError("interface control remapped zero rows")
    return result


def causal_suffix_loss_per_example(logits: Any, labels: Any) -> Any:
    """Mean each example's active-token CE, then give every example equal weight."""
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2:
        raise TrainingGateError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    if tuple(logits.shape[:2]) != tuple(labels.shape) or logits.shape[1] < 2:
        raise TrainingGateError("labels do not align with causal logits")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    active = shifted_labels.ne(-100)
    counts = active.sum(dim=1)
    if bool((counts <= 0).any()):
        raise TrainingGateError("every interface-control example must have a labeled suffix")
    if bool(((shifted_labels < 0) & shifted_labels.ne(-100)).any()):
        raise TrainingGateError("labels contain an unsupported negative token ID")
    if bool((shifted_labels[active] >= logits.shape[-1]).any()):
        raise TrainingGateError("a target token is outside the vocabulary")
    safe = shifted_labels.masked_fill(~active, 0)
    token_losses = F.cross_entropy(
        shifted_logits.reshape(-1, logits.shape[-1]),
        safe.reshape(-1),
        reduction="none",
    ).reshape_as(safe)
    per_example = (token_losses * active).sum(dim=1) / counts
    loss = per_example.mean()
    if not bool(torch.isfinite(loss)):
        raise TrainingGateError("per-example suffix loss is nonfinite")
    return loss


def one_update_per_example(
    *, model: Any, optimizer: Any, batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one interface-control update with explicit example-balanced loss."""
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss_per_example(output.logits, batch["labels"])
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
            raise TrainingGateError(f"nonfinite interface-control gradient: {name}")
        grad_sq += float(parameter.grad.detach().float().square().sum())
        tensors_with_grad += 1
    gradient_norm = math.sqrt(grad_sq)
    if tensors_with_grad == 0 or not math.isfinite(gradient_norm) or gradient_norm == 0:
        raise TrainingGateError("interface-control update has zero or nonfinite gradient")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "trainable_tensors_with_grad": tensors_with_grad,
        "trainable_tensors_without_grad": missing_grad,
        "loss_reduction": "mean_of_per_example_mean_active_suffix_cross_entropy",
    }


def audit_equal_suffix_counts(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove the actual processor labeled every A/B/C candidate equally."""
    grouped: dict[str, set[int]] = defaultdict(set)
    by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate = str(row.get("candidate", ""))
        if candidate not in CODES:
            raise TrainingGateError(f"unexpected interface candidate: {candidate}")
        count = int(row.get("suffix_tokens", 0))
        if count <= 0:
            raise TrainingGateError("interface candidate has no labeled suffix token")
        grouped[candidate].add(count)
        by_item[str(row.get("item_id", ""))].append(row)
    if set(grouped) != set(CODES) or any(len(values) != 1 for values in grouped.values()):
        raise TrainingGateError("candidate suffix counts are incomplete or unstable")
    counts = {candidate: next(iter(grouped[candidate])) for candidate in CODES}
    if len(set(counts.values())) != 1:
        raise TrainingGateError(f"A/B/C suffix counts are unequal: {counts}")

    ranking_checks = 0
    for item_id, rows in by_item.items():
        if not item_id or {str(row["candidate"]) for row in rows} != set(CODES):
            raise TrainingGateError(f"interface candidate group is incomplete: {item_id}")
        if all("log_likelihood_sum" in row and "log_likelihood_mean" in row for row in rows):
            sum_order = sorted(
                rows,
                key=lambda row: (-float(row["log_likelihood_sum"]), str(row["candidate"])),
            )
            mean_order = sorted(
                rows,
                key=lambda row: (-float(row["log_likelihood_mean"]), str(row["candidate"])),
            )
            if [row["candidate"] for row in sum_order] != [row["candidate"] for row in mean_order]:
                raise TrainingGateError(f"sum/mean candidate ranking changed: {item_id}")
            for row in rows:
                if not math.isclose(
                    float(row["log_likelihood_sum"]),
                    float(row["log_likelihood_mean"]) * int(row["suffix_tokens"]),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ):
                    raise TrainingGateError(f"sum/mean score identity failed: {item_id}")
            ranking_checks += 1
    return {
        "status": "GREEN_EQUAL_SUFFIX_COUNTS",
        "candidate_suffix_tokens": counts,
        "equal_count": next(iter(counts.values())),
        "items_with_sum_mean_ranking_identity": ranking_checks,
    }


def semantic_summary(
    items: Sequence[Mapping[str, Any]], transformed_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    semantic_by_item = {
        str(row["item_id"]): str(row["semantic_target"]) for row in transformed_rows
    }
    if len(semantic_by_item) != len(transformed_rows):
        raise TrainingGateError("semantic summary received duplicate transformed items")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        item_id = str(item["item_id"])
        semantic = semantic_by_item.get(item_id)
        if semantic is None:
            raise TrainingGateError(f"scored item escaped transformed population: {item_id}")
        grouped[semantic].append(item)
    if set(grouped) != set(SEMANTIC_LABELS):
        raise TrainingGateError("semantic summary is missing a bean class")

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "items": len(rows),
            "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
            "mean_target_probability": sum(float(row["target_probability"]) for row in rows)
            / len(rows),
        }

    return {
        "aggregate": summarize(items),
        "by_semantic_target": {
            semantic: summarize(grouped[semantic]) for semantic in SEMANTIC_LABELS
        },
    }


validate_codebooks()
