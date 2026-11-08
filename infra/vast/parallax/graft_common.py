"""Pure contracts for the PARALLAX shared-SigLIP graft discovery stage."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class GraftGateError(RuntimeError):
    """Raised when a graft cell would cross a frozen comparison boundary."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select_discovery_panel(
    rows: Iterable[dict[str, Any]], strata: Mapping[str, int]
) -> list[dict[str, Any]]:
    """Select the lexicographically first item IDs within frozen task/target strata."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row["task_id"])
        if task in strata and row.get("image") is not None:
            grouped[(task, str(row["target"]))].append(row)
    selected: list[dict[str, Any]] = []
    for task, per_target in sorted(strata.items()):
        task_groups = sorted((key, values) for key, values in grouped.items() if key[0] == task)
        if not task_groups:
            raise GraftGateError(f"no discovery image rows for {task}")
        for key, values in task_groups:
            ordered = sorted(values, key=lambda row: str(row["item_id"]))
            if len(ordered) < int(per_target):
                raise GraftGateError(f"insufficient rows for stratum {key}: {len(ordered)}")
            selected.extend(ordered[: int(per_target)])
    selected.sort(key=lambda row: (str(row["task_id"]), str(row["target"]), str(row["item_id"])))
    if len({row["item_id"] for row in selected}) != len(selected):
        raise GraftGateError("discovery panel contains duplicate item IDs")
    return selected


def wrong_image_map(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Derange images within task without inspecting model outputs."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    result: dict[str, str] = {}
    for task, task_rows in sorted(by_task.items()):
        ordered = sorted(task_rows, key=lambda row: str(row["item_id"]))
        if len(ordered) < 2:
            raise GraftGateError(f"wrong-image control needs at least two items for {task}")
        for index, row in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            if donor["image_sha256"] == row["image_sha256"]:
                raise GraftGateError(f"wrong-image derangement reused pixels for {row['item_id']}")
            result[str(row["item_id"])] = str(donor["image"])
    return result


def candidate_rows(panel: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in panel:
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise GraftGateError(f"candidate assay requires >=2 candidates: {row['item_id']}")
        if str(row["target"]) not in {str(value) for value in candidates}:
            raise GraftGateError(f"target absent from candidates: {row['item_id']}")
        for candidate in candidates:
            result.append(
                {
                    "item_id": str(row["item_id"]),
                    "task_id": str(row["task_id"]),
                    "prompt": str(row["prompt"]),
                    "candidate": str(candidate),
                    "target": str(row["target"]),
                    "image": str(row["image"]),
                }
            )
    return result


def suffix_log_likelihoods(logits: Any, labels: Any) -> tuple[list[float], list[float], list[int]]:
    """Return summed and mean suffix log likelihood from processor-provided labels."""
    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise GraftGateError(f"logit/label shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    gathered = torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered.masked_fill(~mask, 0.0)
    counts = mask.sum(dim=-1)
    if bool((counts <= 0).any()):
        raise GraftGateError("processor produced a candidate with no suffix labels")
    sums = gathered.sum(dim=-1)
    means = sums / counts
    return sums.cpu().tolist(), means.cpu().tolist(), counts.cpu().tolist()


def item_scores(candidate_scores: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_scores:
        grouped[str(row["item_id"])].append(row)
    items: list[dict[str, Any]] = []
    for item_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: str(row["candidate"]))
        targets = {str(row["target"]) for row in ordered}
        tasks = {str(row["task_id"]) for row in ordered}
        if len(targets) != 1 or len(tasks) != 1:
            raise GraftGateError(f"inconsistent candidate group: {item_id}")
        values = [float(row["log_likelihood_sum"]) for row in ordered]
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        normalizer = sum(weights)
        probabilities = [weight / normalizer for weight in weights]
        target = next(iter(targets))
        target_index = next(i for i, row in enumerate(ordered) if str(row["candidate"]) == target)
        other_values = [value for i, value in enumerate(values) if i != target_index]
        predicted_index = max(range(len(values)), key=lambda index: (values[index], -index))
        items.append(
            {
                "item_id": item_id,
                "task_id": next(iter(tasks)),
                "target": target,
                "prediction": str(ordered[predicted_index]["candidate"]),
                "correct": int(predicted_index == target_index),
                "target_probability": probabilities[target_index],
                "target_margin_nats": values[target_index] - max(other_values),
                "candidate_count": len(ordered),
            }
        )
    return items


def summarize_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        grouped[str(row["task_id"])].append(row)
    result: dict[str, Any] = {}
    for task, rows in sorted(grouped.items()):
        result[task] = {
            "items": len(rows),
            "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
            "mean_target_probability": sum(float(row["target_probability"]) for row in rows) / len(rows),
            "mean_target_margin_nats": sum(float(row["target_margin_nats"]) for row in rows) / len(rows),
        }
    return result


def paired_bootstrap_lower(
    left: Sequence[float], right: Sequence[float], *, seed: int, draws: int = 10_000
) -> float:
    if len(left) != len(right) or not left:
        raise GraftGateError("paired bootstrap requires equal nonempty vectors")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    means.sort()
    return means[max(0, int(0.05 * draws) - 1)]


def discovery_range(
    native: Sequence[dict[str, Any]],
    shuffled: Sequence[dict[str, Any]],
    wrong: Sequence[dict[str, Any]],
    *,
    minimum_raw_effect: float,
    seed: int,
) -> dict[str, Any]:
    indexed = {
        name: {str(row["item_id"]): row for row in rows}
        for name, rows in (("native", native), ("shuffled", shuffled), ("wrong", wrong))
    }
    if not (set(indexed["native"]) == set(indexed["shuffled"]) == set(indexed["wrong"])):
        raise GraftGateError("range controls do not contain the same item IDs")
    tasks = sorted({str(row["task_id"]) for row in native})
    task_results: dict[str, Any] = {}
    for offset, task in enumerate(tasks):
        ids = sorted(item for item, row in indexed["native"].items() if row["task_id"] == task)
        vectors = {
            name: [float(indexed[name][item]["target_probability"]) for item in ids]
            for name in indexed
        }
        effects = {
            null: sum(a - b for a, b in zip(vectors["native"], vectors[null])) / len(ids)
            for null in ("shuffled", "wrong")
        }
        lowers = {
            null: paired_bootstrap_lower(
                vectors["native"], vectors[null], seed=seed + 101 * offset + (0 if null == "shuffled" else 1)
            )
            for null in ("shuffled", "wrong")
        }
        green = all(effects[name] >= minimum_raw_effect and lowers[name] > 0 for name in effects)
        task_results[task] = {
            "status": "GREEN_RANGE" if green else "NO_RANGE",
            "native_minus_null": effects,
            "bootstrap_95pct_lower": lowers,
            "minimum_raw_effect": minimum_raw_effect,
            "items": len(ids),
        }
    green_tasks = [task for task, result in task_results.items() if result["status"] == "GREEN_RANGE"]
    return {
        "status": "GREEN_RANGE" if green_tasks else "VOID_NO_NATIVE_RANGE",
        "green_tasks": green_tasks,
        "tasks": task_results,
    }


def retained_fraction(
    donor: Sequence[dict[str, Any]],
    native: Sequence[dict[str, Any]],
    shuffled: Sequence[dict[str, Any]],
    wrong: Sequence[dict[str, Any]],
    range_tasks: Sequence[str],
) -> dict[str, Any]:
    maps = {
        name: {str(row["item_id"]): row for row in rows}
        for name, rows in (("donor", donor), ("native", native), ("shuffled", shuffled), ("wrong", wrong))
    }
    result: dict[str, Any] = {}
    for task in range_tasks:
        ids = sorted(item for item, row in maps["native"].items() if row["task_id"] == task)
        means = {
            name: sum(float(maps[name][item]["target_probability"]) for item in ids) / len(ids)
            for name in maps
        }
        floor = (means["shuffled"] + means["wrong"]) / 2
        denominator = means["native"] - floor
        result[task] = {
            "donor_score": means["donor"],
            "native_score": means["native"],
            "null_floor": floor,
            "retained_fraction": (means["donor"] - floor) / denominator if denominator > 0 else None,
        }
    return result


def assert_vision_state_compatible(host: Mapping[str, Any], donor: Mapping[str, Any]) -> dict[str, Any]:
    host_keys, donor_keys = set(host), set(donor)
    missing = sorted(host_keys - donor_keys)
    extra = sorted(donor_keys - host_keys)
    shape_mismatch = sorted(key for key in host_keys & donor_keys if tuple(host[key].shape) != tuple(donor[key].shape))
    if missing or extra or shape_mismatch:
        raise GraftGateError(
            f"vision state incompatible: missing={len(missing)} extra={len(extra)} shape={len(shape_mismatch)}"
        )
    return {
        "keys": len(host_keys),
        "elements": sum(int(value.numel()) for value in host.values()),
        "missing": 0,
        "extra": 0,
        "shape_mismatch": 0,
    }


@contextmanager
def shuffled_patch_features(projector: Any):
    """Reverse patch order immediately before the fixed multimodal projector."""
    import torch

    counts = {"calls": 0, "tokens": None}

    def hook(_module: Any, args: tuple[Any, ...]):
        if len(args) != 1 or getattr(args[0], "ndim", None) != 3:
            raise GraftGateError("unexpected projector input for shuffled-feature control")
        hidden = args[0]
        counts["calls"] += 1
        counts["tokens"] = int(hidden.shape[1])
        order = torch.arange(hidden.shape[1] - 1, -1, -1, device=hidden.device)
        return (hidden.index_select(1, order),)

    handle = projector.register_forward_pre_hook(hook)
    try:
        yield counts
    finally:
        handle.remove()
