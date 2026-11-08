"""Run the frozen PG1 train-only optimizer-scale microgate."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import assert_attention_eager, offline_gate, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import DiscoveryTaskBank
from .training_common import TrainingGateError, named_parameter_sha256
from .training_model import (
    _attention_configs,
    attach_free_lora_connector,
    one_update,
    real_training_batch,
    select_split_stratified,
    suffix_anchor,
)


def _verify_contract(root: Path, source_contract: Path) -> dict[str, Any]:
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_artifact_rehearsal":
        raise TrainingGateError("optimizer-scale source contract is not frozen")
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"optimizer-scale source hash mismatch: {name}")
    return contract


def _objective(model: Any, batch: dict[str, Any]) -> float:
    import torch

    from .training_common import causal_suffix_loss

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss(output.logits, batch["labels"])
    value = float(loss)
    if not math.isfinite(value):
        raise TrainingGateError("optimizer-scale probe objective is nonfinite")
    return value


def adjudicate_learning_rates(arms: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    passing = []
    rows = []
    for arm in arms:
        baseline = float(arm["objectives"][0])
        final_ratio = float(arm["objectives"][-1]) / baseline
        maximum_ratio = max(float(value) for value in arm["objectives"]) / baseline
        maximum_gradient = max(float(row["gradient_norm"]) for row in arm["updates"])
        finite = all(
            math.isfinite(value)
            for value in arm["objectives"]
            + [row["loss"] for row in arm["updates"]]
            + [row["gradient_norm"] for row in arm["updates"]]
        )
        passed = bool(
            finite
            and not arm["hard_stopped"]
            and float(thresholds["minimum_final_objective_ratio"]) <= final_ratio
            <= float(thresholds["maximum_final_objective_ratio"])
            and maximum_ratio <= float(thresholds["maximum_objective_ratio"])
            and maximum_gradient <= float(thresholds["maximum_gradient_norm"])
        )
        row = {
            "learning_rate": float(arm["learning_rate"]),
            "finite": finite,
            "hard_stopped": bool(arm["hard_stopped"]),
            "final_objective_ratio": final_ratio,
            "maximum_objective_ratio": maximum_ratio,
            "maximum_gradient_norm": maximum_gradient,
            "passed": passed,
        }
        rows.append(row)
        if passed:
            passing.append(row)
    selected = min(passing, key=lambda row: row["learning_rate"]) if passing else None
    return {
        "verdict": "GREEN_SELECTED_SCALE" if selected else "RED_NO_STABLE_SCALE",
        "selection_rule": "lowest_learning_rate_passing_bounded_target_window",
        "selected_learning_rate": selected["learning_rate"] if selected else None,
        "arms": rows,
    }


def run(
    *,
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    output: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite optimizer-scale result: {output}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("optimizer-scale gate requires an H100")
    contract = _verify_contract(root, source_contract)
    if config_path != (root / contract["artifacts"]["optimizer_scale_config"]["path"]).resolve():
        raise TrainingGateError("optimizer-scale config is outside the frozen source contract")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_h100" or config["lineage"]["id"] != "pg1_pi":
        raise TrainingGateError("optimizer-scale gate is not the frozen PG1 primary microgate")
    offline_gate(cache_root, expected_instance_id=instance_id)
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[("pg1_pi", "fused")]
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)

    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    probe_rows = select_split_stratified(
        rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["train_split"],
        per_target=int(config["probe"]["train_per_target"]),
    )
    if any(row["split"] != "train" for row in probe_rows):
        raise TrainingGateError("non-training item entered optimizer-scale microgate")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    probe_batch = real_training_batch(processor, probe_rows, task_root)
    seed = int(config["optimizer"]["seed"])
    thresholds = config["adjudication"]
    arm_records = []
    initial_trainable_hashes = []
    initial_frozen_hashes = []
    baseline_objectives = []
    zero_delta = []
    attention = []
    torch.cuda.reset_peak_memory_stats()

    for learning_rate in config["optimizer"]["learning_rates"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=config["runtime"]["attention"],
        ).to("cuda")
        model.config.use_cache = False
        attention.append(assert_attention_eager(_attention_configs(model)))
        baseline_logits = suffix_anchor(model, probe_batch)
        model, trainable_report = attach_free_lora_connector(model, config["adapter"])
        attached_logits = suffix_anchor(model, probe_batch)
        arm_zero_delta = all(torch.equal(left, right) for left, right in zip(baseline_logits, attached_logits))
        if not arm_zero_delta:
            raise TrainingGateError("optimizer-scale zero-delta attachment changed logits")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(learning_rate),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        initial_trainable = named_parameter_sha256(model, trainable=True)
        initial_frozen = named_parameter_sha256(model, trainable=False)
        objectives = [_objective(model, probe_batch)]
        updates = []
        hard_stopped = False
        for step in range(1, int(config["optimizer"]["updates_per_arm"]) + 1):
            update = one_update(model=model, optimizer=optimizer, batch=probe_batch)
            objective = _objective(model, probe_batch)
            updates.append({"step": step, **update})
            objectives.append(objective)
            if (
                not math.isfinite(objective)
                or objective / objectives[0] > float(thresholds["hard_stop_objective_ratio"])
                or update["gradient_norm"] > float(thresholds["hard_stop_gradient_norm"])
            ):
                hard_stopped = True
                break
        final_frozen = named_parameter_sha256(model, trainable=False)
        if final_frozen != initial_frozen:
            raise TrainingGateError("optimizer-scale arm changed frozen parameter bytes")
        arm_records.append(
            {
                "learning_rate": float(learning_rate),
                "probe_item_ids": [row["item_id"] for row in probe_rows],
                "objectives": objectives,
                "updates": updates,
                "hard_stopped": hard_stopped,
                "initial_trainable_sha256": initial_trainable,
                "final_trainable_sha256": named_parameter_sha256(model, trainable=True),
                "initial_frozen_sha256": initial_frozen,
                "final_frozen_sha256": final_frozen,
            }
        )
        initial_trainable_hashes.append(initial_trainable)
        initial_frozen_hashes.append(initial_frozen)
        baseline_objectives.append(objectives[0])
        zero_delta.append(arm_zero_delta)
        del model, optimizer, baseline_logits, attached_logits
        release_cuda()

    if len(set(initial_trainable_hashes)) != 1 or len(set(initial_frozen_hashes)) != 1:
        raise TrainingGateError("optimizer-scale arms did not start from identical parameter bytes")
    if len(set(baseline_objectives)) != 1:
        raise TrainingGateError("optimizer-scale arms did not start from an exact common objective")
    adjudication = adjudicate_learning_rates(arm_records, thresholds)
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_optimizer_scale_microgate",
        "status": "GREEN_MECHANICAL_EXECUTION",
        "label": "OPTIMIZER_SCALE_MICROGATE_NOT_PROMOTION_ELIGIBLE",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "lineage": "pg1_pi",
        "role": "primary",
        "checkpoint": {"repo": entity.repo, "revision": entity.revision},
        "probe_split": "train_only",
        "probe_item_ids": [row["item_id"] for row in probe_rows],
        "discovery_items_read": 0,
        "confirmation_items_read": 0,
        "identical_initial_trainable_bytes": len(set(initial_trainable_hashes)) == 1,
        "identical_initial_frozen_bytes": len(set(initial_frozen_hashes)) == 1,
        "identical_baseline_objective": len(set(baseline_objectives)) == 1,
        "zero_delta_logits_bit_exact": all(zero_delta),
        "attention": attention,
        "trainable_contract": trainable_report,
        "arms": arm_records,
        "adjudication": {**adjudication, "thresholds": thresholds, "promotion_eligible": False},
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output, packet)
    del processor, probe_batch
    release_cuda()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        root=args.root.resolve(),
        source_contract=args.source_contract.resolve(),
        matrix=args.matrix.resolve(),
        config_path=args.config.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        instance_id=args.instance_id,
        output=args.output.resolve(),
    )
    print(json.dumps({"status": packet["status"], **packet["adjudication"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
