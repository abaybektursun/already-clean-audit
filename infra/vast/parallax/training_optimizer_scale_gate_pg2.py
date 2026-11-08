"""Run the separately calibrated, train-only PG2 optimizer-scale microgate."""

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
from .training_optimizer_scale_gate import _objective, adjudicate_learning_rates


def _verify_contract(root: Path, source_contract: Path) -> dict[str, Any]:
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "FROZEN"
        or contract.get("stage") != "training_optimizer_scale_microgate"
        or contract.get("cache_scope") != "pg2_supporting_optimizer_scale_only"
    ):
        raise TrainingGateError("PG2 optimizer source contract is not frozen")
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"PG2 optimizer source hash mismatch: {name}")
    checkpoint = contract.get("checkpoint", {})
    manifest = checkpoint.get("input_manifest", {})
    if (
        checkpoint.get("repo") != "google/paligemma2-3b-pt-224"
        or checkpoint.get("revision") != "96eeb174da13ca1a2b247e4d0867436296c36420"
        or sha256_file(root / manifest["path"]) != manifest["sha256"]
    ):
        raise TrainingGateError("PG2 optimizer checkpoint manifest binding changed")
    return contract


def validate_pg2_config(config: dict[str, Any]) -> dict[str, Any]:
    lineage = config.get("lineage", {})
    if (
        config.get("status") != "frozen_before_h100"
        or lineage.get("id") != "pg2"
        or lineage.get("role") != "supporting"
        or lineage.get("checkpoint") != "google/paligemma2-3b-pt-224"
        or lineage.get("revision") != "96eeb174da13ca1a2b247e4d0867436296c36420"
        or lineage.get("promotion_eligible") is not False
        or config.get("panel", {}).get("task_id") != "vr_a_beans_balanced"
        or config.get("probe", {}).get("split") != "train_only"
        or config.get("probe", {}).get("discovery_reads") != 0
        or config.get("probe", {}).get("confirmation_reads") != 0
        or config.get("optimizer", {}).get("learning_rates")
        != [0.0003, 0.0001, 0.00003, 0.00001]
        or config.get("optimizer", {}).get("ladder_origin")
        != "same_frozen_candidate_ladder_not_transfer_of_PG1_selected_scale"
        or config.get("cross_lineage_firewall")
        != {
            "import_PG1_selected_learning_rate": "forbidden",
            "compare_raw_objectives_across_generations": "forbidden",
            "allowed_cross_lineage_output": "normalized_arm_verdicts_only",
            "scientific_claim": False,
        }
        or config.get("claim_boundary")
        != "PG2_OPTIMIZER_SCALE_MICROGATE_NOT_PROMOTION_ELIGIBLE"
    ):
        raise TrainingGateError("optimizer-scale gate is not the frozen PG2 supporting microgate")
    return config


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
        raise FileExistsError(f"refusing to overwrite PG2 optimizer result: {output}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("PG2 optimizer-scale gate requires an H100")
    contract = _verify_contract(root, source_contract)
    if config_path != (root / contract["artifacts"]["optimizer_scale_config"]["path"]).resolve():
        raise TrainingGateError("PG2 optimizer config is outside the frozen source contract")
    config = validate_pg2_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    lineage = config["lineage"]
    offline_gate(cache_root, expected_instance_id=instance_id)
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[("pg2", "fused")]
    if entity.repo != lineage["checkpoint"] or entity.revision != lineage["revision"]:
        raise TrainingGateError("PG2 fused entity differs from the frozen config")
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
        raise TrainingGateError("non-training item entered PG2 optimizer-scale microgate")
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
    trainable_reports = []
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
        trainable_reports.append(trainable_report)
        attached_logits = suffix_anchor(model, probe_batch)
        arm_zero_delta = all(torch.equal(left, right) for left, right in zip(baseline_logits, attached_logits))
        if not arm_zero_delta:
            raise TrainingGateError("PG2 optimizer zero-delta attachment changed logits")
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
            raise TrainingGateError("PG2 optimizer arm changed frozen parameter bytes")
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
        raise TrainingGateError("PG2 optimizer arms did not start from identical parameter bytes")
    if len(set(baseline_objectives)) != 1:
        raise TrainingGateError("PG2 optimizer arms did not start from an exact common objective")
    if any(report != trainable_reports[0] for report in trainable_reports[1:]):
        raise TrainingGateError("PG2 trainable attachment report differed between arms")
    adjudication = adjudicate_learning_rates(arm_records, thresholds)
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_optimizer_scale_microgate",
        "status": "GREEN_MECHANICAL_EXECUTION",
        "label": "PG2_OPTIMIZER_SCALE_MICROGATE_NOT_PROMOTION_ELIGIBLE",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "lineage": "pg2",
        "role": "supporting",
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
        "trainable_contract": trainable_reports[0],
        "arms": arm_records,
        "adjudication": {**adjudication, "thresholds": thresholds, "promotion_eligible": False},
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "cross_lineage_firewall": config["cross_lineage_firewall"],
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
