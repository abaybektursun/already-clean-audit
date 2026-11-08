"""Run the frozen two-lineage zero/one-update gate on one Vast H100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import assert_attention_eager, offline_gate, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import DiscoveryTaskBank
from .training_common import (
    TrainingGateError,
    atomic_training_checkpoint,
    named_parameter_sha256,
    optimizer_state_sha256,
    restore_training_checkpoint,
)
from .training_model import (
    _attention_configs,
    attach_free_lora_connector,
    evaluate_candidates,
    one_update,
    real_training_batch,
    select_split_stratified,
    suffix_anchor,
)


def _validate(root: Path, config: dict[str, Any], task_root: Path) -> dict[str, Any]:
    prerequisite = config["prerequisites"]
    planted_path = root / prerequisite["planted_gate"]
    execution_path = root / prerequisite["execution_freeze"]
    discovery_seal_path = root / prerequisite["discovery_seal"]
    planted = json.loads(planted_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    discovery_seal = json.loads(discovery_seal_path.read_text(encoding="utf-8"))
    if planted.get("status") != prerequisite["required_planted_status"]:
        raise TrainingGateError("weight-free planted gate is not green")
    if execution.get("status") != "FROZEN" or execution.get("h100_launch_authorized") is not False:
        raise TrainingGateError("execution v3 freeze is missing or unexpectedly authorizes H100 science")
    if discovery_seal.get("status") != prerequisite["required_discovery_seal_status"]:
        raise TrainingGateError("discovery-only task seal is not green")
    if task_root.resolve() != (root / "studies/lineage/cross-lineage/data/parallax_tasks_v1").resolve():
        raise TrainingGateError("real gate task root is not the frozen task bank")
    return {
        "planted_gate": {"path": str(planted_path), "sha256": sha256_file(planted_path)},
        "execution_freeze": {"path": str(execution_path), "sha256": sha256_file(execution_path)},
        "discovery_seal": {
            "path": str(discovery_seal_path),
            "sha256": sha256_file(discovery_seal_path),
        },
    }


def run(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from peft import __version__ as peft_version
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite real training gate: {output_dir}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("real training gate requires an H100")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_h100":
        raise TrainingGateError("real training gate config is not frozen")
    bindings = _validate(root, config, task_root)
    cache_gate = offline_gate(cache_root, expected_instance_id=instance_id)
    torch.manual_seed(int(config["optimizer"]["seed"]))
    torch.cuda.manual_seed_all(int(config["optimizer"]["seed"]))
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))

    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    discovery_rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    train_rows = select_split_stratified(
        discovery_rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["train_split"],
        per_target=int(config["panel"]["train_per_target"]),
    )
    evaluation_rows = select_split_stratified(
        discovery_rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["discovery_split"],
        per_target=int(config["panel"]["discovery_per_target"]),
    )
    if any(row["split"] == "confirmation_eval" for row in train_rows + evaluation_rows):
        raise TrainingGateError("confirmation item entered the real training gate")

    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    output_dir.mkdir(parents=True)
    results = {}
    for lineage in config["lineages"]["order"]:
        entity = entities[(lineage, "fused")]
        snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=config["runtime"]["attention"],
        ).to("cuda")
        model.config.use_cache = False
        attention = assert_attention_eager(_attention_configs(model))
        batch = real_training_batch(processor, train_rows, task_root)
        baseline = suffix_anchor(model, batch)
        model, trainable_report = attach_free_lora_connector(model, config["adapter"])
        attached = suffix_anchor(model, batch)
        identity_exact = all(torch.equal(left, right) for left, right in zip(baseline, attached))
        if not identity_exact:
            raise TrainingGateError(f"zero-delta LoRA/connector changed logits for {lineage}")

        frozen_before = named_parameter_sha256(model, trainable=False)
        trainable_before = named_parameter_sha256(model, trainable=True)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        update = one_update(model=model, optimizer=optimizer, batch=batch)
        trainable_after = named_parameter_sha256(model, trainable=True)
        if trainable_before == trainable_after:
            raise TrainingGateError(f"one update did not change trainable bytes for {lineage}")
        frozen_after_update = named_parameter_sha256(model, trainable=False)
        if frozen_before != frozen_after_update:
            raise TrainingGateError(f"frozen parameter bytes changed for {lineage}")

        lineage_dir = output_dir / lineage
        lineage_dir.mkdir()
        checkpoint_path = lineage_dir / "step-000001.pt"
        checkpoint_hash = atomic_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=1,
            schedule_sha256=sha256_file(config_path),
            metadata={"lineage": lineage, "mechanical_only": True},
        )
        optimizer_after = optimizer_state_sha256(optimizer)
        with torch.no_grad():
            first_trainable = next(parameter for parameter in model.parameters() if parameter.requires_grad)
            first_trainable.add_(1.0)
        corrupted_hash = named_parameter_sha256(model, trainable=True)
        if corrupted_hash == trainable_after:
            raise TrainingGateError("checkpoint corruption plant did not change trainable bytes")
        restored = restore_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            expected_schedule_sha256=sha256_file(config_path),
        )
        restored_trainable = named_parameter_sha256(model, trainable=True)
        restored_optimizer = optimizer_state_sha256(optimizer)
        if restored_trainable != trainable_after or restored_optimizer != optimizer_after:
            raise TrainingGateError(f"checkpoint restore was not exact for {lineage}")
        evaluation = evaluate_candidates(
            model=model,
            processor=processor,
            rows=evaluation_rows,
            task_root=task_root,
            batch_size=4,
        )
        frozen_final = named_parameter_sha256(model, trainable=False)
        if frozen_final != frozen_before:
            raise TrainingGateError(f"evaluation changed frozen parameters for {lineage}")
        results[lineage] = {
            "status": "GREEN",
            "role": "primary" if lineage == config["lineages"]["primary"] else "supporting",
            "checkpoint": {"repo": entity.repo, "revision": entity.revision},
            "attention": attention,
            "zero_delta_logits_bit_exact": identity_exact,
            "trainable_contract": trainable_report,
            "update": update,
            "parameter_bytes": {
                "frozen_before_sha256": frozen_before,
                "frozen_after_update_sha256": frozen_after_update,
                "frozen_final_sha256": frozen_final,
                "frozen_exact": frozen_before == frozen_after_update == frozen_final,
                "trainable_before_sha256": trainable_before,
                "trainable_after_sha256": trainable_after,
                "resolved_update": trainable_before != trainable_after,
            },
            "checkpoint_restore": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "restored_sha256": restored["checkpoint_sha256"],
                "trainable_exact": restored_trainable == trainable_after,
                "optimizer_exact": restored_optimizer == optimizer_after,
            },
            "evaluation_after_restore": evaluation,
        }
        del model, processor, optimizer, batch, baseline, attached
        release_cuda()

    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_harness_real_forward_gate",
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "confirmation_items_read": 0,
        "lineage_firewall": {
            "single_chain_across_both": False,
            "direct_cross_gemma_numeric_comparison": False,
            "per_lineage_mechanical_verdicts_only": True,
        },
        "panel": {
            "task_id": config["panel"]["task_id"],
            "train_item_ids": [row["item_id"] for row in train_rows],
            "discovery_item_ids": [row["item_id"] for row in evaluation_rows],
        },
        "bindings": {
            **bindings,
            "config_sha256": sha256_file(config_path),
            "matrix_sha256": sha256_file(matrix),
            "cache_manifest_sha256": cache_gate.get("content_manifest_sha256"),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft_version,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16_with_fp32_trainables",
            "attention": "eager",
            "use_cache": False,
        },
        "lineages": results,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output_dir / "gate.json", packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        root=args.root.resolve(),
        matrix=args.matrix.resolve(),
        config_path=args.config.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        instance_id=args.instance_id,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": packet["status"], "lineages": list(packet["lineages"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
