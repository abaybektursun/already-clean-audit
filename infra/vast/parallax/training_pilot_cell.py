"""Run one frozen fused-base mechanical range cell without promotion eligibility."""

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
from .training_common import (
    TrainingGateError,
    atomic_training_checkpoint,
    causal_suffix_loss,
    deterministic_batches,
    mechanical_adjudicate,
    named_parameter_sha256,
    optimizer_state_sha256,
    schedule_sha256,
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


ALLOWED_LINEAGES = {
    "pg1_pi": {
        "role": "primary",
        "checkpoint_role": "fused",
        "promotion_eligible": False,
    },
    "pg2": {
        "role": "supporting",
        "checkpoint_role": "fused",
        "promotion_eligible": False,
    },
}


def _verify_contract(root: Path, source_contract: Path) -> dict[str, Any]:
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_artifact_rehearsal":
        raise TrainingGateError("pilot-cell source contract is not frozen")
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"pilot-cell source hash mismatch: {name}")
    return contract


def _objective(model: Any, batch: dict[str, Any]) -> float:
    import torch

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss(output.logits, batch["labels"])
    value = float(loss)
    if not math.isfinite(value):
        raise TrainingGateError("pilot objective is nonfinite")
    return value


def run(
    *,
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    started = time.monotonic()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pilot cell: {output_dir}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("pilot cell requires an H100")
    contract = _verify_contract(root, source_contract)
    if config_path != (root / contract["artifacts"]["pilot_config"]["path"]).resolve():
        raise TrainingGateError("pilot config is outside the frozen source contract")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lineage = config.get("lineage", {})
    lineage_id = str(lineage.get("id", ""))
    expected_lineage = ALLOWED_LINEAGES.get(lineage_id)
    if (
        config.get("status") != "frozen_before_h100"
        or expected_lineage is None
        or lineage != {"id": lineage_id, **expected_lineage}
    ):
        raise TrainingGateError("range cell is not an allowed frozen fused-base rehearsal")
    planted_path = root / config["prerequisites"]["planted_gate"]
    planted = json.loads(planted_path.read_text(encoding="utf-8"))
    if planted.get("status") != config["prerequisites"]["required_planted_status"]:
        raise TrainingGateError("pilot prerequisite plants are not green")
    if planted.get("determinism", {}).get("exact_resume") is not True:
        raise TrainingGateError("pilot prerequisite exact-resume plant failed")
    offline_gate(cache_root, expected_instance_id=instance_id)
    seed = int(config["optimizer"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))

    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    train_rows = select_split_stratified(
        rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["train_split"],
        per_target=int(config["panel"]["train_per_target"]),
    )
    discovery_rows = select_split_stratified(
        rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["discovery_split"],
        per_target=int(config["panel"]["discovery_per_target"]),
    )
    if any(row["split"] == "confirmation_eval" for row in train_rows + discovery_rows):
        raise TrainingGateError("confirmation item entered pilot cell")
    schedule = deterministic_batches(
        item_count=len(train_rows),
        batch_size=int(config["optimizer"]["batch_size"]),
        steps=int(config["optimizer"]["steps"]),
        seed=seed,
    )
    schedule_hash = schedule_sha256(schedule)
    snapshots = [int(step) for step in config["optimizer"]["snapshots"]]
    if snapshots != sorted(set(snapshots)) or snapshots[0] != 0 or snapshots[-1] != len(schedule):
        raise TrainingGateError("pilot snapshots do not span the frozen schedule")

    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[(lineage_id, "fused")]
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    release_cuda()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config["runtime"]["attention"],
    ).to("cuda")
    model.config.use_cache = False
    attention = assert_attention_eager(_attention_configs(model))
    anchor_rows = select_split_stratified(
        train_rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["train_split"],
        per_target=1,
    )
    anchor_batch = real_training_batch(processor, anchor_rows, task_root)
    baseline_logits = suffix_anchor(model, anchor_batch)
    model, trainable_report = attach_free_lora_connector(model, config["adapter"])
    attached_logits = suffix_anchor(model, anchor_batch)
    zero_delta = all(torch.equal(left, right) for left, right in zip(baseline_logits, attached_logits))
    if not zero_delta:
        raise TrainingGateError("pilot zero-delta attachment changed logits")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    frozen_initial = named_parameter_sha256(model, trainable=False)
    output_dir.mkdir(parents=True)
    snapshot_records: list[dict[str, Any]] = []
    update_records: list[dict[str, Any]] = []

    def bank_snapshot(step: int) -> None:
        checkpoint = output_dir / f"step-{step:06d}.pt"
        checkpoint_sha = atomic_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=step,
            schedule_sha256=schedule_hash,
            metadata={"lineage": lineage_id, "pilot_mechanical": True},
        )
        objective = _objective(model, anchor_batch)
        evaluation = evaluate_candidates(
            model=model,
            processor=processor,
            rows=discovery_rows,
            task_root=task_root,
            batch_size=4,
        )
        frozen_hash = named_parameter_sha256(model, trainable=False)
        if frozen_hash != frozen_initial:
            raise TrainingGateError(f"pilot snapshot {step} changed frozen parameter bytes")
        snapshot_records.append(
            {
                "step": step,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "bytes": checkpoint.stat().st_size,
                },
                "objective": objective,
                "evaluation": evaluation,
                "trainable_sha256": named_parameter_sha256(model, trainable=True),
                "optimizer_sha256": optimizer_state_sha256(optimizer),
                "frozen_sha256": frozen_hash,
            }
        )

    bank_snapshot(0)
    for step, indices in enumerate(schedule, start=1):
        batch_rows = [train_rows[index] for index in indices]
        batch = real_training_batch(processor, batch_rows, task_root)
        update = one_update(model=model, optimizer=optimizer, batch=batch)
        update_records.append({"step": step, "item_ids": [row["item_id"] for row in batch_rows], **update})
        del batch
        if step in snapshots:
            bank_snapshot(step)

    first = snapshot_records[0]
    final = snapshot_records[-1]
    task_id = config["panel"]["task_id"]
    baseline_summary = first["evaluation"]["summary"][task_id]
    final_summary = final["evaluation"]["summary"][task_id]
    objective_drop = float(first["objective"] - final["objective"])
    heldout_gain = float(
        final_summary["mean_target_probability"] - baseline_summary["mean_target_probability"]
    )
    adjudication_config = config["mechanical_adjudication"]
    shuffle_gain = float(planted["label_shuffle_plant"]["shuffle_heldout_gain"])
    finite = all(
        math.isfinite(value)
        for value in [objective_drop, heldout_gain, shuffle_gain]
        + [record["loss"] for record in update_records]
        + [record["gradient_norm"] for record in update_records]
    )
    verdict = mechanical_adjudicate(
        finite=finite,
        objective_drop=objective_drop,
        heldout_gain=heldout_gain,
        shuffled_gain=shuffle_gain,
        at_floor=float(baseline_summary["mean_target_probability"])
        <= float(adjudication_config["floor_probability"]),
        at_ceiling=float(baseline_summary["mean_target_probability"])
        >= float(adjudication_config["ceiling_probability"]),
        minimum_objective_drop=float(adjudication_config["minimum_objective_drop"]),
        minimum_heldout_gain=float(adjudication_config["minimum_heldout_probability_gain"]),
        maximum_shuffle_gain=float(adjudication_config["maximum_shuffle_gain"]),
    )
    wall_seconds = time.monotonic() - started
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_artifact_rehearsal",
        "status": "GREEN_MECHANICAL_EXECUTION",
        "label": "PILOT_MECHANICAL_NOT_PROMOTION_ELIGIBLE",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "lineage": lineage_id,
        "role": expected_lineage["role"],
        "checkpoint": {"repo": entity.repo, "revision": entity.revision},
        "attention": attention,
        "zero_delta_logits_bit_exact": zero_delta,
        "trainable_contract": trainable_report,
        "schedule": {"sha256": schedule_hash, "batches": schedule},
        "train_item_ids": [row["item_id"] for row in train_rows],
        "discovery_item_ids": [row["item_id"] for row in discovery_rows],
        "confirmation_items_read": 0,
        "updates": update_records,
        "snapshots": snapshot_records,
        "frozen_parameter_bytes_exact": all(
            record["frozen_sha256"] == frozen_initial for record in snapshot_records
        ),
        "mechanical_adjudication": {
            "verdict": verdict,
            "objective_drop": objective_drop,
            "heldout_probability_gain": heldout_gain,
            "shuffle_gain": shuffle_gain,
            "shuffle_control_source": adjudication_config["shuffle_control_source"],
            "thresholds": adjudication_config,
            "promotion_eligible": False,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "checkpoint_bytes": sum(record["checkpoint"]["bytes"] for record in snapshot_records),
        },
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output_dir / "pilot.json", packet)
    del model, optimizer, processor, anchor_batch, baseline_logits, attached_logits
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        root=args.root.resolve(),
        source_contract=args.source_contract.resolve(),
        matrix=args.matrix.resolve(),
        config_path=args.config.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        instance_id=args.instance_id,
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": packet["status"],
                "verdict": packet["mechanical_adjudication"]["verdict"],
                "wall_seconds": packet["resources"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
