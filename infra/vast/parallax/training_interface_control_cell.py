"""Run the frozen three-codebook PARALLAX range interface control on one lineage."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import assert_attention_eager, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import DiscoveryTaskBank
from .training_common import (
    TrainingGateError,
    atomic_training_checkpoint,
    deterministic_batches,
    named_parameter_sha256,
    optimizer_state_sha256,
    schedule_sha256,
)
from .training_interface_control import (
    CODEBOOKS,
    CONTROL_TASK_ID,
    SEMANTIC_LABELS,
    audit_equal_suffix_counts,
    causal_suffix_loss_per_example,
    one_update_per_example,
    remap_rows,
    semantic_summary,
    validate_codebooks,
)
from .training_model import (
    _attention_configs,
    attach_free_lora_connector,
    evaluate_candidates,
    real_training_batch,
    select_split_stratified,
    suffix_anchor,
)


ALLOWED = {
    "pg2": {"role": "supporting_boundary_first", "entity_role": "fused"},
    "pg1_pi": {"role": "primary_only_if_pg2_green", "entity_role": "fused"},
}


def _verify_source_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "FROZEN"
        or contract.get("stage") != "training_interface_control"
        or contract.get("confirmation_items_read") != 0
    ):
        raise TrainingGateError("interface-control source contract is not frozen/no-confirmation")
    for label, row in contract.get("artifacts", {}).items():
        target = root / str(row.get("path", ""))
        if not target.is_file() or sha256_file(target) != row.get("sha256"):
            raise TrainingGateError(f"interface-control source hash mismatch: {label}")
    return contract


def _objective(model: Any, batch: dict[str, Any]) -> float:
    import torch

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss_per_example(output.logits, batch["labels"])
    value = float(loss)
    if not math.isfinite(value):
        raise TrainingGateError("interface-control objective is nonfinite")
    return value


def _mean_score_items(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["item_id"])].append(row)
    items = []
    for item_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: str(row["candidate"]))
        values = [float(row["log_likelihood_mean"]) for row in ordered]
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        probabilities = [weight / sum(weights) for weight in weights]
        target = str(ordered[0]["target"])
        target_index = next(index for index, row in enumerate(ordered) if row["candidate"] == target)
        predicted = max(range(len(values)), key=lambda index: (values[index], -index))
        items.append(
            {
                "item_id": item_id,
                "target": target,
                "prediction": str(ordered[predicted]["candidate"]),
                "correct": int(predicted == target_index),
                "target_probability": probabilities[target_index],
            }
        )
    return items


def _cell_adjudication(snapshots: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    first, final = snapshots[0], snapshots[-1]
    sum_gain = (
        final["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
        - first["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
    )
    mean_gain = (
        final["semantic_summary_mean"]["aggregate"]["mean_target_probability"]
        - first["semantic_summary_mean"]["aggregate"]["mean_target_probability"]
    )
    objective_drop = float(first["objective"] - final["objective"])
    semantic_gains = {
        semantic: (
            final["semantic_summary_sum"]["by_semantic_target"][semantic]["mean_target_probability"]
            - first["semantic_summary_sum"]["by_semantic_target"][semantic]["mean_target_probability"]
        )
        for semantic in SEMANTIC_LABELS
    }
    return {
        "objective_drop": objective_drop,
        "registered_control_probability_gain": sum_gain,
        "diagnostic_mean_score_probability_gain": mean_gain,
        "semantic_probability_gains": semantic_gains,
        "objective_clears_threshold": objective_drop > threshold,
        "positive_heldout_gain": sum_gain > 0.0,
        "all_semantic_gains_positive": all(value > 0.0 for value in semantic_gains.values()),
    }


def _aggregate_adjudication(
    cells: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["provisional_mechanical_adjudication"]
    objective_threshold = float(gate["minimum_objective_drop_per_passing_codebook"])
    gain_threshold = float(gate["minimum_mapping_marginalized_probability_gain"])
    initial = [
        cell["snapshots"][0]["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
        for cell in cells
    ]
    final = [
        cell["snapshots"][-1]["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
        for cell in cells
    ]
    pooled_gain = sum(final) / len(final) - sum(initial) / len(initial)
    semantic_gains = {
        semantic: sum(
            cell["snapshots"][-1]["semantic_summary_sum"]["by_semantic_target"][semantic][
                "mean_target_probability"
            ]
            - cell["snapshots"][0]["semantic_summary_sum"]["by_semantic_target"][semantic][
                "mean_target_probability"
            ]
            for cell in cells
        )
        / len(cells)
        for semantic in SEMANTIC_LABELS
    }
    positive_cells = sum(cell["adjudication"]["positive_heldout_gain"] for cell in cells)
    objective_cells = sum(
        cell["adjudication"]["objective_drop"] > objective_threshold for cell in cells
    )
    finite = all(
        math.isfinite(value)
        for value in [pooled_gain, *semantic_gains.values()]
        + [
            value
            for cell in cells
            for value in (
                cell["adjudication"]["objective_drop"],
                cell["adjudication"]["registered_control_probability_gain"],
            )
        ]
    )
    baseline = sum(initial) / len(initial)
    if not finite:
        verdict = "ABORT_NONFINITE"
    elif baseline <= float(gate["floor_probability"]):
        verdict = "VOID_READOUT_FLOOR"
    elif baseline >= float(gate["ceiling_probability"]):
        verdict = "VOID_READOUT_CEILING"
    elif objective_cells < int(gate["minimum_codebooks_with_positive_probability_gain"]):
        verdict = "RED_OBJECTIVE_NOT_MOVING"
    elif positive_cells < int(gate["minimum_codebooks_with_positive_probability_gain"]):
        verdict = "RED_MAPPING_FRAGILE"
    elif pooled_gain <= gain_threshold:
        verdict = "NO_HELDOUT_RANGE"
    elif not all(value > 0.0 for value in semantic_gains.values()):
        verdict = "RED_SEMANTIC_NONUNIFORM"
    else:
        verdict = "GREEN_PROVISIONAL_INTERFACE_CONTROL"
    return {
        "verdict": verdict,
        "mapping_marginalized_initial_probability": baseline,
        "mapping_marginalized_final_probability": sum(final) / len(final),
        "mapping_marginalized_probability_gain": pooled_gain,
        "mapping_marginalized_semantic_gains": semantic_gains,
        "positive_gain_codebooks": positive_cells,
        "objective_moving_codebooks": objective_cells,
        "thresholds": gate,
        "promotion_eligible": False,
    }


def run(
    *,
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    tokenizer_preflight: Path,
    cache_root: Path,
    task_root: Path,
    lineage: str,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    started = time.monotonic()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite interface-control result: {output_dir}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("interface-control cell requires an H100")
    contract = _verify_source_contract(root, source_contract)
    if config_path != (root / contract["artifacts"]["interface_config"]["path"]).resolve():
        raise TrainingGateError("interface-control config escaped its source contract")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed = ALLOWED.get(lineage)
    lineage_rows = {str(row["id"]): row for row in config.get("lineage_order", [])}
    if (
        config.get("status") != "frozen_design_before_tokenizer_preflight"
        or config.get("paid_authorization") is not False
        or allowed is None
        or lineage not in lineage_rows
        or lineage_rows[lineage]["role"] != allowed["role"]
        or config["population"]["confirmation_reads"] != 0
    ):
        raise TrainingGateError("interface-control lineage/design is not exact")
    preflight = json.loads(tokenizer_preflight.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or preflight.get("lineage") != lineage
        or preflight.get("model_loads") != 0
        or preflight.get("gpu_operations") != 0
        or preflight.get("confirmation_items_read") != 0
        or len(set(preflight.get("candidate_suffix_tokens", {}).values())) != 1
    ):
        raise TrainingGateError("processor-only interface preflight is absent or not green")
    validate_codebooks()
    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["population"]["source_task_id"]))
    train_source = select_split_stratified(
        rows,
        task_id=config["population"]["source_task_id"],
        split=config["population"]["train_split"],
        per_target=int(config["population"]["train_per_semantic_target"]),
    )
    discovery_source = select_split_stratified(
        rows,
        task_id=config["population"]["source_task_id"],
        split=config["population"]["discovery_split"],
        per_target=int(config["population"]["discovery_per_semantic_target"]),
    )
    schedule = deterministic_batches(
        item_count=len(train_source),
        batch_size=int(config["optimizer"]["batch_size"]),
        steps=int(config["optimizer"]["steps_per_codebook"]),
        seed=int(config["optimizer"]["seed"]),
    )
    schedule_hash = schedule_sha256(schedule)
    if schedule_hash != config["optimizer"]["exact_same_schedule_sha256_as_registered_seed17"]:
        raise TrainingGateError("interface-control schedule differs from registered seed17")
    snapshots_requested = [int(value) for value in config["optimizer"]["snapshots"]]
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[(lineage, allowed["entity_role"])]
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    output_dir.mkdir(parents=True)
    cells = []
    initial_frozen_hashes = []
    for codebook_id in CODEBOOKS:
        torch.manual_seed(int(config["optimizer"]["seed"]))
        torch.cuda.manual_seed_all(int(config["optimizer"]["seed"]))
        torch.use_deterministic_algorithms(True)
        train_rows = remap_rows(train_source, codebook_id=codebook_id)
        discovery_rows = remap_rows(discovery_source, codebook_id=codebook_id)
        release_cuda()
        torch.cuda.reset_peak_memory_stats()
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")
        model.config.use_cache = False
        attention = assert_attention_eager(_attention_configs(model))
        anchor_rows = select_split_stratified(
            train_rows,
            task_id=CONTROL_TASK_ID,
            split="train",
            per_target=1,
        )
        anchor_batch = real_training_batch(processor, anchor_rows, task_root)
        baseline_logits = suffix_anchor(model, anchor_batch)
        model, trainable_report = attach_free_lora_connector(model, config["adapter"])
        attached_logits = suffix_anchor(model, anchor_batch)
        if not all(torch.equal(left, right) for left, right in zip(baseline_logits, attached_logits)):
            raise TrainingGateError(f"zero-delta adapter attachment failed: {codebook_id}")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(lineage_rows[lineage]["optimizer_learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        frozen_initial = named_parameter_sha256(model, trainable=False)
        initial_frozen_hashes.append(frozen_initial)
        cell_dir = output_dir / codebook_id
        cell_dir.mkdir()
        snapshot_records = []
        update_records = []

        def bank_snapshot(step: int) -> None:
            checkpoint = cell_dir / f"step-{step:06d}.pt"
            checkpoint_sha = atomic_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                schedule_sha256=schedule_hash,
                metadata={
                    "lineage": lineage,
                    "codebook_id": codebook_id,
                    "interface_control": True,
                },
            )
            evaluation = evaluate_candidates(
                model=model,
                processor=processor,
                rows=discovery_rows,
                task_root=task_root,
                batch_size=4,
            )
            interface_audit = audit_equal_suffix_counts(evaluation["candidate_rows"])
            sum_semantic = semantic_summary(evaluation["items"], discovery_rows)
            mean_items = _mean_score_items(evaluation["candidate_rows"])
            mean_semantic = semantic_summary(mean_items, discovery_rows)
            frozen_hash = named_parameter_sha256(model, trainable=False)
            if frozen_hash != frozen_initial:
                raise TrainingGateError(f"frozen bytes changed: {codebook_id} step {step}")
            snapshot_records.append(
                {
                    "step": step,
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": checkpoint_sha,
                        "bytes": checkpoint.stat().st_size,
                    },
                    "objective": _objective(model, anchor_batch),
                    "evaluation": evaluation,
                    "interface_audit": interface_audit,
                    "semantic_summary_sum": sum_semantic,
                    "semantic_summary_mean": mean_semantic,
                    "trainable_sha256": named_parameter_sha256(model, trainable=True),
                    "optimizer_sha256": optimizer_state_sha256(optimizer),
                    "frozen_sha256": frozen_hash,
                }
            )

        bank_snapshot(0)
        for step, indices in enumerate(schedule, start=1):
            batch_rows = [train_rows[index] for index in indices]
            batch = real_training_batch(processor, batch_rows, task_root)
            update = one_update_per_example(model=model, optimizer=optimizer, batch=batch)
            update_records.append(
                {"step": step, "item_ids": [row["item_id"] for row in batch_rows], **update}
            )
            del batch
            if step in snapshots_requested:
                bank_snapshot(step)
        if [row["step"] for row in snapshot_records] != snapshots_requested:
            raise TrainingGateError(f"snapshot schedule changed: {codebook_id}")
        cell = {
            "codebook_id": codebook_id,
            "mapping": CODEBOOKS[codebook_id],
            "attention": attention,
            "trainable_contract": trainable_report,
            "train_item_ids": [row["item_id"] for row in train_rows],
            "discovery_item_ids": [row["item_id"] for row in discovery_rows],
            "updates": update_records,
            "snapshots": snapshot_records,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        cell["adjudication"] = _cell_adjudication(
            snapshot_records,
            float(
                config["provisional_mechanical_adjudication"][
                    "minimum_objective_drop_per_passing_codebook"
                ]
            ),
        )
        cells.append(cell)
        del model, optimizer, anchor_batch, baseline_logits, attached_logits
        release_cuda()
    if len(set(initial_frozen_hashes)) != 1:
        raise TrainingGateError("codebook cells did not reset to the same exact fused base")
    adjudication = _aggregate_adjudication(cells, config)
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_interface_control",
        "status": "GREEN_MECHANICAL_EXECUTION",
        "label": "INTERFACE_CONTROL_PROVISIONAL_NOT_PROMOTION_ELIGIBLE",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "lineage": lineage,
        "role": allowed["role"],
        "checkpoint": {"repo": entity.repo, "revision": entity.revision},
        "tokenizer_preflight": {
            "path": str(tokenizer_preflight),
            "sha256": sha256_file(tokenizer_preflight),
            "candidate_suffix_tokens": preflight["candidate_suffix_tokens"],
        },
        "schedule": {"sha256": schedule_hash, "batches": schedule},
        "codebooks": cells,
        "fused_base_reset_exact": True,
        "initial_frozen_sha256": initial_frozen_hashes[0],
        "adjudication": adjudication,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "checkpoint_bytes": sum(
                row["checkpoint"]["bytes"]
                for cell in cells
                for row in cell["snapshots"]
            ),
        },
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output_dir / "interface-control.json", packet)
    del processor
    release_cuda()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-preflight", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--lineage", choices=sorted(ALLOWED), required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        root=args.root.resolve(),
        source_contract=args.source_contract.resolve(),
        matrix=args.matrix.resolve(),
        config_path=args.config.resolve(),
        tokenizer_preflight=args.tokenizer_preflight.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        lineage=args.lineage,
        instance_id=args.instance_id,
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["adjudication"]["verdict"],
                "lineage": result["lineage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
