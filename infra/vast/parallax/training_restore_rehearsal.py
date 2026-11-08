"""Rehearse a real PG1 checkpoint volume restore and exact next-step resume."""

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
    one_update,
    real_training_batch,
    select_split_stratified,
)


def _verify_contract(root: Path, source_contract: Path) -> dict[str, Any]:
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_restore_rehearsal":
        raise TrainingGateError("restore-rehearsal source contract is not frozen")
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"restore-rehearsal source hash mismatch: {name}")
    return contract


def _run_branch(
    *,
    snapshot: Path,
    task_root: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint: Path,
    schedule_sha256: str,
) -> tuple[dict[str, Any], Any, Any]:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    seed = int(config["optimizer"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config["runtime"]["attention"],
    ).to("cuda")
    model.config.use_cache = False
    attention = assert_attention_eager(_attention_configs(model))
    batch = real_training_batch(processor, rows, task_root)
    model, trainable_report = attach_free_lora_connector(model, config["adapter"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    restored = restore_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_schedule_sha256=schedule_sha256,
    )
    if restored["step"] != 1 or restored["metadata"].get("lineage") != "pg1_pi":
        raise TrainingGateError("restored checkpoint is not the frozen PG1 step-1 checkpoint")
    initial_trainable = named_parameter_sha256(model, trainable=True)
    initial_optimizer = optimizer_state_sha256(optimizer)
    frozen_before = named_parameter_sha256(model, trainable=False)
    update = one_update(model=model, optimizer=optimizer, batch=batch)
    final_trainable = named_parameter_sha256(model, trainable=True)
    final_optimizer = optimizer_state_sha256(optimizer)
    frozen_after = named_parameter_sha256(model, trainable=False)
    if frozen_before != frozen_after:
        raise TrainingGateError("resume branch changed frozen parameter bytes")
    return (
        {
            "attention": attention,
            "trainable_contract": trainable_report,
            "restored": restored,
            "initial_trainable_sha256": initial_trainable,
            "initial_optimizer_sha256": initial_optimizer,
            "update": update,
            "final_trainable_sha256": final_trainable,
            "final_optimizer_sha256": final_optimizer,
            "frozen_before_sha256": frozen_before,
            "frozen_after_sha256": frozen_after,
            "frozen_exact": frozen_before == frozen_after,
        },
        model,
        optimizer,
    )


def run(
    *,
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    restored_checkpoint: Path,
    expected_checkpoint_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite restore rehearsal: {output_dir}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("restore rehearsal requires an H100")
    if sha256_file(restored_checkpoint) != expected_checkpoint_sha256:
        raise TrainingGateError("volume-restored checkpoint hash mismatch")
    contract = _verify_contract(root, source_contract)
    if matrix != (root / contract["artifacts"]["matrix"]["path"]).resolve():
        raise TrainingGateError("restore rehearsal matrix is outside the frozen contract")
    if config_path != (root / contract["artifacts"]["config"]["path"]).resolve():
        raise TrainingGateError("restore rehearsal training config is outside the frozen contract")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_h100" or config["lineages"]["primary"] != "pg1_pi":
        raise TrainingGateError("restore rehearsal requires the frozen PG1-primary gate config")
    offline_gate(cache_root, expected_instance_id=instance_id)
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[("pg1_pi", "fused")]
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    discovery_rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    train_rows = select_split_stratified(
        discovery_rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["train_split"],
        per_target=int(config["panel"]["train_per_target"]),
    )
    schedule_hash = sha256_file(config_path)

    reference, reference_model, reference_optimizer = _run_branch(
        snapshot=snapshot,
        task_root=task_root,
        rows=train_rows,
        config=config,
        checkpoint=restored_checkpoint,
        schedule_sha256=schedule_hash,
    )
    del reference_model, reference_optimizer
    release_cuda()

    resumed, resumed_model, resumed_optimizer = _run_branch(
        snapshot=snapshot,
        task_root=task_root,
        rows=train_rows,
        config=config,
        checkpoint=restored_checkpoint,
        schedule_sha256=schedule_hash,
    )
    exact_fields = (
        "initial_trainable_sha256",
        "initial_optimizer_sha256",
        "final_trainable_sha256",
        "final_optimizer_sha256",
        "frozen_before_sha256",
        "frozen_after_sha256",
    )
    mismatches = [field for field in exact_fields if reference[field] != resumed[field]]
    if mismatches or reference["update"] != resumed["update"]:
        raise TrainingGateError(f"volume resume diverged from fresh reference: {mismatches}")
    output_dir.mkdir(parents=True)
    resumed_path = output_dir / "pg1-step-000002.pt"
    resumed_sha256 = atomic_training_checkpoint(
        resumed_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        step=2,
        schedule_sha256=schedule_hash,
        metadata={"lineage": "pg1_pi", "mechanical_only": True, "volume_resume": True},
    )
    del resumed_model, resumed_optimizer
    release_cuda()
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_upload_delete_restore_resume_rehearsal",
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "lineage": "pg1_pi",
        "role": "primary",
        "checkpoint_source": {
            "path": str(restored_checkpoint),
            "sha256": expected_checkpoint_sha256,
            "bytes": restored_checkpoint.stat().st_size,
            "external_volume_roundtrip_required": True,
        },
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "train_item_ids": [row["item_id"] for row in train_rows],
        "confirmation_items_read": 0,
        "reference": reference,
        "resumed": resumed,
        "exact_resume": {
            "fields": list(exact_fields),
            "parameter_and_optimizer_hashes_exact": True,
            "update_metrics_exact": True,
        },
        "step2_checkpoint": {
            "path": str(resumed_path),
            "sha256": resumed_sha256,
            "bytes": resumed_path.stat().st_size,
        },
        "claim_boundary": "mechanical_storage_and_resume_only_no_scientific_claim",
    }
    atomic_json(output_dir / "rehearsal.json", packet)
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
    parser.add_argument("--restored-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
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
        restored_checkpoint=args.restored_checkpoint.resolve(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": packet["status"], "exact_resume": packet["exact_resume"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
