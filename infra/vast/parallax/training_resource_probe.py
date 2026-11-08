"""Measure peak CUDA bytes for one frozen PG1 mechanical update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import offline_gate, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import DiscoveryTaskBank
from .training_common import TrainingGateError
from .training_model import select_split_stratified
from .training_restore_rehearsal import _run_branch


def _verify_contract(root: Path, source_contract: Path) -> dict[str, Any]:
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_resource_probe":
        raise TrainingGateError("resource-probe source contract is not frozen")
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"resource-probe source hash mismatch: {name}")
    return contract


def run(
    *,
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    probe_config_path: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    restored_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to overwrite resource probe: {output}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("resource probe requires an H100")
    contract = _verify_contract(root, source_contract)
    expected_checkpoint_sha256 = contract["checkpoint"]["expected_sha256"]
    if sha256_file(restored_checkpoint) != expected_checkpoint_sha256:
        raise TrainingGateError("resource-probe checkpoint hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    probe_config = yaml.safe_load(probe_config_path.read_text(encoding="utf-8"))
    offline_gate(cache_root, expected_instance_id=instance_id)
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[("pg1_pi", "fused")]
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    train_rows = select_split_stratified(
        rows,
        task_id=config["panel"]["task_id"],
        split=config["panel"]["train_split"],
        per_target=int(config["panel"]["train_per_target"]),
    )
    release_cuda()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    branch, model, optimizer = _run_branch(
        snapshot=snapshot,
        task_root=task_root,
        rows=train_rows,
        config=config,
        checkpoint=restored_checkpoint,
        schedule_sha256=sha256_file(config_path),
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    current_allocated = int(torch.cuda.memory_allocated())
    current_reserved = int(torch.cuda.memory_reserved())
    free_after, observed_total = torch.cuda.mem_get_info()
    if observed_total != total:
        raise TrainingGateError("CUDA total memory changed during resource probe")
    peak_fraction = peak_reserved / total
    maximum_fraction = float(probe_config["thresholds"]["maximum_peak_reserved_fraction"])
    if peak_fraction >= maximum_fraction:
        raise TrainingGateError(f"resource probe peak exceeds threshold: {peak_fraction} >= {maximum_fraction}")
    del model, optimizer
    release_cuda()
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_resource_probe",
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "lineage": "pg1_pi",
        "role": "primary",
        "checkpoint_sha256": expected_checkpoint_sha256,
        "train_item_ids": [row["item_id"] for row in train_rows],
        "confirmation_items_read": 0,
        "update": branch["update"],
        "frozen_exact": branch["frozen_exact"],
        "cuda_memory": {
            "total_bytes": int(total),
            "free_before_bytes": int(free_before),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_fraction": peak_fraction,
            "current_allocated_bytes_before_release": current_allocated,
            "current_reserved_bytes_before_release": current_reserved,
            "free_after_update_bytes": int(free_after),
            "maximum_peak_reserved_fraction": maximum_fraction,
        },
        "claim_boundary": "mechanical_resource_measurement_only_no_scientific_claim",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--restored-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        root=args.root.resolve(),
        source_contract=args.source_contract.resolve(),
        matrix=args.matrix.resolve(),
        config_path=args.config.resolve(),
        probe_config_path=args.probe_config.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        instance_id=args.instance_id,
        restored_checkpoint=args.restored_checkpoint.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps({"status": packet["status"], "cuda_memory": packet["cuda_memory"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
