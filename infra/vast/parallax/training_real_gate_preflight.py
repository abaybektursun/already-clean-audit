"""No-science destination preflight for the frozen real-training H100 gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .cache_gate import CACHE_GATE_MARKER
from .g1_common import offline_gate, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .training_common import TrainingGateError
from .training_model import select_split_stratified
from .training_planted_gate import run as run_planted_gate


def _verify_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_harness_real_forward_gate":
        raise TrainingGateError("real-gate source contract is not frozen")
    for name, row in contract["artifacts"].items():
        path = root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise TrainingGateError(f"source contract hash mismatch: {name}")
    return contract


def _runtime_versions(lock_path: Path) -> dict[str, str]:
    required = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        if name in {"torch", "transformers", "peft", "safetensors", "pillow", "pyyaml"}:
            required[name] = version
    observed = {}
    package_names = {"pillow": "Pillow", "pyyaml": "PyYAML"}
    for name, expected in required.items():
        observed[name] = importlib.metadata.version(package_names.get(name, name))
        if observed[name] != expected:
            raise TrainingGateError(f"runtime version mismatch: {name}={observed[name]} expected={expected}")
    return observed


def run(
    *,
    root: Path,
    source_contract: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    output: Path,
) -> dict[str, Any]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to overwrite real-gate preflight: {output}")
    contract = _verify_contract(root, source_contract)
    config_path = root / contract["artifacts"]["config"]["path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_dir = root / contract["destination"]["result_dir"]
    if result_dir.exists():
        raise TrainingGateError("real-gate result directory already exists before science")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("destination is not an H100")
    model_manifest_path = root / contract["artifacts"]["model_cache_manifest"]["path"]
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("status") != "GREEN" or model_manifest.get("mode") != "NO_SCIENCE_CACHE_SUBSET":
        raise TrainingGateError("real-gate model-cache subset manifest is not green")
    cache_errors = []
    for repository in model_manifest["repositories"]:
        snapshot = cache_root / ("models--" + repository["repo"].replace("/", "--")) / "snapshots" / repository["revision"]
        for expected in repository["files"]:
            path = snapshot / expected["path"]
            if not path.is_file():
                cache_errors.append(f"missing:{repository['repo']}:{expected['path']}")
            elif path.stat().st_size != int(expected["bytes"]):
                cache_errors.append(f"size:{repository['repo']}:{expected['path']}")
            elif sha256_file(path) != expected["sha256"]:
                cache_errors.append(f"sha256:{repository['repo']}:{expected['path']}")
    if cache_errors:
        raise TrainingGateError(f"real-gate cache verification failed: {cache_errors[:8]}")
    atomic_json(
        cache_root / CACHE_GATE_MARKER,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN",
            "destination_instance_id": instance_id,
            "content_manifest_sha256": model_manifest["content_manifest_sha256"],
            "network_cache_mutation_after_this_marker": "FORBIDDEN",
            "scope": "two_fused_bases_real_training_gate_only",
        },
    )
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    entities = {(row.lineage, row.role): row for row in frozen_entities(root / contract["artifacts"]["matrix"]["path"])}
    snapshots = {}
    for lineage in config["lineages"]["order"]:
        entity = entities[(lineage, "fused")]
        snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
        files = [path for path in snapshot.rglob("*") if path.is_file()]
        snapshots[lineage] = {
            "repo": entity.repo,
            "revision": entity.revision,
            "path": str(snapshot),
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
    discovery_seal = json.loads((task_root / "manifests/discovery-seal.json").read_text(encoding="utf-8"))
    if discovery_seal.get("status") != "GREEN":
        raise TrainingGateError("destination discovery-only task seal is not green")
    from .parallax_tasks import DiscoveryTaskBank

    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    selected = []
    for split, count_key in ((config["panel"]["train_split"], "train_per_target"),
                             (config["panel"]["discovery_split"], "discovery_per_target")):
        selected.extend(
            select_split_stratified(
                rows,
                task_id=config["panel"]["task_id"],
                split=split,
                per_target=int(config["panel"][count_key]),
            )
        )
    missing_images = [row["image"] for row in selected if not (task_root / row["image"]).is_file()]
    if missing_images:
        raise TrainingGateError(f"destination is missing selected task images: {missing_images[:4]}")
    with tempfile.TemporaryDirectory(prefix="parallax-h100-plants-") as temporary:
        planted = run_planted_gate(Path(temporary) / "gate.json")
    if planted.get("status") != "GREEN":
        raise TrainingGateError("weight-free plants failed on the H100 runtime")
    disk = shutil.disk_usage(result_dir.parent)
    minimum_free = int(contract["destination"]["minimum_free_bytes"])
    if disk.free < minimum_free:
        raise TrainingGateError(f"destination free disk is below contract: {disk.free} < {minimum_free}")
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_harness_real_forward_gate_preflight",
        "status": "GREEN",
        "science_items_evaluated": 0,
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "runtime_versions": _runtime_versions(root / contract["artifacts"]["requirements"]["path"]),
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "cache_gate_sha256": gate.get("content_manifest_sha256"),
        "model_cache_manifest": {
            "path": str(model_manifest_path),
            "sha256": sha256_file(model_manifest_path),
            "verified_files": model_manifest["verified_file_count"],
            "verified_bytes": model_manifest["verified_bytes"],
        },
        "snapshots": snapshots,
        "selected_item_ids": [row["item_id"] for row in selected],
        "confirmation_items_read": 0,
        "h100_planted_gate": {
            "status": planted["status"],
            "exact_resume": planted["determinism"]["exact_resume"],
            "frozen_bytes_exact": planted["frozen_parameter_bytes"]["exact"],
        },
        "disk": {"free_bytes": disk.free, "minimum_free_bytes": minimum_free},
        "result_dir_absent": True,
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        root=args.root.resolve(),
        source_contract=args.source_contract.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        instance_id=args.instance_id,
        output=args.output.resolve(),
    )
    print(json.dumps({"status": packet["status"], "gpu": packet["gpu"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
