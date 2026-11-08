"""No-science H100 preflight for one frozen fused-base mechanical range cell."""

from __future__ import annotations

import argparse
import hashlib
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


def _verify_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_artifact_rehearsal":
        raise TrainingGateError("pilot source contract is not frozen")
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"pilot source contract hash mismatch: {name}")
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


def _repository_subset_sha256(repository: dict[str, Any]) -> str:
    encoded = json.dumps(repository, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_from_manifest(
    model_manifest: dict[str, Any], *, repo: str, revision: str
) -> tuple[dict[str, Any], list[str]]:
    """Normalize the historical multi-repo and current exact S3 manifest schemas."""
    if (
        model_manifest.get("status") == "GREEN"
        and model_manifest.get("mode") == "NO_SCIENCE_CACHE_SUBSET"
    ):
        repositories = model_manifest.get("repositories", [])
        matches = [
            row
            for row in repositories
            if row.get("repo") == repo and row.get("revision") == revision
        ]
        if len(matches) != 1:
            raise TrainingGateError("range repository is not unique in the cache manifest")
        return matches[0], [row["repo"] for row in repositories if row is not matches[0]]
    if (
        model_manifest.get("status") == "FROZEN_INPUT_MANIFEST"
        and model_manifest.get("scope") == "s3_worker_cache_hydration_only"
        and model_manifest.get("repo") == repo
        and model_manifest.get("revision") == revision
        and isinstance(model_manifest.get("files"), list)
    ):
        return {
            "repo": repo,
            "revision": revision,
            "files": model_manifest["files"],
        }, []
    raise TrainingGateError("range model-cache manifest is not an allowed exact schema")


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
        raise FileExistsError(f"refusing to overwrite pilot preflight: {output}")
    contract = _verify_contract(root, source_contract)
    config_path = root / contract["artifacts"]["pilot_config"]["path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lineage = config.get("lineage", {})
    lineage_id = str(lineage.get("id", ""))
    expected_lineage = ALLOWED_LINEAGES.get(lineage_id)
    if (
        config.get("status") != "frozen_before_h100"
        or expected_lineage is None
        or lineage != {"id": lineage_id, **expected_lineage}
    ):
        raise TrainingGateError("range preflight is not scoped to an allowed frozen lineage")
    result_dir = root / contract["destination"]["result_dir"]
    if result_dir.exists():
        raise TrainingGateError("pilot result directory already exists before execution")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("pilot destination is not an H100")

    entities = {(row.lineage, row.role): row for row in frozen_entities(root / contract["artifacts"]["matrix"]["path"])}
    entity = entities[(lineage_id, "fused")]
    model_manifest_path = root / contract["artifacts"]["model_cache_manifest"]["path"]
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    repository, excluded_repositories = _repository_from_manifest(
        model_manifest, repo=entity.repo, revision=entity.revision
    )
    snapshot = cache_root / ("models--" + repository["repo"].replace("/", "--")) / "snapshots" / repository["revision"]
    cache_errors = []
    for expected in repository["files"]:
        path = snapshot / expected["path"]
        if not path.is_file():
            cache_errors.append(f"missing:{expected['path']}")
        elif path.stat().st_size != int(expected["bytes"]):
            cache_errors.append(f"size:{expected['path']}")
        elif sha256_file(path) != expected["sha256"]:
            cache_errors.append(f"sha256:{expected['path']}")
    if cache_errors:
        raise TrainingGateError(f"range cache verification failed: {cache_errors[:8]}")
    subset_sha256 = _repository_subset_sha256(repository)
    atomic_json(
        cache_root / CACHE_GATE_MARKER,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN",
            "destination_instance_id": instance_id,
            "content_manifest_sha256": subset_sha256,
            "network_cache_mutation_after_this_marker": "FORBIDDEN",
            "scope": f"{lineage_id}_mechanical_range_screen_only",
            "repository": entity.repo,
            "revision": entity.revision,
        },
    )
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    resolved = resolve_snapshot(cache_root, entity.repo, entity.revision)

    discovery_seal = json.loads((task_root / "manifests/discovery-seal.json").read_text(encoding="utf-8"))
    if discovery_seal.get("status") != "GREEN":
        raise TrainingGateError("destination discovery-only task seal is not green")
    forbidden_task_paths = [task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256"]
    if any(path.exists() for path in forbidden_task_paths):
        raise TrainingGateError("confirmation-capable task artifact is present on the pilot destination")
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
        raise TrainingGateError(f"destination is missing selected pilot images: {missing_images[:4]}")
    present_images = sorted(path for path in task_root.rglob("*.png") if path.is_file())
    selected_paths = sorted((task_root / row["image"]).resolve() for row in selected)
    if [path.resolve() for path in present_images] != selected_paths:
        raise TrainingGateError("pilot task payload is not exactly the frozen 120-image selection")

    with tempfile.TemporaryDirectory(prefix="parallax-h100-pilot-plants-") as temporary:
        planted = run_planted_gate(Path(temporary) / "gate.json")
    if planted.get("status") != "GREEN":
        raise TrainingGateError("weight-free plants failed on the pilot H100 runtime")
    disk = shutil.disk_usage(result_dir.parent)
    minimum_free = int(contract["destination"]["minimum_free_bytes"])
    if disk.free < minimum_free:
        raise TrainingGateError(f"destination free disk is below pilot contract: {disk.free} < {minimum_free}")

    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_artifact_rehearsal_preflight",
        "status": "GREEN",
        "science_items_evaluated": 0,
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "runtime_versions": _runtime_versions(root / contract["artifacts"]["requirements"]["path"]),
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "cache_gate_sha256": gate["content_manifest_sha256"],
        "cache_scope": gate["scope"],
        "model_cache_manifest": {
            "path": str(model_manifest_path),
            "sha256": sha256_file(model_manifest_path),
            "repository": entity.repo,
            "revision": entity.revision,
            "verified_files": len(repository["files"]),
            "verified_bytes": sum(int(row["bytes"]) for row in repository["files"]),
            "excluded_supporting_repositories": excluded_repositories,
        },
        "snapshot": {"path": str(resolved)},
        "selected_item_ids": [row["item_id"] for row in selected],
        "selected_image_count": len(selected_paths),
        "confirmation_items_read": 0,
        "confirmation_capable_artifacts_present": False,
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
