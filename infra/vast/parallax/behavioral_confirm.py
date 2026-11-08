"""Run the sealed paired behavioral confirmation after a content-addressed arm freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_common import BehavioralGateError, adjudicate_range, select_target_stratified
from .behavioral_model import (
    assert_parameter_versions,
    decoder_layer,
    evaluate_constructions,
    fixed_parameter_versions,
    identity_intervention,
    processor_batch,
    to_device,
)
from .g1_common import assert_attention_eager, offline_gate, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import FrozenTaskBank


def _fused_entities(matrix: Path) -> dict[str, Any]:
    rows = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    return {lineage: rows[(lineage, "fused")] for lineage in ("pg1_pi", "pg2")}


def _configs(model: Any) -> list[Any]:
    return [
        value
        for value in (model.config, model.config.text_config, model.config.vision_config)
        if hasattr(value, "_attn_implementation")
    ]


def _tensor_sha(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def _validate_freeze(
    *, config_path: Path, calibration_dir: Path, task_root: Path, arm_freeze: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(arm_freeze.read_text(encoding="utf-8"))
    calibration_path = calibration_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    checks = {
        "status": freeze.get("status") == "FROZEN_FOR_CONFIRMATION",
        "config": freeze.get("bindings", {}).get("config_sha256") == sha256_file(config_path),
        "calibration": freeze.get("bindings", {}).get("calibration_sha256")
        == sha256_file(calibration_path),
        "calibration_manifest": freeze.get("bindings", {}).get("calibration_manifest_sha256")
        == sha256_file(calibration_dir / "manifest.json"),
        "runner": freeze.get("bindings", {}).get("confirmation_runner_sha256")
        == sha256_file(Path(__file__)),
        "seal": freeze.get("confirmation_seal_sha256")
        == sha256_file(task_root / "manifests/confirmation.sha256"),
        "zero_prior_confirmation": calibration.get("confirmation_items_read") == 0,
    }
    if not all(checks.values()):
        raise BehavioralGateError(f"confirmation freeze binding failed: {checks}")
    return freeze, calibration


def run(
    *,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    calibration_dir: Path,
    arm_freeze: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite behavioral confirmation: {output_dir}")
    output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise BehavioralGateError("behavioral confirmation requires CUDA")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    torch.use_deterministic_algorithms(True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    freeze, calibration = _validate_freeze(
        config_path=config_path,
        calibration_dir=calibration_dir,
        task_root=task_root,
        arm_freeze=arm_freeze,
    )
    bank = FrozenTaskBank(
        task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256"
    )
    # Selection is deliberately order-invariant, so it must inspect every eligible
    # task-confirmation metadata row.  Distinguish that exposure from the subset
    # whose images/model outputs are actually evaluated.
    eligible_confirmation = list(
        bank.iter_confirmation(arm_freeze=arm_freeze, task_id=config["panel"]["task_id"])
    )
    panel = select_target_stratified(
        eligible_confirmation,
        task_id=config["panel"]["task_id"],
        per_target=int(config["panel"]["confirmation_per_target"]),
    )
    if len(panel) != int(config["panel"]["confirmation_items"]):
        raise BehavioralGateError(f"confirmation panel size mismatch: {len(panel)}")
    entities = _fused_entities(matrix)
    lineages: dict[str, Any] = {}
    for offset, lineage in enumerate(("pg1_pi", "pg2")):
        entity = entities[lineage]
        frozen = freeze["lineages"][lineage]
        calibrated = calibration["lineages"][lineage]
        if frozen["checkpoint"] != {"repo": entity.repo, "revision": entity.revision}:
            raise BehavioralGateError(f"checkpoint freeze mismatch: {lineage}")
        direction_path = calibration_dir / frozen["direction_file"]
        if sha256_file(direction_path) != frozen["direction_sha256"]:
            raise BehavioralGateError(f"direction file freeze mismatch: {lineage}")
        direction = load_file(direction_path)["direction"].float()
        if _tensor_sha(direction) != frozen["direction_tensor_sha256"]:
            raise BehavioralGateError(f"direction tensor freeze mismatch: {lineage}")
        if frozen["selected_amplitude"] != calibrated["amplitude"]["selected_amplitude"]:
            raise BehavioralGateError(f"amplitude freeze mismatch: {lineage}")
        if frozen["frozen_mde_nats"] != calibrated["intrinsic_mde"]["frozen_mde_nats"]:
            raise BehavioralGateError(f"MDE freeze mismatch: {lineage}")
        snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda").eval()
        model.config.use_cache = False
        assert_attention_eager(_configs(model))
        layer = decoder_layer(model, int(config["site"]["layer_index"]))
        versions = fixed_parameter_versions(model)
        inputs = to_device(processor_batch(processor, panel[:2], task_root))
        identity = identity_intervention(model, inputs, layer)
        del inputs
        rows, execution = evaluate_constructions(
            model=model,
            processor=processor,
            rows=panel,
            task_root=task_root,
            layer=layer,
            sensitive_direction=direction,
            lineage=lineage,
            namespace=config["direction"]["random_seed_namespace"],
            amplitudes=[float(frozen["selected_amplitude"])],
            constructions=["sensitive_coherent", "rotated_coherent", "white"],
            batch_size=4,
        )
        assert_parameter_versions(model, versions)
        cells = {
            name: {
                row["item_id"]: row["damage_nats"]
                for row in rows
                if row["construction"] == name
            }
            for name in ("sensitive_coherent", "rotated_coherent", "white")
        }
        verdict = adjudicate_range(
            cells["sensitive_coherent"],
            cells["rotated_coherent"],
            cells["white"],
            frozen_mde=float(frozen["frozen_mde_nats"]),
            seed=20260717 + offset * 10,
            draws=int(config["intrinsic_mde"]["confirmation_gate"]["bootstrap_draws"]),
            lower_quantile=float(
                config["intrinsic_mde"]["confirmation_gate"]["familywise_lower_quantile"]
            ),
        )
        means = {
            name: sum(values.values()) / len(values) for name, values in cells.items()
        }
        lineages[lineage] = {
            "role": frozen["role"],
            "status": verdict["status"],
            "checkpoint": frozen["checkpoint"],
            "selected_amplitude": frozen["selected_amplitude"],
            "frozen_mde_nats": frozen["frozen_mde_nats"],
            "identity": identity,
            "parameter_versions": {"status": "GREEN", "count": len(versions)},
            "verdict": verdict,
            "construction_mean_damage_nats": means,
            "rows": rows,
            "execution": execution,
        }
        del model, processor, direction, rows
        release_cuda()
    normalized_boundary = {
        lineage: {
            "range_verdict": row["status"],
            "effect_over_mde": {
                name: comparison["effect_over_mde"]
                for name, comparison in row["verdict"]["comparisons"].items()
            },
        }
        for lineage, row in lineages.items()
    }
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "COMPLETE_SEALED_CONFIRMATION",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "stage": "paired_behavioral_mde_confirmation",
        "confirmation_metadata_rows_parsed": len(eligible_confirmation),
        "confirmation_unique_items_evaluated": len(panel),
        "confirmation_evaluations_per_item": 2,
        "bindings": {
            "arm_freeze_sha256": sha256_file(arm_freeze),
            "config_sha256": sha256_file(config_path),
            "calibration_sha256": sha256_file(calibration_dir / "calibration.json"),
            "cache_manifest_sha256": gate.get("content_manifest_sha256"),
        },
        "lineage_firewall": freeze["lineage_firewall"],
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "attention": "eager",
            "use_cache": False,
        },
        "panel": {
            "items": len(panel),
            "item_ids_sha256": hashlib.sha256(
                json.dumps([row["item_id"] for row in panel], separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "normalized_cross_lineage_boundary": normalized_boundary,
        "lineages": lineages,
    }
    atomic_json(output_dir / "confirmation.json", packet)
    entries = [
        {
            "path": "confirmation.json",
            "bytes": (output_dir / "confirmation.json").stat().st_size,
            "sha256": sha256_file(output_dir / "confirmation.json"),
        }
    ]
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN_CONTENT_HASHED",
            "stage": packet["stage"],
            "entries": entries,
        },
    )
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--arm-freeze", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        matrix=args.matrix,
        config_path=args.config,
        cache_root=args.cache_root,
        task_root=args.task_root,
        calibration_dir=args.calibration_dir,
        arm_freeze=args.arm_freeze,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
