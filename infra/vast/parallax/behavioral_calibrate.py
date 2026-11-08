"""Calibrate lineage-intrinsic amplitude, direction, and MDE without reading confirmation items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_common import BehavioralGateError, select_amplitude, select_target_stratified, sign_flip_mde
from .behavioral_model import (
    assert_parameter_versions,
    decoder_layer,
    discovery_direction,
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
    return [value for value in (model.config, model.config.text_config, model.config.vision_config) if hasattr(value, "_attn_implementation")]


def _direction_sha(direction: Any) -> str:
    return hashlib.sha256(direction.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def _validate_hour6(root: Path, config: dict[str, Any]) -> dict[str, str]:
    path = root / config["prerequisite"]["hour6_packet"]
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet.get("status") != "GREEN" or packet.get("hour_6_decision") != config["prerequisite"]["required_decision"]:
        raise BehavioralGateError("paired G1 does not license behavioral calibration")
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_frozen_task_audit(task_root: Path) -> dict[str, str]:
    """Validate banked integrity evidence without parsing sealed confirmation rows."""
    audit_path = task_root / "manifests/audit.json"
    content_path = task_root / "manifests/content.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    content = json.loads(content_path.read_text(encoding="utf-8"))
    if audit.get("status") != "GREEN" or audit.get("errors"):
        raise BehavioralGateError("frozen task-bank audit is not green")
    if audit.get("items_jsonl_sha256") != content.get("items_jsonl_sha256"):
        raise BehavioralGateError("frozen task audit does not bind the current item manifest")
    if audit.get("content_manifest_sha256") != sha256_file(content_path):
        raise BehavioralGateError("frozen task audit does not bind the current content manifest")
    return {"path": str(audit_path), "sha256": sha256_file(audit_path)}


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
    from safetensors.torch import save_file
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite behavioral calibration: {output_dir}")
    output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise BehavioralGateError("behavioral calibration requires CUDA")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    torch.use_deterministic_algorithms(True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_calibration":
        raise BehavioralGateError("behavioral MDE config is not frozen")
    hour6 = _validate_hour6(root, config)
    task_audit = _validate_frozen_task_audit(task_root)
    bank = FrozenTaskBank(task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256")
    panel = select_target_stratified(
        bank.iter_discovery(task_id=config["panel"]["task_id"]),
        task_id=config["panel"]["task_id"],
        per_target=int(config["panel"]["discovery_per_target"]),
    )
    if len(panel) != int(config["panel"]["discovery_items"]):
        raise BehavioralGateError(f"discovery panel size mismatch: {len(panel)}")
    entities = _fused_entities(matrix)
    batch_size = 4
    lineages: dict[str, Any] = {}
    for lineage in ("pg1_pi", "pg2"):
        entity = entities[lineage]
        snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda").eval()
        model.config.use_cache = False
        assert_attention_eager(_configs(model))
        layer = decoder_layer(model, int(config["site"]["layer_index"]))
        versions = fixed_parameter_versions(model)
        first_inputs = to_device(processor_batch(processor, panel[:2], task_root))
        identity = identity_intervention(model, first_inputs, layer)
        del first_inputs
        direction, gradient_diagnostics = discovery_direction(
            model=model,
            processor=processor,
            rows=panel,
            task_root=task_root,
            layer=layer,
            batch_size=batch_size,
        )
        direction_path = output_dir / f"{lineage}-direction.safetensors"
        save_file({"direction": direction.detach().cpu().float().contiguous()}, direction_path)
        ladder = [float(value) for value in config["energy"]["amplitude_ladder"]]
        sensitive_rows, sensitive_audit = evaluate_constructions(
            model=model,
            processor=processor,
            rows=panel,
            task_root=task_root,
            layer=layer,
            sensitive_direction=direction,
            lineage=lineage,
            namespace=config["direction"]["random_seed_namespace"],
            amplitudes=ladder,
            constructions=["sensitive_coherent"],
            batch_size=batch_size,
        )
        amplitude = select_amplitude(
            sensitive_rows,
            ladder,
            median_kl_min=float(config["energy"]["median_kl_window"][0]),
            median_kl_max=float(config["energy"]["median_kl_window"][1]),
            p90_kl_max=float(config["energy"]["p90_kl_saturation_max"]),
        )
        null_rows: list[dict[str, Any]] = []
        null_audit: dict[str, Any] | None = None
        mde: dict[str, float] | None = None
        if amplitude["selected_amplitude"] is not None:
            null_rows, null_audit = evaluate_constructions(
                model=model,
                processor=processor,
                rows=panel,
                task_root=task_root,
                layer=layer,
                sensitive_direction=direction,
                lineage=lineage,
                namespace=config["direction"]["random_seed_namespace"],
                amplitudes=[float(amplitude["selected_amplitude"])],
                constructions=["rotated_coherent", "white"],
                batch_size=batch_size,
            )
            by_cell = {
                name: {row["item_id"]: row["damage_nats"] for row in null_rows if row["construction"] == name}
                for name in ("rotated_coherent", "white")
            }
            ids = sorted(by_cell["rotated_coherent"])
            differences = [by_cell["rotated_coherent"][item] - by_cell["white"][item] for item in ids]
            mde = sign_flip_mde(
                differences,
                seed=20260716 + (0 if lineage == "pg1_pi" else 1),
                draws=int(config["intrinsic_mde"]["sign_flip_draws"]),
                quantile=float(config["intrinsic_mde"]["absolute_two_sided_quantile"]),
                minimum=float(config["intrinsic_mde"]["minimum_nats"]),
            )
        assert_parameter_versions(model, versions)
        lineages[lineage] = {
            "status": amplitude["status"],
            "role": "primary" if lineage == "pg1_pi" else "supporting",
            "checkpoint": {"repo": entity.repo, "revision": entity.revision},
            "identity": identity,
            "parameter_versions": {"status": "GREEN", "count": len(versions)},
            "direction": {
                "path": direction_path.name,
                "sha256": sha256_file(direction_path),
                "tensor_sha256": _direction_sha(direction),
                "width": int(direction.numel()),
                "norm": float(direction.norm()),
                "gradient_diagnostics": gradient_diagnostics,
            },
            "amplitude": amplitude,
            "intrinsic_mde": mde,
            "sensitive_ladder": sensitive_rows,
            "null_calibration": null_rows,
            "execution_audits": {"sensitive": sensitive_audit, "nulls": null_audit},
        }
        del model, processor, direction
        release_cuda()
    status = (
        "GREEN_CALIBRATED"
        if all(row["status"] == "GREEN" for row in lineages.values())
        else "PARTIAL_OR_VOID_AMPLITUDE"
    )
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": status,
        "provider": "vast.ai",
        "instance_id": instance_id,
        "stage": "paired_behavioral_mde_calibration",
        "confirmation_items_read": 0,
        "lineage_firewall": {
            "primary": "pg1_pi",
            "supporting": "pg2",
            "direct_cross_gemma_numeric_comparison": False,
            "amplitudes_and_mdes_calibrated_independently": True,
        },
        "bindings": {
            "hour6": hour6,
            "config_sha256": sha256_file(config_path),
            "matrix_sha256": sha256_file(matrix),
            "task_content_sha256": sha256_file(task_root / "manifests/content.json"),
            "task_audit": task_audit,
            "confirmation_seal_file_sha256": sha256_file(task_root / "manifests/confirmation.sha256"),
            "cache_manifest_sha256": gate.get("content_manifest_sha256"),
        },
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
        "lineages": lineages,
    }
    packet_path = output_dir / "calibration.json"
    atomic_json(packet_path, packet)
    entries = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.json":
            continue
        entries.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        root=args.root,
        matrix=args.matrix,
        config_path=args.config,
        cache_root=args.cache_root,
        task_root=args.task_root,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
