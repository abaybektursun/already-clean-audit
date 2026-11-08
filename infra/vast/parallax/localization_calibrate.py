"""Calibrate the PG2 localization amplitude and MDE without confirmation access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_common import BehavioralGateError, select_target_stratified, sign_flip_mde
from .behavioral_localization import select_localization_amplitude
from .behavioral_localization_model import evaluate_localization
from .behavioral_model import (
    assert_parameter_versions,
    decoder_layer,
    fixed_parameter_versions,
    identity_intervention,
    processor_batch,
    to_device,
)
from .g1_common import assert_attention_eager, offline_gate, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import FrozenTaskBank


def _tensor_sha(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def _pg2_entity(matrix: Path) -> Any:
    rows = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    return rows[("pg2", "fused")]


def _configs(model: Any) -> list[Any]:
    return [
        value
        for value in (model.config, model.config.text_config, model.config.vision_config)
        if hasattr(value, "_attn_implementation")
    ]


def _validate_prerequisites(root: Path, config: dict[str, Any], task_root: Path) -> dict[str, Any]:
    confirmation = (
        root
        / "studies/lineage/cross-lineage/results/parallax_behavioral_confirmation_v1/confirmation.json"
    )
    provenance = confirmation.parent / "confirmation_provenance_audit.json"
    if sha256_file(confirmation) != config["prerequisites"]["behavioral_confirmation_sha256"]:
        raise BehavioralGateError("localization prerequisite confirmation hash mismatch")
    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    if provenance_payload.get("status") != config["prerequisites"]["provenance_audit_status"]:
        raise BehavioralGateError("localization provenance audit is not green")
    audit_path = task_root / "manifests/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    content_path = task_root / "manifests/content.json"
    if audit.get("status") != "GREEN" or audit.get("errors"):
        raise BehavioralGateError("new localization panel audit is not green")
    if audit.get("content_manifest_sha256") != sha256_file(content_path):
        raise BehavioralGateError("localization panel audit does not bind content manifest")
    return {
        "behavioral_confirmation_sha256": sha256_file(confirmation),
        "provenance_audit_sha256": sha256_file(provenance),
        "panel_audit_sha256": sha256_file(audit_path),
        "panel_content_sha256": sha256_file(content_path),
    }


def run(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    prior_calibration_dir: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite localization calibration: {output_dir}")
    output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise BehavioralGateError("localization calibration requires CUDA")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_new_panel_materialization":
        raise BehavioralGateError("localization design is not frozen")
    prerequisites = _validate_prerequisites(root, config, task_root)
    bank = FrozenTaskBank(
        task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256"
    )
    panel = select_target_stratified(
        bank.iter_discovery(task_id=config["new_panel"]["task_id"]),
        task_id=config["new_panel"]["task_id"],
        per_target=int(config["new_panel"]["discovery_per_target"]),
    )
    if len(panel) != int(config["new_panel"]["discovery_items"]):
        raise BehavioralGateError("localization discovery panel size mismatch")
    direction_path = prior_calibration_dir / "pg2-direction.safetensors"
    if sha256_file(direction_path) != config["direction"]["file_sha256"]:
        raise BehavioralGateError("banked PG2 direction file hash mismatch")
    direction = load_file(direction_path)["direction"].float()
    if _tensor_sha(direction) != config["direction"]["tensor_sha256"]:
        raise BehavioralGateError("banked PG2 direction tensor hash mismatch")
    entity = _pg2_entity(matrix)
    if entity.repo != config["lineage"]["checkpoint"] or entity.revision != config["lineage"]["revision"]:
        raise BehavioralGateError("PG2 entity differs from localization freeze")
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    model.config.use_cache = False
    assert_attention_eager(_configs(model))
    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    torch.use_deterministic_algorithms(True)
    layer = decoder_layer(model, int(config["site"]["layer_index"]))
    versions = fixed_parameter_versions(model)
    first_inputs = to_device(processor_batch(processor, panel[:2], task_root))
    identity = identity_intervention(model, first_inputs, layer)
    del first_inputs
    ladder = [float(value) for value in config["energy"]["common_reference_amplitude_ladder"]]
    sensitive_rows, sensitive_execution = evaluate_localization(
        model=model,
        processor=processor,
        rows=panel,
        task_root=task_root,
        layer=layer,
        sensitive_direction=direction,
        namespace=config["direction"]["random_seed_namespace"],
        amplitudes=ladder,
        arms=["full_sensitive", "image_sensitive", "prompt_sensitive"],
        batch_size=4,
        expected_image_tokens=int(config["site"]["expected_image_tokens_per_item"]),
        per_token_fraction_max=float(
            config["energy"]["maximum_location_per_token_activation_rms_fraction"]
        ),
        total_energy_tolerance=float(
            config["energy"]["injected_bfloat16_max_relative_total_energy_error"]
        ),
    )
    amplitude = select_localization_amplitude(
        sensitive_rows,
        ladder,
        median_kl_min=float(config["energy"]["full_sensitive_median_suffix_kl_window"][0]),
        median_kl_max=float(config["energy"]["full_sensitive_median_suffix_kl_window"][1]),
        p90_kl_max=float(config["energy"]["every_sensitive_location_p90_suffix_kl_max"]),
        per_token_fraction_max=float(
            config["energy"]["maximum_location_per_token_activation_rms_fraction"]
        ),
    )
    null_rows: list[dict[str, Any]] = []
    null_execution: dict[str, Any] | None = None
    intrinsic_mde: dict[str, Any] | None = None
    if amplitude["selected_amplitude"] is not None:
        null_rows, null_execution = evaluate_localization(
            model=model,
            processor=processor,
            rows=panel,
            task_root=task_root,
            layer=layer,
            sensitive_direction=direction,
            namespace=config["direction"]["random_seed_namespace"],
            amplitudes=[float(amplitude["selected_amplitude"])],
            arms=["full_rotated", "full_white", "image_rotated", "prompt_rotated"],
            batch_size=4,
            expected_image_tokens=int(config["site"]["expected_image_tokens_per_item"]),
            per_token_fraction_max=float(
                config["energy"]["maximum_location_per_token_activation_rms_fraction"]
            ),
            total_energy_tolerance=float(
                config["energy"]["injected_bfloat16_max_relative_total_energy_error"]
            ),
        )
        cells = {
            arm: {row["item_id"]: row["damage_nats"] for row in null_rows if row["arm"] == arm}
            for arm in ("full_rotated", "full_white", "image_rotated", "prompt_rotated")
        }
        ids = sorted(cells["full_rotated"])
        full_null = [cells["full_rotated"][item] - cells["full_white"][item] for item in ids]
        location_null = [
            cells["image_rotated"][item] - cells["prompt_rotated"][item] for item in ids
        ]
        mde_config = config["mde"]
        common = {
            "seed": 20260718,
            "draws": int(mde_config["sign_flip_draws"]),
            "quantile": float(mde_config["sign_flip_absolute_two_sided_quantile"]),
            "minimum": float(mde_config["fixed_minimum_nats"]),
        }
        full_band = sign_flip_mde(full_null, **common)
        location_band = sign_flip_mde(location_null, **{**common, "seed": 20260719})
        frozen = max(
            float(mde_config["prior_pg2_frozen_mde_nats"]),
            full_band["frozen_mde_nats"],
            location_band["frozen_mde_nats"],
        )
        intrinsic_mde = {
            "prior_pg2_frozen_mde_nats": float(mde_config["prior_pg2_frozen_mde_nats"]),
            "full_rotated_minus_white": full_band,
            "image_rotated_minus_prompt_rotated": location_band,
            "frozen_mde_nats": frozen,
        }
    assert_parameter_versions(model, versions)
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_CALIBRATED" if amplitude["status"] == "GREEN" else amplitude["status"],
        "provider": "vast.ai",
        "instance_id": instance_id,
        "stage": "pg2_prefix_position_localization_calibration",
        "confirmation_metadata_rows_parsed_by_calibration": 0,
        "confirmation_unique_items_evaluated_by_calibration": 0,
        "bindings": {
            **prerequisites,
            "config_sha256": sha256_file(config_path),
            "matrix_sha256": sha256_file(matrix),
            "direction_file_sha256": sha256_file(direction_path),
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
        "identity": identity,
        "parameter_versions": {"status": "GREEN", "count": len(versions)},
        "amplitude": amplitude,
        "intrinsic_mde": intrinsic_mde,
        "sensitive_ladder": sensitive_rows,
        "null_calibration": null_rows,
        "execution": {"sensitive": sensitive_execution, "nulls": null_execution},
    }
    atomic_json(output_dir / "calibration.json", packet)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN_CONTENT_HASHED",
            "stage": packet["stage"],
            "entries": [
                {
                    "path": "calibration.json",
                    "bytes": (output_dir / "calibration.json").stat().st_size,
                    "sha256": sha256_file(output_dir / "calibration.json"),
                }
            ],
        },
    )
    del model, processor, direction
    release_cuda()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--prior-calibration-dir", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        root=args.root,
        matrix=args.matrix,
        config_path=args.config,
        cache_root=args.cache_root,
        task_root=args.task_root,
        prior_calibration_dir=args.prior_calibration_dir,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
