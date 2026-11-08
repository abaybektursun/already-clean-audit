"""Run the one-shot, fixed-0.001 PG2 localization rescue confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_common import BehavioralGateError
from .behavioral_localization import adjudicate_localization
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


def _validate_freeze(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    task_root: Path,
    direction_path: Path,
    arm_freeze: Path,
) -> dict[str, Any]:
    from .localization_rescue_freeze import SOURCE_FILES

    frozen = json.loads(arm_freeze.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_sources = {path: sha256_file(root / path) for path in SOURCE_FILES}
    prerequisite_hashes = {
        name: {
            "path": str(row["path"]),
            "sha256": sha256_file(root / str(row["path"])),
        }
        for name, row in (
            ("behavioral_confirmation", config["prerequisites"]["behavioral_confirmation"]),
            ("localization_calibration", config["prerequisites"]["localization_calibration"]),
            ("localization_void", config["prerequisites"]["localization_void"]),
        )
    }
    expected_count = int(config["new_panel"]["confirmation_items"])
    checks = {
        "status": frozen.get("status") == "FROZEN_FOR_LOCALIZATION_RESCUE_CONFIRMATION",
        "config": frozen.get("bindings", {}).get("config_sha256") == sha256_file(config_path),
        "matrix": frozen.get("bindings", {}).get("matrix_sha256") == sha256_file(matrix),
        "panel_content": frozen.get("bindings", {}).get("panel_content_sha256")
        == sha256_file(task_root / "manifests/content.json"),
        "panel_audit": frozen.get("bindings", {}).get("panel_audit_sha256")
        == sha256_file(task_root / "manifests/audit.json"),
        "direction": frozen.get("bindings", {}).get("direction_file_sha256")
        == sha256_file(direction_path),
        "prerequisites": frozen.get("bindings", {}).get("prerequisites")
        == prerequisite_hashes,
        "sources": frozen.get("bindings", {}).get("source_sha256") == expected_sources,
        "seal": frozen.get("confirmation_seal_sha256")
        == sha256_file(task_root / "manifests/confirmation.sha256"),
        "aggregate_audit_metadata": frozen.get(
            "confirmation_metadata_rows_parsed_by_frozen_aggregate_audit"
        )
        == expected_count,
        "zero_scientific_metadata": frozen.get(
            "confirmation_metadata_rows_parsed_by_scientific_runners_before_freeze"
        )
        == 0,
        "zero_evaluated": frozen.get("confirmation_unique_items_evaluated_before_freeze") == 0,
        "amplitude": float(frozen.get("selected_amplitude", -1)) == 0.001,
        "mde": float(frozen.get("frozen_mde_nats", -1)) == 0.021652735769748688,
        "second_rescue": frozen.get("second_rescue") == "FORBIDDEN",
    }
    if not all(checks.values()):
        raise BehavioralGateError(f"localization rescue freeze binding failed: {checks}")
    return frozen


def run(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    prior_calibration_dir: Path,
    arm_freeze: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite localization rescue: {output_dir}")
    output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise BehavioralGateError("localization rescue requires CUDA")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    direction_path = prior_calibration_dir / "pg2-direction.safetensors"
    frozen = _validate_freeze(
        root=root,
        matrix=matrix,
        config_path=config_path,
        task_root=task_root,
        direction_path=direction_path,
        arm_freeze=arm_freeze,
    )
    bank = FrozenTaskBank(
        task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256"
    )
    panel = list(
        bank.iter_confirmation(arm_freeze=arm_freeze, task_id=config["new_panel"]["task_id"])
    )
    expected_count = int(config["new_panel"]["confirmation_items"])
    if len(panel) != expected_count:
        raise BehavioralGateError("localization rescue panel size mismatch")
    if Counter(str(row["target"]) for row in panel) != Counter(
        {candidate: int(config["new_panel"]["confirmation_per_target"]) for candidate in "ABCD"}
    ):
        raise BehavioralGateError("localization rescue target balance mismatch")
    direction = load_file(direction_path)["direction"].float()
    if _tensor_sha(direction) != config["direction"]["tensor_sha256"]:
        raise BehavioralGateError("localization rescue direction tensor mismatch")
    entity = _pg2_entity(matrix)
    if entity.repo != config["lineage"]["checkpoint"] or entity.revision != config["lineage"]["revision"]:
        raise BehavioralGateError("PG2 rescue entity differs from freeze")
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    model.config.use_cache = False
    assert_attention_eager(_configs(model))
    torch.manual_seed(20260721)
    torch.cuda.manual_seed_all(20260721)
    torch.use_deterministic_algorithms(True)
    layer = decoder_layer(model, int(config["site"]["layer_index"]))
    versions = fixed_parameter_versions(model)
    first_inputs = to_device(processor_batch(processor, panel[:2], task_root))
    identity = identity_intervention(model, first_inputs, layer)
    del first_inputs
    rows, execution = evaluate_localization(
        model=model,
        processor=processor,
        rows=panel,
        task_root=task_root,
        layer=layer,
        sensitive_direction=direction,
        namespace=config["direction"]["random_seed_namespace"],
        amplitudes=[float(frozen["selected_amplitude"])],
        arms=list(frozen["frozen_arms"]),
        batch_size=4,
        expected_image_tokens=int(config["site"]["expected_image_tokens_per_item"]),
        per_token_fraction_max=float(
            config["energy"]["maximum_location_per_token_activation_rms_fraction"]
        ),
        total_energy_tolerance=float(
            config["energy"]["injected_bfloat16_max_relative_total_energy_error"]
        ),
    )
    assert_parameter_versions(model, versions)
    cells = {
        arm: {row["item_id"]: row["damage_nats"] for row in rows if row["arm"] == arm}
        for arm in frozen["frozen_arms"]
    }
    verdict = adjudicate_localization(
        cells,
        frozen_mde=float(frozen["frozen_mde_nats"]),
        seed=20260721,
        draws=int(config["mde"]["bootstrap_draws"]),
        lower_quantile=float(config["mde"]["familywise_lower_quantile"]),
    )
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "COMPLETE_SEALED_LOCALIZATION_RESCUE",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "stage": "pg2_prefix_position_localization_rescue_confirmation",
        "confirmation_metadata_rows_parsed": len(panel),
        "confirmation_unique_items_evaluated": len(panel),
        "confirmation_evaluations_per_item": len(frozen["frozen_arms"]),
        "bindings": {
            "arm_freeze_sha256": sha256_file(arm_freeze),
            "config_sha256": sha256_file(config_path),
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
        "selected_amplitude": frozen["selected_amplitude"],
        "frozen_mde_nats": frozen["frozen_mde_nats"],
        "identity": identity,
        "parameter_versions": {"status": "GREEN", "count": len(versions)},
        "verdict": verdict,
        "claim_boundary": frozen["claim_boundary"],
        "rows": rows,
        "execution": execution,
        "rescue_boundary": "one_shot_fixed_0.001_no_second_rescue",
    }
    atomic_json(output_dir / "confirmation.json", packet)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN_CONTENT_HASHED",
            "stage": packet["stage"],
            "entries": [
                {
                    "path": "confirmation.json",
                    "bytes": (output_dir / "confirmation.json").stat().st_size,
                    "sha256": sha256_file(output_dir / "confirmation.json"),
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
    parser.add_argument("--arm-freeze", type=Path, required=True)
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
        arm_freeze=args.arm_freeze,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
