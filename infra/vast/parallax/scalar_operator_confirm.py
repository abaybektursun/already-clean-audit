"""Run the sealed PG2 full-prefix scalar operator confirmation cell."""

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
from .scalar_operator import adjudicate_scalar_operator, deterministic_directions
from .scalar_operator_model import evaluate_scalar_operator


def _tensor_sha(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def _configs(model: Any) -> list[Any]:
    return [
        value
        for value in (model.config, model.config.text_config, model.config.vision_config)
        if hasattr(value, "_attn_implementation")
    ]


def _pg2_entity(matrix: Path) -> Any:
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    return entities[("pg2", "fused")]


def _validate_freeze(
    *, root: Path, matrix: Path, config_path: Path, task_root: Path, arm_freeze: Path
) -> dict[str, Any]:
    from .scalar_operator_freeze import SOURCE_FILES

    frozen = json.loads(arm_freeze.read_text(encoding="utf-8"))
    expected_sources = {path: sha256_file(root / path) for path in SOURCE_FILES}
    checks = {
        "status": frozen.get("status") == "FROZEN_FOR_SCALAR_OPERATOR_CONFIRMATION",
        "config": frozen.get("bindings", {}).get("config_sha256") == sha256_file(config_path),
        "matrix": frozen.get("bindings", {}).get("matrix_sha256") == sha256_file(matrix),
        "panel_content": frozen.get("bindings", {}).get("panel_content_sha256")
        == sha256_file(task_root / "manifests/content.json"),
        "panel_audit": frozen.get("bindings", {}).get("panel_audit_sha256")
        == sha256_file(task_root / "manifests/audit.json"),
        "seal": frozen.get("confirmation_seal_sha256")
        == sha256_file(task_root / "manifests/confirmation.sha256"),
        "sources": frozen.get("bindings", {}).get("source_sha256") == expected_sources,
        "zero_scientific_metadata": frozen.get(
            "confirmation_metadata_rows_parsed_by_scientific_runners_before_freeze"
        )
        == 0,
        "zero_evaluated": frozen.get("confirmation_unique_items_evaluated_before_freeze") == 0,
    }
    if not all(checks.values()):
        raise BehavioralGateError(f"scalar operator freeze binding failed: {checks}")
    return frozen


def run(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    arm_freeze: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite scalar operator output: {output_dir}")
    output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise BehavioralGateError("scalar operator confirmation requires CUDA")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = _validate_freeze(
        root=root,
        matrix=matrix,
        config_path=config_path,
        task_root=task_root,
        arm_freeze=arm_freeze,
    )
    bank = FrozenTaskBank(
        task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256"
    )
    panel = list(
        bank.iter_confirmation(arm_freeze=arm_freeze, task_id=config["panel"]["task_id"])
    )
    expected_items = int(config["panel"]["confirmation_items"])
    if len(panel) != expected_items:
        raise BehavioralGateError("scalar operator panel size differs")
    if Counter(str(row["target"]) for row in panel) != Counter(
        {candidate: int(config["panel"]["target_count_each"]) for candidate in "ABCD"}
    ):
        raise BehavioralGateError("scalar operator target balance differs")
    direction_path = root / config["prerequisites"]["direction"]["path"]
    sensitive = load_file(direction_path)["direction"].float()
    if _tensor_sha(sensitive) != config["prerequisites"]["direction"]["tensor_sha256"]:
        raise BehavioralGateError("scalar operator direction tensor hash differs")
    directions = deterministic_directions(
        sensitive,
        count=int(config["directions"]["count"]),
        namespace=str(config["directions"]["namespace"]),
    )
    entity = _pg2_entity(matrix)
    if entity.repo != config["lineage"]["checkpoint"] or entity.revision != config["lineage"]["revision"]:
        raise BehavioralGateError("scalar operator PG2 entity differs")
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    model.config.use_cache = False
    assert_attention_eager(_configs(model))
    torch.manual_seed(20260722)
    torch.cuda.manual_seed_all(20260722)
    torch.use_deterministic_algorithms(True)
    layer = decoder_layer(model, int(config["site"]["layer_index"]))
    versions = fixed_parameter_versions(model)
    first_inputs = to_device(processor_batch(processor, panel[:2], task_root))
    identity = identity_intervention(model, first_inputs, layer)
    del first_inputs
    cells, execution = evaluate_scalar_operator(
        model=model,
        processor=processor,
        rows=panel,
        task_root=task_root,
        layer=layer,
        directions=directions,
        epsilons=[float(value) for value in frozen["epsilon"]["ladder"]],
        namespace=str(frozen["directions"]["namespace"]),
        batch_size=2,
        expected_image_tokens=int(config["site"]["expected_image_tokens_per_item"]),
        per_token_fraction_max=float(frozen["epsilon"]["per_token_fraction_max"]),
        kl_min=float(frozen["epsilon"]["symmetric_kl_min_nats"]),
        kl_max=float(frozen["epsilon"]["symmetric_kl_max_nats"]),
        energy_tolerance=float(frozen["epsilon"]["bfloat16_field_energy_relative_error_max"]),
    )
    assert_parameter_versions(model, versions)
    gates = {
        **frozen["gates"],
        "usable_values_required_per_direction": frozen["epsilon"][
            "usable_values_required_per_direction"
        ],
    }
    verdict = adjudicate_scalar_operator(cells, gates)
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "COMPLETE_SEALED_SCALAR_OPERATOR_CONFIRMATION",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "stage": "pg2_full_prefix_scalar_operator_confirmation",
        "confirmation_metadata_rows_parsed": len(panel),
        "confirmation_unique_items_evaluated": len(panel),
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
            "model_dtype": "bfloat16",
            "operator_accumulation_dtype": "float32",
            "attention": "eager",
            "use_cache": False,
        },
        "panel": {"items": len(panel)},
        "directions": {
            "count": len(directions),
            "tensor_sha256": [_tensor_sha(direction) for direction in directions],
        },
        "epsilon_ladder": frozen["epsilon"]["ladder"],
        "identity": identity,
        "parameter_versions": {"status": "GREEN", "count": len(versions)},
        "cells": cells,
        "execution": execution,
        "verdict": verdict,
        "claim_boundary": frozen["claim_boundary"],
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
    del model, processor, sensitive, directions
    release_cuda()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
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
        arm_freeze=args.arm_freeze,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
