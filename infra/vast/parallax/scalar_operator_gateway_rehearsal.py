"""No-science GPU rehearsal of the PG2 scalar JVP/Fisher/VJP mechanics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_model import (
    assert_parameter_versions,
    decoder_layer,
    fixed_parameter_versions,
    identity_intervention,
    processor_batch,
    to_device,
)
from .g1_common import assert_attention_eager, release_cuda, resolve_snapshot, utc_now
from .localization_gateway_rehearsal import _configs, _mechanical_image, _pg2_entity
from .scalar_operator import deterministic_directions
from .scalar_operator_model import evaluate_scalar_operator


def run(*, matrix: Path, config_path: Path, cache_root: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("scalar operator gateway rehearsal requires CUDA")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    torch.manual_seed(20260722)
    torch.cuda.manual_seed_all(20260722)
    torch.use_deterministic_algorithms(True)
    entity = _pg2_entity(matrix)
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    with tempfile.TemporaryDirectory(prefix="parallax-scalar-operator-mechanical-") as temporary:
        task_root = Path(temporary)
        image_path = task_root / "mechanical.png"
        _mechanical_image(image_path)
        rows = [
            {
                "item_id": "scalar-operator-mechanical-0",
                "target": "alpha",
                "prompt": "Which marker is requested? Answer with exactly one word.",
                "image": image_path.name,
            }
        ]
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda").eval()
        model.config.use_cache = False
        attention = assert_attention_eager(_configs(model))
        layer = decoder_layer(model, int(config["site"]["layer_index"]))
        versions = fixed_parameter_versions(model)
        inputs = to_device(processor_batch(processor, rows, task_root))
        identity = identity_intervention(model, inputs, layer)
        del inputs
        width = int(model.config.text_config.hidden_size)
        sensitive = torch.arange(1, width + 1, dtype=torch.float32)
        directions = deterministic_directions(
            sensitive,
            count=2,
            namespace="parallax-scalar-operator-gateway-no-science",
        )
        cells, execution = evaluate_scalar_operator(
            model=model,
            processor=processor,
            rows=rows,
            task_root=task_root,
            layer=layer,
            directions=directions,
            epsilons=[float(value) for value in config["epsilon"]["ladder"]],
            namespace="parallax-scalar-operator-gateway-no-science",
            batch_size=1,
            expected_image_tokens=int(config["site"]["expected_image_tokens_per_item"]),
            per_token_fraction_max=float(config["epsilon"]["per_token_fraction_max"]),
            kl_min=float(config["epsilon"]["symmetric_kl_min_nats"]),
            kl_max=float(config["epsilon"]["symmetric_kl_max_nats"]),
            energy_tolerance=float(
                config["epsilon"]["bfloat16_field_energy_relative_error_max"]
            ),
        )
        assert_parameter_versions(model, versions)
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN_NO_SCIENCE_SCALAR_OPERATOR_REHEARSAL",
            "provider": "vast.ai",
            "mode": "MECHANICAL_IMAGE_ARTIFICIAL_DIRECTIONS_NO_TASK_BANK",
            "matrix_sha256": sha256_file(matrix),
            "config_sha256": sha256_file(config_path),
            "checkpoint": {"repo": entity.repo, "revision": entity.revision},
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "attention": attention,
            "identity": identity,
            "parameter_versions": {"status": "GREEN", "count": len(versions)},
            "mechanical_directions": 2,
            "epsilon_ladder": config["epsilon"]["ladder"],
            "cells": cells,
            "execution": execution,
            "operator_panel_metadata_rows_parsed": 0,
            "operator_panel_items_evaluated": 0,
        }
        del model, processor, sensitive, directions, cells
        release_cuda()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scalar operator rehearsal: {args.output}")
    result = run(matrix=args.matrix, config_path=args.config, cache_root=args.cache_root)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
