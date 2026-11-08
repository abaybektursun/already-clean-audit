"""No-science rehearsal of paired behavioral intervention mechanics on a cheap GPU."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .behavioral_model import (
    assert_parameter_versions,
    decoder_layer,
    evaluate_constructions,
    fixed_parameter_versions,
    identity_intervention,
    processor_batch,
    to_device,
)
from .g1_common import assert_attention_eager, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities


def _fused_entities(matrix: Path) -> dict[str, Any]:
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    return {lineage: entities[(lineage, "fused")] for lineage in ("pg1_pi", "pg2")}


def _configs(model: Any) -> list[Any]:
    return [
        value
        for value in (model.config, model.config.text_config, model.config.vision_config)
        if hasattr(value, "_attn_implementation")
    ]


def _mechanical_image(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (224, 224))
    pixels = image.load()
    for y in range(224):
        for x in range(224):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256)
    image.save(path, format="PNG", optimize=False)


def run(*, matrix: Path, cache_root: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("behavioral gateway rehearsal requires CUDA")
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    torch.use_deterministic_algorithms(True)
    entities = _fused_entities(matrix)
    lineages: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="parallax-behavioral-mechanical-") as temporary:
        task_root = Path(temporary)
        image_path = task_root / "mechanical.png"
        _mechanical_image(image_path)
        rows = [
            {
                "item_id": f"mechanical-{index}",
                "target": target,
                "prompt": "Which marker is requested? Answer with exactly one word.",
                "image": image_path.name,
            }
            for index, target in enumerate(("alpha", "beta"))
        ]
        # Models are deliberately loaded one at a time so this wiring rehearsal
        # stays on a cheap 12 GB gateway rather than consuming H100 time.
        for lineage in ("pg1_pi", "pg2"):
            entity = entities[lineage]
            snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
            processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                snapshot,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).to("cuda").eval()
            model.config.use_cache = False
            attention = assert_attention_eager(_configs(model))
            layer = decoder_layer(model, 0)
            versions = fixed_parameter_versions(model)
            inputs = to_device(processor_batch(processor, rows, task_root))
            identity = identity_intervention(model, inputs, layer)
            del inputs
            width = int(model.config.text_config.hidden_size)
            direction = torch.arange(1, width + 1, dtype=torch.float32)
            direction = direction / direction.norm()
            metrics, execution = evaluate_constructions(
                model=model,
                processor=processor,
                rows=rows,
                task_root=task_root,
                layer=layer,
                sensitive_direction=direction,
                lineage=lineage,
                namespace="parallax-behavioral-gateway-no-science",
                amplitudes=[0.003],
                constructions=["sensitive_coherent", "rotated_coherent", "white"],
                batch_size=2,
            )
            assert_parameter_versions(model, versions)
            lineages[lineage] = {
                "status": "GREEN",
                "checkpoint": {"repo": entity.repo, "revision": entity.revision},
                "attention": attention,
                "identity": identity,
                "parameter_versions": {"status": "GREEN", "count": len(versions)},
                "mechanical_rows": len(metrics),
                "all_metrics_finite": all(
                    all(torch.isfinite(torch.tensor(row[key])) for key in ("kl_nats", "damage_nats"))
                    for row in metrics
                ),
                "execution": execution,
            }
            del model, processor, direction, metrics
            release_cuda()
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_NO_SCIENCE_REHEARSAL",
        "provider": "vast.ai",
        "mode": "MECHANICAL_IMAGE_AND_FIXED_DIRECTION_ONLY",
        "matrix_sha256": sha256_file(matrix),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "lineages": lineages,
        "discovery_items_read": 0,
        "confirmation_items_read": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite behavioral gateway rehearsal: {args.output}")
    result = run(matrix=args.matrix, cache_root=args.cache_root)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
