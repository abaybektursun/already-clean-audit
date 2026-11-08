"""No-science GPU rehearsal of every rescue arm at the fixed 0.001 amplitude."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .behavioral_localization_model import evaluate_localization
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


def run(*, matrix: Path, cache_root: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("localization rescue gateway rehearsal requires CUDA")
    torch.manual_seed(20260721)
    torch.cuda.manual_seed_all(20260721)
    torch.use_deterministic_algorithms(True)
    entity = _pg2_entity(matrix)
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    with tempfile.TemporaryDirectory(prefix="parallax-localization-rescue-mechanical-") as temporary:
        task_root = Path(temporary)
        image_path = task_root / "mechanical.png"
        _mechanical_image(image_path)
        rows = [
            {
                "item_id": f"localization-rescue-mechanical-{index}",
                "target": target,
                "prompt": "Which marker is requested? Answer with exactly one word.",
                "image": image_path.name,
            }
            for index, target in enumerate(("alpha", "beta"))
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
        layer = decoder_layer(model, 0)
        versions = fixed_parameter_versions(model)
        inputs = to_device(processor_batch(processor, rows, task_root))
        identity = identity_intervention(model, inputs, layer)
        del inputs
        width = int(model.config.text_config.hidden_size)
        direction = torch.arange(1, width + 1, dtype=torch.float32)
        direction = direction / direction.norm()
        metrics, execution = evaluate_localization(
            model=model,
            processor=processor,
            rows=rows,
            task_root=task_root,
            layer=layer,
            sensitive_direction=direction,
            namespace="parallax-localization-rescue-gateway-no-science",
            amplitudes=[0.001],
            arms=[
                "full_sensitive",
                "full_rotated",
                "full_white",
                "image_sensitive",
                "image_rotated",
                "prompt_sensitive",
                "prompt_rotated",
            ],
            batch_size=2,
            expected_image_tokens=256,
            per_token_fraction_max=0.005,
            total_energy_tolerance=0.005,
        )
        assert_parameter_versions(model, versions)
        all_finite = all(
            all(torch.isfinite(torch.tensor(row[key])) for key in ("kl_nats", "damage_nats"))
            for row in metrics
        )
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN_NO_SCIENCE_LOCALIZATION_RESCUE_REHEARSAL",
            "provider": "vast.ai",
            "mode": "MECHANICAL_IMAGE_FIXED_DIRECTION_NO_TASK_BANK",
            "matrix_sha256": sha256_file(matrix),
            "checkpoint": {"repo": entity.repo, "revision": entity.revision},
            "fixed_amplitude": 0.001,
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "attention": attention,
            "identity": identity,
            "parameter_versions": {"status": "GREEN", "count": len(versions)},
            "mechanical_rows": len(metrics),
            "all_metrics_finite": all_finite,
            "execution": execution,
            "discovery_items_read": 0,
            "confirmation_metadata_rows_parsed": 0,
            "confirmation_items_evaluated": 0,
        }
        del model, processor, direction, metrics
        release_cuda()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite localization rescue rehearsal: {args.output}")
    result = run(matrix=args.matrix, cache_root=args.cache_root)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
