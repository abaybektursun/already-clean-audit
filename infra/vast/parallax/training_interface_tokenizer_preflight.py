"""Processor-only confound gate for the PARALLAX A/B/C interface control."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import utc_now
from .training_common import TrainingGateError
from .training_interface_control import CODEBOOKS, CODES, SOURCE_TASK_ID, remap_rows
from .training_interface_control import validate_codebooks
from .training_model import select_split_stratified


def _image_token_id(processor: Any) -> int:
    value = getattr(processor, "image_token_id", None)
    if value is None:
        value = processor.tokenizer.convert_tokens_to_ids("<image>")
    if not isinstance(value, int) or value < 0:
        raise TrainingGateError("processor does not expose a valid image token ID")
    return value


def audit_processor(
    *, processor: Any, source_rows: Sequence[Mapping[str, Any]], task_root: Path
) -> dict[str, Any]:
    """Exercise the exact processor path without loading or evaluating a model."""
    import torch
    from PIL import Image

    validate_codebooks()
    anchors = select_split_stratified(
        source_rows,
        task_id=SOURCE_TASK_ID,
        split="discovery_eval",
        per_target=1,
    )
    image_token_id = _image_token_id(processor)
    observations = []
    for codebook_id in CODEBOOKS:
        rows = remap_rows(anchors, codebook_id=codebook_id)
        for row in rows:
            with Image.open(task_root / str(row["image"])) as handle:
                image = handle.convert("RGB").copy()
            for candidate in CODES:
                batch = processor(
                    images=[image],
                    text=["<image>" + str(row["prompt"])],
                    suffix=[candidate],
                    padding=True,
                    return_tensors="pt",
                )
                required = {"input_ids", "attention_mask", "pixel_values", "labels", "token_type_ids"}
                if not required.issubset(batch):
                    raise TrainingGateError(
                        f"interface processor contract missing: {sorted(required - set(batch))}"
                    )
                if batch["input_ids"].shape != batch["labels"].shape:
                    raise TrainingGateError("interface processor labels do not align")
                active = batch["labels"][:, 1:].ne(-100)
                if int(active.sum()) <= 0:
                    raise TrainingGateError("interface processor produced no causal suffix labels")
                if bool((batch["labels"].ne(-100) & batch["attention_mask"].eq(0)).any()):
                    raise TrainingGateError("interface processor labeled padding")
                suffix_ids = batch["labels"][:, 1:][active].tolist()
                attended = int(batch["attention_mask"].sum())
                image_tokens = int(batch["input_ids"].eq(image_token_id).sum())
                observations.append(
                    {
                        "codebook_id": codebook_id,
                        "item_id": str(row["item_id"]),
                        "semantic_target": str(row["semantic_target"]),
                        "candidate": candidate,
                        "suffix_tokens": len(suffix_ids),
                        "suffix_token_ids": suffix_ids,
                        "attended_tokens": attended,
                        "image_tokens": image_tokens,
                        "prefix_tokens": attended - len(suffix_ids),
                        "pixel_shape": list(batch["pixel_values"].shape),
                    }
                )
                del batch

    suffix_by_candidate: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    counts = set()
    prefix_counts = set()
    image_counts = set()
    pixel_shapes = set()
    for row in observations:
        suffix_by_candidate[row["candidate"]].add(tuple(row["suffix_token_ids"]))
        counts.add(int(row["suffix_tokens"]))
        prefix_counts.add(int(row["prefix_tokens"]))
        image_counts.add(int(row["image_tokens"]))
        pixel_shapes.add(tuple(row["pixel_shape"]))
    if set(suffix_by_candidate) != set(CODES) or any(
        len(values) != 1 for values in suffix_by_candidate.values()
    ):
        raise TrainingGateError("A/B/C suffix token IDs are incomplete or context-dependent")
    if len(counts) != 1:
        raise TrainingGateError(f"A/B/C suffix token counts are unequal: {sorted(counts)}")
    if len(prefix_counts) != 1:
        raise TrainingGateError(
            f"cyclic codebook prompts have unequal processor prefix lengths: {sorted(prefix_counts)}"
        )
    if len(image_counts) != 1 or next(iter(image_counts)) <= 0:
        raise TrainingGateError(f"image token count is invalid or unstable: {sorted(image_counts)}")
    if len(pixel_shapes) != 1:
        raise TrainingGateError("processor pixel shape changed across interface cells")
    suffix_ids = {
        candidate: list(next(iter(suffix_by_candidate[candidate]))) for candidate in CODES
    }
    if len({tuple(value) for value in suffix_ids.values()}) != len(CODES):
        raise TrainingGateError("A/B/C do not resolve to distinct suffix token sequences")
    return {
        "status": "GREEN_PROCESSOR_ONLY_INTERFACE_GATE",
        "codebooks": list(CODEBOOKS),
        "candidates": list(CODES),
        "anchor_item_ids": [str(row["item_id"]) for row in anchors],
        "candidate_suffix_tokens": {candidate: len(suffix_ids[candidate]) for candidate in CODES},
        "candidate_suffix_token_ids": suffix_ids,
        "equal_suffix_token_count": next(iter(counts)),
        "prefix_tokens": next(iter(prefix_counts)),
        "image_tokens": next(iter(image_counts)),
        "pixel_shape": list(next(iter(pixel_shapes))),
        "observations": observations,
        "processor_calls": len(observations),
        "model_loads": 0,
        "gpu_operations": 0,
        "science_items_evaluated": 0,
        "discovery_items_processor_only": len(anchors),
        "confirmation_items_read": 0,
    }


def run(
    *,
    lineage: str,
    config_path: Path,
    snapshot: Path,
    task_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite interface preflight: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("status") != "frozen_design_before_tokenizer_preflight"
        or config.get("paid_authorization") is not False
        or config.get("population", {}).get("confirmation_reads") != 0
        or config.get("codebooks", {}).get("order") != list(CODEBOOKS)
    ):
        raise TrainingGateError("interface-control design is not frozen/no-confirmation")
    allowed = {str(row["id"]): row for row in config.get("lineage_order", [])}
    if lineage not in allowed:
        raise TrainingGateError(f"interface tokenizer preflight lineage is not frozen: {lineage}")
    forbidden = [
        task_root / "manifests/items.jsonl",
        task_root / "manifests/confirmation.sha256",
    ]
    if any(path.exists() for path in forbidden):
        raise TrainingGateError("confirmation-capable artifact entered tokenizer preflight")
    from transformers import AutoProcessor
    from .parallax_tasks import DiscoveryTaskBank

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    rows = list(
        DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl").iter_rows(
            task_id=SOURCE_TASK_ID
        )
    )
    result = audit_processor(processor=processor, source_rows=rows, task_root=task_root)
    result.update(
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "snapshot": str(snapshot),
            "lineage": lineage,
            "role": allowed[lineage]["role"],
        }
    )
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        lineage=args.lineage,
        config_path=args.config.resolve(),
        snapshot=args.snapshot.resolve(),
        task_root=args.task_root.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
