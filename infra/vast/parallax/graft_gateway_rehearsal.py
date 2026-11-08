"""No-science one-host rehearsal for the shared-SigLIP graft mechanics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .g1_common import release_cuda, resolve_snapshot, utc_now
from .graft_common import assert_vision_state_compatible
from .graft_discovery import (
    _anchor_logits,
    _assert_fixed_versions,
    _cpu_state,
    _entity_map,
    _fixed_interface_versions,
    _load_vision,
    _public_parent_state,
    _score_candidates,
    _tensor_sha256,
)


def run(*, matrix: Path, cache_root: Path, task_root: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor, PaliGemmaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("gateway graft rehearsal requires CUDA")
    entities = _entity_map(matrix)
    snapshots = {
        lineage: resolve_snapshot(cache_root, entities[(lineage, "fused")].repo, entities[(lineage, "fused")].revision)
        for lineage in ("pg1_pi", "pg2")
    }
    parent = entities[("pg1_pi", "vision_parent_candidate")]
    parent_snapshot = resolve_snapshot(cache_root, parent.repo, parent.revision)
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    torch.use_deterministic_algorithms(True)

    # Load only one 3B model at a time so this wiring test fits a 12 GB gateway.
    supporting = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshots["pg2"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    cross_state = _cpu_state(supporting.model.vision_tower)
    del supporting
    release_cuda()

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshots["pg1_pi"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(snapshots["pg1_pi"], local_files_only=True)
    native_state = _cpu_state(model.model.vision_tower)
    parent_model = AutoModel.from_pretrained(parent_snapshot, local_files_only=True, dtype=torch.bfloat16).to("cpu").eval()
    parent_state = _public_parent_state(parent_model, native_state)
    del parent_model
    contracts = {
        "native": assert_vision_state_compatible(native_state, native_state),
        "public_parent": assert_vision_state_compatible(native_state, parent_state),
        "cross_lineage": assert_vision_state_compatible(native_state, cross_state),
    }
    rows = [
        {
            "item_id": "mechanical-anchor",
            "task_id": "mechanical-anchor",
            "prompt": "<image>PARALLAX mechanical graft anchor. Answer with one token.",
            "candidate": candidate,
            "target": "alpha",
            "image": "real.png",
        }
        for candidate in ("alpha", "beta")
    ]
    fixed = _fixed_interface_versions(model)
    before = _anchor_logits(model, processor, rows[0], task_root)
    _load_vision(model, native_state)
    _assert_fixed_versions(model, fixed)
    after = _anchor_logits(model, processor, rows[0], task_root)
    if not torch.equal(before, after):
        raise RuntimeError("gateway native reconstruction was not exact")

    cells: dict[str, Any] = {}
    for name, state, shuffled in (
        ("native", native_state, False),
        ("public_parent", parent_state, False),
        ("cross_lineage", cross_state, False),
        ("shuffled_features", native_state, True),
    ):
        _load_vision(model, state)
        _assert_fixed_versions(model, fixed)
        scores, hook = _score_candidates(
            model=model,
            processor=processor,
            rows=rows,
            task_root=task_root,
            batch_size=2,
            shuffle_features=shuffled,
        )
        cells[name] = {
            "status": "GREEN",
            "candidate_count": len(scores),
            "suffix_token_counts": [row["suffix_tokens"] for row in scores],
            "finite": all(
                torch.isfinite(torch.tensor(row["log_likelihood_sum"]))
                and torch.isfinite(torch.tensor(row["log_likelihood_mean"]))
                for row in scores
            ),
            "hook": hook,
        }
    _load_vision(model, native_state)
    _assert_fixed_versions(model, fixed)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_NO_SCIENCE_REHEARSAL",
        "provider": "vast.ai",
        "mode": "MECHANICAL_ANCHOR_ONLY_NO_DISCOVERY_OR_CONFIRMATION_ITEMS",
        "matrix_sha256": sha256_file(matrix),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "identity_reconstruction": {
            "status": "GREEN",
            "exact": True,
            "anchor_sha256": _tensor_sha256(before.float()),
        },
        "fixed_interface": {"status": "GREEN", "parameters": len(fixed)},
        "vision_state_contracts": contracts,
        "cells": cells,
        "discovery_items_read": 0,
        "confirmation_items_read": 0,
    }
    del model, processor, native_state, parent_state, cross_state
    release_cuda()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite gateway rehearsal: {args.output}")
    result = run(matrix=args.matrix, cache_root=args.cache_root, task_root=args.task_root)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
