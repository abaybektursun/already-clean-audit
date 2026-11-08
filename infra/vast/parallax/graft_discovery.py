"""Run the minimum zero-shot shared-SigLIP graft discovery on one H100."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .audit_task_bank import audit as audit_task_bank
from .g1_common import assert_attention_eager, offline_gate, release_cuda, resolve_snapshot, utc_now
from .graft_common import (
    GraftGateError,
    assert_vision_state_compatible,
    candidate_rows,
    canonical_sha256,
    discovery_range,
    item_scores,
    retained_fraction,
    select_discovery_panel,
    shuffled_patch_features,
    suffix_log_likelihoods,
    summarize_items,
    wrong_image_map,
)
from .parallax_entities import frozen_entities
from .parallax_tasks import FrozenTaskBank


def _entity_map(matrix: Path) -> dict[str, Any]:
    rows = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    required = {
        ("pg1_pi", "vision_parent_candidate"),
        ("pg1_pi", "fused"),
        ("pg2", "fused"),
    }
    if not required.issubset(rows):
        raise GraftGateError("shared graft entity set is incomplete")
    if rows[("pg1_pi", "vision_parent_candidate")].repo != rows[("pg2", "vision_parent_candidate")].repo:
        raise GraftGateError("lineages do not name the same public SigLIP candidate")
    return rows


def _attention_configs(model: Any) -> list[Any]:
    return [
        config
        for config in (model.config, model.config.text_config, model.config.vision_config)
        if hasattr(config, "_attn_implementation")
    ]


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().to(device="cpu", dtype=None).contiguous()
    return hashlib.sha256(contiguous.view(-1).numpy().tobytes()).hexdigest()


def _processor_pixels(processor: Any, images: list[Any]) -> list[str]:
    hashes = []
    for image in images:
        pixels = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        hashes.append(_tensor_sha256(pixels.float()))
    return hashes


def _fixed_interface_versions(model: Any) -> dict[str, tuple[int, int, int]]:
    return {
        name: (id(parameter), int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in model.named_parameters()
        if "vision_tower" not in name
    }


def _assert_fixed_versions(model: Any, expected: dict[str, tuple[int, int, int]]) -> None:
    observed = _fixed_interface_versions(model)
    if observed != expected:
        changed = sorted(key for key in set(expected) | set(observed) if expected.get(key) != observed.get(key))
        raise GraftGateError(f"fixed interface parameter changed during vision graft: {changed[:8]}")


def _cpu_state(module: Any) -> dict[str, Any]:
    return {key: value.detach().to("cpu").clone() for key, value in module.state_dict().items()}


def _public_parent_state(parent_model: Any, host_state: dict[str, Any]) -> dict[str, Any]:
    full = parent_model.state_dict()
    filtered = {key: full[key].detach().to("cpu").clone() for key in host_state if key in full}
    assert_vision_state_compatible(host_state, filtered)
    return filtered


def _load_vision(model: Any, state: dict[str, Any]) -> dict[str, Any]:
    host = model.model.vision_tower.state_dict()
    contract = assert_vision_state_compatible(host, state)
    incompatible = model.model.vision_tower.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise GraftGateError("strict vision graft returned incompatible keys")
    return contract


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _score_candidates(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    task_root: Path,
    batch_size: int,
    image_override: dict[str, str] | None = None,
    shuffle_features: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from PIL import Image

    image_cache: dict[str, Any] = {}

    def load(relative: str):
        if relative not in image_cache:
            with Image.open(task_root / relative) as image:
                image_cache[relative] = image.convert("RGB").copy()
        return image_cache[relative]

    output: list[dict[str, Any]] = []
    hook_context = shuffled_patch_features(model.model.multi_modal_projector) if shuffle_features else contextlib.nullcontext(None)
    with hook_context as hook_counts:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            images = [
                load(image_override.get(row["item_id"], row["image"]) if image_override else row["image"])
                for row in batch_rows
            ]
            inputs = processor(
                images=images,
                text=[row["prompt"] for row in batch_rows],
                suffix=[row["candidate"] for row in batch_rows],
                padding=True,
                return_tensors="pt",
            )
            required = {"input_ids", "attention_mask", "pixel_values", "labels", "token_type_ids"}
            if not required.issubset(inputs):
                raise GraftGateError(f"PaliGemma suffix processor contract missing: {sorted(required - set(inputs))}")
            device_inputs = _to_device(dict(inputs), "cuda")
            with torch.inference_mode():
                model_output = model(**device_inputs, use_cache=False, return_dict=True)
            sums, means, counts = suffix_log_likelihoods(model_output.logits, device_inputs["labels"])
            for row, total, mean, count in zip(batch_rows, sums, means, counts):
                if not math.isfinite(total) or not math.isfinite(mean):
                    raise GraftGateError("nonfinite candidate likelihood")
                output.append(
                    {
                        "item_id": row["item_id"],
                        "task_id": row["task_id"],
                        "candidate": row["candidate"],
                        "target": row["target"],
                        "log_likelihood_sum": total,
                        "log_likelihood_mean": mean,
                        "suffix_tokens": int(count),
                    }
                )
            del inputs, device_inputs, model_output
    if shuffle_features:
        expected_calls = math.ceil(len(rows) / batch_size)
        if hook_counts != {"calls": expected_calls, "tokens": 256}:
            raise GraftGateError(f"shuffled feature hook mismatch: {hook_counts} expected_calls={expected_calls}")
    return output, hook_counts or {"calls": 0, "tokens": None}


def _anchor_logits(model: Any, processor: Any, row: dict[str, str], task_root: Path) -> Any:
    import torch
    from PIL import Image

    with Image.open(task_root / row["image"]) as image:
        inputs = processor(
            images=image.convert("RGB"), text=row["prompt"], suffix=row["candidate"], return_tensors="pt"
        )
    device_inputs = _to_device(dict(inputs), "cuda")
    with torch.inference_mode():
        output = model(**device_inputs, use_cache=False, return_dict=True)
    labels = device_inputs["labels"][:, 1:]
    positions = labels.ne(-100).nonzero(as_tuple=False)
    if not len(positions):
        raise GraftGateError("identity anchor has no suffix tokens")
    first = int(positions[0, 1])
    result = output.logits[:, first : first + 1, :256].detach().cpu()
    del inputs, device_inputs, output
    return result


def _validate_prerequisites(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    prereq = config["prerequisites"]
    hour6_path = root / prereq["hour6_packet"]
    g0_path = root / prereq["g0_packet"]
    hour6 = json.loads(hour6_path.read_text(encoding="utf-8"))
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    if hour6.get("hour_6_decision") != prereq["required_hour6_decision"] or hour6.get("status") != "GREEN":
        raise GraftGateError("hour-6 packet does not license graft discovery")
    if g0.get("status") != prereq["required_g0_status"]:
        raise GraftGateError("G0 does not license the coordinate-compatible shared vision bridge")
    if hour6.get("lineage_claims", {}).get("direct_cross_gemma_language_comparison_performed") is not False:
        raise GraftGateError("hour-6 comparison firewall is not green")
    return {
        "hour6": {"path": str(hour6_path), "sha256": sha256_file(hour6_path)},
        "g0": {"path": str(g0_path), "sha256": sha256_file(g0_path)},
    }


def run(
    *,
    root: Path,
    matrix: Path,
    config_path: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int | None,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor, PaliGemmaForConditionalGeneration

    if not torch.cuda.is_available():
        raise GraftGateError("graft discovery requires CUDA")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_discovery":
        raise GraftGateError("graft discovery config is not frozen")
    prerequisite_bindings = _validate_prerequisites(root, config)
    task_audit = audit_task_bank(task_root)
    if task_audit.get("status") != "GREEN":
        raise GraftGateError(f"task bank audit failed: {task_audit.get('errors')}")
    content_path = task_root / "manifests/content.json"
    task_bank = FrozenTaskBank(task_root / "manifests/items.jsonl", task_root / "manifests/confirmation.sha256")
    panel = select_discovery_panel(task_bank.iter_discovery(), config["panel"]["per_target"])
    if len(panel) != int(config["panel"]["expected_items"]):
        raise GraftGateError(f"panel size mismatch: {len(panel)}")
    expanded = candidate_rows(panel)
    if config["scoring"].get("image_marker") != "explicit_paligemma_image_token":
        raise GraftGateError("PaliGemma image-marker grammar is not frozen")
    for row in expanded:
        if row["prompt"].startswith("<image>"):
            raise GraftGateError("task prompt unexpectedly contains the processor image marker")
        row["prompt"] = "<image>" + row["prompt"]
    wrong_map = wrong_image_map(panel)
    entities = _entity_map(matrix)

    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    torch.use_deterministic_algorithms(True)
    fused_snapshots = {
        "pg1_pi": resolve_snapshot(
            cache_root, entities[("pg1_pi", "fused")].repo, entities[("pg1_pi", "fused")].revision
        ),
        "pg2": resolve_snapshot(cache_root, entities[("pg2", "fused")].repo, entities[("pg2", "fused")].revision),
    }
    parent = entities[("pg1_pi", "vision_parent_candidate")]
    parent_snapshot = resolve_snapshot(cache_root, parent.repo, parent.revision)
    processors = {
        lineage: AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        for lineage, snapshot in fused_snapshots.items()
    }
    models = {
        lineage: PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda").eval()
        for lineage, snapshot in fused_snapshots.items()
    }
    for model in models.values():
        assert_attention_eager(_attention_configs(model))
        model.config.use_cache = False

    from PIL import Image

    probe_images = []
    for row in panel:
        with Image.open(task_root / row["image"]) as image:
            probe_images.append(image.convert("RGB").copy())
    pixels = {lineage: _processor_pixels(processor, probe_images) for lineage, processor in processors.items()}
    if pixels["pg1_pi"] != pixels["pg2"]:
        raise GraftGateError("selected-panel pixels differ across host processors")

    native_states = {lineage: _cpu_state(model.model.vision_tower) for lineage, model in models.items()}
    assert_vision_state_compatible(native_states["pg1_pi"], native_states["pg2"])
    parent_model = AutoModel.from_pretrained(parent_snapshot, local_files_only=True, dtype=torch.bfloat16).to("cpu").eval()
    parent_state = _public_parent_state(parent_model, native_states["pg1_pi"])
    del parent_model

    batch_size = int(config["scoring"]["batch_size"])
    host_results: dict[str, Any] = {}
    for lineage in ("pg1_pi", "pg2"):
        model, processor = models[lineage], processors[lineage]
        fixed_versions = _fixed_interface_versions(model)
        native_state = native_states[lineage]
        cross_state = native_states["pg2" if lineage == "pg1_pi" else "pg1_pi"]
        contracts = {
            "native": assert_vision_state_compatible(native_state, native_state),
            "public_parent": assert_vision_state_compatible(native_state, parent_state),
            "cross_lineage": assert_vision_state_compatible(native_state, cross_state),
        }
        anchor_before = _anchor_logits(model, processor, expanded[0], task_root)
        _load_vision(model, native_state)
        _assert_fixed_versions(model, fixed_versions)
        anchor_after = _anchor_logits(model, processor, expanded[0], task_root)
        identity_exact = bool(torch.equal(anchor_before, anchor_after))
        if not identity_exact:
            raise GraftGateError(f"native identity reconstruction changed logits for {lineage}")

        cell_items: dict[str, list[dict[str, Any]]] = {}
        hook_audits: dict[str, Any] = {}
        for cell, state, shuffle in (
            ("native", native_state, False),
            ("public_parent", parent_state, False),
            ("cross_lineage", cross_state, False),
            ("shuffled_features", native_state, True),
        ):
            _load_vision(model, state)
            _assert_fixed_versions(model, fixed_versions)
            candidate_output, hook = _score_candidates(
                model=model,
                processor=processor,
                rows=expanded,
                task_root=task_root,
                batch_size=batch_size,
                shuffle_features=shuffle,
            )
            cell_items[cell] = item_scores(candidate_output)
            hook_audits[cell] = hook
        _load_vision(model, native_state)
        _assert_fixed_versions(model, fixed_versions)
        wrong_candidates, wrong_hook = _score_candidates(
            model=model,
            processor=processor,
            rows=expanded,
            task_root=task_root,
            batch_size=batch_size,
            image_override=wrong_map,
        )
        cell_items["wrong_image_native"] = item_scores(wrong_candidates)
        hook_audits["wrong_image_native"] = wrong_hook
        range_result = discovery_range(
            cell_items["native"],
            cell_items["shuffled_features"],
            cell_items["wrong_image_native"],
            minimum_raw_effect=float(config["range_gate"]["minimum_raw_effect"]),
            seed=20260716 + (0 if lineage == "pg1_pi" else 1000),
        )
        host_results[lineage] = {
            "status": range_result["status"],
            "role": "primary" if lineage == "pg1_pi" else "supporting",
            "identity_reconstruction": {
                "status": "GREEN",
                "exact": identity_exact,
                "anchor_sha256": _tensor_sha256(anchor_before.float()),
            },
            "vision_state_contracts": contracts,
            "fixed_interface_parameters": {"status": "GREEN", "count": len(fixed_versions)},
            "hook_audits": hook_audits,
            "range": range_result,
            "summaries": {cell: summarize_items(items) for cell, items in cell_items.items()},
            "retained_fraction": {
                donor: retained_fraction(
                    cell_items[donor],
                    cell_items["native"],
                    cell_items["shuffled_features"],
                    cell_items["wrong_image_native"],
                    range_result["green_tasks"],
                )
                for donor in ("public_parent", "cross_lineage")
            },
            "items": cell_items,
        }
        _load_vision(model, native_state)

    status = (
        "GREEN_DISCOVERY_RANGE"
        if any(result["range"]["green_tasks"] for result in host_results.values())
        else "VOID_NO_NATIVE_RANGE"
    )
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": status,
        "provider": "vast.ai",
        "instance_id": instance_id,
        "stage": "shared_siglip_zero_shot_graft_discovery",
        "claim_boundary": "Zero-shot graft discovery separates functional range from no-range. A failed cross-graft remains remappable-or-coadapted until connector-only repair.",
        "lineage_firewall": {
            "primary": "pg1_pi",
            "supporting": "pg2",
            "single_chain_across_both": False,
            "direct_cross_gemma_language_comparison_performed": False,
            "cross_lineage_numeric_comparison": "within-host normalized retained fractions only",
        },
        "bindings": {
            **prerequisite_bindings,
            "config_sha256": sha256_file(config_path),
            "matrix_sha256": sha256_file(matrix),
            "task_content_sha256": sha256_file(content_path),
            "cache_manifest_sha256": gate.get("content_manifest_sha256"),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "attention": "eager",
            "dtype": "bfloat16",
            "use_cache": False,
        },
        "panel": {
            "items": len(panel),
            "candidate_sequences": len(expanded),
            "item_ids_sha256": canonical_sha256([row["item_id"] for row in panel]),
            "wrong_image_map_sha256": canonical_sha256(wrong_map),
            "processor_pixels_sha256": canonical_sha256(pixels["pg1_pi"]),
            "confirmation_items_returned": 0,
        },
        "hosts": host_results,
    }
    del models, processors, native_states, parent_state
    release_cuda()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_name("manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite graft discovery output: {args.output}")
    packet = run(
        root=args.root,
        matrix=args.matrix,
        config_path=args.config,
        cache_root=args.cache_root,
        task_root=args.task_root,
        instance_id=args.instance_id,
    )
    atomic_json(args.output, packet)
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN_CONTENT_HASHED",
            "stage": packet["stage"],
            "packet": {
                "path": args.output.name,
                "bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
            },
        },
    )
    print(
        json.dumps(
            {"status": packet["status"], "output": str(args.output), "manifest": str(manifest_path)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
