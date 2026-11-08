"""Run the core-runtime PARALLAX G1 rail on one H100, one model at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import (
    G1GateError,
    assert_attention_eager,
    assert_no_meta_or_nonfinite,
    blank_image,
    identity_replay,
    image_contract,
    load_real_image,
    offline_gate,
    release_cuda,
    resolve_snapshot,
    tensor_fingerprint,
    utc_now,
)
from .parallax_entities import frozen_entities


def _entity_map(matrix: Path, lineage: str) -> dict[str, Any]:
    rows = {row.role: row for row in frozen_entities(matrix) if row.lineage == lineage}
    expected = {"vision_parent_candidate", "language_parent", "fused"}
    if lineage == "pg2":
        expected.add("later_0")
    if not expected.issubset(rows):
        raise G1GateError(f"incomplete entity set for {lineage}")
    return rows


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _model_configs(model: Any) -> list[Any]:
    configs = []
    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
    ):
        if candidate is not None and hasattr(candidate, "_attn_implementation"):
            configs.append(candidate)
    return configs


def _run_siglip(snapshot: Path, images: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    load = assert_no_meta_or_nonfinite(model)
    attention = assert_attention_eager(_model_configs(model))
    cases = {}
    for name, image in images.items():
        inputs = _to_device(processor(images=image, return_tensors="pt"), device)
        pixels = inputs["pixel_values"]
        target = model.vision_model.encoder.layers[0]

        def forward():
            with torch.inference_mode():
                return model.vision_model(pixel_values=pixels, return_dict=True)

        anchor, identity = identity_replay(forward, lambda out: out.last_hidden_state[:, :4, :], {"vision_l0": target})
        cases[name] = {
            "input": image_contract(pixels),
            "patch_tokens": int(
                (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
            ),
            "anchor": tensor_fingerprint(anchor),
            "identity": identity,
        }
    result = {
        "repo_type": "standalone_siglip_parent_candidate",
        "snapshot": str(snapshot),
        "load": load,
        "attention": attention,
        "use_cache": False,
        "cases": cases,
    }
    del model, processor, inputs, pixels, anchor
    release_cuda()
    return result


def _run_language(snapshot: Path, prompt: str, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    load = assert_no_meta_or_nonfinite(model)
    attention = assert_attention_eager(_model_configs(model))
    inputs = _to_device(tokenizer(prompt, return_tensors="pt"), device)
    target = model.model.layers[0]

    def forward():
        with torch.inference_mode():
            return model(**inputs, use_cache=False, return_dict=True)

    anchor, identity = identity_replay(forward, lambda out: out.logits[:, -1, :256], {"language_l0": target})
    result = {
        "repo_type": "standalone_language_parent",
        "snapshot": str(snapshot),
        "load": load,
        "attention": attention,
        "use_cache": False,
        "prompt_sha256": sha256_file_bytes(prompt.encode("utf-8")),
        "token_count": int(inputs["attention_mask"].sum().item()),
        "token_ids_sha256": tensor_fingerprint(inputs["input_ids"])["sample_sha256_float32"],
        "anchor": tensor_fingerprint(anchor),
        "identity": identity,
    }
    del model, tokenizer, inputs, anchor
    release_cuda()
    return result


def sha256_file_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _pg_targets(model: Any) -> dict[str, Any]:
    core = model.model
    vision = core.vision_tower.vision_model.encoder.layers[0]
    language = core.language_model.layers[0]
    return {"vision_l0": vision, "language_l0": language}


def _run_paligemma(snapshot: Path, images: dict[str, Any], prompt: str, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    load = assert_no_meta_or_nonfinite(model)
    attention = assert_attention_eager(_model_configs(model))
    cases = {}
    for name, image in images.items():
        inputs = _to_device(processor(images=image, text=prompt, return_tensors="pt"), device)
        image_tokens = int((inputs["input_ids"] == model.config.image_token_index).sum().item())

        def forward():
            with torch.inference_mode():
                return model(**inputs, use_cache=False, return_dict=True)

        anchor, identity = identity_replay(
            forward, lambda out: out.logits[:, -1, :256], _pg_targets(model)
        )
        cases[name] = {
            "input": image_contract(inputs["pixel_values"], image_token_count=image_tokens),
            "token_count": int(inputs["attention_mask"].sum().item()),
            "token_ids_sha256": tensor_fingerprint(inputs["input_ids"])["sample_sha256_float32"],
            "anchor": tensor_fingerprint(anchor),
            "identity": identity,
        }
    result = {
        "repo_type": "paligemma_fused",
        "snapshot": str(snapshot),
        "load": load,
        "attention": attention,
        "use_cache": False,
        "prompt_sha256": sha256_file_bytes(prompt.encode("utf-8")),
        "image_token_index": int(model.config.image_token_index),
        "vision_patch_size": int(model.config.vision_config.patch_size),
        "vision_num_image_tokens": int(model.config.vision_config.num_image_tokens),
        "language_layers": int(model.config.text_config.num_hidden_layers),
        "language_width": int(model.config.text_config.hidden_size),
        "cases": cases,
    }
    del model, processor, inputs, anchor
    release_cuda()
    return result


def run(
    *,
    matrix: Path,
    anchors: Path,
    cache_root: Path,
    task_root: Path,
    lineage: str,
    instance_id: int | None,
) -> dict[str, Any]:
    import torch

    if lineage not in {"pg1_pi", "pg2"}:
        raise G1GateError("core rail must be pg1_pi or pg2")
    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    if not torch.cuda.is_available():
        raise G1GateError("G1 core runner requires CUDA")
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.use_deterministic_algorithms(True)
    config = yaml.safe_load(anchors.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise G1GateError("G1 anchors are not frozen")
    image_row = config["images"]["real"]
    images = {
        "blank": blank_image(config["images"]["blank_size"]),
        "real": load_real_image(task_root / image_row["relative_path"], image_row["sha256"]),
    }
    entities = _entity_map(matrix, lineage)
    device = "cuda"
    outputs: dict[str, Any] = {}
    vision = entities["vision_parent_candidate"]
    outputs["siglip"] = _run_siglip(resolve_snapshot(cache_root, vision.repo, vision.revision), images, device)
    language = entities["language_parent"]
    outputs["language_parent"] = _run_language(
        resolve_snapshot(cache_root, language.repo, language.revision), config["prompts"]["language_anchor"], device
    )
    fused = entities["fused"]
    outputs["fused"] = _run_paligemma(
        resolve_snapshot(cache_root, fused.repo, fused.revision), images, config["prompts"]["paligemma"], device
    )
    if lineage == "pg2":
        later = entities["later_0"]
        outputs["mix"] = _run_paligemma(
            resolve_snapshot(cache_root, later.repo, later.revision), images, config["prompts"]["paligemma"], device
        )
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "lineage": lineage,
        "runtime_role": "core",
        "anchor_config_sha256": sha256_file(anchors),
        "cache_gate_manifest_sha256": gate.get("content_manifest_sha256"),
        "comparison_scope": "WITHIN_LINEAGE_PLUS_REGISTERED_SHARED_SIGLIP_ONLY",
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--lineage", choices=["pg1_pi", "pg2"], required=True)
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        matrix=args.matrix,
        anchors=args.anchors,
        cache_root=args.cache_root,
        task_root=args.task_root,
        lineage=args.lineage,
        instance_id=args.instance_id,
    )
    atomic_json(args.output, packet)
    print(json.dumps({"status": packet["status"], "lineage": packet["lineage"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
