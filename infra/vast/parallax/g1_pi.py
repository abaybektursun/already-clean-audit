"""Run native π0/π0.5 G1 wrapper anchors inside the isolated π sidecar."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
    load_real_image,
    offline_gate,
    release_cuda,
    resolve_snapshot,
    tensor_fingerprint,
    utc_now,
)
from .parallax_entities import frozen_entities


def pi05_prompt(task: str, state_values: list[float]) -> str:
    """Exact zero-state instance of LeRobot's frozen PI05 prompt grammar."""
    import numpy as np

    state = np.asarray(state_values, dtype=np.float32)
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discretized = np.digitize(state, bins=bins) - 1
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    state_text = " ".join(map(str, discretized.tolist()))
    return f"Task: {cleaned}, State: {state_text};\nAction: "


def _image_tensor(image: Any):
    import numpy as np
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _strict_native_load(policy_class: Any, config: Any, snapshot: Path) -> tuple[Any, dict[str, Any]]:
    """Apply LeRobot's native key fixer but never swallow a strict load error."""
    from safetensors.torch import load_file

    weight_path = snapshot / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    policy = policy_class(config)
    original = load_file(weight_path, device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    incompatible = policy.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise G1GateError(
            f"strict π load returned incompatible keys: {incompatible.missing_keys[:3]} "
            f"{incompatible.unexpected_keys[:3]}"
        )
    source_keys = len(original)
    remapped_keys = len(remapped)
    del original, fixed, remapped
    return policy, {
        "method": "native_policy_plus_frozen_key_fixer_strict_no_exception_swallow",
        "source_keys": source_keys,
        "remapped_keys": remapped_keys,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "weights_sha256": sha256_file(weight_path),
    }


def _inner_targets(policy: Any) -> dict[str, Any]:
    paligemma = policy.model.paligemma_with_expert.paligemma.model
    return {
        "vision_l0": paligemma.vision_tower.vision_model.encoder.layers[0],
        # LeRobot's coupled PaliGemma+expert path calls compute_layer_complete,
        # which invokes layer.mlp directly and deliberately bypasses the whole
        # decoder layer's __call__. Hook an actually executed language module.
        "language_l0_mlp": paligemma.language_model.layers[0].mlp,
    }


def _source_use_cache_contract(model: Any) -> dict[str, Any]:
    source = inspect.getsource(model.forward)
    if "use_cache=False" not in source:
        raise G1GateError("native π forward does not explicitly disable KV cache")
    return {
        "use_cache": False,
        "forward_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _run_policy(
    *,
    kind: str,
    snapshot: Path,
    tokenizer_snapshot: Path,
    images: dict[str, Any],
    prompt_config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from lerobot.configs.policies import PreTrainedConfig

    if kind == "pi0":
        from lerobot.policies.pi0.configuration_pi0 import PI0Config as Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as Policy
    elif kind == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config as Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Policy
    else:
        raise G1GateError(f"unknown π policy kind: {kind}")

    # This deliberately mirrors Policy.from_pretrained. Calling the concrete
    # dataclass loader rejects the serialized `type` discriminator before it
    # can dispatch to PI0Config/PI05Config.
    config = PreTrainedConfig.from_pretrained(snapshot, local_files_only=True)
    if not isinstance(config, Config):
        raise G1GateError(
            f"native policy config dispatch mismatch: expected={Config.__name__} actual={type(config).__name__}"
        )
    config.device = "cuda"
    config.gradient_checkpointing = False
    policy, load = _strict_native_load(Policy, config, snapshot)
    policy.eval()
    load.update(assert_no_meta_or_nonfinite(policy))
    inner = policy.model.paligemma_with_expert.paligemma.model
    configs = [inner.language_model.config, inner.vision_tower.config]
    for inner_config in configs:
        inner_config._attn_implementation = "eager"
    attention = assert_attention_eager(configs)
    cache_contract = _source_use_cache_contract(policy.model)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)
    tokenizer.padding_side = "right"
    zero_state = [0.0] * int(config.max_state_dim)
    if kind == "pi0":
        prompt = str(prompt_config["pi0"])
    else:
        prompt = pi05_prompt(str(prompt_config["pi05_task"]), zero_state)
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=int(config.tokenizer_max_length),
        return_tensors="pt",
    )
    lang_tokens = tokens["input_ids"].to("cuda")
    lang_masks = tokens["attention_mask"].to(device="cuda", dtype=torch.bool)

    action_shape = (1, int(config.chunk_size), int(config.max_action_dim))
    actions = torch.zeros(action_shape, device="cuda", dtype=torch.float32)
    noise = torch.linspace(-1.0, 1.0, actions.numel(), device="cuda", dtype=torch.float32).reshape(action_shape)
    time = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)
    state = torch.zeros((1, int(config.max_state_dim)), device="cuda", dtype=torch.float32)
    cases = {}
    for case_name, image in images.items():
        raw = _image_tensor(image)
        batch = {key: raw.clone() for key in config.image_features}
        prepared_images, image_masks = policy._preprocess_images(batch)

        def forward():
            with torch.inference_mode():
                if kind == "pi0":
                    return policy.model.forward(
                        prepared_images, image_masks, lang_tokens, lang_masks, state, actions, noise, time
                    )
                return policy.model.forward(
                    prepared_images, image_masks, lang_tokens, lang_masks, actions, noise, time
                )

        loss, identity = identity_replay(forward, lambda output: output, _inner_targets(policy))
        with torch.inference_mode():
            prefix, prefix_padding, prefix_attention = policy.model.embed_prefix(
                prepared_images, image_masks, lang_tokens, lang_masks
            )
        cases[case_name] = {
            "camera_streams": len(prepared_images),
            "camera_masks_true": [int(mask.sum().item()) for mask in image_masks],
            "pixels": [tensor_fingerprint(value) for value in prepared_images],
            "prefix": tensor_fingerprint(prefix[:, :16, :]),
            "prefix_tokens": int(prefix.shape[1]),
            "prefix_padding_true": int(prefix_padding.sum().item()),
            "prefix_attention_true": int(prefix_attention.sum().item()),
            "loss": tensor_fingerprint(loss),
            "identity": identity,
        }
    result = {
        "kind": kind,
        "snapshot": str(snapshot),
        "tokenizer_snapshot": str(tokenizer_snapshot),
        "tokenizer_revision_scope": "exact_local_pg1_snapshot",
        "tokenizer_max_length": int(config.tokenizer_max_length),
        "nonpadding_tokens": int(lang_masks.sum().item()),
        "token_ids_sha256": tensor_fingerprint(lang_tokens)["sample_sha256_float32"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_grammar": "pi0_newline" if kind == "pi0" else "pi05_state_conditioned",
        "image_resolution": list(config.image_resolution),
        "language_layers": len(inner.language_model.layers),
        "language_width": int(inner.language_model.config.hidden_size),
        "load": load,
        "attention": attention,
        **cache_contract,
        "cases": cases,
    }
    del policy, tokenizer, actions, noise, time, state, lang_tokens, lang_masks
    del prepared_images, image_masks, loss, prefix, prefix_padding, prefix_attention
    release_cuda()
    return result


def run(
    *,
    matrix: Path,
    anchors: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int | None,
) -> dict[str, Any]:
    import torch

    gate = offline_gate(cache_root, expected_instance_id=instance_id)
    if not torch.cuda.is_available():
        raise G1GateError("π G1 runner requires CUDA")
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.use_deterministic_algorithms(True)
    anchor_config = yaml.safe_load(anchors.read_text(encoding="utf-8"))
    if anchor_config.get("status") != "frozen":
        raise G1GateError("G1 anchors are not frozen")
    image_row = anchor_config["images"]["real"]
    images = {
        "blank": blank_image(anchor_config["images"]["blank_size"]),
        "real": load_real_image(task_root / image_row["relative_path"], image_row["sha256"]),
    }
    rows = {row.role: row for row in frozen_entities(matrix) if row.lineage == "pg1_pi"}
    tokenizer_snapshot = resolve_snapshot(
        cache_root, rows["fused"].repo, rows["fused"].revision
    )
    outputs = {}
    for role, kind in (("later_0", "pi0"), ("later_1", "pi05")):
        entity = rows[role]
        outputs[kind] = _run_policy(
            kind=kind,
            snapshot=resolve_snapshot(cache_root, entity.repo, entity.revision),
            tokenizer_snapshot=tokenizer_snapshot,
            images=images,
            prompt_config=anchor_config["prompts"],
        )
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "lineage": "pg1_pi",
        "runtime_role": "pi_sidecar",
        "anchor_config_sha256": sha256_file(anchors),
        "cache_gate_manifest_sha256": gate.get("content_manifest_sha256"),
        "pg2_imported": False,
        "comparison_scope": "PG1_TO_PI_WRAPPER_ANCHORS_ONLY",
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(
        matrix=args.matrix,
        anchors=args.anchors,
        cache_root=args.cache_root,
        task_root=args.task_root,
        instance_id=args.instance_id,
    )
    atomic_json(args.output, packet)
    print(json.dumps({"status": packet["status"], "lineage": packet["lineage"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
