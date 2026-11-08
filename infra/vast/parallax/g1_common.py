"""Shared, comparison-firewalled machinery for PARALLAX G1 anchors.

The module is intentionally network-free.  Model runners call ``offline_gate``
before importing Transformers or LeRobot and resolve only exact local snapshots.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifacts import sha256_file
from .cache_gate import CACHE_GATE_MARKER


PRIMARY = "pg1_pi"
SUPPORTING = "pg2"
LEGAL_SHARED_OBJECTS = frozenset({"siglip_pixels", "siglip_patch_geometry", "normalized_verdict"})


class G1GateError(RuntimeError):
    """Raised before model execution when a frozen G1 contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def offline_gate(cache_root: Path, *, expected_instance_id: int | None = None) -> dict[str, Any]:
    """Prove that the H100 cache gate exists, then force all model APIs offline."""
    marker = cache_root / CACHE_GATE_MARKER
    if not marker.is_file():
        raise G1GateError("G1 requires the green H100 cache gate")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "GREEN":
        raise G1GateError("H100 cache gate is not green")
    if expected_instance_id is not None and payload.get("destination_instance_id") != expected_instance_id:
        raise G1GateError("cache gate is not bound to this H100 instance")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return payload


def resolve_snapshot(cache_root: Path, repo: str, revision: str) -> Path:
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise G1GateError(f"revision is not a full SHA: {repo}")
    snapshot = cache_root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"exact local snapshot unavailable: {repo}@{revision}")
    return snapshot


def load_real_image(path: Path, expected_sha256: str):
    from PIL import Image

    if sha256_file(path) != expected_sha256:
        raise G1GateError("real-image anchor hash mismatch")
    return Image.open(path).convert("RGB")


def blank_image(size: int = 224):
    from PIL import Image

    return Image.new("RGB", (size, size), color=(0, 0, 0))


def tensor_fingerprint(tensor: Any, *, sample_values: int = 4096) -> dict[str, Any]:
    """Small stable summary; the hash never serializes a full large activation."""
    import torch

    value = tensor.detach()
    if value.is_sparse:
        value = value.to_dense()
    flat = value.reshape(-1)
    finite = bool(torch.isfinite(flat).all().item()) if flat.is_floating_point() else True
    sample = flat[:sample_values].to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
    numeric = flat.to(dtype=torch.float32)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device_type": value.device.type,
        "elements": value.numel(),
        "finite": finite,
        "mean": float(numeric.mean().item()) if value.numel() else None,
        "std": float(numeric.std(unbiased=False).item()) if value.numel() else None,
        "l2": float(torch.linalg.vector_norm(numeric).item()) if value.numel() else 0.0,
        "sample_values": min(sample_values, value.numel()),
        "sample_sha256_float32": digest,
    }


def tensor_difference(left: Any, right: Any) -> dict[str, Any]:
    import torch

    if left.shape != right.shape:
        return {"shape_equal": False, "exact": False, "max_abs": None}
    delta = left.detach().to(dtype=torch.float32) - right.detach().to(dtype=torch.float32)
    return {
        "shape_equal": True,
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
    }


@contextmanager
def observational_hooks(targets: Mapping[str, Any]):
    """Attach hooks that observe but return ``None``, preserving module outputs."""
    counts = {name: 0 for name in targets}
    handles = []
    for name, module in targets.items():
        def hook(_module, _inputs, _output, *, _name=name):
            counts[_name] += 1
            return None

        handles.append(module.register_forward_hook(hook))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def identity_replay(
    forward: Callable[[], Any],
    select: Callable[[Any], Any],
    targets: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Run no-hook and observational-hook forwards and require bit identity."""
    baseline = select(forward()).detach()
    with observational_hooks(targets) as counts:
        replay = select(forward()).detach()
    difference = tensor_difference(baseline, replay)
    if not all(count > 0 for count in counts.values()):
        raise G1GateError(f"identity hook did not fire: {counts}")
    if not difference["exact"] or difference["max_abs"] != 0.0:
        raise G1GateError(f"observational hook changed output: {difference}")
    return baseline, {"status": "GREEN", "hook_counts": counts, **difference}


def assert_no_meta_or_nonfinite(model: Any) -> dict[str, Any]:
    import torch

    parameter_count = 0
    first = None
    for parameter in model.parameters():
        if parameter.is_meta:
            raise G1GateError("model contains unresolved meta parameters")
        parameter_count += parameter.numel()
        if first is None and parameter.numel():
            first = parameter.detach().reshape(-1)[:1024]
    if first is None or not torch.isfinite(first.to(dtype=torch.float32)).all():
        raise G1GateError("model has no finite parameter sentinel")
    return {"parameters": parameter_count, "sentinel": tensor_fingerprint(first)}


def assert_attention_eager(configs: Iterable[Any]) -> list[str]:
    observed = []
    for config in configs:
        implementation = getattr(config, "_attn_implementation", None)
        observed.append(str(implementation))
        if implementation != "eager":
            raise G1GateError(f"attention backend is not eager: {implementation}")
    return observed


def assert_comparison_legal(left_lineage: str, right_lineage: str, object_kind: str) -> None:
    if left_lineage == right_lineage:
        return
    if object_kind not in LEGAL_SHARED_OBJECTS:
        raise G1GateError(
            f"cross-Gemma comparison forbidden: {left_lineage} vs {right_lineage} object={object_kind}"
        )


def image_contract(image_tensor: Any, *, image_token_count: int | None = None) -> dict[str, Any]:
    result = {"pixels": tensor_fingerprint(image_tensor)}
    if image_token_count is not None:
        result["image_token_count"] = int(image_token_count)
    return result


def release_cuda(*objects: Any) -> None:
    import gc
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

