"""Environment and numerical smoke test for the immutable PARALLAX image."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .artifacts import atomic_json


def expected_versions(lock_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            continue
        if " @ git+" in line:
            package, source = line.split(" @ ", 1)
            revision = source.rsplit("@", 1)[-1]
            if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
                raise ValueError(f"git lock entry lacks exact commit: {line}")
            continue
        if "==" not in line:
            raise ValueError(f"non-exact lock entry: {line}")
        package, version = line.split("==", 1)
        expected[package] = version
    return expected


def run(lock_path: Path, *, require_h100: bool) -> dict[str, object]:
    import numpy as np
    import torch
    import transformers
    from transformers import (
        Gemma2Config,
        GemmaConfig,
        PaliGemmaConfig,
        PaliGemmaForConditionalGeneration,
        SiglipVisionConfig,
    )

    expected = expected_versions(lock_path)
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for package, wanted in expected.items():
        got = importlib.metadata.version(package)
        observed[package] = got
        if got.split("+", 1)[0] != wanted:
            mismatches[package] = {"expected": wanted, "observed": got}
    if mismatches:
        raise AssertionError(f"version mismatches: {mismatches}")

    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=14,
        patch_size=14,
    )
    pg1 = PaliGemmaConfig(
        vision_config=vision.to_dict(),
        text_config=GemmaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        ).to_dict(),
        projection_dim=16,
        hidden_size=16,
    )
    pg2 = PaliGemmaConfig(
        vision_config=vision.to_dict(),
        text_config=Gemma2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        ).to_dict(),
        projection_dim=16,
        hidden_size=16,
    )
    assert pg1.text_config.model_type == "gemma"
    assert pg2.text_config.model_type == "gemma2"
    assert PaliGemmaForConditionalGeneration is not None

    # Weight-free operator fixture: a symmetric bilinear form has exact central
    # differences and equal x^TQy / y^TQx to floating-point tolerance.
    generator = torch.Generator(device="cpu").manual_seed(17)
    matrix = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    q = matrix.T @ matrix
    x = torch.randn(8, generator=generator, dtype=torch.float64)
    y = torch.randn(8, generator=generator, dtype=torch.float64)
    symmetry_error = abs(float(x @ q @ y - y @ q @ x))
    assert symmetry_error < 1e-10

    cuda: dict[str, object] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "runtime": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        product = a @ a.T
        assert torch.isfinite(product).all()
    if require_h100:
        assert torch.cuda.is_available(), "CUDA is unavailable"
        assert "H100" in str(cuda.get("name", "")).upper(), cuda
        assert int(cuda.get("total_memory_bytes", 0)) >= 79 * 1024**3, cuda
        assert cuda.get("bf16_supported") is True, cuda

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": observed,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "model_contracts": {
            "paligemma_gemma1_config": "GREEN",
            "paligemma_gemma2_config": "GREEN",
        },
        "operator_fixture": {"symmetry_abs_error": symmetry_error},
        "cuda": cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-h100", action="store_true")
    args = parser.parse_args()
    report = run(args.lock, require_h100=args.require_h100)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
