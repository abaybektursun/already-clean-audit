"""Exact, pre-mapped tensor ancestry assay for the two PARALLAX rails."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_json


PI_ROOT = "paligemma_with_expert.paligemma.model."


def siglip_to_paligemma(key: str) -> str | None:
    # The public SigLIP repository also contains a text tower.  It is not a
    # candidate parent of PaliGemma and must never enter the denominator.
    return f"vision_tower.{key}" if key.startswith("vision_model.") else None


def gemma_to_paligemma(key: str) -> str | None:
    return f"language_model.{key}" if key.startswith("model.") else None


def paligemma_to_pi(key: str) -> str | None:
    if key.startswith("language_model.model."):
        suffix = key.removeprefix("language_model.model.")
        return f"{PI_ROOT}language_model.{suffix}"
    if key.startswith(("vision_tower.", "multi_modal_projector.")):
        return f"{PI_ROOT}{key}"
    return None


def identity(key: str) -> str:
    return key


def canonical_storage_equal(left: Any, right: Any) -> tuple[bool, str]:
    """Test equality after only the unavoidable lower-precision storage cast."""
    import torch

    if left.dtype == right.dtype:
        return torch.equal(left, right), str(left.dtype)
    floating = {left.dtype, right.dtype}
    if torch.bfloat16 in floating:
        canonical = torch.bfloat16
    elif torch.float16 in floating:
        canonical = torch.float16
    else:
        canonical = torch.float32
    return torch.equal(left.to(canonical), right.to(canonical)), str(canonical)


def displacement_sums(left: Any, right: Any, *, chunk_elements: int = 4_000_000) -> dict[str, float]:
    """Accumulate fp64 summary statistics without materializing a full fp32 copy."""
    import torch

    if left.shape != right.shape:
        raise ValueError("displacement tensors must have equal shapes")
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    totals = {"difference_sq": 0.0, "parent_sq": 0.0, "child_sq": 0.0, "dot": 0.0}
    for start in range(0, flat_left.numel(), chunk_elements):
        stop = min(start + chunk_elements, flat_left.numel())
        parent = flat_left[start:stop].float()
        child = flat_right[start:stop].float()
        difference = child - parent
        totals["difference_sq"] += float(torch.sum(difference * difference, dtype=torch.float64))
        totals["parent_sq"] += float(torch.sum(parent * parent, dtype=torch.float64))
        totals["child_sq"] += float(torch.sum(child * child, dtype=torch.float64))
        totals["dot"] += float(torch.sum(parent * child, dtype=torch.float64))
    return totals


def displacement_metrics(sums: dict[str, float]) -> dict[str, float]:
    parent_norm = math.sqrt(sums["parent_sq"])
    child_norm = math.sqrt(sums["child_sq"])
    difference_norm = math.sqrt(sums["difference_sq"])
    return {
        "relative_l2_to_parent": difference_norm / max(parent_norm, 1e-30),
        "cosine": sums["dot"] / max(parent_norm * child_norm, 1e-30),
    }


def triadic_displacement_sums(
    parent: Any, first: Any, second: Any, *, chunk_elements: int = 4_000_000
) -> dict[str, float | int]:
    import torch

    if parent.shape != first.shape or parent.shape != second.shape:
        raise ValueError("triadic displacement tensors must have equal shapes")
    vectors = [tensor.reshape(-1) for tensor in (parent, first, second)]
    totals: dict[str, float | int] = {
        "first_sq": 0.0,
        "second_sq": 0.0,
        "dot": 0.0,
        "nonzero_union": 0,
        "same_sign": 0,
    }
    for start in range(0, vectors[0].numel(), chunk_elements):
        stop = min(start + chunk_elements, vectors[0].numel())
        base, left, right = [vector[start:stop].float() for vector in vectors]
        first_delta = left - base
        second_delta = right - base
        totals["first_sq"] += float(torch.sum(first_delta * first_delta, dtype=torch.float64))
        totals["second_sq"] += float(torch.sum(second_delta * second_delta, dtype=torch.float64))
        totals["dot"] += float(torch.sum(first_delta * second_delta, dtype=torch.float64))
        nonzero = (first_delta != 0) | (second_delta != 0)
        totals["nonzero_union"] += int(nonzero.sum())
        totals["same_sign"] += int(((torch.sign(first_delta) == torch.sign(second_delta)) & nonzero).sum())
    return totals


def triadic_displacement_metrics(sums: dict[str, float | int]) -> dict[str, float]:
    first_norm = math.sqrt(float(sums["first_sq"]))
    second_norm = math.sqrt(float(sums["second_sq"]))
    return {
        "displacement_cosine": float(sums["dot"]) / max(first_norm * second_norm, 1e-30),
        "pg2_to_pg1_displacement_norm_ratio": second_norm / max(first_norm, 1e-30),
        "sign_agreement_nonzero_union": float(sums["same_sign"]) / max(int(sums["nonzero_union"]), 1),
    }


class TensorSet:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.stack = ExitStack()
        self.handles: list[Any] = []
        self.locations: dict[str, int] = {}

    def __enter__(self) -> "TensorSet":
        from safetensors import safe_open

        for path in sorted(self.snapshot.glob("*.safetensors")):
            handle = self.stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            index = len(self.handles)
            self.handles.append(handle)
            for key in handle.keys():
                if key in self.locations:
                    raise ValueError(f"duplicate tensor key: {key}")
                self.locations[key] = index
        if not self.handles:
            raise FileNotFoundError(f"no safetensors in {self.snapshot}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def tensor(self, key: str):
        return self.handles[self.locations[key]].get_tensor(key)


def compare_edge(
    parent_snapshot: Path,
    child_snapshot: Path,
    mapper: Callable[[str], str | None],
) -> dict[str, Any]:
    import torch

    rows: list[dict[str, Any]] = []
    with TensorSet(parent_snapshot) as parent, TensorSet(child_snapshot) as child:
        candidates = [(key, mapper(key)) for key in sorted(parent.locations)]
        candidates = [(key, mapped) for key, mapped in candidates if mapped is not None]
        for parent_key, child_key in candidates:
            assert child_key is not None
            if child_key not in child.locations:
                rows.append({"parent_key": parent_key, "child_key": child_key, "verdict": "MISSING"})
                continue
            left = parent.tensor(parent_key)
            right = child.tensor(child_key)
            element_count = math.prod(left.shape)
            row = {
                "parent_key": parent_key,
                "child_key": child_key,
                "parent_shape": list(left.shape),
                "child_shape": list(right.shape),
                "parent_dtype": str(left.dtype),
                "child_dtype": str(right.dtype),
                "elements": element_count,
            }
            if left.shape == right.shape:
                aligned_right = right
                if torch.equal(left, right):
                    row["verdict"] = "EXACT"
                else:
                    equal, canonical = canonical_storage_equal(left, right)
                    row["canonical_comparison_dtype"] = canonical
                    row["verdict"] = "EXACT_AFTER_STORAGE_CAST" if equal else "CHANGED"
            elif (
                left.ndim >= 1
                and right.ndim == left.ndim
                and right.shape[0] > left.shape[0]
                and right.shape[1:] == left.shape[1:]
            ):
                prefix = right[: left.shape[0]]
                aligned_right = prefix
                if torch.equal(left, prefix):
                    row["verdict"] = "EXACT_PREFIX_EXPANSION"
                else:
                    equal, canonical = canonical_storage_equal(left, prefix)
                    row["canonical_comparison_dtype"] = canonical
                    row["verdict"] = (
                        "EXACT_PREFIX_AFTER_STORAGE_CAST" if equal else "CHANGED_PREFIX_EXPANSION"
                    )
            else:
                row["verdict"] = "SHAPE_MISMATCH"
                aligned_right = None
            if aligned_right is not None:
                sums = displacement_sums(left, aligned_right)
                row["displacement_sums"] = sums
                row["displacement"] = displacement_metrics(sums)
            rows.append(row)

    counts: dict[str, int] = {}
    elements: dict[str, int] = {}
    for row in rows:
        verdict = row["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
        elements[verdict] = elements.get(verdict, 0) + int(row.get("elements", 0))
    global_sums = {"difference_sq": 0.0, "parent_sq": 0.0, "child_sq": 0.0, "dot": 0.0}
    for row in rows:
        for name, value in row.get("displacement_sums", {}).items():
            global_sums[name] += float(value)
    return {
        "mapped_candidate_count": len(rows),
        "verdict_counts": counts,
        "element_counts": elements,
        "global_displacement_sums": global_sums,
        "global_displacement": displacement_metrics(global_sums),
        "rows": rows,
    }


def compare_shared_siglip_displacements(
    parent_snapshot: Path, pg1_snapshot: Path, pg2_snapshot: Path
) -> dict[str, Any]:
    totals: dict[str, float | int] = {
        "first_sq": 0.0,
        "second_sq": 0.0,
        "dot": 0.0,
        "nonzero_union": 0,
        "same_sign": 0,
    }
    compared = 0
    missing: list[str] = []
    with (
        TensorSet(parent_snapshot) as parent,
        TensorSet(pg1_snapshot) as pg1,
        TensorSet(pg2_snapshot) as pg2,
    ):
        for parent_key in sorted(parent.locations):
            child_key = siglip_to_paligemma(parent_key)
            if child_key is None:
                continue
            if child_key not in pg1.locations or child_key not in pg2.locations:
                missing.append(parent_key)
                continue
            base = parent.tensor(parent_key)
            first = pg1.tensor(child_key)
            second = pg2.tensor(child_key)
            if base.shape != first.shape or base.shape != second.shape:
                raise ValueError(f"shared SigLIP shape mismatch: {parent_key}")
            row = triadic_displacement_sums(base, first, second)
            for name, value in row.items():
                totals[name] += value
            compared += 1
    return {
        "compared_tensor_count": compared,
        "missing_parent_keys": missing,
        "sums": totals,
        "metrics": triadic_displacement_metrics(totals),
        "claim_boundary": "Descriptive common-parent displacement in registered tensor coordinates; activation and graft tests remain necessary for functional parentage.",
    }


def run(cache_manifest: Path) -> dict[str, Any]:
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cache.get("status") != "GREEN":
        raise ValueError("model cache manifest is not GREEN")
    snapshots = {row["repo"]: Path(row["snapshot"]) for row in cache["repositories"]}
    definitions = {
        "siglip_to_pg1": ("google/siglip-so400m-patch14-224", "google/paligemma-3b-pt-224", siglip_to_paligemma),
        "gemma1_to_pg1": ("google/gemma-2b", "google/paligemma-3b-pt-224", gemma_to_paligemma),
        "pg1_to_pi0": ("google/paligemma-3b-pt-224", "lerobot/pi0_base", paligemma_to_pi),
        "pg1_to_pi05": ("google/paligemma-3b-pt-224", "lerobot/pi05_base", paligemma_to_pi),
        "siglip_to_pg2": ("google/siglip-so400m-patch14-224", "google/paligemma2-3b-pt-224", siglip_to_paligemma),
        "gemma2_to_pg2": ("google/gemma-2-2b", "google/paligemma2-3b-pt-224", gemma_to_paligemma),
        "pg2_pt_to_mix": ("google/paligemma2-3b-pt-224", "google/paligemma2-3b-mix-224", identity),
    }
    edges = {
        name: {
            "parent_repo": parent,
            "child_repo": child,
            **compare_edge(snapshots[parent], snapshots[child], mapper),
        }
        for name, (parent, child, mapper) in definitions.items()
    }
    shared_siglip = compare_shared_siglip_displacements(
        snapshots["google/siglip-so400m-patch14-224"],
        snapshots["google/paligemma-3b-pt-224"],
        snapshots["google/paligemma2-3b-pt-224"],
    )
    return {
        "schema_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_INSTRUMENT_COMPLETE",
        "mode": "EXACT_CPU_TENSOR_EQUALITY_PREMAPPED_WITH_LOWER_PRECISION_STORAGE_ADJUDICATION",
        "mechanical_revision": "v1 exact-byte results preserved; v2 resolves only observed bf16/fp32 storage crossings by a frozen lower-precision cast",
        "registered_extension": "v3 adds the spec-predeclared displacement map using threshold-free fp64 accumulated relative L2 and cosine",
        "registered_triadic_extension": "v4 adds the spec-predeclared common-SigLIP displacement cosine, norm ratio, and sign agreement with no threshold",
        "shared_siglip_displacement": shared_siglip,
        "claim_boundary": "Exact equality establishes retained bytes under the registered map; changed tensors do not by themselves identify training mechanism or causal contribution.",
        "edges": edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cache_manifest)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
