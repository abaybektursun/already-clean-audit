from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from infra.vast.parallax.task_cache_job import (
    TaskCacheError,
    _semantic_items_sha256,
    materialize,
    validate_task_cache_spec,
)
from infra.vast.parallax import guarded_launch


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_paths = []
    for index in range(4):
        path = tmp_path / f"source-{index}.txt"
        path.write_text(f"source-{index}\n")
        source_paths.append(path)
    final = tmp_path / "cache" / "bank"
    staging = tmp_path / "cache" / ".bank.materializing"
    manifests = final / "manifests"
    manifests.mkdir(parents=True)
    items = manifests / "items.jsonl"
    items.write_text("{}\n")
    content = manifests / "content.json"
    content.write_text(
        json.dumps(
            {
                "status": "GREEN_MATERIALIZED",
                "total_items": 4096,
                "total_images": 2560,
                "total_image_bytes": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )
    audit = manifests / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "GREEN",
                "errors": [],
                "cross_split_image_hashes": 0,
                "cross_split_source_identities": 0,
                "cross_split_nonce_or_interface_semantics": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    spec = {
        "schema_version": 1,
        "status": "FROZEN",
        "provider": "vast.ai",
        "scope": "parallax_task_cache_materialization",
        "run_id": "unit-task-cache",
        "volume_root": str(tmp_path),
        "final_root": str(final),
        "staging_root": str(staging),
        "receipt_path": str(tmp_path / "receipts" / "receipt.json"),
        "progress_path": str(tmp_path / "receipts" / "progress.json"),
        "hf_cache_root": str(tmp_path / "hf"),
        "data_config": str(source_paths[0]),
        "identity_data_config": str(source_paths[0]),
        "task_config": str(source_paths[1]),
        "materialize_timeout_seconds": 10,
        "audit_timeout_seconds": 10,
        "maximum_payload_bytes": 50_000_000,
        "source_bindings": [
            {"path": str(path), "sha256": _digest(path)} for path in source_paths
        ],
        "dataset_metadata": [
            {"repo": "owner/beans", "revision": "a" * 40},
            {"repo": "owner/fashion", "revision": "b" * 40},
        ],
        "expected": {
            "content_sha256": _digest(content),
            "audit_sha256": _digest(audit),
            "items_jsonl_sha256": _digest(items),
            "total_items": 4096,
            "total_images": 2560,
        },
    }
    return spec, final


def test_existing_exact_task_cache_is_reused_without_network_or_commands(tmp_path: Path):
    spec, final = _fixture(tmp_path)
    result = materialize(spec)
    assert result["status"] == "GREEN"
    assert result["reused_existing"] is True
    assert result["commands"] == []
    assert result["verification"]["failed"] == []
    assert result["final_root"] == str(final)
    assert result["gpu_work"] is False
    assert result["model_reads"] == 0


def test_existing_task_cache_hash_drift_fails_closed(tmp_path: Path):
    spec, final = _fixture(tmp_path)
    (final / "manifests/items.jsonl").write_text("drift\n")
    with pytest.raises(TaskCacheError, match="differs from the frozen exact manifests"):
        materialize(spec)


def test_task_cache_spec_rejects_escape_and_wrong_cardinality(tmp_path: Path):
    spec, _ = _fixture(tmp_path)
    spec["final_root"] = "/escape"
    with pytest.raises(TaskCacheError, match="escape"):
        validate_task_cache_spec(spec)
    spec, _ = _fixture(tmp_path / "second")
    spec["expected"]["total_items"] = 4095
    with pytest.raises(TaskCacheError, match="cardinalities"):
        validate_task_cache_spec(spec)


def test_task_cache_spec_accepts_only_exact_frozen_encoder_overlay(tmp_path: Path):
    spec, _ = _fixture(tmp_path)
    spec["python_overlay"] = {
        "root": "/tmp/parallax-pillow-12.1.1",
        "packages": ["pillow==12.1.1"],
        "expected_pillow_version": "12.1.1",
    }
    validate_task_cache_spec(spec)
    spec["python_overlay"]["packages"] = ["pillow==12.3.0"]
    with pytest.raises(TaskCacheError, match="frozen encoder"):
        validate_task_cache_spec(spec)


def test_semantic_items_hash_excludes_only_serialized_image_properties(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    base = {"item_id": "x", "image": "images/x.png", "target": "yes"}
    first.write_text(json.dumps({**base, "image_sha256": "a", "image_bytes": 10}) + "\n")
    second.write_text(json.dumps({**base, "image_sha256": "b", "image_bytes": 20}) + "\n")
    assert _semantic_items_sha256(first) == _semantic_items_sha256(second)


def test_task_gateway_label_matches_exact_local_reaper_identity():
    source = Path(guarded_launch.__file__).read_text()
    start = source.index("def materialize_persistent_task_cache")
    end = source.index("def main()", start)
    body = source[start:end]
    assert '"--label", f"parallax-guard-{run_id}"' in body
    assert '"--label", f"parallax-task-cache-{run_id}"' not in body


def test_v2_data_overlay_changes_only_runtime_repository_resolution():
    root = Path(__file__).resolve().parents[4]
    config_root = root / "studies/lineage/cross-lineage/config/data"
    v1 = yaml.safe_load((config_root / "parallax_truth_sprint_data_v1.yaml").read_text())
    v2 = yaml.safe_load((config_root / "parallax_truth_sprint_data_v2.yaml").read_text())
    assert v2["sources"]["hf_beans_pinned"]["repo"] == "beans"
    assert v2["sources"]["hf_beans_pinned"]["load_repo"] == "AI-Lab-Makerere/beans"
    assert v2["sources"]["hf_beans_pinned"]["revision"] == v1["sources"]["hf_beans_pinned"]["revision"]
    for source in v2["sources"].values():
        source.pop("load_repo", None)
    for key in ("identity_config", "compatibility_boundary"):
        v2.pop(key, None)
    v2["status"] = v1["status"]
    v2["authored"] = v1["authored"]
    assert v2 == v1
