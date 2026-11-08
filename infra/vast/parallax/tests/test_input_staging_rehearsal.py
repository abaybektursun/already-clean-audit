from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from infra.vast.parallax.input_staging_rehearsal import (
    StagingError,
    prepare_task_subset,
    validate_staging_job_spec,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job_spec() -> dict:
    sizes = [1] * 11 + [11_715_866_006]
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": "pg1_input_staging_rehearsal_only",
        "run_id": "parallax-input-staging-v1",
        "uses_gpu": False,
        "scientific_claim": False,
        "model_reads": 0,
        "science_data_reads": 0,
        "runtime": {
            "job": {
                "execution_site": "gateway",
                "environment": {"CUDA_VISIBLE_DEVICES": ""},
                "command": ["python3", "-m", "infra.vast.parallax.input_staging_rehearsal"],
            },
            "artifact_roots": [
                {"remote_root": "/tmp/parallax-input-staging-v1", "durable_prefix": "input-staging"}
            ],
            "required_sentinels": ["input-staging/result.json"],
        },
        "staging": {
            "task_source_root": "/volume/parallax/task-cache/parallax_tasks_v1",
            "task_subset_root": "/volume/parallax/training-inputs/pg1-optimizer-scale-v2",
            "model_source_root": "/volume/parallax/cache/pg1",
            "worker_root": "/root/parallax-inputs-v1",
            "expected_task": {
                "content_sha256": "a" * 64,
                "audit_sha256": "b" * 64,
                "items_jsonl_sha256": "c" * 64,
            },
            "panel": {
                "task_id": "vr_a_beans_balanced",
                "train_per_target": 32,
                "discovery_per_target": 8,
            },
            "model": {
                "repo": "google/paligemma-3b-pt-224",
                "revision": "d" * 40,
                "total_bytes": sum(sizes),
                "files": [
                    {"path": f"file-{index}", "bytes": size, "sha256": f"{index:064x}"}
                    for index, size in enumerate(sizes)
                ],
            },
            "minimum_worker_free_bytes": 15_000_000_000,
            "maximum_transfer_bytes": 12_000_000_000,
            "transfer_timeout_seconds": 2400,
        },
    }


def test_staging_job_contract_is_exact_and_rejects_confirmation_or_unsafe_paths():
    spec = _job_spec()
    validate_staging_job_spec(spec)
    spec["staging"]["task_subset_root"] = "/tmp/escape"
    with pytest.raises(StagingError, match="escapes PARALLAX volume"):
        validate_staging_job_spec(spec)


def test_prepare_task_subset_physically_excludes_confirmation(tmp_path: Path):
    source = tmp_path / "volume/parallax/task-cache/parallax_tasks_v1"
    manifests = source / "manifests"
    manifests.mkdir(parents=True)
    rows = []
    for split, per_target in (("train", 32), ("discovery_eval", 8)):
        for target in ("angular_leaf_spot", "bean_rust", "healthy"):
            for index in range(per_target):
                item_id = hashlib.sha256(f"{split}:{target}:{index}".encode()).hexdigest()
                relative = f"images/vr_a_beans_balanced/{split}/{item_id}.png"
                image = source / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(item_id.encode())
                rows.append({
                    "item_id": item_id,
                    "task_id": "vr_a_beans_balanced",
                    "split": split,
                    "target": target,
                    "image": relative,
                    "image_bytes": image.stat().st_size,
                    "image_sha256": _sha(image),
                })
    for split, total in (("train", 2048), ("discovery_eval", 1024), ("confirmation_eval", 1024)):
        present = sum(row["split"] == split for row in rows)
        for index in range(total - present):
            rows.append({
                "item_id": hashlib.sha256(f"dummy:{split}:{index}".encode()).hexdigest(),
                "task_id": "dummy",
                "split": split,
                "target": "dummy",
                "image": None,
            })
    rows.sort(key=lambda row: (row["task_id"], row["split"], row["item_id"]))
    items = manifests / "items.jsonl"
    items.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    content = manifests / "content.json"
    content.write_text("{}\n")
    audit = manifests / "audit.json"
    audit.write_text(json.dumps({"status": "GREEN", "errors": []}) + "\n")
    final = tmp_path / "volume/parallax/training-inputs/pg1-optimizer-scale-v2"
    spec = {
        "staging": {
            "task_source_root": str(source),
            "task_subset_root": str(final),
            "expected_task": {
                "content_sha256": _sha(content),
                "audit_sha256": _sha(audit),
                "items_jsonl_sha256": _sha(items),
            },
            "panel": {
                "task_id": "vr_a_beans_balanced",
                "train_per_target": 32,
                "discovery_per_target": 8,
            },
        }
    }
    packet = prepare_task_subset(spec)
    assert packet["selected_image_count"] == 120
    assert packet["confirmation_capable_artifacts_present"] is False
    assert len(list(final.rglob("*.png"))) == 120
    assert not (final / "manifests/items.jsonl").exists()
    assert not any("confirmation_eval" in path.as_posix() for path in final.rglob("*"))
    seal = json.loads((final / "manifests/discovery-seal.json").read_text())
    assert seal["confirmation_values_emitted"] == 0
    assert seal["discovery_rows"] == 3072
