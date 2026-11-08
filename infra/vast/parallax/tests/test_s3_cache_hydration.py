from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from infra.vast.parallax.s3_cache_hydration import (
    HydrationError, REMOTE_HYDRATE, validate_spec,
)


def hydration_spec() -> dict:
    names = [
        ("added_tokens.json", 24), ("config.json", 1027),
        ("generation_config.json", 137),
        ("model-00001-of-00003.safetensors", 4_953_412_480),
        ("model-00002-of-00003.safetensors", 4_999_820_608),
        ("model-00003-of-00003.safetensors", 1_740_714_288),
        ("model.safetensors.index.json", 62_552),
        ("preprocessor_config.json", 699), ("special_tokens_map.json", 607),
        ("tokenizer.json", 17_549_604), ("tokenizer.model", 4_264_023),
        ("tokenizer_config.json", 39_968),
    ]
    rows = []
    for index, (name, size) in enumerate(names):
        digest = f"{index + 1:064x}"
        rows.append({
            "path": name, "bytes": size, "sha256": digest,
            "bucket": "pg-lineage-769063704696", "region": "us-east-1",
            "key": f"parallax/input-staging/sha256/{digest}/{name}",
            "version_id": f"v{index}",
        })
    run_id = "parallax-pg1-h100-s3-hydration-v1"
    return {
        "schema_version": 1, "status": "FROZEN",
        "scope": "s3_worker_cache_hydration_only", "run_id": run_id,
        "files": rows, "expected_total_bytes": 11_715_866_017,
        "parallel_streams": 3, "runtime_timeout_seconds": 600,
        "worker_snapshot_root": "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/rev",
        "confirmation_items_read": 0, "science_items_evaluated": 0,
        "model_loads": 0, "gpu_work": 0,
        "runtime": {"job": {"execution_site": "gateway", "environment": {"CUDA_VISIBLE_DEVICES": ""},
            "command": [f"/tmp/parallax-lifecycle-{run_id}/s3-get-urls.secret.json"]}},
    }


def test_hydration_binds_exact_full_pg1_snapshot():
    assert validate_spec(hydration_spec())["parallel_streams"] == 3


@pytest.mark.parametrize("field,value", [
    ("parallel_streams", 4), ("expected_total_bytes", 1),
    ("gpu_work", 1), ("runtime_timeout_seconds", 601),
])
def test_hydration_rejects_scope_expansion(field: str, value: object):
    payload = copy.deepcopy(hydration_spec())
    payload[field] = value
    with pytest.raises(HydrationError):
        validate_spec(payload)


def test_remote_hydration_atomically_publishes_only_after_all_hashes_match(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payload = []
    for index, row in enumerate(hydration_spec()["files"]):
        body = f"tiny-hydration-fixture-{index}".encode()
        path = source / f"source-{index}"
        path.write_bytes(body)
        payload.append({
            "path": row["path"], "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "version_id": row["version_id"], "url": path.as_uri(),
        })
    destination = tmp_path / "cache" / "snapshots" / "revision"
    completed = subprocess.run(
        [sys.executable, "-c", REMOTE_HYDRATE, str(destination), "3"],
        input=json.dumps(payload), text=True, capture_output=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "GREEN_HYDRATED"
    assert result["file_count"] == 12
    assert destination.is_dir()
    assert not destination.with_name(destination.name + ".partial").exists()
    assert all((destination / row["path"]).is_file() for row in payload)
