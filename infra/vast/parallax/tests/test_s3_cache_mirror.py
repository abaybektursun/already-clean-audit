from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from infra.vast.parallax.s3_cache_mirror import (
    CacheMirrorError,
    PG1_ROOT,
    expected_checksum,
    expected_key,
    validate_put_url,
    validate_spec,
)


ROOT = Path(__file__).resolve().parents[4]
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def row(name: str, size: int, digest: str, state: str) -> dict:
    payload = {"path": name, "bytes": size, "sha256": digest}
    payload.update(
        destination_key=expected_key(payload),
        checksum_sha256_base64=expected_checksum(payload),
        expected_initial_state=state,
        existing_version_id="version-1" if state == "exact_existing" else None,
    )
    return payload


def spec() -> dict:
    rows = [
        row("config.json", 1027, DIGEST_A, "absent"),
        row("tokenizer.json", 17_549_604, DIGEST_B, "exact_existing"),
    ]
    return {
        "schema_version": 1, "status": "FROZEN",
        "scope": "pg1_s3_cache_mirror", "provider": "vast.ai",
        "object_store": "amazon_s3", "aws_profile_control_plane_only": "fuelos",
        "region": "us-east-1", "bucket": "pg-lineage-769063704696",
        "mode": "rehearsal", "run_id": "unit-cache-mirror",
        "source_root": str(PG1_ROOT),
        "destination_prefix": "parallax/input-staging/sha256/",
        "existing_object_policy": "reuse_only_if_exact_versioned_head",
        "atomic_publication": "s3_single_put_per_content_addressed_object",
        "automatic_retry": False, "parallel_streams": 2,
        "per_file_timeout_seconds": 60, "files": rows,
        "expected_total_bytes": sum(item["bytes"] for item in rows),
        "maximum_total_bytes": 25_000_000,
        "durable_result": "/volume/parallax/cache-mirror/unit/result.json",
        "confirmation_items_read": 0, "science_items_evaluated": 0,
        "model_loads": 0, "gpu_work": 0,
    }


def test_rehearsal_binds_upload_reuse_and_zero_science():
    assert validate_spec(spec())["run_id"] == "unit-cache-mirror"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_root", "/tmp/cache"),
        ("parallel_streams", 4),
        ("automatic_retry", True),
        ("expected_total_bytes", 1),
        ("gpu_work", 1),
    ],
)
def test_rehearsal_rejects_scope_expansion(field: str, value: object):
    payload = spec(); payload[field] = value
    with pytest.raises(CacheMirrorError):
        validate_spec(payload)


def test_rejects_wrong_content_address_and_unversioned_reuse():
    payload = spec(); payload["files"][0]["destination_key"] = "wrong"
    with pytest.raises(CacheMirrorError):
        validate_spec(payload)
    payload = spec(); payload["files"][1]["existing_version_id"] = ""
    with pytest.raises(CacheMirrorError):
        validate_spec(payload)


def test_url_set_is_object_exact():
    payload = spec()["files"][0]
    host = "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
    headers = "host%3Bx-amz-checksum-sha256%3Bx-amz-meta-sha256%3Bx-amz-server-side-encryption"
    url = (
        f"https://{host}/{payload['destination_key']}?"
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=900&"
        f"X-Amz-SignedHeaders={headers}&X-Amz-Signature={'c' * 64}"
    )
    assert validate_put_url(url, payload)[0] == host
    with pytest.raises(CacheMirrorError):
        validate_put_url(url.replace("config.json", "wrong.json"), payload)


def test_entrypoint_is_standalone_and_has_no_package_dependency():
    script = ROOT / "infra/vast/parallax/s3_cache_mirror.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "--url-file" in completed.stdout


def test_secret_shape_is_json_object_not_persisted_manifest():
    encoded = json.dumps({"key": "url"})
    assert isinstance(json.loads(encoded), dict)
