from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest

from infra.vast.parallax.s3_object_mirror import (
    MirrorError,
    validate_mirror_spec,
    validate_upload_url,
)


DIGEST = "e" * 64
ROOT = Path(__file__).resolve().parents[4]


def spec() -> dict:
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": "pg1_s3_object_mirror_probe_only",
        "provider": "vast.ai",
        "object_store": "amazon_s3",
        "aws_profile_control_plane_only": "fuelos",
        "region": "us-east-1",
        "run_id": "unit-mirror",
        "source": {
            "path": "/volume/parallax/cache/model/tokenizer.json",
            "bytes": 17_549_604,
            "sha256": DIGEST,
        },
        "destination": {
            "bucket": "pg-lineage-769063704696",
            "key": f"parallax/input-staging/sha256/{DIGEST}/tokenizer.json",
            "server_side_encryption": "AES256",
            "checksum_sha256_base64": base64.b64encode(bytes.fromhex(DIGEST)).decode(),
        },
        "durable_result": "/volume/parallax/object-mirror/unit/result.json",
        "require_destination_absent_before_upload": True,
        "upload_timeout_seconds": 60,
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "model_loads": 0,
        "gpu_work": 0,
    }


def test_mirror_spec_binds_real_cache_content_address_and_zero_science():
    assert validate_mirror_spec(spec())["run_id"] == "unit-mirror"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "path"), "/root/model/tokenizer.json"),
        (("source", "bytes"), 25_000_001),
        (("destination", "key"), "parallax/input-staging/wrong"),
        (("destination", "bucket"), "wrong"),
        (("science_items_evaluated",), 1),
        (("gpu_work",), 1),
    ],
)
def test_mirror_spec_rejects_scope_expansion(path: tuple[str, ...], value: object):
    payload = spec()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(MirrorError):
        validate_mirror_spec(payload)


def test_upload_url_is_exact_and_signature_bearing():
    payload = spec()
    host = "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
    headers = "host%3Bx-amz-checksum-sha256%3Bx-amz-meta-sha256%3Bx-amz-server-side-encryption"
    url = (
        f"https://{host}/{payload['destination']['key']}?"
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=900&"
        f"X-Amz-SignedHeaders={headers}&X-Amz-Signature={'a' * 64}"
    )
    assert validate_upload_url(url, payload)[0] == host
    with pytest.raises(MirrorError):
        validate_upload_url(url.replace("tokenizer.json", "wrong.json"), payload)


def test_uploader_is_a_standalone_direct_entrypoint_without_package_imports():
    script = ROOT / "infra/vast/parallax/s3_object_mirror.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "--url-file" in completed.stdout
