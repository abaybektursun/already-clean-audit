from __future__ import annotations

import pytest

from infra.vast.parallax.s3_download_probe import (
    DownloadProbeError, validate_download_probe_spec, validate_download_url,
)


def spec() -> dict:
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": "s3_worker_download_probe_only",
        "run_id": "unit-download",
        "source": {
            "bucket": "pg-lineage-769063704696",
            "region": "us-east-1",
            "key": "parallax/input-staging/sha256/" + "e" * 64 + "/tokenizer.json",
            "version_id": "immutable-version",
            "bytes": 17_549_604,
            "sha256": "e" * 64,
        },
        "repetitions": 48,
        "parallel_streams": 3,
        "maximum_download_bytes": 842_380_992,
        "projected_full_cache_bytes": 11_715_866_017,
        "maximum_projected_full_cache_seconds": 300,
        "runtime_timeout_seconds": 150,
        "worker_scratch_root": "/root/parallax-s3-download-probe",
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "model_loads": 0,
        "gpu_work": 0,
        "runtime": {"job": {
            "execution_site": "gateway",
            "command": ["--url-file", "/tmp/parallax-lifecycle-unit-download/s3-get-url.secret"],
        }},
    }


def test_download_probe_binds_three_streams_exact_bytes_and_five_minute_bar():
    assert validate_download_probe_spec(spec())["maximum_download_bytes"] == 842_380_992


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repetitions", 47),
        ("parallel_streams", 4),
        ("maximum_download_bytes", 842_380_993),
        ("projected_full_cache_bytes", 1),
        ("maximum_projected_full_cache_seconds", 301),
        ("gpu_work", 1),
        ("science_items_evaluated", 1),
    ],
)
def test_download_probe_rejects_scope_or_cost_expansion(field: str, value: object):
    payload = spec(); payload[field] = value
    with pytest.raises(DownloadProbeError):
        validate_download_probe_spec(payload)


def test_download_url_must_bind_exact_object_version_and_expiry():
    payload = spec(); payload["source"]["version_id"] = "immutable-version"
    host = "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
    url = (
        f"https://{host}/{payload['source']['key']}?versionId=immutable-version&"
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-SignedHeaders=host&"
        f"X-Amz-Expires=900&X-Amz-Signature={'a' * 64}"
    )
    assert validate_download_url(url, payload) == url
    with pytest.raises(DownloadProbeError):
        validate_download_url(url.replace("immutable-version", "wrong", 1), payload)
