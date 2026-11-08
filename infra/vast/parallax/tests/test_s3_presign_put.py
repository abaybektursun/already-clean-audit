from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from infra.vast.parallax.s3_presign_put import (
    PresignError,
    presign_get,
    presign_put,
    presign_result_put,
)


def _url(**overrides: object) -> str:
    values = {
        "access_key": "ASIAEXAMPLE",
        "secret_key": "example-secret",
        "session_token": "example-token",
        "bucket": "pg-lineage-769063704696",
        "key": "parallax/input-staging/sha256/ab/object.bin",
        "region": "us-east-1",
        "checksum_sha256_base64": "YWJjZA==",
        "metadata_sha256_hex": "a" * 64,
        "now": datetime(2026, 7, 17, 4, 30, tzinfo=timezone.utc),
        "expires_seconds": 900,
    }
    values.update(overrides)
    return presign_put(**values)  # type: ignore[arg-type]


def test_presigned_put_is_frozen_to_bucket_prefix_region_and_signed_headers():
    parsed = urlsplit(_url())
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
    assert parsed.path == "/parallax/input-staging/sha256/ab/object.bin"
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Expires"] == ["900"]
    assert query["X-Amz-Security-Token"] == ["example-token"]
    assert query["X-Amz-SignedHeaders"] == [
        "host;x-amz-checksum-sha256;x-amz-meta-sha256;x-amz-server-side-encryption"
    ]
    assert len(query["X-Amz-Signature"][0]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bucket", "wrong-bucket"),
        ("region", "us-west-2"),
        ("key", "parallax/other/object.bin"),
        ("key", "parallax/input-staging/../escape"),
        ("expires_seconds", 59),
        ("expires_seconds", 3601),
    ],
)
def test_presigned_put_rejects_scope_expansion(field: str, value: object):
    with pytest.raises(PresignError):
        _url(**{field: value})


def test_presigned_put_signature_binds_checksum_and_object_key():
    first = parse_qs(urlsplit(_url()).query)["X-Amz-Signature"][0]
    checksum_changed = parse_qs(
        urlsplit(_url(checksum_sha256_base64="ZWZnaA==")).query
    )["X-Amz-Signature"][0]
    key_changed = parse_qs(
        urlsplit(_url(key="parallax/input-staging/sha256/ac/object.bin")).query
    )["X-Amz-Signature"][0]
    assert len({first, checksum_changed, key_changed}) == 3


def test_presigned_get_binds_the_immutable_version_id():
    common = {
        "access_key": "ASIAEXAMPLE", "secret_key": "example-secret",
        "session_token": "example-token", "bucket": "pg-lineage-769063704696",
        "key": "parallax/input-staging/sha256/ab/object.bin", "region": "us-east-1",
        "now": datetime(2026, 7, 17, 4, 30, tzinfo=timezone.utc), "expires_seconds": 900,
    }
    first = urlsplit(presign_get(**common, version_id="version-one"))
    second = urlsplit(presign_get(**common, version_id="version-two"))
    first_query = parse_qs(first.query); second_query = parse_qs(second.query)
    assert first_query["versionId"] == ["version-one"]
    assert first_query["X-Amz-SignedHeaders"] == ["host"]
    assert first_query["X-Amz-Signature"] != second_query["X-Amz-Signature"]


def _result_url(**overrides: object) -> str:
    values = {
        "access_key": "ASIAEXAMPLE",
        "secret_key": "example-secret",
        "session_token": "example-token",
        "bucket": "pg-lineage-769063704696",
        "key": "parallax/results/rehearsal-20260718/terminal.json",
        "region": "us-east-1",
        "now": datetime(2026, 7, 18, 10, 30, tzinfo=timezone.utc),
        "expires_seconds": 900,
    }
    values.update(overrides)
    return presign_result_put(**values)  # type: ignore[arg-type]


def test_result_put_binds_exact_key_and_encryption_without_preknown_payload_hash():
    parsed = urlsplit(_result_url())
    query = parse_qs(parsed.query)
    assert parsed.path == "/parallax/results/rehearsal-20260718/terminal.json"
    assert query["X-Amz-SignedHeaders"] == ["host;x-amz-server-side-encryption"]
    assert query["X-Amz-Expires"] == ["900"]
    assert len(query["X-Amz-Signature"][0]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bucket", "wrong-bucket"),
        ("region", "us-west-2"),
        ("key", "parallax/input-staging/run/terminal.json"),
        ("key", "parallax/results/../escape/terminal.json"),
        ("key", "parallax/results/run/result.json"),
        ("expires_seconds", 59),
        ("expires_seconds", 3601),
    ],
)
def test_result_put_rejects_scope_expansion(field: str, value: object):
    with pytest.raises(PresignError):
        _result_url(**{field: value})


def test_result_put_signature_changes_with_terminal_key():
    first = parse_qs(urlsplit(_result_url()).query)["X-Amz-Signature"][0]
    second = parse_qs(
        urlsplit(_result_url(key="parallax/results/another-run/terminal.json")).query
    )["X-Amz-Signature"][0]
    assert first != second
