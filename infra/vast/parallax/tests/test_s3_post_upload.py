from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

import pytest

from infra.vast.parallax import s3_post_upload as upload
from infra.vast.parallax import s3_presign_post as presign


RUN_ID = "parallax-pg1-pilot-post-test-v1"
KEY = f"parallax/results/{RUN_ID}/artifacts.tar"


def _contract(*, minimum: int = 1, maximum: int = 1_000_000) -> dict[str, object]:
    return presign.presign_artifact_post(
        access_key="AKIATEST",
        secret_key="test-secret",
        session_token="test-session",
        run_id=RUN_ID,
        key=KEY,
        minimum_bytes=minimum,
        maximum_bytes=maximum,
        now=datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc),
        expires_seconds=1800,
    )


def test_post_policy_fixes_key_encryption_size_and_checksum_algorithm() -> None:
    contract = upload.validate_contract(_contract(minimum=10, maximum=1000))
    policy = json.loads(base64.b64decode(contract["fields"]["policy"], validate=True))
    assert {"key": KEY} in policy["conditions"]
    assert {"x-amz-server-side-encryption": "AES256"} in policy["conditions"]
    assert {"x-amz-checksum-algorithm": "SHA256"} in policy["conditions"]
    assert ["starts-with", "$x-amz-checksum-sha256", ""] in policy["conditions"]
    assert ["content-length-range", 10, 1000] in policy["conditions"]


def test_post_policy_rejects_a_different_result_key() -> None:
    with pytest.raises(presign.PostPolicyError, match="not exact"):
        presign.presign_artifact_post(
            access_key="AKIATEST",
            secret_key="test-secret",
            session_token=None,
            run_id=RUN_ID,
            key=f"parallax/results/{RUN_ID}/other.tar",
            minimum_bytes=1,
            maximum_bytes=1000,
            now=datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc),
            expires_seconds=1800,
        )


def test_worker_rejects_an_unexpected_post_field() -> None:
    contract = _contract()
    contract["fields"]["unfrozen-field"] = "surprise"
    with pytest.raises(upload.PostUploadError, match="field set changed"):
        upload.validate_contract(contract)


def test_streaming_post_supplies_worker_computed_checksum(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"parallax-worker-computed-checksum\n"
    expected_hex = hashlib.sha256(payload).hexdigest()
    expected_base64 = base64.b64encode(bytes.fromhex(expected_hex)).decode("ascii")

    class Response:
        status = 201

        def read(self, _maximum: int) -> bytes:
            return b"<PostResponse/>"

        def getheader(self, name: str) -> str | None:
            return {
                "x-amz-version-id": "version-1",
                "x-amz-checksum-sha256": expected_base64,
            }.get(name)

    class Connection:
        last: "Connection | None" = None

        def __init__(self, host: str, timeout: int) -> None:
            self.host = host
            self.timeout = timeout
            self.sent = bytearray()
            self.headers: dict[str, str] = {}
            Connection.last = self

        def putrequest(self, method: str, path: str) -> None:
            assert (method, path) == ("POST", "/")

        def putheader(self, name: str, value: str) -> None:
            self.headers[name] = value

        def endheaders(self) -> None:
            pass

        def send(self, value: bytes) -> None:
            self.sent.extend(value)

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    monkeypatch.setattr(upload.http.client, "HTTPSConnection", Connection)
    proof = upload.post_bytes(contract=_contract(), payload=payload)
    assert proof == {
        "http_status": 201,
        "version_id": "version-1",
        "bytes": len(payload),
        "sha256": expected_hex,
        "checksum_sha256_base64": expected_base64,
        "server_side_encryption": "AES256",
        "key": KEY,
    }
    assert payload in bytes(Connection.last.sent)
    assert expected_hex.encode() in bytes(Connection.last.sent)
    assert expected_base64.encode() in bytes(Connection.last.sent)


def test_stream_refuses_a_payload_outside_the_frozen_size_range() -> None:
    with pytest.raises(upload.PostUploadError, match="size escapes"):
        upload.post_bytes(contract=_contract(minimum=100, maximum=200), payload=b"small")
