from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone

import pytest

from infra.vast.parallax import interface_control_s3_post_upload as upload
from infra.vast.parallax import interface_control_s3_presign_post as presign


RUN_ID = "parallax-pg2-interface-control-s3-v1"
KEY = f"parallax/results/{RUN_ID}/artifacts.tar"


def _contract(*, minimum: int, maximum: int) -> dict[str, object]:
    return presign.presign_artifact_post(
        access_key="AKIATEST",
        secret_key="test-secret",
        session_token="test-session",
        run_id=RUN_ID,
        key=KEY,
        minimum_bytes=minimum,
        maximum_bytes=maximum,
        now=datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc),
        expires_seconds=2700,
    )


def test_interface_policy_supports_exact_18_checkpoint_envelope() -> None:
    contract = upload.validate_contract(
        _contract(minimum=1_200_000_000, maximum=2_100_000_000)
    )
    assert contract["status"] == "FROZEN_S3_INTERFACE_ARTIFACT_POST"
    assert contract["minimum_bytes"] == 1_200_000_000
    assert contract["maximum_bytes"] == 2_100_000_000


def test_interface_policy_rejects_blanket_archive_allowance() -> None:
    with pytest.raises(presign.InterfacePostPolicyError, match="byte range"):
        _contract(minimum=1, maximum=2_100_000_001)


def test_interface_stream_supplies_worker_checksum(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"interface-control-archive-fixture\n"
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")

    class Response:
        status = 201

        def read(self, _maximum: int) -> bytes:
            return b"<PostResponse/>"

        def getheader(self, name: str) -> str | None:
            return {
                "x-amz-version-id": "version-1",
                "x-amz-checksum-sha256": checksum,
            }.get(name)

    class Connection:
        last: "Connection | None" = None

        def __init__(self, host: str, timeout: int) -> None:
            self.host = host
            self.timeout = timeout
            self.sent = bytearray()
            Connection.last = self

        def putrequest(self, method: str, path: str) -> None:
            assert (method, path) == ("POST", "/")

        def putheader(self, _name: str, _value: str) -> None:
            pass

        def endheaders(self) -> None:
            pass

        def send(self, value: bytes) -> None:
            self.sent.extend(value)

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    monkeypatch.setattr(upload.http.client, "HTTPSConnection", Connection)
    import io

    proof = upload.post_stream(
        contract=_contract(minimum=1, maximum=1000),
        stream=io.BytesIO(payload),
        size=len(payload),
        sha256_hex=digest,
    )
    assert proof["version_id"] == "version-1"
    assert proof["sha256"] == digest
    assert proof["checksum_sha256_base64"] == checksum
    assert payload in bytes(Connection.last.sent)
