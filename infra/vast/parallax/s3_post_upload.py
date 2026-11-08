"""Stream one file through a frozen S3 POST policy with a computed SHA-256."""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import re
import uuid
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit


class PostUploadError(RuntimeError):
    pass


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN_S3_ARTIFACT_POST":
        raise PostUploadError("artifact POST contract is not frozen schema v1")
    run_id = str(contract.get("run_id", ""))
    key = str(contract.get("key", ""))
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,79}", run_id)
        or key != f"parallax/results/{run_id}/artifacts.tar"
    ):
        raise PostUploadError("artifact POST identity changed")
    parsed = urlsplit(str(contract.get("endpoint", "")))
    if (
        parsed.scheme != "https"
        or parsed.netloc != "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
        or parsed.path != "/"
        or parsed.query
    ):
        raise PostUploadError("artifact POST endpoint changed")
    minimum = contract.get("minimum_bytes")
    maximum = contract.get("maximum_bytes")
    if not isinstance(minimum, int) or not isinstance(maximum, int) or not 1 <= minimum <= maximum <= 1_100_000_000:
        raise PostUploadError("artifact POST byte bounds changed")
    fields = contract.get("fields")
    required = {
        "key",
        "success_action_status",
        "x-amz-server-side-encryption",
        "x-amz-checksum-algorithm",
        "x-amz-credential",
        "x-amz-algorithm",
        "x-amz-date",
        "policy",
        "x-amz-signature",
    }
    allowed_field_sets = (required, required | {"x-amz-security-token"})
    if not isinstance(fields, dict) or set(fields) not in allowed_field_sets:
        raise PostUploadError("artifact POST field set changed")
    if (
        fields.get("key") != key
        or fields.get("success_action_status") != "201"
        or fields.get("x-amz-server-side-encryption") != "AES256"
        or fields.get("x-amz-checksum-algorithm") != "SHA256"
        or fields.get("x-amz-algorithm") != "AWS4-HMAC-SHA256"
    ):
        raise PostUploadError("artifact POST fixed fields changed")
    try:
        import base64 as _base64

        policy = json.loads(_base64.b64decode(str(fields["policy"]), validate=True))
    except (ValueError, json.JSONDecodeError) as error:
        raise PostUploadError("artifact POST policy is malformed") from error
    conditions = policy.get("conditions", [])
    expected = [
        {"bucket": "pg-lineage-769063704696"},
        {"key": key},
        {"success_action_status": "201"},
        {"x-amz-server-side-encryption": "AES256"},
        {"x-amz-checksum-algorithm": "SHA256"},
        ["starts-with", "$x-amz-checksum-sha256", ""],
        ["starts-with", "$x-amz-meta-sha256", ""],
        ["content-length-range", minimum, maximum],
    ]
    if any(row not in conditions for row in expected):
        raise PostUploadError("artifact POST policy lost a required condition")
    return contract


def _part(name: str, value: str, boundary: str) -> bytes:
    return (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
        f"{value}\r\n"
    ).encode("utf-8")


def post_stream(
    *,
    contract: dict[str, Any],
    stream: BinaryIO,
    size: int,
    sha256_hex: str,
    filename: str = "artifacts.tar",
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    contract = validate_contract(contract)
    if not contract["minimum_bytes"] <= size <= contract["maximum_bytes"]:
        raise PostUploadError("artifact file size escapes the POST policy")
    if len(sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in sha256_hex):
        raise PostUploadError("artifact SHA-256 is malformed")
    checksum = base64.b64encode(bytes.fromhex(sha256_hex)).decode("ascii")
    fields = {
        **contract["fields"],
        "x-amz-checksum-sha256": checksum,
        "x-amz-meta-sha256": sha256_hex,
    }
    boundary = "parallax-" + uuid.uuid4().hex
    prefix = b"".join(_part(name, str(value), boundary) for name, value in sorted(fields.items()))
    file_header = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: application/x-tar\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    content_length = len(prefix) + len(file_header) + size + len(suffix)
    parsed = urlsplit(contract["endpoint"])
    connection = http.client.HTTPSConnection(parsed.netloc, timeout=timeout_seconds)
    try:
        connection.putrequest("POST", parsed.path)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()
        connection.send(prefix)
        connection.send(file_header)
        sent = 0
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            connection.send(chunk)
            sent += len(chunk)
        if sent != size:
            raise PostUploadError("artifact stream size changed during upload")
        connection.send(suffix)
        response = connection.getresponse()
        body = response.read(64 * 1024)
        version_id = response.getheader("x-amz-version-id")
        returned_checksum = response.getheader("x-amz-checksum-sha256")
    finally:
        connection.close()
    if response.status != 201 or not version_id:
        raise PostUploadError(
            f"artifact S3 POST lacked HTTP 201/version id: {response.status}; body_bytes={len(body)}"
        )
    if returned_checksum not in {None, checksum}:
        raise PostUploadError("artifact S3 POST returned a different checksum")
    return {
        "http_status": int(response.status),
        "version_id": version_id,
        "bytes": size,
        "sha256": sha256_hex,
        "checksum_sha256_base64": checksum,
        "server_side_encryption": "AES256",
        "key": contract["key"],
    }


def post_file(*, contract: dict[str, Any], path: Path, timeout_seconds: int = 180) -> dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    with path.open("rb") as handle:
        return post_stream(
            contract=contract,
            stream=handle,
            size=size,
            sha256_hex=digest.hexdigest(),
            filename=path.name,
            timeout_seconds=timeout_seconds,
        )


def post_bytes(*, contract: dict[str, Any], payload: bytes, filename: str = "artifacts.tar") -> dict[str, Any]:
    return post_stream(
        contract=contract,
        stream=io.BytesIO(payload),
        size=len(payload),
        sha256_hex=hashlib.sha256(payload).hexdigest(),
        filename=filename,
    )
