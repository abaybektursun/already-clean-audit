"""Stream the 18-checkpoint interface archive through its exact S3 POST policy."""

from __future__ import annotations

import base64
import http.client
import json
import re
import uuid
from typing import Any, BinaryIO
from urllib.parse import urlsplit


MAXIMUM_BYTES = 2_100_000_000


class InterfacePostUploadError(RuntimeError):
    pass


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "FROZEN_S3_INTERFACE_ARTIFACT_POST"
    ):
        raise InterfacePostUploadError("interface POST contract is not frozen schema v1")
    run_id = str(contract.get("run_id", ""))
    key = str(contract.get("key", ""))
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,79}", run_id)
        or key != f"parallax/results/{run_id}/artifacts.tar"
    ):
        raise InterfacePostUploadError("interface POST identity changed")
    parsed = urlsplit(str(contract.get("endpoint", "")))
    if (
        parsed.scheme != "https"
        or parsed.netloc != "pg-lineage-769063704696.s3.us-east-1.amazonaws.com"
        or parsed.path != "/"
        or parsed.query
    ):
        raise InterfacePostUploadError("interface POST endpoint changed")
    minimum = contract.get("minimum_bytes")
    maximum = contract.get("maximum_bytes")
    if (
        not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or not 1 <= minimum <= maximum <= MAXIMUM_BYTES
    ):
        raise InterfacePostUploadError("interface POST byte bounds changed")
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
    if not isinstance(fields, dict) or set(fields) not in (required, required | {"x-amz-security-token"}):
        raise InterfacePostUploadError("interface POST field set changed")
    if (
        fields.get("key") != key
        or fields.get("success_action_status") != "201"
        or fields.get("x-amz-server-side-encryption") != "AES256"
        or fields.get("x-amz-checksum-algorithm") != "SHA256"
        or fields.get("x-amz-algorithm") != "AWS4-HMAC-SHA256"
    ):
        raise InterfacePostUploadError("interface POST fixed fields changed")
    try:
        policy = json.loads(base64.b64decode(str(fields["policy"]), validate=True))
    except (ValueError, json.JSONDecodeError) as error:
        raise InterfacePostUploadError("interface POST policy is malformed") from error
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
    if any(row not in policy.get("conditions", []) for row in expected):
        raise InterfacePostUploadError("interface POST policy lost a required condition")
    return contract


def _part(name: str, value: str, boundary: str) -> bytes:
    return (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
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
        raise InterfacePostUploadError("interface artifact size escapes the POST policy")
    if len(sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in sha256_hex):
        raise InterfacePostUploadError("interface artifact SHA-256 is malformed")
    checksum = base64.b64encode(bytes.fromhex(sha256_hex)).decode("ascii")
    fields = {
        **contract["fields"],
        "x-amz-checksum-sha256": checksum,
        "x-amz-meta-sha256": sha256_hex,
    }
    boundary = "parallax-interface-" + uuid.uuid4().hex
    prefix = b"".join(_part(name, str(value), boundary) for name, value in sorted(fields.items()))
    header = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/x-tar\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    connection = http.client.HTTPSConnection(urlsplit(contract["endpoint"]).netloc, timeout=timeout_seconds)
    try:
        connection.putrequest("POST", "/")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(len(prefix) + len(header) + size + len(suffix)))
        connection.endheaders()
        connection.send(prefix)
        connection.send(header)
        sent = 0
        while chunk := stream.read(8 * 1024 * 1024):
            connection.send(chunk)
            sent += len(chunk)
        if sent != size:
            raise InterfacePostUploadError("interface artifact stream size changed")
        connection.send(suffix)
        response = connection.getresponse()
        body = response.read(64 * 1024)
        version_id = response.getheader("x-amz-version-id")
        returned_checksum = response.getheader("x-amz-checksum-sha256")
    finally:
        connection.close()
    if response.status != 201 or not version_id:
        raise InterfacePostUploadError(
            f"interface S3 POST lacked HTTP 201/version id: {response.status}; body_bytes={len(body)}"
        )
    if returned_checksum not in {None, checksum}:
        raise InterfacePostUploadError("interface S3 POST returned a different checksum")
    return {
        "http_status": int(response.status),
        "version_id": version_id,
        "bytes": size,
        "sha256": sha256_hex,
        "checksum_sha256_base64": checksum,
        "server_side_encryption": "AES256",
        "key": contract["key"],
    }

