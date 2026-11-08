"""Create the exact S3 POST policy for the 18-checkpoint interface archive."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any


BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
MAXIMUM_BYTES = 2_100_000_000


class InterfacePostPolicyError(RuntimeError):
    pass


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def presign_artifact_post(
    *,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    run_id: str,
    key: str,
    minimum_bytes: int,
    maximum_bytes: int,
    now: datetime,
    expires_seconds: int,
) -> dict[str, Any]:
    if not access_key or not secret_key:
        raise InterfacePostPolicyError("AWS access key material is absent")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,79}", run_id):
        raise InterfacePostPolicyError("interface artifact run id violates the frozen grammar")
    if key != f"parallax/results/{run_id}/artifacts.tar":
        raise InterfacePostPolicyError("interface artifact key is not exact")
    if not 1 <= minimum_bytes <= maximum_bytes <= MAXIMUM_BYTES:
        raise InterfacePostPolicyError("interface artifact byte range is invalid")
    if not 300 <= expires_seconds <= 3600 or now.tzinfo is None:
        raise InterfacePostPolicyError("interface POST lifetime or signing time is invalid")
    instant = now.astimezone(timezone.utc)
    amz_date = instant.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = instant.strftime("%Y%m%d")
    scope = f"{date_stamp}/{REGION}/s3/aws4_request"
    credential = f"{access_key}/{scope}"
    expiration = (instant + timedelta(seconds=expires_seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    conditions: list[object] = [
        {"bucket": BUCKET},
        {"key": key},
        {"success_action_status": "201"},
        {"x-amz-server-side-encryption": "AES256"},
        {"x-amz-checksum-algorithm": "SHA256"},
        ["starts-with", "$x-amz-checksum-sha256", ""],
        ["starts-with", "$x-amz-meta-sha256", ""],
        ["content-length-range", minimum_bytes, maximum_bytes],
        {"x-amz-credential": credential},
        {"x-amz-algorithm": "AWS4-HMAC-SHA256"},
        {"x-amz-date": amz_date},
    ]
    fields = {
        "key": key,
        "success_action_status": "201",
        "x-amz-server-side-encryption": "AES256",
        "x-amz-checksum-algorithm": "SHA256",
        "x-amz-credential": credential,
        "x-amz-algorithm": "AWS4-HMAC-SHA256",
        "x-amz-date": amz_date,
    }
    if session_token:
        fields["x-amz-security-token"] = session_token
        conditions.append({"x-amz-security-token": session_token})
    policy = {"expiration": expiration, "conditions": conditions}
    encoded = base64.b64encode(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, REGION)
    service_key = _sign(region_key, "s3")
    signing_key = _sign(service_key, "aws4_request")
    fields["policy"] = encoded
    fields["x-amz-signature"] = hmac.new(
        signing_key, encoded.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "FROZEN_S3_INTERFACE_ARTIFACT_POST",
        "endpoint": f"https://{BUCKET}.s3.{REGION}.amazonaws.com/",
        "run_id": run_id,
        "key": key,
        "minimum_bytes": minimum_bytes,
        "maximum_bytes": maximum_bytes,
        "fields": fields,
    }

