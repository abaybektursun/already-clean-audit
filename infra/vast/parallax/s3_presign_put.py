"""Generate a least-privilege S3 PUT URL without persisting AWS credentials.

Credentials are accepted only as a JSON object on stdin.  The caller is expected to pipe
``aws configure export-credentials --profile fuelos`` into this process.  Only the signed URL is
written to stdout; credentials are never written to disk or included in the URL except through the
normal temporary session-token query parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from datetime import datetime, timezone
from urllib.parse import quote


class PresignError(RuntimeError):
    pass


def _encode(value: str) -> str:
    return quote(value, safe="-_.~")


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def presign_put(
    *,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    bucket: str,
    key: str,
    region: str,
    checksum_sha256_base64: str,
    metadata_sha256_hex: str,
    now: datetime,
    expires_seconds: int,
) -> str:
    if not access_key or not secret_key:
        raise PresignError("AWS access key material is absent")
    if region != "us-east-1":
        raise PresignError("PARALLAX object staging is frozen to us-east-1")
    if bucket != "pg-lineage-769063704696":
        raise PresignError("unexpected PARALLAX staging bucket")
    if not key.startswith("parallax/input-staging/") or ".." in key.split("/"):
        raise PresignError("object key escapes the frozen staging prefix")
    if len(metadata_sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in metadata_sha256_hex):
        raise PresignError("metadata SHA-256 must be lowercase hexadecimal")
    if not checksum_sha256_base64:
        raise PresignError("base64 payload checksum is absent")
    if not 60 <= expires_seconds <= 3600:
        raise PresignError("PUT URL lifetime must be between 60 and 3600 seconds")
    if now.tzinfo is None:
        raise PresignError("signing time must be timezone-aware")

    instant = now.astimezone(timezone.utc)
    amz_date = instant.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = instant.strftime("%Y%m%d")
    service = "s3"
    host = f"{bucket}.s3.{region}.amazonaws.com"
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    signed_header_names = (
        "host;x-amz-checksum-sha256;x-amz-meta-sha256;x-amz-server-side-encryption"
    )
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_seconds),
        "X-Amz-SignedHeaders": signed_header_names,
    }
    if session_token:
        query["X-Amz-Security-Token"] = session_token
    canonical_query = "&".join(
        f"{_encode(name)}={_encode(value)}" for name, value in sorted(query.items())
    )
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-checksum-sha256:{checksum_sha256_base64}\n"
        f"x-amz-meta-sha256:{metadata_sha256_hex}\n"
        "x-amz-server-side-encryption:AES256\n"
    )
    canonical_request = "\n".join(
        (
            "PUT",
            "/" + "/".join(_encode(part) for part in key.split("/")),
            canonical_query,
            canonical_headers,
            signed_header_names,
            "UNSIGNED-PAYLOAD",
        )
    )
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        )
    )
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, service)
    signing_key = _sign(service_key, "aws4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"https://{host}/{'/'.join(_encode(part) for part in key.split('/'))}?{canonical_query}&X-Amz-Signature={signature}"


def presign_result_put(
    *,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    bucket: str,
    key: str,
    region: str,
    now: datetime,
    expires_seconds: int,
) -> str:
    """Sign one small, not-yet-known terminal JSON result for encrypted PUT.

    Input objects use :func:`presign_put`, whose signature binds a checksum
    known before launch.  A worker's terminal result does not exist before the
    job runs, so this variant binds the exact result key and required S3
    encryption header.  The controller subsequently downloads the returned
    immutable version and verifies its full SHA-256 before teardown is
    adjudicated.
    """
    if not access_key or not secret_key:
        raise PresignError("AWS access key material is absent")
    if region != "us-east-1" or bucket != "pg-lineage-769063704696":
        raise PresignError("unexpected PARALLAX result destination")
    parts = key.split("/")
    if (
        not key.startswith("parallax/results/")
        or not key.endswith("/terminal.json")
        or ".." in parts
        or len(parts) < 4
        or any(not part for part in parts)
    ):
        raise PresignError("result key escapes the frozen terminal-result grammar")
    if not 60 <= expires_seconds <= 3600:
        raise PresignError("result PUT URL lifetime must be between 60 and 3600 seconds")
    if now.tzinfo is None:
        raise PresignError("signing time must be timezone-aware")

    instant = now.astimezone(timezone.utc)
    amz_date = instant.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = instant.strftime("%Y%m%d")
    host = f"{bucket}.s3.{region}.amazonaws.com"
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    signed_header_names = "host;x-amz-server-side-encryption"
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_seconds),
        "X-Amz-SignedHeaders": signed_header_names,
    }
    if session_token:
        query["X-Amz-Security-Token"] = session_token
    canonical_query = "&".join(
        f"{_encode(name)}={_encode(value)}" for name, value in sorted(query.items())
    )
    canonical_uri = "/" + "/".join(_encode(part) for part in parts)
    canonical_headers = f"host:{host}\nx-amz-server-side-encryption:AES256\n"
    canonical_request = "\n".join(
        (
            "PUT",
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_header_names,
            "UNSIGNED-PAYLOAD",
        )
    )
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        )
    )
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, "s3")
    signing_key = _sign(service_key, "aws4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"https://{host}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"


def presign_get(
    *,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    bucket: str,
    key: str,
    version_id: str,
    region: str,
    now: datetime,
    expires_seconds: int,
) -> str:
    """Sign one immutable S3 object version for GET without exposing AWS credentials."""
    if not access_key or not secret_key:
        raise PresignError("AWS access key material is absent")
    if bucket != "pg-lineage-769063704696" or region != "us-east-1":
        raise PresignError("unexpected PARALLAX GET destination")
    if not key.startswith("parallax/input-staging/") or ".." in key.split("/"):
        raise PresignError("GET object key escapes the frozen staging prefix")
    if not version_id or len(version_id) > 256:
        raise PresignError("immutable S3 version ID is absent or malformed")
    if not 60 <= expires_seconds <= 3600 or now.tzinfo is None:
        raise PresignError("GET URL lifetime or signing time is invalid")
    instant = now.astimezone(timezone.utc)
    amz_date = instant.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = instant.strftime("%Y%m%d")
    host = f"{bucket}.s3.{region}.amazonaws.com"
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_seconds),
        "X-Amz-SignedHeaders": "host",
        "versionId": version_id,
    }
    if session_token:
        query["X-Amz-Security-Token"] = session_token
    canonical_query = "&".join(
        f"{_encode(name)}={_encode(value)}" for name, value in sorted(query.items())
    )
    canonical_uri = "/" + "/".join(_encode(part) for part in key.split("/"))
    canonical_request = "\n".join(
        ("GET", canonical_uri, canonical_query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD")
    )
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256", amz_date, scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        )
    )
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, "s3")
    signing_key = _sign(service_key, "aws4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"https://{host}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--checksum-sha256-base64", required=True)
    parser.add_argument("--metadata-sha256-hex", required=True)
    parser.add_argument("--expires-seconds", type=int, default=1800)
    args = parser.parse_args()
    credentials = json.load(sys.stdin)
    url = presign_put(
        access_key=str(credentials.get("AccessKeyId", "")),
        secret_key=str(credentials.get("SecretAccessKey", "")),
        session_token=credentials.get("SessionToken"),
        bucket=args.bucket,
        key=args.key,
        region=args.region,
        checksum_sha256_base64=args.checksum_sha256_base64,
        metadata_sha256_hex=args.metadata_sha256_hex,
        now=datetime.now(timezone.utc),
        expires_seconds=args.expires_seconds,
    )
    sys.stdout.write(url + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
