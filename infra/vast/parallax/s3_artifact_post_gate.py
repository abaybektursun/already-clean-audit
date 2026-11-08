"""Exercise the exact worker-to-S3 artifact POST path without GPU or Vast.

The gate writes one tiny, permanent readiness object.  It never persists AWS
credentials, a session token, the signed policy, or the signed endpoint.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.vast.parallax.s3_post_upload import post_bytes
from infra.vast.parallax.s3_presign_post import BUCKET, REGION, presign_artifact_post


AWS_PROFILE = "fuelos"
RUN_ID = "parallax-s3-artifact-post-stock-gate-v1"
KEY = f"parallax/results/{RUN_ID}/artifacts.tar"
RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_s3_artifact_post_stock_gate_v1.json"
)
PAYLOAD = b"PARALLAX S3 artifact POST stock gate v1\n"


class ArtifactPostGateError(RuntimeError):
    pass


def _aws(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        [
            "aws",
            "--profile",
            AWS_PROFILE,
            "--region",
            REGION,
            "--output",
            "json",
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if check and completed.returncode != 0:
        raise ArtifactPostGateError(f"fuelos AWS command failed: {arguments[0]}")
    return completed


def _aws_json(*arguments: str) -> dict[str, Any]:
    completed = _aws(*arguments)
    raw = completed.stdout.decode("utf-8").strip()
    value = json.loads(raw) if raw else {}
    if not isinstance(value, dict):
        raise ArtifactPostGateError("fuelos AWS response is not an object")
    return value


def _require_bucket_safety() -> dict[str, Any]:
    versioning = _aws_json("s3api", "get-bucket-versioning", "--bucket", BUCKET)
    public = _aws_json("s3api", "get-public-access-block", "--bucket", BUCKET)
    encryption = _aws_json("s3api", "get-bucket-encryption", "--bucket", BUCKET)
    blocked = public.get("PublicAccessBlockConfiguration", {})
    algorithms = [
        row.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
        for row in encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
    ]
    if (
        versioning.get("Status") != "Enabled"
        or set(blocked.values()) != {True}
        or "AES256" not in algorithms
    ):
        raise ArtifactPostGateError("S3 safety invariants are not green")
    return {
        "profile": AWS_PROFILE,
        "region": REGION,
        "versioning": "Enabled",
        "all_public_access_blocked": True,
        "default_encryption": "AES256",
    }


def _credentials() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "aws",
            "configure",
            "export-credentials",
            "--profile",
            AWS_PROFILE,
            "--format",
            "process",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ArtifactPostGateError("could not obtain in-memory fuelos credentials")
    value = json.loads(completed.stdout.decode("utf-8"))
    if not value.get("AccessKeyId") or not value.get("SecretAccessKey"):
        raise ArtifactPostGateError("fuelos credentials are incomplete")
    return value


def _require_key_absent() -> None:
    completed = _aws("s3api", "head-object", "--bucket", BUCKET, "--key", KEY, check=False)
    if completed.returncode == 0:
        raise ArtifactPostGateError("one-shot S3 artifact gate key already exists")


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    receipt_path = root / RECEIPT
    if receipt_path.exists():
        raise ArtifactPostGateError("local one-shot S3 artifact gate receipt already exists")
    safety = _require_bucket_safety()
    _require_key_absent()
    credentials = _credentials()
    now = datetime.now(timezone.utc)
    contract = presign_artifact_post(
        access_key=str(credentials["AccessKeyId"]),
        secret_key=str(credentials["SecretAccessKey"]),
        session_token=credentials.get("SessionToken"),
        run_id=RUN_ID,
        key=KEY,
        minimum_bytes=1,
        maximum_bytes=1_000_000,
        now=now,
        expires_seconds=900,
    )
    upload = post_bytes(contract=contract, payload=PAYLOAD, filename="stock-gate.txt")
    del contract
    del credentials

    head = _aws_json(
        "s3api",
        "head-object",
        "--bucket",
        BUCKET,
        "--key",
        KEY,
        "--version-id",
        str(upload["version_id"]),
        "--checksum-mode",
        "ENABLED",
    )
    expected_hex = hashlib.sha256(PAYLOAD).hexdigest()
    expected_base64 = base64.b64encode(bytes.fromhex(expected_hex)).decode("ascii")
    if (
        head.get("VersionId") != upload["version_id"]
        or head.get("ContentLength") != len(PAYLOAD)
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("ChecksumSHA256") != expected_base64
        or head.get("Metadata", {}).get("sha256") != expected_hex
    ):
        raise ArtifactPostGateError("independent controller HEAD did not match the uploaded artifact")

    receipt = {
        "schema_version": 1,
        "status": "GREEN_S3_ARTIFACT_POST_STOCK_GATE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "AWS_S3_ONLY_NO_VAST_NO_GPU_NO_MODEL_NO_DATA",
        "run_id": RUN_ID,
        "s3_safety": safety,
        "artifact": {
            "bucket": BUCKET,
            "key": KEY,
            "version_id": upload["version_id"],
            "bytes": len(PAYLOAD),
            "sha256": expected_hex,
            "checksum_sha256_base64": expected_base64,
            "server_side_encryption": "AES256",
            "controller_head_exact_match": True,
        },
        "security": {
            "aws_credentials_on_worker": False,
            "credentials_or_signed_policy_persisted": False,
            "signed_policy_lifetime_seconds": 900,
            "exact_key_policy": True,
            "s3_validated_worker_checksum": True,
        },
        "vast_resources_created": 0,
        "gpu_seconds": 0,
        "promotion_boundary": "MECHANICAL_ONLY_DOES_NOT_AUTHORIZE_PAID_COMPUTE",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.root)
    print(json.dumps({"status": result["status"], "version_id": result["artifact"]["version_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
