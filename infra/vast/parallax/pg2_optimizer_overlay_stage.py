"""Stage the six tiny PG2 optimizer overlays directly to immutable S3 keys."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from infra.vast.parallax.artifacts import atomic_json, sha256_file
from infra.vast.parallax.ephemeral_pg2_optimizer_scale_job import EXPECTED_OVERLAYS
from infra.vast.parallax.training_optimizer_scale_gate_pg2 import (
    _verify_contract,
    validate_pg2_config,
)


AWS_PROFILE = "fuelos"
REGION = "us-east-1"
BUCKET = "pg-lineage-769063704696"
SOURCE_CONTRACT = Path(
    "studies/lineage/cross-lineage/config/"
    "parallax_training_optimizer_scale_source_contract_pg2_v1.json"
)
CONFIG = Path(
    "studies/lineage/cross-lineage/config/"
    "parallax_training_optimizer_scale_gate_pg2_v1.yaml"
)
RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_optimizer_overlays_s3_v1.json"
)


class OverlayStageError(RuntimeError):
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
        timeout=60,
    )
    if check and completed.returncode != 0:
        raise OverlayStageError(f"fuelos AWS command failed: {arguments[0]}")
    return completed


def _aws_json(*arguments: str) -> dict[str, Any]:
    completed = _aws(*arguments)
    raw = completed.stdout.decode("utf-8").strip()
    value = json.loads(raw) if raw else {}
    if not isinstance(value, dict):
        raise OverlayStageError("fuelos AWS response is not an object")
    return value


def _object(root: Path, relative: str, expected_digest: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise OverlayStageError(f"PG2 overlay is absent or not tiny: {relative}")
    digest = sha256_file(path)
    if digest != expected_digest:
        raise OverlayStageError(f"PG2 overlay source drift: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "checksum_sha256_base64": base64.b64encode(bytes.fromhex(digest)).decode("ascii"),
        "key": f"parallax/input-staging/sha256/{digest}/{path.name}",
    }


def _existing_versions(key: str) -> list[dict[str, Any]]:
    listing = _aws_json(
        "s3api",
        "list-object-versions",
        "--bucket",
        BUCKET,
        "--prefix",
        key,
    )
    markers = [row for row in listing.get("DeleteMarkers", []) if row.get("Key") == key]
    if markers:
        raise OverlayStageError(f"PG2 overlay key has a delete marker: {key}")
    return [row for row in listing.get("Versions", []) if row.get("Key") == key]


def _head(row: dict[str, Any], version_id: str) -> None:
    head = _aws_json(
        "s3api",
        "head-object",
        "--bucket",
        BUCKET,
        "--key",
        row["key"],
        "--version-id",
        version_id,
        "--checksum-mode",
        "ENABLED",
    )
    if (
        head.get("VersionId") != version_id
        or head.get("ContentLength") != row["bytes"]
        or head.get("ChecksumSHA256") != row["checksum_sha256_base64"]
        or head.get("Metadata", {}).get("sha256") != row["sha256"]
        or head.get("ServerSideEncryption") != "AES256"
    ):
        raise OverlayStageError(f"PG2 overlay HEAD mismatch: {row['key']}")


def _stage_one(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    versions = _existing_versions(str(row["key"]))
    if len(versions) > 1:
        raise OverlayStageError(f"PG2 overlay key has multiple versions: {row['key']}")
    reused = bool(versions)
    if reused:
        version_id = str(versions[0]["VersionId"])
    else:
        response = _aws_json(
            "s3api",
            "put-object",
            "--bucket",
            BUCKET,
            "--key",
            row["key"],
            "--body",
            str(root / row["path"]),
            "--server-side-encryption",
            "AES256",
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            row["checksum_sha256_base64"],
            "--metadata",
            f"sha256={row['sha256']}",
        )
        version_id = str(response.get("VersionId", ""))
        if not version_id:
            raise OverlayStageError(f"S3 did not return a PG2 overlay version: {row['key']}")
    _head(row, version_id)
    return {
        **row,
        "bucket": BUCKET,
        "region": REGION,
        "version_id": version_id,
        "server_side_encryption": "AES256",
        "controller_head_exact_match": True,
        "reused_exact_existing_version": reused,
    }


def validate_local(root: Path) -> dict[str, Any]:
    contract = _verify_contract(root, root / SOURCE_CONTRACT)
    config = validate_pg2_config(yaml.safe_load((root / CONFIG).read_text(encoding="utf-8")))
    rows = [_object(root, path, digest) for path, digest in EXPECTED_OVERLAYS.items()]
    if len({row["key"] for row in rows}) != len(EXPECTED_OVERLAYS):
        raise OverlayStageError("PG2 overlay keys are not unique")
    return {
        "status": "GREEN_LOCAL_PG2_OVERLAYS",
        "lineage": config["lineage"],
        "source_contract_sha256": sha256_file(root / SOURCE_CONTRACT),
        "mechanical_prerequisite": contract["mechanical_prerequisite"],
        "objects": rows,
    }


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    local = validate_local(root)
    uploaded = [_stage_one(root, row) for row in local["objects"]]
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_PG2_OPTIMIZER_OVERLAYS_S3",
        "scope": "SIX_TINY_TEXT_OVERLAYS_NO_VAST_NO_GPU_NO_MODEL_OR_DATA_DOWNLOAD",
        "local_validation": {key: value for key, value in local.items() if key != "objects"},
        "aws_profile": AWS_PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "objects": uploaded,
        "total_bytes": sum(int(row["bytes"]) for row in uploaded),
        "object_count": len(uploaded),
        "credentials_or_signed_urls_persisted": False,
        "vast_resources_created": 0,
        "gpu_seconds": 0,
        "paid_compute_authorization": False,
    }
    output = root / RECEIPT
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))

        def normalized(value: dict[str, Any]) -> dict[str, Any]:
            result = {key: row for key, row in value.items() if key != "created_utc"}
            result["objects"] = [
                {key: item for key, item in row.items() if key != "reused_exact_existing_version"}
                for row in value["objects"]
            ]
            return result

        if normalized(prior) != normalized(receipt):
            raise OverlayStageError("existing PG2 overlay receipt disagrees with exact S3 state")
        return prior
    atomic_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"], "objects": result["object_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
