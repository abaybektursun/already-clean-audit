"""Stage four tiny selected-pilot overlays into immutable, versioned S3."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.vast.parallax.artifacts import atomic_json
from infra.vast.parallax.pilot_selected_scale_gate import (
    DELTA,
    SELECTED_CONFIG,
    SOURCE_CONTRACT,
    validate_local,
    sha256_file,
)


AWS_PROFILE = "fuelos"
REGION = "us-east-1"
BUCKET = "pg-lineage-769063704696"
FREEZE = Path("studies/lineage/cross-lineage/config/parallax_execution_freeze_v81.json")
RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pilot_selected_scale_overlays_s3_v1.json"
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


def _object(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise OverlayStageError(f"overlay is absent: {relative}")
    digest = sha256_file(path)
    return {
        "path": str(relative),
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
        raise OverlayStageError(f"overlay key has a delete marker: {key}")
    return [row for row in listing.get("Versions", []) if row.get("Key") == key]


def _head(row: dict[str, Any], version_id: str) -> dict[str, Any]:
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
        raise OverlayStageError(f"overlay HEAD mismatch: {row['key']}")
    return head


def _stage_one(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    versions = _existing_versions(str(row["key"]))
    if len(versions) > 1:
        raise OverlayStageError(f"overlay key has multiple versions: {row['key']}")
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
            raise OverlayStageError(f"S3 did not return an overlay version: {row['key']}")
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


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    selected_gate = validate_local(root)
    rows = [_object(root, relative) for relative in (SOURCE_CONTRACT, SELECTED_CONFIG, FREEZE, DELTA)]
    if len({row["key"] for row in rows}) != 4:
        raise OverlayStageError("pilot overlay keys are not unique")
    uploaded = [_stage_one(root, row) for row in rows]
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_SELECTED_PILOT_OVERLAYS_S3",
        "scope": "FOUR_TINY_TEXT_OVERLAYS_NO_VAST_NO_GPU_NO_MODEL_NO_DATA",
        "selected_scale_gate": selected_gate,
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
        def normalize(value: dict[str, Any]) -> dict[str, Any]:
            normalized = {key: row for key, row in value.items() if key != "created_utc"}
            normalized["objects"] = [
                {key: item for key, item in row.items() if key != "reused_exact_existing_version"}
                for row in value["objects"]
            ]
            return normalized

        prior_comparable = normalize(prior)
        current_comparable = normalize(receipt)
        if current_comparable != prior_comparable:
            raise OverlayStageError("existing overlay receipt disagrees with exact S3 state")
        return prior
    atomic_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.root)
    print(json.dumps({"status": result["status"], "objects": result["object_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
