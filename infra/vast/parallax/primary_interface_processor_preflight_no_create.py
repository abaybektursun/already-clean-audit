"""Zero-mutation rehearsal for the primary PG1 processor-only cloud gate."""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .artifacts import atomic_json, sha256_file
from . import primary_interface_processor_preflight_launch as launch
from .ephemeral_s3_worker import validate_contract as validate_worker_contract
from .primary_interface_processor_preflight_job import RUN_ID, validate_contract as validate_job_contract


class PrimaryProcessorNoCreateError(RuntimeError):
    pass


def _get_url(row: dict[str, Any]) -> str:
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "redacted/20260718/us-east-1/s3/aws4_request",
            "X-Amz-Date": "20260718T000000Z",
            "X-Amz-Expires": "900",
            "X-Amz-SignedHeaders": "host",
            "versionId": row["version_id"],
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://{launch.BUCKET}.s3.{launch.REGION}.amazonaws.com/{row['key']}?{query}"


def _put_url(key: str) -> str:
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "redacted/20260718/us-east-1/s3/aws4_request",
            "X-Amz-Date": "20260718T000000Z",
            "X-Amz-Expires": "900",
            "X-Amz-SignedHeaders": "host;x-amz-server-side-encryption",
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://{launch.BUCKET}.s3.{launch.REGION}.amazonaws.com/{key}?{query}"


def run(root: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite no-create rehearsal: {output}")
    spec = launch.validate_spec(root, spec_path)
    processor_rows = launch._processor_rows(root, spec)
    overlay_rows = launch._overlay_rows(root, spec)
    bundle_row, bundle_receipt = launch._bundle(root, spec)
    processor = [{**row, "url": _get_url(row)} for row in processor_rows]
    bundle = {**bundle_row, "url": _get_url(bundle_row)}
    job = launch._job_contract(
        spec,
        instance_id=123,
        processor=processor,
        bundle=bundle,
        bundle_receipt=bundle_receipt,
        overlays=overlay_rows,
    )
    worker = launch._worker_contract(
        spec,
        instance_id=123,
        result_url=_put_url(str(spec["result"]["key"])),
    )
    validate_job_contract(job)
    validate_worker_contract(worker)
    source = inspect.getsource(launch.run)
    selector = inspect.getsource(launch.select_cheapest_cpu_preflight_offer)
    if (
        "select_cheapest_h100_offer" in source
        or "wait_h100_ssh" in source
        or source.count("_create_instance(") != 1
        or "H100" in selector
        and '"H100" not in' not in selector
        or source.index("_wait_result(") > source.index("_finalize_lifecycle(")
    ):
        raise PrimaryProcessorNoCreateError("H100 or durability ordering entered preflight launch")
    if any("safetensors" in str(row["path"]) for row in job["processor"]):
        raise PrimaryProcessorNoCreateError("model weight entered rehearsed processor job")
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_PRIMARY_PROCESSOR_PREFLIGHT_NO_CREATE",
        "run_id": RUN_ID,
        "launch_spec": {
            "path": str(spec_path.resolve().relative_to(root.resolve())),
            "sha256": sha256_file(spec_path),
        },
        "validated": {
            "processor_objects": len(job["processor"]),
            "processor_bytes": sum(int(row["bytes"]) for row in job["processor"]),
            "model_weight_urls": job["firewall"]["model_weight_urls"],
            "overlays": len(job["overlays"]),
            "bundle_entries": len(job["bundle"]["entries"]),
            "h100_reachable": job["firewall"]["h100_creation_reachable"],
            "gpu_operations": job["firewall"]["gpu_operations"],
            "confirmation_items_read": job["firewall"]["confirmation_items_read"],
            "automatic_retry": job["firewall"]["automatic_retry"],
            "versioned_S3_terminal_then_destroy": worker["post_upload_action"] == "destroy",
            "upload_failure_action": worker["upload_failure_action"],
            "create_calls_in_paid_run": source.count("_create_instance("),
            "h100_selector_references": source.count("select_cheapest_h100_offer"),
        },
        "mutation": {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "s3_objects_written": 0,
            "files_downloaded": 0,
            "gpu_operations": 0,
            "h100_selections": 0,
        },
        "paid_authorization": False,
        "automatic_retry": False,
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root.resolve(), args.spec.resolve(), args.output.resolve())
    print(json.dumps({"status": packet["status"], "run_id": packet["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
