"""Stage seven tiny interface-control overlays to immutable versioned S3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .ephemeral_interface_control_job import OVERLAY_IDENTITIES, RUN_ID
from .interface_control_source_gate import validate_local
from .pilot_overlay_stage import AWS_PROFILE, BUCKET, REGION, _object, _stage_one


RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_interface_control_overlays_s3_v3.json"
)


class InterfaceOverlayError(RuntimeError):
    pass


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    source = validate_local(root)
    rows = [_object(root, Path(path)) for path in sorted(OVERLAY_IDENTITIES)]
    observed = {str(row["path"]): str(row["sha256"]) for row in rows}
    if observed != OVERLAY_IDENTITIES or len({str(row["key"]) for row in rows}) != 9:
        raise InterfaceOverlayError("interface-control overlay identity set is not exact and unique")
    staged = [_stage_one(root, row) for row in rows]
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_PG2_INTERFACE_CONTROL_OVERLAYS_S3",
        "scope": "SEVEN_TINY_TEXT_OVERLAYS_NO_VAST_GPU_MODEL_OR_DATA",
        "run_id": RUN_ID,
        "lineage": "pg2",
        "role": "supporting_boundary_first",
        "source_gate": source,
        "aws_profile": AWS_PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "objects": staged,
        "total_bytes": sum(int(row["bytes"]) for row in staged),
        "object_count": len(staged),
        "credentials_or_signed_urls_persisted": False,
        "vast_resources_created": 0,
        "gpu_seconds": 0,
        "paid_compute_authorization": False,
    }
    output = root / RECEIPT
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))

        def normalized(value: dict[str, Any]) -> dict[str, Any]:
            clean = {key: row for key, row in value.items() if key != "created_utc"}
            clean["objects"] = [
                {key: item for key, item in row.items() if key != "reused_exact_existing_version"}
                for row in value["objects"]
            ]
            return clean

        if normalized(prior) != normalized(receipt):
            raise InterfaceOverlayError("existing interface overlay receipt disagrees with S3")
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
