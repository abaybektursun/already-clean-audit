"""Stage the eight tiny PG2 seed-17 range overlays to immutable S3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .ephemeral_pilot_job import PG2_V2_OVERLAY_IDENTITIES as PG2_OVERLAY_IDENTITIES
from .pg2_range_source_gate import validate_local
from .pilot_overlay_stage import AWS_PROFILE, BUCKET, REGION, _object, _stage_one


RECEIPT = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_seed17_range_overlays_s3_v2.json"
)


class PG2RangeOverlayError(RuntimeError):
    pass


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parity = validate_local(root)
    rows = [_object(root, Path(path)) for path in sorted(PG2_OVERLAY_IDENTITIES)]
    observed = {str(row["path"]): str(row["sha256"]) for row in rows}
    if observed != PG2_OVERLAY_IDENTITIES or len({row["key"] for row in rows}) != 8:
        raise PG2RangeOverlayError("PG2 range overlay identity set is not exact and unique")
    staged = [_stage_one(root, row) for row in rows]
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_PG2_SEED17_RANGE_OVERLAYS_S3",
        "scope": "EIGHT_TINY_TEXT_OVERLAYS_NO_VAST_GPU_MODEL_OR_DATA",
        "run_id": "parallax-pg2-seed17-range-s3-v2",
        "lineage": "pg2",
        "role": "supporting",
        "source_parity": parity,
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
            raise PG2RangeOverlayError("existing PG2 overlay receipt disagrees with S3")
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
