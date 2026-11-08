"""PG1-native exact materialization receipt and staged-byte owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .pg1_execution_capsule import (
    CapsuleError,
    consumer_receipt,
    load_capsule,
    validate_consumer,
    validate_staged_root,
)


def expected_receipt(
    capsule: dict[str, Any],
    staged_root: Path,
    *,
    allow_parameterized_fixture: bool,
) -> dict[str, Any]:
    contract = capsule["contract"]
    return {
        "status": (
            "GREEN_PARAMETERIZED_SOURCE_FIXTURE_V3"
            if allow_parameterized_fixture
            else "GREEN_EXACT_REMOTE_MATERIALIZATION_V3"
        ),
        "scope": (
            "PARAMETERIZED_TRACKED_SOURCE_ONLY"
            if allow_parameterized_fixture
            else "EXACT_DEPLOYMENT_BUNDLE_PLUS_OVERLAYS_AND_NATIVE_V3_WIRING"
        ),
        "capsule_sha256": capsule["content_address"]["digest"],
        "source_root": str(staged_root),
        "bundle_receipt_sha256": contract["materialization"]["bundle_receipt_sha256"],
        "overlay_receipt_sha256": contract["materialization"]["overlay_receipt_sha256"],
        "overlay_paths": sorted(
            row["path"]
            for row in contract["artifacts"].values()
            if row["origin"] == "exact_overlay"
        ),
        "native_runtime_sha256": contract["runtime_wiring"],
        "archive_truth": (
            "NOT_CLAIMED" if allow_parameterized_fixture else "EXACT_REMOTE_REHEARSAL"
        ),
        "confirmation_items_read": 0,
        "claims_written": 0,
        "h100_authority": False,
    }


def capsule_preflight(
    capsule: dict[str, Any],
    staged_root: Path,
    materialization_receipt: dict[str, Any],
    *,
    allow_parameterized_fixture: bool = False,
) -> dict[str, Any]:
    validate_consumer(capsule, "overlay_stage", Path(__file__).resolve())
    root = validate_staged_root(
        staged_root, allow_parameterized_fixture=allow_parameterized_fixture
    )
    expected = expected_receipt(
        capsule, root, allow_parameterized_fixture=allow_parameterized_fixture
    )
    if materialization_receipt != expected:
        raise CapsuleError("native overlay rejected materialization receipt")
    verified = 0
    for label, row in capsule["contract"]["artifacts"].items():
        relative = Path(row["path"])
        if "data" in relative.parts:
            continue
        target = root / relative
        if not target.is_file() or sha256_file(target) != row["sha256"]:
            raise CapsuleError(f"native overlay rejected staged artifact: {label}")
        verified += 1
    for path, digest in capsule["contract"]["runtime_wiring"].items():
        target = root / path
        if not target.is_file() or sha256_file(target) != digest:
            raise CapsuleError(f"native overlay rejected runtime source: {path}")
    return consumer_receipt(
        capsule,
        "overlay_stage",
        {
            "materialization_status": expected["status"],
            "archive_truth": expected["archive_truth"],
            "non_data_artifacts_verified": verified,
            "native_runtime_sources_verified": len(capsule["contract"]["runtime_wiring"]),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-preflight", action="store_true", required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.materialization_receipt.read_text(encoding="utf-8"))
    print(
        json.dumps(
            capsule_preflight(load_capsule(args.capsule), args.staged_root, receipt),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
