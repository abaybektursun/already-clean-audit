"""PG1-native source identity and confirmation-firewall owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .pg1_execution_capsule import (
    SOURCE_CONTRACT_PATH,
    SOURCE_RUN_ID,
    CapsuleError,
    consumer_receipt,
    load_capsule,
    validate_consumer,
    validate_staged_root,
)


def _source_contract(capsule: dict[str, Any], staged_root: Path) -> dict[str, Any]:
    source = capsule["contract"]["source_identity"]
    path = staged_root / SOURCE_CONTRACT_PATH
    if (
        not path.is_file()
        or str(SOURCE_CONTRACT_PATH) != source["source_contract_path"]
        or sha256_file(path) != source["source_contract_sha256"]
    ):
        raise CapsuleError("native source contract path or bytes changed")
    contract = json.loads(path.read_text(encoding="utf-8"))
    artifacts = contract.get("artifacts")
    if (
        contract.get("status") != "FROZEN"
        or contract.get("stage") != "training_interface_control"
        or contract.get("run_id") != SOURCE_RUN_ID
        or contract.get("confirmation_items_read") != 0
        or contract.get("paid_authorization") is not False
        or not isinstance(artifacts, dict)
        or any(
            "confirmation" in str(row.get("path", "")).lower()
            for row in artifacts.values()
        )
    ):
        raise CapsuleError("native source identity or confirmation firewall changed")
    return contract


def capsule_preflight(
    capsule: dict[str, Any],
    staged_root: Path,
    *,
    allow_parameterized_fixture: bool = False,
) -> dict[str, Any]:
    validate_consumer(capsule, "source_gate", Path(__file__).resolve())
    root = validate_staged_root(
        staged_root, allow_parameterized_fixture=allow_parameterized_fixture
    )
    contract = _source_contract(capsule, root)
    attempt_id = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    if contract["run_id"] == attempt_id:
        raise CapsuleError("native source and attempt identities are conflated")
    return consumer_receipt(
        capsule,
        "source_gate",
        {
            "source_contract": str(SOURCE_CONTRACT_PATH),
            "source_artifact_labels": len(contract["artifacts"]),
            "identity_relation": "VALID_DISTINCT_SOURCE_AND_ATTEMPT",
        },
    )


def validate_full_source(
    capsule: dict[str, Any], staged_root: Path
) -> dict[str, Any]:
    """Normal-path full source verification; may read discovery data bytes."""

    root = validate_staged_root(staged_root)
    contract = _source_contract(capsule, root)
    for label, row in contract["artifacts"].items():
        target = root / str(row["path"])
        if not target.is_file() or sha256_file(target) != row["sha256"]:
            raise CapsuleError(f"native full source hash mismatch: {label}")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-preflight", action="store_true", required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            capsule_preflight(load_capsule(args.capsule), args.staged_root),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
