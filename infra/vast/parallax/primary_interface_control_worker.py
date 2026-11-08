"""Phase-aware terminal worker for the primary PG1 interface cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Staged beside the hash-frozen PG2 worker.
    import ephemeral_interface_control_worker as legacy
except ModuleNotFoundError:
    from . import ephemeral_interface_control_worker as legacy


RUN_ID = "parallax-pg1-primary-interface-s3-v2"
EXPECTED_COMMAND = [
    "python3",
    "/opt/parallax/primary_interface_control_job.py",
    "--input-contract",
    "/run/parallax/interface-input-contract.json",
    "--artifact-post",
    "/run/parallax/interface-artifact-post.json",
]


class PrimaryInterfaceWorkerError(RuntimeError):
    pass


def _patch() -> dict[str, Any]:
    original = {"RUN_ID": legacy.RUN_ID, "EXPECTED_COMMAND": legacy.EXPECTED_COMMAND}
    legacy.RUN_ID = RUN_ID
    legacy.EXPECTED_COMMAND = EXPECTED_COMMAND
    return original


def _restore(original: dict[str, Any]) -> None:
    legacy.RUN_ID = original["RUN_ID"]
    legacy.EXPECTED_COMMAND = original["EXPECTED_COMMAND"]


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    original = _patch()
    try:
        return legacy.validate_contract(contract)
    finally:
        _restore(original)


def run(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    original = _patch()
    try:
        return legacy.run(contract_path)
    finally:
        _restore(original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.contract.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
