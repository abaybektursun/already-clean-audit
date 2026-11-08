"""Instance-bound, zero-science H100 preflight for the PG2 localization rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json
from .cache_gate import CACHE_GATE_MARKER
from .g1_common import utc_now
from .localization_preflight import run as base_preflight


def run(**kwargs: Any) -> dict[str, Any]:
    config_path = Path(kwargs["config_path"])
    cache_root = Path(kwargs["cache_root"])
    task_root = Path(kwargs["task_root"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if float(config["energy"]["fixed_reference_amplitude"]) != 0.001:
        raise ValueError("localization rescue preflight amplitude differs from 0.001")
    audit = json.loads((task_root / "manifests/audit.json").read_text(encoding="utf-8"))
    if audit.get("consumed_v1_overlap") != {
        "image_hashes": 0,
        "item_ids": 0,
        "source_identities": 0,
    }:
        raise ValueError("localization rescue panel overlaps consumed v1 confirmation")
    report = base_preflight(**kwargs)
    report["mode"] = "NO_SCIENCE_PG2_LOCALIZATION_RESCUE_PREFLIGHT"
    report["stage"] = "pg2_prefix_position_localization_rescue"
    report["fixed_amplitude"] = 0.001
    report["consumed_v1_overlap"] = audit["consumed_v1_overlap"]
    atomic_json(
        cache_root / CACHE_GATE_MARKER,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "GREEN",
            "stage": report["stage"],
            "destination_instance_id": int(kwargs["instance_id"]),
            "content_manifest_sha256": report["cache"]["content_manifest_sha256"],
            "network_cache_mutation_after_this_marker": "FORBIDDEN",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite localization rescue preflight: {args.output}")
    report = run(
        config_path=args.config,
        lock=args.lock,
        cache_root=args.cache_root,
        cache_contract_path=args.cache_contract,
        source_root=args.source_root,
        source_contract_path=args.source_contract,
        task_root=args.task_root,
        direction_path=args.direction,
        runtime_archive=args.runtime_archive,
        instance_id=args.instance_id,
    )
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
