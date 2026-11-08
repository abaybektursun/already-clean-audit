"""Instance-bound, zero-science H100 preflight for the PG2 scalar operator cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .audit_scalar_operator_panel import audit as audit_panel
from .cache_gate import CACHE_GATE_MARKER
from .g1_common import utc_now
from .image_smoke import run as image_smoke


MAX_PREFLIGHT_MINUTES = 20
STAGE = "pg2_full_prefix_scalar_operator"


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run(
    *,
    root: Path,
    config_path: Path,
    lock: Path,
    cache_root: Path,
    cache_contract_path: Path,
    source_contract_path: Path,
    task_root: Path,
    prior_roots: list[Path],
    direction_path: Path,
    arm_freeze: Path,
    runtime_archive: Path,
    instance_id: int,
) -> dict[str, Any]:
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_operator_panel_materialization":
        raise ValueError("scalar operator config is not frozen")
    if config.get("stage") != STAGE:
        raise ValueError("scalar operator config stage differs")
    if len(prior_roots) != 2:
        raise ValueError("scalar operator preflight requires both consumed localization panels")
    environment = image_smoke(lock, require_h100=True)

    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    if source_contract.get("stage") != STAGE:
        raise ValueError("scalar operator source contract stage differs")
    source_errors = []
    for row in source_contract["entries"]:
        path = root / row["path"]
        if not path.is_file():
            source_errors.append(f"missing:{row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            source_errors.append(f"size:{row['path']}")
        elif sha256_file(path) != row["sha256"]:
            source_errors.append(f"sha256:{row['path']}")
    if source_errors:
        raise IOError(f"scalar operator source verification failed: {source_errors[:10]}")

    cache_contract = json.loads(cache_contract_path.read_text(encoding="utf-8"))
    repositories = cache_contract.get("repositories", [])
    expected_contract_sha = canonical_sha256({"repositories": repositories})
    if cache_contract.get("content_manifest_sha256") != expected_contract_sha:
        raise ValueError("scalar operator cache contract canonical hash mismatch")
    if len(repositories) != 1:
        raise ValueError("scalar operator cache contract must contain exactly one repository")
    repository = repositories[0]
    if (
        repository["repo"] != config["lineage"]["checkpoint"]
        or repository["revision"] != config["lineage"]["revision"]
    ):
        raise ValueError("scalar operator cache contract checkpoint mismatch")
    snapshot = (
        cache_root
        / ("models--" + repository["repo"].replace("/", "--"))
        / "snapshots"
        / repository["revision"]
    )
    cache_errors = []
    verified_cache_bytes = 0
    for row in repository["files"]:
        path = snapshot / row["path"]
        if not path.is_file():
            cache_errors.append(f"missing:{row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            cache_errors.append(f"size:{row['path']}")
        elif sha256_file(path) != row["sha256"]:
            cache_errors.append(f"sha256:{row['path']}")
        else:
            verified_cache_bytes += path.stat().st_size
    if cache_errors:
        raise IOError(f"scalar operator cache verification failed: {cache_errors[:10]}")

    banked_audit = json.loads((task_root / "manifests/audit.json").read_text(encoding="utf-8"))
    observed_audit = audit_panel(task_root, prior_roots)
    audit_keys = (
        "status",
        "errors",
        "split_counts",
        "target_balance",
        "unique_item_ids",
        "unique_image_hashes",
        "unique_source_identities",
        "prior_panel_metadata_rows_parsed",
        "prior_panel_overlap",
        "items_jsonl_sha256",
        "content_manifest_sha256",
    )
    if any(observed_audit[key] != banked_audit[key] for key in audit_keys):
        raise IOError("scalar operator task payload differs from banked aggregate audit")
    if observed_audit["status"] != "GREEN" or observed_audit["prior_panel_overlap"] != {
        "image_hashes": 0,
        "item_ids": 0,
        "source_identities": 0,
    }:
        raise IOError("scalar operator panel is not green and fresh")
    if observed_audit["split_counts"] != {
        "confirmation_eval": int(config["panel"]["confirmation_items"])
    }:
        raise IOError("scalar operator panel count differs")

    if sha256_file(direction_path) != config["prerequisites"]["direction"]["file_sha256"]:
        raise IOError("scalar operator direction file hash mismatch")
    frozen = json.loads(arm_freeze.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_FOR_SCALAR_OPERATOR_CONFIRMATION":
        raise IOError("scalar operator arm is not frozen")
    if frozen.get("bindings", {}).get("config_sha256") != sha256_file(config_path):
        raise IOError("scalar operator freeze/config binding mismatch")
    if frozen.get("bindings", {}).get("panel_audit_sha256") != sha256_file(
        task_root / "manifests/audit.json"
    ):
        raise IOError("scalar operator freeze/panel binding mismatch")
    if frozen.get("bindings", {}).get("direction_file_sha256") != sha256_file(direction_path):
        raise IOError("scalar operator freeze/direction binding mismatch")

    runtime_expected = source_contract["runtime_archive"]
    if runtime_archive.stat().st_size != int(runtime_expected["bytes"]):
        raise IOError("scalar operator runtime archive size mismatch")
    if sha256_file(runtime_archive) != runtime_expected["sha256"]:
        raise IOError("scalar operator runtime archive hash mismatch")
    elapsed = time.monotonic() - started
    if elapsed > MAX_PREFLIGHT_MINUTES * 60:
        raise TimeoutError("scalar operator H100 preflight exceeded its hard time limit")

    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN",
        "mode": "NO_SCIENCE_PG2_SCALAR_OPERATOR_PREFLIGHT",
        "provider": "vast.ai",
        "stage": STAGE,
        "destination_instance_id": instance_id,
        "elapsed_seconds": elapsed,
        "max_minutes": MAX_PREFLIGHT_MINUTES,
        "environment": environment,
        "source": {
            "status": "GREEN",
            "entries": len(source_contract["entries"]),
            "contract_sha256": sha256_file(source_contract_path),
        },
        "cache": {
            "status": "GREEN",
            "repositories": 1,
            "files": len(repository["files"]),
            "verified_bytes": verified_cache_bytes,
            "content_manifest_sha256": expected_contract_sha,
        },
        "task": {
            "status": "GREEN",
            "aggregate_confirmation_metadata_rows_parsed": observed_audit["split_counts"][
                "confirmation_eval"
            ],
            "prior_panel_metadata_rows_parsed": observed_audit[
                "prior_panel_metadata_rows_parsed"
            ],
            "prior_panel_overlap": observed_audit["prior_panel_overlap"],
            "confirmation_items_evaluated": 0,
            "content_manifest_sha256": observed_audit["content_manifest_sha256"],
        },
        "direction_file_sha256": sha256_file(direction_path),
        "arm_freeze_sha256": sha256_file(arm_freeze),
        "runtime_archive_sha256": sha256_file(runtime_archive),
        "science_items_evaluated": 0,
    }
    atomic_json(
        cache_root / CACHE_GATE_MARKER,
        {
            "schema_version": 1,
            "created_utc": report["created_utc"],
            "status": "GREEN",
            "stage": STAGE,
            "destination_instance_id": instance_id,
            "content_manifest_sha256": expected_contract_sha,
            "network_cache_mutation_after_this_marker": "FORBIDDEN",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-contract", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--prior-root", action="append", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--arm-freeze", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scalar operator preflight: {args.output}")
    report = run(
        root=args.root,
        config_path=args.config,
        lock=args.lock,
        cache_root=args.cache_root,
        cache_contract_path=args.cache_contract,
        source_contract_path=args.source_contract,
        task_root=args.task_root,
        prior_roots=args.prior_root,
        direction_path=args.direction,
        arm_freeze=args.arm_freeze,
        runtime_archive=args.runtime_archive,
        instance_id=args.instance_id,
    )
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
