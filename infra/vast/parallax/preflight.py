"""One-command, no-science PARALLAX H100 preflight."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .artifacts import sha256_file
from .cache_gate import CACHE_GATE_MARKER, HYDRATION_ORIGIN
from .image_smoke import run as image_smoke
from .parallax_entities import cache_inventory
from .transfer_receipt import REQUIRED_TRANSFER_SCOPE


MAX_PREFLIGHT_MINUTES = 30


def validate_remote_receipts(
    transfer: dict[str, Any],
    hydration: dict[str, Any],
    *,
    instance_id: int | None,
    content_manifest_sha256: str,
) -> None:
    scope = set(transfer.get("payload_scope", []))
    if not (
        transfer.get("status") == "GREEN"
        and transfer.get("provider") == "vast.ai"
        and transfer.get("direction") == "durable_to_h100_scratch"
        and transfer.get("destination_instance_id") == instance_id
        and REQUIRED_TRANSFER_SCOPE.issubset(scope)
        and transfer.get("cache_contract_manifest_sha256") == content_manifest_sha256
    ):
        raise ValueError("remote durable transfer receipt is invalid or does not bind this H100/input scope")
    if not (
        hydration.get("status") == "GREEN"
        and hydration.get("provider") == "vast.ai"
        and hydration.get("mode") == "NO_SCIENCE_CACHE_HYDRATION"
        and hydration.get("origin") == HYDRATION_ORIGIN
        and hydration.get("destination_instance_id") == instance_id
        and hydration.get("content_manifest_sha256") == content_manifest_sha256
    ):
        raise ValueError("H100 cache hydration receipt is invalid or does not bind this H100/cache")


def run(
    *,
    matrix: Path,
    lock: Path,
    cache_root: Path,
    durable_root: Path | None,
    transfer_receipt: Path | None,
    cache_hydration_receipt: Path | None,
    model_cache_manifest: Path,
    instance_id: int | None,
    require_h100: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    if (durable_root is None) == (transfer_receipt is None):
        raise ValueError("provide exactly one of durable_root or transfer_receipt")
    if durable_root is not None:
        if not durable_root.is_dir():
            raise FileNotFoundError(f"durable store is not mounted: {durable_root}")
        probe = durable_root / ".parallax_preflight_probe"
        probe.write_bytes(b"parallax-preflight-v1\n")
        if probe.read_bytes() != b"parallax-preflight-v1\n":
            raise IOError("durable store round-trip changed bytes")
        probe.unlink()
        durable_access: dict[str, Any] = {"mode": "DIRECT_MOUNT", "roundtrip": "GREEN"}
    else:
        receipt = json.loads(transfer_receipt.read_text(encoding="utf-8"))
        if cache_hydration_receipt is None:
            raise ValueError("remote durable access requires --cache-hydration-receipt")
        hydration = json.loads(cache_hydration_receipt.read_text(encoding="utf-8"))
        cache_evidence = json.loads(model_cache_manifest.read_text(encoding="utf-8"))
        expected = cache_evidence.get("content_manifest_sha256")
        validate_remote_receipts(
            receipt,
            hydration,
            instance_id=instance_id,
            content_manifest_sha256=expected,
        )
        durable_access = {
            "mode": "REMOTE_VAST_GATEWAY_RECEIPT",
            "transfer_receipt": receipt,
            "cache_hydration_receipt": hydration,
        }

    environment = image_smoke(lock, require_h100=require_h100)
    inventory = cache_inventory(matrix, cache_root)
    if not inventory["all_present"]:
        missing = [row["repo"] for row in inventory["entities"] if not row["present"] or not row["has_config"]]
        raise FileNotFoundError(f"frozen model cache is incomplete; downloads are forbidden: {missing}")
    cache_evidence = json.loads(model_cache_manifest.read_text(encoding="utf-8"))
    cache_errors: list[str] = []
    for repo in cache_evidence.get("repositories", []):
        snapshot = cache_root / ("models--" + repo["repo"].replace("/", "--")) / "snapshots" / repo["revision"]
        for file in repo["files"]:
            path = snapshot / file["path"]
            if not path.is_file():
                cache_errors.append(f"missing: {repo['repo']}:{file['path']}")
            elif path.stat().st_size != int(file["bytes"]):
                cache_errors.append(f"size: {repo['repo']}:{file['path']}")
            elif sha256_file(path) != file["sha256"]:
                cache_errors.append(f"sha256: {repo['repo']}:{file['path']}")
    if cache_errors:
        raise IOError(f"local H100 cache verification failed: {cache_errors[:10]}")
    elapsed = time.monotonic() - started
    if elapsed > MAX_PREFLIGHT_MINUTES * 60:
        raise TimeoutError(f"preflight exceeded {MAX_PREFLIGHT_MINUTES} minutes")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "mode": "NO_SCIENCE_PREFLIGHT",
        "elapsed_seconds": elapsed,
        "max_minutes": MAX_PREFLIGHT_MINUTES,
        "environment": environment,
        "cache": inventory,
        "durable_access": durable_access,
        "local_cache_sha256_verification": "GREEN",
    }
    atomic_json(cache_root / CACHE_GATE_MARKER, {
        "schema_version": 1,
        "created_utc": report["created_utc"],
        "status": "GREEN",
        "destination_instance_id": instance_id,
        "content_manifest_sha256": cache_evidence["content_manifest_sha256"],
        "network_cache_mutation_after_this_marker": "FORBIDDEN",
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    access = parser.add_mutually_exclusive_group(required=True)
    access.add_argument("--durable-root", type=Path)
    access.add_argument("--transfer-receipt", type=Path)
    parser.add_argument("--cache-hydration-receipt", type=Path)
    parser.add_argument("--model-cache-manifest", type=Path, required=True)
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-h100", action="store_true")
    args = parser.parse_args()
    report = run(
        matrix=args.matrix,
        lock=args.lock,
        cache_root=args.cache_root,
        durable_root=args.durable_root,
        transfer_receipt=args.transfer_receipt,
        cache_hydration_receipt=args.cache_hydration_receipt,
        model_cache_manifest=args.model_cache_manifest,
        instance_id=args.instance_id,
        require_h100=args.require_h100,
    )
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "mode": report["mode"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
