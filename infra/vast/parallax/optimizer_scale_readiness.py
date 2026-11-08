"""Produce the fail-closed read-only receipt for the PG1 optimizer microgate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .ephemeral_optimizer_scale_launch import (
    RUNTIME_IMAGE,
    _require_input_heads,
    require_machine_quarantine,
    select_cheapest_h100_offer,
)
from .ephemeral_s3_launch import (
    _account_zero,
    _aws_json,
    require_s3_safety,
    select_cheapest_offer,
)


SOURCE_CONTRACT = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_optimizer_scale_source_contract_v1.json"
)
HYDRATION_SPEC = (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_h100_s3_hydration_job_v1.json"
)
BUNDLE_RECEIPT = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)
RUNTIME_RECEIPT = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_full_science_ecr_registry_v1.json"
)
H100_RUNTIME_ADJUDICATION = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_h100_managed_runtime_override_adjudication_v1.json"
)
H100_HYDRATION_ADJUDICATION = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg1_h100_s3_hydration_v1_adjudication.json"
)
MACHINE_QUARANTINE = {
    "path": (
        "studies/lineage/cross-lineage/config/"
        "parallax_h100_machine_quarantine_v1.json"
    ),
    "sha256": "8024eef411b96bfd756febbd3687fad4659b296c6b72160908e300d43ae8ff7e",
}
BUNDLE = {
    "bucket": "pg-lineage-769063704696",
    "region": "us-east-1",
    "key": (
        "parallax/input-staging/sha256/"
        "8b2f16138376e992816f669fb63ab9ff18d6085da3d009aa9931f6e0a834c517/"
        "parallax-training-optimizer-scale-v1.tar.gz"
    ),
    "version_id": "Ohfs88ITOuoRFZxeTfVCfCtx1aRAHRfs",
    "bytes": 12_626_256,
    "sha256": "8b2f16138376e992816f669fb63ab9ff18d6085da3d009aa9931f6e0a834c517",
}


class ReadinessError(RuntimeError):
    pass


def _verify_source(root: Path) -> dict[str, Any]:
    path = root / SOURCE_CONTRACT
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN" or contract.get("stage") != "training_artifact_rehearsal":
        raise ReadinessError("optimizer source contract is not frozen")
    checked = []
    for label, row in contract.get("artifacts", {}).items():
        artifact = root / row["path"]
        observed = sha256_file(artifact)
        if observed != row["sha256"]:
            raise ReadinessError(f"optimizer source mismatch: {label}")
        checked.append({"label": label, "path": row["path"], "sha256": observed})
    return {
        "status": "GREEN_EXACT_FROZEN_SOURCE",
        "source_contract": SOURCE_CONTRACT,
        "source_contract_sha256": sha256_file(path),
        "artifact_count": len(checked),
        "artifacts": checked,
    }


def _verify_bundle(root: Path) -> dict[str, Any]:
    path = root / BUNDLE_RECEIPT
    receipt = json.loads(path.read_text(encoding="utf-8"))
    entries = receipt.get("entries", [])
    task_prefix = "studies/lineage/cross-lineage/data/parallax_tasks_v1/"
    discovery = [
        row for row in entries
        if str(row.get("path", "")).startswith(task_prefix + "images/")
        and "/discovery_eval/" in str(row.get("path", ""))
    ]
    train = [
        row for row in entries
        if str(row.get("path", "")).startswith(task_prefix + "images/")
        and "/train/" in str(row.get("path", ""))
    ]
    forbidden = [
        row["path"] for row in entries
        if str(row.get("path", "")).startswith(task_prefix)
        and (
            "/confirmation_eval/" in str(row["path"])
            or str(row["path"]).endswith("manifests/items.jsonl")
            or str(row["path"]).endswith("manifests/confirmation.sha256")
        )
    ]
    if (
        receipt.get("status") != "GREEN"
        or receipt.get("bundle_sha256") != BUNDLE["sha256"]
        or receipt.get("bundle_bytes") != BUNDLE["bytes"]
        or receipt.get("source_contract_sha256")
        != "fe693a6f9ed9542170a5cba8e539308875f09e1143e33375032136da94c212c9"
        or receipt.get("confirmation_image_included") is not False
        or receipt.get("confirmation_manifest_included") is not False
        or len(entries) != 155
        or len(train) != 96
        or len(discovery) != 24
        or forbidden
    ):
        raise ReadinessError("optimizer source bundle/firewall receipt is not exact")
    return {
        "status": "GREEN_EXACT_BUNDLE",
        "path": BUNDLE_RECEIPT,
        "receipt_sha256": sha256_file(path),
        "entries": len(entries),
        "train_images": len(train),
        "discovery_images_present_for_preflight_then_deleted_before_model_load": len(discovery),
        "confirmation_capable_paths": forbidden,
        "confirmation_manifest_included": False,
        "confirmation_image_included": False,
        "bundle": BUNDLE,
    }


def _verify_runtime(root: Path) -> dict[str, Any]:
    path = root / RUNTIME_RECEIPT
    receipt = json.loads(path.read_text(encoding="utf-8"))
    runtime_gate_path = root / H100_RUNTIME_ADJUDICATION
    runtime_gate = json.loads(runtime_gate_path.read_text(encoding="utf-8"))
    hydration_gate_path = root / H100_HYDRATION_ADJUDICATION
    hydration_gate = json.loads(hydration_gate_path.read_text(encoding="utf-8"))
    digest = RUNTIME_IMAGE.rsplit("@", 1)[1]
    details = _aws_json(
        "ecr",
        "describe-images",
        "--repository-name",
        "parallax/runtime-minimal",
        "--image-ids",
        f"imageDigest={digest}",
    ).get("imageDetails", [])
    if len(details) != 1 or details[0].get("imageDigest") != digest:
        raise ReadinessError("exact private runtime image is absent from ECR")
    if (
        receipt.get("status") != "GREEN_REGISTRY_MANIFEST"
        or receipt.get("immutable_reference") != RUNTIME_IMAGE
        or receipt.get("architecture") != "amd64"
        or runtime_gate.get("status") != "GREEN_H100_MANAGED_PRIVATE_RUNTIME_COMPATIBLE"
        or hydration_gate.get("status") != "GREEN_FULL_PG1_H100_HYDRATION"
        or hydration_gate.get("topology", {}).get("runtime_digest") != digest
        or hydration_gate.get("topology", {}).get("managed_template_hash")
        != "a49dfc59f765f8c5b9c1e0c3a318a486"
    ):
        raise ReadinessError("private runtime/H100 linkage receipts are not exact and green")
    return {
        "status": "GREEN_EXACT_PRIVATE_RUNTIME",
        "reference": RUNTIME_IMAGE,
        "ecr_compressed_bytes": details[0].get("imageSizeInBytes"),
        "receipt": RUNTIME_RECEIPT,
        "receipt_sha256": sha256_file(path),
        "architecture": "amd64",
        "h100_runtime_linkage": {
            "path": H100_RUNTIME_ADJUDICATION,
            "sha256": sha256_file(runtime_gate_path),
            "status": runtime_gate["status"],
        },
        "h100_full_input_hydration": {
            "path": H100_HYDRATION_ADJUDICATION,
            "sha256": sha256_file(hydration_gate_path),
            "status": hydration_gate["status"],
            "verified_bytes": hydration_gate["hydration"]["verified_bytes"],
            "elapsed_seconds": hydration_gate["hydration"]["elapsed_seconds"],
        },
    }


def run(root: Path) -> dict[str, Any]:
    source = _verify_source(root)
    bundle = _verify_bundle(root)
    hydration = json.loads((root / HYDRATION_SPEC).read_text(encoding="utf-8"))
    if hydration.get("status") != "FROZEN" or len(hydration.get("files", [])) != 12:
        raise ReadinessError("PG1 hydration spec is not exact/frozen")
    s3_safety = require_s3_safety()
    input_proof = _require_input_heads(
        {"inputs": {"model": hydration["files"], "bundle": BUNDLE}}
    )
    runtime = _verify_runtime(root)
    quarantine = require_machine_quarantine(
        root,
        {
            "worker": {"launch_mode": "managed_template_private_runtime"},
            "bound_artifacts": {"machine_quarantine": MACHINE_QUARANTINE},
        },
    )
    account = _account_zero()
    if account["active_instances"] or account["active_volumes"]:
        raise ReadinessError("readiness audit requires a zero Vast account")
    guardian_contract = {
        "disk_gb": 10,
        "minimum_reliability": 0.98,
        "minimum_inet_down_mbps": 100,
        "minimum_cuda": 0,
        "price_ceiling_dph": 0.12,
    }
    guardian_market = select_cheapest_offer(role="guardian", contract=guardian_contract)
    worker_contract = {
        "disk_gb": 80,
        "minimum_reliability": 0.98,
        "minimum_gpu_ram_mb": 80_000,
        "minimum_cpu_ram_mb": 64_000,
        "minimum_cpu_cores_effective": 8,
        "minimum_disk_bw_mb_s": 500,
        "minimum_cuda": 13.0,
        "minimum_inet_down_mbps": 500,
        "price_ceiling_dph": 2.25,
    }
    worker_market = select_cheapest_h100_offer(
        worker_contract,
        quarantined_machine_ids=set(quarantine["active_machine_ids"]),
        excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_READ_ONLY_NO_PAID_ACTION",
        "scope": "pg1_optimizer_scale_microgate_readiness",
        "mutation": "none",
        "source": source,
        "bundle": bundle,
        "s3_safety": s3_safety,
        "input_proof": input_proof,
        "runtime": runtime,
        "machine_quarantine": quarantine,
        "vast_account": account,
        "guardian_market": guardian_market,
        "h100_market": worker_market,
        "estimated_maximum_paid_usd": round(
            float(worker_market["selected"]["dph_total"]) * (960 / 3600)
            + float(guardian_market["selected"]["dph_total"]) * (1200 / 3600),
            6,
        ),
        "h100_created": False,
        "model_or_data_downloaded": False,
        "automatic_retry": False,
        "next_boundary": "PLANTED_TESTS_AND_CONTENT_HASHED_ONE_SHOT_EXECUTION_FREEZE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root.resolve())
    atomic_json(args.output.resolve(), packet)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "h100": packet["h100_market"]["selected"],
                "estimated_maximum_paid_usd": packet["estimated_maximum_paid_usd"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
