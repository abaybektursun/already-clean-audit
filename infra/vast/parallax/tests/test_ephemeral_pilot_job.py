from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import ephemeral_pilot_job as job


ROOT = Path(__file__).resolve().parents[4]
HYDRATION = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg1_h100_s3_hydration_job_v1.json"
)
PG2_HYDRATION = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_h100_s3_hydration_job_v1.json"
)
BUNDLE_RECEIPT = ROOT / (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)
OVERLAY_RECEIPT = ROOT / (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pilot_selected_scale_overlays_s3_v1.json"
)


def _url(row: dict[str, object]) -> str:
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "test/20260718/us-east-1/s3/aws4_request",
            "X-Amz-Date": "20260718T000000Z",
            "X-Amz-Expires": "1800",
            "X-Amz-SignedHeaders": "host",
            "versionId": row["version_id"],
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://{job.base.BUCKET}.s3.{job.base.REGION}.amazonaws.com/{row['key']}?{query}"


def _contract() -> dict[str, object]:
    hydration = json.loads(HYDRATION.read_text(encoding="utf-8"))
    bundle_receipt = json.loads(BUNDLE_RECEIPT.read_text(encoding="utf-8"))
    overlay_receipt = json.loads(OVERLAY_RECEIPT.read_text(encoding="utf-8"))
    model = [{**row, "url": _url(row)} for row in hydration["files"]]
    bundle = {
        "bucket": job.base.BUCKET,
        "region": job.base.REGION,
        "key": (
            "parallax/input-staging/sha256/"
            f"{job.base.EXPECTED_BUNDLE_SHA256}/parallax-training-optimizer-scale-v1.tar.gz"
        ),
        "version_id": "Ohfs88ITOuoRFZxeTfVCfCtx1aRAHRfs",
        "bytes": job.base.EXPECTED_BUNDLE_BYTES,
        "sha256": job.base.EXPECTED_BUNDLE_SHA256,
        "entries": bundle_receipt["entries"],
    }
    bundle["url"] = _url(bundle)
    overlays = [{**row, "url": _url(row)} for row in overlay_receipt["objects"]]
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_SELECTED_PILOT_JOB",
        "stage": "training_artifact_rehearsal",
        "provider": "vast.ai",
        "run_id": job.RUN_ID,
        "instance_id": 123,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "ephemeral_scratch_gb": 80,
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": bundle,
        "overlays": overlays,
        "source_contract_sha256": job.SOURCE_CONTRACT_SHA256,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
                "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
            ),
            "preflight_output": "/workspace/parallax/preflight.json",
            "pilot_output_dir": "/workspace/parallax/source/studies/results/pilot",
            "figure_output": "/workspace/parallax/source/studies/results/pilot/pilot.svg",
            "manifest_output": "/workspace/parallax/artifact-manifest.json",
            "archive_output": "/workspace/parallax/artifacts.tar",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": {
            "parallel_downloads": 3,
            "download_timeout_seconds": 600,
            "preflight_timeout_seconds": 180,
            "pilot_timeout_seconds": 600,
            "figure_timeout_seconds": 30,
            "artifact_upload_timeout_seconds": 300,
            "minimum_free_bytes": 50_000_000_000,
        },
        "firewall": {
            "primary_lineage_only": True,
            "confirmation_manifest_present": False,
            "confirmation_images_present": False,
            "train_items": 96,
            "discovery_items": 24,
            "confirmation_items_read": 0,
            "promotion_eligible": False,
            "automatic_retry": False,
        },
    }


def _pg2_contract() -> dict[str, object]:
    contract = copy.deepcopy(_contract())
    hydration = json.loads(PG2_HYDRATION.read_text(encoding="utf-8"))
    contract["run_id"] = job.PG2_RUN_ID
    contract["model"] = [{**row, "url": _url(row)} for row in hydration["files"]]
    contract["overlays"] = []
    for path, digest in job.PG2_OVERLAY_IDENTITIES.items():
        source = ROOT / path
        row = {
            "bucket": job.base.BUCKET,
            "region": job.base.REGION,
            "key": f"parallax/input-staging/sha256/{digest}/{source.name}",
            "version_id": f"version-{source.name}",
            "bytes": source.stat().st_size,
            "sha256": digest,
            "path": path,
        }
        row["url"] = _url(row)
        contract["overlays"].append(row)
    contract["source_contract_sha256"] = job.PG2_SOURCE_CONTRACT_SHA256
    contract["paths"]["snapshot_root"] = hydration["worker_snapshot_root"]
    contract["firewall"] = {
        "supporting_lineage_only": True,
        "confirmation_manifest_present": False,
        "confirmation_images_present": False,
        "train_items": 96,
        "discovery_items": 24,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
        "automatic_retry": False,
    }
    return contract


def _pg2_v2_contract() -> dict[str, object]:
    contract = _pg2_contract()
    contract["run_id"] = job.PG2_V2_RUN_ID
    contract["source_contract_sha256"] = job.PG2_V2_SOURCE_CONTRACT_SHA256
    contract["overlays"] = []
    for path, digest in job.PG2_V2_OVERLAY_IDENTITIES.items():
        source = ROOT / path
        row = {
            "bucket": job.base.BUCKET,
            "region": job.base.REGION,
            "key": f"parallax/input-staging/sha256/{digest}/{source.name}",
            "version_id": f"version-{source.name}",
            "bytes": source.stat().st_size,
            "sha256": digest,
            "path": path,
        }
        row["url"] = _url(row)
        contract["overlays"].append(row)
    return contract


def test_pilot_job_accepts_exact_selected_scale_s3_contract() -> None:
    contract = _contract()
    assert job.validate_contract(contract) is contract


def test_pilot_job_accepts_exact_pg2_seed17_s3_contract() -> None:
    contract = _pg2_contract()
    assert job.validate_contract(contract) is contract


def test_pilot_job_accepts_exact_pg2_seed17_v2_fresh_identity() -> None:
    contract = _pg2_v2_contract()
    assert job.validate_contract(contract) is contract


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(ephemeral_scratch_gb=1000),
        lambda row: row["firewall"].update(supporting_lineage_only=False),
        lambda row: row.update(model=copy.deepcopy(_contract()["model"])),
        lambda row: row["overlays"].pop(),
    ],
)
def test_pg2_job_refuses_blanket_scratch_lineage_model_or_overlay_drift(mutation) -> None:
    contract = _pg2_contract()
    mutation(contract)
    with pytest.raises((job.PilotJobError, job.base.OptimizerJobError)):
        job.validate_contract(contract)


def test_pg2_registered_replacements_are_exact_and_atomic(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    relative = next(iter(job.PG2_REPLACED_BUNDLE_IDENTITIES))
    destination = source / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-bundled-source")
    old_digest = job.PG2_REPLACED_BUNDLE_IDENTITIES[relative]
    row = {
        "path": relative,
        "bytes": 3,
        "sha256": job.PG2_OVERLAY_IDENTITIES[relative],
    }

    def digest(path: Path) -> str:
        if path == destination and path.read_bytes() == b"old-bundled-source":
            return old_digest
        return job.PG2_OVERLAY_IDENTITIES[relative]

    def download(_row, path: Path):
        assert path.name.endswith(".registered-overlay")
        path.write_bytes(b"new")
        return {"path": str(path), "bytes": 3, "sha256": row["sha256"], "version_id": "v"}

    monkeypatch.setattr(job.base, "sha256_file", digest)
    monkeypatch.setattr(job.base, "_download", download)
    result = job._download_overlays(
        {"run_id": job.PG2_RUN_ID, "paths": {"source_root": str(source)}, "overlays": [row]}
    )
    assert destination.read_bytes() == b"new"
    assert result["records"][0]["registered_replacement"] is True
    assert result["records"][0]["replaced_sha256"] == old_digest


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["firewall"].update(confirmation_items_read=1),
        lambda row: row["overlays"][0].update(path="different"),
        lambda row: row.update(source_contract_sha256="0" * 64),
    ],
)
def test_pilot_job_refuses_storage_firewall_overlay_or_source_drift(mutation) -> None:
    contract = _contract()
    mutation(contract)
    with pytest.raises((job.PilotJobError, job.base.OptimizerJobError)):
        job.validate_contract(contract)


def test_artifact_archive_is_exact_six_checkpoints_and_fixed_metadata(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    snapshots = []
    for step in job.EXPECTED_SNAPSHOTS:
        checkpoint = pilot_dir / f"step-{step:06d}.pt"
        checkpoint.write_bytes(f"checkpoint-{step}\n".encode())
        snapshots.append(
            {
                "step": step,
                "checkpoint": {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": job.base.sha256_file(checkpoint),
                },
            }
        )
    pilot = {
        "created_utc": "2026-07-18T00:00:00Z",
        "snapshots": snapshots,
    }
    (pilot_dir / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}\n", encoding="utf-8")
    figure = pilot_dir / "pilot.svg"
    figure.write_text("<svg/>\n", encoding="utf-8")
    contract = copy.deepcopy(_contract())
    contract["paths"].update(
        preflight_output=str(preflight),
        pilot_output_dir=str(pilot_dir),
        figure_output=str(figure),
        manifest_output=str(tmp_path / "manifest.json"),
        archive_output=str(tmp_path / "artifacts.tar"),
    )
    built = job._build_archive(contract, {"pilot": pilot})
    assert built["manifest"]["checkpoint_steps"] == job.EXPECTED_SNAPSHOTS
    assert built["manifest"]["checkpoint_count"] == 6
    assert built["manifest"]["entry_count"] == 9
    with tarfile.open(built["path"], "r") as archive:
        members = archive.getmembers()
    assert [row.name for row in members] == sorted(
        ["artifact-manifest.json", "preflight.json", "pilot/pilot.json", "pilot/pilot.svg"]
        + [f"pilot/step-{step:06d}.pt" for step in job.EXPECTED_SNAPSHOTS]
    )
    assert all(row.mtime == 0 and row.uid == 0 and row.gid == 0 and row.mode == 0o600 for row in members)
