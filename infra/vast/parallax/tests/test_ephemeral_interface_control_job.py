from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import ephemeral_interface_control_job as job


ROOT = Path(__file__).resolve().parents[4]
HYDRATION = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_h100_s3_hydration_job_v1.json"
)
BUNDLE_RECEIPT = ROOT / (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
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
    return (
        f"https://{job.base.BUCKET}.s3.{job.base.REGION}.amazonaws.com/"
        f"{row['key']}?{query}"
    )


def _overlay(path: str, digest: str) -> dict[str, object]:
    local = ROOT / path
    row: dict[str, object] = {
        "path": path,
        "bytes": local.stat().st_size,
        "sha256": digest,
        "bucket": job.base.BUCKET,
        "region": job.base.REGION,
        "key": f"parallax/input-staging/sha256/{digest}/{local.name}",
        "version_id": "test-version-" + digest[:12],
    }
    row["url"] = _url(row)
    return row


def _contract() -> dict[str, object]:
    hydration = json.loads(HYDRATION.read_text(encoding="utf-8"))
    receipt = json.loads(BUNDLE_RECEIPT.read_text(encoding="utf-8"))
    model = [{**row, "url": _url(row)} for row in hydration["files"]]
    bundle: dict[str, object] = {
        "bucket": job.base.BUCKET,
        "region": job.base.REGION,
        "key": (
            "parallax/input-staging/sha256/"
            f"{job.base.EXPECTED_BUNDLE_SHA256}/parallax-training-optimizer-scale-v1.tar.gz"
        ),
        "version_id": "Ohfs88ITOuoRFZxeTfVCfCtx1aRAHRfs",
        "bytes": job.base.EXPECTED_BUNDLE_BYTES,
        "sha256": job.base.EXPECTED_BUNDLE_SHA256,
        "entries": receipt["entries"],
    }
    bundle["url"] = _url(bundle)
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_INTERFACE_CONTROL_JOB",
        "stage": "training_interface_control",
        "provider": "vast.ai",
        "run_id": job.RUN_ID,
        "lineage": "pg2",
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
        "overlays": [_overlay(path, digest) for path, digest in job.OVERLAY_IDENTITIES.items()],
        "source_contract_sha256": job.SOURCE_CONTRACT_SHA256,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "snapshot_root": hydration["worker_snapshot_root"],
            "tokenizer_preflight_output": "/workspace/parallax/tokenizer-preflight.json",
            "control_output_dir": "/workspace/parallax/source/studies/interface-control",
            "manifest_output": "/workspace/parallax/artifact-manifest.json",
            "archive_output": "/workspace/parallax/artifacts.tar",
            "phase_output": "/workspace/parallax/phase.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": {
            "parallel_downloads": 3,
            "processor_download_timeout_seconds": 180,
            "weight_download_timeout_seconds": 600,
            "tokenizer_preflight_timeout_seconds": 120,
            "control_timeout_seconds": 900,
            "artifact_upload_timeout_seconds": 600,
            "minimum_free_bytes": 50_000_000_000,
            "maximum_terminal_bytes": 4 * 1024 * 1024,
        },
        "firewall": {
            "processor_gate_before_weight_hydration": True,
            "model_loads_before_processor_gate": 0,
            "weight_objects_before_processor_gate": 0,
            "train_items": 96,
            "discovery_items": 24,
            "confirmation_manifest_present": False,
            "confirmation_images_present": False,
            "confirmation_items_read": 0,
            "promotion_eligible": False,
            "automatic_retry": False,
        },
    }


def test_job_contract_accepts_exact_pg2_s3_only_processor_first_path() -> None:
    contract = _contract()
    assert job.validate_contract(contract) is contract


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["storage"].update(ephemeral_scratch_gb=1000),
        lambda row: row["firewall"].update(processor_gate_before_weight_hydration=False),
        lambda row: row["firewall"].update(weight_objects_before_processor_gate=1),
        lambda row: row["firewall"].update(confirmation_items_read=1),
        lambda row: row["overlays"].pop(),
    ],
)
def test_job_refuses_storage_weight_gate_firewall_or_overlay_drift(mutation) -> None:
    contract = copy.deepcopy(_contract())
    mutation(contract)
    with pytest.raises((job.InterfaceJobError, job.base.OptimizerJobError)):
        job.validate_contract(contract)


def test_tokenizer_red_cannot_reach_weights_or_science(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = copy.deepcopy(_contract())
    contract["paths"] = {
        "runtime_root": str(tmp_path),
        "source_root": str(tmp_path / "source"),
        "bundle_path": str(tmp_path / "source.tar.gz"),
        "snapshot_root": str(tmp_path / "snapshot"),
        "tokenizer_preflight_output": str(tmp_path / "tokenizer-preflight.json"),
        "control_output_dir": str(tmp_path / "control"),
        "manifest_output": str(tmp_path / "manifest.json"),
        "archive_output": str(tmp_path / "artifact.tar"),
        "phase_output": str(tmp_path / "phase.json"),
        "final_output": str(tmp_path / "job-result.json"),
    }
    contract_path = tmp_path / "input.json"
    post_path = tmp_path / "post.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    post_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(job, "validate_contract", lambda row: row)
    monkeypatch.setattr(job.upload, "validate_contract", lambda row: {"run_id": job.RUN_ID})
    monkeypatch.setattr(job.shutil, "disk_usage", lambda _path: SimpleNamespace(free=60_000_000_000))
    monkeypatch.setattr(job.base, "download_source_bundle", lambda _row: {"status": "GREEN"})
    monkeypatch.setattr(job.base, "extract_and_verify_bundle", lambda _row: {"status": "GREEN"})
    monkeypatch.setattr(job, "_download_overlays", lambda _row: {"status": "GREEN"})
    monkeypatch.setattr(job, "_verify_source", lambda _row: {"status": "GREEN"})
    monkeypatch.setattr(
        job,
        "_hydrate_processor",
        lambda _row: {
            "status": "GREEN_PROCESSOR_FILES_ONLY",
            "weight_objects_downloaded": 0,
        },
    )
    monkeypatch.setattr(
        job,
        "_run_tokenizer_preflight",
        lambda _row: (_ for _ in ()).throw(job.InterfaceJobError("planted unequal suffixes")),
    )
    reached: list[str] = []
    monkeypatch.setattr(job, "_hydrate_weights", lambda _row: reached.append("weights"))
    monkeypatch.setattr(job, "_run_control", lambda *_args: reached.append("science"))
    packet = job.run(contract_path, post_path)
    assert packet["status"] == "RED_TOKENIZER_PREFLIGHT_BEFORE_WEIGHT_HYDRATION"
    assert packet["weight_objects_downloaded"] == packet["model_loads"] == 0
    assert reached == []
    assert not contract_path.exists() and not post_path.exists()


def test_artifact_manifest_requires_exact_three_by_six_checkpoints(tmp_path: Path) -> None:
    output = tmp_path / "control"
    output.mkdir()
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}\n", encoding="utf-8")
    control_path = output / "interface-control.json"
    control_path.write_text("{}\n", encoding="utf-8")
    cells = []
    for codebook in job.EXPECTED_CODEBOOKS:
        cell_dir = output / codebook
        cell_dir.mkdir()
        snapshots = []
        for step in job.EXPECTED_SNAPSHOTS:
            checkpoint = cell_dir / f"step-{step:06d}.pt"
            checkpoint.write_bytes(f"{codebook}:{step}\n".encode())
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
        cells.append({"codebook_id": codebook, "snapshots": snapshots})
    control = {"created_utc": "2026-07-18T00:00:00Z", "codebooks": cells}
    contract = {
        "paths": {
            "tokenizer_preflight_output": str(preflight),
            "control_output_dir": str(output),
            "manifest_output": str(tmp_path / "manifest.json"),
            "archive_output": str(tmp_path / "artifact.tar"),
        }
    }
    archive = job._build_archive(contract, control)
    assert archive["manifest"]["checkpoint_count"] == 18
    assert archive["manifest"]["entry_count"] == 20
    assert archive["bytes"] > 0
