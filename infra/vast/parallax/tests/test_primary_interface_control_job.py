from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import primary_interface_control_job as job
from infra.vast.parallax import primary_interface_control_source_gate as source_gate


ROOT = Path(__file__).resolve().parents[4]
HYDRATION = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg1_h100_s3_hydration_job_v1.json"
)
BUNDLE_RECEIPT = ROOT / source_gate.BUNDLE_RECEIPT


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
        f"https://{job.legacy.base.BUCKET}.s3.{job.legacy.base.REGION}.amazonaws.com/"
        f"{row['key']}?{query}"
    )


def _overlay(path: str, digest: str) -> dict[str, object]:
    local = ROOT / path
    row: dict[str, object] = {
        "path": path,
        "bytes": local.stat().st_size,
        "sha256": digest,
        "bucket": job.legacy.base.BUCKET,
        "region": job.legacy.base.REGION,
        "key": f"parallax/input-staging/sha256/{digest}/{local.name}",
        "version_id": "test-version-" + digest[:12],
    }
    row["url"] = _url(row)
    return row


def _contract() -> dict[str, object]:
    hydration = json.loads(HYDRATION.read_text(encoding="utf-8"))
    receipt = json.loads(BUNDLE_RECEIPT.read_text(encoding="utf-8"))
    source = json.loads((ROOT / job.SOURCE_CONTRACT_PATH).read_text(encoding="utf-8"))
    identities = source_gate.expected_overlay_identities(ROOT, source)
    model = [{**row, "url": _url(row)} for row in hydration["files"]]
    bundle: dict[str, object] = {
        "bucket": job.legacy.base.BUCKET,
        "region": job.legacy.base.REGION,
        "key": (
            "parallax/input-staging/sha256/"
            f"{job.legacy.base.EXPECTED_BUNDLE_SHA256}/"
            "parallax-training-optimizer-scale-v1.tar.gz"
        ),
        "version_id": "Ohfs88ITOuoRFZxeTfVCfCtx1aRAHRfs",
        "bytes": job.legacy.base.EXPECTED_BUNDLE_BYTES,
        "sha256": job.legacy.base.EXPECTED_BUNDLE_SHA256,
        "entries": receipt["entries"],
    }
    bundle["url"] = _url(bundle)
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_INTERFACE_CONTROL_JOB",
        "stage": "training_interface_control",
        "provider": "vast.ai",
        "run_id": job.RUN_ID,
        "lineage": job.LINEAGE,
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
        "overlays": [_overlay(path, digest) for path, digest in identities.items()],
        "source_contract_sha256": identities[job.SOURCE_CONTRACT_PATH],
        "external_processor_gate": {
            "path": job.EXTERNAL_PROCESSOR_TERMINAL_PATH,
            "sha256": job.EXTERNAL_PROCESSOR_TERMINAL_SHA256,
            "s3_version_id": job.EXTERNAL_PROCESSOR_S3_VERSION_ID,
            "status": "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT_EXECUTED",
            "h100_instances_created": 0,
            "model_weight_urls": 0,
            "model_loads": 0,
            "gpu_operations": 0,
            "confirmation_items_read": 0,
        },
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


def _legacy_state() -> dict[str, object]:
    names = (
        "RUN_ID",
        "LINEAGE",
        "SOURCE_CONTRACT_PATH",
        "SOURCE_CONTRACT_SHA256",
        "CONFIG_PATH",
        "MATRIX_PATH",
        "TASK_ROOT",
        "MODEL_FILES",
        "MODEL_BYTES",
        "MODEL_NAMES",
        "WEIGHT_NAMES",
        "OVERLAY_IDENTITIES",
        "EXPECTED_SNAPSHOTS",
        "EXPECTED_CODEBOOKS",
    )
    return {name: getattr(job.legacy, name) for name in names}


def test_primary_contract_accepts_exact_pg1_s3_only_independent_path() -> None:
    contract = _contract()
    before = _legacy_state()
    assert job.validate_contract(contract) is contract
    assert _legacy_state() == before


def test_primary_source_contract_has_one_exact_legacy_config_alias() -> None:
    original_path = ROOT / job.SOURCE_CONTRACT_PATH
    original_bytes = original_path.read_bytes()
    source = json.loads(original_bytes)
    assert "interface_config" not in source["artifacts"]
    assert source["artifacts"]["primary_config"] == {
        "path": job.CONFIG_PATH,
        "sha256": job.OVERLAY_IDENTITIES[job.CONFIG_PATH],
    }
    compatible = job.legacy_compatible_source_contract(source)
    assert compatible["artifacts"]["interface_config"] == source["artifacts"][
        "primary_config"
    ]
    assert len(compatible["artifacts"]) == len(source["artifacts"]) + 1
    assert original_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["artifacts"].pop("primary_config"),
        lambda row: row["artifacts"].update(
            interface_config=copy.deepcopy(row["artifacts"]["primary_config"])
        ),
        lambda row: row["artifacts"]["primary_config"].update(path="wrong.yaml"),
        lambda row: row["artifacts"]["primary_config"].update(sha256="0" * 64),
    ],
)
def test_primary_legacy_config_alias_refuses_missing_ambiguous_or_drifted_source(
    mutation,
) -> None:
    source = json.loads((ROOT / job.SOURCE_CONTRACT_PATH).read_text(encoding="utf-8"))
    mutation(source)
    with pytest.raises(job.PrimaryInterfaceJobError, match="no unique exact"):
        job.legacy_compatible_source_contract(source)


def test_primary_legacy_config_alias_materializes_outside_frozen_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    original = source / job.SOURCE_CONTRACT_PATH
    original.parent.mkdir(parents=True)
    original.write_bytes((ROOT / job.SOURCE_CONTRACT_PATH).read_bytes())
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    materialized = job.materialize_legacy_source_contract(
        source=source, runtime_root=runtime
    )
    assert materialized.parent == runtime
    assert materialized != original
    compatible = json.loads(materialized.read_text(encoding="utf-8"))
    assert compatible["artifacts"]["interface_config"] == compatible["artifacts"][
        "primary_config"
    ]
    with pytest.raises(job.PrimaryInterfaceJobError, match="already exists"):
        job.materialize_legacy_source_contract(source=source, runtime_root=runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["storage"].update(ephemeral_scratch_gb=1000),
        lambda row: row["firewall"].update(confirmation_items_read=1),
        lambda row: row["external_processor_gate"].update(gpu_operations=1),
        lambda row: row["overlays"].pop(),
        lambda row: row["overlays"].append(copy.deepcopy(row["overlays"][0])),
        lambda row: row["overlays"][0].update(sha256="0" * 64),
        lambda row: row.update(source_contract_sha256="0" * 64),
    ],
)
def test_primary_contract_refuses_storage_confirmation_gate_or_overlay_drift(
    mutation,
) -> None:
    contract = copy.deepcopy(_contract())
    mutation(contract)
    before = _legacy_state()
    with pytest.raises((job.PrimaryInterfaceJobError, job.legacy.InterfaceJobError)):
        job.validate_contract(contract)
    assert _legacy_state() == before


def test_external_processor_gate_is_replayed_without_processor_calls(tmp_path: Path) -> None:
    contract = copy.deepcopy(_contract())
    source = tmp_path / "source"
    terminal = source / job.EXTERNAL_PROCESSOR_TERMINAL_PATH
    terminal.parent.mkdir(parents=True)
    terminal.write_bytes((ROOT / job.EXTERNAL_PROCESSOR_TERMINAL_PATH).read_bytes())
    contract["paths"]["source_root"] = str(source)
    contract["paths"]["tokenizer_preflight_output"] = str(tmp_path / "preflight.json")
    preflight, run = job._external_processor_preflight(contract)
    assert preflight["candidate_suffix_token_ids"] == {
        "A": [235280, 1],
        "B": [235305, 1],
        "C": [235288, 1],
    }
    assert run["processor_calls_repeated_on_h100"] == 0
    assert json.loads((tmp_path / "preflight.json").read_text()) == preflight


def test_job_adapter_restores_legacy_bindings_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    contract_path = tmp_path / "contract.json"
    post_path = tmp_path / "post.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    post_path.write_text("{}\n", encoding="utf-8")
    before = _legacy_state()
    original_preflight = job.legacy._run_tokenizer_preflight
    original_control = job.legacy._run_control

    def planted_failure(*_args):
        assert job.legacy.RUN_ID == job.RUN_ID
        assert job.legacy._run_tokenizer_preflight is job._external_processor_preflight
        assert job.legacy._run_control is job._run_primary_control
        raise RuntimeError("planted primary adapter failure")

    monkeypatch.setattr(job.legacy, "run", planted_failure)
    with pytest.raises(RuntimeError, match="planted primary adapter failure"):
        job.run(contract_path, post_path)
    assert _legacy_state() == before
    assert job.legacy._run_tokenizer_preflight is original_preflight
    assert job.legacy._run_control is original_control


def test_primary_control_rejects_lineage_boundary_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = copy.deepcopy(_contract())
    source = tmp_path / "source"
    output = source / "output"
    output.mkdir(parents=True)
    contract["paths"]["source_root"] = str(source)
    contract["paths"]["control_output_dir"] = str(output)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    contract["paths"]["runtime_root"] = str(runtime)
    contract["paths"]["snapshot_root"] = str(
        tmp_path / "cache" / "models--google--paligemma" / "snapshots" / "revision"
    )
    source_contract = source / job.SOURCE_CONTRACT_PATH
    source_contract.parent.mkdir(parents=True)
    source_contract.write_bytes((ROOT / job.SOURCE_CONTRACT_PATH).read_bytes())
    result = {
        "status": "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION",
        "lineage": job.LINEAGE,
        "label": "PRIMARY_PG1_INTERFACE_DISCOVERY_NOT_PROMOTION_ELIGIBLE",
        "supporting_pg2_dependency": "pg2",
        "paired_or_shared_range_claim": False,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
        "fused_base_reset_exact": True,
        "tokenizer_preflight": {"candidate_suffix_token_ids": {"A": [1], "B": [2], "C": [3]}},
        "codebooks": [
            {
                "codebook_id": codebook,
                "snapshots": [{"step": step} for step in job.EXPECTED_SNAPSHOTS],
            }
            for codebook in job.EXPECTED_CODEBOOKS
        ],
    }
    (output / "interface-control.json").write_text(json.dumps(result), encoding="utf-8")
    commands: list[list[str]] = []

    def command(command, *_args, **_kwargs):
        commands.append(command)
        return {"returncode": 0, "elapsed_seconds": 0.0}

    monkeypatch.setattr(job.legacy.base, "_run_command", command)
    with pytest.raises(job.PrimaryInterfaceJobError, match="escaped its frozen boundary"):
        job._run_primary_control(
            contract, {"candidate_suffix_token_ids": {"A": [1], "B": [2], "C": [3]}}
        )
    source_argument = commands[0][commands[0].index("--source-contract") + 1]
    assert source_argument == str(
        runtime / "primary-interface-source-contract-compat.json"
    )
