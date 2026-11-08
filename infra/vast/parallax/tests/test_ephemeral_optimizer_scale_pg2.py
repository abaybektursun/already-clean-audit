from __future__ import annotations

import copy
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
import yaml

from infra.vast.parallax import ephemeral_optimizer_scale_job as base_job
from infra.vast.parallax import ephemeral_pg2_optimizer_scale_job as job
from infra.vast.parallax import ephemeral_pg2_optimizer_scale_launch as launch
from infra.vast.parallax import pg2_optimizer_overlay_stage as overlay_stage
from infra.vast.parallax import training_optimizer_scale_gate_pg2 as gate
from infra.vast.parallax.training_common import TrainingGateError


ROOT = Path(__file__).resolve().parents[4]
HYDRATION = ROOT / job.EXPECTED_HYDRATION_PATH
BASE_RECEIPT = ROOT / (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)
SOURCE_CONTRACT = ROOT / job.EXPECTED_SOURCE_CONTRACT_PATH
CONFIG = ROOT / job.EXPECTED_CONFIG_PATH
LAUNCH_SPEC_V1 = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_optimizer_scale_s3_launch_v1.json"
)
LAUNCH_SPEC_V2 = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_optimizer_scale_s3_launch_v2.json"
)
LAUNCH_SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_optimizer_scale_s3_launch_v3.json"
)
NO_CREATE_V1 = ROOT / (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg2_optimizer_scale_no_create_v1.json"
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
    return f"https://{base_job.BUCKET}.s3.{base_job.REGION}.amazonaws.com/{row['key']}?{query}"


def _overlay(path: str, digest: str) -> dict[str, object]:
    local = ROOT / path
    row: dict[str, object] = {
        "path": path,
        "bytes": local.stat().st_size,
        "sha256": digest,
        "bucket": base_job.BUCKET,
        "region": base_job.REGION,
        "key": f"parallax/input-staging/sha256/{digest}/{local.name}",
        "version_id": "test-version-" + digest[:12],
    }
    row["url"] = _url(row)
    return row


def _job_contract() -> dict[str, object]:
    hydration = json.loads(HYDRATION.read_text(encoding="utf-8"))
    receipt = json.loads(BASE_RECEIPT.read_text(encoding="utf-8"))
    model = [{**row, "url": _url(row)} for row in hydration["files"]]
    bundle: dict[str, object] = {
        "bucket": base_job.BUCKET,
        "region": base_job.REGION,
        "key": (
            "parallax/input-staging/sha256/"
            f"{base_job.EXPECTED_BUNDLE_SHA256}/parallax-training-optimizer-scale-v1.tar.gz"
        ),
        "version_id": "Ohfs88ITOuoRFZxeTfVCfCtx1aRAHRfs",
        "bytes": base_job.EXPECTED_BUNDLE_BYTES,
        "sha256": base_job.EXPECTED_BUNDLE_SHA256,
        "entries": receipt["entries"],
    }
    bundle["url"] = _url(bundle)
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_OPTIMIZER_SCALE_PG2_JOB",
        "stage": "training_optimizer_scale_microgate",
        "provider": "vast.ai",
        "lineage": {
            "id": "pg2",
            "role": "supporting",
            "repo": job.EXPECTED_REPO,
            "revision": job.EXPECTED_REVISION,
        },
        "instance_id": 123,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": bundle,
        "overlays": [_overlay(path, digest) for path, digest in job.EXPECTED_OVERLAYS.items()],
        "source_contract_sha256": job.EXPECTED_SOURCE_CONTRACT_SHA256,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "overlay_root": "/workspace/parallax/input/overlays",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/"
                f"snapshots/{job.EXPECTED_REVISION}"
            ),
            "preflight_output": "/workspace/parallax/preflight.json",
            "optimizer_output": "/workspace/parallax/optimizer.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": copy.deepcopy(job.EXPECTED_BOUNDS),
        "firewall": copy.deepcopy(job.EXPECTED_FIREWALL),
    }


def test_pg2_source_contract_and_config_are_exact_and_separate() -> None:
    contract = gate._verify_contract(ROOT, SOURCE_CONTRACT)
    config = gate.validate_pg2_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    assert contract["checkpoint"]["repo"] == job.EXPECTED_REPO
    assert config["lineage"]["role"] == "supporting"
    assert config["optimizer"]["learning_rates"] == [0.0003, 0.0001, 0.00003, 0.00001]
    assert config["cross_lineage_firewall"]["allowed_cross_lineage_output"] == (
        "normalized_arm_verdicts_only"
    )


def test_pg2_overlay_stage_validates_only_six_tiny_exact_local_sources() -> None:
    result = overlay_stage.validate_local(ROOT)
    assert result["status"] == "GREEN_LOCAL_PG2_OVERLAYS"
    assert len(result["objects"]) == 6
    assert sum(int(row["bytes"]) for row in result["objects"]) < 1024 * 1024
    assert all(row["key"].startswith("parallax/input-staging/sha256/") for row in result["objects"])


def test_pg2_overlay_stage_refuses_duplicate_versions(monkeypatch) -> None:
    row = overlay_stage.validate_local(ROOT)["objects"][0]
    monkeypatch.setattr(
        overlay_stage,
        "_existing_versions",
        lambda key: [{"VersionId": "one"}, {"VersionId": "two"}],
    )
    with pytest.raises(overlay_stage.OverlayStageError, match="multiple versions"):
        overlay_stage._stage_one(ROOT, row)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["lineage"].update(role="primary"),
        lambda row: row["lineage"].update(revision="0" * 40),
        lambda row: row["optimizer"].update(learning_rates=[0.00001]),
        lambda row: row["optimizer"].update(ladder_origin="imported_PG1_selected_scale"),
        lambda row: row["cross_lineage_firewall"].update(
            compare_raw_objectives_across_generations="allowed"
        ),
    ],
)
def test_pg2_config_refuses_lineage_revision_or_selected_lr_transfer(mutation) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutation(config)
    with pytest.raises(TrainingGateError, match="frozen PG2 supporting"):
        gate.validate_pg2_config(config)


def test_pg2_job_contract_accepts_exact_s3_only_inputs_and_overlays() -> None:
    contract = _job_contract()
    assert job.validate_contract(contract) is contract


def test_pg2_launcher_builds_the_exact_worker_job_contract() -> None:
    expected = _job_contract()
    spec = {
        "inputs": {
            "bundle_receipt": {
                "path": str(BASE_RECEIPT.relative_to(ROOT)),
                "sha256": base_job.sha256_file(BASE_RECEIPT),
            }
        },
        "job": {"bounds": copy.deepcopy(job.EXPECTED_BOUNDS)},
    }
    observed = launch._job_contract(
        ROOT,
        spec,
        instance_id=123,
        model=expected["model"],
        bundle=expected["bundle"],
        overlays=expected["overlays"],
    )
    assert job.validate_contract(observed) is observed


def test_pg2_launcher_builds_exact_guardian_assignment() -> None:
    assignment = launch._worker_assignment(run_id="parallax-pg2-test", worker_id=123)
    assert assignment == {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": "parallax-pg2-test",
        "worker_label": "parallax-worker-parallax-pg2-test",
        "worker_id": 123,
    }
    assert launch._worker_assignment_ack(
        run_id="parallax-pg2-test", worker_id=123
    ) == {
        **assignment,
        "status": "GUARDIAN_WORKER_ASSIGNMENT_ACKNOWLEDGED",
    }
    with pytest.raises(launch.PG2OptimizerLaunchError, match="assignment identity"):
        launch._worker_assignment(run_id="parallax-pg2-test", worker_id=0)


def test_pg2_incremental_launch_state_persists_only_sanitized_audit_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "launch-state.json"
    state = {
        "schema_version": 1,
        "run_id": "parallax-pg2-test",
        "status": "ACTIVE",
        "phase": "GUARDIAN_CREATED",
        "guardian_instance_id": 11,
        "worker_instance_id": None,
        "guardian_offer": {"machine_id": 22, "gpu_name": "RTX 3060", "dph_total": 0.05},
        "worker_offer": None,
        "guardian_exact_worker_ID_acknowledged": False,
        "job_released": False,
        "s3_durable": False,
        "failure": None,
        "cleanup": None,
        "automatic_retry": False,
    }
    launch._record_launch_state(path, state)
    observed = json.loads(path.read_text(encoding="utf-8"))
    assert observed["guardian_instance_id"] == 11
    assert observed["phase"] == "GUARDIAN_CREATED"
    assert "updated_utc" in observed
    assert "api_key" not in json.dumps(observed).lower()
    with pytest.raises(launch.PG2OptimizerLaunchError, match="state is malformed"):
        launch._record_launch_state(path, {**state, "secret": "forbidden"})


def test_pg2_launch_state_path_is_derived_from_receipt_directory() -> None:
    spec = {"output_receipt": {"path": "results/v3/receipt.json"}}
    assert launch._launch_state_path(ROOT, spec) == ROOT / "results/v3/launch-state.json"


def test_pg2_launcher_heads_all_sixteen_inputs(monkeypatch) -> None:
    contract = _job_contract()
    rows = [*contract["model"], contract["bundle"], *contract["overlays"]]
    by_key = {row["key"]: row for row in rows}

    def head(*arguments: str):
        key = arguments[arguments.index("--key") + 1]
        row = by_key[key]
        return {
            "ContentLength": row["bytes"],
            "VersionId": row["version_id"],
            "ServerSideEncryption": "AES256",
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(row["sha256"])).decode("ascii"),
            "Metadata": {"sha256": row["sha256"]},
        }

    monkeypatch.setattr(launch, "_aws_json", head)
    result = launch._require_input_heads(
        {"inputs": {"model": contract["model"], "bundle": contract["bundle"], "overlays": contract["overlays"]}}
    )
    assert result["status"] == "GREEN_EXACT_IMMUTABLE_PG2_INPUTS"
    assert result["objects"] == 16


def test_consumed_pg2_v1_refuses_post_incident_implementation_drift() -> None:
    with pytest.raises(launch.PG2OptimizerLaunchError):
        launch.validate_spec(ROOT, LAUNCH_SPEC_V1)


def test_consumed_pg2_v2_refuses_post_stage_retry_implementation_drift() -> None:
    with pytest.raises(launch.PG2OptimizerLaunchError):
        launch.validate_spec(ROOT, LAUNCH_SPEC_V2)


def test_consumed_pg2_v3_refuses_current_program_spec_drift_but_keeps_incident_bindings() -> None:
    with pytest.raises(launch.PG2OptimizerLaunchError, match="research_spec hash mismatch"):
        launch.validate_spec(ROOT, LAUNCH_SPEC)
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    assert spec["run_id"] == launch.EXPECTED_RUN_ID
    assert spec["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert spec["bound_artifacts"]["v1_void_incident"]["sha256"] == (
        "69ea807eaf51b1d8be5562254b7eabf70fc40775523fd5a90dfa99bcd4bd2ed0"
    )
    assert spec["bound_artifacts"]["v2_void_incident"]["sha256"] == (
        "e01624725373544a43a696215d60b24e52cdc28abc995d9b626883e4e0c2f121"
    )
    retry_receipt = ROOT / spec["bound_artifacts"]["stage_retry_tests"]["path"]
    assert json.loads(retry_receipt.read_text(encoding="utf-8"))["status"] == (
        "GREEN_PG2_V3_STAGE_RETRY_TESTS"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(scope="pg1_optimizer_scale_microgate"),
        lambda row: row.update(automatic_paid_retry=True),
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["inputs"]["overlays"][0].update(version_id="different"),
        lambda row: row["invariants"].remove(
            "supporting_PG2_is_not_part_of_the_primary_PG1_to_pi_causal_chain"
        ),
    ],
)
def test_pg2_launch_spec_refuses_scope_retry_storage_input_or_lineage_drift(
    mutation, tmp_path: Path
) -> None:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    mutation(spec)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(launch.PG2OptimizerLaunchError):
        launch.validate_spec(ROOT, candidate)


def test_pg2_paid_execution_requires_a_separate_freeze(tmp_path: Path) -> None:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    with pytest.raises(launch.PG2OptimizerLaunchError, match="paid execution freeze"):
        launch.validate_execution_freeze(ROOT, LAUNCH_SPEC, freeze, spec)


def test_pg2_no_create_receipt_accepts_explicit_empty_id_lists_and_refuses_hidden_ids() -> None:
    receipt = json.loads(NO_CREATE_V1.read_text(encoding="utf-8"))
    spec = json.loads(LAUNCH_SPEC_V1.read_text(encoding="utf-8"))
    launch._require_green_no_create_receipt(
        receipt,
        run_id=spec["run_id"],
        spec_sha256=base_job.sha256_file(LAUNCH_SPEC_V1),
        result_key=spec["result"]["key"],
    )
    receipt["terminal_account"]["instance_ids"] = [999]
    with pytest.raises(launch.PG2OptimizerLaunchError, match="not exact and green"):
        launch._require_green_no_create_receipt(
            receipt,
            run_id=spec["run_id"],
            spec_sha256=base_job.sha256_file(LAUNCH_SPEC_V1),
            result_key=spec["result"]["key"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["lineage"].update(role="primary"),
        lambda row: row["model"][0].update(version_id="different"),
        lambda row: row["overlays"].pop(),
        lambda row: row["firewall"].update(prior_pg2_dynamic_gate_verified=False),
        lambda row: row.update(source_contract_sha256="0" * 64),
    ],
)
def test_pg2_job_refuses_storage_lineage_object_overlay_or_firewall_drift(mutation) -> None:
    contract = _job_contract()
    mutation(contract)
    with pytest.raises(base_job.OptimizerJobError):
        job.validate_contract(contract)


def test_pg2_source_and_prior_dynamic_gate_verify_against_raw_artifacts() -> None:
    contract = _job_contract()
    contract["paths"]["source_root"] = str(ROOT)
    proof = job.verify_source_and_prerequisite(contract)
    assert proof["status"] == "GREEN_SOURCE_AND_PRIOR_PG2_DYNAMIC_GATE"
    assert proof["prior_zero_delta_logits_bit_exact"] is True
    assert proof["prior_gradient_norm"] > 0
    assert proof["same_training_code_and_task_hashes"] is True


def test_pg2_job_proves_all_small_sources_before_model_hydration(monkeypatch, tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    final_output = tmp_path / "job-result.json"
    raw = {
        "model": [{"url": "secret"}],
        "bundle": {"url": "secret"},
        "overlays": [{"url": "secret"}],
        "paths": {"final_output": str(final_output)},
        "bounds": {"minimum_free_bytes": 1},
    }
    contract_path.write_text(json.dumps(raw), encoding="utf-8")
    order = []
    monkeypatch.setattr(job, "validate_contract", lambda value: value)
    monkeypatch.setattr(job.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    monkeypatch.setattr(
        job,
        "download_source_bundle",
        lambda value: order.append("bundle_download") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "extract_and_verify_bundle",
        lambda value: order.append("base_source_verify") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "download_and_apply_overlays",
        lambda value: order.append("overlay_verify") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "verify_source_and_prerequisite",
        lambda value: order.append("source_and_confound_proof") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "hydrate_model",
        lambda value: order.append("model_hydration") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "runtime_preflight",
        lambda value, proof: order.append("runtime_preflight") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "execute",
        lambda value, preflight: order.append("model_execution")
        or {
            "optimizer": {
                "status": "GREEN_MECHANICAL_EXECUTION",
                "adjudication": {"verdict": "GREEN_SELECTED_SCALE"},
            }
        },
    )
    packet = job.run(contract_path)
    assert order == [
        "bundle_download",
        "base_source_verify",
        "overlay_verify",
        "source_and_confound_proof",
        "model_hydration",
        "runtime_preflight",
        "model_execution",
    ]
    assert packet["status"] == "GREEN_MECHANICAL_EXECUTION"
    assert not contract_path.exists()
