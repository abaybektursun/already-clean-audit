from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import primary_interface_control_job as job
from infra.vast.parallax import primary_interface_control_launch as launch
from infra.vast.parallax import primary_interface_control_no_create as no_create
from infra.vast.parallax import ephemeral_s3_launch as s3_launch


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / launch.FRESH_SPEC_PATH
SUPERSEDED_SPECS = [
    ROOT
    / "studies/lineage/cross-lineage/config/"
    / f"parallax_pg1_primary_interface_control_s3_launch_v{version}.json"
    for version in (1, 2, 3, 4, 5)
]


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
        f"https://{launch.legacy.BUCKET}.s3.{launch.legacy.REGION}.amazonaws.com/"
        f"{row['key']}?{query}"
    )


def _result_url() -> str:
    key = f"parallax/results/{job.RUN_ID}/terminal.json"
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Expires": "1800",
            "X-Amz-SignedHeaders": "host;x-amz-server-side-encryption",
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://{launch.legacy.BUCKET}.s3.{launch.legacy.REGION}.amazonaws.com/{key}?{query}"


def _contracts() -> tuple[dict, dict]:
    spec = launch.validate_spec(ROOT, SPEC)
    model = [{**row, "url": _url(row)} for row in launch._model_rows(ROOT, spec)]
    bundle_row, bundle_receipt = launch.legacy._bundle(ROOT, spec)
    bundle = {**bundle_row, "url": _url(bundle_row)}
    overlays = [{**row, "url": _url(row)} for row in launch._overlay_rows(ROOT, spec)]
    job_contract = launch._job_contract(
        spec,
        instance_id=123,
        model=model,
        bundle=bundle,
        bundle_receipt=bundle_receipt,
        overlays=overlays,
    )
    worker_contract = launch._worker_contract(
        spec, instance_id=123, result_url=_result_url()
    )
    return job_contract, worker_contract


def test_primary_contract_is_pg1_only_s3_only_and_nonpromotion() -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    assert spec["run_id"] == job.RUN_ID
    assert spec["authorization"]["PG1"] is True
    assert spec["authorization"]["PG2"] is False
    assert spec["authorization"]["confirmation"] is False
    assert spec["authorization"]["promotion"] is False
    assert spec["authorization"]["paired_or_shared_range_claim"] is False
    assert spec["storage"]["durable_provider"] == "amazon_s3"
    assert spec["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert spec["storage"]["ephemeral_scratch_gb"] == spec["worker"]["disk_gb"] == 80
    assert spec["automatic_paid_retry"] is False


def test_fresh_identity_preserves_every_predecessor_science_field() -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    predecessor = json.loads((ROOT / launch.PREDECESSOR_SPEC_PATH).read_text(encoding="utf-8"))
    for field in (
        "program",
        "scope",
        "provider",
        "automatic_paid_retry",
        "authorization",
        "storage",
        "worker_image",
        "worker_registry_auth",
        "guardian",
        "worker",
        "job",
        "inputs",
        "invariants",
        "promotion_boundary",
    ):
        assert spec[field] == predecessor[field]
    for label, row in predecessor["bound_artifacts"].items():
        assert spec["bound_artifacts"][label] == row
    assert spec["bound_artifacts"]["independent_critic"]["sha256"] == (
        launch.CRITIC_SHA256
    )
    assert spec["run_id"] != predecessor["run_id"]


@pytest.mark.parametrize("superseded", SUPERSEDED_SPECS)
def test_consumed_or_superseded_packets_refuse_the_current_implementation(
    superseded: Path,
) -> None:
    with pytest.raises(RuntimeError):
        launch.validate_spec(ROOT, superseded)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(run_id="reused-run"),
        lambda row: row["predecessor_spec"].update(sha256="0" * 64),
        lambda row: row["operational_identity"].update(claim_path="wrong.json"),
        lambda row: row["operational_identity"].update(output_receipt_path="wrong.json"),
        lambda row: row["independent_critic"].update(sha256="0" * 64),
        lambda row: row["preservation"].update(scientific_change_count=1),
        lambda row: row["authorization"].update(paid_action=True),
        lambda row: row.update(implementation_overrides={}),
    ],
)
def test_primary_launch_refuses_fresh_identity_or_preservation_drift(
    mutation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(spec)
    candidate = tmp_path / "spec.json"
    candidate.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(launch, "FRESH_SPEC_PATH", candidate)
    with pytest.raises(launch.PrimaryInterfaceLaunchError):
        launch.validate_spec(ROOT, candidate)


def test_primary_launcher_builds_exact_external_gate_job_and_worker() -> None:
    job_contract, worker_contract = _contracts()
    assert job.validate_contract(job_contract) is job_contract
    assert launch.validate_worker_contract(worker_contract) is worker_contract
    assert job_contract["lineage"] == "pg1_pi"
    assert job_contract["external_processor_gate"]["s3_version_id"] == (
        job.EXTERNAL_PROCESSOR_S3_VERSION_ID
    )
    assert job_contract["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert worker_contract["archive_durable_action"] == "destroy"
    assert worker_contract["terminal_upload_failure_action"] == "stop"


def test_primary_adapter_stages_both_primary_entrypoints_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def stage(_ssh, *, remote: str, content: bytes, mode: str = "0600") -> None:
        assert content
        calls.append(f"{remote}:{mode}")

    monkeypatch.setattr(launch.legacy, "_stage", stage)
    before = {
        name: getattr(launch.legacy, name)
        for name in (
            "RUN_ID",
            "MODEL_BYTES",
            "MODEL_NAMES",
            "OVERLAY_IDENTITIES",
            "SOURCE_CONTRACT_SHA256",
            "EXPECTED_COMMAND",
            "wait_h100_ssh",
            "_stage",
        )
    }
    with pytest.raises(RuntimeError, match="planted adapter failure"):
        with launch._patched_legacy(ROOT):
            launch.legacy._stage(
                [],
                remote="/opt/parallax/ephemeral_interface_control_job.py",
                content=b"legacy-job",
                mode="0700",
            )
            launch.legacy._stage(
                [],
                remote="/opt/parallax/ephemeral_interface_control_worker.py",
                content=b"legacy-worker",
                mode="0700",
            )
            assert launch.legacy.RUN_ID == job.RUN_ID
            raise RuntimeError("planted adapter failure")
    assert calls == [
        "/opt/parallax/ephemeral_interface_control_job.py:0700",
        "/opt/parallax/primary_interface_control_job.py:0700",
        "/opt/parallax/ephemeral_interface_control_worker.py:0700",
        "/opt/parallax/primary_interface_control_worker.py:0700",
    ]
    assert all(getattr(launch.legacy, name) == value for name, value in before.items())


def test_paid_transport_order_and_no_create_module_have_exact_mutation_boundary() -> None:
    paid = inspect.getsource(launch.legacy.run)
    rehearsal = inspect.getsource(no_create.run)
    assert paid.count("_create_instance(") == 2
    assert paid.index("_wait_remote_json_exact(") < paid.index("wait_h100_ssh(")
    assert paid.index("validate_job_contract(input_contract)") < paid.index(
        'record("JOB_RELEASED"'
    )
    assert paid.index("_wait_interface_result(") < paid.index("_finalize_lifecycle(")
    assert "launch.legacy._create_instance(" not in rehearsal
    assert rehearsal.index("legacy_compatible_source_contract") < rehearsal.index(
        "_claim_absent"
    )
    assert rehearsal.index("legacy_compatible_source_contract") < rehearsal.index(
        "select_cheapest_h100_offer"
    )


def test_worker_create_arguments_use_only_80gb_disposable_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    monkeypatch.setattr(
        s3_launch,
        "_ephemeral_worker_registry_login",
        lambda _row: "temporary-registry-token",
    )
    arguments = launch.legacy._create_arguments(role="worker", spec=spec, run_id=job.RUN_ID)
    assert "--volume" not in arguments
    assert arguments[arguments.index("--disk") + 1] == "80"


def test_local_attempt_paths_must_all_be_absent(tmp_path: Path) -> None:
    spec = {"output_receipt": {"path": "attempt/receipt.json"}}
    result = launch.require_local_outputs_absent(tmp_path, spec)
    assert result["status"] == "GREEN_PRIMARY_LOCAL_ATTEMPT_PATHS_ABSENT"
    (tmp_path / "attempt").mkdir()
    (tmp_path / "attempt" / "launch-state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(launch.PrimaryInterfaceLaunchError, match="refuses to overwrite"):
        launch.require_local_outputs_absent(tmp_path, spec)


def test_fatal_provider_runtime_signature_aborts_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launch.optimizer,
        "_show_instance",
        lambda _instance_id: {
            "status_msg": "failed to create task for container: OCI runtime create failed"
        },
    )

    def delegated(*_args, health_check, **_kwargs):
        health_check()
        raise AssertionError("fatal health check should raise first")

    monkeypatch.setattr(launch.optimizer, "wait_h100_ssh", delegated)
    with pytest.raises(launch.LifecycleError, match="fatal provider bootstrap signature"):
        launch.wait_primary_h100_ssh(
            123,
            Path("/tmp/key"),
            bootstrap_seconds=600,
            running_ssh_deadline_seconds=60,
        )
