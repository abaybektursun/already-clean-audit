from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import ephemeral_interface_control_job as job
from infra.vast.parallax import ephemeral_interface_control_launch as launch
from infra.vast.parallax import ephemeral_s3_launch as s3_launch


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_interface_control_s3_launch_v3.json"
)
CONSUMED_SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_pg2_interface_control_s3_launch_v2.json"
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
    return f"https://{job.base.BUCKET}.s3.{job.base.REGION}.amazonaws.com/{row['key']}?{query}"


def test_launch_spec_is_exact_pg2_first_s3_only_and_nonpromotion() -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    assert spec["run_id"] == job.RUN_ID
    assert spec["storage"]["durable_provider"] == "amazon_s3"
    assert spec["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert spec["storage"]["ephemeral_scratch_gb"] == spec["worker"]["disk_gb"] == 80
    assert spec["authorization"]["PG2"] is True
    assert spec["authorization"]["PG1"] is False
    assert spec["authorization"]["promotion"] is False
    assert spec["automatic_paid_retry"] is False


def test_consumed_v2_refuses_fresh_v3_implementation() -> None:
    with pytest.raises(launch.InterfaceLaunchError, match="one-shot scope"):
        launch.validate_spec(ROOT, CONSUMED_SPEC)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["storage"].update(ephemeral_scratch_gb=1000),
        lambda row: row["worker"].update(disk_gb=1000),
        lambda row: row["authorization"].update(PG1=True),
        lambda row: row["result"].update(pre_science_terminal_durable_action="stop"),
    ],
)
def test_launch_spec_refuses_storage_lineage_or_lifecycle_drift(
    mutation, tmp_path: Path
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(spec)
    candidate = tmp_path / "spec.json"
    candidate.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(launch.InterfaceLaunchError):
        launch.validate_spec(ROOT, candidate)


def test_launcher_builds_exact_processor_first_job_and_phase_aware_worker() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    hydration = json.loads(
        (ROOT / spec["inputs"]["model_manifest"]["path"]).read_text(encoding="utf-8")
    )
    model = [{**row, "url": _url(row)} for row in hydration["files"]]
    bundle = copy.deepcopy(spec["inputs"]["bundle"])
    bundle["url"] = _url(bundle)
    overlays = []
    receipt = json.loads(
        (ROOT / spec["inputs"]["overlay_receipt"]["path"]).read_text(encoding="utf-8")
    )
    for row in receipt["objects"]:
        overlays.append({**row, "url": _url(row)})
    bundle_receipt = json.loads(BUNDLE_RECEIPT.read_text(encoding="utf-8"))
    job_contract = launch._job_contract(
        spec,
        instance_id=123,
        model=model,
        bundle=bundle,
        bundle_receipt=bundle_receipt,
        overlays=overlays,
    )
    assert job.validate_contract(job_contract) is job_contract
    result_url = (
        "https://pg-lineage-769063704696.s3.us-east-1.amazonaws.com/"
        f"parallax/results/{job.RUN_ID}/terminal.json?"
        + urlencode(
            {
                "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                "X-Amz-Expires": "1800",
                "X-Amz-SignedHeaders": "host;x-amz-server-side-encryption",
                "X-Amz-Signature": "0" * 64,
            }
        )
    )
    worker_contract = launch._worker_contract(spec, instance_id=123, result_url=result_url)
    assert launch.validate_worker_contract(worker_contract) is worker_contract


def test_exact_guardian_ack_precedes_h100_ssh_and_job_release() -> None:
    source = inspect.getsource(launch.run)
    assert source.index("_stage_atomic(") < source.index("_wait_remote_json_exact(")
    assert source.index("_wait_remote_json_exact(") < source.index("wait_h100_ssh(")
    assert source.index("validate_job_contract(input_contract)") < source.index(
        'record("JOB_RELEASED"'
    )
    assert "maximum_bytes=int(spec[\"result\"][\"maximum_bytes\"])" in source
    assert "pre_science_terminal_durable" in source


def test_worker_create_arguments_contain_no_persistent_storage(monkeypatch) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    monkeypatch.setattr(s3_launch, "_ephemeral_worker_registry_login", lambda _row: "token")
    arguments = s3_launch._create_arguments(role="worker", spec=spec, run_id=job.RUN_ID)
    assert "--volume" not in arguments
    assert arguments[arguments.index("--disk") + 1] == "80"


def test_exact_initial_interface_state_passes_incremental_recorder(tmp_path: Path) -> None:
    state = {
        "schema_version": 1,
        "run_id": job.RUN_ID,
        "status": "ACTIVE",
        "phase": "CLAIMED_BEFORE_PAID_MUTATION",
        "guardian_instance_id": None,
        "worker_instance_id": None,
        "guardian_offer": None,
        "worker_offer": None,
        "guardian_exact_worker_ID_acknowledged": False,
        "job_released": False,
        "s3_durable": False,
        "failure": None,
        "cleanup": None,
        "automatic_retry": False,
    }
    output = tmp_path / "launch-state.json"
    launch._record_launch_state(output, state)
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed["run_id"] == job.RUN_ID
    assert observed["phase"] == "CLAIMED_BEFORE_PAID_MUTATION"
