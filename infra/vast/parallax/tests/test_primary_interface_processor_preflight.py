from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from infra.vast.parallax import ephemeral_s3_launch as s3_launch
from infra.vast.parallax import primary_interface_processor_preflight_job as job
from infra.vast.parallax import primary_interface_processor_preflight_launch as launch
from infra.vast.parallax import primary_interface_processor_preflight_no_create as no_create


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_primary_processor_preflight_s3_launch_v1.json"
)
FREEZE = ROOT / "studies/lineage/cross-lineage/config/parallax_execution_freeze_v110.json"


def test_launch_spec_is_processor_only_s3_durable_and_h100_false() -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    assert spec["run_id"] == job.RUN_ID
    assert spec["authorization"]["h100"] is False
    assert spec["authorization"]["model_weights"] is False
    assert spec["authorization"]["gpu_operations"] is False
    assert spec["storage"]["durable_provider"] == "amazon_s3"
    assert spec["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert spec["storage"]["ephemeral_scratch_gb"] == spec["worker"]["disk_gb"] == 30
    assert spec["automatic_paid_retry"] is False
    assert spec["worker"]["price_ceiling_dph"] == 0.12
    assert spec["worker"]["hard_destroy_seconds"] == 600


def test_execution_freeze_authorizes_only_one_cheap_non_h100_control() -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    freeze = launch.validate_execution_freeze(ROOT, SPEC, FREEZE, spec)
    assert freeze["authorization"]["create_one_cheap_non_h100_control_instance"] is True
    assert freeze["authorization"]["h100"] is False
    assert freeze["authorization"]["model_weights"] is False
    assert freeze["authorization"]["gpu_operations"] is False
    assert freeze["authorization"]["automatic_retry"] is False


def test_execution_freeze_refuses_h100_authority_drift(tmp_path: Path) -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze["authorization"]["h100"] = True
    candidate = tmp_path / "freeze.json"
    candidate.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(launch.PrimaryProcessorLaunchError):
        launch.validate_execution_freeze(ROOT, SPEC, candidate, spec)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["authorization"].update(h100=True),
        lambda row: row["authorization"].update(model_weights=True),
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["storage"].update(ephemeral_scratch_gb=1000),
        lambda row: row["worker"].update(disk_gb=1000),
        lambda row: row["job"]["environment"].update(CUDA_VISIBLE_DEVICES="0"),
        lambda row: row.update(automatic_paid_retry=True),
    ],
)
def test_launch_spec_refuses_h100_weight_gpu_storage_or_retry_drift(
    mutation, tmp_path: Path
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(spec)
    candidate = tmp_path / "spec.json"
    candidate.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(launch.PrimaryProcessorLaunchError):
        launch.validate_spec(ROOT, candidate)


def test_exact_job_contract_contains_only_processor_objects() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    processor = launch._processor_rows(ROOT, spec)
    bundle, receipt = launch._bundle(ROOT, spec)
    overlays = launch._overlay_rows(ROOT, spec)
    signed = [{**row, "url": no_create._get_url(row)} for row in processor]
    contract = launch._job_contract(
        spec,
        instance_id=123,
        processor=signed,
        bundle={**bundle, "url": no_create._get_url(bundle)},
        bundle_receipt=receipt,
        overlays=overlays,
    )
    assert job.validate_contract(contract) is contract
    assert {row["path"] for row in contract["processor"]} == job.PROCESSOR_NAMES
    assert sum(row["bytes"] for row in contract["processor"]) == job.PROCESSOR_BYTES
    assert not any("safetensors" in row["path"] for row in contract["processor"])
    assert contract["firewall"]["h100_creation_reachable"] is False
    assert contract["firewall"]["model_weight_urls"] == 0


def test_job_contract_rejects_weight_even_when_count_is_preserved() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    processor = launch._processor_rows(ROOT, spec)
    bundle, receipt = launch._bundle(ROOT, spec)
    overlays = launch._overlay_rows(ROOT, spec)
    signed = [{**row, "url": no_create._get_url(row)} for row in processor]
    contract = launch._job_contract(
        spec,
        instance_id=123,
        processor=signed,
        bundle={**bundle, "url": no_create._get_url(bundle)},
        bundle_receipt=receipt,
        overlays=overlays,
    )
    bad = copy.deepcopy(contract)
    bad["processor"][0]["path"] = "model-00001-of-00003.safetensors"
    with pytest.raises(job.PrimaryProcessorPreflightError):
        job.validate_contract(bad)


def test_selector_uses_literal_cheapest_eligible_non_h100(monkeypatch) -> None:
    rows = [
        {"id": 1, "machine_id": 1, "gpu_name": "H100 SXM", "cpu_ram": 64000,
         "cpu_cores_effective": 16, "dph_total": 0.01, "reliability": 0.999},
        {"id": 2, "machine_id": 2, "gpu_name": "RTX 4090", "cpu_ram": 4000,
         "cpu_cores_effective": 8, "dph_total": 0.02, "reliability": 0.99},
        {"id": 3, "machine_id": 3, "gpu_name": "RTX 3090", "cpu_ram": 16000,
         "cpu_cores_effective": 4, "dph_total": 0.04, "reliability": 0.98},
        {"id": 4, "machine_id": 4, "gpu_name": "RTX 4090", "cpu_ram": 16000,
         "cpu_cores_effective": 4, "dph_total": 0.05, "reliability": 0.99},
    ]
    monkeypatch.setattr(launch, "_vast", lambda *args, **kwargs: rows)
    selected = launch.select_cheapest_cpu_preflight_offer(
        json.loads(SPEC.read_text(encoding="utf-8"))["worker"]
    )
    assert selected["selected"]["id"] == 3
    assert "H100" not in selected["selected"]["gpu_name"]


def test_create_arguments_have_disposable_disk_and_no_volume(monkeypatch) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    monkeypatch.setattr(s3_launch, "_ephemeral_worker_registry_login", lambda _spec: "token")
    arguments = s3_launch._create_arguments(role="worker", spec=spec, run_id=job.RUN_ID)
    assert "--volume" not in arguments
    assert arguments[arguments.index("--disk") + 1] == "30"
    assert "--onstart-cmd" in arguments


def test_paid_preflight_executable_has_one_create_and_no_h100_selector() -> None:
    source = inspect.getsource(launch.run)
    assert source.count("_create_instance(") == 1
    assert "select_cheapest_h100_offer" not in source
    assert "wait_h100_ssh" not in source
    assert source.index("_wait_result(") < source.index("_finalize_lifecycle(")


def test_no_create_rehearsal_builds_exact_contracts_without_mutation(tmp_path: Path) -> None:
    output = tmp_path / "no-create.json"
    packet = no_create.run(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PRIMARY_PROCESSOR_PREFLIGHT_NO_CREATE"
    assert set(packet["mutation"].values()) == {0}
    assert packet["validated"]["processor_objects"] == 8
    assert packet["validated"]["model_weight_urls"] == 0
    assert packet["validated"]["h100_reachable"] is False
    assert packet["validated"]["versioned_S3_terminal_then_destroy"] is True
