from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

from infra.vast.parallax import ephemeral_pilot_launch as launch
from infra.vast.parallax import ephemeral_s3_launch as s3_launch
from infra.vast.parallax import pilot_no_create_rehearsal as no_create


ROOT = Path(__file__).resolve().parents[4]
PG1_SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_selected_scale_pilot_s3_launch_v1.json"
)
PG2_V1_SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg2_seed17_range_s3_launch_v1.json"
)
SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg2_seed17_range_s3_launch_v2.json"
)


def test_consumed_selected_pilot_refuses_post_incident_implementation_drift() -> None:
    with pytest.raises(launch.PilotLaunchError, match="one-shot supporting scope"):
        launch.validate_spec(ROOT, PG1_SPEC)


def test_consumed_pg2_range_v1_refuses_fresh_v2_identity() -> None:
    with pytest.raises(launch.PilotLaunchError, match="one-shot supporting scope"):
        launch.validate_spec(ROOT, PG2_V1_SPEC)


def test_pg2_range_contract_is_s3_only_exact_scratch_and_nonpromotion() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["run_id"] == launch.RUN_ID
    assert spec["storage"]["durable_provider"] == "amazon_s3"
    assert spec["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert spec["storage"]["ephemeral_scratch_gb"] == 80
    assert spec["worker"]["disk_gb"] == 80
    assert spec["authorization"]["PG2"] is True
    assert spec["authorization"]["promotion"] is False
    assert spec["automatic_paid_retry"] is False
    assert spec["artifact"]["required_checksum"] == "SHA256"


def test_pg2_range_launch_spec_passes_exact_local_validation() -> None:
    spec = launch.validate_spec(ROOT, SPEC)
    assert spec["run_id"] == launch.RUN_ID


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(ephemeral_scratch_gb=1000),
        lambda row: row["worker"].update(disk_gb=1000),
        lambda row: row["authorization"].update(PG2=False),
    ],
)
def test_pg2_range_launch_refuses_blanket_scratch_or_lineage_drift(
    mutation, tmp_path: Path
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(spec)
    candidate = tmp_path / "spec.json"
    candidate.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(launch.PilotLaunchError):
        launch.validate_spec(ROOT, candidate)


def test_exact_id_ack_precedes_h100_ssh_and_repeated_cleanup_is_terminal() -> None:
    source = inspect.getsource(launch.run)
    assert source.index("_stage_atomic(") < source.index("_wait_remote_json_exact(")
    assert source.index("_wait_remote_json_exact(") < source.index("wait_h100_ssh(")
    assert "_finalize_lifecycle(" in source
    assert source.rindex("_confirm_account_zero(") > source.index("_finalize_lifecycle(")


def test_current_storage_policy_has_no_blanket_or_persistent_vast_storage() -> None:
    amendment_path = ROOT / (
        "studies/lineage/cross-lineage/config/"
        "parallax_s3_execution_amendment_v1.json"
    )
    policy_path = ROOT / (
        "studies/lineage/cross-lineage/config/"
        "parallax_cloud_storage_policy_v3.json"
    )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert (
        "spec_section_12_5_dedicated_persistent_Vast_volume_and_gateway"
        in amendment["superseded_clauses"]
    )
    assert amendment["replacement_contract"]["durable_system_of_record"] == (
        "private_versioned_amazon_s3"
    )
    assert amendment["replacement_contract"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert policy["vast"]["persistent_volumes_forbidden"] is True
    assert policy["vast"]["blanket_scratch_allowance_forbidden"] is True
    assert policy["vast"]["current_seed17_range_cell_scratch_gb"] == 80
    assert policy["cost_invariants"]["recurring_vast_storage_usd_per_hour_after_run"] == 0


def test_pilot_launch_requires_a_separate_exact_execution_freeze(tmp_path: Path) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    with pytest.raises(launch.PilotLaunchError, match="execution freeze"):
        launch.validate_execution_freeze(ROOT, SPEC, freeze, spec)


def test_pilot_worker_mode_reaches_the_proven_create_argument_builder(monkeypatch) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    monkeypatch.setattr(s3_launch, "_ephemeral_worker_registry_login", lambda row: "token")
    arguments = s3_launch._create_arguments(role="worker", spec=spec, run_id=launch.RUN_ID)
    assert arguments[:4] == [
        "--template_hash",
        launch.RUNTIME_TEMPLATE,
        "--image",
        launch.RUNTIME_IMAGE,
    ]
    assert arguments[-2:] == ["--login", "token"]
    assert "--volume" not in arguments


def test_no_create_sanitizer_redacts_login_and_refuses_volume() -> None:
    result = no_create._sanitized_construction(
        ["--image", "exact", "--login", "secret"], role="worker"
    )
    assert result["arguments"][-1] == "<ephemeral-redacted>"
    assert result["secrets_persisted"] is False
    with pytest.raises(no_create.PilotNoCreateError, match="persistent storage"):
        no_create._sanitized_construction(
            ["--image", "exact", "--volume", "wrong"], role="worker"
        )
