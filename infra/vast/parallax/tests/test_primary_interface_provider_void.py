from __future__ import annotations

import json
from pathlib import Path

from infra.vast.parallax.artifacts import sha256_file
from infra.vast.parallax.ephemeral_optimizer_scale_launch import require_machine_quarantine


ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / (
    "studies/lineage/cross-lineage/results/"
    "parallax_training_interface_control_pg1_primary_v1"
)
RESULT_V4 = ROOT / (
    "studies/lineage/cross-lineage/results/"
    "parallax_training_interface_control_pg1_primary_v1_attempt_v4"
)


def test_provider_void_is_pre_science_cleaned_and_all_paid_off() -> None:
    state = json.loads((RESULT / "launch-state.json").read_text(encoding="utf-8"))
    adjudication = json.loads(
        (RESULT / "pre_science_adjudication.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/parallax_execution_freeze_v114.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == "CLEANED_AFTER_FAILURE"
    assert state["phase"] == "TERMINAL_CLEANUP_CONFIRMED"
    assert state["job_released"] is state["s3_durable"] is False
    assert state["cleanup"]["reason"] == "job_not_released"
    assert state["cleanup"]["confirmation"]["consecutive_absences"] == 3
    assert adjudication["status"] == "VOID_PROVIDER_BOOTSTRAP_BEFORE_JOB_RELEASE"
    assert adjudication["scientific_outcome"] == "UNSEEN"
    assert adjudication["firewall"] == {
        "job_released": False,
        "signed_model_urls_released": 0,
        "model_objects_downloaded": 0,
        "model_bytes_downloaded": 0,
        "model_loads": 0,
        "processor_calls_repeated": 0,
        "train_items_read": 0,
        "discovery_items_read": 0,
        "confirmation_items_read": 0,
        "training_steps": 0,
        "gpu_science_operations": 0,
        "science_artifacts_expected": False,
    }
    assert adjudication["s3_result_prefix"]["versions"] == 0
    assert adjudication["cleanup"]["active_instances"] == 0
    assert adjudication["cleanup"]["active_volumes"] == 0
    assert not any(freeze["authorization"].values())
    assert freeze["state"]["active_vast_instances"] == 0
    assert freeze["state"]["active_vast_volumes"] == 0


def test_failed_provider_machine_is_in_the_exact_active_quarantine() -> None:
    spec = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/"
            "parallax_pg1_primary_interface_control_s3_launch_v2.json"
        ).read_text(encoding="utf-8")
    )
    spec["bound_artifacts"]["machine_quarantine"] = {
        "path": "studies/lineage/cross-lineage/config/parallax_h100_machine_quarantine_v3.json",
        "sha256": "b97b3bbde5b07d3321958b95e26d29ff84ed7a65e60d6dbbd65f94bdadf238c6",
    }
    result = require_machine_quarantine(ROOT, spec)
    assert result["status"] == "GREEN_FROZEN_MODE_SPECIFIC_REGISTRY"
    assert 58747 in result["active_machine_ids"]
    assert result["entries"] == 6


def test_second_provider_void_is_pre_science_immediately_cleaned_and_paid_off() -> None:
    state = json.loads((RESULT_V4 / "launch-state.json").read_text(encoding="utf-8"))
    telemetry = json.loads(
        (RESULT_V4 / "provider-fatal-telemetry.json").read_text(encoding="utf-8")
    )
    adjudication = json.loads(
        (RESULT_V4 / "pre_science_adjudication.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/parallax_execution_freeze_v118.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == "CLEANED_AFTER_FAILURE"
    assert state["job_released"] is False
    assert telemetry["fatal_signature"] == "OCI runtime create failed"
    assert telemetry["controller_action"].startswith("IMMEDIATE_DESTROY")
    assert adjudication["scientific_outcome"] == "UNSEEN"
    assert adjudication["worker"]["vast_persistent_volume"] is False
    assert adjudication["worker"]["disposable_scratch_gb"] == 80
    assert not any(adjudication["firewall"].values())
    assert adjudication["firewall"]["job_released"] is False
    assert adjudication["firewall"]["model_bytes_downloaded"] == 0
    assert adjudication["firewall"]["training_steps"] == 0
    assert adjudication["s3_result_prefix"]["versions"] == 0
    assert adjudication["cleanup"]["active_instances"] == 0
    assert adjudication["cleanup"]["active_volumes"] == 0
    assert not any(freeze["authorization"].values())
    assert freeze["state"]["active_vast_instances"] == 0
    assert freeze["state"]["active_vast_volumes"] == 0


def test_both_primary_provider_machines_are_in_the_current_quarantine() -> None:
    spec = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/"
            "parallax_pg1_primary_interface_control_s3_launch_v4.json"
        ).read_text(encoding="utf-8")
    )
    spec["bound_artifacts"]["machine_quarantine"] = {
        "path": "studies/lineage/cross-lineage/config/parallax_h100_machine_quarantine_v4.json",
        "sha256": "a28727a48e0f61e287010e7c362ead3074363b39925aa364eb6f89fe90b923ba",
    }
    result = require_machine_quarantine(ROOT, spec)
    assert result["status"] == "GREEN_FROZEN_MODE_SPECIFIC_REGISTRY"
    assert {58747, 59352}.issubset(result["active_machine_ids"])
    assert result["entries"] == 7


def test_provider_void_teaching_artifact_is_exact() -> None:
    path = ROOT / (
        "studies/lineage/cross-lineage/visuals/parallax_pg1_provider_void_v1.svg"
    )
    assert sha256_file(path) == "31ba111c26c6f9487a4735605082eb178807e3548b729b497613d5ed2cd7e29e"


def test_two_provider_voids_teaching_artifact_is_exact() -> None:
    path = ROOT / (
        "studies/lineage/cross-lineage/visuals/parallax_pg1_two_provider_voids_v1.svg"
    )
    assert sha256_file(path) == "2df48f88c69f5cde4fbe5e803127ba99e86bc5be44b653e298a629f3c0319277"
