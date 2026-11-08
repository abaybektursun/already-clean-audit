from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_endpoint_artifact_launch as launch
from infra.aws_parallax import pi0_endpoint_metadata_launch as lifecycle


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / launch.SPEC_PATH


def zero_account() -> dict:
    return {
        "PARALLAX_AWS_instances": [],
        "PARALLAX_AWS_schedules": [],
        "Vast_instances": [],
        "Vast_volumes": [],
    }


def test_spec_binds_exact_compute_durability_and_historical_amendment() -> None:
    cfg = launch.validate_spec(ROOT, SPEC)
    assert cfg["compute"]["gpu"] is False
    assert cfg["durability"]["prefix"] == launch.PREFIX
    assert cfg["bound_artifacts"]["execution_safety_amendment"]["path"].endswith(
        "parallax_execution_safety_amendment_v2.yaml"
    )
    assert cfg["authorizations"]["paid_action"] is False


def test_temporary_lifecycle_profile_restores_historical_launcher_globals() -> None:
    before = {
        "RUN_ID": lifecycle.RUN_ID,
        "SPEC_PATH": lifecycle.SPEC_PATH,
        "PROBE_PATH": lifecycle.PROBE_PATH,
        "USER_DATA_PATH": lifecycle.USER_DATA_PATH,
        "SCHEDULE_NAME": lifecycle.SCHEDULE_NAME,
        "PREFIX": lifecycle.PREFIX,
    }
    with launch._lifecycle_profile():
        assert lifecycle.RUN_ID == launch.RUN_ID
        assert lifecycle.PREFIX == launch.PREFIX
        assert lifecycle.PROBE_PATH == launch.PROBE_PATH
    assert {name: getattr(lifecycle, name) for name in before} == before


def test_no_create_is_dry_and_restores_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    state = zero_account()
    historical_run_id = lifecycle.RUN_ID
    monkeypatch.setattr(lifecycle, "_account_state", lambda: state)
    monkeypatch.setattr(lifecycle, "_require_zero", lambda value: events.append("zero"))
    monkeypatch.setattr(
        lifecycle,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(
        lifecycle,
        "_require_instance_role_prefix",
        lambda: {"compatible": True, "probe_prefix": lifecycle.PREFIX, "policy_mutation": False},
    )
    monkeypatch.setattr(lifecycle, "_require_prefix_never_used", lambda: events.append("unused"))
    rendered = tmp_path / "rendered.sh"
    rendered.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_with_rendered_user_data", lambda *args: rendered)
    monkeypatch.setattr(lifecycle, "_run", lambda *args, **kwargs: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(lifecycle, "_dry_run_instance", lambda *args: events.append("dry-run"))
    monkeypatch.setattr(launch, "_render_user_data", lambda *args: "#!/bin/bash\n")
    output = tmp_path / "no-create.json"
    packet = launch.no_create(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PI0_ENDPOINT_ARTIFACT_P1B_NO_CREATE"
    assert packet["mutation"]["S3_objects_written"] == 0
    assert packet["mutation"]["external_body_bytes_read"] == 0
    assert events == ["zero", "unused", "dry-run"]
    assert lifecycle.RUN_ID == historical_run_id


def test_version_pinned_readback_requires_s3_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b"exact-result\n"
    sha = hashlib.sha256(body).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(sha)).decode()
    head = {
        "ContentLength": len(body),
        "ServerSideEncryption": "AES256",
        "VersionId": "exact-version",
        "ChecksumSHA256": checksum,
        "Metadata": {"sha256": sha},
    }

    def fake_aws(*args, **kwargs):  # noqa: ANN001
        if args[1] == "head-object":
            assert "--checksum-mode" in args
            return type(
                "R", (), {"returncode": 0, "stdout": json.dumps(head), "stderr": ""}
            )()
        assert args[1] == "get-object"
        assert "--version-id" in args
        assert args[args.index("--version-id") + 1] == "exact-version"
        Path(args[-1]).write_bytes(body)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(lifecycle, "_aws", fake_aws)
    output = tmp_path / "result.json"
    receipt = launch._download_versioned_small(
        "prefix/result.json", output, max_bytes=100, require_checksum=True
    )
    assert receipt["version_id"] == "exact-version"
    assert receipt["checksum_verified"] is True
    head["ChecksumSHA256"] = "wrong"
    with pytest.raises(launch.LaunchError, match="checksum mismatch"):
        launch._download_versioned_small(
            "prefix/result.json", tmp_path / "bad.json", max_bytes=100, require_checksum=True
        )


def test_launch_orders_reaper_before_release_then_terminates_and_proves_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    state = zero_account()
    monkeypatch.setattr(launch, "_validate_freeze", lambda *args: {})
    monkeypatch.setattr(lifecycle, "_account_state", lambda: state)
    monkeypatch.setattr(lifecycle, "_require_zero", lambda value: None)
    monkeypatch.setattr(
        lifecycle,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(
        lifecycle,
        "_require_instance_role_prefix",
        lambda: {"compatible": True, "probe_prefix": lifecycle.PREFIX, "policy_mutation": False},
    )
    monkeypatch.setattr(lifecycle, "_require_prefix_never_used", lambda: None)
    monkeypatch.setattr(
        lifecycle, "_stage_inputs", lambda *args: events.append("stage") or [{"key": "spec"}, {"key": "probe"}]
    )
    rendered = tmp_path / "user-data.sh"
    rendered.write_text("x", encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_with_rendered_user_data", lambda *args: rendered)
    monkeypatch.setattr(lifecycle, "_create_instance", lambda *args: events.append("create") or "i-test")
    monkeypatch.setattr(
        lifecycle,
        "_arm_reaper",
        lambda *args: events.append("arm_reaper") or "2026-07-19T00:00:00+00:00",
    )
    monkeypatch.setattr(
        lifecycle,
        "_put_release",
        lambda *args: events.append("release") or {"key": "release", "version_id": "v"},
    )
    monkeypatch.setattr(
        lifecycle,
        "_wait_status",
        lambda *args: events.append("wait") or {"status": "GREEN_P1B_ARTIFACT_INSPECTION_COMPLETE"},
    )
    result = {
        "hf_libero_wheel": {
            "sha256": "1" * 64,
            "member_count": 10,
            "record_exact_coverage": True,
            "runtime_task_order_evidence": {"all_four_runtime_suite_orders_proven": True},
            "bddl_counts_by_suite": {"object": 10, "spatial": 10, "goal": 10, "long": 10},
        },
        "robot_control": {
            "preprocessor_stats": {"tensor_count": 4},
            "paired_stats_exactly_equal": True,
        },
        "unresolved_fields": [],
        "network": {"request_count": 9, "total_bytes": 3_194_781},
    }
    monkeypatch.setattr(
        launch,
        "_verify_outputs",
        lambda *args: events.append("verify") or (result, {"result": {"version_id": "v"}}),
    )
    monkeypatch.setattr(lifecycle, "_terminate", lambda instance: events.append(f"terminate:{instance}"))
    monkeypatch.setattr(lifecycle, "_cleanup_schedule", lambda: events.append("cleanup_schedule"))
    output = tmp_path / "output"
    packet = launch.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", output)
    assert events.index("arm_reaper") < events.index("release") < events.index("wait")
    assert events[-2:] == ["terminate:i-test", "cleanup_schedule"]
    assert packet["status"] == "GREEN_PI0_ENDPOINT_ARTIFACT_P1B_DURABLE"
    assert packet["artifact_result_summary"]["runtime_task_order_proven"] is True
    assert packet["P2_licensed"] is False
    assert json.loads((output / "terminal-account.json").read_text()) == state


def test_freeze_cannot_authorize_gpu_model_install_or_retry(tmp_path: Path) -> None:
    base_authorization = {
        "paid_action": True,
        "AWS_CPU_create": True,
        "Vast_create": False,
        "H100": False,
        "exact_wheel_body_read": True,
        "exact_small_control_sidecar_body_read": True,
        "model_or_checkpoint_body_read": False,
        "dataset_read": False,
        "package_install_or_import": False,
        "simulator_start": False,
        "model_load": False,
        "training": False,
        "rollout": False,
        "confirmation_items_read": False,
        "automatic_retry": False,
    }
    for field in ("H100", "model_or_checkpoint_body_read", "package_install_or_import", "automatic_retry"):
        authorization = dict(base_authorization)
        authorization[field] = True
        fake = {
            "status": "FROZEN_ONE_SHOT_PI0_ARTIFACT_P1B_AWS_CPU",
            "run_id": launch.RUN_ID,
            "authorization": authorization,
            "spec": {"sha256": launch._sha256(SPEC)},
        }
        path = tmp_path / f"freeze-{field}.json"
        path.write_text(json.dumps(fake), encoding="utf-8")
        with pytest.raises(launch.LaunchError, match="freeze is absent"):
            launch._validate_freeze(ROOT, SPEC, path)
