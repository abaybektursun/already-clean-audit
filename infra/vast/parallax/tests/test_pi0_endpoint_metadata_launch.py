from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_endpoint_metadata_launch as launch


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / launch.SPEC_PATH


def test_spec_and_s3_candidate_graph_are_exact() -> None:
    spec, rows = launch.validate_spec(ROOT, SPEC)
    assert spec["compute"]["gpu"] is False
    assert spec["compute"]["root_disk_gb"] == 8
    assert len(rows) == 16
    assert sum(row["lineage"] == "PG1" for row in rows) == 12
    assert sum(row["lineage"] == "pi0" for row in rows) == 4
    assert all(row["key"] == launch._candidate_key(row) for row in rows)
    assert all(row["version_id"] for row in rows if row["lineage"] == "PG1")
    assert all(row["version_id"] is None for row in rows if row["lineage"] == "pi0")


def test_spec_mutation_cannot_expand_compute(tmp_path: Path) -> None:
    changed = json.loads(SPEC.read_text(encoding="utf-8"))
    changed["compute"]["gpu"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(launch.LaunchError):
        launch.validate_spec(ROOT, path)


def test_rendered_user_data_is_small_inert_and_syntax_green(tmp_path: Path) -> None:
    rendered = launch._render_user_data(ROOT, SPEC)
    assert len(rendered.encode()) <= 15_500
    assert "__RUN_ID__" not in rendered
    assert "__SPEC_SHA256__" not in rendered
    assert "__PROBE_SHA256__" not in rendered
    assert "dnf install" not in rendered
    assert "pip install" not in rendered
    assert "docker" not in rendered.lower()
    assert "shutdown -h +8" in rendered
    assert "put_required \"$STATUS\"" in rendered
    path = tmp_path / "user-data.sh"
    path.write_text(rendered, encoding="utf-8")
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_aws_wrapper_always_appends_literal_fuelos_profile_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(arguments, **kwargs):  # noqa: ANN001
        observed["arguments"] = arguments
        return subprocess.CompletedProcess(arguments, 0, "{}", "")

    monkeypatch.setattr(launch, "_run", fake_run)
    launch._aws("sts", "get-caller-identity")
    assert observed["arguments"][-4:] == ["--profile", "fuelos", "--region", "us-east-1"]


def test_instance_arguments_are_spot_cpu_ephemeral_and_imdsv2(tmp_path: Path) -> None:
    spec, _ = launch.validate_spec(ROOT, SPEC)
    user_data = tmp_path / "user-data.sh"
    user_data.write_text("#!/bin/bash\n", encoding="utf-8")
    arguments = launch._instance_arguments(spec, user_data)
    joined = " ".join(arguments)
    assert "t3a.small" in arguments
    assert "MarketType=spot" in joined
    assert "InstanceInterruptionBehavior=terminate" in joined
    assert "DeleteOnTermination" in joined
    assert "VolumeSize\":8" in joined
    assert "HttpTokens=required" in joined
    assert "H100" not in joined


def test_no_create_is_zero_mutation_and_head_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zero = {
        "PARALLAX_AWS_instances": [],
        "PARALLAX_AWS_schedules": [],
        "Vast_instances": [],
        "Vast_volumes": [],
    }
    monkeypatch.setattr(launch, "_account_state", lambda: zero)
    monkeypatch.setattr(
        launch,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(
        launch,
        "_require_instance_role_prefix",
        lambda: {"compatible": True, "probe_prefix": launch.PREFIX, "policy_mutation": False},
    )
    monkeypatch.setattr(launch, "_require_prefix_never_used", lambda: None)
    monkeypatch.setattr(
        launch,
        "_head_inputs",
        lambda rows: {
            "operation": "HEAD_only_no_get_object",
            "rows": [{**row, "present": row["lineage"] == "PG1", "body_read": False} for row in rows],
            "candidate_count": len(rows),
            "present_count": 12,
            "PG1_present_count": 12,
            "pi0_present_count": 0,
            "body_bytes_read": 0,
        },
    )
    monkeypatch.setattr(launch, "_dry_run_instance", lambda spec, path: None)
    output = tmp_path / "no-create.json"
    packet = launch.no_create(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PI0_ENDPOINT_METADATA_P1_NO_CREATE"
    assert packet["paid_authorization"] is False
    assert set(packet["mutation"].values()) == {0}
    assert packet["s3_input_inventory"]["operation"] == "HEAD_only_no_get_object"
    assert packet["user_data"]["bash_syntax_green"] is True


def test_head_candidate_absence_is_recorded_not_promoted(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "lineage": "pi0",
        "repo": "lerobot/pi0_base",
        "revision": "x" * 40,
        "path": "config.json",
        "bytes": 12,
        "sha256": "a" * 64,
        "key": "parallax/input-staging/sha256/" + "a" * 64 + "/config.json",
        "version_id": None,
        "historical_state": "cache_only",
    }
    monkeypatch.setattr(
        launch,
        "_aws",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 254, "", "An error occurred (404) Not Found"),
    )
    result = launch._head_candidate(row)
    assert result["present"] is False
    assert result["exact"] is None
    assert result["body_read"] is False


def test_head_candidate_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "lineage": "PG1",
        "repo": "google/paligemma-3b-pt-224",
        "revision": "x" * 40,
        "path": "config.json",
        "bytes": 12,
        "sha256": "a" * 64,
        "key": "key",
        "version_id": "version",
        "historical_state": "durable",
    }
    payload = {"ContentLength": 13, "ServerSideEncryption": "AES256", "Metadata": {"sha256": "a" * 64}}
    monkeypatch.setattr(
        launch,
        "_aws",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""),
    )
    with pytest.raises(launch.LaunchError, match="mismatched identity"):
        launch._head_candidate(row)


def test_release_packet_is_exact_and_only_after_external_reaper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed = {}

    def fake_put(path: Path, key: str):
        observed["packet"] = json.loads(path.read_text(encoding="utf-8"))
        observed["key"] = key
        return {"key": key, "version_id": "v1"}

    monkeypatch.setattr(launch, "_put_exact", fake_put)
    result = launch._put_release("i-123", "2026-07-19T00:00:00+00:00")
    assert result["version_id"] == "v1"
    assert observed["packet"]["reaper_armed"] is True
    assert observed["packet"]["instance_id"] == "i-123"
    assert observed["packet"]["automatic_retry"] is False


def test_instance_role_prefix_gate_rejects_the_v1_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = {
        "PolicyDocument": {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"],
                    "Resource": (
                        "arn:aws:s3:::pg-lineage-769063704696/parallax/metadata-probes/*"
                    ),
                },
                {
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": "arn:aws:s3:::pg-lineage-769063704696",
                    "Condition": {"StringLike": {"s3:prefix": "parallax/metadata-probes/*"}},
                },
            ]
        }
    }
    monkeypatch.setattr(launch, "_aws_json", lambda *args, **kwargs: policy)
    with pytest.raises(launch.LaunchError, match="cannot access"):
        launch._require_instance_role_prefix()


def test_instance_role_prefix_gate_accepts_exact_existing_policy_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {
        "PolicyDocument": {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"],
                    "Resource": "arn:aws:s3:::pg-lineage-769063704696/parallax/cloud-builder/*",
                },
                {
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": "arn:aws:s3:::pg-lineage-769063704696",
                    "Condition": {"StringLike": {"s3:prefix": "parallax/cloud-builder/*"}},
                },
            ]
        }
    }
    monkeypatch.setattr(launch, "_aws_json", lambda *args, **kwargs: policy)
    result = launch._require_instance_role_prefix()
    assert result["compatible"] is True
    assert result["policy_mutation"] is False
    assert result["probe_prefix"] == launch.PREFIX


def test_launch_orders_reaper_before_release_and_always_returns_account_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    zero = {
        "PARALLAX_AWS_instances": [],
        "PARALLAX_AWS_schedules": [],
        "Vast_instances": [],
        "Vast_volumes": [],
    }
    monkeypatch.setattr(launch, "_validate_freeze", lambda *args: {})
    monkeypatch.setattr(launch, "_account_state", lambda: zero)
    monkeypatch.setattr(
        launch,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(
        launch,
        "_require_instance_role_prefix",
        lambda: {"compatible": True, "probe_prefix": launch.PREFIX, "policy_mutation": False},
    )
    monkeypatch.setattr(launch, "_require_prefix_never_used", lambda: None)
    monkeypatch.setattr(
        launch,
        "_head_inputs",
        lambda rows: {"operation": "HEAD_only_no_get_object", "rows": [], "body_bytes_read": 0},
    )
    monkeypatch.setattr(
        launch,
        "_stage_inputs",
        lambda *args: events.append("stage") or [{"key": "spec"}, {"key": "probe"}],
    )
    monkeypatch.setattr(launch, "_with_rendered_user_data", lambda *args: tmp_path / "temporary-user-data")
    (tmp_path / "temporary-user-data").write_text("x", encoding="utf-8")
    monkeypatch.setattr(launch, "_create_instance", lambda *args: events.append("create") or "i-test")
    monkeypatch.setattr(
        launch, "_arm_reaper", lambda *args: events.append("arm_reaper") or "2026-07-19T00:00:00+00:00"
    )
    monkeypatch.setattr(
        launch,
        "_put_release",
        lambda *args: events.append("release") or {"key": "release", "version_id": "v"},
    )
    monkeypatch.setattr(
        launch,
        "_wait_status",
        lambda *args: events.append("wait") or {"status": "GREEN_P1_METADATA_PROBE_COMPLETE"},
    )
    result = {
        "robot_positive_control": {"canonical_repo": "lerobot/pi0_libero_finetuned_v044", "revision": "1" * 40},
        "hf_libero_distribution": {"selected_version": "0.1.6"},
        "hf_libero_source_tag": {"commit": "2" * 40},
        "unresolved_fields": [],
        "network": {"request_count": 10, "total_bytes": 1000},
    }
    monkeypatch.setattr(
        launch,
        "_verify_outputs",
        lambda *args: events.append("verify") or (result, {"result": {"version_id": "v"}}),
    )
    monkeypatch.setattr(launch, "_terminate", lambda instance: events.append(f"terminate:{instance}"))
    monkeypatch.setattr(launch, "_cleanup_schedule", lambda: events.append("cleanup_schedule"))
    output = tmp_path / "out"
    packet = launch.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", output)
    assert events.index("arm_reaper") < events.index("release") < events.index("wait")
    assert events[-2:] == ["terminate:i-test", "cleanup_schedule"]
    assert packet["status"] == "GREEN_PI0_ENDPOINT_METADATA_P1_DURABLE"
    assert packet["P2_licensed"] is False
    assert (output / "receipt.json").is_file()
    assert json.loads((output / "terminal-account.json").read_text()) == zero


def test_freeze_cannot_authorize_gpu_or_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "status": "FROZEN_ONE_SHOT_PI0_METADATA_P1_AWS_CPU",
        "run_id": launch.RUN_ID,
        "authorization": {
            "paid_action": True,
            "AWS_CPU_create": True,
            "Vast_create": False,
            "H100": True,
            "model_or_checkpoint_body_read": False,
            "package_or_simulator_body_read": False,
            "dataset_read": False,
            "model_load": False,
            "training": False,
            "rollout": False,
            "confirmation_items_read": False,
            "automatic_retry": False,
        },
        "spec": {"sha256": launch._sha256(SPEC)},
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(launch.LaunchError, match="freeze is absent"):
        launch._validate_freeze(ROOT, SPEC, path)
