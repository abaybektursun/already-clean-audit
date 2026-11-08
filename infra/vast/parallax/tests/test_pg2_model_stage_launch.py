from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from infra.aws_parallax import pg2_model_stage_launch as stage


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / stage.SPEC_PATH
USER_DATA = ROOT / stage.USER_DATA_PATH


def test_pg2_stage_spec_is_exact_cpu_only_and_s3_durable() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    assert spec["source"]["repo"] == "google/paligemma2-3b-pt-224"
    assert spec["source"]["revision"] == "96eeb174da13ca1a2b247e4d0867436296c36420"
    assert spec["expected_file_count"] == 9
    assert spec["expected_total_bytes"] == 6_099_504_596
    assert spec["compute"]["gpu"] is False
    assert spec["compute"]["instance_type"] == "t3a.small"
    assert spec["compute"]["automatic_retry"] is False
    assert spec["destination"]["provider"] == "amazon_s3"
    assert spec["authorizations"]["paid_action"] is False


def test_pg2_stage_user_data_is_syntax_green_bounded_and_token_free() -> None:
    subprocess.run(["bash", "-n", str(USER_DATA)], check=True)
    text = USER_DATA.read_text(encoding="utf-8")
    assert "shutdown -h +35" in text
    assert "WAITING_FOR_EXTERNAL_REAPER_RELEASE" in text
    assert "RESULTS_DURABLE file_count=9 total_bytes=6099504596" in text
    assert "AWS_MAX_ATTEMPTS=1" in text
    assert "--checksum-sha256" in text
    assert "docker" not in text.lower()
    assert "vastai" not in text.lower()
    assert stage.TOKEN_PATH.read_text(encoding="utf-8").strip() not in text
    assert text.count("stage_one ") == 9


def test_pg2_stage_refuses_compute_or_file_drift(tmp_path: Path) -> None:
    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    raw["compute"]["gpu"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(stage.StageError, match="compute contract"):
        stage.validate_spec(ROOT, changed)

    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    raw["files"][0]["bytes"] += 1
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(stage.StageError, match="byte total"):
        stage.validate_spec(ROOT, changed)


def test_rendered_user_data_contains_parameter_name_but_never_token() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    rendered = stage._render_user_data(ROOT, spec)
    assert spec["identity_and_access"]["temporary_ssm_parameter"] in rendered
    assert "__RUN_ID__" not in rendered
    assert "__SSM_PARAMETER__" not in rendered
    assert stage.TOKEN_PATH.read_text(encoding="utf-8").strip() not in rendered


def test_no_create_records_zero_mutation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage, "_account_state", lambda: {
        "PARALLAX_AWS_instances": [],
        "PARALLAX_AWS_schedules": [],
        "Vast_instances": [],
        "Vast_volumes": [],
    })
    monkeypatch.setattr(
        stage,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(stage, "_head_optional", lambda key: None)
    monkeypatch.setattr(stage, "_require_stage_prefix_never_used", lambda: None)
    monkeypatch.setattr(stage, "_dry_run_instance", lambda spec, path: None)
    monkeypatch.setattr(
        stage,
        "_head_huggingface",
        lambda spec: [
            {"path": row["path"], "http_status": 302, "revision_exact": True, "bytes_exact": True,
             "linked_sha_exact": True, "body_downloaded": False}
            for row in spec["files"]
        ],
    )

    def aws_json(*arguments, **kwargs):
        if arguments[:2] == ("ssm", "get-parameter"):
            return None
        if arguments[:2] == ("iam", "list-role-policies"):
            return {"PolicyNames": ["parallax-cloud-builder"]}
        raise AssertionError(arguments)

    monkeypatch.setattr(stage, "_aws_json", aws_json)
    output = tmp_path / "no-create.json"
    packet = stage.no_create(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PG2_MODEL_STAGE_NO_CREATE"
    assert packet["absent_key_count"] == 18
    assert packet["paid_authorization"] is False
    assert set(packet["mutation"].values()) == {0}


def test_create_arguments_are_cpu_spot_ephemeral_and_secret_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    rendered = tmp_path / "user-data.sh"
    rendered.write_text(stage._render_user_data(ROOT, spec), encoding="utf-8")
    captured = {}

    def fake_aws_json(*arguments, **kwargs):
        captured["arguments"] = arguments
        return "i-test123"

    monkeypatch.setattr(stage, "_aws_json", fake_aws_json)
    assert stage._create_instance(spec, rendered) == "i-test123"
    arguments = captured["arguments"]
    joined = " ".join(arguments)
    assert "t3a.small" in arguments
    assert "MarketType=spot" in joined
    assert "InstanceInterruptionBehavior=terminate" in joined
    assert "DeleteOnTermination" in joined
    assert "HttpTokens=required" in joined
    assert stage.TOKEN_PATH.read_text(encoding="utf-8").strip() not in joined
    assert "H100" not in joined


def test_final_keys_are_content_addressed() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    for row in spec["files"]:
        assert stage._final_key(row) == (
            f"parallax/input-staging/sha256/{row['sha256']}/{row['path']}"
        )


def test_ssm_secret_uses_mode_0600_file_not_process_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    token = stage.TOKEN_PATH.read_text(encoding="utf-8").strip()
    observed = {}

    def fake_aws(*arguments, **kwargs):
        joined = " ".join(arguments)
        assert token not in joined
        uri = arguments[arguments.index("--cli-input-json") + 1]
        assert uri.startswith("file://")
        path = Path(uri.removeprefix("file://"))
        observed["path"] = path
        observed["mode"] = path.stat().st_mode & 0o777
        observed["document"] = json.loads(path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess([], 0, "{}", "")

    monkeypatch.setattr(stage, "_aws", fake_aws)
    stage._put_parameter(spec)
    assert observed["mode"] == 0o600
    assert observed["document"]["Value"] == token
    assert not observed["path"].exists()
