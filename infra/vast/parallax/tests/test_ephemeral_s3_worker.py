from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from infra.vast.parallax.ephemeral_s3_worker import (
    WorkerError,
    run,
    validate_contract,
    validate_result_url,
)
from infra.vast.parallax.s3_presign_put import presign_result_put
from datetime import datetime, timezone


def _url(key: str = "parallax/results/rehearsal-20260718/terminal.json") -> str:
    return presign_result_put(
        access_key="ASIAEXAMPLE",
        secret_key="example-secret",
        session_token="example-token",
        bucket="pg-lineage-769063704696",
        key=key,
        region="us-east-1",
        now=datetime(2026, 7, 18, 10, 30, tzinfo=timezone.utc),
        expires_seconds=900,
    )


def _contract(root: Path) -> dict[str, object]:
    key = "parallax/results/rehearsal-20260718/terminal.json"
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "run_id": "rehearsal-20260718",
        "stage": "ephemeral_s3_rehearsal",
        "instance_id": 123,
        "runtime_root": str(root),
        "job_output_path": str(root / "job-result.json"),
        "job_log_path": str(root / "job.log"),
        "terminal_path": str(root / "terminal.json"),
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": [
            "python3",
            "-c",
            f"import json; json.dump({{'green': True}}, open({str(root / 'job-result.json')!r}, 'w'))",
        ],
        "maximum_job_seconds": 10,
        "maximum_result_bytes": 1024 * 1024,
        "result_key": key,
        "result_put_url": _url(key),
        "post_upload_action": "destroy",
        "upload_failure_action": "stop",
        "environment": {},
    }


def test_validate_result_url_rejects_a_different_key():
    with pytest.raises(WorkerError):
        validate_result_url(_url(), result_key="parallax/results/other-run/terminal.json")


def test_contract_rejects_artifacts_outside_runtime_root(tmp_path: Path):
    contract = _contract(tmp_path)
    contract["terminal_path"] = "/tmp/escaped-terminal.json"
    with pytest.raises(WorkerError):
        validate_contract(contract)


def test_green_job_uploads_before_destroy(tmp_path: Path):
    contract = _contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    events = []

    def fake_put(**kwargs):
        events.append(("put", Path(kwargs["terminal_path"]).is_file()))
        return {"http_status": 200, "version_id": "version-one", "bytes": 1}

    def fake_action(**kwargs):
        events.append((kwargs["action"], True))

    with patch("infra.vast.parallax.ephemeral_s3_worker.put_terminal", side_effect=fake_put), patch(
        "infra.vast.parallax.ephemeral_s3_worker.instance_action", side_effect=fake_action
    ):
        packet = run(contract_path)
    assert packet["status"] == "GREEN_JOB_TERMINAL"
    assert events == [("put", True), ("destroy", True)]
    durable = json.loads((tmp_path / "terminal.json").read_text(encoding="utf-8"))
    assert durable["job"]["result"]["payload"] == {"green": True}


def test_upload_failure_stops_and_preserves_terminal(tmp_path: Path):
    contract = _contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    actions = []

    with patch(
        "infra.vast.parallax.ephemeral_s3_worker.put_terminal",
        side_effect=WorkerError("planted upload failure"),
    ), patch(
        "infra.vast.parallax.ephemeral_s3_worker.instance_action",
        side_effect=lambda **kwargs: actions.append(kwargs["action"]),
    ):
        with pytest.raises(WorkerError, match="planted upload failure"):
            run(contract_path)
    assert actions == ["stop"]
    assert (tmp_path / "terminal.json").is_file()
