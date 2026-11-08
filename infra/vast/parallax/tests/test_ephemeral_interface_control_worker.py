from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import ephemeral_interface_control_worker as worker


def _result_url() -> str:
    key = f"parallax/results/{worker.RUN_ID}/terminal.json"
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Expires": "1800",
            "X-Amz-SignedHeaders": "host;x-amz-server-side-encryption",
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://pg-lineage-769063704696.s3.us-east-1.amazonaws.com/{key}?{query}"


def _contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "FROZEN_SELECTED_INTERFACE_WORKER",
        "run_id": worker.RUN_ID,
        "stage": "training_interface_control",
        "instance_id": 123,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "phase_path": "/workspace/parallax/phase.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": worker.EXPECTED_COMMAND,
        "maximum_job_seconds": 1200,
        "maximum_result_bytes": worker.MAXIMUM_RESULT_BYTES,
        "result_key": f"parallax/results/{worker.RUN_ID}/terminal.json",
        "result_put_url": _result_url(),
        "archive_durable_action": "destroy",
        "pre_science_terminal_durable_action": "destroy",
        "science_started_archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
        "automatic_retry": False,
        "environment": {},
    }


def _archive_payload() -> dict[str, object]:
    digest = hashlib.sha256(b"archive").hexdigest()
    return {
        "status": "GREEN_INTERFACE_CONTROL_ARCHIVE_DURABLE",
        "run_id": worker.RUN_ID,
        "artifact_proof": {
            "http_status": 201,
            "version_id": "version-1",
            "bytes": 1_500_000_000,
            "sha256": digest,
            "checksum_sha256_base64": base64.b64encode(bytes.fromhex(digest)).decode(),
            "server_side_encryption": "AES256",
            "key": f"parallax/results/{worker.RUN_ID}/artifacts.tar",
        },
    }


def test_worker_contract_is_exact_s3_only_and_phase_aware() -> None:
    contract = _contract()
    assert worker.validate_contract(contract) is contract
    contract["pre_science_terminal_durable_action"] = "stop"
    with pytest.raises(worker.InterfaceWorkerError):
        worker.validate_contract(contract)


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payload: dict[str, object] | None,
    science_started: bool,
    returncode: int,
    put_failure: bool = False,
) -> tuple[dict[str, object] | None, list[str]]:
    contract = _contract()
    contract.update(
        runtime_root=str(tmp_path),
        job_output_path=str(tmp_path / "job-result.json"),
        job_log_path=str(tmp_path / "job.log"),
        terminal_path=str(tmp_path / "terminal.json"),
        phase_path=str(tmp_path / "phase.json"),
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    (tmp_path / "phase.json").write_text(
        json.dumps(
            {
                "run_id": worker.RUN_ID,
                "phase": "SCIENCE_STARTED" if science_started else "SOURCE_STAGING",
                "science_started": science_started,
            }
        ),
        encoding="utf-8",
    )
    actions: list[str] = []
    monkeypatch.setattr(worker, "validate_contract", lambda row: row)
    def run_job(*_args, **_kwargs):
        if payload is not None:
            (tmp_path / "job-result.json").write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(worker.subprocess, "run", run_job)

    def put(**_kwargs):
        if put_failure:
            raise worker.base.WorkerError("planted terminal upload failure")
        actions.append("put")

    monkeypatch.setattr(worker.base, "put_terminal", put)
    monkeypatch.setattr(
        worker.base,
        "instance_action",
        lambda **kwargs: actions.append(str(kwargs["action"])),
    )
    if put_failure:
        with pytest.raises(worker.base.WorkerError, match="planted terminal upload failure"):
            worker.run(contract_path)
        return None, actions
    return worker.run(contract_path), actions


def test_archive_durable_uploads_terminal_then_destroys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet, actions = _run_case(
        monkeypatch,
        tmp_path,
        payload=_archive_payload(),
        science_started=True,
        returncode=0,
    )
    assert packet is not None and packet["status"] == "GREEN_JOB_TERMINAL"
    assert actions == ["put", "destroy"]


def test_exact_preflight_red_uploads_terminal_then_destroys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "status": "RED_TOKENIZER_PREFLIGHT_BEFORE_WEIGHT_HYDRATION",
        "run_id": worker.RUN_ID,
        "science_started": False,
    }
    packet, actions = _run_case(
        monkeypatch,
        tmp_path,
        payload=payload,
        science_started=False,
        returncode=2,
    )
    assert packet is not None and packet["status"] == "RED_PREFLIGHT_TERMINAL"
    assert actions == ["put", "destroy"]


def test_unknown_pre_science_failure_has_nothing_to_recover_and_destroys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet, actions = _run_case(
        monkeypatch,
        tmp_path,
        payload=None,
        science_started=False,
        returncode=1,
    )
    assert packet is not None and packet["status"] == "RED_JOB_TERMINAL"
    assert packet["durability_contract"]["pre_science"] is True
    assert actions == ["put", "destroy"]


def test_post_science_failure_stops_and_preserves_disposable_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet, actions = _run_case(
        monkeypatch,
        tmp_path,
        payload=None,
        science_started=True,
        returncode=1,
    )
    assert packet is not None and packet["status"] == "RED_JOB_TERMINAL"
    assert actions == ["put", "stop"]


def test_terminal_upload_failure_always_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet, actions = _run_case(
        monkeypatch,
        tmp_path,
        payload=_archive_payload(),
        science_started=True,
        returncode=0,
        put_failure=True,
    )
    assert packet is None
    assert actions == ["stop"]
