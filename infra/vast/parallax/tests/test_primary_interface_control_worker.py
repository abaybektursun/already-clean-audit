from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra.vast.parallax import primary_interface_control_worker as worker


def _contract() -> dict[str, object]:
    key = f"parallax/results/{worker.RUN_ID}/terminal.json"
    url = (
        "https://pg-lineage-769063704696.s3.us-east-1.amazonaws.com/"
        f"{key}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=1800&"
        "X-Amz-SignedHeaders=host%3Bx-amz-server-side-encryption&"
        f"X-Amz-Signature={'0' * 64}"
    )
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
        "maximum_result_bytes": worker.legacy.MAXIMUM_RESULT_BYTES,
        "result_key": key,
        "result_put_url": url,
        "archive_durable_action": "destroy",
        "pre_science_terminal_durable_action": "destroy",
        "science_started_archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
        "automatic_retry": False,
        "environment": {},
    }


def test_primary_worker_contract_is_exact_and_restores_legacy() -> None:
    contract = _contract()
    before = (worker.legacy.RUN_ID, worker.legacy.EXPECTED_COMMAND)
    assert worker.validate_contract(contract) is contract
    assert (worker.legacy.RUN_ID, worker.legacy.EXPECTED_COMMAND) == before


def test_primary_worker_restores_legacy_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract_path = tmp_path / "worker.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    before = (worker.legacy.RUN_ID, worker.legacy.EXPECTED_COMMAND)

    def planted_failure(_path: Path):
        assert worker.legacy.RUN_ID == worker.RUN_ID
        assert worker.legacy.EXPECTED_COMMAND == worker.EXPECTED_COMMAND
        raise RuntimeError("planted worker failure")

    monkeypatch.setattr(worker.legacy, "run", planted_failure)
    with pytest.raises(RuntimeError, match="planted worker failure"):
        worker.run(contract_path)
    assert (worker.legacy.RUN_ID, worker.legacy.EXPECTED_COMMAND) == before
