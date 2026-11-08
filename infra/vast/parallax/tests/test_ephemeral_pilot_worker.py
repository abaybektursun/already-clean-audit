from __future__ import annotations

import base64
import hashlib
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import ephemeral_pilot_worker as worker


def _result_url(run_id: str = worker.RUN_ID) -> str:
    key = f"parallax/results/{run_id}/terminal.json"
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Expires": "1800",
            "X-Amz-SignedHeaders": "host;x-amz-server-side-encryption",
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://pg-lineage-769063704696.s3.us-east-1.amazonaws.com/{key}?{query}"


def _contract(run_id: str = worker.RUN_ID) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "FROZEN_SELECTED_PILOT_WORKER",
        "run_id": run_id,
        "stage": "training_artifact_rehearsal",
        "instance_id": 123,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": worker.EXPECTED_COMMAND,
        "maximum_job_seconds": 900,
        "maximum_result_bytes": 1024 * 1024,
        "result_key": f"parallax/results/{run_id}/terminal.json",
        "result_put_url": _result_url(run_id),
        "archive_durable_action": "destroy",
        "archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
        "automatic_retry": False,
        "environment": {},
    }


def _result(*, exact: bool = True, run_id: str = worker.RUN_ID) -> dict[str, object]:
    digest = hashlib.sha256(b"archive").hexdigest()
    proof = {
        "http_status": 201,
        "version_id": "version-1",
        "bytes": 400_000_000,
        "sha256": digest,
        "checksum_sha256_base64": base64.b64encode(bytes.fromhex(digest)).decode(),
        "server_side_encryption": "AES256",
        "key": f"parallax/results/{run_id}/artifacts.tar",
    }
    if not exact:
        proof["key"] = "wrong"
    return {
        "present": True,
        "payload": {
            "status": "GREEN_SELECTED_PILOT_ARCHIVE_DURABLE",
            "run_id": run_id,
            "artifact_proof": proof,
        },
    }


def test_worker_contract_is_exact_and_fail_safe() -> None:
    contract = _contract()
    assert worker.validate_contract(contract) is contract


def test_worker_accepts_exact_pg2_supporting_range_profile() -> None:
    contract = _contract(worker.PG2_RUN_ID)
    assert worker.validate_contract(contract) is contract


def test_worker_accepts_exact_pg2_v2_fresh_identity() -> None:
    contract = _contract(worker.PG2_V2_RUN_ID)
    assert worker.validate_contract(contract) is contract


def test_worker_accepts_only_exact_archive_proof() -> None:
    assert worker._archive_proof(_result(), run_id=worker.RUN_ID) is not None
    assert worker._archive_proof(_result(exact=False), run_id=worker.RUN_ID) is None
    assert worker._archive_proof({"present": False}, run_id=worker.RUN_ID) is None
    assert worker._archive_proof(
        _result(run_id=worker.PG2_RUN_ID), run_id=worker.PG2_RUN_ID
    ) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_job_seconds", 901),
        ("archive_not_durable_action", "destroy"),
        ("automatic_retry", True),
    ],
)
def test_worker_rejects_timeout_data_loss_or_retry_drift(field: str, value: object) -> None:
    contract = _contract()
    contract[field] = value
    with pytest.raises((worker.PilotWorkerError, worker.base.WorkerError)):
        worker.validate_contract(contract)
