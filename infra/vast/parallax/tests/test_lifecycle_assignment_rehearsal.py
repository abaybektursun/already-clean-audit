from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from infra.vast.parallax import lifecycle_assignment_rehearsal as rehearsal


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_lifecycle_assignment_rehearsal_v1.json"
)


def _minimal_spec() -> dict[str, object]:
    return {
        "guardian": rehearsal._role_contract("guardian"),
        "worker": rehearsal._role_contract("worker"),
        "output": {"terminal_path": "terminal.json"},
        "result": {
            "key": f"parallax/control-plane/{rehearsal.RUN_ID}/terminal.json",
            "maximum_bytes": 262_144,
        },
    }


def test_cheap_live_roles_are_non_h100_ephemeral_and_provider_bounded() -> None:
    guardian = rehearsal._role_contract("guardian")
    worker = rehearsal._role_contract("worker")
    assert guardian["gpu_kind"] == "non_h100_mechanical_guard"
    assert worker["gpu_kind"] == "non_h100_mechanical_worker"
    assert guardian["disk_gb"] == worker["disk_gb"] == 10
    assert guardian["hard_destroy_seconds"] == worker["hard_destroy_seconds"] == 300
    assert guardian["price_ceiling_dph"] == worker["price_ceiling_dph"] == 0.12
    assert worker["launch_mode"] == "stock_direct_ssh"


def test_consumed_cheap_live_spec_refuses_post_stage_retry_transport_drift() -> None:
    with pytest.raises(rehearsal.AssignmentRehearsalError, match="transport hash mismatch"):
        rehearsal.validate_spec(ROOT, SPEC)


def test_cheap_live_frozen_contract_was_exact_and_aws_only() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["authorization"]["h100"] is False
    assert spec["storage"]["durable_provider"] == "amazon_s3"
    assert spec["storage"]["vast_persistent_volumes"] == "FORBIDDEN"
    assert spec["automatic_paid_retry"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(automatic_paid_retry=True),
        lambda row: row["authorization"].update(h100=True),
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["worker"].update(hard_destroy_seconds=3600),
    ],
)
def test_frozen_cheap_live_spec_refuses_retry_h100_storage_or_deadman_drift(
    mutation, tmp_path: Path
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(spec)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(rehearsal.AssignmentRehearsalError):
        rehearsal.validate_spec(ROOT, candidate)


def test_cheap_live_create_construction_contains_no_persistent_storage() -> None:
    spec = _minimal_spec()
    guardian = rehearsal._construction(spec, role="guardian")
    worker = rehearsal._construction(spec, role="worker")
    for row in (guardian, worker):
        rendered = " ".join(row["arguments"])
        assert "--disk 10" in rendered
        assert "--volume" not in rendered
        assert "--link-volume" not in rendered
        assert row["persistent_storage"] is False


def test_fresh_s3_key_requires_zero_versions_and_zero_delete_markers(monkeypatch) -> None:
    monkeypatch.setattr(rehearsal, "_aws_json", lambda *args: {})
    rehearsal._result_absent("fresh")
    monkeypatch.setattr(
        rehearsal,
        "_aws_json",
        lambda *args: {"Versions": [{"Key": "used", "VersionId": "one"}]},
    )
    with pytest.raises(rehearsal.AssignmentRehearsalError, match="not fresh"):
        rehearsal._result_absent("used")


def test_small_terminal_publish_requires_exact_versioned_s3_readback(
    monkeypatch, tmp_path: Path
) -> None:
    spec = _minimal_spec()
    body_holder: dict[str, bytes | str] = {}

    def aws(*arguments: str):
        operation = arguments[1]
        if operation == "put-object":
            source = Path(arguments[arguments.index("--body") + 1])
            body = source.read_bytes()
            body_holder["body"] = body
            digest = hashlib.sha256(body).hexdigest()
            body_holder["digest"] = digest
            return {
                "VersionId": "exact-version",
                "ChecksumSHA256": base64.b64encode(bytes.fromhex(digest)).decode("ascii"),
            }
        if operation == "head-object":
            body = body_holder["body"]
            digest = body_holder["digest"]
            assert isinstance(body, bytes) and isinstance(digest, str)
            return {
                "VersionId": "exact-version",
                "ContentLength": len(body),
                "ServerSideEncryption": "AES256",
                "ChecksumSHA256": base64.b64encode(bytes.fromhex(digest)).decode("ascii"),
                "Metadata": {"sha256": digest},
            }
        if operation == "get-object":
            destination = Path(arguments[-1])
            body = body_holder["body"]
            assert isinstance(body, bytes)
            destination.write_bytes(body)
            return {"VersionId": "exact-version", "ServerSideEncryption": "AES256"}
        raise AssertionError(arguments)

    monkeypatch.setattr(rehearsal, "_aws_json", aws)
    proof = rehearsal._publish_terminal(
        tmp_path,
        spec,
        {"schema_version": 1, "run_id": rehearsal.RUN_ID, "status": "GREEN"},
    )
    assert proof["status"] == "GREEN_EXACT_VERSIONED_S3_READBACK"
    assert proof["version_id"] == "exact-version"
    assert proof["bytes"] == len(body_holder["body"])


def test_small_terminal_publish_refuses_missing_version_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rehearsal, "_aws_json", lambda *args: {})
    with pytest.raises(rehearsal.AssignmentRehearsalError, match="no version ID"):
        rehearsal._publish_terminal(
            tmp_path,
            _minimal_spec(),
            {"schema_version": 1, "run_id": rehearsal.RUN_ID},
        )
