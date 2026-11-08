from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from infra.aws_parallax import audit_pi0_text_image_stage_result as audit


ROOT = Path(__file__).resolve().parents[4]


def _inputs() -> dict[Path, dict]:
    return {
        relative: json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for relative in audit.EXPECTED_HASHES
    }


def _fake_aws(inputs: dict[Path, dict], *, bad_metadata: bool = False, delete_marker: bool = False):
    worker = inputs[audit.WORKER_PATH]
    rows = {row["key"]: row for group in (worker["dependencies"], worker["images"]) for row in group}

    def call(*arguments: str):
        if arguments[:2] == ("s3api", "head-object"):
            key = arguments[arguments.index("--key") + 1]
            row = rows[key]
            if "dependency_id" in row:
                metadata = {
                    "sha256": row["sha256"],
                    "integrity-kind": row["integrity_kind"],
                    "source-integrity": row["source_integrity"],
                }
            else:
                metadata = {
                    "sha256": row["sha256"],
                    "item-id": row["item_id"],
                    "source-index": str(row["source_index"]),
                    "decoded-rgb-sha256": row["decoded_RGB_sha256"],
                }
            if bad_metadata:
                metadata["sha256"] = "wrong"
            return {
                "ContentLength": row["bytes"],
                "ChecksumSHA256": base64.b64encode(bytes.fromhex(row["sha256"])).decode("ascii"),
                "ServerSideEncryption": "AES256",
                "Metadata": metadata,
            }
        if arguments[:2] == ("s3api", "list-object-versions"):
            return {"Versions": [], "DeleteMarkers": [{"Key": "bad"}] if delete_marker else []}
        if arguments[:2] == ("s3api", "list-multipart-uploads"):
            return {"Uploads": []}
        if arguments[:2] in {
            ("ec2", "describe-instances"),
            ("scheduler", "list-schedules"),
        }:
            return []
        raise AssertionError(arguments)

    return call


def test_independent_metadata_audit_preserves_registered_red_and_durable_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs()
    monkeypatch.setattr(audit, "_aws_json", _fake_aws(inputs))
    output = tmp_path / "audit.json"
    packet = audit.audit(ROOT, output)
    assert packet["status"] == (
        "RED_REGISTERED_PNG_BYTES_GREEN_DURABLE_LINUX_CANDIDATE_PIXEL_EQUIVALENCE_UNPROVEN"
    )
    assert packet["registered_gate"] == {
        "PNG_byte_exact_matches": 0,
        "PNG_sha256_exact_matches": 0,
        "historical_manifest_total_bytes": 126_855,
        "Linux_candidate_total_bytes": 128_190,
        "difference_bytes": 1_335,
        "verdict": "RED_PRE_REGISTERED_PNG_BYTE_IDENTITY_GATE",
    }
    assert len(packet["comparisons"]) == 10
    assert len(packet["durable_dependencies"]) == 3
    assert len(packet["durable_images"]) == 10
    assert packet["controller_exact_version_HEAD_count"] == 13
    assert all(not row["PNG_bytes_exact"] and not row["PNG_sha256_exact"] for row in packet["comparisons"])
    assert all(
        row["decoded_RGB_independent_equivalence_to_historical_manifest_proven"] is False
        for row in packet["comparisons"]
    )
    assert output.is_file()


def test_audit_refuses_to_rewrite_the_observed_mismatch_pattern(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs()
    worker = json.loads(json.dumps(inputs[audit.WORKER_PATH]))
    expected = inputs[audit.SPEC_PATH]["selected_items"][0]
    worker["images"][0]["bytes"] = expected["manifest_image_bytes"]
    worker["images"][0]["sha256"] = expected["manifest_image_sha256"]
    worker["images"][0]["readback_sha256"] = expected["manifest_image_sha256"]
    inputs[audit.WORKER_PATH] = worker
    monkeypatch.setattr(audit, "_read_bound", lambda root, relative: inputs[relative])
    monkeypatch.setattr(audit, "_aws_json", _fake_aws(inputs))
    with pytest.raises(audit.AuditError, match="mismatch pattern"):
        audit.audit(ROOT, tmp_path / "audit.json")


def test_audit_rejects_wrong_exact_version_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs()
    monkeypatch.setattr(audit, "_aws_json", _fake_aws(inputs, bad_metadata=True))
    with pytest.raises(audit.AuditError, match="exact S3 metadata mismatch"):
        audit.audit(ROOT, tmp_path / "audit.json")


def test_audit_rejects_delete_markers_or_incomplete_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs()
    monkeypatch.setattr(audit, "_aws_json", _fake_aws(inputs, delete_marker=True))
    with pytest.raises(audit.AuditError, match="delete markers"):
        audit.audit(ROOT, tmp_path / "audit.json")


def test_audit_rejects_nonzero_firewall_counter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs()
    worker = json.loads(json.dumps(inputs[audit.WORKER_PATH]))
    worker["model_loads"] = 1
    inputs[audit.WORKER_PATH] = worker
    monkeypatch.setattr(audit, "_read_bound", lambda root, relative: inputs[relative])
    monkeypatch.setattr(audit, "_aws_json", _fake_aws(inputs))
    with pytest.raises(audit.AuditError, match="firewall counters"):
        audit.audit(ROOT, tmp_path / "audit.json")


def test_audit_refuses_overwrite_before_any_AWS_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "audit.json"
    output.write_text("do not replace", encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "_aws_json",
        lambda *args: (_ for _ in ()).throw(AssertionError("AWS must not be called")),
    )
    with pytest.raises(audit.AuditError, match="refusing to overwrite"):
        audit.audit(ROOT, output)
    assert output.read_text(encoding="utf-8") == "do not replace"


def test_auditor_is_independent_and_uses_literal_AWS_control_plane() -> None:
    text = (ROOT / "infra/aws_parallax/audit_pi0_text_image_stage_result.py").read_text(
        encoding="utf-8"
    )
    assert "pi0_text_image_stage_launch" not in text
    assert 'PROFILE = "fuelos"' in text
    assert 'REGION = "us-east-1"' in text
    assert '"--profile", PROFILE, "--region", REGION' in text
    assert "get-object" not in text
    assert "docker" not in text.lower()
    assert "vastai" not in text.lower()
