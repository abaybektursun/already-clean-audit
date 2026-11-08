from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from infra.vast.parallax.cache_audit import CacheAuditError, audit, validate_audit_spec
from infra.vast.parallax.guarded_launch import select_live_gateway_offer


def _spec(root: Path, payload: bytes = b"exact-cache-bytes") -> dict:
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "provider": "vast.ai",
        "scope": "pg1_model_cache_integrity",
        "audit_id": "unit-cache-audit",
        "cache_root": str(root),
        "expected_total_bytes": len(payload),
        "maximum_audit_bytes": len(payload),
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def test_exact_cache_audit_is_green_without_loading_model_or_network(tmp_path: Path):
    payload = b"exact-cache-bytes"
    (tmp_path / "model.safetensors").write_bytes(payload)
    result = audit(_spec(tmp_path, payload))
    assert result["status"] == "GREEN"
    assert result["verified_bytes"] == len(payload)
    assert result["model_loaded"] is False
    assert result["network_reads"] == 0
    assert result["gpu_created"] is False


def test_cache_audit_fails_closed_on_hash_mismatch(tmp_path: Path):
    payload = b"exact-cache-bytes"
    (tmp_path / "model.safetensors").write_bytes(b"wrong-cache-bytes")
    result = audit(_spec(tmp_path, payload))
    assert result["status"] == "RED"
    assert result["verified_bytes"] == 0


def test_cache_audit_rejects_escape_and_disagreed_total(tmp_path: Path):
    spec = _spec(tmp_path)
    spec["files"][0]["path"] = "../escape"
    with pytest.raises(CacheAuditError, match="relative and safe"):
        validate_audit_spec(spec)
    spec = _spec(tmp_path)
    spec["expected_total_bytes"] += 1
    with pytest.raises(CacheAuditError, match="total-byte binding"):
        validate_audit_spec(spec)
    spec = _spec(tmp_path)
    spec["audit_id"] = "unsafe id"
    with pytest.raises(CacheAuditError, match="id is unsafe"):
        validate_audit_spec(spec)


def test_gateway_only_market_selection_never_queries_for_worker():
    calls = []

    def vast(*args):
        calls.append(args)
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        assert "machine_id=9" in args[2]
        return [
            {"id": 2, "machine_id": 9, "dph_total": 0.08, "reliability": 0.999},
            {"id": 1, "machine_id": 9, "dph_total": 0.06, "reliability": 0.990},
        ]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_gateway_offer(
            spec={"gateway_disk_gb": 20}, volume_id=44
        )
    assert receipt["selected_gateway"]["id"] == 1
    assert receipt["worker_query"] is None
    assert receipt["worker_created"] is False
    assert len(calls) == 2


def test_frozen_pg1_audit_manifest_matches_source_cache_manifest():
    root = Path(__file__).resolve().parents[4]
    audit_spec = json.loads(
        (root / "studies/lineage/cross-lineage/config/parallax_pg1_cache_audit_v1.json").read_text()
    )
    source = json.loads(
        (root / audit_spec["source_manifest"]["path"]).read_text()
    )
    pg1 = next(row for row in source["repositories"] if row["repo"] == audit_spec["repository"])
    assert pg1["revision"] == audit_spec["revision"]
    assert pg1["snapshot"] == audit_spec["cache_root"]
    assert pg1["files"] == audit_spec["files"]
    validate_audit_spec(audit_spec)
