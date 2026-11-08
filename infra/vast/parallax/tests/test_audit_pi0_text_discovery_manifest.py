from __future__ import annotations

import hashlib
import json
from pathlib import Path

from infra.vast.parallax.audit_pi0_text_discovery_manifest import audit
from infra.vast.parallax.pi0_text_discovery_freeze import freeze_manifest


def _rows():
    labels = ["alpha", "beta", "gamma"]
    rows = []
    for target, prefix in zip(labels, "123"):
        for suffix in ("b", "a"):
            item_id = (prefix + suffix) * 32
            rows.append(
                {
                    "item_id": item_id,
                    "task_id": "visual_control",
                    "split": "discovery_eval",
                    "prompt": "Classify alpha, beta, gamma.",
                    "target": target,
                    "candidates": labels,
                    "source_identity": {"id": item_id},
                    "image": f"images/{item_id}.png",
                    "image_sha256": hashlib.sha256(item_id.encode()).hexdigest(),
                    "image_bytes": 42,
                }
            )
    rows.append({**rows[0], "item_id": "f" * 64, "split": "train"})
    return rows


def _contract(source_sha: str):
    return {
        "source": {
            "manifest_sha256": source_sha,
            "allowed_splits_in_physical_manifest": ["train", "discovery_eval"],
            "selected_split": "discovery_eval",
        },
        "selection": {
            "task_id": "visual_control",
            "expected_label_strata": ["alpha", "beta", "gamma"],
            "expected_candidate_set": ["alpha", "beta", "gamma"],
            "items_per_label_stratum": 1,
            "expected_items": 3,
        },
    }


def _execution():
    authorization = {
        "local_discovery_manifest_audit": True,
        "paid_action": False,
        "AWS_CPU_create": False,
        "create_vast_instance": False,
        "create_vast_volume": False,
        "H100": False,
        "image_body_read": False,
        "model_load": False,
        "simulator_start": False,
        "training": False,
        "confirmation_items_read": False,
        "promotion": False,
        "automatic_retry": False,
    }
    return {
        "status": "FROZEN_ONE_SHOT_LOCAL_P2_DISCOVERY_MANIFEST_AUDIT",
        "authorization": authorization,
        "consumed": False,
        "reusable": False,
    }


def _write_fixture(tmp_path: Path):
    rows = _rows()
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    contract_value = _contract(source_sha)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(contract_value), encoding="utf-8")
    frozen_value = freeze_manifest(rows=rows, contract=contract_value, bindings={})
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(frozen_value), encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(_execution()), encoding="utf-8")
    return source, contract, frozen, execution


def test_independent_recomputation_is_exact(tmp_path: Path):
    source, contract, frozen, execution = _write_fixture(tmp_path)
    result = audit(
        source_path=source,
        contract_path=contract,
        frozen_path=frozen,
        execution_freeze_path=execution,
    )
    assert result["status"] == "GREEN_INDEPENDENT_EXACT"
    assert result["errors"] == []
    assert result["recomputation"]["source_rows_parsed"] == 7
    assert result["recomputation"]["eligible_discovery_rows"] == 6
    assert result["recomputation"]["selected_items"] == 3
    assert result["recomputation"]["donor_bijection"] is True
    assert result["recomputation"]["exact_items_match"] is True


def test_audit_detects_tampered_item_and_broad_freeze(tmp_path: Path):
    source, contract, frozen, execution = _write_fixture(tmp_path)
    payload = json.loads(frozen.read_text())
    payload["items"][0]["prompt"] = "tampered"
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    execution_payload = json.loads(execution.read_text())
    execution_payload["authorization"]["image_body_read"] = True
    execution.write_text(json.dumps(execution_payload), encoding="utf-8")
    result = audit(
        source_path=source,
        contract_path=contract,
        frozen_path=frozen,
        execution_freeze_path=execution,
    )
    assert result["status"] == "RED"
    assert "items_exact" in result["errors"]
    assert "execution_freeze_image_body_read" in result["errors"]


def test_audit_detects_confirmation_split_and_source_hash_drift(tmp_path: Path):
    source, contract, frozen, execution = _write_fixture(tmp_path)
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**_rows()[0], "split": "confirmation_eval"}) + "\n")
    result = audit(
        source_path=source,
        contract_path=contract,
        frozen_path=frozen,
        execution_freeze_path=execution,
    )
    assert result["status"] == "RED"
    assert "source_manifest_sha256" in result["errors"]
    assert any(error.startswith("forbidden_split_line_") for error in result["errors"])


def test_auditor_does_not_import_the_producer():
    source = Path("infra/vast/parallax/audit_pi0_text_discovery_manifest.py").read_text(encoding="utf-8")
    assert "pi0_text_discovery_freeze" not in source
    for forbidden in ("import torch", "transformers", "lerobot", "PIL", "boto3", "vastai"):
        assert forbidden not in source
