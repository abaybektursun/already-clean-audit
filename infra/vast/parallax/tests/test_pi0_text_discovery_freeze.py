from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from infra.vast.parallax.pi0_text_discovery_freeze import (
    P2TextDiscoveryFreezeError,
    attach_donors,
    freeze_manifest,
    load_rows,
    select_panel,
    validate_execution_freeze,
)


def _row(item_id: str, target: str, *, split: str = "discovery_eval", image_sha: str | None = None):
    labels = ["alpha", "beta", "gamma"]
    return {
        "item_id": item_id * 64,
        "task_id": "visual_control",
        "split": split,
        "prompt": "Classify the image: alpha, beta, gamma.",
        "target": target,
        "candidates": labels,
        "source_identity": {"index": int(item_id, 16)},
        "image": f"images/{item_id}.png",
        "image_sha256": image_sha or hashlib.sha256(f"image-{item_id}".encode()).hexdigest(),
        "image_bytes": 123,
    }


def _contract():
    return {
        "source": {
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


def test_selects_lowest_item_per_label_and_deranges_within_task():
    rows = [
        _row("a", "alpha"),
        _row("1", "alpha"),
        _row("b", "beta"),
        _row("2", "beta"),
        _row("c", "gamma"),
        _row("3", "gamma"),
        _row("0", "alpha", split="train"),
        {**_row("4", "alpha"), "task_id": "other"},
    ]
    selected = select_panel(rows, _contract())
    assert {row["item_id"] for row in selected} == {"1" * 64, "2" * 64, "3" * 64}
    attached = attach_donors(selected)
    assert len(attached) == 3
    assert all(row["task_id"] == "visual_control" for row in attached)
    assert all(row["item_id"] != row["shuffled_image_donor"]["item_id"] for row in attached)
    assert all(row["target"] != row["shuffled_image_donor"]["target"] for row in attached)
    assert len({row["shuffled_image_donor"]["item_id"] for row in attached}) == 3


def test_manifest_is_deterministic_and_declares_zero_forbidden_access():
    rows = [_row("1", "alpha"), _row("2", "beta"), _row("3", "gamma")]
    left = freeze_manifest(rows=rows, contract=_contract(), bindings={"z": "1", "a": "2"})
    right = freeze_manifest(rows=list(reversed(rows)), contract=_contract(), bindings={"a": "2", "z": "1"})
    assert left == right
    assert left["status"] == "GREEN_P2_TEXT_SELECTION_FROZEN_EXACT_CLOUD_IMAGE_VERSIONS_PENDING"
    assert left["selection"]["items"] == 3
    assert left["firewall"] == {
        "physical_source_splits": ["discovery_eval"],
        "selected_split": "discovery_eval",
        "confirmation_values_emitted": 0,
        "confirmation_items_read": 0,
        "endpoint_outputs_read": 0,
        "image_bodies_opened_on_laptop": 0,
        "model_loads": 0,
        "GPU_work": False,
        "paid_compute": False,
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[:-1],
        lambda rows: [{**rows[0], "candidates": ["alpha", "beta"]}, *rows[1:]],
        lambda rows: [{**rows[0], "image": None}, *rows[1:]],
        lambda rows: [{**rows[0], "target": "unknown"}, *rows[1:]],
    ],
)
def test_selection_fails_closed_on_underfill_or_contract_drift(mutator):
    rows = [_row("1", "alpha"), _row("2", "beta"), _row("3", "gamma")]
    with pytest.raises(P2TextDiscoveryFreezeError):
        select_panel(mutator(rows), _contract())


def test_donor_rejects_same_pixels_or_changed_prompt():
    rows = select_panel(
        [_row("1", "alpha"), _row("2", "beta"), _row("3", "gamma")], _contract()
    )
    rows[1]["manifest_image_sha256"] = rows[0]["manifest_image_sha256"]
    with pytest.raises(P2TextDiscoveryFreezeError, match="reused the recipient image"):
        attach_donors(rows)
    rows = select_panel(
        [_row("1", "alpha"), _row("2", "beta"), _row("3", "gamma")], _contract()
    )
    rows[1]["prompt"] = "changed"
    with pytest.raises(P2TextDiscoveryFreezeError, match="prompt or candidate"):
        attach_donors(rows)


def test_physical_manifest_hash_and_confirmation_firewalls(tmp_path: Path):
    path = tmp_path / "items.discovery.jsonl"
    rows = [_row("1", "alpha"), _row("2", "beta"), _row("3", "gamma")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(load_rows(path, expected_sha256=digest, allowed_splits={"train", "discovery_eval"})) == 3
    with pytest.raises(P2TextDiscoveryFreezeError, match="SHA-256 mismatch"):
        load_rows(path, expected_sha256="0" * 64, allowed_splits={"train", "discovery_eval"})
    path.write_text(json.dumps(_row("f", "alpha", split="confirmation_eval")) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(P2TextDiscoveryFreezeError, match="forbidden split"):
        load_rows(path, expected_sha256=digest, allowed_splits={"train", "discovery_eval"})


def test_execution_freeze_is_narrow_and_one_shot(tmp_path: Path):
    authorization = {
        "local_discovery_manifest_read": True,
        "paid_action": False,
        "AWS_CPU_create": False,
        "create_vast_instance": False,
        "create_vast_volume": False,
        "H100": False,
        "model_load": False,
        "simulator_start": False,
        "training": False,
        "confirmation_items_read": False,
        "promotion": False,
        "automatic_retry": False,
    }
    payload = {
        "status": "FROZEN_ONE_SHOT_LOCAL_P2_DISCOVERY_MANIFEST_READ",
        "authorization": authorization,
        "consumed": False,
        "reusable": False,
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_execution_freeze(path)["authorization"]["local_discovery_manifest_read"] is True
    payload["authorization"]["H100"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(P2TextDiscoveryFreezeError, match="broader work"):
        validate_execution_freeze(path)


def test_source_has_no_heavy_or_model_imports():
    source = Path("infra/vast/parallax/pi0_text_discovery_freeze.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "transformers", "lerobot", "PIL", "boto3", "vastai"):
        assert forbidden not in source
