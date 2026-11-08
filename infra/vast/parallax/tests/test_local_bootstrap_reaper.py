from __future__ import annotations

from pathlib import Path

import pytest

from infra.vast.parallax.lifecycle_supervisor import LifecycleError
from infra.vast.parallax.local_bootstrap_reaper import ReaperPolicy, run_reaper


def policy(tmp_path: Path, **overrides) -> ReaperPolicy:
    key = tmp_path / "lifecycle.key"
    key.write_text("scoped-key\n")
    key.chmod(0o600)
    values = {
        "run_id": "reaper-test",
        "gateway_label": "parallax-guard-reaper-test",
        "api_key_file": key,
        "lease_path": tmp_path / "lease",
        "handoff_path": tmp_path / "handoff",
        "ready_path": tmp_path / "ready.json",
        "receipt_path": tmp_path / "receipt.json",
        "stale_seconds": 10.0,
        "maximum_seconds": 60.0,
        "poll_seconds": 1.0,
    }
    values.update(overrides)
    return ReaperPolicy(**values)


def test_local_reaper_hands_off_without_any_provider_action(tmp_path: Path):
    current = policy(tmp_path)
    current.handoff_path.write_text('{"run_id":"reaper-test","target":"remote_guardian"}\n')
    packet = run_reaper(
        current,
        list_instances=lambda _key: pytest.fail("handoff must not query provider"),
        action=lambda **_kwargs: pytest.fail("handoff must not mutate provider"),
    )
    assert packet["status"] == "HANDED_OFF_TO_REMOTE_GUARDIAN"
    assert current.ready_path.is_file() and current.receipt_path.is_file()


def test_local_reaper_can_handoff_to_injected_scoped_deadman(tmp_path: Path):
    current = policy(tmp_path)
    current.handoff_path.write_text(
        '{"run_id":"reaper-test","target":"remote_scoped_deadman"}\n'
    )
    packet = run_reaper(
        current,
        list_instances=lambda _key: pytest.fail("handoff must not query provider"),
    )
    assert packet["status"] == "HANDED_OFF_TO_REMOTE_SCOPED_DEADMAN"


def test_stale_launcher_reaps_only_exact_gateway_label_stop_first(tmp_path: Path):
    current = policy(tmp_path)
    actions = []
    packet = run_reaper(
        current,
        list_instances=lambda _key: [
            {"id": 12, "label": current.gateway_label},
            {"id": 13, "label": "unrelated"},
        ],
        action=lambda **kwargs: actions.append(kwargs) or {
            "status": "STOPPED" if kwargs["action"] == "stop" else "DESTROYED"
        },
    )
    assert packet["status"] == "REAPED" and packet["matched_instances"] == 1
    assert [(row["instance_id"], row["action"]) for row in actions] == [
        (12, "stop"),
        (12, "destroy"),
    ]


def test_local_reaper_contract_rejects_cross_run_label(tmp_path: Path):
    current = policy(tmp_path)
    contract = {
        "schema_version": 1,
        "status": "FROZEN_LOCAL_REAPER",
        "run_id": current.run_id,
        "gateway_label": "parallax-guard-wrong",
        "api_key_file": str(current.api_key_file),
        "lease_path": str(current.lease_path),
        "handoff_path": str(current.handoff_path),
        "ready_path": str(current.ready_path),
        "receipt_path": str(current.receipt_path),
        "stale_seconds": 10,
        "maximum_seconds": 60,
        "poll_seconds": 1,
    }
    with pytest.raises(LifecycleError, match="bound to run_id"):
        ReaperPolicy.from_contract(contract)
