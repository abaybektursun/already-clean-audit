from __future__ import annotations

import json
from pathlib import Path

from infra.vast.parallax.ephemeral_guardian import GuardianError, read_worker_assignment, run, validate_contract


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def _contract(tmp_path: Path) -> dict[str, object]:
    scoped = tmp_path / "scoped-key"
    own = tmp_path / "self-key"
    scoped.write_text("scoped", encoding="utf-8")
    own.write_text("self", encoding="utf-8")
    return {
        "schema_version": 1,
        "status": "FROZEN_GUARDIAN",
        "run_id": "guard-test-20260718",
        "worker_label": "parallax-worker-guard-test-20260718",
        "self_instance_id": 11,
        "scoped_api_key_path": "/run/parallax/scoped-key",
        "self_api_key_path": "/run/parallax/self-key",
        "receipt_path": str(tmp_path / "receipt.json"),
        "armed_path": str(tmp_path / "armed.json"),
        "worker_assignment_path": "/run/parallax/worker-assignment.json",
        "worker_assignment_ack_path": str(tmp_path / "assignment-ack.json"),
        "discovery_seconds": 30,
        "provider_loading_seconds": 60,
        "hard_runtime_seconds": 120,
        "poll_seconds": 1,
        "consecutive_absent_observations": 3,
    }


def test_validate_guardian_contract(tmp_path: Path):
    assert validate_contract(_contract(tmp_path))["self_instance_id"] == 11


def test_guardian_destroys_loading_worker_then_itself(tmp_path: Path, monkeypatch):
    contract = _contract(tmp_path)
    scoped = tmp_path / "scoped-key"
    own = tmp_path / "self-key"
    monkeypatch.setattr(
        "infra.vast.parallax.ephemeral_guardian._secret",
        lambda path: "scoped" if path.name == "scoped-key" else "self",
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    clock = Clock()
    actions = []

    def rows(_key):
        return [{"id": 22, "label": contract["worker_label"], "actual_status": "loading"}]

    packet = run(
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        list_fn=rows,
        action_fn=lambda **kwargs: actions.append((kwargs["instance_id"], kwargs["action"])),
    )
    assert packet["reason"] == "provider_loading_timeout"
    assert actions == [(22, "destroy"), (11, "destroy")]


def test_guardian_hard_stops_running_worker_but_does_not_delete_scratch(tmp_path: Path, monkeypatch):
    contract = _contract(tmp_path)
    monkeypatch.setattr("infra.vast.parallax.ephemeral_guardian._secret", lambda _path: "key")
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    clock = Clock()
    actions = []
    packet = run(
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        list_fn=lambda _key: [
            {"id": 22, "label": contract["worker_label"], "actual_status": "running"}
        ],
        action_fn=lambda **kwargs: actions.append((kwargs["instance_id"], kwargs["action"])),
    )
    assert packet["reason"] == "hard_runtime_timeout"
    assert actions == [(22, "stop"), (11, "destroy")]


def test_guardian_self_destroys_when_worker_disappears(tmp_path: Path, monkeypatch):
    contract = _contract(tmp_path)
    monkeypatch.setattr("infra.vast.parallax.ephemeral_guardian._secret", lambda _path: "key")
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    clock = Clock()
    calls = 0
    actions = []

    def rows(_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"id": 22, "label": contract["worker_label"], "actual_status": "running"}]
        return []

    packet = run(
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        list_fn=rows,
        action_fn=lambda **kwargs: actions.append((kwargs["instance_id"], kwargs["action"])),
    )
    assert packet["reason"] == "worker_absent_after_confirmed_observations"
    assert actions == [(11, "destroy")]


def test_one_transient_absence_does_not_make_guardian_leave_worker(tmp_path: Path, monkeypatch):
    contract = _contract(tmp_path)
    monkeypatch.setattr("infra.vast.parallax.ephemeral_guardian._secret", lambda _path: "key")
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    clock = Clock()
    calls = 0
    actions = []

    def rows(_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"id": 22, "label": contract["worker_label"], "actual_status": "running"}]
        if calls == 2:
            return []
        return [{"id": 22, "label": contract["worker_label"], "actual_status": "stopped"}]

    packet = run(
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        list_fn=rows,
        action_fn=lambda **kwargs: actions.append((kwargs["instance_id"], kwargs["action"])),
    )
    assert packet["reason"] == "worker_stopped_preserved_for_recovery"
    assert actions == [(11, "destroy")]


def test_exact_worker_assignment_survives_unobservable_label_until_bounded_destroy(
    tmp_path: Path, monkeypatch
):
    contract = _contract(tmp_path)
    monkeypatch.setattr("infra.vast.parallax.ephemeral_guardian._secret", lambda _path: "key")
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    clock = Clock()
    actions = []
    assignment = {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": contract["run_id"],
        "worker_label": contract["worker_label"],
        "worker_id": 22,
    }
    packet = run(
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        list_fn=lambda _key: [],
        action_fn=lambda **kwargs: actions.append((kwargs["instance_id"], kwargs["action"])),
        assignment_fn=lambda _path, _contract: assignment,
    )
    assert packet["reason"] == "assigned_worker_unobserved_provider_timeout"
    assert packet["worker_assigned"] is True
    assert packet["worker_assignment_acknowledged"] is True
    assert packet["worker_observed"] is False
    assert json.loads((tmp_path / "assignment-ack.json").read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "GUARDIAN_WORKER_ASSIGNMENT_ACKNOWLEDGED",
        "run_id": contract["run_id"],
        "worker_label": contract["worker_label"],
        "worker_id": 22,
    }
    assert actions == [(22, "destroy"), (11, "destroy")]


def test_worker_assignment_is_exact_and_refuses_identity_drift(tmp_path: Path):
    contract = _contract(tmp_path)
    assignment_path = tmp_path / "assignment.json"
    payload = {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": contract["run_id"],
        "worker_label": contract["worker_label"],
        "worker_id": 22,
    }
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_worker_assignment(assignment_path, contract)["worker_id"] == 22
    payload["worker_label"] = "other"
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        read_worker_assignment(assignment_path, contract)
    except GuardianError:
        pass
    else:
        raise AssertionError("worker assignment identity drift was accepted")
