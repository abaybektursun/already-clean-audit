from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path

import pytest

from infra.vast.parallax.lifecycle_guardian import GuardianPolicy, run_guardian
from infra.vast.parallax.lifecycle_supervisor import LifecycleError


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def wall(self) -> float:
        return 1000.0 + self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def policy(tmp_path: Path, **overrides) -> GuardianPolicy:
    secret = tmp_path / "lifecycle.key"
    secret.write_text("scoped-lifecycle-key\n")
    secret.chmod(0o600)
    values = {
        "run_id": "test-run",
        "worker_label": "parallax-worker-test-run",
        "state_root": tmp_path / "state",
        "api_key_file": secret,
        "discovery_timeout_seconds": 30,
        "provider_loading_timeout_seconds": 300,
        "billable_prearm_timeout_seconds": 30,
        "heartbeat_timeout_seconds": 5,
        "hard_runtime_seconds": 60,
        "poll_seconds": 1.0,
    }
    values.update(overrides)
    return GuardianPolicy(**values)


def test_guardian_stops_discovered_worker_if_supervisor_never_arms(tmp_path: Path):
    clock = Clock()
    actions = []

    def action(**kwargs):
        actions.append(kwargs)
        return {"status": "STOPPED"}

    packet = run_guardian(
        policy(tmp_path),
        list_instances=lambda _key: [{
            "id": 123,
            "label": "parallax-worker-test-run",
            "actual_status": "running",
        }],
        action=action,
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["status"] == "STOPPED" and packet["reason"] == "billable_prearm_timeout"
    assert actions == [{"instance_id": 123, "api_key": "scoped-lifecycle-key", "action": "stop"}]
    assert json.loads((policy(tmp_path).state_root / "guardian-receipt.json").read_text())["worker_id"] == 123
    registry = json.loads((policy(tmp_path).state_root / "launch-registry.json").read_text())
    assert registry["status"] == "STOPPED" and registry["run_id"] == "test-run"


def test_guardian_stops_on_stale_supervisor_heartbeat(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    current.state_root.mkdir(parents=True)
    (current.state_root / "armed.marker").touch()
    heartbeat = current.state_root / "heartbeat.json"
    heartbeat.write_text("{}\n")
    os.utime(heartbeat, (0, 0))
    packet = run_guardian(
        current,
        list_instances=lambda _key: [{"id": 123, "label": current.worker_label, "actual_status": "running"}],
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "stale_supervisor_heartbeat"


def test_guardian_stops_after_arm_when_supervisor_heartbeat_never_appears(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    current.state_root.mkdir(parents=True)
    armed = current.state_root / "armed.marker"
    armed.touch()
    os.utime(armed, (0, 0))
    packet = run_guardian(
        current,
        list_instances=lambda _key: [{"id": 123, "label": current.worker_label, "actual_status": "running"}],
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "missing_supervisor_heartbeat"


def test_guardian_races_supervisor_on_durable_normal_stop_intent(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    current.state_root.mkdir(parents=True)
    (current.state_root / "armed.marker").touch()
    (current.state_root / "heartbeat.json").write_text("{}\n")
    (current.state_root / "worker-stop-intent.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "DURABILITY_GREEN_STOP_PENDING",
        "run_id": current.run_id,
        "worker_id": 123,
    }))
    actions = []
    packet = run_guardian(
        current,
        list_instances=lambda _key: [{"id": 123, "label": current.worker_label, "actual_status": "running"}],
        action=lambda **kwargs: actions.append(kwargs) or {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "durability_green_stop_intent"
    assert actions[0]["action"] == "stop"


def test_guardian_does_not_spend_billable_prearm_during_provider_loading(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    calls = 0

    def staged_status(_key):
        nonlocal calls
        calls += 1
        return [{
            "id": 123,
            "label": current.worker_label,
            "actual_status": "loading" if clock.value < 250 else "running",
        }]

    packet = run_guardian(
        current,
        list_instances=staged_status,
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "billable_prearm_timeout"
    assert clock.value == 279
    assert calls > 250
    billable = json.loads((current.state_root / "guardian-billable-start.json").read_text())
    assert billable["provider_status"] == "running"


def test_guardian_bounds_unbilled_provider_loading_separately(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path, provider_loading_timeout_seconds=60)
    packet = run_guardian(
        current,
        list_instances=lambda _key: [{
            "id": 123,
            "label": current.worker_label,
            "actual_status": "loading",
        }],
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "provider_loading_timeout"
    assert packet["billable_clock_started"] is False
    assert clock.value == 60


def test_guardian_retries_transient_429_before_discovery(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    calls = 0

    def transient(_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("https://vast.invalid", 429, "limited", {}, None)
        return [{"id": 123, "label": current.worker_label, "actual_status": "running"}]

    packet = run_guardian(
        current,
        list_instances=transient,
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "billable_prearm_timeout" and calls > 2


def test_guardian_assumes_billable_after_provider_outage_post_discovery(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    calls = 0

    def loading_then_outage(_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"id": 123, "label": current.worker_label, "actual_status": "loading"}]
        raise urllib.error.HTTPError("https://vast.invalid", 429, "limited", {}, None)

    packet = run_guardian(
        current,
        list_instances=loading_then_outage,
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "billable_prearm_timeout"
    assert packet["provider_status"] == "provider_listing_unavailable_assume_billable"
    assert clock.value == 30


def test_guardian_recovered_loading_pauses_but_does_not_erase_uncertain_exposure(tmp_path: Path):
    clock = Clock()
    current = policy(tmp_path)
    calls = 0

    def transient_outage(_key):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise urllib.error.HTTPError("https://vast.invalid", 429, "limited", {}, None)
        return [{
            "id": 123,
            "label": current.worker_label,
            "actual_status": "loading" if clock.value < 250 else "running",
        }]

    packet = run_guardian(
        current,
        list_instances=transient_outage,
        action=lambda **_kwargs: {"status": "STOPPED"},
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    assert packet["reason"] == "billable_prearm_timeout"
    assert packet["billable_elapsed_seconds"] == 30
    assert clock.value == 277


def test_guardian_rejects_duplicate_run_labels(tmp_path: Path):
    current = policy(tmp_path)
    with pytest.raises(LifecycleError, match="duplicate"):
        run_guardian(
            current,
            list_instances=lambda _key: [
                {"id": 1, "label": current.worker_label},
                {"id": 2, "label": current.worker_label},
            ],
        )
