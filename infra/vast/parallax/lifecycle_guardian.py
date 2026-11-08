"""Independent gateway guardian that exists before the paid worker does."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_json
from .lifecycle_supervisor import API_BASE, LifecycleError, _read_secret, vast_instance_action


@dataclass(frozen=True)
class GuardianPolicy:
    run_id: str
    worker_label: str
    state_root: Path
    api_key_file: Path
    discovery_timeout_seconds: int
    provider_loading_timeout_seconds: int
    billable_prearm_timeout_seconds: int
    heartbeat_timeout_seconds: int
    hard_runtime_seconds: int
    poll_seconds: float

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "GuardianPolicy":
        if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN_GUARDIAN":
            raise LifecycleError("guardian contract is not frozen")
        parsed = cls(
            run_id=str(contract["run_id"]),
            worker_label=str(contract["worker_label"]),
            state_root=Path(contract["state_root"]),
            api_key_file=Path(contract["api_key_file"]),
            discovery_timeout_seconds=int(contract["discovery_timeout_seconds"]),
            provider_loading_timeout_seconds=int(contract["provider_loading_timeout_seconds"]),
            billable_prearm_timeout_seconds=int(contract["billable_prearm_timeout_seconds"]),
            heartbeat_timeout_seconds=int(contract["heartbeat_timeout_seconds"]),
            hard_runtime_seconds=int(contract["hard_runtime_seconds"]),
            poll_seconds=float(contract["poll_seconds"]),
        )
        if not parsed.run_id or parsed.worker_label != f"parallax-worker-{parsed.run_id}":
            raise LifecycleError("guardian label is not bound to run_id")
        if not parsed.state_root.is_absolute() or not parsed.api_key_file.is_absolute():
            raise LifecycleError("guardian paths must be absolute")
        if not 30 <= parsed.discovery_timeout_seconds <= 600:
            raise LifecycleError("guardian discovery timeout is unsafe")
        if not 60 <= parsed.provider_loading_timeout_seconds <= 3600:
            raise LifecycleError("guardian provider loading timeout is unsafe")
        if not 30 <= parsed.billable_prearm_timeout_seconds <= 300:
            raise LifecycleError("guardian billable prearm timeout is unsafe")
        if not 5 <= parsed.heartbeat_timeout_seconds <= 60:
            raise LifecycleError("guardian heartbeat timeout is unsafe")
        if not parsed.billable_prearm_timeout_seconds < parsed.hard_runtime_seconds <= 24 * 3600:
            raise LifecycleError("guardian hard runtime is unsafe")
        if not 0.5 <= parsed.poll_seconds <= 5:
            raise LifecycleError("guardian polling is unsafe")
        return parsed


def _list_instances(api_key: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{API_BASE}/instances/",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8") or "{}")
    rows = payload.get("instances", payload)
    if not isinstance(rows, list):
        raise LifecycleError("guardian instance listing is malformed")
    return [row for row in rows if isinstance(row, dict)]


def run_guardian(
    policy: GuardianPolicy,
    *,
    list_instances: Callable[[str], list[dict[str, Any]]] = _list_instances,
    action: Callable[..., dict[str, Any]] = vast_instance_action,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    policy.state_root.mkdir(parents=True, exist_ok=True)
    heartbeat_path = policy.state_root / "guardian-heartbeat.json"
    receipt_path = policy.state_root / "guardian-receipt.json"
    key = _read_secret(policy.api_key_file)
    started = monotonic()
    worker_id: int | None = None
    discovered_at: float | None = None
    billable_elapsed_seconds = 0.0
    clock_updated_at = started
    billing_phase = "worker_absent"
    billable_clock_ever_started = False
    provider_status = "worker_absent"
    registry_path = policy.state_root / "launch-registry.json"
    atomic_json(registry_path, {
        "schema_version": 1,
        "status": "GUARDIAN_ARMED_WORKER_ABSENT",
        "run_id": policy.run_id,
        "worker_label": policy.worker_label,
        "discovery_timeout_seconds": policy.discovery_timeout_seconds,
        "provider_loading_timeout_seconds": policy.provider_loading_timeout_seconds,
        "billable_prearm_timeout_seconds": policy.billable_prearm_timeout_seconds,
        "heartbeat_timeout_seconds": policy.heartbeat_timeout_seconds,
        "hard_runtime_seconds": policy.hard_runtime_seconds,
        "wall_time_unix": wall_time(),
    })
    os.sync()

    def set_billable_phase(phase: str) -> None:
        nonlocal billing_phase, billable_clock_ever_started
        if not billable_clock_ever_started:
            billable_clock_ever_started = True
            atomic_json(policy.state_root / "guardian-billable-start.json", {
                "schema_version": 1,
                "status": "BILLABLE_CLOCK_STARTED",
                "run_id": policy.run_id,
                "worker_id": worker_id,
                "provider_status": provider_status,
                "billing_phase": phase,
                "wall_time_unix": wall_time(),
            })
        billing_phase = phase

    while True:
        now = monotonic()
        interval_seconds = max(0.0, now - clock_updated_at)
        interval_counted_as_billable = billing_phase in {
            "provider_running",
            "provider_unknown_assume_billable",
        }
        if interval_counted_as_billable:
            billable_elapsed_seconds += interval_seconds
        clock_updated_at = now
        atomic_json(heartbeat_path, {
            "status": "WATCHING",
            "run_id": policy.run_id,
            "worker_id": worker_id,
            "provider_status": provider_status,
            "billing_phase": billing_phase,
            "billable_clock_started": billable_clock_ever_started,
            "billable_elapsed_seconds": billable_elapsed_seconds,
            "wall_time_unix": wall_time(),
        })
        try:
            rows = list_instances(key)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as error:
            atomic_json(heartbeat_path, {
                "status": "WATCHING_PROVIDER_RETRY",
                "run_id": policy.run_id,
                "worker_id": worker_id,
                "provider_status": provider_status,
                "error_type": type(error).__name__,
                "http_status": getattr(error, "code", None),
                "wall_time_unix": wall_time(),
            })
            if worker_id is None:
                if now - started >= policy.discovery_timeout_seconds:
                    raise LifecycleError("guardian provider listing unavailable through discovery deadline") from error
                sleep(policy.poll_seconds)
                continue
            # Once a worker exists, an unavailable provider listing cannot be
            # assumed free.  Start the short paid clock conservatively so an
            # API outage cannot turn into unbounded GPU billing.
            if not interval_counted_as_billable:
                billable_elapsed_seconds += interval_seconds
                interval_counted_as_billable = True
            provider_status = "provider_listing_unavailable_assume_billable"
            set_billable_phase("provider_unknown_assume_billable")
            rows = []
        matches = [row for row in rows if row.get("label") == policy.worker_label]
        if len(matches) > 1:
            raise LifecycleError("guardian found duplicate run labels")
        if matches:
            row = matches[0]
            row_id = int(row["id"])
            newly_discovered = worker_id is None
            if worker_id is not None and row_id != worker_id:
                raise LifecycleError("guardian worker label changed instance identity")
            if newly_discovered:
                worker_id = row_id
                discovered_at = now
                atomic_json(policy.state_root / "guardian-discovery.json", {
                    "status": "DISCOVERED",
                    "worker_id": worker_id,
                    "worker_label": policy.worker_label,
                })
            raw_status = row.get("actual_status")
            provider_status = str(raw_status or row.get("cur_state") or "unknown").lower()
            if provider_status == "loading":
                # A recovered Loading observation pauses the paid clock but
                # never erases exposure accumulated during an API outage.
                billing_phase = "provider_loading_unbilled"
            else:
                # Vast documents Loading as unbilled.  Every other/unknown
                # state is treated as billable.  Charge the whole observation
                # interval so a transition between polls cannot leak seconds.
                if not interval_counted_as_billable:
                    billable_elapsed_seconds += interval_seconds
                    interval_counted_as_billable = True
                set_billable_phase("provider_running")
            if newly_discovered:
                atomic_json(registry_path, {
                    "schema_version": 1,
                    "status": (
                        "WORKER_DISCOVERED_LOADING_UNBILLED"
                        if provider_status == "loading"
                        else "WORKER_DISCOVERED_ASSUME_BILLABLE"
                    ),
                    "run_id": policy.run_id,
                    "worker_label": policy.worker_label,
                    "worker_id": worker_id,
                    "provider_status": provider_status,
                    "wall_time_unix": wall_time(),
                })
        if worker_id is None:
            if now - started >= policy.discovery_timeout_seconds:
                raise LifecycleError("guardian did not discover worker before deadline")
            sleep(policy.poll_seconds)
            continue
        armed = policy.state_root / "armed.marker"
        supervisor_heartbeat = policy.state_root / "heartbeat.json"
        stop_intent = policy.state_root / "worker-stop-intent.json"
        reason: str | None = None
        if stop_intent.is_file():
            try:
                intent = json.loads(stop_intent.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                intent = {}
            if (
                intent.get("status") in {
                    "DURABILITY_GREEN_STOP_PENDING",
                    "STOP_CONFIRMED_DESTROY_PENDING",
                    "STOP_UNCONFIRMED",
                }
                and intent.get("run_id") == policy.run_id
                and int(intent.get("worker_id", 0)) == worker_id
            ):
                reason = "durability_green_stop_intent"
            else:
                # A malformed or cross-run marker cannot justify continuing to
                # bill.  Fail closed while preserving the worker disk.
                reason = "invalid_stop_intent"
        elif (
            billing_phase == "provider_loading_unbilled"
            and discovered_at is not None
            and now - discovered_at >= policy.provider_loading_timeout_seconds
        ):
            reason = "provider_loading_timeout"
        elif (
            not armed.is_file()
            and billable_elapsed_seconds >= policy.billable_prearm_timeout_seconds
        ):
            reason = "billable_prearm_timeout"
        elif armed.is_file():
            if billing_phase not in {"provider_running", "provider_unknown_assume_billable"}:
                # An armed worker necessarily has a running container.  If the
                # provider row was ambiguous, begin the paid clock now.
                if not interval_counted_as_billable:
                    billable_elapsed_seconds += interval_seconds
                    interval_counted_as_billable = True
                provider_status = "armed_assume_billable"
                set_billable_phase("provider_unknown_assume_billable")
            if billable_elapsed_seconds >= policy.hard_runtime_seconds:
                reason = "hard_runtime"
            elif supervisor_heartbeat.is_file():
                age = wall_time() - supervisor_heartbeat.stat().st_mtime
                if age > policy.heartbeat_timeout_seconds:
                    reason = "stale_supervisor_heartbeat"
            else:
                armed_age = wall_time() - armed.stat().st_mtime
                if armed_age > policy.heartbeat_timeout_seconds:
                    reason = "missing_supervisor_heartbeat"
        if reason is not None:
            stopped = action(instance_id=worker_id, api_key=key, action="stop")
            packet = {
                "schema_version": 1,
                "status": "STOPPED" if stopped.get("status") == "STOPPED" else "STOP_FAILED",
                "run_id": policy.run_id,
                "worker_id": worker_id,
                "reason": reason,
                "provider_status": provider_status,
                "billing_phase": billing_phase,
                "billable_clock_started": billable_clock_ever_started,
                "billable_elapsed_seconds": billable_elapsed_seconds,
                "teardown": stopped,
            }
            atomic_json(receipt_path, packet)
            atomic_json(registry_path, {
                "schema_version": 1,
                "status": packet["status"],
                "run_id": policy.run_id,
                "worker_label": policy.worker_label,
                "worker_id": worker_id,
                "reason": reason,
                "wall_time_unix": wall_time(),
            })
            os.sync()
            return packet
        sleep(policy.poll_seconds)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    packet = run_guardian(GuardianPolicy.from_contract(contract))
    print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
