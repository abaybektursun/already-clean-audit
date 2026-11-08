"""Detached exact-label reaper for the pre-SSH Vast gateway gap.

The reaper is armed before the gateway create request.  It owns only the exact
run label and exits after an explicit handoff written once the durable remote
guardian is live.  If the launcher disappears first, it stops then destroys
the gateway with the endpoint-scoped lifecycle key.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_json
from .lifecycle_guardian import _list_instances
from .lifecycle_supervisor import LifecycleError, _read_secret, vast_instance_action


@dataclass(frozen=True)
class ReaperPolicy:
    run_id: str
    gateway_label: str
    api_key_file: Path
    lease_path: Path
    handoff_path: Path
    ready_path: Path
    receipt_path: Path
    stale_seconds: float
    maximum_seconds: float
    poll_seconds: float

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "ReaperPolicy":
        if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN_LOCAL_REAPER":
            raise LifecycleError("local reaper contract is not frozen")
        policy = cls(
            run_id=str(contract["run_id"]),
            gateway_label=str(contract["gateway_label"]),
            api_key_file=Path(contract["api_key_file"]),
            lease_path=Path(contract["lease_path"]),
            handoff_path=Path(contract["handoff_path"]),
            ready_path=Path(contract["ready_path"]),
            receipt_path=Path(contract["receipt_path"]),
            stale_seconds=float(contract["stale_seconds"]),
            maximum_seconds=float(contract["maximum_seconds"]),
            poll_seconds=float(contract["poll_seconds"]),
        )
        if not policy.run_id or policy.gateway_label != f"parallax-guard-{policy.run_id}":
            raise LifecycleError("local reaper label is not bound to run_id")
        for path in (
            policy.api_key_file,
            policy.lease_path,
            policy.handoff_path,
            policy.ready_path,
            policy.receipt_path,
        ):
            if not path.is_absolute():
                raise LifecycleError("local reaper paths must be absolute")
        if not 5 <= policy.stale_seconds <= 60:
            raise LifecycleError("local reaper stale deadline is unsafe")
        if not 30 <= policy.maximum_seconds <= 600:
            raise LifecycleError("local reaper maximum deadline is unsafe")
        if not 0.25 <= policy.poll_seconds <= 2:
            raise LifecycleError("local reaper poll interval is unsafe")
        return policy


def run_reaper(
    policy: ReaperPolicy,
    *,
    list_instances: Callable[[str], list[dict[str, Any]]] = _list_instances,
    action: Callable[..., dict[str, Any]] = vast_instance_action,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    key = _read_secret(policy.api_key_file)
    started = monotonic()
    atomic_json(policy.ready_path, {
        "status": "ARMED_BEFORE_GATEWAY_CREATE",
        "run_id": policy.run_id,
        "gateway_label": policy.gateway_label,
        "wall_time_unix": wall_time(),
    })
    while True:
        if policy.handoff_path.is_file():
            try:
                handoff = json.loads(policy.handoff_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                handoff = {}
            target = handoff.get("target")
            if handoff.get("run_id") != policy.run_id or target not in {
                "remote_guardian",
                "remote_scoped_deadman",
            }:
                raise LifecycleError("local reaper received an invalid handoff marker")
            packet = {
                "schema_version": 1,
                "status": (
                    "HANDED_OFF_TO_REMOTE_GUARDIAN"
                    if target == "remote_guardian"
                    else "HANDED_OFF_TO_REMOTE_SCOPED_DEADMAN"
                ),
                "run_id": policy.run_id,
                "gateway_label": policy.gateway_label,
                "target": target,
                "elapsed_seconds": monotonic() - started,
            }
            atomic_json(policy.receipt_path, packet)
            return packet
        lease_age = float("inf")
        if policy.lease_path.is_file():
            lease_age = max(0.0, wall_time() - policy.lease_path.stat().st_mtime)
        if lease_age <= policy.stale_seconds and monotonic() - started < policy.maximum_seconds:
            sleep(policy.poll_seconds)
            continue

        # Once the lease is stale, never return to the watching state.  Keep
        # retrying exact-label cleanup for a bounded minute; the provider-side
        # endpoint key itself has no authority over volumes or other resources.
        cleanup_started = monotonic()
        last_error: str | None = None
        while monotonic() - cleanup_started <= 60:
            try:
                rows = list_instances(key)
                matches = [row for row in rows if row.get("label") == policy.gateway_label]
                actions = []
                for row in matches:
                    instance_id = int(row["id"])
                    stopped = action(instance_id=instance_id, api_key=key, action="stop")
                    destroyed = action(instance_id=instance_id, api_key=key, action="destroy")
                    actions.append({"instance_id": instance_id, "stop": stopped, "destroy": destroyed})
                packet = {
                    "schema_version": 1,
                    "status": "REAPED" if matches else "NO_MATCH_ACCOUNT_CLEAN",
                    "run_id": policy.run_id,
                    "gateway_label": policy.gateway_label,
                    "lease_age_seconds": lease_age,
                    "matched_instances": len(matches),
                    "actions": actions,
                    "elapsed_seconds": monotonic() - started,
                }
                atomic_json(policy.receipt_path, packet)
                return packet
            except Exception as error:  # detached safety boundary
                last_error = type(error).__name__
                sleep(policy.poll_seconds)
        packet = {
            "schema_version": 1,
            "status": "REAPER_API_UNAVAILABLE",
            "run_id": policy.run_id,
            "gateway_label": policy.gateway_label,
            "last_error_type": last_error,
            "elapsed_seconds": monotonic() - started,
        }
        atomic_json(policy.receipt_path, packet)
        return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    packet = run_reaper(ReaperPolicy.from_contract(contract))
    print(json.dumps(packet, sort_keys=True))
    return 0 if packet["status"] in {
        "HANDED_OFF_TO_REMOTE_GUARDIAN",
        "HANDED_OFF_TO_REMOTE_SCOPED_DEADMAN",
        "REAPED",
        "NO_MATCH_ACCOUNT_CLEAN",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
