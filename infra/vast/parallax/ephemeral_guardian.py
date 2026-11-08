"""Independent disposable guard for one labelled Vast worker.

The guard is armed before the worker is created.  It uses a narrowly scoped
Vast key that can only list/manage instances, so a controller/laptop loss in
the provider-loading gap cannot leave a GPU running indefinitely.  The guard
contains no model data and destroys itself after the worker is terminal.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "https://console.vast.ai/api/v0"


class GuardianError(RuntimeError):
    pass


def _secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value or any(character.isspace() for character in value):
        raise GuardianError("guardian secret is absent or malformed")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN_GUARDIAN":
        raise GuardianError("guardian contract is not frozen schema v1")
    run_id = str(contract.get("run_id", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,79}", run_id):
        raise GuardianError("guardian run id violates the frozen grammar")
    if contract.get("worker_label") != f"parallax-worker-{run_id}":
        raise GuardianError("guardian worker label is not bound to run id")
    for name in (
        "scoped_api_key_path",
        "self_api_key_path",
        "receipt_path",
        "armed_path",
        "worker_assignment_path",
        "worker_assignment_ack_path",
    ):
        path = Path(str(contract.get(name, "")))
        if not path.is_absolute():
            raise GuardianError(f"{name} must be absolute")
    if not str(contract["scoped_api_key_path"]).startswith("/run/parallax/"):
        raise GuardianError("scoped lifecycle key escapes /run/parallax")
    if not str(contract["self_api_key_path"]).startswith("/run/parallax/"):
        raise GuardianError("self lifecycle key escapes /run/parallax")
    if not str(contract["worker_assignment_path"]).startswith("/run/parallax/"):
        raise GuardianError("worker assignment escapes /run/parallax")
    bounds = {
        "discovery_seconds": (30, 600),
        "provider_loading_seconds": (60, 1800),
        "hard_runtime_seconds": (60, 3600),
    }
    for name, (minimum, maximum) in bounds.items():
        value = contract.get(name)
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise GuardianError(f"{name} escapes the frozen bound")
    poll = contract.get("poll_seconds")
    if not isinstance(poll, (int, float)) or not 0.5 <= float(poll) <= 5:
        raise GuardianError("guardian polling interval is invalid")
    absences = contract.get("consecutive_absent_observations")
    if not isinstance(absences, int) or not 3 <= absences <= 20:
        raise GuardianError("guardian absence confirmation count is invalid")
    self_id = contract.get("self_instance_id")
    if not isinstance(self_id, int) or self_id <= 0:
        raise GuardianError("guardian self instance id is absent")
    return contract


def list_instances(api_key: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{API_BASE}/instances/", headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8") or "[]")
    rows = payload.get("instances", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise GuardianError("Vast instance listing is malformed")
    return [row for row in rows if isinstance(row, dict)]


def instance_action(*, instance_id: int, api_key: str, action: str) -> None:
    if action not in {"stop", "destroy"}:
        raise ValueError(action)
    body = json.dumps({"state": "stopped"}).encode("utf-8") if action == "stop" else None
    request = urllib.request.Request(
        f"{API_BASE}/instances/{instance_id}/",
        data=body,
        method="PUT" if action == "stop" else "DELETE",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if not 200 <= int(response.status) < 300:
            raise GuardianError(f"Vast {action} returned HTTP {response.status}")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_worker_assignment(path: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    worker_id = payload.get("worker_id")
    if payload != {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": contract["run_id"],
        "worker_label": contract["worker_label"],
        "worker_id": worker_id,
    }:
        raise GuardianError("guardian worker assignment is malformed")
    if not isinstance(worker_id, int) or worker_id <= 0:
        raise GuardianError("guardian worker assignment has no valid instance ID")
    return payload


def run(
    contract_path: Path,
    *,
    monotonic=time.monotonic,
    sleep=time.sleep,
    list_fn=list_instances,
    action_fn=instance_action,
    assignment_fn=read_worker_assignment,
) -> dict[str, Any]:
    contract = validate_contract(json.loads(contract_path.read_text(encoding="utf-8")))
    scoped_key = _secret(Path(contract["scoped_api_key_path"]))
    self_key = _secret(Path(contract["self_api_key_path"]))
    receipt_path = Path(contract["receipt_path"])
    armed_path = Path(contract["armed_path"])
    assignment_ack_path = Path(contract["worker_assignment_ack_path"])
    started = monotonic()
    discovered_at: float | None = None
    running_at: float | None = None
    worker_id: int | None = None
    worker_assigned = False
    worker_observed = False
    last_status = "absent"
    provider_errors = 0
    consecutive_absences = 0
    reason = ""
    worker_action: str | None = None
    _atomic_json(
        armed_path,
        {
            "schema_version": 1,
            "status": "GUARDIAN_ARMED_WORKER_ABSENT",
            "run_id": contract["run_id"],
            "worker_label": contract["worker_label"],
        },
    )
    while True:
        now = monotonic()
        assignment = assignment_fn(Path(contract["worker_assignment_path"]), contract)
        if assignment is not None:
            assigned_id = int(assignment["worker_id"])
            if worker_id is not None and worker_id != assigned_id:
                raise GuardianError("worker assignment changed the frozen worker identity")
            if worker_id is None:
                worker_id = assigned_id
                discovered_at = now
                last_status = "assigned_unobserved"
            if not worker_assigned:
                _atomic_json(
                    assignment_ack_path,
                    {
                        "schema_version": 1,
                        "status": "GUARDIAN_WORKER_ASSIGNMENT_ACKNOWLEDGED",
                        "run_id": contract["run_id"],
                        "worker_label": contract["worker_label"],
                        "worker_id": assigned_id,
                    },
                )
            worker_assigned = True
        listing_succeeded = True
        try:
            rows = list_fn(scoped_key)
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
            provider_errors += 1
            listing_succeeded = False
            rows = []
        matches = [row for row in rows if row.get("label") == contract["worker_label"]]
        if len(matches) > 1:
            raise GuardianError("duplicate worker labels")
        if matches:
            consecutive_absences = 0
            worker_observed = True
            row = matches[0]
            candidate_id = int(row["id"])
            if worker_id is not None and candidate_id != worker_id:
                raise GuardianError("worker identity changed under a frozen label")
            if worker_id is None:
                worker_id = candidate_id
                discovered_at = now
            last_status = str(row.get("actual_status") or row.get("cur_state") or "unknown").lower()
            if last_status == "running" and running_at is None:
                running_at = now
        elif worker_id is not None and worker_observed and listing_succeeded:
            consecutive_absences += 1
            if consecutive_absences >= int(contract["consecutive_absent_observations"]):
                reason = "worker_absent_after_confirmed_observations"
                break
        if worker_id is None:
            if now - started >= int(contract["discovery_seconds"]):
                reason = "worker_never_created"
                break
        elif last_status in {"stopped", "exited", "offline"}:
            reason = f"worker_{last_status}_preserved_for_recovery"
            break
        elif running_at is None and discovered_at is not None:
            if now - discovered_at >= int(contract["provider_loading_seconds"]):
                action_fn(instance_id=worker_id, api_key=scoped_key, action="destroy")
                worker_action = "destroy"
                reason = (
                    "provider_loading_timeout"
                    if worker_observed
                    else "assigned_worker_unobserved_provider_timeout"
                )
                break
        elif running_at is not None and now - running_at >= int(contract["hard_runtime_seconds"]):
            action_fn(instance_id=worker_id, api_key=scoped_key, action="stop")
            worker_action = "stop"
            reason = "hard_runtime_timeout"
            break
        sleep(float(contract["poll_seconds"]))
    packet = {
        "schema_version": 1,
        "status": "GUARDIAN_TERMINAL",
        "run_id": contract["run_id"],
        "self_instance_id": contract["self_instance_id"],
        "worker_id": worker_id,
        "worker_assigned": worker_assigned,
        "worker_assignment_acknowledged": assignment_ack_path.is_file(),
        "worker_observed": worker_observed,
        "worker_last_status": last_status,
        "worker_action": worker_action,
        "reason": reason,
        "provider_errors": provider_errors,
        "consecutive_absences_at_terminal": consecutive_absences,
        "running_clock_started": running_at is not None,
        "wall_seconds": monotonic() - started,
    }
    _atomic_json(receipt_path, packet)
    action_fn(instance_id=int(contract["self_instance_id"]), api_key=self_key, action="destroy")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
