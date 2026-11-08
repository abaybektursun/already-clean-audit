"""Find Vast's endpoint permission for instance STOP without creating compute."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .lifecycle_supervisor import API_BASE, LifecycleError
from .scoped_key_setup import (
    _created_payload,
    _key_rows,
    _payload,
    _revoke_key,
    _rollback_named_key,
)


CANDIDATE_ENDPOINTS = (
    "api.instance",
    "api.instance.manage",
    "api.instance.update",
    "api.instance.destroy",
    "api.instances.manage",
    "api.instances.update",
)
NAME_PREFIX = "parallax-stop-permission-probe-"
NONEXISTENT_INSTANCE_ID = 999_999_999


def _show_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["vastai", "show", "api-keys", "--raw"],
        capture_output=True,
        text=True,
        check=False,
    )
    return _key_rows(_payload(completed))


def _stop_http_status(secret: str) -> int:
    request = urllib.request.Request(
        f"{API_BASE}/instances/{NONEXISTENT_INSTANCE_ID}/",
        data=json.dumps({"state": "stopped"}).encode("utf-8"),
        method="PUT",
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def run(output: Path) -> dict[str, Any]:
    before = _show_rows()
    if any(str(row.get("name", "")).startswith(NAME_PREFIX) for row in before):
        raise LifecycleError("stale permission-probe key exists before diagnostic")
    rows = []
    with tempfile.TemporaryDirectory(prefix="parallax-vast-permission-") as temporary:
        temporary_root = Path(temporary)
        for index, endpoint in enumerate(CANDIDATE_ENDPOINTS, start=1):
            name = f"{NAME_PREFIX}{index}"
            permissions = {"api": {"instance_write": {endpoint: {}}}}
            permissions_path = temporary_root / f"permissions-{index}.json"
            atomic_json(permissions_path, permissions)
            completed = subprocess.run(
                [
                    "vastai", "create", "api-key", "--name", name,
                    "--permission_file", str(permissions_path), "--raw",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            try:
                created = _created_payload(completed)
            except LifecycleError as error:
                if not _rollback_named_key(name):
                    raise LifecycleError("permission-probe parse failure and rollback unproven") from error
                raise
            key_id = int(created.get("id", 0))
            secret = str(created.get("key") or created.get("api_key") or "")
            if key_id <= 0 or len(secret) < 16:
                rolled_back = _revoke_key(key_id) if key_id > 0 else _rollback_named_key(name)
                if not rolled_back:
                    raise LifecycleError("invalid permission-probe key and rollback unproven")
                raise LifecycleError("permission probe did not receive a valid one-time key")
            status = None
            revoked = False
            try:
                status = _stop_http_status(secret)
            finally:
                revoked = _revoke_key(key_id)
            if not revoked:
                raise LifecycleError(f"permission probe key {key_id} was not revoked")
            returned = created.get("permissions", {}).get("api", {}).get("instance_write")
            rows.append({
                "candidate": endpoint,
                "api_key_id": key_id,
                "returned_instance_write": returned,
                "nonexistent_instance_stop_http_status": status,
                "authorized_beyond_auth_gate": status not in {401, 403},
                "revoked": True,
                "secret_logged": False,
            })
            if status not in {401, 403}:
                break
    after = _show_rows()
    stale = [
        {"id": row.get("id"), "name": row.get("name")}
        for row in after
        if str(row.get("name", "")).startswith(NAME_PREFIX)
    ]
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_CANDIDATE_FOUND" if any(
            row["authorized_beyond_auth_gate"] for row in rows
        ) else "RED_NO_CANDIDATE",
        "provider": "vast.ai",
        "mutation": "temporary_api_keys_only_no_instances",
        "nonexistent_instance_id": NONEXISTENT_INSTANCE_ID,
        "candidates": rows,
        "stale_probe_keys_after": stale,
        "all_temporary_keys_revoked": not stale and all(row["revoked"] for row in rows),
        "instances_created": 0,
        "gpu_work": False,
        "secrets_logged": False,
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.output.resolve())
    print(json.dumps({
        "status": packet["status"],
        "candidates": len(packet["candidates"]),
        "all_temporary_keys_revoked": packet["all_temporary_keys_revoked"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
