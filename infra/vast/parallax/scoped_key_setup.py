"""Create one dedicated Vast lifecycle key without printing the secret."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .lifecycle_supervisor import LifecycleError


EXPECTED_PERMISSIONS = {
    "api": {
        "instance_read": {"api.instances": {}},
        "instance_write": {"api.instance": {}},
    }
}


def _payload(completed: subprocess.CompletedProcess[str]) -> Any:
    if completed.returncode != 0:
        raise LifecycleError("Vast scoped API key command failed")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise LifecycleError("Vast scoped API key response was not JSON") from error


def _created_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse Vast's create response without ever echoing its one-time secret."""
    if completed.returncode != 0:
        raise LifecycleError("Vast scoped API key creation failed")
    value: Any
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        marker = "api-key created "
        position = completed.stdout.rfind(marker)
        if position < 0:
            raise LifecycleError("Vast scoped API key creation response was unrecognized")
        try:
            payload_line = completed.stdout[position + len(marker):].splitlines()[0].strip()
            value = ast.literal_eval(payload_line)
        except (SyntaxError, ValueError) as error:
            raise LifecycleError("Vast scoped API key creation response was unrecognized") from error
    if not isinstance(value, dict):
        raise LifecycleError("Vast scoped API key creation response is malformed")
    return value


def _key_rows(payload: Any) -> list[dict[str, Any]]:
    values = payload.get("apikeys", payload) if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for value in values or []:
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _revoke_key(key_id: int) -> bool:
    return subprocess.run(
        ["vastai", "delete", "api-key", str(key_id), "--raw"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0


def _rollback_named_key(name: str) -> bool:
    try:
        rows = _key_rows(_payload(subprocess.run(
            ["vastai", "show", "api-keys", "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )))
    except LifecycleError:
        return False
    matches = [int(row.get("id", 0)) for row in rows if row.get("name") == name]
    return len(matches) == 1 and matches[0] > 0 and _revoke_key(matches[0])


def create_scoped_key(
    *, name: str, permissions_file: Path, secret_file: Path, receipt_file: Path
) -> dict[str, Any]:
    if json.loads(permissions_file.read_text(encoding="utf-8")) != EXPECTED_PERMISSIONS:
        raise LifecycleError("lifecycle API permissions are not minimal and exact")
    if secret_file.exists():
        raise LifecycleError("lifecycle API secret file already exists")
    existing = _payload(subprocess.run(
        ["vastai", "show", "api-keys", "--raw"], capture_output=True, text=True, check=False
    ))
    rows = _key_rows(existing)
    if any(row.get("name") == name for row in rows):
        raise LifecycleError("named lifecycle API key already exists")
    completed = subprocess.run(
        [
            "vastai", "create", "api-key", "--name", name,
            "--permission_file", str(permissions_file), "--raw",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        created = _created_payload(completed)
    except LifecycleError as error:
        if not _rollback_named_key(name):
            raise LifecycleError("creation response failed and named-key rollback was unproven") from error
        raise LifecycleError("creation response failed; named key was rolled back") from error
    key_id = int(created.get("id", 0))
    secret = str(created.get("key") or created.get("api_key") or "")
    if key_id <= 0 or len(secret) < 16 or any(character.isspace() for character in secret):
        rolled_back = _revoke_key(key_id) if key_id > 0 else _rollback_named_key(name)
        if not rolled_back:
            raise LifecycleError("invalid one-time secret response and rollback was unproven")
        raise LifecycleError("Vast did not return a valid scoped key exactly once")
    returned_api = created.get("permissions", {}).get("api", {})
    for category, restriction in EXPECTED_PERMISSIONS["api"].items():
        if returned_api.get(category) != restriction:
            if not _revoke_key(key_id):
                raise LifecycleError("endpoint restrictions changed and rollback was unproven")
            observed = json.dumps(returned_api, sort_keys=True, separators=(",", ":"))
            raise LifecycleError(
                f"Vast did not preserve the exact endpoint restrictions; observed_api={observed}"
            )
    try:
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        receipt_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, (secret + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        packet = {
            "schema_version": 1,
            "status": "CREATED_SECRET_NOT_LOGGED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "vast.ai",
            "name": name,
            "api_key_id": key_id,
            "permissions": EXPECTED_PERMISSIONS,
            "permissions_file": str(permissions_file),
            "permissions_sha256": sha256_file(permissions_file),
            "secret_file": str(secret_file),
            "secret_mode": "0600",
            "secret_logged": False,
            "instances_created": 0,
        }
        atomic_json(receipt_file, packet)
    except Exception as error:
        secret_file.unlink(missing_ok=True)
        if not _revoke_key(key_id):
            raise LifecycleError(
                f"local persistence failed and scoped key {key_id} could not be rolled back"
            ) from error
        raise LifecycleError("local persistence failed; scoped key was rolled back") from error
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--permissions-file", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args()
    packet = create_scoped_key(
        name=args.name,
        permissions_file=args.permissions_file.resolve(),
        secret_file=args.secret_file.expanduser().resolve(),
        receipt_file=args.receipt_file.resolve(),
    )
    print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
