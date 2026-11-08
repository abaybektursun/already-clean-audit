"""Read-only helpers for the historical Vast volume rehearsal.

The original experiment produced banked storage evidence, but its direct
mutation path predates the guarded lifecycle and is permanently revoked. The
parsing helpers remain for historical result audits; no function in this module
can create, stop, destroy, or delete a Vast resource.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


GO_PHRASE = "REHEARSE_15GB"  # Historical only; it no longer grants authority.
REVOKED_MESSAGE = "legacy storage rehearsal is revoked; all Vast mutations must use the frozen guarded lifecycle"


class VastMutation:
    """Compatibility wrapper whose only remaining operations are read-only."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.identity = resolve_vast_identity()

    def call(self, arguments: list[str], *, timeout: int = 120) -> Any:
        if not arguments or arguments[0] not in {"search", "show"}:
            raise RuntimeError(REVOKED_MESSAGE)
        completed = subprocess.run(
            ["vastai", *arguments, "--raw"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return parse_vast_output(read_only=True, stdout=completed.stdout, stderr=completed.stderr)


def parse_vast_output(*, read_only: bool, stdout: str, stderr: str) -> Any:
    stdout = stdout.strip()
    stderr = stderr.strip()
    combined = (stdout + "\n" + stderr).lower()
    if any(marker in combined for marker in ("failed with error", "failed to", "error occurred")):
        raise RuntimeError(f"Vast command failed: {(stdout or stderr)[:500]}")
    if read_only:
        if not stdout:
            raise RuntimeError("Vast read command returned an empty response")
        return json.loads(stdout)
    # Retained only to decode already-banked CLI transcripts.
    match = re.search(r"(?:Created|Started)\.\s*(\{.*?\})(?:\n|$)", stdout)
    return ast.literal_eval(match.group(1)) if match else stdout


def parse_ssh_url(stdout: str) -> dict[str, Any] | None:
    match = re.search(r"ssh://([^@\s]+)@([^:\s]+):(\d+)", stdout)
    if not match:
        return None
    return {
        "ssh_user": match.group(1),
        "ssh_host": match.group(2),
        "ssh_port": int(match.group(3)),
    }


def canonical_public_key(value: str) -> str:
    fields = value.strip().split()
    if len(fields) < 2:
        raise ValueError("invalid SSH public key")
    return " ".join(fields[:2])


def resolve_vast_identity(ssh_dir: Path | None = None) -> Path:
    ssh_dir = ssh_dir or (Path.home() / ".ssh")
    completed = subprocess.run(
        ["vastai", "show", "ssh-keys", "--raw"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = json.loads(completed.stdout)
    registered = {
        canonical_public_key(str(row[field]))
        for row in rows
        for field in ("public_key", "ssh_key", "key")
        if row.get(field)
    }
    for public_path in sorted(ssh_dir.glob("*.pub")):
        try:
            public = canonical_public_key(public_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        private_path = public_path.with_suffix("")
        if public in registered and private_path.is_file():
            return private_path
    raise FileNotFoundError("no local private key matches a Vast-registered public key")


def _contract_id(payload: Any, key: str) -> int:
    """Decode IDs in already-banked historical mutation transcripts."""
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError(f"Vast create failed: {payload}")
    for candidate in (key, "new_contract", "volume_name", "id"):
        if candidate in payload:
            value = str(payload[candidate])
            if value.startswith("V."):
                value = value[2:]
            return int(value)
    raise RuntimeError(f"Vast response has no contract ID: {payload}")


def rehearse(_audit_path: Path, _output_path: Path) -> dict[str, Any]:
    raise RuntimeError(REVOKED_MESSAGE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", default="NO_GO")
    parser.add_argument("--market-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    print(f"NO_GO: {REVOKED_MESSAGE}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
