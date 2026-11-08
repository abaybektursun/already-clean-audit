"""Strict offline-only root-rebase contract; production CLIs never select it."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any


class MechanicalRebaseError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise MechanicalRebaseError("mechanical rebase contract is absent or mutable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MechanicalRebaseError("mechanical rebase contract unreadable") from error
    body = value.get("contract") if isinstance(value, dict) else None
    if (
        not isinstance(body, dict)
        or value.get("schema") != "pg1-native-exact-chain-mechanical-contract/v1"
        or value.get("status") != "OFFLINE_DETERMINISTIC_FAKE_SERVICES_ONLY"
        or value.get("content_address") != {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(body)).hexdigest(),
        }
        or set(body) != {
            "kind", "production_authority", "provider_calls", "root_rebase",
            "fake_services", "network_scope", "may_mint_production_authority",
        }
        or body.get("kind") != "MECHANICAL_NO_AUTHORITY"
        or body.get("production_authority") is not False
        or body.get("provider_calls") != 0
        or body.get("fake_services") != "TRACKED_DETERMINISTIC_EXECUTABLES"
        or body.get("network_scope") != "LOOPBACK_OR_NONE"
        or body.get("may_mint_production_authority") is not False
    ):
        raise MechanicalRebaseError("mechanical rebase contract changed")
    root = Path(str(body["root_rebase"]))
    if not root.is_absolute() or root == Path("/") or root.is_symlink() or not root.is_dir():
        raise MechanicalRebaseError("mechanical root rebase changed")
    return {**value, "root": root.resolve()}


def rebased(root: Path, absolute: Path) -> Path:
    if not absolute.is_absolute():
        raise MechanicalRebaseError("only absolute production paths may be rebased")
    return root / absolute.relative_to("/")


def _root_before_marker(source: Path, marker: str) -> Path | None:
    text = str(source.resolve())
    token = f"/{marker.strip('/')}/"
    if token not in text:
        return None
    prefix = text.split(token, 1)[0]
    return Path(prefix or "/").resolve()


def control_contract_from_source(source: Path) -> tuple[dict[str, Any], Path] | None:
    """Discover the offline contract from copied control bytes, never argv/env."""

    for marker in ("opt/parallax/control-stage/source", "opt/parallax/control/source"):
        root = _root_before_marker(source, marker)
        if root is None:
            continue
        if root == Path("/"):
            return None
        path = root / "run/parallax/control-plane/exact-chain-mechanical.json"
        contract = load(path)
        if contract["root"] != root:
            raise MechanicalRebaseError("derived control root and contract root differ")
        return contract, path.resolve()
    raise MechanicalRebaseError("control source is outside canonical or copied layout")


def worker_contract_from_source(
    source: Path,
) -> tuple[dict[str, Any], Path, Path] | None:
    """Discover the offline contract from copied worker bytes, never argv/env."""

    for marker in (
        "opt/parallax/image/source", "opt/parallax/future-freeze/source", "run/parallax",
    ):
        worker = _root_before_marker(source, marker)
        if worker is None:
            continue
        if worker == Path("/"):
            return None
        path = worker / "run/parallax/pg1-native-mechanical-contract.json"
        contract = load(path)
        allowed_workers = {
            contract["root"].resolve(),
            (contract["root"] / ".pg1-fake-services/worker").resolve(),
        }
        if worker not in allowed_workers:
            raise MechanicalRebaseError("derived worker root and contract root differ")
        return contract, path.resolve(), worker
    raise MechanicalRebaseError("worker source is outside canonical or copied layout")
