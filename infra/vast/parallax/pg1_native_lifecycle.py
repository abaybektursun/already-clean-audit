"""Process-safe PG1 lifecycle compare-and-swap primitives.

This module is stdlib-only.  Local state is serialized with an advisory
``fcntl`` lock held across read/validate/replace/fsync.  Durable S3 writers use
the returned sequence with If-None-Match/If-Match; reconciliation never chooses
a lower sequence or regresses an absorbing terminal state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "pg1-native-local-lifecycle/v2"
STATES = (
    "ARMED",
    "RUNNING_HEARTBEAT",
    "FAILURE_STOP_REQUIRED",
    "TERMINAL_STOPPED_PRESERVED",
    "TERMINAL_DESTROYED_DURABLE",
)
TERMINAL_STATES = frozenset(
    {"TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE"}
)
MAX_COMPLETE_ANCESTORS = 256
TRANSITIONS = {
    None: frozenset({"ARMED", "FAILURE_STOP_REQUIRED"}),
    "ARMED": frozenset(
        {"RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED", "TERMINAL_STOPPED_PRESERVED",
         "TERMINAL_DESTROYED_DURABLE"}
    ),
    "RUNNING_HEARTBEAT": frozenset(
        {"RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED", "TERMINAL_STOPPED_PRESERVED",
         "TERMINAL_DESTROYED_DURABLE"}
    ),
    "FAILURE_STOP_REQUIRED": frozenset(
        {"FAILURE_STOP_REQUIRED", "TERMINAL_STOPPED_PRESERVED",
         "TERMINAL_DESTROYED_DURABLE"}
    ),
    "TERMINAL_STOPPED_PRESERVED": frozenset(
        {"TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE"}
    ),
    "TERMINAL_DESTROYED_DURABLE": frozenset({"TERMINAL_DESTROYED_DURABLE"}),
}


class LifecycleCASFailure(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _identity(context: dict[str, Any]) -> tuple[str, str, int, str]:
    contract = context["contract"]
    return (
        str(contract["capsule_sha256"]),
        str(contract["attempt_run_id"]),
        int(contract["instance_id"]),
        str(context["content_address"]["digest"]),
    )


def validate(value: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    capsule, attempt, instance, stop_context = _identity(context)
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("version") != 2
        or value.get("state") not in STATES
        or isinstance(value.get("sequence"), bool)
        or not isinstance(value.get("sequence"), int)
        or value["sequence"] < 1
        or value.get("capsule_sha256") != capsule
        or value.get("attempt_run_id") != attempt
        or value.get("instance_id") != instance
        or value.get("stop_context_sha256") != stop_context
        or not isinstance(value.get("ancestor_sha256"), list)
        or len(value["ancestor_sha256"]) != value["sequence"] - 1
        or len(value["ancestor_sha256"]) > MAX_COMPLETE_ANCESTORS
        or len(set(value["ancestor_sha256"])) != len(value["ancestor_sha256"])
        or any(
            not isinstance(item, str) or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value["ancestor_sha256"]
        )
        or value.get("previous_sha256")
        != (None if value["sequence"] == 1 else value["ancestor_sha256"][-1])
        or value.get("self_sha256") in value["ancestor_sha256"]
        or value.get("self_sha256")
        != digest({key: item for key, item in value.items() if key != "self_sha256"})
    ):
        raise LifecycleCASFailure("lifecycle identity or self digest changed")
    return value


def read(path: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LifecycleCASFailure("lifecycle state is unreadable") from error
    return validate(value, context)


def _atomic_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def transition(
    path: Path,
    context: dict[str, Any],
    state: str,
    *,
    reason: str,
    expected_sequence: int | None,
    terminal_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CAS transition under a process-shared lock with terminal absorption."""

    lock_path = path.with_name(f".{path.name}.cas.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        prior = read(path, context)
        prior_sequence = 0 if prior is None else int(prior["sequence"])
        if expected_sequence is not None and prior_sequence != expected_sequence:
            raise LifecycleCASFailure(
                f"lifecycle CAS sequence mismatch {prior_sequence}!={expected_sequence}"
            )
        prior_state = None if prior is None else str(prior["state"])
        if state not in TRANSITIONS.get(prior_state, frozenset()):
            raise LifecycleCASFailure(f"forbidden lifecycle transition {prior_state!r}->{state}")
        if prior_state in TERMINAL_STATES and state == prior_state:
            return prior
        if prior_sequence >= MAX_COMPLETE_ANCESTORS + 1:
            raise LifecycleCASFailure("complete lifecycle ancestry bound exhausted")
        if state == "TERMINAL_STOPPED_PRESERVED" and (
            not isinstance(terminal_receipt, dict)
            or terminal_receipt.get("receipt_durable") is not True
            or terminal_receipt.get("terminal_state") != "stopped"
            or terminal_receipt.get("consecutive_terminal_observations") != 3
        ):
            raise LifecycleCASFailure("stopped terminal requires three durable observations")
        if state == "TERMINAL_DESTROYED_DURABLE" and (
            not isinstance(terminal_receipt, dict)
            or terminal_receipt.get("terminal_state") != "destroyed"
            or terminal_receipt.get("destroy_gate")
            not in {"COMPLETE_ARTIFACT_DURABILITY", "PRE_SCIENCE_ZERO_BODY_DURABILITY"}
            or terminal_receipt.get("consecutive_absences") < 3
        ):
            raise LifecycleCASFailure("destroy terminal requires durable gate and three absences")
        capsule, attempt, instance, stop_context = _identity(context)
        body = {
            "schema": SCHEMA,
            "version": 2,
            "state": state,
            "sequence": prior_sequence + 1,
            "capsule_sha256": capsule,
            "attempt_run_id": attempt,
            "instance_id": instance,
            "stop_context_sha256": stop_context,
            "previous_sha256": None if prior is None else prior["self_sha256"],
            "ancestor_sha256": (
                [] if prior is None
                else [*prior["ancestor_sha256"], prior["self_sha256"]]
            ),
            "reason": reason,
            "wall_time_unix": time.time(),
            "terminal_receipt_sha256": (
                None if terminal_receipt is None else digest(terminal_receipt)
            ),
        }
        value = {**body, "self_sha256": digest(body)}
        _atomic_replace(path, value)
        return value
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def reconcile(local: dict[str, Any], durable: dict[str, Any]) -> dict[str, Any]:
    """Select only a winner whose complete digest ancestry proves the loser."""

    local_sequence = int(local["sequence"])
    durable_sequence = int(durable["sequence"])
    if local_sequence == durable_sequence:
        if local.get("self_sha256") != durable.get("self_sha256"):
            raise LifecycleCASFailure("same-sequence lifecycle split brain")
        return local
    winner, loser = (
        (local, durable) if local_sequence > durable_sequence else (durable, local)
    )
    loser_sequence = int(loser["sequence"])
    winner_ancestors = winner.get("ancestor_sha256")
    loser_ancestors = loser.get("ancestor_sha256")
    if (
        not isinstance(winner_ancestors, list)
        or not isinstance(loser_ancestors, list)
        or winner_ancestors[: len(loser_ancestors)] != loser_ancestors
        or len(winner_ancestors) <= loser_sequence - 1
        or winner_ancestors[loser_sequence - 1] != loser.get("self_sha256")
    ):
        raise LifecycleCASFailure("lifecycle winner does not prove loser ancestry")
    if loser.get("state") in TERMINAL_STATES and winner.get("state") != loser.get("state"):
        raise LifecycleCASFailure("durable reconciliation would regress terminal state")
    return winner


def install_reconciled(
    path: Path,
    context: dict[str, Any],
    *,
    expected_local: dict[str, Any],
    durable: dict[str, Any],
) -> dict[str, Any]:
    """Converge the local pointer to an ancestry-proven durable winner."""

    lock_path = path.with_name(f".{path.name}.cas.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = read(path, context)
        if current is None or current.get("self_sha256") != expected_local.get("self_sha256"):
            raise LifecycleCASFailure("local lifecycle changed before reconciliation install")
        validate(durable, context)
        winner = reconcile(current, durable)
        if winner.get("self_sha256") == durable.get("self_sha256"):
            _atomic_replace(path, durable)
        return winner
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def s3_put_cas_flags(*, prior_etag: str | None) -> list[str]:
    return ["--if-none-match", "*"] if prior_etag is None else ["--if-match", prior_etag]
