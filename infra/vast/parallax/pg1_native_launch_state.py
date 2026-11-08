"""Pure contracts for the canonical PG1 external launch transaction state."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


SCHEMA = "pg1-native-canonical-launch-state/v1"
MAX_ANCESTORS = 64
SCIENCE_DISABLED = "SCIENCE_DISABLED"
SCIENCE_RELEASED = "SCIENCE_RELEASED"
SCIENCE_RELEASE_UNKNOWN = "SCIENCE_RELEASE_UNKNOWN"
SCIENCE_STATES = frozenset(
    {SCIENCE_DISABLED, SCIENCE_RELEASED, SCIENCE_RELEASE_UNKNOWN}
)
TERMINAL_PHASES = frozenset(
    {"TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE"}
)
TRANSITIONS = {
    None: frozenset({"PRELAUNCH_CLEANUP_INTENT"}),
    "PRELAUNCH_CLEANUP_INTENT": frozenset(
        {"CREATE_INTENT_ARMED", "CLEANUP_STOPPING", "CLEANUP_DESTROYING"}
    ),
    "CREATE_INTENT_ARMED": frozenset(
        {"WORKER_ID_BOUND_SCIENCE_DISABLED", "CLEANUP_STOPPING", "CLEANUP_DESTROYING"}
    ),
    "WORKER_ID_BOUND_SCIENCE_DISABLED": frozenset(
        {"INSTANCE_CONTEXT_BOUND", "CLEANUP_STOPPING", "CLEANUP_DESTROYING"}
    ),
    "INSTANCE_CONTEXT_BOUND": frozenset(
        {"HANDOFF_BOUND_DEADMAN_ACKED", "CLEANUP_STOPPING", "CLEANUP_DESTROYING"}
    ),
    "HANDOFF_BOUND_DEADMAN_ACKED": frozenset(
        {"START_COMMIT_INTENT", "CLEANUP_STOPPING", "CLEANUP_DESTROYING"}
    ),
    "START_COMMIT_INTENT": frozenset(
        {"SCIENCE_RELEASED", "SCIENCE_RELEASE_UNKNOWN", "START_COMMIT_PROVEN_ABSENT",
         "CLEANUP_STOPPING", "CLEANUP_DESTROYING"}
    ),
    "START_COMMIT_PROVEN_ABSENT": frozenset({"CLEANUP_STOPPING", "CLEANUP_DESTROYING"}),
    "SCIENCE_RELEASED": frozenset({"CLEANUP_STOPPING", "CLEANUP_DESTROYING"}),
    "SCIENCE_RELEASE_UNKNOWN": frozenset({"CLEANUP_STOPPING"}),
    "CLEANUP_STOPPING": frozenset({"CLEANUP_DESTROYING", *TERMINAL_PHASES}),
    "CLEANUP_DESTROYING": frozenset(
        {"CLEANUP_STOPPING", "TERMINAL_DESTROYED_DURABLE"}
    ),
    "TERMINAL_STOPPED_PRESERVED": frozenset(),
    "TERMINAL_DESTROYED_DURABLE": frozenset(),
}


class LaunchStateError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def state_key(packet_sha256: str) -> str:
    return f"parallax/authorizations/pg1-native-v3/control-state/{packet_sha256}.json"


def start_commit_key(packet_sha256: str) -> str:
    return f"parallax/authorizations/pg1-native-v3/start-commits/{packet_sha256}.json"


def terminal_receipt_key(packet_sha256: str, action: str) -> str:
    if action not in {"stop", "destroy"}:
        raise LaunchStateError("terminal receipt action changed")
    return (
        "parallax/authorizations/pg1-native-v3/control-terminal-receipts/"
        f"{packet_sha256}/{action}.json"
    )


def _identity_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    contract = packet.get("contract") if isinstance(packet, dict) else None
    if (
        not isinstance(contract, dict)
        or packet.get("content_address", {}).get("algorithm") != "sha256"
        or not isinstance(packet.get("content_address", {}).get("digest"), str)
    ):
        raise LaunchStateError("launch packet identity unavailable")
    return {
        "packet_sha256": packet["content_address"]["digest"],
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"],
        "offer_id": contract["offer_id"],
        "launch_label": contract["launch_label"],
    }


def validate(
    value: dict[str, Any], *, packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("phase") not in TRANSITIONS
        or value.get("science_state") not in SCIENCE_STATES
        or isinstance(value.get("sequence"), bool)
        or not isinstance(value.get("sequence"), int)
        or value["sequence"] < 1
        or not isinstance(value.get("ancestor_state_sha256"), list)
        or len(value["ancestor_state_sha256"]) != value["sequence"] - 1
        or len(value["ancestor_state_sha256"]) > MAX_ANCESTORS
        or len(set(value["ancestor_state_sha256"]))
        != len(value["ancestor_state_sha256"])
        or value.get("previous_state_sha256") != (
            None if value["sequence"] == 1 else value["ancestor_state_sha256"][-1]
        )
        or value.get("previous_raw_body_sha256") is not None
        and (
            not isinstance(value.get("previous_raw_body_sha256"), str)
            or len(value["previous_raw_body_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value["previous_raw_body_sha256"]
            )
        )
        or value["sequence"] == 1 and value.get("previous_raw_body_sha256") is not None
        or value["sequence"] > 1 and value.get("previous_raw_body_sha256") is None
        or value.get("self_sha256") in value["ancestor_state_sha256"]
        or value.get("self_sha256") != digest(
            {key: item for key, item in value.items() if key != "self_sha256"}
        )
        or not isinstance(value.get("packet_sha256"), str)
        or len(value["packet_sha256"]) != 64
        or not isinstance(value.get("capsule_sha256"), str)
        or len(value["capsule_sha256"]) != 64
        or not isinstance(value.get("attempt_run_id"), str)
        or not value["attempt_run_id"]
        or isinstance(value.get("offer_id"), bool)
        or not isinstance(value.get("offer_id"), int)
        or value["offer_id"] <= 0
        or not isinstance(value.get("launch_label"), str)
        or not value["launch_label"]
        or value.get("terminal_receipt") is not None
        and not isinstance(value.get("terminal_receipt"), dict)
    ):
        raise LaunchStateError("canonical launch state identity changed")
    if packet is not None:
        expected = _identity_from_packet(packet)
        if any(value.get(field) != expected[field] for field in expected):
            raise LaunchStateError("canonical launch state packet binding changed")
    phase = str(value["phase"])
    science = str(value["science_state"])
    destroy_reference = value.get("destroy_request")
    destroy_body = value.get("destroy_request_body")
    terminal_reference = value.get("terminal_receipt")
    if (
        phase in {
            "PRELAUNCH_CLEANUP_INTENT", "CREATE_INTENT_ARMED",
            "WORKER_ID_BOUND_SCIENCE_DISABLED", "INSTANCE_CONTEXT_BOUND",
            "HANDOFF_BOUND_DEADMAN_ACKED", "START_COMMIT_INTENT",
            "START_COMMIT_PROVEN_ABSENT",
        }
        and science != SCIENCE_DISABLED
        or phase == "SCIENCE_RELEASED" and science != SCIENCE_RELEASED
        or phase == "SCIENCE_RELEASE_UNKNOWN" and science != SCIENCE_RELEASE_UNKNOWN
        or phase == "CLEANUP_DESTROYING" and (
            not isinstance(destroy_reference, dict)
            or set(destroy_reference) != {"sha256", "storage"}
            or not isinstance(destroy_reference.get("sha256"), str)
            or len(destroy_reference["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in destroy_reference["sha256"]
            )
            or destroy_reference.get("storage") != "CANONICAL_STATE_RAW_BODY"
            or not isinstance(destroy_body, dict)
            or digest(destroy_body) != destroy_reference.get("sha256")
        )
        or phase in TERMINAL_PHASES and (
            not isinstance(terminal_reference, dict)
            or set(terminal_reference) != {
                "key", "sha256", "version_id", "etag", "checksum_sha256", "action",
            }
            or terminal_reference.get("action") not in {"stop", "destroy"}
            or not isinstance(terminal_reference.get("key"), str)
            or not terminal_reference["key"]
            or not isinstance(terminal_reference.get("sha256"), str)
            or len(terminal_reference["sha256"]) != 64
            or not all(
                isinstance(terminal_reference.get(field), str)
                and bool(terminal_reference[field])
                for field in ("version_id", "etag", "checksum_sha256")
            )
        )
    ):
        raise LaunchStateError("canonical launch state science or terminal binding changed")
    return value


def initial(
    packet: dict[str, Any], *, authorization_sha256: str,
    authorization_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    identity = _identity_from_packet(packet)
    body = {
        "schema": SCHEMA, "sequence": 1,
        "phase": "PRELAUNCH_CLEANUP_INTENT",
        "science_state": SCIENCE_DISABLED, **identity,
        "instance_id": None, "machine_id": None, "container_id": None,
        "stop_context_sha256": None, "handoff_layout_sha256": None,
        "start_commit_sha256": None, "authorization_sha256": authorization_sha256,
        "authorization_receipt_sha256": authorization_receipt_sha256,
        "mount_inventory_sha256": None,
        "destroy_request": None,
        "destroy_request_body": None,
        "terminal_receipt": None, "previous_state_sha256": None,
        "previous_raw_body_sha256": None,
        "ancestor_state_sha256": [], "wall_time_unix": time.time(),
    }
    return validate({**body, "self_sha256": digest(body)}, packet=packet)


def transition(
    prior: dict[str, Any], phase: str, *, science_state: str | None = None,
    terminal_receipt: dict[str, Any] | None = None, **fields: Any,
) -> dict[str, Any]:
    validate(prior)
    if phase not in TRANSITIONS.get(str(prior["phase"]), frozenset()):
        raise LaunchStateError(
            f"forbidden canonical launch transition {prior['phase']}->{phase}"
        )
    if prior["sequence"] >= MAX_ANCESTORS + 1:
        raise LaunchStateError("canonical launch state ancestry bound exhausted")
    selected_science = str(science_state or prior["science_state"])
    if (
        prior["science_state"] == SCIENCE_RELEASED
        and selected_science != SCIENCE_RELEASED
        or prior["science_state"] == SCIENCE_RELEASE_UNKNOWN
        and selected_science == SCIENCE_DISABLED
    ):
        raise LaunchStateError("canonical launch science state would regress")
    mutable = {
        key: item for key, item in prior.items()
        if key not in {
            "self_sha256", "sequence", "phase", "science_state",
            "previous_state_sha256", "previous_raw_body_sha256",
            "ancestor_state_sha256", "wall_time_unix",
        }
    }
    mutable.update(fields)
    body = {
        "schema": SCHEMA, "sequence": int(prior["sequence"]) + 1,
        "phase": phase, "science_state": selected_science, **mutable,
        "terminal_receipt": (
            terminal_receipt if terminal_receipt is not None
            else mutable.get("terminal_receipt")
        ),
        "previous_state_sha256": prior["self_sha256"],
        "previous_raw_body_sha256": hashlib.sha256(
            canonical_bytes(prior)
        ).hexdigest(),
        "ancestor_state_sha256": [
            *prior["ancestor_state_sha256"], prior["self_sha256"]
        ],
        "wall_time_unix": time.time(),
    }
    return validate({**body, "self_sha256": digest(body)})


def proves_descendant(winner: dict[str, Any], loser: dict[str, Any]) -> bool:
    validate(winner); validate(loser)
    if winner["sequence"] == loser["sequence"]:
        return winner["self_sha256"] == loser["self_sha256"]
    return (
        winner["sequence"] > loser["sequence"]
        and len(winner["ancestor_state_sha256"]) >= loser["sequence"]
        and winner["ancestor_state_sha256"][loser["sequence"] - 1]
        == loser["self_sha256"]
        and winner["ancestor_state_sha256"][: len(loser["ancestor_state_sha256"])]
        == loser["ancestor_state_sha256"]
    )


def proves_direct_transition(winner: dict[str, Any], prior: dict[str, Any]) -> bool:
    """Prove one legal state edge, including the exact prior raw body digest."""

    validate(winner); validate(prior)
    return (
        winner["phase"] in TRANSITIONS.get(prior["phase"], frozenset())
        and winner["sequence"] == prior["sequence"] + 1
        and winner["previous_state_sha256"] == prior["self_sha256"]
        and winner["previous_raw_body_sha256"]
        == hashlib.sha256(canonical_bytes(prior)).hexdigest()
        and winner["ancestor_state_sha256"]
        == [*prior["ancestor_state_sha256"], prior["self_sha256"]]
    )
