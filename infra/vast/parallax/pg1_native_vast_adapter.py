"""PG1 binding of the repository's tracked Vast CLI response semantics.

The older lifecycle launchers normalize ``show ... --raw`` as either a row
list, a direct row, or a row under ``instances``.  Instance identity is the
provider's ``id`` field and provider state is ``actual_status`` with
``cur_state`` as the tracked fallback.  Mutating command output is not a
state proof: a zero exit status only licenses subsequent show polling.
"""

from __future__ import annotations

from typing import Any


ADAPTER_SCHEMA = "pg1-vast-tracked-show-normalizer/v1"
TERMINAL_STOP_STATES = frozenset({"stopped", "exited", "offline"})


class VastAdapterError(ValueError):
    """A Vast payload cannot be interpreted with the tracked interface."""


def parse_create_response(payload: Any, *, offer_id: int) -> dict[str, Any]:
    """Bind a successful create response without exposing its scoped secret."""

    if isinstance(offer_id, bool) or not isinstance(offer_id, int) or offer_id <= 0:
        raise VastAdapterError("Vast create offer identity is invalid")
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise VastAdapterError("Vast create response is not successful")
    instance_id = payload.get("new_contract")
    api_key = payload.get("instance_api_key")
    if (
        isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0
        or not isinstance(api_key, str) or len(api_key) != 64
        or any(character not in "0123456789abcdef" for character in api_key)
    ):
        raise VastAdapterError("Vast create response lacks instance identity or scoped key")
    return {"instance_id": instance_id, "offer_id": offer_id, "instance_api_key": api_key}


def validate_created_binding(
    payload: Any, *, instance_id: int, launch_label: str
) -> dict[str, Any]:
    """Validate the canonical post-create row used for worker handoff binding."""

    row = select_instance(payload, instance_id)
    if row is None:
        raise VastAdapterError("created Vast instance is absent during binding")
    machine_id = row.get("machine_id")
    container_id = row.get("container_id")
    if (
        row.get("label") != launch_label
        or isinstance(machine_id, bool) or not isinstance(machine_id, int) or machine_id <= 0
        or not isinstance(container_id, str) or not container_id
    ):
        raise VastAdapterError("created Vast binding row changed")
    normalized = normalize_current_row(row, instance_id=instance_id)
    return {**normalized, "machine_id": machine_id, "container_id": container_id,
            "launch_label": launch_label}


def discover_created_by_label(
    payload: Any, *, launch_label: str
) -> dict[str, Any] | None:
    """Recover an orphan identity after a lost create response using one label."""

    rows = normalize_rows(payload)
    matches = [row for row in rows if row.get("label") == launch_label]
    if not matches:
        return None
    if len(matches) != 1:
        raise VastAdapterError("launch label maps to multiple Vast instances")
    row = matches[0]
    instance_id = _instance_id(row)
    return {"instance_id": instance_id, "state": instance_state(row), "row": row}


def _row_list(value: Any) -> tuple[list[dict[str, Any]], bool]:
    """Return normalized rows and whether the payload was a collection.

    The accepted wrappers are intentionally the same narrow shapes already
    tracked by ``guarded_launch._show_instance`` and the lifecycle guardians.
    A direct row is distinguished from a collection so a wrong direct-row
    identity can never be mistaken for canonical absence.
    """

    if isinstance(value, list):
        if any(not isinstance(row, dict) for row in value):
            raise VastAdapterError("Vast instance listing contains a non-row")
        return list(value), True
    if not isinstance(value, dict):
        raise VastAdapterError("Vast show payload is not a row or row list")
    if "id" in value:
        return [value], False
    if "instances" in value:
        nested = value["instances"]
        if isinstance(nested, dict) and "id" not in nested:
            values = list(nested.values())
            if any(not isinstance(row, dict) for row in values):
                raise VastAdapterError("nested Vast instance map contains a non-row")
            return values, True
        return _row_list(nested)
    raise VastAdapterError("Vast show payload has no tracked instance rows")


def normalize_rows(payload: Any) -> list[dict[str, Any]]:
    """Expose the tracked row normalization for fixture and contract tests."""

    rows, _collection = _row_list(payload)
    return rows


def _instance_id(row: dict[str, Any]) -> int:
    value = row.get("id")
    if isinstance(value, bool):
        raise VastAdapterError("Vast instance id is invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise VastAdapterError("Vast instance id is absent") from error
    if parsed <= 0 or str(value).strip() != str(parsed):
        raise VastAdapterError("Vast instance id is invalid")
    return parsed


def select_instance(payload: Any, instance_id: int) -> dict[str, Any] | None:
    """Select one exact instance; ``None`` is canonical collection absence."""

    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise VastAdapterError("expected Vast instance id is invalid")
    rows, collection = _row_list(payload)
    matches = [row for row in rows if _instance_id(row) == instance_id]
    if len(matches) > 1:
        raise VastAdapterError("Vast show returned duplicate instance identity")
    if matches:
        return matches[0]
    if rows and not collection:
        raise VastAdapterError("Vast direct show row has the wrong instance identity")
    return None


def instance_state(row: dict[str, Any]) -> str:
    """Return the established provider state field without synthetic aliases."""

    state = row.get("actual_status") or row.get("cur_state")
    if not isinstance(state, str) or not state.strip():
        raise VastAdapterError("Vast instance state is absent")
    return state.strip().lower()


def validate_current_row(
    row: dict[str, Any],
    *,
    instance_id: int,
    allowed_states: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Validate exact tracked identity and state for one current row."""

    normalized = normalize_current_row(row, instance_id=instance_id)
    if normalized["state"] not in allowed_states:
        raise VastAdapterError("Vast current row state is not allowed")
    return normalized


def normalize_current_row(
    row: dict[str, Any], *, instance_id: int
) -> dict[str, Any]:
    """Normalize exact identity with any nonempty tracked provider state."""

    if _instance_id(row) != instance_id:
        raise VastAdapterError("Vast current row identity changed")
    return {"id": instance_id, "state": instance_state(row), "row": row}
