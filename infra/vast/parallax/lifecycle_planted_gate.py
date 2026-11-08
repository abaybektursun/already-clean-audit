"""Weight-free plants for every lifecycle teardown branch."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.error
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .lifecycle_guardian import GuardianPolicy, run_guardian
from .lifecycle_supervisor import (
    LifecycleError,
    LifecyclePolicy,
    supervise,
    wait_for_execution_release,
)


class FakeJob:
    def __init__(self, backend: "FakeBackend", exit_code: int | None, exit_after_polls: int):
        self.backend = backend
        self.exit_code = exit_code
        self.exit_after_polls = exit_after_polls
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        if self.exit_code is not None and self.polls > self.exit_after_polls:
            return self.exit_code
        return None

    def wait(self) -> int:
        if self.exit_code is None:
            raise AssertionError("never-ending fake job cannot be waited")
        return self.exit_code


class FakeBackend:
    def __init__(
        self,
        *,
        exit_code: int | None,
        exit_after_polls: int = 1,
        durability_green: bool = True,
        durability_green_after_proofs: int = 1,
        destroy_green: bool = True,
        mirror_error: bool = False,
    ):
        self.exit_code = exit_code
        self.exit_after_polls = exit_after_polls
        self.durability_green = durability_green
        self.durability_green_after_proofs = durability_green_after_proofs
        self.durability_proofs = 0
        self.destroy_green = destroy_green
        self.mirror_error = mirror_error
        self.clock = 0.0
        self.events: list[str] = []
        self.receipt: dict[str, Any] | None = None

    def start_job(self) -> FakeJob:
        self.events.append("job_started")
        return FakeJob(self, self.exit_code, self.exit_after_polls)

    def heartbeat(self, _payload: dict[str, Any]) -> None:
        self.events.append("heartbeat")

    def mirror_once(
        self, *, final: bool = False, deadline_monotonic: float | None = None
    ) -> dict[str, Any]:
        self.events.append("mirror_final" if final else "mirror_live")
        if self.mirror_error:
            raise OSError("planted mirror failure")
        return {"final": final, "copied_count": 1}

    def prove_durability(self, *, job_exit_code: int) -> dict[str, Any]:
        self.durability_proofs += 1
        green = self.durability_green and self.durability_proofs >= self.durability_green_after_proofs
        self.events.append("durability_proved" if green else "durability_unproven")
        return {"status": "GREEN" if green else "UNPROVEN", "job_exit_code": job_exit_code}

    def destroy_h100(self) -> dict[str, Any]:
        self.events.append("destroy")
        return {"status": "DESTROYED" if self.destroy_green else "STOPPED_AFTER_DESTROY_FAILURE"}

    def stop_h100(self, *, reason: str) -> dict[str, Any]:
        self.events.append(f"stop:{reason}")
        return {"status": "STOPPED", "reason": reason}

    def write_receipt(self, payload: dict[str, Any]) -> None:
        self.events.append("receipt")
        self.receipt = payload

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def monotonic(self) -> float:
        return self.clock


def _policy(*, maximum_runtime_seconds: int = 10) -> LifecyclePolicy:
    return LifecyclePolicy(
        run_id="plant",
        maximum_runtime_seconds=maximum_runtime_seconds,
        mirror_interval_seconds=1.0,
        heartbeat_timeout_seconds=4.0,
        artifact_copy_timeout_seconds=5.0,
        maximum_finalization_seconds=5.0,
        final_sync_attempts=2,
        final_sync_settle_seconds=0.0,
        required_sentinels=("result.json",),
    )


def run(output: Path) -> dict[str, Any]:
    cases = {}

    success = FakeBackend(exit_code=0, exit_after_polls=2)
    success_packet = supervise(_policy(), success)
    cases["success_durable_before_destroy"] = {
        "status": "GREEN" if success_packet["status"] == "DURABLE_AND_DESTROYED"
        and success.events.index("durability_proved") < success.events.index("destroy")
        and success.events.index("destroy") < success.events.index("receipt")
        and success.events.count("mirror_final") == 1 else "RED",
        "events": success.events,
    }

    incomplete_first_pass = FakeBackend(
        exit_code=0,
        durability_green=True,
        durability_green_after_proofs=2,
    )
    incomplete_packet = supervise(_policy(), incomplete_first_pass)
    cases["exact_artifacts_retry_only_while_incomplete"] = {
        "status": "GREEN" if incomplete_packet["status"] == "DURABLE_AND_DESTROYED"
        and incomplete_first_pass.events.count("mirror_final") == 2
        and incomplete_first_pass.events.index("durability_unproven")
        < incomplete_first_pass.events.index("durability_proved")
        < incomplete_first_pass.events.index("destroy") else "RED",
        "events": incomplete_first_pass.events,
    }

    failed_job = FakeBackend(exit_code=7, exit_after_polls=1)
    failed_packet = supervise(_policy(), failed_job)
    cases["failed_job_salvaged_then_destroyed"] = {
        "status": "GREEN" if failed_packet["status"] == "DURABLE_AND_DESTROYED"
        and failed_packet["job_exit_code"] == 7 else "RED",
        "events": failed_job.events,
    }

    upload_failure = FakeBackend(exit_code=0, durability_green=False)
    upload_packet = supervise(_policy(), upload_failure)
    cases["unproven_durability_stops_and_preserves_disk"] = {
        "status": "GREEN" if upload_packet["status"] == "STOPPED_DISK_PRESERVED"
        and "destroy" not in upload_failure.events
        and "stop:durability_unproven" in upload_failure.events else "RED",
        "events": upload_failure.events,
    }

    timeout = FakeBackend(exit_code=None)
    timeout_packet = supervise(_policy(maximum_runtime_seconds=2), timeout)
    cases["timeout_stops_without_waiting_for_upload"] = {
        "status": "GREEN" if timeout_packet["status"] == "STOPPED_TIMEOUT_DISK_PRESERVED"
        and "stop:hard_runtime_deadline" in timeout.events
        and "durability_proved" not in timeout.events
        and "destroy" not in timeout.events else "RED",
        "events": timeout.events,
    }

    destroy_failure = FakeBackend(exit_code=0, destroy_green=False)
    destroy_packet = supervise(_policy(), destroy_failure)
    cases["destroy_failure_is_visible"] = {
        "status": "GREEN" if destroy_packet["status"] == "DURABLE_TEARDOWN_DEGRADED" else "RED",
        "events": destroy_failure.events,
    }

    internal_failure = FakeBackend(exit_code=0, exit_after_polls=2, mirror_error=True)
    internal_packet = supervise(_policy(), internal_failure)
    cases["unexpected_exception_stops_immediately"] = {
        "status": "GREEN" if internal_packet["status"] == "STOPPED_INTERNAL_FAILURE_DISK_PRESERVED"
        and "stop:supervisor_exception" in internal_failure.events
        and internal_failure.events.index("stop:supervisor_exception") < internal_failure.events.index("receipt")
        and "destroy" not in internal_failure.events else "RED",
        "events": internal_failure.events,
    }

    release_clock = {"value": 0.0}

    def release_sleep(seconds: float) -> None:
        release_clock["value"] += seconds

    released = wait_for_execution_release(
        output.parent / "intentionally-missing-execution-release.marker",
        timeout_seconds=5,
        monotonic=lambda: release_clock["value"],
        sleep=release_sleep,
    )
    cases["missing_execution_release_never_starts_job"] = {
        "status": "GREEN" if released is False and release_clock["value"] >= 5 else "RED"
    }

    unsafe_contract = {
        "schema_version": 1,
        "status": "FROZEN_RUNTIME",
        "run_id": "plant",
        "policy": {
            "maximum_runtime_seconds": 10,
            "mirror_interval_seconds": 10,
            "heartbeat_timeout_seconds": 20,
            "artifact_copy_timeout_seconds": 5,
            "maximum_finalization_seconds": 5,
            "final_sync_attempts": 2,
            "final_sync_settle_seconds": 0,
            "required_sentinels": ["../escape"],
        },
        "teardown": {
            "durability_before_destroy": False,
            "stop_on_unproven_durability": False,
            "h100_api_key_scope": "account_wide",
        },
    }
    rejected = False
    try:
        LifecyclePolicy.from_contract(unsafe_contract)
    except LifecycleError:
        rejected = True
    cases["unsafe_contract_refused"] = {"status": "GREEN" if rejected else "RED"}

    class GuardianClock:
        def __init__(self) -> None:
            self.value = 0.0

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    with tempfile.TemporaryDirectory(prefix="parallax-guardian-plant-") as temporary:
        state_parent = Path(temporary)
        secret = state_parent / "lifecycle.key"
        secret.write_text("planted-scoped-key\n", encoding="utf-8")
        secret.chmod(0o600)

        def guardian_policy(state_name: str) -> GuardianPolicy:
            return GuardianPolicy(
                run_id="plant",
                worker_label="parallax-worker-plant",
                state_root=state_parent / state_name,
                api_key_file=secret,
                discovery_timeout_seconds=30,
                provider_loading_timeout_seconds=300,
                billable_prearm_timeout_seconds=30,
                heartbeat_timeout_seconds=5,
                hard_runtime_seconds=60,
                poll_seconds=1.0,
            )

        loading_clock = GuardianClock()
        loading_packet = run_guardian(
            guardian_policy("loading"),
            list_instances=lambda _key: [{
                "id": 123,
                "label": "parallax-worker-plant",
                "actual_status": "loading" if loading_clock.value < 250 else "running",
            }],
            action=lambda **_kwargs: {"status": "STOPPED"},
            monotonic=lambda: loading_clock.value,
            wall_time=lambda: 1000.0 + loading_clock.value,
            sleep=loading_clock.sleep,
        )
        cases["provider_loading_does_not_consume_billable_prearm"] = {
            "status": "GREEN" if loading_packet["reason"] == "billable_prearm_timeout"
            and loading_clock.value == 279
            and loading_packet["billable_clock_started"] is True else "RED",
            "stop_at_seconds": loading_clock.value,
            "reason": loading_packet["reason"],
        }

        outage_clock = GuardianClock()
        outage_calls = 0

        def loading_then_outage(_key: str) -> list[dict[str, Any]]:
            nonlocal outage_calls
            outage_calls += 1
            if outage_calls == 1:
                return [{
                    "id": 124,
                    "label": "parallax-worker-plant",
                    "actual_status": "loading",
                }]
            raise urllib.error.HTTPError("https://vast.invalid", 429, "limited", {}, None)

        outage_packet = run_guardian(
            guardian_policy("outage"),
            list_instances=loading_then_outage,
            action=lambda **_kwargs: {"status": "STOPPED"},
            monotonic=lambda: outage_clock.value,
            wall_time=lambda: 1000.0 + outage_clock.value,
            sleep=outage_clock.sleep,
        )
        cases["provider_outage_after_discovery_assumes_billable"] = {
            "status": "GREEN" if outage_packet["reason"] == "billable_prearm_timeout"
            and outage_clock.value == 30
            and outage_packet["provider_status"] == "provider_listing_unavailable_assume_billable" else "RED",
            "stop_at_seconds": outage_clock.value,
            "reason": outage_packet["reason"],
        }

        recovery_clock = GuardianClock()
        recovery_calls = 0

        def transient_outage_then_loading(_key: str) -> list[dict[str, Any]]:
            nonlocal recovery_calls
            recovery_calls += 1
            if recovery_calls == 2:
                raise urllib.error.HTTPError("https://vast.invalid", 429, "limited", {}, None)
            return [{
                "id": 125,
                "label": "parallax-worker-plant",
                "actual_status": "loading" if recovery_clock.value < 250 else "running",
            }]

        recovery_packet = run_guardian(
            guardian_policy("recovery"),
            list_instances=transient_outage_then_loading,
            action=lambda **_kwargs: {"status": "STOPPED"},
            monotonic=lambda: recovery_clock.value,
            wall_time=lambda: 1000.0 + recovery_clock.value,
            sleep=recovery_clock.sleep,
        )
        cases["recovered_loading_pauses_without_erasing_uncertain_exposure"] = {
            "status": "GREEN" if recovery_packet["reason"] == "billable_prearm_timeout"
            and recovery_packet["billable_elapsed_seconds"] == 30
            and recovery_clock.value == 277 else "RED",
            "stop_at_seconds": recovery_clock.value,
            "billable_elapsed_seconds": recovery_packet["billable_elapsed_seconds"],
            "reason": recovery_packet["reason"],
        }

    packet = {
        "schema_version": 1,
        "stage": "vast_lifecycle_weight_free_plants",
        "status": "GREEN" if all(row["status"] == "GREEN" for row in cases.values()) else "RED",
        "cases": cases,
        "invariants": {
            "destroy_requires_green_durability": True,
            "unproven_durability_stops_compute_and_preserves_disk": True,
            "hard_deadline_stops_before_any_blocking_salvage": True,
            "job_failure_still_salvages": True,
            "detached_watchdog_required": True,
            "provider_native_TTL_required": True,
            "endpoint_scoped_lifecycle_API_key_only_on_remote_machines": True,
            "handled_supervisor_exceptions_stop_immediately": True,
            "exact_complete_first_pass_stops_without_redundant_sync": True,
            "incomplete_exact_set_never_licenses_destroy": True,
            "missing_phase_two_release_never_starts_job": True,
            "provider_loading_has_a_separate_unbilled_clock": True,
            "provider_status_outage_starts_the_short_billable_clock": True,
            "recovered_loading_pauses_but_never_erases_uncertain_billable_exposure": True,
        },
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.output.resolve())
    print(json.dumps({"status": packet["status"], "cases": len(packet["cases"])}, sort_keys=True))
    return 0 if packet["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
