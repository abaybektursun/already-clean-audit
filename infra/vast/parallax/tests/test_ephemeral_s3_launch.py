from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from infra.vast.parallax.ephemeral_s3_launch import (
    EphemeralLaunchError,
    _confirm_account_zero,
    _create_arguments,
    _enforce_instance_nonbillable,
    _enforce_instances_destroyed,
    _finalize_lifecycle,
    _require_guardian_live,
    _require_worker_can_still_publish,
    _stage_atomic,
    _stage,
    _terminal_worker_cleanup,
    _wait_remote_json_exact,
    _worker_start_command,
    select_cheapest_offer,
)


def _offer_contract():
    return {
        "disk_gb": 16,
        "minimum_reliability": 0.98,
        "minimum_inet_down_mbps": 100,
        "minimum_cuda": 12,
        "price_ceiling_dph": 0.2,
    }


def test_offer_selection_filters_h100_and_takes_literal_cheapest_non_h100():
    rows = [
        {"id": 1, "gpu_name": "H100 NVL", "dph_total": 0.01, "reliability": 0.999},
        {"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.12, "reliability": 0.99},
        {"id": 3, "gpu_name": "RTX 3090", "dph_total": 0.08, "reliability": 0.99},
    ]
    with patch("infra.vast.parallax.ephemeral_s3_launch._vast", return_value=rows):
        packet = select_cheapest_offer(role="worker", contract=_offer_contract())
    assert packet["selected"]["id"] == 3
    assert packet["candidate_count"] == 2


def test_offer_selection_fails_closed_above_price_ceiling():
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._vast",
        return_value=[{"id": 3, "gpu_name": "RTX 3090", "dph_total": 0.21, "reliability": 0.99}],
    ):
        with pytest.raises(EphemeralLaunchError, match="price ceiling"):
            select_cheapest_offer(role="worker", contract=_offer_contract())


def test_worker_offer_excludes_the_guardian_machine():
    rows = [
        {"id": 2, "machine_id": 10, "gpu_name": "RTX 4090", "dph_total": 0.05, "reliability": 0.99},
        {"id": 3, "machine_id": 11, "gpu_name": "RTX 3090", "dph_total": 0.08, "reliability": 0.99},
    ]
    with patch("infra.vast.parallax.ephemeral_s3_launch._vast", return_value=rows):
        packet = select_cheapest_offer(
            role="worker", contract=_offer_contract(), excluded_machine_ids={10}
        )
    assert packet["selected"]["id"] == 3
    assert packet["excluded_machine_ids"] == [10]


def test_worker_create_arguments_have_no_volume_and_arm_provider_destroy():
    spec = {
        "worker": {
            "disk_gb": 32,
            "hard_destroy_seconds": 300,
            "template_hash": "template-hash",
            "image": "private/image@sha256:digest",
        }
    }
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._ephemeral_worker_registry_login",
        return_value="-u AWS -p token registry",
    ):
        arguments = _create_arguments(role="worker", spec=spec, run_id="rehearsal-run")
    rendered = " ".join(arguments)
    assert "--onstart-cmd" in arguments
    assert "--link-volume" not in arguments
    assert "--create-volume" not in arguments
    assert "--template_hash template-hash" in rendered
    assert "private/image@sha256:digest" in rendered


def test_stock_transport_worker_has_no_registry_or_managed_template():
    spec = {
        "worker": {
            "launch_mode": "stock_direct_ssh",
            "disk_gb": 10,
            "hard_destroy_seconds": 300,
            "image": "vastai/base-image@sha256:stock",
        }
    }
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._ephemeral_worker_registry_login"
    ) as registry:
        arguments = _create_arguments(role="worker", spec=spec, run_id="transport-run")
    assert "--ssh" in arguments and "--direct" in arguments
    assert "--template_hash" not in arguments
    assert "--login" not in arguments
    registry.assert_not_called()


def test_controller_fails_worker_bootstrap_if_guardian_disappears():
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._show_instance", return_value=None
    ) as show, patch("infra.vast.parallax.ephemeral_s3_launch.time.sleep"):
        with pytest.raises(EphemeralLaunchError, match="guardian is not live"):
            _require_guardian_live(11)
    assert show.call_count == 3


def test_controller_tolerates_one_transient_empty_guardian_listing():
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._show_instance",
        side_effect=[None, {"actual_status": "running"}],
    ) as show, patch("infra.vast.parallax.ephemeral_s3_launch.time.sleep"):
        _require_guardian_live(11)
    assert show.call_count == 2


def test_controller_does_not_wait_for_s3_after_guardian_stops_worker():
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._show_instance",
        return_value={"actual_status": "running", "intended_status": "stopped"},
    ):
        with pytest.raises(EphemeralLaunchError, match="terminal before S3"):
            _require_worker_can_still_publish(22)


def test_worker_start_creates_runtime_directory_before_shell_redirection():
    command = _worker_start_command()
    assert command.index("mkdir -p /workspace/parallax") < command.index(">/workspace/parallax/worker.log")
    assert "kill -0 $worker_pid" in command


def test_released_nondurable_worker_is_stopped_and_never_destroyed():
    with patch("infra.vast.parallax.ephemeral_s3_launch._show_instance") as show, patch(
        "infra.vast.parallax.ephemeral_s3_launch._vast"
    ) as vast:
        packet = _terminal_worker_cleanup(22, job_released=True, s3_durable=False)
    assert packet["action"] == "stop_preserve"
    show.assert_not_called()
    vast.assert_called_once_with("stop", "instance", "22", check=False)


def test_pre_release_or_durable_worker_is_destroyed():
    with patch("infra.vast.parallax.ephemeral_s3_launch._show_instance") as show, patch(
        "infra.vast.parallax.ephemeral_s3_launch._vast"
    ) as vast:
        first = _terminal_worker_cleanup(22, job_released=False, s3_durable=False)
        second = _terminal_worker_cleanup(23, job_released=True, s3_durable=True)
    assert first["action"] == second["action"] == "destroy"
    show.assert_not_called()
    assert vast.call_count == 2


def test_best_effort_destroy_is_not_suppressed_by_an_empty_listing():
    with patch("infra.vast.parallax.ephemeral_s3_launch._show_instance") as show, patch(
        "infra.vast.parallax.ephemeral_s3_launch._vast"
    ) as vast:
        from infra.vast.parallax.ephemeral_s3_launch import _destroy_best_effort

        _destroy_best_effort(44)
    show.assert_not_called()
    vast.assert_called_once_with("destroy", "instance", "44", check=False)


def test_atomic_assignment_stage_never_exposes_a_partial_final_path():
    completed = subprocess.CompletedProcess([], 0, b"", b"")
    with patch("infra.vast.parallax.ephemeral_s3_launch._stage") as stage, patch(
        "infra.vast.parallax.ephemeral_s3_launch._run", return_value=completed
    ) as run:
        _stage_atomic(["ssh"], remote="/run/parallax/assignment.json", content=b"{}")
    stage.assert_called_once_with(
        ["ssh"],
        remote="/run/parallax/assignment.json.controller-pending",
        content=b"{}",
        mode="0600",
    )
    assert run.call_args.args[0][-1] == (
        "mv -f /run/parallax/assignment.json.controller-pending "
        "/run/parallax/assignment.json"
    )


def test_small_remote_stage_retries_one_transient_ssh_failure_idempotently():
    failed = subprocess.CompletedProcess([], 255, b"", b"transient")
    green = subprocess.CompletedProcess([], 0, b"", b"")
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._run", side_effect=[failed, green]
    ) as run, patch("infra.vast.parallax.ephemeral_s3_launch.time.sleep") as sleep:
        _stage(["ssh"], remote="/run/parallax/key", content=b"secret")
    assert run.call_count == 2
    assert all(call.kwargs["input_bytes"] == b"secret" for call in run.call_args_list)
    sleep.assert_called_once_with(1.0)


def test_small_remote_stage_fails_after_three_bounded_attempts():
    failed = subprocess.CompletedProcess([], 255, b"", b"transient")
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._run", return_value=failed
    ) as run, patch("infra.vast.parallax.ephemeral_s3_launch.time.sleep"):
        with pytest.raises(EphemeralLaunchError, match="after 3 idempotent attempts"):
            _stage(["ssh"], remote="/run/parallax/key", content=b"secret")
    assert run.call_count == 3


def test_controller_requires_guardian_process_ack_not_its_own_assignment_file():
    expected = {"status": "ACK", "worker_id": 22}
    completed = subprocess.CompletedProcess([], 0, json.dumps(expected).encode(), b"")
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._run", return_value=completed
    ) as run:
        assert _wait_remote_json_exact(["ssh"], "/opt/parallax/ack.json", expected) == expected
    assert "cat /opt/parallax/ack.json" in run.call_args.args[0][-1]


def test_controller_refuses_guardian_ack_identity_drift():
    completed = subprocess.CompletedProcess([], 0, b'{"worker_id":23}', b"")
    with patch("infra.vast.parallax.ephemeral_s3_launch._run", return_value=completed):
        with pytest.raises(EphemeralLaunchError, match="identity drift"):
            _wait_remote_json_exact(
                ["ssh"], "/opt/parallax/ack.json", {"worker_id": 22}
            )


def test_cleanup_reissues_destroy_after_transient_false_absence():
    observations = [
        None,
        {"actual_status": "loading"},
        None,
        None,
        None,
    ]
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._show_instance",
        side_effect=observations,
    ), patch("infra.vast.parallax.ephemeral_s3_launch._vast") as vast, patch(
        "infra.vast.parallax.ephemeral_s3_launch.time.sleep"
    ):
        proof = _enforce_instances_destroyed([22], seconds=5)
    assert proof["status"] == "GREEN_KNOWN_INSTANCES_CONFIRMED_ABSENT"
    assert proof["reissued_destroy_requests"] == 1
    vast.assert_called_once_with("destroy", "instance", "22", check=False)


def test_released_nondurable_cleanup_reissues_stop_until_nonbillable():
    observations = [
        {"actual_status": "running"},
        {"actual_status": "stopped"},
        {"actual_status": "stopped"},
        {"actual_status": "stopped"},
    ]
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._show_instance",
        side_effect=observations,
    ), patch("infra.vast.parallax.ephemeral_s3_launch._vast") as vast, patch(
        "infra.vast.parallax.ephemeral_s3_launch.time.sleep"
    ):
        proof = _enforce_instance_nonbillable(22, seconds=5)
    assert proof["status"] == "GREEN_WORKER_CONFIRMED_NONBILLABLE"
    assert proof["reissued_stop_requests"] == 1
    vast.assert_called_once_with("stop", "instance", "22", check=False)


def test_finalizer_proves_pre_release_worker_and_guardian_absent():
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._terminal_worker_cleanup",
        return_value={"action": "destroy", "reason": "job_not_released"},
    ) as cleanup, patch(
        "infra.vast.parallax.ephemeral_s3_launch._destroy_best_effort"
    ) as destroy, patch(
        "infra.vast.parallax.ephemeral_s3_launch._enforce_instances_destroyed",
        return_value={"status": "GREEN_KNOWN_INSTANCES_CONFIRMED_ABSENT"},
    ) as enforce:
        packet = _finalize_lifecycle(
            worker_id=22,
            guardian_id=11,
            job_released=False,
            s3_durable=False,
        )
    cleanup.assert_called_once_with(22, job_released=False, s3_durable=False)
    destroy.assert_called_once_with(11)
    enforce.assert_called_once_with([22, 11])
    assert packet["confirmation"]["status"] == "GREEN_KNOWN_INSTANCES_CONFIRMED_ABSENT"


def test_whole_account_zero_needs_three_consecutive_observations():
    zero = {
        "active_instances": 0,
        "active_volumes": 0,
        "instance_ids": [],
        "volume_ids": [],
    }
    active = {**zero, "active_instances": 1, "instance_ids": [22]}
    with patch(
        "infra.vast.parallax.ephemeral_s3_launch._account_zero",
        side_effect=[zero, active, zero, zero, zero],
    ) as account, patch("infra.vast.parallax.ephemeral_s3_launch.time.sleep"):
        proof = _confirm_account_zero(seconds=5)
    assert account.call_count == 5
    assert proof["status"] == "GREEN_ACCOUNT_ZERO_CONFIRMED"
    assert proof["consecutive_observations"] == 3
