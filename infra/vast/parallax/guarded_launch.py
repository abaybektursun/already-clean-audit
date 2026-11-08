"""The only authorized PARALLAX Vast launcher.

This launcher fails closed: both instances receive provider-native TTL guards,
the detached volume-gateway supervisor must arm successfully, and any bootstrap
failure tears down every created instance before returning.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import atomic_json, sha256_file
from .cache_audit import validate_audit_spec
from .task_cache_job import validate_task_cache_spec
from .input_staging_rehearsal import StagingError, validate_staging_job_spec
from .s3_object_mirror import validate_mirror_spec
from .s3_cache_mirror import validate_spec as validate_cache_mirror_spec
from .s3_download_probe import (
    DownloadProbeError,
    validate_download_probe_spec,
    validate_download_url,
)
from .s3_cache_hydration import (
    HydrationError,
    validate_spec as validate_hydration_spec,
    validate_url as validate_hydration_url,
)
from .pg1_model_smoke import ModelSmokeError, validate_spec as validate_model_smoke_spec
from .s3_presign_put import presign_get, presign_put
from .lifecycle_supervisor import (
    API_BASE,
    LifecycleError,
    validate_runtime_contract,
    vast_instance_action,
)


LIFECYCLE_SOURCE_FILES = (
    "infra/__init__.py",
    "infra/vast/__init__.py",
    "infra/vast/parallax/__init__.py",
    "infra/vast/parallax/artifacts.py",
    "infra/vast/parallax/cache_audit.py",
    "infra/vast/parallax/audit_task_bank.py",
    "infra/vast/parallax/materialize_tasks.py",
    "infra/vast/parallax/task_cache_job.py",
    "infra/vast/parallax/input_staging_rehearsal.py",
    "infra/vast/parallax/s3_object_mirror.py",
    "infra/vast/parallax/s3_cache_mirror.py",
    "infra/vast/parallax/s3_download_probe.py",
    "infra/vast/parallax/s3_cache_hydration.py",
    "infra/vast/parallax/pg1_model_smoke.py",
    "infra/vast/parallax/s3_presign_put.py",
    "infra/vast/parallax/lifecycle_guardian.py",
    "infra/vast/parallax/lifecycle_supervisor.py",
    "studies/lineage/cross-lineage/config/data/parallax_truth_sprint_data_v2.yaml",
    "studies/lineage/cross-lineage/config/parallax_input_staging_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_s3_download_probe_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_s3_download_probe_job_v2.json",
    "studies/lineage/cross-lineage/config/parallax_s3_h100_download_probe_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_s3_h100_download_probe_job_v2.json",
    "studies/lineage/cross-lineage/config/parallax_s3_h100_download_probe_job_v3.json",
    "studies/lineage/cross-lineage/config/parallax_pg1_h100_s3_hydration_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_pg1_model_forward_smoke_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_pg1_model_forward_smoke_job_v2.json",
    "studies/lineage/cross-lineage/config/parallax_pg1_model_forward_smoke_job_v3.json",
    "studies/lineage/cross-lineage/config/parallax_h100_stock_control_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_h100_template_control_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_managed_runtime_override_job_v1.json",
    "studies/lineage/cross-lineage/config/parallax_h100_managed_runtime_override_job_v1.json",
)

SCOPED_LIFECYCLE_PERMISSIONS = {
    "api": {
        "instance_read": {"api.instances": {}},
        "instance_write": {"api.instance": {}},
    }
}

# Retired 2026-07-18 after the durable archive was moved to versioned S3 and
# Vast volume 45042496 was deleted.  This module remains importable so the
# historical receipts and tests can be audited, but its volume-backed mutation
# CLI must never be used for new runs.  The successor path is ephemeral Vast
# scratch with direct, checksum-verified S3 input/output.
PERSISTENT_VOLUME_CONTROLLER_RETIRED = True


def require_persistent_volume_controller_active() -> None:
    if PERSISTENT_VOLUME_CONTROLLER_RETIRED:
        raise LifecycleError(
            "the PARALLAX persistent-Vast-volume controller is retired; "
            "use ephemeral Vast scratch and versioned S3 durability"
        )


def _run(
    arguments: list[str],
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments,
        input=input_bytes,
        check=check,
        capture_output=True,
        timeout=timeout_seconds,
    )


def _run_idempotent_ssh(
    ssh: list[str],
    command: str,
    *,
    attempts: int = 3,
    retry_seconds: float = 1.0,
) -> subprocess.CompletedProcess[bytes]:
    """Retry a side-effect-safe marker command across transient SSH loss."""
    if not 2 <= attempts <= 5:
        raise ValueError("idempotent SSH attempts must be in [2, 5]")
    completed: subprocess.CompletedProcess[bytes] | None = None
    for attempt in range(1, attempts + 1):
        completed = _run([*ssh, command], check=False)
        if completed.returncode == 0:
            return completed
        if attempt < attempts:
            time.sleep(retry_seconds)
    raise LifecycleError(
        f"idempotent SSH marker failed after {attempts} attempts; returncode={completed.returncode}"
    )


def _require_worker_image_capabilities(ssh: list[str]) -> dict[str, Any]:
    required = ("bash", "python3", "rsync", "ssh", "sha256sum")
    command = (
        "set -eu; command -v " + " ".join(required) + " >/dev/null; "
        "python3 -c 'import hashlib,json,os,pathlib'"
    )
    completed = _run([*ssh, command], check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LifecycleError(
            "worker image failed the frozen live command gate: " + stderr[-300:]
        )
    return {"status": "GREEN", "required_commands": list(required)}


def _json_output(completed: subprocess.CompletedProcess[bytes]) -> Any:
    text = completed.stdout.decode("utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some Vast CLI mutations print Python-style dictionaries or prose
        # even under --raw. Cleanup callers must never crash while parsing it.
        return text


def _regenerated_instance_credential_shell() -> str:
    """Official Vast fallback when CONTAINER_API_KEY is absent."""
    return (
        'instance_label="${VAST_CONTAINERLABEL:-}"; test -n "$instance_label"; '
        'instance_id="${instance_label#C.}"; case "$instance_id" in *[!0-9]*|"") exit 70;; esac; '
        'test -s /root/.ssh/authorized_keys; '
        'authorized_hash=$(md5sum /root/.ssh/authorized_keys | cut -d" " -f1); '
        'label_hash=$(printf %s "$instance_label" | md5sum | cut -d" " -f1); '
        'instance_api_key="${authorized_hash}${label_hash}"'
    )


def _persist_instance_label_onstart() -> str:
    return 'umask 077; env | grep -E "^VAST_CONTAINERLABEL=" >> /etc/environment; '


def ttl_onstart(*, action: str, seconds: int) -> str:
    if action not in {"stop", "destroy"}:
        raise ValueError(action)
    if seconds <= 0:
        raise ValueError("TTL must be positive")
    method = "PUT" if action == "stop" else "DELETE"
    data = '-H "Content-Type: application/json" --data-binary "{\\"state\\":\\"stopped\\"}"' if action == "stop" else ""
    return (
        _persist_instance_label_onstart()
        + f"nohup bash -c 'sleep {seconds}; {_regenerated_instance_credential_shell()}; until "
        f"curl -fsS --connect-timeout 3 --max-time 10 --retry 2 --retry-all-errors --retry-delay 1 --retry-max-time 15 "
        f"-H \"Authorization: Bearer $instance_api_key\" "
        f"{data} -X {method} \"https://console.vast.ai/api/v0/instances/$instance_id/\"; "
        "do sleep 5; done' "
        ">/tmp/parallax-ttl-guard.log 2>&1 </dev/null &"
    )


def bootstrap_guard_onstart(
    *,
    action: str,
    seconds: int,
    armed_path: str = "/tmp/parallax-lifecycle-armed",
    lease_path: str = "/tmp/parallax-bootstrap-lease",
    lease_timeout_seconds: int = 30,
) -> str:
    """Provider-side cleanup if the local launcher disappears before arming."""
    if action not in {"stop", "destroy"}:
        raise ValueError(action)
    if not 30 <= seconds <= 300:
        raise ValueError("bootstrap orphan grace must be between 30 and 300 seconds")
    for name, path in (("sentinel", armed_path), ("lease", lease_path)):
        if not path.startswith("/tmp/") or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_/" for character in path):
            raise ValueError(f"unsafe bootstrap {name} path")
    if not 15 <= lease_timeout_seconds <= 60:
        raise ValueError("bootstrap lease timeout must be between 15 and 60 seconds")
    method = "PUT" if action == "stop" else "DELETE"
    data = '-H "Content-Type: application/json" --data-binary "{\\"state\\":\\"stopped\\"}"' if action == "stop" else ""
    return (
        f"nohup bash -c 'sleep {seconds}; {_regenerated_instance_credential_shell()}; while true; do "
        f"test -f {armed_path} && exit 0; now=$(date +%s); lease=0; "
        f"test -f {lease_path} && lease=$(stat -c %Y {lease_path}); "
        f"test $((now-lease)) -le {lease_timeout_seconds} && sleep 5 && continue; "
        f"curl -fsS --connect-timeout 3 --max-time 10 --retry 2 --retry-all-errors --retry-delay 1 --retry-max-time 15 "
        f"-H \"Authorization: Bearer $instance_api_key\" "
        f"{data} -X {method} \"https://console.vast.ai/api/v0/instances/$instance_id/\" && exit; "
        "sleep 5; done' "
        ">/tmp/parallax-bootstrap-guard.log 2>&1 </dev/null &"
    )


def prearm_deadline_onstart(
    *, action: str, seconds: int, armed_path: str = "/tmp/parallax-lifecycle-armed"
) -> str:
    """Non-renewable paid-worker deadline; a live launcher cannot extend it."""
    if action not in {"stop", "destroy"}:
        raise ValueError(action)
    if not 30 <= seconds <= 300:
        raise ValueError("pre-arm deadline must be between 30 and 300 seconds")
    if not armed_path.startswith("/tmp/") or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_/" for character in armed_path
    ):
        raise ValueError("unsafe pre-arm sentinel path")
    method = "PUT" if action == "stop" else "DELETE"
    data = '-H "Content-Type: application/json" --data-binary "{\\"state\\":\\"stopped\\"}"' if action == "stop" else ""
    return (
        f"nohup bash -c 'sleep {seconds}; {_regenerated_instance_credential_shell()}; test -f {armed_path} && exit 0; until "
        f"curl -fsS --connect-timeout 3 --max-time 10 --retry 2 --retry-all-errors --retry-delay 1 --retry-max-time 15 "
        f"-H \"Authorization: Bearer $instance_api_key\" {data} -X {method} "
        f"\"https://console.vast.ai/api/v0/instances/$instance_id/\"; do sleep 5; done' "
        ">/tmp/parallax-prearm-deadline.log 2>&1 </dev/null &"
    )


def scoped_deadman_command(
    *, instance_id: int, key_path: str, seconds: int, action: str, armed_path: str | None = None
) -> str:
    """Detached deadline using a 0600 scoped lifecycle key already on disk."""
    if instance_id <= 0 or seconds <= 0 or action not in {"stop", "destroy"}:
        raise ValueError("invalid scoped deadman")
    for path in (key_path, armed_path):
        if path is not None and (not path.startswith("/") or any(char.isspace() for char in path)):
            raise ValueError("unsafe scoped deadman path")
    method = "PUT" if action == "stop" else "DELETE"
    data = '-H "Content-Type: application/json" --data-binary "{\\"state\\":\\"stopped\\"}"' if action == "stop" else ""
    armed = f"test -f {shlex_quote(armed_path)} && exit 0; " if armed_path else ""
    return (
        f"nohup bash -c 'sleep {seconds}; {armed}until "
        f"curl -fsS --connect-timeout 3 --max-time 10 --retry 2 --retry-all-errors "
        f"--retry-delay 1 --retry-max-time 15 -H \"Authorization: Bearer $(cat {shlex_quote(key_path)})\" "
        f"{data} -X {method} \"https://console.vast.ai/api/v0/instances/{instance_id}/\"; "
        "do sleep 5; done' >/tmp/parallax-scoped-deadman.log 2>&1 </dev/null &"
    )


def scoped_result_stop_command(
    *, instance_id: int, key_path: str, result_path: str, grace_seconds: int = 3
) -> str:
    """STOP billing as soon as an atomic durable result becomes visible."""
    if instance_id <= 0 or not 1 <= grace_seconds <= 5:
        raise ValueError("invalid result-triggered STOP")
    for path in (key_path, result_path):
        if not path.startswith("/") or any(char.isspace() for char in path):
            raise ValueError("unsafe result-triggered STOP path")
    return (
        f"nohup bash -c 'until test -s {result_path}; do sleep 1; done; sleep {grace_seconds}; until "
        f"curl -fsS --connect-timeout 3 --max-time 10 --retry 2 --retry-all-errors "
        f"--retry-delay 1 --retry-max-time 15 -H \"Authorization: Bearer $(cat {key_path})\" "
        f"-H \"Content-Type: application/json\" --data-binary \"{{\\\"state\\\":\\\"stopped\\\"}}\" "
        f"-X PUT \"https://console.vast.ai/api/v0/instances/{instance_id}/\"; do sleep 2; done' "
        ">/tmp/parallax-result-stop.log 2>&1 </dev/null &"
    )


class BootstrapLease:
    """Renew a remote pre-arm lease; staleness is handled inside the instance."""

    def __init__(self, ssh: list[str], *, interval_seconds: float = 5.0, maximum_consecutive_failures: int = 3):
        self.ssh = ssh
        self.interval_seconds = interval_seconds
        self.maximum_consecutive_failures = maximum_consecutive_failures
        if maximum_consecutive_failures < 2 or interval_seconds * maximum_consecutive_failures >= 30:
            raise LifecycleError("bootstrap lease failure threshold must remain below remote staleness")
        self.stop_event = threading.Event()
        self.failed = threading.Event()
        initial = _run([*self.ssh, "touch /tmp/parallax-bootstrap-lease"], check=False)
        if initial.returncode != 0:
            raise LifecycleError("could not establish remote bootstrap lease")
        self.thread = threading.Thread(target=self._worker, name="parallax-bootstrap-lease", daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        consecutive_failures = 0
        while not self.stop_event.wait(self.interval_seconds):
            renewed = _run([*self.ssh, "touch /tmp/parallax-bootstrap-lease"], check=False)
            if renewed.returncode != 0:
                consecutive_failures += 1
                if consecutive_failures >= self.maximum_consecutive_failures:
                    self.failed.set()
                    return
            else:
                consecutive_failures = 0

    def assert_healthy(self) -> None:
        if self.failed.is_set():
            raise LifecycleError("remote bootstrap lease renewal failed")

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        # End the local multiplex master rather than leaving a 60-second
        # ControlPersist process and socket after handoff or cleanup.
        if self.ssh and self.ssh[0] == "ssh" and any(
            str(value).startswith("ControlPath=") for value in self.ssh
        ):
            _run([*self.ssh[:-1], "-O", "exit", self.ssh[-1]], check=False)


class LocalBootstrapReaper:
    """Detached exact-label cleanup until the remote guardian takes ownership."""

    def __init__(
        self,
        state_root: Path,
        *,
        repo_root: Path,
        run_id: str,
        lifecycle_key_path: Path,
        maximum_seconds: int,
        stale_seconds: float = 10.0,
        interval_seconds: float = 2.0,
    ):
        from .local_bootstrap_reaper import ReaperPolicy

        self.root = state_root
        self.lease_path = state_root / "local-reaper.lease"
        self.handoff_path = state_root / "local-reaper.handoff.json"
        self.ready_path = state_root / "local-reaper.ready.json"
        self.receipt_path = state_root / "local-reaper.receipt.json"
        self.contract_path = state_root / "local-reaper.contract.json"
        self.log_path = state_root / "local-reaper.log"
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        atomic_json(self.lease_path, {"run_id": run_id, "status": "LAUNCHER_ALIVE"})
        contract = {
            "schema_version": 1,
            "status": "FROZEN_LOCAL_REAPER",
            "run_id": run_id,
            "gateway_label": f"parallax-guard-{run_id}",
            "api_key_file": str(lifecycle_key_path.resolve()),
            "lease_path": str(self.lease_path.resolve()),
            "handoff_path": str(self.handoff_path.resolve()),
            "ready_path": str(self.ready_path.resolve()),
            "receipt_path": str(self.receipt_path.resolve()),
            "stale_seconds": stale_seconds,
            "maximum_seconds": maximum_seconds,
            "poll_seconds": 0.5,
        }
        ReaperPolicy.from_contract(contract)
        atomic_json(self.contract_path, contract)
        self.log_handle = self.log_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "infra.vast.parallax.local_bootstrap_reaper",
                "--contract",
                str(self.contract_path),
            ],
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self.ready_path.is_file():
            if self.process.poll() is not None:
                break
            time.sleep(0.1)
        if not self.ready_path.is_file():
            self.log_handle.close()
            raise LifecycleError("detached local bootstrap reaper did not arm")
        self.thread = threading.Thread(
            target=self._renew,
            name=f"parallax-local-reaper-{run_id}",
            daemon=True,
        )
        self.thread.start()

    def _renew(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                os.utime(self.lease_path, None)
            except OSError:
                return

    def handoff(self, *, run_id: str, target: str = "remote_guardian") -> dict[str, Any]:
        if target not in {"remote_guardian", "remote_scoped_deadman"}:
            raise ValueError(target)
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 1)
        atomic_json(self.handoff_path, {
            "run_id": run_id,
            "status": "REMOTE_HANDOFF_LIVE",
            "target": target,
        })
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise LifecycleError("local bootstrap reaper did not accept guardian handoff") from error
        self.log_handle.close()
        if not self.receipt_path.is_file():
            raise LifecycleError("local bootstrap reaper produced no handoff receipt")
        packet = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        expected = (
            "HANDED_OFF_TO_REMOTE_GUARDIAN"
            if target == "remote_guardian"
            else "HANDED_OFF_TO_REMOTE_SCOPED_DEADMAN"
        )
        if packet.get("status") != expected:
            raise LifecycleError("local bootstrap reaper did not reach a safe handoff state")
        return packet

    def release_to_reap(self) -> None:
        """Stop renewal without handoff; the detached process now owns cleanup."""
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 1)
        self.log_handle.close()


def _vast(*arguments: str, check: bool = True) -> Any:
    completed = _run(["vastai", *arguments, "--raw"], check=check)
    if completed.returncode != 0:
        return None
    return _json_output(completed)


def _show_instance(instance_id: int) -> dict[str, Any] | None:
    payload = _vast("show", "instance", str(instance_id), check=False)
    if payload is None:
        return None
    if isinstance(payload, list):
        return payload[0] if payload else None
    if isinstance(payload, dict) and isinstance(payload.get("instances"), dict):
        return payload["instances"]
    return payload if isinstance(payload, dict) else None


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise LifecycleError("Vast market query did not return a row list")


def select_live_offers(*, spec: dict[str, Any], volume_id: int) -> dict[str, Any]:
    """Select live offers inside the launcher; callers cannot inject offer ids."""
    volumes = _rows(_vast("show", "volumes"))
    matches = [row for row in volumes if int(row.get("id", -1)) == volume_id]
    if len(matches) != 1 or int(matches[0].get("machine_id", 0)) <= 0:
        raise LifecycleError(f"persistent volume {volume_id} was not found uniquely")
    machine_id = int(matches[0]["machine_id"])
    gateway_query = (
        f"machine_id={machine_id} num_gpus=1 rentable=true rented=false "
        f"disk_space>={int(spec['gateway_disk_gb'])}"
    )
    gateways = _rows(
        _vast("search", "offers", gateway_query, "--limit", "100", "--storage", str(spec["gateway_disk_gb"]), "--order", "dph_total")
    )
    gateways.sort(key=lambda row: (float(row.get("dph_total", float("inf"))), -float(row.get("reliability", 0))))
    if not gateways:
        raise LifecycleError("no rentable gateway exists on the persistent-volume machine")
    gateway = gateways[0]
    lifecycle_h100 = spec.get("lifecycle_worker_kind") == "h100"
    if spec.get("h100_research_authorized") is True or lifecycle_h100:
        minimum_cuda = float(spec.get("nonresearch_min_cuda", 0.0))
        minimum_inet_down = float(spec.get("nonresearch_min_inet_down_mbps", 0.0))
        cuda_clause = f" cuda_max_good>={minimum_cuda:g}" if minimum_cuda > 0 else ""
        inet_clause = f" inet_down>={minimum_inet_down:g}" if minimum_inet_down > 0 else ""
        worker_query = (
            "cpu_arch=amd64 gpu_name in [H100_SXM,H100_NVL,H100_PCIE,H100] num_gpus=1 gpu_ram>=80 "
            f"reliability>=0.98 disk_space>={int(spec['h100_disk_gb'])} duration>=4 "
            f"rentable=true rented=false static_ip=true direct_port_count>=2{cuda_clause}{inet_clause}"
        )
        worker_rule = (
            "literal_cheapest_suitable_h100_all_regions_at_query_time"
            if spec.get("h100_research_authorized") is True
            else "literal_cheapest_suitable_h100_lifecycle_only_all_regions_at_query_time"
        )
    else:
        minimum_reliability = float(spec.get("nonresearch_min_reliability", 0.98))
        minimum_cuda = float(spec.get("nonresearch_min_cuda", 0.0))
        minimum_inet_down = float(spec.get("nonresearch_min_inet_down_mbps", 0.0))
        cuda_clause = f" cuda_max_good>={minimum_cuda:g}" if minimum_cuda > 0 else ""
        inet_clause = f" inet_down>={minimum_inet_down:g}" if minimum_inet_down > 0 else ""
        worker_query = (
            f"cpu_arch=amd64 num_gpus=1 reliability>={minimum_reliability:g} disk_space>={int(spec['h100_disk_gb'])} "
            f"rentable=true rented=false static_ip=true direct_port_count>=2{cuda_clause}{inet_clause}"
        )
        worker_rule = (
            "literal_cheapest_suitable_cuda_compatible_nonresearch_worker_all_regions_at_query_time"
            if minimum_cuda > 0
            else "literal_cheapest_suitable_nonresearch_worker_all_regions_at_query_time"
        )
    workers = _rows(
        _vast("search", "offers", worker_query, "--limit", "100", "--storage", str(spec["h100_disk_gb"]), "--order", "dph_total")
    )
    excluded_offer_ids = {int(value) for value in spec.get("excluded_worker_offer_ids", [])}
    excluded_machine_ids = {int(value) for value in spec.get("excluded_worker_machine_ids", [])}
    workers = [row for row in workers if int(row.get("id", -1)) != int(gateway["id"])]
    workers = [
        row for row in workers
        if int(row.get("id", -1)) not in excluded_offer_ids
        and int(row.get("machine_id", -1)) not in excluded_machine_ids
    ]
    if spec.get("h100_research_authorized") is not True and not lifecycle_h100:
        workers = [row for row in workers if "H100" not in str(row.get("gpu_name", "")).upper()]
        minimum_cuda = float(spec.get("nonresearch_min_cuda", 0.0))
        workers = [
            row for row in workers
            if float(row.get("cuda_max_good", 0.0)) >= minimum_cuda
        ]
    workers.sort(key=lambda row: (float(row.get("dph_total", float("inf"))), -float(row.get("reliability", 0))))
    if not workers:
        raise LifecycleError("no distinct suitable worker offer exists")
    worker = workers[0]
    public_fields = (
        "id", "machine_id", "gpu_name", "num_gpus", "gpu_ram", "reliability",
        "dph_total", "geolocation", "verification", "static_ip", "direct_port_count",
        "cuda_max_good", "driver_version",
        "inet_down", "inet_up",
    )
    sanitize = lambda row: {key: row[key] for key in public_fields if key in row}
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "vast.ai",
        "mutation": "none_read_only",
        "volume_id": volume_id,
        "all_regions": True,
        "gateway_query": gateway_query,
        "worker_query": worker_query,
        "gateway_selection_rule": "literal_cheapest_rentable_offer_on_volume_machine_at_query_time",
        "worker_selection_rule": worker_rule,
        "gateway_candidate_count": len(gateways),
        "worker_candidate_count": len(workers),
        "excluded_worker_offer_ids": sorted(excluded_offer_ids),
        "excluded_worker_machine_ids": sorted(excluded_machine_ids),
        "nonresearch_min_cuda": float(spec.get("nonresearch_min_cuda", 0.0)),
        "nonresearch_min_inet_down_mbps": float(
            spec.get("nonresearch_min_inet_down_mbps", 0.0)
        ),
        "selected_gateway": sanitize(gateway),
        "selected_worker": sanitize(worker),
    }


def select_live_gateway_offer(*, spec: dict[str, Any], volume_id: int) -> dict[str, Any]:
    """Select only the cheapest rentable attachment on the volume machine."""
    volumes = _rows(_vast("show", "volumes"))
    matches = [row for row in volumes if int(row.get("id", -1)) == volume_id]
    if len(matches) != 1 or int(matches[0].get("machine_id", 0)) <= 0:
        raise LifecycleError(f"persistent volume {volume_id} was not found uniquely")
    machine_id = int(matches[0]["machine_id"])
    query = (
        f"machine_id={machine_id} num_gpus=1 rentable=true rented=false "
        f"disk_space>={int(spec['gateway_disk_gb'])}"
    )
    rows = _rows(
        _vast(
            "search", "offers", query, "--limit", "100", "--storage",
            str(spec["gateway_disk_gb"]), "--order", "dph_total",
        )
    )
    rows.sort(
        key=lambda row: (
            float(row.get("dph_total", float("inf"))),
            -float(row.get("reliability", 0)),
        )
    )
    if not rows:
        raise LifecycleError("no rentable gateway exists on the persistent-volume machine")
    public_fields = (
        "id", "machine_id", "gpu_name", "num_gpus", "gpu_ram", "reliability",
        "dph_total", "geolocation", "verification", "static_ip", "direct_port_count",
        "cuda_max_good", "driver_version",
    )
    selected = {key: rows[0][key] for key in public_fields if key in rows[0]}
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "vast.ai",
        "mutation": "none_read_only",
        "volume_id": volume_id,
        "gateway_query": query,
        "gateway_selection_rule": "literal_cheapest_rentable_offer_on_volume_machine_at_query_time",
        "gateway_candidate_count": len(rows),
        "selected_gateway": selected,
        "worker_query": None,
        "worker_created": False,
    }


def wait_ready(instance_id: int, *, timeout_seconds: int, poll_seconds: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    terminal = {"exited", "unknown", "offline"}
    observed = False
    while time.monotonic() < deadline:
        row = _show_instance(instance_id)
        if row is not None:
            observed = True
        elif observed:
            raise LifecycleError(f"instance {instance_id} disappeared during bootstrap")
        status = (row or {}).get("actual_status")
        if status == "running":
            return row or {}
        if status in terminal:
            raise LifecycleError(f"instance {instance_id} entered terminal bootstrap state {status}")
        time.sleep(poll_seconds)
    raise LifecycleError(f"instance {instance_id} did not become ready before bootstrap timeout")


def wait_ssh(
    instance_id: int,
    key: Path,
    *,
    timeout_seconds: int,
    poll_seconds: float = 2.0,
    health_check=lambda: None,
) -> dict[str, Any]:
    """Require a real SSH command; Vast's `running` state alone is not ready."""
    deadline = time.monotonic() + timeout_seconds
    terminal = {"exited", "unknown", "offline"}
    observed = False
    attempts = 0
    last_error = "instance never entered running state"
    while time.monotonic() < deadline:
        health_check()
        row = _show_instance(instance_id)
        if row is not None:
            observed = True
        elif observed:
            raise LifecycleError(f"instance {instance_id} disappeared during SSH bootstrap")
        status = (row or {}).get("actual_status")
        if status in terminal:
            raise LifecycleError(f"instance {instance_id} entered terminal bootstrap state {status}")
        if status == "running":
            probe = _run([*_ssh_base(row, key), "true"], check=False)
            attempts += 1
            if probe.returncode == 0:
                return row
            last_error = probe.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:] or ["empty SSH stderr"]
            last_error = str(last_error[0])[:300]
        time.sleep(poll_seconds)
    raise LifecycleError(
        f"instance {instance_id} never accepted SSH before bootstrap timeout; "
        f"attempts={attempts}; last_error={last_error}"
    )


def _ssh_base(
    row: dict[str, Any], key: Path, *, control_path: Path | None = None
) -> list[str]:
    host, port = _ssh_endpoint(row)
    command = [
        "ssh", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
    ]
    if control_path is not None:
        command.extend([
            "-o", "ControlMaster=auto", "-o", "ControlPersist=60",
            "-o", f"ControlPath={control_path}",
        ])
    return [*command, "-p", str(port), f"root@{host}"]


def _scp_base(
    row: dict[str, Any], key: Path, *, control_path: Path | None = None
) -> list[str]:
    host, port = _ssh_endpoint(row)
    command = [
        "scp", "-q", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
    ]
    if control_path is not None:
        command.extend([
            "-o", "ControlMaster=auto", "-o", "ControlPersist=60",
            "-o", f"ControlPath={control_path}",
        ])
    return [*command, "-P", str(port)]


def _worker_create_arguments(spec: dict[str, Any], *, run_id: str) -> list[str]:
    template_hash = str(spec.get("worker_template_hash", ""))
    common = [
        "--disk", str(spec["h100_disk_gb"]), "--cancel-unavail",
        "--label", f"parallax-worker-{run_id}",
    ]
    if template_hash:
        arguments = ["--template_hash", template_hash]
        if spec.get("worker_template_image_override") is True:
            arguments.extend(["--image", str(spec["worker_image"])])
        return [*arguments, *common]
    return [
        "--image", str(spec.get("worker_image", spec["image"])), *common,
        "--ssh", "--direct", "--onstart-cmd", _persist_instance_label_onstart(),
    ]


def _ssh_endpoint(row: dict[str, Any]) -> tuple[str, int]:
    """Prefer Vast's direct published SSH mapping over its fallible proxy endpoint."""
    public_ip = str(row.get("public_ipaddr") or "")
    mappings = row.get("ports", {}).get("22/tcp", [])
    direct_ports = [
        int(mapping["HostPort"])
        for mapping in mappings
        if isinstance(mapping, dict)
        and str(mapping.get("HostIp", "")) in {"0.0.0.0", public_ip}
        and str(mapping.get("HostPort", "")).isdigit()
    ]
    if public_ip and direct_ports:
        return public_ip, direct_ports[0]
    host = str(row.get("ssh_host") or public_ip)
    port = int(row.get("ssh_port") or row.get("machine_dir_ssh_port"))
    if not host or port <= 0:
        raise LifecycleError("instance has no usable SSH endpoint")
    return host, port


def _short_control_path(*, role: str, instance_id: int) -> Path:
    """Return a host-portable Unix socket path independent of tempdir length."""
    if role not in {
        "bootstrap", "inspection", "cache-audit", "task-cache", "object-mirror",
        "cache-mirror",
    } or instance_id <= 0:
        raise ValueError("invalid SSH control socket identity")
    prefix = {
        "bootstrap": "pb",
        "inspection": "pi",
        "cache-audit": "pa",
        "task-cache": "pt",
        "object-mirror": "po",
        "cache-mirror": "pm",
    }[role]
    path = Path(f"/tmp/{prefix}-{instance_id}-{os.getpid()}.sock")
    if len(os.fsencode(path)) > 80:
        raise LifecycleError("SSH control socket path exceeds frozen portability bound")
    return path


def _semantic_lifecycle_ready_command(
    *, state_root: str, run_id: str, kind: str, maximum_age_seconds: int
) -> str:
    """Validate lifecycle readiness by content and freshness, not existence."""
    if kind == "guardian":
        rules = {
            "guardian-heartbeat.json": ["WATCHING"],
            "launch-registry.json": ["GUARDIAN_ARMED_WORKER_ABSENT"],
        }
    elif kind == "supervisor":
        rules = {"heartbeat.json": ["ARMED_WAITING_RELEASE"]}
    else:
        raise ValueError(kind)
    script = (
        "import json,pathlib,sys,time;"
        "root=pathlib.Path(sys.argv[1]);run=sys.argv[2];age=float(sys.argv[3]);"
        "rules=json.loads(sys.argv[4]);now=time.time();"
        "rows=[(name,json.loads((root/name).read_text())) for name in rules];"
        "assert all(row.get('run_id')==run for name,row in rows);"
        "assert all(row.get('status') in rules[name] for name,row in rows);"
        "assert all(0<=now-float(row.get('wall_time_unix',0))<=age for name,row in rows)"
    )
    arguments = [
        "python3", "-c", script, state_root, run_id, str(maximum_age_seconds),
        json.dumps(rules, separators=(",", ":")),
    ]
    return " ".join(shlex_quote(value) for value in arguments)


def _validated_registered_ssh_key(spec: dict[str, Any], *, role: str) -> Path:
    if role not in {"bootstrap", "relay"}:
        raise ValueError(role)
    private_key = Path(str(spec.get(f"{role}_ssh_private_key", ""))).expanduser()
    public_key = private_key.with_suffix(private_key.suffix + ".pub")
    if not private_key.is_file() or not public_key.is_file():
        raise LifecycleError(f"registered {role} SSH keypair is missing locally")
    if private_key.stat().st_mode & 0o077:
        raise LifecycleError(f"{role} SSH private key permissions are not 0600")
    public_value = public_key.read_text(encoding="utf-8").strip()
    keys = _vast("show", "ssh-keys")
    expected_id = int(spec.get(f"{role}_ssh_public_key_id", 0))
    matches = [
        row for row in keys
        if int(row.get("id", -1)) == expected_id and str(row.get("public_key", "")).strip() == public_value
    ]
    if len(matches) != 1:
        raise LifecycleError(f"local {role} SSH key is not the exact registered Vast key")
    return private_key


def _validated_bootstrap_ssh_key(spec: dict[str, Any]) -> Path:
    return _validated_registered_ssh_key(spec, role="bootstrap")


def _validated_scoped_lifecycle_api_key(spec: dict[str, Any]) -> tuple[Path, int]:
    """Validate the registered key identity without ever printing its value."""
    path = Path(str(spec.get("scoped_lifecycle_api_key_file", ""))).expanduser()
    key_id = int(spec.get("scoped_lifecycle_api_key_id", 0))
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise LifecycleError("scoped lifecycle API key is missing or not mode 0600")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 16 or any(character.isspace() for character in value):
        raise LifecycleError("scoped lifecycle API key file is malformed")
    rows = _vast("show", "api-keys")
    if isinstance(rows, dict):
        rows = rows.get("apikeys", [])
    flattened = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            flattened.append(row)
        elif isinstance(row, list):
            flattened.extend(value for value in row if isinstance(value, dict))
    matches = [
        row for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict)
        and int(row.get("id", -1)) == key_id
        and row.get("name") == spec.get("scoped_lifecycle_api_key_name")
    ]
    if not matches:
        matches = [
            row for row in flattened
            if int(row.get("id", -1)) == key_id
            and row.get("name") == spec.get("scoped_lifecycle_api_key_name")
        ]
    if len(matches) != 1:
        raise LifecycleError("scoped lifecycle API key id is not registered")
    return path, key_id


def _validate_scoped_key_binding(root: Path, spec: dict[str, Any]) -> tuple[Path, int]:
    if spec.get("scoped_lifecycle_api_permissions") != SCOPED_LIFECYCLE_PERMISSIONS:
        raise LifecycleError("lifecycle scope must bind endpoint-only instance permissions")
    if not spec.get("scoped_lifecycle_api_key_file") or int(
        spec.get("scoped_lifecycle_api_key_id", 0)
    ) <= 0:
        raise LifecycleError("lifecycle scope must bind one scoped lifecycle API key")
    if spec.get("scoped_lifecycle_api_key_name") != "parallax-lifecycle-v2":
        raise LifecycleError("lifecycle scope must bind the corrected v2 key name")
    key_receipt = spec.get("scoped_lifecycle_key_receipt", {})
    key_receipt_path = root / str(key_receipt.get("path", ""))
    if not key_receipt_path.is_file() or sha256_file(key_receipt_path) != key_receipt.get("sha256"):
        raise LifecycleError("lifecycle scope must bind the redacted key receipt")
    key_receipt_payload = json.loads(key_receipt_path.read_text(encoding="utf-8"))
    if (
        key_receipt_payload.get("api_key_id") != int(spec["scoped_lifecycle_api_key_id"])
        or key_receipt_payload.get("name") != spec["scoped_lifecycle_api_key_name"]
        or key_receipt_payload.get("permissions") != SCOPED_LIFECYCLE_PERMISSIONS
        or key_receipt_payload.get("secret_logged") is not False
    ):
        raise LifecycleError("redacted lifecycle key receipt does not match scope")
    authority_binding = spec.get("scoped_lifecycle_key_authority", {})
    authority_path = root / str(authority_binding.get("path", ""))
    if not authority_path.is_file() or sha256_file(authority_path) != authority_binding.get("sha256"):
        raise LifecycleError("lifecycle scope must bind the key authority proof")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if not (
        authority.get("status") == "GREEN"
        and authority.get("instance_manage_authorized") is True
        and authority.get("volume_endpoint_denied") is True
        and authority.get("secret_logged") is False
    ):
        raise LifecycleError("lifecycle key authority proof is not GREEN")
    return _validated_scoped_lifecycle_api_key(spec)


def _probe_scoped_key_authority(key_path: Path) -> dict[str, Any]:
    """Prove instance read is allowed while persistent-volume read is denied."""
    key = key_path.read_text(encoding="utf-8").strip()
    instances_request = urllib.request.Request(
        f"{API_BASE}/instances/", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(instances_request, timeout=10) as response:
        instances_payload = json.loads(response.read().decode("utf-8") or "{}")
    rows = instances_payload.get("instances", instances_payload)
    if not isinstance(rows, list):
        raise LifecycleError("scoped key instance listing is malformed")
    volumes_request = urllib.request.Request(
        f"{API_BASE}/volumes/", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(volumes_request, timeout=10):
            pass
    except urllib.error.HTTPError as error:
        if error.code not in {401, 403}:
            raise LifecycleError(f"volume denial returned unexpected HTTP {error.code}") from error
        volume_status = error.code
    else:
        raise LifecycleError("scoped lifecycle key unexpectedly has volume access")

    def instance_mutation_status(method: str, data: bytes | None) -> int:
        request = urllib.request.Request(
            f"{API_BASE}/instances/999999999/",
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
                return int(response.status)
        except urllib.error.HTTPError as error:
            return int(error.code)

    stop_status = instance_mutation_status(
        "PUT", json.dumps({"state": "stopped"}).encode("utf-8")
    )
    destroy_status = instance_mutation_status("DELETE", None)
    if stop_status in {401, 403} or destroy_status in {401, 403}:
        raise LifecycleError("scoped key cannot manage the instance endpoint")
    return {
        "status": "GREEN",
        "instance_list_allowed": True,
        "active_instance_count_before_probe": len(rows),
        "volume_endpoint_denied": True,
        "volume_endpoint_http_status": volume_status,
        "nonexistent_instance_stop_http_status": stop_status,
        "nonexistent_instance_destroy_http_status": destroy_status,
        "instance_manage_authorized": True,
        "secret_logged": False,
    }


def _regenerate_remote_instance_api_key(ssh: list[str], *, instance_id: int) -> str:
    """Regenerate and retrieve the documented restricted key without logging it."""
    if instance_id <= 0:
        raise ValueError("instance_id must be positive")
    expected_label = f"C.{instance_id}"
    command = (
        "set -eu; test -f /etc/environment; . /etc/environment; "
        f'test "${{VAST_CONTAINERLABEL:-}}" = "{expected_label}"; '
        + _regenerated_instance_credential_shell()
        + '; printf %s "$instance_api_key"'
    )
    completed = _run([*ssh, command], check=False)
    value = completed.stdout.decode("utf-8", errors="strict").strip()
    if completed.returncode != 0 or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LifecycleError("could not regenerate the documented per-instance API key")
    return value


def _create_instance(offer_id: int, arguments: list[str]) -> dict[str, Any]:
    # Private-registry credentials may be present in `arguments`. Never use
    # check=True here: CalledProcessError stringifies the command and would
    # disclose the expiring credential. Provider stderr is intentionally not
    # included in the fail-closed error either.
    try:
        completed = _run(
            ["vastai", "create", "instance", str(offer_id), *arguments, "--raw"],
            check=False,
            timeout_seconds=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise LifecycleError(f"Vast instance creation failed for offer {offer_id}") from None
    if completed.returncode != 0:
        raise LifecycleError(f"Vast instance creation failed for offer {offer_id}")
    payload = _json_output(completed)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise LifecycleError(f"Vast instance creation failed for offer {offer_id}")
    if not isinstance(payload.get("new_contract"), int) or not payload.get("instance_api_key"):
        raise LifecycleError("Vast create response lacks instance id or per-instance key")
    return payload


def _validate_worker_registry_auth(spec: dict[str, Any]) -> dict[str, Any] | None:
    auth = spec.get("worker_registry_auth")
    if auth is None:
        return None
    if not isinstance(auth, dict):
        raise LifecycleError("worker registry auth contract must be an object")
    expected = {
        "type": "aws_ecr_ephemeral",
        "profile": "fuelos",
        "region": "us-east-1",
        "registry": "769063704696.dkr.ecr.us-east-1.amazonaws.com",
        "username": "AWS",
        "credential_source": "aws_ecr_get_login_password",
        "persist_secret": False,
        "maximum_token_age_seconds": 43200,
    }
    if auth != expected:
        raise LifecycleError("worker registry auth contract is not the exact fuelos ECR contract")
    reference = str(spec.get("worker_image", ""))
    if not reference.startswith(expected["registry"] + "/"):
        raise LifecycleError("worker registry auth does not match the worker image registry")
    return auth


def _ephemeral_worker_registry_login(spec: dict[str, Any]) -> str | None:
    """Resolve an ECR pull credential in memory immediately before worker create."""
    auth = _validate_worker_registry_auth(spec)
    if auth is None:
        return None
    completed = _run(
        [
            "aws", "ecr", "get-login-password",
            "--profile", str(auth["profile"]),
            "--region", str(auth["region"]),
        ],
        check=False,
        timeout_seconds=30,
    )
    token = completed.stdout.decode("utf-8", errors="strict").strip()
    if completed.returncode != 0 or not token or any(character.isspace() for character in token):
        raise LifecycleError("could not resolve ephemeral worker registry credential")
    return " ".join(
        shlex_quote(value)
        for value in ("-u", str(auth["username"]), "-p", token, str(auth["registry"]))
    )


def _normalized_lifecycle_archive(root: Path, output: Path) -> None:
    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative in LIFECYCLE_SOURCE_FILES:
            path = root / relative
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if relative == "infra/vast/parallax/__init__.py":
                # The repository package initializer imports the full control
                # plane (DAG, YAML launch packet, etc.). The remote lifecycle
                # archive is deliberately minimal and must not import them.
                content = b'"""Minimal detached lifecycle runtime."""\n'
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            else:
                with path.open("rb") as handle:
                    archive.addfile(info, handle)


def _stage_lifecycle_sources_without_local_archive(
    *, root: Path, ssh: list[str], scp: list[str], host: str, remote_root: str
) -> None:
    """Stage the tiny controller source set individually; the laptop never creates an archive."""
    if not remote_root.startswith("/") or any(character.isspace() for character in remote_root):
        raise LifecycleError("unsafe remote lifecycle source root")
    directories = sorted(
        {str(PurePosixPath(remote_root, relative).parent) for relative in LIFECYCLE_SOURCE_FILES}
    )
    _run([*ssh, "mkdir -p " + " ".join(shlex_quote(path) for path in directories)])
    for relative in LIFECYCLE_SOURCE_FILES:
        remote = str(PurePosixPath(remote_root, relative))
        if relative == "infra/vast/parallax/__init__.py":
            _run(
                [*ssh, f"umask 022; cat > {shlex_quote(remote)}"],
                input_bytes=b'"""Minimal detached lifecycle runtime."""\n',
            )
        else:
            _run([*scp, str(root / relative), f"root@{host}:{remote}"])


def _render(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _render(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_render(child, replacements) for child in value]
    if isinstance(value, str):
        rendered = value
        for token, replacement in replacements.items():
            rendered = rendered.replace("{" + token + "}", replacement)
        unresolved = re.search(r"\{[A-Z][A-Z0-9_]*\}", rendered)
        if unresolved is not None:
            raise LifecycleError(f"unresolved lifecycle runtime placeholder: {rendered}")
        return rendered
    return value


def _build_runtime_contract(
    *,
    spec: dict[str, Any],
    job_spec: dict[str, Any],
    run_id: str,
    worker_instance_id: int,
    worker_host: str,
    worker_port: int,
    worker_ssh_key: str,
    worker_api_key: str,
    gateway_instance_id: int,
    gateway_api_key: str,
    volume_run_root: str,
    frozen_spec_sha256: str,
    job_spec_sha256: str,
) -> dict[str, Any]:
    """Build and validate the exact detached contract used by paid launch."""
    policy = spec["policy"]
    rendered_job = _render(
        job_spec["runtime"],
        {
            "RUN_ID": run_id,
            "H100_HOST": worker_host,
            "H100_PORT": str(worker_port),
            "H100_SSH_KEY": worker_ssh_key,
            "VOLUME_RUN_ROOT": volume_run_root,
        },
    )
    contract = {
        "schema_version": 1,
        "status": "FROZEN_RUNTIME",
        "run_id": run_id,
        "policy": {
            "maximum_runtime_seconds": int(policy["maximum_runtime_seconds"]),
            "mirror_interval_seconds": float(policy["mirror_interval_seconds"]),
            "heartbeat_timeout_seconds": float(policy["heartbeat_timeout_seconds"]),
            "artifact_copy_timeout_seconds": float(policy["artifact_copy_timeout_seconds"]),
            "maximum_finalization_seconds": float(policy["maximum_finalization_seconds"]),
            "final_sync_attempts": int(policy["final_sync_attempts"]),
            "final_sync_settle_seconds": float(policy["final_sync_settle_seconds"]),
            "required_sentinels": rendered_job["required_sentinels"],
            "execution_release_timeout_seconds": float(
                policy.get("supervisor_arm_timeout_seconds", 30)
            ),
        },
        "teardown": {
            "durability_before_destroy": True,
            "stop_on_unproven_durability": True,
            "h100_api_key_scope": "scoped_lifecycle_only",
        },
        "h100": {
            "instance_id": worker_instance_id,
            "ssh_host": worker_host,
            "ssh_port": worker_port,
            "ssh_private_key_file": worker_ssh_key,
            "instance_api_key_file": worker_api_key,
        },
        "gateway": {
            "instance_id": gateway_instance_id,
            "volume_run_root": volume_run_root,
            "instance_api_key_file": gateway_api_key,
            "self_destroy": True,
        },
        "job": rendered_job["job"],
        "artifacts": {"roots": rendered_job["artifact_roots"]},
        "expected_artifacts": job_spec.get("expected_artifacts"),
        "source": {
            "frozen_spec_sha256": frozen_spec_sha256,
            "job_spec_sha256": job_spec_sha256,
        },
    }
    validate_runtime_contract(contract)
    return contract


def _write_remote_secret(ssh: list[str], path: str, value: str) -> None:
    command = f"umask 077; mkdir -p {shlex_quote(str(Path(path).parent))}; cat > {shlex_quote(path)}; chmod 600 {shlex_quote(path)}"
    _run([*ssh, command], input_bytes=(value + "\n").encode("utf-8"))


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def validate_frozen_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise LifecycleError("lifecycle launch spec is not frozen")
    if spec.get("provider") != "vast.ai" or spec.get("direct_vast_create_forbidden") is not True:
        raise LifecycleError("lifecycle spec does not enforce guarded Vast launch")
    orphan_grace = int(spec.get("policy", {}).get("bootstrap_orphan_grace_seconds", 0))
    if not 30 <= orphan_grace <= 120:
        raise LifecycleError("bootstrap orphan guard must stop paid compute within 120 seconds")
    copy_timeout = float(spec.get("policy", {}).get("artifact_copy_timeout_seconds", 0))
    finalization_timeout = float(spec.get("policy", {}).get("maximum_finalization_seconds", 0))
    if not 5 <= copy_timeout <= 300:
        raise LifecycleError("artifact operations must have a bounded timeout")
    if not 5 <= finalization_timeout <= 300:
        raise LifecycleError("final result salvage must have a bounded timeout")
    if spec.get("paid_launch_authorized") is True:
        if spec.get("scope") == "h100_official_template_control_only":
            _validate_official_template_contract(root, spec)
        elif spec.get("scope") in {
            "s3_worker_download_probe_only", "s3_worker_cache_hydration_only",
            "pg1_model_forward_smoke_only"
        } and spec.get(
            "worker_template_image_override"
        ) is True:
            _validate_official_template_contract(root, spec)
            _validate_worker_image_contract(root, spec)
        elif spec.get("scope") in {
            "managed_template_runtime_override_probe",
            "h100_managed_template_runtime_override_probe",
        }:
            _validate_official_template_contract(root, spec)
            _validate_worker_image_contract(root, spec)
        else:
            _validate_worker_image_contract(root, spec)
        if spec.get("scope") == "runtime_image_compatibility_probe":
            if spec.get("lifecycle_worker_kind") != "non_h100":
                raise LifecycleError("runtime image probe must use a non-H100 worker")
            minimum_cuda = float(spec.get("nonresearch_min_cuda", 0.0))
            if not 13.0 <= minimum_cuda <= 14.0:
                raise LifecycleError("CUDA 13 runtime probe must bind a compatible host-driver floor")
        if spec.get("scope") == "pg1_input_staging_rehearsal_only":
            if spec.get("lifecycle_worker_kind") != "non_h100" or spec.get("h100_research_authorized") is True:
                raise LifecycleError("input staging rehearsal must use a non-H100 worker")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            if not market_path.is_file() or sha256_file(market_path) != market_binding.get("sha256"):
                raise LifecycleError("input staging lacks exact read-only market preflight")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            selected_worker = market.get("selected_worker", {})
            selected_gateway = market.get("selected_gateway", {})
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or int(market.get("worker_candidate_count", 0)) < 1
                or "H100" in str(selected_worker.get("gpu_name", "")).upper()
                or float(selected_worker.get("cuda_max_good", 0)) < float(spec.get("nonresearch_min_cuda", 0))
                or float(selected_worker.get("dph_total", float("inf"))) > float(spec.get("worker_price_ceiling_dph", 0))
                or float(selected_gateway.get("dph_total", float("inf"))) > float(spec.get("gateway_price_ceiling_dph", 0))
            ):
                raise LifecycleError("input staging read-only market preflight is not suitable")
            for binding in spec.get("required_input_evidence", []):
                path = root / str(binding.get("path", ""))
                if not path.is_file() or sha256_file(path) != binding.get("sha256"):
                    raise LifecycleError("input staging readiness evidence is missing or changed")
            if len(spec.get("required_input_evidence", [])) < 2:
                raise LifecycleError("input staging readiness evidence is incomplete")
            task_receipt = json.loads(
                (root / spec["required_input_evidence"][0]["path"]).read_text(encoding="utf-8")
            )
            cache_audit = json.loads(
                (root / spec["required_input_evidence"][1]["path"]).read_text(encoding="utf-8")
            )
            if (
                task_receipt.get("status") != "GREEN"
                or task_receipt.get("final_root") != "/volume/parallax/task-cache/parallax_tasks_v1"
                or task_receipt.get("verification", {}).get("failed") != []
                or cache_audit.get("status") != "GREEN"
                or int(cache_audit.get("file_count", 0)) != 12
                or int(cache_audit.get("verified_bytes", 0)) != 11_715_866_017
                or cache_audit.get("final_account_audit", {}).get("active_instances") != 0
            ):
                raise LifecycleError("input staging readiness evidence is not GREEN and exact")
        if spec.get("scope") == "s3_worker_download_probe_only":
            worker_kind = spec.get("lifecycle_worker_kind")
            if worker_kind not in {"non_h100", "h100"} or spec.get("h100_research_authorized") is True:
                raise LifecycleError("S3 download probe must remain infrastructure-only")
            minimum_inet = float(spec.get("nonresearch_min_inet_down_mbps", 0))
            if not 500 <= minimum_inet <= 5000:
                raise LifecycleError("S3 download probe must bind a 500-5000 Mbps advertised floor")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            if not market_path.is_file() or sha256_file(market_path) != market_binding.get("sha256"):
                raise LifecycleError("S3 download probe lacks exact read-only market preflight")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            selected_worker = market.get("selected_worker", {})
            selected_is_h100 = "H100" in str(selected_worker.get("gpu_name", "")).upper()
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or int(market.get("worker_candidate_count", 0)) < 1
                or selected_is_h100 != (worker_kind == "h100")
                or float(selected_worker.get("cuda_max_good", 0))
                < float(spec.get("nonresearch_min_cuda", 0))
                or float(selected_worker.get("inet_down", 0)) < minimum_inet
                or float(selected_worker.get("dph_total", float("inf")))
                > float(spec.get("worker_price_ceiling_dph", 0))
            ):
                raise LifecycleError("S3 download probe market preflight is not suitable")
            if spec.get("worker_template_image_override") is True:
                adjudication_binding = spec.get(
                    "required_h100_managed_runtime_override_adjudication", {}
                )
                adjudication_path = root / str(adjudication_binding.get("path", ""))
                if (
                    worker_kind != "h100"
                    or not adjudication_path.is_file()
                    or sha256_file(adjudication_path) != adjudication_binding.get("sha256")
                ):
                    raise LifecycleError("managed H100 rate probe lacks exact runtime evidence")
                adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
                if (
                    adjudication.get("status")
                    != "GREEN_H100_MANAGED_PRIVATE_RUNTIME_COMPATIBLE"
                    or adjudication.get("next_gate")
                    != "RUN_THE_FROZEN_842380992_BYTE_H100_S3_RATE_PROBE_USING_THIS_EXACT_MANAGED_TEMPLATE_PRIVATE_RUNTIME_CONSTRUCTION"
                ):
                    raise LifecycleError("managed H100 rate probe runtime evidence is unsuitable")
        if spec.get("scope") in {
            "s3_worker_cache_hydration_only", "pg1_model_forward_smoke_only"
        }:
            if (
                spec.get("lifecycle_worker_kind") != "h100"
                or spec.get("h100_research_authorized") is True
                or spec.get("worker_template_image_override") is not True
            ):
                raise LifecycleError("H100 cache hydration must remain infrastructure-only")
            minimum_inet = float(spec.get("nonresearch_min_inet_down_mbps", 0))
            if not 500 <= minimum_inet <= 5000:
                raise LifecycleError("H100 hydration must bind a 500-5000 Mbps advertised floor")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            if not market_path.is_file() or sha256_file(market_path) != market_binding.get("sha256"):
                raise LifecycleError("H100 hydration lacks exact read-only market preflight")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            worker = market.get("selected_worker", {})
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or int(market.get("worker_candidate_count", 0)) < 1
                or "H100" not in str(worker.get("gpu_name", "")).upper()
                or float(worker.get("cuda_max_good", 0)) < float(spec.get("nonresearch_min_cuda", 0))
                or float(worker.get("inet_down", 0)) < minimum_inet
                or float(worker.get("dph_total", float("inf"))) > float(spec.get("worker_price_ceiling_dph", 0))
            ):
                raise LifecycleError("H100 hydration market preflight is unsuitable")
            mirror_binding = spec.get("required_full_s3_mirror_adjudication", {})
            mirror_path = root / str(mirror_binding.get("path", ""))
            if not mirror_path.is_file() or sha256_file(mirror_path) != mirror_binding.get("sha256"):
                raise LifecycleError("H100 hydration lacks exact full-mirror adjudication")
            mirror = json.loads(mirror_path.read_text(encoding="utf-8"))
            if (
                mirror.get("status") != "GREEN_FULL_PG1_S3_MIRROR"
                or mirror.get("decision") != "PASS"
                or mirror.get("independent_versioned_head_audit", {}).get("files_green") != 12
                or mirror.get("independent_versioned_head_audit", {}).get("bytes_green") != 11_715_866_017
                or mirror.get("controls", {}).get("final_active_instances") != 0
            ):
                raise LifecycleError("H100 hydration full-mirror evidence is unsuitable")
            if spec.get("scope") == "pg1_model_forward_smoke_only":
                hydration_binding = spec.get("required_h100_hydration_adjudication", {})
                hydration_path = root / str(hydration_binding.get("path", ""))
                if (
                    not hydration_path.is_file()
                    or sha256_file(hydration_path) != hydration_binding.get("sha256")
                ):
                    raise LifecycleError("PG1 model smoke lacks exact hydration adjudication")
                hydration = json.loads(hydration_path.read_text(encoding="utf-8"))
                if (
                    hydration.get("status") != "GREEN_FULL_PG1_H100_HYDRATION"
                    or hydration.get("decision") != "PASS"
                    or hydration.get("controls", {}).get("confirmation_items_read") != 0
                    or hydration.get("lifecycle", {}).get("final_active_instances") != 0
                ):
                    raise LifecycleError("PG1 model smoke hydration evidence is unsuitable")
        if spec.get("scope") == "h100_stock_ssh_control_only":
            if (
                spec.get("lifecycle_worker_kind") != "h100"
                or spec.get("h100_research_authorized") is True
                or spec.get("worker_image")
                != "vastai/base-image@sha256:a264d0c242bebbdaa6f50b34fb81c7e06cb21d8487a202f27cce22794a2e68e4"
            ):
                raise LifecycleError("H100 stock control must remain stock-image infrastructure only")
            minimum_cuda = float(spec.get("nonresearch_min_cuda", 0))
            minimum_inet = float(spec.get("nonresearch_min_inet_down_mbps", 0))
            if minimum_cuda != 13.0 or minimum_inet != 500.0:
                raise LifecycleError("H100 stock control driver and downlink floors changed")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            boundary_binding = spec.get("required_h100_runtime_boundary", {})
            boundary_path = root / str(boundary_binding.get("path", ""))
            if (
                not market_path.is_file()
                or sha256_file(market_path) != market_binding.get("sha256")
                or not boundary_path.is_file()
                or sha256_file(boundary_path) != boundary_binding.get("sha256")
            ):
                raise LifecycleError("H100 stock control evidence is missing or changed")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
            selected_worker = market.get("selected_worker", {})
            excluded_machines = {int(value) for value in spec.get("excluded_worker_machine_ids", [])}
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or int(market.get("worker_candidate_count", 0)) < 1
                or "H100" not in str(selected_worker.get("gpu_name", "")).upper()
                or float(selected_worker.get("cuda_max_good", 0)) < minimum_cuda
                or float(selected_worker.get("inet_down", 0)) < minimum_inet
                or float(selected_worker.get("dph_total", float("inf")))
                > float(spec.get("worker_price_ceiling_dph", 0))
                or int(selected_worker.get("machine_id", -1)) in excluded_machines
                or not {141549, 141956}.issubset(excluded_machines)
                or boundary.get("status") != "RED_REPRODUCED_PRE_SSH_RUNTIME_START_BOUNDARY"
                or boundary.get("decision") != "NO_THIRD_FULL_IMAGE_H100_RETRY"
            ):
                raise LifecycleError("H100 stock control preflight is not exact and suitable")
        if spec.get("scope") == "h100_official_template_control_only":
            if (
                spec.get("lifecycle_worker_kind") != "h100"
                or spec.get("h100_research_authorized") is True
            ):
                raise LifecycleError("official H100 template control cannot authorize research")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            boundary_binding = spec.get("required_h100_pre_ssh_boundary", {})
            boundary_path = root / str(boundary_binding.get("path", ""))
            if (
                not market_path.is_file()
                or sha256_file(market_path) != market_binding.get("sha256")
                or not boundary_path.is_file()
                or sha256_file(boundary_path) != boundary_binding.get("sha256")
            ):
                raise LifecycleError("official H100 template control evidence changed")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
            selected = market.get("selected_worker", {})
            excluded = {int(value) for value in spec.get("excluded_worker_machine_ids", [])}
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or "H100" not in str(selected.get("gpu_name", "")).upper()
                or float(selected.get("cuda_max_good", 0)) < 13.0
                or float(selected.get("inet_down", 0)) < 500.0
                or float(selected.get("dph_total", float("inf")))
                > float(spec.get("worker_price_ceiling_dph", 0))
                or int(selected.get("machine_id", -1)) in excluded
                or not {81035, 141549, 141956}.issubset(excluded)
                or boundary.get("status")
                != "RED_REPRODUCED_ACROSS_CUSTOM_AND_STOCK_DIGEST_IMAGES"
                or boundary.get("next_gate")
                != "ONE_CURRENT_OFFICIAL_RECOMMENDED_NVIDIA_CUDA_TEMPLATE_CONTROL"
            ):
                raise LifecycleError("official H100 template control preflight is unsuitable")
        if spec.get("scope") == "managed_template_runtime_override_probe":
            if (
                spec.get("lifecycle_worker_kind") != "non_h100"
                or spec.get("h100_research_authorized") is True
                or spec.get("worker_template_image_override") is not True
                or spec.get("worker_image")
                != "769063704696.dkr.ecr.us-east-1.amazonaws.com/parallax/runtime-minimal@sha256:f7407a831951f4c744a9afecf59d248bfadd31d44d012f200bc5055b6d4d4ed6"
            ):
                raise LifecycleError("managed runtime override must remain exact and non-H100")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            adjudication_binding = spec.get("required_h100_launch_mode_adjudication", {})
            adjudication_path = root / str(adjudication_binding.get("path", ""))
            if (
                not market_path.is_file()
                or sha256_file(market_path) != market_binding.get("sha256")
                or not adjudication_path.is_file()
                or sha256_file(adjudication_path) != adjudication_binding.get("sha256")
            ):
                raise LifecycleError("managed runtime override evidence is missing or changed")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
            selected = market.get("selected_worker", {})
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or int(market.get("worker_candidate_count", 0)) < 1
                or "H100" in str(selected.get("gpu_name", "")).upper()
                or float(selected.get("cuda_max_good", 0)) < 13.0
                or float(selected.get("dph_total", float("inf")))
                > float(spec.get("worker_price_ceiling_dph", 0))
                or adjudication.get("status")
                != "GREEN_OFFICIAL_TEMPLATE_ISOLATES_RAW_LAUNCH_MODE_FAILURE"
                or adjudication.get("next_gate")
                != "ON_CHEAP_NON_H100_OVERRIDE_THE_GREEN_OFFICIAL_TEMPLATE_WITH_THE_EXACT_PRIVATE_FULL_RUNTIME_IMAGE_AND_EPHEMERAL_ECR_LOGIN"
            ):
                raise LifecycleError("managed runtime override preflight is unsuitable")
        if spec.get("scope") == "h100_managed_template_runtime_override_probe":
            if (
                spec.get("lifecycle_worker_kind") != "h100"
                or spec.get("h100_research_authorized") is True
                or spec.get("worker_template_image_override") is not True
                or spec.get("worker_image")
                != "769063704696.dkr.ecr.us-east-1.amazonaws.com/parallax/runtime-minimal@sha256:f7407a831951f4c744a9afecf59d248bfadd31d44d012f200bc5055b6d4d4ed6"
            ):
                raise LifecycleError("H100 managed runtime override must remain exact and nonresearch")
            market_binding = spec.get("required_market_preflight", {})
            market_path = root / str(market_binding.get("path", ""))
            adjudication_binding = spec.get("required_managed_runtime_override_adjudication", {})
            adjudication_path = root / str(adjudication_binding.get("path", ""))
            if (
                not market_path.is_file()
                or sha256_file(market_path) != market_binding.get("sha256")
                or not adjudication_path.is_file()
                or sha256_file(adjudication_path) != adjudication_binding.get("sha256")
            ):
                raise LifecycleError("H100 managed runtime override evidence is missing or changed")
            market = json.loads(market_path.read_text(encoding="utf-8"))
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
            selected = market.get("selected_worker", {})
            if (
                market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or int(market.get("worker_candidate_count", 0)) < 1
                or "H100" not in str(selected.get("gpu_name", "")).upper()
                or float(selected.get("cuda_max_good", 0)) < 13.0
                or float(selected.get("dph_total", float("inf")))
                > float(spec.get("worker_price_ceiling_dph", 0))
                or adjudication.get("status")
                != "GREEN_MANAGED_TEMPLATE_PRIVATE_RUNTIME_COMPATIBLE"
                or adjudication.get("next_gate")
                != "ONE_TINY_NO_DOWNLOAD_NO_GPU_H100_RUN_WITH_THE_IDENTICAL_MANAGED_TEMPLATE_PLUS_EXACT_PRIVATE_RUNTIME_CONSTRUCTION"
            ):
                raise LifecycleError("H100 managed runtime override preflight is unsuitable")
        loading_seconds = int(spec.get("policy", {}).get("maximum_provider_loading_seconds", 0))
        if not 60 <= loading_seconds <= 3600:
            raise LifecycleError("paid lifecycle spec must bound the unbilled provider loading phase")
        guardian_poll_seconds = float(spec.get("policy", {}).get("guardian_poll_seconds", 0))
        if not 1 <= guardian_poll_seconds <= 5:
            raise LifecycleError("paid lifecycle guardian polling must be rate-limited and bounded")
        maximum_prearm_cost = float(spec.get("policy", {}).get("maximum_prearm_worker_cost_usd", 0))
        prearm_seconds = int(spec.get("policy", {}).get("maximum_worker_prearm_seconds", 0))
        if not 30 <= prearm_seconds <= 300:
            raise LifecycleError("paid lifecycle spec must bound worker pre-arm time")
        frozen_ceiling_cost = float(spec.get("worker_price_ceiling_dph", 0)) * prearm_seconds / 3600
        if maximum_prearm_cost <= 0 or frozen_ceiling_cost > maximum_prearm_cost + 1e-9:
            raise LifecycleError("frozen worker ceiling exceeds the maximum pre-arm cost")
        minimum_hard_stop = (
            prearm_seconds
            + int(spec.get("policy", {}).get("maximum_runtime_seconds", 0))
            + int(spec.get("policy", {}).get("maximum_finalization_seconds", 0))
            + int(spec.get("policy", {}).get("heartbeat_timeout_seconds", 0))
        )
        hard_stop = int(spec.get("policy", {}).get("maximum_worker_hard_stop_seconds", 0))
        if not minimum_hard_stop <= hard_stop <= minimum_hard_stop + 30:
            raise LifecycleError("worker hard STOP must tightly cover only the frozen lifecycle windows")
        gateway_ttl = (
            int(spec.get("policy", {}).get("bootstrap_timeout_seconds", 0))
            + loading_seconds
            + hard_stop
            + 300
        )
        maximum_gateway_cost = float(
            spec.get("policy", {}).get("maximum_gateway_lifecycle_cost_usd", 0)
        )
        frozen_gateway_cost = float(spec.get("gateway_price_ceiling_dph", 0)) * gateway_ttl / 3600
        if maximum_gateway_cost <= 0 or frozen_gateway_cost > maximum_gateway_cost + 1e-9:
            raise LifecycleError("frozen gateway ceiling exceeds maximum lifecycle cost")
        if int(spec.get("authorized_volume_id", 0)) <= 0:
            raise LifecycleError("paid lifecycle spec must bind one persistent volume")
        if not spec.get("relay_ssh_private_key") or int(spec.get("relay_ssh_public_key_id", 0)) <= 0:
            raise LifecycleError("paid lifecycle spec must bind one dedicated registered relay key")
        _validate_live_control_evidence(root, spec)
        _validate_scoped_key_binding(root, spec)
    if spec.get("paid_control_probe_authorized") is True:
        if int(spec.get("authorized_volume_id", 0)) <= 0:
            raise LifecycleError("paid control probe must bind one persistent volume")
        if float(spec.get("gateway_price_ceiling_dph", 0)) <= 0:
            raise LifecycleError("paid control probe must bind a gateway price ceiling")
        if not spec.get("relay_ssh_private_key") or int(spec.get("relay_ssh_public_key_id", 0)) <= 0:
            raise LifecycleError("paid control probe must bind one dedicated registered relay key")
        stop_seconds = int(spec.get("policy", {}).get("scoped_stop_proof_seconds", 0))
        if not 5 <= stop_seconds <= 60:
            raise LifecycleError("paid control probe must bind a 5-60 second scoped STOP deadline")
        maximum_control_cost = float(spec.get("policy", {}).get("maximum_control_cost_usd", 0))
        maximum_control_seconds = int(spec.get("policy", {}).get("bootstrap_timeout_seconds", 0)) + stop_seconds + 60
        frozen_control_cost = float(spec.get("gateway_price_ceiling_dph", 0)) * maximum_control_seconds / 3600
        if maximum_control_cost <= 0 or frozen_control_cost > maximum_control_cost + 1e-9:
            raise LifecycleError("frozen gateway ceiling exceeds maximum control-proof cost")
        _validate_scoped_key_binding(root, spec)
    if spec.get("paid_inspection_authorized") is True:
        gateway_contract = spec.get("inspection_gateway_image_contract", {})
        gateway_reference = str(spec.get("image", ""))
        gateway_receipt_binding = gateway_contract.get("registry_receipt", {})
        gateway_receipt_path = root / str(gateway_receipt_binding.get("path", ""))
        if (
            gateway_contract.get("reference") != gateway_reference
            or gateway_contract.get("architecture") != "amd64"
            or int(gateway_contract.get("compressed_bytes", 0)) <= 0
            or int(gateway_contract.get("compressed_bytes", 0))
            > int(gateway_contract.get("maximum_compressed_bytes", 0))
            or not gateway_receipt_path.is_file()
            or sha256_file(gateway_receipt_path) != gateway_receipt_binding.get("sha256")
        ):
            raise LifecycleError("paid inspection lacks an exact bounded gateway image contract")
        gateway_receipt = json.loads(gateway_receipt_path.read_text(encoding="utf-8"))
        if (
            gateway_receipt.get("status") != "GREEN_REGISTRY_MANIFEST"
            or gateway_receipt.get("reference") != gateway_reference
            or int(gateway_receipt.get("compressed_bytes", 0))
            != int(gateway_contract.get("compressed_bytes", 0))
        ):
            raise LifecycleError("paid inspection gateway registry receipt does not match")
        if not 20 <= int(spec.get("gateway_disk_gb", 0)) <= 50:
            raise LifecycleError("paid inspection gateway disk must be bounded in [20, 50] GB")
        if int(spec.get("authorized_volume_id", 0)) <= 0:
            raise LifecycleError("paid inspection must bind one persistent volume")
        if float(spec.get("gateway_price_ceiling_dph", 0)) <= 0:
            raise LifecycleError("paid inspection must bind a gateway price ceiling")
        deadman_seconds = int(spec.get("policy", {}).get("inspection_deadman_seconds", 0))
        if not 30 <= deadman_seconds <= 300:
            raise LifecycleError("paid inspection must bind a short scoped deadman")
        maximum_seconds = (
            int(spec.get("policy", {}).get("bootstrap_timeout_seconds", 0))
            + deadman_seconds
            + 30
        )
        frozen_cost = float(spec.get("gateway_price_ceiling_dph", 0)) * maximum_seconds / 3600
        maximum_cost = float(spec.get("policy", {}).get("maximum_inspection_cost_usd", 0))
        if maximum_cost <= 0 or frozen_cost > maximum_cost + 1e-9:
            raise LifecycleError("frozen inspection ceiling exceeds maximum inspection cost")
        maximum_file_bytes = int(
            spec.get("policy", {}).get("maximum_local_inspection_file_bytes", 0)
        )
        maximum_total_bytes = int(
            spec.get("policy", {}).get("maximum_local_inspection_artifact_bytes", 0)
        )
        if not 1 <= maximum_file_bytes <= 50_000_000:
            raise LifecycleError("paid inspection must bind a bounded local per-file limit")
        if not maximum_file_bytes <= maximum_total_bytes <= 100_000_000:
            raise LifecycleError("paid inspection must bind a bounded local total-artifact limit")
        if spec.get("scope") == "s3_worker_cache_hydration_only":
            names = ("required_launch_receipt", "required_launch_market", "required_launch_claim")
            bindings = [spec.get(name, {}) for name in names]
            paths = [root / str(binding.get("path", "")) for binding in bindings]
            if any(
                not path.is_file() or sha256_file(path) != binding.get("sha256")
                for path, binding in zip(paths, bindings, strict=True)
            ):
                raise LifecycleError("hydration inspection lacks exact launch bindings")
            launch, market, claim = [
                json.loads(path.read_text(encoding="utf-8")) for path in paths
            ]
            if (
                launch.get("status") != "SUPERVISOR_RELEASED"
                or launch.get("run_id") != "parallax-pg1-h100-s3-hydration-v1"
                or int(launch.get("h100_instance_id", 0)) <= 0
                or launch.get("local_controller_required_after_arm") is not False
                or launch.get("secrets_logged") is not False
                or launch.get("market_receipt_sha256") != sha256_file(paths[1])
                or launch.get("authorization_claim_sha256") != sha256_file(paths[2])
                or market.get("mutation") != "none_read_only"
                or market.get("all_regions") is not True
                or "H100" not in str(market.get("selected_worker", {}).get("gpu_name", "")).upper()
                or claim.get("status") != "CONSUMED_BEFORE_MARKET_MUTATION"
                or claim.get("mode") != "launch"
                or claim.get("authorization_id") != launch.get("run_id")
            ):
                raise LifecycleError("hydration inspection launch evidence is unsuitable")
        _validate_live_control_evidence(root, spec)
        _validate_scoped_key_binding(root, spec)
    if spec.get("paid_cache_audit_authorized") is True:
        gateway_contract = spec.get("inspection_gateway_image_contract", {})
        gateway_reference = str(spec.get("image", ""))
        receipt_binding = gateway_contract.get("registry_receipt", {})
        receipt_path = root / str(receipt_binding.get("path", ""))
        if (
            gateway_contract.get("reference") != gateway_reference
            or gateway_contract.get("architecture") != "amd64"
            or int(gateway_contract.get("compressed_bytes", 0)) <= 0
            or int(gateway_contract.get("compressed_bytes", 0))
            > int(gateway_contract.get("maximum_compressed_bytes", 0))
            or not receipt_path.is_file()
            or sha256_file(receipt_path) != receipt_binding.get("sha256")
        ):
            raise LifecycleError("paid cache audit lacks an exact bounded gateway image contract")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "GREEN_REGISTRY_MANIFEST"
            or receipt.get("reference") != gateway_reference
            or int(receipt.get("compressed_bytes", 0))
            != int(gateway_contract.get("compressed_bytes", 0))
        ):
            raise LifecycleError("paid cache audit gateway registry receipt does not match")
        if not 20 <= int(spec.get("gateway_disk_gb", 0)) <= 50:
            raise LifecycleError("paid cache audit gateway disk must be bounded in [20, 50] GB")
        if int(spec.get("authorized_volume_id", 0)) <= 0:
            raise LifecycleError("paid cache audit must bind one persistent volume")
        if float(spec.get("gateway_price_ceiling_dph", 0)) <= 0:
            raise LifecycleError("paid cache audit must bind a gateway price ceiling")
        deadman_seconds = int(spec.get("policy", {}).get("cache_audit_deadman_seconds", 0))
        if not 60 <= deadman_seconds <= 900:
            raise LifecycleError("paid cache audit must bind a 60-900 second deadman")
        maximum_seconds = int(spec.get("policy", {}).get("bootstrap_timeout_seconds", 0)) + deadman_seconds + 30
        frozen_cost = float(spec.get("gateway_price_ceiling_dph", 0)) * maximum_seconds / 3600
        maximum_cost = float(spec.get("policy", {}).get("maximum_cache_audit_cost_usd", 0))
        if maximum_cost <= 0 or frozen_cost > maximum_cost + 1e-9:
            raise LifecycleError("frozen cache-audit ceiling exceeds maximum cache-audit cost")
        _validate_bound_cache_audit_spec(root, spec)
        _validate_live_control_evidence(root, spec)
        _validate_scoped_key_binding(root, spec)
    if spec.get("paid_object_mirror_authorized") is True:
        gateway_contract = spec.get("inspection_gateway_image_contract", {})
        gateway_reference = str(spec.get("image", ""))
        receipt_binding = gateway_contract.get("registry_receipt", {})
        receipt_path = root / str(receipt_binding.get("path", ""))
        if (
            gateway_contract.get("reference") != gateway_reference
            or gateway_contract.get("architecture") != "amd64"
            or int(gateway_contract.get("compressed_bytes", 0)) <= 0
            or int(gateway_contract.get("compressed_bytes", 0))
            > int(gateway_contract.get("maximum_compressed_bytes", 0))
            or not receipt_path.is_file()
            or sha256_file(receipt_path) != receipt_binding.get("sha256")
        ):
            raise LifecycleError("paid object mirror lacks an exact bounded gateway image contract")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "GREEN_REGISTRY_MANIFEST"
            or receipt.get("reference") != gateway_reference
            or int(receipt.get("compressed_bytes", 0))
            != int(gateway_contract.get("compressed_bytes", 0))
        ):
            raise LifecycleError("paid object mirror gateway registry receipt does not match")
        if not 20 <= int(spec.get("gateway_disk_gb", 0)) <= 50:
            raise LifecycleError("paid object mirror gateway disk must be bounded in [20, 50] GB")
        if int(spec.get("authorized_volume_id", 0)) <= 0:
            raise LifecycleError("paid object mirror must bind one persistent volume")
        if float(spec.get("gateway_price_ceiling_dph", 0)) <= 0:
            raise LifecycleError("paid object mirror must bind a gateway price ceiling")
        deadman_seconds = int(spec.get("policy", {}).get("object_mirror_deadman_seconds", 0))
        cache_mirror_scope = spec.get("scope") in {
            "pg1_s3_cache_mirror_rehearsal", "pg1_s3_cache_mirror_full"
        }
        maximum_deadman = 1800 if cache_mirror_scope else 600
        if not 60 <= deadman_seconds <= maximum_deadman:
            raise LifecycleError(
                f"paid object mirror must bind a 60-{maximum_deadman} second deadman"
            )
        maximum_seconds = (
            int(spec.get("policy", {}).get("bootstrap_timeout_seconds", 0))
            + deadman_seconds
            + 30
        )
        frozen_cost = float(spec.get("gateway_price_ceiling_dph", 0)) * maximum_seconds / 3600
        maximum_cost = float(spec.get("policy", {}).get("maximum_object_mirror_cost_usd", 0))
        if maximum_cost <= 0 or frozen_cost > maximum_cost + 1e-9:
            raise LifecycleError("frozen object-mirror ceiling exceeds maximum cost")
        if cache_mirror_scope:
            _validate_bound_cache_mirror_spec(root, spec)
        else:
            _validate_bound_object_mirror_spec(root, spec)
        _validate_live_control_evidence(root, spec)
        _validate_scoped_key_binding(root, spec)
    if spec.get("paid_task_materialization_authorized") is True:
        _validate_worker_image_contract(root, spec)
        if spec.get("image") != spec.get("worker_image"):
            raise LifecycleError("task materialization gateway must use the exact qualified worker image")
        qualification_binding = spec.get("required_full_image_qualification", {})
        qualification_path = root / str(qualification_binding.get("path", ""))
        if (
            not qualification_path.is_file()
            or sha256_file(qualification_path) != qualification_binding.get("sha256")
        ):
            raise LifecycleError("task materialization lacks exact full-image qualification evidence")
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        if (
            qualification.get("status") != "GREEN_FULL_SCIENCE_RUNTIME_COMPATIBLE"
            or qualification.get("image_reference") != spec.get("image")
            or qualification.get("lifecycle", {}).get("final_vast_active_instances") != 0
        ):
            raise LifecycleError("task materialization full-image qualification is not green")
        if not 20 <= int(spec.get("gateway_disk_gb", 0)) <= 50:
            raise LifecycleError("paid task materialization gateway disk must be bounded in [20, 50] GB")
        if int(spec.get("authorized_volume_id", 0)) <= 0:
            raise LifecycleError("paid task materialization must bind one persistent volume")
        if float(spec.get("gateway_price_ceiling_dph", 0)) <= 0:
            raise LifecycleError("paid task materialization must bind a gateway price ceiling")
        deadman_seconds = int(spec.get("policy", {}).get("task_materialization_deadman_seconds", 0))
        if not 300 <= deadman_seconds <= 1800:
            raise LifecycleError("paid task materialization must bind a 300-1800 second deadman")
        maximum_seconds = int(spec.get("policy", {}).get("bootstrap_timeout_seconds", 0)) + deadman_seconds + 30
        frozen_cost = float(spec.get("gateway_price_ceiling_dph", 0)) * maximum_seconds / 3600
        maximum_cost = float(spec.get("policy", {}).get("maximum_task_materialization_cost_usd", 0))
        if maximum_cost <= 0 or frozen_cost > maximum_cost + 1e-9:
            raise LifecycleError("frozen task-materialization ceiling exceeds maximum cost")
        _validate_bound_task_cache_spec(root, spec)
        _validate_live_control_evidence(root, spec)
        _validate_scoped_key_binding(root, spec)
    plant = spec.get("required_plant", {})
    plant_path = root / plant.get("path", "")
    if not plant_path.is_file() or sha256_file(plant_path) != plant.get("sha256"):
        raise LifecycleError("required lifecycle plant receipt is missing or changed")
    plant_payload = json.loads(plant_path.read_text(encoding="utf-8"))
    if plant_payload.get("status") != "GREEN":
        raise LifecycleError("lifecycle plants are not green")
    for name, binding in spec.get("implementation", {}).items():
        bound_path = root / str(binding.get("path", ""))
        if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
            raise LifecycleError(f"frozen lifecycle implementation changed: {name}")
    if not spec.get("implementation"):
        raise LifecycleError("lifecycle spec does not bind its implementation")
    for mode, field in (
        ("launch", "paid_launch_authorized"),
        ("inspect", "paid_inspection_authorized"),
        ("control", "paid_control_probe_authorized"),
        ("audit", "paid_cache_audit_authorized"),
        ("mirror", "paid_object_mirror_authorized"),
        ("task", "paid_task_materialization_authorized"),
    ):
        if spec.get(field) is True:
            authorization = spec.get("one_shot_authorization", {})
            claim_path = authorization.get("claim_paths", {}).get(mode)
            if not authorization.get("id") or not claim_path:
                raise LifecycleError(f"paid Vast {mode} requires an explicit one-shot claim path")
    if spec.get("h100_research_authorized") is True:
        from .lifecycle_launch_packet import validate_lifecycle_launch_packet

        binding = spec.get("required_launch_packet", {})
        packet_path = root / str(binding.get("path", ""))
        if not packet_path.is_file() or sha256_file(packet_path) != binding.get("sha256"):
            raise LifecycleError("research lifecycle spec lacks its exact GO launch packet")
        validate_lifecycle_launch_packet(packet_path, repo_root=root, verify_evidence=True)
    return spec


def _validate_bound_cache_audit_spec(
    root: Path, spec: dict[str, Any], audit_path: Path | None = None
) -> dict[str, Any]:
    binding = spec.get("authorized_cache_audit_spec", {})
    bound_path = root / str(binding.get("path", ""))
    if audit_path is not None and bound_path.resolve() != audit_path.resolve():
        raise LifecycleError("cache audit spec path is not authorized by lifecycle freeze")
    if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
        raise LifecycleError("authorized cache audit spec is missing or changed")
    payload = json.loads(bound_path.read_text(encoding="utf-8"))
    validate_audit_spec(payload)
    source_binding = payload.get("source_manifest", {})
    source_path = root / str(source_binding.get("path", ""))
    if not source_path.is_file() or sha256_file(source_path) != source_binding.get("sha256"):
        raise LifecycleError("cache audit source manifest is missing or changed")
    volume_mount = PurePosixPath(str(spec.get("volume_mount", "")))
    cache_root = PurePosixPath(str(payload.get("cache_root", "")))
    if not cache_root.is_relative_to(volume_mount):
        raise LifecycleError("cache audit root is outside the authorized persistent volume")
    if int(payload["maximum_audit_bytes"]) > int(
        spec.get("policy", {}).get("maximum_cache_audit_bytes", 0)
    ):
        raise LifecycleError("cache audit byte ceiling exceeds lifecycle authorization")
    return payload


def _validate_bound_object_mirror_spec(
    root: Path, spec: dict[str, Any], mirror_path: Path | None = None
) -> dict[str, Any]:
    binding = spec.get("authorized_object_mirror_spec", {})
    bound_path = root / str(binding.get("path", ""))
    if mirror_path is not None and bound_path.resolve() != mirror_path.resolve():
        raise LifecycleError("object mirror spec path is not authorized by lifecycle freeze")
    if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
        raise LifecycleError("authorized object mirror spec is missing or changed")
    payload = json.loads(bound_path.read_text(encoding="utf-8"))
    validate_mirror_spec(payload)
    volume_mount = PurePosixPath(str(spec.get("volume_mount", "")))
    source_path = PurePosixPath(str(payload.get("source", {}).get("path", "")))
    durable_result = PurePosixPath(str(payload.get("durable_result", "")))
    if not source_path.is_relative_to(volume_mount) or not durable_result.is_relative_to(volume_mount):
        raise LifecycleError("object mirror paths are outside the authorized persistent volume")
    evidence_bindings = spec.get("required_cache_audit_evidence", [])
    if not isinstance(evidence_bindings, list) or len(evidence_bindings) != 2:
        raise LifecycleError("object mirror requires audit envelope and exact result")
    evidence_paths = [root / str(binding.get("path", "")) for binding in evidence_bindings]
    if any(
        not path.is_file() or sha256_file(path) != binding.get("sha256")
        for path, binding in zip(evidence_paths, evidence_bindings, strict=True)
    ):
        raise LifecycleError("object mirror lacks exact cache-audit evidence")
    envelope = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
    evidence = json.loads(evidence_paths[1].read_text(encoding="utf-8"))
    expected_source = payload["source"]
    audited_files = {
        row.get("path"): row for row in evidence.get("files", []) if isinstance(row, dict)
    }
    audited = audited_files.get(PurePosixPath(str(expected_source["path"])).name)
    if (
        envelope.get("status") != "GREEN"
        or envelope.get("final_account_audit", {}).get("active_instances") != 0
        or envelope.get("audit_result_sha256") != sha256_file(evidence_paths[1])
        or evidence.get("status") != "GREEN"
        or audited is None
        or int(audited.get("bytes", -1)) != int(expected_source["bytes"])
        or audited.get("sha256") != expected_source["sha256"]
    ):
        raise LifecycleError("object mirror source is not present in the GREEN exact cache audit")
    return payload


def _validate_bound_cache_mirror_spec(
    root: Path, spec: dict[str, Any], mirror_path: Path | None = None
) -> dict[str, Any]:
    binding = spec.get("authorized_cache_mirror_spec", {})
    bound_path = root / str(binding.get("path", ""))
    if mirror_path is not None and bound_path.resolve() != mirror_path.resolve():
        raise LifecycleError("cache mirror spec is not authorized by lifecycle freeze")
    if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
        raise LifecycleError("authorized cache mirror spec is missing or changed")
    payload = json.loads(bound_path.read_text(encoding="utf-8"))
    validate_cache_mirror_spec(payload)
    expected_mode = (
        "rehearsal" if spec.get("scope") == "pg1_s3_cache_mirror_rehearsal" else "full"
    )
    if payload.get("mode") != expected_mode:
        raise LifecycleError("cache mirror lifecycle scope and mode disagree")
    volume_mount = PurePosixPath(str(spec.get("volume_mount", "")))
    if (
        not PurePosixPath(str(payload["source_root"])).is_relative_to(volume_mount)
        or not PurePosixPath(str(payload["durable_result"])).is_relative_to(volume_mount)
    ):
        raise LifecycleError("cache mirror paths escape the authorized persistent volume")
    evidence_bindings = spec.get("required_cache_audit_evidence", [])
    if not isinstance(evidence_bindings, list) or len(evidence_bindings) != 2:
        raise LifecycleError("cache mirror requires audit envelope and exact result")
    evidence_paths = [root / str(item.get("path", "")) for item in evidence_bindings]
    if any(
        not path.is_file() or sha256_file(path) != item.get("sha256")
        for path, item in zip(evidence_paths, evidence_bindings, strict=True)
    ):
        raise LifecycleError("cache mirror lacks exact cache-audit evidence")
    envelope = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
    audit = json.loads(evidence_paths[1].read_text(encoding="utf-8"))
    audited = {
        str(row.get("path")): row for row in audit.get("files", []) if isinstance(row, dict)
    }
    for row in payload["files"]:
        source = audited.get(str(row["path"]))
        if (
            source is None
            or source.get("match") is not True
            or int(source.get("bytes", -1)) != int(row["bytes"])
            or source.get("sha256") != row["sha256"]
        ):
            raise LifecycleError("cache mirror file is absent from the exact audit")
    if (
        envelope.get("status") != "GREEN"
        or envelope.get("final_account_audit", {}).get("active_instances") != 0
        or envelope.get("audit_result_sha256") != sha256_file(evidence_paths[1])
        or audit.get("status") != "GREEN"
        or int(audit.get("verified_bytes", 0)) != 11_715_866_017
    ):
        raise LifecycleError("cache mirror audit evidence is not GREEN and exact")
    if expected_mode == "rehearsal":
        expected_paths = {
            name for name in audited if not name.endswith(".safetensors")
        }
        if {str(row["path"]) for row in payload["files"]} != expected_paths:
            raise LifecycleError("cache mirror rehearsal must cover all nine non-weight files")
    else:
        rehearsal_bindings = spec.get("required_cache_mirror_rehearsal_evidence", [])
        if not isinstance(rehearsal_bindings, list) or len(rehearsal_bindings) != 4:
            raise LifecycleError("full cache mirror requires four exact rehearsal receipts")
        rehearsal_paths = [
            root / str(item.get("path", "")) for item in rehearsal_bindings
        ]
        if any(
            not path.is_file() or sha256_file(path) != item.get("sha256")
            for path, item in zip(
                rehearsal_paths, rehearsal_bindings, strict=True
            )
        ):
            raise LifecycleError("full cache mirror lacks exact rehearsal evidence")
        controller, remote, manifest, account = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in rehearsal_paths
        ]
        if (
            controller.get("status") != "GREEN"
            or controller.get("mode") != "rehearsal"
            or controller.get("final_account_audit", {}).get("active_instances") != 0
            or controller.get("worker_created") is not False
            or controller.get("gpu_work") is not False
            or controller.get("model_loaded") is not False
            or controller.get("automatic_retry") is not False
            or controller.get("aws_credentials_transmitted") is not False
            or controller.get("url_persisted") is not False
            or int(controller.get("verified_bytes", -1)) != 21_918_641
            or int(controller.get("uploaded_bytes", -1)) != 4_369_037
            or int(controller.get("reused_bytes", -1)) != 17_549_604
            or controller.get("mirror_result_sha256") != sha256_file(rehearsal_paths[1])
            or controller.get("s3_manifest_sha256") != sha256_file(rehearsal_paths[2])
            or remote.get("status") != "GREEN_MIRRORED"
            or remote.get("mode") != "rehearsal"
            or int(remote.get("expected_file_count", -1)) != 9
            or int(remote.get("file_count", -1)) != 9
            or int(remote.get("expected_total_bytes", -1)) != 21_918_641
            or remote.get("failures") != []
            or remote.get("url_file_deleted_before_source_reads") is not True
            or remote.get("url_persisted") is not False
            or remote.get("automatic_retry") is not False
            or remote.get("aws_credentials_present_on_gateway") is not False
            or manifest.get("status") != "GREEN"
            or manifest.get("mode") != "rehearsal"
            or int(manifest.get("file_count", -1)) != 9
            or int(manifest.get("total_bytes", -1)) != 21_918_641
            or manifest.get("errors") != []
            or any(row.get("status") != "GREEN" for row in manifest.get("files", []))
            or account != {
                "active_instances": 0,
                "instance_ids": [],
                "query_succeeded": True,
            }
        ):
            raise LifecycleError("cache mirror rehearsal evidence is not GREEN and exact")
    rate_binding = spec.get("required_h100_rate_gate", {})
    rate_path = root / str(rate_binding.get("path", ""))
    if not rate_path.is_file() or sha256_file(rate_path) != rate_binding.get("sha256"):
        raise LifecycleError("cache mirror lacks exact H100 rate evidence")
    rate = json.loads(rate_path.read_text(encoding="utf-8"))
    if (
        rate.get("status") != "GREEN_H100_S3_RATE_GATE"
        or rate.get("adjudication", {}).get("decision") != "PASS"
        or float(rate.get("adjudication", {}).get("projected_full_cache_seconds", 9999)) > 300
        or rate.get("controls", {}).get("final_active_instances") != 0
    ):
        raise LifecycleError("cache mirror H100 rate evidence is unsuitable")
    return payload


def _validate_bound_task_cache_spec(
    root: Path, spec: dict[str, Any], task_spec_path: Path | None = None
) -> dict[str, Any]:
    binding = spec.get("authorized_task_cache_spec", {})
    bound_path = root / str(binding.get("path", ""))
    if task_spec_path is not None and bound_path.resolve() != task_spec_path.resolve():
        raise LifecycleError("task cache spec path is not authorized by lifecycle freeze")
    if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
        raise LifecycleError("authorized task cache spec is missing or changed")
    payload = json.loads(bound_path.read_text(encoding="utf-8"))
    validate_task_cache_spec(payload)
    volume_mount = PurePosixPath(str(spec.get("volume_mount", "")))
    if not PurePosixPath(str(payload["volume_root"])).is_relative_to(volume_mount):
        raise LifecycleError("task cache root is outside the authorized persistent volume")
    if int(payload["maximum_payload_bytes"]) > int(
        spec.get("policy", {}).get("maximum_task_payload_bytes", 0)
    ):
        raise LifecycleError("task payload ceiling exceeds lifecycle authorization")
    for provenance in payload.get("provenance_bindings", []):
        path = root / str(provenance.get("path", ""))
        if not path.is_file() or sha256_file(path) != provenance.get("sha256"):
            raise LifecycleError("task cache provenance binding is missing or changed")
    if len(payload.get("provenance_bindings", [])) < 2:
        raise LifecycleError("task cache provenance bindings are incomplete")
    return payload


def _validate_worker_image_contract(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    contract = spec.get("worker_image_contract", {})
    reference = str(spec.get("worker_image", ""))
    if not reference or contract.get("reference") != reference:
        raise LifecycleError("paid worker image is not bound to its exact contract")
    repository, separator, digest = reference.partition("@sha256:")
    if not repository or separator != "@sha256:" or len(digest) != 64:
        raise LifecycleError("paid worker image must use an exact sha256 digest reference")
    try:
        int(digest, 16)
    except ValueError as error:
        raise LifecycleError("paid worker image digest is malformed") from error
    if contract.get("architecture") != "amd64" or contract.get("manifest_sha256") != digest:
        raise LifecycleError("paid worker image architecture or manifest digest is not exact")
    compressed_bytes = int(contract.get("compressed_bytes", 0))
    maximum_bytes = int(contract.get("maximum_compressed_bytes", 0))
    if compressed_bytes <= 0 or maximum_bytes <= 0 or compressed_bytes > maximum_bytes:
        raise LifecycleError("paid worker image exceeds its frozen compressed-size gate")
    if contract.get("required_live_commands") != ["bash", "python3", "rsync", "ssh", "sha256sum"]:
        raise LifecycleError("paid worker image live command gate is incomplete")
    binding = contract.get("registry_receipt", {})
    receipt_path = root / str(binding.get("path", ""))
    if not receipt_path.is_file() or sha256_file(receipt_path) != binding.get("sha256"):
        raise LifecycleError("paid worker image registry receipt is missing or changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "GREEN_REGISTRY_MANIFEST"
        or receipt.get("reference") != reference
        or receipt.get("architecture") != "amd64"
        or receipt.get("manifest_sha256") != digest
        or int(receipt.get("compressed_bytes", 0)) != compressed_bytes
    ):
        raise LifecycleError("paid worker image registry receipt does not match its contract")
    _validate_worker_registry_auth(spec)
    return contract


def _validate_official_template_contract(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    binding = spec.get("worker_template_receipt", {})
    path = root / str(binding.get("path", ""))
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise LifecycleError("official worker template receipt is missing or changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    template = receipt.get("template", {})
    expected_hash = "a49dfc59f765f8c5b9c1e0c3a318a486"
    if (
        spec.get("worker_template_hash") != expected_hash
        or receipt.get("status") != "GREEN_READ_ONLY_CURRENT_OFFICIAL_TEMPLATE"
        or receipt.get("mutation") != "none_read_only"
        or template.get("hash_id") != expected_hash
        or template.get("name") != "NVIDIA CUDA"
        or template.get("recommended") is not True
        or template.get("cached") is not True
        or template.get("image") != "vastai/base-image"
        or template.get("tag") != "@vastai-automatic-tag"
        or template.get("runtype") != "jupyter"
        or template.get("use_ssh") is not True
        or template.get("ssh_direct") is not True
        or float(template.get("recommended_disk_space_gb", 0)) != 32.0
        or int(spec.get("h100_disk_gb", 0)) < 32
    ):
        raise LifecycleError("official NVIDIA CUDA template contract changed")
    return receipt


def _validate_live_control_evidence(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Require the exact cheap live proof before any paid worker may exist."""
    binding = spec.get("required_live_control_evidence", {})
    relative = Path(str(binding.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise LifecycleError("paid lifecycle spec lacks exact live control evidence")
    path = root / relative
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise LifecycleError("paid lifecycle spec lacks exact live control evidence")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "status": evidence.get("status") == "GREEN",
        "relay": evidence.get("relay_ssh") == "GREEN",
        "stop": evidence.get("endpoint_scoped_STOP", {}).get("status") == "STOPPED",
        "authority": evidence.get("scoped_key_authority", {}).get("status") == "GREEN",
        "volume_denied": evidence.get("scoped_key_authority", {}).get("volume_endpoint_denied") is True,
        "account_zero": evidence.get("final_account_audit", {}).get("active_instances") == 0,
        "no_h100": evidence.get("H100_created") is False,
        "no_gpu_work": evidence.get("gpu_work_executed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise LifecycleError("live control evidence is not GREEN: " + ", ".join(failed))
    return evidence


def validate_job_spec(root: Path, spec: dict[str, Any], job_spec_path: Path) -> dict[str, Any]:
    binding = spec.get("authorized_job_spec", {})
    bound_path = root / str(binding.get("path", ""))
    if bound_path.resolve() != job_spec_path.resolve():
        raise LifecycleError("job spec path is not authorized by lifecycle freeze")
    if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
        raise LifecycleError("authorized job spec is missing or changed")
    job_spec = json.loads(bound_path.read_text(encoding="utf-8"))
    if job_spec.get("schema_version") != 1 or job_spec.get("status") != "FROZEN":
        raise LifecycleError("job spec is not frozen")
    if job_spec.get("scope") != spec.get("scope"):
        raise LifecycleError("job scope is not authorized by lifecycle freeze")
    if spec.get("paid_launch_authorized") is True:
        authorization = spec.get("one_shot_authorization", {})
        run_id = str(job_spec.get("run_id", ""))
        expected_claim_name = run_id.replace("-", "_") + "_launch.json"
        claim_path = Path(str(authorization.get("claim_paths", {}).get("launch", "")))
        if authorization.get("id") != run_id or claim_path.name != expected_claim_name:
            raise LifecycleError("paid launch authorization id, run id, and claim path disagree")
    if spec.get("h100_research_authorized") is not True and job_spec.get("uses_gpu") is not False:
        bounded_runtime_smoke = (
            spec.get("scope") == "runtime_image_compatibility_probe"
            and spec.get("lifecycle_worker_kind") == "non_h100"
            and job_spec.get("gpu_work_class") == "single_tensor_capability_smoke"
            and job_spec.get("scientific_claim") is False
            and int(job_spec.get("model_reads", -1)) == 0
            and int(job_spec.get("data_reads", -1)) == 0
        )
        bounded_model_smoke = (
            spec.get("scope") == "pg1_model_forward_smoke_only"
            and spec.get("lifecycle_worker_kind") == "h100"
            and job_spec.get("gpu_work_class") == "single_model_forward_smoke"
            and job_spec.get("scientific_claim") is False
            and int(job_spec.get("synthetic_items", -1)) == 1
            and int(job_spec.get("discovery_items_read", -1)) == 0
            and int(job_spec.get("confirmation_items_read", -1)) == 0
        )
        if not bounded_runtime_smoke and not bounded_model_smoke:
            raise LifecycleError("non-research lifecycle scope cannot run a GPU workload")
    if job_spec.get("artifact_protocol") != "atomic_rename_no_background_writers":
        raise LifecycleError(
            "job must guarantee atomic artifact publication and no writers after foreground exit"
        )
    local_artifacts = job_spec.get("required_local_artifacts")
    if spec.get("paid_inspection_authorized") is True:
        if not isinstance(local_artifacts, list) or not local_artifacts:
            raise LifecycleError("paid inspection requires declared local artifacts")
    if local_artifacts is not None:
        if not isinstance(local_artifacts, list) or not local_artifacts:
            raise LifecycleError("required_local_artifacts must be a nonempty list")
        normalized = [_safe_local_artifact_path(value).as_posix() for value in local_artifacts]
        if len(normalized) != len(set(normalized)):
            raise LifecycleError("required_local_artifacts contains duplicates")
    if job_spec.get("scope") in {
        "lifecycle_live_rehearsal_only", "h100_stock_ssh_control_only",
        "h100_official_template_control_only", "managed_template_runtime_override_probe",
        "h100_managed_template_runtime_override_probe"
    }:
        expected = job_spec.get("expected_artifacts")
        if not isinstance(expected, dict) or not expected:
            raise LifecycleError("lifecycle rehearsal must freeze expected artifact bytes")
        for relative, binding in expected.items():
            path = Path(str(relative))
            digest = str(binding.get("sha256", "")) if isinstance(binding, dict) else ""
            size = int(binding.get("bytes", -1)) if isinstance(binding, dict) else -1
            if path.is_absolute() or ".." in path.parts or size < 0:
                raise LifecycleError("lifecycle rehearsal has an unsafe expected artifact")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise LifecycleError("lifecycle rehearsal has an invalid expected artifact hash")
        sentinels = set(job_spec.get("runtime", {}).get("required_sentinels", []))
        if not sentinels or not sentinels.issubset(expected):
            raise LifecycleError("every lifecycle sentinel must have frozen expected bytes")
        if job_spec.get("scope") == "h100_stock_ssh_control_only":
            runtime_job = job_spec.get("runtime", {}).get("job", {})
            environment = runtime_job.get("environment", {})
            total_expected_bytes = sum(
                int(binding.get("bytes", -1)) for binding in expected.values()
            )
            if (
                runtime_job.get("execution_site") != "h100"
                or environment.get("CUDA_VISIBLE_DEVICES") != ""
                or job_spec.get("uses_gpu") is not False
                or job_spec.get("scientific_claim") is not False
                or total_expected_bytes != 2_752_643
            ):
                raise LifecycleError("H100 stock control job boundary changed")
        if job_spec.get("scope") == "h100_official_template_control_only":
            runtime_job = job_spec.get("runtime", {}).get("job", {})
            environment = runtime_job.get("environment", {})
            total_expected_bytes = sum(
                int(binding.get("bytes", -1)) for binding in expected.values()
            )
            if (
                runtime_job.get("execution_site") != "h100"
                or environment.get("CUDA_VISIBLE_DEVICES") != ""
                or job_spec.get("uses_gpu") is not False
                or job_spec.get("scientific_claim") is not False
                or total_expected_bytes != 2_752_646
            ):
                raise LifecycleError("official H100 template control job boundary changed")
        if job_spec.get("scope") == "managed_template_runtime_override_probe":
            runtime_job = job_spec.get("runtime", {}).get("job", {})
            environment = runtime_job.get("environment", {})
            total_expected_bytes = sum(
                int(binding.get("bytes", -1)) for binding in expected.values()
            )
            if (
                runtime_job.get("execution_site") != "h100"
                or environment.get("CUDA_VISIBLE_DEVICES") != ""
                or job_spec.get("uses_gpu") is not False
                or job_spec.get("scientific_claim") is not False
                or total_expected_bytes != 2_752_649
            ):
                raise LifecycleError("managed runtime override job boundary changed")
        if job_spec.get("scope") == "h100_managed_template_runtime_override_probe":
            runtime_job = job_spec.get("runtime", {}).get("job", {})
            environment = runtime_job.get("environment", {})
            total_expected_bytes = sum(
                int(binding.get("bytes", -1)) for binding in expected.values()
            )
            if (
                runtime_job.get("execution_site") != "h100"
                or environment.get("CUDA_VISIBLE_DEVICES") != ""
                or job_spec.get("uses_gpu") is not False
                or job_spec.get("scientific_claim") is not False
                or total_expected_bytes != 2_752_654
            ):
                raise LifecycleError("H100 managed runtime override job boundary changed")
    if job_spec.get("scope") == "pg1_input_staging_rehearsal_only":
        try:
            validate_staging_job_spec(job_spec)
        except (StagingError, ValueError, TypeError, KeyError) as error:
            raise LifecycleError(f"input staging job is invalid: {error}") from error
        bindings = job_spec.get("source_bindings")
        if not isinstance(bindings, list) or len(bindings) < 5:
            raise LifecycleError("input staging source bindings are incomplete")
        for binding in bindings:
            path = root / str(binding.get("path", ""))
            if not path.is_file() or sha256_file(path) != binding.get("sha256"):
                raise LifecycleError("input staging source binding is missing or changed")
        if spec.get("paid_inspection_authorized") is True and (
            local_artifacts != ["input-staging/result.json"]
            or job_spec.get("expected_artifacts") is not None
        ):
            raise LifecycleError(
                "input staging inspection must retrieve only its content-addressed result"
            )
    if job_spec.get("scope") == "s3_worker_download_probe_only":
        try:
            validate_download_probe_spec(job_spec)
        except (DownloadProbeError, ValueError, TypeError, KeyError) as error:
            raise LifecycleError(f"S3 download probe job is invalid: {error}") from error
        bindings = job_spec.get("source_bindings")
        if not isinstance(bindings, list) or len(bindings) != 3:
            raise LifecycleError("S3 download probe source bindings are incomplete")
        for source_binding in bindings:
            path = root / str(source_binding.get("path", ""))
            if not path.is_file() or sha256_file(path) != source_binding.get("sha256"):
                raise LifecycleError("S3 download probe source binding is missing or changed")
        mirror = json.loads((root / bindings[1]["path"]).read_text(encoding="utf-8"))
        head = json.loads((root / bindings[2]["path"]).read_text(encoding="utf-8"))
        source = job_spec["source"]
        if (
            mirror.get("status") != "GREEN"
            or mirror.get("final_account_audit", {}).get("active_instances") != 0
            or mirror.get("url_persisted") is not False
            or mirror.get("aws_credentials_transmitted") is not False
            or head.get("status") != "GREEN"
            or head.get("bucket") != source["bucket"]
            or head.get("key") != source["key"]
            or head.get("version_id") != source["version_id"]
            or int(head.get("content_length", -1)) != int(source["bytes"])
            or head.get("metadata_sha256") != source["sha256"]
            or head.get("checksum_sha256_base64") != source["checksum_sha256_base64"]
        ):
            raise LifecycleError("S3 download probe source is not exact and GREEN")
        if spec.get("paid_inspection_authorized") is True and (
            local_artifacts != ["download-probe/result.json"]
            or job_spec.get("expected_artifacts") is not None
        ):
            raise LifecycleError(
                "S3 download inspection must retrieve only its content-addressed result"
            )
    if job_spec.get("scope") == "s3_worker_cache_hydration_only":
        try:
            validate_hydration_spec(job_spec)
        except (HydrationError, ValueError, TypeError, KeyError) as error:
            raise LifecycleError(f"S3 hydration job is invalid: {error}") from error
        bindings = job_spec.get("source_bindings")
        if not isinstance(bindings, list) or len(bindings) != 3:
            raise LifecycleError("S3 hydration source bindings are incomplete")
        for source_binding in bindings:
            path = root / str(source_binding.get("path", ""))
            if not path.is_file() or sha256_file(path) != source_binding.get("sha256"):
                raise LifecycleError("S3 hydration source binding is missing or changed")
        adjudication = json.loads((root / bindings[1]["path"]).read_text(encoding="utf-8"))
        manifest = json.loads((root / bindings[2]["path"]).read_text(encoding="utf-8"))
        versions = {str(row.get("path")): row for row in manifest.get("files", [])}
        if (
            adjudication.get("status") != "GREEN_FULL_PG1_S3_MIRROR"
            or adjudication.get("decision") != "PASS"
            or manifest.get("status") != "GREEN"
            or int(manifest.get("file_count", -1)) != 12
            or int(manifest.get("total_bytes", -1)) != 11_715_866_017
            or manifest.get("errors") != []
        ):
            raise LifecycleError("S3 hydration mirror evidence is not GREEN")
        for row in job_spec["files"]:
            source = versions.get(row["path"])
            if (
                source is None
                or source.get("status") != "GREEN"
                or int(source.get("bytes", -1)) != int(row["bytes"])
                or source.get("sha256") != row["sha256"]
                or source.get("key") != row["key"]
                or source.get("version_id") != row["version_id"]
            ):
                raise LifecycleError("S3 hydration file is not an exact mirrored version")
        if spec.get("paid_inspection_authorized") is True and (
            local_artifacts != ["cache-hydration/result.json"]
            or job_spec.get("expected_artifacts") is not None
        ):
            raise LifecycleError(
                "S3 hydration inspection must retrieve only its content-addressed result"
            )
    if job_spec.get("scope") == "pg1_model_forward_smoke_only":
        try:
            validate_model_smoke_spec(job_spec)
        except (ModelSmokeError, HydrationError, ValueError, TypeError, KeyError) as error:
            raise LifecycleError(f"PG1 model smoke job is invalid: {error}") from error
        bindings = job_spec.get("source_bindings", [])
        if not isinstance(bindings, list) or len(bindings) != 3:
            raise LifecycleError("PG1 model smoke source bindings are incomplete")
        for binding in bindings:
            path = root / str(binding.get("path", ""))
            if not path.is_file() or sha256_file(path) != binding.get("sha256"):
                raise LifecycleError("PG1 model smoke source binding is missing or changed")
    return job_spec


def _expected_manifest_errors(
    manifest_path: Path,
    expected: dict[str, Any] | None,
    *,
    content_addressed_result_path: str | None = None,
) -> list[str]:
    if not manifest_path.is_file():
        return ["manifest unavailable for expected-byte comparison"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {
        str(row.get("path")): {
            "bytes": int(row.get("bytes", -1)),
            "sha256": str(row.get("sha256", "")),
        }
        for row in payload.get("entries", [])
        if isinstance(row, dict)
    }
    if expected is None:
        dynamic_result = actual.get(content_addressed_result_path or "")
        if (
            content_addressed_result_path is not None
            and set(actual) == {content_addressed_result_path}
            and dynamic_result is not None
            and dynamic_result["bytes"] >= 0
            and re.fullmatch(r"[0-9a-f]{64}", dynamic_result["sha256"])
        ):
            return []
        return ["exact content-addressed result binding is absent"]
    errors = [
        f"expected artifact mismatch: {relative}"
        for relative, binding in expected.items()
        if actual.get(relative) != binding
    ]
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        errors.append("unexpected artifacts: " + ", ".join(unexpected))
    return errors


def _safe_local_artifact_path(value: object) -> Path:
    """Return a conservative manifest-relative path safe for scp and local materialization."""
    text = str(value)
    path = Path(text)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
    if (
        not text
        or text.startswith("/")
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or any(character not in allowed for character in text)
    ):
        raise LifecycleError(f"unsafe local artifact path: {text!r}")
    return path


def _local_artifact_retrieval_plan(
    *,
    manifest_path: Path,
    required_artifacts: list[str],
    maximum_file_bytes: int,
    maximum_total_bytes: int,
) -> dict[str, Any]:
    """Select bounded manifest entries, prioritizing explicitly required local results."""
    if not 1 <= maximum_file_bytes <= 50_000_000:
        raise LifecycleError("local artifact per-file limit must be in [1, 50000000]")
    if not maximum_file_bytes <= maximum_total_bytes <= 100_000_000:
        raise LifecycleError("local artifact total limit must cover one file and be <= 100000000")
    if not manifest_path.is_file():
        raise LifecycleError("durable manifest unavailable for local artifact retrieval")
    if not isinstance(required_artifacts, list) or not required_artifacts:
        raise LifecycleError("inspection requires at least one declared local artifact")
    required = [str(_safe_local_artifact_path(value).as_posix()) for value in required_artifacts]
    if len(required) != len(set(required)):
        raise LifecycleError("required local artifacts contain duplicates")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LifecycleError("durable manifest has no entries for local retrieval")
    entries: dict[str, dict[str, Any]] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise LifecycleError("durable manifest contains a malformed entry")
        relative = _safe_local_artifact_path(raw.get("path")).as_posix()
        if relative in entries:
            raise LifecycleError(f"durable manifest contains duplicate path: {relative}")
        try:
            size = int(raw.get("bytes", -1))
        except (TypeError, ValueError) as error:
            raise LifecycleError(f"invalid durable artifact size: {relative}") from error
        digest = str(raw.get("sha256", ""))
        if size < 0 or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise LifecycleError(f"invalid durable artifact binding: {relative}")
        entries[relative] = {"path": relative, "bytes": size, "sha256": digest}

    missing = sorted(set(required) - set(entries))
    if missing:
        raise LifecycleError("required local artifacts absent from durable manifest: " + ", ".join(missing))
    oversized_required = sorted(
        relative for relative in required if entries[relative]["bytes"] > maximum_file_bytes
    )
    if oversized_required:
        raise LifecycleError(
            "required local artifacts exceed the per-file limit: " + ", ".join(oversized_required)
        )
    required_bytes = sum(entries[relative]["bytes"] for relative in required)
    if required_bytes > maximum_total_bytes:
        raise LifecycleError("required local artifacts exceed the total retrieval limit")

    selected = [entries[relative] for relative in required]
    selected_set = set(required)
    skipped: list[dict[str, Any]] = []
    total_bytes = required_bytes
    for relative in sorted(set(entries) - selected_set):
        row = entries[relative]
        if row["bytes"] > maximum_file_bytes:
            skipped.append({"path": relative, "bytes": row["bytes"], "reason": "PER_FILE_LIMIT"})
        elif total_bytes + row["bytes"] > maximum_total_bytes:
            skipped.append({"path": relative, "bytes": row["bytes"], "reason": "TOTAL_LIMIT"})
        else:
            selected.append(row)
            total_bytes += row["bytes"]
    return {
        "required_artifacts": required,
        "selected": selected,
        "skipped": skipped,
        "selected_bytes": total_bytes,
        "maximum_file_bytes": maximum_file_bytes,
        "maximum_total_bytes": maximum_total_bytes,
    }


def _retrieve_local_artifacts(
    *,
    plan: dict[str, Any],
    scp: list[str],
    host: str,
    volume_run_root: str,
    output_dir: Path,
    copy_timeout_seconds: float,
) -> dict[str, Any]:
    """Atomically retrieve and verify the bounded artifact plan from the durable volume."""
    retrieved: list[dict[str, Any]] = []
    errors: list[str] = []
    local_root = output_dir / "artifacts"
    for row in plan["selected"]:
        relative = _safe_local_artifact_path(row["path"])
        destination = local_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        try:
            completed = _run(
                [
                    *scp,
                    f"root@{host}:{volume_run_root}/artifacts/{relative.as_posix()}",
                    str(temporary),
                ],
                check=False,
                timeout_seconds=copy_timeout_seconds,
            )
            if completed.returncode != 0:
                errors.append(f"copy failed: {relative.as_posix()} returncode={completed.returncode}")
                continue
            actual_bytes = temporary.stat().st_size if temporary.is_file() else -1
            if actual_bytes != row["bytes"]:
                errors.append(
                    f"size mismatch: {relative.as_posix()} expected={row['bytes']} actual={actual_bytes}"
                )
                continue
            actual_sha256 = sha256_file(temporary)
            if actual_sha256 != row["sha256"]:
                errors.append(f"sha256 mismatch: {relative.as_posix()}")
                continue
            os.replace(temporary, destination)
            retrieved.append(dict(row))
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"copy error: {relative.as_posix()} type={type(error).__name__}")
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    retrieved_paths = {row["path"] for row in retrieved}
    required_missing = sorted(set(plan["required_artifacts"]) - retrieved_paths)
    return {
        "status": "GREEN" if not errors and not required_missing else "RED",
        "retrieved": retrieved,
        "retrieved_bytes": sum(row["bytes"] for row in retrieved),
        "required_missing": required_missing,
        "errors": errors,
        "skipped": plan["skipped"],
        "maximum_file_bytes": plan["maximum_file_bytes"],
        "maximum_total_bytes": plan["maximum_total_bytes"],
    }


def require_paid_authorization(spec: dict[str, Any], *, mode: str) -> None:
    field = {
        "launch": "paid_launch_authorized",
        "inspect": "paid_inspection_authorized",
        "control": "paid_control_probe_authorized",
        "audit": "paid_cache_audit_authorized",
        "mirror": "paid_object_mirror_authorized",
        "task": "paid_task_materialization_authorized",
    }.get(mode)
    if field is None:
        raise ValueError(mode)
    if spec.get(field) is not True:
        raise LifecycleError(f"paid Vast {mode} is revoked by the active lifecycle freeze")


def _validate_paid_target(spec: dict[str, Any], *, volume_id: int) -> None:
    if int(spec.get("authorized_volume_id", 0)) != volume_id:
        raise LifecycleError("paid Vast action targets a volume not authorized by the frozen spec")


def _enforce_market_budget(spec: dict[str, Any], market: dict[str, Any]) -> None:
    worker = float(market["selected_worker"].get("dph_total", float("inf")))
    gateway = float(market["selected_gateway"].get("dph_total", float("inf")))
    worker_ceiling = float(spec.get("worker_price_ceiling_dph", 0))
    gateway_ceiling = float(spec.get("gateway_price_ceiling_dph", 0))
    if worker_ceiling <= 0 or worker > worker_ceiling:
        raise LifecycleError(f"selected worker price {worker:g}/h exceeds frozen ceiling {worker_ceiling:g}/h")
    if gateway_ceiling <= 0 or gateway > gateway_ceiling:
        raise LifecycleError(f"selected gateway price {gateway:g}/h exceeds frozen ceiling {gateway_ceiling:g}/h")


def _enforce_gateway_budget(spec: dict[str, Any], market: dict[str, Any]) -> None:
    gateway = float(market["selected_gateway"].get("dph_total", float("inf")))
    gateway_ceiling = float(spec.get("gateway_price_ceiling_dph", 0))
    if gateway_ceiling <= 0 or gateway > gateway_ceiling:
        raise LifecycleError(f"selected gateway price {gateway:g}/h exceeds frozen ceiling {gateway_ceiling:g}/h")


def _wait_account_zero(*, timeout_seconds: float = 30.0) -> dict[str, Any]:
    """Wait for Vast's control plane to reflect cleanup; stopped time is not GPU time."""
    deadline = time.monotonic() + timeout_seconds
    last_rows: Any = None
    while True:
        last_rows = _vast("show", "instances", check=False)
        if isinstance(last_rows, list) and not last_rows:
            return {"active_instances": 0, "instance_ids": [], "query_succeeded": True}
        if time.monotonic() >= deadline:
            break
        time.sleep(1)
    active = last_rows if isinstance(last_rows, list) else []
    return {
        "active_instances": len(active),
        "instance_ids": sorted(
            int(row["id"])
            for row in active
            if isinstance(row, dict) and str(row.get("id", "")).isdigit()
        ),
        "query_succeeded": isinstance(last_rows, list),
    }


def _wait_instance_stopped(instance_id: int, *, timeout_seconds: float = 180.0) -> dict[str, Any]:
    """Observe the provider-side guard; this function never performs STOP itself."""
    deadline = time.monotonic() + timeout_seconds
    observations = 0
    last_status: str | None = None
    while True:
        row = _show_instance(instance_id)
        observations += 1
        if isinstance(row, dict):
            last_status = str(row.get("actual_status") or row.get("cur_state") or "")
            intended = str(row.get("intended_status") or "")
            if last_status == "stopped" or intended == "stopped":
                return {
                    "status": "STOPPED",
                    "operator": "provider_observed_stop",
                    "observations": observations,
                }
        if time.monotonic() >= deadline:
            raise LifecycleError(f"provider-side STOP was not observed; last status={last_status}")
        time.sleep(2)


def _claim_one_shot_authorization(*, root: Path, spec_path: Path, spec: dict[str, Any], mode: str) -> Path:
    """Consume a frozen paid authorization exactly once, before any market mutation."""
    authorization = spec.get("one_shot_authorization", {})
    relative = Path(str(authorization.get("claim_paths", {}).get(mode, "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise LifecycleError(f"unsafe or missing one-shot {mode} claim path")
    claim_path = (root / relative).resolve()
    try:
        claim_path.relative_to(root.resolve())
    except ValueError as error:
        raise LifecycleError("one-shot claim path escapes the repository") from error
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "CONSUMED_BEFORE_MARKET_MUTATION",
        "authorization_id": str(authorization["id"]),
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lifecycle_spec": str(spec_path),
        "lifecycle_spec_sha256": sha256_file(spec_path),
        "pid": os.getpid(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise LifecycleError(f"one-shot paid Vast {mode} authorization was already consumed") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return claim_path


def launch(
    *,
    root: Path,
    spec_path: Path,
    job_spec_path: Path,
    volume_id: int,
    output: Path,
) -> dict[str, Any]:
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="launch")
    job_spec = validate_job_spec(root, spec, job_spec_path)
    _validate_paid_target(spec, volume_id=volume_id)
    authorization_claim = _claim_one_shot_authorization(root=root, spec_path=spec_path, spec=spec, mode="launch")
    run_id = str(job_spec["run_id"])
    try:
        market = select_live_offers(spec=spec, volume_id=volume_id)
        _enforce_market_budget(spec, market)
    except Exception as error:
        atomic_json(output, {
            "schema_version": 1,
            "status": "RED_PRECREATE_ACCOUNT_ZERO",
            "run_id": run_id,
            "stage": "LIVE_MARKET_SELECTION_AND_BUDGET",
            "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            "authorization_claim": str(authorization_claim),
            "authorization_claim_sha256": sha256_file(authorization_claim),
            "created_instance_ids": [],
            "gpu_work_released": False,
            "automatic_retry": False,
            "final_account_audit": _wait_account_zero(),
        })
        raise
    market_path = output.with_name(output.stem + ".market.json")
    atomic_json(market_path, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    h100_offer_id = int(market["selected_worker"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    relay_key = _validated_registered_ssh_key(spec, role="relay")
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    created: list[int] = []
    failure_packet: dict[str, Any] | None = None
    launch_stage = "VALIDATED_BEFORE_INSTANCE_CREATE"
    with tempfile.TemporaryDirectory(prefix="parallax-guarded-launch-") as temporary:
        temporary_path = Path(temporary)
        leases: list[BootstrapLease] = []
        local_reaper: LocalBootstrapReaper | None = None
        local_reaper_handed_off = False
        local_reaper_receipt: dict[str, Any] | None = None
        try:
            policy = spec["policy"]
            provider_loading_seconds = int(policy["maximum_provider_loading_seconds"])
            worker_hard_stop_seconds = int(policy["maximum_worker_hard_stop_seconds"])
            gateway_ttl = (
                int(policy["bootstrap_timeout_seconds"])
                + provider_loading_seconds
                + worker_hard_stop_seconds
                + 300
            )
            gateway_image = str(spec.get("gateway_image", spec["image"]))
            worker_image = str(spec.get("worker_image", spec["image"]))
            gateway_arguments = [
                "--image", gateway_image, "--disk", str(spec["gateway_disk_gb"]), "--ssh", "--direct",
                "--cancel-unavail", "--link-volume", str(volume_id), "--mount-path", str(spec["volume_mount"]),
                "--label", f"parallax-guard-{run_id}", "--onstart-cmd", _persist_instance_label_onstart(),
            ]
            worker_template_hash = str(spec.get("worker_template_hash", ""))
            h100_arguments = _worker_create_arguments(spec, run_id=run_id)
            worker_registry_login = _ephemeral_worker_registry_login(spec)
            if worker_registry_login is not None:
                h100_arguments.extend(["--login", worker_registry_login])

            def create_and_wait(
                offer_id: int, arguments: list[str], *, timeout_seconds: int
            ) -> tuple[dict[str, Any], dict[str, Any], list[str], Path]:
                response = _create_instance(offer_id, arguments)
                instance_id = int(response["new_contract"])
                created.append(instance_id)
                row = wait_ssh(
                    instance_id,
                    bootstrap_key,
                    timeout_seconds=timeout_seconds,
                    health_check=lambda: [lease.assert_healthy() for lease in leases],
                )
                control_path = _short_control_path(
                    role="bootstrap", instance_id=instance_id
                )
                ssh = _ssh_base(row, bootstrap_key, control_path=control_path)
                leases.append(BootstrapLease(ssh))
                return response, row, ssh, control_path

            # Arm the only pre-SSH cleanup layer before the first paid create
            # request.  It owns only this run's exact gateway label.
            local_reaper = LocalBootstrapReaper(
                temporary_path,
                repo_root=root,
                run_id=run_id,
                lifecycle_key_path=lifecycle_key_path,
                maximum_seconds=int(policy["bootstrap_timeout_seconds"]) + 60,
            )

            # No paid worker may exist until the independent cheap guardian is
            # running from the persistent volume with the scoped lifecycle key.
            launch_stage = "CREATING_AND_ARMING_GATEWAY"
            gateway_create, gateway_row, gateway_ssh, gateway_control = create_and_wait(
                gateway_offer_id,
                gateway_arguments,
                timeout_seconds=int(policy["bootstrap_timeout_seconds"]),
            )
            gateway_id = int(gateway_create["new_contract"])
            remote_bootstrap = f"/tmp/parallax-lifecycle-{run_id}"
            volume_run_root = f"{spec['volume_mount']}/parallax/lifecycle-runs/{run_id}"
            _run([*gateway_ssh, f"mkdir -p {shlex_quote(remote_bootstrap)} /opt/parallax-lifecycle"])
            gateway_scp = _scp_base(
                gateway_row, bootstrap_key, control_path=gateway_control
            )
            gateway_host, _gateway_port = _ssh_endpoint(gateway_row)
            _stage_lifecycle_sources_without_local_archive(
                root=root,
                ssh=gateway_ssh,
                scp=gateway_scp,
                host=gateway_host,
                remote_root="/opt/parallax-lifecycle",
            )
            lifecycle_secret = f"{remote_bootstrap}/scoped-lifecycle.key"
            _write_remote_secret(
                gateway_ssh, lifecycle_secret, lifecycle_key_path.read_text(encoding="utf-8").strip()
            )
            _run([
                *gateway_ssh,
                scoped_deadman_command(
                    instance_id=gateway_id,
                    key_path=lifecycle_secret,
                    seconds=gateway_ttl,
                    action="destroy",
                ),
            ])
            guardian_contract = {
                "schema_version": 1,
                "status": "FROZEN_GUARDIAN",
                "run_id": run_id,
                "worker_label": f"parallax-worker-{run_id}",
                "state_root": f"{volume_run_root}/lifecycle",
                "api_key_file": lifecycle_secret,
                "discovery_timeout_seconds": int(policy["bootstrap_timeout_seconds"]),
                "provider_loading_timeout_seconds": provider_loading_seconds,
                "billable_prearm_timeout_seconds": int(
                    policy.get("maximum_worker_prearm_seconds", 120)
                ),
                "heartbeat_timeout_seconds": int(policy["heartbeat_timeout_seconds"]),
                "hard_runtime_seconds": worker_hard_stop_seconds,
                "poll_seconds": float(policy.get("guardian_poll_seconds", 5.0)),
            }
            from .lifecycle_guardian import GuardianPolicy

            GuardianPolicy.from_contract(guardian_contract)
            guardian_path = temporary_path / "guardian-contract.json"
            atomic_json(guardian_path, guardian_contract)
            remote_guardian = f"{remote_bootstrap}/guardian-contract.json"
            _run([*gateway_scp, str(guardian_path), f"root@{gateway_host}:{remote_guardian}"])
            guardian_command = (
                f"mkdir -p {shlex_quote(volume_run_root + '/lifecycle')}; "
                f"nohup env PYTHONPATH=/opt/parallax-lifecycle python3 -m infra.vast.parallax.lifecycle_guardian "
                f"--contract {shlex_quote(remote_guardian)} "
                f">{shlex_quote(volume_run_root + '/lifecycle/guardian.log')} 2>&1 </dev/null &"
            )
            _run([*gateway_ssh, guardian_command])
            guardian_ready = False
            guardian_ready_command = _semantic_lifecycle_ready_command(
                state_root=volume_run_root + "/lifecycle",
                run_id=run_id,
                kind="guardian",
                maximum_age_seconds=int(policy["heartbeat_timeout_seconds"]),
            )
            for _ in range(30):
                check = _run([*gateway_ssh, guardian_ready_command], check=False)
                if check.returncode == 0:
                    guardian_ready = True
                    break
                time.sleep(1)
            if not guardian_ready:
                raise LifecycleError("independent gateway guardian did not arm before H100 creation")
            local_reaper_receipt = local_reaper.handoff(run_id=run_id)
            local_reaper_handed_off = True

            launch_stage = "CREATING_AND_QUALIFYING_WORKER"
            h100_create, h100_row, h100_bootstrap_ssh, _h100_control = create_and_wait(
                h100_offer_id,
                h100_arguments,
                timeout_seconds=(
                    provider_loading_seconds
                    + int(policy.get("maximum_worker_prearm_seconds", 120))
                ),
            )
            h100_id = int(h100_create["new_contract"])
            if worker_template_hash:
                _run([*h100_bootstrap_ssh, _persist_instance_label_onstart()])
            launch_stage = "WORKER_IMAGE_LIVE_COMMAND_GATE"
            worker_capabilities = _require_worker_image_capabilities(h100_bootstrap_ssh)
            h100_local_secret = f"{remote_bootstrap}/h100-local-lifecycle.key"
            _write_remote_secret(
                h100_bootstrap_ssh,
                h100_local_secret,
                lifecycle_key_path.read_text(encoding="utf-8").strip(),
            )
            launch_stage = "ARMING_WORKER_SCOPED_DEADMEN"
            _run([
                *h100_bootstrap_ssh,
                scoped_deadman_command(
                    instance_id=h100_id,
                    key_path=h100_local_secret,
                    seconds=worker_hard_stop_seconds,
                    action="stop",
                ) + " " + scoped_deadman_command(
                    instance_id=h100_id,
                    key_path=h100_local_secret,
                    seconds=int(policy.get("maximum_worker_prearm_seconds", 120)),
                    action="stop",
                    armed_path="/tmp/parallax-lifecycle-armed",
                ),
            ])

            gateway_secret = lifecycle_secret
            h100_secret = lifecycle_secret
            if spec.get("scope") == "s3_worker_download_probe_only":
                credentials_command = _run(
                    [
                        "aws", "configure", "export-credentials", "--format", "process",
                        "--profile", "fuelos", "--region", "us-east-1",
                    ]
                )
                credentials = json.loads(credentials_command.stdout.decode("utf-8"))
                source = job_spec["source"]
                s3_download_url = presign_get(
                    access_key=str(credentials.get("AccessKeyId", "")),
                    secret_key=str(credentials.get("SecretAccessKey", "")),
                    session_token=credentials.get("SessionToken"),
                    bucket=str(source["bucket"]),
                    key=str(source["key"]),
                    version_id=str(source["version_id"]),
                    region="us-east-1",
                    now=datetime.now(timezone.utc),
                    expires_seconds=900,
                )
                del credentials
                validate_download_url(s3_download_url, job_spec)
                remote_download_url = f"{remote_bootstrap}/s3-get-url.secret"
                _write_remote_secret(gateway_ssh, remote_download_url, s3_download_url)
                s3_download_url = ""
            if spec.get("scope") in {
                "s3_worker_cache_hydration_only", "pg1_model_forward_smoke_only"
            }:
                credentials_command = _run(
                    [
                        "aws", "configure", "export-credentials", "--format", "process",
                        "--profile", "fuelos", "--region", "us-east-1",
                    ]
                )
                credentials = json.loads(credentials_command.stdout.decode("utf-8"))
                hydration_urls = []
                for source in job_spec["files"]:
                    url = presign_get(
                        access_key=str(credentials.get("AccessKeyId", "")),
                        secret_key=str(credentials.get("SecretAccessKey", "")),
                        session_token=credentials.get("SessionToken"),
                        bucket=str(source["bucket"]), key=str(source["key"]),
                        version_id=str(source["version_id"]), region="us-east-1",
                        now=datetime.now(timezone.utc), expires_seconds=1800,
                    )
                    validate_hydration_url(url, source)
                    hydration_urls.append({"path": source["path"], "url": url})
                del credentials
                remote_urls = f"{remote_bootstrap}/s3-get-urls.secret.json"
                _write_remote_secret(gateway_ssh, remote_urls, json.dumps(hydration_urls))
                hydration_urls.clear()
            remote_ssh_key = f"{remote_bootstrap}/h100-ssh.key"
            _write_remote_secret(gateway_ssh, remote_ssh_key, relay_key.read_text(encoding="utf-8"))
            h100_host, h100_port = _ssh_endpoint(h100_row)
            launch_stage = "RENDERING_AND_VALIDATING_RUNTIME_CONTRACT"
            runtime_contract = _build_runtime_contract(
                spec=spec,
                job_spec=job_spec,
                run_id=run_id,
                worker_instance_id=h100_id,
                worker_host=h100_host,
                worker_port=h100_port,
                worker_ssh_key=remote_ssh_key,
                worker_api_key=h100_secret,
                gateway_instance_id=gateway_id,
                gateway_api_key=gateway_secret,
                volume_run_root=volume_run_root,
                frozen_spec_sha256=sha256_file(spec_path),
                job_spec_sha256=sha256_file(job_spec_path),
            )
            runtime_path = temporary_path / "runtime-contract.json"
            atomic_json(runtime_path, runtime_contract)
            remote_contract = f"{remote_bootstrap}/runtime-contract.json"
            _run([*gateway_scp, str(runtime_path), f"root@{gateway_host}:{remote_contract}"])
            supervisor_command = (
                f"mkdir -p {shlex_quote(volume_run_root + '/lifecycle')}; "
                f"nohup env PYTHONPATH=/opt/parallax-lifecycle python3 -m infra.vast.parallax.lifecycle_supervisor "
                f"supervise --contract {shlex_quote(remote_contract)} "
                f">{shlex_quote(volume_run_root + '/lifecycle/supervisor.log')} 2>&1 </dev/null &"
            )
            launch_stage = "ARMING_DETACHED_SUPERVISOR"
            _run([*gateway_ssh, supervisor_command])
            armed = False
            supervisor_ready_command = _semantic_lifecycle_ready_command(
                state_root=volume_run_root + "/lifecycle",
                run_id=run_id,
                kind="supervisor",
                maximum_age_seconds=int(policy["heartbeat_timeout_seconds"]),
            )
            for _ in range(int(policy.get("supervisor_arm_timeout_seconds", 30))):
                for lease in leases:
                    lease.assert_healthy()
                check = _run([*gateway_ssh, supervisor_ready_command], check=False)
                if check.returncode == 0:
                    armed = True
                    break
                time.sleep(1)
            if not armed:
                raise LifecycleError("detached lifecycle supervisor did not arm")
            launch_stage = "RELEASING_WORKER_EXECUTION"
            _run_idempotent_ssh(
                h100_bootstrap_ssh,
                "touch /tmp/parallax-lifecycle-armed",
            )
            _run_idempotent_ssh(
                gateway_ssh,
                "touch "
                + " ".join(
                    shlex_quote(path)
                    for path in (
                        "/tmp/parallax-lifecycle-armed",
                        volume_run_root + "/lifecycle/armed.marker",
                        volume_run_root + "/lifecycle/execution-release.marker",
                    )
                ),
            )
            for lease in leases:
                lease.close()
            leases.clear()
            packet = {
                "schema_version": 1,
                "status": "SUPERVISOR_RELEASED",
                "run_id": run_id,
                "gateway_instance_id": gateway_id,
                "h100_instance_id": h100_id,
                "bootstrap_ssh_public_key_id": int(spec["bootstrap_ssh_public_key_id"]),
                "relay_ssh_public_key_id": int(spec["relay_ssh_public_key_id"]),
                "scoped_lifecycle_api_key_id": lifecycle_key_id,
                "account_ssh_key_mutations": 0,
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "volume_run_root": volume_run_root,
                "runtime_contract_sha256": sha256_file(runtime_path),
                "market_receipt": str(market_path),
                "market_receipt_sha256": sha256_file(market_path),
                "gateway_hourly_price": market["selected_gateway"].get("dph_total"),
                "worker_hourly_price": market["selected_worker"].get("dph_total"),
                "worker_image": worker_image,
                "worker_capability_preflight": worker_capabilities,
                "secrets_logged": False,
                "local_controller_required_after_arm": False,
                "local_bootstrap_reaper": local_reaper_receipt,
            }
            atomic_json(output, packet)
            created.clear()  # Detached supervisor and TTL guards own teardown now.
            return packet
        except Exception as error:
            failure_packet = {
                "schema_version": 1,
                "status": "RED_LAUNCH_FAILED_CLEANUP_PENDING",
                "run_id": run_id,
                "stage": launch_stage,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error)[:1000],
                },
                "created_instance_ids": list(created),
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "market_receipt": str(market_path),
                "market_receipt_sha256": sha256_file(market_path),
                "worker_image": str(spec.get("worker_image", spec["image"])),
                "gpu_work_released": launch_stage == "RELEASING_WORKER_EXECUTION",
                "automatic_retry": False,
                "secrets_logged": False,
            }
            atomic_json(output, failure_packet)
            raise
        finally:
            for lease in leases:
                lease.close()
            for instance_id in reversed(created):
                # STOP is the billing boundary. Destroy is only local-disk
                # cleanup and must never consume teardown time on an active GPU.
                _vast("stop", "instance", str(instance_id), check=False)
                _vast("destroy", "instance", str(instance_id), check=False)
            if local_reaper is not None and not local_reaper_handed_off:
                local_reaper.release_to_reap()
            if failure_packet is not None:
                audit = _wait_account_zero()
                failure_packet["final_account_audit"] = audit
                failure_packet["status"] = (
                    "RED_LAUNCH_FAILED_ACCOUNT_ZERO"
                    if audit.get("query_succeeded") and audit.get("active_instances") == 0
                    else "RED_LAUNCH_FAILED_ACCOUNT_NOT_ZERO"
                )
                atomic_json(output, failure_packet)


def probe_gateway_controls(
    *,
    root: Path,
    spec_path: Path,
    volume_id: int,
    output: Path,
) -> dict[str, Any]:
    """Prove endpoint-scoped STOP control without renting an H100."""
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="control")
    _validate_paid_target(spec, volume_id=volume_id)
    authorization_claim = _claim_one_shot_authorization(
        root=root, spec_path=spec_path, spec=spec, mode="control"
    )
    market = select_live_offers(spec=spec, volume_id=volume_id)
    _enforce_gateway_budget(spec, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    relay_key = _validated_registered_ssh_key(spec, role="relay")
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    authority = _probe_scoped_key_authority(lifecycle_key_path)
    instance_id: int | None = None
    lease: BootstrapLease | None = None
    return_packet: dict[str, Any] | None = None
    market_path = output.with_name(output.stem + ".market.json")
    atomic_json(market_path, market)
    progress = {
        "schema_version": 1,
        "status": "RUNNING",
        "stage": "vast_gateway_control_probe",
        "instance_id": None,
        "gpu_work": False,
        "relay_ssh": "PENDING",
        "scoped_deadman_armed": "PENDING",
        "endpoint_scoped_STOP": "PENDING",
        "scoped_key_authority": authority,
        "scoped_lifecycle_api_key_id": lifecycle_key_id,
        "authorization_claim": str(authorization_claim),
        "account_destroy_requested_in_finally": True,
    }
    atomic_json(output, progress)
    try:
        policy = spec["policy"]
        stop_seconds = int(policy["scoped_stop_proof_seconds"])
        onstart = _persist_instance_label_onstart()
        created = _create_instance(
            gateway_offer_id,
            [
                "--image", str(spec["image"]), "--disk", str(spec["gateway_disk_gb"]),
                "--ssh", "--direct", "--cancel-unavail", "--link-volume", str(volume_id),
                "--mount-path", str(spec["volume_mount"]), "--label", "parallax-control-probe",
                "--onstart-cmd", onstart,
            ],
        )
        instance_id = int(created["new_contract"])
        progress.update({"instance_id": instance_id, "status": "BOOTSTRAPPING"})
        atomic_json(output, progress)
        row = wait_ssh(
            instance_id,
            bootstrap_key,
            timeout_seconds=int(policy["bootstrap_timeout_seconds"]),
        )
        bootstrap_ssh = _ssh_base(row, bootstrap_key)
        lease = BootstrapLease(bootstrap_ssh)
        remote_secret = "/tmp/parallax-scoped-stop-proof.key"
        _write_remote_secret(
            bootstrap_ssh,
            remote_secret,
            lifecycle_key_path.read_text(encoding="utf-8").strip(),
        )
        _run([*bootstrap_ssh, scoped_deadman_command(
            instance_id=instance_id,
            key_path=remote_secret,
            seconds=stop_seconds,
            action="stop",
        )])
        progress["scoped_deadman_armed"] = "GREEN"
        atomic_json(output, progress)
        relay = _run([*_ssh_base(row, relay_key), "true"], check=False)
        if relay.returncode != 0:
            raise LifecycleError("dedicated registered relay key was rejected by Vast SSH")
        progress["relay_ssh"] = "GREEN"
        atomic_json(output, progress)
        lease.close()
        lease = None
        stopped = _wait_instance_stopped(instance_id, timeout_seconds=stop_seconds + 60)
        if stopped.get("status") != "STOPPED":
            raise LifecycleError("endpoint-scoped key failed live STOP proof")
        packet = {**progress,
            "status": "GREEN",
            "endpoint_scoped_STOP": stopped,
            "stop_deadline_seconds": stop_seconds,
            "H100_created": False,
            "gpu_work_executed": False,
            "authorization_claim_sha256": sha256_file(authorization_claim),
            "market_receipt": str(market_path),
            "market_receipt_sha256": sha256_file(market_path),
            "gateway_dph_total": market["selected_gateway"].get("dph_total"),
            "account_destroy_requested_in_finally": True,
        }
        return_packet = packet
        atomic_json(output, packet)
        return packet
    except Exception as error:
        atomic_json(
            output,
            {**progress,
                "status": "RED",
                "instance_id": instance_id,
                "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            },
        )
        raise
    finally:
        if lease is not None:
            lease.close()
        if instance_id is not None:
            _vast("destroy", "instance", str(instance_id), check=False)
        audit = _wait_account_zero()
        persisted = json.loads(output.read_text(encoding="utf-8"))
        persisted["final_account_audit"] = audit
        if not audit["query_succeeded"] or audit["active_instances"] != 0:
            persisted["status"] = "RED_ACCOUNT_NOT_ZERO"
        if return_packet is not None:
            return_packet.update({
                "status": persisted["status"],
                "final_account_audit": audit,
            })
        atomic_json(output, persisted)
        if sys.exc_info()[0] is None and persisted["status"] != "GREEN":
            raise LifecycleError("gateway control probe cleanup did not return Vast account to zero")


def inspect_persisted_run(
    *,
    root: Path,
    spec_path: Path,
    volume_id: int,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Re-attach, verify, and retrieve a run under short provider guards."""
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="inspect")
    authorized_job = validate_job_spec(
        root,
        spec,
        root / str(spec.get("authorized_job_spec", {}).get("path", "")),
    )
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in run_id):
        raise LifecycleError("unsafe persisted run id")
    _validate_paid_target(spec, volume_id=volume_id)
    authorization_claim = _claim_one_shot_authorization(root=root, spec_path=spec_path, spec=spec, mode="inspect")
    market = select_live_offers(spec=spec, volume_id=volume_id)
    _enforce_gateway_budget(spec, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    instance_id: int | None = None
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "inspection-market.json", market)
    with tempfile.TemporaryDirectory(prefix="parallax-guarded-inspect-") as temporary:
        temporary_path = Path(temporary)
        lease: BootstrapLease | None = None
        reaper: LocalBootstrapReaper | None = None
        reaper_handed_off = False
        reaper_receipt: dict[str, Any] | None = None
        packet: dict[str, Any] | None = None
        try:
            reaper_run_id = f"inspect-{run_id}"
            reaper = LocalBootstrapReaper(
                temporary_path,
                repo_root=root,
                run_id=reaper_run_id,
                lifecycle_key_path=lifecycle_key_path,
                maximum_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]) + 60,
            )
            created = _create_instance(
                gateway_offer_id,
                [
                    "--image", str(spec["image"]), "--disk", str(spec["gateway_disk_gb"]), "--ssh", "--direct",
                    "--cancel-unavail", "--link-volume", str(volume_id), "--mount-path", str(spec["volume_mount"]),
                    "--label", f"parallax-guard-{reaper_run_id}", "--onstart-cmd", _persist_instance_label_onstart(),
                ],
            )
            instance_id = int(created["new_contract"])
            row = wait_ssh(
                instance_id,
                bootstrap_key,
                timeout_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]),
            )
            control_path = _short_control_path(
                role="inspection", instance_id=instance_id
            )
            ssh = _ssh_base(row, bootstrap_key, control_path=control_path)
            lease = BootstrapLease(ssh)
            remote_secret = f"/tmp/parallax-inspect-{run_id}/scoped-lifecycle.key"
            _write_remote_secret(
                ssh,
                remote_secret,
                lifecycle_key_path.read_text(encoding="utf-8").strip(),
            )
            inspection_deadman = int(spec["policy"]["inspection_deadman_seconds"])
            _run([
                *ssh,
                scoped_deadman_command(
                    instance_id=instance_id,
                    key_path=remote_secret,
                    seconds=inspection_deadman,
                    action="stop",
                )
                + " "
                + scoped_deadman_command(
                    instance_id=instance_id,
                    key_path=remote_secret,
                    seconds=inspection_deadman + 30,
                    action="destroy",
                ),
            ])
            reaper_receipt = reaper.handoff(
                run_id=reaper_run_id,
                target="remote_scoped_deadman",
            )
            reaper_handed_off = True
            scp = _scp_base(row, bootstrap_key, control_path=control_path)
            archive = temporary_path / "lifecycle.tar.gz"
            _normalized_lifecycle_archive(root, archive)
            _run([*ssh, "mkdir -p /opt/parallax-lifecycle"])
            host, _port = _ssh_endpoint(row)
            _run([*scp, str(archive), f"root@{host}:/tmp/parallax-lifecycle.tar.gz"])
            _run([*ssh, "tar -C /opt/parallax-lifecycle -xzf /tmp/parallax-lifecycle.tar.gz"])
            volume_run_root = f"{spec['volume_mount']}/parallax/lifecycle-runs/{run_id}"
            verify_command = _run([
                *ssh,
                f"PYTHONPATH=/opt/parallax-lifecycle python3 -m infra.vast.parallax.lifecycle_supervisor "
                f"verify-volume --run-root {shlex_quote(volume_run_root)}",
            ], check=False)
            atomic_json(
                output_dir / "verify-command.json",
                {
                    "returncode": verify_command.returncode,
                    "stdout": verify_command.stdout.decode("utf-8", errors="replace")[-4000:],
                    "stderr": verify_command.stderr.decode("utf-8", errors="replace")[-4000:],
                },
            )
            remote_files = (
                "lifecycle/receipt.json",
                "lifecycle/manifest.json",
                "lifecycle/manifest.previous.json",
                "lifecycle/upload-journal.jsonl",
                "lifecycle/reattach-verification.json",
                "lifecycle/guardian.log",
                "lifecycle/guardian-heartbeat.json",
                "lifecycle/guardian-discovery.json",
                "lifecycle/guardian-receipt.json",
                "lifecycle/launch-registry.json",
                "lifecycle/supervisor.log",
                "lifecycle/watchdog.log",
                "lifecycle/heartbeat.json",
                "lifecycle/relay-ssh-probe.json",
                "lifecycle/worker-stop-intent.json",
                "lifecycle/gateway-teardown-intent.json",
                "job.log",
            )
            copied_files = []
            copy_timeout_seconds = float(spec["policy"]["artifact_copy_timeout_seconds"])
            for relative in remote_files:
                exists = _run([*ssh, f"test -f {shlex_quote(volume_run_root + '/' + relative)}"], check=False)
                if exists.returncode == 0:
                    _run(
                        [
                            *scp,
                            f"root@{host}:{volume_run_root}/{relative}",
                            str(output_dir / Path(relative).name),
                        ],
                        timeout_seconds=copy_timeout_seconds,
                    )
                    copied_files.append(relative)
            verification_path = output_dir / "reattach-verification.json"
            verification = (
                json.loads(verification_path.read_text(encoding="utf-8"))
                if verification_path.is_file()
                else {"status": "RED", "errors": ["reattach verification command produced no packet"]}
            )
            manifest_file = output_dir / "manifest.json"
            expected_errors = _expected_manifest_errors(
                manifest_file,
                authorized_job.get("expected_artifacts"),
                content_addressed_result_path={
                    "pg1_input_staging_rehearsal_only": "input-staging/result.json",
                    "s3_worker_download_probe_only": "download-probe/result.json",
                    "s3_worker_cache_hydration_only": "cache-hydration/result.json",
                    "pg1_model_forward_smoke_only": "model-smoke/result.json",
                }.get(str(authorized_job.get("scope"))),
            )
            local_plan = _local_artifact_retrieval_plan(
                manifest_path=manifest_file,
                required_artifacts=authorized_job.get("required_local_artifacts", []),
                maximum_file_bytes=int(
                    spec["policy"]["maximum_local_inspection_file_bytes"]
                ),
                maximum_total_bytes=int(
                    spec["policy"]["maximum_local_inspection_artifact_bytes"]
                ),
            )
            local_retrieval = _retrieve_local_artifacts(
                plan=local_plan,
                scp=scp,
                host=host,
                volume_run_root=volume_run_root,
                output_dir=output_dir,
                copy_timeout_seconds=copy_timeout_seconds,
            )
            atomic_json(output_dir / "local-artifact-retrieval.json", local_retrieval)
            inspection_status = (
                "GREEN"
                if (
                    verification.get("status") == "GREEN"
                    and not expected_errors
                    and local_retrieval["status"] == "GREEN"
                )
                else "RED"
            )
            packet = {
                "schema_version": 1,
                "status": inspection_status,
                "run_id": run_id,
                "volume_id": volume_id,
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "gateway_instance_id": instance_id,
                "scoped_lifecycle_api_key_id": lifecycle_key_id,
                "local_bootstrap_reaper": reaper_receipt,
                "copied_files": copied_files,
                "verification_errors": verification.get("errors", []),
                "expected_artifact_errors": expected_errors,
                "expected_artifact_count": len(
                    authorized_job.get("expected_artifacts") or {}
                ),
                "local_artifact_retrieval": local_retrieval,
                "local_artifact_retrieval_sha256": sha256_file(
                    output_dir / "local-artifact-retrieval.json"
                ),
                "reattach_verification_sha256": sha256_file(verification_path) if verification_path.is_file() else None,
                "receipt_sha256": sha256_file(output_dir / "receipt.json") if (output_dir / "receipt.json").is_file() else None,
                "manifest_sha256": sha256_file(output_dir / "manifest.json") if (output_dir / "manifest.json").is_file() else None,
            }
            atomic_json(output_dir / "inspection.json", packet)
            return packet
        finally:
            if lease is not None:
                lease.close()
            if instance_id is not None:
                _vast("stop", "instance", str(instance_id), check=False)
                _vast("destroy", "instance", str(instance_id), check=False)
            if reaper is not None and not reaper_handed_off:
                reaper.release_to_reap()
            audit = _wait_account_zero()
            atomic_json(output_dir / "final-account-audit.json", audit)
            account_is_zero = bool(
                audit.get("query_succeeded")
                and audit.get("active_instances") == 0
            )
            if packet is not None:
                packet["final_account_audit"] = audit
                if not account_is_zero:
                    packet["status"] = "RED_ACCOUNT_NOT_ZERO"
                atomic_json(output_dir / "inspection.json", packet)
            if sys.exc_info()[0] is None and not account_is_zero:
                raise LifecycleError(
                    "volume inspection cleanup did not return Vast account to zero"
                )


def audit_persistent_cache(
    *,
    root: Path,
    spec_path: Path,
    audit_spec_path: Path,
    volume_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Hash one frozen cache subset on its volume; never create a worker."""
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="audit")
    audit_spec = _validate_bound_cache_audit_spec(root, spec, audit_spec_path)
    _validate_paid_target(spec, volume_id=volume_id)
    authorization_claim = _claim_one_shot_authorization(
        root=root, spec_path=spec_path, spec=spec, mode="audit"
    )
    market = select_live_gateway_offer(spec=spec, volume_id=volume_id)
    _enforce_gateway_budget(spec, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "market.json", market)
    instance_id: int | None = None
    packet: dict[str, Any] | None = None
    lease: BootstrapLease | None = None
    reaper: LocalBootstrapReaper | None = None
    reaper_handed_off = False
    with tempfile.TemporaryDirectory(prefix="parallax-cache-audit-") as temporary:
        temporary_path = Path(temporary)
        try:
            run_id = str(audit_spec["audit_id"])
            reaper = LocalBootstrapReaper(
                temporary_path,
                repo_root=root,
                run_id=run_id,
                lifecycle_key_path=lifecycle_key_path,
                maximum_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]) + 60,
            )
            created = _create_instance(
                gateway_offer_id,
                [
                    "--image", str(spec["image"]), "--disk", str(spec["gateway_disk_gb"]),
                    "--ssh", "--direct", "--cancel-unavail", "--link-volume", str(volume_id),
                    "--mount-path", str(spec["volume_mount"]), "--label", f"parallax-guard-{run_id}",
                    "--onstart-cmd", _persist_instance_label_onstart(),
                ],
            )
            instance_id = int(created["new_contract"])
            row = wait_ssh(
                instance_id,
                bootstrap_key,
                timeout_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]),
            )
            control_path = _short_control_path(role="cache-audit", instance_id=instance_id)
            ssh = _ssh_base(row, bootstrap_key, control_path=control_path)
            scp = _scp_base(row, bootstrap_key, control_path=control_path)
            lease = BootstrapLease(ssh)
            remote_root = f"/tmp/parallax-cache-audit-{run_id}"
            remote_secret = f"{remote_root}/scoped-lifecycle.key"
            _run([*ssh, f"umask 077; mkdir -p {shlex_quote(remote_root)}"])
            _write_remote_secret(
                ssh,
                remote_secret,
                lifecycle_key_path.read_text(encoding="utf-8").strip(),
            )
            deadman = int(spec["policy"]["cache_audit_deadman_seconds"])
            _run(
                [
                    *ssh,
                    scoped_deadman_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        seconds=deadman,
                        action="stop",
                    )
                    + " "
                    + scoped_deadman_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        seconds=deadman + 30,
                        action="destroy",
                    ),
                ]
            )
            reaper_receipt = reaper.handoff(run_id=run_id, target="remote_scoped_deadman")
            reaper_handed_off = True
            archive = temporary_path / "lifecycle.tar.gz"
            _normalized_lifecycle_archive(root, archive)
            host, _port = _ssh_endpoint(row)
            _run([*ssh, "mkdir -p /opt/parallax-lifecycle"])
            _run([*scp, str(archive), f"root@{host}:/tmp/parallax-lifecycle.tar.gz"])
            _run([*ssh, "tar -C /opt/parallax-lifecycle -xzf /tmp/parallax-lifecycle.tar.gz"])
            _run([*scp, str(audit_spec_path), f"root@{host}:{remote_root}/audit-spec.json"])
            durable_root = f"{spec['volume_mount']}/parallax/cache-audits/{run_id}"
            remote_result = f"{durable_root}/audit-result.json"
            _run(
                [
                    *ssh,
                    f"mkdir -p {shlex_quote(durable_root)}; test ! -e {shlex_quote(remote_result)}",
                ]
            )
            _run(
                [
                    *ssh,
                    scoped_result_stop_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        result_path=remote_result,
                    ),
                ]
            )
            completed = _run(
                [
                    *ssh,
                    f"PYTHONPATH=/opt/parallax-lifecycle python3 -m infra.vast.parallax.cache_audit "
                    f"--spec {shlex_quote(remote_root + '/audit-spec.json')} "
                    f"--output {shlex_quote(remote_result)}",
                ],
                check=False,
                timeout_seconds=max(30, deadman - 20),
            )
            atomic_json(
                output_dir / "command.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.decode("utf-8", errors="replace")[-4000:],
                    "stderr": completed.stderr.decode("utf-8", errors="replace")[-4000:],
                },
            )
            local_temporary = output_dir / ".audit-result.json.partial"
            copied = _run(
                [*scp, f"root@{host}:{remote_result}", str(local_temporary)],
                check=False,
                timeout_seconds=30,
            )
            if copied.returncode != 0 or not local_temporary.is_file():
                raise LifecycleError("cache audit produced no retrievable result")
            if local_temporary.stat().st_size > 2_000_000:
                raise LifecycleError("cache audit result exceeded its local byte ceiling")
            result_path = output_dir / "audit-result.json"
            os.replace(local_temporary, result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = "GREEN" if completed.returncode == 0 and result.get("status") == "GREEN" else "RED"
            packet = {
                "schema_version": 1,
                "status": status,
                "audit_id": run_id,
                "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "gateway_dph_total": market["selected_gateway"].get("dph_total"),
                "worker_created": False,
                "gpu_work": False,
                "model_loaded": False,
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "scoped_lifecycle_api_key_id": lifecycle_key_id,
                "local_bootstrap_reaper": reaper_receipt,
                "audit_result": str(result_path),
                "audit_result_sha256": sha256_file(result_path),
                "verified_bytes": result.get("verified_bytes", 0),
                "file_count": result.get("file_count", 0),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "stop_before_destroy_requested": True,
                "result_triggered_stop_armed": True,
                "durable_volume_result": remote_result,
                "automatic_retry": False,
            }
            atomic_json(output_dir / "audit.json", packet)
            return packet
        except Exception as error:
            packet = {
                "schema_version": 1,
                "status": "RED",
                "audit_id": audit_spec.get("audit_id"),
                "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "worker_created": False,
                "gpu_work": False,
                "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
                "automatic_retry": False,
            }
            atomic_json(output_dir / "audit.json", packet)
            raise
        finally:
            if lease is not None:
                lease.close()
            if instance_id is not None:
                _vast("stop", "instance", str(instance_id), check=False)
                _vast("destroy", "instance", str(instance_id), check=False)
            if reaper is not None and not reaper_handed_off:
                reaper.release_to_reap()
            account = _wait_account_zero()
            atomic_json(output_dir / "final-account-audit.json", account)
            if packet is not None:
                packet["final_account_audit"] = account
                if not account.get("query_succeeded") or account.get("active_instances") != 0:
                    packet["status"] = "RED_ACCOUNT_NOT_ZERO"
                atomic_json(output_dir / "audit.json", packet)
            if sys.exc_info()[0] is None and packet is not None and packet["status"] != "GREEN":
                raise LifecycleError("cache audit did not finish GREEN with Vast account zero")


def mirror_persistent_object_to_s3(
    *,
    root: Path,
    spec_path: Path,
    mirror_spec_path: Path,
    volume_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Upload one exact small cache object through a single-use signed URL; never create a worker."""
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="mirror")
    mirror_spec = _validate_bound_object_mirror_spec(root, spec, mirror_spec_path)
    _validate_paid_target(spec, volume_id=volume_id)
    destination = mirror_spec["destination"]
    absent = _run(
        [
            "aws", "s3api", "head-object",
            "--bucket", str(destination["bucket"]),
            "--key", str(destination["key"]),
            "--profile", "fuelos", "--region", "us-east-1", "--output", "json",
        ],
        check=False,
    )
    absent_error = absent.stderr.decode("utf-8", errors="replace")
    if absent.returncode == 0:
        raise LifecycleError("content-addressed probe destination already exists")
    if "404" not in absent_error and "Not Found" not in absent_error:
        raise LifecycleError("could not prove content-addressed probe destination absent")
    credentials_command = _run(
        [
            "aws", "configure", "export-credentials", "--format", "process",
            "--profile", "fuelos", "--region", "us-east-1",
        ]
    )
    credentials = json.loads(credentials_command.stdout.decode("utf-8"))
    upload_url = presign_put(
        access_key=str(credentials.get("AccessKeyId", "")),
        secret_key=str(credentials.get("SecretAccessKey", "")),
        session_token=credentials.get("SessionToken"),
        bucket=str(destination["bucket"]),
        key=str(destination["key"]),
        region="us-east-1",
        checksum_sha256_base64=str(destination["checksum_sha256_base64"]),
        metadata_sha256_hex=str(mirror_spec["source"]["sha256"]),
        now=datetime.now(timezone.utc),
        expires_seconds=int(mirror_spec["presigned_url_lifetime_seconds"]),
    )
    del credentials
    authorization_claim = _claim_one_shot_authorization(
        root=root, spec_path=spec_path, spec=spec, mode="mirror"
    )
    market = select_live_gateway_offer(spec=spec, volume_id=volume_id)
    _enforce_gateway_budget(spec, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "market.json", market)
    instance_id: int | None = None
    packet: dict[str, Any] | None = None
    lease: BootstrapLease | None = None
    reaper: LocalBootstrapReaper | None = None
    reaper_handed_off = False
    with tempfile.TemporaryDirectory(prefix="parallax-object-mirror-") as temporary:
        temporary_path = Path(temporary)
        try:
            run_id = str(mirror_spec["run_id"])
            reaper = LocalBootstrapReaper(
                temporary_path,
                repo_root=root,
                run_id=run_id,
                lifecycle_key_path=lifecycle_key_path,
                maximum_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]) + 60,
            )
            created = _create_instance(
                gateway_offer_id,
                [
                    "--image", str(spec["image"]), "--disk", str(spec["gateway_disk_gb"]),
                    "--ssh", "--direct", "--cancel-unavail", "--link-volume", str(volume_id),
                    "--mount-path", str(spec["volume_mount"]), "--label", f"parallax-guard-{run_id}",
                    "--onstart-cmd", _persist_instance_label_onstart(),
                ],
            )
            instance_id = int(created["new_contract"])
            row = wait_ssh(
                instance_id,
                bootstrap_key,
                timeout_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]),
            )
            control_path = _short_control_path(role="object-mirror", instance_id=instance_id)
            ssh = _ssh_base(row, bootstrap_key, control_path=control_path)
            scp = _scp_base(row, bootstrap_key, control_path=control_path)
            lease = BootstrapLease(ssh)
            remote_root = f"/tmp/parallax-object-mirror-{run_id}"
            remote_secret = f"{remote_root}/scoped-lifecycle.key"
            remote_url = f"{remote_root}/put-url.secret"
            _run([*ssh, f"umask 077; mkdir -p {shlex_quote(remote_root)}"])
            _write_remote_secret(
                ssh, remote_secret, lifecycle_key_path.read_text(encoding="utf-8").strip()
            )
            _write_remote_secret(ssh, remote_url, upload_url)
            upload_url = ""
            deadman = int(spec["policy"]["object_mirror_deadman_seconds"])
            _run(
                [
                    *ssh,
                    scoped_deadman_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        seconds=deadman,
                        action="stop",
                    )
                    + " "
                    + scoped_deadman_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        seconds=deadman + 30,
                        action="destroy",
                    ),
                ]
            )
            reaper_receipt = reaper.handoff(run_id=run_id, target="remote_scoped_deadman")
            reaper_handed_off = True
            host, _port = _ssh_endpoint(row)
            remote_source_root = "/opt/parallax-object-mirror"
            remote_uploader = f"{remote_source_root}/s3_object_mirror.py"
            _run([*ssh, f"mkdir -p {remote_source_root}"])
            _run(
                [
                    *scp,
                    str(root / "infra/vast/parallax/s3_object_mirror.py"),
                    f"root@{host}:{remote_uploader}",
                ]
            )
            remote_spec = f"{remote_root}/mirror-spec.json"
            _run([*scp, str(mirror_spec_path), f"root@{host}:{remote_spec}"])
            remote_result = str(mirror_spec["durable_result"])
            _run(
                [
                    *ssh,
                    f"mkdir -p {shlex_quote(str(PurePosixPath(remote_result).parent))}; "
                    f"test ! -e {shlex_quote(remote_result)}",
                ]
            )
            _run(
                [
                    *ssh,
                    scoped_result_stop_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        result_path=remote_result,
                    ),
                ]
            )
            completed = _run(
                [
                    *ssh,
                    f"python3 {remote_uploader} "
                    f"--spec {shlex_quote(remote_spec)} --url-file {shlex_quote(remote_url)} "
                    f"--output {shlex_quote(remote_result)}",
                ],
                check=False,
                timeout_seconds=max(30, deadman - 20),
            )
            atomic_json(
                output_dir / "command.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.decode("utf-8", errors="replace")[-4000:],
                    "stderr": completed.stderr.decode("utf-8", errors="replace")[-4000:],
                },
            )
            local_temporary = output_dir / ".mirror-result.json.partial"
            copied = _run(
                [*scp, f"root@{host}:{remote_result}", str(local_temporary)],
                check=False,
                timeout_seconds=30,
            )
            if copied.returncode != 0 or not local_temporary.is_file():
                raise LifecycleError("object mirror produced no retrievable result")
            if local_temporary.stat().st_size > 2_000_000:
                raise LifecycleError("object mirror result exceeded its local byte ceiling")
            result_path = output_dir / "mirror-result.json"
            os.replace(local_temporary, result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            head = _run(
                [
                    "aws", "s3api", "head-object",
                    "--bucket", str(destination["bucket"]),
                    "--key", str(destination["key"]),
                    "--checksum-mode", "ENABLED",
                    "--profile", "fuelos", "--region", "us-east-1", "--output", "json",
                ],
                check=False,
            )
            head_payload = (
                json.loads(head.stdout.decode("utf-8")) if head.returncode == 0 else {}
            )
            head_green = (
                int(head_payload.get("ContentLength", -1)) == int(mirror_spec["source"]["bytes"])
                and head_payload.get("ChecksumSHA256")
                == destination["checksum_sha256_base64"]
                and head_payload.get("Metadata", {}).get("sha256")
                == mirror_spec["source"]["sha256"]
                and head_payload.get("ServerSideEncryption") == "AES256"
            )
            atomic_json(
                output_dir / "s3-head.json",
                {
                    "status": "GREEN" if head_green else "RED",
                    "bucket": destination["bucket"],
                    "key": destination["key"],
                    "content_length": head_payload.get("ContentLength"),
                    "checksum_sha256_base64": head_payload.get("ChecksumSHA256"),
                    "metadata_sha256": head_payload.get("Metadata", {}).get("sha256"),
                    "server_side_encryption": head_payload.get("ServerSideEncryption"),
                    "version_id": head_payload.get("VersionId"),
                },
            )
            status = (
                "GREEN"
                if completed.returncode == 0
                and result.get("status") == "GREEN_UPLOADED"
                and head_green
                else "RED"
            )
            packet = {
                "schema_version": 1,
                "status": status,
                "run_id": run_id,
                "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "gateway_dph_total": market["selected_gateway"].get("dph_total"),
                "worker_created": False,
                "gpu_work": False,
                "model_loaded": False,
                "science_items_evaluated": 0,
                "confirmation_items_read": 0,
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "scoped_lifecycle_api_key_id": lifecycle_key_id,
                "local_bootstrap_reaper": reaper_receipt,
                "mirror_result": str(result_path),
                "mirror_result_sha256": sha256_file(result_path),
                "s3_head": str(output_dir / "s3-head.json"),
                "s3_head_sha256": sha256_file(output_dir / "s3-head.json"),
                "verified_bytes": result.get("source", {}).get("bytes", 0),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "url_persisted": False,
                "aws_credentials_transmitted": False,
                "stop_before_destroy_requested": True,
                "result_triggered_stop_armed": True,
                "durable_volume_result": remote_result,
                "automatic_retry": False,
            }
            atomic_json(output_dir / "mirror.json", packet)
            return packet
        except Exception as error:
            packet = {
                "schema_version": 1,
                "status": "RED",
                "run_id": mirror_spec.get("run_id"),
                "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "worker_created": False,
                "gpu_work": False,
                "url_persisted": False,
                "aws_credentials_transmitted": False,
                "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
                "automatic_retry": False,
            }
            atomic_json(output_dir / "mirror.json", packet)
            raise
        finally:
            upload_url = ""
            if lease is not None:
                lease.close()
            if instance_id is not None:
                _vast("stop", "instance", str(instance_id), check=False)
                _vast("destroy", "instance", str(instance_id), check=False)
            if reaper is not None and not reaper_handed_off:
                reaper.release_to_reap()
            account = _wait_account_zero()
            atomic_json(output_dir / "final-account-audit.json", account)
            if packet is not None:
                packet["final_account_audit"] = account
                if not account.get("query_succeeded") or account.get("active_instances") != 0:
                    packet["status"] = "RED_ACCOUNT_NOT_ZERO"
                atomic_json(output_dir / "mirror.json", packet)
            if sys.exc_info()[0] is None and packet is not None and packet["status"] != "GREEN":
                raise LifecycleError("object mirror did not finish GREEN with Vast account zero")


def _cache_mirror_head(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    completed = _run(
        [
            "aws", "s3api", "head-object",
            "--bucket", "pg-lineage-769063704696",
            "--key", str(row["destination_key"]),
            "--checksum-mode", "ENABLED",
            "--profile", "fuelos", "--region", "us-east-1", "--output", "json",
        ],
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")
        if "404" in error or "Not Found" in error:
            return "absent", {}
        raise LifecycleError("could not establish S3 cache-object state")
    payload = json.loads(completed.stdout.decode("utf-8"))
    exact = (
        int(payload.get("ContentLength", -1)) == int(row["bytes"])
        and payload.get("ChecksumSHA256") == row["checksum_sha256_base64"]
        and payload.get("Metadata", {}).get("sha256") == row["sha256"]
        and payload.get("ServerSideEncryption") == "AES256"
        and bool(payload.get("VersionId"))
    )
    return ("exact" if exact else "mismatch"), payload


def mirror_persistent_cache_to_s3(
    *,
    root: Path,
    spec_path: Path,
    mirror_spec_path: Path,
    volume_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Mirror an audited cache batch with exact reuse, atomic PUTs, and immediate teardown."""
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="mirror")
    mirror_spec = _validate_bound_cache_mirror_spec(root, spec, mirror_spec_path)
    _validate_paid_target(spec, volume_id=volume_id)
    preflight_rows: list[dict[str, Any]] = []
    absent_rows: list[dict[str, Any]] = []
    for source in mirror_spec["files"]:
        row = dict(source)
        state, head = _cache_mirror_head(row)
        expected = row["expected_initial_state"]
        if expected == "absent" and state != "absent":
            raise LifecycleError("cache mirror expected an absent destination")
        if expected == "exact_existing" and (
            state != "exact" or head.get("VersionId") != row["existing_version_id"]
        ):
            raise LifecycleError("cache mirror exact existing destination changed")
        if expected == "absent":
            absent_rows.append(row)
        preflight_rows.append(
            {
                "path": row["path"], "destination_key": row["destination_key"],
                "expected": expected, "observed": state,
                "version_id": head.get("VersionId"),
            }
        )
    if not absent_rows:
        raise LifecycleError("cache mirror has no absent object to publish")
    credentials_command = _run(
        [
            "aws", "configure", "export-credentials", "--format", "process",
            "--profile", "fuelos", "--region", "us-east-1",
        ]
    )
    credentials = json.loads(credentials_command.stdout.decode("utf-8"))
    upload_urls = {
        row["destination_key"]: presign_put(
            access_key=str(credentials.get("AccessKeyId", "")),
            secret_key=str(credentials.get("SecretAccessKey", "")),
            session_token=credentials.get("SessionToken"),
            bucket="pg-lineage-769063704696",
            key=str(row["destination_key"]),
            region="us-east-1",
            checksum_sha256_base64=str(row["checksum_sha256_base64"]),
            metadata_sha256_hex=str(row["sha256"]),
            now=datetime.now(timezone.utc),
            expires_seconds=int(mirror_spec["presigned_url_lifetime_seconds"]),
        )
        for row in absent_rows
    }
    del credentials
    authorization_claim = _claim_one_shot_authorization(
        root=root, spec_path=spec_path, spec=spec, mode="mirror"
    )
    market = select_live_gateway_offer(spec=spec, volume_id=volume_id)
    _enforce_gateway_budget(spec, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "market.json", market)
    atomic_json(
        output_dir / "s3-preflight.json",
        {
            "schema_version": 1, "status": "GREEN", "mutation": "none_read_only",
            "rows": preflight_rows, "absent_count": len(absent_rows),
            "exact_existing_count": len(preflight_rows) - len(absent_rows),
        },
    )
    instance_id: int | None = None
    packet: dict[str, Any] | None = None
    lease: BootstrapLease | None = None
    reaper: LocalBootstrapReaper | None = None
    reaper_handed_off = False
    with tempfile.TemporaryDirectory(prefix="parallax-cache-mirror-") as temporary:
        temporary_path = Path(temporary)
        try:
            run_id = str(mirror_spec["run_id"])
            reaper = LocalBootstrapReaper(
                temporary_path,
                repo_root=root,
                run_id=run_id,
                lifecycle_key_path=lifecycle_key_path,
                maximum_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]) + 60,
            )
            created = _create_instance(
                gateway_offer_id,
                [
                    "--image", str(spec["image"]), "--disk", str(spec["gateway_disk_gb"]),
                    "--ssh", "--direct", "--cancel-unavail", "--link-volume", str(volume_id),
                    "--mount-path", str(spec["volume_mount"]),
                    "--label", f"parallax-guard-{run_id}",
                    "--onstart-cmd", _persist_instance_label_onstart(),
                ],
            )
            instance_id = int(created["new_contract"])
            live = wait_ssh(
                instance_id, bootstrap_key,
                timeout_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]),
            )
            control_path = _short_control_path(role="cache-mirror", instance_id=instance_id)
            ssh = _ssh_base(live, bootstrap_key, control_path=control_path)
            scp = _scp_base(live, bootstrap_key, control_path=control_path)
            lease = BootstrapLease(ssh)
            remote_root = f"/tmp/parallax-cache-mirror-{run_id}"
            remote_secret = f"{remote_root}/scoped-lifecycle.key"
            remote_urls = f"{remote_root}/put-urls.secret.json"
            _run([*ssh, f"umask 077; mkdir -p {shlex_quote(remote_root)}"])
            _write_remote_secret(
                ssh, remote_secret, lifecycle_key_path.read_text(encoding="utf-8").strip()
            )
            _write_remote_secret(ssh, remote_urls, json.dumps(upload_urls, sort_keys=True))
            upload_urls = {}
            deadman = int(spec["policy"]["object_mirror_deadman_seconds"])
            _run(
                [
                    *ssh,
                    scoped_deadman_command(
                        instance_id=instance_id, key_path=remote_secret,
                        seconds=deadman, action="stop",
                    )
                    + " "
                    + scoped_deadman_command(
                        instance_id=instance_id, key_path=remote_secret,
                        seconds=deadman + 30, action="destroy",
                    ),
                ]
            )
            reaper_receipt = reaper.handoff(run_id=run_id, target="remote_scoped_deadman")
            reaper_handed_off = True
            host, _port = _ssh_endpoint(live)
            remote_source_root = "/opt/parallax-cache-mirror"
            remote_uploader = f"{remote_source_root}/s3_cache_mirror.py"
            _run([*ssh, f"mkdir -p {remote_source_root}"])
            _run(
                [
                    *scp, str(root / "infra/vast/parallax/s3_cache_mirror.py"),
                    f"root@{host}:{remote_uploader}",
                ]
            )
            remote_spec = f"{remote_root}/mirror-spec.json"
            _run([*scp, str(mirror_spec_path), f"root@{host}:{remote_spec}"])
            remote_result = str(mirror_spec["durable_result"])
            _run(
                [
                    *ssh,
                    f"mkdir -p {shlex_quote(str(PurePosixPath(remote_result).parent))}; "
                    f"test ! -e {shlex_quote(remote_result)}",
                ]
            )
            _run(
                [
                    *ssh,
                    scoped_result_stop_command(
                        instance_id=instance_id, key_path=remote_secret,
                        result_path=remote_result,
                    ),
                ]
            )
            completed = _run(
                [
                    *ssh,
                    f"python3 {remote_uploader} --spec {shlex_quote(remote_spec)} "
                    f"--url-file {shlex_quote(remote_urls)} --output {shlex_quote(remote_result)}",
                ],
                check=False,
                timeout_seconds=max(30, deadman - 20),
            )
            atomic_json(
                output_dir / "command.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.decode("utf-8", errors="replace")[-4000:],
                    "stderr": completed.stderr.decode("utf-8", errors="replace")[-4000:],
                },
            )
            local_temporary = output_dir / ".mirror-result.json.partial"
            copied = _run(
                [*scp, f"root@{host}:{remote_result}", str(local_temporary)],
                check=False,
                timeout_seconds=60,
            )
            if copied.returncode != 0 or not local_temporary.is_file():
                raise LifecycleError("cache mirror produced no retrievable result")
            if local_temporary.stat().st_size > 5_000_000:
                raise LifecycleError("cache mirror result exceeded local byte ceiling")
            result_path = output_dir / "mirror-result.json"
            os.replace(local_temporary, result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result_rows = {
                str(item.get("path")): item
                for item in result.get("files", []) if isinstance(item, dict)
            }
            final_rows: list[dict[str, Any]] = []
            final_errors: list[str] = []
            for source in mirror_spec["files"]:
                state, head = _cache_mirror_head(source)
                result_row = result_rows.get(str(source["path"]), {})
                version_matches = (
                    head.get("VersionId") == result_row.get("version_id")
                    if result_row.get("status") == "GREEN_UPLOADED"
                    else head.get("VersionId") == source.get("existing_version_id")
                )
                green = state == "exact" and version_matches
                if not green:
                    final_errors.append(str(source["path"]))
                final_rows.append(
                    {
                        "path": source["path"], "bytes": source["bytes"],
                        "sha256": source["sha256"],
                        "key": source["destination_key"],
                        "version_id": head.get("VersionId"),
                        "status": "GREEN" if green else "RED",
                    }
                )
            s3_manifest = {
                "schema_version": 1,
                "status": "GREEN" if not final_errors else "RED",
                "bucket": "pg-lineage-769063704696", "region": "us-east-1",
                "run_id": run_id, "mode": mirror_spec["mode"],
                "file_count": len(final_rows),
                "total_bytes": sum(int(item["bytes"]) for item in final_rows),
                "files": final_rows, "errors": final_errors,
            }
            atomic_json(output_dir / "s3-manifest.json", s3_manifest)
            status = (
                "GREEN" if completed.returncode == 0
                and result.get("status") == "GREEN_MIRRORED"
                and len(result_rows) == len(mirror_spec["files"])
                and not final_errors else "RED"
            )
            packet = {
                "schema_version": 1, "status": status, "run_id": run_id,
                "mode": mirror_spec["mode"], "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "gateway_dph_total": market["selected_gateway"].get("dph_total"),
                "worker_created": False, "gpu_work": False, "model_loaded": False,
                "science_items_evaluated": 0, "confirmation_items_read": 0,
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "scoped_lifecycle_api_key_id": lifecycle_key_id,
                "local_bootstrap_reaper": reaper_receipt,
                "mirror_result": str(result_path),
                "mirror_result_sha256": sha256_file(result_path),
                "s3_manifest": str(output_dir / "s3-manifest.json"),
                "s3_manifest_sha256": sha256_file(output_dir / "s3-manifest.json"),
                "verified_bytes": s3_manifest["total_bytes"],
                "uploaded_bytes": result.get("uploaded_bytes", 0),
                "reused_bytes": result.get("reused_bytes", 0),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "url_persisted": False, "aws_credentials_transmitted": False,
                "stop_before_destroy_requested": True,
                "result_triggered_stop_armed": True,
                "durable_volume_result": remote_result, "automatic_retry": False,
            }
            atomic_json(output_dir / "mirror.json", packet)
            return packet
        except Exception as error:
            packet = {
                "schema_version": 1, "status": "RED",
                "run_id": mirror_spec.get("run_id"), "mode": mirror_spec.get("mode"),
                "volume_id": volume_id, "gateway_instance_id": instance_id,
                "worker_created": False, "gpu_work": False,
                "url_persisted": False, "aws_credentials_transmitted": False,
                "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
                "automatic_retry": False,
            }
            atomic_json(output_dir / "mirror.json", packet)
            raise
        finally:
            upload_urls = {}
            if lease is not None:
                lease.close()
            if instance_id is not None:
                _vast("stop", "instance", str(instance_id), check=False)
                _vast("destroy", "instance", str(instance_id), check=False)
            if reaper is not None and not reaper_handed_off:
                reaper.release_to_reap()
            account = _wait_account_zero()
            atomic_json(output_dir / "final-account-audit.json", account)
            if packet is not None:
                packet["final_account_audit"] = account
                if not account.get("query_succeeded") or account.get("active_instances") != 0:
                    packet["status"] = "RED_ACCOUNT_NOT_ZERO"
                atomic_json(output_dir / "mirror.json", packet)
            if sys.exc_info()[0] is None and packet is not None and packet["status"] != "GREEN":
                raise LifecycleError("cache mirror did not finish GREEN with Vast account zero")


def materialize_persistent_task_cache(
    *,
    root: Path,
    spec_path: Path,
    task_spec_path: Path,
    volume_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the exact task bank in the qualified image directly on the volume."""
    spec = validate_frozen_spec(root, spec_path)
    require_paid_authorization(spec, mode="task")
    task_spec = _validate_bound_task_cache_spec(root, spec, task_spec_path)
    _validate_paid_target(spec, volume_id=volume_id)
    authorization_claim = _claim_one_shot_authorization(
        root=root, spec_path=spec_path, spec=spec, mode="task"
    )
    market = select_live_gateway_offer(spec=spec, volume_id=volume_id)
    _enforce_gateway_budget(spec, market)
    gateway_offer_id = int(market["selected_gateway"]["id"])
    bootstrap_key = _validated_bootstrap_ssh_key(spec)
    lifecycle_key_path, lifecycle_key_id = _validated_scoped_lifecycle_api_key(spec)
    registry_login = _ephemeral_worker_registry_login(spec)
    if registry_login is None:
        raise LifecycleError("task materialization lacks ephemeral registry authentication")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "market.json", market)
    instance_id: int | None = None
    packet: dict[str, Any] | None = None
    lease: BootstrapLease | None = None
    reaper: LocalBootstrapReaper | None = None
    reaper_handed_off = False
    with tempfile.TemporaryDirectory(prefix="parallax-task-cache-") as temporary:
        temporary_path = Path(temporary)
        try:
            run_id = str(task_spec["run_id"])
            reaper = LocalBootstrapReaper(
                temporary_path,
                repo_root=root,
                run_id=run_id,
                lifecycle_key_path=lifecycle_key_path,
                maximum_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]) + 60,
            )
            created = _create_instance(
                gateway_offer_id,
                [
                    "--image", str(spec["image"]), "--disk", str(spec["gateway_disk_gb"]),
                    "--ssh", "--direct", "--cancel-unavail", "--link-volume", str(volume_id),
                    "--mount-path", str(spec["volume_mount"]), "--label", f"parallax-guard-{run_id}",
                    "--onstart-cmd", _persist_instance_label_onstart(), "--login", registry_login,
                ],
            )
            instance_id = int(created["new_contract"])
            row = wait_ssh(
                instance_id,
                bootstrap_key,
                timeout_seconds=int(spec["policy"]["bootstrap_timeout_seconds"]),
            )
            control_path = _short_control_path(role="task-cache", instance_id=instance_id)
            ssh = _ssh_base(row, bootstrap_key, control_path=control_path)
            scp = _scp_base(row, bootstrap_key, control_path=control_path)
            lease = BootstrapLease(ssh)
            remote_root = f"/tmp/parallax-task-cache-{run_id}"
            remote_secret = f"{remote_root}/scoped-lifecycle.key"
            _run([*ssh, f"umask 077; mkdir -p {shlex_quote(remote_root)}"])
            _write_remote_secret(
                ssh,
                remote_secret,
                lifecycle_key_path.read_text(encoding="utf-8").strip(),
            )
            deadman = int(spec["policy"]["task_materialization_deadman_seconds"])
            _run(
                [
                    *ssh,
                    scoped_deadman_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        seconds=deadman,
                        action="stop",
                    )
                    + " "
                    + scoped_deadman_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        seconds=deadman + 30,
                        action="destroy",
                    ),
                ]
            )
            reaper_receipt = reaper.handoff(run_id=run_id, target="remote_scoped_deadman")
            reaper_handed_off = True
            archive = temporary_path / "lifecycle.tar.gz"
            _normalized_lifecycle_archive(root, archive)
            host, _port = _ssh_endpoint(row)
            _run([*ssh, "mkdir -p /opt/parallax-lifecycle"])
            _run([*scp, str(archive), f"root@{host}:/tmp/parallax-lifecycle.tar.gz"])
            _run([*ssh, "tar -C /opt/parallax-lifecycle -xzf /tmp/parallax-lifecycle.tar.gz"])
            _run([*scp, str(task_spec_path), f"root@{host}:{remote_root}/task-cache-spec.json"])
            durable_receipt = str(task_spec["receipt_path"])
            _run(
                [
                    *ssh,
                    f"mkdir -p {shlex_quote(str(PurePosixPath(durable_receipt).parent))}; "
                    f"test ! -e {shlex_quote(durable_receipt)}",
                ]
            )
            _run(
                [
                    *ssh,
                    scoped_result_stop_command(
                        instance_id=instance_id,
                        key_path=remote_secret,
                        result_path=durable_receipt,
                    ),
                ]
            )
            completed = _run(
                [
                    *ssh,
                    f"PYTHONPATH=/opt/parallax-lifecycle:/opt/parallax "
                    f"python3 -m infra.vast.parallax.task_cache_job "
                    f"--spec {shlex_quote(remote_root + '/task-cache-spec.json')}",
                ],
                check=False,
                timeout_seconds=max(60, deadman - 20),
            )
            atomic_json(
                output_dir / "command.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.decode("utf-8", errors="replace")[-8000:],
                    "stderr": completed.stderr.decode("utf-8", errors="replace")[-8000:],
                },
            )
            local_temporary = output_dir / ".task-cache-receipt.json.partial"
            copied = _run(
                [*scp, f"root@{host}:{durable_receipt}", str(local_temporary)],
                check=False,
                timeout_seconds=30,
            )
            if copied.returncode != 0 or not local_temporary.is_file():
                raise LifecycleError("task materialization produced no retrievable durable receipt")
            if local_temporary.stat().st_size > 2_000_000:
                raise LifecycleError("task materialization receipt exceeded its local byte ceiling")
            result_path = output_dir / "task-cache-receipt.json"
            os.replace(local_temporary, result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = "GREEN" if completed.returncode == 0 and result.get("status") == "GREEN" else "RED"
            verification = result.get("verification", {})
            packet = {
                "schema_version": 1,
                "status": status,
                "run_id": run_id,
                "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "gateway_dph_total": market["selected_gateway"].get("dph_total"),
                "gateway_image": spec["image"],
                "worker_created": False,
                "gpu_work": False,
                "model_reads": 0,
                "authorization_claim": str(authorization_claim),
                "authorization_claim_sha256": sha256_file(authorization_claim),
                "scoped_lifecycle_api_key_id": lifecycle_key_id,
                "local_bootstrap_reaper": reaper_receipt,
                "task_cache_receipt": str(result_path),
                "task_cache_receipt_sha256": sha256_file(result_path),
                "durable_volume_receipt": durable_receipt,
                "final_root": result.get("final_root"),
                "reused_existing": result.get("reused_existing"),
                "payload_bytes": verification.get("payload_bytes"),
                "total_items": verification.get("total_items"),
                "total_images": verification.get("total_images"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_triggered_stop_armed": True,
                "stop_before_destroy_requested": True,
                "automatic_retry": False,
            }
            atomic_json(output_dir / "materialization.json", packet)
            return packet
        except Exception as error:
            packet = {
                "schema_version": 1,
                "status": "RED",
                "run_id": task_spec.get("run_id"),
                "volume_id": volume_id,
                "gateway_instance_id": instance_id,
                "worker_created": False,
                "gpu_work": False,
                "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
                "automatic_retry": False,
            }
            atomic_json(output_dir / "materialization.json", packet)
            raise
        finally:
            if lease is not None:
                lease.close()
            if instance_id is not None:
                _vast("stop", "instance", str(instance_id), check=False)
                _vast("destroy", "instance", str(instance_id), check=False)
            if reaper is not None and not reaper_handed_off:
                reaper.release_to_reap()
            account = _wait_account_zero()
            atomic_json(output_dir / "final-account-audit.json", account)
            if packet is not None:
                packet["final_account_audit"] = account
                if not account.get("query_succeeded") or account.get("active_instances") != 0:
                    packet["status"] = "RED_ACCOUNT_NOT_ZERO"
                atomic_json(output_dir / "materialization.json", packet)
            if sys.exc_info()[0] is None and packet is not None and packet["status"] != "GREEN":
                raise LifecycleError("task materialization did not finish GREEN with Vast account zero")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--root", type=Path, required=True)
    launch_parser.add_argument("--spec", type=Path, required=True)
    launch_parser.add_argument("--job-spec", type=Path, required=True)
    launch_parser.add_argument("--volume-id", type=int, required=True)
    launch_parser.add_argument("--output", type=Path, required=True)
    control_parser = subparsers.add_parser("control-probe")
    control_parser.add_argument("--root", type=Path, required=True)
    control_parser.add_argument("--spec", type=Path, required=True)
    control_parser.add_argument("--volume-id", type=int, required=True)
    control_parser.add_argument("--output", type=Path, required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--root", type=Path, required=True)
    inspect_parser.add_argument("--spec", type=Path, required=True)
    inspect_parser.add_argument("--volume-id", type=int, required=True)
    inspect_parser.add_argument("--run-id", required=True)
    inspect_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit-cache")
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument("--spec", type=Path, required=True)
    audit_parser.add_argument("--audit-spec", type=Path, required=True)
    audit_parser.add_argument("--volume-id", type=int, required=True)
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    mirror_parser = subparsers.add_parser("mirror-object")
    mirror_parser.add_argument("--root", type=Path, required=True)
    mirror_parser.add_argument("--spec", type=Path, required=True)
    mirror_parser.add_argument("--mirror-spec", type=Path, required=True)
    mirror_parser.add_argument("--volume-id", type=int, required=True)
    mirror_parser.add_argument("--output-dir", type=Path, required=True)
    cache_mirror_parser = subparsers.add_parser("mirror-cache")
    cache_mirror_parser.add_argument("--root", type=Path, required=True)
    cache_mirror_parser.add_argument("--spec", type=Path, required=True)
    cache_mirror_parser.add_argument("--mirror-spec", type=Path, required=True)
    cache_mirror_parser.add_argument("--volume-id", type=int, required=True)
    cache_mirror_parser.add_argument("--output-dir", type=Path, required=True)
    task_parser = subparsers.add_parser("materialize-tasks")
    task_parser.add_argument("--root", type=Path, required=True)
    task_parser.add_argument("--spec", type=Path, required=True)
    task_parser.add_argument("--task-spec", type=Path, required=True)
    task_parser.add_argument("--volume-id", type=int, required=True)
    task_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require_persistent_volume_controller_active()
    if args.command == "launch":
        packet = launch(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            job_spec_path=args.job_spec.resolve(),
            volume_id=args.volume_id,
            output=args.output.resolve(),
        )
    elif args.command == "control-probe":
        packet = probe_gateway_controls(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            volume_id=args.volume_id,
            output=args.output.resolve(),
        )
    elif args.command == "inspect":
        packet = inspect_persisted_run(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            volume_id=args.volume_id,
            run_id=args.run_id,
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "audit-cache":
        packet = audit_persistent_cache(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            audit_spec_path=args.audit_spec.resolve(),
            volume_id=args.volume_id,
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "mirror-object":
        packet = mirror_persistent_object_to_s3(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            mirror_spec_path=args.mirror_spec.resolve(),
            volume_id=args.volume_id,
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "mirror-cache":
        packet = mirror_persistent_cache_to_s3(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            mirror_spec_path=args.mirror_spec.resolve(),
            volume_id=args.volume_id,
            output_dir=args.output_dir.resolve(),
        )
    else:
        packet = materialize_persistent_task_cache(
            root=args.root.resolve(),
            spec_path=args.spec.resolve(),
            task_spec_path=args.task_spec.resolve(),
            volume_id=args.volume_id,
            output_dir=args.output_dir.resolve(),
        )
    print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
