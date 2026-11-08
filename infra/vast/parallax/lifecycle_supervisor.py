"""Fail-closed Vast lifecycle supervisor with continuous durable mirroring.

The cheap volume gateway runs this process detached.  The H100 is destroyed
only after a final volume manifest is green.  If durability is unproven, the
instance is stopped instead, ending compute billing while preserving its disk.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .artifacts import ArtifactManifest, atomic_json, sha256_file


API_BASE = "https://console.vast.ai/api/v0"


class LifecycleError(RuntimeError):
    """Raised when a lifecycle contract or transition is unsafe."""


@dataclass(frozen=True)
class LifecyclePolicy:
    run_id: str
    maximum_runtime_seconds: int
    mirror_interval_seconds: float
    heartbeat_timeout_seconds: float
    artifact_copy_timeout_seconds: float
    maximum_finalization_seconds: float
    final_sync_attempts: int
    final_sync_settle_seconds: float
    required_sentinels: tuple[str, ...]
    execution_release_timeout_seconds: float = 30.0

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "LifecyclePolicy":
        if contract.get("schema_version") != 1 or contract.get("status") != "FROZEN_RUNTIME":
            raise LifecycleError("runtime lifecycle contract is not frozen")
        policy = contract.get("policy", {})
        run_id = str(contract.get("run_id", ""))
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in run_id):
            raise LifecycleError("run_id is not a safe lowercase identifier")
        parsed = cls(
            run_id=run_id,
            maximum_runtime_seconds=int(policy["maximum_runtime_seconds"]),
            mirror_interval_seconds=float(policy["mirror_interval_seconds"]),
            heartbeat_timeout_seconds=float(policy["heartbeat_timeout_seconds"]),
            artifact_copy_timeout_seconds=float(policy["artifact_copy_timeout_seconds"]),
            maximum_finalization_seconds=float(policy["maximum_finalization_seconds"]),
            final_sync_attempts=int(policy["final_sync_attempts"]),
            final_sync_settle_seconds=float(policy["final_sync_settle_seconds"]),
            required_sentinels=tuple(str(value) for value in policy["required_sentinels"]),
            execution_release_timeout_seconds=float(
                policy.get("execution_release_timeout_seconds", 30.0)
            ),
        )
        if parsed.maximum_runtime_seconds <= 0 or parsed.maximum_runtime_seconds > 24 * 60 * 60:
            raise LifecycleError("maximum runtime must be in (0, 24h]")
        if not 0.5 <= parsed.mirror_interval_seconds <= 5.0:
            raise LifecycleError("mirror interval must be between 0.5 and 5 seconds")
        if parsed.heartbeat_timeout_seconds < 3 * parsed.mirror_interval_seconds:
            raise LifecycleError("heartbeat timeout must cover at least three mirror intervals")
        if not 5 <= parsed.artifact_copy_timeout_seconds <= 300:
            raise LifecycleError("artifact copy timeout must be in [5, 300] seconds")
        if not 5 <= parsed.maximum_finalization_seconds <= 300:
            raise LifecycleError("maximum finalization must be in [5, 300] seconds")
        if not 2 <= parsed.final_sync_attempts <= 10:
            raise LifecycleError("final sync attempts must be between 2 and 10")
        if not 0.0 <= parsed.final_sync_settle_seconds <= 2.0:
            raise LifecycleError("final sync settle must be in [0, 2] seconds")
        if not parsed.required_sentinels:
            raise LifecycleError("at least one durable result sentinel is required")
        if not 5 <= parsed.execution_release_timeout_seconds <= 60:
            raise LifecycleError("execution release timeout must be in [5, 60] seconds")
        for sentinel in parsed.required_sentinels:
            path = PurePosixPath(sentinel)
            if path.is_absolute() or ".." in path.parts:
                raise LifecycleError(f"unsafe sentinel path: {sentinel}")
        teardown = contract.get("teardown", {})
        if teardown.get("durability_before_destroy") is not True:
            raise LifecycleError("destroy must require proven durability")
        if teardown.get("stop_on_unproven_durability") is not True:
            raise LifecycleError("unproven durability must stop rather than destroy")
        if teardown.get("h100_api_key_scope") not in {"per_instance_only", "scoped_lifecycle_only"}:
            raise LifecycleError("H100 teardown key must be narrowly scoped")
        return parsed


def validate_runtime_contract(contract: dict[str, Any]) -> LifecyclePolicy:
    """Validate the complete remote contract before any job can start."""
    policy = LifecyclePolicy.from_contract(contract)
    job = contract.get("job", {})
    if job.get("execution_site") not in {"gateway", "h100"}:
        raise LifecycleError("job.execution_site must be gateway or h100")
    command = job.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
        raise LifecycleError("job.command must be a nonempty string list")
    cwd = PurePosixPath(str(job.get("cwd", "")))
    if not cwd.is_absolute() or ".." in cwd.parts:
        raise LifecycleError("job.cwd must be an absolute safe path")
    roots = contract.get("artifacts", {}).get("roots")
    if not isinstance(roots, list) or not roots:
        raise LifecycleError("at least one artifact root is required")
    prefixes: set[str] = set()
    for row in roots:
        remote_root = PurePosixPath(str(row.get("remote_root", "")))
        prefix = PurePosixPath(str(row.get("durable_prefix", "")))
        if not remote_root.is_absolute() or ".." in remote_root.parts:
            raise LifecycleError("artifact remote_root must be absolute and safe")
        if not prefix.parts or prefix.is_absolute() or ".." in prefix.parts:
            raise LifecycleError("artifact durable_prefix must be relative and safe")
        if prefix.as_posix() in prefixes:
            raise LifecycleError("artifact durable_prefix values must be unique")
        prefixes.add(prefix.as_posix())
    if any(sentinel.split("/", 1)[0] not in prefixes for sentinel in policy.required_sentinels):
        raise LifecycleError("every sentinel must belong to a declared durable prefix")
    expected = contract.get("expected_artifacts")
    if expected is not None:
        if not isinstance(expected, dict) or not expected:
            raise LifecycleError("expected_artifacts must be a nonempty exact-byte map")
        for relative, binding in expected.items():
            path = PurePosixPath(str(relative))
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise LifecycleError("unsafe expected artifact path")
            if path.parts[0] not in prefixes or not isinstance(binding, dict):
                raise LifecycleError("expected artifact is outside the declared roots")
            digest = str(binding.get("sha256", ""))
            if int(binding.get("bytes", -1)) < 0 or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise LifecycleError("invalid expected artifact byte binding")
        if not set(policy.required_sentinels).issubset(expected):
            raise LifecycleError("every sentinel must have an exact expected byte binding")
    h100 = contract.get("h100", {})
    gateway = contract.get("gateway", {})
    if int(h100.get("instance_id", 0)) <= 0 or int(gateway.get("instance_id", 0)) <= 0:
        raise LifecycleError("both instance ids must be positive")
    if int(h100["instance_id"]) == int(gateway["instance_id"]):
        raise LifecycleError("gateway and accelerator must be different instances")
    for key in ("ssh_private_key_file", "instance_api_key_file"):
        if not PurePosixPath(str(h100.get(key, ""))).is_absolute():
            raise LifecycleError(f"h100.{key} must be absolute")
    if not PurePosixPath(str(gateway.get("instance_api_key_file", ""))).is_absolute():
        raise LifecycleError("gateway.instance_api_key_file must be absolute")
    if not PurePosixPath(str(gateway.get("volume_run_root", ""))).is_absolute():
        raise LifecycleError("gateway.volume_run_root must be absolute")
    return policy


class JobHandle(Protocol):
    def poll(self) -> int | None: ...

    def wait(self) -> int: ...


class SupervisorBackend(Protocol):
    def start_job(self) -> JobHandle: ...

    def heartbeat(self, payload: dict[str, Any]) -> None: ...

    def mirror_once(
        self, *, final: bool = False, deadline_monotonic: float | None = None
    ) -> dict[str, Any]: ...

    def prove_durability(self, *, job_exit_code: int) -> dict[str, Any]: ...

    def destroy_h100(self) -> dict[str, Any]: ...

    def stop_h100(self, *, reason: str) -> dict[str, Any]: ...

    def write_receipt(self, payload: dict[str, Any]) -> None: ...

    def sleep(self, seconds: float) -> None: ...

    def monotonic(self) -> float: ...


def _supervise_inner(policy: LifecyclePolicy, backend: SupervisorBackend) -> dict[str, Any]:
    """Run the normal lifecycle state machine; callers add the emergency boundary."""
    started = backend.monotonic()
    deadline = started + policy.maximum_runtime_seconds
    mirrors: list[dict[str, Any]] = []
    job = backend.start_job()
    job_exit_code: int | None = None
    timed_out = False
    while job_exit_code is None:
        now = backend.monotonic()
        backend.heartbeat(
            {
                "status": "RUNNING",
                "run_id": policy.run_id,
                "monotonic": now,
                "deadline_monotonic": deadline,
            }
        )
        job_exit_code = job.poll()
        if job_exit_code is not None:
            break
        if now >= deadline:
            timed_out = True
            break
        mirrors.append(backend.mirror_once(final=False))
        backend.sleep(min(policy.mirror_interval_seconds, max(0.0, deadline - now)))

    if timed_out:
        # Stop first: this ends GPU billing while preserving all unsynced local bytes.
        teardown = backend.stop_h100(reason="hard_runtime_deadline")
        receipt = {
            "schema_version": 1,
            "run_id": policy.run_id,
            "status": "STOPPED_TIMEOUT_DISK_PRESERVED",
            "job_exit_code": None,
            "durability": {"status": "UNPROVEN", "reason": "hard_runtime_deadline"},
            "teardown": teardown,
            "mirror_passes": mirrors,
            "elapsed_seconds": backend.monotonic() - started,
        }
        backend.write_receipt(receipt)
        return receipt

    job_exit_code = job.wait()
    job_finished = backend.monotonic()
    # Continuous mirroring is the durability mechanism. Final sync is only a
    # bounded salvage window; it must never strand an idle paid accelerator.
    finalization_deadline = backend.monotonic() + policy.maximum_finalization_seconds
    durability: dict[str, Any] | None = None
    for _ in range(policy.final_sync_attempts):
        if backend.monotonic() >= finalization_deadline:
            break
        mirrors.append(
            backend.mirror_once(final=True, deadline_monotonic=finalization_deadline)
        )
        # Exact durability is a teardown license. Once it is proven there is
        # no reason to spend another paid second on a redundant mirror pass.
        durability = backend.prove_durability(job_exit_code=job_exit_code)
        if durability.get("status") == "GREEN":
            break
        if policy.final_sync_settle_seconds:
            remaining = max(0.0, finalization_deadline - backend.monotonic())
            backend.sleep(min(policy.final_sync_settle_seconds, remaining))
    if durability is None:
        durability = backend.prove_durability(job_exit_code=job_exit_code)
    durability_proven = backend.monotonic()
    if durability.get("status") == "GREEN":
        teardown = backend.destroy_h100()
        status = "DURABLE_AND_DESTROYED" if teardown.get("status") == "DESTROYED" else "DURABLE_TEARDOWN_DEGRADED"
    else:
        teardown = backend.stop_h100(reason="durability_unproven")
        status = "STOPPED_DISK_PRESERVED"
    receipt = {
        "schema_version": 1,
        "run_id": policy.run_id,
        "status": status,
        "job_exit_code": job_exit_code,
        "durability": durability,
        "teardown": teardown,
        "mirror_passes": mirrors,
        "job_finished_elapsed_seconds": job_finished - started,
        "durability_proven_elapsed_seconds": durability_proven - started,
        "post_job_finalization_seconds": durability_proven - job_finished,
        "elapsed_seconds": backend.monotonic() - started,
    }
    backend.write_receipt(receipt)
    return receipt


def supervise(policy: LifecyclePolicy, backend: SupervisorBackend) -> dict[str, Any]:
    """Run fail-closed: every ordinary exception immediately requests STOP.

    Signals and a dead Python process are handled by the detached watchdog and
    the provider TTL.  Exceptions that Python can still handle take the faster
    path here, before any attempt to salvage more bytes.
    """
    started = backend.monotonic()
    try:
        return _supervise_inner(policy, backend)
    except Exception as error:
        try:
            teardown = backend.stop_h100(reason="supervisor_exception")
        except Exception as teardown_error:
            teardown = {
                "status": "TEARDOWN_FAILED",
                "error_type": type(teardown_error).__name__,
            }
        receipt = {
            "schema_version": 1,
            "run_id": policy.run_id,
            "status": (
                "STOPPED_INTERNAL_FAILURE_DISK_PRESERVED"
                if teardown.get("status") == "STOPPED"
                else "INTERNAL_FAILURE_TEARDOWN_UNPROVEN"
            ),
            "job_exit_code": None,
            "durability": {"status": "UNPROVEN", "reason": "supervisor_exception"},
            "teardown": teardown,
            "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            "elapsed_seconds": backend.monotonic() - started,
        }
        # If this durable write fails, propagate to run_gateway.  It must not
        # self-destroy; the watchdog retains its key and keeps trying STOP.
        backend.write_receipt(receipt)
        return receipt


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete, fsynced event without rewriting prior journal history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_secret(path: Path) -> str:
    stat = path.stat()
    if stat.st_mode & 0o077:
        raise LifecycleError(f"secret file permissions are not 0600: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise LifecycleError(f"secret file is empty: {path}")
    return value


def vast_instance_action(
    *,
    instance_id: int,
    api_key: str,
    action: str,
    retries: int = 4,
    retry_seconds: float = 1.0,
) -> dict[str, Any]:
    if action not in {"destroy", "stop"}:
        raise ValueError(action)
    url = f"{API_BASE}/instances/{instance_id}/"
    data = None if action == "destroy" else json.dumps({"state": "stopped"}).encode("utf-8")
    method = "DELETE" if action == "destroy" else "PUT"
    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
            if payload.get("success") is True:
                return {"status": "DESTROYED" if action == "destroy" else "STOPPED", "attempts": attempt}
            last_error = f"unsuccessful response: {payload}"
        except urllib.error.HTTPError as error:
            if error.code == 404:
                # Absence is terminal for both operations: no accelerator can
                # still be billing after the contract has disappeared.  This
                # also makes redundant guardian/supervisor STOP races benign.
                return {
                    "status": "DESTROYED" if action == "destroy" else "STOPPED",
                    "attempts": attempt,
                    "already_absent": True,
                }
            last_error = f"HTTP {error.code}"
        except (OSError, ValueError) as error:
            last_error = type(error).__name__
        if attempt < retries:
            time.sleep(retry_seconds)
    return {"status": "TEARDOWN_FAILED", "attempts": retries, "error": last_error}


class GatewayBackend:
    """Concrete backend executed on the persistent-volume gateway."""

    def __init__(self, contract: dict[str, Any], contract_path: Path):
        self.contract = contract
        self.contract_path = contract_path
        self.policy = validate_runtime_contract(contract)
        h100 = contract["h100"]
        gateway = contract["gateway"]
        self.host = str(h100["ssh_host"])
        self.port = int(h100["ssh_port"])
        self.instance_id = int(h100["instance_id"])
        self.control_path = Path(f"/tmp/parallax-ssh-{self.instance_id}.sock")
        self.ssh_key = Path(h100["ssh_private_key_file"])
        self.api_key_file = Path(h100["instance_api_key_file"])
        self.volume_root = Path(gateway["volume_run_root"])
        self.partial_root = self.volume_root / ".partial"
        self.artifact_root = self.volume_root / "artifacts"
        self.state_root = self.volume_root / "lifecycle"
        self.receipt_path = self.state_root / "receipt.json"
        self.upload_journal_path = self.state_root / "upload-journal.jsonl"
        self.heartbeat_path = self.state_root / "heartbeat.json"
        self.complete_path = self.state_root / "complete.marker"
        self.job_log = self.volume_root / "job.log"
        self._observed: dict[str, tuple[int, str, int]] = {}
        self._synced: dict[str, tuple[int, str, str]] = {}
        self._job: subprocess.Popen[bytes] | None = None
        self._job_log_handle = None
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.partial_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        if self.ssh_key.stat().st_mode & 0o077:
            raise LifecycleError("SSH private key permissions are not 0600")

    @property
    def ssh_base(self) -> list[str]:
        return [
            "ssh", "-i", str(self.ssh_key), "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", "-o", "ControlMaster=auto",
            "-o", "ControlPersist=60", "-o", f"ControlPath={self.control_path}",
            "-p", str(self.port), f"root@{self.host}",
        ]

    @property
    def scp_base(self) -> list[str]:
        return [
            "scp", "-q", "-i", str(self.ssh_key), "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=60",
            "-o", f"ControlPath={self.control_path}",
            "-P", str(self.port),
        ]

    def start_job(self) -> subprocess.Popen[bytes]:
        job = self.contract["job"]
        command = [str(value) for value in job["command"]]
        environment = {str(key): str(value) for key, value in job.get("environment", {}).items()}
        self._job_log_handle = self.job_log.open("ab", buffering=0)
        if job.get("execution_site") == "gateway":
            process_environment = dict(os.environ)
            process_environment.update(environment)
            self._job = subprocess.Popen(
                command,
                cwd=str(job["cwd"]),
                env=process_environment,
                stdout=self._job_log_handle,
                stderr=subprocess.STDOUT,
            )
        elif job.get("execution_site") == "h100":
            exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(environment.items()))
            remote = f"cd {shlex.quote(str(job['cwd']))} && {exports} exec {shlex.join(command)}"
            self._job = subprocess.Popen(
                [*self.ssh_base, remote],
                stdout=self._job_log_handle,
                stderr=subprocess.STDOUT,
            )
        else:
            raise LifecycleError("job.execution_site must be gateway or h100")
        return self._job

    def heartbeat(self, payload: dict[str, Any]) -> None:
        atomic_json(self.heartbeat_path, {**payload, "wall_time_unix": time.time()})

    def probe_worker_ssh(self) -> dict[str, Any]:
        completed = subprocess.run([*self.ssh_base, "true"], check=False, capture_output=True)
        packet = {
            "status": "GREEN" if completed.returncode == 0 else "RED",
            "returncode": completed.returncode,
            "stderr": completed.stderr.decode("utf-8", errors="replace")[-1000:],
        }
        atomic_json(self.state_root / "relay-ssh-probe.json", packet)
        if packet["status"] != "GREEN":
            raise LifecycleError("gateway relay key cannot reach worker before arm")
        return packet

    def _scan(self, *, timeout_seconds: float) -> list[dict[str, Any]]:
        script = """import json,os,pathlib,sys\nrows=[]\nfor item in json.loads(sys.argv[1]):\n root=pathlib.Path(item['remote_root'])\n if not root.exists(): continue\n for path in sorted(p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.') and p.suffix not in {'.tmp','.part','.partial'}):\n  stat=path.stat(); rows.append({'remote':str(path),'relative':str(path.relative_to(root)),'prefix':item['durable_prefix'],'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})\nprint(json.dumps(rows,separators=(',',':')))"""
        roots = self.contract["artifacts"]["roots"]
        remote_command = shlex.join(["python3", "-c", script, json.dumps(roots, separators=(",", ":"))])
        completed = subprocess.run(
            [*self.ssh_base, remote_command],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        rows = json.loads(completed.stdout)
        if not isinstance(rows, list):
            raise LifecycleError("remote artifact scan did not return a list")
        return rows

    def mirror_once(
        self, *, final: bool = False, deadline_monotonic: float | None = None
    ) -> dict[str, Any]:
        pass_deadline = time.monotonic() + self.policy.artifact_copy_timeout_seconds
        if deadline_monotonic is not None:
            pass_deadline = min(pass_deadline, deadline_monotonic)

        def remaining() -> float:
            value = pass_deadline - time.monotonic()
            if value <= 0:
                raise subprocess.TimeoutExpired("artifact mirror pass", 0)
            return value

        copied = []
        errors = []
        try:
            rows = self._scan(timeout_seconds=remaining())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ValueError) as error:
            return {
                "final": final,
                "copied": [],
                "copied_count": 0,
                "errors": [{"stage": "scan", "type": type(error).__name__}],
            }
        for row in rows:
            remote = str(row["remote"])
            size = int(row["bytes"])
            mtime = str(row["mtime_ns"])
            prior = self._observed.get(remote)
            stable_count = prior[2] + 1 if prior and prior[:2] == (size, mtime) else 1
            self._observed[remote] = (size, mtime, stable_count)
            # During execution, require two identical observations so a writer
            # is never copied mid-update.  A final pass runs only after the
            # foreground job exits under an atomic-output/no-background-writer
            # contract, so its first observation is already stable.
            if stable_count < (1 if final else 2):
                continue
            if self._synced.get(remote, ())[:2] == (size, mtime):
                continue
            relative = PurePosixPath(str(row["relative"]))
            prefix = PurePosixPath(str(row["prefix"]))
            if relative.is_absolute() or ".." in relative.parts or prefix.is_absolute() or ".." in prefix.parts:
                raise LifecycleError("unsafe mirrored artifact path")
            destination = self.artifact_root / Path(prefix.as_posix()) / Path(relative.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = self.partial_root / f"{sha256_file(self.contract_path)[:12]}-{len(copied)}-{destination.name}.part"
            try:
                completed = subprocess.run(
                    [*self.scp_base, f"root@{self.host}:{remote}", str(partial)],
                    check=False,
                    capture_output=True,
                    timeout=remaining(),
                )
            except (subprocess.TimeoutExpired, OSError) as error:
                partial.unlink(missing_ok=True)
                errors.append({"stage": "copy", "path": remote, "type": type(error).__name__})
                break
            if completed.returncode != 0:
                partial.unlink(missing_ok=True)
                continue
            if partial.stat().st_size != size:
                partial.unlink(missing_ok=True)
                continue
            digest = sha256_file(partial)
            os.replace(partial, destination)
            self._synced[remote] = (size, mtime, digest)
            copied_row = {
                "path": str(destination.relative_to(self.volume_root)),
                "bytes": size,
                "sha256": digest,
            }
            _append_jsonl(
                self.upload_journal_path,
                {
                    "schema_version": 1,
                    "event": "COPY_COMMITTED",
                    "run_id": self.policy.run_id,
                    "final_pass": final,
                    "remote_path": remote,
                    **copied_row,
                    "wall_time_unix": time.time(),
                },
            )
            copied.append(copied_row)
        return {
            "final": final,
            "copied": copied,
            "copied_count": len(copied),
            "errors": errors,
        }

    def prove_durability(self, *, job_exit_code: int) -> dict[str, Any]:
        missing = [value for value in self.policy.required_sentinels if not (self.artifact_root / value).is_file()]
        if missing:
            return {"status": "UNPROVEN", "missing_sentinels": missing, "job_exit_code": job_exit_code}
        files = [path for path in self.artifact_root.rglob("*") if path.is_file()]
        if not files:
            return {"status": "UNPROVEN", "reason": "no mirrored artifacts", "job_exit_code": job_exit_code}
        manifest = ArtifactManifest(self.artifact_root, self.policy.run_id)
        manifest.add(*files)
        manifest_path = self.state_root / "manifest.json"
        previous_manifest_path = self.state_root / "manifest.previous.json"
        if manifest_path.is_file():
            _atomic_bytes(previous_manifest_path, manifest_path.read_bytes())
        manifest.write(manifest_path)
        errors = ArtifactManifest.verify(manifest_path, self.artifact_root)
        if errors:
            return {"status": "UNPROVEN", "manifest_errors": errors, "job_exit_code": job_exit_code}
        journal_errors: list[str] = []
        committed: dict[str, dict[str, Any]] = {}
        if not self.upload_journal_path.is_file():
            journal_errors.append("upload journal missing")
        else:
            try:
                for line_number, line in enumerate(
                    self.upload_journal_path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    row = json.loads(line)
                    if (
                        row.get("event") == "COPY_COMMITTED"
                        and row.get("run_id") == self.policy.run_id
                    ):
                        committed[str(row.get("path"))] = row
            except (OSError, ValueError) as error:
                journal_errors.append(f"upload journal invalid: {type(error).__name__}")
        expected_paths = set(self.contract.get("expected_artifacts") or {})
        journal_files = (
            [path for path in files if path.relative_to(self.artifact_root).as_posix() in expected_paths]
            if expected_paths
            else files
        )
        for path in journal_files:
            relative = path.relative_to(self.artifact_root).as_posix()
            journal_path = "artifacts/" + relative
            binding = committed.get(journal_path)
            if not binding or (
                int(binding.get("bytes", -1)) != path.stat().st_size
                or str(binding.get("sha256", "")) != sha256_file(path)
            ):
                journal_errors.append(f"upload journal lacks exact commit: {relative}")
        if journal_errors:
            return {
                "status": "UNPROVEN",
                "upload_journal_errors": journal_errors,
                "job_exit_code": job_exit_code,
            }
        expected = self.contract.get("expected_artifacts")
        if expected is not None:
            actual = {
                str(path.relative_to(self.artifact_root)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            }
            mismatches = sorted(
                relative for relative, binding in expected.items()
                if actual.get(relative) != binding
            )
            unexpected = sorted(set(actual) - set(expected))
            if mismatches or unexpected:
                return {
                    "status": "UNPROVEN",
                    "job_exit_code": job_exit_code,
                    "expected_artifact_mismatches": mismatches,
                    "unexpected_artifacts": unexpected,
                }
        os.sync()
        return {
            "status": "GREEN",
            "job_exit_code": job_exit_code,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "previous_manifest_sha256": (
                sha256_file(previous_manifest_path) if previous_manifest_path.is_file() else None
            ),
            "upload_journal": str(self.upload_journal_path),
            "upload_journal_sha256": sha256_file(self.upload_journal_path),
            "entry_count": len(files),
            "total_bytes": manifest.total_bytes,
        }

    def destroy_h100(self) -> dict[str, Any]:
        key = _read_secret(self.api_key_file)
        atomic_json(
            self.state_root / "worker-stop-intent.json",
            {
                "schema_version": 1,
                "status": "DURABILITY_GREEN_STOP_PENDING",
                "run_id": self.policy.run_id,
                "worker_id": self.instance_id,
                "wall_time_unix": time.time(),
            },
        )
        os.sync()
        # End compute billing first.  Destruction is the storage cleanup step,
        # and may be eventually consistent even after a successful response.
        # The independent guardian watches the durable intent above and races
        # this request, so either process can end billing on normal completion.
        stopped = vast_instance_action(instance_id=self.instance_id, api_key=key, action="stop")
        atomic_json(
            self.state_root / "worker-stop-intent.json",
            {
                "schema_version": 1,
                "status": "STOP_CONFIRMED_DESTROY_PENDING" if stopped.get("status") == "STOPPED" else "STOP_UNCONFIRMED",
                "run_id": self.policy.run_id,
                "worker_id": self.instance_id,
                "stop": stopped,
                "wall_time_unix": time.time(),
            },
        )
        os.sync()
        destroyed = vast_instance_action(instance_id=self.instance_id, api_key=key, action="destroy")
        if destroyed.get("status") == "DESTROYED":
            return {"status": "DESTROYED", "stop": stopped, "destroy": destroyed}
        if stopped.get("status") == "STOPPED":
            return {"status": "STOPPED_AFTER_DESTROY_FAILURE", "stop": stopped, "destroy": destroyed}
        return {"status": "TEARDOWN_FAILED", "stop": stopped, "destroy": destroyed}

    def stop_h100(self, *, reason: str) -> dict[str, Any]:
        return {**vast_instance_action(instance_id=self.instance_id, api_key=_read_secret(self.api_key_file), action="stop"), "reason": reason}

    def write_receipt(self, payload: dict[str, Any]) -> None:
        atomic_json(self.receipt_path, payload)
        teardown_status = payload.get("teardown", {}).get("status")
        if teardown_status in {"DESTROYED", "STOPPED", "STOPPED_AFTER_DESTROY_FAILURE"}:
            _atomic_text(self.complete_path, payload["status"] + "\n")
        os.sync()
        if self._job_log_handle is not None:
            self._job_log_handle.close()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def monotonic(self) -> float:
        return time.monotonic()


def watchdog(
    contract: dict[str, Any],
    contract_path: Path,
    *,
    action=vast_instance_action,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
) -> dict[str, Any]:
    policy = validate_runtime_contract(contract)
    h100 = contract["h100"]
    root = Path(contract["gateway"]["volume_run_root"]) / "lifecycle"
    heartbeat_path = root / "heartbeat.json"
    complete_path = root / "complete.marker"
    receipt_path = root / "watchdog.json"
    api_key_file = Path(h100["instance_api_key_file"])
    started = monotonic()
    hard_deadline = (
        started
        + policy.execution_release_timeout_seconds
        + policy.maximum_runtime_seconds
        + policy.heartbeat_timeout_seconds
    )
    while True:
        if complete_path.is_file():
            packet = {"status": "EXIT_GREEN_SUPERVISOR_COMPLETE", "run_id": policy.run_id}
            atomic_json(receipt_path, packet)
            return packet
        heartbeat_age = monotonic() - started
        if heartbeat_path.is_file():
            heartbeat_age = max(0.0, wall_time() - heartbeat_path.stat().st_mtime)
        if heartbeat_age > policy.heartbeat_timeout_seconds or monotonic() >= hard_deadline:
            result = action(
                instance_id=int(h100["instance_id"]),
                api_key=_read_secret(api_key_file),
                action="stop",
            )
            packet = {
                "status": "H100_STOP_REQUESTED_DISK_PRESERVED",
                "run_id": policy.run_id,
                "reason": "stale_heartbeat" if heartbeat_age > policy.heartbeat_timeout_seconds else "hard_deadline",
                "heartbeat_age_seconds": heartbeat_age,
                "teardown": result,
            }
            atomic_json(receipt_path, packet)
            os.sync()
            if result.get("status") == "STOPPED":
                gateway = contract.get("gateway", {})
                if gateway.get("self_destroy") is True:
                    gateway_key_file = Path(gateway["instance_api_key_file"])
                    gateway_key = _read_secret(gateway_key_file)
                    atomic_json(
                        root / "gateway-teardown-intent.json",
                        {
                            "status": "WATCHDOG_GATEWAY_DESTROY_REQUEST_PENDING",
                            "run_id": policy.run_id,
                            "gateway_instance_id": int(gateway["instance_id"]),
                        },
                    )
                    os.sync()
                    action(
                        instance_id=int(gateway["instance_id"]),
                        api_key=gateway_key,
                        action="destroy",
                    )
                return packet
            # Keep the independent process alive and retry.  The provider TTL
            # remains the final backstop if the API is unreachable throughout.
            sleep(min(policy.mirror_interval_seconds, 2.0))
            continue
        sleep(min(policy.mirror_interval_seconds, 2.0))


def wait_for_execution_release(
    marker: Path,
    *,
    timeout_seconds: float,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> bool:
    """Wait for the launcher's explicit phase-two release under a hard bound."""
    deadline = monotonic() + timeout_seconds
    while not marker.is_file():
        if monotonic() >= deadline:
            return False
        sleep(min(0.1, max(0.0, deadline - monotonic())))
    return True


def run_gateway(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    policy = validate_runtime_contract(contract)
    lock_path = Path(contract["gateway"]["volume_run_root"]) / "lifecycle/supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LifecycleError(f"supervisor already owns run {policy.run_id}") from error
        backend = GatewayBackend(contract, contract_path)
        backend.probe_worker_ssh()
        backend.heartbeat({
            "status": "ARMED_WAITING_RELEASE",
            "run_id": policy.run_id,
            "wall_time_unix": time.time(),
        })
        watchdog_log = lock_path.parent / "watchdog.log"
        with watchdog_log.open("ab", buffering=0) as log:
            subprocess.Popen(
                [sys.executable, "-m", "infra.vast.parallax.lifecycle_supervisor", "watchdog", "--contract", str(contract_path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        heartbeat_stop = threading.Event()
        execution_released = threading.Event()

        def heartbeat_worker() -> None:
            while not heartbeat_stop.wait(min(policy.mirror_interval_seconds, 1.0)):
                backend.heartbeat(
                    {
                        "status": "RUNNING" if execution_released.is_set() else "ARMED_WAITING_RELEASE",
                        "run_id": policy.run_id,
                        "wall_time_unix": time.time(),
                    }
                )

        heartbeat_thread = threading.Thread(
            target=heartbeat_worker,
            name=f"parallax-heartbeat-{policy.run_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            release_marker = backend.state_root / "execution-release.marker"
            if wait_for_execution_release(
                release_marker,
                timeout_seconds=policy.execution_release_timeout_seconds,
            ):
                execution_released.set()
                packet = supervise(policy, backend)
            else:
                teardown = backend.stop_h100(reason="execution_release_timeout")
                packet = {
                    "schema_version": 1,
                    "run_id": policy.run_id,
                    "status": "STOPPED_EXECUTION_NEVER_RELEASED_DISK_PRESERVED",
                    "job_exit_code": None,
                    "durability": {"status": "UNPROVEN", "reason": "execution_release_timeout"},
                    "teardown": teardown,
                    "mirror_passes": [],
                    "elapsed_seconds": policy.execution_release_timeout_seconds,
                }
                backend.write_receipt(packet)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.0)
        gateway = contract["gateway"]
        terminal_teardown = packet.get("teardown", {}).get("status") in {
            "DESTROYED",
            "STOPPED",
            "STOPPED_AFTER_DESTROY_FAILURE",
        }
        receipt_is_durable = backend.receipt_path.is_file()
        if receipt_is_durable:
            persisted = json.loads(backend.receipt_path.read_text(encoding="utf-8"))
            receipt_is_durable = persisted.get("run_id") == policy.run_id and persisted.get("status") == packet.get("status")
        if gateway.get("self_destroy") is True and terminal_teardown and receipt_is_durable:
            gateway_key_path = Path(gateway["instance_api_key_file"])
            gateway_key = _read_secret(gateway_key_path)
            for secret_path in (
                Path(contract["h100"]["instance_api_key_file"]),
                Path(contract["h100"]["ssh_private_key_file"]),
                gateway_key_path,
            ):
                secret_path.unlink(missing_ok=True)
            intent = {
                "status": "GATEWAY_DESTROY_REQUEST_PENDING",
                "run_id": policy.run_id,
                "gateway_instance_id": int(gateway["instance_id"]),
            }
            atomic_json(lock_path.parent / "gateway-teardown-intent.json", intent)
            os.sync()
            # No result can be written after this succeeds; the intent on the
            # attached volume is therefore the terminal local record.
            destroyed = vast_instance_action(
                instance_id=int(gateway["instance_id"]),
                api_key=gateway_key,
                action="destroy",
            )
            if destroyed.get("status") != "DESTROYED":
                atomic_json(
                    lock_path.parent / "gateway-teardown-intent.json",
                    {**intent, "status": "GATEWAY_DESTROY_FAILED_STOP_PENDING", "destroy": destroyed},
                )
                os.sync()
                vast_instance_action(
                    instance_id=int(gateway["instance_id"]),
                    api_key=gateway_key,
                    action="stop",
                )
        return packet


def verify_persisted_run(run_root: Path) -> dict[str, Any]:
    """Verify a run after the volume has been detached and re-attached."""
    manifest_path = run_root / "lifecycle/manifest.json"
    previous_manifest_path = run_root / "lifecycle/manifest.previous.json"
    upload_journal_path = run_root / "lifecycle/upload-journal.jsonl"
    receipt_path = run_root / "lifecycle/receipt.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        errors.append("manifest missing after reattach")
    if not receipt_path.is_file():
        errors.append("receipt missing after reattach")
    receipt: dict[str, Any] = {}
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors.append("receipt is not valid JSON")
    if manifest_path.is_file():
        try:
            errors.extend(ArtifactManifest.verify(manifest_path, run_root / "artifacts"))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"manifest verification error: {type(error).__name__}")
    if receipt.get("durability", {}).get("status") != "GREEN":
        errors.append("receipt durability is not GREEN")
    durability = receipt.get("durability", {})
    if manifest_path.is_file() and durability.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("receipt does not bind the current manifest")
    expected_previous_sha256 = durability.get("previous_manifest_sha256")
    if expected_previous_sha256 is not None and (
        not previous_manifest_path.is_file()
        or sha256_file(previous_manifest_path) != expected_previous_sha256
    ):
        errors.append("receipt does not bind the previous manifest generation")
    if not upload_journal_path.is_file():
        errors.append("upload journal missing after reattach")
    elif durability.get("upload_journal_sha256") != sha256_file(upload_journal_path):
        errors.append("receipt does not bind the upload journal")
    if receipt.get("teardown", {}).get("status") not in {
        "DESTROYED",
        "STOPPED_AFTER_DESTROY_FAILURE",
    }:
        errors.append("worker teardown is not terminal")
    packet = {
        "schema_version": 1,
        "status": "GREEN" if not errors else "RED",
        "run_id": receipt.get("run_id"),
        "volume_reattach_verified": True,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "previous_manifest_sha256": (
            sha256_file(previous_manifest_path) if previous_manifest_path.is_file() else None
        ),
        "upload_journal_sha256": (
            sha256_file(upload_journal_path) if upload_journal_path.is_file() else None
        ),
        "receipt_sha256": sha256_file(receipt_path) if receipt_path.is_file() else None,
        "errors": errors,
    }
    atomic_json(run_root / "lifecycle/reattach-verification.json", packet)
    os.sync()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    supervise_parser = subparsers.add_parser("supervise")
    supervise_parser.add_argument("--contract", type=Path, required=True)
    watchdog_parser = subparsers.add_parser("watchdog")
    watchdog_parser.add_argument("--contract", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-volume")
    verify_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "supervise":
        contract_path = args.contract.resolve()
        packet = run_gateway(contract_path)
    elif args.command == "watchdog":
        contract_path = args.contract.resolve()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        packet = watchdog(contract, contract_path)
    else:
        packet = verify_persisted_run(args.run_root.resolve())
    print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
