#!/usr/local/bin/python3
"""Stdlib-first supervised PG1 production bootstrap.

No project module is imported until a trusted minimal stop context has been
parsed, a pointer has been preserved, and the guardian process has started.
Setup/import/materialization failures publish the canonical marker through the
fixed AWS path and keep supervising/restarting the guardian until a durable
verified STOP terminal receipt appears.  A supervision deadline transitions
to STOP-required and begins another bounded cycle; it never exits unverified.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


PYTHON_EXECUTABLE = Path("/usr/local/bin/python3")
AWS_EXECUTABLE = Path("/usr/local/bin/aws")
CAPSULE_RUNTIME_PATH = Path("/run/parallax/pg1-execution-capsule.json")
STOP_CONTEXT_RUNTIME_PATH = Path("/run/parallax/pg1-native-stop-context.json")
ENTRY_CONTRACT_RUNTIME_PATH = Path("/run/parallax/pg1-native-entry-contract.json")
BOOTSTRAP_SOURCE_ROOT = Path("/opt/parallax/future-freeze/source")
IMMUTABLE_IMAGE_SOURCE_ROOT = Path("/opt/parallax/image/source")
STAGED_ROOT = Path("/workspace/parallax/source")
WORKSPACE_ROOT = Path("/workspace/parallax")
FAILURE_POINTER_PATH = WORKSPACE_ROOT / "failure-stop-pointer.json"
GUARDIAN_RESULT_PATH = WORKSPACE_ROOT / "pg1-native-guardian-result.json"
GUARDIAN_HEARTBEAT_PATH = WORKSPACE_ROOT / "pg1-native-guardian-heartbeat.json"
BOOTSTRAP_HEARTBEAT_PATH = WORKSPACE_ROOT / "pg1-native-bootstrap-heartbeat.json"
AUTHORIZATION_BUCKET = "pg-lineage-769063704696"
AUTHORIZATION_PREFIX = "parallax/authorizations/pg1-native-v3"
EXPECTED_BUCKET_OWNER = "769063704696"
PROVIDER_TIMEOUT_SECONDS = 30
SUPERVISOR_TIMEOUT_SECONDS = 10_800 + 300
SUPERVISOR_POLL_SECONDS = 2


class BootstrapError(RuntimeError):
    pass


def _entry_contract(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o400:
        raise BootstrapError("bootstrap entry contract is absent or mutable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError("bootstrap entry contract unreadable") from error
    if (
        not isinstance(value, dict)
        or (value.get("schema"), value.get("kind")) not in {
            ("pg1-native-entry-dispatch-contract/v1", "PRODUCTION_AUTHORITY_BOUND"),
            ("pg1-native-entry-dispatch-contract/v1", "MECHANICAL_NO_AUTHORITY"),
            ("pg1-native-entry-dispatch-contract/v2",
             "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"),
        }
        or value.get("may_mint_authorization") is not False
    ):
        raise BootstrapError("bootstrap entry contract kind changed")
    return value


def _mechanical_entry_from_source() -> tuple[dict[str, Any], Path] | None:
    """Derive offline dispatch only from copied bootstrap bytes."""

    source = Path(__file__).resolve()
    canonical = {
        Path("/opt/parallax/image/source/infra/vast/parallax/pg1_native_bootstrap.py"),
        Path("/opt/parallax/future-freeze/source/infra/vast/parallax/pg1_native_bootstrap.py"),
    }
    if source in canonical:
        return None
    if not any(
        f"/{marker}/" in str(source)
        for marker in ("opt/parallax/image/source", "opt/parallax/future-freeze/source")
    ):
        raise BootstrapError("bootstrap source is outside canonical or copied layout")
    module_path = source.parent / "pg1_native_mechanical_rebase.py"
    spec = importlib.util.spec_from_file_location(
        "_pg1_native_copied_mechanical_rebase", module_path
    )
    if spec is None or spec.loader is None:
        raise BootstrapError("copied mechanical source contract is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        discovered = module.worker_contract_from_source(source)
    except Exception as error:
        raise BootstrapError("copied bootstrap source contract changed") from error
    if discovered is None:
        raise BootstrapError("copied bootstrap resolved as canonical source")
    _root_contract, _root_contract_path, worker = discovered
    entry_path = worker / "run/parallax/pg1-native-entry-contract.json"
    contract = _entry_contract(entry_path)
    if (
        contract.get("kind") == "PRODUCTION_AUTHORITY_BOUND"
        or Path(str(contract.get("rebase_root", ""))).resolve() != worker
        or Path(str(contract.get("image_root", ""))).resolve()
        != worker / "opt/parallax/image/source"
        or Path(str(contract.get("bootstrap_root", ""))).resolve()
        != worker / "opt/parallax/future-freeze/source"
    ):
        raise BootstrapError("copied bootstrap entry contract changed")
    return contract, entry_path.resolve()


def _mechanical_dispatch(contract: dict[str, Any], argv: list[str]) -> int:
    worker_argv = contract.get("worker_argv")
    worker_receipt_path = Path(str(contract.get("worker_receipt_path", "")))
    worker_enabled = contract.get("kind") == "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"
    if (
        (contract.get("schema"), contract.get("kind")) not in {
            ("pg1-native-entry-dispatch-contract/v1", "MECHANICAL_NO_AUTHORITY"),
            ("pg1-native-entry-dispatch-contract/v2",
             "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"),
        }
        or contract.get("dispatch_scope") != "EXACT_REBASED_BYTES_ONLY"
        or contract.get("production_authority") is not False
        or contract.get("bootstrap_argv") != argv
        or not isinstance(contract.get("runtime_wiring"), dict)
        or worker_enabled and (
            not isinstance(worker_argv, list)
            or not worker_argv or not all(isinstance(item, str) for item in worker_argv)
            or worker_receipt_path.is_symlink()
            or "worker_argv" not in contract or "worker_receipt_path" not in contract
        )
    ):
        raise BootstrapError("mechanical bootstrap contract escaped its scope")
    relative = "infra/vast/parallax/pg1_native_bootstrap.py"
    expected = contract["runtime_wiring"].get(relative)
    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected != digest or source != Path(str(contract["bootstrap_root"])) / relative:
        raise BootstrapError("executed bootstrap bytes differ from capsule wiring")
    worker_receipt: dict[str, Any] | None = None
    if worker_enabled:
        completed = subprocess.run(
            [__import__("sys").executable, *worker_argv], check=False,
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(str(contract["bootstrap_root"]))),
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "PYTHONPATH": str(contract["bootstrap_root"]),
            }, close_fds=True,
        )
        if completed.returncode != 0 or not worker_receipt_path.is_file():
            raise BootstrapError("exact copied worker bytes exited nonzero")
        try:
            worker_receipt = json.loads(worker_receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BootstrapError("exact worker receipt unreadable") from error
        if (
            worker_receipt.get("status") != "EXACT_COPIED_WORKER_BYTES_EXECUTED"
            or worker_receipt.get("argv") != worker_argv
            or worker_receipt.get("provider_calls") != 0
            or worker_receipt.get("production_authority") is not False
        ):
            raise BootstrapError("exact worker execution receipt changed")
    receipt = {
        "schema": "pg1-native-exact-bootstrap-execution/v1",
        "status": "EXACT_COPIED_BOOTSTRAP_BYTES_EXECUTED",
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"],
        "bootstrap_path": str(source),
        "bootstrap_sha256": digest,
        "argv": argv,
        "production_authority": False,
        "may_mint_authorization": False,
        "provider_calls": 0,
    }
    if worker_receipt is not None:
        receipt.update({
            "worker_status": worker_receipt["status"],
            "worker_sha256": worker_receipt["worker_sha256"],
            "worker_receipt_sha256": hashlib.sha256(
                worker_receipt_path.read_bytes()
            ).hexdigest(),
        })
    _atomic_json(Path(str(contract["receipt_path"])), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _fixed_env() -> dict[str, str]:
    guardian_source = (
        BOOTSTRAP_SOURCE_ROOT
        if (BOOTSTRAP_SOURCE_ROOT / "infra/vast/parallax/pg1_native_guardian.py").is_file()
        else IMMUTABLE_IMAGE_SOURCE_ROOT
    )
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "AWS_PROFILE": "fuelos",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": "/run/parallax/aws/config",
        "AWS_SHARED_CREDENTIALS_FILE": "/run/parallax/aws/credentials",
        "AWS_EC2_METADATA_DISABLED": "true",
        "PYTHONPATH": str(guardian_source),
    }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_minimal_context(path: Path = STOP_CONTEXT_RUNTIME_PATH) -> dict[str, Any]:
    if (
        path != STOP_CONTEXT_RUNTIME_PATH
        or path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o400
    ):
        raise BootstrapError("fixed stop context is absent before bootstrap imports")
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError("fixed stop context is unreadable before bootstrap imports") from error
    if not isinstance(context, dict) or set(context) != {
        "schema_version", "status", "content_address", "contract", "issuance"
    }:
        raise BootstrapError("minimal stop context envelope changed")
    contract = context.get("contract")
    address = context.get("content_address")
    if (
        context.get("schema_version") != 1
        or context.get("status") != "ISSUED_FOR_ONE_FUTURE_PRODUCTION_INSTANCE"
        or not isinstance(contract, dict)
        or address != {"algorithm": "sha256", "digest": _digest(contract)}
        or contract.get("schema") != "pg1-native-production-stop-context/v1"
        or contract.get("issuer") != "PG1_PRODUCTION_DEPLOYMENT_CONTEXT_ISSUER_V1"
        or contract.get("issuer_account") != EXPECTED_BUCKET_OWNER
        or isinstance(contract.get("instance_id"), bool)
        or not isinstance(contract.get("instance_id"), int)
        or contract["instance_id"] <= 0
        or not isinstance(contract.get("capsule_sha256"), str)
        or len(contract["capsule_sha256"]) != 64
        or not isinstance(contract.get("attempt_run_id"), str)
        or not contract["attempt_run_id"]
        or contract.get("context_key")
        != (
            f"{AUTHORIZATION_PREFIX}/deployment-context/"
            f"{contract.get('capsule_sha256')}/{contract.get('attempt_run_id')}.json"
        )
    ):
        raise BootstrapError("minimal stop context identity changed")
    for field in ("offer_id", "machine_id"):
        if (
            isinstance(contract.get(field), bool)
            or not isinstance(contract.get(field), int)
            or contract[field] <= 0
        ):
            raise BootstrapError(f"minimal stop context {field} changed")
    if not isinstance(contract.get("container_id"), str) or not contract["container_id"]:
        raise BootstrapError("minimal stop context container identity changed")
    return context


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _untrusted_context_failure_pointer(error: BaseException) -> None:
    _write_exclusive(
        FAILURE_POINTER_PATH,
        {
            "schema": "pg1-native-bootstrap-untrusted-context-failure/v2",
            "status": "POINTER_PRESERVED_NO_UNTRUSTED_PROVIDER_ACTION",
            "failure": type(error).__name__,
            "instance_id": "FORBIDDEN_UNTRUSTED",
        },
    )


def _read_lifecycle(context: dict[str, Any]) -> dict[str, Any] | None:
    module = importlib.import_module("infra.vast.parallax.pg1_native_lifecycle")
    try:
        return module.read(FAILURE_POINTER_PATH, context)
    except module.LifecycleCASFailure as error:
        raise BootstrapError(str(error)) from error


def _transition_lifecycle(
    context: dict[str, Any], state: str, *, reason: str
) -> dict[str, Any]:
    module = importlib.import_module("infra.vast.parallax.pg1_native_lifecycle")
    for _attempt in range(8):
        prior = _read_lifecycle(context)
        try:
            return module.transition(
                FAILURE_POINTER_PATH,
                context,
                state,
                reason=reason,
                expected_sequence=0 if prior is None else int(prior["sequence"]),
            )
        except module.LifecycleCASFailure as error:
            if "sequence mismatch" not in str(error):
                raise BootstrapError(str(error)) from error
    raise BootstrapError("bootstrap lifecycle CAS retry budget exhausted")


def _pointer_first(context: dict[str, Any], failure: str) -> None:
    """Compatibility wrapper: writing a pointer always means STOP is required."""

    _transition_lifecycle(context, "FAILURE_STOP_REQUIRED", reason=failure)


def _marker(context: dict[str, Any], failure: str) -> dict[str, Any]:
    contract = context["contract"]
    pointer_sha = (
        hashlib.sha256(FAILURE_POINTER_PATH.read_bytes()).hexdigest()
        if FAILURE_POINTER_PATH.is_file()
        else None
    )
    return {
        "schema": "pg1-native-stop-guardian-request/v3",
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"],
        "stop_context_sha256": context["content_address"]["digest"],
        "instance_id": contract["instance_id"],
        "offer_id": contract["offer_id"],
        "machine_id": contract["machine_id"],
        "container_id": contract["container_id"],
        "failure": failure,
        "failure_pointer": str(FAILURE_POINTER_PATH),
        "failure_pointer_sha256": pointer_sha,
        "artifact_manifest": str(WORKSPACE_ROOT / "artifacts.manifest.json"),
        "scratch_preserved": True,
    }


def _aws_prefix() -> list[str]:
    return [
        str(AWS_EXECUTABLE), "--profile", "fuelos", "--region", "us-east-1"
    ]


def _marker_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/stop-guardian/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def _heartbeat_key(context: dict[str, Any]) -> str:
    contract = context["contract"]
    return (
        f"{AUTHORIZATION_PREFIX}/deadman-heartbeats/{contract['capsule_sha256']}/"
        f"{contract['attempt_run_id']}.json"
    )


def _marker_metadata(context: dict[str, Any]) -> dict[str, str]:
    contract = context["contract"]
    return {
        "capsule-sha256": contract["capsule_sha256"],
        "attempt-run-id": contract["attempt_run_id"],
        "instance-id": str(contract["instance_id"]),
        "object-kind": "stop-guardian-request",
    }


def _run_json(command: list[str]) -> Any:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROVIDER_TIMEOUT_SECONDS,
            env=_fixed_env(),
        )
    except subprocess.TimeoutExpired as error:
        raise BootstrapError("fixed bootstrap AWS command timed out") from error
    if completed.returncode != 0:
        raise BootstrapError("fixed bootstrap AWS command failed")
    try:
        return json.loads(completed.stdout or "")
    except json.JSONDecodeError as error:
        raise BootstrapError("fixed bootstrap AWS response is malformed") from error


def _verify_download(
    response: Any, data: bytes, context: dict[str, Any]
) -> dict[str, Any]:
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    if (
        not isinstance(response, dict)
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("VersionId")
        or not response.get("ETag")
        or response.get("ContentLength") != len(data)
        or response.get("ContentType") != "application/json"
        or response.get("Metadata") != _marker_metadata(context)
    ):
        raise BootstrapError("bootstrap guardian marker readback changed")
    return response


def _publish_marker(context: dict[str, Any], failure: str) -> dict[str, Any]:
    """Publish canonical fallback without loading or referencing the Vast key."""

    marker = _marker(context, failure)
    data = _canonical_bytes(marker)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="pg1-bootstrap-marker-") as directory:
        body = Path(directory) / "marker.json"
        body.write_bytes(data)
        put = [
            *_aws_prefix(), "s3api", "put-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", _marker_key(context),
            "--body", str(body),
            "--if-none-match", "*",
            "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum,
            "--content-length", str(len(data)),
            "--content-type", "application/json",
            "--server-side-encryption", "AES256",
            "--metadata", json.dumps(
                _marker_metadata(context), sort_keys=True, separators=(",", ":")
            ),
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        ]
        try:
            response = _run_json(put)
            if (
                not isinstance(response, dict)
                or response.get("ChecksumSHA256") != checksum
                or response.get("ServerSideEncryption") != "AES256"
                or not response.get("VersionId")
                or not response.get("ETag")
            ):
                raise BootstrapError("bootstrap marker upload evidence changed")
            replay = False
        except Exception:
            target = Path(directory) / "existing.json"
            response = _run_json(
                [
                    *_aws_prefix(), "s3api", "get-object",
                    "--bucket", AUTHORIZATION_BUCKET,
                    "--key", _marker_key(context),
                    "--checksum-mode", "ENABLED",
                    "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
                    str(target),
                ]
            )
            existing = target.read_bytes()
            if existing != data:
                raise BootstrapError("bootstrap marker replay body changed")
            _verify_download(response, existing, context)
            replay = True
    return {
        "status": "CANONICAL_GUARDIAN_MARKER_DURABLE",
        "key": _marker_key(context),
        "version_id": str(response["VersionId"]),
        "replayed_exact": replay,
    }


def _publish_supervisor_heartbeat(
    context: dict[str, Any], *, sequence: int, status: str
) -> dict[str, Any]:
    contract = context["contract"]
    value = {
        "schema": "pg1-native-bootstrap-supervisor-heartbeat/v1",
        "status": status,
        "sequence": sequence,
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"],
        "instance_id": contract["instance_id"],
        "stop_context_sha256": context["content_address"]["digest"],
        "wall_time_unix": time.time(),
        "destroy_authority": False,
    }
    _atomic_json(BOOTSTRAP_HEARTBEAT_PATH, value)
    data = _canonical_bytes(value)
    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="pg1-bootstrap-heartbeat-") as directory:
        body = Path(directory) / "heartbeat.json"
        body.write_bytes(data)
        response = _run_json(
            [
                *_aws_prefix(), "s3api", "put-object",
                "--bucket", AUTHORIZATION_BUCKET,
                "--key", _heartbeat_key(context),
                "--body", str(body),
                "--checksum-algorithm", "SHA256",
                "--checksum-sha256", checksum,
                "--content-length", str(len(data)),
                "--content-type", "application/json",
                "--server-side-encryption", "AES256",
                "--metadata", json.dumps(
                    _expected_metadata(context, kind="bootstrap-supervisor-heartbeat"),
                    sort_keys=True, separators=(",", ":")
                ),
                "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            ]
        )
    if (
        not isinstance(response, dict)
        or response.get("ChecksumSHA256") != checksum
        or response.get("ServerSideEncryption") != "AES256"
        or not response.get("VersionId")
        or not response.get("ETag")
    ):
        raise BootstrapError("bootstrap supervisor heartbeat upload unverified")
    return value


def guardian_command(_capsule: dict[str, Any] | None = None) -> list[str]:
    return [
        str(PYTHON_EXECUTABLE),
        "-m",
        "infra.vast.parallax.pg1_native_guardian",
        "--guard",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--stop-context",
        str(STOP_CONTEXT_RUNTIME_PATH),
    ]


def _start_guardian() -> subprocess.Popen[str]:
    return subprocess.Popen(
        guardian_command(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_fixed_env(),
        start_new_session=True,
        close_fds=True,
        text=True,
    )


def _terminal_receipt(context: dict[str, Any]) -> dict[str, Any] | None:
    if not GUARDIAN_RESULT_PATH.is_file():
        return None
    try:
        value = json.loads(GUARDIAN_RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError("guardian terminal receipt is unreadable") from error
    contract = context["contract"]
    if (
        not isinstance(value, dict)
        or value.get("schema") != "pg1-native-guardian-result/v2"
        or value.get("capsule_sha256") != contract["capsule_sha256"]
        or value.get("attempt_run_id") != contract["attempt_run_id"]
        or value.get("instance_id") != contract["instance_id"]
        or value.get("terminal_state") != "stopped"
        or value.get("receipt_durable") is not True
    ):
        raise BootstrapError("guardian terminal receipt identity changed")
    return value


def _supervise_guardian(
    guardian: subprocess.Popen[str] | None,
    context: dict[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    current = guardian
    sequence = 0
    guardian_started = time.monotonic()
    while True:
        receipt = _terminal_receipt(context)
        if receipt is not None:
            return receipt
        sequence += 1
        try:
            _publish_supervisor_heartbeat(
                context, sequence=sequence, status="SUPERVISING_LOCAL_GUARDIAN"
            )
        except BaseException:
            # The external deadman treats a stale/missing S3 heartbeat as STOP.
            pass
        heartbeat_ok = False
        if GUARDIAN_HEARTBEAT_PATH.is_file():
            try:
                heartbeat = json.loads(
                    GUARDIAN_HEARTBEAT_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                heartbeat = None
            contract = context["contract"]
            heartbeat_ok = bool(
                isinstance(heartbeat, dict)
                and heartbeat.get("capsule_sha256") == contract["capsule_sha256"]
                and heartbeat.get("attempt_run_id") == contract["attempt_run_id"]
                and heartbeat.get("instance_id") == contract["instance_id"]
                and heartbeat.get("destroy_authority") is False
                and isinstance(heartbeat.get("wall_time_unix"), (int, float))
                and time.time() - float(heartbeat["wall_time_unix"]) <= 30
            )
        if (
            current is None
            or current.poll() is not None
            or (time.monotonic() - guardian_started > 30 and not heartbeat_ok)
        ):
            if current is not None and current.poll() is None:
                current.terminate()
            current = _start_guardian()
            guardian_started = time.monotonic()
        if time.monotonic() >= deadline:
            _pointer_first(context, "GUARDIAN_SUPERVISOR_DEADLINE_STOP_REQUIRED")
            deadline = time.monotonic() + SUPERVISOR_TIMEOUT_SECONDS
        time.sleep(float(SUPERVISOR_POLL_SECONDS))


def _load_minimal_capsule(path: Path = CAPSULE_RUNTIME_PATH) -> dict[str, Any]:
    if path != CAPSULE_RUNTIME_PATH or path.is_symlink() or not path.is_file():
        raise BootstrapError("fixed capsule is absent")
    try:
        capsule = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError("fixed capsule is unreadable") from error
    contract = capsule.get("contract") if isinstance(capsule, dict) else None
    if (
        not isinstance(contract, dict)
        or capsule.get("content_address")
        != {"algorithm": "sha256", "digest": _digest(contract)}
    ):
        raise BootstrapError("minimal capsule content address changed")
    return capsule


def materialize_staged_runtime(*, capsule: dict[str, Any]) -> dict[str, Any]:
    capsule_module = importlib.import_module("infra.vast.parallax.pg1_execution_capsule")
    artifacts_module = importlib.import_module("infra.vast.parallax.artifacts")
    CapsuleError = capsule_module.CapsuleError
    safe_relative = capsule_module.safe_relative
    sha256_file = artifacts_module.sha256_file
    if STAGED_ROOT.exists():
        raise CapsuleError("refusing to merge or overwrite staged production root")
    if BOOTSTRAP_SOURCE_ROOT.is_symlink() or not BOOTSTRAP_SOURCE_ROOT.is_dir():
        raise CapsuleError("fixed future-freeze source root is absent")
    identities: dict[str, str] = {}
    for row in capsule["contract"]["artifacts"].values():
        identities[str(safe_relative(row["path"], label="bootstrap artifact"))] = row[
            "sha256"
        ]
    identities.update(capsule["contract"]["runtime_wiring"])
    copied: list[dict[str, Any]] = []
    try:
        for relative_text, expected_sha256 in sorted(identities.items()):
            relative = safe_relative(relative_text, label="bootstrap source")
            source = BOOTSTRAP_SOURCE_ROOT / relative
            target = STAGED_ROOT / relative
            if source.is_symlink() or not source.is_file() or sha256_file(source) != expected_sha256:
                raise CapsuleError(f"fixed bootstrap source body changed: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.chmod(target, 0o400)
            if sha256_file(target) != expected_sha256:
                raise CapsuleError(f"materialized bootstrap body changed: {relative}")
            copied.append(
                {"path": str(relative), "sha256": expected_sha256, "bytes": target.stat().st_size}
            )
    except Exception:
        shutil.rmtree(STAGED_ROOT, ignore_errors=True)
        raise
    return {
        "schema": "pg1-native-production-bootstrap-materialization/v2",
        "status": "GREEN_EXACT_FUTURE_FREEZE_RUNTIME_MATERIALIZED",
        "capsule_sha256": capsule["content_address"]["digest"],
        "source_root": str(BOOTSTRAP_SOURCE_ROOT),
        "staged_root": str(STAGED_ROOT),
        "files": copied,
        "file_count": len(copied),
        "data_image_bodies_read": 0,
        "confirmation_items_read": 0,
    }


def _native_command(capsule: dict[str, Any]) -> list[str]:
    expected = [
        str(PYTHON_EXECUTABLE), "-m", "infra.vast.parallax.pg1_native_launcher",
        "--execute", "--capsule", str(CAPSULE_RUNTIME_PATH),
    ]
    if capsule["contract"]["entrypoints"].get("native_launcher_argv") != expected:
        raise BootstrapError("native launcher argv changed before bootstrap dispatch")
    return expected


def _run_native_supervised(
    capsule: dict[str, Any],
    context: dict[str, Any],
    guardian: subprocess.Popen[str],
    *,
    deadline: float,
) -> dict[str, Any]:
    environment = _fixed_env()
    environment["PYTHONPATH"] = str(STAGED_ROOT)
    process = subprocess.Popen(
        _native_command(capsule),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        close_fds=True,
        text=True,
    )
    current_guardian = guardian
    sequence = 0
    guardian_started = time.monotonic()
    while process.poll() is None and time.monotonic() < deadline:
        receipt = _terminal_receipt(context)
        if receipt is not None and receipt["terminal_state"] == "stopped":
            process.terminate()
            raise BootstrapError("guardian reached terminal state while native process was live")
        sequence += 1
        try:
            _publish_supervisor_heartbeat(
                context, sequence=sequence, status="NATIVE_RUNNING_GUARDIAN_SUPERVISED"
            )
        except BaseException:
            pass
        heartbeat_ok = False
        if GUARDIAN_HEARTBEAT_PATH.is_file():
            try:
                heartbeat = json.loads(
                    GUARDIAN_HEARTBEAT_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                heartbeat = None
            heartbeat_ok = bool(
                isinstance(heartbeat, dict)
                and heartbeat.get("capsule_sha256")
                == context["contract"]["capsule_sha256"]
                and heartbeat.get("attempt_run_id")
                == context["contract"]["attempt_run_id"]
                and heartbeat.get("instance_id") == context["contract"]["instance_id"]
                and heartbeat.get("destroy_authority") is False
                and isinstance(heartbeat.get("wall_time_unix"), (int, float))
                and time.time() - float(heartbeat["wall_time_unix"]) <= 30
            )
        if (
            current_guardian.poll() is not None
            or (time.monotonic() - guardian_started > 30 and not heartbeat_ok)
        ):
            if current_guardian.poll() is None:
                current_guardian.terminate()
            current_guardian = _start_guardian()
            guardian_started = time.monotonic()
        time.sleep(min(float(SUPERVISOR_POLL_SECONDS), deadline - time.monotonic()))
    if process.poll() is None:
        process.terminate()
        raise BootstrapError("native process exceeded supervised deadline")
    if process.returncode != 0:
        raise BootstrapError("native process exited nonzero")
    return _supervise_guardian(current_guardian, context, deadline=deadline)


def deploy(*, capsule: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Validate imported contracts, materialize, and invoke exact native argv."""

    capsule_module = importlib.import_module("infra.vast.parallax.pg1_execution_capsule")
    deployment = importlib.import_module("infra.vast.parallax.pg1_native_deployment")
    provider = importlib.import_module("infra.vast.parallax.pg1_native_provider")
    capsule_module.validate_capsule(capsule)
    deployment.validate_layout_receipt(capsule)
    provider.validate_stop_context(context, capsule=capsule)
    provider.provider_identity_proof(capsule=capsule, context=context)
    materialization = materialize_staged_runtime(capsule=capsule)
    return materialization


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else __import__("sys").argv[1:])
    entry_parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    entry_parser.add_argument("--entry-contract", type=Path)
    entry_args, _unknown = entry_parser.parse_known_args(raw_argv)
    if entry_args.entry_contract is None:
        raise BootstrapError("bootstrap entry contract path is required")
    discovered = _mechanical_entry_from_source()
    actual_argv = [str(Path(__file__).resolve()), *raw_argv]
    if discovered is not None:
        dispatch, expected_entry = discovered
        if entry_args.entry_contract.resolve() != expected_entry:
            raise BootstrapError("mechanical bootstrap entry path is not fixed")
        return _mechanical_dispatch(dispatch, actual_argv)
    if entry_args.entry_contract != ENTRY_CONTRACT_RUNTIME_PATH:
        raise BootstrapError("production bootstrap entry path is not fixed")
    dispatch = _entry_contract(entry_args.entry_contract)
    if dispatch["kind"] != "PRODUCTION_AUTHORITY_BOUND":
        raise BootstrapError("mechanical contract cannot select canonical bootstrap bytes")
    try:
        context = _load_minimal_context(STOP_CONTEXT_RUNTIME_PATH)
    except BaseException as error:
        _untrusted_context_failure_pointer(error)
        raise
    deadline = time.monotonic() + SUPERVISOR_TIMEOUT_SECONDS
    guardian: subprocess.Popen[str] | None = None
    try:
        if _read_lifecycle(context) is None:
            _transition_lifecycle(
                context, "ARMED", reason="BOOTSTRAP_ARMED_BEFORE_CLI"
            )
        _transition_lifecycle(
            context,
            "RUNNING_HEARTBEAT",
            reason="BOOTSTRAP_POINTER_BOUNDARY_ENTERED",
        )
        guardian = _start_guardian()
        parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
        parser.add_argument("--deploy", action="store_true")
        parser.add_argument("--capsule", type=Path)
        parser.add_argument("--stop-context", type=Path)
        parser.add_argument("--entry-contract", type=Path)
        args = parser.parse_args(raw_argv)
        if (
            not args.deploy
            or args.capsule != CAPSULE_RUNTIME_PATH
            or args.stop_context != STOP_CONTEXT_RUNTIME_PATH
            or args.entry_contract != ENTRY_CONTRACT_RUNTIME_PATH
        ):
            raise BootstrapError("production bootstrap argv changed")
        capsule = _load_minimal_capsule(args.capsule)
        if capsule["content_address"]["digest"] != context["contract"]["capsule_sha256"]:
            raise BootstrapError("minimal capsule and stop context identity changed")
        expected_bootstrap = [
            str(BOOTSTRAP_SOURCE_ROOT / "infra/vast/parallax/pg1_native_bootstrap.py"),
            "--deploy", "--capsule",
            str(CAPSULE_RUNTIME_PATH), "--stop-context", str(STOP_CONTEXT_RUNTIME_PATH),
            "--entry-contract", str(ENTRY_CONTRACT_RUNTIME_PATH),
        ]
        if capsule["contract"]["entrypoints"].get("bootstrap_argv") != expected_bootstrap:
            raise BootstrapError("capsule bootstrap entrypoint changed")
        if (
            dispatch.get("kind") != "PRODUCTION_AUTHORITY_BOUND"
            or dispatch.get("dispatch_scope") != "PRODUCTION_ENTRY_ONLY"
            or dispatch.get("capsule_sha256") != capsule["content_address"]["digest"]
            or dispatch.get("attempt_run_id")
            != capsule["contract"]["attempt_identity"]["attempt_run_id"]
            or dispatch.get("bootstrap_argv_sha256")
            != hashlib.sha256(_canonical_bytes(expected_bootstrap)).hexdigest()
            or actual_argv != expected_bootstrap
        ):
            raise BootstrapError("production bootstrap dispatch contract changed")
        deploy(capsule=capsule, context=context)
        receipt = _run_native_supervised(
            capsule, context, guardian, deadline=deadline
        )
        if receipt["terminal_state"] != "stopped":
            raise BootstrapError("guardian failed closed without verified terminal state")
        return 0
    except BaseException as error:
        _pointer_first(context, type(error).__name__)
        marker_error: BaseException | None = None
        for attempt in range(3):
            try:
                _publish_marker(context, type(error).__name__)
                marker_error = None
                break
            except BaseException as candidate:
                marker_error = candidate
                if attempt < 2:
                    time.sleep(float(attempt + 1))
        try:
            receipt = _supervise_guardian(guardian, context, deadline=deadline)
            if receipt["terminal_state"] != "stopped":
                raise BootstrapError("guardian terminal state remained unverified")
        except BaseException as guardian_error:
            if marker_error is not None:
                raise BootstrapError(
                    "bootstrap marker and guardian supervision both failed"
                ) from guardian_error
            raise
        raise


if __name__ == "__main__":
    raise SystemExit(main())
