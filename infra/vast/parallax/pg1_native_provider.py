"""Fixed production provider identity and fail-closed subprocess semantics.

This module never consults caller PATH, HOME, AWS profile variables, or a
caller-selected executable.  Secrets are read only from their fixed files and
are never included in returned evidence or exception text.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .pg1_execution_capsule import (
    AWS_CLI_VERSION,
    AUTHORIZATION_BUCKET,
    AWS_CONFIG_PATH,
    AWS_CREDENTIALS_PATH,
    AWS_EXECUTABLE,
    AWS_PROFILE,
    AWS_REGION,
    EXPECTED_BUCKET_OWNER,
    PROVIDER_TIMEOUT_SECONDS,
    PYTHON_EXECUTABLE,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_RECEIPT_PATH,
    STOP_CONTEXT_RUNTIME_PATH,
    VAST_API_KEY_PATH,
    VAST_CLI_VERSION,
    VAST_EXECUTABLE,
    CapsuleError,
    canonical_bytes,
    content_sha256,
)
from .pg1_native_vast_adapter import (
    TERMINAL_STOP_STATES,
    VastAdapterError,
    normalize_current_row,
    select_instance,
    validate_current_row,
)


CONTEXT_SCHEMA = "pg1-native-production-stop-context/v1"
CONTEXT_STATUS = "ISSUED_FOR_ONE_FUTURE_PRODUCTION_INSTANCE"
CONTEXT_ISSUER = "PG1_PRODUCTION_DEPLOYMENT_CONTEXT_ISSUER_V1"
FIXED_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "AWS_PROFILE": AWS_PROFILE,
    "AWS_REGION": AWS_REGION,
    "AWS_DEFAULT_REGION": AWS_REGION,
    "AWS_CONFIG_FILE": str(AWS_CONFIG_PATH),
    "AWS_SHARED_CREDENTIALS_FILE": str(AWS_CREDENTIALS_PATH),
    "AWS_EC2_METADATA_DISABLED": "true",
}


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CapsuleError(f"invalid stop-context {label} digest")
    return text


def stop_context_key(capsule_sha256: str, attempt_run_id: str) -> str:
    return (
        "parallax/authorizations/pg1-native-v3/deployment-context/"
        f"{capsule_sha256}/{attempt_run_id}.json"
    )


def validate_stop_context(
    context: dict[str, Any], *, capsule: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(context, dict) or set(context) != {
        "schema_version", "status", "content_address", "contract", "issuance"
    }:
        raise CapsuleError("production stop context envelope changed")
    contract = context.get("contract")
    if (
        context.get("schema_version") != 1
        or context.get("status") != CONTEXT_STATUS
        or not isinstance(contract, dict)
        or context.get("content_address")
        != {"algorithm": "sha256", "digest": content_sha256(contract)}
        or set(contract) != {
            "schema", "issuer", "issuer_account", "capsule_sha256",
            "attempt_run_id", "instance_id", "offer_id", "machine_id",
            "container_id", "runtime_image", "runtime_receipts", "context_key",
        }
        or contract.get("schema") != CONTEXT_SCHEMA
        or contract.get("issuer") != CONTEXT_ISSUER
        or contract.get("issuer_account") != EXPECTED_BUCKET_OWNER
        or not isinstance(contract.get("instance_id"), int)
        or contract["instance_id"] <= 0
        or not isinstance(contract.get("offer_id"), int)
        or contract["offer_id"] <= 0
        or not isinstance(contract.get("machine_id"), int)
        or contract["machine_id"] <= 0
        or not isinstance(contract.get("container_id"), str)
        or not contract["container_id"]
        or contract.get("runtime_image") != RUNTIME_IMAGE
        or contract.get("context_key")
        != stop_context_key(
            str(contract.get("capsule_sha256")), str(contract.get("attempt_run_id"))
        )
    ):
        raise CapsuleError("production stop context identity changed")
    issuance = context.get("issuance")
    if (
        not isinstance(issuance, dict)
        or set(issuance) != {
            "schema", "key", "version_id", "etag", "checksum_sha256",
            "server_side_encryption",
        }
        or issuance.get("schema") != "pg1-native-stop-context-issuance/v1"
        or issuance.get("key") != contract["context_key"]
        or issuance.get("server_side_encryption") != "AES256"
        or not all(
            isinstance(issuance.get(field), str) and issuance[field]
            for field in ("version_id", "etag", "checksum_sha256")
        )
    ):
        raise CapsuleError("production stop context issuance evidence changed")
    expected_receipts = {
        "python": (str(PYTHON_EXECUTABLE), "0755"),
        "aws": (str(AWS_EXECUTABLE), "0755"),
        "vast": (str(VAST_EXECUTABLE), "0755"),
        "aws_config": (str(AWS_CONFIG_PATH), "0400"),
        "aws_credentials": (str(AWS_CREDENTIALS_PATH), "0400"),
        "vast_api_key": (str(VAST_API_KEY_PATH), "0400"),
        "runtime_image": (str(RUNTIME_IMAGE_RECEIPT_PATH), "0400"),
    }
    receipts = contract.get("runtime_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(expected_receipts):
        raise CapsuleError("production runtime receipt set changed")
    for label, (expected_path, expected_mode) in expected_receipts.items():
        row = receipts[label]
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "mode"}
            or row.get("path") != expected_path
            or row.get("mode") != expected_mode
        ):
            raise CapsuleError(f"production {label} provenance changed")
        _digest(row.get("sha256"), label)
    if capsule is not None:
        attempt = capsule["contract"]["attempt_identity"]
        if (
            contract.get("capsule_sha256") != capsule["content_address"]["digest"]
            or contract.get("attempt_run_id") != attempt["attempt_run_id"]
        ):
            raise CapsuleError("stop context is not bound to this capsule attempt")
    return context


def load_stop_context(
    path: Path = STOP_CONTEXT_RUNTIME_PATH, *, capsule: dict[str, Any] | None = None
) -> dict[str, Any]:
    if (
        path != STOP_CONTEXT_RUNTIME_PATH
        or path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o400
    ):
        raise CapsuleError("fixed production stop context is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_stop_context(value, capsule=capsule)


def _secure_file(path: Path, receipt: dict[str, Any], *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CapsuleError(f"fixed production {label} file is absent")
    mode = stat.S_IMODE(path.stat().st_mode)
    expected_mode = int(str(receipt.get("mode", "")), 8)
    if mode != expected_mode:
        raise CapsuleError(f"fixed production {label} permissions changed")
    if sha256_file(path) != receipt.get("sha256"):
        raise CapsuleError(f"fixed production {label} provenance changed")


def aws_prefix() -> list[str]:
    return [
        str(AWS_EXECUTABLE), "--profile", "fuelos", "--region", "us-east-1"
    ]


def sts_command() -> list[str]:
    return [*aws_prefix(), "sts", "get-caller-identity"]


def stop_context_command(context: dict[str, Any], target: Path) -> list[str]:
    contract = context["contract"]
    return [
        *aws_prefix(), "s3api", "get-object",
        "--bucket", AUTHORIZATION_BUCKET,
        "--key", contract["context_key"],
        "--version-id", context["issuance"]["version_id"],
        "--checksum-mode", "ENABLED",
        "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        str(target),
    ]


def vast_version_command(api_key: str) -> list[str]:
    return [str(VAST_EXECUTABLE), "--api-key", api_key, "--version"]


def vast_show_command(instance_id: int, api_key: str) -> list[str]:
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise CapsuleError("fixed Vast instance id is invalid")
    return [
        str(VAST_EXECUTABLE), "show", "instances", "--api-key", api_key, "--raw",
    ]


def vast_stop_command(instance_id: int, api_key: str) -> list[str]:
    return [
        str(VAST_EXECUTABLE), "stop", "instance", str(instance_id),
        "--api-key", api_key, "--raw",
    ]


def vast_destroy_command(instance_id: int, api_key: str) -> list[str]:
    return [
        str(VAST_EXECUTABLE), "destroy", "instance", str(instance_id),
        "--api-key", api_key, "--raw",
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROVIDER_TIMEOUT_SECONDS,
            env=FIXED_ENV,
        )
    except subprocess.TimeoutExpired as error:
        raise CapsuleError("fixed provider command timed out") from error
    if completed.returncode != 0:
        raise CapsuleError("fixed provider command failed")
    return completed


def run_text(command: list[str]) -> str:
    completed = _run(command)
    return (completed.stdout + completed.stderr).strip()


def run_json(command: list[str]) -> Any:
    text = run_text(command)
    try:
        value = json.loads(text or "")
    except json.JSONDecodeError as error:
        raise CapsuleError("fixed provider returned invalid JSON") from error
    return value


def run_vast_mutation(command: list[str]) -> dict[str, Any]:
    """Accept a mutation only by its bounded zero process exit status.

    Vast mutation stdout is deliberately ignored.  The lifecycle proof is a
    later show observation, and destruction is only consecutive absence.
    """

    _run(command)
    return {"schema": "pg1-vast-mutation-exit/v1", "exit_code": 0}


def validate_vast_show(
    response: Any, context: dict[str, Any], *, allowed_states: set[str]
) -> dict[str, Any]:
    instance_id = context["contract"]["instance_id"]
    try:
        row = select_instance(response, instance_id)
        if row is None:
            raise VastAdapterError("bound Vast instance is absent")
        return validate_current_row(
            row, instance_id=instance_id, allowed_states=allowed_states
        )
    except VastAdapterError as error:
        raise CapsuleError("Vast show identity or state changed") from error


def observe_vast_instance(
    context: dict[str, Any], api_key: str
) -> dict[str, Any] | None:
    """Return the exact tracked row proof, or canonical listing absence."""

    instance_id = context["contract"]["instance_id"]
    try:
        row = select_instance(
            run_json(vast_show_command(instance_id, api_key)), instance_id
        )
        if row is None:
            return None
        state = normalize_current_row(row, instance_id=instance_id)
    except VastAdapterError as error:
        raise CapsuleError("Vast show response shape changed") from error
    return state


def wait_vast_terminal(
    context: dict[str, Any], api_key: str, *, terminal_state: str
) -> dict[str, Any]:
    if terminal_state not in {"stopped", "destroyed"}:
        raise CapsuleError("unsupported Vast terminal state")
    last_error: Exception | None = None
    consecutive_absences = 0
    consecutive_terminal = 0
    for attempt in range(1, 10):
        try:
            observed = observe_vast_instance(context, api_key)
            if terminal_state == "destroyed":
                consecutive_absences = consecutive_absences + 1 if observed is None else 0
                if consecutive_absences >= 3:
                    return {
                        "id": context["contract"]["instance_id"],
                        "state": "destroyed",
                        "consecutive_absences": consecutive_absences,
                    }
            elif observed is not None and observed["state"] in TERMINAL_STOP_STATES:
                consecutive_terminal += 1
                if consecutive_terminal >= 3:
                    return {
                        **observed,
                        "terminal_state": "stopped",
                        "consecutive_terminal_observations": consecutive_terminal,
                    }
            else:
                consecutive_terminal = 0
        except Exception as error:
            last_error = error
            consecutive_absences = 0
            consecutive_terminal = 0
        if attempt < 9:
            time.sleep(2)
    raise CapsuleError("Vast instance did not reach the bound terminal state") from last_error


def load_vast_key(context: dict[str, Any]) -> str:
    receipt = context["contract"]["runtime_receipts"]["vast_api_key"]
    _secure_file(VAST_API_KEY_PATH, receipt, label="Vast API key")
    value = VAST_API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not value:
        raise CapsuleError("fixed scoped Vast API key is empty")
    return value


def provider_identity_proof(
    *, capsule: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    validate_stop_context(context, capsule=capsule)
    receipts = context["contract"]["runtime_receipts"]
    for label, path in (
        ("python", PYTHON_EXECUTABLE), ("aws", AWS_EXECUTABLE),
        ("vast", VAST_EXECUTABLE), ("aws_config", AWS_CONFIG_PATH),
        ("aws_credentials", AWS_CREDENTIALS_PATH),
        ("runtime_image", RUNTIME_IMAGE_RECEIPT_PATH),
    ):
        _secure_file(path, receipts[label], label=label)
    hostname = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    if hostname != context["contract"]["container_id"]:
        raise CapsuleError("current container identity changed")
    if RUNTIME_IMAGE_RECEIPT_PATH.read_text(encoding="utf-8").strip() != RUNTIME_IMAGE:
        raise CapsuleError("current runtime image digest receipt changed")
    api_key = load_vast_key(context)
    with tempfile.TemporaryDirectory(prefix="pg1-stop-context-") as directory:
        context_target = Path(directory) / "context.json"
        context_response = run_json(stop_context_command(context, context_target))
        context_data = context_target.read_bytes()
    issued_context = {key: value for key, value in context.items() if key != "issuance"}
    expected_context_data = canonical_bytes(issued_context)
    if (
        context_data != expected_context_data
        or context_response.get("VersionId")
        != context["issuance"]["version_id"]
        or context_response.get("ETag") != context["issuance"]["etag"]
        or context_response.get("ChecksumSHA256")
        != base64.b64encode(hashlib.sha256(expected_context_data).digest()).decode("ascii")
        or context_response.get("ChecksumSHA256")
        != context["issuance"]["checksum_sha256"]
        or context_response.get("ServerSideEncryption") != "AES256"
        or context_response.get("ContentLength") != len(expected_context_data)
    ):
        raise CapsuleError("canonical deployment context S3 identity changed")
    aws_version = run_text([*aws_prefix(), "--version"])
    vast_version = run_text(vast_version_command(api_key))
    caller = run_json(sts_command())
    if (
        AWS_CLI_VERSION not in aws_version
        or VAST_CLI_VERSION not in vast_version
        or caller.get("Account") != EXPECTED_BUCKET_OWNER
        or not str(caller.get("Arn", "")).startswith(
            f"arn:aws:sts::{EXPECTED_BUCKET_OWNER}:assumed-role/"
            "parallax-pg1-native-executor/"
        )
    ):
        raise CapsuleError("production provider caller identity changed")
    instance = validate_vast_show(
        run_json(vast_show_command(context["contract"]["instance_id"], api_key)),
        context,
        allowed_states={"running"},
    )
    return {
        "schema": "pg1-native-production-provider-proof/v1",
        "status": "GREEN_FIXED_PROVIDER_AND_CURRENT_INSTANCE_IDENTITY",
        "capsule_sha256": capsule["content_address"]["digest"],
        "stop_context_sha256": context["content_address"]["digest"],
        "account": caller["Account"],
        "arn": caller["Arn"],
        "instance_id": instance["id"],
        "offer_id": context["contract"]["offer_id"],
        "machine_id": context["contract"]["machine_id"],
        "container_id": context["contract"]["container_id"],
        "provider_state": instance["state"],
        "aws_version": AWS_CLI_VERSION,
        "vast_version": VAST_CLI_VERSION,
        "runtime_image": RUNTIME_IMAGE,
        "secret_logged": False,
    }
