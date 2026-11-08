"""Fixed PG1 production materializer and controller dispatcher."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .pg1_execution_capsule import (
    CAPSULE_RUNTIME_PATH,
    GUARDIAN_TIMEOUT_SECONDS,
    PYTHON_EXECUTABLE,
    STAGED_ROOT,
    STOP_CONTEXT_RUNTIME_PATH,
    CapsuleError,
    load_capsule,
    production_argv,
    validate_capsule,
)
from .pg1_native_provider import (
    FIXED_ENV,
    aws_prefix,
    load_stop_context,
    provider_identity_proof,
    vast_version_command,
)


def aws_version_command() -> list[str]:
    return [*aws_prefix(), "--version"]


def aws_profile_region_command() -> list[str]:
    return [*aws_prefix(), "configure", "get", "region"]


def runtime_availability_proof(capsule: dict[str, Any]) -> dict[str, Any]:
    validate_capsule(capsule)
    context = load_stop_context(STOP_CONTEXT_RUNTIME_PATH, capsule=capsule)
    return provider_identity_proof(capsule=capsule, context=context)


def materialize(capsule: dict[str, Any]) -> dict[str, Any]:
    from .pg1_native_overlay_stage import capsule_preflight, expected_receipt

    receipt_path = Path(capsule["contract"]["materialization"]["remote_receipt_path"])
    if receipt_path.exists():
        raise CapsuleError("refusing to overwrite native materialization receipt")
    receipt = expected_receipt(capsule, STAGED_ROOT, allow_parameterized_fixture=False)
    capsule_preflight(capsule, STAGED_ROOT, receipt)
    atomic_json(receipt_path, receipt)
    return receipt


def dispatch_command(capsule: dict[str, Any]) -> list[str]:
    command = capsule["contract"]["entrypoints"]["normal_controller_argv"]
    expected = [
        str(PYTHON_EXECUTABLE), "-m", "infra.vast.parallax.pg1_native_controller",
        "--execute", "--capsule", str(CAPSULE_RUNTIME_PATH),
    ]
    if command != expected:
        raise CapsuleError("normal controller argv is not the fixed native dispatcher")
    return list(command)


def execute(*, capsule: dict[str, Any]) -> dict[str, Any]:
    from .pg1_native_backend import ProductionStopper

    stopper = ProductionStopper(capsule=capsule)
    try:
        validate_capsule(capsule)
        runtime = runtime_availability_proof(capsule)
        receipt = materialize(capsule)
        try:
            completed = subprocess.run(
                dispatch_command(capsule),
                check=False,
                capture_output=True,
                text=True,
                timeout=GUARDIAN_TIMEOUT_SECONDS,
                env={**FIXED_ENV, "PYTHONPATH": str(STAGED_ROOT)},
            )
        except subprocess.TimeoutExpired as error:
            raise CapsuleError("native controller timed out") from error
        if completed.returncode != 0:
            raise CapsuleError("native controller exited nonzero")
        try:
            controller = json.loads(completed.stdout or "")
        except json.JSONDecodeError as error:
            raise CapsuleError("native controller returned invalid JSON") from error
        if not isinstance(controller, dict):
            raise CapsuleError("native controller returned non-object output")
    except Exception as error:
        stopper.stop_preserve_scratch(failure=type(error).__name__)
        raise
    return {
        "status": "GREEN_PG1_NATIVE_PRODUCTION_DISPATCH_COMPLETE",
        "capsule_sha256": capsule["content_address"]["digest"],
        "production_argv": production_argv(capsule["contract"]),
        "normal_controller_argv": dispatch_command(capsule),
        "runtime": runtime,
        "materialization": receipt,
        "controller": controller,
    }


def main(argv: list[str] | None = None) -> int:
    from .pg1_native_backend import ProductionStopper

    context = load_stop_context(STOP_CONTEXT_RUNTIME_PATH)
    stopper = ProductionStopper(capsule=None, context=context)
    try:
        parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
        parser.add_argument("--execute", action="store_true")
        parser.add_argument("--capsule", type=Path)
        args = parser.parse_args(argv)
        if not args.execute or args.capsule != CAPSULE_RUNTIME_PATH:
            raise CapsuleError("production launcher argv changed")
        capsule = load_capsule(args.capsule)
        if capsule["contract"]["entrypoints"]["production_argv"] != production_argv(
            capsule["contract"]
        ):
            raise CapsuleError("production launcher argv drift")
        result = execute(capsule=capsule)
    except BaseException as error:
        stopper.stop_preserve_scratch(failure=type(error).__name__)
        raise
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
