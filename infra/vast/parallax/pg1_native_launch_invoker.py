#!/usr/local/bin/python3
"""Install the fixed PG1 control root and execute its one native launch chain."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 300


class LaunchInvokerError(RuntimeError):
    pass


def _installer() -> Any:
    """Load copied installer bytes using only this executable's own location."""

    source_root = Path(__file__).resolve().parents[3]
    expected = source_root / "infra/vast/parallax/pg1_native_launch_invoker.py"
    if Path(__file__).resolve() != expected or not (source_root / "infra").is_dir():
        raise LaunchInvokerError("native launch invoker copied source layout changed")
    sys.path.insert(0, str(source_root))
    return importlib.import_module(
        "infra.vast.parallax.pg1_native_control_installer"
    )


def _environment(installer: Any) -> dict[str, str]:
    return {
        "PATH": f"{installer.CONTROL_BIN_ROOT}:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/nonexistent", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(installer.CONTROL_SOURCE_ROOT),
    }


def invoke(
    *, capsule_path: Path, inputs_path: Path, receipt_path: Path,
    offer_id: int, attempt_run_id: str, one_shot_nonce: str,
) -> dict[str, Any]:
    installer = _installer()
    mechanical = installer.configure_from_source_location()
    if (
        capsule_path != installer.CAPSULE or inputs_path != installer.INPUTS
        or receipt_path != installer.RECEIPT
    ):
        raise LaunchInvokerError("native launch invoker fixed inputs changed")
    install_receipt = installer.install_layout(
        capsule=installer._json(capsule_path), inputs=installer._json(inputs_path),
        source_root=installer.STAGE_SOURCE_ROOT,
        control_root=installer.CONTROL_SOURCE_ROOT,
        stage_bin_root=installer.STAGE_BIN_ROOT,
        control_bin_root=installer.CONTROL_BIN_ROOT,
        material_paths={
            "aws_config": installer.AWS_CONFIG,
            "aws_credentials": installer.AWS_CREDENTIALS,
            "control_vast_key": installer.VAST_KEY,
            "ssh_private_key": installer.SSH_KEY,
        },
        receipt_path=receipt_path, mechanical_contract=mechanical,
    )
    commands = installer.production_invocation_argv(
        one_shot_nonce=one_shot_nonce, offer_id=offer_id,
        attempt_run_id=attempt_run_id,
    )
    steps: list[dict[str, Any]] = []
    for name, command in zip(("compile", "issue", "gateway"), commands, strict=True):
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, cwd=str(installer.CONTROL_SOURCE_ROOT),
                env=_environment(installer), close_fds=True,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LaunchInvokerError(f"native {name} boundary failed") from error
        if completed.returncode != 0:
            raise LaunchInvokerError(
                f"native {name} boundary returned nonzero: {completed.stderr.strip()}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise LaunchInvokerError(f"native {name} boundary output changed") from error
        steps.append({"name": name, "result": result})
    return {
        "schema": "pg1-native-causal-launch-invocation/v1",
        "status": "INSTALL_COMPILE_AUTHORITY_GATEWAY_EXECUTED",
        "capsule_sha256": install_receipt["capsule_sha256"],
        "attempt_run_id": attempt_run_id,
        "installer": install_receipt,
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--launch-future-native", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--attempt-run-id")
    parser.add_argument("--one-shot-nonce")
    args = parser.parse_args(argv)
    if (
        not args.launch_future_native or args.capsule is None or args.inputs is None
        or args.receipt is None or args.offer_id is None or not args.attempt_run_id
        or not args.one_shot_nonce
    ):
        raise LaunchInvokerError("native launch invoker argv changed")
    print(json.dumps(invoke(
        capsule_path=args.capsule, inputs_path=args.inputs,
        receipt_path=args.receipt, offer_id=args.offer_id,
        attempt_run_id=args.attempt_run_id,
        one_shot_nonce=args.one_shot_nonce,
    ), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
