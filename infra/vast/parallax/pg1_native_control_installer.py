#!/usr/local/bin/python3
"""Install and verify the fixed PG1 control root before any launch command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

try:
    from . import pg1_native_mechanical_rebase as mechanical_rebase
except ImportError:
    from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase


STAGE_SOURCE_ROOT = Path("/opt/parallax/control-stage/source")
CONTROL_SOURCE_ROOT = Path("/opt/parallax/control/source")
STAGE_BIN_ROOT = Path("/opt/parallax/control-stage/bin")
CONTROL_BIN_ROOT = Path("/opt/parallax/control/bin")
CAPSULE = Path("/run/parallax/control-plane/pg1-execution-capsule.json")
INPUTS = Path("/run/parallax/control-plane/pg1-future-launch-inputs.json")
AWS_CONFIG = Path("/run/parallax/control-plane/aws/config")
AWS_CREDENTIALS = Path("/run/parallax/control-plane/aws/credentials")
VAST_KEY = Path("/run/secrets/pg1-native-control-plane-vast-api-key")
SSH_KEY = Path("/run/secrets/pg1-native-control-plane-ssh-private-key")
RECEIPT = Path("/run/parallax/control-plane/pg1-control-layout-receipt.json")
EXECUTABLE_SOURCES = frozenset({
    "infra/vast/parallax/pg1_native_control_installer.py",
    "infra/vast/parallax/pg1_native_launch_invoker.py",
    "infra/vast/parallax/pg1_native_launch_packet.py",
    "infra/vast/parallax/pg1_native_launch_authority.py",
    "infra/vast/parallax/pg1_native_launch_gateway.py",
    "infra/vast/parallax/pg1_native_control_deadman.py",
    "infra/vast/parallax/pg1_native_onstart_materializer.py",
})


class ControlInstallerError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _fixed(path: Path, mode: int) -> bytes:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != mode:
        raise ControlInstallerError(f"fixed control material changed: {path}")
    return path.read_bytes()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_fixed(path, 0o400))
    except json.JSONDecodeError as error:
        raise ControlInstallerError(f"fixed control JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ControlInstallerError("fixed control JSON must be an object")
    return value


def _atomic_copy(source: Path, target: Path, *, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_text)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer); writer.flush(); os.fsync(writer.fileno())
        os.chmod(temporary, mode); os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _mechanical_roots_allowed(contract: dict[str, Any] | None) -> bool:
    body = contract.get("contract") if isinstance(contract, dict) else None
    return bool(
        isinstance(body, dict)
        and body.get("kind") == "MECHANICAL_NO_AUTHORITY"
        and body.get("production_authority") is False
        and body.get("provider_calls") == 0
        and body.get("fake_services") == "TRACKED_DETERMINISTIC_EXECUTABLES"
        and body.get("may_mint_production_authority") is False
    )


def install_layout(
    *, capsule: dict[str, Any], inputs: dict[str, Any], source_root: Path,
    control_root: Path, stage_bin_root: Path, control_bin_root: Path,
    material_paths: dict[str, Path], receipt_path: Path,
    mechanical_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    production_roots = (
        mechanical_contract is None
        and
        source_root == STAGE_SOURCE_ROOT and control_root == CONTROL_SOURCE_ROOT
        and stage_bin_root == STAGE_BIN_ROOT and control_bin_root == CONTROL_BIN_ROOT
        and material_paths == {
            "aws_config": AWS_CONFIG, "aws_credentials": AWS_CREDENTIALS,
            "control_vast_key": VAST_KEY, "ssh_private_key": SSH_KEY,
        }
        and receipt_path == RECEIPT
    )
    mechanical = _mechanical_roots_allowed(mechanical_contract)
    if mechanical:
        root = mechanical_contract.get("root")
        expected_material = {
            "aws_config": root / "run/parallax/control-plane/aws/config",
            "aws_credentials": root / "run/parallax/control-plane/aws/credentials",
            "control_vast_key": root / "run/secrets/pg1-native-control-plane-vast-api-key",
            "ssh_private_key": root / "run/secrets/pg1-native-control-plane-ssh-private-key",
        } if isinstance(root, Path) else None
        mechanical = bool(
            isinstance(root, Path)
            and source_root == root / "opt/parallax/control-stage/source"
            and control_root == root / "opt/parallax/control/source"
            and stage_bin_root == root / "opt/parallax/control-stage/bin"
            and control_bin_root == root / "opt/parallax/control/bin"
            and material_paths == expected_material
            and receipt_path
            == root / "run/parallax/control-plane/pg1-control-layout-receipt.json"
        )
    if production_roots == mechanical:
        raise ControlInstallerError("control root is neither production nor bounded mechanical")
    contract = capsule.get("contract") if isinstance(capsule, dict) else None
    wiring = contract.get("runtime_wiring") if isinstance(contract, dict) else None
    if (
        not isinstance(wiring, dict) or not wiring
        or capsule.get("content_address", {}).get("digest")
        != hashlib.sha256(canonical_bytes(contract)).hexdigest()
        or inputs.get("schema") != "pg1-native-future-launch-inputs/v1"
    ):
        raise ControlInstallerError("control install capsule or inputs changed")
    installed: dict[str, dict[str, str]] = {}
    for relative_text, expected in wiring.items():
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ControlInstallerError("control source path escaped fixed root")
        source = source_root / relative
        if source.is_symlink() or not source.is_file() or _sha256(source) != expected:
            raise ControlInstallerError(f"control source changed: {relative_text}")
        mode = 0o755 if relative_text in EXECUTABLE_SOURCES else 0o644
        target = control_root / relative
        _atomic_copy(source, target, mode=mode)
        if _sha256(target) != expected or stat.S_IMODE(target.stat().st_mode) != mode:
            raise ControlInstallerError(f"installed control source changed: {relative_text}")
        installed[relative_text] = {"sha256": expected, "mode": f"{mode:04o}"}
    executable_inputs = inputs.get("control_executables")
    if not isinstance(executable_inputs, dict):
        raise ControlInstallerError("control executable inventory absent")
    for name in ("aws", "vastai"):
        source = stage_bin_root / name
        expected = executable_inputs.get(f"{name}_sha256")
        if (
            source.is_symlink() or not source.is_file()
            or stat.S_IMODE(source.stat().st_mode) != 0o755 or _sha256(source) != expected
        ):
            raise ControlInstallerError(f"staged control executable changed: {name}")
        target = control_bin_root / name
        _atomic_copy(source, target, mode=0o755)
        if _sha256(target) != expected:
            raise ControlInstallerError(f"installed control executable changed: {name}")
    mount_inputs = inputs.get("control_mounts")
    for label, path in material_paths.items():
        expected = mount_inputs.get(f"{label}_sha256")
        if hashlib.sha256(_fixed(path, 0o400)).hexdigest() != expected:
            raise ControlInstallerError(f"fixed control credential provenance changed: {label}")
    receipt = {
        "schema": "pg1-native-control-layout-receipt/v1",
        "status": "FIXED_CONTROL_LAYOUT_VERIFIED",
        "capsule_sha256": capsule["content_address"]["digest"],
        "runtime_sources": installed,
        "executables_verified": ["aws", "vastai"],
        "credentials_verified_without_receipt_content": True,
        "production_layout": production_roots, "mechanical_layout": mechanical,
        "provider_calls": 0,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", dir=receipt_path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(receipt)); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o400); os.replace(temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def production_invocation_argv(
    *, one_shot_nonce: str, offer_id: int, attempt_run_id: str,
) -> list[list[str]]:
    if (
        len(one_shot_nonce) != 64
        or isinstance(offer_id, bool) or not isinstance(offer_id, int) or offer_id <= 0
        or not isinstance(attempt_run_id, str) or not attempt_run_id
    ):
        raise ControlInstallerError("control invocation identity changed")
    root = CONTROL_SOURCE_ROOT / "infra/vast/parallax"
    packet = CAPSULE.parent / "pg1-future-launch-packet.json"
    authority_receipt = CAPSULE.parent / "pg1-launch-authority-receipt.json"
    return [
        [str(root / "pg1_native_launch_packet.py"), "--compile", "--capsule", str(CAPSULE),
         "--inputs", str(INPUTS), "--offer-id", str(offer_id),
         "--attempt-run-id", attempt_run_id, "--output",
         str(packet)],
        [str(root / "pg1_native_launch_authority.py"), "--issue", "--capsule", str(CAPSULE),
         "--packet", str(packet),
         "--one-shot-nonce", one_shot_nonce, "--receipt",
         str(authority_receipt)],
        [str(root / "pg1_native_launch_gateway.py"), "--launch", "--capsule", str(CAPSULE),
         "--packet", str(packet)],
    ]


def configure_from_source_location() -> dict[str, Any] | None:
    """Rebase only when these exact bytes live under a contract-bound copied root."""

    global STAGE_SOURCE_ROOT, CONTROL_SOURCE_ROOT, STAGE_BIN_ROOT, CONTROL_BIN_ROOT
    global CAPSULE, INPUTS, AWS_CONFIG, AWS_CREDENTIALS, VAST_KEY, SSH_KEY, RECEIPT
    discovered = mechanical_rebase.control_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    contract, _contract_path = discovered; root = contract["root"]
    def at(path: str) -> Path:
        return mechanical_rebase.rebased(root, Path(path))
    STAGE_SOURCE_ROOT = at("/opt/parallax/control-stage/source")
    CONTROL_SOURCE_ROOT = at("/opt/parallax/control/source")
    STAGE_BIN_ROOT = at("/opt/parallax/control-stage/bin")
    CONTROL_BIN_ROOT = at("/opt/parallax/control/bin")
    CAPSULE = at("/run/parallax/control-plane/pg1-execution-capsule.json")
    INPUTS = at("/run/parallax/control-plane/pg1-future-launch-inputs.json")
    AWS_CONFIG = at("/run/parallax/control-plane/aws/config")
    AWS_CREDENTIALS = at("/run/parallax/control-plane/aws/credentials")
    VAST_KEY = at("/run/secrets/pg1-native-control-plane-vast-api-key")
    SSH_KEY = at("/run/secrets/pg1-native-control-plane-ssh-private-key")
    RECEIPT = at("/run/parallax/control-plane/pg1-control-layout-receipt.json")
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    mechanical = configure_from_source_location()
    if (
        not args.install or args.capsule != CAPSULE or args.inputs != INPUTS
        or args.receipt != RECEIPT
    ):
        raise ControlInstallerError("production control installer argv changed")
    receipt = install_layout(
        capsule=_json(args.capsule), inputs=_json(args.inputs),
        source_root=STAGE_SOURCE_ROOT, control_root=CONTROL_SOURCE_ROOT,
        stage_bin_root=STAGE_BIN_ROOT, control_bin_root=CONTROL_BIN_ROOT,
        material_paths={
            "aws_config": AWS_CONFIG, "aws_credentials": AWS_CREDENTIALS,
            "control_vast_key": VAST_KEY, "ssh_private_key": SSH_KEY,
        }, receipt_path=args.receipt, mechanical_contract=mechanical,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
