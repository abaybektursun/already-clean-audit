"""PG1-native binding for the byte-restored independent registered critic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .audit_primary_interface_independent import build_audit
from .pg1_execution_capsule import (
    REGISTERED_CRITIC_PATH,
    CapsuleError,
    consumer_receipt,
    critic_execution_argv,
    load_capsule,
    validate_authorization_receipt,
    validate_consumer,
    validate_execution_authorization,
    validate_runtime_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capsule_preflight(
    capsule: dict[str, Any],
    *,
    expected_run_id: str,
    terminal_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    validate_consumer(capsule, "critic", Path(__file__).resolve())
    validate_runtime_source(
        capsule,
        REGISTERED_CRITIC_PATH,
        Path(__file__).with_name("audit_primary_interface_independent.py").resolve(),
    )
    attempt_id = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    if expected_run_id != attempt_id or expected_run_id == capsule["contract"][
        "source_identity"
    ]["source_run_id"]:
        raise CapsuleError("native critic rejected wrong or conflated expected-run-id")
    if terminal_path != Path("/workspace/parallax/terminal.json") or output_path != Path(
        "/workspace/parallax/independent-critic.json"
    ):
        raise CapsuleError("native critic terminal/output path drift")
    expected_argv = critic_execution_argv(capsule["contract"])
    if capsule["contract"]["consumers"]["critic"]["execution_argv"] != expected_argv:
        raise CapsuleError("native critic execution argv changed")
    return consumer_receipt(
        capsule,
        "critic",
        {
            "expected_run_id": expected_run_id,
            "terminal_path": str(terminal_path),
            "output_path": str(output_path),
            "execution_argv": expected_argv,
            "registered_critic": str(REGISTERED_CRITIC_PATH),
            "terminal_body_reads": 0,
        },
    )


def run(
    *,
    capsule: dict[str, Any],
    authorization: dict[str, Any],
    authorization_receipt: dict[str, Any],
) -> dict[str, Any]:
    from .pg1_native_authority import ProductionAuthorizationStore

    validate_execution_authorization(authorization, capsule)
    validate_authorization_receipt(authorization_receipt, authorization, capsule)
    ProductionAuthorizationStore().consume_owner(
        capsule, authorization, authorization_receipt, owner="critic"
    )
    expected_run_id = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    terminal_path = Path(capsule["contract"]["canonical_paths"]["raw_terminal"])
    output_path = Path(capsule["contract"]["canonical_paths"]["critic_output"])
    capsule_preflight(
        capsule,
        expected_run_id=expected_run_id,
        terminal_path=terminal_path,
        output_path=output_path,
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    registered_path = Path(__file__).with_name(
        "audit_primary_interface_independent.py"
    ).resolve()
    audit = build_audit(
        terminal,
        expected_run_id=expected_run_id,
        terminal_path=str(terminal_path.resolve()),
        terminal_sha256=_sha256(terminal_path),
        critic_sha256=_sha256(registered_path),
    )
    _atomic_json(output_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capsule-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--staged-root", type=Path)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--expected-run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capsule = load_capsule(args.capsule)
    if args.capsule_preflight:
        if (
            args.staged_root != Path("/workspace/parallax/source")
            or args.terminal is None
            or args.expected_run_id is None
            or args.output is None
        ):
            raise CapsuleError("native critic staged-root path drift")
        result = capsule_preflight(
            capsule,
            expected_run_id=args.expected_run_id,
            terminal_path=args.terminal,
            output_path=args.output,
        )
    else:
        if args.authorization is None or args.authorization_receipt is None:
            raise CapsuleError("native critic execution requires one-shot authorization")
        result = run(
            capsule=capsule,
            authorization=json.loads(args.authorization.read_text(encoding="utf-8")),
            authorization_receipt=json.loads(
                args.authorization_receipt.read_text(encoding="utf-8")
            ),
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
