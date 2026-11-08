"""PG1 exact durability, critic, and all-failure lifecycle owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path
from typing import Any

try:
    from .artifacts import atomic_json
    from . import pg1_native_mechanical_rebase as mechanical_rebase
    from .pg1_execution_capsule import (
        CapsuleError,
        canonical_bytes,
        consumer_receipt,
        content_sha256,
        critic_execution_argv,
        load_capsule,
        validate_authorization_receipt,
        validate_consumer,
        validate_execution_authorization,
        validate_staged_root,
    )
except ImportError:
    from infra.vast.parallax.artifacts import atomic_json
    from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase
    from infra.vast.parallax.pg1_execution_capsule import (
        CapsuleError,
        canonical_bytes,
        consumer_receipt,
        content_sha256,
        critic_execution_argv,
        load_capsule,
        validate_authorization_receipt,
        validate_consumer,
        validate_execution_authorization,
        validate_staged_root,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mechanical_entry_from_source() -> dict[str, Any] | None:
    discovered = mechanical_rebase.worker_contract_from_source(Path(__file__))
    if discovered is None:
        return None
    _root_contract, _root_contract_path, worker = discovered
    contract_path = worker / "run/parallax/pg1-native-entry-contract.json"
    if (
        contract_path.is_symlink() or not contract_path.is_file()
        or stat.S_IMODE(contract_path.stat().st_mode) != 0o400
    ):
        raise CapsuleError("mechanical worker contract is absent or mutable")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CapsuleError("mechanical worker contract unreadable") from error
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "pg1-native-entry-dispatch-contract/v2"
        or contract.get("kind") != "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"
        or contract.get("dispatch_scope") != "EXACT_REBASED_BYTES_ONLY"
        or contract.get("production_authority") is not False
        or contract.get("may_mint_authorization") is not False
        or Path(str(contract.get("rebase_root", ""))).resolve() != worker
        or Path(str(contract.get("image_root", ""))).resolve()
        != worker / "opt/parallax/image/source"
        or Path(str(contract.get("bootstrap_root", ""))).resolve()
        != worker / "opt/parallax/future-freeze/source"
        or Path(__file__).resolve()
        != worker / "opt/parallax/future-freeze/source/infra/vast/parallax/pg1_native_worker.py"
    ):
        raise CapsuleError("mechanical worker source contract changed")
    return contract


def _mechanical_execution(contract: dict[str, Any], invoked_argv: list[str]) -> int:
    source = Path(__file__).resolve()
    relative = "infra/vast/parallax/pg1_native_worker.py"
    receipt_path = Path(str(contract.get("worker_receipt_path", ""))).resolve()
    rebase_root = Path(str(contract.get("rebase_root", ""))).resolve()
    bootstrap_root = Path(str(contract.get("bootstrap_root", ""))).resolve()
    if (
        contract.get("schema") != "pg1-native-entry-dispatch-contract/v2"
        or contract.get("kind") != "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2"
        or contract.get("dispatch_scope") != "EXACT_REBASED_BYTES_ONLY"
        or contract.get("production_authority") is not False
        or contract.get("may_mint_authorization") is not False
        or contract.get("worker_argv") != invoked_argv
        or contract.get("runtime_wiring", {}).get(relative) != _sha256(source)
        or not rebase_root.is_absolute() or rebase_root == Path("/")
        or rebase_root not in source.parents
        or source != bootstrap_root / relative
        or rebase_root not in receipt_path.parents
    ):
        raise CapsuleError("mechanical worker execution escaped its exact scope")
    receipt = {
        "schema": "pg1-native-exact-worker-execution/v1",
        "status": "EXACT_COPIED_WORKER_BYTES_EXECUTED",
        "capsule_sha256": contract["capsule_sha256"],
        "attempt_run_id": contract["attempt_run_id"],
        "worker_path": str(source), "worker_sha256": _sha256(source),
        "argv": invoked_argv, "production_authority": False,
        "may_mint_authorization": False, "provider_calls": 0,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


def capsule_preflight(
    capsule: dict[str, Any],
    staged_root: Path,
    *,
    allow_parameterized_fixture: bool = False,
) -> dict[str, Any]:
    validate_consumer(capsule, "worker", Path(__file__).resolve())
    root = validate_staged_root(
        staged_root, allow_parameterized_fixture=allow_parameterized_fixture
    )
    attempt = capsule["contract"]["attempt_identity"]
    attempt_id = attempt["attempt_run_id"]
    if (
        attempt["terminal_key"] != f"parallax/results/{attempt_id}/terminal.json"
        or attempt["scratch_key"]
        != f"parallax/results/{attempt_id}/scratch-pointer.json"
        or attempt["critic_key"]
        != f"parallax/results/{attempt_id}/independent-critic.json"
        or capsule["contract"]["consumers"]["critic"]["execution_argv"]
        != critic_execution_argv(capsule["contract"])
    ):
        raise CapsuleError("native worker attempt or critic binding changed")
    return consumer_receipt(
        capsule,
        "worker",
        {
            "staged_root": str(root),
            "attempt_run_id": attempt_id,
            "terminal_key": attempt["terminal_key"],
            "scratch_key": attempt["scratch_key"],
            "critic_key": attempt["critic_key"],
            "critic_argv": critic_execution_argv(capsule["contract"]),
            "failure_action": "IDEMPOTENT_PRIMARY_STOP_THEN_S3_GUARDIAN",
            "destroy_after": "EXACT_DOWNLOADED_TERMINAL_SCRATCH_AND_CRITIC_VERIFIED",
            "automatic_retry": False,
        },
    )


def validate_durability_receipt(
    receipt: dict[str, Any],
    *,
    capsule: dict[str, Any],
    kind: str,
    source_path: Path,
    object_key: str,
) -> dict[str, Any]:
    expected_metadata = {
        "attempt-run-id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "capsule-sha256": capsule["content_address"]["digest"],
        "instance-id": str(receipt.get("metadata", {}).get("instance-id", "")),
        "object-kind": kind,
        "source-sha256": _sha256(source_path),
    }
    expected = {
        "schema": "pg1-native-durability-receipt/v3",
        "status": "DURABLE_CONDITIONAL_CREATE_RETURNED",
        "kind": kind,
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "object_key": object_key,
        "source_sha256": _sha256(source_path),
        "checksum_sha256": receipt.get("checksum_sha256"),
        "content_length": source_path.stat().st_size,
        "content_type": (
            "application/x-tar" if source_path.suffix == ".tar" else "application/json"
        ),
        "server_side_encryption": "AES256",
        "metadata": expected_metadata,
        "version_id": receipt.get("version_id"),
        "etag": receipt.get("etag"),
    }
    if receipt != expected or not all(
        isinstance(receipt.get(field), str) and receipt[field]
        for field in ("checksum_sha256", "version_id", "etag")
    ):
        raise CapsuleError(f"{kind} exact durability receipt is absent or wrong")
    return receipt


def validate_durability_verification(
    verification: dict[str, Any], receipt: dict[str, Any], *, kind: str
) -> dict[str, Any]:
    expected = {
        "schema": "pg1-native-durability-verification/v3",
        "status": "VERIFIED_EXACT_DOWNLOADED_DURABLE_OBJECT",
        "kind": kind,
        "receipt_sha256": content_sha256(receipt),
        "downloaded_sha256": receipt["source_sha256"],
        "downloaded_bytes": receipt["content_length"],
    }
    if verification != expected:
        raise CapsuleError(f"{kind} downloaded durability verification failed")
    return verification


def finalize_attempt(
    *,
    capsule: dict[str, Any],
    authorization: dict[str, Any],
    authorization_receipt: dict[str, Any],
) -> dict[str, Any]:
    from .pg1_native_authority import ProductionAuthorizationStore
    from .pg1_native_artifacts import (
        final_durability_receipt,
        package_required_artifacts,
        verified_teardown_evidence,
    )
    from .pg1_native_backend import ProductionBackend, ProductionStopper
    from .pg1_native_critic import run as run_critic

    stopper = ProductionStopper(capsule=capsule)
    trace: list[str] = []
    try:
        validate_consumer(capsule, "worker", Path(__file__).resolve())
        validate_execution_authorization(authorization, capsule)
        stopper.bind_instance(authorization["contract"]["instance_id"])
        validate_authorization_receipt(authorization_receipt, authorization, capsule)
        ProductionAuthorizationStore().consume_owner(
            capsule, authorization, authorization_receipt, owner="worker"
        )
        attempt = capsule["contract"]["attempt_identity"]
        attempt_id = attempt["attempt_run_id"]
        paths = capsule["contract"]["canonical_paths"]
        raw_terminal_path = Path(paths["raw_terminal"])
        critic_output_path = Path(paths["critic_output"])
        scratch_pointer_path = Path(paths["scratch_pointer"])
        backend = ProductionBackend(
            capsule=capsule, instance_id=authorization["contract"]["instance_id"]
        )
        if (
            not raw_terminal_path.is_file()
            or critic_output_path.exists()
            or scratch_pointer_path.exists()
        ):
            raise CapsuleError("native worker terminal/critic scratch state changed")
        terminal = json.loads(raw_terminal_path.read_text(encoding="utf-8"))
        if terminal.get("run_id") != attempt_id:
            raise CapsuleError("native worker raw terminal attempt identity changed")
        atomic_json(
            scratch_pointer_path,
            {
                "schema": "pg1-native-scratch-pointer/v2",
                "status": "SCRATCH_PRESERVED_UNTIL_VERIFIED_DESTROY",
                "capsule_sha256": capsule["content_address"]["digest"],
                "attempt_run_id": attempt_id,
                "instance_id": authorization["contract"]["instance_id"],
                "raw_terminal": str(raw_terminal_path),
                "output_dir": paths["output_dir"],
            },
        )
        trace.extend(("RAW_TERMINAL_READY", "SCRATCH_POINTER_READY"))
        receipts: dict[str, dict[str, Any]] = {}
        verifications: dict[str, dict[str, Any]] = {}
        for kind, source_path, object_key in (
            ("raw_terminal", raw_terminal_path, attempt["terminal_key"]),
            ("scratch_pointer", scratch_pointer_path, attempt["scratch_key"]),
        ):
            receipt = backend.durable_sink(
                kind=kind, source_path=source_path, object_key=object_key
            )
            validate_durability_receipt(
                receipt,
                capsule=capsule,
                kind=kind,
                source_path=source_path,
                object_key=object_key,
            )
            verification = backend.verify_durability(kind=kind, receipt=receipt)
            validate_durability_verification(verification, receipt, kind=kind)
            receipts[kind] = receipt
            verifications[kind] = verification
            trace.extend((f"{kind.upper()}_DURABLE", f"{kind.upper()}_DURABILITY_VERIFIED"))
        critic_result = run_critic(
            capsule=capsule,
            authorization=authorization,
            authorization_receipt=authorization_receipt,
        )
        if (
            not critic_output_path.is_file()
            or critic_result.get("status") != "GREEN_INDEPENDENT_EXACT"
            or critic_result.get("run_id") != attempt_id
            or critic_result.get("bound_inputs", {}).get("terminal")
            != {"path": str(raw_terminal_path.resolve()), "sha256": _sha256(raw_terminal_path)}
        ):
            raise CapsuleError("native independent critic did not complete exactly")
        trace.append("INDEPENDENT_CRITIC_COMPLETE")
        critic_receipt = backend.durable_sink(
            kind="critic_audit",
            source_path=critic_output_path,
            object_key=attempt["critic_key"],
        )
        validate_durability_receipt(
            critic_receipt,
            capsule=capsule,
            kind="critic_audit",
            source_path=critic_output_path,
            object_key=attempt["critic_key"],
        )
        critic_verification = backend.verify_durability(
            kind="critic_audit", receipt=critic_receipt
        )
        validate_durability_verification(
            critic_verification, critic_receipt, kind="critic_audit"
        )
        receipts["critic_audit"] = critic_receipt
        verifications["critic_audit"] = critic_verification
        trace.extend(
            (
                "CRITIC_AUDIT_DURABLE",
                "CRITIC_AUDIT_DURABILITY_VERIFIED",
            )
        )
        bundle_evidence = package_required_artifacts(
            capsule=capsule, terminal=terminal
        )
        if bundle_evidence.get("status") != (
            "COMPLETE_REQUIRED_BODIES_PACKAGED_AND_READ_BACK"
        ):
            raise CapsuleError("required artifact package is incomplete")
        trace.append("ARTIFACT_INVENTORY_COMPLETE")
        for kind, source_path, object_key in (
            ("artifact_bundle", Path(paths["artifact_bundle"]), attempt["artifact_key"]),
            (
                "artifact_manifest",
                Path(paths["artifact_manifest"]),
                attempt["artifact_manifest_key"],
            ),
        ):
            receipt = backend.durable_sink(
                kind=kind, source_path=source_path, object_key=object_key
            )
            validate_durability_receipt(
                receipt,
                capsule=capsule,
                kind=kind,
                source_path=source_path,
                object_key=object_key,
            )
            verification = backend.verify_durability(kind=kind, receipt=receipt)
            validate_durability_verification(verification, receipt, kind=kind)
            receipts[kind] = receipt
            verifications[kind] = verification
            trace.extend(
                (f"{kind.upper()}_DURABLE", f"{kind.upper()}_DURABILITY_VERIFIED")
            )
        final_receipt = final_durability_receipt(
            capsule=capsule, receipts=receipts, verifications=verifications
        )
        final_receipt_path = Path(paths["durability_receipts"])
        atomic_json(final_receipt_path, final_receipt)
        receipt_durability = backend.durable_sink(
            kind="artifact_receipts",
            source_path=final_receipt_path,
            object_key=attempt["artifact_receipts_key"],
        )
        validate_durability_receipt(
            receipt_durability,
            capsule=capsule,
            kind="artifact_receipts",
            source_path=final_receipt_path,
            object_key=attempt["artifact_receipts_key"],
        )
        receipt_verification = backend.verify_durability(
            kind="artifact_receipts", receipt=receipt_durability
        )
        validate_durability_verification(
            receipt_verification, receipt_durability, kind="artifact_receipts"
        )
        teardown_evidence = verified_teardown_evidence(
            capsule=capsule,
            final_receipt=final_receipt,
            receipt_durability=receipt_durability,
            receipt_verification=receipt_verification,
        )
        trace.extend(
            ("ARTIFACT_RECEIPTS_DURABLE", "ARTIFACT_RECEIPTS_DURABILITY_VERIFIED")
        )
        trace.append("DURABILITY_VERIFIED")
        destroy = backend.request_destroy(evidence=teardown_evidence)
        if (
            destroy.get("status") != "DURABLE_EXTERNAL_DESTROY_REQUEST"
            or destroy.get("worker_provider_mutations") != 0
            or destroy.get("worker_absence_polls") != 0
        ):
            raise CapsuleError("native destroy receipt changed")
        trace.append("DESTROY_REQUESTED")
    except Exception as error:
        stopper.stop_preserve_scratch(failure=type(error).__name__)
        raise
    return {
        "status": "GREEN_NATIVE_TERMINAL_CRITIC_DURABLE_DESTROY_REQUESTED",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": attempt_id,
        "trace": trace,
        "receipts": receipts,
        "verifications": verifications,
        "destroy": destroy,
        "final_durability_receipt": final_receipt,
        "receipt_durability": receipt_durability,
        "receipt_verification": receipt_verification,
        "teardown_evidence": teardown_evidence,
        "scratch_preserved_until_destroy_request": True,
        "automatic_retry": False,
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capsule-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--mechanical-no-model-proof", action="store_true")
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--staged-root", type=Path)
    args = parser.parse_args(raw_argv)
    mechanical = _mechanical_entry_from_source()
    if mechanical is not None:
        expected_capsule = Path(str(mechanical["rebase_root"])) / (
            "run/parallax/pg1-execution-capsule.json"
        )
        if (
            not args.mechanical_no_model_proof
            or args.capsule_preflight or args.execute
            or args.capsule != expected_capsule
            or args.staged_root != Path(str(mechanical.get("bootstrap_root", "")))
            or args.authorization is not None or args.authorization_receipt is not None
        ):
            raise CapsuleError("mechanical worker normal production argv changed")
        return _mechanical_execution(
            mechanical, [str(Path(__file__).resolve()), *raw_argv]
        )
    if args.mechanical_no_model_proof:
        raise CapsuleError("mechanical no-model proof is unavailable in production")
    from .pg1_native_backend import ProductionStopper
    from .pg1_native_provider import load_stop_context

    context = load_stop_context()
    stopper = ProductionStopper(capsule=None, context=context)
    try:
        capsule = load_capsule(args.capsule)
        if args.capsule_preflight:
            if args.staged_root is None:
                raise CapsuleError("native worker preflight requires staged root")
            result = capsule_preflight(capsule, args.staged_root)
        else:
            if args.authorization is None or args.authorization_receipt is None:
                raise CapsuleError("native worker execution requires one-shot authority")
            result = finalize_attempt(
                capsule=capsule,
                authorization=json.loads(args.authorization.read_text(encoding="utf-8")),
                authorization_receipt=json.loads(
                    args.authorization_receipt.read_text(encoding="utf-8")
                ),
            )
    except BaseException as error:
        stopper.stop_preserve_scratch(failure=type(error).__name__)
        raise
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
