"""PG1-native controller with separate one-shot authority and explicit owners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .pg1_execution_capsule import (
    AUTHORIZATION_RECEIPT_RUNTIME_PATH,
    AUTHORIZATION_RUNTIME_PATH,
    CONSUMER_ORDER,
    PRIMARY_CONFIG_PATH,
    SOURCE_CONTRACT_PATH,
    CapsuleError,
    consumer_receipt,
    load_capsule,
    validate_consumer,
    validate_staged_root,
)


def capsule_preflight(capsule: dict[str, Any]) -> dict[str, Any]:
    validate_consumer(capsule, "controller", Path(__file__).resolve())
    if capsule["contract"]["authorization_boundary"] != {
        "embedded_execution_authority": False,
        "preflight_can_grant_authority": False,
        "separate_authorization_required": True,
        "one_shot_consumption_receipt_required": True,
        "h100_authority": False,
    }:
        raise CapsuleError("native controller preflight acquired authority")
    return consumer_receipt(
        capsule,
        "controller",
        {
            "native_modules_distinct": True,
            "separate_one_shot_authorization_required": True,
            "preflight_can_grant_authority": False,
            "h100_authority": False,
        },
    )


def capsule_preflight_graph(
    capsule: dict[str, Any],
    staged_root: Path,
    materialization_receipt: dict[str, Any],
    *,
    allow_parameterized_fixture: bool = False,
) -> dict[str, Any]:
    from . import pg1_native_cell as cell
    from . import pg1_native_critic as critic
    from . import pg1_native_job as job
    from . import pg1_native_overlay_stage as overlay
    from . import pg1_native_source_gate as source
    from . import pg1_native_worker as worker

    root = validate_staged_root(
        staged_root, allow_parameterized_fixture=allow_parameterized_fixture
    )
    receipts = [capsule_preflight(capsule)]
    receipts.append(
        source.capsule_preflight(
            capsule,
            root,
            allow_parameterized_fixture=allow_parameterized_fixture,
        )
    )
    receipts.append(
        overlay.capsule_preflight(
            capsule,
            root,
            materialization_receipt,
            allow_parameterized_fixture=allow_parameterized_fixture,
        )
    )
    receipts.append(
        job.capsule_preflight(
            capsule,
            root,
            allow_parameterized_fixture=allow_parameterized_fixture,
        )
    )
    receipts.append(
        cell.capsule_preflight(
            capsule,
            root=root,
            source_contract=root / SOURCE_CONTRACT_PATH,
            config_path=root / PRIMARY_CONFIG_PATH,
            allow_parameterized_fixture=allow_parameterized_fixture,
        )
    )
    receipts.append(
        worker.capsule_preflight(
            capsule,
            root,
            allow_parameterized_fixture=allow_parameterized_fixture,
        )
    )
    receipts.append(
        critic.capsule_preflight(
            capsule,
            expected_run_id=capsule["contract"]["attempt_identity"]["attempt_run_id"],
            terminal_path=Path("/workspace/parallax/terminal.json"),
            output_path=Path("/workspace/parallax/independent-critic.json"),
        )
    )
    trace = [row["consumer"] for row in receipts]
    if trace != list(CONSUMER_ORDER) or {
        row["capsule_sha256"] for row in receipts
    } != {capsule["content_address"]["digest"]}:
        raise CapsuleError("native preflight call trace or capsule digest changed")
    return {
        "status": "GREEN_NATIVE_V3_ALL_CONSUMERS_PREFLIGHT",
        "capsule_sha256": capsule["content_address"]["digest"],
        "source_run_id": capsule["contract"]["source_identity"]["source_run_id"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "call_trace": trace,
        "receipts": receipts,
        "remote_archive_truth": (
            "NOT_CLAIMED_PARAMETERIZED_FIXTURE"
            if allow_parameterized_fixture
            else "BOUND_TO_SEPARATE_REMOTE_RECEIPT"
        ),
        "archive_body_reads": 0,
        "model_body_reads": 0,
        "data_body_reads": 0,
        "confirmation_items_read": 0,
        "torch_imports": 0,
        "transformers_imports": 0,
        "cuda_operations": 0,
        "hydration_operations": 0,
        "science_operations": 0,
        "claims_written": 0,
        "automatic_retry": False,
        "h100_authority": False,
    }


def execute(
    *, capsule: dict[str, Any]
) -> dict[str, Any]:
    from .pg1_native_authority import ProductionAuthorizationStore
    from .pg1_native_backend import ProductionStopper
    from .pg1_native_job import execute as execute_job
    from .pg1_native_worker import finalize_attempt

    stopper = ProductionStopper(capsule=capsule)
    try:
        validate_consumer(capsule, "controller", Path(__file__).resolve())
        store = ProductionAuthorizationStore()
        authorization = store.load_issued(capsule)
        stopper.bind_instance(authorization["contract"]["instance_id"])
        authorization_receipt = store.consume(capsule, authorization)
        materialization_path = Path(
            capsule["contract"]["materialization"]["remote_receipt_path"]
        )
        materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
        capsule_preflight_graph(
            capsule,
            Path(capsule["contract"]["canonical_paths"]["staged_root"]),
            materialization,
        )
        atomic_json(AUTHORIZATION_RUNTIME_PATH, authorization)
        atomic_json(AUTHORIZATION_RECEIPT_RUNTIME_PATH, authorization_receipt)
        execute_job(
            capsule=capsule,
            authorization=authorization,
            authorization_receipt=authorization_receipt,
        )
        final = finalize_attempt(
            capsule=capsule,
            authorization=authorization,
            authorization_receipt=authorization_receipt,
        )
    except Exception as error:
        stopper.stop_preserve_scratch(failure=type(error).__name__)
        raise
    return {
        "status": final["status"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "trace": [
            "AUTHORIZATION_LOADED_FROM_PRODUCTION_STORE",
            "ONE_SHOT_EXECUTION_AUTHORIZATION_CONSUMED",
            "ALL_NATIVE_CONSUMER_PREFLIGHTS_GREEN",
            "NATIVE_JOB_COMPLETE",
            *final["trace"],
        ],
        "authorization_receipt": authorization_receipt,
        "finalization": final,
    }


def main(argv: list[str] | None = None) -> int:
    from .pg1_native_backend import ProductionStopper
    from .pg1_native_provider import load_stop_context

    context = load_stop_context()
    stopper = ProductionStopper(capsule=None, context=context)
    try:
        parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--capsule-preflight", action="store_true")
        mode.add_argument("--execute", action="store_true")
        parser.add_argument("--capsule", type=Path, required=True)
        parser.add_argument("--staged-root", type=Path)
        parser.add_argument("--materialization-receipt", type=Path)
        args = parser.parse_args(argv)
        capsule = load_capsule(args.capsule)
        if args.execute:
            result = execute(capsule=capsule)
        else:
            if args.staged_root is None or args.materialization_receipt is None:
                raise CapsuleError("controller preflight requires staged root and receipt")
            materialization = json.loads(
                args.materialization_receipt.read_text(encoding="utf-8")
            )
            result = capsule_preflight_graph(capsule, args.staged_root, materialization)
    except BaseException as error:
        stopper.stop_preserve_scratch(failure=type(error).__name__)
        raise
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
