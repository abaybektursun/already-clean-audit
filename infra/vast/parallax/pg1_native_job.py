"""PG1-native job owner with immutable explicit cell inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .pg1_execution_capsule import (
    PRIMARY_CONFIG_PATH,
    LINEAGE,
    TOKENIZER_TERMINAL_SHA256,
    CapsuleError,
    content_sha256,
    consumer_receipt,
    load_capsule,
    validate_authorization_receipt,
    validate_consumer,
    validate_execution_authorization,
    validate_staged_root,
)
from .pg1_native_source_gate import capsule_preflight as source_preflight


def capsule_preflight(
    capsule: dict[str, Any],
    staged_root: Path,
    *,
    allow_parameterized_fixture: bool = False,
) -> dict[str, Any]:
    validate_consumer(capsule, "job", Path(__file__).resolve())
    root = validate_staged_root(
        staged_root, allow_parameterized_fixture=allow_parameterized_fixture
    )
    source_preflight(
        capsule,
        root,
        allow_parameterized_fixture=allow_parameterized_fixture,
    )
    config = capsule["contract"]["artifacts"]["primary_config"]
    target = root / PRIMARY_CONFIG_PATH
    if (
        config["path"] != str(PRIMARY_CONFIG_PATH)
        or not target.is_file()
        or sha256_file(target) != config["sha256"]
    ):
        raise CapsuleError("native job primary_config binding changed")
    return consumer_receipt(
        capsule,
        "job",
        {
            "config_label": "primary_config",
            "config_path": str(PRIMARY_CONFIG_PATH),
            "preflight_before_processor_hydration": True,
            "preflight_before_weight_hydration": True,
        },
    )


def _materialize_tokenizer_evidence(capsule: dict[str, Any]) -> Path:
    paths = capsule["contract"]["canonical_paths"]
    terminal_path = Path(paths["tokenizer_terminal"])
    output_path = Path(paths["tokenizer_evidence"])
    if (
        not terminal_path.is_file()
        or sha256_file(terminal_path) != TOKENIZER_TERMINAL_SHA256
        or output_path.exists()
    ):
        raise CapsuleError("capsule-bound tokenizer terminal is absent, changed, or reused")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    preflight = payload.get("preflight")
    if (
        terminal.get("status") != "GREEN_JOB_TERMINAL"
        or payload.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or payload.get("lineage") != LINEAGE
        or payload.get("model_loads") != 0
        or payload.get("gpu_operations") != 0
        or payload.get("h100_instances_created") != 0
        or payload.get("confirmation_items_read") != 0
        or not isinstance(preflight, dict)
    ):
        raise CapsuleError("capsule-bound tokenizer evidence is not processor-only green")
    atomic_json(output_path, preflight)
    return output_path


def execute(
    *,
    capsule: dict[str, Any],
    authorization: dict[str, Any],
    authorization_receipt: dict[str, Any],
) -> dict[str, Any]:
    from .pg1_native_authority import ProductionAuthorizationStore
    from .pg1_native_artifacts import build_payload_manifest
    from .pg1_native_cell import run
    from .pg1_native_hydration import verify_runtime_hydration

    validate_consumer(capsule, "job", Path(__file__).resolve())
    validate_execution_authorization(authorization, capsule)
    validate_authorization_receipt(authorization_receipt, authorization, capsule)
    ProductionAuthorizationStore().consume_owner(
        capsule, authorization, authorization_receipt, owner="job"
    )
    paths = capsule["contract"]["canonical_paths"]
    raw_terminal_path = Path(paths["raw_terminal"])
    output_dir = Path(paths["output_dir"])
    if raw_terminal_path.exists():
        raise CapsuleError("native job refuses to overwrite raw terminal")
    _materialize_tokenizer_evidence(capsule)
    hydration = verify_runtime_hydration(capsule=capsule)
    control = run(
        capsule=capsule,
        authorization=authorization,
        authorization_receipt=authorization_receipt,
    )
    attempt_id = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    if (
        control.get("status") != "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION"
        or control.get("lineage") != LINEAGE
        or control.get("confirmation_items_read") != 0
        or control.get("promotion_eligible") is not False
    ):
        raise CapsuleError("native cell result escaped primary discovery boundary")
    artifact_manifest = build_payload_manifest(capsule=capsule, control=control)
    artifact_manifest_path = Path(paths["artifact_manifest"])
    payload = {
        "schema_version": 1,
        "status": "GREEN_NATIVE_RAW_TERMINAL_READY",
        "run_id": attempt_id,
        "control": control,
        "runtime_hydration": hydration,
        "artifact_manifest": {
            "run_id": attempt_id,
            "control_path": str(output_dir / "interface-control.json"),
            "control_sha256": sha256_file(output_dir / "interface-control.json"),
            "path": str(artifact_manifest_path),
            "sha256": sha256_file(artifact_manifest_path),
            "content_sha256": content_sha256(artifact_manifest),
            "file_count": artifact_manifest["file_count"],
            "total_bytes": artifact_manifest["total_bytes"],
            "confirmation_items_read": 0,
            "promotion_eligible": False,
        },
        "confirmation_items_read": 0,
        "automatic_retry": False,
    }
    terminal = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_JOB_TERMINAL",
        "run_id": attempt_id,
        "stage": "training_interface_control",
        "job": {"result": {"payload": payload}},
        "durability_state": "RAW_NOT_YET_UPLOADED",
        "confirmation_items_read": 0,
        "automatic_retry": False,
    }
    atomic_json(raw_terminal_path, terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--staged-root", type=Path)
    args = parser.parse_args()
    capsule = load_capsule(args.capsule)
    if args.capsule_preflight:
        if args.staged_root is None:
            raise CapsuleError("native job preflight requires staged root")
        result = capsule_preflight(capsule, args.staged_root)
    else:
        if not args.execute or args.authorization is None or args.authorization_receipt is None:
            raise CapsuleError("native job execution arguments are incomplete")
        result = execute(
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
