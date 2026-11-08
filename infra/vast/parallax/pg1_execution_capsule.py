"""Authority-free contract primitives for the fresh PG1-native execution path.

The capsule is immutable metadata.  It cannot authorize execution.  A normal
native entrypoint additionally requires a separately issued, content-addressed,
one-shot authorization and a digest-bound consumption receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file


CAPSULE_SCHEMA = "pg1-equal-token-native-execution-capsule/v3"
CAPSULE_STATUS = "COMPILED_V3_AUTHORITY_FREE_EXECUTION_CONTRACT"
AUTHORIZATION_SCHEMA = "pg1-native-production-execution-authorization/v2"
AUTHORIZATION_STATUS = "AUTHORIZED_ONE_SHOT_EXECUTION"
AUTHORIZATION_RECEIPT_SCHEMA = "pg1-native-production-authorization-receipt/v2"
PRODUCTION_AUTHORIZATION_ISSUER = "PG1_PRODUCTION_S3_CONDITIONAL_STORE_V1"
AUTHORIZATION_BUCKET = "pg-lineage-769063704696"
AUTHORIZATION_PREFIX = "parallax/authorizations/pg1-native-v3"
EXPECTED_BUCKET_OWNER = "769063704696"
AWS_PROFILE = "fuelos"
AWS_REGION = "us-east-1"
AWS_CLI_VERSION = "aws-cli/2.33.2"
VAST_CLI_VERSION = "0.5.0"
VAST_COMMAND_SCHEMA = "vastai-cli/0.5.0-tracked-rows-exit-then-show-v3"
PYTHON_EXECUTABLE = Path("/usr/local/bin/python3")
AWS_EXECUTABLE = Path("/usr/local/bin/aws")
VAST_EXECUTABLE = Path("/usr/local/bin/vastai")
AWS_CONFIG_PATH = Path("/run/parallax/aws/config")
AWS_CREDENTIALS_PATH = Path("/run/parallax/aws/credentials")
VAST_API_KEY_PATH = Path("/run/secrets/pg1-native-vast-api-key")
RUNTIME_IMAGE_RECEIPT_PATH = Path("/run/parallax/runtime-image-digest")
STOP_CONTEXT_RUNTIME_PATH = Path("/run/parallax/pg1-native-stop-context.json")
ENTRY_CONTRACT_RUNTIME_PATH = Path("/run/parallax/pg1-native-entry-contract.json")
START_LEASE_RUNTIME_PATH = Path("/run/parallax/pg1-native-start-lease.json")
HANDOFF_LAYOUT_RUNTIME_PATH = Path("/run/parallax/pg1-native-handoff-layout.json")
BOOTSTRAP_SOURCE_ROOT = Path("/opt/parallax/future-freeze/source")
IMMUTABLE_IMAGE_SOURCE_ROOT = Path("/opt/parallax/image/source")
CONTROL_PLANE_SOURCE_ROOT = Path("/opt/parallax/control/source")
CONTROL_PLANE_STAGE_SOURCE_ROOT = Path("/opt/parallax/control-stage/source")
DEPLOYMENT_LAYOUT_RECEIPT_PATH = Path(
    "/run/parallax/pg1-native-deployment-layout.json"
)
LOCAL_LIFECYCLE_PATH = Path("/workspace/parallax/failure-stop-pointer.json")
PROVIDER_TIMEOUT_SECONDS = 30
GUARDIAN_POLL_SECONDS = 5
GUARDIAN_TIMEOUT_SECONDS = 10_800
RUNTIME_IMAGE = (
    "769063704696.dkr.ecr.us-east-1.amazonaws.com/parallax/runtime-minimal@sha256:"
    "f7407a831951f4c744a9afecf59d248bfadd31d44d012f200bc5055b6d4d4ed6"
)
SOURCE_RUN_ID = "parallax-pg1-primary-interface-s3-v1"
BASELINE_COMMIT = "8bba643"
STAGED_ROOT = Path("/workspace/parallax/source")
CAPSULE_RUNTIME_PATH = Path("/run/parallax/pg1-execution-capsule.json")
MECHANICAL_CONTRACT_RUNTIME_PATH = Path(
    "/run/parallax/pg1-mechanical-rehearsal-contract.json"
)
AUTHORIZATION_RUNTIME_PATH = Path("/run/parallax/pg1-execution-authorization.json")
AUTHORIZATION_RECEIPT_RUNTIME_PATH = Path(
    "/run/parallax/pg1-execution-authorization-receipt.json"
)
SOURCE_CONTRACT_PATH = Path(
    "studies/lineage/cross-lineage/config/"
    "parallax_training_interface_control_source_contract_pg1_primary_v1.json"
)
SOURCE_CONTRACT_SHA256 = "a95f65a78d5dc6b7f4b80416d40c6b9dcc45cc1c91fde74154da90dd117e5544"
PRIMARY_CONFIG_PATH = Path(
    "studies/lineage/cross-lineage/config/"
    "parallax_training_interface_control_pg1_primary_v1.yaml"
)
PRIMARY_CONFIG_SHA256 = "fa5d8ccefeacfbb57fcd223b3b4486d4adef80388ae06d48452221d738e53078"
MATRIX_PATH = Path("studies/lineage/cross-lineage/config/parallax_matrix.yaml")
MATRIX_SHA256 = "9df29f6e559f1d2d12651dec3623faa1d515139504e69ef5b1bcfaff325db91c"
TASK_ROOT_PATH = Path("studies/lineage/cross-lineage/data/parallax_tasks_v1")
DISCOVERY_MANIFEST_PATH = TASK_ROOT_PATH / "manifests/items.discovery.jsonl"
DISCOVERY_MANIFEST_SHA256 = "12a8c184087dffc949ae9fcbacc92e8fbbf92bf59d36b14c02547202e2033902"
TOKENIZER_TERMINAL_PATH = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg1_primary_processor_preflight_v1/terminal.json"
)
TOKENIZER_TERMINAL_SHA256 = "f751ef90acd6a96cafe3dea4b491ab8df9d06d6f7faa8388ebda798c5f4b2ba9"
MODEL_MANIFEST_PATH = Path(
    "studies/lineage/cross-lineage/config/parallax_pg1_h100_s3_hydration_job_v1.json"
)
MODEL_MANIFEST_SHA256 = "fc186748f7a19ca922aac1a25f09745b366d1b4efb62353dc2254a5bc303fbab"
LINEAGE = "pg1_pi"
LINEAGE_ROLE = "primary_within_lineage_discovery_independent_of_supporting_pg2"
ENTITY_ROLE = "fused"
MODEL_REPO = "google/paligemma-3b-pt-224"
MODEL_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
BUNDLE_RECEIPT_PATH = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)
BUNDLE_RECEIPT_SHA256 = "3924d5b01dde6f5e6f2a4991c8ae03bdbfb2ed343e764be4a25ec7613e4ccedd"
OVERLAY_RECEIPT_PATH = Path(
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg1_primary_interface_control_overlays_s3_v1.json"
)
OVERLAY_RECEIPT_SHA256 = "29725b294fdffd088b77317e331beaa1eb8384cdbdda9ee0746d06e014cd4133"
CORE_PATH = Path("infra/vast/parallax/pg1_execution_capsule.py")
REGISTERED_CRITIC_PATH = Path(
    "infra/vast/parallax/audit_primary_interface_independent.py"
)
AUTHORITY_PATH = Path("infra/vast/parallax/pg1_native_authority.py")
BACKEND_PATH = Path("infra/vast/parallax/pg1_native_backend.py")
REHEARSAL_PATH = Path("infra/vast/parallax/pg1_native_rehearsal.py")
LAUNCHER_PATH = Path("infra/vast/parallax/pg1_native_launcher.py")
HYDRATION_PATH = Path("infra/vast/parallax/pg1_native_hydration.py")
PROVIDER_PATH = Path("infra/vast/parallax/pg1_native_provider.py")
VAST_ADAPTER_PATH = Path("infra/vast/parallax/pg1_native_vast_adapter.py")
ARTIFACTS_PATH = Path("infra/vast/parallax/pg1_native_artifacts.py")
GUARDIAN_PATH = Path("infra/vast/parallax/pg1_native_guardian.py")
BOOTSTRAP_PATH = Path("infra/vast/parallax/pg1_native_bootstrap.py")
DEPLOYMENT_PATH = Path("infra/vast/parallax/pg1_native_deployment.py")
LIFECYCLE_PATH = Path("infra/vast/parallax/pg1_native_lifecycle.py")
CONTROL_DEADMAN_PATH = Path("infra/vast/parallax/pg1_native_control_deadman.py")
LAUNCH_GATEWAY_PATH = Path("infra/vast/parallax/pg1_native_launch_gateway.py")
LAUNCH_PACKET_PATH = Path("infra/vast/parallax/pg1_native_launch_packet.py")
IDLE_BOOTSTRAP_PATH = Path("infra/vast/parallax/pg1_native_idle_bootstrap.py")
LAUNCH_AUTHORITY_PATH = Path("infra/vast/parallax/pg1_native_launch_authority.py")
LAUNCH_STATE_PATH = Path("infra/vast/parallax/pg1_native_launch_state.py")
ONSTART_MATERIALIZER_PATH = Path(
    "infra/vast/parallax/pg1_native_onstart_materializer.py"
)
CONTROL_INSTALLER_PATH = Path("infra/vast/parallax/pg1_native_control_installer.py")
LAUNCH_INVOKER_PATH = Path("infra/vast/parallax/pg1_native_launch_invoker.py")
MECHANICAL_REBASE_PATH = Path("infra/vast/parallax/pg1_native_mechanical_rebase.py")
RUNTIME_DEPENDENCY_PATHS = tuple(
    Path(f"infra/vast/parallax/{name}.py") for name in (
        "artifacts", "behavioral_common", "behavioral_model", "cache_gate", "dag",
        "g1_common", "graft_common", "parallax_entities", "parallax_tasks",
        "launch_packet", "training_common", "training_interface_control",
        "training_interface_control_cell", "training_interface_control_primary_cell",
        "training_interface_interaction", "training_model",
    )
)
PRODUCTION_ENTRY_PATH = IMMUTABLE_IMAGE_SOURCE_ROOT / DEPLOYMENT_PATH
CONTROL_DEADMAN_IMAGE_PATH = CONTROL_PLANE_SOURCE_ROOT / CONTROL_DEADMAN_PATH
LAUNCH_GATEWAY_IMAGE_PATH = CONTROL_PLANE_SOURCE_ROOT / LAUNCH_GATEWAY_PATH
LAUNCH_PACKET_IMAGE_PATH = CONTROL_PLANE_SOURCE_ROOT / LAUNCH_PACKET_PATH
IDLE_BOOTSTRAP_IMAGE_PATH = IMMUTABLE_IMAGE_SOURCE_ROOT / IDLE_BOOTSTRAP_PATH
CONTROL_INSTALLER_IMAGE_PATH = CONTROL_PLANE_SOURCE_ROOT / CONTROL_INSTALLER_PATH
LAUNCH_INVOKER_STAGE_PATH = CONTROL_PLANE_STAGE_SOURCE_ROOT / LAUNCH_INVOKER_PATH
LAUNCH_AUTHORITY_IMAGE_PATH = CONTROL_PLANE_SOURCE_ROOT / LAUNCH_AUTHORITY_PATH
ONSTART_MATERIALIZER_RUNTIME_PATH = Path(
    "/run/parallax/pg1-native-onstart-materializer.py"
)

NATIVE_BINDINGS = (
    (
        "controller",
        "infra.vast.parallax.pg1_native_controller",
        Path("infra/vast/parallax/pg1_native_controller.py"),
    ),
    (
        "source_gate",
        "infra.vast.parallax.pg1_native_source_gate",
        Path("infra/vast/parallax/pg1_native_source_gate.py"),
    ),
    (
        "overlay_stage",
        "infra.vast.parallax.pg1_native_overlay_stage",
        Path("infra/vast/parallax/pg1_native_overlay_stage.py"),
    ),
    (
        "job",
        "infra.vast.parallax.pg1_native_job",
        Path("infra/vast/parallax/pg1_native_job.py"),
    ),
    (
        "cell",
        "infra.vast.parallax.pg1_native_cell",
        Path("infra/vast/parallax/pg1_native_cell.py"),
    ),
    (
        "worker",
        "infra.vast.parallax.pg1_native_worker",
        Path("infra/vast/parallax/pg1_native_worker.py"),
    ),
    (
        "critic",
        "infra.vast.parallax.pg1_native_critic",
        Path("infra/vast/parallax/pg1_native_critic.py"),
    ),
)
CONSUMER_ORDER = tuple(row[0] for row in NATIVE_BINDINGS)
EXECUTION_CALLABLES = (
    ("controller", "execute"),
    ("source_gate", "capsule_preflight"),
    ("overlay_stage", "capsule_preflight"),
    ("job", "execute"),
    ("cell", "run"),
    ("worker", "finalize_attempt"),
    ("critic", "run"),
)
RUNTIME_PATHS = (
    CORE_PATH,
    REGISTERED_CRITIC_PATH,
    AUTHORITY_PATH,
    BACKEND_PATH,
    REHEARSAL_PATH,
    LAUNCHER_PATH,
    HYDRATION_PATH,
    PROVIDER_PATH,
    VAST_ADAPTER_PATH,
    ARTIFACTS_PATH,
    GUARDIAN_PATH,
    BOOTSTRAP_PATH,
    DEPLOYMENT_PATH,
    LIFECYCLE_PATH,
    CONTROL_DEADMAN_PATH,
    LAUNCH_GATEWAY_PATH,
    LAUNCH_PACKET_PATH,
    IDLE_BOOTSTRAP_PATH,
    LAUNCH_AUTHORITY_PATH,
    LAUNCH_STATE_PATH,
    ONSTART_MATERIALIZER_PATH,
    CONTROL_INSTALLER_PATH,
    LAUNCH_INVOKER_PATH,
    MECHANICAL_REBASE_PATH,
    *RUNTIME_DEPENDENCY_PATHS,
    *(row[2] for row in NATIVE_BINDINGS),
)
PHASE_ORDER = (
    "SOURCE_STAGING",
    "EXACT_REMOTE_MATERIALIZATION_GREEN",
    "ALL_NATIVE_CONSUMER_PREFLIGHTS_GREEN",
    "ONE_SHOT_EXECUTION_AUTHORIZATION_CONSUMED",
    "PROCESSOR_HYDRATION",
    "WEIGHT_HYDRATION",
    "RUNTIME_HYDRATION_VERIFIED",
    "SCIENCE_STARTED",
    "RAW_TERMINAL_DURABLE",
    "SCRATCH_POINTER_DURABLE",
    "INDEPENDENT_CRITIC_COMPLETE",
    "CRITIC_AUDIT_DURABLE",
    "ARTIFACT_INVENTORY_COMPLETE",
    "ARTIFACT_BUNDLE_DURABLE",
    "ARTIFACT_MANIFEST_DURABLE",
    "DURABILITY_VERIFIED",
    "DESTROY_REQUESTED",
)
ARTIFACT_LABELS = frozenset(
    {
        "infra_init",
        "vast_init",
        "parallax_init",
        "artifacts_module",
        "g1_common",
        "entities",
        "parallax_tasks",
        "training_common",
        "training_model",
        "interface_mechanics",
        "interface_cell",
        "primary_cell",
        "interaction_metric",
        "primary_config",
        "primary_spec",
        "metric_spec",
        "metric_config",
        "matrix",
        "discovery_manifest",
        "discovery_seal",
        "model_cache_manifest",
        "external_processor_terminal",
        "external_processor_adjudication",
        "post_processor_freeze",
        "next_cell_audit",
        "requirements",
        "source_contract",
    }
)


class CapsuleError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def production_authorization_id(capsule: dict[str, Any]) -> str:
    return (
        "pg1-native-production:"
        f"{capsule['content_address']['digest']}:"
        f"{capsule['contract']['attempt_identity']['attempt_run_id']}"
    )


def production_issued_key(capsule: dict[str, Any]) -> str:
    return (
        f"{AUTHORIZATION_PREFIX}/issued/{capsule['content_address']['digest']}/"
        f"{capsule['contract']['attempt_identity']['attempt_run_id']}.json"
    )


def production_consumed_key(
    capsule: dict[str, Any], authorization: dict[str, Any]
) -> str:
    return (
        f"{AUTHORIZATION_PREFIX}/consumed/{capsule['content_address']['digest']}/"
        f"{capsule['contract']['attempt_identity']['attempt_run_id']}/"
        f"{authorization['content_address']['digest']}.json"
    )


def production_consumption_marker(
    capsule: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "pg1-native-production-consumption-marker/v2",
        "authorization_sha256": authorization["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "issued_key": production_issued_key(capsule),
    }


def addressed(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": CAPSULE_STATUS,
        "content_address": {"algorithm": "sha256", "digest": content_sha256(contract)},
        "contract": copy.deepcopy(contract),
    }


def _digest(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CapsuleError(f"invalid {label} sha256")
    return digest


def safe_relative(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CapsuleError(f"unsafe {label} path")
    return path


def validate_staged_root(
    staged_root: Path, *, allow_parameterized_fixture: bool = False
) -> Path:
    resolved = staged_root.resolve()
    if not allow_parameterized_fixture and resolved != STAGED_ROOT:
        raise CapsuleError("native consumer staged-root path drift")
    return resolved


def _binding(name: str) -> tuple[str, Path]:
    for candidate, module, path in NATIVE_BINDINGS:
        if candidate == name:
            return module, path
    raise CapsuleError(f"unknown native consumer: {name}")


def _execution_callable(name: str) -> str:
    for candidate, function in EXECUTION_CALLABLES:
        if candidate == name:
            return function
    raise CapsuleError(f"unknown native execution callable: {name}")


def consumer_preflight_argv(name: str, contract: dict[str, Any]) -> list[str]:
    module, _ = _binding(name)
    argv = [
        str(PYTHON_EXECUTABLE),
        "-m",
        module,
        "--capsule-preflight",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--staged-root",
        str(STAGED_ROOT),
    ]
    if name in {"controller", "overlay_stage"}:
        argv.extend(
            [
                "--materialization-receipt",
                "/run/parallax/pg1-materialization-receipt.json",
            ]
        )
    if name == "critic":
        argv.extend(
            [
                "--terminal",
                "/workspace/parallax/terminal.json",
                "--expected-run-id",
                str(contract["attempt_identity"]["attempt_run_id"]),
                "--output",
                "/workspace/parallax/independent-critic.json",
            ]
        )
    return argv


def critic_execution_argv(contract: dict[str, Any]) -> list[str]:
    module, _ = _binding("critic")
    return [
        str(PYTHON_EXECUTABLE),
        "-m",
        module,
        "--execute",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--authorization",
        str(AUTHORIZATION_RUNTIME_PATH),
        "--authorization-receipt",
        str(AUTHORIZATION_RECEIPT_RUNTIME_PATH),
    ]


def consumer_execution_argv(name: str, contract: dict[str, Any]) -> list[str]:
    if name == "critic":
        return critic_execution_argv(contract)
    module, _ = _binding(name)
    if name in {"source_gate", "overlay_stage"}:
        return consumer_preflight_argv(name, contract)
    if name == "controller":
        return [
            str(PYTHON_EXECUTABLE),
            "-m",
            module,
            "--execute",
            "--capsule",
            str(CAPSULE_RUNTIME_PATH),
        ]
    argv = [
        str(PYTHON_EXECUTABLE),
        "-m",
        module,
        "--execute",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--authorization",
        str(AUTHORIZATION_RUNTIME_PATH),
        "--authorization-receipt",
        str(AUTHORIZATION_RECEIPT_RUNTIME_PATH),
    ]
    return argv


def production_argv(contract: dict[str, Any]) -> list[str]:
    return [
        str(PRODUCTION_ENTRY_PATH),
        "--launch",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--stop-context",
        str(STOP_CONTEXT_RUNTIME_PATH),
        "--entry-contract",
        str(ENTRY_CONTRACT_RUNTIME_PATH),
    ]


def deadman_argv(contract: dict[str, Any]) -> list[str]:
    return [
        str(CONTROL_DEADMAN_IMAGE_PATH),
        "--control-deadman",
        "--capsule",
        "/run/parallax/control-plane/pg1-execution-capsule.json",
        "--launch-transaction",
        "/run/parallax/control-plane/pg1-worker-launch-transaction.json",
    ]


def launch_gateway_argv(contract: dict[str, Any]) -> list[str]:
    return [
        str(LAUNCH_GATEWAY_IMAGE_PATH),
        "--launch",
        "--capsule",
        "/run/parallax/control-plane/pg1-execution-capsule.json",
        "--packet",
        "/run/parallax/control-plane/pg1-future-launch-packet.json",
    ]


def native_launcher_argv(contract: dict[str, Any]) -> list[str]:
    return [
        str(PYTHON_EXECUTABLE),
        "-m",
        "infra.vast.parallax.pg1_native_launcher",
        "--execute",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
    ]


def mechanical_rehearsal_argv(contract: dict[str, Any]) -> list[str]:
    return [
        "python3",
        "-m",
        "infra.vast.parallax.pg1_native_rehearsal",
        "--mechanical-rehearsal",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--contract",
        str(MECHANICAL_CONTRACT_RUNTIME_PATH),
        "--rehearsal-root",
        "${REHEARSAL_ROOT}",
        "--failure",
        "${FAILURE}",
    ]


def mechanical_bootstrap_argv(contract: dict[str, Any]) -> list[str]:
    return [
        "python3", "-m", "infra.vast.parallax.pg1_native_rehearsal",
        "--mechanical-bootstrap", "--capsule", str(CAPSULE_RUNTIME_PATH),
        "--contract", str(MECHANICAL_CONTRACT_RUNTIME_PATH),
        "--rehearsal-root", "${REHEARSAL_ROOT}",
        "--failure", "${FAILURE}",
    ]


def bootstrap_argv(contract: dict[str, Any]) -> list[str]:
    return [
        str(BOOTSTRAP_SOURCE_ROOT / BOOTSTRAP_PATH),
        "--deploy",
        "--capsule",
        str(CAPSULE_RUNTIME_PATH),
        "--stop-context",
        str(STOP_CONTEXT_RUNTIME_PATH),
        "--entry-contract",
        str(ENTRY_CONTRACT_RUNTIME_PATH),
    ]


def artifact_contract(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "pg1-native-required-artifact-inventory/v1",
        "bundle_path": "/workspace/parallax/artifacts.tar",
        "bundle_key": attempt["artifact_key"],
        "manifest_path": "/workspace/parallax/artifacts.manifest.json",
        "manifest_key": attempt["artifact_manifest_key"],
        "receipts_path": "/workspace/parallax/durability-receipts.json",
        "receipts_key": attempt["artifact_receipts_key"],
        "required_singletons": {
            "authorization": str(AUTHORIZATION_RUNTIME_PATH),
            "authorization_receipt": str(AUTHORIZATION_RECEIPT_RUNTIME_PATH),
            "materialization_receipt": "/run/parallax/pg1-materialization-receipt.json",
            "deployment_layout_receipt": str(DEPLOYMENT_LAYOUT_RECEIPT_PATH),
            "stop_context": str(STOP_CONTEXT_RUNTIME_PATH),
            "tokenizer_evidence": "/workspace/parallax/tokenizer-preflight.json",
            "hydration_evidence": "/workspace/parallax/runtime-hydration.json",
            "interface_control": (
                "/workspace/parallax/output/interface-control/interface-control.json"
            ),
            "raw_terminal": "/workspace/parallax/terminal.json",
            "critic_audit": "/workspace/parallax/independent-critic.json",
            "scratch_pointer": "/workspace/parallax/scratch-pointer.json",
        },
        "checkpoint_source": "RAW_TERMINAL_CONTROL_CODEBOOK_SNAPSHOTS",
        "checkpoint_suffix": ".pt",
        "minimum_checkpoints": 1,
        "unexpected_output_files_forbidden": True,
        "teardown_requires_all_inventory_verified": True,
    }


def validate_contract_schema(contract: dict[str, Any]) -> dict[str, Any]:
    if set(contract) != {
        "capsule_schema",
        "source_identity",
        "attempt_identity",
        "artifacts",
        "materialization",
        "consumers",
        "runtime_wiring",
        "phase_order",
        "firewall",
        "authorization_boundary",
        "science_inputs",
        "canonical_paths",
        "production_runtime",
        "artifact_contract",
        "entrypoints",
    } or contract.get("capsule_schema") != CAPSULE_SCHEMA:
        raise CapsuleError("canonical v3 contract field set changed")
    if contract.get("source_identity") != {
        "source_run_id": SOURCE_RUN_ID,
        "source_contract_path": str(SOURCE_CONTRACT_PATH),
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
    }:
        raise CapsuleError("canonical source identity changed")
    attempt = contract.get("attempt_identity")
    if not isinstance(attempt, dict) or set(attempt) != {
        "attempt_run_id",
        "artifact_key",
        "artifact_manifest_key",
        "artifact_receipts_key",
        "terminal_key",
        "scratch_key",
        "critic_key",
    }:
        raise CapsuleError("attempt identity field set changed")
    attempt_id = attempt.get("attempt_run_id")
    if (
        not isinstance(attempt_id, str)
        or not attempt_id
        or attempt_id == SOURCE_RUN_ID
        or attempt.get("artifact_key") != f"parallax/results/{attempt_id}/artifacts.tar"
        or attempt.get("artifact_manifest_key")
        != f"parallax/results/{attempt_id}/artifacts.manifest.json"
        or attempt.get("artifact_receipts_key")
        != f"parallax/results/{attempt_id}/durability-receipts.json"
        or attempt.get("terminal_key") != f"parallax/results/{attempt_id}/terminal.json"
        or attempt.get("scratch_key")
        != f"parallax/results/{attempt_id}/scratch-pointer.json"
        or attempt.get("critic_key") != f"parallax/results/{attempt_id}/independent-critic.json"
    ):
        raise CapsuleError("attempt identity is wrong or conflated")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACT_LABELS:
        raise CapsuleError("canonical artifact label set changed")
    for label, row in artifacts.items():
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "origin"}:
            raise CapsuleError(f"artifact {label} schema changed")
        safe_relative(row["path"], label=f"artifact {label}")
        _digest(row["sha256"], label=f"artifact {label}")
        if row["origin"] not in {"base_bundle", "exact_overlay"}:
            raise CapsuleError(f"artifact {label} origin changed")
    if artifacts["primary_config"]["path"] != str(PRIMARY_CONFIG_PATH) or artifacts[
        "primary_config"
    ]["sha256"] != PRIMARY_CONFIG_SHA256:
        raise CapsuleError("canonical primary_config changed")
    if artifacts["source_contract"]["path"] != str(SOURCE_CONTRACT_PATH) or artifacts[
        "source_contract"
    ]["sha256"] != SOURCE_CONTRACT_SHA256:
        raise CapsuleError("canonical source contract artifact changed")
    if contract.get("materialization") != {
        "bundle_receipt_path": str(BUNDLE_RECEIPT_PATH),
        "bundle_receipt_sha256": BUNDLE_RECEIPT_SHA256,
        "overlay_receipt_path": str(OVERLAY_RECEIPT_PATH),
        "overlay_receipt_sha256": OVERLAY_RECEIPT_SHA256,
        "remote_receipt_path": "/run/parallax/pg1-materialization-receipt.json",
        "exact_archive_rehearsal": "PENDING_BOUNDED_REMOTE_CPU_REHEARSAL",
    }:
        raise CapsuleError("materialization identity changed")
    expected_science_inputs = {
        "matrix": {"path": str(MATRIX_PATH), "sha256": MATRIX_SHA256},
        "primary_config": {
            "path": str(PRIMARY_CONFIG_PATH),
            "sha256": PRIMARY_CONFIG_SHA256,
        },
        "task_root": str(TASK_ROOT_PATH),
        "discovery_manifest": {
            "path": str(DISCOVERY_MANIFEST_PATH),
            "sha256": DISCOVERY_MANIFEST_SHA256,
        },
        "tokenizer_terminal": {
            "path": str(TOKENIZER_TERMINAL_PATH),
            "sha256": TOKENIZER_TERMINAL_SHA256,
        },
        "lineage": LINEAGE,
        "lineage_role": LINEAGE_ROLE,
        "entity_role": ENTITY_ROLE,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "manifest_path": str(MODEL_MANIFEST_PATH),
            "manifest_sha256": MODEL_MANIFEST_SHA256,
        },
    }
    if contract.get("science_inputs") != expected_science_inputs:
        raise CapsuleError("capsule-bound scientific inputs changed")
    expected_paths = {
        "staged_root": str(STAGED_ROOT),
        "source_contract": str(STAGED_ROOT / SOURCE_CONTRACT_PATH),
        "matrix": str(STAGED_ROOT / MATRIX_PATH),
        "primary_config": str(STAGED_ROOT / PRIMARY_CONFIG_PATH),
        "task_root": str(STAGED_ROOT / TASK_ROOT_PATH),
        "discovery_manifest": str(STAGED_ROOT / DISCOVERY_MANIFEST_PATH),
        "tokenizer_terminal": str(STAGED_ROOT / TOKENIZER_TERMINAL_PATH),
        "model_manifest": str(STAGED_ROOT / MODEL_MANIFEST_PATH),
        "tokenizer_evidence": "/workspace/parallax/tokenizer-preflight.json",
        "cache_root": "/root/.cache/huggingface/hub",
        "output_dir": "/workspace/parallax/output/interface-control",
        "raw_terminal": "/workspace/parallax/terminal.json",
        "critic_output": "/workspace/parallax/independent-critic.json",
        "scratch_pointer": "/workspace/parallax/scratch-pointer.json",
        "failure_pointer": "/workspace/parallax/failure-stop-pointer.json",
        "hydration_evidence": "/workspace/parallax/runtime-hydration.json",
        "artifact_manifest": "/workspace/parallax/artifacts.manifest.json",
        "artifact_bundle": "/workspace/parallax/artifacts.tar",
        "durability_receipts": "/workspace/parallax/durability-receipts.json",
    }
    if contract.get("canonical_paths") != expected_paths:
        raise CapsuleError("canonical native path set changed")
    consumers = contract.get("consumers")
    wiring = contract.get("runtime_wiring")
    if not isinstance(consumers, dict) or set(consumers) != set(CONSUMER_ORDER):
        raise CapsuleError("native consumer set changed")
    if not isinstance(wiring, dict) or set(wiring) != {str(path) for path in RUNTIME_PATHS}:
        raise CapsuleError("native runtime path set changed")
    for path, digest in wiring.items():
        safe_relative(path, label="native runtime")
        _digest(digest, label=f"native runtime {path}")
    for name, module, path in NATIVE_BINDINGS:
        row = consumers[name]
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "schema",
                "module",
                "path",
                "sha256",
                "preflight_argv",
                "execution_argv",
                "execution_callable",
                "receipt_path",
            }
            or row["schema"] != f"pg1-native-{name}/v3"
            or row["module"] != module
            or row["path"] != str(path)
            or row["sha256"] != wiring[str(path)]
            or row["preflight_argv"] != consumer_preflight_argv(name, contract)
            or row["execution_argv"] != consumer_execution_argv(name, contract)
            or row["execution_callable"] != f"{module}:{_execution_callable(name)}"
        ):
            raise CapsuleError(f"consumer {name} binding changed")
        safe_relative(row["receipt_path"], label=f"consumer {name} receipt")
    if contract.get("phase_order") != list(PHASE_ORDER):
        raise CapsuleError("native phase order changed")
    if contract.get("firewall") != {
        "confirmation_items_read": 0,
        "archive_body_reads_local": 0,
        "model_body_reads_preflight": 0,
        "data_body_reads_preflight": 0,
        "torch_imports_preflight": 0,
        "transformers_imports_preflight": 0,
        "cuda_operations_preflight": 0,
        "hydration_operations_preflight": 0,
        "science_operations_preflight": 0,
        "claims_written": 0,
        "automatic_retry": False,
    }:
        raise CapsuleError("native preflight firewall changed")
    if contract.get("authorization_boundary") != {
        "embedded_execution_authority": False,
        "preflight_can_grant_authority": False,
        "separate_authorization_required": True,
        "one_shot_consumption_receipt_required": True,
        "h100_authority": False,
    }:
        raise CapsuleError("capsule authorization boundary changed")
    if contract.get("production_runtime") != {
        "image": RUNTIME_IMAGE,
        "python": {
            "executable": str(PYTHON_EXECUTABLE),
            "hash_receipt": str(STOP_CONTEXT_RUNTIME_PATH),
        },
        "aws": {
            "executable": str(AWS_EXECUTABLE),
            "version": AWS_CLI_VERSION,
            "profile": AWS_PROFILE,
            "region": AWS_REGION,
            "config_path": str(AWS_CONFIG_PATH),
            "credentials_path": str(AWS_CREDENTIALS_PATH),
            "expected_account": EXPECTED_BUCKET_OWNER,
            "expected_arn_prefix": (
                f"arn:aws:sts::{EXPECTED_BUCKET_OWNER}:assumed-role/"
                "parallax-pg1-native-executor/"
            ),
            "hash_receipt": str(STOP_CONTEXT_RUNTIME_PATH),
        },
        "vast": {
            "executable": str(VAST_EXECUTABLE),
            "version": VAST_CLI_VERSION,
            "command_schema": VAST_COMMAND_SCHEMA,
            "api_key_path": str(VAST_API_KEY_PATH),
            "hash_receipt": str(STOP_CONTEXT_RUNTIME_PATH),
            "adapter_path": str(VAST_ADAPTER_PATH),
            "adapter_sha256": wiring[str(VAST_ADAPTER_PATH)],
        },
        "authority": {
            "issuer": PRODUCTION_AUTHORIZATION_ISSUER,
            "bucket": AUTHORIZATION_BUCKET,
            "prefix": AUTHORIZATION_PREFIX,
            "expected_account": EXPECTED_BUCKET_OWNER,
        },
        "stop": {
            "context_path": str(STOP_CONTEXT_RUNTIME_PATH),
            "primary_attempts": 2,
            "fallback_attempts": 2,
            "fallback": "TRACKED_PERSISTENT_GUARDIAN_CONSUMER_V3",
            "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
            "guardian_poll_seconds": GUARDIAN_POLL_SECONDS,
            "guardian_timeout_seconds": GUARDIAN_TIMEOUT_SECONDS,
        },
        "deployment": {
            "actual_future_entrypoint": production_argv(contract),
            "external_deadman_argv": deadman_argv(contract),
            "immutable_image_source_root": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
            "bootstrap_source_root": str(BOOTSTRAP_SOURCE_ROOT),
            "fixed_cwd": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
            "fixed_pythonpath": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
            "fixed_environment": {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/nonexistent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONPATH": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
            },
            "outer_entry": {
                "path": str(PRODUCTION_ENTRY_PATH),
                "sha256": wiring[str(DEPLOYMENT_PATH)],
                "mode": "0755",
            },
            "external_deadman": {
                "path": str(CONTROL_DEADMAN_IMAGE_PATH),
                "sha256": wiring[str(CONTROL_DEADMAN_PATH)],
                "mode": "0755",
                "destroy_authority": False,
                "aws_config": "/run/parallax/control-plane/aws/config",
                "aws_credentials": "/run/parallax/control-plane/aws/credentials",
                "vast_key": "/run/secrets/pg1-native-control-plane-vast-api-key",
                "terminal_observations": 3,
            },
            "launch_invoker": {
                "path": str(LAUNCH_INVOKER_STAGE_PATH),
                "sha256": wiring[str(LAUNCH_INVOKER_PATH)],
                "mode": "0755",
                "causal_order": ["installer", "compiler", "authority", "gateway"],
                "consumes_production_invocation_argv": True,
                "test_only_cli_modes": False,
            },
            "launch_gateway": {
                "path": str(LAUNCH_GATEWAY_IMAGE_PATH),
                "sha256": wiring[str(LAUNCH_GATEWAY_PATH)],
                "argv": launch_gateway_argv(contract),
                "arms_deadman_before_worker_create": True,
                "consumes_production_argv": True,
                "packet_compiler_path": str(LAUNCH_PACKET_IMAGE_PATH),
                "packet_compiler_sha256": wiring[str(LAUNCH_PACKET_PATH)],
                "authority_issuer_sha256": wiring[str(LAUNCH_AUTHORITY_PATH)],
                "canonical_state_sha256": wiring[str(LAUNCH_STATE_PATH)],
                "control_installer_sha256": wiring[str(CONTROL_INSTALLER_PATH)],
                "control_installer_path": str(CONTROL_INSTALLER_IMAGE_PATH),
                "authority_issuer_path": str(LAUNCH_AUTHORITY_IMAGE_PATH),
                "control_install_before_compile_authority_gateway": True,
                "onstart_materializer_sha256": wiring[str(ONSTART_MATERIALIZER_PATH)],
                "idle_bootstrap_path": str(IDLE_BOOTSTRAP_IMAGE_PATH),
                "idle_bootstrap_sha256": wiring[str(IDLE_BOOTSTRAP_PATH)],
                "create_bind_deadman_start_order": True,
                "prelaunch_label_orphan_recovery": True,
            },
            "local_lifecycle": {
                "path": str(LOCAL_LIFECYCLE_PATH),
                "schema": "pg1-native-local-lifecycle/v2",
                "version": 2,
                "cas": "fcntl-lock-expected-sequence-atomic-replace-fsync",
                "ancestry": "complete-bounded-self-digest-chain",
                "states": [
                    "ARMED", "RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED",
                    "TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE",
                ],
                "guardian_action_state": "FAILURE_STOP_REQUIRED",
                "stopped_state": "TERMINAL_STOPPED_PRESERVED",
                "destroyed_state": "TERMINAL_DESTROYED_DURABLE",
            },
            "durable_lease": {
                "bucket": AUTHORIZATION_BUCKET,
                "lease_key_template": (
                    f"{AUTHORIZATION_PREFIX}/external-deadman-leases/"
                    "{capsule_sha256}/{attempt_run_id}.json"
                ),
                "heartbeat_key_template": (
                    f"{AUTHORIZATION_PREFIX}/deadman-heartbeats/"
                    "{capsule_sha256}/{attempt_run_id}.json"
                ),
                "guardian_receipt_key_template": (
                    f"{AUTHORIZATION_PREFIX}/stop-guardian-receipts/"
                    "{capsule_sha256}/{attempt_run_id}.json"
                ),
                "conditional_create": "If-None-Match:*",
                "conditional_update": "If-Match:canonical-etag",
            },
            "layout_receipt_path": str(DEPLOYMENT_LAYOUT_RECEIPT_PATH),
            "bootstrap_argv": bootstrap_argv(contract),
            "fixed_layout_modes": {
                "/opt/parallax/future-freeze": "0700",
                "/run/parallax": "0700",
                "/run/parallax/aws": "0700",
                "/run/secrets": "0700",
                "/workspace/parallax": "0700",
            },
            "fixed_mount_modes": {
                "/run/parallax/pg1-execution-capsule.json": "0400",
                "/run/parallax/pg1-native-stop-context.json": "0400",
                "/run/parallax/aws/config": "0400",
                "/run/parallax/aws/credentials": "0400",
                "/run/secrets/pg1-native-vast-api-key": "0400",
                "/run/parallax/runtime-image-digest": "0400",
                "/run/parallax/pg1-external-deadman-arm.json": "0400",
                "/run/parallax/pg1-native-entry-contract.json": "0400",
                "/run/parallax/pg1-native-start-lease.json": "0400",
                "/run/parallax/pg1-native-handoff-layout.json": "0400",
                "/run/parallax/pg1-future-launch-packet.json": "0400",
                "/run/parallax/pg1-native-installed-source-layout.json": "0400",
            },
        },
    }:
        raise CapsuleError("production runtime identity changed")
    if contract.get("artifact_contract") != artifact_contract(attempt):
        raise CapsuleError("required artifact inventory contract changed")
    if contract.get("entrypoints") != {
        "bootstrap_argv": bootstrap_argv(contract),
        "deadman_argv": deadman_argv(contract),
        "launch_gateway_argv": launch_gateway_argv(contract),
        "production_argv": production_argv(contract),
        "native_launcher_argv": native_launcher_argv(contract),
        "normal_controller_argv": consumer_execution_argv("controller", contract),
        "mechanical_bootstrap_argv": mechanical_bootstrap_argv(contract),
        "mechanical_rehearsal_argv": mechanical_rehearsal_argv(contract),
        "production_deployment": "infra.vast.parallax.pg1_native_deployment:provision_and_launch",
        "production_launcher": "infra.vast.parallax.pg1_native_launcher:execute",
        "production_bootstrap": "infra.vast.parallax.pg1_native_bootstrap:deploy",
        "production_guardian": "infra.vast.parallax.pg1_native_guardian:guard",
        "production_control_deadman": (
            "infra.vast.parallax.pg1_native_control_deadman:run"
        ),
        "production_launch_gateway": (
            "infra.vast.parallax.pg1_native_launch_gateway:launch"
        ),
        "production_backend": "infra.vast.parallax.pg1_native_backend:ProductionBackend",
        "production_authority_store": (
            "infra.vast.parallax.pg1_native_authority:ProductionAuthorizationStore"
        ),
        "rehearsal_harness": "infra.vast.parallax.pg1_native_rehearsal:run",
    }:
        raise CapsuleError("native entrypoint wiring changed")
    return contract


def validate_runtime_source(
    capsule: dict[str, Any], declared_path: Path, runtime_path: Path
) -> None:
    expected = capsule["contract"]["runtime_wiring"].get(str(declared_path))
    if expected is None or not runtime_path.is_file() or sha256_file(runtime_path) != expected:
        raise CapsuleError(f"native runtime source disagrees: {declared_path}")


def validate_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    if set(capsule) != {"schema_version", "status", "content_address", "contract"}:
        raise CapsuleError("capsule envelope field set changed")
    contract = capsule.get("contract")
    if (
        capsule.get("schema_version") != 3
        or capsule.get("status") != CAPSULE_STATUS
        or not isinstance(contract, dict)
        or capsule.get("content_address")
        != {"algorithm": "sha256", "digest": content_sha256(contract)}
    ):
        raise CapsuleError("capsule content address disagrees")
    validate_contract_schema(contract)
    validate_runtime_source(capsule, CORE_PATH, Path(__file__).resolve())
    return capsule


def validate_consumer(
    capsule: dict[str, Any], name: str, runtime_path: Path
) -> dict[str, Any]:
    validate_capsule(capsule)
    _, declared_path = _binding(name)
    validate_runtime_source(capsule, declared_path, runtime_path.resolve())
    return capsule["contract"]["consumers"][name]


def load_capsule(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CapsuleError("capsule is not an object")
    return validate_capsule(value)


def validate_execution_authorization(
    authorization: dict[str, Any], capsule: dict[str, Any]
) -> dict[str, Any]:
    validate_capsule(capsule)
    if not isinstance(authorization, dict) or set(authorization) != {
        "schema_version",
        "status",
        "content_address",
        "contract",
    }:
        raise CapsuleError("execution authorization envelope changed")
    contract = authorization.get("contract")
    if (
        authorization.get("schema_version") != 2
        or authorization.get("status") != AUTHORIZATION_STATUS
        or not isinstance(contract, dict)
        or authorization.get("content_address")
        != {"algorithm": "sha256", "digest": content_sha256(contract)}
        or set(contract)
        != {
            "authorization_id",
            "capsule_sha256",
            "source_run_id",
            "attempt_run_id",
            "scope",
            "one_shot",
            "h100_authority",
            "issuer",
            "issuer_backend",
            "issuer_account",
            "store_uri",
            "issued_key",
            "instance_id",
            "stop_context_sha256",
            "offer_id",
            "machine_id",
            "container_id",
        }
    ):
        raise CapsuleError("execution authorization schema or address changed")
    capsule_contract = capsule["contract"]
    if (
        contract.get("authorization_id") != production_authorization_id(capsule)
        or contract.get("capsule_sha256") != capsule["content_address"]["digest"]
        or contract.get("source_run_id")
        != capsule_contract["source_identity"]["source_run_id"]
        or contract.get("attempt_run_id")
        != capsule_contract["attempt_identity"]["attempt_run_id"]
        or contract.get("scope") != "PG1_NATIVE_ONE_SHOT_EXECUTION"
        or contract.get("one_shot") is not True
        or contract.get("h100_authority") is not True
        or contract.get("issuer") != PRODUCTION_AUTHORIZATION_ISSUER
        or contract.get("issuer_backend") != "S3_CONDITIONAL_CREATE_ISSUER_V2"
        or contract.get("issuer_account") != EXPECTED_BUCKET_OWNER
        or contract.get("store_uri")
        != f"s3://{AUTHORIZATION_BUCKET}/{AUTHORIZATION_PREFIX}"
        or contract.get("issued_key") != production_issued_key(capsule)
        or not isinstance(contract.get("instance_id"), int)
        or contract["instance_id"] <= 0
        or not isinstance(contract.get("stop_context_sha256"), str)
        or len(contract["stop_context_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in contract["stop_context_sha256"]
        )
        or not isinstance(contract.get("offer_id"), int)
        or contract["offer_id"] <= 0
        or not isinstance(contract.get("machine_id"), int)
        or contract["machine_id"] <= 0
        or not isinstance(contract.get("container_id"), str)
        or not contract["container_id"]
    ):
        raise CapsuleError("execution authorization is false, wrong, or unbound")
    return authorization


def validate_authorization_receipt(
    receipt: dict[str, Any],
    authorization: dict[str, Any],
    capsule: dict[str, Any],
) -> dict[str, Any]:
    validate_execution_authorization(authorization, capsule)
    expected = {
        "schema": AUTHORIZATION_RECEIPT_SCHEMA,
        "status": "CONSUMED_EXACTLY_ONCE",
        "authorization_sha256": authorization["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "source_run_id": capsule["contract"]["source_identity"]["source_run_id"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "one_shot_consumed": True,
        "h100_authority": True,
        "issuer": authorization["contract"]["issuer"],
        "issuer_backend": authorization["contract"]["issuer_backend"],
        "issuer_account": authorization["contract"]["issuer_account"],
        "store_uri": authorization["contract"]["store_uri"],
        "consume_key": production_consumed_key(capsule, authorization),
        "consume_version_id": receipt.get("consume_version_id"),
        "consume_etag": receipt.get("consume_etag"),
        "consume_checksum_sha256": receipt.get("consume_checksum_sha256"),
        "consume_marker_sha256": content_sha256(
            production_consumption_marker(capsule, authorization)
        ),
    }
    if (
        receipt != expected
        or not all(
            isinstance(receipt.get(field), str) and receipt[field]
            for field in (
                "consume_key",
                "consume_version_id",
                "consume_etag",
                "consume_checksum_sha256",
                "consume_marker_sha256",
            )
        )
    ):
        raise CapsuleError("one-shot authorization receipt is absent, wrong, or reused")
    return receipt


def consumer_receipt(
    capsule: dict[str, Any], name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    validate_capsule(capsule)
    _binding(name)
    return {
        "status": "GREEN_NATIVE_CONSUMER_PREFLIGHT_V3",
        "consumer": name,
        "capsule_sha256": capsule["content_address"]["digest"],
        "source_run_id": capsule["contract"]["source_identity"]["source_run_id"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "payload": copy.deepcopy(payload),
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


def compile_contract(root: Path, *, attempt_run_id: str) -> dict[str, Any]:
    if not attempt_run_id or attempt_run_id == SOURCE_RUN_ID:
        raise CapsuleError("attempt_run_id must be explicit and distinct")
    root = root.resolve()
    source_path = root / SOURCE_CONTRACT_PATH
    if not source_path.is_file() or sha256_file(source_path) != SOURCE_CONTRACT_SHA256:
        raise CapsuleError("frozen source contract changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("run_id") != SOURCE_RUN_ID
        or source.get("confirmation_items_read") != 0
        or source.get("paid_authorization") is not False
        or not isinstance(source.get("artifacts"), dict)
    ):
        raise CapsuleError("frozen source identity or firewall changed")
    bundle_path = root / BUNDLE_RECEIPT_PATH
    overlay_path = root / OVERLAY_RECEIPT_PATH
    if sha256_file(bundle_path) != BUNDLE_RECEIPT_SHA256:
        raise CapsuleError("bundle receipt changed")
    if sha256_file(overlay_path) != OVERLAY_RECEIPT_SHA256:
        raise CapsuleError("overlay receipt changed")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    bundled = {str(row["path"]): str(row["sha256"]) for row in bundle["entries"]}
    overlaid = {str(row["path"]): str(row["sha256"]) for row in overlay["objects"]}
    rows = copy.deepcopy(source["artifacts"])
    rows["source_contract"] = {
        "path": str(SOURCE_CONTRACT_PATH),
        "sha256": SOURCE_CONTRACT_SHA256,
    }
    artifacts: dict[str, dict[str, str]] = {}
    for label, row in sorted(rows.items()):
        path = str(safe_relative(row["path"], label=f"artifact {label}"))
        digest = _digest(row["sha256"], label=f"artifact {label}")
        origin = (
            "exact_overlay"
            if overlaid.get(path) == digest
            else "base_bundle"
            if bundled.get(path) == digest
            else None
        )
        if origin is None:
            raise CapsuleError(f"artifact {label} has no receipt-bound origin")
        artifacts[label] = {"path": path, "sha256": digest, "origin": origin}
    runtime_wiring = {str(path): sha256_file(root / path) for path in RUNTIME_PATHS}
    contract: dict[str, Any] = {
        "capsule_schema": CAPSULE_SCHEMA,
        "source_identity": {
            "source_run_id": SOURCE_RUN_ID,
            "source_contract_path": str(SOURCE_CONTRACT_PATH),
            "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        },
        "attempt_identity": {
            "attempt_run_id": attempt_run_id,
            "artifact_key": f"parallax/results/{attempt_run_id}/artifacts.tar",
            "artifact_manifest_key": (
                f"parallax/results/{attempt_run_id}/artifacts.manifest.json"
            ),
            "artifact_receipts_key": (
                f"parallax/results/{attempt_run_id}/durability-receipts.json"
            ),
            "terminal_key": f"parallax/results/{attempt_run_id}/terminal.json",
            "scratch_key": f"parallax/results/{attempt_run_id}/scratch-pointer.json",
            "critic_key": f"parallax/results/{attempt_run_id}/independent-critic.json",
        },
        "artifacts": artifacts,
        "materialization": {
            "bundle_receipt_path": str(BUNDLE_RECEIPT_PATH),
            "bundle_receipt_sha256": BUNDLE_RECEIPT_SHA256,
            "overlay_receipt_path": str(OVERLAY_RECEIPT_PATH),
            "overlay_receipt_sha256": OVERLAY_RECEIPT_SHA256,
            "remote_receipt_path": "/run/parallax/pg1-materialization-receipt.json",
            "exact_archive_rehearsal": "PENDING_BOUNDED_REMOTE_CPU_REHEARSAL",
        },
        "consumers": {},
        "runtime_wiring": runtime_wiring,
        "phase_order": list(PHASE_ORDER),
        "firewall": {
            "confirmation_items_read": 0,
            "archive_body_reads_local": 0,
            "model_body_reads_preflight": 0,
            "data_body_reads_preflight": 0,
            "torch_imports_preflight": 0,
            "transformers_imports_preflight": 0,
            "cuda_operations_preflight": 0,
            "hydration_operations_preflight": 0,
            "science_operations_preflight": 0,
            "claims_written": 0,
            "automatic_retry": False,
        },
        "authorization_boundary": {
            "embedded_execution_authority": False,
            "preflight_can_grant_authority": False,
            "separate_authorization_required": True,
            "one_shot_consumption_receipt_required": True,
            "h100_authority": False,
        },
        "science_inputs": {
            "matrix": {"path": str(MATRIX_PATH), "sha256": MATRIX_SHA256},
            "primary_config": {
                "path": str(PRIMARY_CONFIG_PATH),
                "sha256": PRIMARY_CONFIG_SHA256,
            },
            "task_root": str(TASK_ROOT_PATH),
            "discovery_manifest": {
                "path": str(DISCOVERY_MANIFEST_PATH),
                "sha256": DISCOVERY_MANIFEST_SHA256,
            },
            "tokenizer_terminal": {
                "path": str(TOKENIZER_TERMINAL_PATH),
                "sha256": TOKENIZER_TERMINAL_SHA256,
            },
            "lineage": LINEAGE,
            "lineage_role": LINEAGE_ROLE,
            "entity_role": ENTITY_ROLE,
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "manifest_path": str(MODEL_MANIFEST_PATH),
                "manifest_sha256": MODEL_MANIFEST_SHA256,
            },
        },
        "canonical_paths": {
            "staged_root": str(STAGED_ROOT),
            "source_contract": str(STAGED_ROOT / SOURCE_CONTRACT_PATH),
            "matrix": str(STAGED_ROOT / MATRIX_PATH),
            "primary_config": str(STAGED_ROOT / PRIMARY_CONFIG_PATH),
            "task_root": str(STAGED_ROOT / TASK_ROOT_PATH),
            "discovery_manifest": str(STAGED_ROOT / DISCOVERY_MANIFEST_PATH),
            "tokenizer_terminal": str(STAGED_ROOT / TOKENIZER_TERMINAL_PATH),
            "model_manifest": str(STAGED_ROOT / MODEL_MANIFEST_PATH),
            "tokenizer_evidence": "/workspace/parallax/tokenizer-preflight.json",
            "cache_root": "/root/.cache/huggingface/hub",
            "output_dir": "/workspace/parallax/output/interface-control",
            "raw_terminal": "/workspace/parallax/terminal.json",
            "critic_output": "/workspace/parallax/independent-critic.json",
            "scratch_pointer": "/workspace/parallax/scratch-pointer.json",
            "failure_pointer": "/workspace/parallax/failure-stop-pointer.json",
            "hydration_evidence": "/workspace/parallax/runtime-hydration.json",
            "artifact_manifest": "/workspace/parallax/artifacts.manifest.json",
            "artifact_bundle": "/workspace/parallax/artifacts.tar",
            "durability_receipts": "/workspace/parallax/durability-receipts.json",
        },
        "production_runtime": {
            "image": RUNTIME_IMAGE,
            "python": {
                "executable": str(PYTHON_EXECUTABLE),
                "hash_receipt": str(STOP_CONTEXT_RUNTIME_PATH),
            },
            "aws": {
                "executable": str(AWS_EXECUTABLE),
                "version": AWS_CLI_VERSION,
                "profile": AWS_PROFILE,
                "region": AWS_REGION,
                "config_path": str(AWS_CONFIG_PATH),
                "credentials_path": str(AWS_CREDENTIALS_PATH),
                "expected_account": EXPECTED_BUCKET_OWNER,
                "expected_arn_prefix": (
                    f"arn:aws:sts::{EXPECTED_BUCKET_OWNER}:assumed-role/"
                    "parallax-pg1-native-executor/"
                ),
                "hash_receipt": str(STOP_CONTEXT_RUNTIME_PATH),
            },
            "vast": {
                "executable": str(VAST_EXECUTABLE),
                "version": VAST_CLI_VERSION,
                "command_schema": VAST_COMMAND_SCHEMA,
                "api_key_path": str(VAST_API_KEY_PATH),
                "hash_receipt": str(STOP_CONTEXT_RUNTIME_PATH),
                "adapter_path": str(VAST_ADAPTER_PATH),
                "adapter_sha256": runtime_wiring[str(VAST_ADAPTER_PATH)],
            },
            "authority": {
                "issuer": PRODUCTION_AUTHORIZATION_ISSUER,
                "bucket": AUTHORIZATION_BUCKET,
                "prefix": AUTHORIZATION_PREFIX,
                "expected_account": EXPECTED_BUCKET_OWNER,
            },
            "stop": {
                "context_path": str(STOP_CONTEXT_RUNTIME_PATH),
                "primary_attempts": 2,
                "fallback_attempts": 2,
                "fallback": "TRACKED_PERSISTENT_GUARDIAN_CONSUMER_V3",
                "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
                "guardian_poll_seconds": GUARDIAN_POLL_SECONDS,
                "guardian_timeout_seconds": GUARDIAN_TIMEOUT_SECONDS,
            },
            "deployment": {
                "actual_future_entrypoint": production_argv({}),
                "external_deadman_argv": deadman_argv({}),
                "immutable_image_source_root": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
                "bootstrap_source_root": str(BOOTSTRAP_SOURCE_ROOT),
                "fixed_cwd": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
                "fixed_pythonpath": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
                "fixed_environment": {
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/nonexistent",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONPATH": str(IMMUTABLE_IMAGE_SOURCE_ROOT),
                },
                "outer_entry": {
                    "path": str(PRODUCTION_ENTRY_PATH),
                    "sha256": runtime_wiring[str(DEPLOYMENT_PATH)],
                    "mode": "0755",
                },
                "external_deadman": {
                    "path": str(CONTROL_DEADMAN_IMAGE_PATH),
                    "sha256": runtime_wiring[str(CONTROL_DEADMAN_PATH)],
                    "mode": "0755",
                    "destroy_authority": False,
                    "aws_config": "/run/parallax/control-plane/aws/config",
                    "aws_credentials": "/run/parallax/control-plane/aws/credentials",
                    "vast_key": "/run/secrets/pg1-native-control-plane-vast-api-key",
                    "terminal_observations": 3,
                },
                "launch_invoker": {
                    "path": str(LAUNCH_INVOKER_STAGE_PATH),
                    "sha256": runtime_wiring[str(LAUNCH_INVOKER_PATH)],
                    "mode": "0755",
                    "causal_order": ["installer", "compiler", "authority", "gateway"],
                    "consumes_production_invocation_argv": True,
                    "test_only_cli_modes": False,
                },
                "launch_gateway": {
                    "path": str(LAUNCH_GATEWAY_IMAGE_PATH),
                    "sha256": runtime_wiring[str(LAUNCH_GATEWAY_PATH)],
                    "argv": launch_gateway_argv({}),
                    "arms_deadman_before_worker_create": True,
                    "consumes_production_argv": True,
                    "packet_compiler_path": str(LAUNCH_PACKET_IMAGE_PATH),
                    "packet_compiler_sha256": runtime_wiring[str(LAUNCH_PACKET_PATH)],
                    "authority_issuer_sha256": runtime_wiring[str(LAUNCH_AUTHORITY_PATH)],
                    "canonical_state_sha256": runtime_wiring[str(LAUNCH_STATE_PATH)],
                    "control_installer_sha256": runtime_wiring[str(CONTROL_INSTALLER_PATH)],
                    "control_installer_path": str(CONTROL_INSTALLER_IMAGE_PATH),
                    "authority_issuer_path": str(LAUNCH_AUTHORITY_IMAGE_PATH),
                    "control_install_before_compile_authority_gateway": True,
                    "onstart_materializer_sha256": runtime_wiring[str(ONSTART_MATERIALIZER_PATH)],
                    "idle_bootstrap_path": str(IDLE_BOOTSTRAP_IMAGE_PATH),
                    "idle_bootstrap_sha256": runtime_wiring[str(IDLE_BOOTSTRAP_PATH)],
                    "create_bind_deadman_start_order": True,
                    "prelaunch_label_orphan_recovery": True,
                },
                "local_lifecycle": {
                    "path": str(LOCAL_LIFECYCLE_PATH),
                    "schema": "pg1-native-local-lifecycle/v2",
                    "version": 2,
                    "cas": "fcntl-lock-expected-sequence-atomic-replace-fsync",
                    "ancestry": "complete-bounded-self-digest-chain",
                    "states": [
                        "ARMED", "RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED",
                        "TERMINAL_STOPPED_PRESERVED", "TERMINAL_DESTROYED_DURABLE",
                    ],
                    "guardian_action_state": "FAILURE_STOP_REQUIRED",
                    "stopped_state": "TERMINAL_STOPPED_PRESERVED",
                    "destroyed_state": "TERMINAL_DESTROYED_DURABLE",
                },
                "durable_lease": {
                    "bucket": AUTHORIZATION_BUCKET,
                    "lease_key_template": (
                        f"{AUTHORIZATION_PREFIX}/external-deadman-leases/"
                        "{capsule_sha256}/{attempt_run_id}.json"
                    ),
                    "heartbeat_key_template": (
                        f"{AUTHORIZATION_PREFIX}/deadman-heartbeats/"
                        "{capsule_sha256}/{attempt_run_id}.json"
                    ),
                    "guardian_receipt_key_template": (
                        f"{AUTHORIZATION_PREFIX}/stop-guardian-receipts/"
                        "{capsule_sha256}/{attempt_run_id}.json"
                    ),
                    "conditional_create": "If-None-Match:*",
                    "conditional_update": "If-Match:canonical-etag",
                },
                "layout_receipt_path": str(DEPLOYMENT_LAYOUT_RECEIPT_PATH),
                "bootstrap_argv": bootstrap_argv({}),
                "fixed_layout_modes": {
                    "/opt/parallax/future-freeze": "0700",
                    "/run/parallax": "0700",
                    "/run/parallax/aws": "0700",
                    "/run/secrets": "0700",
                    "/workspace/parallax": "0700",
                },
                "fixed_mount_modes": {
                    "/run/parallax/pg1-execution-capsule.json": "0400",
                    "/run/parallax/pg1-native-stop-context.json": "0400",
                    "/run/parallax/aws/config": "0400",
                    "/run/parallax/aws/credentials": "0400",
                    "/run/secrets/pg1-native-vast-api-key": "0400",
                    "/run/parallax/runtime-image-digest": "0400",
                    "/run/parallax/pg1-external-deadman-arm.json": "0400",
                    "/run/parallax/pg1-native-entry-contract.json": "0400",
                    "/run/parallax/pg1-native-start-lease.json": "0400",
                    "/run/parallax/pg1-native-handoff-layout.json": "0400",
                    "/run/parallax/pg1-future-launch-packet.json": "0400",
                    "/run/parallax/pg1-native-installed-source-layout.json": "0400",
                },
            },
        },
        "artifact_contract": {},
        "entrypoints": {},
    }
    for name, module, path in NATIVE_BINDINGS:
        contract["consumers"][name] = {
            "schema": f"pg1-native-{name}/v3",
            "module": module,
            "path": str(path),
            "sha256": runtime_wiring[str(path)],
            "preflight_argv": [],
            "execution_argv": [],
            "execution_callable": f"{module}:{_execution_callable(name)}",
            "receipt_path": f"infra/vast/parallax/operations/readiness/pg1_v3/{name}.json",
        }
    for name in CONSUMER_ORDER:
        contract["consumers"][name]["preflight_argv"] = consumer_preflight_argv(
            name, contract
        )
        contract["consumers"][name]["execution_argv"] = consumer_execution_argv(
            name, contract
        )
    contract["artifact_contract"] = artifact_contract(contract["attempt_identity"])
    contract["entrypoints"] = {
        "bootstrap_argv": bootstrap_argv(contract),
        "deadman_argv": deadman_argv(contract),
        "launch_gateway_argv": launch_gateway_argv(contract),
        "production_argv": production_argv(contract),
        "native_launcher_argv": native_launcher_argv(contract),
        "normal_controller_argv": consumer_execution_argv("controller", contract),
        "mechanical_bootstrap_argv": mechanical_bootstrap_argv(contract),
        "mechanical_rehearsal_argv": mechanical_rehearsal_argv(contract),
        "production_deployment": "infra.vast.parallax.pg1_native_deployment:provision_and_launch",
        "production_launcher": "infra.vast.parallax.pg1_native_launcher:execute",
        "production_bootstrap": "infra.vast.parallax.pg1_native_bootstrap:deploy",
        "production_guardian": "infra.vast.parallax.pg1_native_guardian:guard",
        "production_control_deadman": (
            "infra.vast.parallax.pg1_native_control_deadman:run"
        ),
        "production_launch_gateway": (
            "infra.vast.parallax.pg1_native_launch_gateway:launch"
        ),
        "production_backend": "infra.vast.parallax.pg1_native_backend:ProductionBackend",
        "production_authority_store": (
            "infra.vast.parallax.pg1_native_authority:ProductionAuthorizationStore"
        ),
        "rehearsal_harness": "infra.vast.parallax.pg1_native_rehearsal:run",
    }
    validate_contract_schema(contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise CapsuleError("refusing to overwrite capsule")
    capsule = addressed(
        compile_contract(args.root.resolve(), attempt_run_id=args.attempt_run_id)
    )
    atomic_json(args.output, capsule)
    print(json.dumps({"status": capsule["status"], "sha256": capsule["content_address"]["digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
