"""Launch one S3-durable primary PG1 interface discovery on the cheapest H100."""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import ephemeral_interface_control_launch as legacy
from . import ephemeral_optimizer_scale_launch as optimizer
from .artifacts import atomic_json, sha256_file
from .lifecycle_supervisor import LifecycleError
from .primary_interface_control_job import (
    EXTERNAL_PROCESSOR_S3_VERSION_ID,
    EXTERNAL_PROCESSOR_TERMINAL_PATH,
    EXTERNAL_PROCESSOR_TERMINAL_SHA256,
    LINEAGE,
    MODEL_BYTES,
    MODEL_NAMES,
    OVERLAY_IDENTITIES,
    RUN_ID,
    SOURCE_RUN_ID,
    SOURCE_CONTRACT_PATH,
    validate_contract as validate_job_contract,
)
from .primary_interface_control_source_gate import (
    expected_overlay_identities,
    validate_local as validate_primary_source,
)
from .primary_interface_control_worker import (
    EXPECTED_COMMAND,
    validate_contract as validate_worker_contract,
)


MODEL_MANIFEST_PATH = (
    "studies/lineage/cross-lineage/config/parallax_pg1_h100_s3_hydration_job_v1.json"
)
OVERLAY_RECEIPT_PATH = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg1_primary_interface_control_overlays_s3_v1.json"
)
FREEZE_STATUS = "FROZEN_ONE_SHOT_PRIMARY_PG1_INTERFACE_AFTER_NO_CREATE_GREEN"
NO_CREATE_STATUS = "GREEN_PRIMARY_PG1_INTERFACE_CONTROL_NO_CREATE"
PRIMARY_SCOPE = "pg1_primary_interface_discovery_nonpromotion_independent_of_pg2"
PRIMARY_PROMOTION_BOUNDARY = (
    "PRIMARY_PG1_DISCOVERY_ONLY_NO_CONFIRMATION_PROMOTION_OR_PG1_TO_PG2_CAUSAL_CHAIN"
)
FRESH_SPEC_STATUS = "FROZEN_FRESH_OPERATIONAL_IDENTITY_BEFORE_NO_CREATE"
FRESH_SPEC_PATH = Path(
    "infra/vast/parallax/operations/readiness/"
    "pg1_equal_token_interface_seed17_fresh_v2_spec.json"
)
PREDECESSOR_SPEC_PATH = Path(
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_primary_interface_control_s3_launch_v5.json"
)
PREDECESSOR_SPEC_SHA256 = "eb3e53611bde17192c8270f67fca1abcac84e65649cb7f99b0cc2a0c5b6554b8"
FRESH_CLAIM_PATH = (
    "infra/vast/parallax/operations/readiness/claims/"
    "pg1_equal_token_interface_seed17_fresh_v2.json"
)
FRESH_OUTPUT_RECEIPT_PATH = (
    "infra/vast/parallax/operations/readiness/"
    "pg1_equal_token_interface_seed17_fresh_v2_attempt/receipt.json"
)
CRITIC_COMMIT = "b1ad4f86b1926c7d0fda6020267672b57adf1c8f"
CRITIC_PATH = "infra/vast/parallax/audit_primary_interface_independent.py"
CRITIC_SHA256 = "d6b03945ed389a48b764fdeef509aa22818c0051654534f789559c4ba05ee8c4"
CRITIC_FREEZE_PATH = (
    "studies/lineage/cross-lineage/research/"
    "2026-07-19_parallax_primary_critic_freeze.md"
)
CRITIC_FREEZE_SHA256 = "ff016face127f54a35a82170893d52872474c675b2978874cd6e247137cdf4af"
FATAL_PROVIDER_BOOTSTRAP_SIGNATURES = (
    "OCI runtime create failed",
    "failed to create task for container",
    "kaalia_docker_shim did not terminate successfully",
)


class PrimaryInterfaceLaunchError(RuntimeError):
    pass


def _source_contract(root: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    path = root / SOURCE_CONTRACT_PATH
    contract = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    identities = expected_overlay_identities(root, contract)
    if identities != OVERLAY_IDENTITIES or identities.get(SOURCE_CONTRACT_PATH) != digest:
        raise PrimaryInterfaceLaunchError("primary source contract overlay identity changed")
    return contract, digest, identities


def _model_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = legacy._resolve_bound(root, spec["inputs"]["model_manifest"], label="PG1 model manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if (
        spec["inputs"]["model_manifest"].get("path") != MODEL_MANIFEST_PATH
        or manifest.get("status") != "FROZEN"
        or manifest.get("run_id") != "parallax-pg1-h100-s3-hydration-v1"
        or manifest.get("scope") != "s3_worker_cache_hydration_only"
        or manifest.get("expected_total_bytes") != MODEL_BYTES
        or manifest.get("parallel_streams") != 3
        or manifest.get("runtime_timeout_seconds") != 600
        or manifest.get("worker_snapshot_root")
        != (
            "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
            "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
        )
        or manifest.get("confirmation_items_read") != 0
        or manifest.get("science_items_evaluated") != 0
        or manifest.get("model_loads") != 0
        or manifest.get("gpu_work") != 0
        or manifest.get("automatic_retry") is not False
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise PrimaryInterfaceLaunchError("primary PG1 model manifest is not exact")
    for index, row in enumerate(rows):
        legacy._validate_input(row, label=f"primary PG1 model {index}")
    if (
        {str(row.get("path", "")) for row in rows} != MODEL_NAMES
        or sum(int(row["bytes"]) for row in rows) != MODEL_BYTES
    ):
        raise PrimaryInterfaceLaunchError("primary PG1 model inventory changed")
    return rows


def _overlay_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = legacy._resolve_bound(root, spec["inputs"]["overlay_receipt"], label="primary overlay receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _contract, _digest, identities = _source_contract(root)
    rows = receipt.get("objects")
    if (
        spec["inputs"]["overlay_receipt"].get("path") != OVERLAY_RECEIPT_PATH
        or receipt.get("status") != "GREEN_PG1_PRIMARY_INTERFACE_OVERLAYS_S3"
        or receipt.get("run_id") != SOURCE_RUN_ID
        or receipt.get("lineage") != LINEAGE
        or receipt.get("role")
        != "primary_within_lineage_discovery_independent_of_supporting_pg2"
        or receipt.get("aws_profile") != "fuelos"
        or receipt.get("region") != legacy.REGION
        or receipt.get("bucket") != legacy.BUCKET
        or receipt.get("object_count") != len(identities)
        or receipt.get("vast_resources_created") != 0
        or receipt.get("vast_persistent_volumes_created") != 0
        or receipt.get("gpu_seconds") != 0
        or receipt.get("paid_compute_authorization") is not False
        or not isinstance(rows, list)
        or len(rows) != len(identities)
    ):
        raise PrimaryInterfaceLaunchError("primary interface overlay receipt is not exact")
    observed: dict[str, str] = {}
    for index, row in enumerate(rows):
        legacy._validate_input(row, label=f"primary interface overlay {index}")
        path_name = str(row.get("path", ""))
        if path_name in observed or row.get("controller_head_exact_match") is not True:
            raise PrimaryInterfaceLaunchError("primary overlay is duplicated or lacks exact HEAD")
        observed[path_name] = str(row["sha256"])
    if observed != identities:
        raise PrimaryInterfaceLaunchError("primary interface overlay identities changed")
    return rows


def _exact_guardian() -> dict[str, Any]:
    return {
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "gpu_kind": "non_h100_mechanical_guard",
        "image": (
            "vastai/base-image@sha256:"
            "a264d0c242bebbdaa6f50b34fb81c7e06cb21d8487a202f27cce22794a2e68e4"
        ),
        "disk_gb": 10,
        "minimum_reliability": 0.98,
        "minimum_inet_down_mbps": 100,
        "minimum_cuda": 0,
        "price_ceiling_dph": 0.12,
        "bootstrap_seconds": 300,
        "discovery_seconds": 180,
        "provider_loading_seconds": 900,
        "worker_hard_runtime_seconds": 1320,
        "hard_destroy_seconds": 1800,
    }


def _exact_worker() -> dict[str, Any]:
    return {
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "gpu_kind": "single_h100_80gb_or_larger",
        "launch_mode": "managed_template_private_runtime",
        "image": legacy.RUNTIME_IMAGE,
        "template_hash": legacy.RUNTIME_TEMPLATE,
        "disk_gb": 80,
        "minimum_gpu_ram_mb": 80000,
        "minimum_cpu_ram_mb": 64000,
        "minimum_cpu_cores_effective": 8,
        "minimum_disk_bw_mb_s": 500,
        "minimum_reliability": 0.98,
        "minimum_inet_down_mbps": 500,
        "minimum_cuda": 13.0,
        "price_ceiling_dph": 2.25,
        "bootstrap_seconds": 600,
        "hard_destroy_seconds": 1500,
        "running_ssh_deadline_seconds": 60,
    }


def _exact_bounds() -> dict[str, int]:
    return {
        "parallel_downloads": 3,
        "processor_download_timeout_seconds": 180,
        "weight_download_timeout_seconds": 600,
        "tokenizer_preflight_timeout_seconds": 120,
        "control_timeout_seconds": 900,
        "artifact_upload_timeout_seconds": 600,
        "minimum_free_bytes": 50_000_000_000,
        "maximum_terminal_bytes": legacy.MAXIMUM_RESULT_BYTES,
    }


def _materialize_fresh_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    if spec_path.resolve() != (root / FRESH_SPEC_PATH).resolve():
        raise PrimaryInterfaceLaunchError("fresh operational spec path changed")
    fresh = json.loads(spec_path.read_text(encoding="utf-8"))
    if set(fresh) != {
        "schema_version",
        "created_utc",
        "status",
        "run_id",
        "predecessor_spec",
        "operational_identity",
        "implementation_overrides",
        "independent_critic",
        "preservation",
        "authorization",
    }:
        raise PrimaryInterfaceLaunchError("fresh operational spec schema changed")
    if (
        fresh.get("schema_version") != 1
        or fresh.get("status") != FRESH_SPEC_STATUS
        or fresh.get("run_id") != RUN_ID
        or fresh.get("predecessor_spec")
        != {
            "path": str(PREDECESSOR_SPEC_PATH),
            "sha256": PREDECESSOR_SPEC_SHA256,
            "run_id": SOURCE_RUN_ID,
        }
        or fresh.get("operational_identity")
        != {
            "artifact_key": f"parallax/results/{RUN_ID}/artifacts.tar",
            "terminal_key": f"parallax/results/{RUN_ID}/terminal.json",
            "claim_path": FRESH_CLAIM_PATH,
            "output_receipt_path": FRESH_OUTPUT_RECEIPT_PATH,
        }
        or fresh.get("independent_critic")
        != {
            "commit": CRITIC_COMMIT,
            "path": CRITIC_PATH,
            "sha256": CRITIC_SHA256,
            "freeze_path": CRITIC_FREEZE_PATH,
            "freeze_sha256": CRITIC_FREEZE_SHA256,
        }
        or fresh.get("preservation")
        != {
            "mode": "EXACT_PREDECESSOR_DERIVATION_OPERATIONAL_FIELDS_ONLY",
            "scientific_change_count": 0,
            "preserved_exact": [
                "model_inputs",
                "training_and_discovery_items",
                "seed17",
                "optimizer_and_step_schedule",
                "primary_metric_and_thresholds",
                "confirmation_firewall",
                "frozen_science_specs_and_configs",
            ],
        }
        or fresh.get("authorization")
        != {
            "paid_action": False,
            "execution_freeze": False,
            "no_create": True,
            "current_source_and_market_reads": True,
        }
    ):
        raise PrimaryInterfaceLaunchError("fresh operational identity or preservation changed")
    predecessor_path = root / PREDECESSOR_SPEC_PATH
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path) != PREDECESSOR_SPEC_SHA256
    ):
        raise PrimaryInterfaceLaunchError("fresh identity predecessor spec changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("run_id") != SOURCE_RUN_ID
        or predecessor.get("status") != "FROZEN_ONE_SHOT_BEFORE_H100"
        or predecessor.get("automatic_paid_retry") is not False
    ):
        raise PrimaryInterfaceLaunchError("fresh identity predecessor is not exact")
    overrides = fresh.get("implementation_overrides")
    expected_overrides = {
        "primary_launcher",
        "primary_no_create",
        "primary_job",
        "primary_worker",
        "primary_source_gate",
        "primary_overlay_stage",
    }
    if not isinstance(overrides, dict) or set(overrides) != expected_overrides:
        raise PrimaryInterfaceLaunchError("fresh implementation override inventory changed")
    for label, row in overrides.items():
        if row.get("path") != predecessor["implementation"][label]["path"]:
            raise PrimaryInterfaceLaunchError("fresh implementation override path changed")
    spec = copy.deepcopy(predecessor)
    spec["created_utc"] = fresh["created_utc"]
    spec["run_id"] = RUN_ID
    spec["artifact"]["key"] = fresh["operational_identity"]["artifact_key"]
    spec["result"]["key"] = fresh["operational_identity"]["terminal_key"]
    spec["one_shot_authorization"] = {
        "id": RUN_ID,
        "claim_path": fresh["operational_identity"]["claim_path"],
    }
    spec["output_receipt"] = {
        "path": fresh["operational_identity"]["output_receipt_path"]
    }
    spec["implementation"].update(copy.deepcopy(overrides))
    spec["bound_artifacts"].update(
        {
            "independent_critic": {
                "path": CRITIC_PATH,
                "sha256": CRITIC_SHA256,
            },
            "independent_critic_freeze": {
                "path": CRITIC_FREEZE_PATH,
                "sha256": CRITIC_FREEZE_SHA256,
            },
        }
    )
    return spec


def validate_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    spec = _materialize_fresh_spec(root, spec_path)
    expected_top = {
        "schema_version", "created_utc", "program", "status", "scope", "run_id",
        "provider", "automatic_paid_retry", "authorization", "storage",
        "ssh_private_key", "scoped_lifecycle_api_key_file", "worker_image",
        "worker_registry_auth", "guardian", "worker", "job", "inputs", "artifact",
        "result", "one_shot_authorization", "output_receipt", "bound_artifacts",
        "implementation", "invariants", "promotion_boundary",
    }
    if set(spec) != expected_top or spec.get("schema_version") != 1:
        raise PrimaryInterfaceLaunchError("primary interface launch schema changed")
    if (
        spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_H100"
        or spec.get("program") != "parallax"
        or spec.get("scope") != PRIMARY_SCOPE
        or spec.get("provider") != "vast.ai"
        or spec.get("run_id") != RUN_ID
        or spec.get("one_shot_authorization", {}).get("id") != RUN_ID
        or spec.get("automatic_paid_retry") is not False
        or spec.get("promotion_boundary") != PRIMARY_PROMOTION_BOUNDARY
    ):
        raise PrimaryInterfaceLaunchError("primary interface launch escaped its one-shot scope")
    if (
        spec.get("ssh_private_key") != "~/.ssh/parallax_relay_ed25519"
        or spec.get("scoped_lifecycle_api_key_file")
        != "~/.config/parallax/vast_lifecycle_api_key_v2"
        or spec.get("worker_registry_auth")
        != {
            "type": "aws_ecr_ephemeral",
            "profile": "fuelos",
            "region": "us-east-1",
            "registry": "769063704696.dkr.ecr.us-east-1.amazonaws.com",
            "username": "AWS",
            "credential_source": "aws_ecr_get_login_password",
            "persist_secret": False,
            "maximum_token_age_seconds": 43200,
        }
        or spec.get("one_shot_authorization")
        != {
            "id": RUN_ID,
            "claim_path": FRESH_CLAIM_PATH,
        }
        or spec.get("output_receipt")
        != {"path": FRESH_OUTPUT_RECEIPT_PATH}
    ):
        raise PrimaryInterfaceLaunchError("primary interface credential or output binding changed")
    if spec.get("authorization") != {
        "paid_control": True,
        "h100": True,
        "external_processor_gate_replay": True,
        "model_weight_reads_after_external_processor_gate": True,
        "train_data_reads": True,
        "discovery_evaluation": True,
        "training": True,
        "confirmation": False,
        "promotion": False,
        "PG1": True,
        "PG2": False,
        "paired_or_shared_range_claim": False,
        "broad_run": False,
        "science_claim": False,
    }:
        raise PrimaryInterfaceLaunchError("primary interface paid authorization changed")
    if spec.get("storage") != {
        "durable_provider": "amazon_s3",
        "aws_profile": "fuelos",
        "region": legacy.REGION,
        "bucket": legacy.BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": 80,
    }:
        raise PrimaryInterfaceLaunchError("primary interface storage boundary changed")
    if (
        spec.get("worker_image") != legacy.RUNTIME_IMAGE
        or spec.get("guardian") != _exact_guardian()
        or spec.get("worker") != _exact_worker()
        or spec.get("job")
        != {
            "stage": "training_interface_control",
            "maximum_seconds": 1200,
            "command": EXPECTED_COMMAND,
            "environment": {},
            "bounds": _exact_bounds(),
        }
    ):
        raise PrimaryInterfaceLaunchError("primary interface runtime construction changed")
    if spec.get("artifact") != {
        "key": f"parallax/results/{RUN_ID}/artifacts.tar",
        "minimum_bytes": 1_200_000_000,
        "maximum_bytes": 2_100_000_000,
        "post_policy_seconds": 2700,
        "required_http_status": 201,
        "required_version_id": True,
        "required_checksum": "SHA256",
        "required_encryption": "AES256",
    }:
        raise PrimaryInterfaceLaunchError("primary interface artifact boundary changed")
    if spec.get("result") != {
        "key": f"parallax/results/{RUN_ID}/terminal.json",
        "maximum_bytes": legacy.MAXIMUM_RESULT_BYTES,
        "put_url_seconds": 2700,
        "wait_seconds": 1400,
        "required_http_status": 200,
        "required_version_id": True,
        "required_encryption": "AES256",
        "archive_durable_action": "destroy",
        "pre_science_terminal_durable_action": "destroy",
        "science_started_archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
    }:
        raise PrimaryInterfaceLaunchError("primary interface terminal lifecycle changed")
    expected_bound_artifacts = {
        "research_spec", "design_audit", "matrix", "primary_interface_spec",
        "primary_metric_spec", "primary_config", "primary_metric_config",
        "source_contract", "post_processor_freeze", "external_processor_terminal",
        "external_processor_adjudication", "checkpoint_hydration_manifest",
        "overlay_receipt", "base_bundle_receipt", "storage_policy",
        "program_s3_amendment", "runtime_registry", "runtime_h100_linkage",
        "managed_template", "scoped_lifecycle_authority", "machine_quarantine",
        "lifecycle_proof_receipt", "independent_critic",
        "independent_critic_freeze",
    }
    expected_implementation = {
        "primary_launcher", "primary_no_create", "primary_job", "primary_worker",
        "primary_source_gate", "primary_overlay_stage", "transport_launcher",
        "transport_job", "transport_worker", "guardian", "base_job_helper",
        "terminal_transport_worker", "interface_artifact_uploader",
        "interface_artifact_presigner", "terminal_presigner", "s3_transport_launcher",
        "vast_launcher", "artifacts",
    }
    if (
        set(spec.get("inputs", {}))
        != {"model_manifest", "bundle", "bundle_receipt", "overlay_receipt"}
        or spec.get("inputs", {}).get("bundle_receipt", {}).get("path")
        != (
            "studies/lineage/cross-lineage/results/readiness/"
            "parallax_training_optimizer_scale_transfer_v1/receipt.json"
        )
        or set(spec.get("bound_artifacts", {})) != expected_bound_artifacts
        or set(spec.get("implementation", {})) != expected_implementation
    ):
        raise PrimaryInterfaceLaunchError("primary interface binding inventory changed")
    for label, row in spec["bound_artifacts"].items():
        legacy._resolve_bound(root, row, label=f"primary bound artifact {label}")
    for label, row in spec["implementation"].items():
        legacy._resolve_bound(root, row, label=f"primary implementation {label}")
    source = validate_primary_source(root)
    if source.get("external_processor_s3_version_id") != EXTERNAL_PROCESSOR_S3_VERSION_ID:
        raise PrimaryInterfaceLaunchError("primary external processor gate changed")
    _model_rows(root, spec)
    _overlay_rows(root, spec)
    legacy._bundle(root, spec)
    legacy.require_machine_quarantine(root, spec)
    if set(spec.get("invariants", [])) != {
        "primary_SigLIP_plus_Gemma1_to_PaliGemma1_to_pi0_lineage",
        "supporting_PG2_dependency_none_within_lineage",
        "no_single_PG1_to_PG2_causal_chain",
        "external_processor_terminal_precedes_H100_freeze",
        "external_processor_gate_replayed_without_processor_calls_on_H100",
        "weights_hydrate_only_after_exact_external_gate_replay",
        "token_prior_invariant_association_is_registered_primary_metric",
        "three_cyclic_codebooks_counterbalance_letter_prior",
        "exact_registered_seed17_items_schedule_adapter_steps_and_LR",
        "fused_base_reset_between_codebooks",
        "confirmation_reads_zero",
        "nonpromotion",
        "no_automatic_retry",
        "literal_cheapest_current_eligible_H100_all_regions",
        "AWS_S3_is_sole_durable_store",
        "no_Vast_persistent_storage",
        "exact_80_GB_disposable_scratch_no_blanket_allowance",
        "pre_science_terminal_durable_then_destroy",
        "artifact_archive_durable_before_destroy",
        "post_science_archive_nondurable_stops_and_preserves_scratch",
        "controller_HEAD_exact_archive_and_terminal_versions",
        "guardian_acknowledges_exact_worker_ID_before_H100_SSH",
        "known_instance_cleanup_and_account_zero_each_require_three_observations",
    }:
        raise PrimaryInterfaceLaunchError("primary interface invariants changed")
    return spec


def validate_execution_freeze(
    root: Path, spec_path: Path, freeze_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("run_id") != RUN_ID
        or freeze.get("automatic_paid_retry") is not False
        or freeze.get("authorization")
        != {
            "create_one_guardian_and_one_h100": True,
            "create_vast_volumes": False,
            "h100": True,
            "external_processor_gate_replay": True,
            "model_weight_reads_after_external_processor_gate": True,
            "train_data_reads": True,
            "discovery_evaluation": True,
            "training": True,
            "confirmation": False,
            "promotion": False,
            "PG1": True,
            "PG2": False,
            "paired_or_shared_range_claim": False,
            "broad_run": False,
            "automatic_retry": False,
        }
        or freeze.get("launch_spec", {}).get("path")
        != str(spec_path.resolve().relative_to(root.resolve()))
        or freeze.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
        or freeze.get("processor_gate", {}).get("terminal_s3_version_id")
        != EXTERNAL_PROCESSOR_S3_VERSION_ID
        or freeze.get("processor_gate", {}).get("terminal_sha256")
        != EXTERNAL_PROCESSOR_TERMINAL_SHA256
    ):
        raise PrimaryInterfaceLaunchError("primary interface paid execution freeze is absent")
    rehearsal_path = legacy._resolve_bound(
        root, freeze.get("no_create_rehearsal", {}), label="primary no-create rehearsal"
    )
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if (
        rehearsal.get("status") != NO_CREATE_STATUS
        or rehearsal.get("run_id") != RUN_ID
        or rehearsal.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
        or rehearsal.get("mutation")
        != {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "s3_objects_written": 0,
            "files_downloaded": 0,
            "gpu_operations": 0,
        }
        or rehearsal.get("paid_authorization") is not False
    ):
        raise PrimaryInterfaceLaunchError("primary freeze lacks exact zero-create rehearsal")
    plants_path = legacy._resolve_bound(root, freeze.get("planted_tests", {}), label="primary plants")
    plants = json.loads(plants_path.read_text(encoding="utf-8"))
    if (
        plants.get("status") != "GREEN"
        or plants.get("exit_code") != 0
        or plants.get("paid_action") is not False
        or plants.get("confirmation_items_read") != 0
        or plants.get("vast_persistent_volumes_created") != 0
    ):
        raise PrimaryInterfaceLaunchError("primary freeze lacks exact green planted tests")
    return freeze


def _job_contract(
    spec: dict[str, Any],
    *,
    instance_id: int,
    model: list[dict[str, Any]],
    bundle: dict[str, Any],
    bundle_receipt: dict[str, Any],
    overlays: list[dict[str, Any]],
) -> dict[str, Any]:
    identities = {str(row["path"]): str(row["sha256"]) for row in overlays}
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_INTERFACE_CONTROL_JOB",
        "stage": "training_interface_control",
        "provider": "vast.ai",
        "run_id": RUN_ID,
        "lineage": LINEAGE,
        "instance_id": instance_id,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "ephemeral_scratch_gb": 80,
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": {**bundle, "entries": bundle_receipt["entries"]},
        "overlays": overlays,
        "source_contract_sha256": identities[SOURCE_CONTRACT_PATH],
        "external_processor_gate": {
            "path": EXTERNAL_PROCESSOR_TERMINAL_PATH,
            "sha256": EXTERNAL_PROCESSOR_TERMINAL_SHA256,
            "s3_version_id": EXTERNAL_PROCESSOR_S3_VERSION_ID,
            "status": "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT_EXECUTED",
            "h100_instances_created": 0,
            "model_weight_urls": 0,
            "model_loads": 0,
            "gpu_operations": 0,
            "confirmation_items_read": 0,
        },
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
                "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
            ),
            "tokenizer_preflight_output": "/workspace/parallax/tokenizer-preflight.json",
            "control_output_dir": (
                "/workspace/parallax/source/studies/lineage/cross-lineage/results/"
                "parallax_training_interface_control_pg1_primary_v1"
            ),
            "manifest_output": "/workspace/parallax/artifact-manifest.json",
            "archive_output": "/workspace/parallax/artifacts.tar",
            "phase_output": "/workspace/parallax/phase.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": spec["job"]["bounds"],
        "firewall": {
            "processor_gate_before_weight_hydration": True,
            "model_loads_before_processor_gate": 0,
            "weight_objects_before_processor_gate": 0,
            "train_items": 96,
            "discovery_items": 24,
            "confirmation_manifest_present": False,
            "confirmation_images_present": False,
            "confirmation_items_read": 0,
            "promotion_eligible": False,
            "automatic_retry": False,
        },
    }


def _worker_contract(
    spec: dict[str, Any], *, instance_id: int, result_url: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_SELECTED_INTERFACE_WORKER",
        "run_id": RUN_ID,
        "stage": "training_interface_control",
        "instance_id": instance_id,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "phase_path": "/workspace/parallax/phase.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": EXPECTED_COMMAND,
        "maximum_job_seconds": 1200,
        "maximum_result_bytes": legacy.MAXIMUM_RESULT_BYTES,
        "result_key": spec["result"]["key"],
        "result_put_url": result_url,
        "archive_durable_action": "destroy",
        "pre_science_terminal_durable_action": "destroy",
        "science_started_archive_not_durable_action": "stop",
        "terminal_upload_failure_action": "stop",
        "automatic_retry": False,
        "environment": {},
    }


def _worker_start_command() -> str:
    return (
        "set -eu; mkdir -p /workspace/parallax; "
        "nohup python3 /opt/parallax/primary_interface_control_worker.py --contract "
        "/run/parallax/worker-contract.json >/workspace/parallax/worker.log 2>&1 </dev/null & "
        "worker_pid=$!; kill -0 $worker_pid 2>/dev/null || "
        "test -s /workspace/parallax/terminal.json"
    )


def wait_primary_h100_ssh(
    instance_id: int,
    key: Path,
    *,
    bootstrap_seconds: int,
    running_ssh_deadline_seconds: int,
    poll_seconds: float = 2.0,
    health_check=lambda: None,
) -> dict[str, Any]:
    """Retain frozen SSH bounds but abort immediately on fatal provider runtime errors."""

    def provider_health_check() -> None:
        health_check()
        row = optimizer._show_instance(instance_id)
        message = str((row or {}).get("status_msg", ""))
        matched = next(
            (signature for signature in FATAL_PROVIDER_BOOTSTRAP_SIGNATURES if signature in message),
            None,
        )
        if matched is not None:
            raise LifecycleError(
                f"instance {instance_id} reported fatal provider bootstrap signature: {matched}"
            )

    return optimizer.wait_h100_ssh(
        instance_id,
        key,
        bootstrap_seconds=bootstrap_seconds,
        running_ssh_deadline_seconds=running_ssh_deadline_seconds,
        poll_seconds=poll_seconds,
        health_check=provider_health_check,
    )


def require_local_outputs_absent(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(spec.get("output_receipt", {}).get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PrimaryInterfaceLaunchError("primary local output receipt path is unsafe")
    output = root.resolve() / relative
    paths = [output, output.parent / "launch-state.json", output.parent / "terminal.json"]
    present = [path for path in paths if path.exists()]
    if present:
        raise PrimaryInterfaceLaunchError(
            "primary launch refuses to overwrite local attempt evidence: "
            + ", ".join(path.name for path in present)
        )
    return {
        "status": "GREEN_PRIMARY_LOCAL_ATTEMPT_PATHS_ABSENT",
        "output_receipt": str(relative),
        "launch_state": str(relative.parent / "launch-state.json"),
        "terminal": str(relative.parent / "terminal.json"),
    }


@contextmanager
def _patched_legacy(root: Path) -> Iterator[None]:
    _contract, source_digest, identities = _source_contract(root)
    replacements: dict[str, Any] = {
        "RUN_ID": RUN_ID,
        "MODEL_BYTES": MODEL_BYTES,
        "MODEL_NAMES": MODEL_NAMES,
        "OVERLAY_IDENTITIES": identities,
        "SOURCE_CONTRACT_SHA256": source_digest,
        "EXPECTED_COMMAND": EXPECTED_COMMAND,
        "validate_job_contract": validate_job_contract,
        "validate_worker_contract": validate_worker_contract,
        "validate_interface_source": validate_primary_source,
        "_model_rows": _model_rows,
        "_overlay_rows": _overlay_rows,
        "validate_spec": validate_spec,
        "validate_execution_freeze": validate_execution_freeze,
        "_job_contract": _job_contract,
        "_worker_contract": _worker_contract,
        "_worker_start_command": _worker_start_command,
        "wait_h100_ssh": wait_primary_h100_ssh,
    }
    original = {name: getattr(legacy, name) for name in replacements}
    original_stage = legacy._stage

    def stage_with_primary_entrypoints(
        ssh: list[str], *, remote: str, content: bytes, mode: str = "0600"
    ) -> None:
        original_stage(ssh, remote=remote, content=content, mode=mode)
        if remote == "/opt/parallax/ephemeral_interface_control_job.py":
            original_stage(
                ssh,
                remote="/opt/parallax/primary_interface_control_job.py",
                content=(root / "infra/vast/parallax/primary_interface_control_job.py").read_bytes(),
                mode="0700",
            )
        elif remote == "/opt/parallax/ephemeral_interface_control_worker.py":
            original_stage(
                ssh,
                remote="/opt/parallax/primary_interface_control_worker.py",
                content=(root / "infra/vast/parallax/primary_interface_control_worker.py").read_bytes(),
                mode="0700",
            )

    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        legacy._stage = stage_with_primary_entrypoints
        yield
    finally:
        legacy._stage = original_stage
        for name, value in original.items():
            setattr(legacy, name, value)


def run(root: Path, spec_path: Path, freeze_path: Path) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    freeze_path = freeze_path.resolve()
    spec = validate_spec(root, spec_path)
    validate_execution_freeze(root, spec_path, freeze_path, spec)
    require_local_outputs_absent(root, spec)
    with _patched_legacy(root):
        packet = legacy.run(root, spec_path, freeze_path)
    packet.update(
        {
            "status": (
                "GREEN_PRIMARY_PG1_INTERFACE_CONTROL_EXECUTED"
                if packet.get("status") == "GREEN_INTERFACE_CONTROL_EXECUTED"
                else "RED_PRIMARY_PG1_INTERFACE_PREFLIGHT"
                if packet.get("status") == "RED_INTERFACE_CONTROL_PREFLIGHT"
                else "RED_PRIMARY_PG1_INTERFACE_EXECUTION"
            ),
            "scope": "PRIMARY_PG1_INTERFACE_DISCOVERY_PROVISIONAL_NONPROMOTION",
            "lineage": LINEAGE,
            "role": "primary_within_lineage_discovery_independent_of_supporting_pg2",
            "supporting_pg2_dependency": "none_within_lineage",
            "paired_or_shared_range_claim": False,
            "promotion_boundary": PRIMARY_PROMOTION_BOUNDARY,
        }
    )
    output = root / Path(str(spec["output_receipt"]["path"]))
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root, args.spec, args.freeze)
    print(json.dumps({"run_id": RUN_ID, "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"] in {
        "GREEN_PRIMARY_PG1_INTERFACE_CONTROL_EXECUTED",
        "RED_PRIMARY_PG1_INTERFACE_PREFLIGHT",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
