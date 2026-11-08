"""Validate the effective PARALLAX execution policy without rewriting frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


AMENDMENT_PATH = Path(
    "studies/lineage/cross-lineage/config/parallax_execution_safety_amendment_v2.yaml"
)
LAUNCH_OVERLAY_PATH = Path(
    "studies/lineage/cross-lineage/config/parallax_launch_packet.s3_v2.template.yaml"
)
EXPECTED_RUNTIME = (
    "769063704696.dkr.ecr.us-east-1.amazonaws.com/parallax/runtime-minimal@"
    "sha256:f7407a831951f4c744a9afecf59d248bfadd31d44d012f200bc5055b6d4d4ed6"
)
EXPECTED_TEMPLATE_HASH = "a49dfc59f765f8c5b9c1e0c3a318a486"
RETIRED_LIFECYCLE_FIELDS = {
    "continuous_mirror",
    "gateway_bootstrap_cleanup",
    "lifecycle_key_name",
    "volume_endpoint_access",
}


class ProgramControlError(RuntimeError):
    """Raised when historical or effective program control drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProgramControlError(f"control document is not a mapping: {path}")
    return value


def _bound_path(root: Path, row: dict[str, Any], *, label: str) -> Path:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ProgramControlError(f"unsafe {label} path")
    target = (root / relative).resolve()
    if not target.is_file() or sha256_file(target) != row.get("sha256"):
        raise ProgramControlError(f"{label} hash mismatch")
    return target


def validate_policy_data(
    root: Path,
    amendment: dict[str, Any],
    launch_overlay: dict[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    if (
        amendment.get("schema_version") != 2
        or amendment.get("status") != "FROZEN_EXECUTION_SAFETY_AMENDMENT_AFTER_V104"
        or amendment.get("scope") != "operational_and_cost_safety_only"
    ):
        raise ProgramControlError("execution amendment identity changed")

    frozen = amendment.get("frozen_historical_inputs")
    if not isinstance(frozen, dict) or set(frozen) != {
        "spec",
        "design_audit",
        "matrix",
        "launch_packet_template",
        "predecessor_freeze",
    }:
        raise ProgramControlError("historical input inventory changed")
    for label, row in frozen.items():
        _bound_path(root, row, label=f"frozen historical {label}")

    precedence = amendment.get("precedence", {})
    if (
        precedence.get("scientific_design_changed") is not False
        or precedence.get("outcome_thresholds_changed") is not False
        or precedence.get("data_or_confirmation_firewalls_changed") is not False
        or precedence.get("prior_run_identities_reusable") is not False
        or precedence.get("prior_results_reinterpreted") is not False
        or precedence.get("future_launchers_must_bind_historical_inputs_and_this_amendment")
        is not True
        or precedence.get("frozen_historical_files_must_remain_byte_identical") is not True
    ):
        raise ProgramControlError("execution amendment escaped its non-scientific scope")

    execution = amendment.get("effective_execution", {})
    durability = amendment.get("durability", {})
    lifecycle = amendment.get("lifecycle", {})
    laptop = amendment.get("laptop_boundary", {})
    authorization = amendment.get("authorization", {})
    if (
        execution.get("provider") != "vast.ai"
        or execution.get("gpu_count") != 1
        or execution.get("concurrent_gpu_jobs") != 1
        or execution.get("offer_selection")
        != "literal_cheapest_eligible_all_regions_at_mutation_time"
        or execution.get("fixed_local_scratch_minimum") != "forbidden"
        or execution.get("scratch_projection_rule") != "measured_peak_plus_30pct"
        or execution.get("current_rehearsed_range_cell_scratch_gb") != 80
        or execution.get("current_s3_planning_allocation_tb") != 1
        or execution.get("automatic_paid_retry") != "forbidden"
    ):
        raise ProgramControlError("effective execution policy changed")
    if (
        amendment.get("runtime", {}).get("image") != EXPECTED_RUNTIME
        or amendment.get("runtime", {}).get("managed_template_hash") != EXPECTED_TEMPLATE_HASH
        or amendment.get("runtime", {}).get("dependency_install_on_H100") != "forbidden"
    ):
        raise ProgramControlError("runtime identity changed")
    if (
        durability.get("sole_system_of_record") != "amazon_s3"
        or durability.get("aws_profile") != "fuelos"
        or durability.get("region") != "us-east-1"
        or durability.get("vast_persistent_volumes") != "forbidden"
        or durability.get("vast_storage_gateways") != "forbidden"
        or durability.get("worker_aws_credentials") != "forbidden"
        or durability.get("recurring_vast_storage_after_success") != 0
    ):
        raise ProgramControlError("durability policy changed")
    if (
        lifecycle.get("guardian_created_before_H100") != "required"
        or lifecycle.get("guardian_exact_worker_id_ack_before_release") != "required"
        or lifecycle.get("known_instance_absence_consecutive_observations") != 3
        or lifecycle.get("whole_account_zero_after_destroy") != "required"
    ):
        raise ProgramControlError("lifecycle policy changed")
    if (
        laptop.get("role") != "lightweight_control_plane_only"
        or any(laptop.get(field) != 0 for field in ("model_bytes", "dataset_bytes", "container_bytes", "research_archive_bytes"))
    ):
        raise ProgramControlError("laptop boundary changed")
    if any(authorization.get(field) is not False for field in (
        "paid_action",
        "create_vast_instance",
        "create_vast_volume",
        "H100",
        "model_or_data_read",
        "confirmation",
        "promotion",
        "automatic_retry",
    )):
        raise ProgramControlError("amendment accidentally authorizes paid or scientific work")

    if (
        launch_overlay.get("schema_version") != 2
        or launch_overlay.get("status") != "FROZEN_S3_EPHEMERAL_LAUNCH_TEMPLATE_OVERLAY"
    ):
        raise ProgramControlError("S3 launch overlay identity changed")
    composition = launch_overlay.get("composition", {})
    _bound_path(root, composition.get("base_template", {}), label="launch overlay base")
    _bound_path(root, composition.get("execution_amendment", {}), label="launch overlay amendment")
    if (
        composition.get("strategy")
        != "inherit_all_base_sections_replace_lifecycle_add_resource_projection_fields"
        or composition.get("old_lifecycle_fields_authoritative") is not False
    ):
        raise ProgramControlError("launch overlay composition changed")

    overlay_lifecycle = launch_overlay.get("lifecycle", {})
    if RETIRED_LIFECYCLE_FIELDS.intersection(overlay_lifecycle):
        raise ProgramControlError("retired lifecycle fields returned")
    if (
        overlay_lifecycle.get("launcher") != "REQUIRED_HASH_BOUND_EPHEMERAL_S3_LAUNCHER"
        or overlay_lifecycle.get("runtime_image") != EXPECTED_RUNTIME
        or overlay_lifecycle.get("offer_selection")
        != "literal_cheapest_eligible_all_regions_at_mutation_time"
        or overlay_lifecycle.get("worker", {}).get("persistent_volume") != "forbidden"
        or overlay_lifecycle.get("worker", {}).get("storage_gateway") != "forbidden"
        or overlay_lifecycle.get("worker", {}).get("aws_credentials") != "forbidden"
        or overlay_lifecycle.get("aws_s3", {}).get("role") != "sole_durable_system_of_record"
        or overlay_lifecycle.get("automatic_paid_retry") != "forbidden"
        or overlay_lifecycle.get("teardown", {}).get("whole_account_zero_required") is not True
        or launch_overlay.get("go") is not False
    ):
        raise ProgramControlError("S3 launch overlay escaped its boundary")

    return {
        "schema_version": 1,
        "status": "GREEN_EFFECTIVE_PROGRAM_CONTROL",
        "historical_inputs_verified": len(frozen),
        "scientific_design_changed": False,
        "paid_action": False,
        "provider": "vast.ai",
        "offer_selection": execution["offer_selection"],
        "scratch": {
            "fixed_minimum": "forbidden",
            "current_rehearsed_range_cell_gb": 80,
        },
        "durability": {
            "sole_system_of_record": "amazon_s3",
            "aws_profile": "fuelos",
            "region": "us-east-1",
            "vast_persistent_storage": "forbidden",
        },
        "runtime_image": EXPECTED_RUNTIME,
        "laptop_role": "lightweight_control_plane_only",
        "automatic_paid_retry": False,
    }


def validate_local(root: Path) -> dict[str, Any]:
    root = root.resolve()
    amendment_path = root / AMENDMENT_PATH
    overlay_path = root / LAUNCH_OVERLAY_PATH
    return validate_policy_data(root, _load_yaml(amendment_path), _load_yaml(overlay_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_local(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
