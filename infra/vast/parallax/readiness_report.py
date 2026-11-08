"""Recompute the local PARALLAX H100 GO/NO_GO readiness state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactManifest, atomic_json, sha256_file
from .image_smoke import expected_versions
from .launch_packet import LaunchPacketError, validate_launch_packet


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def build_report(root: Path) -> dict[str, Any]:
    cross = root / "studies/lineage/cross-lineage"
    spec = cross / "research/2026-07-15_parallax_spec.md"
    matrix_path = cross / "config/parallax_matrix.yaml"
    packet_template = cross / "config/parallax_launch_packet.template.yaml"
    dockerfile = root / "infra/vast/parallax/Dockerfile"
    lock = root / "infra/vast/parallax/requirements.lock"
    market_path = cross / "results/readiness/vast_market_2026-07-15.json"
    genealogy_path = cross / "results/readiness/genealogy_config_2026-07-15.json"
    storage_path = cross / "results/readiness/storage_rehearsal_2026-07-15.json"
    image_gpu_path = cross / "results/readiness/image_h100_smoke_2026-07-15.json"
    planted_path = cross / "results/readiness/planted_preflight_2026-07-15.json"
    download_path = cross / "results/readiness/download_inventory_2026-07-15.json"
    durable_active_path = cross / "results/readiness/durable_volume_active_2026-07-15.json"
    runtime_bundle_path = cross / "results/readiness/runtime_bundle_2026-07-15.json"
    pi_runtime_bundle_path = cross / "results/readiness/runtime_pi_bundle_2026-07-15.json"
    model_cache_path = cross / "results/readiness/model_cache_2026-07-15.json"
    transfer_rehearsal_path = cross / "results/readiness/intermachine_transfer_rehearsal_2026-07-15.json"
    h100_execution_path = cross / "results/readiness/h100_preflight_execution_2026-07-15.json"
    g1_rehearsal_path = cross / "results/readiness/g1_runtime_rehearsal_2026-07-16.json"
    h100_packet_path = cross / "results/readiness/h100_preflight_45049292"
    freeze_path = cross / "config/parallax_design_freeze_v1.json"
    execution_freeze_path = cross / "config/parallax_execution_freeze_v2.json"
    pi_freeze_path = cross / "config/parallax_pi_sidecar_freeze_v1.json"
    g1_freeze_path = cross / "config/parallax_g1_freeze_v3.json"
    required_preflight = [
        root / "infra/vast/parallax/parallax_entities.py",
        root / "infra/vast/parallax/parallax_tasks.py",
        root / "infra/vast/parallax/parallax_operator.py",
        root / "infra/vast/parallax/parallax_towers.py",
        root / "infra/vast/parallax/parallax_adjudicate.py",
        root / "infra/vast/parallax/preflight.py",
        root / "infra/vast/parallax/g1_common.py",
        root / "infra/vast/parallax/g1_core.py",
        root / "infra/vast/parallax/g1_pi.py",
        root / "infra/vast/parallax/g1_adjudicate.py",
        root / "infra/vast/parallax/g1_sequence.py",
        root / "infra/vast/parallax_run.py",
    ]

    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    matrix_text = matrix_path.read_text(encoding="utf-8")
    market = _json_or_none(market_path)
    genealogy = _json_or_none(genealogy_path)
    storage = _json_or_none(storage_path)
    image_gpu = _json_or_none(image_gpu_path)
    planted = _json_or_none(planted_path)
    download = _json_or_none(download_path)
    durable_active = _json_or_none(durable_active_path)
    runtime_bundle = _json_or_none(runtime_bundle_path)
    pi_runtime_bundle = _json_or_none(pi_runtime_bundle_path)
    model_cache = _json_or_none(model_cache_path)
    transfer_rehearsal = _json_or_none(transfer_rehearsal_path)
    h100_execution = _json_or_none(h100_execution_path)
    g1_rehearsal = _json_or_none(g1_rehearsal_path)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, green: bool, evidence: Any, detail: str) -> None:
        checks[name] = {
            "status": "GREEN" if green else "RED",
            "evidence": evidence,
            "detail": detail,
        }

    branch = _git(root, "branch", "--show-current")
    worktree_list = _git(root, "worktree", "list", "--porcelain")
    check(
        "isolated_worktree",
        branch == "science/parallax" and str(root.resolve()) in worktree_list,
        {"branch": branch, "root": str(root.resolve())},
        "PARALLAX must remain isolated from all active lineage worktrees.",
    )
    check(
        "design_files_present",
        all(path.is_file() for path in (spec, matrix_path, packet_template)),
        {
            "spec_sha256": sha256_file(spec),
            "matrix_sha256": sha256_file(matrix_path),
            "launch_template_sha256": sha256_file(packet_template),
        },
        "All design and refusal artifacts exist and are content-addressed.",
    )
    check(
        "checkpoint_revisions_pinned",
        "RESOLVE_AT" not in matrix_text,
        str(genealogy_path.relative_to(root)),
        "Every referenced checkpoint must be a full commit SHA before weights download.",
    )
    genealogy_green = bool(
        genealogy and genealogy.get("verdict") == "GREEN_CONFIG_ONLY_WEIGHTS_UNPROVEN"
    )
    check(
        "weight_free_genealogy",
        genealogy_green,
        str(genealogy_path.relative_to(root)),
        "Both SigLIP+Gemma fusion shells and the declared pi backbone pass config-only checks; ancestry remains scoped as unproven.",
    )

    try:
        validate_launch_packet(packet_template, repo_root=root, verify_evidence=False)
        template_refuses = False
    except LaunchPacketError:
        template_refuses = True
    check(
        "launch_template_refuses",
        template_refuses,
        str(packet_template.relative_to(root)),
        "The distributed template must never authorize compute.",
    )

    docker_text = dockerfile.read_text(encoding="utf-8")
    digest_match = re.search(r"FROM\s+\S+@sha256:[0-9a-f]{64}", docker_text)
    versions = expected_versions(lock)
    check(
        "immutable_image_recipe",
        digest_match is not None and all(versions.values()),
        {
            "dockerfile_sha256": sha256_file(dockerfile),
            "requirements_sha256": sha256_file(lock),
            "top_level_pin_count": len(versions),
        },
        "The PARALLAX-only base is digest-pinned and all top-level dependencies are exact.",
    )
    check(
        "h100_image_smoke",
        bool(image_gpu and image_gpu.get("status") == "GREEN"),
        str(image_gpu_path.relative_to(root)),
        "Requires the built image on the actual selected H100; CPU resolution is not enough.",
    )
    h100_packet_errors: list[str] = []
    if not h100_packet_path.is_dir():
        h100_packet_errors.append("returned H100 packet missing")
    else:
        h100_packet_errors.extend(
            ArtifactManifest.verify(h100_packet_path / "manifest.json", h100_packet_path)
        )
    packet_preflight = _json_or_none(h100_packet_path / "h100-preflight.json")
    packet_hydration = _json_or_none(h100_packet_path / "cache-hydration.json")
    packet_gate = _json_or_none(h100_packet_path / "cache-gate.json")
    packet_transfer = _json_or_none(h100_packet_path / "inbound-transfer.json")
    h100_instance_id = (h100_execution or {}).get("h100", {}).get("instance_id")
    h100_cache_contract = (packet_hydration or {}).get("content_manifest_sha256")
    h100_preflight_green = bool(
        h100_execution
        and h100_execution.get("status") == "GREEN"
        and h100_execution.get("science_started") is False
        and h100_execution.get("cleanup", {}).get("status") == "GREEN"
        and h100_execution.get("cleanup", {}).get("active_instance_count_after_cleanup") == 0
        and packet_preflight
        and packet_preflight.get("status") == "GREEN"
        and float(packet_preflight.get("elapsed_seconds", float("inf"))) <= 30 * 60
        and packet_preflight.get("local_cache_sha256_verification") == "GREEN"
        and packet_hydration
        and packet_hydration.get("status") == "GREEN"
        and packet_hydration.get("verified_bytes") == (model_cache or {}).get("verified_bytes")
        and h100_cache_contract == (model_cache or {}).get("content_manifest_sha256")
        and packet_hydration.get("destination_instance_id") == h100_instance_id
        and packet_gate
        and packet_gate.get("status") == "GREEN"
        and packet_gate.get("destination_instance_id") == h100_instance_id
        and packet_gate.get("content_manifest_sha256") == h100_cache_contract
        and packet_transfer
        and packet_transfer.get("status") == "GREEN"
        and packet_transfer.get("destination_instance_id") == h100_instance_id
        and not h100_packet_errors
    )
    check(
        "h100_no_science_preflight",
        h100_preflight_green,
        {
            "execution": str(h100_execution_path.relative_to(root)),
            "returned_packet": str(h100_packet_path.relative_to(root)),
            "packet_errors": h100_packet_errors,
        },
        "The selected H100 must pass exact runtime, cache, transfer, gate, durable-return, credential-cleanup, and resource-cleanup checks before G1.",
    )

    market_green = bool(
        market
        and market.get("selected_h100")
        and market.get("selected_volume_bundle")
        and market.get("mutation") == "none_read_only"
    )
    check(
        "live_vast_market",
        market_green,
        str(market_path.relative_to(root)),
        "A current all-region H100 and an accessible persistent-volume bundle are visible; both must be re-queried at creation.",
    )
    check(
        "durable_store_rehearsal",
        bool(
            storage
            and storage.get("status") == "GREEN"
            and storage.get("cleanup_status") == "GREEN"
            and storage.get("attach_cycles") == 2
        ),
        str(storage_path.relative_to(root)),
        "A disposable volume must survive destroy/re-attach with exact bytes and leave no paid resources.",
    )

    tasks_dir = cross / "config/tasks"
    data_dir = cross / "config/data"
    materialized_content = _json_or_none(cross / "data/parallax_tasks_v1/manifests/content.json")
    task_bank_audit = _json_or_none(cross / "data/parallax_tasks_v1/manifests/audit.json")
    check(
        "task_and_data_manifests",
        tasks_dir.is_dir()
        and data_dir.is_dir()
        and any(tasks_dir.glob("*.yaml"))
        and any(data_dir.glob("*.yaml"))
        and bool(materialized_content and materialized_content.get("status") == "GREEN_MATERIALIZED")
        and bool(task_bank_audit and task_bank_audit.get("status") == "GREEN"),
        {
            "tasks": str(tasks_dir.relative_to(root)),
            "data": str(data_dir.relative_to(root)),
            "content_manifest": "studies/lineage/cross-lineage/data/parallax_tasks_v1/manifests/content.json",
            "confound_audit": "studies/lineage/cross-lineage/data/parallax_tasks_v1/manifests/audit.json",
        },
        "Frozen task/data manifests must be materialized and pass split, balance, target, and contamination audits.",
    )
    check(
        "design_frozen",
        matrix.get("status") == "frozen",
        {"matrix_status": matrix.get("status"), "required_tag": "parallax-design-v1"},
        "Freeze only after all offline confound and mechanics gates are green.",
    )
    freeze = _json_or_none(freeze_path)
    freeze_errors: list[str] = []
    if freeze is None:
        freeze_errors.append("freeze manifest missing")
    else:
        for name, row in freeze.get("artifacts", {}).items():
            path = root / row["path"]
            if not path.is_file():
                freeze_errors.append(f"{name}: missing {row['path']}")
            elif sha256_file(path) != row["sha256"]:
                freeze_errors.append(f"{name}: SHA mismatch")
    check(
        "design_freeze_hashes",
        not freeze_errors,
        {"manifest": str(freeze_path.relative_to(root)), "errors": freeze_errors},
        "Every frozen scientific and execution artifact must still match its registered bytes.",
    )
    execution_freeze = _json_or_none(execution_freeze_path)
    execution_freeze_errors: list[str] = []
    if execution_freeze is None or execution_freeze.get("status") != "FROZEN":
        execution_freeze_errors.append("execution freeze manifest missing or not FROZEN")
    else:
        for name, row in execution_freeze.get("artifacts", {}).items():
            path = root / row["path"]
            if not path.is_file():
                execution_freeze_errors.append(f"{name}: missing {row['path']}")
            elif sha256_file(path) != row["sha256"]:
                execution_freeze_errors.append(f"{name}: SHA mismatch")
    check(
        "execution_order_frozen",
        not execution_freeze_errors,
        {"manifest": str(execution_freeze_path.relative_to(root)), "errors": execution_freeze_errors},
        "The high-leverage truth-sprint order must be frozen before any H100 science.",
    )
    pi_freeze = _json_or_none(pi_freeze_path)
    pi_freeze_errors: list[str] = []
    if pi_freeze is None or pi_freeze.get("status") != "FROZEN_BEFORE_G1":
        pi_freeze_errors.append("pi sidecar freeze missing or not frozen")
    else:
        for name, row in pi_freeze.get("artifacts", {}).items():
            path = root / row["path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                pi_freeze_errors.append(f"{name}: missing or SHA mismatch")
    check(
        "pi_sidecar_frozen",
        not pi_freeze_errors,
        {"manifest": str(pi_freeze_path.relative_to(root)), "errors": pi_freeze_errors},
        "The incompatible native pi wrapper dependency must remain isolated and content-frozen before G1.",
    )
    g1_freeze = _json_or_none(g1_freeze_path)
    g1_freeze_errors: list[str] = []
    if g1_freeze is None or g1_freeze.get("status") != "FROZEN_BEFORE_G1":
        g1_freeze_errors.append("G1 freeze missing or not frozen")
    else:
        row = g1_freeze.get("anchor_contract", {})
        path = root / str(row.get("path", ""))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            g1_freeze_errors.append("G1 anchor contract missing or SHA mismatch")
        for module in g1_freeze.get("entry_points", {}).values():
            module_path = root / (str(module).replace(".", "/") + ".py")
            if not module_path.is_file():
                g1_freeze_errors.append(f"G1 entry point missing: {module}")
        for name, artifact in g1_freeze.get("runner_artifacts", {}).items():
            artifact_path = root / str(artifact.get("path", ""))
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
                g1_freeze_errors.append(f"G1 runner artifact missing or SHA mismatch: {name}")
    check(
        "g1_runner_frozen",
        not g1_freeze_errors,
        {"manifest": str(g1_freeze_path.relative_to(root)), "errors": g1_freeze_errors},
        "Both firewalled G1 runtimes and their paired adjudicator must be frozen before H100 science.",
    )
    check(
        "g1_runtime_rehearsed",
        bool(
            g1_rehearsal
            and g1_rehearsal.get("status") == "GREEN_READY_H100_G1"
            and g1_rehearsal.get("provider") == "vast.ai"
            and g1_rehearsal.get("science_started") is False
            and g1_rehearsal.get("model_forward_executed") is False
            and g1_rehearsal.get("gateway", {}).get("cleanup_status") == "GREEN"
            and g1_rehearsal.get("gateway", {}).get("active_instance_count_after_cleanup") == 0
            and g1_rehearsal.get("gateway", {}).get("ephemeral_ssh_key_present_after_cleanup") is False
            and g1_rehearsal.get("processor_rehearsal", {}).get(
                "pixel_exact_across_siglip_pg1_pg2pt_pg2mix"
            )
            and g1_rehearsal.get("pi_runtime", {}).get("pi0", {}).get("missing_keys") == 0
            and g1_rehearsal.get("pi_runtime", {}).get("pi05", {}).get("missing_keys") == 0
            and g1_rehearsal.get("source", {}).get("anchor_sha256")
            == (g1_freeze or {}).get("anchor_contract", {}).get("sha256")
        ),
        str(g1_rehearsal_path.relative_to(root)),
        "Cheap Vast rehearsal must close runtime, preprocessing, structure, and strict pi-load confounds without a model forward.",
    )
    check(
        "executable_preflight_entrypoints",
        all(path.is_file() and path.stat().st_size > 0 for path in required_preflight),
        [str(path.relative_to(root)) for path in required_preflight],
        "A no-science runner and planted-testable first-six-hour primitives must exist before any H100 rental.",
    )
    source_hash_errors: list[str] = []
    if not planted or planted.get("status") != "GREEN":
        source_hash_errors.append("planted evidence missing or red")
    else:
        for name, expected in planted.get("source_sha256", {}).items():
            path = root / "infra/vast/parallax" / name
            if not path.is_file() or sha256_file(path) != expected:
                source_hash_errors.append(f"stale planted evidence: {name}")
    check(
        "planted_preflight_gates",
        not source_hash_errors,
        {"report": str(planted_path.relative_to(root)), "errors": source_hash_errors},
        "All planted branches and mathematical identities must pass against the current source bytes.",
    )
    check(
        "download_and_disk_projection",
        bool(
            download
            and download.get("status") == "GREEN"
            and float(download.get("projection", {}).get("projected_disk_fraction", 1.0)) < 0.80
        ),
        str(download_path.relative_to(root)),
        "Pinned file metadata and a conservative first-six-hour disk projection must fit below 80% of requested scratch.",
    )
    check(
        "active_durable_volume",
        bool(
            durable_active
            and durable_active.get("status") == "GREEN"
            and float(durable_active.get("capacity_gb", 0)) >= 4000
            and durable_active.get("provider") == "vast.ai"
        ),
        str(durable_active_path.relative_to(root)),
        "The dedicated >=4 TB Vast volume must be active before an H100 is rented.",
    )
    check(
        "runtime_bundle_on_durable_store",
        bool(
            runtime_bundle
            and runtime_bundle.get("status") == "GREEN"
            and runtime_bundle.get("sha256")
            and int(runtime_bundle.get("bundle_bytes", 0)) > 0
        ),
        str(runtime_bundle_path.relative_to(root)),
        "The exact runtime must already be content-addressed on durable storage; installation on H100 is forbidden.",
    )
    check(
        "pi_sidecar_runtime_on_durable_store",
        bool(
            pi_runtime_bundle
            and pi_runtime_bundle.get("status") == "GREEN"
            and pi_runtime_bundle.get("provider") == "vast.ai"
            and pi_runtime_bundle.get("location") == "persistent_volume"
            and pi_runtime_bundle.get("runtime_role") == "pi_sidecar"
            and pi_runtime_bundle.get("requirements_sha256")
            == sha256_file(root / "infra/vast/parallax/requirements_pi.lock")
            and pi_runtime_bundle.get("pi_sidecar_smoke_status") == "GREEN"
        ),
        str(pi_runtime_bundle_path.relative_to(root)),
        "Native pi0/pi0.5 G1 requires its separately pinned, gateway-built sidecar; it may not be assembled on H100.",
    )
    check(
        "model_cache_on_durable_store",
        bool(
            model_cache
            and model_cache.get("status") == "GREEN"
            and model_cache.get("provider") == "vast.ai"
            and model_cache.get("location") == "persistent_volume"
            and int(model_cache.get("verified_bytes", 0))
            == int((download or {}).get("required_cache_bytes", -1))
            and model_cache.get("content_manifest_sha256")
        ),
        str(model_cache_path.relative_to(root)),
        "Every exact pinned model byte must be downloaded and verified on the durable Vast volume before H100 time starts.",
    )
    check(
        "intermachine_durable_transfer",
        bool(
            transfer_rehearsal
            and transfer_rehearsal.get("status") == "GREEN"
            and transfer_rehearsal.get("provider") == "vast.ai"
            and transfer_rehearsal.get("forward_sha256") == transfer_rehearsal.get("source_sha256")
            and transfer_rehearsal.get("reverse_sha256") == transfer_rehearsal.get("source_sha256")
            and transfer_rehearsal.get("cleanup_status") == "GREEN"
        ),
        str(transfer_rehearsal_path.relative_to(root)),
        "A cross-machine Vast-host transfer must preserve representative bytes in both directions before the all-region H100 can use remote durable storage.",
    )

    red = [name for name, row in checks.items() if row["status"] != "GREEN"]
    if not red:
        decision = "GO_TRUTH_SPRINT_FIRST_6H"
    elif red == ["h100_image_smoke"]:
        decision = "GO_H100_PREFLIGHT_ONLY"
    else:
        decision = "NO_GO_H100"
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "program": "parallax",
        "decision": decision,
        "red_checks": red,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root.resolve())
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "red_checks": report["red_checks"]}))
    return 0 if report["decision"].startswith("GO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
