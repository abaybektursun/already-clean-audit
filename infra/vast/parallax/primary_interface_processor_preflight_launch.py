"""Launch one disposable CPU-only PG1 processor gate before any H100 exists."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .ephemeral_optimizer_scale_launch import PUBLIC_OFFER_FIELDS, RUNTIME_IMAGE, RUNTIME_TEMPLATE
from .ephemeral_pilot_launch import (
    _bundle,
    _require_input_heads,
    _require_result_absent,
    _resolve_bound,
    _signed,
    _validate_input,
)
from .ephemeral_s3_launch import (
    BUCKET,
    REGION,
    _aws_credentials,
    _claim,
    _confirm_account_zero,
    _create_arguments,
    _finalize_lifecycle,
    _require_initial_zero,
    _stage,
    _wait_result,
    _worker_start_command,
    require_s3_safety,
)
from .ephemeral_s3_worker import validate_contract as validate_worker_contract
from .guarded_launch import _create_instance, _rows, _run, _ssh_base, _vast, wait_ssh
from .primary_interface_processor_preflight_job import (
    CONFIG_PATH,
    LINEAGE,
    OVERLAY_IDENTITIES,
    PROCESSOR_BYTES,
    PROCESSOR_NAMES,
    RUN_ID,
    validate_contract as validate_job_contract,
)
from .s3_presign_put import presign_result_put


MODEL_MANIFEST_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_primary_processor_s3_manifest_v1.json"
)
EXPECTED_COMMAND = [
    "python3",
    "/opt/parallax/primary_interface_processor_preflight_job.py",
    "--contract",
    "/run/parallax/processor-preflight-contract.json",
]


class PrimaryProcessorLaunchError(RuntimeError):
    pass


def _processor_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_bound(root, spec["inputs"]["processor_manifest"], label="PG1 processor manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if (
        manifest.get("status") != "FROZEN_PROCESSOR_ONLY_INPUT_MANIFEST"
        or manifest.get("scope") != "primary_pg1_processor_preflight_before_any_h100_create"
        or manifest.get("lineage") != LINEAGE
        or manifest.get("repo") != "google/paligemma-3b-pt-224"
        or manifest.get("revision") != "35e4f46485b4d07967e7e9935bc3786aad50687c"
        or manifest.get("expected_file_count") != 8
        or manifest.get("expected_total_bytes") != PROCESSOR_BYTES
        or manifest.get("model_weight_objects") != 0
        or manifest.get("model_weight_bytes") != 0
        or manifest.get("model_loads") != 0
        or manifest.get("gpu_operations") != 0
        or manifest.get("confirmation_items_read") != 0
        or manifest.get("authorization", {}).get("paid_action") is not False
        or manifest.get("storage", {}).get("durable_provider") != "amazon_s3"
        or manifest.get("storage", {}).get("vast_persistent_volumes") != "FORBIDDEN"
        or not isinstance(rows, list)
        or len(rows) != 8
    ):
        raise PrimaryProcessorLaunchError("PG1 processor-only manifest is not exact")
    for index, row in enumerate(rows):
        _validate_input(row, label=f"PG1 processor {index}")
    if (
        {str(row.get("path", "")) for row in rows} != PROCESSOR_NAMES
        or sum(int(row["bytes"]) for row in rows) != PROCESSOR_BYTES
        or any("safetensors" in str(row.get("path", "")) for row in rows)
    ):
        raise PrimaryProcessorLaunchError("PG1 processor-only inventory changed")
    source = _resolve_bound(root, manifest["source_binding"], label="full PG1 S3 manifest")
    full = json.loads(source.read_text(encoding="utf-8"))
    if full.get("bucket") != BUCKET or full.get("region") != REGION:
        raise PrimaryProcessorLaunchError("full PG1 S3 mirror destination changed")
    full_rows = {str(row["path"]): row for row in full.get("files", [])}
    for row in rows:
        original = full_rows.get(str(row["path"]))
        if original is None or any(
            original.get(key) != row.get(key)
            for key in ("bytes", "sha256", "key", "version_id")
        ):
            raise PrimaryProcessorLaunchError("processor subset disagrees with full PG1 S3 mirror")
    return rows


def _overlay_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = spec.get("inputs", {}).get("overlays")
    if not isinstance(rows, list) or len(rows) != len(OVERLAY_IDENTITIES):
        raise PrimaryProcessorLaunchError("processor preflight overlay bindings are absent")
    observed: dict[str, str] = {}
    result = []
    for index, row in enumerate(rows):
        path = _resolve_bound(root, row, label=f"processor preflight overlay {index}")
        relative = str(row["path"])
        observed[relative] = str(row["sha256"])
        result.append(
            {
                "path": relative,
                "sha256": str(row["sha256"]),
                "bytes": path.stat().st_size,
                "staged_path": f"/run/parallax/preflight-overlays/{index:02d}.blob",
            }
        )
    if observed != OVERLAY_IDENTITIES or len(observed) != len(rows):
        raise PrimaryProcessorLaunchError("processor preflight overlay identities changed")
    return result


def select_cheapest_cpu_preflight_offer(contract: dict[str, Any]) -> dict[str, Any]:
    disk = int(contract["disk_gb"])
    query = (
        f"cpu_arch=amd64 num_gpus=1 reliability>={float(contract['minimum_reliability']):g} "
        f"disk_space>={disk} duration>=1 rentable=true rented=false static_ip=true "
        f"direct_port_count>=2 inet_down>={float(contract['minimum_inet_down_mbps']):g} "
        f"cuda_max_good>={float(contract['minimum_cuda']):g}"
    )
    rows = _rows(
        _vast(
            "search", "offers", query,
            "--limit", "100", "--storage", str(disk), "--order", "dph_total",
        )
    )
    eligible = [
        row for row in rows
        if "H100" not in str(row.get("gpu_name", "")).upper()
        and int(row.get("cpu_ram", 0)) >= int(contract["minimum_cpu_ram_mb"])
        and float(row.get("cpu_cores_effective", 0)) >= float(contract["minimum_cpu_cores_effective"])
    ]
    eligible.sort(
        key=lambda row: (
            float(row.get("dph_total", float("inf"))),
            -float(row.get("reliability", 0)),
        )
    )
    if not eligible:
        raise PrimaryProcessorLaunchError("no eligible non-H100 CPU preflight offer exists")
    selected = eligible[0]
    if float(selected.get("dph_total", float("inf"))) > float(contract["price_ceiling_dph"]):
        raise PrimaryProcessorLaunchError("literal-cheapest CPU preflight exceeds price ceiling")
    return {
        "query": query,
        "candidate_count": len(eligible),
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "selected": {name: selected[name] for name in PUBLIC_OFFER_FIELDS if name in selected},
    }


def validate_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if set(spec) != {
        "schema_version", "created_utc", "program", "status", "scope", "run_id",
        "provider", "automatic_paid_retry", "authorization", "storage",
        "ssh_private_key", "worker_image", "worker_registry_auth", "worker", "job",
        "inputs", "result", "one_shot_authorization", "output_receipt",
        "bound_artifacts", "implementation", "invariants", "promotion_boundary",
    } or spec.get("schema_version") != 1:
        raise PrimaryProcessorLaunchError("processor preflight launch schema changed")
    if (
        spec.get("status") != "FROZEN_ONE_SHOT_BEFORE_CHEAP_CONTROL"
        or spec.get("program") != "parallax"
        or spec.get("scope") != "primary_pg1_processor_only_preflight_before_h100"
        or spec.get("provider") != "vast.ai"
        or spec.get("run_id") != RUN_ID
        or spec.get("one_shot_authorization", {}).get("id") != RUN_ID
        or spec.get("automatic_paid_retry") is not False
    ):
        raise PrimaryProcessorLaunchError("processor preflight escaped its one-shot scope")
    if spec.get("authorization") != {
        "paid_control": True,
        "h100": False,
        "model_weights": False,
        "model_loads": False,
        "gpu_operations": False,
        "processor_reads": True,
        "discovery_processor_anchors": True,
        "training": False,
        "confirmation": False,
        "promotion": False,
        "science_claim": False,
    }:
        raise PrimaryProcessorLaunchError("processor preflight authorization changed")
    if spec.get("storage") != {
        "durable_provider": "amazon_s3",
        "aws_profile": "fuelos",
        "region": REGION,
        "bucket": BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
        "ephemeral_scratch_gb": 30,
    }:
        raise PrimaryProcessorLaunchError("processor preflight storage boundary changed")
    if spec.get("worker_image") != RUNTIME_IMAGE:
        raise PrimaryProcessorLaunchError("processor preflight runtime image changed")
    worker = spec.get("worker", {})
    if worker != {
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "gpu_kind": "non_h100_cpu_only_processor_control",
        "launch_mode": "managed_template_private_runtime",
        "image": RUNTIME_IMAGE,
        "template_hash": RUNTIME_TEMPLATE,
        "disk_gb": 30,
        "minimum_cpu_ram_mb": 8000,
        "minimum_cpu_cores_effective": 2,
        "minimum_reliability": 0.95,
        "minimum_inet_down_mbps": 100,
        "minimum_cuda": 13.0,
        "price_ceiling_dph": 0.12,
        "bootstrap_seconds": 600,
        "hard_destroy_seconds": 600,
    }:
        raise PrimaryProcessorLaunchError("processor preflight worker construction changed")
    if spec.get("job") != {
        "stage": "primary_pg1_interface_processor_preflight",
        "command": EXPECTED_COMMAND,
        "maximum_seconds": 300,
        "maximum_terminal_bytes": 1024 * 1024,
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
        "bounds": {
            "parallel_downloads": 4,
            "download_timeout_seconds": 180,
            "preflight_timeout_seconds": 120,
            "minimum_free_bytes": 5_000_000_000,
        },
    }:
        raise PrimaryProcessorLaunchError("processor preflight job bounds changed")
    if spec.get("result") != {
        "key": f"parallax/results/{RUN_ID}/terminal.json",
        "maximum_bytes": 1024 * 1024,
        "put_url_seconds": 900,
        "wait_seconds": 360,
        "required_version_id": True,
        "required_encryption": "AES256",
        "durable_action": "destroy",
        "upload_failure_action": "stop",
    }:
        raise PrimaryProcessorLaunchError("processor preflight durability contract changed")
    for label, row in spec.get("bound_artifacts", {}).items():
        _resolve_bound(root, row, label=f"processor preflight bound artifact {label}")
    for label, row in spec.get("implementation", {}).items():
        _resolve_bound(root, row, label=f"processor preflight implementation {label}")
    _processor_rows(root, spec)
    _overlay_rows(root, spec)
    bundle, _receipt = _bundle(root, spec)
    if bundle["sha256"] != base_bundle_sha256():
        raise PrimaryProcessorLaunchError("processor preflight bundle identity changed")
    if set(spec.get("invariants", [])) != {
        "processor_preflight_terminal_must_precede_any_H100_selection_or_create",
        "H100_unreachable_in_this_executable",
        "no_model_weight_URL_or_byte_enters_control_instance",
        "CUDA_devices_hidden_and_gpu_operations_zero",
        "model_loads_zero",
        "confirmation_reads_zero",
        "no_automatic_retry",
        "literal_cheapest_eligible_non_H100_all_regions",
        "no_Vast_persistent_storage",
        "30_GB_disposable_scratch_only",
        "versioned_encrypted_S3_terminal_before_destroy",
        "upload_failure_stops_billing_and_preserves_scratch",
    }:
        raise PrimaryProcessorLaunchError("processor preflight invariants changed")
    return spec


def base_bundle_sha256() -> str:
    return "8b2f16138376e992816f669fb63ab9ff18d6085da3d009aa9931f6e0a834c517"


def validate_execution_freeze(
    root: Path, spec_path: Path, freeze_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "FROZEN_ONE_SHOT_PRIMARY_PG1_PROCESSOR_PREFLIGHT"
        or freeze.get("run_id") != RUN_ID
        or freeze.get("automatic_paid_retry") is not False
        or freeze.get("authorization") != {
            "create_one_cheap_non_h100_control_instance": True,
            "create_vast_volumes": False,
            "h100": False,
            "model_weights": False,
            "model_loads": False,
            "gpu_operations": False,
            "processor_reads": True,
            "confirmation": False,
            "training": False,
            "automatic_retry": False,
        }
        or freeze.get("launch_spec", {}).get("path")
        != str(spec_path.resolve().relative_to(root.resolve()))
        or freeze.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
    ):
        raise PrimaryProcessorLaunchError("processor preflight execution freeze is absent")
    rehearsal_path = _resolve_bound(
        root, freeze.get("no_create_rehearsal", {}), label="processor preflight no-create"
    )
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if (
        rehearsal.get("status") != "GREEN_PRIMARY_PROCESSOR_PREFLIGHT_NO_CREATE"
        or rehearsal.get("run_id") != RUN_ID
        or rehearsal.get("launch_spec", {}).get("sha256") != sha256_file(spec_path)
        or rehearsal.get("mutation") != {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "s3_objects_written": 0,
            "files_downloaded": 0,
            "gpu_operations": 0,
            "h100_selections": 0,
        }
        or rehearsal.get("paid_authorization") is not False
    ):
        raise PrimaryProcessorLaunchError("processor preflight no-create rehearsal is not exact")
    plants_path = _resolve_bound(root, freeze.get("planted_tests", {}), label="processor plants")
    plants = json.loads(plants_path.read_text(encoding="utf-8"))
    if (
        plants.get("status") != "GREEN"
        or plants.get("exit_code") != 0
        or plants.get("paid_action") is not False
        or plants.get("h100_instances_created") != 0
        or plants.get("confirmation_items_read") != 0
    ):
        raise PrimaryProcessorLaunchError("processor preflight plants are not exact green")
    return freeze


def _job_contract(
    spec: dict[str, Any], *, instance_id: int, processor: list[dict[str, Any]],
    bundle: dict[str, Any], bundle_receipt: dict[str, Any], overlays: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT_JOB",
        "stage": "primary_pg1_interface_processor_preflight",
        "provider": "vast.ai",
        "run_id": RUN_ID,
        "lineage": LINEAGE,
        "instance_id": instance_id,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "ephemeral_scratch_gb": 30,
            "aws_credentials_on_worker": False,
        },
        "processor": processor,
        "bundle": {**bundle, "entries": bundle_receipt["entries"]},
        "overlays": overlays,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "snapshot_root": (
                "/workspace/parallax/cache/models--google--paligemma-3b-pt-224/"
                "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
            ),
            "preflight_output": "/workspace/parallax/processor-preflight.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": spec["job"]["bounds"],
        "firewall": {
            "h100_creation_reachable": False,
            "model_weight_urls": 0,
            "model_loads": 0,
            "gpu_operations": 0,
            "science_items_evaluated": 0,
            "discovery_items_processor_only": 3,
            "confirmation_items_read": 0,
            "automatic_retry": False,
        },
    }


def _worker_contract(spec: dict[str, Any], *, instance_id: int, result_url: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "run_id": RUN_ID,
        "stage": "primary_pg1_interface_processor_preflight",
        "instance_id": instance_id,
        "runtime_root": "/workspace/parallax",
        "job_output_path": "/workspace/parallax/job-result.json",
        "job_log_path": "/workspace/parallax/job.log",
        "terminal_path": "/workspace/parallax/terminal.json",
        "instance_api_key_path": "/run/parallax/instance-api-key",
        "command": EXPECTED_COMMAND,
        "maximum_job_seconds": int(spec["job"]["maximum_seconds"]),
        "maximum_result_bytes": int(spec["result"]["maximum_bytes"]),
        "result_key": spec["result"]["key"],
        "result_put_url": result_url,
        "post_upload_action": "destroy",
        "upload_failure_action": "stop",
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
    }


def _state_path(root: Path, spec: dict[str, Any]) -> Path:
    output = Path(str(spec.get("output_receipt", {}).get("path", "")))
    if output.is_absolute() or not output.parts or ".." in output.parts:
        raise PrimaryProcessorLaunchError("processor preflight output path is unsafe")
    return root / output.parent / "launch-state.json"


def run(root: Path, spec_path: Path, freeze_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    validate_execution_freeze(root, spec_path, freeze_path, spec)
    initial = _require_initial_zero()
    s3_safety = require_s3_safety()
    processor_rows = _processor_rows(root, spec)
    overlays = _overlay_rows(root, spec)
    bundle_row, bundle_receipt = _bundle(root, spec)
    input_proof = _require_input_heads(processor_rows, bundle_row, [])
    _require_result_absent(spec["result"]["key"])
    _claim(root, spec)
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    if not key_path.is_file():
        raise PrimaryProcessorLaunchError("SSH private key is absent")

    instance_id: int | None = None
    market: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    terminal_body: bytes | None = None
    s3_proof: dict[str, Any] | None = None
    job_released = False
    cleanup: dict[str, Any] | None = None
    state_path = _state_path(root, spec)
    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "CLAIMED_BEFORE_PAID_MUTATION",
        "instance_id": None,
        "offer": None,
        "job_released": False,
        "s3_durable": False,
        "automatic_retry": False,
    }

    def record(phase: str, **updates: Any) -> None:
        state.update(updates)
        state["phase"] = phase
        atomic_json(state_path, state)

    record("CLAIMED_BEFORE_PAID_MUTATION")
    started = time.monotonic()
    try:
        market = select_cheapest_cpu_preflight_offer(spec["worker"])
        record("CHEAP_NON_H100_SELECTED", offer=market["selected"])
        created = _create_instance(
            int(market["selected"]["id"]),
            _create_arguments(role="worker", spec=spec, run_id=RUN_ID),
        )
        instance_id = int(created["new_contract"])
        record("CHEAP_NON_H100_CREATED", instance_id=instance_id)
        row = wait_ssh(instance_id, key_path, timeout_seconds=int(spec["worker"]["bootstrap_seconds"]))
        ssh = _ssh_base(row, key_path)
        credentials = _aws_credentials()
        now = datetime.now(timezone.utc)
        processor = [_signed(item, credentials, now) for item in processor_rows]
        bundle = _signed(bundle_row, credentials, now)
        result_url = presign_result_put(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            bucket=BUCKET,
            key=spec["result"]["key"],
            region=REGION,
            now=now,
            expires_seconds=int(spec["result"]["put_url_seconds"]),
        )
        credentials.clear()
        job_contract = _job_contract(
            spec, instance_id=instance_id, processor=processor, bundle=bundle,
            bundle_receipt=bundle_receipt, overlays=overlays,
        )
        worker_contract = _worker_contract(spec, instance_id=instance_id, result_url=result_url)
        validate_job_contract(job_contract)
        validate_worker_contract(worker_contract)
        for remote, relative in {
            "/opt/parallax/ephemeral_s3_worker.py": "infra/vast/parallax/ephemeral_s3_worker.py",
            "/opt/parallax/ephemeral_optimizer_scale_job.py": (
                "infra/vast/parallax/ephemeral_optimizer_scale_job.py"
            ),
            "/opt/parallax/primary_interface_processor_preflight_job.py": (
                "infra/vast/parallax/primary_interface_processor_preflight_job.py"
            ),
        }.items():
            _stage(ssh, remote=remote, content=(root / relative).read_bytes(), mode="0700")
        for source_row, staged in zip(spec["inputs"]["overlays"], overlays, strict=True):
            _stage(
                ssh,
                remote=str(staged["staged_path"]),
                content=(root / str(source_row["path"])).read_bytes(),
            )
        _stage(
            ssh,
            remote="/run/parallax/processor-preflight-contract.json",
            content=json.dumps(job_contract, sort_keys=True).encode("utf-8"),
        )
        _stage(
            ssh,
            remote="/run/parallax/worker-contract.json",
            content=json.dumps(worker_contract, sort_keys=True).encode("utf-8"),
        )
        _stage(
            ssh,
            remote="/run/parallax/instance-api-key",
            content=str(created["instance_api_key"]).encode("utf-8"),
        )
        processor.clear()
        bundle.clear()
        job_contract.clear()
        job_released = True
        record("PROCESSOR_PREFLIGHT_RELEASED", job_released=True)
        started_worker = _run([*ssh, _worker_start_command()], check=False, timeout_seconds=30)
        if started_worker.returncode != 0:
            raise PrimaryProcessorLaunchError("processor preflight worker did not start")
        terminal, terminal_body, s3_proof = _wait_result(
            key=spec["result"]["key"],
            run_id=RUN_ID,
            instance_id=instance_id,
            seconds=int(spec["result"]["wait_seconds"]),
        )
        record("VERSIONED_S3_TERMINAL_DURABLE", s3_durable=True)
    finally:
        cleanup = _finalize_lifecycle(
            worker_id=instance_id,
            guardian_id=None,
            job_released=job_released,
            s3_durable=s3_proof is not None,
        )
    final = _confirm_account_zero(seconds=45)
    if terminal is None or terminal_body is None or s3_proof is None:
        raise PrimaryProcessorLaunchError("processor preflight terminal is absent")
    payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    scientific_green = (
        terminal.get("status") == "GREEN_JOB_TERMINAL"
        and payload.get("status") == "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT"
        and payload.get("h100_instances_created") == 0
        and payload.get("model_weight_urls") == 0
        and payload.get("model_loads") == 0
        and payload.get("gpu_operations") == 0
        and payload.get("confirmation_items_read") == 0
    )
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT_EXECUTED"
            if scientific_green else "RED_PRIMARY_PG1_PROCESSOR_PREFLIGHT_EXECUTED"
        ),
        "run_id": RUN_ID,
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "initial_account": initial,
        "s3_safety": s3_safety,
        "input_proof": input_proof,
        "market": market,
        "instance_id": instance_id,
        "terminal": terminal,
        "processor_preflight": payload,
        "s3_proof": s3_proof,
        "cleanup": cleanup,
        "terminal_account": final,
        "h100_instances_created": 0,
        "vast_persistent_volumes_created": 0,
        "wall_seconds": time.monotonic() - started,
        "listed_rate_upper_bound_usd": (
            float(market["selected"]["dph_total"]) * (time.monotonic() - started) / 3600
            if market is not None else None
        ),
        "automatic_retry": False,
        "promotion_boundary": "PROCESSOR_CONFOUND_GATE_ONLY_NO_SCIENCE_CLAIM_OR_H100_AUTHORIZATION",
    }
    output = root / str(spec["output_receipt"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, packet)
    (output.parent / "terminal.json").write_bytes(terminal_body)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root.resolve(), args.spec.resolve(), args.freeze.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
