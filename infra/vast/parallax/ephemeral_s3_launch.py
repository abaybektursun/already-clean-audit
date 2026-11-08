"""Launch one disposable Vast worker guarded before creation and durable in S3.

This is the successor to the retired persistent-volume controller.  It has no
volume operations.  A cheap disposable guardian is armed first, the literal
cheapest eligible worker is then selected from the live all-region market, and
the worker publishes a small terminal JSON to versioned S3 before destroying
itself.  There is no automatic paid retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .guarded_launch import (
    LifecycleError,
    _create_instance,
    _ephemeral_worker_registry_login,
    _rows,
    _run,
    _scp_base,
    _show_instance,
    _ssh_base,
    _vast,
    ttl_onstart,
    wait_ssh,
)
from .s3_presign_put import presign_result_put


BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
AWS_PROFILE = "fuelos"
PUBLIC_OFFER_FIELDS = (
    "id",
    "machine_id",
    "gpu_name",
    "gpu_ram",
    "reliability",
    "dph_total",
    "geolocation",
    "verification",
    "cuda_max_good",
    "inet_down",
    "inet_up",
)


class EphemeralLaunchError(LifecycleError):
    pass


def _sha256_json_file(path: Path) -> str:
    return sha256_file(path)


def _resolve_bound(root: Path, row: dict[str, Any], *, label: str) -> Path:
    relative = Path(str(row.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise EphemeralLaunchError(f"unsafe {label} path")
    path = (root / relative).resolve()
    if _sha256_json_file(path) != row.get("sha256"):
        raise EphemeralLaunchError(f"{label} hash mismatch")
    return path


def _materialize_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") == 1:
        return raw
    if raw.get("schema_version") == 3:
        expected_transport_keys = {
            "schema_version",
            "created_utc",
            "status",
            "scope",
            "run_id",
            "base_spec",
            "justification",
            "one_shot_authorization",
            "result_key",
            "output_receipt",
            "worker_override",
            "implementation",
        }
        if set(raw) != expected_transport_keys:
            raise EphemeralLaunchError("stock transport overlay has an unexpected schema")
        if raw.get("status") != "FROZEN_BEFORE_PAID_REHEARSAL" or raw.get("scope") != "ephemeral_s3_terminal_transport_gate":
            raise EphemeralLaunchError("stock transport overlay escapes its frozen scope")
        base_path = _resolve_bound(root, raw["base_spec"], label="base rehearsal spec")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if base.get("run_id") != "parallax-ephemeral-s3-terminal-rehearsal-v1":
            raise EphemeralLaunchError("stock transport gate does not derive from exact v1")
        justification = raw["justification"]
        _resolve_bound(root, justification["v2_receipt"], label="v2 RED receipt")
        if justification.get("unresolved_boundary") != "combined_live_S3_HTTP_200_version_id_then_worker_self_DESTROY":
            raise EphemeralLaunchError("stock gate does not isolate the registered missing boundary")
        if justification.get("only_semantic_changes") != [
            "new_one_shot_run_identity_and_S3_key",
            "replace_4.34GB_private_runtime_with_1.31GB_proven_stock_image",
            "reduce_ephemeral_worker_disk_from_32GB_to_10GB",
            "remove_ECR_login_and_managed_template_from_transport_only_worker",
        ]:
            raise EphemeralLaunchError("stock transport gate changes more than its startup confound")
        worker_override = raw["worker_override"]
        if worker_override != {
            "launch_mode": "stock_direct_ssh",
            "image": "vastai/base-image@sha256:a264d0c242bebbdaa6f50b34fb81c7e06cb21d8487a202f27cce22794a2e68e4",
            "disk_gb": 10,
            "minimum_cuda": 0,
            "bootstrap_seconds": 180,
        }:
            raise EphemeralLaunchError("stock worker override is not exact")
        effective = json.loads(json.dumps(base))
        effective["created_utc"] = raw["created_utc"]
        effective["scope"] = raw["scope"]
        effective["run_id"] = raw["run_id"]
        effective["one_shot_authorization"] = raw["one_shot_authorization"]
        effective["result"]["key"] = raw["result_key"]
        effective["output_receipt"] = raw["output_receipt"]
        effective["worker"].update(worker_override)
        effective["worker"].pop("template_hash", None)
        effective["implementation"] = raw["implementation"]
        effective["justification"] = justification
        effective["source_schema_version"] = 3
        return effective
    if raw.get("schema_version") == 4:
        expected_correction_keys = {
            "schema_version",
            "created_utc",
            "status",
            "scope",
            "run_id",
            "base_stock_gate",
            "correction_from",
            "one_shot_authorization",
            "result_key",
            "output_receipt",
            "implementation",
        }
        if set(raw) != expected_correction_keys:
            raise EphemeralLaunchError("corrected stock transport overlay has an unexpected schema")
        if raw.get("status") != "FROZEN_BEFORE_PAID_REHEARSAL" or raw.get("scope") != "ephemeral_s3_terminal_transport_gate":
            raise EphemeralLaunchError("corrected stock gate escapes its frozen scope")
        base_path = _resolve_bound(root, raw["base_stock_gate"], label="base stock transport gate")
        base = _materialize_spec(root, base_path)
        if base.get("run_id") != "parallax-ephemeral-s3-terminal-stock-gate-v1":
            raise EphemeralLaunchError("corrected stock gate does not derive from exact v1")
        correction = raw["correction_from"]
        _resolve_bound(root, correction["receipt"], label="stock v1 RED receipt")
        if correction.get("failure") != "RED_REMOTE_WORKER_NOT_STARTED":
            raise EphemeralLaunchError("corrected stock gate does not bind its exact failure")
        if correction.get("only_semantic_changes") != [
            "new_one_shot_run_identity_and_S3_key",
            "mkdir_runtime_root_before_detached_log_redirection",
            "verify_spawned_worker_PID_or_terminal_file",
            "STOP_and_preserve_if_job_released_without_S3_durability",
        ]:
            raise EphemeralLaunchError("corrected stock gate changes more than the audited defects")
        effective = json.loads(json.dumps(base))
        effective["created_utc"] = raw["created_utc"]
        effective["run_id"] = raw["run_id"]
        effective["one_shot_authorization"] = raw["one_shot_authorization"]
        effective["result"]["key"] = raw["result_key"]
        effective["output_receipt"] = raw["output_receipt"]
        effective["implementation"] = raw["implementation"]
        effective["correction_from"] = correction
        effective["source_schema_version"] = 4
        return effective
    expected_keys = {
        "schema_version",
        "created_utc",
        "status",
        "scope",
        "run_id",
        "base_spec",
        "correction_from",
        "one_shot_authorization",
        "result_key",
        "output_receipt",
        "implementation",
    }
    if raw.get("schema_version") != 2 or set(raw) != expected_keys:
        raise EphemeralLaunchError("corrected rehearsal overlay has an unexpected schema")
    if raw.get("status") != "FROZEN_BEFORE_PAID_REHEARSAL" or raw.get("scope") != "ephemeral_s3_terminal_rehearsal":
        raise EphemeralLaunchError("corrected rehearsal overlay escapes the frozen scope")
    base_path = _resolve_bound(root, raw["base_spec"], label="base rehearsal spec")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("run_id") != "parallax-ephemeral-s3-terminal-rehearsal-v1":
        raise EphemeralLaunchError("corrected rehearsal does not derive from exact v1")
    correction = raw["correction_from"]
    _resolve_bound(root, correction["receipt"], label="v1 RED receipt")
    if correction.get("run_id") != base["run_id"] or correction.get("failure") != "RED_GUARDIAN_LIVENESS_RACE":
        raise EphemeralLaunchError("corrected rehearsal does not bind the v1 failure")
    if correction.get("only_semantic_changes") != [
        "new_one_shot_run_identity_and_S3_key",
        "ten_consecutive_successful_absent_observations_required",
        "provider_listing_errors_never_count_as_absence",
        "controller_requires_guardian_liveness_during_worker_bootstrap",
    ]:
        raise EphemeralLaunchError("corrected rehearsal changes more than the audited defects")
    effective = json.loads(json.dumps(base))
    effective["created_utc"] = raw["created_utc"]
    effective["run_id"] = raw["run_id"]
    effective["one_shot_authorization"] = raw["one_shot_authorization"]
    effective["result"]["key"] = raw["result_key"]
    effective["output_receipt"] = raw["output_receipt"]
    effective["implementation"] = raw["implementation"]
    effective["correction_from"] = correction
    effective["source_schema_version"] = 2
    return effective


def validate_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = _materialize_spec(root, spec_path)
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN_BEFORE_PAID_REHEARSAL":
        raise EphemeralLaunchError("ephemeral launch spec is not frozen rehearsal schema v1")
    if spec.get("scope") not in {
        "ephemeral_s3_terminal_rehearsal",
        "ephemeral_s3_terminal_transport_gate",
    } or spec.get("provider") != "vast.ai":
        raise EphemeralLaunchError("only the non-science S3 terminal rehearsal is authorized")
    if spec.get("run_id") != spec.get("one_shot_authorization", {}).get("id"):
        raise EphemeralLaunchError("one-shot authorization is not bound to run id")
    if spec.get("automatic_paid_retry") is not False:
        raise EphemeralLaunchError("automatic paid retry must be explicitly false")
    storage = spec.get("storage", {})
    if storage != {
        "durable_provider": "amazon_s3",
        "aws_profile": AWS_PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "vast_persistent_volumes": "FORBIDDEN",
        "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
    }:
        raise EphemeralLaunchError("storage contract is not exact S3-durable/Vast-ephemeral")
    authorization = spec.get("authorization", {})
    if authorization != {
        "paid_rehearsal": True,
        "h100": False,
        "model_reads": False,
        "data_reads": False,
        "training": False,
        "science": False,
    }:
        raise EphemeralLaunchError("rehearsal authorization escapes the mechanical boundary")
    for role in ("guardian", "worker"):
        row = spec.get(role, {})
        if not isinstance(row.get("disk_gb"), int) or not 8 <= row["disk_gb"] <= 64:
            raise EphemeralLaunchError(f"{role} disk is outside the frozen ephemeral bound")
        if not 0 < float(row.get("price_ceiling_dph", 0)) <= 0.25:
            raise EphemeralLaunchError(f"{role} price ceiling is invalid")
    if spec["worker"].get("gpu_kind") != "non_h100_mechanical_worker":
        raise EphemeralLaunchError("worker kind escapes the non-H100 rehearsal boundary")
    if spec["worker"].get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time":
        raise EphemeralLaunchError("worker selection is not literal-cheapest")
    if spec["guardian"].get("selection_rule") != "literal_cheapest_eligible_all_regions_at_query_time":
        raise EphemeralLaunchError("guardian selection is not literal-cheapest")
    result = spec.get("result", {})
    expected_key = f"parallax/results/{spec['run_id']}/terminal.json"
    if result.get("key") != expected_key or result.get("maximum_bytes") != 1024 * 1024:
        raise EphemeralLaunchError("terminal result destination or size is not exact")
    if result.get("put_url_seconds") not in range(300, 1801):
        raise EphemeralLaunchError("terminal PUT lifetime is unsafe")
    job = spec.get("job", {})
    if job.get("stage") != "ephemeral_s3_terminal_rehearsal" or job.get("maximum_seconds") != 30:
        raise EphemeralLaunchError("rehearsal job is not the exact 30-second mechanical probe")
    command = job.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
        raise EphemeralLaunchError("rehearsal command is malformed")
    for label, row in spec.get("bound_artifacts", {}).items():
        _resolve_bound(root, row, label=label)
    expected_implementations = {
        "launcher": "infra/vast/parallax/ephemeral_s3_launch.py",
        "worker": "infra/vast/parallax/ephemeral_s3_worker.py",
        "guardian": "infra/vast/parallax/ephemeral_guardian.py",
        "presigner": "infra/vast/parallax/s3_presign_put.py",
    }
    for label, relative in expected_implementations.items():
        row = spec.get("implementation", {}).get(label, {})
        if row.get("path") != relative or sha256_file(root / relative) != row.get("sha256"):
            raise EphemeralLaunchError(f"{label} implementation hash mismatch")
    return spec


def _sanitized_offer(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in PUBLIC_OFFER_FIELDS if name in row}


def select_cheapest_offer(
    *, role: str, contract: dict[str, Any], excluded_machine_ids: set[int] | None = None
) -> dict[str, Any]:
    disk = int(contract["disk_gb"])
    query = (
        f"cpu_arch=amd64 num_gpus=1 reliability>={float(contract['minimum_reliability']):g} "
        f"disk_space>={disk} duration>=1 rentable=true rented=false static_ip=true "
        f"direct_port_count>=2 inet_down>={float(contract['minimum_inet_down_mbps']):g}"
    )
    minimum_cuda = float(contract.get("minimum_cuda", 0))
    if minimum_cuda:
        query += f" cuda_max_good>={minimum_cuda:g}"
    rows = _rows(
        _vast(
            "search",
            "offers",
            query,
            "--limit",
            "100",
            "--storage",
            str(disk),
            "--order",
            "dph_total",
        )
    )
    excluded = excluded_machine_ids or set()
    rows = [
        row
        for row in rows
        if "H100" not in str(row.get("gpu_name", "")).upper()
        and int(row.get("machine_id", -1)) not in excluded
    ]
    rows.sort(key=lambda row: (float(row.get("dph_total", float("inf"))), -float(row.get("reliability", 0))))
    if not rows:
        raise EphemeralLaunchError(f"no eligible non-H100 {role} offer exists")
    selected = rows[0]
    if float(selected.get("dph_total", float("inf"))) > float(contract["price_ceiling_dph"]):
        raise EphemeralLaunchError(f"literal-cheapest {role} exceeds the frozen price ceiling")
    return {
        "query": query,
        "candidate_count": len(rows),
        "excluded_machine_ids": sorted(excluded),
        "selection_rule": "literal_cheapest_eligible_all_regions_at_query_time",
        "selected": _sanitized_offer(selected),
    }


def _account_zero() -> dict[str, Any]:
    instances = _vast("show", "instances", check=False)
    volumes = _vast("show", "volumes", check=False)
    if not isinstance(instances, list) or not isinstance(volumes, list):
        raise EphemeralLaunchError("could not prove the Vast account state")
    return {
        "active_instances": len(instances),
        "active_volumes": len(volumes),
        "instance_ids": sorted(int(row["id"]) for row in instances if str(row.get("id", "")).isdigit()),
        "volume_ids": sorted(int(row["id"]) for row in volumes if str(row.get("id", "")).isdigit()),
    }


def _require_initial_zero() -> dict[str, Any]:
    packet = _account_zero()
    if packet["active_instances"] != 0 or packet["active_volumes"] != 0:
        raise EphemeralLaunchError("paid rehearsal requires zero pre-existing Vast resources")
    return packet


def _aws_json(*arguments: str, timeout_seconds: int = 30) -> Any:
    completed = _run(
        [
            "aws",
            "--profile",
            AWS_PROFILE,
            "--region",
            REGION,
            "--output",
            "json",
            *arguments,
        ],
        check=False,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        raise EphemeralLaunchError(f"fuelos AWS command failed: {arguments[0]}")
    raw = completed.stdout.decode("utf-8").strip()
    return json.loads(raw) if raw else {}


def require_s3_safety() -> dict[str, Any]:
    versioning = _aws_json("s3api", "get-bucket-versioning", "--bucket", BUCKET)
    public = _aws_json("s3api", "get-public-access-block", "--bucket", BUCKET)
    encryption = _aws_json("s3api", "get-bucket-encryption", "--bucket", BUCKET)
    blocked = public.get("PublicAccessBlockConfiguration", {})
    algorithms = [
        row.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
        for row in encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
    ]
    if versioning.get("Status") != "Enabled" or set(blocked.values()) != {True} or "AES256" not in algorithms:
        raise EphemeralLaunchError("S3 versioning/public-block/encryption gate is not green")
    return {
        "versioning": "Enabled",
        "all_public_access_blocked": True,
        "default_encryption": "AES256",
        "profile": AWS_PROFILE,
        "region": REGION,
    }


def _aws_credentials() -> dict[str, Any]:
    completed = _run(
        ["aws", "configure", "export-credentials", "--profile", AWS_PROFILE, "--format", "process"],
        check=False,
        timeout_seconds=30,
    )
    if completed.returncode != 0:
        raise EphemeralLaunchError("could not obtain in-memory fuelos credentials")
    credentials = json.loads(completed.stdout.decode("utf-8"))
    if not credentials.get("AccessKeyId") or not credentials.get("SecretAccessKey"):
        raise EphemeralLaunchError("fuelos credential export is incomplete")
    return credentials


def _claim(root: Path, spec: dict[str, Any]) -> Path:
    relative = Path(str(spec["one_shot_authorization"].get("claim_path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise EphemeralLaunchError("one-shot claim path is unsafe")
    path = (root / relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "CLAIMED_BEFORE_PAID_MUTATION",
        "run_id": spec["run_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_retry": False,
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise EphemeralLaunchError("one-shot paid authorization was already consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _stage(
    ssh: list[str],
    *,
    remote: str,
    content: bytes,
    mode: str = "0600",
    attempts: int = 3,
    retry_seconds: float = 1.0,
) -> None:
    """Idempotently overwrite one small pre-release control file across transient SSH loss."""
    if not remote.startswith("/") or any(character.isspace() for character in remote):
        raise EphemeralLaunchError("unsafe remote staging path")
    if not 2 <= attempts <= 5 or not 0 <= retry_seconds <= 5:
        raise ValueError("remote staging retry bounds are invalid")
    parent = str(Path(remote).parent)
    completed: subprocess.CompletedProcess[bytes] | None = None
    for attempt in range(1, attempts + 1):
        completed = _run(
            [*ssh, f"umask 077; mkdir -p {parent}; cat > {remote}; chmod {mode} {remote}"],
            input_bytes=content,
            check=False,
            timeout_seconds=30,
        )
        if completed.returncode == 0:
            return
        if attempt < attempts:
            time.sleep(retry_seconds)
    raise EphemeralLaunchError(
        f"remote staging failed after {attempts} idempotent attempts; "
        f"returncode={completed.returncode if completed is not None else 'absent'}"
    )


def _wait_remote_file(ssh: list[str], path: str, *, seconds: int = 30) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        completed = _run([*ssh, f"test -s {path}"], check=False, timeout_seconds=15)
        if completed.returncode == 0:
            return
        time.sleep(1)
    raise EphemeralLaunchError(f"remote readiness file did not appear: {path}")


def _stage_atomic(
    ssh: list[str], *, remote: str, content: bytes, mode: str = "0600"
) -> None:
    """Publish a small controller file without exposing a partial final path."""
    pending = f"{remote}.controller-pending"
    _stage(ssh, remote=pending, content=content, mode=mode)
    completed = _run(
        [*ssh, f"mv -f {pending} {remote}"],
        check=False,
        timeout_seconds=30,
    )
    if completed.returncode != 0:
        raise EphemeralLaunchError("atomic remote staging failed")


def _wait_remote_json_exact(
    ssh: list[str], path: str, expected: dict[str, Any], *, seconds: int = 30
) -> dict[str, Any]:
    """Wait for a remote process to publish the exact expected acknowledgement."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        completed = _run(
            [*ssh, f"test -s {path} && cat {path}"],
            check=False,
            timeout_seconds=15,
        )
        if completed.returncode == 0:
            try:
                observed = json.loads(completed.stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                observed = None
            if observed == expected:
                return observed
            if observed is not None:
                raise EphemeralLaunchError("remote acknowledgement identity drift")
        time.sleep(1)
    raise EphemeralLaunchError(f"exact remote acknowledgement did not appear: {path}")


def _require_guardian_live(
    instance_id: int, *, absent_observations: int = 3, poll_seconds: float = 1.0
) -> None:
    """Tolerate brief empty provider listings without accepting a dead guardian."""
    if absent_observations < 1:
        raise ValueError("absent_observations must be positive")
    for attempt in range(absent_observations):
        row = _show_instance(instance_id)
        status = str((row or {}).get("actual_status") or "").lower()
        if status == "running":
            return
        if row is not None:
            raise EphemeralLaunchError(f"independent guardian is not live: {status or 'unknown'}")
        if attempt + 1 < absent_observations:
            time.sleep(poll_seconds)
    raise EphemeralLaunchError("independent guardian is not live: absent_after_confirmed_observations")


def _create_arguments(*, role: str, spec: dict[str, Any], run_id: str) -> list[str]:
    row = spec[role]
    hard_deadman = ttl_onstart(action="destroy", seconds=int(row["hard_destroy_seconds"]))
    common = [
        "--disk",
        str(row["disk_gb"]),
        "--cancel-unavail",
        "--label",
        f"parallax-{role}-{run_id}" if role == "guardian" else f"parallax-worker-{run_id}",
        "--onstart-cmd",
        hard_deadman,
    ]
    if role == "guardian":
        return ["--image", str(row["image"]), "--ssh", "--direct", *common]
    launch_mode = str(row.get("launch_mode", "managed_private_runtime"))
    if launch_mode == "stock_direct_ssh":
        return ["--image", str(row["image"]), "--ssh", "--direct", *common]
    if launch_mode not in {"managed_private_runtime", "managed_template_private_runtime"}:
        raise EphemeralLaunchError("worker launch mode is not frozen")
    login = _ephemeral_worker_registry_login(spec)
    if not login:
        raise EphemeralLaunchError("worker ECR login was not resolved")
    return [
        "--template_hash",
        str(row["template_hash"]),
        "--image",
        str(row["image"]),
        *common,
        "--login",
        login,
    ]


def _frozen_worker_assignment(*, run_id: str, worker_id: int) -> dict[str, Any]:
    if worker_id <= 0 or not run_id:
        raise EphemeralLaunchError("guardian worker assignment identity is invalid")
    return {
        "schema_version": 1,
        "status": "FROZEN_WORKER_ASSIGNMENT",
        "run_id": run_id,
        "worker_label": f"parallax-worker-{run_id}",
        "worker_id": worker_id,
    }


def _expected_guardian_assignment_ack(*, run_id: str, worker_id: int) -> dict[str, Any]:
    return {
        **_frozen_worker_assignment(run_id=run_id, worker_id=worker_id),
        "status": "GUARDIAN_WORKER_ASSIGNMENT_ACKNOWLEDGED",
    }


def _destroy_best_effort(instance_id: int | None) -> None:
    if instance_id is None:
        return
    # A transient empty listing must never suppress a billing-ending action.
    # Vast destroy is idempotent under check=False, so request it directly.
    _vast("destroy", "instance", str(instance_id), check=False)


def _terminal_worker_cleanup(
    instance_id: int | None, *, job_released: bool, s3_durable: bool
) -> dict[str, Any]:
    """Destroy only before release or after durability; otherwise STOP and preserve."""
    if instance_id is None:
        return {"action": "none", "reason": "worker_not_created"}
    if s3_durable or not job_released:
        # Never gate cleanup on a preceding provider listing. The incident on
        # 2026-07-18 proved that a transient empty listing can coexist with a
        # billable loading H100.
        _vast("destroy", "instance", str(instance_id), check=False)
        return {
            "action": "destroy",
            "reason": "S3_durable" if s3_durable else "job_not_released",
        }
    _vast("stop", "instance", str(instance_id), check=False)
    return {
        "action": "stop_preserve",
        "reason": "job_released_without_S3_durability",
    }


def _enforce_instances_destroyed(
    instance_ids: list[int | None],
    *,
    seconds: int = 45,
    consecutive_absences: int = 3,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    """Repeat idempotent destroys until every known instance is confirmed absent."""
    ids = sorted({int(value) for value in instance_ids if value is not None})
    if consecutive_absences < 2 or seconds < 1:
        raise ValueError("destroy confirmation bounds are invalid")
    if not ids:
        return {
            "status": "GREEN_NO_KNOWN_INSTANCES",
            "instance_ids": [],
            "consecutive_absences": consecutive_absences,
            "reissued_destroy_requests": 0,
        }
    confirmations = {instance_id: 0 for instance_id in ids}
    reissued = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for instance_id in ids:
            row = _show_instance(instance_id)
            if row is None:
                confirmations[instance_id] += 1
            else:
                confirmations[instance_id] = 0
                _vast("destroy", "instance", str(instance_id), check=False)
                reissued += 1
        if all(value >= consecutive_absences for value in confirmations.values()):
            return {
                "status": "GREEN_KNOWN_INSTANCES_CONFIRMED_ABSENT",
                "instance_ids": ids,
                "consecutive_absences": consecutive_absences,
                "reissued_destroy_requests": reissued,
            }
        time.sleep(poll_seconds)
    raise EphemeralLaunchError(
        "known Vast instances were not confirmed absent before the cleanup deadline"
    )


def _enforce_instance_nonbillable(
    instance_id: int | None,
    *,
    seconds: int = 45,
    consecutive_observations: int = 3,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    """Confirm a released nondurable worker is stopped or absent, reissuing STOP."""
    if instance_id is None:
        return {"status": "GREEN_WORKER_NOT_CREATED", "instance_id": None}
    confirmations = 0
    last_state = "unknown"
    stop_requests = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        row = _show_instance(instance_id)
        if row is None:
            last_state = "absent"
            confirmations += 1
        else:
            actual = str(row.get("actual_status") or row.get("cur_state") or "").lower()
            intended = str(row.get("intended_status") or "").lower()
            if actual in {"stopped", "exited", "offline"} or intended == "stopped":
                last_state = actual or intended
                confirmations += 1
            else:
                last_state = actual or intended or "unknown"
                confirmations = 0
                _vast("stop", "instance", str(instance_id), check=False)
                stop_requests += 1
        if confirmations >= consecutive_observations:
            return {
                "status": "GREEN_WORKER_CONFIRMED_NONBILLABLE",
                "instance_id": instance_id,
                "terminal_state": last_state,
                "consecutive_observations": confirmations,
                "reissued_stop_requests": stop_requests,
            }
        time.sleep(poll_seconds)
    raise EphemeralLaunchError(
        "released nondurable Vast worker was not confirmed nonbillable before deadline"
    )


def _finalize_lifecycle(
    *,
    worker_id: int | None,
    guardian_id: int | None,
    job_released: bool,
    s3_durable: bool,
) -> dict[str, Any]:
    """End billing and prove the terminal action even while another exception unwinds."""
    cleanup = _terminal_worker_cleanup(
        worker_id,
        job_released=job_released,
        s3_durable=s3_durable,
    )
    _destroy_best_effort(guardian_id)
    if s3_durable or not job_released:
        proof = _enforce_instances_destroyed([worker_id, guardian_id])
    else:
        proof = {
            "status": "GREEN_NONDURABLE_WORKER_STOPPED_GUARDIAN_DESTROYED",
            "guardian": _enforce_instances_destroyed([guardian_id]),
            "worker": _enforce_instance_nonbillable(worker_id),
        }
    return {**cleanup, "confirmation": proof}


def _confirm_account_zero(
    *, seconds: int = 45, consecutive_observations: int = 3, poll_seconds: float = 1.0
) -> dict[str, Any]:
    """Require repeated whole-account zero observations; one empty listing is insufficient."""
    confirmations = 0
    last: dict[str, Any] | None = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        last = _account_zero()
        if last["active_instances"] == 0 and last["active_volumes"] == 0:
            confirmations += 1
            if confirmations >= consecutive_observations:
                return {
                    **last,
                    "status": "GREEN_ACCOUNT_ZERO_CONFIRMED",
                    "consecutive_observations": confirmations,
                }
        else:
            confirmations = 0
        time.sleep(poll_seconds)
    raise EphemeralLaunchError(
        f"terminal Vast account-zero confirmation failed; last={last}"
    )


def _worker_start_command() -> str:
    return (
        "set -eu; mkdir -p /workspace/parallax; "
        "nohup python3 /opt/parallax/ephemeral_s3_worker.py --contract "
        "/run/parallax/worker-contract.json >/workspace/parallax/worker.log 2>&1 </dev/null & "
        "worker_pid=$!; kill -0 $worker_pid 2>/dev/null || "
        "test -s /workspace/parallax/terminal.json"
    )


def _require_worker_can_still_publish(instance_id: int) -> bool:
    """Return whether the worker is absent; fail immediately for a stopped worker."""
    row = _show_instance(instance_id)
    if row is None:
        return True
    actual = str(row.get("actual_status") or row.get("cur_state") or "").lower()
    intended = str(row.get("intended_status") or "").lower()
    if actual in {"stopped", "exited", "offline"} or intended == "stopped":
        raise EphemeralLaunchError("worker became terminal before S3 durability")
    return False


def _wait_result(*, key: str, run_id: str, instance_id: int, seconds: int) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    deadline = time.monotonic() + seconds
    head: dict[str, Any] | None = None
    consecutive_absences = 0
    while time.monotonic() < deadline:
        try:
            candidate = _aws_json(
                "s3api",
                "head-object",
                "--bucket",
                BUCKET,
                "--key",
                key,
                "--checksum-mode",
                "ENABLED",
            )
        except EphemeralLaunchError:
            candidate = None
        if isinstance(candidate, dict) and candidate.get("VersionId"):
            head = candidate
            break
        if _require_worker_can_still_publish(instance_id):
            consecutive_absences += 1
            if consecutive_absences >= 3:
                raise EphemeralLaunchError("worker disappeared before S3 durability")
        else:
            consecutive_absences = 0
        time.sleep(2)
    if head is None:
        raise EphemeralLaunchError("versioned S3 terminal result did not appear before deadline")
    version_id = str(head["VersionId"])
    with tempfile.TemporaryDirectory(prefix="parallax-small-result-") as temporary:
        path = Path(temporary) / "terminal.json"
        response = _aws_json(
            "s3api",
            "get-object",
            "--bucket",
            BUCKET,
            "--key",
            key,
            "--version-id",
            version_id,
            "--checksum-mode",
            "ENABLED",
            str(path),
            timeout_seconds=60,
        )
        body = path.read_bytes()
    if len(body) != int(head.get("ContentLength", -1)) or len(body) > 1024 * 1024:
        raise EphemeralLaunchError("terminal result length is not exact or exceeds 1 MiB")
    terminal = json.loads(body)
    if terminal.get("run_id") != run_id or terminal.get("instance_id") != instance_id:
        raise EphemeralLaunchError("terminal result identity does not match the worker")
    if head.get("ServerSideEncryption") != "AES256" or response.get("ServerSideEncryption") != "AES256":
        raise EphemeralLaunchError("terminal result is not AES256 encrypted")
    proof = {
        "bucket": BUCKET,
        "key": key,
        "version_id": version_id,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "server_side_encryption": "AES256",
        "checksum_sha256_base64": head.get("ChecksumSHA256") or response.get("ChecksumSHA256"),
    }
    return terminal, body, proof


def run(root: Path, spec_path: Path) -> dict[str, Any]:
    spec = validate_spec(root, spec_path)
    initial = _require_initial_zero()
    s3_safety = require_s3_safety()
    _claim(root, spec)
    run_id = str(spec["run_id"])
    key_path = Path(str(spec["ssh_private_key"])).expanduser()
    scoped_key_path = Path(str(spec["scoped_lifecycle_api_key_file"])).expanduser()
    if not key_path.is_file() or not scoped_key_path.is_file():
        raise EphemeralLaunchError("SSH or scoped lifecycle key is absent")
    guardian_id: int | None = None
    worker_id: int | None = None
    guardian_market: dict[str, Any] | None = None
    worker_market: dict[str, Any] | None = None
    started = time.monotonic()
    terminal: dict[str, Any] | None = None
    terminal_body: bytes | None = None
    s3_proof: dict[str, Any] | None = None
    job_released = False
    try:
        guardian_market = select_cheapest_offer(role="guardian", contract=spec["guardian"])
        guardian_create = _create_instance(
            int(guardian_market["selected"]["id"]),
            _create_arguments(role="guardian", spec=spec, run_id=run_id),
        )
        guardian_id = int(guardian_create["new_contract"])
        guardian_row = wait_ssh(
            guardian_id,
            key_path,
            timeout_seconds=int(spec["guardian"]["bootstrap_seconds"]),
        )
        guardian_ssh = _ssh_base(guardian_row, key_path)
        guardian_contract = {
            "schema_version": 1,
            "status": "FROZEN_GUARDIAN",
            "run_id": run_id,
            "worker_label": f"parallax-worker-{run_id}",
            "self_instance_id": guardian_id,
            "scoped_api_key_path": "/run/parallax/scoped-lifecycle-key",
            "self_api_key_path": "/run/parallax/self-instance-key",
            "receipt_path": "/opt/parallax/guardian-receipt.json",
            "armed_path": "/opt/parallax/guardian-armed.json",
            "worker_assignment_path": "/run/parallax/worker-assignment.json",
            "worker_assignment_ack_path": "/opt/parallax/guardian-assignment-ack.json",
            "discovery_seconds": int(spec["guardian"]["discovery_seconds"]),
            "provider_loading_seconds": int(spec["guardian"]["provider_loading_seconds"]),
            "hard_runtime_seconds": int(spec["guardian"]["worker_hard_runtime_seconds"]),
            "poll_seconds": 1,
            "consecutive_absent_observations": 10,
        }
        _stage(
            guardian_ssh,
            remote="/opt/parallax/ephemeral_guardian.py",
            content=(root / "infra/vast/parallax/ephemeral_guardian.py").read_bytes(),
            mode="0700",
        )
        _stage(
            guardian_ssh,
            remote=guardian_contract["scoped_api_key_path"],
            content=scoped_key_path.read_bytes(),
        )
        _stage(
            guardian_ssh,
            remote=guardian_contract["self_api_key_path"],
            content=str(guardian_create["instance_api_key"]).encode("utf-8"),
        )
        _stage(
            guardian_ssh,
            remote="/run/parallax/guardian-contract.json",
            content=json.dumps(guardian_contract, sort_keys=True).encode("utf-8"),
        )
        started_guardian = _run(
            [
                *guardian_ssh,
                "nohup python3 /opt/parallax/ephemeral_guardian.py --contract "
                "/run/parallax/guardian-contract.json >/opt/parallax/guardian.log 2>&1 </dev/null &",
            ],
            check=False,
            timeout_seconds=30,
        )
        if started_guardian.returncode != 0:
            raise EphemeralLaunchError("guardian process did not start")
        _wait_remote_file(guardian_ssh, guardian_contract["armed_path"])
        _require_guardian_live(guardian_id)

        worker_market = select_cheapest_offer(
            role="worker",
            contract=spec["worker"],
            excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
        )
        worker_create = _create_instance(
            int(worker_market["selected"]["id"]),
            _create_arguments(role="worker", spec=spec, run_id=run_id),
        )
        worker_id = int(worker_create["new_contract"])
        _stage_atomic(
            guardian_ssh,
            remote=guardian_contract["worker_assignment_path"],
            content=json.dumps(
                _frozen_worker_assignment(run_id=run_id, worker_id=worker_id),
                sort_keys=True,
            ).encode("utf-8"),
        )
        _wait_remote_json_exact(
            guardian_ssh,
            guardian_contract["worker_assignment_ack_path"],
            _expected_guardian_assignment_ack(run_id=run_id, worker_id=worker_id),
        )
        _require_guardian_live(guardian_id)
        worker_row = wait_ssh(
            worker_id,
            key_path,
            timeout_seconds=int(spec["worker"]["bootstrap_seconds"]),
            health_check=lambda: _require_guardian_live(guardian_id),
        )
        _require_guardian_live(guardian_id)
        worker_ssh = _ssh_base(worker_row, key_path)
        credentials = _aws_credentials()
        result_url = presign_result_put(
            access_key=str(credentials["AccessKeyId"]),
            secret_key=str(credentials["SecretAccessKey"]),
            session_token=credentials.get("SessionToken"),
            bucket=BUCKET,
            key=spec["result"]["key"],
            region=REGION,
            now=datetime.now(timezone.utc),
            expires_seconds=int(spec["result"]["put_url_seconds"]),
        )
        worker_contract = {
            "schema_version": 1,
            "status": "FROZEN",
            "run_id": run_id,
            "stage": spec["job"]["stage"],
            "instance_id": worker_id,
            "runtime_root": "/workspace/parallax",
            "job_output_path": "/workspace/parallax/job-result.json",
            "job_log_path": "/workspace/parallax/job.log",
            "terminal_path": "/workspace/parallax/terminal.json",
            "instance_api_key_path": "/run/parallax/instance-api-key",
            "command": spec["job"]["command"],
            "maximum_job_seconds": int(spec["job"]["maximum_seconds"]),
            "maximum_result_bytes": int(spec["result"]["maximum_bytes"]),
            "result_key": spec["result"]["key"],
            "result_put_url": result_url,
            "post_upload_action": "destroy",
            "upload_failure_action": "stop",
            "environment": spec["job"].get("environment", {}),
        }
        _stage(
            worker_ssh,
            remote="/opt/parallax/ephemeral_s3_worker.py",
            content=(root / "infra/vast/parallax/ephemeral_s3_worker.py").read_bytes(),
            mode="0700",
        )
        _stage(
            worker_ssh,
            remote=worker_contract["instance_api_key_path"],
            content=str(worker_create["instance_api_key"]).encode("utf-8"),
        )
        _stage(
            worker_ssh,
            remote="/run/parallax/worker-contract.json",
            content=json.dumps(worker_contract, sort_keys=True).encode("utf-8"),
        )
        # From this point forward scratch may contain the only result.  Mark
        # release conservatively before SSH so even a connection loss cannot
        # make cleanup destroy unverified bytes.
        job_released = True
        started_worker = _run(
            [
                *worker_ssh,
                _worker_start_command(),
            ],
            check=False,
            timeout_seconds=30,
        )
        if started_worker.returncode != 0:
            raise EphemeralLaunchError("worker process did not start")
        terminal, terminal_body, s3_proof = _wait_result(
            key=spec["result"]["key"],
            run_id=run_id,
            instance_id=worker_id,
            seconds=int(spec["result"]["wait_seconds"]),
        )
    finally:
        cleanup = _finalize_lifecycle(
            worker_id=worker_id,
            guardian_id=guardian_id,
            job_released=job_released,
            s3_durable=s3_proof is not None,
        )
    final = _confirm_account_zero(seconds=45)
    if terminal is None or terminal_body is None or s3_proof is None:
        raise EphemeralLaunchError("terminal S3 result is absent")
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_EPHEMERAL_S3_TERMINAL_REHEARSAL" if terminal.get("status") == "GREEN_JOB_TERMINAL" else "RED_REHEARSAL_JOB",
        "run_id": run_id,
        "provider": "vast.ai",
        "durable_provider": "amazon_s3",
        "scope": "MECHANICAL_ONLY_NO_MODEL_NO_DATA_NO_SCIENCE",
        "initial_account": initial,
        "s3_safety": s3_safety,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "cleanup": cleanup,
        "guardian_instance_id": guardian_id,
        "worker_instance_id": worker_id,
        "terminal": terminal,
        "s3_proof": s3_proof,
        "terminal_account": final,
        "wall_seconds": time.monotonic() - started,
        "automatic_retry": False,
        "promotion_boundary": "PASS_AUTHORIZES_ONLY_A_SEPARATELY_FROZEN_H100_GATE",
    }
    output_relative = Path(str(spec["output_receipt"].get("path", "")))
    if not output_relative.parts or output_relative.is_absolute() or ".." in output_relative.parts:
        raise EphemeralLaunchError("output receipt path is unsafe")
    output = root / output_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, packet)
    terminal_output = output.parent / "terminal.json"
    terminal_output.write_bytes(terminal_body)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root.resolve(), args.spec.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0 if packet["status"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
