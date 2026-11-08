"""Materialize and verify the frozen PARALLAX task bank on cloud storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class TaskCacheError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task_cache_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise TaskCacheError("task cache spec is not frozen")
    if spec.get("provider") != "vast.ai" or spec.get("scope") != "parallax_task_cache_materialization":
        raise TaskCacheError("task cache scope is invalid")
    run_id = str(spec.get("run_id", ""))
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise TaskCacheError("task cache run id is unsafe")
    volume_root = PurePosixPath(str(spec.get("volume_root", "")))
    final_root = PurePosixPath(str(spec.get("final_root", "")))
    staging_root = PurePosixPath(str(spec.get("staging_root", "")))
    receipt_path = PurePosixPath(str(spec.get("receipt_path", "")))
    progress_path = PurePosixPath(str(spec.get("progress_path", "")))
    for name, path in (
        ("volume_root", volume_root),
        ("final_root", final_root),
        ("staging_root", staging_root),
        ("receipt_path", receipt_path),
        ("progress_path", progress_path),
    ):
        if not path.is_absolute() or ".." in path.parts:
            raise TaskCacheError(f"{name} must be absolute and safe")
    if not final_root.is_relative_to(volume_root) or not staging_root.is_relative_to(volume_root):
        raise TaskCacheError("task cache roots escape the persistent volume")
    if not receipt_path.is_relative_to(volume_root) or not progress_path.is_relative_to(volume_root):
        raise TaskCacheError("task cache receipt or progress escapes the persistent volume")
    if receipt_path == progress_path:
        raise TaskCacheError("task cache receipt and progress paths must be distinct")
    if final_root == staging_root or staging_root.parent != final_root.parent:
        raise TaskCacheError("task cache staging must be a distinct atomic sibling")
    bindings = spec.get("source_bindings")
    if not isinstance(bindings, list) or len(bindings) < 4:
        raise TaskCacheError("task cache source bindings are incomplete")
    seen: set[str] = set()
    for row in bindings:
        path = PurePosixPath(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        if not path.is_absolute() or ".." in path.parts or path.as_posix() in seen:
            raise TaskCacheError("task cache source path is unsafe or duplicated")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise TaskCacheError("task cache source digest is malformed")
        seen.add(path.as_posix())
    for field in ("data_config", "identity_data_config", "task_config"):
        path = PurePosixPath(str(spec.get(field, "")))
        if not path.is_absolute() or ".." in path.parts or path.as_posix() not in seen:
            raise TaskCacheError(f"task cache {field} is not source-bound")
    hf_cache_root = PurePosixPath(str(spec.get("hf_cache_root", "")))
    if not hf_cache_root.is_absolute() or not hf_cache_root.is_relative_to(volume_root):
        raise TaskCacheError("task cache Hugging Face cache escapes the persistent volume")
    expected = spec.get("expected", {})
    for key in ("content_sha256", "audit_sha256", "items_jsonl_sha256"):
        digest = str(expected.get(key, ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise TaskCacheError(f"task cache expected {key} is malformed")
    for key in ("semantic_items_sha256", "decoded_pixels_sha256"):
        if key in expected:
            digest = str(expected[key])
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise TaskCacheError(f"task cache expected {key} is malformed")
    if int(expected.get("total_items", 0)) != 4096 or int(expected.get("total_images", 0)) != 2560:
        raise TaskCacheError("task cache expected cardinalities are not exact")
    metadata = spec.get("dataset_metadata")
    if not isinstance(metadata, list) or len(metadata) != 2:
        raise TaskCacheError("task cache dataset metadata gate must bind two repositories")
    for row in metadata:
        repo = str(row.get("repo", ""))
        revision = str(row.get("revision", ""))
        if repo.count("/") != 1 or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise TaskCacheError("task cache dataset metadata binding is malformed")
    if not 50_000_000 <= int(spec.get("maximum_payload_bytes", 0)) <= 100_000_000:
        raise TaskCacheError("task cache payload ceiling is outside the frozen range")
    overlay = spec.get("python_overlay")
    if overlay is not None:
        if not isinstance(overlay, dict):
            raise TaskCacheError("task cache Python overlay is malformed")
        root = PurePosixPath(str(overlay.get("root", "")))
        if not root.is_absolute() or ".." in root.parts:
            raise TaskCacheError("task cache Python overlay root is unsafe")
        if overlay.get("packages") != ["pillow==12.1.1"]:
            raise TaskCacheError("task cache Python overlay is not the frozen encoder")
        if overlay.get("expected_pillow_version") != "12.1.1":
            raise TaskCacheError("task cache Python overlay version is not exact")
    return spec


def _run(arguments: list[str], *, environment: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        arguments,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "arguments": arguments,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "elapsed_seconds": time.monotonic() - started,
    }


def _payload_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _semantic_items_sha256(items_path: Path) -> str:
    digest = hashlib.sha256()
    with items_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row.pop("image_sha256", None)
            row.pop("image_bytes", None)
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _decoded_pixels_sha256(root: Path, items_path: Path) -> str:
    from PIL import Image

    digest = hashlib.sha256()
    with items_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("image") is None:
                continue
            with Image.open(root / row["image"]) as image:
                image.load()
                digest.update(row["item_id"].encode())
                digest.update(b"\0" + image.mode.encode() + b"\0")
                digest.update(f"{image.width}x{image.height}".encode() + b"\0")
                digest.update(image.tobytes())
    return digest.hexdigest()


def _verify_bank(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    content_path = root / "manifests/content.json"
    audit_path = root / "manifests/audit.json"
    items_path = root / "manifests/items.jsonl"
    expected = spec["expected"]
    actual: dict[str, Any] = {
        "content_sha256": _sha256(content_path),
        "audit_sha256": _sha256(audit_path),
        "items_jsonl_sha256": _sha256(items_path),
    }
    try:
        actual["semantic_items_sha256"] = _semantic_items_sha256(items_path)
        actual["decoded_pixels_sha256"] = _decoded_pixels_sha256(root, items_path)
    except (OSError, ValueError, KeyError) as error:
        actual["semantic_items_sha256"] = "UNREADABLE"
        actual["decoded_pixels_sha256"] = "UNREADABLE"
        actual["derived_manifest_error"] = f"{type(error).__name__}: {error}"[:500]
    content = json.loads(content_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload_bytes = _payload_bytes(root)
    checks = {
        "content_sha256": actual["content_sha256"] == expected["content_sha256"],
        "audit_sha256": actual["audit_sha256"] == expected["audit_sha256"],
        "items_jsonl_sha256": actual["items_jsonl_sha256"] == expected["items_jsonl_sha256"],
        "content_green": content.get("status") == "GREEN_MATERIALIZED",
        "audit_green": audit.get("status") == "GREEN" and not audit.get("errors"),
        "total_items": int(content.get("total_items", 0)) == int(expected["total_items"]),
        "total_images": int(content.get("total_images", 0)) == int(expected["total_images"]),
        "payload_ceiling": payload_bytes <= int(spec["maximum_payload_bytes"]),
        "split_image_firewall": audit.get("cross_split_image_hashes") == 0,
        "split_source_firewall": audit.get("cross_split_source_identities") == 0,
        "split_semantic_firewall": audit.get("cross_split_nonce_or_interface_semantics") == 0,
    }
    if "total_image_bytes" in expected:
        checks["total_image_bytes"] = int(content.get("total_image_bytes", 0)) == int(
            expected["total_image_bytes"]
        )
    for name in ("semantic_items_sha256", "decoded_pixels_sha256"):
        if name in expected:
            checks[name] = actual[name] == expected[name]
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed": failed,
        "actual": actual,
        "payload_bytes": payload_bytes,
        "total_items": content.get("total_items"),
        "total_images": content.get("total_images"),
        "total_image_bytes": content.get("total_image_bytes"),
    }


def materialize(spec: dict[str, Any]) -> dict[str, Any]:
    validate_task_cache_spec(spec)
    started = time.monotonic()
    source_checks: list[dict[str, Any]] = []
    for binding in spec["source_bindings"]:
        path = Path(binding["path"])
        actual = _sha256(path)
        match = actual == binding["sha256"]
        source_checks.append({"path": str(path), "sha256": actual, "match": match})
        if not match:
            raise TaskCacheError(f"source binding mismatch: {path}")
    final_root = Path(spec["final_root"])
    staging_root = Path(spec["staging_root"])
    if staging_root.exists():
        raise TaskCacheError("task cache staging path already exists")
    commands: list[dict[str, Any]] = []
    progress_path = Path(spec["progress_path"])
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "status": "SOURCE_BINDINGS_GREEN",
            "run_id": spec["run_id"],
            "source_checks": source_checks,
            "commands": commands,
        },
    )
    reused_existing = final_root.exists()
    if not reused_existing:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": "/opt/parallax-lifecycle:/opt/parallax",
                "HF_HOME": spec["hf_cache_root"],
                "HF_DATASETS_CACHE": str(Path(spec["hf_cache_root"]) / "datasets"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        overlay = spec.get("python_overlay")
        if overlay is not None:
            overlay_root = Path(overlay["root"])
            shutil.rmtree(overlay_root, ignore_errors=True)
            install_result = _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--target",
                    str(overlay_root),
                    *overlay["packages"],
                ],
                environment=environment,
                timeout_seconds=120,
            )
            commands.append(install_result)
            _atomic_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "PYTHON_OVERLAY_INSTALL_FINISHED",
                    "run_id": spec["run_id"],
                    "source_checks": source_checks,
                    "commands": commands,
                },
            )
            if install_result["returncode"] != 0:
                detail = install_result["stderr_tail"] or install_result["stdout_tail"]
                raise TaskCacheError("Python overlay install failed: " + detail[-500:])
            environment["PYTHONPATH"] = f"{overlay_root}:/opt/parallax-lifecycle:/opt/parallax"
            version_result = _run(
                [
                    sys.executable,
                    "-c",
                    "import PIL,sys; print(PIL.__version__); sys.exit(0 if PIL.__version__ == sys.argv[1] else 2)",
                    overlay["expected_pillow_version"],
                ],
                environment=environment,
                timeout_seconds=30,
            )
            commands.append(version_result)
            _atomic_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "PYTHON_OVERLAY_VERSION_FINISHED",
                    "run_id": spec["run_id"],
                    "source_checks": source_checks,
                    "commands": commands,
                },
            )
            if version_result["returncode"] != 0:
                raise TaskCacheError("Python overlay version gate failed")
        metadata_script = (
            "import json,sys;from huggingface_hub import HfApi;"
            "rows=json.loads(sys.argv[1]);api=HfApi();"
            "out=[{'repo':r['repo'],'expected':r['revision'],'actual':api.dataset_info(r['repo'],revision=r['revision']).sha} for r in rows];"
            "assert all(x['actual']==x['expected'] for x in out),out;print(json.dumps(out,sort_keys=True))"
        )
        metadata_result = _run(
            [sys.executable, "-c", metadata_script, json.dumps(spec["dataset_metadata"], sort_keys=True)],
            environment=environment,
            timeout_seconds=60,
        )
        commands.append(metadata_result)
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "DATASET_METADATA_PREFLIGHT_FINISHED",
                "run_id": spec["run_id"],
                "source_checks": source_checks,
                "commands": commands,
            },
        )
        if metadata_result["returncode"] != 0:
            detail = metadata_result["stderr_tail"] or metadata_result["stdout_tail"]
            raise TaskCacheError("dataset metadata preflight failed: " + detail[-500:])
        materialize_result = _run(
            [
                sys.executable,
                "-m",
                "infra.vast.parallax.materialize_tasks",
                "--data-config",
                spec["data_config"],
                "--identity-data-config",
                spec["identity_data_config"],
                "--task-config",
                spec["task_config"],
                "--root",
                str(staging_root),
            ],
            environment=environment,
            timeout_seconds=int(spec["materialize_timeout_seconds"]),
        )
        commands.append(materialize_result)
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "MATERIALIZE_COMMAND_FINISHED",
                "run_id": spec["run_id"],
                "source_checks": source_checks,
                "commands": commands,
            },
        )
        if materialize_result["returncode"] != 0:
            detail = materialize_result["stderr_tail"] or materialize_result["stdout_tail"]
            raise TaskCacheError("task materialization command failed: " + detail[-500:])
        audit_result = _run(
            [
                sys.executable,
                "-m",
                "infra.vast.parallax.audit_task_bank",
                "--root",
                str(staging_root),
                "--output",
                str(staging_root / "manifests/audit.json"),
            ],
            environment=environment,
            timeout_seconds=int(spec["audit_timeout_seconds"]),
        )
        commands.append(audit_result)
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "AUDIT_COMMAND_FINISHED",
                "run_id": spec["run_id"],
                "source_checks": source_checks,
                "commands": commands,
            },
        )
        if audit_result["returncode"] != 0:
            detail = audit_result["stderr_tail"] or audit_result["stdout_tail"]
            raise TaskCacheError("task bank audit command failed: " + detail[-500:])
        verification = _verify_bank(staging_root, spec)
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "EXACT_VERIFICATION_FINISHED",
                "run_id": spec["run_id"],
                "source_checks": source_checks,
                "commands": commands,
                "verification": verification,
            },
        )
        if verification["status"] != "GREEN":
            raise TaskCacheError("task bank differs from the frozen exact manifests")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, final_root)
    else:
        verification = _verify_bank(final_root, spec)
        if verification["status"] != "GREEN":
            raise TaskCacheError("existing task bank differs from the frozen exact manifests")
    return {
        "schema_version": 1,
        "status": "GREEN",
        "run_id": spec["run_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checks": source_checks,
        "commands": commands,
        "verification": verification,
        "final_root": str(final_root),
        "reused_existing": reused_existing,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_work": False,
        "model_reads": 0,
        "network_policy": "pinned_huggingface_datasets_only_if_cache_absent",
        "progress_path": str(progress_path),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    receipt_path: Path | None = None
    spec: dict[str, Any] | None = None
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        receipt_path = Path(spec["receipt_path"])
        result = materialize(spec)
        exit_code = 0
    except Exception as error:
        staging_cleaned = False
        if spec is not None:
            staging = Path(str(spec.get("staging_root", "")))
            final = Path(str(spec.get("final_root", "")))
            if staging.is_absolute() and staging.exists() and not final.exists():
                shutil.rmtree(staging)
                staging_cleaned = True
        progress = None
        if spec is not None:
            progress_path = Path(str(spec.get("progress_path", "")))
            if progress_path.is_file() and progress_path.stat().st_size <= 2_000_000:
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    progress = {"status": "UNREADABLE"}
        result = {
            "schema_version": 1,
            "status": "RED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {"type": type(error).__name__, "message": str(error)[:500]},
            "gpu_work": False,
            "model_reads": 0,
            "staging_cleaned": staging_cleaned,
            "progress": progress,
        }
        exit_code = 2
        if receipt_path is None:
            raise
    _atomic_json(receipt_path, result)
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
