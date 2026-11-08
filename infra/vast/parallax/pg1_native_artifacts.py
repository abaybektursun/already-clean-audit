"""Authoritative PG1 payload inventory and deterministic result packaging."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .pg1_execution_capsule import CapsuleError, content_sha256


PAYLOAD_SCHEMA = "pg1-native-payload-manifest/v1"
FINAL_RECEIPT_SCHEMA = "pg1-native-final-durability-receipt/v1"
TEARDOWN_SCHEMA = "pg1-native-verified-teardown-evidence/v1"


def _within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise CapsuleError(f"{label} escaped canonical output root") from error
    return resolved


def _checkpoint_rows(control: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in control.get("codebooks", []):
        if not isinstance(cell, dict) or not isinstance(cell.get("snapshots"), list):
            raise CapsuleError("control checkpoint inventory is incomplete")
        for snapshot in cell["snapshots"]:
            checkpoint = snapshot.get("checkpoint") if isinstance(snapshot, dict) else None
            if not isinstance(checkpoint, dict):
                raise CapsuleError("control checkpoint identity is absent")
            path = _within(Path(str(checkpoint.get("path"))), output_dir, label="checkpoint")
            if path.suffix != ".pt":
                raise CapsuleError("control checkpoint suffix changed")
            rows.append(
                {
                    "kind": "checkpoint_snapshot",
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_dir.resolve())),
                    "bytes": checkpoint.get("bytes"),
                    "sha256": checkpoint.get("sha256"),
                }
            )
    if not rows:
        raise CapsuleError("control produced no required checkpoints")
    return rows


def build_payload_manifest(
    *, capsule: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    paths = capsule["contract"]["canonical_paths"]
    output_dir = Path(paths["output_dir"]).resolve()
    manifest_path = Path(paths["artifact_manifest"])
    if manifest_path.exists():
        raise CapsuleError("refusing to reuse payload artifact manifest")
    interface_path = output_dir / "interface-control.json"
    rows = [
        {
            "kind": "interface_control_result",
            "path": str(interface_path),
            "relative_path": "interface-control.json",
            "bytes": interface_path.stat().st_size if interface_path.is_file() else -1,
            "sha256": sha256_file(interface_path) if interface_path.is_file() else "",
        },
        *_checkpoint_rows(control, output_dir),
    ]
    expected_paths = {Path(row["path"]).resolve() for row in rows}
    observed_paths = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
    if (
        len(expected_paths) != len(rows)
        or observed_paths != expected_paths
        or any(
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
            for path, row in ((Path(row["path"]), row) for row in rows)
        )
    ):
        raise CapsuleError("payload artifact inventory is missing, extra, or changed")
    manifest = {
        "schema": PAYLOAD_SCHEMA,
        "status": "COMPLETE_EXACT_RESULT_AND_CHECKPOINT_INVENTORY",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "artifact_key": capsule["contract"]["attempt_identity"]["artifact_key"],
        "manifest_key": capsule["contract"]["attempt_identity"]["artifact_manifest_key"],
        "files": sorted(rows, key=lambda row: row["relative_path"]),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "unexpected_output_files": 0,
        "confirmation_items_read": 0,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def validate_payload_manifest(
    *, capsule: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    paths = capsule["contract"]["canonical_paths"]
    manifest_path = Path(paths["artifact_manifest"])
    output_dir = Path(paths["output_dir"]).resolve()
    if (
        manifest.get("schema") != PAYLOAD_SCHEMA
        or manifest.get("status") != "COMPLETE_EXACT_RESULT_AND_CHECKPOINT_INVENTORY"
        or manifest.get("capsule_sha256") != capsule["content_address"]["digest"]
        or manifest.get("attempt_run_id")
        != capsule["contract"]["attempt_identity"]["attempt_run_id"]
        or manifest.get("artifact_key")
        != capsule["contract"]["attempt_identity"]["artifact_key"]
        or manifest.get("manifest_key")
        != capsule["contract"]["attempt_identity"]["artifact_manifest_key"]
        or not isinstance(manifest.get("files"), list)
        or manifest.get("file_count") != len(manifest["files"])
        or manifest.get("unexpected_output_files") != 0
        or manifest.get("confirmation_items_read") != 0
    ):
        raise CapsuleError("payload artifact manifest identity changed")
    expected: set[Path] = set()
    checkpoint_count = 0
    for row in manifest["files"]:
        if not isinstance(row, dict) or set(row) != {
            "kind", "path", "relative_path", "bytes", "sha256"
        }:
            raise CapsuleError("payload artifact manifest row changed")
        path = _within(Path(row["path"]), output_dir, label="payload")
        if (
            path in expected
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise CapsuleError("payload artifact body changed")
        expected.add(path)
        checkpoint_count += row["kind"] == "checkpoint_snapshot"
    observed = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
    if (
        observed != expected
        or checkpoint_count < capsule["contract"]["artifact_contract"]["minimum_checkpoints"]
        or not manifest_path.is_file()
    ):
        raise CapsuleError("payload artifact inventory is incomplete")
    return manifest


def _tar_add_path(archive: tarfile.TarFile, name: str, path: Path) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = path.stat().st_size
    info.mode = 0o400
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def package_required_artifacts(
    *, capsule: dict[str, Any], terminal: dict[str, Any]
) -> dict[str, Any]:
    paths = capsule["contract"]["canonical_paths"]
    manifest_path = Path(paths["artifact_manifest"])
    bundle_path = Path(paths["artifact_bundle"])
    if bundle_path.exists() or not manifest_path.is_file():
        raise CapsuleError("artifact bundle path or payload manifest state changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_payload_manifest(capsule=capsule, manifest=manifest)
    terminal_binding = terminal.get("job", {}).get("result", {}).get("payload", {}).get(
        "artifact_manifest"
    )
    expected_terminal_binding = {
        "run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "control_path": str(Path(paths["output_dir"]) / "interface-control.json"),
        "control_sha256": sha256_file(
            Path(paths["output_dir"]) / "interface-control.json"
        ),
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "content_sha256": content_sha256(manifest),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "confirmation_items_read": 0,
        "promotion_eligible": False,
    }
    if terminal_binding != expected_terminal_binding:
        raise CapsuleError("raw terminal does not bind the payload manifest")
    sources: list[tuple[str, Path]] = [
        (f"payload/{row['relative_path']}", Path(row["path"]))
        for row in manifest["files"]
    ]
    sources.append(("evidence/payload_manifest.json", manifest_path))
    singletons = capsule["contract"]["artifact_contract"]["required_singletons"]
    for label, source in singletons.items():
        path = Path(source)
        if label == "interface_control":
            continue
        sources.append((f"evidence/{label}{path.suffix}", path))
    if len({name for name, _ in sources}) != len(sources):
        raise CapsuleError("artifact archive name collision")
    for _name, path in sources:
        if path.is_symlink() or not path.is_file():
            raise CapsuleError("required artifact body is absent")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "x", format=tarfile.PAX_FORMAT) as archive:
        for name, path in sorted(sources):
            _tar_add_path(archive, name, path)
    expected = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sources
    }
    with tarfile.open(bundle_path, "r") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(members) != set(expected):
            raise CapsuleError("artifact bundle member inventory changed")
        for name, identity in expected.items():
            handle = archive.extractfile(members[name])
            if handle is None:
                raise CapsuleError("artifact bundle member is unreadable")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
            if size != identity["bytes"] or digest.hexdigest() != identity["sha256"]:
                raise CapsuleError("artifact bundle member body changed")
    return {
        "schema": "pg1-native-artifact-bundle-evidence/v1",
        "status": "COMPLETE_REQUIRED_BODIES_PACKAGED_AND_READ_BACK",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "artifact_key": capsule["contract"]["attempt_identity"]["artifact_key"],
        "path": str(bundle_path),
        "sha256": sha256_file(bundle_path),
        "bytes": bundle_path.stat().st_size,
        "member_count": len(expected),
        "payload_manifest_sha256": sha256_file(manifest_path),
    }


def final_durability_receipt(
    *,
    capsule: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    verifications: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "raw_terminal", "scratch_pointer", "critic_audit",
        "artifact_bundle", "artifact_manifest",
    }
    if set(receipts) != required or set(verifications) != required:
        raise CapsuleError("final durability inventory is incomplete")
    for kind in required:
        if (
            receipts[kind].get("kind") != kind
            or verifications[kind].get("kind") != kind
            or verifications[kind].get("status")
            != "VERIFIED_EXACT_DOWNLOADED_DURABLE_OBJECT"
        ):
            raise CapsuleError("final durability evidence changed")
    return {
        "schema": FINAL_RECEIPT_SCHEMA,
        "status": "ALL_REQUIRED_ARTIFACTS_DOWNLOADED_AND_VERIFIED_BEFORE_TEARDOWN",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "payload_manifest_sha256": sha256_file(
            Path(capsule["contract"]["canonical_paths"]["artifact_manifest"])
        ),
        "receipt_sha256": {
            kind: content_sha256(receipts[kind]) for kind in sorted(required)
        },
        "verification_sha256": {
            kind: content_sha256(verifications[kind]) for kind in sorted(required)
        },
        "teardown_authorized": True,
    }


def verified_teardown_evidence(
    *,
    capsule: dict[str, Any],
    final_receipt: dict[str, Any],
    receipt_durability: dict[str, Any],
    receipt_verification: dict[str, Any],
) -> dict[str, Any]:
    if (
        final_receipt.get("schema") != FINAL_RECEIPT_SCHEMA
        or final_receipt.get("teardown_authorized") is not True
        or receipt_durability.get("kind") != "artifact_receipts"
        or receipt_verification.get("kind") != "artifact_receipts"
        or receipt_verification.get("status")
        != "VERIFIED_EXACT_DOWNLOADED_DURABLE_OBJECT"
        or receipt_verification.get("receipt_sha256")
        != content_sha256(receipt_durability)
    ):
        raise CapsuleError("final durability receipt was not independently persisted")
    return {
        "schema": TEARDOWN_SCHEMA,
        "status": "FINAL_DURABILITY_RECEIPT_DOWNLOADED_VERIFIED_FOR_TEARDOWN",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "final_receipt_sha256": content_sha256(final_receipt),
        "receipt_durability_sha256": content_sha256(receipt_durability),
        "receipt_verification_sha256": content_sha256(receipt_verification),
        "teardown_authorized": True,
    }
