"""Separate stdlib deterministic analogue of the PG1 production deployment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .pg1_execution_capsule import (
    CONSUMER_ORDER,
    CapsuleError,
    canonical_bytes,
    content_sha256,
    load_capsule,
    validate_capsule,
)


MECHANICAL_SCHEMA = "pg1-native-mechanical-rehearsal-contract/v3"
MECHANICAL_AUTH_SCHEMA = "pg1-native-mechanical-authorization/v3"
MECHANICAL_RECEIPT_SCHEMA = "pg1-native-mechanical-consumption-receipt/v3"
FAILURES = (
    "none", "pre_setup", "model_body_mismatch", "tokenizer_asset_mismatch",
    "task_manifest_mismatch", "missing_image_body", "extra_image_body",
    "wrong_image_body", "artifact_missing_body", "artifact_extra_body",
    "artifact_wrong_body", "terminal_upload", "critic_execution", "audit_upload",
    "verification", "primary_stop_failure", "fallback_stop_failure",
    "guardian_marker_failure", "guardian_race", "guardian_replay",
    "guardian_timeout", "provider_nonzero", "provider_timeout",
    "guardian_transient_key", "guardian_transient_show_timeout",
    "guardian_transient_nonzero", "guardian_transient_malformed",
    "guardian_transient_parse", "guardian_transient_rate_limit",
    "guardian_deadline_fail_closed", "guardian_repeated_absence",
    "guardian_wrong_row", "bootstrap_import_failure",
)


def _fixture_bytes(attempt_id: str) -> dict[str, bytes]:
    visual_train = f"tiny-visual-train:{attempt_id}\n".encode()
    visual_eval = f"tiny-visual-eval:{attempt_id}\n".encode()
    rows = [
        {
            "item_id": f"selected-train-visual-{attempt_id}",
            "image": "images/train-visual.bin",
            "image_bytes": len(visual_train),
            "image_sha256": hashlib.sha256(visual_train).hexdigest(),
            "split": "train", "target": "visual", "task_id": "tiny-task",
        },
        {
            "item_id": f"selected-train-text-{attempt_id}",
            "image": None, "split": "train", "target": "text",
            "task_id": "tiny-task", "text": "manifest-bound interface item",
        },
        {
            "item_id": f"selected-eval-visual-{attempt_id}",
            "image": "images/eval-visual.bin",
            "image_bytes": len(visual_eval),
            "image_sha256": hashlib.sha256(visual_eval).hexdigest(),
            "split": "discovery_eval", "target": "visual", "task_id": "tiny-task",
        },
        {
            "item_id": f"selected-eval-text-{attempt_id}",
            "image": None, "split": "discovery_eval", "target": "text",
            "task_id": "tiny-task", "text": "manifest-bound text item",
        },
        {
            "item_id": f"unselected-interface-{attempt_id}",
            "image": None, "split": "train", "target": "interface",
            "task_id": "other-interface-task", "text": "valid image-less mixed row",
        },
    ]
    manifest = b"".join(canonical_bytes(row) for row in rows)
    return {
        "model.bin": f"tiny-model:{attempt_id}\n".encode(),
        "tokenizer.json": canonical_bytes({"attempt": attempt_id, "kind": "tokenizer"}),
        "tokenizer.model": f"tiny-tokenizer-model:{attempt_id}\n".encode(),
        "preprocessor_config.json": b'{"size":224}\n',
        "train-visual.bin": visual_train,
        "eval-visual.bin": visual_eval,
        "task-manifest.jsonl": manifest,
    }


def compile_mechanical_contract(capsule: dict[str, Any]) -> dict[str, Any]:
    validate_capsule(capsule)
    attempt_id = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    fixtures = _fixture_bytes(attempt_id)
    contract = {
        "schema": MECHANICAL_SCHEMA,
        "production_capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": attempt_id,
        "owner_order": list(CONSUMER_ORDER),
        "selected_task_id": "tiny-task",
        "selection": {
            "train_split": "train", "train_per_target": 1,
            "discovery_split": "discovery_eval", "discovery_per_target": 1,
        },
        "fixture_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in sorted(fixtures.items())
        },
        "external_commands": 0,
        "h100_instances_created": 0,
        "deployment_analogue": {
            "production_argv": capsule["contract"]["entrypoints"]["production_argv"],
            "bootstrap_argv": capsule["contract"]["entrypoints"]["bootstrap_argv"],
            "native_launcher_argv": capsule["contract"]["entrypoints"][
                "native_launcher_argv"
            ],
            "fixed_layout_modes": capsule["contract"]["production_runtime"][
                "deployment"
            ]["fixed_layout_modes"],
            "fixed_mount_modes": capsule["contract"]["production_runtime"][
                "deployment"
            ]["fixed_mount_modes"],
            "runtime_wiring": capsule["contract"]["runtime_wiring"],
        },
    }
    return {
        "schema_version": 3,
        "status": "COMPILED_MECHANICAL_ONLY_NO_PRODUCTION_AUTHORITY",
        "content_address": {"algorithm": "sha256", "digest": content_sha256(contract)},
        "contract": contract,
    }


def validate_mechanical_contract(
    mechanical: dict[str, Any], capsule: dict[str, Any]
) -> dict[str, Any]:
    if mechanical != compile_mechanical_contract(capsule):
        raise CapsuleError("mechanical contract identity or fixtures changed")
    return mechanical


def load_mechanical_contract(path: Path, capsule: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_mechanical_contract(value, capsule)


class DeterministicAuthorizationStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @staticmethod
    def _exclusive(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())

    def issue(
        self, capsule: dict[str, Any], mechanical: dict[str, Any]
    ) -> dict[str, Any]:
        contract = {
            "schema": MECHANICAL_AUTH_SCHEMA,
            "mechanical_contract_sha256": mechanical["content_address"]["digest"],
            "production_capsule_sha256": capsule["content_address"]["digest"],
            "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
            "instance_id": 100000 + int(capsule["content_address"]["digest"][:8], 16) % 800000,
            "store_type": "LOCAL_O_EXCL_REHEARSAL_ONLY",
        }
        value = {
            "schema_version": 3,
            "status": "MECHANICAL_AUTHORIZED_NO_PRODUCTION_AUTHORITY",
            "content_address": {"algorithm": "sha256", "digest": content_sha256(contract)},
            "contract": contract,
        }
        self._exclusive(self.root / "issued" / f"{contract['attempt_run_id']}.json", value)
        return value

    def consume(
        self,
        capsule: dict[str, Any],
        mechanical: dict[str, Any],
        authorization: dict[str, Any],
    ) -> dict[str, Any]:
        if authorization["contract"].get("schema") != MECHANICAL_AUTH_SCHEMA:
            raise CapsuleError("mechanical authorization schema changed")
        marker = {
            "schema": "pg1-native-mechanical-consumption-marker/v3",
            "mechanical_contract_sha256": mechanical["content_address"]["digest"],
            "authorization_sha256": authorization["content_address"]["digest"],
        }
        relative = Path("consumed") / f"{capsule['content_address']['digest']}.json"
        self._exclusive(self.root / relative, marker)
        return {
            "schema": MECHANICAL_RECEIPT_SCHEMA,
            "status": "MECHANICAL_CONSUMED_EXACTLY_ONCE",
            "authorization_sha256": authorization["content_address"]["digest"],
            "mechanical_contract_sha256": mechanical["content_address"]["digest"],
            "marker_sha256": content_sha256(marker),
            "local_key": str(relative),
        }


def _write_fixtures(
    root: Path, contract: dict[str, Any], *, failure: str
) -> dict[str, Path]:
    values = _fixture_bytes(contract["attempt_run_id"])
    paths: dict[str, Path] = {}
    for name, data in values.items():
        if name in {"train-visual.bin", "eval-visual.bin"}:
            target = root / "fixture" / "tasks" / "images" / name
        elif name == "task-manifest.jsonl":
            target = root / "fixture" / "tasks" / "manifest.jsonl"
        else:
            target = root / "fixture" / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        paths[name] = target
    corrupt = {
        "model_body_mismatch": "model.bin",
        "tokenizer_asset_mismatch": "tokenizer.json",
        "task_manifest_mismatch": "task-manifest.jsonl",
        "wrong_image_body": "train-visual.bin",
    }.get(failure)
    if corrupt:
        paths[corrupt].write_bytes(paths[corrupt].read_bytes() + b"PLANTED_DRIFT")
    if failure == "missing_image_body":
        paths["eval-visual.bin"].unlink()
    if failure == "extra_image_body":
        (root / "fixture" / "tasks" / "images" / "extra-selected.bin").write_bytes(b"extra")
    return paths


def _select(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["task_id"] == "tiny-task" and row["split"] == split:
            grouped.setdefault(row["target"], []).append(row)
    selected = [
        sorted(values, key=lambda row: row["item_id"])[0]
        for _target, values in sorted(grouped.items())
    ]
    return sorted(selected, key=lambda row: (row["target"], row["item_id"]))


def _verify_fixture_hydration(
    mechanical: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    expected = mechanical["contract"]["fixture_sha256"]
    for name, digest in expected.items():
        if not paths[name].is_file() or sha256_file(paths[name]) != digest:
            raise CapsuleError(f"mechanical hydration mismatch: {name}")
    rows = [json.loads(line) for line in paths["task-manifest.jsonl"].read_text().splitlines()]
    for row in rows:
        if row.get("image") is None:
            if "image_sha256" in row or "image_bytes" in row:
                raise CapsuleError("mechanical image-less item has extra image identity")
        elif not all(field in row for field in ("image", "image_sha256", "image_bytes")):
            raise CapsuleError("mechanical visual item image identity is incomplete")
    selected = [*_select(rows, "train"), *_select(rows, "discovery_eval")]
    visual = [row for row in selected if row["image"] is not None]
    imageless = [row for row in selected if row["image"] is None]
    image_root = paths["task-manifest.jsonl"].parent / "images"
    if {path.name for path in image_root.iterdir() if path.is_file()} != {
        "train-visual.bin", "eval-visual.bin"
    }:
        raise CapsuleError("mechanical selected image body inventory has extras")
    for row in visual:
        target = paths[Path(row["image"]).name]
        if target.stat().st_size != row["image_bytes"] or sha256_file(target) != row["image_sha256"]:
            raise CapsuleError("mechanical selected image body changed")
    return {
        "schema": "pg1-native-mechanical-hydration-evidence/v3",
        "status": "GREEN_MIXED_ROWS_EXACT_SELECTION_AND_BODIES",
        "manifest_items": len(rows),
        "selected_item_ids": [row["item_id"] for row in selected],
        "selected_visual_bodies": len(visual),
        "selected_imageless_items": len(imageless),
        "unselected_imageless_items": 1,
        "science_started": False,
    }


class DeterministicLifecycleTransport:
    def __init__(self, root: Path, failure: str) -> None:
        self.root = root
        self.failure = failure
        self.events_path = root / "events.jsonl"
        self.verified: set[str] = set()

    def event(self, name: str, **evidence: Any) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": name, **evidence}, sort_keys=True) + "\n")

    def durable_sink(self, *, kind: str, source_path: Path, object_key: str) -> dict[str, Any]:
        self.event(f"{kind}_upload_started")
        if (self.failure == "terminal_upload" and kind == "raw_terminal") or (
            self.failure == "audit_upload" and kind == "critic_audit"
        ):
            raise CapsuleError(f"planted {self.failure} failure")
        target = self.root / "durable" / object_key
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("mechanical conditional durability replay")
        shutil.copyfile(source_path, target)
        digest = sha256_file(source_path)
        receipt = {
            "schema": "pg1-native-mechanical-durability-receipt/v3",
            "kind": kind, "object_key": object_key, "source_sha256": digest,
            "checksum_sha256": base64.b64encode(bytes.fromhex(digest)).decode("ascii"),
            "content_length": source_path.stat().st_size,
            "content_type": "application/x-tar" if source_path.suffix == ".tar" else "application/json",
            "server_side_encryption": "AES256",
            "metadata": {"object-kind": kind, "source-sha256": digest},
            "version_id": f"local-{digest[:24]}", "etag": digest,
            "conditional_create": True,
        }
        self.event(f"{kind}_durable")
        return receipt

    def verify(self, *, kind: str, receipt: dict[str, Any]) -> None:
        if self.failure == "verification" and kind == "critic_audit":
            raise CapsuleError("planted downloaded verification failure")
        target = self.root / "durable" / receipt["object_key"]
        if (
            not target.is_file()
            or target.stat().st_size != receipt["content_length"]
            or sha256_file(target) != receipt["source_sha256"]
            or receipt["server_side_encryption"] != "AES256"
            or receipt["conditional_create"] is not True
        ):
            raise CapsuleError(f"mechanical {kind} exact durability changed")
        self.verified.add(kind)
        self.event(f"{kind}_downloaded_verified")

    def stop(self, *, failure: str) -> dict[str, Any]:
        pointer = self.root / "scratch" / "failure-stop-pointer.json"
        atomic_json(pointer, {"status": "ARTIFACT_POINTERS_PRESERVED", "failure": failure})
        self.event("failure_pointer_preserved")
        mode = self.failure
        if mode in {"provider_timeout", "guardian_transient_show_timeout"}:
            self.event("primary_provider_timeout_rejected")
        elif mode in {"provider_nonzero", "guardian_transient_nonzero"}:
            self.event("primary_provider_nonzero_rejected")
        elif mode in {
            "primary_stop_failure", "fallback_stop_failure", "guardian_marker_failure",
            "guardian_race", "guardian_replay", "guardian_timeout",
            "guardian_transient_key", "guardian_transient_malformed",
            "guardian_transient_parse", "guardian_transient_rate_limit",
            "guardian_deadline_fail_closed", "guardian_repeated_absence",
            "guardian_wrong_row", "bootstrap_import_failure",
        }:
            self.event("primary_stop_failed")
        else:
            self.event("primary_stop_semantics_verified")
            return {"status": "PRIMARY_STOP_VERIFIED_TERMINAL"}
        if mode in {"fallback_stop_failure", "guardian_marker_failure"}:
            self.event("guardian_marker_failed")
            raise CapsuleError("mechanical guardian marker failed")
        marker = self.root / "guardian" / "canonical-marker.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(marker, {"schema": "mechanical-guardian-marker/v1", "instance_id": 1})
        self.event("guardian_marker_durable")
        if mode == "guardian_replay":
            self.event("guardian_marker_replay_rejected")
        if mode == "guardian_race":
            self.event("guardian_race_one_winner_seven_rejected")
        if mode == "guardian_timeout":
            self.event("guardian_timeout_fail_closed")
        transient_events = {
            "guardian_transient_key": "guardian_key_error_retry_recovered",
            "guardian_transient_show_timeout": "guardian_show_timeout_retry_recovered",
            "guardian_transient_nonzero": "guardian_nonzero_retry_recovered",
            "guardian_transient_malformed": "guardian_malformed_retry_recovered",
            "guardian_transient_parse": "guardian_parse_retry_recovered",
            "guardian_transient_rate_limit": "guardian_rate_limit_retry_recovered",
            "guardian_wrong_row": "guardian_wrong_row_rejected_then_recovered",
            "bootstrap_import_failure": "bootstrap_import_failure_marker_supervised",
        }
        if mode in transient_events:
            self.event(transient_events[mode])
        if mode == "guardian_deadline_fail_closed":
            self.event("guardian_terminal_receipt_durable_unverified")
            return {
                "status": "GUARDIAN_DEADLINE_EXHAUSTED_FAIL_CLOSED",
                "terminal_state": "unverified",
                "receipt_durable": True,
            }
        if mode == "guardian_repeated_absence":
            for observation in range(1, 4):
                self.event(
                    "guardian_canonical_absence_observed",
                    consecutive_absences=observation,
                )
            self.event("guardian_destroy_verified_repeated_absence")
            return {
                "status": "GUARDIAN_DESTROY_VERIFIED_REPEATED_ABSENCE",
                "terminal_state": "destroyed",
                "consecutive_absences": 3,
                "receipt_durable": True,
            }
        self.event("guardian_stop_verified_terminal")
        return {"status": "GUARDIAN_STOP_VERIFIED_TERMINAL", "marker_actions": 1}

    def destroy(self) -> None:
        required = {
            "raw_terminal", "scratch_pointer", "critic_audit", "artifact_bundle",
            "artifact_manifest", "artifact_receipts",
        }
        if self.verified != required:
            raise CapsuleError("mechanical teardown refused incomplete artifact durability")
        self.event("destroy_verified_terminal_after_complete_artifacts")


def _payload_artifacts(root: Path, failure: str) -> tuple[Path, Path]:
    output = root / "scratch" / "output"
    output.mkdir(parents=True)
    interface = output / "interface-control.json"
    checkpoint = output / "step-000001.pt"
    interface.write_bytes(b'{"status":"tiny-control"}\n')
    checkpoint.write_bytes(b"tiny-checkpoint\n")
    if failure == "artifact_missing_body":
        checkpoint.unlink()
    elif failure == "artifact_wrong_body":
        checkpoint.write_bytes(b"wrong-checkpoint\n")
    elif failure == "artifact_extra_body":
        (output / "unexpected.bin").write_bytes(b"extra")
    expected_checkpoint = hashlib.sha256(b"tiny-checkpoint\n").hexdigest()
    expected = {"interface-control.json", "step-000001.pt"}
    if {path.name for path in output.iterdir() if path.is_file()} != expected:
        raise CapsuleError("mechanical artifact inventory missing or extra")
    if sha256_file(checkpoint) != expected_checkpoint:
        raise CapsuleError("mechanical checkpoint body changed")
    manifest = root / "scratch" / "artifacts.manifest.json"
    atomic_json(
        manifest,
        {
            "schema": "mechanical-payload-manifest/v1",
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (interface, checkpoint)
            ],
        },
    )
    return output, manifest


def _mechanical_deployment_layout(
    root: Path, capsule: dict[str, Any]
) -> dict[str, Any]:
    """Stage exact bytes and execute the exact path-rebased outer argv analogue."""

    repository = Path(__file__).resolve().parents[3]
    layout = root / "fixed-layout"
    production = capsule["contract"]["production_runtime"]["deployment"]
    directories: dict[str, str] = {}
    for absolute, mode_text in production["fixed_layout_modes"].items():
        target = layout / absolute.lstrip("/")
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, int(mode_text, 8))
        if target.stat().st_mode & 0o777 != int(mode_text, 8):
            raise CapsuleError("mechanical fixed directory mode changed")
        directories[absolute] = mode_text
    mounts: dict[str, str] = {}
    for absolute, mode_text in production["fixed_mount_modes"].items():
        target = layout / absolute.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if absolute == "/run/parallax/pg1-native-entry-contract.json":
            # The mechanical dispatcher is the provisioner for this distinct
            # no-authority contract.  Materialize its final bytes below, then
            # apply the same immutable mode production receives from its mount.
            mounts[absolute] = mode_text
            continue
        target.write_bytes(
            canonical_bytes(
                {
                    "schema": "pg1-native-mechanical-fixed-mount/v1",
                    "production_path": absolute,
                }
            )
        )
        os.chmod(target, int(mode_text, 8))
        if target.stat().st_mode & 0o777 != int(mode_text, 8):
            raise CapsuleError("mechanical fixed mount mode changed")
        mounts[absolute] = mode_text
    image_root = layout / production["immutable_image_source_root"].lstrip("/")
    stage_root = layout / production["bootstrap_source_root"].lstrip("/")
    staged: list[dict[str, Any]] = []
    for relative_text, expected_sha256 in sorted(
        capsule["contract"]["runtime_wiring"].items()
    ):
        relative = Path(relative_text)
        source = repository / relative
        image = image_root / relative
        target = stage_root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, image)
        shutil.copyfile(image, target)
        image_mode = 0o755 if relative_text == "infra/vast/parallax/pg1_native_deployment.py" else 0o400
        staged_mode = 0o755 if relative_text == "infra/vast/parallax/pg1_native_bootstrap.py" else 0o400
        os.chmod(image, image_mode)
        os.chmod(target, staged_mode)
        if sha256_file(image) != expected_sha256 or sha256_file(target) != expected_sha256:
            raise CapsuleError("mechanical deployment runtime staging changed")
        staged.append(
            {
                "path": relative_text,
                "sha256": expected_sha256,
                "image_mode": f"{image_mode:04o}",
                "staged_mode": f"{staged_mode:04o}",
            }
        )
    production_argv = list(capsule["contract"]["entrypoints"]["production_argv"])
    rebased_argv = [
        str(layout / value.lstrip("/")) if value.startswith("/") else value
        for value in production_argv
    ]
    bootstrap_argv = [
        str(layout / value.lstrip("/")) if value.startswith("/") else value
        for value in capsule["contract"]["entrypoints"]["bootstrap_argv"]
    ]
    entry_contract_path = Path(rebased_argv[rebased_argv.index("--entry-contract") + 1])
    bootstrap_argv[bootstrap_argv.index("--entry-contract") + 1] = str(entry_contract_path)
    execution_receipt = layout / "mechanical-exact-bootstrap-executed.json"
    mechanical_contract = {
        "schema": "pg1-native-entry-dispatch-contract/v1",
        "kind": "MECHANICAL_NO_AUTHORITY",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "argv_sha256": hashlib.sha256(canonical_bytes(rebased_argv)).hexdigest(),
        "argv": rebased_argv,
        "rebase_root": str(layout),
        "image_root": str(image_root),
        "bootstrap_root": str(stage_root),
        "runtime_wiring": capsule["contract"]["runtime_wiring"],
        "bootstrap_argv": bootstrap_argv,
        "receipt_path": str(execution_receipt),
        "dispatch_scope": "EXACT_REBASED_BYTES_ONLY",
        "production_authority": False,
        "may_mint_authorization": False,
    }
    root_body = {
        "kind": "MECHANICAL_NO_AUTHORITY", "production_authority": False,
        "provider_calls": 0, "root_rebase": str(layout.resolve()),
        "fake_services": "TRACKED_DETERMINISTIC_EXECUTABLES",
        "network_scope": "LOOPBACK_OR_NONE",
        "may_mint_production_authority": False,
    }
    root_contract = {
        "schema": "pg1-native-exact-chain-mechanical-contract/v1",
        "status": "OFFLINE_DETERMINISTIC_FAKE_SERVICES_ONLY",
        "content_address": {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical_bytes(root_body)).hexdigest(),
        },
        "contract": root_body,
    }
    root_contract_path = (
        layout / "run/parallax/pg1-native-mechanical-contract.json"
    )
    root_contract_path.write_bytes(canonical_bytes(root_contract))
    os.chmod(root_contract_path, 0o400)
    entry_contract_path.write_bytes(canonical_bytes(mechanical_contract))
    os.chmod(
        entry_contract_path,
        int(production["fixed_mount_modes"]["/run/parallax/pg1-native-entry-contract.json"], 8),
    )
    exact_entry = Path(rebased_argv[0])
    if sha256_file(exact_entry) != production["outer_entry"]["sha256"]:
        raise CapsuleError("mechanical executable is not exact production provisioner bytes")
    completed = subprocess.run(
        rebased_argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=image_root,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPATH": str(image_root),
        },
    )
    if completed.returncode != 0 or not execution_receipt.is_file():
        raise CapsuleError(f"exact production provisioner/bootstrap bytes failed: {completed.stderr}")
    executed = json.loads(execution_receipt.read_text(encoding="utf-8"))
    if (
        executed.get("status") != "EXACT_COPIED_BOOTSTRAP_BYTES_EXECUTED"
        or executed.get("argv") != bootstrap_argv
        or executed.get("provider_calls") != 0
    ):
        raise CapsuleError("mechanical path-rebased execution proof changed")
    receipt = {
        "schema": "pg1-native-mechanical-deployment-layout/v1",
        "status": "FIXED_LAYOUT_AND_EXACT_RUNTIME_STAGED_NO_PROVIDER",
        "directories": directories,
        "mounts": mounts,
        "runtime_files": staged,
        "runtime_file_count": len(staged),
        "production_argv": production_argv,
        "path_rebased_argv": rebased_argv,
        "argv_equivalence": [
            {"production": source, "path_rebased": target}
            for source, target in zip(production_argv, rebased_argv, strict=True)
        ],
        "production_entry_sha256": production["outer_entry"]["sha256"],
        "path_rebased_production_sha256": sha256_file(exact_entry),
        "exact_bootstrap_sha256": executed["bootstrap_sha256"],
        "executed": True,
        "execution_returncode": completed.returncode,
        "execution_receipt_sha256": sha256_file(execution_receipt),
        "invoked_argv": capsule["contract"]["entrypoints"]["bootstrap_argv"],
        "next_argv": capsule["contract"]["entrypoints"]["native_launcher_argv"],
        "provider_calls": 0,
    }
    receipt_path = layout / production["layout_receipt_path"].lstrip("/")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    os.chmod(receipt_path, 0o400)
    receipt["layout_receipt"] = {
        "path": production["layout_receipt_path"],
        "mode": "0400",
        "sha256": sha256_file(receipt_path),
    }
    return receipt


def _package(root: Path, sources: list[Path]) -> Path:
    bundle = root / "scratch" / "artifacts.tar"
    with tarfile.open(bundle, "x") as archive:
        for source in sources:
            data = source.read_bytes()
            info = tarfile.TarInfo(source.name)
            info.size = len(data); info.mode = 0o400; info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(bundle, "r") as archive:
        if len([member for member in archive.getmembers() if member.isfile()]) != len(sources):
            raise CapsuleError("mechanical artifact bundle incomplete")
    return bundle


def run(
    *, capsule: dict[str, Any], mechanical: dict[str, Any],
    rehearsal_root: Path, failure: str, bootstrap_analogue: bool = False,
) -> dict[str, Any]:
    from .pg1_native_controller import capsule_preflight_graph
    from .pg1_native_overlay_stage import expected_receipt

    validate_mechanical_contract(mechanical, capsule)
    if failure not in FAILURES:
        raise CapsuleError("unknown mechanical failure plant")
    root = rehearsal_root.resolve()
    if root.exists():
        raise CapsuleError("refusing to overwrite mechanical rehearsal root")
    root.mkdir(parents=True)
    transport = DeterministicLifecycleTransport(root, failure)
    attempt = capsule["contract"]["attempt_identity"]
    deployment = (
        _mechanical_deployment_layout(root, capsule) if bootstrap_analogue else None
    )
    bootstrap_trace = [
        "MECHANICAL_FIXED_LAYOUT_CREATED",
        "MECHANICAL_EXACT_RUNTIME_STAGED",
        "MECHANICAL_REBASED_PRODUCTION_ARGV_EXECUTED",
        "MECHANICAL_PRODUCTION_ENTRYPOINT_INVOKED_BOOTSTRAP",
        "MECHANICAL_STOP_CONTEXT_VALIDATED", "MECHANICAL_GUARDIAN_STARTED",
        "MECHANICAL_RUNTIME_MATERIALIZED", "MECHANICAL_NATIVE_ARGV_INVOKED",
        "MECHANICAL_GUARDIAN_TERMINAL_RECEIPT_SUPERVISED",
    ] if bootstrap_analogue else []
    try:
        if failure in {
            "pre_setup", "primary_stop_failure", "fallback_stop_failure",
            "guardian_marker_failure", "guardian_race", "guardian_replay",
            "guardian_timeout", "provider_nonzero", "provider_timeout",
            "guardian_transient_key", "guardian_transient_show_timeout",
            "guardian_transient_nonzero", "guardian_transient_malformed",
            "guardian_transient_parse", "guardian_transient_rate_limit",
            "guardian_deadline_fail_closed", "guardian_repeated_absence",
            "guardian_wrong_row", "bootstrap_import_failure",
        }:
            raise CapsuleError("planted pre-setup/lifecycle failure")
        repository = Path(__file__).resolve().parents[3]
        preflight = capsule_preflight_graph(
            capsule, repository,
            expected_receipt(capsule, repository, allow_parameterized_fixture=True),
            allow_parameterized_fixture=True,
        )
        if preflight["call_trace"] != list(CONSUMER_ORDER):
            raise CapsuleError("mechanical command did not traverse seven owners")
        store = DeterministicAuthorizationStore(root / "authority")
        authorization = store.issue(capsule, mechanical)
        receipt = store.consume(capsule, mechanical, authorization)
        fixtures = _write_fixtures(root, mechanical["contract"], failure=failure)
        hydration = _verify_fixture_hydration(mechanical, fixtures)
        transport.event("runtime_hydration_verified_before_science")
        output, manifest_path = _payload_artifacts(root, failure)
        terminal_path = root / "scratch" / "terminal.json"
        scratch_path = root / "scratch" / "scratch-pointer.json"
        critic_path = root / "scratch" / "independent-critic.json"
        atomic_json(
            terminal_path,
            {
                "status": "GREEN_JOB_TERMINAL", "run_id": attempt["attempt_run_id"],
                "payload_manifest_sha256": sha256_file(manifest_path),
            },
        )
        atomic_json(scratch_path, {"status": "SCRATCH_PRESERVED", "run_id": attempt["attempt_run_id"]})
        receipts: dict[str, Any] = {}
        for kind, path, key in (
            ("raw_terminal", terminal_path, attempt["terminal_key"]),
            ("scratch_pointer", scratch_path, attempt["scratch_key"]),
        ):
            receipts[kind] = transport.durable_sink(kind=kind, source_path=path, object_key=key)
            transport.verify(kind=kind, receipt=receipts[kind])
        transport.event("critic_execution_started")
        if failure == "critic_execution":
            raise CapsuleError("planted critic execution failure")
        atomic_json(
            critic_path,
            {
                "status": "GREEN_INDEPENDENT_EXACT", "run_id": attempt["attempt_run_id"],
                "terminal_sha256": sha256_file(terminal_path),
                "payload_manifest_sha256": sha256_file(manifest_path),
            },
        )
        transport.event("independent_critic_complete")
        receipts["critic_audit"] = transport.durable_sink(
            kind="critic_audit", source_path=critic_path, object_key=attempt["critic_key"]
        )
        transport.verify(kind="critic_audit", receipt=receipts["critic_audit"])
        bundle = _package(
            root,
            [*sorted(output.iterdir()), manifest_path, terminal_path, scratch_path, critic_path],
        )
        for kind, path, key in (
            ("artifact_bundle", bundle, attempt["artifact_key"]),
            ("artifact_manifest", manifest_path, attempt["artifact_manifest_key"]),
        ):
            receipts[kind] = transport.durable_sink(kind=kind, source_path=path, object_key=key)
            transport.verify(kind=kind, receipt=receipts[kind])
        final_path = root / "scratch" / "durability-receipts.json"
        atomic_json(
            final_path,
            {
                "schema": "mechanical-final-durability/v1",
                "verified": sorted(transport.verified),
                "payload_manifest_sha256": sha256_file(manifest_path),
                "terminal_sha256": sha256_file(terminal_path),
                "critic_sha256": sha256_file(critic_path),
            },
        )
        receipts["artifact_receipts"] = transport.durable_sink(
            kind="artifact_receipts", source_path=final_path,
            object_key=attempt["artifact_receipts_key"],
        )
        transport.verify(kind="artifact_receipts", receipt=receipts["artifact_receipts"])
        transport.destroy()
    except Exception as error:
        stop = None; stop_error = None
        try:
            stop = transport.stop(failure=type(error).__name__)
        except Exception as candidate:
            stop_error = candidate
        result = {
            "schema": "pg1-native-mechanical-result/v3",
            "status": "FAILED_STOPPED_OR_EXHAUSTED_ARTIFACTS_PRESERVED",
            "failure": failure, "attempt_run_id": attempt["attempt_run_id"],
            "bootstrap_trace": bootstrap_trace, "stop": stop,
            "deployment": deployment,
            "stop_error": type(stop_error).__name__ if stop_error else None,
            "destroy_invoked": False,
            "failure_pointer_preserved": (root / "scratch" / "failure-stop-pointer.json").is_file(),
            "external_commands": 0, "h100_instances_created": 0,
        }
        atomic_json(root / "rehearsal-result.json", result)
        raise CapsuleError("mechanical failure plant completed") from (stop_error or error)
    result = {
        "schema": "pg1-native-mechanical-result/v3",
        "status": "GREEN_MECHANICAL_BOOTSTRAP_ARTIFACT_GUARDIAN_PROVED",
        "attempt_run_id": attempt["attempt_run_id"],
        "bootstrap_trace": bootstrap_trace,
        "deployment": deployment,
        "owner_trace": list(CONSUMER_ORDER),
        "authorization_schema": authorization["contract"]["schema"],
        "receipt_schema": receipt["schema"], "hydration": hydration,
        "durable_kinds": sorted(transport.verified),
        "external_commands": 0, "archive_body_reads": 0,
        "confirmation_items_read": 0, "ml_imports": 0, "h100_instances_created": 0,
    }
    atomic_json(root / "rehearsal-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mechanical-rehearsal", action="store_true")
    mode.add_argument("--mechanical-bootstrap", action="store_true")
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rehearsal-root", type=Path, required=True)
    parser.add_argument("--failure", choices=FAILURES, required=True)
    args = parser.parse_args()
    capsule = load_capsule(args.capsule)
    mechanical = load_mechanical_contract(args.contract, capsule)
    print(
        json.dumps(
            run(
                capsule=capsule, mechanical=mechanical,
                rehearsal_root=args.rehearsal_root, failure=args.failure,
                bootstrap_analogue=args.mechanical_bootstrap,
            ),
            sort_keys=True, separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
