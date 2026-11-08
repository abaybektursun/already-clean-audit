"""Pinned production durability and fail-closed PG1 lifecycle backend."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .pg1_execution_capsule import (
    AUTHORIZATION_BUCKET,
    AUTHORIZATION_PREFIX,
    EXPECTED_BUCKET_OWNER,
    CapsuleError,
    canonical_bytes,
    content_sha256,
)
from .pg1_native_artifacts import TEARDOWN_SCHEMA
from . import pg1_native_lifecycle as lifecycle
from . import pg1_native_launch_state as launch_state
from .pg1_native_provider import (
    aws_prefix,
    load_stop_context,
    load_vast_key,
    run_json,
    run_vast_mutation,
    validate_stop_context,
    vast_show_command,
    vast_stop_command,
    wait_vast_terminal,
)


PRE_SCIENCE_TEARDOWN_SCHEMA = "pg1-native-pre-science-verified-teardown-evidence/v1"
LIFECYCLE_PATH = Path("/workspace/parallax/failure-stop-pointer.json")
LAUNCH_PACKET_PATH = Path("/run/parallax/pg1-future-launch-packet.json")


def validate_destroy_evidence(
    evidence: dict[str, Any], capsule: dict[str, Any]
) -> str:
    """Authorize destroy only after one of the two independently verified gates."""

    capsule_sha256 = capsule["content_address"]["digest"]
    attempt_run_id = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    final_digest_fields = {
        "final_receipt_sha256",
        "receipt_durability_sha256",
        "receipt_verification_sha256",
    }
    if (
        set(evidence)
        == {
            "schema", "status", "capsule_sha256", "attempt_run_id",
            *final_digest_fields, "teardown_authorized",
        }
        and evidence.get("schema") == TEARDOWN_SCHEMA
        and evidence.get("status")
        == "FINAL_DURABILITY_RECEIPT_DOWNLOADED_VERIFIED_FOR_TEARDOWN"
        and evidence.get("capsule_sha256") == capsule_sha256
        and evidence.get("attempt_run_id") == attempt_run_id
        and evidence.get("teardown_authorized") is True
        and all(
            isinstance(evidence.get(field), str)
            and len(evidence[field]) == 64
            and all(character in "0123456789abcdef" for character in evidence[field])
            for field in final_digest_fields
        )
    ):
        return "COMPLETE_ARTIFACT_DURABILITY"
    zero_reads = {
        "archive_body_reads": 0,
        "model_body_reads": 0,
        "data_body_reads": 0,
        "confirmation_items_read": 0,
    }
    expected_kinds = {
        "pre_science_terminal",
        "pre_science_audit",
        "deployment_layout_receipt",
    }
    verifications = evidence.get("verified_objects")
    if (
        set(evidence)
        != {
            "schema", "status", "capsule_sha256", "attempt_run_id",
            "science_started", "protected_body_reads", "output_files",
            "checkpoints", "verified_objects", "teardown_authorized",
        }
        or evidence.get("schema") != PRE_SCIENCE_TEARDOWN_SCHEMA
        or evidence.get("status")
        != "PRE_SCIENCE_TERMINAL_AUDIT_AND_LAYOUT_DOWNLOADED_VERIFIED_FOR_TEARDOWN"
        or evidence.get("capsule_sha256") != capsule_sha256
        or evidence.get("attempt_run_id") != attempt_run_id
        or evidence.get("science_started") is not False
        or evidence.get("protected_body_reads") != zero_reads
        or evidence.get("output_files") != 0
        or evidence.get("checkpoints") != 0
        or evidence.get("teardown_authorized") is not True
        or not isinstance(verifications, dict)
        or set(verifications) != expected_kinds
    ):
        raise CapsuleError("destroy requires complete final or pre-science durability evidence")
    for kind, verification in verifications.items():
        if (
            not isinstance(verification, dict)
            or set(verification)
            != {
                "status", "kind", "receipt_sha256", "downloaded_sha256",
                "downloaded_bytes",
            }
            or verification.get("status")
            != "VERIFIED_EXACT_DOWNLOADED_DURABLE_OBJECT"
            or verification.get("kind") != kind
            or not isinstance(verification.get("receipt_sha256"), str)
            or len(verification["receipt_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in verification["receipt_sha256"]
            )
            or not isinstance(verification.get("downloaded_sha256"), str)
            or len(verification["downloaded_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in verification["downloaded_sha256"]
            )
            or not isinstance(verification.get("downloaded_bytes"), int)
            or verification["downloaded_bytes"] <= 0
        ):
            raise CapsuleError("pre-science destroy verification is incomplete")
    return "PRE_SCIENCE_ZERO_BODY_DURABILITY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum(path: Path) -> str:
    return base64.b64encode(bytes.fromhex(_sha256(path))).decode("ascii")


def _aws_prefix() -> list[str]:
    return aws_prefix()


class ProductionStopper:
    """Pointer-first, locally serialized stop with a tracked guardian fallback."""

    def __init__(
        self,
        *,
        capsule: dict[str, Any] | None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.capsule = capsule
        self.context = validate_stop_context(
            context if context is not None else load_stop_context(),
            capsule=capsule,
        )
        deployment = self.context["contract"]
        self.capsule_sha256 = deployment["capsule_sha256"]
        self.attempt_run_id = deployment["attempt_run_id"]
        self.instance_id = deployment["instance_id"]
        self.failure_pointer = Path("/workspace/parallax/failure-stop-pointer.json")
        self.state_path = Path("/workspace/parallax/pg1-native-stop-state.json")
        self.lock_path = Path("/workspace/parallax/pg1-native-stop.lock")

    def bind_instance(self, instance_id: int) -> None:
        if instance_id != self.instance_id:
            raise CapsuleError("authorization instance differs from deployment context")

    def primary_command(self, api_key: str) -> list[str]:
        return vast_stop_command(self.instance_id, api_key)

    def show_command(self, api_key: str) -> list[str]:
        return vast_show_command(self.instance_id, api_key)

    def guardian_key(self) -> str:
        return (
            f"{AUTHORIZATION_PREFIX}/stop-guardian/{self.capsule_sha256}/"
            f"{self.attempt_run_id}.json"
        )

    def fallback_command(self, body: Path, data: bytes) -> list[str]:
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        return [
            *_aws_prefix(), "s3api", "put-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", self.guardian_key(),
            "--body", str(body),
            "--if-none-match", "*",
            "--checksum-algorithm", "SHA256",
            "--checksum-sha256", checksum,
            "--content-length", str(len(data)),
            "--content-type", "application/json",
            "--server-side-encryption", "AES256",
            "--metadata", json.dumps(
                {
                    "capsule-sha256": self.capsule_sha256,
                    "attempt-run-id": self.attempt_run_id,
                    "instance-id": str(self.instance_id),
                    "object-kind": "stop-guardian-request",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        ]

    def fallback_verify_command(self, target: Path) -> list[str]:
        return [
            *_aws_prefix(), "s3api", "get-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", self.guardian_key(),
            "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            str(target),
        ]

    @staticmethod
    def _run_json(command: list[str]) -> dict[str, Any]:
        return run_json(command)

    def _persist_pointer(self, failure: str) -> None:
        for _attempt in range(8):
            prior = lifecycle.read(self.failure_pointer, self.context)
            if prior is not None and prior["state"] in {
                "FAILURE_STOP_REQUIRED", "TERMINAL_STOPPED_PRESERVED"
            }:
                return
            try:
                lifecycle.transition(
                    self.failure_pointer,
                    self.context,
                    "FAILURE_STOP_REQUIRED",
                    reason=failure,
                    expected_sequence=0 if prior is None else int(prior["sequence"]),
                )
                return
            except lifecycle.LifecycleCASFailure as error:
                if "sequence mismatch" not in str(error):
                    raise CapsuleError(str(error)) from error
        raise CapsuleError("stopper lifecycle CAS retry budget exhausted")

    def _prior(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("capsule_sha256") != self.capsule_sha256
            or value.get("attempt_run_id") != self.attempt_run_id
            or value.get("instance_id") != self.instance_id
        ):
            raise CapsuleError("stop state identity changed")
        return value

    def _claim(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
            )
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{self.capsule_sha256}:{self.attempt_run_id}:{self.instance_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def _guardian_marker(self, failure: str) -> dict[str, Any]:
        pointer_sha = _sha256(self.failure_pointer) if self.failure_pointer.is_file() else None
        return {
            "schema": "pg1-native-stop-guardian-request/v3",
            "capsule_sha256": self.capsule_sha256,
            "attempt_run_id": self.attempt_run_id,
            "stop_context_sha256": self.context["content_address"]["digest"],
            "instance_id": self.instance_id,
            "offer_id": self.context["contract"]["offer_id"],
            "machine_id": self.context["contract"]["machine_id"],
            "container_id": self.context["contract"]["container_id"],
            "failure": failure,
            "failure_pointer": str(self.failure_pointer),
            "failure_pointer_sha256": pointer_sha,
            "artifact_manifest": "/workspace/parallax/artifacts.manifest.json",
            "scratch_preserved": True,
        }

    def _publish_guardian_marker(
        self, marker: dict[str, Any]
    ) -> dict[str, Any]:
        """Conditionally publish or exact-readback a concurrent canonical marker."""

        data = canonical_bytes(marker)
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        with tempfile.TemporaryDirectory(prefix="pg1-stop-marker-") as directory:
            body = Path(directory) / "marker.json"
            body.write_bytes(data)
            try:
                result = self._run_json(self.fallback_command(body, data))
                replay = False
                if (
                    not isinstance(result, dict)
                    or not result.get("VersionId")
                    or not result.get("ETag")
                    or result.get("ChecksumSHA256") != checksum
                    or result.get("ServerSideEncryption") != "AES256"
                ):
                    raise CapsuleError("guardian marker response lacks exact S3 evidence")
            except Exception:
                target = Path(directory) / "existing-marker.json"
                result = self._run_json(self.fallback_verify_command(target))
                if target.read_bytes() != data:
                    raise CapsuleError("concurrent guardian marker body changed")
                if (
                    not isinstance(result, dict)
                    or not result.get("VersionId")
                    or not result.get("ETag")
                    or result.get("ChecksumSHA256") != checksum
                    or result.get("ServerSideEncryption") != "AES256"
                    or result.get("ContentLength") != len(data)
                    or result.get("ContentType") != "application/json"
                    or result.get("Metadata")
                    != {
                        "capsule-sha256": self.capsule_sha256,
                        "attempt-run-id": self.attempt_run_id,
                        "instance-id": str(self.instance_id),
                        "object-kind": "stop-guardian-request",
                    }
                ):
                    raise CapsuleError("concurrent guardian marker evidence changed")
                replay = True
        return {
            "version_id": str(result["VersionId"]),
            "etag": str(result["ETag"]),
            "checksum_sha256": checksum,
            "replayed_exact": replay,
        }

    def stop_preserve_scratch(self, *, failure: str) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        try:
            self._persist_pointer(failure)
            events.append({"action": "FAILURE_POINTER", "status": "PRESERVED"})
        except Exception as error:
            events.append(
                {"action": "FAILURE_POINTER", "status": "FAILED", "error": type(error).__name__}
            )
        try:
            prior = self._prior()
        except Exception as error:
            prior = None
            events.append(
                {"action": "PRIOR_STATE", "status": "INVALID", "error": type(error).__name__}
            )
        if prior is not None and prior.get("status") in {
            "PRIMARY_STOP_VERIFIED_TERMINAL", "GUARDIAN_MARKER_DURABLE"
        }:
            return prior
        if not self._claim():
            try:
                prior = self._prior()
            except Exception as error:
                prior = None
                events.append(
                    {
                        "action": "CONCURRENT_STATE", "status": "INVALID_TAKEOVER",
                        "error": type(error).__name__,
                    }
                )
            if prior is not None and prior.get("status") in {
                "PRIMARY_STOP_VERIFIED_TERMINAL", "GUARDIAN_MARKER_DURABLE"
            }:
                return prior
            events.append(
                {"action": "CONCURRENT_STOP_OWNER", "status": "BOUNDED_IDEMPOTENT_TAKEOVER"}
            )
        for attempt in range(1, 3):
            try:
                # Key provenance is inside the primary attempt boundary.  A
                # missing/wrong key therefore falls through to the independent
                # fixed-AWS marker channel below.
                api_key = load_vast_key(self.context)
                mutation = run_vast_mutation(self.primary_command(api_key))
                terminal = wait_vast_terminal(
                    self.context, api_key, terminal_state="stopped"
                )
                events.append(
                    {"action": "PRIMARY_VAST_STOP", "attempt": attempt, "status": "VERIFIED"}
                )
                state = {
                    "schema": "pg1-native-stop-state/v3",
                    "status": "PRIMARY_STOP_VERIFIED_TERMINAL",
                    "capsule_sha256": self.capsule_sha256,
                    "attempt_run_id": self.attempt_run_id,
                    "instance_id": self.instance_id,
                    "failure": failure,
                    "events": events,
                    "mutation_exit_code": mutation["exit_code"],
                    "terminal_state": terminal["state"],
                    "destroy_requested": False,
                }
                atomic_json(self.state_path, state)
                return state
            except Exception as error:
                events.append(
                    {
                        "action": "PRIMARY_VAST_STOP", "attempt": attempt,
                        "status": "FAILED", "error": type(error).__name__,
                    }
                )
        marker = self._guardian_marker(failure)
        try:
            fallback: dict[str, Any] | None = None
            last_error: Exception | None = None
            for attempt in range(1, 3):
                try:
                    fallback = self._publish_guardian_marker(marker)
                    events.append(
                        {
                            "action": "GUARDIAN_MARKER",
                            "attempt": attempt,
                            "status": "DURABLE",
                            "replayed_exact": fallback["replayed_exact"],
                        }
                    )
                    break
                except Exception as error:
                    last_error = error
                    events.append(
                        {
                            "action": "GUARDIAN_MARKER",
                            "attempt": attempt,
                            "status": "RETRY",
                            "error": type(error).__name__,
                        }
                    )
            if fallback is None:
                raise CapsuleError("guardian marker publication attempts failed") from last_error
            state = {
                "schema": "pg1-native-stop-state/v3",
                "status": "GUARDIAN_MARKER_DURABLE",
                "capsule_sha256": self.capsule_sha256,
                "attempt_run_id": self.attempt_run_id,
                "instance_id": self.instance_id,
                "failure": failure,
                "events": events,
                "fallback_key": self.guardian_key(),
                "fallback_version_id": fallback["version_id"],
                "fallback_replayed_exact": fallback["replayed_exact"],
                "destroy_requested": False,
            }
            atomic_json(self.state_path, state)
            return state
        except Exception as error:
            events.append(
                {"action": "GUARDIAN_MARKER", "status": "FAILED", "error": type(error).__name__}
            )
            state = {
                "schema": "pg1-native-stop-state/v3",
                "status": "STOP_CHANNELS_EXHAUSTED",
                "capsule_sha256": self.capsule_sha256,
                "attempt_run_id": self.attempt_run_id,
                "instance_id": self.instance_id,
                "failure": failure,
                "events": events,
                "destroy_requested": False,
            }
            atomic_json(self.state_path, state)
            raise CapsuleError("primary and guardian stop channels failed") from error


class ProductionBackend:
    """Collision-safe exact-byte S3 backend with verified Vast teardown."""

    def __init__(
        self,
        *,
        capsule: dict[str, Any],
        instance_id: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.capsule = capsule
        self.context = validate_stop_context(
            context if context is not None else load_stop_context(), capsule=capsule
        )
        if instance_id != self.context["contract"]["instance_id"]:
            raise CapsuleError("backend instance differs from current deployment")
        self.instance_id = instance_id

    def metadata(self, *, kind: str, source_sha256: str) -> dict[str, str]:
        return {
            "attempt-run-id": self.capsule["contract"]["attempt_identity"]["attempt_run_id"],
            "capsule-sha256": self.capsule["content_address"]["digest"],
            "instance-id": str(self.instance_id),
            "object-kind": kind,
            "source-sha256": source_sha256,
        }

    def upload_command(
        self, *, kind: str, source_path: Path, object_key: str
    ) -> list[str]:
        digest = _sha256(source_path)
        content_type = "application/x-tar" if source_path.suffix == ".tar" else "application/json"
        return [
            *_aws_prefix(), "s3api", "put-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", object_key,
            "--body", str(source_path),
            "--if-none-match", "*",
            "--checksum-algorithm", "SHA256",
            "--checksum-sha256", _checksum(source_path),
            "--content-length", str(source_path.stat().st_size),
            "--content-type", content_type,
            "--server-side-encryption", "AES256",
            "--metadata", json.dumps(
                self.metadata(kind=kind, source_sha256=digest),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        ]

    @staticmethod
    def download_command(object_key: str, version_id: str, target: Path) -> list[str]:
        return [
            *_aws_prefix(), "s3api", "get-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", object_key,
            "--version-id", version_id,
            "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            str(target),
        ]

    def stop_command(self, api_key: str) -> list[str]:
        return vast_stop_command(self.instance_id, api_key)

    @staticmethod
    def _run_json(command: list[str]) -> dict[str, Any]:
        return run_json(command)

    def durable_sink(
        self, *, kind: str, source_path: Path, object_key: str
    ) -> dict[str, Any]:
        source_sha256 = _sha256(source_path)
        checksum = _checksum(source_path)
        metadata = self.metadata(kind=kind, source_sha256=source_sha256)
        content_type = "application/x-tar" if source_path.suffix == ".tar" else "application/json"
        result = self._run_json(
            self.upload_command(kind=kind, source_path=source_path, object_key=object_key)
        )
        receipt = {
            "schema": "pg1-native-durability-receipt/v3",
            "status": "DURABLE_CONDITIONAL_CREATE_RETURNED",
            "kind": kind,
            "capsule_sha256": self.capsule["content_address"]["digest"],
            "attempt_run_id": self.capsule["contract"]["attempt_identity"]["attempt_run_id"],
            "object_key": object_key,
            "source_sha256": source_sha256,
            "checksum_sha256": checksum,
            "content_length": source_path.stat().st_size,
            "content_type": content_type,
            "server_side_encryption": "AES256",
            "metadata": metadata,
            "version_id": str(result.get("VersionId", "")),
            "etag": str(result.get("ETag", "")),
        }
        if (
            not receipt["version_id"]
            or not receipt["etag"]
            or result.get("ChecksumSHA256") != checksum
            or result.get("ServerSideEncryption") != "AES256"
        ):
            raise CapsuleError(f"{kind} upload lacks exact durability evidence")
        return receipt

    def verify_durability(
        self, *, kind: str, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="pg1-durable-") as directory:
            target = Path(directory) / f"{kind}.body"
            result = self._run_json(
                self.download_command(receipt["object_key"], receipt["version_id"], target)
            )
            downloaded_sha256 = _sha256(target)
            downloaded_bytes = target.stat().st_size
        if (
            result.get("VersionId") != receipt["version_id"]
            or result.get("ETag") != receipt["etag"]
            or result.get("ChecksumSHA256") != receipt["checksum_sha256"]
            or result.get("ServerSideEncryption") != receipt["server_side_encryption"]
            or result.get("ContentLength") != receipt["content_length"]
            or result.get("ContentType") != receipt["content_type"]
            or result.get("Metadata") != receipt["metadata"]
            or downloaded_bytes != receipt["content_length"]
            or downloaded_sha256 != receipt["source_sha256"]
        ):
            raise CapsuleError(f"{kind} downloaded durability proof changed")
        return {
            "schema": "pg1-native-durability-verification/v3",
            "status": "VERIFIED_EXACT_DOWNLOADED_DURABLE_OBJECT",
            "kind": kind,
            "receipt_sha256": content_sha256(receipt),
            "downloaded_sha256": downloaded_sha256,
            "downloaded_bytes": downloaded_bytes,
        }

    def stop_preserve_scratch(self, *, failure: str) -> dict[str, Any]:
        return ProductionStopper(
            capsule=self.capsule, context=self.context
        ).stop_preserve_scratch(failure=failure)

    def request_destroy(self, *, evidence: dict[str, Any]) -> dict[str, Any]:
        destroy_gate = validate_destroy_evidence(evidence, self.capsule)
        if destroy_gate != "COMPLETE_ARTIFACT_DURABILITY":
            raise CapsuleError("worker may request destroy only after complete artifact durability")
        prior_lifecycle = lifecycle.read(LIFECYCLE_PATH, self.context)
        if prior_lifecycle is None or prior_lifecycle["state"] not in {
            "RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED", "TERMINAL_STOPPED_PRESERVED"
        }:
            raise CapsuleError("destroy requires the shared armed lifecycle")
        capsule_sha = self.capsule["content_address"]["digest"]
        attempt = self.capsule["contract"]["attempt_identity"]["attempt_run_id"]
        try:
            if (
                LAUNCH_PACKET_PATH.is_symlink() or not LAUNCH_PACKET_PATH.is_file()
                or stat.S_IMODE(LAUNCH_PACKET_PATH.stat().st_mode) != 0o400
            ):
                raise CapsuleError("destroy request launch packet is not fixed")
            packet = json.loads(LAUNCH_PACKET_PATH.read_text(encoding="utf-8"))
            packet_sha = str(packet["content_address"]["digest"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise CapsuleError("destroy request launch packet unavailable") from error
        state_key = launch_state.state_key(packet_sha)
        def read_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            head = self._run_json([
                *_aws_prefix(), "s3api", "head-object", "--bucket", AUTHORIZATION_BUCKET,
                "--key", state_key, "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            ])
            version_id = head.get("VersionId")
            if not version_id:
                raise CapsuleError("destroy canonical launch state version unavailable")
            with tempfile.TemporaryDirectory(prefix="pg1-destroy-state-") as directory:
                target = Path(directory) / "state.json"
                downloaded = self._run_json([
                    *_aws_prefix(), "s3api", "get-object", "--bucket",
                    AUTHORIZATION_BUCKET, "--key", state_key, "--version-id",
                    str(version_id), "--checksum-mode", "ENABLED",
                    "--expected-bucket-owner", EXPECTED_BUCKET_OWNER, str(target),
                ])
                try:
                    state_data = target.read_bytes()
                    canonical_state = json.loads(state_data)
                    launch_state.validate(canonical_state, packet=packet)
                except (
                    OSError, json.JSONDecodeError, launch_state.LaunchStateError
                ) as error:
                    raise CapsuleError("destroy canonical launch state changed") from error
            exact_raw = launch_state.canonical_bytes(canonical_state)
            state_checksum = base64.b64encode(
                hashlib.sha256(state_data).digest()
            ).decode("ascii")
            expected_state_metadata = {
                "object-kind": "canonical-launch-state",
                "packet-sha256": packet_sha, "capsule-sha256": capsule_sha,
                "attempt-run-id": attempt,
                "state-sha256": canonical_state["self_sha256"],
                "state-sequence": str(canonical_state["sequence"]),
                "raw-body-sha256": hashlib.sha256(exact_raw).hexdigest(),
            }
            if (
                state_data != exact_raw
                or downloaded.get("VersionId") != version_id
                or downloaded.get("ETag") != head.get("ETag")
                or downloaded.get("ChecksumSHA256") != state_checksum
                or downloaded.get("Metadata") != expected_state_metadata
                or head.get("Metadata") != expected_state_metadata
                or downloaded.get("ServerSideEncryption") != "AES256"
                or head.get("ServerSideEncryption") != "AES256"
                or downloaded.get("ContentLength") != len(state_data)
                or head.get("ContentLength") != len(state_data)
                or downloaded.get("ContentType") != "application/json"
                or head.get("ContentType") != "application/json"
            ):
                raise CapsuleError("destroy canonical launch state provenance changed")
            return canonical_state, head, downloaded

        canonical_state, state_head, _state_download = read_state()
        if (
            canonical_state.get("capsule_sha256") != capsule_sha
            or canonical_state.get("attempt_run_id") != attempt
            or canonical_state.get("instance_id") != self.instance_id
            or canonical_state.get("science_state") != launch_state.SCIENCE_RELEASED
            or not canonical_state.get("start_commit_sha256")
        ):
            raise CapsuleError("destroy canonical launch state is not released")
        request = {
            "schema": "pg1-native-external-destroy-request/v2",
            "status": "DURABLE_EXTERNAL_DESTROY_REQUEST",
            "capsule_sha256": self.capsule["content_address"]["digest"],
            "attempt_run_id": attempt,
            "instance_id": self.instance_id,
            "destroy_gate": destroy_gate,
            "evidence_sha256": content_sha256(evidence),
            "teardown_authorized": True,
            "science_released": destroy_gate == "COMPLETE_ARTIFACT_DURABILITY",
            "launch_state_sequence": canonical_state["sequence"],
            "launch_state_sha256": canonical_state["self_sha256"],
            "start_commit_sha256": canonical_state["start_commit_sha256"],
            "start_commit_classification": "SCIENCE_RELEASED",
            "worker_provider_mutations": 0,
            "worker_absence_polls": 0,
        }
        request_reference = {
            "sha256": launch_state.digest(request),
            "storage": "CANONICAL_STATE_RAW_BODY",
        }
        candidate = launch_state.transition(
            canonical_state, "CLEANUP_DESTROYING",
            destroy_request=request_reference, destroy_request_body=request,
            destroy_gate=destroy_gate,
        )
        data = launch_state.canonical_bytes(candidate)
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        candidate_metadata = {
            "object-kind": "canonical-launch-state", "packet-sha256": packet_sha,
            "capsule-sha256": capsule_sha, "attempt-run-id": attempt,
            "state-sha256": candidate["self_sha256"],
            "state-sequence": str(candidate["sequence"]),
            "raw-body-sha256": hashlib.sha256(data).hexdigest(),
        }
        response: dict[str, Any] | None
        with tempfile.TemporaryDirectory(prefix="pg1-external-destroy-cas-") as directory:
            body = Path(directory) / "state.json"
            body.write_bytes(data)
            command = [
                *_aws_prefix(), "s3api", "put-object", "--bucket", AUTHORIZATION_BUCKET,
                "--key", state_key, "--body", str(body), "--if-match",
                str(state_head["ETag"]),
                "--checksum-algorithm", "SHA256", "--checksum-sha256", checksum,
                "--content-length", str(len(data)), "--content-type", "application/json",
                "--server-side-encryption", "AES256", "--metadata", json.dumps(
                    candidate_metadata,
                    sort_keys=True, separators=(",", ":")
                ), "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            ]
            try:
                response = self._run_json(command)
            except CapsuleError:
                response = None
        winner, winner_head, winner_download = read_state()
        if (
            winner != candidate
            or not launch_state.proves_direct_transition(winner, canonical_state)
            or response is not None and (
                response.get("ChecksumSHA256") != checksum
                or response.get("ServerSideEncryption") != "AES256"
                or response.get("VersionId") != winner_head.get("VersionId")
                or response.get("ETag") != winner_head.get("ETag")
                or winner_download.get("ChecksumSHA256") != checksum
            )
        ):
            raise CapsuleError("external destroy canonical CAS is not exact")
        return {
            **request,
            "canonical_state_key": state_key,
            "canonical_state_sha256": candidate["self_sha256"],
            "canonical_state_version_id": str(winner_head["VersionId"]),
            "canonical_state_etag": str(winner_head["ETag"]),
            "canonical_state_checksum_sha256": checksum,
        }
