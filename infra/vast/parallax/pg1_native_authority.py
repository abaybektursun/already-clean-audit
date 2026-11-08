"""Canonical production S3 issuer and atomic PG1 authorization consumer."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .pg1_execution_capsule import (
    AUTHORIZATION_BUCKET,
    AUTHORIZATION_PREFIX,
    AUTHORIZATION_RECEIPT_SCHEMA,
    AUTHORIZATION_STATUS,
    EXPECTED_BUCKET_OWNER,
    PRODUCTION_AUTHORIZATION_ISSUER,
    CapsuleError,
    canonical_bytes,
    content_sha256,
    production_authorization_id,
    production_consumed_key,
    production_consumption_marker,
    production_issued_key,
    validate_authorization_receipt,
    validate_execution_authorization,
)
from .pg1_native_provider import (
    aws_prefix,
    load_stop_context,
    run_json,
    validate_stop_context,
)


def _checksum(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


def _aws_prefix() -> list[str]:
    return aws_prefix()


class ProductionAuthorizationStore:
    """Fixed production trust root; no caller-selected bucket, key, or runner."""

    @staticmethod
    def issued_key(capsule: dict[str, Any]) -> str:
        return production_issued_key(capsule)

    @staticmethod
    def consumed_key(
        capsule: dict[str, Any], authorization: dict[str, Any]
    ) -> str:
        return production_consumed_key(capsule, authorization)

    @classmethod
    def owner_key(
        cls, capsule: dict[str, Any], authorization: dict[str, Any], owner: str
    ) -> str:
        if owner not in {"job", "cell", "worker", "critic"}:
            raise CapsuleError("unknown production execution owner")
        return f"{cls.consumed_key(capsule, authorization)}/owners/{owner}.json"

    @staticmethod
    def _put_command(
        *, key: str, body: Path, data: bytes, metadata: dict[str, str]
    ) -> list[str]:
        return [
            *_aws_prefix(), "s3api", "put-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", key,
            "--body", str(body),
            "--if-none-match", "*",
            "--checksum-algorithm", "SHA256",
            "--checksum-sha256", _checksum(data),
            "--content-length", str(len(data)),
            "--content-type", "application/json",
            "--server-side-encryption", "AES256",
            "--metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
        ]

    @classmethod
    def issue_command(
        cls, capsule: dict[str, Any], body: Path, data: bytes
    ) -> list[str]:
        return cls._put_command(
            key=cls.issued_key(capsule),
            body=body,
            data=data,
            metadata={
                "capsule-sha256": capsule["content_address"]["digest"],
                "attempt-run-id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
                "object-kind": "production-authorization",
            },
        )

    @classmethod
    def load_command(cls, capsule: dict[str, Any], target: Path) -> list[str]:
        return [
            *_aws_prefix(), "s3api", "get-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", cls.issued_key(capsule),
            "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            str(target),
        ]

    @classmethod
    def consume_command(
        cls,
        capsule: dict[str, Any],
        authorization: dict[str, Any],
        body: Path,
        data: bytes,
    ) -> list[str]:
        return cls._put_command(
            key=cls.consumed_key(capsule, authorization),
            body=body,
            data=data,
            metadata={
                "capsule-sha256": capsule["content_address"]["digest"],
                "attempt-run-id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
                "authorization-sha256": authorization["content_address"]["digest"],
                "object-kind": "production-authorization-consumption",
            },
        )

    @staticmethod
    def verify_consume_command(
        consume_key: str, version_id: str, target: Path
    ) -> list[str]:
        return [
            *_aws_prefix(), "s3api", "get-object",
            "--bucket", AUTHORIZATION_BUCKET,
            "--key", consume_key,
            "--version-id", version_id,
            "--checksum-mode", "ENABLED",
            "--expected-bucket-owner", EXPECTED_BUCKET_OWNER,
            str(target),
        ]

    @classmethod
    def owner_consume_command(
        cls,
        capsule: dict[str, Any],
        authorization: dict[str, Any],
        owner: str,
        body: Path,
        data: bytes,
    ) -> list[str]:
        return cls._put_command(
            key=cls.owner_key(capsule, authorization, owner),
            body=body,
            data=data,
            metadata={
                "capsule-sha256": capsule["content_address"]["digest"],
                "attempt-run-id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
                "authorization-sha256": authorization["content_address"]["digest"],
                "execution-owner": owner,
                "object-kind": "production-owner-consumption",
            },
        )

    @staticmethod
    def _aws(command: list[str]) -> dict[str, Any]:
        return run_json(command)

    @staticmethod
    def build_authorization(
        capsule: dict[str, Any], *, stop_context: dict[str, Any]
    ) -> dict[str, Any]:
        validate_stop_context(stop_context, capsule=capsule)
        deployment = stop_context["contract"]
        capsule_contract = capsule["contract"]
        contract = {
            "authorization_id": production_authorization_id(capsule),
            "capsule_sha256": capsule["content_address"]["digest"],
            "source_run_id": capsule_contract["source_identity"]["source_run_id"],
            "attempt_run_id": capsule_contract["attempt_identity"]["attempt_run_id"],
            "scope": "PG1_NATIVE_ONE_SHOT_EXECUTION",
            "one_shot": True,
            "h100_authority": True,
            "issuer": PRODUCTION_AUTHORIZATION_ISSUER,
            "issuer_backend": "S3_CONDITIONAL_CREATE_ISSUER_V2",
            "issuer_account": EXPECTED_BUCKET_OWNER,
            "store_uri": f"s3://{AUTHORIZATION_BUCKET}/{AUTHORIZATION_PREFIX}",
            "issued_key": production_issued_key(capsule),
            "instance_id": deployment["instance_id"],
            "stop_context_sha256": stop_context["content_address"]["digest"],
            "offer_id": deployment["offer_id"],
            "machine_id": deployment["machine_id"],
            "container_id": deployment["container_id"],
        }
        authorization = {
            "schema_version": 2,
            "status": AUTHORIZATION_STATUS,
            "content_address": {
                "algorithm": "sha256", "digest": content_sha256(contract)
            },
            "contract": contract,
        }
        return validate_execution_authorization(authorization, capsule)

    def issue(
        self, capsule: dict[str, Any], *, stop_context: dict[str, Any]
    ) -> dict[str, Any]:
        authorization = self.build_authorization(
            capsule, stop_context=stop_context
        )
        data = canonical_bytes(authorization)
        with tempfile.NamedTemporaryFile("wb") as handle:
            handle.write(data)
            handle.flush()
            response = self._aws(
                self.issue_command(capsule, Path(handle.name), data)
            )
        fields = {
            "version_id": str(response.get("VersionId", "")),
            "etag": str(response.get("ETag", "")),
            "checksum_sha256": str(response.get("ChecksumSHA256", "")),
            "server_side_encryption": str(response.get("ServerSideEncryption", "")),
        }
        if (
            not all(fields.values())
            or fields["checksum_sha256"] != _checksum(data)
            or fields["server_side_encryption"] != "AES256"
        ):
            raise CapsuleError("production issuance lacks exact S3 identity evidence")
        return {
            "authorization": authorization,
            "issuance": {
                "schema": "pg1-native-production-issuance-receipt/v2",
                "key": self.issued_key(capsule),
                **fields,
            },
        }

    def load_issued(self, capsule: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="pg1-authority-") as directory:
            target = Path(directory) / "authorization.json"
            response = self._aws(self.load_command(capsule, target))
            data = target.read_bytes()
            authorization = json.loads(data)
        if not isinstance(authorization, dict):
            raise CapsuleError("trusted authorization object is not an object")
        validate_execution_authorization(authorization, capsule)
        context = load_stop_context(capsule=capsule)
        deployment = context["contract"]
        authorized = authorization["contract"]
        if (
            authorized["stop_context_sha256"]
            != context["content_address"]["digest"]
            or authorized["instance_id"] != deployment["instance_id"]
            or authorized["offer_id"] != deployment["offer_id"]
            or authorized["machine_id"] != deployment["machine_id"]
            or authorized["container_id"] != deployment["container_id"]
        ):
            raise CapsuleError("authorization does not bind the current deployment context")
        if (
            data != canonical_bytes(authorization)
            or str(response.get("ChecksumSHA256", "")) != _checksum(data)
            or str(response.get("ServerSideEncryption", "")) != "AES256"
            or not response.get("VersionId")
            or not response.get("ETag")
        ):
            raise CapsuleError("canonical authorization read lacks exact S3 evidence")
        return authorization

    def consume(
        self, capsule: dict[str, Any], authorization: dict[str, Any]
    ) -> dict[str, Any]:
        validate_execution_authorization(authorization, capsule)
        marker = production_consumption_marker(capsule, authorization)
        data = canonical_bytes(marker)
        with tempfile.NamedTemporaryFile("wb") as handle:
            handle.write(data)
            handle.flush()
            response = self._aws(
                self.consume_command(capsule, authorization, Path(handle.name), data)
            )
        version_id = str(response.get("VersionId", ""))
        etag = str(response.get("ETag", ""))
        checksum = str(response.get("ChecksumSHA256", ""))
        if (
            not version_id
            or not etag
            or checksum != _checksum(data)
            or str(response.get("ServerSideEncryption", "")) != "AES256"
        ):
            raise CapsuleError("atomic authorization consumption lacks exact S3 evidence")
        contract = authorization["contract"]
        receipt = {
            "schema": AUTHORIZATION_RECEIPT_SCHEMA,
            "status": "CONSUMED_EXACTLY_ONCE",
            "authorization_sha256": authorization["content_address"]["digest"],
            "capsule_sha256": capsule["content_address"]["digest"],
            "source_run_id": capsule["contract"]["source_identity"]["source_run_id"],
            "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
            "one_shot_consumed": True,
            "h100_authority": True,
            "issuer": contract["issuer"],
            "issuer_backend": contract["issuer_backend"],
            "issuer_account": contract["issuer_account"],
            "store_uri": contract["store_uri"],
            "consume_key": self.consumed_key(capsule, authorization),
            "consume_version_id": version_id,
            "consume_etag": etag,
            "consume_checksum_sha256": checksum,
            "consume_marker_sha256": content_sha256(marker),
        }
        return validate_authorization_receipt(receipt, authorization, capsule)

    def _verify_consumption(
        self,
        capsule: dict[str, Any],
        authorization: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        validate_authorization_receipt(receipt, authorization, capsule)
        marker = production_consumption_marker(capsule, authorization)
        expected_data = canonical_bytes(marker)
        with tempfile.TemporaryDirectory(prefix="pg1-consume-") as directory:
            target = Path(directory) / "consumed.json"
            response = self._aws(
                self.verify_consume_command(
                    self.consumed_key(capsule, authorization),
                    receipt["consume_version_id"],
                    target,
                )
            )
            observed_data = target.read_bytes()
        if (
            observed_data != expected_data
            or str(response.get("VersionId", "")) != receipt["consume_version_id"]
            or str(response.get("ETag", "")) != receipt["consume_etag"]
            or str(response.get("ChecksumSHA256", ""))
            != receipt["consume_checksum_sha256"]
            or receipt["consume_checksum_sha256"] != _checksum(expected_data)
            or str(response.get("ServerSideEncryption", "")) != "AES256"
        ):
            raise CapsuleError("canonical consumption marker body or identity changed")

    def consume_owner(
        self,
        capsule: dict[str, Any],
        authorization: dict[str, Any],
        receipt: dict[str, Any],
        *,
        owner: str,
    ) -> dict[str, Any]:
        self._verify_consumption(capsule, authorization, receipt)
        marker = {
            "schema": "pg1-native-production-owner-marker/v2",
            "owner": owner,
            "authorization_sha256": authorization["content_address"]["digest"],
            "base_receipt_sha256": content_sha256(receipt),
        }
        data = canonical_bytes(marker)
        with tempfile.NamedTemporaryFile("wb") as handle:
            handle.write(data)
            handle.flush()
            response = self._aws(
                self.owner_consume_command(
                    capsule, authorization, owner, Path(handle.name), data
                )
            )
        if (
            not response.get("VersionId")
            or not response.get("ETag")
            or str(response.get("ChecksumSHA256", "")) != _checksum(data)
            or str(response.get("ServerSideEncryption", "")) != "AES256"
        ):
            raise CapsuleError(f"{owner} atomic admission lacks exact S3 evidence")
        return {
            "status": "OWNER_CONSUMED_EXACTLY_ONCE",
            "owner": owner,
            "key": self.owner_key(capsule, authorization, owner),
            "version_id": str(response["VersionId"]),
            "etag": str(response["ETag"]),
            "checksum_sha256": str(response["ChecksumSHA256"]),
        }
