from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from infra.vast.parallax.lifecycle_supervisor import LifecycleError
from infra.vast.parallax.scoped_key_setup import create_scoped_key


def test_scoped_key_secret_is_mode_0600_and_never_enters_receipt(tmp_path: Path):
    permissions = tmp_path / "permissions.json"
    exact = {
        "api": {
            "instance_read": {"api.instances": {}},
            "instance_write": {"api.instance": {}},
        }
    }
    permissions.write_text(json.dumps(exact))
    secret = tmp_path / "secret.key"
    receipt = tmp_path / "receipt.json"
    responses = [
        subprocess.CompletedProcess([], 0, "[]", ""),
        subprocess.CompletedProcess([], 0, json.dumps({
            "id": 77, "key": "vast-secret-value-123", "permissions": exact,
        }), ""),
    ]
    with patch("infra.vast.parallax.scoped_key_setup.subprocess.run", side_effect=responses):
        packet = create_scoped_key(
            name="parallax-lifecycle-v1",
            permissions_file=permissions,
            secret_file=secret,
            receipt_file=receipt,
        )
    assert secret.read_text().strip() == "vast-secret-value-123"
    assert secret.stat().st_mode & 0o777 == 0o600
    assert "vast-secret-value-123" not in receipt.read_text()
    assert packet["api_key_id"] == 77 and packet["secret_logged"] is False


def test_scoped_key_parses_vast_prose_without_printing_secret(tmp_path: Path):
    exact = {
        "api": {
            "instance_read": {"api.instances": {}},
            "instance_write": {"api.instance": {}},
        }
    }
    permissions = tmp_path / "permissions.json"
    permissions.write_text(json.dumps(exact))
    response = f"api-key created {dict(id=88, key='prose-secret-value-123', permissions=exact)!r}\nnull\n"
    with patch("infra.vast.parallax.scoped_key_setup.subprocess.run", side_effect=[
        subprocess.CompletedProcess([], 0, '{"apikeys": [[]]}', ""),
        subprocess.CompletedProcess([], 0, response, ""),
    ]):
        packet = create_scoped_key(
            name="parallax-lifecycle-v1",
            permissions_file=permissions,
            secret_file=tmp_path / "secret.key",
            receipt_file=tmp_path / "receipt.json",
        )
    assert packet["api_key_id"] == 88
    assert "prose-secret-value-123" not in (tmp_path / "receipt.json").read_text()


def test_scoped_key_refuses_broader_permissions_before_vast_mutation(tmp_path: Path):
    permissions = tmp_path / "permissions.json"
    permissions.write_text(json.dumps({"api": {"instance_write": {}, "billing_read": {}}}))
    with (
        patch("infra.vast.parallax.scoped_key_setup.subprocess.run") as run,
        pytest.raises(LifecycleError, match="minimal and exact"),
    ):
        create_scoped_key(
            name="bad",
            permissions_file=permissions,
            secret_file=tmp_path / "secret",
            receipt_file=tmp_path / "receipt",
        )
    run.assert_not_called()


def test_scoped_key_rolls_back_if_redacted_receipt_cannot_be_written(tmp_path: Path):
    permissions = tmp_path / "permissions.json"
    exact = {
        "api": {
            "instance_read": {"api.instances": {}},
            "instance_write": {"api.instance": {}},
        }
    }
    permissions.write_text(json.dumps(exact))
    secret = tmp_path / "secret.key"
    responses = [
        subprocess.CompletedProcess([], 0, "[]", ""),
        subprocess.CompletedProcess([], 0, json.dumps({
            "id": 77, "key": "vast-secret-value-123", "permissions": exact,
        }), ""),
        subprocess.CompletedProcess([], 0, '{"success":true}', ""),
    ]
    with (
        patch("infra.vast.parallax.scoped_key_setup.subprocess.run", side_effect=responses) as run,
        patch("infra.vast.parallax.scoped_key_setup.atomic_json", side_effect=OSError("disk full")),
        pytest.raises(LifecycleError, match="rolled back"),
    ):
        create_scoped_key(
            name="parallax-lifecycle-v1",
            permissions_file=permissions,
            secret_file=secret,
            receipt_file=tmp_path / "receipt.json",
        )
    assert not secret.exists()
    assert run.call_args_list[-1].args[0] == ["vastai", "delete", "api-key", "77", "--raw"]
