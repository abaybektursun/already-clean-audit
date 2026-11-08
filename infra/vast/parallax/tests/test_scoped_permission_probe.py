from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from infra.vast.parallax.scoped_permission_probe import run


def test_permission_probe_stops_at_first_authorized_candidate_and_revokes(tmp_path: Path):
    returned = {
        "api": {
            "instance_write": {"api.instance": {}},
            "*": {"api.instances": {"GET": {}}},
        }
    }
    create = (
        "api-key created "
        + repr({"id": 91, "key": "temporary-secret-value-123", "permissions": returned})
        + "\nnull\n"
    )
    responses = [
        subprocess.CompletedProcess([], 0, '{"apikeys":[[]]}', ""),
        subprocess.CompletedProcess([], 0, create, ""),
        subprocess.CompletedProcess([], 0, '{"apikeys":[[]]}', ""),
    ]
    output = tmp_path / "probe.json"
    with (
        patch("infra.vast.parallax.scoped_permission_probe.subprocess.run", side_effect=responses),
        patch("infra.vast.parallax.scoped_permission_probe._stop_http_status", return_value=404),
        patch("infra.vast.parallax.scoped_permission_probe._revoke_key", return_value=True),
    ):
        packet = run(output)
    assert packet["status"] == "GREEN_CANDIDATE_FOUND"
    assert len(packet["candidates"]) == 1
    assert packet["candidates"][0]["candidate"] == "api.instance"
    assert packet["all_temporary_keys_revoked"] is True
    assert "temporary-secret-value-123" not in output.read_text()
    assert json.loads(output.read_text())["instances_created"] == 0
