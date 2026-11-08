from __future__ import annotations

import base64
import gzip
import json
import subprocess
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_text_image_pixel_audit_launch as audit


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / audit.SPEC_PATH
USER_DATA = ROOT / audit.USER_DATA_PATH


def _zero_account() -> dict[str, list[object]]:
    return {
        "PARALLAX_AWS_instances": [],
        "PARALLAX_AWS_schedules": [],
        "Vast_instances": [],
        "Vast_volumes": [],
    }


def _inputs() -> tuple[dict, list[dict], list[dict]]:
    spec = audit.validate_spec(ROOT, SPEC)
    dependencies, images = audit._load_inputs(ROOT, spec)
    return spec, dependencies, images


def _worker(images: list[dict]) -> dict:
    rows = []
    for source in images:
        rows.append(
            {
                "item_id": source["item_id"],
                "target": source["target"],
                "source_index": source["source_index"],
                "key": source["key"],
                "version_id": source["version_id"],
                "PNG_bytes": source["bytes"],
                "PNG_sha256": source["sha256"],
                "semantic_item_ID_exact": True,
                "raw_28x28_SHA256": source["raw_28x28_sha256"],
                "raw_matches_first_worker": True,
                "reconstructed_decoded_RGB_SHA256": source["decoded_RGB_sha256"],
                "candidate_decoded_RGB_SHA256": source["decoded_RGB_sha256"],
                "first_worker_decoded_RGB_SHA256": source["decoded_RGB_sha256"],
                "reconstructed_candidate_first_worker_decoded_RGB_exact": True,
            }
        )
    return {
        "schema_version": 1,
        "status": "GREEN_LINUX_RUNTIME_PACKAGE_PIXEL_EXACT",
        "run_id": audit.RUN_ID,
        "pillow_version": "12.1.1",
        "exact_input_versions_read": 13,
        "exact_input_body_bytes": 33_621_125,
        "semantic_item_IDs_exact": 10,
        "raw_28x28_hashes_exact": 10,
        "decoded_RGB_hashes_exact": 10,
        "items": rows,
        "new_or_rewritten_image_objects": 0,
        "public_network_dataset_wheel_or_candidate_body_requests": 0,
        "Fashion_MNIST_test_or_confirmation_source_reads": 0,
        "confirmation_items_read": 0,
        "model_body_bytes": 0,
        "model_loads": 0,
        "simulator_starts": 0,
        "gpu_work": 0,
        "laptop_body_bytes": 0,
        "automatic_retry": False,
    }


def test_spec_binds_nonproducing_exact_thirteen_version_pixel_audit() -> None:
    spec, dependencies, images = _inputs()
    assert spec["input_envelope"]["exact_total_versions"] == 13
    assert spec["input_envelope"]["total_body_bytes"] == 33_621_125
    assert len(dependencies) == 3
    assert len(images) == 10
    assert sum(row["bytes"] for row in dependencies + images) == 33_621_125
    assert spec["audit_contract"]["new_or_rewritten_image_object_count"] == 0
    assert spec["authorizations"]["paid_action"] is False
    assert spec["authorizations"]["new_image_write"] is False
    assert spec["compute"]["gpu"] is False
    assert spec["compute"]["root_disk_gb"] == 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["input_envelope"].update(exact_total_versions=12), "input envelope"),
        (lambda row: row["audit_contract"].update(green_requires_all_ten_exact=False), "contract"),
        (lambda row: row["compute"].update(gpu=True), "compute contract"),
        (lambda row: row["cost_gate"].update(maximum_spot_USD_per_hour=1.0), "cost gate"),
        (lambda row: row["authorizations"].update(paid_action=True), "must not authorize"),
        (lambda row: row["forbidden"].pop(), "firewall"),
        (lambda row: row["stop_on"].pop(), "stop rules"),
    ],
)
def test_spec_refuses_input_contract_compute_cost_authority_or_firewall_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    packet = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(packet)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(audit.StageError, match=message):
        audit.validate_spec(ROOT, changed)


def test_source_receipt_drift_cannot_enter_control_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    receipt = json.loads(
        (ROOT / spec["bound_artifacts"]["first_worker_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    receipt["images"][0]["decoded_RGB_sha256"] = "0" * 64
    changed_receipt = tmp_path / "changed-receipt.json"
    changed_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    semantic = ROOT / spec["bound_artifacts"]["semantic_manifest"]["path"]

    def fake_bound(root, row, label):
        if label == "first_worker_receipt":
            return changed_receipt
        if label == "semantic_manifest":
            return semantic
        return ROOT / row["path"]

    monkeypatch.setattr(audit, "_bound", fake_bound)
    with pytest.raises(audit.StageError, match="candidate input changed"):
        audit._load_inputs(ROOT, spec)


def test_control_payload_contains_only_exact_preexisting_versions_and_hashes() -> None:
    _, dependencies, images = _inputs()
    packet = json.loads(audit._control_payload(dependencies, images))
    assert packet["run_id"] == audit.RUN_ID
    assert len(packet["d"]) == 3
    assert len(packet["i"]) == 10
    assert all(set(row) == {"n", "v", "b", "s", "p"} for row in packet["d"])
    assert all(set(row) == {"j", "t", "x", "v", "b", "s", "r", "q"} for row in packet["i"])
    assert "https://" not in json.dumps(packet)
    assert "manifest_image" not in json.dumps(packet)
    assert "t10k" not in json.dumps(packet)


def test_user_data_is_small_syntax_green_exact_version_only_nonproducing_and_self_terminating() -> None:
    _, dependencies, images = _inputs()
    subprocess.run(["bash", "-n", str(USER_DATA)], check=True)
    text = USER_DATA.read_text(encoding="utf-8")
    rendered = audit._render_user_data(ROOT, dependencies, images)
    assert len(rendered.encode("utf-8")) < 16_000
    assert 16_384 - len(rendered.encode("utf-8")) >= 2_000
    encoded = rendered.split("CONTROL_B64=", 1)[1].splitlines()[0]
    decoded = gzip.decompress(base64.b64decode(encoded))
    assert decoded == audit._control_payload(dependencies, images)
    assert "shutdown -h +10" in text
    assert "WAITING_FOR_EXTERNAL_REAPER_RELEASE" in text
    assert "AWS_MAX_ATTEMPTS=1" in text
    assert "--version-id \"$version\"" in text
    assert "--version-id \"$RESULT_VERSION\"" in text
    assert "--no-index --no-deps" in text
    assert 'deps={r["n"]:root/"dependencies"/r["p"] for r in control["d"]}' in text
    assert "r[5]" not in text
    assert "Image.Resampling.BICUBIC" in text
    assert "candidate_sha=hashlib.sha256(candidate.tobytes()).hexdigest()" in text
    assert "reconstructed_sha==candidate_sha==first_decoded" in text
    assert "new_or_rewritten_image_objects\":0" in text
    assert "aws s3api put-object" in text
    assert text.count("aws s3api put-object") == 1
    assert "public_network_dataset_wheel_or_candidate_body_requests\":0" in text
    assert "fashion-mnist.s3" not in text
    assert "files.pythonhosted.org" not in text
    assert "t10k-images" not in text
    assert "t10k-labels" not in text
    assert "docker" not in text.lower()
    assert "vastai" not in text.lower()
    assert "h100" not in text.lower()
    assert "model.safetensors" not in text


def test_exact_input_head_requires_version_size_checksum_encryption_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dependencies, _ = _inputs()
    row = dependencies[0]
    response = {
        "ContentLength": row["bytes"],
        "ChecksumSHA256": base64.b64encode(bytes.fromhex(row["sha256"])).decode("ascii"),
        "ServerSideEncryption": "AES256",
        "Metadata": audit._metadata(row),
    }
    calls = []

    def fake_aws_json(*args, **kwargs):
        calls.append(args)
        return response

    monkeypatch.setattr(audit.lifecycle, "_aws_json", fake_aws_json)
    result = audit._head_exact_input(row)
    assert result["metadata_exact"] is True
    assert result["body_read"] is False
    assert "--version-id" in calls[0]
    response["Metadata"] = {**audit._metadata(row), "source-integrity": "wrong"}
    with pytest.raises(audit.StageError, match="exact input HEAD failed"):
        audit._head_exact_input(row)


def test_all_thirteen_input_heads_are_body_free_and_aggregate_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dependencies, images = _inputs()
    calls = []

    def fake_head(row):
        calls.append(row["version_id"])
        return {"key": row["key"], "version_id": row["version_id"], "bytes": row["bytes"]}

    monkeypatch.setattr(audit, "_head_exact_input", fake_head)
    rows = audit._head_all_inputs(dependencies, images)
    assert len(rows) == len(calls) == 13
    assert sum(row["bytes"] for row in rows) == 33_621_125


def test_worker_green_requires_every_semantic_raw_and_decoded_comparison() -> None:
    _, _, images = _inputs()
    worker = _worker(images)
    assert len(audit._validate_worker(worker, images)) == 10

    missing = json.loads(json.dumps(worker))
    missing["items"].pop()
    with pytest.raises(audit.StageError, match="item set"):
        audit._validate_worker(missing, images)

    bad_raw = json.loads(json.dumps(worker))
    bad_raw["items"][0]["raw_28x28_SHA256"] = "0" * 64
    with pytest.raises(audit.StageError, match="comparison failed"):
        audit._validate_worker(bad_raw, images)

    bad_decoded = json.loads(json.dumps(worker))
    bad_decoded["items"][0]["candidate_decoded_RGB_SHA256"] = "0" * 64
    with pytest.raises(audit.StageError, match="comparison failed"):
        audit._validate_worker(bad_decoded, images)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exact_input_versions_read", 12),
        ("semantic_item_IDs_exact", 9),
        ("raw_28x28_hashes_exact", 9),
        ("decoded_RGB_hashes_exact", 9),
        ("new_or_rewritten_image_objects", 1),
        ("public_network_dataset_wheel_or_candidate_body_requests", 1),
        ("Fashion_MNIST_test_or_confirmation_source_reads", 1),
        ("confirmation_items_read", 1),
        ("model_body_bytes", 1),
        ("model_loads", 1),
        ("simulator_starts", 1),
        ("gpu_work", 1),
        ("laptop_body_bytes", 1),
        ("automatic_retry", True),
    ],
)
def test_worker_rejects_incomplete_or_firewall_positive_aggregate(field: str, value) -> None:
    _, _, images = _inputs()
    worker = _worker(images)
    worker[field] = value
    with pytest.raises(audit.StageError, match="aggregate or firewall"):
        audit._validate_worker(worker, images)


def test_result_head_and_download_are_exact_version_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sha = "a" * 64
    checksum = base64.b64encode(bytes.fromhex(sha)).decode("ascii")
    response = {
        "ContentLength": 100,
        "ChecksumSHA256": checksum,
        "ServerSideEncryption": "AES256",
        "Metadata": {"sha256": sha, "run-id": audit.RUN_ID},
    }
    monkeypatch.setattr(audit.lifecycle, "_aws_json", lambda *args, **kwargs: response)
    assert audit._head_result("key", "version", 100, sha)["metadata_exact"] is True

    calls = []

    def fake_aws(*args, **kwargs):
        calls.append(args)
        Path(args[-1]).write_bytes(b"x" * 100)

    monkeypatch.setattr(audit.lifecycle, "_aws", fake_aws)
    output = tmp_path / "result.json"
    audit._download_exact_small("key", "version", output, 100)
    assert output.stat().st_size == 100
    assert "--version-id" in calls[0]

    with pytest.raises(audit.StageError, match="exceeded"):
        audit._download_exact_small("key", "version", tmp_path / "too-big.json", 99)


def test_no_create_records_zero_mutation_and_zero_input_body_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audit.lifecycle, "_account_state", _zero_account)
    monkeypatch.setattr(audit.lifecycle, "_require_zero", lambda state: None)
    monkeypatch.setattr(
        audit.lifecycle,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(audit, "_role_compatibility", lambda: {"compatible": True})
    monkeypatch.setattr(audit, "_spot_price", lambda spec: 0.0084)
    monkeypatch.setattr(audit, "_require_prefix_never_used", lambda: None)
    monkeypatch.setattr(
        audit,
        "_head_all_inputs",
        lambda d, i: [
            {"key": row["key"], "version_id": row["version_id"], "bytes": row["bytes"]}
            for row in [*d, *i]
        ],
    )
    monkeypatch.setattr(audit.lifecycle, "_dry_run_instance", lambda spec, path: None)
    output = tmp_path / "no-create.json"
    packet = audit.no_create(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PI0_TEXT_IMAGE_PIXEL_AUDIT_NO_CREATE"
    assert packet["exact_input_HEAD_count"] == 13
    assert packet["exact_input_HEAD_bytes"] == 33_621_125
    assert packet["input_body_bytes_read"] == 0
    assert packet["destination_prefix_history"] == 0
    assert packet["paid_authorization"] is False
    assert packet["user_data"]["safe_margin_bytes"] >= 2_000
    assert set(packet["mutation"].values()) == {0}


def test_spot_price_gate_rejects_small_cpu_price_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _, _ = _inputs()
    monkeypatch.setattr(
        audit.lifecycle,
        "_aws_json",
        lambda *args, **kwargs: {"SpotPriceHistory": [{"SpotPrice": "0.0121"}]},
    )
    with pytest.raises(audit.StageError, match="exceeds"):
        audit._spot_price(spec)


def test_instance_construction_is_cpu_spot_ephemeral_and_8gb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, dependencies, images = _inputs()
    path = tmp_path / "user-data.sh"
    path.write_text(audit._render_user_data(ROOT, dependencies, images), encoding="utf-8")
    captured = {}

    def fake_aws_json(*arguments, **kwargs):
        captured["arguments"] = arguments
        return "i-test"

    monkeypatch.setattr(audit.lifecycle, "_aws_json", fake_aws_json)
    with audit._lifecycle_profile():
        assert audit.lifecycle._create_instance(spec, path) == "i-test"
    joined = " ".join(captured["arguments"])
    assert "t3a.small" in joined
    assert "MarketType=spot" in joined
    assert '"VolumeSize":8' in joined
    assert '"DeleteOnTermination":true' in joined
    assert "HttpTokens=required" in joined
    assert "H100" not in joined


def test_launch_arms_reaper_then_releases_then_terminates_before_result_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    monkeypatch.setattr(audit, "_validate_freeze", lambda *args: {})
    monkeypatch.setattr(audit.lifecycle, "_account_state", _zero_account)
    monkeypatch.setattr(audit.lifecycle, "_require_zero", lambda state: None)
    monkeypatch.setattr(audit.lifecycle, "_require_s3_safety", lambda: {})
    monkeypatch.setattr(audit, "_role_compatibility", lambda: {})
    monkeypatch.setattr(audit, "_spot_price", lambda spec: 0.0084)
    monkeypatch.setattr(audit, "_require_prefix_never_used", lambda: None)
    monkeypatch.setattr(audit, "_head_all_inputs", lambda d, i: [])
    monkeypatch.setattr(
        audit.lifecycle, "_create_instance", lambda spec, path: events.append("create") or "i-test"
    )
    monkeypatch.setattr(
        audit.lifecycle,
        "_arm_reaper",
        lambda instance, minutes: events.append("reaper") or "deadline",
    )

    def put(key, packet):
        assert events[-1] == "reaper"
        events.append("release")
        return {}

    monkeypatch.setattr(audit.lifecycle, "_put_json_object", put)
    monkeypatch.setattr(
        audit.lifecycle,
        "_wait_status",
        lambda *args, **kwargs: events.append("status")
        or {
            "status": "GREEN_LINUX_RUNTIME_PACKAGE_PIXEL_EXACT",
            "exit_code": 0,
            "result_key": f"{audit.STAGE_PREFIX}/pixel-audit-receipt.json",
            "result_version_id": "version",
            "result_bytes": 100,
            "result_sha256": "a" * 64,
            "exact_input_versions_read": 13,
            "exact_input_body_bytes": 33_621_125,
            "result_exact_version_readback": True,
            "new_or_rewritten_image_objects": 0,
            "confirmation_items_read": 0,
            "model_loads": 0,
            "simulator_starts": 0,
            "gpu_work": 0,
            "laptop_body_bytes": 0,
        },
    )
    monkeypatch.setattr(
        audit.lifecycle,
        "_terminate",
        lambda instance: events.append("terminate") if instance is not None else None,
    )

    def download(*args, **kwargs):
        assert events[-1] == "terminate"
        events.append("download-status")
        raise audit.StageError("stop after ordering assertion")

    monkeypatch.setattr(audit.lifecycle, "_download_small", download)
    monkeypatch.setattr(audit, "_cleanup_control", lambda: events.append("cleanup"))
    with pytest.raises(audit.StageError, match="ordering assertion"):
        audit.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", tmp_path / "output")
    assert events == [
        "create",
        "reaper",
        "release",
        "status",
        "terminate",
        "download-status",
        "cleanup",
    ]


def test_launch_failure_before_status_still_terminates_and_cleans_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    monkeypatch.setattr(audit, "_validate_freeze", lambda *args: {})
    monkeypatch.setattr(audit.lifecycle, "_account_state", _zero_account)
    monkeypatch.setattr(audit.lifecycle, "_require_zero", lambda state: None)
    monkeypatch.setattr(audit.lifecycle, "_require_s3_safety", lambda: {})
    monkeypatch.setattr(audit, "_role_compatibility", lambda: {})
    monkeypatch.setattr(audit, "_spot_price", lambda spec: 0.0084)
    monkeypatch.setattr(audit, "_require_prefix_never_used", lambda: None)
    monkeypatch.setattr(audit, "_head_all_inputs", lambda d, i: [])
    monkeypatch.setattr(
        audit.lifecycle, "_create_instance", lambda spec, path: events.append("create") or "i-test"
    )
    monkeypatch.setattr(
        audit.lifecycle, "_arm_reaper", lambda *args: events.append("reaper") or "deadline"
    )
    monkeypatch.setattr(
        audit.lifecycle, "_put_json_object", lambda *args: events.append("release") or {}
    )
    monkeypatch.setattr(
        audit.lifecycle,
        "_wait_status",
        lambda *args, **kwargs: (_ for _ in ()).throw(audit.StageError("worker failed")),
    )
    monkeypatch.setattr(
        audit.lifecycle,
        "_terminate",
        lambda instance: events.append("terminate") if instance is not None else None,
    )
    monkeypatch.setattr(audit, "_cleanup_control", lambda: events.append("cleanup"))
    with pytest.raises(audit.StageError, match="worker failed"):
        audit.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", tmp_path / "output")
    assert events == ["create", "reaper", "release", "terminate", "cleanup"]


def test_bad_or_unbound_freeze_cannot_authorize_paid_action(tmp_path: Path) -> None:
    bad = tmp_path / "bad-freeze.json"
    bad.write_text(
        json.dumps({"status": "FROZEN_ONE_SHOT_PI0_TEXT_IMAGE_PIXEL_AUDIT_AWS_CPU"}),
        encoding="utf-8",
    )
    with pytest.raises(audit.StageError, match="freeze is absent"):
        audit._validate_freeze(ROOT, SPEC, bad)


def test_launcher_does_not_import_producer_or_first_controller_and_has_literal_control_plane() -> None:
    text = (ROOT / audit.LAUNCHER_PATH).read_text(encoding="utf-8")
    assert "pi0_text_image_stage_launch" not in text
    assert "audit_pi0_text_image_stage_result" not in text
    assert 'PROFILE = "fuelos"' in text
    assert 'REGION = "us-east-1"' in text
    assert "subprocess.run([\"docker\"" not in text
    assert "vastai create" not in text.lower()
