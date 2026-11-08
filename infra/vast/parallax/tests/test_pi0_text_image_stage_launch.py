from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_text_image_stage_launch as stage


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / stage.SPEC_PATH
USER_DATA = ROOT / stage.USER_DATA_PATH


def _zero_account() -> dict[str, list[object]]:
    return {
        "PARALLAX_AWS_instances": [],
        "PARALLAX_AWS_schedules": [],
        "Vast_instances": [],
        "Vast_volumes": [],
    }


def _worker(spec: dict) -> dict:
    dependencies = []
    for index, expected in enumerate(spec["dependencies"]):
        sha256 = expected["integrity"] if expected["integrity_kind"] == "sha256" else f"{index + 1}" * 64
        dependencies.append(
            {
                "dependency_id": expected["dependency_id"],
                "url": expected["url"],
                "path": expected["path"],
                "key": stage._dependency_key(expected),
                "version_id": f"dependency-version-{index}",
                "bytes": expected["bytes"],
                "integrity_kind": expected["integrity_kind"],
                "source_integrity": expected["integrity"],
                "sha256": sha256,
                "checksum_sha256_base64": base64.b64encode(bytes.fromhex(sha256)).decode("ascii"),
                "server_side_encryption": "AES256",
                "exact_version_full_body_readback": True,
                "readback_sha256": sha256,
            }
        )
    images = []
    for index, expected in enumerate(spec["selected_items"]):
        sha256 = expected["manifest_image_sha256"]
        images.append(
            {
                "item_id": expected["item_id"],
                "target": expected["target"],
                "source_index": expected["source_index"],
                "key": stage._image_key(expected),
                "version_id": f"image-version-{index}",
                "bytes": expected["manifest_image_bytes"],
                "sha256": sha256,
                "raw_28x28_sha256": "a" * 64,
                "decoded_RGB_sha256": "b" * 64,
                "checksum_sha256_base64": base64.b64encode(bytes.fromhex(sha256)).decode("ascii"),
                "pillow_version": "12.1.1",
                "server_side_encryption": "AES256",
                "exact_version_full_body_readback": True,
                "readback_sha256": sha256,
                "readback_decoded_RGB_sha256": "b" * 64,
            }
        )
    return {
        "status": "GREEN_PI0_P2_TEXT_IMAGES_STAGED_TO_DURABLE_S3",
        "dependencies": dependencies,
        "images": images,
        "dependency_count": 3,
        "dependency_bytes": 33_492_935,
        "dependency_exact_version_readback_bytes": 33_492_935,
        "image_count": 10,
        "image_bytes": 126_855,
        "image_exact_version_readback_bytes": 126_855,
        "confirmation_items_read": 0,
        "model_body_bytes": 0,
        "model_loads": 0,
        "simulator_starts": 0,
        "gpu_work": 0,
        "laptop_dependency_or_image_bytes": 0,
        "automatic_retry": False,
    }


def test_spec_binds_exact_dependencies_transform_and_ten_frozen_images() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    assert spec["expected_dependency_count"] == 3
    assert spec["expected_dependency_bytes"] == 33_492_935
    assert spec["expected_selected_image_count"] == 10
    assert spec["expected_selected_image_bytes"] == 126_855
    assert [row["target"] for row in spec["selected_items"]] == [
        "ankle_boot",
        "sneaker",
        "bag",
        "sandal",
        "dress",
        "shirt",
        "pullover",
        "t_shirt",
        "trouser",
        "coat",
    ]
    assert spec["image_transform"]["exact_Pillow_version"] == "12.1.1"
    assert spec["compute"]["gpu"] is False
    assert spec["compute"]["root_disk_gb"] == 8
    assert spec["authorizations"]["paid_action"] is False
    assert spec["authorizations"]["selected_discovery_image_body_read"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["dependencies"][0].update(bytes=1), "dependency set"),
        (lambda row: row["selected_items"][0].update(source_index=1), "selected image set"),
        (lambda row: row["image_transform"].update(optimize=False), "transform"),
        (lambda row: row["compute"].update(gpu=True), "compute contract"),
        (lambda row: row["cost_gate"].update(maximum_spot_USD_per_hour=1.0), "cost gate"),
        (lambda row: row["authorizations"].update(paid_action=True), "must not authorize"),
        (lambda row: row["forbidden"].pop(), "safety invariants"),
    ],
)
def test_spec_refuses_dependency_selection_transform_compute_cost_authority_or_safety_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    packet = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(packet)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(stage.StageError, match=message):
        stage.validate_spec(ROOT, changed)


def test_selected_payload_contains_only_preoutput_identity_target_and_source_index() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    payload = json.loads(stage._selected_payload(spec))
    assert len(payload) == 10
    assert all(isinstance(row, list) and len(row) == 3 for row in payload)
    assert payload[0] == [
        "000d3fc36407d4c3411cf71df7c315efeaee79f854f0abd60d80103880483b8a",
        "ankle_boot",
        450,
    ]
    assert not any("manifest_image" in str(row) for row in payload)


def test_user_data_is_syntax_green_small_exact_cloud_only_and_self_terminating() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    subprocess.run(["bash", "-n", str(USER_DATA)], check=True)
    text = USER_DATA.read_text(encoding="utf-8")
    rendered = stage._render_user_data(ROOT, spec)
    assert len(rendered.encode("utf-8")) < 16_000
    assert 16_384 - len(rendered.encode("utf-8")) >= 900
    assert "shutdown -h +15" in text
    assert "WAITING_FOR_EXTERNAL_REAPER_RELEASE" in text
    assert "AWS_MAX_ATTEMPTS=1" in text
    assert "python3.12 -m pip install" in text
    assert "--no-index --no-deps" in text
    assert "Pillow.Image.Resampling.BICUBIC" not in text
    assert "Image.Resampling.BICUBIC" in text
    assert "optimize=True" in text
    assert "--version-id \"$version_id\"" in text
    assert text.count("exact_version_full_body_readback\":True") == 2
    assert text.index('rm -f "$file"') < text.index("aws s3api get-object")
    assert "train-images-idx3-ubyte.gz" in text
    assert "train-labels-idx1-ubyte.gz" in text
    assert "prefix_items=max(selected_indices)+1" in text
    assert "f.read(prefix_items*28*28)" in text
    assert "raw_images=f.read()" not in text
    assert "raw_labels=f.read()" not in text
    assert "t10k-images" not in text
    assert "t10k-labels" not in text
    assert "confirmation" in text
    assert "docker" not in text.lower()
    assert "vastai" not in text.lower()
    assert "h100" not in text.lower()
    assert "model.safetensors" not in text


def test_source_head_is_body_free_direct_and_requires_frozen_transport_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    row = spec["dependencies"][0]
    monkeypatch.setattr(
        stage,
        "_https_head",
        lambda url: (200, {"content-length": str(row["bytes"]), "etag": f'"{row["head_etag"]}"'}),
    )
    result = stage._head_source(row)
    assert result["bytes_exact"] is True
    assert result["transport_etag_exact"] is True
    assert result["head_etag_is_source_integrity"] is False
    assert result["redirects_followed"] == 0
    assert result["body_downloaded"] is False

    monkeypatch.setattr(
        stage,
        "_https_head",
        lambda url: (200, {"content-length": str(row["bytes"]), "etag": '"wrong"'}),
    )
    with pytest.raises(stage.StageError, match="transport ETag drift"):
        stage._head_source(row)


def test_source_head_rejects_unfrozen_authority() -> None:
    changed = dict(stage.DEPENDENCIES[0], url="https://example.com/train-images-idx3-ubyte.gz")
    with pytest.raises(stage.StageError, match="escaped"):
        stage._head_source(changed)


def test_all_thirteen_durable_keys_are_run_scoped_and_collision_free() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    keys = [stage._dependency_key(row) for row in spec["dependencies"]]
    keys.extend(stage._image_key(row) for row in spec["selected_items"])
    assert len(keys) == len(set(keys)) == 13
    assert all(key.startswith(f"{stage.STAGE_PREFIX}/") for key in keys)


def test_worker_gate_requires_every_exact_body_readback_and_byte_exact_image() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    worker = _worker(spec)
    dependencies, images = stage._validate_worker(spec, worker)
    assert len(dependencies) == 3
    assert len(images) == 10

    missing = json.loads(json.dumps(worker))
    missing["images"].pop()
    with pytest.raises(stage.StageError, match="image set"):
        stage._validate_worker(spec, missing)

    bad_dependency = json.loads(json.dumps(worker))
    bad_dependency["dependencies"][0]["readback_sha256"] = "0" * 64
    with pytest.raises(stage.StageError, match="dependency integrity/readback"):
        stage._validate_worker(spec, bad_dependency)

    bad_image = json.loads(json.dumps(worker))
    bad_image["images"][0]["sha256"] = "0" * 64
    bad_image["images"][0]["readback_sha256"] = "0" * 64
    bad_image["images"][0]["checksum_sha256_base64"] = base64.b64encode(
        bytes.fromhex("0" * 64)
    ).decode("ascii")
    with pytest.raises(stage.StageError, match="exact image reproduction"):
        stage._validate_worker(spec, bad_image)

    bad_decoded = json.loads(json.dumps(worker))
    bad_decoded["images"][0]["readback_decoded_RGB_sha256"] = "c" * 64
    with pytest.raises(stage.StageError, match="exact image reproduction"):
        stage._validate_worker(spec, bad_decoded)


def test_worker_gate_rejects_confirmation_model_simulator_gpu_or_retry_counters() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    for field, value in (
        ("confirmation_items_read", 1),
        ("model_body_bytes", 1),
        ("model_loads", 1),
        ("simulator_starts", 1),
        ("gpu_work", 1),
        ("laptop_dependency_or_image_bytes", 1),
        ("automatic_retry", True),
    ):
        worker = _worker(spec)
        worker[field] = value
        with pytest.raises(stage.StageError, match="aggregate"):
            stage._validate_worker(spec, worker)


def test_exact_head_requires_version_size_checksum_encryption_and_all_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    row = _worker(spec)["images"][0]
    metadata = {
        "sha256": row["sha256"],
        "item-id": row["item_id"],
        "source-index": str(row["source_index"]),
        "decoded-rgb-sha256": row["decoded_RGB_sha256"],
    }
    response = {
        "ContentLength": row["bytes"],
        "ChecksumSHA256": row["checksum_sha256_base64"],
        "ServerSideEncryption": "AES256",
        "Metadata": metadata,
    }
    monkeypatch.setattr(stage.lifecycle, "_aws_json", lambda *args, **kwargs: response)
    assert stage._head_exact(row["key"], row["version_id"], row, metadata)["metadata_exact"]

    response["Metadata"] = {**metadata, "source-index": "wrong"}
    with pytest.raises(stage.StageError, match="immutable S3"):
        stage._head_exact(row["key"], row["version_id"], row, metadata)


def test_durable_verification_heads_all_thirteen_exact_versions_without_body_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    dependencies, images = stage._validate_worker(spec, _worker(spec))
    calls = []

    def head(key, version_id, row, metadata):
        calls.append((key, version_id, metadata))
        return {"key": key, "version_id": version_id, "bytes": row["bytes"], "metadata_exact": True}

    monkeypatch.setattr(stage, "_head_exact", head)
    durable_dependencies, durable_images = stage._verify_durable(dependencies, images)
    assert len(durable_dependencies) == 3
    assert len(durable_images) == 10
    assert len(calls) == 13
    assert all(row["byte_exact_to_frozen_cloud_task_cache"] is True for row in durable_images)


def test_no_create_records_zero_mutation_and_no_dependency_or_image_body_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stage.lifecycle, "_account_state", _zero_account)
    monkeypatch.setattr(stage.lifecycle, "_require_zero", lambda state: None)
    monkeypatch.setattr(
        stage.lifecycle,
        "_require_s3_safety",
        lambda: {"versioning": "Enabled", "public_access_blocked": True, "encryption": "AES256"},
    )
    monkeypatch.setattr(stage, "_role_compatibility", lambda: {"compatible": True, "policy_mutation": False})
    monkeypatch.setattr(stage, "_spot_price", lambda spec: 0.0084)
    monkeypatch.setattr(stage, "_require_absence", lambda spec: [f"absent-{index}" for index in range(13)])
    monkeypatch.setattr(
        stage,
        "_head_all_sources",
        lambda spec: [
            {"dependency_id": row["dependency_id"], "body_downloaded": False}
            for row in spec["dependencies"]
        ],
    )
    monkeypatch.setattr(stage.lifecycle, "_dry_run_instance", lambda spec, path: None)
    output = tmp_path / "no-create.json"
    packet = stage.no_create(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PI0_P2_TEXT_IMAGE_STAGE_NO_CREATE"
    assert packet["source_HEAD_count"] == 3
    assert packet["source_body_bytes_read"] == 0
    assert packet["absent_key_count"] == 13
    assert packet["user_data"]["safe_margin_bytes"] >= 900
    assert packet["paid_authorization"] is False
    assert set(packet["mutation"].values()) == {0}


def test_spot_price_gate_rejects_even_small_cpu_price_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    monkeypatch.setattr(
        stage.lifecycle,
        "_aws_json",
        lambda *args, **kwargs: {"SpotPriceHistory": [{"SpotPrice": "0.0121"}]},
    )
    with pytest.raises(stage.StageError, match="exceeds"):
        stage._spot_price(spec)


def test_instance_construction_is_cpu_spot_ephemeral_and_8gb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    user_data = tmp_path / "user-data.sh"
    user_data.write_text(stage._render_user_data(ROOT, spec), encoding="utf-8")
    captured = {}

    def fake_aws_json(*arguments, **kwargs):
        captured["arguments"] = arguments
        return "i-test"

    monkeypatch.setattr(stage.lifecycle, "_aws_json", fake_aws_json)
    with stage._lifecycle_profile():
        assert stage.lifecycle._create_instance(spec, user_data) == "i-test"
    joined = " ".join(captured["arguments"])
    assert "t3a.small" in joined
    assert "MarketType=spot" in joined
    assert '"VolumeSize":8' in joined
    assert '"DeleteOnTermination":true' in joined
    assert "HttpTokens=required" in joined
    assert "H100" not in joined


def test_launch_arms_reaper_before_release_and_terminates_at_status_before_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    monkeypatch.setattr(stage, "_validate_freeze", lambda *args: {})
    monkeypatch.setattr(stage.lifecycle, "_account_state", _zero_account)
    monkeypatch.setattr(stage.lifecycle, "_require_zero", lambda state: None)
    monkeypatch.setattr(stage.lifecycle, "_require_s3_safety", lambda: {})
    monkeypatch.setattr(stage, "_role_compatibility", lambda: {})
    monkeypatch.setattr(stage, "_spot_price", lambda spec: 0.0084)
    monkeypatch.setattr(stage, "_require_absence", lambda spec: [])
    monkeypatch.setattr(stage, "_head_all_sources", lambda spec: [])
    monkeypatch.setattr(stage.lifecycle, "_create_instance", lambda spec, path: events.append("create") or "i-test")
    monkeypatch.setattr(
        stage.lifecycle,
        "_arm_reaper",
        lambda instance, minutes: events.append("reaper") or "deadline",
    )

    def put(key, packet):
        assert events[-1] == "reaper"
        events.append("release")
        return {}

    monkeypatch.setattr(stage.lifecycle, "_put_json_object", put)
    monkeypatch.setattr(
        stage.lifecycle,
        "_wait_status",
        lambda *args, **kwargs: events.append("status")
        or {
            "status": "GREEN_PI0_P2_TEXT_IMAGES_STAGED_TO_DURABLE_S3",
            "exit_code": 0,
            "dependency_count": 3,
            "image_count": 10,
        },
    )
    monkeypatch.setattr(
        stage.lifecycle,
        "_terminate",
        lambda instance: events.append("terminate") if instance is not None else None,
    )

    def download(*args, **kwargs):
        assert events[-1] == "terminate"
        events.append("download")
        raise stage.StageError("stop after ordering assertion")

    monkeypatch.setattr(stage.lifecycle, "_download_small", download)
    monkeypatch.setattr(stage, "_cleanup_control", lambda: events.append("cleanup"))
    with pytest.raises(stage.StageError, match="ordering assertion"):
        stage.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", tmp_path / "output")
    assert events == ["create", "reaper", "release", "status", "terminate", "download", "cleanup"]


def test_launch_failure_before_status_still_terminates_and_cleans_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    monkeypatch.setattr(stage, "_validate_freeze", lambda *args: {})
    monkeypatch.setattr(stage.lifecycle, "_account_state", _zero_account)
    monkeypatch.setattr(stage.lifecycle, "_require_zero", lambda state: None)
    monkeypatch.setattr(stage.lifecycle, "_require_s3_safety", lambda: {})
    monkeypatch.setattr(stage, "_role_compatibility", lambda: {})
    monkeypatch.setattr(stage, "_spot_price", lambda spec: 0.0084)
    monkeypatch.setattr(stage, "_require_absence", lambda spec: [])
    monkeypatch.setattr(stage, "_head_all_sources", lambda spec: [])
    monkeypatch.setattr(stage.lifecycle, "_create_instance", lambda spec, path: events.append("create") or "i-test")
    monkeypatch.setattr(stage.lifecycle, "_arm_reaper", lambda *args: events.append("reaper") or "deadline")
    monkeypatch.setattr(stage.lifecycle, "_put_json_object", lambda *args: events.append("release") or {})
    monkeypatch.setattr(
        stage.lifecycle,
        "_wait_status",
        lambda *args, **kwargs: (_ for _ in ()).throw(stage.StageError("worker failed")),
    )
    monkeypatch.setattr(
        stage.lifecycle,
        "_terminate",
        lambda instance: events.append("terminate") if instance is not None else None,
    )
    monkeypatch.setattr(stage, "_cleanup_control", lambda: events.append("cleanup"))
    with pytest.raises(stage.StageError, match="worker failed"):
        stage.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", tmp_path / "output")
    assert events == ["create", "reaper", "release", "terminate", "cleanup"]


def test_bad_or_unbound_freeze_cannot_authorize_paid_action(tmp_path: Path) -> None:
    bad = tmp_path / "bad-freeze.json"
    bad.write_text(
        json.dumps({"status": "FROZEN_ONE_SHOT_PI0_P2_TEXT_IMAGE_STAGE_AWS_CPU"}),
        encoding="utf-8",
    )
    with pytest.raises(stage.StageError, match="freeze is absent"):
        stage._validate_freeze(ROOT, SPEC, bad)
