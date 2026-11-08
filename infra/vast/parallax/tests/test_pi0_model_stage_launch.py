from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_model_stage_launch as stage


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
    sources = stage._source_map(spec)
    rows = []
    for source_row in spec["files"]:
        sha256 = (
            source_row["integrity"]
            if source_row["integrity_kind"] in {"sha256", "lfs_sha256"}
            else "f" * 64
        )
        source = sources[source_row["source_id"]]
        rows.append(
            {
                "source_id": source_row["source_id"],
                "repo": source["repo"],
                "revision": source["revision"],
                "path": source_row["path"],
                "key": stage._temporary_key(source_row),
                "version_id": f"version-{len(rows)}",
                "bytes": source_row["bytes"],
                "integrity_kind": source_row["integrity_kind"],
                "source_integrity": source_row["integrity"],
                "sha256": sha256,
                "full_sha256_base64": base64.b64encode(bytes.fromhex(sha256)).decode("ascii"),
                "s3_checksum_sha256": "service-checksum",
                "s3_checksum_type": "FULL_OBJECT",
                "server_side_encryption": "AES256",
                "exact_version_full_body_readback": True,
                "readback_sha256": sha256,
            }
        )
    return {
        "status": "GREEN_PI0_P2_MODELS_STAGED_TO_DURABLE_S3",
        "files": rows,
        "file_count": 13,
        "total_bytes": 21_017_202_151,
        "source_body_bytes": 21_017_202_151,
        "s3_exact_version_readback_bytes": 21_017_202_151,
        "scratch_peak_model_bytes": 14_005_618_584,
        "model_loads": 0,
        "simulator_starts": 0,
        "gpu_work": 0,
        "laptop_model_bytes": 0,
        "automatic_retry": False,
    }


def test_spec_is_exact_public_cpu_only_and_s3_durable() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    assert [(row["repo"], row["revision"]) for row in spec["sources"]] == [
        ("lerobot/pi0_base", "25c379b52ba2ff8788cab921758a3cc3fe3f77f2"),
        (
            "lerobot/pi0_libero_finetuned_v044",
            "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
        ),
    ]
    assert spec["expected_file_count"] == 13
    assert spec["expected_total_bytes"] == 21_017_202_151
    assert spec["compute"]["gpu"] is False
    assert spec["compute"]["root_disk_gb"] == 30
    assert spec["compute"]["sequential_single_file_scratch"] is True
    assert spec["cost_gate"]["maximum_one_shot_infrastructure_USD"] == 0.03
    assert spec["authorizations"]["paid_action"] is False
    assert spec["authorizations"]["model_body_read"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["files"][0].update(bytes=2), "audited exact inventory"),
        (lambda row: row["compute"].update(gpu=True), "compute contract"),
        (lambda row: row["cost_gate"].update(maximum_spot_USD_per_hour=1.0), "cost gate"),
        (lambda row: row["authorizations"].update(paid_action=True), "must not authorize"),
        (lambda row: row["sources"][0].update(revision="main"), "identity changed"),
    ],
)
def test_spec_refuses_source_compute_cost_or_authority_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    packet = json.loads(SPEC.read_text(encoding="utf-8"))
    mutation(packet)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(stage.StageError, match=message):
        stage.validate_spec(ROOT, changed)


def test_user_data_is_small_syntax_green_sequential_and_self_terminating() -> None:
    subprocess.run(["bash", "-n", str(USER_DATA)], check=True)
    text = USER_DATA.read_text(encoding="utf-8")
    assert len(stage._render_user_data(ROOT).encode("utf-8")) < 16_384
    assert text.count("stage_one ") == 13
    assert "shutdown -h +90" in text
    assert "WAITING_FOR_EXTERNAL_REAPER_RELEASE" in text
    assert "AWS_MAX_ATTEMPTS=1" in text
    assert "--speed-limit 8388608 --speed-time 180" in text
    assert 'aws s3 cp "$file"' in text
    assert "--version-id \"$version_id\"" in text
    assert "exact_version_readback_bytes=21017202151" in text
    assert text.index('rm -f "$file"') < text.index("aws s3api get-object")
    assert "docker" not in text.lower()
    assert "vastai" not in text.lower()
    assert "dnf install" not in text.lower()
    assert "pip install" not in text.lower()
    assert "bearer" not in text.lower()
    assert "hf_token" not in text.lower()
    assert "H100" not in text


def test_user_data_stage_rows_equal_the_frozen_inventory_exactly() -> None:
    import shlex

    spec = stage.validate_spec(ROOT, SPEC)
    sources = stage._source_map(spec)
    observed = []
    for line in USER_DATA.read_text(encoding="utf-8").splitlines():
        if not line.startswith("stage_one "):
            continue
        _, source_id, repo, revision, path, size, integrity_kind, integrity = shlex.split(line)
        observed.append(
            {
                "source_id": source_id,
                "repo": repo,
                "revision": revision,
                "path": path,
                "bytes": int(size),
                "integrity_kind": integrity_kind,
                "integrity": integrity,
            }
        )
    expected = [
        {
            **row,
            "repo": sources[row["source_id"]]["repo"],
            "revision": sources[row["source_id"]]["revision"],
        }
        for row in spec["files"]
    ]
    assert observed == expected


def test_regular_git_source_head_follows_only_exact_cache_and_checks_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    source = stage._source_map(spec)["robot_positive_control"]
    row = next(row for row in spec["files"] if row["path"] == ".gitattributes")
    calls = []

    def head(url: str):
        calls.append(url)
        if len(calls) == 1:
            return 307, {
                "x-repo-commit": source["revision"],
                "x-linked-etag": f'"{row["integrity"]}"',
                "location": (
                    f"/api/resolve-cache/models/{source['repo']}/{source['revision']}/"
                    ".gitattributes?etag=exact"
                ),
            }
        return 200, {"content-length": str(row["bytes"]), "etag": f'"{row["integrity"]}"'}

    monkeypatch.setattr(stage, "_https_head", head)
    result = stage._head_source(source, row)
    assert result["revision_exact"] is True
    assert result["bytes_exact"] is True
    assert result["linked_integrity_exact"] is True
    assert result["redirects_followed"] == 1
    assert result["body_downloaded"] is False


def test_lfs_source_head_closes_from_linked_size_and_sha_without_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    source = stage._source_map(spec)["pi0_base"]
    row = next(row for row in spec["files"] if row["source_id"] == "pi0_base" and row["path"] == "model.safetensors")
    monkeypatch.setattr(
        stage,
        "_https_head",
        lambda url: (
            302,
            {
                "x-repo-commit": source["revision"],
                "x-linked-size": str(row["bytes"]),
                "x-linked-etag": f'"{row["integrity"]}"',
                "location": "https://us.aws.cdn.hf.co/signed",
            },
        ),
    )
    result = stage._head_source(source, row)
    assert result["redirects_followed"] == 0
    assert result["linked_integrity_exact"] is True
    assert result["body_downloaded"] is False


def test_source_head_rejects_revision_integrity_size_and_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    source = stage._source_map(spec)["robot_positive_control"]
    row = next(row for row in spec["files"] if row["path"] == ".gitattributes")
    monkeypatch.setattr(
        stage,
        "_https_head",
        lambda url: (307, {"x-repo-commit": "wrong", "x-linked-etag": f'"{row["integrity"]}"'}),
    )
    with pytest.raises(stage.StageError, match="revision drift"):
        stage._head_source(source, row)

    calls = 0

    def escaped(url: str):
        nonlocal calls
        calls += 1
        return (
            307,
            {
                "x-repo-commit": source["revision"],
                "x-linked-etag": f'"{row["integrity"]}"',
                "location": "https://example.com/escape",
            },
        )

    monkeypatch.setattr(stage, "_https_head", escaped)
    with pytest.raises(stage.StageError, match="authority"):
        stage._head_source(source, row)
    assert calls == 1


def test_all_thirteen_durable_keys_are_run_scoped_and_collision_free() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    keys = [stage._temporary_key(row) for row in spec["files"]]
    assert len(keys) == len(set(keys)) == 13
    assert all(key.startswith(f"{stage.STAGE_PREFIX}/files/") for key in keys)


def test_worker_gate_requires_every_source_identity_and_full_version_readback() -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    worker = _worker(spec)
    assert len(stage._validate_worker(spec, worker)) == 13

    missing = json.loads(json.dumps(worker))
    missing["files"].pop()
    with pytest.raises(stage.StageError, match="file set"):
        stage._validate_worker(spec, missing)

    bad_readback = json.loads(json.dumps(worker))
    bad_readback["files"][0]["readback_sha256"] = "0" * 64
    with pytest.raises(stage.StageError, match="integrity/readback"):
        stage._validate_worker(spec, bad_readback)

    bad_sha = json.loads(json.dumps(worker))
    bad_sha["files"][0]["sha256"] = "0" * 64
    bad_sha["files"][0]["readback_sha256"] = "0" * 64
    bad_sha["files"][0]["full_sha256_base64"] = base64.b64encode(bytes.fromhex("0" * 64)).decode("ascii")
    with pytest.raises(stage.StageError, match="differs from frozen source"):
        stage._validate_worker(spec, bad_sha)


def test_exact_head_accepts_full_small_and_composite_large_service_checksums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    worker = _worker(spec)
    small = worker["files"][0]
    large = next(row for row in worker["files"] if row["bytes"] > 5_000_000_000)

    def response(row, checksum, checksum_type):
        return {
            "ContentLength": row["bytes"],
            "ChecksumSHA256": checksum,
            "ChecksumType": checksum_type,
            "ServerSideEncryption": "AES256",
            "Metadata": {"sha256": row["sha256"]},
        }

    expected = base64.b64encode(bytes.fromhex(small["sha256"])).decode("ascii")
    monkeypatch.setattr(stage.lifecycle, "_aws_json", lambda *args, **kwargs: response(small, expected, "FULL_OBJECT"))
    assert stage._head_exact(small["key"], small["version_id"], small)["s3_checksum_type"] == "FULL_OBJECT"

    monkeypatch.setattr(stage.lifecycle, "_aws_json", lambda *args, **kwargs: response(large, "Y29tcG9zaXRl-27", "COMPOSITE"))
    assert stage._head_exact(large["key"], large["version_id"], large)["s3_checksum_type"] == "COMPOSITE"

    monkeypatch.setattr(stage.lifecycle, "_aws_json", lambda *args, **kwargs: response(large, "", "COMPOSITE"))
    with pytest.raises(stage.StageError, match="immutable S3 verification failed"):
        stage._head_exact(large["key"], large["version_id"], large)


def test_durable_verification_binds_every_exact_run_key_and_version_without_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    rows = stage._validate_worker(spec, _worker(spec))
    calls = []

    def head(key, version, row):
        calls.append((key, version))
        return {
            "key": key,
            "version_id": version,
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "full_sha256_base64_from_exact_version_readback": "exact",
            "s3_service_checksum_sha256": "service",
            "s3_checksum_type": "FULL_OBJECT",
            "server_side_encryption": "AES256",
        }

    monkeypatch.setattr(stage, "_head_exact", head)
    durable = stage._verify_durable_files(spec, rows)
    assert len(durable) == 13
    assert calls == [(row["key"], row["version_id"]) for row in rows]
    assert all(row["exact_version_full_body_readback"] is True for row in durable)
    assert all(row["content_addressed_identity"] == row["sha256"] for row in durable)


def test_no_create_records_zero_mutation_and_no_model_body_read(
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
            {"source_id": row["source_id"], "path": row["path"], "body_downloaded": False}
            for row in spec["files"]
        ],
    )
    monkeypatch.setattr(stage.lifecycle, "_dry_run_instance", lambda spec, path: None)
    output = tmp_path / "no-create.json"
    packet = stage.no_create(ROOT, SPEC, output)
    assert packet["status"] == "GREEN_PI0_P2_MODEL_STAGE_NO_CREATE"
    assert packet["source_HEAD_count"] == 13
    assert packet["source_body_bytes_read"] == 0
    assert packet["absent_key_count"] == 13
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


def test_instance_construction_is_cpu_spot_ephemeral_and_30gb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = stage.validate_spec(ROOT, SPEC)
    user_data = tmp_path / "user-data.sh"
    user_data.write_text(stage._render_user_data(ROOT), encoding="utf-8")
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
    assert '"VolumeSize":30' in joined
    assert '"DeleteOnTermination":true' in joined
    assert "HttpTokens=required" in joined
    assert "H100" not in joined


def test_launch_arms_reaper_before_release_and_cleans_up_on_worker_failure(
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
        lambda *args, **kwargs: (_ for _ in ()).throw(stage.StageError("worker failed")),
    )
    monkeypatch.setattr(stage.lifecycle, "_terminate", lambda instance: events.append("terminate"))
    monkeypatch.setattr(stage, "_cleanup_control", lambda: events.append("cleanup"))
    with pytest.raises(stage.StageError, match="worker failed"):
        stage.launch(ROOT, SPEC, tmp_path / "unused-freeze.json", tmp_path / "output")
    assert events == ["create", "reaper", "release", "terminate", "cleanup"]


def test_bad_or_unbound_freeze_cannot_authorize_paid_action(tmp_path: Path) -> None:
    bad = tmp_path / "bad-freeze.json"
    bad.write_text(json.dumps({"status": "FROZEN_ONE_SHOT_PI0_P2_MODEL_STAGE_AWS_CPU"}), encoding="utf-8")
    with pytest.raises(stage.StageError, match="freeze is absent"):
        stage._validate_freeze(ROOT, SPEC, bad)
