from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from infra.vast.parallax import ephemeral_optimizer_scale_job as job
from infra.vast.parallax import ephemeral_optimizer_scale_launch as launch
from infra.vast.parallax import ephemeral_s3_launch as s3_launch
from infra.vast.parallax import optimizer_scale_no_create_rehearsal as no_create


ROOT = Path(__file__).resolve().parents[4]
HYDRATION = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_h100_s3_hydration_job_v1.json"
)
RECEIPT = ROOT / (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_training_optimizer_scale_transfer_v1/receipt.json"
)
LAUNCH_SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_optimizer_scale_s3_launch_v3.json"
)
QUARANTINE = ROOT / (
    "studies/lineage/cross-lineage/config/parallax_h100_machine_quarantine_v1.json"
)


def _url(row: dict[str, object]) -> str:
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "test/20260718/us-east-1/s3/aws4_request",
            "X-Amz-Date": "20260718T000000Z",
            "X-Amz-Expires": "1800",
            "X-Amz-SignedHeaders": "host",
            "versionId": row["version_id"],
            "X-Amz-Signature": "0" * 64,
        }
    )
    return f"https://{job.BUCKET}.s3.{job.REGION}.amazonaws.com/{row['key']}?{query}"


def _job_contract() -> dict[str, object]:
    hydration = json.loads(HYDRATION.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    model = [{**row, "url": _url(row)} for row in hydration["files"]]
    bundle = {
        "bucket": job.BUCKET,
        "region": job.REGION,
        "key": (
            "parallax/input-staging/sha256/"
            f"{job.EXPECTED_BUNDLE_SHA256}/parallax-training-optimizer-scale-v1.tar.gz"
        ),
        "version_id": "Ohfs88ITOuoRFZxeTfVCfCtx1aRAHRfs",
        "bytes": job.EXPECTED_BUNDLE_BYTES,
        "sha256": job.EXPECTED_BUNDLE_SHA256,
        "entries": receipt["entries"],
    }
    bundle["url"] = _url(bundle)
    return {
        "schema_version": 1,
        "status": "FROZEN_H100_OPTIMIZER_SCALE_JOB",
        "stage": "training_optimizer_scale_microgate",
        "provider": "vast.ai",
        "instance_id": 123,
        "storage": {
            "durable_provider": "amazon_s3",
            "vast_persistent_volumes": "FORBIDDEN",
            "vast_scratch": "EPHEMERAL_INSTANCE_DISK_ONLY",
            "aws_credentials_on_worker": False,
        },
        "model": model,
        "bundle": bundle,
        "source_contract_sha256": job.EXPECTED_SOURCE_CONTRACT_SHA256,
        "paths": {
            "runtime_root": "/workspace/parallax",
            "source_root": "/workspace/parallax/source",
            "bundle_path": "/workspace/parallax/input/source.tar.gz",
            "snapshot_root": (
                "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/"
                "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
            ),
            "preflight_output": "/workspace/parallax/preflight.json",
            "optimizer_output": "/workspace/parallax/optimizer.json",
            "final_output": "/workspace/parallax/job-result.json",
        },
        "bounds": {
            "parallel_downloads": 3,
            "download_timeout_seconds": 600,
            "preflight_timeout_seconds": 180,
            "optimizer_timeout_seconds": 600,
            "minimum_free_bytes": 50_000_000_000,
        },
        "firewall": {
            "confirmation_manifest_present": False,
            "confirmation_images_present": False,
            "discovery_images_removed_before_model_load": True,
            "discovery_images_opened_by_optimizer": 0,
            "discovery_items_evaluated": 0,
            "promotion_eligible": False,
            "automatic_retry": False,
        },
    }


def test_optimizer_job_contract_accepts_exact_s3_only_inputs() -> None:
    contract = _job_contract()
    assert job.validate_contract(contract) is contract


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["storage"].update(vast_persistent_volumes="allowed"),
        lambda row: row["model"][0].update(bytes=row["model"][0]["bytes"] + 1),
        lambda row: row["bundle"].update(version_id="different"),
        lambda row: row["firewall"].update(discovery_items_evaluated=1),
    ],
)
def test_optimizer_job_contract_rejects_storage_identity_or_firewall_drift(mutation) -> None:
    contract = _job_contract()
    mutation(contract)
    with pytest.raises(job.OptimizerJobError):
        job.validate_contract(contract)


def test_optimizer_job_url_rejects_wrong_immutable_version() -> None:
    contract = _job_contract()
    row = contract["model"][0]
    row["url"] = row["url"].replace(str(row["version_id"]), "wrong-version")
    with pytest.raises(job.OptimizerJobError, match="version-bound"):
        job.validate_contract(contract)


def test_tar_inventory_accepts_receipt_and_rejects_traversal() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    members = []
    for row in receipt["entries"]:
        member = tarfile.TarInfo(row["path"])
        member.size = int(row["bytes"])
        members.append(member)
    job.validate_tar_members(members, receipt["entries"])
    traversal = tarfile.TarInfo("../escape")
    traversal.size = 1
    with pytest.raises(job.OptimizerJobError, match="path traversal"):
        job.validate_tar_members([*members, traversal], receipt["entries"])


def test_discovery_images_are_physically_removed_before_model_load(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks"
    discovery = task_root / "images" / "task" / "discovery_eval"
    train = task_root / "images" / "task" / "train"
    discovery.mkdir(parents=True)
    train.mkdir(parents=True)
    (discovery / "one.png").write_bytes(b"discovery")
    (train / "one.png").write_bytes(b"train")
    result = job._remove_discovery_images(task_root)
    assert result["status"] == "GREEN_DISCOVERY_IMAGES_ABSENT_BEFORE_MODEL_LOAD"
    assert result["deleted_files"] == 1
    assert not discovery.exists()
    assert (train / "one.png").read_bytes() == b"train"


def test_job_fails_source_first_before_large_model_hydration(monkeypatch, tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    final_output = tmp_path / "job-result.json"
    raw = {
        "model": [{"url": "secret"}],
        "bundle": {"url": "secret"},
        "paths": {"final_output": str(final_output)},
        "bounds": {"minimum_free_bytes": 1},
    }
    contract_path.write_text(json.dumps(raw), encoding="utf-8")
    order = []
    monkeypatch.setattr(job, "validate_contract", lambda value: value)
    monkeypatch.setattr(job.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    monkeypatch.setattr(
        job,
        "download_source_bundle",
        lambda value: order.append("bundle_download") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "extract_and_verify_bundle",
        lambda value: order.append("source_verify") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "hydrate_model",
        lambda value: order.append("model_hydration") or {"status": "GREEN"},
    )
    monkeypatch.setattr(
        job,
        "execute",
        lambda value: order.append("model_execution")
        or {
            "optimizer": {
                "status": "GREEN_MECHANICAL_EXECUTION",
                "adjudication": {"verdict": "GREEN_SELECTED_SCALE"},
            }
        },
    )
    packet = job.run(contract_path)
    assert order == ["bundle_download", "source_verify", "model_hydration", "model_execution"]
    assert packet["status"] == "GREEN_MECHANICAL_EXECUTION"
    assert not contract_path.exists()


def test_consumed_pg1_optimizer_v3_refuses_post_incident_implementation_drift() -> None:
    with pytest.raises(launch.OptimizerLaunchError, match="implementation hash mismatch"):
        launch.validate_spec(ROOT, LAUNCH_SPEC)


def test_frozen_optimizer_launch_mode_reaches_exact_worker_argument_builder(monkeypatch) -> None:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    monkeypatch.setattr(s3_launch, "_ephemeral_worker_registry_login", lambda row: "token")
    arguments = s3_launch._create_arguments(
        role="worker", spec=spec, run_id=str(spec["run_id"])
    )
    assert arguments[:4] == [
        "--template_hash",
        launch.RUNTIME_TEMPLATE,
        "--image",
        launch.RUNTIME_IMAGE,
    ]
    assert arguments[-2:] == ["--login", "token"]


def test_paid_execution_requires_a_separate_exact_freeze(tmp_path: Path) -> None:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    with pytest.raises(launch.OptimizerLaunchError, match="paid execution freeze"):
        launch.validate_execution_freeze(ROOT, LAUNCH_SPEC, freeze, spec)


def test_paid_freeze_rejects_any_nonzero_no_create_mutation() -> None:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    receipt_path = ROOT / (
        "studies/lineage/cross-lineage/results/readiness/"
        "parallax_optimizer_scale_no_create_v3.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["mutation"]["volumes_created"] = 1
    with pytest.raises(launch.OptimizerLaunchError, match="not exact and green"):
        launch._require_green_no_create_receipt(
            receipt,
            run_id=str(spec["run_id"]),
            spec_sha256=str(receipt["launch_spec"]["sha256"]),
            result_key=str(spec["result"]["key"]),
        )


def test_paid_freeze_rejects_non_green_planted_tests() -> None:
    receipt_path = ROOT / (
        "studies/lineage/cross-lineage/results/readiness/"
        "parallax_optimizer_scale_planted_tests_v3.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["paid_action"] = True
    with pytest.raises(launch.OptimizerLaunchError, match="not exact and green"):
        launch._require_green_planted_tests(ROOT, receipt)


def test_no_create_rehearsal_redacts_login_and_refuses_volume_arguments() -> None:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    arguments = [
        "--template_hash",
        launch.RUNTIME_TEMPLATE,
        "--image",
        launch.RUNTIME_IMAGE,
        "--disk",
        "80",
        "--onstart-cmd",
        "deadman",
        "--login",
        "-u AWS -p secret 769063704696.dkr.ecr.us-east-1.amazonaws.com",
    ]
    result = no_create._sanitized_create_arguments(arguments, role="worker", spec=spec)
    assert result["status"] == "GREEN_EXACT_CREATE_ARGUMENTS"
    assert result["arguments"][-1] == "<EPHEMERAL_ECR_LOGIN_REDACTED>"
    assert "-p secret" not in json.dumps(result)
    with pytest.raises(no_create.NoCreateRehearsalError, match="volume"):
        no_create._sanitized_create_arguments(
            [*arguments, "--volume", "forbidden"], role="worker", spec=spec
        )


def _worker_contract() -> dict[str, object]:
    return {
        "disk_gb": 80,
        "minimum_reliability": 0.98,
        "minimum_gpu_ram_mb": 80_000,
        "minimum_cpu_ram_mb": 64_000,
        "minimum_cpu_cores_effective": 8,
        "minimum_disk_bw_mb_s": 500,
        "minimum_cuda": 13.0,
        "minimum_inet_down_mbps": 500,
        "price_ceiling_dph": 2.25,
    }


def _offer(*, offer_id: int, machine_id: int, price: float, ram: int = 95_830) -> dict[str, object]:
    return {
        "id": offer_id,
        "machine_id": machine_id,
        "gpu_name": "H100 NVL",
        "num_gpus": 1,
        "gpu_ram": ram,
        "cpu_ram": 96_000,
        "cpu_cores_effective": 16,
        "reliability": 0.999,
        "dph_total": price,
        "disk_space": 100,
        "disk_bw": 1000,
        "cuda_max_good": 13.1,
        "inet_down": 800,
        "inet_up": 500,
        "static_ip": True,
        "direct_port_count": 10,
    }


def test_h100_selector_is_literal_cheapest_after_exact_suitability_and_machine_exclusion(monkeypatch) -> None:
    offers = [
        _offer(offer_id=1, machine_id=10, price=1.80),
        _offer(offer_id=2, machine_id=20, price=2.01),
        _offer(offer_id=3, machine_id=30, price=1.90, ram=79_000),
        _offer(offer_id=4, machine_id=40, price=2.10),
    ]
    monkeypatch.setattr(launch, "_vast", lambda *args, **kwargs: copy.deepcopy(offers))
    selected = launch.select_cheapest_h100_offer(
        _worker_contract(), quarantined_machine_ids={10}, excluded_machine_ids=set()
    )
    assert selected["selected"]["id"] == 2
    assert selected["quarantined_machine_ids"] == [10]
    assert selected["quarantine_applied_before_price_ordering"] is True
    assert selected["selection_rule"] == "literal_cheapest_eligible_all_regions_at_query_time"


def test_h100_selector_refuses_market_above_frozen_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(
        launch,
        "_vast",
        lambda *args, **kwargs: [_offer(offer_id=1, machine_id=10, price=2.26)],
    )
    with pytest.raises(launch.OptimizerLaunchError, match="price ceiling"):
        launch.select_cheapest_h100_offer(
            _worker_contract(), quarantined_machine_ids=set(), excluded_machine_ids=set()
        )


def test_mode_specific_registry_excludes_active_but_allows_rehabilitated_host(monkeypatch) -> None:
    binding = {
        "path": str(QUARANTINE.relative_to(ROOT)),
        "sha256": launch.sha256_file(QUARANTINE),
    }
    result = launch.require_machine_quarantine(
        ROOT,
        {
            "worker": {"launch_mode": "managed_template_private_runtime"},
            "bound_artifacts": {"machine_quarantine": binding},
        },
    )
    assert result["active_machine_ids"] == [59184, 81035, 141549]
    assert result["rehabilitated_machine_ids"] == [141956]
    offers = [
        _offer(offer_id=1, machine_id=81035, price=1.80),
        _offer(offer_id=2, machine_id=141956, price=1.90),
        _offer(offer_id=3, machine_id=54285, price=2.00),
    ]
    monkeypatch.setattr(launch, "_vast", lambda *args, **kwargs: copy.deepcopy(offers))
    selected = launch.select_cheapest_h100_offer(
        _worker_contract(),
        quarantined_machine_ids=set(result["active_machine_ids"]),
        excluded_machine_ids=set(),
    )
    assert selected["selected"]["machine_id"] == 141956


def test_machine_registry_is_required_and_content_hash_bound() -> None:
    spec = {
        "worker": {"launch_mode": "managed_template_private_runtime"},
        "bound_artifacts": {},
    }
    with pytest.raises(launch.OptimizerLaunchError, match="no bound"):
        launch.require_machine_quarantine(ROOT, spec)
    spec["bound_artifacts"]["machine_quarantine"] = {
        "path": str(QUARANTINE.relative_to(ROOT)),
        "sha256": "0" * 64,
    }
    with pytest.raises(launch.OptimizerLaunchError, match="hash mismatch"):
        launch.require_machine_quarantine(ROOT, spec)


def test_h100_ssh_wait_stops_at_frozen_post_running_deadline(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(launch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(
        launch,
        "_show_instance",
        lambda instance_id: {"actual_status": "running"},
    )
    monkeypatch.setattr(launch, "_ssh_base", lambda row, key: ["ssh"])
    monkeypatch.setattr(
        launch,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=255, stderr=b"Permission denied (publickey)"
        ),
    )
    with pytest.raises(launch.LifecycleError, match="post-running deadline"):
        launch.wait_h100_ssh(
            123,
            Path("key"),
            bootstrap_seconds=100,
            running_ssh_deadline_seconds=5,
            poll_seconds=2,
        )
    assert clock[0] <= 6


def test_h100_ssh_wait_accepts_success_before_post_running_deadline(monkeypatch) -> None:
    clock = [0.0]
    attempts = [0]
    monkeypatch.setattr(launch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(
        launch,
        "_show_instance",
        lambda instance_id: {"actual_status": "running", "machine_id": 42},
    )
    monkeypatch.setattr(launch, "_ssh_base", lambda row, key: ["ssh"])

    def probe(*args, **kwargs):
        attempts[0] += 1
        return SimpleNamespace(returncode=0 if attempts[0] == 2 else 255, stderr=b"not yet")

    monkeypatch.setattr(launch, "_run", probe)
    row = launch.wait_h100_ssh(
        123,
        Path("key"),
        bootstrap_seconds=100,
        running_ssh_deadline_seconds=5,
        poll_seconds=2,
    )
    assert row["machine_id"] == 42
    assert attempts[0] == 2


def test_result_key_absence_check_does_not_treat_other_prefix_keys_as_collision(monkeypatch) -> None:
    monkeypatch.setattr(
        launch,
        "_aws_json",
        lambda *args, **kwargs: {
            "Versions": [{"Key": "parallax/results/run/terminal.json.extra"}],
            "DeleteMarkers": [],
        },
    )
    launch._require_result_absent("parallax/results/run/terminal.json")
    monkeypatch.setattr(
        launch,
        "_aws_json",
        lambda *args, **kwargs: {
            "Versions": [{"Key": "parallax/results/run/terminal.json"}],
            "DeleteMarkers": [],
        },
    )
    with pytest.raises(launch.OptimizerLaunchError, match="already exists"):
        launch._require_result_absent("parallax/results/run/terminal.json")
