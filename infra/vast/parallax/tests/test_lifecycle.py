from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from infra.vast.parallax.guarded_launch import (
    bootstrap_guard_onstart,
    BootstrapLease,
    _build_runtime_contract,
    _claim_one_shot_authorization,
    _create_instance,
    _ephemeral_worker_registry_login,
    _expected_manifest_errors,
    _enforce_gateway_budget,
    _local_artifact_retrieval_plan,
    _retrieve_local_artifacts,
    _render,
    _json_output,
    _run_idempotent_ssh,
    _normalized_lifecycle_archive,
    _semantic_lifecycle_ready_command,
    _scp_base,
    _short_control_path,
    _ssh_endpoint,
    _ssh_base,
    _regenerate_remote_instance_api_key,
    _regenerated_instance_credential_shell,
    _require_worker_image_capabilities,
    _wait_instance_stopped,
    _worker_create_arguments,
    _validate_live_control_evidence,
    _validate_worker_registry_auth,
    _validated_bootstrap_ssh_key,
    _validated_registered_ssh_key,
    _validated_scoped_lifecycle_api_key,
    launch,
    LocalBootstrapReaper,
    inspect_persisted_run,
    probe_gateway_controls,
    require_persistent_volume_controller_active,
    require_paid_authorization,
    scoped_deadman_command,
    prearm_deadline_onstart,
    select_live_offers,
    ttl_onstart,
    validate_frozen_spec,
    validate_job_spec,
    wait_ready,
)
from infra.vast.parallax.lifecycle_planted_gate import run as run_plants
from infra.vast.parallax.lifecycle_launch_packet import validate_lifecycle_launch_packet
from infra.vast.parallax.launch_packet import LaunchPacketError
from infra.vast.parallax.lifecycle_supervisor import (
    LifecycleError,
    LifecyclePolicy,
    GatewayBackend,
    vast_instance_action,
    verify_persisted_run,
    watchdog,
    wait_for_execution_release,
    sha256_file,
)
from infra.vast.parallax.storage_rehearsal import VastMutation, rehearse


ROOT = Path(__file__).resolve().parents[4]


def test_persistent_vast_volume_controller_is_hard_retired():
    with pytest.raises(LifecycleError, match="persistent-Vast-volume controller is retired"):
        require_persistent_volume_controller_active()


def _ecr_auth_spec() -> dict[str, object]:
    return {
        "worker_image": (
            "769063704696.dkr.ecr.us-east-1.amazonaws.com/parallax/runtime-minimal@sha256:"
            + "a" * 64
        ),
        "worker_registry_auth": {
            "type": "aws_ecr_ephemeral",
            "profile": "fuelos",
            "region": "us-east-1",
            "registry": "769063704696.dkr.ecr.us-east-1.amazonaws.com",
            "username": "AWS",
            "credential_source": "aws_ecr_get_login_password",
            "persist_secret": False,
            "maximum_token_age_seconds": 43200,
        },
    }


def test_ephemeral_ecr_login_uses_only_fuelos_and_never_persists_secret():
    spec = _ecr_auth_spec()
    completed = subprocess.CompletedProcess([], 0, b"expiring-secret\n", b"")
    with patch("infra.vast.parallax.guarded_launch._run", return_value=completed) as run:
        login = _ephemeral_worker_registry_login(spec)
    assert login == (
        "-u AWS -p expiring-secret 769063704696.dkr.ecr.us-east-1.amazonaws.com"
    )
    assert run.call_args.args[0] == [
        "aws", "ecr", "get-login-password", "--profile", "fuelos", "--region", "us-east-1"
    ]
    assert "expiring-secret" not in json.dumps(spec)


def test_ecr_auth_rejects_any_non_fuelos_profile():
    spec = _ecr_auth_spec()
    spec["worker_registry_auth"]["profile"] = "default"
    with pytest.raises(LifecycleError, match="exact fuelos ECR contract"):
        _validate_worker_registry_auth(spec)


def test_gateway_only_budget_ignores_uncreated_worker_price():
    spec = {"gateway_price_ceiling_dph": 0.15, "worker_price_ceiling_dph": 0.0}
    market = {
        "selected_gateway": {"dph_total": 0.1016296296},
        "selected_worker": {"dph_total": 99.0},
    }
    _enforce_gateway_budget(spec, market)


def test_official_template_worker_create_preserves_template_launch_mode():
    spec = {
        "h100_disk_gb": 32,
        "worker_template_hash": "a49dfc59f765f8c5b9c1e0c3a318a486",
    }
    arguments = _worker_create_arguments(spec, run_id="parallax-h100-template-control-v1")
    assert arguments == [
        "--template_hash", "a49dfc59f765f8c5b9c1e0c3a318a486",
        "--disk", "32", "--cancel-unavail",
        "--label", "parallax-worker-parallax-h100-template-control-v1",
    ]
    assert "--image" not in arguments
    assert "--onstart-cmd" not in arguments


def test_managed_template_can_override_only_the_exact_worker_image():
    image = "registry.example/runtime@sha256:" + "a" * 64
    spec = {
        "h100_disk_gb": 32,
        "worker_template_hash": "a49dfc59f765f8c5b9c1e0c3a318a486",
        "worker_template_image_override": True,
        "worker_image": image,
    }
    arguments = _worker_create_arguments(spec, run_id="override-v1")
    assert arguments[:4] == [
        "--template_hash", "a49dfc59f765f8c5b9c1e0c3a318a486", "--image", image,
    ]
    assert "--onstart-cmd" not in arguments
    assert "--ssh" not in arguments and "--direct" not in arguments


def test_private_create_failure_never_discloses_registry_secret():
    secret = "do-not-log-this-secret"
    completed = subprocess.CompletedProcess([], 2, b"", b"provider detail")
    with patch("infra.vast.parallax.guarded_launch._run", return_value=completed):
        with pytest.raises(LifecycleError) as error:
            _create_instance(123, ["--image", "private/image", "--login", secret])
    assert secret not in str(error.value)


def test_private_create_timeout_never_discloses_registry_secret():
    secret = "do-not-log-this-timeout-secret"
    timeout = subprocess.TimeoutExpired(["vastai", "create", "--login", secret], 60)
    with patch("infra.vast.parallax.guarded_launch._run", side_effect=timeout):
        with pytest.raises(LifecycleError) as error:
            _create_instance(123, ["--image", "private/image", "--login", secret])
    assert secret not in str(error.value)


def test_lifecycle_plants_all_green(tmp_path: Path):
    packet = run_plants(tmp_path / "plants.json")
    assert packet["status"] == "GREEN"
    assert len(packet["cases"]) == 12
    assert all(row["status"] == "GREEN" for row in packet["cases"].values())


def test_ttl_onstart_regenerates_only_the_documented_per_instance_key():
    stop = ttl_onstart(action="stop", seconds=60)
    destroy = ttl_onstart(action="destroy", seconds=90)
    assert "VAST_CONTAINERLABEL" in stop and "/root/.ssh/authorized_keys" in stop
    assert "authorized_hash" in stop and "label_hash" in stop
    assert "CONTAINER_API_KEY" not in stop and "CONTAINER_ID" not in stop
    assert "state" in stop and "stopped" in stop
    assert "-X PUT" in stop
    assert "-X DELETE" in destroy
    assert "VAST_API_KEY" not in stop + destroy
    assert "--retry-max-time 15" in stop + destroy
    assert "until curl" in stop + destroy
    assert "do sleep 5; done" in stop + destroy
    assert subprocess.run(["bash", "-n", "-c", stop], check=False).returncode == 0
    assert subprocess.run(["bash", "-n", "-c", destroy], check=False).returncode == 0


def test_bootstrap_orphan_guards_are_provider_side_and_shell_valid():
    stop = bootstrap_guard_onstart(action="stop", seconds=90)
    destroy = bootstrap_guard_onstart(action="destroy", seconds=90)
    assert "parallax-lifecycle-armed" in stop + destroy
    assert "VAST_CONTAINERLABEL" in stop + destroy
    assert "/root/.ssh/authorized_keys" in stop + destroy
    assert "CONTAINER_API_KEY" not in stop + destroy
    assert "VAST_API_KEY" not in stop + destroy
    assert "--retry-max-time 15" in stop + destroy
    assert subprocess.run(["bash", "-n", "-c", stop], check=False).returncode == 0
    assert subprocess.run(["bash", "-n", "-c", destroy], check=False).returncode == 0
    assert "parallax-bootstrap-lease" in stop + destroy
    assert "stat -c %Y" in stop + destroy


def test_paid_worker_has_nonrenewable_prearm_deadline():
    stop = prearm_deadline_onstart(action="stop", seconds=120)
    assert "parallax-lifecycle-armed" in stop
    assert "parallax-bootstrap-lease" not in stop
    assert "sleep 120" in stop and "until curl" in stop
    assert subprocess.run(["bash", "-n", "-c", stop], check=False).returncode == 0


def test_scoped_deadman_reads_secret_from_mode600_file_at_execution_time():
    stop = scoped_deadman_command(
        instance_id=123,
        key_path="/run/secrets/lifecycle.key",
        seconds=120,
        action="stop",
        armed_path="/tmp/parallax-lifecycle-armed",
    )
    assert "$(cat /run/secrets/lifecycle.key)" in stop
    assert "instances/123" in stop and "sleep 120" in stop
    assert "state" in stop and "stopped" in stop
    assert subprocess.run(["bash", "-n", "-c", stop], check=False).returncode == 0


def test_bootstrap_lease_reports_renewal_failure():
    calls = []

    def run(arguments, **_kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0 if len(calls) == 1 else 255, b"", b"refused")

    with patch("infra.vast.parallax.guarded_launch._run", side_effect=run):
        lease = BootstrapLease(["ssh", "host"], interval_seconds=0.001)
        assert lease.failed.wait(0.2)
        with pytest.raises(LifecycleError, match="renewal failed"):
            lease.assert_healthy()
        lease.close()


def test_idempotent_arm_marker_retries_transient_ssh_only_within_bound():
    failed = subprocess.CompletedProcess([], 255, b"", b"transient")
    green = subprocess.CompletedProcess([], 0, b"", b"")
    with (
        patch("infra.vast.parallax.guarded_launch._run", side_effect=[failed, green]) as run,
        patch("infra.vast.parallax.guarded_launch.time.sleep"),
    ):
        assert _run_idempotent_ssh(["ssh", "worker"], "touch /tmp/armed").returncode == 0
    assert run.call_count == 2


def test_worker_image_capability_gate_fails_before_release():
    failed = subprocess.CompletedProcess([], 127, b"", b"rsync: not found")
    with (
        patch("infra.vast.parallax.guarded_launch._run", return_value=failed) as run,
        pytest.raises(LifecycleError, match="live command gate"),
    ):
        _require_worker_image_capabilities(["ssh", "worker"])
    command = run.call_args.args[0][-1]
    assert "command -v bash python3 rsync ssh sha256sum" in command
    assert "import hashlib,json,os,pathlib" in command


def test_runtime_image_probe_job_is_no_gpu_no_model_no_data_and_atomic():
    path = (
        ROOT
        / "studies/lineage/cross-lineage/config/parallax_runtime_image_probe_job_v1.json"
    )
    job = json.loads(path.read_text())
    assert job["scope"] == "runtime_image_compatibility_probe"
    assert job["uses_gpu"] is False
    assert job["model_reads"] == job["data_reads"] == 0
    assert job["automatic_retry"] is False
    assert job["required_local_artifacts"] == ["runtime-probe/probe.json"]
    runtime = job["runtime"]
    assert runtime["job"]["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert runtime["required_sentinels"] == ["runtime-probe/probe.json"]
    command = runtime["job"]["command"][2].replace("{RUN_ID}", job["run_id"])
    probe_root = Path("/tmp/parallax-runtime-image-probe-v1")
    shutil.rmtree(probe_root, ignore_errors=True)
    try:
        completed = subprocess.run(
            ["python3", "-c", command],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        packet = json.loads((probe_root / "probe.json").read_text())
        assert packet["status"] in {"GREEN_COMPATIBLE", "RED_INCOMPATIBLE"}
        assert packet["gpu_work"] is False
        assert packet["model_reads"] == packet["data_reads"] == 0
        assert not (probe_root / ".probe.json.partial").exists()
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)


def test_non_h100_runtime_probe_allows_only_exact_bounded_gpu_capability_smoke(tmp_path: Path):
    job = {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": "runtime_image_compatibility_probe",
        "uses_gpu": True,
        "gpu_work_class": "single_tensor_capability_smoke",
        "scientific_claim": False,
        "model_reads": 0,
        "data_reads": 0,
        "artifact_protocol": "atomic_rename_no_background_writers",
        "required_local_artifacts": ["runtime-probe/probe.json"],
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    spec = {
        "scope": "runtime_image_compatibility_probe",
        "lifecycle_worker_kind": "non_h100",
        "h100_research_authorized": False,
        "paid_launch_authorized": False,
        "authorized_job_spec": {
            "path": "job.json",
            "sha256": sha256_file(path),
        },
    }
    assert validate_job_spec(tmp_path, spec, path)["gpu_work_class"] == (
        "single_tensor_capability_smoke"
    )

    for mutation in (
        {"gpu_work_class": "training"},
        {"scientific_claim": True},
        {"model_reads": 1},
        {"data_reads": 1},
    ):
        candidate = dict(job)
        candidate.update(mutation)
        path.write_text(json.dumps(candidate))
        spec["authorized_job_spec"]["sha256"] = sha256_file(path)
        with pytest.raises(LifecycleError, match="cannot run a GPU workload"):
            validate_job_spec(tmp_path, spec, path)


def test_paid_job_refuses_mismatched_authorization_run_and_claim_path(tmp_path: Path):
    job = {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": "runtime_image_compatibility_probe",
        "run_id": "parallax-runtime-image-probe-v3",
        "uses_gpu": False,
        "artifact_protocol": "atomic_rename_no_background_writers",
        "required_local_artifacts": ["runtime-probe/probe.json"],
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    spec = {
        "scope": job["scope"],
        "paid_launch_authorized": True,
        "h100_research_authorized": False,
        "authorized_job_spec": {"path": "job.json", "sha256": sha256_file(path)},
        "one_shot_authorization": {
            "id": job["run_id"],
            "claim_paths": {"launch": "claims/parallax_runtime_image_probe_v2_launch.json"},
        },
    }
    with pytest.raises(LifecycleError, match="authorization id, run id, and claim path disagree"):
        validate_job_spec(tmp_path, spec, path)


def test_runtime_renderer_distinguishes_python_braces_from_frozen_placeholders():
    source = ROOT / "studies/lineage/cross-lineage/config/parallax_runtime_image_probe_job_v3.json"
    runtime = json.loads(source.read_text())["runtime"]
    rendered = _render(
        runtime,
        {
            "RUN_ID": "parallax-runtime-image-probe-v3",
            "H100_HOST": "worker.example",
            "H100_PORT": "22",
            "H100_SSH_KEY": "/tmp/key",
            "VOLUME_RUN_ROOT": "/volume/run",
        },
    )
    command = rendered["job"]["command"][2]
    compile(command, str(source), "exec")
    assert "{RUN_ID}" not in command
    assert "{'torch':'torch'" in command
    with pytest.raises(LifecycleError, match="unresolved lifecycle runtime placeholder"):
        _render("echo {MISSING}", {})


def test_v3_exact_runtime_contract_reproduces_execution_site_red():
    spec = json.loads(
        (ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v36.json").read_text()
    )
    job = json.loads(
        (ROOT / "studies/lineage/cross-lineage/config/parallax_runtime_image_probe_job_v3.json").read_text()
    )
    with pytest.raises(LifecycleError, match="job.execution_site must be gateway or h100"):
        _build_runtime_contract(
            spec=spec,
            job_spec=job,
            run_id=job["run_id"],
            worker_instance_id=2,
            worker_host="worker.example",
            worker_port=22,
            worker_ssh_key="/tmp/worker.key",
            worker_api_key="/tmp/worker.api",
            gateway_instance_id=1,
            gateway_api_key="/tmp/gateway.api",
            volume_run_root="/volume/parallax/lifecycle-runs/unit-v3",
            frozen_spec_sha256="a" * 64,
            job_spec_sha256="b" * 64,
        )


def test_v4_shared_builder_produces_complete_valid_runtime_contract():
    spec_path = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v39.json"
    job_path = ROOT / "studies/lineage/cross-lineage/config/parallax_runtime_image_probe_job_v4.json"
    # Historical freezes deliberately bind the implementation that existed at
    # authorization time. Read the historical fixture directly after the
    # launcher evolves; active freezes still use validate_frozen_spec.
    spec = json.loads(spec_path.read_text())
    job = validate_job_spec(ROOT, spec, job_path)
    contract = _build_runtime_contract(
        spec=spec,
        job_spec=job,
        run_id=job["run_id"],
        worker_instance_id=2,
        worker_host="worker.example",
        worker_port=22,
        worker_ssh_key="/tmp/worker.key",
        worker_api_key="/tmp/worker.api",
        gateway_instance_id=1,
        gateway_api_key="/tmp/gateway.api",
        volume_run_root="/volume/parallax/lifecycle-runs/unit-v4",
        frozen_spec_sha256=sha256_file(spec_path),
        job_spec_sha256=sha256_file(job_path),
    )
    assert contract["job"]["execution_site"] == "h100"
    assert contract["run_id"] == "parallax-runtime-image-probe-v4"
    compile(contract["job"]["command"][2], str(job_path), "exec")
    assert contract["policy"]["required_sentinels"] == ["runtime-probe/probe.json"]


def test_bootstrap_ssh_and_scp_share_one_control_socket(tmp_path: Path):
    row = {"ssh_host": "worker", "ssh_port": 2200}
    key = tmp_path / "key"
    control = tmp_path / "bootstrap.sock"
    ssh = _ssh_base(row, key, control_path=control)
    scp = _scp_base(row, key, control_path=control)
    expected = f"ControlPath={control}"
    assert expected in ssh and expected in scp
    assert "ControlMaster=auto" in ssh and "ControlMaster=auto" in scp
    assert "ControlPersist=60" in ssh and "ControlPersist=60" in scp


def test_ssh_endpoint_prefers_proven_direct_mapping_over_proxy():
    row = {
        "ssh_host": "ssh8.vast.ai",
        "ssh_port": 21650,
        "public_ipaddr": "141.195.21.72",
        "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41778"}]},
    }
    assert _ssh_endpoint(row) == ("141.195.21.72", 41778)
    assert _ssh_base(row, Path("/key"))[-3:] == ["-p", "41778", "root@141.195.21.72"]


def test_ssh_endpoint_falls_back_to_proxy_without_direct_mapping():
    assert _ssh_endpoint({"ssh_host": "worker", "ssh_port": 2200}) == ("worker", 2200)


def test_control_socket_path_is_short_even_when_host_tempdir_is_too_long():
    long_old_path = (
        "/var/folders/9j/yjj9b3jj47v01_299q09_nc40000gn/T/"
        "parallax-guarded-launch-kx1kzhol/bootstrap-45117000.sock"
    )
    assert len(os.fsencode(long_old_path)) == 105
    bootstrap = _short_control_path(role="bootstrap", instance_id=45117000)
    inspection = _short_control_path(role="inspection", instance_id=45117000)
    object_mirror = _short_control_path(role="object-mirror", instance_id=45117000)
    cache_mirror = _short_control_path(role="cache-mirror", instance_id=45117000)
    assert len(os.fsencode(bootstrap)) <= 80
    assert len(os.fsencode(inspection)) <= 80
    assert len(os.fsencode(object_mirror)) <= 80
    assert len(os.fsencode(cache_mirror)) <= 80
    assert bootstrap.parent == inspection.parent == object_mirror.parent == cache_mirror.parent == Path("/tmp")


def test_execution_release_gate_never_starts_on_timeout(tmp_path: Path):
    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += seconds

    assert wait_for_execution_release(
        tmp_path / "missing.marker",
        timeout_seconds=5,
        monotonic=lambda: clock["now"],
        sleep=sleep,
    ) is False
    marker = tmp_path / "released.marker"
    marker.touch()
    assert wait_for_execution_release(marker, timeout_seconds=5) is True


def test_cleanup_parser_tolerates_vast_non_json_output():
    completed = subprocess.CompletedProcess(["vastai"], 0, b"{'success': True}\n", b"")
    assert _json_output(completed) == "{'success': True}"


def test_legacy_storage_rehearsal_cannot_bypass_guarded_launcher(tmp_path: Path):
    mutation = object.__new__(VastMutation)
    mutation.events = []
    with (
        patch("infra.vast.parallax.storage_rehearsal.subprocess.run") as run,
        pytest.raises(RuntimeError, match="revoked"),
    ):
        mutation.call(["create", "instance", "123"])
    run.assert_not_called()
    with pytest.raises(RuntimeError, match="revoked"):
        rehearse(tmp_path / "missing-audit.json", tmp_path / "never-written.json")


def test_bootstrap_key_is_existing_registered_key_and_never_mutates_account(tmp_path: Path):
    private = tmp_path / "id_ed25519"
    public = tmp_path / "id_ed25519.pub"
    private.write_text("PRIVATE\n")
    private.chmod(0o600)
    public.write_text("ssh-ed25519 AAAATEST local\n")
    spec = {"bootstrap_ssh_private_key": str(private), "bootstrap_ssh_public_key_id": 77}
    with patch(
        "infra.vast.parallax.guarded_launch._vast",
        return_value=[{"id": 77, "public_key": "ssh-ed25519 AAAATEST local"}],
    ) as vast:
        assert _validated_bootstrap_ssh_key(spec) == private
    vast.assert_called_once_with("show", "ssh-keys")


def test_dedicated_relay_key_must_match_exact_registered_id(tmp_path: Path):
    private = tmp_path / "relay"
    public = tmp_path / "relay.pub"
    private.write_text("PRIVATE\n")
    private.chmod(0o600)
    public.write_text("ssh-ed25519 RELAY dedicated\n")
    spec = {"relay_ssh_private_key": str(private), "relay_ssh_public_key_id": 88}
    with patch(
        "infra.vast.parallax.guarded_launch._vast",
        return_value=[{"id": 88, "public_key": "ssh-ed25519 RELAY dedicated"}],
    ):
        assert _validated_registered_ssh_key(spec, role="relay") == private


def test_scoped_lifecycle_key_is_registered_and_mode_0600(tmp_path: Path):
    secret = tmp_path / "lifecycle.key"
    secret.write_text("scoped-secret-value\n")
    secret.chmod(0o600)
    spec = {
        "scoped_lifecycle_api_key_file": str(secret),
        "scoped_lifecycle_api_key_id": 55,
        "scoped_lifecycle_api_key_name": "parallax-lifecycle-v2",
    }
    with patch("infra.vast.parallax.guarded_launch._vast", return_value={"apikeys": [[{
        "id": 55,
        "name": "parallax-lifecycle-v2",
        "key": "scoped-secret-value",
    }]]}):
        assert _validated_scoped_lifecycle_api_key(spec) == (secret, 55)
    secret.chmod(0o644)
    with pytest.raises(LifecycleError, match="mode 0600"):
        _validated_scoped_lifecycle_api_key(spec)


def test_paid_authorization_claim_is_atomic_and_one_shot(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}\n")
    spec = {
        "one_shot_authorization": {
            "id": "bounded-proof",
            "claim_paths": {"launch": "claims/launch.json"},
        }
    }
    claim = _claim_one_shot_authorization(root=tmp_path, spec_path=spec_path, spec=spec, mode="launch")
    payload = json.loads(claim.read_text())
    assert payload["status"] == "CONSUMED_BEFORE_MARKET_MUTATION"
    assert payload["mode"] == "launch"
    with pytest.raises(LifecycleError, match="already consumed"):
        _claim_one_shot_authorization(root=tmp_path, spec_path=spec_path, spec=spec, mode="launch")


def test_documented_instance_key_is_regenerated_without_putting_secret_in_argv():
    secret = b"a" * 64
    completed = subprocess.CompletedProcess([], 0, secret, b"")
    with patch("infra.vast.parallax.guarded_launch._run", return_value=completed) as run:
        assert _regenerate_remote_instance_api_key(["ssh", "worker"], instance_id=123) == secret.decode()
    command = run.call_args.args[0][-1]
    assert 'VAST_CONTAINERLABEL:-}" = "C.123"' in command
    assert "/root/.ssh/authorized_keys" in command and "md5sum" in command
    assert secret.decode() not in " ".join(run.call_args.args[0])


def test_documented_shell_regeneration_matches_two_md5_concatenation(tmp_path: Path):
    import hashlib

    authorized = tmp_path / "authorized_keys"
    authorized.write_bytes(b"ssh-ed25519 AAAATEST first\nssh-ed25519 AAAATEST2 second\n")
    fragment = _regenerated_instance_credential_shell().replace(
        "/root/.ssh/authorized_keys", str(authorized)
    )
    completed = subprocess.run(
        ["bash", "-c", fragment + '; printf %s "$instance_api_key"'],
        env={"PATH": os.environ["PATH"], "VAST_CONTAINERLABEL": "C.123"},
        capture_output=True,
        check=True,
    )
    expected = hashlib.md5(authorized.read_bytes()).hexdigest() + hashlib.md5(b"C.123").hexdigest()
    assert completed.stdout.decode() == expected


def test_documented_key_regeneration_rejects_bad_length_or_label_failure():
    bad = subprocess.CompletedProcess([], 1, b"short", b"label mismatch")
    with (
        patch("infra.vast.parallax.guarded_launch._run", return_value=bad),
        pytest.raises(LifecycleError, match="could not regenerate"),
    ):
        _regenerate_remote_instance_api_key(["ssh", "worker"], instance_id=123)


def test_provider_stop_observer_never_issues_a_mutating_action():
    with (
        patch(
            "infra.vast.parallax.guarded_launch._show_instance",
            side_effect=[{"actual_status": "running"}, {"actual_status": "stopped"}],
        ),
        patch("infra.vast.parallax.guarded_launch.time.sleep"),
        patch("infra.vast.parallax.guarded_launch.vast_instance_action") as action,
    ):
        packet = _wait_instance_stopped(123, timeout_seconds=10)
    assert packet["status"] == "STOPPED"
    assert packet["operator"] == "provider_observed_stop"
    action.assert_not_called()


def test_minimal_lifecycle_archive_imports_without_full_control_plane(tmp_path: Path):
    archive = tmp_path / "lifecycle.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    _normalized_lifecycle_archive(ROOT, archive)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extracted)
    completed = subprocess.run(
        [sys.executable, "-c", "import infra.vast.parallax.lifecycle_supervisor"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(extracted)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_detached_local_reaper_arms_and_hands_off_without_provider_mutation(tmp_path: Path):
    key = tmp_path / "lifecycle.key"
    key.write_text("scoped-key\n")
    key.chmod(0o600)
    reaper = LocalBootstrapReaper(
        tmp_path,
        repo_root=ROOT,
        run_id="detached-reaper-plant",
        lifecycle_key_path=key,
        maximum_seconds=60,
    )
    packet = reaper.handoff(run_id="detached-reaper-plant")
    assert packet["status"] == "HANDED_OFF_TO_REMOTE_GUARDIAN"


def test_bootstrap_detects_instance_disappearing_after_provider_guard():
    with (
        patch("infra.vast.parallax.guarded_launch._show_instance", side_effect=[{"actual_status": "loading"}, None]),
        patch("infra.vast.parallax.guarded_launch.time.sleep"),
        pytest.raises(LifecycleError, match="disappeared"),
    ):
        wait_ready(123, timeout_seconds=5)


def test_runtime_policy_rejects_slow_mirroring_and_path_escape():
    contract = {
        "schema_version": 1,
        "status": "FROZEN_RUNTIME",
        "run_id": "test",
        "policy": {
            "maximum_runtime_seconds": 10,
            "mirror_interval_seconds": 6,
            "heartbeat_timeout_seconds": 20,
            "artifact_copy_timeout_seconds": 5,
            "maximum_finalization_seconds": 5,
            "final_sync_attempts": 2,
            "final_sync_settle_seconds": 0,
            "required_sentinels": ["../result.json"],
        },
        "teardown": {
            "durability_before_destroy": True,
            "stop_on_unproven_durability": True,
            "h100_api_key_scope": "per_instance_only",
        },
    }
    with pytest.raises(LifecycleError):
        LifecyclePolicy.from_contract(contract)


def _watchdog_contract(tmp_path: Path) -> tuple[dict, Path]:
    secret = tmp_path / "instance.key"
    secret.write_text("per-instance-secret\n")
    secret.chmod(0o600)
    gateway_secret = tmp_path / "gateway.key"
    gateway_secret.write_text("gateway-instance-secret\n")
    gateway_secret.chmod(0o600)
    run_root = tmp_path / "run"
    contract = {
        "schema_version": 1,
        "status": "FROZEN_RUNTIME",
        "run_id": "watchdog-test",
        "policy": {
            "maximum_runtime_seconds": 10,
            "mirror_interval_seconds": 1,
            "heartbeat_timeout_seconds": 3,
            "artifact_copy_timeout_seconds": 5,
            "maximum_finalization_seconds": 5,
            "final_sync_attempts": 2,
            "final_sync_settle_seconds": 0,
            "required_sentinels": ["result/result.json"],
        },
        "teardown": {
            "durability_before_destroy": True,
            "stop_on_unproven_durability": True,
            "h100_api_key_scope": "per_instance_only",
        },
        "h100": {
            "instance_id": 123,
            "ssh_host": "127.0.0.1",
            "ssh_port": 22,
            "ssh_private_key_file": str(secret),
            "instance_api_key_file": str(secret),
        },
        "gateway": {
            "instance_id": 456,
            "instance_api_key_file": str(gateway_secret),
            "self_destroy": True,
            "volume_run_root": str(run_root),
        },
        "job": {"execution_site": "h100", "cwd": "/tmp", "command": ["true"]},
        "artifacts": {"roots": [{"remote_root": "/tmp/output", "durable_prefix": "result"}]},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return contract, contract_path


def test_watchdog_stops_h100_when_supervisor_never_heartbeats(tmp_path: Path):
    contract, contract_path = _watchdog_contract(tmp_path)
    clock = {"value": 0.0}
    actions = []

    def monotonic():
        return clock["value"]

    def sleep(seconds):
        clock["value"] += seconds

    def action(**kwargs):
        actions.append(kwargs)
        return {"status": "DESTROYED" if kwargs["action"] == "destroy" else "STOPPED"}

    packet = watchdog(
        contract,
        contract_path,
        action=action,
        monotonic=monotonic,
        wall_time=lambda: clock["value"],
        sleep=sleep,
    )
    assert packet["status"] == "H100_STOP_REQUESTED_DISK_PRESERVED"
    assert actions == [
        {"instance_id": 123, "api_key": "per-instance-secret", "action": "stop"},
        {"instance_id": 456, "api_key": "gateway-instance-secret", "action": "destroy"},
    ]


def test_watchdog_exits_without_api_call_after_supervisor_completion(tmp_path: Path):
    contract, contract_path = _watchdog_contract(tmp_path)
    complete = Path(contract["gateway"]["volume_run_root"]) / "lifecycle/complete.marker"
    complete.parent.mkdir(parents=True)
    complete.write_text("DURABLE_AND_DESTROYED\n")
    packet = watchdog(
        contract,
        contract_path,
        action=lambda **_kwargs: pytest.fail("watchdog must not call API after completion"),
    )
    assert packet["status"] == "EXIT_GREEN_SUPERVISOR_COMPLETE"


class _HTTPResponse:
    def __init__(self, payload: bytes = b'{"success":true}'):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_vast_instance_action_uses_per_instance_bearer_and_correct_methods():
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return _HTTPResponse()

    with patch("infra.vast.parallax.lifecycle_supervisor.urllib.request.urlopen", side_effect=urlopen):
        assert vast_instance_action(instance_id=12, api_key="instance-only", action="stop")["status"] == "STOPPED"
        assert vast_instance_action(instance_id=12, api_key="instance-only", action="destroy")["status"] == "DESTROYED"
    stop, destroy = requests
    assert stop[0].method == "PUT"
    assert json.loads(stop[0].data) == {"state": "stopped"}
    assert destroy[0].method == "DELETE" and destroy[0].data is None
    assert stop[0].get_header("Authorization") == "Bearer instance-only"
    assert destroy[0].get_header("Authorization") == "Bearer instance-only"
    assert stop[1] == destroy[1] == 10


def test_redundant_stop_treats_already_absent_worker_as_terminal():
    missing = urllib.error.HTTPError(
        "https://console.vast.ai/api/v0/instances/12/", 404, "missing", {}, None
    )
    with patch(
        "infra.vast.parallax.lifecycle_supervisor.urllib.request.urlopen",
        side_effect=missing,
    ):
        packet = vast_instance_action(instance_id=12, api_key="instance-only", action="stop")
    assert packet == {"status": "STOPPED", "attempts": 1, "already_absent": True}


def test_green_teardown_stops_compute_before_destroy(tmp_path: Path):
    key = tmp_path / "instance.key"
    key.write_text("instance-only\n")
    key.chmod(0o600)
    backend = object.__new__(GatewayBackend)
    backend.instance_id = 777
    backend.api_key_file = key
    backend.state_root = tmp_path / "lifecycle"
    backend.state_root.mkdir()
    backend.policy = type("Policy", (), {"run_id": "teardown-test"})()
    actions = []

    def action(**kwargs):
        actions.append(kwargs["action"])
        return {"status": "STOPPED" if kwargs["action"] == "stop" else "DESTROYED"}

    with patch("infra.vast.parallax.lifecycle_supervisor.vast_instance_action", side_effect=action):
        packet = backend.destroy_h100()
    assert packet["status"] == "DESTROYED"
    assert actions == ["stop", "destroy"]
    intent = json.loads((backend.state_root / "worker-stop-intent.json").read_text())
    assert intent["status"] == "STOP_CONFIRMED_DESTROY_PENDING"


def test_gateway_probes_relay_ssh_before_supervisor_can_arm(tmp_path: Path):
    key = tmp_path / "relay.key"
    key.write_text("PRIVATE\n")
    backend = object.__new__(GatewayBackend)
    backend.ssh_key = key
    backend.host = "worker"
    backend.port = 22
    backend.control_path = tmp_path / "control.sock"
    backend.state_root = tmp_path / "lifecycle"
    backend.state_root.mkdir()
    completed = subprocess.CompletedProcess([], 0, b"", b"")
    with patch("infra.vast.parallax.lifecycle_supervisor.subprocess.run", return_value=completed):
        packet = backend.probe_worker_ssh()
    assert packet["status"] == "GREEN"
    assert json.loads((backend.state_root / "relay-ssh-probe.json").read_text())["status"] == "GREEN"


def test_concrete_gateway_mirror_is_stable_atomic_and_manifested(tmp_path: Path):
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote = remote_root / "result.json"
    remote.write_text('{"generation":1}\n')
    ssh_key = tmp_path / "ssh.key"
    api_key = tmp_path / "api.key"
    for path in (ssh_key, api_key):
        path.write_text("secret\n")
        path.chmod(0o600)
    run_root = tmp_path / "volume-run"
    contract = {
        "schema_version": 1,
        "status": "FROZEN_RUNTIME",
        "run_id": "mirror-test",
        "policy": {
            "maximum_runtime_seconds": 10,
            "mirror_interval_seconds": 1,
            "heartbeat_timeout_seconds": 3,
            "artifact_copy_timeout_seconds": 5,
            "maximum_finalization_seconds": 5,
            "final_sync_attempts": 2,
            "final_sync_settle_seconds": 0,
            "required_sentinels": ["result/result.json"],
        },
        "teardown": {
            "durability_before_destroy": True,
            "stop_on_unproven_durability": True,
            "h100_api_key_scope": "per_instance_only",
        },
        "h100": {
            "instance_id": 1,
            "ssh_host": "worker",
            "ssh_port": 22,
            "ssh_private_key_file": str(ssh_key),
            "instance_api_key_file": str(api_key),
        },
        "gateway": {
            "instance_id": 2,
            "instance_api_key_file": str(api_key),
            "volume_run_root": str(run_root),
        },
        "job": {"execution_site": "h100", "cwd": "/tmp", "command": ["true"]},
        "artifacts": {"roots": [{"remote_root": str(remote_root), "durable_prefix": "result"}]},
        "expected_artifacts": {
            "result/result.json": {
                "bytes": remote.stat().st_size,
                "sha256": __import__("hashlib").sha256(remote.read_bytes()).hexdigest(),
            }
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    backend = GatewayBackend(contract, contract_path)

    def scan(*, timeout_seconds):
        assert 0 < timeout_seconds <= 5
        stat = remote.stat()
        return [{
            "remote": str(remote),
            "relative": remote.name,
            "prefix": "result",
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }]

    backend._scan = scan

    def copy_run(arguments, **_kwargs):
        source = arguments[-2].split(":", 1)[1]
        shutil.copyfile(source, arguments[-1])
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("infra.vast.parallax.lifecycle_supervisor.subprocess.run", side_effect=copy_run):
        assert backend.mirror_once()["copied_count"] == 0
        assert backend.mirror_once()["copied_count"] == 1
    durable = run_root / "artifacts/result/result.json"
    assert durable.read_bytes() == remote.read_bytes()
    proof = backend.prove_durability(job_exit_code=0)
    assert proof["status"] == "GREEN" and proof["entry_count"] == 1
    extra = run_root / "artifacts/result/unexpected.bin"
    extra.write_bytes(b"unexpected")
    refused = backend.prove_durability(job_exit_code=0)
    assert refused["status"] == "UNPROVEN"
    assert refused["unexpected_artifacts"] == ["result/unexpected.bin"]
    extra.unlink()
    final_proof = backend.prove_durability(job_exit_code=0)
    assert final_proof["status"] == "GREEN"
    assert (backend.state_root / "manifest.previous.json").is_file()
    journal_lines = (backend.state_root / "upload-journal.jsonl").read_text().splitlines()
    assert len(journal_lines) == 1
    assert json.loads(journal_lines[0])["event"] == "COPY_COMMITTED"
    backend.write_receipt({
        "status": "DURABLE_AND_DESTROYED",
        "run_id": "mirror-test",
        "durability": final_proof,
        "teardown": {"status": "DESTROYED"},
    })
    reattached = verify_persisted_run(run_root)
    assert reattached["status"] == "GREEN" and reattached["volume_reattach_verified"] is True
    with (backend.state_root / "upload-journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event":"TAMPER"}\n')
    tampered = verify_persisted_run(run_root)
    assert tampered["status"] == "RED"
    assert "receipt does not bind the upload journal" in tampered["errors"]


def test_final_mirror_copies_atomic_output_on_first_post_exit_observation(tmp_path: Path):
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote = remote_root / "result.json"
    remote.write_text('{"status":"GREEN"}\n')
    ssh_key = tmp_path / "ssh.key"
    api_key = tmp_path / "api.key"
    for path in (ssh_key, api_key):
        path.write_text("secret\n")
        path.chmod(0o600)
    run_root = tmp_path / "run"
    contract = _watchdog_contract(tmp_path)[0]
    contract["run_id"] = "final-copy-test"
    contract["h100"]["ssh_private_key_file"] = str(ssh_key)
    contract["h100"]["instance_api_key_file"] = str(api_key)
    contract["gateway"]["instance_api_key_file"] = str(api_key)
    contract["gateway"]["volume_run_root"] = str(run_root)
    contract["artifacts"]["roots"] = [{"remote_root": str(remote_root), "durable_prefix": "result"}]
    contract_path = tmp_path / "final-contract.json"
    contract_path.write_text(json.dumps(contract))
    backend = GatewayBackend(contract, contract_path)
    stat = remote.stat()
    backend._scan = lambda **_kwargs: [{
        "remote": str(remote), "relative": remote.name, "prefix": "result",
        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }]

    def copy_run(arguments, **_kwargs):
        shutil.copyfile(arguments[-2].split(":", 1)[1], arguments[-1])
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("infra.vast.parallax.lifecycle_supervisor.subprocess.run", side_effect=copy_run):
        assert backend.mirror_once(final=True)["copied_count"] == 1

    backend._scan = lambda **_kwargs: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("scan", 1)
    )
    failed_pass = backend.mirror_once(final=True)
    assert failed_pass["copied_count"] == 0
    assert failed_pass["errors"] == [{"stage": "scan", "type": "TimeoutExpired"}]


def test_job_contract_must_license_fast_final_atomic_copy(tmp_path: Path):
    job = tmp_path / "job.json"
    payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": "lifecycle",
        "uses_gpu": False,
    }
    job.write_text(json.dumps(payload))
    binding = {
        "path": "job.json",
        "sha256": __import__("hashlib").sha256(job.read_bytes()).hexdigest(),
    }
    spec = {"scope": "lifecycle", "h100_research_authorized": False, "authorized_job_spec": binding}
    with pytest.raises(LifecycleError, match="atomic artifact"):
        validate_job_spec(tmp_path, spec, job)
    payload["artifact_protocol"] = "atomic_rename_no_background_writers"
    payload["expected_artifacts"] = {
        "result/result.json": {"bytes": 1, "sha256": "a" * 64}
    }
    payload["runtime"] = {"required_sentinels": ["result/result.json"]}
    job.write_text(json.dumps(payload))
    binding["sha256"] = __import__("hashlib").sha256(job.read_bytes()).hexdigest()
    assert validate_job_spec(tmp_path, spec, job)["uses_gpu"] is False


def test_expected_manifest_comparison_rejects_missing_extra_or_changed_bytes(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    expected = {"result/result.json": {"bytes": 7, "sha256": "a" * 64}}
    manifest.write_text(json.dumps({"entries": [
        {"path": "result/result.json", "bytes": 7, "sha256": "a" * 64}
    ]}))
    assert _expected_manifest_errors(manifest, expected) == []
    manifest.write_text(json.dumps({"entries": [
        {"path": "result/result.json", "bytes": 8, "sha256": "a" * 64},
        {"path": "result/extra.bin", "bytes": 1, "sha256": "b" * 64},
    ]}))
    errors = _expected_manifest_errors(manifest, expected)
    assert errors == [
        "expected artifact mismatch: result/result.json",
        "unexpected artifacts: result/extra.bin",
    ]


def test_content_addressed_dynamic_result_is_allowed_only_when_explicit(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"path": "input-staging/result.json", "bytes": 99, "sha256": "c" * 64}
    ]}))
    assert _expected_manifest_errors(manifest, None) == [
        "exact content-addressed result binding is absent"
    ]
    assert _expected_manifest_errors(
        manifest, None, content_addressed_result_path="input-staging/result.json"
    ) == []

    manifest.write_text(json.dumps({"entries": [
        {"path": "input-staging/result.json", "bytes": 99, "sha256": "c" * 64},
        {"path": "unexpected.bin", "bytes": 1, "sha256": "d" * 64},
    ]}))
    assert _expected_manifest_errors(
        manifest, None, content_addressed_result_path="input-staging/result.json"
    ) == ["exact content-addressed result binding is absent"]


def test_local_artifact_plan_prioritizes_required_and_leaves_large_bytes_on_volume(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"path": "result/required.json", "bytes": 7, "sha256": "a" * 64},
        {"path": "result/optional.json", "bytes": 5, "sha256": "b" * 64},
        {"path": "result/weights.safetensors", "bytes": 50, "sha256": "c" * 64},
    ]}))
    plan = _local_artifact_retrieval_plan(
        manifest_path=manifest,
        required_artifacts=["result/required.json"],
        maximum_file_bytes=10,
        maximum_total_bytes=12,
    )
    assert [row["path"] for row in plan["selected"]] == [
        "result/required.json", "result/optional.json"
    ]
    assert plan["selected_bytes"] == 12
    assert plan["skipped"] == [{
        "path": "result/weights.safetensors", "bytes": 50, "reason": "PER_FILE_LIMIT"
    }]


@pytest.mark.parametrize("path", ["../escape.json", "/absolute.json", "result/has space.json"])
def test_local_artifact_plan_rejects_unsafe_required_paths(tmp_path: Path, path: str):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"path": "result/safe.json", "bytes": 1, "sha256": "a" * 64}
    ]}))
    with pytest.raises(LifecycleError, match="unsafe local artifact"):
        _local_artifact_retrieval_plan(
            manifest_path=manifest,
            required_artifacts=[path],
            maximum_file_bytes=10,
            maximum_total_bytes=10,
        )


def test_local_artifact_retrieval_is_atomic_and_hash_verified(tmp_path: Path):
    content = b'{"status":"GREEN"}\n'
    digest = __import__("hashlib").sha256(content).hexdigest()
    plan = {
        "required_artifacts": ["result/probe.json"],
        "selected": [{"path": "result/probe.json", "bytes": len(content), "sha256": digest}],
        "skipped": [],
        "maximum_file_bytes": 100,
        "maximum_total_bytes": 100,
    }

    def run(arguments, **_kwargs):
        Path(arguments[-1]).write_bytes(content)
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("infra.vast.parallax.guarded_launch._run", side_effect=run):
        receipt = _retrieve_local_artifacts(
            plan=plan,
            scp=["scp"],
            host="gateway",
            volume_run_root="/volume/run",
            output_dir=tmp_path / "inspection",
            copy_timeout_seconds=5,
        )
    assert receipt["status"] == "GREEN"
    assert (tmp_path / "inspection/artifacts/result/probe.json").read_bytes() == content
    assert not (tmp_path / "inspection/artifacts/result/.probe.json.partial").exists()

    bad_content = b"wrong"
    def bad_run(arguments, **_kwargs):
        Path(arguments[-1]).write_bytes(bad_content)
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch(
        "infra.vast.parallax.guarded_launch._run",
        side_effect=bad_run,
    ):
        refused = _retrieve_local_artifacts(
            plan=plan,
            scp=["scp"],
            host="gateway",
            volume_run_root="/volume/run",
            output_dir=tmp_path / "bad-inspection",
            copy_timeout_seconds=5,
        )
    assert refused["status"] == "RED"
    assert refused["required_missing"] == ["result/probe.json"]
    assert not (tmp_path / "bad-inspection/artifacts/result/probe.json").exists()


def test_semantic_readiness_requires_matching_fresh_run_state(tmp_path: Path):
    now = __import__("time").time()
    (tmp_path / "guardian-heartbeat.json").write_text(json.dumps({
        "status": "WATCHING", "run_id": "bound-run", "wall_time_unix": now,
    }))
    (tmp_path / "launch-registry.json").write_text(json.dumps({
        "status": "GUARDIAN_ARMED_WORKER_ABSENT", "run_id": "bound-run", "wall_time_unix": now,
    }))
    command = _semantic_lifecycle_ready_command(
        state_root=str(tmp_path), run_id="bound-run", kind="guardian", maximum_age_seconds=5,
    )
    assert subprocess.run(shlex.split(command), check=False).returncode == 0
    stale = json.loads((tmp_path / "guardian-heartbeat.json").read_text())
    stale["run_id"] = "wrong-run"
    (tmp_path / "guardian-heartbeat.json").write_text(json.dumps(stale))
    assert subprocess.run(shlex.split(command), check=False, capture_output=True).returncode != 0


def test_consumed_v35_detects_post_launch_implementation_change():
    spec = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v35.json"
    with pytest.raises(LifecycleError, match="implementation changed: launcher"):
        validate_frozen_spec(ROOT, spec)


def test_consumed_v36_detects_shared_contract_builder_change():
    spec = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v36.json"
    with pytest.raises(LifecycleError, match="implementation changed: launcher"):
        validate_frozen_spec(ROOT, spec)


def test_v39_inspection_binds_small_exact_gateway_and_larger_bounded_disk():
    path = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v39.json"
    spec = json.loads(path.read_text())
    assert spec["paid_launch_authorized"] is False
    assert spec["paid_inspection_authorized"] is True
    assert spec["gateway_disk_gb"] == 20
    assert spec["inspection_gateway_image_contract"]["compressed_bytes"] == 1_312_938_153


def test_v27_baseline_revokes_every_paid_mode_after_v12_red():
    spec_path = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v27.json"
    spec = json.loads(spec_path.read_text())
    assert spec["lifecycle_worker_kind"] == "non_h100"
    assert spec["h100_research_authorized"] is False
    with pytest.raises(LifecycleError, match="revoked"):
        require_paid_authorization(spec, mode="launch")
    with pytest.raises(LifecycleError, match="revoked"):
        require_paid_authorization(spec, mode="inspect")
    with pytest.raises(LifecycleError, match="revoked"):
        require_paid_authorization(spec, mode="control")


def test_v26_paid_packet_cannot_replay_after_exact_image_gate_lands():
    spec_path = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v26.json"
    with pytest.raises(LifecycleError, match="exact contract"):
        validate_frozen_spec(ROOT, spec_path)


def test_v28_historically_authorized_one_exact_stock_image_v13_plant():
    path = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v28.json"
    spec = json.loads(path.read_text())
    require_paid_authorization(spec, mode="launch")
    assert spec["worker_image"].endswith(spec["worker_image_contract"]["manifest_sha256"])
    assert spec["authorized_job_spec"]["path"].endswith("job_v13.json")
    assert spec["h100_research_authorized"] is False


def test_current_post_probe_red_spec_revokes_paid_paths_and_preserves_confound_bounds():
    spec_path = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v34.json"
    spec = json.loads(spec_path.read_text())
    policy = spec["policy"]
    assert policy["maximum_provider_loading_seconds"] == 600
    assert policy["guardian_poll_seconds"] == 5.0
    assert policy["maximum_worker_prearm_seconds"] == 60
    assert policy["maximum_worker_hard_stop_seconds"] == 150
    assert spec["nonresearch_min_cuda"] == 13.0
    assert spec["h100_research_authorized"] is False
    assert spec["paid_launch_authorized"] is False
    assert spec["worker_image"].endswith(spec["worker_image_contract"]["manifest_sha256"])
    assert spec["worker_image_contract"]["compressed_bytes"] <= 3_100_000_000


def test_paid_worker_refuses_mutable_or_unbound_image_reference(tmp_path: Path):
    source = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v27.json"
    payload = json.loads(source.read_text())
    payload["paid_launch_authorized"] = True
    payload["worker_image"] = "vastai/base-image:stock-ubuntu24.04-py312-2026-06-15"
    candidate = tmp_path / "mutable-image.json"
    candidate.write_text(json.dumps(payload))
    with pytest.raises(LifecycleError, match="exact contract"):
        validate_frozen_spec(ROOT, candidate)


def test_future_h100_scope_refuses_without_exact_green_lifecycle_packet(tmp_path: Path):
    source = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v37.json"
    payload = json.loads(source.read_text())
    payload["scope"] = "future_h100_unit"
    payload["paid_launch_authorized"] = True
    payload["paid_inspection_authorized"] = False
    payload["scoped_lifecycle_api_permissions"] = {
        "api": {
            "instance_read": {"api.instances": {}},
            "instance_write": {"api.instance": {}},
        }
    }
    payload["policy"]["maximum_worker_prearm_seconds"] = 120
    payload["policy"]["maximum_prearm_worker_cost_usd"] = 1.0
    payload["policy"]["maximum_worker_hard_stop_seconds"] = 210
    payload["policy"]["maximum_gateway_lifecycle_cost_usd"] = 1.0
    payload["one_shot_authorization"] = {
        "id": "unit-future-launch",
        "claim_paths": {"launch": "studies/lineage/cross-lineage/results/readiness/claims/unit.json"},
    }
    payload["h100_research_authorized"] = True
    payload["implementation"]["launcher"]["sha256"] = sha256_file(
        ROOT / "infra/vast/parallax/guarded_launch.py"
    )
    candidate = tmp_path / "research-spec.json"
    candidate.write_text(json.dumps(payload))
    with pytest.raises(LifecycleError, match="GO launch packet"):
        validate_frozen_spec(ROOT, candidate)


def test_runtime_probe_paid_spec_refuses_without_cuda13_host_floor(tmp_path: Path):
    source = ROOT / "studies/lineage/cross-lineage/config/parallax_vast_lifecycle_spec_v33.json"
    payload = json.loads(source.read_text())
    payload["nonresearch_min_cuda"] = 0
    candidate = tmp_path / "bad-host-floor.json"
    candidate.write_text(json.dumps(payload))
    with pytest.raises(LifecycleError, match="host-driver floor"):
        validate_frozen_spec(ROOT, candidate)


def test_paid_worker_admission_requires_exact_green_live_control_receipt(tmp_path: Path):
    receipt = tmp_path / "control.json"
    payload = {
        "status": "GREEN",
        "relay_ssh": "GREEN",
        "endpoint_scoped_STOP": {"status": "STOPPED"},
        "scoped_key_authority": {
            "status": "GREEN",
            "volume_endpoint_denied": True,
        },
        "final_account_audit": {"active_instances": 0},
        "H100_created": False,
        "gpu_work_executed": False,
    }
    receipt.write_text(json.dumps(payload))
    spec = {
        "required_live_control_evidence": {
            "path": "control.json",
            "sha256": __import__("hashlib").sha256(receipt.read_bytes()).hexdigest(),
        }
    }
    assert _validate_live_control_evidence(tmp_path, spec) == payload
    for field, replacement in (
        ("status", "RED"),
        ("relay_ssh", "RED"),
        ("H100_created", True),
        ("gpu_work_executed", True),
    ):
        broken = dict(payload)
        broken[field] = replacement
        receipt.write_text(json.dumps(broken))
        spec["required_live_control_evidence"]["sha256"] = __import__("hashlib").sha256(
            receipt.read_bytes()
        ).hexdigest()
        with pytest.raises(LifecycleError, match="not GREEN"):
            _validate_live_control_evidence(tmp_path, spec)


def test_lifecycle_big_run_packet_requires_every_live_control(tmp_path: Path):
    lifecycle = {
        "provider": "vast.ai",
        "launcher": "infra.vast.parallax.guarded_launch",
        "direct_vast_create_forbidden": True,
        "h100_api_key_scope": "per_instance_only",
        "local_controller_required_after_arm": False,
        "maximum_mirror_interval_seconds": 1,
        "maximum_idle_gpu_seconds_after_durable_result": 5,
        "maximum_bootstrap_orphan_seconds": 90,
    }
    for name in (
        "weight_free_plants", "continuous_mirror", "detached_watchdog", "provider_ttl_guard",
        "bootstrap_orphan_guard", "durability_before_destroy", "stop_preserves_disk", "live_rehearsal",
    ):
        lifecycle[name] = {"status": "GREEN"}
    payload = {"lifecycle": lifecycle}
    with patch("infra.vast.parallax.lifecycle_launch_packet.validate_launch_packet", return_value=payload):
        assert validate_lifecycle_launch_packet(tmp_path / "packet.yaml") is payload
        lifecycle["live_rehearsal"] = {"status": "RED"}
        with pytest.raises(LaunchPacketError, match="live_rehearsal"):
            validate_lifecycle_launch_packet(tmp_path / "packet.yaml")


def test_launcher_itself_selects_cheapest_distinct_live_offers():
    spec = {"gateway_disk_gb": 10, "h100_disk_gb": 100, "h100_research_authorized": False}

    def vast(*args):
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        query = args[2]
        if "machine_id=9" in query:
            return [
                {"id": 2, "gpu_name": "T4", "dph_total": 0.2, "reliability": 0.99},
                {"id": 1, "gpu_name": "T4", "dph_total": 0.1, "reliability": 0.99},
            ]
        return [
            {"id": 1, "gpu_name": "T4", "dph_total": 0.01, "reliability": 0.99},
            {"id": 3, "gpu_name": "H100 PCIe", "dph_total": 0.02, "reliability": 0.99},
            {"id": 4, "gpu_name": "RTX 3060", "dph_total": 0.03, "reliability": 0.99},
        ]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_offers(spec=spec, volume_id=44)
    assert receipt["selected_gateway"]["id"] == 1
    assert receipt["selected_worker"]["id"] == 4
    assert receipt["all_regions"] is True


def test_nonresearch_worker_uses_same_direct_connectivity_gate_as_h100():
    spec = {
        "gateway_disk_gb": 10,
        "h100_disk_gb": 10,
        "h100_research_authorized": False,
        "lifecycle_worker_kind": "non_h100",
        "nonresearch_min_reliability": 0.99,
    }
    queries = []

    def vast(*args):
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        queries.append(args[2])
        if "machine_id=9" in args[2]:
            return [{"id": 1, "gpu_name": "T4", "dph_total": 0.1, "reliability": 0.99}]
        return [{"id": 2, "gpu_name": "RTX 3060", "dph_total": 0.05, "reliability": 0.999}]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_offers(spec=spec, volume_id=44)
    assert "cpu_arch=amd64" in receipt["worker_query"]
    assert "static_ip=true" in receipt["worker_query"]
    assert "direct_port_count>=2" in receipt["worker_query"]


def test_h100_infrastructure_probe_filters_driver_and_downlink_before_cost_sorting():
    spec = {
        "gateway_disk_gb": 10,
        "h100_disk_gb": 25,
        "h100_research_authorized": False,
        "lifecycle_worker_kind": "h100",
        "nonresearch_min_cuda": 13.0,
        "nonresearch_min_inet_down_mbps": 500.0,
    }

    def vast(*args):
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        if "machine_id=9" in args[2]:
            return [{"id": 1, "gpu_name": "T4", "dph_total": 0.1, "reliability": 0.99}]
        assert "cuda_max_good>=13" in args[2]
        assert "inet_down>=500" in args[2]
        return [{
            "id": 2, "machine_id": 20, "gpu_name": "H100 PCIE", "dph_total": 1.9,
            "reliability": 0.99, "cuda_max_good": 13.0, "inet_down": 3000.0,
        }]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_offers(spec=spec, volume_id=44)
    assert receipt["selected_worker"]["gpu_name"] == "H100 PCIE"


def test_runtime_image_probe_filters_host_driver_before_cost_sorting():
    spec = {
        "gateway_disk_gb": 10,
        "h100_disk_gb": 20,
        "h100_research_authorized": False,
        "lifecycle_worker_kind": "non_h100",
        "nonresearch_min_reliability": 0.99,
        "nonresearch_min_cuda": 13.0,
    }

    def vast(*args):
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        if "machine_id=9" in args[2]:
            return [{"id": 1, "gpu_name": "T4", "dph_total": 0.1, "reliability": 0.99}]
        assert "cuda_max_good>=13" in args[2]
        return [
            {
                "id": 2, "machine_id": 20, "gpu_name": "RTX 3060", "dph_total": 0.01,
                "reliability": 0.999, "cuda_max_good": 12.4,
            },
            {
                "id": 3, "machine_id": 30, "gpu_name": "RTX 3060", "dph_total": 0.02,
                "reliability": 0.999, "cuda_max_good": 13.0,
            },
        ]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_offers(spec=spec, volume_id=44)
    assert receipt["selected_worker"]["id"] == 3
    assert receipt["nonresearch_min_cuda"] == 13.0
    assert receipt["worker_selection_rule"].startswith(
        "literal_cheapest_suitable_cuda_compatible"
    )


def test_research_market_selection_is_h100_only_and_cost_sorted():
    spec = {"gateway_disk_gb": 10, "h100_disk_gb": 100, "h100_research_authorized": True}

    def vast(*args):
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        if "machine_id=9" in args[2]:
            return [{"id": 1, "gpu_name": "T4", "dph_total": 0.1, "reliability": 0.99}]
        assert "H100" in args[2]
        return [
            {"id": 8, "gpu_name": "H100 SXM", "dph_total": 2.0, "reliability": 0.999},
            {"id": 7, "gpu_name": "H100 PCIe", "dph_total": 1.8, "reliability": 0.99},
        ]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_offers(spec=spec, volume_id=44)
    assert receipt["selected_worker"]["id"] == 7
    assert receipt["worker_selection_rule"] == "literal_cheapest_suitable_h100_all_regions_at_query_time"


def test_live_failed_worker_machine_is_excluded_before_cost_sorting():
    spec = {
        "gateway_disk_gb": 10,
        "h100_disk_gb": 10,
        "h100_research_authorized": False,
        "lifecycle_worker_kind": "h100",
        "excluded_worker_offer_ids": [70],
        "excluded_worker_machine_ids": [900],
    }

    def vast(*args):
        if args[:2] == ("show", "volumes"):
            return [{"id": 44, "machine_id": 9}]
        if "machine_id=9" in args[2]:
            return [{"id": 1, "gpu_name": "T4", "dph_total": 0.1, "reliability": 0.99}]
        return [
            {"id": 70, "machine_id": 800, "gpu_name": "H100 SXM", "dph_total": 1.0, "reliability": 0.99},
            {"id": 71, "machine_id": 900, "gpu_name": "H100 SXM", "dph_total": 1.1, "reliability": 0.99},
            {"id": 72, "machine_id": 901, "gpu_name": "H100 SXM", "dph_total": 1.2, "reliability": 0.99},
        ]

    with patch("infra.vast.parallax.guarded_launch._vast", side_effect=vast):
        receipt = select_live_offers(spec=spec, volume_id=44)
    assert receipt["selected_worker"]["id"] == 72
    assert receipt["excluded_worker_offer_ids"] == [70]
    assert receipt["excluded_worker_machine_ids"] == [900]


def _launch_spec_and_job():
    spec = {
        "image": "test-image",
        "gateway_disk_gb": 10,
        "h100_disk_gb": 100,
        "volume_mount": "/volume",
        "bootstrap_ssh_public_key_id": 77,
        "relay_ssh_public_key_id": 88,
        "authorized_volume_id": 44,
        "worker_price_ceiling_dph": 1.0,
        "gateway_price_ceiling_dph": 1.0,
        "h100_research_authorized": True,
        "paid_launch_authorized": True,
        "policy": {
            "bootstrap_timeout_seconds": 60,
            "bootstrap_orphan_grace_seconds": 30,
            "maximum_provider_loading_seconds": 60,
            "maximum_runtime_seconds": 60,
            "maximum_worker_prearm_seconds": 30,
            "maximum_gateway_lifecycle_cost_usd": 1.0,
            "maximum_worker_hard_stop_seconds": 100,
            "scoped_stop_proof_seconds": 5,
            "mirror_interval_seconds": 1,
            "heartbeat_timeout_seconds": 5,
            "artifact_copy_timeout_seconds": 5,
            "maximum_local_inspection_file_bytes": 10_000_000,
            "maximum_local_inspection_artifact_bytes": 50_000_000,
            "maximum_finalization_seconds": 5,
            "final_sync_attempts": 2,
            "final_sync_settle_seconds": 0,
        },
    }
    job = {
        "run_id": "unit-launch",
        "required_local_artifacts": ["result/result.json"],
        "runtime": {
            "job": {"execution_site": "h100", "cwd": "/tmp", "command": ["true"]},
            "artifact_roots": [{"remote_root": "/tmp/output", "durable_prefix": "result"}],
            "required_sentinels": ["result/result.json"],
        },
    }
    market = {
        "selected_gateway": {"id": 10, "dph_total": 0.1},
        "selected_worker": {"id": 20, "dph_total": 0.2},
    }
    return spec, job, market


class _MockLocalReaper:
    armed = False

    def __init__(self, *_args, **_kwargs):
        type(self).armed = True

    def handoff(self, *, run_id):
        return {"status": "HANDED_OFF_TO_REMOTE_GUARDIAN", "run_id": run_id}

    def release_to_reap(self):
        pass


def test_bootstrap_failure_destroys_every_created_instance_without_account_key_mutation(tmp_path: Path):
    spec, job, market = _launch_spec_and_job()
    vast_calls = []
    lifecycle_key = tmp_path / "lifecycle.key"
    lifecycle_key.write_text("scoped-key\n")
    claim = tmp_path / "claim.json"
    claim.write_text("{}\n")

    def create(offer_id, _arguments):
        if offer_id == 10:
            return {"new_contract": 101, "instance_api_key": "gateway-key", "success": True}
        raise LifecycleError("planted worker create failure")

    with (
        patch("infra.vast.parallax.guarded_launch.validate_frozen_spec", return_value=spec),
        patch("infra.vast.parallax.guarded_launch.validate_job_spec", return_value=job),
        patch("infra.vast.parallax.guarded_launch.select_live_offers", return_value=market),
        patch("infra.vast.parallax.guarded_launch._claim_one_shot_authorization", return_value=claim),
        patch("infra.vast.parallax.guarded_launch._validated_bootstrap_ssh_key", return_value=tmp_path / "bootstrap.key"),
        patch("infra.vast.parallax.guarded_launch._validated_registered_ssh_key", return_value=tmp_path / "relay.key"),
        patch(
            "infra.vast.parallax.guarded_launch._validated_scoped_lifecycle_api_key",
            return_value=(lifecycle_key, 99),
        ),
        patch("infra.vast.parallax.guarded_launch._create_instance", side_effect=create),
        patch("infra.vast.parallax.guarded_launch.wait_ssh", return_value={"ssh_host": "host", "ssh_port": 22}),
        patch("infra.vast.parallax.guarded_launch.BootstrapLease"),
        patch("infra.vast.parallax.guarded_launch.LocalBootstrapReaper", _MockLocalReaper),
        patch(
            "infra.vast.parallax.guarded_launch._run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._vast",
            side_effect=lambda *args, **kwargs: [] if args == ("show", "instances") else vast_calls.append(args),
        ),
        pytest.raises(LifecycleError, match="planted worker"),
    ):
        launch(
            root=ROOT,
            spec_path=tmp_path / "spec.json",
            job_spec_path=tmp_path / "job.json",
            volume_id=44,
            output=tmp_path / "launch.json",
        )
    assert ("stop", "instance", "101") in vast_calls
    assert vast_calls.index(("stop", "instance", "101")) < vast_calls.index(("destroy", "instance", "101"))
    assert not any(call[:2] in {("create", "ssh-key"), ("delete", "ssh-key")} for call in vast_calls)
    failure = json.loads((tmp_path / "launch.json").read_text())
    assert failure["status"] == "RED_LAUNCH_FAILED_ACCOUNT_ZERO"
    assert failure["stage"] == "CREATING_AND_QUALIFYING_WORKER"
    assert failure["automatic_retry"] is False
    assert failure["final_account_audit"]["active_instances"] == 0


def test_fully_mocked_happy_path_arms_without_account_key_mutation_or_local_cleanup(tmp_path: Path):
    spec, job, market = _launch_spec_and_job()
    spec["h100_research_authorized"] = False
    spec["lifecycle_worker_kind"] = "h100"
    spec["gateway_image"] = "example/gateway@sha256:" + "a" * 64
    spec["worker_image"] = "example/worker@sha256:" + "b" * 64
    spec_path = tmp_path / "spec.json"
    job_path = tmp_path / "job.json"
    spec_path.write_text("{}\n")
    job_path.write_text("{}\n")
    output = tmp_path / "launch.json"
    claim = tmp_path / "claim.json"
    claim.write_text("{}\n")
    relay = tmp_path / "relay.key"
    relay.write_text("RELAY PRIVATE\n")
    lifecycle_key = tmp_path / "lifecycle.key"
    lifecycle_key.write_text("scoped-key\n")
    vast_calls = []
    run_calls = []
    creation_order = []
    creation_arguments = []
    ready_timeouts = []
    guardian_started = False
    _MockLocalReaper.armed = False

    def create(offer_id, _arguments):
        nonlocal guardian_started
        assert _MockLocalReaper.armed, "gateway was created before the detached local reaper"
        creation_order.append(offer_id)
        creation_arguments.append(_arguments)
        if offer_id == 10:
            return {"new_contract": 101, "instance_api_key": "gateway-key", "success": True}
        assert guardian_started, "H100 was created before the independent gateway guardian"
        return {"new_contract": 202, "instance_api_key": "worker-key", "success": True}

    def run(arguments, **_kwargs):
        nonlocal guardian_started
        run_calls.append(arguments)
        if any("lifecycle_guardian" in str(value) for value in arguments):
            guardian_started = True
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    def ready(instance_id, *_args, **_kwargs):
        ready_timeouts.append(_kwargs["timeout_seconds"])
        return {"ssh_host": f"host-{instance_id}", "ssh_port": 2200 + instance_id}

    class Lease:
        def __init__(self, _ssh):
            pass

        def assert_healthy(self):
            pass

        def close(self):
            pass

    completed = subprocess.CompletedProcess([], 0, b"", b"")
    with (
        patch("infra.vast.parallax.guarded_launch.validate_frozen_spec", return_value=spec),
        patch("infra.vast.parallax.guarded_launch.validate_job_spec", return_value=job),
        patch("infra.vast.parallax.guarded_launch.select_live_offers", return_value=market),
        patch("infra.vast.parallax.guarded_launch._claim_one_shot_authorization", return_value=claim),
        patch("infra.vast.parallax.guarded_launch._validated_bootstrap_ssh_key", return_value=tmp_path / "bootstrap.key"),
        patch("infra.vast.parallax.guarded_launch._validated_registered_ssh_key", return_value=relay),
        patch(
            "infra.vast.parallax.guarded_launch._validated_scoped_lifecycle_api_key",
            return_value=(lifecycle_key, 99),
        ),
        patch("infra.vast.parallax.guarded_launch._regenerate_remote_instance_api_key", return_value="a" * 64),
        patch("infra.vast.parallax.guarded_launch._create_instance", side_effect=create),
        patch("infra.vast.parallax.guarded_launch.wait_ssh", side_effect=ready),
        patch("infra.vast.parallax.guarded_launch.BootstrapLease", Lease),
        patch("infra.vast.parallax.guarded_launch.LocalBootstrapReaper", _MockLocalReaper),
        patch("infra.vast.parallax.guarded_launch._run", side_effect=run),
        patch("infra.vast.parallax.guarded_launch._vast", side_effect=lambda *args, **kwargs: vast_calls.append(args)),
    ):
        packet = launch(
            root=ROOT,
            spec_path=spec_path,
            job_spec_path=job_path,
            volume_id=44,
            output=output,
        )

    assert packet["status"] == "SUPERVISOR_RELEASED"
    assert packet["account_ssh_key_mutations"] == 0
    assert packet["bootstrap_ssh_public_key_id"] == 77
    assert packet["relay_ssh_public_key_id"] == 88
    assert packet["scoped_lifecycle_api_key_id"] == 99
    assert creation_order == [10, 20]
    assert creation_arguments[0][1] == spec["gateway_image"]
    assert creation_arguments[1][1] == spec["worker_image"]
    assert ready_timeouts == [60, 90]
    assert guardian_started is True
    supervisor_index = next(
        index for index, arguments in enumerate(run_calls)
        if any("lifecycle_supervisor" in str(value) for value in arguments)
    )
    release_index = next(
        index for index, arguments in enumerate(run_calls)
        if any("execution-release.marker" in str(value) for value in arguments)
    )
    assert supervisor_index < release_index
    assert any(
        any("command -v bash python3 rsync ssh sha256sum" in str(value) for value in arguments)
        for arguments in run_calls
    )
    assert packet["local_bootstrap_reaper"]["status"] == "HANDED_OFF_TO_REMOTE_GUARDIAN"
    assert output.is_file()
    assert vast_calls == []


def test_worker_image_command_gate_failure_is_durable_red_and_account_zero(tmp_path: Path):
    spec, job, market = _launch_spec_and_job()
    spec["h100_research_authorized"] = False
    spec["lifecycle_worker_kind"] = "non_h100"
    spec["worker_image"] = "example/runtime@sha256:" + "b" * 64
    claim = tmp_path / "claim.json"
    claim.write_text("{}\n")
    lifecycle_key = tmp_path / "lifecycle.key"
    lifecycle_key.write_text("scoped-key\n")
    vast_calls = []
    created = iter((101, 202))

    class Lease:
        def __init__(self, _ssh): pass
        def assert_healthy(self): pass
        def close(self): pass

    def create(_offer_id, _arguments):
        return {"new_contract": next(created), "instance_api_key": "instance-key", "success": True}

    with (
        patch("infra.vast.parallax.guarded_launch.validate_frozen_spec", return_value=spec),
        patch("infra.vast.parallax.guarded_launch.validate_job_spec", return_value=job),
        patch("infra.vast.parallax.guarded_launch.select_live_offers", return_value=market),
        patch("infra.vast.parallax.guarded_launch._claim_one_shot_authorization", return_value=claim),
        patch("infra.vast.parallax.guarded_launch._validated_bootstrap_ssh_key", return_value=tmp_path / "bootstrap"),
        patch("infra.vast.parallax.guarded_launch._validated_registered_ssh_key", return_value=tmp_path / "relay"),
        patch(
            "infra.vast.parallax.guarded_launch._validated_scoped_lifecycle_api_key",
            return_value=(lifecycle_key, 99),
        ),
        patch("infra.vast.parallax.guarded_launch._create_instance", side_effect=create),
        patch(
            "infra.vast.parallax.guarded_launch.wait_ssh",
            side_effect=lambda instance_id, *_args, **_kwargs: {
                "ssh_host": f"host-{instance_id}", "ssh_port": 2200 + instance_id
            },
        ),
        patch("infra.vast.parallax.guarded_launch.BootstrapLease", Lease),
        patch("infra.vast.parallax.guarded_launch.LocalBootstrapReaper", _MockLocalReaper),
        patch(
            "infra.vast.parallax.guarded_launch._run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._require_worker_image_capabilities",
            side_effect=LifecycleError("worker image failed the frozen live command gate: rsync not found"),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._vast",
            side_effect=lambda *args, **_kwargs: vast_calls.append(args),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._wait_account_zero",
            return_value={"query_succeeded": True, "active_instances": 0, "instance_ids": []},
        ),
        pytest.raises(LifecycleError, match="rsync not found"),
    ):
        launch(
            root=ROOT,
            spec_path=tmp_path / "spec.json",
            job_spec_path=tmp_path / "job.json",
            volume_id=44,
            output=tmp_path / "image-probe-launch.json",
        )
    packet = json.loads((tmp_path / "image-probe-launch.json").read_text())
    assert packet["status"] == "RED_LAUNCH_FAILED_ACCOUNT_ZERO"
    assert packet["stage"] == "WORKER_IMAGE_LIVE_COMMAND_GATE"
    assert packet["gpu_work_released"] is False
    assert packet["automatic_retry"] is False
    assert packet["created_instance_ids"] == [101, 202]
    assert ("stop", "instance", "202") in vast_calls
    assert ("destroy", "instance", "101") in vast_calls


def test_gateway_control_probe_proves_relay_and_exact_stop_then_account_cleanup(tmp_path: Path):
    spec, _job, market = _launch_spec_and_job()
    spec["paid_control_probe_authorized"] = True
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}\n")
    claim = tmp_path / "claim.json"
    claim.write_text("{}\n")
    output = tmp_path / "control.json"
    vast_calls = []
    run_calls = []
    lifecycle_key = tmp_path / "lifecycle.key"
    lifecycle_key.write_text("endpoint-scoped-secret\n")

    class Lease:
        def __init__(self, _ssh):
            pass

        def close(self):
            pass

    completed = subprocess.CompletedProcess([], 0, b"", b"")

    def run(arguments, **_kwargs):
        run_calls.append(arguments)
        return completed

    with (
        patch("infra.vast.parallax.guarded_launch.validate_frozen_spec", return_value=spec),
        patch("infra.vast.parallax.guarded_launch._claim_one_shot_authorization", return_value=claim),
        patch("infra.vast.parallax.guarded_launch.select_live_offers", return_value=market),
        patch("infra.vast.parallax.guarded_launch._validated_bootstrap_ssh_key", return_value=tmp_path / "bootstrap"),
        patch("infra.vast.parallax.guarded_launch._validated_registered_ssh_key", return_value=tmp_path / "relay"),
        patch(
            "infra.vast.parallax.guarded_launch._validated_scoped_lifecycle_api_key",
            return_value=(lifecycle_key, 99),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._probe_scoped_key_authority",
            return_value={"status": "GREEN", "volume_endpoint_denied": True},
        ),
        patch("infra.vast.parallax.guarded_launch._create_instance", return_value={"new_contract": 303}),
        patch("infra.vast.parallax.guarded_launch.wait_ssh", return_value={"ssh_host": "host", "ssh_port": 22}),
        patch("infra.vast.parallax.guarded_launch.BootstrapLease", Lease),
        patch("infra.vast.parallax.guarded_launch._run", side_effect=run),
        patch(
            "infra.vast.parallax.guarded_launch._wait_instance_stopped",
            return_value={"status": "STOPPED", "operator": "endpoint_scoped_deadline"},
        ),
        patch("infra.vast.parallax.guarded_launch._vast", side_effect=lambda *args, **kwargs: vast_calls.append(args)),
        patch(
            "infra.vast.parallax.guarded_launch._wait_account_zero",
            return_value={"active_instances": 0, "instance_ids": [], "query_succeeded": True},
        ),
    ):
        packet = probe_gateway_controls(
            root=ROOT,
            spec_path=spec_path,
            volume_id=44,
            output=output,
        )

    assert packet["status"] == "GREEN"
    assert packet["relay_ssh"] == "GREEN"
    assert packet["endpoint_scoped_STOP"]["status"] == "STOPPED"
    assert packet["scoped_key_authority"]["volume_endpoint_denied"] is True
    assert any("parallax-scoped-stop-proof.key" in str(call) for call in run_calls)
    assert packet["final_account_audit"]["active_instances"] == 0
    assert packet["H100_created"] is False and packet["gpu_work_executed"] is False
    assert ("destroy", "instance", "303") in vast_calls
    assert output.is_file()


def test_gateway_control_probe_fails_closed_if_account_is_not_zero(tmp_path: Path):
    spec, _job, market = _launch_spec_and_job()
    spec["paid_control_probe_authorized"] = True
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}\n")
    claim = tmp_path / "claim.json"
    claim.write_text("{}\n")
    output = tmp_path / "control.json"
    lifecycle_key = tmp_path / "lifecycle.key"
    lifecycle_key.write_text("endpoint-scoped-secret\n")

    class Lease:
        def __init__(self, _ssh): pass
        def close(self): pass

    completed = subprocess.CompletedProcess([], 0, b"", b"")
    with (
        patch("infra.vast.parallax.guarded_launch.validate_frozen_spec", return_value=spec),
        patch("infra.vast.parallax.guarded_launch._claim_one_shot_authorization", return_value=claim),
        patch("infra.vast.parallax.guarded_launch.select_live_offers", return_value=market),
        patch("infra.vast.parallax.guarded_launch._validated_bootstrap_ssh_key", return_value=tmp_path / "bootstrap"),
        patch("infra.vast.parallax.guarded_launch._validated_registered_ssh_key", return_value=tmp_path / "relay"),
        patch(
            "infra.vast.parallax.guarded_launch._validated_scoped_lifecycle_api_key",
            return_value=(lifecycle_key, 99),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._probe_scoped_key_authority",
            return_value={"status": "GREEN", "volume_endpoint_denied": True},
        ),
        patch("infra.vast.parallax.guarded_launch._create_instance", return_value={"new_contract": 303}),
        patch("infra.vast.parallax.guarded_launch.wait_ssh", return_value={"ssh_host": "host", "ssh_port": 22}),
        patch("infra.vast.parallax.guarded_launch.BootstrapLease", Lease),
        patch("infra.vast.parallax.guarded_launch._run", return_value=completed),
        patch(
            "infra.vast.parallax.guarded_launch._wait_instance_stopped",
            return_value={"status": "STOPPED", "operator": "endpoint_scoped_deadline"},
        ),
        patch("infra.vast.parallax.guarded_launch._vast"),
        patch(
            "infra.vast.parallax.guarded_launch._wait_account_zero",
            return_value={"active_instances": 1, "instance_ids": [303], "query_succeeded": True},
        ),
        pytest.raises(LifecycleError, match="account to zero"),
    ):
        probe_gateway_controls(root=ROOT, spec_path=spec_path, volume_id=44, output=output)
    assert json.loads(output.read_text())["status"] == "RED_ACCOUNT_NOT_ZERO"


@pytest.mark.parametrize("active_instances", [0, 1])
def test_volume_inspection_persists_and_enforces_account_zero_after_gateway_cleanup(
    tmp_path: Path, active_instances: int
):
    spec, job, market = _launch_spec_and_job()
    spec["paid_launch_authorized"] = False
    spec["paid_inspection_authorized"] = True
    spec["authorized_job_spec"] = {"path": "job.json", "sha256": "ignored-by-mock"}
    spec["policy"]["inspection_deadman_seconds"] = 180
    result_bytes = b"1234567"
    result_sha256 = __import__("hashlib").sha256(result_bytes).hexdigest()
    job["expected_artifacts"] = {
        "result/result.json": {"bytes": len(result_bytes), "sha256": result_sha256}
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}\n")
    claim = tmp_path / "claim.json"
    claim.write_text("{}\n")
    lifecycle_key = tmp_path / "lifecycle.key"
    lifecycle_key.write_text("endpoint-scoped-secret\n")
    output_dir = tmp_path / "inspection"
    verification = {"status": "GREEN", "errors": []}
    manifest = {
        "entries": [
            {"path": path, "bytes": row["bytes"], "sha256": row["sha256"]}
            for path, row in job["expected_artifacts"].items()
        ]
    }
    run_calls = []
    vast_calls = []

    class Lease:
        def __init__(self, _ssh): pass
        def close(self): pass

    class Reaper:
        def __init__(self, *_args, **_kwargs): pass
        def handoff(self, *, run_id, target):
            return {"status": "HANDED_OFF", "run_id": run_id, "target": target}
        def release_to_reap(self): pass

    completed = subprocess.CompletedProcess([], 0, b"", b"")

    def run(arguments, **_kwargs):
        run_calls.append(arguments)
        destination = Path(str(arguments[-1]))
        is_scp = arguments[0] == "scp"
        if is_scp and destination.name == "reattach-verification.json":
            destination.write_text(json.dumps(verification))
        elif is_scp and destination.name == "manifest.json":
            destination.write_text(json.dumps(manifest))
        elif is_scp and destination.name == "receipt.json":
            destination.write_text(json.dumps({"status": "DURABLE_AND_DESTROYED"}))
        elif is_scp and destination.name == ".result.json.partial":
            destination.write_bytes(result_bytes)
        return completed

    with (
        patch("infra.vast.parallax.guarded_launch.validate_frozen_spec", return_value=spec),
        patch("infra.vast.parallax.guarded_launch.validate_job_spec", return_value=job),
        patch("infra.vast.parallax.guarded_launch._claim_one_shot_authorization", return_value=claim),
        patch("infra.vast.parallax.guarded_launch.select_live_offers", return_value=market),
        patch("infra.vast.parallax.guarded_launch._validated_bootstrap_ssh_key", return_value=tmp_path / "bootstrap"),
        patch(
            "infra.vast.parallax.guarded_launch._validated_scoped_lifecycle_api_key",
            return_value=(lifecycle_key, 99),
        ),
        patch(
            "infra.vast.parallax.guarded_launch._create_instance",
            return_value={"new_contract": 404},
        ) as create_instance,
        patch("infra.vast.parallax.guarded_launch.wait_ssh", return_value={"ssh_host": "host", "ssh_port": 22}),
        patch("infra.vast.parallax.guarded_launch.BootstrapLease", Lease),
        patch("infra.vast.parallax.guarded_launch._normalized_lifecycle_archive"),
        patch("infra.vast.parallax.guarded_launch._run", side_effect=run),
        patch("infra.vast.parallax.guarded_launch._vast", side_effect=lambda *args, **kwargs: vast_calls.append(args)),
        patch(
            "infra.vast.parallax.guarded_launch._wait_account_zero",
            return_value={
                "active_instances": active_instances,
                "instance_ids": [404] if active_instances else [],
                "query_succeeded": True,
            },
        ),
        patch("infra.vast.parallax.guarded_launch.LocalBootstrapReaper", Reaper),
    ):
        if active_instances:
            with pytest.raises(LifecycleError, match="account to zero"):
                inspect_persisted_run(
                    root=tmp_path,
                    spec_path=spec_path,
                    volume_id=44,
                    run_id="unit-inspection",
                    output_dir=output_dir,
                )
        else:
            packet = inspect_persisted_run(
                root=tmp_path,
                spec_path=spec_path,
                volume_id=44,
                run_id="unit-inspection",
                output_dir=output_dir,
            )
            assert packet["status"] == "GREEN"

    audit = json.loads((output_dir / "final-account-audit.json").read_text())
    persisted = json.loads((output_dir / "inspection.json").read_text())
    assert audit["active_instances"] == active_instances
    assert persisted["final_account_audit"]["active_instances"] == active_instances
    assert persisted["status"] == ("RED_ACCOUNT_NOT_ZERO" if active_instances else "GREEN")
    assert ("stop", "instance", "404") in vast_calls
    assert ("destroy", "instance", "404") in vast_calls
    assert len(create_instance.call_args.args) == 2
    assert create_instance.call_args.args[0] == 10
