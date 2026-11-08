from __future__ import annotations

import ast
import base64
import copy
import hashlib
import inspect
import importlib.util
import io
import json
import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from infra.vast.parallax import pg1_execution_capsule as capsule_module
from infra.vast.parallax import pg1_native_backend as backend_module
from infra.vast.parallax import pg1_native_bootstrap as bootstrap
from infra.vast.parallax import pg1_native_cell as cell
from infra.vast.parallax import pg1_native_control_deadman as control_deadman
from infra.vast.parallax import pg1_native_controller as controller
from infra.vast.parallax import pg1_native_critic as critic
from infra.vast.parallax import pg1_native_deployment as deployment
from infra.vast.parallax import pg1_native_deadman as deadman
from infra.vast.parallax import pg1_native_job as job
from infra.vast.parallax import pg1_native_launcher as launcher
from infra.vast.parallax import pg1_native_launch_gateway as launch_gateway
from infra.vast.parallax import pg1_native_launch_authority as launch_authority
from infra.vast.parallax import pg1_native_launch_invoker as launch_invoker
from infra.vast.parallax import pg1_native_launch_packet as launch_packet
from infra.vast.parallax import pg1_native_launch_state as launch_state
from infra.vast.parallax import pg1_native_mechanical_rebase as mechanical_rebase
from infra.vast.parallax import pg1_native_idle_bootstrap as idle_bootstrap
from infra.vast.parallax import pg1_native_onstart_materializer as onstart_materializer
from infra.vast.parallax import pg1_native_control_installer as control_installer
from infra.vast.parallax import pg1_native_guardian as guardian
from infra.vast.parallax import pg1_native_lifecycle as lifecycle_module
from infra.vast.parallax import pg1_native_provider as provider
from infra.vast.parallax import pg1_native_vast_adapter as vast_adapter
from infra.vast.parallax import pg1_native_overlay_stage as overlay
from infra.vast.parallax import pg1_native_source_gate as source_gate
from infra.vast.parallax import pg1_native_worker as worker
from infra.vast.parallax import pg1_science_ast_guard as science_guard
from infra.vast.parallax.pg1_execution_capsule import CapsuleError
from infra.vast.parallax.pg1_native_authority import ProductionAuthorizationStore
from infra.vast.parallax.pg1_native_rehearsal import (
    DeterministicAuthorizationStore,
    compile_mechanical_contract,
)


ROOT = Path(__file__).resolve().parents[4]
VAST_SHAPES = json.loads(
    (
        ROOT
        / "infra/vast/parallax/tests/fixtures/pg1_vast_tracked_real_shapes_v1.json"
    ).read_text(encoding="utf-8")
)
ATTEMPT_IDS = (
    "pg1-native-future-source-test-a",
    "pg1-native-future-2041-01",
    "arbitrary-later-attempt-7f3c",
)
LEGACY_RESTORED = (
    "infra/vast/parallax/audit_primary_interface_independent.py",
    "infra/vast/parallax/primary_interface_control_job.py",
    "infra/vast/parallax/primary_interface_control_launch.py",
    "infra/vast/parallax/primary_interface_control_overlay_stage.py",
    "infra/vast/parallax/primary_interface_control_source_gate.py",
    "infra/vast/parallax/primary_interface_control_worker.py",
    "infra/vast/parallax/training_interface_control_cell.py",
    "infra/vast/parallax/training_interface_control_primary_cell.py",
)


def _capsule(attempt_run_id: str = ATTEMPT_IDS[0]) -> dict[str, Any]:
    return capsule_module.addressed(
        capsule_module.compile_contract(ROOT, attempt_run_id=attempt_run_id)
    )


def _readdress(value: dict[str, Any]) -> dict[str, Any]:
    return capsule_module.addressed(copy.deepcopy(value["contract"]))


def _gateway_packet(capsule: dict[str, Any]) -> dict[str, Any]:
    source_hashes = {
        **{row["path"]: row["sha256"]
           for label, row in capsule["contract"]["artifacts"].items()
           if label not in launch_packet.PROTECTED_BODY_ARTIFACT_LABELS},
        **capsule["contract"]["runtime_wiring"],
    }
    source_members = {
        relative: {
            "sha256": expected,
            "bytes": (ROOT / relative).stat().st_size,
            "mode": f"{stat.S_IMODE((ROOT / relative).stat().st_mode):04o}",
        }
        for relative, expected in source_hashes.items()
    }
    inputs = {
        "schema": launch_packet.INPUT_SCHEMA,
        "worker_image": capsule["contract"]["production_runtime"]["image"],
        "worker_disk_gb": 80,
        "source_bundle": {"bucket": "offline-fixture", "key": "future/source.tar",
                          "version_id": "offline-version", "sha256": "a" * 64,
                          "bytes": 4096, "members": source_members},
        "runtime_image": {"image": capsule["contract"]["production_runtime"]["image"],
                          "receipt_sha256": "b" * 64},
        "control_mounts": {"aws_config_sha256": "c" * 64,
                           "aws_credentials_sha256": "d" * 64,
                           "control_vast_key_sha256": "e" * 64,
                           "ssh_private_key_sha256": "f" * 64},
        "control_executables": {"aws_sha256": "7" * 64,
                                "vastai_sha256": "8" * 64},
        "worker_runtime_receipts": {"python_sha256": "1" * 64,
                                    "aws_sha256": "2" * 64,
                                    "vast_sha256": "3" * 64},
        "maximum_prestart_seconds": 120,
    }
    return launch_packet.compile_packet(
        capsule, inputs, offer_id=424242,
        attempt_run_id=capsule["contract"]["attempt_identity"]["attempt_run_id"],
    )


def _gateway_arm(capsule: dict[str, Any]) -> dict[str, Any]:
    packet = _gateway_packet(capsule)
    return {"schema": "pg1-native-prelaunch-cleanup-lease/v2",
            "status": "PRELAUNCH_CLEANUP_ARMED",
            "packet_sha256": packet["content_address"]["digest"],
            "capsule_sha256": capsule["content_address"]["digest"],
            "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
            "offer_id": packet["contract"]["offer_id"],
            "launch_label": packet["contract"]["launch_label"],
            "transaction_sha256": "a" * 64, "instance_id": None,
            "science_released": False, "destroy_authority": False,
            "etag": "arm-etag", "version_id": "arm-version"}


def _gateway_authorization(capsule: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    return launch_authority.build_authorization(
        capsule, packet, one_shot_nonce="6" * 64
    )


def _stored_object(
    body: dict[str, Any], metadata: dict[str, str], *,
    etag: str = "fixture-etag", version_id: str = "fixture-version",
    key: str | None = None,
) -> dict[str, Any]:
    raw = launch_state.canonical_bytes(body)
    selected_key = key
    if selected_key is None and metadata.get("object-kind") == "canonical-launch-state":
        selected_key = launch_state.state_key(body["packet_sha256"])
    elif selected_key is None and metadata.get("object-kind") == "gateway-orphan-cleanup":
        selected_key = (
            f"{control_deadman.PREFIX}/gateway-cleanup/{body['packet_sha256']}.json"
        )
    elif (
        selected_key is None
        and metadata.get("object-kind") == "immutable-external-terminal-receipt"
    ):
        selected_key = launch_state.terminal_receipt_key(
            body["packet_sha256"], body["action"]
        )
    return {
        "status": "PRESENT", "body": copy.deepcopy(body),
        "raw_bytes": raw, "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "etag": etag, "version_id": version_id,
        "checksum_sha256": base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii"),
        "metadata": copy.deepcopy(metadata),
        **({"key": selected_key} if selected_key is not None else {}),
    }


def _fixed_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mechanical_rebase.canonical_bytes(value))
    os.chmod(path, 0o400)


def _mechanical_source_layout(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = tmp_path / "source-location-root"
    worker_root = root / ".pg1-fake-services/worker"
    body = {
        "kind": "MECHANICAL_NO_AUTHORITY", "production_authority": False,
        "provider_calls": 0, "root_rebase": str(root.resolve()),
        "fake_services": "TRACKED_DETERMINISTIC_EXECUTABLES",
        "network_scope": "LOOPBACK_OR_NONE",
        "may_mint_production_authority": False,
    }
    root_contract = {
        "schema": "pg1-native-exact-chain-mechanical-contract/v1",
        "status": "OFFLINE_DETERMINISTIC_FAKE_SERVICES_ONLY",
        "content_address": {
            "algorithm": "sha256",
            "digest": hashlib.sha256(
                mechanical_rebase.canonical_bytes(body)
            ).hexdigest(),
        },
        "contract": body,
    }
    _fixed_json(
        worker_root / "run/parallax/pg1-native-mechanical-contract.json",
        root_contract,
    )
    _fixed_json(
        root / "run/parallax/control-plane/exact-chain-mechanical.json",
        root_contract,
    )
    entry = {
        "schema": "pg1-native-entry-dispatch-contract/v2",
        "kind": "MECHANICAL_NO_AUTHORITY_EXACT_WORKER_V2",
        "dispatch_scope": "EXACT_REBASED_BYTES_ONLY",
        "production_authority": False, "may_mint_authorization": False,
        "rebase_root": str(worker_root.resolve()),
        "image_root": str(
            worker_root / "opt/parallax/image/source"
        ),
        "bootstrap_root": str(
            worker_root / "opt/parallax/future-freeze/source"
        ),
    }
    entry_path = worker_root / "run/parallax/pg1-native-entry-contract.json"
    _fixed_json(entry_path, entry)
    return root, worker_root, entry_path, entry


def _lifecycle_race_writer(
    path_text: str,
    context: dict[str, Any],
    target_state: str,
    queue: Any,
) -> None:
    try:
        value = lifecycle_module.transition(
            Path(path_text), context, target_state,
            reason=f"RACE_{target_state}", expected_sequence=1,
        )
        queue.put(("ok", value["state"], value["sequence"]))
    except BaseException as error:
        queue.put(("error", type(error).__name__, str(error)))


def _fixture_receipt(
    capsule: dict[str, Any], staged_root: Path = ROOT
) -> dict[str, Any]:
    return overlay.expected_receipt(
        capsule, staged_root.resolve(), allow_parameterized_fixture=True
    )


def _graph(capsule: dict[str, Any]) -> dict[str, Any]:
    return controller.capsule_preflight_graph(
        capsule,
        ROOT,
        _fixture_receipt(capsule),
        allow_parameterized_fixture=True,
    )


def _write_capsule(capsule: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(capsule, sort_keys=True), encoding="utf-8")


def _stop_context(
    capsule: dict[str, Any], *, instance_id: int = 778899
) -> dict[str, Any]:
    attempt = capsule["contract"]["attempt_identity"]["attempt_run_id"]
    receipts = {
        label: {"path": str(path), "sha256": character * 64, "mode": mode}
        for (label, path, mode), character in zip(
            (
                ("python", capsule_module.PYTHON_EXECUTABLE, "0755"),
                ("aws", capsule_module.AWS_EXECUTABLE, "0755"),
                ("vast", capsule_module.VAST_EXECUTABLE, "0755"),
                ("aws_config", capsule_module.AWS_CONFIG_PATH, "0400"),
                ("aws_credentials", capsule_module.AWS_CREDENTIALS_PATH, "0400"),
                ("vast_api_key", capsule_module.VAST_API_KEY_PATH, "0400"),
                ("runtime_image", capsule_module.RUNTIME_IMAGE_RECEIPT_PATH, "0400"),
            ),
            "1234567",
        )
    }
    contract = {
        "schema": provider.CONTEXT_SCHEMA,
        "issuer": provider.CONTEXT_ISSUER,
        "issuer_account": capsule_module.EXPECTED_BUCKET_OWNER,
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": attempt,
        "instance_id": instance_id,
        "offer_id": 661122,
        "machine_id": 771133,
        "container_id": "future-container-abc",
        "runtime_image": capsule_module.RUNTIME_IMAGE,
        "runtime_receipts": receipts,
        "context_key": provider.stop_context_key(
            capsule["content_address"]["digest"], attempt
        ),
    }
    context = {
        "schema_version": 1,
        "status": provider.CONTEXT_STATUS,
        "content_address": {
            "algorithm": "sha256",
            "digest": capsule_module.content_sha256(contract),
        },
        "contract": contract,
    }
    import base64
    import hashlib

    data = capsule_module.canonical_bytes(context)
    context["issuance"] = {
        "schema": "pg1-native-stop-context-issuance/v1",
        "key": contract["context_key"],
        "version_id": "context-version-1",
        "etag": "context-etag-1",
        "checksum_sha256": base64.b64encode(hashlib.sha256(data).digest()).decode(
            "ascii"
        ),
        "server_side_encryption": "AES256",
    }
    return context


def _mechanical_command(
    capsule: dict[str, Any],
    capsule_path: Path,
    contract_path: Path,
    root: Path,
    failure: str,
) -> list[str]:
    argv = capsule["contract"]["entrypoints"]["mechanical_bootstrap_argv"]
    replacements = {
        str(capsule_module.CAPSULE_RUNTIME_PATH): str(capsule_path),
        str(capsule_module.MECHANICAL_CONTRACT_RUNTIME_PATH): str(contract_path),
        "${REHEARSAL_ROOT}": str(root),
        "${FAILURE}": failure,
    }
    return [sys.executable if item == "python3" else replacements.get(item, item) for item in argv]


def _run_mechanical(
    tmp_path: Path, *, attempt_id: str, failure: str = "none"
) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, Any]]:
    capsule = _capsule(attempt_id)
    capsule_path = tmp_path / f"{attempt_id}.capsule.json"
    rehearsal_root = tmp_path / f"{attempt_id}.{failure}"
    contract_path = tmp_path / f"{attempt_id}.mechanical.json"
    _write_capsule(capsule, capsule_path)
    contract_path.write_text(
        json.dumps(compile_mechanical_contract(capsule), sort_keys=True),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        _mechanical_command(
            capsule, capsule_path, contract_path, rehearsal_root, failure
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, rehearsal_root, capsule


@pytest.mark.parametrize("attempt_run_id", ATTEMPT_IDS)
def test_arbitrary_attempts_traverse_exactly_seven_native_owners(
    attempt_run_id: str,
) -> None:
    capsule = _capsule(attempt_run_id)
    result = _graph(capsule)
    assert result["call_trace"] == list(capsule_module.CONSUMER_ORDER)
    assert capsule["contract"]["source_identity"]["source_run_id"] != attempt_run_id
    assert {row["attempt_run_id"] for row in result["receipts"]} == {attempt_run_id}
    for field in (
        "archive_body_reads",
        "model_body_reads",
        "data_body_reads",
        "confirmation_items_read",
        "torch_imports",
        "transformers_imports",
        "cuda_operations",
        "hydration_operations",
        "science_operations",
        "claims_written",
    ):
        assert result[field] == 0
    assert result["h100_authority"] is False


def test_preflight_graph_opens_no_protected_model_data_or_archive_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("preflight-zero-body-reads")
    original = Path.open

    def guarded(path: Path, *args: Any, **kwargs: Any):
        resolved = path.resolve()
        text = str(resolved)
        if (
            "/data/" in text
            or "/models--" in text
            or resolved.suffix in {".safetensors", ".bin", ".jpg", ".jpeg", ".png"}
        ):
            raise AssertionError(f"preflight attempted protected body read: {resolved}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert _graph(capsule)["call_trace"] == list(capsule_module.CONSUMER_ORDER)


def test_legacy_hash_bound_modules_remain_exact_8bba643_bytes() -> None:
    for relative in LEGACY_RESTORED:
        baseline = subprocess.run(
            ["git", "show", f"8bba643:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert (ROOT / relative).read_bytes() == baseline


def test_declared_normal_argv_is_complete_and_controller_traverses_execution() -> None:
    capsule = _capsule()
    consumers = capsule["contract"]["consumers"]
    for name, module, _ in capsule_module.NATIVE_BINDINGS:
        argv = consumers[name]["execution_argv"]
        assert argv[:3] == [str(capsule_module.PYTHON_EXECUTABLE), "-m", module]
        assert "--capsule" in argv
        assert consumers[name]["execution_callable"].startswith(f"{module}:")
    assert "--execute" in consumers["controller"]["execution_argv"]
    assert "--execute" in consumers["job"]["execution_argv"]
    assert "--execute" in consumers["cell"]["execution_argv"]
    assert "--execute" in consumers["worker"]["execution_argv"]
    assert "--execute" in consumers["critic"]["execution_argv"]
    assert consumers["source_gate"]["execution_argv"] == consumers["source_gate"]["preflight_argv"]
    assert consumers["overlay_stage"]["execution_argv"] == consumers["overlay_stage"]["preflight_argv"]
    assert capsule["contract"]["entrypoints"]["production_argv"] == [
        str(capsule_module.PRODUCTION_ENTRY_PATH),
        "--launch", "--capsule", str(capsule_module.CAPSULE_RUNTIME_PATH),
        "--stop-context", str(capsule_module.STOP_CONTEXT_RUNTIME_PATH),
        "--entry-contract", str(capsule_module.ENTRY_CONTRACT_RUNTIME_PATH),
    ]
    assert capsule["contract"]["entrypoints"]["deadman_argv"] == (
        capsule_module.deadman_argv(capsule["contract"])
    )
    assert capsule["contract"]["entrypoints"]["launch_gateway_argv"] == (
        capsule_module.launch_gateway_argv(capsule["contract"])
    )
    assert capsule["contract"]["entrypoints"]["native_launcher_argv"] == [
        str(capsule_module.PYTHON_EXECUTABLE), "-m",
        "infra.vast.parallax.pg1_native_launcher",
        "--execute", "--capsule", str(capsule_module.CAPSULE_RUNTIME_PATH),
    ]
    assert capsule["contract"]["entrypoints"]["bootstrap_argv"] == (
        capsule_module.bootstrap_argv(capsule["contract"])
    )
    assert bootstrap.guardian_command(capsule)[0] == str(
        capsule_module.PYTHON_EXECUTABLE
    )
    assert deployment.bootstrap_command(capsule) == capsule["contract"]["entrypoints"][
        "bootstrap_argv"
    ]
    assert launcher.dispatch_command(capsule) == capsule["contract"]["entrypoints"][
        "normal_controller_argv"
    ]
    controller_source = inspect.getsource(controller.execute)
    for fixed_call in (
        "ProductionAuthorizationStore",
        "ProductionStopper",
        "capsule_preflight_graph",
        "execute_job",
        "finalize_attempt",
    ):
        assert fixed_call in controller_source


def test_production_functions_offer_no_substitutable_callbacks_or_science_inputs() -> None:
    expected = {
        controller.execute: {"capsule"},
        job.execute: {"capsule", "authorization", "authorization_receipt"},
        cell.run: {"capsule", "authorization", "authorization_receipt"},
        worker.finalize_attempt: {"capsule", "authorization", "authorization_receipt"},
        critic.run: {"capsule", "authorization", "authorization_receipt"},
    }
    for function, names in expected.items():
        assert set(inspect.signature(function).parameters) == names
        source = inspect.getsource(function)
        for forbidden in (
            "Callable", "run_cell", "durable_sink:", "verify_durability:",
            "request_destroy:", "lineage_policy:", "matrix:", "task_root:",
        ):
            assert forbidden not in source


def test_canonical_launch_authority_is_remote_exact_and_not_a_launch_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("canonical-remote-authority-plant-v1")
    packet = _gateway_packet(capsule)
    authorization = _gateway_authorization(capsule, packet)
    data = launch_authority.canonical_bytes(authorization)
    storage = _stored_object(
        authorization, launch_authority.metadata(authorization),
        version_id="canonical-version-1", etag="canonical-etag-1",
    )
    assert launch_gateway.validate_launch_authorization(
        authorization, storage, packet, capsule
    ) == authorization
    assert list(inspect.signature(launch_gateway.launch).parameters) == ["capsule", "packet"]
    consumed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        launch_gateway, "_aws_put",
        lambda **values: (consumed.append(values) or {"version_id": "consume-version"}),
    )
    launch_gateway.consume_launch_authorization(authorization, storage, packet)
    assert consumed[0]["key"] == launch_authority.consumption_key(
        authorization["content_address"]["digest"]
    )
    assert consumed[0]["value"]["authorization_version_id"] == storage["version_id"]
    assert consumed[0]["value"]["authorization_checksum_sha256"] == storage[
        "checksum_sha256"
    ]

    fabricated = copy.deepcopy(authorization)
    fabricated["contract"]["one_shot_nonce"] = "8" * 64
    fabricated["content_address"]["digest"] = launch_authority.digest(
        fabricated["contract"]
    )
    for planted in (
        {**storage, "body": fabricated},
        {**storage, "metadata": {"object-kind": "locally-fabricated"}},
        {**storage, "status": "ABSENT"},
    ):
        with pytest.raises(launch_gateway.LaunchGatewayError):
            launch_gateway.validate_launch_authorization(
                authorization, planted, packet, capsule,
            )

    monkeypatch.setattr(
        launch_gateway, "_read_object", lambda **_kwargs: {"status": "ABSENT"}
    )
    with pytest.raises(launch_gateway.LaunchGatewayError, match="unavailable"):
        launch_gateway.fetch_launch_authorization(capsule, packet)


@pytest.mark.parametrize(
    ("observations", "expected"),
    (
        (("PRESENT",), "SCIENCE_RELEASED"),
        (("ABSENT", "ABSENT", "ABSENT"), "START_COMMIT_PROVEN_ABSENT"),
        (("ABSENT", "UNKNOWN"), "SCIENCE_RELEASE_UNKNOWN"),
        (("CONTRADICTORY",), "SCIENCE_RELEASE_UNKNOWN"),
    ),
)
def test_start_commit_classifier_has_released_absent_and_unknown_outcomes(
    observations: tuple[str, ...], expected: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule(f"start-classifier-{expected.lower()}-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="a" * 64)
    for phase in (
        "CREATE_INTENT_ARMED", "WORKER_ID_BOUND_SCIENCE_DISABLED",
        "INSTANCE_CONTEXT_BOUND", "HANDOFF_BOUND_DEADMAN_ACKED",
        "START_COMMIT_INTENT",
    ):
        state = launch_state.transition(state, phase)
    context = _stop_context(capsule, instance_id=778899)
    expected_body = launch_gateway.start_commit_body(
        capsule, packet, state, context, {"schema": "layout"}, {"status": "ack"}
    )
    planted = iter(observations)

    def read(**_kwargs: Any) -> dict[str, Any]:
        status = next(planted)
        return {"status": status, "body": expected_body} if status == "PRESENT" else {
            "status": status
        }

    monkeypatch.setattr(launch_gateway, "_read_object", read)
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    result = launch_gateway.classify_start_commit(packet, expected_body)
    assert result["classification"] == expected

    if expected == "SCIENCE_RELEASED":
        released = launch_state.transition(
            state, "SCIENCE_RELEASED", science_state=launch_state.SCIENCE_RELEASED,
            start_commit_sha256=expected_body["content_address"]["digest"],
        )
        assert released["science_state"] == launch_state.SCIENCE_RELEASED
    elif expected == "START_COMMIT_PROVEN_ABSENT":
        absent = launch_state.transition(state, "START_COMMIT_PROVEN_ABSENT")
        destroy_body = {"schema": "planted-exact-body/v1", "plant": "offline"}
        request_reference = {
            "sha256": launch_state.digest(destroy_body),
            "storage": "CANONICAL_STATE_RAW_BODY",
        }
        assert launch_state.transition(
            absent, "CLEANUP_DESTROYING", destroy_request=request_reference,
            destroy_request_body=destroy_body,
        )["phase"] == (
            "CLEANUP_DESTROYING"
        )
    else:
        unknown = launch_state.transition(
            state, "SCIENCE_RELEASE_UNKNOWN",
            science_state=launch_state.SCIENCE_RELEASE_UNKNOWN,
        )
        with pytest.raises(launch_state.LaunchStateError, match="forbidden"):
            launch_state.transition(unknown, "CLEANUP_DESTROYING")


def test_deadman_restart_accepts_only_canonical_descendant_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("deadman-canonical-restart-v1")
    packet = _gateway_packet(capsule)
    cached = launch_state.initial(packet, authorization_sha256="b" * 64)
    canonical = launch_state.transition(cached, "CREATE_INTENT_ARMED")
    transaction = tmp_path / "transaction.json"
    transaction.write_bytes(launch_state.canonical_bytes(cached)); os.chmod(transaction, 0o400)
    observed = _stored_object(
        canonical, control_deadman._state_metadata(canonical),
        etag="state-etag-2", version_id="state-version-2",
    )
    monkeypatch.setattr(control_deadman, "_read_object", lambda _key: observed)
    restarted = control_deadman.arm(capsule, transaction)
    assert restarted["status"] == "RESTART_RESUMED_CANONICAL_STATE"
    assert json.loads(transaction.read_text())["self_sha256"] == canonical["self_sha256"]

    divergent = copy.deepcopy(canonical)
    divergent["ancestor_state_sha256"] = ["d" * 64]
    divergent["previous_state_sha256"] = "d" * 64
    divergent["self_sha256"] = launch_state.digest({
        key: value for key, value in divergent.items() if key != "self_sha256"
    })
    observed.clear()
    observed.update(_stored_object(
        divergent, control_deadman._state_metadata(divergent),
        etag="divergent-etag", version_id="divergent-version",
    ))
    with pytest.raises(control_deadman.ControlDeadmanError, match="ancestry"):
        control_deadman._load_state(capsule, transaction)


def test_deadman_restart_rejects_remote_ancestor_of_locally_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("deadman-reject-canonical-regression-v1")
    packet = _gateway_packet(capsule)
    remote = launch_state.initial(packet, authorization_sha256="a" * 64)
    cached = launch_state.transition(remote, "CREATE_INTENT_ARMED")
    transaction = tmp_path / "transaction.json"
    transaction.write_bytes(launch_state.canonical_bytes(cached)); os.chmod(transaction, 0o400)
    monkeypatch.setattr(
        control_deadman, "_read_object",
        lambda _key: _stored_object(
            remote, control_deadman._state_metadata(remote), etag="old-etag"
        ),
    )
    with pytest.raises(control_deadman.ControlDeadmanError, match="ancestry"):
        control_deadman._load_state(capsule, transaction)


def test_deadman_bound_ack_replay_is_stable_after_post_bound_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("deadman-post-bound-ack-restart-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(
        packet, authorization_sha256="1" * 64,
        authorization_receipt_sha256="4" * 64,
    )
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {
            "machine_id": 771133, "container_id": "future-container-abc",
            "stop_context_sha256": "2" * 64,
            "mount_inventory_sha256": "5" * 64,
        }),
    ):
        state = launch_state.transition(state, phase, **fields)
    planted: dict[str, Any] = {}

    def immutable_put(*, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if planted:
            assert payload == planted
            replayed = True
        else:
            planted.update(copy.deepcopy(payload)); replayed = False
        return {
            "etag": "bound-etag", "version_id": "bound-version",
            "checksum_sha256": base64.b64encode(hashlib.sha256(
                control_deadman.canonical_bytes(payload)
            ).digest()).decode("ascii"),
            "replayed_exact": replayed,
        }

    monkeypatch.setattr(control_deadman, "BOUND_ACK", tmp_path / "bound-ack.json")
    monkeypatch.setattr(control_deadman, "_put", immutable_put)
    first = control_deadman._bind_ack(state)
    assert launch_gateway.validate_bound_ack(first, state, packet) == first
    for invalid_ack in (
        {**first, "unexpected": "field"},
        {**first, "ack_nonce": "0" * 64},
        {**first, "checksum_sha256": "wrong-checksum"},
    ):
        with pytest.raises(launch_gateway.LaunchGatewayError, match="exact envelope"):
            launch_gateway.validate_bound_ack(invalid_ack, state, packet)
    restarted = launch_state.transition(
        state, "HANDOFF_BOUND_DEADMAN_ACKED", handoff_layout_sha256="3" * 64,
    )
    restarted = launch_state.transition(restarted, "START_COMMIT_INTENT")
    second = control_deadman._bind_ack(restarted)
    assert first["schema"] == "pg1-native-instance-bound-deadman-ack/v3"
    assert second == first
    assert {key: second[key] for key in planted} == planted
    assert "canonical_state_sequence" not in planted
    assert "transaction_sha256" not in planted


def test_production_deployment_accepts_exact_v3_bound_ack_and_rejects_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("deployment-v3-bound-ack-v1")
    packet = _gateway_packet(capsule)
    context = _stop_context(capsule, instance_id=778899)
    context["contract"]["offer_id"] = packet["contract"]["offer_id"]
    context["content_address"]["digest"] = capsule_module.content_sha256(
        context["contract"]
    )
    arm_path = tmp_path / "pg1-external-deadman-arm.json"
    packet_path = tmp_path / "pg1-future-launch-packet.json"
    binding_path = tmp_path / "pg1-launch-authority-binding.json"
    packet_path.write_bytes(launch_state.canonical_bytes(packet))
    state = launch_state.initial(
        packet, authorization_sha256="a" * 64,
        authorization_receipt_sha256="b" * 64,
    )
    state = launch_state.transition(state, "CREATE_INTENT_ARMED")
    state = launch_state.transition(
        state, "WORKER_ID_BOUND_SCIENCE_DISABLED", instance_id=778899,
    )
    state = launch_state.transition(
        state, "INSTANCE_CONTEXT_BOUND",
        machine_id=context["contract"]["machine_id"],
        container_id=context["contract"]["container_id"],
        stop_context_sha256=context["content_address"]["digest"],
        mount_inventory_sha256="c" * 64,
    )
    monkeypatch.setattr(control_deadman, "BOUND_ACK", arm_path)
    monkeypatch.setattr(
        control_deadman, "_put",
        lambda **values: {
            "etag": "ack-etag", "version_id": "ack-version",
            "checksum_sha256": base64.b64encode(hashlib.sha256(
                control_deadman.canonical_bytes(values["payload"])
            ).digest()).decode("ascii"),
        },
    )
    v3 = control_deadman._bind_ack(state)
    binding = {
        "schema": "pg1-native-worker-authority-binding/v1",
        "packet_sha256": v3["packet_sha256"],
        "capsule_sha256": v3["capsule_sha256"],
        "attempt_run_id": v3["attempt_run_id"],
        "authorization_body_sha256": v3["authorization_body_sha256"],
        "authorization_receipt_sha256": v3["authorization_receipt_sha256"],
        "mount_inventory_sha256": v3["mount_inventory_sha256"],
    }
    binding_path.write_bytes(launch_state.canonical_bytes(binding))
    monkeypatch.setattr(deployment, "EXTERNAL_ARM_RECEIPT_PATH", arm_path)
    monkeypatch.setattr(deployment, "PACKET_RUNTIME_PATH", packet_path)
    monkeypatch.setattr(deployment, "AUTHORITY_BINDING_RUNTIME_PATH", binding_path)
    assert deployment._verify_external_deadman_arm(capsule, context) == v3
    for name, planted in (
        ("extra", {**v3, "unexpected": "field"}),
        ("nonce", {**v3, "ack_nonce": "0" * 64}),
        ("checksum", {**v3, "checksum_sha256": "wrong-checksum"}),
    ):
        os.chmod(arm_path, 0o600)
        arm_path.write_text(json.dumps(planted, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(arm_path, 0o400)
        with pytest.raises(deployment.DeploymentError, match="identity changed"):
            deployment._verify_external_deadman_arm(capsule, context)
    os.chmod(arm_path, 0o600)
    arm_path.write_text(json.dumps(v3, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(arm_path, 0o400)
    v1 = {**v3, "schema": "pg1-native-instance-bound-deadman-ack/v1",
          "transaction_sha256": "e" * 64}
    os.chmod(arm_path, 0o600)
    arm_path.write_text(json.dumps(v1, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(arm_path, 0o400)
    with pytest.raises(deployment.DeploymentError, match="identity changed"):
        deployment._verify_external_deadman_arm(capsule, context)


def test_production_clis_have_no_test_only_root_or_probe_options() -> None:
    paths = (
        capsule_module.CONTROL_INSTALLER_PATH, capsule_module.LAUNCH_INVOKER_PATH,
        capsule_module.LAUNCH_AUTHORITY_PATH, capsule_module.LAUNCH_GATEWAY_PATH,
        capsule_module.CONTROL_DEADMAN_PATH, capsule_module.ONSTART_MATERIALIZER_PATH,
        capsule_module.IDLE_BOOTSTRAP_PATH,
        Path("infra/vast/parallax/pg1_native_worker.py"),
    )
    options: set[str] = set()
    for path in paths:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        options.update(
            argument.value
            for call in ast.walk(tree)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            for argument in call.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        )
    assert not {option for option in options if "offline" in option}
    assert {option for option in options if "mechanical" in option} == {
        "--mechanical-no-model-proof"
    }
    worker_source = (
        ROOT / "infra/vast/parallax/pg1_native_worker.py"
    ).read_text(encoding="utf-8")
    assert "mechanical no-model proof is unavailable in production" in worker_source
    assert "--offline-root-contract" not in options
    assert "--mechanical-exact-probe" not in options


def test_production_invocation_argv_is_fully_resolved_and_placeholder_free() -> None:
    attempt = "resolved-production-invocation-v1"
    commands = control_installer.production_invocation_argv(
        one_shot_nonce="d" * 64, offer_id=424242, attempt_run_id=attempt,
    )
    assert len(commands) == 3
    assert all(Path(command[0]).is_absolute() for command in commands)
    assert all(
        "<" not in item and ">" not in item and "${" not in item
        for command in commands for item in command
    )
    assert commands[0][commands[0].index("--offer-id") + 1] == "424242"
    assert commands[0][commands[0].index("--attempt-run-id") + 1] == attempt
    assert commands[1][commands[1].index("--one-shot-nonce") + 1] == "d" * 64
    packet_paths = {
        command[command.index("--packet") + 1]
        for command in commands[1:]
    }
    assert packet_paths == {"/run/parallax/control-plane/pg1-future-launch-packet.json"}


def test_gateway_cas_ambiguity_requires_the_complete_attempted_canonical_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("gateway-descendant-cas-recovery-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="b" * 64)
    for phase in (
        "CREATE_INTENT_ARMED", "WORKER_ID_BOUND_SCIENCE_DISABLED",
        "INSTANCE_CONTEXT_BOUND", "HANDOFF_BOUND_DEADMAN_ACKED", "START_COMMIT_INTENT",
    ):
        state = launch_state.transition(state, phase, instance_id=778899)
    attempted = launch_state.transition(
        state, "SCIENCE_RELEASED", science_state=launch_state.SCIENCE_RELEASED,
        start_commit_sha256="c" * 64,
    )
    winner = launch_state.transition(attempted, "CLEANUP_STOPPING")
    monkeypatch.setattr(launch_gateway, "TRANSACTION", tmp_path / "transaction.json")
    monkeypatch.setattr(launch_gateway, "STATE_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(
        launch_gateway, "_aws_put",
        lambda **_kwargs: (_ for _ in ()).throw(
            launch_gateway.LaunchGatewayError("planted response loss")
        ),
    )
    monkeypatch.setattr(
        launch_gateway, "_read_object",
        lambda **_kwargs: _stored_object(
            winner, launch_gateway._state_metadata(winner),
            etag="winner-etag", version_id="winner-version",
        ),
    )
    with pytest.raises(launch_gateway.LaunchGatewayError, match="CAS inconclusive"):
        launch_gateway._publish_state(packet, attempted, prior_etag="intent-etag")
    monkeypatch.setattr(
        launch_gateway, "_read_object",
        lambda **_kwargs: _stored_object(
            attempted, launch_gateway._state_metadata(attempted),
            etag="winner-etag", version_id="winner-version",
        ),
    )
    selected, receipt = launch_gateway._publish_state(
        packet, attempted, prior_etag="intent-etag", prior_state=state,
    )
    assert selected == attempted
    assert receipt["recovered_after_ambiguous_put"] is True
    assert json.loads(launch_gateway.TRANSACTION.read_text()) == attempted


def test_deadman_terminal_receipt_precedes_state_reference_and_restart_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("deadman-terminal-convergence-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="e" * 64)
    transitions = (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {"machine_id": 771133,
                                    "container_id": "future-container-abc",
                                    "stop_context_sha256": "f" * 64}),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "1" * 64}),
        ("START_COMMIT_INTENT", {}),
        ("SCIENCE_RELEASED", {"science_state": launch_state.SCIENCE_RELEASED,
                              "start_commit_sha256": "2" * 64}),
        ("CLEANUP_STOPPING", {}),
    )
    for phase, fields in transitions:
        science = fields.pop("science_state", None)
        state = launch_state.transition(state, phase, science_state=science, **fields)
    storage = {"etag": "cleanup-etag", "version_id": "cleanup-version"}
    memory: dict[str, dict[str, Any]] = {}
    writes: list[str] = []
    monkeypatch.setattr(control_deadman, "TRANSACTION", tmp_path / "transaction.json")
    monkeypatch.setattr(control_deadman, "STATE_RECEIPT", tmp_path / "state-receipt.json")

    def read(key: str) -> dict[str, Any]:
        return memory.get(key, {"status": "ABSENT"})

    def put(*, key: str, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        writes.append(key)
        memory[key] = _stored_object(
            payload, _kwargs["metadata"], etag=f"etag-{len(writes)}",
            version_id=f"version-{len(writes)}", key=key,
        )
        receipt = {
            field: memory[key][field]
            for field in ("etag", "version_id", "checksum_sha256")
        }
        return receipt

    monkeypatch.setattr(control_deadman, "_read_object", read)
    monkeypatch.setattr(control_deadman, "_put", put)
    payload, terminal, terminal_storage = control_deadman._terminal_receipt(
        state, storage, action="stop", attempts=2, errors=0,
        observations=3, destroy_gate=None,
    )
    assert payload["status"] == "TERMINAL_STOPPED_PRESERVED"
    assert terminal["phase"] == "TERMINAL_STOPPED_PRESERVED"
    assert writes == [
        launch_state.terminal_receipt_key(packet["content_address"]["digest"], "stop"),
        launch_state.state_key(packet["content_address"]["digest"]),
    ]
    reread_state, _reread_storage = control_deadman._validate_terminal_state_reread(
        terminal_storage, terminal, state,
    )
    assert reread_state == terminal
    contradictory_state = copy.deepcopy(terminal_storage)
    contradictory_state["body"] = launch_state.transition(
        state, "TERMINAL_STOPPED_PRESERVED",
        terminal_receipt=terminal["terminal_receipt"],
        planted_omitted_identity="otherwise-valid-but-not-the-attempted-body",
    )
    with pytest.raises(control_deadman.ControlDeadmanError, match="reread invalid"):
        control_deadman._validate_terminal_state_reread(
            contradictory_state, terminal, state,
        )
    memory[terminal["terminal_receipt"]["key"]].update({
        "etag": terminal["terminal_receipt"]["etag"],
        "version_id": terminal["terminal_receipt"]["version_id"],
    })
    before = list(writes)
    converged = control_deadman._converge_terminal_orphan(terminal, terminal_storage)
    assert converged is not None and converged[1] == terminal
    assert writes == before


@pytest.mark.parametrize(
    "corruption",
    ("raw_bytes", "raw_sha256", "checksum_sha256", "metadata", "key"),
)
def test_deadman_ambiguous_put_rejects_nonexact_storage_replay(
    monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    key = "parallax/test/exact-replay.json"
    payload = {"schema": "test-exact-replay/v1", "value": 1}
    metadata = {"object-kind": "test-exact-replay"}
    observed = _stored_object(payload, metadata, key=key)
    if corruption == "raw_bytes":
        observed[corruption] = observed[corruption] + b" "
    elif corruption == "raw_sha256":
        observed[corruption] = "0" * 64
    elif corruption == "checksum_sha256":
        observed[corruption] = base64.b64encode(b"0" * 32).decode("ascii")
    elif corruption == "metadata":
        observed[corruption] = {**metadata, "planted": "extra"}
    else:
        observed[corruption] = f"{key}.changed"
    monkeypatch.setattr(
        control_deadman, "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "planted ambiguous put"
        ),
    )
    monkeypatch.setattr(control_deadman, "_read_object", lambda _key: observed)
    with pytest.raises(control_deadman.ControlDeadmanError, match="not exact replay"):
        control_deadman._put(key=key, payload=payload, metadata=metadata)


def test_deadman_ambiguous_put_accepts_only_exact_raw_storage_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "parallax/test/exact-replay.json"
    payload = {"schema": "test-exact-replay/v1", "value": 1}
    metadata = {"object-kind": "test-exact-replay"}
    observed = _stored_object(payload, metadata, key=key)
    monkeypatch.setattr(
        control_deadman, "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "planted ambiguous put"
        ),
    )
    monkeypatch.setattr(control_deadman, "_read_object", lambda _key: observed)
    result = control_deadman._put(key=key, payload=payload, metadata=metadata)
    assert result == {
        "etag": observed["etag"], "version_id": observed["version_id"],
        "checksum_sha256": observed["checksum_sha256"], "replayed_exact": True,
    }


@pytest.mark.parametrize(
    "corruption", ("extra_field", "raw_bytes", "checksum_sha256", "metadata")
)
def test_deadman_terminal_receipt_rejects_open_or_nonexact_storage(
    corruption: str,
) -> None:
    capsule = _capsule(f"terminal-exact-storage-{corruption}")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="a" * 64)
    state = launch_state.transition(state, "CREATE_INTENT_ARMED")
    state = launch_state.transition(
        state, "WORKER_ID_BOUND_SCIENCE_DISABLED", instance_id=778899,
    )
    state = launch_state.transition(state, "CLEANUP_STOPPING")
    payload = {
        "schema": "pg1-native-external-control-terminal/v3",
        "status": "TERMINAL_STOPPED_PRESERVED", "action": "stop",
        "packet_sha256": state["packet_sha256"],
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"], "instance_id": 778899,
        "parent_state_sha256": state["self_sha256"],
        "parent_state_sequence": state["sequence"], "terminal_state": "stopped",
        "consecutive_observations": 3, "attempts": 1, "errors": 0,
        "destroy_gate": None, "science_state": state["science_state"],
        "external_control_plane": True, "receipt_durable": True,
    }
    key = launch_state.terminal_receipt_key(state["packet_sha256"], "stop")
    observed = _stored_object(
        payload, control_deadman._terminal_metadata(state), key=key,
    )
    if corruption == "extra_field":
        observed["body"]["planted_extra"] = True
        raw = launch_state.canonical_bytes(observed["body"])
        observed.update({
            "raw_bytes": raw,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "checksum_sha256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
        })
    elif corruption == "raw_bytes":
        observed[corruption] += b" "
    elif corruption == "checksum_sha256":
        observed[corruption] = "changed"
    else:
        observed[corruption] = {
            **observed[corruption], "planted-extra": "changed",
        }
    with pytest.raises(control_deadman.ControlDeadmanError):
        control_deadman._validate_terminal_observation(
            observed, state, action="stop"
        )


def test_mechanical_dispatch_is_derived_from_exact_copied_source_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, worker_root, entry_path, entry = _mechanical_source_layout(tmp_path)
    image_source = (
        worker_root
        / "opt/parallax/image/source/infra/vast/parallax/pg1_native_deployment.py"
    )
    bootstrap_source = (
        worker_root
        / "opt/parallax/future-freeze/source/infra/vast/parallax/pg1_native_bootstrap.py"
    )
    worker_source = bootstrap_source.with_name("pg1_native_worker.py")
    rebase_source = bootstrap_source.with_name("pg1_native_mechanical_rebase.py")
    for target, source in (
        (image_source, ROOT / "infra/vast/parallax/pg1_native_deployment.py"),
        (bootstrap_source, ROOT / "infra/vast/parallax/pg1_native_bootstrap.py"),
        (worker_source, ROOT / "infra/vast/parallax/pg1_native_worker.py"),
        (rebase_source, ROOT / "infra/vast/parallax/pg1_native_mechanical_rebase.py"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    monkeypatch.setattr(deployment, "__file__", str(image_source))
    assert deployment._mechanical_entry_from_source() == (
        entry, entry_path.resolve()
    )
    monkeypatch.setattr(deployment, "_mechanical_dispatch", lambda *_args: 17)
    assert deployment.main(["--entry-contract", str(entry_path)]) == 17
    monkeypatch.setattr(bootstrap, "__file__", str(bootstrap_source))
    assert bootstrap._mechanical_entry_from_source() == (
        entry, entry_path.resolve()
    )
    monkeypatch.setattr(bootstrap, "_mechanical_dispatch", lambda *_args: 19)
    assert bootstrap.main(["--entry-contract", str(entry_path)]) == 19
    monkeypatch.setattr(worker, "__file__", str(worker_source))
    assert worker._mechanical_entry_from_source() == entry

    control_source = (
        root
        / "opt/parallax/control-stage/source/infra/vast/parallax/pg1_native_control_installer.py"
    )
    control_source.parent.mkdir(parents=True, exist_ok=True)
    control_source.write_bytes(
        (ROOT / "infra/vast/parallax/pg1_native_control_installer.py").read_bytes()
    )
    monkeypatch.setattr(control_installer, "__file__", str(control_source))
    for name in (
        "STAGE_SOURCE_ROOT", "CONTROL_SOURCE_ROOT", "STAGE_BIN_ROOT",
        "CONTROL_BIN_ROOT", "CAPSULE", "INPUTS", "AWS_CONFIG",
        "AWS_CREDENTIALS", "VAST_KEY", "SSH_KEY", "RECEIPT",
    ):
        monkeypatch.setattr(control_installer, name, getattr(control_installer, name))
    installer_contract = control_installer.configure_from_source_location()
    assert installer_contract is not None
    assert installer_contract["root"] == root.resolve()

    invoker_source = control_source.with_name("pg1_native_launch_invoker.py")
    invoker_source.write_bytes(
        (ROOT / "infra/vast/parallax/pg1_native_launch_invoker.py").read_bytes()
    )
    sentinel = object()
    monkeypatch.setattr(launch_invoker, "__file__", str(invoker_source))
    monkeypatch.setattr(
        launch_invoker.importlib, "import_module", lambda _name: sentinel,
    )
    assert launch_invoker._installer() is sentinel
    invoker_tree = ast.parse(
        (ROOT / "infra/vast/parallax/pg1_native_launch_invoker.py").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "configure_from_source_location"
        for node in ast.walk(invoker_tree)
    )


@pytest.mark.parametrize("variant", ("production", "mismatched_root"))
def test_copied_deployment_rejects_production_or_mismatched_entry_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    _root, worker_root, entry_path, entry = _mechanical_source_layout(tmp_path)
    source = (
        worker_root
        / "opt/parallax/image/source/infra/vast/parallax/pg1_native_deployment.py"
    )
    changed = copy.deepcopy(entry)
    if variant == "production":
        changed.update({
            "schema": "pg1-native-entry-dispatch-contract/v1",
            "kind": "PRODUCTION_AUTHORITY_BOUND",
        })
    else:
        changed["rebase_root"] = str(tmp_path / "planted-mismatch")
    os.chmod(entry_path, 0o600)
    _fixed_json(entry_path, changed)
    monkeypatch.setattr(deployment, "__file__", str(source))
    with pytest.raises(deployment.DeploymentError, match="copied deployment"):
        deployment._mechanical_entry_from_source()


@pytest.mark.parametrize("module", (deployment, bootstrap))
def test_canonical_entry_rejects_mechanical_contract_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any,
) -> None:
    _root, _worker_root, entry_path, _entry = _mechanical_source_layout(tmp_path)
    monkeypatch.setattr(module, "_mechanical_entry_from_source", lambda: None)
    monkeypatch.setattr(module, "ENTRY_CONTRACT_RUNTIME_PATH", entry_path)
    with pytest.raises(
        (deployment.DeploymentError, bootstrap.BootstrapError),
        match="mechanical contract cannot select canonical",
    ):
        module.main(["--entry-contract", str(entry_path)])


def test_stale_pre_science_destroy_request_can_only_stop_released_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("deadman-stale-pre-science-request-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="4" * 64)
    for phase in (
        "CREATE_INTENT_ARMED", "WORKER_ID_BOUND_SCIENCE_DISABLED",
        "INSTANCE_CONTEXT_BOUND", "HANDOFF_BOUND_DEADMAN_ACKED", "START_COMMIT_INTENT",
    ):
        state = launch_state.transition(state, phase, instance_id=778899)
    state = launch_state.transition(
        state, "SCIENCE_RELEASED", science_state=launch_state.SCIENCE_RELEASED,
        start_commit_sha256="5" * 64,
    )
    stale = {
        "schema": "pg1-native-external-destroy-request/v1",
        "status": "DURABLE_EXTERNAL_DESTROY_REQUEST",
        "capsule_sha256": state["capsule_sha256"],
        "attempt_run_id": state["attempt_run_id"], "instance_id": 778899,
        "destroy_gate": "PRE_SCIENCE_ZERO_BODY_DURABILITY",
        "teardown_authorized": True, "science_released": False,
        "evidence_sha256": "6" * 64, "launch_state_sequence": state["sequence"],
        "launch_state_sha256": state["self_sha256"],
        "start_commit_classification": "START_COMMIT_PROVEN_ABSENT",
    }
    monkeypatch.setattr(
        control_deadman, "_read_object",
        lambda _key: {"status": "PRESENT", "body": stale},
    )
    assert control_deadman._action_required(
        capsule, state, {"status": "PRESENT", "start_commit_sha256": "5" * 64}
    ) == ("stop", None)


def test_gateway_pre_science_request_advances_deadman_to_destroy_only_when_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("gateway-deadman-pre-science-exact-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="6" * 64)
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {
            "machine_id": 771133, "container_id": "future-container-abc",
            "stop_context_sha256": "7" * 64,
        }),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "8" * 64}),
        ("START_COMMIT_INTENT", {}),
        ("START_COMMIT_PROVEN_ABSENT", {}),
    ):
        state = launch_state.transition(state, phase, **fields)
    objects: dict[str, dict[str, Any]] = {}

    def put(*, key: str, value: dict[str, Any], metadata: dict[str, str],
            **_kwargs: Any) -> dict[str, Any]:
        objects[key] = _stored_object(
            value, metadata, etag=f"etag-{len(objects) + 1}",
            version_id=f"version-{len(objects) + 1}", key=key,
        )
        receipt = {
            field: objects[key][field]
            for field in ("version_id", "etag", "checksum_sha256")
        }
        return receipt

    monkeypatch.setattr(
        launch_gateway, "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "{}" if "stop" in command else json.dumps(
                [{"id": 778899, "actual_status": "running"}]
            ), "",
        ),
    )
    monkeypatch.setattr(launch_gateway, "_aws_put", put)
    monkeypatch.setattr(
        launch_gateway, "_read_object",
        lambda *, key, **_kwargs: copy.deepcopy(objects.get(key, {"status": "ABSENT"})),
    )
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    cleanup = launch_gateway._cleanup(
        778899, packet, failure="PLANTED_STOP_UNCONFIRMED",
        science_state=launch_state.SCIENCE_DISABLED, canonical_state=state,
        start_commit_classification="START_COMMIT_PROVEN_ABSENT",
    )
    destroy_body = cleanup["destroy_request_body"]
    reference = {
        "sha256": launch_state.digest(destroy_body),
        "storage": "CANONICAL_STATE_RAW_BODY",
    }
    destroying = launch_state.transition(
        state, "CLEANUP_DESTROYING", destroy_request=reference,
        destroy_request_body=destroy_body,
        destroy_gate="PRE_SCIENCE_ZERO_BODY_DURABILITY",
    )
    monkeypatch.setattr(
        control_deadman, "_read_object",
        lambda key: copy.deepcopy(objects.get(key, {"status": "ABSENT"})),
    )
    assert control_deadman._action_required(
        capsule, state, {"status": "ABSENT"},
    ) == (None, None)
    assert control_deadman._action_required(
        capsule, destroying, {"status": "ABSENT"},
    ) == ("destroy", "PRE_SCIENCE_ZERO_BODY_DURABILITY")
    contradictory = copy.deepcopy(destroying)
    contradictory["destroy_request_body"]["pre_science_zero_body_evidence"][
        "science_output_files"
    ] = 1
    assert control_deadman._action_required(
        capsule, contradictory, {"status": "ABSENT"},
    ) == ("stop", None)


def test_pre_science_cleanup_crash_is_adopted_only_by_running_deadman(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("pre-science-request-first-crash-convergence-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="9" * 64)
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {
            "machine_id": 771133, "container_id": "future-container-abc",
            "stop_context_sha256": "a" * 64,
        }),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "b" * 64}),
        ("START_COMMIT_INTENT", {}),
        ("START_COMMIT_PROVEN_ABSENT", {}),
    ):
        state = launch_state.transition(state, phase, **fields)
    objects: dict[str, dict[str, Any]] = {}
    writes: list[str] = []
    crash_after: int | None = 1
    state_key = launch_state.state_key(packet["content_address"]["digest"])
    objects[state_key] = _stored_object(
        state, launch_gateway._state_metadata(state), etag="state-etag",
        version_id="state-version", key=state_key,
    )

    class PlantedCrash(RuntimeError):
        pass

    def put(*, key: str, value: dict[str, Any], metadata: dict[str, str],
            **_kwargs: Any) -> dict[str, Any]:
        prior_etag = _kwargs.get("prior_etag")
        if key in objects and prior_etag is None:
            raise launch_gateway.LaunchGatewayError("conditional create conflict")
        if key in objects and objects[key].get("etag") != prior_etag:
            raise launch_gateway.LaunchGatewayError("conditional update conflict")
        writes.append(key)
        objects[key] = _stored_object(
            value, metadata, etag=f"etag-{len(writes)}",
            version_id=f"version-{len(writes)}", key=key,
        )
        receipt = {
            field: objects[key][field]
            for field in ("version_id", "etag", "checksum_sha256")
        }
        if crash_after == len(writes):
            raise PlantedCrash(f"planted crash after immutable write {len(writes)}")
        return receipt

    def read(*, key: str, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(objects.get(key, {"status": "ABSENT"}))

    monkeypatch.setattr(
        launch_gateway, "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "{}" if "stop" in command else json.dumps(
                [{"id": 778899, "actual_status": "running"}]
            ), "",
        ),
    )
    monkeypatch.setattr(launch_gateway, "_aws_put", put)
    monkeypatch.setattr(launch_gateway, "_read_object", read)
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(launch_gateway, "_cache_state", lambda *_args: None)
    monkeypatch.setattr(
        launch_gateway, "_start_recovery_deadman", lambda *_args: None,
    )
    monkeypatch.setattr(
        control_deadman, "_read_object",
        lambda key: copy.deepcopy(objects.get(key, {"status": "ABSENT"})),
    )
    monkeypatch.setattr(
        control_deadman, "_put",
        lambda *, key, payload, metadata, prior_etag=None: put(
            key=key, value=payload, metadata=metadata, prior_etag=prior_etag,
        ),
    )
    monkeypatch.setattr(control_deadman, "TRANSACTION", tmp_path / "transaction.json")
    monkeypatch.setattr(control_deadman, "STATE_RECEIPT", tmp_path / "state-receipt.json")
    monkeypatch.setattr(
        control_deadman.time, "time", lambda: float(state["wall_time_unix"]) + 31,
    )
    with pytest.raises(PlantedCrash, match="write 1"):
        launch_gateway._cleanup(
            778899, packet, failure="PLANT_AFTER_CLEANUP_RECEIPT",
            science_state=launch_state.SCIENCE_DISABLED, canonical_state=state,
            start_commit_classification="START_COMMIT_PROVEN_ABSENT",
        )
    assert len(writes) == 1

    crash_after = None
    destroying, destroying_storage = control_deadman._recover_cleanup_intent(
        state, objects[state_key],
    )
    assert len(writes) == 2
    assert destroying["phase"] == "CLEANUP_DESTROYING"
    assert destroying["destroy_request"]["storage"] == "CANONICAL_STATE_RAW_BODY"
    assert objects[state_key]["body"] == destroying
    assert destroying_storage["key"] == state_key
    assert control_deadman._action_required(
        capsule, destroying, {"status": "ABSENT"}
    ) == ("destroy", "PRE_SCIENCE_ZERO_BODY_DURABILITY")


@pytest.mark.parametrize(
    "corruption", ("extra_field", "raw_bytes", "checksum_sha256", "metadata")
)
def test_deadman_cleanup_adoption_rejects_open_or_nonexact_receipt(
    monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    capsule = _capsule(f"cleanup-exact-storage-{corruption}")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="a" * 64)
    state = launch_state.transition(state, "CREATE_INTENT_ARMED")
    state = launch_state.transition(
        state, "WORKER_ID_BOUND_SCIENCE_DISABLED", instance_id=778899,
    )
    receipt = {
        "schema": "pg1-native-gateway-orphan-cleanup/v2",
        "status": "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED",
        "packet_sha256": state["packet_sha256"], "instance_id": 778899,
        "failure": "PLANTED_EXACT_RECEIPT", "stop_attempts": 6, "errors": 0,
        "consecutive_terminal_observations": 0, "science_released": False,
        "science_state": launch_state.SCIENCE_DISABLED,
        "destroy_gate": "PRE_SCIENCE_ZERO_BODY_DURABILITY",
        "start_commit_classification": "START_COMMIT_PROVEN_ABSENT",
    }
    key = f"{control_deadman.PREFIX}/gateway-cleanup/{state['packet_sha256']}.json"
    observed = _stored_object(
        receipt, launch_gateway._cleanup_metadata(receipt), key=key,
    )
    if corruption == "extra_field":
        observed["body"]["planted_extra"] = True
        raw = launch_state.canonical_bytes(observed["body"])
        observed.update({
            "raw_bytes": raw, "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "checksum_sha256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
        })
    elif corruption == "raw_bytes":
        observed[corruption] += b" "
    elif corruption == "checksum_sha256":
        observed[corruption] = "changed"
    else:
        observed[corruption] = {
            **observed[corruption], "planted-extra": "changed",
        }
    monkeypatch.setattr(control_deadman, "_read_object", lambda _key: observed)
    with pytest.raises(control_deadman.ControlDeadmanError):
        control_deadman._recover_cleanup_intent(
            state, _stored_object(
                state, control_deadman._state_metadata(state),
                key=launch_state.state_key(state["packet_sha256"]),
            ),
        )


def test_concurrent_deadman_recovery_entries_adopt_one_destroying_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("concurrent-production-cleanup-recovery-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="a" * 64)
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {"machine_id": 771133,
                                    "container_id": "future-container-abc",
                                    "stop_context_sha256": "b" * 64}),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "c" * 64}),
        ("START_COMMIT_INTENT", {}),
        ("START_COMMIT_PROVEN_ABSENT", {}),
    ):
        state = launch_state.transition(state, phase, **fields)
    cleanup = {
        "schema": "pg1-native-gateway-orphan-cleanup/v2",
        "status": "NO_SCIENCE_DESTROY_REQUEST_REQUIRED_STOP_UNCONFIRMED",
        "packet_sha256": packet["content_address"]["digest"],
        "instance_id": 778899, "failure": "PLANTED_CONCURRENT_RESTART",
        "stop_attempts": 6, "errors": 0,
        "consecutive_terminal_observations": 0, "science_released": False,
        "science_state": launch_state.SCIENCE_DISABLED,
        "destroy_gate": "PRE_SCIENCE_ZERO_BODY_DURABILITY",
        "start_commit_classification": "START_COMMIT_PROVEN_ABSENT",
    }
    state_key = launch_state.state_key(packet["content_address"]["digest"])
    cleanup_key = launch_gateway._cleanup_key(packet)
    memory = {
        state_key: _stored_object(
            state, launch_gateway._state_metadata(state), etag="state-etag",
            key=state_key,
        ),
        cleanup_key: _stored_object(
            cleanup, launch_gateway._cleanup_metadata(cleanup), etag="cleanup-etag",
            key=cleanup_key,
        ),
    }
    lock = multiprocessing.Lock()

    def read(key: str) -> dict[str, Any]:
        with lock:
            return copy.deepcopy(memory.get(key, {"status": "ABSENT"}))

    def put(*, key: str, payload: dict[str, Any], metadata: dict[str, str],
            prior_etag: str | None = None) -> dict[str, Any]:
        with lock:
            current = memory[key]
            if current["etag"] != prior_etag:
                raise control_deadman.ControlDeadmanError("planted concurrent CAS loss")
            memory[key] = _stored_object(
                payload, metadata, etag="destroying-etag",
                version_id="destroying-version", key=key,
            )
            return {
                "etag": "destroying-etag", "version_id": "destroying-version",
                "checksum_sha256": memory[key]["checksum_sha256"],
            }

    monkeypatch.setattr(control_deadman, "_read_object", read)
    monkeypatch.setattr(control_deadman, "_put", put)
    monkeypatch.setattr(launch_state.time, "time", lambda: 1234567890.0)
    monkeypatch.setattr(control_deadman, "TRANSACTION", tmp_path / "transaction.json")
    monkeypatch.setattr(control_deadman, "STATE_RECEIPT", tmp_path / "state-receipt.json")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda _index: control_deadman._recover_cleanup_intent(
                state, copy.deepcopy(memory[state_key])
            ),
            range(2),
        ))
    winner = memory[state_key]["body"]
    assert winner["phase"] == "CLEANUP_DESTROYING"
    assert winner["sequence"] == state["sequence"] + 1
    assert launch_state.proves_direct_transition(winner, state)
    assert all(result[0]["phase"] == "CLEANUP_DESTROYING" for result in results)


def test_launch_gateway_production_and_mechanical_authority_consumers_are_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("disjoint-mechanical-authority-consumer-v1")
    packet = _gateway_packet(capsule)
    production = launch_authority.build_authorization(
        capsule, packet, one_shot_nonce="1" * 64
    )
    mechanical = launch_authority.build_mechanical_authorization(
        capsule, packet, one_shot_nonce="2" * 64
    )
    mechanical_metadata = {
        "object-kind": "mechanical-launch-authorization",
        "issuer": launch_authority.MECHANICAL_ISSUER,
        "packet-sha256": packet["content_address"]["digest"],
        "capsule-sha256": capsule["content_address"]["digest"],
        "attempt-run-id": packet["contract"]["attempt_run_id"],
    }
    monkeypatch.setattr(launch_gateway, "MECHANICAL_ROOT_CONTRACT", None)
    assert launch_gateway.validate_launch_authorization(
        production,
        _stored_object(production, launch_authority.metadata(production)),
        packet, capsule,
    ) == production
    with pytest.raises(launch_authority.LaunchAuthorityError):
        launch_gateway.validate_launch_authorization(
            mechanical, _stored_object(mechanical, mechanical_metadata),
            packet, capsule,
        )
    monkeypatch.setattr(
        launch_gateway, "MECHANICAL_ROOT_CONTRACT", Path("/offline/mechanical.json")
    )
    assert launch_gateway.validate_launch_authorization(
        mechanical, _stored_object(mechanical, mechanical_metadata), packet, capsule,
    ) == mechanical
    with pytest.raises(launch_authority.LaunchAuthorityError):
        launch_gateway.validate_launch_authorization(
            production,
            _stored_object(production, launch_authority.metadata(production)),
            packet, capsule,
        )


def test_mechanical_authority_cli_never_calls_production_issuer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    capsule = _capsule("mechanical-cli-never-production-issuer-v1")
    packet = _gateway_packet(capsule)
    calls: list[str] = []
    monkeypatch.setattr(
        launch_authority, "configure_from_source_location", lambda: {
            "contract": {"may_mint_production_authority": False}
        },
    )
    monkeypatch.setattr(
        launch_authority, "_fixed_json",
        lambda path: capsule if path == launch_authority.CAPSULE else packet,
    )
    monkeypatch.setattr(
        launch_authority, "issue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production issuer was called")
        ),
    )
    monkeypatch.setattr(
        launch_authority, "issue_mechanical",
        lambda *_args, **_kwargs: (
            calls.append("mechanical")
            or {"schema": "pg1-native-mechanical-authority-receipt/v1"}
        ),
    )
    assert launch_authority.main([
        "--issue", "--capsule", str(launch_authority.CAPSULE),
        "--packet", str(launch_authority.PACKET), "--one-shot-nonce", "3" * 64,
        "--receipt", str(launch_authority.RECEIPT),
    ]) == 0
    assert calls == ["mechanical"]
    assert "pg1-native-mechanical-authority-receipt/v1" in capsys.readouterr().out


@pytest.mark.parametrize("corruption", ("raw-bytes", "raw-sha256"))
def test_gateway_rejects_parsed_state_with_corrupt_raw_storage_identity(
    corruption: str,
) -> None:
    capsule = _capsule(f"raw-state-corruption-{corruption}-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="d" * 64)
    observed = _stored_object(state, launch_gateway._state_metadata(state))
    if corruption == "raw-bytes":
        observed["raw_bytes"] = observed["raw_bytes"] + b" "
    else:
        observed["raw_sha256"] = "0" * 64
    with pytest.raises(launch_gateway.LaunchGatewayError, match="raw identity"):
        launch_gateway._verify_stored_state_observation(observed, packet=packet)


def test_gateway_unknown_release_cleanup_never_publishes_destroy_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("gateway-unknown-release-stop-only-v1")
    packet = _gateway_packet(capsule)
    publications: list[str] = []
    monkeypatch.setattr(
        launch_gateway, "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0,
            "{}" if "stop" in command else json.dumps(
                [{"id": 778899, "actual_status": "running"}]
            ), "",
        ),
    )
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        launch_gateway, "_put_immutable_verified",
        lambda *, key, **_kwargs: (
            publications.append(key)
            or {"version_id": "v", "etag": "e", "checksum_sha256": "c"}
        ),
    )
    receipt = launch_gateway._cleanup(
        778899, packet, failure="PLANTED_AMBIGUOUS_START",
        science_state=launch_state.SCIENCE_RELEASE_UNKNOWN,
        start_commit_classification="SCIENCE_RELEASE_UNKNOWN",
    )
    assert receipt["science_state"] == launch_state.SCIENCE_RELEASE_UNKNOWN
    assert receipt["destroy_gate"] is None
    assert len(publications) == 1 and "/gateway-cleanup/" in publications[0]


@pytest.mark.parametrize(
    "forbidden",
    (
        "--matrix", "--config", "--task-root", "--tokenizer-preflight",
        "--cache-root", "--lineage", "--lineage-role", "--entity-role",
        "--instance-id", "--output-dir", "--raw-terminal", "--terminal",
        "--expected-run-id", "--output",
    ),
)
def test_normal_execution_argv_has_no_bound_input_override(forbidden: str) -> None:
    capsule = _capsule()
    for name in ("controller", "job", "cell", "worker", "critic"):
        assert forbidden not in capsule["contract"]["consumers"][name]["execution_argv"]


TAMPER_PLANTS = (
    ("science_inputs.matrix.path", "studies/wrong-matrix.yaml"),
    ("science_inputs.matrix.sha256", "0" * 64),
    ("science_inputs.primary_config.path", "studies/wrong-config.yaml"),
    ("science_inputs.primary_config.sha256", "5" * 64),
    ("science_inputs.task_root", "studies/wrong-tasks"),
    ("science_inputs.discovery_manifest.path", "studies/wrong-manifest.jsonl"),
    ("science_inputs.discovery_manifest.sha256", "1" * 64),
    ("science_inputs.tokenizer_terminal.path", "studies/wrong-tokenizer.json"),
    ("science_inputs.tokenizer_terminal.sha256", "2" * 64),
    ("science_inputs.lineage", "pg2_sigma"),
    ("science_inputs.lineage_role", "supporting"),
    ("science_inputs.entity_role", "base"),
    ("science_inputs.model.repo", "wrong/model"),
    ("science_inputs.model.revision", "3" * 40),
    ("science_inputs.model.manifest_path", "studies/wrong-model.json"),
    ("science_inputs.model.manifest_sha256", "4" * 64),
    ("canonical_paths.staged_root", "/tmp/override-root"),
    ("canonical_paths.source_contract", "/tmp/override-source-contract"),
    ("canonical_paths.matrix", "/tmp/override-matrix"),
    ("canonical_paths.primary_config", "/tmp/override-config"),
    ("canonical_paths.task_root", "/tmp/override-tasks"),
    ("canonical_paths.discovery_manifest", "/tmp/override-manifest"),
    ("canonical_paths.tokenizer_terminal", "/tmp/override-tokenizer-terminal"),
    ("canonical_paths.cache_root", "/tmp/override-cache"),
    ("canonical_paths.tokenizer_evidence", "/tmp/override-tokenizer"),
    ("canonical_paths.model_manifest", "/tmp/override-model"),
    ("canonical_paths.output_dir", "/tmp/override-output"),
    ("canonical_paths.raw_terminal", "/tmp/override-terminal"),
    ("canonical_paths.critic_output", "/tmp/override-critic"),
    ("canonical_paths.scratch_pointer", "/tmp/override-scratch"),
)


@pytest.mark.parametrize(("field", "value"), TAMPER_PLANTS)
def test_every_bound_science_input_and_path_tamper_fails(field: str, value: str) -> None:
    capsule = _capsule()
    target: Any = capsule["contract"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(CapsuleError):
        controller.capsule_preflight(_readdress(capsule))


def test_production_authority_commands_are_fixed_conditional_s3_operations(tmp_path: Path) -> None:
    capsule = _capsule()
    context = _stop_context(capsule, instance_id=730001)
    authorization = ProductionAuthorizationStore.build_authorization(
        capsule, stop_context=context
    )
    data = capsule_module.canonical_bytes(authorization)
    issue = ProductionAuthorizationStore.issue_command(
        capsule, tmp_path / "body", data
    )
    consume = ProductionAuthorizationStore.consume_command(
        capsule, authorization, tmp_path / "marker", b"marker"
    )
    load = ProductionAuthorizationStore.load_command(capsule, tmp_path / "target")
    owner = ProductionAuthorizationStore.owner_consume_command(
        capsule, authorization, "critic", tmp_path / "owner-marker", b"owner"
    )
    verify = ProductionAuthorizationStore.verify_consume_command(
        ProductionAuthorizationStore.consumed_key(capsule, authorization),
        "version-7",
        tmp_path / "verified-marker",
    )
    for command in (issue, consume, load, owner, verify):
        assert command[:5] == [
            str(capsule_module.AWS_EXECUTABLE),
            "--profile", "fuelos", "--region", "us-east-1"
        ]
        assert command[5] == "s3api"
        assert command[command.index("--bucket") + 1] == capsule_module.AUTHORIZATION_BUCKET
        assert capsule_module.AUTHORIZATION_PREFIX in command[command.index("--key") + 1]
        assert command[command.index("--expected-bucket-owner") + 1] == capsule_module.EXPECTED_BUCKET_OWNER
    assert issue[issue.index("--if-none-match") + 1] == "*"
    assert consume[consume.index("--if-none-match") + 1] == "*"
    assert "--if-none-match" not in load
    assert owner[owner.index("--if-none-match") + 1] == "*"
    assert "--if-none-match" not in verify
    for command in (issue, consume, owner):
        for flag in (
            "--checksum-algorithm", "--checksum-sha256", "--content-length",
            "--content-type", "--server-side-encryption", "--metadata",
        ):
            assert flag in command
    assert authorization["contract"]["issuer"] == capsule_module.PRODUCTION_AUTHORIZATION_ISSUER
    assert authorization["contract"]["authorization_id"] == capsule_module.production_authorization_id(capsule)
    assert authorization["contract"]["issued_key"] == capsule_module.production_issued_key(capsule)


def test_production_backend_commands_are_fixed_s3_stop_and_destroy(tmp_path: Path) -> None:
    capsule = _capsule()
    context = _stop_context(capsule, instance_id=771991)
    backend = backend_module.ProductionBackend(
        capsule=capsule, instance_id=771991, context=context
    )
    source = tmp_path / "terminal.json"
    source.write_text('{"status":"tiny"}\n', encoding="utf-8")
    key = capsule["contract"]["attempt_identity"]["terminal_key"]
    upload = backend.upload_command(
        kind="raw_terminal", source_path=source, object_key=key
    )
    verify = backend.download_command(key, "version-7", tmp_path / "download")
    for command in (upload, verify):
        assert command[:5] == [
            str(capsule_module.AWS_EXECUTABLE),
            "--profile", "fuelos", "--region", "us-east-1"
        ]
        assert command[5] == "s3api"
        assert command[command.index("--bucket") + 1] == capsule_module.AUTHORIZATION_BUCKET
    for flag in (
        "--if-none-match", "--checksum-sha256", "--content-length",
        "--content-type", "--server-side-encryption", "--metadata",
    ):
        assert flag in upload
    assert "--version-id" in verify and "--checksum-mode" in verify
    assert backend.stop_command("scoped-key") == [
        str(capsule_module.VAST_EXECUTABLE), "stop", "instance", "771991",
        "--api-key", "scoped-key", "--raw"
    ]
    assert not hasattr(backend, "destroy_command")
    assert "vast_destroy_command" not in inspect.getsource(backend_module.ProductionBackend)
    assert "wait_vast_terminal" not in inspect.getsource(
        backend_module.ProductionBackend.request_destroy
    )


def test_production_runtime_and_stopper_commands_are_fully_pinned(tmp_path: Path) -> None:
    capsule = _capsule()
    runtime = capsule["contract"]["production_runtime"]
    assert runtime["aws"]["executable"] == str(capsule_module.AWS_EXECUTABLE)
    assert runtime["aws"]["profile"] == "fuelos"
    assert runtime["aws"]["region"] == "us-east-1"
    assert runtime["aws"]["config_path"] == str(capsule_module.AWS_CONFIG_PATH)
    assert runtime["aws"]["credentials_path"] == str(
        capsule_module.AWS_CREDENTIALS_PATH
    )
    assert runtime["vast"]["version"] == "0.5.0"
    assert runtime["vast"]["api_key_path"] == str(capsule_module.VAST_API_KEY_PATH)
    assert launcher.aws_version_command() == [
        str(capsule_module.AWS_EXECUTABLE), "--profile", "fuelos",
        "--region", "us-east-1", "--version"
    ]
    context = _stop_context(capsule, instance_id=441122)
    stopper = backend_module.ProductionStopper(capsule=capsule, context=context)
    stopper.bind_instance(441122)
    assert stopper.primary_command("scoped-key") == [
        str(capsule_module.VAST_EXECUTABLE), "stop", "instance", "441122",
        "--api-key", "scoped-key", "--raw"
    ]
    marker = b'{"tiny":"guardian"}'
    fallback = stopper.fallback_command(tmp_path / "guardian.json", marker)
    assert fallback[:5] == [
        str(capsule_module.AWS_EXECUTABLE),
        "--profile", "fuelos", "--region", "us-east-1"
    ]
    assert fallback[fallback.index("--key") + 1] == stopper.guardian_key()
    assert "--if-none-match" in fallback and "--checksum-sha256" in fallback


def test_tracked_vast_list_nested_rows_and_state_fields_are_accepted() -> None:
    capsule = _capsule()
    context = _stop_context(capsule)
    running = provider.validate_vast_show(
        VAST_SHAPES["show_list_running"], context, allowed_states={"running"}
    )
    stopped = provider.validate_vast_show(
        VAST_SHAPES["show_nested_stopped"], context, allowed_states={"stopped"}
    )
    loading = provider.validate_vast_show(
        VAST_SHAPES["show_nested_list_loading"], context, allowed_states={"loading"}
    )
    assert (running["id"], running["state"]) == (778899, "running")
    assert stopped["state"] == "stopped"
    assert loading["state"] == "loading"


def test_tracked_vast_wrong_direct_row_and_malformed_rows_are_rejected() -> None:
    context = _stop_context(_capsule())
    for response in (
        VAST_SHAPES["show_direct_wrong_row"],
        {"instances": "malformed"},
        [{"id": 778899, "label": "state-is-absent"}],
    ):
        with pytest.raises(CapsuleError):
            provider.validate_vast_show(response, context, allowed_states={"running"})


def test_destroy_terminal_proof_requires_three_tracked_absences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _stop_context(_capsule())
    sequence = iter(VAST_SHAPES["destroy_absence_sequence"])
    monkeypatch.setattr(
        provider,
        "run_json",
        lambda _command: next(sequence),
    )
    monkeypatch.setattr(provider.time, "sleep", lambda _seconds: None)
    proof = provider.wait_vast_terminal(context, "tracked-key", terminal_state="destroyed")
    assert proof == {"id": 778899, "state": "destroyed", "consecutive_absences": 3}


def test_show_instance_and_guardian_marker_bind_full_current_instance() -> None:
    capsule = _capsule()
    context = _stop_context(capsule)
    identity = context["contract"]
    response = VAST_SHAPES["show_list_running"]
    assert provider.validate_vast_show(
        response, context, allowed_states={"running"}
    )["id"] == identity["instance_id"]
    marker = {
        "schema": "pg1-native-stop-guardian-request/v3",
        "capsule_sha256": identity["capsule_sha256"],
        "attempt_run_id": identity["attempt_run_id"],
        "stop_context_sha256": context["content_address"]["digest"],
        "instance_id": identity["instance_id"],
        "offer_id": identity["offer_id"],
        "machine_id": identity["machine_id"],
        "container_id": identity["container_id"],
        "failure": "CapsuleError",
        "failure_pointer": "/workspace/parallax/failure-stop-pointer.json",
        "failure_pointer_sha256": None,
        "artifact_manifest": "/workspace/parallax/artifacts.manifest.json",
        "scratch_preserved": True,
    }
    assert guardian.validate_marker(marker, context) == marker
    marker["instance_id"] = 0
    with pytest.raises(CapsuleError):
        guardian.validate_marker(marker, context)


def test_artifact_inventory_and_bootstrap_are_authoritative_production_contracts() -> None:
    capsule = _capsule()
    attempt = capsule["contract"]["attempt_identity"]
    inventory = capsule["contract"]["artifact_contract"]
    assert inventory["bundle_key"] == attempt["artifact_key"]
    assert inventory["manifest_key"] == attempt["artifact_manifest_key"]
    assert inventory["receipts_key"] == attempt["artifact_receipts_key"]
    assert inventory["teardown_requires_all_inventory_verified"] is True
    for label in (
        "authorization", "authorization_receipt", "materialization_receipt",
        "hydration_evidence", "interface_control", "raw_terminal",
        "critic_audit", "scratch_pointer",
    ):
        assert label in inventory["required_singletons"]
    for function in (launcher.execute, controller.main, worker.main):
        source = inspect.getsource(function)
        assert "stop_preserve_scratch" in source
    bootstrap_source = (ROOT / capsule_module.BOOTSTRAP_PATH).read_text(encoding="utf-8")
    assert "_publish_marker" in bootstrap_source
    assert "_supervise_guardian" in bootstrap_source
    production = "\n".join(
        inspect.getsource(function)
        for function in (
            bootstrap.deploy,
            deployment.provision_and_launch,
            launcher.execute,
            controller.execute,
            worker.finalize_attempt,
        )
    )
    assert "pg1_native_rehearsal" not in production
    assert "mechanical_bootstrap" not in production


def test_deterministic_authorization_rejects_second_consume(tmp_path: Path) -> None:
    capsule = _capsule()
    mechanical = compile_mechanical_contract(capsule)
    store = DeterministicAuthorizationStore(tmp_path / "authority")
    authorization = store.issue(capsule, mechanical)
    receipt = store.consume(capsule, mechanical, authorization)
    assert receipt["status"] == "MECHANICAL_CONSUMED_EXACTLY_ONCE"
    with pytest.raises(FileExistsError):
        store.consume(capsule, mechanical, authorization)


def test_deterministic_authorization_atomic_race_has_exactly_one_winner(tmp_path: Path) -> None:
    capsule = _capsule(ATTEMPT_IDS[1])
    mechanical = compile_mechanical_contract(capsule)
    store = DeterministicAuthorizationStore(tmp_path / "authority")
    authorization = store.issue(capsule, mechanical)

    def consume() -> str:
        try:
            store.consume(capsule, mechanical, authorization)
            return "consumed"
        except FileExistsError:
            return "replay-rejected"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _index: consume(), range(8)))
    assert outcomes.count("consumed") == 1
    assert outcomes.count("replay-rejected") == 7


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("issuer", "SELF_ASSERTED"),
        ("store_uri", "file://caller-controlled"),
        ("capsule_sha256", "0" * 64),
        ("attempt_run_id", "wrong-attempt"),
        ("instance_id", 0),
        ("h100_authority", False),
    ),
)
def test_forged_or_unbound_authorization_is_rejected(
    field: str, value: Any, tmp_path: Path
) -> None:
    capsule = _capsule()
    authorization = ProductionAuthorizationStore.build_authorization(
        capsule, stop_context=_stop_context(capsule)
    )
    authorization["contract"][field] = value
    authorization["content_address"]["digest"] = capsule_module.content_sha256(
        authorization["contract"]
    )
    with pytest.raises(CapsuleError):
        capsule_module.validate_execution_authorization(authorization, capsule)


def test_production_validator_rejects_mechanical_authority_and_arbitrary_keys(
    tmp_path: Path,
) -> None:
    capsule = _capsule()
    mechanical = compile_mechanical_contract(capsule)
    rehearsal = DeterministicAuthorizationStore(tmp_path / "authority").issue(
        capsule, mechanical
    )
    with pytest.raises(CapsuleError):
        capsule_module.validate_execution_authorization(rehearsal, capsule)
    authorization = ProductionAuthorizationStore.build_authorization(
        capsule, stop_context=_stop_context(capsule)
    )
    authorization["contract"]["issued_key"] = "caller/chosen.json"
    authorization["content_address"]["digest"] = capsule_module.content_sha256(
        authorization["contract"]
    )
    with pytest.raises(CapsuleError):
        capsule_module.validate_execution_authorization(authorization, capsule)

    authorization = ProductionAuthorizationStore.build_authorization(
        capsule, stop_context=_stop_context(capsule)
    )
    marker = capsule_module.production_consumption_marker(capsule, authorization)
    receipt = {
        "schema": capsule_module.AUTHORIZATION_RECEIPT_SCHEMA,
        "status": "CONSUMED_EXACTLY_ONCE",
        "authorization_sha256": authorization["content_address"]["digest"],
        "capsule_sha256": capsule["content_address"]["digest"],
        "source_run_id": capsule["contract"]["source_identity"]["source_run_id"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "one_shot_consumed": True,
        "h100_authority": True,
        "issuer": authorization["contract"]["issuer"],
        "issuer_backend": authorization["contract"]["issuer_backend"],
        "issuer_account": authorization["contract"]["issuer_account"],
        "store_uri": authorization["contract"]["store_uri"],
        "consume_key": "unrelated/caller-selected-object.json",
        "consume_version_id": "version-1",
        "consume_etag": "etag-1",
        "consume_checksum_sha256": "checksum-1",
        "consume_marker_sha256": capsule_module.content_sha256(marker),
    }
    with pytest.raises(CapsuleError):
        capsule_module.validate_authorization_receipt(
            receipt, authorization, capsule
        )


@pytest.mark.parametrize("attempt_id", ATTEMPT_IDS)
def test_declared_mechanical_argv_runs_end_to_end_without_cloud_or_ml(
    attempt_id: str, tmp_path: Path
) -> None:
    completed, root, _capsule_value = _run_mechanical(tmp_path, attempt_id=attempt_id)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["owner_trace"] == list(capsule_module.CONSUMER_ORDER)
    assert result["attempt_run_id"] == attempt_id
    assert result["bootstrap_trace"] == [
        "MECHANICAL_FIXED_LAYOUT_CREATED",
        "MECHANICAL_EXACT_RUNTIME_STAGED",
        "MECHANICAL_REBASED_PRODUCTION_ARGV_EXECUTED",
        "MECHANICAL_PRODUCTION_ENTRYPOINT_INVOKED_BOOTSTRAP",
        "MECHANICAL_STOP_CONTEXT_VALIDATED", "MECHANICAL_GUARDIAN_STARTED",
        "MECHANICAL_RUNTIME_MATERIALIZED", "MECHANICAL_NATIVE_ARGV_INVOKED",
        "MECHANICAL_GUARDIAN_TERMINAL_RECEIPT_SUPERVISED",
    ]
    assert result["deployment"]["runtime_file_count"] == len(
        capsule_module.RUNTIME_PATHS
    )
    assert result["deployment"]["production_argv"] == _capsule(attempt_id)["contract"][
        "entrypoints"
    ]["production_argv"]
    assert result["deployment"]["invoked_argv"] == _capsule(attempt_id)["contract"][
        "entrypoints"
    ]["bootstrap_argv"]
    assert result["deployment"]["provider_calls"] == 0
    assert result["deployment"]["executed"] is True
    assert result["deployment"]["execution_returncode"] == 0
    for row in result["deployment"]["argv_equivalence"]:
        if row["production"].startswith("/"):
            assert row["path_rebased"].endswith(row["production"])
        else:
            assert row["path_rebased"] == row["production"]
    assert set(result["deployment"]["mounts"]) == {
        str(path) for path in deployment.FIXED_MOUNTS
    }
    assert result["deployment"]["layout_receipt"]["mode"] == "0400"
    assert result["hydration"]["selected_visual_bodies"] == 2
    assert result["hydration"]["selected_imageless_items"] == 2
    assert result["hydration"]["unselected_imageless_items"] == 1
    assert result["durable_kinds"] == [
        "artifact_bundle", "artifact_manifest", "artifact_receipts",
        "critic_audit", "raw_terminal", "scratch_pointer",
    ]
    for field in (
        "external_commands", "archive_body_reads",
        "confirmation_items_read", "ml_imports",
        "h100_instances_created",
    ):
        assert result[field] == 0
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert events[-1] == "destroy_verified_terminal_after_complete_artifacts"


@pytest.mark.parametrize(
    "failure",
    (
        "pre_setup", "model_body_mismatch", "tokenizer_asset_mismatch",
        "task_manifest_mismatch", "missing_image_body", "extra_image_body",
        "wrong_image_body", "artifact_missing_body", "artifact_extra_body",
        "artifact_wrong_body", "terminal_upload", "critic_execution",
        "audit_upload", "verification", "primary_stop_failure",
        "fallback_stop_failure", "guardian_marker_failure", "guardian_race",
        "guardian_replay", "guardian_timeout", "provider_nonzero",
        "provider_timeout",
    ),
)
def test_each_lifecycle_failure_stops_and_preserves_without_destroy(
    failure: str, tmp_path: Path
) -> None:
    completed, root, _capsule_value = _run_mechanical(
        tmp_path, attempt_id=f"failure-{failure}-91", failure=failure
    )
    assert completed.returncode != 0
    result = json.loads((root / "rehearsal-result.json").read_text())
    assert result["status"] == "FAILED_STOPPED_OR_EXHAUSTED_ARTIFACTS_PRESERVED"
    assert result["destroy_invoked"] is False
    assert result["failure_pointer_preserved"] is True
    assert result["external_commands"] == result["h100_instances_created"] == 0
    assert (root / "scratch" / "failure-stop-pointer.json").is_file()
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    if failure in {"fallback_stop_failure", "guardian_marker_failure"}:
        assert result["stop_error"] == "CapsuleError"
        assert events[-1] == "guardian_marker_failed"
    elif failure in {
        "primary_stop_failure", "guardian_race", "guardian_replay",
        "guardian_timeout", "provider_nonzero", "provider_timeout",
    }:
        assert result["stop"]["status"] == "GUARDIAN_STOP_VERIFIED_TERMINAL"
        assert events[-1] == "guardian_stop_verified_terminal"
    else:
        assert result["stop"]["status"] == "PRIMARY_STOP_VERIFIED_TERMINAL"
        assert events[-1] == "primary_stop_semantics_verified"
    assert "destroy_verified_terminal_after_complete_artifacts" not in events


def test_success_order_is_terminal_and_scratch_then_critic_audit_then_destroy(tmp_path: Path) -> None:
    completed, root, _capsule_value = _run_mechanical(
        tmp_path, attempt_id="ordering-proof-009"
    )
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    required = (
        "runtime_hydration_verified_before_science",
        "raw_terminal_durable", "raw_terminal_downloaded_verified",
        "scratch_pointer_durable", "scratch_pointer_downloaded_verified",
        "critic_execution_started", "independent_critic_complete",
        "critic_audit_durable", "critic_audit_downloaded_verified",
        "artifact_bundle_durable", "artifact_bundle_downloaded_verified",
        "artifact_manifest_durable", "artifact_manifest_downloaded_verified",
        "artifact_receipts_durable", "artifact_receipts_downloaded_verified",
        "destroy_verified_terminal_after_complete_artifacts",
    )
    assert [events.index(event) for event in required] == sorted(events.index(event) for event in required)
    assert "failure_pointer_preserved" not in events


def test_production_owners_validate_authority_before_science_and_worker_is_stop_wrapped() -> None:
    for function in (job.execute, cell.run, worker.finalize_attempt, critic.run):
        source = inspect.getsource(function)
        assert "validate_execution_authorization" in source
        assert "validate_authorization_receipt" in source
    worker_source = inspect.getsource(worker.finalize_attempt)
    assert "ProductionStopper" in worker_source
    assert "stopper.stop_preserve_scratch" in worker_source
    assert worker_source.index("ProductionStopper") < worker_source.index("validate_consumer")
    assert worker_source.index("failure=type(error).__name__") > worker_source.index("except Exception")


def test_wrong_capsule_and_wrong_runtime_bytes_fail() -> None:
    capsule = _capsule()
    capsule["content_address"]["digest"] = "0" * 64
    with pytest.raises(CapsuleError, match="content address"):
        controller.capsule_preflight(capsule)
    capsule = _capsule()
    native_path = str(capsule_module.NATIVE_BINDINGS[1][2])
    capsule["contract"]["runtime_wiring"][native_path] = "0" * 64
    capsule["contract"]["consumers"]["source_gate"]["sha256"] = "0" * 64
    with pytest.raises(CapsuleError, match="runtime source"):
        source_gate.capsule_preflight(
            _readdress(capsule), ROOT, allow_parameterized_fixture=True
        )


def test_scientific_ast_and_fixed_native_call_trace_match_8bba643() -> None:
    baseline = json.loads((ROOT / science_guard.BASELINE_PATH).read_text(encoding="utf-8"))
    proof = science_guard.verify(ROOT, ROOT / science_guard.BASELINE_PATH)
    native = proof["native_equivalence"]
    assert proof["status"] == "GREEN_SCIENTIFIC_AST_IDENTICAL_TO_8BBA643"
    assert native["scientific_slice_sha256"] == baseline["files"][str(science_guard.CELL_PATH)]["protected_slices"]["run_from_lineage_rows"]
    assert native["controller_call_trace"] == [
        "load_issued", "consume", "capsule_preflight_graph", "execute_job",
        "finalize_attempt", "stop_preserve_scratch",
    ]
    assert native["worker_call_trace"][-2:] == ["request_destroy", "stop_preserve_scratch"]


def test_native_modules_have_no_callback_or_legacy_patch_architecture() -> None:
    paths = [ROOT / path for path in capsule_module.RUNTIME_PATHS]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.name.startswith("pg1_native"))
    for forbidden in (
        "_patched_legacy", "legacy_compatible_source_contract", "monkeypatch",
        "import boto3", "import vastai", "Callable[...,", "os.environ",
    ):
        assert forbidden not in combined
    assert not (ROOT / "infra/vast/parallax/pg1_execution_capsule_preflight.py").exists()


def test_phase_order_requires_scratch_and_verified_audit_before_destroy() -> None:
    phases = _capsule()["contract"]["phase_order"]
    assert phases.index("RAW_TERMINAL_DURABLE") < phases.index("SCRATCH_POINTER_DURABLE")
    assert phases.index("SCRATCH_POINTER_DURABLE") < phases.index("INDEPENDENT_CRITIC_COMPLETE")
    assert phases.index("CRITIC_AUDIT_DURABLE") < phases.index("DURABILITY_VERIFIED")
    assert phases[-1] == "DESTROY_REQUESTED"


def test_python_sources_parse_and_production_execute_never_imports_rehearsal() -> None:
    for path in capsule_module.RUNTIME_PATHS:
        ast.parse((ROOT / path).read_text(encoding="utf-8"))
    assert "pg1_native_rehearsal" not in inspect.getsource(controller.execute)


def test_tracked_vast_adapter_and_deployment_are_hash_bound_runtime_sources() -> None:
    capsule = _capsule("adapter-deployment-binding-v1")
    wiring = capsule["contract"]["runtime_wiring"]
    assert wiring[str(capsule_module.VAST_ADAPTER_PATH)] == capsule_module.sha256_file(
        ROOT / capsule_module.VAST_ADAPTER_PATH
    )
    assert wiring[str(capsule_module.DEPLOYMENT_PATH)] == capsule_module.sha256_file(
        ROOT / capsule_module.DEPLOYMENT_PATH
    )
    vast = capsule["contract"]["production_runtime"]["vast"]
    assert vast["adapter_path"] == str(capsule_module.VAST_ADAPTER_PATH)
    assert vast["adapter_sha256"] == wiring[str(capsule_module.VAST_ADAPTER_PATH)]
    assert VAST_SHAPES["provenance"] == {
        "normalizer": "infra/vast/parallax/guarded_launch.py:_show_instance",
        "row_identity": "infra/vast/parallax/ephemeral_guardian.py:run",
        "state_fields": ["actual_status", "cur_state"],
        "destruction_proof": (
            "infra/vast/parallax/ephemeral_s3_launch.py:_enforce_instances_destroyed"
        ),
    }


def test_vast_mutation_uses_exit_code_only_before_show_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        [str(capsule_module.VAST_EXECUTABLE), "stop"], 0, "", ""
    )
    monkeypatch.setattr(provider, "_run", lambda _command: completed)
    assert provider.run_vast_mutation([str(capsule_module.VAST_EXECUTABLE), "stop"]) == {
        "schema": "pg1-vast-mutation-exit/v1",
        "exit_code": 0,
    }


def test_vast_key_provenance_failure_still_publishes_fixed_aws_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("key-failure-fallback-v1")
    stopper = backend_module.ProductionStopper(
        capsule=capsule, context=_stop_context(capsule)
    )
    stopper.failure_pointer = tmp_path / "failure-pointer.json"
    stopper.state_path = tmp_path / "stop-state.json"
    stopper.lock_path = tmp_path / "stop.lock"
    monkeypatch.setattr(
        backend_module,
        "load_vast_key",
        lambda _context: (_ for _ in ()).throw(CapsuleError("key provenance")),
    )
    calls: list[dict[str, Any]] = []

    def publish(marker: dict[str, Any]) -> dict[str, Any]:
        calls.append(marker)
        return {
            "version_id": "marker-version-1",
            "etag": "marker-etag-1",
            "checksum_sha256": "tracked",
            "replayed_exact": False,
        }

    monkeypatch.setattr(stopper, "_publish_guardian_marker", publish)
    result = stopper.stop_preserve_scratch(failure="CapsuleError")
    assert result["status"] == "GUARDIAN_MARKER_DURABLE"
    assert len(calls) == 1
    assert calls[0]["instance_id"] == stopper.instance_id
    assert any(
        row["action"] == "PRIMARY_VAST_STOP" and row["status"] == "FAILED"
        for row in result["events"]
    )


def test_guardian_marker_conditional_race_exact_readback_is_replay_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("marker-replay-v1")
    stopper = backend_module.ProductionStopper(
        capsule=capsule, context=_stop_context(capsule)
    )
    stopper.failure_pointer = tmp_path / "failure-pointer.json"
    stopper._persist_pointer("CapsuleError")
    marker = stopper._guardian_marker("CapsuleError")
    data = capsule_module.canonical_bytes(marker)
    import base64
    import hashlib

    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    calls = 0

    def run(command: list[str]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CapsuleError("conditional create lost race")
        Path(command[-1]).write_bytes(data)
        return {
            "VersionId": "existing-version",
            "ETag": "existing-etag",
            "ChecksumSHA256": checksum,
            "ServerSideEncryption": "AES256",
            "ContentLength": len(data),
            "ContentType": "application/json",
            "Metadata": {
                "capsule-sha256": stopper.capsule_sha256,
                "attempt-run-id": stopper.attempt_run_id,
                "instance-id": str(stopper.instance_id),
                "object-kind": "stop-guardian-request",
            },
        }

    monkeypatch.setattr(stopper, "_run_json", run)
    result = stopper._publish_guardian_marker(marker)
    assert result["replayed_exact"] is True
    assert result["version_id"] == "existing-version"


class _GuardianClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _guardian_test_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardian, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(guardian, "FAILURE_POINTER_PATH", tmp_path / "failure.json")
    monkeypatch.setattr(guardian, "LOCK_PATH", tmp_path / "guardian.lock")
    monkeypatch.setattr(guardian, "PENDING_RECEIPT_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(guardian, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(guardian, "HEARTBEAT_PATH", tmp_path / "heartbeat.json")


def _guardian_persist_fixture(
    payload: dict[str, Any], _context: dict[str, Any]
) -> dict[str, Any]:
    return {
        **payload,
        "schema": "pg1-native-guardian-result/v2",
        "receipt_durable": True,
    }


def test_guardian_transient_key_failure_recovers_without_real_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("guardian-transient-key-v1")
    context = _stop_context(capsule)
    _guardian_test_paths(tmp_path, monkeypatch)
    clock = _GuardianClock()
    monkeypatch.setattr(guardian.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(guardian.time, "sleep", clock.sleep)
    keys: list[Any] = [CapsuleError("transient key"), *("scoped",) * 4]

    def load(_context: dict[str, Any]) -> str:
        value = keys.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    observations = iter(
        [
            {"id": context["contract"]["instance_id"], "state": "running"},
            {"id": context["contract"]["instance_id"], "state": "stopped"},
            {"id": context["contract"]["instance_id"], "state": "stopped"},
            {"id": context["contract"]["instance_id"], "state": "stopped"},
        ]
    )
    monkeypatch.setattr(guardian, "load_vast_key", load)
    monkeypatch.setattr(guardian, "observe_vast_instance", lambda *_args: next(observations))
    monkeypatch.setattr(
        guardian,
        "run_vast_mutation",
        lambda _command: {"schema": "pg1-vast-mutation-exit/v1", "exit_code": 0},
    )
    monkeypatch.setattr(guardian, "_persist_terminal", _guardian_persist_fixture)
    result = guardian._act(
        guardian._synthetic_marker(context, "TEST_TRANSIENT"),
        context,
        deadline=40,
    )
    assert result["terminal_state"] == "stopped"
    assert any(row["action"] == "provider_retry" for row in result["events"])
    assert clock.value > 0


def test_guardian_repeated_absence_never_manufactures_destroy_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("guardian-absence-v1")
    context = _stop_context(capsule)
    _guardian_test_paths(tmp_path, monkeypatch)
    clock = _GuardianClock()
    monkeypatch.setattr(guardian.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(guardian.time, "sleep", clock.sleep)
    monkeypatch.setattr(guardian, "load_vast_key", lambda _context: "scoped")
    monkeypatch.setattr(guardian, "observe_vast_instance", lambda *_args: None)
    monkeypatch.setattr(
        guardian,
        "run_vast_mutation",
        lambda _command: {"schema": "pg1-vast-mutation-exit/v1", "exit_code": 0},
    )
    monkeypatch.setattr(guardian, "_persist_terminal", _guardian_persist_fixture)
    with pytest.raises(CapsuleError, match="STOP remains unconfirmed"):
        guardian._act(
            guardian._synthetic_marker(context, "TEST_ABSENCE"),
            context,
            deadline=40,
        )
    assert not guardian.RESULT_PATH.exists()


def test_guardian_three_failed_stops_never_bypass_backend_destroy_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("guardian-stop-fallback-v1")
    context = _stop_context(capsule)
    _guardian_test_paths(tmp_path, monkeypatch)
    clock = _GuardianClock()
    monkeypatch.setattr(guardian.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(guardian.time, "sleep", clock.sleep)
    monkeypatch.setattr(guardian, "load_vast_key", lambda _context: "scoped")
    observations: list[dict[str, Any] | None] = [
        {"id": context["contract"]["instance_id"], "state": "running"},
        {"id": context["contract"]["instance_id"], "state": "running"},
        {"id": context["contract"]["instance_id"], "state": "running"},
        None,
        None,
        None,
    ]
    monkeypatch.setattr(
        guardian, "observe_vast_instance", lambda *_args: observations.pop(0)
    )
    mutations: list[str] = []

    def mutate(command: list[str]) -> dict[str, Any]:
        action = command[1]
        mutations.append(action)
        if action == "stop":
            raise CapsuleError("tracked mutation nonzero")
        return {"schema": "pg1-vast-mutation-exit/v1", "exit_code": 0}

    monkeypatch.setattr(guardian, "run_vast_mutation", mutate)
    monkeypatch.setattr(guardian, "_persist_terminal", _guardian_persist_fixture)
    with pytest.raises(CapsuleError, match="STOP remains unconfirmed"):
        guardian._act(
            guardian._synthetic_marker(context, "TEST_STOP_FALLBACK"),
            context,
            deadline=100,
        )
    assert mutations[:3] == ["stop", "stop", "stop"]
    assert set(mutations) == {"stop"}
    assert not guardian.RESULT_PATH.exists()


def test_guardian_deadline_requires_restart_without_unverified_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("guardian-deadline-v1")
    context = _stop_context(capsule)
    _guardian_test_paths(tmp_path, monkeypatch)
    clock = _GuardianClock()
    monkeypatch.setattr(guardian.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(guardian.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        guardian,
        "load_vast_key",
        lambda _context: (_ for _ in ()).throw(CapsuleError("rate limited")),
    )
    monkeypatch.setattr(guardian, "_persist_terminal", _guardian_persist_fixture)
    with pytest.raises(CapsuleError, match="STOP remains unconfirmed"):
        guardian._act(
            guardian._synthetic_marker(context, "TEST_DEADLINE"),
            context,
            deadline=20,
        )
    heartbeat = json.loads(guardian.HEARTBEAT_PATH.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "STOP_UNCONFIRMED_RESTART_REQUIRED"
    assert not guardian.RESULT_PATH.exists()


def test_failed_stop_cannot_destroy_without_complete_durability_gate() -> None:
    capsule = _capsule("undurable-stop-destroy-gate-v1")
    incomplete = {
        "schema": backend_module.TEARDOWN_SCHEMA,
        "status": "FINAL_DURABILITY_RECEIPT_DOWNLOADED_VERIFIED_FOR_TEARDOWN",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "teardown_authorized": True,
    }
    with pytest.raises(CapsuleError, match="complete final or pre-science"):
        backend_module.validate_destroy_evidence(incomplete, capsule)
    assert "vast_destroy_command" not in inspect.getsource(guardian)
    assert "vast_destroy_command" not in inspect.getsource(deadman)


def test_local_failure_lifecycle_triggers_stop_during_s3_marker_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("local-stop-during-s3-outage-v1")
    context = _stop_context(capsule)
    _guardian_test_paths(tmp_path, monkeypatch)
    lifecycle_module.transition(
        guardian.FAILURE_POINTER_PATH,
        context,
        "FAILURE_STOP_REQUIRED",
        reason="PLANTED_S3_OUTAGE",
        expected_sequence=0,
    )
    observations = iter(
        [
            {"id": context["contract"]["instance_id"], "state": "running"},
            {"id": context["contract"]["instance_id"], "state": "stopped"},
            {"id": context["contract"]["instance_id"], "state": "stopped"},
            {"id": context["contract"]["instance_id"], "state": "stopped"},
        ]
    )
    mutations: list[list[str]] = []
    monkeypatch.setattr(
        guardian, "_probe_marker", lambda _context: (_ for _ in ()).throw(
            CapsuleError("planted S3 outage")
        )
    )
    monkeypatch.setattr(guardian, "load_vast_key", lambda _context: "scoped")
    monkeypatch.setattr(guardian, "observe_vast_instance", lambda *_args: next(observations))
    monkeypatch.setattr(
        guardian,
        "run_vast_mutation",
        lambda command: (mutations.append(command) or {"exit_code": 0}),
    )
    monkeypatch.setattr(guardian, "_persist_terminal", _guardian_persist_fixture)
    monkeypatch.setattr(guardian.time, "sleep", lambda _seconds: None)
    result = guardian.guard(capsule=capsule, context=context)
    assert result["terminal_state"] == "stopped"
    assert len(mutations) == 3
    assert "stop" in mutations[0]


def test_local_lifecycle_is_monotonic_and_guardian_ignores_nonfailure_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("monotonic-local-lifecycle-v1")
    context = _stop_context(capsule)
    lifecycle_path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(deployment, "LIFECYCLE_PATH", lifecycle_path)
    armed = deployment._transition_lifecycle(context, "ARMED", reason="test")
    running = deployment._transition_lifecycle(
        context, "RUNNING_HEARTBEAT", reason="test"
    )
    monkeypatch.setattr(guardian, "FAILURE_POINTER_PATH", lifecycle_path)
    assert guardian._probe_local_pointer(context) is None
    failed = deployment._transition_lifecycle(
        context, "FAILURE_STOP_REQUIRED", reason="test"
    )
    assert [armed["sequence"], running["sequence"], failed["sequence"]] == [1, 2, 3]
    assert guardian._probe_local_pointer(context) is not None
    stopped = deployment._transition_lifecycle(
        context,
        "TERMINAL_STOPPED_PRESERVED",
        reason="test",
        terminal_receipt={
            "receipt_durable": True,
            "terminal_state": "stopped",
            "consecutive_terminal_observations": 3,
        },
    )
    assert stopped["sequence"] == 4
    assert guardian._probe_local_pointer(context) is None
    with pytest.raises(deployment.DeploymentError, match="forbidden lifecycle transition"):
        deployment._transition_lifecycle(
            context, "RUNNING_HEARTBEAT", reason="forbidden regression"
        )
    with pytest.raises(deployment.DeploymentError, match="destroy terminal requires"):
        deployment._transition_lifecycle(
            context,
            "TERMINAL_DESTROYED_DURABLE",
            reason="missing gate",
            terminal_receipt={"receipt_durable": True, "terminal_state": "destroyed"},
        )
    destroyed = deployment._transition_lifecycle(
        context,
        "TERMINAL_DESTROYED_DURABLE",
        reason="complete durability gate",
        terminal_receipt={
            "receipt_durable": True,
            "terminal_state": "destroyed",
            "destroy_gate": "COMPLETE_ARTIFACT_DURABILITY",
            "consecutive_absences": 3,
        },
    )
    assert destroyed["sequence"] == 5


def test_gateway_requires_deadman_ack_before_constructing_or_running_worker_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("gateway-arm-before-create-v1")
    packet = _gateway_packet(capsule)
    authorization = _gateway_authorization(capsule, packet)
    events: list[str] = []

    class DeadmanProcess:
        def poll(self) -> None:
            return None

    transactions: list[dict[str, Any]] = []
    def publish_state(
        _packet: dict[str, Any], state: dict[str, Any], *, prior_etag: str | None,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        transactions.append({"status": state["phase"], "sequence": state["sequence"],
                             "instance_id": state.get("instance_id"),
                             "science_released": state["science_state"] == "SCIENCE_RELEASED"})
        return state, {"etag": f"state-{state['sequence']}", "version_id": "offline"}

    monkeypatch.setattr(launch_gateway, "_publish_state", publish_state)
    monkeypatch.setattr(
        launch_gateway, "fetch_launch_authorization",
        lambda *_args: (authorization, {"version_id": "v", "etag": "e",
                                        "checksum_sha256": "c", "metadata": {}}),
    )
    monkeypatch.setattr(launch_gateway, "consume_launch_authorization", lambda *_args: {})
    monkeypatch.setattr(launch_gateway, "_aws_put", lambda **_kwargs: {})
    monkeypatch.setattr(
        launch_gateway, "classify_start_commit",
        lambda *_args: {"classification": "SCIENCE_RELEASED", "observed": {}},
    )
    monkeypatch.setattr(launch_gateway, "_read_object", lambda **_kwargs: {"status": "ABSENT"})
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(launch_gateway, "fixed_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        launch_gateway.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (events.append("deadman_started") or DeadmanProcess()),
    )
    monkeypatch.setattr(
        launch_gateway, "_wait_ack",
        lambda _path, _packet, status: (
            events.append(status)
            or {"status": status, "packet_sha256": packet["content_address"]["digest"],
                "transaction_sha256": "a" * 64, "destroy_authority": False,
                "etag": f"{status}-etag", "version_id": f"{status}-version"}
        ),
    )
    original_construct = launch_gateway.worker_create_command

    def construct(packet_value: dict[str, Any], capsule_value: dict[str, Any]) -> list[str]:
        events.append("worker_command_constructed")
        return original_construct(packet_value, capsule_value)

    monkeypatch.setattr(launch_gateway, "worker_create_command", construct)

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "create" in command:
            events.append("worker_create_run")
            onstart = command[command.index("--onstart-cmd") + 1]
            assert "pg1_native_onstart_materializer.py" in onstart
            assert "PG1_SCIENCE_DISABLED=1" in onstart
            assert str(capsule_module.PRODUCTION_ENTRY_PATH) not in onstart
            return subprocess.CompletedProcess(
                command, 0,
                json.dumps({"success": True, "new_contract": 778899,
                            "instance_api_key": "1" * 64}), "",
            )
        if "show" in command:
            return subprocess.CompletedProcess(
                command, 0,
                json.dumps([{"id": 778899, "machine_id": 771133,
                             "container_id": "future-container-abc",
                             "label": packet["contract"]["launch_label"],
                             "actual_status": "running"}]), "",
            )
        if "copy" in command:
            events.append("copy_start" if str(launch_gateway.START_LEASE_REMOTE) in command[-2]
                          else "copy_handoff")
            return subprocess.CompletedProcess(command, 0, "{}", "")
        raise AssertionError(command)

    monkeypatch.setattr(launch_gateway, "_run", run)
    monkeypatch.setattr(
        launch_gateway, "source_materializer_onstart",
        lambda *_args: "PG1_SCIENCE_DISABLED=1 pg1_native_onstart_materializer.py",
    )
    monkeypatch.setattr(
        launch_gateway, "_publish_stop_context",
        lambda *_args: _stop_context(capsule, instance_id=778899),
    )
    monkeypatch.setattr(
        launch_gateway, "_materialize_handoff",
        lambda *_args: ({Path("/run/parallax/fixed"): tmp_path / "fixed"},
                        {"schema": "offline-layout"}),
    )
    monkeypatch.setattr(
        launch_gateway, "_copy_all", lambda *_args: events.append("copy_handoff")
    )
    result = launch_gateway.launch(capsule, packet)
    assert result["instance_id"] == 778899
    assert events.index("PRELAUNCH_CLEANUP_ARMED") < events.index("worker_create_run")
    assert events.index("worker_create_run") < events.index("INSTANCE_BOUND_DEADMAN_ARMED")
    assert events.index("INSTANCE_BOUND_DEADMAN_ARMED") < events.index("copy_start")
    assert transactions[-1]["status"] == "SCIENCE_RELEASED"
    assert transactions[-1]["science_released"] is True
    assert all(not row["science_released"] for row in transactions[:-1])

    events.clear()
    monkeypatch.setattr(
        launch_gateway, "_wait_ack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            launch_gateway.LaunchGatewayError("planted missing durable ACK")
        ),
    )
    monkeypatch.setattr(launch_gateway, "_discover", lambda _packet: None)
    with pytest.raises(launch_gateway.LaunchGatewayError, match="missing durable ACK"):
        launch_gateway.launch(capsule, packet)
    assert events == ["deadman_started"]


@pytest.mark.cloud_cpu_only
def test_exact_copied_compiler_gateway_idle_and_production_chain_fake_provider(
    tmp_path: Path,
) -> None:
    if os.environ.get("PG1_APPROVED_AWS_FUELOS_CLOUD_RUNNER") != "1":
        pytest.skip(
            "cloud_cpu_only: refuses archive materialization outside the approved "
            "AWS fuelos cloud runner"
        )
    attempt = "exact-copied-launch-chain-v1"
    capsule = _capsule(attempt)
    root = tmp_path / "exact-root"
    services = root / ".pg1-fake-services"
    worker_root = services / "worker"
    services.mkdir(parents=True)

    def fixed(path: Path, data: bytes, mode: int = 0o400) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.chmod(path, mode)

    mechanical_body = {
        "kind": "MECHANICAL_NO_AUTHORITY", "production_authority": False,
        "provider_calls": 0, "root_rebase": str(root.resolve()),
        "fake_services": "TRACKED_DETERMINISTIC_EXECUTABLES",
        "network_scope": "LOOPBACK_OR_NONE", "may_mint_production_authority": False,
    }
    mechanical_contract = {
        "schema": "pg1-native-exact-chain-mechanical-contract/v1",
        "status": "OFFLINE_DETERMINISTIC_FAKE_SERVICES_ONLY",
        "content_address": {"algorithm": "sha256", "digest": hashlib.sha256(
            mechanical_rebase.canonical_bytes(mechanical_body)
        ).hexdigest()},
        "contract": mechanical_body,
    }
    mechanical_path = root / "run/parallax/control-plane/exact-chain-mechanical.json"
    fixed(mechanical_path, mechanical_rebase.canonical_bytes(mechanical_contract))
    assert mechanical_rebase.load(mechanical_path)["root"] == root.resolve()

    stage_source = root / "opt/parallax/control-stage/source"
    for relative, expected in capsule["contract"]["runtime_wiring"].items():
        source = ROOT / relative
        target = stage_source / relative
        fixed(target, source.read_bytes(), stat.S_IMODE(source.stat().st_mode))
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected

    fixture_root = root / "infra/vast/parallax/tests/fixtures"
    fake_aws = root / "opt/parallax/control-stage/bin/aws"
    fake_vast = root / "opt/parallax/control-stage/bin/vastai"
    fixed(fake_aws, (ROOT / "infra/vast/parallax/tests/fixtures/pg1_deterministic_fake_aws.py").read_bytes(), 0o755)
    fixed(fake_vast, (ROOT / "infra/vast/parallax/tests/fixtures/pg1_deterministic_fake_vast.py").read_bytes(), 0o755)
    fixed(services / "control.json", b'{"faults":[]}\n', 0o600)

    capsule_path = root / "run/parallax/control-plane/pg1-execution-capsule.json"
    inputs_path = root / "run/parallax/control-plane/pg1-future-launch-inputs.json"
    packet_path = root / "run/parallax/control-plane/pg1-future-launch-packet.json"
    fixed(capsule_path, capsule_module.canonical_bytes(capsule))
    aws_config = b"[profile pg1-control-deadman]\nregion=us-east-1\n"
    aws_credentials = b"[pg1-control-deadman]\naws_access_key_id=offline\naws_secret_access_key=offline\n"
    control_key = b"offline-control-vast-key\n"
    ssh_key = b"offline-control-ssh-key\n"
    runtime_receipt = (capsule["contract"]["production_runtime"]["image"] + "\n").encode()
    fixed(root / "run/parallax/control-plane/aws/config", aws_config)
    fixed(root / "run/parallax/control-plane/aws/credentials", aws_credentials)
    fixed(root / "run/secrets/pg1-native-control-plane-vast-api-key", control_key)
    fixed(root / "run/secrets/pg1-native-control-plane-ssh-private-key", ssh_key)
    fixed(root / "run/parallax/control-plane/runtime-image-digest", runtime_receipt)

    archive_hashes = {
        **{row["path"]: row["sha256"]
           for label, row in capsule["contract"]["artifacts"].items()
           if label not in launch_packet.PROTECTED_BODY_ARTIFACT_LABELS},
        **capsule["contract"]["runtime_wiring"],
    }
    assert all("/data/" not in relative for relative in archive_hashes)
    archive = tmp_path / "exact-source.tar"
    members: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as handle:
        for relative, expected in sorted(archive_hashes.items()):
            data = (ROOT / relative).read_bytes()
            mode = stat.S_IMODE((ROOT / relative).stat().st_mode)
            row = tarfile.TarInfo(relative)
            row.size = len(data); row.mode = mode; row.mtime = 0; row.uid = 0; row.gid = 0
            row.uname = ""; row.gname = ""
            handle.addfile(row, io.BytesIO(data))
            members[relative] = {"sha256": expected, "bytes": len(data),
                                 "mode": f"{mode:04o}"}
    source_data = archive.read_bytes()
    source_checksum = base64.b64encode(hashlib.sha256(source_data).digest()).decode("ascii")
    source_key = "offline/exact/source.tar"
    uploaded = subprocess.run(
        [str(fake_aws), "s3api", "put-object", "--bucket", "offline-source",
         "--key", source_key, "--body", str(archive), "--if-none-match", "*",
         "--checksum-algorithm", "SHA256", "--checksum-sha256", source_checksum,
         "--content-length", str(len(source_data)), "--content-type", "application/x-tar",
         "--server-side-encryption", "AES256", "--metadata", '{"object-kind":"exact-source"}'],
        check=False, capture_output=True, text=True,
    )
    assert uploaded.returncode == 0, uploaded.stderr
    source_version = json.loads(uploaded.stdout)["VersionId"]

    worker_aws = worker_root / "usr/local/bin/aws"
    worker_vast = worker_root / "usr/local/bin/vastai"
    worker_python = worker_root / "usr/local/bin/python3"
    fixed(worker_aws, fake_aws.read_bytes(), 0o755)
    fixed(worker_vast, fake_vast.read_bytes(), 0o755)
    fixed(worker_python, Path(sys.executable).read_bytes(), 0o755)
    inputs = {
        "schema": launch_packet.INPUT_SCHEMA,
        "worker_image": capsule["contract"]["production_runtime"]["image"],
        "worker_disk_gb": 80,
        "source_bundle": {"bucket": "offline-source", "key": source_key,
                          "version_id": source_version,
                          "sha256": hashlib.sha256(source_data).hexdigest(),
                          "bytes": len(source_data), "members": members},
        "runtime_image": {"image": capsule["contract"]["production_runtime"]["image"],
                          "receipt_sha256": hashlib.sha256(runtime_receipt).hexdigest()},
        "control_mounts": {
            "aws_config_sha256": hashlib.sha256(aws_config).hexdigest(),
            "aws_credentials_sha256": hashlib.sha256(aws_credentials).hexdigest(),
            "control_vast_key_sha256": hashlib.sha256(control_key).hexdigest(),
            "ssh_private_key_sha256": hashlib.sha256(ssh_key).hexdigest(),
        },
        "control_executables": {
            "aws_sha256": hashlib.sha256(fake_aws.read_bytes()).hexdigest(),
            "vastai_sha256": hashlib.sha256(fake_vast.read_bytes()).hexdigest(),
        },
        "worker_runtime_receipts": {
            "python_sha256": hashlib.sha256(worker_python.read_bytes()).hexdigest(),
            "aws_sha256": hashlib.sha256(worker_aws.read_bytes()).hexdigest(),
            "vast_sha256": hashlib.sha256(worker_vast.read_bytes()).hexdigest(),
        },
        "maximum_prestart_seconds": 120,
    }
    fixed(inputs_path, launch_packet.canonical_bytes(inputs))

    bootstrap_root = worker_root / "opt/parallax/future-freeze/source"
    entry_runtime_path = worker_root / "run/parallax/pg1-native-entry-contract.json"
    bootstrap_receipt = worker_root / "run/parallax/exact-bootstrap-receipt.json"
    worker_receipt = worker_root / "run/parallax/exact-worker-receipt.json"
    stage_invoker = stage_source / "infra/vast/parallax/pg1_native_launch_invoker.py"
    downstream = (
        root / "opt/parallax/control/source",
        packet_path,
        worker_root / "opt/parallax/image/source",
        bootstrap_root,
        entry_runtime_path,
        worker_root / "run/parallax/pg1-external-deadman-arm.json",
        worker_root / "run/parallax/pg1-launch-authority-binding.json",
    )
    assert all(not path.exists() for path in downstream)
    arbitrary_cwd = tmp_path / "arbitrary-cwd-before-install"
    arbitrary_cwd.mkdir()
    isolated_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    }
    helped = subprocess.run(
        [str(stage_invoker), "--help"], check=False, capture_output=True,
        text=True, timeout=30, cwd=arbitrary_cwd, env=isolated_env,
    )
    assert helped.returncode == 0, helped.stderr
    assert all(not path.exists() for path in downstream)
    invoked = subprocess.run(
        [str(stage_invoker), "--launch-future-native", "--capsule", str(capsule_path),
         "--inputs", str(inputs_path), "--receipt",
         str(root / "run/parallax/control-plane/pg1-control-layout-receipt.json"),
         "--offer-id", "424242", "--attempt-run-id", attempt,
         "--one-shot-nonce", "9" * 64],
        check=False, capture_output=True, text=True, timeout=30, cwd=arbitrary_cwd,
        env=isolated_env,
    )
    assert invoked.returncode == 0, invoked.stderr
    causal = json.loads(invoked.stdout)
    assert causal["status"] == "INSTALL_COMPILE_AUTHORITY_GATEWAY_EXECUTED"
    assert [step["name"] for step in causal["steps"]] == [
        "compile", "issue", "gateway",
    ]
    control_source = root / "opt/parallax/control/source"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert launch_packet.validate_packet(packet, capsule) == packet
    launch_result = causal["steps"][-1]["result"]
    assert launch_result["science_released"] is True
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (
        bootstrap_receipt.is_file() and worker_receipt.is_file()
    ):
        time.sleep(0.05)
    onstart_error = (services / "onstart.stderr").read_text(encoding="utf-8")
    assert bootstrap_receipt.is_file(), onstart_error
    assert worker_receipt.is_file(), onstart_error
    bootstrap_result = json.loads(bootstrap_receipt.read_text())
    worker_result = json.loads(worker_receipt.read_text())
    assert bootstrap_result["status"] == "EXACT_COPIED_BOOTSTRAP_BYTES_EXECUTED"
    assert bootstrap_result["worker_status"] == "EXACT_COPIED_WORKER_BYTES_EXECUTED"
    assert worker_result["status"] == "EXACT_COPIED_WORKER_BYTES_EXECUTED"
    assert bootstrap_result["provider_calls"] == worker_result["provider_calls"] == 0
    assert all(path.exists() for path in downstream)
    assert (worker_root / "run/parallax/pg1-native-installed-source-layout.json").is_file()
    assert (worker_root / "run/parallax/pg1-idle-bootstrap-pointer.json").is_file()
    gateway_source = (
        root / "opt/parallax/control/source/infra/vast/parallax/pg1_native_launch_gateway.py"
    ).read_text()
    gateway_tree = ast.parse(gateway_source)
    parser_options = {
        argument.value for call in ast.walk(gateway_tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        for argument in call.args if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
    }
    assert "--mechanical-contract" not in parser_options
    assert "fake_provider" not in gateway_source


@pytest.mark.parametrize(
    "failure",
    ("create_response_loss", "label_ambiguity", "label_absence", "context_bind",
     "mount_handoff", "bound_deadman_ack", "start_lease_publish"),
)
def test_gateway_post_create_failure_matrix_never_releases_science_or_orphans(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule(f"gateway-failure-{failure}-v1")
    packet = _gateway_packet(capsule)
    transactions: list[dict[str, Any]] = []
    cleanup: list[int] = []
    deadman_started: list[bool] = []
    authorization = _gateway_authorization(capsule, packet)
    monkeypatch.setattr(
        launch_gateway, "fetch_launch_authorization",
        lambda *_args: (authorization, {
            "version_id": "authorization-version", "etag": "authorization-etag",
            "checksum_sha256": "a" * 64,
            "metadata": launch_authority.metadata(authorization),
        }),
    )
    monkeypatch.setattr(launch_gateway, "consume_launch_authorization", lambda *_args: {})
    monkeypatch.setattr(launch_gateway, "fixed_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        launch_gateway, "_publish_state",
        lambda _packet, state, **_kwargs: (
            transactions.append(copy.deepcopy(state))
            or (state, {"etag": f"state-etag-{len(transactions)}"})
        ),
    )

    class Process:
        pass

    monkeypatch.setattr(
        launch_gateway.subprocess, "Popen",
        lambda *_args, **_kwargs: (deadman_started.append(True) or Process()),
    )
    ack_calls = 0

    def wait_ack(_path: Path, _packet: dict[str, Any], *, status: str) -> dict[str, Any]:
        nonlocal ack_calls
        ack_calls += 1
        if failure == "bound_deadman_ack" and ack_calls == 2:
            raise launch_gateway.LaunchGatewayError("planted bound ACK failure")
        return {"status": status, "packet_sha256": packet["content_address"]["digest"],
                "schema": "pg1-native-instance-bound-deadman-ack/v3",
                "issuer": control_deadman.ISSUER,
                "capsule_sha256": capsule["content_address"]["digest"],
                "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
                "instance_id": 778899, "machine_id": 771133,
                "container_id": "future-container-abc",
                "stop_context_sha256": "b" * 64, "destroy_authority": False,
                "etag": "ack-etag", "version_id": "ack-version"}

    monkeypatch.setattr(launch_gateway, "_wait_ack", wait_ack)

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "create" in command:
            if failure in {"create_response_loss", "label_ambiguity", "label_absence"}:
                return subprocess.CompletedProcess(command, 0, "{lost-response", "")
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"success": True, "new_contract": 778899,
                                        "instance_api_key": "1" * 64}), ""
            )
        if "show" in command:
            return subprocess.CompletedProcess(
                command, 0,
                json.dumps([{"id": 778899, "machine_id": 771133,
                             "container_id": "future-container-abc",
                             "label": packet["contract"]["launch_label"],
                             "actual_status": "running"}]), "",
            )
        if "copy" in command:
            return subprocess.CompletedProcess(
                command, 1 if failure == "start_lease_publish" else 0, "{}", ""
            )
        raise AssertionError(command)

    monkeypatch.setattr(launch_gateway, "_run", run)
    monkeypatch.setattr(
        launch_gateway, "_read_object", lambda **_kwargs: {"status": "ABSENT"}
    )
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    if failure == "label_ambiguity":
        monkeypatch.setattr(
            launch_gateway, "_discover",
            lambda _packet: (_ for _ in ()).throw(
                launch_gateway.LaunchGatewayError("planted ambiguous label")
            ),
        )
    elif failure == "label_absence":
        monkeypatch.setattr(launch_gateway, "_discover", lambda _packet: None)
    elif failure == "create_response_loss":
        monkeypatch.setattr(launch_gateway, "_discover", lambda _packet: 778899)
    monkeypatch.setattr(
        launch_gateway, "_publish_stop_context",
        lambda *_args: (
            (_ for _ in ()).throw(launch_gateway.LaunchGatewayError("planted context bind"))
            if failure == "context_bind" else _stop_context(capsule, instance_id=778899)
        ),
    )
    monkeypatch.setattr(
        launch_gateway, "_materialize_handoff",
        lambda *_args: ({Path("/run/parallax/fixed"): tmp_path / "fixed"},
                        {"schema": "offline-layout"}),
    )
    monkeypatch.setattr(
        launch_gateway, "_copy_all",
        lambda *_args: (
            (_ for _ in ()).throw(launch_gateway.LaunchGatewayError("planted handoff"))
            if failure == "mount_handoff" else None
        ),
    )
    monkeypatch.setattr(
        launch_gateway, "classify_start_commit",
        lambda *_args: {
            "classification": "START_COMMIT_PROVEN_ABSENT",
            "consecutive_absences": 3,
        },
    )
    monkeypatch.setattr(
        launch_gateway, "_cleanup",
        lambda instance_id, *_args, **_kwargs: (
            cleanup.append(instance_id)
            or {"status": "ORPHAN_STOPPED_SCIENCE_NEVER_RELEASED",
                "science_released": False}
        ),
    )
    with pytest.raises((launch_gateway.LaunchGatewayError, json.JSONDecodeError)):
        launch_gateway.launch(capsule, packet)
    assert deadman_started == [True]
    assert all(row["science_state"] == launch_state.SCIENCE_DISABLED for row in transactions)
    if failure in {"create_response_loss", "context_bind", "mount_handoff",
                   "bound_deadman_ack", "start_lease_publish"}:
        assert cleanup == [778899]
    else:
        assert cleanup == []


def test_gateway_cleanup_retries_stop_to_three_terminal_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("gateway-stop-cleanup-v1")
    packet = _gateway_packet(capsule)
    states = iter(("running", "stopped", "stopped", "stopped"))
    calls: list[str] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "stop" in command:
            calls.append("STOP"); return subprocess.CompletedProcess(command, 0, "{}", "")
        state = next(states); calls.append(f"SHOW:{state}")
        return subprocess.CompletedProcess(
            command, 0, json.dumps([{"id": 778899, "actual_status": state}]), ""
        )

    monkeypatch.setattr(launch_gateway, "_run", run)
    monkeypatch.setattr(
        launch_gateway, "_put_immutable_verified",
        lambda **_kwargs: {
            "version_id": "cleanup-version", "etag": "cleanup-etag",
            "checksum_sha256": "cleanup-checksum",
        },
    )
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    receipt = launch_gateway._cleanup(778899, packet, failure="PLANTED")
    assert receipt["status"] == "ORPHAN_STOPPED_SCIENCE_NEVER_RELEASED"
    assert receipt["consecutive_terminal_observations"] == 3
    assert receipt["science_released"] is False
    assert calls.count("STOP") == 4


def test_gateway_failure_after_start_release_can_only_stop_and_preserve_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("gateway-post-start-stop-only-v1")
    packet = _gateway_packet(capsule)
    publications: list[str] = []
    monkeypatch.setattr(
        launch_gateway, "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0,
            "{}" if "stop" in command else json.dumps(
                [{"id": 778899, "actual_status": "running"}]
            ), "",
        ),
    )
    monkeypatch.setattr(launch_gateway.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        launch_gateway, "_put_immutable_verified",
        lambda *, key, **_kwargs: (
            publications.append(key)
            or {"version_id": "v", "etag": "e", "checksum_sha256": "c"}
        ),
    )
    receipt = launch_gateway._cleanup(
        778899, packet, failure="PLANTED_AFTER_START_RELEASE",
        science_released=True,
    )
    assert receipt["status"] == "RELEASED_WORKER_STOP_UNCONFIRMED_SCRATCH_PRESERVED"
    assert receipt["science_released"] is True
    assert receipt["destroy_gate"] is None
    assert len(publications) == 1
    assert "/gateway-cleanup/" in publications[0]
    assert all("/destroy-requests/" not in key for key in publications)


def test_idle_bootstrap_validates_bound_materialization_and_fails_before_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("idle-bound-materialization-v1")
    packet = _gateway_packet(capsule)
    root = tmp_path / "worker"
    source_root = root / "opt/parallax/image/source"
    runtime_bytes = {
        "aws_config": b"[profile fuelos]\nregion=us-east-1\n",
        "aws_credentials": b"[fuelos]\ncredential_process=offline-fixture\n",
        "vast_api_key": b"1" * 64 + b"\n",
        "runtime_image": (capsule["contract"]["production_runtime"]["image"] + "\n").encode(),
    }
    context = _stop_context(capsule, instance_id=778899)
    for label, data in runtime_bytes.items():
        context["contract"]["runtime_receipts"][label]["sha256"] = hashlib.sha256(data).hexdigest()
    context["contract"]["offer_id"] = packet["contract"]["offer_id"]
    context["content_address"]["digest"] = capsule_module.content_sha256(context["contract"])
    entry = launch_gateway.production_entry_contract(capsule)
    state = launch_state.initial(
        packet, authorization_sha256="2" * 64,
        authorization_receipt_sha256="3" * 64,
    )
    state = launch_state.transition(state, "CREATE_INTENT_ARMED")
    state = launch_state.transition(
        state, "WORKER_ID_BOUND_SCIENCE_DISABLED", instance_id=778899,
    )
    state = launch_state.transition(
        state, "INSTANCE_CONTEXT_BOUND", machine_id=771133,
        container_id="future-container-abc",
        stop_context_sha256=context["content_address"]["digest"],
        mount_inventory_sha256=launch_gateway.prebound_mount_inventory_digest(
            capsule, packet, context,
            hashlib.sha256(runtime_bytes["vast_api_key"]).hexdigest(),
        ),
    )
    arm_path = root / "run/parallax/pg1-external-deadman-arm.json"
    monkeypatch.setattr(control_deadman, "BOUND_ACK", arm_path)
    monkeypatch.setattr(
        control_deadman, "_put",
        lambda **values: {
            "etag": "ack-etag", "version_id": "ack-version",
            "checksum_sha256": base64.b64encode(hashlib.sha256(
                control_deadman.canonical_bytes(values["payload"])
            ).digest()).decode("ascii"),
        },
    )
    arm = control_deadman._bind_ack(state)
    authority_binding = {
        "schema": "pg1-native-worker-authority-binding/v1",
        "packet_sha256": arm["packet_sha256"],
        "capsule_sha256": arm["capsule_sha256"],
        "attempt_run_id": arm["attempt_run_id"],
        "authorization_body_sha256": arm["authorization_body_sha256"],
        "authorization_receipt_sha256": arm["authorization_receipt_sha256"],
        "mount_inventory_sha256": arm["mount_inventory_sha256"],
    }
    path_values: dict[str, tuple[Path, bytes]] = {
        "CAPSULE": (root / "run/parallax/pg1-execution-capsule.json",
                    capsule_module.canonical_bytes(capsule)),
        "STOP_CONTEXT": (root / "run/parallax/pg1-native-stop-context.json",
                         capsule_module.canonical_bytes(context)),
        "ENTRY_CONTRACT": (root / "run/parallax/pg1-native-entry-contract.json",
                           capsule_module.canonical_bytes(entry)),
        "EXTERNAL_ARM": (arm_path,
                           capsule_module.canonical_bytes(arm)),
        "PACKET": (root / "run/parallax/pg1-future-launch-packet.json",
                   capsule_module.canonical_bytes(packet)),
        "AWS_CONFIG": (root / "run/parallax/aws/config", runtime_bytes["aws_config"]),
        "AWS_CREDENTIALS": (root / "run/parallax/aws/credentials",
                            runtime_bytes["aws_credentials"]),
        "VAST_KEY": (root / "run/secrets/pg1-native-vast-api-key",
                     runtime_bytes["vast_api_key"]),
        "RUNTIME_RECEIPT": (root / "run/parallax/runtime-image-digest",
                            runtime_bytes["runtime_image"]),
        "AUTHORITY_BINDING": (
            root / "run/parallax/pg1-launch-authority-binding.json",
            capsule_module.canonical_bytes(authority_binding),
        ),
    }
    mounts: dict[str, Any] = {}
    for constant, (path, data) in path_values.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.write_bytes(data)
        os.chmod(path, 0o400)
        monkeypatch.setattr(idle_bootstrap, constant, path)
        mounts[str(path)] = {"mode": "0400", "sha256": hashlib.sha256(data).hexdigest(),
                             "bytes": len(data)}
    for relative_text, expected in packet["contract"]["source_runtime_sha256"].items():
        target = source_root / relative_text
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_text, target)
        assert capsule_module.sha256_file(target) == expected
    monkeypatch.setattr(idle_bootstrap, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(idle_bootstrap, "PRODUCTION_ENTRY", capsule_module.PRODUCTION_ENTRY_PATH)
    layout_path = root / "run/parallax/pg1-native-handoff-layout.json"
    layout = {"schema": "pg1-native-bound-handoff-layout/v1",
              "packet_sha256": packet["content_address"]["digest"],
              "capsule_sha256": capsule["content_address"]["digest"],
              "instance_id": 778899,
              "source_bundle": packet["contract"]["frozen_inputs"]["source_bundle"],
              "source_runtime_sha256": packet["contract"]["source_runtime_sha256"],
              "mounts": mounts, "science_released": False}
    layout_path.write_bytes(capsule_module.canonical_bytes(layout)); os.chmod(layout_path, 0o400)
    monkeypatch.setattr(idle_bootstrap, "LAYOUT_RECEIPT", layout_path)
    start_path = root / "run/parallax/pg1-native-start-lease.json"
    installed_path = root / "run/parallax/pg1-native-installed-source-layout.json"
    installed = {
        "schema": "pg1-native-installed-source-layout/v1",
        "packet_sha256": packet["content_address"]["digest"],
        "source_bundle_sha256": packet["contract"]["frozen_inputs"]["source_bundle"]["sha256"],
        "members": packet["contract"]["frozen_inputs"]["source_bundle"]["members"],
        "science_released": False,
    }
    installed_path.write_bytes(capsule_module.canonical_bytes(installed))
    os.chmod(installed_path, 0o400)
    monkeypatch.setattr(idle_bootstrap, "INSTALLED_SOURCE", installed_path)
    start = {"schema": "pg1-native-bound-start-intent/v1",
             "status": "SCIENCE_DISABLED_WAITING_FOR_CANONICAL_START_COMMIT",
             "packet_sha256": packet["content_address"]["digest"],
             "capsule_sha256": capsule["content_address"]["digest"],
             "attempt_run_id": "idle-bound-materialization-v1", "instance_id": 778899,
             "stop_context_sha256": context["content_address"]["digest"],
             "deadman_ack_sha256": hashlib.sha256(capsule_module.canonical_bytes(arm)).hexdigest(),
             "handoff_layout_sha256": hashlib.sha256(capsule_module.canonical_bytes(layout)).hexdigest(),
             "canonical_state_sequence": 7, "canonical_state_sha256": "7" * 64,
             "start_commit_key": launch_state.start_commit_key(
                 packet["content_address"]["digest"]
             ),
             "start_commit_sha256": "8" * 64,
             "not_after_unix": time.time() + 30, "science_released": False}
    start_path.write_bytes(capsule_module.canonical_bytes(start)); os.chmod(start_path, 0o400)
    monkeypatch.setattr(idle_bootstrap, "START_LEASE", start_path)
    monkeypatch.setattr(idle_bootstrap, "_start_commit", lambda _intent: {"status": "PRESENT"})
    command = idle_bootstrap.validate_and_command(
        packet_sha256=packet["content_address"]["digest"],
        attempt_run_id="idle-bound-materialization-v1",
    )
    assert command[-len(capsule["contract"]["entrypoints"]["production_argv"]):] == (
        capsule["contract"]["entrypoints"]["production_argv"]
    )
    os.chmod(path_values["AWS_CREDENTIALS"][0], 0o600)
    with pytest.raises(idle_bootstrap.IdleBootstrapError, match="handoff file changed"):
        idle_bootstrap.validate_and_command(
            packet_sha256=packet["content_address"]["digest"],
            attempt_run_id="idle-bound-materialization-v1",
        )


def test_external_control_deadman_resets_three_observations_and_uses_no_worker_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule("external-control-common-mode-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="a" * 64)
    state = launch_state.transition(state, "CREATE_INTENT_ARMED")
    state = launch_state.transition(
        state, "WORKER_ID_BOUND_SCIENCE_DISABLED", instance_id=778899,
    )
    state = launch_state.transition(state, "CLEANUP_STOPPING")
    storage = {"etag": "cleanup-etag", "version_id": "cleanup-version"}
    monkeypatch.setattr(control_deadman, "_api_key", lambda: "control-plane-key")
    monkeypatch.setattr(control_deadman.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        control_deadman, "_persist_event", lambda *_args, **_kwargs: {}
    )
    states = iter(("stopped", "running", "stopped", "stopped", "stopped"))
    stop_calls: list[list[str]] = []
    monkeypatch.setattr(
        control_deadman, "_observe",
        lambda *_args: (next(states), {"id": 778899}),
    )
    monkeypatch.setattr(
        control_deadman, "_run",
        lambda command, **_kwargs: (
            stop_calls.append(command) or subprocess.CompletedProcess(command, 0, "{}", "")
        ),
    )
    monkeypatch.setattr(control_deadman, "_rest_mutation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        control_deadman, "_terminal_receipt",
        lambda current, current_storage, **values: (
            {"receipt_durable": True,
             "consecutive_terminal_observations": values["observations"], **values},
            current, current_storage,
        ),
    )
    result = control_deadman.supervise_action(
        capsule, state, storage, action="stop", destroy_gate=None
    )
    assert result["consecutive_terminal_observations"] == 3
    assert stop_calls
    assert all("control-plane-key" not in command for command in stop_calls)
    source = (ROOT / capsule_module.CONTROL_DEADMAN_PATH).read_text(encoding="utf-8")
    assert "pg1_native_provider" not in source
    assert "/run/secrets/pg1-native-vast-api-key" not in source
    assert "/run/secrets/pg1-native-control-plane-vast-api-key" in source


def test_external_control_deadman_stops_stale_science_worker_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("external-control-stale-worker-start-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="b" * 64)
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {
            "machine_id": 771133, "container_id": "future-container-abc",
            "stop_context_sha256": "c" * 64,
        }),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "d" * 64}),
        ("START_COMMIT_INTENT", {}),
    ):
        state = launch_state.transition(state, phase, **fields)
    state = launch_state.transition(
        state, "SCIENCE_RELEASED", science_state=launch_state.SCIENCE_RELEASED,
        start_commit_sha256="e" * 64,
    )
    reads = iter((
        {"status": "ABSENT"},  # no destroy request
        {"status": "ABSENT"},  # stale/missing science heartbeat
    ))
    monkeypatch.setattr(
        control_deadman, "_read_object", lambda _key: next(reads),
    )
    assert control_deadman._action_required(
        capsule, state, {"status": "PRESENT", "start_commit_sha256": "e" * 64}
    ) == ("stop", None)


def test_external_control_deadman_gated_destroy_survives_error_and_proves_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("external-control-destroy-gate-v1")
    packet = _gateway_packet(capsule)
    state = launch_state.initial(packet, authorization_sha256="8" * 64)
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": 778899}),
        ("INSTANCE_CONTEXT_BOUND", {
            "machine_id": 771133, "container_id": "future-container-abc",
            "stop_context_sha256": "7" * 64,
        }),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "6" * 64}),
        ("START_COMMIT_INTENT", {}),
    ):
        state = launch_state.transition(state, phase, **fields)
    released = launch_state.transition(
        state, "SCIENCE_RELEASED", science_state=launch_state.SCIENCE_RELEASED,
        start_commit_sha256="5" * 64,
    )
    request = {
        "schema": "pg1-native-external-destroy-request/v2",
        "status": "DURABLE_EXTERNAL_DESTROY_REQUEST",
        "capsule_sha256": released["capsule_sha256"],
        "attempt_run_id": released["attempt_run_id"],
        "instance_id": 778899, "destroy_gate": "COMPLETE_ARTIFACT_DURABILITY",
        "teardown_authorized": True, "science_released": True,
        "evidence_sha256": "9" * 64,
        "launch_state_sequence": released["sequence"],
        "launch_state_sha256": released["self_sha256"],
        "start_commit_sha256": "5" * 64,
    }
    start = {"status": "PRESENT", "start_commit_sha256": "5" * 64}
    request_reference = {
        "sha256": control_deadman.digest(request),
        "storage": "CANONICAL_STATE_RAW_BODY",
    }
    state = launch_state.transition(
        released, "CLEANUP_DESTROYING", destroy_request=request_reference,
        destroy_request_body=request,
        destroy_gate="COMPLETE_ARTIFACT_DURABILITY",
    )
    assert control_deadman._validate_destroy_request(
        request, state, start,
    ) == "COMPLETE_ARTIFACT_DURABILITY"
    storage = {"etag": "destroying-etag", "version_id": "destroying-version"}
    events: list[tuple[str, dict[str, Any]]] = []
    puts: list[tuple[str, dict[str, Any]]] = []
    memory: dict[str, dict[str, Any]] = {}
    cli: list[list[str]] = []
    rest: list[tuple[int, str]] = []
    observations = iter(("ERROR", "running", None, None, None))

    def observe(*_args: Any) -> tuple[str | None, dict[str, Any] | None]:
        state = next(observations)
        if state == "ERROR":
            raise control_deadman.ControlDeadmanError("planted control read failure")
        return state, None if state is None else {"id": 778899}

    monkeypatch.setattr(control_deadman, "_api_key", lambda: "control-plane-key")
    monkeypatch.setattr(control_deadman, "_observe", observe)
    monkeypatch.setattr(control_deadman.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(control_deadman, "TRANSACTION", tmp_path / "transaction.json")
    monkeypatch.setattr(control_deadman, "STATE_RECEIPT", tmp_path / "state-receipt.json")
    monkeypatch.setattr(
        control_deadman, "_read_object",
        lambda key: copy.deepcopy(memory.get(key, {"status": "ABSENT"})),
    )
    monkeypatch.setattr(
        control_deadman, "_persist_event",
        lambda _transaction, *, kind, sequence, payload: (
            events.append((kind, payload)) or {"sequence": sequence}
        ),
    )
    monkeypatch.setattr(
        control_deadman, "_run",
        lambda command, **_kwargs: (
            cli.append(command) or subprocess.CompletedProcess(command, 1, "", "planted")
        ),
    )
    monkeypatch.setattr(
        control_deadman, "_rest_mutation",
        lambda instance_id, _key, *, action: (
            rest.append((instance_id, action)) or {"http_status": 200}
        ),
    )
    def put_exact(*, key: str, payload: dict[str, Any],
                  metadata: dict[str, str], **_kwargs: Any) -> dict[str, Any]:
        puts.append((key, copy.deepcopy(payload)))
        memory[key] = _stored_object(
            payload, metadata, etag=f"etag-{len(puts)}",
            version_id=f"version-{len(puts)}", key=key,
        )
        return {
            field: memory[key][field]
            for field in ("etag", "version_id", "checksum_sha256")
        }

    monkeypatch.setattr(control_deadman, "_put", put_exact)
    result = control_deadman.supervise_action(
        capsule, state, storage, action="destroy",
        destroy_gate="COMPLETE_ARTIFACT_DURABILITY",
    )
    assert result["status"] == "TERMINAL_DESTROYED_DURABLE"
    assert result["receipt_durable"] is True
    assert result["consecutive_observations"] == 3
    assert result["errors"] > 1
    assert cli and rest
    assert all("control-plane-key" not in command for command in cli)
    assert all(action == "destroy" for _instance_id, action in rest)
    assert any(kind == "error" and payload["process_continues"] for kind, payload in events)
    assert puts[-1][1]["phase"] == "TERMINAL_DESTROYED_DURABLE"
    assert puts[-1][1]["terminal_receipt"]["sha256"] == control_deadman.digest(
        puts[-2][1]
    )


def test_lifecycle_multiprocess_lost_update_replay_and_terminal_absorption(
    tmp_path: Path,
) -> None:
    capsule = _capsule("lifecycle-process-cas-v1")
    context = _stop_context(capsule)
    path = tmp_path / "lifecycle.json"
    armed = lifecycle_module.transition(
        path, context, "ARMED", reason="TEST_ARM", expected_sequence=0
    )
    assert armed["sequence"] == 1
    process_context = multiprocessing.get_context("fork")
    queue = process_context.Queue()
    processes = [
        process_context.Process(
            target=_lifecycle_race_writer,
            args=(str(path), context, state, queue),
        )
        for state in ("RUNNING_HEARTBEAT", "FAILURE_STOP_REQUIRED")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=2), queue.get(timeout=2)]
    assert sorted(row[0] for row in outcomes) == ["error", "ok"]
    assert "sequence mismatch" in next(row[2] for row in outcomes if row[0] == "error")
    current = lifecycle_module.read(path, context)
    assert current is not None and current["sequence"] == 2
    if current["state"] == "RUNNING_HEARTBEAT":
        current = lifecycle_module.transition(
            path, context, "FAILURE_STOP_REQUIRED",
            reason="TEST_FAILURE", expected_sequence=2,
        )
    terminal = lifecycle_module.transition(
        path, context, "TERMINAL_STOPPED_PRESERVED",
        reason="TEST_TERMINAL", expected_sequence=current["sequence"],
        terminal_receipt={
            "receipt_durable": True, "terminal_state": "stopped",
            "consecutive_terminal_observations": 3,
        },
    )
    replay = lifecycle_module.transition(
        path, context, "TERMINAL_STOPPED_PRESERVED",
        reason="REPLAY", expected_sequence=terminal["sequence"],
        terminal_receipt={
            "receipt_durable": True, "terminal_state": "stopped",
            "consecutive_terminal_observations": 3,
        },
    )
    assert replay == terminal
    with pytest.raises(lifecycle_module.LifecycleCASFailure, match="forbidden"):
        lifecycle_module.transition(
            path, context, "FAILURE_STOP_REQUIRED",
            reason="PLANTED_TERMINAL_REGRESSION", expected_sequence=terminal["sequence"],
        )
    assert lifecycle_module.reconcile(terminal, copy.deepcopy(terminal)) == terminal
    split_brain = copy.deepcopy(terminal)
    split_brain["self_sha256"] = "f" * 64
    with pytest.raises(lifecycle_module.LifecycleCASFailure, match="split brain"):
        lifecycle_module.reconcile(terminal, split_brain)


def test_s3_conditions_conflict_reconcile_retry_and_terminal_no_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("durable-lease-conflict-v1")
    context = _stop_context(capsule)
    local_path = tmp_path / "local.json"
    armed = lifecycle_module.transition(
        local_path, context, "ARMED", reason="ARM", expected_sequence=0
    )
    running = lifecycle_module.transition(
        local_path, context, "RUNNING_HEARTBEAT",
        reason="RUNNING", expected_sequence=1,
    )
    durable_payload = {
        "schema": "pg1-native-external-deadman-lease/v2",
        "status": armed["state"],
        "version": armed["version"],
        "sequence": 2,
        "local_sequence": armed["sequence"],
        "capsule_sha256": context["contract"]["capsule_sha256"],
        "attempt_run_id": context["contract"]["attempt_run_id"],
        "instance_id": context["contract"]["instance_id"],
        "stop_context_sha256": context["content_address"]["digest"],
        "local_lifecycle_sha256": armed["self_sha256"],
        "local_lifecycle": armed,
    }
    assert deployment._reconcile_lease_conflict(
        context, running, durable_payload
    ) == running
    terminal_path = tmp_path / "terminal.json"
    lifecycle_module.transition(
        terminal_path, context, "ARMED", reason="ARM", expected_sequence=0
    )
    failed = lifecycle_module.transition(
        terminal_path, context, "FAILURE_STOP_REQUIRED",
        reason="FAIL", expected_sequence=1,
    )
    forked_terminal = lifecycle_module.transition(
        terminal_path, context, "TERMINAL_STOPPED_PRESERVED",
        reason="STOPPED", expected_sequence=failed["sequence"],
        terminal_receipt={
            "receipt_durable": True, "terminal_state": "stopped",
            "consecutive_terminal_observations": 3,
        },
    )
    forked_payload = {**durable_payload, "sequence": 4,
                      "local_sequence": forked_terminal["sequence"],
                      "local_lifecycle_sha256": forked_terminal["self_sha256"],
                      "local_lifecycle": forked_terminal,
                      "status": forked_terminal["state"]}
    with pytest.raises(deployment.DeploymentError, match="does not prove loser ancestry"):
        deployment._reconcile_lease_conflict(context, running, forked_payload)
    failed_descendant = lifecycle_module.transition(
        local_path, context, "FAILURE_STOP_REQUIRED", reason="FAIL", expected_sequence=2
    )
    terminal = lifecycle_module.transition(
        local_path, context, "TERMINAL_STOPPED_PRESERVED", reason="STOPPED",
        expected_sequence=failed_descendant["sequence"],
        terminal_receipt={"receipt_durable": True, "terminal_state": "stopped",
                          "consecutive_terminal_observations": 3},
    )
    terminal_payload = {**durable_payload, "sequence": 4,
                        "local_sequence": terminal["sequence"],
                        "local_lifecycle_sha256": terminal["self_sha256"],
                        "local_lifecycle": terminal, "status": terminal["state"]}
    assert deployment._reconcile_lease_conflict(
        context, running, terminal_payload
    ) == terminal

    arm_receipt = tmp_path / "arm.json"
    arm_receipt.write_text(json.dumps({"etag": "arm-etag"}), encoding="utf-8")
    os.chmod(arm_receipt, 0o400)
    monkeypatch.setattr(deployment, "EXTERNAL_ARM_RECEIPT_PATH", arm_receipt)
    monkeypatch.setattr(deployment, "LEASE_CAS_RECEIPT_PATH", tmp_path / "lease-cas.json")
    monkeypatch.setattr(
        deployment, "_read_durable_lease",
        lambda _context: (
            durable_payload,
            {"ETag": "durable-etag", "VersionId": "durable-version"},
        ),
    )
    puts: list[list[str]] = []

    def put(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        puts.append(command)
        if len(puts) == 1:
            return subprocess.CompletedProcess(command, 1, "", "conditional conflict")
        checksum = command[command.index("--checksum-sha256") + 1]
        return subprocess.CompletedProcess(
            command, 0,
            json.dumps(
                {"ChecksumSHA256": checksum, "ServerSideEncryption": "AES256",
                 "VersionId": "retry-version", "ETag": "retry-etag"}
            ), "",
        )

    monkeypatch.setattr(deployment.subprocess, "run", put)
    published = deployment._publish_lease(context, running)
    assert published["reconciled_conflict"] is True
    assert published["sequence"] == 3
    assert puts[0][puts[0].index("--if-match") + 1] == "arm-etag"
    assert puts[1][puts[1].index("--if-match") + 1] == "durable-etag"
    assert lifecycle_module.s3_put_cas_flags(prior_etag=None) == [
        "--if-none-match", "*"
    ]
    assert lifecycle_module.s3_put_cas_flags(prior_etag="prior-etag") == [
        "--if-match", "prior-etag"
    ]


def test_backend_emits_only_durable_external_destroy_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("backend-destroyed-lifecycle-v1")
    context = _stop_context(capsule)
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_module.transition(
        lifecycle_path, context, "ARMED", reason="ARMED", expected_sequence=0
    )
    lifecycle_module.transition(
        lifecycle_path, context, "RUNNING_HEARTBEAT", reason="RUNNING",
        expected_sequence=1,
    )
    monkeypatch.setattr(backend_module, "LIFECYCLE_PATH", lifecycle_path)
    evidence = {
        "schema": backend_module.TEARDOWN_SCHEMA,
        "status": "FINAL_DURABILITY_RECEIPT_DOWNLOADED_VERIFIED_FOR_TEARDOWN",
        "capsule_sha256": capsule["content_address"]["digest"],
        "attempt_run_id": capsule["contract"]["attempt_identity"]["attempt_run_id"],
        "final_receipt_sha256": "a" * 64,
        "receipt_durability_sha256": "b" * 64,
        "receipt_verification_sha256": "c" * 64,
        "teardown_authorized": True,
    }
    backend = backend_module.ProductionBackend(
        capsule=capsule, instance_id=context["contract"]["instance_id"], context=context
    )
    packet = _gateway_packet(capsule)
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(capsule_module.canonical_bytes(packet)); os.chmod(packet_path, 0o400)
    monkeypatch.setattr(backend_module, "LAUNCH_PACKET_PATH", packet_path)
    state = launch_state.initial(packet, authorization_sha256="7" * 64)
    for phase, fields in (
        ("CREATE_INTENT_ARMED", {}),
        ("WORKER_ID_BOUND_SCIENCE_DISABLED", {"instance_id": context["contract"]["instance_id"]}),
        ("INSTANCE_CONTEXT_BOUND", {"machine_id": context["contract"]["machine_id"],
                                    "container_id": context["contract"]["container_id"],
                                    "stop_context_sha256": context["content_address"]["digest"]}),
        ("HANDOFF_BOUND_DEADMAN_ACKED", {"handoff_layout_sha256": "8" * 64}),
        ("START_COMMIT_INTENT", {}),
    ):
        state = launch_state.transition(state, phase, **fields)
    state = launch_state.transition(
        state, "SCIENCE_RELEASED", science_state=launch_state.SCIENCE_RELEASED,
        start_commit_sha256="9" * 64,
    )
    state_data = launch_state.canonical_bytes(state)
    state_checksum = base64.b64encode(hashlib.sha256(state_data).digest()).decode("ascii")
    state_metadata = control_deadman._state_metadata(state)
    stored_state: dict[str, Any] = {
        "data": state_data, "checksum": state_checksum, "metadata": state_metadata,
        "version": "state-version", "etag": "state-etag",
    }

    def run_json(command: list[str]) -> dict[str, Any]:
        if "head-object" in command:
            return {
                "VersionId": stored_state["version"], "ETag": stored_state["etag"],
                "Metadata": stored_state["metadata"],
                "ServerSideEncryption": "AES256",
                "ContentLength": len(stored_state["data"]),
                "ContentType": "application/json",
            }
        if "get-object" in command:
            Path(command[-1]).write_bytes(stored_state["data"])
            return {
                "VersionId": stored_state["version"], "ETag": stored_state["etag"],
                "ChecksumSHA256": stored_state["checksum"],
                "Metadata": stored_state["metadata"],
                "ServerSideEncryption": "AES256",
                "ContentLength": len(stored_state["data"]),
                "ContentType": "application/json",
            }
        stored_state.update({
            "data": Path(command[command.index("--body") + 1]).read_bytes(),
            "checksum": command[command.index("--checksum-sha256") + 1],
            "metadata": json.loads(command[command.index("--metadata") + 1]),
            "version": "destroy-state-version", "etag": "destroy-state-etag",
        })
        return {
            "ChecksumSHA256": stored_state["checksum"],
            "ServerSideEncryption": "AES256",
            "VersionId": stored_state["version"], "ETag": stored_state["etag"],
        }

    monkeypatch.setattr(backend, "_run_json", run_json)
    result = backend.request_destroy(evidence=evidence)
    assert result["status"] == "DURABLE_EXTERNAL_DESTROY_REQUEST"
    assert result["schema"] == "pg1-native-external-destroy-request/v2"
    assert json.loads(stored_state["data"])["phase"] == "CLEANUP_DESTROYING"
    assert result["worker_provider_mutations"] == 0
    assert result["worker_absence_polls"] == 0
    assert lifecycle_module.read(lifecycle_path, context)["state"] == "RUNNING_HEARTBEAT"
    source = inspect.getsource(backend_module.ProductionBackend.request_destroy)
    assert "vast_destroy" not in source and "wait_vast_terminal" not in source


def test_external_deadman_cli_failures_add_rest_and_never_exit_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _capsule("external-deadman-retry-v1")
    context = _stop_context(capsule)
    clock = _GuardianClock()
    observations = [
        {"state": "running"}, {"state": "running"}, {"state": "running"},
        {"state": "running"}, {"state": "running"}, {"state": "stopped"},
    ]

    class Provider:
        @staticmethod
        def load_vast_key(_context: dict[str, Any]) -> str:
            return "scoped"

        @staticmethod
        def observe_vast_instance(_context: dict[str, Any], _key: str) -> dict[str, Any]:
            return observations.pop(0)

        @staticmethod
        def vast_stop_command(_instance_id: int, _key: str) -> list[str]:
            return ["vastai", "stop"]

        @staticmethod
        def run_vast_mutation(_command: list[str]) -> None:
            raise CapsuleError("planted CLI STOP failure")

    rest_calls: list[tuple[int, str]] = []
    monkeypatch.setattr(deadman, "_probe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deadman, "_local_failure_required", lambda _context: True)
    monkeypatch.setattr(deadman, "_heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        deadman.importlib, "import_module", lambda _name: Provider
    )
    monkeypatch.setattr(deadman.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(deadman.time, "sleep", clock.sleep)
    monkeypatch.setattr(deadman, "ALTERNATE_STOP_AFTER_SECONDS", 30)
    monkeypatch.setattr(
        deadman, "_rest_stop",
        lambda instance_id, key: (
            rest_calls.append((instance_id, key)) or {"http_status": 200}
        ),
    )
    monkeypatch.setattr(
        deadman, "_put_receipt",
        lambda _context, payload: {**payload, "receipt_durable": True},
    )
    result = deadman.run_deadman(context)
    assert result["terminal_state"] == "stopped"
    assert result["receipt_durable"] is True
    assert rest_calls
    assert result.get("terminal_state") != "unverified"


def test_bootstrap_has_no_project_import_before_pointer_and_import_failure_is_supervised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (ROOT / capsule_module.BOOTSTRAP_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and isinstance(node.module, str)
            and node.module.startswith("infra")
        )
        for node in top_imports
    )
    capsule = _capsule("bootstrap-import-failure-v1")
    context = _stop_context(capsule)
    events: list[str] = []
    monkeypatch.setattr(
        bootstrap, "_entry_contract",
        lambda _path: launch_gateway.production_entry_contract(capsule),
    )
    monkeypatch.setattr(
        bootstrap, "__file__", capsule["contract"]["entrypoints"]["bootstrap_argv"][0]
    )

    class Process:
        def poll(self) -> None:
            return None

    monkeypatch.setattr(bootstrap, "_load_minimal_context", lambda _path: context)
    monkeypatch.setattr(bootstrap, "_read_lifecycle", lambda _context: None)
    monkeypatch.setattr(
        bootstrap,
        "_transition_lifecycle",
        lambda _context, state, **_kwargs: (events.append(state) or {"state": state}),
    )
    monkeypatch.setattr(
        bootstrap, "_pointer_first", lambda _context, _failure: events.append("pointer")
    )
    monkeypatch.setattr(
        bootstrap, "_start_guardian", lambda: (events.append("guardian") or Process())
    )
    monkeypatch.setattr(bootstrap, "_load_minimal_capsule", lambda _path: capsule)
    monkeypatch.setattr(
        bootstrap,
        "deploy",
        lambda **_kwargs: (_ for _ in ()).throw(ImportError("planted import failure")),
    )
    monkeypatch.setattr(
        bootstrap,
        "_publish_marker",
        lambda _context, _failure: (events.append("marker") or {"status": "durable"}),
    )
    monkeypatch.setattr(
        bootstrap,
        "_supervise_guardian",
        lambda *_args, **_kwargs: (
            events.append("supervised")
            or {"terminal_state": "stopped", "receipt_durable": True}
        ),
    )
    with pytest.raises(ImportError):
        bootstrap.main(
            [
                "--deploy",
                "--capsule",
                str(capsule_module.CAPSULE_RUNTIME_PATH),
                    "--stop-context",
                    str(capsule_module.STOP_CONTEXT_RUNTIME_PATH),
                    "--entry-contract",
                    str(capsule_module.ENTRY_CONTRACT_RUNTIME_PATH),
            ]
        )
    assert events[:4] == ["ARMED", "RUNNING_HEARTBEAT", "guardian", "pointer"]
    assert events[-2:] == ["marker", "supervised"]


@pytest.mark.parametrize("failure", ("guardian_spawn", "argv"))
def test_bootstrap_guardian_spawn_and_argv_failures_transition_to_stop(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule(f"bootstrap-{failure}-failure-v1")
    context = _stop_context(capsule)
    events: list[str] = []
    monkeypatch.setattr(
        bootstrap, "_entry_contract",
        lambda _path: launch_gateway.production_entry_contract(capsule),
    )
    monkeypatch.setattr(
        bootstrap, "__file__", capsule["contract"]["entrypoints"]["bootstrap_argv"][0]
    )

    class Process:
        def poll(self) -> None:
            return None

    monkeypatch.setattr(bootstrap, "_load_minimal_context", lambda _path: context)
    monkeypatch.setattr(bootstrap, "_read_lifecycle", lambda _context: None)
    monkeypatch.setattr(
        bootstrap,
        "_transition_lifecycle",
        lambda _context, state, **_kwargs: (events.append(state) or {"state": state}),
    )
    starts = 0

    def start() -> Process:
        nonlocal starts
        starts += 1
        if failure == "guardian_spawn" and starts == 1:
            raise RuntimeError("planted guardian spawn failure")
        return Process()

    monkeypatch.setattr(bootstrap, "_start_guardian", start)
    monkeypatch.setattr(
        bootstrap, "_pointer_first", lambda _context, _failure: events.append("pointer")
    )
    monkeypatch.setattr(
        bootstrap,
        "_publish_marker",
        lambda _context, _failure: (events.append("marker") or {"status": "durable"}),
    )
    monkeypatch.setattr(
        bootstrap,
        "_supervise_guardian",
        lambda *_args, **_kwargs: (
            events.append("supervised")
            or {"terminal_state": "stopped", "receipt_durable": True}
        ),
    )
    argv = [
        "--deploy", "--entry-contract", str(capsule_module.ENTRY_CONTRACT_RUNTIME_PATH)
    ] if failure == "argv" else [
        "--deploy", "--capsule", str(capsule_module.CAPSULE_RUNTIME_PATH),
        "--stop-context", str(capsule_module.STOP_CONTEXT_RUNTIME_PATH),
        "--entry-contract", str(capsule_module.ENTRY_CONTRACT_RUNTIME_PATH),
    ]
    with pytest.raises((RuntimeError, bootstrap.BootstrapError)):
        bootstrap.main(argv)
    assert events[:2] == ["ARMED", "RUNNING_HEARTBEAT"]
    assert events[-3:] == ["pointer", "marker", "supervised"]


def test_production_provisioner_has_fixed_layout_and_no_mechanical_selector() -> None:
    capsule = _capsule("provisioner-fixed-layout-v1")
    runtime = capsule["contract"]["production_runtime"]["deployment"]
    assert runtime["actual_future_entrypoint"] == capsule["contract"]["entrypoints"][
        "production_argv"
    ]
    assert runtime["bootstrap_argv"] == deployment.bootstrap_command(capsule)
    assert runtime["fixed_layout_modes"] == {
        "/opt/parallax/future-freeze": "0700",
        "/run/parallax": "0700",
        "/run/parallax/aws": "0700",
        "/run/secrets": "0700",
        "/workspace/parallax": "0700",
    }
    assert runtime["fixed_mount_modes"] == {
        str(path): f"{mode:04o}" for path, mode in deployment.FIXED_MOUNTS.items()
    }
    source = (ROOT / capsule_module.DEPLOYMENT_PATH).read_text(encoding="utf-8")
    assert "pg1_native_rehearsal" not in source
    assert "mechanical" not in inspect.signature(deployment.main).parameters
    assert "--rehearsal" not in source


@pytest.mark.parametrize(
    "failure",
    (
        "guardian_transient_key",
        "guardian_transient_show_timeout",
        "guardian_transient_nonzero",
        "guardian_transient_malformed",
        "guardian_transient_parse",
        "guardian_transient_rate_limit",
        "guardian_wrong_row",
        "bootstrap_import_failure",
    ),
)
def test_mechanical_transient_and_import_plants_recover_through_guardian(
    failure: str, tmp_path: Path
) -> None:
    completed, root, _capsule_value = _run_mechanical(
        tmp_path, attempt_id=f"plant-{failure}-v1", failure=failure
    )
    assert completed.returncode != 0
    result = json.loads((root / "rehearsal-result.json").read_text(encoding="utf-8"))
    assert result["stop"]["status"] == "GUARDIAN_STOP_VERIFIED_TERMINAL"
    events = [
        json.loads(line)["event"]
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "guardian_marker_durable" in events
    assert events[-1] == "guardian_stop_verified_terminal"


def test_mechanical_guardian_deadline_and_repeated_absence_plants(
    tmp_path: Path,
) -> None:
    deadline, deadline_root, _ = _run_mechanical(
        tmp_path,
        attempt_id="plant-guardian-deadline-v1",
        failure="guardian_deadline_fail_closed",
    )
    assert deadline.returncode != 0
    deadline_result = json.loads(
        (deadline_root / "rehearsal-result.json").read_text(encoding="utf-8")
    )
    assert deadline_result["stop"] == {
        "status": "GUARDIAN_DEADLINE_EXHAUSTED_FAIL_CLOSED",
        "terminal_state": "unverified",
        "receipt_durable": True,
    }
    absent, absent_root, _ = _run_mechanical(
        tmp_path,
        attempt_id="plant-guardian-absence-v1",
        failure="guardian_repeated_absence",
    )
    assert absent.returncode != 0
    absent_result = json.loads(
        (absent_root / "rehearsal-result.json").read_text(encoding="utf-8")
    )
    assert absent_result["stop"]["status"] == (
        "GUARDIAN_DESTROY_VERIFIED_REPEATED_ABSENCE"
    )
    assert absent_result["stop"]["consecutive_absences"] == 3
