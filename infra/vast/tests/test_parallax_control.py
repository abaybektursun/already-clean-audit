from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from infra.vast.parallax.artifacts import ArtifactManifest
from infra.vast.parallax.dag import SprintDAG
from infra.vast.parallax.launch_packet import (
    LaunchPacketError,
    canonical_sha256,
    validate_launch_packet,
)
from infra.vast.parallax.offers import H100Policy, rank_h100_offers, rank_volume_bundles
from infra.vast.parallax.genealogy_audit import _same_except
from infra.vast.parallax.image_smoke import expected_versions
from infra.vast.parallax.storage_rehearsal import (
    GO_PHRASE,
    _contract_id,
    canonical_public_key,
    parse_ssh_url,
    parse_vast_output,
)
from infra.vast.parallax.runtime_bundle import activate
from infra.vast.parallax.parallax_adjudicate import graft_world, operator_world
from infra.vast.parallax.parallax_entities import frozen_entities
from infra.vast.parallax.parallax_operator import (
    bilinear_symmetry_error,
    central_directional_derivative,
    linearity_error,
)
from infra.vast.parallax.parallax_tasks import ConfirmationFirewallError, FrozenTaskBank
from infra.vast.parallax.parallax_towers import key_role, replacement_plan
from infra.vast.parallax.preflight import MAX_PREFLIGHT_MINUTES
from infra.vast.parallax.download_inventory import _required
from infra.vast.parallax.populate_cache import ACK, populate
from infra.vast.parallax.hydrate_h100_cache import (
    ACK as HYDRATE_ACK,
    hydrate,
)
from infra.vast.parallax.cache_gate import CACHE_GATE_MARKER
from infra.vast.parallax.preflight import validate_remote_receipts
from infra.vast.parallax.transfer_receipt import (
    REQUIRED_TRANSFER_SCOPE,
    build_manifest as build_transfer_manifest,
    verify_inbound,
)
from infra.vast.parallax.genealogy_weights import (
    canonical_storage_equal,
    displacement_metrics,
    displacement_sums,
    triadic_displacement_metrics,
    triadic_displacement_sums,
    gemma_to_paligemma,
    paligemma_to_pi,
    siglip_to_paligemma,
)


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "studies/lineage/cross-lineage/config/parallax_matrix.yaml"
TEMPLATE = ROOT / "studies/lineage/cross-lineage/config/parallax_launch_packet.template.yaml"
SPRINT_V2 = ROOT / "studies/lineage/cross-lineage/config/parallax_truth_sprint_v2.yaml"


def test_template_refuses_launch() -> None:
    with pytest.raises(LaunchPacketError, match="launch refused"):
        validate_launch_packet(TEMPLATE, repo_root=ROOT, verify_evidence=False)


def test_manifest_detects_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "result.json"
    artifact.write_text('{"answer": 42}\n', encoding="utf-8")
    manifest = ArtifactManifest(source, "fixture")
    manifest.add(artifact)
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    assert ArtifactManifest.verify(manifest_path, source) == []
    artifact.write_text('{"answer": 43}\n', encoding="utf-8")
    assert ArtifactManifest.verify(manifest_path, source) == ["sha256 mismatch: result.json"]


def test_banked_h100_packet_matches_its_durable_manifest() -> None:
    packet = ROOT / "studies/lineage/cross-lineage/results/readiness/h100_preflight_45049292"
    assert ArtifactManifest.verify(packet / "manifest.json", packet) == []
    preflight = json.loads((packet / "h100-preflight.json").read_text(encoding="utf-8"))
    hydration = json.loads((packet / "cache-hydration.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "GREEN"
    assert preflight["elapsed_seconds"] < 60
    assert hydration["verified_bytes"] == 71_411_428_681


def test_truth_sprint_is_contiguous_and_refuses_hour_six_without_packet(tmp_path: Path) -> None:
    dag = SprintDAG(MATRIX, tmp_path / "state.json")
    assert sum(stage.budget_hours for stage in dag.stages) == 72
    state = dag.load()
    for stage in dag.stages[:2]:
        state["stages"][stage.stage_id]["status"] = "succeeded"
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="launch packet required"):
        dag.start(dag.stages[2].stage_id)


def test_execution_v2_frontloads_decisions_without_relaxing_gates() -> None:
    sprint = yaml.safe_load(SPRINT_V2.read_text(encoding="utf-8"))
    stages = sprint["stages"]
    assert stages[0]["hours"] == [0.0, 0.5]
    assert stages[-1]["hours"] == [68.0, 72.0]
    for left, right in zip(stages, stages[1:]):
        assert float(left["hours"][1]) == float(right["hours"][0])
    assert sprint["earliest_decisive_targets"] == {
        "paired_g1_mechanical_definition_by_hour": 6,
        "shared_siglip_discovery_by_hour": 14,
        "pg1_vs_pg2_behavioral_boundary_by_hour": 18,
        "sealed_graft_world_by_hour": 22,
        "operator_scalar_decision_by_hour": 34,
    }
    assert "science_before_both_g1_packets" in sprint["hard_rules"]
    assert "continue_after_hour_6_without_content_hashed_go_packet" in sprint["hard_rules"]
    assert "connector_repair_before_zero_shot_graft_range" in sprint["hard_rules"]
    assert "broad_training_grid" in sprint["hard_rules"]


def test_cheapest_suitable_offer_uses_measured_scratch_not_fixed_three_tb() -> None:
    offers = [
        {
            "id": 1,
            "gpu_name": "H100 NVL",
            "num_gpus": 1,
            "gpu_ram": 95.8,
            "reliability": 0.99,
            "disk_space": 2802.5,
            "duration": 10,
            "verified": True,
            "rentable": True,
            "rented": False,
            "dph_base": 2.0,
            "storage_cost": 0.1,
        },
        {
            "id": 2,
            "gpu_name": "H100 PCIE",
            "num_gpus": 1,
            "gpu_ram": 81.6,
            "reliability": 0.997,
            "disk_space": 23_000,
            "duration": 10,
            "verified": True,
            "rentable": True,
            "rented": False,
            "dph_base": 2.2,
            "storage_cost": 0.2,
        },
    ]
    ranked = rank_h100_offers(offers, H100Policy())
    assert [offer["id"] for offer in ranked] == [1, 2]


def test_valid_packet_hash_round_trip(tmp_path: Path) -> None:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    # This test exercises canonicalization without pretending the template is GO.
    payload["review"]["packet_sha256"] = canonical_sha256(payload)
    assert payload["review"]["packet_sha256"] == canonical_sha256(payload)


def test_parallax_stack_does_not_import_old_mlops() -> None:
    package = ROOT / "infra/vast/parallax"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package.glob("*.py")
    )
    assert "infra.image_base" not in text
    assert "infra.aws_pg" not in text
    assert "modal." not in text


def test_shell_comparison_ignores_only_registered_vocab_expansion() -> None:
    green, differences = _same_except(
        {"hidden_size": 2048, "vocab_size": 256000},
        {"hidden_size": 2048, "vocab_size": 257216},
        {"vocab_size"},
    )
    assert green and differences == {}
    red, differences = _same_except(
        {"hidden_size": 2048, "vocab_size": 256000},
        {"hidden_size": 2304, "vocab_size": 257216},
        {"vocab_size"},
    )
    assert not red and differences == {"hidden_size": [2048, 2304]}


def test_inaccessible_volume_can_never_win() -> None:
    bundles = [
        {"volume": {"id": 1, "storage_cost": 0.01, "reliability": 0.999}},
        {
            "volume": {"id": 2, "storage_cost": 0.10, "reliability": 0.99},
            "gateway": {"id": 20, "dph_total": 0.20},
        },
    ]
    ranked = rank_volume_bundles(bundles, volume_gb=4000)
    assert [bundle["volume"]["id"] for bundle in ranked] == [2]


def test_environment_lock_is_exact_and_parallax_only() -> None:
    lock = ROOT / "infra/vast/parallax/requirements.lock"
    versions = expected_versions(lock)
    assert versions["torch"] == "2.13.0"
    assert versions["transformers"] == "5.5.4"
    assert "nnsight" not in versions
    assert "wandb" not in versions
    assert "dictionary_learning" not in versions


def test_pi_lock_parser_accepts_only_pinned_index_git_and_cuda_build() -> None:
    versions = expected_versions(ROOT / "infra/vast/parallax/requirements_pi.lock")
    assert versions["torch"] == "2.11.0"
    assert versions["transformers"] == "5.5.4"
    assert "lerobot" not in versions


def test_storage_rehearsal_requires_exact_go_phrase() -> None:
    assert GO_PHRASE == "REHEARSE_15GB"


def test_vast_mutation_parser_handles_documented_cli_forms() -> None:
    assert parse_vast_output(
        read_only=False,
        stdout="Created. {'success': True, 'new_contract': 123}\n",
        stderr="",
    ) == {"success": True, "new_contract": 123}
    assert parse_vast_output(
        read_only=False,
        stdout="Started. {'success': True, 'new_contract': 456}\nInstance launched successfully: 456\n",
        stderr="",
    ) == {"success": True, "new_contract": 456}
    with pytest.raises(RuntimeError, match="failed"):
        parse_vast_output(read_only=False, stdout="failed with error 404", stderr="")


def test_volume_contract_id_parses_vast_volume_name() -> None:
    assert _contract_id({"success": True, "volume_name": "V.45040105"}, "volume_name") == 45040105


def test_ssh_endpoint_uses_vast_ssh_url_not_show_instance_proxy() -> None:
    assert parse_ssh_url("ssh://root@141.195.21.72:41652\n") == {
        "ssh_user": "root",
        "ssh_host": "141.195.21.72",
        "ssh_port": 41652,
    }


def test_public_key_matching_ignores_comment_only() -> None:
    first = "ssh-ed25519 AAAATEST researcher@host"
    second = "ssh-ed25519 AAAATEST another-comment"
    assert canonical_public_key(first) == canonical_public_key(second)


def test_runtime_activation_refuses_wrong_hash(tmp_path: Path) -> None:
    bundle = tmp_path / "runtime.tar.gz"
    bundle.write_bytes(b"not a runtime")
    with pytest.raises(ValueError, match="SHA mismatch"):
        activate(bundle, "0" * 64, tmp_path / "target")


def test_pi_sidecar_is_pinned_and_firewalled_from_core() -> None:
    core = (ROOT / "infra/vast/parallax/requirements.lock").read_text(encoding="utf-8")
    sidecar = (ROOT / "infra/vast/parallax/requirements_pi.lock").read_text(encoding="utf-8")
    assert "lerobot" not in core
    assert "torch==2.13.0" in core
    assert "torch==2.11.0" in sidecar
    assert "3f2179f3b69708b6ad009b2e7685dd9d05269ee1" in sidecar
    assert "transformers==5.5.4" in sidecar
    smoke = (ROOT / "infra/vast/parallax/pi_sidecar_smoke.py").read_text(encoding="utf-8")
    assert "PI0Policy.config_class is PI0Config" in smoke
    assert "PI05Policy.config_class is PI05Config" in smoke
    assert "core_runtime_imported" in smoke
    freeze = json.loads(
        (ROOT / "studies/lineage/cross-lineage/config/parallax_pi_sidecar_freeze_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze["firewall"]["pg2_may_import_pi_sidecar"] is False
    assert freeze["firewall"]["pi_may_import_core_site_packages"] is False


def test_frozen_entities_include_both_parent_fusion_jumps_and_later_models() -> None:
    entities = frozen_entities(MATRIX)
    roles = {(row.lineage, row.role, row.repo) for row in entities}
    assert ("pg1_pi", "vision_parent_candidate", "google/siglip-so400m-patch14-224") in roles
    assert ("pg1_pi", "language_parent", "google/gemma-2b") in roles
    assert ("pg1_pi", "fused", "google/paligemma-3b-pt-224") in roles
    assert ("pg1_pi", "later_0", "lerobot/pi0_base") in roles
    assert ("pg2", "vision_parent_candidate", "google/siglip-so400m-patch14-224") in roles
    assert ("pg2", "language_parent", "google/gemma-2-2b") in roles
    assert ("pg2", "fused", "google/paligemma2-3b-pt-224") in roles


def test_confirmation_bank_is_unreadable_without_bound_arm_freeze(tmp_path: Path) -> None:
    manifest = tmp_path / "items.jsonl"
    manifest.write_text(
        '\n'.join([
            json.dumps({"split": "train", "task_id": "t", "item_id": "train"}),
            json.dumps({"split": "confirmation_eval", "task_id": "t", "item_id": "sealed"}),
        ]) + '\n',
        encoding="utf-8",
    )
    seal = tmp_path / "confirmation.sha256"
    seal.write_text("sealed-hash\n", encoding="utf-8")
    bank = FrozenTaskBank(manifest, seal)
    assert [row["item_id"] for row in bank.iter_discovery()] == ["train"]
    with pytest.raises(ConfirmationFirewallError):
        list(bank.iter_confirmation(arm_freeze=None))


def test_planted_operator_identities() -> None:
    matrix = ((2.0, 1.0), (1.0, 3.0))
    apply = lambda vector: [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
    assert bilinear_symmetry_error(apply, [1.0, 2.0], [-2.0, 0.5]) < 1e-12
    assert linearity_error(apply, [1.0, 2.0], [-2.0, 0.5], 0.7) < 1e-12
    quadratic = lambda point: point[0] ** 2 + 3.0 * point[1] ** 2
    observed = central_directional_derivative(quadratic, [2.0, -1.0], [0.5, 2.0], 1e-5)
    assert observed == pytest.approx(-10.0, rel=1e-9)


def test_tower_replacement_never_crosses_fixed_interface() -> None:
    shapes = {
        "vision_tower.encoder.weight": (2, 2),
        "multi_modal_projector.weight": (2, 2),
        "language_model.layers.0.weight": (2, 2),
    }
    assert key_role("multi_modal_projector.weight") == "fixed_interface"
    plan = replacement_plan(shapes, shapes, role="vision")
    assert plan == {"vision_tower.encoder.weight": "replace"}


def test_all_planted_adjudicator_branches() -> None:
    assert graft_world(native_effect=2, graft_effect=2, repaired_effect=0, mde=1) == "PORTABLE"
    assert graft_world(native_effect=2, graft_effect=0, repaired_effect=2, mde=1) == "REMAPPABLE"
    assert graft_world(native_effect=2, graft_effect=0, repaired_effect=0, mde=1) == "CO_ADAPTED_OR_BRIDGE_DISABLED"
    assert graft_world(native_effect=0, graft_effect=0, repaired_effect=0, mde=1) == "VOID_NO_NATIVE_RANGE"
    assert operator_world(scalar_green=True, replay_green=True, behavioral_range=True) == "VALID"
    assert operator_world(scalar_green=False, replay_green=True, behavioral_range=True) == "REDESIGN"
    assert operator_world(scalar_green=True, replay_green=False, behavioral_range=True) == "SITE_LOCAL_VOID"
    assert operator_world(scalar_green=True, replay_green=True, behavioral_range=False) == "VOID_NO_DIRECTIONAL_RANGE"


def test_preflight_is_no_science_and_hard_capped() -> None:
    assert MAX_PREFLIGHT_MINUTES == 30
    text = (ROOT / "infra/vast/parallax/preflight.py").read_text(encoding="utf-8")
    assert "NO_SCIENCE_PREFLIGHT" in text
    assert "huggingface_hub" not in text
    assert "snapshot_download" not in text
    assert "REMOTE_VAST_GATEWAY_RECEIPT" in text
    assert "local_cache_sha256_verification" in text


def test_download_inventory_excludes_duplicate_weight_formats() -> None:
    assert _required("model-00001-of-00002.safetensors")
    assert _required("tokenizer.json")
    assert not _required("gemma-2b.gguf")
    assert not _required("pytorch_model.bin")


def test_model_cache_population_requires_gateway_acknowledgement(tmp_path: Path) -> None:
    assert ACK == "POPULATE_DURABLE_CACHE_ON_GATEWAY"
    with pytest.raises(PermissionError, match="refusing cache mutation"):
        populate(tmp_path / "missing.json", tmp_path / "cache", acknowledge="no")


def test_h100_hydration_refuses_after_cache_gate(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / CACHE_GATE_MARKER).write_text('{"status":"GREEN"}\n', encoding="utf-8")
    with pytest.raises(PermissionError, match="cache gate already exists"):
        hydrate(
            tmp_path / "unused.json",
            cache,
            destination_instance_id=123,
            acknowledge=HYDRATE_ACK,
        )


def test_h100_hydration_verifies_durable_hash_contract(tmp_path: Path) -> None:
    source = tmp_path / "snapshot"
    source.mkdir()
    artifact = source / "config.json"
    artifact.write_bytes(b"exact bytes\n")
    import hashlib

    manifest = {
        "status": "GREEN",
        "provider": "vast.ai",
        "content_manifest_sha256": "contract-sha",
        "repositories": [{
            "repo": "fixture/repo",
            "revision": "a" * 40,
            "files": [{
                "path": "config.json",
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }],
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = hydrate(
        manifest_path,
        tmp_path / "cache",
        destination_instance_id=123,
        acknowledge=HYDRATE_ACK,
        downloader=lambda **_: str(source),
    )
    assert report["status"] == "GREEN"
    assert report["verified_bytes"] == artifact.stat().st_size


def test_remote_receipts_bind_h100_scope_and_separate_cache_origin() -> None:
    transfer = {
        "status": "GREEN",
        "provider": "vast.ai",
        "direction": "durable_to_h100_scratch",
        "destination_instance_id": 123,
        "payload_scope": sorted(REQUIRED_TRANSFER_SCOPE),
        "cache_contract_manifest_sha256": "cache-sha",
    }
    hydration = {
        "status": "GREEN",
        "provider": "vast.ai",
        "mode": "NO_SCIENCE_CACHE_HYDRATION",
        "origin": "huggingface_exact_revisions_before_cache_gate",
        "destination_instance_id": 123,
        "content_manifest_sha256": "cache-sha",
    }
    validate_remote_receipts(
        transfer,
        hydration,
        instance_id=123,
        content_manifest_sha256="cache-sha",
    )
    transfer["payload_scope"].remove("task_data")
    with pytest.raises(ValueError, match="input scope"):
        validate_remote_receipts(
            transfer,
            hydration,
            instance_id=123,
            content_manifest_sha256="cache-sha",
        )


def test_inbound_transfer_receipt_detects_corruption(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    scoped = {}
    for scope in REQUIRED_TRANSFER_SCOPE:
        path = root / f"{scope}.bin"
        path.write_bytes(scope.encode())
        scoped[scope] = [path]
    manifest = build_transfer_manifest(root, scoped)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = verify_inbound(
        manifest_path,
        root,
        destination_instance_id=123,
        cache_contract_manifest_sha256="cache-sha",
    )
    assert receipt["verified_entry_count"] == 4
    (root / "runtime_bundle.bin").write_bytes(b"corrupt")
    with pytest.raises(IOError, match="verification failed"):
        verify_inbound(
            manifest_path,
            root,
            destination_instance_id=123,
            cache_contract_manifest_sha256="cache-sha",
        )


def test_weight_genealogy_key_maps_are_frozen_before_value_reads() -> None:
    assert siglip_to_paligemma("vision_model.encoder.layers.0.weight") == (
        "vision_tower.vision_model.encoder.layers.0.weight"
    )
    assert siglip_to_paligemma("text_model.encoder.layers.0.weight") is None
    assert gemma_to_paligemma("model.layers.0.mlp.down_proj.weight") == (
        "language_model.model.layers.0.mlp.down_proj.weight"
    )
    assert paligemma_to_pi("language_model.model.layers.0.mlp.down_proj.weight") == (
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight"
    )
    assert paligemma_to_pi("vision_tower.vision_model.post_layernorm.weight") == (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    )
    assert paligemma_to_pi("multi_modal_projector.linear.weight") == (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    )


def test_weight_genealogy_storage_cast_is_narrow_and_planted() -> None:
    import torch

    parent = torch.tensor([1.125, -2.25], dtype=torch.bfloat16)
    child = parent.float()
    assert canonical_storage_equal(parent, child) == (True, "torch.bfloat16")
    changed = child.clone()
    changed[0] += 0.5
    assert canonical_storage_equal(parent, changed) == (False, "torch.bfloat16")


def test_weight_genealogy_displacement_fixture() -> None:
    import torch

    parent = torch.tensor([3.0, 4.0])
    child = torch.tensor([0.0, 4.0])
    metrics = displacement_metrics(displacement_sums(parent, child, chunk_elements=1))
    assert metrics["relative_l2_to_parent"] == pytest.approx(0.6)
    assert metrics["cosine"] == pytest.approx(0.8)


def test_shared_parent_displacement_fixture() -> None:
    import torch

    parent = torch.tensor([0.0, 0.0])
    pg1 = torch.tensor([1.0, -1.0])
    pg2 = torch.tensor([2.0, -2.0])
    metrics = triadic_displacement_metrics(
        triadic_displacement_sums(parent, pg1, pg2, chunk_elements=1)
    )
    assert metrics["displacement_cosine"] == pytest.approx(1.0)
    assert metrics["pg2_to_pg1_displacement_norm_ratio"] == pytest.approx(2.0)
    assert metrics["sign_agreement_nonzero_union"] == pytest.approx(1.0)
