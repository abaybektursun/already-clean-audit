from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from infra.vast.parallax.parallax_tasks import ConfirmationFirewallError, DiscoveryTaskBank
from infra.vast.parallax.training_common import TrainingGateError
from infra.vast.parallax.training_common import deterministic_batches
from infra.vast.parallax.training_model import select_split_stratified
from infra.vast.parallax.training_pilot_figure import render as render_pilot_figure
from infra.vast.parallax.training_optimizer_scale_gate import adjudicate_learning_rates
from infra.vast.parallax.training_transfer_bundle import build as build_transfer_bundle


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "studies/lineage/cross-lineage/config/parallax_training_real_gate_v1.yaml"
TASK_ROOT = ROOT / "studies/lineage/cross-lineage/data/parallax_tasks_v1"


def test_real_gate_config_is_small_frozen_and_two_lineage():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["status"] == "frozen_before_h100"
    assert config["provider"] == {
        "name": "vast.ai",
        "gpu": "H100",
        "market": "all_regions",
        "offer_rule": "literal_cheapest_offer_passing_suitability_gate",
        "maximum_wall_minutes": 30,
        "forbidden": ["modal", "aws", "legacy_mlops"],
    }
    assert config["lineages"]["order"] == ["pg1_pi", "pg2"]
    assert config["lineages"]["direct_cross_gemma_numeric_comparison"] == "forbidden"
    assert config["optimizer"]["steps"] == 1
    assert config["panel"]["confirmation_reads"] == 0
    assert config["claim_boundary"] == "mechanical_only_no_task_range_or_scientific_claim"


def test_real_gate_panel_is_deterministic_and_confirmation_free():
    bank = DiscoveryTaskBank(TASK_ROOT / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id="vr_a_beans_balanced"))
    train = select_split_stratified(
        rows, task_id="vr_a_beans_balanced", split="train", per_target=1
    )
    discovery = select_split_stratified(
        rows, task_id="vr_a_beans_balanced", split="discovery_eval", per_target=1
    )
    assert len(train) == len(discovery) == 3
    assert {row["target"] for row in train} == {"angular_leaf_spot", "bean_rust", "healthy"}
    assert all(row["split"] == "train" for row in train)
    assert all(row["split"] == "discovery_eval" for row in discovery)
    assert not ({row["item_id"] for row in train} & {row["item_id"] for row in discovery})
    assert train == select_split_stratified(
        rows, task_id="vr_a_beans_balanced", split="train", per_target=1
    )


def test_stratified_selection_fails_closed():
    with pytest.raises(TrainingGateError, match="no train rows"):
        select_split_stratified([], task_id="missing", split="train", per_target=1)
    with pytest.raises(TrainingGateError, match="underfilled"):
        select_split_stratified(
            [{"task_id": "x", "split": "train", "target": "a", "item_id": "1"}],
            task_id="x",
            split="train",
            per_target=2,
        )


def test_discovery_bank_rejects_a_physically_mixed_manifest(tmp_path: Path):
    path = tmp_path / "mixed.jsonl"
    path.write_text('{"split":"train","task_id":"x"}\n'
                    '{"split":"confirmation_eval","task_id":"x"}\n')
    with pytest.raises(ConfirmationFirewallError, match="sealed split"):
        list(DiscoveryTaskBank(path).iter_rows())


def test_new_training_surface_has_no_legacy_provider_imports():
    paths = [
        ROOT / "infra/vast/parallax/training_common.py",
        ROOT / "infra/vast/parallax/training_model.py",
        ROOT / "infra/vast/parallax/training_real_gate.py",
        ROOT / "infra/vast/parallax/training_real_gate_preflight.py",
        ROOT / "infra/vast/parallax/training_planted_gate.py",
        ROOT / "infra/vast/parallax/training_transfer_bundle.py",
        ROOT / "infra/vast/parallax/training_restore_rehearsal.py",
        ROOT / "infra/vast/parallax/training_resource_probe.py",
        ROOT / "infra/vast/parallax/training_pilot_cell.py",
        ROOT / "infra/vast/parallax/training_pilot_figure.py",
    ]
    forbidden = {"modal", "boto3", "infra.pg", "infra.aws_pg"}
    observed = set()
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                observed.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                observed.add(node.module)
    assert not {name for name in observed if any(name == value or name.startswith(value + ".") for value in forbidden)}


def test_transfer_bundle_excludes_confirmation_payload(tmp_path: Path):
    contract = ROOT / "studies/lineage/cross-lineage/config/parallax_training_real_gate_source_contract_v2.json"
    bundle = tmp_path / "bundle.tar.gz"
    receipt = tmp_path / "receipt.json"
    packet = build_transfer_bundle(root=ROOT, contract_path=contract, output=bundle, receipt=receipt)
    assert packet["confirmation_manifest_included"] is False
    assert packet["confirmation_image_included"] is False
    assert packet["selected_image_count"] == 6
    assert packet["isolated_import_smoke"] == {
        "status": "GREEN",
        "modules": [
            "infra.vast.parallax.training_real_gate_preflight",
            "infra.vast.parallax.training_real_gate",
        ],
        "repo_worktree_on_sys_path": False,
    }
    with tarfile.open(bundle, "r:gz") as archive:
        names = archive.getnames()
    assert not any("confirmation_eval" in name or name.endswith("/items.jsonl") for name in names)
    assert not any(name.endswith("/audit.json") or name.endswith("/content.json") for name in names)
    assert "studies/lineage/cross-lineage/data/parallax_tasks_v1/manifests/items.discovery.jsonl" in names
    assert "infra/vast/parallax/dag.py" in names
    assert "infra/vast/parallax/launch_packet.py" in names


def test_restore_rehearsal_bundle_imports_outside_worktree(tmp_path: Path):
    contract = ROOT / "studies/lineage/cross-lineage/config/parallax_training_restore_rehearsal_source_contract_v1.json"
    bundle = tmp_path / "restore-bundle.tar.gz"
    receipt = tmp_path / "restore-receipt.json"
    packet = build_transfer_bundle(root=ROOT, contract_path=contract, output=bundle, receipt=receipt)
    assert packet["confirmation_manifest_included"] is False
    assert packet["confirmation_image_included"] is False
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extracted)
    completed = subprocess.run(
        [sys.executable, "-c", "import infra.vast.parallax.training_restore_rehearsal"],
        cwd=extracted,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plants = json.loads(
        (ROOT / "studies/lineage/cross-lineage/results/readiness/training_harness_weight_free_gates_v1.json").read_text()
    )
    assert plants["determinism"]["exact_resume"] is True


def test_resource_probe_bundle_imports_outside_worktree(tmp_path: Path):
    contract = ROOT / "studies/lineage/cross-lineage/config/parallax_training_resource_probe_source_contract_v1.json"
    bundle = tmp_path / "resource-bundle.tar.gz"
    receipt = tmp_path / "resource-receipt.json"
    packet = build_transfer_bundle(root=ROOT, contract_path=contract, output=bundle, receipt=receipt)
    assert packet["confirmation_manifest_included"] is False
    extracted = tmp_path / "resource-extracted"
    extracted.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extracted)
    completed = subprocess.run(
        [sys.executable, "-c", "import infra.vast.parallax.training_resource_probe"],
        cwd=extracted,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_pilot_schedule_and_figure_stub_are_frozen_and_deterministic(tmp_path: Path):
    config = yaml.safe_load(
        (ROOT / "studies/lineage/cross-lineage/config/parallax_training_pilot_cell_v1.yaml").read_text()
    )
    assert config["status"] == "frozen_before_h100"
    assert config["lineage"] == {
        "id": "pg1_pi",
        "role": "primary",
        "checkpoint_role": "fused",
        "promotion_eligible": False,
    }
    assert config["optimizer"]["snapshots"] == [0, 1, 2, 4, 8, 16]
    schedule = deterministic_batches(item_count=96, batch_size=6, steps=16, seed=17)
    assert schedule == deterministic_batches(item_count=96, batch_size=6, steps=16, seed=17)
    assert sorted(index for batch in schedule for index in batch) == list(range(96))

    result = tmp_path / "pilot.json"
    result.write_text(
        json.dumps(
            {
                "label": "PILOT_MECHANICAL_NOT_PROMOTION_ELIGIBLE",
                "mechanical_adjudication": {"verdict": "GREEN_MECHANICAL", "thresholds": {}},
                "snapshots": [
                    {
                        "step": 0,
                        "objective": 2.0,
                        "evaluation": {"summary": {"vr_a_beans_balanced": {"mean_target_probability": 0.3}}},
                    },
                    {
                        "step": 1,
                        "objective": 1.0,
                        "evaluation": {"summary": {"vr_a_beans_balanced": {"mean_target_probability": 0.6}}},
                    },
                ],
            },
            sort_keys=True,
        )
    )
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    render_pilot_figure(result, first)
    render_pilot_figure(result, second)
    assert first.read_bytes() == second.read_bytes()
    assert b"NOT PROMOTION ELIGIBLE" in first.read_bytes()


def test_consumed_pilot_bundle_contract_refuses_shared_range_code_drift(tmp_path: Path):
    contract = ROOT / "studies/lineage/cross-lineage/config/parallax_training_pilot_source_contract_v2.json"
    bundle = tmp_path / "pilot-bundle.tar.gz"
    receipt = tmp_path / "pilot-receipt.json"
    with pytest.raises(TrainingGateError, match="source hash mismatch"):
        build_transfer_bundle(root=ROOT, contract_path=contract, output=bundle, receipt=receipt)


def test_optimizer_scale_adjudication_selects_lowest_bounded_pass():
    def arm(lr, objectives, gradients, hard_stopped=False):
        return {
            "learning_rate": lr,
            "objectives": objectives,
            "updates": [
                {"loss": objective, "gradient_norm": gradient}
                for objective, gradient in zip(objectives[1:], gradients)
            ],
            "hard_stopped": hard_stopped,
        }

    thresholds = {
        "minimum_final_objective_ratio": 0.5,
        "maximum_final_objective_ratio": 0.9,
        "maximum_objective_ratio": 1.25,
        "maximum_gradient_norm": 500.0,
    }
    packet = adjudicate_learning_rates(
        [
            arm(3e-4, [1.0, 2.0], [100.0], hard_stopped=True),
            arm(1e-4, [1.0, 0.9, 0.8, 0.7, 0.6], [10.0] * 4),
            arm(3e-5, [1.0, 0.97, 0.93, 0.89, 0.85], [3.0] * 4),
            arm(1e-5, [1.0, 0.99, 0.97, 0.95, 0.94], [1.0] * 4),
        ],
        thresholds,
    )
    assert packet["verdict"] == "GREEN_SELECTED_SCALE"
    assert packet["selected_learning_rate"] == 3e-5


def test_optimizer_scale_adjudication_red_when_no_arm_passes():
    thresholds = {
        "minimum_final_objective_ratio": 0.5,
        "maximum_final_objective_ratio": 0.9,
        "maximum_objective_ratio": 1.25,
        "maximum_gradient_norm": 500.0,
    }
    packet = adjudicate_learning_rates(
        [{
            "learning_rate": 1e-4,
            "objectives": [1.0, 3.0],
            "updates": [{"loss": 3.0, "gradient_norm": 600.0}],
            "hard_stopped": False,
        }],
        thresholds,
    )
    assert packet["verdict"] == "RED_NO_STABLE_SCALE"
    assert packet["selected_learning_rate"] is None


def test_consumed_optimizer_bundle_contract_refuses_shared_range_code_drift(tmp_path: Path):
    contract = ROOT / "studies/lineage/cross-lineage/config/parallax_training_optimizer_scale_source_contract_v1.json"
    bundle = tmp_path / "optimizer-scale-bundle.tar.gz"
    receipt = tmp_path / "optimizer-scale-receipt.json"
    with pytest.raises(TrainingGateError, match="source hash mismatch"):
        build_transfer_bundle(root=ROOT, contract_path=contract, output=bundle, receipt=receipt)
