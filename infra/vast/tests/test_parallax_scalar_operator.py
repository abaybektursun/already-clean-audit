from __future__ import annotations

from pathlib import Path

import torch
import yaml

from infra.vast.parallax.audit_scalar_operator_panel import audit
from infra.vast.parallax.materialize_localization_panel import materialize as materialize_localization
from infra.vast.parallax.materialize_scalar_operator_panel import materialize
from infra.vast.parallax.scalar_operator import (
    adjudicate_scalar_operator,
    categorical_fisher_apply,
    categorical_fisher_rayleigh,
    deterministic_directions,
)
from infra.vast.parallax.scalar_operator_audit import recompute


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "studies/lineage/cross-lineage/config/parallax_pg2_scalar_operator_v1.yaml"


def test_categorical_fisher_matches_linear_softmax_kl_curvature() -> None:
    torch.manual_seed(7)
    matrix = torch.randn(6, 4, dtype=torch.float64)
    point = torch.randn(4, dtype=torch.float64)
    direction = torch.randn(4, dtype=torch.float64)
    logits = matrix @ point
    tangent = matrix @ direction
    probabilities = torch.softmax(logits, dim=-1)
    analytic = categorical_fisher_rayleigh(probabilities, tangent)
    qv = matrix.T @ categorical_fisher_apply(probabilities, tangent)
    assert torch.allclose(analytic, torch.dot(direction, qv), rtol=1e-12, atol=1e-12)

    epsilon = 1e-4
    base_logp = torch.log_softmax(logits, dim=-1)
    plus_logp = torch.log_softmax(logits + epsilon * tangent, dim=-1)
    minus_logp = torch.log_softmax(logits - epsilon * tangent, dim=-1)
    kl_plus = torch.sum(probabilities * (base_logp - plus_logp))
    kl_minus = torch.sum(probabilities * (base_logp - minus_logp))
    direct = (kl_plus + kl_minus) / epsilon**2
    assert torch.allclose(direct, analytic, rtol=1e-6, atol=1e-8)


def test_registered_directions_are_deterministic_and_orthonormal() -> None:
    sensitive = torch.arange(1, 65, dtype=torch.float32)
    first = deterministic_directions(sensitive, count=8, namespace="fixture")
    second = deterministic_directions(sensitive, count=8, namespace="fixture")
    matrix = torch.stack(first)
    assert torch.equal(matrix, torch.stack(second))
    assert torch.allclose(matrix @ matrix.T, torch.eye(8), atol=2e-6, rtol=0)


def _gates() -> dict[str, float | int]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {
        **config["gates"],
        "usable_values_required_per_direction": config["epsilon"][
            "usable_values_required_per_direction"
        ],
    }


def _cells() -> list[dict[str, float | int | bool | None]]:
    rows = []
    for direction in range(8):
        rayleigh = 10.0 + direction
        previous = None
        for epsilon in (0.0005, 0.001, 0.002, 0.003):
            jvp_change = None if previous is None else 0.01
            rows.append(
                {
                    "direction_index": direction,
                    "epsilon": epsilon,
                    "usable": True,
                    "symmetric_kl_nats": rayleigh * epsilon**2,
                    "symmetric_direct_kl_curvature": rayleigh * 1.001,
                    "finite_difference_fisher_rayleigh": rayleigh * 0.999,
                    "vjp_rayleigh": rayleigh,
                    "jvp_relative_change_from_previous": jvp_change,
                }
            )
            previous = epsilon
    return rows


def test_scalar_adjudicator_exercises_green_and_redesign_worlds() -> None:
    gates = _gates()
    assert adjudicate_scalar_operator(_cells(), gates)["status"] == "SCALAR_ROUTE_GREEN"

    floor = _cells()
    for row in floor:
        if row["epsilon"] == 0.003:
            row["usable"] = False
        if row["epsilon"] == 0.002:
            row["usable"] = False
    assert (
        adjudicate_scalar_operator(floor, gates)["status"]
        == "REDESIGN_NUMERICAL_FLOOR_OR_SATURATION"
    )

    identity = _cells()
    for row in identity:
        if row["direction_index"] == 0:
            row["symmetric_direct_kl_curvature"] = 1000.0
    assert adjudicate_scalar_operator(identity, gates)["status"] == "REDESIGN_KL_QUADRATIC_IDENTITY"

    route = _cells()
    for row in route:
        if row["direction_index"] == 0:
            row["finite_difference_fisher_rayleigh"] = 1000.0
    assert adjudicate_scalar_operator(route, gates)["status"] == "REDESIGN_FISHER_VJP_ROUTE"

    epsilon = _cells()
    for row in epsilon:
        if row["epsilon"] == 0.001:
            row["jvp_relative_change_from_previous"] = 0.2
    assert adjudicate_scalar_operator(epsilon, gates)["status"] == "REDESIGN_EPSILON_STABILITY"

    psd = _cells()
    for row in psd:
        if row["direction_index"] == 0:
            row["symmetric_direct_kl_curvature"] = -10.0
            row["finite_difference_fisher_rayleigh"] = -10.0
            row["vjp_rayleigh"] = -10.0
            row["symmetric_kl_nats"] = 10.0 * float(row["epsilon"]) ** 2
    assert adjudicate_scalar_operator(psd, gates)["status"] == "REDESIGN_PSD"


def test_scalar_operator_panel_is_balanced_and_fresh(tmp_path: Path) -> None:
    original_root = tmp_path / "localization-v1"
    rescue_root = tmp_path / "localization-rescue-v1"
    operator_root = tmp_path / "operator-v1"
    materialize_localization(
        ROOT / "studies/lineage/cross-lineage/config/parallax_pg2_localization_v1.yaml",
        original_root,
    )
    materialize_localization(
        ROOT
        / "studies/lineage/cross-lineage/config/parallax_pg2_localization_rescue_v1.yaml",
        rescue_root,
    )
    materialize(CONFIG, operator_root)
    report = audit(operator_root, [original_root, rescue_root])
    assert report["status"] == "GREEN"
    assert report["split_counts"] == {"confirmation_eval": 8}
    assert report["target_balance"] == {candidate: 2 for candidate in "ABCD"}
    assert report["prior_panel_overlap"] == {
        "item_ids": 0,
        "image_hashes": 0,
        "source_identities": 0,
    }


def test_scalar_scope_forbids_spectrum_and_cross_lineage_claims() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["scope"]["cell_count"] == 1
    assert config["scope"]["spectrum"] == "forbidden"
    assert config["scope"]["operator_atlas"] == "forbidden"
    assert config["scope"]["pg1_and_pi_implications"] == "none"
    assert config["branch_rules"]["top8"] == "forbidden"
    assert config["branch_rules"]["top32"] == "forbidden"


def test_scalar_freezer_has_no_item_level_access_and_confirmation_is_sealed() -> None:
    package = ROOT / "infra/vast/parallax"
    freezer = (package / "scalar_operator_freeze.py").read_text(encoding="utf-8")
    confirmation = (package / "scalar_operator_confirm.py").read_text(encoding="utf-8")
    assert "items.jsonl" not in freezer
    assert "iter_discovery" not in freezer
    assert "iter_confirmation" not in freezer
    assert "iter_discovery" not in confirmation
    assert "iter_confirmation" in confirmation
    assert "SOURCE_FILES" in confirmation
    for name in (
        "materialize_scalar_operator_panel.py",
        "audit_scalar_operator_panel.py",
        "scalar_operator.py",
        "scalar_operator_model.py",
        "scalar_operator_freeze.py",
        "scalar_operator_confirm.py",
    ):
        assert name in freezer


def test_scalar_gateway_rehearsal_has_no_scientific_panel_access() -> None:
    source = (
        ROOT / "infra/vast/parallax/scalar_operator_gateway_rehearsal.py"
    ).read_text(encoding="utf-8")
    assert "FrozenTaskBank" not in source
    assert "iter_discovery" not in source
    assert "iter_confirmation" not in source
    assert '"operator_panel_metadata_rows_parsed": 0' in source
    assert '"operator_panel_items_evaluated": 0' in source
    assert "evaluate_scalar_operator" in source


def test_scalar_h100_preflight_is_stage_specific_zero_science_and_instance_bound() -> None:
    source = (ROOT / "infra/vast/parallax/scalar_operator_preflight.py").read_text(
        encoding="utf-8"
    )
    assert "audit_scalar_operator_panel" in source
    assert "localization_preflight" not in source
    assert '"science_items_evaluated": 0' in source
    assert '"confirmation_items_evaluated": 0' in source
    assert '"destination_instance_id": instance_id' in source
    assert '"stage": STAGE' in source
    assert "require_h100=True" in source
    assert "len(prior_roots) != 2" in source
    assert "prior_panel_overlap" in source
    assert "runtime archive hash mismatch" in source


def test_scalar_result_independent_recompute_matches_registered_verdict() -> None:
    raw = (
        ROOT
        / "studies/lineage/cross-lineage/results/"
        "parallax_pg2_scalar_operator_confirmation_v1/confirmation.json"
    )
    result = recompute(raw, CONFIG)
    assert result["status"] == "GREEN_INDEPENDENT_RECOMPUTE"
    assert result["registered_verdict_recomputed"] == "REDESIGN_KL_QUADRATIC_IDENTITY"
    assert result["verdict_exact"] is True
    assert result["diagnostics_exact"] is True
    assert result["mechanical"]["expected_hook_calls"] == 388
    assert result["mechanical"]["observed_hook_calls"] == 388
    assert result["gate_pass"] == {
        "usable_range": True,
        "kl_quadratic_identity": False,
        "fisher_vjp_route": False,
        "epsilon_stability": False,
        "psd": False,
    }
