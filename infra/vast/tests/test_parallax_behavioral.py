from __future__ import annotations

import pytest
import torch
from pathlib import Path

from infra.vast.parallax.behavioral_common import (
    BehavioralGateError,
    adjudicate_range,
    energy_audit,
    make_field,
    select_amplitude,
    select_target_stratified,
    sign_flip_mde,
)
from infra.vast.parallax.behavioral_model import affected_prefix_mask, site_hook, suffix_metrics


def _row(target: str, index: int) -> dict[str, object]:
    return {
        "task_id": "task",
        "target": target,
        "item_id": f"{target}-{index:02d}",
        "image": f"{target}-{index}.png",
    }


def test_target_panel_selection_is_order_invariant_and_stratified() -> None:
    rows = [_row(target, index) for target in "AB" for index in range(4)]
    selected = select_target_stratified(reversed(rows), task_id="task", per_target=2)
    assert [row["item_id"] for row in selected] == ["A-00", "A-01", "B-00", "B-01"]


@pytest.mark.parametrize("construction", ["sensitive_coherent", "rotated_coherent", "white"])
def test_all_fields_have_exact_matched_per_token_energy(construction: str) -> None:
    direction = torch.arange(1, 9, dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    rms = torch.tensor([2.0, 3.0])
    field = make_field(
        construction=construction,
        sensitive_direction=direction,
        active_mask=mask,
        activation_rms=rms,
        amplitude=0.01,
        namespace="test",
        lineage="pg1_pi",
        item_ids=["one", "two"],
    )
    audit = energy_audit(field, mask, rms, 0.01)
    assert audit["status"] == "GREEN"
    assert audit["active_tokens"] == 5


def test_rotated_and_white_fields_are_orthogonal_to_sensitive_direction() -> None:
    direction = torch.arange(1, 9, dtype=torch.float32)
    mask = torch.ones(1, 4)
    for construction in ("rotated_coherent", "white"):
        field = make_field(
            construction=construction,
            sensitive_direction=direction,
            active_mask=mask,
            activation_rms=torch.ones(1),
            amplitude=0.1,
            namespace="test",
            lineage="pg2",
            item_ids=["one"],
        )
        cosines = torch.nn.functional.cosine_similarity(field[0], direction.expand_as(field[0]))
        assert torch.max(torch.abs(cosines)) < 1e-5


def test_amplitude_rule_selects_smallest_readable_unsaturated_value() -> None:
    rows = []
    for amplitude, values in ((0.001, [0.0001] * 4), (0.003, [0.002] * 4), (0.01, [0.02] * 4)):
        rows.extend({"amplitude": amplitude, "kl_nats": value} for value in values)
    result = select_amplitude(
        rows, [0.001, 0.003, 0.01], median_kl_min=0.001, median_kl_max=0.05, p90_kl_max=0.2
    )
    assert result["status"] == "GREEN"
    assert result["selected_amplitude"] == 0.003


def test_lineage_specific_mde_is_never_below_fixed_numerical_floor() -> None:
    result = sign_flip_mde([0.001, -0.001] * 8, seed=1, draws=1000, quantile=0.975, minimum=0.01)
    assert result["frozen_mde_nats"] == 0.01


def test_range_requires_sensitive_to_beat_both_nulls() -> None:
    sensitive = {str(i): 0.4 for i in range(16)}
    rotated = {str(i): 0.1 for i in range(16)}
    white = {str(i): 0.2 for i in range(16)}
    green = adjudicate_range(
        sensitive, rotated, white, frozen_mde=0.05, seed=1, draws=1000, lower_quantile=0.0125
    )
    assert green["status"] == "GREEN_DIRECTIONAL_RANGE"
    no_range = adjudicate_range(
        sensitive,
        rotated,
        {str(i): 0.39 for i in range(16)},
        frozen_mde=0.05,
        seed=1,
        draws=1000,
        lower_quantile=0.0125,
    )
    assert no_range["status"] == "NO_RANGE"


def test_field_contract_refuses_negative_amplitude() -> None:
    with pytest.raises(BehavioralGateError, match="nonnegative"):
        make_field(
            construction="white",
            sensitive_direction=torch.ones(4),
            active_mask=torch.ones(1, 2),
            activation_rms=torch.ones(1),
            amplitude=-1,
            namespace="test",
            lineage="pg1_pi",
            item_ids=["one"],
        )


def test_site_hook_zero_add_is_exact_and_nonzero_add_is_local() -> None:
    layer = torch.nn.Identity()
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    with site_hook(layer, mode="add", field=torch.zeros_like(hidden)) as zero:
        replay = layer(hidden)
    assert zero["calls"] == 1
    assert torch.equal(replay, hidden)
    field = torch.ones_like(hidden)
    with site_hook(layer, mode="add", field=field) as added:
        changed = layer(hidden)
    assert added["calls"] == 1
    assert torch.equal(changed, hidden + 1)


def test_suffix_metrics_use_only_shifted_labeled_positions() -> None:
    labels = torch.tensor([[-100, -100, 2, 1]])
    base = [torch.tensor([[0.0, 0.0, 4.0], [0.0, 4.0, 0.0]])]
    perturbed = [torch.zeros(2, 3)]
    result = suffix_metrics(base, perturbed, labels)
    assert result[0]["suffix_tokens"] == 2
    assert result[0]["damage_nats"] > 0


def test_intervention_mask_excludes_teacher_forced_suffix_and_padding() -> None:
    inputs = {
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 7, 8, -100], [-100, -100, 9, -100, -100]]),
    }
    mask = affected_prefix_mask(inputs)
    assert torch.equal(mask, torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool))
    assert not bool((mask & inputs["labels"].ne(-100)).any())


def test_bfloat16_injection_energy_is_audited_after_cast() -> None:
    direction = torch.arange(1, 129, dtype=torch.float32)
    mask = torch.ones(2, 4)
    rms = torch.tensor([0.7, 1.3])
    field = make_field(
        construction="white",
        sensitive_direction=direction,
        active_mask=mask,
        activation_rms=rms,
        amplitude=0.003,
        namespace="test",
        lineage="pg1_pi",
        item_ids=["one", "two"],
    ).to(torch.bfloat16)
    audit = energy_audit(field, mask, rms, 0.003, relative_tolerance=0.005)
    assert audit["status"] == "GREEN"
    assert audit["injection_dtype"] == "torch.bfloat16"


def test_gateway_rehearsal_cannot_read_scientific_task_bank() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "infra/vast/parallax/behavioral_gateway_rehearsal.py"
    ).read_text(encoding="utf-8")
    assert "FrozenTaskBank" not in source
    assert "iter_discovery" not in source
    assert "iter_confirmation" not in source
    assert '"discovery_items_read": 0' in source
    assert '"confirmation_items_read": 0' in source
    assert 'for lineage in ("pg1_pi", "pg2")' in source


def test_calibration_runner_does_not_parse_confirmation_for_integrity_audit() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "infra/vast/parallax/behavioral_calibrate.py"
    ).read_text(encoding="utf-8")
    assert "audit_task_bank" not in source
    assert "iter_confirmation" not in source
    assert "_validate_frozen_task_audit" in source
    assert "bank.iter_discovery" in source


def test_confirmation_freeze_cannot_read_item_manifest() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "infra/vast/parallax/behavioral_freeze_confirmation.py"
    ).read_text(encoding="utf-8")
    assert "items.jsonl" not in source
    assert "iter_discovery" not in source
    assert "iter_confirmation" not in source
    assert '"confirmation_items_read_before_freeze": 0' in source


def test_confirmation_runner_has_only_sealed_task_access() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "infra/vast/parallax/behavioral_confirm.py"
    ).read_text(encoding="utf-8")
    assert "iter_discovery" not in source
    assert "iter_confirmation" in source
    assert "FROZEN_FOR_CONFIRMATION" in source
    assert "confirmation_runner_sha256" in source
    assert '"lineage_firewall": freeze["lineage_firewall"]' in source
    assert '"confirmation_metadata_rows_parsed": len(eligible_confirmation)' in source
    assert '"confirmation_unique_items_evaluated": len(panel)' in source
    assert '"confirmation_unique_items_read"' not in source
