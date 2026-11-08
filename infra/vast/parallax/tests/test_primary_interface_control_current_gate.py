from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra.vast.parallax import primary_interface_control_current_gate as gate


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / (
    "studies/lineage/cross-lineage/config/"
    "parallax_pg1_primary_interface_control_s3_launch_v5.json"
)


def _zero() -> dict:
    return {
        "active_instances": 0,
        "active_volumes": 0,
        "instance_ids": [],
        "volume_ids": [],
    }


def test_market_observer_preserves_frozen_policy_but_applies_ceiling_after_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    captured = {}

    def select(contract, *, quarantined_machine_ids, excluded_machine_ids):
        captured["contract"] = contract
        captured["quarantine"] = quarantined_machine_ids
        captured["excluded"] = excluded_machine_ids
        return {
            "query": "all-region-exact-query",
            "candidate_count": 3,
            "selected": {"id": 1, "machine_id": 2, "dph_total": 2.26},
        }

    monkeypatch.setattr(gate.launch.legacy, "select_cheapest_h100_offer", select)
    result = gate._observe_worker_market(
        spec["worker"], quarantined_machine_ids={10}, excluded_machine_ids={20}
    )
    assert captured["contract"]["price_ceiling_dph"] == float("inf")
    assert spec["worker"]["price_ceiling_dph"] == 2.25
    assert captured["quarantine"] == {10}
    assert captured["excluded"] == {20}
    assert result["selected_dph_total"] == 2.26
    assert result["frozen_price_ceiling_dph"] == 2.25
    assert result["at_or_below_frozen_ceiling"] is False


def test_current_gate_binds_exact_pre_result_critic_commit_and_requires_post_result_audit() -> None:
    binding = gate._critic_binding(ROOT)
    assert binding["commit"] == "b1ad4f86b1926c7d0fda6020267672b57adf1c8f"
    assert binding["sha256"] == gate.CRITIC_SHA256
    assert binding["imports_producer_metric"] is False
    assert binding["required_post_result_before_interpretation"] is True


def test_current_gate_no_go_runs_source_and_head_checks_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    monkeypatch.setattr(gate.launch, "validate_spec", lambda root, path: spec)
    monkeypatch.setattr(gate.launch.legacy, "_confirm_account_zero", lambda **kwargs: _zero())
    monkeypatch.setattr(gate.launch.legacy, "require_s3_safety", lambda: {"green": True})
    monkeypatch.setattr(
        gate, "validate_primary_source", lambda root: {"status": "GREEN", "confirmation_items_read": 0}
    )
    monkeypatch.setattr(gate.no_create, "_claim_absent", lambda root, spec: {"status": "GREEN_ABSENT"})
    monkeypatch.setattr(
        gate.launch,
        "require_local_outputs_absent",
        lambda root, spec: {"status": "GREEN_ALL_ABSENT"},
    )
    monkeypatch.setattr(gate.launch.legacy, "_require_result_absent", lambda key: None)
    monkeypatch.setattr(
        gate.launch.legacy,
        "require_machine_quarantine",
        lambda root, spec: {"active_machine_ids": [10]},
    )
    monkeypatch.setattr(gate.launch, "_model_rows", lambda root, spec: [{}] * 12)
    monkeypatch.setattr(gate.launch, "_overlay_rows", lambda root, spec: [{}] * 14)
    monkeypatch.setattr(gate.launch.legacy, "_bundle", lambda root, spec: ({}, {}))
    monkeypatch.setattr(
        gate.launch.legacy,
        "_require_input_heads",
        lambda model, bundle, overlays: {"object_count": 27, "all_exact": True},
    )
    monkeypatch.setattr(
        gate.launch.legacy,
        "select_cheapest_offer",
        lambda **kwargs: {"selected": {"machine_id": 99, "dph_total": 0.05}},
    )
    monkeypatch.setattr(
        gate,
        "_observe_worker_market",
        lambda *args, **kwargs: {
            "selected_dph_total": 2.26,
            "frozen_price_ceiling_dph": 2.25,
            "at_or_below_frozen_ceiling": False,
            "selected": {"id": 1, "machine_id": 2, "dph_total": 2.26},
        },
    )
    output = tmp_path / "gate.json"
    packet = gate.run(ROOT, SPEC, output)
    assert packet["status"] == gate.NO_GO_STATUS
    assert packet["decision"] == "STOP_NO_GO_NO_CEILING_RELAXATION_AFTER_OBSERVATION"
    assert packet["source_gate"]["confirmation_items_read"] == 0
    assert packet["input_proof"]["object_count"] == 27
    assert packet["initial_account"] == packet["terminal_account"] == _zero()
    assert set(packet["mutation"].values()) == {0}
    assert packet["paid_authorization"] is False
    assert packet["automatic_retry"] is False
    assert packet["independent_critic"]["required_post_result_before_interpretation"] is True


def test_current_gate_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "exists.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.CurrentGateError, match="overwrite"):
        gate.run(ROOT, SPEC, output)
