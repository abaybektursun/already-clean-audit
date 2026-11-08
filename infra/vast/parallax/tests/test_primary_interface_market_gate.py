from __future__ import annotations

import json
from pathlib import Path

from infra.vast.parallax.artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "studies/lineage/cross-lineage/config"
RESULTS = ROOT / "studies/lineage/cross-lineage/results"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_market_gate_refuses_spend_and_leaves_the_account_empty() -> None:
    gate = _load(
        RESULTS
        / "readiness/parallax_pg1_primary_interface_control_market_gate_v5.json"
    )
    assert gate["status"] == "NO_GO_CHEAPEST_ELIGIBLE_H100_ABOVE_FROZEN_CEILING"
    assert gate["market"]["cheapest_eligible"]["dph_total"] > gate["market"][
        "frozen_price_ceiling_dph"
    ]
    assert gate["live_no_create"]["green_receipt_emitted"] is False
    assert not any(
        gate["live_no_create"][field]
        for field in (
            "instances_created",
            "volumes_created",
            "claims_created",
            "S3_objects_written",
            "model_or_data_bytes_downloaded",
            "gpu_operations",
        )
    )
    assert gate["terminal_account"]["active_instances"] == 0
    assert gate["terminal_account"]["active_volumes"] == 0
    assert not any(gate["authorization"].values())


def test_v119_keeps_the_valid_v5_packet_unlicensed() -> None:
    freeze = _load(CONFIG / "parallax_execution_freeze_v119.json")
    assert freeze["status"] == "FROZEN_PRIMARY_PG1_V5_MARKET_NO_GO_ALL_PAID_ACTION_OFF"
    assert not any(freeze["authorization"].values())
    assert freeze["unlicensed_launch_spec"]["fresh_claim_exists"] is False
    assert freeze["unlicensed_launch_spec"]["separate_execution_freeze_exists"] is False
    assert freeze["market_gate"]["observed_cheapest_eligible_dph"] > freeze[
        "market_gate"
    ]["frozen_ceiling_dph"]
    for row in (
        freeze["predecessor_freeze"],
        freeze["unlicensed_launch_spec"],
        freeze["market_gate"],
        freeze["machine_quarantine"],
    ):
        assert sha256_file(ROOT / row["path"]) == row["sha256"]


def test_coverage_audit_keeps_the_two_lineages_and_missing_primary_fact_honest() -> None:
    audit_path = RESULTS / "parallax_scientific_coverage_audit_v1.json"
    audit = _load(audit_path)
    assert audit["lineage_contract"]["single_cross_generation_weight_space_or_causal_chain"] is False
    assert audit["shared_story_audit"]["same_experiment_grammar"] is True
    assert audit["shared_story_audit"]["same_scientific_story"] is False
    assert audit["evidence"]["primary_interface_science"]["scientific_outcome"] == "UNSEEN"
    assert audit["next_decisive_cell"]["cell_id"] == "pg1_equal_token_interface_seed17_discovery_v1"
    assert audit["completion_audit_section_16"]["program_complete"] is False
    for row in audit["evidence"].values():
        if "path" in row:
            assert sha256_file(ROOT / row["path"]) == row["sha256"]

    visual = ROOT / (
        "studies/lineage/cross-lineage/visuals/"
        "parallax_lineage_scoreboard_2026-07-18_v1.svg"
    )
    assert sha256_file(audit_path) == "19b3bba09d70de8c4cd38f6075ea87ac50c550db944a6b31471b0d0c28c4a803"
    assert sha256_file(visual) == "97546a8935ba26acf1971f7e8ff1eb93965d978e04b519d359ccdf4d04cbc838"
