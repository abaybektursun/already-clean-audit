from __future__ import annotations

from pathlib import Path

from infra.vast.parallax.audit_training_range_label_length import audit
from infra.vast.parallax.audit_training_range_label_length import rescore_candidate_rows
from infra.vast.parallax.render_training_range_label_length import render


ROOT = Path(__file__).resolve().parents[4]
PG1 = ROOT / "studies/lineage/cross-lineage/results/parallax_training_pilot_cell_v2_selected_scale/terminal.json"
PG2 = ROOT / "studies/lineage/cross-lineage/results/parallax_training_range_screen_pg2_seed17_v2/terminal.json"


def _row(candidate: str, *, target: str, total: float, tokens: int) -> dict:
    return {
        "item_id": "item-1",
        "task_id": "plant",
        "candidate": candidate,
        "target": target,
        "log_likelihood_sum": total,
        "log_likelihood_mean": total / tokens,
        "suffix_tokens": tokens,
    }


def test_equal_token_lengths_make_sum_and_mean_ranking_identical() -> None:
    rows = [
        _row("a", target="a", total=-2.0, tokens=2),
        _row("b", target="a", total=-3.0, tokens=2),
        _row("c", target="a", total=-4.0, tokens=2),
    ]
    summed = rescore_candidate_rows(rows, score_field="log_likelihood_sum")[0]
    meaned = rescore_candidate_rows(rows, score_field="log_likelihood_mean")[0]
    assert summed["prediction"] == meaned["prediction"] == "a"
    # Dividing every score by a common token count changes softmax temperature,
    # so probabilities need not match even though the candidate order does.
    assert summed["target_probability"] != meaned["target_probability"]


def test_unequal_token_lengths_can_reverse_the_answer() -> None:
    rows = [
        _row("long", target="long", total=-3.0, tokens=3),
        _row("short", target="long", total=-2.0, tokens=1),
    ]
    summed = rescore_candidate_rows(rows, score_field="log_likelihood_sum")[0]
    meaned = rescore_candidate_rows(rows, score_field="log_likelihood_mean")[0]
    assert summed["prediction"] == "short"
    assert meaned["prediction"] == "long"


def test_banked_pilots_fail_closed_for_shared_range_promotion() -> None:
    result = audit(pg1_terminal=PG1, pg2_terminal=PG2, root=ROOT)
    assert result["status"] == "RED_FOR_SHARED_RANGE_PROMOTION"
    assert result["interpretive_verdict"] == "VOID_SHARED_RANGE_LABEL_LENGTH_CONFOUND"
    assert result["confound"]["candidate_suffix_tokens"] == {
        "angular_leaf_spot": 6,
        "bean_rust": 4,
        "healthy": 2,
    }
    pg1 = result["lineages"]["pg1_pi"]
    pg2 = result["lineages"]["pg2"]
    assert pg1["registered_sum_score"]["clears_threshold"] is True
    assert pg1["diagnostic_mean_score"]["would_clear_registered_threshold"] is True
    assert pg2["registered_sum_score"]["clears_threshold"] is True
    assert pg2["diagnostic_mean_score"]["would_clear_registered_threshold"] is False
    assert pg2["registered_sum_score"]["direction_counts_by_target"]["healthy"] == {
        "improved": 0,
        "unchanged": 0,
        "worsened": 8,
    }
    assert result["next_gate"]["seed29"] == "FORBIDDEN"
    assert result["next_gate"]["g3"] == "NOT_GREEN"


def test_teaching_figure_refuses_unfrozen_input_and_renders_audit(tmp_path: Path) -> None:
    import json

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit(pg1_terminal=PG1, pg2_terminal=PG2, root=ROOT)))
    output = tmp_path / "audit.svg"
    result = render(audit_path, output)
    assert result["status"] == "GREEN_RENDERED"
    content = output.read_text(encoding="utf-8")
    assert "answer length is controlled" in content
    assert "SEED 29 BLOCKED" in content
    assert "PaliGemma 1" in content and "PaliGemma 2" in content
