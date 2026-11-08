from __future__ import annotations

import math
from pathlib import Path

import pytest

from infra.vast.parallax import audit_primary_interface_independent as critic


def _matrix(mapping: dict[str, str], values: dict[str, float]) -> dict[str, dict[str, float]]:
    result = {semantic: {code: 0.0 for code in critic.CODES} for semantic in critic.SEMANTIC_LABELS}
    for semantic, value in values.items():
        result[semantic][mapping[semantic]] = value
    return result


def _rows(
    mapping: dict[str, str], matrix: dict[str, dict[str, float]], *, suffix_tokens: int = 2
) -> list[dict]:
    return [
        {
            "item_id": f"{semantic}-{index}",
            "target": mapping[semantic],
            "candidate": code,
            "suffix_tokens": suffix_tokens,
            "log_likelihood_sum": matrix[semantic][code],
        }
        for semantic in critic.SEMANTIC_LABELS
        for index in range(4)
        for code in critic.CODES
    ]


def _cube(values: dict[str, dict[str, float]] | None = None) -> list[dict]:
    values = values or {
        codebook_id: {semantic: 1.0 for semantic in critic.SEMANTIC_LABELS}
        for codebook_id in critic.CODEBOOKS
    }
    result = []
    for codebook_id, mapping in critic.CODEBOOKS.items():
        result.append(
            {
                "codebook_id": codebook_id,
                "mapping": mapping,
                "base_rows": _rows(mapping, _matrix(mapping, {})),
                "final_rows": _rows(mapping, _matrix(mapping, values[codebook_id])),
            }
        )
    return result


def _producer(independent: dict) -> dict:
    return {
        "metric_id": critic.METRIC_ID,
        "verdict": independent["verdict"],
        "mapping_marginalized_interaction_gain": independent[
            "mapping_marginalized_interaction_gain"
        ],
        "mapping_marginalized_semantic_interaction_gains": independent["semantic_gains"],
        "bootstrap": independent["bootstrap"],
        "permutation_test": {
            "p_value": independent["permutation"]["p_value"],
            "permutations_at_least_observed": independent["permutation"]["at_least_observed"],
        },
        "codebooks": [
            {"codebook_id": codebook_id, "interaction_gain": gain}
            for codebook_id, gain in independent["codebook_gains"].items()
        ],
    }


def _terminal(cube: list[dict], *, run_id: str = "primary-test-run") -> dict:
    independent = critic.recompute(cube, bootstrap_replicates=200, permutation_replicates=199)
    codebooks = []
    for row in cube:
        snapshots = []
        for step in critic.EXPECTED_STEPS:
            candidate_rows = row["final_rows"] if step == 16 else row["base_rows"]
            snapshots.append({"step": step, "evaluation": {"candidate_rows": candidate_rows}})
        codebooks.append({"codebook_id": row["codebook_id"], "mapping": row["mapping"], "snapshots": snapshots})
    return {
        "run_id": run_id,
        "status": "GREEN_JOB_TERMINAL",
        "job": {
            "result": {
                "payload": {
                    "run_id": run_id,
                    "artifact_manifest": {
                        "run_id": run_id,
                        "confirmation_items_read": 0,
                        "promotion_eligible": False,
                    },
                    "control": {
                        "status": "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION",
                        "label": "PRIMARY_PG1_INTERFACE_DISCOVERY_NOT_PROMOTION_ELIGIBLE",
                        "lineage": "pg1_pi",
                        "supporting_pg2_dependency": "none_within_lineage",
                        "paired_or_shared_range_claim": False,
                        "promotion_eligible": False,
                        "confirmation_items_read": 0,
                        "fused_base_reset_exact": True,
                        "adjudication": _producer(independent),
                        "codebooks": codebooks,
                    },
                }
            }
        },
    }


def _audit(terminal: dict) -> dict:
    return critic.build_audit(
        terminal,
        expected_run_id="primary-test-run",
        terminal_path="terminal.json",
        terminal_sha256="a" * 64,
        critic_sha256="b" * 64,
        bootstrap_replicates=200,
        permutation_replicates=199,
    )


def test_independent_green_recomputation_and_terminal_audit() -> None:
    recomputed = critic.recompute(_cube(), bootstrap_replicates=200, permutation_replicates=199)
    assert recomputed["verdict"] == "GREEN_PROVISIONAL_IMAGE_CONDITIONED_RANGE"
    assert recomputed["mapping_marginalized_interaction_gain"] == pytest.approx(2 / 3)
    audit = _audit(_terminal(_cube()))
    assert audit["status"] == "GREEN_INDEPENDENT_EXACT"
    assert audit["independent_result"]["verdict"] == recomputed["verdict"]
    assert audit["registration"] == "DISCOVERY_ONLY_NONPROMOTIONAL"


def test_independent_critic_exposes_bad_codebook_and_bad_semantic() -> None:
    bad_codebook = {
        codebook_id: {
            semantic: (-1.0 if codebook_id == "cycle_0" else 2.0)
            for semantic in critic.SEMANTIC_LABELS
        }
        for codebook_id in critic.CODEBOOKS
    }
    assert (
        critic.recompute(
            _cube(bad_codebook), bootstrap_replicates=200, permutation_replicates=199
        )["verdict"]
        == "RED_CODEBOOK_NONUNIFORM"
    )

    bad_semantic = {
        codebook_id: {"angular_leaf_spot": 10.0, "bean_rust": -6.0, "healthy": 10.0}
        for codebook_id in critic.CODEBOOKS
    }
    assert (
        critic.recompute(
            _cube(bad_semantic), bootstrap_replicates=200, permutation_replicates=199
        )["verdict"]
        == "RED_SEMANTIC_NONUNIFORM"
    )


def test_independent_critic_rejects_token_prior_and_structural_voids() -> None:
    cube = _cube()
    for row in cube:
        for candidate_row in row["final_rows"]:
            candidate_row["log_likelihood_sum"] = {"A": 5.0, "B": -2.0, "C": 0.75}[
                candidate_row["candidate"]
            ]
    result = critic.recompute(cube, bootstrap_replicates=200, permutation_replicates=199)
    assert result["mapping_marginalized_interaction_gain"] == pytest.approx(0.0, abs=1e-12)
    assert result["verdict"] == "RED_NO_RELIABLE_IMAGE_CONDITIONED_GAIN"

    unequal = _cube()
    unequal[0]["final_rows"][0]["suffix_tokens"] = 3
    assert (
        critic.recompute(unequal, bootstrap_replicates=200, permutation_replicates=199)["verdict"]
        == "VOID_UNEQUAL_SUFFIX_COUNTS"
    )

    nonfinite = _cube()
    nonfinite[0]["final_rows"][0]["log_likelihood_sum"] = math.nan
    assert (
        critic.recompute(nonfinite, bootstrap_replicates=200, permutation_replicates=199)["verdict"]
        == "VOID_NONFINITE_SCORE"
    )


def test_terminal_firewalls_and_producer_disagreement_fail_closed() -> None:
    wrong_identity = _terminal(_cube())
    wrong_identity["job"]["result"]["payload"]["run_id"] = "other"
    with pytest.raises(critic.IndependentPrimaryAuditError, match="identity"):
        _audit(wrong_identity)

    confirmation = _terminal(_cube())
    confirmation["job"]["result"]["payload"]["control"]["confirmation_items_read"] = 1
    with pytest.raises(critic.IndependentPrimaryAuditError, match="firewall"):
        _audit(confirmation)

    disagreement = _terminal(_cube())
    disagreement["job"]["result"]["payload"]["control"]["adjudication"][
        "mapping_marginalized_interaction_gain"
    ] += 0.01
    with pytest.raises(critic.IndependentPrimaryAuditError, match="disagree"):
        _audit(disagreement)


def test_critic_has_no_import_of_producer_metric() -> None:
    source = Path(critic.__file__).read_text(encoding="utf-8")
    assert "training_interface_interaction" not in source
    assert "training_interface_control" not in source
