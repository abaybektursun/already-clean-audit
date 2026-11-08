from __future__ import annotations

import math

import pytest

from infra.vast.parallax.training_interface_control import CODEBOOKS, CODES, SEMANTIC_LABELS
from infra.vast.parallax.audit_interface_interaction import build_audit
from infra.vast.parallax.training_interface_interaction import adjudicate_codebooks
from infra.vast.parallax.training_interface_interaction import compare_snapshots
from infra.vast.parallax.training_interface_interaction import double_center


def _rows(mapping: dict[str, str], matrix: dict[str, dict[str, float]], *, suffix: int = 2):
    rows = []
    for semantic in SEMANTIC_LABELS:
        for item_index in range(4):
            item_id = f"{semantic}-{item_index}"
            for code in CODES:
                rows.append(
                    {
                        "item_id": item_id,
                        "target": mapping[semantic],
                        "candidate": code,
                        "suffix_tokens": suffix,
                        "log_likelihood_sum": matrix[semantic][code],
                    }
                )
    return rows


def _zero_matrix():
    return {semantic: {code: 0.0 for code in CODES} for semantic in SEMANTIC_LABELS}


def _mapped_matrix(mapping: dict[str, str], diagonal: dict[str, float]):
    matrix = _zero_matrix()
    for semantic, value in diagonal.items():
        matrix[semantic][mapping[semantic]] = value
    return matrix


def _cube(final_matrices=None, *, suffix=2):
    final_matrices = final_matrices or {
        codebook_id: _mapped_matrix(mapping, {semantic: 1.0 for semantic in SEMANTIC_LABELS})
        for codebook_id, mapping in CODEBOOKS.items()
    }
    return [
        {
            "codebook_id": codebook_id,
            "mapping": mapping,
            "base_rows": _rows(mapping, _zero_matrix(), suffix=suffix),
            "final_rows": _rows(mapping, final_matrices[codebook_id], suffix=suffix),
        }
        for codebook_id, mapping in CODEBOOKS.items()
    ]


def _adjudicate(cube):
    return adjudicate_codebooks(
        cube,
        bootstrap_replicates=200,
        bootstrap_seed=7,
        permutation_replicates=199,
        permutation_seed=11,
    )


def test_double_center_exactly_removes_additive_row_and_column_shifts() -> None:
    matrix = {
        semantic: {
            code: 10.0 * semantic_index + code_index
            for code_index, code in enumerate(CODES)
        }
        for semantic_index, semantic in enumerate(SEMANTIC_LABELS)
    }
    centered = double_center(matrix)
    assert all(
        centered[semantic][code] == pytest.approx(0.0, abs=1e-12)
        for semantic in SEMANTIC_LABELS
        for code in CODES
    )


def test_true_mapped_interaction_is_green() -> None:
    result = _adjudicate(_cube())
    assert result["verdict"] == "GREEN_PROVISIONAL_IMAGE_CONDITIONED_RANGE"
    assert result["mapping_marginalized_interaction_gain"] == pytest.approx(2 / 3)
    assert result["bootstrap"]["lower"] == pytest.approx(2 / 3)
    assert result["permutation_test"]["p_value"] <= 0.05


def test_pure_token_prior_shift_cannot_create_a_gain() -> None:
    shifted = {
        codebook_id: {
            semantic: {"A": 5.0, "B": -2.0, "C": 0.75}
            for semantic in SEMANTIC_LABELS
        }
        for codebook_id in CODEBOOKS
    }
    result = _adjudicate(_cube(shifted))
    assert result["mapping_marginalized_interaction_gain"] == pytest.approx(0.0, abs=1e-12)
    assert result["permutation_test"]["p_value"] > 0.05
    assert result["verdict"] == "RED_NO_RELIABLE_IMAGE_CONDITIONED_GAIN"


def test_pure_class_difficulty_shift_cannot_create_a_gain() -> None:
    shifted = {
        codebook_id: {
            semantic: {code: float(index + 1) for code in CODES}
            for index, semantic in enumerate(SEMANTIC_LABELS)
        }
        for codebook_id in CODEBOOKS
    }
    result = _adjudicate(_cube(shifted))
    assert result["mapping_marginalized_interaction_gain"] == pytest.approx(0.0, abs=1e-12)
    assert result["verdict"] == "RED_NO_RELIABLE_IMAGE_CONDITIONED_GAIN"


def test_one_bad_codebook_is_not_hidden_by_positive_average() -> None:
    matrices = {}
    for codebook_id, mapping in CODEBOOKS.items():
        value = -1.0 if codebook_id == "cycle_0" else 2.0
        matrices[codebook_id] = _mapped_matrix(
            mapping, {semantic: value for semantic in SEMANTIC_LABELS}
        )
    result = _adjudicate(_cube(matrices))
    assert result["mapping_marginalized_interaction_gain"] > 0
    assert result["bootstrap"]["lower"] > 0
    assert result["verdict"] == "RED_CODEBOOK_NONUNIFORM"


def test_one_bad_semantic_is_not_hidden_by_positive_average() -> None:
    matrices = {
        codebook_id: _mapped_matrix(
            mapping,
            {"angular_leaf_spot": 10.0, "bean_rust": -6.0, "healthy": 10.0},
        )
        for codebook_id, mapping in CODEBOOKS.items()
    }
    result = _adjudicate(_cube(matrices))
    assert result["mapping_marginalized_interaction_gain"] > 0
    assert result["mapping_marginalized_semantic_interaction_gains"]["bean_rust"] < 0
    assert result["verdict"] == "RED_SEMANTIC_NONUNIFORM"


def test_unequal_suffix_counts_void_before_scoring() -> None:
    cube = _cube()
    cube[0]["final_rows"][0]["suffix_tokens"] = 3
    result = _adjudicate(cube)
    assert result["verdict"] == "VOID_UNEQUAL_SUFFIX_COUNTS"


def test_missing_candidate_and_item_drift_void() -> None:
    missing = _cube()
    missing[1]["final_rows"].pop()
    assert _adjudicate(missing)["verdict"] == "VOID_INCOMPLETE_CANDIDATE_CUBE"

    drift = _cube()
    drift[1]["final_rows"][0]["item_id"] = "different"
    assert _adjudicate(drift)["verdict"] in {
        "VOID_INCOMPLETE_CANDIDATE_CUBE",
        "VOID_UNBALANCED_SEMANTIC_CLASSES",
        "VOID_ITEM_SET_DRIFT",
    }


def test_nonfinite_and_codebook_drift_void() -> None:
    nonfinite = _cube()
    nonfinite[2]["final_rows"][0]["log_likelihood_sum"] = math.nan
    assert _adjudicate(nonfinite)["verdict"] == "VOID_NONFINITE_SCORE"

    wrong_mapping = _cube()
    wrong_mapping[0]["mapping"] = CODEBOOKS["cycle_1"]
    assert _adjudicate(wrong_mapping)["verdict"] == "VOID_CODEBOOK_DRIFT"


def test_compare_snapshots_is_invariant_to_combined_additive_shifts() -> None:
    mapping = CODEBOOKS["cycle_0"]
    base = _mapped_matrix(mapping, {semantic: 0.5 for semantic in SEMANTIC_LABELS})
    shifted = {
        semantic: {
            code: base[semantic][code] + 7.0 * semantic_index - 3.0 * code_index
            for code_index, code in enumerate(CODES)
        }
        for semantic_index, semantic in enumerate(SEMANTIC_LABELS)
    }
    result = compare_snapshots(
        base_rows=_rows(mapping, base), final_rows=_rows(mapping, shifted), mapping=mapping
    )
    assert result["interaction_gain"] == pytest.approx(0.0, abs=1e-12)


def test_banked_terminal_schema_reads_run_id_from_artifact_manifest() -> None:
    codebooks = []
    for row in _cube():
        snapshots = []
        for step in (0, 1, 2, 4, 8, 16):
            candidate_rows = row["final_rows"] if step == 16 else row["base_rows"]
            snapshots.append({"step": step, "evaluation": {"candidate_rows": candidate_rows}})
        codebooks.append(
            {
                "codebook_id": row["codebook_id"],
                "mapping": row["mapping"],
                "snapshots": snapshots,
            }
        )
    terminal = {
        "job": {
            "result": {
                "payload": {
                    "artifact_manifest": {
                        "run_id": "parallax-pg2-interface-control-s3-v3",
                        "confirmation_items_read": 0,
                    },
                    "control": {
                        "adjudication": {
                            "verdict": "RED_SEMANTIC_NONUNIFORM",
                            "mapping_marginalized_probability_gain": 0.0244,
                        },
                        "codebooks": codebooks,
                    },
                }
            }
        }
    }
    result = build_audit(
        terminal,
        terminal_path="terminal.json",
        terminal_sha256="a" * 64,
        metric_spec_path="metric.yaml",
        metric_spec_sha256="b" * 64,
        bootstrap_replicates=200,
        permutation_replicates=199,
    )
    assert result["run_id"] == "parallax-pg2-interface-control-s3-v3"
    assert result["status"] == "DISCOVERY_POSTHOC_GREEN_PROVISIONAL_IMAGE_CONDITIONED_RANGE"
    assert result["registered_result_preserved"]["verdict"] == "RED_SEMANTIC_NONUNIFORM"
