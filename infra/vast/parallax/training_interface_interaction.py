"""Token-prior-invariant image-conditioned association for interface controls.

The statistic double-centers a semantic-class x output-code mean log-likelihood
matrix at each checkpoint.  Final-minus-base therefore cancels additive class
difficulty, additive output-token preference, and fixed base interface affinity.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .training_common import TrainingGateError
from .training_interface_control import CODEBOOKS, CODES, SEMANTIC_LABELS, validate_codebooks


METRIC_ID = "token_prior_invariant_association_v1"
SCORE_FIELD = "log_likelihood_sum"
BOOTSTRAP_SEED = 20260718
BOOTSTRAP_REPLICATES = 10_000
CONFIDENCE_LEVEL = 0.95
PERMUTATION_SEED = 20260719
PERMUTATION_REPLICATES = 9_999
MAX_ONE_SIDED_PERMUTATION_P = 0.05


class InteractionMetricError(TrainingGateError):
    """A metric input failure with a stable non-promotional verdict."""

    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def _fail(verdict: str, message: str) -> None:
    raise InteractionMetricError(verdict, message)


def _validate_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in mapping.items()}
    if set(normalized) != set(SEMANTIC_LABELS) or set(normalized.values()) != set(CODES):
        _fail("VOID_NONBIJECTIVE_CODEBOOK", "interaction metric requires a semantic/code bijection")
    return normalized


def _parse_rows(
    rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]
) -> dict[str, Any]:
    normalized_mapping = _validate_mapping(mapping)
    semantic_for_code = {code: semantic for semantic, code in normalized_mapping.items()}
    by_item: dict[str, dict[str, float]] = defaultdict(dict)
    target_by_item: dict[str, str] = {}
    suffix_counts: set[int] = set()

    if not rows:
        _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", "interaction metric received zero rows")
    for raw in rows:
        item_id = str(raw.get("item_id", ""))
        candidate = str(raw.get("candidate", ""))
        target = str(raw.get("target", ""))
        if not item_id or candidate not in CODES or target not in CODES:
            _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", "row identity escaped the A/B/C interface")
        try:
            suffix_tokens = int(raw.get("suffix_tokens", 0))
            score = float(raw[SCORE_FIELD])
        except (KeyError, TypeError, ValueError) as error:
            _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", f"candidate row is incomplete: {error}")
        if suffix_tokens <= 0:
            _fail("VOID_UNEQUAL_SUFFIX_COUNTS", "candidate has no labeled suffix")
        if not math.isfinite(score):
            _fail("VOID_NONFINITE_SCORE", "candidate score is nonfinite")
        if candidate in by_item[item_id]:
            _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", f"duplicate item/candidate row: {item_id}/{candidate}")
        if item_id in target_by_item and target_by_item[item_id] != target:
            _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", f"item target changed within cube: {item_id}")
        by_item[item_id][candidate] = score
        target_by_item[item_id] = target
        suffix_counts.add(suffix_tokens)

    if suffix_counts == set() or len(suffix_counts) != 1:
        _fail("VOID_UNEQUAL_SUFFIX_COUNTS", "A/B/C suffix counts are unequal or unstable")
    for item_id, candidate_scores in by_item.items():
        if set(candidate_scores) != set(CODES):
            _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", f"candidate triplet is incomplete: {item_id}")

    item_ids_by_semantic: dict[str, list[str]] = {semantic: [] for semantic in SEMANTIC_LABELS}
    for item_id, target in target_by_item.items():
        semantic = semantic_for_code.get(target)
        if semantic is None:
            _fail("VOID_NONBIJECTIVE_CODEBOOK", f"target code is not mapped: {target}")
        item_ids_by_semantic[semantic].append(item_id)
    class_sizes = {semantic: len(ids) for semantic, ids in item_ids_by_semantic.items()}
    if min(class_sizes.values(), default=0) <= 0 or len(set(class_sizes.values())) != 1:
        _fail("VOID_UNBALANCED_SEMANTIC_CLASSES", f"semantic class sizes are unbalanced: {class_sizes}")
    for ids in item_ids_by_semantic.values():
        ids.sort()
    return {
        "by_item": dict(by_item),
        "target_by_item": target_by_item,
        "item_ids_by_semantic": item_ids_by_semantic,
        "class_size": next(iter(class_sizes.values())),
        "suffix_tokens": next(iter(suffix_counts)),
    }


def _matrix_from_parsed(
    parsed: Mapping[str, Any], sampled_ids: Mapping[str, Sequence[str]] | None = None
) -> dict[str, dict[str, float]]:
    groups = sampled_ids or parsed["item_ids_by_semantic"]
    return {
        semantic: {
            code: sum(float(parsed["by_item"][item_id][code]) for item_id in groups[semantic])
            / len(groups[semantic])
            for code in CODES
        }
        for semantic in SEMANTIC_LABELS
    }


def double_center(
    matrix: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    """Remove additive semantic-row and token-column effects from a complete cube."""
    if set(matrix) != set(SEMANTIC_LABELS) or any(
        set(matrix[semantic]) != set(CODES) for semantic in SEMANTIC_LABELS
    ):
        _fail("VOID_INCOMPLETE_CANDIDATE_CUBE", "matrix is not a complete 3x3 interface cube")
    values = [float(matrix[semantic][code]) for semantic in SEMANTIC_LABELS for code in CODES]
    if not all(math.isfinite(value) for value in values):
        _fail("VOID_NONFINITE_SCORE", "matrix contains a nonfinite value")
    row_mean = {
        semantic: sum(float(matrix[semantic][code]) for code in CODES) / len(CODES)
        for semantic in SEMANTIC_LABELS
    }
    column_mean = {
        code: sum(float(matrix[semantic][code]) for semantic in SEMANTIC_LABELS)
        / len(SEMANTIC_LABELS)
        for code in CODES
    }
    grand_mean = sum(values) / len(values)
    return {
        semantic: {
            code: float(matrix[semantic][code])
            - row_mean[semantic]
            - column_mean[code]
            + grand_mean
            for code in CODES
        }
        for semantic in SEMANTIC_LABELS
    }


def score_matrix(
    matrix: Mapping[str, Mapping[str, float]], mapping: Mapping[str, str]
) -> dict[str, Any]:
    normalized_mapping = _validate_mapping(mapping)
    interaction = double_center(matrix)
    mapped_by_semantic = {
        semantic: interaction[semantic][normalized_mapping[semantic]]
        for semantic in SEMANTIC_LABELS
    }
    return {
        "matrix": {semantic: dict(matrix[semantic]) for semantic in SEMANTIC_LABELS},
        "double_centered": interaction,
        "mapped_interaction_by_semantic": mapped_by_semantic,
        "mapped_interaction_score": sum(mapped_by_semantic.values()) / len(SEMANTIC_LABELS),
    }


def compare_snapshots(
    *,
    base_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Compute a paired final-minus-base interaction gain for one codebook."""
    base = _parse_rows(base_rows, mapping)
    final = _parse_rows(final_rows, mapping)
    if base["item_ids_by_semantic"] != final["item_ids_by_semantic"]:
        _fail("VOID_ITEM_SET_DRIFT", "base/final semantic item sets differ")
    if base["suffix_tokens"] != final["suffix_tokens"]:
        _fail("VOID_UNEQUAL_SUFFIX_COUNTS", "base/final suffix counts differ")
    base_score = score_matrix(_matrix_from_parsed(base), mapping)
    final_score = score_matrix(_matrix_from_parsed(final), mapping)
    semantic_gain = {
        semantic: final_score["mapped_interaction_by_semantic"][semantic]
        - base_score["mapped_interaction_by_semantic"][semantic]
        for semantic in SEMANTIC_LABELS
    }
    return {
        "base": base_score,
        "final": final_score,
        "interaction_gain": final_score["mapped_interaction_score"]
        - base_score["mapped_interaction_score"],
        "interaction_gain_by_semantic": semantic_gain,
        "class_size": base["class_size"],
        "suffix_tokens": base["suffix_tokens"],
        "_parsed_base": base,
        "_parsed_final": final,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_gain(
    comparisons: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if replicates < 100:
        _fail("VOID_INVALID_BOOTSTRAP", "at least 100 bootstrap replicates are required")
    reference = comparisons[0]["_parsed_base"]["item_ids_by_semantic"]
    if any(comparison["_parsed_base"]["item_ids_by_semantic"] != reference for comparison in comparisons):
        _fail("VOID_ITEM_SET_DRIFT", "codebooks do not share exact semantic item IDs")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled = {
            semantic: [rng.choice(reference[semantic]) for _ in reference[semantic]]
            for semantic in SEMANTIC_LABELS
        }
        codebook_gains = []
        for comparison in comparisons:
            base_matrix = _matrix_from_parsed(comparison["_parsed_base"], sampled)
            final_matrix = _matrix_from_parsed(comparison["_parsed_final"], sampled)
            base_score = score_matrix(base_matrix, comparison["mapping"])["mapped_interaction_score"]
            final_score = score_matrix(final_matrix, comparison["mapping"])["mapped_interaction_score"]
            codebook_gains.append(final_score - base_score)
        draws.append(sum(codebook_gains) / len(codebook_gains))
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "method": "paired_stratified_nonparametric_percentile",
        "seed": seed,
        "replicates": replicates,
        "confidence_level": CONFIDENCE_LEVEL,
        "lower": _percentile(draws, tail),
        "median": _percentile(draws, 0.5),
        "upper": _percentile(draws, 1.0 - tail),
    }


def _permutation_test(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    observed: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Balanced item-label randomization null for image-conditioned association."""
    if replicates < 100:
        _fail("VOID_INVALID_PERMUTATION", "at least 100 permutation replicates are required")
    reference = comparisons[0]["_parsed_base"]["item_ids_by_semantic"]
    if any(comparison["_parsed_base"]["item_ids_by_semantic"] != reference for comparison in comparisons):
        _fail("VOID_ITEM_SET_DRIFT", "codebooks do not share exact semantic item IDs")
    item_ids = [item_id for semantic in SEMANTIC_LABELS for item_id in reference[semantic]]
    class_size = len(reference[SEMANTIC_LABELS[0]])
    rng = random.Random(seed)
    at_least_observed = 0
    null_values: list[float] = []
    for _ in range(replicates):
        shuffled = list(item_ids)
        rng.shuffle(shuffled)
        assigned = {
            semantic: shuffled[index * class_size : (index + 1) * class_size]
            for index, semantic in enumerate(SEMANTIC_LABELS)
        }
        gains = []
        for comparison in comparisons:
            base_matrix = _matrix_from_parsed(comparison["_parsed_base"], assigned)
            final_matrix = _matrix_from_parsed(comparison["_parsed_final"], assigned)
            base_score = score_matrix(base_matrix, comparison["mapping"])["mapped_interaction_score"]
            final_score = score_matrix(final_matrix, comparison["mapping"])["mapped_interaction_score"]
            gains.append(final_score - base_score)
        statistic = sum(gains) / len(gains)
        null_values.append(statistic)
        if statistic >= observed - 1e-15:
            at_least_observed += 1
    return {
        "method": "shared_balanced_semantic_label_permutation_one_sided",
        "seed": seed,
        "replicates": replicates,
        "null_median": _percentile(null_values, 0.5),
        "null_95th_percentile": _percentile(null_values, 0.95),
        "permutations_at_least_observed": at_least_observed,
        "p_value": (at_least_observed + 1) / (replicates + 1),
        "maximum_p_value": MAX_ONE_SIDED_PERMUTATION_P,
    }


def adjudicate_codebooks(
    codebooks: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_replicates: int = PERMUTATION_REPLICATES,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Adjudicate three counterbalanced codebooks without hiding a bad rotation."""
    try:
        validate_codebooks()
        by_id = {str(row.get("codebook_id", "")): row for row in codebooks}
        if set(by_id) != set(CODEBOOKS) or len(by_id) != len(codebooks):
            _fail("VOID_INCOMPLETE_CODEBOOK_SET", "exactly cycle_0/cycle_1/cycle_2 are required")
        comparisons = []
        public_codebooks = []
        for codebook_id in CODEBOOKS:
            row = by_id[codebook_id]
            mapping = _validate_mapping(row.get("mapping", {}))
            if mapping != CODEBOOKS[codebook_id]:
                _fail("VOID_CODEBOOK_DRIFT", f"mapping changed: {codebook_id}")
            comparison = compare_snapshots(
                base_rows=row.get("base_rows", ()),
                final_rows=row.get("final_rows", ()),
                mapping=mapping,
            )
            comparison["mapping"] = mapping
            comparisons.append(comparison)
            public_codebooks.append(
                {
                    "codebook_id": codebook_id,
                    "mapping": mapping,
                    **{key: value for key, value in comparison.items() if not key.startswith("_")},
                }
            )
        gains = [float(row["interaction_gain"]) for row in comparisons]
        mean_gain = sum(gains) / len(gains)
        bootstrap = _bootstrap_mean_gain(
            comparisons, replicates=bootstrap_replicates, seed=bootstrap_seed
        )
        permutation = _permutation_test(
            comparisons,
            observed=mean_gain,
            replicates=permutation_replicates,
            seed=permutation_seed,
        )
        semantic_gains = {
            semantic: sum(
                float(row["interaction_gain_by_semantic"][semantic]) for row in comparisons
            )
            / len(comparisons)
            for semantic in SEMANTIC_LABELS
        }
        if (
            mean_gain <= 0
            or bootstrap["lower"] <= 0
            or permutation["p_value"] > MAX_ONE_SIDED_PERMUTATION_P
        ):
            verdict = "RED_NO_RELIABLE_IMAGE_CONDITIONED_GAIN"
        elif any(gain <= 0 for gain in gains):
            verdict = "RED_CODEBOOK_NONUNIFORM"
        elif any(gain <= 0 for gain in semantic_gains.values()):
            verdict = "RED_SEMANTIC_NONUNIFORM"
        else:
            verdict = "GREEN_PROVISIONAL_IMAGE_CONDITIONED_RANGE"
        return {
            "schema_version": 1,
            "metric_id": METRIC_ID,
            "score_field": SCORE_FIELD,
            "formula": "I_ct=M_ct-rowmean_c-colmean_t+grandmean; S=mean_c(I_c,m(c)); gain=S_final-S_base",
            "invariances": [
                "additive_semantic_class_difficulty",
                "additive_output_token_prior",
                "fixed_base_codebook_affinity_via_final_minus_base",
            ],
            "codebooks": public_codebooks,
            "mapping_marginalized_interaction_gain": mean_gain,
            "mapping_marginalized_semantic_interaction_gains": semantic_gains,
            "bootstrap": bootstrap,
            "permutation_test": permutation,
            "gates": {
                "bootstrap_lower_strictly_positive": bootstrap["lower"] > 0,
                "image_label_permutation_p_at_most_0_05": permutation["p_value"]
                <= MAX_ONE_SIDED_PERMUTATION_P,
                "every_codebook_gain_strictly_positive": all(gain > 0 for gain in gains),
                "every_semantic_gain_strictly_positive": all(
                    gain > 0 for gain in semantic_gains.values()
                ),
            },
            "verdict": verdict,
        }
    except InteractionMetricError as error:
        return {
            "schema_version": 1,
            "metric_id": METRIC_ID,
            "verdict": error.verdict,
            "failure": str(error),
        }
