"""Independent critic for the frozen primary PG1 equal-token discovery cell.

This module intentionally imports no PARALLAX producer or metric implementation. It recomputes the
registered interaction statistic, resampling procedures, gates, and verdict from terminal candidate
rows using only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEMANTIC_LABELS = ("angular_leaf_spot", "bean_rust", "healthy")
CODES = ("A", "B", "C")
CODEBOOKS = {
    "cycle_0": {"angular_leaf_spot": "A", "bean_rust": "B", "healthy": "C"},
    "cycle_1": {"angular_leaf_spot": "B", "bean_rust": "C", "healthy": "A"},
    "cycle_2": {"angular_leaf_spot": "C", "bean_rust": "A", "healthy": "B"},
}
EXPECTED_STEPS = (0, 1, 2, 4, 8, 16)
METRIC_ID = "token_prior_invariant_association_v1"
BOOTSTRAP_SEED = 20260718
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_SEED = 20260719
PERMUTATION_REPLICATES = 9_999
MAX_P = 0.05


class IndependentPrimaryAuditError(RuntimeError):
    pass


class IndependentMetricVoid(IndependentPrimaryAuditError):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def _void(verdict: str, message: str) -> None:
    raise IndependentMetricVoid(verdict, message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mapping(raw: Mapping[str, Any]) -> dict[str, str]:
    mapping = {str(key): str(value) for key, value in raw.items()}
    if set(mapping) != set(SEMANTIC_LABELS) or set(mapping.values()) != set(CODES):
        _void("VOID_NONBIJECTIVE_CODEBOOK", "codebook is not a semantic-to-token bijection")
    return mapping


def _parse_rows(rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]) -> dict[str, Any]:
    inverse = {code: semantic for semantic, code in _mapping(mapping).items()}
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    targets: dict[str, str] = {}
    suffix_counts: set[int] = set()
    if not rows:
        _void("VOID_INCOMPLETE_CANDIDATE_CUBE", "candidate rows are empty")
    for raw in rows:
        item_id = str(raw.get("item_id", ""))
        candidate = str(raw.get("candidate", ""))
        target = str(raw.get("target", ""))
        if not item_id or candidate not in CODES or target not in CODES:
            _void("VOID_INCOMPLETE_CANDIDATE_CUBE", "row escaped the frozen A/B/C interface")
        try:
            suffix_tokens = int(raw["suffix_tokens"])
            value = float(raw["log_likelihood_sum"])
        except (KeyError, TypeError, ValueError) as error:
            _void("VOID_INCOMPLETE_CANDIDATE_CUBE", f"candidate row is incomplete: {error}")
        if suffix_tokens <= 0:
            _void("VOID_UNEQUAL_SUFFIX_COUNTS", "candidate suffix has no labeled token")
        if not math.isfinite(value):
            _void("VOID_NONFINITE_SCORE", "candidate score is nonfinite")
        if candidate in scores[item_id]:
            _void("VOID_INCOMPLETE_CANDIDATE_CUBE", f"duplicate row: {item_id}/{candidate}")
        if item_id in targets and targets[item_id] != target:
            _void("VOID_INCOMPLETE_CANDIDATE_CUBE", f"target drift within item: {item_id}")
        scores[item_id][candidate] = value
        targets[item_id] = target
        suffix_counts.add(suffix_tokens)
    if len(suffix_counts) != 1:
        _void("VOID_UNEQUAL_SUFFIX_COUNTS", "suffix counts are unequal or unstable")
    if any(set(candidate_scores) != set(CODES) for candidate_scores in scores.values()):
        _void("VOID_INCOMPLETE_CANDIDATE_CUBE", "one or more candidate triplets is incomplete")
    item_ids = {semantic: [] for semantic in SEMANTIC_LABELS}
    for item_id, target in targets.items():
        semantic = inverse.get(target)
        if semantic is None:
            _void("VOID_NONBIJECTIVE_CODEBOOK", f"unmapped target token: {target}")
        item_ids[semantic].append(item_id)
    for ids in item_ids.values():
        ids.sort()
    class_sizes = {semantic: len(ids) for semantic, ids in item_ids.items()}
    if min(class_sizes.values(), default=0) <= 0 or len(set(class_sizes.values())) != 1:
        _void("VOID_UNBALANCED_SEMANTIC_CLASSES", f"class sizes changed: {class_sizes}")
    return {
        "scores": dict(scores),
        "item_ids": item_ids,
        "suffix_tokens": next(iter(suffix_counts)),
        "class_size": next(iter(class_sizes.values())),
    }


def _matrix(
    parsed: Mapping[str, Any], sampled_ids: Mapping[str, Sequence[str]] | None = None
) -> dict[str, dict[str, float]]:
    groups = sampled_ids or parsed["item_ids"]
    return {
        semantic: {
            code: sum(float(parsed["scores"][item_id][code]) for item_id in groups[semantic])
            / len(groups[semantic])
            for code in CODES
        }
        for semantic in SEMANTIC_LABELS
    }


def _mapped_interaction(
    matrix: Mapping[str, Mapping[str, float]], mapping: Mapping[str, str]
) -> tuple[float, dict[str, float]]:
    values = [float(matrix[semantic][code]) for semantic in SEMANTIC_LABELS for code in CODES]
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        _void("VOID_NONFINITE_SCORE", "interaction matrix is incomplete or nonfinite")
    row_means = {
        semantic: sum(float(matrix[semantic][code]) for code in CODES) / 3.0
        for semantic in SEMANTIC_LABELS
    }
    column_means = {
        code: sum(float(matrix[semantic][code]) for semantic in SEMANTIC_LABELS) / 3.0
        for code in CODES
    }
    grand_mean = sum(values) / 9.0
    mapped = {
        semantic: float(matrix[semantic][mapping[semantic]])
        - row_means[semantic]
        - column_means[mapping[semantic]]
        + grand_mean
        for semantic in SEMANTIC_LABELS
    }
    return sum(mapped.values()) / 3.0, mapped


def _compare(
    base_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    base = _parse_rows(base_rows, mapping)
    final = _parse_rows(final_rows, mapping)
    if base["item_ids"] != final["item_ids"]:
        _void("VOID_ITEM_SET_DRIFT", "base and final item sets differ")
    if base["suffix_tokens"] != final["suffix_tokens"]:
        _void("VOID_UNEQUAL_SUFFIX_COUNTS", "base and final suffix counts differ")
    base_score, base_semantic = _mapped_interaction(_matrix(base), mapping)
    final_score, final_semantic = _mapped_interaction(_matrix(final), mapping)
    return {
        "gain": final_score - base_score,
        "semantic_gains": {
            semantic: final_semantic[semantic] - base_semantic[semantic]
            for semantic in SEMANTIC_LABELS
        },
        "base": base,
        "final": final,
        "mapping": dict(mapping),
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


def _sampled_gain(comparisons: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[str]]) -> float:
    gains = []
    for comparison in comparisons:
        base_score, _ = _mapped_interaction(_matrix(comparison["base"], groups), comparison["mapping"])
        final_score, _ = _mapped_interaction(_matrix(comparison["final"], groups), comparison["mapping"])
        gains.append(final_score - base_score)
    return sum(gains) / len(gains)


def recompute(
    codebooks: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_replicates: int = PERMUTATION_REPLICATES,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    try:
        if bootstrap_replicates < 100 or permutation_replicates < 100:
            _void("VOID_INVALID_RESAMPLING", "at least 100 bootstrap and permutation draws are required")
        by_id = {str(row.get("codebook_id", "")): row for row in codebooks}
        if len(by_id) != len(codebooks) or set(by_id) != set(CODEBOOKS):
            _void("VOID_INCOMPLETE_CODEBOOK_SET", "exactly three frozen codebooks are required")
        comparisons = []
        for codebook_id, expected_mapping in CODEBOOKS.items():
            row = by_id[codebook_id]
            mapping = _mapping(row.get("mapping", {}))
            if mapping != expected_mapping:
                _void("VOID_CODEBOOK_DRIFT", f"mapping drifted: {codebook_id}")
            comparison = _compare(row.get("base_rows", ()), row.get("final_rows", ()), mapping)
            comparison["codebook_id"] = codebook_id
            comparisons.append(comparison)
        reference = comparisons[0]["base"]["item_ids"]
        if any(comparison["base"]["item_ids"] != reference for comparison in comparisons):
            _void("VOID_ITEM_SET_DRIFT", "codebooks do not share exact semantic item IDs")
        gains = [float(comparison["gain"]) for comparison in comparisons]
        mean_gain = sum(gains) / len(gains)
        semantic_gains = {
            semantic: sum(float(row["semantic_gains"][semantic]) for row in comparisons)
            / len(comparisons)
            for semantic in SEMANTIC_LABELS
        }

        bootstrap_rng = random.Random(bootstrap_seed)
        bootstrap_values = []
        for _ in range(bootstrap_replicates):
            sampled = {
                semantic: [bootstrap_rng.choice(reference[semantic]) for _ in reference[semantic]]
                for semantic in SEMANTIC_LABELS
            }
            bootstrap_values.append(_sampled_gain(comparisons, sampled))
        bootstrap = {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "lower": _percentile(bootstrap_values, 0.025),
            "median": _percentile(bootstrap_values, 0.5),
            "upper": _percentile(bootstrap_values, 0.975),
        }

        all_item_ids = [item_id for semantic in SEMANTIC_LABELS for item_id in reference[semantic]]
        class_size = len(reference[SEMANTIC_LABELS[0]])
        permutation_rng = random.Random(permutation_seed)
        at_least_observed = 0
        permutation_values = []
        for _ in range(permutation_replicates):
            shuffled = list(all_item_ids)
            permutation_rng.shuffle(shuffled)
            assigned = {
                semantic: shuffled[index * class_size : (index + 1) * class_size]
                for index, semantic in enumerate(SEMANTIC_LABELS)
            }
            value = _sampled_gain(comparisons, assigned)
            permutation_values.append(value)
            if value >= mean_gain - 1e-15:
                at_least_observed += 1
        permutation = {
            "seed": permutation_seed,
            "replicates": permutation_replicates,
            "at_least_observed": at_least_observed,
            "null_median": _percentile(permutation_values, 0.5),
            "null_95th_percentile": _percentile(permutation_values, 0.95),
            "p_value": (at_least_observed + 1) / (permutation_replicates + 1),
        }
        if mean_gain <= 0 or bootstrap["lower"] <= 0 or permutation["p_value"] > MAX_P:
            verdict = "RED_NO_RELIABLE_IMAGE_CONDITIONED_GAIN"
        elif any(value <= 0 for value in gains):
            verdict = "RED_CODEBOOK_NONUNIFORM"
        elif any(value <= 0 for value in semantic_gains.values()):
            verdict = "RED_SEMANTIC_NONUNIFORM"
        else:
            verdict = "GREEN_PROVISIONAL_IMAGE_CONDITIONED_RANGE"
        return {
            "metric_id": METRIC_ID,
            "mapping_marginalized_interaction_gain": mean_gain,
            "codebook_gains": dict(zip(CODEBOOKS, gains, strict=True)),
            "semantic_gains": semantic_gains,
            "bootstrap": bootstrap,
            "permutation": permutation,
            "verdict": verdict,
        }
    except IndependentMetricVoid as error:
        return {"metric_id": METRIC_ID, "verdict": error.verdict, "failure": str(error)}


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _assert_producer_agreement(producer: Mapping[str, Any], independent: Mapping[str, Any]) -> None:
    if producer.get("metric_id") != METRIC_ID or producer.get("verdict") != independent["verdict"]:
        raise IndependentPrimaryAuditError("producer metric identity or verdict disagrees")
    comparisons = [
        (producer.get("mapping_marginalized_interaction_gain"), independent["mapping_marginalized_interaction_gain"]),
        (producer.get("bootstrap", {}).get("lower"), independent["bootstrap"]["lower"]),
        (producer.get("bootstrap", {}).get("median"), independent["bootstrap"]["median"]),
        (producer.get("bootstrap", {}).get("upper"), independent["bootstrap"]["upper"]),
        (producer.get("permutation_test", {}).get("p_value"), independent["permutation"]["p_value"]),
    ]
    if not all(_close(left, right) for left, right in comparisons):
        raise IndependentPrimaryAuditError("producer aggregate or resampling values disagree")
    if int(producer.get("permutation_test", {}).get("permutations_at_least_observed", -1)) != int(
        independent["permutation"]["at_least_observed"]
    ):
        raise IndependentPrimaryAuditError("producer permutation count disagrees")
    producer_codebooks = {
        str(row.get("codebook_id", "")): row for row in producer.get("codebooks", [])
    }
    if set(producer_codebooks) != set(CODEBOOKS) or any(
        not _close(producer_codebooks[codebook_id].get("interaction_gain"), gain)
        for codebook_id, gain in independent["codebook_gains"].items()
    ):
        raise IndependentPrimaryAuditError("producer codebook gains disagree")
    producer_semantic = producer.get("mapping_marginalized_semantic_interaction_gains", {})
    if any(
        not _close(producer_semantic.get(semantic), gain)
        for semantic, gain in independent["semantic_gains"].items()
    ):
        raise IndependentPrimaryAuditError("producer semantic gains disagree")


def build_audit(
    terminal: Mapping[str, Any],
    *,
    expected_run_id: str,
    terminal_path: str,
    terminal_sha256: str,
    critic_sha256: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = PERMUTATION_REPLICATES,
) -> dict[str, Any]:
    payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    control = payload.get("control", {})
    manifest = payload.get("artifact_manifest", {})
    identities = {
        str(terminal.get("run_id", "")),
        str(payload.get("run_id", "")),
        str(manifest.get("run_id", "")),
    }
    if identities != {expected_run_id}:
        raise IndependentPrimaryAuditError(f"run identity disagreement: {sorted(identities)}")
    if (
        terminal.get("status") != "GREEN_JOB_TERMINAL"
        or control.get("status") != "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION"
        or control.get("label") != "PRIMARY_PG1_INTERFACE_DISCOVERY_NOT_PROMOTION_ELIGIBLE"
        or control.get("lineage") != "pg1_pi"
        or control.get("supporting_pg2_dependency") != "none_within_lineage"
        or control.get("paired_or_shared_range_claim") is not False
        or control.get("promotion_eligible") is not False
        or control.get("confirmation_items_read") != 0
        or control.get("fused_base_reset_exact") is not True
        or manifest.get("confirmation_items_read") != 0
        or manifest.get("promotion_eligible") is not False
    ):
        raise IndependentPrimaryAuditError("terminal escaped the primary discovery firewall")
    codebooks = []
    for row in control.get("codebooks", []):
        snapshots = {int(snapshot["step"]): snapshot for snapshot in row.get("snapshots", [])}
        if tuple(sorted(snapshots)) != EXPECTED_STEPS:
            raise IndependentPrimaryAuditError(f"snapshot schedule changed: {row.get('codebook_id')}")
        codebooks.append(
            {
                "codebook_id": row.get("codebook_id"),
                "mapping": row.get("mapping"),
                "base_rows": snapshots[0].get("evaluation", {}).get("candidate_rows", []),
                "final_rows": snapshots[16].get("evaluation", {}).get("candidate_rows", []),
            }
        )
    independent = recompute(
        codebooks,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )
    if str(independent.get("verdict", "")).startswith("VOID"):
        raise IndependentPrimaryAuditError(f"independent input gate voided: {independent}")
    _assert_producer_agreement(control.get("adjudication", {}), independent)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "program": "parallax",
        "status": "GREEN_INDEPENDENT_EXACT",
        "run_id": expected_run_id,
        "lineage": "SigLIP_plus_Gemma1_to_PaliGemma1_primary",
        "registration": "DISCOVERY_ONLY_NONPROMOTIONAL",
        "bound_inputs": {
            "terminal": {"path": terminal_path, "sha256": terminal_sha256},
            "critic_sha256": critic_sha256,
        },
        "independent_result": independent,
        "producer_agreement": "EXACT_VERDICT_AND_NUMERICS_WITHIN_1E_12",
        "claim_boundary": {
            "confirmation": False,
            "promotion": False,
            "paired_or_shared_range": False,
            "single_pg1_pg2_causal_chain": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    terminal_path = args.terminal.resolve()
    critic_path = Path(__file__).resolve()
    audit = build_audit(
        json.loads(terminal_path.read_text(encoding="utf-8")),
        expected_run_id=args.expected_run_id,
        terminal_path=str(terminal_path),
        terminal_sha256=_sha256(terminal_path),
        critic_sha256=_sha256(critic_path),
    )
    _atomic_json(args.output.resolve(), audit)
    print(json.dumps({"status": audit["status"], "verdict": audit["independent_result"]["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
