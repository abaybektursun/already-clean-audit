"""Audit PARALLAX range pilots for answer-length and score-normalization confounds.

This is a diagnostic audit of already-banked candidate rows.  It preserves the
registered summed-log-likelihood verdict and independently asks whether the same
rows support that verdict under mean-log-likelihood scoring.  The latter is not
a preregistered replacement metric and cannot promote a task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .artifacts import atomic_json, sha256_file
from .training_common import TrainingGateError


TASK_ID = "vr_a_beans_balanced"
REGISTERED_SCORE = "log_likelihood_sum"
DIAGNOSTIC_SCORE = "log_likelihood_mean"


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    normalizer = sum(weights)
    return [weight / normalizer for weight in weights]


def rescore_candidate_rows(
    candidate_rows: Sequence[dict[str, Any]], *, score_field: str
) -> list[dict[str, Any]]:
    """Recompute item predictions and target probabilities from banked rows."""
    if score_field not in {REGISTERED_SCORE, DIAGNOSTIC_SCORE}:
        raise TrainingGateError(f"unsupported score field: {score_field}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["item_id"])].append(row)
    if not grouped:
        raise TrainingGateError("candidate audit received no items")

    items: list[dict[str, Any]] = []
    for item_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: str(row["candidate"]))
        targets = {str(row["target"]) for row in ordered}
        tasks = {str(row["task_id"]) for row in ordered}
        candidates = [str(row["candidate"]) for row in ordered]
        if len(targets) != 1 or len(tasks) != 1 or len(candidates) != len(set(candidates)):
            raise TrainingGateError(f"inconsistent candidate group: {item_id}")
        target = next(iter(targets))
        if target not in candidates:
            raise TrainingGateError(f"target absent from candidates: {item_id}")
        values = [float(row[score_field]) for row in ordered]
        if not all(math.isfinite(value) for value in values):
            raise TrainingGateError(f"nonfinite candidate score: {item_id}")
        probabilities = _softmax(values)
        target_index = candidates.index(target)
        predicted_index = max(range(len(values)), key=lambda index: (values[index], -index))
        items.append(
            {
                "item_id": item_id,
                "task_id": next(iter(tasks)),
                "target": target,
                "prediction": candidates[predicted_index],
                "correct": int(predicted_index == target_index),
                "target_probability": probabilities[target_index],
            }
        )
    return items


def _summarize(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise TrainingGateError("cannot summarize zero items")
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_target[str(item["target"])].append(item)

    def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "items": len(rows),
            "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
            "mean_target_probability": sum(
                float(row["target_probability"]) for row in rows
            )
            / len(rows),
        }

    return {
        "aggregate": summary(items),
        "by_target": {
            target: summary(rows) for target, rows in sorted(by_target.items())
        },
    }


def _snapshot_audit(snapshot: dict[str, Any], *, score_field: str) -> dict[str, Any]:
    items = rescore_candidate_rows(snapshot["evaluation"]["candidate_rows"], score_field=score_field)
    return {
        "step": int(snapshot["step"]),
        "objective": float(snapshot["objective"]),
        **_summarize(items),
        "items_detail": items,
    }


def _token_lengths(snapshots: Sequence[dict[str, Any]]) -> dict[str, int]:
    observed: dict[str, set[int]] = defaultdict(set)
    for snapshot in snapshots:
        for row in snapshot["evaluation"]["candidate_rows"]:
            observed[str(row["candidate"])].add(int(row["suffix_tokens"]))
    inconsistent = {candidate: sorted(lengths) for candidate, lengths in observed.items() if len(lengths) != 1}
    if inconsistent:
        raise TrainingGateError(f"candidate token counts changed across rows: {inconsistent}")
    return {candidate: next(iter(lengths)) for candidate, lengths in sorted(observed.items())}


def _direction_counts(first: dict[str, Any], last: dict[str, Any]) -> dict[str, int]:
    initial = {row["item_id"]: row for row in first["items_detail"]}
    final = {row["item_id"]: row for row in last["items_detail"]}
    if initial.keys() != final.keys():
        raise TrainingGateError("evaluated item population changed across snapshots")
    changes = [
        float(final[item_id]["target_probability"])
        - float(initial[item_id]["target_probability"])
        for item_id in sorted(initial)
    ]
    return {
        "improved": sum(change > 0.0 for change in changes),
        "unchanged": sum(change == 0.0 for change in changes),
        "worsened": sum(change < 0.0 for change in changes),
    }


def _direction_counts_by_target(
    first: dict[str, Any], last: dict[str, Any]
) -> dict[str, dict[str, int]]:
    targets = sorted({str(row["target"]) for row in first["items_detail"]})
    result: dict[str, dict[str, int]] = {}
    for target in targets:
        first_rows = [row for row in first["items_detail"] if row["target"] == target]
        last_rows = [row for row in last["items_detail"] if row["target"] == target]
        result[target] = _direction_counts(
            {"items_detail": first_rows}, {"items_detail": last_rows}
        )
    return result


def _curve(snapshots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": row["step"],
            "objective": row["objective"],
            "accuracy": row["aggregate"]["accuracy"],
            "mean_target_probability": row["aggregate"]["mean_target_probability"],
            "by_target": row["by_target"],
        }
        for row in snapshots
    ]


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise TrainingGateError(f"audit input is outside repository root: {path}") from error


def _audit_lineage(
    path: Path, *, expected_lineage: str, root: Path
) -> dict[str, Any]:
    terminal = json.loads(path.read_text(encoding="utf-8"))
    if terminal.get("status") != "GREEN_JOB_TERMINAL":
        raise TrainingGateError(f"terminal is not green: {path}")
    pilot = terminal["job"]["result"]["payload"]["pilot"]
    if pilot.get("lineage") != expected_lineage:
        raise TrainingGateError(f"wrong lineage in {path}: {pilot.get('lineage')}")
    snapshots = pilot["snapshots"]
    if [int(row["step"]) for row in snapshots] != [0, 1, 2, 4, 8, 16]:
        raise TrainingGateError("unexpected range-screen schedule")
    if int(pilot.get("confirmation_items_read", -1)) != 0:
        raise TrainingGateError("range screen crossed the confirmation firewall")

    registered = [_snapshot_audit(row, score_field=REGISTERED_SCORE) for row in snapshots]
    diagnostic = [_snapshot_audit(row, score_field=DIAGNOSTIC_SCORE) for row in snapshots]
    task_summaries = [row["evaluation"]["summary"][TASK_ID] for row in snapshots]
    for recomputed, recorded in zip(registered, task_summaries):
        if not math.isclose(
            recomputed["aggregate"]["mean_target_probability"],
            float(recorded["mean_target_probability"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            recomputed["aggregate"]["accuracy"],
            float(recorded["accuracy"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TrainingGateError(f"registered metric failed independent recompute: {path}")

    threshold = float(
        pilot["mechanical_adjudication"]["thresholds"]["minimum_heldout_probability_gain"]
    )
    registered_gain = (
        registered[-1]["aggregate"]["mean_target_probability"]
        - registered[0]["aggregate"]["mean_target_probability"]
    )
    diagnostic_gain = (
        diagnostic[-1]["aggregate"]["mean_target_probability"]
        - diagnostic[0]["aggregate"]["mean_target_probability"]
    )
    recorded_gain = float(pilot["mechanical_adjudication"]["heldout_probability_gain"])
    if not math.isclose(registered_gain, recorded_gain, rel_tol=0.0, abs_tol=1e-12):
        raise TrainingGateError(f"registered gain mismatch: {path}")

    return {
        "lineage": expected_lineage,
        "role": pilot["role"],
        "terminal_path": _repo_relative(path, root),
        "terminal_sha256": sha256_file(path),
        "run_id": terminal["run_id"],
        "schedule_sha256": pilot["schedule"]["sha256"],
        "confirmation_items_read": pilot["confirmation_items_read"],
        "candidate_suffix_tokens": _token_lengths(snapshots),
        "registered_sum_score": {
            "status": "VERIFIED_REGISTERED_RESULT",
            "verdict": pilot["mechanical_adjudication"]["verdict"],
            "threshold": threshold,
            "gain": registered_gain,
            "clears_threshold": registered_gain >= threshold,
            "direction_counts": _direction_counts(registered[0], registered[-1]),
            "direction_counts_by_target": _direction_counts_by_target(
                registered[0], registered[-1]
            ),
            "curve": _curve(registered),
        },
        "diagnostic_mean_score": {
            "status": "POST_HOC_DIAGNOSTIC_NOT_PREREGISTERED_NOT_PROMOTIONAL",
            "threshold_shown_for_sensitivity_only": threshold,
            "gain": diagnostic_gain,
            "would_clear_registered_threshold": diagnostic_gain >= threshold,
            "direction_counts": _direction_counts(diagnostic[0], diagnostic[-1]),
            "direction_counts_by_target": _direction_counts_by_target(
                diagnostic[0], diagnostic[-1]
            ),
            "curve": _curve(diagnostic),
        },
    }


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit(
    *,
    pg1_terminal: Path,
    pg2_terminal: Path,
    root: Path,
) -> dict[str, Any]:
    pg1 = _audit_lineage(pg1_terminal, expected_lineage="pg1_pi", root=root)
    pg2 = _audit_lineage(pg2_terminal, expected_lineage="pg2", root=root)
    parity = {
        "same_schedule": pg1["schedule_sha256"] == pg2["schedule_sha256"],
        "same_candidates_and_token_counts": (
            pg1["candidate_suffix_tokens"] == pg2["candidate_suffix_tokens"]
        ),
    }
    if not all(parity.values()):
        raise TrainingGateError(f"cross-lineage pilot parity failed: {parity}")
    token_counts = pg1["candidate_suffix_tokens"]
    unequal_token_counts = len(set(token_counts.values())) != 1
    pg2_verdict_reversal = (
        pg2["registered_sum_score"]["clears_threshold"]
        != pg2["diagnostic_mean_score"]["would_clear_registered_threshold"]
    )
    if not unequal_token_counts or not pg2_verdict_reversal:
        raise TrainingGateError("expected label-length confound signature is absent")

    source_paths = {
        "training_loss": root / "infra/vast/parallax/training_common.py",
        "registered_item_score": root / "infra/vast/parallax/graft_common.py",
        "evaluation_callsite": root / "infra/vast/parallax/training_model.py",
    }
    sources = {
        name: {"path": _repo_relative(path, root), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    result = {
        "schema_version": 1,
        "status": "RED_FOR_SHARED_RANGE_PROMOTION",
        "interpretive_verdict": "VOID_SHARED_RANGE_LABEL_LENGTH_CONFOUND",
        "registered_result_disposition": (
            "PRESERVED_AS_GREEN_MECHANICAL_UNDER_REGISTERED_SUM_SCORE_ONLY"
        ),
        "claim_boundary": {
            "licensed": [
                "PG1 and PG2 each moved under the registered summed-score mechanical screen",
                "the PG2 registered aggregate is exactly reproducible from banked rows",
            ],
            "not_licensed": [
                "clean shared task range",
                "class-uniform PG2 range",
                "promotion to G3",
                "seed29 launch before interface control",
            ],
        },
        "confound": {
            "candidate_suffix_tokens": token_counts,
            "unequal_answer_token_counts": unequal_token_counts,
            "training_normalization": (
                "cross_entropy_mean_over_all_active_suffix_tokens_in_batch"
            ),
            "registered_evaluation_normalization": (
                "softmax_over_candidate_log_likelihood_sums"
            ),
            "diagnostic_evaluation_normalization": (
                "softmax_over_candidate_log_likelihood_means"
            ),
            "mechanism": (
                "longer target labels contribute more active training tokens, while summed "
                "candidate likelihood penalizes longer answer strings at evaluation"
            ),
            "decisive_observation": (
                "PG2 clears the frozen 0.01 gain threshold under registered sum scoring "
                "but not under the post-hoc mean-score diagnostic"
            ),
        },
        "parity": parity,
        "lineages": {"pg1_pi": pg1, "pg2": pg2},
        "source_evidence": sources,
        "next_gate": {
            "paid_action": "OFF",
            "seed29": "FORBIDDEN",
            "g3": "NOT_GREEN",
            "required_before_any_paid_range_continuation": [
                "freeze equal-token answer verbalizers",
                "make per-example training normalization explicit and plant-tested",
                "verify candidate ranking is length-invariant under the frozen evaluation rule",
                "calibrate any probability-gain threshold to that one frozen evaluation rule",
                "run a tiny two-lineage interface-control cell before any second range seed",
            ],
        },
        "audit_inputs_sha256": _canonical_sha(
            {
                "pg1": pg1["terminal_sha256"],
                "pg2": pg2["terminal_sha256"],
                "sources": sources,
            }
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg1-terminal", type=Path, required=True)
    parser.add_argument("--pg2-terminal", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        pg1_terminal=args.pg1_terminal.resolve(),
        pg2_terminal=args.pg2_terminal.resolve(),
        root=args.root.resolve(),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
