"""Replay the frozen interaction metric on a banked interface-control terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .training_interface_interaction import adjudicate_codebooks, compare_snapshots


EXPECTED_RUN_ID = "parallax-pg2-interface-control-s3-v3"
EXPECTED_REGISTERED_VERDICT = "RED_SEMANTIC_NONUNIFORM"
EXPECTED_STEPS = (0, 1, 2, 4, 8, 16)


class InterfaceInteractionAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit(
    terminal: dict[str, Any],
    *,
    terminal_path: str,
    terminal_sha256: str,
    metric_spec_path: str,
    metric_spec_sha256: str,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 9_999,
) -> dict[str, Any]:
    payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    control = payload.get("control", {})
    run_id = str(payload.get("artifact_manifest", {}).get("run_id", ""))
    if run_id != EXPECTED_RUN_ID:
        raise InterfaceInteractionAuditError(f"unexpected run identity: {run_id}")
    registered = control.get("adjudication", {})
    if registered.get("verdict") != EXPECTED_REGISTERED_VERDICT:
        raise InterfaceInteractionAuditError("registered verdict changed")
    if int(payload.get("artifact_manifest", {}).get("confirmation_items_read", -1)) != 0:
        raise InterfaceInteractionAuditError("confirmation firewall changed")

    metric_inputs = []
    trajectories = []
    for codebook in control.get("codebooks", []):
        snapshots = {int(row["step"]): row for row in codebook.get("snapshots", [])}
        if tuple(sorted(snapshots)) != EXPECTED_STEPS:
            raise InterfaceInteractionAuditError(
                f"snapshot schedule changed: {codebook.get('codebook_id')}"
            )
        mapping = codebook["mapping"]
        base_rows = snapshots[0]["evaluation"]["candidate_rows"]
        final_rows = snapshots[16]["evaluation"]["candidate_rows"]
        metric_inputs.append(
            {
                "codebook_id": codebook["codebook_id"],
                "mapping": mapping,
                "base_rows": base_rows,
                "final_rows": final_rows,
            }
        )
        trajectory = []
        for step in EXPECTED_STEPS:
            comparison = compare_snapshots(
                base_rows=base_rows,
                final_rows=snapshots[step]["evaluation"]["candidate_rows"],
                mapping=mapping,
            )
            trajectory.append(
                {
                    "step": step,
                    "mapped_interaction_score": comparison["final"][
                        "mapped_interaction_score"
                    ],
                    "interaction_gain_from_base": comparison["interaction_gain"],
                }
            )
        trajectories.append(
            {"codebook_id": codebook["codebook_id"], "trajectory": trajectory}
        )

    result = adjudicate_codebooks(
        metric_inputs,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )
    if result["verdict"].startswith("VOID"):
        raise InterfaceInteractionAuditError(
            f"banked terminal failed the frozen metric input gate: {result}"
        )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "program": "parallax",
        "status": f"DISCOVERY_POSTHOC_{result['verdict']}",
        "registration_status": "EXPLORATORY_INSTRUMENT_AUDIT_NOT_CONFIRMATION_OR_RESCUE",
        "run_id": run_id,
        "lineage": "SigLIP_plus_Gemma2_to_PaliGemma2_supporting",
        "bound_inputs": {
            "terminal": {"path": terminal_path, "sha256": terminal_sha256},
            "metric_spec": {"path": metric_spec_path, "sha256": metric_spec_sha256},
        },
        "registered_result_preserved": {
            "verdict": EXPECTED_REGISTERED_VERDICT,
            "mapping_marginalized_probability_gain": registered.get(
                "mapping_marginalized_probability_gain"
            ),
            "promotion_eligible": False,
        },
        "frozen_metric_replay": result,
        "trajectories": trajectories,
        "interpretation": (
            "global output-token priors are canceled, but the learned association remains "
            "codebook-nonuniform; one rotation reverses rather than merely shrinking"
            if result["verdict"] == "RED_CODEBOOK_NONUNIFORM"
            else "see frozen metric gates; this post-hoc replay cannot alter registration"
        ),
        "claim_boundary": {
            "existing_pg2_promotion": "forbidden",
            "existing_pg2_confirmation": "forbidden",
            "pg1_state": "unchanged_not_run_in_this_control",
            "single_pg1_pg2_causal_chain": False,
            "fresh_future_use_requires_unseen_frozen_source": True,
        },
        "paid_action": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--metric-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    terminal = json.loads(args.terminal.read_text(encoding="utf-8"))
    audit = build_audit(
        terminal,
        terminal_path=str(args.terminal),
        terminal_sha256=_sha256(args.terminal),
        metric_spec_path=str(args.metric_spec),
        metric_spec_sha256=_sha256(args.metric_spec),
    )
    atomic_json(args.output, audit)
    print(json.dumps({"output": str(args.output), "status": audit["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
