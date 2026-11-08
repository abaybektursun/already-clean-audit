"""Primary-only PG1 interface discovery with the frozen interaction adjudicator.

The completed PG2 controller remains hash-frozen.  This adapter reuses its exact
training mechanics without retaining the old paired-branch dependency, then makes
the token-prior-invariant interaction metric the primary within-lineage verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from . import training_interface_control_cell as legacy_cell
from .artifacts import atomic_json
from .training_common import TrainingGateError
from .training_interface_interaction import METRIC_ID, adjudicate_codebooks


LINEAGE = "pg1_pi"
ROLE = "primary_within_lineage_discovery_independent_of_supporting_pg2"
DESIGN_ID = "parallax-pg1-primary-interface-control-v1"


def _validate_primary_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = {str(row.get("id", "")): row for row in config.get("lineage_order", [])}
    primary = config.get("primary_within_lineage", {})
    metric = config.get("primary_adjudication", {})
    if (
        config.get("status") != "frozen_design_before_tokenizer_preflight"
        or config.get("design_id") != DESIGN_ID
        or config.get("paid_authorization") is not False
        or set(rows) != {LINEAGE}
        or rows[LINEAGE].get("role") != ROLE
        or primary.get("supporting_pg2_dependency") != "none_within_lineage"
        or primary.get("paired_or_shared_range_claim") != "forbidden"
        or primary.get("confirmation_or_promotion") != "forbidden"
        or metric.get("metric_id") != METRIC_ID
        or metric.get("existing_pg1_outcome_status") != "unseen_equal_token_training_outcome"
        or metric.get("use") != "primary_discovery_nonpromotion"
    ):
        raise TrainingGateError("primary PG1 interface design escaped its frozen boundary")
    return config


def _metric_inputs(packet: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = []
    for codebook in packet.get("codebooks", []):
        snapshots = codebook.get("snapshots", [])
        if not snapshots or int(snapshots[0].get("step", -1)) != 0 or int(
            snapshots[-1].get("step", -1)
        ) != 16:
            raise TrainingGateError("primary interface snapshot endpoints changed")
        inputs.append(
            {
                "codebook_id": codebook["codebook_id"],
                "mapping": codebook["mapping"],
                "base_rows": snapshots[0]["evaluation"]["candidate_rows"],
                "final_rows": snapshots[-1]["evaluation"]["candidate_rows"],
            }
        )
    return inputs


def adjudicate_primary_packet(
    packet: dict[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 9_999,
) -> dict[str, Any]:
    if packet.get("lineage") != LINEAGE or packet.get("confirmation_items_read") != 0:
        raise TrainingGateError("primary packet lineage or confirmation firewall changed")
    probability_diagnostic = packet.get("adjudication")
    if not isinstance(probability_diagnostic, dict):
        raise TrainingGateError("legacy probability diagnostic is absent")
    interaction = adjudicate_codebooks(
        _metric_inputs(packet),
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )
    if interaction.get("verdict", "").startswith("VOID"):
        packet_status = "VOID_PRIMARY_INTERFACE_METRIC_INPUT"
    else:
        packet_status = "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION"
    result = dict(packet)
    result.update(
        {
            "status": packet_status,
            "label": "PRIMARY_PG1_INTERFACE_DISCOVERY_NOT_PROMOTION_ELIGIBLE",
            "role": ROLE,
            "diagnostic_probability_adjudication": probability_diagnostic,
            "adjudication": interaction,
            "registered_primary_metric": METRIC_ID,
            "supporting_pg2_dependency": "none_within_lineage",
            "paired_or_shared_range_claim": False,
            "promotion_eligible": False,
        }
    )
    return result


def run(
    *,
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    tokenizer_preflight: Path,
    cache_root: Path,
    task_root: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_primary_config(config_path)
    original = dict(legacy_cell.ALLOWED[LINEAGE])
    legacy_cell.ALLOWED[LINEAGE] = {"role": ROLE, "entity_role": "fused"}
    try:
        packet = legacy_cell.run(
            root=root,
            source_contract=source_contract,
            matrix=matrix,
            config_path=config_path,
            tokenizer_preflight=tokenizer_preflight,
            cache_root=cache_root,
            task_root=task_root,
            lineage=LINEAGE,
            instance_id=instance_id,
            output_dir=output_dir,
        )
    finally:
        legacy_cell.ALLOWED[LINEAGE] = original
    result = adjudicate_primary_packet(packet)
    atomic_json(output_dir / "interface-control.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-preflight", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        root=args.root.resolve(),
        source_contract=args.source_contract.resolve(),
        matrix=args.matrix.resolve(),
        config_path=args.config.resolve(),
        tokenizer_preflight=args.tokenizer_preflight.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        instance_id=args.instance_id,
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["adjudication"]["verdict"],
                "lineage": result["lineage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
