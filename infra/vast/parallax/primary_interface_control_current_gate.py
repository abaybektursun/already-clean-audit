"""Read-only current-source and live-market gate for the frozen PG1 equal-token cell."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from . import primary_interface_control_launch as launch
from . import primary_interface_control_no_create as no_create
from .primary_interface_control_job import RUN_ID
from .primary_interface_control_source_gate import validate_local as validate_primary_source


GREEN_STATUS = "GREEN_PG1_EQUAL_TOKEN_CURRENT_GATE_REQUIRES_SEPARATE_NO_CREATE"
NO_GO_STATUS = "NO_GO_CHEAPEST_ELIGIBLE_H100_ABOVE_FROZEN_CEILING"
CRITIC_COMMIT = "b1ad4f86b1926c7d0fda6020267672b57adf1c8f"
CRITIC_PATH = Path("infra/vast/parallax/audit_primary_interface_independent.py")
CRITIC_SHA256 = "d6b03945ed389a48b764fdeef509aa22818c0051654534f789559c4ba05ee8c4"
CRITIC_FREEZE_PATH = Path(
    "studies/lineage/cross-lineage/research/2026-07-19_parallax_primary_critic_freeze.md"
)
CRITIC_FREEZE_SHA256 = "ff016face127f54a35a82170893d52872474c675b2978874cd6e247137cdf4af"


class CurrentGateError(RuntimeError):
    pass


def _critic_binding(root: Path) -> dict[str, Any]:
    critic = root / CRITIC_PATH
    freeze = root / CRITIC_FREEZE_PATH
    if (
        not critic.is_file()
        or sha256_file(critic) != CRITIC_SHA256
        or not freeze.is_file()
        or sha256_file(freeze) != CRITIC_FREEZE_SHA256
    ):
        raise CurrentGateError("independent pre-result critic binding changed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CRITIC_COMMIT, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise CurrentGateError("independent pre-result critic commit is not in current HEAD")
    committed = subprocess.run(
        ["git", "show", f"{CRITIC_COMMIT}:{CRITIC_PATH}"],
        cwd=root,
        capture_output=True,
    )
    if committed.returncode != 0 or hashlib.sha256(committed.stdout).hexdigest() != CRITIC_SHA256:
        raise CurrentGateError("independent critic bytes differ from the bound commit")
    return {
        "commit": CRITIC_COMMIT,
        "path": str(CRITIC_PATH),
        "sha256": CRITIC_SHA256,
        "freeze_memo": {"path": str(CRITIC_FREEZE_PATH), "sha256": CRITIC_FREEZE_SHA256},
        "imports_producer_metric": False,
        "required_post_result_before_interpretation": True,
    }


def _observe_worker_market(
    contract: dict[str, Any], *, quarantined_machine_ids: set[int], excluded_machine_ids: set[int]
) -> dict[str, Any]:
    observation_contract = copy.deepcopy(contract)
    frozen_ceiling = float(observation_contract["price_ceiling_dph"])
    observation_contract["price_ceiling_dph"] = float("inf")
    market = launch.legacy.select_cheapest_h100_offer(
        observation_contract,
        quarantined_machine_ids=quarantined_machine_ids,
        excluded_machine_ids=excluded_machine_ids,
    )
    selected_price = float(market["selected"]["dph_total"])
    return {
        **market,
        "frozen_price_ceiling_dph": frozen_ceiling,
        "selected_dph_total": selected_price,
        "at_or_below_frozen_ceiling": selected_price <= frozen_ceiling,
        "ceiling_excess_fraction": (
            max(0.0, selected_price - frozen_ceiling) / frozen_ceiling
        ),
    }


def run(root: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    output = output.resolve()
    if output.exists():
        raise CurrentGateError(f"refusing to overwrite current gate: {output}")
    spec = launch.validate_spec(root, spec_path)
    completed: list[str] = []
    critic = _critic_binding(root)
    completed.append("independent_pre_result_critic_commit_and_bytes")
    initial = launch.legacy._confirm_account_zero(seconds=10, poll_seconds=0.2)
    completed.append("account_zero_before")
    safety = launch.legacy.require_s3_safety()
    completed.append("S3_versioning_encryption_public_block")
    source = validate_primary_source(root)
    completed.append("current_source_closure_and_confirmation_absence")
    claim = no_create._claim_absent(root, spec)
    completed.append("fresh_claim_absence")
    local_outputs = launch.require_local_outputs_absent(root, spec)
    completed.append("fresh_local_attempt_paths_absence")
    launch.legacy._require_result_absent(spec["artifact"]["key"])
    launch.legacy._require_result_absent(spec["result"]["key"])
    completed.append("artifact_and_terminal_S3_key_absence")
    quarantine = launch.legacy.require_machine_quarantine(root, spec)
    completed.append("machine_quarantine")
    model_rows = launch._model_rows(root, spec)
    overlay_rows = launch._overlay_rows(root, spec)
    bundle_row, _bundle_receipt = launch.legacy._bundle(root, spec)
    input_proof = launch.legacy._require_input_heads(model_rows, bundle_row, overlay_rows)
    completed.append("twenty_seven_exact_S3_input_HEADs")
    guardian_market = launch.legacy.select_cheapest_offer(
        role="guardian", contract=spec["guardian"]
    )
    completed.append("live_all_region_guardian_market")
    worker_market = _observe_worker_market(
        spec["worker"],
        quarantined_machine_ids=set(quarantine["active_machine_ids"]),
        excluded_machine_ids={int(guardian_market["selected"]["machine_id"])},
    )
    completed.append("live_all_region_literal_cheapest_eligible_H100_market")
    terminal = launch.legacy._confirm_account_zero(seconds=10, poll_seconds=0.2)
    completed.append("account_zero_after")
    price_green = worker_market["at_or_below_frozen_ceiling"]
    packet = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": GREEN_STATUS if price_green else NO_GO_STATUS,
        "run_id": RUN_ID,
        "cell_id": "pg1_equal_token_interface_seed17_fresh_v2",
        "scope": "READ_ONLY_CURRENT_SOURCE_AND_LIVE_ALL_REGION_MARKET_GATE",
        "launch_spec": {
            "path": str(spec_path.relative_to(root)),
            "sha256": sha256_file(spec_path),
        },
        "independent_critic": critic,
        "completed_checks": completed,
        "source_gate": source,
        "initial_account": initial,
        "terminal_account": terminal,
        "s3_safety": safety,
        "claim": claim,
        "local_outputs": local_outputs,
        "result_keys": {
            "artifact": {
                "key": spec["artifact"]["key"],
                "status": "GREEN_ZERO_VERSIONS_AND_DELETE_MARKERS",
            },
            "terminal": {
                "key": spec["result"]["key"],
                "status": "GREEN_ZERO_VERSIONS_AND_DELETE_MARKERS",
            },
        },
        "input_proof": input_proof,
        "machine_quarantine": quarantine,
        "guardian_market": guardian_market,
        "worker_market": worker_market,
        "decision": (
            "CONTINUE_TO_SEPARATE_COMPLETE_NO_CREATE_WITH_FRESH_MARKET_REQUERY"
            if price_green
            else "STOP_NO_GO_NO_CEILING_RELAXATION_AFTER_OBSERVATION"
        ),
        "mutation": {
            "instances_created": 0,
            "volumes_created": 0,
            "claims_created": 0,
            "S3_objects_written": 0,
            "model_or_data_body_bytes_read": 0,
            "GPU_operations": 0,
        },
        "paid_authorization": False,
        "automatic_retry": False,
        "claim_boundary": "CURRENT_SOURCE_AND_MARKET_GATE_ONLY_NO_SCIENTIFIC_RESULT",
    }
    atomic_json(output, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.root, args.spec, args.output)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "selected_dph_total": packet["worker_market"]["selected_dph_total"],
                "frozen_price_ceiling_dph": packet["worker_market"][
                    "frozen_price_ceiling_dph"
                ],
                "mutation": packet["mutation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
