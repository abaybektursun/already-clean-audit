"""Independently recompute and visualize a frozen graft-discovery packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_lower(left: list[float], right: list[float], seed: int, draws: int = 10_000) -> float:
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    means = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(draws)
    )
    return means[int(0.05 * draws) - 1]


def recompute(packet: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    recomputed: dict[str, Any] = {}
    for lineage_index, lineage in enumerate(("pg1_pi", "pg2")):
        host = packet["hosts"][lineage]
        cells = host["items"]
        maps = {
            name: {row["item_id"]: row for row in cells[name]}
            for name in ("native", "shuffled_features", "wrong_image_native")
        }
        if not (set(maps["native"]) == set(maps["shuffled_features"]) == set(maps["wrong_image_native"])):
            errors.append(f"{lineage}: item IDs differ across range cells")
        tasks: dict[str, Any] = {}
        green: list[str] = []
        task_names = sorted({row["task_id"] for row in cells["native"]})
        for task_index, task in enumerate(task_names):
            ids = sorted(item for item, row in maps["native"].items() if row["task_id"] == task)
            vectors = {
                name: [float(maps[name][item]["target_probability"]) for item in ids]
                for name in maps
            }
            effects = {
                "shuffled": sum(a - b for a, b in zip(vectors["native"], vectors["shuffled_features"]))
                / len(ids),
                "wrong": sum(a - b for a, b in zip(vectors["native"], vectors["wrong_image_native"]))
                / len(ids),
            }
            base_seed = 20260716 + 1000 * lineage_index + 101 * task_index
            lowers = {
                "shuffled": _bootstrap_lower(
                    vectors["native"], vectors["shuffled_features"], base_seed
                ),
                "wrong": _bootstrap_lower(
                    vectors["native"], vectors["wrong_image_native"], base_seed + 1
                ),
            }
            is_green = all(effects[name] >= 0.02 and lowers[name] > 0 for name in effects)
            if is_green:
                green.append(task)
            stored = host["range"]["tasks"][task]
            for key in effects:
                if abs(effects[key] - float(stored["native_minus_null"][key])) > 1e-12:
                    errors.append(f"{lineage}/{task}: effect mismatch {key}")
                if abs(lowers[key] - float(stored["bootstrap_95pct_lower"][key])) > 1e-12:
                    errors.append(f"{lineage}/{task}: bootstrap mismatch {key}")
            expected_status = "GREEN_RANGE" if is_green else "NO_RANGE"
            if stored["status"] != expected_status:
                errors.append(f"{lineage}/{task}: status mismatch")
            tasks[task] = {
                "items": len(ids),
                "native_mean": sum(vectors["native"]) / len(ids),
                "shuffled_mean": sum(vectors["shuffled_features"]) / len(ids),
                "wrong_image_mean": sum(vectors["wrong_image_native"]) / len(ids),
                "native_minus_null": effects,
                "bootstrap_95pct_lower": lowers,
                "status": expected_status,
            }
        expected_host = "GREEN_RANGE" if green else "VOID_NO_NATIVE_RANGE"
        if host["range"]["status"] != expected_host or host["range"]["green_tasks"] != green:
            errors.append(f"{lineage}: host range mismatch")
        recomputed[lineage] = {"status": expected_host, "green_tasks": green, "tasks": tasks}
    expected_packet = (
        "GREEN_DISCOVERY_RANGE"
        if any(value["green_tasks"] for value in recomputed.values())
        else "VOID_NO_NATIVE_RANGE"
    )
    if packet["status"] != expected_packet:
        errors.append("packet status mismatch")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_INDEPENDENT_RECOMPUTE" if not errors else "RED",
        "errors": errors,
        "scientific_status": expected_packet,
        "hosts": recomputed,
        "branch_decision": {
            "connector_repair_microcurves": "CANCEL_NO_NATIVE_RANGE",
            "graft_confirmation": "CANCEL_NO_NATIVE_RANGE",
            "shared_vision_bridge_extensions": "CANCEL_NO_NATIVE_RANGE",
            "portable_remappable_or_coadapted_verdict": "VOID",
            "independent_next_stage_still_licensed": "paired_behavioral_mde_after_its_own_freeze_and_gates",
        },
        "claim_boundary": "This packet establishes failure of the registered zero-shot graft assay to show native image-dependent range. It does not establish tower portability, remappability, or co-adaptation.",
    }


def plot(audit: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    task_order = ("rel_a_spatial_selection", "rel_b_relation_comparison", "vr_a_beans_balanced")
    short = {"rel_a_spatial_selection": "spatial", "rel_b_relation_comparison": "relation", "vr_a_beans_balanced": "beans"}
    chance = {"rel_a_spatial_selection": 0.25, "rel_b_relation_comparison": 0.5, "vr_a_beans_balanced": 1 / 3}
    colors = ("#315f85", "#d97a43", "#8c8c8c")
    labels = ("native", "patch-shuffled", "wrong image")
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.19, hspace=0.28, wspace=0.16)
    for row_index, lineage in enumerate(("pg1_pi", "pg2")):
        for column, task in enumerate(task_order):
            ax = axes[row_index, column]
            values = audit["hosts"][lineage]["tasks"][task]
            heights = [values["native_mean"], values["shuffled_mean"], values["wrong_image_mean"]]
            ax.bar(np.arange(3), heights, color=colors, width=0.72)
            ax.axhline(chance[task], color="#222222", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_ylim(0, max(0.62, max(heights) * 1.18))
            ax.set_xticks(np.arange(3), labels if row_index == 1 else ("", "", ""), rotation=18, ha="right")
            ax.set_title(short[task] if row_index == 0 else "")
            ax.text(0.03, 0.94, "NO RANGE", transform=ax.transAxes, va="top", fontsize=9, weight="bold", color="#9b2c2c")
            if column == 0:
                ax.set_ylabel(("PG1 primary\n" if lineage == "pg1_pi" else "PG2 supporting\n") + "target probability")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Registered zero-shot graft assay is VOID: native hosts do not beat both image nulls", fontsize=14, weight="bold")
    fig.text(0.5, 0.025, "Bars are within-host diagnostics only; raw heights are not compared across Gemma generations. Dashed line = candidate chance.", ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    packet = json.loads(raw)
    audit = recompute(packet)
    audit["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    plot(audit, args.figure)
    print(json.dumps({"status": audit["status"], "scientific_status": audit["scientific_status"]}))
    return 0 if audit["status"] == "GREEN_INDEPENDENT_RECOMPUTE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
