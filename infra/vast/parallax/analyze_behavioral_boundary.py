"""Independently recompute and visualize the normalized paired behavioral boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .behavioral_common import adjudicate_range
from .g1_common import utc_now


def analyze(calibration: Path, confirmation: Path, output_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    cal = json.loads(calibration.read_text(encoding="utf-8"))
    con = json.loads(confirmation.read_text(encoding="utf-8"))
    recomputed = {}
    for offset, lineage in enumerate(("pg1_pi", "pg2")):
        row = con["lineages"][lineage]
        cells = {
            name: {
                item["item_id"]: item["damage_nats"]
                for item in row["rows"]
                if item["construction"] == name
            }
            for name in ("sensitive_coherent", "rotated_coherent", "white")
        }
        verdict = adjudicate_range(
            cells["sensitive_coherent"],
            cells["rotated_coherent"],
            cells["white"],
            frozen_mde=float(row["frozen_mde_nats"]),
            seed=20260717 + offset * 10,
            draws=20000,
            lower_quantile=0.0125,
        )
        if verdict != row["verdict"]:
            raise RuntimeError(f"independent verdict mismatch: {lineage}")
        recomputed[lineage] = {
            "status": verdict["status"],
            "selected_amplitude": cal["lineages"][lineage]["amplitude"]["selected_amplitude"],
            "frozen_mde_nats": row["frozen_mde_nats"],
            "comparisons": verdict["comparisons"],
            "items": verdict["items"],
        }
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_INDEPENDENT_RECOMPUTE",
        "calibration_sha256": sha256_file(calibration),
        "confirmation_sha256": sha256_file(confirmation),
        "comparison_firewall": "effect_over_lineage_specific_frozen_mde_only",
        "lineages": recomputed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "independent_recompute.json", result)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    positions = np.array([0.0, 1.0])
    width = 0.34
    colors = {"rotated": "#456990", "white": "#E07A5F"}
    for shift, null in ((-width / 2, "rotated"), (width / 2, "white")):
        effects = []
        lower = []
        for lineage in ("pg1_pi", "pg2"):
            comparison = recomputed[lineage]["comparisons"][null]
            effects.append(comparison["effect_over_mde"])
            lower.append(
                comparison["bootstrap_familywise_lower"]
                / recomputed[lineage]["frozen_mde_nats"]
            )
        effects_array = np.array(effects)
        lower_array = np.array(lower)
        ax.bar(
            positions + shift,
            effects_array,
            width,
            label=f"sensitive − {null}",
            color=colors[null],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.errorbar(
            positions + shift,
            effects_array,
            yerr=np.vstack([effects_array - lower_array, np.zeros(2)]),
            fmt="none",
            ecolor="#202124",
            capsize=4,
            linewidth=1.4,
        )
    ax.axhline(1.0, color="#202124", linestyle="--", linewidth=1.3, label="frozen MDE gate")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(
        positions,
        [
            "Primary: PaliGemma-1\nNO RANGE",
            "Supporting: PaliGemma-2\nDIRECTIONAL RANGE",
        ],
    )
    ax.set_ylabel("Sensitive advantage / lineage-specific frozen MDE")
    ax.set_title("PARALLAX sealed behavioral boundary\nNormalized verdicts only; raw Gemma scales are not compared")
    ax.set_ylim(0, 1.65)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_dir / "behavioral_boundary.png", dpi=180)
    plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.calibration, args.confirmation, args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
