"""Independently recompute and visualize the sealed PG2 localization rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_localization import adjudicate_localization
from .g1_common import utc_now


def analyze(*, confirmation_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arms = tuple(config["cells"]["arms"])
    cells = {
        arm: {
            row["item_id"]: row["damage_nats"]
            for row in confirmation["rows"]
            if row["arm"] == arm
        }
        for arm in arms
    }
    verdict = adjudicate_localization(
        cells,
        frozen_mde=float(config["mde"]["fixed_nats"]),
        seed=20260721,
        draws=int(config["mde"]["bootstrap_draws"]),
        lower_quantile=float(config["mde"]["familywise_lower_quantile"]),
    )
    if verdict != confirmation["verdict"]:
        raise RuntimeError("independent rescue verdict mismatch")
    energy = confirmation["execution"]["energy_audits"]
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_INDEPENDENT_RECOMPUTE",
        "confirmation_sha256": sha256_file(confirmation_path),
        "config_sha256": sha256_file(config_path),
        "selected_amplitude": confirmation["selected_amplitude"],
        "frozen_mde_nats": confirmation["frozen_mde_nats"],
        "verdict": verdict,
        "integrity": {
            "identity": confirmation["identity"],
            "parameter_versions": confirmation["parameter_versions"],
            "hook_calls": confirmation["execution"]["hook_calls"],
            "max_relative_total_energy_error": max(
                row["total"]["max_relative_total_energy_error"] for row in energy
            ),
            "max_relative_per_token_norm_error": max(
                row["per_token"]["max_relative_norm_error"] for row in energy
            ),
        },
        "interpretation": (
            "full_prefix_range_replicates; image_and_prompt_partitions_are_each_positive_but_"
            "below_mde; distribution_or_interaction_is_unresolved"
        ),
        "claim_boundary": "token_positions_only_not_image_content_semantics",
        "branch": "license_one_full_prefix_scalar_g2_instrumentation_cell_only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "independent_recompute.json", result)

    order = ("full_rotated", "full_white", "image", "prompt", "image_minus_prompt")
    labels = ("Full - rotated", "Full - white", "Image slots", "Prompt/BOS", "Image - prompt")
    effects = np.array([verdict["gates"][name]["effect_over_mde"] for name in order])
    lowers = np.array(
        [
            verdict["gates"][name]["positive_familywise_lower"]
            / verdict["frozen_mde_nats"]
            for name in order
        ]
    )
    colors = ["#2E7D5B", "#3D8F69", "#D89B3C", "#D17C3F", "#8B7C8E"]
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.80, bottom=0.23)
    positions = np.arange(len(order))
    ax.bar(positions, effects, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(
        positions,
        effects,
        yerr=np.vstack([effects - lowers, np.zeros(len(effects))]),
        fmt="none",
        ecolor="#202124",
        capsize=5,
        linewidth=1.5,
    )
    ax.axhspan(1.0, 1.55, color="#EAF5EE", alpha=0.8, zorder=0)
    ax.axhline(1.0, color="#202124", linestyle="--", linewidth=1.3, label="frozen MDE gate")
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Sensitive advantage / frozen localization MDE")
    ax.set_ylim(-1.15, 1.55)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.10,
        0.955,
        "PG2 range needs the full prefix",
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.10,
        0.905,
        "Fresh panel - fixed amplitude 0.001 - full replay passes both nulls",
        ha="left",
        fontsize=10.5,
        color="#444444",
    )
    fig.text(
        0.10,
        0.055,
        "Image and prompt positions are each positive but sub-MDE; neither dominates. "
        "Verdict: distributed or unresolved position range.",
        fontsize=10,
        color="#284A3B",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EFF8F1", "edgecolor": "#9DC5AA"},
    )
    fig.savefig(output_dir / "localization_rescue.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        confirmation_path=args.confirmation,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
