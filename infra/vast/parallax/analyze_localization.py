"""Independently recompute and visualize the sealed PG2 localization VOID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .behavioral_localization import adjudicate_localization
from .g1_common import utc_now


def analyze(
    *, calibration_path: Path, confirmation_path: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arms = (
        "full_sensitive",
        "full_rotated",
        "full_white",
        "image_sensitive",
        "image_rotated",
        "prompt_sensitive",
        "prompt_rotated",
    )
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
        frozen_mde=float(confirmation["frozen_mde_nats"]),
        seed=20260720,
        draws=int(config["mde"]["bootstrap_draws"]),
        lower_quantile=float(config["mde"]["familywise_lower_quantile"]),
    )
    if verdict != confirmation["verdict"]:
        raise RuntimeError("independent localization verdict mismatch")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "GREEN_INDEPENDENT_RECOMPUTE",
        "calibration_sha256": sha256_file(calibration_path),
        "confirmation_sha256": sha256_file(confirmation_path),
        "config_sha256": sha256_file(config_path),
        "selected_amplitude": calibration["amplitude"]["selected_amplitude"],
        "prior_green_amplitude": 0.001,
        "frozen_mde_nats": confirmation["frozen_mde_nats"],
        "verdict": verdict,
        "claim_boundary": (
            "void_at_new_0.0003_localization_calibration_not_a_reversal_of_prior_0.001_pg2_range"
        ),
        "branch": "no_g2; only a separately frozen 0.001 fresh-panel rescue can reopen localization",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "independent_recompute.json", result)

    order = ("full_rotated", "full_white", "image", "prompt", "image_minus_prompt")
    labels = (
        "Full − rotated",
        "Full − white",
        "Image slots",
        "Prompt/BOS",
        "Image − prompt",
    )
    effects = np.array([verdict["gates"][name]["effect_over_mde"] for name in order])
    lowers = np.array(
        [
            verdict["gates"][name]["positive_familywise_lower"]
            / verdict["frozen_mde_nats"]
            for name in order
        ]
    )
    colors = ["#5B6C8F", "#7C8DAA", "#4C956C", "#D68C45", "#8E6C88"]
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.22)
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
    ax.axhline(1.0, color="#202124", linestyle="--", linewidth=1.3, label="frozen MDE gate")
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Sensitive advantage / frozen localization MDE")
    ax.set_ylim(-0.65, 1.35)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.10,
        0.96,
        "PG2 localization stopped at its positive control",
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.10,
        0.91,
        "Fresh panel · selected amplitude 0.0003 · full-prefix replay failed both nulls",
        ha="left",
        fontsize=10.5,
        color="#444444",
    )
    fig.text(
        0.10,
        0.06,
        "VOID means no location claim. The earlier PG2 range remains valid at 0.001; "
        "this lower-amplitude assay does not reverse it.",
        fontsize=10,
        color="#5A2A27",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF3E8", "edgecolor": "#E2B792"},
    )
    fig.savefig(output_dir / "localization_void.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        calibration_path=args.calibration,
        confirmation_path=args.confirmation,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
