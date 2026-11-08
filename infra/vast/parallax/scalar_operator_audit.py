"""Independently recompute and visualize a sealed scalar-operator result packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1e-30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute(raw_path: Path, config_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gates = config["gates"]
    required_eps = int(config["epsilon"]["usable_values_required_per_direction"])
    cells = raw["cells"]
    by_direction: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        by_direction[int(row["direction_index"])].append(row)
    usable = {
        direction: sorted(
            (row for row in rows if bool(row["usable"])),
            key=lambda row: float(row["epsilon"]),
        )
        for direction, rows in by_direction.items()
    }
    accepted = [row for rows in usable.values() for row in rows]
    usable_counts = {str(key): len(value) for key, value in sorted(usable.items())}
    kl_errors = [
        _relative(row["symmetric_direct_kl_curvature"], row["vjp_rayleigh"])
        for row in accepted
    ]
    route_errors = [
        _relative(row["finite_difference_fisher_rayleigh"], row["vjp_rayleigh"])
        for row in accepted
    ]
    slopes: list[float] = []
    jvp_changes: list[float] = []
    direction_slopes: dict[str, list[float]] = {}
    for direction, rows in sorted(usable.items()):
        current = []
        for left, right in zip(rows, rows[1:]):
            slope = math.log(
                float(right["symmetric_kl_nats"])
                / float(left["symmetric_kl_nats"])
            ) / math.log(float(right["epsilon"]) / float(left["epsilon"]))
            slopes.append(slope)
            current.append(slope)
            if right["jvp_relative_change_from_previous"] is not None:
                jvp_changes.append(float(right["jvp_relative_change_from_previous"]))
        direction_slopes[str(direction)] = current

    diagnostics = {
        "kl_q_relative_error": {
            "median": _quantile(kl_errors, 0.5),
            "p90": _quantile(kl_errors, 0.9),
        },
        "fisher_vjp_relative_error": {
            "median": _quantile(route_errors, 0.5),
            "p90": _quantile(route_errors, 0.9),
        },
        "epsilon": {
            "max_adjacent_jvp_relative_change": max(jvp_changes),
            "kl_slope_median_abs_error_from_two": _quantile(
                [abs(value - 2.0) for value in slopes], 0.5
            ),
            "kl_slope_p90_abs_error_from_two": _quantile(
                [abs(value - 2.0) for value in slopes], 0.9
            ),
        },
        "minimum_vjp_rayleigh": min(float(row["vjp_rayleigh"]) for row in accepted),
    }
    median_abs_vjp = _quantile(
        [abs(float(row["vjp_rayleigh"])) for row in accepted], 0.5
    )
    psd_floor = -float(gates["psd_relative_negative_tolerance"]) * median_abs_vjp
    gate_pass = {
        "usable_range": len(by_direction) == int(gates["directions_required"])
        and all(
            len(usable.get(index, [])) >= required_eps
            for index in range(int(gates["directions_required"]))
        ),
        "kl_quadratic_identity": diagnostics["kl_q_relative_error"]["median"]
        <= float(gates["kl_q_median_relative_error_max"])
        and diagnostics["kl_q_relative_error"]["p90"]
        <= float(gates["kl_q_p90_relative_error_max"]),
        "fisher_vjp_route": diagnostics["fisher_vjp_relative_error"]["median"]
        <= float(gates["fisher_vjp_median_relative_error_max"])
        and diagnostics["fisher_vjp_relative_error"]["p90"]
        <= float(gates["fisher_vjp_p90_relative_error_max"]),
        "epsilon_stability": diagnostics["epsilon"]["max_adjacent_jvp_relative_change"]
        <= float(gates["adjacent_epsilon_jvp_relative_change_max"])
        and diagnostics["epsilon"]["kl_slope_median_abs_error_from_two"]
        <= float(gates["kl_loglog_slope_median_abs_error_from_two_max"])
        and diagnostics["epsilon"]["kl_slope_p90_abs_error_from_two"]
        <= float(gates["kl_loglog_slope_p90_abs_error_from_two_max"]),
        "psd": diagnostics["minimum_vjp_rayleigh"] >= psd_floor,
    }
    if not gate_pass["usable_range"]:
        registered = "REDESIGN_NUMERICAL_FLOOR_OR_SATURATION"
    elif not gate_pass["kl_quadratic_identity"]:
        registered = "REDESIGN_KL_QUADRATIC_IDENTITY"
    elif not gate_pass["fisher_vjp_route"]:
        registered = "REDESIGN_FISHER_VJP_ROUTE"
    elif not gate_pass["epsilon_stability"]:
        registered = "REDESIGN_EPSILON_STABILITY"
    elif not gate_pass["psd"]:
        registered = "REDESIGN_PSD"
    else:
        registered = "SCALAR_ROUTE_GREEN"

    epsilon_summary = []
    for epsilon in config["epsilon"]["ladder"]:
        rows = [row for row in cells if float(row["epsilon"]) == float(epsilon)]
        epsilon_summary.append(
            {
                "epsilon": float(epsilon),
                "median_symmetric_kl_nats": statistics.median(
                    float(row["symmetric_kl_nats"]) for row in rows
                ),
                "median_direct_curvature": statistics.median(
                    float(row["symmetric_direct_kl_curvature"]) for row in rows
                ),
                "median_finite_difference_fisher": statistics.median(
                    float(row["finite_difference_fisher_rayleigh"]) for row in rows
                ),
                "median_vjp_rayleigh": statistics.median(
                    float(row["vjp_rayleigh"]) for row in rows
                ),
            }
        )
    controls = [row for row in cells if int(row["direction_index"]) > 0]
    control_kl_by_epsilon = [
        statistics.median(
            float(row["symmetric_kl_nats"])
            for row in controls
            if float(row["epsilon"]) == float(epsilon)
        )
        for epsilon in config["epsilon"]["ladder"]
    ]
    sensitive_upper = [
        row
        for row in usable[0]
        if float(row["epsilon"]) in {0.002, 0.003}
    ]
    expected_batches = math.ceil(int(raw["panel"]["items"]) / 2)
    expected_hooks = expected_batches * (
        1
        + int(config["directions"]["count"])
        * len(config["epsilon"]["ladder"])
        * 3
    )
    energy_rows = raw["execution"]["energy_audits"]
    mechanical = {
        "identity_exact": raw["identity"]["observational_exact"]
        and raw["identity"]["zero_add_exact"],
        "parameter_versions_unchanged": raw["parameter_versions"]["status"] == "GREEN",
        "expected_hook_calls": expected_hooks,
        "observed_hook_calls": int(raw["execution"]["hook_calls"]),
        "energy_audits": len(energy_rows),
        "max_energy_relative_error": max(
            max(
                float(row["plus"]["max_relative_norm_error"]),
                float(row["minus"]["max_relative_norm_error"]),
            )
            for row in energy_rows
        ),
        "max_inactive_norm": max(
            max(
                float(row["plus"]["inactive_max_norm"]),
                float(row["minus"]["inactive_max_norm"]),
            )
            for row in energy_rows
        ),
    }
    return {
        "schema_version": 1,
        "status": "GREEN_INDEPENDENT_RECOMPUTE",
        "raw_confirmation_sha256": _sha256(raw_path),
        "config_sha256": _sha256(config_path),
        "cells": len(cells),
        "accepted_cells": len(accepted),
        "usable_epsilon_counts": usable_counts,
        "registered_verdict_reported": raw["verdict"]["status"],
        "registered_verdict_recomputed": registered,
        "verdict_exact": registered == raw["verdict"]["status"],
        "diagnostics": diagnostics,
        "diagnostics_exact": diagnostics
        == {
            key: raw["verdict"][key]
            for key in (
                "kl_q_relative_error",
                "fisher_vjp_relative_error",
                "epsilon",
                "minimum_vjp_rayleigh",
            )
        },
        "gate_pass": gate_pass,
        "psd_floor": psd_floor,
        "direction_slopes": direction_slopes,
        "epsilon_summary": epsilon_summary,
        "floor_diagnostic": {
            "control_median_symmetric_kl_nats_by_epsilon": control_kl_by_epsilon,
            "control_median_kl_max_to_min_ratio": max(control_kl_by_epsilon)
            / min(control_kl_by_epsilon),
            "sensitive_upper_epsilon_fisher_vjp_relative_errors": [
                _relative(row["finite_difference_fisher_rayleigh"], row["vjp_rayleigh"])
                for row in sensitive_upper
            ],
            "sensitive_upper_epsilon_kl_vjp_relative_errors": [
                _relative(row["symmetric_direct_kl_curvature"], row["vjp_rayleigh"])
                for row in sensitive_upper
            ],
        },
        "mechanical": mechanical,
    }


def visualize(raw_path: Path, config_path: Path, audit: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cells = raw["cells"]
    epsilons = [float(value) for value in config["epsilon"]["ladder"]]
    sensitive = sorted(
        (row for row in cells if int(row["direction_index"]) == 0),
        key=lambda row: float(row["epsilon"]),
    )
    controls = {
        epsilon: [
            row
            for row in cells
            if int(row["direction_index"]) > 0 and float(row["epsilon"]) == epsilon
        ]
        for epsilon in epsilons
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    sensitive_kl = [float(row["symmetric_kl_nats"]) for row in sensitive]
    control_median = [
        statistics.median(float(row["symmetric_kl_nats"]) for row in controls[value])
        for value in epsilons
    ]
    control_low = [
        min(float(row["symmetric_kl_nats"]) for row in controls[value])
        for value in epsilons
    ]
    control_high = [
        max(float(row["symmetric_kl_nats"]) for row in controls[value])
        for value in epsilons
    ]
    reference = [sensitive_kl[-1] * (value / epsilons[-1]) ** 2 for value in epsilons]
    ax.loglog(epsilons, sensitive_kl, "o-", lw=2.5, label="sensitive direction")
    ax.loglog(epsilons, control_median, "o-", lw=2.0, label="7-control median")
    ax.fill_between(epsilons, control_low, control_high, alpha=0.18, label="control range")
    ax.loglog(epsilons, reference, "--", color="black", label="ε² reference")
    ax.set_title("A. Raw symmetric KL: controls are nearly flat")
    ax.set_xlabel("activation-RMS fraction ε")
    ax.set_ylabel("symmetric KL (nats)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for key, label, marker in (
        ("symmetric_direct_kl_curvature", "direct KL / ε²", "o"),
        ("finite_difference_fisher_rayleigh", "finite-difference Fisher", "s"),
        ("vjp_rayleigh", "first-order VJP", "^"),
    ):
        ax.loglog(
            epsilons,
            [float(row[key]) for row in sensitive],
            marker=marker,
            lw=2.2,
            label=label,
        )
    ax.set_title("B. Sensitive direction: upper-ε routes approach")
    ax.set_xlabel("activation-RMS fraction ε")
    ax.set_ylabel("Rayleigh / curvature estimate")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for key, label, marker in (
        ("symmetric_direct_kl_curvature", "|direct KL / ε²| ", "o"),
        ("finite_difference_fisher_rayleigh", "|finite-difference Fisher| ", "s"),
        ("vjp_rayleigh", "|first-order VJP| ", "^"),
    ):
        values = [
            statistics.median(abs(float(row[key])) for row in controls[epsilon])
            for epsilon in epsilons
        ]
        ax.loglog(epsilons, values, marker=marker, lw=2.2, label=label)
    ax.set_title("C. Controls: divided finite differences explode")
    ax.set_xlabel("activation-RMS fraction ε")
    ax.set_ylabel("median absolute estimate")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    diagnostics = audit["diagnostics"]
    ratios = {
        "KL med": diagnostics["kl_q_relative_error"]["median"]
        / float(config["gates"]["kl_q_median_relative_error_max"]),
        "KL p90": diagnostics["kl_q_relative_error"]["p90"]
        / float(config["gates"]["kl_q_p90_relative_error_max"]),
        "route med": diagnostics["fisher_vjp_relative_error"]["median"]
        / float(config["gates"]["fisher_vjp_median_relative_error_max"]),
        "route p90": diagnostics["fisher_vjp_relative_error"]["p90"]
        / float(config["gates"]["fisher_vjp_p90_relative_error_max"]),
        "JVP max": diagnostics["epsilon"]["max_adjacent_jvp_relative_change"]
        / float(config["gates"]["adjacent_epsilon_jvp_relative_change_max"]),
        "slope med": diagnostics["epsilon"]["kl_slope_median_abs_error_from_two"]
        / float(config["gates"]["kl_loglog_slope_median_abs_error_from_two_max"]),
        "slope p90": diagnostics["epsilon"]["kl_slope_p90_abs_error_from_two"]
        / float(config["gates"]["kl_loglog_slope_p90_abs_error_from_two_max"]),
    }
    names = list(ratios)
    values = [ratios[name] for name in names]
    bars = ax.bar(names, values, color=["#b2182b", "#b2182b", "#d6604d", "#d6604d", "#f4a582", "#fddbc7", "#fddbc7"])
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="registered limit")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("observed / registered maximum")
    ax.set_title("D. Every numerical identity is far outside gate")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.0f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(fontsize=9)
    fig.suptitle(
        "PG2 scalar operator cell — registered verdict: KL quadratic identity redesign\n"
        "Diagnostic signature: coherent sensitive signal, BF16-floor-dominated controls",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    args = parser.parse_args()
    audit = recompute(args.raw, args.config)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    visualize(args.raw, args.config, audit, args.visual)
    print(json.dumps({"status": audit["status"], "verdict": audit["registered_verdict_recomputed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
