"""PLENUM figures F1–F6 (spec §10). Weight-free: every figure is drawn from banked JSON.

`feedback_complete_sweep_visualization`: a swept parameter gets its WHOLE sweep drawn — every point,
every component curve (raw + diff + floor + CI + null), not the summary the verdict happened to
need. `feedback_post_critic_artifact_desync`: regenerate these AFTER any critic fold, or they keep
showing the pre-critic verdict.

  python -m infra.aws_pg.jobs.plenum_figures --out figs --agg <aggregate metrics.json> \
      [--cells <cells.json>] [--patch <patch metrics.json>] [--sae <sae metrics.json>]
"""
from __future__ import annotations

import argparse
import json
import pathlib

MARK = {"measured": "#2c7fb8", "impossible": "#d9d9d9", "VOID": "#e6550d",
        "rule_selected_out": "#f0e442", "pending": "#000000"}
LAB = {"retained": "#2c7fb8", "partial": "#fdae61", "consumed": "#d7191c"}


def _load(p):
    p = str(p)
    if p.startswith("s3://"):
        import subprocess
        return json.loads(subprocess.run(["aws", "s3", "cp", p, "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def f1_map(out, records):
    """F1 — the completed map: units × probes, coloured by TERMINAL STATE. The deliverable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from infra.pg import plenum_map as PM
    frozen = PM.all_cells()
    units = [u["id"] for u in PM.units()]
    probes = [p["id"] for p in PM.probes()]
    idx = {c["cell"]: c for c in frozen}
    states = list(MARK)
    M = np.zeros((len(units), len(probes)), dtype=int)
    for i, u in enumerate(units):
        for j, p in enumerate(probes):
            fz = idx[PM.cell_id(u, p)]
            M[i, j] = states.index(PM.state_of(fz, records.get(fz["cell"])))
    fig, ax = plt.subplots(figsize=(15, 11))
    cmap = matplotlib.colors.ListedColormap([MARK[s] for s in states])
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=len(states) - 1,
              interpolation="nearest")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(probes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels(units, fontsize=7)
    n_pending = int((M == states.index("pending")).sum())
    ax.set_title(f"F1 — the PLENUM map, {len(units)}×{len(probes)} = {M.size} cells "
                 f"(pending = {n_pending})", fontsize=11)
    ax.legend(handles=[Patch(facecolor=MARK[s], label=f"{s} ({int((M==k).sum())})")
                       for k, s in enumerate(states)],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F1_map.png", dpi=160)
    plt.close(fig)
    return {"n_cells": int(M.size), "pending": n_pending}


def f2_law(out, agg, sealed):
    """F2 — the survival grid (units × faculties) with the SEALED per-cell prediction overlaid as a
    hit/miss mark. Prediction and outcome in one frame: the point of a prediction ledger."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    # One faculty can have cells under BOTH readouts. Plot the one the LAW uses for that faculty
    # (the aggregator's mechanical routing), or the grid would silently show whichever readout
    # happened to be written last — including a VOIDed one.
    routing = agg.get("law_readout_per_faculty", {})
    # Film checkpoints enter the battery as `ckpt:<uri>` unit ids so they are measured by the SAME
    # code as the grid. They are points on a trajectory, not roster units — the aggregator already
    # keeps them out of the MAP, and the law figure must keep them out too or it grows a row per
    # checkpoint (46 rows instead of 32) and silently changes what the figure is OF.
    cells = [c for c in agg.get("cells", [])
             if c.get("label") is not None
             and not str(c.get("unit", "")).startswith("ckpt:")
             and (not routing or routing.get(c["faculty"]) == c.get("readout"))]
    if not cells:
        return {"skipped": "no behavioural cells with a label"}
    units = sorted({c["unit"] for c in cells})
    facs = sorted({c["faculty"] for c in cells})
    S = np.full((len(units), len(facs)), np.nan)
    L = {}
    for c in cells:
        S[units.index(c["unit"]), facs.index(c["faculty"])] = c["S"]
        L[(c["unit"], c["faculty"])] = c["label"]
    fig, ax = plt.subplots(figsize=(1.6 * len(facs) + 4, 0.42 * len(units) + 3))
    im = ax.imshow(S, aspect="auto", cmap="RdYlBu", vmin=0, vmax=1)
    for i, u in enumerate(units):
        for j, f in enumerate(facs):
            if np.isnan(S[i, j]):
                continue
            pred = (sealed or {}).get("predictions", {}).get(f"{u}|{f}", {}).get("PROGRAM")
            if pred is None:
                continue
            hit = (pred == L[(u, f)])
            ax.text(j, i, "✓" if hit else "✗", ha="center", va="center",
                    fontsize=8, color=("#111111" if hit else "#7a0000"),
                    fontweight=("normal" if hit else "bold"))
    ax.set_xticks(range(len(facs))); ax.set_xticklabels(facs, rotation=30, fontsize=8)
    ax.set_yticks(range(len(units))); ax.set_yticklabels(units, fontsize=7)
    law = agg.get("law", {})
    d = law.get("delta_loo_membership", {})
    ro = ", ".join(f"{k}:{v}" for k, v in sorted(routing.items())) or "forced choice"
    ax.set_title(f"F2 — survival S with sealed PROGRAM predictions  [readout per faculty: {ro}]\n"
                 f"ΔLOO(membership | ‖ΔW‖, similarity) = {d.get('delta', float('nan')):.4f}  "
                 f"p_perm = {d.get('p_perm', float('nan')):.4f}  "
                 f"H_recipe supported = {law.get('H_recipe_supported')}", fontsize=10)
    fig.colorbar(im, ax=ax, label="S = (score − FLOOR)/(CEILING − FLOOR)")
    ax.legend(handles=[Patch(facecolor="w", edgecolor="k", label="✓ sealed prediction hit"),
                       Patch(facecolor="w", edgecolor="k", label="✗ miss")],
              loc="upper left", bbox_to_anchor=(1.18, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F2_law.png", dpi=160)
    plt.close(fig)
    return {"units": len(units), "faculties": len(facs)}


def f5_mechanism(out, patch):
    """F5 — the WHOLE depth sweep: ΔS at every layer, every dose, every position block, with the
    same-depth-count null band drawn behind it (the complete-sweep rule)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cells = [c for c in patch.get("cells", []) if c.get("arm") == "locus"
             and c.get("kind") == "patched"]
    nulls = [c for c in patch.get("cells", []) if c.get("arm") == "locus_null_n3"]
    if not cells:
        return {"skipped": "no locus cells"}
    tasks = sorted({c["task"] for c in cells})
    doses = sorted({c["dose"] for c in cells})
    fig, axes = plt.subplots(len(tasks), len(doses), squeeze=False,
                             figsize=(4.6 * len(doses), 3.4 * len(tasks)), sharey=True)
    for r, task in enumerate(tasks):
        for k, dose in enumerate(doses):
            ax = axes[r][k]
            nd = [c["dS"] for c in nulls if c["task"] == task and c["dose"] == dose]
            if nd:
                lo, hi = float(np.percentile(nd, 2.5)), float(np.percentile(nd, 97.5))
                ax.axhspan(lo, hi, color="#cccccc", alpha=0.7,
                           label=f"N3 same-depth-count null (n={len(nd)})")
            for pk, col in (("all", "#2c7fb8"), ("last", "#d95f02"), ("image", "#1b9e77")):
                pts = sorted([c for c in cells if c["task"] == task and c["dose"] == dose
                              and c["positions"] == pk], key=lambda c: c["window"][0])
                if not pts:
                    continue
                x = [c["window"][0] for c in pts]
                y = [c["dS"] for c in pts]
                lo = [c["dS"] - c["dS_lo95"] for c in pts]
                hi = [c["dS_hi95"] - c["dS"] for c in pts]
                ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=3, lw=1.2, capsize=2,
                            color=col, label=f"positions={pk}")
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(f"{task} · dose={dose}", fontsize=9)
            ax.set_xlabel("window start layer")
            if k == 0:
                ax.set_ylabel("ΔS (patched − native π0.5)")
            ax.legend(fontsize=6)
    fig.suptitle("F5 — PG1→π0.5 depth-patch sweep: every layer, every dose, every position block",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "F5_mechanism.png", dpi=160)
    plt.close(fig)
    return {"n_locus_cells": len(cells), "n_null_draws": len(nulls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--sae", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    made = {}
    records = _load(a.cells) if a.cells else {}
    made["F1"] = f1_map(out, records)
    if a.agg:
        agg = _load(a.agg)
        made["F2"] = f2_law(out, agg, _load(a.predictions) if a.predictions else None)
    if a.patch:
        made["F5"] = f5_mechanism(out, _load(a.patch))
    (out / "figures_manifest.json").write_text(json.dumps(made, indent=2, default=str))
    print("PLENUM_FIGURES_OK", json.dumps(made, default=str)[:400], flush=True)


if __name__ == "__main__":
    main()
