"""LATMECH figure suite (§10) — weight-free, from cell JSONs + latmech_metrics.json only.

Each figure is a number + CI + null (never evidence on its own). Reads the banked per-item arrays
(no model, no GPU); runs as a CPU pg job (matplotlib is on the AMI) or anywhere the deps exist.

  F1 (headline): RF_MCQ(16k) per weight condition with the energy-matched-random null overlaid —
       the double-dissociation bar chart (content vs emrand vs shrink_perturb), bootstrap CI bars,
       title = the §6 verdict in plain language.
  F2: RF_MCQ vs the data-budget ladder per condition (the re-entry curves; floor/ceiling anchored).

    python -m infra.aws_pg.jobs.latmech_figures --cells <dir> --metrics <latmech_metrics.json> \
        --split confirmation --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _load_cells(cells_dir, split, arm="cap"):
    by = {}
    for p in sorted(pathlib.Path(cells_dir).rglob("cell_*.json")):
        r = json.loads(p.read_text())
        if r["eval_split"] != split or r["arm"] != arm:
            continue
        by[(r["condition_info"]["condition"], int(r["n"]), int(r["seed"]))] = r
    return by


def _fetch_cells_s3(prefixes, dest):
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    for pre in prefixes:
        subprocess.run(["aws", "s3", "cp", pre, str(dest), "--recursive",
                        "--exclude", "*", "--include", "*cell_*.json", "--region", "us-east-1"], check=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--cells-s3", default="", help="comma-list of S3 prefixes to pull cell_*.json from")
    ap.add_argument("--metrics", default="", help="latmech_metrics.json (for the verdict in F1 title)")
    ap.add_argument("--metrics-s3", default="", help="s3 path to latmech_metrics.json")
    ap.add_argument("--split", default="confirmation", choices=["bucket", "confirmation"])
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=1337, help="seed whose cells anchor the figures")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cells_s3:
        args.cells = str(_fetch_cells_s3([p.strip() for p in args.cells_s3.split(",")], out / "cells"))
    if args.metrics_s3:
        import subprocess
        args.metrics = str(out / "metrics.json")
        subprocess.run(["aws", "s3", "cp", args.metrics_s3, args.metrics, "--region", "us-east-1"], check=True)
    assert args.cells, "need --cells or --cells-s3"

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from infra.pg import reprice_stats as RS

    cells = _load_cells(args.cells, args.split)
    metrics = json.loads(pathlib.Path(args.metrics).read_text()) if args.metrics else {}
    verdict = metrics.get("verdict", "verdict pending")
    s, B = args.seed, args.budget

    def rf_ci(cond_S, floor_S, ceil_S):
        b = RS.bootstrap_recovery_fraction(cond_S, floor_S, ceil_S, n=args.boot, seed=args.seed)
        return b["point"], b["lo"], b["hi"]

    # ---- F1: RF_MCQ(16k) per condition, emrand null overlaid, CI bars ----
    ceil_S = cells[("deleted", B, s)]["mcq"]["per_item"]
    floor_S = cells[("random_floor", B, s)]["mcq"]["per_item"]
    conds = [("deleted", "deleted"), ("content", "content-ablated"),
             ("emrand", "emrand (null)"), ("shrink_perturb", "shrink-perturb")]
    labels, pts, los, his, colors = [], [], [], [], []
    palette = {"deleted": "#0072B2", "content": "#D55E00", "emrand": "#999999",
               "shrink_perturb": "#009E73"}
    for key, lab in conds:
        if (key, B, s) not in cells:
            continue
        p, lo, hi = rf_ci(cells[(key, B, s)]["mcq"]["per_item"], floor_S, ceil_S)
        labels.append(lab); pts.append(p); los.append(p - lo); his.append(hi - p)
        colors.append(palette[key])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, pts, yerr=[los, his], capsize=5, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="#0072B2", lw=0.8, ls=":", label="ceiling (deleted)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(f"RF_MCQ(n={B})  (1=deleted ceiling, 0=random floor)")
    ax.set_title(f"LATMECH F1 — re-entry recovery by weight condition\nverdict: {verdict}")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, f"split={args.split} seed={s} boot={args.boot} | content vs emrand = the MDE",
             fontsize=6)
    fig.savefig(out / "F1_mde_recovery.png", bbox_inches="tight", dpi=150)
    fig.savefig(out / "F1_mde_recovery.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- F2: RF_MCQ vs data-budget ladder per condition ----
    budgets = sorted({n for (_, n, _) in cells})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lab in [("deleted", "deleted"), ("content", "content-ablated"),
                     ("emrand", "emrand (null)"), ("random_floor", "random floor")]:
        xs, ys, yl, yh = [], [], [], []
        for n in budgets:
            if (key, n, s) not in cells or ("deleted", n, s) not in cells or ("random_floor", n, s) not in cells:
                continue
            cS = cells[(key, n, s)]["mcq"]["per_item"]
            fS = cells[("random_floor", n, s)]["mcq"]["per_item"]
            ceS = cells[("deleted", n, s)]["mcq"]["per_item"]
            p, lo, hi = rf_ci(cS, fS, ceS)
            xs.append(n); ys.append(p); yl.append(p - lo); yh.append(hi - p)
        if xs:
            ax.errorbar(xs, ys, yerr=[yl, yh], marker="o", capsize=3, label=lab,
                        color=palette.get(key, "#444444"))
    ax.set_xscale("symlog"); ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="#0072B2", lw=0.6, ls=":")
    ax.set_xlabel("relearning budget n (examples)"); ax.set_ylabel("RF_MCQ (floor/ceiling anchored)")
    ax.set_title("LATMECH F2 — re-entry recovery vs data budget")
    ax.legend(fontsize=8)
    fig.savefig(out / "F2_ladder.png", bbox_inches="tight", dpi=150)
    fig.savefig(out / "F2_ladder.pdf", bbox_inches="tight")
    plt.close(fig)

    (out / "figures.json").write_text(json.dumps(
        {"verdict": verdict, "split": args.split, "seed": s, "budget": B,
         "figures": ["F1_mde_recovery", "F2_ladder"]}, indent=1))
    print(f"[LATMECH-FIGURES] OK — F1+F2 written ({args.split}, seed {s}, verdict {verdict})")


if __name__ == "__main__":
    main()
