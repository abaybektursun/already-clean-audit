"""REPRICE figures — weight-free, reads only banked cell/preflight JSONs.

Headline: per-axis recovery curves (MCQ letter vs cloze content) across the data
ladder, with ceiling/floor reference lines. The content axis being flat while the
letter axis climbs is the H_head result.

    python -m infra.pg.reprice_figures --discovery <dir> --confirmation <dir> --out fig.png
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib


def _load(run_dir):
    pf = json.load(open(glob.glob(f"{run_dir}/**/preflight.json", recursive=True)[0]))
    b = pf["baselines_blankimg"]
    cells = {}
    for f in glob.glob(f"{run_dir}/**/cell_*.json", recursive=True):
        c = json.load(open(f))
        cells[(c["arm"], c["rank"], c["n"])] = c
    budgets = [0, 1000, 4000, 16000]
    mcq = [cells[("cap", 512, n)]["mcq"]["acc_pct"] for n in budgets]
    cloze = [cells[("cap", 512, n)]["cloze"]["acc_pct"] for n in budgets]
    return {"budgets": budgets, "mcq": mcq, "cloze": cloze,
            "ceil_mcq": b["ceiling"]["mcq_acc_pct"], "ceil_cloze": b["ceiling"]["cloze_acc_pct"],
            "n": cells[("cap", 512, 16000)]["mcq"]["n_items"]}


def _panel(ax, d, title):
    x = list(range(len(d["budgets"])))
    ax.plot(x, d["mcq"], "o-", color="#1f4e8c", lw=2.2, ms=7, label="MCQ (letter readout)")
    ax.plot(x, d["cloze"], "s-", color="#d2691e", lw=2.2, ms=7, label="cloze (answer content)")
    ax.axhline(d["ceil_mcq"], ls="--", color="#1f4e8c", alpha=0.5, lw=1.2)
    ax.axhline(d["ceil_cloze"], ls="--", color="#d2691e", alpha=0.5, lw=1.2)
    ax.axhline(25, ls=":", color="gray", alpha=0.6, lw=1, label="chance (25%)")
    ax.text(len(x) - 1, d["ceil_mcq"] + 0.4, "MCQ ceiling", color="#1f4e8c", fontsize=8, ha="right")
    ax.text(len(x) - 1, d["ceil_cloze"] - 1.2, "cloze ceiling", color="#d2691e", fontsize=8, ha="right")
    fm, fc = d["mcq"][0], d["cloze"][0]
    rm = (d["mcq"][-1] - fm) / (d["ceil_mcq"] - fm) * 100
    rc = (d["cloze"][-1] - fc) / (d["ceil_cloze"] - fc) * 100
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1k", "4k", "16k"])
    ax.set_xlabel("relearning data budget (examples)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{title}  (n={d['n']})\nrecovers @16k:  MCQ {rm:+.0f}%   cloze {rc:+.0f}%", fontsize=10)
    ax.set_ylim(20, 50)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    _panel(axes[0], _load(a.confirmation), "SEALED CONFIRMATION (is_confirmation)")
    _panel(axes[1], _load(a.discovery), "discovery bucket (is_grid_subset)")
    fig.suptitle("REPRICE: the letter readout recovers; the content does not — relearning re-prices "
                 "the head, not the knowledge", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
