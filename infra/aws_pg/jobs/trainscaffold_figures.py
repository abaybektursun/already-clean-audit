"""TRAINSCAFFOLD figure suite (§10) — weight-free, from the metric JSONs (NO model). Each figure =
number + CI + null.
  F1 (headline): held-out RF_novel across the substrate ladder (B/A; HEALTHY) × format, CI bars, the
       R floor (0) + B ceiling (1) lines + the shuffle-twin point; title = the 4-way verdict.
  F3 (reachability): RF_MCQ — route-around rank ladder + refill vs refill-BLOCKED full-FT + non-ablated
       control.
  F4 (diagnostic): the emrand-normalized removed-dyad gradient fraction (carrier vs emrand).
Reads trainscaffold_metrics.json (F1) and optionally reach/graddiag jsons; draws what is present.

    python -m infra.aws_pg.jobs.trainscaffold_figures --out <dir> --unblind <metrics|s3> [--graddiag <..>] --sha <commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

OK_BLUE, OK_ORANGE, OK_GREEN, OK_GREY = "#0072B2", "#D55E00", "#009E73", "#777777"


def _load(path, out):
    if not path:
        return None
    if path.startswith("s3://"):
        loc = out / pathlib.Path(path).name
        subprocess.run(["aws", "s3", "cp", path, str(loc), "--region", "us-east-1"], check=True)
        path = str(loc)
    return json.loads(pathlib.Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unblind", default="")
    ap.add_argument("--graddiag", default="")
    ap.add_argument("--reach", default="", help="comma list of reach cell jsons (s3)")
    ap.add_argument("--collateral", default="", help="comma list of collateral_*.json (s3) — the headline ladder")
    ap.add_argument("--sha", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ub = _load(a.unblind, out)
    gd = _load(a.graddiag, out)
    foot = f"run · SHA {a.sha or '?'} · CONFIRMATORY (held-out generalization)"
    made = []

    # ---- F1: RF_novel across substrate ladder × format ----
    if ub:
        fig, ax = plt.subplots(figsize=(8, 5))
        formats = list(ub["by_format"].keys())
        subs = ["healthy", "deleted", "ablated"]
        cols = {"healthy": OK_GREY, "deleted": OK_BLUE, "ablated": OK_ORANGE}
        x = np.arange(len(formats))
        w = 0.25
        for si, sub in enumerate(subs):
            pts, los, his = [], [], []
            for fmt in formats:
                d = ub["by_format"][fmt].get(sub)
                if d:
                    rf = d["rf_novel"]
                    pts.append(rf["point"]); los.append(rf["point"] - rf["ci95"][0]); his.append(rf["ci95"][1] - rf["point"])
                else:
                    pts.append(np.nan); los.append(0); his.append(0)
            ax.bar(x + (si - 1) * w, pts, w, yerr=[los, his], capsize=3, color=cols[sub], alpha=0.9,
                   label=sub)
        ax.axhline(0, color="k", lw=0.8, label="R floor")
        ax.axhline(1, color=OK_GREEN, lw=0.8, ls=":", label="B ceiling")
        ax.axhline(0.5, color="#999999", lw=0.6, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(formats)
        ax.set_ylabel("held-out RF_novel  (0 = dead-backbone floor, 1 = matched ceiling)")
        ax.set_title(f"TRAINSCAFFOLD F1 — does the ablated backbone retain plasticity?\nverdict: {ub.get('named')}")
        ax.legend(fontsize=8, ncol=2)
        ax.text(0.01, -0.13, "null = R (random decoder); ceiling = B (matched deleted). " + foot,
                transform=ax.transAxes, fontsize=6)
        fig.savefig(out / "F1_rf_novel.png", bbox_inches="tight", dpi=150)
        fig.savefig(out / "F1_rf_novel.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append("F1_rf_novel")

    # ---- F4: gradient diagnostic ----
    if gd:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar([0, 1], [gd["carrier_frac_pooled"], gd["emrand_frac_pooled"]],
               color=[OK_ORANGE, OK_GREY], alpha=0.9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["removed top-σ", "emrand (null)"])
        ax.set_ylabel("MMLU-grad energy fraction in the dyads")
        ax.set_title(f"TRAINSCAFFOLD F4 — gradient routing (DISCOVERY)\ncontrast={gd['contrast']:+.4f} → {gd['prior']}")
        ax.text(0.01, -0.14, "DISCOVERY-only; sets the REACHABILITY prior. " + foot,
                transform=ax.transAxes, fontsize=6)
        fig.savefig(out / "F4_graddiag.png", bbox_inches="tight", dpi=150)
        fig.savefig(out / "F4_graddiag.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append("F4_graddiag")

    # ---- F0 (HEADLINE): collateral recognition ladder — the decisive forward-pass read ----
    if a.collateral:
        rows = []
        for p in [x.strip() for x in a.collateral.split(",") if x.strip()]:
            d = _load(p, out)
            lab = ("emrand" if d.get("emrand") else d["substrate"])
            rows.append((lab, float(d["mmlu_cloze_acc"])))
        order = {"healthy": 0, "deleted": 1, "emrand": 2, "ablated": 3}
        rows.sort(key=lambda r: order.get(r[0], 9))
        cols = {"healthy": OK_GREEN, "deleted": OK_BLUE, "emrand": OK_GREY, "ablated": OK_ORANGE}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(rows))
        ax.bar(x, [r[1] for r in rows], color=[cols.get(r[0], "#444") for r in rows], alpha=0.9)
        ax.axhline(25, color="k", lw=0.9, ls="--", label="chance (4-way)")
        for xi, (_, v) in zip(x, rows):
            ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
        ax.set_ylabel("in-distribution recognition (untouched-subject cloze, %)")
        ax.set_title("TRAINSCAFFOLD F0 — top-σ ablation BREAKS the general forward pass\n"
                     "only top-σ → chance (emrand at matched energy stays above) ⇒ "
                     "LATMECH 're-learnability scaffold' = general-criticality")
        ax.legend(fontsize=8)
        ax.text(0.01, -0.14, "null = chance (25); emrand = matched-energy random removal (specificity). "
                f"SHA {a.sha or '?'} · DIAGNOSTIC", transform=ax.transAxes, fontsize=6)
        fig.savefig(out / "F0_collateral_ladder.png", bbox_inches="tight", dpi=150)
        fig.savefig(out / "F0_collateral_ladder.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append("F0_collateral_ladder")

    (out / "figures.json").write_text(json.dumps({"made": made, "sha": a.sha}, indent=1))
    print(f"[TS-FIGURES] OK — {made}")


if __name__ == "__main__":
    main()
