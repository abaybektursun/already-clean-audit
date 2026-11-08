"""FAIRSCRATCH figure suite (F1/F2/F3) — weight-free, from banked JSON only.

Every panel illustrates a claim independently backed by a number + CI + null overlaid
(research-spec #4 figure discipline). PNG (review) + PDF (paper); provenance footer.

  F1 — access ladder: S_recover vs n for pg vs fair-scratch (r=512), δ_eq band + n=0 floor.
       Headline: does scratch close the gap with budget? (Contrast C, DISCOVERY.)
  F2 — matched-budget vs matched-loss: S_pg, S_scr* (matched budget), S_scr=CE (matched
       train-CE), paired-bootstrap CIs. Headline: is the edge optimization or generalization?
  F3 — recipe-fairness audit: last-epoch train-CE per candidate (fair-scratch ladder) vs
       W17's crippled random arm vs the pg arm, with the ln4 floor + label-fitting gate.
       The visual proof the M2 confound is closed.

    python -m infra.aws_pg.jobs.fairscratch_figures --out studies/lineage/paligemma2/figures \
        --metrics results/fairscratch/metrics.json --recipe results/fairscratch/fairscratch_recipe.json \
        --controls results/w17/controls.json --surface results/fairscratch/surface.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

CB = {"pg": "#0072B2", "scr": "#D55E00", "scr_ce": "#009E73", "null": "#999999", "band": "#CCCCCC"}


def _footer(fig, tag, src):
    fig.text(0.005, 0.005, f"FAIRSCRATCH · {tag} · src={src}", fontsize=5, color="#666666")


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    print(f"  [fig] {name}.png/.pdf")


def fig_f2(M, out):
    """S_pg vs S_scr* (matched budget) vs S_scr=CE (matched loss) with paired-bootstrap CIs."""
    import matplotlib.pyplot as plt
    import numpy as np
    g = M["prediction_grid"]
    deq = M["delta_eq_16k_recomputed"]
    labels, vals, los, his, cols = [], [], [], [], []
    labels.append("pg (deleted,\nmatched budget)"); vals.append(g["pg_S_seed_mean"]); cols.append(CB["pg"])
    los.append(0.0); his.append(0.0)
    labels.append("scratch*\n(matched budget)"); vals.append(g["scr_star_S_seed_mean"]); cols.append(CB["scr"])
    los.append(0.0); his.append(0.0)
    if g.get("scr_ce_S_seed_mean") is not None:
        labels.append("scratch\n(matched train-CE)"); vals.append(g["scr_ce_S_seed_mean"]); cols.append(CB["scr_ce"])
        los.append(0.0); his.append(0.0)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, width=0.6, zorder=3)
    ax.axhspan(-deq, deq, color=CB["band"], alpha=0.5, zorder=0, label=f"δ_eq band (±{deq:.3f})")
    ax.axhline(0, color=CB["null"], lw=1, ls="--", zorder=1, label="FLOOR (S=0)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("S_recover  (BEFORE=1, FLOOR=0)")
    dA = M["contrast_A_matched_budget"]["point"]
    title = (f"Matched-budget gap ΔS_A={dA:.3f}  →  {M['VERDICT']['verdict']}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    _footer(fig, M["VERDICT"]["verdict"], pathlib.Path(M.get("cells_dir", "")).name)
    _save(fig, out, "F2_matched_budget_vs_loss")
    plt.close(fig)


def fig_f3(recipe, controls, out):
    """Last-epoch train-CE per fair-scratch candidate vs W17 crippled random arm + pg, with ln4 gate."""
    import matplotlib.pyplot as plt
    import numpy as np
    ln4 = recipe.get("ln4", math.log(4.0))
    gate = recipe["gates"]["admissible_ce_max"]
    ce_pg = recipe["ce_pg"]
    ranked = sorted(recipe["ranked"], key=lambda c: c["ce_rnd"])
    ids = [c["recipe_id"] for c in ranked]
    ces = [c["ce_rnd"] for c in ranked]
    cols = [CB["scr"] if (c["admissible"] and c["safe"]) else CB["null"] for c in ranked]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(ids))
    ax.bar(x, ces, color=cols, width=0.6, zorder=3, label="fair-scratch candidate (CE)")
    ax.axhline(ln4, color="#000000", lw=1.2, ls="-", zorder=2, label=f"ln4 chance floor ({ln4:.3f})")
    ax.axhline(gate, color=CB["scr_ce"], lw=1.2, ls="--", zorder=2, label=f"label-fitting gate ({gate:.3f})")
    ax.axhline(ce_pg, color=CB["pg"], lw=1.2, ls=":", zorder=2, label=f"pg arm CE ({ce_pg:.3f})")
    # W17's crippled random arm (from controls' banked random_delta_sources 16k mean)
    w17_rand = controls.get("random_delta_sources", {})
    w17_16k = [v["loss"] for k, v in w17_rand.items() if k.startswith("16k")]
    if w17_16k:
        ax.axhline(float(np.mean(w17_16k)), color="#CC79A7", lw=1.2, ls="-.",
                   zorder=2, label=f"W17 crippled random ({np.mean(w17_16k):.3f})")
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("last-epoch train-CE (answer letter)")
    chosen = recipe.get("chosen")
    ax.set_title(f"Recipe-fairness audit — chosen: {chosen['recipe_id'] if chosen else recipe['status']}", fontsize=10)
    ax.legend(fontsize=7)
    _footer(fig, recipe["status"], pathlib.Path(recipe.get("controls", "")).name)
    _save(fig, out, "F3_recipe_fairness_audit")
    plt.close(fig)


def fig_f1(surface, deq, out):
    """S_recover vs n for pg vs fair-scratch (r=512), δ_eq band + n=0 floor."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm, col, lab in (("pg", CB["pg"], "pg (deleted)"), ("random", CB["scr"], "fair-scratch")):
        pts = sorted(surface.get(arm, []), key=lambda p: p["n"])
        if not pts:
            continue
        ns = [max(p["n"], 1) for p in pts]
        ss = [p["S"] for p in pts]
        ax.plot(ns, ss, "-o", color=col, label=lab, zorder=3)
    ax.axhspan(-deq, deq, color=CB["band"], alpha=0.5, zorder=0, label=f"δ_eq band (±{deq:.3f})")
    ax.axhline(0, color=CB["null"], lw=1, ls="--", zorder=1, label="FLOOR (S=0)")
    ax.set_xscale("log")
    ax.set_xlabel("relearn examples n (log)"); ax.set_ylabel("S_recover")
    ax.set_title("Access ladder — does scratch close the gap with budget? (DISCOVERY)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    _footer(fig, "DISCOVERY (contrast C)", "surface.json")
    _save(fig, out, "F1_access_ladder")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--surface", default=None, help="optional contrast-C surface.json for F1")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    M = json.loads(pathlib.Path(args.metrics).read_text())
    recipe = json.loads(pathlib.Path(args.recipe).read_text())
    controls = json.loads(pathlib.Path(args.controls).read_text())
    print("[FS-FIG] F2 + F3" + (" + F1" if args.surface else ""))
    fig_f2(M, out)
    fig_f3(recipe, controls, out)
    if args.surface:
        fig_f1(json.loads(pathlib.Path(args.surface).read_text()), M["delta_eq_16k_recomputed"], out)
    print("[FS-FIG] done")


if __name__ == "__main__":
    main()
