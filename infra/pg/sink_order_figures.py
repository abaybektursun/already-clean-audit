"""Figures for sink-order (prereg §10) — weight-free, reads sink_order.json only.

    python3 infra/pg/sink_order_figures.py \
        --json studies/lineage/paligemma1-pi0/results/sink_order/sink_order.json \
        --out studies/lineage/paligemma1-pi0/results/sink_order --sha <freeze-sha>

F1  DOSE-RESPONSE — position-corr@L9 (and Δppl) vs the achieved text→BOS sink mass, for the SINK-INDUCED
    arm vs the MATCHED-ATTENTION-MASS null, with baseline + U_unt-ablation levels overlaid and the
    headline-level position Excess CI annotated. The naming chart: if the blue (sink) line drops below the
    orange (matched-mass null) line as BOS mass rises, the sink is sink-specific.
F2  ARM LADDER at the headline level — position-corr / Δppl / copy-acc for {baseline, matched-null,
    sink-induced, U_unt-ablation}, so a GENERIC attention hit (sink≈null) is visually distinct from a
    SINK-SPECIFIC one (sink between null and ablation). Excess CI + θ in each title.
F3  MEDIATION + SUPPRESSION — the mediation fraction D_sink/D_abl (position) ± CI vs the matched-null's own
    fraction; and the suppression arm (U_unt-ablation → +suppress-BOS) recovery of position-corr / Δppl.

Every panel: CI / null overlaid, a plain finding title, a "how to read" line, a provenance footer. PNG+PDF.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colourblind-safe (Wong)
C_BASE = "#000000"     # black      — baseline
C_SINK = "#0072B2"     # blue       — sink-induced (the treatment)
C_NULL = "#D55E00"     # vermillion — matched-attention-mass null (the decisive comparator)
C_ABL = "#009E73"      # green      — U_unt-ablation (full-damage reference)


def _footer(ax, txt):
    ax.annotate(txt, xy=(0, 0), xytext=(0, -46), textcoords="offset points", xycoords="axes fraction",
                fontsize=6, color="#666666", va="top")


def _prov(rep, sha):
    return (f"L{rep['headline_layer']} · k={rep['k']} · n_ft={rep.get('n_ft','?')} · "
            f"base_sha={str(rep.get('base_sha','?'))[:8]} · freeze={sha} · CONFIRMATORY")


def fig_f1(rep, out, sha):
    sink_dose = sorted(rep["dose_response"]["sink"], key=lambda d: d["sink_bos_mass"])
    null_dose = sorted(rep["dose_response"]["null"], key=lambda d: d["null_div_mass"])
    h = rep["headline"]
    pc = rep["positive_control"]
    base_corr = rep["base_bands"][str(rep["headline_layer"])]["corr_clean"]
    fig, (axp, axl) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # position-corr vs achieved BOS mass
    xs = [d["sink_bos_mass"] for d in sink_dose]
    axp.plot(xs, [d["sink_pos_corr"] for d in sink_dose], "o-", color=C_SINK, label="sink-induced (→BOS)")
    axp.plot([d["null_div_mass"] for d in null_dose], [d["null_pos_corr"] for d in null_dose], "s--",
             color=C_NULL, label="matched-mass null (→random text)")
    axp.axhline(base_corr, color=C_BASE, lw=0.9, ls=":", label=f"baseline ({base_corr:.2f})")
    axp.axhline(pc["pos_corr"], color=C_ABL, lw=1.2, ls="-.", label=f"U_unt-ablation ({pc['pos_corr']:.2f})")
    axp.axvline(pc["sink_mass"], color=C_ABL, lw=0.8, ls=":", alpha=0.6)
    axp.set_xlabel("achieved text→BOS sink mass (induced)")
    axp.set_ylabel("position-corr @ L%d (higher = decodable)" % rep["headline_layer"])
    ex = h["pos_excess"]
    axp.set_title(f"position vs induced sink mass\nheadline Excess(null−sink)={ex['contrast']:+.3f} "
                  f"[{ex['lo']:+.3f},{ex['hi']:+.3f}] θ={h['theta']:.3f}", fontsize=8)
    axp.legend(fontsize=6, loc="best")

    # Δppl vs achieved BOS mass
    axl.plot(xs, [d["sink_dppl"] for d in sink_dose], "o-", color=C_SINK, label="sink-induced")
    axl.plot([d["null_div_mass"] for d in null_dose], [d["null_dppl"] for d in null_dose], "s--",
             color=C_NULL, label="matched-mass null")
    axl.axhline(pc["dppl"], color=C_ABL, lw=1.2, ls="-.", label=f"U_unt-ablation (+{pc['dppl']:.0f})")
    axl.set_xlabel("achieved text→BOS sink mass (induced)")
    axl.set_ylabel("Δ wikitext ppl vs baseline")
    axl.set_title(f"perplexity vs induced sink mass\nheadline ppl Excess={h['ppl_excess']['contrast']:+.2f} "
                  f"[{h['ppl_excess']['lo']:+.2f},{h['ppl_excess']['hi']:+.2f}]", fontsize=8)
    axl.legend(fontsize=6, loc="best")

    fig.suptitle(f"F1 · sink-order: does inducing attention concentration crater position? — verdict "
                 f"{rep['verdict']['named']} (concentration, NOT BOS-specific)\n"
                 f"How to read: BOTH the BOS sink (blue) and the non-BOS matched-mass null (orange) crater "
                 f"position as mass concentrates ⇒ not BOS-exclusive. Matched-mass ranking is "
                 f"violence-confounded (null needs a far larger bias; see result doc).", fontsize=9)
    _footer(axp, _prov(rep, sha))
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F1_dose_response.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_f2(rep, out, sha):
    h = rep["headline"]
    pc = rep["positive_control"]
    L = str(rep["headline_layer"])
    base_corr = rep["base_bands"][L]["corr_clean"]
    # headline arm values for each readout
    di = next(d for d in rep["dose_response"]["sink"] if d["b"] == h["b_sink"])
    dn = next(d for d in rep["dose_response"]["null"] if d["b"] == h["b_null"])
    panels = [
        ("position-corr@L%d" % rep["headline_layer"],
         [base_corr, dn["null_pos_corr"], di["sink_pos_corr"], pc["pos_corr"]],
         f"Excess={h['pos_excess']['contrast']:+.3f} θ={h['theta']:.3f}", "higher = decodable"),
        ("Δ wikitext ppl",
         [0.0, dn["null_dppl"], di["sink_dppl"], pc["dppl"]],
         f"Excess={h['ppl_excess']['contrast']:+.2f}", "higher = worse"),
        ("copy-acc",
         [rep["copy"]["base"], rep["copy"]["null"], rep["copy"]["sink"], rep["copy"]["abl"]],
         f"Excess={rep['copy']['excess']['contrast']:+.3f}", "higher = intact"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    labels = ["baseline", "matched-\nnull", "sink-\ninduced", "U_unt-\nablation"]
    colors = [C_BASE, C_NULL, C_SINK, C_ABL]
    for ax, (title, vals, ex, howread) in zip(axes, panels):
        ax.bar(range(4), vals, color=colors)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{title}\n{ex}", fontsize=8)
        ax.axhline(vals[0], color="#bbbbbb", lw=0.8, ls=":")
        ax.annotate(howread, xy=(0, 1), xytext=(0, 8), textcoords="offset points",
                    xycoords="axes fraction", fontsize=6, color="#666666")
    fig.suptitle(f"F2 · sink-order arm ladder at the headline level (sink BOS mass "
                 f"{h['sink_bos_mass']:.2f} ≈ ablation {h['abl_sink_mass']:.2f}) — verdict "
                 f"{rep['verdict']['named']}\nHow to read: sink bar ≈ matched-null ⇒ GENERIC; sink between "
                 f"null and ablation ⇒ SINK-SPECIFIC partial; sink ≈ ablation ⇒ sink-PRIMARY.", fontsize=9)
    _footer(axes[0], _prov(rep, sha))
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F2_arm_ladder.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_f3(rep, out, sha):
    h = rep["headline"]
    sup = rep["suppression"]
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(11, 4.3))

    # mediation fraction: sink vs matched-null (of the U_unt-ablation position crater)
    med = h["mediation"]
    nself = h["null_self_mediation"]
    med_lo = [med["fraction"] - med["lo"], nself["fraction"] - nself["lo"]]
    med_hi = [med["hi"] - med["fraction"], nself["hi"] - nself["fraction"]]
    axm.bar([0, 1], [med["fraction"], nself["fraction"]], color=[C_SINK, C_NULL],
            yerr=[med_lo, med_hi], capsize=5)
    axm.set_xticks([0, 1]); axm.set_xticklabels(["sink-induced", "matched-null"], fontsize=8)
    axm.axhline(0.5, color="#888888", lw=0.8, ls="--")
    axm.axhline(0.25, color="#cccccc", lw=0.8, ls=":")
    axm.set_ylabel("mediation fraction D_arm / D_ablation (position)")
    axm.set_title(f"how much of the ablation crater each arm reproduces\nsink={med['fraction']:.0%} "
                  f"[{med['lo']:.0%},{med['hi']:.0%}] (≥50% primary, <25% inert)", fontsize=8)

    # suppression: U_unt-ablation -> +suppress BOS, position-corr & Δppl recovery
    axs2 = axs.twinx()
    axs.bar([0, 1], [sup["abl_pos_corr"], sup["pos_corr"]], width=0.35, color=C_ABL, label="position-corr")
    axs2.bar([0.45, 1.45], [sup["abl_dppl"], sup["dppl"]], width=0.35, color=C_SINK, label="Δppl")
    axs.set_xticks([0.22, 1.22]); axs.set_xticklabels(["U_unt-ablation", "ablation\n+suppress BOS"], fontsize=8)
    axs.set_ylabel("position-corr (green)")
    axs2.set_ylabel("Δppl (blue)")
    axs.set_title(f"necessity (secondary): does suppressing the sink recover?\n"
                  f"sink {sup.get('sink_mass', float('nan')):.2f}; pos recovered={sup['pos_recovered']}, "
                  f"ppl recovered={sup['ppl_recovered']}", fontsize=8)

    fig.suptitle(f"F3 · sink-order mediation + suppression — verdict {rep['verdict']['named']}\n"
                 f"How to read: sink mediation ≥0.5 ⇒ sink-primary; ≪ matched-null ⇒ not sink-specific; "
                 f"suppression recovery corroborates the sufficiency direction.", fontsize=9)
    _footer(axm, _prov(rep, sha))
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F3_mediation_suppression.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sha", default="?")
    a = ap.parse_args()
    rep = json.loads(pathlib.Path(a.json).read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_f1(rep, out, a.sha)
    fig_f2(rep, out, a.sha)
    fig_f3(rep, out, a.sha)
    print(f"wrote F1/F2/F3 (png+pdf) to {out}")


if __name__ == "__main__":
    main()
