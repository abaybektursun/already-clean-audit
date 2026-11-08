"""Figures for SAV-C (prereg §10) — weight-free, reads savc_metrics.json only.

    python3 infra/pg/savc_figures.py --json studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json \
        --out studies/lineage/paligemma1-pi0/results/savc --sha <freeze-sha>

F1  overlap→S(16k) scatter — 18 units colored by {deep-consumed outlier (overlap<0.62) / baseline cluster /
    NEW gap-filler}, within-F region shaded, full-pool ρ + within-F ρ in the title. Between-regime trend
    clean; within-family scatter flat/noisy => the resolution chart.
F2  ΔLOO-R²(overlap|l2,lmc) — full-pool vs within-F vs scicap-leave-out, perm-null line at 0. The fragility
    chart (within-F sign-flips on dropping scicap).
F3  frob→overlap (did the frob selection axis fill the gap?) + the new-unit overlap distribution vs the
    previously-empty 0.62-0.82 gap.

Every panel: number + null overlaid, plain finding title, how-to-read, provenance footer. PNG+PDF.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_OUT = "#D55E00"     # vermillion — deep-consumed outliers (overlap<0.62)
C_CLU = "#0072B2"     # blue       — baseline near-task cluster
C_NEW = "#009E73"     # green      — NEW units (this study)
WITHIN_CUT = 0.62


def _footer(ax, txt):
    ax.annotate(txt, xy=(0, 0), xytext=(0, -44), textcoords="offset points", xycoords="axes fraction",
                fontsize=6, color="#666666", va="top")


def _region(u):
    if u["overlap"] <= WITHIN_CUT:
        return "out"
    return "new" if u.get("kind") == "n5c_new" else "clu"


def fig_f1(rep, out, sha):
    pool = rep["pool"]
    a = rep["analysis"]
    fp_rho = a["full_pool"]["S_16k"]["spearman_overlap"]
    wf_rho = a["within_family"]["S_16k"]["spearman_overlap"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvspan(WITHIN_CUT, 0.86, color="#f0f0f0", zorder=0, label="within-family (overlap>0.62)")
    for label, c, nm in [("out", C_OUT, "deep-consumed (overlap<0.62)"),
                         ("clu", C_CLU, "baseline cluster"), ("new", C_NEW, "NEW units (SAV-C)")]:
        xs = [u["overlap"] for u in pool if _region(u) == label]
        ys = [u["S_16k"] for u in pool if _region(u) == label]
        ax.scatter(xs, ys, c=c, s=70 if label == "new" else 45, label=nm,
                   edgecolor="black" if label == "new" else "none", linewidth=0.6, zorder=3)
    for u in pool:
        if _region(u) == "new":
            ax.annotate(u["unit"].replace("paligemma-3b-ft-", "").replace("-224", ""),
                        (u["overlap"], u["S_16k"]), fontsize=6, xytext=(4, -8),
                        textcoords="offset points", color="#333333")
    ax.set_xlabel("L9 representation overlap with gemma-2b base (cheap, one forward pass)")
    ax.set_ylabel("relearnability S(16k)  (1=recovered to base, 0=floor)")
    ax.set_title(f"F1 · SAV-C: overlap predicts relearnability BETWEEN regimes, not WITHIN a family\n"
                 f"full-pool ρ={fp_rho['rho']:+.3f} [{fp_rho['ci95'][0]:+.2f},{fp_rho['ci95'][1]:+.2f}]  ·  "
                 f"within-family ρ={wf_rho['rho']:+.3f} [{wf_rho['ci95'][0]:+.2f},{wf_rho['ci95'][1]:+.2f}] "
                 f"(CI brackets 0)\nHow to read: clean rising trend across the whole range, but a "
                 f"flat/noisy scatter inside the shaded family ⇒ no within-family resolution.", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    _footer(ax, f"n=18 (13 baseline + 5 new) · k=128 r512 · freeze={sha} · CONFIRMATORY/DISCOVERY")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F1_overlap_scatter.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_f2(rep, out, sha):
    a = rep["analysis"]
    pcr = rep.get("post_critic_robustness", {})
    rows = [("full-pool\noverlap|l2,lmc\n(n=18)", a["full_pool"]["S_16k"]["loo_overlap"], C_CLU),
            ("within-F\noverlap|l2,lmc\n(n=15, COLLINEAR)", a["within_family"]["S_16k"]["loo_overlap"], C_OUT)]
    if pcr:
        rows.append(("within-F\noverlap-ONLY\n(n=15)",
                     {"delta": pcr["overlap_only_within_f_loo_r2"], "p_perm": pcr["within_f_spearman_perm_p_2sided"]}, C_NEW))
    if "within_family_no_scicap" in a:
        rows.append(("within-F\nno-scicap (n=14)", a["within_family_no_scicap"]["S_16k"]["loo_overlap"], "#56B4E9"))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    xs = range(len(rows))
    ax.bar(xs, [r[1]["delta"] for r in rows], color=[r[2] for r in rows])
    ax.axhline(0, color="#888888", lw=1.0)
    for i, (lab, loo, _) in enumerate(rows):
        ax.annotate(f"Δ={loo['delta']:+.2f}\np={loo['p_perm']:.3f}", (i, loo["delta"]),
                    ha="center", va="bottom" if loo["delta"] >= 0 else "top",
                    fontsize=8, xytext=(0, 4 if loo["delta"] >= 0 else -4), textcoords="offset points")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("LOO-R²  (>0 = overlap adds out-of-sample power)")
    ax.set_title("F2 · SAV-C: overlap adds power BETWEEN regimes (Δ+0.86, p=.003); within-family is weak + not robust\n"
                 "the within-F overlap|l2,lmc bar (−1.97) is a COLLINEARITY artifact (corr −0.87, 88% scicap "
                 "fold) — the honest read is overlap-ONLY LOO +0.61 / Spearman p=.054 (fails the CI gate)\n"
                 "How to read: full-pool ≫0 = real gauge; within-family weak-positive overlap-only but "
                 "CI-insignificant ⇒ no robust within-family triage.", fontsize=7.5)
    _footer(ax, f"perm-null at 0 (n_perm=2000) · freeze={sha} · DISCOVERY")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F2_loo_fragility.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_f3(rep, out, sha):
    new = rep["new_units"]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.4))
    # frob -> overlap (did the selection axis populate the gap?)
    axl.scatter([u["frob_cached"] for u in new], [u["overlap"] for u in new], c=C_NEW, s=70,
                edgecolor="black", linewidth=0.6)
    for u in new:
        axl.annotate(u["unit"].replace("paligemma-3b-ft-", "").replace("-224", ""),
                     (u["frob_cached"], u["overlap"]), fontsize=7, xytext=(5, 0), textcoords="offset points")
    axl.set_xscale("log"); axl.set_xlabel("ΔW Frobenius (frob, the cached SELECTION axis, log)")
    axl.set_ylabel("L9 overlap"); axl.axhspan(0.62, 0.82, color="#fff0e0", zorder=0)
    axl.set_title("frob selection axis vs achieved overlap\n(shaded = the previously-EMPTY 0.62-0.82 gap)", fontsize=8)
    # overlap distribution: baseline vs new, the gap fill
    pool = rep["pool"]
    base_ov = [u["overlap"] for u in pool if u.get("kind") != "n5c_new"]
    new_ov = [u["overlap"] for u in new]
    axr.hist([base_ov, new_ov], bins=[0.45, 0.55, 0.62, 0.70, 0.78, 0.82, 0.86], stacked=True,
             color=[C_CLU, C_NEW], label=["13 baseline", "5 new (SAV-C)"])
    axr.axvspan(0.62, 0.82, color="#fff0e0", zorder=0)
    axr.set_xlabel("L9 overlap"); axr.set_ylabel("# units")
    axr.set_title("overlap distribution: the new units FILLED the gap\n(3 gap-fillers at 0.65/0.74/0.80 + 2 cluster)", fontsize=8)
    axr.legend(fontsize=7)
    fig.suptitle("F3 · SAV-C: the frob-spread selection populated the empty overlap gap as designed", fontsize=9)
    _footer(axl, f"freeze={sha} · DISCOVERY")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F3_gap_fill.{ext}", bbox_inches="tight", dpi=150)
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
