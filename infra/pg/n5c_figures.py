"""N5c figures (pi-trajectory N5 result). WEIGHT-FREE: reads only the frozen n5c_cloud.json (24 ft members
+ π0) + n5c_verdict.json — no models, no GPU. Each figure backs a number + the critic-folded interpretation.
Captions encode the result-critic qualifications (OUTSIDE carried by survives_outlier, not the gap; π0 is the
unique above-band member, not extreme on every axis; the task-distance gradient runs opposite to FT1-inflation).
GEOMETRY-ONLY. Renders locally (matplotlib from JSON is featherweight).

  python -m infra.pg.n5c_figures --cloud /tmp/n5c_cloud.json --out studies/lineage/paligemma1-pi0/figures/n5c_figures
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _slug(repo):
    return repo.split("/")[-1].replace("paligemma-3b-ft-", "").replace("-224", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.loads(pathlib.Path(a.cloud).read_text())
    members, pi0 = d["members"], d["pi0_member"]
    R = ("32", "64")

    # ---- Fig 1: the cloud distribution — π0 is the UNIQUE above-band member at both r ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, r in zip(axes, R):
        a_cl = sorted((m["A"][r]["a_pooled_exL17mlp"], _slug(m["repo_id"]), m["task_distance"]) for m in members)
        ys = [x[0] for x in a_cl]
        xs = list(range(len(ys)))
        band = float(np.mean([m["A"][r]["a_ub_exL17mlp"] for m in members]))   # survive band (≈constant)
        cols = ["#d2691e" if x[2] == "distant" else "#888888" for x in a_cl]
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.axhspan(band, 1.0, color="#cfe8cf", alpha=0.5, zorder=0)
        ax.axhline(band, color="#2e7d32", lw=1.2, ls="--", label=f"N1 survive band ≈{band:.3f}")
        ax.scatter(xs, ys, c=cols, s=45, zorder=3, label="ft cloud (orange = distant: refcoco-seg)")
        p = pi0["A"][r]["a_pooled_exL17mlp"]
        ax.scatter([len(ys) + 1], [p], marker="*", s=420, color="#c62828", zorder=4, label=f"π0 = {p:+.3f}")
        ax.annotate("π0 — the ONLY member\nabove the band", (len(ys) + 1, p), xytext=(len(ys) - 6, p - 0.18),
                    fontsize=9, color="#c62828", arrowprops=dict(arrowstyle="->", color="#c62828"))
        n_neg = sum(1 for v in ys if v < 0)
        ax.set_title(f"r={r}:  cloud median {np.median(ys):+.3f}   ({n_neg}/24 oppose, a<0)", fontsize=10)
        ax.set_xlabel("rank (sorted ft cloud)  →  π0")
        if r == "32":
            ax.set_ylabel("signed alignment  a_pooled_exL17mlp")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("N5c — π0 vs 24 same-init PaliGemma-1 fine-tunes  (GEOMETRY-ONLY)\n"
                 "π0 is the unique above-band member; OUTSIDE carried by survives_outlier, NOT a gap",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(out / "n5c_fig1_cloud_distribution.png", dpi=130)
    plt.close(fig)

    # ---- Fig 2: task-distance gradient (runs OPPOSITE to FT1-family inflation) ----
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    rng = np.random.default_rng(0)
    for grp, gx, col in (("near", 0, "#888888"), ("distant", 1, "#d2691e")):
        vals = [m["A"]["32"]["a_pooled_exL17mlp"] for m in members if m["task_distance"] == grp]
        ax.scatter(gx + 0.06 * rng.standard_normal(len(vals)), vals, c=col, s=55, zorder=3,
                   label=f"{grp} (n={len(vals)}, mean {np.mean(vals):+.3f})")
    ax.scatter([2], [pi0["A"]["32"]["a_pooled_exL17mlp"]], marker="*", s=420, color="#c62828",
               zorder=4, label=f"π0 = {pi0['A']['32']['a_pooled_exL17mlp']:+.3f}")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["near\n(VQA/caption)", "distant\n(refcoco-seg)", "π0"])
    ax.set_ylabel("signed alignment  a_pooled_exL17mlp (r=32)")
    ax.set_title("Task-distance gradient runs OPPOSITE to FT1-family inflation:\nnear family OPPOSES ΔW₁ "
                 "(mean −0.19); the lone distant task is the most positive. (n_distant=1 — NOT a powered test.)",
                 fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "n5c_fig2_task_distance.png", dpi=130); plt.close(fig)

    # ---- Fig 3: π0 is extreme on signed-alignment but NOT on every axis (ρ_L-enrichment below the cloud) ----
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax_a = [m["A"]["32"]["a_pooled_exL17mlp"] for m in members]
    ax_e = [m["rhoL_enrichment"]["32"] for m in members]
    ax.scatter(ax_a, ax_e, c="#888888", s=45, zorder=3, label="ft cloud")
    ax.scatter([pi0["A"]["32"]["a_pooled_exL17mlp"]], [pi0["rhoL_enrichment"]["32"]], marker="*", s=420,
               color="#c62828", zorder=4, label="π0")
    elo, ehi = min(ax_e), max(ax_e)
    ax.axhspan(elo, ehi, color="#e3f2fd", alpha=0.6, zorder=0, label=f"cloud ρ_L-enrichment band [{elo:.2f},{ehi:.2f}]")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed alignment  a_pooled_exL17mlp (r=32)")
    ax.set_ylabel("ρ_L enrichment  (ρ_L / Haar-p95)")
    ax.set_title("π0 is the extreme on SIGNED ALIGNMENT (x) but sits BELOW the cloud on\nspectral enrichment (y) "
                 f"— π0 enrichment {pi0['rhoL_enrichment']['32']:.2f} < cloud band. Not extreme on every axis.",
                 fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "n5c_fig3_multiaxis.png", dpi=130); plt.close(fig)

    print("N5C_FIGS_OK", [p.name for p in sorted(out.glob("n5c_fig*.png"))])


if __name__ == "__main__":
    main()
