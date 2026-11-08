"""W16 figure suite — weight-free from results/w16/{metrics,controls}.json (spec §3.4).

Four panels, one claim each, every number+CI+null in metrics.json; null overlaid in-panel;
provenance footer (run id · commit SHA · artifact sha256 · tag); Okabe–Ito colourblind-safe
palette; PNG + PDF; bbox_inches="tight". Run on the control plane (no GPU/weights):

    python -m infra.pg.w16_figures
"""
from __future__ import annotations

import json
import pathlib
import subprocess

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "sky": "#56B4E9"}
ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "studies/lineage/paligemma2/figures/w16_figures"


def _footer(ax, m, tag):
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h"],
                         capture_output=True, text=True, check=True).stdout.strip()
    ax.figure.text(0.01, 0.005,
                   f"{tag} · unblind {m['run']['prefix']} ({m['run']['instance']}) · "
                   f"commit {sha} · s3 sha256 {m['s3_full_sha256'][:12]}",
                   fontsize=6, color="#555555")


def _save(fig, name, index, claim, tag):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    index[name] = {"claim": claim, "tag": tag, "files": [f"{name}.png", f"{name}.pdf"]}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = json.loads((ROOT / "results/w16/metrics.json").read_text())
    c = json.loads((ROOT / "results/w16/controls.json").read_text())
    index: dict = {}

    # --- F1 headline: AFTER probe grid vs frozen FLOOR null band + confirmation kill ----
    s = m["screening"]
    cells = sorted(s["per_cell_delta_s"], key=lambda k: s["per_cell_delta_s"][k])
    vals = [s["per_cell_delta_s"][k] for k in cells]
    band = c["screening_null_band"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(cells)), vals,
           color=[OI["red"] if k == s["argmax_cell"] else OI["blue"] for k in cells])
    ax.axhline(band["band_hi"], color=OI["grey"], ls="--",
               label=f"frozen FLOOR null band p95 = {band['band_hi']:.3f}")
    ax.axhline(band["p50"], color=OI["grey"], ls=":", label=f"null p50 = {band['p50']:.3f}")
    conf = m["confirmation"]
    ax.scatter([cells.index(conf["cell"])], [conf["stat_seed_mean"]], marker="X", s=120,
               color=OI["purple"], zorder=5,
               label=f"int4 CONFIRMATION (7,453 items): ΔS = {conf['stat_seed_mean']:.3f}, p = {conf['p_one_sided']:.2f}")
    ax.set_xticks(range(len(cells)), cells, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ΔS_text (probed − unprobed, FLOOR-scale)")
    ax.set_title("No probe cell survives screen→confirm: int4's screening spike "
                 f"(ΔS={s['max']:.3f}, p={s['p_one_sided']:.3f}) dies at confirmation — H_deletion stands")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(0.99, 0.02, "How to read: bars = AFTER grid-bucket cells (exploratory); "
            "X = the only verdict-bearing read.", transform=ax.transAxes,
            fontsize=6.5, ha="right", color="#444444")
    _footer(ax, m, "CONFIRMATORY (X) / DISCOVERY (bars)")
    _save(fig, "f1_probe_grid_vs_null", index,
          "Zero-gradient probes do not resurrect capability (confirmation ΔS=−0.013, p=0.98)",
          "CONFIRMATORY")
    plt.close(fig)

    # --- F2: Arm A paired diffs with CIs vs 0 (the metric-induced trigger) ----
    a = m["arm_a"]
    bh = m["bh_family"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, point, ci, padj, ylab in (
            (axes[0], "armA_margin", a["margin_diff_point"], a["margin_diff_ci95"],
             bh["armA_margin"]["p_adj"], "mean margin diff (nats)"),
            (axes[1], "armA_auc", a["auc_diff_point"], a["auc_diff_ci95"],
             bh["armA_auc"]["p_adj"], "rank-AUC diff")):
        ax.errorbar([0], [point], yerr=[[point - ci[0]], [ci[1] - point]],
                    fmt="o", color=OI["blue"], capsize=6, ms=8)
        ax.axhline(0, color=OI["grey"], ls="--", label="null (AFTER ≡ FLOOR)")
        ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_ylabel(ylab)
        ax.set_title(f"p_adj = {padj:.4f} ({'sig' if bh[key]['significant'] else 'ns'})", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Arm A (confirmation half, paired AFTER−FLOOR): margin fires the frozen rule; "
                 "rank-AUC does not", fontsize=10)
    _footer(axes[0], m, "CONFIRMATORY")
    _save(fig, "f2_arm_a_paired_diffs", index,
          "metric-induced (graded): margin diff +1.70 [1.63,1.77] BH-sig; AUC diff ns", "CONFIRMATORY")
    plt.close(fig)

    # --- F3: dispersion characterization (post-critic descriptive) ----
    d = m["post_critic_descriptive"]
    qa = d["after_margin_quantiles_conf"]; qf = d["floor_margin_quantiles_conf"]
    ps = ["0.05", "0.25", "0.5", "0.75", "0.95"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot([float(p) for p in ps], [qa[p] for p in ps], "o-", color=OI["blue"], label="AFTER")
    ax.plot([float(p) for p in ps], [qf[p] for p in ps], "s-", color=OI["orange"], label="FLOOR (1337)")
    ax.axhline(0, color=OI["grey"], ls=":")
    ax.set_xlabel("quantile"); ax.set_ylabel("per-item margin (nats)")
    ax.set_title("AFTER margins compressed near 0;\nFLOOR wide — dispersion, not a signal tail", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = ["letter-logit\nspread", "Brier\n(0.75=uniform)", "frac\nmargin>0"]
    av = [d["letter_logit_spread_conf"]["AFTER"], d["brier_conf"]["AFTER"],
          d["frac_margin_positive_conf"]["AFTER"]]
    fv = [d["letter_logit_spread_conf"]["FLOOR_1337"], d["brier_conf"]["FLOOR_1337"],
          d["frac_margin_positive_conf"]["FLOOR_1337"]]
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], av, width=0.36, color=OI["blue"], label="AFTER")
    ax.bar([i + 0.18 for i in x], fv, width=0.36, color=OI["orange"], label="FLOOR")
    ax.axhline(0.75, color=OI["grey"], ls=":", lw=1)
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_title("Brier ≈ uniform; frac(margin>0) BELOW floor:\nno item-level retained signal", fontsize=9)
    ax.legend(fontsize=8)
    _footer(axes[0], m, "DESCRIPTIVE (post-critic)")
    _save(fig, "f3_dispersion_characterization", index,
          "+1.70-nat margin diff is dispersion-dominated (spread 5.5×, Brier≈uniform, frac>0 below floor)",
          "DESCRIPTIVE")
    plt.close(fig)

    # --- F4: BEFORE damage reference (descriptive) ----
    bc = c["before_cells_bucket_acc"]
    before_unp_bucket = c["scale_validity"]["grid_bucket"]["floor_pct"] + \
        100 * c["scale_validity"]["grid_bucket"]["denom"]
    names = sorted(bc)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(names)), [bc[k] for k in names], color=OI["green"])
    ax.axhline(before_unp_bucket, color=OI["grey"], ls="--",
               label=f"BEFORE unprobed (bucket) = {before_unp_bucket:.1f}%")
    ax.axhline(c["scale_validity"]["grid_bucket"]["floor_pct"], color=OI["orange"], ls=":",
               label="FLOOR (chance)")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("bucket accuracy (%)")
    ax.set_title("Damage reference: int4 costs −8.8 pts on the capability-intact side — "
                 "why FLOOR (not BEFORE) is the null", fontsize=9)
    ax.legend(fontsize=7)
    _footer(ax, m, "DESCRIPTIVE")
    _save(fig, "f4_before_damage_reference", index,
          "Probe damage is capability-dependent (int4 −8.8 pts on BEFORE; ≈0 on FLOOR)", "DESCRIPTIVE")
    plt.close(fig)

    (FIGDIR / "fig_index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} figures → {FIGDIR}")


if __name__ == "__main__":
    main()
