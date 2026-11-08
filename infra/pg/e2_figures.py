"""E2b figure suite — WEIGHT-FREE (pi-trajectory spec §E2b + follow-up §C). Renders from e2_geometry.json
(frozen extraction) + e2_reagg.json (re-frozen aggregation, pi-e2-reagg-v3) — no models, no GPU. Every
panel backs a number + its null + a GEOMETRY-ONLY tag + a provenance footer.

  python -c "from infra.pg.e2_figures import render; render('e2_geometry.json','figs/','<prov>','e2_reagg.json')"
"""
from __future__ import annotations

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GEOM = "GEOMETRY-ONLY — computation untested (Tier-2 function arm gates promotion)"
# corrected per-r verdict labels (follow-up §C): r8 is NOT a named waterfall bin (G-power-inconclusive).
CORRECTED_R_LABEL = {8: "ρ_L-only / INCONCLUSIVE (G-power)", 16: "extend (borderline, a-hw≈0.500)",
                     32: "extend", 64: "extend"}


def render(geometry_json, out_dir, provenance="(provenance)", reagg_json=None):
    import json
    import pathlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.loads(pathlib.Path(geometry_json).read_text())
    summ, per_tensor = d["summary"], d["per_tensor"]
    agg = summ["aggregation"]
    reagg = json.loads(pathlib.Path(reagg_json).read_text()) if reagg_json else None
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    r_grid = sorted((int(r) for r in agg), key=int)
    foot = lambda fig: fig.text(0.005, 0.004, f"{provenance} | {GEOM}", fontsize=6, alpha=0.6)

    # fig1 — pooled ρ_L vs r (+ Haar-p95) and pooled signed a vs r (+ frozen G-power band).
    fig, (axL, axA) = plt.subplots(1, 2, figsize=(11, 4))
    rs = [int(r) for r in r_grid]
    axL.plot(rs, [agg[str(r)]["pooled_energy_weighted_rho_L"] for r in rs], "o-", label="pooled ρ_L (energy-wt)", color="C0")
    axL.plot(rs, [agg[str(r)]["pooled_haar_L_p95"] for r in rs], "s--", label="Haar-p95 null", color="grey")
    axL.set_xlabel("r (ΔW₁ top-r)"); axL.set_ylabel("ρ_L"); axL.set_title("Reuse: ρ_L vs architecture-intrinsic null")
    axL.legend(fontsize=8); axL.set_xticks(rs)
    gp = summ.get("e1_gpower_halfwidth", {})
    axA.axhline(0, color="k", lw=0.5)
    axA.plot(rs, [agg[str(r)]["pooled_energy_weighted_a"] for r in rs], "o-", color="C3", label="pooled signed a")
    for r in rs:
        hw = gp.get(str(r), gp.get(r, 0))
        axA.errorbar(r, 0, yerr=hw, color="grey", alpha=0.4, capsize=4)
        if hw > 0.5:
            axA.annotate("ρ_L-only\n(a blunt)", (r, 0), fontsize=6, ha="center", va="bottom", color="darkred")
    axA.set_ylim(-1.05, 1.05); axA.set_xlabel("r"); axA.set_ylabel("signed a")
    axA.set_title("Sign vs frozen G-power band"); axA.legend(fontsize=8); axA.set_xticks(rs)
    foot(fig); fig.tight_layout(); fig.savefig(out / "fig1_reuse_primary.png", dpi=140); plt.close(fig)

    # fig2 — per-type ρ_L (enrichment + absolute) + per-type signed-a panel (re-frozen by_type_a CIs).
    r_head = 32 if 32 in rs else rs[len(rs) // 2]
    bt = agg[str(r_head)]["by_type_rho_L"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    xs = np.arange(len(PROJ))
    rhoL = [bt[t]["point"] if bt.get(t) else np.nan for t in PROJ]
    p95 = [np.mean([rec["reuse"][str(r_head)]["haar_rho_L"]["p95.0"] for rec in per_tensor if rec["type"] == t]) for t in PROJ]
    enrich = [rhoL[i] / p95[i] if p95[i] else np.nan for i in range(len(PROJ))]
    ax1b = ax1.twinx()
    ax1.bar(xs - 0.2, rhoL, 0.4, color="C0", alpha=0.8, label="abs ρ_L")
    ax1b.bar(xs + 0.2, enrich, 0.4, color="C2", alpha=0.6, label="enrichment ρ_L/null")
    ax1b.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax1.set_xticks(xs); ax1.set_xticklabels(PROJ, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("absolute ρ_L", color="C0"); ax1b.set_ylabel("enrichment ×null", color="C2")
    ax1.set_title(f"Per-type ρ_L at r={r_head}: k/v tall-but-near-null (d_out=256, null≈0.125)")
    ax1.annotate("k/v: d_out=256 → null≈0.125\n(tall ρ_L is near-null, NOT high enrichment)",
                 (1.5, max(v for v in rhoL if v == v) * 0.75), fontsize=6.5, color="dimgray")
    if reagg:
        ba = reagg["by_type_a"][str(r_head)]
        pts = [ba[t]["point"] for t in PROJ]; los = [ba[t]["point"] - ba[t]["lo"] for t in PROJ]; his = [ba[t]["hi"] - ba[t]["point"] for t in PROJ]
        cols = ["C2" if ba[t]["lo"] > 0 else ("C3" if ba[t]["hi"] < 0 else "grey") for t in PROJ]
        ax2.axhline(0, color="k", lw=0.6)
        ax2.bar(xs, pts, yerr=[los, his], capsize=4, color=cols, alpha=0.85)
        ax2.set_xticks(xs); ax2.set_xticklabels(PROJ, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("signed a [95% CI, n=18]"); ax2.set_ylim(-1.05, 1.05)
        ax2.set_title(f"Per-type signed a at r={r_head} (green=CI>0 extends, grey=CI crosses 0)")
        ax2.annotate("q/k CIs cross 0 → sign-AMBIGUOUS\n(not 'reverse')", (4.0, -0.85), fontsize=6.5, color="dimgray")
    foot(fig); fig.tight_layout(); fig.savefig(out / "fig2_by_type.png", dpi=140); plt.close(fig)

    # fig3 — (type, layer) ρ_L heatmap.
    layers = sorted({rec["layer"] for rec in per_tensor})
    M = np.full((len(PROJ), len(layers)), np.nan)
    for rec in per_tensor:
        M[PROJ.index(rec["type"]), layers.index(rec["layer"])] = rec["reuse"][str(r_head)]["rho_L"]
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(PROJ))); ax.set_yticklabels(PROJ)
    ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
    ax.set_xlabel("decoder layer"); ax.set_title(f"ρ_L by (type, layer) at r={r_head}")
    fig.colorbar(im, ax=ax, label="ρ_L"); foot(fig); fig.tight_layout()
    fig.savefig(out / "fig3_depth_heatmap.png", dpi=140); plt.close(fig)

    # fig4 — FORM E_gain/C_route/f_perp per hop, with the cross-hop NON-comparability caveat on-panel.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, metric in zip(axes, ("E_gain", "C_route", "f_perp")):
        ft1 = [rec["form_ft1"][metric] for rec in per_tensor if rec["form_ft1"].get(metric) is not None]
        ft2 = [rec["form_ft2"][metric] for rec in per_tensor if rec["form_ft2"].get(metric) is not None]
        ax.bar([0, 1], [np.mean(ft1), np.mean(ft2)], color=["C0", "C1"], alpha=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["FT#1\n(Gemma-1 basis)", "FT#2\n(PG1 basis)"], fontsize=8)
        ax.set_title(metric)
        if metric != "f_perp":
            ax.axhline(1.0, color="grey", ls="--", lw=0.8, label="null=1"); ax.legend(fontsize=7)
    fig.suptitle("Form (§3.1) per hop — each z-scored vs its OWN N1 null; NOT numerically comparable across "
                 "hops (different substrate/basis); no declining-gain trend implied", fontsize=9)
    foot(fig); fig.tight_layout(); fig.savefig(out / "fig4_form_per_hop.png", dpi=140); plt.close(fig)

    # fig5 — decision summary with CORRECTED per-r labels (r8 hatched = inconclusive, NOT a bin).
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bins_order = ["build_disjoint", "partial_reuse", "extend", "overwrite_undo"]
    cols = {"build_disjoint": "grey", "partial_reuse": "C0", "extend": "C2", "overwrite_undo": "C3"}
    bottom = np.zeros(len(rs))
    for b in bins_order:
        vals = [agg[str(r)]["per_tensor_bin_counts"].get(b, 0) for r in rs]
        ax.bar([str(r) for r in rs], vals, bottom=bottom, label=b, color=cols[b], alpha=0.85)
        bottom += np.array(vals)
    # hatch the r=8 column to mark it signed-inconclusive (G-power), not a named bin.
    if 8 in rs:
        ax.bar(str(8), bottom[rs.index(8)], color="none", edgecolor="black", hatch="//", lw=0)
    ax.set_xlabel("r"); ax.set_ylabel("# tensors (per-tensor diagnostic)"); ax.legend(fontsize=8, loc="upper right")
    label = "  ".join(f"r{r}:{CORRECTED_R_LABEL.get(r, '?')}" for r in rs)
    ax.set_title("HEADLINE (energy-wt pooled, corrected): " + label, fontsize=8)
    foot(fig); fig.tight_layout(); fig.savefig(out / "fig5_decision.png", dpi=140); plt.close(fig)

    # fig6 (NEW) — the L17 story: MLP a-per-layer profile + a_pooled vs a_pooled_exL17 vs N1 band.
    if reagg:
        fig, (axP, axB) = plt.subplots(1, 2, figsize=(12, 4.2))
        prof = reagg["mlp_a_per_layer"]["32"]
        Ls = list(range(18)); ys = [prof[str(L)] for L in Ls]
        axP.axhline(0, color="k", lw=0.6)
        axP.plot(Ls, ys, "o-", color="C1")
        axP.axvspan(16.5, 17.5, color="C3", alpha=0.15)
        axP.annotate("L17 (readout\nthe expert reads)", (17, max(y for y in ys if y is not None)), fontsize=6.5, ha="right", color="C3")
        axP.set_xlabel("decoder layer"); axP.set_ylabel("MLP energy-wt signed a (r=32)")
        v = reagg["verdict"]
        axP.set_title(f"MLP a-per-layer: {v['n_early_layers_pos']}/17 of L0..16 >0 → {v['bin']}", fontsize=8.5)
        # a_pooled full vs exL17 vs band, at r=32/64
        rr = ["32", "64"]; x = np.arange(2); wd = 0.25
        full = [reagg["A"][r]["a_pooled_full"] for r in rr]
        exm = [reagg["A"][r]["a_pooled_exL17mlp"] for r in rr]
        ub = [reagg["A"][r]["a_ub_exL17mlp"] for r in rr]
        axB.axhline(0, color="k", lw=0.6)
        axB.bar(x - wd, full, wd, label="a_pooled (full)", color="C3", alpha=0.8)
        axB.bar(x, exm, wd, label="a_pooled (excl L17-MLP)", color="C2", alpha=0.8)
        axB.plot(x, ub, "k_", ms=22, mew=2, label="N1 a-band upper (exL17)")
        axB.set_xticks(x); axB.set_xticklabels([f"r={r}" for r in rr]); axB.set_ylim(-0.1, 1.05)
        axB.set_ylabel("pooled signed a"); axB.legend(fontsize=7)
        axB.set_title("Decisive: does signed-extend survive removing L17-MLP?", fontsize=8.5)
        foot(fig); fig.tight_layout(); fig.savefig(out / "fig6_L17_story.png", dpi=140); plt.close(fig)

    return sorted(str(p.name) for p in out.glob("*.png"))
