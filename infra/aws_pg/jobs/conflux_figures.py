"""CONFLUX figure suite (CPU, r7i) — the 6 panels (spec §10), weight-free from the disc+conf JSONs.
Every panel: CI band where available, nulls overlaid, plain-language title, provenance footer;
colourblind-safe; PNG+PDF; bbox_inches='tight'.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.conflux_figures \
      --type r7i.2xlarge --no-weights --s3-prefix results/conflux_figures --budget-min 30
"""
import argparse
import json
import os
import pathlib

# colourblind-safe (Wong)
C = {"real": "#000000", "haar": "#999999", "within": "#0072B2", "energy": "#D55E00",
     "uwr": "#009E73", "uunt": "#CC79A7", "rstar": "#E69F00"}


def _footer(fig, text):
    fig.text(0.005, 0.005, text, fontsize=5, color="#666666", ha="left", va="bottom")


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc-prefix", default="results/conflux_disc")
    ap.add_argument("--conf-prefix", default="results/conflux_conf")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wd = pathlib.Path("/tmp/conflux"); wd.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from infra.aws_pg.jobs import _conflux_shared as CS

    disc = json.loads(CS.fetch(f"{CS.S3_BUCKET}/{a.disc_prefix}/out/metrics.json", wd / "d.json").read_text())
    conf = json.loads(CS.fetch(f"{CS.S3_BUCKET}/{a.conf_prefix}/out/metrics.json", wd / "c.json").read_text())
    prov = f"conflux_disc={disc.get('object')} conflux_conf={conf.get('object')} r*={conf.get('rstar')}"
    lad = disc["ladder"]; ks = [int(k) for k in lad["ks"]]
    rstar = int(conf["rstar"])
    rungs = lad["rungs"]

    def col(key):
        return [rungs[str(k)][key] if str(k) in rungs else rungs[k][key] for k in ks]

    def rg(k):
        return rungs.get(str(k), rungs.get(k))

    # ---- Panel 1: the headline overlap ladder ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    o_real = [rg(k)["o_real"] for k in ks]
    haar = [rg(k)["haar"]["mean"] for k in ks]
    within = [rg(k)["within_stiff_null"]["o_ubb_median"] for k in ks]
    within_p95 = [rg(k)["within_stiff_null"]["o_ubb_p95"] for k in ks]
    energy = [rg(k)["energy_null"]["o_ubb_median"] for k in ks]
    uwr = [rg(k)["partners"].get("U_wr", np.nan) for k in ks]
    uunt = [rg(k)["partners"].get("U_unt", np.nan) for k in ks]
    ax.plot(ks, o_real, "-o", color=C["real"], lw=2, label="o(U_bb, U_sus)  [discovery]")
    ax.plot(ks, within, "--", color=C["within"], label="within-stiff null (decisive)")
    ax.fill_between(ks, within, within_p95, color=C["within"], alpha=0.15)
    ax.plot(ks, energy, ":", color=C["energy"], label="energy-matched null")
    ax.plot(ks, haar, "-", color=C["haar"], lw=1, label="Haar-2048 floor (r/d)")
    ax.plot(ks, uwr, "-.", color=C["uwr"], label="partner o(U_bb, U_wr)")
    ax.plot(ks, uunt, "-.", color=C["uunt"], label="partner o(U_bb, U_unt)")
    # confirmation r* sealed point
    ch = conf["headline"]
    ax.plot([rstar], [ch["o_real"]], "*", color=C["rstar"], ms=18, label=f"confirmation r*={rstar} (sealed)")
    for k in ks:
        if rg(k)["void"]:
            ax.axvspan(k * 0.92, k * 1.08, color="#dddddd", alpha=0.4, zorder=0)
    ax.axvline(rstar, color=C["rstar"], ls=":", alpha=0.5)
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("rank r (ladder rung; grey = VOID by saturation)")
    ax.set_ylabel("rank-normalised subspace overlap")
    ax.set_title("Does the fine-tune backbone U_bb coincide with the sensitive subspace U_sus?")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))  # outside — never hide U_wr points
    ax.grid(alpha=0.2)
    _footer(fig, prov); _save(fig, out, "fig1_overlap_ladder"); plt.close(fig)

    # ---- Panel 2: energy adjudication — excess vs within-stiff null with CI + c_sat ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ex, lo, hi = [], [], []
    for k in ks:
        r = rg(k)
        ci = r.get("excess_within_ci")
        ex.append(r["excess_within_point"])
        lo.append(ci["ci95"][0] if ci else np.nan)
        hi.append(ci["ci95"][1] if ci else np.nan)
    ax.axhline(0, color="#000000", lw=0.8)
    ax.plot(ks, ex, "-o", color=C["within"], label="excess over within-stiff null")
    ax.fill_between(ks, lo, hi, color=C["within"], alpha=0.2, label="95% FT-bootstrap CI")
    che = conf["headline"].get("excess_within_ci")
    if che:
        # center on the bootstrap median (yerr ≥ 0 by construction); the full-sample point can sit
        # just outside the resampled CI (mean-projector bootstrap bias), so plot it separately.
        med = che["median"]
        ax.errorbar([rstar], [med],
                    yerr=[[max(0.0, med - che["ci95"][0])], [max(0.0, che["ci95"][1] - med)]],
                    fmt="*", color=C["rstar"], ms=16, capsize=4, label="confirmation r* (sealed, boot median)")
        ax.plot([rstar], [conf["headline"]["excess_within_point"]], "x", color=C["rstar"],
                ms=10, label="confirmation r* (full-sample point)")
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("rank r"); ax.set_ylabel("excess_within = o(U_bb,U_sus) − within-stiff null")
    ax.set_title("Does U_bb align with the sensitive TOP beyond a random stiff direction?")
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    _footer(fig, prov); _save(fig, out, "fig2_energy_adjudication"); plt.close(fig)

    # ---- Panel 3: per-layer heatmap (Arm D) ----
    pl = conf.get("arm_d_per_layer", {})
    if pl:
        Ls = sorted(int(L) for L in pl.keys())
        o = np.array([[pl[str(L)]["o_real"], pl[str(L)]["within_null_median"]] for L in Ls])
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(o.T, aspect="auto", cmap="viridis", origin="lower")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["o(U_bb^L, U_sus)", "within-stiff null"])
        ax.set_xticks(range(len(Ls))); ax.set_xticklabels(Ls)
        ax.set_xlabel("decoder layer L (U_sus fixed at L0)")
        ax.set_title(f"Arm D — per-layer backbone alignment with the L0 sensitive subspace (r*={rstar})")
        fig.colorbar(im, ax=ax, label="overlap")
        _footer(fig, prov + " | Arm D uses the single L0 Q (only L0 banked)")
        _save(fig, out, "fig3_per_layer_heatmap"); plt.close(fig)

    # ---- Panel 4: three-way (Arm C) ----
    tw = conf.get("arm_c_three_way", {})
    if tw:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        keys = ["bb_sus", "bb_carrier", "sus_carrier", "triple"]
        ax.bar(keys, [tw[k] for k in keys], color=[C["within"], C["energy"], C["uunt"], C["rstar"]])
        ax.set_ylabel(f"overlap (rank {tw.get('r')})")
        ax.set_title("Arm C — backbone ∩ sensitive ∩ carrier")
        for i, k in enumerate(keys):
            ax.text(i, tw[k], f"{tw[k]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(alpha=0.2, axis="y")
        _footer(fig, prov); _save(fig, out, "fig4_three_way"); plt.close(fig)

    # ---- Panel 5: residual characterisation (Arm N) ----
    rn = conf.get("arm_n_residual", {})
    if rn:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["off-U_sus energy frac", "sink mass frac"],
               [rn["off_sus_energy_frac"], rn["sink_mass_frac"]], color=[C["within"], C["haar"]])
        ax.set_title(f"Arm N — what is left of U_bb outside U_sus (r*={rstar})")
        ax.set_ylabel("fraction"); ax.grid(alpha=0.2, axis="y")
        _footer(fig, prov); _save(fig, out, "fig5_residual"); plt.close(fig)

    # ---- Panel 6: predictive scatter (Arm B) ----
    ab = conf.get("arm_b_predictive", {})
    if ab and not ab.get("instrument_null"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        rows = ab["rows"]
        g = [r["gain"] for r in rows]
        ax.scatter([r["off_bb"] for r in rows], g, color=C["within"], label=f"off-U_bb (ρ={ab['rho_off_bb_vs_gain']:.2f})")
        ax.scatter([r["off_sus"] for r in rows], g, color=C["energy"], marker="s", label=f"off-U_sus (ρ={ab['rho_off_sus_vs_gain']:.2f})")
        ax.set_xlabel("off-subspace ΔW energy fraction"); ax.set_ylabel("banked behavioural gain")
        ax.set_title("Arm B — does backbone-membership predict behaviour ≥ susceptibility?")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        _footer(fig, prov); _save(fig, out, "fig6_predictive"); plt.close(fig)

    rep = {"object": "conflux_figures_v1", "figures": sorted(p.name for p in out.glob("*.png")),
           "green": True}
    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    print("CONFLUX_FIGURES_OK", json.dumps(rep))


if __name__ == "__main__":
    main()
