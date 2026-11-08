"""REGRAIN figures (CPU) — the §14 suite. Figures are STAGE OUTPUTS, not a closeout afterthought.

Discipline (binds every quantitative figure):
  * it backs a claim independently carried by a number + CI + null overlay
  * provenance footer: run id · SHA · input hash · DISCOVERY / CONFIRMATORY / EXPLORATORY tag
  * colorblind-safe; PNG (review) + PDF (vector); bbox_inches="tight"
  * every SWEEP shows ALL swept points and ALL component curves (raw + difference + floor +
    intact + CI bands + null bands) on one axis — never just the headline diff (house rule)
  * a panel whose input artifact is missing is rendered as an explicit SKIPPED placeholder and
    listed in `skipped`; it is never silently dropped

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_figures --type r7i.4xlarge --no-weights \
      --s3-prefix results/regrain_figs --budget-min 60 --args "--sha <git-sha> --inputs '<json>'"
"""
import argparse
import hashlib
import json
import pathlib

# Okabe–Ito, colorblind-safe
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "black": "#000000",
      "grey": "#999999"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sha", default="unknown")
    ap.add_argument("--cdisc-s3", default=None)
    ap.add_argument("--csealed-s3", default=None)
    ap.add_argument("--factorial-s3", default=None)
    ap.add_argument("--frontier-s3", default=None)
    ap.add_argument("--d-s3", default=None)
    ap.add_argument("--e-s3", default=None)
    ap.add_argument("--h-s3", default=None)
    ap.add_argument("--b-s3", default=None)
    ap.add_argument("--f-s3", default=None)
    ap.add_argument("--g-s3", default=None)
    ap.add_argument("--gallery-s3", default=None)
    ap.add_argument("--ce-s3", default=None, help="regrain_c_energy (the energy ladder)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from infra.aws_pg.jobs import _regrain_shared as RS

    registry, skipped = {}, []

    def load(uri, name):
        if not uri:
            return None
        p = RS.fetch(f"{uri}/{name}", f"/tmp/regrain/fig_{abs(hash(uri+name))}.json")
        return json.loads(p.read_text())

    def stamp(fig, tag, run, inputs):
        h = hashlib.sha1(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:10]
        fig.text(0.005, 0.005, f"{run} · {a.sha[:8]} · in:{h} · {tag}", fontsize=5.2,
                 color=CB["grey"], ha="left", va="bottom")

    def save(fig, name, tag, run, inputs):
        stamp(fig, tag, run, inputs)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=190, bbox_inches="tight")
        plt.close(fig)
        registry[name] = {"script": "regrain_figures.py", "tag": tag, "run": run}

    def skip(name, why):
        fig, ax = plt.subplots(figsize=(5, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{name}\nSKIPPED\n{why}", ha="center", va="center", fontsize=9,
                color=CB["red"])
        save(fig, name, "SKIPPED", "n/a", {"why": why})
        skipped.append({"figure": name, "reason": why})

    cd = load(a.cdisc_s3, "c_disc.json")
    cs = load(a.csealed_s3, "metrics.json")
    fac = load(a.factorial_s3, "gates.json")
    fr = load(a.frontier_s3, "frontier.json")
    dd = load(a.d_s3, "metrics.json")
    ee = load(a.e_s3, "metrics.json")
    hh = load(a.h_s3, "metrics.json")
    bb = load(a.b_s3, "metrics.json")
    ff = load(a.f_s3, "metrics.json")
    gg = load(a.g_s3, "metrics.json")
    ce = load(a.ce_s3, "metrics.json")

    # ---------------- F-C1: the program headline ----------------
    if cs:
        eff, rf = cs["effects"], cs["rf"]
        order = ["full_read", "bstar", "rand_uniform_0", "rand_uniform_1", "rand_uniform_2",
                 "rand_usage_0", "rand_usage_1", "rand_usage_2", "em_span", "equal_rank_span"]
        order = [c for c in order if c in eff]
        colors = {"full_read": CB["black"], "bstar": CB["red"], "em_span": CB["purple"],
                  "equal_rank_span": CB["green"]}
        fig, ax = plt.subplots(figsize=(9.6, 4.6))
        for i, c in enumerate(order):
            lo, hi = rf[c]["ci"]
            v = rf[c]["rf"]
            col = colors.get(c, CB["blue"] if "uniform" in c else CB["orange"])
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=3, ms=6)
        uni = [rf[f"rand_uniform_{i}"]["rf"] for i in range(3) if f"rand_uniform_{i}" in rf]
        usg = [rf[f"rand_usage_{i}"]["rf"] for i in range(3) if f"rand_usage_{i}" in rf]
        if uni:
            ax.axhspan(min(uni), max(uni), color=CB["blue"], alpha=0.10,
                       label="uniform random-union band")
        if usg:
            ax.axhspan(min(usg), max(usg), color=CB["orange"], alpha=0.14,
                       label="usage-matched band (A2)")
        ax.axhline(1.0, color=CB["grey"], ls=":", lw=1, label="intact (RF=1)")
        ax.axhline(0.0, color=CB["grey"], ls="--", lw=1, label="dead-model floor (RF=0)")
        ax.axhline(rf["bstar"]["rf"] + cs["delta_C"], color=CB["red"], ls="-.", lw=0.9,
                   label=r"$RF(B^*)+\delta_C$")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("RF (dead-model-floor convention)")
        ax.set_title(f"F-C1 · sealed atom-handle adjudication @ L{cs['L']} · "
                     f"|B*|={cs['bstar_rank']} · verdict: {cs['verdict']}")
        ax.legend(fontsize=7, loc="lower left")
        save(fig, "regrain_F_C1_sealed_handle", "CONFIRMATORY", "regrain_c_sealed", cs["gaps"])
    else:
        skip("regrain_F_C1_sealed_handle", "regrain_c_sealed artifact absent")

    # ---------------- F-C2: the dose curve (all points, all seeds) ----------------
    if cd:
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        for s, v in sorted(cd["per_seed"].items()):
            xs = [p["n_atoms"] for p in v["dose_curve"]]
            ys = [p["effect"] for p in v["dose_curve"]]
            ax.plot(xs, ys, "-o", ms=2.6, lw=1.2, label=f"seed {s} (|B*|={v['bstar_len']})",
                    color=CB["red"] if int(s) == cd["median_seed"] else CB["sky"])
        se = cd["reference"]["screen_effect_se"]
        ef = cd["reference"]["effect_full_screen"]
        ax.axhline(ef, color=CB["black"], ls="--", lw=1.1, label="full read (k=128)")
        ax.axhspan(ef - se, ef + se, color=CB["black"], alpha=0.08, label="±1 screen SE")
        ax.axhline(cd["reference"]["rho_threshold_on_screen"], color=CB["green"], ls=":",
                   lw=1.1, label=r"$\rho\cdot$effect(full)  [gates |B*| only]")
        ax.axvline(cd["bstar_rank"], color=CB["red"], ls="-.", lw=0.9, alpha=0.7)
        ax.set_xlabel("atoms added (greedy, descending single-atom effect)")
        ax.set_ylabel("ablation effect on the 320-item screen slice")
        ax.set_title("F-C2 · dose curve — non-monotone under nesting (SE = %.3f)" % se)
        ax.legend(fontsize=7.5)
        save(fig, "regrain_F_C2_dose_curve", "DISCOVERY", "regrain_c_disc", cd["reference"])
    else:
        skip("regrain_F_C2_dose_curve", "regrain_c_disc artifact absent")

    # ---------------- F-C1b: the energy ladder (the mechanism behind R1) ----------------
    if ce and ce.get("cells") and ce.get("headline"):
        cells, h = ce["cells"], ce["headline"]
        el = ce["energy_ladder"]
        groups = [("full_read", [("full_read", None)], CB["black"], "full read (k=128)"),
                  ("bstar", [("bstar", 1.0)], CB["red"], "B* (25 atoms)"),
                  ("energy_core", [("energy_core", el["energy_core_E_ratio"])], CB["purple"],
                   "energy core (top-25 by energy)"),
                  ("marg", [(f"marginal_matched_{i}", r) for i, r in
                            enumerate(el["marginal_matched_E_ratios"])], CB["orange"],
                   "marginal-energy-matched ×3"),
                  ("uni", [(f"union_matched_{i}", r) for i, r in
                           enumerate(el["union_matched_E_ratios"])], CB["blue"],
                   "UNION-energy-matched ×3  (the valid null)")]
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        x = 0
        for _, members, col, lab in groups:
            first = True
            for name, ratio in members:
                if name not in cells:
                    continue
                ax.bar(x, cells[name]["effect"], color=col, alpha=0.9,
                       label=lab if first else None)
                if ratio is not None:
                    ax.text(x, cells[name]["effect"] + 0.012, f"E/E$_b$\n{ratio:.2f}",
                            ha="center", fontsize=5.8, color=CB["grey"])
                first = False
                x += 1
            x += 0.6
        ax.axhline(cells["bstar"]["effect"], color=CB["red"], ls="--", lw=1.1,
                   label="B* effect = %.3f" % cells["bstar"]["effect"])
        ax.axhline(h["effect_union_matched_mean"], color=CB["blue"], ls=":", lw=1.3,
                   label="union-matched MEAN = %.3f" % h["effect_union_matched_mean"])
        ax.set_xticks([])
        ax.set_ylabel("ablation effect  (1 − RF), discovery half")
        ax.set_ylim(0, max(c["effect"] for c in cells.values()) * 1.22)
        ax.set_title("F-C1b · the energy ladder — at MATCHED union energy, random atoms reproduce "
                     "B*\n(B*/control = %.2f)  EXPLORATORY · adjudicates nothing"
                     % h["bstar_over_union_matched"], fontsize=10)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95)
        save(fig, "regrain_F_C1b_energy_ladder", "EXPLORATORY", "regrain_c_energy", h)
    else:
        skip("regrain_F_C1b_energy_ladder", "regrain_c_energy artifact absent")

    # ---------------- F-C3: handle geometry vs the Haar band ----------------
    if ee and "c3_geometry" in ee:
        g = ee["c3_geometry"]
        band = g["stable_rank_haar_band"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.axhspan(band["p05"], band["p95"], color=CB["grey"], alpha=0.25,
                   label="Haar band (|B*| random unit columns), p05–p95")
        ax.axhline(band["median"], color=CB["grey"], lw=1)
        ax.plot([0], [g["stable_rank_bstar_decoders"]], "o", ms=11, color=CB["red"],
                label="stable rank of B* decoders")
        ax.axhline(8.0, color=CB["green"], ls=":", lw=1.2, label="H_tiling leg-1 rule (≤ 8)")
        ax.set_xticks([])
        ax.set_ylabel("stable rank")
        ax.set_title("F-C3 · handle geometry — leg 1 %s (beats Haar: %s)"
                     % ("HOLDS" if g["tiling_leg1_holds"] else "FAILS", g["tiling_leg1_beats_haar"]))
        ax.legend(fontsize=7.5)
        save(fig, "regrain_F_C3_handle_geometry", "DESCRIPTIVE", "regrain_e_atlas", g)
    else:
        skip("regrain_F_C3_handle_geometry", "regrain_e_atlas c3_geometry absent")

    # ---------------- F-I1: the attribution 2×2 ----------------
    if fac:
        cells = fac["configs"]
        names = [n for n in ("vanilla_rect_m1", "vanilla_sign_m1", "vanilla_rect_m4",
                             "vanilla_sign_m4") if n in cells]
        if names:
            fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6))
            for ax, fld, leg, lab in ((axs[0], "dvqa", "vqa_ok", "ΔVQA"),
                                      (axs[1], "dppl", "ppl_ok", "Δppl  (symlog)")):
                for i, n in enumerate(names):
                    c = cells[n]
                    # colour by THIS panel's leg, not the overall gate — the two legs disagree
                    ok = c["gate_parts"][leg]
                    ax.bar(i, c["splice"][fld], color=CB["green"] if ok else CB["red"],
                           alpha=0.85, label=("leg passes" if ok else "leg fails")
                           if i == 0 or (ok != cells[names[0]]["gate_parts"][leg]
                                         and i == 1) else None)
                    ax.plot(i, c["rotated_error_null"][fld], "D", color=CB["black"], ms=6,
                            zorder=4, label="rotated-error null" if i == 0 else None)
                    band = c["em_vqa_band"] if fld == "dvqa" else c["em_ppl_band"]
                    ax.errorbar(i, band["mean"], yerr=band["halfwidth"], fmt="_",
                                color=CB["purple"], capsize=7, lw=1.8, zorder=5,
                                label="em band (this config's own)" if i == 0 else None)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([n.replace("vanilla_", "") for n in names], rotation=20,
                                   fontsize=8)
                ax.set_ylabel(lab)
                ax.axhline(0, color=CB["grey"], lw=0.7)
                ax.legend(fontsize=6.6, loc="best")
            # Δppl spans 2.3 → 159.5; a linear axis hides the R12 story, namely that sign_m4's bar
            # sits BELOW its own band (leg passes) while rect_m1's sits ABOVE (leg fails).
            axs[1].set_yscale("symlog", linthresh=1.0)
            fig.suptitle("F-I1 · sign × block-size attribution — verdict: %s\n"
                         "each band is energy-matched to that config's OWN reconstruction error"
                         % fac.get("sign_verdict"), fontsize=10)
            save(fig, "regrain_F_I1_attribution_2x2", "CONFIRMATORY", "regrain_i_factorial",
                 fac.get("attribution_2x2", {}))
        else:
            skip("regrain_F_I1_attribution_2x2", "no vanilla configs in the factorial artifact")

        # ---------------- F-I3: training landscape + MDL ----------------
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
        cids = list(cells)
        axs[0].bar(range(len(cids)), [cells[c]["quality"]["fvu"] for c in cids],
                   color=[CB["green"] if cells[c].get("gate_pass") else CB["red"] for c in cids])
        axs[0].set_xticks(range(len(cids)))
        axs[0].set_xticklabels(cids, rotation=40, ha="right", fontsize=7)
        axs[0].set_ylabel("held-out FVU")
        axs[0].set_title("all %d configs (green = splice gate passed)" % len(cids))
        for frac, col in (("0.01", CB["blue"]), ("0.05", CB["sky"]), ("0.10", CB["orange"]),
                          ("0.20", CB["red"])):
            axs[1].plot(range(len(cids)), [cells[c]["mdl"][frac]["total"] for c in cids], "-o",
                        ms=3.5, color=col, label=f"δ = {frac}·var")
        axs[1].set_xticks(range(len(cids)))
        axs[1].set_xticklabels(cids, rotation=40, ha="right", fontsize=7)
        axs[1].set_ylabel("MDL (bits / token, Eq. 6)")
        axs[1].legend(fontsize=7)
        axs[1].set_title("description length at four distortions")
        save(fig, "regrain_F_I3_landscape_mdl", "DESCRIPTIVE", "regrain_i_factorial",
             fac.get("mdl_ranking_20pct", {}))
    else:
        skip("regrain_F_I1_attribution_2x2", "regrain_i_factorial artifact absent")
        skip("regrain_F_I3_landscape_mdl", "regrain_i_factorial artifact absent")

    # ---------------- F-I2: the distortion frontier (the decision-relevant figure) ------------
    if fr:
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
        fam_col = {"quant": CB["blue"], "gauss": CB["orange"], "em_abl": CB["green"],
                   "rot_err": CB["purple"]}
        # Δppl spans 10 orders of magnitude (a rank-512 removal costs +33,090; a rot_err at α=8
        # costs 1.4e18). A linear axis renders every informative point as a flat line at zero.
        # symlog keeps the near-zero regime readable AND shows the blow-up. The R²<0.45 tail is
        # off-scale for every readout and is cropped, with the crop stated in the caption.
        R2_MIN = 0.55
        for ax, fld, lab in ((axs[0], "dvqa", "ΔVQA"), (axs[1], "dppl", "Δppl  (symlog)")):
            for fam, col in fam_col.items():
                pts = sorted([p for p in fr["points"] if p["family"] == fam
                              and p["r2"] >= R2_MIN], key=lambda p: p["r2"])
                if pts:
                    ax.plot([p["r2"] for p in pts], [p[fld] for p in pts], "-o", ms=4.5,
                            color=col, label=fam)
            anc = fr.get("m1_splice_anchor")
            if anc:
                ax.plot(anc["r2"], anc[fld], "*", ms=16, color=CB["red"], zorder=5,
                        label="m=1 true splice")
            if fac:
                for cid, c in fac["configs"].items():
                    if "recon_energy_frac" in c and "splice" in c:
                        r2c = 1.0 - c["recon_energy_frac"]
                        if r2c >= R2_MIN:
                            ax.plot(r2c, c["splice"][fld], "s", ms=4.6, color=CB["black"],
                                    alpha=0.85, zorder=4)
                            ax.annotate(cid, (r2c, c["splice"][fld]), fontsize=5.2,
                                        xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color=CB["grey"], lw=0.7)
            ax.set_xlabel(r"realized reconstruction fidelity $R^2(x, x')$")
            ax.set_ylabel(lab)
            ax.set_xlim(1.005, R2_MIN)
            ax.legend(fontsize=7, loc="lower left")
        axs[1].set_yscale("symlog", linthresh=1.0)
        tol = fr.get("tolerance", {})
        fig.suptitle("F-I2 · distortion frontier of the PG1 forward — at matched $R^2$ the "
                     "corruption GEOMETRY sets the behavioural cost\n"
                     "(App. G.1 protocol on the LIVE forward, not a frozen probe; "
                     "$R^2<%.2f$ cropped)" % R2_MIN, fontsize=9.5)
        save(fig, "regrain_F_I2_frontier", "CONFIRMATORY", "regrain_i_frontier", tol)
    else:
        skip("regrain_F_I2_frontier", "regrain_i_frontier artifact absent")

    # ---------------- F-D1: draw agreement with BOTH nulls ----------------
    if dd and "svl" in dd:
        s = dd["svl"]
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        ax.axhline(s["permutation_p95"], color=CB["grey"], ls="--", lw=1.4,
                   label="permutation null p95 (= %.3f)" % s["permutation_p95"])
        ax.axhline(s["surrogate_p95"], color=CB["purple"], ls="-.", lw=1.4,
                   label="canonicity-matched SURROGATE p95 (A1) = %.3f" % s["surrogate_p95"])
        ax.axhspan(0, s["surrogate_p95"], color=CB["purple"], alpha=0.07)
        # the decisive comparison: the observed MEAN sits on the surrogate MEAN
        ax.axhline(s["surrogate_mean"], color=CB["purple"], ls="-", lw=1.0, alpha=0.75,
                   label="surrogate null MEAN = %.3f" % s["surrogate_mean"])
        ax.scatter(range(len(s["jaccard_pairs"])), s["jaccard_pairs"], color=CB["blue"], s=44,
                   label="observed pairwise Jaccard", zorder=3)
        ax.axhline(s["jaccard_mean"], color=CB["blue"], lw=1.4,
                   label="observed MEAN = %.3f" % s["jaccard_mean"])
        if s.get("jaccard_cilo") is not None:
            ax.axhline(s["jaccard_cilo"], color=CB["red"], ls=":", lw=1.3,
                       label="J CI-lo = %.3f (the rule's input)" % s["jaccard_cilo"])
        ax.set_xlabel("draw pair")
        ax.set_ylabel("top-8 aligned-atom Jaccard")
        ax.set_ylim(0, 1.02)
        ax.set_title("F-D1 · S_vl, %d draws · canonicity(k=%d)=%.3f · verdict: %s"
                     % (s["n_draws"], dd["k_readout"], s["canonicity_at_readout_rank"],
                        s["verdict"]))
        ax.legend(fontsize=6.8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95)
        save(fig, "regrain_F_D1_draw_agreement", "CONFIRMATORY", "regrain_d", s)
    else:
        skip("regrain_F_D1_draw_agreement", "regrain_d artifact absent")

    # ---------------- F-E1: the atlas + its matched-noise null ----------------
    if ee and ee.get("atlas"):
        cids = [c for c, v in ee["atlas"].items() if v.get("rank_triplet")]
        if cids:
            fig, ax = plt.subplots(figsize=(10.4, 4.4))
            w = 0.26
            for k, (stat, col) in enumerate((("stable_rank", CB["blue"]),
                                             ("participation_ratio", CB["orange"]),
                                             ("effective_rank", CB["green"]))):
                xs = np.arange(len(cids)) + (k - 1) * w
                ax.bar(xs, [ee["atlas"][c]["rank_triplet"][stat]["median"] for c in cids],
                       width=w, color=col, label=stat)
                nulls = [ee["atlas"][c].get("rank_triplet_matched_noise_null") for c in cids]
                ax.plot(xs, [n[stat]["median"] if n else np.nan for n in nulls], "k_", ms=9,
                        label="matched-noise null" if k == 0 else None)
            ax.set_xticks(range(len(cids)))
            ax.set_xticklabels(cids, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("median per-block rank statistic")
            ax.set_title("F-E1 · atlas: the paper's C.4 triplet, each beside its App.-E null")
            ax.legend(fontsize=7)
            save(fig, "regrain_F_E1_atlas", "DESCRIPTIVE", "regrain_e_atlas",
                 {"cids": cids})
        else:
            skip("regrain_F_E1_atlas", "no dictionary produced a rank triplet")
    else:
        skip("regrain_F_E1_atlas", "regrain_e_atlas artifact absent")

    # ---------------- F-H1: the L14 stream × dictionary matrix ----------------
    if hh and hh.get("arms"):
        arms = [k for k in ("pure_text", "pure_vl", "mixed") if k in hh["arms"]
                and "gate" in hh["arms"][k]]
        if arms:
            fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.2))
            for i, k in enumerate(arms):
                g = hh["arms"][k]["gate"]
                axs[0].bar(i, g["delta_ppl"], color=CB["green"] if g["ppl_ok"] else CB["red"])
                axs[0].plot(i, g["rotated_error_dppl"], "D", color=CB["black"], ms=6,
                            label="rotated-error null" if i == 0 else None)
                axs[0].errorbar(i, g["em_ppl"]["mean"], yerr=g["em_ppl"]["halfwidth"], fmt="_",
                                color=CB["purple"], capsize=6, label="em band" if i == 0 else None)
                axs[1].bar(i - 0.18, hh["arms"][k]["fvu_on_text"], width=0.34, color=CB["blue"],
                           label="FVU on text" if i == 0 else None)
                axs[1].bar(i + 0.18, hh["arms"][k]["fvu_on_vl"], width=0.34, color=CB["orange"],
                           label="FVU on VL" if i == 0 else None)
            for ax, lab in ((axs[0], "Δppl (blank-image wikitext)"), (axs[1], "held-out FVU")):
                ax.set_xticks(range(len(arms)))
                ax.set_xticklabels(arms, fontsize=8)
                ax.set_ylabel(lab)
                ax.legend(fontsize=7)
            fig.suptitle("F-H1 · L14 anatomy — verdict: %s" % hh.get("verdict"))
            save(fig, "regrain_F_H1_stream_anatomy", "CONFIRMATORY", "regrain_h",
                 hh.get("adjudication_inputs", {}))
        else:
            skip("regrain_F_H1_stream_anatomy", "no gated arms in regrain_h")
    else:
        skip("regrain_F_H1_stream_anatomy", "regrain_h artifact absent")

    # ---------------- F-F1: steering dose-response ----------------
    if ff and ff.get("sweeps"):
        fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.2))
        style = {"pc1": (CB["red"], 2.0), "pc2": (CB["orange"], 1.3),
                 "pc1_rotated": (CB["purple"], 1.2), "rand0": (CB["grey"], 0.9),
                 "rand1": (CB["grey"], 0.9), "rand2": (CB["grey"], 0.9)}
        for tag, pts in ff["sweeps"].items():
            col, lw = style.get(tag, (CB["sky"], 1.0))
            al = [p["alpha"] for p in pts]
            axs[0].plot(al, [p["dvqa"] for p in pts], "-o", ms=3, color=col, lw=lw, label=tag)
            axs[1].plot(al, [p["answer_shift_tv"] for p in pts], "-o", ms=3, color=col, lw=lw)
        axs[0].set_ylabel("ΔVQA")
        axs[1].set_ylabel("answer-distribution shift (TV)")
        for ax in axs:
            ax.set_xlabel(r"$\alpha$  (units of $\sigma$ along the coordinate)")
            ax.axvline(0, color=CB["grey"], lw=0.7)
        axs[0].legend(fontsize=7)
        fig.suptitle("F-F1 · steering the handle · monotone=%s (%s)"
                     % (ff["dose_response"]["monotone"], ff["label"]))
        save(fig, "regrain_F_F1_steering", "EXPLORATORY", "regrain_f", ff["dose_response"])
    else:
        skip("regrain_F_F1_steering", "regrain_f artifact absent")

    # ---------------- F-B1 / F-B2 ----------------
    if bb and bb.get("verdict") not in (None, "UNRUN"):
        eff = bb["effects"]
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        ks = [k for k in eff]
        ax.bar(range(len(ks)), [eff[k] for k in ks],
               color=[CB["red"] if k == "bstar" else CB["grey"] for k in ks])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("ablation effect (1 − RF)")
        ax.set_title("F-B1 · block-grain sealed adjudication — %s" % bb["verdict"])
        save(fig, "regrain_F_B1_block_adjudication", "CONFIRMATORY", "regrain_b", bb["gaps"])
        t2 = bb["tiling_leg2"]
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.plot([p["n_blocks"] for p in t2["collapse_curve"]],
                [p["bstar_span_energy_covered"] for p in t2["collapse_curve"]], "-o",
                color=CB["blue"], label="B*-span energy covered")
        ax.axhline(0.90, color=CB["green"], ls=":", label="90% rule")
        ax.axvline(4, color=CB["red"], ls="-.", label="collapse ≤ 4 rule")
        ax.axhline(t2["haar_coverage_at_4_blocks"], color=CB["grey"], ls="--",
                   label="Haar floor @ 4 blocks")
        ax.set_xlabel("blocks (top-aligned to B*)")
        ax.set_ylabel("fraction of B*-span energy")
        ax.set_title("F-B2 · tiling leg 2 — collapse=%s" % t2["collapse_count"])
        ax.legend(fontsize=7)
        save(fig, "regrain_F_B2_tiling_leg2", "CONFIRMATORY", "regrain_b", t2)
    else:
        why = ("Arm B closed with the I4 certificate (no gate-passing block featurizer)"
               if bb else "regrain_b artifact absent")
        skip("regrain_F_B1_block_adjudication", why)
        skip("regrain_F_B2_tiling_leg2", why)

    # ---------------- F-I4: utilization panel (paper Fig. 22 analog) ----------------
    if ee and ee.get("atlas"):
        cids = [c for c, v in ee["atlas"].items()
                if v.get("rank_triplet") and v.get("utilization_eff_over_b") is not None]
        if cids:
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            bs = [ee["atlas"][c]["b"] for c in cids]
            ut = [ee["atlas"][c]["utilization_eff_over_b"] for c in cids]
            ax.plot(bs, ut, "o", ms=8, color=CB["blue"])
            for b_, u_, c in zip(bs, ut, cids):
                ax.annotate(c, (b_, u_), fontsize=6, xytext=(4, 3), textcoords="offset points")
            ax.axhline(1.0, color=CB["grey"], ls="--", lw=1, label="full occupancy")
            ax.set_xlabel("block dimension b")
            ax.set_ylabel("utilization  (effective rank / b)")
            ax.set_ylim(0, 1.08)
            ax.legend(fontsize=7)
            ax.set_title("F-I4 · how much of each block's allotted dimension is used")
            save(fig, "regrain_F_I4_utilization", "DESCRIPTIVE", "regrain_e_atlas",
                 {"cids": cids})
        else:
            skip("regrain_F_I4_utilization", "no b>1 dictionary with a rank triplet")
    else:
        skip("regrain_F_I4_utilization", "regrain_e_atlas artifact absent")

    # ---------------- F-H2: error anatomy by position type ----------------
    if hh and hh.get("error_anatomy_by_position_type"):
        an = hh["error_anatomy_by_position_type"]
        ks = list(an)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        axs[0].bar(range(len(ks)), [an[k]["rel_err"] for k in ks], color=CB["red"])
        axs[0].set_ylabel("relative splice error  ‖x̂−x‖ / ‖x‖")
        axs[1].bar(range(len(ks)), [an[k]["cosine"] for k in ks], color=CB["blue"])
        axs[1].set_ylabel("cosine(x̂, x)")
        for ax in axs:
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels(ks, fontsize=8)
        fig.suptitle("F-H2 · L14 splice-error anatomy by position type")
        save(fig, "regrain_F_H2_error_anatomy", "DESCRIPTIVE", "regrain_h", an)
    else:
        skip("regrain_F_H2_error_anatomy", "regrain_h error anatomy absent")

    # ---------------- F-G1: the bridge footprints ----------------
    if gg and gg.get("write_footprint"):
        w = gg["write_footprint"]
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        names = ["Haar null"] + [f"write · {m}" for m in w] + ["task footprint"]
        vals = [gg["haar_null"]["mean"]] + [w[m]["top8_share_mean"] for m in w] \
            + [gg["task_footprint"]["top8_share"]]
        cols = [CB["grey"]] + [CB["blue"]] * len(w) + [CB["orange"]]
        ax.bar(range(len(vals)), vals, color=cols)
        ax.errorbar(len(vals) - 1, gg["task_footprint"]["top8_share"],
                    yerr=[[gg["task_footprint"]["top8_share"] - gg["task_footprint"]["ci95"][0]],
                          [gg["task_footprint"]["ci95"][1] - gg["task_footprint"]["top8_share"]]],
                    fmt="none", ecolor=CB["black"], capsize=4)
        ax.axhline(8.0 / gg["dictionary"]["n_atoms"], color=CB["red"], ls=":", lw=1,
                   label="perfectly flat (8/W)")
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=7.5)
        ax.set_ylabel("top-8 share of the footprint (log)")
        ax.legend(fontsize=7)
        ax.set_title("F-G1 · fine-tuning's consensus write vs the task, in the L%d dictionary "
                     "(EXPLORATORY)" % gg["dictionary"]["site"])
        save(fig, "regrain_F_G1_bridge_footprints", "EXPLORATORY", "regrain_g", gg["headline"])
    else:
        skip("regrain_F_G1_bridge_footprints", "regrain_g artifact absent")

    # ---------------- F-S1: the synthesis dashboard (the §1 tree with landed verdicts) --------
    def _f(v, yes, no):
        return None if v is None else (yes if v else no)

    verdicts = [
        ("Arm C · atom handle (MDE)", (cs or {}).get("verdict"),
         "sealed, 400 items, touched once"),
        ("Arm C · mechanism (exploratory)",
         _f(((ce or {}).get("headline") or {}).get("union_matched_reproduces_bstar"),
            "energy effect — the union-energy-matched null reproduces B*", "direction-carried"),
         "discovery half; adjudicates nothing"),
        ("Arm I · sign / class boundary", (fac or {}).get("sign_verdict"),
         "vanilla 2×2 + Grassmannian b=1..4"),
        ("Arm B · block grain", (bb or {}).get("verdict"), "gated on I4"),
        ("Arm D · draw agreement", ((dd or {}).get("headline") or {}).get("verdict"),
         "S_vl, 5 disjoint draws"),
        ("Arm H · L14 stream", (hh or {}).get("verdict"), "per-stream dictionaries"),
        ("Arm F · steering", _f(((ff or {}).get("dose_response") or {}).get("monotone"),
                                "monotone dose-response", "no monotone dose-response"),
         "contribution-PCA coordinates"),
        ("Arm G · bridge (R11)", _f(((gg or {}).get("headline") or {})
                                    .get("write_more_concentrated_than_task"),
                                    "write MORE concentrated than task",
                                    "write LESS concentrated than task"),
         "exploratory; L14 only (structural)"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.axis("off")
    ax.set_title("F-S1 · REGRAIN synthesis — landed verdicts\n"
                 "(a verdict is neither good nor bad; UNRUN is shown, never imputed)", fontsize=11)
    for i, (name, v, note) in enumerate(verdicts):
        y = 0.96 - i * 0.122
        landed = v is not None
        ax.text(0.01, y, name, fontsize=9, va="center",
                color=CB["black"] if landed else CB["grey"])
        ax.text(0.40, y, "UNRUN" if not landed else str(v), fontsize=9, va="center",
                color=CB["grey"] if not landed else CB["blue"], fontweight="bold")
        ax.text(0.40, y - 0.045, note, fontsize=6.2, va="center", color=CB["grey"])
    save(fig, "regrain_F_S1_synthesis", "SYNTHESIS", "regrain_figs",
         {n: str(v) for n, v, _ in verdicts})

    (out / "figure_registry.json").write_text(json.dumps(
        {"registry": registry, "skipped": skipped, "sha": a.sha,
         "n_rendered": len(registry) - len(skipped), "n_skipped": len(skipped)},
        indent=2, default=str))
    print("REGRAIN_FIGS_OK", json.dumps({"rendered": len(registry) - len(skipped),
                                         "skipped": [s["figure"] for s in skipped]}))


if __name__ == "__main__":
    main()
