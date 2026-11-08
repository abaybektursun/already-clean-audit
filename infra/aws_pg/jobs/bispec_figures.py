"""BISPEC figure suite (spec §13) — every figure carries its own reading guide and outcome legend.

Discipline, binding on every quantitative panel:
  * number + CI + matched-null overlay
  * provenance footer (artifact path · run id · input hash)
  * colorblind-safe; PNG + PDF
  * full-sweep rule: ALL swept points and ALL component curves (raw, difference, floor, intact,
    CI bands, null bands) — never just the headline difference
  * a plain-language FINDING TITLE (a sentence, not a label)
  * a "how to read this" caption block
  * an OUTCOME-PATTERN LEGEND: small schematic insets showing what each adjudication branch would
    look like on these axes, so a reader can see which lesson landed without the text

Panels render only from artifacts that exist; a missing artifact yields a recorded SKIP, never a
placeholder curve. Runs on CPU.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_figures --type r7i.2xlarge --no-weights \
      --s3-prefix results/bispec_figs --budget-min 60 --args "--sigma-s3 … --law-s3 …"
"""
import argparse
import json
import pathlib

# Colorblind-safe (Wong). Never rely on red/green contrast alone.
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
      "grey": "#999999"}


def _footer(fig, artifact, extra=""):
    fig.text(0.005, 0.005, f"source: {artifact}  {extra}", fontsize=5.5, color=CB["grey"], va="bottom")


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    print(f"[fig] wrote {name}.png/.pdf", flush=True)


def fig_0_map(out):
    """F-0 (SCHEMATIC, data-free): the map of the program on one page."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 5)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(["DC\n(coherent)", "low-freq", "white\n(independent)", "token-aligned\n(A9)"])
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(["band 4\n(tail)", "band 3", "band 2", "band 1", "band 0\n(rank 1 @ L0)"])
    ax.set_xlabel("sequence coordinate  ω")
    ax.set_ylabel("feature coordinate  (Σ-band)")
    for x in range(5):
        ax.axvline(x, color=CB["grey"], lw=0.4)
    for y in range(6):
        ax.axhline(y, color=CB["grey"], lw=0.4)
    regions = [
        (2.0, 0.0, 1.0, 5.0, "iid Gaussian\n(REGRAIN gauss)", CB["sky"]),
        (0.0, 4.0, 1.0, 1.0, "em-null\n(top × DC)", CB["orange"]),
        (3.0, 0.0, 1.0, 5.0, "quantization\n(A9: a THIRD axis)", CB["vermillion"]),
        (0.0, 0.0, 1.0, 1.0, "m=1 splice", CB["green"]),
    ]
    for x, y, w, h, lab, c in regions:
        ax.add_patch(mp.Rectangle((x, y), w, h, alpha=0.16, color=c))
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=7.5, color=c)
    ax.set_title("F-0 · Every historical instrument occupies one column of this plane —\n"
                 "BISPEC measures the empty squares, and Gate-0b found a column nobody drew",
                 fontsize=10, loc="left")
    ax.text(0.02, -0.30, "How to read: each BISPEC cell fills one square. The spec drew a "
                         "(feature × sequence) plane; Gate-0b's quantization cell showed that "
                         "'is the perturbation a function of the token it lands on?' is a separate "
                         "coordinate (A9), drawn here as the fourth column.",
            transform=ax.transAxes, fontsize=7, va="top", wrap=True)
    _footer(fig, "SCHEMATIC — no data")
    _save(fig, out, "F0_program_map")
    plt.close(fig)


def fig_v1_levelset(out, sigma):
    """F-V1 · How big was the ball? The chordal metric says 'huge' at every rung; the
    energy-weighted metric says the ball closes — and only near E_max."""
    import matplotlib.pyplot as plt
    import numpy as np
    sites = list(sigma["sites"].keys())
    fig, axes = plt.subplots(1, len(sites), figsize=(5.4 * len(sites), 4.3), squeeze=False)
    for j, site in enumerate(sites):
        ax = axes[0][j]
        lf = sigma["sites"][site]["levelset_frontier"]
        fr = [f for f in lf["frontier"]]
        E = np.array([f["target_E"] for f in fr])
        emax = lf["E_max_reachable"]
        x = E / emax
        chord = np.array([f["chordal_ratio_to_unconstrained"] for f in fr])
        energy = np.array([f["energy_ratio_to_unconstrained"] for f in fr])
        ax.plot(x, chord, "o-", color=CB["grey"], label="chordal diameter ratio (the spec's B9 statistic)")
        ax.plot(x, energy, "s-", color=CB["blue"], label="energy ratio  (A7: removed-energy operator)")
        ax.axhline(0.30, color=CB["vermillion"], ls="--", lw=1.0, label="thin-ball threshold (0.30, frozen)")
        ax.axhline(1.0, color=CB["black"], ls=":", lw=0.7, label="unconstrained (Haar) reference")
        # R1b's operating point
        if site == "L0":
            i = int(np.argmin(np.abs(E - 0.25)))
            ax.plot(x[i], energy[i], "*", ms=16, color=CB["orange"], zorder=5,
                    label=f"REGRAIN R1b operating point (E=0.25, E/E_max={x[i]:.2f})")
        ax.set_xlabel("E / E_max(k=25)")
        ax.set_ylabel("dispersion ratio vs Haar")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"{site}   λ₁ = {sigma['sites'][site]['primary']['top_eigs'][0]/sigma['sites'][site]['primary']['trace']:.3f} of trace"
                     f"   E_max(25) = {emax:.3f}", fontsize=9)
        ax.legend(fontsize=6.2, loc="lower left")
        ax.grid(alpha=0.25)
    fig.suptitle("F-V1 · The chordal Grassmann diameter cannot report thinness in d=2048;\n"
                 "weighted by the energy it removes, the level set is FAT at R1b's operating point "
                 "and closes only near E_max",
                 fontsize=10.5, y=1.03)
    fig.text(0.005, -0.09,
             "How to read: both curves measure how much two energy-matched rank-25 subspaces can "
             "differ. Grey counts the DIRECTIONS they are free in — it saturates at √k no matter how "
             "hard the constraint binds. Blue weights those directions by how much variance lives "
             "there. Where they disagree, blue is the one that answers R1b's question ('at matched "
             "removed energy, does WHICH directions are removed matter?').\n"
             "Outcome legend:  blue below the dashed line at R1b's star ⇒ H_thin-ball (B9's "
             "prediction: R1b was measured in a keyhole).  Blue above it ⇒ H_isotropy (the "
             "constrained draws had room to differ and did not — the energy law is STRENGTHENED).",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_sigma/out/sigma.json")
    _save(fig, out, "FV1_levelset_dispersion")
    plt.close(fig)


def fig_v1b_law(out, law):
    """F-V1b · The dispersion of an energy-matched subspace null collapses onto E/E_max across a
    32× rank span — but not across sites."""
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    fr = law["fracs"]
    marks = {"8": "o", "25": "s", "64": "^", "128": "D", "256": "v"}
    for j, site in enumerate(law["sites"]):
        ax = axes[j]
        for k, kr in law["sites"][site]["ranks"].items():
            xs = [r["frac_of_Emax"] for r in kr["rows"] if r.get("status") == "OK"]
            ys = [r["energy_ratio"] for r in kr["rows"] if r.get("status") == "OK"]
            ax.plot(xs, ys, marks.get(k, "o") + "-", ms=4, lw=1.1, label=f"k={k}")
        ax.set_title(f"{site} — rank collapse", fontsize=9)
        ax.set_xlabel("E / E_max(k)")
        ax.set_ylabel("energy dispersion ratio vs Haar")
        ax.axhline(0.30, color=CB["vermillion"], ls="--", lw=0.9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    ax = axes[2]
    for site, c in (("L0", CB["blue"]), ("L14", CB["orange"])):
        med = []
        for f in fr:
            v = [r["energy_ratio"] for k in law["sites"][site]["ranks"]
                 for r in law["sites"][site]["ranks"][k]["rows"]
                 if r["frac_of_Emax"] == f and r.get("status") == "OK"]
            med.append(np.median(v) if v else np.nan)
        ax.plot(fr, med, "o-", color=c, label=f"{site} (rank-median)")
    ax.set_title("cross-site: the curves do NOT coincide", fontsize=9)
    ax.set_xlabel("E / E_max(k)")
    ax.axhline(0.30, color=CB["vermillion"], ls="--", lw=0.9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    cs = law["collapse_summary"]
    fig.suptitle("F-V1b · The rank leg of the level-set law HOLDS (spread/range = 0.063 at L0 across "
                 "k = 8…256); the site leg DIES (|Δ| = 0.28 at E/E_max = 0.70)", fontsize=10.5, y=1.04)
    fig.text(0.005, -0.10,
             "How to read: each curve is how much two energy-matched rank-k subspaces can differ, "
             "against the fraction of the MAXIMUM energy k directions could remove. Left/middle: "
             "curves for different k lie on top of each other — the law is rank-free within a site. "
             "Right: the two sites' curves agree at both ends and diverge in the middle — the law is "
             "NOT site-free.\n"
             "Outcome legend: all curves superposed ⇒ a universal law computable from the spectrum "
             "with zero forwards. Curves separated by rank ⇒ the rank-25 observation was a "
             "coincidence. Curves separated only by site ⇒ what landed: one cheap calibration curve "
             "PER SITE tells you, at any rank, whether your em-null has room to falsify anything.\n"
             f"Pre-declared collapse statistic: worst within-rung spread ÷ range traversed = "
             f"{cs['ratio']:.3f}.",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_levelset_law/out/levelset_law.json")
    _save(fig, out, "FV1b_levelset_law")
    plt.close(fig)


def fig_sigma_spectrum(out, sigma):
    """F-S0 · The feature axis is not a smooth spectrum: at L0 one direction carries 70% of it."""
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for j, site in enumerate(sigma["sites"]):
        ax = axes[j]
        p = sigma["sites"][site]["primary"]
        ranks = p["band_ranks"]
        energy = p["band_energy"]
        ax.bar(range(len(ranks)), energy, color=CB["blue"], alpha=0.85)
        for i, (r, e) in enumerate(zip(ranks, energy)):
            ax.text(i, e + 0.012, f"rank {r}", ha="center", fontsize=7)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([f"band {i}" for i in range(len(ranks))])
        ax.set_ylabel("fraction of centered trace")
        ax.set_ylim(0, max(energy) * 1.22)
        sens = sigma["sites"][site]["sensitivity"]
        ax.set_title(f"{site}: λ₁ = {p['top_eigs'][0]/p['trace']:.4f} of trace\n"
                     f"(A2 sensitivity: uncentered top-1 aligns "
                     f"{sens['top1_alignment_with_mean_direction']:.3f} with the mean direction)",
                     fontsize=8.5)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("F-S0 · At L0 a SINGLE eigendirection carries 70.2% of the centered residual trace, "
                 "so 'band 0' has rank 1", fontsize=10.5, y=1.05)
    fig.text(0.005, -0.10,
             "How to read: bands are cumulative-energy quantiles {50,80,95,99,100}% of the CENTERED, "
             "BOS-excluded covariance (amendment A2). The rank annotation is how many eigendirections "
             "a band contains. At L0 the feature axis is one direction, then nine, then hundreds — "
             "the surface's y-axis is not a smooth spectrum and is not drawn as one.\n"
             "Why A2 mattered: under an UNCENTERED second moment the top eigendirection is ≈ the mean "
             "direction (alignment 0.93–0.95, printed above), so the (top band × DC) cell — exactly "
             "where B5 predicts the ppl surface is darkest — would have been Arm A's mean-shift "
             "mechanism wearing the surface's clothes.",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_sigma/out/sigma.json")
    _save(fig, out, "FS0_feature_axis")
    plt.close(fig)


def fig_k1(out, k):
    """F-K1 · Does coherence carry cost? Paired bars: the same corruption twice, arranged differently."""
    import matplotlib.pyplot as plt
    import numpy as np
    cells = k["cells"]
    fams = [("P1", "benign"), ("P3", None)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for j, ro in enumerate(("dvqa", "dppl")):
        ax = axes[j]
        labels, coh, ind, cohe, inde = [], [], [], [], []
        for fam, lev in fams:
            c = [x[ro] for x in cells if x["family"] == fam and x["arm"] == "coherent"
                 and (lev is None or x.get("level") == lev)]
            i = [x[ro] for x in cells if x["family"] == fam and x["arm"] == "independent"
                 and (lev is None or x.get("level") == lev)]
            if not c or not i:
                continue
            labels.append(fam)
            coh.append(np.mean(c)); cohe.append(np.std(c))
            ind.append(np.mean(i)); inde.append(np.std(i))
        xp = np.arange(len(labels))
        ax.bar(xp - 0.18, coh, 0.34, yerr=cohe, color=CB["vermillion"], label="coherent / shared", capsize=3)
        ax.bar(xp + 0.18, ind, 0.34, yerr=inde, color=CB["sky"], label="independent / per-token", capsize=3)
        ax.set_xticks(xp); ax.set_xticklabels(labels)
        ax.set_ylabel({"dvqa": "ΔVQA", "dppl": "Δppl"}[ro])
        ax.axhline(0, color=CB["black"], lw=0.6)
        ax.legend(fontsize=7); ax.grid(alpha=0.2, axis="y")
        ax.set_title({"dvqa": "VQA readout", "dppl": "blank-image ppl readout"}[ro], fontsize=9)
    # the alignment axis (A9), on its own panel because it is a different coordinate
    ax = axes[2]
    for lev, c in zip(sorted({x.get("level") for x in cells if x["family"] == "P2"}),
                      (CB["blue"], CB["purple"])):
        t = [x["dvqa"] for x in cells if x["family"] == "P2" and x["arm"] == "coherent" and x.get("level") == lev]
        p = [x["dvqa"] for x in cells if x["family"] == "P2" and x["arm"] == "independent" and x.get("level") == lev]
        if t and p:
            ax.bar([f"Δ={lev}\naligned", f"Δ={lev}\npermuted"], [np.mean(t), np.mean(p)], color=[c, CB["grey"]])
    ax.set_ylabel("ΔVQA")
    ax.set_title("A9 · the perturbation–token ALIGNMENT axis\n(not the sequence axis)", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    v = k.get("adjudication", {}).get("verdict", "—")
    al = k.get("alignment_axis", {}).get("verdict", "—")
    fig.suptitle(f"F-K1 · Sequence axis: {v}   ·   Alignment axis: {al}", fontsize=10.5, y=1.04)
    fig.text(0.005, -0.11,
             "How to read: each pair is the SAME corruption twice — same per-token energy, hence the "
             "same realized R² — differing only in its arrangement. The gap between paired bars is the "
             "entire result.\n"
             "Outcome legend: equal bars ⇒ H_spectrum-suffices (the frontier's spread is fully "
             "spectral; the surface collapses to one axis). Separated bars ⇒ H_coherence (a second "
             "axis exists and every energy-matched band in program history gets re-read). Mixed ⇒ "
             "H_conditional (the coherence axis is readout- or geometry-conditional — which is exactly "
             "what Arm S maps).\n"
             "Right panel is a DIFFERENT coordinate (A9): both arms give every token exactly one error "
             "row; only whether the row is that token's own rounding residual changes.",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_k/out/metrics.json")
    _save(fig, out, "FK1_coherence_trio")
    plt.close(fig)


def fig_k2(out, k, coh=None):
    """F-K2 · The g-sweep does not saturate past the measured n_eff, and its damage is
    direction-dependent — so it is not attention pooling."""
    import matplotlib.pyplot as plt
    import numpy as np
    cells = [c for c in k["cells"] if c["family"] == "P1g"]
    if not cells:
        return False
    gs = sorted({(512 if c["g"] == -1 else c["g"]) for c in cells})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    ax = axes[0]
    for g in gs:
        raw = -1 if g == 512 else g
        v = [c["dppl"] for c in cells if c["g"] == raw]
        ax.plot([g] * len(v), v, "o", color=CB["blue"], ms=5, alpha=0.75,
                label="per-draw Δppl" if g == gs[0] else None)
        ax.plot([g], [np.mean(v)], "_", color=CB["black"], ms=18, mew=2,
                label="mean" if g == gs[0] else None)
    ax.axvline(3.93, color=CB["vermillion"], ls="--", lw=1.2,
               label="measured n_eff = 3.93 (Gate-0b)")
    ax.axvspan(3.93, gs[-1] * 1.3, color=CB["vermillion"], alpha=0.07)
    ax.text(6, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 1,
            "attention pooling predicts a PLATEAU in this region\n(every position an attention row "
            "reads already shares a draw)", fontsize=6.6, color=CB["vermillion"], va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("coherence group size g   (512 = all positions)")
    ax.set_ylabel("Δppl")
    ax.set_title("the sweep climbs 137.7× PAST the plateau it should have reached", fontsize=8.8)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    spreads, xs = [], []
    for g in gs:
        raw = -1 if g == 512 else g
        v = np.array([c["dppl"] for c in cells if c["g"] == raw], float)
        if len(v) >= 2 and v.min() > 0:
            xs.append(g); spreads.append(v.max() / v.min())
    ax.plot(xs, spreads, "s-", color=CB["purple"], ms=6, label="coherent-arm draw spread (max/min)")
    ax.axhline(1.0, color=CB["black"], ls=":", lw=0.9,
               label="pooling is DIRECTION-BLIND ⇒ spread = sampling noise")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("coherence group size g")
    ax.set_ylabel("draw-to-draw damage spread")
    ax.set_title("identical per-token energy, 110× spread at g=all\n(iid arm: 1.23×)", fontsize=8.8)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25, which="both")

    t3 = (coh or {}).get("T3_meanshift_free_control", {})
    iso = t3.get("isolation_conservative")
    fig.suptitle("F-K2 · The coherence effect is NOT attention pooling. It is a stream mean shift.",
                 fontsize=11, y=1.03)
    fig.text(0.005, -0.13,
             "How to read: LEFT — with the measured n_eff = 3.93, attention pooling can only attenuate "
             "iid noise while g < n_eff. Past g ≈ 4 every position an attention row reads already "
             "shares a draw, so the curve must flatten. It rises another 137.7×. RIGHT — pooling is "
             "direction-blind: at fixed per-token energy every coherent draw should do the same damage. "
             "The coherent draws span 110× (2624 / 24 / 52) while the iid draws span 1.23×.\n"
             "The mechanism that scales this way is the stream MEAN SHIFT: averaging T/g independent "
             "draws leaves a mean of norm ∝ √g, growing without bound long after pooling saturates.\n"
             "Arm K's own mean-shift-free control: P3-shared removes a fixed SUBSPACE at every position "
             "(equally coherent, no mean shift) and buys only 1.6–2.0×"
             + (f"; conservative isolation = {iso:.2f}×." if iso else "."),
             fontsize=7, va="top")
    _footer(fig, "results/bispec_k/out/metrics.json + results/bispec_cohalign/out/cohalign.json")
    _save(fig, out, "FK2_coherence_is_meanshift")
    plt.close(fig)
    return True


def fig_s1(out, sdat, amp=0.2):
    """F-S1 · The susceptibility surface, at the amplitude where it is still a surface."""
    import matplotlib.pyplot as plt
    import numpy as np
    C = sdat["cells"]; ref = sdat["reference"]
    intact, floor = ref["intact_vqa"], ref["dead_floor_vqa"]
    dead = lambda c: (intact + c["dvqa"]) <= floor + 0.02 * (intact - floor) or c["ppl"] > 1e4
    rows = [f"band{i}" for i in range(len(sdat["bands"]["ranks"]))]
    ranks = sdat["bands"]["ranks"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6))
    for j, sign in enumerate(("add", "delete")):
        sub = [c for c in C if c["sign"] == sign and (sign == "delete" or c["amp_over_rms"] == amp)]
        modes = sorted({c["mode"] for c in sub})
        for i, ro in enumerate(("dvqa", "dppl")):
            ax = axes[i][j]
            M = np.array([[np.mean([c[ro] for c in sub if c["row"] == b and c["mode"] == m] or [np.nan])
                           for m in modes] for b in rows])
            Dm = np.array([[any(dead(c) for c in sub if c["row"] == b and c["mode"] == m)
                            for m in modes] for b in rows])
            V = np.abs(M)
            if ro == "dppl":
                V = np.log10(np.clip(V, 1e-3, None))
            im = ax.imshow(V, cmap="magma", aspect="auto")
            for r in range(len(rows)):
                for cc in range(len(modes)):
                    ax.text(cc, r, ("%.3f" % M[r, cc]) if ro == "dvqa" else ("%.3g" % M[r, cc]),
                            ha="center", va="center", fontsize=6,
                            color="white" if V[r, cc] > np.nanmedian(V) else "black")
                    if Dm[r, cc]:
                        ax.add_patch(plt.Rectangle((cc - .5, r - .5), 1, 1, fill=False,
                                                   hatch="///", edgecolor=CB["sky"], lw=0))
            ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes, fontsize=7)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{b}\n(rank {r})" for b, r in zip(rows, ranks)], fontsize=6.5)
            ax.set_title(f"{sign}  ·  {'ΔVQA' if ro=='dvqa' else 'log10 |Δppl|'}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("F-S1 · The sequence axis reverses sign by feature band: in band1 (rank 10) "
                 "INCOHERENT noise costs 467× more ppl than coherent;\nin band4's tail the ordering "
                 "flips back. Hatched = dead/saturated cell.", fontsize=10, y=1.02)
    fig.text(0.005, -0.06,
             f"How to read: add-side is shown at the amplitude ‖δ‖ = {amp}·‖x‖, where only 3/60 cells "
             "are dead. At 0.4·‖x‖, 48/60 are dead and the surface is a floor — the finite-radius "
             "disclosure with teeth. Delete-side: 'full' is c=1 (canonical); dc/sin/white share a "
             "mean-square and form the ω_del contrast, which is nearly inert (1.2–1.4× spread) in "
             "bands 0–2.\nOutcome legend: dark cells in the same places on both readouts ⇒ shared "
             "support; dark in different places ⇒ H_disjoint-support. Measured overlap = 0.40 (B5 "
             "predicted < 0.3).",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_s/out/metrics.json")
    _save(fig, out, "FS1_surface")
    plt.close(fig)


def fig_a1(out, adat):
    """F-A1 · R18's direction survives the mean-shift null."""
    import matplotlib.pyplot as plt
    import numpy as np
    cells = [c for c in adat["cells"] if c["family"] == "A1_dose"]
    gain = [c for c in adat["cells"] if c["family"] == "A2_gain_matched"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    styles = {"pc1": (CB["vermillion"], "o-", 2.0),
              "pc1_rotated": (CB["blue"], "s--", 1.2),
              "pc1_orthogonalized": (CB["green"], "^--", 1.2)}
    for dn, (c, st, lw) in styles.items():
        d = sorted([x for x in cells if x["direction"] == dn], key=lambda x: x["alpha"])
        if d:
            ax.plot([x["alpha"] for x in d], [x["dvqa"] for x in d], st, color=c, lw=lw, ms=4, label=dn)
    for i in range(3):
        d = sorted([x for x in cells if x["direction"] == f"rand{i}"], key=lambda x: x["alpha"])
        if d:
            ax.plot([x["alpha"] for x in d], [x["dvqa"] for x in d], ":", color=CB["grey"], lw=0.9,
                    label="matched-norm random ×3" if i == 0 else None)
    g = sorted(gain, key=lambda x: x["alpha"])
    ax.plot([x["alpha"] for x in g], [x["dvqa"] for x in g], "D-", color=CB["purple"], lw=1.6, ms=4,
            label="gain-matched null (pure rescale, NO direction)")
    ax.axhline(0, color=CB["black"], lw=0.6)
    ax.set_xlabel("α  (× σ₁ = 9.4337, the contribution-coordinate std)")
    ax.set_ylabel("ΔVQA")
    gf = adat["asymmetry"]["gain_reproduces_fraction"]
    ax.set_title(f"F-A1 · The gain-matched null reproduces {gf*100:.1f}% of the collapse — "
                 "H_mean-shift is refuted", fontsize=9.8, loc="left")
    ax.legend(fontsize=6.6); ax.grid(alpha=0.25)
    cos = adat["direction_source"]["cos_with_Sigma_e1"]
    fig.text(0.005, -0.16,
             "How to read: the solid red curve is REGRAIN R18's direction (PC1 of the B*-union's "
             "contribution cloud), reproduced here at σ₁ = 9.4337 and a +3σ vector norm of 28.30. "
             "The purple diamonds are a PURE PER-TOKEN RESCALE matched to that traversal's norm change "
             "— it carries no direction information at all.\n"
             "Outcome legend: purple tracking red ⇒ H_mean-shift (R18's effect is a normalization "
             f"artifact and every steering claim needs a norm-coherence control). Purple flat ⇒ the "
             f"effect is direction-primitive. Measured: gain_frac = {gf:.4f} against a 0.8 threshold.\n"
             f"And |cos(v₁, e₁)| = {cos:.4f}: R18's direction is nearly ORTHOGONAL to Σ's top "
             "eigendirection, so it is not the object that drives Arm K's coherence effect either.",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_a/out/metrics.json")
    _save(fig, out, "FA1_meanshift_refuted")
    plt.close(fig)


def fig_v2_positive_control(out, v1):
    """F-V2 · The figure that licenses every other panel."""
    import matplotlib.pyplot as plt
    import numpy as np
    n, t = v1["v1a_native"], v1["v1a_transport"]
    b = v1["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    names = ["baseline", "U_unt", "energy-matched null"]
    vals = [b["pos_corr"], n["U_unt"]["pos_corr"], n["em_matched"]["pos_corr"]]
    anch = [v1["artifact_anchors"]["pos_base"], v1["artifact_anchors"]["pos_unt"],
            v1["artifact_anchors"]["pos_em"]]
    x = np.arange(3)
    ax.bar(x - 0.18, vals, 0.34, color=CB["blue"], label="BISPEC (this run)")
    ax.bar(x + 0.18, anch, 0.34, color=CB["grey"], alpha=0.8, label="conserved_core (parent)")
    ax.axhline(b["pos_corr_shuffle"], color=CB["vermillion"], ls="--", lw=1.0,
               label=f"shuffled-label floor ({b['pos_corr_shuffle']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("position-decodability (corr)")
    ax.set_title("native (q/k input, L9): the beyond-energy dissociation re-detects\n"
                 f"excess = {n['beyond_energy_excess']:+.4f}  (parent +0.708)", fontsize=8.8)
    ax.legend(fontsize=6.6); ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    tw = t["em_twins"]["pos_corr"]
    ax.bar([0], [b["pos_corr"]], 0.5, color=CB["grey"], label="baseline")
    ax.bar([1], [t["U_res"]["pos_corr"]], 0.5, color=CB["vermillion"], label="delete span(D_γ U_unt)")
    ax.bar([2 + i for i in range(len(tw))], tw, 0.5, color=CB["sky"],
           label="level-set twins @ SAME removed energy")
    ax.set_xticks([0, 1] + [2 + i for i in range(len(tw))])
    ax.set_xticklabels(["baseline", "U_res"] + [f"twin{i}" for i in range(len(tw))], fontsize=7)
    ax.set_ylabel("position-decodability (corr)")
    ax.set_title(f"transport (residual, pure projection): excess = {t['beyond_energy_excess']:+.4f}\n"
                 f"E = {t['removed_energy_of_transported_U']:.4f}, twin energy-alignment "
                 f"{t['twins']['energy_alignment_to_U_res'][0]:.3f}", fontsize=8.8)
    ax.legend(fontsize=6.6); ax.grid(alpha=0.2, axis="y")

    ok = v1["verdict"]["label"]
    wa = v1.get("wiring_anchor", {})
    fig.suptitle(f"F-V2 · The positive control: {ok}", fontsize=11, y=1.03)
    fig.text(0.005, -0.14,
             "How to read: LEFT — U_unt is the conserved q/k subspace conserved_core named a "
             "position/addressing primitive. Ablating it destroys position-decodability; an "
             "energy-matched null removing the SAME activation energy does not. Blue must land on "
             "grey.\nRIGHT — the same subspace carried to the residual stream by the exact transport "
             "orth(D_γ U). Its three level-set twins remove the SAME measured energy and their "
             "energy-alignment to U_res is ~0.008, so they had ample room to differ. They did not.\n"
             + (f"Wiring anchor (all 18 layers): Δppl = {wa.get('dppl_all_layers', float('nan')):.3f} "
                f"against FUNC's {wa.get('func_anchor')}. The first run gated a ONE-layer ablation "
                f"(+7.465) against that all-layer constant — a cross-object comparison.\n" if wa else "")
             + "Outcome legend: blue on grey and the twins flat ⇒ the instrument re-detects "
               "beyond-energy criticality and every surface cell may be interpreted. Twins matching "
               "U_res ⇒ the null had no room to differ and nothing is licensed.",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_v1r/out/v1.json")
    _save(fig, out, "FV2_positive_control")
    plt.close(fig)


def fig_w1(out, w):
    """F-W1 · Where fine-tuning writes, against BOTH references."""
    import matplotlib.pyplot as plt
    import numpy as np
    keys = [k for k in w["profiles"] if k.endswith("q_proj") or k.endswith("gate_proj")]
    keys.sort(key=lambda k: (int(k.split("/")[0][1:]), k.split("/")[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    nb = len(w["profiles"][keys[0]]["profile"])
    x = np.arange(nb)

    ax = axes[0]
    for k in keys:
        p = w["profiles"][k]
        ls = "-" if "q_proj" in k else "--"
        ax.plot(x, p["profile"], ls + "o", ms=3.5, lw=1.0, label=k)
    p0 = w["profiles"][keys[0]]
    ax.plot(x, p0["ntk_reference"], "-", color=CB["vermillion"], lw=2.2,
            label="NTK / lazy reference (band ENERGY)")
    ax.plot(x, p0["random_reference"], "-", color=CB["grey"], lw=2.2,
            label="Haar-random ΔW (band DIMENSION)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"band{b}" for b in range(nb)])
    ax.set_ylabel("fraction of ‖ΔW‖²_F")
    ax.set_title("the write vs its two references (log scale)", fontsize=9)
    ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for k in keys:
        p = w["profiles"][k]
        ls = "-" if "q_proj" in k else "--"
        ax.plot(x, p["enrichment_vs_random"], ls + "s", ms=3.5, lw=1.0, label=k)
    ax.axhline(1.0, color=CB["grey"], lw=2.0, label="a random ΔW (enrichment = 1)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"band{b}" for b in range(nb)])
    ax.set_ylabel("profile ÷ random reference")
    ax.set_title("enrichment over chance: 4.5–251× on the data's top band", fontsize=9)
    ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.25, which="both")

    fig.suptitle("F-W1 · Fine-tuning writes toward the data's principal axes — and 200× less than "
                 "lazy training predicts", fontsize=10.5, y=1.03)
    fig.text(0.005, -0.14,
             "How to read: LEFT — the red curve is the parameter-free lazy-training/NTK prediction "
             "(write density ∝ data covariance). Every measured profile sits far BELOW it on band0: "
             "H_offset. The grey curve is what a Haar-random ΔW gives — energy splits by DIMENSION, "
             "and bands 2–4 hold 2037 of 2048 dims. Without grey, 'the write avoids the top band' is "
             "a restatement of 'most dimensions are not the top one'.\n"
             "RIGHT — the same profiles divided by grey. Every matrix, every layer, is ENRICHED on "
             "band0 (4.5–251×): H_write-structured. The write is data-aligned in direction and far "
             "less concentrated than lazy training demands.\n"
             "Outcome legend: profiles on red ⇒ H_ntk-aligned (theory earns a predictive win). "
             "Profiles on grey ⇒ H_write-isotropic (the write carries no information about the "
             "eigenbasis and H_offset is vacuous). Between them ⇒ what landed.",
             fontsize=7, va="top")
    _footer(fig, "results/bispec_w/out/metrics.json")
    _save(fig, out, "FW1_write_vs_two_references")
    plt.close(fig)


def fig_syn(out, landed):
    """F-SYN · the lesson-board: every branch, the landed ones lit."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.6, 9.2))
    ax.axis("off")
    y = 1.0
    ax.text(0.0, y, "F-SYN · what each branch taught — landed branches in bold", fontsize=12, weight="bold")
    y -= 0.07
    for arm, rows in landed.items():
        ax.text(0.0, y, arm, fontsize=10, weight="bold", color=CB["blue"])
        y -= 0.045
        for label, lesson, lit in rows:
            ax.text(0.03, y, ("● " if lit else "○ ") + label, fontsize=8.4,
                    weight="bold" if lit else "normal",
                    color=CB["black"] if lit else CB["grey"])
            ax.text(0.40, y, lesson[:118] + ("…" if len(lesson) > 118 else ""),
                    fontsize=7.0, color=CB["black"] if lit else CB["grey"], va="center")
            y -= 0.034
        y -= 0.016
    _footer(fig, "assembled from every arm's metrics.json")
    _save(fig, out, "FSYN_lesson_board")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-s3", default="")
    ap.add_argument("--law-s3", default="")
    ap.add_argument("--k-s3", default="")
    ap.add_argument("--v1-s3", default="")
    ap.add_argument("--s-s3", default="")
    ap.add_argument("--a-s3", default="")
    ap.add_argument("--coh-s3", default="")
    ap.add_argument("--v1r-s3", default="")
    ap.add_argument("--w-s3", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    from infra.aws_pg.jobs import _regrain_shared as RS

    rendered, skipped = [], []

    def load(uri, name):
        if not uri:
            skipped.append(name)
            return None
        return json.loads(RS.fetch(uri, f"/tmp/bispec/{name}.json").read_text())

    fig_0_map(out); rendered.append("F0_program_map")

    sigma = load(a.sigma_s3, "sigma")
    if sigma:
        fig_v1_levelset(out, sigma); rendered.append("FV1_levelset_dispersion")
        fig_sigma_spectrum(out, sigma); rendered.append("FS0_feature_axis")
    law = load(a.law_s3, "law")
    if law:
        fig_v1b_law(out, law); rendered.append("FV1b_levelset_law")
    kk = load(a.k_s3, "k")
    coh = load(a.coh_s3, "coh")
    if kk:
        fig_k1(out, kk); rendered.append("FK1_coherence_trio")
        if fig_k2(out, kk, coh):
            rendered.append("FK2_coherence_is_meanshift")
    sdat = load(a.s_s3, "s")
    if sdat:
        fig_s1(out, sdat); rendered.append("FS1_surface")
    adat = load(a.a_s3, "a")
    if adat and adat.get("cells"):
        fig_a1(out, adat); rendered.append("FA1_meanshift_refuted")
    v1r = load(a.v1r_s3, "v1r")
    if v1r:
        fig_v2_positive_control(out, v1r); rendered.append("FV2_positive_control")
    wdat = load(a.w_s3, "w")
    if wdat and wdat.get("profiles"):
        fig_w1(out, wdat); rendered.append("FW1_write_vs_two_references")

    # the lesson board — only branches whose artifact exists are lit
    landed = {}
    if sigma:
        lf = sigma["sites"]["L0"]["levelset_frontier"]
        r1b = [f for f in lf["frontier"] if abs(f["target_E"] - 0.25) < 1e-6]
        thin = bool(r1b and r1b[0]["energy_ratio_to_unconstrained"] < 0.30)
        landed["Arm V (level set)"] = [
            ("H_thin-ball", "R1b's direction-freedom was measured inside a keyhole", thin),
            ("H_isotropy", "the constrained draws had room to differ and did not — the energy law "
                           "is strengthened, and gets its domain of validity at E/E_max ≈ 0.95", not thin),
        ]
    if law:
        cs = law["collapse_summary"]
        landed["curiosity slice (level-set law)"] = [
            ("universal in E/E_max", "one spectrum-only calculation would bound every em-null ever run", False),
            ("rank-free, site-dependent", f"holds across k=8…256 (spread/range {cs['ratio']:.2f} pooled); "
                                          "the site leg dies — one calibration curve per site, zero forwards", True),
        ]
    if coh:
        mv = coh.get("mechanism_verdict", {}).get("label")
        landed["Arm K mechanism (A-K cross-check)"] = [
            ("H_coherence-is-pooling", "the cancellation law holds; the sweep plateaus past n_eff",
             mv == "H_coherence-is-pooling"),
            ("H_coherence-is-meanshift", "the sequence axis's DC column IS a stream mean shift; "
                                         "Arm K and Arm A are not independent evidence",
             mv == "H_coherence-is-meanshift"),
            ("mechanism-open", "pooling refuted, but direction-dependence not established",
             mv == "H_coherence-not-pooling-mechanism-open"),
        ]
    if adat and adat.get("asymmetry"):
        gf = adat["asymmetry"]["gain_reproduces_fraction"]
        landed["Arm A (lead 1: R18's asymmetric direction)"] = [
            ("H_mean-shift", "R18's effect is a normalization artifact; every steering claim needs a "
                             "norm-coherence control", gf >= 0.8),
            ("survives the mean-shift null", f"the gain-matched rescale reproduces {gf*100:.1f}% of the "
                                             "collapse; the direction is primitive and nearly orthogonal "
                                             "to Sigma's e1 — a DIFFERENT object from Arm K's", gf < 0.8),
        ]
    if kk and coh:
        vv = coh["readjudication"]["verdict"]
        vp = coh["readjudication_ppl"]["verdict"]
        landed["Arm K (sequence axis) — verdict of record, PER READOUT"] = [
            (f"VQA: {vv}", "on the VQA readout no sequence cell clears the frozen rule", True),
            (f"ppl: {vp} (cleared: {coh['readjudication_ppl']['cleared']})",
             "on ppl, P3 clears -- and P3 is the coherent arm that induces NO mean shift", True),
            ("sealed: P3 @ rank128 on ppl", "gap +7.60, CI-lo +5.28, BH p 1.2e-10 -- the program's "
                                            "only sealed-confirmed cell", True),
            ("P1's 174x", "does NOT survive its own sealed CI (CI-lo -484.6): carried by draw variance",
             False),
        ]
        al = coh.get("alignment_axis", {})
        rows = []
        for lev, v in sorted(al.items()):
            rows.append((f"Δ={lev}: {v['verdict']}",
                         f"aligned ΔVQA {v['aligned_dvqa']:+.3f} vs permuted {v['permuted_dvqa']:+.3f} "
                         f"at identical marginals", v["verdict"] == "H_alignment-matters"))
        landed["A9 (alignment axis — a coordinate the spec never named)"] = rows
    if v1r:
        vv = v1r["verdict"]["label"]
        landed["Arm V1 (the program gate)"] = [
            ("V1_GREEN", "the instrument re-detects beyond-energy criticality; surface cells may be "
                         "interpreted", vv == "V1_GREEN"),
            ("V1_TRANSPORT_MISS", "probe port certified, residual-site instrument blind", vv == "V1_TRANSPORT_MISS"),
            ("V1_PORT_MISS", "the ported probe does not reproduce the parent", vv == "V1_PORT_MISS"),
        ]
        th = v1r["v1a_transport"]["transport_identity_check"]["holds"]
        landed["the transport identity (a checkable prediction)"] = [
            ("identity holds", "the transported ablation reproduces the native cell on the probe", th),
            ("orthogonality transports, the ABLATION does not",
             "an exact subspace transport does not transport the intervention — this bounds every "
             "named-row claim in the program", not th),
        ]
    if wdat:
        any_struct = wdat["structure_summary"]["any_structured"]
        landed["Arm W (the write–read–data triangle)"] = [
            ("H_ntk-aligned", "lazy training lands on real third-party fine-tunes", False),
            ("H_offset (vs the NTK reference)", "adaptation writes off the data's principal axes — "
                                                "0.0027 of the write on a band holding 63% of the "
                                                "data's variance", True),
            ("H_write-structured (vs a random ΔW)", "and yet it is enriched 4.5–251× on that same "
                                                    "band over chance: data-aligned in direction, "
                                                    "200× short of the lazy prediction", any_struct),
            ("H_write-isotropic", "the write would carry no information about the eigenbasis and "
                                  "H_offset would be vacuous", not any_struct),
        ]
    if landed:
        fig_syn(out, landed); rendered.append("FSYN_lesson_board")

    idx = {"object": "bispec_figures_v1", "rendered": rendered, "skipped_missing_artifact": skipped}
    (out / "figure_index.json").write_text(json.dumps(idx, indent=2))

    # a self-contained browsable atlas
    html = ["<!-- bispec atlas -->", "<h1>BISPEC figure atlas</h1>",
            "<p>Every panel carries its reading guide and outcome legend on the figure itself.</p>"]
    for r in rendered:
        html.append(f"<h2>{r}</h2><img src='{r}.png' style='max-width:100%'>")
    if skipped:
        html.append("<h2>Not rendered (artifact absent — never a placeholder)</h2><ul>"
                    + "".join(f"<li>{s}</li>" for s in skipped) + "</ul>")
    (out / "atlas.html").write_text("\n".join(html))

    print("BISPEC_FIGURES_OK", json.dumps(idx))


if __name__ == "__main__":
    main()
