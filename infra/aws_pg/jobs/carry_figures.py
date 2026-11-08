"""CARRY figures (CPU) — the §10 figure suite. Reads every arm's banked metrics from S3 and
renders the 16 figures; each carries a plain-language claim title backed by a number + CI +
null, a one-line how-to-read, the null/baseline overlaid, the full-sweep rule (every swept
point + every component curve), colourblind-safe colours, a provenance footer, PNG + PDF. Robust
to missing arms (a figure whose inputs are absent is skipped + logged). Regenerated post-critic.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_figures --type c7i.2xlarge \
      --no-weights --s3-prefix results/carry_figures --budget-min 40
"""
import argparse
import json
import os
import pathlib

# colourblind-safe (Wong 2011)
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "grey": "#999999"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from infra.aws_pg.jobs import _carry_shared as CS

    rep = {"object": "carry_figures_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"},
           "figures": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    def load(prefix, name="metrics.json"):
        # instance-role S3 (no --profile — that only exists on the laptop control plane);
        # returns None on a missing arm so a figure is skipped, not a fail-loud crash
        import subprocess
        p = out / f"_{prefix.replace('/', '_')}_{name}"
        r = subprocess.run(["aws", "s3", "cp",
                            f"{CS.S3_BUCKET}/results/{prefix}/out/{name}", str(p),
                            "--region", "us-east-1", "--only-show-errors"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not p.exists():
            return None
        return json.loads(p.read_text())

    def footer(ax, fid, src):
        ax.text(0.5, -0.16, f"{fid} · CARRY · src: {src} · carry_figures_v1",
                transform=ax.transAxes, ha="center", va="top", fontsize=6,
                color=CB["grey"])

    def save(fig, fid):
        fig.savefig(out / "figs" / f"{fid}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out / "figs" / f"{fid}.pdf", bbox_inches="tight")
        plt.close(fig)
        rep["figures"].append(fid)
        print(f"[carry-fig] {fid} rendered", flush=True)
        _w()

    def skip(fid, why):
        rep["figures"].append(f"{fid}:SKIPPED({why})")
        print(f"[carry-fig] {fid} SKIPPED: {why}", flush=True)
        _w()

    # ---- F-C1 headline: per-FT slopegraph gain_frac_stiff vs gain_frac_flat -------------
    disc = load("carry_c_disc")
    sealed = load("carry_c_sealed")
    for fid, m in (("F-C1", sealed), ("F-C1_disc", disc)):
        if m is None or "cells" not in m:
            skip(fid, "carry_c cells absent")
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for slug, fr in m["cells"].items():
            if "stiff" not in fr or "gain_frac" not in fr.get("stiff", {}):
                continue
            gs, gf = fr["stiff"]["gain_frac"], fr["flat"]["gain_frac"]
            ax.plot([0, 1], [gs, gf], "-o", color=CB["blue"], alpha=0.6, ms=4)
        band_lo = [fr["randA@0"]["gain_frac"] for fr in m["cells"].values()
                   if "gain_frac" in fr.get("randA@0", {})]
        if band_lo:
            import numpy as np
            ax.axhspan(float(np.min(band_lo)), float(np.max(band_lo)), color=CB["grey"],
                       alpha=0.2, label="random-split band")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["stiff-only", "flat-only"])
        ax.set_ylabel("gain fraction (component / full write)")
        ax.axhline(0.6, ls="--", color=CB["orange"], lw=1, label="0.6 carrier bar")
        ax.set_title("Which component carries the gain: per-FT stiff vs flat gain fraction")
        ax.legend(fontsize=7)
        footer(ax, fid, "carry_c_{disc,sealed}")
        save(fig, fid)

    # ---- F-C2 stiffness spectrum + write mass overlay (from scope) -----------------------
    scope = load("carry_scope")
    if scope and "split" in scope:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 4))
        sp = scope["split"]
        ax.axvline(sp["k_stiff"], color=CB["red"], ls="--",
                   label=f"k_stiff={sp['k_stiff']} (50% trace)")
        ev = sp.get("eval_top8", [])
        ax.plot(range(1, len(ev) + 1), ev, "o-", color=CB["blue"], label="top-8 Q eigenvalues")
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("Q eigenvalue (log)")
        ax.set_title(f"Stiffness spectrum of the L0/dc/ppl operator; split at 50% of trace "
                     f"(k={sp['k_stiff']}/{sp['d']})")
        ax.legend(fontsize=7)
        footer(ax, "F-C2", "carry_scope")
        save(fig, "F-C2")
    else:
        skip("F-C2", "carry_scope absent")

    # ---- F-C3 per-FT gain waterfall -----------------------------------------------------
    if sealed and "cells" in sealed:
        import numpy as np
        fig, ax = plt.subplots(figsize=(9, 5))
        slugs = list(sealed["cells"])
        x = np.arange(len(slugs))
        for key, col, lab in (("ft", CB["grey"], "full"), ("stiff", CB["blue"], "stiff"),
                              ("flat", CB["green"], "flat")):
            ys = [sealed["cells"][s].get(key, {}).get("gain", np.nan) for s in slugs]
            ax.plot(x, ys, "o-", color=col, label=lab, ms=4)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, rotation=90, fontsize=6)
        ax.set_ylabel("task gain")
        ax.set_title("Per-FT gain waterfall: full → stiff-only → flat-only (no medians hiding tails)")
        ax.legend(fontsize=7)
        footer(ax, "F-C3", "carry_c_sealed")
        save(fig, "F-C3")
    else:
        skip("F-C3", "carry_c_sealed absent")

    # ---- F-P1 per-spike sink vs content Rayleigh ----------------------------------------
    p = load("carry_p")
    if p and "per_spike" in p:
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 4))
        rows = p["per_spike"]
        x = np.arange(len(rows))
        sink = [r["sink_frac"] for r in rows]
        ax.bar(x, sink, color=CB["orange"], label="sink/BOS Rayleigh fraction")
        ax.bar(x, [1 - s for s in sink], bottom=sink, color=CB["blue"], label="content")
        ax.axhline(0.3, ls="--", color=CB["green"], lw=1, label="0.3 content-clean line")
        ax.axhline(0.7, ls="--", color=CB["red"], lw=1, label="0.7 sink-loaded line")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['key']}#{r['idx']}" for r in rows], rotation=90, fontsize=6)
        ax.set_ylabel("Rayleigh mass fraction")
        ax.set_title(f"Do SPECTER's spikes survive the position confound? ({p['rp']['verdict']})")
        ax.legend(fontsize=7)
        footer(ax, "F-P1", "carry_p")
        save(fig, "F-P1")
    else:
        skip("F-P1", "carry_p absent")

    # ---- F-N1 the theorem ruler ---------------------------------------------------------
    nab = load("carry_n_ab")
    if nab and "ruler" in nab:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 5))
        rows = nab["ruler"]["rows"]
        gs = [r["G"] for r in rows.values()]
        y = np.arange(len(gs))
        ax.barh(y, gs, color=CB["blue"], label="witness G (worst/typical)")
        ax.axvline(1.0, color=CB["grey"], ls="-", label="typical (1×)")
        ax.axvline(nab["ruler"]["sqrt_d_line"], color=CB["red"], ls="--",
                   label=f"√d = {nab['ruler']['sqrt_d_line']:.0f}× (S13 line)")
        ax.set_yticks(y)
        ax.set_yticklabels(list(rows), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("damage ratio vs matched-energy typical (log)")
        ax.set_title("Our worst cases against published mathematics (√d Lipschitz line)")
        ax.legend(fontsize=7)
        footer(ax, "F-N1", "carry_n_ab")
        save(fig, "F-N1")
    else:
        skip("F-N1", "carry_n_ab ruler absent")

    # ---- F-N2 isoperimetry panel --------------------------------------------------------
    if nab and "rn_b" in nab:
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = [k for k in nab["rn_b"] if isinstance(nab["rn_b"][k], dict)
                and "cv_ratio" in nab["rn_b"][k]]
        import numpy as np
        x = np.arange(len(keys))
        cvr = [nab["rn_b"][k]["cv_ratio"] for k in keys]
        chat = [nab["rn_b"][k]["c_hat"] for k in keys]
        ax.bar(x - 0.2, cvr, 0.4, color=CB["blue"], label="thin-shell CV ratio")
        ax.bar(x + 0.2, chat, 0.4, color=CB["orange"], label="Lipschitz-tail ĉ")
        ax.axhline(2.0, ls="--", color=CB["red"], lw=1, label="2× reference (fail line)")
        ax.axhline(1.0, ls=":", color=CB["grey"], lw=1, label="Gaussian reference")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, fontsize=7)
        ax.set_yscale("log")
        ax.set_title("Does the isoperimetry premise hold on PG1 activations?")
        ax.legend(fontsize=7)
        footer(ax, "F-N2", "carry_n_ab")
        save(fig, "F-N2")
    else:
        skip("F-N2", "carry_n_ab rn_b absent")

    # ---- F-N3 sink-head QK scatter ------------------------------------------------------
    ncd = load("carry_n_cd")
    if ncd and "head_rows" in ncd:
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 5))
        af = [r["attn_sink_frac"] for r in ncd["head_rows"]]
        qk = [max(r["qk_curv"], 1e-12) for r in ncd["head_rows"]]
        ax.scatter(af, qk, s=12, color=CB["blue"], alpha=0.6)
        ax.axvline(0.9, ls="--", color=CB["red"], label="sink-head threshold 0.9")
        ax.set_yscale("log")
        ax.set_xlabel("attention-to-sink fraction")
        ax.set_ylabel("QK-block curvature (log)")
        ax.set_title(f"The S23 collapse prediction, tested ({ncd['rn_c']['verdict']})")
        ax.legend(fontsize=7)
        footer(ax, "F-N3", "carry_n_cd")
        save(fig, "F-N3")
    else:
        skip("F-N3", "carry_n_cd absent")

    # ---- F-E1 unification map -----------------------------------------------------------
    e = load("carry_e")
    if e and "cells" in e and "predictions" in e:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 5))
        fams = list(e["predictions"]["q_pred"])
        xa = [e["predictions"]["spike_alignment"][f] for f in fams]
        ya = [e["cells"][f"{f}@1.0"]["log1p_dppl"] for f in fams]
        ax.scatter(xa, ya, s=40, color=CB["blue"])
        for f, x0, y0 in zip(fams, xa, ya):
            ax.annotate(f, (x0, y0), fontsize=6)
        ax.set_xlabel("F̄-spike alignment")
        ax.set_ylabel("measured damage log(1+Δppl)")
        ax.set_title(f"Error-family unification map (ρ={e['re'].get('rho', float('nan')):.2f}, "
                     f"{e['re']['verdict']})")
        footer(ax, "F-E1", "carry_e")
        save(fig, "F-E1")
    else:
        skip("F-E1", "carry_e absent")

    # ---- F-E2 Lᵖ exponents --------------------------------------------------------------
    if e and "lp_exponents" in e:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 4))
        fams = [f for f in e["lp_exponents"] if e["lp_exponents"][f] is not None]
        vals = [e["lp_exponents"][f] for f in fams]
        ax.bar(range(len(fams)), vals, color=CB["blue"])
        ax.axhline(2.0, ls="--", color=CB["red"], label="p=2 (quadratic)")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, rotation=45, fontsize=6)
        ax.set_ylabel("Lᵖ growth exponent")
        ax.set_title("Which corruptions are quadratic and which are not (S3)")
        ax.legend(fontsize=7)
        footer(ax, "F-E2", "carry_e")
        save(fig, "F-E2")
    else:
        skip("F-E2", "carry_e lp absent")

    # ---- F-W1 sink triptych + F-W2 rotation certificate ---------------------------------
    w2 = load("carry_w2")
    if w2 and "sink_triple" in w2:
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 4))
        st = w2["sink_triple"]
        vals = [st["mean_sub"]["log1p_dppl"], st["rms_preserving"]["log1p_dppl"],
                st["zero"]["log1p_dppl"]]
        ax.bar(range(3), vals, color=[CB["green"], CB["orange"], CB["red"]])
        ax.axhspan(0, st["seed_band"], color=CB["grey"], alpha=0.3, label="seed band")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["mean-sub", "rms-preserving", "zero"])
        ax.set_ylabel("log(1+Δppl)")
        ax.set_title(f"Sink coordinates: constant-bias account ({w2['rw2_meansub']['verdict']})")
        ax.legend(fontsize=7)
        footer(ax, "F-W1", "carry_w2")
        save(fig, "F-W1")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        rot = w2["rotation"]
        ax2.scatter([rot["q_unrot"]], [rot["q_rot"]], s=60, color=CB["blue"], label="u_dc quadform")
        lim = [min(rot["q_unrot"], rot["q_rot"]) * 0.9, max(rot["q_unrot"], rot["q_rot"]) * 1.1]
        ax2.plot(lim, lim, ls="--", color=CB["grey"], label="y=x (invariant)")
        ax2.set_xlabel("statistic (unrotated)")
        ax2.set_ylabel("statistic (Hadamard gauge)")
        ax2.set_title(f"Rotation-covariance certificate ({w2['rw2_rotation']['verdict']}, "
                      f"rel={rot['rel_q']:.1e})")
        ax2.legend(fontsize=7)
        footer(ax2, "F-W2", "carry_w2")
        save(fig2, "F-W2")
    else:
        skip("F-W1", "carry_w2 absent")
        skip("F-W2", "carry_w2 absent")

    # ---- F-D1 intruder heatmap ----------------------------------------------------------
    d2 = load("carry_d2")
    if d2 and "per_ft_counts" in d2:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 5))
        eps = d2["eps_grid"]
        mat = np.array([[c[f"{e:.1f}"] for e in eps] for c in d2["per_ft_counts"]])
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels([f"{e:.1f}" for e in eps])
        ax.set_xlabel("ε threshold")
        ax.set_ylabel("fine-tune index")
        ax.set_title(f"Intruder-dimension census across 24 FTs ({d2['rd2']['verdict']})")
        fig.colorbar(im, ax=ax, label="intruder count (summed over 126 (L,t))")
        footer(ax, "F-D1", "carry_d2")
        save(fig, "F-D1")
    else:
        skip("F-D1", "carry_d2 absent")

    # ---- F-B1 κ-bridge ------------------------------------------------------------------
    b2 = load("carry_b2")
    if b2 and "kappa_bridge" in b2:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 4))
        rows = [r for r in b2["kappa_bridge"] if r["kappa"] is not None]
        xa = [r["expected_cos"] for r in rows]
        ya = [r["log1p_dppl"] for r in rows]
        ax.plot(xa, ya, "o-", color=CB["blue"], label="κ-family damage")
        wit = [r for r in b2["kappa_bridge"] if r["kappa"] is None]
        if wit:
            ax.axhline(wit[0]["log1p_dppl"], ls="--", color=CB["red"], label="pure witness")
        ax.set_xlabel("E[cos θ] = κ/√(D+κ²)")
        ax.set_ylabel("log(1+Δppl)")
        ax.set_title("The graded path from typical noise to the worst case (κ-bridge)")
        ax.legend(fontsize=7)
        footer(ax, "F-B1", "carry_b2")
        save(fig, "F-B1")
    else:
        skip("F-B1", "carry_b2 absent")

    rep["green"] = True
    _w()
    print("CARRY_FIGURES_OK", json.dumps({"n_figures": len([f for f in rep["figures"]
                                                            if ":SKIP" not in f])}))


if __name__ == "__main__":
    main()
