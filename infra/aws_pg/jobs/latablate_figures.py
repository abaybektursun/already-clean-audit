"""latablate figure suite (prereg §10) — weight-free; reads the banked latablate JSONs and writes
the figures. PNG (review) + PDF (paper), colourblind-safe, bbox_inches='tight'. Run on a pg CPU
instance (`--no-weights`); no model, no GPU.

F1 is the MANDATED COMPLETE k-sweep visualization (`feedback_complete_sweep_visualization`): every
k on the grid, ALL component curves overlaid (deleted-floor, intact ceiling, latloc recoverable
ceiling, real RF(k)+CI, random-null band, shuffle-null), with the k* marker and δ_rand margin — no
k hidden, no component dropped.

    python -m infra.aws_pg.jobs.latablate_figures --discovery-prefix results/latablate_disc \
        --confirm-prefix results/latablate_confirm --recoverable-ceiling 0.83 --sha <git-sha>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


# Okabe-Ito colourblind-safe palette.
C_REAL = "#0072B2"      # real-direction RF(k) (blue)
C_RAND = "#999999"      # random-direction null band (grey)
C_SHUF = "#CC79A7"      # shuffle-subspace null (pink)
C_ENERGY = "#A6761D"    # energy-matched null (brown; colourblind-safe ColorBrewer Dark2)
C_FLOOR = "#000000"     # deleted floor (black)
C_CEIL = "#009E73"      # intact ceiling (green)
C_RECOV = "#56B4E9"     # latloc recoverable ceiling (sky)
C_KSTAR = "#D55E00"     # k* marker (vermillion)
C_TAU = "#E69F00"       # τ_restore threshold (orange)


def _pull(prefix, dest):
    """Download <prefix>/out/latablate.json from S3 to dest; return the parsed dict (or None)."""
    if not prefix:
        return None
    from infra.aws_pg import config as C
    uri = f"s3://{C.BUCKET}/{prefix}/out/latablate.json"
    subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", C.REGION], check=True)
    return json.loads(pathlib.Path(dest).read_text())


def _footer(ax, text):
    ax.text(0.0, -0.17, text, transform=ax.transAxes, fontsize=6, color="#555555", va="top")


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140)


def fig_complete_sweep(disc, out, sha, recoverable=None):
    """F1 — the MANDATED complete k-sweep. Every k; ALL components: deleted-floor (RF=0), intact
    ceiling (RF=1), latloc recoverable ceiling (if given), real RF(k)+95% CI, 5-seed random-null
    band (mean±sd), shuffle-null, k* marker, τ_restore line + δ_rand margin shaded above random."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = disc["discovery"]
    sweep = d["sweep"]
    ks = sorted(int(k) for k in sweep)
    x = np.array(ks)
    real = np.array([sweep[str(k)]["rf_real"] for k in ks])
    real_lo = np.array([sweep[str(k)].get("rf_real_ci_lo", sweep[str(k)]["rf_real"]) for k in ks])
    real_hi = np.array([sweep[str(k)].get("rf_real_ci_hi", sweep[str(k)]["rf_real"]) for k in ks])
    rmean = np.array([sweep[str(k)]["rf_random_mean"] for k in ks])
    rsd = np.array([sweep[str(k)]["rf_random_sd"] for k in ks])
    smean = np.array([sweep[str(k)]["rf_shuffle_mean"] for k in ks])
    ssd = np.array([sweep[str(k)]["rf_shuffle_sd"] for k in ks])
    energy = np.array([sweep[str(k)].get("rf_energy", np.nan) for k in ks])
    tau = disc["tau_restore"]
    delta = disc["delta_rand"]
    kstar = d.get("minimal_k_star")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # reference lines (anchors)
    ax.axhline(0.0, color=C_FLOOR, lw=1.0, ls="-", label="deleted floor (RF=0)")
    ax.axhline(1.0, color=C_CEIL, lw=1.0, ls="-", label="intact ceiling (RF=1)")
    if recoverable is not None:
        ax.axhline(recoverable, color=C_RECOV, lw=1.2, ls="-.",
                   label=f"latloc recoverable ceiling (intact-patch L16) = {recoverable:.2f}")
    ax.axhline(tau, color=C_TAU, lw=1.0, ls="--", label=f"τ_restore = {tau}")
    # null bands (random = load-bearing; shuffle = characterization)
    ax.fill_between(x, rmean - rsd, rmean + rsd, color=C_RAND, alpha=0.30,
                    label="random-direction null (5-seed mean±sd)")
    ax.plot(x, rmean, "o--", color=C_RAND, ms=4, lw=1.2)
    ax.fill_between(x, rmean + delta, np.maximum(rmean + delta, real), where=(real > rmean + delta),
                    color=C_REAL, alpha=0.10, label=f"real − random margin > δ_rand={delta}")
    ax.plot(x, smean, "s:", color=C_SHUF, ms=4, lw=1.2, label="shuffle-subspace null (5-seed mean)")
    ax.fill_between(x, smean - ssd, smean + ssd, color=C_SHUF, alpha=0.12)
    if np.isfinite(energy).any():
        ax.plot(x, energy, "D-.", color=C_ENERGY, ms=4, lw=1.2,
                label="energy-matched null (top-k of deleted residual)")
    # real-direction RF(k) with bootstrap CI
    ax.errorbar(x, real, yerr=[real - real_lo, real_hi - real], fmt="o-", color=C_REAL, ms=6, lw=2.0,
                capsize=3, label="real-direction RF(k) (top-k SVD of deleted−intact) + 95% CI")
    if kstar is not None:
        ax.axvline(kstar, color=C_KSTAR, ls=":", lw=1.6, label=f"k* = {kstar} (minimal τ+random)")
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    verdict = ("LOW-RANK removable" if kstar is not None and kstar <= disc["k_small"]
               else "DISTRIBUTED (no small-k fix)")
    ax.set_title(f"Is the L16 deletion a low-rank removable direction? — {verdict}", fontsize=11, pad=10)
    ax.set_xlabel("k  (number of ablated directions, projected out of the DELETED forward at L16)")
    ax.set_ylabel("RF(k) = (acc_ablate(k) − floor) / (intact − floor)")
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _footer(ax, f"F1 COMPLETE k-SWEEP · latablate {disc['eval_split']} · n_eval={disc['n_eval_items']} "
                f"d={disc['d_model']} · real beats RANDOM (load-bearing) by δ; shuffle≈real⇒generic shift "
                f"· SHA {sha} · DISCOVERY")
    _save(fig, out, "latablate_F1_complete_k_sweep")
    plt.close(fig)


def fig_spectrum(disc, out, sha):
    """F2 — singular-value spectrum of (deleted − intact) at L16 + cumulative energy. Steep ⇒ low-rank."""
    import matplotlib.pyplot as plt
    import numpy as np

    sp = disc["singular_spectrum"]
    sv = np.array(sp["singular_values"])
    cum = np.array(sp["cum_energy_fraction"])
    i = np.arange(1, len(sv) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(i, sv / sv[0], "o-", color=C_REAL, ms=4, label="singular value (norm. to σ₁)")
    ax2 = ax.twinx()
    ax2.plot(i, cum, "s--", color=C_CEIL, ms=3, label="cumulative energy fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("cumulative energy fraction", color=C_CEIL)
    ax.set_yscale("log")
    ax.set_xlabel("singular index")
    ax.set_ylabel("σ_i / σ_1 (log)")
    top1 = cum[0] if len(cum) else float("nan")
    ax.set_title(f"(deleted−intact) L16 spectrum — top-1 energy = {top1:.2f}", fontsize=10, pad=8)
    ax.legend(fontsize=7, loc="upper right")
    _footer(ax, f"F2 · latablate · steep spectrum ⇒ low-rank disabling direction · SHA {sha} · DISCOVERY")
    _save(fig, out, "latablate_F2_singular_spectrum")
    plt.close(fig)


def fig_headline(conf, out, sha):
    """F3 — confirmation headline: RF(k*) CI vs τ_restore and vs the equal-k random-direction null."""
    import matplotlib.pyplot as plt

    c = conf["confirmation"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.axhline(conf["tau_restore"], color=C_TAU, ls="--", lw=1, label=f"τ_restore = {conf['tau_restore']}")
    ax.axhline(0, color="#000000", lw=0.6)
    ax.bar([0], [c["rf_real"]], width=0.5, color=C_REAL,
           yerr=[[c["rf_real"] - c["rf_ci_lo"]], [c["rf_ci_hi"] - c["rf_real"]]], capsize=5,
           label=f"real RF(k*={c['k_star']})")
    ax.bar([1], [c["rf_random_mean"]], width=0.5, color=C_RAND,
           yerr=[[c["rf_random_sd"]], [c["rf_random_sd"]]], capsize=5, label="random null (k*)")
    v = "H_lowrank CONFIRMED" if c["verdict_H_lowrank"] else "NOT low-rank (practical leg open)"
    ax.set_title(f"Confirmation at frozen k*={c['k_star']}: {v}\n"
                 f"RF={c['rf_real']:.3f} CI[{c['rf_ci_lo']:.3f},{c['rf_ci_hi']:.3f}] "
                 f"margin={c['margin_over_random']:+.3f}", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real V_k*", "random k*"])
    ax.set_ylabel("RF = (acc_ablate − floor)/(intact − floor)")
    ax.legend(fontsize=7)
    _footer(ax, f"F3 · latablate confirmation · n_eval={conf['n_eval_items']} · "
                f"CI-lower={c['rf_ci_lo']:.3f} vs τ={conf['tau_restore']} · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "latablate_F3_headline")
    plt.close(fig)


def fig_cloze(conf, out, sha):
    """F4 — cloze-under-projector: does ablating V_k* harm answer CONTENT (cloze) while it restores
    the MCQ readout? Content should NOT collapse."""
    import matplotlib.pyplot as plt

    c = conf["confirmation"]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.bar([0, 1], [c["cloze_acc_floor"], c["cloze_acc_ablated"]], width=0.55, color=[C_RAND, C_REAL])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cloze\n(deleted, k=0)", f"cloze\n(+V_k*={c['k_star']})"])
    ax.set_ylabel("MMLU cloze (answer-content) accuracy")
    delta_cloze = c["cloze_acc_ablated"] - c["cloze_acc_floor"]
    keep = "content retained" if delta_cloze > -0.05 else "content HARMED"
    ax.set_title(f"Cloze under the projector: {keep} (Δ={delta_cloze:+.3f})", fontsize=9.5)
    _footer(ax, f"F4 · latablate cloze-under-projector · n_eval={conf['n_eval_items']} · "
                f"SHA {sha} · CONFIRMATORY")
    _save(fig, out, "latablate_F4_cloze_under_projector")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--discovery-prefix", default="")
    ap.add_argument("--confirm-prefix", default="")
    ap.add_argument("--recoverable-ceiling", type=float, default=None,
                    help="latloc intact-patch RF(correct,L16), recomputed from the latloc bank @ SHA")
    ap.add_argument("--sha", default="uncommitted")
    args = ap.parse_args()

    import importlib.util
    import sys
    if importlib.util.find_spec("matplotlib") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib"], check=True)

    out = pathlib.Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)

    disc = _pull(args.discovery_prefix, out / "_disc.json")
    conf = _pull(args.confirm_prefix, out / "_confirm.json")
    if disc and disc.get("discovery"):
        fig_complete_sweep(disc, out, args.sha, recoverable=args.recoverable_ceiling)
        fig_spectrum(disc, out, args.sha)
        print("[latablate-fig] F1 (complete k-sweep) + F2 (spectrum) written")
    if conf and conf.get("confirmation"):
        fig_headline(conf, out, args.sha)
        fig_cloze(conf, out, args.sha)
        print("[latablate-fig] F3 (headline) + F4 (cloze) written")
    print("[latablate-fig] done")


if __name__ == "__main__":
    main()
