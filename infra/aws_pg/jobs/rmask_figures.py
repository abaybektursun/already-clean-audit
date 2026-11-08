"""rmask figure suite (prereg §10) — weight-free; reads the banked rmask JSON and writes the figures.
PNG (review) + PDF (paper), colourblind-safe (Okabe-Ito), bbox_inches='tight'. Run on a pg CPU instance
(`--no-weights`); no model, no GPU.

F1 is the MANDATED DECODABILITY-vs-DEPTH profile (`feedback_complete_sweep_visualization`): every layer,
intact-probe (ceiling curve) + deleted-probe (test curve) + chance + deleted/intact MCQ + the L_e floor +
shuffled-null band + the L16 headline marker. No layer hidden, no curve dropped.

    python -m infra.aws_pg.jobs.rmask_figures --prefix results/rmask --sha <git-sha>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

C_DEL = "#0072B2"     # deleted-model probe (blue)
C_INT = "#009E73"     # intact-model probe / ceiling (green)
C_FLOOR = "#A6761D"   # early-layer surface floor (brown)
C_SHUF = "#999999"    # shuffled null (grey)
C_MCQ = "#CC79A7"     # model's own MCQ (pink)
C_CHANCE = "#000000"  # chance (black)
C_HEAD = "#D55E00"    # L16 headline marker (vermillion)


def _pull(prefix, dest):
    if not prefix:
        return None
    from infra.aws_pg import config as C
    subprocess.run(["aws", "s3", "cp", f"s3://{C.BUCKET}/{prefix}/out/rmask.json", str(dest),
                    "--region", C.REGION], check=True)
    return json.loads(pathlib.Path(dest).read_text())


def _footer(ax, text):
    ax.text(0.0, -0.17, text, transform=ax.transAxes, fontsize=6, color="#555555", va="top")


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140)


def fig_profile(d, out, sha):
    """F1 — decodability vs depth. Intact-probe (ceiling), deleted-probe (test), chance, deleted/intact MCQ,
    the L_e floor, shuffled-null band, the L16 headline. mask ⟺ deleted tracks intact at L16; loss ⟺ deleted
    at the floor while intact decodes."""
    import matplotlib.pyplot as plt
    import numpy as np

    pa = d["profile_acc"]
    layers = sorted(int(L) for L in pa["deleted"])
    x = np.array(layers)
    dele = np.array([pa["deleted"][str(L)] for L in layers])
    inta = np.array([pa["intact"][str(L)] for L in layers])
    hl = d["headline"]; chance = d["chance"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(chance, color=C_CHANCE, lw=1.0, ls="-", label=f"chance = {chance}")
    ax.axhline(d["mcq"]["deleted"], color=C_MCQ, lw=1.0, ls=":", label=f"deleted model MCQ = {d['mcq']['deleted']:.2f}")
    ax.axhline(d["mcq"]["intact"], color=C_INT, lw=1.0, ls=":", alpha=0.6, label=f"intact model MCQ = {d['mcq']['intact']:.2f}")
    ax.axhspan(chance, hl["shuffled_null_mean"], color=C_SHUF, alpha=0.20, label=f"shuffled-null ≈ {hl['shuffled_null_mean']:.2f}")
    ax.plot(x, inta, "o-", color=C_INT, ms=6, lw=2.0, label="INTACT-probe acc (ceiling)")
    ax.plot(x, dele, "s-", color=C_DEL, ms=6, lw=2.0, label="DELETED-probe acc (the test)")
    ax.axvline(d["L_floor"], color=C_FLOOR, ls="--", lw=1.2, label=f"L_e floor = {d['L_floor']}")
    ax.axvline(d["L_headline"], color=C_HEAD, ls=":", lw=1.6, label=f"L_head = {d['L_headline']}")
    ax.set_xticks(layers); ax.set_xlabel("decoder layer (last-token residual)")
    ax.set_ylabel("held-out 4-way probe accuracy (gold answer)")
    ax.set_ylim(min(chance - 0.05, dele.min() - 0.03), 1.02)
    v = d["verdict"]
    ax.set_title(f"Is the answer still decodable from the DELETED residual? — {v}", fontsize=11, pad=10)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _footer(ax, f"F1 DECODABILITY×DEPTH · rmask · n_conf={d['n_conf']} d_model={d['d_model']} · deleted tracks "
                f"intact ⇒ MASK; deleted at floor ⇒ LOSS · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "rmask_F1_decodability_profile")
    plt.close(fig)


def fig_headline(d, out, sha):
    """F2 — the headline contrast at L_head: deleted-L16 probe vs {deleted-MCQ, L_e floor, intact ceiling,
    shuffled null} with CIs, vs chance."""
    import matplotlib.pyplot as plt

    h = d["headline"]; ch = d["chance"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.axhline(ch, color=C_CHANCE, lw=0.8, ls="-", label=f"chance = {ch}")
    vals = [("deleted\nL_head", h["deleted_L16_acc"], h["deleted_L16_ci"], C_DEL),
            ("intact\nceiling", h["intact_L16_acc"], None, C_INT),
            ("L_e\nfloor", h["deleted_floor_acc"], None, C_FLOOR),
            ("deleted\nMCQ", d["mcq"]["deleted"], None, C_MCQ),
            ("shuffled\nnull", h["shuffled_null_mean"], None, C_SHUF)]
    for i, (lab, v, ci, c) in enumerate(vals):
        yerr = [[v - ci[0]], [ci[1] - v]] if ci else None
        ax.bar([i], [v], width=0.6, color=c, yerr=yerr, capsize=5)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([v[0] for v in vals], fontsize=8)
    ax.set_ylabel("held-out 4-way probe / MCQ accuracy")
    ax.set_title(f"L_head={d['L_headline']}: {d['verdict']}\n"
                 f"mask_gap {h['masking_gap']:+.3f} · comp_gap {h['computation_gap']:+.3f} "
                 f"CI-lo {h['computation_gap_ci'][0]:+.3f} · ceil_gap {h['ceiling_gap']:+.3f}", fontsize=9)
    _footer(ax, f"F2 · rmask · n_conf={d['n_conf']} · δ_decode={d['delta_decode']} γ={d['gamma_ceiling']} "
                f"· SHA {sha} · CONFIRMATORY")
    _save(fig, out, "rmask_F2_headline")
    plt.close(fig)


def fig_dataeff(d, out, sha):
    """F3 — data-efficiency at L_head (deleted): a PRESERVED signal saturates by few examples (H_mask read
    cheaply); a RE-LEARNED head keeps climbing (REPRICE-class). When the deleted probe is AT CHANCE at every
    size (H_loss), the saturation index is a ratio of two near-zero differences ⇒ N/A — plotted on a
    chance→ceiling y-axis so the flatness-at-chance is not misread as a climbing trend."""
    import matplotlib.pyplot as plt
    import numpy as np

    de = d["data_efficiency_deleted"]
    sizes = sorted(int(k) for k in de if k != "all")
    x = np.array(sizes); y = np.array([de[str(k)] for k in sizes])
    ch = d["chance"]; ceil = d["headline"]["intact_L16_acc"]
    at_chance = de["all"] <= ch + d["delta_decode"]          # no real signal → saturation is meaningless
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.axhline(ch, color=C_CHANCE, lw=0.9, label=f"chance = {ch}")
    ax.axhline(ceil, color=C_INT, ls=":", lw=1.4, label=f"intact ceiling (L{d['L_headline']}) = {ceil:.2f}")
    ax.axhline(de["all"], color=C_DEL, ls="--", lw=1.0, alpha=0.7, label=f"deleted all-disc = {de['all']:.3f}")
    ax.plot(x, y, "s-", color=C_DEL, ms=6, lw=2.0, label="deleted-probe acc vs train size")
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("probe training-set size (discovery)")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(ch - 0.03, ceil + 0.03)                       # chance→ceiling: flatness-at-chance is honest
    sat = d.get("saturation_index_deleted")
    if at_chance:
        ax.set_title(f"Data-efficiency @L{d['L_headline']} (deleted): FLAT AT CHANCE at every train size\n"
                     f"→ saturation index ({sat:.2f}) is N/A — no signal to be data-efficient about (H_loss)",
                     fontsize=8.8)
    else:
        ax.set_title(f"Data-efficiency @L{d['L_headline']}: saturation index = {sat:.2f}\n"
                     f"(≈1 ⇒ preserved signal READ; ≪1 & climbing ⇒ head RE-LEARNED)", fontsize=9.0)
    ax.legend(fontsize=7, loc="center right")
    _footer(ax, f"F3 · rmask · preserved-vs-repriced discriminator · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "rmask_F3_data_efficiency")
    plt.close(fig)


def fig_alignment(d, out, sha):
    """F4 — readout-alignment (rough cross-layer descriptor): the probe class direction vs the model's
    choice-letter unembedding, deleted + intact, vs a random-direction null."""
    import matplotlib.pyplot as plt

    ra = d["readout_alignment"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    labs, vals, cols = [], [], []
    for name, c in (("deleted", C_DEL), ("intact", C_INT)):
        r = ra.get(name)
        if r:
            labs.append(f"{name}\nprobe·readout"); vals.append(r["mean_abscos"]); cols.append(c)
            labs.append(f"{name}\nrandom null"); vals.append(r["random_null_mean"]); cols.append(C_SHUF)
    ax.bar(range(len(vals)), vals, width=0.6, color=cols)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("|cosine| (probe class dir · choice-letter unembedding)")
    ax.set_title("F4 — is the decoded direction the model's OWN readout?\n(rough cross-layer descriptor)", fontsize=9.5)
    _footer(ax, f"F4 · rmask · L{d['L_headline']} probe vs final-layer unembedding (cross-layer proxy) · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "rmask_F4_readout_alignment")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--sha", default="uncommitted")
    args = ap.parse_args()
    import importlib.util
    import sys
    if importlib.util.find_spec("matplotlib") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib"], check=True)
    out = pathlib.Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    d = _pull(args.prefix, out / "_rmask.json")
    if d:
        fig_profile(d, out, args.sha)
        fig_headline(d, out, args.sha)
        fig_dataeff(d, out, args.sha)
        fig_alignment(d, out, args.sha)
        print("[rmask-fig] F1 (profile) + F2 (headline) + F3 (data-eff) + F4 (alignment) written")
    print("[rmask-fig] done")


if __name__ == "__main__":
    main()
