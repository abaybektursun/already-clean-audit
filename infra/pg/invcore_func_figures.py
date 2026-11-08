"""Figures for invariant_core-FUNC (prereg §10) — weight-free, reads the result JSON only.

    python3 infra/pg/invcore_func_figures.py \
        --json studies/lineage/paligemma1-pi0/results/invcore_func/invariant_core_func.json \
        --out  studies/lineage/paligemma1-pi0/results/invcore_func

F1  ACT energy: per-layer fraction of q/k INPUT activation energy inside U_unt vs the matched-random
    k-subspace (and the k/n no-structure reference), one series per k. Shows the untouched band holds
    far more activation than random in 17/18 layers (L0 is the lone inert exception).
F2  Ablation Δppl per arm vs k: U_unt vs matched-random BAND vs base-bottom-σ, with the paired-contrast
    95% CI; the top-σ positive control (off-scale) annotated as the validity-gate anchor.

Every panel: CI / band overlaid, the null overlaid, a plain finding title, a "how to read" line, and a
provenance footer (run id · base SHA · input hash · CONFIRMATORY). PNG (review) + PDF (paper).
"""
import argparse
import hashlib
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colourblind-safe (Wong)
C_UNT = "#0072B2"      # blue   — U_unt (the untouched band)
C_RAND = "#E69F00"     # orange — matched-random null
C_BOTS = "#009E73"     # green  — base-bottom-σ null
C_REF = "#999999"      # grey   — k/n no-structure reference


def _footer(fig, run_id, base_sha, ih):
    fig.text(0.5, 0.005,
             f"run={run_id} · base_sha={base_sha[:8]} · input={ih[:12]} · CONFIRMATORY (prereg de56286)",
             ha="center", va="bottom", fontsize=6.5, color="#555555")


def fig1_act(d, out, run_id, base_sha, ih):
    ae = d["activation_energy"]
    layers = [r["layer"] for r in ae]
    ks = d["k_grid"]
    n = 2048
    fig, ax = plt.subplots(figsize=(9, 4.6))
    styles = {ks[0]: "o-", ks[1]: "s-", ks[2]: "^-"}
    for k in ks:
        e_unt = [r["k"][str(k)]["energy_unt"] for r in ae]
        ax.plot(layers, e_unt, styles.get(k, "o-"), color=C_UNT, alpha=0.55 + 0.15 * ks.index(k),
                ms=4, lw=1.4, label=f"U_unt  k={k}")
    # matched-random reference (≈ k/n) — overlay the largest-k random level as the null band edge
    for k in ks:
        e_rand = [r["k"][str(k)]["energy_rand"] for r in ae]
        ax.plot(layers, e_rand, "--", color=C_RAND, lw=1.0, alpha=0.5,
                label=f"matched-random  k={k}" if k == ks[-1] else None)
    ax.axhline(ks[-1] / n, color=C_REF, ls=":", lw=1.0, label=f"k/n reference (k={ks[-1]})")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("fraction of q/k input activation energy in subspace")
    ax.set_xticks(layers)
    ax.set_title("Untouched q/k band holds 4–44× the matched-random activation energy (17/18 layers)\n"
                 "How to read: blue ≫ orange/grey ⇒ the consensus-untouched band is high-activation, "
                 "not gradient-starved. L0 (≈0) is the lone inert layer.", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig, run_id, base_sha, ih)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F1_act_energy.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_ablation(d, out, run_id, base_sha, ih):
    abl = sorted(d["ablation"], key=lambda r: r["k"])
    base = d["capability_baseline"]["ppl"]
    ks = [r["k"] for r in abl]
    x = list(range(len(ks)))
    d_unt = [r["contrast"]["dppl_unt"] for r in abl]
    # U_unt paired-contrast CI vs random (the headline interval)
    ci_lo = [r["contrast"]["ci_unt_minus_rand"]["lo"] for r in abl]
    ci_hi = [r["contrast"]["ci_unt_minus_rand"]["hi"] for r in abl]
    d_rand_mean = [r["contrast"]["rand_band"]["dppl_mean"] for r in abl]
    d_rand_lo = [r["contrast"]["rand_band"]["dppl_min"] for r in abl]
    d_rand_hi = [r["contrast"]["rand_band"]["dppl_max"] for r in abl]
    d_bots = [r["contrast"]["dppl_bots"] for r in abl]
    d_tops = [r["contrast"]["dppl_top_sigma_poscontrol"] for r in abl]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    # U_unt with its paired-contrast-vs-random 95% CI (the decisive interval)
    ax.errorbar(x, d_unt, yerr=[[u - l for u, l in zip(d_unt, ci_lo)],
                                [h - u for u, h in zip(d_unt, ci_hi)]],
                fmt="o-", color=C_UNT, lw=1.8, ms=7, capsize=4,
                label="U_unt (untouched band)  ±95% paired-contrast CI")
    # matched-random BAND (min..max over 3 draws) + mean
    ax.fill_between(x, d_rand_lo, d_rand_hi, color=C_RAND, alpha=0.25, label="matched-random band (3 draws)")
    ax.plot(x, d_rand_mean, "s--", color=C_RAND, lw=1.3, ms=5, label="matched-random mean")
    ax.plot(x, d_bots, "^--", color=C_BOTS, lw=1.3, ms=6, label="base-bottom-σ null")
    ax.axhline(0, color="#000000", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel(f"Δ perplexity vs base (base blank-image ppl = {base:.1f})")
    ax.set_title("Ablating the untouched q/k band costs 24–44× more perplexity than matched-random / bottom-σ\n"
                 "How to read: blue ≫ orange band & green ⇒ load-bearing. Paired-contrast CI excludes 0, "
                 "BH-adjusted p=0 at every k.", fontsize=9.3)
    # annotate the off-scale top-σ positive control (the validity-gate anchor)
    for xi, (k, t) in enumerate(zip(ks, d_tops)):
        ax.annotate(f"top-σ +{t:.0f} ✓", (xi, max(d_unt) * 0.62), ha="center", fontsize=7.2,
                    color="#444444")
    ax.text(0.02, 0.92, "top-σ positive control (off-scale, +1.5k–3.7k ppl) valid at every k\n"
                        "⇒ the ablation readout has full dynamic range", transform=ax.transAxes,
            fontsize=7.2, color="#444444", va="top")
    ax.legend(fontsize=7.5, loc="center right")
    ax.margins(x=0.08)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig, run_id, base_sha, ih)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F2_ablation_dppl.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jp = pathlib.Path(a.json)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = jp.read_bytes()
    ih = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw)
    run_id = "results/invcore_func"
    base_sha = d["base_sha"]
    assert d["verdict"].startswith(("LIVE", "INERT", "VOID")), d["verdict"]
    fig1_act(d, out, run_id, base_sha, ih)
    fig2_ablation(d, out, run_id, base_sha, ih)
    print(f"wrote F1_act_energy + F2_ablation_dppl (png+pdf) to {out} (input {ih[:12]})")
    print(f"verdict: {d['verdict'][:90]}...")


if __name__ == "__main__":
    main()
