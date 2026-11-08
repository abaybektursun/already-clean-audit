"""latloc figure suite (prereg §10) — weight-free; reads the banked Arm-1 JSONs and writes the
figures. Each panel carries: a CI band / error bars, the NULL overlaid, a plain-language finding in
the title, a one-line "how to read", and a provenance footer (run prefix · SHA · tag). PNG (review)
+ PDF (paper), colourblind-safe, bbox_inches='tight'. Run on a pg CPU instance (`--no-weights`); no
model, no GPU.

    python -m infra.aws_pg.jobs.latloc_figures --discovery-prefix results/latloc_disc \
        --confirm-prefix results/latloc_confirm --sha <git-sha>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


# Okabe-Ito colourblind-safe palette.
C_DRF = "#0072B2"      # ΔRF (blue)
C_NULL = "#999999"     # null / wrong-donor band (grey)
C_ONSET = "#D55E00"    # onset marker (vermillion)
C_DELTA = "#009E73"    # δ threshold (green)
C_BAR = "#0072B2"
C_BAR2 = "#E69F00"


def _pull(prefix, dest):
    """Download <prefix>/out/latloc_arm1.json from S3 to dest; return the parsed dict (or None)."""
    if not prefix:
        return None
    from infra.aws_pg import config as C
    uri = f"s3://{C.BUCKET}/{prefix}/out/latloc_arm1.json"
    subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", C.REGION], check=True)
    return json.loads(pathlib.Path(dest).read_text())


def _footer(ax, text):
    ax.text(0.0, -0.16, text, transform=ax.transAxes, fontsize=6, color="#555555", va="top")


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140)


def fig_profile(disc, out, sha):
    """F1 — discovery ΔRF-vs-depth profile with per-layer 95% CI, the 5-seed wrong-donor SD band,
    the δ line, and the stable-onset marker."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = disc["discovery"]
    prof = np.array(d["delta_rf_profile"])
    lo = np.array(d["delta_rf_ci_lo"])
    hi = np.array(d["delta_rf_ci_hi"])
    sd = np.array(d["delta_rf_seed_sd"])
    L = np.arange(len(prof))
    onset = d["stable_onset_depth"]
    delta = disc["delta"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(delta, color=C_DELTA, ls="--", lw=1, label=f"δ = {delta}")
    ax.axhline(0, color="#000000", lw=0.6)
    ax.fill_between(L, prof - sd, prof + sd, color=C_NULL, alpha=0.35, label="5-seed wrong-donor SD")
    ax.errorbar(L, prof, yerr=[prof - lo, hi - prof], fmt="o-", color=C_DRF, ms=4, lw=1.4,
                capsize=2, label="ΔRF (correct − wrong-donor)")
    if onset is not None:
        ax.axvline(onset, color=C_ONSET, ls=":", lw=1.5, label=f"stable onset L{onset}")
    band = "FINAL-layer (trivial)" if (onset is not None and onset >= len(prof) - 2) else \
        ("mid-stack (NOVEL)" if onset is not None else "no stable onset")
    ax.set_title(f"Item-specific restoration: stable onset L{onset} — {band}", fontsize=10, pad=10)
    ax.set_xlabel("decoder layer L (patched intact→deleted residual)")
    ax.set_ylabel("ΔRF(L) = RF(correct) − RF(wrong-donor)")
    ax.legend(fontsize=7, loc="upper left")
    _footer(ax, f"F1 · latloc discovery · n_used={disc['n_items_used']} retention="
                f"{disc['length_bias']['retention']:.0%} · onset=earliest depth ΔRF stays ≥ δ "
                f"(final 1-2 layers=trivial, mid-stack=novel) · SHA {sha} · DISCOVERY")
    _save(fig, out, "F1_delta_rf_profile")
    plt.close(fig)


def fig_headline(conf, out, sha):
    """F3 — confirmation headline: ΔRF(L_d) at the FROZEN onset with 95% CI vs δ (the verdict)."""
    import matplotlib.pyplot as plt

    h = conf["headline"]
    delta = conf["delta"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.axhline(delta, color=C_DELTA, ls="--", lw=1, label=f"δ = {delta}")
    ax.axhline(0, color="#000000", lw=0.6)
    ax.bar([0], [h["delta_rf"]], width=0.5, color=C_BAR,
           yerr=[[h["delta_rf"] - h["ci_lo"]], [h["ci_hi"] - h["delta_rf"]]], capsize=5)
    verdict = "exceeds δ (item-specific)" if h["exceeds_delta"] else "within δ (not item-specific)"
    nov = "NOVEL (mid-stack)" if h["novel_per_gate"] else "trivial-confirmed (final-layer)"
    ax.set_title(f"Confirmation headline at frozen L{h['layer']}: {verdict}\n{nov}", fontsize=9.5)
    ax.set_xticks([0]); ax.set_xticklabels([f"ΔRF(L{h['layer']})"])
    ax.set_ylabel("ΔRF = RF(correct) − RF(wrong-donor)")
    ax.legend(fontsize=7)
    _footer(ax, f"F3 · latloc confirmation · n_used={conf['n_items_used']} · "
                f"CI-lower={h['ci_lo']:.3f} · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "F3_headline")
    plt.close(fig)


def fig_corrupt(conf, out, sha):
    """F4 — symmetric corrupt patch at L_d: intact MCQ degradation, correct-donor vs wrong-donor."""
    import matplotlib.pyplot as plt

    c = conf.get("corrupt_patch")
    if not c:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.bar([0, 1], [c["degrade_correct"], c["degrade_wrong"]], width=0.55,
           color=[C_BAR, C_BAR2])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["correct-donor\n(this item)", "wrong-donor\n(other item)"])
    ax.set_ylabel("intact MCQ degradation (clean − corrupted)")
    spec = "item-specific (correct degrades MORE)" if c["item_specific"] else "NOT item-specific"
    ax.set_title(f"Corrupt patch at L{c['layer']} is {spec}\n"
                 f"gap={c['item_specific_gap']:.3f} CI[{c['gap_ci_lo']:.3f},{c['gap_ci_hi']:.3f}]",
                 fontsize=9.5)
    _footer(ax, f"F4 · latloc corrupt-direction · n_used={conf['n_items_used']} · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "F4_corrupt_patch")
    plt.close(fig)


def fig_arm3_verdict(disc, arm3, out, sha):
    """F7 — the latent-vs-erased verdict as TWO INDEPENDENT per-generation categorical panels. NO
    shared numeric axis / normalized depth (firewall): each panel shows only its own model's onset
    layer (out of its own decoder count) and its categorical band {early|mid|late|none}."""
    import matplotlib.pyplot as plt

    panels = []
    if disc and disc.get("discovery"):
        d = disc["discovery"]
        onset = d["stable_onset_depth"]
        n = len(d["delta_rf_profile"])
        band = ("none" if onset is None else
                ("late" if onset >= -(-2 * n // 3) else ("early" if onset < n // 3 else "mid")))
        panels.append(("Gemma-2 PG2 (latent)", onset, n, band))
    if arm3:
        panels.append((f"Gemma-1 π0 (erased){'' if arm3['adjudicable'] else ' — cannot adjudicate'}",
                       arm3["within_g1_stable_onset"], arm3["firewall_n_layers"], arm3["band_label"]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, onset, n, band) in zip(axes, panels):
        ax.barh([0], [n], color="#EEEEEE", height=0.5)
        if onset is not None:
            ax.axvline(onset, color=C_ONSET, lw=2.5)
            ax.text(onset, 0.0, f" L{onset}", color=C_ONSET, fontsize=9, ha="left", va="center")
        ax.set_title(f"{title}\nonset L{onset} · band = {band.upper()}", fontsize=9)
        ax.set_xlim(0, n - 1); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
        ax.set_xlabel(f"this model's own decoder depth (0..{n - 1})")
    fig.suptitle("Latent vs erased — independent per-generation labels (no shared axis)",
                 fontsize=10, y=1.04)
    fig.subplots_adjust(top=0.78, wspace=0.2)
    _footer(axes[0], f"F7 · latloc Arm-3 · firewall: labels only, never numerically compared · SHA {sha}")
    _save(fig, out, "F7_arm3_verdict")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--discovery-prefix", default="")
    ap.add_argument("--confirm-prefix", default="")
    ap.add_argument("--arm3-prefix", default="")
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
    arm3 = None
    if args.arm3_prefix:   # latloc_arm3 banks latloc_arm3.json (not latloc_arm1.json)
        from infra.aws_pg import config as C
        uri = f"s3://{C.BUCKET}/{args.arm3_prefix}/out/latloc_arm3.json"
        subprocess.run(["aws", "s3", "cp", uri, str(out / "_arm3.json"), "--region", C.REGION], check=True)
        arm3 = json.loads((out / "_arm3.json").read_text())
    if disc and disc.get("discovery"):
        fig_profile(disc, out, args.sha)
        print("[latloc-fig] F1 written")
    if conf and conf.get("headline"):
        fig_headline(conf, out, args.sha)
        print("[latloc-fig] F3 written")
    if conf and conf.get("corrupt_patch"):
        fig_corrupt(conf, out, args.sha)
        print("[latloc-fig] F4 written")
    if disc or arm3:
        fig_arm3_verdict(disc, arm3, out, args.sha)
        print("[latloc-fig] F7 written")
    print("[latloc-fig] done")


if __name__ == "__main__":
    main()
