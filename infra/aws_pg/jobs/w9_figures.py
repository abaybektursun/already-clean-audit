"""W9 figure suite (§10.2) — each figure illustrates a claim independently backed by a number +
CI + null. PNG (review) + PDF (paper), null overlaid, provenance footer. GPU-independent: reads
the unblind metrics.json + controls.json + per-cell features_*.npz (download from S3 first).

    python -m infra.aws_pg.jobs.w9_figures --metrics metrics.json --controls controls.json \
        --features-dir <dir-of-npz> --out <dir> --freeze-sha 18abbf5

Implemented (data-backed): novel_spectrum, kl_faithfulness, fve_vs_faithfulness, stability,
controls. Deferred (need data not harvested this pass — logged, not silently dropped):
  - cross_instrument (per-feature Diff-SAE↔crosscoder matching across distinct dictionaries)
  - delta_map (per-feature image-token mass)
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _footer(fig, prov):
    fig.text(0.005, 0.005, prov, fontsize=5, color="#666", ha="left", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-sha", default="")
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    M = json.loads(pathlib.Path(args.metrics).read_text())
    C = json.loads(pathlib.Path(args.controls).read_text())
    fdir = pathlib.Path(args.features_dir) if args.features_dir else None
    prov = f"W9 Diff-SAE · freeze {args.freeze_sha} · gpu {M.get('gpu')} · {M.get('torch')}"
    layers = sorted(M["layers"].keys(), key=int)
    widths = M.get("widths", [16, 32])
    CB = {"diffsae": "#0072B2", "cc_recon": "#D55E00", "cc_error": "#009E73", "null": "#999999"}

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {name}")

    # 1) novel spectrum — headline re-weighting-vs-new chart (L20, base width, seed 0).
    if fdir is not None:
        w0 = widths[0]
        f = fdir / f"features_L20_x{w0}_s0.npz"
        if f.exists():
            z = np.load(f)
            e = z["out_of_span_energy"]; nf = z["novel_flag"]
            floor = M["layers"]["20"]["grid"][f"x{w0}_s0"]["out_of_span_floor_pct95"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(e, bins=60, color=CB["diffsae"], alpha=0.8)
            ax.axvline(floor, color=CB["null"], ls="--", lw=1.5,
                       label=f"N3 floor (95th pct)={floor:.3f}")
            ax.set_xlabel("decoder direction out-of-span energy (method i)")
            ax.set_ylabel("# Diff-SAE features")
            ax.set_title(f"W9 L20: does the LM-rewrite delta live in Gemma's existing span?\n"
                         f"novel fraction ρ={nf.mean():.4f} ({int(nf.sum())}/{len(nf)})")
            ax.legend(fontsize=8)
            _footer(fig, prov); save(fig, "fig_w9_novel_spectrum")

    # 2) KL faithfulness — the crux (per layer: diffsae / cc_recon / cc_error R_KL bars + CI).
    w0 = widths[0]
    def _rkl(L, key):
        return M["layers"][L]["R_KL"]["R_KL"].get(key, {})
    haveR = [L for L in layers if "R_KL" in M["layers"][L]]
    if haveR:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(haveR)); w = 0.25
        for i, key in enumerate([f"diffsae_x{w0}", "crosscoder_recon", "crosscoder_error"]):
            vals = [_rkl(L, key).get("point", float("nan")) for L in haveR]
            err = [[_rkl(L, key).get("point", 0) - _rkl(L, key).get("ci", {}).get("lo", _rkl(L, key).get("point", 0)) for L in haveR],
                   [_rkl(L, key).get("ci", {}).get("hi", _rkl(L, key).get("point", 0)) - _rkl(L, key).get("point", 0) for L in haveR]]
            ax.bar(x + (i-1)*w, vals, w, yerr=err, capsize=2, label=key.replace(f"_x{w0}", ""),
                   color=[CB["diffsae"], CB["cc_recon"], CB["cc_error"]][i])
        ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in haveR])
        ax.axhline(1.0, color=CB["null"], ls=":", lw=1, label="full delta recovered")
        ax.set_ylabel("R_KL (behavioural KL recovered)")
        ax.set_title("W9 crux: is the crosscoder residual-blind?\n"
                     "(cc_error ≈ cc_recon ⇒ crosscoder leaves behavioural signal unmodelled)")
        ax.legend(fontsize=8); _footer(fig, prov); save(fig, "fig_w9_kl_faithfulness")

    # 3) FVE vs faithfulness — the lesson plot (high explained variance ≠ faithfulness).
    if haveR:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for L in haveR:
            cell = M["layers"][L]["grid"].get(f"x{w0}_s0")
            if not cell:
                continue
            fve = 1.0 - cell["fvu"]; rk = _rkl(L, f"diffsae_x{w0}").get("point", float("nan"))
            ax.scatter(fve, rk, color=CB["diffsae"], label="Diff-SAE" if L == haveR[0] else None, s=40)
            ax.annotate(f"L{L}", (fve, rk), fontsize=7)
        ax.set_xlabel("FVE = 1 − FVU (fit quality)"); ax.set_ylabel("R_KL (faithfulness)")
        ax.set_title("W9 lesson: fit quality (FVE) ≠ behavioural faithfulness (R_KL)")
        ax.legend(fontsize=8); _footer(fig, prov); save(fig, "fig_w9_fve_vs_faithfulness")

    # 4) stability — ρ_novel heatmap across (layer × width), seed-mean ± across-seed.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    grid = np.full((len(layers), len(widths)), np.nan)
    for r, L in enumerate(layers):
        for c, wexp in enumerate(widths):
            v = M["layers"][L]["grid"].get(f"x{wexp}_rho_seed_mean")
            if v is not None:
                grid[r, c] = v
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(widths))); ax.set_xticklabels([f"×{w}" for w in widths])
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{L}" for L in layers])
    for r in range(len(layers)):
        for c in range(len(widths)):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, f"{grid[r,c]:.3f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="ρ_novel (seed mean)")
    ax.set_title("W9 stability: ρ_novel across layer × dict width (C4)")
    _footer(fig, prov); save(fig, "fig_w9_stability")

    # 5) controls — the panel that licenses 1-4 (positive recovery, N4 collapse, N3 floors).
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    Lc = sorted(C["layers"].keys(), key=int)
    pos = [C["layers"][L]["n4_positive"]["positive_recovered_max_cos"] for L in Lc]
    n4 = [C["layers"][L]["n4_positive"]["n4_frac_above_n3_floor"] for L in Lc]
    axes[0].bar([f"L{L}" for L in Lc], pos, color=CB["diffsae"])
    axes[0].axhline(0.8, color=CB["null"], ls="--", label="recovery gate 0.8")
    axes[0].set_title("positive control\n(planted dir recovered)"); axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7)
    axes[1].bar([f"L{L}" for L in Lc], n4, color=CB["cc_error"])
    axes[1].axhline(0.05, color=CB["null"], ls="--", label="N3 floor ≈0.05")
    axes[1].set_title("N4 shuffle collapse\n(frac above N3 floor)"); axes[1].legend(fontsize=7)
    for L in Lc:
        fl = [C["layers"][L]["widths"][str(w)]["n3_out_of_span_floor_pct95"] for w in widths]
        axes[2].plot([f"×{w}" for w in widths], fl, marker="o", label=f"L{L}")
    axes[2].set_title("N3 random-delta floor\n(per width)"); axes[2].legend(fontsize=7)
    fig.suptitle("W9 controls — the instrument works (licenses figs 1-4)")
    _footer(fig, prov); save(fig, "fig_w9_controls")

    print("[fig] DEFERRED (data not harvested this pass): cross_instrument (cross-dict feature "
          "matching), delta_map (per-feature image-token mass).")


if __name__ == "__main__":
    main()
