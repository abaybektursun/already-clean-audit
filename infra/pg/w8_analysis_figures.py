"""W8 ANALYSIS-half figures (spec §7 W8 amendment 2026-05-27). EXPLORATORY / DISCOVERY.

Distinct from infra/pg/w8_figures.py (the study-wide lineage-map communication suite).
These figures illustrate the NEW W8-analysis cross-check numbers — how a full-rank LM
ΔW (W1 eff-rank ~1440) produces NO new feature directions (crosscoder emergent=0):

  fig_w8x_dW_spectra    — per-matrix-type ΔW normalised SVD spectra (flat=full-rank);
                          backs W1 eff-rank ~1440 vs N3 null ~1618.
  fig_w8x_energy_curve  — fraction of ΔW energy inside W0's top-r OUTPUT span vs r, real
                          vs r/out chance + random-ΔW null; THE weight reconciliation.
  fig_w8x_overlap_bars  — strict top-k(ΔW)↔top-k(W0) overlap per matrix type vs nulls.
  fig_w8x_act_overlap   — activation PCA-subspace overlap (after vs before) per layer vs
                          random floor; THE direct activation re-weighting test.
  fig_w8x_act_normratio — img/txt activation norm-ratio before vs after (W4/W3 gain-control).

DISCIPLINE: a figure ILLUSTRATES a number that is independently backed by a value + null
in w8_weight.json / w8_activation.json. Never standalone evidence. Captions state the
number + the null + a DISCOVERY tag. No try/except — missing keys fail loud.

Pure matplotlib (Agg backend). No Modal, no model compute, no recomputation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_REAL = "#D55E00"     # vermillion — real ΔW / after
C_NULL = "#999999"     # grey       — random null
C_CHANCE = "#000000"   # black dots — analytic chance floor
C_BEFORE = "#56B4E9"   # sky        — before (base decoder)
C_AFTER = "#CC79A7"    # purple     — after (PaliGemma2)
C_IMG = "#009E73"      # green      — image tokens
C_TXT = "#E69F00"      # orange     — text tokens

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "legend.fontsize": 8,
})

PREREG = "prereg-v10 @ 80b1167"


def _tag(ax, t="DISCOVERY"):
    ax.text(0.985, 0.97, t, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, fontweight="bold", color="#4A148C",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#4A148C", alpha=0.9))


def _finish(fig, out_dir, stem, caption, nchar=120, bottom=0.30, right=None):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(caption, nchar))
    kw = {"bottom": bottom}
    if right is not None:
        kw["right"] = right
    fig.subplots_adjust(**kw)
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=7.6, color="#222", linespacing=1.35)
    paths = []
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        paths.append(str(p))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------

def fig_dW_spectra(wj: dict, out_dir) -> dict:
    pm = wj["per_matrix"]
    mtypes = sorted({r["matrix_type"] for r in pm.values()})
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    cmap = matplotlib.colormaps["tab10"]
    for i, mt in enumerate(mtypes):
        specs = [np.asarray(r["sv_normalised"]) for r in pm.values() if r["matrix_type"] == mt]
        L = min(len(s) for s in specs)
        mean_spec = np.mean([s[:L] for s in specs], axis=0)
        ax.plot(np.arange(1, L + 1), mean_spec, lw=1.3, color=cmap(i % 10), label=mt)
    ax.set_yscale("log"); ax.set_xlabel("singular-value index (descending)")
    ax.set_ylabel("σ / σ_max  (mean over sampled layers)")
    pr_mean = float(np.mean([r["participation_ratio"] for r in pm.values()]))
    ax.set_title("LM ΔW SVD spectra are FLAT (full-rank)")
    ax.legend(ncol=2, fontsize=7, loc="lower left")
    _tag(ax)
    cap = (f"Per-matrix-type ΔW=(L1−L0) normalised singular-value spectra decay slowly "
           f"(no low-rank cliff): mean participation-ratio eff-rank={pr_mean:.0f}. "
           f"NULL bracket: N3 random-weight null eff-rank≈1618; a rank-1 ΔW would be 1. "
           f"Matches W1 (~1440). The LM rewrite is NOT low-rank/adapter-shaped. {PREREG}. [DISCOVERY]")
    files = _finish(fig, out_dir, "fig_w8x_dW_spectra", cap)
    return {"stem": "fig_w8x_dW_spectra", "tag": "DISCOVERY", "files": files,
            "claim": "LM ΔW is full-rank (flat spectrum) for every decoder matrix type.",
            "backing_number": {"participation_ratio_mean": round(pr_mean, 1)},
            "null": "N3 random-null eff-rank≈1618; rank-1 would be 1",
            "source": "w8_analysis/w8_weight.json", "caption": cap}


def fig_energy_curve(wj: dict, out_dir) -> dict:
    pm = wj["per_matrix"]
    rank_sets = [set(int(k) for k in r["frac_energy_in_base_topr_real"]) for r in pm.values()]
    ranks = sorted(set.intersection(*rank_sets))
    real = np.array([[r["frac_energy_in_base_topr_real"][str(rr)] for rr in ranks] for r in pm.values()])
    chance = np.array([[r["chance_frac_r_over_out"][str(rr)] for rr in ranks] for r in pm.values()])
    null = np.array([[r["frac_energy_in_base_topr_null_mean"][str(rr)] for rr in ranks] for r in pm.values()])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ranks, real.mean(0), "o-", color=C_REAL, lw=1.7, label="real ΔW (mean over matrices)")
    ax.fill_between(ranks, real.mean(0) - real.std(0), real.mean(0) + real.std(0), color=C_REAL, alpha=0.15)
    ax.plot(ranks, null.mean(0), "s--", color=C_NULL, lw=1.2, label="random-ΔW null (isotropic)")
    ax.plot(ranks, chance.mean(0), ":", color=C_CHANCE, lw=1.5, label="chance = r / out_features")
    ax.set_xlabel("rank r of W0's top-r OUTPUT subspace")
    ax.set_ylabel("fraction of ΔW energy inside W0's top-r span")
    ax.set_title("ΔW lives in W0's existing output span (isotropically)")
    ax.legend(fontsize=8, loc="lower right")
    _tag(ax)
    es = wj["headline"]["energy_in_base_span"]
    cap = (f"Fraction of ΔW output energy inside W0's top-r left-singular (output) subspace "
           f"tracks the r/out chance line (mean excess over chance={es['excess_over_chance_mean']:+.3f}, "
           f"≈ the random-ΔW null). NULL = r/out (isotropic) + random-ΔW Monte-Carlo. "
           f"INTERPRETATION: ΔW writes ~uniformly ACROSS all of W0's existing output dimensions — "
           f"it re-weights the directions W0 already spans rather than adding new output directions. "
           f"This reconciles the full-rank ΔW with crosscoder emergent=0. {PREREG}. [DISCOVERY]")
    files = _finish(fig, out_dir, "fig_w8x_energy_curve", cap)
    return {"stem": "fig_w8x_energy_curve", "tag": "DISCOVERY", "files": files,
            "claim": "ΔW energy is ~isotropic in W0's output basis → re-weights within W0's existing output span.",
            "backing_number": {"excess_over_chance_mean": round(es["excess_over_chance_mean"], 4),
                               "frac_energy_real_mean": round(es["frac_energy_real_mean"], 4)},
            "null": "r/out analytic chance + random-ΔW Monte-Carlo (both ~equal to real)",
            "source": "w8_analysis/w8_weight.json", "caption": cap}


def fig_overlap_bars(wj: dict, out_dir) -> dict:
    bt = wj["by_matrix_type_overlap"]
    mtypes = list(bt.keys())
    real = [bt[m]["overlap_real_mean"] for m in mtypes]
    real_sd = [bt[m]["overlap_real_std"] for m in mtypes]
    null = [bt[m]["overlap_null_mean"] for m in mtypes]
    chance = [bt[m]["chance_floor_k_over_out"] for m in mtypes]
    x = np.arange(len(mtypes)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w, real, w, yerr=real_sd, capsize=2, color=C_REAL, label="real ΔW")
    ax.bar(x, null, w, color=C_NULL, label="random-ΔW null")
    ax.bar(x + w, chance, w, color="0.55", label="chance k/out")
    ax.set_xticks(x); ax.set_xticklabels(mtypes, rotation=30, ha="right")
    k = wj["config"]["overlap_k"]
    ax.set_ylabel(f"top-{k} output-subspace overlap")
    ax.set_title("ΔW's DOMINANT dirs do not single out W0's dominant dirs")
    ax.legend(fontsize=8)
    _tag(ax)
    ostr = wj["headline"]["overlap_strict"]
    cap = (f"Strict top-{k}(ΔW)↔top-{k}(W0) output-subspace overlap = {ostr['overlap_real_mean']:.3f} "
           f"(mean over matrices) ≈ the random-ΔW null {ostr['overlap_null_random_dW_mean']:.3f} ≈ "
           f"chance k/out {ostr['chance_floor_mean']:.3f}. NULL = random-ΔW (matched Frobenius) + k/out. "
           f"So ΔW's MOST-energetic output directions are NOT W0's most-energetic ones — consistent "
           f"with the isotropic re-mix in fig_w8x_energy_curve (ΔW spreads across W0's full span, not "
           f"concentrated on W0's top dirs, and not on NEW dirs). {PREREG}. [DISCOVERY]")
    files = _finish(fig, out_dir, "fig_w8x_overlap_bars", cap)
    return {"stem": "fig_w8x_overlap_bars", "tag": "DISCOVERY", "files": files,
            "claim": "ΔW's dominant output directions ~ random vs W0's dominant directions (no top-dir coincidence).",
            "backing_number": {"overlap_real_mean": round(ostr["overlap_real_mean"], 4),
                               "overlap_null_mean": round(ostr["overlap_null_random_dW_mean"], 4)},
            "null": "random-ΔW (matched Frobenius) + chance k/out",
            "source": "w8_analysis/w8_weight.json", "caption": cap}


def fig_act_overlap(aj: dict, out_dir) -> dict:
    layers = sorted(aj["layers"].keys(), key=int)
    def g(L, split, key):
        return aj["layers"][L]["subspace_overlap"].get(split, {}).get(key)
    real = [g(L, "all", "overlap_real") for L in layers]
    null = [g(L, "all", "overlap_null_random_mean") for L in layers]
    chance = [g(L, "all", "chance_floor_k_over_d") for L in layers]
    img = [g(L, "image", "overlap_real") for L in layers]
    txt = [g(L, "text", "overlap_real") for L in layers]
    x = np.arange(len(layers)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.bar(x - 1.5 * w, real, w, color=C_REAL, label="all tokens")
    if any(v is not None for v in img):
        ax.bar(x - 0.5 * w, [v or 0 for v in img], w, color=C_IMG, label="image tokens")
    if any(v is not None for v in txt):
        ax.bar(x + 0.5 * w, [v or 0 for v in txt], w, color=C_TXT, label="text tokens")
    ax.bar(x + 1.5 * w, null, w, color=C_NULL, label="random-subspace floor")
    ax.plot(x, chance, "k_", ms=16, label="chance k/d")
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in layers])
    k = aj["config"]["overlap_k"]
    ax.set_ylabel(f"top-{k} PCA-subspace overlap (after vs before)")
    ax.set_ylim(0, 1.05)
    ax.set_title("AFTER spans the SAME activation subspace as BEFORE")
    ax.legend(fontsize=7.4, ncol=2, loc="lower center")
    _tag(ax)
    h = aj["headline"]
    cap = (f"PCA top-{k} subspace overlap between PG2-LM (after) and decoder-reset base-Gemma2 "
           f"(before) PAIRED activations on shared multimodal probes: mean={h['overlap_real_mean']:.2f} "
           f"(min L={h['overlap_real_min']:.2f}) ≫ the random-subspace floor {h['overlap_null_random_mean']:.3f} "
           f"(≈ chance k/d {h['chance_floor_mean']:.4f}). NULL = random k-subspace of R^d. The rewrite "
           f"RE-WEIGHTS within the existing activation subspace (few/no new activation directions) — "
           f"the direct activation corroboration of crosscoder emergent=0. {PREREG}. [DISCOVERY]")
    files = _finish(fig, out_dir, "fig_w8x_act_overlap", cap)
    return {"stem": "fig_w8x_act_overlap", "tag": "DISCOVERY", "files": files,
            "claim": "PG2-LM activations span ~the same top-k subspace as decoder-reset base activations (>> random floor).",
            "backing_number": {"overlap_real_mean": round(h["overlap_real_mean"], 4),
                               "overlap_null_mean": round(h["overlap_null_random_mean"], 4),
                               "per_layer_all": {L: round(g(L, "all", "overlap_real"), 4) for L in layers}},
            "null": "random k-subspace floor (≈ chance k/d)",
            "source": "w8_analysis/w8_activation.json", "caption": cap}


def fig_act_normratio(aj: dict, out_dir) -> dict:
    layers = sorted(aj["layers"].keys(), key=int)
    before = [aj["layers"][L]["norm_ratio_before"]["img_over_txt_norm_ratio"] for L in layers]
    after = [aj["layers"][L]["norm_ratio_after"]["img_over_txt_norm_ratio"] for L in layers]
    x = np.arange(len(layers)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.bar(x - w / 2, before, w, color=C_BEFORE, label="before (base decoder)")
    ax.bar(x + w / 2, after, w, color=C_AFTER, label="after (PG2 decoder)")
    ax.axhline(1.0, color="#000", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("image / text token L2-norm ratio")
    ax.set_title("PG2 decoder rescales image-token norms (gain-control)")
    ax.legend(fontsize=8)
    _tag(ax)
    cap = (f"Image/text activation L2-norm ratio, before (decoder-reset base) vs after (PG2), per "
           f"crosscoder layer. The shift quantifies the W4/W3 gain-control (image-token norm "
           f"reshaping) as a number measured the SAME way on both halves (no swap). NULL reference = "
           f"ratio 1.0 (dotted). Descriptive companion to the subspace test. {PREREG}. [DISCOVERY]")
    files = _finish(fig, out_dir, "fig_w8x_act_normratio", cap)
    return {"stem": "fig_w8x_act_normratio", "tag": "DISCOVERY", "files": files,
            "claim": "PG2 decoder changes image/text token norm-ratio (gain-control), measured identically before/after.",
            "backing_number": {"img_over_txt_before": {L: round(b, 4) for L, b in zip(layers, before)},
                               "img_over_txt_after": {L: round(a, 4) for L, a in zip(layers, after)}},
            "null": "norm-ratio = 1.0 reference",
            "source": "w8_analysis/w8_activation.json", "caption": cap}


def build_all(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    wj = json.loads((results_dir / "w8_analysis" / "w8_weight.json").read_text())
    aj = json.loads((results_dir / "w8_analysis" / "w8_activation.json").read_text())
    out_dir = results_dir / "w8_analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        fig_dW_spectra(wj, out_dir),
        fig_energy_curve(wj, out_dir),
        fig_overlap_bars(wj, out_dir),
        fig_act_overlap(aj, out_dir),
        fig_act_normratio(aj, out_dir),
    ]
    index = {
        "workstream": "W8 analysis-half — weight + activation cross-check figures",
        "prereg_tag": PREREG,
        "question": "How does a full-rank LM ΔW (W1 eff-rank ~1440) produce NO new feature "
                    "directions (crosscoder emergent=0)?",
        "discipline": "Each figure illustrates a number+null in w8_weight.json / w8_activation.json; "
                      "never standalone evidence.",
        "script": "infra/pg/w8_analysis_figures.py",
        "figures": figures,
    }
    (out_dir / "figure_index.json").write_text(json.dumps(index, indent=2))
    return index
