"""P3 — fig11 scaling figure: per-tower split holds at scale (3B → 10B).

Pure matplotlib renderer. Reads the already-landed result JSONs on the
`pg-results` volume VERBATIM — no GPU, no model loads, no recomputation.

INPUTS (all under /vol/results):
  a1_10b/aggregate.json    — 10B S_A / S_B + CIs + per-config means
  a1_contrast/aggregate.json — 3B VQAv2-task S_A / S_B (like-for-like source)

OUTPUT (RESULTS_DIR/w8_figures_extra):
  fig11_scaling_3b_10b.{png,pdf}
  figure_index_p3.json

DISCIPLINE:
  This figure ILLUSTRATES the pre-registered §2.5 external-validity check.
  The claim is independently backed by numbers + CIs adjudicated in the
  lineage journal. A figure is never evidence on its own. Caption states
  number + CI + null + CONFIRMATORY tag. No try/except (CLAUDE.md).
"""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette — identical to w8_figures.py (colour-blind-safe Wong 2011)
# ---------------------------------------------------------------------------
C_VISION = "#0072B2"   # blue   — vision tower (translated)
C_LM = "#D55E00"       # vermillion — LM tower (rewritten)
C_NULL = "#999999"     # grey   — floor / null reference
C_ACCENT = "#E69F00"   # orange — 10B bar highlight (second scale)
C_3B = "#56B4E9"       # sky    — 3B bars (lighter tones)
C_10B_VIS = "#0072B2"  # blue   — 10B vision bar
C_10B_LM = "#D55E00"   # vermillion — 10B LM bar
C_3B_VIS = "#56B4E9"   # sky blue — 3B vision bar (lighter)
C_3B_LM = "#F0A07A"    # light vermillion — 3B LM bar

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

PREREG = "prereg-v10 @ 80b1167"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(results_dir: Path, rel: str) -> dict:
    p = results_dir / rel
    if not p.exists():
        raise FileNotFoundError(f"[p3-fig] required input missing on volume: {p}")
    return json.loads(p.read_text())


def _sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def _tag_box(ax, tag: str):
    colour = "#1B5E20" if tag.startswith("CONFIRM") else "#4A148C"
    ax.text(0.985, 0.97, tag, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, fontweight="bold", color=colour,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour, alpha=0.9))


def _finish(fig, out_dir: Path, stem: str, caption: str, *,
            nchar: int = 130, bottom: float = 0.30, right: float | None = None) -> list[str]:
    wrapped = "\n".join(textwrap.wrap(caption, nchar))
    kw = {"bottom": bottom}
    if right is not None:
        kw["right"] = right
    fig.subplots_adjust(**kw)
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=7.6,
             color="#222222", linespacing=1.35)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        paths.append(str(p))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig 11 — the per-tower split holds at scale (3B → 10B)
# ---------------------------------------------------------------------------

def fig11_scaling_3b_10b(results_dir: Path, out_dir: Path) -> dict:
    agg10 = _load(results_dir, "a1_10b/aggregate.json")
    agg3  = _load(results_dir, "a1_contrast/aggregate.json")

    # ---- 10B numbers (from aggregate.json, all fields are canonical) ----
    S_A_10b = agg10["S_A_vision"]["S_point"]
    S_B_10b = agg10["S_B_lm"]["S_point"]
    SA_lo10 = agg10["S_A_vision"]["S_ci95_low"]
    SA_hi10 = agg10["S_A_vision"]["S_ci95_high"]
    SB_lo10 = agg10["S_B_lm"]["S_ci95_low"]
    SB_hi10 = agg10["S_B_lm"]["S_ci95_high"]
    verdict_10b = agg10["VERDICT"]

    # per-config means at 10B (CEILING / PRIMARY_A / PRIMARY_B / FLOOR)
    pc10 = agg10["per_config"]
    pc10_means = {
        "CEILING":   pc10["CEILING"]["mean_pct"],
        "PRIMARY_A": pc10["PRIMARY_A"]["mean_pct"],
        "PRIMARY_B": pc10["PRIMARY_B"]["mean_pct"],
        "FLOOR":     pc10["FLOOR"]["mean_pct"],
    }

    # ---- 3B numbers — from a1_contrast/aggregate.json (VQAv2-task, like-for-like) ----
    S_A_3b = agg3["S_A_vision"]["S_point"]
    S_B_3b = agg3["S_B_lm"]["S_point"]
    SA_lo3 = agg3["S_A_vision"]["S_ci95_low"]
    SA_hi3 = agg3["S_A_vision"]["S_ci95_high"]
    SB_lo3 = agg3["S_B_lm"]["S_ci95_low"]
    SB_hi3 = agg3["S_B_lm"]["S_ci95_high"]

    # per-config means at 3B
    pc3 = agg3["per_config"]
    pc3_means = {
        "CEILING":   pc3["CEILING"]["mean_pct"],
        "PRIMARY_A": pc3["PRIMARY_A"]["mean_pct"],
        "PRIMARY_B": pc3["PRIMARY_B"]["mean_pct"],
        "FLOOR":     pc3["FLOOR"]["mean_pct"],
    }

    # -----------------------------------------------------------------------
    # Figure: 2 panels side-by-side
    #   Left:  grouped bars for S_A / S_B at 3B vs 10B + threshold lines
    #   Right: per-config raw accuracy (4 configs × 2 scales) as grouped bars
    # -----------------------------------------------------------------------
    fig, (axS, axC) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.subplots_adjust(wspace=0.30)

    # ---- LEFT panel: S_A vs S_B at 3B / 10B ----
    groups = ["Vision ($S_A$)", "LM ($S_B$)"]
    s_3b  = [S_A_3b,  S_B_3b]
    s_10b = [S_A_10b, S_B_10b]
    err_3b_lo  = [S_A_3b - SA_lo3,  S_B_3b - SB_lo3]
    err_3b_hi  = [SA_hi3 - S_A_3b,  SB_hi3 - S_B_3b]
    err_10b_lo = [S_A_10b - SA_lo10, S_B_10b - SB_lo10]
    err_10b_hi = [SA_hi10 - S_A_10b, SB_hi10 - S_B_10b]

    err_3b  = np.clip(np.array([err_3b_lo,  err_3b_hi]),  0, None)
    err_10b = np.clip(np.array([err_10b_lo, err_10b_hi]), 0, None)

    x = np.array([0, 1])
    w = 0.32
    # Colour by tower (vision=blue tones, LM=vermillion tones); lighter for 3B
    bar_3b_cols  = [C_3B_VIS, C_3B_LM]
    bar_10b_cols = [C_10B_VIS, C_10B_LM]

    bars_3b  = axS.bar(x - w / 2, s_3b,  w, color=bar_3b_cols,
                       yerr=err_3b, capsize=4, error_kw=dict(lw=1.2, ecolor="#333"),
                       label="3B (VQAv2, a1_contrast)", zorder=3)
    bars_10b = axS.bar(x + w / 2, s_10b, w, color=bar_10b_cols,
                       yerr=err_10b, capsize=4, error_kw=dict(lw=1.2, ecolor="#333"),
                       label="10B (VQAv2, a1_10b)", zorder=3)

    # Hatch 10B bars so scale is visually coded independently of colour
    for b in bars_10b:
        b.set_hatch("//")
        b.set_edgecolor("#333")
        b.set_linewidth(0.8)

    # Threshold lines — the frozen rule the verdict is read against
    axS.axhline(0.90, ls="--", lw=1.2, color="#444",
                label="0.90 = translate threshold")
    axS.axhline(0.70, ls=":",  lw=1.2, color="#444",
                label="0.70 = repurpose threshold")
    axS.axhspan(0.70, 0.90, color="#000", alpha=0.04, zorder=0)

    # S=0 floor reference (base+base)
    axS.axhline(0.0, lw=0.7, color=C_NULL, ls="-", alpha=0.6, label="S=0 (FLOOR = base+base)")

    # numeric labels
    for xi, v3, v10 in zip(x, s_3b, s_10b):
        axS.text(xi - w / 2, v3  + 0.035, f"{v3:.2f}",  ha="center", fontsize=8.0,
                 color="#333")
        axS.text(xi + w / 2, v10 + 0.035, f"{v10:.2f}", ha="center", fontsize=8.0,
                 color="#333")

    axS.set_xticks(x)
    axS.set_xticklabels(groups, fontsize=10)
    axS.set_ylabel("transfer-necessity score  $S$  (0=FLOOR, 1=CEILING)")
    axS.set_ylim(-0.05, 1.12)
    axS.set_title("Per-tower split: 3B vs 10B (VQAv2)")
    _tag_box(axS, "CONFIRMATORY · pre-reg §2.5")

    # Manual legend (split between 3B / 10B and lines)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=C_3B_VIS,  label="3B vision $S_A$"),
        Patch(facecolor=C_3B_LM,   label="3B LM $S_B$"),
        Patch(facecolor=C_10B_VIS, hatch="//", edgecolor="#333", label="10B vision $S_A$"),
        Patch(facecolor=C_10B_LM,  hatch="//", edgecolor="#333", label="10B LM $S_B$"),
        Line2D([0], [0], ls="--", color="#444", lw=1.2, label="0.90 translate"),
        Line2D([0], [0], ls=":",  color="#444", lw=1.2, label="0.70 repurpose"),
    ]
    axS.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=7.6)

    # ---- RIGHT panel: raw per-config accuracy (%) at 3B vs 10B ----
    cfg_labels = ["CEILING\n(PG+PG)", "PRIMARY_A\n(base V+PG LM)", "PRIMARY_B\n(PG V+base LM)", "FLOOR\n(base+base)"]
    cfg_keys   = ["CEILING", "PRIMARY_A", "PRIMARY_B", "FLOOR"]
    v3_raw  = [pc3_means[k]  for k in cfg_keys]
    v10_raw = [pc10_means[k] for k in cfg_keys]

    xc = np.arange(len(cfg_keys))
    wc = 0.34
    axC.bar(xc - wc / 2, v3_raw,  wc, color=C_3B,   label="3B", alpha=0.85, zorder=3)
    axC.bar(xc + wc / 2, v10_raw, wc, color=C_ACCENT, label="10B", zorder=3,
            hatch="//", edgecolor="#333", linewidth=0.8)

    for xi, v3, v10 in zip(xc, v3_raw, v10_raw):
        axC.text(xi - wc / 2, v3  + 0.5, f"{v3:.1f}",  ha="center", fontsize=7.4, color="#333")
        axC.text(xi + wc / 2, v10 + 0.5, f"{v10:.1f}", ha="center", fontsize=7.4, color="#333")

    axC.set_xticks(xc)
    axC.set_xticklabels(cfg_labels, fontsize=7.8)
    axC.set_ylabel("VQAv2 soft-accuracy  (%)")
    axC.set_ylim(0, max(max(v3_raw), max(v10_raw)) * 1.14)
    axC.set_title("Raw per-config accuracy (3B vs 10B inset)")
    axC.legend(loc="lower right", fontsize=8.5)

    # -----------------------------------------------------------------------
    # Caption (number + CI + null + tag + honest caveat)
    # -----------------------------------------------------------------------
    cap = (
        f"P3 — scaling external-validity check (spec §2.5). The per-tower split holds at 10B: "
        f"vision S_A={S_A_10b:.3f} CI95[{SA_lo10:.3f},{SA_hi10:.3f}] (→TRANSLATED, ≥0.90 threshold) "
        f"and LM S_B={S_B_10b:.3f} CI95[{SB_lo10:.3f},{SB_hi10:.3f}] (→REWRITTEN, <0.70 threshold) "
        f"under the IDENTICAL frozen recipe (VQAv2, LoRA r16/α32/1500 steps). "
        f"3B reference (a1_contrast/aggregate.json, same task): S_A={S_A_3b:.2f} CI95[{SA_lo3:.3f},{SA_hi3:.3f}], "
        f"S_B={S_B_3b:.2f} CI95[{SB_lo3:.3f},{SB_hi3:.3f}]. "
        f"The split does NOT flip at scale. "
        f"NULL = base+base FLOOR (S=0) + frozen rule (S≥0.90→translate; S<0.70→repurpose). "
        f"Right inset: raw accuracy (%) — 10B CEILING is higher, preserving the informative gap. "
        f"3B numbers: VQAv2-task from a1_contrast (like-for-like with 10B, single-task). "
        f"Honest caveat: 10B uses the SAME frozen recipe; if critics later flag undertrained, "
        f"the verdict stands directionally but should be re-evaluated at matched budget. "
        f"{PREREG}. [CONFIRMATORY · pre-registered §2.5]"
    )
    paths = _finish(fig, out_dir, "fig11_scaling_3b_10b", cap, nchar=148, bottom=0.30, right=0.72)

    return {
        "stem": "fig11_scaling_3b_10b",
        "tag": "CONFIRMATORY",
        "files": paths,
        "claim": (
            "The per-tower split (vision TRANSLATED, LM REWRITTEN) holds at 10B under the "
            "identical frozen recipe; the split does NOT flip at scale (pre-registered §2.5 check)."
        ),
        "backing_number": {
            "S_A_10b": round(S_A_10b, 4),
            "S_B_10b": round(S_B_10b, 4),
            "S_A_3b":  round(S_A_3b,  4),
            "S_B_3b":  round(S_B_3b,  4),
            "per_config_10b_pct": {k: round(v, 3) for k, v in pc10_means.items()},
            "per_config_3b_pct":  {k: round(v, 3) for k, v in pc3_means.items()},
        },
        "ci": {
            "S_A_10b": [round(SA_lo10, 4), round(SA_hi10, 4)],
            "S_B_10b": [round(SB_lo10, 4), round(SB_hi10, 4)],
            "S_A_3b":  [round(SA_lo3,  4), round(SA_hi3,  4)],
            "S_B_3b":  [round(SB_lo3,  4), round(SB_hi3,  4)],
        },
        "null": "base+base FLOOR (S=0); frozen rule S>=0.90 translate / S<0.70 repurpose",
        "source": {
            "10b": "a1_10b/aggregate.json",
            "3b":  "a1_contrast/aggregate.json (VQAv2-task, like-for-like with 10B)",
        },
        "3b_source_choice": (
            "VQAv2-task numbers from a1_contrast/aggregate.json (NOT the battery S̄). "
            "Reason: the 10B experiment is VQAv2-only; using the VQAv2-specific 3B contrast "
            "is the like-for-like comparison. The battery S̄ is multi-task and would conflate "
            "task-selection with scale effects."
        ),
        "caption": cap,
        "verdict_10b": verdict_10b,
    }


# ---------------------------------------------------------------------------
# Driver — called by the Modal wrapper
# ---------------------------------------------------------------------------

_INPUTS = [
    "a1_10b/aggregate.json",
    "a1_contrast/aggregate.json",
]


def build(results_dir: str | Path) -> dict:
    """Render fig11 + write a per-figure index. Returns the index dict."""
    results_dir = Path(results_dir)
    out_dir = results_dir / "w8_figures_extra"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {}
    for rel in _INPUTS:
        p = results_dir / rel
        input_hashes[rel] = _sha8(p) if p.exists() else "MISSING"

    fig_entry = fig11_scaling_3b_10b(results_dir, out_dir)

    index = {
        "workstream": "P3 — scaling external-validity figure (spec §2.5)",
        "prereg_tag": PREREG,
        "discipline": (
            "Figure ILLUSTRATES a claim independently backed by numbers + CIs already "
            "adjudicated in the lineage journal; never standalone evidence. Caption states "
            "number + CI + null + CONFIRMATORY tag. 3B numbers from a1_contrast/aggregate.json "
            "(VQAv2-task, like-for-like with the 10B VQAv2-only experiment)."
        ),
        "script": "infra/pg/p3_scaling_fig.py",
        "input_hashes": input_hashes,
        "figures": [fig_entry],
    }
    (out_dir / "figure_index_p3.json").write_text(json.dumps(index, indent=2))
    return index
