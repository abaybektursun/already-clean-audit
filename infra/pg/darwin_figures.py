"""DARWIN — figure assembly (§10 `out/figures/`).

House rule (feedback_complete_sweep_visualization): a swept figure shows ALL points and ALL
component curves — the real banks AND every twin AND the CI — never a summary of them. A figure
that plots core32 alone is a defect here, for the same reason a table without its control column
is: the reader cannot see the comparison the claim rests on.

Every figure carries a provenance footer (run ids + commit sha + row count), so a figure lifted
out of the directory still resolves to the rows that produced it (§0.18).

Zero interpretation (§0.1): no title, caption or annotation in this module states a conclusion.
Titles name the statistic and the cell; that is all.

Pure-matplotlib, Agg backend, no seaborn. Heavy imports live inside functions.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

# Bank roles drive colour and z-order, so the reader can always tell the treatment from its
# controls without consulting a legend entry-by-entry.
REAL_BANKS = ("core32", "anti32")
STRUCTURED = ("topsigma32", "bottomsigma32")
HAAR = ("haar32_s0", "haar32_s1", "haar32_s2")
COMPLEMENT = ("complement32",)

STYLE = {
    "core32":        {"color": "#C2410C", "lw": 2.4, "z": 5, "marker": "o", "ms": 4},
    "anti32":        {"color": "#1D4ED8", "lw": 2.4, "z": 5, "marker": "s", "ms": 4},
    "topsigma32":    {"color": "#047857", "lw": 1.6, "z": 4, "marker": "^", "ms": 3},
    "bottomsigma32": {"color": "#7C3AED", "lw": 1.6, "z": 4, "marker": "v", "ms": 3},
    "complement32":  {"color": "#9CA3AF", "lw": 1.2, "z": 3, "marker": "d", "ms": 3},
    "haar32_s0":     {"color": "#D1D5DB", "lw": 1.0, "z": 2, "marker": None, "ms": 0},
    "haar32_s1":     {"color": "#D1D5DB", "lw": 1.0, "z": 2, "marker": None, "ms": 0},
    "haar32_s2":     {"color": "#D1D5DB", "lw": 1.0, "z": 2, "marker": None, "ms": 0},
}
FALLBACK = {"color": "#111827", "lw": 1.2, "z": 1, "marker": "x", "ms": 3}

# Series that are NOT bank ids — quantization arms, pruning schemes, census bases — still need to
# be told apart. A single fallback colour renders every one of them in the same dark line, which
# is how the first W1-N4 contrast figure came out: six curves, one colour, unreadable. Distinct
# hue AND distinct marker, because a reader may be colour-blind or printing in grey.
FALLBACK_PALETTE = (
    ("#C2410C", "o"), ("#1D4ED8", "s"), ("#047857", "^"), ("#7C3AED", "v"),
    ("#B45309", "D"), ("#0E7490", "P"), ("#BE185D", "X"), ("#4D7C0F", "*"),
    ("#334155", "h"), ("#9333EA", "8"),
)


def use_agg():
    import matplotlib
    matplotlib.use("Agg")


def provenance_footer(fig, df, extra: str = ""):
    """Stamp run ids + commit shas + row count under every figure."""
    runs = sorted(set(df["run_id"].dropna())) if "run_id" in df else []
    shas = sorted({str(s)[:10] for s in df["commit_sha"].dropna()}) if "commit_sha" in df else []
    txt = (f"n_rows={len(df)}  runs={','.join(runs)[:110]}  sha={','.join(shas)[:60]}"
           + (f"  {extra}" if extra else ""))
    fig.text(0.005, 0.004, txt, fontsize=5.2, color="#6B7280", ha="left", va="bottom")


def style_map(ids) -> dict:
    """{id: style} for one figure's whole series set.

    Bank ids keep their fixed role-based style. Everything else is assigned a palette slot by
    SORTED POSITION within this figure, which guarantees the curves are distinguishable — hashing
    each name independently does not: six arms into ten slots collide about 78% of the time, and
    the first attempt at this duly produced two identical colours. Sorted position is also
    deterministic, so a given statistic's arms keep their colours across runs.
    """
    unknown = sorted(str(i) for i in ids if i not in STYLE)
    assert len(unknown) <= len(FALLBACK_PALETTE), (
        f"{len(unknown)} non-bank series exceeds the {len(FALLBACK_PALETTE)}-colour palette; "
        f"extend FALLBACK_PALETTE rather than let two curves share a colour")
    out = {i: STYLE[i] for i in ids if i in STYLE}
    for slot, name in enumerate(unknown):
        color, marker = FALLBACK_PALETTE[slot]
        out[name] = {"color": color, "lw": 1.6, "z": 3, "marker": marker, "ms": 4}
    return out


def _style(bid):
    """Single-series style, for callers with no set context. Prefer `style_map`."""
    return STYLE.get(bid, FALLBACK)


def bank_sweep(df, *, x: str, y: str, title: str, xlabel: str, ylabel: str, out_path,
               reference: float = None, reference_label: str = None, logx: bool = False,
               extra_footer: str = ""):
    """One sweep, every bank as its own curve, CI as a shaded band where the rows carry one.

    `df` must already be filtered to ONE statistic and ONE cell family; the function does not
    decide what to plot, it plots what it is given. Rows are grouped by `bank_id`, sorted by `x`.

    Returns the set of bank_ids drawn, so the caller can assert control completeness on the
    FIGURE — a figure silently missing its twin curves is the defect this guards.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert len(df), f"bank_sweep: empty frame for {title}"
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
    smap = style_map(sorted(set(df["bank_id"].dropna())))
    drawn = set()
    for bid, g in df.groupby("bank_id", dropna=True):
        g = g.sort_values(x)
        st = smap[bid]
        xv = g[x].to_numpy(dtype=float)
        yv = g[y].to_numpy(dtype=float)
        ax.plot(xv, yv, label=bid, color=st["color"], lw=st["lw"], zorder=st["z"],
                marker=st["marker"], markersize=st["ms"])
        if {"ci_lo", "ci_hi"} <= set(g.columns) and g["ci_lo"].notna().any():
            lo = g["ci_lo"].to_numpy(dtype=float)
            hi = g["ci_hi"].to_numpy(dtype=float)
            m = np.isfinite(lo) & np.isfinite(hi)
            if m.any():
                ax.fill_between(xv[m], lo[m], hi[m], color=st["color"], alpha=0.16,
                                lw=0, zorder=st["z"] - 1)
        drawn.add(bid)
    if reference is not None:
        ax.axhline(reference, color="#111827", ls=":", lw=1.0, zorder=1,
                   label=reference_label or "reference")
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, ncol=3, framealpha=0.9, loc="best")
    provenance_footer(fig, df, extra_footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return drawn


def bank_bars(df, *, y: str, title: str, ylabel: str, out_path, group: str = "bank_id",
              reference: float = None, reference_label: str = None, extra_footer: str = ""):
    """Per-bank distribution as a bar with its own spread — for statistics with no natural sweep
    axis. Bars are ordered treatment-first so the eye lands on the comparison, not on the
    alphabet."""
    import matplotlib.pyplot as plt
    import numpy as np

    assert len(df), f"bank_bars: empty frame for {title}"
    order = [b for b in (REAL_BANKS + STRUCTURED + COMPLEMENT + HAAR)
             if b in set(df[group].dropna())]
    order += sorted(set(df[group].dropna()) - set(order))
    smap = style_map(order)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=170)
    for i, bid in enumerate(order):
        v = df[df[group] == bid][y].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        st = smap[bid]
        ax.bar(i, v.mean(), color=st["color"], alpha=0.85, zorder=3,
               yerr=(v.std(ddof=1) if v.size > 1 else 0.0), capsize=3,
               error_kw={"lw": 0.8, "ecolor": "#374151"})
        ax.scatter(np.full(v.size, i) + np.random.default_rng(0).normal(0, 0.06, v.size), v,
                   s=3, color="#111827", alpha=0.35, zorder=4)     # ALL points, per house rule
    if reference is not None:
        ax.axhline(reference, color="#111827", ls=":", lw=1.0, zorder=2,
                   label=reference_label or "reference")
        ax.legend(fontsize=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=6.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    provenance_footer(fig, df, extra_footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return set(order)


def layer_type_heatmap(df, *, value: str, title: str, out_path, cbar_label: str = "",
                       extra_footer: str = ""):
    """(type x layer) heatmap of one statistic for one bank — the coverage-at-a-glance view."""
    import matplotlib.pyplot as plt
    import numpy as np

    assert len(df), f"heatmap: empty frame for {title}"
    types = sorted(set(df["type"].dropna()))
    layers = sorted(int(x) for x in set(df["layer"].dropna()))
    M = np.full((len(types), len(layers)), np.nan)
    for _, r in df.iterrows():
        if r["type"] in types and r["layer"] == r["layer"]:
            M[types.index(r["type"]), layers.index(int(r["layer"]))] = r[value]
    fig, ax = plt.subplots(figsize=(7.6, 2.6 + 0.22 * len(types)), dpi=170)
    im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=6)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=7)
    ax.set_xlabel("layer", fontsize=8)
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(cbar_label or value, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    provenance_footer(fig, df, extra_footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return {"n_types": len(types), "n_layers": len(layers),
            "n_filled": int(np.isfinite(M).sum())}


def assert_controls_drawn(drawn, *, figure: str, treatment_is_bank: bool = True):
    """A figure that lost its control curves is a defect (§0.1, §3). Checked on the FIGURE, not
    only on the table, because the figure is what a reader actually looks at."""
    if not treatment_is_bank:
        return sorted(drawn)
    need = set(REAL_BANKS) | set(STRUCTURED) | set(HAAR) | set(COMPLEMENT)
    missing = sorted(need - set(drawn))
    assert not missing, (
        f"{figure}: drawn without control curves {missing}. §3 requires every control column in "
        f"the same view as the treatment; a Haar-only comparison is exactly what it forbids")
    return sorted(drawn)


def manifest(figures: dict, out_dir):
    """out/figures/manifest.json — what was drawn, from which rows, with which banks."""
    p = pathlib.Path(out_dir) / "figures" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(figures, indent=2, sort_keys=True, default=str))
    return str(p)
