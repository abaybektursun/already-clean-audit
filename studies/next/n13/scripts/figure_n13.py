"""N13 figure (studies/next/07_PICKS/N13_SPEC.md §10.3): every plotted value from verdict_n13.json, the rndT m_k labels from the S4 arm
records. Laptop only.

    python -m studies.next.n13.scripts.figure_n13 --verdict verdict_n13.json --arm-records studies/next/n13/s4 \
        --out-dir studies/next/n13/figures [--title "..."] [--footer "..."]

MATH-500 accuracy (%) against k (percent of MLP neurons, log scale), then "all MLP" (mlp) and "attention + norms" (attn) ticks. One line
per arm type (top, comp, rndE, rndT, dare), each point with its Wilson 95% bar; damaged arms drawn as x; base (B) and rl (L) as
horizontal lines with shaded Wilson bands, B' and L' thin, H dotted. Palette: the dataviz reference categorical slots 1-5 (validated,
light surface); every series also has its own marker and a direct label, and the verdict table is the table view.
Every plotted value comes from verdict_n13.json. The verdict does not carry m_k (the neuron count of each rndT_k set), so the label
under each rndT point is read from --arm-records: the S4 directory holding <launch>/rndT_<k>/arm_record.json, field
build.n_neurons_in_set; exactly one record per k (the second draw rndT_k1_d2 does not match). render() without arm_records draws no
m_k labels (the offline dry run has no arm records).
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

KX = {"k0.1": 0.1, "k0.3": 0.3, "k1": 1.0, "k3": 3.0}
SERIES = [("top", "#2a78d6", "o"), ("comp", "#eb6834", "s"), ("rndE", "#1baf7a", "^"), ("rndT", "#eda100", "D"), ("dare", "#e87ba4", "v")]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def rndT_m(arm_records):
    m = {}
    for k in KX:
        paths = sorted(pathlib.Path(arm_records).glob(f"*/rndT_{k}/arm_record.json"))
        assert len(paths) == 1, f"rndT_{k}: expected one arm_record.json under {arm_records}, found {paths}"
        R = json.loads(paths[0].read_text())
        assert R["arm"] == f"rndT_{k}", paths[0]
        m[k] = R["build"]["n_neurons_in_set"]
    return m


def render(verdict_path, out_dir, title=None, footer="", arm_records=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = json.loads(pathlib.Path(verdict_path).read_text())
    assert V["verdict"] == "COMPLETE", f"no figure for a verdict of {V['verdict']}"
    arms, T = V["arms"], V["thresholds"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs_extra = {"mlp": 10.0, "attn": 30.0}
    for nm, ls, lw in (("B", "-", 1.5), ("L", "-", 1.5), ("B_prime", ":", 0.8), ("L_prime", ":", 0.8)):
        ax.axhline(T[nm], color=MUTED, ls=ls, lw=lw, zorder=1)
    for arm in ("base", "rl"):
        w = arms[arm]["wilson"]
        ax.axhspan(w[0], w[1], color=MUTED, alpha=0.10, lw=0, zorder=0)
    ax.axhline(T["H"], color=MUTED, ls=(0, (1, 2)), lw=1.0, zorder=1)
    for nm, txt in (("H", "halfway (frozen rule)"), ("B", "base (B)"), ("L", "rl (L)")):
        ax.text(0.075, T[nm] + 1.8, txt, color=MUTED, fontsize=8, ha="left")
    for name, color, marker in SERIES:
        pts = [(KX[k], arms[f"{name}_{k}"]) for k in KX if arms.get(f"{name}_{k}", {}).get("acc") is not None]
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1]["acc"] for p in pts]
        ax.plot(x, y, color=color, lw=2, zorder=2)
        for xi, r in pts:
            lo, hi = r["wilson"]
            ax.vlines(xi, lo, hi, colors=color, linewidths=1.2, linestyles="solid", zorder=3)
            ax.plot([xi], [r["acc"]], marker="x" if r["label"] == "DAMAGED" else marker, ms=8, color=color, mec=SURFACE if r["label"] != "DAMAGED" else color,
                    mew=1.5, ls="none", zorder=4)
        ax.annotate(name, (x[-1], y[-1]), xytext=(8, {"top": 10, "rndT": -10}.get(name, 0)), textcoords="offset points", color=INK, fontsize=8, va="center")
    if arm_records is not None:
        y_m = min([arms["base"]["wilson"][0]] + [arms[f"{n}_{k}"]["wilson"][0] for n in ("top", "rndE", "rndT") for k in KX])
        for k, m in rndT_m(arm_records).items():
            ax.annotate(f"rndT\nm = {m:,}", (KX[k], y_m), xytext=(0, -6), textcoords="offset points", color=INK, fontsize=7, ha="center",
                        va="top")
    for arm, xv in xs_extra.items():
        r = arms.get(arm, {})
        if r.get("acc") is None:
            continue
        lo, hi = r["wilson"]
        ax.vlines(xv, lo, hi, colors=INK, linewidths=1.2, linestyles="solid", zorder=3)
        ax.plot([xv], [r["acc"]], marker="x" if r["label"] == "DAMAGED" else "o", ms=8, color=INK, ls="none", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "all MLP", "attention\n+ norms"])
    ax.set_xlim(0.07, 45)
    ax.set_ylim(0, 100)
    ax.set_xlabel("k, percent of MLP neurons (log scale)", color=INK)
    ax.set_ylabel("MATH-500 accuracy (%)", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6, zorder=0)
    handles = [plt.Line2D([0], [0], color=c, marker=m, lw=2, mec=SURFACE, label=n) for n, c, m in SERIES]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    ax.set_title(title or "Title: the finding, one sentence, filled at closeout", color=INK, fontsize=10, loc="left")
    cap = (f"n = 500, one sample per problem, temperature 0.6, top-p 0.95, cap C = {T['C']}, seed set A, grader math-verify 0.9.0. "
           f"Verdict {V['verdict']}; sha256(verdict) {hashlib.sha256(pathlib.Path(verdict_path).read_bytes()).hexdigest()[:16]}. {footer}")
    fig.text(0.01, 0.01, cap, fontsize=6.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_n13.png", dpi=150, facecolor=SURFACE)
    fig.savefig(out / "fig_n13.pdf", facecolor=SURFACE)
    plt.close(fig)
    return [str(out / "fig_n13.png"), str(out / "fig_n13.pdf")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--arm-records", required=True, help="S4 directory with <launch>/rndT_<k>/arm_record.json (m_k labels only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--footer", default="")
    a = ap.parse_args()
    print(render(a.verdict, a.out_dir, a.title, a.footer, a.arm_records))


if __name__ == "__main__":
    main()
