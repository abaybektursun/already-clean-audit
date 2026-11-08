#!/usr/bin/env python3
"""AMUNDSEN figures — one per SWEPT parameter, null band overlaid, provenance footer.

House rule: every swept parameter gets its full-sweep figure, not a summary
number. Here that is AM-2's lr sweep and AM-3's dose ladder; AM-1 gets the
paired per-window contrast (the object the verdict is computed on) and AM-5 the
trajectory the crossing rule reads.

Every figure prints, in its footer, the artifact it was drawn from and the git
SHA of the code that drew it — so a figure can always be traced to its rows.

  python3 figures_amundsen.py am2 --dir <collected dir> --out figures/amundsen
"""
import argparse
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def _sha():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                       text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    return r.stdout.strip() or "unknown"


def _footer(fig, source):
    fig.subplots_adjust(bottom=0.18 if len(fig.axes) <= 3 else 0.12)
    fig.text(0.01, 0.005, f"source: {source}   ·   code: figures_amundsen.py @ {_sha()}",
             fontsize=6, color="#555555")


def _save(fig, out, name, source):
    os.makedirs(out, exist_ok=True)
    _footer(fig, source)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=160)
    print("wrote", p)
    plt.close(fig)


def am1(D, out):
    w = json.load(open(f"{D}/v1abl_window_nll.json"))["per_window"]
    ppl = lambda a: np.exp(np.asarray(w[a]["nll"]) / np.asarray(w[a]["ntok"]))
    base = ppl("none")
    draws = lambda p: sorted(a for a in w if a.startswith(p + "_s"))
    series = {"core32": ppl("core32") - base,
              "energymatched32 (mean of draws)":
                  np.mean([ppl(a) - base for a in draws("energymatched32")], axis=0),
              "magmatched32 (planted, mean)":
                  np.mean([ppl(a) - base for a in draws("magmatched32")], axis=0),
              "haar32 (mean)":
                  np.mean([ppl(f"haar32_s{i}") - base for i in range(3)], axis=0)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[2, 1])
    x = np.arange(base.size)
    for k, v in series.items():
        ax[0].plot(x, v, lw=0.9, alpha=0.85, label=k)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("window index (frozen corpus order)")
    ax[0].set_ylabel(r"$\Delta$ wikitext ppl vs none")
    ax[0].set_title("AM-1 · per-window paired contrast, Qwen2.5-14B q_proj_in, k=32")
    ax[0].legend(fontsize=7)
    names, means, los, his = [], [], [], []
    rng = np.random.default_rng(1337)
    for k, v in series.items():
        idx = rng.integers(0, v.size, (2000, v.size))
        b = v[idx].mean(axis=1)
        names.append(k.split(" ")[0]); means.append(v.mean())
        los.append(np.percentile(b, 2.5)); his.append(np.percentile(b, 97.5))
    y = np.arange(len(names))
    ax[1].errorbar(means, y, xerr=[np.array(means) - los, np.array(his) - means],
                   fmt="o", capsize=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_xlabel(r"mean $\Delta$ ppl (95% paired bootstrap)")
    ax[1].set_title("AM-1 · arm means")
    _save(fig, out, "am1_per_window_contrast.png", f"{D}/v1abl_window_nll.json")


def am2(D, out):
    cells = json.load(open(f"{D}/am2_cells.json"))
    horizons = sorted({c["steps"] for c in cells})
    fig, axes = plt.subplots(2, len(horizons), figsize=(5.4 * len(horizons), 8.0))
    fig.subplots_adjust(hspace=0.42)
    axes = np.atleast_2d(axes)
    for col, steps in enumerate(horizons):
        for row, (key, label, logy) in enumerate(
                [("ppl", "wikitext ppl (blank-image)", True),
                 ("task_loss", "held-out training-task loss", False)]):
            ax = axes[row][col]
            for gauge, color in (("unrotated", "#1f77b4"), ("hadamard", "#d62728")):
                pts = [c for c in cells if c["steps"] == steps and c["gauge"] == gauge
                       and not c["diverged"]]
                lrs = sorted({c["lr"] for c in pts})
                m = [np.mean([c[key] for c in pts if c["lr"] == lr]) for lr in lrs]
                ax.plot(lrs, m, "o-", color=color, label=gauge)
                ax.scatter([c["lr"] for c in pts], [c[key] for c in pts], s=12,
                           color=color, alpha=0.45)
                best = lrs[int(np.argmin(m))]
                ax.scatter([best], [min(m)], s=140, facecolors="none", edgecolors=color, lw=1.6)
            for c in [c for c in cells if c["steps"] == steps and c["diverged"]]:
                ax.scatter([c["lr"]], [ax.get_ylim()[1]], marker="x", color="k")
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("learning rate"); ax.set_ylabel(label)
            ax.set_title(f"AM-2 · {steps} steps (rings = each side's best)", fontsize=10)
            ax.legend(fontsize=8)
    _save(fig, out, "am2_lr_sweep.png", f"{D}/am2_cells.json")


def am3(D, out):
    rows = [json.loads(l) for l in open(f"{D}/softread_rows.jsonl") if l.strip()]
    base = [r for r in rows if r["arm"] == "none"][0]["ppl"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("matched", "nominal")):
        def ser(arm, md):
            rs = sorted([r for r in rows if r["arm"] == arm and r["mode"] == md],
                        key=lambda r: r["s"])
            return [r["s"] for r in rs], [r["ppl"] - base for r in rs]
        s, d = ser("core32", "nominal")
        tw = {t: ser(t, mode)[1] for t in ("energymatched32", "haar32")}
        lo = [min(tw[t][i] for t in tw) for i in range(len(s))]
        hi = [max(tw[t][i] for t in tw) for i in range(len(s))]
        ax.fill_between(s, lo, hi, color="#999999", alpha=0.35,
                        label=f"twin band ({mode} dose)")
        ax.plot(s, d, "o-", color="#d62728", label="core32")
        for t in tw:
            ax.plot(s, tw[t], ".--", lw=0.8, alpha=0.7, label=t)
        sp, dp = ser("topenergy32", "nominal")
        ax.plot(sp, dp, "s:", color="#2ca02c", alpha=0.8, label="topenergy32 (pos. control)")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("dose s (effective-gain multiplier on the subspace)")
        ax.set_ylabel(r"$\Delta$ wikitext ppl vs baseline")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_title(f"AM-3 · dose ladder, gemma-2b ({mode} twins)")
        ax.legend(fontsize=7)
    _save(fig, out, "am3_dose_ladder.png", f"{D}/softread_rows.jsonl")


def am4(D, out):
    meta = json.load(open(f"{D}/am4_meta.json"))
    nl, W = meta["n_layers"], meta["window"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, f, color in [("this arm (batch a)", f"{D}/am4_actstats_a.npz", "#d62728"),
                           ("this arm (batch b)", f"{D}/am4_actstats_b.npz", "#ff7f0e"),
                           ("reference family (banked)", f"{D}/reference.npz", "#1f77b4")]:
        if not os.path.exists(f):
            continue
        d = dict(np.load(f))
        for site, ls in (("gate_proj_in", "-"), ("o_proj_in", ":"), ("q_proj_in", "--")):
            key = f"L00_{site}_pos_energy"
            if key not in d:
                continue
            y = [float(d[f"L{L:02d}_{site}_pos_energy"][0]
                       / d[f"L{L:02d}_{site}_pos_energy"].mean()) for L in range(nl)]
            ax.plot(range(nl), y, ls, color=color, lw=1.0, alpha=0.85,
                    label=f"{name} · {site}")
    ax.axhline(1.0, color="k", lw=0.6, label="uniform (no concentration)")
    ax.set_yscale("log"); ax.set_xlabel("layer")
    ax.set_ylabel("first-token energy concentration (x uniform)")
    ax.set_title("AM-4 · first-token relocation by depth (verdicts compare; numbers do not cross)")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "am4_depth_profile.png", f"{D}/am4_actstats_a.npz")


def am5(D, out):
    rows = [json.loads(l) for l in open(f"{D}/v2b_rows.jsonl") if l.strip()]
    keys = [("fresh_bottom32_motion_share", "cumulative (the frozen rule's tracker)"),
            ("fresh_bottom32_motion_share_win", "windowed (horizon-matched, the fix)")]
    keys = [(k, t) for k, t in keys if k in rows[0]]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(5.2 * (len(keys) + 1), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, keys):
        for gauge, color in (("unrotated", "#1f77b4"), ("hadamard", "#d62728")):
            steps = sorted({r["step"] for r in rows if r["gauge"] == gauge})
            med = [np.median([r[key] for r in rows if r["gauge"] == gauge and r["step"] == s])
                   for s in steps]
            ax.plot(steps, med, "o-", ms=3, color=color, label=f"{gauge} (seed median)")
            for seed in sorted({r["seed"] for r in rows if r["gauge"] == gauge}):
                pts = sorted([r for r in rows if r["gauge"] == gauge and r["seed"] == seed],
                             key=lambda r: r["step"])
                ax.plot([p["step"] for p in pts], [p[key] for p in pts], lw=0.6,
                        alpha=0.35, color=color)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_ylabel("fresh bottom-32 motion share")
        ax.set_title(f"AM-5 · {title}", fontsize=10)
        ax.legend(fontsize=8)
    ax = axes[-1]
    for key, title in keys:
        steps = sorted({r["step"] for r in rows})
        ratio = [np.median([r[key] for r in rows if r["gauge"] == "hadamard" and r["step"] == s])
                 / np.median([r[key] for r in rows if r["gauge"] == "unrotated" and r["step"] == s])
                 for s in steps]
        ax.plot(steps, ratio, "o-", ms=3, label=title)
    ax.axhline(1.0, color="k", lw=0.8, label="parity")
    ax.axhline(1.25, color="#2ca02c", ls="--", lw=0.8, label="RE-FORMATION bar (1.25)")
    ax.axhline(2.0, color="#d62728", ls="--", lw=0.8, label="SCAR bar (2.0)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("hadamard / unrotated")
    ax.set_title("AM-5 · the ratio the frozen rule reads", fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, out, "am5_trajectory.png", f"{D}/v2b_rows.jsonl")


def am6(D, out):
    new = [json.loads(l) for l in open(f"{D}/v3_rows_new.jsonl") if l.strip()]
    old = [json.loads(l) for l in open(f"{D}/v3_rows_banked.jsonl") if l.strip()]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    def curves(rows, style, alpha):
        by = {}
        for r in rows:
            if r.get("layer") is None:
                by.setdefault(r.get("sample_seed", "banked"), {})[r["batch"]] = r["delivery"]
        for seed, d in sorted(by.items(), key=lambda kv: str(kv[0])):
            Bs = sorted(d)
            for bank, color in (("core32", "#d62728"), ("anti32", "#ff7f0e"),
                                ("topsigma32", "#9467bd"), ("bottomsigma32", "#8c564b")):
                ax.plot(Bs, [d[B][bank] for B in Bs], style, color=color, alpha=alpha,
                        label=f"{bank} s{seed}" if style == "o-" else None)
            for h in ("haar32_s0", "haar32_s1", "haar32_s2"):
                ax.plot(Bs, [d[B][h] for B in Bs], style, color="#999999", alpha=alpha * 0.7)
    curves(old, "s--", 0.5)
    curves(new, "o-", 0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size (noise dial: sigma^2/mu^2 falls to the right)")
    ax.set_ylabel("delivery = motion share / gradient share")
    ax.set_title("AM-6 · delivery vs batch (dashed = banked seed, solid = new seeds; grey = haar)")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "am6_batch_dial.png", f"{D}/v3_rows_new.jsonl")


def am7(D, out):
    rows = [json.loads(l) for l in open(f"{D}/am7_rows.jsonl") if l.strip()]
    bms = ["mmlu_blankimg", "mmlu_cloze"]
    fig, axes = plt.subplots(1, len(bms), figsize=(5.6 * len(bms), 4.4))
    for ax, bm in zip(np.atleast_1d(axes), bms):
        for gauge, color in (("unrotated", "#1f77b4"), ("hadamard", "#d62728")):
            pts = [r for r in rows if r["gauge"] == gauge and r["kind"] == "step"]
            steps = sorted({r["step"] for r in pts})
            m = [np.mean([r[bm]["acc"] for r in pts if r["step"] == s]) for s in steps]
            sd = [np.std([r[bm]["acc"] for r in pts if r["step"] == s], ddof=1)
                  if len([r for r in pts if r["step"] == s]) > 1 else 0.0 for s in steps]
            ax.errorbar(steps, m, yerr=sd, fmt="o-", color=color, capsize=3,
                        label=f"{gauge} (mean ± SD over seeds)")
            w = [r for r in rows if r["gauge"] == gauge and r["kind"] == "wallclock"]
            if w:
                ws = sorted({r["step"] for r in w})
                ax.errorbar(ws, [np.mean([r[bm]["acc"] for r in w if r["step"] == s]) for s in ws],
                            yerr=[np.std([r[bm]["acc"] for r in w if r["step"] == s], ddof=1)
                                  if len([r for r in w if r["step"] == s]) > 1 else 0.0
                                  for s in ws],
                            fmt="s", ms=8, mfc="none", color=color, capsize=3,
                            label=f"{gauge} @ matched wall-clock")
        base = [r for r in rows if r["kind"] == "base"]
        if base:
            ax.axhline(np.mean([r[bm]["acc"] for r in base]), color="k", ls=":", lw=1.0,
                       label="step 0 (identical in both gauges)")
        # chance, so a benchmark sitting on the floor cannot look like signal
        ax.axhline(0.25, color="#7f7f7f", ls="--", lw=1.0, label="chance (4-way)")
        lo = min(0.24, ax.get_ylim()[0])
        ax.set_ylim(lo, max(ax.get_ylim()[1], 0.26))
        ax.set_xlabel("step"); ax.set_ylabel(f"{bm} accuracy (456 items)")
        ax.set_title(f"AM-7 · {bm}", fontsize=10)
        ax.legend(fontsize=7)
    _save(fig, out, "am7_benchmarks.png", f"{D}/am7_rows.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["am1", "am2", "am3", "am4", "am5", "am6", "am7"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="figures/amundsen")
    a = ap.parse_args()
    {"am1": am1, "am2": am2, "am3": am3, "am4": am4, "am5": am5, "am6": am6,
     "am7": am7}[a.arm](
        a.dir, a.out)


if __name__ == "__main__":
    main()
