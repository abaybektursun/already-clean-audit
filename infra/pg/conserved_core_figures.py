"""Figures for conserved-core (prereg §10) — weight-free, reads the result JSONs only.

    python3 infra/pg/conserved_core_figures.py \
        --t1 studies/lineage/paligemma1-pi0/results/conserved_core/conserved_core.json \
        --t2 studies/lineage/paligemma1-pi0/results/policy_subject_structure/policy_subject_structure.json \
        --out studies/lineage/paligemma1-pi0/results/conserved_core --sha <freeze-sha>

F1  probe-degradation ladder (Tier 1) — per probe (position-corr / induction-F1 / BOS-sink), the metric
    under {baseline, U_unt, ENERGY-MATCHED, random, bottom-σ} at the headline k, with the energy removed
    per arm annotated. An energy-only hit drops every probe ∝ energy and U_unt ≈ energy-matched; a named
    function shows U_unt dropping ITS probe past the energy-matched null. The naming chart.
F2  dose-response (Tier 1) — each probe's energy-controlled excess (U_unt − energy-matched) ± CI vs
    k∈{8,32,128}, with θ_X overlaid. Monotone-in-k is a §6 naming conjunct.
F3  per-subject policy map (Tier 2) — π0/π0.5 per-subject cloze−floor gap, bars coloured by the frozen
    spatial proxy, floor at 0; Spearman ρ + CI in the title (uniform vs structured).

Every panel: CI / null overlaid, a plain finding title, a "how to read" line, a provenance footer. PNG+PDF.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colourblind-safe (Wong)
C_BASE = "#000000"     # black  — baseline (no ablation)
C_UNT = "#0072B2"      # blue   — U_unt
C_EM = "#D55E00"       # vermillion — ENERGY-MATCHED null (the decisive comparator)
C_RAND = "#E69F00"     # orange — low-energy random null
C_BOTS = "#009E73"     # green  — bottom-σ null
C_SPATIAL = "#0072B2"  # blue   — spatial/embodied subject
C_VERBAL = "#E69F00"   # orange — abstract/verbal subject
PROBES = [("pos", "position (corr)"), ("indu", "induction (F1)"), ("sink", "BOS-sink (mass)")]


def _footer(ax, txt):
    ax.annotate(txt, xy=(0, 0), xytext=(0, -42), textcoords="offset points", xycoords="axes fraction",
                fontsize=6, color="#666666", va="top")


def fig_f1(t1, out, sha):
    L = str(t1["headline_layer"])
    k = t1["headline_k"]
    hk = next(r for r in t1["ablation"] if r["k"] == k)
    en = t1["energy"][f"{t1['headline_layer']},{k}"]
    e_em_mean = sum(en["e_em"]) / len(en["e_em"])
    verdict = t1["verdict"]["named"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (key, label) in zip(axes, PROBES):
        raw = hk["probes"][L][key]["raw"]
        arms = ["base", "unt", "em", "rand", "bots"]
        vals = [raw[a] for a in arms]
        colors = [C_BASE, C_UNT, C_EM, C_RAND, C_BOTS]
        ax.bar(range(5), vals, color=colors)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["baseline", "U_unt", "energy-\nmatched", "random", "bottom-σ"], fontsize=7)
        ex = hk["probes"][L][key]["excess_vs_em"]
        th = hk["probes"][L][key]["theta_em"]
        ax.set_title(f"{label}\nexcess(U_unt−em)={ex['contrast']:+.3f} [{ex['lo']:+.3f},{ex['hi']:+.3f}] θ={th:.3f}",
                     fontsize=8)
        ax.axhline(raw["base"], color="#bbbbbb", lw=0.8, ls=":")
    axes[0].set_ylabel("probe metric (higher = more decodable)")
    fig.suptitle(f"F1 · conserved-core: does U_unt carry a function past the ENERGY-MATCHED null? — "
                 f"verdict {verdict}\nU_unt removes {en['e_unt']:.2f} of activation energy; energy-matched "
                 f"null removes {e_em_mean:.2f} (matched). How to read: a function shows U_unt's bar BELOW "
                 f"the energy-matched bar for ITS probe only.", fontsize=9)
    _footer(axes[0], f"k={k} · L{t1['headline_layer']} · base_sha={t1.get('base_sha','?')[:8]} · "
                     f"freeze={sha} · CONFIRMATORY")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/F1_probe_ladder.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_f2(t1, out, sha):
    L = str(t1["headline_layer"])
    ks = [r["k"] for r in t1["ablation"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label in PROBES:
        xs, ys, los, his = [], [], [], []
        for r in t1["ablation"]:
            ex = r["probes"][L][key]["excess_vs_em"]
            xs.append(r["k"]); ys.append(ex["contrast"]); los.append(ex["contrast"] - ex["lo"]); his.append(ex["hi"] - ex["contrast"])
        col = {"pos": C_UNT, "indu": C_BOTS, "sink": C_EM}[key]
        ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=label, color=col)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (conserved dims removed)"); ax.set_ylabel("energy-controlled excess (U_unt − energy-matched)")
    ax.legend(fontsize=8)
    ax.set_title("F2 · dose-response: excess past the energy-matched null vs k\n"
                 "How to read: a confirmatory function rises monotonically with k (more conserved dims removed → more loss).",
                 fontsize=9)
    _footer(ax, f"L{t1['headline_layer']} · base_sha={t1.get('base_sha','?')[:8]} · freeze={sha} · CONFIRMATORY")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/F2_dose_response.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_f3(t2, out, sha):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, pol in zip(axes, ("pi0", "pi05")):
        r = t2["gap_vs_proxy"][pol]
        gaps = r["per_subject_gap"]
        proxy = r["proxy"]
        items = sorted(gaps.items(), key=lambda x: x[1])
        vals = [g for _, g in items]
        cols = [C_SPATIAL if proxy[s] else C_VERBAL for s, _ in items]
        ax.bar(range(len(items)), vals, color=cols)
        ax.axhline(0, color="#000000", lw=0.8)
        ax.set_xticks([]); ax.set_xlabel(f"57 MMLU subjects (sorted by gap; blue=spatial, orange=verbal)")
        ax.set_ylabel("cloze − floor (per-subject)")
        verdict = "STRUCTURED" if r["structured"] else "UNIFORM"
        ax.set_title(f"{pol}: {verdict} — ρ(gap, spatial-proxy)={r['rho']:+.3f} CI[{r['lo']:+.3f},{r['hi']:+.3f}]",
                     fontsize=9)
    fig.suptitle("F3 · per-subject deployed-policy erasure structure (Tier 2, exploratory). "
                 "How to read: bars near 0 = erased to the floor; a spatial/verbal colour gradient would mean structured.",
                 fontsize=9)
    _footer(axes[0], f"n_items={t2['n_items']} · freeze={sha} · EXPLORATORY/DESCRIPTIVE")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/F3_subject_map.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", default="")
    ap.add_argument("--t2", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sha", default="?")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.t1:
        t1 = json.loads(pathlib.Path(a.t1).read_text())
        fig_f1(t1, a.out, a.sha); fig_f2(t1, a.out, a.sha)
        print(f"[fig] F1, F2 written to {a.out}")
    if a.t2:
        t2 = json.loads(pathlib.Path(a.t2).read_text())
        fig_f3(t2, a.out, a.sha)
        print(f"[fig] F3 written to {a.out}")


if __name__ == "__main__":
    main()
