"""SAV-A figure suite — weight-free from results/sav/metrics.json (post-fold).

    python -m infra.pg.sav_figures
"""
from __future__ import annotations

import json
import pathlib
import subprocess

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "studies/lineage/paligemma1-pi0/figures/sav_figures"
LADDER = [256, 1000, 4000, 16000]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = json.loads((ROOT / "results/sav/metrics.json").read_text())
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h"],
                         capture_output=True, text=True, check=True).stdout.strip()
    index = {}

    def footer(fig, tag):
        fig.text(0.01, 0.005, f"{tag} · sav-a-lineage-savings-prereg@61dc325 · commit {sha}",
                 fontsize=6, color="#555555")

    def save(fig, name, claim, tag):
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
        index[name] = {"claim": claim, "tag": tag, "files": [f"{name}.png", f"{name}.pdf"]}

    # F1: the four ladders (seed-mean S per cell from spreads)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    cols = {"PG1": OI["blue"], "pi0": OI["red"], "pi05": OI["purple"], "random": OI["grey"]}
    for ck, col in cols.items():
        ys = [sum(m["spreads"][f"{ck}_{n}"]["per_seed_S"]) / 3 for n in LADDER]
        ax.plot(LADDER, ys, "o-", color=col, label=f"{ck} (3-seed mean, sealed half)")
    ax.axhline(0, color=OI["grey"], ls=":")
    ax.axhline(0.5, color=OI["grey"], ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n distinct QA examples (log)")
    ax.set_ylabel("S_recover (BASE−FLOOR scale)")
    s = m["savings_observed"]
    ax.set_title("Latent capability survives the VL hop and collapses at the robot hop: "
                 f"savings PG1 +{float(s['PG1']):.2f} (p_adj .002) vs π0 +{float(s['pi0']):.2f} ns / "
                 f"π0.5 +{float(s['pi05']):.2f} ns")
    ax.legend(fontsize=8)
    ax.text(0.99, 0.02, "Verdict: H_collapsed (consumed-to-floor qualifier). "
            "Scope: 16k-budget radius, frozen recipe.", transform=ax.transAxes,
            fontsize=6.5, ha="right", color="#444444")
    footer(fig, "CONFIRMATORY")
    save(fig, "f1_lineage_ladders", "the three-way verdict", "CONFIRMATORY")
    plt.close(fig)

    # F2: savings forest (m=6)
    tests = m["primary_tests"]
    bh = m["VERDICTS"]["bh"]
    order = ["T1_sav_PG1", "T2_sav_pi0", "T3_sav_pi05", "T4_drop1", "T5_drop2", "T6_droptotal"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, t in enumerate(order):
        d = tests[t]
        col = OI["blue"] if bh[t]["significant"] else OI["grey"]
        ax.errorbar([d["point"]], [i], xerr=[[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]],
                    fmt="o", color=col, capsize=4)
        ax.text(d["ci95"][1] + 0.02, i, f"p_adj={bh[t]['p_adj']:.3f}", va="center", fontsize=8)
    ax.axvline(0, color=OI["grey"], ls="--")
    dq = m["delta_auc_ci_legs"]
    ax.axvspan(-dq, dq, color=OI["orange"], alpha=0.08)
    ax.text(dq, -0.45, f"±δ_AUC={dq:.2f} (inflated; CI legs fit ±0.11)", fontsize=6.5, color="#7a5200")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("savings / drop (curve-AUC units)")
    ax.set_title("Frozen m=6 family: levels (T1 only) and drops (T4/T6) — BH q=0.05")
    footer(fig, "CONFIRMATORY")
    save(fig, "f2_family_forest", "m=6 tests with the δ band shown honestly", "CONFIRMATORY")
    plt.close(fig)

    # F3: the fit-check panel (train CE by arm)
    ce = m["post_critic_descriptive"]["train_ce_last_epoch_by_arm"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for ck, col in cols.items():
        ys = [sum(ce[ck][str(n)]) / len(ce[ck][str(n)]) for n in LADDER]
        ax.plot(LADDER, ys, "o-", color=col, label=ck)
    ax.axhline(m["post_critic_descriptive"]["ln4"], color=OI["red"], ls="--", lw=0.8,
               label="ln 4 (uniform letters)")
    ax.set_xscale("log")
    ax.set_xlabel("n (log)"); ax.set_ylabel("last-epoch train CE (seed mean)")
    ax.set_title("π0 TRAINS like the random decoder (CE rises with n; never fits) — "
                 "PG1 fits to 0.11 under identical conditions", fontsize=9)
    ax.legend(fontsize=8)
    footer(fig, "DESCRIPTIVE (critic M2 check)")
    save(fig, "f3_train_ce_fit_check", "no-prior training dynamics on the π side", "DESCRIPTIVE")
    plt.close(fig)

    (FIGDIR / "fig_index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} figures → {FIGDIR}")


if __name__ == "__main__":
    main()
