"""W17 figure suite — weight-free from results/w17/{metrics,controls}.json.

Three panels, one claim each, null overlaid, provenance footer, Okabe–Ito,
PNG+PDF, bbox tight. Run on the control plane:

    python -m infra.pg.w17_figures
"""
from __future__ import annotations

import json
import pathlib
import subprocess

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "sky": "#56B4E9"}
ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "studies/lineage/paligemma2/figures/w17_figures"
NS = [64, 256, 1000, 4000, 16000]


def _footer(fig, m, tag):
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h"],
                         capture_output=True, text=True, check=True).stdout.strip()
    fig.text(0.01, 0.005, f"{tag} · {m['provenance']['freeze_tag']} · commit {sha} · "
             f"controls sha {m['provenance']['controls_sha256'][:12]}",
             fontsize=6, color="#555555")


def _save(fig, name, index, claim, tag):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    index[name] = {"claim": claim, "tag": tag, "files": [f"{name}.png", f"{name}.pdf"]}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = json.loads((ROOT / "results/w17/metrics.json").read_text())
    grid = m["discovery_grid_bucket"]["cells"]
    index: dict = {}

    # --- F1 headline: r=512 recovery curves + confirmation decision points ----
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    curves = {"capability (pg)": ("pg_cap_r512_n{n}_s1337_bucket", OI["blue"], "o-"),
              "random decoder (no prior)": ("random_cap_r512_n{n}_s1337_bucket", OI["orange"], "s-")}
    for label, (pat, col, st) in curves.items():
        ys = [grid[pat.format(n=n)]["S"] for n in NS]
        ax.plot(NS, ys, st, color=col, label=f"{label} — bucket, DISCOVERY")
    ax.scatter([1000, 16000], [m["cell_S"]["cap_1k"]["S_seed_mean"], m["cell_S"]["cap_16k"]["S_seed_mean"]],
               marker="*", s=220, color=OI["blue"], zorder=5,
               label="capability CONFIRMATION (7,453 items, 3 seeds)")
    ax.scatter([1000, 16000], [m["cell_S"]["rnd_1k"]["S_seed_mean"], m["cell_S"]["rnd_16k"]["S_seed_mean"]],
               marker="*", s=220, color=OI["orange"], zorder=5,
               label="random CONFIRMATION")
    for key, n, col in (("ret_16k", 16000, OI["green"]), ("shf_16k", 16000, OI["purple"]),
                        ("shf_1k", 1000, OI["purple"])):
        ax.scatter([n], [m["cell_S"][key]["S_seed_mean"]], marker="D", s=60, color=col, zorder=4)
    ax.scatter([], [], marker="D", s=60, color=OI["green"], label="retain-only (displacement floor)")
    ax.scatter([], [], marker="D", s=60, color=OI["purple"], label="shuffled-label (format floor)")
    ax.axhline(0, color=OI["grey"], ls=":")
    ax.axhline(0.5, color=OI["grey"], ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n distinct QA examples (log)")
    ax.set_ylabel("S_recover (BEFORE−FLOOR scale)")
    ax.set_title("Latent capability survives the deletion: cap@16k ΔS = +0.64 over the no-prior "
                 "ladder (p_adj 0.0008); retain/shuffled flat — content, not normalization")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(0.99, 0.03, "How to read: lines = single-seed bucket (DISCOVERY); "
            "stars/diamonds = sealed-half decision cells (3 seeds).",
            transform=ax.transAxes, fontsize=6.5, ha="right", color="#444444")
    _footer(fig, m, "CONFIRMATORY (stars) / DISCOVERY (lines)")
    _save(fig, "f1_recovery_curves", index,
          "H_savings supported via 16k; displacement+format killed", "CONFIRMATORY")
    plt.close(fig)

    # --- F2: the m=5 primary family forest plot ----
    tests = m["primary_tests"]
    bh = m["VERDICTS"]["bh"]
    order = ["sav_sup_1k", "sav_floor_1k", "sav_sup_16k", "sav_floor_16k", "disp_16k"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, t in enumerate(order):
        d = tests[t]
        sig = bh[t]["significant"]
        col = OI["blue"] if sig else OI["grey"]
        ax.errorbar([d["point"]], [i], xerr=[[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]],
                    fmt="o", color=col, capsize=4, ms=7)
        ax.text(max(d["ci95"][1], 0.02) + 0.02, i,
                f"p_adj={bh[t]['p_adj']:.4f}{' *' if sig else ''}", va="center", fontsize=8)
    ax.axvline(0, color=OI["grey"], ls="--", label="null (no difference)")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("ΔS_recover (seed-mean, confirmation half)")
    ax.set_title("Frozen m=5 family, BH q=0.05: all three 16k contrasts decisive; 1k sup ns")
    ax.legend(fontsize=7)
    _footer(fig, m, "CONFIRMATORY")
    _save(fig, "f2_primary_family_forest", index,
          "BH-governed verdicts on the sealed half", "CONFIRMATORY")
    plt.close(fig)

    # --- F3: rank × training-floor panel ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    for r, col in ((8, OI["sky"]), (64, OI["green"]), (512, OI["blue"])):
        ys = [grid[f"pg_cap_r{r}_n{n}_s1337_bucket"]["S"] for n in NS]
        ax.plot(NS, ys, "o-", color=col, label=f"r={r}")
    ax.axhline(0, color=OI["grey"], ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("n (log)"); ax.set_ylabel("bucket S_recover")
    ax.set_title("Only r=512 recovers (DISCOVERY)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    ce = m["post_critic_descriptive"]["low_rank_train_ce_16k"]
    ranks = ["r8", "r64", "r512"]
    ax.bar(range(3), [ce[r] for r in ranks], color=[OI["sky"], OI["green"], OI["blue"]])
    ax.axhline(ce["ln4_chance_floor"], color=OI["red"], ls="--",
               label="ln 4 chance-CE floor")
    ax.set_xticks(range(3), ranks)
    ax.set_ylabel("last-epoch train CE @ 16k")
    ax.set_title("r≤64 never exits the TRAINING floor —\ncapacity vs under-optimization unresolved", fontsize=9)
    ax.legend(fontsize=7)
    _footer(fig, m, "DISCOVERY / DESCRIPTIVE")
    _save(fig, "f3_rank_gate_training_floor", index,
          "Rank-resolved recovery with the training-floor caveat (critic M4)", "DISCOVERY")
    plt.close(fig)

    (FIGDIR / "fig_index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} figures → {FIGDIR}")


if __name__ == "__main__":
    main()
