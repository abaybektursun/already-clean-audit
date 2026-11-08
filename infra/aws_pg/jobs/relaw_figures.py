"""RELAW figure suite (§10) — weight-free, from the metric JSONs only (NO model, NO GPU).

Each figure illustrates a claim independently backed by a number + CI + null:
  F1 (headline): G vs R_prin scatter across the SAV family, the R_dist baseline + the held-out
       member starred; subtitle = ΔLOO-R²(R_prin|R_dist) + perm-p + the law verdict.
  F3 (causal):  Arm-3 RF under standard / insulated(off) / anti(into), CI bars (the lever).
  F4 (partials): ΔLOO-R² partials (R_prin|R_dist, R_prin|R_ovl, R_ovl|R_prin) with perm-p.
Reads relaw_metrics.json (F1/F4) and relaw_insulate_metrics.json (F3); draws what is present.

    python -m infra.aws_pg.jobs.relaw_figures --out <dir> \
        --lawfit <relaw_metrics.json|s3> [--insulate <relaw_insulate_metrics.json|s3>] --sha <commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

OK_BLUE, OK_ORANGE, OK_GREEN, OK_GREY = "#0072B2", "#D55E00", "#009E73", "#777777"


def _load(path, out):
    if not path:
        return None
    if path.startswith("s3://"):
        loc = out / pathlib.Path(path).name
        subprocess.run(["aws", "s3", "cp", path, str(loc), "--region", "us-east-1"], check=True)
        path = str(loc)
    return json.loads(pathlib.Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lawfit", default="")
    ap.add_argument("--insulate", default="")
    ap.add_argument("--reads", default="", help="reads_sav.json for the scatter points")
    ap.add_argument("--sha", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    law = _load(a.lawfit, out)
    reads = _load(a.reads, out)
    ins = _load(a.insulate, out)
    foot = f"run · SHA {a.sha or '?'} · CONFIRMATORY (held-out + LOO)"
    made = []

    # ---- F1: G vs R_prin headline scatter ----
    if law and reads:
        r_star = law["r_star"]
        units = sorted(reads["units"], key=lambda u: u["unit"])
        G = np.array([u["G"] for u in units])
        Rp = np.array([u["R_prin"][str(r_star)] for u in units])
        held = law["held_out"]
        f = law["fit"]["R_prin_over_dist"]
        fig, ax = plt.subplots(figsize=(7, 5))
        for u, g, rp in zip(units, G, Rp):
            star = u["unit"] == held
            ax.scatter(rp, g, s=120 if star else 55, marker="*" if star else "o",
                       color=OK_ORANGE if star else OK_BLUE, zorder=3,
                       edgecolor="k" if star else "none",
                       label="held-out (sealed)" if star else None)
        # LOO/OLS guide line (illustration; the CLAIM is the ΔLOO-R² number below, not the line).
        b1, b0 = np.polyfit(Rp, G, 1)
        xs = np.linspace(Rp.min(), Rp.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=OK_GREEN, lw=1.5, ls="--", label="OLS guide")
        ax.set_xlabel(f"R_prin = principal-subspace survival (top-{r_star})")
        ax.set_ylabel("G = re-learnability S(16k)")
        ax.set_title(f"RELAW F1 — does weight-principal survival predict re-learnability?\n"
                     f"ΔLOO-R²(R_prin|R_dist)={f['delta']:+.3f}  perm-p={f['p_perm']:.3f}  "
                     f"verdict: {law['named']}")
        ax.legend(fontsize=8, loc="best")
        ax.text(0.01, -0.13, "read = R_prin vs banked G; null = R_dist baseline (ΔLOO-R² beats it?). "
                + foot, transform=ax.transAxes, fontsize=6)
        fig.savefig(out / "F1_law_scatter.png", bbox_inches="tight", dpi=150)
        fig.savefig(out / "F1_law_scatter.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append("F1_law_scatter")

    # ---- F4: ΔLOO-R² partials ----
    if law:
        keys = [("R_prin_over_dist", "R_prin | R_dist"), ("R_ovl_over_dist", "R_ovl | R_dist"),
                ("R_prin_over_ovl", "R_prin | R_ovl"), ("R_ovl_over_prin", "R_ovl | R_prin")]
        labs = [lab for k, lab in keys if k in law["fit"]]
        dels = [law["fit"][k]["delta"] for k, _ in keys if k in law["fit"]]
        ps = [law["fit"][k]["p_perm"] for k, _ in keys if k in law["fit"]]
        cols = [OK_GREEN if p < 0.05 else OK_GREY for p in ps]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(labs))
        ax.bar(x, dels, color=cols, alpha=0.9)
        for xi, d, p in zip(x, dels, ps):
            ax.text(xi, d, f"p={p:.3f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel("ΔLOO-R² (added read | baseline)")
        ax.set_title(f"RELAW F4 — partial predictive power (perm-null)\npartials verdict: "
                     f"{law.get('partial_verdict')}")
        ax.text(0.01, -0.18, "green = perm-p<0.05; null = permuted (read,G) pairing. " + foot,
                transform=ax.transAxes, fontsize=6)
        fig.savefig(out / "F4_partials.png", bbox_inches="tight", dpi=150)
        fig.savefig(out / "F4_partials.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append("F4_partials")

    # ---- F3: Arm-3 insulation RF bars ----
    if ins:
        ps = ins["per_seed"]
        modes = [("RF_standard", "standard"), ("RF_off", "insulated\n(off)"), ("RF_into", "anti\n(into)")]
        means = {m: np.mean([ps[s][m] for s in ps]) for m, _ in modes}
        sds = {m: (np.std([ps[s][m] for s in ps]) if len(ps) > 1 else 0.0) for m, _ in modes}
        cols = {"RF_standard": OK_GREY, "RF_off": OK_GREEN, "RF_into": OK_ORANGE}
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = np.arange(len(modes))
        ax.bar(x, [means[m] for m, _ in modes], yerr=[sds[m] for m, _ in modes], capsize=5,
               color=[cols[m] for m, _ in modes], alpha=0.9)
        ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color=OK_BLUE, lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in modes])
        ax.set_ylabel("RF re-entry (1=deleted ceiling, 0=floor)")
        po = ins["pooled"]
        ax.set_title(f"RELAW F3 — substrate insulation lever\noff−standard=[{po['off_minus_standard']['lo']:+.2f},"
                     f"{po['off_minus_standard']['hi']:+.2f}] off−into=[{po['off_minus_into']['lo']:+.2f},"
                     f"{po['off_minus_into']['hi']:+.2f}] — {ins['named']}")
        ax.text(0.01, -0.16, "err=seed SD; null=anti (magnitude-matched into-subspace). " + foot,
                transform=ax.transAxes, fontsize=6)
        fig.savefig(out / "F3_insulation.png", bbox_inches="tight", dpi=150)
        fig.savefig(out / "F3_insulation.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append("F3_insulation")

    (out / "figures.json").write_text(json.dumps({"made": made, "sha": a.sha}, indent=1))
    print(f"[RELAW-FIGURES] OK — {made}")


if __name__ == "__main__":
    main()
