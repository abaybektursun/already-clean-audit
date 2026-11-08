"""REFEREE figures — complete-sweep suite. f1 MDE endpoints with VOID shading · f2 THE
instrument figure (nominal vs delivered dose, band shaded) · f3 loss curves (the P5
divergence asymmetry) · f4 R1 vfreeze (evals + in-core motion vs baseline). CPU-only.

    python -m infra.aws_pg.jobs.referee_figures --out <dir> --sha <commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "grey": "#7f7f7f"}
SEEDS = (0, 1, 2)
TARGET = 0.00517875318702413


def _pull(rel, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{rel}", str(dest),
                    "--region", "us-east-1"], check=True, capture_output=True)
    return json.loads(dest.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sha", default="unknown")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    agg = _pull("referee_aggregate2/out/metrics.json", out / "in/agg.json")
    C = {}
    for s in SEEDS:
        for arm in ("baseline", "incore16", "leverage16"):
            C[(arm, s)] = _pull(f"referee_mde_s{s}/out/{arm}/metrics.json",
                                out / f"in/{arm}_s{s}.json")
    sm1 = _pull("referee_smoke_incore/out/metrics.json", out / "in/smoke1.json")
    sm2 = _pull("referee_smoke_incore2/out/metrics.json", out / "in/smoke2.json")
    R1 = {}
    for s in (0, 1):
        R1[s] = _pull(f"referee_r1/out/vfreeze_s{s}/metrics.json", out / f"in/vf_{s}.json")
    made = []
    voided = {v["cell"] for v in agg["void"]}

    def save(fig, name, foot):
        fig.text(0.01, 0.005, f"{foot} · sha {a.sha}", fontsize=5, color="#666666")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(name)
        print(f"[FIG] {name}", flush=True)

    # ---- f1: MDE endpoints, VOID-shaded ------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, ttl in ((axs[0], "task_acc", "task fc (textvqa)"),
                         (axs[1], "battery_acc", "battery fc (cococap)"),
                         (axs[2], "ppl", "wikitext ppl")):
        for i, arm in enumerate(("baseline", "incore16", "leverage16")):
            for s in SEEDS:
                v = C[(arm, s)]["evals"]["end"][key]
                is_void = f"{arm}_s{s}" in voided
                ax.plot(i + (s - 1) * 0.12, v, "o", ms=7,
                        color=CB["grey"] if is_void else
                        {"baseline": CB["blue"], "incore16": CB["red"],
                         "leverage16": CB["orange"]}[arm],
                        alpha=0.5 if is_void else 1.0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["baseline", "incore λ16", "leverage λ16"], fontsize=8)
        ax.set_title(ttl, fontsize=9)
    axs[0].set_ylabel("endpoint value (grey = VOID cell, shown for census only)")
    fig.suptitle("MDE endpoints — 6/6 dose cells VOID (P4 undelivered and/or P5 diverged); "
                 "NO §1 verdict readable; census display", fontsize=10)
    save(fig, "f1_mde_endpoints_void", "referee_mde_s*/out/*/metrics.json")

    # ---- f2: THE instrument figure — nominal vs delivered dose -------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pts = []
    for sm, lam in ((sm1, 1), (sm2, 1)):
        aud = sm["motion_audit"][-1]["audit"]
        insub = [aud[k]["in_sub"] for k in sorted(aud)]
        realized = float(np.median(np.sqrt(insub))) / sm["n_steps"]
        pts.append((lam, realized / (lam * TARGET),
                    "v1 η_c" if sm is sm1 else "v2 η_c (frozen)"))
    for s in SEEDS:
        for arm, col in (("incore16", CB["red"]), ("leverage16", CB["orange"])):
            r = C[(arm, s)]
            aud = r["motion_audit"][-1]["audit"]
            insub = [aud[k]["in_sub"] for k in sorted(aud)]
            realized = float(np.median(np.sqrt(insub))) / r["n_steps"]
            ax.plot(16, realized / (16 * TARGET), "o", ms=7, color=col,
                    label=arm if s == 0 else None)
    ax.plot([1], [pts[1][1]], "s", ms=9, color=CB["blue"], label="λ=1 verify (v2 η_c)")
    ax.plot([1], [pts[0][1]], "s", ms=7, color=CB["grey"], label="λ=1 (v1 η_c, pre-P3)")
    ax.axhspan(0.5, 2.0, color=CB["green"], alpha=0.15, label="delivered band [0.5, 2.0]")
    ax.axhline(1.0, color=CB["green"], lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("nominal dose λ")
    ax.set_ylabel("delivered / nominal (realized in-sub motion ÷ λ·target)")
    ax.set_title("The dose ladder cannot deliver λ=16: delivery collapses from 1.03× (λ=1)\n"
                 "to 0.19–0.23× (in-core) — large raw steps cancel harder; the deadlock "
                 "resists forcing", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f2_delivered_dose", "referee_{smoke*,mde_s*}/…/metrics.json")

    # ---- f3: loss stability asymmetry --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, col in (("baseline", CB["blue"]), ("incore16", CB["red"]),
                     ("leverage16", CB["orange"])):
        for s in SEEDS:
            fl = C[(arm, s)]["final_loss_mean5"]
            ax.plot(s, fl, "o", ms=8, color=col, label=arm if s == 0 else None)
    ref = np.mean([C[("baseline", s)]["final_loss_mean5"] for s in SEEDS])
    ax.axhline(3 * ref, color="black", lw=1, ls="--", label="P5 divergence bound (3× baseline)")
    ax.set_yscale("log")
    ax.set_xticks(list(SEEDS))
    ax.set_xlabel("seed")
    ax.set_ylabel("final loss (mean of last 5 steps)")
    ax.set_title("Equal nominal dose, opposite stability: leverage-subspace forcing diverges "
                 "3/3;\nin-core forcing is stable 3/3 (census — endpoint channel VOID)",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f3_stability_asymmetry", "referee_mde_s*/out/*/metrics.json")

    # ---- f4: R1 vfreeze ----------------------------------------------------------------------
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4))
    for s in (0, 1):
        r = R1[s]
        b = C[("baseline", s)]
        for m_, col, lbl in ((r, CB["red"], "vfreeze"), (b, CB["blue"], "baseline")):
            steps = [row["step"] for row in m_["motion_audit"]]
            incore = [float(np.median([np.sqrt(row["audit"][k]["in_core"])
                                       for k in sorted(row["audit"])]))
                      for row in m_["motion_audit"]]
            axl.plot(steps, incore, "-", lw=1.2, color=col,
                     label=lbl if s == 0 else None)
        axr.plot(0 + s * 0.1, r["evals"]["end"]["task_acc"], "o", color=CB["red"],
                 label="vfreeze" if s == 0 else None)
        axr.plot(1 + s * 0.1, b["evals"]["end"]["task_acc"], "o", color=CB["blue"],
                 label="baseline" if s == 0 else None)
        axr.plot(2 + s * 0.1, r["evals"]["end"]["ppl"] / 100, "s", color=CB["red"])
        axr.plot(3 + s * 0.1, b["evals"]["end"]["ppl"] / 100, "s", color=CB["blue"])
    axl.set_xlabel("step")
    axl.set_ylabel("median in-core cumulative displacement")
    axl.set_title("R1: does freezing v in-core release the core?", fontsize=9)
    axl.legend(fontsize=7)
    axr.set_xticks([0, 1, 2, 3])
    axr.set_xticklabels(["task vf", "task base", "ppl/100 vf", "ppl/100 base"], fontsize=7)
    axr.set_title("R1 endpoints", fontsize=9)
    axr.legend(fontsize=7)
    save(fig, "f4_r1_vfreeze", "referee_r1/out/*/metrics.json")

    (out / "manifest.json").write_text(json.dumps({"figures": made, "sha": a.sha}, indent=1))
    print("REFEREE_FIGURES_OK", json.dumps(made), flush=True)


if __name__ == "__main__":
    main()
