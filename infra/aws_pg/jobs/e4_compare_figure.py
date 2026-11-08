"""E4 install-vs-task figure — fig_install_vs_task (GraftLM headline comparison).

Overlays the function-space compressibility curves of the two LM deltas (base->pt modality install
vs pt->mix task tuning) and decomposes their CEILING_edit (2-D weight matrices) vs the norm/gain
contribution. Reads the two corrected E4 JSONs from S3. GPU-independent; --no-weights. PNG + PDF.

    pg run infra.aws_pg.jobs.e4_compare_figure --no-weights --type c7i.xlarge --s3-prefix results/e4_cmp_fig

No try/except — fail loud.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
C_INSTALL = "#0072B2"   # blue — base->pt (modality install)
C_TASK = "#D55E00"      # vermillion — pt->mix (task tuning)


def _fetch(key, dst):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--region", "us-east-1"], check=True)


def _load(key, dst):
    _fetch(key, dst)
    return json.loads(dst.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basept-key", default="results/e4_basept_v2/out/e4_compress_curve.json")
    ap.add_argument("--ptmix-key", default="results/e4_ptmix_v2/out/e4_compress_curve.json")
    ap.add_argument("--freeze-sha", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    inst = _load(a.basept_key, out / "basept.json")
    task = _load(a.ptmix_key, out / "ptmix.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def curve(d):
        ks = sorted(int(k) for k in d["curve"])
        R = [d["curve"][str(k)]["R_KL"] for k in ks]
        lo = [d["curve"][str(k)]["R_KL_ci"]["lo"] for k in ks]
        hi = [d["curve"][str(k)]["R_KL_ci"]["hi"] for k in ks]
        ce = d["anchors"]["ceiling_edit_2d_only"]["R_KL"]
        plumb = d["anchors"]["plumbing_full_after"]["R_KL"]
        return ks, R, lo, hi, ce, plumb

    iks, iR, ilo, ihi, ice, ipl = curve(inst)
    tks, tR, tlo, thi, tce, tpl = curve(task)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Panel A: raw R_KL recovery curves, both axes ---------------------------------------
    ax = axes[0]
    ax.errorbar(iks, iR, yerr=[np.array(iR) - ilo, np.array(ihi) - iR], fmt="-o", color=C_INSTALL,
                capsize=3, lw=2, ms=6, label=f"base→pt (install)  baseKL={inst['baseline_kl_full_base']:.1f}")
    ax.errorbar(tks, tR, yerr=[np.array(tR) - tlo, np.array(thi) - tR], fmt="-s", color=C_TASK,
                capsize=3, lw=2, ms=6, label=f"pt→mix (task)  baseKL={task['baseline_kl_full_base']:.1f}")
    ax.axhline(ice, color=C_INSTALL, ls="--", lw=1, alpha=0.7)
    ax.axhline(tce, color=C_TASK, ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(iks); ax.set_xticklabels([str(k) for k in iks])
    ax.set_xlabel("rank-k per decoder matrix (log₂)")
    ax.set_ylabel("R_KL  (KL-mass-weighted divergence closure)")
    ax.set_ylim(0, 1.02)
    ax.annotate(f"install head\nrank-8 = {iR[0]:.2f}", xy=(iks[0], iR[0]),
                xytext=(iks[0] * 1.6, iR[0] + 0.06), fontsize=8, color=C_INSTALL)
    ax.annotate(f"task head\nrank-8 = {tR[0]:.2f}", xy=(tks[0], tR[0]),
                xytext=(tks[0] * 1.6, tR[0] - 0.16), fontsize=8, color=C_TASK)
    ax.set_title("Install has the bigger low-rank head; both heavy-tailed\n"
                 "(R_KL = divergence closure, NOT capability; dashed = each CEILING_edit)", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # --- Panel B: where the delta lives — weight matrices vs norms/embed ----------------------
    ax = axes[1]
    labels = ["base→pt\n(install)", "pt→mix\n(task)"]
    x = np.arange(2)
    ce_vals = [ice, tce]                                  # 2-D weight matrices
    norm_vals = [ipl - ice, tpl - tce]                    # norms (+embed for pt-mix) = PLUMBING - CEILING_edit
    ax.bar(x, ce_vals, 0.5, color=[C_INSTALL, C_TASK], label="decoder weight matrices (CEILING_edit)")
    ax.bar(x, norm_vals, 0.5, bottom=ce_vals, color="#999", label="norms/gains (+embed) = PLUMBING − CEILING_edit")
    for xi, ce, nv in zip(x, ce_vals, norm_vals):
        ax.text(xi, ce / 2, f"{ce:.3f}", ha="center", va="center", color="white", fontsize=9)
        ax.text(xi, ce + nv / 2 + 0.005, f"{nv:.3f}", ha="center", va="center", fontsize=8)
    ax.axhline(1.0, color="#444", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("R_KL recovered (full edit)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Task-tuning leans on norms/gains far more than install\n"
                 "(~9% vs ~0.6%) — refines E2's 'identical-structure' (E2 excluded norms)", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower center")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("E4 — modality install vs task tuning in function space: same heavy-tailed shape, but "
                 "install is bigger + more head-concentrated + weight-matrix-borne; task leans on norms",
                 fontsize=11, y=1.03)
    prov = (f"install s3://{BUCKET}/{a.basept_key} · task s3://{BUCKET}/{a.ptmix_key} · commit {a.freeze_sha} · "
            f"native R_KL no-retrain, all probe text positions · EXPLORATORY")
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_install_vs_task.{ext}", bbox_inches="tight", dpi=150)
    (out / "fig_index.json").write_text(json.dumps(
        {"figures": ["fig_install_vs_task"], "install": a.basept_key, "task": a.ptmix_key,
         "claims": ["install low-rank head (rank-8) > task head",
                    "task leans on norms/gains ~9% vs install ~0.6% (CEILING_edit gap)",
                    "install ~3x larger behavioural shift (baseline KL)",
                    "both heavy-tailed; neither cheap-graftable to ceiling"]}, indent=2))
    print("E4_CMP_FIG_OK", json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
