"""E2 figure suite — fig_pt_vs_task (GraftLM plan fig #2, weight-space edition).

Renders the PT-vs-task LM weight decomposition: modality-install (base->pt) vs task-tuning
(pt->mix). Each panel illustrates a claim backed by the E2 numbers (eff-rank, rel-Frob, in-span
energy vs chance). GPU-independent; launch with --no-weights (no model load, no weight sync).
Fetches the E2 result JSON from S3, writes PNG (review) + PDF (paper) with a provenance footer.

    pg run infra.aws_pg.jobs.e2_figures --no-weights --type c7i.xlarge --s3-prefix results/e2_figs

No try/except — fail loud.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
E2_KEY = "results/e2_pt_vs_task/out/e2_pt_vs_task.json"
C_INSTALL = "#0072B2"   # blue  — base->pt (modality install)
C_TASK = "#D55E00"      # vermillion — pt->mix (task tuning)
MT_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _fetch(key, dst):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--region", "us-east-1"],
                   check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--e2-key", default=E2_KEY)
    ap.add_argument("--freeze-sha", default="8ce82c6")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src = out / "e2_pt_vs_task.json"
    _fetch(a.e2_key, src)
    d = json.loads(src.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bp = d["axes"]["base_pt"]["per_matrix"]
    pm = d["axes"]["pt_mix"]["per_matrix"]
    head = d["headline"]
    prov = (f"E2 weight decomposition · s3://{BUCKET}/{a.e2_key} · commit {a.freeze_sha} · "
            f"base={d['provenance']['base']} pt={d['provenance']['pt']} mix={d['provenance']['mix']} · EXPLORATORY")

    def col(per, field, mt=None, layer=None):
        return [m[field] for m in per.values()
                if (mt is None or m["matrix_type"] == mt) and (layer is None or m["layer"] == layer)]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # --- Panel 1 (headline): eff-rank vs magnitude, per matrix, by axis -----------------------
    ax = axes[0]
    ax.scatter(col(bp, "participation_ratio"), col(bp, "rel_frob"), s=18, alpha=0.7,
               color=C_INSTALL, label=f"base→pt (install)  eff-rank μ={head['base_pt_eff_rank_mean']:.0f}")
    ax.scatter(col(pm, "participation_ratio"), col(pm, "rel_frob"), s=18, alpha=0.7,
               color=C_TASK, label=f"pt→mix (task)  eff-rank μ={head['pt_mix_eff_rank_mean']:.0f}")
    ax.axhline(head["base_pt_rel_frob_mean"], color=C_INSTALL, ls="--", lw=1, alpha=0.6)
    ax.axhline(head["pt_mix_rel_frob_mean"], color=C_TASK, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("eff-rank  (SVD participation ratio, max 2304)")
    ax.set_ylabel("rel-Frobenius  ‖ΔW‖/‖W₀‖  (magnitude)")
    ax.set_title("Same rank, different magnitude\n(install ≈ 2.2× task; both full-rank)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # --- Panel 2: depth profile — rel-Frob by layer ------------------------------------------
    ax = axes[1]
    Ls = list(range(d["config"]["n_layers"]))
    bp_by_L = [np.mean(col(bp, "rel_frob", layer=L)) for L in Ls]
    pm_by_L = [np.mean(col(pm, "rel_frob", layer=L)) for L in Ls]
    ax.plot(Ls, bp_by_L, "-o", ms=3, color=C_INSTALL, label="base→pt (install)")
    ax.plot(Ls, pm_by_L, "-o", ms=3, color=C_TASK, label="pt→mix (task)")
    ax.set_xlabel("decoder layer (0–25)")
    ax.set_ylabel("rel-Frobenius (mean over 7 matrices)")
    ax.set_ylim(0, max(bp_by_L) * 1.25)
    ax.set_title("Depth-uniform at both stages\n(no concentration in any layer band)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 3: per-module eff-rank, both axes (identical module shape) ----------------------
    ax = axes[2]
    x = np.arange(len(MT_ORDER))
    bp_mt = [np.mean(col(bp, "participation_ratio", mt=mt)) for mt in MT_ORDER]
    pm_mt = [np.mean(col(pm, "participation_ratio", mt=mt)) for mt in MT_ORDER]
    ax.bar(x - 0.2, bp_mt, 0.4, color=C_INSTALL, label="base→pt (install)")
    ax.bar(x + 0.2, pm_mt, 0.4, color=C_TASK, label="pt→mix (task)")
    ax.axhline(2304, color="#444", ls=":", lw=1)
    ax.text(0.1, 2304 * 0.97, "full rank 2304", fontsize=7, color="#444", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_proj", "") for m in MT_ORDER], rotation=30, fontsize=8)
    ax.set_ylabel("eff-rank (participation ratio)")
    ax.set_title("Identical per-module rank shape\n(MLP ≫ attn k/v in BOTH axes)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("E2 — becoming-a-VLM vs learning-the-task: same dense full-rank re-weighting, "
                 "install just larger in magnitude", fontsize=12.5, y=1.02)
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_pt_vs_task.{ext}", bbox_inches="tight", dpi=150)
    (out / "fig_index.json").write_text(json.dumps(
        {"figures": ["fig_pt_vs_task"], "source": f"s3://{BUCKET}/{a.e2_key}",
         "commit": a.freeze_sha, "claims": [
             "both axes full-rank (eff-rank ~1440 / ~1361 of 2304)",
             "install ~2.2x task in magnitude (rel-Frob 0.37 vs 0.17)",
             "depth-uniform; identical per-module rank shape"]}, indent=2))
    print("E2_FIGS_OK", json.dumps({"out": str(out), "figures": ["fig_pt_vs_task"]}))


if __name__ == "__main__":
    main()
