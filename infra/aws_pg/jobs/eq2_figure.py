"""E-Q2 cliff-decomposition figure (GraftLM trained-exploit prereg §10, fig_eq2_cliff_decomp).

Renders the gate result: is the no-retrain RefCOCO acc@0.5 cliff a discontinuous-metric artifact
(H_metric) or genuine generation-policy non-installation (H_policy)? Plots, across the rank ladder:
  - R_cap(acc@0.5)         — the thresholded metric
  - R_cap(continuous IoU)  — no 0.5 threshold
  - <loc>-emission fraction
all with bootstrap/binomial CI bands. The claim is independently backed by the numbers + CIs in the
metrics JSON: the continuous metric EQUALS the thresholded metric (both ~0) until emission flips on at
rank-1024 -> no smooth sub-threshold recovery -> the cliff is genuine policy non-installation, not a
metric artifact.

Reads s3://.../results/eq2_continuous_rescore (EXPLORATORY-DESCRIPTIVE). GPU-independent; --no-weights.
PNG + PDF. No try/except.

    pg run infra.aws_pg.jobs.eq2_figure --no-weights --type c7i.xlarge \
        --s3-prefix results/eq2_figs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
EQ2_KEY = "results/eq2_continuous_rescore/out/eq2_continuous_rescore.json"

C_ACC = "#0072B2"    # blue  — R_cap acc@0.5 (thresholded)
C_IOU = "#56B4E9"    # light blue — R_cap continuous IoU (overlays acc until the flip — the point)
C_EMIT = "#E69F00"   # amber — <loc>-emission fraction
C_CEIL = "#009E73"   # green — CEILING


def _fetch(key, dst):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--region", "us-east-1"],
                   check=True)


def _ci(d, lo="lo", hi="hi"):
    return (d[lo], d[hi]) if d else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eq2-key", default=EQ2_KEY)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    _fetch(a.eq2_key, out / "eq2.json")
    d = json.loads((out / "eq2.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [int(c[4:]) for c in d["configs"] if c.startswith("rank")]   # [8,32,128,512,1024]
    n = d["n"]
    ria = d["R_cap_iou_continuous"]["configs"]
    raa = d["R_cap_acc05"]["configs"]
    em = d["emission_fraction"]

    def series(src, key="R_cap"):
        ys = [src[f"rank{k}"][key] for k in ks]
        cis = [src[f"rank{k}"].get("R_cap_ci") for k in ks]
        lo = [(c["lo"] if c else y) for c, y in zip(cis, ys)]
        hi = [(c["hi"] if c else y) for c, y in zip(cis, ys)]
        return ys, lo, hi

    acc, acc_lo, acc_hi = series(raa)
    iou, iou_lo, iou_hi = series(ria)
    emit = [em[f"rank{k}"]["mean"] for k in ks]
    emit_lo = [em[f"rank{k}"]["lo"] for k in ks]
    emit_hi = [em[f"rank{k}"]["hi"] for k in ks]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    # emission fraction (the mechanism behind the cliff)
    ax.fill_between(ks, emit_lo, emit_hi, color=C_EMIT, alpha=0.15)
    ax.plot(ks, emit, "-^", color=C_EMIT, lw=2.0, ms=7, label="<loc>-emission fraction")
    # continuous IoU R_cap (no threshold)
    ax.fill_between(ks, iou_lo, iou_hi, color=C_IOU, alpha=0.20)
    ax.plot(ks, iou, "--D", color=C_IOU, lw=2.2, ms=6, label="R_cap (continuous IoU, no threshold)")
    # acc@0.5 R_cap (thresholded)
    ax.fill_between(ks, acc_lo, acc_hi, color=C_ACC, alpha=0.18)
    ax.plot(ks, acc, "-o", color=C_ACC, lw=2.4, ms=7, label="R_cap (acc@0.5, thresholded)")
    ax.axhline(1.0, color=C_CEIL, ls="--", lw=1.2,
               label=f"CEILING (PG2-pt): emit={em['CEILING']['mean']:.2f}, R_cap=1")
    ax.axhline(0.0, color="#888", ls=":", lw=1, label="FLOOR (base) = 0")

    # annotate the policy flip (rank-512 -> rank-1024)
    ax.annotate("no boxes emitted → continuous IoU = acc@0.5 = 0\n(NOT a thresholding artifact)",
                xy=(128, 0.0), xytext=(20, 0.34), fontsize=8.6,
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.annotate(f"<loc> policy flips on\n(emit {em['rank512']['mean']:.0%}→{em['rank1024']['mean']:.0%})",
                xy=(1024, emit[-1]), xytext=(300, 0.80), fontsize=8.6,
                arrowprops=dict(arrowstyle="->", color="#444"))

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("rank-k per decoder matrix (no-retrain SVD edit; log₂)")
    ax.set_ylabel("recovery fraction / emission fraction")
    ax.set_ylim(-0.08, 1.12)
    ax.set_title("E-Q2: the RefCOCO cliff is genuine policy non-installation, NOT a metric artifact\n"
                 "continuous-IoU R_cap = acc@0.5 R_cap = 0 until <loc>-emission flips on at ~eff-rank "
                 "(H_policy; metric-artifact confound ruled out)", fontsize=10.3)
    ax.legend(fontsize=8.2, loc="center left")
    ax.grid(alpha=0.25)
    prov = (f"s3://{BUCKET}/{a.eq2_key} (EXPLORATORY-DESCRIPTIVE) · RefCOCO testA N={n} · "
            f"bootstrap R_cap CI + binomial emission CI · greedy+seed-1337 (bit-identical to the frozen sweep)")
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_eq2_cliff_decomp.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    idx = {"figures": ["fig_eq2_cliff_decomp"], "tag": "EXPLORATORY-DESCRIPTIVE", "eq2": a.eq2_key, "n": n,
           "verdict": "H_policy (continuous = thresholded = 0 below emission flip; confound ruled out)",
           "by_rank": {f"rank{k}": {"emit": em[f"rank{k}"]["mean"], "R_cap_acc05": raa[f"rank{k}"]["R_cap"],
                                    "R_cap_iou_cont": ria[f"rank{k}"]["R_cap"]} for k in ks}}
    (out / "fig_index.json").write_text(json.dumps(idx, indent=2))
    print("EQ2_FIGS_OK", json.dumps(idx, indent=2))


if __name__ == "__main__":
    main()
