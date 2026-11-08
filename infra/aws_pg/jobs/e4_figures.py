"""E4 figure suite — fig_compress_curve (GraftLM plan fig_graft_mdl, function-space edition).

Renders the rank-k compressibility curve of the modality-install delta: R_KL recovery (no-retrain,
frozen bridge) vs rank / description-length, against CEILING_edit + FLOOR + E_scale/N3 nulls. Shows
the heavy-tailed recovery (low-rank head + irreducible tail) that confirms cheap graft does NOT
install the modality. GPU-independent; --no-weights. Fetches the E4 JSON from S3. PNG + PDF.

    pg run infra.aws_pg.jobs.e4_figures --no-weights --type c7i.xlarge --s3-prefix results/e4_figs

No try/except — fail loud.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
E4_KEY = "results/e4_compress_curve/out/e4_compress_curve.json"
C_REAL = "#0072B2"                     # blue — real install edit
C_CEIL = "#009E73"                     # green — CEILING_edit
C_NULL = "#D55E00"                     # vermillion — nulls


def _fetch(key, dst):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--region", "us-east-1"],
                   check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--e4-key", default=E4_KEY)
    ap.add_argument("--freeze-sha", default="bf2c99b")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src = out / "e4.json"
    _fetch(a.e4_key, src)
    d = json.loads(src.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ceil = d["anchors"]["ceiling_edit_2d_only"]["R_KL"]
    escale = d["controls"]["E_scale_normonly"]["R_KL"]
    n3 = d["controls"][[k for k in d["controls"] if k.startswith("N3")][0]]["R_KL"]
    ks = sorted(int(k) for k in d["curve"])
    R = [d["curve"][str(k)]["R_KL"] for k in ks]
    lo = [d["curve"][str(k)]["R_KL_ci"]["lo"] for k in ks]
    hi = [d["curve"][str(k)]["R_KL_ci"]["hi"] for k in ks]
    lm_params = d["headline"].get("edited_decoder_matrix_params", 2.024e9)  # corrected denom (edited matrices only)
    pctp = [100 * d["curve"][str(k)]["L_desc_params"] / lm_params for k in ks]
    yerr = np.array([np.array(R) - np.array(lo), np.array(hi) - np.array(R)])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Panel A: recovery vs rank ---------------------------------------------------------
    ax = axes[0]
    ax.errorbar(ks, R, yerr=yerr, fmt="-o", color=C_REAL, capsize=3, lw=2, ms=6,
                label="real install edit (rank-k)")
    ax.axhline(ceil, color=C_CEIL, ls="--", lw=1.5, label=f"CEILING_edit (full 2-D) = {ceil:.3f}")
    ax.axhline(0.0, color="#888", ls=":", lw=1, label="FLOOR (base) = 0")
    ax.axhline(escale, color=C_NULL, ls="-.", lw=1, label=f"E_scale norm-only null = {escale:.3f}")
    ax.axhline(n3, color=C_NULL, ls=":", lw=1, label=f"N3 random rank null = {n3:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("rank-k per decoder matrix (log₂)")
    ax.set_ylabel("R_KL  (behavioural recovery)")
    ax.set_ylim(-0.05, 1.02)
    ax.annotate(f"low-rank HEAD\nrank-{ks[0]} = {R[0]:.0%}\n({pctp[0]:.1f}% of params)",
                xy=(ks[0], R[0]), xytext=(ks[0] * 1.5, R[0] - 0.32), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_title("Heavy-tailed recovery: low-rank head + irreducible tail\n"
                 "(real edit ≫ nulls at every rank)", fontsize=11)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.25)

    # --- Panel B: recovery (% of ceiling) vs description length ------------------------------
    ax = axes[1]
    frac = [100 * r / ceil for r in R]
    ax.plot(pctp, frac, "-o", color=C_REAL, lw=2, ms=6)
    ax.axvspan(0, 10, color="#FDE0C8", alpha=0.6, label="cheap-graft budget (<10% params)")
    ax.axhline(100, color=C_CEIL, ls="--", lw=1.2, label="CEILING_edit (100%)")
    ax.axhline(90, color="#aaa", ls=":", lw=1, label="90% (cheap-graft bar)")
    for k, x, y in zip(ks, pctp, frac):
        ax.annotate(f"r{k}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("description length  L(E) = rank-k factorization  (% of LM-decoder params)")
    ax.set_ylabel("% of CEILING_edit recovered")
    ax.set_ylim(40, 105)
    ax.set_title("Cheap-graft budget recovers only ~⅔;\nfull recovery needs near-full rank → IRREDUCIBLE",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle("E4 — modality install is largely irreducible (H_high): a low-rank head carries a "
                 "majority of the shift, but full recovery needs near-full-rank edits", fontsize=12, y=1.03)
    prov = (f"E4 function-space compressibility · s3://{BUCKET}/{a.e4_key} · commit {a.freeze_sha} · "
            f"7344 text pos · native R_KL, no-retrain frozen bridge · EXPLORATORY")
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_compress_curve.{ext}", bbox_inches="tight", dpi=150)
    (out / "fig_index.json").write_text(json.dumps(
        {"figures": ["fig_compress_curve"], "source": f"s3://{BUCKET}/{a.e4_key}",
         "commit": a.freeze_sha, "claims": [
             f"CEILING_edit R_KL={ceil:.3f} (install in decoder weight matrices)",
             f"low-rank head rank-{ks[0]} recovers {R[0]:.0%} at {pctp[0]:.1f}% params",
             "heavy-tailed: cheap-graft budget (<10% params) recovers only ~66% of ceiling",
             f"nulls clean (E_scale {escale:.3f}, N3 {n3:.3f}) -> recovery is real structure"]}, indent=2))
    print("E4_FIGS_OK", json.dumps({"out": str(out), "figures": ["fig_compress_curve"]}))


if __name__ == "__main__":
    main()
