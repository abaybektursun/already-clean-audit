"""W11 figure suite (§10.2) — each figure illustrates a claim backed by a number + CI + null.
PNG (review) + PDF (paper), colourblind-safe, null/floor overlaid, plain finding in the title,
"how to read" line, provenance footer. Reads the frozen result JSONs from S3 (no GPU needed).

    python -m infra.aws_pg.jobs.w11_figures \
        --legA-unblind-s3 s3://.../w11_legA_unblind/out/legA_unblind.json \
        --legA-controls-s3 s3://.../w11_legA_controls/out/legA_controls.json \
        --legB-unblind-s3 s3://.../w11_legB_unblind/out/legB_unblind.json \
        --legB-controls-s3 s3://.../w11_legB_controls_v3/out/legB_controls.json \
        --freeze-sha 3374082 --out <dir>

Implemented: fig_w11a_retention, fig_w11b_concentration, fig_w11_controls.
Deferred (logged, not silently dropped): fig_w11b_content_map — the OCR/object/spatial
selective-ablation attribution was not harvested (exploratory; the ablation is the evidence).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

CB = {"base": "#0072B2", "before": "#56B4E9", "after": "#D55E00", "floor": "#999999",
      "structured": "#D55E00", "random": "#0072B2", "null": "#999999", "ok": "#009E73"}


def _pull(uri, dest):
    subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", "us-east-1"], check=True)
    return json.loads(pathlib.Path(dest).read_text())


def _footer(fig, prov):
    fig.text(0.005, 0.004, prov, fontsize=5, color="#666", ha="left", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legA-unblind-s3", required=True)
    ap.add_argument("--legA-controls-s3", required=True)
    ap.add_argument("--legB-unblind-s3", required=True)
    ap.add_argument("--legB-controls-s3", required=True)
    ap.add_argument("--freeze-sha", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    AU = _pull(args.legA_unblind_s3, out / "legA_unblind.json")
    AC = _pull(args.legA_controls_s3, out / "legA_controls.json")
    BU = _pull(args.legB_unblind_s3, out / "legB_unblind.json")
    BC = _pull(args.legB_controls_s3, out / "legB_controls.json")
    prov = f"W11 behavioral-closure · freeze {args.freeze_sha} · PG2-pt(LegA)/PG2-mix(LegB)"

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {name}")

    # ============================================================= fig 1: retention (Leg A)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    mm = AC["mmlu"]
    accs = {"base": mm["base"]["acc_pct"], "BEFORE\n(decoder-reset)": mm["BEFORE"]["acc_pct"],
            "AFTER\n(PG2-LM)": AU["mmlu"]["acc_after_pct"], "FLOOR\n(random-dec)": mm["FLOOR"]["acc_pct"]}
    cols = [CB["base"], CB["before"], CB["after"], CB["floor"]]
    ax1.bar(range(4), list(accs.values()), color=cols)
    ax1.set_xticks(range(4)); ax1.set_xticklabels(list(accs.keys()), fontsize=8)
    ax1.axhline(25.0, ls=":", color=CB["floor"], lw=1, label="4-way chance (25%)")
    St = AU["S_text_capability"]
    ax1.set_ylabel("MMLU 5-shot accuracy (%)")
    ax1.set_title(f"Leg A — text-only LM capability\nS_text={St['point']:.3f} "
                  f"CI[{St['ci95'][0]:.3f},{St['ci95'][1]:.3f}]  →  {AU['VERDICT']}", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right")
    for i, v in enumerate(accs.values()):
        ax1.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    pp = AU["perplexity"]
    ax2.bar([0, 1], [pp["ppl_base"], pp["ppl_after"]], color=[CB["base"], CB["after"]])
    ax2.set_yscale("log"); ax2.set_xticks([0, 1]); ax2.set_xticklabels(["base", "AFTER (PG2-LM)"])
    ax2.set_ylabel("wikitext-2 perplexity (log)")
    ax2.set_title(f"Leg A — fluency (secondary)\nppl ratio={pp['ratio_point']:.2f}× "
                  f"CI[{pp['ratio_ci95'][0]:.1f},{pp['ratio_ci95'][1]:.1f}]", fontsize=10)
    for i, v in enumerate([pp["ppl_base"], pp["ppl_after"]]):
        ax2.text(i, v * 1.1, f"{v:.0f}", ha="center", fontsize=8)
    fig.text(0.5, 0.965, "How to read: AFTER at the FLOOR ⇒ text-only capability lost; "
             "BEFORE≈base ⇒ the decoder-reset is a valid base anchor.", ha="center", fontsize=7.5, color="#444")
    _footer(fig, prov)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_w11a_retention")

    # ============================================================= fig 2: concentration (Leg B)
    tasks = list(BU["per_task"].keys())
    sig = BU["fdr"]["significant"]
    n = len(tasks)
    ncol = 2; nrow = (n + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.0 * nrow))
    axes = axes.ravel()
    for ax, t in zip(axes, tasks):
        pt = BU["per_task"][t]
        curve = pt["curve"]
        ks = sorted((int(k) for k in curve), key=int)
        sd = [curve[str(k)]["structured_drop"] for k in ks]
        rd = [curve[str(k)]["random_drop"] for k in ks]
        ax.plot(ks, sd, "-o", color=CB["structured"], label="structured top-k", ms=4)
        ax.plot(ks, rd, "--s", color=CB["random"], label="random-k (floor)", ms=3)
        ax.set_xscale("log")
        for k in ks:
            if sig.get(f"{t}@k{k}", False):
                ax.plot(k, curve[str(k)]["structured_drop"], "*", color=CB["ok"], ms=11, zorder=5)
        if pt.get("knee_k"):
            ax.axvline(pt["knee_k"], ls=":", color="#444", lw=1, label=f"knee k={pt['knee_k']}")
        ax.set_title(f"{t}  (base={pt['baseline']:.3f})", fontsize=9)
        ax.set_xlabel("# directions ablated (k)"); ax.set_ylabel("capability drop")
        ax.legend(fontsize=6.5)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Leg B — projector concentration: structured vs random  →  {BU['VERDICT']}\n"
                 "How to read: structured ≫ random at small k = info concentrated in the bridge "
                 "(installs); structured≈random until large k = diffuse re-basis. ★=BH-FDR sig.",
                 fontsize=9.5)
    _footer(fig, prov)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_w11b_concentration")

    # ============================================================= fig 3: controls (instrument-works)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a = [mm["base"]["acc_pct"], mm["BEFORE"]["acc_pct"], mm["FLOOR"]["acc_pct"]]
    ax1.bar(range(3), a, color=[CB["base"], CB["before"], CB["floor"]])
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["base", "BEFORE", "FLOOR"])
    ax1.axhline(25, ls=":", color=CB["floor"], lw=1)
    ax1.set_ylabel("MMLU (%)")
    ax1.set_title(f"Leg A controls — extraction_ok={AC['checks']['extraction_ok']} "
                  f"(BEFORE≈base), FLOOR≈chance", fontsize=9)
    btasks = [t for t in BC["per_task"]]
    import numpy as np
    x = np.arange(len(btasks)); w = 0.27
    base_v = [BC["per_task"][t]["baseline"] for t in btasks]
    all_v = [BC["per_task"][t]["all_ablate"] for t in btasks]
    n4_v = [BC["per_task"][t]["n4_shuffle"] for t in btasks]
    ax2.bar(x - w, base_v, w, label="baseline", color=CB["base"])
    ax2.bar(x, all_v, w, label="all-ablate (dead)", color=CB["after"])
    ax2.bar(x + w, n4_v, w, label="N4 shuffle", color=CB["floor"])
    ax2.set_xticks(x); ax2.set_xticklabels(btasks, fontsize=7, rotation=15)
    ax2.set_ylabel("task capability")
    ax2.set_title(f"Leg B controls — all-ablate bites + N4 collapses "
                  f"(random within ±{BC['checks'].get('rand_band', 0.05)})", fontsize=9)
    ax2.legend(fontsize=7)
    fig.text(0.5, 0.96, "How to read: instruments validated before unblinding — "
             "the contrast scales (Leg A) and the ablation bites/floor (Leg B) are real.",
             ha="center", fontsize=7.5, color="#444")
    _footer(fig, prov)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_w11_controls")

    print("[W11-FIG] deferred (logged): fig_w11b_content_map (OCR/object/spatial attribution "
          "not harvested — exploratory; the ablation is the evidence).")
    (out / "figures_manifest.json").write_text(json.dumps({
        "figures": ["fig_w11a_retention", "fig_w11b_concentration", "fig_w11_controls"],
        "deferred": ["fig_w11b_content_map"],
        "legA_verdict": AU["VERDICT"], "legB_verdict": BU["VERDICT"],
        "freeze_sha": args.freeze_sha}, indent=2))


if __name__ == "__main__":
    main()
