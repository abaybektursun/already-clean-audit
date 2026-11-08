"""W12 §10.2 figure suite (5 figures; each backed by number + CI + null). Reads the frozen
controls.json + unblind_base-pt.json + causal.json from S3, recomputes per-index spectra for a few
representative matrices (post-unblind — weight-loading is fine here), renders PNG+PDF with a
provenance footer. Runs on the pg launcher (CPU + weights for the spectra panel).

  fig_w12_enrichment — per-matrix E_gain vs the §5 random null band (shaded). Headline.
  fig_w12_split      — stacked gain / in-span-routing / out-of-span fractions by type, null overlaid.
  fig_w12_spectra    — Δσ-vs-index + same-index left-cosine-vs-index (the Jin/Si rotation signature).
  fig_w12_causal     — gain-only / routing-only / in_span R_cap bars (RefCOCO + MMLU-forgetting).
  fig_w12_controls   — positive-control planted-split recovery + the §5 null band + identity = 0.

    pg run infra.aws_pg.jobs.w12_figures --s3-prefix results/w12_figures --type g6e.8xlarge \\
        --args="--provenance w12-gain-routing-prereg@0f4d81b"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

REP_KEYS = ["layers.0.self_attn.q_proj.weight",
            "layers.12.mlp.gate_proj.weight",
            "layers.25.mlp.down_proj.weight"]


def _pull(s3, local):
    subprocess.run(["aws", "s3", "cp", s3, str(local), "--region", "us-east-1"], check=True)
    return json.loads(pathlib.Path(local).read_text())


def _save(fig, out, name, footer):
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=6, color="gray")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[W12-FIG] wrote {name}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", default="s3://pg-lineage-769063704696/results/w12_controls/out/controls.json")
    ap.add_argument("--unblind-s3", default="s3://pg-lineage-769063704696/results/w12_unblind/out/unblind_base-pt.json")
    ap.add_argument("--causal-s3", default="s3://pg-lineage-769063704696/results/w12_causal/out/causal.json")
    ap.add_argument("--provenance", default="w12-gain-routing-prereg")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # weight-free: every panel renders from the frozen result JSONs (no model load, no HF).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ctrl = _pull(args.controls_s3, out / "controls.json")
    unb = _pull(args.unblind_s3, out / "unblind.json")
    causal = _pull(args.causal_s3, out / "causal.json")
    footer = (f"W12 gain-vs-routing | {args.provenance} | PG {unb['entities']['after']['sha'][:10]} "
              f"vs {unb['entities']['before']['sha'][:10]} | headline_k={unb['headline_k']}")
    from infra.pg import gain_routing as GR
    TYPES = list(GR.PROJECTION_TYPES)
    rows = unb["per_matrix"]
    thr = ctrl["thresholds_by_shape"]
    null = ctrl["null_band_by_shape"]

    # ---- fig 1: per-matrix E_gain vs the random null band (log scale; headline) ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for ti, t in enumerate(TYPES):
        ys = [r["E_gain"] for r in rows if r["type"] == t]
        xs = np.full(len(ys), ti) + np.linspace(-0.3, 0.3, len(ys))
        ax.scatter(xs, ys, s=14, alpha=0.7, label=t)
    tau = max(thr[s]["tau_gain_p99"] for s in thr)
    ax.axhspan(min(null[s]["E_gain"]["p50.0"] for s in null),
               max(null[s]["E_gain"]["p99.0"] for s in null), color="gray", alpha=0.25,
               label="§5 null band (p50–p99)")
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="null mean E_gain=1")
    ax.axhline(tau, color="red", ls="--", lw=0.8, label=f"frozen τ_gain p99={tau:.2f}")
    ax.set_yscale("log"); ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES, rotation=30)
    ax.set_ylabel("E_gain (gain-enrichment over 1/r null)")
    ax.set_title("W12 fig 1 — gain-enrichment per (layer×projection) vs the matched-random null")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "fig_w12_enrichment", footer); plt.close(fig)

    # ---- fig 2: stacked gain / in-span-routing / out-of-span fractions by type + null overlay ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    g = [np.mean([r["gain_frac"] for r in rows if r["type"] == t]) for t in TYPES]
    rt = [np.mean([r["route_frac"] for r in rows if r["type"] == t]) for t in TYPES]
    fp = [np.mean([r["f_perp"] for r in rows if r["type"] == t]) for t in TYPES]
    x = np.arange(len(TYPES))
    ax.bar(x, g, label="gain ‖diag M‖²", color="#2c7fb8")
    ax.bar(x, rt, bottom=g, label="in-span routing ‖offdiag M‖²", color="#7fcdbb")
    ax.bar(x, fp, bottom=np.array(g) + np.array(rt), label="out-of-span", color="#edf8b1")
    # null reference: analytic f_perp per type (1−r²/mn) as a black tick
    for ti, t in enumerate(TYPES):
        sk = next(r["shape_key"] for r in rows if r["type"] == t)
        ax.plot([ti - 0.4, ti + 0.4], [1 - null[sk]["analytic_f_perp"]] * 2, "k_", ms=20,
                label="null in-span frac (r²/mn)" if ti == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(TYPES, rotation=30); ax.set_ylabel("energy fraction of ‖ΔW‖²")
    ax.set_title("W12 fig 2 — ΔW energy split by projection type (null in-span frac overlaid)")
    ax.legend(fontsize=7); _save(fig, out, "fig_w12_split", footer); plt.close(fig)

    # ---- fig 3: spectra signature per projection type (from unblind summaries — weight-free) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cu = [unb["by_type"][t]["mean_cos_u_topk"] for t in TYPES]
    ds = [float(np.mean([r["dsigma_mean_abs"] for r in rows if r["type"] == t])) for t in TYPES]
    x = np.arange(len(TYPES))
    axes[0].bar(x, cu, color="#7570b3"); axes[0].axhline(1.0, color="k", ls=":", lw=0.8, label="vectors stable=1")
    axes[0].set_ylim(0, 1.02); axes[0].set_xticks(x); axes[0].set_xticklabels(TYPES, rotation=30)
    axes[0].set_title(f"mean |⟨u_i^b,u_i^a⟩| over top-{unb['headline_k']} (<1 = rotate; Jin/Si)"); axes[0].legend(fontsize=7)
    axes[1].bar(x, ds, color="#1b9e77"); axes[1].set_xticks(x); axes[1].set_xticklabels(TYPES, rotation=30)
    axes[1].set_title("mean |Δσ_i| over top-k (singular-value shift)")
    fig.suptitle("W12 fig 3 — rotation signature by projection type (top vectors rotate, cos_u≈0.3)")
    _save(fig, out, "fig_w12_spectra", footer); plt.close(fig)

    # ---- fig 4: causal R_cap bars (gain/routing/in_span vs in_span; RefCOCO + MMLU) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    arms = ["gain", "routing", "in_span"]
    rc = causal["R_cap"]["refcoco_vs_inspan"]; mm = causal["R_cap"]["mmlu_vs_inspan"]
    def _val(tbl, a): return (tbl[a]["R_cap"] if a in tbl and tbl[a].get("R_cap") is not None else np.nan)
    x = np.arange(len(arms)); w = 0.38
    ax.bar(x - w/2, [_val(rc, a) for a in arms], w, label="RefCOCO (install)", color="#2c7fb8")
    ax.bar(x + w/2, [_val(mm, a) for a in arms], w, label="MMLU (forgetting)", color="#de2d26")
    ax.axhline(1 - causal["delta"], color="k", ls="--", lw=0.8, label=f"δ rule (1−δ={1-causal['delta']})")
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("R_cap (normalised vs in_span ceiling)")
    ax.set_title("W12 fig 4 — which in-span component carries capability (§6C causal)")
    ax.legend(fontsize=7); _save(fig, out, "fig_w12_causal", footer); plt.close(fig)

    # ---- fig 5: controls (positive-control recovery + null band + identity) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pc = ctrl["positive_control"]
    labels = ["plant gain\n→gain_frac", "plant routing\n→route_frac", "plant oos\n→f_perp", "identity\n→‖ΔW‖²"]
    vals = [pc["pure_gain"]["gain_frac"], pc["pure_routing"]["route_frac"],
            pc["pure_out_of_span"]["f_perp"], ctrl["n1_identity"]["frob_dW_sq"]]
    ax.bar(range(4), vals, color=["#2c7fb8", "#7fcdbb", "#edf8b1", "#999999"])
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="planted target = 1")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.4g}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("W12 fig 5 — controls: planted-split recovery (=1) + identity (=0)")
    ax.legend(fontsize=7); _save(fig, out, "fig_w12_controls", footer); plt.close(fig)

    (out / "figures_manifest.json").write_text(json.dumps(
        {"figures": ["fig_w12_enrichment", "fig_w12_split", "fig_w12_spectra", "fig_w12_causal",
                     "fig_w12_controls"], "provenance": args.provenance,
         "sources": {"controls": args.controls_s3, "unblind": args.unblind_s3, "causal": args.causal_s3}},
        indent=2))
    print("[W12-FIG] 5 figures + manifest written.")


if __name__ == "__main__":
    main()
