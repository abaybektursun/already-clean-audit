"""TENANT figures — complete-sweep suite (all arms/seeds plotted, CIs + bands + power floors
annotated; colourblind-safe; PNG+PDF; provenance footers). CPU-only; run AFTER the aggregate.

    python -m infra.aws_pg.jobs.tenant_figures --out <dir> --sha <commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "grey": "#7f7f7f"}
SEEDS = (1337, 1338)
ARMS = (("incore", "blue"), ("leverage", "orange"), ("random", "green"))
X_STEPS = ("adamw_s0", "adamw_s1", "sgd_s0", "sgd_s1")


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

    agg = _pull("tenant_aggregate2/out/metrics.json", out / "in/agg.json")
    cells = {}
    for arm, _ in ARMS:
        for s in SEEDS:
            pre = (f"tenant_mde_s{s}/out/incore" if arm == "incore"
                   else f"tenant_mde2_s{s}/out/{arm}")
            cells[(arm, s)] = _pull(f"{pre}/metrics.json", out / f"in/{arm}_{s}.json")
    X_SRC = {"adamw_s0": "tenant_x4/out/adamw_s0", "adamw_s1": "tenant_x_full3/out/adamw_s1",
             "sgd_s0": "tenant_x4/out/sgd_s0", "sgd_s1": "tenant_x4/out/sgd_s1"}
    X = {st: _pull(f"{X_SRC[st]}/metrics.json", out / f"in/x_{st}.json") for st in X_STEPS}
    made = []

    def save(fig, name, foot):
        fig.text(0.01, 0.005, f"{foot} · sha {a.sha}", fontsize=5, color="#666666")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(name)
        print(f"[FIG] {name}", flush=True)

    rngb = np.random.default_rng(7)

    def ci(pi):
        x = np.asarray(pi, float)
        x = x[~np.isnan(x)]
        idx = rngb.integers(0, x.size, size=(4000, x.size))
        m = x[idx].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    # ---- f1: MDE endpoints + PAIRED contrasts (critic m22) -----------------------------------
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11, 4))
    for i, (arm, col) in enumerate(ARMS):
        for j, s in enumerate(SEEDS):
            m, lo, hi = ci(cells[(arm, s)]["endpoint"]["per_item"])
            x = i + (j - 0.5) * 0.25
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=6, capsize=4,
                        color=CB[col], label=f"{arm}" if j == 0 else None)
            ax.annotate(f"s{s}", (x, lo - 0.012), fontsize=6, ha="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels([a0 for a0, _ in ARMS])
    ax.set_ylabel("endpoint fc accuracy (n=395 scorable; UNPAIRED per-arm CIs)")
    ps = {m["name"]: m["p"] for m in agg["families"]["F-placement"]["members"]}
    n2 = agg.get("n2_reprint_post_critic", {})
    ax.set_title("MDE (cococap, r512): UNDERPOWERED NON-REPLICATION (post-critic)\n"
                 f"p(in>lev)={ps['incore_vs_leverage_cococap']:.3f}, "
                 f"p(in>rand)={ps['incore_vs_random_cococap']:.3f}; power at the registered "
                 f"effect {n2.get('alpha_bh_m7', float('nan')):.2f} (BH m=7); "
                 f"live items {agg['mde_census']['frac_live']:.1%}", fontsize=9)
    ax.legend(fontsize=7)
    pos = 0
    for ctrl in ("leverage", "random"):
        for s in SEEDS:
            A = np.asarray(cells[("incore", s)]["endpoint"]["per_item"], float)
            B = np.asarray(cells[(ctrl, s)]["endpoint"]["per_item"], float)
            msk = ~(np.isnan(A) | np.isnan(B))
            d = A[msk] - B[msk]
            idx = rngb.integers(0, d.size, size=(4000, d.size))
            bs = d[idx].mean(axis=1)
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            axd.errorbar(pos, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o",
                         ms=6, capsize=4, color=CB["blue" if ctrl == "leverage" else "green"])
            axd.annotate(f"s{s}", (pos, lo - 0.004), fontsize=6, ha="center")
            pos += 1
    axd.axhline(0, color=CB["grey"], lw=1, ls=":")
    axd.set_xticks(range(4))
    axd.set_xticklabels(["in-lev s1337", "in-lev s1338", "in-rand s1337", "in-rand s1338"],
                        rotation=20, fontsize=7)
    axd.set_ylabel("ITEM-PAIRED contrast (in-core - control), 95% CI")
    axd.set_title("paired contrasts (the tested quantity)", fontsize=9)
    save(fig, "f1_mde_endpoints", "tenant_mde*/out/*/metrics.json")

    # ---- f2: capture vs conversion -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, col in ARMS:
        for s in SEEDS:
            cap = cells[(arm, s)].get("q5_captured_share") or 0
            m, lo, hi = ci(cells[(arm, s)]["endpoint"]["per_item"])
            base = ci(cells[("random", s)]["endpoint"]["per_item"])[0]
            ax.errorbar(cap, m - base, yerr=[[m - lo], [hi - m]], fmt="o", ms=6,
                        color=CB[col], capsize=3,
                        label=arm if s == SEEDS[0] else None)
    ax.axhline(0, color=CB["grey"], lw=1, ls=":")
    ax.set_xlabel("Q5: captured gradient share in the arm's subspace")
    ax.set_ylabel("endpoint advantage vs the random arm")
    ax.set_title("Capture != conversion - and capture is LEVERAGE-driven, not core-specific:\n"
                 "leverage 0.40 > core 0.26 >> random 0.016 = Haar; none converts (cococap)",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f2_capture_conversion", "tenant_mde*/out/*/metrics.json")

    # ---- f3: the mechanism headline ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fm = agg["families"]["F-mechanism"]["adam_suppression"]
    labels = ["Adam s0", "Adam s1", "SGD s0", "SGD s1"]
    grads = fm["adam_grad_range"] + fm["sgd_grad_range"]
    disps = fm["adam_disp_range"] + fm["sgd_disp_range"]
    xs = np.arange(4)
    ax.bar(xs - 0.18, grads, width=0.36, color=CB["blue"], label="in-core gradient ratio")
    ax.bar(xs + 0.18, disps, width=0.36, color=CB["red"], label="in-core displacement ratio")
    ax.axhline(1.0, color=CB["grey"], lw=1, ls=":", label="parity (=matched-random)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("in-core / matched-random (median over 18 q_proj)")
    ax.set_title("Adam takes 16x gradient and moves the core 0.2x.\n"
                 "SGD/adamfrozen contrasts: v4 re-run values (the v1 SGD arm DIVERGED, "
                 "critic F1)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f3_mechanism", "tenant_x_full3/out/*/metrics.json")

    # ---- f4: lag-autocorr spectra ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"adamw_s0": CB["blue"], "adamw_s1": "#67a9cf",
              "sgd_s0": CB["red"], "sgd_s1": "#ef8a62"}
    for st in X_STEPS:
        for k, v in X[st]["lag_autocorr"].items():
            lay = k.split("__")[2]
            ax.plot(range(1, len(v) + 1), v, "o-", ms=3, color=colors[st], alpha=0.8,
                    label=f"{st}" if lay == "5" else None)
    ax.axhline(0, color=CB["grey"], lw=1, ls=":")
    ax.axhline(-0.9, color="black", lw=1, ls="--", label="take-revert signature (≈ −1)")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("in-core step autocorrelation (L5/L9/L14)")
    ax.set_title("H_take-revert KILLED: in-core steps are strongly PERSISTENT (+0.9), "
                 "not reverted", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f4_lag_autocorr", "tenant_x_full3/out/*/metrics.json")

    # ---- f5: qk cancellation index over training ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for st in X_STEPS:
        steps, means = [], []
        for row in X[st]["qk_index"]:
            vals = [v["mean"] for v in row.values()
                    if isinstance(v, dict) and v.get("mean") is not None]
            if vals:
                steps.append(row["step"])
                means.append(float(np.mean(vals)))
        ax.plot(steps, means, "-", lw=1, color=colors[st], label=st)
    ni = agg["families"]["F-mechanism"]["qk_cancel"].get("no_interaction_value")
    if ni is not None and ni == ni:
        ax.axhline(ni, color=CB["grey"], lw=1.2, ls=":",
                   label=f"no-interaction value {ni:.3f} (measured norms; NOT 1.0)")
    ax.axhline(0.05, color="black", lw=1, ls="--", label="pairing-cancel signature (≈0)")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("step")
    ax.set_ylabel("head-sliced q/k product cancellation index (mean of 3 layers × 8 heads)")
    ax.set_title("H_qk-cancel not supported: index 0.75–0.92, far from the ≈0 signature",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f5_qk_index", "tenant_x_full3/out/*/metrics.json")

    # ---- f6: confinement census (A-TENANT-1) -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, col in ARMS:
        for s in SEEDS:
            cc = cells[(arm, s)]["confinement"]
            ax.plot([c["step"] for c in cc], [c["max_leak"] for c in cc], "-", lw=1.2,
                    color=CB[col], alpha=0.8, label=arm if s == SEEDS[0] else None)
    ax.set_xlabel("step")
    ax.set_ylabel("max B-factor span leakage (A-TENANT-1 census)")
    ax.set_title("Adam carries subspace-seeded adapters off-span (leak → 0.87–0.9);\n"
                 "the random arm leaks least (0.17) — the arms are gradient-confined",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f6_confinement", "tenant_mde*/out/*/metrics.json")

    anch = _pull("tenant_anchor2/out/metrics.json", out / "in/anchor.json")["tasks"]["cococap"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, col in ARMS:
        for s in SEEDS:
            tr = cells[(arm, s)]["trajectory"]
            ax.plot([r["step"] for r in tr], [r["acc"] for r in tr], "-", lw=1.2,
                    color=CB[col], alpha=0.8, label=arm if s == SEEDS[0] else None)
    ax.axhline(anch["base_acc"], color="black", lw=1, ls=":",
               label=f"anchor base {anch['base_acc']:.3f}")
    ax.axhline(anch["ft_acc"], color="black", lw=1, ls="--",
               label=f"anchor FT {anch['ft_acc']:.3f}")
    ax.set_xlabel("step")
    ax.set_ylabel("trajectory fc accuracy (first-100 sub-pool)")
    ax.set_title("All arms operate ABOVE the anchor-validated range (critic M6);\n"
                 "the trajectory pool is ~0.15 harder than the endpoint pool", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f7_trajectories", "tenant_mde*/out/*/metrics.json")

    (out / "manifest.json").write_text(json.dumps({"figures": made, "sha": a.sha}, indent=1))
    print("TENANT_FIGURES_OK", json.dumps(made), flush=True)


if __name__ == "__main__":
    main()
