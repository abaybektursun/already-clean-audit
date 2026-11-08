"""GESTALT figure suite — weight-free, from banked JSONs only (house rule: figures recompute
nothing; every panel shows the number + CI + the null that backs it; twin/null curves always
overlaid; complete-sweep rule: every K / step / pair plotted).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_figures --type r7i.2xlarge --no-weights \
      --args "--aggregate results/gestalt_aggregate/out --git-sha <sha>" \
      --s3-prefix results/gestalt_figures --budget-min 30

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

REAL_C = "#1f77b4"
TWIN_C = "#d62728"
NULL_C = "#7f7f7f"


def _pull(prefix, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"s3://pg-lineage-769063704696/{prefix}", str(dest),
                    "--region", "us-east-1", "--only-show-errors"], check=True)


def _save(fig, out, name, foot):
    fig.text(0.01, 0.005, foot, fontsize=6, color="#666666")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"[FIG] {name}", flush=True)


def fig_s_curves(agg, site_tag, out, foot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = agg[f"arm_s_{site_tag}"]
    ks = sorted(int(k) for k in m["per_k"] if "p_K" in m["per_k"][k])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for src, col in (("real", REAL_C), ("twin", TWIN_C)):
        seeds = sorted(m["per_k"][str(ks[0])][f"{src}_scores"])
        for s in seeds:
            ys = [m["per_k"][str(k)][f"{src}_scores"][s] for k in ks]
            ax.plot(ks, ys, "o--", color=col, alpha=0.45, ms=4, lw=0.8)
        mean = [sum(m["per_k"][str(k)][f"{src}_scores"][s] for s in seeds) / len(seeds)
                for k in ks]
        ax.plot(ks, mean, "-", color=col, lw=2.2,
                label=f"{src} fit (seeds {','.join(map(str, seeds))})")
    base = m["unswapped"]
    ax.axhline(base["score"], color="k", lw=1.2, label="unswapped forward")
    ax.axhspan(base["ci95"][0], base["ci95"][1], color="k", alpha=0.08)
    p3 = m.get("p3_identity")
    if p3:
        ax.axhline(p3["score"], color="#2ca02c", lw=1.0, ls=":",
                   label="P3 identity swap (hook ceiling)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K (splat components; rank r=2 frozen at dry-run)")
    ax.set_ylabel("VQAv2 soft-acc under splat-swap (800 frozen discovery items)")
    ax.set_title(f"Arm S {site_tag}: K-vs-fidelity, real vs moment-matched twin\n"
                 f"(member p={m['member_p']:.4g} at K={m['k_at_min_p']}; "
                 f"how to read: separation of blue above red = beyond-order-2 shape content)",
                 fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, out, f"gestalt_s_curve_{site_tag}", foot)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k in ks:
        pr = m["per_k"][str(k)]["pairings"]
        worst = max(pr, key=lambda p: p["p_le0"])
        ax.errorbar([k], [worst["diff"]],
                    yerr=[[worst["diff"] - worst["lo95"]], [worst["hi95"] - worst["diff"]]],
                    fmt="o", color=REAL_C, capsize=4)
        ax.annotate(f"p={m['per_k'][str(k)]['p_K']:.3g}", (k, worst["hi95"]),
                    fontsize=7, ha="center", va="bottom")
    ax.axhline(0.0, color=NULL_C, lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K")
    ax.set_ylabel("fidelity(real) − fidelity(twin), worst seed pairing")
    ax.set_title(f"Arm S {site_tag}: per-K real−twin paired difference "
                 f"(whiskers = item-paired 95% CI; null = 0 line)", fontsize=10)
    _save(fig, out, f"gestalt_s_diff_{site_tag}", foot)
    plt.close(fig)


def fig_f1(agg, out, foot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f1 = (agg.get("arm_f") or {}).get("F1")
    if not f1:
        return
    steps = sorted(int(s) for s in f1["per_step"])
    diff = [f1["per_step"][str(s)]["diff"] if str(s) in f1["per_step"] else
            f1["per_step"][s]["diff"] for s in steps]

    def _e(s):
        return f1["per_step"].get(str(s), f1["per_step"].get(s))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(steps, [d["d_rel"] for d in diff], "o-", color=REAL_C, lw=2,
            label="D_rel(U3c@s − U3@s)")
    ax.fill_between(steps, [d["lo95"] for d in diff], [d["hi95"] for d in diff],
                    color=REAL_C, alpha=0.2, label="probe-bootstrap 95% CI")
    drift = [(_e(s) or {}).get("drift_band") for s in steps]
    ds = [(s, x) for s, x in zip(steps, drift) if x is not None]
    if ds:
        ax.plot([s for s, _ in ds], [x for _, x in ds], "s--", color=NULL_C, lw=1.5,
                label="FR-3(b) null: max adjacent-step drift")
    ax.set_xlabel("training step (film checkpoints, seed0)")
    ax.set_ylabel("normalized field difference D_rel")
    ax.set_title("Arm F F1: does VL co-training (U3c) shape the action-expert field beyond "
                 "training drift?\n(how to read: blue above gray at matched step = develops)",
                 fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, out, "gestalt_f1_steps", foot)
    plt.close(fig)


def fig_f2(agg, out, foot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lad = (agg.get("arm_f") or {}).get("F2_ladder")
    if not lad:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lvls = list(lad)
    xs = [lad[l]["params"] for l in lvls]
    ax.plot(xs, [lad[l]["behavior_mse"] for l in lvls], "o-", color=REAL_C,
            label="surrogate action-MSE (10-step, held obs)")
    ax.plot(xs, [lad[l]["proxy_mse_held"] for l in lvls], "^--", color="#9467bd",
            label="field-MSE proxy (held probes)")
    thr = lad[lvls[0]]["behavior_thresh_5v10"]
    ax.axhline(thr, color=NULL_C, ls="--",
               label="FR-7 gate: true field's own 5-vs-10-step deviation")
    for l, x in zip(lvls, xs):
        ax.annotate(l, (x, lad[l]["behavior_mse"]), fontsize=8, ha="left", va="bottom")
    exp = (agg.get("arm_f") or {}).get("expert_params")
    if exp:
        ax.axvline(exp, color="k", lw=1.0, ls=":", label=f"expert params ({exp:.2e})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("surrogate parameter count")
    ax.set_ylabel("MSE")
    v = (agg.get("arm_f") or {}).get("F2_verdict") or {}
    ax.set_title(f"Arm F F2: behavior-certified surrogate ladder — Borges ratio "
                 f"{v.get('borges_ratio')}\n(a rung below the dashed line is behaviorally "
                 f"certified; ≈1 ratio is a result either way)", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, out, "gestalt_f2_ladder", foot)
    plt.close(fig)


def fig_d(agg, out, foot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = agg.get("arm_d")
    if not cells:
        return
    fig, axes = plt.subplots(1, len(cells), figsize=(5.5 * len(cells), 5), squeeze=False)
    for ax, site_rec in zip(axes[0], cells):
        if "VOID" in site_rec:
            ax.set_title(f"site {site_rec.get('site')}: VOID ({site_rec['VOID']})")
            continue
        tgts = site_rec.get("targets", [])
        names = [t["target"] for t in tgts]
        for i, t in enumerate(tgts):
            pairs = [p for p in t["seed_pairs"] if "survival" in p]
            ys = [p["survival"] for p in pairs]
            fl = [p["twin_floor"] for p in pairs]
            ax.scatter([i] * len(ys), ys, color=REAL_C, zorder=3, label="survival" if i == 0 else None)
            ax.scatter([i] * len(fl), fl, color=TWIN_C, marker="v", zorder=3,
                       label="twin floor" if i == 0 else None)
        ax.axhline(site_rec["ceiling_selfhalf_survival"], color="k", ls="--",
                   label="split-half ceiling (P4)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("PG1-component survival fraction")
        ax.set_title(f"site {site_rec['site']} (τ_move={site_rec['p4_band']['tau_move']:.3g})",
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Arm D: atlas survival across the lineage vs twin floors + split-half ceiling",
                 fontsize=11)
    _save(fig, out, "gestalt_d_survival", foot)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--aggregate", required=True, help="aggregate OUT prefix on S3")
    ap.add_argument("--git-sha", default="unknown")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pathlib.Path("/tmp/figs_agg")
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"s3://pg-lineage-769063704696/{a.aggregate}", str(d),
                    "--region", "us-east-1", "--only-show-errors"], check=True)
    agg = json.loads((d / "aggregate.json").read_text())
    foot = (f"GESTALT · aggregate={a.aggregate} · git={a.git_sha} · "
            f"figures recomputed from aggregate.json only")
    for site_tag in ("site9", "site14"):
        if f"arm_s_{site_tag}" in agg:
            fig_s_curves(agg, site_tag, out, foot)
    fig_f1(agg, out, foot)
    fig_f2(agg, out, foot)
    fig_d(agg, out, foot)
    print("GESTALT_FIGURES_OK", flush=True)


if __name__ == "__main__":
    main()
