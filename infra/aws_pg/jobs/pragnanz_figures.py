"""PRAGNANZ figures — the complete-sweep suite (every swept point + ALL component curves +
CIs + nulls overlaid; colourblind-safe; PNG + PDF; provenance footers). CPU-only, reads the
banks + the aggregate's metrics.json from S3. Run AFTER the aggregate.

    python -m infra.aws_pg.jobs.pragnanz_figures --out <dir> --sha <commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "black": "#000000"}
M4_ARMS = [("baseline", "black"), ("core", "blue"), ("random", "orange"), ("dome", "purple")]
M5_ARMS = [("m5_perp", "green"), ("m5_rand", "orange"), ("m5_incore", "red")]
SEEDS = [1337, 1338]


def _pull(rel, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{rel}", str(dest),
                    "--region", "us-east-1"], check=True, capture_output=True)
    return json.loads(dest.read_text())


def boot_ci(x, rng, n_boot=4000):
    import numpy as np
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


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

    rng = np.random.default_rng(20260805)
    made = []

    def save(fig, name, foot):
        fig.text(0.01, 0.005, f"{foot} · sha {a.sha}", fontsize=5, color="#666666")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(name)
        print(f"[FIG] {name}", flush=True)

    # ---- F1: the ladder (MDE headline) -------------------------------------------------------
    lb = _pull("pragnanz_ladder/out/ladder_banks.json", out / "in/ladder_banks.json")
    agg = _pull("pragnanz_aggregate2/out/metrics.json", out / "in/aggregate.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, unit in zip(axes, ("pg1", "ft-aokvqa-da")):
        xs, labels = [], []
        pos = 0
        for K in (32, 16):
            for rung in ("r1", "r2"):
                for s in (0, 1):
                    m, lo, hi = boot_ci(lb[f"{unit}|K{K}|{rung}|s{s}"]["per_item"], rng)
                    c = CB["blue"] if rung == "r1" else CB["orange"]
                    ax.errorbar(pos, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=5,
                                capsize=3)
                    xs.append(pos)
                    labels.append(f"K{K}·{rung}·s{s}")
                    pos += 1
            pos += 0.6
        m0, lo0, hi0 = boot_ci(lb[f"{unit}|r0"]["per_item"], rng)
        ax.axhline(m0, color=CB["green"], lw=1.2, label=f"r0 baseline {m0:.3f}")
        ax.axhspan(lo0, hi0, color=CB["green"], alpha=0.12)
        f3 = [boot_ci(lb[f"{unit}|r3|s{s}"]["per_item"], rng) for s in (0, 1)]
        ax.axhspan(min(f[1] for f in f3), max(f[2] for f in f3), color=CB["red"], alpha=0.12,
                   label=f"r3 floor per-seed CIs [{f3[0][0]:.3f},{f3[1][0]:.3f}]")
        for f3s in f3:
            ax.axhline(f3s[0], color=CB["red"], lw=0.9, ls="--")
        ax.axhline(0.25, color=CB["grey"], lw=1.0, ls=":", label="chance 0.25")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, fontsize=6)
        p = next(f["p"] for f in agg["families"]["F-read"]["members"]
                 if f["name"] == f"ladder_{unit}_K32")
        ax.set_title(f"{unit}: Δ(r1−r2) member p={p:.3f} (frozen worst-of-4, K32)", fontsize=9)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("forced-choice accuracy")
    fig.suptitle("PRAGNANZ ladder — resample (r1, blue) vs permuted-resample (r2, orange); "
                 "every rung/seed with 95% item-bootstrap CIs", fontsize=10)
    save(fig, "f1_ladder", "results/pragnanz_ladder/out/ladder_banks.json")

    # ---- F2: R2 corroboration + restore ------------------------------------------------------
    r2b = _pull("pragnanz_r2/out/r2_banks.json", out / "in/r2_banks.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, keys, title in (
            (axes[0], ["pg1|rt", "pg1|shuf|s0", "pg1|shuf|s1"],
             "BSF corroboration (pg1, 800 items)"),
            (axes[1], ["pi0|none", "pi0|transport", "pi0|randtrans|s0", "pi0|randtrans|s1"],
             "restore (pi0-reshell, 400 items)")):
        for i, k in enumerate(keys):
            m, lo, hi = boot_ci(r2b[k]["per_item"], rng)
            ax.bar(i, m, color=CB["sky"], width=0.6)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", capsize=4)
        ax.axhline(0.25, color=CB["grey"], ls=":", lw=1, label="chance 0.25")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.split("|", 1)[1] for k in keys], rotation=30, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    ps = {f["name"]: f["p"] for f in agg["families"]["F-read"]["members"]}
    axes[0].set_ylabel("forced-choice accuracy")
    fig.suptitle(f"Wing-R completion — corroboration p={ps['bsf_corroboration']:.4f}, "
                 f"restore-vs-random p={ps['restore_vs_random']:.4f} (both worst-of-seeds)",
                 fontsize=10)
    save(fig, "f2_r2_cells", "results/pragnanz_r2/out/r2_banks.json")

    # ---- F3: M1 ratio-of-ratios censuses -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dst in zip(axes, ("textvqa", "cococap")):
        m1 = _pull(f"pragnanz_m1_{dst}/out/metrics.json", out / f"in/m1_{dst}.json")["m1"]
        lays, cores, rands = [], [], []
        for k in sorted(m1, key=lambda x: int(x.split("__")[2])):
            lays.append(int(k.split("__")[2]))
            cores.append(m1[k]["core"]["ratio"])
            rands.append([m1[k][f"rand{j}"]["ratio"] for j in range(3)])
        rands = np.array(rands)
        ax.plot(lays, cores, "o-", color=CB["blue"], label="in-core ratio")
        ax.plot(lays, rands.mean(1), "s--", color=CB["orange"], label="matched-random (mean of 3)")
        ax.fill_between(lays, rands.min(1), rands.max(1), color=CB["orange"], alpha=0.2,
                        label="random min–max")
        ax.set_yscale("log")
        ax.set_xlabel("layer (q_proj)")
        ax.set_title(f"{dst}: RoR median "
                     f"{np.median(np.array(cores)/rands.mean(1)):.2f} (registered ≪ 1)",
                     fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean²/var (mean-cancellation ratio)")
    fig.suptitle("M1 — in-core vs matched-random mean-cancellation; the registered RoR≪1 "
                 "signature is ABSENT (medians > 1)", fontsize=10)
    save(fig, "f3_m1_meancancel", "results/pragnanz_m1_*/out/metrics.json")

    # ---- F4: M2 DOME-core overlap spectra ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dst in zip(axes, ("textvqa", "cococap")):
        m2 = json.loads((out / f"in/m1_{dst}.json").read_text())["m2"]
        for k, col in zip(sorted(m2, key=lambda x: int(x.split("__")[2])),
                          (CB["blue"], CB["green"], CB["purple"])):
            lay = k.split("__")[2]
            v = m2[k]
            ax.plot(range(1, 33), v["dome_core_fracs"], "o-", ms=3, color=col,
                    label=f"L{lay} (null [{v['null']['lo']:.3f},{v['null']['hi']:.3f}])")
            ax.axhspan(v["null"]["lo"], v["null"]["hi"], color=CB["grey"], alpha=0.25)
        ax.set_xlabel("DOME basis column index (QR order, unsorted — not eigenvalue rank)")
        ax.set_title(dst, fontsize=9)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("fraction inside the inherited core")
    fig.suptitle("M2 — DOME nuisance eigenvectors vs the ERLANGEN core; grey band = 24-draw "
                 "null (≈0.125). Every vector, every layer, both destinations above the band.",
                 fontsize=10)
    save(fig, "f4_m2_sameobject", "results/pragnanz_m1_*/out/metrics.json")

    # ---- F5: M3 ceiling ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for dst, col in (("textvqa", CB["blue"]), ("cococap", CB["orange"])):
        m3 = json.loads((out / f"in/m1_{dst}.json").read_text())["m3"]
        y = m3["deadlocked_fraction_per_seg"]
        ax.plot(range(1, len(y) + 1), y, "o-", color=col, label=dst)
    ax.axhline(0.01, color=CB["grey"], ls=":", lw=1, label="1% (spec branch line)")
    ax.set_ylim(0, 0.015)
    ax.set_xlabel("training segment (100 steps each)")
    ax.set_ylabel("in-core q_proj energy / total grad energy")
    ax.set_title("M3 — in-core energy share ≈ 0.8% on the VLM fine-tunes\n"
                 "(~20× an isotropic 32-dim share; scope: NOT measured on the M4/M5 recipe)",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f5_m3_ceiling", "results/pragnanz_m1_*/out/metrics.json")

    # ---- F6/F7: M4 + M5 trajectories ---------------------------------------------------------
    crit = agg["criterion"]
    for name, arms, ttl in (("f6_m4_trajectories", M4_ARMS, "M4 gradient-projection arms"),
                            ("f7_m5_trajectories", M5_ARMS, "M5 placement arms (rank-32)")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, seed in zip(axes, SEEDS):
            plot_arms = arms if name == "f6_m4_trajectories" else \
                [("baseline", "black")] + arms          # critic #9: f7 needs the baseline
            for arm, col in plot_arms:
                key = f"{arm}_s{seed}"
                cell = agg["m4_cells"].get(key)
                if cell is None:
                    continue
                tr = cell["trajectory"]
                ax.plot([t["step"] for t in tr], [t["S"] for t in tr], "-", lw=1.4,
                        color=CB[col], label=f"{arm} (S_end {cell['S_end']:.3f})")
            b = agg["m4_cells"][f"baseline_s{seed}"].get("p4")
            if b:
                ax.axhspan(b["banked_ci"][0], b["banked_ci"][1], color=CB["green"],
                           alpha=0.12, label="banked SAV CI (P4)")
            ax.axhline(crit["S_star_lo"], color="black", ls="--", lw=1,
                       label=f"S*_lo {crit['S_star_lo']:.3f} (primary)")
            ax.axhline(crit["S_star_hi"], color="black", ls=":", lw=1,
                       label=f"S*_hi {crit['S_star_hi']:.3f}")
            ax.set_title(f"seed {seed}", fontsize=9)
            ax.set_xlabel("step")
            ax.legend(fontsize=6)
        axes[0].set_ylabel("S_text (trajectory sub-bucket)")
        fig.suptitle(f"{ttl} — every arm/seed, criterion lines, banked anchor", fontsize=10)
        save(fig, name, "results/pragnanz_m4_*|m5_*/out/metrics.json")

    (out / "manifest.json").write_text(json.dumps({"figures": made, "sha": a.sha}, indent=1))
    print("PRAGNANZ_FIGURES_OK", json.dumps(made), flush=True)


if __name__ == "__main__":
    main()
