"""REFEREE figures v2 — POST-CRITIC complete-sweep suite (supersedes referee_figures.py,
kept as the pre-critic snapshot). Every panel binds `referee_aggregate3`.

f1 dose-response, three axes, all arms/seeds (the ladder headline; controls shaded)
f2 delivered dose HORIZON-MATCHED (the RF-E2 fix made visual: same-T comparison, both bands)
f3 in-core displacement vs step for every arm (the saturation finding, RF-E7 — replaces the
   inverted "sub-linear in lambda" panel)
f4 collateral map: ppl and battery per arm (log ppl; baseline band)
f5 P5 chain: final losses vs the 3x seed-matched bound, incl. the halved-eta retries
f6 R1 vfreeze (in-core motion vs baseline + post-snapshot increment)
f7 task-2 replication

    python -m infra.aws_pg.jobs.referee_figures2 --out <dir> --sha <commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f"}
SEEDS = (0, 1, 2)
CELLS = {"baseline": "referee_mde_s{s}/out/baseline",
         "t2incore_r": "referee_ladder_T2r/out/t2incore_r_s{s}",
         "t2lev_r": "referee_ladder_T2r/out/t2lev_r_s{s}",
         "incore1": "referee_ladder_L1/out/incore1_s{s}",
         "incore4": "referee_ladder_L2/out/incore4_s{s}",
         "incore16": "referee_mde_s{s}/out/incore16",
         "leverage16": "referee_mde_s{s}/out/leverage16",
         "levretry": "referee_ladder_L4/out/levretry_s{s}",
         "haar16": "referee_ladder_L3/out/haar16_s{s}",
         "t2base": "referee_ladder_T2a/out/t2base_s{s}",
         "t2incore": "referee_ladder_T2b/out/t2incore_s{s}",
         "t2lev": "referee_ladder_T2c/out/t2lev_s{s}"}
ONLY_SEEDS = {"t2incore_r": (0,), "t2lev_r": (0, 1)}
COL = {"baseline": "blue", "t2incore_r": "red", "t2lev_r": "orange", "incore1": "sky", "incore4": "purple", "incore16": "red",
       "leverage16": "orange", "levretry": "orange", "haar16": "green",
       "t2base": "blue", "t2incore": "red", "t2lev": "orange"}


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

    agg = _pull("referee_aggregate4/out/metrics.json", out / "in/agg4.json")
    C = {}
    for name, tpl in CELLS.items():
        for s in ONLY_SEEDS.get(name, SEEDS):
            C[(name, s)] = _pull(f"{tpl.format(s=s)}/metrics.json",
                                 out / f"in/{name}_{s}.json")
    R1 = {s: _pull(f"referee_r1/out/vfreeze_s{s}/metrics.json", out / f"in/vf{s}.json")
          for s in (0, 1)}
    made = []
    voided = {v["cell"] for v in agg["void"]}

    def save(fig, name, foot):
        fig.text(0.01, 0.005, f"{foot} · sha {a.sha}", fontsize=5, color="#666666")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(name)
        print(f"[FIG] {name}", flush=True)

    def med_sqrt(r, key):
        aud = r["motion_audit"][-1]["audit"]
        k0 = sorted(aud)[0]
        kk = key if key in aud[k0] else "in_core"
        return float(np.median([np.sqrt(aud[k][kk]) for k in sorted(aud)]))

    # ---- f1: dose-response, three axes -------------------------------------------------------
    ladder = ["baseline", "incore1", "incore4", "incore16"]
    lam_x = [0, 1, 4, 16]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, ttl in ((axs[0], "task_acc", "task fc (textvqa)"),
                         (axs[1], "battery_acc", "battery fc (cococap)"),
                         (axs[2], "ppl", "wikitext ppl (log)")):
        for i, nm in enumerate(ladder):
            vals = [C[(nm, s)]["evals"]["end"][key] for s in SEEDS
                    if f"{nm}_s{s}" not in voided or nm == "baseline"]
            if not vals:
                continue
            x = max(lam_x[i], 0.5)
            ax.errorbar(x, np.mean(vals),
                        yerr=[[np.mean(vals) - min(vals)], [max(vals) - np.mean(vals)]],
                        fmt="o-", ms=7, capsize=4, color=CB[COL[nm]],
                        label=nm if ax is axs[0] else None)
        for nm in ("leverage16", "levretry", "haar16"):
            vals = [C[(nm, s)]["evals"]["end"][key] for s in SEEDS
                    if f"{nm}_s{s}" not in voided]
            if vals:
                ax.plot([16] * len(vals), vals, "s", ms=6, color=CB[COL[nm]], alpha=0.75,
                        label=nm if ax is axs[0] else None)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("dose λ (0 = unhooked baseline)")
        ax.set_title(ttl, fontsize=9)
        if key == "ppl":
            ax.set_yscale("log")
    axs[0].legend(fontsize=6)
    fig.suptitle("REFEREE dose-response — in-core ladder (circles) vs matched-dose control "
                 "subspaces (squares); seed min/max bars", fontsize=10)
    save(fig, "f1_dose_response", "referee_{mde_s*,ladder_*}/…/metrics.json")

    # ---- f2: delivered dose, horizon-matched -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for nm, lam in (("incore1", 1), ("incore4", 4), ("incore16", 16),
                    ("leverage16", 16), ("levretry", 16), ("haar16", 16)):
        for s in SEEDS:
            cell = agg["cells"].get(f"{nm}_s{s}", {})
            r = cell.get("p4_ratio_horizon_matched")
            if r is None:
                continue
            ax.plot(lam, r, "o", ms=7, color=CB[COL[nm]], alpha=0.85,
                    label=nm if s == 0 else None)
    ax.axhspan(0.7, 1.4, color=CB["green"], alpha=0.20,
               label="frozen band [0.7,1.4] (P3, pre-data)")
    ax.axhspan(0.5, 2.0, color=CB["green"], alpha=0.08,
               label="post-authored band [0.5,2.0] (RF-E4)")
    ax.axhline(1.0, color=CB["green"], lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("nominal dose λ")
    ax.set_ylabel("delivered / nominal (HORIZON-MATCHED, RF-E2 fix)")
    ax.set_title("Delivered dose, corrected: the in-core dose WAS delivered at λ=16\n"
                 "(the pre-critic 0.19–0.23× was a T=500-vs-T=25 horizon mismatch)",
                 fontsize=9)
    ax.legend(fontsize=6)
    save(fig, "f2_delivered_dose_corrected", "referee_aggregate3/out/metrics.json")

    # ---- f3: in-core displacement saturation -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for nm in ("baseline", "incore1", "incore4", "incore16", "leverage16", "haar16"):
        r = C[(nm, 0)]
        steps = [row["step"] for row in r["motion_audit"]]
        key = "in_sub" if nm != "baseline" else "in_core"
        vals = []
        for row in r["motion_audit"]:
            aud = row["audit"]
            k0 = sorted(aud)[0]
            kk = key if key in aud[k0] else "in_core"
            vals.append(float(np.median([np.sqrt(aud[k][kk]) for k in sorted(aud)])))
        ax.plot(steps, vals, "-", lw=1.4, color=CB[COL[nm]], label=nm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel("median cumulative in-subspace displacement (log)")
    ax.set_title("Cumulative in-subspace motion SATURATES within ~25–50 steps at every dose\n"
                 "(slope ≈ T^0.1) — the corrected instrument finding (RF-E7)", fontsize=9)
    ax.legend(fontsize=6)
    save(fig, "f3_motion_saturation", "referee_{mde_s0,ladder_*}/…/metrics.json")

    # ---- f4: collateral map -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [n for n in ("baseline", "incore1", "incore4", "incore16", "leverage16",
                         "levretry", "haar16")]
    for i, nm in enumerate(names):
        for s in SEEDS:
            if f"{nm}_s{s}" in voided and nm != "baseline":
                continue
            ax.plot(i + (s - 1) * 0.12, C[(nm, s)]["evals"]["end"]["ppl"], "o", ms=7,
                    color=CB[COL[nm]])
    bl = [C[("baseline", s)]["evals"]["end"]["ppl"] for s in SEEDS]
    ax.axhspan(min(bl), max(bl), color=CB["blue"], alpha=0.15, label="baseline ppl band")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, fontsize=7)
    ax.set_ylabel("wikitext ppl (log)")
    ax.set_title("Collateral: raw-step dosing damages every subspace; the ordering across "
                 "arms is the finding", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f4_collateral", "referee_{mde_s*,ladder_*}/…/metrics.json")

    # ---- f5: P5 chain -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4))
    chain = agg["p5_chain"]
    keys = sorted(chain)
    for i, k in enumerate(keys):
        v = chain[k]
        ax.plot(i, v["final"], "o", ms=7,
                color=CB["red"] if v["verdict"] != "ok" else CB["green"])
        ax.plot(i, 3 * v["ref"], "_", ms=12, color="black")
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=80, fontsize=5)
    ax.set_ylabel("final loss (log); black dash = 3× seed-matched bound")
    ax.set_title("P5 divergence chain incl. the halved-η retries (RF-E3 fix)", fontsize=9)
    save(fig, "f5_p5_chain", "referee_aggregate3/out/metrics.json")

    # ---- f6: R1 -------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in (0, 1):
        for m_, col, lbl in ((R1[s], CB["red"], "vfreeze"),
                             (C[("baseline", s)], CB["blue"], "baseline")):
            steps = [row["step"] for row in m_["motion_audit"]]
            vals = [float(np.median([np.sqrt(row["audit"][k]["in_core"])
                                     for k in sorted(row["audit"])]))
                    for row in m_["motion_audit"]]
            ax.plot(steps, vals, "-", lw=1.3, color=col, label=lbl if s == 0 else None)
    ax.axvline(100, color=CB["grey"], ls=":", lw=1, label="v snapshot (step 100)")
    ax.set_xlabel("step")
    ax.set_ylabel("median in-core cumulative displacement")
    ax.set_title("R1: freezing v in-core releases ×1.45–1.50 of post-snapshot motion "
                 "(≪ SGD's ×16–80)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f6_r1_vfreeze", "referee_r1/out/*/metrics.json")

    # ---- f7: task-2 replication ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ttl in ((axs[0], "task_acc", "task fc (vqav2)"),
                         (axs[1], "ppl", "wikitext ppl (log)")):
        for i, nm in enumerate(("t2base", "t2incore", "t2lev", "t2incore_r", "t2lev_r")):
            for s in ONLY_SEEDS.get(nm, SEEDS):
                if f"{nm}_s{s}" in voided and nm != "t2base":
                    continue
                ax.plot(i + (s - 1) * 0.12, C[(nm, s)]["evals"]["end"][key], "o", ms=7,
                        color=CB[COL[nm]])
        ax.set_xticks(range(5))
        ax.set_xticklabels(["baseline", "incore λ16", "leverage λ16", "incore retry",
                            "leverage retry"], rotation=20, fontsize=7)
        ax.set_title(ttl, fontsize=9)
        if key == "ppl":
            ax.set_yscale("log")
    fig.suptitle("Task-2 replication (vqav2): does the arm ordering hold?", fontsize=10)
    save(fig, "f7_task2", "referee_ladder_T2*/out/*/metrics.json")

    # ---- f8: THE motion-vs-damage law (the program's adjudicating figure) --------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rows = agg["motion_damage_census"]["rows"]
    style = {("core", "textvqa"): ("o", CB["red"], "in-core (textvqa)"),
             ("core", "vqav2"): ("^", CB["purple"], "in-core (vqav2)"),
             ("leverage", "textvqa"): ("s", CB["orange"], "leverage (textvqa)"),
             ("leverage", "vqav2"): ("D", CB["sky"], "leverage (vqav2)")}
    seen = set()
    for r_ in rows:
        k = (r_["subspace"], r_["task"])
        mk, col, lbl = style[k]
        ax.plot(r_["motion_units"], r_["ppl"] / r_["ppl_base"], mk, ms=8, color=col,
                label=None if k in seen else lbl)
        seen.add(k)
    core_tv = [(r_["motion_units"], r_["ppl"] / r_["ppl_base"]) for r_ in rows
               if r_["subspace"] == "core" and r_["task"] == "textvqa"]
    if len(core_tv) >= 2:
        xs = np.array([x for x, _ in core_tv])
        ys = np.array([y for _, y in core_tv])
        b1, b0 = np.polyfit(np.log(xs), np.log(ys), 1)
        gx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(gx, np.exp(b0) * gx ** b1, "-", lw=1.2, color=CB["grey"],
                label=f"in-core textvqa power law (exp {b1:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("realized in-subspace motion (units of Adam's own out-of-core per-step motion)")
    ax.set_ylabel("collateral cost: ppl / seed-matched baseline ppl")
    ax.set_title("Collateral damage is governed by REALIZED MOTION, not by which subspace\n"
                 "receives it (power law ≈ motion²; leverage lies on the core's own curve on\n"
                 "textvqa, above it on the single vqav2 cell) — post-hoc census, labeled",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "f8_motion_damage_law", "referee_aggregate4/out/metrics.json")

    (out / "manifest.json").write_text(json.dumps({"figures": made, "sha": a.sha}, indent=1))
    print("REFEREE_FIGURES2_OK", json.dumps(made), flush=True)


if __name__ == "__main__":
    main()
