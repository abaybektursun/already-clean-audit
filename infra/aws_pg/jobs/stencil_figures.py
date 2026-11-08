"""STENCIL figure suite (CPU) — spec §10. Reads the discovery + confirmation artifacts and renders
the primary-path panels (colourblind-safe Okabe-Ito palette; PNG+PDF; provenance footer). Robust to
partial / VOID landings — a panel with no data is skipped and logged, never faked.

Panels rendered from disc.json + conf.json (+ banks metrics if present):
  1  HEADLINE frontier — task-gain vs forgetting, one curve per arm (up-and-left is better)
  2  rank-efficiency  — task-gain vs rank per arm
  3  collateral       — forgetting vs rank per arm
  6  learning curves  — task-gain & forgetting vs step at r* (from confirmation trajectories)
  8  cross-task replication — per-subject E_adapt
Panels 4/5/7 (natural-write census / per-layer / leverage) render only when the corroboration arms
have run; skipped + logged otherwise.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_figures \
      --args "--disc-s3 results/stencil_disc/out/stencil_disc.json --conf-s3 results/stencil_conf/out/stencil_conf.json --disc-metrics results/stencil_disc/out/metrics.json --conf-metrics results/stencil_conf/out/metrics.json" \
      --type r7i.2xlarge --no-weights --s3-prefix results/stencil_figs --budget-min 30
"""
import argparse
import json
import os
import pathlib

# Okabe-Ito colourblind-safe categorical palette (arm → colour)
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
ARM_COLOR = {"susceptibility": OI["vermillion"], "energy": OI["blue"], "write": OI["green"],
             "random": OI["yellow"], "susceptibility_shuffle": OI["purple"],
             "free_lora": OI["skyblue"], "full_ft": OI["black"]}
ARM_LABEL = {"susceptibility": "susceptibility (Q)", "energy": "energy (Σₓ)", "write": "write (U_wr)",
             "random": "random", "susceptibility_shuffle": "shuffle (ordering null)",
             "free_lora": "free-LoRA", "full_ft": "full-FT"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc-s3", default="")
    ap.add_argument("--conf-s3", default="")
    ap.add_argument("--disc-metrics", default="")
    ap.add_argument("--conf-metrics", default="")
    ap.add_argument("--banks-metrics", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from infra.aws_pg.jobs import _stencil_shared as SS

    def load(s3):
        if not s3:
            return None
        p = SS.fetch(f"{SS.S3_BUCKET}/{s3}", f"/tmp/stencil/{pathlib.Path(s3).name}")
        return json.loads(pathlib.Path(p).read_text())

    disc = load(a.disc_s3)
    conf = load(a.conf_s3)
    disc_m = load(a.disc_metrics)
    conf_m = load(a.conf_metrics)
    banks_m = load(a.banks_metrics)
    rep = {"object": "stencil_figures_v1", "panels": [], "skipped": [], "gpu_seconds": {"total": 0.0}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    def _save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140)
        plt.close(fig)
        rep["panels"].append(name)
        _w()

    def _foot(ax, prov):
        ax.figure.text(0.01, 0.005, prov, fontsize=6, color="#666666")

    frontier = (disc_m or {}).get("frontier") if disc_m else None
    ranks = (disc_m or {}).get("ranks") if disc_m else None

    # --- panel 1: HEADLINE frontier (task-gain vs forgetting) ------------------------------
    if frontier:
        fig, ax = plt.subplots(figsize=(6, 5))
        for arm, per in frontier.items():
            xs = [per[str(r)]["forget"] if str(r) in per else per.get(r, {}).get("forget") for r in (ranks or [])]
            ys = [per[str(r)]["gain"] if str(r) in per else per.get(r, {}).get("gain") for r in (ranks or [])]
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2 = zip(*pts)
            marker = "*" if arm == "full_ft" else "o"
            ax.plot(xs2, ys2, marker=marker, color=ARM_COLOR.get(arm, "#888"),
                    label=ARM_LABEL.get(arm, arm), lw=1.6, ms=9 if arm == "full_ft" else 5)
        ax.set_xlabel("forgetting  (composite battery drop) →  worse")
        ax.set_ylabel("task-gain  (held-out subject acc) →  better")
        ax.set_title("STENCIL headline: adaptation frontier per arm\n(up-and-left is better; if Q sits above-left of Σₓ, the map prescribes)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        _foot(ax, f"disc={a.disc_metrics}  |  DISCOVERY frontier (points=ranks)")
        _save(fig, "fig1_headline_frontier")
    else:
        rep["skipped"].append("fig1_headline_frontier (no frontier)")

    # --- panels 2 & 3: rank-efficiency + collateral ----------------------------------------
    for key, ylab, name, title in (
            ("gain", "task-gain", "fig2_rank_efficiency", "Rank-efficiency: task-gain vs rank"),
            ("forget", "forgetting", "fig3_collateral", "Collateral: forgetting vs rank")):
        if not (frontier and ranks):
            rep["skipped"].append(f"{name} (no frontier)")
            continue
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for arm, per in frontier.items():
            ys = [per.get(str(r), per.get(r, {})).get(key) for r in ranks]
            pts = [(r, y) for r, y in zip(ranks, ys) if y is not None]
            if not pts:
                continue
            rs, ys2 = zip(*pts)
            ax.plot(rs, ys2, marker="o", color=ARM_COLOR.get(arm, "#888"), label=ARM_LABEL.get(arm, arm), lw=1.5)
        ax.set_xscale("log"); ax.set_xlabel("adapter rank r (log)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.25)
        _foot(ax, f"disc={a.disc_metrics}")
        _save(fig, name)

    # --- panel 6: learning curves at r* (from confirmation trajectories) --------------------
    conf_cells = (conf_m or {}).get("cells") if conf_m else None
    if conf_cells:
        fig, (axg, axf) = plt.subplots(1, 2, figsize=(10, 4))
        seen = set()
        for c in conf_cells:
            arm = c["arm"]
            traj = c.get("trajectory") or []
            if not traj:
                continue
            steps = [t["step"] for t in traj]
            lbl = ARM_LABEL.get(arm, arm) if arm not in seen else None
            seen.add(arm)
            axg.plot(steps, [t["task_gain"] for t in traj], color=ARM_COLOR.get(arm, "#888"), alpha=0.6, label=lbl)
            axf.plot(steps, [t["forgetting"] for t in traj], color=ARM_COLOR.get(arm, "#888"), alpha=0.6)
        tau = (disc or {}).get("tau_star")
        if tau is not None:
            axg.axhline(tau, ls="--", color="#444", lw=1, label=f"τ*={tau:.3f}")
        axg.set_xlabel("training step"); axg.set_ylabel("task-gain"); axg.set_title("Task-gain vs step (r*)")
        axf.set_xlabel("training step"); axf.set_ylabel("forgetting"); axf.set_title("Forgetting vs step (r*)")
        axg.legend(fontsize=6); axg.grid(alpha=0.25); axf.grid(alpha=0.25)
        _foot(axg, f"conf={a.conf_metrics}  |  raw trajectories behind the iso-gain interpolation (T5)")
        _save(fig, "fig6_learning_curves")
    else:
        rep["skipped"].append("fig6_learning_curves (no confirmation cells)")

    # --- panel 8: cross-task replication (per-subject E_adapt) ------------------------------
    per_sub = (conf or {}).get("per_subject_E_adapt") if conf else None
    if per_sub:
        subs = [s for s in per_sub if per_sub[s].get("E") is not None]
        if subs:
            fig, ax = plt.subplots(figsize=(6, 4))
            vals = [per_sub[s]["E"] for s in subs]
            ax.barh(range(len(subs)), vals, color=[OI["vermillion"] if v > 0 else OI["blue"] for v in vals])
            ax.axvline(0, color="#000", lw=1)
            ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs, fontsize=7)
            ax.set_xlabel("E_adapt = forget(energy) − forget(susceptibility)   (>0 ⇒ map forgets less)")
            E = (conf or {}).get("E_adapt", {})
            ax.set_title(f"Cross-task replication of E_adapt\npooled E={E.get('E')}, CI=[{E.get('ci_lo')}, {E.get('ci_hi')}]")
            ax.grid(alpha=0.25, axis="x")
            _foot(ax, f"conf={a.conf_s3}  |  verdict: {(conf or {}).get('verdict', {}).get('verdict')}")
            _save(fig, "fig8_cross_task_replication")
    else:
        rep["skipped"].append("fig8_cross_task_replication (no per-subject E_adapt)")

    # --- panel 5b: THE MID-STACK CONTRAST MAP (headline first-class finding) ----------------
    pl = (banks_m or {}).get("per_layer") if banks_m else None
    if pl:
        layers = sorted(int(k) for k in pl)
        ov = [pl[str(L)].get("contrast_overlap", {}).get("50") for L in layers]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [OI["vermillion"] if (o is not None and o < 0.1) else
                  (OI["yellow"] if (o is not None and o < 0.4) else OI["skyblue"]) for o in ov]
        ax.bar(layers, [o if o is not None else 0 for o in ov], color=colors)
        ax.axhline(0.1, ls="--", color="#444", lw=1, label="strong-contrast threshold (0.1)")
        ax.set_xlabel("decoder layer"); ax.set_ylabel("subspace_overlap(U_sus, U_energy) @ r=50")
        ax.set_title("STENCIL: WHERE Q⊥energy holds — mid-stack + L0 separability\n"
                     "(low = susceptibility subspace distinct from energy; high = they coincide)")
        ax.set_xticks(layers); ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")
        _foot(ax, f"banks={a.banks_metrics}  |  0=orthogonal, 1=identical span")
        _save(fig, "fig5b_contrast_map")
    else:
        rep["skipped"].append("fig5b_contrast_map (no banks metrics)")

    # --- panel 7b: arm task-gain (the R2 mechanism: Q is the worst adapt axis) ---------------
    if conf_cells:
        import numpy as np
        by = {}
        for c in conf_cells:
            by.setdefault(c["arm"], []).append(max(c["gains"]) if c.get("gains") else None)
        arms_o = [a2 for a2 in ("susceptibility", "energy", "random") if a2 in by]
        if arms_o:
            fig, ax = plt.subplots(figsize=(5, 4))
            means = [np.nanmean([x for x in by[a2] if x is not None]) for a2 in arms_o]
            ax.bar(range(len(arms_o)), means, color=[ARM_COLOR[a2] for a2 in arms_o])
            tau = (disc or {}).get("tau_star")
            if tau is not None:
                ax.axhline(tau, ls="--", color="#000", lw=1, label=f"τ*={tau:.3f}")
            ax.set_xticks(range(len(arms_o))); ax.set_xticklabels([ARM_LABEL[a2] for a2 in arms_o], fontsize=8)
            ax.set_ylabel("mean max task-gain"); ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")
            ax.set_title("Q is the WORST adaptation axis\n(susceptibility < energy < random; scale-matched)")
            _foot(ax, f"conf={a.conf_metrics}  |  high-Q ≠ task-gradient-aligned")
            _save(fig, "fig7b_arm_task_gain")

    rep["green"] = True
    _w()
    print("STENCIL_FIGURES_OK", json.dumps({"panels": rep["panels"], "skipped": rep["skipped"]}), flush=True)


if __name__ == "__main__":
    main()
