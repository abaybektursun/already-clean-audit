"""CLEAVE figure suite (CPU) — spec §10. The frontier IS the verdict, so the panels carry it.

Colourblind-safe Okabe-Ito palette; PNG+PDF; every panel gets a plain-language title, a "how to read"
line, the null overlaid, and a provenance footer. Robust to partial / VOID landings — a panel with no
data is SKIPPED AND LOGGED, never faked (`feedback_signals_that_cannot_fail`), and the skip list is
returned in the report.

  1  HEADLINE — the protective frontier: task-retention vs forgetting per policy; (r*, ρ*) marked.
                Up-and-LEFT (keep the task, shed the forgetting) is better. If off-sus sits below
                off-random at matched retention, the map protects.
  2  the cleave — forgetting and task-retention carried by ΔW_in vs ΔW_off (the decomposition).
  3  targeting frontier — iso-forgetting task-gain, into-sus vs into-energy vs random (STENCIL redone).
  4  task-gradient census — where the task gradient's mass lives, against the Haar floor (MECHANISM).
  5  operator resolution certification — per-layer stability vs rank (STENCIL's T1, closed and shown).
  6  per-layer Q⊥energy contrast — where the map is a distinct object at all (STENCIL R3 replicated).
  7  post-hoc vs trained — removable vs learnable-in-the-complement, side by side.
  8  cross-task replication — per-subject E_protect with CIs.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_figures \
      --args "--disc results/cleave_disc/out/metrics.json --conf results/cleave_conf/out/metrics.json \
              --qcert results/cleave_qcert/out/metrics.json --mech results/cleave_mech/out/metrics.json \
              --target results/cleave_target/out/metrics.json --trained results/cleave_trained/out/metrics.json" \
      --type r7i.2xlarge --no-weights --s3-prefix results/cleave_figs --budget-min 30
"""
import argparse
import json
import os
import pathlib

OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
POL_COLOR = {"susceptibility": OI["vermillion"], "energy": OI["blue"], "random": OI["yellow"],
             "susceptibility_shuffle": OI["purple"]}
POL_LABEL = {"susceptibility": "off-susceptibility (the map)", "energy": "off-energy (Σₓ)",
             "random": "off-random (the null)", "susceptibility_shuffle": "off-shuffle (ordering null)"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc", default="")
    ap.add_argument("--conf", default="")
    ap.add_argument("--qcert", default="")
    ap.add_argument("--mech", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--trained", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from infra.aws_pg.jobs import _cleave_shared as CLS

    def load(s3):
        if not s3:
            return None
        p = CLS.fetch_s3(s3, f"/tmp/cleave/figs/{s3.replace('/', '_')}")
        return json.loads(pathlib.Path(p).read_text())

    disc, conf = load(a.disc), load(a.conf)
    qcert, mech = load(a.qcert), load(a.mech)
    target, trained = load(a.target), load(a.trained)
    rep = {"object": "cleave_figures_v1", "rendered": [], "skipped": [], "gpu_seconds": {"total": 0.0}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    def _save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140, pad_inches=0.35)
        plt.close(fig)
        rep["rendered"].append(name)
        print(f"[figs] rendered {name}", flush=True)
        _w()

    def _skip(name, why):
        rep["skipped"].append({"panel": name, "reason": why})
        print(f"[figs] SKIP {name}: {why}", flush=True)
        _w()

    def _foot(ax, prov):
        ax.figure.text(0.005, 0.005, prov, fontsize=5.5, color="#666666", ha="left", va="bottom")

    def _how(ax, text):
        # below the x-label: at y=1.015 it collided with the title and both became unreadable
        ax.text(0.0, -0.185, text, transform=ax.transAxes, fontsize=7.5, color="#444444",
                ha="left", va="top", wrap=True)

    # ---- 1. HEADLINE: the protective frontier ------------------------------------------
    src = conf or disc
    if src and src.get("frontiers"):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for pol, fr in src["frontiers"].items():
            fr = sorted(fr, key=lambda p: -p["retention"])
            ax.plot([p["retention"] for p in fr], [p["forgetting"] for p in fr], "o-",
                    color=POL_COLOR.get(pol, "#888888"), label=POL_LABEL.get(pol, pol),
                    lw=2 if pol == "susceptibility" else 1.4, ms=4,
                    zorder=3 if pol == "susceptibility" else 2)
        fz = (disc or {}).get("freeze") or src.get("frozen")
        if fz and fz.get("rho_star") is not None:
            ax.axvline(float(fz["rho_star"]), color=OI["black"], ls="--", lw=1, alpha=0.7)
            ax.text(float(fz["rho_star"]), ax.get_ylim()[1], f"  ρ*={float(fz['rho_star']):.3f}",
                    fontsize=7, va="top")
        nf = src.get("natural_forgetting_mean")
        if nf is not None:
            ax.axhline(nf, color=OI["black"], ls=":", lw=1, alpha=0.6)
            ax.text(ax.get_xlim()[0], nf, " natural fine-tune (nothing removed)", fontsize=7, va="bottom")
        ax.set_xlabel("task-retention  gain(ΔW_off) / gain(ΔW_full)   →  keeps more task")
        ax.set_ylabel("multimodal forgetting  (VQAv2 + COCO-CIDEr)")
        ax.set_title("Does removing the map's directions keep the task and shed the forgetting?",
                     fontsize=11)
        _how(ax, "How to read: each curve sweeps removal STRENGTH alpha (0=nothing removed, "
                 "1=the map's component fully removed) at the certified r*. DOWN-and-RIGHT is better. "
                 "If red sits below yellow at the same retention, the map protects.")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        _foot(ax, f"CLEAVE · {src.get('object')} · phase={src.get('phase')} · sha={src.get('pinned_sha')}")
        _save(fig, "fig1_protective_frontier")
    else:
        _skip("fig1_protective_frontier", "no frontiers in disc/conf")

    # ---- 2. the cleave: what does each part of ΔW carry? -------------------------------
    if src and src.get("frontiers") and src.get("natural_forgetting_mean") is not None:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        pols = [p for p in ("susceptibility", "energy", "random") if p in src["frontiers"]]
        if pols:
            fz = (disc or {}).get("freeze") or src.get("frozen") or {}
            rstar = fz.get("r_star")
            x = np.arange(len(pols))
            ret, forg = [], []
            for p in pols:
                fr = src["frontiers"][p]
                # alpha=1 = the map's component FULLY removed (A-CLEAVE-12's sweep axis)
                pt = max(fr, key=lambda q: q.get("alpha", 0.0))
                ret.append(pt["retention"])
                forg.append(pt["forgetting"])
            ax.bar(x - 0.2, ret, 0.4, label="task-retention kept", color=OI["green"])
            ax.bar(x + 0.2, forg, 0.4, label="forgetting remaining", color=OI["vermillion"])
            ax.axhline(src["natural_forgetting_mean"], color=OI["black"], ls=":", lw=1,
                       label="natural-FT forgetting")
            ax.set_xticks(x)
            ax.set_xticklabels([POL_LABEL.get(p, p) for p in pols], fontsize=8)
            ax.set_title(f"What FULL removal costs and buys (alpha=1, r*={rstar})", fontsize=11)
            _how(ax, "How to read: tall green + short red = the ideal cleave (task kept, forgetting gone).")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25, axis="y")
            _foot(ax, f"CLEAVE · {src.get('object')} · r*={rstar}")
            _save(fig, "fig2_the_cleave")
        else:
            _skip("fig2_the_cleave", "no policies in frontiers")
    else:
        _skip("fig2_the_cleave", "no frontier/natural data")

    # ---- 3. targeting frontier (STENCIL redone) ----------------------------------------
    if target and target.get("iso_forgetting"):
        rows = [(int(r), v) for r, v in target["iso_forgetting"].items() if not v.get("insufficient")]
        if rows:
            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            for arm, col in (("susceptibility", OI["vermillion"]), ("energy", OI["blue"]),
                             ("random", OI["yellow"])):
                xs, ys = [], []
                for r, v in sorted(rows):
                    vals = [x for x in (v.get("gain_at_phi", {}).get(arm) or {}).values() if x is not None]
                    if vals:
                        xs.append(r)
                        ys.append(float(np.mean(vals)))
                if xs:
                    ax.plot(xs, ys, "o-", color=col, label=f"into-{arm}", lw=1.6, ms=5)
            ax.set_xscale("log")
            ax.set_xlabel("adapter rank")
            ax.set_ylabel("task-gain at iso-forgetting φ*")
            ax.set_title("Which subspace carries the task? (writing INTO each)", fontsize=11)
            _how(ax, "How to read: higher = learns more at the SAME collateral. STENCIL R2 predicts "
                     "susceptibility lowest.")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            _foot(ax, f"CLEAVE · target · certified_max={target.get('certification', {}).get('certified_max_min_over_layers')}")
            _save(fig, "fig3_targeting_frontier")
        else:
            _skip("fig3_targeting_frontier", "no iso-forgetting rows")
    else:
        _skip("fig3_targeting_frontier", "target artifact absent")

    # ---- 4. task-gradient census (MECHANISM) -------------------------------------------
    if mech and mech.get("mechanism"):
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ranks = sorted(int(r) for r in mech["mechanism"])
        sus = [mech["mechanism"][str(r) if str(r) in mech["mechanism"] else r]["task_gradient_in_susceptibility"] for r in ranks]
        en = [mech["mechanism"][str(r) if str(r) in mech["mechanism"] else r]["task_gradient_in_energy"] for r in ranks]
        rnd = [mech["mechanism"][str(r) if str(r) in mech["mechanism"] else r]["task_gradient_in_random"] for r in ranks]
        fl = [mech["mechanism"][str(r) if str(r) in mech["mechanism"] else r]["chance_floor"] for r in ranks]
        x = np.arange(len(ranks))
        ax.bar(x - 0.25, sus, 0.25, label="in U_sus (the map)", color=OI["vermillion"])
        ax.bar(x, en, 0.25, label="in U_energy (Σₓ)", color=OI["blue"])
        ax.bar(x + 0.25, rnd, 0.25, label="in random", color=OI["yellow"])
        for xi, f in zip(x, fl):          # a segment spanning the whole group reads as a floor
            ax.hlines(f, xi - 0.42, xi + 0.42, color=OI["black"], lw=1.4, zorder=4)
        ax.hlines([], [], [], color=OI["black"], lw=1.4, label="Haar chance floor r/d")
        for xi, e, f in zip(x, en, fl):   # the number that carries the result
            ax.annotate(f"{e / f:.0f}×", (xi, e), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=OI["blue"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("subspace rank r")
        ax.set_ylabel("fraction of task-gradient energy captured")
        ax.set_title("Where does the task gradient actually live? (measured, not inferred)", fontsize=11)
        _how(ax, "How to read: bars above the floor = the subspace captures real gradient mass. The "
                 "unifying hypothesis-to-kill predicts blue > red.")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        _foot(ax, f"CLEAVE · mechanism · confirms_unifying={mech.get('mechanism_confirms')}")
        _save(fig, "fig4_task_gradient_census")
    else:
        _skip("fig4_task_gradient_census", "mechanism artifact absent")

    # ---- 5. operator resolution certification (STENCIL's T1, shown) ---------------------
    if qcert and qcert.get("per_layer"):
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        pl = qcert["per_layer"]
        layers = sorted(int(L) for L in pl)
        ranks = sorted(int(r) for r in pl[str(layers[0]) if str(layers[0]) in pl else layers[0]]
                       ["certification"]["per_rank"])
        n_dead = 0
        for L in layers:
            cert = pl[str(L) if str(L) in pl else L]["certification"]
            per = cert["per_rank"]
            ys = [per[str(r) if str(r) in per else r]["mean_overlap"] for r in ranks]
            dead = not cert.get("certified_ranks")
            n_dead += int(dead)
            # layers that certify at NO rank are the headline — draw them red, not buried in viridis
            ax.plot(ranks, ys, "-", alpha=0.85 if dead else 0.45,
                    lw=1.8 if dead else 1.0,
                    color=OI["vermillion"] if dead else plt.cm.viridis(L / max(layers, default=1)),
                    zorder=3 if dead else 2)
        ax.plot([], [], "-", color=OI["vermillion"], lw=1.8,
                label=f"certifies at NO rank ({n_dead}/{len(layers)} layers)")
        ax.plot(ranks, [r / 2048 for r in ranks], "k--", lw=1.2, label="Haar floor r/d (chance)")
        ax.axhline(qcert.get("cert_tol", 0.9), color=OI["vermillion"], ls=":", lw=1.5,
                   label=f"certification floor ({qcert.get('cert_tol', 0.9)})")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("rank r")
        ax.set_ylabel("mean pairwise subspace overlap across independent Q measurements")
        ax.set_title("Is U_sus the map, or is it noise? (one line per layer)", fontsize=11)
        _how(ax, "How to read: a line above the red floor = U_sus reproduces across independent "
                 "measurements at that rank. Near the dashed line = indistinguishable from chance.")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        _foot(ax, f"CLEAVE · qcert · repeats={qcert.get('repeats')} · {qcert.get('cert_scope', '')[:90]}")
        _save(fig, "fig5_resolution_certification")
    else:
        _skip("fig5_resolution_certification", "qcert artifact absent")

    # ---- 6. per-layer Q ⊥ energy contrast (STENCIL R3 replicated) -----------------------
    if qcert and qcert.get("per_layer"):
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        pl = qcert["per_layer"]
        layers = sorted(int(L) for L in pl)
        key = "contrast_overlap_sus_vs_energy"
        first = pl[str(layers[0]) if str(layers[0]) in pl else layers[0]]
        if key in first:
            ranks = sorted(int(r) for r in first[key])
            for r in ranks:
                ys = [pl[str(L) if str(L) in pl else L][key][str(r) if str(r) in
                      pl[str(L) if str(L) in pl else L][key] else r] for L in layers]
                ax.plot(layers, ys, "o-", ms=3, lw=1.2, label=f"r={r}")
            ax.set_xlabel("decoder layer")
            ax.set_ylabel("subspace overlap(U_sus, U_energy)")
            ax.set_title("Where is the map a DISTINCT object from energy? (0 = orthogonal)", fontsize=11)
            _how(ax, "How to read: low = Q and Σₓ genuinely differ there, so 'adapt in Q' and 'adapt "
                     "in energy' are different interventions. STENCIL R3: L0 + mid-stack ≈ 0.")
            ax.legend(fontsize=7, ncol=4)
            ax.grid(alpha=0.25)
            _foot(ax, f"CLEAVE · qcert · sha={qcert.get('pinned_sha')}")
            _save(fig, "fig6_q_perp_energy_by_layer")
        else:
            _skip("fig6_q_perp_energy_by_layer", "no contrast table in qcert")
    else:
        _skip("fig6_q_perp_energy_by_layer", "qcert artifact absent")

    # ---- 7. post-hoc vs trained ---------------------------------------------------------
    if trained and trained.get("mean_by_policy"):
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        pols = [p for p in ("susceptibility", "energy", "random") if p in trained["mean_by_policy"]]
        x = np.arange(len(pols))
        tr_f = [trained["mean_by_policy"][p]["forgetting"] for p in pols]
        ax.bar(x - 0.2, tr_f, 0.4, label="trained OFF the subspace", color=OI["skyblue"])
        if src and src.get("frontiers"):
            fz = (disc or {}).get("freeze") or src.get("frozen") or {}
            rstar = fz.get("r_star") or trained.get("rank")
            ph = []
            for p in pols:
                fr = src["frontiers"].get(p)
                if not fr:
                    ph.append(np.nan)
                    continue
                pt = max(fr, key=lambda q: q.get("alpha", 0.0))   # alpha=1: full post-hoc removal
                ph.append(pt["forgetting"])
            ax.bar(x + 0.2, ph, 0.4, label="post-hoc projection", color=OI["orange"])
        ax.set_xticks(x)
        ax.set_xticklabels([POL_LABEL.get(p, p) for p in pols], fontsize=8)
        ax.set_ylabel("multimodal forgetting")
        ax.set_title("Removable after the fact vs learnable in the complement", fontsize=11)
        _how(ax, "How to read: agreement means the projection story is not an artifact of WHEN you "
                 "project. Disagreement means the optimizer compensates.")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        _foot(ax, f"CLEAVE · trained r={trained.get('rank')} · adamw_leak="
                  f"{(trained.get('adamw_inspan_leak') or {}).get('mean_over_cells')}")
        _save(fig, "fig7_posthoc_vs_trained")
    else:
        _skip("fig7_posthoc_vs_trained", "trained artifact absent")

    # ---- 8. cross-task replication ------------------------------------------------------
    if conf and conf.get("E_protect_by_subject"):
        by = conf["E_protect_by_subject"]
        subs = [s for s in sorted(by) if by[s].get("E") is not None]
        if subs:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            y = np.arange(len(subs))
            E = [by[s]["E"] for s in subs]
            lo = [by[s]["E"] - by[s]["ci_lo"] for s in subs]
            hi = [by[s]["ci_hi"] - by[s]["E"] for s in subs]
            ax.errorbar(E, y, xerr=[lo, hi], fmt="o", color=OI["vermillion"], capsize=3)
            ax.axvline(0, color=OI["black"], lw=1)
            ov = conf.get("E_protect") or {}
            if ov.get("E") is not None:
                ax.axvline(ov["E"], color=OI["blue"], ls="--", lw=1.2,
                           label=f"pooled E_protect={ov['E']:+.4f}")
                ax.legend(fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(subs, fontsize=8)
            ax.set_xlabel("E_protect = forgetting(off-random) − forgetting(off-sus)   →  map protects")
            ax.set_title("Does the result replicate across tasks?", fontsize=11)
            _how(ax, "How to read: points right of 0 = removing the map's directions beat removing a "
                     "random subspace on that task. CIs crossing 0 = no effect there.")
            ax.grid(alpha=0.25, axis="x")
            _foot(ax, f"CLEAVE · conf · n_subjects={len(subs)}")
            _save(fig, "fig8_cross_task_replication")
        else:
            _skip("fig8_cross_task_replication", "no per-subject E_protect with data")
    else:
        _skip("fig8_cross_task_replication", "conf artifact absent / no per-subject table")

    rep["green"] = True
    _w()
    print("CLEAVE_FIGURES_OK", json.dumps({"rendered": rep["rendered"], "skipped": rep["skipped"]},
                                          default=str), flush=True)


if __name__ == "__main__":
    main()
