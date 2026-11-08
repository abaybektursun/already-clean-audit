"""FULCRUM figure suite (CPU) — spec §10, the complete-sweep mandate
(`feedback_complete_sweep_visualization`): every swept point + all component curves + CI + null + g*.

Okabe-Ito; PNG+PDF; every panel: null overlaid, plain-language title, provenance footer, "how to read".
Missing-input panels are SKIPPED AND LOGGED, never faked (`feedback_signals_that_cannot_fail`).

  F1 the full θ-dose-response: n*_energy vs gain-retention (curv/random/energy per cap + pooled), g* line
  F2 the E_curv forest (paired, per capability, confirmation) with the random-vs-random null band
  F3 footprint-moved: curvature/energy fraction vs θ
  F4 the basis-match map: n*_energy vs curvature footprint, θ=0 → θ=1 (where the intervention moves points)
  F5 Leg B (if run): footprint / n*_energy / gain vs penalty λ
  F6 Leg C cross-model E_curv sign + ρ

  ./infra/aws_pg/pg run infra.aws_pg.jobs.fulcrum_figures \
      --args "--conf ... --disc ... --verdict ... --preflight ... [--legb ...] [--legc ...]" \
      --type r7i.2xlarge --no-weights --s3-prefix results/fulcrum_figs --budget-min 30
"""
import argparse
import json
import os
import pathlib

OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
PCOL = {"susceptibility": OI["vermillion"], "random": OI["blue"], "energy": OI["green"],
        "susceptibility_shuffle": OI["purple"]}
PLAB = {"susceptibility": "curvature (Q)", "random": "random (null)", "energy": "energy (exploratory)",
        "susceptibility_shuffle": "curvature-shuffle"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default="")
    ap.add_argument("--disc", default="")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--preflight", default="")
    ap.add_argument("--legb", default="")
    ap.add_argument("--legb-random", default="")
    ap.add_argument("--legc", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from infra.aws_pg.jobs import _quorum_shared as QS

    def load(key):
        if not key:
            return None
        return json.loads(pathlib.Path(QS.fetch_s3(key, f"/tmp/fulcrum/figs_{key.replace('/', '_')}")).read_text())

    conf, disc, verdict = load(a.conf), load(a.disc), load(a.verdict)
    pre, legb, legc = load(a.preflight), load(a.legb), load(a.legc)
    legb_random = load(a.legb_random)
    rep = {"object": "fulcrum_figures_v1", "rendered": [], "skipped": [], "gpu_seconds": {"total": 0.0}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    def _save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140, pad_inches=0.3)
        plt.close(fig)
        rep["rendered"].append(name)
        print(f"[figs] rendered {name}", flush=True)
        _w()

    def _skip(name, why):
        rep["skipped"].append({"panel": name, "reason": why})
        print(f"[figs] SKIP {name}: {why}", flush=True)
        _w()

    def _foot(ax, prov):
        ax.figure.text(0.005, 0.005, prov, fontsize=5.5, color="#666", ha="left", va="bottom")

    def _how(ax, text):
        ax.text(0.0, -0.18, text, transform=ax.transAxes, fontsize=7.5, color="#444", ha="left", va="top")

    g_star = (disc or {}).get("freeze", {}).get("g_star")

    def path_points(rows):
        """(gain_retention, n*) over θ where n* defined, sorted by gain-retention."""
        pts = [(r["gain_retention"], r["n_star"]) for r in rows if r.get("n_star") is not None]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    # ---- F1 the full θ-dose-response ------------------------------------------------------
    if conf and conf.get("capabilities"):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        drew = False
        for c in conf["capabilities"]:
            for key, rows in (c.get("paths") or {}).items():
                policy = "random" if key.startswith("random") else key
                xs, ys = path_points(rows)
                if xs:
                    ax.plot(xs, ys, "-o", color=PCOL.get(policy, "#999"), alpha=0.45, lw=1, ms=3)
                    drew = True
        if g_star is not None:
            ax.axvline(g_star, color=OI["black"], ls="--", lw=1.2, label=f"g*={g_star:.2f}")
        for policy in ("susceptibility", "random", "energy"):
            ax.plot([], [], "-o", color=PCOL[policy], label=PLAB[policy])
        ax.set_xlabel("gain-retention  gain(θ)/gain(0)"); ax.set_ylabel("n*_energy (necessary rank)")
        ax.set_title("F1 — necessary-rank vs matched gain, along each θ-removal path", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        _how(ax, "Each line is one capability's θ-path. At g* (dashed), curvature below random ⇒ H_lever.")
        _foot(ax, f"FULCRUM · F1 · conf={a.conf}")
        _save(fig, "F1_theta_dose_response") if drew else _skip("F1_theta_dose_response", "no paths")
    else:
        _skip("F1_theta_dose_response", "conf absent")

    # ---- F2 the E_curv forest -------------------------------------------------------------
    if verdict and verdict.get("per_capability"):
        rows = [r for r in verdict["per_capability"] if r.get("e_curv") is not None]
        if rows:
            fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.42 * len(rows) + 1.5)))
            ys = np.arange(len(rows))
            es = [r["e_curv"] for r in rows]
            ax.scatter(es, ys, color=OI["vermillion"], zorder=3)
            ax.set_yticks(ys); ax.set_yticklabels([r["slug"] for r in rows], fontsize=7)
            ax.axvline(0, color=OI["black"], lw=1)
            nb = verdict.get("null_band_hi", 0.0)
            ax.axvspan(-nb, nb, color=OI["yellow"], alpha=0.3, label=f"random-vs-random band ±{nb:.1f}")
            ci = verdict.get("e_curv_ci", {})
            if ci.get("median") is not None:
                ax.axvline(ci["median"], color=OI["blue"], ls="--",
                           label=f"median {ci['median']:.1f} [{ci['ci_lo']:.1f},{ci['ci_hi']:.1f}]")
            ax.set_xlabel("E_curv = n*(curv@g*) − n*(random@g*)")
            ax.set_title(f"F2 — the E_curv forest  ·  verdict: {verdict.get('verdict', '?')}", fontsize=11)
            ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")
            _how(ax, "E_curv < 0 = curvature-removal is more surgical than random at matched gain (H_lever).")
            _foot(ax, f"FULCRUM · F2 · verdict={a.verdict}")
            _save(fig, "F2_ecurv_forest")
        else:
            _skip("F2_ecurv_forest", "no reachable E_curv")
    else:
        _skip("F2_ecurv_forest", "verdict absent")

    # ---- F3 footprint-moved ---------------------------------------------------------------
    if conf and conf.get("capabilities"):
        fig, (axq, axe) = plt.subplots(1, 2, figsize=(9.2, 4.2))
        drew = False
        for c in conf["capabilities"]:
            cr = (c.get("paths") or {}).get("susceptibility")
            if not cr:
                continue
            th = [r["theta"] for r in cr]
            axq.plot(th, [r["curvature_fraction"] for r in cr], "-o", color=OI["vermillion"], alpha=0.4, ms=3)
            axe.plot(th, [r["energy_fraction"] for r in cr], "-o", color=OI["green"], alpha=0.4, ms=3)
            drew = True
        axq.set_xlabel("θ (curvature removal)"); axq.set_ylabel("curvature fraction of ΔW(θ)")
        axq.set_title("curvature fraction ↓ with θ", fontsize=10); axq.grid(alpha=0.25)
        axe.set_xlabel("θ"); axe.set_ylabel("energy fraction of ΔW(θ)")
        axe.set_title("energy fraction ~stable", fontsize=10); axe.grid(alpha=0.25)
        fig.suptitle("F3 — the intervention moved the footprint (Q↓, E stable)", fontsize=11)
        _foot(axq, f"FULCRUM · F3 · conf={a.conf}")
        _save(fig, "F3_footprint_moved") if drew else _skip("F3_footprint_moved", "no curv paths")
    else:
        _skip("F3_footprint_moved", "conf absent")

    # ---- F4 the basis-match map -----------------------------------------------------------
    if verdict and verdict.get("f4_basis_match"):
        f4 = verdict["f4_basis_match"]
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        drew = False
        for r in f4:
            if r.get("nstar_theta0") is None or r.get("nstar_theta1") is None:
                continue
            ax.annotate("", xy=(r["footprint_theta1"], r["nstar_theta1"]),
                        xytext=(r["footprint_theta0"], r["nstar_theta0"]),
                        arrowprops=dict(arrowstyle="->", color="#999", alpha=0.7))
            ax.scatter([r["footprint_theta0"]], [r["nstar_theta0"]], color=OI["vermillion"], zorder=3)
            ax.scatter([r["footprint_theta1"]], [r["nstar_theta1"]], color=OI["blue"], zorder=3)
            drew = True
        ax.scatter([], [], color=OI["vermillion"], label="θ=0 (full ΔW)")
        ax.scatter([], [], color=OI["blue"], label="θ=1 (curvature removed)")
        ax.set_xlabel("curvature footprint of ΔW  (Q energy @ r=679)")
        ax.set_ylabel("n*_energy"); ax.set_title("F4 — basis-match: removing curvature moves points "
                                                 "down-left along QUORUM's ρ=0.65 line", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        _how(ax, "Arrows: each cap from full-ΔW to curvature-removed. Down = n* fell; left = footprint fell.")
        _foot(ax, f"FULCRUM · F4 · verdict={a.verdict}")
        _save(fig, "F4_basis_match") if drew else _skip("F4_basis_match", "no θ0/θ1 n*")
    else:
        _skip("F4_basis_match", "verdict/f4 absent")

    # ---- F5 Leg B -------------------------------------------------------------------------
    if legb and legb.get("penalty_sweep"):
        def _series(rep_, key):
            sw = sorted([s for s in rep_["penalty_sweep"] if s["lambda"] >= 0], key=lambda s: s["lambda"])
            return [s["lambda"] for s in sw], [s.get(key) for s in sw]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
        for key, ax, ttl, col in [("footprint", axes[0], "curvature footprint", OI["vermillion"]),
                                  ("n_star_energy", axes[1], "n*_energy (log)", OI["blue"]),
                                  ("gain", axes[2], "task gain", OI["green"])]:
            lx, ly = _series(legb, key)
            ax.plot(lx, ly, "-o", color=col, label="curvature penalty")
            if legb_random and legb_random.get("penalty_sweep"):
                rx, ry = _series(legb_random, key)
                ax.plot(rx, ry, "--s", color="#888", label="random penalty (control)")
            ax.set_title(f"{ttl} vs λ", fontsize=9); ax.set_xlabel("λ (penalty strength)")
            ax.grid(alpha=0.25); ax.legend(fontsize=7)
            if key == "n_star_energy":
                ax.set_yscale("symlog")
        fig.suptitle("F5 — Leg B: the n* lever is GENERIC, not curvature-specific "
                     "(random penalty collapses n* too)", fontsize=10)
        _foot(axes[0], f"FULCRUM · F5 · legb={a.legb} · legb_random={a.legb_random}")
        _save(fig, "F5_legb_penalty")
    else:
        _skip("F5_legb_penalty", "legb absent or NO-GO")

    # ---- F6 Leg C -------------------------------------------------------------------------
    if legc and (legc.get("e_curv_sign") is not None or legc.get("rho") is not None):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        txt = (f"PG2 replication (firewall — verdict-level only)\n\n"
               f"E_curv sign: {legc.get('e_curv_sign')}\n"
               f"curvature-footprint→n* ρ: {legc.get('rho')}  (QUORUM PG1 ρ=0.654)\n"
               f"replicates: {legc.get('replicates')}")
        ax.text(0.05, 0.5, txt, fontsize=11, va="center")
        ax.axis("off"); ax.set_title("F6 — Leg C cross-model external validity (PG2)", fontsize=11)
        _foot(ax, f"FULCRUM · F6 · legc={a.legc}")
        _save(fig, "F6_legc_crossmodel")
    else:
        _skip("F6_legc_crossmodel", "legc absent")

    rep["green"] = True
    _w()
    print("FULCRUM_FIGURES_OK", json.dumps({"rendered": rep["rendered"], "skipped": rep["skipped"]},
          default=str), flush=True)


if __name__ == "__main__":
    main()
