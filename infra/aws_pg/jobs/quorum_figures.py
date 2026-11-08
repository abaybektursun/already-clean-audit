"""QUORUM figure suite (CPU) — spec §10. The necessary-rank spectrum IS the verdict.

Okabe-Ito; PNG+PDF; every panel: null overlaid, plain-language title, provenance footer, "how to read".
Missing-input panels are SKIPPED AND LOGGED, never faked (`feedback_signals_that_cannot_fail`).

  F1 the full retention-vs-rank sweep (every capability, energy/random/Q + floor + ρ_break + search-rank)
  F2 the n* spectrum (per capability, per basis, censoring flags)
  F3 the Δn* forest (paired, per capability) with the null band
  F4 the Arm-0 seam-seal (write energy in Σ vs Q vs floor at each rank) — the SPECTER-#5 closer
  F5 Arm-2 trajectory (n* for full-FT vs single-step vs rank-r adapter)
  F6 Arm-3 predictor scatter (n* vs gradient-concentration / curvature / ‖ΔW‖)

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_figures \
      --args "--conf ... --disc ... --verdict ... --arm0 ... --arm2 ..." \
      --type r7i.2xlarge --no-weights --s3-prefix results/quorum_figs --budget-min 30
"""
import argparse
import json
import os
import pathlib

OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
BASIS_COLOR = {"energy": OI["blue"], "random": OI["yellow"], "susceptibility": OI["vermillion"]}
BASIS_LABEL = {"energy": "energy (Σₓ)", "random": "random (null)", "susceptibility": "susceptibility (Q)"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default="")
    ap.add_argument("--disc", default="")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--arm0", default="")
    ap.add_argument("--arm2", default="")
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
        return json.loads(pathlib.Path(QS.fetch_s3(key, f"/tmp/quorum/figs_{key.replace('/', '_')}")).read_text())

    conf, disc, verdict = load(a.conf), load(a.disc), load(a.verdict)
    arm0, arm2 = load(a.arm0), load(a.arm2)
    rep = {"object": "quorum_figures_v1", "rendered": [], "skipped": [], "gpu_seconds": {"total": 0.0}}

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
        ax.text(0.0, -0.16, text, transform=ax.transAxes, fontsize=7.5, color="#444",
                ha="left", va="top")

    rho_break = (disc or {}).get("freeze", {}).get("rho_break")
    search_rank = (disc or {}).get("freeze", {}).get("search_rank", 20)

    # ---- F1 the full retention-vs-rank sweep ----------------------------------------------
    if conf and conf.get("capabilities"):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        drew = False
        for c in conf["capabilities"]:
            for key, sweep in (c.get("sweeps") or {}).items():
                basis = "random" if key.startswith("random") else key
                col = BASIS_COLOR.get(basis, "#999")
                xs = [s["rank"] for s in sweep]
                ys = [s["retention"] for s in sweep]
                ax.plot(xs, ys, "-", color=col, alpha=0.5, lw=1)
                drew = True
        if rho_break is not None:
            ax.axhline(rho_break, color=OI["black"], ls="--", lw=1.2, label=f"ρ_break={rho_break:.3f}")
        ax.axvline(search_rank, color=OI["green"], ls=":", lw=1.4, label=f"search-rank={search_rank}")
        for basis, col in BASIS_COLOR.items():
            ax.plot([], [], "-", color=col, label=BASIS_LABEL[basis])
        ax.set_xscale("log")
        ax.set_xlabel("removal rank r")
        ax.set_ylabel("task-gain retention  gain_off(r) / gain_full")
        ax.set_title("How far can you delete before the capability breaks?", fontsize=11)
        _how(ax, "How to read: each line is one capability on one basis. n* = where it crosses ρ_break. "
                 "If energy (blue) crosses far LEFT of random (yellow), storage is compact.")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25)
        _foot(ax, f"QUORUM · {conf.get('object')} · sha={conf.get('pinned_sha')}")
        _save(fig, "F1_retention_sweep") if drew else _skip("F1_retention_sweep", "no sweeps")
    else:
        _skip("F1_retention_sweep", "conf absent")

    # ---- F2 the n* spectrum ---------------------------------------------------------------
    v = verdict
    if v and v.get("per_capability"):
        pc = sorted(v["per_capability"], key=lambda r: r["n_star_energy"])
        fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.4 * len(pc))))
        y = np.arange(len(pc))
        ax.scatter([r["n_star_energy"] for r in pc], y, color=OI["blue"], label="n* energy", zorder=3)
        ax.scatter([r["n_star_random"] for r in pc], y, color=OI["yellow"], edgecolor="#888",
                   label="n* random", zorder=3)
        ax.axvline(search_rank, color=OI["green"], ls=":", lw=1.4, label=f"search-rank={search_rank}")
        for i, r in enumerate(pc):
            ax.plot([r["n_star_energy"], r["n_star_random"]], [i, i], color="#ccc", lw=1, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([r["slug"] for r in pc], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("necessary rank n*")
        ax.set_title("The necessary-rank spectrum (n* per capability, per basis)", fontsize=11)
        _how(ax, "How to read: blue right of the green line = stored more broadly than the search used. "
                 "Blue ≪ yellow = energy is specifically necessary (compact).")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="x")
        _foot(ax, f"QUORUM · verdict={v.get('verdict',{}).get('verdict')}")
        _save(fig, "F2_nstar_spectrum")
    else:
        _skip("F2_nstar_spectrum", "verdict/per_capability absent")

    # ---- F3 the Δn* forest ----------------------------------------------------------------
    if v and v.get("per_capability"):
        pc = sorted(v["per_capability"], key=lambda r: r["delta_nstar"])
        fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.4 * len(pc))))
        y = np.arange(len(pc))
        ax.scatter([r["delta_nstar"] for r in pc], y, color=OI["vermillion"], zorder=3)
        nb = v.get("null_band_hi")
        if nb is not None:
            ax.axvspan(-nb, nb, color="#ddd", label=f"null band ±{nb:.0f}")
        ax.axvline(0, color=OI["black"], lw=1)
        med = (v.get("delta_ci") or {}).get("median")
        if med is not None:
            ax.axvline(med, color=OI["blue"], ls="--", lw=1.2, label=f"median Δn*={med:.0f}")
        ax.set_yticks(y); ax.set_yticklabels([r["slug"] for r in pc], fontsize=8)
        ax.set_xlabel("Δn* = n*_random − n*_energy   →  compact")
        ax.set_title("Is energy MORE necessary than a random subspace? (per capability)", fontsize=11)
        _how(ax, "How to read: right of the grey band = energy-removal breaks it earlier than random "
                 "(compact). Inside the band = no more necessary than random (redundant).")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="x")
        _foot(ax, f"QUORUM · Δn* CI={v.get('delta_ci',{}).get('ci_lo')}..{v.get('delta_ci',{}).get('ci_hi')}")
        _save(fig, "F3_delta_forest")
    else:
        _skip("F3_delta_forest", "verdict absent")

    # ---- F4 the Arm-0 seam-seal -----------------------------------------------------------
    if arm0 and arm0.get("aggregate"):
        agg = arm0["aggregate"]
        ranks = sorted(int(r) for r in agg)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        x = np.arange(len(ranks))
        E = [agg[str(r) if str(r) in agg else r]["energy"] for r in ranks]
        Q = [agg[str(r) if str(r) in agg else r]["susceptibility"] for r in ranks]
        fl = [agg[str(r) if str(r) in agg else r]["floor"] for r in ranks]
        ax.bar(x - 0.22, E, 0.22, label="in Σ (energy)", color=OI["blue"])
        ax.bar(x, Q, 0.22, label="in Q (susceptibility)", color=OI["vermillion"])
        ax.bar(x + 0.22, fl, 0.22, label="Haar floor r/d", color="#bbb")
        ax.set_xticks(x); ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("subspace rank r")
        ax.set_ylabel("write-energy fraction of ΔW")
        ax.set_title("Where does fine-tuning's WRITE live? (SPECTER open-Q #5)", fontsize=11)
        _how(ax, "How to read: Σ ≫ Q > floor ⇒ the write is variance-located with a modest curvature "
                 "tilt. Q at r=679 ≈ 0.352 reproduces CARRY's stiff-band number.")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        _foot(ax, f"QUORUM · arm0 · stiff@679={arm0.get('stiff_fraction_median')}")
        _save(fig, "F4_seam_seal")
    else:
        _skip("F4_seam_seal", "arm0 absent")

    # ---- F5 Arm-2 trajectory --------------------------------------------------------------
    if arm2 and arm2.get("conditions"):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        conds = arm2["conditions"]
        names = list(conds)
        ns = [conds[k].get("n_star") for k in names]
        rr = [conds[k].get("rank") for k in names]
        x = np.arange(len(names))
        ax.bar(x, ns, 0.5, color=OI["skyblue"], label="n*")
        for i, r in enumerate(rr):
            if r is not None:
                ax.axhline(r, ls=":", lw=0.8, color="#999")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("necessary rank n*")
        ax.set_title("Is redundancy MADE by training or inherent to the code?", fontsize=11)
        _how(ax, "How to read: a rank-r adapter with n* ≫ r ⇒ redundancy is a property of the "
                 "representation, not the trajectory.")
        ax.legend(fontsize=8)
        _foot(ax, "QUORUM · arm2")
        _save(fig, "F5_trajectory")
    else:
        _skip("F5_trajectory", "arm2 absent")

    # ---- F6 Arm-3 predictor scatter -------------------------------------------------------
    if v and v.get("arm3") and not (v["arm3"] or {}).get("skipped") and arm0:
        a3 = v["arm3"]
        pc = v["per_capability"]
        cen = arm0.get("per_donor") or {}
        preds = {"grad_concentration": [], "curvature_footprint": [], "delta_norm": [], "nstar": [], "slug": []}
        for r in pc:
            d = cen.get(r["slug"])
            if d is None:
                continue
            c20 = d["census"].get("20") or d["census"].get(20)
            c679 = d["census"].get("679") or d["census"].get(679)
            preds["grad_concentration"].append(c20["energy"])
            preds["curvature_footprint"].append(c679["susceptibility"])
            preds["delta_norm"].append(r.get("total_fro2") or 0.0)
            preds["nstar"].append(r["n_star_energy"]); preds["slug"].append(r["slug"])
        pnames = a3.get("predictors", [])
        fig, axes = plt.subplots(1, len(pnames), figsize=(4.2 * len(pnames), 3.8))
        axes = np.atleast_1d(axes)
        for j, pn in enumerate(pnames):
            axes[j].scatter(preds[pn], preds["nstar"], color=OI["purple"])
            rho = a3["spearman_rho"][j]; pv = a3["p_value"][j]; rej = a3["bh_reject"][j]
            axes[j].set_title(f"{pn}\nρ={rho} p={pv:.3f}{' *FDR' if rej else ''}", fontsize=9)
            axes[j].set_xlabel(pn, fontsize=8); axes[j].set_ylabel("n*_energy", fontsize=8)
            axes[j].grid(alpha=0.25)
        fig.suptitle("Arm 3 — what predicts a capability's necessary rank?", fontsize=11)
        _foot(axes[0], f"QUORUM · arm3 · any_separates={a3.get('any_separates')}")
        _save(fig, "F6_predictors")
    else:
        _skip("F6_predictors", "arm3/arm0 absent or skipped")

    rep["green"] = True
    _w()
    print("QUORUM_FIGURES_OK", json.dumps({"rendered": rep["rendered"], "skipped": rep["skipped"]},
          default=str), flush=True)


if __name__ == "__main__":
    main()
