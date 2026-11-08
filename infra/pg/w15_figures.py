"""W15 carrier-gate figure suite (prereg §10) — WEIGHT-FREE: every panel renders from the
result JSONs in results/w15/ only (no model, no GPU, no HF). Runs on the laptop (matplotlib).
Mirrors the w12_figures house style (Agg backend, PNG+PDF, provenance footer per figure).

    python -m infra.pg.w15_figures \
        --results studies/lineage/paligemma2/results/w15 \
        --out studies/lineage/paligemma2/figures/w15_figures \
        --provenance w15-carrier-gate@<freeze-sha>

Figures (each = a claim backed by number + CI + null; title = the verdict):
  F1 install     R_cap by selection {oracle, leverage, energy, whole-loss} at r* + random-
                 rotation null band + recomputed energy@r* anchor + full-rank ceiling.
  F2 retention   O(forget,install) ± CI vs permuted-null band + planted anchors (0,1) +
                 magnitude-overlap references.
  F3 validity    base↔after leverage rank-corr (install) + magnitude-overlap controls (retention).
  F4 out-of-span install-gradient f_perp (the in-span-edge-map validity disclosure).
  F5 robustness  R_cap across seeds (install) + O across seeds × k∈{25,50,75}% (retention).

Regenerate after any critic fold (feedback_post_critic_artifact_desync). Provenance footer
records the generating script + the freeze SHA + DISCOVERY/CONFIRMATORY tag.
"""
from __future__ import annotations

import argparse
import json
import pathlib

# the full-rank install ceiling (no-retrain rank-1024 R_cap), recomputed §9.0 (BASELINES_W15).
FULL_RANK_CEILING_RCAP = 0.7974


def _load(results, name):
    p = pathlib.Path(results) / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, out, name, footer):
    fig.text(0.5, 0.004, footer, ha="center", va="bottom", fontsize=6, color="gray")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[W15-FIG] wrote {name}.png/.pdf")


def _seed_mean(per_seed, arm, field="R_cap"):
    """Mean over seeds of per_seed[sd][arm][field]; returns (mean, lo, hi=min/max over seeds)."""
    vals = [s[arm][field] for s in per_seed.values()]
    return sum(vals) / len(vals), min(vals), max(vals)


def fig_install(m, out, prov):
    """F1 — the install fork. R_cap of each selection vs the random-rotation null band."""
    import matplotlib.pyplot as plt
    null = m["random_rotation_null"]
    per_seed = m["per_seed"]
    energy = m["energy"]["R_cap"]
    oracle = m["oracle"]["R_cap"]
    lev_mean, lev_lo, lev_hi = _seed_mean(per_seed, "leverage")
    whl_mean, whl_lo, whl_hi = _seed_mean(per_seed, "wholeloss")

    labels = ["energy", "whole-loss", "leverage", "oracle"]
    vals = [energy, whl_mean, lev_mean, oracle]
    errs = [[0, 0, lev_mean - lev_lo, 0], [0, 0, lev_hi - lev_mean, 0]]
    colors = ["#9aa0ad", "#0ea5e9", "#4f46e5", "#059669"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = range(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, width=0.62, capsize=4, zorder=3)
    # random-rotation null band (p5–p95) + the energy@r* anchor + full-rank ceiling
    ax.axhspan(null["p5"], null["p95"], color="#e11d48", alpha=0.10, zorder=1,
               label=f"random-rotation null p5–p95 (n={null['n_draws']})")
    ax.axhline(null["p50"], color="#e11d48", ls=":", lw=1, zorder=2)
    ax.axhline(FULL_RANK_CEILING_RCAP, color="#15151d", ls="--", lw=1, zorder=2,
               label=f"full-rank install ceiling ({FULL_RANK_CEILING_RCAP:.3f})")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel(r"RefCOCO $R_{cap}$ @ r*=512 (no-retrain graft)")
    ax.set_ylim(-0.02, max(0.2, FULL_RANK_CEILING_RCAP * 1.08))
    v = m["verdict"]["headline"]
    ax.set_title(f"INSTALL ({m['pg_variant']}): {v}"
                 + ("" if m["verdict"]["seed_stable"] else "  [SEED-UNSTABLE]"))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out, f"fig_w15_install_{m['pg_variant']}",
          f"{prov} · CONFIRMATORY · energy/oracle seed-invariant; leverage/whole-loss mean±range over "
          f"{len(per_seed)} seeds · δ_install={m['config']['delta_install']}")
    plt.close(fig)


def fig_retention(m, out, prov):
    """F2 — the retention fork. Pooled O(forget,install) ± CI vs the permuted-null band."""
    import matplotlib.pyplot as plt
    head = str(m["config"]["fracs"][0])
    seeds = list(m["per_seed"].keys())
    # seed-0 headline frac block carries the bands; points across seeds overlaid
    blk0 = m["per_seed"][seeds[0]][head]
    null = blk0["permuted_null"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # planted anchors (0 and 1) as the estimator rails
    ax.axhline(0.0, color="#059669", ls=":", lw=1, alpha=0.7)
    ax.axhline(1.0, color="#e11d48", ls=":", lw=1, alpha=0.5)
    # permuted-null band p5–p95
    ax.axhspan(null["p5"], null["p95"], color="#62626f", alpha=0.14, zorder=1,
               label=f"permuted-null p5–p95 (analytic {null['analytic']:.3f})")
    # magnitude-overlap references (p5 lower bounds — O must sit below these)
    ax.axhline(blk0["O_forget_mag"]["p5"], color="#f59e0b", ls="--", lw=1,
               label=f"O(forget,|M|) p5 = {blk0['O_forget_mag']['p5']:.3f}")
    ax.axhline(blk0["O_install_mag"]["p5"], color="#0ea5e9", ls="--", lw=1,
               label=f"O(install,|M|) p5 = {blk0['O_install_mag']['p5']:.3f}")
    # O(forget,install) per seed with cluster-bootstrap CI
    for i, sd in enumerate(seeds):
        b = m["per_seed"][sd][head]["O_forget_install"]
        ax.errorbar(i, b["point"], yerr=[[b["point"] - b["p5"]], [b["p95"] - b["point"]]],
                    fmt="o", color="#4f46e5", capsize=4, zorder=4,
                    label="O(forget,install) ± cluster-boot" if i == 0 else None)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(r"pooled top-k overlap $O$")
    ax.set_ylim(-0.03, max(0.4, null["p95"] * 1.3, blk0["O_install_mag"]["p5"] * 1.2))
    v = m["verdict"]["headline"]
    ax.set_title(f"RETENTION ({m['pg_variant']}): {v}"
                 + ("" if m["verdict"]["seed_stable"] else "  [SEED-UNSTABLE]"))
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out, f"fig_w15_retention_{m['pg_variant']}",
          f"{prov} · CONFIRMATORY · k=50% |M| energy/matrix · cluster-bootstrap over matrices · "
          f"f_perp(install)={blk0['f_perp_install_mean']:.3f}")
    plt.close(fig)


def fig_validity(inst, ret, out, prov):
    """F3 — instrument validity: install base↔after leverage rank-corr + retention magnitude controls."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    if inst is not None:
        seeds = list(inst["per_seed"].keys())
        rc = [inst["per_seed"][s]["base_after_rank_corr"] for s in seeds]
        axes[0].bar(range(len(seeds)), rc, color="#4f46e5", width=0.6)
        axes[0].axhline(0, color="k", lw=0.6)
        axes[0].set_xticks(range(len(seeds))); axes[0].set_xticklabels([f"s{s}" for s in seeds])
        axes[0].set_ylabel("base↔after leverage rank-corr")
        axes[0].set_title("INSTALL: is the base map resolving?")
        axes[0].grid(axis="y", alpha=0.25)
    if ret is not None:
        head = str(ret["config"]["fracs"][0]); s0 = list(ret["per_seed"].keys())[0]
        b = ret["per_seed"][s0][head]
        names = ["O(F,I)", "O(F,|M|)", "O(I,|M|)"]
        pts = [b["O_forget_install"]["point"], b["O_forget_mag"]["point"], b["O_install_mag"]["point"]]
        axes[1].bar(names, pts, color=["#4f46e5", "#f59e0b", "#0ea5e9"], width=0.6)
        axes[1].set_ylabel("pooled overlap O")
        axes[1].set_title("RETENTION: function beyond magnitude?")
        axes[1].grid(axis="y", alpha=0.25)
    _save(fig, out, "fig_w15_validity",
          f"{prov} · CONTROL · low base↔after corr ⇒ leverage non-resolving (oracle is the verdict); "
          f"O(F,I) must sit below both magnitude controls")
    plt.close(fig)


def fig_outofspan(ret, out, prov):
    """F4 — out-of-span disclosure: install-gradient f_perp across seeds/fracs."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for sd in ret["per_seed"]:
        fracs = sorted(ret["per_seed"][sd].keys(), key=float)
        fp = [ret["per_seed"][sd][f]["f_perp_install_mean"] for f in fracs]
        ax.plot([float(f) for f in fracs], fp, "o-", label=f"seed {sd}")
    ax.axhline(0.5, color="#e11d48", ls=":", lw=1, label="0.5 (in-span map blind above this)")
    ax.set_xlabel("top-edge energy fraction k"); ax.set_ylabel(r"install-gradient $f_\perp$")
    ax.set_ylim(0, 1)
    ax.set_title(f"OUT-OF-SPAN ({ret['pg_variant']}): can the in-span edge map see the install carrier?")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    _save(fig, out, f"fig_w15_outofspan_{ret['pg_variant']}",
          f"{prov} · DISCLOSURE · large f_perp ⇒ install lives out-of-span ⇒ in-span gate cannot freeze it")
    plt.close(fig)


def fig_robustness(inst, ret, out, prov):
    """F5 — seed/frac robustness: install R_cap across seeds; retention O across seeds × k."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    if inst is not None:
        seeds = list(inst["per_seed"].keys())
        for arm, c in [("leverage", "#4f46e5"), ("wholeloss", "#0ea5e9")]:
            axes[0].plot([int(s) for s in seeds],
                         [inst["per_seed"][s][arm]["R_cap"] for s in seeds], "o-", color=c, label=arm)
        axes[0].axhline(inst["oracle"]["R_cap"], color="#059669", ls="--", lw=1, label="oracle")
        axes[0].axhline(inst["energy"]["R_cap"], color="#9aa0ad", ls=":", lw=1, label="energy")
        axes[0].set_xlabel("seed"); axes[0].set_ylabel(r"$R_{cap}$")
        axes[0].set_title("INSTALL seed-stability"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)
    if ret is not None:
        for sd in ret["per_seed"]:
            fracs = sorted(ret["per_seed"][sd].keys(), key=float)
            axes[1].plot([float(f) for f in fracs],
                         [ret["per_seed"][sd][f]["O_forget_install"]["point"] for f in fracs],
                         "o-", label=f"seed {sd}")
        axes[1].set_xlabel("top-edge energy fraction k"); axes[1].set_ylabel("O(forget,install)")
        axes[1].set_title("RETENTION seed×k robustness"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.25)
    _save(fig, out, "fig_w15_robustness",
          f"{prov} · ROBUSTNESS · a verdict that flips across seeds = SEED-UNSTABLE non-result")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="studies/lineage/paligemma2/results/w15")
    ap.add_argument("--out", default="studies/lineage/paligemma2/figures/w15_figures")
    ap.add_argument("--provenance", default="w15-carrier-gate")
    ap.add_argument("--variant", default="pt", help="which axis's metrics to lead with (pt/mix)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    inst = _load(a.results, f"install_{a.variant}_metrics.json")
    ret = _load(a.results, f"retention_{a.variant}_metrics.json")
    prov = a.provenance
    if inst is not None:
        fig_install(inst, out, prov)
    if ret is not None:
        fig_retention(ret, out, prov)
    if inst is not None or ret is not None:
        fig_validity(inst, ret, out, prov)
    if ret is not None:
        fig_outofspan(ret, out, prov)
        fig_robustness(inst, ret, out, prov)
    elif inst is not None:
        fig_robustness(inst, None, out, prov)
    print("[W15-FIG] done:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
