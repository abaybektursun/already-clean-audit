"""VLAT §B — figures (weight-free, from S3 JSON). CPU --no-weights, fail-loud.

F1  retention trajectory — S_text across base→PG1→π0→π0.5 with bootstrap CI band, FLOOR line, recipe
    annotation per checkpoint.
F2  recipe-predicts — registered predicted label vs measured label per checkpoint; prediction-commit-SHA
    in the footer (the blinding proof).
F3  mechanism link — per-hop ΔW {median f_perp, median rel-Frob} vs the matched-random null, colored by
    the behavioral label.

S_text(c) = (c − FLOOR)/(BASE − FLOOR) computed here from raw MMLU acc (validated §9.1). Each figure
carries a provenance footer: run-id · commit · input-hash · BEHAVIORAL/WEIGHT · FIREWALL:within-Gemma-1.

  pg run infra.aws_pg.jobs.vlat_figures --no-weights --type c7i.xlarge --s3-prefix results/vlat_figs \
      --args "--disc-s3 s3://.../retention.json --conf-s3 s3://.../retention.json \
              --mech-s3 s3://.../mechanism.json --freeze-sha <sha> --theta 0.15" --budget-min 30
"""
import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
ORDER = ["base", "PG1", "pi0", "pi05"]
RECIPE = {"base": "pretrain", "PG1": "VLM FT", "pi0": "naive joint FT", "pi05": "co-train + insulation"}
PRED = {"PG1": "(reuse XGEN)", "pi0": "forget", "pi05": "retain"}


def _fetch(s3, dst):
    if not s3:
        return None
    subprocess.run(["aws", "s3", "cp", s3, str(dst), "--region", "us-east-1"], check=True)
    return json.loads(pathlib.Path(dst).read_text())


def _s_text_ci(acc_items, base_items, floor_items, n_boot=2000, seed=1337):
    """Nonparametric bootstrap of S_text = (mean(acc) − mean(floor))/(mean(base) − mean(floor)).
    Resamples each per-item 0/1 array independently (acc, base AND floor treated as random) → a
    CONSERVATIVE CI; a fixed-effect CI holding the §9.0 base/floor moments as plug-ins would be slightly
    tighter. Conclusion is unchanged under either. Returns (point, lo95, hi95)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a, b, f = (np.asarray(x, dtype=float) for x in (acc_items, base_items, floor_items))
    point = (a.mean() - f.mean()) / (b.mean() - f.mean())
    draws = []
    for _ in range(n_boot):
        ba = a[rng.integers(0, a.size, a.size)].mean()
        bb = b[rng.integers(0, b.size, b.size)].mean()
        bf = f[rng.integers(0, f.size, f.size)].mean()
        if bb - bf != 0:
            draws.append((ba - bf) / (bb - bf))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc-s3", required=True, help="retention.json with base,PG1,floor")
    ap.add_argument("--conf-s3", default="", help="retention.json with pi0,pi05 (sealed)")
    ap.add_argument("--mech-s3", default="", help="mechanism.json")
    ap.add_argument("--freeze-sha", required=True)
    ap.add_argument("--theta", type=float, required=True, help="θ_forget on the S_text scale")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disc = _fetch(a.disc_s3, out / "disc.json")["results"]
    conf = (_fetch(a.conf_s3, out / "conf.json") or {}).get("results", {}) if a.conf_s3 else {}
    mech = (_fetch(a.mech_s3, out / "mech.json") or {}).get("hops", {}) if a.mech_s3 else {}
    res = {**disc, **conf}
    assert "base" in res and "floor" in res, "need base + floor in the discovery retention.json"
    base_items = res["base"]["mmlu"]["per_item"]
    floor_items = res["floor"]["mmlu"]["per_item"]

    rows, foot = {}, f"freeze:{a.freeze_sha[:10]} · θ={a.theta} · BEHAVIORAL · FIREWALL:within-Gemma-1"
    for c in ORDER:
        if c not in res:
            continue
        pt, lo, hi = _s_text_ci(res[c]["mmlu"]["per_item"], base_items, floor_items)
        rows[c] = {"s_text": pt, "lo": lo, "hi": hi, "acc_pct": res[c]["mmlu"]["acc_pct"],
                   "ppl": res[c]["ppl"]["ppl"], "measured": "forget" if hi < a.theta else "retain"}

    # ---- F1 retention trajectory -------------------------------------------------------------
    xs = [c for c in ORDER if c in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = [rows[c]["s_text"] for c in xs]
    yerr = [[rows[c]["s_text"] - rows[c]["lo"] for c in xs], [rows[c]["hi"] - rows[c]["s_text"] for c in xs]]
    ax.errorbar(range(len(xs)), y, yerr=yerr, marker="o", capsize=4, lw=2, color="#1f3b73")
    ax.axhline(0.0, ls="--", c="#999", label="FLOOR (random decoder)")
    ax.axhline(a.theta, ls=":", c="#b3402a", label=f"θ_forget={a.theta}")
    for i, c in enumerate(xs):
        ax.annotate(RECIPE[c], (i, y[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_ylabel("S_text  (retained-reasoning fraction)")
    ax.set_title("VLAT F1 — reasoning retention across the Gemma-1 VLA lineage")
    ax.legend(fontsize=8, loc="lower left"); ax.text(0, -0.18, foot, transform=ax.transAxes, fontsize=6, color="#666")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"vlat_f1_retention.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- F2 recipe-predicts (honest: the boundary as S_text vs θ, NOT a label-match checkmark) --
    # Both sealed hops sit at floor; the "retain" label on π0.5 is label-agreement only
    # (behavioral_match=false), so the figure shows S_text+CI vs θ/floor, not a ✓MATCH that would
    # read as confirmation (post-critic desync fix).
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cells = [c for c in ("pi0", "pi05") if c in rows]
    xs2 = list(range(len(cells)))
    y2 = [rows[c]["s_text"] for c in cells]
    err2 = [[rows[c]["s_text"] - rows[c]["lo"] for c in cells], [rows[c]["hi"] - rows[c]["s_text"] for c in cells]]
    ax.bar(xs2, y2, yerr=err2, capsize=6, color="#b3b3b3", width=0.5, label="measured S_text")
    ax.axhline(0.0, ls="--", c="#999", label="FLOOR")
    ax.axhline(a.theta, ls=":", c="#b3402a", label=f"θ_forget={a.theta}")
    for i, c in enumerate(cells):
        ax.annotate(f"predicted: {PRED[c].upper()}", (i, rows[c]["hi"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#1f3b73")
    ax.set_xticks(xs2); ax.set_xticklabels([f"{c}\nS_text={rows[c]['s_text']:.3f}" for c in cells])
    ax.set_ylabel("S_text"); ax.set_ylim(-0.15, max(0.30, a.theta * 2.2))
    ax.set_title("VLAT F2 — recipe did NOT predict a split: both sealed VLA hops at floor\n"
                 "(H_recipe-predicts NOT CONFIRMED; π0.5 'retain' = label-only, behavioral_match=false)")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0, -0.30, foot + " · prediction registered pre-measurement (freeze precedes run)",
            transform=ax.transAxes, fontsize=6, color="#666")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"vlat_f2_recipe_predicts.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- F3 mechanism link -------------------------------------------------------------------
    if mech:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        hop_lbl = {"PG1__pi0": "PG1→π0", "PG1__pi05": "PG1→π0.5"}
        for h, m in mech.items():
            c = "pi0" if h.split("__")[1] == "pi0" else "pi05"
            st = rows.get(c, {}).get("s_text", float("nan"))
            at_floor = st < a.theta              # point-estimate at floor (honest, not the θ-label)
            col = "#b3402a" if at_floor else "#1a7a3a"
            tag = "at-floor" if at_floor else "retain"
            ax.scatter(m["median_rel_frob"], m["median_f_perp"], s=90, color=col,
                       label=f"{hop_lbl.get(h, h)}: ΔW large, behavior {tag} (S_text={st:.3f})")
            ax.annotate(hop_lbl.get(h, h), (m["median_rel_frob"], m["median_f_perp"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("median rel-Frob  ‖ΔW‖/‖W_PG1‖"); ax.set_ylabel("median f_perp (raw-ΔW out-of-span)")
        ax.set_title("VLAT F3 — weight-space mechanism vs behavioral label")
        ax.legend(fontsize=8); ax.text(0, -0.18, foot + " · WEIGHT · f_perp=raw-ΔW (≠PG2 cotangent)",
                                       transform=ax.transAxes, fontsize=6, color="#666")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"vlat_f3_mechanism.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)

    figs = ["vlat_f1_retention", "vlat_f2_recipe_predicts"] + (["vlat_f3_mechanism"] if mech else [])
    (out / "fig_index.json").write_text(json.dumps({
        "figures": figs, "rows": rows, "freeze_sha": a.freeze_sha, "theta": a.theta,
        "sources": {"disc": a.disc_s3, "conf": a.conf_s3, "mech": a.mech_s3},
        "claims": ["F1 retention trajectory", "F2 recipe-predicts on sealed {π0,π0.5}",
                   "F3 weight-space mechanism vs behavior"]}, indent=2))
    print("VLAT_FIGURES_OK", json.dumps({"figs": figs, "rows": {c: round(rows[c]["s_text"], 3) for c in rows}}))


if __name__ == "__main__":
    main()
