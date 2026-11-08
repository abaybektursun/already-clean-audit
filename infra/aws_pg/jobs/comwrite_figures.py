"""COMWRITE figure suite (spec §11; CPU) — every panel = number + CI + matched null; every
sweep FULL (all swept points + all component curves; the complete-sweep rule). PNG+PDF,
colourblind-safe, provenance footers, missing-arm panels skipped LOUD (printed, listed in
figures.json). Arm inputs are the banked metrics.json artifacts.

  F-A1  eigen-share ladder per type + Haar bands; A3 sealed pooled ladder + k*; mean-dir
        cosine + LOO stability panels.
  F-A2  star test: intact vs residual CCC against BOTH null distributions + verdict.
  F-A3  origin overlaps (PG1 / gemma-2b) vs floor + the hop row vs matched-random.
  F-B1  four-cell pattern per donor × readout family (score / damage bars, em bands,
        FT/base anchors, interaction annotated).
  F-B2  rank ladder on base+shared (vqav2): all k, all three readout families.
  F-B3  location sub-arm (if run).
  F-C1  savings per constraint cell with δ_C line + train-CE panel.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_figures --type r7i.2xlarge --no-weights \
      --s3-prefix results/comwrite_figs --budget-min 25 --args "--a-s3 s3://.../comwrite_a_anatomy/out \
      --b-s3 s3://.../comwrite_b_vqav2_disc/out,... --c-s3 s3://.../comwrite_c_into/out,..."
"""
import argparse
import json
import pathlib

CB = {"real": "#0072B2", "null": "#999999", "floor": "#D55E00", "seal": "#009E73",
      "pg1": "#0072B2", "g2b": "#56B4E9", "hop": "#CC79A7", "rand": "#999999",
      "shared": "#0072B2", "residual": "#009E73", "ft": "#000000", "base": "#666666",
      "em": "#D55E00", "kstar": "#CC79A7"}
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TAG = "comwrite-a-freeze-2026-07-05"


def _fetch_json(uri, dst):
    from infra.aws_pg.jobs._inherit_shared import fetch
    return json.loads(fetch(uri, dst).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--a-s3", default="", help=".../comwrite_a_anatomy/out")
    ap.add_argument("--b-s3", default="", help="comma list of B out prefixes (disc first)")
    ap.add_argument("--c-s3", default="", help="comma list of C out prefixes")
    ap.add_argument("--loc-s3", default="",
                    help="comma list of location sub-arm out prefixes (F-B3)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    made, skipped = [], []

    def _footer(fig, tag):
        fig.text(0.01, 0.005, f"COMWRITE · {tag} · {TAG}", fontsize=6, color="#666")

    def _save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        made.append(name)
        print(f"[figs] made {name}", flush=True)

    A = _fetch_json(f"{a.a_s3}/metrics.json", "/tmp/comwrite/fig_a.json") if a.a_s3 else None
    B = [_fetch_json(f"{p}/metrics.json", f"/tmp/comwrite/fig_b{i}.json")
         for i, p in enumerate([x for x in a.b_s3.split(",") if x])]
    C = [_fetch_json(f"{p}/metrics.json", f"/tmp/comwrite/fig_c{i}.json")
         for i, p in enumerate([x for x in a.c_s3.split(",") if x])]

    # ---------------- F-A1: shares + ladder + structure ----------------
    if A:
        pm = A["per_matrix"]
        k_grid = [int(k) for k in A["k_grid"]]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for t in PROJ:
            rows = [m for m in pm if m["type"] == t]
            if not rows:
                continue
            real = [float(np.mean([m["a1_share_by_k"][str(k)]["share"] for m in rows]))
                    for k in k_grid]
            null = [float(np.mean([m["a1_share_by_k"][str(k)]["null_p99.5"] for m in rows]))
                    for k in k_grid]
            ln, = ax[0, 0].plot(k_grid, real, marker="o", ms=3, label=t)
            ax[0, 0].plot(k_grid, null, ls=":", color=ln.get_color(), alpha=0.6)
        ax[0, 0].set(xscale="log", xlabel="k", ylabel="top-k trace share",
                     title="A1 consensus concentration (solid=real, dotted=Haar p99.5)")
        ax[0, 0].legend(fontsize=7, ncol=2)

        lad = {int(k): v for k, v in A["a3_pooled_ladder"].items()}
        ks = sorted(lad)
        ax[0, 1].plot(ks, [lad[k] for k in ks], marker="o", color=CB["seal"],
                      label="sealed capture (pooled)")
        d0 = pm[0]["d"]
        ax[0, 1].plot(ks, [k / d0 for k in ks], ls="--", color=CB["floor"],
                      label="chance k/d")
        ax[0, 1].axvline(A["k_star"], color=CB["kstar"], ls="-.",
                         label=f"k*={A['k_star']}")
        ax[0, 1].set(xscale="log", xlabel="k", ylabel="sealed-half energy capture",
                     title="A3 out-of-sample ladder (discovery basis)")
        ax[0, 1].legend(fontsize=8)

        for t in PROJ:
            rows = [m for m in pm if m["type"] == t]
            if rows:
                ax[1, 0].scatter([m["layer"] for m in rows],
                                 [m["mean_dir_cos_top1"] for m in rows], s=12, label=t)
        ax[1, 0].set(xlabel="layer", ylabel="|cos(top-1, mean-dir)|", ylim=(0, 1.02),
                     title="A1 star comparator (H_star-rank1)")
        ax[1, 0].legend(fontsize=7, ncol=2)

        for t in PROJ:
            rows = [m for m in pm if m["type"] == t]
            if rows:
                ax[1, 1].scatter([m["layer"] for m in rows],
                                 [m["loo_stability_mean"] for m in rows], s=12, label=t)
        if pm:
            ax[1, 1].axhline(pm[0]["loo_stability_floor"], ls="--", color=CB["floor"],
                             label="Haar floor k/d")
        ax[1, 1].set(xlabel="layer", ylabel=f"LOO overlap @k={pm[0]['k_loo']}",
                     title="A1 LOO stability")
        ax[1, 1].legend(fontsize=7, ncol=2)
        _footer(fig, "F-A1")
        _save(fig, "F-A1_structure_ladder")

        # ---------------- F-A2: star test ----------------
        a2 = A.get("a2")
        if a2:
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, (name, rec) in enumerate((("intact", a2["intact"]),
                                             ("residual", a2["residual"]))):
                for j, null in enumerate(("cloud_null", "perm_null")):
                    s = rec[null]["samples"]
                    ax.violinplot([s], positions=[i * 3 + j], widths=0.8)
                ax.scatter([i * 3 + 0.5], [rec["ccc"]], color=CB["real"], zorder=5,
                           label="real CCC" if i == 0 else None)
                ax.text(i * 3 + 0.5, rec["ccc"] + 0.01,
                        f"{name}\nCCC={rec['ccc']:.3f}\ntriple={rec['triple_statistic']:.2f}",
                        ha="center", fontsize=8)
            ax.set_xticks([0, 1, 3, 4])
            ax.set_xticklabels(["intact/cloud", "intact/perm", "resid/cloud", "resid/perm"],
                               fontsize=8)
            ax.set(ylabel="CCC", title=f"A2 star test — {a2['verdict']}")
            ax.legend(fontsize=8)
            _footer(fig, "F-A2")
            _save(fig, "F-A2_star_test")
        else:
            skipped.append("F-A2 (no a2 in A metrics)")

        # ---------------- F-A3: origin + hop ----------------
        a4, a5 = A.get("a4", []), A.get("a5", {})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        if a4:
            xs = np.arange(len(PROJ))
            for tag_o, off in (("pg1", -0.15), ("g2b", 0.15)):
                key = f"overlap_{tag_o}"
                means = [np.mean([r[key] for r in a4 if r["type"] == t and key in r])
                         if any(r["type"] == t and key in r for r in a4) else np.nan
                         for t in PROJ]
                ax[0].bar(xs + off, means, width=0.3, color=CB[tag_o], label=f"W_{tag_o}")
            ax[0].axhline(a4[0]["floor"], ls="--", color=CB["floor"], label="k*/d floor")
            ax[0].set_xticks(xs)
            ax[0].set_xticklabels(PROJ, rotation=30, fontsize=7)
            ax[0].set(ylabel=f"overlap(U_wr(k*), base top-k*)", title="A4 origin")
            ax[0].legend(fontsize=8)
        else:
            skipped.append("F-A3 origin (no a4)")
        if a5 and a5.get("by_type"):
            xs, hop_m, hop_lo, hop_hi, rnd_m = [], [], [], [], []
            for i, t in enumerate([t for t in PROJ if t in a5["by_type"]]):
                b = a5["by_type"][t]
                xs.append(i)
                hop_m.append(b["hop"]["mean"])
                hop_lo.append(b["hop"]["mean"] - b["hop"]["ci_lo"])
                hop_hi.append(b["hop"]["ci_hi"] - b["hop"]["mean"])
                rnd_m.append(b["rand"]["mean"])
            ax[1].errorbar(xs, hop_m, yerr=[hop_lo, hop_hi], fmt="o", color=CB["hop"],
                           label="π0 hop into U_wr(k*)", capsize=3)
            ax[1].plot(xs, rnd_m, "s", color=CB["rand"], label="matched-random mean")
            ax[1].set_xticks(xs)
            ax[1].set_xticklabels([t for t in PROJ if t in a5["by_type"]], rotation=30,
                                  fontsize=7)
            ax[1].set(ylabel="energy fraction", title="A5 hop row (EXPLORATORY, n=1 hop)")
            ax[1].legend(fontsize=8)
        else:
            skipped.append("F-A3 hop (no a5)")
        _footer(fig, "F-A3")
        _save(fig, "F-A3_origin_hop")
    else:
        skipped += ["F-A1", "F-A2", "F-A3"]

    # ---------------- F-B1/F-B2: the decomposition ----------------
    for bi, b in enumerate(B):
        donor, half = b["donor"], b["half"]
        rd = b["readouts"]
        cells = list(rd)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        xs = np.arange(len(cells))
        base_s = rd.get("identity@0", {}).get("donor", {}).get("score")
        for j, (panel, get) in enumerate((
                ("donor score", lambda r: (r["donor"]["score"], r["donor"]["ci95"])),
                ("cross NLL (mean of tasks)", lambda r: (
                    float(np.mean([r[k]["nll"] for k in r if k.startswith("nll_")])
                          ) if any(k.startswith("nll_") for k in r) else np.nan, None)),
                ("blank-image ppl", lambda r: (r["ppl"]["ppl"], None)))):
            vals, err = [], []
            for c in cells:
                if "donor" not in rd[c]:
                    vals.append(np.nan)
                    err.append(0)
                    continue
                v, ci = get(rd[c])
                vals.append(v)
                err.append(0 if ci is None else (ci[1] - ci[0]) / 2)
            colors = [CB.get(c.split("@")[0], "#888888") for c in cells]
            ax[j].bar(xs, vals, yerr=err, color=colors, capsize=2)
            ax[j].set_xticks(xs)
            ax[j].set_xticklabels(cells, rotation=45, fontsize=6, ha="right")
            ax[j].set(title=f"{panel}")
        need = {"shared@0", "residual@0", "identity@0", "ft@0"}
        if need <= set(cells) and all("donor" in rd[c] for c in need):
            I = (rd["shared@0"]["donor"]["score"] + rd["residual@0"]["donor"]["score"]
                 - rd["identity@0"]["donor"]["score"] - rd["ft@0"]["donor"]["score"])
            fig.suptitle(f"F-B1 {donor} ({half}) k={b['k']}  interaction I(score)={I:+.4f}")
        else:
            fig.suptitle(f"F-B1 {donor} ({half}) k={b['k']}")
        _footer(fig, f"F-B1 {donor} {half}")
        _save(fig, f"F-B1_cells_{donor}_{half}")

        lad_cells = sorted([c for c in cells if c.startswith("shared_k")],
                           key=lambda c: int(c.split("shared_k")[1].split("@")[0]))
        if lad_cells:
            ks = [int(c.split("shared_k")[1].split("@")[0]) for c in lad_cells]
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for j, (panel, fn) in enumerate((
                    ("donor score", lambda r: r["donor"]["score"]),
                    ("cross NLL", lambda r: float(np.mean(
                        [r[k]["nll"] for k in r if k.startswith("nll_")]))),
                    ("ppl", lambda r: r["ppl"]["ppl"]))):
                ax[j].plot(ks, [fn(rd[c]) for c in lad_cells], marker="o")
                if base_s is not None and j == 0:
                    ax[j].axhline(base_s, ls="--", color=CB["base"], label="base")
                    ax[j].legend(fontsize=8)
                ax[j].set(xscale="log", xlabel="k", title=f"F-B2 {panel}")
            _footer(fig, f"F-B2 {donor}")
            _save(fig, f"F-B2_ladder_{donor}_{half}")

    # ---------------- F-B3: location sub-arm (partial-surgery groups) ----------------
    LOC = [_fetch_json(f"{p}/metrics.json", f"/tmp/comwrite/fig_l{i}.json")
           for i, p in enumerate([x for x in a.loc_s3.split(",") if x])]
    if LOC:
        full_b = next((b for b in B if b["donor"] == LOC[0]["donor"]
                       and b["half"] == LOC[0]["half"]), None)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        groups = ["+".join(t.split("_")[0] for t in PROJ
                           if t not in {t2 for _L, t2 in m["excluded_lt"]})
                  for m in LOC]
        xs = np.arange(len(LOC))
        for j, (panel, fn) in enumerate((
                ("donor score", lambda r: r["donor"]["score"]),
                ("blank-image ppl", lambda r: r["ppl"]["ppl"]))):
            sh = [fn(m["readouts"]["shared@0"]) for m in LOC]
            re_ = [fn(m["readouts"]["residual@0"]) for m in LOC]
            ax[j].bar(xs - 0.17, sh, width=0.34, color=CB["shared"], label="base+shared(group)")
            ax[j].bar(xs + 0.17, re_, width=0.34, color=CB["residual"],
                      label="base+residual(group)")
            if full_b:
                ax[j].axhline(fn(full_b["readouts"]["shared@0"]), ls="--",
                              color=CB["shared"], alpha=0.6, label="full-grid shared")
                ax[j].axhline(fn(full_b["readouts"]["identity@0"]), ls=":",
                              color=CB["base"], label="base")
            ax[j].set_xticks(xs)
            ax[j].set_xticklabels(groups, fontsize=7)
            ax[j].set(title=f"F-B3 {panel} ({LOC[0]['donor']} {LOC[0]['half']})")
            ax[j].legend(fontsize=7)
        _footer(fig, "F-B3")
        _save(fig, "F-B3_location")
    else:
        skipped.append("F-B3 (no loc metrics)")

    # ---------------- F-C1: savings per constraint cell ----------------
    if C:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        names = [c["cell"] for c in C]
        g = [c["best_checkpoint"]["acc_pct"] - c["checkpoints"]["0"]["acc_pct"] for c in C]
        ax[0].bar(np.arange(len(names)), g, color=[CB.get(n.split("@")[0], "#888") for n in names])
        ax[0].set_xticks(np.arange(len(names)))
        ax[0].set_xticklabels(names, rotation=30, fontsize=8)
        ax[0].set(ylabel="G = best-ckpt − step-0 (MMLU pts)", title="F-C1 savings per cell")
        for c in C:
            ce = c.get("train_ce", {}).get("curve_every50", [])
            ax[1].plot(np.arange(len(ce)) * 50, ce, label=c["cell"])
        ax[1].set(xlabel="step", ylabel="train CE", title="train-CE (identification)")
        ax[1].legend(fontsize=8)
        _footer(fig, "F-C1")
        _save(fig, "F-C1_savings")
    else:
        skipped.append("F-C1 (no C metrics)")

    (out / "figures.json").write_text(json.dumps({"made": made, "skipped": skipped},
                                                 indent=2))
    print("FIGS_OK", json.dumps({"made": made, "skipped": skipped}))


if __name__ == "__main__":
    main()
