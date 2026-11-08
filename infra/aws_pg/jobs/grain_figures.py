"""GRAIN figure suite (CPU, weight-free — from the stage metrics JSONs only, the W12
convention). Every figure = number + CI + matched null; full-sweep (all swept points + all
component curves), per the house sweep-visualization rule. Okabe–Ito CVD-safe fixed palette.

F-T1 faithfulness frontier · F-G1 sealed RF per cell (+ effect-vs-rank inset) · F-G2 support
spread · F-S1 atlas · F-S2 canonicity re-read · F-S3 steering dose-response · F-X1 bridges.
A stage whose artifact is absent/UNRUN gets a SKIPPED entry (the §11.3 pivot path), never a
silent hole.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_figures --type r7i.4xlarge --no-weights \
      --s3-prefix results/grain_figs --budget-min 60 --args "--root s3://pg-lineage-769063704696/results"
"""
import argparse
import json
import pathlib
import subprocess

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}


def _try_fetch(uri, dst):
    r = subprocess.run(["aws", "s3", "cp", uri, str(dst), "--region", "us-east-1",
                        "--only-show-errors"], capture_output=True)
    return dst if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="s3://…/results")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    status = {}

    def load(rel, name):
        p = _try_fetch(f"{a.root}/{rel}", f"/tmp/grain_figs/{name}")
        return json.loads(pathlib.Path(p).read_text()) if p else None

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    # ---------------- F-T1: faithfulness frontier ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    any_t1 = False
    for site, mark in (("primary", "o"), ("secondary", "s")):
        rel = ("grain_t3diag/out/gates.json" if site == "primary"
               else "grain_ts2/out/gates.json")
        g = load(rel, f"gates_{site}.json")
        if not g:
            continue
        any_t1 = True
        for cid, c in g["configs"].items():
            if c.get("state") == "FAILED":
                continue
            m = c["cfg"]["m"]
            col = OI["blue"] if m == 1 else OI["verm"]
            ok = c.get("gate_pass")
            axes[0].scatter(c["quality"]["fvu"], c["delta_vqa"], marker=mark,
                            facecolor=col if ok else "none", edgecolor=col, s=55, zorder=3)
            axes[0].annotate(f"{site[0]}:{cid}", (c["quality"]["fvu"], c["delta_vqa"]),
                             fontsize=6, xytext=(3, 3), textcoords="offset points")
            axes[1].scatter(c["quality"]["fvu"], c["delta_ppl"], marker=mark,
                            facecolor=col if ok else "none", edgecolor=col, s=55, zorder=3)
            band = c["em_band"]
            axes[0].hlines(band["mean"] - max(c["delta_splice"], 0.01),
                           c["quality"]["fvu"] - 0.01, c["quality"]["fvu"] + 0.01,
                           color=OI["black"], linewidth=1)
    if any_t1:
        axes[0].set_xlabel("held-out FVU")
        axes[0].set_ylabel("splice ΔVQA (gate line = em-noise bound)")
        axes[1].set_xlabel("held-out FVU")
        axes[1].set_ylabel("splice Δppl")
        axes[1].axhline(0.5, color=OI["black"], linewidth=1)
        axes[1].axhline(-0.5, color=OI["black"], linewidth=1)
        for ax in axes:
            style(ax)
        fig.suptitle("F-T1 faithfulness frontier — filled=pass, blue m=1, verm m>1; o=primary s=secondary")
        fig.tight_layout()
        fig.savefig(out / "F_T1_faithfulness_frontier.png", dpi=180)
        status["F_T1"] = "ok"
    else:
        status["F_T1"] = "SKIPPED (no gates.json)"
    plt.close(fig)

    # ---------------- F-G1 headline + F-G2 support, per read ----------------
    for read in ("saction", "svl"):
        sealed = load(f"grain_g_sealed_{read}/out/metrics.json", f"sealed_{read}.json")
        disc = load(f"grain_gdisc_{read}/out/g_disc.json", f"gdisc_{read}.json")
        if not sealed or sealed.get("verdict") == "UNRUN" or "rf" not in sealed:
            status[f"F_G1_{read}"] = f"SKIPPED ({(sealed or {}).get('verdict', 'missing')})"
            continue
        order = ["full_read", "bstar", "rand_union_0", "rand_union_1", "rand_union_2",
                 "em_span", "equal_rank_span", "grain1_union"]
        order = [c for c in order if c in sealed["rf"]]
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_axes([0.08, 0.12, 0.6, 0.76])
        cols = {"full_read": OI["black"], "bstar": OI["verm"], "em_span": OI["green"],
                "equal_rank_span": OI["blue"], "grain1_union": OI["purple"]}
        rvals = [sealed["rf"][f"rand_union_{i}"]["rf"] for i in range(3)
                 if f"rand_union_{i}" in sealed["rf"]]
        if rvals:
            ax.axhspan(min(rvals), max(rvals), color=OI["orange"], alpha=0.18,
                       label="random-union band")
        for i, cell in enumerate(order):
            r = sealed["rf"][cell]
            col = cols.get(cell, OI["orange"])
            ax.errorbar(i, r["rf"], yerr=[[r["rf"] - r["ci"][0]], [r["ci"][1] - r["rf"]]],
                        fmt="o", color=col, capsize=4, markersize=7)
        ax.axhline(1.0, color=OI["black"], linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color=OI["black"], linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("sealed RF (dead-floor convention)")
        ax.set_title(f"F-G1 {read} — verdict: {sealed['verdict']} "
                     f"(δ_G={disc['random_unions']['delta_G']:.3f}, ρ={disc['rho']})")
        style(ax)
        ax.legend(fontsize=8)
        ins = fig.add_axes([0.74, 0.14, 0.24, 0.7])
        med = str(disc["median_seed"])
        curve = disc["per_seed"][med]["union_curve"]
        m_star = disc["m_star"]
        ins.plot([len(c["blocks"]) * m_star for c in curve], [c["effect"] for c in curve],
                 "o-", color=OI["verm"], label="B* unions")
        if disc.get("grain1"):
            ins.plot([c["n_latents"] for c in disc["grain1"]["curve"]],
                     [c["effect"] for c in disc["grain1"]["curve"]],
                     "s-", color=OI["purple"], label="m=1 unions")
        ins.axhline(disc["rho"] * disc["reference"]["effect_full_screen"],
                    color=OI["black"], linewidth=0.8, linestyle="--", label="ρ·full")
        ins.scatter([128], [disc["reference"]["effect_full_screen"]], marker="*", s=90,
                    color=OI["black"], label="full read")
        ins.set_xlabel("ablated rank")
        ins.set_ylabel("discovery effect")
        ins.set_title("effect vs rank", fontsize=9)
        ins.legend(fontsize=6)
        style(ins)
        fig.savefig(out / f"F_G1_{read}_sealed.png", dpi=180)
        plt.close(fig)
        status[f"F_G1_{read}"] = "ok"

        ss = sealed["support_spread"]
        fig, ax = plt.subplots(figsize=(6, 4))
        cum = ss["cum_share"]
        ax.plot(range(1, len(cum) + 1), cum, color=OI["blue"], linewidth=2)
        ax.axhline(0.9, color=OI["black"], linestyle="--", linewidth=0.8)
        ax.axvline(ss["n_for_90"], color=OI["verm"], linestyle=":",
                   label=f"n₉₀={ss['n_for_90']} ({ss['frac_active_for_90']:.2%} of active)")
        ax.axvline(0.25 * ss["n_active"], color=OI["green"], linestyle="--",
                   label="H_oblique bound (25% of active)")
        ax.set_xscale("log")
        ax.set_xlabel("blocks (sorted by alignment energy)")
        ax.set_ylabel("cumulative share of read-span energy")
        ax.set_title(f"F-G2 {read} block-support spread")
        ax.legend(fontsize=8)
        style(ax)
        fig.tight_layout()
        fig.savefig(out / f"F_G2_{read}_support.png", dpi=180)
        plt.close(fig)
        status[f"F_G2_{read}"] = "ok"

    # ---------------- F-G1d: descriptive grain-1 screening (E8 pivot) ----------------
    dsc = load("grain_gdisc_saction/out/g_disc.json", "gdisc_saction_d.json")
    if dsc and dsc.get("grain1_descriptive"):
        d = dsc["grain1_descriptive"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [c["n_latents"] for c in d["curve"]]
        ys = [c["effect"] for c in d["curve"]]
        ax.plot(xs, ys, "o-", color=OI["purple"], label="grain-1 cumulative unions")
        ax.axhline(dsc["reference"]["effect_full_screen"], color=OI["black"], linestyle="--",
                   linewidth=1, label=f"full read (k=128): {dsc['reference']['effect_full_screen']:.3f}")
        ax.set_xlabel("ablated grain-1 latents (alignment→effect order)")
        ax.set_ylabel("discovery effect (1 − RF, dead-floor)")
        ax.set_title("F-G1d S_action@L0 — DESCRIPTIVE (no null; no sealed pass — E8 pivot)")
        ax.legend(fontsize=8)
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "F_G1d_saction_descriptive.png", dpi=180)
        plt.close(fig)
        status["F_G1d"] = "ok"
    else:
        status["F_G1d"] = "SKIPPED"

    # ---------------- F-S1 atlas ----------------
    atlas = load("grain_s_atlas/out/metrics.json", "atlas.json")
    if atlas and any(v.get("seeds") for v in atlas.get("sites", {}).values()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for si, (site, sv) in enumerate(atlas["sites"].items()):
            if not sv.get("seeds"):
                continue
            blocks0 = load(f"grain_s_atlas/out/blocks_{site}_s0.json", f"blocks_{site}.json")
            if blocks0:
                sr = np.asarray(blocks0["stable_rank"], float)
                fr = np.asarray(blocks0["freq"], float)
                axes[0].hist(sr[fr > 0], bins=30, alpha=0.55, label=f"{site} (seed 0)",
                             color=[OI["blue"], OI["verm"]][si])
            meds = [sv["seeds"][s]["stable_rank"]["median"] for s in sv["seeds"]]
            axes[1].errorbar([si], [np.mean(meds)],
                             yerr=[[np.mean(meds) - min(meds)], [max(meds) - np.mean(meds)]],
                             fmt="o", capsize=6, color=[OI["blue"], OI["verm"]][si],
                             markersize=8, label=f"{site} seed band (n={len(meds)})")
        axes[0].set_xlabel("block stable rank (activation-weighted)")
        axes[0].set_ylabel("active blocks")
        axes[0].legend(fontsize=8)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(list(atlas["sites"]))
        axes[1].set_ylabel("median block stable rank")
        axes[1].legend(fontsize=8)
        for ax in axes:
            style(ax)
        fig.suptitle("F-S1 the block-dimension spectrum of PG1's VL stream")
        fig.tight_layout()
        fig.savefig(out / "F_S1_atlas.png", dpi=180)
        plt.close(fig)
        status["F_S1"] = "ok"
    else:
        status["F_S1"] = "SKIPPED"

    # ---------------- F-S2 canonicity re-read ----------------
    if atlas and atlas.get("s2"):
        fig, ax = plt.subplots(figsize=(7, 4))
        xt = []
        for i, (read, e) in enumerate(atlas["s2"].items()):
            if "jaccard_pairs" not in e:
                continue
            ax.scatter([i] * len(e["jaccard_pairs"]), e["jaccard_pairs"], s=60,
                       color=OI["verm"] if e.get("adjudicated") else OI["orange"],
                       label=None, zorder=3)
            ax.hlines(e["null_p95"], i - 0.25, i + 0.25, color=OI["black"], linewidth=1.5)
            ax.hlines(2 * e["null_p95"], i - 0.25, i + 0.25, color=OI["black"],
                      linewidth=1.0, linestyle="--")
            ax.scatter([i + 0.15], [e["subspace_canonicity_kcan16"]], marker="D",
                       color=OI["blue"], s=50, zorder=3)
            xt.append((i, f"{read}\n{'ADJ' if e.get('adjudicated') else 'descr.'}"
                          f"\n{e.get('verdict', '')}"))
        ax.set_xticks([t[0] for t in xt])
        ax.set_xticklabels([t[1] for t in xt], fontsize=8)
        ax.set_ylabel("Jaccard (dots) / subspace canonicity (diamond)")
        ax.set_title("F-S2 block-set agreement vs permutation null (solid=p95, dashed=2×)")
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "F_S2_canonicity.png", dpi=180)
        plt.close(fig)
        status["F_S2"] = "ok"
    else:
        status["F_S2"] = "SKIPPED"

    # ---------------- F-S3 steering ----------------
    steer = load("grain_s_steer/out/metrics.json", "steer.json")
    if steer and steer.get("curves"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for tag, curve in steer["curves"].items():
            xs = sorted(float(x) for x in curve)
            real = tag.startswith("real")
            col = OI["verm"] if tag == "real_pc0" else (OI["purple"] if real else OI["sky"])
            axes[0].plot(xs, [curve[str(x)]["shift_rate"] for x in xs], "o-",
                         color=col, alpha=1.0 if real else 0.6, label=tag)
            axes[1].plot(xs, [curve[str(x)]["cider"] for x in xs], "o-",
                         color=col, alpha=1.0 if real else 0.6, label=tag)
        axes[1].axhline(steer["baseline"]["splice"]["cider"], color=OI["black"],
                        linestyle="--", linewidth=0.8, label="splice α=0")
        axes[0].set_xlabel("α (·σ)")
        axes[0].set_ylabel("caption shift rate vs α=0")
        axes[1].set_xlabel("α (·σ)")
        axes[1].set_ylabel("CIDEr")
        axes[0].legend(fontsize=7)
        for ax in axes:
            style(ax)
        fig.suptitle(f"F-S3 steering dose-response — target block {steer['target']['block']} "
                     f"({steer['target']['source']}); signature: {steer.get('signature')}")
        fig.tight_layout()
        fig.savefig(out / "F_S3_steering.png", dpi=180)
        plt.close(fig)
        status["F_S3"] = "ok"
    else:
        status["F_S3"] = "SKIPPED"

    # ---------------- F-X1 bridges ----------------
    xb = load("grain_x_bridge/out/metrics.json", "xbridge.json")
    if xb and xb.get("x1"):
        fig, ax = plt.subplots(figsize=(7, 4))
        sites = list(xb["x1"])
        states = ("shared", "residual", "ft")
        w = 0.25
        for j, st in enumerate(states):
            vals = [xb["x1"][s][st]["top8_share"] for s in sites]
            ax.bar([i + (j - 1) * w for i in range(len(sites))], vals, width=w - 0.03,
                   color=[OI["verm"], OI["blue"], OI["green"]][j], label=st)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels([f"{s} (L{xb['sites'][s]})" for s in sites])
        ax.set_ylabel("top-8 blocks' share of shift energy")
        ax.set_title("F-X1 damage vs task footprint block-concentration (EXPLORATORY)")
        ax.legend(fontsize=8)
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "F_X1_bridge.png", dpi=180)
        plt.close(fig)
        status["F_X1"] = "ok"
    else:
        status["F_X1"] = "SKIPPED"

    (out / "metrics.json").write_text(json.dumps({"figures": status}, indent=2))
    print("GRAIN_FIGS_OK", json.dumps(status))


if __name__ == "__main__":
    main()
