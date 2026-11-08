"""LACUNA figure suite (spec §10) — weight-free; reads the banked JSONs, writes PNG + PDF.

Every panel carries the number, its CI, and the NULL it is judged against; the complete-sweep rule
(`feedback_complete_sweep_visualization`) is enforced literally — every measured (row, task,
condition) point appears in F1, including the ones that landed at the floor. Regenerated AFTER any
critic-driven verdict change (`feedback_post_critic_artifact_desync`).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_figures --type r7i.2xlarge --no-weights \
      --args "--agg-prefix results/lacuna_agg" --s3-prefix results/lacuna_figs --budget-min 45
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

S3 = "s3://pg-lineage-769063704696"
# Okabe-Ito colourblind-safe.
C_REAL = "#0072B2"
C_GRAY = "#E69F00"
C_NULL = "#999999"
C_CEIL = "#009E73"
C_FLOOR = "#D55E00"

ROW_ORDER = ["gemma2b_qonly", "stitch0_s1", "stitch0_s2", "rand_dec", "matched_dw",
             "pi0", "pi05", "pi0_pg1embed", "pi05_pg1embed", "pg1"]
ROW_LABEL = {"gemma2b_qonly": "R1 gemma-2b\n(question-only)", "stitch0_s1": "R2 stitch-0\ns1",
             "stitch0_s2": "R2 stitch-0\ns2", "pg1": "R3 PG1\n(CEILING)", "pi0": "R4 π0\n(re-shell)",
             "pi05": "R5 π0.5\n(re-shell)", "matched_dw": "R6 matched-ΔW\n(FLOOR)",
             "rand_dec": "R7 random\ndecoder", "pi0_pg1embed": "R4b π0\n(PG1 embed)",
             "pi05_pg1embed": "R5b π0.5\n(PG1 embed)"}
TASK_LABEL = {"vqav2": "VQAv2 (soft acc)", "coco_captions": "COCO (CIDEr-D)",
              "textvqa": "TextVQA (ANLS)", "refcoco": "RefCOCO (acc@0.5 IoU)"}


def _pull(prefix, dst):
    dst = pathlib.Path(dst); dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{S3}/{prefix}/out/", str(dst), "--region", "us-east-1"],
                   check=True, capture_output=True)
    return dst


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(pathlib.Path(out) / f"{name}.{ext}", bbox_inches="tight", dpi=140)


def _footer(ax, text):
    ax.text(0.0, -0.30, text, transform=ax.transAxes, fontsize=6, color="#555", va="top")


def fig_ladder(agg, out, prov, half):
    import matplotlib.pyplot as plt
    import numpy as np
    tasks = [t for t in ("vqav2", "coco_captions", "textvqa", "refcoco")
             if agg["per_task"].get(t, {}).get("halves", {}).get(half, {}).get("rows")]
    if not tasks:
        return None
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.1 * len(tasks), 7.2), squeeze=False)
    for j, task in enumerate(tasks):
        hb = agg["per_task"][task]["halves"][half]
        rows = [r for r in ROW_ORDER if r in hb["rows"]]
        corpus = agg["per_task"][task]["corpus"]
        ax = axes[0][j]
        xs = np.arange(len(rows))
        real, lo, hi, gray = [], [], [], []
        for r in rows:
            e = hb["rows"][r]
            if corpus:
                real.append(e.get("cider", np.nan)); lo.append(np.nan); hi.append(np.nan)
                gray.append(e.get("cider_gray", np.nan))
            else:
                m = e.get("mean", {})
                real.append(m.get("mean", np.nan))
                lo.append(m.get("mean", np.nan) - m.get("lo95", np.nan))
                hi.append(m.get("hi95", np.nan) - m.get("mean", np.nan))
                dp = e.get("delta_perception")
                gray.append((m.get("mean", np.nan) - dp["diff"]) if isinstance(dp, dict) else np.nan)
        if not corpus:
            ax.errorbar(xs, real, yerr=[lo, hi], fmt="o", color=C_REAL, capsize=3,
                        label="real image", zorder=3)
        else:
            ax.plot(xs, real, "o", color=C_REAL, label="real image", zorder=3)
        ax.plot(xs, gray, "o", mfc="none", color=C_GRAY, label="gray-128", zorder=3)
        fl = hb["rows"].get("matched_dw", {})
        ce = hb["rows"].get("pg1", {})
        fv = fl.get("cider") if corpus else fl.get("mean", {}).get("mean")
        cv = ce.get("cider") if corpus else ce.get("mean", {}).get("mean")
        if fv is not None and cv is not None:
            th = hb.get("theta_null", {}).get("theta_null")
            if th is not None:
                ax.axhspan(fv, fv + th * (cv - fv), color=C_NULL, alpha=0.30, zorder=0,
                           label=f"null band (θ={th:.3f})")
            ax.axhline(cv, color=C_CEIL, ls="--", lw=1, label="CEILING (PG1)")
            ax.axhline(fv, color=C_FLOOR, ls=":", lw=1, label="FLOOR (matched-ΔW)")
        ax.set_xticks(xs)
        ax.set_xticklabels([ROW_LABEL.get(r, r) for r in rows], fontsize=6, rotation=45, ha="right")
        pw = hb.get("G3_power", {})
        ax.set_title(f"{TASK_LABEL.get(task, task)}\nG3 power {'PASS' if pw.get('passes') else 'FAIL'}"
                     f" (gap {pw.get('gap', float('nan')):.3f})", fontsize=8)
        ax.set_ylabel("raw score")
        if j == 0:
            ax.legend(fontsize=6, loc="upper left")
        # ---- lower panel: S with CI + the null band -------------------------------------
        ax2 = axes[1][j]
        srows = [r for r in rows if "S" in hb["rows"][r]]
        sx = np.arange(len(srows))
        sp = [hb["rows"][r]["S"]["S_point"] for r in srows]
        slo = [sp[i] - hb["rows"][r]["S"]["S_lo95"] for i, r in enumerate(srows)]
        shi = [hb["rows"][r]["S"]["S_hi95"] - sp[i] for i, r in enumerate(srows)]
        ax2.errorbar(sx, sp, yerr=[slo, shi], fmt="s", color=C_REAL, capsize=3, zorder=3)
        th = hb.get("theta_null", {}).get("theta_null")
        if th is not None:
            ax2.axhspan(0, th, color=C_NULL, alpha=0.30, label=f"null band (θ={th:.3f})")
        ax2.axhline(0.5, color=C_CEIL, ls="--", lw=1, label="retained band (S ≥ 0.5)")
        for i, r in enumerate(srows):
            lab = hb["rows"][r].get("label", {}).get("label", "")
            if lab:
                ax2.annotate(lab, (sx[i], sp[i]), fontsize=5, rotation=90,
                             textcoords="offset points", xytext=(6, 2))
        ax2.set_xticks(sx)
        ax2.set_xticklabels([ROW_LABEL.get(r, r) for r in srows], fontsize=6, rotation=45, ha="right")
        ax2.set_ylabel("S = (score − FLOOR)/(CEILING − FLOOR)")
        ax2.set_ylim(-0.1, 1.1)
        if j == 0:
            ax2.legend(fontsize=6, loc="upper left")
    _footer(axes[1][0], prov)
    fig.suptitle(f"F1 THE LADDER — every row × every task, {half} half "
                 "(complete sweep: every measured point shown)", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save(fig, out, "F1_ladder")
    plt.close(fig)
    return "F1_ladder"


def fig_perception(agg, out, prov, half):
    import matplotlib.pyplot as plt
    import numpy as np
    tasks = [t for t in ("vqav2", "textvqa", "refcoco")
             if agg["per_task"].get(t, {}).get("halves", {}).get(half, {}).get("rows")]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.8 * len(tasks), 3.8), squeeze=False)
    for j, task in enumerate(tasks):
        hb = agg["per_task"][task]["halves"][half]
        rows = [r for r in ROW_ORDER if r in hb["rows"]
                and isinstance(hb["rows"][r].get("delta_perception"), dict)]
        xs = np.arange(len(rows))
        d = [hb["rows"][r]["delta_perception"]["diff"] for r in rows]
        lo = [d[i] - hb["rows"][r]["delta_perception"]["lo95"] for i, r in enumerate(rows)]
        hi = [hb["rows"][r]["delta_perception"]["hi95"] - d[i] for i, r in enumerate(rows)]
        ax = axes[0][j]
        ax.errorbar(xs, d, yerr=[lo, hi], fmt="o", color=C_REAL, capsize=3)
        ax.axhline(0, color="#444", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([ROW_LABEL.get(r, r) for r in rows], fontsize=6, rotation=45, ha="right")
        ax.set_title(f"Δ_perception — {TASK_LABEL.get(task, task)}", fontsize=8)
        ax.set_ylabel("acc(real) − acc(gray-128), paired")
    _footer(axes[0][0], prov)
    fig.suptitle("F2 Δ_perception — does image information FLOW, independent of task score?",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    _save(fig, out, "F2_perception")
    plt.close(fig)
    return "F2_perception"


def fig_vision(byte_m, out, prov):
    import matplotlib.pyplot as plt
    import numpy as np
    r2a = (byte_m or {}).get("results", {}).get("2a")
    if not r2a:
        return None
    pols = sorted(r2a)
    fig, axes = plt.subplots(1, len(pols) + 1, figsize=(4.2 * (len(pols) + 1), 3.9), squeeze=False)
    for j, pol in enumerate(pols):
        blk = r2a[pol]
        ax = axes[0][j]
        prof = blk.get("profile", {})
        if not prof:
            ax.text(0.5, 0.5, f"{pol}: BYTE-FROZEN\nmax|Δ| = 0", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_axis_off()
            continue
        keys = sorted(prof, key=lambda k: (int(k.split("layers.")[1].split(".")[0])
                                           if "layers." in k else -1, k))
        rel = [prof[k]["rel_frob"] for k in keys]
        ov = [prof[k]["left"]["overlap"] for k in keys]
        nl = [prof[k]["left"]["null_p95"] for k in keys]
        xs = np.arange(len(keys))
        ax.plot(xs, rel, ".", ms=3, color=C_FLOOR, label="‖Δ‖/‖W‖")
        ax.set_ylabel("‖Δ‖/‖W‖", color=C_FLOOR)
        axr = ax.twinx()
        axr.plot(xs, ov, ".", ms=3, color=C_REAL, label="top-k overlap")
        axr.plot(xs, nl, "-", lw=0.8, color=C_NULL, label="matched-rotation null p95")
        axr.set_ylabel("subspace overlap", color=C_REAL)
        ax.set_xlabel("vision-tower matrix (depth-ordered)")
        ax.set_title(f"{pol} vs PG1 vision — max|Δ| = {blk['global_max_abs_delta']:.4g}\n"
                     f"byte-frozen = {blk['byte_frozen']} · preserved = {blk.get('preserved')}",
                     fontsize=8)
        axr.legend(fontsize=6, loc="lower right")
    ax = axes[0][-1]
    r2b = (byte_m or {}).get("results", {}).get("2b")
    if r2b:
        ax.bar([0, 1], [r2b["median_left_overlap"], r2b["median_null_p95"]],
               color=[C_REAL, C_NULL])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["PG1 vs SigLIP\nrelease", "matched-rotation\nnull p95"],
                                                  fontsize=7)
        ax.set_ylabel("median top-k overlap")
        ax.set_title(f"2b (ASSUMPTION-GATED EXPLORATORY)\nrel‖Δ‖ med = {r2b['median_rel_frob']:.4f}",
                     fontsize=8)
    else:
        ax.set_axis_off()
    _footer(axes[0][0], prov)
    fig.suptitle("F3 vision tower across the VLA hop — byte verdict + depth profile vs the null",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    _save(fig, out, "F3_vision")
    plt.close(fig)
    return "F3_vision"


def fig_lm_census(byte_m, geom_m, out, prov):
    import matplotlib.pyplot as plt
    import numpy as np
    r2c = (byte_m or {}).get("results", {}).get("2c")
    r2e = (byte_m or {}).get("results", {}).get("2e", {})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    if r2c:
        per = [r for r in r2c["per_matrix"] if r.get("srank_delta") is not None]
        xs = np.arange(len(per))
        axes[0].plot(xs, [r["srank_delta"] for r in per], ".", ms=4, color=C_REAL, label="srank(ΔW)")
        axes[0].fill_between(xs, [r["srank_null"]["p2_5"] for r in per],
                             [r["srank_null"]["p97_5"] for r in per], color=C_NULL, alpha=0.35,
                             label="Frobenius-matched random band")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("LM matrix (depth × type)")
        axes[0].set_ylabel("stable rank")
        axes[0].set_title(f"2c gemma-2b → PG1 ΔW\nmedian {r2c['median_srank_delta']:.1f} · "
                          f"full-rank-like = {r2c['fullrank_like']}", fontsize=8)
        axes[0].legend(fontsize=6)
    for i, (tag, blk) in enumerate(sorted(r2e.items())[:1]):
        per = [r for r in blk["per_matrix"] if r.get("raw_out_of_span") is not None]
        axes[1].plot(np.arange(len(per)), [r["rel_frob"] for r in per], ".", ms=4, color=C_FLOOR,
                     label="‖Δ‖/‖W‖")
        ax2 = axes[1].twinx()
        ax2.plot(np.arange(len(per)), [r["raw_out_of_span"] for r in per], ".", ms=4,
                 color=C_REAL, label="RAW out-of-span")
        axes[1].set_title(f"2e {tag} (EXPLORATORY, RAW)\nsrank med "
                          f"{blk['median_srank_delta']:.1f}", fontsize=8)
        axes[1].set_xlabel("LM matrix (depth × type)")
        axes[1].set_ylabel("‖Δ‖/‖W‖", color=C_FLOOR)
        ax2.set_ylabel("RAW out-of-span fraction", color=C_REAL)
    if geom_m and geom_m.get("per_layer"):
        pl = geom_m["per_layer"]
        Ls = sorted(pl, key=int)
        xs = [int(L) for L in Ls]
        axes[2].plot(xs, [pl[L]["overlap"] for L in Ls], "o-", ms=3, color=C_REAL, label="base↔PG1")
        # v2 of the 2d job renamed `ceiling` → `ceiling_base` (it also banks PG1's own
        # self-overlap as a second, independent ceiling); accept either spelling.
        _ceil = [pl[L].get("ceiling_base", pl[L].get("ceiling")) for L in Ls]
        axes[2].plot(xs, _ceil, "^--", ms=3, color=C_CEIL, label="CEILING (base split-half)")
        if all("ceiling_pg1" in pl[L] for L in Ls):
            axes[2].plot(xs, [pl[L]["ceiling_pg1"] for L in Ls], "^:", ms=2, color="#666",
                         label="2nd ceiling (PG1 self-overlap)")
        axes[2].plot(xs, [pl[L]["floor"] for L in Ls], "v:", ms=3, color=C_FLOOR,
                     label="FLOOR (window-permuted)")
        pooled = geom_m.get("pooled", {})
        axes[2].set_title(f"2d text geometry per layer\npooled P = {pooled.get('P', float('nan')):.3f} "
                          f"({pooled.get('P_label')})", fontsize=8)
        axes[2].set_xlabel("decoder layer")
        axes[2].set_ylabel("top-k activation-subspace overlap")
        axes[2].legend(fontsize=6)
    _footer(axes[0], prov)
    fig.suptitle("F4 weight + geometry census — the gemma→PG1 install and the VLA-hop ΔW",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    _save(fig, out, "F4_lm_census")
    plt.close(fig)
    return "F4_lm_census"


def fig_arm3(arm3, out, prov):
    import matplotlib.pyplot as plt
    if not arm3 or not arm3.get("cells"):
        return None
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"U1": C_REAL, "U2": C_CEIL, "U3": C_GRAY, "U4": C_FLOOR}
    for u in sorted({c["unit"] for c in arm3["cells"]}):
        pts = sorted([c for c in arm3["cells"] if c["unit"] == u], key=lambda c: c["steps"])
        ax.plot([p["steps"] for p in pts], [p["pc_success"] for p in pts], "o-",
                color=colors.get(u, "#444"), label=u)
        for p in pts:
            if p.get("ci95"):
                ax.errorbar([p["steps"]], [p["pc_success"]],
                            yerr=[[p["pc_success"] - p["ci95"][0]], [p["ci95"][1] - p["pc_success"]]],
                            fmt="none", ecolor=colors.get(u, "#444"), capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("fine-tuning steps"); ax.set_ylabel("pc_success (LIBERO-Spatial)")
    ax.set_title("F5 Arm-3 control savings — every unit × budget × seed", fontsize=9)
    ax.legend(fontsize=7)
    _footer(ax, prov)
    fig.tight_layout()
    _save(fig, out, "F5_arm3")
    plt.close(fig)
    return "F5_arm3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg-prefix", default="results/lacuna_agg")
    ap.add_argument("--bytes-prefix", default="results/lacuna_a2_bytes")
    ap.add_argument("--geom-prefix", default="results/lacuna_a2_geom")
    ap.add_argument("--arm3-prefix", default="")
    ap.add_argument("--half", default="confirmation")
    ap.add_argument("--sha", default="")
    ap.add_argument("--work", default="/work/figs")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")

    agg = json.loads((_pull(a.agg_prefix, pathlib.Path(a.work) / "agg") / "metrics.json").read_text())
    byte_m = json.loads((_pull(a.bytes_prefix, pathlib.Path(a.work) / "b") / "metrics.json").read_text())
    geom_p = _pull(a.geom_prefix, pathlib.Path(a.work) / "g") / "metrics.json"
    geom_m = json.loads(geom_p.read_text()) if geom_p.exists() else None
    arm3 = None
    if a.arm3_prefix:
        p = _pull(a.arm3_prefix, pathlib.Path(a.work) / "a3") / "metrics.json"
        arm3 = json.loads(p.read_text()) if p.exists() else None

    prov = (f"LACUNA · {a.agg_prefix} · half={a.half} · sha={a.sha or 'n/a'} · "
            f"tag lacuna-freeze-program-2026-07-28 · every point recomputed from banked per-item scores")
    made = [f for f in (fig_ladder(agg["results"], out, prov, a.half),
                        fig_perception(agg["results"], out, prov, a.half),
                        fig_vision(byte_m, out, prov),
                        fig_lm_census(byte_m, geom_m, out, prov),
                        fig_arm3(arm3, out, prov)) if f]
    (out / "figures.json").write_text(json.dumps({"figures": made, "provenance": prov}, indent=2))
    print("LACUNA_FIGS_OK", json.dumps(made), flush=True)


if __name__ == "__main__":
    main()
