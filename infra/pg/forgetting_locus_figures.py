"""Figures for forgetting-locus (prereg §10) — weight-free, reads the result JSON only.

    python3 infra/pg/forgetting_locus_figures.py \
        --json studies/lineage/paligemma1-pi0/results/forget_locus/forgetting_locus.json \
        --out  studies/lineage/paligemma1-pi0/results/forget_locus

F1  per-fine-tune whole-model general-ppl forgetting (the robust headline) with the q/k-recoverable
    SHARE annotated — VL fine-tuning catastrophically forgets general LM competence, and for the biggest
    forgetters / most tasks the majority is NOT recoverable by removing attention q/k weights.
F2  the locus test (honest): fraction of q/k-recoverable forgetting captured by U_unt vs matched-random
    vs bottom-σ, per valid fine-tune, at BOTH k=32 and k=128 — core-locus is NOT demonstrated (U_unt is
    not disproportionately efficient; the contrast is k-fragile and n=4, dominated by one outlier).

Provenance footer on each. PNG (review) + PDF (paper). Corrected post-critic framing (no "U_unt worse"
claim; bootstrap is uninformative, reported as such).
"""
import argparse
import hashlib
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_FORGET = "#D55E00"   # vermillion — forgetting
C_QK = "#0072B2"       # blue   — q/k-recoverable share
C_UNT = "#0072B2"      # blue   — U_unt
C_RAND = "#E69F00"     # orange — matched-random
C_BOTS = "#009E73"     # green  — bottom-σ


def _footer(fig, run_id, base_sha, ih):
    fig.text(0.5, 0.005, f"run={run_id} · base_sha={base_sha[:8]} · input={ih[:12]} · "
             f"CONFIRMATORY (prereg) · eager forward (base_ppl 22.43 = FUNC anchor)",
             ha="center", va="bottom", fontsize=6.5, color="#555555")


def fig1_forgetting(d, out, run_id, base_sha, ih):
    base = d["base_ppl"]
    sg = d["surgery"]
    fts = d["forgetting_surgered"]
    rows = []
    for r in fts:
        wm = d["whole_model_forgetting_precheck"][r]
        qk = sg[r]["rec_full_qk"]
        rows.append((r.split("ft-")[-1].replace("-224", ""), wm, max(qk, 0.0) / wm if wm > 0 else 0.0))
    rows.sort(key=lambda x: -x[1])
    names = [x[0] for x in rows]
    wm = [x[1] for x in rows]
    share = [x[2] for x in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(names))
    ax.bar(x, wm, color=C_FORGET, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("whole-model forgetting  Δppl(ft − base)  [log]")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    for xi, (w, s) in enumerate(zip(wm, share)):
        ax.annotate(f"q/k {100*s:.0f}%", (xi, w), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6.8, color=C_QK)
    ax.set_title(f"VL fine-tuning catastrophically forgets general LM competence; most of it is NOT in q/k\n"
                 f"How to read: bars = blank-image wikitext ppl rise over base ({base:.1f}); "
                 f"'q/k N%' = share recoverable by removing ALL attention q/k ΔW (small ⇒ forgetting is in "
                 f"the complement).", fontsize=9.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, run_id, base_sha, ih)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F1_forgetting_magnitude.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_locus(d, out, run_id, base_sha, ih):
    vk = d["verdict_by_k"]
    ks = sorted(vk.keys(), key=int)
    # valid ft = those present as rows (rec_full_qk >= forget_min); use k=128 rows for the ft set
    rows128 = {r["ft"]: r for r in vk[ks[-1]]["rows"]}
    fts = list(rows128.keys())
    names = [f.split("ft-")[-1].replace("-224", "") for f in fts]
    fig, axes = plt.subplots(1, len(ks), figsize=(11, 4.6), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    width = 0.26
    for ax, k in zip(axes, ks):
        rows = {r["ft"]: r for r in vk[k]["rows"]}
        xs = range(len(fts))
        fu = [rows[f]["frac_unt"] for f in fts]
        fr = [rows[f]["frac_rand"] for f in fts]
        fb = [rows[f]["frac_bots"] for f in fts]
        ax.bar([x - width for x in xs], fu, width, color=C_UNT, label="U_unt (invariant core)")
        ax.bar(list(xs), fr, width, color=C_RAND, label="matched-random")
        ax.bar([x + width for x in xs], fb, width, color=C_BOTS, label="bottom-σ")
        ax.axhline(0, color="#000000", lw=0.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        med = vk[k]["median_frac_unt"]
        ax.set_title(f"k={k}  (median frac U_unt={med:.3f} vs rand={vk[k]['median_frac_rand']:.3f} / "
                     f"bots={vk[k]['median_frac_bots']:.3f})", fontsize=8.5)
        if k == ks[0]:
            ax.set_ylabel("fraction of q/k-recoverable forgetting captured")
            ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Core-locus NOT demonstrated: U_unt is not disproportionately efficient at recovering forgetting\n"
                 "How to read: if the invariant core were the forgetting locus, blue would tower over orange/green. "
                 "It does not — and the ranking is k-fragile (n=4 valid ft, refcoco-seg outlier; bootstrap uninformative).",
                 fontsize=9.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _footer(fig, run_id, base_sha, ih)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F2_locus_test.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jp = pathlib.Path(a.json)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = jp.read_bytes()
    ih = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw)
    fig1_forgetting(d, out, "results/forget_locus_eager", d["base_sha"], ih)
    fig2_locus(d, out, "results/forget_locus_eager", d["base_sha"], ih)
    print(f"wrote F1_forgetting_magnitude + F2_locus_test (png+pdf) to {out} (input {ih[:12]})")
    print(f"verdict: {d['verdict'][:90]}...")


if __name__ == "__main__":
    main()
