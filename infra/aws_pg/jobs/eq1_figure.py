"""E-Q1 trained-adapter ladder figures (GraftLM trained-exploit prereg §10).

fig_trained_vs_noretrain — RefCOCO R_cap vs rank for the TRAINED topk adapter (seed 0; seed-1 point at
  rank-512 shown), over the no-retrain baseline (= eval-at-init, ~0 ≤512), FLOOR=0 / CEILING=1, and the
  random-init control (R_cap 0 at 512 & 1024 -> the ΔW-subspace is necessary). Title states the verdict.
fig_loc_emission — <loc>-emission fraction vs rank (the policy flip; onset ~512, robust by 1024).

Reads the per-cell eq1_trained_adapter.json artifacts from S3 (CONFIRMATORY, frozen
graftlm-trained-exploit-prereg). GPU-independent; --no-weights. PNG + PDF. No try/except.

    pg run infra.aws_pg.jobs.eq1_figure --no-weights --type c7i.xlarge --s3-prefix results/eq1_figs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
# rank -> S3 prefix for the seed-0 topk cells (+ the rank-1024 positive control)
TOPK_S0 = {8: "eq1_r8_topk_s0", 32: "eq1_r32_topk_s0", 128: "eq1_mde128_s0",
           256: "eq1_r256_topk_s0", 512: "eq1_r512_topk_s0", 1024: "eq1_pc3b"}
RAND = {512: "eq1_r512_rand_s0", 1024: "eq1_pc_rand3"}     # random-init control
S1 = {512: "eq1_r512_topk_s1"}                              # seed-1 stability at the threshold rank

C_TRAIN = "#0072B2"   # blue  — trained topk R_cap
C_NORET = "#555555"   # gray  — no-retrain (= eval-at-init)
C_RAND = "#D55E00"    # vermillion — random-init control
C_EMIT = "#E69F00"    # amber — <loc>-emission
C_CEIL = "#009E73"    # green — CEILING


def _fetch_json(prefix):
    dst = f"/tmp/{prefix.replace('/', '_')}.json"
    r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/results/{prefix}/out/eq1_trained_adapter.json",
                        dst, "--region", "us-east-1"], capture_output=True, text=True)
    assert r.returncode == 0, f"fetch failed for {prefix}: {r.stderr}"
    return json.loads(pathlib.Path(dst).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sorted(TOPK_S0)
    cells = {r: _fetch_json(TOPK_S0[r]) for r in ranks}
    rand = {r: _fetch_json(RAND[r]) for r in RAND}
    s1 = {r: _fetch_json(S1[r]) for r in S1}

    def tr(c, k):     # trained-eval field for refcoco
        return c["trained_eval"]["refcoco"][k]

    def ini(c, k):    # init-eval (= no-retrain) field
        return c["init_eval"]["refcoco"][k]

    rcap = [tr(cells[r], "R_cap") for r in ranks]
    emit = [tr(cells[r], "emit") for r in ranks]
    noret = [ini(cells[r], "R_cap") for r in ranks]              # eval-at-init = no-retrain rank-k edit

    # ---- fig_trained_vs_noretrain ----
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axhline(1.0, color=C_CEIL, ls="--", lw=1.2, label="CEILING (PG2-pt) = 1")
    ax.axhline(0.0, color="#888", ls=":", lw=1, label="FLOOR (base) = 0")
    ax.plot(ranks, noret, "--s", color=C_NORET, lw=1.8, ms=6, label="no-retrain SVD edit (= train init)")
    ax.plot(ranks, rcap, "-o", color=C_TRAIN, lw=2.4, ms=7, label="TRAINED topk adapter (seed 0)")
    # seed-1 at the threshold rank (show instability)
    for r, c in s1.items():
        ax.plot([r], [tr(c, "R_cap")], "D", color=C_TRAIN, ms=8, mfc="white", mec=C_TRAIN,
                label="TRAINED topk (seed 1) — threshold rank")
    # random-init control
    rr = sorted(rand)
    ax.plot(rr, [tr(rand[r], "R_cap") for r in rr], "^", color=C_RAND, ms=9,
            label="random-init control (R_cap 0)")
    ax.annotate("onset ~rank-512\n(seed-unstable: s0 emit 54% / s1 22%)", xy=(512, rcap[ranks.index(512)]),
                xytext=(40, 0.55), fontsize=8.6, arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank-k per decoder matrix (log₂)")
    ax.set_ylabel("RefCOCO R_cap (recovery fraction)")
    ax.set_ylim(-0.08, 1.12)
    ax.set_title("E-Q1: trained low-rank adapter installs the modality capability — H_threshold\n"
                 "graftable WITH training in the ΔW-subspace; onset ~512 (seed-unstable), robust by ~eff-rank "
                 "1024; ≤256 dead; random-init = 0 (subspace necessary)", fontsize=9.8)
    ax.legend(fontsize=7.6, loc="center left")
    ax.grid(alpha=0.25)
    prov = (f"topk seed0 {list(TOPK_S0.values())} · random {list(RAND.values())} · seed1 {list(S1.values())} "
            f"· s3://{BUCKET}/results/* · CONFIRMATORY graftlm-trained-exploit-prereg · RefCOCO testA N=1975")
    fig.text(0.005, 0.002, prov, fontsize=4.6, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_trained_vs_noretrain.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- fig_loc_emission ----
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.axhline(0.5, color="#aaa", ls="--", lw=1, label="policy-flip threshold (50%)")
    ax.plot(ranks, emit, "-^", color=C_EMIT, lw=2.2, ms=7, label="TRAINED topk <loc>-emission (seed 0)")
    for r, c in s1.items():
        ax.plot([r], [tr(c, "emit")], "D", color=C_EMIT, ms=8, mfc="white", mec=C_EMIT,
                label="seed 1 (threshold rank)")
    ax.annotate("emission flips on\nat rank-512 (seed-dependent)", xy=(512, emit[ranks.index(512)]),
                xytext=(40, 0.7), fontsize=8.6, arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank-k per decoder matrix (log₂)")
    ax.set_ylabel("<loc>-emission fraction (RefCOCO)")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("E-Q1: the <loc> generation-policy flip is a mid-rank phenomenon (onset ~512)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.text(0.005, 0.002, prov, fontsize=4.6, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_loc_emission.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    idx = {"figures": ["fig_trained_vs_noretrain", "fig_loc_emission"], "tag": "CONFIRMATORY",
           "verdict": "H_threshold; onset ~512 (seed-unstable), robust by 1024; random=0 (subspace necessary)",
           "ladder": {r: {"R_cap": rcap[i], "emit": emit[i], "no_retrain_R_cap": noret[i]}
                      for i, r in enumerate(ranks)},
           "seed1_512": {"R_cap": tr(s1[512], "R_cap"), "emit": tr(s1[512], "emit")},
           "random": {r: tr(rand[r], "R_cap") for r in rr}}
    (out / "fig_index.json").write_text(json.dumps(idx, indent=2))
    print("EQ1_FIGS_OK", json.dumps(idx, indent=2))


if __name__ == "__main__":
    main()
