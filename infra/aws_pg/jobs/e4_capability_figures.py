"""E4 capability-readout figure suite (GraftLM prereg §10).

Renders the headline of the capability readout: does the no-retrain rank-k edit's behavioural
KL-closure R_KL(k) (E4, prompt tokens) correspond to recovery of zero-shot TASK CAPABILITY R_cap(k)?
Both curves are on the SAME k-grid and the SAME base->pt rank-k edit family, so they overlay directly
as recovery fractions in [0,1].

  fig_kl_vs_capability  (headline) — per task: R_cap(k) (this sweep, CI band) vs R_KL(k) (E4
                        e4_basept_v2, CI band), FLOOR=0 / CEILING=1, the E_scale/N3 null band, and the
                        rank-8 head annotated. Title states the per-task verdict (cliff / prop / lead).
  fig_head_bar          — rank-8 head: R_KL vs R_cap per task, CIs + the frozen delta band.

R_cap from s3://.../results/e4_capreadout_sweep (frozen, CONFIRMATORY). R_KL from
s3://.../results/e4_basept_v2 (E4, EXPLORATORY). GPU-independent; --no-weights. PNG + PDF.

    pg run infra.aws_pg.jobs.e4_capability_figures --no-weights --type c7i.xlarge \
        --s3-prefix results/e4_capreadout_figs

No try/except — fail loud.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"
SWEEP_KEY = "results/e4_capreadout_sweep/out/e4_capability_sweep.json"
RKL_KEY = "results/e4_basept_v2/out/e4_compress_curve.json"   # E4 install-axis R_KL (prompt tokens)
ANSWERKL_KEY = "results/e4_capreadout_answerkl/out/e4_capability_answerkl.json"  # answer-token R_KL bridge
DELTA = 0.10                                                  # frozen verdict margin (prereg §6)

C_CAP = "#0072B2"    # blue — capability recovery R_cap
C_KL = "#555555"     # gray — behavioural KL closure R_KL (E4, prompt tokens)
C_AKL = "#E69F00"    # amber — answer-token R_KL (bridge)
C_NULL = "#D55E00"   # vermillion — E_scale / N3 nulls
C_CEIL = "#009E73"   # green — CEILING


def _fetch(key, dst):
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--region", "us-east-1"],
                   check=True)


def _verdict(rcap, rcap_ci, rkl):
    """Per-task rank-8 verdict from the frozen rule (prereg §6): compare the R_cap CI to R_KL +- delta.
    Conservative single-readout call (the headline confirmatory verdict uses the paired (R_cap-R_KL)
    bootstrap; this labels the figure)."""
    if rcap_ci is None:
        return "n/a"
    lo, hi = rcap_ci["lo"], rcap_ci["hi"]
    if hi < rkl - DELTA:
        return f"H_cliff (capability lags KL: R_cap CI<{rkl:.2f}-δ)"
    if lo > rkl + DELTA:
        return f"H_lead (capability leads KL: R_cap CI>{rkl:.2f}+δ)"
    if lo >= rkl - DELTA and hi <= rkl + DELTA:
        return f"H_prop (R_cap≈R_KL≈{rkl:.2f} within δ)"
    return "inconclusive (CI straddles δ)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep-key", default=SWEEP_KEY)
    ap.add_argument("--rkl-key", default=RKL_KEY)
    ap.add_argument("--answerkl-key", default=ANSWERKL_KEY)
    ap.add_argument("--no-answerkl", action="store_true", help="skip Fig.4 (answer-token bridge)")
    ap.add_argument("--freeze-tag", default="graftlm-capreadout-prereg")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    _fetch(a.sweep_key, out / "sweep.json")
    _fetch(a.rkl_key, out / "rkl.json")
    sweep = json.loads((out / "sweep.json").read_text())
    rkl = json.loads((out / "rkl.json").read_text())
    answerkl = None
    if not a.no_answerkl:
        _fetch(a.answerkl_key, out / "answerkl.json")
        answerkl = json.loads((out / "answerkl.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ks = sweep["config"]["k_sweep"]
    n = sweep["config"]["n"]
    rkl_curve = {int(k): rkl["curve"][str(k)] for k in ks}          # E4 R_KL(k), same grid
    tasks = list(sweep["tasks"])

    # ---- fig_kl_vs_capability (headline) --------------------------------------------------------
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.6 * len(tasks), 5.4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        cf = sweep["tasks"][t]["configs"]
        rcap = [cf[f"rank{k}"]["R_cap"] for k in ks]
        rcap_lo = [cf[f"rank{k}"]["R_cap_ci"]["lo"] if cf[f"rank{k}"]["R_cap_ci"] else cf[f"rank{k}"]["R_cap"] for k in ks]
        rcap_hi = [cf[f"rank{k}"]["R_cap_ci"]["hi"] if cf[f"rank{k}"]["R_cap_ci"] else cf[f"rank{k}"]["R_cap"] for k in ks]
        rk = [rkl_curve[k]["R_KL"] for k in ks]
        rk_lo = [rkl_curve[k]["R_KL_ci"]["lo"] for k in ks]
        rk_hi = [rkl_curve[k]["R_KL_ci"]["hi"] for k in ks]
        null = max(abs(cf["E_scale"]["R_cap"]),
                   abs(cf.get("N3_rank8", cf.get("N3_rank128"))["R_cap"]))

        ax.fill_between(ks, rk_lo, rk_hi, color=C_KL, alpha=0.15)
        ax.plot(ks, rk, "--s", color=C_KL, lw=2, ms=6, label="R_KL(k) — behavioural KL closure (E4)")
        ax.fill_between(ks, rcap_lo, rcap_hi, color=C_CAP, alpha=0.18)
        ax.plot(ks, rcap, "-o", color=C_CAP, lw=2.4, ms=7, label="R_cap(k) — task-capability recovery")
        ax.axhline(1.0, color=C_CEIL, ls="--", lw=1.2, label="CEILING (PG2-pt) = 1")
        ax.axhline(0.0, color="#888", ls=":", lw=1, label="FLOOR (base) = 0")
        ax.axhspan(-abs(null), abs(null), color=C_NULL, alpha=0.10,
                   label=f"E_scale/N3 null band (≤{null:.02f})")

        rcap8 = cf["rank8"]["R_cap"]
        rkl8 = rkl_curve[8]["R_KL"]
        verdict = _verdict(rcap8, cf["rank8"]["R_cap_ci"], rkl8)
        ax.annotate(f"rank-8 head\nR_KL={rkl8:.2f}\nR_cap={rcap8:.2f}",
                    xy=(8, rcap8), xytext=(8 * 1.5, max(rcap8, 0.0) + 0.30), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#444"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("rank-k per decoder matrix (log₂)")
        ax.set_ylabel("recovery fraction")
        ax.set_ylim(-0.12, 1.12)
        ax.set_title(f"{t} (acc@0.5 / ANLS, N={n})\n{verdict}", fontsize=10.5)
        ax.legend(fontsize=7.2, loc="center right")
        ax.grid(alpha=0.25)
    fig.suptitle("Does cheap KL-closure buy capability? R_cap(k) vs R_KL(k) on the SAME base→pt "
                 "rank-k edit family\n(below R_KL = the cheap low-rank head is 'cheap talk'; "
                 "on R_KL = KL is a faithful capability proxy)", fontsize=11.5, y=1.04)
    prov = (f"R_cap s3://{BUCKET}/{a.sweep_key} (CONFIRMATORY, {a.freeze_tag}) · "
            f"R_KL s3://{BUCKET}/{a.rkl_key} (E4, EXPLORATORY) · paired bootstrap CI, N={n}/task")
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_kl_vs_capability.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- fig_head_bar (rank-8 head: R_KL vs R_cap per task) -------------------------------------
    fig, ax = plt.subplots(figsize=(1.8 + 2.0 * len(tasks), 5.0))
    x = np.arange(len(tasks))
    w = 0.36
    rkl8 = [rkl_curve[8]["R_KL"]] * len(tasks)
    rkl8_err = np.array([[rkl_curve[8]["R_KL"] - rkl_curve[8]["R_KL_ci"]["lo"]] * len(tasks),
                         [rkl_curve[8]["R_KL_ci"]["hi"] - rkl_curve[8]["R_KL"]] * len(tasks)])
    rcap8 = [sweep["tasks"][t]["configs"]["rank8"]["R_cap"] for t in tasks]
    rc_ci = [sweep["tasks"][t]["configs"]["rank8"]["R_cap_ci"] for t in tasks]
    rcap8_err = np.array([[rcap8[i] - (rc_ci[i]["lo"] if rc_ci[i] else rcap8[i]) for i in range(len(tasks))],
                          [(rc_ci[i]["hi"] if rc_ci[i] else rcap8[i]) - rcap8[i] for i in range(len(tasks))]])
    ax.bar(x - w / 2, rkl8, w, yerr=rkl8_err, capsize=4, color=C_KL, label="R_KL (KL closure, E4)")
    ax.bar(x + w / 2, rcap8, w, yerr=rcap8_err, capsize=4, color=C_CAP, label="R_cap (capability)")
    for i, t in enumerate(tasks):
        ax.fill_between([i - 0.5, i + 0.5], rkl_curve[8]["R_KL"] - DELTA, rkl_curve[8]["R_KL"] + DELTA,
                        color=C_KL, alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("rank-8 recovery fraction")
    ax.set_ylim(-0.05, 1.12)
    ax.axhline(0, color="#888", ls=":", lw=1)
    ax.set_title(f"Rank-8 head: does {rkl_curve[8]['R_KL']:.0%} KL-closure buy {rkl_curve[8]['R_KL']:.0%} "
                 f"capability?\n(gray band = R_KL ± δ={DELTA}; bar below it = capability lags)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    fig.text(0.005, 0.002, prov, fontsize=5, color="#666", ha="left", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_head_bar.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- fig_answertoken_kl (prereg Fig.4) — the same-token bridge ------------------------------
    figures = ["fig_kl_vs_capability", "fig_head_bar"]
    if answerkl is not None:
        fig, axes = plt.subplots(1, len(tasks), figsize=(6.6 * len(tasks), 5.4), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            cf = sweep["tasks"][t]["configs"]
            akl = answerkl["tasks"][t]["configs"]
            rcap = [cf[f"rank{k}"]["R_cap"] for k in ks]
            rk = [rkl_curve[k]["R_KL"] for k in ks]
            rak = [akl[f"rank{k}"]["R_KL_ans"] for k in ks]
            rak_lo = [akl[f"rank{k}"]["lo"] for k in ks]
            rak_hi = [akl[f"rank{k}"]["hi"] for k in ks]
            ax.plot(ks, rk, "--s", color=C_KL, lw=2, ms=6,
                    label="R_KL — prompt tokens (E4)")
            ax.fill_between(ks, rak_lo, rak_hi, color=C_AKL, alpha=0.18)
            ax.plot(ks, rak, "-.D", color=C_AKL, lw=2.2, ms=6,
                    label="R_KL — ANSWER tokens (bridge)")
            ax.plot(ks, rcap, "-o", color=C_CAP, lw=2.4, ms=7,
                    label="R_cap — task capability")
            ax.axhline(1.0, color=C_CEIL, ls="--", lw=1.2)
            ax.axhline(0.0, color="#888", ls=":", lw=1)
            rak8, rcap8 = akl["rank8"]["R_KL_ans"], cf["rank8"]["R_cap"]
            ax.annotate(f"rank-8: answer-KL closes {rak8:.0%}\nbut capability = {rcap8:.0%}",
                        xy=(8, rak8), xytext=(8 * 1.4, rak8 - 0.42), fontsize=8.5,
                        arrowprops=dict(arrowstyle="->", color="#444"))
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_xlabel("rank-k per decoder matrix (log₂)")
            ax.set_ylabel("recovery fraction")
            ax.set_ylim(-0.12, 1.12)
            ax.set_title(f"{t}: distributional closure ≫ capability on the SAME tokens", fontsize=10.5)
            ax.legend(fontsize=8, loc="center right")
            ax.grid(alpha=0.25)
        fig.suptitle("Fig.4 — Answer-token bridge: the cheap edit closes most of the KL gap even on the "
                     "ANSWER tokens, yet recovers little capability\n(kills the prompt-vs-answer "
                     "population-mismatch objection — the cliff is capability-vs-distribution, not token-set)",
                     fontsize=11, y=1.04)
        akl_prov = prov + f" · answer-R_KL s3://{BUCKET}/{a.answerkl_key}"
        fig.text(0.005, 0.002, akl_prov, fontsize=5, color="#666", ha="left", va="bottom")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_answertoken_kl.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        figures.append("fig_answertoken_kl")

    idx = {"figures": figures, "tag": "CONFIRMATORY",
           "sweep": a.sweep_key, "rkl": a.rkl_key, "freeze_tag": a.freeze_tag, "delta": DELTA,
           "rank8": {t: {"R_cap": sweep["tasks"][t]["configs"]["rank8"]["R_cap"],
                         "R_cap_ci": sweep["tasks"][t]["configs"]["rank8"]["R_cap_ci"],
                         "R_KL": rkl_curve[8]["R_KL"],
                         "verdict": _verdict(sweep["tasks"][t]["configs"]["rank8"]["R_cap"],
                                             sweep["tasks"][t]["configs"]["rank8"]["R_cap_ci"],
                                             rkl_curve[8]["R_KL"])}
                     for t in tasks}}
    (out / "fig_index.json").write_text(json.dumps(idx, indent=2))
    print("E4_CAP_FIGS_OK", json.dumps(idx, indent=2))


if __name__ == "__main__":
    main()
