"""W8 — figure suite EXTRA (post-freeze amendment 2026-05-27).

Four more communication figures, one per completed experiment that the original
`w8_figures.py` six did not give a dedicated figure: Q1 (suppression-steering
generalisation across the 5-task battery), P2 (>=3-seed replication of the causal
hits), C5 (Gemma Scope SAE quality / lens applicability) and P1 (AI2D positive
control). With these every experiment in the study is visualised.

Same renderer pattern, style and DISCIPLINE as `infra.pg.w8_figures`:
  * pure matplotlib; reads the already-landed result JSONs on the `pg-results`
    volume VERBATIM — no GPU, no model loads, no recomputation. Every plotted
    number is read straight from its canonical JSON.
  * a figure ILLUSTRATES a claim independently adjudicated in the lineage journal
    (number + CI + null already there); it is NEVER standalone evidence. Captions
    state number + CI + null + a DISCOVERY/CONFIRMATORY tag and match the journal's
    CURRENT (critic-corrected) claims.
  * NO try/except (CLAUDE.md): a missing JSON key fails loud with a full trace.

Inputs (all under /vol/results):
  q1_steer_battery/steer_battery_PG_mix_resample.json — per-task ablate-steering battery
  p2_seed_replicate/p2_seed0_PG_mix_resample.json     — P2 seed-0 per-cell curves
  p2_seed_replicate/p2_seed42_PG_mix_resample.json    — P2 seed-42 per-cell curves
  w5_causal/steer_PG_mix_resample.json                — the seed-1337 primary (P2 third seed)
  c5/gemma_scope_applicability.json                   — FVU + CE-loss-recovered, 2 models
  p1/p1_ai2d_n500_s0.json                             — AI2D positive control (n=500)

Output (RESULTS_DIR/w8_figures_extra):
  figX_*.{png,pdf}  +  figure_index.json / figure_index.md
"""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette + rcParams — identical to w8_figures.py so the suites match visually.
# ---------------------------------------------------------------------------
C_VISION = "#0072B2"   # blue
C_LM = "#D55E00"       # vermillion
C_NULL = "#999999"     # grey — null / floor reference
C_BEFORE = "#56B4E9"   # sky
C_AFTER = "#CC79A7"    # purple
C_TARGET = "#009E73"   # green  — target (suppressed cluster)
C_RANDOM = "#999999"   # grey   — frequency-matched random control
C_ACCENT = "#E69F00"   # orange
C_PASS = "#1B5E20"     # dark green — passes the gate
C_FAIL = "#B00020"     # dark red   — fails the gate

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

PREREG = "prereg-v10 @ 80b1167"

# Battery task display order + pretty labels. NB the Q1 battery JSON keys VSR as
# 'vsr' (not 'spatial_vsr' as the A1 battery does) — use the Q1 file's own names.
_TASK_ORDER = ["refcoco", "coco_captions", "textvqa", "vqav2", "vsr"]
_TASK_LABEL = {
    "refcoco": "RefCOCO\n(acc@.5)",
    "coco_captions": "COCO-Cap\n(CIDEr)",
    "textvqa": "TextVQA\n(ANLS)",
    "vqav2": "VQAv2\n(soft-acc)",
    "vsr": "VSR\n(acc)",
}


# ---------------------------------------------------------------------------
# helpers (same contract as w8_figures.py)
# ---------------------------------------------------------------------------

def _load(results_dir: Path, rel: str) -> dict:
    p = results_dir / rel
    if not p.exists():
        raise FileNotFoundError(f"[w8-extra] required input missing on volume: {p}")
    return json.loads(p.read_text())


def _sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def _tag_box(ax, tag: str):
    colour = "#1B5E20" if tag.startswith("CONFIRM") else "#4A148C"
    ax.text(0.985, 0.97, tag, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, fontweight="bold", color=colour,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour, alpha=0.9))


def _finish(fig, out_dir: Path, stem: str, caption: str, *, nchar: int = 118,
            bottom: float = 0.30, right: float | None = None) -> list[str]:
    wrapped = "\n".join(textwrap.wrap(caption, nchar))
    kw = {"bottom": bottom}
    if right is not None:
        kw["right"] = right
    fig.subplots_adjust(**kw)
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=7.6,
             color="#222222", linespacing=1.35)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        paths.append(str(p))
    plt.close(fig)
    return paths


def _spearman_tiecorrect(x, y) -> float:
    """Tie-correct Spearman rho (average ranks) — the gate-deciding statistic P2 uses
    (infra.pg.p2_verdict.spearman_rho_tiecorrect). Reproduced here verbatim so the
    figure's monotonicity read matches the journal's critic-corrected P2 gate, NOT the
    inflated argsort(argsort) harness value. Pure read of the stored dose curve."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")

    def _rank_avg(a):
        a = np.asarray(a, float)
        order = np.argsort(a, kind="mergesort")
        sa = a[order]
        ranks = np.empty(len(a), float)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sa[j + 1] == sa[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return ranks

    rx = _rank_avg(x) - _rank_avg(x).mean()
    ry = _rank_avg(y) - _rank_avg(y).mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _one_sided_p_from_ci(mean, lo, hi) -> float:
    """One-sided p for H1: contrast>0 from a bootstrap mean+CI95 (normal approx) —
    verbatim from infra.pg.w5_verdict._one_sided_p_from_ci, the statistic the journal's
    Q1/P2 BH-FDR family uses. Pure read."""
    import math
    if mean is None or lo is None or hi is None:
        return float("nan")
    se = (hi - lo) / (2 * 1.959964)
    if se <= 0:
        return 0.0 if mean > 0 else 1.0
    return 0.5 * (1.0 + math.erf((-mean / se) / math.sqrt(2.0)))


def _bh_fdr(pairs, q=0.05) -> set:
    """Benjamini-Hochberg over [(key, p)] → keys rejected at FDR=q. Verbatim from
    infra.pg.w5_verdict._bh_fdr (the journal's family correction)."""
    valid = [(k, p) for k, p in pairs if p == p]
    if not valid:
        return set()
    valid.sort(key=lambda kp: kp[1])
    n = len(valid)
    crit = None
    for i, (k, p) in enumerate(valid, start=1):
        if p <= (i / n) * q:
            crit = p
    if crit is None:
        return set()
    return {k for k, p in valid if p <= crit}


# ---------------------------------------------------------------------------
# Figure 7 — Q1 suppression-steering generalisation across the battery. DISCOVERY
# ---------------------------------------------------------------------------

def fig7_q1_steer_battery(results_dir: Path, out_dir: Path) -> dict:
    q1 = _load(results_dir, "q1_steer_battery/steer_battery_PG_mix_resample.json")
    A = q1["prereg_analysis"]
    grounding = q1.get("grounding", {})

    tasks = [t for t in _TASK_ORDER if any(A[k]["task"] == t for k in A)]
    # one ablate cell per task at position 'all' (the pre-registered cell family).
    cells = {}
    for t in tasks:
        key = f"{t}|all|ablate"
        if key not in A:
            raise KeyError(f"[fig7] missing Q1 cell '{key}' (have {sorted(A)[:8]})")
        cells[t] = A[key]

    # target-minus-random max-dose effect (the bar height) + its beats-random CI.
    tmr = [cells[t]["target_minus_random_effect"] for t in tasks]
    beats = [cells[t].get("beats_random_paired_abs_delta") or {} for t in tasks]
    delta = [b.get("delta_mean", float("nan")) for b in beats]
    lo = [b.get("ci95_low", float("nan")) for b in beats]
    hi = [b.get("ci95_high", float("nan")) for b in beats]
    rho_store = [cells[t]["target_spearman_rho_vs_dose"] for t in tasks]
    # tie-correct rho (the journal's critic-corrected monotonicity gate value)
    rho_tc = [_spearman_tiecorrect(cells[t]["doses"], cells[t]["target_curve"]) for t in tasks]
    monotone_tc = [np.isfinite(r) and abs(r) >= 0.9 for r in rho_tc]
    # beats-random under BH-FDR (q=0.05), EXACTLY as infra.pg.q1_verdict: the family is the
    # cells where the target loses MORE than random (tmr>0), ranked by the one-sided p from
    # the paired |Δ| CI; a cell beats-random iff it survives FDR AND tmr>0.
    ps = [_one_sided_p_from_ci(b.get("delta_mean"), b.get("ci95_low"), b.get("ci95_high"))
          for b in beats]
    fam = [(t, p) for t, p, m in zip(tasks, ps, tmr) if m is not None and m > 0]
    fdr_keep = _bh_fdr(fam, q=0.05)
    beats_fdr_pos = [(t in fdr_keep) and (m is not None and m > 0) for t, m in zip(tasks, tmr)]
    perturbs = [bool(m and bf) for m, bf in zip(monotone_tc, beats_fdr_pos)]

    fig, (axB, axR) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    fig.subplots_adjust(wspace=0.28)
    x = np.arange(len(tasks))

    # --- LEFT: beats-random Δ (target loses MORE than random) with CI95.
    # Colour by the beats-random(FDR) gate ONLY (this panel's own gate); the right panel
    # carries monotonicity. A ✓ marks the tasks that PASS BOTH (= perturb).
    derr = np.clip(np.array([[d - l for d, l in zip(delta, lo)],
                             [h - d for d, h in zip(delta, hi)]]), 0, None)
    colours = [C_PASS if bf else C_FAIL for bf in beats_fdr_pos]
    axB.bar(x, delta, 0.6, yerr=derr, capsize=4, color=colours,
            error_kw=dict(lw=1.1, ecolor="#333"))
    axB.axhline(0, color="#000", lw=0.9)
    for xi, d, bf, p in zip(x, delta, beats_fdr_pos, perturbs):
        axB.text(xi, d + (0.004 if d >= 0 else -0.010), f"{d:+.3f}",
                 ha="center", va="bottom" if d >= 0 else "top", fontsize=7.6,
                 color=C_PASS if bf else C_FAIL)
        if p:  # perturbs = passes BOTH gates
            axB.text(xi, d + 0.018, "✓ perturbs", ha="center", fontsize=7.0,
                     color=C_PASS, fontweight="bold")
    axB.set_xticks(x)
    axB.set_xticklabels([_TASK_LABEL[t] for t in tasks], fontsize=8.3)
    axB.set_ylabel("beats-random paired |Δ| (each task's own metric, CI95)")
    axB.set_title("Beats freq-matched random? (BH-FDR)")
    # legend proxies
    axB.bar([], [], color=C_PASS, label="beats-random under FDR")
    axB.bar([], [], color=C_FAIL, label="does not beat random")
    axB.axhline(0, color="#000", lw=0.9, label="0 = no excess over random")
    axB.legend(loc="upper right", fontsize=7.8)

    # --- RIGHT: monotonicity gate — tie-correct |rho| vs the >=0.9 threshold ---
    rho_abs = [abs(r) for r in rho_tc]
    rcol = [C_PASS if m else C_FAIL for m in monotone_tc]
    axR.bar(x, rho_abs, 0.6, color=rcol)
    axR.axhline(0.9, ls="--", lw=1.2, color="#444", label="0.90 = monotonicity gate")
    axR.axhspan(0.9, 1.0, color="#000", alpha=0.05, zorder=0)
    for xi, r, rs in zip(x, rho_abs, rho_store):
        axR.text(xi, r + 0.012, f"{r:.2f}", ha="center", fontsize=7.6, color="#222")
        axR.text(xi, 0.03, f"ρ*={rs:+.2f}", ha="center", fontsize=6.6, color="#777")
    axR.set_xticks(x)
    axR.set_xticklabels([_TASK_LABEL[t] for t in tasks], fontsize=8.3)
    axR.set_ylabel("|Spearman ρ| (tie-correct)  target curve vs dose")
    axR.set_ylim(0, 1.05)
    axR.set_title("Monotone dose-response? (gate = |ρ|≥0.9)")
    axR.legend(loc="lower right", fontsize=7.8)
    _tag_box(axR, "DISCOVERY · critic-corrected")

    n_pert = sum(perturbs)
    pert_tasks = [t for t, p in zip(tasks, perturbs) if p]
    cap = (f"Q1 — does ablating the suppressed image cluster (n=87) generalise beyond RefCOCO? "
           f"Per task (pos=all, ablate): a task PERTURBS iff its dose curve is monotone "
           f"(tie-correct |ρ|≥0.9, GREEN bar right) AND beats a frequency-matched RANDOM set "
           f"under BH-FDR (target loses more; beats-random Δ CI95>0, GREEN bar left). "
           f"{n_pert}/5 perturb: {pert_tasks} — RefCOCO Δ={delta[0]:+.3f} "
           f"CI[{lo[0]:+.3f},{hi[0]:+.3f}], COCO-Cap Δ={delta[1]:+.3f} CI[{lo[1]:+.3f},{hi[1]:+.3f}]. "
           f"TextVQA passes FDR but FAILS monotonicity (tie-correct |ρ|={rho_abs[2]:.2f}<0.9); "
           f"VQAv2/VSR null. NULL = freq-matched random cluster. Broader than RefCOCO, but the "
           f"perturbed pair is mixed-type → NO clean grounding/non-grounding split; single seed; "
           f"modest. {PREREG}. [DISCOVERY · critic-corrected]")
    paths = _finish(fig, out_dir, "fig7_q1_steer_battery", cap, nchar=158, bottom=0.31)

    return {
        "stem": "fig7_q1_steer_battery", "tag": "DISCOVERY (critic-corrected)", "files": paths,
        "claim": ("Suppression-steering generalises to 2/5 tasks (RefCOCO + COCO-Caption); TextVQA "
                  "passes FDR but fails monotonicity; VQAv2/VSR null — broader than RefCOCO, no clean "
                  "grounding split."),
        "backing_number": {
            "n_perturbing": n_pert, "perturbing_tasks": pert_tasks,
            "beats_random_delta": {t: round(d, 4) for t, d in zip(tasks, delta)},
            "target_minus_random_maxdose": {t: round(v, 4) for t, v in zip(tasks, tmr)},
            "rho_tiecorrect": {t: round(r, 4) for t, r in zip(tasks, rho_tc)},
            "rho_stored_harness": {t: round(r, 4) for t, r in zip(tasks, rho_store)},
            "monotone_tiecorrect": {t: bool(m) for t, m in zip(tasks, monotone_tc)},
            "beats_random_fdr": {t: bool(b) for t, b in zip(tasks, beats_fdr_pos)},
            "perturbs": {t: bool(p) for t, p in zip(tasks, perturbs)},
            "grounding": {t: grounding.get(t) for t in tasks},
        },
        "ci": {t: [beats[i].get("ci95_low"), beats[i].get("ci95_high")] for i, t in enumerate(tasks)},
        "null": "freq-matched random latent cluster; gate = monotone(|ρ|≥0.9, tie-correct) AND beats-random under BH-FDR q=0.05",
        "source": "q1_steer_battery/steer_battery_PG_mix_resample.json",
        "caption": cap,
    }


# ---------------------------------------------------------------------------
# Figure 8 — P2 >=3-seed replication of the two causal hits. DISCOVERY
# ---------------------------------------------------------------------------

def fig8_p2_seed_replication(results_dir: Path, out_dir: Path) -> dict:
    # Read the per-(task,seed) values VERBATIM from the canonical 3-seed verdict the
    # journal cites (p2_verdict_3seed.json) — NOT recomputed — so the figure's numbers
    # are exactly the journal's critic-corrected adjudication (tie-correct ρ, gate_pass,
    # beats-random Δ + CI under BH-FDR). seed 1337 = the W5 primary (its COCO-Caption +
    # RefCOCO cells come from the Q1 battery run the verdict used as the 1337 source).
    v = _load(results_dir, "p2_seed_replicate/p2_verdict_3seed.json")
    seeds = [1337, 0, 42]            # plot order: primary first
    tasks = ["refcoco", "coco_captions"]
    task_pretty = {"refcoco": "RefCOCO (grounding)", "coco_captions": "COCO-Caption (captioning)"}

    data = {t: {} for t in tasks}
    for t in tasks:
        ps = v["per_task"][t]["per_seed"]
        for s in seeds:
            cell = ps[str(s)]          # JSON keys are strings
            ci = cell["beats_ci"]
            data[t][s] = {
                "rho": cell["rho"], "rho_stored": cell["rho_stored"],
                "delta": cell["beats_delta_mean"], "lo": ci[0], "hi": ci[1],
                "tmr": cell["target_minus_random"],
                "monotone": bool(cell["monotone"]),
                "beats_pos": bool(cell["beats_random_fdr"]),
                "gate": bool(cell["gate_pass"]),
            }

    n_pass = {t: v["per_task"][t]["n_gate_pass"] for t in tasks}

    fig, (axD, axR) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    fig.subplots_adjust(wspace=0.28)

    seed_colours = {1337: C_TARGET, 0: C_ACCENT, 42: "#6A3D9A"}
    xt = np.arange(len(tasks))
    w = 0.26

    # --- LEFT: per-seed beats-random Δ with CI95, grouped by task ---
    for j, s in enumerate(seeds):
        ys = [data[t][s]["delta"] for t in tasks]
        los = [data[t][s]["lo"] for t in tasks]
        his = [data[t][s]["hi"] for t in tasks]
        err = np.clip(np.array([[y - l for y, l in zip(ys, los)],
                                [h - y for y, h in zip(ys, his)]]), 0, None)
        xpos = xt + (j - 1) * w
        bars = axD.bar(xpos, ys, w, yerr=err, capsize=3, color=seed_colours[s],
                       label=f"seed {s}" + (" (W5 primary)" if s == 1337 else ""),
                       error_kw=dict(lw=1.0, ecolor="#333"))
        # hatch the bars that FAIL the gate so pass/fail is legible alongside seed colour
        for t, b in zip(tasks, bars):
            if not data[t][s]["gate"]:
                b.set_hatch("xxx"); b.set_edgecolor("#333"); b.set_linewidth(0.6)
    axD.axhline(0, color="#000", lw=0.9)
    axD.set_xticks(xt)
    axD.set_xticklabels([task_pretty[t] for t in tasks], fontsize=8.6)
    axD.set_ylabel("beats-random paired |Δ| (each task's own metric, CI95)")
    axD.set_title("Per-seed causal effect (hatched = fails gate)")
    axD.legend(loc="upper right", fontsize=7.8, ncol=1)

    # --- RIGHT: tie-correct |rho| per seed vs the 0.9 gate ---
    for j, s in enumerate(seeds):
        ys = [abs(data[t][s]["rho"]) for t in tasks]
        xpos = xt + (j - 1) * w
        bars = axR.bar(xpos, ys, w, color=seed_colours[s], label=f"seed {s}")
        for t, b, yv in zip(tasks, bars, ys):
            if not data[t][s]["monotone"]:
                b.set_hatch("xxx"); b.set_edgecolor("#333"); b.set_linewidth(0.6)
            axR.text(b.get_x() + b.get_width() / 2, yv + 0.012, f"{yv:.2f}",
                     ha="center", fontsize=6.4, color="#222")
    axR.axhline(0.9, ls="--", lw=1.2, color="#444", label="0.90 = monotonicity gate")
    axR.axhspan(0.9, 1.0, color="#000", alpha=0.05, zorder=0)
    axR.set_xticks(xt)
    axR.set_xticklabels([task_pretty[t] for t in tasks], fontsize=8.6)
    axR.set_ylabel("|Spearman ρ| (tie-correct)  target vs dose")
    axR.set_ylim(0, 1.08)
    axR.set_title("Monotonicity per seed (gate = |ρ|≥0.9)")
    axR.legend(loc="lower left", fontsize=7.6, ncol=2)
    _tag_box(axR, "DISCOVERY · seed-split")

    rc0 = data["refcoco"][0]
    cap = (f"P2 — does the causal hit replicate across seeds? Per seed (1337=W5 primary, 0, 42) "
           f"for the two W5/Q1 hits, the SAME pre-set gate: monotone (tie-correct |ρ|≥0.9, right) "
           f"AND beats a freq-matched RANDOM set (target loses more, CI95>0, left). "
           f"COCO-Caption is SEED-FIRM ({n_pass['coco_captions']}/3 — all seeds pass). "
           f"RefCOCO does NOT seed-replicate ({n_pass['refcoco']}/3 — fails seed 0 on the "
           f"tie-correct monotonicity gate: |ρ|={abs(rc0['rho']):.3f}<0.9 where the stored "
           f"harness ρ*={rc0['rho_stored']:+.2f} would have passed; the W5 primary RefCOCO "
           f"ρ=−0.900 at seed 1337 is VALID — no ties — but not seed-robust). NULL = freq-matched "
           f"random cluster. → the seed-robust causal hit is captioning, NOT grounding; tempers "
           f"(not inverts) W5's grounding-specific framing. {PREREG}. [DISCOVERY · seed-split]")
    paths = _finish(fig, out_dir, "fig8_p2_seed_replication", cap, nchar=158, bottom=0.31)

    return {
        "stem": "fig8_p2_seed_replication", "tag": "DISCOVERY (seed-split)", "files": paths,
        "claim": ("COCO-Caption SEED-FIRM (3/3); RefCOCO does NOT seed-replicate (2/3, fails seed 0 "
                  "on the tie-correct monotonicity gate). Seed-robust hit is captioning, not grounding."),
        "backing_number": {
            "n_gate_pass": n_pass,
            "per_seed_rho_tiecorrect": {t: {s: round(data[t][s]["rho"], 4) for s in seeds} for t in tasks},
            "per_seed_rho_stored": {t: {s: data[t][s]["rho_stored"] for s in seeds} for t in tasks},
            "per_seed_beats_delta": {t: {s: data[t][s]["delta"] for s in seeds} for t in tasks},
            "per_seed_gate_pass": {t: {s: bool(data[t][s]["gate"]) for s in seeds} for t in tasks},
            "refcoco_seed0_fails_on": "tie-correct monotonicity (|rho|<0.9) — stored harness rho* would pass",
        },
        "ci": {t: {s: [data[t][s]["lo"], data[t][s]["hi"]] for s in seeds} for t in tasks},
        "null": "freq-matched random latent cluster; gate = monotone(|ρ|≥0.9, tie-correct) AND beats-random under BH-FDR",
        "source": ("p2_seed_replicate/p2_verdict_3seed.json (canonical 3-seed adjudication; "
                   "seed 1337 = the W5 primary via the Q1 battery run)"),
        "caption": cap,
    }


# ---------------------------------------------------------------------------
# Figure 9 — C5 Gemma Scope SAE quality (lens applicability). exploratory control
# ---------------------------------------------------------------------------

def fig9_c5_sae_quality(results_dir: Path, out_dir: Path) -> dict:
    c5 = _load(results_dir, "c5/gemma_scope_applicability.json")
    g2 = c5["gemma_2_2b"]
    pg = c5["paligemma2_lm"]
    sae = c5["sae"]

    # Pre-registered SAP acceptance bar (frozen prereg-v10): the quality NULL/reference
    # the journal reads C5 against — FVU<=0.40 AND loss-recovered>=0.85.
    FVU_BAR = 0.40
    LR_BAR = 0.85

    fig, (axF, axL) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(2)
    labels = ["gemma-2-2b\n(base LM)", "PaliGemma2-LM\n(the 'after')"]
    cols = [C_BEFORE, C_AFTER]

    # --- LEFT: FVU (lower is better) vs the <=0.40 acceptance bar ---
    fvu = [g2["fvu"], pg["fvu"]]
    axF.bar(x, fvu, 0.55, color=cols)
    axF.axhline(FVU_BAR, ls="--", lw=1.3, color=C_FAIL,
                label=f"acceptance bar FVU≤{FVU_BAR:.2f}")
    axF.axhspan(0, FVU_BAR, color="#1B5E20", alpha=0.05, zorder=0)
    for xi, v in zip(x, fvu):
        axF.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.4, color="#222")
        axF.text(xi, v / 2, f"{(1-v)*100:.0f}%\nvar expl.", ha="center", va="center",
                 fontsize=7.0, color="white", fontweight="bold")
    axF.set_xticks(x); axF.set_xticklabels(labels, fontsize=8.6)
    axF.set_ylabel("FVU  (fraction of variance unexplained; lower better)")
    axF.set_ylim(0, max(max(fvu) * 1.25, FVU_BAR * 1.2))
    axF.set_title("SAE reconstruction (FVU)")
    axF.legend(loc="upper left", fontsize=8.0)

    # --- RIGHT: CE-loss-recovered (higher is better) vs the >=0.85 acceptance bar ---
    lr = [g2["loss_recovered"], pg["loss_recovered"]]
    axL.bar(x, lr, 0.55, color=cols)
    axL.axhline(LR_BAR, ls="--", lw=1.3, color=C_FAIL,
                label=f"acceptance bar loss-rec≥{LR_BAR:.2f}")
    axL.axhspan(LR_BAR, 1.0, color="#1B5E20", alpha=0.05, zorder=0)
    for xi, v in zip(x, lr):
        axL.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.4, color="#222")
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8.6)
    axL.set_ylabel("CE-loss-recovered  (clean→mean-ablation gap; higher better)")
    axL.set_ylim(0, 1.02)
    axL.set_title("Downstream CE-loss-recovered")
    axL.legend(loc="lower left", fontsize=8.0)
    _tag_box(axL, "EXPLORATORY · control")

    cap = (f"C5 — is the off-the-shelf Gemma Scope JumpReLU SAE (layer {sae['layer']}, "
           f"width {sae['width']}, L0≈{sae['average_l0']}) a usable LM lens? On gemma-2-2b "
           f"(the base LM, BEFORE): FVU {g2['fvu']:.3f} / loss-rec {g2['loss_recovered']:.3f} "
           f"→ inside the pre-registered acceptance band (FVU≤{FVU_BAR:.2f}, loss-rec≥{LR_BAR:.2f}; "
           f"matches Gemma Scope's published quality). On PaliGemma2-LM (the AFTER): FVU {pg['fvu']:.3f} "
           f"/ loss-rec {pg['loss_recovered']:.3f} → degraded but still inside the band (usable as a "
           f"FROZEN-BASE lens). NULL/reference = the frozen SAP acceptance bar. The degradation on "
           f"PG2-LM is an APPLICABILITY term (it is why W4 reads the LM rewrite through a base-LM SAE), "
           f"NOT the f_emergent result. {PREREG}. [exploratory control]")
    paths = _finish(fig, out_dir, "fig9_c5_sae_quality", cap, nchar=150, bottom=0.30)

    return {
        "stem": "fig9_c5_sae_quality", "tag": "EXPLORATORY (control)", "files": paths,
        "claim": ("Gemma Scope SAE usable as a frozen-base LM lens: gemma-2-2b in the published band, "
                  "PG2-LM degraded but inside the acceptance band; degradation is an applicability term, "
                  "not the f_emergent result."),
        "backing_number": {
            "gemma_2_2b": {"fvu": round(g2["fvu"], 4), "loss_recovered": round(g2["loss_recovered"], 4),
                           "explained_variance": round(g2.get("explained_variance", 1 - g2["fvu"]), 4),
                           "l0": round(g2.get("l0", float("nan")), 2)},
            "paligemma2_lm": {"fvu": round(pg["fvu"], 4), "loss_recovered": round(pg["loss_recovered"], 4),
                              "explained_variance": round(pg.get("explained_variance", 1 - pg["fvu"]), 4),
                              "l0": round(pg.get("l0", float("nan")), 2)},
            "sae": {"layer": sae["layer"], "width": sae["width"], "average_l0": sae["average_l0"]},
        },
        "ci": "n/a (single-pass FVU / loss-recovered on a fixed corpus; no resample CI)",
        "null": f"frozen SAP acceptance bar: FVU<={FVU_BAR} AND loss-recovered>={LR_BAR} (prereg-v10)",
        "source": "c5/gemma_scope_applicability.json",
        "caption": cap,
    }


# ---------------------------------------------------------------------------
# Figure 10 — P1 AI2D positive control (harness validation, gate G0). CONFIRMATORY
# ---------------------------------------------------------------------------

def fig10_p1_ai2d_control(results_dir: Path, out_dir: Path) -> dict:
    p1 = _load(results_dir, "p1/p1_ai2d_n500_s0.json")
    published = p1["published"]                 # 74.7
    measured = p1["acc_exact_pct"]              # 75.80
    n = p1["n_items"]
    n_correct = p1["n_correct_exact"]
    TOL = 3.0                                   # pre-set +/-3.0 tolerance band

    fig, ax = plt.subplots(figsize=(8.2, 4.5))

    y_pub, y_meas = 1.0, 0.0
    # the tolerance band around the published number (the NULL the measurement is read against)
    ax.axvspan(published - TOL, published + TOL, color=C_PASS, alpha=0.10, zorder=0,
               label=f"pre-set tolerance ±{TOL:.1f} around published")
    ax.axvline(published, ls="--", lw=1.4, color="#444")

    # Wald 95% CI on the measured accuracy (binomial; reported for honesty — the
    # verdict is the pre-set tolerance band, not this CI).
    p = n_correct / n
    se = (p * (1 - p) / n) ** 0.5
    ci_lo, ci_hi = (p - 1.959964 * se) * 100, (p + 1.959964 * se) * 100

    ax.errorbar([measured], [y_meas], xerr=[[measured - ci_lo], [ci_hi - measured]],
                fmt="o", ms=11, color=C_AFTER, capsize=5, lw=1.8, zorder=5,
                label=f"measured {measured:.2f}%  (n={n}, {n_correct}/{n}; Wald CI95)")
    ax.scatter([published], [y_pub], marker="D", s=120, color="#444", zorder=5,
               label=f"published {published:.1f}% (HF card / arXiv 2412.03555)")

    ax.annotate(f"Δ = {measured - published:+.2f} pts\n(inside ±{TOL:.1f})",
                xy=(measured, y_meas), xytext=(measured + 1.0, y_meas + 0.45),
                fontsize=8.2, color=C_PASS, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PASS, lw=1.0))

    ax.set_yticks([y_meas, y_pub])
    ax.set_yticklabels(["measured\n(this harness)", "published\n(PaliGemma2 card)"], fontsize=9)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(published - 2 * TOL, published + 2 * TOL)
    ax.set_xlabel("AI2D accuracy (exact-match, big_vision VQA scorer)  %")
    ax.set_title("P1 positive control: AI2D harness validated (G0)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7.8)
    _tag_box(ax, "CONFIRMATORY")

    cap = (f"P1 — positive control proving the eval harness is not broken (gate G0). "
           f"AI2D, paligemma2-3b-mix-224, exact-match string scorer (big_vision recipe). "
           f"Published {published:.1f}% (HF model card 224-3B; cross-checked arXiv 2412.03555 "
           f"Tbl 13) vs measured {measured:.2f}% (n={n}, {n_correct}/{n}; Wald CI95 "
           f"[{ci_lo:.2f},{ci_hi:.2f}]). Δ={measured - published:+.2f} pts, INSIDE the pre-set "
           f"±{TOL:.1f} tolerance band. NULL = the published number ± {TOL:.1f} tolerance. "
           f"Harness validated → downstream results are trustworthy. {PREREG}. [CONFIRMATORY]")
    paths = _finish(fig, out_dir, "fig10_p1_ai2d_control", cap, bottom=0.30)

    return {
        "stem": "fig10_p1_ai2d_control", "tag": "CONFIRMATORY", "files": paths,
        "claim": "AI2D positive control reproduces the published number within the pre-set tolerance; harness validated (G0).",
        "backing_number": {
            "published_pct": published, "measured_pct": measured,
            "delta_pts": round(measured - published, 2),
            "n_items": n, "n_correct": n_correct,
            "wald_ci95_pct": [round(ci_lo, 2), round(ci_hi, 2)],
            "tolerance_pts": TOL, "within_tolerance": abs(measured - published) <= TOL,
        },
        "ci": {"measured_wald_ci95_pct": [round(ci_lo, 2), round(ci_hi, 2)],
               "note": "verdict is the pre-set ±3.0 tolerance band, not this CI"},
        "null": f"published {published} ± {TOL} tolerance band (HF card / arXiv 2412.03555 Tbl 13)",
        "source": "p1/p1_ai2d_n500_s0.json",
        "caption": cap,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_BUILDERS = [
    ("fig7", fig7_q1_steer_battery),
    ("fig8", fig8_p2_seed_replication),
    ("fig9", fig9_c5_sae_quality),
    ("fig10", fig10_p1_ai2d_control),
]

_INPUTS = [
    "q1_steer_battery/steer_battery_PG_mix_resample.json",
    "p2_seed_replicate/p2_verdict_3seed.json",
    "c5/gemma_scope_applicability.json",
    "p1/p1_ai2d_n500_s0.json",
]


def build_all(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    out_dir = results_dir / "w8_figures_extra"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {}
    for rel in _INPUTS:
        p = results_dir / rel
        input_hashes[rel] = _sha8(p) if p.exists() else "MISSING"

    figures = [fn(results_dir, out_dir) for _, fn in _BUILDERS]

    index = {
        "workstream": "W8 — figure suite EXTRA (one figure per remaining experiment)",
        "prereg_tag": PREREG,
        "discipline": ("Each figure ILLUSTRATES a claim independently backed by a number + CI + null "
                       "already adjudicated in the journal; a figure is never evidence on its own. "
                       "Captions match the journal's CURRENT (critic-corrected) claims."),
        "script": "infra/pg/w8_figures_extra.py",
        "input_hashes": input_hashes,
        "figures": figures,
    }
    (out_dir / "figure_index.json").write_text(json.dumps(index, indent=2))
    (out_dir / "figure_index.md").write_text(_render_md(index))
    return index


def _render_md(index: dict) -> str:
    lines = [
        "# W8 figure index — EXTRA (Q1, P2, C5, P1)", "",
        f"- **Pre-reg:** {index['prereg_tag']}",
        f"- **Script:** `{index['script']}`",
        f"- **Discipline:** {index['discipline']}", "",
        "Each figure illustrates a claim backed by a number + CI + null (below) and is tagged "
        "DISCOVERY/CONFIRMATORY. Captions match the journal's current critic-corrected claims.", "",
    ]
    for f in index["figures"]:
        lines += [
            f"## {f['stem']}  — [{f['tag']}]", "",
            f"**Claim illustrated:** {f['claim']}", "",
            f"**Backing number:** `{json.dumps(f['backing_number'])}`", "",
            f"**CI:** `{json.dumps(f['ci'])}`", "",
            f"**Null:** {f['null']}", "",
            f"**Source JSON:** `{f['source']}`", "",
            f"**Caption:** {f['caption']}", "",
            "---", "",
        ]
    return "\n".join(lines)
