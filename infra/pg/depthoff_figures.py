"""DEPTHOFF figures, drawn from `results/verdict.json` only.

Every panel that has no input is skipped and named in the report, so a session draws what has landed rather
than failing on a missing cell. Nothing here recomputes a number — the verdict file is the single source, so
a corrected verdict redraws every figure (CLAUDE.md provenance 4).
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

COND_ORDER = ("ON", "ON2", "ZERO", "NOGEN", "BROKEN")
COND_COLOUR = {"ON": "#1f4e79", "ON2": "#5b9bd5", "ZERO": "#c55a11", "NOGEN": "#7030a0",
               "BROKEN": "#808080"}
COND_LABEL = {"ON": "ON\n(as released)", "ON2": "ON2\n(identical re-run)",
              "ZERO": "ZERO\n(scalars forced to 0)", "NOGEN": "NOGEN\n(no depth block)",
              "BROKEN": "BROKEN\n(wrong image)"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def f1_rates(out: pathlib.Path, verdict: Dict) -> Optional[str]:
    """Pooled success per condition with Wilson intervals, the frozen margin drawn against ON."""
    rates = verdict.get("pooled_rates") or {}
    conds = [c for c in COND_ORDER if c in rates]
    if not conds:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = range(len(conds))
    vals = [rates[c]["rate"] * 100 for c in conds]
    los = [(rates[c]["rate"] - rates[c]["lo"]) * 100 for c in conds]
    his = [(rates[c]["hi"] - rates[c]["rate"]) * 100 for c in conds]
    ax.bar(xs, vals, yerr=[los, his], capsize=4,
           color=[COND_COLOUR[c] for c in conds], width=0.62)
    on = rates.get("ON", {}).get("rate")
    delta = ((verdict.get("decided") or {}).get("frozen") or {}).get("delta_points", 5.0)
    if on is not None:
        ax.axhline(on * 100, color="#1f4e79", lw=1.0, ls="--", zorder=0)
        ax.axhspan((on * 100) - delta, (on * 100) + delta, color="#1f4e79", alpha=0.10, zorder=0,
                   label=f"ON +/- {delta:g} pt margin (frozen)")
    for x, c in zip(xs, conds):
        ax.text(x, vals[list(conds).index(c)] + his[list(conds).index(c)] + 1.4,
                f"{rates[c]['k']}/{rates[c]['n']}", ha="center", fontsize=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=8.5)
    ax.set_ylabel("LIBERO success rate (%)")
    ax.set_ylim(0, 108)
    n = verdict.get("n_paired_episodes")
    ax.set_title(f"MolmoAct2-Think-LIBERO with the depth pathway removed\n"
                 f"{n} paired episodes, 4 suites, same task / init state / seed", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = out / "F1_rates.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def f2_paired_differences(out: pathlib.Path, verdict: Dict) -> Optional[str]:
    """The paired difference ON - X with its bootstrap interval, against the frozen margin."""
    arms = ((verdict.get("decided") or {}).get("arms") or {})
    if not arms:
        return None
    plt = _mpl()
    rows: List = []
    for name, payload in arms.items():
        iv = payload["rule1"]["diff_on_minus_arm"]
        rows.append((name, iv["point"], iv["lo"], iv["hi"], payload["rule1"]["verdict"]))
    # the noise floor, ON vs ON2, drawn as the reference the removal arms are judged against
    floor = (verdict.get("decided") or {}).get("gates", {}).get("G-floor")
    delta = ((verdict.get("decided") or {}).get("frozen") or {}).get("delta_points", 5.0)
    fig, ax = plt.subplots(figsize=(7.2, 2.6 + 0.5 * len(rows)))
    ax.axvspan(-delta, delta, color="#2e7d32", alpha=0.12, label=f"+/- {delta:g} pt equivalence margin")
    ax.axvline(0, color="#444", lw=0.9)
    for i, (name, pt, lo, hi, verd) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=COND_COLOUR.get(name, "#333"), lw=2.6, solid_capstyle="round")
        ax.plot([pt], [y], "o", color=COND_COLOUR.get(name, "#333"), ms=7)
        ax.text(hi + 0.35, y, f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]  {verd}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlabel("ON minus arm, success rate difference (points). Positive = removing depth costs success")
    title = "Paired difference on identical episodes"
    if floor:
        title += f"\nnoise floor: ON vs an identical re-run disagrees on {floor['n_discordant']} of " \
                 f"{floor['n_pairs']} episodes"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out / "F2_paired_differences.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def f3_discordance(out: pathlib.Path, verdict: Dict) -> Optional[str]:
    """Per-arm disagreement with ON, split by direction, against the ON-vs-ON2 floor."""
    decided = verdict.get("decided") or {}
    arms = decided.get("arms") or {}
    floor = (decided.get("gates") or {}).get("G-floor")
    if not arms or not floor:
        return None
    plt = _mpl()
    names = ["ON2 (floor)"] + list(arms)
    on_only = [floor["a_only"]] + [arms[a]["rule2"]["d_test_on_vs_arm"]["a_only"] for a in arms]
    arm_only = [floor["b_only"]] + [arms[a]["rule2"]["d_test_on_vs_arm"]["b_only"] for a in arms]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    xs = range(len(names))
    ax.bar(xs, on_only, width=0.6, color="#1f4e79", label="ON succeeded, arm failed")
    ax.bar(xs, arm_only, width=0.6, bottom=on_only, color="#c55a11",
           label="arm succeeded, ON failed")
    ax.axhline(floor["n_discordant"], color="#555", ls="--", lw=1.0,
               label="noise floor (ON vs identical re-run)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("episodes disagreeing with ON")
    ax.set_title("Disagreement with ON, and the floor an identical re-run already produces", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "F3_discordance.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def f4_per_suite(out: pathlib.Path, verdict: Dict) -> Optional[str]:
    """Per-suite rates. Labelled as not a verdict, because 50 pairs cannot decide the margin."""
    per = verdict.get("per_suite_rates_NOT_A_VERDICT") or {}
    if not per:
        return None
    plt = _mpl()
    suites = sorted(per)
    conds = [c for c in COND_ORDER if any(c in per[s] for s in suites)]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    width = 0.8 / max(1, len(conds))
    for j, c in enumerate(conds):
        xs = [i + j * width - 0.4 + width / 2 for i in range(len(suites))]
        vals = [per[s].get(c, {}).get("rate", 0) * 100 for s in suites]
        ax.bar(xs, vals, width=width, color=COND_COLOUR[c], label=c)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(suites, fontsize=9)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Per suite — reported for the record, NOT a verdict:\n"
                 "50 paired episodes leave a 7.1 point bound, wider than the 5 point margin", fontsize=10)
    ax.legend(fontsize=8, ncol=len(conds))
    fig.tight_layout()
    p = out / "F4_per_suite.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def f5_cost(out: pathlib.Path, verdict: Dict) -> Optional[str]:
    """The measured inference cost of the depth block — the number this lineage lacked an artifact for."""
    lat = verdict.get("latency") or {}
    cost = lat.get("_depth_cost")
    conds = [c for c in COND_ORDER if c in lat and isinstance(lat.get(c), dict)
             and lat[c].get("median_seconds_per_call")]
    if not conds:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    vals = [lat[c]["median_seconds_per_call"] for c in conds]
    ax.bar(range(len(conds)), vals, color=[COND_COLOUR[c] for c in conds], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.3f}s", ha="center", fontsize=9)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c for c in conds], fontsize=9)
    ax.set_ylabel("median seconds per model call")
    title = "Inference cost per action call"
    if cost:
        title += f"\ndepth block costs {cost['multiple']:.2f}x  ({cost['saving_pct']:.1f}% saved by NOGEN)"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    p = out / "F5_cost.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def draw_all(out_dir: pathlib.Path, verdict: Dict) -> Dict[str, object]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = {"F1_rates": f1_rates, "F2_paired_differences": f2_paired_differences,
              "F3_discordance": f3_discordance, "F4_per_suite": f4_per_suite, "F5_cost": f5_cost}
    drawn: Dict[str, str] = {}
    skipped: List[str] = []
    for name, fn in panels.items():
        path = fn(out_dir, verdict)
        if path:
            drawn[name] = path
        else:
            skipped.append(name)
    return {"drawn": drawn, "skipped": skipped,
            "source": "results/verdict.json only — no figure recomputes a number"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", default="studies/depthoff/results/verdict.json")
    ap.add_argument("--out", default="studies/depthoff/results/figures")
    a = ap.parse_args()
    v = json.loads(pathlib.Path(a.verdict).read_text())
    rep = draw_all(pathlib.Path(a.out), v)
    (pathlib.Path(a.out) / "figures.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
