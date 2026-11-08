"""DEPTHOPEN figures — SPEC §10, drawn from `results/verdict.json` and the per-arm artifacts.

Same contract as DEPTHANCHOR's figure module: every figure carries the finding in its title, the counts in
its caption, and the script, cell and input hash in its footer; PNG and PDF; each `draw_*` returns None
when its inputs are absent, so a session draws what it has and never invents a panel.

  F1  the gate per layer per arm — the one panel that says whether opening the gate stayed open
  F2  shuffle, constant grid and removal per arm, each against that arm's OWN floor and margin
  F3  cross-attention share on the depth positions, per layer per arm
  F4  standard LIBERO by suite per arm, with Wilson intervals and the ±8-point band
  F5  LIBERO-plus camera success by level per arm, with Wilson intervals and the published reference rows
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np

SCRIPT = "infra/pg/depthopen_figures.py"
ARM_ORDER = ("A0", "A1", "A2", "A3", "A4")
ARM_COLOR = {"A0": "#6b6b6b", "A1": "#c1666b", "A2": "#d9a13b", "A3": "#3b6ea5", "A4": "#4f8a5b"}
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# LIBERO-plus's own README table, camera dimension — reference rows, never a baseline for our arms
PUBLISHED_CAMERA = {"π₀": 13.8, "π₀-Fast": 65.1, "OpenVLA-OFT": 56.4, "OpenVLA-OFT mixed": 92.8}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _finish(fig, out_dir: pathlib.Path, name: str, caption: str, cell: str,
            input_hash: str) -> List[str]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(0.0, -0.03, caption, fontsize=6, color="#333333", ha="left", va="top", wrap=True)
    fig.text(0.0, -0.11, f"{SCRIPT} · cell {cell} · inputs {input_hash[:12]}", fontsize=5,
             color="#666666", ha="left", va="top")
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    _plt().close(fig)
    return paths


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _arms_present(arms: Dict[str, object]) -> List[str]:
    return [a for a in ARM_ORDER if a in arms]


def _err(rate: float, lo: float, hi: float) -> List[float]:
    """Error-bar halves that can never be negative — the defect that broke DEPTHANCHOR's F7 twice."""
    return [max(0.0, rate - lo), max(0.0, hi - rate)]


# ------------------------------------------------------------------ F1 the gate per layer
def draw_f1(out_dir, arms: Dict[str, object], gate_layers: Dict[str, Sequence[float]]
            ) -> Optional[List[str]]:
    """Per-layer sigmoid of the trained gate, one line per gated arm, with each arm's initialization."""
    present = [a for a in _arms_present(arms) if gate_layers.get(a)]
    if not present:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for arm in present:
        g = np.asarray(gate_layers[arm], dtype=float)
        ax.plot(np.arange(len(g)), g, marker="o", ms=2.5, lw=1.2, color=ARM_COLOR[arm],
                label=f"{arm} (init bias {arms[arm]['flags']['init_bias']})")
        init = 1.0 / (1.0 + np.exp(-float(arms[arm]["flags"]["init_bias"])))
        ax.axhline(init, color=ARM_COLOR[arm], lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel("action-expert layer")
    ax.set_ylabel("gate = sigmoid(bias)")
    ax.set_ylim(0, 1)
    ax.set_title("F1 · the depth gate after training, per layer\n"
                 "dotted lines are each arm's initialization")
    ax.legend(fontsize=6, ncol=2)
    caption = ("Per-layer gate read from the exported checkpoint's `action_expert_depth_gate.*.bias` "
               "(no forward pass). An arm whose line sits on its dotted line never moved its gate.")
    return _finish(fig, out_dir, "F1_gate_per_layer", caption, "arm/*", _hash(gate_layers))


# ------------------------------------------------------------------ F2 the edits, per arm
def draw_f2(out_dir, arms: Dict[str, object], reads: Dict[str, object]) -> Optional[List[str]]:
    """Each arm's shuffle, constant grid and removal, as ratios to that arm's own RGB ceiling, with the
    arm's own margin m drawn where the rule reads it."""
    present = [a for a in _arms_present(arms) if (arms[a].get("r2") or {}).get("m") is not None]
    if not present:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.26
    x = np.arange(len(present))
    for j, (key, label) in enumerate((("shuffle_over_ceiling", "shuffle all 100 cells"),
                                      ("const_over_ceiling", "one code everywhere"),
                                      ("remove_over_ceiling", "remove the prefix"))):
        vals = [float((arms[a]["r2"] or {}).get(key) or 0.0) for a in present]
        ax.bar(x + (j - 1) * width, vals, width, label=label,
               color=["#8fb3d9", "#d9c48f", "#b58fd9"][j], edgecolor="#444444", lw=0.4)
    for i, a in enumerate(present):
        m = float(arms[a]["r2"]["m"])
        ax.plot([i - 1.5 * width, i + 1.5 * width], [m, m], color="#c1666b", lw=1.6)
        v = (reads.get(a) or {}).get("verdict", "")
        ax.annotate(v, (i, m), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=6, color="#c1666b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\ninit {arms[a]['flags']['init_bias'] if arms[a]['flags']['gate'] else '—'}"
                        for a in present])
    ax.set_yscale("log")
    ax.set_ylabel("Δ action ÷ the frame's RGB ceiling (log)")
    ax.set_title("F2 · what each edit does to the action, per arm\n"
                 "red line is that arm's own margin m; rule 1 compares the shuffle bar with it")
    ax.legend(fontsize=6)
    caption = ("Per-frame medians over the 40 held-out frames, each divided by that frame's own RGB "
               "ceiling. m is the 95th percentile of seed floor ÷ ceiling for that arm.")
    return _finish(fig, out_dir, "F2_edits_per_arm", caption, "arm/*", _hash([arms, reads]))


# ------------------------------------------------------------------ F3 attention share
def draw_f3(out_dir, arms: Dict[str, object], per_layer: Dict[str, Sequence[float]]
            ) -> Optional[List[str]]:
    present = [a for a in _arms_present(arms) if per_layer.get(a)]
    if not present:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    uniform = None
    for arm in present:
        v = np.asarray(per_layer[arm], dtype=float)
        ax.plot(np.arange(len(v)), v, lw=1.2, marker="o", ms=2.2, color=ARM_COLOR[arm], label=arm)
        u = (arms[arm].get("r3") or {}).get("uniform_share")
        uniform = float(u) if u is not None else uniform
    if uniform is not None:
        ax.axhline(uniform, color="#444444", lw=0.9, ls="--",
                   label=f"uniform share {uniform:.3f}")
    ax.set_xlabel("action-expert layer")
    ax.set_ylabel("share of cross-attention on the 100 depth positions")
    ax.set_title("F3 · where the action expert's cross-attention goes")
    ax.legend(fontsize=6, ncol=3)
    caption = ("Median over 20 held-out frames and the flow-matching steps, per layer. The dashed line is "
               "the share the depth positions would get if attention were spread uniformly.")
    return _finish(fig, out_dir, "F3_attention_share", caption, "arm/*", _hash(per_layer))


# ------------------------------------------------------------------ F4 standard LIBERO
def draw_f4(out_dir, arms: Dict[str, object]) -> Optional[List[str]]:
    present = [a for a in _arms_present(arms) if (arms[a].get("r4") or {}).get("pooled")]
    if not present:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    width = 0.8 / max(1, len(present))
    x = np.arange(len(SUITE_ORDER) + 1)
    for i, arm in enumerate(present):
        r4 = arms[arm]["r4"]
        rates, errs = [], [[], []]
        for suite in SUITE_ORDER + ("pooled",):
            w = (r4.get("by_suite") or {}).get(suite) if suite != "pooled" else r4.get("pooled")
            if not w:
                rates.append(0.0)
                errs[0].append(0.0)
                errs[1].append(0.0)
                continue
            rates.append(100 * w["rate"])
            lo, hi = _err(100 * w["rate"], 100 * w["lo"], 100 * w["hi"])
            errs[0].append(lo)
            errs[1].append(hi)
        ax.bar(x + i * width - 0.4, rates, width, yerr=np.asarray(errs), capsize=1.5,
               color=ARM_COLOR[arm], edgecolor="#333333", lw=0.4, label=arm,
               error_kw={"lw": 0.7, "ecolor": "#333333"})
    base = arms.get("A0", {}).get("r4", {}).get("pooled")
    if base:
        ax.axhline(100 * base["rate"], color="#6b6b6b", lw=0.8, ls="--")
        ax.axhspan(100 * base["rate"] - 8, 100 * base["rate"] + 8, color="#6b6b6b", alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("libero_", "") for s in SUITE_ORDER] + ["pooled"])
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("F4 · standard LIBERO per arm — a collapse check, not a comparison\n"
                 "shaded band is ±8 points around A0's pooled rate")
    ax.legend(fontsize=6, ncol=5)
    caption = ("50 episodes per suite per arm (10 tasks × 5 initial states), Wilson 95 % intervals. "
               "§0: no arm here is compared with the released checkpoint, which trained on about forty "
               "times more samples.")
    return _finish(fig, out_dir, "F4_standard_libero", caption, "r4/*", _hash(arms))


# ------------------------------------------------------------------ F5 LIBERO-plus camera
def draw_f5(out_dir, arms: Dict[str, object], released: Dict[str, object],
            dimension: str = "camera") -> Optional[List[str]]:
    rows = [(a, arms[a][f"r5_{dimension}"]) for a in _arms_present(arms)
            if arms[a].get(f"r5_{dimension}")]
    ref = (released.get(dimension) or [])
    if not rows and not ref:
        return None
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2), width_ratios=[2.0, 1.0])
    levels = sorted({k for _, r in rows for k in (r.get("by_level") or {})} |
                    {k for run in ref for k in ((run.get("verdict") or {}).get("by_level") or {})})
    x = np.arange(len(levels))
    series = [(f"released run {i + 1}", (run.get("verdict") or {}).get("by_level") or {}, "#6b6b6b")
              for i, run in enumerate(ref)]
    series += [(a, r.get("by_level") or {}, ARM_COLOR[a]) for a, r in rows]
    width = 0.8 / max(1, len(series))
    for i, (label, by_level, color) in enumerate(series):
        rates, errs = [], [[], []]
        for lv in levels:
            w = by_level.get(lv)
            rates.append(100 * w["rate"] if w else 0.0)
            lo, hi = _err(100 * w["rate"], 100 * w["lo"], 100 * w["hi"]) if w else (0.0, 0.0)
            errs[0].append(lo)
            errs[1].append(hi)
        ax.bar(x + i * width - 0.4, rates, width, yerr=np.asarray(errs), capsize=1.2, label=label,
               color=color, edgecolor="#333333", lw=0.3, error_kw={"lw": 0.6, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{lv}" for lv in levels])
    ax.set_ylabel("success rate (%)")
    ax.set_xlabel("LIBERO-plus difficulty level")
    ax.set_title(f"F5 · LIBERO-plus {dimension} dimension, by level")
    ax.legend(fontsize=6, ncol=2)

    labels, vals, colors = [], [], []
    for name, v in PUBLISHED_CAMERA.items():
        labels.append(name)
        vals.append(v)
        colors.append("#bbbbbb")
    for i, run in enumerate(ref):
        w = (run.get("verdict") or {}).get("overall")
        if w:
            labels.append(f"released {i + 1}")
            vals.append(100 * w["rate"])
            colors.append("#6b6b6b")
    for a, r in rows:
        w = r.get("overall")
        if w:
            labels.append(a)
            vals.append(100 * w["rate"])
            colors.append(ARM_COLOR[a])
    ax2.barh(np.arange(len(labels)), vals, color=colors, edgecolor="#333333", lw=0.3)
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels, fontsize=6)
    ax2.invert_yaxis()
    ax2.set_xlabel("overall success (%)")
    ax2.set_title("overall, with the benchmark's\npublished rows for scale", fontsize=8)
    caption = ("One trial per task, the benchmark's own protocol, on the banked 200-task sample. Grey "
               "rows are LIBERO-plus's published numbers for other policies and are context, not a "
               "baseline for these arms.")
    return _finish(fig, out_dir, f"F5_liberoplus_{dimension}", caption, "r5/*",
                   _hash([arms, released]))


def draw_all(out_dir, *, verdict: Dict[str, object], gate_layers: Optional[Dict[str, Sequence[float]]] = None,
             attention_layers: Optional[Dict[str, Sequence[float]]] = None) -> Dict[str, object]:
    arms = verdict.get("arms") or {}
    released = verdict.get("released_r5") or {}
    reads = (verdict.get("decided") or {}).get("reads") or {}
    drew: Dict[str, object] = {}
    drew["F1"] = draw_f1(out_dir, arms, gate_layers or {})
    drew["F2"] = draw_f2(out_dir, arms, reads)
    drew["F3"] = draw_f3(out_dir, arms, attention_layers or {})
    drew["F4"] = draw_f4(out_dir, arms)
    for dim in ("camera", "layout"):
        drew[f"F5_{dim}"] = draw_f5(out_dir, arms, released, dimension=dim)
    return {"drew": {k: v for k, v in drew.items() if v}, "skipped": [k for k, v in drew.items() if not v]}


# ------------------------------------------------------------------ F6 the two films
def draw_f6(out_dir, films: Dict[str, Dict[str, np.ndarray]], *, task_label: str = "",
            fps: int = 4) -> Optional[List[str]]:
    """SPEC §10's last item: the SAME perturbed LIBERO-plus task under two arms, RGB above the depth grid
    the policy was conditioned on.

    DEPTHANCHOR's strip drawer already does exactly this and is torch-free, so the decoded grid is passed
    in rather than decoded here; `films[arm]` needs `frames` and may carry `decoded`.
    """
    from infra.pg import depthanchor_figures as daf
    if not films:
        return None
    return daf.draw_f8(out_dir, films, suite=task_label, fps=fps)
