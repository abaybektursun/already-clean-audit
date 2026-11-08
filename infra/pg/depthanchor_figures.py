"""DEPTHANCHOR figures — F1..F9 of spec §10. The visual record is a deliverable, not an afterthought.

Every figure carries the finding in its title, the raw counts in its caption, and the generating script,
the cell id and the input hash in its footer; PNG and PDF. Each `draw_*` returns None when its inputs are
absent, so the same call draws what a session has and never invents a panel — and `draw_all` reports what
it drew, which is what the figure job banks.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from infra.pg import depthanchor_core as core

SCRIPT = "infra/pg/depthanchor_figures.py"
SUITE_COLOR = {"libero_spatial": "#3b6ea5", "libero_object": "#c1666b", "libero_goal": "#4f8a5b",
               "libero_10": "#8a6bb5"}


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
    # BELOW the axes in figure coordinates, so `bbox_inches="tight"` grows the canvas to include them
    # instead of laying them over the x-labels.
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


def _by_frame(records, cell, key="delta_norm_cont", variant=None):
    out: Dict[str, List[float]] = {}
    suite: Dict[str, str] = {}
    for r in records:
        if r["cell"] != cell or (variant is not None and r["variant"] != variant):
            continue
        out.setdefault(r["frame_id"], []).append(float(r[key]))
        suite[r["frame_id"]] = r["suite"]
    return {f: float(np.mean(v)) for f, v in out.items()}, suite


# ------------------------------------------------------------------ F1 the code ladder
def draw_f1(out_dir, ladder: Optional[core.CodeLadder], gates=None) -> Optional[List[str]]:
    if ladder is None:
        return None
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    order = ladder.order
    x = np.arange(core.N_CODES)
    ax.plot(x, ladder.d_primary[order], lw=1.2, label="uniform grid of the code")
    ax.plot(x, ladder.d_local[order], lw=1.0, ls="--", label="the code alone, on its own patch")
    ax.set_xlabel("code, ordered by decoded depth")
    ax.set_ylabel("decoded depth (VQ-VAE units)")
    ax.legend(frameon=False, fontsize=6)
    gap = np.abs(ladder.d_primary - ladder.d_local)
    sep = np.abs(np.diff(ladder.d_primary[order]))
    ax2.hist(sep, bins=40, color="#3b6ea5", alpha=0.85, label="neighbour separation")
    ax2.axvline(float(np.median(gap)), color="#c1666b", lw=1.2,
                label=f"placement noise (median {np.median(gap):.4f})")
    ax2.set_xlabel("|d(c) − d(next rank)|")
    ax2.set_ylabel("codes")
    ax2.legend(frameon=False, fontsize=6)
    n_distinct = int(np.sum(_neighbour_min(ladder.d_primary[order]) > np.median(gap)))
    verdict = None if gates is None else (gates.get("G2") or {}).get("verdict")
    fig.suptitle(f"F1 the 128 depth codes are an order: {n_distinct}/128 separated from their rank "
                 f"neighbours by more than the decoder's placement noise (G2 {verdict})", fontsize=9)
    return _finish(fig, out_dir, "F1_depth_ladder",
                   ("Left: d(c) under both placements, codes sorted by the primary placement. "
                   "Right: the distribution of neighbour separations against the median placement "
                    "noise. n = 128 codes, decoded twice each."), "G2",
                   _hash([ladder.d_primary.tolist(), ladder.d_local.tolist()]))


def _neighbour_min(sorted_d: np.ndarray) -> np.ndarray:
    left = np.concatenate([[np.inf], np.abs(np.diff(sorted_d))])
    right = np.concatenate([np.abs(np.diff(sorted_d)), [np.inf]])
    return np.minimum(left, right)


# ------------------------------------------------------------------ F2 depth fidelity
def draw_f2(out_dir, fidelity: Optional[Dict[str, object]]) -> Optional[List[str]]:
    if not fidelity or not fidelity.get("per_frame"):
        return None
    plt = _plt()
    rows = fidelity["per_frame"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for suite in core.SUITES:
        vals = [r["spearman_decoded_vs_da2"] for r in rows if r["suite"] == suite]
        if vals:
            ax.scatter(np.full(len(vals), core.SUITES.index(suite)) + np.linspace(-0.2, 0.2, len(vals)),
                       vals, s=8, color=SUITE_COLOR[suite], label=suite)
    shuf = [r["spearman_shuffled"] for r in rows if r.get("spearman_shuffled") is not None]
    if shuf:
        ax.axhline(float(np.percentile(shuf, 95)), color="#c1666b", lw=1.0,
                   label="shuffled-bins 95th percentile")
    ax.set_xticks(range(len(core.SUITES)))
    ax.set_xticklabels([s.replace("libero_", "") for s in core.SUITES])
    ax.set_ylabel("per-frame Spearman, decoded bins vs Depth-Anything")
    ax.legend(frameon=False, fontsize=6)
    agree = [r["bin_agreement"] for r in rows if r.get("bin_agreement") is not None]
    if agree:
        ax2.hist(agree, bins=20, color="#4f8a5b", alpha=0.85)
        ax2.set_xlabel("fraction of the 100 cells whose code equals the reference pipeline's")
        ax2.set_ylabel("frames")
    med = float(np.median([r["spearman_decoded_vs_da2"] for r in rows]))
    fig.suptitle(f"F2 the model's depth grid is depth: median per-frame Spearman {med:+.2f} against "
                 f"the pipeline that wrote its training targets", fontsize=9)
    return _finish(fig, out_dir, "F2_fidelity",
                   f"n = {len(rows)} frames. Left: per-frame rank correlation between the decoded "
                   f"predicted grid and Depth-Anything-V2 depth pooled over the 100 cells, against the "
                   f"same statistic under spatially shuffled bins. Right: exact code agreement with the "
                   f"released annotation pipeline.", "G1", _hash(rows))


# ------------------------------------------------------------------ F3 anchor maps
def draw_f3(out_dir, offline: Optional[Dict[str, object]]) -> Optional[List[str]]:
    maps = (offline or {}).get("per_cell_maps") or []
    if not maps:
        return None
    plt = _plt()
    by_suite: Dict[str, List[np.ndarray]] = {}
    for entry in maps:
        imp = (np.abs(np.asarray(entry["delta_plus8"])) + np.abs(np.asarray(entry["delta_minus8"])))
        by_suite.setdefault(entry["suite"], []).append(imp.reshape(core.GRID, core.GRID))
    suites = [s for s in core.SUITES if s in by_suite]
    fig, axes = plt.subplots(1, len(suites) + 1, figsize=(2.0 * (len(suites) + 1), 2.4))
    axes = np.atleast_1d(axes)
    pooled = np.mean([m for ms in by_suite.values() for m in ms], axis=0)
    for ax, suite in zip(axes[:-1], suites):
        m = np.mean(by_suite[suite], axis=0)
        im = ax.imshow(m, cmap="magma")
        ax.set_title(suite.replace("libero_", ""), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    im = axes[-1].imshow(pooled, cmap="magma")
    axes[-1].set_title("pooled", fontsize=7)
    axes[-1].set_xticks([]); axes[-1].set_yticks([]); axes[-1].grid(False)
    fig.colorbar(im, ax=axes[-1], fraction=0.046)
    top = np.unravel_index(int(np.argmax(pooled)), pooled.shape)
    fig.suptitle(f"F3 per-cell importance of the depth grid: |Δ| from shifting one cell ±8 ranks; "
                 f"the largest cell is (row {top[0]}, col {top[1]})", fontsize=9)
    return _finish(fig, out_dir, "F3_anchor_maps",
                   f"n = {len(maps)} frames × 200 single-cell edits. Each panel is the mean over that "
                   f"suite's frames of |Δ(+8)| + |Δ(−8)| per cell, primary units.", "O-cell",
                   _hash(maps))


# ------------------------------------------------------------------ F5 dose curves
def _sig(x, digits: int = 2) -> str:
    """A signed number at `digits` significant figures, in fixed notation when that is readable and in
    scientific notation when it is not. Figure titles carry the finding, so a title must never round a
    measured effect to zero."""
    if x is None:
        return "—"
    x = float(x)
    if x == 0.0:
        return "0"
    return f"{x:+.{digits}e}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:+.{digits}g}"


def draw_f5(out_dir, aggregate: Optional[Dict[str, object]]) -> Optional[List[str]]:
    per_k = (aggregate or {}).get("per_k")
    if not per_k:
        return None
    plt = _plt()
    ks = sorted(int(k) for k in per_k)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for cell, color, label in (("target", "#3b6ea5", "target cells"), ("bg", "#c1666b", "background cells")):
        pt = [per_k[str(k)][cell]["point"] for k in ks]
        lo = [per_k[str(k)][cell]["ci95"][0] for k in ks]
        hi = [per_k[str(k)][cell]["ci95"][1] for k in ks]
        ax.plot(ks, pt, "-o", ms=3, color=color, label=label)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18)
    seed = (aggregate.get("absolute") or {}).get("seed_floor")
    ceil = (aggregate.get("absolute") or {}).get("ceiling_rgb")
    gate0 = (aggregate.get("absolute") or {}).get("ceiling_gate0")
    for band, color, label in ((seed, "#888888", "seed floor S(f)"),
                               (ceil, "#000000", "RGB ceiling C(f)"),
                               (gate0, "#4f8a5b", "depth ablation ceiling")):
        if band:
            ax.axhspan(band["ci95"][0], band["ci95"][1], color=color, alpha=0.12)
            ax.axhline(band["point"], color=color, lw=0.8, ls=":", label=label)
    ax.set_xlabel("requested shift k (ranks on the code ladder)")
    ax.set_ylabel("Δ, primary units (median over frames)")
    ax.legend(frameon=False, fontsize=6)
    pt = [per_k[str(k)]["paired_target_minus_bg"]["point"] for k in ks]
    lo = [per_k[str(k)]["paired_target_minus_bg"]["ci95"][0] for k in ks]
    hi = [per_k[str(k)]["paired_target_minus_bg"]["ci95"][1] for k in ks]
    ax2.axhline(0.0, color="#000000", lw=0.8)
    # clamped for the same reason as F7: a banked interval can sit a floating-point hair inside its point
    ax2.errorbar(ks, pt, yerr=[np.maximum(0.0, np.array(pt) - np.array(lo)),
                               np.maximum(0.0, np.array(hi) - np.array(pt))],
                 fmt="o", ms=3, color="#3b6ea5")
    ax2.set_xlabel("requested shift k")
    ax2.set_ylabel("paired Δ(target) − Δ(background)")
    primary = per_k.get("8", {}).get("paired_target_minus_bg")
    # A fixed number of decimals is wrong for this figure: the paired difference at |k|=8 is of order
    # 1e-5, and `.4f` printed it as "-0.0000 [-0.0000, +0.0000]" — a title that says "no effect" over a
    # panel that plots a real one. Two significant figures, in whatever notation fits the magnitude.
    head = ("not measured" if not primary else
            f"{_sig(primary['point'])} [{_sig(primary['ci95'][0])}, {_sig(primary['ci95'][1])}]")
    fig.suptitle(f"F5 dose: target minus count-matched background at k=+8 is {head} in primary units",
                 fontsize=9)
    return _finish(fig, out_dir, "F5_dose_curves",
                   f"n = {aggregate.get('n_frames')} frames, paired within frame, suite-stratified "
                   f"bootstrap, {core.N_BOOT} resamples. Left: absolute Δ with the control bands. "
                   f"Right: the paired contrast that rule 2(a) reads.", "O-target-k/O-bg-k",
                   _hash(per_k))


# ------------------------------------------------------------------ F6 distributions
def draw_f6(out_dir, offline: Optional[Dict[str, object]],
            aggregate: Optional[Dict[str, object]]) -> Optional[List[str]]:
    records = (offline or {}).get("records")
    if not records:
        return None
    plt = _plt()
    order = ["seed", "ident", "shuffle", "const", "random", "target", "bg", "gate0", "gate1", "ceiling"]
    have = [c for c in order if any(r["cell"] == c for r in records)]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(have) + 1.5))
    for i, cell in enumerate(have):
        table, _ = _by_frame(records, cell)
        vals = np.array(list(table.values()))
        if vals.size == 0:
            continue
        ax.scatter(vals, np.full(vals.size, i) + core.rng_for("f6", cell).normal(0, 0.06, vals.size),
                   s=6, alpha=0.6, color="#3b6ea5")
        ax.plot([np.median(vals)], [i], "|", ms=18, color="#c1666b")
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels(have)
    ax.set_xlabel("Δ, primary units (mean over that cell's variants, one point per frame)")
    ax.set_xscale("symlog", linthresh=1e-4)
    m = None if not aggregate else aggregate.get("m")
    fig.suptitle("F6 every offline edit against the seed floor" + ("" if m is None else
                 f" (frozen margin m = {m:.4f} of the ceiling)"), fontsize=9)
    return _finish(fig, out_dir, "F6_distributions",
                   f"One point per frame per cell; the red bar is the median. Symlog x, linear below "
                   f"1e-4. n = {len(set(r['frame_id'] for r in records))} frames.", "all offline",
                   _hash([len(records)]))


# ------------------------------------------------------------------ F7 closed-loop bars
def draw_f7(out_dir, closed_loop: Optional[Dict[str, object]]) -> Optional[List[str]]:
    if not closed_loop or not closed_loop.get("by_arm_suite"):
        return None
    plt = _plt()
    data = closed_loop["by_arm_suite"]
    arms = list(data.keys())
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    width = 0.8 / max(1, len(core.SUITES))
    for si, suite in enumerate(core.SUITES):
        xs, ys, los, his = [], [], [], []
        for ai, arm in enumerate(arms):
            entry = data[arm].get(suite)
            if not entry:
                continue
            xs.append(ai + si * width - 0.4 + width / 2)
            ys.append(entry["rate"])
            # clamped, because a banked interval from an earlier run can still carry the 1e-16 gap
            los.append(max(0.0, entry["rate"] - entry["ci95"][0]))
            his.append(max(0.0, entry["ci95"][1] - entry["rate"]))
        if xs:
            ax.bar(xs, ys, width=width * 0.92, color=SUITE_COLOR[suite], label=suite.replace("libero_", ""))
            ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#333333", lw=0.7, capsize=1.5)
    normal = closed_loop.get("pooled", {}).get("C-normal")
    if normal:
        ax.axhline(normal["rate"], color="#000000", lw=0.9, ls="--",
                   label=f"C-normal pooled {normal['rate']:.2f}")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("LIBERO success rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6, ncol=3)
    fig.suptitle("F7 closed-loop success by depth regime, Wilson 95% intervals", fontsize=9)
    n_ep = closed_loop.get("episodes_per_cell")
    if isinstance(n_ep, (list, tuple, set)):  # one entry per (arm, suite); say the range, not the last
        n_ep = sorted(n_ep)
        n_ep = str(n_ep[0]) if len(n_ep) == 1 else f"{n_ep[0]}-{n_ep[-1]}"
    return _finish(fig, out_dir, "F7_closed_loop",
                   f"{n_ep} episodes per arm per suite; per-episode seeding from the master seed.",
                   "closed loop", _hash(data))


# ------------------------------------------------------------------ F9 divergence traces
def draw_f9(out_dir, closed_loop: Optional[Dict[str, object]]) -> Optional[List[str]]:
    traces = (closed_loop or {}).get("traces")
    if not traces:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for arm, path in traces.items():
        p = np.asarray(path, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2:
            continue
        axes[0].plot(p[:, 0], p[:, 1], lw=1.0, label=arm)
        axes[1].plot(np.arange(p.shape[0]), p[:, 2], lw=1.0, label=arm)
    axes[0].set_xlabel("end-effector x (m)"); axes[0].set_ylabel("y (m)")
    axes[1].set_xlabel("environment step"); axes[1].set_ylabel("z (m)")
    axes[1].legend(frameon=False, fontsize=6)
    fig.suptitle("F9 one episode, the same task and seed under each depth regime", fontsize=9)
    return _finish(fig, out_dir, "F9_divergence", f"{len(traces)} regimes, one episode each.",
                   "closed loop", _hash(list(traces)))


# ------------------------------------------------------------------ F4 edit strips (needs images)
def draw_f4(out_dir, strips: Optional[Sequence[Dict[str, object]]]) -> Optional[List[str]]:
    """`strips` entries: {frame_id, rgb, decoded_pred, decoded_edited, target_cells, a_ref, a_edit,
    axis}. Drawn only when the GPU cell banked the decoded depth images."""
    if not strips:
        return None
    plt = _plt()
    n = min(4, len(strips))
    fig, axes = plt.subplots(n, 4, figsize=(8.4, 2.1 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        s = strips[r]
        panels = [("agentview RGB", np.asarray(s["rgb"]), None),
                  ("decoded predicted depth", np.asarray(s["decoded_pred"]), "viridis"),
                  ("decoded edited depth", np.asarray(s["decoded_edited"]), "viridis"),
                  ("edited − predicted", np.asarray(s["decoded_edited"]) - np.asarray(s["decoded_pred"]),
                   "coolwarm")]
        for c, (title, img, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(title, fontsize=7)
            if c == 0:
                ax.set_ylabel(str(s["frame_id"]), fontsize=5)
            if c in (1, 2, 3):
                for cell in np.asarray(s["target_cells"], dtype=int):
                    rr, cc = divmod(int(cell), core.GRID)
                    side = img.shape[0] / core.GRID
                    ax.add_patch(plt.Rectangle((cc * side, rr * side), side, side, fill=False,
                                              edgecolor="#ffcc00", lw=0.6))
    fig.suptitle("F4 what an edit does to the decoded depth map, target cells outlined", fontsize=9)
    return _finish(fig, out_dir, "F4_edit_strips", f"{n} frames of the banked sample set.",
                   "O-target-k", _hash([s["frame_id"] for s in strips]))


# ------------------------------------------------------------------ F8 rollout films
def draw_f8(out_dir, films: Optional[Dict[str, Dict[str, np.ndarray]]], suite: str = "",
            fps: int = 4) -> Optional[List[str]]:
    """One animated strip per suite: the same episode under each depth regime, RGB above the decoded
    depth map the policy was conditioned on. Written as a GIF with Pillow (no extra dependency) and, if
    imageio is importable, also as an mp4; the caller records which formats were produced.

    `films[arm]` needs `frames` (T,H,W,3 uint8) and may carry `decoded` (T,h,w float) — the decoded
    depth image of that step's grid, decoded by the caller, so this module stays torch-free."""
    if not films:
        return None
    from PIL import Image
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in films if films[a].get("frames") is not None]
    if not arms:
        return None
    n = min(int(np.asarray(films[a]["frames"]).shape[0]) for a in arms)
    if n < 2:
        return None
    panels = []
    for t in range(n):
        row = []
        for a in arms:
            rgb = np.asarray(films[a]["frames"])[t]
            h, w = rgb.shape[:2]
            dec = films[a].get("decoded")
            if dec is not None:
                d = np.asarray(dec)[t]
                d = (d - d.min()) / max(1e-8, float(d.max() - d.min()))
                small = np.asarray(Image.fromarray((d * 255).astype(np.uint8)).resize((w, h),
                                                                                     Image.NEAREST))
                small = np.stack([small] * 3, axis=-1)
                row.append(np.concatenate([rgb, small], axis=0))
            else:
                row.append(rgb)
        panels.append(np.concatenate(row, axis=1))
    images = [Image.fromarray(p.astype(np.uint8)) for p in panels]
    gif = out_dir / f"F8_rollout_{suite or 'suite'}.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=int(1000 / max(1, fps)),
                   loop=0)
    made = [str(gif)]
    try:
        import imageio.v2 as imageio
        mp4 = out_dir / f"F8_rollout_{suite or 'suite'}.mp4"
        imageio.mimsave(mp4, panels, fps=fps)
        made.append(str(mp4))
    except Exception as exc:  # recorded, not hidden: a missing encoder is not a science failure
        print(f"[figures] F8 mp4 not written ({type(exc).__name__}: {exc}); the GIF is the record",
              flush=True)
    (out_dir / f"F8_rollout_{suite or 'suite'}.txt").write_text(
        f"arms left to right: {', '.join(arms)}; top row RGB, bottom row the decoded depth map the "
        f"policy was conditioned on; {n} policy calls at {fps} fps\n")
    return made


# ------------------------------------------------------------------ everything
def draw_all(out_dir, *, offline=None, aggregate=None, pack=None, ladder=None, gates=None,
             closed_loop=None, fidelity=None, strips=None, films=None) -> Dict[str, object]:
    made: Dict[str, object] = {}
    for name, paths in (("F1", draw_f1(out_dir, ladder, gates)),
                        ("F2", draw_f2(out_dir, fidelity)),
                        ("F3", draw_f3(out_dir, offline)),
                        ("F4", draw_f4(out_dir, strips)),
                        ("F5", draw_f5(out_dir, aggregate)),
                        ("F6", draw_f6(out_dir, offline, aggregate)),
                        ("F7", draw_f7(out_dir, closed_loop)),
                        ("F8", draw_f8(out_dir, films, suite=(closed_loop or {}).get("suite", ""))),
                        ("F9", draw_f9(out_dir, closed_loop))):
        if paths:
            made[name] = paths
    return made
