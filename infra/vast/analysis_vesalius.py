#!/usr/bin/env python3
"""VESALIUS offline mining of the banked V1 actstats sufficient statistics.

Frozen BEFORE any pos0_sq / rest_sq / sum_quad / pos_energy[1:] value is read
(blind-analysis discipline, spec §9). Zero GPU: every statistic below is an exact
function of the four banked arrays per (layer, site) described in
infra/vast/cells/v1_actstats.py.

The central object: rest_sq = sum_sq - pos0_sq, the per-coordinate energy at
positions 1..W-1 only. It removes the circularity that pos0_sq is a component of
sum_sq, and it is exact in float64 (dryrun_v1 asserts the conservation identity).

FAIL LOUD: no try/except, no fallbacks.

Usage: python3 -m infra.vast.analysis_vesalius <manifest.json> --out-dir <dir>
"""
import argparse
import json
import os

import numpy as np

K_LADDER = (8, 32, 128)
K_PRIMARY = 32
N_BOOT = 10000
SEED = 1337
SPIKE_BAR = 20.0          # x uniform; stratum definition only (spec §4.3)


# ---------------------------------------------------------------- pure core

def rest_sq(sum_sq, pos0_sq):
    """Energy at positions 1..W-1. Exact; asserts no material cancellation."""
    r = sum_sq - pos0_sq
    worst = r.min()
    assert worst > -1e-9 * sum_sq.max(), f"rest_sq cancellation {worst:.3e}"
    return np.maximum(r, 0.0)


def topk(v, k):
    """Indices of the k largest entries; ties broken by ascending index."""
    return np.sort(np.argsort(-v, kind="stable")[:k])


def jaccard(a, b):
    m = np.intersect1d(a, b, assume_unique=True).size
    return m / (a.size + b.size - m)


def uniform_null_jaccard(k, d):
    """E[|A n B|] = k^2/d for two independent uniform k-sets; E[J] ~ m/(2k-m)."""
    m = k * k / d
    return m / (2 * k - m)


def layer_shuffle_null(pos0_top, rest_tops_other, k):
    """J of this cell's position-0 address against every OTHER layer's rest address.

    Preserves model, site, magnitude class and residual-stream basis; destroys
    only the layer-and-position pairing (spec §5).
    """
    return np.array([jaccard(pos0_top, t) for t in rest_tops_other])


def participation_ratio(v):
    """(Sum v)^2 / (d * Sum v^2); PR * d is the effective coordinate count."""
    return float(v.sum() ** 2 / (v.size * (v ** 2).sum()))


def spearman(x, y):
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def topk_share(v, k):
    return float(np.sort(v)[::-1][:k].sum() / v.sum())


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg; returns a boolean array aligned with pvals."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p, kind="stable")
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        out[order[: np.nonzero(passed)[0].max() + 1]] = True
    return out


def boot_median_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, dtype=np.float64)
    draws = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.quantile(draws, 0.025)), \
        float(np.quantile(draws, 0.975))


# ---------------------------------------------------------------- per cell

def cell_stats(sum_sq, sum_quad, pos_energy, pos0_sq):
    """Every per-cell statistic the spec's Q1-Q4 rules consume."""
    d = sum_sq.size
    w = pos_energy.size
    rest = rest_sq(sum_sq, pos0_sq)
    total = pos_energy.sum()

    row = {
        "d": int(d), "window": int(w),
        # --- Q4 / stratum (SEEN quantities, re-derived not transcribed)
        "pos0_share": float(pos_energy[0] / total),
        "pos0_x_uniform": float(pos_energy[0] / total * w),
        "pos_share_tail_mean": float(pos_energy[1:].mean() / total),
        # --- Q3
        "top32_share_sum": topk_share(sum_sq, K_PRIMARY),
        "top32_share_rest": topk_share(rest, K_PRIMARY),
        # --- Q2 (unseen: pos_energy[1:] beyond its mean)
        "pos_ref_median": float(np.median(pos_energy[4:]) / total),
        # --- Q1 width riders
        "pr_pos0_eff": participation_ratio(pos0_sq) * d,
        "pr_rest_eff": participation_ratio(rest) * d,
        "pr_sum_eff": participation_ratio(sum_sq) * d,
        "spearman_pos0_rest": spearman(pos0_sq, rest),
        "pos0_top_coord_share": float(pos0_sq.max() / pos0_sq.sum()),
        # --- provenance / degeneracy
        "n_zero_coord": int((sum_sq <= 0).sum()),
        "median_kurtosis": float(np.median(
            (sum_quad[sum_sq > 0] / sum_sq[sum_sq > 0] ** 2) * sum_sq.size)),
    }
    for p in (1, 2, 3):
        row[f"pos{p}_rel"] = float(pos_energy[p] / np.median(pos_energy[4:]))
    for k in K_LADDER:
        tp, tr = topk(pos0_sq, k), topk(rest, k)
        row[f"J{k}"] = jaccard(tp, tr)
        row[f"J{k}_uniform_null"] = uniform_null_jaccard(k, d)
        # degenerate iff either address reaches into zero-energy coordinates
        row[f"degenerate{k}"] = bool((pos0_sq[tp] <= 0).any() or (rest[tr] <= 0).any())
    return row, {k: (topk(pos0_sq, k), topk(rest, k)) for k in K_LADDER}


# ---------------------------------------------------------------- sweep

def sweep(npz_path, flags_path, tag):
    """All cells of one banked sweep, with the layer-shuffle null attached."""
    z = np.load(npz_path)
    flags = json.load(open(flags_path))
    rows, addr = [], {}
    for meta in flags["per_site"]:
        key = f"L{meta['layer']:02d}_{meta['site']}"
        row, tops = cell_stats(z[f"{key}_sum_sq"], z[f"{key}_sum_quad"],
                               z[f"{key}_pos_energy"], z[f"{key}_pos0_sq"])
        row.update(rung=tag, layer=int(meta["layer"]), site=meta["site"],
                   n_layers=int(flags["n_layers"]),
                   depth_frac=meta["layer"] / (flags["n_layers"] - 1),
                   cell=f"{tag}|L{meta['layer']:02d}|{meta['site']}")
        rows.append(row)
        addr[(meta["layer"], meta["site"])] = tops

    for row in rows:
        for k in K_LADDER:
            others = [addr[(l, s)][k][1] for (l, s) in addr
                      if s == row["site"] and l != row["layer"]]
            null = layer_shuffle_null(addr[(row["layer"], row["site"])][k][0],
                                      others, k)
            row[f"J{k}_layer_null_med"] = float(np.median(null))
            row[f"J{k}_layer_null_p90"] = float(np.quantile(null, 0.90))
            # one-sided p with the +1/(n+1) correction
            row[f"J{k}_p"] = float(((null >= row[f"J{k}"]).sum() + 1)
                                   / (null.size + 1))
    return rows


def preflight(rows, atlas_jsonl, tol=1e-9):
    """W1: banked atlas values must be re-derived from the raw npz to < tol."""
    banked = {f"L{r['layer']:02d}|{r['site']}": r
              for r in (json.loads(l) for l in open(atlas_jsonl))}
    checked = 0
    for row in rows:
        b = banked[f"L{row['layer']:02d}|{row['site']}"]
        assert abs(b["pos0_share"] - row["pos0_share"]) < tol, row["cell"]
        assert abs(b["share_curves"]["32"]["top_share"]
                   - row["top32_share_sum"]) < tol, row["cell"]
        assert abs(b["pos0_top_coord_share"]
                   - row["pos0_top_coord_share"]) < tol, row["cell"]
        checked += 1
    return checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    man = json.load(open(a.manifest))
    os.makedirs(a.out_dir, exist_ok=True)

    all_rows, preflight_log = [], {}
    for s in man["sweeps"]:
        rows = sweep(s["npz"], s["flags"], s["tag"])
        if s.get("atlas_jsonl"):
            preflight_log[s["tag"]] = preflight(rows, s["atlas_jsonl"])
        all_rows.extend(rows)
        print(f"[vesalius] {s['tag']}: {len(rows)} cells")

    cols = sorted({k for r in all_rows for k in r})
    with open(os.path.join(a.out_dir, "vesalius_cells.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    json.dump({"n_cells": len(all_rows), "preflight_cells_checked": preflight_log,
               "k_ladder": list(K_LADDER), "seed": SEED, "spike_bar": SPIKE_BAR},
              open(os.path.join(a.out_dir, "vesalius_sweep_summary.json"), "w"),
              indent=2)
    print(f"[vesalius] {len(all_rows)} cells -> {a.out_dir}")


if __name__ == "__main__":
    main()
