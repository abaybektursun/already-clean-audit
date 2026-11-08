"""W17 §6 decision arithmetic (pure; the truth-table-tested verdict path).

Implements the post-critic frozen procedure exactly:
  * paired_arm_diff_s — seed-mean per-item arrays for two arms on the SAME
    items, ONE shared-index item bootstrap of ΔS = (ā−b̄)/(BEFORE−FLOOR);
    one-sided +1-smoothed p; den≤0 draws dropped+counted (≥50% or HALT).
  * delta_eq — the only band: pooled p95 of |seed-pair S differences| at the
    named band-source cells, via the same shared-index engine.
  * w17_verdicts — the full §6 truth table (BH over the m=5 primary family;
    any-of-across-n*/both-conjuncts-within-n*; equivalence-supported branches
    qualified; indeterminate branch; per-cell seed-instability rule).

No model, no GPU, numpy only inside functions. Fail-loud.
"""
from __future__ import annotations

PRIMARY_TESTS = ("sav_sup_1k", "sav_floor_1k", "sav_sup_16k", "sav_floor_16k", "disp_16k")


def seed_mean_items(per_seed_arrays):
    """Per-item mean across seed runs (lists of equal-length 0/1 arrays)."""
    import numpy as np
    A = np.asarray(per_seed_arrays, dtype=float)
    assert A.ndim == 2 and A.shape[0] >= 1, f"need (seeds, items), got {A.shape}"
    return A.mean(axis=0)


def paired_arm_diff_s(a_seed_arrays, b_seed_arrays, before_pi, floor_pi,
                      n_boot: int = 2000, seed: int = 1337) -> dict:
    """ΔS between two arms (a − b) on identical items; statistics per prereg §6."""
    import numpy as np
    from infra.pg.w16_probes import _boot_draws
    a = seed_mean_items(a_seed_arrays)
    b = seed_mean_items(b_seed_arrays)
    bb = np.asarray(before_pi, dtype=float)
    ff = np.asarray(floor_pi, dtype=float)
    n = a.size
    assert b.size == n == bb.size == ff.size, \
        f"misaligned: {a.size}/{b.size}/{bb.size}/{ff.size}"

    den0 = bb.mean() - ff.mean()
    assert den0 > 0, f"point denom ≤ 0 ({den0:.4f})"
    point = float((a.mean() - b.mean()) / den0)

    def stat(idx):
        den = bb[idx].mean() - ff[idx].mean()
        if den <= 0:
            return None
        return float((a[idx].mean() - b[idx].mean()) / den)

    draws, dropped = _boot_draws(stat, n, n_boot, seed)
    p = float((1 + (draws <= 0).sum()) / (1 + draws.size))
    return {"point": point,
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "p_one_sided": p, "n_items": int(n),
            "n_draws_kept": int(draws.size), "n_draws_dropped": int(dropped)}


def per_seed_s(per_seed_arrays, before_pi, floor_pi):
    """Per-seed S_recover points (for spreads + δ sources)."""
    import numpy as np
    bb = np.asarray(before_pi, dtype=float)
    ff = np.asarray(floor_pi, dtype=float)
    den = bb.mean() - ff.mean()
    assert den > 0
    return [float((np.asarray(s, dtype=float).mean() - ff.mean()) / den)
            for s in per_seed_arrays]


def delta_eq(band_source_cells, before_pi, floor_pi, n_boot: int = 2000,
             seed: int = 1337, pct: float = 95.0) -> dict:
    """δ_eq: pooled p95 of |seed-pair S differences| across the named band-source
    cells. band_source_cells = list of per-seed-array lists (each cell: ≥2 seeds).
    Shared-index bootstrap: per draw, resample items once, recompute every
    seed-pair |ΔS| in every cell, pool all values of that draw."""
    import numpy as np
    from itertools import combinations

    from infra.pg.w16_probes import _boot_draws
    bb = np.asarray(before_pi, dtype=float)
    ff = np.asarray(floor_pi, dtype=float)
    cells = [[np.asarray(s, dtype=float) for s in cell] for cell in band_source_cells]
    assert cells and all(len(c) >= 2 for c in cells), "each band-source cell needs ≥2 seeds"
    n = bb.size

    def stat(idx):
        den = bb[idx].mean() - ff[idx].mean()
        if den <= 0:
            return None
        vals = []
        for cell in cells:
            ms = [s[idx].mean() for s in cell]
            vals.extend(abs(x - y) / den for x, y in combinations(ms, 2))
        return float(max(vals))          # per-draw pooled max |seed-pair ΔS|

    draws, dropped = _boot_draws(stat, n, n_boot, seed)
    return {"delta_eq": float(np.percentile(draws, pct)), "pct": pct,
            "n_cells": len(cells), "n_draws_kept": int(draws.size),
            "n_draws_dropped": int(dropped)}


def w17_verdicts(tests: dict, deq: dict, spreads: dict, q: float = 0.05) -> dict:
    """The §6 truth table, mechanical.

    tests: {name in PRIMARY_TESTS: paired_arm_diff_s output}
    deq:   {"1k": δ, "16k": δ}
    spreads: {cell_name: {"spread": max−min per-seed S, "n_star": "1k"|"16k"}}
             — cells entering a test; instability ⇒ that test → non-result.
    """
    from infra.pg.w16_probes import bh_fdr
    assert set(tests) == set(PRIMARY_TESTS), f"family mismatch: {sorted(tests)}"
    bh = bh_fdr({k: v["p_one_sided"] for k, v in tests.items()}, q=q)

    unstable_tests = set()
    for cell, info in spreads.items():
        if info["spread"] > deq[info["n_star"]]:
            unstable_tests.update(info["tests"])

    def res(name):
        if name in unstable_tests:
            return "non-result"
        return "sig" if bh[name]["significant"] else "ns"

    def ci_inside(name, n_star):
        lo, hi = tests[name]["ci95"]
        return bool(-deq[n_star] <= lo and hi <= deq[n_star])

    r = {name: res(name) for name in PRIMARY_TESTS}
    savings_1k = r["sav_sup_1k"] == "sig" and r["sav_floor_1k"] == "sig"
    savings_16k = r["sav_sup_16k"] == "sig" and r["sav_floor_16k"] == "sig"
    h_savings = bool(savings_1k or savings_16k)

    four = ["sav_sup_1k", "sav_floor_1k", "sav_sup_16k", "sav_floor_16k"]
    all_ns = all(r[t] == "ns" for t in four)
    all_inside = (ci_inside("sav_sup_1k", "1k") and ci_inside("sav_floor_1k", "1k")
                  and ci_inside("sav_sup_16k", "16k") and ci_inside("sav_floor_16k", "16k"))
    if h_savings:
        h_no_savings = "killed"
    elif all_ns and all_inside:
        h_no_savings = "supported (qualified bounded-difference)"
    elif any(r[t] == "non-result" for t in four):
        h_no_savings = "non-result (instability)"
    else:
        h_no_savings = "indeterminate-savings"

    if r["disp_16k"] == "sig":
        h_disp = "killed (capability > retain)"
    elif r["disp_16k"] == "non-result":
        h_disp = "non-result (instability)"
    elif ci_inside("disp_16k", "16k"):
        h_disp = "supported (qualified bounded-difference)"
    else:
        h_disp = "indeterminate"

    if r["sav_floor_16k"] == "sig":
        h_format = "killed (capability > shuffled)"
    elif r["sav_floor_16k"] == "non-result":
        h_format = "non-result (instability)"
    elif ci_inside("sav_floor_16k", "16k"):
        h_format = "supported (qualified bounded-difference)"
    else:
        h_format = "indeterminate"

    return {"bh": bh, "test_results": r, "unstable_tests": sorted(unstable_tests),
            "savings_per_nstar": {"1k": savings_1k, "16k": savings_16k},
            "H_savings": "supported" if h_savings else "not demonstrated",
            "H_no_savings": h_no_savings, "H_displacement": h_disp, "H_format": h_format}
