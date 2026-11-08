"""STAGEDIFF statistics — spec §7 (studies/stagediff/2026-09-04_SPEC.md).

Every interval is a paired ITEM bootstrap (2000 resamples, percentile 2.5/97.5), rates carry Wilson
intervals, and the six hop phenomenon tests go through Benjamini–Hochberg at q = 0.10. Per-item vectors
are aligned by item hash BEFORE they reach this module (`align`), so a resample draws the same items on
both sides. NaN in → assertion out: a dead readout is stamped VOID upstream, never scored here.

Weight-free; numpy only. `_dryrun()` checks each estimator on inputs with a known answer and that each
gate can fail (a rule with no failing input is not a rule).
"""
from __future__ import annotations

import math

import numpy as np

N_BOOT = 2000
SEED = 1337
Q_BH = 0.10


def _vec(x, name="x"):
    v = np.asarray(x, float)
    assert v.ndim == 1 and v.size >= 1, f"{name}: need a 1-d vector, got shape {v.shape}"
    assert np.isfinite(v).all(), f"{name}: {int((~np.isfinite(v)).sum())} non-finite entries — VOID upstream, not scored"
    return v


def align(per_item_a: dict, per_item_b: dict, *more: dict):
    """Intersect {hash: value} dicts on their keys (sorted) → tuple of aligned float arrays. Asserts the
    overlap is total: a missing hash is a pairing defect, not a row to drop silently."""
    ds = (per_item_a, per_item_b) + more
    keys = sorted(set(ds[0]))
    for i, d in enumerate(ds[1:], 1):
        missing = set(keys) ^ set(d)
        assert not missing, f"pairing defect: arm {i} differs on {len(missing)} item hashes, e.g. {sorted(missing)[:3]}"
    return tuple(np.asarray([d[k] for k in keys], float) for d in ds)


def boot_mean(x, n_boot=N_BOOT, seed=SEED) -> dict:
    v = _vec(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(1)
    return {"point": float(v.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "se": float(m.std(ddof=1)), "n": int(v.size)}


def boot_paired_diff(a, b, n_boot=N_BOOT, seed=SEED) -> dict:
    """mean(b) − mean(a) over paired item resamples. `p_le0` = fraction of draws with diff ≥ 0 when the
    point is negative (a LOSS test: how often the resampled child is not below the parent); the BH
    family in §7 uses this one-sided bootstrap p for 'hop loses'."""
    va, vb = _vec(a, "a"), _vec(b, "b")
    assert va.size == vb.size, (va.size, vb.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, va.size, size=(n_boot, va.size))
    d = vb[idx].mean(1) - va[idx].mean(1)
    return {"point": float(vb.mean() - va.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "se": float(d.std(ddof=1)),
            "p_not_below": float((d >= 0).mean()), "n": int(va.size)}


def boot_loss_fraction(parent, child, n_boot=N_BOOT, seed=SEED, return_draws=False) -> dict:
    """L = (acc_P − acc_C) / acc_P with a paired bootstrap; draws with acc_P = 0 are excluded and counted.
    `return_draws` adds the kept draws (for an interval comparison of two L's; review CK-12, 2026-09-05)."""
    p, c = _vec(parent, "parent"), _vec(child, "child")
    assert p.size == c.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, p.size, size=(n_boot, p.size))
    ap, ac = p[idx].mean(1), c[idx].mean(1)
    ok = ap > 0
    L = (ap[ok] - ac[ok]) / ap[ok]
    point = float((p.mean() - c.mean()) / p.mean()) if p.mean() > 0 else None
    if ok.sum() == 0:            # the parent scores 0 on every draw: the fraction is undefined, said so
        return {"point": None, "lo": None, "hi": None, "excluded_draws": int(n_boot), "n": int(p.size),
                "status": "undefined (parent accuracy 0)", **({"draws": []} if return_draws else {})}
    out = {"point": point, "lo": float(np.percentile(L, 2.5)), "hi": float(np.percentile(L, 97.5)),
           "excluded_draws": int((~ok).sum()), "n": int(p.size), "status": "reported"}
    if return_draws:
        out["draws"] = L
    return out


def interval_compare(draws_a, draws_b, scale_b: float = 0.5) -> dict:
    """CI of D = A − scale_b·B from two INDEPENDENT bootstrap draw vectors (different item populations, so no pairing):
    'below' when the 97.5th percentile of D is < 0, 'above' when the 2.5th is ≥ 0, else 'undecided'."""
    a, b = np.asarray(draws_a, float), np.asarray(draws_b, float)
    n = min(a.size, b.size)
    assert n >= 100, f"too few draws for an interval comparison ({n})"
    D = a[:n] - scale_b * b[:n]
    lo, hi = float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))
    return {"lo": lo, "hi": hi, "n_draws": int(n), "verdict": "below" if hi < 0 else ("above" if lo >= 0 else "undecided")}


def boot_recovery(parent, child, arm, n_boot=N_BOOT, seed=SEED) -> dict:
    """R_x = (acc_arm − acc_child) / (acc_parent − acc_child), paired. The ratio is reported only when the
    lost margin's CI excludes 0 (else 'N/A'); the absolute difference arm − child always is. Mirrors
    `matrix_core.bootstrap_fraction` (same seed recipe) with the R naming of spec §2."""
    p, c, x = _vec(parent, "parent"), _vec(child, "child"), _vec(arm, "arm")
    assert p.size == c.size == x.size, (p.size, c.size, x.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, p.size, size=(n_boot, p.size))
    lost = p[idx].mean(1) - c[idx].mean(1)
    diff = x[idx].mean(1) - c[idx].mean(1)
    out = {"n": int(p.size), "lost_point": float(p.mean() - c.mean()),
           "lost_lo": float(np.percentile(lost, 2.5)), "lost_hi": float(np.percentile(lost, 97.5)),
           "diff_point": float(x.mean() - c.mean()),
           "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5)),
           "diff_se": float(diff.std(ddof=1))}
    if out["lost_lo"] > 0:
        ok = lost > 0
        r = diff[ok] / lost[ok]
        out.update({"R": float(out["diff_point"] / out["lost_point"]), "R_lo": float(np.percentile(r, 2.5)),
                    "R_hi": float(np.percentile(r, 97.5)), "R_se": float(r.std(ddof=1)),
                    "R_status": "reported", "R_excluded_draws": int((~ok).sum())})
    else:
        out.update({"R": None, "R_lo": None, "R_hi": None, "R_se": None,
                    "R_status": "N/A (lost-margin CI includes 0)"})
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    assert 0 <= k <= n, (k, n)
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": p, "lo": max(0.0, mid - half), "hi": min(1.0, mid + half), "k": int(k), "n": int(n)}


def bh_reject(pvals, q: float = Q_BH) -> list:
    p = np.asarray(pvals, float)
    assert p.size >= 1 and np.isfinite(p).all(), p
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(ranked <= thresh)
    k = int(below.max()) + 1 if below.size else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej.tolist()


N_BOOT_EXPLORATORY = 20000   # review 2026-09-05 (plan C5): exploratory re-estimates converge here; the frozen rule stays at N_BOOT


def mcnemar_exact(lost: int, gained: int) -> dict:
    """Exact one-sided McNemar on the discordant pairs of a parent→child comparison: P(X ≤ gained | n = lost + gained,
    p = ½) — how often this few gains would be seen if losses and gains were equally likely. Pure math, no scipy."""
    assert lost >= 0 and gained >= 0, (lost, gained)
    n = lost + gained
    if n == 0:
        return {"lost": 0, "gained": 0, "n_discordant": 0, "p_one_sided": None}
    p = sum(math.comb(n, k) for k in range(0, gained + 1)) / 2 ** n
    return {"lost": int(lost), "gained": int(gained), "n_discordant": int(n), "p_one_sided": float(p)}


def boot_median(x, n_boot=N_BOOT, seed=SEED) -> dict:
    v = _vec(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = np.median(v[idx], axis=1)
    return {"point": float(np.median(v)), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "n": int(v.size)}


def _dryrun():
    rng = np.random.default_rng(0)
    p = (rng.random(200) < 0.8).astype(float)
    c = (rng.random(200) < 0.5).astype(float)
    x = np.where(rng.random(200) < 0.5, p, c)          # recovers about half of the margin
    r = boot_recovery(p, c, x)
    assert r["R_status"] == "reported" and 0.2 < r["R"] < 0.8, r
    assert r["lost_lo"] > 0
    same = boot_recovery(p, p, p)
    assert same["R_status"].startswith("N/A"), same          # no loss → ratio refused
    d = boot_paired_diff(p, c)
    assert d["point"] < 0 and d["hi"] < 0 and d["p_not_below"] < 0.01, d
    lf = boot_loss_fraction(p, c)
    assert 0.2 < lf["point"] < 0.5 and lf["lo"] < lf["point"] < lf["hi"], lf
    assert boot_loss_fraction(np.zeros(30), np.zeros(30))["point"] is None       # undefined, not a number
    lfd = boot_loss_fraction(p, c, return_draws=True)
    assert len(lfd["draws"]) == N_BOOT - lfd["excluded_draws"]
    ic = interval_compare(lfd["draws"], lfd["draws"], scale_b=0.5)               # L − ½·L > 0 on every draw → 'above'
    assert ic["verdict"] == "above" and ic["lo"] > 0, ic
    assert interval_compare(lfd["draws"], lfd["draws"], scale_b=2.0)["verdict"] == "below"
    perm = np.random.default_rng(1).permutation(lfd["draws"])               # same distribution, independent order → D straddles 0
    assert interval_compare(lfd["draws"], perm, scale_b=1.0)["verdict"] == "undecided"
    w = wilson(0, 10)
    assert w["lo"] == 0.0 and w["hi"] > 0.2
    w = wilson(10, 10)
    assert w["hi"] > 0.999 and w["lo"] < 0.8
    assert bh_reject([0.001, 0.02, 0.9, 0.5, 0.7, 0.8]) == [True, True, False, False, False, False]
    assert bh_reject([0.5] * 6) == [False] * 6
    mc = mcnemar_exact(12, 3)                                   # H5′: 12 lost, 3 gained → 576 / 32768
    assert abs(mc["p_one_sided"] - 0.017578125) < 1e-12 and mc["n_discordant"] == 15, mc
    assert mcnemar_exact(0, 0)["p_one_sided"] is None and mcnemar_exact(5, 5)["p_one_sided"] > 0.5
    assert N_BOOT_EXPLORATORY > N_BOOT
    a, b = align({"h1": 1, "h2": 0}, {"h2": 1, "h1": 1})
    assert a.tolist() == [1.0, 0.0] and b.tolist() == [1.0, 1.0]
    try:
        align({"h1": 1}, {"h2": 1}); raise SystemExit("align must fail on a pairing defect")
    except AssertionError:
        pass
    try:
        boot_mean([1.0, float("nan")]); raise SystemExit("NaN must fail loud")
    except AssertionError:
        pass
    print("PASS stagediff_stats dry-run (recovery ratio, N/A branch, paired diff, loss fraction, Wilson, BH, align, NaN)")


if __name__ == "__main__":
    _dryrun()
