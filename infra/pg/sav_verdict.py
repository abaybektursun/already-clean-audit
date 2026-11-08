"""SAV-A §6 decision arithmetic (pure; truth-table-tested before freeze).

savings(c) = AUC_log2n[S_c] − AUC_log2n[S_random] on the confirmation half,
trapezoid over the pre-named ladder of seed-mean per-item arrays; ONE shared
item-index resample per bootstrap draw across every cell of every checkpoint;
den ≤ 0 draws dropped + counted (≥50% survival or HALT). The m=6 family
(T1–T3 levels one-sided, T4–T6 drops two-sided), BH q=0.05, and the
pre-registered precedence: collapsed → consumed → conserved → indeterminate.
"""
from __future__ import annotations

LADDER = (256, 1000, 4000, 16000)
CKPTS = ("PG1", "pi0", "pi05")          # tested checkpoints (base anchors the scale)


def _auc_log2n(s_values, ns):
    """Trapezoid AUC of S over log2(n), normalized by the log2-span (so the
    unit is mean-S-over-the-ladder; comparable across checkpoints)."""
    import math
    assert len(s_values) == len(ns) >= 2
    xs = [math.log2(n) for n in ns]
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (s_values[i] + s_values[i + 1]) * (xs[i + 1] - xs[i])
    return area / (xs[-1] - xs[0])


def savings_draws(ckpt_ladders: dict, random_ladder: dict, base_pi, floor_pi,
                  ns=LADDER, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Bootstrap engine over ITEMS AND SEEDS (spec §7; the W17-M5 gap closed):
    ckpt_ladders[c][n] and random_ladder[n] = LISTS of per-seed per-item arrays
    (confirmation half, aligned). Per draw: ONE shared item-index resample AND,
    per cell, a resample-with-replacement of its seed arrays before averaging.
    Observed point uses the plain seed mean. Returns draws + observed + drops."""
    import numpy as np
    b = np.asarray(base_pi, float)
    f = np.asarray(floor_pi, float)
    n_items = b.size
    C = {c: {n: [np.asarray(a, float) for a in ckpt_ladders[c][n]] for n in ns}
         for c in ckpt_ladders}
    R = {n: [np.asarray(a, float) for a in random_ladder[n]] for n in ns}
    for c in C:
        for n in ns:
            assert all(a.size == n_items for a in C[c][n]), f"{c}@{n} misaligned"
            assert len(C[c][n]) >= 1
    rng = np.random.default_rng(seed)

    def cell_mean(arrs, sl, seed_idx=None):
        chosen = arrs if seed_idx is None else [arrs[j] for j in seed_idx[:len(arrs)]]
        return float(np.mean([a[sl].mean() for a in chosen]))

    def point(idx=None, resample_seeds=False):
        sl = slice(None) if idx is None else idx
        den = b[sl].mean() - f[sl].mean()
        if den <= 0:
            return None
        out = {}
        sidx = None
        if resample_seeds:
            k = max(len(next(iter(R.values()))), max(len(C[c][ns[0]]) for c in C))
            sidx = rng.integers(0, 10 ** 9, k).tolist()      # per-draw seed picks (mod len below)
        def pick(arrs):
            if sidx is None:
                return arrs
            return [arrs[s % len(arrs)] for s in sidx[:len(arrs)]]
        r_auc = _auc_log2n([(cell_mean(pick(R[n]), sl) - f[sl].mean()) / den for n in ns], ns)
        for c in C:
            c_auc = _auc_log2n([(cell_mean(pick(C[c][n]), sl) - f[sl].mean()) / den
                                for n in ns], ns)
            out[c] = c_auc - r_auc
        return out

    obs = point()
    assert obs is not None, "point denom ≤ 0"
    draws = {c: [] for c in C}
    dropped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        v = point(idx, resample_seeds=True)
        if v is None:
            dropped += 1
            continue
        for c in C:
            draws[c].append(v[c])
    kept = len(next(iter(draws.values())))
    assert kept >= n_boot // 2, f"degenerate bootstrap: {dropped}/{n_boot} dropped"
    return {"observed": obs, "draws": {c: np.asarray(d) for c, d in draws.items()},
            "n_draws_kept": kept, "n_draws_dropped": dropped}


def family_tests(sd: dict) -> dict:
    """The m=6 family from savings_draws output. One/two-sided +1-smoothed
    percentile-inversion p-values; CIs."""
    import numpy as np

    def ci(d):
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    def p1(d):                                   # one-sided: > 0
        return float((1 + (d <= 0).sum()) / (1 + d.size))

    def p2(d):                                   # two-sided vs 0
        lo = (1 + (d <= 0).sum()) / (1 + d.size)
        hi = (1 + (d >= 0).sum()) / (1 + d.size)
        return float(min(1.0, 2.0 * min(lo, hi)))

    D = sd["draws"]
    obs = sd["observed"]
    out = {}
    for t, c in (("T1_sav_PG1", "PG1"), ("T2_sav_pi0", "pi0"), ("T3_sav_pi05", "pi05")):
        out[t] = {"point": float(obs[c]), "ci95": ci(D[c]), "p": p1(D[c])}
    for t, (a, bb) in (("T4_drop1", ("PG1", "pi0")), ("T5_drop2", ("pi0", "pi05")),
                       ("T6_droptotal", ("PG1", "pi05"))):
        d = D[a] - D[bb]
        out[t] = {"point": float(obs[a] - obs[bb]), "ci95": ci(d), "p": p2(d)}
    return out


def auc_weight_l2(ns=LADDER) -> float:
    """‖w‖₂ of the normalized trapezoid weights over log2(ns) — the iid-seed-noise
    scale factor mapping per-cell δ_S to the savings-AUC scale (critic M1a)."""
    import math
    xs = [math.log2(n) for n in ns]
    span = xs[-1] - xs[0]
    w = [0.0] * len(xs)
    for i in range(len(xs) - 1):
        h = (xs[i + 1] - xs[i]) / (2 * span)
        w[i] += h
        w[i + 1] += h
    return math.sqrt(sum(x * x for x in w))


def sav_verdicts(tests: dict, delta_level: float, delta_pair: float,
                 unstable_tests=(), q: float = 0.05) -> dict:
    """Pre-registered mapping with precedence collapsed → consumed → conserved."""
    from infra.pg.w16_probes import bh_fdr
    names = ["T1_sav_PG1", "T2_sav_pi0", "T3_sav_pi05", "T4_drop1", "T5_drop2", "T6_droptotal"]
    assert set(tests) == set(names), f"family mismatch: {sorted(tests)}"
    bh = bh_fdr({k: v["p"] for k, v in tests.items()}, q=q)

    def res(t):
        if t in unstable_tests:
            return "non-result"
        return "sig" if bh[t]["significant"] else "ns"

    def inside(t, delta):
        lo, hi = tests[t]["ci95"]
        return -delta <= lo and hi <= delta

    r = {t: res(t) for t in names}
    pts = {c: tests[t]["point"] for t, c in
           (("T1_sav_PG1", "PG1"), ("T2_sav_pi0", "pi0"), ("T3_sav_pi05", "pi05"))}

    # PER-TEST voiding (prereg §6): a non-result test FAILS any branch condition
    # that needs it; branches not using it are unaffected. Global non-result only
    # when every test is voided.
    if all(r[t] == "non-result" for t in names):
        verdict = "non-result (instability: all tests voided)"
    elif (r["T2_sav_pi0"] == "ns" and inside("T2_sav_pi0", delta_level)
          and r["T3_sav_pi05"] == "ns" and inside("T3_sav_pi05", delta_level)):
        verdict = "H_collapsed"
        if r["T6_droptotal"] == "sig" and r["T1_sav_PG1"] == "sig":
            verdict += " (consumed-to-floor qualifier: T1+T6 significant)"
    elif (r["T6_droptotal"] == "sig" and tests["T6_droptotal"]["point"] > 0
          and pts["PG1"] > pts["pi0"] and pts["pi0"] >= pts["pi05"]   # strict hop-1 (spec §7)
          and r["T1_sav_PG1"] == "sig"):
        verdict = "H_consumed"
    elif (all(r[t] == "sig" for t in ("T1_sav_PG1", "T2_sav_pi0", "T3_sav_pi05"))
          and all(r[t] == "ns" and inside(t, delta_pair)
                  for t in ("T4_drop1", "T5_drop2", "T6_droptotal"))):
        verdict = "H_conserved (equality legs: qualified bounded-difference)"
    else:
        verdict = "indeterminate"
        if any(r[t] == "non-result" for t in names):
            verdict += f" (instability-touched: {[t for t in names if r[t]=='non-result']})"
    return {"bh": bh, "test_results": r, "points": pts, "VERDICT": verdict,
            "pi0_vs_pi05_T5": {"point": tests["T5_drop2"]["point"],
                               "ci95": tests["T5_drop2"]["ci95"],
                               "result": r["T5_drop2"]}}
