#!/usr/bin/env python3
"""AMUNDSEN scoring — the §6 decision rules, evaluated verbatim and offline.

One function per arm. Each takes ALREADY-LOADED artifacts (so every rule is
unit-testable weight-free) and returns {"verdict": TOKEN, ...evidence}. Every
validity gate is CONSUMED INSIDE the verdict expression (the B1/X2 lesson): a
failed gate returns VOID-INSTRUMENT / COVERAGE-IMPOSSIBLE as THE verdict, it is
never recorded beside a verdict that ignored it.

House statistics: bootstrap CIs with >= 2000 resamples, paired designs resample
PAIRS (the same window / item indices across every arm in the comparison).

CLI:  python3 analysis_amundsen.py <arm> --dir <dir of downloaded artifacts>
"""
import argparse
import json
import os
import zlib

import numpy as np

N_BOOT = 2000


# ---------------------------------------------------------------- primitives
def boot_mean_ci(x, n_boot=N_BOOT, seed=1337, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    b = x[idx].mean(axis=1)
    return {"mean": float(x.mean()),
            "lo": float(np.percentile(b, 100 * alpha / 2)),
            "hi": float(np.percentile(b, 100 * (1 - alpha / 2))), "n": int(x.size)}


def boot_paired_ci(a, b, n_boot=N_BOOT, seed=1337, alpha=0.05):
    """CI of mean(a) - mean(b) resampling the SHARED index."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n_boot, a.size))
    d = (a[idx] - b[idx]).mean(axis=1)
    return {"diff": float(a.mean() - b.mean()),
            "lo": float(np.percentile(d, 100 * alpha / 2)),
            "hi": float(np.percentile(d, 100 * (1 - alpha / 2))), "n": int(a.size)}


def overlap(ci_a, ci_b):
    return not (ci_a["hi"] < ci_b["lo"] or ci_b["hi"] < ci_a["lo"])


def spearman_exact_p(x, y):
    """Two-sided exact permutation p for Spearman rho (n <= 8 -> exact)."""
    from itertools import permutations
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    rx, ry = _rank(x), _rank(y)
    rho = _pearson(rx, ry)
    n = x.size
    assert n <= 8, f"exact permutation needs n <= 8, got {n}"
    hits = tot = 0
    for p in permutations(range(n)):
        tot += 1
        if abs(_pearson(rx, ry[list(p)])) >= abs(rho) - 1e-12:
            hits += 1
    return float(rho), hits / tot


def _rank(x):
    """Ranks with TIES AVERAGED. Without this a constant series ranks 1..n in
    index order and Spearman returns rho = 1 against any increasing ladder — a
    dead readout would pass the monotonicity gate. (Caught by the dry-run's
    dead-positive-control case.)"""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1)
    xs = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def bh_reject(pvals, q=0.10):
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out.tolist()


def loglog_slope(steps, values):
    s, v = np.asarray(steps, np.float64), np.asarray(values, np.float64)
    ok = (s > 0) & (v > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(s[ok]), np.log(v[ok]), 1)[0])


# ---------------------------------------------------------------------- AM-1
def am1(rows, window_nll, banked_rows, n_boot=N_BOOT):
    """Energy account vs core account at 14B (§6.1).

    C = mean paired dppl(core32); E = mean paired dppl(energymatched32) where
    the arm is the MEAN OVER ITS DRAWS (per-draw verdicts banked alongside).
    Validity, inside the verdict: topenergy32 reproduces the banked cost's sign
    and order of magnitude, AND the magnitude-matched planted arm does not
    exceed 0.5*C.
    """
    per = window_nll["per_window"]

    def dppl(arm):
        n0 = np.asarray(per["none"]["nll"], np.float64)
        t0 = np.asarray(per["none"]["ntok"], np.float64)
        na = np.asarray(per[arm]["nll"], np.float64)
        ta = np.asarray(per[arm]["ntok"], np.float64)
        assert np.array_equal(t0, ta), f"window alignment broken for {arm}"
        return np.exp(na / ta) - np.exp(n0 / t0)

    draws = lambda pfx: sorted(a for a in per if a.startswith(pfx + "_s"))
    e_arms, m_arms = draws("energymatched32"), draws("magmatched32")
    assert e_arms and m_arms, (list(per),)
    C_w = dppl("core32")
    E_w = np.mean([dppl(a) for a in e_arms], axis=0)
    M_w = np.mean([dppl(a) for a in m_arms], axis=0)
    ci_C, ci_E = boot_mean_ci(C_w, n_boot), boot_mean_ci(E_w, n_boot, seed=1338)
    ci_M = boot_mean_ci(M_w, n_boot, seed=1339)
    ci_CE = boot_paired_ci(C_w, E_w, n_boot, seed=1340)
    # the fraction of core's cost the energy control reproduces. The frozen rule
    # is a THREE-WAY test on overlap, which a precise instrument can push to
    # CORE-EXCEEDS even when energy explains ~all of the cost; this ratio is
    # registered (pre-launch) as the reading that carries the science either way.
    _rng = np.random.default_rng(1341)
    _idx = _rng.integers(0, C_w.size, (n_boot, C_w.size))
    _r = E_w[_idx].mean(axis=1) / C_w[_idx].mean(axis=1)
    ci_ratio = {"ratio": float(E_w.mean() / C_w.mean()),
                "lo": float(np.percentile(_r, 2.5)),
                "hi": float(np.percentile(_r, 97.5))}

    ppl = {r["arm"]: r["ppl"] for r in rows["arms"]}
    bank = {r["arm"]: r["ppl"] for r in banked_rows["arms"]}
    d_top, d_top_bank = ppl["topenergy32"] - ppl["none"], bank["topenergy32"] - bank["none"]
    pos_ok = bool(np.sign(d_top) == np.sign(d_top_bank)
                  and int(np.floor(np.log10(abs(d_top))))
                  == int(np.floor(np.log10(abs(d_top_bank)))))
    planted_ok = bool(ci_M["mean"] <= 0.5 * ci_C["mean"])

    if not (pos_ok and planted_ok):
        verdict = "VOID-INSTRUMENT"
    elif ci_E["mean"] >= 0.7 * ci_C["mean"] and overlap(ci_C, ci_E):
        verdict = "ENERGY-ACCOUNT"
    elif ci_CE["lo"] > 0 and not overlap(ci_C, ci_E):
        verdict = "CORE-EXCEEDS-ENERGY"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "C": ci_C, "E": ci_E, "planted": ci_M,
            "C_minus_E": ci_CE, "E_over_C": ci_ratio,
            "validity": {"positive_control_reproduces": pos_ok,
                         "dppl_topenergy32": d_top, "dppl_topenergy32_banked": d_top_bank,
                         "planted_below_half_C": planted_ok},
            "per_draw": {a: boot_mean_ci(dppl(a), n_boot, seed=17 + i)
                         for i, a in enumerate(e_arms + m_arms)},
            "arm_ppl": ppl}


# ---------------------------------------------------------------------- AM-2
def am2(cells, horizon=1200, ratio_hi=1.10, ratio_lo=1.03,
        min_lrs=4, min_span=8.0):
    """Retune closes the rotated arm's wikitext-ppl cost, or it does not (§6.2).

    `cells` = [{"lr":, "steps":, "gauge":, "seed":, "ppl":, "task_loss":,
    "diverged": bool}, ...] flattened from every lr cell's M-Q_stage2_result.json.
    """
    at_h = [c for c in cells if c["steps"] == horizon]
    live = [c for c in at_h if not c["diverged"]]
    by = {}
    for c in live:
        by.setdefault((c["gauge"], c["lr"]), []).append(c["ppl"])
    sides = {}
    for gauge in ("hadamard", "unrotated"):
        lrs = sorted({lr for (g, lr) in by if g == gauge})
        sides[gauge] = {"lrs": lrs,
                        "mean_ppl_by_lr": {lr: float(np.mean(by[(gauge, lr)]))
                                           for lr in lrs}}
    cov = {g: (len(s["lrs"]) >= min_lrs
               and (max(s["lrs"]) / min(s["lrs"]) >= min_span if s["lrs"] else False))
           for g, s in sides.items()}
    if not all(cov.values()):
        return {"verdict": "COVERAGE-IMPOSSIBLE", "sides": sides, "coverage": cov}
    if not sides["hadamard"]["lrs"] or not sides["unrotated"]["lrs"]:
        return {"verdict": "VOID-INSTRUMENT", "sides": sides, "coverage": cov}

    best = {g: min(sides[g]["mean_ppl_by_lr"], key=sides[g]["mean_ppl_by_lr"].get)
            for g in sides}
    # the other criterion, reported beside it: AM-7 tunes on the training-task
    # loss (a practitioner's criterion), AM-2's own verdict on wikitext ppl
    task = {}
    for gauge in sides:
        lrs = sides[gauge]["lrs"]
        m = {lr: float(np.mean([c["task_loss"] for c in live
                                if c["gauge"] == gauge and c["lr"] == lr])) for lr in lrs}
        sides[gauge]["mean_task_loss_by_lr"] = m
        task[gauge] = min(m, key=m.get) if m else None
    R = sides["hadamard"]["mean_ppl_by_lr"][best["hadamard"]]
    Uu = sides["unrotated"]["mean_ppl_by_lr"][best["unrotated"]]
    seed_pairs = []
    for s in sorted({c["seed"] for c in live}):
        r = [c["ppl"] for c in live if c["gauge"] == "hadamard"
             and c["lr"] == best["hadamard"] and c["seed"] == s]
        u = [c["ppl"] for c in live if c["gauge"] == "unrotated"
             and c["lr"] == best["unrotated"] and c["seed"] == s]
        if r and u:
            seed_pairs.append({"seed": s, "rotated": r[0], "unrotated": u[0],
                               "ratio": r[0] / u[0]})
    n_up = sum(p["ratio"] > 1 for p in seed_pairs)
    ratio = R / Uu
    all_up = bool(seed_pairs) and n_up == len(seed_pairs)
    all_dn = bool(seed_pairs) and n_up == 0
    if ratio >= ratio_hi and all_up:
        verdict = "INTRINSIC-COST"
    elif ratio <= ratio_lo and (all_up or all_dn):
        verdict = "RETUNE-CLOSES"
    else:
        verdict = "PARTIAL"
    return {"verdict": verdict, "ratio_best_rotated_over_best_unrotated": ratio,
            "best_lr": best, "best_lr_task": task, "R": R, "U": Uu, "per_seed": seed_pairs,
            "n_seeds_ratio_gt_1": n_up, "sides": sides, "coverage": cov,
            "diverged": [{k: c[k] for k in ("gauge", "lr", "seed", "steps")}
                         for c in at_h if c["diverged"]]}


# ---------------------------------------------------------------------- AM-3
def am3(rows, mode="matched", p_max=0.05):
    """Does the model READ the sink coordinates (§6.3)?

    PRIMARY (the frozen rule): Spearman of dppl(core32) against the SIGNED dose
    ladder s. SECONDARY, pre-registered before launch because the frozen rule
    cannot express a symmetric response: Spearman against |log s|, the dose
    MAGNITUDE — a model that reads the coordinates is hurt by moving them in
    either direction, and that response is non-monotone in signed s by
    construction. Both are reported; the primary decides the token.
    """
    base = [r for r in rows if r["arm"] == "none"][0]["ppl"]

    def series(arm, md):
        rs = [r for r in rows if r["arm"] == arm and r["mode"] == md]
        rs = sorted(rs, key=lambda r: r["s"])
        return [r["s"] for r in rs], [r["ppl"] - base for r in rs], rs

    s_core, d_core, _ = series("core32", "nominal")
    twins = {}
    for t in ("energymatched32", "haar32"):
        s_t, d_t, rt = series(t, mode)
        assert s_t == s_core, (t, s_t, s_core)
        twins[t] = d_t
    s_pc, d_pc, _ = series("topenergy32", "nominal")

    rho, p = spearman_exact_p(s_core, d_core)
    mag = [abs(np.log(s)) for s in s_core]
    rho_mag, p_mag = spearman_exact_p(mag, d_core)
    rho_pc, p_pc = spearman_exact_p(s_pc, d_pc)
    rho_pc_mag, p_pc_mag = spearman_exact_p([abs(np.log(s)) for s in s_pc], d_pc)

    lo = [min(twins[t][i] for t in twins) for i in range(len(s_core))]
    hi = [max(twins[t][i] for t in twins) for i in range(len(s_core))]
    outside = [bool(d_core[i] > hi[i] or d_core[i] < lo[i]) for i in range(len(s_core))]
    inside = [not o for o in outside]
    band_overlap = sum(inside)

    ident = [r for r in rows if r["s"] == 1.0 and r["mode"] == "nominal"]
    ident_ok = all(abs(r["ppl"] - base) / base < 1e-3 for r in ident)
    # the positive control must have RANGE, in either the signed or the
    # magnitude reading — otherwise the ladder cannot resolve anything
    pc_ok = bool((abs(rho_pc) == 1.0 and p_pc <= p_max)
                 or (rho_pc_mag > 0 and p_pc_mag <= p_max))

    # §6.3's ROUTED-AROUND ("inside the band at all rungs") and GENERIC ("band
    # overlap at >= 6/7 rungs") overlap as written — all-inside implies 7/7. The
    # discriminator the two branches were written to express is whether the
    # readout MOVES AT ALL: routed-around = scaling the sink does nothing;
    # generic = it does something, and so does every twin. Motion is measured
    # against the range the positive control demonstrates on the same ladder.
    # (Disambiguation frozen pre-launch, journalled; no data was seen.)
    moved = bool(max(abs(v) for v in d_core)
                 >= 0.05 * max(abs(v) for v in d_pc)) if any(d_pc) else False
    if not (ident_ok and pc_ok):
        verdict = "VOID-INSTRUMENT"
    elif p <= p_max and sum(outside) >= 2:
        verdict = "READS-CORE"
    elif all(inside) and not moved:
        verdict = "ROUTED-AROUND"
    elif band_overlap >= 6:
        verdict = "GENERIC"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "ladder": s_core, "dppl_core": d_core,
            "twin_band_lo": lo, "twin_band_hi": hi, "outside_band": outside,
            "spearman_signed": {"rho": rho, "p": p},
            "spearman_magnitude": {"rho": rho_mag, "p": p_mag},
            "positive_control": {"rho": rho_pc, "p": p_pc,
                                 "rho_mag": rho_pc_mag, "p_mag": p_pc_mag,
                                 "dppl": d_pc},
            "twins": twins, "mode": mode, "core_moved_vs_control": moved,
            "validity": {"identity_control_ok": ident_ok,
                         "positive_control_has_range": pc_ok}}


# ---------------------------------------------------------------------- AM-4
def am4(actstats_batches, n_layers, window, early_max_layer=2, hi=20.0, lo=5.0):
    """First-token relocation on a second family (§6.4).

    actstats_batches: [{ "L{layer:02d}_{site}_pos_energy": array }, ...] — one
    dict per eval batch. Concentration = pos_energy[0] / mean(pos_energy), i.e.
    pos0 share x n_positions. The verdict takes the MIN over batches of each
    batch's max (the frozen stability rule), and both median and max are banked.
    """
    def conc(b, L, site):
        pe = np.asarray(b[f"L{L:02d}_{site}_pos_energy"], np.float64)
        return float(pe[0] / pe.mean())

    sites = sorted({k.split("_", 1)[1].rsplit("_pos_energy", 1)[0]
                    for k in actstats_batches[0] if k.endswith("_pos_energy")})
    early = list(range(0, min(early_max_layer + 1, n_layers)))
    late = list(range(max(0, n_layers - 3), n_layers))
    per_batch = []
    for b in actstats_batches:
        allc = {(L, s): conc(b, L, s) for L in range(n_layers) for s in sites}
        per_batch.append({
            "early_max": max(allc[(L, s)] for L in early for s in sites),
            "late_max": max(allc[(L, s)] for L in late for s in sites),
            "all_layer_median": float(np.median([allc[(L, s)] for L in range(n_layers)
                                                 for s in sites])),
            "per_site_max": {s: max(allc[(L, s)] for L in range(n_layers)) for s in sites},
            "per_site_median": {s: float(np.median([allc[(L, s)] for L in range(n_layers)]))
                                for s in sites},
            "argmax_layer": max(((allc[(L, s)], L, s) for L in range(n_layers)
                                 for s in sites))[1]})
    early_max = min(b["early_max"] for b in per_batch)
    late_max = min(b["late_max"] for b in per_batch)
    if early_max >= hi and early_max > late_max:
        verdict = "REPLICATES"
    elif early_max < lo:
        verdict = "DOES-NOT"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "early_max_min_over_batches": early_max,
            "late_max_min_over_batches": late_max, "n_batches": len(per_batch),
            "per_batch": per_batch, "uniform_reference": 1.0,
            "window": window, "n_layers": n_layers, "sites": sites}


# ---------------------------------------------------------------------- AM-5
def am5(rows, step=3000, key="fresh_bottom32_motion_share",
        reform_hi=1.25, scar_lo=2.0, plateau_slope=0.15, _secondary=True):
    """Scar or slow re-formation at >= 3000 steps (§6.5).

    The established-sink level is the UNROTATED arm's value at the SAME step in
    the SAME run (horizon-matched control), which is why the control continues
    alongside. Its own validity gate — the unrotated arm must keep its sink,
    final delivery(core32) < 1 — is consumed in the verdict.

    PRIMARY key is the CUMULATIVE fresh_bottom32_motion_share, because that is
    the tracker §6.5's 1.25x / 2.0x thresholds were calibrated against on the
    banked V2b trajectory; scoring the windowed variant against thresholds set
    on the cumulative one would be a rule swap. The windowed variant — a better
    question ("is the sink re-forming NOW?") — is computed and returned as the
    SECONDARY reading, never as the token.
    """
    def at(gauge, st):
        return [r for r in rows if r["gauge"] == gauge and r["step"] == st]

    ctrl = at("unrotated", step)
    trt = at("hadamard", step)
    if not ctrl or not trt:
        return {"verdict": "VOID-INSTRUMENT", "why": f"missing rows at step {step}",
                "n_rows": len(rows)}
    dkey = "delivery_win" if "delivery_win" in ctrl[0] else "delivery"
    keeps_sink = all(r[dkey]["core32"] < 1.0 for r in ctrl)
    distinct_ok = all(r.get("act_distinct_max", 2) >= 2 for r in trt + ctrl)

    est = float(np.median([r[key] for r in ctrl]))
    obs = float(np.median([r[key] for r in trt]))
    ratio = obs / est
    steps_seen = sorted({r["step"] for r in rows if r["gauge"] == "hadamard"})
    last3 = steps_seen[-3:]
    med = [float(np.median([r[key] for r in at("hadamard", s)])) for s in last3]
    slope = loglog_slope(last3, med)
    plateau = bool(abs(slope) < plateau_slope)

    if not (keeps_sink and distinct_ok):
        verdict = "VOID-INSTRUMENT"
    elif ratio <= reform_hi:
        verdict = "RE-FORMATION"
    elif ratio >= scar_lo and plateau:
        verdict = "SCAR"
    else:
        verdict = "AMBIGUOUS-EXTEND"
    secondary = None
    if _secondary and "fresh_bottom32_motion_share_win" in ctrl[0]:
        secondary = am5(rows, step, "fresh_bottom32_motion_share_win",
                        reform_hi, scar_lo, plateau_slope, _secondary=False)
    return {"verdict": verdict, "ratio": ratio, "established_level": est,
            "observed": obs, "key": key, "step": step, "secondary_windowed": secondary,
            "loglog_slope_last3": slope, "plateau": plateau, "last3_steps": last3,
            "validity": {"unrotated_keeps_sink": keeps_sink,
                         "per_layer_trackers_distinct": distinct_ok,
                         "delivery_key_used": dkey}}


# ---------------------------------------------------------------------- AM-6
def am6(new_rows, banked_rows, banks=("core32", "anti32", "topsigma32", "bottomsigma32"),
        haar=("haar32_s0", "haar32_s1", "haar32_s2"), real_frac=5 / 6, noise_frac=3 / 6):
    """Is the wrong-signed noise trend real (§6.6)?

    Frozen rule: the structured-bank delivery-vs-noise slope keeps its sign in
    >= 5 of 6 (bank x new-seed) cells while haar stays within +-1 sigma of flat.
    The literal "6" presumes 3 banks x 2 seeds; the FRACTIONS are what the rule
    means, so with n_banks x n_seeds cells the thresholds are the same fractions
    (recorded pre-launch, never re-derived after seeing the cells).
    """
    def deliv(rows):
        out = {}
        for r in rows:
            if r.get("layer") is None and "delivery" in r:
                out.setdefault(r.get("sample_seed", "banked"), {})[r["batch"]] = r["delivery"]
        return out

    ref = deliv(banked_rows)
    assert len(ref) == 1, f"banked rows carry {len(ref)} seeds"
    ref_seed = list(ref)[0]
    ref_slope = {b: np.polyfit(np.log([B for B in sorted(ref[ref_seed])]),
                               [ref[ref_seed][B][b] for B in sorted(ref[ref_seed])], 1)[0]
                 for b in banks + haar}
    new = deliv(new_rows)
    cells, n_keep = [], 0
    for seed in sorted(new):
        Bs = sorted(new[seed])
        for b in banks:
            sl = float(np.polyfit(np.log(Bs), [new[seed][B][b] for B in Bs], 1)[0])
            keep = bool(np.sign(sl) == np.sign(ref_slope[b]) and sl != 0)
            n_keep += keep
            cells.append({"seed": seed, "bank": b, "slope": sl,
                          "banked_slope": float(ref_slope[b]), "sign_kept": keep})
    n_cells = len(cells)
    haar_slopes = [float(np.polyfit(np.log(sorted(new[seed])),
                                    [new[seed][B][h] for B in sorted(new[seed])], 1)[0])
                   for seed in sorted(new) for h in haar]
    haar_flat = bool(abs(float(np.mean(haar_slopes)))
                     <= float(np.std(haar_slopes, ddof=1)) if len(haar_slopes) > 1 else False)
    if n_cells == 0:
        verdict = "VOID-INSTRUMENT"
    elif n_keep >= np.ceil(real_frac * n_cells) and haar_flat:
        verdict = "REAL"
    elif n_keep <= np.floor(noise_frac * n_cells):
        verdict = "SEED-NOISE"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "n_cells": n_cells, "n_sign_kept": n_keep,
            "threshold_real": float(np.ceil(real_frac * n_cells)),
            "threshold_noise": float(np.floor(noise_frac * n_cells)),
            "haar_slopes": haar_slopes, "haar_flat": haar_flat,
            "cells": cells, "banked_slopes": {k: float(v) for k, v in ref_slope.items()}}


# ---------------------------------------------------------------------- AM-7
def am7(rows, benchmarks=("mmlu_blankimg", "mmlu_cloze"), treat="hadamard",
        ctrl="unrotated", q=0.10, n_boot=N_BOOT):
    """Does de-sinking buy task capability (§6.7)?

    Paired item bootstrap of treat-minus-control accuracy, pooled over seeds at
    the matched-STEP horizon; the matched-WALL-CLOCK rows must agree in sign.
    BH-FDR at q=0.10 across the benchmark family.
    """
    def pick(gauge, kind, at_last_step=True):
        """Rows for one arm at one kind. The matched-STEP rows are taken at the
        common last step; the WALL-CLOCK rows are taken as they fall — each seed
        crosses the budget at its own step, so filtering them to a single step
        would silently drop seeds."""
        rs = [r for r in rows if r["gauge"] == gauge and r["kind"] == kind]
        last = max((r["step"] for r in rs), default=None)
        if not at_last_step:
            return rs, sorted({r["step"] for r in rs})
        return [r for r in rs if r["step"] == last], last

    t_step, step_t = pick(treat, "step")
    c_step, step_c = pick(ctrl, "step")
    t_wall, step_w = pick(treat, "wallclock", at_last_step=False)
    if not t_wall:
        t_wall, step_w = t_step, step_t       # wall-clock never bound (same speed)
    by_seed = lambda rs: {r["seed"]: r for r in rs}
    out, pvals = {}, []
    for bm in benchmarks:
        # arms are paired BY POSITION within a seed, and seeds are matched
        # explicitly — never by concatenation order
        seeds = sorted(set(by_seed(t_step)) & set(by_seed(c_step)))
        assert seeds, "no seed appears in both arms at the matched-step horizon"
        a = np.concatenate([np.asarray(by_seed(t_step)[s][bm]["per_item"], np.float64)
                            for s in seeds])
        b = np.concatenate([np.asarray(by_seed(c_step)[s][bm]["per_item"], np.float64)
                            for s in seeds])
        assert a.shape == b.shape, (bm, a.shape, b.shape)
        # zlib.crc32, NOT hash(): Python salts hash() per process, so a rerun
        # would draw a different bootstrap and print a different CI for the same
        # rows — a number that cannot be reproduced is not a result
        ci = boot_paired_ci(a, b, n_boot, seed=zlib.crc32(bm.encode()) % 10_000)
        wseeds = sorted(set(by_seed(t_wall)) & set(by_seed(c_step)))
        aw = np.concatenate([np.asarray(by_seed(t_wall)[s][bm]["per_item"], np.float64)
                             for s in wseeds])
        bw = np.concatenate([np.asarray(by_seed(c_step)[s][bm]["per_item"], np.float64)
                             for s in wseeds])
        wall = float(aw.mean() - bw.mean())
        # bootstrap two-sided p as the mass on the other side of zero
        rng = np.random.default_rng(4242)
        idx = rng.integers(0, a.size, (n_boot, a.size))
        d = (a[idx] - b[idx]).mean(axis=1)
        p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        pvals.append(p)
        out[bm] = {**ci, "p": p, "wallclock_diff": wall,
                   "same_sign_wallclock": bool(np.sign(wall) == np.sign(ci["diff"])
                                               and ci["diff"] != 0)}
    # headroom diagnostic (declared from the smoke cell BEFORE these numbers
    # existed; it does not enter the verdict rule): a benchmark whose untrained
    # accuracy is indistinguishable from chance cannot evidence "no effect"
    chance = 0.25
    head = {}
    for bm in benchmarks:
        base = [r for r in rows if r["kind"] == "base"]
        if not base:
            continue
        acc = float(np.mean([r[bm]["acc"] for r in base]))
        n = int(np.mean([r[bm]["n"] for r in base]))
        se = float(np.sqrt(chance * (1 - chance) / n))
        head[bm] = {"base_acc": acc, "n": n, "chance": chance,
                    "sigma_above_chance": (acc - chance) / se,
                    "informative": bool((acc - chance) / se >= 2.0)}
    rej = bh_reject(pvals, q)
    lever = [bm for i, bm in enumerate(benchmarks)
             if out[bm]["diff"] > 0 and out[bm]["lo"] > 0 and rej[i]
             and out[bm]["same_sign_wallclock"]]
    any_move = [bm for i, bm in enumerate(benchmarks)
                if (out[bm]["lo"] > 0 or out[bm]["hi"] < 0) and rej[i]]
    if lever:
        verdict = "LEVER"
    elif not any_move:
        verdict = "TRANSIENT-ONLY"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "benchmarks": out, "bh_reject": rej, "q": q,
            "lever_benchmarks": lever,
            "matched_step": {treat: step_t, ctrl: step_c},
            "wallclock_step": {treat: step_w}, "seeds_used": seeds,
            "headroom": head}


# ------------------------------------------------------------------------ CLI
def _load(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path) if l.strip()]
    if path.endswith(".npz"):
        return dict(np.load(path))
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["am1", "am2", "am3", "am4", "am5", "am6", "am7"])
    ap.add_argument("--dir", required=True, help="directory of downloaded artifacts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = a.dir
    j = lambda n: _load(os.path.join(D, n))
    if a.arm == "am1":
        res = am1(j("v1abl_rows.json"), j("v1abl_window_nll.json"),
                  j("banked_v1abl_rows.json"))
    elif a.arm == "am2":
        res = am2(j("am2_cells.json"))
    elif a.arm == "am3":
        res = am3(j("softread_rows.jsonl"))
    elif a.arm == "am4":
        meta = j("am4_meta.json")
        res = am4([j(n) for n in meta["batches"]], meta["n_layers"], meta["window"])
    elif a.arm == "am5":
        res = am5(j("v2b_rows.jsonl"), step=j("am5_meta.json")["step"])
    elif a.arm == "am6":
        res = am6(j("v3_rows_new.jsonl"), j("v3_rows_banked.jsonl"))
    else:
        res = am7(j("am7_rows.jsonl"))
    txt = json.dumps(res, indent=2, default=float)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
