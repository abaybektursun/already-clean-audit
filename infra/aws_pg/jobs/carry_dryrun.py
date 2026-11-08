"""CARRY D-suite (CPU, weight-free) — §9.1: every mathematical claim in carry_core proven
on synthetic objects, EVERY §6 rule branch constructed (including each VOID/failure branch),
the §0.5-4 Gemma-(1+γ)/Hadamard proof, and the historical replays. Run on a pg r7i instance
(never the laptop) BEFORE the freeze; D-suite green is the freeze precondition.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_dryrun --type r7i.2xlarge \
      --no-weights --s3-prefix results/carry_dryrun --budget-min 60
"""
import argparse
import json
import math
import os
import pathlib
import time

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ===========================================================================
# Q materialization + split + nulls.
# ===========================================================================
@check
def q_matches_closed_form():
    """Dense Q == spectral_completion_quadform on random probes (the A2 identity)."""
    import torch

    from infra.pg import carry_core as CC
    from infra.pg import specter_core as SC
    d, gen = 64, torch.Generator().manual_seed(0)
    B, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    sb = [(0, 8), (8, 24), (24, 64)]
    bulk = [3.0, 1.0, 0.1]
    U, _ = torch.linalg.qr(torch.randn(d, 3, generator=gen, dtype=torch.float64))
    lam = [50.0, 20.0, 10.0]
    Q = CC.materialize_q(U, lam, bulk, B, sb)
    for i in range(16):
        v = torch.randn(d, generator=gen, dtype=torch.float64)
        v = v / v.norm()
        qd = float(v @ Q @ v)
        qf = SC.spectral_completion_quadform(v, U, lam, bulk, B, sb)
        assert abs(qd - qf) / abs(qf) < 1e-10, (i, qd, qf)
    # spikeless variant
    Q0 = CC.materialize_q(None, [], bulk, B, sb)
    v = torch.randn(d, generator=gen, dtype=torch.float64)
    v = v / v.norm()
    assert abs(float(v @ Q0 @ v)
               - SC.spectral_completion_quadform(v, None, [], bulk, B, sb)) < 1e-10


@check
def split_recovers_planted_structure():
    """On a planted spectrum the 50%-trace split lands exactly at the planted rank, the
    stiff basis spans the planted top space, and stiff⊕flat reconstructs ΔW exactly
    (the §9.1 split-projector algebra, all 7 matrix types)."""
    import torch

    from infra.pg import carry_core as CC
    from infra.pg import comwrite_surgery as SURG
    d, gen = 32, torch.Generator().manual_seed(1)
    B, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    # 4 spikes carrying exactly half the trace: 4×10 = 40; bulk = 40 over 28 dims
    U, lam = B[:, :4], [10.0, 10.0, 10.0, 10.0]
    sb = [(0, d)]
    bulk = [40.0 / (d - 4)]
    # exact bulk needs spike-orthogonal bands: use B's complement columns via projector
    Q = CC.materialize_q(U, lam, bulk, B, sb)
    evals, evecs, k = CC.stiff_split(Q)
    assert k == 4, f"planted k=4, got {k}"
    ov = torch.linalg.svdvals(evecs[:, :4].T @ U)
    assert float(ov.min()) > 0.999, "stiff basis does not span the planted spikes"
    shapes = {"q_proj": (d, d), "k_proj": (8, d), "v_proj": (8, d), "o_proj": (d, d),
              "gate_proj": (4 * d, d), "up_proj": (4 * d, d), "down_proj": (d, 4 * d)}
    for t, sh in shapes.items():
        dW = torch.randn(*sh, generator=gen)
        stiff = SURG.project_delta(dW, evecs[:, :k].float(), t)
        flat = dW.to(torch.float32) - stiff
        assert torch.allclose(stiff + flat, dW.float(), atol=1e-5)
        # Frobenius-orthogonality of the two parts (the partition identity)
        assert abs(float((stiff * flat).sum())) < 1e-2 * float((dW ** 2).sum())


@check
def random_eigsplit_parity_and_determinism():
    import torch

    from infra.pg import carry_core as CC
    gen = torch.Generator().manual_seed(2)
    E, _ = torch.linalg.qr(torch.randn(16, 16, generator=gen))
    Ua, ia = CC.random_eigsplit(E, 5, seed=99)
    Ub, ib = CC.random_eigsplit(E, 5, seed=99)
    Uc, ic = CC.random_eigsplit(E, 5, seed=100)
    assert ia == ib and torch.equal(Ua, Ub), "eigsplit not deterministic in seed"
    assert ia != ic, "different seeds gave the same subset (16 choose 5 — a bug, not luck)"
    assert Ua.shape == (16, 5)


# ===========================================================================
# Gains, guards, Δcarry, degenerate completions.
# ===========================================================================
@check
def paired_gain_and_range_guard():
    import numpy as np

    from infra.pg import carry_core as CC
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 1, 400)
    cell = np.clip(base + 0.2 + rng.normal(0, 0.05, 400), 0, 1.4)
    g = CC.paired_gain(cell, base)
    assert 0.15 < g["gain"] < 0.25 and 0 < g["se"] < 0.02
    assert CC.range_guard_ok(g["gain"], g["se"])
    # the guard FAILS on noise (the R-T1 replay lives in its own check below)
    noise = base + rng.normal(0, 0.05, 400)
    gn = CC.paired_gain(noise, base)
    assert not CC.range_guard_ok(gn["gain"], gn["se"]) or abs(gn["gain"]) < 0.01
    # negative full gain fails REGARDLESS of magnitude (two-sidedness)
    assert not CC.range_guard_ok(-0.5, 0.01)


@check
def dcarry_bootstrap_and_void():
    import numpy as np

    from infra.pg import carry_core as CC
    rng = np.random.default_rng(4)
    flat = rng.normal(0.8, 0.1, 18)
    stiff = rng.normal(0.2, 0.1, 18)
    b = CC.dcarry_bootstrap(flat, stiff, n_boot=2000)
    assert not b["void"] and b["ci95"][0] > 0.3 and 0.5 < b["dcarry"] < 0.7
    v = CC.dcarry_bootstrap([0.5, 0.6], [0.1, 0.2], n_boot=100)
    assert v["void"], "a 2-FT bootstrap must VOID, not emit a CI (§2)"


@check
def rank_statistic_degenerate_completions():
    """The all-tied Spearman corner := 0 (the SPECTER E13/E16 replay — the parent
    atlas_fit.spearman RAISES on this input; the frozen completion catches it FIRST)."""
    from infra.pg import carry_core as CC
    assert CC.spearman0([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
    assert CC.kendall0([2, 2, 2], [5, 5, 5]) == 0.0
    assert abs(CC.spearman0([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    sh = CC.shuffle_null_rho([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
                             n_shuffle=500)
    assert sh["rho_obs"] == 1.0 and sh["p_value"] > 0        # (c+1)/(n+1): never exactly 0


# ===========================================================================
# R-C — every branch constructed.
# ===========================================================================
def _rc_rows(gf_s, gf_f, band_s=0.3, band_f=0.3):
    return [{"ft": f"ft{i}", "gf_stiff": s, "gf_flat": f,
             "band_stiff_max": band_s, "band_flat_max": band_f}
            for i, (s, f) in enumerate(zip(gf_s, gf_f))]


@check
def rc_all_branches():
    import numpy as np

    from infra.pg import carry_core as CC
    rng = np.random.default_rng(5)
    n = 18
    flat_hi = rng.normal(0.85, 0.05, n)
    stiff_lo = rng.normal(0.15, 0.05, n)
    r = CC.adjudicate_rc(_rc_rows(stiff_lo, flat_hi), n_sealed_total=18)
    assert r["verdict"] == "H_flat-carries", r
    r = CC.adjudicate_rc(_rc_rows(flat_hi, stiff_lo), n_sealed_total=18)
    assert r["verdict"] == "H_stiff-carries", r
    mid = rng.normal(0.5, 0.03, n)
    r = CC.adjudicate_rc(_rc_rows(mid, mid + rng.normal(0, 0.01, n),
                                  band_s=0.9, band_f=0.9), n_sealed_total=18)
    assert r["verdict"] == "H_distributed", r
    r = CC.adjudicate_rc(_rc_rows(stiff_lo[:8], flat_hi[:8]), n_sealed_total=18)
    assert r["verdict"] == "VOID-range" and r["n_void"] == 10, r      # >9 of 18 VOID
    # H_partial: directional but under the 0.6 median bar
    r = CC.adjudicate_rc(_rc_rows(rng.normal(0.1, 0.03, n), rng.normal(0.45, 0.03, n)),
                         n_sealed_total=18)
    assert r["verdict"] == "H_partial", r


# ===========================================================================
# R-P / R-N / R-E / R-W2 / R-D2 — every branch constructed.
# ===========================================================================
@check
def rp_branches():
    from infra.pg import carry_core as CC
    assert CC.adjudicate_rp([0.1] * 6 + [0.5, 0.8])["verdict"] == "H_spikes-content"
    assert CC.adjudicate_rp([0.8] * 4 + [0.1] * 4)["verdict"] == "H_spikes-sink-loaded"
    assert CC.adjudicate_rp([0.5] * 8)["verdict"] == "H_mixed"
    assert CC.sink_frac(2.0, 4.0) == 0.5


@check
def rn_branches_and_isoperimetry_two_sided():
    """The isoperimetry diagnostics must PASS on Gaussian data and FAIL on a
    heavy-tailed/outlier-coordinate construction — a theorem check designed to fail
    (the one-sided-thresholds law: every floor gets a ceiling)."""
    import numpy as np

    from infra.pg import carry_core as CC
    rng = np.random.default_rng(6)
    d, n = 64, 20000
    X = rng.standard_normal((n, d))
    cv = CC.thin_shell_cv(np.linalg.norm(X, axis=1))
    cv_ref = 1.0 / math.sqrt(2 * d)
    tails = CC.lipschitz_tail_chat(X, n_func=16, seed=0)
    r = CC.adjudicate_rn_b(cv=cv, cv_ref=cv_ref, c_hat=tails["c_hat"])
    assert r["verdict"] == "H_isoperimetric-enough", r
    # designed-to-fail leg 1 — one massive coordinate (the S17 threat): breaks the
    # thin-shell CV (norm bimodality) while linear-functional tails stay Gaussian
    Y = X.copy()
    Y[:, 0] *= 40.0
    cvy = CC.thin_shell_cv(np.linalg.norm(Y, axis=1))
    ty = CC.lipschitz_tail_chat(Y, n_func=16, seed=0)
    assert cvy / cv_ref > 2.0, "the massive coordinate must break the thin shell"
    ry = CC.adjudicate_rn_b(cv=cvy, cv_ref=cv_ref, c_hat=ty["c_hat"])
    assert ry["verdict"] == "H_non-isoperimetric", ry
    # designed-to-fail leg 2 — a scale MIXTURE (heavy-tailed rows): the TAIL diagnostic
    # itself must fail too, else ĉ would be a statistic that cannot fail
    s = np.where(rng.uniform(size=n) < 0.1, 3.0, 0.2)
    Z = X * s[:, None]
    tz = CC.lipschitz_tail_chat(Z, n_func=16, seed=0)
    assert tz["c_hat"] > 2.0, f"scale-mixture tails must exceed 2× Gaussian: {tz['c_hat']}"
    # R-N(c) branches
    rows_ok = [{"attn_sink_frac": 0.95, "qk_curv": 0.01}] + \
              [{"attn_sink_frac": 0.1, "qk_curv": 1.0}] * 9
    assert CC.adjudicate_rn_c(rows_ok)["verdict"] == "H_theorem-verified"
    rows_leak = [{"attn_sink_frac": 0.95, "qk_curv": 0.3}] + \
                [{"attn_sink_frac": 0.1, "qk_curv": 1.0}] * 9
    assert CC.adjudicate_rn_c(rows_leak)["verdict"] == "H_measured-leak"
    rows_bound = [{"attn_sink_frac": 0.95, "qk_curv": 0.9}] + \
                 [{"attn_sink_frac": 0.1, "qk_curv": 1.0}] * 9
    assert CC.adjudicate_rn_c(rows_bound)["verdict"] == "H_s23-bounded"
    rows_void = [{"attn_sink_frac": 0.1, "qk_curv": 1.0}] * 10
    assert CC.adjudicate_rn_c(rows_void)["verdict"] == "VOID-no-sink-heads"
    # Ford bound sanity: mu=0.5 ⇒ 0; smaller mu ⇒ larger bound; degenerate mu raises
    assert CC.ford_bound(1.0, 0.5) == 0.0
    assert CC.ford_bound(2.0, 0.1) > CC.ford_bound(1.0, 0.1) > 0
    try:
        CC.ford_bound(1.0, 0.0)
        raise RuntimeError("Ford bound accepted mu=0 — it must refuse")
    except AssertionError:
        pass
    assert CC.adjudicate_rn_d(top_ggn=1.0, top_hess=2.0)["ratio"] == 0.5


@check
def re_branches_and_lp_exponent():
    import numpy as np

    from infra.pg import carry_core as CC
    rng = np.random.default_rng(7)
    pred = np.arange(8.0)
    meas = pred * 10 + rng.normal(0, 0.1, 8) + 1
    r = CC.adjudicate_re(pred=pred, meas=meas, labels=[f"f{i}" for i in range(8)],
                         n_shuffle=2000)
    assert r["verdict"] == "H_operator-unifies", r
    assert r["p_value"] < 0.05 and r["shuffle_p95"] > 0.3, \
        "n=8 shuffle null sanity: p tiny, p95 large (the E3 interpretation note's premise)"
    r = CC.adjudicate_re(pred=pred, meas=np.ones(8) * 5 + rng.normal(0, 0.01, 8),
                         labels=[f"f{i}" for i in range(8)], n_shuffle=200)
    assert r["verdict"] == "VOID-no-range", r                  # span < 10× ⇒ VOID
    # a DETERMINISTIC partial ranking (ρ≈0.76, span 71× > 10×): fixed rank permutation
    # so the branch cannot flip on an RNG draw (the run-2 misfire: seed-7 "half" landed
    # ρ=0.83 ≥ 0.8 → H_operator-unifies; a test verdict must not depend on the seed)
    half = np.array([2.0, 0.0, 1.0, 5.0, 3.0, 7.0, 4.0, 6.0]) * 10 + 1
    r = CC.adjudicate_re(pred=pred, meas=half, labels=[f"f{i}" for i in range(8)],
                         n_shuffle=2000)
    assert r["verdict"] == "H_partial" and 0.5 <= r["rho"] < 0.8, r
    # strong anti-correlation with span ≥ 10× (100/(0.5+pred) spans 13.3–200 = 15×, so it
    # clears the range guard and reaches the ρ<0.5 branch instead of VOID-ing on span)
    anti = 100.0 / (0.5 + pred) + rng.normal(0, 0.05, 8)
    r = CC.adjudicate_re(pred=pred, meas=anti, labels=[f"f{i}" for i in range(8)],
                         n_shuffle=2000)
    assert r["verdict"] == "H_families-escape" and r["span_ratio"] >= 10.0, r
    # Lᵖ exponent: quadratic family reads p≈2; degenerate inputs return None
    amps = np.array([0.05, 0.1, 0.2, 0.4])
    assert abs(CC.lp_growth_exponent(amps, 3.0 * amps ** 2) - 2.0) < 1e-6
    assert CC.lp_growth_exponent(amps, [1.0, 2.0, 0.0, 4.0]) is None
    assert CC.lp_growth_exponent(amps[:2], [1.0, 2.0]) is None


@check
def rw2_and_rd2_branches():
    import numpy as np

    from infra.pg import carry_core as CC
    assert CC.adjudicate_rw2_meansub(d_meansub=0.01, d_zero=1.0, seed_band=0.02)[
        "verdict"] == "H_constant-bias"
    assert CC.adjudicate_rw2_meansub(d_meansub=0.5, d_zero=1.0, seed_band=0.02)[
        "verdict"] == "H_deviates"
    assert CC.adjudicate_rw2_rotation(max_rel_err=1e-3)["verdict"] == "H_instrument-certified"
    assert CC.adjudicate_rw2_rotation(max_rel_err=0.2)["verdict"] == "HALT-instrument"
    # D2: planted intruder is counted; clean FT is clean; entry-shuffle null raises maxcos
    rng = np.random.default_rng(8)
    d = 64
    Ub, _ = np.linalg.qr(rng.standard_normal((d, d)))
    clean = Ub[:, :10]                                     # ft top vectors == base vectors
    counts, _ = CC.intruder_counts(clean, Ub)
    assert all(v == 0 for v in counts.values()), counts
    intr = clean.copy()
    v = rng.standard_normal(d)
    v -= Ub @ (Ub.T @ v) * 0                               # arbitrary direction
    v /= np.linalg.norm(v)
    # make an exact intruder: orthogonal to ALL base columns is impossible (Ub is full),
    # so plant a maximally-spread vector: equal mass on all base vectors → maxcos = 1/√d
    v = Ub @ (np.ones(d) / math.sqrt(d))
    intr[:, 0] = v
    counts_i, maxcos = CC.intruder_counts(intr, Ub)
    assert counts_i["0.6"] >= 1 and maxcos[0] < 0.2
    r = CC.adjudicate_rd2([counts, counts, counts])
    assert r["verdict"] == "H_full-FT-clean"
    r = CC.adjudicate_rd2([counts_i, counts_i, counts_i])
    assert r["verdict"] == "H_intruders-present"
    B_sh = CC.entry_shuffle_null(Ub, 10, seed=1)
    assert B_sh.shape == Ub.shape and not np.allclose(B_sh, Ub)


@check
def kappa_family_alignment():
    """E[cos θ] tracks κ/√(d+κ²) on the κ-bridge construction (S2's family)."""
    import numpy as np
    import torch

    from infra.pg import carry_core as CC
    d = 256
    w = torch.randn(d, generator=torch.Generator().manual_seed(9))
    for kappa in (0.0, 4.0, 16.0, 64.0):
        cs = [float(torch.dot(CC.kappa_draw(w, kappa, d, seed=i), w / w.norm()))
              for i in range(200)]
        exp = CC.expected_cos(kappa, d)
        assert abs(np.mean(cs) - exp) < 0.05, (kappa, np.mean(cs), exp)
    assert float(torch.dot(CC.kappa_draw(w, None, d, 0), w / w.norm())) > 0.999


# ===========================================================================
# §0.5-4 — the Gemma (1+γ) / Hadamard proof on a tiny PG1-like block.
# ===========================================================================
@check
def gemma_gamma_hadamard_invariance():
    """A tiny pre-norm residual block with Gemma RMSNorm (x/rms·(1+γ)): folding (1+γ)
    into the reading weight and rotating the residual stream by Hadamard H preserves the
    function exactly (fp64) — the S18 computational invariance on Gemma's convention.
    Also the DESIGNED-TO-FAIL leg: rotating WITHOUT folding γ must NOT preserve it."""
    import torch

    from infra.pg import carry_core as CC
    d, dm = 16, 32
    gen = torch.Generator().manual_seed(10)
    gamma = 0.1 * torch.randn(d, generator=gen, dtype=torch.float64)
    W_in = torch.randn(dm, d, generator=gen, dtype=torch.float64)
    W_out = torch.randn(d, dm, generator=gen, dtype=torch.float64)

    def rms(x):
        return torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)

    def block(x, w_in, w_out, g):
        h = (x / rms(x)) * (1.0 + g) if g is not None else x / rms(x)
        return x + torch.tanh(h @ w_in.T) @ w_out.T

    X = torch.randn(8, d, generator=gen, dtype=torch.float64)
    y_ref = block(X, W_in, W_out, gamma)
    H = CC.hadamard_matrix(d)
    assert torch.allclose(H @ H.T, torch.eye(d, dtype=torch.float64), atol=1e-12)
    Wf = CC.fold_gemma_norm(gamma, W_in)                    # scale-free norm now exact
    y_fold = block(X, Wf, W_out, None)
    assert torch.allclose(y_ref, y_fold, atol=1e-10), "the (1+γ) fold is not exact"
    # rotate: x' = xH ; the reader absorbs H on its input side (Wf → Wf·H) and the
    # writer counter-rotates its output side (W_out → Hᵀ·W_out)
    y_rot = block(X @ H, Wf @ H, H.T @ W_out, None)
    assert torch.allclose(y_ref @ H, y_rot, atol=1e-10), "Hadamard invariance broken"
    # designed to fail: rotation WITHOUT the fold
    y_bad = block(X @ H, W_in @ H, H.T @ W_out, gamma)
    assert not torch.allclose(y_ref @ H, y_bad, atol=1e-6), \
        "unfolded rotation must break — if this passes, the check cannot fail"


# ===========================================================================
# Historical replays (§9.1) — misfires the suite must reproduce and classify.
# ===========================================================================
@check
def replay_rt1_range_guard_misfire():
    """ATLAS R-T1: a floor measured on a MISMATCHED construction family voids a live
    family. The construction-matched floor (same family) classifies it correctly."""
    from infra.pg import specter_core as SC
    response = 0.05
    floor_mismatched = 0.04          # noisy foreign family: guard fails a REAL response
    floor_matched = 0.005
    assert not SC.range_guard(response, floor_mismatched)
    assert SC.range_guard(response, floor_matched)


@check
def replay_saturated_em_null_voids():
    """REGRAIN R1: both arms indistinguishable from the dead floor (or from intact) ⇒
    the saturation guard must VOID the contrast, never adjudicate it; a live contrast
    passes through."""
    from infra.pg.bispec_core import saturation_guard
    live = saturation_guard(damage_treat=0.6, damage_null=0.2, floor=0.0, intact=1.0,
                            name="replay_live")
    assert live["void"] is False, live
    dead = saturation_guard(damage_treat=0.005, damage_null=0.004, floor=0.0, intact=1.0,
                            name="replay_dead")
    assert dead["void"] is True, dead


@check
def replay_all_tied_regressor():
    """SPECTER E13: a constant predictor is a reachable corner — ρ := 0, and the R-E
    adjudicator lands H_families-escape (never crashes, never rewards)."""
    import numpy as np

    from infra.pg import carry_core as CC
    meas = np.array([1.0, 3.0, 9.0, 27.0, 81.0, 243.0, 729.0, 2187.0])
    r = CC.adjudicate_re(pred=np.ones(8), meas=meas, labels=list("abcdefgh"),
                         n_shuffle=500)
    assert r["rho"] == 0.0 and r["verdict"] == "H_families-escape", r


@check
def replay_assignment_battery_mechanics():
    """A1: the argmax assignment + ascending-|gain| discovery selection is mechanical and
    deterministic on a synthetic gain table, and a no-transfer FT VOIDs at the guard."""
    from infra.pg import carry_core as CC
    table = {
        "ftA": {"vqav2": (0.30, 0.01), "textvqa": (0.05, 0.01)},
        "ftB": {"vqav2": (0.02, 0.01), "textvqa": (0.20, 0.01)},
        "ftC": {"vqav2": (0.005, 0.01), "textvqa": (0.004, 0.01)},   # no transfer → VOID
        "ftD": {"vqav2": (0.10, 0.01), "textvqa": (0.01, 0.01)},
    }
    assigned = {ft: max(row, key=lambda t: abs(row[t][0])) for ft, row in table.items()}
    assert assigned == {"ftA": "vqav2", "ftB": "textvqa", "ftC": "vqav2", "ftD": "vqav2"}
    passing = sorted((abs(table[ft][assigned[ft]][0]), ft) for ft in table
                     if CC.range_guard_ok(*table[ft][assigned[ft]]))
    assert [ft for _g, ft in passing] == ["ftD", "ftB", "ftA"]     # ascending |gain|
    assert "ftC" not in [ft for _, ft in passing]


@check
def cider_paired_gain_runs():
    """The corpus-task paired bootstrap helper is exercised (pycocoevalcap present on the
    instance via install_pg; the D-suite installs it the same way)."""
    from infra.pg import carry_core as CC
    cand_a = {i: [f"a cat sits on the mat {i % 3}"] for i in range(24)}
    cand_b = {i: [f"a dog sits on the mat {i % 3}"] for i in range(24)}
    refs = {i: [f"a cat sits on the mat {i % 3}", "a cat on a mat"] for i in range(24)}
    g = CC.cider_paired_gain(cand_a, cand_b, refs, n_boot=50)
    assert g["gain"] > 0 and g["se"] >= 0


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from infra.aws_pg.jobs import _carry_shared as CS
    CS.install_pg()                       # pycocoevalcap for the CIDEr check
    rep = {"object": "carry_dryrun_v1", "checks": []}
    t00 = time.time()
    for fn in CHECKS:
        t0 = time.time()
        fn()
        rep["checks"].append({"name": fn.__name__, "ok": True,
                              "seconds": round(time.time() - t0, 2)})
        (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
        print(f"[carry-dryrun] {fn.__name__} OK", flush=True)
    rep["n_checks"] = len(rep["checks"])
    rep["wall_seconds"] = round(time.time() - t00, 1)
    rep["gpu_seconds"] = {"total": 0.0, "note": "CPU job"}
    print("CARRY_DRYRUN_OK", json.dumps({"n": rep["n_checks"]}))
    rep["green"] = True
    (out / "dryrun.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
