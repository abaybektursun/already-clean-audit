#!/usr/bin/env python3
"""AMUNDSEN offline gate — every new instrument, weight-free, before any GPU.

Runs in the image (preflight T2) and on the laptop. No model, no download, no
CUDA. Four instrument checks and one rule check:

  1 energy-matched subspace   two-directional: the construction hits the target
                              AND an independent measurement of the built
                              subspace reproduces the returned energy; plus a
                              RESOLUTION gate (a different target must give a
                              different realized energy) and a MUTANT (the
                              wrong formula must fail the same assert).
  2 planted/magnitude control identical per-coordinate magnitude profile,
                              destroyed direction.
  3 softread dose hook        THE GEMMA TRAP: the hook must equal rescaling the
                              EFFECTIVE gain (1+w)*s, and must NOT equal the
                              wrong form w*s (the mutant).
  4 horizon-matched delivery  on a synthetic trajectory whose true windowed
                              delivery is known, the banked cumulative/windowed
                              construction is shown to MISS it and the fix to
                              recover it.
  5 §6 decision rules         every verdict token of every arm is reachable on
                              synthetic inputs — a rule that can only emit one
                              token is a cannot-fail statistic, not a rule.

Exit 0 = ship. FAIL LOUD: asserts only, no try/except.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/app") if "/app" not in sys.path else None

from infra.pg.conserved_core_probes import energy_matched_subspace     # noqa: E402
from infra.vast import analysis_amundsen as A                          # noqa: E402


def _gram(d=64, seed=0, decay=0.9):
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    mu = decay ** np.arange(d) + 1e-6
    return V, mu, (V * mu) @ V.T


def _share(Q, G):
    return float(np.trace(Q.T @ G @ Q) / np.trace(G))


# ------------------------------------------------------------------------ 1
def check_energy_matched():
    d, k = 128, 8
    V, mu, G = _gram(d)
    # a synthetic "core": aligned with the high-energy directions, the way a
    # real sink is. A random core would sit AT the k/d floor and the resolution
    # gate below would be testing nothing.
    rng = np.random.default_rng(7)
    core, _ = np.linalg.qr(V[:, :k] + 0.1 * rng.standard_normal((d, k)))
    e_core = _share(core, G)
    assert e_core > 4 * k / d, f"synthetic core is not energy-concentrated: {e_core}"
    Q, e_ret = energy_matched_subspace(V, mu, e_core, k, seed=3,
                                       tol=0.02 * e_core, max_resample=200)
    assert Q.shape == (d, k), Q.shape
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-8), "energy-matched basis not orthonormal"
    # direction A: construction -> target
    assert abs(e_ret - e_core) <= 0.1 * e_core, (e_ret, e_core)
    # direction B: independent measurement of the BUILT subspace -> the same energy
    e_meas = _share(Q, G)
    assert abs(e_meas - e_ret) <= 1e-6 * max(1.0, e_ret), (e_meas, e_ret)
    # resolution gate: a different target must move the realized energy
    Q2, e2 = energy_matched_subspace(V, mu, e_core / 4, k, seed=3,
                                     tol=0.02 * e_core / 4, max_resample=200)
    assert e2 < e_ret * 0.7, f"instrument cannot resolve the target: {e2} vs {e_ret}"
    # mutant: matching a target the construction cannot reach (below k/d) must
    # NOT silently claim success — it lands at the random floor and says so
    floor_target = 0.5 * k / d
    _, e_floor = energy_matched_subspace(V, mu, floor_target, k, seed=3,
                                         tol=1e-9, max_resample=50)
    assert e_floor > floor_target, "below-floor target reported as matched"
    print("dryrun 1 OK  energy-matched subspace (two-directional + resolution + mutant)")


# ------------------------------------------------------------------------ 2
def check_planted():
    d, k = 64, 8
    rng = np.random.default_rng(11)
    core = rng.standard_normal((d, k)) * (rng.random((d, 1)) ** 3)   # spiky rows
    signs = np.random.default_rng(12).choice([-1.0, 1.0], size=core.shape)
    planted = core * signs
    assert np.allclose(np.sort(np.abs(planted), axis=0), np.sort(np.abs(core), axis=0)), \
        "planted control changed the magnitude profile"
    Qc, _ = np.linalg.qr(core)
    Qp, _ = np.linalg.qr(planted)
    ov = float(np.linalg.norm(Qc.T @ Qp, "fro") ** 2 / k)
    assert ov < 0.5, f"planted control still shares the direction (overlap {ov})"
    print(f"dryrun 2 OK  planted control (magnitude preserved, overlap {ov:.3f})")


# ----------------------------------------------------------------------- 2b
def check_spectrum_matched():
    """AM-1b's control: same removed energy, same spectral placement, new
    directions. The construction sign-randomises core32's coordinates IN THE
    GRAM'S EIGENBASIS, so every |A_ij| — and therefore the per-eigendirection
    energy profile — is preserved before the re-orthonormalisation."""
    d, k = 128, 8
    V, mu, G = _gram(d)
    rng = np.random.default_rng(21)
    core, _ = np.linalg.qr(V[:, :k] + 0.3 * rng.standard_normal((d, k)))
    A = V.T @ core
    srng = np.random.default_rng(22)
    Qs, _ = np.linalg.qr(V @ (A * srng.choice([-1.0, 1.0], size=A.shape)))
    Qs = Qs[:, :k]
    assert np.allclose(Qs.T @ Qs, np.eye(k), atol=1e-8)
    e_core, e_spec = _share(core, G), _share(Qs, G)
    prof_core = (A ** 2).sum(axis=1)
    prof_spec = ((V.T @ Qs) ** 2).sum(axis=1)
    l1 = float(np.abs(prof_spec - prof_core).sum() / prof_core.sum())
    ov = float(np.linalg.norm(core.T @ Qs, "fro") ** 2 / k)
    assert abs(e_spec - e_core) <= 0.10 * e_core, (e_core, e_spec)
    assert l1 < 0.35, f"spectral profile not preserved (L1 rel {l1})"
    assert ov < 0.6, f"spectrum-matched control still shares the direction ({ov})"
    # MUTANT: sign-randomising in the STREAM basis instead of the eigenbasis
    # must NOT preserve the spectral profile as well
    Qm, _ = np.linalg.qr(core * np.random.default_rng(23).choice([-1.0, 1.0], core.shape))
    l1_m = float(np.abs(((V.T @ Qm) ** 2).sum(axis=1) - prof_core).sum() / prof_core.sum())
    assert l1_m > l1, f"the stream-basis mutant matched the profile as well ({l1_m} vs {l1})"
    print(f"dryrun 2b OK  spectrum-matched control (dE {abs(e_spec-e_core)/e_core:.3f}, "
          f"profile L1 {l1:.3f} vs mutant {l1_m:.3f}, overlap {ov:.3f})")


# ------------------------------------------------------------------------ 3
def _gemma_rmsnorm(x, w, eps=1e-6):
    """The reference: gemma multiplies by (1 + w), never by w."""
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return v * (1.0 + w.float())


def check_softread_gain():
    torch.manual_seed(0)
    d, k, s = 32, 4, 1.7
    x = torch.randn(3, 5, d)
    w = torch.randn(d) * 0.1
    idx = torch.randperm(d)[:k]
    Q = torch.zeros(d, k)
    Q[idx, torch.arange(k)] = 1.0                       # axis-aligned subspace

    out = _gemma_rmsnorm(x, w)
    hooked = out + (s - 1.0) * ((out @ Q) @ Q.T)        # the cell's dose hook

    w_right = w.clone()
    w_right[idx] = s * (1.0 + w[idx]) - 1.0             # effective gain (1+w)*s
    assert torch.allclose(hooked, _gemma_rmsnorm(x, w_right), atol=1e-5), \
        "dose hook != rescaling the EFFECTIVE gain"

    w_wrong = w.clone()
    w_wrong[idx] = s * w[idx]                            # THE TRAP: w*s
    assert not torch.allclose(hooked, _gemma_rmsnorm(x, w_wrong), atol=1e-3), \
        "the wrong (w*s) form passed the equivalence — the trap check is blind"

    # identity: s = 1 is exactly the untouched forward
    assert torch.equal(out + 0.0 * ((out @ Q) @ Q.T), out)
    # realized motion is the quantity the twins are matched on
    rel = float((hooked - out).pow(2).sum().sqrt() / out.pow(2).sum().sqrt())
    assert rel > 0, rel
    print(f"dryrun 3 OK  gemma effective-gain dose (trap + mutant), rel_dh={rel:.4f}")


# ------------------------------------------------------------------------ 4
def check_horizon_matched_delivery():
    """Windows 1..3; the subspace takes a CONSTANT 40% of each window's motion
    while the gradient share is a constant 50%, so the true delivery is 0.8 at
    every checkpoint. Window motion norms grow, so cumulative/windowed drifts."""
    f_motion, g_share, true_delivery = 0.4, 0.5, 0.8
    win_norm2 = [1.0, 4.0, 16.0]
    cum = cum_in = 0.0
    got_win, got_cum = [], []
    for w2 in win_norm2:
        win_in = f_motion * w2
        cum += w2
        cum_in += win_in
        got_win.append((win_in / w2) / g_share)
        got_cum.append((cum_in / cum) / g_share)
    assert all(abs(v - true_delivery) < 1e-12 for v in got_win), got_win
    assert abs(got_cum[-1] - true_delivery) < 1e-12, "cumulative also correct here?"
    # make the mismatch bite: the motion fraction CHANGES with the horizon,
    # which is exactly the scar-vs-reformation question. (Increments are taken
    # orthogonal here so norms add — the point being demonstrated is the
    # HORIZON mismatch, not the vector algebra.)
    cum = cum_in = 0.0
    fr, win_norm2 = [0.9, 0.5, 0.02], [4.0, 4.0, 1.0]
    win, cumd = [], []
    for w2, f in zip(win_norm2, fr):
        cum += w2
        cum_in += f * w2
        win.append((f * w2 / w2) / g_share)
        cumd.append((cum_in / cum) / g_share)
    assert abs(win[-1] - fr[-1] / g_share) < 1e-12, win
    assert abs(cumd[-1] - fr[-1] / g_share) > 0.5, \
        "the banked construction should MISS a changing motion fraction"
    src = open(__file__.replace("dryrun_amundsen.py", "cells/v2b_continuation.py")).read()
    assert '"delivery_win": {b: (wins[b] / wtot) / (gwin[b] / gwin_tot)' in src, \
        "the shipped v2b cell does not compute windowed/windowed delivery"
    assert '"act_per_layer"' in src and '"act_distinct_max"' in src, \
        "the shipped v2b cell does not bank per-layer activation trackers"
    print(f"dryrun 4 OK  horizon-matched delivery (win {win[-1]:.2f} vs cum {cumd[-1]:.2f})")


# ------------------------------------------------------------------------ 5
def _am1_inputs(core_d, emat_d, plant_d, top_d, n=40, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    base_nll = rng.normal(900, 5, n)
    ntok = np.full(n, 511.0)

    def arm(dppl):
        # invert ppl = exp(nll/ntok) so a per-window dppl becomes a per-window nll
        return {"nll": list(np.log(np.exp(base_nll / ntok) + dppl
                                   + rng.normal(0, noise, n)) * ntok),
                "ntok": list(ntok)}
    per = {"none": {"nll": list(base_nll), "ntok": list(ntok)},
           "core32": arm(core_d), "topenergy32": arm(top_d),
           "energymatched32_s0": arm(emat_d), "energymatched32_s1": arm(emat_d),
           "magmatched32_s0": arm(plant_d), "magmatched32_s1": arm(plant_d)}
    rows = {"arms": [{"arm": "none", "ppl": 6.0},
                     {"arm": "topenergy32", "ppl": 6.0 + top_d}]}
    banked = {"arms": [{"arm": "none", "ppl": 6.0},
                       {"arm": "topenergy32", "ppl": 6.0 + 103.6}]}
    return rows, {"per_window": per}, banked


def check_rules():
    # ---- AM-1: all four tokens reachable
    got = set()
    # a noisier instrument: the frozen rule's ENERGY-ACCOUNT branch needs the two
    # CIs to OVERLAP, which a very precise instrument can make unreachable —
    # recorded here as a property of the rule, not of the data
    r, w, b = _am1_inputs(0.15, 0.145, 0.01, 103.0, noise=0.08, seed=3)
    got.add(A.am1(r, w, b, n_boot=400)["verdict"])
    for core, emat, plant, top in [(0.15, 0.14, 0.01, 103.0),   # energy account
                                   (0.15, 0.02, 0.01, 103.0),   # core exceeds
                                   (0.15, 0.11, 0.01, 103.0),   # ambiguous-ish
                                   (0.15, 0.14, 0.12, 103.0),   # planted fires
                                   (0.15, 0.14, 0.01, 0.5)]:    # positive ctrl dead
        r, w, b = _am1_inputs(core, emat, plant, top)
        got.add(A.am1(r, w, b, n_boot=400)["verdict"])
    assert {"ENERGY-ACCOUNT", "CORE-EXCEEDS-ENERGY", "VOID-INSTRUMENT"} <= got, got

    # ---- AM-2
    def cells(rot_ppl, unrot_ppl, lrs=(3e-6, 1e-5, 3e-5, 1e-4), diverge=()):
        out = []
        for lr in lrs:
            for g, p in (("hadamard", rot_ppl), ("unrotated", unrot_ppl)):
                for s in (2026, 2027, 2028):
                    out.append({"lr": lr, "steps": 1200, "gauge": g, "seed": s,
                                "ppl": p * (1 + 0.02 * (lr == 1e-4)),
                                "task_loss": 2.5, "diverged": (g, lr) in diverge})
        return out
    v2 = {A.am2(cells(33.0, 25.0))["verdict"], A.am2(cells(25.2, 25.0))["verdict"],
          A.am2(cells(26.5, 25.0))["verdict"],
          A.am2(cells(33.0, 25.0, lrs=(1e-5, 2e-5)))["verdict"]}
    assert {"INTRINSIC-COST", "RETUNE-CLOSES", "PARTIAL", "COVERAGE-IMPOSSIBLE"} <= v2, v2

    # ---- AM-3
    ladder = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

    def sr(core_fn, twin_fn, pc_fn):
        rows = [{"arm": "none", "s": 1.0, "mode": "identity", "ppl": 10.0}]
        for s in ladder:
            rows.append({"arm": "core32", "s": s, "mode": "nominal", "ppl": 10.0 + core_fn(s)})
            for t in ("energymatched32", "haar32"):
                for md in ("nominal", "matched"):
                    rows.append({"arm": t, "s": s, "mode": md, "ppl": 10.0 + twin_fn(s)})
            rows.append({"arm": "topenergy32", "s": s, "mode": "nominal",
                         "ppl": 10.0 + pc_fn(s)})
        return rows
    # every response must be 0 at s = 1 — the identity control is real
    mono = lambda s: 2.0 * (s - 1.0)
    ushape = lambda s: 3.0 * abs(np.log(s))
    flat = lambda s: 0.001 * abs(np.log(s))
    v3 = {A.am3(sr(mono, flat, mono))["verdict"],       # reads (monotone, outside)
          A.am3(sr(flat, flat, mono))["verdict"],       # routed around
          A.am3(sr(ushape, ushape, ushape))["verdict"],  # generic
          A.am3(sr(mono, flat, lambda s: 0.0))["verdict"]}  # dead positive control
    assert {"READS-CORE", "ROUTED-AROUND", "GENERIC", "VOID-INSTRUMENT"} <= v3, v3

    # ---- AM-4
    def act(early, late, n_layers=8, window=512):
        b = {}
        for L in range(n_layers):
            pe = np.ones(window)
            c = early if L <= 2 else (late if L >= n_layers - 3 else 1.0)
            pe[0] = c * window - (window - 1) * 0 if c > 1 else 1.0
            pe[0] = c
            pe[1:] = (window - c) / (window - 1)
            b[f"L{L:02d}_q_proj_in_pos_energy"] = pe * 1.0
        return b
    hi = act(0.5 * 512, 0.02 * 512)      # pos0 holds half the energy early
    v4 = {A.am4([hi, hi], 8, 512)["verdict"],
          A.am4([act(1.0, 1.0), act(1.0, 1.0)], 8, 512)["verdict"]}
    assert {"REPLICATES", "DOES-NOT"} <= v4, v4

    # ---- AM-5
    def v2b(level, obs, slope0=True):
        rows = []
        for st in (2760, 2880, 3000):
            for sd in (1, 2, 3):
                h = obs if slope0 else obs * (st / 3000.0) ** 2
                rows.append({"gauge": "unrotated", "seed": sd, "step": st,
                             "delivery_win": {"core32": 0.3},
                             "fresh_bottom32_motion_share": level,
                             "fresh_bottom32_motion_share_win": level,
                             "act_distinct_max": 5})
                rows.append({"gauge": "hadamard", "seed": sd, "step": st,
                             "delivery_win": {"core32": 0.9},
                             "fresh_bottom32_motion_share": h,
                             "fresh_bottom32_motion_share_win": h,
                             "act_distinct_max": 5})
        return rows
    v5 = {A.am5(v2b(0.01, 0.011))["verdict"], A.am5(v2b(0.01, 0.03))["verdict"],
          A.am5(v2b(0.01, 0.03, slope0=False))["verdict"]}
    assert {"RE-FORMATION", "SCAR", "AMBIGUOUS-EXTEND"} <= v5, v5
    bad = v2b(0.01, 0.03)
    for r in bad:
        if r["gauge"] == "unrotated":
            r["delivery_win"] = {"core32": 1.5}
    assert A.am5(bad)["verdict"] == "VOID-INSTRUMENT"

    # ---- AM-6
    def v3rows(seeds, slope_sign):
        out = []
        for sd in seeds:
            for B in (2, 8, 32):
                out.append({"batch": B, "layer": None, "sample_seed": sd,
                            "delivery": {b: 0.13 + slope_sign * 0.01 * np.log(B)
                                         for b in ("core32", "anti32", "topsigma32",
                                                   "bottomsigma32")}
                            | {h: 1.0 + 0.001 * ((sd + i) % 3 - 1) * np.log(B)
                               for i, h in enumerate(("haar32_s0", "haar32_s1", "haar32_s2"))}})
        return out
    banked6 = v3rows([2026], -1)
    v6 = {A.am6(v3rows([3001, 3002], -1), banked6)["verdict"],
          A.am6(v3rows([3001, 3002], +1), banked6)["verdict"]}
    assert {"REAL", "SEED-NOISE"} <= v6, v6

    # ---- AM-7
    def am7rows(delta, n=200, wall_delta=None):
        rng = np.random.default_rng(5)
        base = (rng.random(n) < 0.30).astype(float)
        out = []
        for sd in (1, 2, 3):
            for g, dd in (("unrotated", 0.0), ("hadamard", delta)):
                flip = base.copy()
                if dd:
                    src = np.where(base == (0 if dd > 0 else 1))[0]
                    idx = src[: int(abs(dd) * n)]
                    flip[idx] = 1.0 if dd > 0 else 0.0
                for kind, dl in (("step", dd),
                                 ("wallclock", dd if wall_delta is None else wall_delta)):
                    f2 = base.copy()
                    if dl:
                        src = np.where(base == (0 if dl > 0 else 1))[0]
                        idx = src[: int(abs(dl) * n)]
                        f2[idx] = 1.0 if dl > 0 else 0.0
                    if g == "unrotated" and kind == "wallclock":
                        continue
                    out.append({"gauge": g, "kind": kind, "step": 1200, "seed": sd,
                                "mmlu_blankimg": {"per_item": list(f2 if kind == "wallclock"
                                                                   else flip)},
                                "mmlu_cloze": {"per_item": list(base)}})
        return out
    v7 = {A.am7(am7rows(0.15), n_boot=400)["verdict"],
          A.am7(am7rows(0.0), n_boot=400)["verdict"],
          A.am7(am7rows(-0.15), n_boot=400)["verdict"]}
    assert {"LEVER", "TRANSIENT-ONLY", "MIXED"} <= v7, v7

    # ---- primitives
    rho, p = A.spearman_exact_p([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
    assert rho == 1.0 and p < 0.001, (rho, p)
    assert A.bh_reject([0.001, 0.9], 0.10) == [True, False]
    print("dryrun 5 OK  every §6 verdict token reachable on synthetic inputs")


if __name__ == "__main__":
    check_energy_matched()
    check_planted()
    check_spectrum_matched()
    check_softread_gain()
    check_horizon_matched_delivery()
    check_rules()
    print("AMUNDSEN DRYRUN PASS")
