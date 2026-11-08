"""Weight-free CPU dry-run gate for conserved-core Tier 1 — the prereg §9.1 / §13 gate-1 logic test.
NO model, NO datasets, NO GPU — pure logic on tiny planted tensors, seconds on a laptop.

    python3 -m pytest infra/aws_pg/tests/test_conserved_core.py -q

Covers every §9.1 checklist item + the post-review energy-matched-null machinery:
  * projector idempotency (sided variant) removes the band on the chosen side
  * position probe R²>0.95 (shuffle→0); induction F1>0.9 (shuffle→low); label correctness
  * paired block-bootstrap brackets a planted contrast / straddles 0
  * energy-matched subspace hits the target energy, orthonormal, distinct per seed
  * synthetic contrast bootstrap + BH-FDR
  * synthetic corpora gold == the induction / position target
  * the FROZEN four-conjunct naming rule maps each scenario to the right verdict
"""
import math

import numpy as np
import pytest
import torch

from infra.pg import conserved_core_probes as P
from infra.aws_pg.jobs import conserved_core as J


# ---------------------------------------------------------------------------
# Split, probes, labels (pure core) — unchanged from v1.
# ---------------------------------------------------------------------------
def test_is_confirmation_deterministic_and_balanced():
    assert P.is_confirmation("hello world") == P.is_confirmation("hello world")
    frac = sum(P.is_confirmation(f"window number {i} text") for i in range(400)) / 400
    assert 0.4 < frac < 0.6, frac


def test_position_probe_recovers_planted_signal_and_shuffle_collapses():
    rng = np.random.default_rng(0)
    N, d = 600, 16
    y = rng.uniform(0, 1, N)
    X = rng.standard_normal((N, d)) * 0.1
    X[:, 0] = y + 0.02 * rng.standard_normal(N)
    tr, te = slice(0, 400), slice(400, N)
    r2 = P.r2_score(y[te], P.linear_predict(P.ridge_fit(X[tr], y[tr], 1.0), X[te]))
    assert r2 > 0.95, r2
    r2s = P.r2_score(y[te], P.linear_predict(P.ridge_fit(X[tr], P.shuffle_labels(y[tr], 1), 1.0), X[te]))
    assert r2s < 0.1, r2s


def test_corr_score_bounded_shift_invariant():
    rng = np.random.default_rng(20)
    y = rng.uniform(0, 1, 100)
    assert abs(P.corr_score(y, y) - 1.0) < 1e-9
    assert abs(P.corr_score(y, -y) + 1.0) < 1e-9
    assert abs(P.corr_score(y, 5 * y + 3) - 1.0) < 1e-9       # scale/shift invariant (the smoke fix)
    assert P.corr_score(y, np.zeros(100)) == 0.0              # constant pred => no signal
    assert -1.0 <= P.corr_score(y, rng.uniform(0, 1, 100)) <= 1.0


def test_induction_probe_separates_planted_and_shuffle_collapses():
    rng = np.random.default_rng(1)
    N, d = 600, 16
    y = (rng.uniform(0, 1, N) > 0.5).astype(np.int64)
    X = rng.standard_normal((N, d)) * 0.1
    X[:, 0] = (2.0 * y - 1.0) + 0.2 * rng.standard_normal(N)
    tr, te = slice(0, 400), slice(400, N)
    f1 = P.f1_at_zero(P.linear_predict(P.lsclf_fit(X[tr], y[tr], 1.0), X[te]), y[te])
    assert f1 > 0.9, f1
    f1s = P.f1_at_zero(P.linear_predict(P.lsclf_fit(X[tr], P.shuffle_labels(y[tr], 2), 1.0), X[te]), y[te])
    assert f1s < 0.7, f1s


def test_induction_labels():
    assert P.induction_labels(np.array([10, 11, 10, 12, 11])).tolist() == [0, 0, 0, 1, 0]


def test_position_labels():
    assert np.allclose(P.position_labels(5), [0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(AssertionError):
        P.position_labels(1)


def _make_blocks(n_win, tok_per, treat_noise, null_noise, seed):
    rng = np.random.default_rng(seed)
    return [{"y": (yy := rng.uniform(0, 1, tok_per)),
             "treat": yy + treat_noise * rng.standard_normal(tok_per),
             "null": yy + null_noise * rng.standard_normal(tok_per)} for _ in range(n_win)]


def test_probe_contrast_block_ci_brackets_planted_positive():
    out = P.probe_contrast_block_ci(_make_blocks(60, 40, 0.5, 0.02, 3), P.r2_metric, "treat", "null", 2000, 7)
    assert out["contrast"] > 0.1 and out["lo"] > 0 and out["p_le0"] < 0.05, out


def test_probe_contrast_block_ci_straddles_zero_when_exchangeable():
    out = P.probe_contrast_block_ci(_make_blocks(60, 40, 0.2, 0.2, 4), P.r2_metric, "treat", "null", 2000, 8)
    assert out["lo"] < 0 < out["hi"] and 0.1 < out["p_le0"] < 0.9, out


def test_probe_contrast_block_ci_fails_loud_on_empty():
    with pytest.raises(AssertionError):
        P.probe_contrast_block_ci([], P.r2_metric, "treat", "null")


# ---------------------------------------------------------------------------
# Energy-matched subspace: hits the target energy, orthonormal, distinct per seed.
# ---------------------------------------------------------------------------
def test_energy_matched_subspace_hits_target_energy():
    rng = np.random.default_rng(10)
    n, k = 128, 16
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))            # orthonormal eigvecs
    mu = np.sort(np.exp(-np.linspace(0, 6, n)))[::-1]           # descending, concentrated energy
    G = (V * mu) @ V.T
    e_target = 0.30
    Q, e = P.energy_matched_subspace(V, mu, e_target, k, seed=1, tol=0.03)
    assert abs(e - e_target) < 0.06, e                          # near the target energy
    actual = float(np.trace(Q.T @ G @ Q)) / float(np.trace(G))  # re-measure on G
    assert abs(actual - e) < 1e-6, (actual, e)
    assert np.abs(Q.T @ Q - np.eye(k)).max() < 1e-8             # orthonormal
    Q2, _ = P.energy_matched_subspace(V, mu, e_target, k, seed=2, tol=0.03)
    assert not np.allclose(Q, Q2)                               # distinct per seed (=> calibrates θ)


def test_synth_contrast_bootstrap():
    rng = np.random.default_rng(11)
    N = 200
    base = (rng.uniform(size=N) > 0.1).astype(float)            # high base accuracy
    treat = (rng.uniform(size=N) > 0.8).astype(float)           # U_unt collapses it
    null = (rng.uniform(size=N) > 0.2).astype(float)            # null barely moves it
    out = P.synth_contrast_bootstrap(base, treat, null, 2000, 7)
    assert out["contrast"] > 0.1 and out["lo"] > 0, out         # treat collapses more than null
    null2 = (rng.uniform(size=N) > 0.2).astype(float)           # independent, same distribution as null
    out0 = P.synth_contrast_bootstrap(base, null, null2, 2000, 7)
    assert out0["lo"] < 0 < out0["hi"], out0                    # exchangeable arms straddle 0


def test_bh_adjust():
    adj = P.bh_adjust([0.001, 0.5, 0.02])
    assert abs(adj[0] - 0.003) < 1e-9 and abs(adj[2] - 0.03) < 1e-9 and all(0 <= x <= 1 for x in adj)


def test_copy_corpus_gold_is_induction_target():
    for prompt, gold in P.copy_corpus(20, 6, 5):
        toks = prompt.split()
        assert len(toks) == 11 and toks[:6] == toks[6:] + [gold] and gold == toks[5]


def test_position_corpus_gold_at_queried_position():
    for prompt, gold in P.position_corpus(20, 6, 6):
        toks = prompt.split()
        assert toks[2 * (int(toks[-1]) - 1) + 1] == gold


def test_sided_projector_hook_removes_band_on_chosen_side():
    import torch.nn as nn
    n, k = 32, 4
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((n, n)))
    Qk = Q[:, :k]

    class Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj, s.k_proj = nn.Linear(n, n, bias=False), nn.Linear(n, n, bias=False)
            with torch.no_grad():
                s.q_proj.weight.copy_(torch.eye(n)); s.k_proj.weight.copy_(torch.eye(n))

    class Layer(nn.Module):
        def __init__(s):
            super().__init__(); s.self_attn = Attn()

    layer = Layer()
    x = torch.randn(1, 3, n)
    with J._InputProjectSided([layer], {0: Qk}, "cpu", torch.float32, "k"):
        yk, yq = layer.self_attn.k_proj(x), layer.self_attn.q_proj(x)
    assert (yk.double() @ torch.tensor(Qk)).abs().max().item() < 1e-5      # band removed from k
    assert (yq - x).abs().max().item() < 1e-6                              # q untouched


def test_ppl_from_pw():
    assert abs(J._ppl_from_pw([{"ppl_pw": [10.0, 5]}, {"ppl_pw": [20.0, 5]}]) - math.exp(3.0)) < 1e-9


def test_monotone():
    assert J._monotone([0.01, 0.05, 0.10]) and J._monotone([-0.01, -0.05, -0.1])
    assert not J._monotone([0.01, 0.15, 0.1])


# ---------------------------------------------------------------------------
# The FROZEN four-conjunct naming rule (§6, energy-matched primary). Build minimal abl/synth/bands.
# ---------------------------------------------------------------------------
def _mk_abl(specs, k_grid=(8, 32, 128), L=9, poscontrol=True, dppl_unt=49.0):
    """specs: {probe: {contrast, lo, hi, p, theta, monotone}}. Builds a 3-k abl with the headline
    (last k) carrying the real lo/hi/p/theta; the k-trajectory is monotone-increasing to the headline
    contrast (or up-then-down if monotone=False)."""
    abl = []
    for ki, k in enumerate(k_grid):
        last = ki == len(k_grid) - 1
        probes = {str(L): {}}
        for w, s in specs.items():
            h = s["contrast"]
            traj = [0.33 * h, 0.66 * h, h] if s.get("monotone", True) else [0.33 * h, 1.6 * h, h]
            ex = traj[ki]
            probes[str(L)][w] = {"excess_vs_em": {"contrast": ex,
                                 "lo": s["lo"] if last else ex - 1, "hi": s.get("hi", ex + 0.1) if last else ex + 1,
                                 "p_le0": s["p"] if last else 0.5}, "theta_em": s["theta"], "em_null_disc": []}
        abl.append({"k": k, "dppl_unt": dppl_unt, "dppl_em": [1.0], "dppl_rand": [1.0], "dppl_bots": 1.0,
                    "dppl_tops": 1500.0, "poscontrol_valid": poscontrol, "e_unt": 0.28, "probes": probes})
    return abl


_BANDS = {"pos_r2_clean": 0.6, "pos_r2_shuffle": 0.0, "pos_corr_clean": 0.85, "pos_corr_shuffle": 0.0,
          "indu_f1_clean": 0.8, "indu_f1_shuffle": 0.3, "sink_base": 0.2}
_SBASE = {"copy_has_range": True, "posq_has_range": True}
_NOSYN = {"contrast": 0.0, "lo": -0.1, "hi": 0.1, "p_le0": 0.5}            # synthetic null (no collapse)
_YESSYN = {"contrast": 0.3, "lo": 0.2, "hi": 0.4, "p_le0": 0.0}            # synthetic collapse


def _syn(posq, copy):
    return {"posq_contrast_unt_vs_em": posq, "copy_contrast_unt_vs_em": copy}


def _clear(contrast, theta=0.01, p=0.0, monotone=True, lo=None, hi=None):
    return {"contrast": contrast, "lo": (contrast - 0.02 if lo is None else lo),
            "hi": (contrast + 0.05 if hi is None else hi), "p": p, "theta": theta, "monotone": monotone}


def _null(contrast=0.001):
    return {"contrast": contrast, "lo": -0.02, "hi": 0.05, "p": 0.5, "theta": 0.01, "monotone": True}


def test_verdict_h_pos_single_dominant():
    abl = _mk_abl({"pos": _clear(0.10), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"].startswith("H_pos"), v


def test_verdict_h_indu_single_dominant():
    abl = _mk_abl({"pos": _null(), "indu": _clear(0.10), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _YESSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"].startswith("H_indu"), v


def test_verdict_h_sink_two_sided():
    # sink mass INCREASES under U_unt ablation => negative contrast; two-sided test still fires.
    abl = _mk_abl({"pos": _null(), "indu": _null(),
                   "sink": {"contrast": -0.02, "lo": -0.03, "hi": -0.008, "p": 0.99, "theta": 0.002, "monotone": True}})
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"].startswith("H_sink"), v


def test_verdict_h_compound():
    abl = _mk_abl({"pos": _clear(0.10), "indu": _clear(0.10), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _YESSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_compound", v


def test_verdict_single_dominant_margin_collapses_compound():
    abl = _mk_abl({"pos": _clear(0.30), "indu": _clear(0.10), "sink": _null()})   # pos >> 1.5x indu
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _YESSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"].startswith("H_pos"), v


def test_verdict_h_generic_none_beats_em_but_core_live():
    abl = _mk_abl({"pos": _null(), "indu": _null(), "sink": _null()}, dppl_unt=49.0)
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v


def test_verdict_h_other_core_not_live():
    abl = _mk_abl({"pos": _null(), "indu": _null(), "sink": _null()}, dppl_unt=2.0)
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_other", v


def test_verdict_instrument_broken():
    abl = _mk_abl({"pos": _clear(0.10), "indu": _null(), "sink": _null()}, poscontrol=False)
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "INSTRUMENT_BROKEN", v


def test_verdict_synthetic_gate_blocks_pos():
    # pos probe clears the wikitext contrast but the synthetic posq task does NOT collapse => not named.
    abl = _mk_abl({"pos": _clear(0.10), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v


def test_verdict_monotone_gate_blocks_pos():
    abl = _mk_abl({"pos": _clear(0.10, monotone=False), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v


def test_verdict_theta_gate_blocks_pos():
    # contrast positive & CI excludes 0 but below the discovery-calibrated θ => not named.
    abl = _mk_abl({"pos": _clear(0.10, theta=0.5), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v


def test_verdict_construct_invalid_blocks_pos():
    bands = dict(_BANDS, pos_corr_clean=0.02, pos_corr_shuffle=0.0)             # probe ≈ shuffle => VOID
    abl = _mk_abl({"pos": _clear(0.10), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _NOSYN), bands, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v


def test_verdict_synthetic_waived_when_no_range_names_pos():
    # posq at floor (no baseline range) => synthetic conjunct WAIVED => H_pos named on the probe excess
    # (this is the smoke-2026-06-19 reality: PG1 can't do zero-shot position retrieval).
    sbase = dict(_SBASE, posq_has_range=False)
    abl = _mk_abl({"pos": _clear(0.10), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _NOSYN), _BANDS, sbase, [8, 32, 128])
    assert v["named"].startswith("H_pos"), v
    assert "waived" in v["detail"]["pos"]["synthetic_flag"], v


def test_verdict_sink_floor_blocks_sink():
    bands = dict(_BANDS, sink_base=0.01)                                       # below §8 sink floor
    abl = _mk_abl({"pos": _null(), "indu": _null(),
                   "sink": {"contrast": -0.02, "lo": -0.03, "hi": -0.008, "p": 0.99, "theta": 0.002, "monotone": True}})
    v = J._verdict(abl, 128, 9, _syn(_NOSYN, _NOSYN), bands, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v


def test_verdict_bh_gate_blocks_pos():
    # pos clears CI/θ/monotone/synth but its p is large => BH-adjusted not significant => not named.
    abl = _mk_abl({"pos": _clear(0.10, p=0.30), "indu": _null(), "sink": _null()})
    v = J._verdict(abl, 128, 9, _syn(_YESSYN, _NOSYN), _BANDS, _SBASE, [8, 32, 128])
    assert v["named"] == "H_generic", v
