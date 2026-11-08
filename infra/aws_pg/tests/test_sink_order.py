"""Weight-free CPU dry-run gate for sink-order — prereg §9.1 / §13.3 gate-1 logic test.
NO model, NO datasets, NO GPU — pure logic on tiny planted tensors, seconds on a laptop.

    python3 -m pytest infra/aws_pg/tests/test_sink_order.py -q

Covers: the additive-bias builders (sink / matched-mass null / suppress) — correct rows/cols, never the
BOS or image cols for the null, determinism; the softmax SEMANTIC (a +b bias RAISES that col's mass,
monotone); closest_level selection; the position Excess sign (corr(null)-corr(sink)>0 ⇔ sink degrades
more) + θ; and the FROZEN §6 verdict rule mapping each scenario to the right label.
"""
import numpy as np
import pytest

from infra.pg import sink_clamp as SC
from infra.pg import conserved_core_probes as P
from infra.aws_pg.jobs import sink_order as J


# geometry used across tests: [0..bos_idx-1 image][bos_idx=BOS][pos..pos+n_text-1 text][trailing]
POS, N_TEXT, BOS, T = 5, 6, 4, 14          # image cols 0..3, BOS=4, text 5..10, trailing 11..13


# ---------------------------------------------------------------------------
# Bias builders.
# ---------------------------------------------------------------------------
def test_bos_bias_hits_only_bos_col_for_text_rows():
    bias, tgt = SC.bos_bias(POS, N_TEXT, BOS, T, b=3.0)
    assert bias.shape == (T, T)
    text_rows = np.arange(POS, POS + N_TEXT)
    assert np.allclose(bias[text_rows, BOS], 3.0)
    # every other entry is zero (only the BOS col of text rows is biased)
    m = np.zeros((T, T), bool)
    m[text_rows, BOS] = True
    assert np.allclose(bias[~m], 0.0)
    assert (tgt[text_rows] == BOS).all()
    assert (tgt[:POS] == -1).all() and (tgt[POS + N_TEXT:] == -1).all()


def test_matched_mass_null_picks_visible_text_cols_never_bos_or_image():
    bias, tgt = SC.matched_mass_bias(POS, N_TEXT, BOS, T, b=3.0, seed=0)
    for t in range(POS, POS + N_TEXT):
        c = tgt[t]
        assert POS <= c <= t, (t, c)            # causally visible TEXT col
        assert c != BOS and c >= POS, (t, c)    # never BOS, never image
        assert bias[t, c] == 3.0
        assert (bias[t, :c] == 0).all() and (bias[t, c + 1:] == 0).all()  # exactly one biased col / row
    assert (bias[:POS] == 0).all()              # no image-row bias
    # never touches the BOS column or any image column
    assert np.allclose(bias[:, BOS], 0.0)
    assert np.allclose(bias[:, :BOS], 0.0)


def test_matched_mass_null_deterministic_by_seed():
    a1, t1 = SC.matched_mass_bias(POS, N_TEXT, BOS, T, 3.0, seed=7)
    a2, t2 = SC.matched_mass_bias(POS, N_TEXT, BOS, T, 3.0, seed=7)
    a3, t3 = SC.matched_mass_bias(POS, N_TEXT, BOS, T, 3.0, seed=8)
    assert np.array_equal(t1, t2) and np.array_equal(a1, a2)
    assert not np.array_equal(t1, t3)           # different seed => different draw (with high prob)


def test_suppress_bias_is_negative_on_bos():
    bias, tgt = SC.suppress_bias(POS, N_TEXT, BOS, T, b_supp=8.0)
    text_rows = np.arange(POS, POS + N_TEXT)
    assert np.allclose(bias[text_rows, BOS], -8.0)
    assert (tgt[text_rows] == BOS).all()


def test_geometry_asserts_fail_loud():
    with pytest.raises(AssertionError):
        SC.bos_bias(POS, N_TEXT, BOS + 1, T, 1.0)          # bos_idx != pos-1
    with pytest.raises(AssertionError):
        SC.bos_bias(POS, N_TEXT, BOS, POS + N_TEXT - 1, 1.0)  # text overflows T


# ---------------------------------------------------------------------------
# Softmax semantic: a +b additive logit bias RAISES that column's post-softmax mass (monotone).
# ---------------------------------------------------------------------------
def test_additive_bias_raises_mass_monotone():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(20)
    col = 3
    masses = [SC.softmax_mass_after_bias(logits, col, b) for b in [0, 1, 2, 4, 8]]
    assert all(masses[i + 1] >= masses[i] for i in range(len(masses) - 1)), masses
    assert masses[-1] > masses[0] + 0.3        # b=8 substantially raises it
    base = np.exp(logits - logits.max()); base = base / base.sum()
    assert abs(masses[0] - base[col]) < 1e-9    # b=0 == baseline


# ---------------------------------------------------------------------------
# closest_level selection.
# ---------------------------------------------------------------------------
def test_closest_level():
    assert SC.closest_level([0.2, 0.4, 0.64, 0.9], 0.6) == 2
    assert SC.closest_level([0.2, 0.4, 0.64, 0.9], 0.95) == 3
    with pytest.raises(AssertionError):
        SC.closest_level([], 0.5)


# ---------------------------------------------------------------------------
# Position Excess sign + θ (reuses the probe core).
# ---------------------------------------------------------------------------
def _planted_window(text, is_conf, sink_drop, null_drop, n=20, seed=0):
    """A window whose feats[L]'s first coord = position (+ noise scaled by *_drop): a larger drop scales
    down the position signal, simulating more degradation. Returns the per-window pos-pred dicts."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, n)
    # the baseline-trained probe is ~identity on coord 0; degradation = shrink the signal toward noise.
    pred_sink = (1 - sink_drop) * y + sink_drop * rng.uniform(0, 1, n)
    pred_null = (1 - null_drop) * y + null_drop * rng.uniform(0, 1, n)
    return y, pred_sink, pred_null


def test_pos_excess_positive_when_sink_degrades_more():
    sink_preds, null_preds = {}, {}
    for i in range(12):
        txt = f"w{i}"
        y, ps, pn = _planted_window(txt, True, sink_drop=0.8, null_drop=0.1, seed=i)
        sink_preds[txt] = {"is_conf": True, "y": y, "pred": ps}
        null_preds[txt] = {"is_conf": True, "y": y, "pred": pn}
    ex = J._pos_excess(sink_preds, null_preds, n_boot=300, seed=1)
    assert ex["contrast"] > 0 and ex["lo"] > 0, ex      # sink degrades more => corr(null)-corr(sink) > 0


def test_pos_excess_brackets_zero_when_equal():
    sink_preds, null_preds = {}, {}
    for i in range(12):
        txt = f"w{i}"
        y, ps, pn = _planted_window(txt, True, sink_drop=0.3, null_drop=0.3, seed=i)
        sink_preds[txt] = {"is_conf": True, "y": y, "pred": ps}
        null_preds[txt] = {"is_conf": True, "y": y, "pred": pn}
    ex = J._pos_excess(sink_preds, null_preds, n_boot=300, seed=1)
    assert ex["lo"] < 0 < ex["hi"], ex                 # equal degradation => Excess straddles 0


# ---------------------------------------------------------------------------
# The FROZEN §6 verdict rule.
# ---------------------------------------------------------------------------
def _ex(contrast, lo, hi, p):
    return {"contrast": contrast, "lo": lo, "hi": hi, "p_le0": p}


def _med(frac, lo, hi):
    return {"fraction": frac, "lo": lo, "hi": hi}


def test_verdict_instrument_broken():
    v = J._verdict(_ex(0.5, 0.3, 0.7, 0.0), _ex(10, 5, 15, 0.0), _ex(0.2, 0.1, 0.3, 0.0),
                   0.02, _med(0.6, 0.5, 0.7), True, poscontrol_valid=False, clamp_bite_ok=True,
                   null_self_frac=0.1)
    assert v["named"] == "INSTRUMENT_BROKEN", v
    v2 = J._verdict(_ex(0.5, 0.3, 0.7, 0.0), _ex(10, 5, 15, 0.0), _ex(0.2, 0.1, 0.3, 0.0),
                    0.02, _med(0.6, 0.5, 0.7), True, poscontrol_valid=True, clamp_bite_ok=False,
                    null_self_frac=0.1)
    assert v2["named"] == "INSTRUMENT_BROKEN", v2


def test_verdict_sink_primary():
    v = J._verdict(_ex(0.5, 0.3, 0.7, 0.0), _ex(10, 5, 15, 0.0), _ex(0.2, 0.1, 0.3, 0.0),
                   theta=0.05, mediation=_med(0.7, 0.5, 0.85), monotone_pos=True,
                   poscontrol_valid=True, clamp_bite_ok=True, null_self_frac=0.1)
    assert v["named"] == "H_sink-primary", v


def test_verdict_compound_when_sink_specific_but_small_mediation():
    v = J._verdict(_ex(0.3, 0.1, 0.5, 0.001), _ex(6, 2, 10, 0.001), _ex(0.1, 0.02, 0.18, 0.001),
                   theta=0.05, mediation=_med(0.30, 0.18, 0.42), monotone_pos=True,
                   poscontrol_valid=True, clamp_bite_ok=True, null_self_frac=0.1)
    assert v["named"] == "H_compound", v


def test_verdict_addressing_primary():
    # sink does ~nothing (mediation<0.25, Excess brackets 0) AND the matched-null is also inert.
    v = J._verdict(_ex(0.02, -0.05, 0.09, 0.4), _ex(0.5, -1, 2, 0.4), _ex(0.0, -0.05, 0.05, 0.5),
                   theta=0.05, mediation=_med(0.10, -0.02, 0.22), monotone_pos=False,
                   poscontrol_valid=True, clamp_bite_ok=True, null_self_frac=0.08)
    assert v["named"] == "H_addressing-primary", v


def test_verdict_generic_when_null_also_degrades():
    # Excess brackets 0 but the matched-mass null ITSELF reproduces most of the crater => generic.
    v = J._verdict(_ex(0.03, -0.05, 0.11, 0.35), _ex(1, -2, 4, 0.35), _ex(0.0, -0.05, 0.05, 0.5),
                   theta=0.05, mediation=_med(0.6, 0.45, 0.75), monotone_pos=True,
                   poscontrol_valid=True, clamp_bite_ok=True, null_self_frac=0.6)
    assert v["named"] == "H_generic", v


def test_monotone_helper():
    assert J._monotone([0.1, 0.3, 0.3, 0.5])
    assert not J._monotone([0.1, 0.5, 0.2])
