"""Unit tests for the pure logic of DARWIN's W1-A/B/C and W0-9/11/13 job modules.

Only the parts that can be exercised without a GPU or the network: the paired-accuracy bootstrap,
the intercom probe's information accounting, the checkpoint-suite frame picker, and the
bias/amplifier shape contracts. Run by scripts/darwin_check.sh with every other darwin test.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from infra.aws_pg.jobs import darwin_w0_ext as EXT
from infra.aws_pg.jobs import darwin_w1_icl as ICL


def test_paired_acc_ci_is_paired():
    """A constant +0.1 improvement on every item has a CI that excludes 0 and a point of 0.1."""
    rng = np.random.default_rng(0)
    null = (rng.random(400) < 0.4).astype(float)
    treat = null.copy()
    flip = np.where(null == 0)[0][:40]
    treat[flip] = 1.0
    r = ICL.paired_acc_ci(treat, null, n_boot=500, seed=3)
    assert abs(r["contrast"] - 0.1) < 1e-9, r
    assert r["ci_lo"] > 0.0, r
    assert r["n"] == 400
    # identical arms -> a contrast of exactly zero and a CI that contains it
    z = ICL.paired_acc_ci(null, null, n_boot=200, seed=3)
    assert z["contrast"] == 0.0 and z["ci_lo"] == 0.0 and z["ci_hi"] == 0.0, z


def test_paired_acc_ci_rejects_unpaired_lengths():
    try:
        ICL.paired_acc_ci(np.zeros(5), np.zeros(6))
    except AssertionError as e:
        assert "different item counts" in str(e)
    else:
        raise AssertionError("mismatched arms must fail loud")


def test_probe_bits_recovers_a_clean_channel():
    """A message written into an orthonormal frame and read from a noiseless copy is ~32 bits."""
    rng = np.random.default_rng(1)
    k, n, T = 8, 64, 600
    U = np.linalg.qr(rng.standard_normal((n, k)))[0]
    M = rng.integers(0, 2, (T, k)) * 2.0 - 1.0
    read = M @ U.T + 0.001 * rng.standard_normal((T, n))
    r = ICL.probe_bits(M, read, ICL.RIDGE, T // 2)
    assert r["mean_bit_acc"] > 0.99, r
    assert r["bits_per_token"] > k - 0.5, r
    assert r["n_fit"] == T // 2 and r["n_held"] == T - T // 2


def test_probe_bits_on_pure_noise_is_near_zero():
    rng = np.random.default_rng(2)
    k, n, T = 8, 64, 600
    M = rng.integers(0, 2, (T, k)) * 2.0 - 1.0
    read = rng.standard_normal((T, n))
    r = ICL.probe_bits(M, read, ICL.RIDGE, T // 2)
    assert r["bits_per_token"] < 1.5, r          # chance decoding carries ~no information


def test_probe_bits_requires_held_out_tokens():
    M = np.ones((4, 2))
    try:
        ICL.probe_bits(M, np.ones((4, 3)), ICL.RIDGE, 4)
    except AssertionError as e:
        assert "held-out" in str(e)
    else:
        raise AssertionError("a probe with no held-out half must fail loud")


def test_pick_frames_keeps_ends_and_spreads():
    revs = [(i * 1000, f"step{i * 1000}") for i in range(1, 51)]
    f = EXT.pick_frames(revs, 12)
    assert f[0] == revs[0] and f[-1] == revs[-1], f
    assert len(f) == 12, len(f)
    assert f == sorted(set(f)), "frames must be sorted and unique"
    assert EXT.pick_frames(revs[:5], 12) == revs[:5], "fewer revisions than frames -> take all"


def test_pick_frames_rejects_a_degenerate_suite():
    try:
        EXT.pick_frames([(0, "step0")], 4)
    except AssertionError as e:
        assert "need >= 2" in str(e)
    else:
        raise AssertionError("a one-checkpoint suite must fail loud, not silently produce nothing")


def test_step_regex_parses_the_real_branch_shapes():
    for name, want in (("step1000-tokens4B", 1000), ("stage1-step140000-tokens587B", 140000),
                       ("step0", 0)):
        m = EXT.STEP_RE.search(name)
        assert m and int(m.group(1)) == want, (name, m)
    assert EXT.STEP_RE.search("main") is None


def test_window_banks_shapes_and_count():
    rng = np.random.default_rng(4)
    n, k, ndelta, window = 24, 4, 6, 3
    keys = [("q_proj", 0)]
    terms = []
    for i in range(ndelta):
        A = rng.standard_normal((10, n))
        terms.append({"from_step": i * 100, "to_step": (i + 1) * 100,
                      "term": {keys[0]: A.T @ A / float((A ** 2).sum())}})
    wins = EXT.window_banks(terms, keys, window, k)
    assert len(wins) == ndelta - window + 1, len(wins)
    U = wins[0]["banks"][keys[0]]
    assert U.shape == (n, k), U.shape
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-8), "window bank must be orthonormal"
    assert wins[0]["lo_step"] == 0 and wins[0]["hi_step"] == window * 100


def test_confined_bias_parameter_count_matches_the_spec_incumbent():
    """576 = 18 layers x 32 — the count §4/W1-A matches its unconstrained incumbent against."""
    assert 18 * 32 == 576
    coords = EXT_free = ICL.freecoords_by_layer(2048, 32, list(range(18)), 1337)
    assert len(coords) == 18 and all(len(v) == 32 for v in coords.values())
    flat = np.concatenate([v for v in coords.values()])
    assert flat.max() < 2048 and flat.min() >= 0
    for v in coords.values():
        assert len(set(v.tolist())) == 32, "coordinates must be sampled without replacement"
    assert EXT_free is coords


def test_freecoords_are_layer_specific_but_reproducible():
    a = ICL.freecoords_by_layer(2048, 32, [0, 1], 1337)
    b = ICL.freecoords_by_layer(2048, 32, [0, 1], 1337)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], a[1]), "different layers must draw different coordinates"


def test_prompt_builder_is_monotone_in_shots():
    ex = {"algebra": [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4}
                      for i in range(16)]}
    p0 = ICL.prompt_for("algebra", ex, 0)
    p4 = ICL.prompt_for("algebra", ex, 4)
    p16 = ICL.prompt_for("algebra", ex, 16)
    assert p0 in p4 and p4 in p16, "shot prefixes must nest, or the ladder is not one prompt family"
    assert len(p16) > len(p4) > len(p0)
    item = {"question": "target", "choices": ["w", "x", "y", "z"], "answer": 0}
    assert ICL.prompt_for("algebra", ex, 4, item).startswith(p4)
    assert "target" in ICL.prompt_for("algebra", ex, 4, item)


def test_joint_least_squares_edit_is_exact_and_the_sequential_sum_is_not():
    """The defect this test exists for: ten rank-1 inserts summed do NOT satisfy their own keys.

    W1-L places every key inside one 32-dim subspace, so the keys overlap by ~1/sqrt(32) and each
    insert corrupts the other nine. The measured residual was 0.20 on the real model and 0.62 here
    -- i.e. most of every edit was gone before a single training step, and the arm would have
    reported edit interference as edit durability.
    """
    from infra.aws_pg.jobs import darwin_w1_edit as ED
    rng = np.random.default_rng(0)
    Wm = rng.standard_normal((64, 128)) / 8
    U = np.linalg.qr(rng.standard_normal((128, 32)))[0]
    dW, pairs = ED.edit_batch(rng, Wm, U, 10, 0.25)
    assert ED.edit_recovery(Wm + dW, pairs)[0] < 1e-10, "the joint solve must satisfy every key"
    seq = np.zeros_like(Wm)
    for k, v, _b in pairs:
        seq += np.outer(v - Wm @ k, k) / float(k @ k)
    assert ED.edit_recovery(Wm + seq, pairs)[0] > 0.1, (
        "if the sequential sum were also exact this test would be guarding nothing")


def test_edit_magnitude_is_bank_independent():
    """Every bank must get the same SIZE of edit, or durability is confounded with base gain."""
    from infra.aws_pg.jobs import darwin_w1_edit as ED
    Wm = np.diag(np.concatenate([np.full(16, 10.0), np.full(16, 0.001)]))
    hi = np.eye(32)[:, :8]                       # the high-gain input directions
    lo = np.eye(32)[:, 24:]                      # the low-gain ones
    mags = []
    for U in (hi, lo):
        _, pairs = ED.edit_batch(np.random.default_rng(2), Wm, U, 4, 0.25)
        mags.append(np.mean([np.linalg.norm(v - b) for _k, v, b in pairs]))
    assert abs(mags[0] - mags[1]) / mags[0] < 0.2, (
        f"edit magnitude differs {mags} across a 10000x gain ratio — the old ||W k||-scaled "
        f"target made bottomsigma32's edits microscopic and its readout meaningless")


def test_edit_keys_must_be_independent():
    from infra.aws_pg.jobs import darwin_w1_edit as ED
    rng = np.random.default_rng(3)
    Wm = rng.standard_normal((16, 32))
    U = np.linalg.qr(rng.standard_normal((32, 4)))[0]
    try:
        ED.edit_batch(rng, Wm, U, 8, 0.25)       # 8 edits in a 4-dim subspace
    except AssertionError as e:
        assert "cannot be placed independently" in str(e)
    else:
        raise AssertionError("over-subscribed placements must fail loud")


def test_spec_expectations_cover_every_arm_the_module_runs():
    assert set(ICL.RUNNERS) == set(ICL.SPEC_EXPECTATIONS), (
        "an arm with no pre-registered refuter would write rows with no expectation to fail")
    assert set(EXT.RUNNERS) == set(EXT.SPEC_EXPECTATIONS)


def test_dose_ladders_satisfy_the_section_3_grid():
    from infra.pg import darwin_core as D
    D.check_dose_grid(list(ICL.LAM_LADDER), arm="W1-A")
    D.check_dose_grid(list(ICL.AMP_LADDER), arm="W1-C")
    D.check_dose_grid(list(ICL.GAIN_LADDER), arm="W1-B", log_spaced=False)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("test_darwin_w1_icl: all tests passed")
