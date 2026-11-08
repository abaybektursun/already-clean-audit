"""Weight-free unit tests for every pure HUMBOLDT statistic (§9 step 2).

Run: PYTHONPATH=. python3 -m pytest infra/pg/tests/test_humboldt_core.py -q
     PYTHONPATH=. python3 infra/pg/tests/test_humboldt_core.py     (no pytest needed)

Each test states the property the statistic must have and the observation that would falsify it.
"""
import json
import pathlib
import tempfile

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S


def test_principal_angles_identity_and_orthogonal():
    U = H.haar_subspace(64, 8, 0)
    assert float(H.principal_angles(U, U).max()) < 1e-7          # a subspace against itself: 0
    V = H.complement_subspace(U, 1)
    assert abs(float(H.principal_angles(U, V).min()) - np.pi / 2) < 1e-7   # orthogonal: all π/2


def test_leverage_sums_to_k_and_pr_bounds():
    U = H.haar_subspace(128, 16, 3)
    lev = H.leverage(U)
    assert abs(float(lev.sum()) - 16) < 1e-9
    assert 0.0 <= float(lev.min()) and float(lev.max()) <= 1.0
    flat = np.full(100, 0.01)
    spike = np.zeros(100); spike[0] = 1.0
    assert abs(H.participation_ratio(flat) - 100) < 1e-9
    assert abs(H.participation_ratio(spike) - 1.0) < 1e-9


def test_reconstruction_residual_can_fail():
    """The G1 gate must distinguish a complete eigenbasis from an incomplete one. A version that
    reports ~0 for BOTH is the bug this test exists to catch."""
    rng = np.random.default_rng(0)
    n, m, k = 64, 32, 8
    V = np.linalg.qr(rng.standard_normal((n, n)))[0]
    W = rng.standard_normal((m, n))
    assert H.reconstruction_residual(W, V, k) < 1e-12
    Vt = V.copy(); Vt[:, k:] = 0.0
    assert H.reconstruction_residual(W, Vt, k) > 0.5
    assert H.reconstruction_residual(W, V[:, :n - 3], k) > 0.05


def test_sigma_subspaces_economy_on_rank_deficient():
    """C5: for a 256×2048-shaped matrix the economy basis spans min(m, n) directions, and the
    bottom-σ twin must come from those, not from the exact kernel."""
    rng = np.random.default_rng(2)
    W = rng.standard_normal((16, 64))
    bots, tops, sig = H.sigma_subspaces(W, 4)
    assert sig.size == 16 and bots.shape == (64, 4) and tops.shape == (64, 4)
    H.orthonormal_check(bots)
    H.orthonormal_check(tops)
    assert float(np.linalg.norm(W @ tops)) > float(np.linalg.norm(W @ bots))


def test_build_banks_battery_complete_and_orthonormal():
    rng = np.random.default_rng(4)
    n, m = 96, 48
    A = rng.standard_normal((n, n))
    G = A @ A.T
    _lam, V = np.linalg.eigh(G)
    W = rng.standard_normal((m, n))
    banks = H.build_banks(V, W, k=8, seed_base=11)
    assert set(banks) == set(H.BANK_IDS)
    for bid, U in banks.items():
        assert U.shape == (n, 8)
        H.orthonormal_check(U)
    # core32 and anti32 are the two ends of the same eigenbasis → mutually orthogonal.
    assert abs(float(H.principal_angles(banks["core32"], banks["anti32"]).min()) - np.pi / 2) < 1e-7
    # complement32 lives strictly outside core32.
    assert float(np.abs(banks["core32"].T @ banks["complement32"]).max()) < 1e-8


def test_rope_coupling_normalises_and_localises():
    hd, n = 16, 6
    nb = hd // 2
    U = np.zeros((n, 1)); U[2, 0] = 1.0
    W = np.zeros((hd, n)); W[3, 2] = 1.0; W[3 + nb, 2] = 1.0
    c = H.rope_coupling(W, U, head_dim=hd, n_out_heads=1)
    assert c.size == nb and abs(float(c.sum()) - 1.0) < 1e-12 and int(np.argmax(c)) == 3


def test_rope_bands_accepts_default_rope_and_rejects_rescaled():
    """transformers reports plain RoPE as None, {}, or {"rope_type": "default"} depending on the
    version; a rescaled rope must fail loud rather than be silently mis-binned."""
    class Cfg:
        head_dim = 256
        num_attention_heads = 8
        num_key_value_heads = 1
        rope_theta = 10000.0
        rope_scaling = None

        def to_dict(self):
            return {k: v for k, v in vars(type(self)).items() if not k.startswith("_")}

    for rs in (None, {}, {"rope_type": "default"}, {"type": "default"}):
        Cfg.rope_scaling = rs
        b = H.rope_bands(Cfg())
        assert b["n_bands"] == 128 and len(b["inv_freq"]) == 128
        assert b["rope_theta"] == 10000.0 and b["rope_theta_source"] == "cfg.rope_theta"
    for rs in ({"rope_type": "linear", "factor": 2.0}, {"type": "yarn"}):
        Cfg.rope_scaling = rs
        try:
            H.rope_bands(Cfg())
        except AssertionError:
            continue
        raise AssertionError(f"rope_bands accepted a rescaled rope: {rs}")

    # the base may live under rope_parameters instead (transformers moved it)
    class Cfg2(Cfg):
        rope_theta = None
        rope_scaling = None
        rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    b = H.rope_bands(Cfg2())
    assert b["rope_theta"] == 10000.0 and "rope_parameters" in b["rope_theta_source"]

    # and if it is nowhere, fail loud rather than assume a default
    class Cfg3(Cfg):
        rope_theta = None
        rope_scaling = None
        rope_parameters = None
    try:
        H.rope_bands(Cfg3())
    except AssertionError as e:
        assert "no rope base" in str(e)
    else:
        raise AssertionError("rope_bands invented a rope base")


def test_energy_fraction_bounds_and_chance():
    rng = np.random.default_rng(5)
    n, k = 256, 16
    U = H.haar_subspace(n, k, 6)
    e_in = H.energy_fraction(rng.standard_normal((32, k)) @ U.T, U)
    e_iso = H.energy_fraction(rng.standard_normal((4096, n)), U)
    assert abs(float(e_in.mean()) - 1.0) < 1e-9
    assert abs(float(e_iso.mean()) - k / n) < 0.02


def test_autocorr_and_spectral_peaks_find_a_planted_period():
    t = np.arange(1024)
    x = np.sin(2 * np.pi * t / 32)
    assert abs(H.spectral_peaks(x, top=1)["period"][0] - 32) < 1.0
    assert H.autocorr(x, 64)[31] > 0.9
    assert np.isnan(H.autocorr(np.ones(10), 4)).all()          # a constant series has no structure


def test_mean_ci_regimes_agree():
    rng = np.random.default_rng(7)
    small = rng.standard_normal(400)
    s = H.mean_ci(small)
    assert s["ci_method"] == "percentile_bootstrap" and s["ci_lo"] < s["mean"] < s["ci_hi"]
    big = rng.standard_normal(50_000)
    b = H.mean_ci(big)
    assert b["ci_method"] == "exact_bootstrap_se"
    se = float(np.sqrt(((big - big.mean()) ** 2).sum() / big.size ** 2))
    assert abs((b["ci_hi"] - b["ci_lo"]) - 2 * H.Z95 * se) < 1e-12
    one = H.mean_ci([3.0])
    assert one["ci_lo"] == one["ci_hi"] == 3.0


def test_atlas_schema_and_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        at = H.Atlas(d, arm="T", run_id="t0")
        at.add("scalar", value=1.5, layer=0, type="q_proj", bank_id="core32", flags={"a": 1})
        at.add_ci("sampled", [1.0, 2.0, 3.0], layer=1, type="q_proj", bank_id="haar32_s0")
        at.add("vec", vector_ref=at.vector("v0", np.arange(5)), n=5)
        two_d = at.vector("v1", np.zeros((4, 3)))
        at.add("col", vector_ref=f"{two_d}#2", n=4)
        at.close()
        rep = at.to_parquet()
        assert rep["n_rows"] == 4 and rep["n_vectors"] == 2
        rows = [json.loads(x) for x in open(at.path)]
        assert all(set(r) == set(H.ATLAS_COLUMNS) for r in rows)
        assert json.loads(rows[1]["flags"])["ci_method"] == "percentile_bootstrap"
        # exactly one of value / vector_ref, always
        for r in rows:
            assert (r["value"] is None) != (r["vector_ref"] is None)


def test_vector_ref_resolves_relative_to_the_out_dir():
    """The ref convention is `out_dir / ref` with the `vectors/` prefix INCLUDED. Every consumer
    (B2, the figures job, the validity audit) joins it that way; stripping the prefix is the bug
    this test exists to catch."""
    with tempfile.TemporaryDirectory() as d:
        at = H.Atlas(d, arm="T", run_id="t2")
        ref = at.vector("v", np.arange(6, dtype=float))
        at.close()
        assert ref.startswith("vectors/") and ref.endswith(".npy")
        assert (pathlib.Path(d) / ref).exists(), f"{ref} does not resolve under {d}"
        assert np.load(pathlib.Path(d) / ref).size == 6
        block = at.vector("w", np.zeros((3, 4)))
        col_ref = f"{block}#2"
        path, _, col = col_ref.partition("#")
        arr = np.load(pathlib.Path(d) / path)
        assert arr.ndim == 2 and int(col) < arr.shape[1]


def test_atlas_rejects_bad_rows():
    with tempfile.TemporaryDirectory() as d:
        at = H.Atlas(d, arm="T", run_id="t1")
        for bad in (dict(), dict(value=1.0, vector_ref="v"), dict(value=float("nan"))):
            try:
                at.add("x", **bad)
            except AssertionError:
                continue
            raise AssertionError(f"Atlas.add accepted a bad row: {bad}")
        at.vector("dup", [1.0])
        try:
            at.vector("dup", [2.0])
        except AssertionError:
            at.close()
            return
        raise AssertionError("Atlas.vector silently overwrote an existing vector")


def test_cache_dir_is_outside_the_synced_tree():
    """Fetched inputs must never land inside out/ — run_job.sh re-uploads everything under it, and
    the pipeline runner hands each step out/<step>, so a sibling-of-out rule would fail."""
    for out in ("/work/out", "/work/out/a2", "/tmp/whatever/out"):
        d = H.cache_dir(out, "banks")
        assert "hum_cache_banks" in str(d)
        assert not str(d).startswith(str(pathlib.Path(out).resolve()) + "/"), (out, d)
    # the same name resolves to the same directory from every step → fetched once per instance
    assert H.cache_dir("/work/out/a2", "banks") == H.cache_dir("/work/out/a3", "banks")


def test_type_flags_carry_the_caveats():
    assert H.type_flags("k_proj")["mqa_kernel_caveat"] is True
    assert H.type_flags("v_proj")["mqa_kernel_caveat"] is True
    assert H.type_flags("q_proj")["mqa_kernel_caveat"] is False
    assert H.type_flags("down_proj")["rank_deficient"] is True
    assert H.type_flags("down_proj")["n_kernel_dims"] == 16384 - 2048
    assert H.type_flags("gate_proj")["site"] == "gate_proj_in"


def test_all_offline_smokes_pass():
    for name, ev in S.run_all_offline().items():
        assert ev["ok"], f"{name}: {ev}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"{len(fns)} tests passed")
