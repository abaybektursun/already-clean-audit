"""REGRAIN weight-free dry-run suite (spec §3 / freeze checklist) — tiny tensors, CPU, no model.

Mandatory pre-freeze gates:
  * signed-path PARITY: signed m=1 on non-negative pre-activations ≡ rectified m=1 (and the two
    differ in general — the flag is not a no-op).
  * Grassmannian charts orthonormal at init AND after retraction; decode = γ·Σ Π_ℓ.
  * the planted-manifold DIAGNOSTIC reproduced weight-free: signed b=2 holds a planted circle in
    ~one block; rectified m=2 SHATTERS it across ≥3 (the cone restriction, BSF App. D/F).
  * MDL estimator on synthetic spectra (closed forms) + the infeasible branch.
  * rotated-error hook NORM-exactness; Haar rotation orthogonality.
  * A2 usage-matched unions; A1 canonicity-matched surrogates (closed form cross-check).
  * every adjudicator label reachable; guard rails RAISE.

Runs on the remote CPU box (pg r-family). Never on the laptop.
"""
import math

import pytest
import torch

from infra.pg import grain_bsf as GB
from infra.pg import regrain_core as RC
from infra.pg.ctrliface_core import principal_angle_overlap
from infra.pg.endow_atlas import normalized_canonicity


# ===========================================================================
# Signed path: parity on the non-negative cone, and non-triviality off it.
# ===========================================================================
def _nonneg_pre_sae(d, W, k, seed, signed):
    """An SAE whose pre-activations are non-negative for non-negative x: encoder weight ≥ 0,
    bias 0, b_dec 0 → pre = |E| x ≥ 0, so relu is the identity and signed ≡ rectified."""
    torch.manual_seed(seed)
    ae = GB.BlockTopKSAE(d, W, m=1, k_blk=k, signed=signed)
    with torch.no_grad():
        ae.encoder.weight.abs_()
        ae.encoder.bias.zero_()
        ae.b_dec.zero_()
        ae.threshold.fill_(0.05)
    return ae


def test_signed_parity_on_nonnegative_cone():
    d, W, k = 16, 64, 4
    rect = _nonneg_pre_sae(d, W, k, seed=11, signed=False)
    sign = _nonneg_pre_sae(d, W, k, seed=11, signed=True)
    x = torch.rand(32, d)                                  # x ≥ 0 → pre ≥ 0
    assert torch.equal(sign.encode(x, use_threshold=False), rect.encode(x, use_threshold=False))
    assert torch.equal(sign.encode(x, use_threshold=True), rect.encode(x, use_threshold=True))
    assert torch.equal(sign.decode(sign.encode(x)), rect.decode(rect.encode(x)))


def test_signed_differs_from_rectified_in_general():
    d, W, k = 16, 64, 4
    torch.manual_seed(3)
    rect = GB.BlockTopKSAE(d, W, m=1, k_blk=k, signed=False)
    torch.manual_seed(3)
    sign = GB.BlockTopKSAE(d, W, m=1, k_blk=k, signed=True)
    rect.threshold.fill_(0.05)
    sign.threshold.fill_(0.05)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(32, d, generator=g)                    # sign-symmetric → relu bites
    assert not torch.equal(sign.encode(x, use_threshold=False),
                           rect.encode(x, use_threshold=False))


def test_signed_flag_defaults_false_and_v1_behaviour_unchanged():
    """The v1 bank contract: a dict constructed without `signed` is rectified, and its encode is
    bit-identical to the pre-REGRAIN code path (relu ∘ encoder)."""
    d, W, k = 16, 64, 4
    torch.manual_seed(7)
    ae = GB.BlockTopKSAE(d, W, m=1, k_blk=k)
    assert ae.signed is False
    g = torch.Generator().manual_seed(9)
    x = torch.randn(8, d, generator=g)
    expected = torch.nn.functional.relu(ae.encoder(x - ae.b_dec))
    f = ae.encode(x, use_threshold=False)
    assert torch.equal(f[f > 0], expected[f > 0])
    assert float(f.min()) >= 0.0


# ===========================================================================
# Grassmannian charts.
# ===========================================================================
def test_block_quality_counts_negative_only_signed_latents_as_alive():
    """`f > 0` would call a signed latent that only ever fires NEGATIVE 'dead'. Descriptive, but it
    would mis-report every signed config in the factorial and the atlas."""
    d, W = 8, 4
    torch.manual_seed(1)
    ae = GB.BlockTopKSAE(d, W, m=1, k_blk=2, signed=True)
    with torch.no_grad():                       # latent 0 strictly negative, latent 1 positive
        ae.encoder.weight.zero_()
        ae.encoder.weight[0, 0] = -1.0
        ae.encoder.weight[1, 0] = 1.0
        ae.encoder.bias.zero_()
        ae.b_dec.zero_()
        ae.threshold.fill_(0.01)
    g = torch.Generator().manual_seed(2)
    X = torch.randn(64, d, generator=g)         # needs real variance: block_quality asserts den > 0
    X[:, 0] = torch.rand(64, generator=g) + 1.0  # → latent 0 code < 0, latent 1 code > 0, always
    q = GB.block_quality(ae, X, "cpu", n_tokens=32)
    # 2 of 4 latents fire (one only-negative, one only-positive); the other 2 never fire
    assert q["dead_latent_frac"] == pytest.approx(0.5), q
    f = ae.encode(X, use_threshold=True)
    assert float(f[:, 0].max()) < 0 and float(f[:, 1].min()) > 0
    assert bool((f[:, 2:] == 0).all())


def test_grassmann_charts_orthonormal_at_init_and_after_retraction():
    """Charts are stored fp32 (the encoder runs fp32), so the QR retraction — exact in fp64 —
    lands at fp32 orthonormality (~1e-7). 1e-8 would test the storage dtype, not the retraction."""
    d, W, b = 32, 24, 3
    ATOL = 1e-6
    ae = RC.GrassmannBSF(d, W, b, k_blk=2)
    assert ae.charts.dtype == torch.float32
    for blk in range(ae.n_blocks):
        U = ae.charts.detach()[:, blk * b:(blk + 1) * b]
        assert torch.allclose(U.T.double() @ U.double(),
                              torch.eye(b, dtype=torch.float64), atol=ATOL)
    with torch.no_grad():
        ae.charts.add_(0.3 * torch.randn_like(ae.charts))    # push off the manifold
        RC._batched_orthonormalize_(ae.charts.data, b)
    for blk in range(ae.n_blocks):
        U = ae.charts.detach()[:, blk * b:(blk + 1) * b]
        assert torch.allclose(U.T.double() @ U.double(),
                              torch.eye(b, dtype=torch.float64), atol=ATOL)


def test_batched_orthonormalize_chunking_is_exact():
    """Chunk boundaries must not change the result (the d=2048 memory guard)."""
    d, W, b = 16, 24, 2
    g = torch.Generator().manual_seed(1)
    A = torch.randn(d, W, generator=g)
    one = RC._batched_orthonormalize_(A.clone(), b, chunk=1000)
    many = RC._batched_orthonormalize_(A.clone(), b, chunk=2)
    assert torch.allclose(one, many, atol=1e-6)


def test_grassmann_decode_is_gamma_times_sum_of_projections():
    d, W, b = 32, 8, 2
    ae = RC.GrassmannBSF(d, W, b, k_blk=1)
    with torch.no_grad():
        ae.b_dec.copy_(torch.randn(d))
        ae.log_gamma.fill_(0.0)                      # γ = 1
    assert float(ae.gamma) == pytest.approx(1.0)
    x = torch.randn(4, d)
    z = (x - ae.b_dec) @ ae.charts                   # γ=1 ⇒ this is the code
    # one-block reconstruction must equal the orthogonal projection onto that chart
    keep = torch.zeros_like(z)
    keep[:, 0:b] = z[:, 0:b]
    rec = ae.decode(keep) - ae.b_dec
    U = ae.charts[:, 0:b]
    proj = (x - ae.b_dec) @ U @ U.T
    assert torch.allclose(rec, proj, atol=1e-5)


def test_grassmann_gamma_is_positive_and_log_parameterised():
    ae = RC.GrassmannBSF(16, 8, 2, k_blk=1)
    with torch.no_grad():
        ae.log_gamma.fill_(-3.0)
    assert float(ae.gamma) == pytest.approx(math.exp(-3.0))
    assert float(ae.gamma) > 0


def test_grassmann_ignores_the_threshold_at_inference():
    """A Grassmannian BSF has NO inference threshold: Π_ℓ is a hard per-sample count at train AND
    inference. The house BatchTopK threshold (smallest selected block norm) is only a valid global
    cutoff under BATCH-level selection; under per-sample selection it is the smallest k-th-largest
    norm across rows, and a thresholded forward admits far more than ℓ blocks. Measured on the
    first factorial run: mean_block_l0 = 3365.7 against k_blk = 96, held-out FVU 8.75."""
    d, W, b, k = 32, 16, 2, 3
    ae = RC.GrassmannBSF(d, W, b, k_blk=k)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(12, d, generator=g)
    for th in (-1.0, 0.0, 1e-6, 1e6):        # any threshold, including a pathological one
        with torch.no_grad():
            ae.threshold.fill_(th)
        for ut in (True, False):
            f = ae.encode(x, use_threshold=ut)
            assert torch.equal((ae.block_norms(f) > 0).sum(1), torch.full((12,), k)), (th, ut)


def test_grassmann_trainer_leaves_a_positive_threshold_sentinel():
    """`load_bank_any` asserts a positive threshold (it guards the BatchTopK banks). The Grassmann
    trainer must therefore leave a positive sentinel even though `encode` never reads it."""
    ae, cfg = RC.train_grassmann_bsf(torch.randn(300, 16), b=2, seed=0, device="cpu", dict_size=16,
                                     r_active=4, steps=40, lr=1e-3, batch=64, warmup=5)
    assert float(ae.threshold) > 0 and cfg["threshold"] > 0


def test_grassmann_selection_is_per_sample_not_batch_flattened():
    """Paper App. D: 'Selection is per-sample.' Every row must activate EXACTLY k_blk blocks —
    a batch-flattened top-(k·batch) would let one row take slots from another."""
    d, W, b, k = 24, 12, 2, 2
    ae = RC.GrassmannBSF(d, W, b, k_blk=k)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(9, d, generator=g)
    x[0] *= 100.0                                    # a row that would hog batch-level slots
    f = ae.encode(x, use_threshold=False)
    active_per_row = (ae.block_norms(f) > 0).sum(1)
    assert torch.equal(active_per_row, torch.full((9,), k))


# ===========================================================================
# The planted-manifold diagnostic: signed holds the circle, rectified shatters it.
# ===========================================================================
def _train_circle(X, signed, freeze_b_dec, steps=1400):
    return GB.train_block_sae(X, m=2, seed=0, device="cpu", dict_size=32, r_active=2,
                              steps=steps, lr=3e-3, batch=256, warmup=100,
                              signed=signed, freeze_b_dec=freeze_b_dec)[0]


def test_planted_circle_signed_holds_rectified_shatters_at_a_pinned_apex():
    """The BSF cone argument (App. D §2.3), isolated: with the code's apex PINNED at the data's
    geometric median, a rectified block's non-negative cone spans <180° of the planted plane, so
    a full circle cannot live in one block. A signed block spans the whole plane."""
    d, n = 48, 4000
    X, U = RC.gen_planted_circle(d, n, seed=17, noise=0.01)
    sgn = _train_circle(X, signed=True, freeze_b_dec=True)
    rct = _train_circle(X, signed=False, freeze_b_dec=True)

    assert RC.b_dec_offset_ratio(sgn, X) < 0.2 and RC.b_dec_offset_ratio(rct, X) < 0.2
    n_sgn = RC.blocks_covering_usage(sgn, X)
    n_rct = RC.blocks_covering_usage(rct, X)
    assert n_sgn <= 2, f"signed BSF should hold the circle in ~1 block, used {n_sgn}"
    assert n_rct >= 3, f"rectified BSF must shatter the circle (cone restriction), used {n_rct}"
    assert n_rct > n_sgn

    best = max(principal_angle_overlap(GB.block_subspace(sgn.decoder.weight.detach(), 2, b), U)[0]
               for b in range(sgn.n_blocks))
    assert best > 0.9, f"a signed block should span the planted plane; best overlap {best:.3f}"


def test_learned_b_dec_defeats_the_cone_restriction_for_one_manifold():
    """MEASURED, and load-bearing for H_sign-fix: with b_dec FREE the trainer moves the apex off
    the data, widening the rectified cone until ONE block covers the whole circle. So the paper's
    cone argument does not by itself imply that rectification must shatter a curved factor in the
    real instrument (where b_dec is learned). Recorded, not asserted away."""
    d, n = 48, 4000
    X, _U = RC.gen_planted_circle(d, n, seed=17, noise=0.01)
    rct_free = _train_circle(X, signed=False, freeze_b_dec=False)
    rct_pinned = _train_circle(X, signed=False, freeze_b_dec=True)

    off_free = RC.b_dec_offset_ratio(rct_free, X)
    off_pinned = RC.b_dec_offset_ratio(rct_pinned, X)
    assert off_free > off_pinned, (off_free, off_pinned)
    assert RC.blocks_covering_usage(rct_free, X) < RC.blocks_covering_usage(rct_pinned, X)


def test_freeze_b_dec_actually_pins_the_parameter():
    d = 16
    g = torch.Generator().manual_seed(2)
    X = torch.randn(300, d, generator=g)
    ae = GB.train_block_sae(X, m=2, seed=1, device="cpu", dict_size=16, r_active=2, steps=40,
                            lr=1e-2, batch=64, warmup=5, freeze_b_dec=True)[0]
    assert ae.b_dec.requires_grad is False
    free = GB.train_block_sae(X, m=2, seed=1, device="cpu", dict_size=16, r_active=2, steps=40,
                              lr=1e-2, batch=64, warmup=5, freeze_b_dec=False)[0]
    assert free.b_dec.requires_grad is True
    assert not torch.allclose(ae.b_dec, free.b_dec)


# ===========================================================================
# MDL.
# ===========================================================================
def test_rate_bits_closed_forms_smooth_and_clipped():
    lam = torch.tensor([4.0, 4.0], dtype=torch.float64)
    assert RC.rate_bits(lam, 1.0, smooth=False) == pytest.approx(2 * 0.5 * math.log2(4.0))
    assert RC.rate_bits(lam, 1.0, smooth=True) == pytest.approx(2 * 0.5 * math.log2(5.0))
    # clipped drops sub-δ eigenvalues entirely; smooth keeps a small positive contribution
    tiny = torch.tensor([1e-9], dtype=torch.float64)
    assert RC.rate_bits(tiny, 1.0, smooth=False) == 0.0
    assert 0.0 < RC.rate_bits(tiny, 1.0, smooth=True) < 1e-8
    with pytest.raises(AssertionError):
        RC.rate_bits(lam, 0.0)


def test_rate_bits_monotone_decreasing_in_delta():
    lam = torch.tensor([4.0, 2.0, 1.0, 0.25], dtype=torch.float64)
    bits = [RC.rate_bits(lam, d) for d in (0.1, 0.5, 1.0, 2.0, 4.0)]
    assert all(bits[i] > bits[i + 1] for i in range(len(bits) - 1)), bits


def test_log2_binom():
    assert abs(RC.log2_binom(5, 2) - math.log2(10)) < 1e-9
    assert RC.log2_binom(7, 0) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(AssertionError):
        RC.log2_binom(3, 4)


def test_dictionary_param_counts_match_the_paper():
    # Grassmannian: G·b·(d−b) free params of G points on Gr(b, d); free decoder: G·b·d
    assert RC.grassmann_free_params(4096, 4, 2048) == 4096 * 4 * (2048 - 4)
    assert RC.free_decoder_params(4096, 4, 2048) == 4096 * 4 * 2048
    assert RC.grassmann_free_params(10, 1, 5) < RC.free_decoder_params(10, 1, 5)


def _mdl_kw(G=1024, b=2, ell=8, d=64, N=1_000_000, sigma=1.0, lam=0.01):
    return dict(n_blocks=G, k_blk=ell, n_tokens=N,
                block_fire_rates=torch.full((G,), ell / G, dtype=torch.float64),
                block_code_eigs=torch.full((G, b), sigma, dtype=torch.float64),
                residual_eigs=torch.full((d,), lam, dtype=torch.float64),
                dict_free_params=RC.grassmann_free_params(G, b, d))


def test_mdl_matches_eq6_term_by_term():
    G, b, ell, d, N, sigma, lam, delta = 1024, 2, 8, 64, 1_000_000, 1.0, 0.01, 0.05
    out = RC.mdl_bits_per_token(delta=delta, **_mdl_kw(G, b, ell, d, N, sigma, lam))
    assert out["support"] == pytest.approx(RC.log2_binom(G, ell))
    # code: Σ_g p_g Σ_j ½log2(1+σ²/δ) = ℓ · b · ½log2(1+σ²/δ)
    assert out["code"] == pytest.approx(ell * b * 0.5 * math.log2(1 + sigma / delta))
    # residual: d · ½log2(1+λ/δ)
    assert out["residual"] == pytest.approx(d * 0.5 * math.log2(1 + lam / delta))
    assert out["dictionary"] == pytest.approx(
        0.5 * RC.grassmann_free_params(G, b, d) * math.log2(N) / N)
    assert out["expected_active_blocks"] == pytest.approx(ell)
    assert out["total"] == pytest.approx(out["support"] + out["code"] + out["residual"]
                                         + out["dictionary"])


def test_mdl_total_decreases_with_looser_distortion():
    kw = _mdl_kw()
    assert RC.mdl_bits_per_token(delta=0.30, **kw)["total"] \
        < RC.mdl_bits_per_token(delta=0.05, **kw)["total"]


def test_mdl_residual_is_transmitted_not_a_budget():
    """Eq. 5 has FOUR terms: a config with a huge residual is still describable — it just pays
    more residual bits. It must never return +inf (the estimator I first wrote did)."""
    kw = _mdl_kw(lam=1e6)
    out = RC.mdl_bits_per_token(delta=0.01, **kw)
    assert math.isfinite(out["total"]) and out["residual"] > out["code"]


def test_mdl_spectra_matches_a_dense_reference():
    """The chunked collector must equal the obvious (memory-hungry) computation."""
    d, W, m = 12, 8, 2
    torch.manual_seed(4)
    ae = GB.BlockTopKSAE(d, W, m=m, k_blk=2)
    ae.threshold.fill_(0.02)
    g = torch.Generator().manual_seed(5)
    X = torch.randn(400, d, generator=g)
    spec = RC.mdl_spectra(ae, X, micro=64)

    with torch.no_grad():
        f = ae.encode(X, use_threshold=True)
        r = (X - ae.decode(f)).double()
    zb = f.double().reshape(-1, W // m, m)
    act = zb.norm(dim=2) > 0
    assert torch.allclose(spec["block_fire_rates"], act.double().mean(0), atol=1e-9)
    gram = torch.einsum("ngi,ngj->gij", zb, zb) / act.double().sum(0).clamp_min(1).reshape(-1, 1, 1)
    assert torch.allclose(spec["block_code_eigs"], torch.linalg.eigvalsh(gram).clamp_min(0.0),
                          atol=1e-8)
    rc = r - r.mean(0)
    ref = torch.linalg.eigvalsh(rc.T @ rc / X.shape[0]).clamp_min(0.0)
    assert torch.allclose(spec["residual_eigs"], ref, atol=1e-7)
    assert spec["var_ref"] == pytest.approx(float(X.double().var(0, unbiased=False).mean()),
                                            abs=1e-9)
    assert spec["n_tokens"] == 400


def test_mdl_spectra_feeds_mdl_bits_per_token():
    d, W, m = 12, 8, 2
    torch.manual_seed(6)
    ae = GB.BlockTopKSAE(d, W, m=m, k_blk=2)
    ae.threshold.fill_(0.02)
    spec = RC.mdl_spectra(ae, torch.randn(300, d), micro=64)
    out = RC.mdl_bits_per_token(
        n_blocks=ae.n_blocks, k_blk=ae.k_blk, n_tokens=spec["n_tokens"],
        block_fire_rates=spec["block_fire_rates"], block_code_eigs=spec["block_code_eigs"],
        residual_eigs=spec["residual_eigs"], delta=0.10 * spec["var_ref"],
        dict_free_params=RC.free_decoder_params(ae.n_blocks, ae.m, d))
    assert math.isfinite(out["total"]) and out["total"] > 0


def test_unit_decoder_hoist_does_not_change_top_aligned_atoms():
    d, W, k = 32, 40, 3
    g = torch.Generator().manual_seed(7)
    Wd = torch.randn(d, W, generator=g, dtype=torch.float64)
    V = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]
    assert RC.top_aligned_atoms(Wd, V, 6) == \
        RC.top_aligned_atoms(RC.unit_decoder(Wd), V, 6, already_unit=True)


def test_mdl_wider_blocks_trade_support_against_code():
    """The paper's core MDL mechanism: at fixed active rank ℓ·b, growing b shrinks the support
    term (fewer, larger blocks to name) while the code term stays put — so the optimum is
    interior. Verified here at the estimator level."""
    d, N, delta = 64, 10 ** 6, 0.05
    thin = RC.mdl_bits_per_token(delta=delta, **_mdl_kw(G=4096, b=1, ell=16, d=d, N=N))
    wide = RC.mdl_bits_per_token(delta=delta, **_mdl_kw(G=2048, b=2, ell=8, d=d, N=N))
    assert wide["support"] < thin["support"]
    assert wide["code"] == pytest.approx(thin["code"])          # ℓ·b = 16 both ways


# ===========================================================================
# Rotated error + corruptions.
# ===========================================================================
def test_haar_rotation_is_orthogonal_and_seeded():
    Q = RC.haar_rotation(24, seed=5, dtype=torch.float64)
    assert torch.allclose(Q.T @ Q, torch.eye(24, dtype=torch.float64), atol=1e-10)
    assert torch.equal(Q, RC.haar_rotation(24, seed=5, dtype=torch.float64))
    assert not torch.equal(Q, RC.haar_rotation(24, seed=6, dtype=torch.float64))


def test_rotate_error_preserves_per_row_error_norm_exactly():
    d = 32
    g = torch.Generator().manual_seed(2)
    x = torch.randn(16, d, generator=g, dtype=torch.float64)
    x_hat = x + 0.3 * torch.randn(16, d, generator=g, dtype=torch.float64)
    Q = RC.haar_rotation(d, seed=1, dtype=torch.float64)
    out = RC.rotate_error(x, x_hat, Q)
    assert torch.allclose((out - x).norm(dim=1), (x_hat - x).norm(dim=1), atol=1e-10)
    # direction destroyed: the rotated error is not the original error
    assert not torch.allclose(out, x_hat, atol=1e-6)


def test_rotate_error_alpha_scales_the_error_norm():
    d = 32
    g = torch.Generator().manual_seed(11)
    x = torch.randn(8, d, generator=g, dtype=torch.float64)
    x_hat = x + 0.3 * torch.randn(8, d, generator=g, dtype=torch.float64)
    Q = RC.haar_rotation(d, seed=2, dtype=torch.float64)
    base = (RC.rotate_error(x, x_hat, Q) - x).norm(dim=1)
    for al in (0.5, 2.0, 4.0):
        got = (RC.rotate_error(x, x_hat, Q, alpha=al) - x).norm(dim=1)
        assert torch.allclose(got, al * base, atol=1e-10)
    assert torch.allclose(RC.rotate_error(x, x_hat, Q, alpha=0.0), x)


def test_x_scale_default_reproduces_the_v1_recipe_bitwise():
    X = torch.randn(300, 16)
    a1 = GB.train_block_sae(X, m=2, seed=3, device="cpu", dict_size=32, r_active=4, steps=60,
                            lr=1e-3, batch=64, warmup=10)[0]
    a2 = GB.train_block_sae(X, m=2, seed=3, device="cpu", dict_size=32, r_active=4, steps=60,
                            lr=1e-3, batch=64, warmup=10, x_scale=1.0)[0]
    assert torch.equal(a1.decoder.weight, a2.decoder.weight)


def test_x_scale_changes_training_and_is_banked():
    X = torch.randn(300, 16)
    kw = dict(m=2, seed=3, device="cpu", dict_size=32, r_active=4, steps=60, lr=1e-3,
              batch=64, warmup=10)
    a1, c1 = GB.train_block_sae(X, x_scale=1.0, **kw)
    a2, c2 = GB.train_block_sae(X, x_scale=0.4, **kw)
    assert c1["x_scale"] == 1.0 and c2["x_scale"] == 0.4
    assert not torch.equal(a1.decoder.weight, a2.decoder.weight)


def test_mdl_spectra_x_scale_rescales_var_ref_quadratically():
    d = 12
    torch.manual_seed(8)
    ae = GB.BlockTopKSAE(d, 8, m=2, k_blk=2)
    ae.threshold.fill_(0.02)
    X = torch.randn(200, d)
    s1 = RC.mdl_spectra(ae, X, micro=64, x_scale=1.0)
    s2 = RC.mdl_spectra(ae, X, micro=64, x_scale=3.0)
    assert s2["var_ref"] == pytest.approx(9.0 * s1["var_ref"], rel=1e-6)


def test_save_bank_any_raises_on_x_scale_mismatch(tmp_path):
    d, W = 8, 8
    ae, cfg = GB.train_block_sae(torch.randn(100, d), m=1, seed=0, device="cpu", dict_size=W,
                                 r_active=2, steps=30, lr=1e-3, batch=32, warmup=5, x_scale=0.5)
    ae.threshold.fill_(0.01)
    with pytest.raises(AssertionError):
        RC.save_bank_any(ae, cfg, {}, tmp_path / "x.pt", "sha", x_scale=0.9)
    RC.save_bank_any(ae, cfg, {}, tmp_path / "ok.pt", "sha", x_scale=0.5)
    back, cfg2, _q, _s = RC.load_bank_any(tmp_path / "ok.pt")
    assert isinstance(back, RC.ScaledAE) and back.x_scale == 0.5


def test_r2_of_matches_definition():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(64, 8, generator=g, dtype=torch.float64)
    mu = x.mean(0)
    assert RC.r2_of(x, x, mu) == pytest.approx(1.0)
    assert RC.r2_of(x, mu.expand_as(x), mu) == pytest.approx(0.0, abs=1e-12)


def test_quantize_step_matches_the_paper_and_bounds_error_by_half_a_step():
    """App. G.1: x_q = Δ·round(x/Δ). Error ≤ Δ/2 everywhere; no clipping."""
    g = torch.Generator().manual_seed(6)
    x = torch.randn(200, 5, generator=g, dtype=torch.float64)
    for step in (0.05, 0.5, 2.0):
        xq = RC.quantize_step(x, step)
        assert ((xq - x).abs() <= step / 2 + 1e-9).all()
        assert torch.allclose(xq / step, torch.round(xq / step))    # lands on the lattice
    assert RC.r2_of(x, RC.quantize_step(x, 0.01), x.mean(0)) \
        > RC.r2_of(x, RC.quantize_step(x, 1.0), x.mean(0))
    with pytest.raises(AssertionError):
        RC.quantize_step(x, 0.0)


# ===========================================================================
# A2 — usage-matched unions.
# ===========================================================================
def test_usage_strata_partitions_active_atoms():
    scale = torch.rand(200).double() + 0.01
    active = torch.zeros(200, dtype=torch.bool)
    active[:150] = True
    st = RC.usage_strata(scale, active, n_strata=10)
    assert int(st[active].min()) >= 0 and int(st[active].max()) <= 9
    assert (st[~active] == -1).all()
    counts = torch.bincount(st[active], minlength=10)
    assert int(counts.min()) >= 10, counts        # deciles of 150 → ~15 each


def test_usage_matched_union_matches_stratum_and_excludes_bstar():
    scale = torch.linspace(0.01, 1.0, 200).double()
    active = torch.ones(200, dtype=torch.bool)
    st = RC.usage_strata(scale, active, n_strata=10)
    bstar = [3, 55, 120, 190]
    picked = RC.usage_matched_union(bstar, st, seed=1)
    assert len(set(picked)) == len(picked)
    assert not (set(picked) & set(bstar))
    for a, p in zip(bstar, picked):
        assert int(st[a]) == int(st[p])


def test_usage_matched_union_fails_loud_on_exhausted_stratum():
    scale = torch.ones(4).double()
    active = torch.ones(4, dtype=torch.bool)
    st = torch.zeros(4, dtype=torch.long)          # all one stratum
    with pytest.raises(AssertionError):
        RC.usage_matched_union([0, 1, 2, 3, 0], st, seed=1)   # 5 picks from a 4-atom stratum


def test_usage_matched_union_rejects_inactive_bstar_atom():
    st = torch.tensor([-1, 0, 0, 0], dtype=torch.long)
    with pytest.raises(AssertionError):
        RC.usage_matched_union([0], st, seed=1)


def test_uniform_random_union_size_and_exclusion():
    u = RC.uniform_random_union(list(range(100)), 8, seed=2, exclude=[0, 1, 2])
    assert len(u) == 8 and len(set(u)) == 8
    assert not (set(u) & {0, 1, 2})
    with pytest.raises(AssertionError):
        RC.uniform_random_union([1, 2], 8, seed=2)


# ===========================================================================
# A1 — canonicity-matched surrogates.
# ===========================================================================
def test_surrogate_hits_target_canonicity_and_matches_closed_form():
    d, k = 128, 8
    g = torch.Generator().manual_seed(3)
    V = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]
    for target in (0.20, 0.55, 0.80):
        Vp, c = RC.surrogate_at_canonicity(V, target, d, seed=1)
        assert abs(c - target) < 5e-3, (target, c)
        o, _ = principal_angle_overlap(V, Vp)
        assert abs(normalized_canonicity(o, k, d) - c) < 1e-9
        # closed form: overlap = cos²θ ⇒ c = (cos²θ − k/d)/(1 − k/d)
        floor = k / d
        assert abs(o - (target * (1 - floor) + floor)) < 1e-2


def test_surrogate_canonicity_is_monotone_in_target():
    d, k = 64, 4
    g = torch.Generator().manual_seed(8)
    V = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]
    cs = [RC.surrogate_at_canonicity(V, t, d, seed=2)[1] for t in (0.1, 0.4, 0.7, 0.95)]
    assert all(cs[i] < cs[i + 1] for i in range(len(cs) - 1)), cs


def test_surrogate_rejects_impossible_target():
    d, k = 32, 4
    V = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))[0]
    with pytest.raises(AssertionError):
        RC.surrogate_at_canonicity(V, -0.5, d, seed=1)     # below the θ=π/2 floor


def test_canonicity_matched_jaccard_null_is_monotone_in_canonicity():
    """The property Arm D's rule depends on: the surrogate Jaccard band RISES with the matched
    canonicity, so comparing the observed J against the band AT THE OBSERVED canonicity is the
    like-for-like test. (It does NOT saturate at 1 for a generic dictionary even at c≈1: with
    closely-spaced alignment scores a few-degree subspace tilt reshuffles the top-n. That is a
    property of the readout, and it is exactly what the null is there to absorb.)"""
    d, W, k = 64, 96, 4
    g = torch.Generator().manual_seed(4)
    Wd = torch.randn(d, W, generator=g, dtype=torch.float64)
    V = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]
    means = []
    for c in (0.10, 0.50, 0.90, 0.995):
        vals, cs = RC.canonicity_matched_jaccard_null(Wd, V, c, d, top_n=8, n_draws=12, seed=1)
        assert all(abs(x - c) < 5e-3 for x in cs), cs
        assert 0.0 <= float(vals.min()) and float(vals.max()) <= 1.0
        means.append(float(vals.mean()))
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1)), means
    assert means[0] < 0.3 and means[-1] > 0.5


def test_canonicity_matched_jaccard_null_is_exactly_one_at_zero_tilt():
    """Sanity anchor: a surrogate built at θ→0 IS V, so its atom set is V's → Jaccard 1.0."""
    d, W, k = 64, 96, 4
    g = torch.Generator().manual_seed(4)
    Wd = torch.randn(d, W, generator=g, dtype=torch.float64)
    V = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]
    base = RC.top_aligned_atoms(Wd, V, 8)
    Vp, _c = RC.surrogate_at_canonicity(V, 1.0, d, seed=1, tol=1e-9, max_iter=80)
    assert GB.jaccard(base, RC.top_aligned_atoms(Wd, Vp, 8)) == 1.0


def test_top_aligned_atoms_agrees_with_block_alignment_at_m1():
    d, W, k = 32, 40, 3
    g = torch.Generator().manual_seed(6)
    Wd = torch.randn(d, W, generator=g, dtype=torch.float64)
    Wd = Wd / Wd.norm(dim=0, keepdim=True)
    V = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]
    mine = RC.top_aligned_atoms(Wd, V, 6)
    theirs = torch.topk(GB.block_alignment(Wd, 1, V), 6).indices.tolist()
    assert mine == theirs


# ===========================================================================
# Adjudicators — every label reachable, guards raise.
# ===========================================================================
_C_OK = dict(effect_bstar=0.60, effect_full=0.50, effect_em=0.05,
             effect_rand_uniform_mean=0.05, effect_rand_usage_mean=0.08,
             gap_uniform_cilo=0.20, gap_usage_cilo=0.15, delta_C=0.05, rho=0.5, bh_pass=True,
             dppl_bstar=3.0, dppl_full=6.0, dppl_rand_band_hi=1.0)


def test_adjudicate_atom_handle_all_labels_reachable():
    assert RC.adjudicate_atom_handle(**_C_OK) == "H_atom-handle"
    # over-damaging off-target → nonselective
    assert RC.adjudicate_atom_handle(**{**_C_OK, "dppl_bstar": 20.0}) \
        == "H_atom-handle-nonselective"
    # usage-matched null explains it away (uniform still passes) → not compact
    assert RC.adjudicate_atom_handle(**{**_C_OK, "gap_usage_cilo": 0.0,
                                        "effect_rand_usage_mean": 0.40}) == "H_capacity-artifact"
    assert RC.adjudicate_atom_handle(**{**_C_OK, "gap_usage_cilo": 0.0}) == "H_no-atom-handle"
    # em-span reproduces B* → not compact
    assert RC.adjudicate_atom_handle(**{**_C_OK, "effect_em": 0.60}) == "H_no-atom-handle"
    # BH fails → not compact
    assert RC.adjudicate_atom_handle(**{**_C_OK, "bh_pass": False}) == "H_no-atom-handle"
    assert set(RC.C_LABELS) == {"H_atom-handle", "H_atom-handle-nonselective",
                                "H_capacity-artifact", "H_no-atom-handle"}


def test_adjudicate_atom_handle_requires_both_nulls():
    """A2's whole point: passing the uniform null is not enough."""
    assert RC.adjudicate_atom_handle(**{**_C_OK, "gap_usage_cilo": 0.04}) == "H_no-atom-handle"
    assert RC.adjudicate_atom_handle(**{**_C_OK, "gap_uniform_cilo": 0.04}) == "H_no-atom-handle"


def test_adjudicate_atom_handle_guard_raises_without_dynamic_range():
    with pytest.raises(AssertionError):
        RC.adjudicate_atom_handle(**{**_C_OK, "effect_full": 0.0})


def test_adjudicate_draws_all_labels_reachable():
    assert RC.adjudicate_draws(jaccard_cilo=0.9, permutation_p95=0.0, surrogate_p95=0.6) \
        == "H_basis-ambiguity"
    assert RC.adjudicate_draws(jaccard_cilo=0.5, permutation_p95=0.0, surrogate_p95=0.6) \
        == "H_subspace-explains"
    assert RC.adjudicate_draws(jaccard_cilo=0.0, permutation_p95=0.0, surrogate_p95=0.6) \
        == "H_genuinely-family"
    assert set(RC.D_LABELS) == {"H_basis-ambiguity", "H_subspace-explains", "H_genuinely-family"}


def test_adjudicate_draws_guard_raises_on_inverted_nulls():
    with pytest.raises(AssertionError):
        RC.adjudicate_draws(jaccard_cilo=0.5, permutation_p95=0.7, surrogate_p95=0.2)


def test_adjudicate_draws_boundary_is_strict():
    """J exactly at the surrogate band is NOT basis-ambiguity (the rule must not round up)."""
    assert RC.adjudicate_draws(jaccard_cilo=0.6, permutation_p95=0.0, surrogate_p95=0.6) \
        == "H_subspace-explains"


def test_adjudicate_sign_all_labels_reachable():
    assert RC.adjudicate_sign(signed_m4_pass=True, rectified_m4_pass=False,
                              any_signed_small_b_pass=True) == "H_sign-fix"
    assert RC.adjudicate_sign(signed_m4_pass=False, rectified_m4_pass=False,
                              any_signed_small_b_pass=False) == "H_class-boundary"
    assert RC.adjudicate_sign(signed_m4_pass=True, rectified_m4_pass=True,
                              any_signed_small_b_pass=True) == "H_grain-cost"
    assert set(RC.I_LABELS) == {"H_sign-fix", "H_class-boundary", "H_grain-cost"}


def test_adjudicate_stream_all_labels_reachable():
    assert RC.adjudicate_stream(pure_text_passes_text_gate=True, pure_text_fvu=0.09,
                                mixed_fvu=0.07) == "H_mixture"
    assert RC.adjudicate_stream(pure_text_passes_text_gate=False, pure_text_fvu=0.071,
                                mixed_fvu=0.070) == "H_prior-mismatch"
    assert RC.adjudicate_stream(pure_text_passes_text_gate=False, pure_text_fvu=0.30,
                                mixed_fvu=0.070) == "H_stream-null"
    assert set(RC.H_LABELS) == {"H_mixture", "H_prior-mismatch", "H_stream-null"}


def test_adjudicate_stream_a7_better_reconstruction_is_prior_mismatch_not_stream_null():
    """A7: a pure-text dictionary that reconstructs text MUCH BETTER than the mixed one and still
    breaks the text splice is the STRONGEST evidence for the language-mismatch conjecture. A
    two-sided |Δfvu| ≤ tol band would have labelled it 'undertrained' — the exact opposite."""
    assert RC.adjudicate_stream(pure_text_passes_text_gate=False, pure_text_fvu=0.030,
                                mixed_fvu=0.070) == "H_prior-mismatch"
    # and genuinely worse reconstruction still reaches H_stream-null
    assert RC.adjudicate_stream(pure_text_passes_text_gate=False, pure_text_fvu=0.120,
                                mixed_fvu=0.070) == "H_stream-null"
    # the boundary is one-sided at mixed + tol
    assert RC.adjudicate_stream(pure_text_passes_text_gate=False, pure_text_fvu=0.0899,
                                mixed_fvu=0.070, fvu_tol=0.02) == "H_prior-mismatch"
    assert RC.adjudicate_stream(pure_text_passes_text_gate=False, pure_text_fvu=0.0901,
                                mixed_fvu=0.070, fvu_tol=0.02) == "H_stream-null"


# ===========================================================================
# Structure statistics.
# ===========================================================================
def test_stable_rank_known_cases():
    I = torch.eye(10, dtype=torch.float64)
    assert RC.stable_rank(I) == pytest.approx(10.0)
    rank1 = torch.ones(10, 4, dtype=torch.float64)
    assert RC.stable_rank(rank1) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(AssertionError):
        RC.stable_rank(torch.zeros(4, 3, dtype=torch.float64))


def test_stable_rank_haar_band_is_near_n_cols_and_not_degenerate():
    band = RC.stable_rank_haar_band(512, 16, n_draws=8, seed=1)
    assert 12.0 < float(band.mean()) < 16.0            # unit columns, not an orthonormal basis
    assert float(band.std()) > 0.0


# ===========================================================================
# √d scaling + unified bank IO.
# ===========================================================================
def test_sqrt_d_scale_normalizes_mean_norm():
    d = 64
    g = torch.Generator().manual_seed(1)
    X = 7.5 * torch.randn(1000, d, generator=g)
    s = RC.sqrt_d_scale(X, d)
    assert abs(float((s * X).norm(dim=1).mean()) - math.sqrt(d)) < 1e-3 * math.sqrt(d)


def test_scaled_ae_is_transparent_in_original_units():
    d, W = 16, 32
    torch.manual_seed(2)
    inner = GB.BlockTopKSAE(d, W, m=1, k_blk=4)
    inner.threshold.fill_(0.05)
    s = 3.0
    wrapped = RC.ScaledAE(inner, s)
    x = torch.randn(8, d)
    assert torch.allclose(wrapped.encode(x), inner.encode(s * x))
    f = wrapped.encode(x)
    assert torch.allclose(wrapped.decode(f), inner.decode(f) / s)
    assert wrapped.m == inner.m and wrapped.n_blocks == inner.n_blocks
    # decoder DIRECTIONS untouched by scaling → geometry is invariant
    assert torch.equal(RC.decoder_matrix(wrapped), RC.decoder_matrix(inner))


def test_bank_roundtrip_blocktopk_signed_and_scaled(tmp_path):
    d, W = 16, 32
    # the trainer records x_scale in cfg; save_bank_any RAISES if the arg disagrees, so the bank
    # can never be written at a scale the featurizer was not trained at.
    ae, cfg = GB.train_block_sae(torch.randn(500, d), m=2, seed=0, device="cpu", dict_size=W,
                                 r_active=4, steps=60, lr=1e-3, batch=64, warmup=10, signed=True,
                                 x_scale=2.5)
    ae.threshold.fill_(0.01)
    p = tmp_path / "b.pt"
    RC.save_bank_any(ae, cfg, {"fvu": 0.1}, p, "sha", x_scale=2.5)
    back, cfg2, _q, sha = RC.load_bank_any(p)
    assert sha == "sha" and cfg2["signed"] is True and cfg2["x_scale"] == 2.5
    assert isinstance(back, RC.ScaledAE) and back.x_scale == 2.5
    x = torch.randn(4, d)
    assert torch.allclose(back.encode(x), ae.encode(2.5 * x))
    with pytest.raises(AssertionError):
        RC.save_bank_any(ae, cfg, {}, tmp_path / "bad.pt", "sha", x_scale=1.0)


def test_bank_roundtrip_grassmann(tmp_path):
    d, W, b = 24, 12, 3
    ae, cfg = RC.train_grassmann_bsf(torch.randn(400, d), b=b, seed=0, device="cpu", dict_size=W,
                                     r_active=3, steps=60, lr=1e-3, batch=64, warmup=10)
    assert cfg["kind"] == "grassmann" and cfg["signed"] is True
    ae.threshold.fill_(0.01)
    p = tmp_path / "g.pt"
    RC.save_bank_any(ae, cfg, {"fvu": 0.2}, p, "sha2")
    back, cfg2, _q, _s = RC.load_bank_any(p)
    assert isinstance(back, RC.GrassmannBSF) and cfg2["b"] == b
    x = torch.randn(4, d)
    assert torch.allclose(back.encode(x), ae.encode(x), atol=1e-6)
    assert RC.decoder_matrix(back).shape == (d, W)


def test_grain_bsf_load_dict_bank_rejects_grassmann(tmp_path):
    d, W, b = 24, 12, 3
    ae, cfg = RC.train_grassmann_bsf(torch.randn(200, d), b=b, seed=0, device="cpu", dict_size=W,
                                     r_active=3, steps=40, lr=1e-3, batch=64, warmup=10)
    ae.threshold.fill_(0.01)
    p = tmp_path / "g.pt"
    RC.save_bank_any(ae, cfg, {}, p, "sha")
    with pytest.raises(AssertionError):
        GB.load_dict_bank(p)


# ===========================================================================
# Trainer parity: the signed flag must not perturb the rectified code path.
# ===========================================================================
def test_train_block_sae_rectified_unchanged_by_the_signed_refactor():
    """The v1 recipe must reproduce bit-for-bit: same seed, same data, signed=False."""
    X = torch.randn(400, 16)
    a1, c1 = GB.train_block_sae(X, m=2, seed=5, device="cpu", dict_size=32, r_active=4,
                                steps=80, lr=1e-3, batch=64, warmup=10)
    a2, c2 = GB.train_block_sae(X, m=2, seed=5, device="cpu", dict_size=32, r_active=4,
                                steps=80, lr=1e-3, batch=64, warmup=10, signed=False)
    assert torch.equal(a1.decoder.weight, a2.decoder.weight)
    assert c1["final_loss"] == c2["final_loss"] and c1["signed"] is False
