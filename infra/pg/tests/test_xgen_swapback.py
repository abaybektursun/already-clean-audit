"""§9.1 WEIGHT-FREE DRY-RUN GATE for the XGEN cross-generation seal.

Validates the PURE logic of the NEW instruments (infra/pg/swap_back.py) on tiny tensors
+ planted arrays — NO model, NO datasets, NO GPU, NO transformers — so it runs in <1s on a
laptop and gates the harness BEFORE freeze (prereg §9.1: "a NEW instrument is not frozen
until its logic is dry-run-validated"). The model-assembly WIRING is validated separately on
the box at §9.2. Mirrors infra/pg/tests/test_w14_localize.py.

    python -m pytest infra/pg/tests/test_xgen_swapback.py -q
    # or:  python infra/pg/tests/test_xgen_swapback.py
"""
from __future__ import annotations

import pytest

from infra.pg import swap_back as SB
from infra.pg import gain_routing as GR


# --------------------------------------------------------------------------- #
# tiny synthetic Gemma-1-style decoder state_dicts (bare GemmaModel keys).
# --------------------------------------------------------------------------- #
def _toy_decoder_sds(d=4, ff=8, base_vocab=10, ext_vocab=12, seed=0):
    """Return (base_lm_sd, target_dec_sd): identical decoder KEYS/shapes, different VALUES.
    target's embed_tokens is bigger (extended vocab) — it must be DROPPED by the swap-back."""
    import torch
    g = torch.Generator().manual_seed(seed)

    def _dec(vocab, off):
        t = lambda *s: torch.randn(*s, generator=g) + off  # noqa: E731
        return {
            "embed_tokens.weight": torch.randn(vocab, d, generator=g) + off,
            "layers.0.self_attn.q_proj.weight": t(d, d),
            "layers.0.mlp.gate_proj.weight": t(ff, d),
            "layers.0.input_layernorm.weight": torch.zeros(d) + off,
            "layers.1.self_attn.q_proj.weight": t(d, d),
            "layers.1.mlp.gate_proj.weight": t(ff, d),
            "norm.weight": torch.zeros(d) + off,
        }
    return _dec(base_vocab, 0.0), _dec(ext_vocab, 5.0)


def _decoder_keys(sd):
    return {k for k in sd if not SB._is_embed(k)}


# --------------------------------------------------------------------------- #
# 1. swap-back state_dict: keeps base embed, swaps every decoder key, strict-clean.
# --------------------------------------------------------------------------- #
def test_swapback_keeps_embed_and_swaps_decoder():
    import torch
    base, target = _toy_decoder_sds()
    new_sd, info = SB.swapback_state_dict(base, target)

    # EXACTLY the base keys (→ a strict-clean 0-missing/0-unexpected load).
    assert set(new_sd) == set(base), "new_sd keys must equal the base shell keys exactly"

    # embed_tokens KEPT from base (base vocab 10, NOT target's extended 12).
    assert torch.equal(new_sd["embed_tokens.weight"], base["embed_tokens.weight"])
    assert new_sd["embed_tokens.weight"].shape[0] == 10

    # every decoder key (matrices AND norms) taken from target.
    for k in _decoder_keys(base):
        assert torch.equal(new_sd[k], target[k]), f"decoder key {k} not swapped to target"

    assert info["n_embed_keys_kept"] == 1
    assert info["n_decoder_keys_swapped"] == len(_decoder_keys(base)) == 6


def test_swapback_identity_when_target_is_base():
    """BEFORE arm: target == base → an identity swap (same forward path as AFTER)."""
    import torch
    base, _ = _toy_decoder_sds()
    new_sd, info = SB.swapback_state_dict(base, base)
    for k in base:
        assert torch.equal(new_sd[k], base[k])
    assert info["n_decoder_keys_swapped"] == 6


def test_swapback_failloud_missing_target_key():
    base, target = _toy_decoder_sds()
    del target["layers.1.mlp.gate_proj.weight"]
    with pytest.raises(RuntimeError, match="absent from the target"):
        SB.swapback_state_dict(base, target)


def test_swapback_failloud_shape_mismatch():
    import torch
    base, target = _toy_decoder_sds()
    target["layers.0.self_attn.q_proj.weight"] = torch.randn(4, 5)  # wrong shape
    with pytest.raises(RuntimeError, match="shape mismatch"):
        SB.swapback_state_dict(base, target)


# --------------------------------------------------------------------------- #
# 2. FLOOR: scale-matched random decoder, embed + 1-D norms kept, deterministic.
# --------------------------------------------------------------------------- #
def test_random_decoder_floor_structure_and_scale():
    import torch
    base, _ = _toy_decoder_sds()
    rnd = SB.random_decoder_state_dict(base, seed=1337)

    assert set(rnd) == set(base)
    # embed_tokens + 1-D norms KEPT verbatim.
    assert torch.equal(rnd["embed_tokens.weight"], base["embed_tokens.weight"])
    assert torch.equal(rnd["norm.weight"], base["norm.weight"])
    assert torch.equal(rnd["layers.0.input_layernorm.weight"], base["layers.0.input_layernorm.weight"])

    # 2-D matrices REPLACED (different values) but per-tensor std matched.
    for k in ("layers.0.self_attn.q_proj.weight", "layers.0.mlp.gate_proj.weight"):
        assert not torch.equal(rnd[k], base[k]), f"{k} should be re-randomised"
        base_std = base[k].float().std().item()
        rnd_std = rnd[k].float().std().item()
        assert abs(rnd_std - base_std) / base_std < 0.35, f"{k} std not scale-matched"


def test_random_decoder_floor_deterministic():
    import torch
    base, _ = _toy_decoder_sds()
    a = SB.random_decoder_state_dict(base, seed=7)
    b = SB.random_decoder_state_dict(base, seed=7)
    c = SB.random_decoder_state_dict(base, seed=8)
    k = "layers.0.self_attn.q_proj.weight"
    assert torch.equal(a[k], b[k]), "same seed must reproduce"
    assert not torch.equal(a[k], c[k]), "different seed must differ"


# --------------------------------------------------------------------------- #
# 3. V2: row-shuffle collapses f_perp toward the dimension-counting null.
# --------------------------------------------------------------------------- #
def test_row_shuffle_collapses_fperp_toward_null():
    import torch
    m, n = 40, 20
    g = torch.Generator().manual_seed(0)
    w_before = torch.randn(m, n, generator=g, dtype=torch.float64)
    w_after = w_before + 0.3 * w_before                     # a purely IN-SPAN ΔW (gain only)

    structured = GR.decompose(w_before, w_after, check_recon=False)["f_perp"]
    assert structured < 0.02, f"in-span ΔW must give f_perp≈0, got {structured}"

    # analytic dimension-counting null for this shape: 1 − r²/(m·n) = 1 − n/m = 0.5.
    null_fp = 1.0 - (min(m, n) ** 2) / (m * n)
    fps = []
    sh = torch.Generator().manual_seed(123)
    for _ in range(50):
        w_shuf = SB.row_shuffled_after(w_before, w_after, sh)
        fps.append(GR.decompose(w_before, w_shuf, check_recon=False)["f_perp"])
    mean_shuf = sum(fps) / len(fps)
    assert mean_shuf > 10 * structured, "shuffle must raise f_perp off the in-span floor"
    assert abs(mean_shuf - null_fp) < 0.15, f"shuffled f_perp {mean_shuf:.3f} not near null {null_fp}"


def test_square_full_rank_fperp_is_zero():
    """Attention (square) matrices have f_perp ≡ 0 by construction → V2 is MLP-only."""
    import torch
    g = torch.Generator().manual_seed(1)
    w_before = torch.randn(8, 8, generator=g, dtype=torch.float64)
    w_after = w_before + torch.randn(8, 8, generator=g, dtype=torch.float64)
    out = GR.decompose(w_before, w_after, check_recon=True)
    assert out["is_square_full_rank"] is True
    assert out["f_perp"] < 1e-9, f"square full-rank f_perp must be 0, got {out['f_perp']}"


def test_decompose_recon_identity_fp64():
    """The fp64 reconstruction identity (recon_residual < 1e-10) — the V2 recon round-trip gate."""
    import torch
    g = torch.Generator().manual_seed(2)
    w_before = torch.randn(40, 20, generator=g, dtype=torch.float64)
    w_after = w_before + 0.1 * torch.randn(40, 20, generator=g, dtype=torch.float64)
    out = GR.decompose(w_before, w_after, check_recon=True)
    assert out["recon_residual"] < 1e-10, f"recon_residual {out['recon_residual']} ≥ 1e-10 (fp64 broken)"


def test_null_band_analytic_fperp_matches_dimension_counting():
    """The V2 threshold source: null_band's analytic f_perp == 1 − r²/(m·n), p95 sane."""
    nb = GR.null_band((40, 20), n_draws=200, seed=0, top_k=8)
    assert abs(nb["analytic_f_perp"] - 0.5) < 1e-9
    # NB: percentile keys are f"p{float}" → "p95.0", NOT "p95" (the carrier job must match).
    assert 0.0 < nb["f_perp"]["p95.0"] < 1.0
    assert nb["rank_r"] == 20


# --------------------------------------------------------------------------- #
# 4. V1 arithmetic + the sha1 discovery/confirmation split (prereg §4.1).
# --------------------------------------------------------------------------- #
def test_s_text_arithmetic_planted():
    after = [1, 1, 0, 0]   # mean 0.5
    before = [1, 1, 1, 1]  # mean 1.0
    floor = [0, 0, 0, 0]   # mean 0.0
    assert SB.s_text(after, before, floor) == pytest.approx(0.5)
    # AFTER == BEFORE → S_text == 1 (full retention); AFTER == FLOOR → 0 (deletion).
    assert SB.s_text(before, before, floor) == pytest.approx(1.0)
    assert SB.s_text(floor, before, floor) == pytest.approx(0.0)


def test_boot_s_text_ci_brackets_point():
    after = [1, 1, 0, 0, 1, 0, 1, 0]
    before = [1] * 8
    floor = [0] * 8
    mean, lo, hi, nb = SB.boot_s_text(after, before, floor, n_boot=500, seed=1337)
    assert nb == 500                          # den=1>0 every draw
    assert lo <= mean <= hi
    assert lo <= SB.s_text(after, before, floor) <= hi


def test_boot_s_text_failloud_misaligned():
    with pytest.raises(AssertionError, match="misaligned"):
        SB.boot_s_text([1, 0, 1], [1, 1], [0, 0, 0])


def test_boot_s_text_failloud_empty_when_before_le_floor():
    # BEFORE < FLOOR on every draw → denom<=0 always → S_text undefined → fail-loud (not opaque).
    with pytest.raises(AssertionError, match="denom<=0"):
        SB.boot_s_text([0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1])


def test_theta_forget_positive_and_bounded():
    # θ_forget = FLOOR + eval-noise band in S_text units: small, positive, well below 1 (retention).
    before = [1] * 200                          # BEFORE ≈ ceiling
    floor = [1, 0, 0, 0] * 50                   # FLOOR ≈ 0.25 (4-way chance) with real sampling noise
    theta = SB.theta_forget(before, floor, n_boot=500, seed=1337)
    assert 0.0 < theta < 0.5, f"θ_forget {theta} not a small positive S_text-noise band"
    # a clearly-deleted AFTER (== FLOOR) has S_text ≈ 0 < θ_forget → would read H_forget;
    # a retained AFTER (== BEFORE) has S_text ≈ 1 > θ_forget → would read H_retain.
    assert SB.s_text(floor, before, floor) < theta < SB.s_text(before, before, floor)


def test_theta_forget_zero_noise_floor():
    # a noiseless FLOOR (all identical) yields θ_forget == 0 (no eval-noise band).
    assert SB.theta_forget([1] * 8, [0] * 8, n_boot=200) == 0.0


def test_hash_split_partition_disjoint_and_covers():
    arr = [10, 11, 12, 13, 14]
    hashes = ["aaaaaaaaaaa0", "aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3", "aaaaaaaaaaa5"]
    disc = SB.subset_by_hash(arr, hashes, "discovery")     # even last hex
    conf = SB.subset_by_hash(arr, hashes, "confirmation")  # odd  last hex
    allv = SB.subset_by_hash(arr, hashes, "all")
    assert disc == [10, 12]                                 # ...0, ...2
    assert conf == [11, 13, 14]                             # ...1, ...3, ...5
    assert sorted(disc + conf) == sorted(allv) == arr       # covers
    assert set(disc).isdisjoint(conf)                       # disjoint


def test_hash_split_aligned_across_arms():
    """Same hashes → discovery/confirmation pick the SAME items in AFTER/BEFORE/FLOOR."""
    hashes = ["%012x" % i for i in range(20)]
    after = list(range(100, 120))
    before = list(range(200, 220))
    a_d = SB.subset_by_hash(after, hashes, "discovery")
    b_d = SB.subset_by_hash(before, hashes, "discovery")
    assert len(a_d) == len(b_d)                             # identical mask
    assert [x - 100 for x in a_d] == [x - 200 for x in b_d]


def test_hash_split_failloud_length_mismatch():
    with pytest.raises(AssertionError, match="length mismatch"):
        SB.subset_by_hash([1, 2, 3], ["a", "b"], "all")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
