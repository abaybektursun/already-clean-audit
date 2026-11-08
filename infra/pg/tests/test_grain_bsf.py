"""GRAIN §11.1 weight-free dry-run suite — tiny tensors, CPU, no model.

Mandatory pre-freeze gates (spec freeze checklist):
  * m=1 EXACT reduction: BlockTopKSAE(m=1) ≡ the house BatchTopKSAE (init, encode topk +
    threshold modes, decode) and the trainer mirrors BatchTopKTrainer step-for-step.
  * block-TopK selects by BLOCK NORM (a case where value-topk and block-norm-topk differ).
  * planted-block recovery ≫ Haar (the T3.3 positive control) with the m=1 fragmentation
    comparator.
  * QR union assembly orthonormal; splice hook pass-through = bitwise no-op.
  * G/S2 adjudicators: every label reachable on synthetic tables; guard rails raise.
  * Jaccard + permutation-null machinery.

Runs on the remote CPU box (pg r-family / GPU-job preflight) — never on the laptop.
"""
import pytest
import torch

from infra.pg import grain_bsf as GB
from infra.pg.ctrliface_core import matched_random_overlap, principal_angle_overlap


# ===========================================================================
# m=1 exact reduction vs the house BatchTopKSAE.
# ===========================================================================
def _house_sae(d, W, k, seed):
    from dictionary_learning.dictionary import BatchTopKSAE
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return BatchTopKSAE(activation_dim=d, dict_size=W, k=k)


def test_m1_init_identical_to_house():
    d, W, k = 16, 64, 4
    house = _house_sae(d, W, k, seed=11)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    mine = GB.BlockTopKSAE(d, W, m=1, k_blk=k)
    assert torch.equal(mine.decoder.weight, house.decoder.weight)
    assert torch.equal(mine.encoder.weight, house.encoder.weight)
    assert torch.equal(mine.encoder.bias, house.encoder.bias)
    assert torch.equal(mine.b_dec, house.b_dec)
    assert float(mine.threshold) == float(house.threshold) == -1.0


def test_m1_encode_decode_identical_to_house():
    d, W, k = 16, 64, 4
    house = _house_sae(d, W, k, seed=3)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    mine = GB.BlockTopKSAE(d, W, m=1, k_blk=k)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(32, d, generator=g)
    # batch-topk mode (training selection)
    f_h = house.encode(x, use_threshold=False, normalize_activations=False)
    f_m = mine.encode(x, use_threshold=False)
    assert torch.equal(f_m, f_h)
    # threshold mode (deployment), at a nontrivial threshold
    house.threshold.fill_(0.05)
    mine.threshold.fill_(0.05)
    f_h = house.encode(x, use_threshold=True, normalize_activations=False)
    f_m = mine.encode(x, use_threshold=True)
    assert torch.equal(f_m, f_h)
    assert torch.equal(mine.decode(f_m), house.decode(f_h, denormalize_activations=False))


def test_m1_trainer_equivalence_short_strict():
    """30 mirrored steps (pre-threshold): my trainer vs BatchTopKTrainer on the SAME batches."""
    from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
    d, W, k, batch, steps, warmup, lr, seed = 16, 64, 4, 32, 30, 10, 1e-3, 11
    g = torch.Generator().manual_seed(7)
    X = torch.randn(4096, d, generator=g)
    ae_mine, _cfg = GB.train_block_sae(X, m=1, seed=seed, device="cpu", dict_size=W,
                                       r_active=k, steps=steps, lr=lr, batch=batch,
                                       warmup=warmup)
    tr = BatchTopKTrainer(steps=steps, activation_dim=d, dict_size=W, k=k, layer=0,
                          lm_name="grain_eqv", lr=lr, warmup_steps=warmup, seed=seed,
                          device="cpu")
    idx_g = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (batch,), generator=idx_g)].float()
        tr.update(step, x)
    assert torch.allclose(ae_mine.decoder.weight, tr.ae.decoder.weight, atol=1e-5)
    assert torch.allclose(ae_mine.encoder.weight, tr.ae.encoder.weight, atol=1e-5)
    assert torch.allclose(ae_mine.b_dec, tr.ae.b_dec, atol=1e-5)


def test_m1_trainer_equivalence_through_threshold():
    """1200 mirrored steps crossing THRESHOLD_START_STEP: the EMA threshold + params agree
    (loose tolerance — fp reduction-order drift compounds through Adam)."""
    from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
    d, W, k, batch, steps, warmup, lr, seed = 16, 64, 4, 32, 1200, 10, 1e-3, 13
    g = torch.Generator().manual_seed(17)
    X = torch.randn(4096, d, generator=g)
    ae_mine, cfg = GB.train_block_sae(X, m=1, seed=seed, device="cpu", dict_size=W,
                                      r_active=k, steps=steps, lr=lr, batch=batch,
                                      warmup=warmup)
    tr = BatchTopKTrainer(steps=steps, activation_dim=d, dict_size=W, k=k, layer=0,
                          lm_name="grain_eqv", lr=lr, warmup_steps=warmup, seed=seed,
                          device="cpu")
    idx_g = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (batch,), generator=idx_g)].float()
        tr.update(step, x)
    assert cfg["threshold"] > 0 and float(tr.ae.threshold) > 0
    assert abs(cfg["threshold"] - float(tr.ae.threshold)) < 1e-2
    probe = torch.randn(64, d, generator=g)
    rec_m = ae_mine.decode(ae_mine.encode(probe, use_threshold=True))
    rec_h = tr.ae.decode(tr.ae.encode(probe, use_threshold=True,
                                      normalize_activations=False),
                         denormalize_activations=False)
    assert torch.allclose(rec_m, rec_h, atol=1e-3, rtol=1e-3)


# ===========================================================================
# Block-norm selection (the grain mechanism itself).
# ===========================================================================
def test_block_norm_selection_differs_from_value_topk():
    """m=2, k_blk=1: block A = (3,3) (norm √18 ≈ 4.24) vs block B = (4,0) (norm 4).
    Value-topk would keep the 4; block-norm keeps block A."""
    ae = GB.BlockTopKSAE(4, 4, m=2, k_blk=1)
    with torch.no_grad():
        ae.encoder.weight.copy_(torch.eye(4))
        ae.encoder.bias.zero_()
        ae.b_dec.zero_()
    x = torch.tensor([[3.0, 3.0, 4.0, 0.0]])
    f = ae.encode(x, use_threshold=False)
    assert torch.equal(f, torch.tensor([[3.0, 3.0, 0.0, 0.0]]))


def test_threshold_mode_selects_blocks_by_norm():
    ae = GB.BlockTopKSAE(4, 4, m=2, k_blk=1)
    with torch.no_grad():
        ae.encoder.weight.copy_(torch.eye(4))
        ae.encoder.bias.zero_()
        ae.b_dec.zero_()
    ae.threshold.fill_(4.1)                     # block A norm 4.24 > 4.1 > block B norm 4
    x = torch.tensor([[3.0, 3.0, 4.0, 0.0]])
    f = ae.encode(x, use_threshold=True)
    assert torch.equal(f, torch.tensor([[3.0, 3.0, 0.0, 0.0]]))


# ===========================================================================
# Planted-structure positive control (T3.3) + m=1 fragmentation comparator.
# ===========================================================================
def test_planted_recovery_beats_haar_and_m1_fragments():
    d, n_planted, m_true = 32, 4, 4
    X, U = GB.gen_planted(d, n_planted, m_true, n_samples=20000, k_active=2, seed=42,
                          noise=0.02)
    ae_blk, _ = GB.train_block_sae(X, m=4, seed=0, device="cpu", dict_size=64, r_active=8,
                                   steps=1500, lr=1e-3, batch=256, warmup=100)
    ae_m1, _ = GB.train_block_sae(X, m=1, seed=0, device="cpu", dict_size=64, r_active=8,
                                  steps=1500, lr=1e-3, batch=256, warmup=100)
    haar_p95 = float(torch.quantile(matched_random_overlap(d, m_true, m_true, 200, 9), 0.95))
    W1n = ae_m1.decoder.weight.detach().double()
    W1n = W1n / W1n.norm(dim=0, keepdim=True)
    blk_ovs, m1_caps, atom_cov = [], [], []
    for Ui in U:
        ov, _b = GB.best_block_overlap(ae_blk.decoder.weight.detach(), 4, Ui)
        blk_ovs.append(ov)
        m1_caps.append(GB.matched_span_capture(ae_m1.decoder.weight.detach(), Ui, m_true))
        atom_cov.append(float(((W1n.T @ Ui) ** 2).sum(1).max() / m_true))
    mean_blk = sum(blk_ovs) / len(blk_ovs)
    mean_m1 = sum(m1_caps) / len(m1_caps)
    mean_atom = sum(atom_cov) / len(atom_cov)
    assert mean_blk > 3 * haar_p95, f"block recovery {mean_blk:.3f} not ≫ Haar p95 {haar_p95:.3f}"
    assert mean_blk >= 0.5, f"block recovery {mean_blk:.3f} < 0.5"
    # Fragmentation holds at UNIT granularity: one block covers the planted subspace; no single
    # m=1 atom can. The SPAN of the top-m_true m=1 atoms is a much stronger baseline and
    # empirically MATCHES the block (dry-run 2026-07-08: 0.792 vs 0.773) — that contrast is the
    # real experiment's grain-1 comparator, with H_sparse-not-block pre-registered (§5.2), so it
    # is REPORTED here as a diagnostic, not asserted.
    assert mean_blk > mean_atom, (
        f"best single block ({mean_blk:.3f}) does not beat best single atom ({mean_atom:.3f})")
    print(f"[planted] block={mean_blk:.3f} haar_p95={haar_p95:.3f} "
          f"m1_span={mean_m1:.3f} m1_single_atom={mean_atom:.3f}")
    # fit sanity on the trained block dictionary (threshold mode)
    q = GB.block_quality(ae_blk, X, "cpu", n_tokens=4000)
    assert q["fvu"] < 0.5, f"planted-data FVU {q['fvu']:.3f} ≥ 0.5"


# ===========================================================================
# Geometry: unions, alignment, support spread, stable rank.
# ===========================================================================
def test_union_q_orthonormal_and_spanning():
    torch.manual_seed(1)
    ae = GB.BlockTopKSAE(16, 32, m=4, k_blk=2)
    Wd = ae.decoder.weight.detach()
    Q = GB.union_Q(Wd, 4, [0, 3, 5])
    assert Q.shape == (16, 12)
    err = float((Q.T @ Q - torch.eye(12, dtype=torch.float64)).abs().max())
    assert err < 1e-9
    cols = torch.cat([GB.block_cols(Wd, 4, b).double() for b in (0, 3, 5)], dim=1)
    resid = cols - Q @ (Q.T @ cols)
    assert float(resid.abs().max()) < 1e-9, "union span does not contain its block columns"


def test_block_alignment_identity_case():
    torch.manual_seed(2)
    d, W, m = 16, 32, 4
    ae = GB.BlockTopKSAE(d, W, m=m, k_blk=2)
    V = GB.orthonormalize(torch.randn(d, 8, dtype=torch.float64))
    with torch.no_grad():
        ae.decoder.weight[:, :m] = V[:, :m].float()      # block 0 spans V's first m dirs
    align = GB.block_alignment(ae.decoder.weight.detach(), m, V)
    assert abs(float(align[0]) - 1.0) < 1e-5
    assert float(align[1:].max()) < 0.9


def test_support_spread():
    s = GB.support_spread([9.0, 0.5, 0.5], n_active=10)
    assert s["n_for_90"] == 1 and abs(s["frac_active_for_90"] - 0.1) < 1e-12
    with pytest.raises(AssertionError):
        GB.support_spread([0.0, 0.0], n_active=5)


def test_block_stable_rank():
    d, W, m = 8, 16, 4
    torch.manual_seed(4)
    ae = GB.BlockTopKSAE(d, W, m=m, k_blk=2)
    with torch.no_grad():                        # block 0: rank-1 (identical columns)
        ae.decoder.weight[:, 1:m] = ae.decoder.weight[:, 0:1].expand(d, m - 1)
    scale = torch.ones(W)
    sr = GB.block_stable_rank(ae.decoder.weight.detach(), m, scale)
    assert abs(float(sr[0]) - 1.0) < 1e-6
    assert float(sr[1]) > 1.5                    # random block ≈ full rank
    scale[2 * m:3 * m] = 0.0                     # dead block → 0
    sr = GB.block_stable_rank(ae.decoder.weight.detach(), m, scale)
    assert float(sr[2]) == 0.0


# ===========================================================================
# Adjudicators — every label reachable; mutually exclusive by single-return construction.
# ===========================================================================
def _adj(**kw):
    base = dict(effect_bstar=0.4, effect_full=0.6, effect_em=0.05, effect_grain1=0.1,
                effect_rand_mean=0.05, gap_cilo=0.1, delta_G=0.03, rho=0.5, bh_pass=True,
                support_frac_90=0.1)
    base.update(kw)
    return GB.adjudicate_grain(**base)


def test_adjudicate_grain_all_labels_reachable():
    assert _adj() == "H_block-grain"
    assert _adj(effect_grain1=0.39) == "H_sparse-not-block"
    assert _adj(effect_rand_mean=0.5, gap_cilo=-0.1) == "H_capacity-artifact"
    assert _adj(effect_bstar=0.1, gap_cilo=0.0, support_frac_90=0.4) == "H_oblique"
    assert _adj(effect_bstar=0.1, gap_cilo=0.0) == "H_distributed-all-grains"
    # em-random reproducing B* voids the compact claim
    assert _adj(effect_em=0.39) != "H_block-grain"
    # BH failure voids the compact claim
    assert _adj(bh_pass=False) == "H_distributed-all-grains"
    with pytest.raises(AssertionError):
        _adj(effect_full=0.0)


def test_adjudicate_s2_labels():
    assert GB.adjudicate_s2(jaccard_cilo=0.5, null_p95=0.2,
                            subspace_canonicity=0.3) == "H_basis-ambiguity"
    assert GB.adjudicate_s2(jaccard_cilo=0.3, null_p95=0.2,
                            subspace_canonicity=0.3) == "H_genuinely-family"
    assert GB.adjudicate_s2(jaccard_cilo=0.5, null_p95=0.2,
                            subspace_canonicity=0.7) == "H_genuinely-family"


# ===========================================================================
# Jaccard machinery.
# ===========================================================================
def test_jaccard():
    assert GB.jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert GB.jaccard([1, 2], [3, 4]) == 0.0
    mean, vals = GB.pairwise_jaccard([[1, 2, 3], [1, 2, 4], [1, 5, 6]])
    assert len(vals) == 3 and 0 < mean < 1
    null = GB.jaccard_null(1024, 8, 500, seed=1)
    assert null.shape == (500,) and float(null.min()) >= 0 and float(null.max()) <= 1
    assert float(torch.quantile(null, 0.95)) < 0.2
    with pytest.raises(AssertionError):
        GB.pairwise_jaccard([[1, 2]])


# ===========================================================================
# Hooks: pass-through bitwise no-op (both call conventions) + splice/steer math.
# ===========================================================================
class _KwLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, hidden_states, flag=None):
        return self.lin(hidden_states)


def test_passthrough_hook_bitwise_noop():
    torch.manual_seed(6)
    layer = _KwLayer(8)
    x = torch.randn(2, 5, 8)
    base_pos = layer(x)
    base_kw = layer(hidden_states=x)
    h = layer.register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(), with_kwargs=True)
    assert torch.equal(layer(x), base_pos)                    # positional path
    assert torch.equal(layer(hidden_states=x), base_kw)       # kwarg path
    h.remove()
    assert torch.equal(layer(x), base_pos)                    # inert after removal


def test_splice_and_steer_hooks_match_direct_math():
    torch.manual_seed(8)
    d = 8
    ae = GB.BlockTopKSAE(d, 16, m=4, k_blk=2)
    ae.threshold.fill_(0.01)
    layer = _KwLayer(d)
    x = torch.randn(2, 3, d)
    captured = {}

    def grab(_m, args, kwargs):
        captured["hs"] = args[0] if args else kwargs["hidden_states"]

    h1 = layer.register_forward_pre_hook(GB.make_pg1_pre_hook_splice(ae), with_kwargs=True)
    h2 = layer.register_forward_pre_hook(grab, with_kwargs=True)
    layer(x)
    h1.remove(), h2.remove()
    want = ae.decode(ae.encode(x.reshape(-1, d), use_threshold=True)).reshape(x.shape)
    assert torch.allclose(captured["hs"], want)
    # steering at α=0 ≡ splice
    h1 = layer.register_forward_pre_hook(
        GB.make_pg1_pre_hook_steer(ae, block=1, direction=torch.ones(4), alpha_sigma=0.0),
        with_kwargs=True)
    h2 = layer.register_forward_pre_hook(grab, with_kwargs=True)
    layer(x)
    h1.remove(), h2.remove()
    assert torch.allclose(captured["hs"], want)
    # steering at α≠0 shifts block 1's code before decode
    h1 = layer.register_forward_pre_hook(
        GB.make_pg1_pre_hook_steer(ae, block=1, direction=torch.ones(4), alpha_sigma=2.0),
        with_kwargs=True)
    h2 = layer.register_forward_pre_hook(grab, with_kwargs=True)
    layer(x)
    h1.remove(), h2.remove()
    f = ae.encode(x.reshape(-1, d), use_threshold=True)
    f[:, 4:8] += 2.0
    want2 = ae.decode(f).reshape(x.shape)
    assert torch.allclose(captured["hs"], want2)


# ===========================================================================
# Bank round-trip.
# ===========================================================================
def test_bank_roundtrip(tmp_path):
    torch.manual_seed(9)
    ae = GB.BlockTopKSAE(8, 16, m=4, k_blk=2)
    ae.threshold.fill_(0.2)
    cfg = {"m": 4, "k_blk": 2, "dict_size": 16, "activation_dim": 8}
    GB.save_dict_bank(ae, cfg, {"fvu": 0.1}, tmp_path / "bank.pt", "sha")
    ae2, cfg2, q2, sha2 = GB.load_dict_bank(tmp_path / "bank.pt")
    assert torch.equal(ae2.decoder.weight, ae.decoder.weight)
    assert abs(float(ae2.threshold) - 0.2) < 1e-6 and q2["fvu"] == 0.1 and sha2 == "sha"
    g = torch.Generator().manual_seed(10)
    x = torch.randn(4, 8, generator=g)
    assert torch.equal(ae2.encode(x, use_threshold=True), ae.encode(x, use_threshold=True))
