"""Weight-free CPU dry-run gate for INHERIT (prereg §9.1 gate; extends the CTRLIFACE 27-test suite).

NO model, NO datasets, NO GPU — tiny planted tensors only, seconds on a CPU box. Proves the PURE
logic of everything INHERIT adds: the bank loader (transported-projector: span from the bank, HOST
h̄), the kwarg-robust PG1 pre-hook adapter, mass-matched tail/mid-band construction, linear CKA +
the mismatched-layer alignment bound, the overlap null band with rejection room, the sha1
content-hash split, paired per-item RF/gap bootstrap + bootstrap-p + BH, the frozen power rule,
U_unt derivation (consensus Gram → bottom-k) + RMSNorm pullback, the ridge probe kit, the frozen
layer-promotion rule, and the input-residual grad-Gram capture (S_vl) vs manual autograd.

Run (on a remote `pg` CPU instance — never locally; torch is a large lib):
    python -m pytest infra/pg/tests/test_inherit_core.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from infra.pg import ctrliface_core as CC
from infra.pg import inherit_core as IC
from infra.pg import inherit_pg1 as IP


def _orthobasis(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64)).Q[:, :k].contiguous()


def _tiny_bank(d=16, n_layers=3, kmax=6):
    bases = {L: _orthobasis(d, kmax, seed=10 + L) for L in range(n_layers)}
    hbar = {L: torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(20 + L))
            for L in range(n_layers)}
    trace = {L: float(3.0 + L) for L in range(n_layers)}
    return {"bases": bases, "hbar": hbar, "trace": trace, "d": d, "n_prefix": None}


# ---------------------------------------------------------------------------
# 1. Bank loader / transported projector.
# ---------------------------------------------------------------------------
def test_validate_bank_roundtrip_and_rejects():
    b = _tiny_bank()
    bases, hbars, traces = IC.validate_bank(b, d=16, n_layers=3, k_min=6)
    assert set(bases) == {0, 1, 2} and bases[0].shape == (16, 6)
    bad = _tiny_bank()
    bad["bases"][1] = bad["bases"][1] * 2.0                      # not orthonormal
    with pytest.raises(AssertionError):
        IC.validate_bank(bad, d=16, n_layers=3, k_min=6)
    with pytest.raises(AssertionError):
        IC.validate_bank(_tiny_bank(), d=32, n_layers=3, k_min=6)  # wrong d
    with pytest.raises(AssertionError):
        IC.validate_bank(_tiny_bank(), d=16, n_layers=4, k_min=6)  # missing layer


def test_slice_k_orthonormal_and_bounds():
    Q = _orthobasis(12, 5)
    Vk = IC.slice_k(Q, 3)
    assert Vk.shape == (12, 3)
    assert torch.allclose(Vk.T @ Vk, torch.eye(3, dtype=torch.float64), atol=1e-9)
    with pytest.raises(AssertionError):
        IC.slice_k(Q, 6)


def test_transported_projector_uses_host_hbar():
    """The B1 contract: span from the (π0.5) bank, replacement value from the HOST (PG1)."""
    d, k = 10, 3
    Q = _orthobasis(d, k, seed=4)                                # "bank" span
    h_host = torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    x = torch.randn(7, d, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    x2 = CC.apply_mean_ablation(x, Q, h_host)
    # span(Q) component of (x' − h_host) is zero; complement untouched
    assert float((((x2 - h_host) @ Q)).abs().max()) < 1e-9
    P = torch.eye(d, dtype=torch.float64) - Q @ Q.T
    assert torch.allclose(x2 @ P, x @ P, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. PG1 pre-hook adapter (kwarg-robust; ALL positions incl. decode steps).
# ---------------------------------------------------------------------------
class _EchoLayer(torch.nn.Module):
    def forward(self, hidden_states, flag=None):
        return hidden_states


def _mk_xqh(d=8, k=2, n=5, seed=7):
    Q = _orthobasis(d, k, seed=seed)
    h = torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 1))
    x = torch.randn(2, n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 2))
    return x, Q, h


def test_pg1_pre_hook_ablate_positional_and_kwarg():
    x, Q, h = _mk_xqh()
    want = CC.apply_mean_ablation(x, Q, h)
    layer = _EchoLayer()
    hd = layer.register_forward_pre_hook(IC.make_pg1_pre_hook_ablate(Q, h), with_kwargs=True)
    assert torch.allclose(layer(x), want, atol=1e-9)             # positional
    assert torch.allclose(layer(hidden_states=x), want, atol=1e-9)  # kwarg
    hd.remove()
    assert torch.allclose(layer(x), x)                           # hook removed → identity


def test_pg1_pre_hook_replace_and_capture():
    x, _Q, h = _mk_xqh()
    layer = _EchoLayer()
    hd = layer.register_forward_pre_hook(IC.make_pg1_pre_hook_replace(h), with_kwargs=True)
    out = layer(x)
    assert torch.allclose(out, h.expand_as(x), atol=1e-9)
    hd.remove()
    acc = {}
    hd = layer.register_forward_pre_hook(IC.make_pg1_pre_hook_capture(acc, 8, x_cap=6),
                                         with_kwargs=True)
    layer(x)
    layer(hidden_states=x)
    hd.remove()
    hbar, trace, X, n = IC.finalize_capture(acc)
    flat = torch.cat([x.reshape(-1, 8)] * 2, 0)
    assert n == flat.shape[0]
    assert torch.allclose(hbar, flat.mean(0), atol=1e-9)
    assert abs(trace - float(((flat - flat.mean(0)) ** 2).sum())) < 1e-6
    assert X.shape == (6, 8) and torch.allclose(X, flat[:6], atol=1e-12)   # cap respected


# ---------------------------------------------------------------------------
# 3. Arm O — mass-matched tail / mid-band.
# ---------------------------------------------------------------------------
def test_mass_matched_tail_minimal_and_infeasible():
    ev = torch.tensor([4.0, 3.0, 2.0] + [0.5] * 12, dtype=torch.float64)
    j, mass, target = IC.mass_matched_tail(ev, k=1)              # target 4.0
    assert target == 4.0 and mass >= target
    assert float(ev[j + 1:].sum()) < target                      # minimal (one shorter fails)
    assert j >= 1                                                # disjoint from top-k
    with pytest.raises(AssertionError):                          # complement can't match
        IC.mass_matched_tail(torch.tensor([100.0, 1.0, 1.0, 1.0], dtype=torch.float64), k=1)


def test_mass_matched_mid_contains_center_and_matches():
    d = 20
    ev = torch.linspace(5.0, 0.1, d, dtype=torch.float64)
    lo, hi, mass, target = IC.mass_matched_mid(ev, k=2)
    assert mass >= target and lo > 2                             # disjoint from top-k block
    assert lo <= d // 2 <= hi
    with pytest.raises(AssertionError):
        IC.mass_matched_mid(torch.tensor([50.0, 50.0, 1e-6, 1e-6, 1e-6, 1e-6],
                                         dtype=torch.float64), k=2)


# ---------------------------------------------------------------------------
# 4. Transport gate — CKA + mismatched-layer bound.
# ---------------------------------------------------------------------------
def test_linear_cka_rails():
    g = torch.Generator().manual_seed(11)
    X = torch.randn(200, 12, generator=g, dtype=torch.float64)
    R = torch.linalg.qr(torch.randn(12, 12, generator=g, dtype=torch.float64)).Q
    assert abs(IC.linear_cka(X, X) - 1.0) < 1e-9
    assert abs(IC.linear_cka(X, X @ R) - 1.0) < 1e-9             # rotation-invariant
    Y = torch.randn(200, 12, generator=g, dtype=torch.float64)
    assert IC.linear_cka(X, Y) < 0.2                             # independent ⇒ near-0


def test_alignment_bound_gate():
    null = [0.1, 0.2, 0.15, 0.12, 0.3, 0.22, 0.18, 0.25, 0.11, 0.28, 0.19, 0.21]
    b = IC.alignment_bound(null, q=0.95)
    assert max(null) >= b >= np.quantile(null, 0.5)
    assert 0.9 > b                                                # a matched CKA of 0.9 clears it


# ---------------------------------------------------------------------------
# 5. Arm P-geo — null band with rejection room.
# ---------------------------------------------------------------------------
def test_overlap_null_band_and_beyond():
    d, kA, kB = 24, 4, 4
    ref = _orthobasis(d, kA, seed=13)
    band = IC.overlap_null_band(ref, d, kB, n_draws=40, seed=17)
    assert 0 < band["mean"] < 3 * kA / d and band["spread"] > 0
    ov_self, _ = CC.principal_angle_overlap(ref, ref)
    assert IC.beyond_band(ov_self, band)                         # identical subspace: separation
    ov_rand, _ = CC.principal_angle_overlap(ref, _orthobasis(d, kB, seed=999))
    assert not IC.beyond_band(ov_rand, band)                     # a random draw: inside the band


# ---------------------------------------------------------------------------
# 6. sha1 content-hash split.
# ---------------------------------------------------------------------------
def test_sha1_split_content_stable_and_nonempty():
    hashes = [f"item{i:03d}" for i in range(64)]
    s1 = IC.sha1_split(hashes)
    assert s1["n_disc"] + s1["n_conf"] == 64 and s1["n_disc"] > 0 and s1["n_conf"] > 0
    perm = np.random.default_rng(0).permutation(64)
    s2 = IC.sha1_split([hashes[i] for i in perm])
    m1 = {hashes[i]: bool(i in set(s1["conf_idx"].tolist())) for i in range(64)}
    m2 = {hashes[perm[i]]: bool(i in set(s2["conf_idx"].tolist())) for i in range(64)}
    assert m1 == m2                                              # bucket follows CONTENT, not index


# ---------------------------------------------------------------------------
# 7. §7 statistics — paired RF/gap bootstrap, bootstrap-p, BH, power rule.
# ---------------------------------------------------------------------------
def test_rf_item_bootstrap_point_and_ci():
    rng = np.random.default_rng(23)
    intact = (rng.random(4000) < 0.8).astype(float)
    ablate = (rng.random(4000) < 0.45).astype(float)
    r = IC.rf_item_bootstrap(ablate, intact, floor=0.1)
    want = (ablate.mean() - 0.1) / (intact.mean() - 0.1)
    assert abs(r["rf"] - want) < 1e-12
    assert r["ci"][0] < want < r["ci"][1] and (r["ci"][1] - r["ci"][0]) < 0.15


def test_gap_item_bootstrap_detects_planted_gap():
    rng = np.random.default_rng(29)
    intact = (rng.random(4000) < 0.8).astype(float)
    nullarm = (rng.random(4000) < 0.78).astype(float)            # RF ≈ 0.97
    saction = (rng.random(4000) < 0.40).astype(float)            # RF ≈ 0.43
    g = IC.gap_item_bootstrap(saction, nullarm, intact, floor=0.1)
    assert g["gap"] > 0.4 and g["ci_lo"] > 0.3                   # planted gap detected
    g0 = IC.gap_item_bootstrap(nullarm, nullarm, intact, floor=0.1)
    assert g0["ci"][0] <= 0.0 <= g0["ci"][1]                     # no self-gap


def test_boot_p_and_bh_adjust():
    boots = np.concatenate([np.full(99, 0.5), [0.0]])
    assert IC.boot_p_gt(boots, 0.1) < 0.03                       # nearly all draws above thresh
    assert IC.boot_p_gt(boots, 0.9) > 0.97
    adj = IC.bh_adjust([0.01, 0.04, 0.03, 0.02])
    assert np.allclose(adj, [0.04, 0.04, 0.04, 0.04])            # classic BH example
    adj2 = IC.bh_adjust([0.001, 0.5])
    assert adj2[0] < 0.01 and adj2[1] == 0.5


def test_choose_n_items_power_rule():
    assert IC.choose_n_items(sd_intact=0.5, dynamic_range=0.5, frac=0.05) == 2048   # clamp low
    assert IC.choose_n_items(sd_intact=0.5, dynamic_range=0.1, frac=0.05) == 8192   # clamp high
    n = IC.choose_n_items(sd_intact=0.5, dynamic_range=0.2, frac=0.05, n_min=1, n_max=10 ** 9)
    assert n == int(np.ceil((1.96 * 0.5 / (0.05 * 0.2)) ** 2))
    with pytest.raises(AssertionError):
        IC.choose_n_items(0.5, 0.0)


# ---------------------------------------------------------------------------
# 8. Arm P — U_unt derivation + RMSNorm pullback.
# ---------------------------------------------------------------------------
def test_consensus_gram_bottom_recovers_untouched():
    d, k = 12, 2
    U = _orthobasis(d, k, seed=31)                               # the planted untouched band
    P = torch.eye(d, dtype=torch.float64) - U @ U.T
    g = torch.Generator().manual_seed(37)
    deltas = [torch.randn(d, d, generator=g, dtype=torch.float64) @ P for _ in range(8)]
    G = IC.consensus_gram(deltas)
    Ub, evals = IC.bottom_eigvecs(G, k)
    assert float(evals.max()) < 1e-18                            # untouched ⇒ ~0 eigvals
    ov, _ = CC.principal_angle_overlap(Ub, U)
    assert ov > 1 - 1e-9
    with pytest.raises(AssertionError):
        IC.consensus_gram([torch.zeros(d, d, dtype=torch.float64)])


def test_rmsnorm_pullback_recovers_source_span():
    d, k = 10, 3
    gamma = torch.linspace(-0.5, 2.0, d, dtype=torch.float64)
    Ux = _orthobasis(d, k, seed=41)                              # x-space (residual) span
    Uy = CC.orthonormalize((1.0 + gamma).unsqueeze(1) * Ux)      # its post-RMSNorm image
    back = IC.rmsnorm_pullback(Uy, gamma)
    ov, _ = CC.principal_angle_overlap(back, Ux)
    assert ov > 1 - 1e-9
    same = IC.rmsnorm_pullback(Ux, torch.zeros(d, dtype=torch.float64))
    ov2, _ = CC.principal_angle_overlap(same, Ux)
    assert ov2 > 1 - 1e-9                                        # γ=0 ⇒ identity on spans


# ---------------------------------------------------------------------------
# 9. Arm C — ridge probe kit.
# ---------------------------------------------------------------------------
def test_ridge_r2_signal_vs_noise():
    g = torch.Generator().manual_seed(43)
    Ztr = torch.randn(300, 6, generator=g, dtype=torch.float64)
    Zte = torch.randn(120, 6, generator=g, dtype=torch.float64)
    W = torch.randn(6, 2, generator=g, dtype=torch.float64)
    r2, per = IC.ridge_r2(Ztr, Ztr @ W, Zte, Zte @ W, alpha=1e-6)
    assert r2 > 0.999 and len(per) == 2
    ynoise_tr = torch.randn(300, 2, generator=g, dtype=torch.float64)
    ynoise_te = torch.randn(120, 2, generator=g, dtype=torch.float64)
    r2n, _ = IC.ridge_r2(Ztr, ynoise_tr, Zte, ynoise_te, alpha=1e-6)
    assert r2n < 0.2


# ---------------------------------------------------------------------------
# 10. Frozen layer-promotion rule (the spec §3 selection, mechanical).
# ---------------------------------------------------------------------------
def test_promote_layers_ctrliface_shape():
    prof = {"0": 0.50, "3": 1.15, "6": 0.90, "9": 0.80, "12": 0.75, "14": 0.65, "15": 1.00}
    sel = IC.promote_layers(prof)
    assert sel == {"primary": 0, "secondary": 14, "anchors": [6, 15]}


# ---------------------------------------------------------------------------
# 11. S_vl input-residual grad-Gram capture vs manual autograd (frozen toy net).
# ---------------------------------------------------------------------------
class _TinyStack(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        g = torch.Generator().manual_seed(47)
        self.l0 = torch.nn.Linear(d, d, dtype=torch.float64)
        self.l1 = torch.nn.Linear(d, d, dtype=torch.float64)
        with torch.no_grad():
            for lin in (self.l0, self.l1):
                lin.weight.copy_(torch.randn(d, d, generator=g, dtype=torch.float64) / d ** 0.5)
                lin.bias.zero_()

    def forward(self, x):
        return self.l1(torch.tanh(self.l0(x)))


def test_accumulate_gram_pre_matches_autograd():
    d = 6
    net = _TinyStack(d)
    layers = [net.l0, net.l1]
    g = torch.Generator().manual_seed(53)
    batches = [torch.randn(2, 3, d, generator=g, dtype=torch.float64) for _ in range(2)]

    def loss_fn(b):
        return net(b).pow(2).sum()

    G, npos = IP.accumulate_gram_pre(net, layers, loss_fn, batches, [0, 1], d=d)
    assert npos == sum(b.shape[0] * b.shape[1] for b in batches)

    G_manual = {0: torch.zeros(d, d, dtype=torch.float64), 1: torch.zeros(d, d, dtype=torch.float64)}
    for b in batches:
        x = b.detach().clone().requires_grad_(True)              # fresh leaf (IP pass mutated b)
        h1 = torch.tanh(net.l0(x))                               # input of l1
        h1.retain_grad()
        loss = net.l1(h1).pow(2).sum()
        loss.backward()
        for L, grad in ((0, x.grad), (1, h1.grad)):
            fl = grad.reshape(-1, d)
            G_manual[L] += fl.T @ fl
    for L in (0, 1):
        assert torch.allclose(G[L], G_manual[L], atol=1e-9), f"layer {L} Gram mismatch"


def test_harvest_matches_manual():
    d = 6
    net = _TinyStack(d)
    layers = [net.l0, net.l1]
    g = torch.Generator().manual_seed(59)
    batches = [torch.randn(2, 4, d, generator=g, dtype=torch.float64) for _ in range(2)]
    res = IP.harvest(layers, lambda b: net(b), batches, [0, 1], d=d, x_cap=5)
    flat0 = torch.cat([b.reshape(-1, d) for b in batches], 0)
    hbar0, trace0, X0, n0 = res[0]
    assert n0 == flat0.shape[0]
    assert torch.allclose(hbar0, flat0.mean(0), atol=1e-9)
    assert abs(trace0 - float(((flat0 - flat0.mean(0)) ** 2).sum())) < 1e-6
    assert X0.shape == (5, d)
    hbar1 = res[1][0]                                            # input of l1 = tanh(l0(x))
    want1 = torch.cat([torch.tanh(net.l0(b)).reshape(-1, d) for b in batches], 0).mean(0)
    assert torch.allclose(hbar1, want1, atol=1e-9)
