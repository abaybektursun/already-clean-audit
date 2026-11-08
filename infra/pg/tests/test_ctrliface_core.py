"""Weight-free CPU dry-run gate for CTRLIFACE (frozen prereg §9.1 gate).

NO model, NO datasets, NO GPU, NO lerobot — tiny planted tensors only, <1s on a CPU box.
Proves the PURE logic (activation-Gram top-k eigen-recovery, the mean-ablation projector
algebra + removed-energy accounting, energy-matched-random rank, principal-angle overlap
rails + matched-random null centring, the prefix-masked layer hook on a tuple/tensor output,
and the frozen §6 verdict predicates) BEFORE any GPU minute is spent. Distinct from the §9.0
runtime preflight (which proves the WIRING on real weights + lerobot).

Run (on a remote `pg` CPU instance — never locally; torch is a large lib):
    python -m pytest infra/pg/tests/test_ctrliface_core.py -q
"""
from __future__ import annotations

import numpy as np
import torch

from infra.pg import ctrliface_core as CC
from infra.pg import ctrliface_run as CR
from infra.pg import ctrliface_rollout as RO
from infra.pg import ctrliface_pi05_hook as PH


def _orthobasis(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64)).Q[:, :k].contiguous()


# ---------------------------------------------------------------------------
# 1. Activation-Gram → top-k eigenvectors recovers a planted subspace.
# ---------------------------------------------------------------------------
def test_gram_topk_recovers_planted_subspace():
    d, k = 16, 3
    torch.manual_seed(0)
    U = _orthobasis(d, k, seed=1)                      # the "true" S(k,L)
    scales = torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64)
    # gradients that live in span(U) with the given per-direction energies
    G = torch.zeros(d, d, dtype=torch.float64)
    for j in range(k):
        # a block of n gradient rows all along U[:,j], summing to scales[j] energy
        rows = U[:, j].reshape(1, d) * scales[j].sqrt()
        CC.gram_update(G, rows)
    Q, evals = CC.topk_eigvecs(G, k)
    # eigenvalues match planted energies (descending)
    assert torch.allclose(evals, scales.sort(descending=True).values, atol=1e-9)
    # recovered subspace == planted subspace (overlap 1)
    ov, cos = CC.principal_angle_overlap(CC.orthonormalize(Q), U)
    assert ov > 1 - 1e-9, ov
    assert torch.allclose(cos, torch.ones(k, dtype=torch.float64), atol=1e-6)


def test_gram_update_is_additive_and_psd():
    d = 8
    g = torch.Generator().manual_seed(3)
    A = torch.randn(20, d, generator=g, dtype=torch.float64)
    B = torch.randn(13, d, generator=g, dtype=torch.float64)
    G1 = torch.zeros(d, d, dtype=torch.float64)
    CC.gram_update(G1, A); CC.gram_update(G1, B)
    G2 = torch.cat([A, B], 0).transpose(0, 1) @ torch.cat([A, B], 0)
    assert torch.allclose(G1, G2, atol=1e-9)
    _, evals = CC.topk_eigvecs(G1, d)
    assert float(evals.min()) >= -1e-9                 # PSD


# ---------------------------------------------------------------------------
# 2. Mean-ablation projector algebra + removed energy.
# ---------------------------------------------------------------------------
def test_mean_ablation_removes_span_to_hbar():
    d, k = 10, 2
    Q = _orthobasis(d, k, seed=2)
    hbar = torch.randn(d, dtype=torch.float64)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(5, d, generator=g, dtype=torch.float64)
    xp = CC.apply_mean_ablation(x, Q, hbar)
    # the span(Q) component of the ablated x equals that of hbar (component replaced by h̄'s)
    assert torch.allclose((xp - hbar) @ Q, torch.zeros(5, k, dtype=torch.float64), atol=1e-9)
    # the complement is untouched
    P_perp = torch.eye(d, dtype=torch.float64) - Q @ Q.transpose(0, 1)
    assert torch.allclose(xp @ P_perp, x @ P_perp, atol=1e-9)


def test_removed_energy_matches_projection():
    d, k = 12, 3
    Q = _orthobasis(d, k, seed=5)
    hbar = torch.zeros(d, dtype=torch.float64)
    g = torch.Generator().manual_seed(6)
    X = torch.randn(7, d, generator=g, dtype=torch.float64)
    e = CC.removed_energy(X, Q, hbar)
    assert abs(e - float(((X @ Q) ** 2).sum())) < 1e-9


def test_energy_matched_rank_roundtrip():
    d = 100
    trace_C = 500.0                                    # tr(C) over the activation set
    # a rank-7 random subspace removes E ≈ 7·tr/d in expectation → recover 7
    E_target = 7 * trace_C / d
    assert CC.energy_matched_rank(E_target, trace_C, d) == 7
    assert CC.energy_matched_rank(1e-9, trace_C, d) == 1          # clamp low
    assert CC.energy_matched_rank(1e9, trace_C, d) == d           # clamp high


def test_energy_matched_random_removes_matched_energy():
    # a random subspace at the energy-matched rank removes ≈ E_target on isotropic data
    d = 200
    g = torch.Generator().manual_seed(7)
    X = torch.randn(4000, d, generator=g, dtype=torch.float64)   # ~isotropic, tr(C)=Σ‖x‖²
    hbar = X.mean(0)
    trace_C = float((((X - hbar) ** 2).sum()).item())
    E_target = trace_C * 0.10                                     # remove ~10% of the energy
    Q, r_e = CC.energy_matched_random(d, E_target, trace_C, seed=11)
    e = CC.removed_energy(X, Q, hbar)
    assert 0.7 * E_target < e < 1.3 * E_target, (e, E_target, r_e)


# ---------------------------------------------------------------------------
# 3. Principal-angle overlap — rails + matched-random null centring.
# ---------------------------------------------------------------------------
def test_overlap_rails():
    d, k = 20, 4
    # A and B are disjoint orthonormal column-blocks of ONE orthonormal basis → mutually
    # orthogonal by construction (can't rely on randn(d,k)/(d,2k) sharing columns: row-major fill).
    full = _orthobasis(d, 2 * k, seed=8)
    A = full[:, :k].contiguous()
    B = full[:, k:2 * k].contiguous()
    ov_self, cos_self = CC.principal_angle_overlap(A, A)
    assert ov_self > 1 - 1e-9
    ov_orth, _ = CC.principal_angle_overlap(A, B)
    assert ov_orth < 1e-9, ov_orth


def test_matched_random_overlap_centres_on_analytic():
    d, kA, kB = 256, 4, 6
    samp = CC.matched_random_overlap(d, kA, kB, n_draws=400, seed=1)
    # analytic chance overlap ≈ max(kA,kB)/d for small k
    assert abs(float(samp.mean()) - max(kA, kB) / d) < 0.02, float(samp.mean())
    assert float(samp.min()) >= 0.0 and float(samp.max()) <= 1.0


# ---------------------------------------------------------------------------
# 4. The prefix-masked layer ablation hook (tuple + tensor outputs).
# ---------------------------------------------------------------------------
def test_layer_hook_tuple_and_prefix_mask():
    d, k, seq, npref = 8, 2, 6, 3
    Q = _orthobasis(d, k, seed=9)
    hbar = torch.randn(d, dtype=torch.float64)
    hook = CC.make_layer_ablation_hook(Q, hbar, n_prefix=npref)
    g = torch.Generator().manual_seed(10)
    hs = torch.randn(1, seq, d, generator=g, dtype=torch.float64)
    out = hook(None, None, (hs, "extra"))
    assert isinstance(out, tuple) and out[1] == "extra"
    hs2 = out[0]
    # prefix positions ablated (span(Q) component == h̄'s), suffix untouched
    assert torch.allclose((hs2[:, :npref] - hbar) @ Q, torch.zeros(1, npref, k, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(hs2[:, npref:], hs[:, npref:], atol=1e-9)
    # bare-tensor output path
    out_t = hook(None, None, hs)
    assert torch.is_tensor(out_t) and out_t.shape == hs.shape


def test_mean_replace_hook():
    d, seq = 6, 4
    hbar = torch.arange(d, dtype=torch.float64)
    hook = CC.make_mean_replace_hook(hbar, n_prefix=2)
    hs = torch.randn(1, seq, d, dtype=torch.float64)
    out = hook(None, None, (hs, "x"))
    assert isinstance(out, tuple) and out[1] == "x"
    assert torch.allclose(out[0][:, :2], hbar.expand(1, 2, d))       # prefix replaced by h̄
    assert torch.allclose(out[0][:, 2:], hs[:, 2:])                  # suffix untouched


def test_install_ablation_kinds():
    import torch.nn as nn
    d, k, seq = 6, 2, 3
    layers = nn.ModuleList([nn.Identity() for _ in range(4)])
    bases = {L: _orthobasis(d, 4, seed=100 + L) for L in range(4)}   # d×kmax(4) per layer
    hbars = {L: torch.zeros(d, dtype=torch.float64) for L in range(4)}
    x = torch.randn(1, seq, d, dtype=torch.float64)
    L = 1

    # identity → no handles, output unchanged
    hh = RO.install_ablation(layers, "identity", L, k, bases, hbars, d, seed=0)
    assert hh == []
    assert torch.allclose(layers[L](x), x)

    # saction → span(Q) removed (h̄=0 ⇒ projection to 0)
    hh = RO.install_ablation(layers, "saction", L, k, bases, hbars, d, seed=0)
    out = layers[L](x)
    for h in hh:
        h.remove()
    assert torch.allclose(out @ bases[L][:, :k], torch.zeros(1, seq, k, dtype=torch.float64), atol=1e-9)

    # positive (mean-replace, h̄=0) → output all zeros
    hh = RO.install_ablation(layers, "positive", L, k, bases, hbars, d, seed=0)
    out = layers[L](x)
    for h in hh:
        h.remove()
    assert torch.allclose(out, torch.zeros_like(x))

    # shuffle → uses a DIFFERENT layer's basis at L (structure broken); still removes rank-k
    hh = RO.install_ablation(layers, "shuffle", L, k, bases, hbars, d, seed=0)
    out = layers[L](x)
    for h in hh:
        h.remove()
    assert torch.allclose(out @ bases[(L + 2) % 4][:, :k], torch.zeros(1, seq, k, dtype=torch.float64), atol=1e-9)

    # random_energy → runs, removes some rank (needs E_target + trace_C)
    hh = RO.install_ablation(layers, "random_energy", L, k, bases, hbars, d, seed=0,
                             E_target=1.0, trace_C=float(d))
    out = layers[L](x)
    for h in hh:
        h.remove()
    assert out.shape == x.shape


def test_layer_hook_all_positions_when_no_prefix():
    d, k, seq = 8, 2, 5
    Q = _orthobasis(d, k, seed=12)
    hbar = torch.zeros(d, dtype=torch.float64)
    hook = CC.make_layer_ablation_hook(Q, hbar, n_prefix=None)
    hs = torch.randn(1, seq, d, dtype=torch.float64)
    hs2 = hook(None, None, hs)
    assert torch.allclose(hs2 @ Q, torch.zeros(1, seq, k, dtype=torch.float64), atol=1e-9)


# ---------------------------------------------------------------------------
# 5. Frozen §6 verdict predicates (the auditable decision rules).
# ---------------------------------------------------------------------------
def test_rf_control_and_dynamic_range_guard():
    assert abs(CC.rf_control(0.9, 1.0, 0.0) - 0.9) < 1e-12
    assert abs(CC.rf_control(0.0, 0.8, 0.0) - 0.0) < 1e-12
    # no dynamic range (intact ≈ floor) must RAISE (P7 halt, not a silent NaN)
    import pytest
    with pytest.raises(AssertionError):
        CC.rf_control(0.2, 0.25, 0.25)


def test_rf_cluster_bootstrap_rails():
    import numpy as np
    tasks = [f"t{i}" for i in range(8)]
    intact = {t: np.ones(5) for t in tasks}          # intact always succeeds
    ablate0 = {t: np.zeros(5) for t in tasks}        # ablation always fails
    r = CC.rf_cluster_bootstrap(ablate0, intact, floor=0.0, n_boot=300, seed=1)
    assert abs(r["rf"] - 0.0) < 1e-9 and r["ci"][0] <= 0.0 <= r["ci"][1] + 1e-9
    ablate1 = {t: np.ones(5) for t in tasks}         # ablation harmless
    r1 = CC.rf_cluster_bootstrap(ablate1, intact, floor=0.0, n_boot=300, seed=1)
    assert abs(r1["rf"] - 1.0) < 1e-9


def test_gap_cluster_bootstrap_decisive():
    import numpy as np
    tasks = [f"t{i}" for i in range(10)]
    intact = {t: np.ones(6) for t in tasks}
    saction = {t: np.zeros(6) for t in tasks}        # S_action ablation craters control (RF=0)
    randen = {t: np.ones(6) for t in tasks}          # energy-random preserves control (RF=1)
    g = CC.gap_cluster_bootstrap(saction, randen, intact, floor=0.0, n_boot=300, seed=2)
    assert abs(g["gap"] - 1.0) < 1e-9 and g["ci_lo"] > 0.3          # decisive: beats the null
    # null case: S_action ablation == energy-random → gap ~0, CI straddles 0
    g0 = CC.gap_cluster_bootstrap(intact, intact, intact, floor=0.0, n_boot=300, seed=2)
    assert abs(g0["gap"]) < 1e-9


def test_interface_verdict_rule():
    # crater + beats energy-null + monotone → H_interface
    assert CC.interface_verdict(0.1, 0.5, tau_low=0.3, delta=0.3, monotone_ok=True) == "H_interface"
    # no crater → H_distributed
    assert CC.interface_verdict(0.6, 0.5, 0.3, 0.3, True) == "H_distributed"
    # crater but gap doesn't beat energy-null (matched-random floors control too) → H_distributed
    assert CC.interface_verdict(0.1, 0.2, 0.3, 0.3, True) == "H_distributed"
    # crater + gap but non-monotone → H_distributed
    assert CC.interface_verdict(0.1, 0.5, 0.3, 0.3, False) == "H_distributed"


def test_pi05_ablate_transform():
    d, k, seq = 6, 2, 4
    Q = _orthobasis(d, k, seed=20); hbar = torch.zeros(d, dtype=torch.float64)
    prefix = torch.randn(1, seq, d, dtype=torch.float64); expert = torch.randn(1, 3, d, dtype=torch.float64)
    tf = PH.make_ablate_transform(L_target=5, Q=Q, hbar=hbar)
    out = tf(3, [prefix, expert])                       # non-target layer untouched
    assert out[0] is prefix and out[1] is expert
    out2 = tf(5, [prefix, expert])                      # target: prefix span(Q)→0 (h̄=0), expert intact
    assert torch.allclose(out2[0] @ Q, torch.zeros(1, seq, k, dtype=torch.float64), atol=1e-9)
    assert out2[1] is expert


def test_pi05_capture_and_meanreplace_transforms():
    d, seq = 5, 3
    store = {}
    h = torch.randn(1, seq, d, dtype=torch.float64, requires_grad=True)
    out = PH.make_capture_transform(store)(7, [h, "exp"])
    assert store[7] is h and out[0] is h and out[1] == "exp"
    hbars = {L: torch.arange(d, dtype=torch.float64) for L in range(3)}
    out2 = PH.make_mean_replace_transform(hbars)(1, [torch.randn(1, seq, d, dtype=torch.float64), "e"])
    assert torch.allclose(out2[0], hbars[1].expand(1, seq, d)) and out2[1] == "e"


def test_pi05_pre_hook_ablate_and_replace():
    d, k, seq = 6, 2, 4
    Q = _orthobasis(d, k, seed=21); hbar = torch.zeros(d, dtype=torch.float64)
    hs = torch.randn(1, seq, d, dtype=torch.float64)
    out = PH.make_pre_hook_ablate(Q, hbar)(None, (hs, "pos"))       # ablate the layer INPUT (args[0])
    assert torch.allclose(out[0] @ Q, torch.zeros(1, seq, k, dtype=torch.float64), atol=1e-9)
    assert out[1] == "pos"                                          # trailing args preserved
    hb = torch.arange(d, dtype=torch.float64)
    out2 = PH.make_pre_hook_replace(hb)(None, (hs,))
    assert torch.allclose(out2[0], hb.expand(1, seq, d))


def test_pi05_mean_accum_transform():
    d = 4
    acc = {0: {"sum": torch.zeros(d, dtype=torch.float64), "sq": 0.0, "n": 0}}
    h = torch.randn(1, 3, d, dtype=torch.float64)
    PH.make_mean_accum_transform(acc, d)(0, [h, None])
    assert acc[0]["n"] == 3
    assert torch.allclose(acc[0]["sum"], h.reshape(-1, d).sum(0), atol=1e-9)
    assert abs(acc[0]["sq"] - float((h.reshape(-1, d) ** 2).sum())) < 1e-9


def test_disjoint_verdict_rule():
    # overlap below null AND text-ablation preserves control → H_disjoint
    assert CC.disjoint_verdict(0.02, 0.10, 0.9, tau_high=0.7) == "H_disjoint"
    # overlap above null → H_shared
    assert CC.disjoint_verdict(0.30, 0.10, 0.9, 0.7) == "H_shared"
    # text-ablation craters control → H_shared
    assert CC.disjoint_verdict(0.02, 0.10, 0.4, 0.7) == "H_shared"


# ---------------------------------------------------------------------------
# 6. run-orchestration wiring (ctrliface_run) — the activation-gradient capture path,
#    validated weight-free against manual autograd on a toy FROZEN nn.Module.
# ---------------------------------------------------------------------------
def _toy_frozen_net(d, seed=0):
    import torch.nn as nn
    torch.manual_seed(seed)
    lin1, lin2 = nn.Linear(d, d).double(), nn.Linear(d, d).double()
    net = nn.Sequential(lin1, lin2)
    for p in net.parameters():
        p.requires_grad_(False)                    # backbone FROZEN — the real regime
    return net, lin1, lin2


def test_accumulate_gram_matches_autograd_on_frozen_net():
    d = 6
    net, lin1, lin2 = _toy_frozen_net(d)
    y = torch.randn(1, 3, d, dtype=torch.float64)
    xs = [torch.randn(1, 3, d, dtype=torch.float64) for _ in range(2)]

    def loss_fn(x):
        return ((net(x) - y) ** 2).sum()

    # capture-hook path (grad reaches the frozen layer's OUTPUT even though params are frozen)
    G, npos = CR.accumulate_gram(net, lin1, loss_fn, xs, d, n_prefix=None)
    assert npos == 2 * 3

    # manual reference: anchor grad at h = lin1(x), backprop dL/dh
    G_ref = torch.zeros(d, d, dtype=torch.float64)
    for x in xs:
        h = lin1(x).detach().requires_grad_(True)
        (((lin2(h) - y) ** 2).sum()).backward()
        g = h.grad.reshape(-1, d).double()
        G_ref += g.transpose(0, 1) @ g
    assert torch.allclose(G, G_ref, atol=1e-9), (G - G_ref).abs().max()


def test_accumulate_gram_multi_matches_per_layer():
    import torch.nn as nn
    d = 5
    torch.manual_seed(7)
    l0, l1, l2 = nn.Linear(d, d).double(), nn.Linear(d, d).double(), nn.Linear(d, d).double()
    net = nn.Sequential(l0, l1, l2)
    for p in net.parameters():
        p.requires_grad_(False)
    y = torch.randn(1, 3, d, dtype=torch.float64)
    xs = [torch.randn(1, 3, d, dtype=torch.float64) for _ in range(2)]

    def loss_fn(x):
        return ((net(x) - y) ** 2).sum()

    # multi-layer: both l0 and l1 grads from ONE backward per batch
    Gm, npos = CR.accumulate_gram_multi(net, {"L0": l0, "L1": l1}, loss_fn, xs, d)
    # per-layer reference
    G0, _ = CR.accumulate_gram(net, l0, loss_fn, xs, d)
    G1, _ = CR.accumulate_gram(net, l1, loss_fn, xs, d)
    assert torch.allclose(Gm["L0"], G0, atol=1e-9)
    assert torch.allclose(Gm["L1"], G1, atol=1e-9)
    assert npos == 2 * 3


def test_harvest_multi_matches_single():
    import torch.nn as nn
    d = 4
    net, lin1, _ = _toy_frozen_net(d, seed=8)
    xs = [torch.randn(1, 3, d, dtype=torch.float64) for _ in range(2)]
    res, n = CR.harvest_hbar_trace_multi({"L0": lin1}, lambda x: net(x), xs, d)
    hbar_s, trace_s, _ = CR.harvest_hbar_trace(lin1, lambda x: net(x), xs, d)
    assert torch.allclose(res["L0"][0], hbar_s, atol=1e-9)
    assert abs(res["L0"][1] - trace_s) < 1e-6


def test_accumulate_gram_prefix_mask():
    d = 5
    net, lin1, lin2 = _toy_frozen_net(d, seed=1)
    y = torch.randn(1, 4, d, dtype=torch.float64)
    xs = [torch.randn(1, 4, d, dtype=torch.float64)]

    def loss_fn(x):
        return ((net(x) - y) ** 2).sum()

    G, npos = CR.accumulate_gram(net, lin1, loss_fn, xs, d, n_prefix=2)
    assert npos == 2                              # only the first 2 positions counted


def test_harvest_hbar_trace_matches_manual():
    d = 4
    net, lin1, _ = _toy_frozen_net(d, seed=2)
    xs = [torch.randn(1, 3, d, dtype=torch.float64) for _ in range(3)]
    hbar, trace_C, n = CR.harvest_hbar_trace(lin1, lambda x: net(x), xs, d, n_prefix=None)
    H = torch.cat([lin1(x).reshape(-1, d) for x in xs], 0).double()
    assert n == H.shape[0]
    assert torch.allclose(hbar, H.mean(0), atol=1e-9)
    assert abs(trace_C - float(((H - H.mean(0)) ** 2).sum())) < 1e-6


def test_proxy_deviation_positive_for_relevant_direction():
    d = 5
    net, lin1, _ = _toy_frozen_net(d, seed=3)
    xs = [torch.randn(1, 3, d, dtype=torch.float64) for _ in range(2)]
    hbar, _, _ = CR.harvest_hbar_trace(lin1, lambda x: net(x), xs, d)
    Q = _orthobasis(d, 2, seed=9)                 # a real 2-dim subspace of the residual
    D, per = CR.proxy_deviation(lin1, lambda x: net(x), xs, Q, hbar, n_prefix=None)
    assert D > 0 and len(per) == 2 and all(p >= 0 for p in per)
