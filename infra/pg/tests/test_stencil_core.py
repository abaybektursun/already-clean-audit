"""STENCIL weight-free dry-run logic gate (spec §9 step 2 / §10.1). Tiny tensors, no model —
seconds on CPU. Proves: the fixed-basis pin confines ΔW to span(U_policy) (energy_fraction=1 in-span,
0 off); zero-delta init is byte-identity; α=r ⇒ scale 1; the gradient-projection partition identity;
iso-gain interpolation on a synthetic trajectory; degenerate-basis fallback; the §6 adjudicator's
every branch. Runs under the repo's pytest (infra/pg/tests). No try/except (CLAUDE.md)."""
import numpy as np
import torch

from infra.pg import comwrite_core as CW
from infra.pg import stencil_core as SC


def _bank(d=32, seed=0):
    """A tiny per-layer bank: a random Q eigenbasis, a DISTINCT Σ eigenbasis, a u_wr, d."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    W, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return {"Q_evecs": Q, "sigma_evecs": S, "u_wr": W[:, :16], "d": d}


# ---------------------------------------------------------------------------
# 1. Fixed-basis pin: zero-delta init, scale=α/r, ΔW confined to span(U_policy).
# ---------------------------------------------------------------------------
def test_zero_delta_init_is_identity():
    d_in, d_out, r = 24, 32, 5
    base = torch.nn.Linear(d_in, d_out, bias=False)
    U = torch.linalg.qr(torch.randn(d_out, d_out, dtype=torch.float64))[0][:, :r]
    adin = SC.make_fixed_adapter(base, U, r, alpha=r)
    x = torch.randn(4, d_in)
    assert torch.allclose(adin(x), base(x), atol=0, rtol=0), "zero-delta init must be byte-identity"
    assert not adin.A.requires_grad, "A must be a frozen (buffer) factor"
    assert adin.B.requires_grad and float(adin.B.abs().max()) == 0.0, "B trained, init exactly 0"


def test_scale_is_alpha_over_r():
    base = torch.nn.Linear(10, 12, bias=False)
    U = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))[0][:, :4]
    assert abs(SC.make_fixed_adapter(base, U, 4, alpha=4).scale - 1.0) < 1e-12, "α=r ⇒ scale 1"
    assert abs(SC.make_fixed_adapter(base, U, 4, alpha=8).scale - 2.0) < 1e-12, "α=2r ⇒ scale 2"


def test_delta_confined_to_policy_span():
    """ΔW = scale·A·B with A=U ⇒ ALL Frobenius energy in span(U) on the output side; ZERO off it."""
    d_in, d_out, r = 20, 32, 6
    base = torch.nn.Linear(d_in, d_out, bias=False)
    U = torch.linalg.qr(torch.randn(d_out, d_out, dtype=torch.float64))[0][:, :r]
    ad = SC.make_fixed_adapter(base, U, r, alpha=r)
    ad.B.data = torch.randn(r, d_in)                              # a nonzero trained B
    dW = (ad.scale * (ad.A.double() @ ad.B.detach().double())).numpy()   # (out, in); A is a float32 buffer
    frac_in = CW.energy_fraction(dW, U.numpy(), "o_proj")         # output-side projector UUᵀ·
    assert abs(frac_in - 1.0) < 1e-5, f"in-span energy {frac_in} != 1 (float32 A tol)"
    # off-span (complement) energy must be ~0 (float32 A ⇒ ~1e-7 residual)
    Uc = torch.linalg.qr(torch.cat([U, torch.randn(d_out, d_out - r, dtype=torch.float64)], 1))[0][:, r:]
    frac_off = CW.energy_fraction(dW, Uc.numpy(), "o_proj")
    assert frac_off < 1e-6, f"off-span energy {frac_off} != 0"


# ---------------------------------------------------------------------------
# 2. attach_fixed_adapters scope + trainability on a mock decoder tree.
# ---------------------------------------------------------------------------
class _Attn(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.o_proj = torch.nn.Linear(d, d, bias=False)


class _MLP(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.up_proj = torch.nn.Linear(d, h, bias=False)
        self.down_proj = torch.nn.Linear(h, d, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d, h)


class _Decoder(torch.nn.Module):
    def __init__(self, n, d, h):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(d, h) for _ in range(n)])


def test_attach_scope_and_trainability():
    d, h, n, r = 32, 40, 3, 5
    lm = _Decoder(n, d, h)
    banks = {L: _bank(d, seed=L) for L in range(n)}
    info = SC.attach_fixed_adapters(lm, "susceptibility", banks, rank=r, alpha=r, seed=SC.SEED)
    # exactly o_proj + down_proj across n layers → 2n targets
    assert info["n_modules"] == 2 * n, info["n_modules"]
    assert all(nm.rsplit(".", 1)[-1] in SC.TARGET_SUFFIXES for nm in info["names"])
    # only B params trainable; q_proj/up_proj untouched (still plain Linear, frozen)
    trainable = [nm for nm, p in lm.named_parameters() if p.requires_grad]
    assert trainable and all(nm.endswith(".B") for nm in trainable), trainable
    assert not info["degenerate"], "clean banks must not trigger fallback"
    # the pinned A at layer L must equal that layer's top-r Q eigvecs (o/down share the layer basis)
    o0 = lm.layers[0].self_attn.o_proj
    Uexp = banks[0]["Q_evecs"][:, :r].float()
    assert torch.allclose(o0.A, Uexp, atol=1e-6), "A not pinned to the layer's top-r Q basis"


# ---------------------------------------------------------------------------
# 3. Policy-basis builders: orthonormality + arm semantics.
# ---------------------------------------------------------------------------
def test_policy_bases_orthonormal_and_semantic():
    b = _bank(32, seed=7)
    for arm in SC.FIXED_ARMS:
        U = SC.build_policy_basis(arm, b, rank=6, seed=3)
        g = U.transpose(0, 1) @ U
        assert float((g - torch.eye(6, dtype=torch.float64)).abs().max()) < 1e-8, f"{arm} not orthonormal"
    # susceptibility = top-r Q eigvecs exactly
    assert torch.allclose(SC.build_policy_basis("susceptibility", b, 6), b["Q_evecs"][:, :6])
    # energy = top-r Σ eigvecs exactly
    assert torch.allclose(SC.build_policy_basis("energy", b, 6), b["sigma_evecs"][:, :6])
    # shuffle = a column subset of Q (spanned by Q's columns) but NOT the top-r ordering
    Ush = SC.build_policy_basis("susceptibility_shuffle", b, 6, seed=1)
    overlap_top = float(CW.subspace_overlap(Ush.numpy(), b["Q_evecs"][:, :6].numpy()))
    assert overlap_top < 0.999, "shuffle must not coincide with the top-r block"


def test_random_arm_reproducible_and_distinct():
    b = _bank(32, seed=2)
    u1 = SC.random_orthonormal(32, 5, seed=11)
    u2 = SC.random_orthonormal(32, 5, seed=11)
    u3 = SC.random_orthonormal(32, 5, seed=12)
    assert torch.allclose(u1, u2), "random basis must be deterministic in seed"
    assert not torch.allclose(u1, u3), "different seed → different basis"


def test_degenerate_basis_fallback():
    """A NaN-poisoned Q bank at one layer → that cell falls back to a random orthonormal basis + flag."""
    d, h, n, r = 24, 30, 2, 4
    lm = _Decoder(n, d, h)
    banks = {L: _bank(d, seed=L) for L in range(n)}
    banks[1]["Q_evecs"] = banks[1]["Q_evecs"].clone()
    banks[1]["Q_evecs"][:, 0] = float("nan")                      # poison layer 1's top eigvec
    info = SC.attach_fixed_adapters(lm, "susceptibility", banks, rank=r, alpha=r)
    assert any(".1." in nm for nm in info["degenerate"]), "layer-1 cells must fall back + be flagged"
    # the fallback basis is finite orthonormal
    o1 = lm.layers[1].self_attn.o_proj
    assert torch.isfinite(o1.A).all()


# ---------------------------------------------------------------------------
# 4. Gradient-projection partition identity (Arm O; comwrite_core).
# ---------------------------------------------------------------------------
def test_grad_projection_partition():
    # project_grad_residual computes in fp32 regardless of g's dtype → use fp32 g so the partition
    # identity holds to fp32 precision (not double).
    out_d, in_d, k = 32, 20, 6
    g = torch.randn(out_d, in_d, dtype=torch.float32)
    U = torch.linalg.qr(torch.randn(out_d, out_d, dtype=torch.float32))[0][:, :k]
    into = CW.project_grad_residual(g, U, "o_proj", "into")
    off = CW.project_grad_residual(g, U, "o_proj", "off")
    lhs = float((into ** 2).sum() + (off ** 2).sum())
    rhs = float((g ** 2).sum())
    assert abs(lhs - rhs) / rhs < 1e-5, f"‖into‖²+‖off‖² ({lhs}) != ‖g‖² ({rhs})"
    # 'into' energy lands entirely in span(U) on the output side
    assert abs(CW.energy_fraction(into.numpy(), U.numpy(), "o_proj") - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 5. Composite forgetting + iso-gain interpolation.
# ---------------------------------------------------------------------------
def test_composite_forgetting_polarity():
    base = {"mmlu_broad": 0.40, "vqav2": 0.60, "coco_cider": 1.00, "wikitext_ppl": 22.0}
    adapted = {"mmlu_broad": 0.36, "vqav2": 0.54, "coco_cider": 0.90, "wikitext_ppl": 24.2}
    w = {"mmlu_broad": 1.0, "vqav2": 1.0, "coco_cider": 1.0, "wikitext_ppl": 1.0}
    r = SC.composite_forgetting(base, adapted, w)
    # each component degraded by exactly 10% relative → composite 0.10
    for c, fc in r["per_component"].items():
        assert abs(fc - 0.10) < 1e-9, (c, fc)
    assert abs(r["composite"] - 0.10) < 1e-9
    # an IMPROVED component → negative forgetting (no absolute-value hiding of sign)
    adapted2 = {**adapted, "mmlu_broad": 0.44}
    assert SC.composite_forgetting(base, adapted2, w)["per_component"]["mmlu_broad"] < 0


def test_iso_gain_interpolation():
    gains = [0.0, 0.1, 0.25, 0.4]
    forgets = [0.0, 0.05, 0.20, 0.5]
    # exact node
    assert abs(SC.interp_forget_at_gain(gains, forgets, 0.25) - 0.20) < 1e-12
    # linear midpoint between 0.1→0.25
    mid = SC.interp_forget_at_gain(gains, forgets, 0.175)
    assert abs(mid - (0.05 + 0.5 * (0.20 - 0.05))) < 1e-9
    # outside the arm's reach → None (excluded from the paired primary)
    assert SC.interp_forget_at_gain(gains, forgets, 0.5) is None
    assert SC.interp_forget_at_gain(gains, forgets, -0.1) is None


def test_common_tau_is_min_of_maxes():
    g = {"a": [0.0, 0.3, 0.42], "b": [0.1, 0.35], "c": [0.05, 0.38]}
    assert abs(SC.common_tau(g, quantile=1.0) - 0.35) < 1e-12       # min over arms of each max
    assert abs(SC.common_tau(g, quantile=0.9) - 0.315) < 1e-12


# ---------------------------------------------------------------------------
# 6. Natural-write census + contrast-range gate.
# ---------------------------------------------------------------------------
def test_natural_write_census_captures_planted_subspace():
    b = _bank(32, seed=5)
    r = 6
    U_en = b["sigma_evecs"][:, :r]                                 # plant ΔW entirely in energy span
    dW = (U_en @ torch.randn(r, 18, dtype=torch.float64)).numpy()
    cen = SC.natural_write_census(dW, b, rank=r, mtype="down_proj", seed=1)
    assert abs(cen["energy"] - 1.0) < 1e-9, cen                    # energy arm captures all of it
    assert cen["susceptibility"] < 0.99                            # a distinct subspace captures less
    assert abs(cen["chance_floor"] - r / b["d"]) < 1e-12


def test_contrast_range_reads_overlap():
    b = _bank(64, seed=9)
    cr = SC.contrast_range(b, rank=8)
    assert 0.0 <= cr["overlap"] <= 1.0 and cr["haar_floor"] == 8 / 64
    # a random Q vs random Σ (independent QR) sit near the Haar floor, far from identical span
    assert cr["overlap"] < 0.6, cr


# ---------------------------------------------------------------------------
# 7. adjudicate_stencil — every frozen branch fires on the right inputs.
# ---------------------------------------------------------------------------
def _pe(lo, hi=None, E=None):
    """A minimal paired-excess-shaped dict."""
    return {"ci_lo": lo, "ci_hi": hi if hi is not None else lo + 0.1, "E": E if E is not None else lo + 0.05}


def test_adjudicate_void_no_range():
    v = SC.adjudicate_stencil(E_adapt=_pe(0.1), sus_vs_random=_pe(0.1), sus_vs_shuffle=_pe(0.1),
                              gain_range_ok=False, forget_range_ok=True,
                              random_saturates_frontier=False, free_strictly_dominates=False,
                              structure_beats_random=True)
    assert v["verdict"] == "VOID-no-range"


def test_adjudicate_subspace_agnostic():
    v = SC.adjudicate_stencil(E_adapt=_pe(0.1), sus_vs_random=_pe(-0.01), sus_vs_shuffle=_pe(0.1),
                              gain_range_ok=True, forget_range_ok=True,
                              random_saturates_frontier=True, free_strictly_dominates=False,
                              structure_beats_random=False)
    assert v["verdict"] == "H_subspace-agnostic"


def test_adjudicate_free_dominates():
    v = SC.adjudicate_stencil(E_adapt=_pe(0.1), sus_vs_random=_pe(0.1), sus_vs_shuffle=_pe(0.1),
                              gain_range_ok=True, forget_range_ok=True,
                              random_saturates_frontier=False, free_strictly_dominates=True,
                              structure_beats_random=True)
    assert v["verdict"] == "H_free-dominates"


def test_adjudicate_prescriptive_needs_all_three():
    common = dict(gain_range_ok=True, forget_range_ok=True, random_saturates_frontier=False,
                  free_strictly_dominates=False, structure_beats_random=True)
    win = SC.adjudicate_stencil(E_adapt=_pe(0.02), sus_vs_random=_pe(0.03), sus_vs_shuffle=_pe(0.01),
                                **common)
    assert win["verdict"] == "H_prescriptive"
    # drop ANY of the three killers → energy-suffices (cannot be rescued by the others)
    for kill in ("E_adapt", "sus_vs_random", "sus_vs_shuffle"):
        kw = dict(E_adapt=_pe(0.02), sus_vs_random=_pe(0.03), sus_vs_shuffle=_pe(0.01))
        kw[kill] = _pe(-0.01)
        v = SC.adjudicate_stencil(**kw, **common)
        assert v["verdict"] == "H_energy-suffices", (kill, v["verdict"])


def test_adjudicate_energy_suffices_when_tie():
    v = SC.adjudicate_stencil(E_adapt=_pe(-0.02), sus_vs_random=_pe(0.05), sus_vs_shuffle=_pe(0.05),
                              gain_range_ok=True, forget_range_ok=True,
                              random_saturates_frontier=False, free_strictly_dominates=False,
                              structure_beats_random=True)
    assert v["verdict"] == "H_energy-suffices"


# ---------------------------------------------------------------------------
# 8. Reference-arm attach paths (free-LoRA scope; full-FT decoder-block scope) + dispatch.
# ---------------------------------------------------------------------------
def test_free_lora_scope_and_trainability():
    d, h, n, r = 32, 40, 2, 5
    lm = _Decoder(n, d, h)
    info = SC.attach_free_lora(lm, rank=r, alpha=r, seed=SC.SEED)
    assert info["n_modules"] == 2 * n and info["arm"] == "free_lora"
    # both A and B trainable (unconstrained low-rank), on o/down only
    trainable = [nm for nm, p in lm.named_parameters() if p.requires_grad]
    assert trainable and all(nm.endswith(".A") or nm.endswith(".B") for nm in trainable), trainable
    assert all(any(t in nm for t in SC.TARGET_SUFFIXES) for nm in trainable)


def test_full_ft_unfreezes_decoder_blocks_only():
    d, h, n = 24, 30, 3
    lm = _Decoder(n, d, h)
    info = SC.mark_full_ft_trainable(lm)
    trainable = {nm for nm, p in lm.named_parameters() if p.requires_grad}
    # every decoder-block weight (q/o/up/down) unfrozen; count = 4 linears × n layers
    assert len(trainable) == 4 * n, (len(trainable), 4 * n)
    import re
    assert all(re.search(r"layers\.\d+\.", nm) for nm in trainable)


def test_attach_arm_dispatch():
    d, h, n, r = 32, 40, 2, 4
    banks = {L: _bank(d, seed=L) for L in range(n)}
    for arm in SC.ARMS:
        lm = _Decoder(n, d, h)
        info = SC.attach_arm(lm, arm, banks, rank=r, alpha=r)
        assert info["arm"] == arm
        assert any(p.requires_grad for p in lm.parameters()), f"{arm}: nothing trainable"


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    """Run every dry-run test; return the pass count (the offline gate asserts == len(ALL))."""
    for fn in ALL:
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nALL {len(ALL)} STENCIL dry-run logic-gate tests PASSED", flush=True)
    return len(ALL)


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
