"""Weight-free tests for the DARWIN training harness (§9 step 2, §13 NEW row).

Run: PYTHONPATH=. python3 infra/pg/tests/test_darwin_train.py

These run on tiny torch modules with no model and no GPU. What they pin is the arithmetic the
retention arms' validity rests on: that the optimizer really is exact fp32 Adam, that the
subspace restriction really confines the update, and that the anchor and its EWC-lite control
differ in the way §4/W1-U's REFUTER assumes they do.
"""
import numpy as np
import torch

from infra.pg import darwin_core as D
from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H


def _lin(n_in=16, n_out=8, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(n_in, n_out, bias=False).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# ExactAdam.
# ---------------------------------------------------------------------------
def test_exact_adam_matches_torch_adamw_in_fp32():
    """The whole point of writing it out: it must BE AdamW, not merely resemble it."""
    torch.manual_seed(0)
    g = [torch.randn(8, 16) for _ in range(5)]

    ref = torch.nn.Parameter(torch.zeros(8, 16))
    opt = torch.optim.AdamW([ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for gi in g:
        opt.zero_grad()
        ref.grad = gi.clone()
        opt.step()

    p = torch.nn.Parameter(torch.zeros(8, 16))
    mine = T.ExactAdam([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for gi in g:
        mine.zero_grad()
        p.grad = gi.clone()
        mine.step()

    assert torch.allclose(ref.detach(), p.detach(), atol=1e-6), \
        (ref.detach() - p.detach()).abs().max().item()


def test_exact_adam_keeps_fp32_state_behind_a_bf16_parameter():
    m = _lin()
    opt = T.ExactAdam(list(m.parameters()), lr=1e-2)
    assert all(x.dtype == torch.float32 for x in opt.master + opt.m + opt.v)
    assert m.weight.dtype == torch.bfloat16
    # 3 fp32 tensors per parameter: master + m + v.
    assert opt.state_bytes == m.weight.numel() * 12
    m.weight.grad = torch.ones_like(m.weight)
    before = m.weight.detach().clone().to(torch.float32)
    opt.step()
    after = m.weight.detach().to(torch.float32)
    assert not torch.allclose(before, after)                    # it actually moved
    assert m.weight.dtype == torch.bfloat16                     # ...and stayed bf16


def test_exact_adam_is_not_an_approximate_optimizer_by_policy():
    """§0.2.7: the arms this harness serves may not use 8-bit/approximate Adam."""
    for arm in ("W1-T", "W1-U"):
        r = D.check_optimizer(arm, "darwin_train.ExactAdam", offload_exact_fp32=False)
        assert not r["is_approximate"] and r["optimizer_mechanics_arm"]


# ---------------------------------------------------------------------------
# SubspaceProjector — W1-T's restriction.
# ---------------------------------------------------------------------------
def test_subspace_projector_confines_the_update_to_the_basis():
    n_in, n_out, k = 32, 8, 4
    m = _lin(n_in, n_out).to(torch.float32)
    U = H.haar_subspace(n_in, k, 3)
    proj = T.SubspaceProjector({("q_proj", 0): U}, device="cpu")
    m.weight.grad = torch.randn(n_out, n_in)
    raw = m.weight.grad.clone()
    proj.apply({("q_proj", 0): m})
    g = m.weight.grad
    Ut = torch.as_tensor(U, dtype=torch.float32)
    # The projected gradient lives entirely in span(U): its complement component is zero.
    comp = g - (g @ Ut) @ Ut.T
    assert float(comp.abs().max()) < 1e-5, float(comp.abs().max())
    assert not torch.allclose(raw, g)                            # it really changed something
    # Idempotent: projecting twice is projecting once.
    once = g.clone()
    proj.apply({("q_proj", 0): m})
    assert torch.allclose(once, m.weight.grad, atol=1e-6)


def test_subspace_projector_leaves_unlisted_cells_alone():
    m = _lin(32, 8).to(torch.float32)
    proj = T.SubspaceProjector({("q_proj", 0): H.haar_subspace(32, 4, 1)}, device="cpu")
    m.weight.grad = torch.randn(8, 32)
    raw = m.weight.grad.clone()
    proj.apply({("o_proj", 5): m})                               # key not in the basis map
    assert torch.allclose(raw, m.weight.grad)


# ---------------------------------------------------------------------------
# The anchor and its control must differ in the way the REFUTER assumes.
# ---------------------------------------------------------------------------
def test_ewc_lite_penalises_weight_drift_and_is_zero_at_the_start():
    n_in, n_out, k = 24, 6, 3
    m = _lin(n_in, n_out).to(torch.float32)
    key = ("q_proj", 0)
    U = H.haar_subspace(n_in, k, 7)
    ewc = T.EwcLiteCore({key: m}, {key: U}, device="cpu", lam=1.0)
    assert float(ewc.penalty().detach()) < 1e-12                          # W == W0 at construction
    with torch.no_grad():                                        # move INSIDE the core subspace
        Ut = torch.as_tensor(U, dtype=torch.float32)
        m.weight.add_((torch.randn(n_out, k) @ Ut.T) * 0.1)
    moved = float(ewc.penalty().detach())
    assert moved > 1e-6
    # A move in the COMPLEMENT must not be penalised — the control is core-specific too.
    m2 = _lin(n_in, n_out).to(torch.float32)
    ewc2 = T.EwcLiteCore({key: m2}, {key: U}, device="cpu", lam=1.0)
    with torch.no_grad():
        Ut = torch.as_tensor(U, dtype=torch.float32)
        d = torch.randn(n_out, n_in)
        m2.weight.add_((d - (d @ Ut) @ Ut.T) * 0.1)
    assert float(ewc2.penalty().detach()) < 1e-6


def test_ewc_lite_scales_linearly_with_lambda():
    key = ("q_proj", 0)
    U = H.haar_subspace(24, 3, 11)
    vals = []
    for lam in (0.5, 1.0, 2.0):
        m = _lin(24, 6).to(torch.float32)
        e = T.EwcLiteCore({key: m}, {key: U}, device="cpu", lam=lam)
        with torch.no_grad():
            torch.manual_seed(4)
            Ut = torch.as_tensor(U, dtype=torch.float32)
            m.weight.add_((torch.randn(6, 3) @ Ut.T) * 0.1)
        vals.append(float(e.penalty().detach()))
    assert abs(vals[1] / vals[0] - 2.0) < 1e-4
    assert abs(vals[2] / vals[1] - 2.0) < 1e-4


def test_core_activation_anchor_is_zero_against_its_own_reference():
    """The anchor compares ACTIVATIONS AT MODULE INPUTS — the site the banks are defined in
    (humboldt_core's site discipline: U_l is an INPUT-space basis). A weight change is therefore
    visible to the anchor only through the activations it feeds DOWNSTREAM, so the test needs a
    stack: perturbing layer 0 moves layer 1's input. On a single module the penalty is
    identically zero by construction, which is correct, not a bug."""
    k0, k1 = ("q_proj", 0), ("q_proj", 1)
    m0, m1 = _lin(16, 16, seed=1).to(torch.float32), _lin(16, 8, seed=2).to(torch.float32)
    U = {k0: H.haar_subspace(16, 4, 5), k1: H.haar_subspace(16, 4, 6)}
    x = torch.randn(6, 16)
    anchor = T.CoreActivationAnchor({k0: m0, k1: m1}, U, device="cpu", lam=1.0)
    with anchor:
        anchor.record_reference()
        m1(m0(x))
        anchor.record_current()
        m1(m0(x))
        assert float(anchor.penalty().detach()) < 1e-12
        # Perturbing layer 0 changes layer 1's INPUT, which the anchor does see.
        with torch.no_grad():
            m0.weight.add_(torch.randn_like(m0.weight) * 0.5)
        anchor.record_current()
        m1(m0(x))
        assert float(anchor.penalty().detach()) > 1e-6


def test_anchor_and_ewc_are_different_objects():
    """A weight move that is invisible to the activation anchor on a given batch is still seen by
    EWC-lite — the two penalties are not the same regulariser wearing two names, which is what
    §4/W1-U's REFUTER hinges on."""
    key = ("q_proj", 0)
    n_in, k = 16, 4
    U = H.haar_subspace(n_in, k, 9)
    Ut = torch.as_tensor(U, dtype=torch.float32)
    m = _lin(n_in, 8).to(torch.float32)
    # A batch that lies ENTIRELY in the complement of U: core-projected activations are ~0, so a
    # weight change in the core columns cannot show up in this batch's activations.
    xc = torch.randn(6, n_in)
    xc = xc - (xc @ Ut) @ Ut.T
    ewc = T.EwcLiteCore({key: m}, {key: U}, device="cpu", lam=1.0)
    anchor = T.CoreActivationAnchor({key: m}, {key: U}, device="cpu", lam=1.0)
    with anchor:
        anchor.record_reference()
        m(xc)
        with torch.no_grad():
            m.weight.add_((torch.randn(8, k) @ Ut.T) * 0.5)
        anchor.record_current()
        m(xc)
        a_pen = float(anchor.penalty().detach())
    e_pen = float(ewc.penalty().detach())
    assert e_pen > 1e-6, "EWC-lite must see a core weight move"
    assert a_pen < e_pen, (a_pen, e_pen)


# ---------------------------------------------------------------------------
# Scope + dose grid.
# ---------------------------------------------------------------------------
def test_lam_ladder_is_four_log_spaced_doses():
    lad = T.lam_ladder()
    assert len(lad) == 4
    D.check_dose_grid(lad, arm="W1-U")                            # §3 compliance, checked here


def test_trainable_scope_is_stated_and_symmetric():
    class _Blk:
        def __init__(self):
            self.self_attn = type("A", (), {})()
            self.self_attn.q_proj = _lin()
            self.self_attn.o_proj = _lin()
    layers = [_Blk() for _ in range(3)]
    mods = T.trainable_modules(layers)
    assert set(mods) == {(t, L) for t in ("q_proj", "o_proj") for L in range(3)}
    assert T.TRAINABLE_TYPES == ("q_proj", "o_proj")


def test_core_energy_gauge_is_a_readout_not_a_side_effect():
    key = ("q_proj", 0)
    m = _lin(16, 8).to(torch.float32)
    U = H.haar_subspace(16, 4, 2)
    before = m.weight.detach().clone()
    g = T.core_energy_gauge({key: m}, {key: U})
    assert torch.allclose(before, m.weight.detach())              # no mutation
    assert 0.0 <= g[key] <= 1.0
    assert abs(g[key] - 4 / 16) < 0.25                            # Haar reference is k/n


# ---------------------------------------------------------------------------
# LoRA — W1-T's incumbent.
# ---------------------------------------------------------------------------
def test_lora_starts_as_an_exact_noop():
    """B is zero-init, so at step 0 the model must be bit-identical to the base — otherwise the
    comparison against a full fine-tune is unfair from the first step."""
    key = ("q_proj", 0)
    m = _lin(16, 8).to(torch.float32)
    x = torch.randn(4, 16)
    before = m(x).clone()
    lora = T.LoRAAdapter({key: m}, rank=4, alpha=8.0, device="cpu")
    with lora:
        assert torch.allclose(before, m(x), atol=1e-7)
        # ...and once B moves, the output changes.
        with torch.no_grad():
            lora.B[key].add_(torch.randn_like(lora.B[key]))
        assert not torch.allclose(before, m(x), atol=1e-5)
    assert torch.allclose(before, m(x), atol=1e-7)          # hook removed on exit


def test_lora_scale_is_alpha_over_rank():
    key = ("q_proj", 0)
    m = _lin(16, 8).to(torch.float32)
    for r, alpha in ((4, 8.0), (8, 8.0), (32, 64.0)):
        lora = T.LoRAAdapter({key: m}, rank=r, alpha=alpha, device="cpu")
        assert abs(lora.scale - alpha / r) < 1e-12
    # Parameter count is r*(in+out) per matrix — the number the matched-memory rule compares.
    lora = T.LoRAAdapter({key: m}, rank=4, alpha=8.0, device="cpu")
    assert lora.n_params == 4 * (16 + 8)


def test_lora_gradients_reach_only_the_adapter():
    key = ("q_proj", 0)
    m = _lin(16, 8).to(torch.float32)
    m.weight.requires_grad_(False)
    lora = T.LoRAAdapter({key: m}, rank=4, alpha=8.0, device="cpu")
    with lora:
        loss = m(torch.randn(4, 16)).pow(2).sum()
        loss.backward()
    assert m.weight.grad is None                              # the base stays frozen
    assert lora.A[key].grad is not None and lora.B[key].grad is not None
    assert float(lora.B[key].grad.abs().sum()) > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
