"""CTRLIFACE run-orchestration — the (still lerobot-agnostic) glue between the pure core
(ctrliface_core) and the deployed policy. Everything here takes the backbone layer as an
opaque nn.Module and the policy's forward/loss/action as opaque CALLABLES, so the hard
activation-gradient wiring is unit-testable weight-free on a toy nn.Module BEFORE any GPU
minute (test_ctrliface_core.py::test_accumulate_gram_*). The CTRLIFACE GPU jobs bind these
callables to lerobot's PI05Policy at run time (the bindings are pinned at Gate 0).

Objects (frozen spec §2):
  * accumulate_gram — ∂ℓ/∂h_L capture over demo/text batches → the activation-space Gram G_L
    whose top-k eigvecs are S(k,L). The capture hook forces h_L to require grad + retain_grad,
    so gradients reach it EVEN when the backbone params are frozen (requires_grad=False) — the
    graph is anchored at the activation, not the weights. loss_fn MUST run under grad mode
    (no torch.no_grad / inference_mode around the backbone — Gate 0 verifies this).
  * harvest_hbar_trace — the mean-ablation target h̄_L and tr(C_L)=Σ‖h−h̄‖² (energy-match input),
    one pass, no grad.
  * proxy_deviation — the cheap S_action-sweep proxy D = E‖a_ablate − a_intact‖ (no rollout).

Design rules (CLAUDE.md): NO try/except hiding; a wrong assumption RAISES. Heavy imports inside
functions so the module py_compiles on the control plane.
"""
from __future__ import annotations

from infra.pg.ctrliface_core import gram_update, make_layer_ablation_hook


def _layer_hidden(out):
    """The residual-stream hidden tensor of a decoder-layer output (tuple or bare tensor)."""
    return out[0] if isinstance(out, tuple) else out


def capture_grad_hook(store, key="h"):
    """Forward hook that anchors grad at the layer output: force requires_grad + retain_grad and
    stash the tensor in `store[key]`, so store[key].grad = ∂loss/∂h_L after loss.backward().
    Returns the output UNCHANGED (capture only). Works with a frozen backbone because the anchor
    is the activation, not any parameter."""
    def hook(_m, _i, out):
        hs = _layer_hidden(out)
        if not hs.requires_grad:
            hs.requires_grad_(True)
        hs.retain_grad()
        store[key] = hs
        return out
    return hook


def accumulate_gram(model, layer, loss_fn, batches, d, n_prefix=None):
    """Accumulate the activation-space Gram G_L = Σ ∇_{h_L}ℓ ∇_{h_L}ℓᵀ (d×d fp64) over `batches`.
    `model` = the module to zero_grad between batches; `layer` = the hooked decoder layer;
    `loss_fn(batch) -> scalar loss` (differentiable to h_L, run under grad mode). `n_prefix`:
    accumulate only over the first n_prefix positions (the prefix the expert reads), else all.
    Returns (G, n_positions). Fail-loud if grad never reaches h_L (a no_grad wrap / wrong layer)."""
    import torch
    G = torch.zeros(d, d, dtype=torch.float64)
    n_pos = 0
    for batch in batches:
        store = {}
        h = layer.register_forward_hook(capture_grad_hook(store))
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_fn(batch)
            assert loss.requires_grad, "loss does not require grad — backbone likely under no_grad"
            loss.backward()
        h.remove()
        hs = store["h"]
        assert hs.grad is not None, "∂loss/∂h_L is None — grad did not reach the hooked layer"
        g = hs.grad
        assert g.shape[-1] == d, f"h_L dim {g.shape[-1]} != d {d}"
        g = g[:, :n_prefix, :] if n_prefix is not None else g
        g = g.reshape(-1, d).double()
        assert torch.isfinite(g).all(), "non-finite activation gradient"
        gram_update(G, g)
        n_pos += g.shape[0]
    assert n_pos > 0, "no positions accumulated"
    return G, n_pos


def accumulate_gram_multi(model, layers, loss_fn, batches, d, n_prefix=None):
    """Like accumulate_gram but for MANY layers in ONE backward per batch: register a capture hook
    on every layer in `layers` (dict {name: module}), run forward+backward ONCE, and read each
    layer's ∂loss/∂h from its retained grad. 18× cheaper than per-layer backward. Returns
    ({name: G (d×d fp64)}, n_positions). Fail-loud if any layer's grad is None (wrong layer / no_grad)."""
    import torch
    G = {name: torch.zeros(d, d, dtype=torch.float64) for name in layers}
    n_pos = 0
    for batch in batches:
        stores = {name: {} for name in layers}
        handles = [layers[name].register_forward_hook(capture_grad_hook(stores[name]))
                   for name in layers]
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_fn(batch)
            assert loss.requires_grad, "loss does not require grad — backbone likely under no_grad"
            loss.backward()
        for h in handles:
            h.remove()
        batch_pos = None
        for name in layers:
            hs = stores[name]["h"]
            assert hs.grad is not None, f"∂loss/∂h is None for layer {name}"
            g = hs.grad
            g = g[:, :n_prefix, :] if n_prefix is not None else g
            g = g.reshape(-1, d).double()
            assert torch.isfinite(g).all(), f"non-finite grad at layer {name}"
            gram_update(G[name], g)
            batch_pos = g.shape[0] if batch_pos is None else batch_pos
        n_pos += batch_pos
    assert n_pos > 0, "no positions accumulated"
    return G, n_pos


def harvest_hbar_trace(layer, forward_fn, batches, d, n_prefix=None):
    """One no-grad pass: mean prefix residual h̄_L (d,) and tr(C_L)=Σ‖h−h̄‖² over the same
    positions (the energy-match denominator + the RF/ablation target). forward_fn(batch) runs a
    plain forward (the hook reads the layer output). Returns (hbar fp64, trace_C float, n)."""
    import torch
    acc = {"sum": torch.zeros(d, dtype=torch.float64), "sq": 0.0, "n": 0}

    def hook(_m, _i, out):
        hs = _layer_hidden(out).detach().double()
        hs = hs[:, :n_prefix, :] if n_prefix is not None else hs
        flat = hs.reshape(-1, d)
        acc["sum"] += flat.sum(0).cpu()
        acc["sq"] += float((flat ** 2).sum().item())
        acc["n"] += flat.shape[0]
        return out

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        for batch in batches:
            forward_fn(batch)
    h.remove()
    assert acc["n"] > 0, "harvest saw no positions"
    hbar = acc["sum"] / acc["n"]
    trace_C = acc["sq"] - acc["n"] * float((hbar ** 2).sum().item())   # Σ‖h‖² − n‖h̄‖²
    return hbar, trace_C, acc["n"]


def harvest_hbar_trace_multi(layers, forward_fn, batches, d, n_prefix=None):
    """Multi-layer h̄/trace in ONE no-grad forward per batch (hook every layer). Returns
    {name: (hbar fp64, trace_C float)} + n_positions."""
    import torch
    acc = {name: {"sum": torch.zeros(d, dtype=torch.float64), "sq": 0.0, "n": 0} for name in layers}

    def _mk(name):
        def hook(_m, _i, out):
            hs = _layer_hidden(out).detach().double()
            hs = hs[:, :n_prefix, :] if n_prefix is not None else hs
            flat = hs.reshape(-1, d)
            acc[name]["sum"] += flat.sum(0).cpu()
            acc[name]["sq"] += float((flat ** 2).sum().item())
            acc[name]["n"] += flat.shape[0]
            return out
        return hook

    handles = [layers[name].register_forward_hook(_mk(name)) for name in layers]
    with torch.no_grad():
        for batch in batches:
            forward_fn(batch)
    for h in handles:
        h.remove()
    res, n = {}, 0
    for name in layers:
        a = acc[name]
        assert a["n"] > 0, f"harvest saw no positions for {name}"
        hbar = a["sum"] / a["n"]
        trace_C = a["sq"] - a["n"] * float((hbar ** 2).sum().item())
        res[name] = (hbar, trace_C)
        n = a["n"]
    return res, n


def proxy_deviation(layer, action_fn, batches, Q, hbar, n_prefix=None, seed=0):
    """The cheap S-sweep proxy (no rollout): D = mean_batch ‖a_ablate − a_intact‖ where a_* is
    action_fn(batch) with / without the mean-ablation hook (span(Q)→h̄) on `layer`. Larger D ⇒
    the ablated subspace matters more to the controller's output.

    NOISE CONTROL (load-bearing for a flow-matching policy): π0.5's action sampler draws random
    noise, so two bare calls differ even with NO ablation. We `torch.manual_seed(seed+i)` before
    BOTH the intact and the ablated call of batch i, so the sampled noise is identical and the
    deviation reflects ONLY the ablation. Returns (D float, per_batch)."""
    import torch
    per = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            torch.manual_seed(seed + i)
            a_intact = action_fn(batch).detach().double()
            h = layer.register_forward_hook(make_layer_ablation_hook(Q, hbar, n_prefix=n_prefix))
            torch.manual_seed(seed + i)                      # SAME noise draw as the intact call
            a_ablate = action_fn(batch).detach().double()
            h.remove()
            per.append(float((a_ablate - a_intact).norm(dim=-1).mean().item()))
    D = float(sum(per) / len(per))
    return D, per
