"""CTRLIFACE π0.5 backbone-residual hook — the CORRECT injection point for PaliGemmaWithExpert.

Gate-0 discovery (modeling_pi05.py): the joint backbone+expert forward does NOT call the decoder
layer modules' `forward()` — it runs a module-level function `compute_layer_complete(inputs_embeds,
…, layers=…)` per layer that calls the SUBMODULES (q/k/v/o_proj, mlp, layernorms) directly. So a
forward hook on `language_model.layers[L]` never fires (the KeyError we hit). The backbone residual
stream h_L is the returned `outputs_embeds[0]`.

Fix: monkeypatch the module-global `compute_layer_complete` to CAPTURE (grad-Gram) or ABLATE
(rollout) `outputs_embeds[0]` per layer index (matched via `id(layers[0])`). In `.eval()` mode
gradient-checkpointing is OFF (gates on `self.training`, modeling_pi05:501-505) → the direct call
path fires the patch + retain_grad is meaningful. The ablation math is the SAME tested projector
(ctrliface_core.apply_mean_ablation). The transform factories below are module-level + weight-free
unit-tested; only `_patched` touches lerobot. No try/except hiding.
"""
from __future__ import annotations

import contextlib

from infra.pg.ctrliface_core import apply_mean_ablation, gram_update


# ---------------------------------------------------------------------------
# Transform factories (pure; weight-free unit-tested). Each maps (L, out) -> out',
# where `out` is compute_layer_complete's return list [backbone_residual, expert_residual].
# ---------------------------------------------------------------------------
def make_capture_transform(store):
    """Stash the backbone residual out[0] per layer L into `store` (retain_grad if it requires grad)."""
    def tf(L, out):
        h = out[0]
        if h.requires_grad:
            h.retain_grad()
        store[L] = h
        return out
    return tf


def make_mean_accum_transform(acc, d, n_prefix=None):
    """No-grad accumulate per-layer sum/sumsq/count of out[0] into `acc[L]` (for h̄ + tr(C))."""
    def tf(L, out):
        h = out[0].detach().double()
        h = h[:, :n_prefix, :] if n_prefix is not None else h
        flat = h.reshape(-1, d)
        a = acc[L]
        a["sum"] = a["sum"] + flat.sum(0).cpu()
        a["sq"] += float((flat ** 2).sum().item())
        a["n"] += flat.shape[0]
        return out
    return tf


def make_ablate_transform(L_target, Q, hbar, n_prefix=None):
    """Mean-ablate span(Q)→h̄ on out[0] at layer L_target (same projector as apply_mean_ablation)."""
    def tf(L, out):
        if L != L_target:
            return out
        h = out[0]
        Qd, hd = Q.to(h.device), hbar.to(h.device)
        if n_prefix is None:
            h2 = apply_mean_ablation(h, Qd, hd)
        else:
            h2 = h.clone()
            h2[:, :n_prefix, :] = apply_mean_ablation(h[:, :n_prefix, :], Qd, hd)
        return [h2] + list(out[1:])
    return tf


def make_mean_replace_transform(hbars, n_prefix=None):
    """REPLACE out[0] with h̄ at EVERY layer (dead-backbone floor)."""
    def tf(L, out):
        h = out[0]
        hd = hbars[L].to(h.device).to(h.dtype)
        h2 = h.clone()
        if n_prefix is None:
            h2[:] = hd
        else:
            h2[:, :n_prefix, :] = hd
        return [h2] + list(out[1:])
    return tf


def make_pre_hook_ablate(Q, hbar, n_prefix=None):
    """forward_PRE_hook for a standard decoder layer (Path-1 inference `language_model.forward`):
    mean-ablate span(Q)→h̄ on the layer INPUT hidden_states (args[0]) — the residual ENTERING layer L,
    the SAME point S_action[L] characterizes. Returns modified positional args."""
    def hook(_module, args):
        hs = args[0]
        Qd, hd = Q.to(hs.device), hbar.to(hs.device)
        if n_prefix is None:
            hs2 = apply_mean_ablation(hs, Qd, hd)
        else:
            hs2 = hs.clone()
            hs2[:, :n_prefix, :] = apply_mean_ablation(hs[:, :n_prefix, :], Qd, hd)
        return (hs2,) + tuple(args[1:])
    return hook


def make_pre_hook_replace(hbar, n_prefix=None):
    """forward_PRE_hook: REPLACE the layer input residual with h̄ (positive/floor, Path-1)."""
    def hook(_module, args):
        hs = args[0]
        hd = hbar.to(hs.device).to(hs.dtype)
        hs2 = hs.clone()
        if n_prefix is None:
            hs2[:] = hd
        else:
            hs2[:, :n_prefix, :] = hd
        return (hs2,) + tuple(args[1:])
    return hook


def make_mean_replace_at_transform(L_target, hbar, n_prefix=None):
    """REPLACE out[0] with h̄ at ONE layer (the full-read positive control at L_target)."""
    def tf(L, out):
        if L != L_target:
            return out
        h = out[0]
        hd = hbar.to(h.device).to(h.dtype)
        h2 = h.clone()
        if n_prefix is None:
            h2[:] = hd
        else:
            h2[:, :n_prefix, :] = hd
        return [h2] + list(out[1:])
    return tf


# ---------------------------------------------------------------------------
# The monkeypatch (touches lerobot; not weight-free).
# ---------------------------------------------------------------------------
def _backbone_layers(policy):
    return policy.model.paligemma_with_expert.paligemma.model.language_model.layers


@contextlib.contextmanager
def _patched(policy, transform):
    """Monkeypatch compute_layer_complete so `transform(L, inputs_embeds)` runs on the residual
    ENTERING backbone layer L (the list [prefix_residual, expert_residual]) BEFORE the layer computes.
    We transform the INPUT (not the output): the residual entering layer L is consumed by layer L
    (its K/V feed the joint attention → the action loss), so ∂loss/∂h_L is well-defined for ALL 18
    layers — capturing the OUTPUT of the last layer gives None (the final backbone residual is not
    consumed by the action loss). Ablating the input is also the correct residual-stream intervention
    (it changes what layer L reads)."""
    import lerobot.policies.pi05.modeling_pi05 as M
    idx = {id(l): i for i, l in enumerate(_backbone_layers(policy))}
    orig = M.compute_layer_complete

    def patched(inputs_embeds, attention_mask, position_ids, adarms_cond, layers=None, rotary_emb=None):
        L = idx.get(id(layers[0]))
        if L is not None:
            inputs_embeds = transform(L, inputs_embeds)
        return orig(inputs_embeds, attention_mask, position_ids, adarms_cond,
                    layers=layers, rotary_emb=rotary_emb)

    M.compute_layer_complete = patched
    try:
        yield
    finally:
        M.compute_layer_complete = orig


# ---------------------------------------------------------------------------
# Public: capture the Gram / harvest h̄ / ablate during rollout.
# ---------------------------------------------------------------------------
def capture_gram(policy, loss_fn, batches, d, n_prefix=None, n_layers=18):
    """G_L = Σ ∇_{h_L}ℓ ∇_{h_L}ℓᵀ over `batches`, all layers per backward. policy .eval() + params
    requiring grad. Returns ({L: G d×d fp64}, n_positions)."""
    import torch
    dev = next(policy.parameters()).device                # accumulate the Gram on the policy's GPU
    G = {L: torch.zeros(d, d, dtype=torch.float64, device=dev) for L in range(n_layers)}
    n_pos = 0
    for batch in batches:
        store = {}
        policy.zero_grad(set_to_none=True)
        with _patched(policy, make_capture_transform(store)):
            with torch.enable_grad():
                loss = loss_fn(batch)
                assert loss.requires_grad, "loss does not require grad (need unfrozen params + eval fwd)"
                loss.backward()
        assert len(store) == n_layers, f"captured {len(store)}/{n_layers} backbone layers"
        batch_pos = None
        for L in range(n_layers):
            g = store[L].grad
            assert g is not None, f"∂loss/∂h_L is None at layer {L}"
            g = g[:, :n_prefix, :] if n_prefix is not None else g
            g = g.reshape(-1, d).double()
            assert torch.isfinite(g).all(), f"non-finite grad at layer {L}"
            gram_update(G[L], g)
            batch_pos = g.shape[0] if batch_pos is None else batch_pos
        n_pos += batch_pos
    assert n_pos > 0, "no positions accumulated"
    return G, n_pos


def harvest_hbar_trace(policy, forward_fn, batches, d, n_prefix=None, n_layers=18):
    """Per-layer h̄_L + tr(C_L)=Σ‖h−h̄‖² over `batches` (one no-grad pass). Returns ({L:(hbar,trace)}, n)."""
    import torch
    acc = {L: {"sum": torch.zeros(d, dtype=torch.float64), "sq": 0.0, "n": 0} for L in range(n_layers)}
    with _patched(policy, make_mean_accum_transform(acc, d, n_prefix)):
        with torch.no_grad():
            for batch in batches:
                forward_fn(batch)
    res, n = {}, 0
    for L in range(n_layers):
        a = acc[L]
        assert a["n"] > 0, f"no positions for layer {L}"
        hbar = a["sum"] / a["n"]
        res[L] = (hbar, a["sq"] - a["n"] * float((hbar ** 2).sum().item()))
        n = a["n"]
    return res, n


@contextlib.contextmanager
def ablate_residual(policy, L_target, Q, hbar, n_prefix=None):
    """Mean-ablate the residual entering layer L_target for the duration — installs BOTH paths so the
    ablation fires whether the forward is the joint training path (Path-3 compute_layer_complete →
    monkeypatch) or the standard inference path (Path-1 language_model.forward → forward_pre_hook on
    layers[L]). Idempotent (re-projecting an already-projected residual is a no-op)."""
    layers = _backbone_layers(policy)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(policy, make_ablate_transform(L_target, Q, hbar, n_prefix)))
        h = layers[L_target].register_forward_pre_hook(make_pre_hook_ablate(Q, hbar, n_prefix))
        stack.callback(h.remove)
        yield


@contextlib.contextmanager
def mean_replace_all(policy, hbars, n_prefix=None):
    """Dead-backbone floor — replace h_L with h̄ at EVERY layer (both paths)."""
    layers = _backbone_layers(policy)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(policy, make_mean_replace_transform(hbars, n_prefix)))
        for i in range(len(layers)):
            h = layers[i].register_forward_pre_hook(make_pre_hook_replace(hbars[i], n_prefix))
            stack.callback(h.remove)
        yield


@contextlib.contextmanager
def mean_replace_at(policy, L_target, hbar, n_prefix=None):
    """Full-read positive control — replace h_L with h̄ at L_target only (both paths)."""
    layers = _backbone_layers(policy)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(policy, make_mean_replace_at_transform(L_target, hbar, n_prefix)))
        h = layers[L_target].register_forward_pre_hook(make_pre_hook_replace(hbar, n_prefix))
        stack.callback(h.remove)
        yield


def proxy_deviation(policy, action_fn, batches, L, Q, hbar, seed=0, n_prefix=None):
    """Cheap S-sweep proxy (no rollout): D = mean ‖a_ablate − a_intact‖ over `batches`, with the
    flow-sampler noise fixed (same torch seed) across the intact and ablated calls so the deviation
    reflects only the ablation of span(Q) at layer L. action_fn(batch) -> predicted action tensor."""
    import torch
    per = []
    with torch.no_grad():
        for i, b in enumerate(batches):
            torch.manual_seed(seed + i)
            a_int = action_fn(b).detach().double()
            with ablate_residual(policy, L, Q, hbar, n_prefix):
                torch.manual_seed(seed + i)
                a_abl = action_fn(b).detach().double()
            per.append(float((a_abl - a_int).norm(dim=-1).mean().item()))
    return float(sum(per) / len(per)), per


def capture_activations(policy, forward_fn, batches, d, layers_wanted, n_prefix=None):
    """Collect the backbone residual h_L (concatenated over batches×positions) for each L in
    `layers_wanted` — for the energy-matched-null E_target = removed_energy(X_L, S_action[:, :k], h̄_L).
    Returns {L: X_L (n, d) fp64 cpu}. No grad."""
    import torch
    buf = {L: [] for L in layers_wanted}
    want = set(layers_wanted)

    def tf(L, out):
        if L in want:
            h = out[0].detach().double()
            h = h[:, :n_prefix, :] if n_prefix is not None else h
            buf[L].append(h.reshape(-1, d).cpu())
        return out

    with _patched(policy, tf):
        with torch.no_grad():
            for b in batches:
                forward_fn(b)
    return {L: torch.cat(buf[L], 0) for L in layers_wanted}
