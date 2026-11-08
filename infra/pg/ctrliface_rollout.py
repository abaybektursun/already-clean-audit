"""CTRLIFACE ablation-rollout harness — the ablation SPEC → basis/hook construction (spec §3/§5
arms) plus the hook installation on the backbone layer. The env + rollout LOOP itself is bound to
lerobot's in-process rollout API in the job (pinned at Gate-0c); this module holds the part that is
independent of that API and is unit-tested weight-free.

Ablation kinds (each gates a claim, spec §5):
  identity       — no hook (RF≈1; harness inert).
  saction        — S_action top-k at layer L (the treatment).
  random_k       — a fresh random d×k orthonormal basis (equal-k artifact null).
  random_energy  — a random subspace whose EXPECTED removed energy matches S_action(L,k)'s
                   (energy_matched_random; the DECISIVE null). Needs E_target (removed energy of
                   S_action(L,k) on activations) + tr(C_L); both measured at confirmation time.
  shuffle        — S_action basis from a DIFFERENT layer applied at L (structure rolled across
                   layers → must collapse to the random floor).
  positive       — full-read mean-replace (h_L←h̄; control must floor → the harness has teeth).

Design (CLAUDE.md): no try/except hiding; a bad kind/rank RAISES.
"""
from __future__ import annotations

from infra.pg.ctrliface_core import (
    make_layer_ablation_hook, make_mean_replace_hook, random_basis, energy_matched_random,
)

KINDS = ("identity", "saction", "random_k", "random_energy", "shuffle", "positive")


def build_Q_hbar(kind, L, k, bases, hbars, d, seed, E_target=None, trace_C=None, n_layers=18):
    """Return (Q, hbar) for the ablation hook, or (None, None) for identity/positive (handled by a
    dedicated hook). `bases` = {L: Q_full (d,kmax)} S_action per layer; `hbars` = {L: h̄ (d,)}.
    For random_energy, E_target (removed energy of S_action(L,k)) + trace_C=tr(C_L) set the rank."""
    import torch
    assert kind in KINDS, f"unknown ablation kind {kind}"
    if kind in ("identity", "positive"):
        return None, hbars.get(L)
    hbar = hbars[L]
    if kind == "saction":
        Q = bases[L][:, :k]
    elif kind == "random_k":
        Q = random_basis(d, k, seed, "cpu", torch.float64)
    elif kind == "random_energy":
        assert E_target is not None and trace_C is not None, "random_energy needs E_target + trace_C"
        Q, _r = energy_matched_random(d, E_target, trace_C, seed, "cpu", torch.float64)
    elif kind == "shuffle":
        L_roll = (L + n_layers // 2) % n_layers            # a different layer's S_action, applied at L
        assert L_roll != L, "shuffle rolled to the same layer"
        Q = bases[L_roll][:, :k]
    return Q, hbar


def install_ablation(layers, kind, L, k, bases, hbars, d, seed, n_prefix=None,
                     E_target=None, trace_C=None):
    """Register the ablation hook(s) on layers[L] for the given kind. Returns a list of hook handles
    (EMPTY for identity → no perturbation). Caller removes the handles after the rollout."""
    if kind == "identity":
        return []
    if kind == "positive":
        return [layers[L].register_forward_hook(make_mean_replace_hook(hbars[L], n_prefix=n_prefix))]
    Q, hbar = build_Q_hbar(kind, L, k, bases, hbars, d, seed, E_target=E_target,
                           trace_C=trace_C, n_layers=len(layers))
    return [layers[L].register_forward_hook(make_layer_ablation_hook(Q, hbar, n_prefix=n_prefix))]
