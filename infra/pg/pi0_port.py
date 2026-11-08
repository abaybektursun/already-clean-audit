"""π0 port remap — the single source of the backbone-LM key remap (pi-trajectory spec §E0b-load/§E1.5).

lerobot/pi0_base is a LeRobot policy safetensors. Its Gemma-1 backbone LM decoder lives under
  paligemma_with_expert.paligemma.model.language_model.<gemma-1 key>
alongside the SigLIP vision tower (…paligemma.model.vision_tower.*) and the 300M action expert
(…gemma_expert.*). The port DROPS the tied embed_tokens (openpi #669 / lerobot #2151), so the spine of
analysis is the 18×7 decoder MATRICES + norms — embeddings are untestable by the weight gate (stamped).

Pure, fp64-agnostic, no try/except. Discovered structure (e0b_discover): backbone bucket = 603 keys,
expert = 164, layer-0 k/v = [256,2048] (== PG1, MQA single-KV head_dim 256).
"""
from __future__ import annotations

BACKBONE_LM_PREFIX = "paligemma_with_expert.paligemma.model.language_model."
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def remap_backbone_lm(pi0_sd):
    """{gemma1_key: tensor} for the π0 backbone Gemma-1 LM decoder — strip the prefix, EXCLUDE the vision
    tower and the action expert. Fail-loud if any gemma_expert key would enter (compartment leak)."""
    out = {}
    for k, v in pi0_sd.items():
        if not k.startswith(BACKBONE_LM_PREFIX):
            continue
        if "vision_tower" in k:                          # shares the paligemma prefix; not the LM
            continue
        assert "gemma_expert" not in k, f"action-expert key leaked into backbone set: {k}"
        out[k[len(BACKBONE_LM_PREFIX):]] = v
    return out


def decoder_matrices(sd):
    """The 18×7 decoder WEIGHT MATRICES (2-D, q/k/v/o/gate/up/down) keyed as in a bare GemmaModel."""
    return {k: v for k, v in sd.items()
            if any(t in k for t in PROJ) and getattr(v, "ndim", None) == 2}


def assert_18x7(matrices, n_layers=18):
    """Explicit layer-count + per-(type,layer) presence — a key-intersection would not catch a gap."""
    expect = {f"layers.{i}.{('self_attn' if p.startswith(('q', 'k', 'v', 'o')) else 'mlp')}.{p}.weight"
              for i in range(n_layers) for p in PROJ}
    missing = sorted(expect - set(matrices))
    assert not missing, f"18×7 incomplete: {len(missing)} missing (e.g. {missing[:5]})"
    assert len(matrices) == n_layers * 7, f"expected {n_layers*7} matrices, got {len(matrices)}"
    return sorted(matrices)


def _rel_frob(a, b):
    """fp64 ‖a−b‖_F/‖a‖_F."""
    import torch
    x = a.detach().to(torch.float64)
    y = b.detach().to(torch.float64)
    nx = x.norm().item()
    return (y - x).norm().item() / nx if nx > 0 else float("nan")


def kv_convention_probe(pg_k, pg_v, pi0_k, pi0_v):
    """Distinguish a real FT#2 change from a CONVENTION artifact (within-head permutation / K↔V slab-swap
    — both shape-identical, so a shape check is vacuous). The CORRECT convention keeps the identity mapping
    (pi0_k vs pg_k) CLOSEST — its residual is the FT (~0.3) — while a swap/permute reads near the √2 floor.

    Returns the three relative-Frobenius distances + whether identity is the clear winner (FT-scale AND
    < half the nearest alternative)."""
    import torch
    d_identity = _rel_frob(pg_k, pi0_k)
    d_kv_swap = _rel_frob(pg_k, pi0_v)                    # K vs the port's V slab (openpi K=[i,0] V=[i,1,0])
    d_rowperm = _rel_frob(pg_k, torch.flip(pi0_k, dims=[0]))   # a within-head 256-row permutation candidate
    nearest_alt = min(d_kv_swap, d_rowperm)
    identity_wins = (d_identity < 0.7) and (d_identity < 0.5 * nearest_alt)
    return {"d_identity": d_identity, "d_kv_swap": d_kv_swap, "d_rowperm": d_rowperm,
            "identity_is_closest": bool(identity_wins)}
