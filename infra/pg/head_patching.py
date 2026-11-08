# Q2 — head-level activation patching to localize the LM rewrite at HEAD granularity.
"""Locate WHICH attention HEADS carry the installed image→answer capability, and
whether that head set is shared or task-specific across RefCOCO vs VQAv2.

This EXTENDS the W5 layer-level residual patching (infra.pg.lm_patching) down one
level of granularity: instead of patching a whole decoder layer's residual output,
we patch a SINGLE attention head's output, per (layer, head), and measure recovery
of the clean answer in the corrupt run.

THE INTERVENTION SITE (Gemma2 / PaliGemma2 LM). A Gemma2 decoder layer's attention
computes per-head outputs o_h (head_dim each), CONCATENATES them into a (b, T,
num_heads*head_dim) tensor, and projects with o_proj → (b, T, d_model). The
concatenated pre-o_proj tensor is therefore the per-head output stream: head h
occupies the contiguous block [h*head_dim : (h+1)*head_dim]. We hook o_proj with a
forward_PRE_hook (its INPUT is exactly that concatenated tensor) → we can read one
head's block (cache) or overwrite it (patch) cleanly, with NO change to attention
math. This is the standard per-head patching site (o_proj input slice).

Why a pre-hook on o_proj and not the attn module output: the attn module's output is
already o_proj-mixed (all heads summed into d_model) — heads are no longer separable
there. The pre-o_proj concatenation is the last point where heads are addressable.

Best practice (Zhang & Nanda ICLR 2024 / Heimersheim & Nanda): VARY the corruption
method, the metric/target token; report the sensitivity, not one number. Floor =
resample-patch (copy the CLEAN head output) — NEVER zero (§5.1). Internal-validity
control: patching from an UNCORRUPTED clean pair must give ~0 recovery (no spurious
recovery from the patch mechanism itself).

Discipline (CLAUDE.md / queue guardrails):
  * NEVER zero. Patch = overwrite the corrupt head block with the CLEAN head block
    (resample-patch). NEVER zeros.
  * NEVER try/except to hide errors. Shape/position mismatch RAISES with a trace.
  * PURE module (torch/numpy only, no Modal) — unit-importable, shared by analysis.
"""
from __future__ import annotations

import numpy as np
import torch


# ===========================================================================
# Resolve the per-head geometry of a Gemma2 decoder layer's attention.
# ===========================================================================

def resolve_head_geometry(layer_module) -> dict:
    """Read (num_heads, head_dim, o_proj) off ONE decoder layer's self_attn.

    Gemma2 stores num_attention_heads + head_dim on the attention config; o_proj maps
    (num_heads*head_dim) → hidden_size. We DON'T assume the attribute names — we read
    the o_proj weight shape (the ground truth: in_features = num_heads*head_dim) and
    cross-check against the config head count. Fails loud on any inconsistency."""
    attn = layer_module.self_attn
    o_proj = attn.o_proj
    in_features = o_proj.weight.shape[1]            # = num_heads * head_dim (ground truth)
    # head count: prefer the attention module's own config; cross-check via head_dim.
    cfg = getattr(attn, "config", None)
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and cfg is not None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and cfg is not None:
        head_dim = getattr(cfg, "head_dim", None)
    if num_heads is None and head_dim is not None:
        num_heads = in_features // int(head_dim)
    if head_dim is None and num_heads is not None:
        head_dim = in_features // int(num_heads)
    if num_heads is None or head_dim is None:
        raise RuntimeError(
            f"Cannot resolve (num_heads, head_dim) from self_attn: "
            f"num_heads={num_heads} head_dim={head_dim} o_proj.in_features={in_features}. "
            f"attn attrs: {[a for a in dir(attn) if not a.startswith('__')][:30]}")
    num_heads, head_dim = int(num_heads), int(head_dim)
    if num_heads * head_dim != in_features:
        raise RuntimeError(
            f"Head geometry mismatch: num_heads({num_heads})*head_dim({head_dim}) "
            f"= {num_heads*head_dim} != o_proj.in_features({in_features}). "
            f"o_proj input is NOT a plain head concatenation for this architecture — "
            f"per-head slicing is unsafe; halt.")
    return {"num_heads": num_heads, "head_dim": head_dim, "o_proj_in": in_features}


# ===========================================================================
# Cache the clean per-head attention output (the o_proj INPUT) at every layer.
# ===========================================================================

class AttnHeadCache:
    """Capture the o_proj INPUT (b, T, num_heads*head_dim) at EVERY layer in one
    forward — the concatenated per-head attention output (before o_proj mixes heads).

    Context-managed so the pre-hooks are always removed. This is the clean source the
    head-patch overwrites into the corrupt run (resample-patch). We hook each layer's
    self_attn.o_proj with a forward_pre_hook; args[0] is the concatenated head tensor."""

    def __init__(self, layers: torch.nn.ModuleList):
        self.layers = layers
        self.cache: dict[int, torch.Tensor] = {}
        self._handles = []

    def _mk(self, i):
        def pre_hook(module, args):
            x = args[0]                                  # (b, T, num_heads*head_dim)
            self.cache[i] = x.detach()
        return pre_hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._mk(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ===========================================================================
# Patch hook: overwrite ONE head's output block with cached clean values (never zero).
# ===========================================================================

class HeadPatchHook:
    """Forward_PRE_hook on ONE layer's o_proj that overwrites a SINGLE head's output
    block at chosen positions with the cached CLEAN head output (resample-patch). The
    rest of the concatenated head tensor (all other heads) is left as the corrupt run
    produced it → only head `head` at layer `layer` is patched. NEVER zero (§5.1).

    head_dim block: head h occupies columns [h*head_dim : (h+1)*head_dim] of the
    o_proj input (b, T, num_heads*head_dim).

    positions: a (b, T) bool mask of patchable positions (set by the harness; the
    PROMPT region only — never the answer suffix, which would trivially leak)."""

    def __init__(self, layer_module, head: int, head_dim: int,
                 clean_head_input: torch.Tensor, pos_mask: torch.Tensor):
        self.o_proj = layer_module.self_attn.o_proj
        self.head = int(head)
        self.head_dim = int(head_dim)
        self.clean = clean_head_input                    # (b, T, num_heads*head_dim) CLEAN
        self.pos_mask = pos_mask.bool()
        self._handle = None

    def __enter__(self):
        self._handle = self.o_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args):
        x = args[0]                                      # (b, T, H*hd) corrupt concat
        src = self.clean.to(x.dtype).to(x.device)
        if src.shape != x.shape:
            raise RuntimeError(
                f"Head-patch shape mismatch: clean {tuple(src.shape)} vs corrupt "
                f"{tuple(x.shape)} — clean/corrupt must be aligned (same tokenization).")
        lo = self.head * self.head_dim
        hi = lo + self.head_dim
        pm = self.pos_mask.to(x.device)[:, :, None]      # (b, T, 1)
        new = x.clone()
        # overwrite ONLY head `head`'s block, ONLY at patchable positions.
        head_slice = torch.where(pm, src[:, :, lo:hi], x[:, :, lo:hi])
        new[:, :, lo:hi] = head_slice
        return (new,) + tuple(args[1:])


def recovery(patched: float, clean: float, corrupt: float) -> float:
    """Normalized recovery: 0 = no better than corrupt, 1 = fully clean.
    recovery = (patched - corrupt) / (clean - corrupt). Reused verbatim from W5."""
    denom = clean - corrupt
    if abs(denom) < 1e-9:
        return float("nan")
    return (patched - corrupt) / denom
