# W5 — Activation patching to localize the LM rewrite (spec §7 W5; Zhang & Nanda 2024).
"""Locate WHICH LM decoder layers/positions carry the installed image→answer capability.

CONFIRMATORY localization. On clean/corrupt pairs (RefCOCO primary — S_B=0.00, the
capability is entirely LM-installed — + VQAv2), we patch the CLEAN LM residual into
the CORRUPT forward across all 26 LM decoder layers, separately at image-token vs
text-token positions, and measure recovery of the clean answer. The layer where
patching recovers the answer = where the image→answer signal is carried.

Best practice (Zhang & Nanda, ICLR 2024 / Heimersheim & Nanda): VARY the corruption
method, the metric, and the target token; report the sensitivity grid, not a single
number. Floor = resample/mean (copying the clean activation IS resample-patching);
NEVER zero (§5.1).

Discipline (CLAUDE.md / spec hard rules):
  * NEVER zero. Patch = overwrite corrupt residual with the CLEAN residual at L
    (resample-patch), or with the batch-mean (mean-patch) — never zeros.
  * NEVER try/except to hide errors. Position/shape mismatch RAISES.
  * PURE module (torch/numpy only, no Modal) — unit-importable.

The patch (pre-reg §4): run CLEAN forward, cache residual h_clean[L] at every layer.
Run CORRUPT forward with a hook at layer L that REPLACES the corrupt residual with
h_clean[L] at the chosen positions (image / text / all). Read the clean-answer
logit/logprob (and the task metric) → recovery = (patched - corrupt)/(clean - corrupt).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ===========================================================================
# Cache the clean residual stream at every LM decoder layer (one forward).
# ===========================================================================

class ResidualCache:
    """Capture the output residual of EVERY layer in a ModuleList (one forward).

    Stores hidden_states[L] (b, T, d) per layer. Context-managed so hooks are always
    removed. Used to grab the clean activations that get patched into the corrupt run."""

    def __init__(self, layers: torch.nn.ModuleList):
        self.layers = layers
        self.cache: dict[int, torch.Tensor] = {}
        self._handles = []

    def _mk(self, i):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.cache[i] = hs.detach()
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._mk(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ===========================================================================
# Patch hook: overwrite ONE layer's residual with cached clean values (never zero).
# ===========================================================================

class PatchHook:
    """Forward hook on ONE layer that overwrites the (corrupt) residual at chosen
    positions with cached CLEAN residual values (resample-patch) or the batch-mean
    (mean-patch). NEVER zero (§5.1).

    positions = 'all' | 'image' | 'text'; pos_mask (b,T) bool set by the harness.
    source = the cached clean residual (b, T, d) for this layer (resample-patch), or
             None to mean-patch with the corrupt residual's own batch-mean."""

    def __init__(self, layer_module, clean_residual: torch.Tensor | None,
                 pos_mask: torch.Tensor, mode: str = "resample"):
        assert mode in ("resample", "mean"), mode
        self.layer = layer_module
        self.clean = clean_residual
        self.pos_mask = pos_mask.bool()
        self.mode = mode
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output      # (b,T,d) corrupt
        if self.mode == "resample":
            assert self.clean is not None, "resample-patch needs the cached clean residual"
            src = self.clean.to(hs.dtype).to(hs.device)              # clean (b,T,d)
            if src.shape != hs.shape:
                raise RuntimeError(
                    f"Patch shape mismatch: clean {tuple(src.shape)} vs corrupt {tuple(hs.shape)} "
                    f"— clean/corrupt must be aligned (same tokenization/length).")
        else:  # mean-patch: replace with the corrupt residual's own batch+token mean
            src = hs.float().mean(dim=(0, 1), keepdim=True).expand_as(hs).to(hs.dtype)
        pm = self.pos_mask.to(hs.device)[:, :, None]                 # (b,T,1)
        new = torch.where(pm, src, hs)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


# ===========================================================================
# Answer-token logprob (the continuous patching metric) + recovery.
# ===========================================================================

def first_answer_token_ids(processor, answer_text: str) -> list[int]:
    """Token ids of the answer's FIRST token(s) under the PG tokenizer. For RefCOCO the
    answer_text is the GT `<loc>` box string; for VQA the majority answer string. We
    return ALL ids of the encoded answer so the harness can score the first or all."""
    enc = processor.tokenizer(answer_text, add_special_tokens=False)
    ids = enc["input_ids"]
    if not ids:
        raise RuntimeError(f"Answer text {answer_text!r} encoded to zero tokens.")
    return ids


@torch.inference_mode()
def teacher_forced_answer_logprob(model, proc, image, prompt: str, answer_text: str,
                                  device, target: str = "first",
                                  patch_ctx=None, set_pos_mask=None) -> float:
    """Mean logprob of the answer tokens under teacher forcing (prompt+answer fed in;
    score the answer-token positions). This is the continuous patching metric — robust
    where the discrete acc@0.5 is flat. `target` ∈ {'first','all'} selects the first vs
    all answer tokens (the target-token sensitivity axis).

    patch_ctx: optional context manager (PatchHook) active during the forward.
    set_pos_mask: optional callable(enc) that sets patch_ctx.pos_mask from the built
                  inputs (needs the realised image-token positions of THIS pair)."""
    full = proc(text=prompt, images=image.convert("RGB"), suffix=answer_text,
                return_tensors="pt", padding="longest")
    full = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full.items()}
    full["pixel_values"] = full["pixel_values"].to(torch.bfloat16)
    labels = full.get("labels")
    if labels is None:
        raise RuntimeError("processor did not return labels for suffix= ; cannot score answer logprob.")

    if set_pos_mask is not None and patch_ctx is not None:
        set_pos_mask(full)

    def _run():
        out = model(input_ids=full["input_ids"], attention_mask=full["attention_mask"],
                    pixel_values=full["pixel_values"])
        return out.logits.float()

    logits = (_run() if patch_ctx is None else _with(patch_ctx, _run))
    # answer-token positions = where labels != -100; logits[t] predicts token t+1.
    lab = labels[0]
    ans_pos = (lab != -100).nonzero(as_tuple=True)[0]
    if ans_pos.numel() == 0:
        raise RuntimeError("No answer-label positions (labels all -100).")
    if target == "first":
        ans_pos = ans_pos[:1]
    lp = torch.log_softmax(logits[0], dim=-1)
    tok_lp = [lp[p - 1, lab[p]].item() for p in ans_pos.tolist()]
    return float(np.mean(tok_lp))


def _with(ctx, fn):
    with ctx:
        return fn()


def recovery(patched: float, clean: float, corrupt: float) -> float:
    """Normalized recovery: 0 = no better than corrupt, 1 = fully clean.
    recovery = (patched - corrupt) / (clean - corrupt). Reported per layer/position."""
    denom = clean - corrupt
    if abs(denom) < 1e-9:
        return float("nan")
    return (patched - corrupt) / denom


# ===========================================================================
# Clean/corrupt pairing (corruption methods varied — Zhang & Nanda best practice).
# ===========================================================================

@dataclass
class CleanCorruptPair:
    clean_prompt: str
    clean_image: object              # PIL
    clean_answer: str                # the supervised answer string (loc box / VQA answer)
    corrupt_prompt: str
    corrupt_image: object
    corruption: str                  # 'img_resample' | 'expr_swap' | 'q_swap'


def make_pairs(rows: list, task: str, prompt_fn, target_fn, corruption: str,
               seed: int = 0) -> list:
    """Build clean/corrupt pairs from eval rows under one corruption method.

    img_resample: corrupt = clean prompt + a DIFFERENT image (next row's image).
    expr_swap (RefCOCO): corrupt = clean image + a different row's referring expr.
    q_swap (VQA): corrupt = clean image + a different row's question.

    The clean ANSWER (loc box of clean row / clean VQA answer) is the recovery target;
    corruption removes the image→answer signal so patching the clean residual restores it.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    perm = rng.permutation(n)
    # ensure the donor != self
    donor = np.where(perm == np.arange(n), (perm + 1) % n, perm)
    pairs = []
    for i, row in enumerate(rows):
        clean_prompt = prompt_fn(row)
        clean_image = row["image"]
        clean_answer = target_fn(row)
        d = rows[int(donor[i])]
        if corruption == "img_resample":
            corrupt_prompt = clean_prompt
            corrupt_image = d["image"]
        elif corruption in ("expr_swap", "q_swap"):
            corrupt_prompt = prompt_fn(d)
            corrupt_image = clean_image
        else:
            raise RuntimeError(f"unknown corruption {corruption!r}")
        pairs.append(CleanCorruptPair(
            clean_prompt=clean_prompt, clean_image=clean_image, clean_answer=clean_answer,
            corrupt_prompt=corrupt_prompt, corrupt_image=corrupt_image, corruption=corruption,
        ))
    return pairs
