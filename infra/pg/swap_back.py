"""XGEN cross-generation seal — the NEW swap-back shell + V2 carrier helpers.

The load-bearing NEW instruments for the `gemma-2b → PaliGemma-1` hop (prereg §13).
XGEN's swap-back is the INVERSE of `contrast_a1._lm_swap_state_dict` (W11-A / PG2):

  W11-A : keep PG's *extended-vocab* embed_tokens/lm_head, swap the decoder INTO the
          PaliGemma harness  (PG vocab 257152 ≠ base 256000 → can't use a bare shell).
  XGEN  : keep *base* Gemma-1 embed_tokens/lm_head, load the target decoder into a pure
          `GemmaForCausalLM` text shell. The decoder matrices + norms carry no vocab
          dependency, so they strict-load into the base shell; PG1's embed_tokens
          (extended vocab) is DROPPED by design and the base embed/head kept.

Split by design (so §9.1 can validate the logic weight-free, no transformers/GPU):
  * PURE state-dict core — `swapback_state_dict`, `random_decoder_state_dict`,
    `row_shuffled_after` + the V1 arithmetic (`s_text`, `boot_s_text`, `subset_by_hash`).
    Tiny-tensor / planted-array testable on a laptop in <1s.
  * THIN model wrappers — `load_xgen_decoders`, `assemble_swapback_model`. Real
    Gemma-1 / PaliGemma-1; run on the box, wiring validated at §9.2.

Heavy imports (torch/numpy/transformers) live inside functions so this module
py_compiles on the control plane. Fail-loud; no try/except hiding.
"""
from __future__ import annotations

EMBED_KEYS = ("embed_tokens", "lm_head")


def _is_embed(k: str) -> bool:
    return any(t in k for t in EMBED_KEYS)


# ===========================================================================
# PURE state-dict core (no model, no GPU) — the §9.1 dry-run gate exercises this.
# ===========================================================================
def swapback_state_dict(base_lm_sd, target_dec_sd):
    """The swap-back shell, as a pure state_dict transform.

    Returns (new_sd, info) where new_sd has EXACTLY base_lm_sd's keys/shapes:
    embed_tokens/lm_head are KEPT from base, every decoder key (attention + MLP
    matrices + every norm) comes from target_dec_sd. Iterating over the BASE keys
    guarantees a strict-clean (0 missing / 0 unexpected) load back into the base
    model. Fail-loud on a missing target counterpart (a partial swap that would
    silently leak base weights) or a shape mismatch (architecture regression).
    """
    new_sd, kept, swapped, missing = {}, [], [], []
    for k, base_w in base_lm_sd.items():
        if _is_embed(k):
            new_sd[k] = base_w
            kept.append(k)
            continue
        if k not in target_dec_sd:
            missing.append(k)
            continue
        tw = target_dec_sd[k]
        if tuple(tw.shape) != tuple(base_w.shape):
            raise RuntimeError(
                f"swap-back shape mismatch on {k!r}: base={list(base_w.shape)} "
                f"target={list(tw.shape)}. Architectures must align.")
        new_sd[k] = tw
        swapped.append(k)
    if missing:
        raise RuntimeError(
            f"swap-back: {len(missing)} base decoder keys absent from the target "
            f"(a partial swap would leak base weights). e.g. {sorted(missing)[:8]}")
    info = {
        "scope": ("decoder layers + every norm loaded from target; "
                  "embed_tokens + lm_head KEPT from base Gemma-1"),
        "n_decoder_keys_swapped": len(swapped),
        "n_embed_keys_kept": len(kept),
        "embed_keys_kept": sorted(kept),
    }
    return new_sd, info


def random_decoder_state_dict(base_lm_sd, seed):
    """The V1 FLOOR: every 2-D decoder matrix → scale-matched Gaussian noise (per-tensor
    std of the base matrix it replaces); embed_tokens/lm_head + 1-D norms/biases KEPT from
    base (structural init). Deterministic in `seed`. Port of the build_random_after loop.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    new_sd = {}
    for k, v in base_lm_sd.items():
        if _is_embed(k):
            new_sd[k] = v
            continue
        vf = v.detach().cpu().float()
        if vf.ndim >= 2:                                   # weight matrices → scale-matched noise
            std = vf.std().clamp_min(1e-8)
            new_sd[k] = (torch.randn(vf.shape, generator=g) * std).to(v.dtype)
        else:                                              # 1-D (norms/biases) → keep structural
            new_sd[k] = v
    return new_sd


def row_shuffled_after(w_before, w_after, generator):
    """The V2 shuffle control. Permute the ROWS of ΔW = w_after − w_before (destroys the
    column-space alignment with w_before; preserves the Frobenius norm + the row marginal)
    and return w_before + ΔW_perm, so `gain_routing.decompose(w_before, ·)` collapses
    f_perp toward the dimension-counting null. A real, structured ΔW₁ sits off that null.
    """
    import torch
    dW = w_after - w_before
    perm = torch.randperm(dW.shape[0], generator=generator)
    return w_before + dW[perm]


# ===========================================================================
# V1 capability arithmetic + the sha1 discovery/confirmation split (prereg §4.1).
# Mirrors w14_arm2._subset + w11_legA_unblind._boot_S (kept pure + importable here).
# ===========================================================================
def subset_by_hash(arr, item_hashes, which):
    """Split per-item values by sha1 hash parity on the verified alignment key
    (`per_item_hash = sha1(question[:64])[:12]`): discovery = even (int(h,16)%2==0),
    confirmation = odd, all = full. The analyst calibrates on discovery; the headline
    is read once on confirmation."""
    assert len(arr) == len(item_hashes), \
        f"arr/hash length mismatch: {len(arr)}/{len(item_hashes)}"
    if which == "all":
        return list(arr)
    assert which in ("discovery", "confirmation"), f"unknown half {which!r}"
    par = 1 if which == "confirmation" else 0
    return [v for v, h in zip(arr, item_hashes) if int(h, 16) % 2 == par]


def s_text(after_pi, before_pi, floor_pi):
    """Point capability S_text = (mean(AFTER) − mean(FLOOR)) / (mean(BEFORE) − mean(FLOOR))."""
    import numpy as np
    a, b, f = (np.asarray(x, dtype=float) for x in (after_pi, before_pi, floor_pi))
    denom = b.mean() - f.mean()
    return float((a.mean() - f.mean()) / denom) if denom > 0 else float("nan")


def boot_s_text(after_pi, before_pi, floor_pi, n_boot=2000, seed=1337):
    """Paired nonparametric item-bootstrap of S_text (AFTER/BEFORE/FLOOR are per-item-aligned).
    Returns (boot_mean, lo95, hi95, n_valid). Draws with a den>0 only (mirrors _boot_S)."""
    import numpy as np
    a, b, f = (np.asarray(x, dtype=float) for x in (after_pi, before_pi, floor_pi))
    n = a.size
    assert b.size == n == f.size, f"per-item arrays misaligned: {a.size}/{b.size}/{f.size}"
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        den = b[idx].mean() - f[idx].mean()
        if den > 0:
            out.append((a[idx].mean() - f[idx].mean()) / den)
    out = np.asarray(out)
    assert out.size > 0, "all bootstrap draws had denom<=0 (BEFORE<=FLOOR on this split) — S_text undefined"
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(out.size)


def theta_forget(before_pi, floor_pi, n_boot=2000, seed=1337, pct=97.5):
    """θ_forget = FLOOR + the eval-noise band, expressed in S_text units (prereg §6).

    S_text normalizes FLOOR→0, BEFORE→1, so "FLOOR + noise band" is the one-sided upper edge of
    the S_text sampling noise at the FLOOR — i.e. the bootstrap `pct` percentile of S_text computed
    with AFTER := FLOOR (a null-AFTER). H_forget iff the confirmation-half S_text upper-CI < θ_forget
    (AFTER is statistically indistinguishable from a random-decoder FLOOR). Calibrated on the
    discovery half; never sees confirmation. Returns θ_forget (float, ≳0)."""
    import numpy as np
    b, f = (np.asarray(x, dtype=float) for x in (before_pi, floor_pi))
    n = b.size
    assert f.size == n, f"per-item arrays misaligned: before={b.size} floor={f.size}"
    rng = np.random.default_rng(seed)
    # AFTER:=FLOOR is identically 0, so the noise band is read from two INDEPENDENT FLOOR-equivalent
    # resamples: (FLOOR' − FLOOR)/(BEFORE − FLOOR) — the S_text-scale sampling spread at the floor.
    out = []
    for _ in range(n_boot):
        i1 = rng.integers(0, n, n)            # FLOOR' (a second random-decoder-equivalent draw)
        i2 = rng.integers(0, n, n)            # reference FLOOR
        den = b[i2].mean() - f[i2].mean()
        if den > 0:
            out.append((f[i1].mean() - f[i2].mean()) / den)
    out = np.asarray(out)
    assert out.size > 0, "θ_forget: all denom<=0 (BEFORE<=FLOOR on the discovery half)"
    return float(np.percentile(np.abs(out), pct))


# ===========================================================================
# THIN model wrappers (transformers; run on the box — validated at §9.2, NOT §9.1).
# ===========================================================================
def load_xgen_decoders(device="cpu", base_revision="main", dtype=None):
    """Load base Gemma-1 (google/gemma-2b) + the PaliGemma-1 LM tower and return their
    normalized decoder state_dicts for the V2 carrier and the swap-back shell.

    Returns (base_model, base_dec_sd, pg1_dec_sd, meta). fp32 CPU eager by default
    (matches e0a). base_dec_sd / pg1_dec_sd are bare GemmaModel decoders (embed dropped by
    `matched_decoder_weight_matrices`; kept here for the shell). REUSES e0a's extractors.
    """
    import torch
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.e0a_initfamily_gate import _lm_decoder_sd, _norm_keys

    base_repo = ENTITIES["L0_g1"].repo_id            # google/gemma-2b  (Gemma-1, NOT gemma-2-2b)
    pg1_repo = ENTITIES["PG1"].repo_id               # google/paligemma-3b-pt-224
    td = dtype or torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_repo, revision=base_revision, torch_dtype=td,
        device_map=device, attn_implementation="eager")
    pg1 = PaliGemmaForConditionalGeneration.from_pretrained(
        pg1_repo, torch_dtype=td, device_map=device, attn_implementation="eager")
    base_dec = _norm_keys(base.model.state_dict())
    pg1_dec = _lm_decoder_sd(pg1)
    meta = {"base_repo": base_repo, "base_revision": base_revision, "pg1_repo": pg1_repo,
            "base_model_type": getattr(base.config, "model_type", None)}
    assert meta["base_model_type"] == "gemma", f"base is not Gemma-1: {meta['base_model_type']}"
    del pg1
    return base, base_dec, pg1_dec, meta


def assemble_swapback_model(base_model, target_dec_sd, device):
    """Load target_dec_sd into a FRESH base Gemma-1 GemmaForCausalLM, keeping base
    embed_tokens/lm_head, and return a model `text_eval.mmlu_accuracy` can run.

    `base_model` is mutated in place (load it fresh per arm; its attn_implementation is
    fixed at from_pretrained time). Strict-clean load + the embed-kept post-load gate
    (lm_head is tied to embed_tokens and lives outside base_model.model, so keeping base
    embed keeps base lm_head). The decoder-differs check is NOT done here (BEFORE is an
    identity swap; the AFTER decoder difference is certified separately by E0a). Returns
    (base_model, info).
    """
    import torch
    from infra.aws_pg.jobs.e0a_initfamily_gate import _norm_keys
    base_lm_sd = _norm_keys(base_model.model.state_dict())     # bare GemmaModel decoder keys
    pre_embed = base_lm_sd[[k for k in base_lm_sd if "embed_tokens" in k][0]].detach().cpu().clone()
    new_sd, info = swapback_state_dict(base_lm_sd, target_dec_sd)
    missing, unexpected = base_model.model.load_state_dict(
        {k: v.to(device) for k, v in new_sd.items()}, strict=True)
    assert not missing and not unexpected, \
        f"swap-back load not strict-clean: missing={missing} unexpected={unexpected}"
    # post-load gate: embed_tokens unchanged (base interface kept).
    now_embed = _norm_keys(base_model.model.state_dict())[
        [k for k in base_lm_sd if "embed_tokens" in k][0]].detach().cpu()
    assert torch.equal(now_embed, pre_embed), "swap-back changed embed_tokens — base interface must be KEPT."
    return base_model, info
