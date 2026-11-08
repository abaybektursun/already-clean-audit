# A1 CONTRAST — the tower-swap engine for PRIMARY-A / PRIMARY-B / FLOOR (spec §2.1).
"""The one-shot unblinding (gate G3). Builds the three blinded A1 configs by
swapping a tower's INIT into PaliGemma2-3b-pt-224, then applies the FROZEN
transfer recipe (imported verbatim from transfer_a1.RECIPE) identically to every
config. Only the tower init differs between configs — the recipe never changes.

Configs (spec §2.1 table):
  PRIMARY_A: base SigLIP vision (V0) + PG LM   — "is the vision change necessary?"
  PRIMARY_B: PG vision           + base Gemma2 LM (L0) — "is the LM change necessary?"
  FLOOR:     base SigLIP vision  + base Gemma2 LM       — borrowed substrate + recipe

CEILING (PG + PG) is NOT rebuilt here — its 5-seed mean is reused from the pilot
(/vol/results/a1_pilot/sigma_seed.json, mean 0.7403). This module builds ONLY the
three swapped configs.

V0 resolution (spec §3 SigLIP-init match):
  PG2-3b-pt-224 vision_config = siglip_vision_model, image_size 224, patch14,
  hidden 1152, 27 layers, 16 heads, intermediate 4304 (= SigLIP-So400m, 256 image
  tokens, 256 position embeddings). The public checkpoint with the IDENTICAL config
  AND native 224 resolution (so 256 position embeddings — NO resize needed) is
  google/siglip-so400m-patch14-224. PaliGemma's big_vision Stage-1 initialises the
  vision tower from the raw public SigLIP-So400m checkpoint at 224 (PG2 paper §2).
  We load its SiglipVisionModel weights directly into PG2's vision_tower.
  * If the standalone 224 checkpoint's vision_model.* keys + shapes match PG2's
    vision_tower exactly (verified by `probe_swap_structure`), the swap is a direct
    state_dict load with NO position-embedding interpolation → FULL confidence.
  * If shapes differ (e.g. PG used the 384 checkpoint resized to 224), we apply the
    same deterministic bicubic pos-embed interpolation PaliGemma uses and disclose
    vision-side as LOWER-CONFIDENCE (spec §3). The realised path is recorded in the
    swap report (`vision_pos_embed_action`), not assumed.

LM swap (PRIMARY_B) — vocab subtlety (spec, handled explicitly):
  PG2 extended the token embedding table with multimodal tokens (image/loc/seg) that
  base Gemma 2 never had. The PIPELINE needs those rows to exist, so we KEEP PG's
  embedding table + lm_head untouched and swap ONLY the DECODER (the 26 Gemma2
  transformer layers + the final norm). This is the documented, conservative choice:
  we reset the learned *computation* (the layers) to base Gemma2, while preserving the
  extended-vocab embedding interface the multimodal pipeline requires. The report
  states exactly what was reset (`lm_reset_scope = "decoder layers + final norm;
  embed_tokens + lm_head KEPT (PG, extended vocab)"`).

Design rules (CLAUDE.md + spec §6):
  * NO try/except hiding — a wrong assumption raises with a full trace.
  * NEVER touch the frozen transfer_a1.RECIPE / data / eval — imported verbatim.
  * Verification gates (build-time, fail loud) BEFORE any training:
      (a) the swapped submodule's params now DIFFER from PG's originals;
      (b) every NON-swapped submodule is BITWISE-IDENTICAL to PG (a leak = corrupt);
      (c) shapes / param-counts sane;
      (d) a forward pass on one VQA item is finite.
    Any failure → RuntimeError (HALT), never train on a bad swap.
"""
from __future__ import annotations

import time

from infra.pg.transfer_a1 import (
    A1_CONFIGS,
    RECIPE,
    VQA_DATASET,
    VQA_EVAL_SPLIT,
    VQA_PREFIX,
    VQA_TRAIN_SPLIT,
    _seed_everything,
    _vqa_majority_answer,
    _collate_train,
    vqa_soft_accuracy,
)

# The public SigLIP-So400m init for PG2's 224 vision tower (native 256 positions).
V0_SIGLIP_REPO = "google/siglip-so400m-patch14-224"
# Base Gemma 2 2B (L0) — PG2's LM init (entities.ENTITIES["L0"]).
L0_GEMMA_REPO = "google/gemma-2-2b"

# Which configs this module builds (CEILING is reused from the pilot, not rebuilt).
SWAP_CONFIGS = ("PRIMARY_A", "PRIMARY_B", "FLOOR")


# ===========================================================================
# Structure probe — bind the swap to REAL keys/shapes (introspect, don't assume).
# ===========================================================================

def probe_swap_structure() -> dict:
    """Load PG2 + the two init checkpoints; report exact state_dict keys/shapes
    for the vision_tower and language_model so the swap binds to real structure.

    Returns a JSON-able dict; called by the Modal `probe` entrypoint before any
    swap is trusted. No swap, no training here.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        PaliGemmaForConditionalGeneration,
        SiglipVisionModel,
    )

    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        ENTITIES["PG"].repo_id, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa",
    )
    paths = resolve_submodule_paths(pg)
    vt = pg.get_submodule(paths["vision_tower"])
    lm = pg.get_submodule(paths["language_model"])

    pg_vt_sd = vt.state_dict()
    pg_lm_sd = lm.state_dict()

    # Standalone public SigLIP-So400m @224 vision tower.
    sig = SiglipVisionModel.from_pretrained(
        V0_SIGLIP_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    sig_sd = sig.state_dict()

    # Standalone base Gemma 2 2B. PG's `language_model` submodule is the Gemma2
    # *decoder* (Gemma2Model) — keys `embed_tokens.weight`, `layers.N.*`, `norm.weight`,
    # NO `model.` prefix and NO `lm_head`. The standalone Gemma2ForCausalLM nests the
    # decoder at `.model` and adds a top-level `lm_head`, so `g2.state_dict()` keys are
    # `model.layers.N.*` + `lm_head.weight` (namespace mismatch). The matching decoder
    # is `g2.model` (the Gemma2Model) → its state_dict aligns with PG's language_model.
    g2 = AutoModelForCausalLM.from_pretrained(
        L0_GEMMA_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    )
    g2_full_sd = g2.state_dict()              # Gemma2ForCausalLM (model.* + lm_head)
    g2_sd = g2.model.state_dict()             # Gemma2Model decoder — aligns with PG LM

    def _pe_shape(sd, needle):
        for k, v in sd.items():
            if needle in k:
                return {"key": k, "shape": list(v.shape)}
        return None

    pg_vt_keys = set(pg_vt_sd)
    sig_keys = set(sig_sd)
    # Shape agreement on the shared keys (the swap requires identical shapes).
    shared_vt = pg_vt_keys & sig_keys
    vt_shape_mismatch = {
        k: {"pg": list(pg_vt_sd[k].shape), "siglip": list(sig_sd[k].shape)}
        for k in shared_vt if pg_vt_sd[k].shape != sig_sd[k].shape
    }

    pg_lm_keys = set(pg_lm_sd)
    g2_keys = set(g2_sd)            # decoder keys (g2.model) — should align with PG LM
    g2_full_keys = set(g2_full_sd)  # ForCausalLM keys (model.* + lm_head)
    shared_lm = pg_lm_keys & g2_keys
    lm_shape_mismatch = {
        k: {"pg": list(pg_lm_sd[k].shape), "gemma2": list(g2_sd[k].shape)}
        for k in shared_lm if pg_lm_sd[k].shape != g2_sd[k].shape
    }
    # The swap KEEPS PG's embed_tokens (extended vocab) → the only expected shape
    # mismatch on shared keys is embed_tokens (257216 vs 256000), which we do NOT load.
    lm_decoder_keys_to_swap = {k for k in shared_lm if "embed_tokens" not in k}
    lm_decoder_shape_ok = all(pg_lm_sd[k].shape == g2_sd[k].shape
                              for k in lm_decoder_keys_to_swap)

    return {
        "submodule_paths": paths,
        "vision_tower": {
            "pg_n_keys": len(pg_vt_keys),
            "siglip_n_keys": len(sig_keys),
            "only_in_pg": sorted(pg_vt_keys - sig_keys)[:20],
            "only_in_siglip": sorted(sig_keys - pg_vt_keys)[:20],
            "n_shared": len(shared_vt),
            "n_shape_mismatch": len(vt_shape_mismatch),
            "shape_mismatch_sample": dict(list(vt_shape_mismatch.items())[:8]),
            "pg_pos_embed": _pe_shape(pg_vt_sd, "position_embedding"),
            "siglip_pos_embed": _pe_shape(sig_sd, "position_embedding"),
        },
        "language_model": {
            "_note": "g2_* below are the Gemma2 DECODER (g2.model), which aligns with "
                     "PG's language_model submodule. g2_FULL_* are Gemma2ForCausalLM "
                     "(model.* + lm_head) — namespace-mismatched, NOT used for the swap.",
            "pg_n_keys": len(pg_lm_keys),
            "gemma2_decoder_n_keys": len(g2_keys),
            "gemma2_FULL_n_keys": len(g2_full_keys),
            "gemma2_FULL_only_sample": sorted(g2_full_keys - pg_lm_keys)[:10],
            "only_in_pg_vs_decoder": sorted(pg_lm_keys - g2_keys)[:20],
            "only_in_decoder_vs_pg": sorted(g2_keys - pg_lm_keys)[:20],
            "n_shared_with_decoder": len(shared_lm),
            "n_shape_mismatch_shared": len(lm_shape_mismatch),
            "shape_mismatch_sample": dict(list(lm_shape_mismatch.items())[:8]),
            "n_decoder_keys_to_swap": len(lm_decoder_keys_to_swap),
            "decoder_shape_ok_excl_embed": bool(lm_decoder_shape_ok),
            "pg_embed_tokens": _pe_shape(pg_lm_sd, "embed_tokens"),
            "gemma2_embed_tokens": _pe_shape(g2_sd, "embed_tokens"),
            "decoder_layer_keys_sample": [k for k in sorted(pg_lm_keys)
                                          if k.startswith("layers.0.")][:20],
            "pg_has_final_norm": any(k == "norm.weight" for k in pg_lm_keys),
            "decoder_has_final_norm": any(k == "norm.weight" for k in g2_keys),
        },
    }


# ===========================================================================
# Pos-embed reconciliation (only used if the standalone init resolution differs).
# ===========================================================================

def _interpolate_pos_embed(src_weight, n_target: int):
    """Bicubic-interpolate a [n_src, dim] position-embedding table to [n_target, dim].

    Used ONLY if the public SigLIP init has a different number of positions than
    PG2's 224 vision tower (e.g. a 384-resolution checkpoint, 729 → 256). PaliGemma
    resizes pos-embeds deterministically; we mirror that with bicubic interpolation
    on the square patch grid. NOT used when the native-224 checkpoint matches
    (256 → 256), which is the expected, full-confidence path.
    """
    import math

    import torch
    import torch.nn.functional as F

    n_src, dim = src_weight.shape
    g_src = int(round(math.sqrt(n_src)))
    g_tgt = int(round(math.sqrt(n_target)))
    assert g_src * g_src == n_src, f"src pos-embed {n_src} not a square grid"
    assert g_tgt * g_tgt == n_target, f"target pos-embed {n_target} not a square grid"
    grid = src_weight.float().reshape(1, g_src, g_src, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(g_tgt, g_tgt), mode="bicubic", align_corners=False)
    out = grid.permute(0, 2, 3, 1).reshape(n_target, dim)
    return out.to(src_weight.dtype)


# ===========================================================================
# The swaps. Each returns (state_dict_to_load, info) for the target submodule.
# ===========================================================================

def _vision_swap_state_dict(pg_vt_sd: dict, sig_sd: dict) -> tuple[dict, dict]:
    """Build the state_dict to load into PG2's vision_tower from the public SigLIP.

    The standalone SiglipVisionModel and PG2's vision_tower share architecture, so
    keys align. If pos-embed counts differ, interpolate (disclosed, lower-confidence).
    Returns (new_sd, info). new_sd has EXACTLY PG2's vision_tower keys/shapes.
    """
    new_sd = {}
    pos_action = "none (native 224, 256 positions — direct load)"
    resized = []
    missing_in_siglip = []
    for k, pg_w in pg_vt_sd.items():
        if k not in sig_sd:
            missing_in_siglip.append(k)
            continue
        s_w = sig_sd[k]
        if s_w.shape == pg_w.shape:
            new_sd[k] = s_w.clone()
        elif "position_embedding" in k and s_w.ndim == 2 and pg_w.ndim == 2 \
                and s_w.shape[1] == pg_w.shape[1]:
            # Reconcile position embeddings to PG2's 224 grid (lower-confidence path).
            new_sd[k] = _interpolate_pos_embed(s_w, pg_w.shape[0])
            resized.append({"key": k, "from": list(s_w.shape), "to": list(pg_w.shape)})
            pos_action = (f"bicubic-interpolated {s_w.shape[0]}->{pg_w.shape[0]} positions "
                          f"(public init resolution differs; LOWER-CONFIDENCE per spec §3)")
        else:
            raise RuntimeError(
                f"Vision swap: shape mismatch on {k!r} that is not a pos-embed resize: "
                f"siglip={list(s_w.shape)} pg={list(pg_w.shape)}. Resolve before training."
            )
    # The new state_dict must cover EVERY PG vision_tower key (else load_state_dict
    # would leave PG-original weights in place → a partial, corrupt swap).
    if missing_in_siglip:
        raise RuntimeError(
            f"Vision swap: {len(missing_in_siglip)} PG vision_tower keys absent from the "
            f"public SigLIP checkpoint — partial swap would leak PG weights. "
            f"Examples: {missing_in_siglip[:8]}"
        )
    info = {
        "vision_pos_embed_action": pos_action,
        "vision_pos_embed_resized": resized,
        "vision_confidence": "full" if not resized else "approximate-per-§3",
        "n_keys_loaded": len(new_sd),
    }
    return new_sd, info


def _lm_swap_state_dict(pg_lm_sd: dict, g2_sd: dict) -> tuple[dict, dict]:
    """Build the state_dict to load into PG2's language_model from base Gemma 2 2B.

    Resets ONLY the decoder layers + final norm to base Gemma2; KEEPS PG's
    embed_tokens + lm_head (the extended multimodal vocab the pipeline needs).
    Returns (new_sd, info). new_sd has EXACTLY PG2's language_model keys/shapes —
    embedding/lm_head rows come from PG (unchanged), everything else from Gemma2.
    """
    new_sd = {}
    kept_embedding_keys = []
    swapped_keys = []
    g2_missing = []
    for k, pg_w in pg_lm_sd.items():
        is_embed = ("embed_tokens" in k) or ("lm_head" in k)
        if is_embed:
            # KEEP PG's extended-vocab embedding/lm_head verbatim.
            new_sd[k] = pg_w.clone()
            kept_embedding_keys.append(k)
            continue
        if k not in g2_sd:
            g2_missing.append(k)
            continue
        s_w = g2_sd[k]
        if s_w.shape != pg_w.shape:
            raise RuntimeError(
                f"LM swap: shape mismatch on decoder key {k!r}: "
                f"gemma2={list(s_w.shape)} pg={list(pg_w.shape)}. Architectures must align."
            )
        new_sd[k] = s_w.clone()
        swapped_keys.append(k)
    if g2_missing:
        raise RuntimeError(
            f"LM swap: {len(g2_missing)} PG decoder keys absent from base Gemma 2 — "
            f"partial swap would leak PG weights. Examples: {g2_missing[:8]}"
        )
    info = {
        "lm_reset_scope": ("decoder layers + final norm reset to base Gemma2; "
                           "embed_tokens + lm_head KEPT (PG, extended multimodal vocab)"),
        "n_decoder_keys_swapped": len(swapped_keys),
        "n_embedding_keys_kept": len(kept_embedding_keys),
        "embedding_keys_kept": kept_embedding_keys,
    }
    return new_sd, info


# ===========================================================================
# Build a swapped model + attach the FROZEN recipe (connector + LM-LoRA).
# ===========================================================================

def build_swapped_model(config_name: str, seed: int):
    """Load PG2-3b-pt-224, swap the tower(s) per `config_name`, run the build-time
    verification gates, then attach the frozen recipe (connector trainable + LoRA
    scoped to the LM subtree). Returns (model, processor, report).

    The recipe attachment is IDENTICAL to build_ceiling_model (same regex-anchored
    LoRA, same connector-trainable, same vision-frozen invariant) — only the base
    weights differ. We re-implement the attach here (not import build_ceiling_model)
    because that function asserts CEILING; the recipe constants come from the SAME
    frozen RECIPE dict, so the recipe is identical by construction.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
        SiglipVisionModel,
    )

    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    assert config_name in SWAP_CONFIGS, (
        f"build_swapped_model handles only {SWAP_CONFIGS}; CEILING is reused from "
        f"the pilot, not rebuilt. Got {config_name!r}."
    )
    cfg = A1_CONFIGS[config_name]
    swap_vision = cfg.vision_init == "V0"
    swap_lm = cfg.lm_init == "L0"
    assert swap_vision or swap_lm, f"{config_name} swaps nothing — not a contrast config"

    repo = ENTITIES["PG"].repo_id
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    )
    proc = PaliGemmaProcessor.from_pretrained(repo)
    paths = resolve_submodule_paths(model)

    # --- snapshot PG-ORIGINAL state_dicts for the bitwise-leak gate (b) ---------
    # Clone to CPU so they survive the in-place swaps (used as the ground truth).
    pg_orig = {
        name: {k: v.detach().clone().cpu()
               for k, v in model.get_submodule(path).state_dict().items()}
        for name, path in paths.items()
    }

    swap_info: dict = {}

    # --- VISION swap (PRIMARY_A, FLOOR) -----------------------------------------
    if swap_vision:
        sig = SiglipVisionModel.from_pretrained(
            V0_SIGLIP_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        vt = model.get_submodule(paths["vision_tower"])
        new_vt_sd, vinfo = _vision_swap_state_dict(vt.state_dict(), sig.state_dict())
        # strict=True: every key present, shapes match → no silent partial load.
        new_vt_sd = {k: v.to("cuda") for k, v in new_vt_sd.items()}
        missing, unexpected = vt.load_state_dict(new_vt_sd, strict=True)
        assert not missing and not unexpected, (
            f"Vision load_state_dict not strict-clean: missing={missing} unexpected={unexpected}"
        )
        swap_info["vision"] = {"source": V0_SIGLIP_REPO, **vinfo}
        del sig

    # --- LM swap (PRIMARY_B, FLOOR) ---------------------------------------------
    if swap_lm:
        g2 = AutoModelForCausalLM.from_pretrained(
            L0_GEMMA_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
            attn_implementation="eager",
        )
        lm = model.get_submodule(paths["language_model"])
        # PG's language_model is the Gemma2 DECODER (Gemma2Model); the matching
        # standalone decoder is g2.model (NOT g2 = Gemma2ForCausalLM, whose keys carry
        # a `model.` prefix + lm_head). Pass the decoder state_dict so keys align.
        new_lm_sd, linfo = _lm_swap_state_dict(lm.state_dict(), g2.model.state_dict())
        new_lm_sd = {k: v.to("cuda") for k, v in new_lm_sd.items()}
        missing, unexpected = lm.load_state_dict(new_lm_sd, strict=True)
        assert not missing and not unexpected, (
            f"LM load_state_dict not strict-clean: missing={missing} unexpected={unexpected}"
        )
        swap_info["lm"] = {"source": L0_GEMMA_REPO, **linfo}
        del g2

    # =======================================================================
    # VERIFICATION GATES (build-time, fail loud) — spec hard rules (a)-(d).
    # =======================================================================
    gate = _verify_swap(model, paths, pg_orig, swap_vision, swap_lm, proc)

    # --- attach the FROZEN recipe (identical to build_ceiling_model) ------------
    for p in model.parameters():
        p.requires_grad_(False)

    lm_leaf = paths["language_model"].rsplit(".", 1)[-1]
    leaves = "|".join(RECIPE["lora_target_modules"])
    target_regex = rf".*{lm_leaf}\..*\.({leaves})$"
    lora = LoraConfig(
        r=RECIPE["lora_rank"],
        lora_alpha=RECIPE["lora_alpha"],
        lora_dropout=RECIPE["lora_dropout"],
        bias="none",
        target_modules=target_regex,
    )
    model = get_peft_model(model, lora)

    proj_leaf = paths["multi_modal_projector"].rsplit(".", 1)[-1]
    n_proj_params = 0
    for name, p in model.named_parameters():
        if f".{proj_leaf}." in name or name.startswith(f"{proj_leaf}."):
            p.requires_grad_(True)
            n_proj_params += p.numel()
    assert n_proj_params > 0, f"No connector params matched leaf '{proj_leaf}'."

    trainable, total, n_lora = 0, 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if "lora_" in name:
                n_lora += p.numel()
    assert n_lora > 0, "LoRA adapters did not attach (no trainable lora_ params)."

    vision_trainable = [n for n, p in model.named_parameters()
                        if p.requires_grad and "vision_tower" in n]
    assert not vision_trainable, (
        f"Vision tower must stay frozen; {len(vision_trainable)} trainable vision params "
        f"(LoRA leak). Examples: {vision_trainable[:4]}"
    )

    report = {
        "config": config_name,
        "vision_init": cfg.vision_init,
        "lm_init": cfg.lm_init,
        "swap_info": swap_info,
        "swap_gate": gate,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / total, 4),
        "lora_params": n_lora,
        "connector_params": n_proj_params,
        "submodule_paths": paths,
    }
    return model, proc, report


def _verify_swap(model, paths, pg_orig, swap_vision, swap_lm, proc) -> dict:
    """The four build-time gates (spec hard rules). Raises on any failure (HALT).

    (a) swapped submodule params DIFFER from PG originals;
    (b) every NON-swapped submodule is BITWISE-IDENTICAL to PG;
    (c) shapes / param-counts sane (covered by strict load + key-set equality);
    (d) a forward pass on one VQA item is finite.
    """
    import torch
    from PIL import Image
    import numpy as np

    def _sd_now(name):
        return {k: v.detach().cpu() for k, v in model.get_submodule(paths[name]).state_dict().items()}

    def _bitwise_identical(a: dict, b: dict) -> tuple[bool, list]:
        if set(a) != set(b):
            return False, sorted(set(a) ^ set(b))[:8]
        diffs = []
        for k in a:
            if a[k].shape != b[k].shape:
                diffs.append(f"{k}:shape")
            elif not torch.equal(a[k], b[k]):
                diffs.append(k)
        return (len(diffs) == 0), diffs[:8]

    def _max_abs_delta(a: dict, b: dict) -> float:
        m = 0.0
        for k in a:
            d = (a[k].float() - b[k].float()).abs().max().item()
            m = max(m, d)
        return m

    swapped = {"vision_tower": swap_vision, "language_model": swap_lm}
    gate: dict = {"swapped": {}, "non_swapped_bitwise_identical": {}, "param_counts": {}}

    for name in ("vision_tower", "language_model", "multi_modal_projector"):
        now = _sd_now(name)
        orig = pg_orig[name]
        # (c) key-set + param-count sanity for every submodule.
        assert set(now) == set(orig), (
            f"Gate(c) {name}: key-set changed by swap: {sorted(set(now) ^ set(orig))[:8]}"
        )
        gate["param_counts"][name] = {
            "n_keys": len(now),
            "n_params": sum(v.numel() for v in now.values()),
        }
        if name == "multi_modal_projector":
            # The connector is NEVER swapped in any config → must be PG-identical.
            identical, diffs = _bitwise_identical(now, orig)
            gate["non_swapped_bitwise_identical"][name] = identical
            assert identical, (
                f"Gate(b) LEAK: connector ({name}) changed by a swap (must be PG-identical). "
                f"Diff keys: {diffs}"
            )
            continue
        if swapped[name]:
            # (a) swapped tower must now DIFFER from PG original.
            mad = _max_abs_delta(now, orig)
            gate["swapped"][name] = {"max_abs_delta_vs_PG": mad}
            assert mad > 0.0, (
                f"Gate(a) {name} was swapped but is bitwise-equal to PG (swap did nothing). "
                f"max|Δ|={mad}"
            )
        else:
            # (b) non-swapped tower must be BITWISE-IDENTICAL to PG.
            identical, diffs = _bitwise_identical(now, orig)
            gate["non_swapped_bitwise_identical"][name] = identical
            assert identical, (
                f"Gate(b) LEAK: {name} was NOT swapped in this config but differs from PG. "
                f"Diff keys: {diffs}"
            )

    # (d) forward pass on one VQA item is finite (gray image + a real prompt).
    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    inp = proc(text=f"{VQA_PREFIX}what color is this?", images=img, return_tensors="pt")
    moved = {}
    for k, v in inp.items():
        if torch.is_tensor(v):
            v = v.to("cuda")
            if k == "pixel_values":
                v = v.to(torch.bfloat16)
        moved[k] = v
    model.eval()
    with torch.inference_mode():
        out = model(**moved)
    finite = bool(torch.isfinite(out.logits).all())
    gate["forward_finite"] = finite
    assert finite, "Gate(d): forward pass produced non-finite logits on a VQA item."
    return gate


# ===========================================================================
# Full transfer + eval for ONE (config, seed). Mirrors run_transfer_ceiling but
# builds a swapped model. The data load + train loop + eval scorer are the SAME
# frozen recipe paths (imported from transfer_a1) — identical by construction.
# ===========================================================================

def run_transfer_contrast(config_name: str, seed: int, recipe: dict | None = None) -> dict:
    """Transfer + VQAv2 eval for one swapped config + seed. Returns the per-seed
    result dict with the VQAv2 soft-accuracy score (the estimand variable input)."""
    import math

    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    rcp = dict(RECIPE)
    if recipe:
        rcp.update(recipe)

    assert torch.cuda.is_available(), "A1 contrast must run on GPU (B200/H200)."
    t0 = time.time()
    print(f"[A1/{config_name} s{seed}] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    _seed_everything(seed)

    model, proc, report = build_swapped_model(config_name, seed)
    print(f"[A1/{config_name} s{seed}] swap_gate={report['swap_gate']} "
          f"trainable={report['trainable_params']:,} ({report['trainable_pct']}%)")

    # --- data: IDENTICAL load path to run_transfer_ceiling (same fixed shuffle) ---
    DATA_SHUFFLE_SEED = 0
    n_train_shards = math.ceil(rcp["n_train"] / 3000) + 1
    train_shards = [f"data/train-{i:05d}-of-00136.parquet" for i in range(n_train_shards)]
    train_ds = load_dataset(VQA_DATASET, data_files={"train": train_shards}, split="train",
                            verification_mode="no_checks")
    eval_ds = load_dataset(VQA_DATASET, split=VQA_EVAL_SPLIT)
    assert len(train_ds) >= rcp["n_train"], (
        f"train shard subset has {len(train_ds)} < n_train={rcp['n_train']}"
    )
    train_ds = train_ds.shuffle(seed=DATA_SHUFFLE_SEED).select(range(rcp["n_train"]))
    eval_ds = eval_ds.shuffle(seed=DATA_SHUFFLE_SEED).select(range(min(rcp["n_eval"], len(eval_ds))))
    train_ds = train_ds.map(lambda r: {"_target": _vqa_majority_answer(r)})
    print(f"[A1/{config_name} s{seed}] train={len(train_ds)} eval={len(eval_ds)}")

    def collate(batch_rows):
        keys = batch_rows[0].keys()
        rows = {k: [r[k] for r in batch_rows] for k in keys}
        return _collate_train(rows, proc, rcp["max_label_len"])

    loader = DataLoader(
        train_ds, batch_size=rcp["train_batch_size"], shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rcp["lr"], weight_decay=rcp["weight_decay"])
    sched = get_scheduler(
        rcp["lr_scheduler"], optimizer=opt,
        num_warmup_steps=int(rcp["warmup_ratio"] * rcp["max_steps"]),
        num_training_steps=rcp["max_steps"],
    )

    model.train()
    step = 0
    losses = []
    done = False
    while not done:
        for batch in loader:
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[A1/{config_name} s{seed}] non-finite loss at step {step}: {loss.item()} "
                    f"— diverged run (estimand scores this worst-case; reported, not hidden)."
                )
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"[A1/{config_name} s{seed}] step {step}/{rcp['max_steps']} loss={loss.item():.4f}")
            if step >= rcp["max_steps"]:
                done = True
                break

    train_secs = time.time() - t0
    print(f"[A1/{config_name} s{seed}] training done in {train_secs:.1f}s "
          f"loss[start={losses[0]:.3f} end={np.mean(losses[-10:]):.3f}]")

    # --- eval: greedy generate, official VQA soft-accuracy (IDENTICAL to CEILING) ---
    model.eval()
    n = len(eval_ds)
    acc_sum = 0.0
    n_scored = 0
    samples = []
    for start in range(0, n, rcp["eval_batch_size"]):
        rows = eval_ds[start:start + rcp["eval_batch_size"]]
        questions = rows["question"]
        images = [im.convert("RGB") for im in rows["image"]]
        gt_lists = [[a["answer"] for a in ann] for ann in rows["answers"]]
        prompts = [f"{VQA_PREFIX}{q}" for q in questions]

        inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        in_len = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=rcp["max_new_tokens"], do_sample=False)
        decoded = proc.batch_decode(gen[:, in_len:], skip_special_tokens=True)

        for q, pred, gts in zip(questions, decoded, gt_lists):
            acc = vqa_soft_accuracy(pred.strip(), gts)
            acc_sum += acc
            n_scored += 1
            if len(samples) < 30:
                samples.append({"question": q, "pred": pred.strip(),
                                "gt_sample": gts[:3], "acc": acc})
    vqa_acc = acc_sum / n_scored
    elapsed = time.time() - t0
    print(f"[A1/{config_name} s{seed}] DONE vqa_acc={vqa_acc:.4f} (N={n_scored}) total={elapsed:.1f}s")

    return {
        "config": config_name,
        "seed": seed,
        "task": rcp["task"],
        "metric": "VQAv2 soft-accuracy (official vqaEval, min(1,#match/3))",
        "score": vqa_acc,
        "score_pct": round(100.0 * vqa_acc, 2),
        "n_eval": n_scored,
        "n_train": len(train_ds),
        "recipe": rcp,
        "train_loss_start": losses[0],
        "train_loss_end_mean10": float(np.mean(losses[-10:])),
        "trainable_report": report,
        "train_secs": round(train_secs, 1),
        "elapsed_sec": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
        "dataset": VQA_DATASET,
        "eval_split": VQA_EVAL_SPLIT,
        "train_split": VQA_TRAIN_SPLIT,
        "sample_predictions": samples,
    }
