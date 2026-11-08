"""W9 Diff-SAE — paired before/after activation harvest (faithful port of the W4 harness).

Reproduces, with NO Modal dependency, the EXACT paired-contrast harvest the Delta-Crosscoder
(modal_pg_w4_crosscoder.py) used, so the two instruments are directly comparable (prereg §2,
§4 forward-condition parity C2):

  AFTER  = PG2-3b-pt-224 model.language_model residual at layer L.
  BEFORE = the SAME PaliGemma forward with the 26 decoder layers + final norm reset to base
           gemma-2-2b (contrast_a1._lm_swap_state_dict; vision_tower + projector +
           extended-vocab embeddings KEPT). Only the decoder weights differ → Δa = a_after −
           a_before is the weight-induced change at matched inputs.

This module also provides:
  * build_random_after  — the N3 random-delta null: a third PG2 whose decoder layers are
    re-initialised to scale-matched Gaussian noise (parent N3, random-weight model). Δa_random
    = a(random-decoder) − a(before) sets the ρ_novel / R_KL floor.
  * train_crosscoder / crosscoder_fvu — the lib BatchTopK crosscoder reused for the §9.1
    preflight FVU reproduction and the §6.2 R_KL crosscoder recon / error terms.

Discipline: no try/except hiding; lib/shape mismatches RAISE. Heavy imports inside functions
(so the module py_compiles without torch on the control plane).
"""
from __future__ import annotations

# frozen harvest config — VERBATIM the W4 MM_PROBES + layers so the regime matches the result
# this instrument cross-checks (prereg §2 layers chosen to align with the crosscoder readouts).
LAYERS = (5, 12, 20)


def hf_online_setup():
    """Set HF_TOKEN (from SSM via the instance role) + online mode, so the GATED PaliGemma/Gemma
    repos load (cache-backed but authenticated — run_job.sh's HF_HUB_OFFLINE=1 would force an
    unauthenticated offline path that still 401s on a gated metadata check) AND the probe datasets
    can stream. Call FIRST in a job main(), before any transformers/datasets import. Fail-loud."""
    import os
    import subprocess
    if os.environ.get("HF_TOKEN"):
        # already provisioned (e.g. a Vast box with no aws CLI/creds — the launcher ships the
        # read-scoped token via env); just flip online mode.
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        return
    tok = subprocess.run(
        ["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token", "--with-decryption",
         "--query", "Parameter.Value", "--output", "text", "--region", "us-east-1"],
        check=True, capture_output=True, text=True).stdout.strip()
    if not tok:
        raise RuntimeError("empty HF token from SSM /pg-lineage/hf-token")
    os.environ["HF_TOKEN"] = tok
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"


def ensure_wandb():
    """dictionary_learning/training.py imports wandb at module top (we always pass use_wandb=False).
    The Modal image had it; the AWS AMI bake didn't — pip-install it once at job start (idempotent,
    fail-loud). bake_env.sh now also pins it so future AMIs include it and this is a no-op."""
    import importlib.util
    import subprocess
    import sys
    if importlib.util.find_spec("wandb") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)


MM_PROBES = (
    {"task": "refcoco", "dataset": "lmms-lab/RefCOCO", "split": "testA",
     "prompt_fmt": "detect {expr}\n", "expr_keys": ("answer", "sentences", "sent"),
     "n_images": 384},
    {"task": "textvqa", "dataset": "lmms-lab/textvqa", "split": "validation",
     "prompt_fmt": "answer en {q}\n", "q_key": "question", "n_images": 384},
)
SEED = 1337


# ===========================================================================
# BEFORE (decoder-reset PG2) + AFTER (PG2) — the W3 leak-gated build.
# ===========================================================================

def build_before_after(device, pg_key="PG", attn_impl="sdpa"):
    """Return (pg_after, pg_before, proc, paths, image_token_index, gate).

    AFTER = PG2 as released (pg_key="PG"=pt default, or "PG_mix" for the base→mix second-ΔW
    generality axis — same Gemma-2-2b architecture/interface, so the base swap is identical).
    BEFORE = a second PG2 with decoder reset to base gemma-2-2b.
    Fail-loud gates: swapped decoder DIFFERS from PG; vision_tower + projector + embed_tokens
    BITWISE-IDENTICAL to PG (no leak); both forwards finite.

    attn_impl: attention backend for the AFTER/BEFORE PaliGemma forwards. Default "sdpa"
    (unchanged for existing residual-harvest callers, W9/W6 — they read mid-stack residuals,
    not the prefix-LM read-out). Pass "eager" when SCORING the prefix-LM read-out (MMLU
    MCQ/cloze logits) — sdpa mishandles PaliGemma's prefix-LM bidirectional mask and gives the
    wrong logits/ppl (memory reference_paligemma_sdpa_wrong_forward; the latloc study §4)."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (
        AutoModelForCausalLM, PaliGemmaForConditionalGeneration, PaliGemmaProcessor,
    )

    from infra.pg.contrast_a1 import L0_GEMMA_REPO, _lm_swap_state_dict
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    repo = ENTITIES[pg_key].repo_id
    proc = PaliGemmaProcessor.from_pretrained(repo)

    pg_after = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=attn_impl,
    ).eval()
    paths = resolve_submodule_paths(pg_after)
    image_token_index = getattr(pg_after.config, "image_token_index", None)
    if image_token_index is None:
        raise RuntimeError(
            f"PG2 config has no image_token_index; cannot locate image-token positions. "
            f"config keys: {sorted(vars(pg_after.config).keys())}")

    pg_before = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=attn_impl,
    ).eval()
    lm_before = pg_before.get_submodule(paths["language_model"])

    pg_lm_orig = {k: v.detach().clone().cpu() for k, v in lm_before.state_dict().items()}
    proj_orig = {k: v.detach().clone().cpu()
                 for k, v in pg_before.get_submodule(paths["multi_modal_projector"]).state_dict().items()}
    vt_orig = {k: v.detach().clone().cpu()
               for k, v in pg_before.get_submodule(paths["vision_tower"]).state_dict().items()}

    g2 = AutoModelForCausalLM.from_pretrained(
        L0_GEMMA_REPO, revision=ENTITIES["L0"].revision, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager")
    new_lm_sd, linfo = _lm_swap_state_dict(lm_before.state_dict(), g2.model.state_dict())
    new_lm_sd = {k: v.to(device) for k, v in new_lm_sd.items()}
    missing, unexpected = lm_before.load_state_dict(new_lm_sd, strict=True)
    assert not missing and not unexpected, (
        f"BEFORE LM load_state_dict not strict-clean: missing={missing} unexpected={unexpected}")
    del g2
    torch.cuda.empty_cache()

    def _identical(now_sd, orig):
        if set(now_sd) != set(orig):
            return False, sorted(set(now_sd) ^ set(orig))[:8]
        bad = [k for k in orig if not torch.equal(now_sd[k].detach().cpu(), orig[k])]
        return len(bad) == 0, bad[:8]

    lm_now = {k: v for k, v in lm_before.state_dict().items()}
    dec_keys = [k for k in pg_lm_orig if "embed_tokens" not in k and "lm_head" not in k]
    dec_delta = max(
        (lm_now[k].detach().cpu().float() - pg_lm_orig[k].float()).abs().max().item()
        for k in dec_keys)
    assert dec_delta > 0.0, f"BEFORE decoder is bitwise-equal to PG (swap did nothing). max|Δ|={dec_delta}"
    emb_keys = [k for k in pg_lm_orig if "embed_tokens" in k or "lm_head" in k]
    for k in emb_keys:
        assert torch.equal(lm_now[k].detach().cpu(), pg_lm_orig[k]), (
            f"BEFORE embed/lm_head key {k} changed — must be KEPT (extended-vocab interface).")
    vt_id, vt_bad = _identical(
        {k: v for k, v in pg_before.get_submodule(paths["vision_tower"]).state_dict().items()}, vt_orig)
    assert vt_id, f"LEAK: BEFORE vision_tower changed (must be PG-identical). {vt_bad}"
    proj_id, proj_bad = _identical(
        {k: v for k, v in pg_before.get_submodule(paths["multi_modal_projector"]).state_dict().items()}, proj_orig)
    assert proj_id, f"LEAK: BEFORE projector changed (must be PG-identical). {proj_bad}"

    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    inp = proc(text="answer en what color is this?\n", images=img, return_tensors="pt")
    moved = {k: (v.to(device).to(torch.bfloat16) if k == "pixel_values" else v.to(device))
             for k, v in inp.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        oa = pg_after(**moved); ob = pg_before(**moved)
    assert torch.isfinite(oa.logits).all() and torch.isfinite(ob.logits).all(), \
        "Non-finite logits from before/after on a VQA item."

    gate = {
        "lm_swap_info": linfo,
        "decoder_max_abs_delta_vs_PG": dec_delta,
        "embed_lm_head_kept_identical": True,
        "vision_tower_PG_identical": vt_id,
        "projector_PG_identical": proj_id,
        "forward_finite_before": bool(torch.isfinite(ob.logits).all()),
        "forward_finite_after": bool(torch.isfinite(oa.logits).all()),
    }
    return pg_after, pg_before, proc, paths, image_token_index, gate


def build_pt_mix(device):
    """W6 PT→Mix axis. Return (pg_after, pg_before, proc, paths, image_token_index, gate) where
    AFTER = PG2-3b-MIX-224 (task-tuned) and BEFORE = PG2-3b-PT-224 (the W9 'after'). Both are
    released models with identical architecture + tokenizer (mix is a fine-tune of pt), so the
    SAME multimodal forward gives token-paired residuals — Δa_mix = a(mix) − a(pt) isolates what
    the SECOND fine-tune (task/instruction tuning) changed on top of multimodal pretraining.

    No decoder surgery and no leak gate (unlike build_before_after): both halves are real models.
    Fail-loud gates: the mix LM decoder DIFFERS from pt (the fine-tune did something); both
    forwards finite. The processor comes from pt (shared tokenizer)."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    pt_repo = ENTITIES["PG"].repo_id
    mix_repo = ENTITIES["PG_mix"].repo_id
    proc = PaliGemmaProcessor.from_pretrained(pt_repo)

    pg_after = PaliGemmaForConditionalGeneration.from_pretrained(
        mix_repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    pg_before = PaliGemmaForConditionalGeneration.from_pretrained(
        pt_repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    paths = resolve_submodule_paths(pg_after)
    image_token_index = getattr(pg_after.config, "image_token_index", None)
    if image_token_index is None:
        raise RuntimeError(f"PG2-mix config has no image_token_index. keys: {sorted(vars(pg_after.config).keys())}")

    # gate: the mix decoder must differ from pt (otherwise the fine-tune is a no-op and Δa≡0).
    lm_a = pg_after.get_submodule(paths["language_model"]).state_dict()
    lm_b = pg_before.get_submodule(paths["language_model"]).state_dict()
    dec_keys = [k for k in lm_b if "embed_tokens" not in k and "lm_head" not in k]
    dec_delta = max((lm_a[k].detach().cpu().float() - lm_b[k].detach().cpu().float()).abs().max().item()
                    for k in dec_keys)
    assert dec_delta > 0.0, f"MIX decoder is bitwise-equal to PT (fine-tune did nothing). max|Δ|={dec_delta}"

    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    inp = proc(text="answer en what color is this?\n", images=img, return_tensors="pt")
    moved = {k: (v.to(device).to(torch.bfloat16) if k == "pixel_values" else v.to(device))
             for k, v in inp.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        oa = pg_after(**moved); ob = pg_before(**moved)
    assert torch.isfinite(oa.logits).all() and torch.isfinite(ob.logits).all(), \
        "Non-finite logits from mix/pt on a VQA item."

    gate = {
        "pair": "pt-mix", "after_repo": mix_repo, "before_repo": pt_repo,
        "decoder_max_abs_delta_mix_vs_pt": dec_delta,
        "forward_finite_before": bool(torch.isfinite(ob.logits).all()),
        "forward_finite_after": bool(torch.isfinite(oa.logits).all()),
    }
    return pg_after, pg_before, proc, paths, image_token_index, gate


def build_pair(pair, device):
    """Dispatch the before/after build by axis: 'base-pt' (W9: Gemma→PT) or 'pt-mix' (W6:
    PT→Mix). Both return the same 6-tuple, so all downstream W9 machinery is pair-agnostic."""
    if pair == "base-mix":
        return build_before_after(device, pg_key="PG_mix")
    if pair == "base-pt":
        return build_before_after(device)
    if pair == "pt-mix":
        return build_pt_mix(device)
    raise ValueError(f"unknown pair {pair!r}; expected 'base-pt' or 'pt-mix'")


def build_random_after(pg_before, paths, device, seed=SEED):
    """N3 random-delta null (parent N3, random-weight model). Clone the BEFORE build and
    re-initialise its 26 decoder layers + final norm to scale-matched Gaussian noise (per
    weight tensor: same shape, mean 0, std matched to the base-gemma2 tensor it replaces;
    biases/norm-weights kept at their structural init). Vision tower + projector + embeddings
    are untouched (PG-identical), so a(random-decoder) − a(before) isolates the achievable
    'random change' floor at matched multimodal inputs.

    Returns pg_random (a fresh PaliGemma instance); the caller harvests a_after_random from it.
    """
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    from infra.pg.entities import ENTITIES

    repo = ENTITIES["PG"].repo_id
    pg_random = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    # copy the BEFORE (base-gemma2 decoder) weights in, then randomise the decoder in place so
    # the random null shares the before's scale per tensor.
    lm_random = pg_random.get_submodule(paths["language_model"])
    lm_before_sd = pg_before.get_submodule(paths["language_model"]).state_dict()
    lm_random.load_state_dict({k: v.to(device) for k, v in lm_before_sd.items()}, strict=True)

    g = torch.Generator(device="cpu").manual_seed(seed)
    new_sd = {}
    for k, v in lm_before_sd.items():
        if "embed_tokens" in k or "lm_head" in k:
            new_sd[k] = v.to(device)                      # keep the extended-vocab interface
            continue
        vf = v.detach().cpu().float()
        if vf.ndim >= 2:                                  # weight matrices → scale-matched noise
            std = vf.std().clamp_min(1e-8)
            noise = torch.randn(vf.shape, generator=g) * std
            new_sd[k] = noise.to(v.dtype).to(device)
        else:                                             # 1-D (norms/biases) → keep structural
            new_sd[k] = v.to(device)
    miss, unexp = lm_random.load_state_dict(new_sd, strict=True)
    assert not miss and not unexp, f"random-after load not strict-clean: {miss} {unexp}"
    return pg_random


# ===========================================================================
# Multimodal probe batch (image tokens) — mirrors W4 _build_mm_batch.
# ===========================================================================

def _mm_expr(row, probe) -> str:
    if probe["task"] == "refcoco":
        for k in probe["expr_keys"]:
            if k in row and row[k]:
                v = row[k]
                if isinstance(v, str):
                    return probe["prompt_fmt"].format(expr=v)
                if isinstance(v, list) and v:
                    expr = (v[0].get("raw") or v[0].get("sent") or next(iter(v[0].values()))) \
                        if isinstance(v[0], dict) else v[0]
                    return probe["prompt_fmt"].format(expr=expr)
        raise RuntimeError(f"refcoco row missing expr; keys={sorted(row.keys())}")
    if probe["task"] == "textvqa":
        return probe["prompt_fmt"].format(q=row[probe["q_key"]])
    raise RuntimeError(f"unknown probe task {probe['task']}")


def build_mm_batch(proc, image_token_index, device):
    """Load probe images+prompts, run the processor, build position masks + a content-hash
    held-out split (sha1(image_bytes) first nibble < 4 ≈ 25%, the prereg §4.confirmation half)."""
    import hashlib
    import torch
    from datasets import load_dataset

    images, prompts, is_held = [], [], []
    for probe in MM_PROBES:
        ds = load_dataset(probe["dataset"], split=probe["split"], streaming=True)
        n = 0
        for row in ds:
            img = row["image"]
            if img is None:
                continue
            img = img.convert("RGB")
            prompts.append(_mm_expr(row, probe))
            images.append(img)
            h = int(hashlib.sha1(img.tobytes()).hexdigest()[0], 16)
            is_held.append(h < 4)
            n += 1
            if n >= probe["n_images"]:
                break
        if n < probe["n_images"]:
            raise RuntimeError(f"{probe['task']}: only {n}/{probe['n_images']} images.")

    enc = proc(text=prompts, images=images, return_tensors="pt", padding="longest")
    enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    is_image = (input_ids == image_token_index)
    is_text_keep = attn.bool() & (~is_image)
    held_ex = torch.tensor(is_held, dtype=torch.bool)
    held_tok = held_ex[:, None].expand_as(input_ids)
    return {
        "input_ids": input_ids.to(device), "attention_mask": attn.to(device),
        "pixel_values": enc["pixel_values"].to(device),
        "is_image": is_image.to(device), "is_text_keep": is_text_keep.to(device),
        "held_out": held_tok.to(device),
        "n_examples": len(prompts),
    }


# ===========================================================================
# Paired residual harvest at one layer.
# ===========================================================================

def harvest_paired_layer(pg_after, pg_before, layer_idx, mm_batch, device,
                         keep_image=True, keep_text=True):
    """Forward both models micro-batched, hook the LM decoder layer L residual on each, keep
    (image | text-keep) positions. Returns (acts_before, acts_after, is_image, held) as float32
    CPU tensors, row-aligned (token i is the SAME token in both models)."""
    import torch
    from infra.pg.gemma_scope import ResidualHarvest, resolve_gemma2_layers

    a_layers = resolve_gemma2_layers(pg_after.model.language_model)
    b_layers = resolve_gemma2_layers(pg_before.model.language_model)

    ii = mm_batch["input_ids"]; am = mm_batch["attention_mask"]; pv = mm_batch["pixel_values"]
    is_image = mm_batch["is_image"]; is_text_keep = mm_batch["is_text_keep"]; held = mm_batch["held_out"]

    keep_mask = torch.zeros_like(is_image, dtype=torch.bool)
    if keep_image:
        keep_mask |= is_image
    if keep_text:
        keep_mask |= is_text_keep

    MICRO = 4
    bef_chunks, aft_chunks, img_chunks, held_chunks = [], [], [], []
    with ResidualHarvest(a_layers, layer_idx) as ha, ResidualHarvest(b_layers, layer_idx) as hb:
        for s in range(0, ii.shape[0], MICRO):
            ids = ii[s:s+MICRO]; a_ = am[s:s+MICRO]; p_ = pv[s:s+MICRO]
            km = keep_mask[s:s+MICRO].reshape(-1)
            with torch.inference_mode():
                pg_after(input_ids=ids, attention_mask=a_, pixel_values=p_)
            acts_a = ha.captured.reshape(-1, ha.captured.shape[-1])[km].float().cpu()
            with torch.inference_mode():
                pg_before(input_ids=ids, attention_mask=a_, pixel_values=p_)
            acts_b = hb.captured.reshape(-1, hb.captured.shape[-1])[km].float().cpu()
            if acts_a.shape != acts_b.shape:
                raise RuntimeError(
                    f"paired shape mismatch at L{layer_idx}: after {acts_a.shape} before {acts_b.shape}")
            aft_chunks.append(acts_a); bef_chunks.append(acts_b)
            img_chunks.append(is_image[s:s+MICRO].reshape(-1)[km].cpu())
            held_chunks.append(held[s:s+MICRO].reshape(-1)[km].cpu())
    acts_before = torch.cat(bef_chunks, 0)
    acts_after = torch.cat(aft_chunks, 0)
    is_img = torch.cat(img_chunks, 0)
    held_tok = torch.cat(held_chunks, 0)
    return acts_before, acts_after, is_img, held_tok


# ===========================================================================
# Crosscoder (lib BatchTopK) — reused for §9.1 preflight + §6.2 R_KL crosscoder terms.
# ===========================================================================

# matches the frozen W4 regime (read from the reference JSON's config at runtime; these are the
# defaults the preflight uses to reproduce the cached FVU@L20).
DICT_EXPANSION = 16
BATCHTOPK_K = 100
CC_LR = 1e-4
CC_TRAIN_STEPS = 4000
CC_BATCH = 2048
WARMUP_STEPS = 500


def train_crosscoder(acts_before, acts_after, layer_idx, seed, device,
                     shuffle_pairing=False, dict_size_override=None,
                     steps=CC_TRAIN_STEPS):
    """Train a BatchTopKCrossCoder on the paired (N,2,d) stack (faithful W4 port)."""
    import torch
    from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer
    from dictionary_learning.training import trainSAE

    g = torch.Generator().manual_seed(seed)
    if shuffle_pairing:
        perm = torch.randperm(acts_after.shape[0], generator=g)
        acts_after = acts_after[perm]

    N, d = acts_before.shape
    stack = torch.stack([acts_before, acts_after], dim=1)
    if stack.shape[1] != 2:
        raise RuntimeError(f"crosscoder stack must be (N,2,d); got {stack.shape}")
    dict_size = int(dict_size_override) if dict_size_override is not None else DICT_EXPANSION * d
    act_mean = stack.mean(0)
    act_std = stack.std(0).clamp_min(1e-6)

    def data_iter():
        idx_g = torch.Generator().manual_seed(seed + 1)
        for _ in range(steps):
            idx = torch.randint(0, N, (CC_BATCH,), generator=idx_g)
            yield stack[idx]

    trainer_cfg = {
        "trainer": BatchTopKCrossCoderTrainer,
        "steps": steps, "activation_dim": d, "dict_size": dict_size, "k": BATCHTOPK_K,
        "layer": layer_idx, "lm_name": "gemma-2-2b__paligemma2-3b-pt", "num_layers": 2,
        "lr": CC_LR, "warmup_steps": WARMUP_STEPS, "seed": seed, "device": device,
        "wandb_name": f"deltacc_L{layer_idx}_s{seed}{'_shuf' if shuffle_pairing else ''}",
        "activation_mean": act_mean.to(device), "activation_std": act_std.to(device),
    }
    ae = trainSAE(data=data_iter(), trainer_config=trainer_cfg, steps=steps,
                  use_wandb=False, dtype=torch.float32).to("cpu")
    cfg = {"dict_size": dict_size, "activation_dim": d, "k": BATCHTOPK_K, "steps": steps,
           "lr": CC_LR, "expansion": dict_size / d, "n_paired_tokens": int(N),
           "shuffle_pairing": shuffle_pairing}
    return ae, cfg


def crosscoder_fvu(ae, acts_before, acts_after, device, n_tokens=20000, seed=SEED):
    """Crosscoder FVU per stream (matches W4 _fvu — the §9.1 preflight reproduction metric)."""
    import numpy as np
    import torch
    N = acts_before.shape[0]
    ntok = min(n_tokens, N)
    sel = np.random.default_rng(seed).choice(N, ntok, replace=False)
    b = acts_before[sel].to(device); a = acts_after[sel].to(device)
    stack = torch.stack([b, a], dim=1)
    ae = ae.to(device).eval()
    with torch.inference_mode():
        recon = ae.decode(ae.encode(stack))

    def _f(x, xr):
        num = ((x - xr) ** 2).sum().item()
        den = ((x - x.mean(0, keepdim=True)) ** 2).sum().item()
        return (num / den) if den > 0 else float("nan")
    return {"fvu_before": _f(stack[:, 0], recon[:, 0]),
            "fvu_after": _f(stack[:, 1], recon[:, 1]), "n_tokens": int(ntok)}
