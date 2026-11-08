# P3 — A1 contrast retargeted to PaliGemma2-10b (pre-registered scaling check, spec §2.5).
"""The 10B replication of the A1 transfer-necessity contrast (Q3).

Question: does the per-tower split measured at 3B — vision TRANSLATED (S̄_A≈0.93),
LM REWRITTEN (S̄_B≈0.16) — HOLD at 10B, or flip? PG2 itself shows the 27B-from-scratch
model behaves unlike the distilled ones, so this is a genuine open question.

This module is the 10B analogue of ``infra/pg/contrast_a1.py``. The swap LOGIC is
reused VERBATIM (the helpers bind to real state_dict keys/shapes — architecture-agnostic,
they HALT on any mismatch). Only three things change for 10B:

  1. **L0 (base LM) = google/gemma-2-9b** — PG2-10b's LM is Gemma2-**9B** (not 2B).
     Verified at probe time against ``model.language_model`` (vocab/hidden/layers).
  2. **V0 (base vision) = the SAME SigLIP-So400m @224** — PG2-10b is also a 224 model,
     so the identical public init is reused; verified to match the 10b ``vision_tower``.
  3. **CEILING is BUILT here** (PG vision + PG LM). The 3B contrast reused a 5-seed pilot
     CEILING; 10B has no pilot, so S needs CEILING measured under the SAME frozen recipe.
     CEILING is just PG2-10b loaded whole (no swap) + the frozen recipe attached.

EVERYTHING ELSE is the FROZEN A1 recipe imported verbatim from transfer_a1 (RECIPE,
VQAv2 data path, official soft-accuracy eval, seeding, collator). No recipe change.

Design rules (CLAUDE.md + spec §6 + queue guardrails):
  * NO try/except hiding — a wrong assumption raises with a full trace.
  * NEVER touch the frozen transfer_a1.RECIPE / data / eval — imported verbatim.
  * Verification gates (build-time, fail loud) BEFORE any training, reusing
    contrast_a1._verify_swap unchanged: (a) swapped tower differs from PG;
    (b) every non-swapped submodule bitwise-identical to PG; (c) shapes sane;
    (d) a forward on one VQA item is finite.
  * Fail loud on a gemma-2-9b ↔ PG2-10b-LM mismatch (vocab/hidden/layers).
"""
from __future__ import annotations

import time

# The FROZEN A1 recipe + data + eval — imported verbatim, identical by construction.
from infra.pg.transfer_a1 import (
    A1_CONFIGS,
    RECIPE,
    VQA_DATASET,
    VQA_EVAL_SPLIT,
    VQA_PREFIX,
    VQA_TRAIN_SPLIT,
    _collate_train,
    _seed_everything,
    _vqa_majority_answer,
    vqa_soft_accuracy,
)
# The swap LOGIC, reused verbatim (key/shape-bound, architecture-agnostic, fail-loud).
from infra.pg.contrast_a1 import (
    _lm_swap_state_dict,
    _verify_swap,
    _vision_swap_state_dict,
)

# ---------------------------------------------------------------------------
# 10B entities (the ONLY thing retargeted vs the 3B module).
# ---------------------------------------------------------------------------
# PaliGemma2-10b-pt-224. Its LM init is Gemma2-9B (PG2 paper §2: the 3b/10b/28b
# variants are built on Gemma2 2b/9b/27b). Its vision tower is the SAME SigLIP-So400m
# @224 as the 3b — both are 224 models, 256 image tokens, 256 position embeddings.
PG_10B_REPO = "google/paligemma2-10b-pt-224"
V0_SIGLIP_REPO = "google/siglip-so400m-patch14-224"   # same public init as 3B
L0_GEMMA_REPO = "google/gemma-2-9b"                    # PG2-10b's LM init (NOT 2b)

# All FOUR configs are built here (CEILING included — no pilot to reuse at 10B).
ALL_CONFIGS = ("CEILING", "PRIMARY_A", "PRIMARY_B", "FLOOR")
SWAP_CONFIGS = ("PRIMARY_A", "PRIMARY_B", "FLOOR")


# ===========================================================================
# Structure probe — bind the 10B swap to REAL keys/shapes AND verify the
# gemma-2-9b ↔ PG2-10b-LM init match (fail loud on any dim mismatch).
# ===========================================================================

def probe_swap_structure_10b() -> dict:
    """Load PG2-10b + the two init checkpoints; report exact vision_tower / language_model
    keys & shapes so the swap binds to real structure, and ASSERT that gemma-2-9b is
    PG2-10b's actual LM init (vocab/hidden/layers) and the SigLIP matches its vision tower.

    HALTS (raises) on any structural mismatch. No swap, no training here.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        PaliGemmaForConditionalGeneration,
        SiglipVisionModel,
    )

    from infra.pg.entities import resolve_submodule_paths

    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG_10B_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa",
    )
    paths = resolve_submodule_paths(pg)
    vt = pg.get_submodule(paths["vision_tower"])
    lm = pg.get_submodule(paths["language_model"])
    pg_vt_sd = vt.state_dict()
    pg_lm_sd = lm.state_dict()

    # --- PG2-10b LM config (what gemma-2-9b must match) -----------------------
    pg_text_cfg = pg.config.text_config
    pg_lm_dims = {
        "vocab_size": int(pg_text_cfg.vocab_size),
        "hidden_size": int(pg_text_cfg.hidden_size),
        "num_hidden_layers": int(pg_text_cfg.num_hidden_layers),
        "num_attention_heads": int(pg_text_cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(pg_text_cfg, "num_key_value_heads", pg_text_cfg.num_attention_heads)),
        "intermediate_size": int(pg_text_cfg.intermediate_size),
        "head_dim": int(getattr(pg_text_cfg, "head_dim", pg_text_cfg.hidden_size // pg_text_cfg.num_attention_heads)),
    }
    n_pg_decoder_layers = sum(1 for k in pg_lm_sd if k.startswith("layers.") and k.endswith(".input_layernorm.weight"))

    # --- standalone base Gemma2-9B (L0) ---------------------------------------
    g2 = AutoModelForCausalLM.from_pretrained(
        L0_GEMMA_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    )
    g2_cfg = g2.config
    g2_dims = {
        "vocab_size": int(g2_cfg.vocab_size),
        "hidden_size": int(g2_cfg.hidden_size),
        "num_hidden_layers": int(g2_cfg.num_hidden_layers),
        "num_attention_heads": int(g2_cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(g2_cfg, "num_key_value_heads", g2_cfg.num_attention_heads)),
        "intermediate_size": int(g2_cfg.intermediate_size),
        "head_dim": int(getattr(g2_cfg, "head_dim", g2_cfg.hidden_size // g2_cfg.num_attention_heads)),
    }
    g2_sd = g2.model.state_dict()  # the decoder (Gemma2Model) — aligns with PG's language_model

    # ASSERT the init match on the dims that the swap depends on. PG2 extends the
    # vocab (image/loc/seg tokens), so vocab_size legitimately DIFFERS (PG > Gemma2);
    # we KEEP PG's embeddings and swap only the decoder, so the swap-critical dims are
    # hidden_size / num_hidden_layers / heads / intermediate_size — those MUST match.
    swap_critical = ["hidden_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "intermediate_size", "head_dim"]
    mismatched = {k: {"pg10b_lm": pg_lm_dims[k], "gemma2_9b": g2_dims[k]}
                  for k in swap_critical if pg_lm_dims[k] != g2_dims[k]}
    if mismatched:
        raise RuntimeError(
            f"P3 HALT: gemma-2-9b is NOT PG2-10b's LM init — swap-critical dims differ: "
            f"{mismatched}. PG2-10b LM dims={pg_lm_dims}; gemma-2-9b dims={g2_dims}. "
            f"Re-confirm the base-LM repo before any swap."
        )
    if n_pg_decoder_layers != g2_dims["num_hidden_layers"]:
        raise RuntimeError(
            f"P3 HALT: PG2-10b decoder has {n_pg_decoder_layers} layers but gemma-2-9b "
            f"has {g2_dims['num_hidden_layers']} — architectures do not align."
        )
    vocab_note = (
        f"PG2-10b vocab={pg_lm_dims['vocab_size']} > gemma-2-9b vocab={g2_dims['vocab_size']} "
        f"(PG2 extended the table with multimodal tokens; embeddings are KEPT from PG, "
        f"only the decoder is swapped — expected, not a mismatch)."
    )

    # --- standalone SigLIP-So400m @224 (V0) -----------------------------------
    sig = SiglipVisionModel.from_pretrained(
        V0_SIGLIP_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    sig_sd = sig.state_dict()

    def _pe_shape(sd, needle):
        for k, v in sd.items():
            if needle in k:
                return {"key": k, "shape": list(v.shape)}
        return None

    pg_vt_keys, sig_keys = set(pg_vt_sd), set(sig_sd)
    shared_vt = pg_vt_keys & sig_keys
    vt_shape_mismatch = {
        k: {"pg": list(pg_vt_sd[k].shape), "siglip": list(sig_sd[k].shape)}
        for k in shared_vt if pg_vt_sd[k].shape != sig_sd[k].shape
    }
    vision_init_match = (not (pg_vt_keys - sig_keys)) and (len(vt_shape_mismatch) == 0)

    pg_lm_keys, g2_keys = set(pg_lm_sd), set(g2_sd)
    shared_lm = pg_lm_keys & g2_keys
    lm_decoder_keys_to_swap = {k for k in shared_lm if "embed_tokens" not in k}
    lm_decoder_shape_ok = all(pg_lm_sd[k].shape == g2_sd[k].shape for k in lm_decoder_keys_to_swap)

    return {
        "pg_repo": PG_10B_REPO,
        "submodule_paths": paths,
        "lm_init_match": {
            "base_lm_repo": L0_GEMMA_REPO,
            "pg10b_lm_dims": pg_lm_dims,
            "gemma2_9b_dims": g2_dims,
            "swap_critical_match": True,           # asserted above (else raised)
            "n_pg_decoder_layers": n_pg_decoder_layers,
            "vocab_note": vocab_note,
        },
        "vision_tower": {
            "v0_repo": V0_SIGLIP_REPO,
            "pg_n_keys": len(pg_vt_keys),
            "siglip_n_keys": len(sig_keys),
            "n_shared": len(shared_vt),
            "n_shape_mismatch": len(vt_shape_mismatch),
            "shape_mismatch_sample": dict(list(vt_shape_mismatch.items())[:8]),
            "only_in_pg": sorted(pg_vt_keys - sig_keys)[:20],
            "vision_init_direct_match": bool(vision_init_match),
            "pg_pos_embed": _pe_shape(pg_vt_sd, "position_embedding"),
            "siglip_pos_embed": _pe_shape(sig_sd, "position_embedding"),
        },
        "language_model": {
            "pg_n_keys": len(pg_lm_keys),
            "gemma2_9b_decoder_n_keys": len(g2_keys),
            "n_shared_with_decoder": len(shared_lm),
            "n_decoder_keys_to_swap": len(lm_decoder_keys_to_swap),
            "decoder_shape_ok_excl_embed": bool(lm_decoder_shape_ok),
            "pg_embed_tokens": _pe_shape(pg_lm_sd, "embed_tokens"),
            "gemma2_embed_tokens": _pe_shape(g2_sd, "embed_tokens"),
            "pg_has_final_norm": any(k == "norm.weight" for k in pg_lm_keys),
            "decoder_has_final_norm": any(k == "norm.weight" for k in g2_keys),
        },
    }


# ===========================================================================
# Recipe attachment — IDENTICAL to contrast_a1 / build_ceiling_model by construction
# (same regex-anchored LoRA on the LM subtree, connector trainable, vision frozen).
# Factored out so CEILING (no swap) and the swapped configs share ONE attach path.
# ===========================================================================

def _attach_frozen_recipe(model, paths) -> dict:
    """Freeze base, attach the frozen RECIPE: regex-anchored LoRA on the LM subtree +
    connector trainable. Returns the trainable report. Asserts vision stays frozen
    (LoRA must NOT leak into the SigLIP tower's same-named q/k/v/o projections)."""
    from peft import LoraConfig, get_peft_model

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
    return {
        "model": model,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / total, 4),
        "lora_params": n_lora,
        "connector_params": n_proj_params,
    }


# ===========================================================================
# Build one (config) model on PG2-10b: load whole, swap tower(s) per config (none
# for CEILING), run the four build-time gates, attach the frozen recipe.
# ===========================================================================

def build_model_10b(config_name: str, seed: int):
    """Load PaliGemma2-10b-pt-224, optionally swap a tower's init per `config_name`
    (CEILING swaps nothing), run the build-time verification gates (skipped for
    CEILING — nothing is swapped), then attach the frozen recipe. Returns
    (model, processor, report).

    Memory: PG2-10b in bf16 (~20GB) loads on cuda; base init checkpoints (SigLIP ~0.9GB,
    gemma-2-9b ~18GB) load on CPU, are copied into PG's submodules, then freed. A single
    B200/H200 holds the 10B model + LoRA comfortably.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
        SiglipVisionModel,
    )

    from infra.pg.entities import resolve_submodule_paths

    assert config_name in ALL_CONFIGS, f"unknown config {config_name!r}; expected {ALL_CONFIGS}"
    cfg = A1_CONFIGS[config_name]
    swap_vision = cfg.vision_init == "V0"
    swap_lm = cfg.lm_init == "L0"

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PG_10B_REPO, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    )
    proc = PaliGemmaProcessor.from_pretrained(PG_10B_REPO)
    paths = resolve_submodule_paths(model)

    swap_info: dict = {}
    gate: dict | None = None

    if swap_vision or swap_lm:
        # snapshot PG-ORIGINAL state_dicts (CPU) for the bitwise-leak gate (b).
        pg_orig = {
            name: {k: v.detach().clone().cpu()
                   for k, v in model.get_submodule(path).state_dict().items()}
            for name, path in paths.items()
        }

        if swap_vision:
            sig = SiglipVisionModel.from_pretrained(
                V0_SIGLIP_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
            )
            vt = model.get_submodule(paths["vision_tower"])
            new_vt_sd, vinfo = _vision_swap_state_dict(vt.state_dict(), sig.state_dict())
            new_vt_sd = {k: v.to("cuda") for k, v in new_vt_sd.items()}
            missing, unexpected = vt.load_state_dict(new_vt_sd, strict=True)
            assert not missing and not unexpected, (
                f"Vision load_state_dict not strict-clean: missing={missing} unexpected={unexpected}"
            )
            swap_info["vision"] = {"source": V0_SIGLIP_REPO, **vinfo}
            del sig

        if swap_lm:
            g2 = AutoModelForCausalLM.from_pretrained(
                L0_GEMMA_REPO, torch_dtype=torch.bfloat16, device_map="cpu",
                attn_implementation="eager",
            )
            lm = model.get_submodule(paths["language_model"])
            # PG's language_model is the Gemma2 DECODER (Gemma2Model); g2.model is the
            # matching standalone decoder (keys align — no `model.` prefix, no lm_head).
            new_lm_sd, linfo = _lm_swap_state_dict(lm.state_dict(), g2.model.state_dict())
            new_lm_sd = {k: v.to("cuda") for k, v in new_lm_sd.items()}
            missing, unexpected = lm.load_state_dict(new_lm_sd, strict=True)
            assert not missing and not unexpected, (
                f"LM load_state_dict not strict-clean: missing={missing} unexpected={unexpected}"
            )
            swap_info["lm"] = {"source": L0_GEMMA_REPO, **linfo}
            del g2

        # The FOUR build-time gates, reused verbatim from the 3B engine.
        gate = _verify_swap(model, paths, pg_orig, swap_vision, swap_lm, proc)

    # attach the frozen recipe (identical to the 3B path by construction).
    att = _attach_frozen_recipe(model, paths)
    model = att["model"]

    report = {
        "config": config_name,
        "model": "paligemma2-10b-pt-224",
        "vision_init": cfg.vision_init,
        "lm_init": cfg.lm_init,
        "swap_info": swap_info,
        "swap_gate": gate,                       # None for CEILING (nothing swapped)
        "trainable_params": att["trainable_params"],
        "total_params": att["total_params"],
        "trainable_pct": att["trainable_pct"],
        "lora_params": att["lora_params"],
        "connector_params": att["connector_params"],
        "submodule_paths": paths,
    }
    return model, proc, report


# ===========================================================================
# Full transfer + eval for ONE (config, seed). The data load + train loop + eval
# scorer are the SAME frozen recipe paths as transfer_a1.run_transfer_ceiling /
# contrast_a1.run_transfer_contrast — identical by construction, retargeted to 10B.
# ===========================================================================

def run_transfer_10b(config_name: str, seed: int, recipe: dict | None = None) -> dict:
    """Transfer + VQAv2 eval for one 10B config + seed. Returns the per-seed result
    dict with the VQAv2 soft-accuracy score (the estimand variable input)."""
    import math

    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    rcp = dict(RECIPE)
    if recipe:
        rcp.update(recipe)

    assert torch.cuda.is_available(), "P3 (10B) must run on GPU (B200/H200)."
    t0 = time.time()
    print(f"[P3/{config_name} s{seed}] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    _seed_everything(seed)

    model, proc, report = build_model_10b(config_name, seed)
    print(f"[P3/{config_name} s{seed}] swap_gate={report['swap_gate']} "
          f"trainable={report['trainable_params']:,} ({report['trainable_pct']}%)")

    # --- data: IDENTICAL load path to the frozen recipe (same fixed shuffle seed) ---
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
    print(f"[P3/{config_name} s{seed}] train={len(train_ds)} eval={len(eval_ds)}")

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
                    f"[P3/{config_name} s{seed}] non-finite loss at step {step}: {loss.item()} "
                    f"— diverged run (estimand scores this worst-case; reported, not hidden)."
                )
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"[P3/{config_name} s{seed}] step {step}/{rcp['max_steps']} loss={loss.item():.4f}")
            if step >= rcp["max_steps"]:
                done = True
                break

    train_secs = time.time() - t0
    print(f"[P3/{config_name} s{seed}] training done in {train_secs:.1f}s "
          f"loss[start={losses[0]:.3f} end={np.mean(losses[-10:]):.3f}]")

    # --- eval: greedy generate, official VQA soft-accuracy (IDENTICAL to the recipe) ---
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
    print(f"[P3/{config_name} s{seed}] DONE vqa_acc={vqa_acc:.4f} (N={n_scored}) total={elapsed:.1f}s")

    return {
        "config": config_name,
        "model": "paligemma2-10b-pt-224",
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
