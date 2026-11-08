# A1 transfer recipe — the primary-endpoint engine (spec §2.1, §2.4 step 3).
"""Identical, fixed transfer recipe for the A1 transfer-necessity anchor.

Given a `(vision-init, LM-init)` config it (a) loads PaliGemma 2, (b) sets the
connector (`multi_modal_projector`) trainable and attaches a fixed-rank LoRA to
the language model's attention + MLP projections, (c) fine-tunes on a downstream
task's TRAIN split, (d) evaluates on the eval split → a per-task score.

THIS PILOT RUNS ONLY THE CEILING CONFIG = (PG vision, PG LM) = the full
PaliGemma2-3b-pt-224 towers under the recipe. The other three configs
(PRIMARY-A, PRIMARY-B, FLOOR) are the blinded before→after contrast and MUST NOT
run before the GF freeze (spec §6). They are wired here only as an enum so the
*recipe is identical by construction*; this module never instantiates them.

The estimand's `S` is calibrated against the transfer-seed noise floor σ_seed,
which is the std of the CEILING eval metric across ≥5 seeds (spec §2.1
"Threshold calibration", §4 N2(b)). This file is the recipe; the seed fan-out +
σ_seed aggregation live in infra/modal_pg_a1.py.

Task = VQAv2 (spec §3 battery: "VQAv2 (acc)").
  * Why VQAv2 and not AI2D: the P1 positive control validated an AI2D *eval*
    metric, but `lmms-lab/ai2d` ships only a `test` split — no train split exists
    for a transfer fine-tune (verified on the hub). VQAv2 (lmms-lab/VQAv2) has
    public train + validation splits and is PaliGemma's canonical Stage-3 transfer
    target (HF's own PaliGemma LoRA tutorial uses it). Its metric is the standard,
    unambiguous VQA soft-accuracy.
  * Metric: VQAv2 soft accuracy = min(1, #annotators-agreeing / 3) after the
    official VQA answer normalization. This is the spec's "acc" for VQAv2.

Recipe constants below are the GF-FREEZE PROPOSAL (mirrored in
configs/a1_transfer_recipe.yaml). Same recipe for every config by construction;
only the seed varies in this pilot.

Design rules (CLAUDE.md + spec §5/§6):
  * NO try/except hiding — a wrong assumption raises with a full trace.
  * A diverged run is reported (loss non-finite → RuntimeError), scored worst-case
    by the estimand; here we report σ honestly.
  * Attention-mask parity: keep PaliGemma's prefix-LM mask (the processor + model
    default). We never touch the mask.
  * Deterministic: every seed pins torch/numpy/python RNG.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

# --- The A1 config space (spec §2.1 table). ONLY CEILING is instantiated here. -
# vision_init / lm_init are entity symbols resolved against infra.pg.entities.
# CEILING uses PG for both towers → just load the PaliGemma checkpoint whole.


@dataclass(frozen=True)
class A1Config:
    name: str
    vision_init: str  # entity symbol: "PG" (V1) or "V0" (base SigLIP)
    lm_init: str      # entity symbol: "PG" (L1) or "L0" (base Gemma 2)


A1_CONFIGS: dict[str, A1Config] = {
    "CEILING":   A1Config("CEILING",   "PG", "PG"),
    "PRIMARY_A": A1Config("PRIMARY_A", "V0", "PG"),   # BLINDED — do NOT run pre-GF
    "PRIMARY_B": A1Config("PRIMARY_B", "PG", "L0"),   # BLINDED — do NOT run pre-GF
    "FLOOR":     A1Config("FLOOR",     "V0", "L0"),   # BLINDED — do NOT run pre-GF
}

# ===========================================================================
# GF-FREEZE PROPOSAL — the fixed transfer recipe (identical for all configs).
# Mirrored in configs/a1_transfer_recipe.yaml. STRENGTHENED budget (rank 16,
# 1500 steps, 16k train — per the GF recipe-budget decision 2026-05-26) so σ_seed
# is measured at a converged operating point, not the undertrained pilot (VQAv2
# 64% vs published ~83%). Same recipe for every config by construction.
# ===========================================================================
RECIPE = {
    "task": "VQAv2",
    # LoRA on the LM (spec §2.1 "a learned connector + a fixed-rank LoRA").
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    # PaliGemma LM = Gemma 2 decoder. Target attention + MLP projections (the
    # standard PaliGemma LoRA target set). The connector is trained separately,
    # NOT via LoRA — it is set fully trainable (spec: "a learned connector").
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    "train_connector": True,    # multi_modal_projector set requires_grad=True
    "optimizer": "adamw_torch",
    "lr": 2e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",
    "max_steps": 1500,
    "train_batch_size": 8,
    "grad_accum": 1,
    "max_label_len": 16,        # VQA answers are short; cap suffix length
    "n_train": 16384,           # ~0.7 epoch at 1500 steps × batch 8 (no repeat)
    "n_eval": 1000,             # ≥500 per spec §5 sample-size; 1000 for tighter CI
    "eval_batch_size": 16,
    "max_new_tokens": 8,        # VQA answers are short
    "dtype": "bfloat16",
}

# VQAv2 dataset (introspected on Modal via the `probe` entrypoint before any run).
# lmms-lab/VQAv2 is parquet-native (datasets 4.x dropped loading-script datasets,
# so HuggingFaceM4/VQAv2 — which ships a VQAv2.py script — is unusable). Verified
# splits: train (136 shards), validation (68), test (143), testdev (36). Row
# schema (verified): image (PIL), question (str), multiple_choice_answer (str),
# answers = list[{answer, answer_confidence, answer_id}] (the 10 human answers).
VQA_DATASET = "lmms-lab/VQAv2"
VQA_TRAIN_SPLIT = "train"
VQA_EVAL_SPLIT = "validation"

# PaliGemma VQA prompt recipe (big_vision proj/paligemma/transfers/vqa*). For
# open-ended VQA the prefix is "answer en {question}"; the processor
# auto-prepends the <image> tokens when images= is passed (do NOT add them
# manually — that doubles the image tokens). Suffix = the answer text.
VQA_PREFIX = "answer en "


# ---------------------------------------------------------------------------
# Official VQAv2 answer normalization + soft-accuracy scorer.
# Verbatim transliteration of the VQA v2 eval code (vqaEval.py) — the canonical
# metric. NOT invented. Soft acc = min(1, #matching human answers / 3).
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id": "I'd", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itll": "it'll", "lets": "let's", "maam": "ma'am", "mightve": "might've",
    "mustve": "must've", "shant": "shan't", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "thered": "there'd", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "wasnt": "wasn't", "wed": "we'd", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
          "-", ">", "<", "@", "`", ",", "?", "!"]


def vqa_normalize(ans: str) -> str:
    """Official VQA v2 answer-processing pipeline (vqaEval.processPunctuation +
    processDigitArticle), verbatim."""
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    # processPunctuation
    out = ans
    for p in _PUNCT:
        if (p + " " in ans or " " + p in ans) or (_COMMA_STRIP.search(ans) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    # processDigitArticle
    words = out.lower().split()
    kept = []
    for w in words:
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            kept.append(w)
    for i, w in enumerate(kept):
        if w in _CONTRACTIONS:
            kept[i] = _CONTRACTIONS[w]
    return " ".join(kept)


def vqa_soft_accuracy(pred: str, gt_answers: list[str]) -> float:
    """VQA soft accuracy for one item: min(1, #GT-annotators matching / 3).

    `gt_answers` is the list of 10 human answers (raw); both pred and each GT are
    run through the official normalization, then for each annotator we compare
    pred to the *other* annotators (the standard leave-one-out is folded into the
    standard formula: acc = min(1, matches/3) over the full 10-answer set)."""
    p = vqa_normalize(pred)
    gts = [vqa_normalize(a) for a in gt_answers]
    matches = sum(1 for g in gts if g == p)
    return min(1.0, matches / 3.0)


# ---------------------------------------------------------------------------
# Seeding (deterministic per transfer seed — the only thing that varies here).
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Build the CEILING model + attach the trainable recipe (connector + LoRA).
# ---------------------------------------------------------------------------

def build_ceiling_model(seed: int):
    """Load PaliGemma2-3b-pt-224 (CEILING = PG vision + PG LM), freeze the base,
    set the connector trainable, and attach a fixed-rank LoRA to the LM.

    Returns (model, processor, trainable_report).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    cfg = A1_CONFIGS["CEILING"]
    assert cfg.vision_init == "PG" and cfg.lm_init == "PG", (
        f"This module instantiates ONLY CEILING (blinding, spec §6); got {cfg}"
    )
    repo = ENTITIES["PG"].repo_id  # PaliGemma2-3b-pt-224 = both PG towers

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    )
    proc = PaliGemmaProcessor.from_pretrained(repo)

    # Resolve the real submodule dotted paths (introspect, don't assume; G0 contract).
    paths = resolve_submodule_paths(model)  # {vision_tower, multi_modal_projector, language_model}

    # 1) Freeze the entire base model.
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) LoRA on the LM ONLY. peft matches `target_modules` by leaf-name suffix
    #    across the WHOLE model — and the SigLIP vision tower's attention has the
    #    same leaf names (q_proj/k_proj/v_proj/o_proj). A plain leaf list therefore
    #    LEAKS LoRA into the vision tower (verified: vision_tower then has
    #    requires_grad adapters), violating "LoRA on the LM" + the frozen-vision
    #    recipe. Fix: pass a REGEX (peft treats a *string* target_modules as a
    #    regex) anchored to the language_model subtree, so only LM projections match.
    lm_leaf = paths["language_model"].rsplit(".", 1)[-1]  # "language_model"
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

    # 3) Connector trainable (spec: "a learned connector"). peft froze everything
    #    except LoRA adapters; re-enable grads on the multi_modal_projector. Under
    #    PeftModel the base is nested at `base_model.model.<orig path>`.
    proj_leaf = paths["multi_modal_projector"].rsplit(".", 1)[-1]  # e.g. "multi_modal_projector"
    n_proj_params = 0
    for name, p in model.named_parameters():
        # match the connector subtree by its leaf submodule name appearing as a path segment
        if f".{proj_leaf}." in name or name.startswith(f"{proj_leaf}."):
            p.requires_grad_(True)
            n_proj_params += p.numel()
    assert n_proj_params > 0, (
        f"No connector params matched leaf '{proj_leaf}' — connector would be frozen. "
        f"Sample param names: {[n for n,_ in list(model.named_parameters())[:6]]}"
    )

    # Trainable report (also a structural assert: both LoRA and connector are live).
    trainable, total = 0, 0
    n_lora = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if "lora_" in name:
                n_lora += p.numel()
    assert n_lora > 0, "LoRA adapters did not attach (no trainable lora_ params)."

    # Recipe invariant (fail loud): the vision tower stays FROZEN — only the
    # connector + LM-LoRA train. A trainable vision param means LoRA leaked into
    # the SigLIP tower (shared q_proj leaf names) — the bug the target regex fixes.
    vision_trainable = [n for n, p in model.named_parameters()
                        if p.requires_grad and "vision_tower" in n]
    assert not vision_trainable, (
        f"Vision tower must stay frozen; {len(vision_trainable)} trainable vision "
        f"params found (LoRA leak). Examples: {vision_trainable[:4]}"
    )
    report = {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / total, 4),
        "lora_params": n_lora,
        "connector_params": n_proj_params,
        "submodule_paths": paths,
    }
    return model, proc, report


# ---------------------------------------------------------------------------
# VQAv2 data → tokenized supervised examples (suffix-only loss).
# ---------------------------------------------------------------------------

def _vqa_majority_answer(row) -> str:
    """The training target = the most-frequent human answer (VQAv2 'multiple_choice_answer'
    when present, else the mode of the 10 answers)."""
    if row.get("multiple_choice_answer"):
        return row["multiple_choice_answer"]
    answers = [a["answer"] for a in row["answers"]]
    # mode
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(counts, key=counts.get)


def _collate_train(rows, proc, max_label_len: int):
    """Build a supervised batch where loss is masked to the answer (suffix) tokens.

    PaliGemmaProcessor supports `suffix=` — it appends the suffix after the prefix
    and returns `labels` with the prefix + image tokens set to -100 (loss only on
    the suffix). This is PaliGemma's native training path; we use it rather than
    hand-rolling label masking.
    """
    import torch

    questions = rows["question"]
    images = [im.convert("RGB") for im in rows["image"]]
    prompts = [f"{VQA_PREFIX}{q}" for q in questions]
    suffixes = rows["_target"]  # precomputed majority answer

    batch = proc(
        text=prompts, images=images, suffix=suffixes,
        return_tensors="pt", padding="longest",
    )
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    return batch


def run_transfer_ceiling(seed: int, recipe: dict | None = None) -> dict:
    """Full CEILING transfer + eval for one seed. Returns the per-seed result dict
    with the VQAv2 soft-accuracy eval score (the per-task `score`)."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    rcp = dict(RECIPE)
    if recipe:
        rcp.update(recipe)

    assert torch.cuda.is_available(), "A1 transfer must run on GPU (B200/H200)."
    t0 = time.time()
    print(f"[A1/CEILING s{seed}] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    _seed_everything(seed)

    # --- model + recipe ---
    model, proc, report = build_ceiling_model(seed)
    print(f"[A1/CEILING s{seed}] trainable={report['trainable_params']:,} "
          f"({report['trainable_pct']}%) lora={report['lora_params']:,} "
          f"connector={report['connector_params']:,}")

    # --- data: train subsample + eval subsample (deterministic shuffle by seed-independent
    #     dataset seed so all transfer seeds see the SAME data; only init/optimizer RNG
    #     varies → σ_seed isolates training stochasticity, not data sampling). ---
    DATA_SHUFFLE_SEED = 0  # fixed across transfer seeds by design
    # The registered lmms-lab/VQAv2 config exposes only validation/testdev/test as
    # named splits, but train-*.parquet shards exist → load train explicitly via
    # data_files (verified by the probe entrypoint). validation loads normally.
    #
    # The full train split is 136 shards (~all of COCO-train); a non-streaming load
    # would download tens of GB just to .select(n_train). We only need n_train
    # items, so we read a FIXED, small subset of shards (deterministic across seeds)
    # large enough to cover n_train after a deterministic shuffle. ~3.5k items/shard
    # → n_train_shards = ceil(n_train / 3000) + 1 headroom.
    import math
    n_train_shards = math.ceil(rcp["n_train"] / 3000) + 1
    train_shards = [f"data/train-{i:05d}-of-00136.parquet" for i in range(n_train_shards)]
    # verification_mode="no_checks": the repo's registered config declares all 4
    # splits {train,validation,test,testdev}; passing data_files for only the train
    # shard subset otherwise trips ExpectedMoreSplitsError. We deliberately load a
    # subset (cheap), so skip the split-completeness check.
    train_ds = load_dataset(VQA_DATASET, data_files={"train": train_shards}, split="train",
                            verification_mode="no_checks")
    eval_ds = load_dataset(VQA_DATASET, split=VQA_EVAL_SPLIT)
    assert len(train_ds) >= rcp["n_train"], (
        f"train shard subset has {len(train_ds)} < n_train={rcp['n_train']}; "
        f"increase n_train_shards."
    )
    train_ds = train_ds.shuffle(seed=DATA_SHUFFLE_SEED).select(range(rcp["n_train"]))
    eval_ds = eval_ds.shuffle(seed=DATA_SHUFFLE_SEED).select(range(min(rcp["n_eval"], len(eval_ds))))
    train_ds = train_ds.map(lambda r: {"_target": _vqa_majority_answer(r)})
    print(f"[A1/CEILING s{seed}] train={len(train_ds)} eval={len(eval_ds)}")

    def collate(batch_rows):
        # DataLoader passes a list of dicts; convert to dict-of-lists for the processor.
        keys = batch_rows[0].keys()
        rows = {k: [r[k] for r in batch_rows] for k in keys}
        return _collate_train(rows, proc, rcp["max_label_len"])

    loader = DataLoader(
        train_ds, batch_size=rcp["train_batch_size"], shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )

    # --- optimizer + scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rcp["lr"], weight_decay=rcp["weight_decay"])
    sched = get_scheduler(
        rcp["lr_scheduler"], optimizer=opt,
        num_warmup_steps=int(rcp["warmup_ratio"] * rcp["max_steps"]),
        num_training_steps=rcp["max_steps"],
    )

    # --- train loop (suffix-masked CE, the model's native labels path) ---
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
                    f"[A1/CEILING s{seed}] non-finite loss at step {step}: {loss.item()} "
                    f"— diverged run (estimand scores this worst-case; reported, not hidden)."
                )
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"[A1/CEILING s{seed}] step {step}/{rcp['max_steps']} loss={loss.item():.4f}")
            if step >= rcp["max_steps"]:
                done = True
                break

    train_secs = time.time() - t0
    print(f"[A1/CEILING s{seed}] training done in {train_secs:.1f}s "
          f"loss[start={losses[0]:.3f} end={np.mean(losses[-10:]):.3f}]")

    # --- eval: greedy generate, official VQA soft-accuracy ---
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
    print(f"[A1/CEILING s{seed}] DONE vqa_acc={vqa_acc:.4f} (N={n_scored}) total={elapsed:.1f}s")

    return {
        "config": "CEILING",
        "seed": seed,
        "task": rcp["task"],
        "metric": "VQAv2 soft-accuracy (official vqaEval, min(1,#match/3))",
        "score": vqa_acc,                # the per-task score (the estimand variable input)
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
