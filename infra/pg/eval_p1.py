# GPU policy: this runs on B200/H200 only (see infra/modal_pg_p1.py).
"""P1 positive control (spec §4 "P1 [V34]", §0.1 step 2, gate G0).

Reproduce ONE advertised PaliGemma 2 number to prove the eval harness is not
broken. If we cannot reproduce a published number, nothing downstream is
trustworthy → HALT.

TARGET (chosen, with source):
    Task   : AI2D (science-diagram multiple-choice VQA), accuracy.
    Model  : google/paligemma2-3b-mix-224  (the *mix* checkpoint — task-ready;
             the pt checkpoint is NOT task-ready, per the spec).
    Number : 74.7  (AI2D accuracy, 3B @ 224, "mix" / "224-3B" column).
    Source : model card https://huggingface.co/google/paligemma2-3b-mix-224
             (cross-checked on Google's card
              https://ai.google.dev/gemma/docs/paligemma/model-card-2
              and the PG2 paper arXiv:2412.03555 Table 13 → AI2D 74.7 ±0.5).

WHY AI2D (most reliably reproducible of the candidates):
    AI2D is *multiple-choice*: the metric is plain exact-match string equality of
    the generated answer against the correct option text (big_vision VQA scorer,
    single ground truth → `float(answer == gt)`). No VQAv2 10-annotator soft
    accuracy, no ANLS, no answer-normalization ambiguity. It also sits at a
    sensitive mid-range (74.7, not near-ceiling like ScienceQA 96.1), so a broken
    prompt/parse misses obviously rather than hiding.

PROMPTING + SCORING (the AUTHORITATIVE big_vision recipe, not invented):
    google-research/big_vision .../proj/paligemma/transfers/ai2d.py
        PREFIX     = 'answer en '
        PROMPT     = 'choose from:'
        PROMPT_SEP = ' \\t '            # space-TAB-space
        prefix = f"{PREFIX} {{question}} {PROMPT} {{ansstr}}"
        ansstr = PROMPT_SEP.join(options)
        suffix = the full answer TEXT (copy of `answer`)   # not a letter
    Scorer .../evaluators/proj/paligemma/transfers/vqa.py
        single GT  → accuracies.append(float(answer == gt))   # exact string ==
        to_lower=False for AI2D ("Model sees options in the prompt").
    The processor auto-prepends the <image> tokens when images= is passed, so we
    do NOT manually add "<image>" (doing so would double the image tokens).

NOTE on mix-vs-transfer: Google's card frames these as per-task transfer fine-tunes,
but the HF `paligemma2-3b-mix-224` card advertises 74.7 as this multi-task *mix*
checkpoint's AI2D score, and the mix model is explicitly trained to handle the
`answer en … choose from: …` prompt family. The spec directs us to the mix
checkpoint, so 74.7 is the advertised number we reproduce. Disclosed, not hidden.

Design rules (CLAUDE.md + spec §5): NO try/except hiding — a wrong assumption
raises with a full trace. Deterministic greedy decode. Distinct output subdir.
"""
from __future__ import annotations

import json
import time

# --- big_vision AI2D recipe constants (verbatim) ---------------------------
AI2D_REPO = "google/paligemma2-3b-mix-224"
AI2D_DATASET = "lmms-lab/ai2d"
AI2D_SPLIT = "test"  # big_vision 'eval' split == test (the publishable-score split)

PREFIX = "answer en "
PROMPT = "choose from:"
PROMPT_SEP = " \t "  # space-TAB-space, exactly as big_vision PROMPT_SEP


def build_prompt(question: str, options: list[str]) -> str:
    """Exact big_vision AI2D prefix: f'{PREFIX} {question} {PROMPT} {ansstr}'.

    Note the doubled space after 'en' comes from PREFIX's own trailing space plus
    the format string's space — this matches big_vision byte-for-byte.
    """
    ansstr = PROMPT_SEP.join(options)
    return f"{PREFIX} {question} {PROMPT} {ansstr}"


def _norm(s: str) -> str:
    """Whitespace-normalised, case-folded — for the *diagnostic* lenient metric only."""
    return " ".join(s.strip().lower().split())


def run_ai2d_eval(n_items: int = 500, seed: int = 0, batch_size: int = 16) -> dict:
    """Load the mix model, run AI2D on `n_items` test examples, return metrics.

    Primary metric `acc_exact`: case-sensitive exact string match of the generated
    answer against the gold option text (== big_vision's single-GT scorer with
    to_lower=False), after stripping surrounding whitespace from the generation.
    Diagnostic `acc_norm`: whitespace/case-normalised match (looser; reported for
    transparency, NOT the verdict metric).
    """
    import torch
    from datasets import load_dataset
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    t0 = time.time()
    print(f"[P1] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "P1 must run on GPU (B200/H200) — no CUDA visible."
    print(f"[P1] GPU={torch.cuda.get_device_name(0)}")

    # --- model + processor (validated load recipe from the engineering plan) ---
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        AI2D_REPO,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    proc = PaliGemmaProcessor.from_pretrained(AI2D_REPO)
    print(f"[P1] loaded {AI2D_REPO} in {time.time() - t0:.1f}s")

    # --- dataset: AI2D test split, deterministically shuffled, first n_items ----
    ds = load_dataset(AI2D_DATASET, split=AI2D_SPLIT)
    full_n = len(ds)
    n_items = min(n_items, full_n)
    ds = ds.shuffle(seed=seed).select(range(n_items))
    print(f"[P1] AI2D {AI2D_SPLIT}: using {n_items}/{full_n} items (seed={seed})")

    n_correct_exact = 0
    n_correct_norm = 0
    records = []

    for start in range(0, n_items, batch_size):
        rows = ds[start : start + batch_size]
        # `datasets` slicing returns dict-of-lists.
        questions = rows["question"]
        options_b = rows["options"]
        answers_b = rows["answer"]      # STRING index into options (e.g. "1")
        images_b = rows["image"]

        prompts = [build_prompt(q, opts) for q, opts in zip(questions, options_b)]
        images = [im.convert("RGB") for im in images_b]

        model_inputs = proc(
            text=prompts, images=images, return_tensors="pt", padding="longest"
        ).to(model.device)
        # pixel_values must be bf16 to match the model; ids/masks stay integer.
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.bfloat16)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            gen = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        gen_new = gen[:, input_len:]
        decoded = proc.batch_decode(gen_new, skip_special_tokens=True)

        for q, opts, ans_idx, pred_raw in zip(questions, options_b, answers_b, decoded):
            # `lmms-lab/ai2d` stores `answer` as a STRING index (e.g. "1"), not int
            # (verified via the probe entrypoint) → cast before indexing options.
            gold = opts[int(ans_idx)]
            pred = pred_raw.strip()
            exact = pred == gold                 # big_vision scorer (to_lower=False)
            norm = _norm(pred) == _norm(gold)    # diagnostic only
            n_correct_exact += int(exact)
            n_correct_norm += int(norm)
            if len(records) < 40:  # keep a small qualitative sample, not all N
                records.append({
                    "question": q, "options": opts, "gold": gold,
                    "pred": pred, "exact": exact, "norm": norm,
                })

        done = min(start + batch_size, n_items)
        if (start // batch_size) % 5 == 0 or done == n_items:
            print(f"[P1] {done}/{n_items}  running acc_exact="
                  f"{n_correct_exact / done:.4f}")

    acc_exact = n_correct_exact / n_items
    acc_norm = n_correct_norm / n_items
    elapsed = time.time() - t0
    print(f"[P1] DONE acc_exact={acc_exact:.4f} acc_norm={acc_norm:.4f} "
          f"N={n_items} in {elapsed:.1f}s")

    return {
        "task": "AI2D",
        "model": AI2D_REPO,
        "dataset": AI2D_DATASET,
        "split": AI2D_SPLIT,
        "metric": "accuracy (exact-match string == gold option text, big_vision VQA scorer, to_lower=False)",
        "published": 74.7,
        "published_source": "https://huggingface.co/google/paligemma2-3b-mix-224 (model card 224-3B AI2D); cross-checked ai.google.dev model-card-2 + arXiv:2412.03555 Table 13",
        "n_items": n_items,
        "n_full_split": full_n,
        "seed": seed,
        "batch_size": batch_size,
        "acc_exact": acc_exact,           # PRIMARY verdict metric (×100 → compare to 74.7)
        "acc_exact_pct": round(acc_exact * 100, 2),
        "acc_norm": acc_norm,             # diagnostic
        "acc_norm_pct": round(acc_norm * 100, 2),
        "n_correct_exact": n_correct_exact,
        "n_correct_norm": n_correct_norm,
        "elapsed_sec": round(elapsed, 1),
        "prompt_recipe": {
            "PREFIX": PREFIX, "PROMPT": PROMPT, "PROMPT_SEP": PROMPT_SEP,
            "example_prompt": build_prompt(
                records[0]["question"], records[0]["options"]) if records else None,
            "source": "google-research/big_vision .../proj/paligemma/transfers/ai2d.py",
        },
        "sample_records": records,
    }
