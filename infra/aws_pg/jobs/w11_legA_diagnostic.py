"""W11 Leg-A DIAGNOSTIC (EXPLORATORY, post-hoc — NOT the registered unblind).

The registered Leg-A unblind ran the LMs TEXT-ONLY (no image tokens) and found H_forget
(PG2-LM MMLU = random-decoder floor). But PaliGemma2 is ALWAYS trained with 256 image tokens;
a text-only forward is OUT-OF-DISTRIBUTION for it. This disambiguates "genuinely forgot language"
from "requires the image-token forward context" by re-running MMLU with a BLANK (gray) image
PRESENT — PG2's in-distribution forward (the §2A "image tokens blanked" reading).

  base   = gemma-2-2b, TEXT-ONLY (its native mode; the published anchor)
  AFTER  = PG2-LM, with a gray 224 image  (in-distribution)
  BEFORE = decoder-reset PG2, with a gray 224 image
  FLOOR  = random-decoder PG2, with a gray 224 image
  S_text_blankimg = (AFTER − FLOOR) / (BEFORE − FLOOR)

Read:
  * AFTER_bi MMLU ≈ base (~50%)  → text-only was an OOD ARTIFACT; PG2 RETAINS language with image
    context → reframe Leg A as "image-context-dependent, not forgotten" (H_preserve-in-context).
  * AFTER_bi MMLU ≈ floor (~25%) → genuine forgetting survives the in-distribution forward → H_forget.

Fail-loud; no try/except. Single read.

    python -m infra.aws_pg.jobs.w11_legA_diagnostic --out <dir> [--n-per-subject 30]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def mmlu_blankimg(model, processor, device, n_per_subject, batch_size=4, gray=128):
    """5-shot MMLU with a gray 224 image present (PG2 in-distribution forward). Reads the
    next-token logit at the four choice letters after 'Answer:'. Left-pad so -1 is the real last."""
    import numpy as np
    import torch
    from collections import defaultdict
    from datasets import load_dataset
    from PIL import Image

    from infra.pg.text_eval import MMLU_REPO, CHOICES, _subject_header, _q_block

    ds = load_dataset(MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]
    shots_by = defaultdict(list)
    for r in dev:
        shots_by[r["subject"]].append(r)
    test_by = defaultdict(list)
    for r in test:
        test_by[r["subject"]].append(r)

    tok = processor.tokenizer
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in CHOICES]
    gray_img = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    orig = tok.padding_side
    tok.padding_side = "left"

    n_correct = 0
    n_total = 0
    per_item = []
    for subj in sorted(test_by):
        shots = shots_by[subj][:5]
        prefix = _subject_header(subj) + "".join(
            _q_block(s["question"], s["choices"], s["answer"]) for s in shots)
        rows = test_by[subj][:n_per_subject] if n_per_subject else test_by[subj]
        for st in range(0, len(rows), batch_size):
            b = rows[st:st + batch_size]
            prompts = [prefix + _q_block(r["question"], r["choices"]) for r in b]
            inp = processor(text=prompts, images=[gray_img] * len(b),
                            return_tensors="pt", padding="longest").to(device)
            # pass ONLY the logits inputs (no token_type_ids/labels → PaliGemma's training-loss
            # path stays off; that path + logits_to_keep mismatches shapes). Image present via
            # pixel_values = the in-distribution factor being tested.
            with torch.inference_mode():
                # logits_to_keep=1 (last token only → no full 257k-vocab OOM) AND no
                # token_type_ids/labels (→ PaliGemma loss-path stays off). Both needed together.
                last = model(input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"],
                             pixel_values=inp["pixel_values"].to(torch.bfloat16),
                             logits_to_keep=1).logits[:, -1, :]
            preds = last[:, letter_ids].argmax(dim=-1).tolist()
            for r, p in zip(b, preds):
                ok = int(p == int(r["answer"]))
                per_item.append(ok)
                n_correct += ok
                n_total += 1
    tok.padding_side = orig
    return {"acc": n_correct / n_total, "acc_pct": round(100 * n_correct / n_total, 2),
            "n": n_total, "per_item": per_item}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-subject", type=int, default=30, help="cap per subject (exploratory; speed)")
    ap.add_argument("--seed-floor", type=int, default=1337)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PaliGemmaProcessor

    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "Leg-A diagnostic needs GPU."
    device = "cuda"
    nps = args.n_per_subject

    # base gemma-2-2b TEXT-ONLY (control anchor; sdpa parity)
    L0 = ENTITIES["L0"]
    gemma_tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    base = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="sdpa").eval()
    base_mmlu = TE.mmlu_accuracy(base, gemma_tok, device, n_per_subject=nps)
    print(f"[W11-A-DIAG] base text-only MMLU={base_mmlu['acc_pct']:.2f}")
    del base
    torch.cuda.empty_cache()

    # AFTER + BEFORE (PaliGemma), then FLOOR — all with a gray image (in-distribution)
    pg_after, pg_before, proc, paths, _, gate = H.build_before_after(device)
    after_bi = mmlu_blankimg(pg_after, proc, device, nps)
    print(f"[W11-A-DIAG] AFTER blank-image MMLU={after_bi['acc_pct']:.2f}")
    del pg_after
    torch.cuda.empty_cache()
    before_bi = mmlu_blankimg(pg_before, proc, device, nps)
    print(f"[W11-A-DIAG] BEFORE blank-image MMLU={before_bi['acc_pct']:.2f}")
    pg_random = H.build_random_after(pg_before, paths, device, seed=args.seed_floor)
    floor_bi = mmlu_blankimg(pg_random, proc, device, nps)
    print(f"[W11-A-DIAG] FLOOR blank-image MMLU={floor_bi['acc_pct']:.2f}")
    del pg_random, pg_before
    torch.cuda.empty_cache()

    denom = before_bi["acc"] - floor_bi["acc"]
    s_text_bi = (after_bi["acc"] - floor_bi["acc"]) / denom if denom > 0 else float("nan")
    # interpretation: AFTER near base ⇒ OOD artifact; AFTER near floor ⇒ genuine forget
    if after_bi["acc_pct"] >= 0.6 * base_mmlu["acc_pct"]:
        interp = "OOD_artifact: PG2 RETAINS language with image context (text-only was OOD) → H_preserve-in-context"
    elif after_bi["acc_pct"] <= floor_bi["acc_pct"] + 5.0:
        interp = "GENUINE_forget: forgetting survives the in-distribution forward → H_forget"
    else:
        interp = "PARTIAL: AFTER between floor and base → partial in-context degradation"

    result = {
        "torch": torch.__version__, "exploratory": True,
        "note": "post-hoc disambiguation of the registered text-only H_forget; NOT a registered result",
        "n_per_subject": nps,
        "base_text_only_mmlu_pct": base_mmlu["acc_pct"],
        "blank_image_mmlu": {"AFTER": after_bi["acc_pct"], "BEFORE": before_bi["acc_pct"],
                             "FLOOR": floor_bi["acc_pct"]},
        "S_text_blankimg": s_text_bi,
        "registered_text_only": {"S_text": 0.028, "AFTER_mmlu_pct": 25.42, "verdict": "H_forget"},
        "interpretation": interp,
        "build_gate": gate,
    }
    (out / "legA_diagnostic.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W11-A-DIAG] S_text_blankimg={s_text_bi:.4f}  →  {interp}")


if __name__ == "__main__":
    main()
