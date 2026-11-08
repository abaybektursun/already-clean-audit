# W14-R eval layer — MMLU loading (English + MMMLU translations) + the scoring primitive that
# the W11 forward/extraction is factored into. Heavy imports (torch/datasets) live INSIDE the
# functions so this module py_compiles on the control plane.
"""Binds the two freeze-time externals of the recovery gate against REAL sources:

  * `load_mmlu` — builds the §2 item schema {question, options[4], answer, subject, i18n} from
    `cais/mmlu` (English base, the W11 hash source) + `openai/MMMLU` (PROFESSIONAL human translations
    for fr/es/de/ja/zh). The English item is the alignment anchor; each translation is paired by
    (subject, in-subject position) with a fail-loud answer-key cross-check (the translated correct
    letter must match the English answer index) so a reordered translation can NEVER align silently.

  * `score_prompts` — the W11 `text_eval.mmlu_accuracy` forward + extraction, factored to score an
    ARBITRARY list of prompts against a per-transform answer-surface set (the §2 "W11 scorer wrapped
    by apply_transform" design). Same recipe: left-pad, content-relative position_ids, single forward
    with logits_to_keep=1, argmax over the 4 candidate-letter token ids. IDENTITY prompts built by
    `recovery_transforms.apply_transform` reproduce `mmlu_accuracy` byte-for-byte (so the per-item
    0/1 it returns equals the §0 anchor — asserted at preflight).

No try/except hiding — shape/format/alignment errors RAISE with a trace (fail loud).
"""
from __future__ import annotations

# openai/MMMLU per-language config names → our i18n lang keys (ratified fr/es/de/ja/zh; ru→ja).
MMMLU_CONFIGS = {"fr": "FR_FR", "es": "ES_LA", "de": "DE_DE", "ja": "JA_JP", "zh": "ZH_CN"}


def load_mmlu(held_out_subjects, n_per_subject=None, langs=("fr", "es", "de", "ja", "zh"), n_shots=5,
              max_drop_frac=0.05):
    """Return (dev_items, held_out_items, exemplars_by_subject). Each item:
        {question, options[4], answer∈0..3, subject, gidx, hash, i18n:{lang:{question,options}}}
    `gidx` is a unique global index (the cross-transform PAIRING key — NOT the hash, which collides:
    1637 MMLU items share a 64-char question prefix and even q+opts has ~105 true duplicates). `hash`
    is sha1(question+options) for the manifest only.

    Alignment (journal C8): MMMLU is paired to cais by (subject, Unnamed:0-SORTED) position — verified
    to give 0 answer-mismatches for fr/es, and the answer cross-check DROPS the ~1.3% of items (mostly
    zh: 176, de: 5, ja: 1) whose translation is misordered/relabeled, so every kept item has a
    verified translation in ALL langs. A drop rate above `max_drop_frac` HALTS (systemic, not warts).

    exemplars_by_subject[subj] = the first `n_shots` ENGLISH `cais/mmlu` DEV items (W11's few-shot
    source, no i18n — translations are test-only). dev = 47 dev subjects, held_out = the §4 reserved 10.
    n_per_subject caps test items/subject for a fast preflight (None = full)."""
    import hashlib
    from collections import defaultdict
    from datasets import load_dataset

    ds_en = load_dataset("cais/mmlu", "all")
    en, dev = ds_en["test"], ds_en["dev"]
    en_by_subj = defaultdict(list)
    for r in en:
        en_by_subj[r["subject"]].append(r)               # dataset order within subject = MMLU canonical
    exemplars_by_subject = defaultdict(list)
    for r in dev:                                         # W11 few-shot pool (English, dev split)
        exemplars_by_subject[r["subject"]].append(
            {"question": r["question"], "options": list(r["choices"]),
             "answer": int(r["answer"]), "subject": r["subject"]})
    exemplars_by_subject = {s: rows[:n_shots] for s, rows in exemplars_by_subject.items()}

    # MMMLU translations, indexed by (subject, Unnamed:0-SORTED position) — the verified alignment.
    tr_by_lang = {}
    for lang in langs:
        cfg = MMMLU_CONFIGS[lang]
        ds = load_dataset("openai/MMMLU", cfg)
        split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        by_subj = defaultdict(list)
        for r in split:
            by_subj[r["Subject"]].append(r)
        for s in by_subj:                                    # dataset iteration order != Unnamed:0 for zh
            by_subj[s].sort(key=lambda r: int(r["Unnamed: 0"]))
        tr_by_lang[lang] = by_subj

    subjects = sorted(en_by_subj)
    held = set(held_out_subjects)
    assert held <= set(subjects), f"held-out subjects not in MMLU: {held - set(subjects)}"

    dev_items, held_items = [], []
    gidx = 0
    n_seen = n_dropped = 0
    for subj in subjects:
        en_rows = en_by_subj[subj]
        if n_per_subject:
            en_rows = en_rows[:n_per_subject]
        for i, r in enumerate(en_rows):
            n_seen += 1
            q, opts, ans = r["question"], list(r["choices"]), int(r["answer"])
            assert len(opts) == 4, f"{subj}[{i}]: {len(opts)} options"
            i18n = {}
            ok = True
            for lang in langs:
                trows = tr_by_lang[lang].get(subj, [])
                if i >= len(trows):                          # translation missing this position
                    ok = False
                    break
                tr = trows[i]
                # the translated correct letter must equal the English answer index; a mismatch means
                # this item's translation is misordered/relabeled in this lang → DROP it (drop-and-log,
                # not a silent mispairing and not a hard halt for sporadic data warts).
                if ord(tr["Answer"].strip().upper()) - ord("A") != ans:
                    ok = False
                    break
                i18n[lang] = {"question": tr["Question"],
                              "options": [tr["A"], tr["B"], tr["C"], tr["D"]]}
            if not ok:
                n_dropped += 1
                continue
            item = {"question": q, "options": opts, "answer": ans, "subject": subj, "gidx": gidx,
                    "hash": hashlib.sha1((q + "||" + "|".join(opts)).encode()).hexdigest()[:16],
                    "i18n": i18n}
            gidx += 1
            (held_items if subj in held else dev_items).append(item)

    frac = n_dropped / max(n_seen, 1)
    assert frac <= max_drop_frac, (f"translation-misalignment drop {frac:.3f} > {max_drop_frac} — "
                                   "systemic alignment failure, not sporadic data warts; investigate")
    print(f"[load_mmlu] kept {gidx} items (dev {len(dev_items)} / held {len(held_items)}); dropped "
          f"{n_dropped} ({frac:.2%}) lacking a verified translation in all {len(langs)} langs")
    return dev_items, held_items, exemplars_by_subject


def _candidate_token_ids(tokenizer, answer_surfaces):
    """Token id for each answer surface (e.g. ' A',' 1',' a'); assert each is a single token and
    the four are distinct (the W11 single-token-answer contract, per transform)."""
    ids = []
    for s in answer_surfaces:
        toks = tokenizer.encode(s, add_special_tokens=False)
        assert len(toks) == 1, f"answer surface {s!r} → {len(toks)} tokens, expected 1"
        ids.append(toks[0])
    assert len(set(ids)) == len(ids), f"answer-surface token ids collide: {ids}"
    return ids


def score_prompts(model, tokenizer, device, prompts, answer_surfaces, gold_idx, batch_size=4):
    """W11 text-only forward + extraction over arbitrary prompts. `answer_surfaces` = the 4 candidate
    answer tokens for this transform (recovery_transforms.answer_candidates); `gold_idx` = per-prompt
    correct index 0..3. Returns a per-prompt 0/1 list aligned to `prompts`. Same recipe as
    text_eval.mmlu_accuracy: left-pad, content-relative position_ids, logits_to_keep=1, argmax."""
    import torch

    assert len(prompts) == len(gold_idx), f"prompts {len(prompts)} != gold {len(gold_idx)}"
    letter_ids = _candidate_token_ids(tokenizer, answer_surfaces)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        pos = enc["attention_mask"].long().cumsum(-1) - 1
        pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
        with torch.inference_mode():
            last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
        preds = last[:, letter_ids].argmax(dim=-1).tolist()
        for j, p in enumerate(preds):
            out.append(int(p == int(gold_idx[start + j])))
    tokenizer.padding_side = orig_side
    return out


def score_prompts_blankimg(model, processor, device, prompts, answer_surfaces, gold_idx,
                           batch_size=4, gray=128):
    """Blank-image SECONDARY-arm scorer (W11 `mmlu_blankimg`-faithful): the SAME prompt text run
    through the processor with a gray 224² image PRESENT (PG2 in-distribution forward). No
    position_ids / token_type_ids / labels (PaliGemma's loss path stays off — it shape-clashes with
    logits_to_keep); pixel_values in bf16; logits_to_keep=1; argmax over the per-transform answer
    surfaces. Returns per-prompt 0/1 aligned to `prompts`."""
    import numpy as np
    import torch
    from PIL import Image

    assert len(prompts) == len(gold_idx), f"prompts {len(prompts)} != gold {len(gold_idx)}"
    tok = processor.tokenizer
    letter_ids = _candidate_token_ids(tok, answer_surfaces)
    gray_img = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    orig_side = tok.padding_side
    tok.padding_side = "left"

    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inp = processor(text=chunk, images=[gray_img] * len(chunk),
                        return_tensors="pt", padding="longest").to(device)
        with torch.inference_mode():
            last = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"].to(torch.bfloat16),
                         logits_to_keep=1).logits[:, -1, :]
        preds = last[:, letter_ids].argmax(dim=-1).tolist()
        for j, p in enumerate(preds):
            out.append(int(p == int(gold_idx[start + j])))
    tok.padding_side = orig_side
    return out
