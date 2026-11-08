"""read-from-passage — the no-knowledge reading control (prereg 2026-06-20).

Does the deployed-policy "deletion" (PCR `H_genuine-deletion`) mean an erased FACT, or a dead READER?
Every readout that floors on π0/π0.5 presupposes the backbone can read text. This module supplies a
task whose answer is PRESENT in the prompt and needs NO stored fact (fictional entities ⇒ no parametric
retrieval path), scored through the IDENTICAL cloze / MCQ-letter readout as PCR. Above FLOOR ⇒ the
reader works (PCR's deletion is knowledge-specific); at FLOOR ⇒ the reader is dead (PCR measured a
dead reader). See `studies/lineage/paligemma1-pi0/research/2026-06-20_read_from_passage_control_prereg.md`.

Two pieces:
  * build_synth()  — a SEEDED, content-hashed fictional-entity 4-way in-context-recall generator. Each
    item names two entities, binds each to a distinct attribute value, and asks for ONE entity's value.
    The 4 choices include BOTH entities' values, so answering requires RESOLVING the question's subject
    to the right sentence (genuine reading, not keyword spotting). Fictional names ⇒ zero world knowledge.
  * passage_cloze / passage_choice_logits — the readout pair. They REUSE text_eval's verified pure
    scoring core (_score_cloze_window / _common_prefix_len / _common_suffix_len) and the exact left-pad +
    content-relative position_ids + logits_to_keep forward of `mmlu_cloze` / `mmlu_accuracy`, but consume
    the read-from-passage item list. They emit the IDENTICAL return-dict shape so the entire PCR
    orchestration (paired bootstrap, split, own-vs-base-embed, label-shuffle) is reused unchanged.

Discipline (CLAUDE.md): no try/except hiding; any mismatch RAISES with a trace. Heavy imports live
inside functions so this module py_compiles on a torch-free control plane.
"""
from __future__ import annotations

import hashlib
import json
import random

# ---------------------------------------------------------------------------
# Synthetic fictional-entity pools. ALL ASCII common words → tokenize into shared
# base-vocab rows (<256000) under the base gemma-2b tokenizer (prereg §13 gate 2).
# Attribute VALUES are arbitrary preferences of FICTIONAL people ⇒ no parametric path.
# ---------------------------------------------------------------------------
NAMES = [
    "Toran", "Wessel", "Marla", "Brennick", "Dalia", "Foran", "Galwin", "Hesper",
    "Joris", "Kessa", "Lorne", "Mabry", "Nessa", "Orin", "Perrin", "Quilla",
    "Roban", "Selka", "Tamsin", "Ulric", "Varna", "Welby", "Yorin", "Zella",
    "Abner", "Cterra", "Doreen", "Elgar", "Faraday", "Greta",
]
PLACES = [
    "Quemby", "Dremfork", "Halloway", "Pellmoor", "Corvale", "Brackin", "Stowe",
    "Marrow", "Fendle", "Garrow", "Wend", "Throstle", "Ashby", "Lund", "Crale",
]
PROFESSIONS = [
    "baker", "sailor", "weaver", "smith", "scribe", "cooper", "mason", "tanner",
    "farrier", "glazier", "potter", "carter",
]
ATTRS = {
    "color": ["amber", "teal", "crimson", "indigo", "olive", "scarlet", "azure",
              "maroon", "violet", "ochre", "russet", "slate"],
    "animal": ["heron", "marten", "falcon", "otter", "badger", "lynx", "stoat",
               "raven", "ibis", "newt", "vole", "shrew"],
    "drink": ["cider", "mead", "broth", "nectar", "cordial", "tonic", "squash", "posset"],
    "tool": ["mallet", "chisel", "needle", "ledger", "trowel", "auger", "plane", "rasp"],
    "fruit": ["plum", "fig", "quince", "pear", "damson", "medlar", "sloe", "gage"],
}
ATTR_NOUN = {"color": "favorite color", "animal": "favorite animal",
             "drink": "favorite drink", "tool": "favorite tool", "fruit": "favorite fruit"}

# Last 6 names / 4 places are RESERVED for the few-shot exemplars only → disjoint from scored items
# (prereg §4 contamination control). Asserted in build_synth.
_N_NAMES_EX = 6
_N_PLACES_EX = 4

RFP_HEADER = "Read each passage and answer the question using only the passage.\n\n"


def _make_item(rng, names, places):
    """One 2-entity attribute-binding item. The 4 choices carry BOTH entities' values + 2 fillers, so
    the model must bind the question's subject to the right sentence (reading, not keyword spotting)."""
    attr = rng.choice(list(ATTRS))
    pool = ATTRS[attr]
    subj, other = rng.sample(names, 2)
    subj_prof, other_prof = rng.sample(PROFESSIONS, 2)
    subj_place, other_place = rng.sample(places, 2)
    v_subj, v_other, f1, f2 = rng.sample(pool, 4)           # 4 distinct values from the same category
    noun = ATTR_NOUN[attr]
    passage = (f"{subj} is a {subj_prof} who lives in {subj_place}. {subj}'s {noun} is {v_subj}. "
               f"{other} is a {other_prof} who lives in {other_place}. {other}'s {noun} is {v_other}.")
    question = f"What is {subj}'s {noun}?"
    choices = [v_subj, v_other, f1, f2]
    rng.shuffle(choices)
    return {"passage": passage, "question": question, "choices": choices,
            "answer": choices.index(v_subj), "attr": attr, "subject": subj, "distractor_entity": other}


def build_synth(seed: int, n: int, n_exemplars: int = 5):
    """Deterministic fictional-entity in-context-recall set. Returns (items, exemplars, meta).
    `meta` banks the seed, pools, per-item content sha1, and the full-set content hash (prereg §4)."""
    names_items = NAMES[:-_N_NAMES_EX]
    names_ex = NAMES[-_N_NAMES_EX:]
    places_items = PLACES[:-_N_PLACES_EX]
    places_ex = PLACES[-_N_PLACES_EX:]
    assert len(names_items) >= 2 and len(places_items) >= 2, "item pools too small"
    assert not (set(names_items) & set(names_ex)), "name pools not disjoint"
    assert not (set(places_items) & set(places_ex)), "place pools not disjoint"

    rng = random.Random(seed)
    items = [_make_item(rng, names_items, places_items) for _ in range(n)]
    ex_rng = random.Random(seed + 1)
    exemplars = [_make_item(ex_rng, names_ex, places_ex) for _ in range(n_exemplars)]

    # contamination guard: no scored-item entity appears in any exemplar (prereg §4, fail-loud)
    ex_names = {e["subject"] for e in exemplars} | {e["distractor_entity"] for e in exemplars}
    it_names = {it["subject"] for it in items} | {it["distractor_entity"] for it in items}
    assert not (ex_names & it_names), f"exemplar∩item entity leak: {sorted(ex_names & it_names)}"

    for it in items:
        assert len(it["choices"]) == 4 and 0 <= it["answer"] < 4
        assert it["choices"][it["answer"]] in it["passage"], "gold value not stated in passage"

    meta = {"seed": seed, "n": n, "n_exemplars": n_exemplars,
            "names_items": names_items, "names_ex": names_ex,
            "places_items": places_items, "places_ex": places_ex,
            "attrs": {k: v for k, v in ATTRS.items()},
            "content_sha1": content_hash(items),
            "exemplar_sha1": content_hash(exemplars)}
    return items, exemplars, meta


def item_hash(it) -> str:
    """Position-pairing key + reorder tripwire (prereg §2): sha1 of the FULL item content, 12 hex.
    Distinct items → distinct hashes (well-distributed for the is_discovery split)."""
    blob = it["passage"] + "\n" + it["question"] + "\n" + "||".join(it["choices"])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def content_hash(items) -> str:
    canon = [{"passage": it["passage"], "question": it["question"],
              "choices": it["choices"], "answer": it["answer"]} for it in items]
    return hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompt blocks — mirror text_eval._q_block_cloze / _q_block, with a passage prepended.
# drop_passage=True removes the passage (the §5 no-passage leakage control).
# ---------------------------------------------------------------------------
def _block_cloze(passage, question, answer_text=None, drop_passage=False):
    body = "" if drop_passage else passage.strip() + "\n"
    s = body + question.strip() + "\nAnswer:"
    if answer_text is not None:
        s += f" {answer_text.strip()}\n\n"
    return s


def _block_mcq(passage, question, choices, answer_idx=None, drop_passage=False):
    from infra.pg.text_eval import CHOICES
    assert len(choices) == 4, f"item must have 4 choices, got {len(choices)}"
    body = "" if drop_passage else passage.strip() + "\n"
    s = body + question.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{CHOICES[i]}. {c.strip()}\n"
    s += "Answer:"
    if answer_idx is not None:
        s += f" {CHOICES[answer_idx]}\n\n"
    return s


def _build_prefix_cloze(exemplars, n_shots, header, drop_passage):
    return header + "".join(
        _block_cloze(e["passage"], e["question"], e["choices"][e["answer"]], drop_passage)
        for e in exemplars[:n_shots])


def _build_prefix_mcq(exemplars, n_shots, header, drop_passage):
    return header + "".join(
        _block_mcq(e["passage"], e["question"], e["choices"], e["answer"], drop_passage)
        for e in exemplars[:n_shots])


# ---------------------------------------------------------------------------
# Readouts. The CONTENT (cloze) and LETTER (MCQ) axes — the two PCR axes, applied to the
# read-from-passage items. Verified pure core + forward reused VERBATIM from text_eval.mmlu_cloze /
# mmlu_accuracy; the only change vs PCR is the item source. Return-dict shape is identical to its
# MMLU counterpart so the PCR orchestration consumes it unchanged.
# ---------------------------------------------------------------------------
def passage_cloze(model, tokenizer, device, items, exemplars, *, n_shots=5,
                  drop_passage=False, header=RFP_HEADER):
    """Score the 4 choice TEXTS as continuations of `header + exemplars + passage + question + 'Answer:'`
    (choices hidden). Banks per-candidate (Σlogprob, n_tok); λ=1 per_item. Mirrors text_eval.mmlu_cloze
    (text-only forward), reusing its scoring core and left-pad+position_ids+logits_to_keep recipe."""
    import torch
    from infra.pg.text_eval import (_score_cloze_window, _common_prefix_len, _common_suffix_len)

    tok = tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"

    prefix = _build_prefix_cloze(exemplars, n_shots, header, drop_passage)
    cloze_logprobs, cloze_ntoks, correct_idx, per_item, per_item_hash = [], [], [], [], []
    for it in items:
        prompt = prefix + _block_cloze(it["passage"], it["question"], None, drop_passage)
        choices = [c.strip() for c in it["choices"]]
        assert len(choices) == 4, f"item must have 4 choices, got {len(choices)}"
        pe = tok([prompt], return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        prompt_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
        enc = tok([prompt + " " + c for c in choices], return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        ids, am = enc["input_ids"], enc["attention_mask"]
        real = am.sum(dim=1).tolist()
        cont_ids, ntok = [], []
        lcp = lcs = None
        for k in range(4):
            full_real = ids[k][ids.shape[1] - int(real[k]):].tolist()
            p = _common_prefix_len(prompt_real, full_real)
            s = _common_suffix_len(prompt_real, full_real)
            assert p + s == len(prompt_real), \
                f"cloze prompt not [shared|sep] (choice {k}): lcp={p} lcs={s} |p|={len(prompt_real)}"
            cont = full_real[p:len(full_real) - s]
            assert len(cont) >= 1, f"empty cloze continuation (choice {k})"
            if lcp is None:
                lcp, lcs = p, s
            else:
                assert (p, s) == (lcp, lcs), "inconsistent cloze framing across candidates"
            cont_ids.append(cont)
            ntok.append(len(cont))
        K = max(ntok) + lcs + 1
        with torch.inference_mode():
            pos = am.long().cumsum(-1) - 1
            pos = pos.masked_fill(am == 0, 1)
            out = model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=K)
            lw = torch.log_softmax(out.logits.float(), dim=-1)
        sums = _score_cloze_window(lw, cont_ids, ntok, trailing=lcs)
        assert torch.isfinite(sums).all(), "non-finite cloze logprob"
        cloze_logprobs.append([float(x) for x in sums.tolist()])
        cloze_ntoks.append([int(m) for m in ntok])
        ci = int(it["answer"])
        correct_idx.append(ci)
        mean_lp = sums / torch.tensor(ntok, dtype=torch.float64)
        per_item.append(int(int(mean_lp.argmax()) == ci))
        per_item_hash.append(item_hash(it))

    tok.padding_side = orig_side
    n = len(per_item)
    assert n > 0, "passage_cloze produced no items"
    return {"task": "RFP-cloze", "metric": "read-from-passage GEN-cloze (length-normed λ=1 argmax)",
            "forward": "text-only", "drop_passage": drop_passage,
            "acc": sum(per_item) / n, "acc_pct": round(100 * sum(per_item) / n, 2), "n": n,
            "n_shots": n_shots, "cloze_logprobs": cloze_logprobs, "cloze_ntoks": cloze_ntoks,
            "correct_idx": correct_idx, "per_item": per_item, "per_item_hash": per_item_hash}


def passage_choice_logits(model, tokenizer, device, items, exemplars, *, n_shots=5,
                          drop_passage=False, header=RFP_HEADER, batch_size=4):
    """Letter axis: render `... + passage + question + A./B./C./D. + 'Answer:'`, read next-token logits
    at the 4 choice-letter ids, argmax. Mirrors text_eval.mmlu_accuracy / mmlu_choice_logits."""
    import torch
    from infra.pg.text_eval import CHOICES

    tok = tokenizer
    letter_ids = []
    for L in CHOICES:
        lid = tok.encode(" " + L, add_special_tokens=False)
        assert len(lid) == 1, f"' {L}' → {len(lid)} tokens, expected 1 (Gemma '▁{L}')"
        letter_ids.append(lid[0])
    assert len(set(letter_ids)) == 4, f"choice-letter token ids collide: {letter_ids}"

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"

    prefix = _build_prefix_mcq(exemplars, n_shots, header, drop_passage)
    choice_logits, correct_idx, per_item, per_item_hash = [], [], [], []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [prefix + _block_mcq(it["passage"], it["question"], it["choices"], None, drop_passage)
                   for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        pos = enc["attention_mask"].long().cumsum(-1) - 1
        pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
        with torch.inference_mode():
            last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
        rows = last[:, letter_ids].float()
        preds = rows.argmax(dim=-1).tolist()
        for it, lg, p in zip(batch, rows.tolist(), preds):
            ci = int(it["answer"])
            choice_logits.append([float(x) for x in lg])
            correct_idx.append(ci)
            per_item.append(int(p == ci))
            per_item_hash.append(item_hash(it))

    tok.padding_side = orig_side
    n = len(per_item)
    assert n > 0, "passage_choice_logits produced no items"
    return {"task": "RFP-mcq", "metric": "read-from-passage 5-shot single-token-answer accuracy",
            "forward": "text-only", "drop_passage": drop_passage,
            "acc": sum(per_item) / n, "acc_pct": round(100 * sum(per_item) / n, 2), "n": n,
            "n_shots": n_shots, "choice_logits": choice_logits, "correct_idx": correct_idx,
            "per_item": per_item, "per_item_hash": per_item_hash, "letter_token_ids": letter_ids}
