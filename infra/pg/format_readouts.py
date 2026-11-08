"""MENDELEEV readouts (MATRIX_CONTRACT §4; spec §1.2). Every readout depends ONLY on the four `Forward`
methods of contract §3 — `next_token_logits`, `continuation_logprobs`, `generate`, `letter_ids` — plus
`batch_size`; nothing here imports torch or touches a tokenizer. Every readout returns ONE record shape:

    {"col", "acc", "n", "floor", "per_item", "per_item_hash", "per_item_scores", "pred", "extra"}

with acc == mean(per_item) (the paired-item bootstrap in matrix_core consumes per_item directly).

Prompts. S is byte-identical to `text_eval.mmlu_accuracy` / `mmlu_blankimg`: `_subject_header` + k
`_q_block` shots + `_q_block(item)` ending at "Answer:", read at the " A".." D" letter ids (the §3 parity
gate). K reproduces `text_eval.mmlu_cloze` through `continuation_logprobs`: `_subject_header` + 5 cloze
exemplars (`_q_block_cloze` with the gold TEXT) + `_q_block_cloze(item)`, the four option texts scored as
" " + text.strip(), per-token-mean log-likelihood, 4-way argmax. RFP reproduces
`read_from_passage.passage_cloze` the same way (its header, prefix builder and item hash are reused).

Formats (all share S's shots and header; only the scored block's shape changes):
  F1  option ORDER deranged per item (random.Random(1337), no fixed point), letters A–D unchanged, gold tracked.
  F2  labels "1."–"4.", read at " 1".." 4".
  F3  verification: 4 prompts per item ("Proposed answer: <opt>." → " Yes"/" No" logits), one true, three
      false; balanced accuracy over (tp, fn, tn, fp), floor 0.5. per_item is the per-item balanced score
      0.5·(1[gold→Yes] + n_wrong→No / 3), whose mean IS the balanced accuracy (asserted in the dry-run;
      `extra["per_item_kind"]` says so). The item's lettered option list IS shown (the F family holds item
      + options fixed and varies only the answer format). Shots alternate true/false proposals.
  F4  "(A)" labels; the scored block ends "Answer: (" and is read at the no-space letter ids.
  F5  options shown with letters, the four option TEXTS scored after "Answer:" (shots show the gold text).
  K-null: option texts of item i replaced by those of item π(i) (item-level derangement), gold index kept.
  A   arithmetic: greedy generate (8 tokens), first integer == gold (em, the column); cand = 4-option
      candidate-likelihood (extra). TQ: TriviaQA 4-way option-text scoring. U / U_shortcut: S on the
      two_hop_items bundle with its own exemplars.
  H(pred): matrix_core.letter_entropy; H_record gives it beside the answer key's entropy.
  tokenizer_gate_stats(fwd, s_rec): raw letter token counts (`Forward.letter_token_counts`, no assert, so
      VOID-tokenizer is reachable), argmax_in_letters_frac, p_any_letter (mean full-vocab probability mass
      on the four letter tokens), and BOTH surfaces' accuracy plus their per-item agreement (overall and
      split by whether the scored surface was right) — from the S record so the forward is not run twice;
      tokenizer_gate(before, after) → matrix_core.tokenizer_gate v5 (accuracy invariance is the gate;
      per-item agreement is a diagnostic).
Forward methods used: next_token_logits, continuation_logprobs, generate, letter_ids, letter_token_counts, batch_size.
"""
from __future__ import annotations

import random
import re

import numpy as np

from infra.pg import matrix_core as MC
from infra.pg import matrix_items as MI
from infra.pg import read_from_passage as RFP
from infra.pg.text_eval import CHOICES, _q_block, _q_block_cloze, _subject_header

SEED = 1337
FLOOR_LETTER, FLOOR_F3, FLOOR_RFP, FLOOR_A = 0.25, 0.5, 0.27, 0.0
DIGITS = ("1", "2", "3", "4")
YESNO = ("Yes", "No")
F3_ASK = "Is the proposed answer correct? Answer (Yes/No):"
A_HEADER = "Solve each problem. Give only the final number.\n\n"
TQ_HEADER = "Answer each trivia question.\n\n"


# ------------------------------------------------------------------------------------------ plumbing
def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _derangement(rng: random.Random, n: int) -> list:
    assert n >= 2, "a derangement needs ≥ 2 elements"
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def _letter_read(fwd, prompts: list, ids: list, alt_ids: list = None):
    """Per prompt: logits at `ids` (n, k), argmax-in-ids flag (n,), probability mass on ids (n,).
    Reduced per batch so the (B, V) logits are never kept.
    Also banked from the SAME forward at zero extra cost (smoke 4, 2026-09-03): the argmax token id and
    its probability per item (`top`), and, when `alt_ids` is given, the same three statistics for a second
    answer surface. Why: in the blank-image shell the PaliGemma processor appends a trailing newline, so
    the model answers with the NO-SPACE letter and the space-prefixed surface carries ~1e-7 of the mass —
    the healthy model reads as 'off-format' while its restricted 4-way ranking still works. Both surfaces
    are recorded so the format artifact can be separated from the fine-tune's effect."""
    ids = [int(i) for i in ids]
    assert len(ids) == len(set(ids)), f"letter ids collide: {ids}"
    alt_ids = [int(i) for i in alt_ids] if alt_ids else None
    bs = int(fwd.batch_size)
    assert bs >= 1
    scores, in_ids, p_any = [], [], []
    top_ids, top_p = [], []
    alt_scores, alt_in, alt_p = [], [], []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s:s + bs]
        L = _np(fwd.next_token_logits(chunk))
        assert L.ndim == 2 and L.shape[0] == len(chunk), L.shape
        assert np.isfinite(L).all(), "non-finite logits"
        sub = L[:, ids]
        m = L.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(L - m).sum(axis=1))
        scores.append(sub)
        am = L.argmax(axis=1)
        in_ids.append(np.isin(am, ids))
        p_any.append(np.exp(sub - lse[:, None]).sum(axis=1))
        top_ids.append(am)
        top_p.append(np.exp(L[np.arange(L.shape[0]), am] - lse))
        if alt_ids is not None:
            asub = L[:, alt_ids]
            alt_scores.append(asub)
            alt_in.append(np.isin(am, alt_ids))
            alt_p.append(np.exp(asub - lse[:, None]).sum(axis=1))
    out = {"scores": np.concatenate(scores), "in_ids": np.concatenate(in_ids), "p_any": np.concatenate(p_any),
           "top_ids": np.concatenate(top_ids), "top_p": np.concatenate(top_p)}
    if alt_ids is not None:
        out["alt"] = {"ids": alt_ids, "scores": np.concatenate(alt_scores),
                      "in_ids": np.concatenate(alt_in), "p_any": np.concatenate(alt_p)}
    return out


def _cloze_read(fwd, prompts: list, options: list):
    """4 continuations (" " + text.strip()) per prompt through continuation_logprobs → per-token-mean (n, 4),
    sums (n, 4), ntoks (n, 4)."""
    flat_p, flat_c = [], []
    for p, opts in zip(prompts, options):
        assert len(opts) == 4, f"need 4 options, got {len(opts)}"
        for o in opts:
            flat_p.append(p)
            flat_c.append(" " + str(o).strip())
    bs = int(fwd.batch_size)
    res = []
    for s in range(0, len(flat_p), bs):
        res.extend(fwd.continuation_logprobs(flat_p[s:s + bs], flat_c[s:s + bs]))
    assert len(res) == len(flat_p), f"{len(res)} continuation scores for {len(flat_p)} requests"
    sums = np.array([float(r[0]) for r in res]).reshape(len(prompts), 4)
    ntok = np.array([int(r[1]) for r in res]).reshape(len(prompts), 4)
    assert (ntok >= 1).all() and np.isfinite(sums).all()
    return sums / ntok, sums, ntok


def _generate(fwd, prompts: list, max_new_tokens: int) -> list:
    bs = int(fwd.batch_size)
    out = []
    for s in range(0, len(prompts), bs):
        out.extend(fwd.generate(prompts[s:s + bs], max_new_tokens))
    assert len(out) == len(prompts)
    return [str(o) for o in out]




def collect_samples(obj, path=(), out=None) -> dict:
    """Walk any banked structure and pull out every readout record that carries `extra.samples`, keyed by
    its path. Used by the behaviour, graft and recovery jobs alike so one `samples.json` per cell covers
    every eval that cell ran (spec §12.11) without each job knowing the others' layouts."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        ex = obj.get("extra") if isinstance(obj.get("extra"), dict) else {}
        sm = ex.get("samples")
        if sm:
            # acc/n/floor fall back to `extra` because not every readout carries them at the top level:
            # P banks its record keyed by LAYER (`cols.P.after.<half>.<L>`) with no scalar accuracy of its
            # own, and ppl banks a perplexity. Without the fallback those blocks render with no number at
            # all beside their samples.
            def _f(k):
                v = obj.get(k)
                return ex.get(k) if v is None else v
            out["/".join(str(p) for p in path)] = {"col": _f("col"), "acc": _f("acc"),
                                                   "n": _f("n"), "floor": _f("floor"), "samples": sm}
        for k, v in obj.items():
            if k != "extra":
                collect_samples(v, path + (k,), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            collect_samples(v, path + (i,), out)
    return out

# ------------------------------------------------------------------ real-inference samples (report)
SAMPLES_N = 3
SAMPLE_PROMPT_CHARS = 700


def _samples(prompts, per_item, pred, golds, scores=None, options=None, decoded=None, labels=None,
             n=SAMPLES_N, extra_per_item=None, index_into_candidates=True, score_label=None) -> list:
    """Real inference samples for the sample report: the ACTUAL prompt (tail), what the model actually
    predicted, the gold, the per-candidate scores it chose from, and the decoded generation where the
    readout generates. Chosen to be informative rather than first-n: the BEST-scoring item, the
    WORST-scoring item, then fill in order. Every eval in the program banks these, so the report shows
    real input and real output for each readout rather than a summary statistic.

    Selection is by score rank, not by an equality test against 1.0 and 0.0 (2026-09-03 defect). The old
    rule took `per >= 0.999` as "right" and `per <= 0.001` as "wrong" and, when either list came back
    empty, filled the remaining slots with items 0, 1, 2 IN ORDER — so on a FRACTIONAL readout the
    "wrong" sample was not the worst item, just the first one. F3 scores
    0.5·(1[gold→Yes] + n_wrong→No / 3) and is exactly such a readout.

    Measured against the 24 F3 blocks banked on S3 (2026-09-03): 7 of 24 banked no clearly-right /
    clearly-wrong PAIR and 2 of 24 banked three all-partial samples. Re-selecting from the same 24 banked
    per-item vectors, this rule changes the chosen items on 4 and shows a strictly worse worst-item on 3
    (e.g. 3a_beh after/discovery: the old rule showed 0.5 as its worst, the real minimum is 0.167). The
    other no-pair blocks are irreducible — 3 of those vectors contain no item at 1.0 and 6 contain none
    at 0.0, so no selection rule can produce a pair that is not in the data; what changes there is that
    the report now SAYS so instead of showing three partial scores with no explanation. O-textvqa (ANLS)
    is the same shape and would hit this on the first vision cell.

    Ranking degenerates to the old behaviour on a binary readout (argmax picks a 1.0, argmin a 0.0).
    Every sample says which it is in `selection`, and `score_kind` says whether partial credit was
    possible at all.

    `index_into_candidates` says whether `pred`/`golds` are positions in `candidates`. A generates an
    INTEGER answer while its candidate list is four option texts, so on A they are not (and the report's
    chosen/gold markers landed on option 0-3 whenever the arithmetic answer happened to be 0-3). Callers
    whose predictions do index the candidates get explicit `chosen_index` / `gold_index`; the report marks
    ONLY those, never a bare `predicted`."""
    per = [float(x) for x in per_item]
    frac = any(0.0 < v < 1.0 for v in per)
    ranked = [(int(np.argmax(per)), "best-scoring"), (int(np.argmin(per)), "worst-scoring")]
    order, why = [], {}
    for pick, tag in ranked + [(i, "fill (next in order)") for i in range(len(per))]:
        if pick in why:
            continue
        why[pick] = tag; order.append(pick)
        if len(order) >= n:
            break
    out = []
    for i in order:
        p = str(prompts[i]) if i < len(prompts) else None
        d = {"item_index": int(i),
             "selection": why[i],
             "score_kind": "fractional (partial credit possible)" if frac else "binary (right / wrong)",
             "prompt_tail": (p[-SAMPLE_PROMPT_CHARS:] if p else None),
             "prompt_chars": (len(p) if p else None),
             "predicted": (int(pred[i]) if not isinstance(pred[i], str) else pred[i]),
             "gold": (int(golds[i]) if golds is not None and not isinstance(golds[i], str) else
                      (golds[i] if golds is not None else None)),
             "scored_correct": per[i]}
        if score_label is not None and i < len(score_label):
            d["score_label"] = str(score_label[i])
        if labels is not None:
            d["answer_labels"] = list(labels)
        if options is not None and i < len(options):
            d["candidates"] = [str(o) for o in options[i]]
            if index_into_candidates:
                if not isinstance(pred[i], str):
                    d["chosen_index"] = int(pred[i])
                if golds is not None and not isinstance(golds[i], str):
                    d["gold_index"] = int(golds[i])
        if scores is not None and i < len(scores):
            d["candidate_scores"] = [float(x) for x in scores[i]]
        if decoded is not None and i < len(decoded):
            d["model_output_text"] = str(decoded[i])
        if extra_per_item:
            for k, v in extra_per_item.items():
                if i < len(v):
                    d[k] = v[i]
        out.append(d)
    return out


def _attach_top_tokens(fwd, prompts, samples) -> None:
    """§12.11 requires the ACTUAL top token over the full vocabulary, with its probability, on EVERY
    readout. The cloze family (K, K-null, F5, RFP, TQ) and A score continuations and never look at the
    read position, so they banked none. Taken here for the SAMPLED prompts only — one extra forward of
    three sequences per readout, not per item, so the cost is three prompts against n·4 continuations."""
    idx = [int(s["item_index"]) for s in samples
           if s.get("item_index") is not None and int(s["item_index"]) < len(prompts)]
    if not idx:
        return
    L = _np(fwd.next_token_logits([prompts[i] for i in idx]))
    assert L.ndim == 2 and L.shape[0] == len(idx), L.shape
    am = L.argmax(axis=1)
    m = L.max(axis=1)
    lse = m + np.log(np.exp(L - m[:, None]).sum(axis=1))
    top = {i: (int(am[j]), float(np.exp(L[j, am[j]] - lse[j]))) for j, i in enumerate(idx)}
    for s in samples:
        t = top.get(s.get("item_index"))
        if t is not None:
            s["model_top_token_id"], s["model_top_token_p"] = t


def _record(col, floor, per_item, hashes, scores, pred, extra) -> dict:
    per_item = [float(x) for x in per_item]
    n = len(per_item)
    assert n > 0 and len(hashes) == n and len(pred) == n and (scores is None or len(scores) == n)
    per_item = [int(x) if x in (0.0, 1.0) else x for x in per_item]
    return {"col": col, "acc": float(np.mean(per_item)), "n": n, "floor": float(floor), "per_item": per_item,
            "per_item_hash": list(hashes), "per_item_scores": scores, "pred": [int(p) for p in pred],
            "extra": extra}


def _shots(dev_shots, subject, k):
    rows = list(dev_shots(subject, k))
    assert len(rows) == k, f"{subject}: {len(rows)} shots for k={k}"
    return rows


def _top_summary(top_ids, top_p, k: int = 5) -> list:
    """The k most frequent argmax tokens with their share of items and mean probability — the diagnostic
    that names what the model actually wants to say at the read position (smoke 4)."""
    ids, counts = np.unique(top_ids, return_counts=True)
    order = np.argsort(-counts)[:k]
    return [{"id": int(ids[i]), "frac_items": float(counts[i] / len(top_ids)),
             "mean_p": float(top_p[top_ids == ids[i]].mean())} for i in order]


def _letter_readout(fwd, col, prompts, ids, golds, hashes, extra, floor=FLOOR_LETTER, alt_ids=None) -> dict:
    r = _letter_read(fwd, prompts, ids, alt_ids)
    scores, in_ids, p_any = r["scores"], r["in_ids"], r["p_any"]
    pred = scores.argmax(axis=1)
    golds = np.asarray(golds, int)
    extra = {**extra, "argmax_in_letters_frac": float(in_ids.mean()), "p_any_letter": float(p_any.mean()),
             "letter_ids": [int(i) for i in ids], "key": golds.tolist(), "prompt_example": prompts[0],
             "top_tokens": _top_summary(r["top_ids"], r["top_p"])}
    extra["samples"] = _samples(prompts, (pred == golds).astype(float), pred, golds, scores.tolist(),
                                labels=[str(i) for i in ids],
                                extra_per_item={"model_top_token_id": [int(x) for x in r["top_ids"]],
                                                "model_top_token_p": [float(x) for x in r["top_p"]]})
    if "alt" in r:
        a = r["alt"]
        apred = a["scores"].argmax(axis=1)
        extra["alt_surface"] = {"letter_ids": a["ids"], "acc": float((apred == golds).mean()),
                                "per_item": [int(x) for x in (apred == golds)], "pred": [int(p) for p in apred],
                                "argmax_in_letters_frac": float(a["in_ids"].mean()),
                                "p_any_letter": float(a["p_any"].mean()),
                                "per_item_scores": a["scores"].tolist()}
        # §12.11 on `<col>-native`. `matrix_behaviour.Job.bank` promotes this surface to a FULL column and
        # every number in it was unsampled — and this is the exact readout the 2026-09-03 instrument alarm
        # was about (the blank-image shell scores a letter surface the model does not emit), so it is the
        # last one that may be a number alone. The samples are selected on the ALT surface's own
        # correctness, so `-native` shows the items IT got right and wrong, not S's. Same forward, no
        # extra compute: the full-vocabulary top token is a property of the read position, not of which
        # four columns are scored, so `r["top_ids"]` is the honest top token for both surfaces.
        extra["alt_surface"]["samples"] = _samples(
            prompts, (apred == golds).astype(float), apred, golds, a["scores"].tolist(),
            labels=[str(i) for i in a["ids"]],
            extra_per_item={"model_top_token_id": [int(x) for x in r["top_ids"]],
                            "model_top_token_p": [float(x) for x in r["top_p"]]})
    return _record(col, floor, pred == golds, hashes, scores.tolist(), pred, extra)


# ------------------------------------------------------------------------------------------ blocks
def _q_block_labeled(question, choices, labels, answer_idx=None) -> str:
    """`_q_block` with arbitrary labels: with CHOICES it is byte-identical to text_eval._q_block."""
    assert len(choices) == 4 and len(labels) == 4
    s = question.strip() + "\n"
    for lab, c in zip(labels, choices):
        s += f"{lab}. {c}\n"
    s += "Answer:"
    if answer_idx is not None:
        s += f" {labels[answer_idx]}\n\n"
    return s


def _q_block_paren(question, choices, answer_idx=None) -> str:
    assert len(choices) == 4
    s = question.strip() + "\n"
    for L, c in zip(CHOICES, choices):
        s += f"({L}) {c}\n"
    s += "Answer: ("
    if answer_idx is not None:
        s += f"{CHOICES[answer_idx]})\n\n"
    return s


def _q_block_answerword(question, choices, answer_idx=None) -> str:
    s = _q_block_labeled(question, choices, CHOICES)
    if answer_idx is not None:
        s += f" {str(choices[answer_idx]).strip()}\n\n"
    return s


def _block_verify(question, choices, option, label=None) -> str:
    """The item WITH its lettered option list (the F family holds item + options fixed and varies only the
    answer format), then one proposed option and the Yes/No question."""
    assert len(choices) == 4
    s = question.strip() + "\n" + "".join(f"{L}. {c}\n" for L, c in zip(CHOICES, choices))
    s += "Proposed answer: " + str(option).strip() + ".\n" + F3_ASK
    if label is not None:
        s += f" {label}\n\n"
    return s


def s_prompt(item, shot_rows) -> str:
    return (_subject_header(item["subject"])
            + "".join(_q_block(s["question"], s["choices"], s["answer"]) for s in shot_rows)
            + _q_block(item["question"], item["choices"]))


def k_prompt(item, shot_rows) -> str:
    return (_subject_header(item["subject"])
            + "".join(_q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in shot_rows)
            + _q_block_cloze(item["question"]))


def _hashes(items):
    """The per-item alignment key as the SET BUILDER minted it (`matrix_items.item_hash`, uniqueness
    asserted at build time). Never re-derived here: one recipe, one place."""
    return [it["hash"] for it in items]


# ------------------------------------------------------------------------------------------ S, F1–F5
def S(fwd, items, shots: int, dev_shots=None) -> dict:
    dev_shots = dev_shots or MI.mmlu_dev_shots
    prompts = [s_prompt(it, _shots(dev_shots, it["subject"], shots)) for it in items]
    # alt surface = the NO-SPACE letters: in the blank-image shell the processor's trailing newline makes
    # them the on-format answer (smoke 4). Same forward, four more logit columns, no extra compute.
    return _letter_readout(fwd, "S" if shots == 5 else f"S-{shots}", prompts, fwd.letter_ids(CHOICES, True),
                           [it["answer"] for it in items], _hashes(items), {"shots": shots},
                           alt_ids=fwd.letter_ids(CHOICES, False))


def f1_items(items, seed=SEED) -> list:
    rng = random.Random(seed)
    out = []
    for it in items:
        perm = _derangement(rng, 4)
        out.append({**it, "choices": [it["choices"][perm[j]] for j in range(4)],
                    "answer": perm.index(int(it["answer"])), "f1_perm": perm})
    return out


def F1(fwd, items, shots=5, dev_shots=None, seed=SEED) -> dict:
    rec = S(fwd, f1_items(items, seed), shots, dev_shots)
    rec["col"], rec["extra"]["seed"] = "F1", seed
    return rec


def F2(fwd, items, shots=5, dev_shots=None) -> dict:
    dev_shots = dev_shots or MI.mmlu_dev_shots
    prompts = []
    for it in items:
        rows = _shots(dev_shots, it["subject"], shots)
        prompts.append(_subject_header(it["subject"])
                       + "".join(_q_block_labeled(s["question"], s["choices"], DIGITS, s["answer"]) for s in rows)
                       + _q_block_labeled(it["question"], it["choices"], DIGITS))
    return _letter_readout(fwd, "F2", prompts, fwd.letter_ids(DIGITS, True), [it["answer"] for it in items],
                           _hashes(items), {"shots": shots, "labels": list(DIGITS)})


def F3(fwd, items, shots=5, dev_shots=None) -> dict:
    dev_shots = dev_shots or MI.mmlu_dev_shots
    prompts, is_gold = [], []
    for it in items:
        rows = _shots(dev_shots, it["subject"], shots)
        prefix = _subject_header(it["subject"])
        for j, s in enumerate(rows):                       # alternate true / false proposals
            k = int(s["answer"]) if j % 2 == 0 else (int(s["answer"]) + 1) % 4
            prefix += _block_verify(s["question"], s["choices"], s["choices"][k], YESNO[0 if j % 2 == 0 else 1])
        for k in range(4):
            prompts.append(prefix + _block_verify(it["question"], it["choices"], it["choices"][k]))
            is_gold.append(k == int(it["answer"]))
    _r3 = _letter_read(fwd, prompts, fwd.letter_ids(YESNO, True))
    scores, in_ids, p_any = _r3["scores"], _r3["in_ids"], _r3["p_any"]
    margin = (scores[:, 0] - scores[:, 1]).reshape(len(items), 4)
    said_yes = margin > 0
    gold = np.asarray(is_gold, bool).reshape(len(items), 4)
    tp, fn = int((said_yes & gold).sum()), int((~said_yes & gold).sum())
    tn, fp = int((~said_yes & ~gold).sum()), int((said_yes & ~gold).sum())
    per_item = 0.5 * (said_yes[gold].astype(float) + (~said_yes & ~gold).sum(axis=1) / 3.0)
    golds = [int(it["answer"]) for it in items]
    extra = {"shots": shots, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
             "per_item_kind": "fractional (balanced-accuracy decomposition)",
             "balanced_accuracy": MC.balanced_accuracy(tp, fn, tn, fp), "yes_rate": float(said_yes.mean()),
             "argmax_in_letters_frac": float(in_ids.mean()), "p_any_letter": float(p_any.mean()),
             "letter_ids": [int(i) for i in fwd.letter_ids(YESNO, True)], "key": golds, "prompt_example": prompts[0]}
    # F3 asks four yes/no questions per item; the sample shows all four prompts and the model's yes/no
    # margin on each, so the report can show the actual verification behaviour rather than one number.
    per4 = np.asarray(prompts, object).reshape(len(items), 4)
    extra["samples"] = _samples([p[0] for p in per4], per_item, margin.argmax(axis=1), golds,
                                margin.reshape(len(items), 4).tolist(),
                                options=[list(it["choices"]) for it in items],
                                extra_per_item={"four_prompts_tail": [[str(x)[-260:] for x in row] for row in per4],
                                                "said_yes": said_yes.reshape(len(items), 4).tolist(),
                                                "is_gold": gold.tolist()})
    rec = _record("F3", FLOOR_F3, per_item, _hashes(items), margin.tolist(), margin.argmax(axis=1), extra)
    assert abs(rec["acc"] - extra["balanced_accuracy"]) < 1e-12
    return rec


def F4(fwd, items, shots=5, dev_shots=None) -> dict:
    dev_shots = dev_shots or MI.mmlu_dev_shots
    prompts = []
    for it in items:
        rows = _shots(dev_shots, it["subject"], shots)
        prompts.append(_subject_header(it["subject"])
                       + "".join(_q_block_paren(s["question"], s["choices"], s["answer"]) for s in rows)
                       + _q_block_paren(it["question"], it["choices"]))
    return _letter_readout(fwd, "F4", prompts, fwd.letter_ids(CHOICES, False), [it["answer"] for it in items],
                           _hashes(items), {"shots": shots, "labels": "(A)"})


def _cloze_readout(fwd, col, prompts, options, golds, hashes, extra, floor=FLOOR_LETTER) -> dict:
    mean, sums, ntok = _cloze_read(fwd, prompts, options)
    pred = mean.argmax(axis=1)
    golds = np.asarray(golds, int)
    extra = {**extra, "cloze_logprobs": sums.tolist(), "cloze_ntoks": ntok.tolist(), "key": golds.tolist(),
             "prompt_example": prompts[0]}
    extra["samples"] = _samples(prompts, (pred == golds).astype(float), pred, golds, mean.tolist(),
                                options=options,
                                extra_per_item={"candidate_sum_logprob": sums.tolist(),
                                                "candidate_n_tokens": ntok.tolist()})
    _attach_top_tokens(fwd, prompts, extra["samples"])   # §12.11: K / K-null / F5 / RFP / TQ banked none
    return _record(col, floor, pred == golds, hashes, mean.tolist(), pred, extra)


def F5(fwd, items, shots=5, dev_shots=None) -> dict:
    dev_shots = dev_shots or MI.mmlu_dev_shots
    prompts = []
    for it in items:
        rows = _shots(dev_shots, it["subject"], shots)
        prompts.append(_subject_header(it["subject"])
                       + "".join(_q_block_answerword(s["question"], s["choices"], s["answer"]) for s in rows)
                       + _q_block_answerword(it["question"], it["choices"]))
    return _cloze_readout(fwd, "F5", prompts, [it["choices"] for it in items], [it["answer"] for it in items],
                          _hashes(items), {"shots": shots})


# ------------------------------------------------------------------------------------------ K, RFP, A, TQ, U
def K(fwd, items, shots=5, dev_shots=None, col="K") -> dict:
    dev_shots = dev_shots or MI.mmlu_dev_shots
    prompts = [k_prompt(it, _shots(dev_shots, it["subject"], shots)) for it in items]
    return _cloze_readout(fwd, col, prompts, [it["choices"] for it in items], [it["answer"] for it in items],
                          _hashes(items), {"shots": shots})


def K_shuffle_null(fwd, items, shots=5, seed=SEED, dev_shots=None) -> dict:
    perm = _derangement(random.Random(seed), len(items))
    null_items = [{**it, "choices": list(items[perm[i]]["choices"])} for i, it in enumerate(items)]
    rec = K(fwd, null_items, shots, dev_shots, col="K-null")
    rec["extra"].update({"seed": seed, "perm": perm})
    return rec


def RFP_readout(fwd, items, exemplars, n_shots=5, drop_passage=False) -> dict:
    prefix = RFP._build_prefix_cloze(exemplars, n_shots, RFP.RFP_HEADER, drop_passage)
    prompts = [prefix + RFP._block_cloze(it["passage"], it["question"], None, drop_passage) for it in items]
    return _cloze_readout(fwd, "RFP", prompts, [it["choices"] for it in items], [it["answer"] for it in items],
                          [RFP.item_hash(it) for it in items], {"n_shots": n_shots, "drop_passage": drop_passage},
                          floor=FLOOR_RFP)


def _first_int(text: str) -> int:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else -1


def A(fwd, items, exemplars=None, max_new_tokens=8) -> dict:
    exemplars = exemplars if exemplars is not None else MI.arith_exemplars()
    prefix = A_HEADER + "".join(_q_block_cloze(e["question"], e["answer"]) for e in exemplars)
    prompts = [prefix + _q_block_cloze(it["question"]) for it in items]
    gens = _generate(fwd, prompts, max_new_tokens)
    pred = [_first_int(g) for g in gens]
    em = [int(p == int(it["answer"])) for p, it in zip(pred, items)]
    mean, sums, ntok = _cloze_read(fwd, prompts, [it["options"] for it in items])
    cand_pred = mean.argmax(axis=1)
    cand = (cand_pred == np.asarray([it["gold_idx"] for it in items], int)).astype(int)
    extra = {"max_new_tokens": max_new_tokens, "generations": gens, "cand_acc": float(cand.mean()),
             "cand_per_item": cand.tolist(), "cand_pred": cand_pred.tolist(), "cloze_logprobs": sums.tolist(),
             "cloze_ntoks": ntok.tolist(), "key": [int(it["answer"]) for it in items],
             "gold_idx": [int(it["gold_idx"]) for it in items], "prompt_example": prompts[0]}
    # A's `pred` is the INTEGER the model generated and its gold is that integer's value — neither is a
    # position in the four candidate texts, so the report's chosen/gold markers mismarked option 0-3
    # whenever the arithmetic answer happened to be 0-3 (2026-09-03 defect). The candidate positions A
    # really chose and was scored against travel as their own fields.
    extra["samples"] = _samples(prompts, em, pred, [int(it["answer"]) for it in items], mean.tolist(),
                                options=[it["options"] for it in items], decoded=gens,
                                index_into_candidates=False,
                                extra_per_item={"candidate_correct": cand.tolist(),
                                                "chosen_index": cand_pred.tolist(),
                                                "gold_index": [int(it["gold_idx"]) for it in items]})
    _attach_top_tokens(fwd, prompts, extra["samples"])   # §12.11: A banked no top token either
    return _record("A", FLOOR_A, em, _hashes(items), mean.tolist(), pred, extra)


def TQ(fwd, items, exemplars) -> dict:
    prefix = TQ_HEADER + "".join(_q_block_cloze(e["question"], e["answer"]) for e in exemplars)
    prompts = [prefix + _q_block_cloze(it["question"]) for it in items]
    return _cloze_readout(fwd, "TQ", prompts, [it["options"] for it in items], [it["gold_idx"] for it in items],
                          _hashes(items), {"n_shots": len(exemplars)})


def U(fwd, bundle, shots=5) -> dict:
    rec = S(fwd, bundle["items"], shots, dev_shots=lambda subj, k: bundle["exemplars"][:k])
    rec["col"] = "U"
    return rec


def U_shortcut(fwd, bundle, shots=5) -> dict:
    rec = S(fwd, bundle["shortcut"], shots, dev_shots=lambda subj, k: bundle["shortcut_exemplars"][:k])
    rec["col"] = "U-shortcut"
    return rec


# ------------------------------------------------------------------------------------------ H, tokenizer gate
def H(pred, n_choices=4) -> float:
    return MC.letter_entropy(pred, n_choices)


def H_record(rec) -> dict:
    return {"col": rec["col"], "H_pred_bits": H(rec["pred"]), "H_key_bits": H(rec["extra"]["key"])}


def tokenizer_gate_stats(fwd, s_rec) -> dict:
    """`letter_token_counts` reports the raw encode counts WITHOUT asserting, so a multi-token letter reaches
    matrix_core.tokenizer_gate as a count ≠ 1 (VOID-tokenizer reachable); ids are read only when all are 1."""
    counts = [int(c) for c in fwd.letter_token_counts(CHOICES, True)]
    assert len(counts) == 4, counts
    ids = [int(i) for i in fwd.letter_ids(CHOICES, True)] if all(c == 1 for c in counts) else None
    alt = s_rec["extra"].get("alt_surface") or {}
    out = {"letter_ids": ids, "letter_token_counts": counts,
           "argmax_in_letters_frac": float(s_rec["extra"]["argmax_in_letters_frac"]),
           "p_any_letter": float(s_rec["extra"]["p_any_letter"]),
           "acc": float(s_rec["acc"])}
    if alt:
        # v4: the second surface, and how often the two surfaces pick the same option per item.
        # v5 (D-B): BOTH surfaces' accuracy — that pair is what the gate now tests (the measurement must
        # not move when the other surface is read). Agreement stays, split by whether the scored surface
        # got the item right: near the floor a guessing model disagrees with itself on the items it gets
        # wrong, which is expected and must not void anything.
        a, b = s_rec["pred"], alt["pred"]
        agree = np.asarray([int(x == y) for x, y in zip(a, b)])
        right = np.asarray(s_rec["per_item"], float) > 0
        out.update({"native_argmax_frac": float(alt["argmax_in_letters_frac"]),
                    "native_p_any_letter": float(alt["p_any_letter"]),
                    "native_acc": float(alt["acc"]),
                    "surface_agreement": float(agree.mean()),
                    "surface_agreement_when_right": float(agree[right].mean()) if right.any() else None,
                    "surface_agreement_when_wrong": float(agree[~right].mean()) if (~right).any() else None})
    return out


def tokenizer_gate(stats_before, stats_after) -> dict:
    """matrix_core.tokenizer_gate v5 (2026-09-03): single-token letters; the healthy model on-format on
    SOME scored surface (space-prefixed or no-space); the two surfaces giving the SAME accuracy
    (|Δ| ≤ 0.03); and the after model's letter mass taken as the max over surfaces. The v3 rule tested
    only the scored surface and would have voided the shell readout, which the two-surface measurement
    showed measures the right thing. The v4 rule tested per-item AGREEMENT between the surfaces and voided
    ctrl-g1 at 0.74/0.77 even though both surfaces returned identical accuracy — agreement is now a
    reported diagnostic and accuracy invariance is the gate."""
    return {**MC.tokenizer_gate(stats_before["letter_token_counts"], stats_before["argmax_in_letters_frac"],
                                stats_before["p_any_letter"], stats_after["p_any_letter"],
                                native_argmax_frac=stats_before.get("native_argmax_frac"),
                                native_p_after=stats_after.get("native_p_any_letter"),
                                surface_agreement=stats_before.get("surface_agreement"),
                                acc_scored=stats_before.get("acc"), acc_native=stats_before.get("native_acc"),
                                agreement_when_right=stats_before.get("surface_agreement_when_right"),
                                agreement_when_wrong=stats_before.get("surface_agreement_when_wrong")),
            "before": stats_before, "after": stats_after}


# ------------------------------------------------------------------------------------------ dry-run
class FakeForward:
    """Contract-§3-shaped stub: deterministic pseudo-random logits keyed on the prompt text (numpy only)."""
    n_layers, family, shell, batch_size, V = 2, "fake", "text", 3, 64

    def __init__(self, multi_token=False):
        self.calls = {"logits": 0, "cont": 0, "gen": 0}
        self.multi_token = multi_token

    def letter_token_counts(self, letters=CHOICES, space=True):
        return [2 if self.multi_token else 1 for _ in letters]

    def _rng(self, key):
        import hashlib
        return np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))

    def next_token_logits(self, prompts):
        self.calls["logits"] += len(prompts)
        return np.stack([self._rng("L|" + p).normal(size=self.V) for p in prompts])

    def continuation_logprobs(self, prompts, continuations):
        assert len(prompts) == len(continuations)
        self.calls["cont"] += len(prompts)
        return [(float(-abs(self._rng("C|" + p + "|" + c).normal()) * len(c.split())), len(c.split()))
                for p, c in zip(prompts, continuations)]

    def generate(self, prompts, max_new_tokens):
        self.calls["gen"] += len(prompts)
        return [f" {int(self._rng('G|' + p).integers(0, 60))}\n\nnext" for p in prompts]

    def letter_ids(self, letters=CHOICES, space=True):
        ids = [(10 if space else 40) + sum(map(ord, L)) % 23 for L in letters]
        assert len(set(ids)) == len(ids)
        return ids


def _fake_items(n_per_subject=3):
    rng = random.Random(3)
    subjects = ("anatomy", "world_religions")
    dev = {s: [{"subject": s, "question": f"Dev {i} of {s}?", "choices": [f"d{i}{j} {s[:3]}" for j in range(4)],
                "answer": rng.randrange(4)} for i in range(5)] for s in subjects}
    items = [{"subject": s, "question": f"Test {i} of {s}?", "choices": [f"opt{i}{j} {s[:3]} word" for j in range(4)],
              "answer": rng.randrange(4), "hash": MI.item_hash(s, f"Test {i} of {s}?")}
             for s in subjects for i in range(n_per_subject)]
    return items, (lambda subj, k: dev[subj][:k])


def _check_shape(rec, n):
    assert set(rec) == {"col", "acc", "n", "floor", "per_item", "per_item_hash", "per_item_scores", "pred", "extra"}
    assert rec["n"] == n == len(rec["per_item"]) == len(rec["per_item_hash"]) == len(rec["pred"])
    assert abs(rec["acc"] - float(np.mean(rec["per_item"]))) < 1e-12
    assert rec["per_item_scores"] is None or all(len(s) == 4 for s in rec["per_item_scores"])


def _dryrun():
    from infra.pg import two_hop_items as TH
    fwd = FakeForward()
    items, dev_shots = _fake_items()
    n = len(items)

    # S: byte-identical to the text_eval 5-shot prompt; deterministic; k ∈ {0, 1, 5}
    rec = S(fwd, items, 5, dev_shots)
    _check_shape(rec, n)
    ref = (_subject_header("anatomy") + "".join(_q_block(s["question"], s["choices"], s["answer"]) for s in dev_shots("anatomy", 5))
           + _q_block(items[0]["question"], items[0]["choices"]))
    assert rec["extra"]["prompt_example"] == ref and ref.endswith("Answer:")
    assert rec == S(fwd, items, 5, dev_shots) and rec["col"] == "S"
    assert S(fwd, items, 0, dev_shots)["col"] == "S-0" and "Answer: " not in S(fwd, items, 0, dev_shots)["extra"]["prompt_example"]
    assert 0.0 <= rec["extra"]["p_any_letter"] <= 1.0 and rec["extra"]["letter_ids"] == fwd.letter_ids(CHOICES, True)
    assert rec["per_item_hash"] == [it["hash"] for it in items] == [MI.item_hash(it["subject"], it["question"]) for it in items]
    assert len(set(rec["per_item_hash"])) == len(items)          # the key is an identity

    # F1: deranged option order, gold tracked, hashes unchanged
    f1 = F1(fwd, items, 5, dev_shots)
    _check_shape(f1, n)
    for it, fi in zip(items, f1_items(items)):
        assert all(fi["f1_perm"][j] != j for j in range(4)) and fi["choices"][fi["answer"]] == it["choices"][it["answer"]]
    assert f1["per_item_hash"] == rec["per_item_hash"] and f1_items(items) == f1_items(items)

    # F2: numbered labels, digit ids; the labeled block with CHOICES equals text_eval._q_block
    f2 = F2(fwd, items, 5, dev_shots)
    _check_shape(f2, n)
    assert "\n1. " in f2["extra"]["prompt_example"] and re.search(r"Answer: [1-4]\n\n", f2["extra"]["prompt_example"])
    assert f2["extra"]["prompt_example"].endswith("4. " + items[0]["choices"][3] + "\nAnswer:")
    assert f2["extra"]["letter_ids"] == fwd.letter_ids(DIGITS, True)
    assert _q_block_labeled("q", ["a", "b", "c", "d"], CHOICES, 2) == _q_block("q", ["a", "b", "c", "d"], 2)

    # F3: 4 prompts per item, counts sum right, mean(per_item) == balanced accuracy
    before = fwd.calls["logits"]
    f3 = F3(fwd, items, 5, dev_shots)
    _check_shape(f3, n)
    e = f3["extra"]
    assert fwd.calls["logits"] - before == 4 * n and e["tp"] + e["fn"] == n and e["tn"] + e["fp"] == 3 * n
    assert abs(f3["acc"] - MC.balanced_accuracy(e["tp"], e["fn"], e["tn"], e["fp"])) < 1e-12 and f3["floor"] == 0.5
    assert e["prompt_example"].endswith(F3_ASK) and e["prompt_example"].count("Proposed answer:") == 6
    assert " Yes\n\n" in e["prompt_example"] and " No\n\n" in e["prompt_example"]
    assert e["prompt_example"].count("\nA. ") == 6 and e["per_item_kind"].startswith("fractional")
    assert _q_block(items[0]["question"], items[0]["choices"])[:-len("Answer:")] in e["prompt_example"]

    # F4: "(A)" labels, read after "(" with no-space ids
    f4 = F4(fwd, items, 5, dev_shots)
    _check_shape(f4, n)
    assert f4["extra"]["prompt_example"].endswith("Answer: (") and "(A) " in f4["extra"]["prompt_example"]
    assert "Answer: (" in f4["extra"]["prompt_example"][:-9] and f4["extra"]["letter_ids"] == fwd.letter_ids(CHOICES, False)

    # F5: options shown, option TEXT scored; K: mmlu_cloze prompt + per-token-mean argmax
    before = fwd.calls["cont"]
    f5 = F5(fwd, items, 5, dev_shots)
    _check_shape(f5, n)
    assert fwd.calls["cont"] - before == 4 * n and "\nA. " in f5["extra"]["prompt_example"]
    k = K(fwd, items, 5, dev_shots)
    _check_shape(k, n)
    kref = (_subject_header("anatomy") + "".join(_q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in dev_shots("anatomy", 5))
            + _q_block_cloze(items[0]["question"]))
    assert k["extra"]["prompt_example"] == kref and "A. " not in kref
    lp = [fwd.continuation_logprobs([kref], [" " + c.strip()])[0] for c in items[0]["choices"]]
    assert k["pred"][0] == int(np.argmax([s / m for s, m in lp])) and len(k["extra"]["cloze_logprobs"][0]) == 4
    kn = K_shuffle_null(fwd, items, 5, dev_shots=dev_shots)
    _check_shape(kn, n)
    assert all(kn["extra"]["perm"][i] != i for i in range(n)) and kn["extra"]["key"] == k["extra"]["key"]
    assert kn["per_item_hash"] == k["per_item_hash"] and kn["col"] == "K-null"

    # RFP through the Forward: header + cloze prefix from read_from_passage
    r_items, r_ex, _ = RFP.build_synth(1, 8)
    rf = RFP_readout(fwd, r_items, r_ex)
    _check_shape(rf, 8)
    assert rf["extra"]["prompt_example"].startswith(RFP.RFP_HEADER) and rf["floor"] == 0.27
    assert rf["per_item_hash"] == [RFP.item_hash(it) for it in r_items]

    # A: em via first integer of the generation; cand beside it. TQ.
    ar = MI.arith_items()[:5]
    a = A(fwd, ar, MI.arith_exemplars())
    _check_shape(a, 5)
    assert a["pred"] == [_first_int(g) for g in a["extra"]["generations"]] and a["floor"] == 0.0
    assert a["per_item"] == [int(p == int(it["answer"])) for p, it in zip(a["pred"], ar)]
    assert _first_int("The answer is 1,234.") == 1234 and _first_int("none") == -1
    assert len(a["extra"]["cand_per_item"]) == 5 and a["extra"]["prompt_example"].startswith(A_HEADER)
    tq_items = [{"subject": MI.TRIVIAQA_SUBJECT, "question": f"Who wrote book {i}?", "answer": f"Author{i}",
                 "hash": MI.item_hash(MI.TRIVIAQA_SUBJECT, f"Who wrote book {i}?"),
                 "options": [f"Author{(i + j) % 4}" for j in range(4)], "gold_idx": 0} for i in range(4)]
    tq = TQ(fwd, tq_items, tq_items[:2])
    _check_shape(tq, 4)

    # U / U-shortcut on the two-hop bundle; H; tokenizer gate
    bundle = TH.build_two_hop(TH.fake_dev_rows())
    u, us = U(fwd, bundle, 5), U_shortcut(fwd, bundle, 5)
    _check_shape(u, 100)
    _check_shape(us, 100)
    assert u["col"] == "U" and us["col"] == "U-shortcut" and "Inner question" not in us["extra"]["prompt_example"]
    assert u["extra"]["prompt_example"].startswith(_subject_header(TH.U_SUBJECT))
    h = H_record(rec)
    assert 0.0 <= h["H_pred_bits"] <= 2.0 and abs(H([0, 1, 2, 3]) - 2.0) < 1e-12 and H([1, 1, 1]) == 0.0
    st = tokenizer_gate_stats(fwd, rec)
    g = tokenizer_gate(st, tokenizer_gate_stats(FakeForward(), rec))
    assert g["S_family"].startswith(("VALID", "VOID-tokenizer")) and {"letter_ids", "letter_token_counts",
        "argmax_in_letters_frac", "p_any_letter", "acc"} <= set(st) and {"native_argmax_frac", "native_acc",
        "surface_agreement", "surface_agreement_when_right", "surface_agreement_when_wrong"} <= set(st)
    assert st["letter_token_counts"] == [1, 1, 1, 1] and st["letter_ids"] == fwd.letter_ids(CHOICES, True)
    assert st["acc"] == rec["acc"] and st["native_acc"] == rec["extra"]["alt_surface"]["acc"]
    assert g["acc_scored"] == st["acc"] and g["acc_native"] == st["native_acc"]
    bad = tokenizer_gate_stats(FakeForward(multi_token=True), rec)      # VOID-tokenizer must be reachable
    assert bad["letter_ids"] is None and tokenizer_gate(bad, st)["S_family"].startswith("VOID-tokenizer")
    # v5 (D-B) both branches of the surface clause, through the readout layer:
    on = {**st, "argmax_in_letters_frac": 1.0}   # hold the other two clauses fixed at PASS
    #  (a) accuracy invariant but per-item agreement low → VALID (the v4 rule voided this)
    inv = {**on, "acc": 0.4386, "native_acc": 0.4386, "surface_agreement": 0.7368}
    assert tokenizer_gate(inv, st)["S_family"] == "VALID"
    #  (b) accuracy surface-DEPENDENT though the surfaces mostly agree per item → VOID
    dep = {**on, "acc": 0.44, "native_acc": 0.36, "surface_agreement": 0.97}
    assert "surface-dependent" in tokenizer_gate(dep, st)["S_family"]
    assert MC.u_valid(u["acc"], us["acc"])["U"] in ("VALID", "VOID-U")
    print(f"S acc {rec['acc']:.3f} F3 ba {f3['acc']:.3f} K acc {k['acc']:.3f} K-null acc {kn['acc']:.3f} "
          f"U acc {u['acc']:.3f}; forward calls {fwd.calls}")
    print("PASS format_readouts dry-run (FakeForward; S/F1-F5/K/K-null/RFP/A/TQ/U/H/tokenizer gate)")


if __name__ == "__main__":
    _dryrun()
