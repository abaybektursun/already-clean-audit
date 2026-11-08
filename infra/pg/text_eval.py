"""W11 Leg-A text-competence primitives (pure, importable; no Modal/AWS deps).

Minimal, transparent text-only instruments for the forgetting-axis test:

  * mmlu_accuracy        — 5-shot MMLU, lm-eval-style single-token-answer
    loglik (read the next-token logit at the four choice letters after
    "Answer:"). The §9-step-1a positive control: base gemma-2-2b must
    reproduce its published 51.3 within tolerance → the text harness is valid.
  * mmlu_blankimg        — the SAME 5-shot MMLU but with a blank gray 224 image
    present (PaliGemma 2's in-distribution forward; text-only is OOD). The W14
    PRIMARY readout; item-order/hash aligned to mmlu_accuracy for paired bootstrap.
  * clean_text_perplexity — wikitext-2-raw per-token cross-entropy (secondary
    fluency read; no published target → sanity only at preflight). Calls the model
    BARE (`model(input_ids=...)`): correct for a text model, WRONG for a decoder
    living in a VLM shell (no pixel_values → not the forward the model is run with).
  * shell_text_perplexity — the same corpus through a `matrix_forward.Forward`, i.e.
    the member's own shell (blank image present on the VLM shells). The MENDELEEV
    `ppl` column; use it whenever the object under test is a shell member.
  * tokenizer_parity     — assert gemma-2-2b's tokenizer and PG2's processor
    tokenizer encode shared-vocab clean text to identical ids (the §8 threat
    "extended-vocab mismatch → spurious perplexity gap").

All take an already-loaded (model, tokenizer); reused at unblind for the
S_text capability readout (base vs PG2-LM vs random-decoder FLOOR). No
try/except hiding — shape/format errors RAISE with a trace. Heavy imports
live inside functions so the module py_compiles on the control plane.
"""
from __future__ import annotations

import hashlib
import math

MMLU_REPO = "cais/mmlu"
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
CHOICES = ("A", "B", "C", "D")

# clean shared-vocab probes for tokenizer parity (no rendered-text / image / loc tokens)
PARITY_PROBES = (
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, Apollo 11 landed on the Moon.",
    "def add(a, b):\n    return a + b\n",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "The capital of France is Paris, and its population exceeds two million.",
)


# ===========================================================================
# MMLU — 5-shot, single-token-answer loglik (lm-eval-harness default style).
# ===========================================================================
def _subject_header(subject: str) -> str:
    return (f"The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n")


def _q_block(question: str, choices, answer_idx: int | None = None) -> str:
    """One MMLU item rendered as `Q\\nA. ..\\n..\\nAnswer:[ L\\n\\n]`. The answer
    (with its leading space + trailing blank line) is appended only for the
    few-shot exemplars; the scored item ends exactly at "Answer:"."""
    assert len(choices) == 4, f"MMLU item must have 4 choices, got {len(choices)}"
    s = question.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{CHOICES[i]}. {c}\n"
    s += "Answer:"
    if answer_idx is not None:
        s += f" {CHOICES[answer_idx]}\n\n"
    return s


def mmlu_accuracy(model, tokenizer, device, n_shots: int = 5,
                  n_per_subject: int | None = None, batch_size: int = 4) -> dict:
    """5-shot MMLU accuracy. For each test item build the subject header + 5 dev
    exemplars + the item ending at "Answer:", run ONE forward, read the next-token
    logits at the four choice-letter token ids (" A".." D"), argmax → prediction.
    Left-padding so position -1 is the true final token for every row.

    n_per_subject=None → the full test split (tightest match to the published
    number). Returns per-subject + overall accuracy."""
    import torch
    from collections import defaultdict
    from datasets import load_dataset

    ds = load_dataset(MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]

    shots_by_subj: dict = defaultdict(list)
    for r in dev:
        shots_by_subj[r["subject"]].append(r)
    test_by_subj: dict = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)

    # token id of the continuation token after "Answer:" for each letter (Gemma "▁A".."▁D")
    letter_ids = []
    for L in CHOICES:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        assert len(ids) == 1, f"' {L}' → {len(ids)} tokens, expected 1 (Gemma '▁{L}')"
        letter_ids.append(ids[0])
    assert len(set(letter_ids)) == 4, f"choice-letter token ids collide: {letter_ids}"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    n_correct = 0
    n_total = 0
    per_subject: dict = {}
    per_item: list = []   # 0/1 in deterministic (sorted-subject, row) order — aligned across models
    per_item_hash: list = []   # sha1(question[:64]) per item — cross-checked at unblind to catch reordering
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:n_shots]
        assert len(shots) == n_shots, f"{subj}: only {len(shots)} dev shots (<{n_shots})"
        prefix = _subject_header(subj) + "".join(
            _q_block(s["question"], s["choices"], s["answer"]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        subj_correct = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [prefix + _q_block(r["question"], r["choices"]) for r in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=True).to(device)
            # explicit content-relative positions (lm-eval recipe): left-pad RoPE
            # must start at the first real token, not at pad — version-proof.
            pos = enc["attention_mask"].long().cumsum(-1) - 1
            pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
            with torch.inference_mode():
                # logits_to_keep=1: compute lm_head for the LAST token only — avoids
                # materialising the (B, S, 256k-vocab) logits that OOMs on long prompts.
                last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
            preds = last[:, letter_ids].argmax(dim=-1).tolist()
            for r, p in zip(batch, preds):
                ok = int(p == int(r["answer"]))
                per_item.append(ok)
                per_item_hash.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])
                subj_correct += ok
                n_correct += ok
                n_total += 1
        per_subject[subj] = subj_correct / len(rows)

    tokenizer.padding_side = orig_side
    acc = n_correct / n_total
    return {
        "task": "MMLU", "metric": "5-shot single-token-answer accuracy",
        "acc": acc, "acc_pct": round(acc * 100, 2),
        "n": n_total, "n_subjects": len(per_subject), "n_shots": n_shots,
        "n_per_subject": n_per_subject, "per_subject": per_subject,
        "per_item": per_item, "per_item_hash": per_item_hash,
        "letter_token_ids": letter_ids,
        "published": 51.3,
        "published_source": ("google/gemma-2-2b card + Gemma 2 tech report arXiv:2408.00118 "
                             "(MMLU 5-shot 51.3); recompute, never transcribed as a baseline"),
    }


# ===========================================================================
# MMLU — blank-image (PaliGemma in-distribution forward). W14 PRIMARY readout.
# ===========================================================================
def mmlu_blankimg(model, processor, device, n_shots: int = 5,
                  n_per_subject: int | None = None, batch_size: int = 4,
                  gray: int = 128, item_filter=None) -> dict:
    """5-shot MMLU with a BLANK (uniform gray 224²) image PRESENT — PaliGemma 2's
    in-distribution forward (it is always trained with 256 image tokens; a text-only forward
    is OOD). This is the W14 PRIMARY readout (prereg §1); `mmlu_accuracy` (text-only) is the
    secondary sensitivity regime. IDENTICAL item order, 5-shot construction, choice-letter
    argmax, and per_item / per_item_hash to `mmlu_accuracy`, so the blank-image and text-only
    reads are paired-bootstrap aligned to each other and to the re-frozen controls.

    Reads the next-token logit at the four choice letters after "Answer:"; left-pad so -1 is
    the real last token; passes ONLY input_ids/attention_mask/pixel_values + logits_to_keep=1
    (no token_type_ids/labels → PaliGemma's training-loss path stays off, which would mismatch
    shapes). gray=128 is the fixed neutral image — it MUST match across controls and every arm.
    """
    import numpy as np
    import torch
    from collections import defaultdict
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]
    shots_by_subj: dict = defaultdict(list)
    for r in dev:
        shots_by_subj[r["subject"]].append(r)
    test_by_subj: dict = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)

    tok = processor.tokenizer
    letter_ids = []
    for L in CHOICES:
        ids = tok.encode(" " + L, add_special_tokens=False)
        assert len(ids) == 1, f"' {L}' → {len(ids)} tokens, expected 1 (Gemma '▁{L}')"
        letter_ids.append(ids[0])
    assert len(set(letter_ids)) == 4, f"choice-letter token ids collide: {letter_ids}"

    gray_img = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    orig_side = tok.padding_side
    tok.padding_side = "left"

    n_correct = 0
    n_total = 0
    per_subject: dict = {}
    per_item: list = []
    per_item_hash: list = []     # sha1(question[:64]) — IDENTICAL recipe to mmlu_accuracy
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:n_shots]
        assert len(shots) == n_shots, f"{subj}: only {len(shots)} dev shots (<{n_shots})"
        prefix = _subject_header(subj) + "".join(
            _q_block(s["question"], s["choices"], s["answer"]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        if item_filter is not None:
            rows = [r for r in rows if item_filter(
                hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])]
        subj_correct = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [prefix + _q_block(r["question"], r["choices"]) for r in batch]
            inp = processor(text=prompts, images=[gray_img] * len(batch),
                            return_tensors="pt", padding="longest").to(device)
            with torch.inference_mode():
                last = model(input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"],
                             pixel_values=inp["pixel_values"].to(torch.bfloat16),
                             logits_to_keep=1).logits[:, -1, :]
            preds = last[:, letter_ids].argmax(dim=-1).tolist()
            for r, p in zip(batch, preds):
                ok = int(p == int(r["answer"]))
                per_item.append(ok)
                per_item_hash.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])
                subj_correct += ok
                n_correct += ok
                n_total += 1
        if rows:                                  # item_filter may empty a subject (collision split)
            per_subject[subj] = subj_correct / len(rows)

    tok.padding_side = orig_side
    assert n_total > 0, "mmlu_blankimg produced no items (over-restrictive item_filter?)"
    acc = n_correct / n_total
    return {
        "task": "MMLU", "metric": "5-shot single-token-answer accuracy (blank-image, in-distribution)",
        "forward": "blank-image", "gray": gray,
        "acc": acc, "acc_pct": round(acc * 100, 2),
        "n": n_total, "n_subjects": len(per_subject), "n_shots": n_shots,
        "n_per_subject": n_per_subject, "per_subject": per_subject,
        "per_item": per_item, "per_item_hash": per_item_hash,
        "letter_token_ids": letter_ids,
    }


# ===========================================================================
# MMLU — choice-logit capture (W16-A read-side addition; mmlu_accuracy UNTOUCHED).
# ===========================================================================
def mmlu_choice_logits(model, tokenizer, device, n_shots: int = 5,
                       n_per_subject: int | None = None, batch_size: int = 4,
                       item_filter=None) -> dict:
    """IDENTICAL prompt construction, item order, letter-id read, and per-item
    hashing to `mmlu_accuracy`, but banks the raw last-token logits at the four
    choice-letter ids per item (`choice_logits`, (N,4) row-major in CHOICES
    order) plus `correct_idx`. The W16-A continuous readouts (margin / Brier /
    rank-AUC, infra.pg.w16_probes) are pure functions of these two arrays; the
    derived per_item (argmax == answer) must EQUAL mmlu_accuracy's on the same
    model/cap — asserted at preflight, not assumed.

    item_filter: optional predicate on the per-item hash (sha1(question[:64])
    [:12]) applied BEFORE the forward — the W16 eval-economics surface (grid
    cells read a fixed discovery sub-bucket; decision cells the confirmation
    half). Filtering preserves the deterministic (sorted-subject, row) order of
    the surviving items, so two filtered runs are per-item aligned."""
    import torch
    from collections import defaultdict
    from datasets import load_dataset

    ds = load_dataset(MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]

    shots_by_subj: dict = defaultdict(list)
    for r in dev:
        shots_by_subj[r["subject"]].append(r)
    test_by_subj: dict = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)

    letter_ids = []
    for L in CHOICES:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        assert len(ids) == 1, f"' {L}' → {len(ids)} tokens, expected 1 (Gemma '▁{L}')"
        letter_ids.append(ids[0])
    assert len(set(letter_ids)) == 4, f"choice-letter token ids collide: {letter_ids}"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    choice_logits: list = []   # (N,4) raw last-token logits at letter ids, CHOICES order
    correct_idx: list = []     # (N,) answer index 0..3
    per_item: list = []        # derived argmax==answer (cross-checked vs mmlu_accuracy)
    per_item_hash: list = []
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:n_shots]
        assert len(shots) == n_shots, f"{subj}: only {len(shots)} dev shots (<{n_shots})"
        prefix = _subject_header(subj) + "".join(
            _q_block(s["question"], s["choices"], s["answer"]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        if item_filter is not None:
            rows = [r for r in rows if item_filter(
                hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [prefix + _q_block(r["question"], r["choices"]) for r in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=True).to(device)
            pos = enc["attention_mask"].long().cumsum(-1) - 1
            pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
            with torch.inference_mode():
                last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
            rows_logits = last[:, letter_ids].float().cpu()
            assert torch.isfinite(rows_logits).all(), "non-finite choice-letter logits"
            preds = rows_logits.argmax(dim=-1).tolist()
            for r, p, cl in zip(batch, preds, rows_logits.tolist()):
                choice_logits.append(cl)
                correct_idx.append(int(r["answer"]))
                per_item.append(int(p == int(r["answer"])))
                per_item_hash.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])

    tokenizer.padding_side = orig_side
    n_total = len(per_item)
    acc = sum(per_item) / n_total
    return {
        "task": "MMLU", "metric": "5-shot choice-letter logit capture",
        "acc": acc, "acc_pct": round(acc * 100, 2),
        "n": n_total, "n_shots": n_shots, "n_per_subject": n_per_subject,
        "choice_logits": choice_logits, "correct_idx": correct_idx,
        "per_item": per_item, "per_item_hash": per_item_hash,
        "letter_token_ids": letter_ids,
    }


# ===========================================================================
# Perplexity — wikitext-2-raw, non-overlapping windows (secondary fluency read).
# ===========================================================================
def clean_text_perplexity(model, tokenizer, device, window: int = 2048,
                          max_tokens: int | None = 120_000) -> dict:
    """Per-token cross-entropy perplexity on wikitext-2-raw test, non-overlapping
    windows (each scored without prior context → a mild over-estimate vs sliding,
    fine for a fluency sanity / relative base-vs-PG2 comparison). No published
    target → preflight gates only on finite + reasonable.

    BOS: every window is prefixed with `tokenizer.bos_token_id` and only the
    non-BOS targets are scored. Gemma is trained with a BOS and scoring raw
    windows without one destabilises the forward — in ERLANGEN c6b the no-BOS
    protocol made a matched-RANDOM ablation *improve* gemma-2b's perplexity by
    4.7 (an instrument statement, not a finding) and put base ppl at 51.8 where
    the BOS-prefixed protocol gives 22.222 for PaliGemma-1 and 11.790 for
    gemma-2b (the house wikitext anchor is ≈ 22.4). That execution carries a
    VOID-INSTRUMENT-SUPERSEDED tag for exactly this reason
    (RESEARCH_INDEX.md:133). This function mirrors
    `infra/aws_pg/jobs/erlangen_c6b.py:92-109`; any value banked before
    2026-09-03 came from the no-BOS protocol and its `ppl` is NOT comparable
    to values produced here (the returned `protocol` field names which one ran).
    """
    import torch
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]

    bos = tokenizer.bos_token_id
    assert bos is not None, "tokenizer has no BOS — the text forward would be mis-specified"

    windows = [ids[i:i + window] for i in range(0, ids.size(0), window)]
    windows = [w for w in windows if w.numel() >= 8]

    nll_sum = 0.0
    n_tok = 0
    per_window: list = []   # [nll_sum, n_tok] per window — for a token-weighted bootstrap of ppl
    for w in windows:
        inp = torch.cat([torch.tensor([bos]), w]).unsqueeze(0).to(device)
        with torch.inference_mode():
            lg = model(input_ids=inp).logits[0, :-1, :].float()
        tg = inp[0, 1:]
        loss = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
        per_window.append([float(loss), int(tg.numel())])
        nll_sum += float(loss)
        n_tok += int(tg.numel())

    ppl = math.exp(nll_sum / n_tok)
    return {
        "task": "wikitext-2-raw perplexity", "metric": "per-token CE perplexity",
        "ppl": ppl, "mean_nll": nll_sum / n_tok, "n_tokens": n_tok, "window": window,
        "per_window": per_window,
        "protocol": "bos-prefixed windows (ERLANGEN c6b)",
        "published": None, "note": "no published 2B base ppl; recompute self-consistent",
    }


def _shell_text_block(tok, text, in_ids):
    """(pos, block_ids): where the window's own text sits inside the shell's REAL input_ids.

    A VLM processor wraps text as `[image tokens .. <bos> .. text .. "\\n"]`, so the text is
    neither the head nor the exact tail; a plain text shell wraps it as `[<bos> .. text]` with no
    trailing newline. Worse, whether the appended newline is a token of its own or MERGES into the
    window's last token depends on that window's final character — the trap that killed M-D45 and
    M-D51 (`invariant_core_func._processor_text_ids`). So both candidate encodings are checked
    against the shell's actual ids, newline-appended first (the longer, more specific block), and
    the one that is present wins. Absent both, FAIL LOUD — a silent misalignment here would score
    the wrong targets and look like a fluency result."""
    import torch
    hay = in_ids.cpu()
    for cand in (tok(text + "\n", return_tensors="pt", add_special_tokens=False).input_ids[0],
                 tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]):
        n = int(cand.shape[0])
        for i in range(hay.shape[0] - n, -1, -1):
            if torch.equal(hay[i:i + n], cand):
                return i, cand
    raise AssertionError(f"window text not found in the shell's input_ids (len {int(hay.shape[0])})")


def shell_text_perplexity(fwd, window: int = 2048, max_tokens: int | None = 120_000) -> dict:
    """Clean-text perplexity through a `matrix_forward.Forward` — i.e. the member's OWN shell.

    WHY THIS EXISTS (2026-09-03). `clean_text_perplexity` calls `model(input_ids=...)` directly.
    On an install row the member is a language decoder living inside a VLM shell, and that call
    passes NO pixel_values: a text-only forward of a model that is only ever run with an image
    block present. It is neither the §6.5 blank-image condition every other column of the row uses
    nor a bare text model — the worst of both. Measured on MENDELEEV row 2's `before` member it
    read **18558.6** where the same decoder reads **61.28** bare (7.966 once BOS-prefixed): 300x,
    an instrument statement, not a fluency number. The real PaliGemma-2 through the same broken
    forward read 42.8, so the defect lands far harder on the installed decoder than on the shell's
    own — the ratio inverts and the one-sided §1.2 flag stays off.

    Here every window goes through `fwd._encode` / `fwd._call_kwargs`, EXACTLY the call every other
    column of that row makes: blank gray-128 image + processor framing on the VLM shells, plain
    tokenizer on the text shell. BOS is inserted by the shell's own processor/tokenizer
    (`add_special_tokens=True`), so the ERLANGEN c6b BOS correction holds by construction. Targets
    are the window's own tokens, located as an exact block inside the shell's input_ids
    (`_ppl_blankimg` semantics generalised to any shell): the image prefix is never a target, and
    the first text token is scored from the prefix that precedes it. On VLM shells the block
    carries the processor's appended newline, so ONE extra target per window (≤ 0.05 % at
    window 2048) is that newline — identical for both arms of a row, which is what parity needs.

    NO-START-TOKEN SHELLS (2026-09-03, row 3a). Not every tokenizer prepends one. Qwen3's does not,
    so on a Qwen text shell the block starts at position 0 and the first target has NO preceding
    token to be predicted from — the case the old `assert pos >= 1` died on (row 3a's ppl cell,
    `AssertionError: text block starts at position 0`; row 3b's before-member would have followed).
    There is no honest logit for that token, so it is DROPPED rather than scored against a padding
    or a wrapped-around position: `off = 1 if pos == 0 else 0`, targets `block[off:]`, logits from
    `pos + off - 1`. It costs ONE token per window (≤ 0.05 % at window 2048) and is applied by the
    same rule to both arms of a row — the arms share a tokenizer, so §1.2 parity is untouched. The
    returned record carries `dropped_leading_target` / `n_dropped_leading_targets` so an artifact
    can never hide it. Every OTHER misalignment still fails loud (`_shell_text_block` raises when
    the block is absent; the length assert below stays).

    Not comparable to any `clean_text_perplexity` number (different forward, different framing);
    the returned `forward` field names which shell produced it."""
    import torch
    from datasets import load_dataset

    tok = fwd.tok
    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]

    nll_sum = 0.0
    n_tok = 0
    n_dropped = 0
    per_window: list = []
    win_samples: list = []
    for start in range(0, ids.size(0), window):
        chunk = ids[start:start + window]
        if chunk.numel() < 8:
            break
        win_text = tok.decode(chunk.tolist())
        enc = fwd._encode([win_text])                                       # noqa: SLF001 — the shell's own call
        in_ids = enc["input_ids"][0]
        pos, block = _shell_text_block(tok, win_text, in_ids)
        off = 1 if pos == 0 else 0          # no-start-token shell: the first target is unscorable
        n_dropped += off
        n_text = int(block.shape[0])
        assert n_text - off >= 1, f"window has no scorable target (n_text={n_text}, off={off})"
        with torch.inference_mode():
            logits = fwd.model(**fwd._call_kwargs(enc)).logits[0]           # noqa: SLF001
        lg = logits[pos + off - 1:pos + n_text - 1, :].float()
        tg = block[off:].to(lg.device)
        assert lg.shape[0] == tg.shape[0], (lg.shape, tg.shape)
        loss = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
        per_window.append([float(loss), int(tg.numel())])
        nll_sum += float(loss)
        n_tok += int(tg.numel())
        # §12.11 sample capture. ppl banked a number with nothing behind it — no text, no prediction,
        # nothing a reader could look at to check the number means what its name says. Everything a
        # sample needs is already computed here: the actual window the model saw, its actual next-token
        # distribution at the last scored position, the token that actually came next, and this window's
        # own per-token NLL. Kept per window (tail only); the three that are shown are chosen after the
        # loop, best and worst by NLL.
        lp = torch.log_softmax(lg[-1], -1)
        top = int(lp.argmax())
        win_samples.append({"item_index": len(per_window) - 1,
                            "prompt_tail": win_text[-700:], "prompt_chars": len(win_text),
                            "n_tokens": int(tg.numel()), "nll_per_token": float(loss) / int(tg.numel()),
                            "model_top_token_id": top, "model_top_token_p": float(lp[top].exp()),
                            "model_top_token_text": tok.decode([top]),
                            "predicted": tok.decode([top]), "gold": tok.decode([int(tg[-1])]),
                            "gold_token_id": int(tg[-1]), "scored_correct": None})

    assert n_tok > 0, "perplexity saw no tokens"
    order = sorted(range(len(win_samples)), key=lambda i: win_samples[i]["nll_per_token"])
    picks, seen = [], set()
    for i, tag in ([(order[0], "best-scoring (lowest per-token NLL)"),
                    (order[-1], "worst-scoring (highest per-token NLL)")]
                   + [(i, "fill (next in order)") for i in range(len(win_samples))]):
        if i in seen:
            continue
        seen.add(i)
        picks.append({**win_samples[i], "selection": tag, "score_kind": "per-token NLL (not right/wrong)",
                      "score_label": f"per-token NLL {win_samples[i]['nll_per_token']:.4f} "
                                     f"(window perplexity {math.exp(win_samples[i]['nll_per_token']):.2f}) "
                                     f"over {win_samples[i]['n_tokens']} tokens"})
        if len(picks) >= 3:
            break
    return {
        "task": "wikitext-2-raw perplexity", "metric": "per-token CE perplexity",
        "samples": picks,
        "ppl": math.exp(nll_sum / n_tok), "mean_nll": nll_sum / n_tok, "n_tokens": n_tok,
        "window": window, "n_windows": len(per_window), "per_window": per_window,
        "dropped_leading_target": bool(n_dropped), "n_dropped_leading_targets": n_dropped,
        "protocol": "shell-framed windows, BOS from the shell's own encoder (ERLANGEN c6b)",
        "forward": f"{fwd.shell} shell" + ("" if fwd.shell == "text" else ", blank gray image present (§6.5)"),
        "published": None, "note": "shell-framed; NOT comparable to clean_text_perplexity values",
    }


# ===========================================================================
# Tokenizer parity — shared-vocab encoding must be identical (§8 threat).
# ===========================================================================
def tokenizer_parity(tok_a, tok_b, probes=PARITY_PROBES) -> dict:
    """Assert two tokenizers encode the clean-text probes to identical ids
    (add_special_tokens=False, so BOS/multimodal specials don't confound). PG2's
    extended vocab only adds image/loc/seg rows; on shared text the ids must match."""
    rows = []
    all_ok = True
    for p in probes:
        ia = tok_a(p, add_special_tokens=False).input_ids
        ib = tok_b(p, add_special_tokens=False).input_ids
        ok = (ia == ib)
        all_ok = all_ok and ok
        rows.append({"text": p, "ids_a": ia, "ids_b": ib, "match": ok})
    return {"match": bool(all_ok), "n": len(probes), "probes": rows}


# ===========================================================================
# WMDP-bio — choice-logit capture (W18 read-side addition; 0-shot lm-eval
# template; mechanical sibling of mmlu_choice_logits — MMLU functions UNTOUCHED).
# ===========================================================================
WMDP_REPO = "cais/wmdp"


def wmdp_choice_logits(model, tokenizer, device, config: str = "wmdp-bio",
                       batch_size: int = 4, item_filter=None) -> dict:
    """0-shot WMDP letter-logit capture (lm-eval wmdp convention: question +
    lettered choices + "Answer:"; single-token ' A'..' D' read). Same hashing
    (sha1(question[:64])[:12]), same left-pad + content-relative position_ids +
    logits_to_keep=1, same item_filter semantics as mmlu_choice_logits. Items
    in dataset order (deterministic load; no subjects)."""
    import torch
    from datasets import load_dataset

    ds = load_dataset(WMDP_REPO, config, split="test")

    from infra.pg.relearn import letter_token_ids
    letter_ids = letter_token_ids(tokenizer)   # Gemma fast path / Mistral ▁-fallback

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    rows = [dict(r) for r in ds]
    if item_filter is not None:
        rows = [r for r in rows if item_filter(
            hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])]

    choice_logits, correct_idx, per_item, per_item_hash = [], [], [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [_q_block(r["question"], r["choices"]) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=True).to(device)
        pos = enc["attention_mask"].long().cumsum(-1) - 1
        pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
        with torch.inference_mode():
            last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
        rl = last[:, letter_ids].float().cpu()
        assert torch.isfinite(rl).all(), "non-finite choice-letter logits"
        preds = rl.argmax(dim=-1).tolist()
        for r, pr, cl in zip(batch, preds, rl.tolist()):
            choice_logits.append(cl)
            correct_idx.append(int(r["answer"]))
            per_item.append(int(pr == int(r["answer"])))
            per_item_hash.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])

    tokenizer.padding_side = orig_side
    n_total = len(per_item)
    assert n_total > 0, "wmdp eval produced no items"
    return {"task": f"WMDP::{config}", "metric": "0-shot single-token-answer (letter-logit capture)",
            "acc": sum(per_item) / n_total, "acc_pct": round(100 * sum(per_item) / n_total, 2),
            "n": n_total, "choice_logits": choice_logits, "correct_idx": correct_idx,
            "per_item": per_item, "per_item_hash": per_item_hash,
            "letter_token_ids": letter_ids}


# ===========================================================================
# MMLU — GEN-cloze (answer-CONTENT axis; REPRICE's only NEW measurement).
#   Same subject header + 5 dev exemplars + question as `mmlu_accuracy`, but the
#   'A./B./C./D.' choice scaffold and the choice-letter read are GONE. For each
#   of the 4 choice TEXTS, score Σ log p(' '+choice_text | prompt). The single
#   variable changed vs the MCQ axis: letter-readout → answer-content. Item order
#   + per_item_hash recipe are IDENTICAL to `mmlu_accuracy` → paired per item.
# ===========================================================================
def _q_block_cloze(question: str, answer_text: str | None = None) -> str:
    """One MMLU item in CLOZE format: the question, then 'Answer:' followed by the
    answer TEXT (few-shot exemplars) or ending exactly at 'Answer:' (scored item).
    No lettered choice list, no choice letter."""
    s = question.strip() + "\nAnswer:"
    if answer_text is not None:
        s += f" {answer_text.strip()}\n\n"
    return s


def _common_prefix_len(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def _score_cloze_window(logprob_window, cont_ids_rows, ntok_rows, trailing=0):
    """Pure gather (unit-tested weight-free). `logprob_window` is (B, K, V) — the
    next-token log-probs at the LAST K decode positions of a LEFT-padded batch
    whose every row ends at a fixed `trailing`-token separator (the PaliGemma
    processor appends one '\\n'; text-only has none → trailing=0). Row r's
    continuation has `ntok_rows[r]` tokens sitting just BEFORE that separator; the
    logits PREDICTING them are at window indices [end-ntok : end] with
    end = K-1-trailing. Returns (B,) summed continuation log-prob (float64)."""
    import torch
    B, K, _ = logprob_window.shape
    end = K - 1 - trailing
    out = torch.empty(B, dtype=torch.float64)
    for r in range(B):
        m = int(ntok_rows[r])
        assert 1 <= m and end - m >= 0, f"row {r}: ntok {m}, trailing {trailing}, K {K}"
        pred = logprob_window[r, end - m:end, :]              # (m, V): predicts cont tokens
        tgt = torch.as_tensor(cont_ids_rows[r][:m], dtype=torch.long, device=pred.device)
        out[r] = pred[torch.arange(m), tgt].sum().double()
    return out


def mmlu_cloze(model, tokenizer, device, *, processor=None, n_shots: int = 5,
               n_per_subject: int | None = None, gray: int = 128,
               item_filter=None) -> dict:
    """The content axis. Per item, score the 4 choice TEXTS as continuations of
    `header + 5 cloze exemplars + question + 'Answer:'` (choices NOT shown). Banks
    per candidate (sum log-prob, n_tok) so any length-normalization
    (logprob / ntok**λ) is a pure downstream function; the banked default
    `per_item` uses λ=1 (per-token mean). Deterministic, no decode, no grader.

    forward: text-only when `processor is None` (uses `tokenizer`); blank-image
    (PaliGemma in-distribution, uniform gray) when `processor` is given — mirrors
    `mmlu_blankimg` (pixel_values, no manual position_ids on the image path).
    `logits_to_keep=K` caps lm_head to the continuation window (no (B,S,256k)
    materialization). Continuation length derived from real-token counts (no
    boundary-tokenization assumption; prompt-prefix stability asserted, fail-loud).
    """
    import torch
    from collections import defaultdict
    from datasets import load_dataset

    tok = processor.tokenizer if processor is not None else tokenizer
    blank = None
    if processor is not None:
        import numpy as np
        from PIL import Image
        blank = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))

    def encode(texts):
        if processor is not None:
            return processor(text=texts, images=[blank] * len(texts),
                             return_tensors="pt", padding="longest").to(device)
        return tok(texts, return_tensors="pt", padding=True,
                   add_special_tokens=True).to(device)

    ds = load_dataset(MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]
    shots_by_subj: dict = defaultdict(list)
    for r in dev:
        shots_by_subj[r["subject"]].append(r)
    test_by_subj: dict = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"

    cloze_logprobs: list = []   # (N,4) Σ log p(choice_text), CHOICES order
    cloze_ntoks: list = []      # (N,4) continuation token counts
    correct_idx: list = []
    per_item: list = []         # λ=1 argmax == answer
    per_item_hash: list = []
    n_subjects = 0
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:n_shots]
        assert len(shots) == n_shots, f"{subj}: only {len(shots)} dev shots (<{n_shots})"
        prefix = _subject_header(subj) + "".join(
            _q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        if item_filter is not None:
            rows = [r for r in rows if item_filter(
                hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])]
        if rows:
            n_subjects += 1
        for r in rows:
            prompt = prefix + _q_block_cloze(r["question"])
            choices = [c.strip() for c in r["choices"]]
            assert len(choices) == 4, f"MMLU item must have 4 choices, got {len(choices)}"
            pe = encode([prompt])
            prompt_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
            enc = encode([prompt + " " + c for c in choices])
            ids, am = enc["input_ids"], enc["attention_mask"]
            real = am.sum(dim=1).tolist()
            # Locate each continuation by longest common prefix/suffix vs the bare prompt
            # — robust to the processor's framing (PaliGemma appends a trailing '\n';
            # text-only appends nothing). prompt = [shared|sep]; full = [shared|cont|sep].
            cont_ids, ntok = [], []
            lcp = lcs = None
            for k in range(4):
                full_real = ids[k][ids.shape[1] - int(real[k]):].tolist()
                p = _common_prefix_len(prompt_real, full_real)
                s = _common_suffix_len(prompt_real, full_real)
                assert p + s == len(prompt_real), \
                    f"cloze prompt not [shared|sep] ({subj}, choice {k}): lcp={p} lcs={s} |p|={len(prompt_real)}"
                cont = full_real[p:len(full_real) - s]
                assert len(cont) >= 1, f"empty cloze continuation ({subj}, choice {k})"
                if lcp is None:
                    lcp, lcs = p, s
                else:
                    assert (p, s) == (lcp, lcs), f"inconsistent cloze framing across candidates ({subj})"
                cont_ids.append(cont)
                ntok.append(len(cont))
            K = max(ntok) + lcs + 1
            with torch.inference_mode():
                if processor is not None:
                    out = model(input_ids=ids, attention_mask=am,
                                pixel_values=enc["pixel_values"].to(torch.bfloat16),
                                logits_to_keep=K)
                else:
                    pos = am.long().cumsum(-1) - 1
                    pos = pos.masked_fill(am == 0, 1)
                    out = model(input_ids=ids, attention_mask=am, position_ids=pos,
                                logits_to_keep=K)
                lw = torch.log_softmax(out.logits.float(), dim=-1)   # (4, K, V)
            sums = _score_cloze_window(lw, cont_ids, ntok, trailing=lcs)   # (4,)
            assert torch.isfinite(sums).all(), f"non-finite cloze logprob ({subj})"
            cloze_logprobs.append([float(x) for x in sums.tolist()])
            cloze_ntoks.append([int(m) for m in ntok])
            ci = int(r["answer"])
            correct_idx.append(ci)
            mean_lp = sums / torch.tensor(ntok, dtype=torch.float64)
            per_item.append(int(int(mean_lp.argmax()) == ci))
            per_item_hash.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])

    tok.padding_side = orig_side
    n_total = len(per_item)
    assert n_total > 0, "mmlu_cloze produced no items"
    return {
        "task": "MMLU", "metric": "5-shot GEN-cloze answer-content (length-normed λ=1 argmax)",
        "forward": "blank-image" if processor is not None else "text-only",
        "gray": gray if processor is not None else None,
        "acc": sum(per_item) / n_total, "acc_pct": round(100 * sum(per_item) / n_total, 2),
        "n": n_total, "n_subjects": n_subjects, "n_shots": n_shots,
        "n_per_subject": n_per_subject,
        "cloze_logprobs": cloze_logprobs, "cloze_ntoks": cloze_ntoks,
        "correct_idx": correct_idx,
        "per_item": per_item, "per_item_hash": per_item_hash,
    }


# ===========================================================================
# RE-EXPRESS — production readout (NEW; prereg §2 keystone + secondary).
#
#   The four position-paired axes of ARM G, all in the SAME blank-image gray=128
#   forward, same (sorted-subject, row) order + per_item_hash recipe as
#   mmlu_accuracy/mmlu_blankimg/mmlu_cloze:
#     (M) MCQ-letter   — mmlu_blankimg / mmlu_choice_logits (REUSED): rank 4 letters.
#     (Z) cloze        — mmlu_cloze (PORT): rank the 4 candidate TEXTS.
#     (P) gen2opt      — mmlu_gen2opt (NEW, KEYSTONE): GENERATE, then map the
#                        generation to one of the 4 candidate texts (_match_to_option).
#                        Same cloze prompt as (Z) → differs from (Z) ONLY
#                        production-vs-ranking; same 4-option construct as (M)/(Z).
#     (S) strict-EM    — mmlu_freegen (NEW, secondary): the SAME generation graded by
#                        open-vocab normalized strict-EM vs gold. One decode → P and S
#                        paired per item, removing the production≫recognition confound.
#
#   The blank-image greedy decode loop is NEW (only _boot_ci is shared from
#   e4_capability; e4's _generate needs a REAL image and is NOT reused). The pure
#   graders (_normalize_answer / _match_to_option / _free_gen_em) are weight-free and
#   unit-tested; the _match_to_option heuristic is tuned on DISCOVERY only (§2), so
#   what is frozen here is the no-match/tie CONTRACT, not a final threshold.
# ===========================================================================
def _normalize_answer(s: str) -> str:
    """SQuAD-style normalization shared by the gen2opt matcher and the strict-EM
    axis: lowercase → drop punctuation → drop articles (a/an/the) → collapse
    whitespace. Deterministic; the single shared normalizer keeps (P) and (S)
    construct-comparable."""
    import re
    import string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _match_to_option(decoded, options):
    """Map a free generation to one of the supplied candidate strings by DIRECTIONAL
    whole-word containment: a candidate matches iff its normalized text occurs as a
    contiguous word-span of the normalized generation. The containment SCORE is the
    candidate's word length (a longer contained candidate is the more specific match,
    which resolves nested candidates like 'cell' ⊂ 'cell wall'). Returns
    (matched_idx, flag):

      'ok'        — exactly one top-scoring candidate → matched_idx in 0..len-1.
      'unmatched' — max score 0 (no candidate contained — e.g. a bare letter or a
                    no-overlap paraphrase) → matched_idx None, scored INCORRECT (§2).
      'tie'       — ≥2 candidates share the max score → matched_idx None, INCORRECT.

    Directional (candidate ⊆ generation), so format-only output (e.g. 'A') matches no
    CONTENT candidate → 'unmatched' — exactly the §2 no-match policy. The unmatched
    rate is banked as the §2 U_max diagnostic; the heuristic is tuned on DISCOVERY."""
    g = _normalize_answer(decoded)
    if not g:
        return (None, "unmatched")
    gpad = f" {g} "
    scores = []
    for o in options:
        no = _normalize_answer(o)
        contained = bool(no) and (f" {no} " in gpad)
        scores.append(len(no.split()) if contained else 0)
    top = max(scores)
    if top == 0:
        return (None, "unmatched")
    winners = [k for k, sc in enumerate(scores) if sc == top]
    if len(winners) > 1:
        return (None, "tie")
    return (winners[0], "ok")


def _free_gen_em(decoded, gold) -> int:
    """Open-vocab strict EM (axis S): 1 iff the generation and the gold answer agree
    after _normalize_answer. No candidate set — the labelled secondary axis."""
    return int(_normalize_answer(decoded) == _normalize_answer(gold))


def mmlu_freegen(decoded, golds) -> dict:
    """Secondary axis (S) as a PURE re-grade over banked generations: per-item
    strict normalized-EM vs gold. Weight-free (no model); mmlu_gen2opt calls this on
    its own decode so P and S are paired per item by construction."""
    assert len(decoded) == len(golds), f"decoded/golds misaligned: {len(decoded)} vs {len(golds)}"
    per_item = [_free_gen_em(d, g) for d, g in zip(decoded, golds)]
    n = len(per_item)
    assert n > 0, "mmlu_freegen over empty set"
    return {"task": "MMLU", "metric": "open-vocab strict normalized-EM (axis S, secondary)",
            "acc": sum(per_item) / n, "acc_pct": round(100 * sum(per_item) / n, 2),
            "n": n, "per_item": per_item}


def _blank_image_generate(model, processor, device, prompts, gray: int = 128,
                          max_new_tokens: int = 16, batch_size: int = 4) -> list:
    """NEW blank-image greedy decode loop (the (P)/(S) production read). Uniform gray
    224² image (parity with mmlu_blankimg), LEFT-pad the batch, generate
    do_sample=False, decode ONLY the new tokens (gen[:, in_len:]), truncate at the
    first newline. Returns decoded strings aligned 1:1 to `prompts`. Only `_boot_ci`
    is shared from e4_capability; e4's `_generate` requires a real image (RefCOCO box
    decode) and is deliberately not reused (prereg §2/§12)."""
    import numpy as np
    import torch
    from PIL import Image

    tok = processor.tokenizer
    blank = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"

    out: list = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inp = processor(text=chunk, images=[blank] * len(chunk),
                        return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        in_len = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        dec = tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)
        out.extend(d.split("\n")[0].strip() for d in dec)

    tok.padding_side = orig_side
    assert len(out) == len(prompts), f"decode count {len(out)} != prompts {len(prompts)}"
    return out


def mmlu_gen2opt(model, processor, device, n_shots: int = 5,
                 n_per_subject: int | None = None, gray: int = 128,
                 max_new_tokens: int = 16, batch_size: int = 4,
                 item_filter=None, decoded_sample: int = 64) -> dict:
    """KEYSTONE (axis P). Per item, build the SAME 5-shot cloze prompt as `mmlu_cloze`
    (header + cloze exemplars + question + 'Answer:'), GENERATE freely on the
    blank-image forward, then map the generation to one of the 4 candidate texts via
    `_match_to_option` (correct iff the matched candidate is the gold answer; no-match
    and tie score INCORRECT, §2). Differs from cloze (Z) ONLY production-vs-ranking.

    Banks the open-vocab strict-EM axis (S) from the SAME decode via `mmlu_freegen`
    (one generation → P and S paired per item). Item order + per_item_hash recipe are
    IDENTICAL to mmlu_accuracy/mmlu_blankimg/mmlu_cloze, so all four ARM-G axes are
    position-paired. `item_filter` (hash predicate) selects the discovery / confirmation
    split before the forward, preserving order."""
    import hashlib
    from collections import defaultdict
    from datasets import load_dataset

    ds = load_dataset(MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]
    shots_by_subj: dict = defaultdict(list)
    for r in dev:
        shots_by_subj[r["subject"]].append(r)
    test_by_subj: dict = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)

    prompts: list = []
    options_list: list = []
    answers: list = []
    golds: list = []
    hashes: list = []
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:n_shots]
        assert len(shots) == n_shots, f"{subj}: only {len(shots)} dev shots (<{n_shots})"
        prefix = _subject_header(subj) + "".join(
            _q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        if item_filter is not None:
            rows = [r for r in rows if item_filter(
                hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])]
        for r in rows:
            choices = [str(c).strip() for c in r["choices"]]
            assert len(choices) == 4, f"MMLU item must have 4 choices, got {len(choices)}"
            ai = int(r["answer"])
            prompts.append(prefix + _q_block_cloze(r["question"]))
            options_list.append(choices)
            answers.append(ai)
            golds.append(choices[ai])
            hashes.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])
    assert prompts, "mmlu_gen2opt produced no items"

    decoded = _blank_image_generate(model, processor, device, prompts, gray=gray,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)

    per_item: list = []
    flags: list = []
    matched_idx: list = []
    for dec, opts, ans in zip(decoded, options_list, answers):
        idx, flag = _match_to_option(dec, opts)
        per_item.append(int(idx == ans))   # idx None (unmatched/tie) → 0 (incorrect)
        flags.append(flag)
        matched_idx.append(idx)

    n = len(per_item)
    unmatched = sum(1 for f in flags if f == "unmatched")
    ties = sum(1 for f in flags if f == "tie")
    free = mmlu_freegen(decoded, golds)   # axis S, same decode, position-paired

    return {
        "task": "MMLU", "metric": "gen-into-4-options (axis P, keystone)",
        "forward": "blank-image", "gray": gray, "max_new_tokens": max_new_tokens,
        "acc": sum(per_item) / n, "acc_pct": round(100 * sum(per_item) / n, 2),
        "n": n, "n_shots": n_shots, "n_per_subject": n_per_subject,
        "per_item": per_item, "per_item_hash": hashes,
        "matched_idx": matched_idx, "match_flag": flags, "answers": answers,
        "unmatched_rate": unmatched / n, "tie_rate": ties / n,
        # secondary axis (S), same decode, position-paired per item:
        "freegen_per_item": free["per_item"], "freegen_acc": free["acc"],
        # decoded_raw sample for the U_max audit / journal (not the full transcript):
        "decoded_sample": decoded[:decoded_sample],
    }
