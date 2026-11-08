"""W14-R OFFLINE INTEGRATION GATE — everything that needs real data/tokenizers but NOT a GPU.

This is the harness that must go green BEFORE any pg launch. It exercises the full pipeline minus
the model forward, on the REAL cais/mmlu + openai/MMMLU + the real gemma/paligemma tokenizers — the
class of failure that previously surfaced only on GPU (journal C3/C8/C9/C10/C11). The single thing it
canNOT check is "does the model produce 0.517 on before" — that, and only that, is the GPU's job.

Needs `datasets` + `transformers` + a HF token (gated repos). Skips cleanly if absent, so the fast
unit dry-run (`test_recovery_transforms.py`) still runs on a bare interpreter.

Run (in a venv with the deps, with the token):
    HF_TOKEN=hf_... python -m pytest infra/pg/tests/test_recovery_integration.py -v
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")
if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
    pytest.skip("no HF token — gated repos unavailable", allow_module_level=True)

from transformers import AutoTokenizer  # noqa: E402

import infra.pg.recovery_transforms as RT  # noqa: E402
from infra.pg.recovery_eval import load_mmlu  # noqa: E402
from infra.pg.text_eval import _q_block, _subject_header  # noqa: E402

GEMMA = "google/gemma-2-2b"           # the text-scoring tokenizer (ENTITIES['L0'])
PG = "google/paligemma2-3b-pt-224"    # the processor tokenizer (text-only path must NOT use this)
LANGS = tuple(RT.CONJ_LANG.values())


@pytest.fixture(scope="module")
def data():
    # a couple of subjects, enough shots for the translation pool (>= n_shots+1)
    dev, held, ex = load_mmlu(["nutrition", "astronomy"], n_per_subject=12, langs=LANGS)
    return dev, held, ex


@pytest.fixture(scope="module")
def gemma_tok():
    return AutoTokenizer.from_pretrained(GEMMA)


# ── data alignment (C8) ────────────────────────────────────────────────────────────────────────

def test_load_mmlu_alignment(data):
    dev, held, ex = data
    items = dev + held
    assert len(items) > 0
    assert len({it["gidx"] for it in items}) == len(items)          # gidx unique (collision-proof key)
    for it in items:
        assert sorted(it["i18n"].keys()) == sorted(LANGS)           # every kept item has all langs
        assert len(it["options"]) == 4 and 0 <= it["answer"] <= 3
    # the answer cross-check that gates alignment is enforced inside load_mmlu (drop-and-log); here we
    # just assert it produced a sane, non-degenerate set
    assert len(ex) >= 2 and all(len(v) >= 5 for v in ex.values())


# ── the BOS bug (C11) — the one that cost three GPU launches ─────────────────────────────────────

def test_text_tokenizer_prepends_bos(gemma_tok, data):
    dev, _, ex = data
    it = dev[0]
    prompt = RT.apply_transform(it, ("identity", "canonical", None, "5_default"), ex[it["subject"]])
    ids = gemma_tok.encode(prompt, add_special_tokens=True)
    assert ids[0] == gemma_tok.bos_token_id, "text tokenizer must prepend BOS (the decoder needs it)"
    # and the PaliGemma processor tokenizer must NOT — so it must never be used for the text path
    pg_tok = AutoTokenizer.from_pretrained(PG)
    assert pg_tok.encode(prompt, add_special_tokens=True)[0] != pg_tok.bos_token_id


# ── answer surfaces single-token under the real tokenizer (C9) ───────────────────────────────────

def test_all_answer_surfaces_single_token(gemma_tok):
    for t in RT.transform_set():
        for surface in RT.answer_candidates(t):
            toks = gemma_tok.encode(surface, add_special_tokens=False)
            assert len(toks) == 1, f"{t[0]}: answer surface {surface!r} → {len(toks)} tokens (must be 1)"
        # the 4 candidates must be distinct tokens
        ids = [gemma_tok.encode(s, add_special_tokens=False)[0] for s in RT.answer_candidates(t)]
        assert len(set(ids)) == 4, f"{t[0]}: candidate tokens collide {ids}"


# ── every transform renders on real data, deterministically (C3) ─────────────────────────────────

def test_all_transforms_render(data):
    dev, held, ex = data
    it = dev[0]
    same_subj = [x for x in dev + held if x["subject"] == it["subject"] and x["gidx"] != it["gidx"]][:5]
    for t in RT.transform_set():
        shots = same_subj if t[2] in RT.CONJ_LANG else ex[it["subject"]]
        out = RT.apply_transform(it, t, shots)
        assert out and out == RT.apply_transform(it, t, shots)       # non-empty + deterministic
        assert out.rstrip().endswith(RT.FORMATS[t[1]][1])            # ends on the transform's cue


def test_identity_byte_exact_w11(data):
    dev, _, ex = data
    it = dev[0]
    w11 = (_subject_header(it["subject"])
           + "".join(_q_block(e["question"], e["options"], e["answer"]) for e in ex[it["subject"]][:5])
           + _q_block(it["question"], it["options"]))
    assert RT.apply_transform(it, ("identity", "canonical", None, "5_default"), ex[it["subject"]]) == w11


def test_translation_surface(data):
    dev, held, ex = data
    it = dev[0]
    same = [x for x in dev + held if x["subject"] == it["subject"] and x["gidx"] != it["gidx"]][:5]
    out = RT.apply_transform(it, ("f2_zh", "canonical", "f2_zh", "5_default"), same)
    assert it["i18n"]["zh"]["question"][:10] in out                 # translated question present
    assert out.rstrip().endswith("Answer:")                         # canonical English answer surface
