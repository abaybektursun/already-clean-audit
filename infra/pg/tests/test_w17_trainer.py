"""Weight-free CPU tests for the W17 trainer logic (relearn.train_adapter_qa & co).
NO real model/tokenizer/GPU — a stub LM + stub tokenizer prove the LOOP: loss
masking (single letter-token CE), seed determinism, max_steps cap, schedule
wiring, token matching. Part of the W17 §9.1 offline dry-run gate.

    python -m pytest infra/pg/tests/test_w17_trainer.py -q
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from infra.pg import relearn as R


# ---------------------------------------------------------------------------
# Stubs: char-level tokenizer (letters → fixed ids) + a 1-layer "LM".
# ---------------------------------------------------------------------------
class StubTok:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"
    VOCAB = 128

    def encode(self, text, add_special_tokens=False):
        if text in (" A", " B", " C", " D"):          # Gemma '▁A'-style single token
            return [65 + " ABCD".index(text[1]) - 1]
        return [min(127, max(1, ord(c))) for c in text][:32] or [1]

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=True):
        seqs = [self.encode(t) for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        am = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            if self.padding_side == "left":
                ids[i, n - len(s):] = torch.tensor(s); am[i, n - len(s):] = 1
            else:
                ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1

        class Enc(dict):
            def to(self, device):
                return self
        return Enc(input_ids=ids, attention_mask=am)


class StubLM(nn.Module):
    """Logits = bias only (trainable) — CE is exactly fittable to any constant
    target distribution; proves the loop optimizes the right token."""
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(StubTok.VOCAB))

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                logits_to_keep=None, **kw):
        B = input_ids.shape[0]
        out = type("O", (), {})()
        out.logits = self.bias.expand(B, 1, StubTok.VOCAB)
        return out


def _examples(n, answer_idx=0):
    return [R.format_qa(f"Stub question {i}?", ["w", "x", "y", "z"], answer_idx)
            for i in range(n)]


# ---------------------------------------------------------------------------
def test_letter_ids_distinct_and_single():
    assert R.letter_token_ids(StubTok()) == [65, 66, 67, 68]


def test_letter_ids_mistral_fallback():
    # Mistral/Llama SP splits the leading space: encode(' A') → 2 tokens; the
    # resolver must fall back to the explicit '▁A' vocab entry (W18 incident).
    class MistralStub(StubTok):
        VOCAB_MAP = {"\u2581A": 330, "\u2581B": 365, "\u2581C": 334, "\u2581D": 384}
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text in (" A", " B", " C", " D"):
                return [29871, 65 + " ABCD".index(text[1]) - 1]   # split: ▁ + letter
            return super().encode(text, add_special_tokens)

        def convert_tokens_to_ids(self, tok):
            return self.VOCAB_MAP.get(tok, self.unk_token_id)

    assert R.letter_token_ids(MistralStub()) == [330, 365, 334, 384]

    class BrokenStub(MistralStub):
        VOCAB_MAP = {}
    import pytest as _pt
    with _pt.raises(AssertionError):
        R.letter_token_ids(BrokenStub())


def test_batches_partition():
    b = R.qa_train_step_batches(list(range(10)), 4)
    assert b == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(AssertionError):
        R.qa_train_step_batches([], 0)


def test_trainer_fits_constant_target_and_is_deterministic():
    tok = StubTok()
    ids = R.letter_token_ids(tok)
    assert len(set(ids)) == 4, "stub must give distinct letter ids"
    ex = _examples(24, answer_idx=2)
    runs = []
    for _ in range(2):
        m = StubLM()
        rep = R.train_adapter_qa(m, StubTok(), "cpu", ex, lr=0.5, epochs=6,
                                 batch_size=8, warmup_ratio=0.1, seed=7)
        runs.append((rep["final_loss"], rep["steps"], rep["tokens_seen"]))
        assert rep["final_loss"] < 0.1, f"constant-target CE must fit, got {rep['final_loss']}"
        assert rep["steps"] == 6 * 3 and rep["tokens_seen"] > 0
        # the loop optimized THE letter token: bias argmax == letter id of answer 2
        assert int(m.bias.argmax()) == ids[2]
    assert runs[0] == runs[1], "trainer not deterministic in seed"


def test_trainer_max_steps_cap_and_padding_side_restored():
    tok = StubTok()
    rep = R.train_adapter_qa(StubLM(), tok, "cpu", _examples(24), lr=0.1, epochs=999,
                             batch_size=8, warmup_ratio=0.0, seed=1, max_steps=5)
    assert rep["steps"] == 5
    assert tok.padding_side == "right"          # restored


def test_trainer_requires_trainable_params():
    m = StubLM()
    m.bias.requires_grad_(False)
    with pytest.raises(AssertionError):
        R.train_adapter_qa(m, StubTok(), "cpu", _examples(8), lr=0.1, epochs=1,
                           batch_size=4, warmup_ratio=0.0, seed=0)


def test_trainer_micro_batch_math_equivalent():
    # accumulation with equal micro-chunks == full-batch step exactly
    ex = _examples(16, answer_idx=1)
    runs = []
    for mb in (None, 2):
        m = StubLM()
        rep = R.train_adapter_qa(m, StubTok(), "cpu", ex, lr=0.3, epochs=3,
                                 batch_size=4, warmup_ratio=0.1, seed=5, micro_batch=mb)
        runs.append((rep["final_loss"], m.bias.detach().clone()))
    assert abs(runs[0][0] - runs[1][0]) < 1e-6
    assert torch.allclose(runs[0][1], runs[1][1], atol=1e-6)


def test_retain_token_matching():
    tok = StubTok()
    texts = ["aaaa", "bbbbbb", "cc", "dddddddd"]      # 4+6+2+8 char-tokens
    out = R.retain_token_matched_slice(texts, tok, token_budget=11)
    assert out["n_texts"] == 3 and out["n_tokens"] == 12      # 4+6 <11 → +2 = 12
    with pytest.raises(AssertionError):
        R.retain_token_matched_slice(["ab"], tok, token_budget=99)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
