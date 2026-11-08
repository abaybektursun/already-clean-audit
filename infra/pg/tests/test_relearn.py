"""Weight-free CPU unit tests for the relearning-ladder core (infra/pg/relearn.py).
NO model, NO datasets, NO GPU — the W17/W18/SAV-π §9.1 offline dry-run gate.

    python -m pytest infra/pg/tests/test_relearn.py -q

Checklist:
  * QA formatting mirrors the eval _q_block + single-letter target   (test_format_*)
  * hash recipe identical to text_eval per_item_hash                 (test_hash_recipe)
  * dedup audit counts; ladder determinism + nesting; label shuffle
    preserves marginals + is deterministic                           (test_ladder_*, test_shuffle_*)
  * adapter: zero delta at init, only A/B trainable, scope excludes
    embed/lm_head, per-module seeds deterministic, rank clamped      (test_adapter_*)
"""
from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn

from infra.pg import relearn as R
from infra.pg.text_eval import _q_block


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        out.append({"question": f"Question number {i} about thing {i*7%13}?",
                    "choices": [f"opt{j}" for j in range(4)],
                    "answer": int(torch.randint(0, 4, (1,), generator=g))})
    return out


# ---------------------------------------------------------------------------
# Formatting + hash recipe.
# ---------------------------------------------------------------------------
def test_format_matches_eval_qblock():
    r = _rows(1)[0]
    ex = R.format_qa(r["question"], r["choices"], r["answer"])
    assert ex["prompt"] == _q_block(r["question"], r["choices"])          # identical render
    assert ex["target_text"] == f" {R.CHOICES[r['answer']]}"
    assert ex["prompt"].endswith("Answer:")


def test_format_rejects_bad_inputs():
    with pytest.raises(AssertionError):
        R.format_qa("q", ["a", "b", "c"], 0)
    with pytest.raises(AssertionError):
        R.format_qa("q", ["a", "b", "c", "d"], 4)


def test_hash_recipe_identical_to_eval():
    q = "Some question with a moderately long prefix that exceeds sixty-four characters easily."
    assert R.qa_item_hash(q) == hashlib.sha1(q[:64].encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Dedup + ladder + shuffle.
# ---------------------------------------------------------------------------
def test_ladder_dedup_counts():
    rows = _rows(50)
    ev = [R.qa_item_hash(r["question"]) for r in rows[:10]]
    d = R.dedup_against_eval(rows, ev)
    assert d["n_in"] == 50 and d["n_dropped"] == 10 and len(d["kept"]) == 40


def test_ladder_deterministic_and_nested():
    rows = _rows(200)
    s64 = R.ladder_subset(rows, 64)
    s64b = R.ladder_subset(rows, 64)
    s128 = R.ladder_subset(rows, 128)
    assert s64 == s64b                                                    # deterministic
    ids64 = {R.qa_item_hash(r["question"]) for r in s64}
    ids128 = {R.qa_item_hash(r["question"]) for r in s128}
    assert ids64 <= ids128                                                # nested
    with pytest.raises(AssertionError):
        R.ladder_subset(rows, 201)                                        # exceeds pool


def test_ladder_salt_changes_order():
    rows = _rows(100)
    assert R.ladder_subset(rows, 10) != R.ladder_subset(rows, 10, salt="other-salt")


def test_shuffle_preserves_marginals_and_changes_rows():
    rows = _rows(100, seed=3)
    sh = R.shuffle_labels(rows, seed=7)
    assert sorted(r["answer"] for r in sh) == sorted(r["answer"] for r in rows)
    assert any(a["answer"] != b["answer"] for a, b in zip(rows, sh))
    assert sh == R.shuffle_labels(rows, seed=7)                           # deterministic
    assert [r["question"] for r in sh] == [r["question"] for r in rows]   # inputs untouched


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------
class TinyDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(7, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.mlp_up = nn.Linear(8, 16, bias=False)
        self.lm_head = nn.Linear(8, 7, bias=False)


def test_adapter_scope_excludes_embed_lmhead():
    lm = TinyDec()
    assert R.adapter_target_names(lm) == ["mlp_up", "q_proj"]


def test_adapter_zero_delta_at_init_and_trainability():
    lm = TinyDec()
    x = torch.randn(3, 8)
    y0 = lm.q_proj(x)
    rep = R.attach_adapters(lm, rank=4, seed=11)
    assert rep["n_modules"] == 2 and rep["rank"] == 4
    y1 = lm.q_proj(x)
    assert torch.allclose(y0, y1, atol=0), "adapter must start at exactly zero delta (B=0)"
    train_names = [n for n, p in lm.named_parameters() if p.requires_grad]
    assert train_names and all((".A" in n or ".B" in n) for n in train_names)
    # base weights frozen, embed untouched by scope
    assert not lm.q_proj.base.weight.requires_grad
    assert isinstance(lm.lm_head, nn.Linear) and isinstance(lm.embed_tokens, nn.Embedding)


def test_adapter_nonzero_after_B_update_and_param_count():
    lm = TinyDec()
    R.attach_adapters(lm, rank=4, seed=11)
    with torch.no_grad():
        lm.q_proj.B.add_(torch.ones_like(lm.q_proj.B))
    x = torch.randn(3, 8)
    base_only = lm.q_proj.base(x)
    assert not torch.allclose(lm.q_proj(x), base_only)
    n_adapter = sum(p.numel() for n, p in lm.named_parameters() if p.requires_grad)
    assert n_adapter == (8 * 4 + 4 * 8) + (16 * 4 + 4 * 8)               # A+B per module


def test_adapter_seeds_deterministic_per_module():
    a, b = TinyDec(), TinyDec()
    R.attach_adapters(a, rank=4, seed=11)
    R.attach_adapters(b, rank=4, seed=11)
    assert torch.equal(a.q_proj.A, b.q_proj.A)
    c = TinyDec()
    R.attach_adapters(c, rank=4, seed=12)
    assert not torch.equal(a.q_proj.A, c.q_proj.A)
    assert not torch.equal(a.q_proj.A[:, :], a.mlp_up.A[:8, :])           # per-module seeds differ


def test_adapter_rank_clamped():
    lm = TinyDec()
    R.attach_adapters(lm, rank=64, seed=0)
    assert lm.q_proj.rank == 8                                            # min(out,in)=8


def test_adapter_bf16_model_forward_and_train_step():
    # FATAL-1 regression (W17 harness critic): deployed models are bf16; the
    # adapter must forward AND train on a bf16 base without dtype errors,
    # keeping A/B fp32 masters.
    lm = TinyDec().to(torch.bfloat16)
    R.attach_adapters(lm, rank=4, seed=11)
    assert lm.q_proj.A.dtype == torch.float32 and lm.q_proj.B.dtype == torch.float32
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    y = lm.q_proj(x)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, lm.q_proj.base(x))          # zero delta at init, exact
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-2)
    loss = lm.q_proj(x).float().square().mean()
    loss.backward()
    opt.step()
    with torch.no_grad():
        assert lm.q_proj.B.abs().max() > 0            # update landed in fp32 master
    y2 = lm.q_proj(x)
    assert y2.dtype == torch.bfloat16 and not torch.equal(y2, y)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
