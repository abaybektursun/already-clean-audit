"""Relearning-ladder core (W17/W18/SAV-π shared; program spec §3.1 `infra/pg/relearn.py`).

Pure, importable science for the capability-data relearning curves:

  * QA formatting — MMLU auxiliary-train rows rendered EXACTLY like the eval
    instrument's `_q_block` (text_eval), answer appended; the training loss
    reads the single answer-letter token (matches the single-token eval read).
  * data ladder — deterministic sha1-bucketed subsets of the relearn pool at
    sizes n ∈ ladder; dedup against the eval population by the SAME item hash
    recipe (sha1(question[:64])[:12]); audit counts returned, never hidden.
  * shuffled-label arm — deterministic derangement-style permutation of the
    answer indices (the EDL≈0 no-information floor).
  * rank-r adapter — hand-rolled W + (α/r)·A·B on every 2-D decoder linear
    (NOT peft; eq1 precedent). From-scratch init: A ~ N(0, 1/r), B = 0 ⇒ the
    adapter starts at exactly zero delta. Target scope = the W16 probe-scope
    convention (2-D, embed/lm_head excluded).

Heavy imports inside functions; fail-loud; no try/except. The GPU trainer loop
lives in the job (thin); everything testable on a laptop is here.
"""
from __future__ import annotations

import hashlib

CHOICES = ("A", "B", "C", "D")
LADDER_N = (0, 64, 256, 1000, 4000, 16000)        # program spec §5 data axis
DEFAULT_RANK = 512                                 # primary budget (spec §2)


# ===========================================================================
# QA example formatting (mirrors text_eval._q_block; answer appended).
# ===========================================================================
def qa_item_hash(question: str) -> str:
    """THE alignment-key recipe (identical to text_eval per_item_hash)."""
    return hashlib.sha1(question[:64].encode()).hexdigest()[:12]


def format_qa(question: str, choices, answer_idx: int) -> dict:
    """One supervised example: prompt ends at 'Answer:'; target is the single
    letter token (leading space, the eval convention). Returns
    {prompt, target_text, answer_idx, hash}."""
    assert len(choices) == 4, f"need 4 choices, got {len(choices)}"
    assert 0 <= int(answer_idx) <= 3, f"answer_idx {answer_idx} out of range"
    s = question.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{CHOICES[i]}. {c}\n"
    s += "Answer:"
    return {"prompt": s, "target_text": f" {CHOICES[int(answer_idx)]}",
            "answer_idx": int(answer_idx), "hash": qa_item_hash(question)}


# ===========================================================================
# Ladder subsets + dedup audit.
# ===========================================================================
def dedup_against_eval(rows: list, eval_hashes) -> dict:
    """Drop relearn rows whose item hash collides with the eval population.
    rows: LIST of dicts with 'question'. Returns {kept, n_in, n_dropped}."""
    assert isinstance(rows, list), "rows must be a list (audit needs sizes)"
    ev = set(eval_hashes)
    kept = [r for r in rows if qa_item_hash(r["question"]) not in ev]
    return {"kept": kept, "n_in": len(rows), "n_dropped": len(rows) - len(kept)}


def ladder_subset(rows, n: int, salt: str = "w17-ladder") -> list:
    """Deterministic n-subset: order rows by sha1(salt + item-hash), take the
    first n. Nested property: subset(n1) ⊂ subset(n2) for n1 < n2 — pre-named
    ladder cells share their smaller cells' examples (a frozen, auditable
    choice; no RNG state involved)."""
    assert n >= 0
    keyed = sorted(rows, key=lambda r: hashlib.sha1(
        (salt + qa_item_hash(r["question"])).encode()).hexdigest())
    assert n <= len(keyed), f"ladder n={n} exceeds pool {len(keyed)}"
    return keyed[:n]


def shuffle_labels(rows, seed: int = 1337) -> list:
    """The EDL≈0 floor arm: deterministically permute answers so that every
    row's label is replaced by a label drawn from the row-set's label
    distribution, with the constraint that the permutation is a fixed seeded
    shuffle (label marginals preserved exactly). Returns NEW row dicts."""
    import numpy as np
    rows = list(rows)
    labels = [int(r["answer"]) for r in rows]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    return [{**r, "answer": labels[perm[i]]} for i, r in enumerate(rows)]


# ===========================================================================
# Rank-r adapter (hand-rolled; NOT peft — eq1 convention, from-scratch init).
# ===========================================================================
def adapter_target_names(lm) -> list:
    """Same scope convention as w16_probes.probe_param_names, expressed at the
    MODULE level: every nn.Linear under the decoder whose weight is 2-D and
    whose name carries no embed/lm_head."""
    import torch.nn as nn
    return sorted(n for n, m in lm.named_modules()
                  if isinstance(m, nn.Linear) and m.weight.ndim == 2
                  and "embed_tokens" not in n and "lm_head" not in n)


def make_adapter(base_linear, rank: int, seed: int, alpha: float = 1.0):
    """Wrap a frozen nn.Linear as W·x + (alpha/rank)·A(B(x)).
    Init: B = 0, A ~ N(0, 1/rank) ⇒ zero delta at step 0 (from-scratch
    relearning; the no-retrain graft inits of eq1 are deliberately NOT here).
    Returns the wrapper module; only A, B require grad."""
    import torch
    import torch.nn as nn

    class RankRAdapter(nn.Module):
        def __init__(self, base, r, a_scale):
            super().__init__()
            out_d, in_d = base.weight.shape
            r = min(r, out_d, in_d)
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            g = torch.Generator().manual_seed(int(seed))
            self.A = nn.Parameter(torch.randn(out_d, r, generator=g) * (1.0 / r) ** 0.5)
            self.B = nn.Parameter(torch.zeros(r, in_d))
            self.scale = a_scale / r
            self.rank = r

        def forward(self, x):
            # A/B are fp32 MASTERS (AdamW updates land in fp32 — bf16 states
            # underflow near the mantissa floor); the matmuls run in x's dtype
            # via per-forward casts. An fp32 x.float() path would make autograd
            # save a full-precision activation copy per adapter (182×) — OOM at
            # long sequences (W17 preflight gate-2b incident). Saved tensors
            # here: x (already saved by base, shared storage) + a (B,S,r)
            # low-rank intermediate in x's dtype.
            B = self.B.to(x.dtype).transpose(0, 1)
            A = self.A.to(x.dtype).transpose(0, 1)
            return self.base(x) + self.scale * ((x @ B) @ A)

    return RankRAdapter(base_linear, rank, alpha)


def letter_token_ids(tokenizer) -> list:
    """The four answer-letter target ids. Fast path: ' A' encodes to ONE token
    (Gemma family). Fallback (Mistral/Llama SentencePiece, which split the
    leading space): the explicit '▁A' vocab entry via convert_tokens_to_ids —
    the same continuation token an answer ' A' begins with. Fail-loud if
    neither resolves to 4 distinct, known ids."""
    ids = []
    for L in CHOICES:
        enc = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
            continue
        tid = tokenizer.convert_tokens_to_ids("\u2581" + L)
        unk = getattr(tokenizer, "unk_token_id", None)
        assert isinstance(tid, int) and tid >= 0 and tid != unk, \
            f"' {L}': {len(enc)} tokens and no '▁{L}' vocab entry"
        ids.append(tid)
    assert len(set(ids)) == 4, f"letter ids collide: {ids}"
    return ids


def qa_train_step_batches(examples: list, batch_size: int) -> list:
    """Deterministic batch index ranges over a FIXED example order (the ladder
    order). Epoch shuffling is seeded inside the trainer; this helper only
    partitions. Returns list of (start, end)."""
    assert batch_size > 0
    return [(s, min(s + batch_size, len(examples)))
            for s in range(0, len(examples), batch_size)]


def train_adapter_qa(model, tokenizer, device, examples: list, *, lr: float,
                     epochs: int, batch_size: int, warmup_ratio: float,
                     seed: int, max_steps: int | None = None,
                     micro_batch: int | None = None,
                     grad_hook=None, step_hook=None,
                     weight_decay: float = 0.01, loss_log: list | None = None) -> dict:
    """The ONE relearning trainer (frozen-recipe discipline; W17 prereg §5).

    CE on the single answer-letter token, computed EXACTLY like the eval read:
    left-padded prompts ending at 'Answer:', content-relative position_ids,
    logits_to_keep=1, loss = CE(last-token logits, letter_id). AdamW + cosine
    + warmup over trainable (adapter) params only. Deterministic in `seed`
    (epoch shuffles from one torch.Generator). Fail-loud on non-finite loss.
    `weight_decay` defaults to torch's AdamW default (0.01 — what every prior ladder
    ran); MENDELEEV passes the frozen recipes-file value. `loss_log`, when a list,
    receives the per-step answer-token CE (the recovery train curve).
    Returns {steps, final_loss, mean_loss_last_epoch, tokens_seen, ...}."""
    import math

    import torch

    def _cosine_warmup(opt, n_warm, n_total):
        def f(s):
            if s < n_warm:
                return s / max(1, n_warm)
            p = (s - n_warm) / max(1, n_total - n_warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)

    letters = letter_token_ids(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    prompts = [e["prompt"] for e in examples]
    targets = [letters[e["answer_idx"]] for e in examples]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params — attach adapters first"
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    n_batches = len(qa_train_step_batches(examples, batch_size))
    total_steps = max_steps if max_steps is not None else epochs * n_batches
    sched = _cosine_warmup(opt, int(warmup_ratio * total_steps), total_steps)
    g = torch.Generator().manual_seed(int(seed))

    model.train()
    step = 0
    tokens_seen = 0
    loss_tokens = 0
    losses: list = []
    done = False
    for _ep in range(epochs if max_steps is None else 10 ** 9):
        order = torch.randperm(len(examples), generator=g).tolist()
        for s, e in qa_train_step_batches(examples, batch_size):
            idx = order[s:e]
            loss_tokens += len(idx)
            mb = micro_batch or len(idx)
            step_loss = 0.0
            for ms in range(0, len(idx), mb):
                midx = idx[ms:ms + mb]
                enc = tokenizer([prompts[i] for i in midx], return_tensors="pt",
                                padding=True, add_special_tokens=True).to(device)
                pos = enc["attention_mask"].long().cumsum(-1) - 1
                pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
                tgt = torch.tensor([targets[i] for i in midx], device=device)
                last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
                # scale so accumulated grads == the full-batch mean-CE gradient
                loss = (torch.nn.functional.cross_entropy(last.float(), tgt)
                        * (len(midx) / len(idx)))
                assert torch.isfinite(loss), f"non-finite loss at step {step} — HALT (no silent retry)"
                loss.backward()
                step_loss += float(loss.item())
                tokens_seen += int(enc["attention_mask"].sum().item())
            if grad_hook is not None:      # gradient-space intervention (PRAGNANZ M4);
                grad_hook()                # runs between backward and step, no-op when None
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(step_loss)
            if loss_log is not None:
                loss_log.append(step_loss)
            step += 1
            if step_hook is not None:
                step_hook(step)
            if max_steps is not None and step >= max_steps:
                done = True
                break
        if done:
            break
    model.eval()
    tokenizer.padding_side = orig_side
    last_ep = losses[-n_batches:] if len(losses) >= n_batches else losses
    first_ep = losses[:n_batches] if len(losses) >= n_batches else losses
    return {"steps": step, "final_loss": losses[-1],
            "mean_loss_first_epoch": sum(first_ep) / len(first_ep),
            "mean_loss_last_epoch": sum(last_ep) / len(last_ep),
            "tokens_seen": tokens_seen,
            "loss_tokens": loss_tokens,
            "n_examples": len(examples),
            "recipe": {"lr": lr, "epochs": epochs, "batch_size": batch_size,
                       "warmup_ratio": warmup_ratio, "seed": int(seed),
                       "max_steps": max_steps, "micro_batch": micro_batch,
                       "weight_decay": weight_decay}}


def retain_token_matched_slice(texts: list, tokenizer, token_budget: int) -> dict:
    """Greedy prefix of `texts` whose total token count first reaches
    `token_budget` (the retain-only arm's token matching). Returns
    {texts, n_tokens, n_texts}; fail-loud if the pool is too small."""
    assert token_budget > 0
    out, total = [], 0
    for t in texts:
        if total >= token_budget:
            break
        n = len(tokenizer.encode(t, add_special_tokens=False))
        if n == 0:
            continue
        out.append(t)
        total += n
    assert total >= token_budget, \
        f"retain pool exhausted: {total} < budget {token_budget}"
    return {"texts": out, "n_tokens": total, "n_texts": len(out)}


def attach_adapters(lm, rank: int, seed: int, alpha: float = 1.0) -> dict:
    """Replace every target Linear in `lm` with its adapter wrapper, in place.
    Per-module seeds derived deterministically from (seed, module-name).
    Freezes EVERY existing param first (embed/lm_head/norms included) — after
    attach, exactly the adapter A/B matrices are trainable.
    Returns {n_modules, trainable_params, frozen_params, names}."""
    names = adapter_target_names(lm)
    assert names, "no adapter targets found"
    for p in lm.parameters():
        p.requires_grad_(False)
    for n in names:
        parent_name, _, leaf = n.rpartition(".")
        parent = lm.get_submodule(parent_name) if parent_name else lm
        base = getattr(parent, leaf)
        sub_seed = int(hashlib.sha1(f"{seed}:{n}".encode()).hexdigest()[:8], 16)
        setattr(parent, leaf, make_adapter(base, rank, sub_seed, alpha))
    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lm.parameters() if not p.requires_grad)
    return {"n_modules": len(names), "trainable_params": int(trainable),
            "frozen_params": int(frozen), "names": names, "rank": rank, "alpha": alpha}
