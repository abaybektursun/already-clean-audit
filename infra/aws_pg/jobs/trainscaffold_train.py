"""TRAINSCAFFOLD trainers + recognition scorer (GPU). NEW per §13:

  * train_adapter_lm     — the masked rank-r LM-loss adapter trainer (free-gen format). Loss is CE on
                           the FACT-SPAN tokens only (prompt + padding → −100; matched to the letter-CE
                           single-target span, §2.3/T7). Optimizer/schedule/recipe matched to
                           relearn.train_adapter_qa so the format contrast is recipe-clean. Returns the
                           train report incl. the per-step loss trajectory.
  * recognition_cloze    — the generic invented-MC cloze-recognition scorer (held-out PROBE; the
                           readout-AGNOSTIC RF_novel axis). 0-shot, fixed neutral framing, NO cross-fact
                           exemplars. Mirrors text_eval.mmlu_cloze's continuation scoring (reuses its
                           module-level helpers) over arbitrary (prompt, candidates) items. Returns
                           per_item 0/1 + acc.

Heavy imports inside functions; fail-loud; no try/except.
"""
from __future__ import annotations


def train_adapter_lm(model, tokenizer, device, examples, *, lr, epochs, batch_size,
                     warmup_ratio, seed, micro_batch=None):
    """examples = [{"prompt","target"}]; CE on the target (fact-span) tokens only. Right-padded;
    HF causal-LM internal shift handles label alignment (prompt+pad labels = −100). AdamW + cosine +
    warmup over the trainable (adapter) params (recipe-matched to train_adapter_qa). Fail-loud on
    non-finite loss. Returns {steps, final_loss, mean_loss_last_epoch, loss_trajectory, ...}."""
    import math

    import torch

    def _cosine_warmup(opt, n_warm, n_total):
        def f(s):
            if s < n_warm:
                return s / max(1, n_warm)
            p = (s - n_warm) / max(1, n_total - n_warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "right"           # right-pad: the fact-span is at the end, clean span CE

    # precompute (input_ids, labels) per example; labels mask prompt + padding (−100).
    enc_ex = []
    for ex in examples:
        pid = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
        fid = tokenizer(ex["prompt"] + ex["target"], add_special_tokens=True)["input_ids"]
        assert fid[:len(pid)] == pid and len(fid) > len(pid), \
            f"target not a clean continuation of prompt: {ex['prompt']!r} + {ex['target']!r}"
        labels = [-100] * len(pid) + fid[len(pid):]
        enc_ex.append((fid, labels))

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params — attach adapters first"
    opt = torch.optim.AdamW(trainable, lr=lr)
    n_batches = (len(examples) + batch_size - 1) // batch_size
    total_steps = epochs * n_batches
    sched = _cosine_warmup(opt, int(warmup_ratio * total_steps), total_steps)
    g = torch.Generator().manual_seed(int(seed))
    pad_id = tokenizer.pad_token_id

    model.train()
    step = 0
    losses: list = []
    for _ep in range(epochs):
        order = torch.randperm(len(examples), generator=g).tolist()
        for s in range(0, len(examples), batch_size):
            idx = order[s:s + batch_size]
            mb = micro_batch or len(idx)
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for ms in range(0, len(idx), mb):
                midx = idx[ms:ms + mb]
                maxlen = max(len(enc_ex[i][0]) for i in midx)
                input_ids = torch.full((len(midx), maxlen), pad_id, dtype=torch.long)
                labels = torch.full((len(midx), maxlen), -100, dtype=torch.long)
                attn = torch.zeros((len(midx), maxlen), dtype=torch.long)
                for j, i in enumerate(midx):
                    fid, lab = enc_ex[i]
                    input_ids[j, :len(fid)] = torch.tensor(fid)
                    labels[j, :len(lab)] = torch.tensor(lab)
                    attn[j, :len(fid)] = 1
                input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                loss = loss * (len(midx) / len(idx))
                assert torch.isfinite(loss), f"non-finite loss at step {step} — HALT"
                loss.backward()
                step_loss += float(loss.item())
            opt.step()
            sched.step()
            losses.append(step_loss)
            step += 1
    model.eval()
    tokenizer.padding_side = orig_side
    last_ep = losses[-n_batches:] if len(losses) >= n_batches else losses
    first_ep = losses[:n_batches] if len(losses) >= n_batches else losses
    return {"steps": step, "final_loss": losses[-1],
            "mean_loss_first_epoch": sum(first_ep) / len(first_ep),
            "mean_loss_last_epoch": sum(last_ep) / len(last_ep),
            "loss_trajectory": losses, "n_examples": len(examples),
            "recipe": {"lr": lr, "epochs": epochs, "batch_size": batch_size,
                       "warmup_ratio": warmup_ratio, "seed": int(seed), "micro_batch": micro_batch,
                       "format": "free-gen-masked-lm"}}


def recognition_cloze(model, processor, device, items, batch_size=8, gray=128):
    """Generic cloze-recognition over (prompt, candidates, answer_idx) items — the readout-agnostic
    RF_novel axis. Per item: score each candidate as a continuation of `prompt` (length-normed λ=1
    log-prob), argmax. Blank-image (PaliGemma in-distribution) forward; reuses text_eval's continuation
    helpers verbatim. Returns {per_item:[0/1], acc, n}."""
    import numpy as np
    import torch
    from PIL import Image

    from infra.pg.text_eval import (_common_prefix_len, _common_suffix_len, _score_cloze_window)

    tok = processor.tokenizer
    blank = Image.fromarray(np.full((224, 224, 3), gray, dtype=np.uint8))

    def encode(texts):
        return processor(text=texts, images=[blank] * len(texts),
                         return_tensors="pt", padding="longest").to(device)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"

    per_item = []
    model.eval()
    for start in range(0, len(items), batch_size):
        for it in items[start:start + batch_size]:
            prompt, cands, ci = it["prompt"], it["candidates"], int(it["answer_idx"])
            assert len(cands) == 4, f"recognition item needs 4 candidates, got {len(cands)}"
            pe = encode([prompt])
            prompt_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
            enc = encode([prompt + " " + c for c in cands])
            ids, am = enc["input_ids"], enc["attention_mask"]
            real = am.sum(dim=1).tolist()
            cont_ids, ntok, lcp, lcs = [], [], None, None
            for k in range(4):
                full_real = ids[k][ids.shape[1] - int(real[k]):].tolist()
                p = _common_prefix_len(prompt_real, full_real)
                ssfx = _common_suffix_len(prompt_real, full_real)
                assert p + ssfx == len(prompt_real), \
                    f"cloze prompt not [shared|sep] (cand {k}): lcp={p} lcs={ssfx} |p|={len(prompt_real)}"
                cont = full_real[p:len(full_real) - ssfx]
                assert len(cont) >= 1, f"empty cloze continuation (cand {k})"
                if lcp is None:
                    lcp, lcs = p, ssfx
                else:
                    assert (p, ssfx) == (lcp, lcs), "inconsistent cloze framing across candidates"
                cont_ids.append(cont)
                ntok.append(len(cont))
            K = max(ntok) + lcs + 1
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=am,
                            pixel_values=enc["pixel_values"].to(torch.bfloat16), logits_to_keep=K)
                lw = torch.log_softmax(out.logits.float(), dim=-1)
            sums = _score_cloze_window(lw, cont_ids, ntok, trailing=lcs)
            assert torch.isfinite(sums).all(), "non-finite cloze logprob"
            mean_lp = sums / torch.tensor(ntok, dtype=torch.float64)
            per_item.append(int(int(mean_lp.argmax()) == ci))
    tok.padding_side = orig_side
    n = len(per_item)
    assert n > 0, "recognition_cloze produced no items"
    return {"per_item": per_item, "acc": sum(per_item) / n, "n": n}
