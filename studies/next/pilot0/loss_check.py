"""GPU loss check for one reconstructed batch (00b_DATA_ORDER.md Step 8).

Forwards the 512 x 8192 tokens of batch_<s>/tokens.npy through allenai/Olmo-3-1025-7B at one revision and computes
the stage-1 training loss: mean next-token cross-entropy over all 512 x 8191 positions. Labels are the inputs
shifted by one with nothing masked: the run set no label_mask, attention_mask or instance filter, so pad tokens
(100277) are targets too. Attention is plain causal across document boundaries (no attention_mask is passed);
the sliding window comes from the revision's config.json.

The training log did not count pad targets (pilot0/GPU_CHECK.md, "Pad tokens"). So the JSON also holds the mean with
pad targets excluded, and per row: the cross-entropy sum and the count of non-pad target positions, and their ratio.

--rows N scores only the first N rows (default: all 512).

The --out JSON is rewritten (atomically) every 16 rows with "complete": false and the rows done so far, and a last
time with "complete": true.

usage: python loss_check.py --tokens batch_60001/tokens.npy --revision stage1-step60000 \
           --target 2.1240015029907227 --sd 0.0248 --attn sdpa --out batch_60001/loss_check_sdpa.json
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "allenai/Olmo-3-1025-7B"
PAD_ID = 100277
HF_VOCAB = 100278
TRAIN_VOCAB = 100352  # training run config.json -> model.vocab_size (pilot0 README, "Known differences")

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--target", type=float, required=True, help="W&B train/CE loss logged at the batch's trainer step")
ap.add_argument("--sd", type=float, required=True, help="sd of train/CE loss across steps 60,001-61,000")
ap.add_argument("--attn", required=True, help="attn_implementation: sdpa, eager or flash_attention_2")
ap.add_argument("--out", required=True)
ap.add_argument("--instance-type", default=None, help="recorded only")
ap.add_argument("--rows", type=int, default=512, help="score the first ROWS rows")
args = ap.parse_args()

tokens = np.load(args.tokens)
meta = json.load(open(os.path.join(os.path.dirname(args.tokens), "meta.json")))
assert tokens.dtype == np.uint32 and tokens.shape == (512, 8192), (tokens.dtype, tokens.shape)
assert hashlib.sha256(tokens.tobytes()).hexdigest() == meta["token_bytes_sha256"], "tokens.npy differs from meta.json"
assert int(tokens.max()) < HF_VOCAB, int(tokens.max())
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

t_load = time.time()
model = AutoModelForCausalLM.from_pretrained(REPO, revision=args.revision, dtype=torch.bfloat16,
                                             attn_implementation=args.attn, device_map="cuda").eval()
tok = AutoTokenizer.from_pretrained(REPO, revision=args.revision)
load_seconds = time.time() - t_load
cfg = model.config
assert model.dtype == torch.bfloat16, model.dtype
assert cfg._attn_implementation == args.attn, (cfg._attn_implementation, args.attn)
assert cfg.sliding_window == 4096 and cfg.layer_types[:4] == ["sliding_attention"] * 3 + ["full_attention"], cfg
assert cfg.vocab_size == HF_VOCAB and model.lm_head.weight.shape[0] == HF_VOCAB, (cfg.vocab_size, model.lm_head.weight.shape)
assert tok.convert_ids_to_tokens(PAD_ID) == "<|pad|>", tok.convert_ids_to_tokens(PAD_ID)

assert 1 <= args.rows <= tokens.shape[0], args.rows
n_rows, n_pred = args.rows, tokens.shape[1] - 1
ce_rows, top1_rows, pad_n_rows, pad_ce_rows, sample = [], [], [], [], []


def write(complete, seconds):
    done = len(ce_rows)
    n_pos = done * n_pred
    mean_ce = sum(ce_rows) / n_pos if done else None
    n_pad = sum(pad_n_rows)
    res = {
        "complete": complete, "rows_done": done, "rows_total": n_rows,
        "model": REPO, "revision": args.revision, "attn_implementation": args.attn,
        "tokens": args.tokens, "token_bytes_sha256": meta["token_bytes_sha256"],
        "trainer_step": meta["trainer_step"], "batch_index_0based": meta["batch_index_0based"],
        "positions": n_pos, "mean_ce": mean_ce, "target_wandb_ce": args.target,
        "diff": None if mean_ce is None else mean_ce - args.target,
        "diff_in_step_sd": None if mean_ce is None else (mean_ce - args.target) / args.sd, "step_sd": args.sd,
        "top1_accuracy": sum(top1_rows) / n_pos if done else None,
        "pad": {"pad_id": PAD_ID, "handling": "pad tokens are inputs and targets, no mask; mean_ce counts every "
                                              "position, mean_ce_excluding_pad_targets drops pad targets",
                "pad_target_positions": n_pad, "ce_sum_on_pad_targets": sum(pad_ce_rows),
                "mean_ce_excluding_pad_targets": (sum(ce_rows) - sum(pad_ce_rows)) / (n_pos - n_pad) if done else None,
                "per_row_pad_target_positions": pad_n_rows},
        "vocab": {"hf_vocab_size": cfg.vocab_size, "training_vocab_size": TRAIN_VOCAB,
                  "note": f"the softmax here runs over {cfg.vocab_size} classes; the training LM head had "
                          f"{TRAIN_VOCAB}, so {TRAIN_VOCAB - cfg.vocab_size} logits in the training denominator "
                          "are absent here"},
        "dtype": {"parameters": str(model.dtype), "logits_for_ce": "float32 (bf16 logits cast)",
                  "row_sums": "float64"},
        "per_row_mean_ce": [c / n_pred for c in ce_rows], "per_row_top1": top1_rows,
        "per_row_nonpad_target_positions": [n_pred - k for k in pad_n_rows],
        "per_row_ce_sum_excluding_pad_targets": [c - q for c, q in zip(ce_rows, pad_ce_rows)],
        "per_row_mean_ce_excluding_pad_targets": [(c - q) / (n_pred - k) if k < n_pred else None
                                                  for c, q, k in zip(ce_rows, pad_ce_rows, pad_n_rows)],
        "row0_first32": sample,
        "seconds_load": round(load_seconds, 1), "seconds_forward": round(seconds, 1),
        "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "instance_type": args.instance_type, "model_config": json.loads(cfg.to_json_string(use_diff=False)),
    }
    tmp = args.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, args.out)
    return res


t0 = time.time()
with torch.inference_mode():
    for b in range(n_rows):
        x = torch.from_numpy(tokens[b].astype(np.int64)).cuda()
        logits = model(input_ids=x[None], use_cache=False).logits[0, :-1].float()
        assert logits.shape == (n_pred, HF_VOCAB), logits.shape
        target = x[1:]
        ce = F.cross_entropy(logits, target, reduction="none").double()
        assert torch.isfinite(ce).all(), b
        is_pad = target == PAD_ID
        ce_rows.append(ce.sum().item())
        pad_n_rows.append(int(is_pad.sum().item()))
        pad_ce_rows.append(ce[is_pad].sum().item())
        top1 = logits.argmax(-1)
        top1_rows.append(int((top1 == target).sum().item()))
        if b == 0:  # bank real targets, predictions and probabilities for the first 32 positions of row 0
            p = logits[:32].softmax(-1)
            sample = [{"pos": i, "context_last": tok.convert_ids_to_tokens(int(x[i])),
                       "target": int(target[i]), "target_str": tok.convert_ids_to_tokens(int(target[i])),
                       "p_target": p[i, target[i]].item(),
                       "top1": int(top1[i]), "top1_str": tok.convert_ids_to_tokens(int(top1[i])),
                       "p_top1": p[i, top1[i]].item(), "correct": bool(top1[i] == target[i])} for i in range(32)]
        del logits
        if (b + 1) % 16 == 0:
            write(False, time.time() - t0)

res = write(True, time.time() - t0)
print(f"step {meta['trainer_step']} (batch {meta['batch_index_0based']}) at {args.revision}, attn {args.attn}: "
      f"mean CE {res['mean_ce']:.7f}  W&B {args.target:.7f}  diff {res['diff']:+.7f} = {res['diff_in_step_sd']:+.3f} "
      f"step sd  top-1 {res['top1_accuracy']:.4f}  {res['seconds_forward']} s  -> {args.out}", flush=True)
