"""Smoke cell (SHACKLETON §9.9): smallest end-to-end unit on the baked model.

Loads the model from the image's /models dir (offline), runs one forward on a
fixed probe, banks logit statistics. Doubles as the wiring proof that the
measurement path is unquantized bf16 (dtype asserted on every parameter).
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(cell, out_dir):
    model_dir = cell["model_dir"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=cell.get("device_map", dev))
    for name, p in model.named_parameters():
        assert p.dtype == dtype, f"quantized/wrong-dtype weight in path: {name} {p.dtype}"

    t_load = time.time() - t0
    t1 = time.time()
    ids = tok(cell.get("probe_text", "The capital of France is"),
              return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids)
    lg = out.logits[0, -1].float()
    stats = {
        "cell": cell["id"], "model_dir": model_dir, "device": dev,
        "t_load_s": round(t_load, 2), "t_forward_s": round(time.time() - t1, 2),
        "dtype": str(dtype), "n_params": sum(p.numel() for p in model.parameters()),
        "logit_mean": lg.mean().item(), "logit_std": lg.std().item(),
        "argmax_token": tok.decode(lg.argmax().item()),
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "smoke_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
