#!/usr/bin/env python3
"""Build-time (off-meter): freeze + tokenize the V1 corpus for one model.

wikitext-103-raw-v1 test split, joined, tokenized with the model's tokenizer,
first 98 windows x 512 tokens (~50k tokens — the house n_tokens=50_000 /
window=512 from humboldt_act). Saves shackleton_corpus.npy INSIDE the model
dir so gen_dockerfile bakes it into the image, and records the content hash.

Usage: python3 prepare_corpus.py <model_dir>
"""
import hashlib
import json
import sys

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

N_WINDOWS, WINDOW = 98, 512

model_dir = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_dir)
# namespaced id (bare "wikitext" is invalid in current datasets — D4);
# revision pinned so the baked corpus is deterministic across builds
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test",
                  revision="b08601e04326c79dfdd32d625aee71d232d685c3")
text = "\n".join(r["text"] for r in ds)
if tok.bos_token_id is not None:
    # BOS-family model (gemma/PG convention: prepend BOS to every forward):
    # each WINDOW starts with BOS so position 0 is the BOS token — the very
    # object cell (c) measures. Content tokens tokenized without specials.
    body = tok(text, return_tensors="np", add_special_tokens=False).input_ids[0]
    need = N_WINDOWS * (WINDOW - 1)
    assert len(body) >= need, f"corpus too short: {len(body)} < {need}"
    content = body[:need].reshape(N_WINDOWS, WINDOW - 1)
    bos = np.full((N_WINDOWS, 1), tok.bos_token_id, dtype=content.dtype)
    windows = np.concatenate([bos, content], axis=1).astype(np.int64)
else:
    ids = tok(text, return_tensors="np").input_ids[0]
    need = N_WINDOWS * WINDOW
    assert len(ids) >= need, f"corpus too short: {len(ids)} < {need}"
    windows = ids[:need].reshape(N_WINDOWS, WINDOW).astype(np.int64)

out = f"{model_dir}/shackleton_corpus.npy"
np.save(out, windows)
h = hashlib.sha256(windows.tobytes()).hexdigest()
json.dump({"corpus": "wikitext-103-raw-v1:test", "n_windows": N_WINDOWS,
           "window": WINDOW, "sha256": h, "tokenizer": model_dir},
          open(f"{model_dir}/shackleton_corpus_hash.json", "w"), indent=2)
print(f"[corpus] {out} sha256={h[:16]}")
