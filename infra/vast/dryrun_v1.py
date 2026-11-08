#!/usr/bin/env python3
"""§9.1 weight-free dry-run of the V1 actstats cell on a tiny random Qwen2.

No downloads: builds a 2-layer random-weight Qwen2 in-process, a synthetic
corpus, runs cells/v1_actstats.run end-to-end on CPU, then validates the
offline-share math on the banked sums (shares in [0,1], token counts exact,
position-0 share consistent between npz and flags).
"""
import json
import os
import tempfile

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

from infra.vast.cells import v1_actstats

tmp = tempfile.mkdtemp()
model_dir = os.path.join(tmp, "model")
cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
                  max_position_embeddings=64)
torch.manual_seed(0)
Qwen2ForCausalLM(cfg).save_pretrained(model_dir)

corpus = np.random.default_rng(0).integers(0, 512, size=(6, 16)).astype(np.int64)
corpus_path = os.path.join(tmp, "corpus.npy")
np.save(corpus_path, corpus)

out_dir = os.path.join(tmp, "out")
os.makedirs(out_dir)
cell = {"id": "dryrun_v1", "model_dir": model_dir, "corpus_path": corpus_path,
        "batch": 4, "device_map": "cpu"}
v1_actstats.run(cell, out_dir)

z = np.load(os.path.join(out_dir, "actstats.npz"))
flags = json.load(open(os.path.join(out_dir, "actstats_flags.json")))
assert flags["n_layers"] == 2 and len(flags["per_site"]) == 6
for row in flags["per_site"]:
    key = f"L{row['layer']:02d}_{row['site']}"
    sum_sq = z[f"{key}_sum_sq"]
    pos_energy = z[f"{key}_pos_energy"]
    assert row["tokens"] == 6 * 16, row
    assert sum_sq.shape == (row["d"],) and (sum_sq >= 0).all()
    # offline top-k share from sums: in (0,1], monotone in k
    order = np.argsort(sum_sq)[::-1]
    s32 = sum_sq[order[:32]].sum() / sum_sq.sum()
    s8 = sum_sq[order[:8]].sum() / sum_sq.sum()
    assert 0 < s8 <= s32 <= 1.0
    # position-0 share consistent between npz and flags
    assert abs(pos_energy[0] / pos_energy.sum() - row["pos0_share"]) < 1e-12
    # energy conservation across decompositions: Σ_j sum_sq == Σ_pos pos_energy
    assert abs(sum_sq.sum() - pos_energy.sum()) / pos_energy.sum() < 1e-12
    # pos0_sq is exactly the position-0 slice of the energy
    assert abs(z[f"{key}_pos0_sq"].sum() - pos_energy[0]) / pos_energy[0] < 1e-12
# tiny-Gemma leg: the anchor image's family must pass the same machinery
from transformers import GemmaConfig, GemmaForCausalLM
gdir = os.path.join(tmp, "gmodel")
gcfg = GemmaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
                   head_dim=16, max_position_embeddings=64)
GemmaForCausalLM(gcfg).save_pretrained(gdir)
gout = os.path.join(tmp, "gout")
os.makedirs(gout)
v1_actstats.run({"id": "dryrun_v1_gemma", "model_dir": gdir,
                 "corpus_path": corpus_path, "batch": 4,
                 "device_map": "cpu"}, gout)
gflags = json.load(open(os.path.join(gout, "actstats_flags.json")))
assert gflags["n_layers"] == 2 and len(gflags["per_site"]) == 6
print("DRYRUN-V1 PASS: hooks, sums, offline share math, energy conservation (qwen2+gemma)")
