"""V1 (b)+(c) sufficient statistics — activation share and position-0 share.

One forward sweep over the frozen, pre-tokenized corpus (baked into the image)
with forward-pre-hooks at the three house sites per layer (humboldt_act site
discipline: q_proj_in for q/k/v banks, o_proj_in, gate_proj_in). Banked per
(layer, site), all in float64:

    sum_sq[d]    Σ_tokens h_j²      → per-coordinate RMS, top-k share at ANY k
    sum_quad[d]  Σ_tokens h_j⁴      → spikiness/kurtosis
    pos_energy[w] Σ_coords h² at each position → position-0 (BOS) share
    pos0_sq[d]   Σ h_j² at position 0 only     → which coordinates spike at BOS

Because the banks under study are coordinate sets (axis-aligned projections),
aggregate share statistics AND their random-set twin bands are exact functions
of these sums — so every gate/threshold computation happens offline (§6), and
billed GPU time is one forward sweep. FAIL LOUD: no try/except.
"""
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

SITES = ("q_proj_in", "o_proj_in", "gate_proj_in")
SITE_MODULE = {"q_proj_in": "self_attn.q_proj",
               "o_proj_in": "self_attn.o_proj",
               "gate_proj_in": "mlp.gate_proj"}


def run(cell, out_dir):
    model_dir = cell["model_dir"]
    corpus_path = cell.get("corpus_path", os.path.join(model_dir, "shackleton_corpus.npy"))
    # AM-4 additions, both defaulting to the banked behaviour:
    #   corpus_s3     a download-mode model has no corpus baked beside it, so the
    #                 tokenized corpus for ITS tokenizer is fetched at cell start
    #   window slice  the frozen ">= 2 eval batches" control needs two DISJOINT
    #                 slices of the same corpus, scored independently
    if cell.get("corpus_s3"):
        import subprocess
        corpus_path = "/workspace/actstats_corpus.npy"
        subprocess.run(["aws", "s3", "cp", cell["corpus_s3"], corpus_path],
                       check=True, capture_output=True)
    batch = cell.get("batch", 8)
    w_off = int(cell.get("window_offset", 0))
    w_n = cell.get("n_windows")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    tokens = np.load(corpus_path)  # (n_windows, window) int64
    if w_off or w_n:
        tokens = tokens[w_off:w_off + int(w_n)] if w_n else tokens[w_off:]
        assert len(tokens) > 0, (w_off, w_n)
    n_win, window = tokens.shape

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype,
        device_map=cell.get("device_map", dev),
        attn_implementation=cell.get("attn", "sdpa"))
    model.eval()
    for name, p in model.named_parameters():
        assert p.dtype == dtype, f"non-{dtype} weight in measurement path: {name}"

    t_load = time.time() - t0
    layers = model.model.layers
    n_layers = len(layers)
    acc = {}
    for li in range(n_layers):
        for site in SITES:
            mod = layers[li]
            for part in SITE_MODULE[site].split("."):
                mod = getattr(mod, part)
            d = mod.in_features
            a = {"sum_sq": torch.zeros(d, dtype=torch.float64, device=mod.weight.device),
                 "sum_quad": torch.zeros(d, dtype=torch.float64, device=mod.weight.device),
                 "pos_energy": torch.zeros(window, dtype=torch.float64, device=mod.weight.device),
                 "pos0_sq": torch.zeros(d, dtype=torch.float64, device=mod.weight.device),
                 "tokens": 0}
            acc[(li, site)] = a

            def hook(module, args, key=(li, site)):
                h = args[0]
                assert h.dtype == dtype, f"hooked tensor dtype {h.dtype} at {key}"
                h64 = h.detach().to(torch.float64)
                sq = h64.pow(2)                       # (B, T, d)
                a = acc[key]
                a["sum_sq"] += sq.sum(dim=(0, 1))
                a["sum_quad"] += sq.pow(2).sum(dim=(0, 1))
                a["pos_energy"] += sq.sum(dim=(0, 2))
                a["pos0_sq"] += sq[:, 0, :].sum(dim=0)
                a["tokens"] += h.shape[0] * h.shape[1]
            mod.register_forward_pre_hook(hook)

    t1 = time.time()
    with torch.no_grad():
        for i in range(0, n_win, batch):
            ids = torch.from_numpy(tokens[i:i + batch]).to(model.device)
            model(input_ids=ids)

    arrays, meta = {}, []
    for (li, site), a in acc.items():
        key = f"L{li:02d}_{site}"
        arrays[f"{key}_sum_sq"] = a["sum_sq"].cpu().numpy()
        arrays[f"{key}_sum_quad"] = a["sum_quad"].cpu().numpy()
        arrays[f"{key}_pos_energy"] = a["pos_energy"].cpu().numpy()
        arrays[f"{key}_pos0_sq"] = a["pos0_sq"].cpu().numpy()
        pos0_share = float(a["pos_energy"][0] / a["pos_energy"].sum())
        meta.append({"layer": li, "site": site, "d": int(a["sum_sq"].shape[0]),
                     "tokens": a["tokens"], "pos0_share": pos0_share})
    np.savez_compressed(os.path.join(out_dir, "actstats.npz"), **arrays)

    corpus_hash = __import__("hashlib").sha256(tokens.tobytes()).hexdigest()
    flags = {"cell": cell["id"], "model_dir": model_dir, "n_layers": n_layers,
             "t_load_s": round(t_load, 2), "t_sweep_s": round(time.time() - t1, 2),
             "n_windows": int(n_win), "window": int(window),
             "window_offset": w_off, "corpus_s3": cell.get("corpus_s3"),
             "corpus_sha256": corpus_hash, "dtype": str(dtype),
             "attn": cell.get("attn", "sdpa"), "torch": torch.__version__,
             "sites": list(SITES), "per_site": meta}
    with open(os.path.join(out_dir, "actstats_flags.json"), "w") as f:
        json.dump(flags, f, indent=2)
