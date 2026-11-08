"""V3 — the batch-size dial (spec §1/§2/§3), gemma-2b substrate.

Per batch size B in the ladder, per layer (q_proj, all 18): train n_steps with
TRUE batch B under ExactAdam fp32, accumulating ELEMENTWISE sum_g / sum_g2
(M-A3 stage-1, reused verbatim: settling_ratio -> per_coordinate median down
rows -> subspace_mean per bank; never column-summed, M-D18) plus the mq2-style
delivery ratio per bank.

Pre-registered (expectations row): H1 = sigma^2/mu^2 falls with B AND delivery
into core32 rises toward the haar twin band; H0 = suppression batch-invariant;
VOID-INSTRUMENT if the ladder does not move sigma^2/mu^2. Scoring offline from
this cell's banked rows with thresholds from the run's own twin spread (§6).

Banks: census core32/anti32 from banks_s3; haar x3 in-cell (recorded seeds);
top/bottom-sigma32 via GPU SVD of base q_proj. complement32 out of scope
(recorded). FAIL LOUD.
"""
import json
import os
import subprocess
import time


def run(cell, out_dir):
    from infra.pg import humboldt_core as H
    H.hf_setup()  # BEFORE any hub/transformers import (D17)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from infra.pg import darwin_train as T
    from infra.aws_pg.jobs.mercator_a3 import settling_ratio, per_coordinate, subspace_mean

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model_dir = cell["model_dir"]
    ladder = [int(b) for b in str(cell["batch_ladder"]).split(",")]
    n_steps = int(cell.get("n_steps", 60))
    # AM-6: the banked V3 drew its minibatches from default_rng(2026+B), i.e. the
    # seed was a function of the RUNG — so "another seed" was inexpressible and
    # the -17% ordering had n=1 per rung. sample_seed makes the seed an argument;
    # the default reproduces the banked run bit-for-bit.
    sample_seed = int(cell.get("sample_seed", 2026))
    lr = float(cell.get("lr", 1e-5))
    corpus = np.load(os.path.join(model_dir, "shackleton_corpus.npy"))

    bdir = "/workspace/g2b_banks"
    subprocess.run(["aws", "s3", "cp", "--recursive", cell["banks_s3"], bdir],
                   check=True, capture_output=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map="cuda",
        attn_implementation="eager")   # gemma: eager per house convention
    # D19: fp32 full-graph backward at B=32x512 exceeds 95GB — checkpointing
    # trades compute for activations; per-step summed gradients are unchanged,
    # so the sigma^2/mu^2-across-steps estimand is untouched
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.eval()
    layers = model.model.layers
    NL = len(layers)
    d = layers[0].self_attn.q_proj.in_features
    K = 32

    def orth_t(A):
        q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
        return np.ascontiguousarray(q)

    rng = np.random.default_rng(1337)
    banks = {}
    for L in range(NL):
        banks[("core32", L)] = orth_t(np.load(f"{bdir}/q_proj__L{L}__core32.npy"))
        banks[("anti32", L)] = orth_t(np.load(f"{bdir}/q_proj__L{L}__anti32.npy"))
        for s in range(3):
            banks[(f"haar32_s{s}", L)] = orth_t(rng.standard_normal((d, K)))
        W = layers[L].self_attn.q_proj.weight.detach().to(torch.float32)
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        banks[("bottomsigma32", L)] = Vh[-K:, :].T.cpu().numpy().astype(np.float64)
        banks[("topsigma32", L)] = Vh[:K, :].T.cpu().numpy().astype(np.float64)
        del Vh
    BANK_IDS = ("core32", "anti32", "haar32_s0", "haar32_s1", "haar32_s2",
                "bottomsigma32", "topsigma32")
    U_gpu = {k: torch.as_tensor(v, device="cuda", dtype=torch.float32)
             for k, v in banks.items()}

    mods = {L: layers[L].self_attn.q_proj for L in range(NL)}
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)
    pristine = {L: m.weight.detach().clone() for L, m in mods.items()}

    rows_f = open(os.path.join(out_dir, "v3_rows.jsonl"), "w")
    for B in ladder:
        for L, m in mods.items():
            m.weight.data.copy_(pristine[L])
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=lr)
        srng = np.random.default_rng(sample_seed + B)
        sum_g = {L: torch.zeros_like(mods[L].weight, dtype=torch.float32)
                 for L in mods}
        sum_g2 = {L: torch.zeros_like(mods[L].weight, dtype=torch.float32)
                  for L in mods}
        grad_in = {b: 0.0 for b in BANK_IDS}
        grad_tot = 0.0
        model.train()
        t0 = time.time()
        for _ in range(n_steps):
            idx = srng.integers(0, corpus.shape[0], size=B)
            ids = torch.as_tensor(corpus[idx], device="cuda")
            opt.zero_grad()
            model(input_ids=ids, labels=ids).loss.backward()
            with torch.no_grad():
                for L, m in mods.items():
                    g = m.weight.grad
                    sum_g[L] += g
                    sum_g2[L] += g * g
                    grad_tot += float((g * g).sum())
                    for b in BANK_IDS:
                        grad_in[b] += float((g @ U_gpu[(b, L)]).pow(2).sum())
            opt.step()
        model.eval()
        move_tot, move_in = 0.0, {b: 0.0 for b in BANK_IDS}
        with torch.no_grad():
            for L, m in mods.items():
                dW = m.weight.detach() - pristine[L]
                move_tot += float((dW * dW).sum())
                for b in BANK_IDS:
                    move_in[b] += float((dW @ U_gpu[(b, L)]).pow(2).sum())
        # M-A3 stage-1 per layer, collapsed per bank (reused functions verbatim)
        for L in mods:
            sr = settling_ratio(sum_g[L].cpu().numpy().astype(np.float64),
                                sum_g2[L].cpu().numpy().astype(np.float64), n_steps)
            pc = per_coordinate(sr["ratio"])
            row = {"batch": B, "layer": L, "n_steps": n_steps,
                   "sample_seed": sample_seed,
                   "median_elementwise_ratio": float(np.median(
                       sr["ratio"][np.isfinite(sr["ratio"])])),
                   "n_dead": sr["n_dead"],
                   "bank_ratio": {b: float(subspace_mean(pc, banks[(b, L)]))
                                  for b in BANK_IDS}}
            rows_f.write(json.dumps(row) + "\n")
        deliv = {b: (move_in[b] / move_tot) / (grad_in[b] / grad_tot)
                 for b in BANK_IDS}
        rows_f.write(json.dumps({"batch": B, "layer": None,
                                 "sample_seed": sample_seed,
                                 "delivery": deliv,
                                 "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        rows_f.flush()
        print(f"[v3] B={B}: delivery(core32)={deliv['core32']:.3f} "
              f"({time.time()-t0:.0f}s)")
    rows_f.close()
    with open(os.path.join(out_dir, "v3_config.json"), "w") as f:
        json.dump({"ladder": ladder, "n_steps": n_steps, "lr": lr,
                   "bank_scope": "q_proj all layers; complement32 out of scope",
                   "haar_seed": 1337,
                   "sample_seed_base": sample_seed,
                   "sample_seed_rule": "default_rng(sample_seed + B)"}, f, indent=2)
