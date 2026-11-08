"""W9 Diff-SAE — a BatchTopK sparse autoencoder trained DIRECTLY on the activation delta
Δa = a_after − a_before (Bussmann et al. 2412.06410; activation-difference-SAE method,
science-of-finetuning Diffing Toolkit). Unlike the W4 crosscoder (which must spend capacity on
the large shared structure of both models), the Diff-SAE allocates its ENTIRE dictionary to the
change → the sharper lens on the rewrite's content (prereg §3).

Uses the SAME maintained library + trainSAE wrapper as the W4 crosscoder (jkminder/
dictionary_learning: BatchTopKSAE + BatchTopKTrainer) so the two instruments share training
regime and are fairly comparable.

Discipline: fail-loud (no try/except hiding); heavy imports inside functions (module py_compiles
without torch on the control plane).
"""
from __future__ import annotations

# Diff-SAE training regime. k matches the crosscoder BatchTopK budget (BATCHTOPK_K=100) so the
# two instruments sit in the same L0 band. dict_size = expansion × d_model (d_model=2304 for
# Gemma-2-2B). C4 width-stability (prereg §5) sweeps ≥2 expansions.
DIFFSAE_K = 100
DIFFSAE_LR = 1e-4
DIFFSAE_TRAIN_STEPS = 4000
DIFFSAE_BATCH = 2048
DIFFSAE_WARMUP = 500
DEFAULT_EXPANSIONS = (16, 32)        # C4 ≥2 widths
SEED = 1337


def train_diff_sae(delta, layer_idx, seed, device, expansion=16, dict_size_override=None,
                   steps=DIFFSAE_TRAIN_STEPS, k=DIFFSAE_K):
    """Train ONE BatchTopK Diff-SAE on the delta stack.

    delta: (N, d) float32 CPU tensor of Δa = a_after − a_before, row-aligned per token.
    Returns (ae on CPU, cfg dict). The decoder is unit-norm-constrained by the lib; feature j's
    decoder direction = ae.decoder.weight[:, j] (a d-vector).
    """
    import torch
    from dictionary_learning.dictionary import BatchTopKSAE
    from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
    from dictionary_learning.training import trainSAE

    N, d = delta.shape
    dict_size = int(dict_size_override) if dict_size_override is not None else int(expansion) * d

    def data_iter():
        idx_g = torch.Generator().manual_seed(seed + 1)
        for _ in range(steps):
            idx = torch.randint(0, N, (DIFFSAE_BATCH,), generator=idx_g)
            yield delta[idx]

    # NOTE: deliberately NO activation_mean/std. BatchTopKTrainer.update() sets
    # b_dec=geometric_median(x) on the RAW step-0 batch BEFORE normalizing — passing mean/std
    # then puts b_dec in raw scale while the encoder sees normalized acts (encode(x_norm-b_dec)),
    # a mismatch that starves the encoder (dead latents, FVU≫1). Training on raw Δa with b_dec
    # median-centering is the standard BatchTopK recipe (Bussmann 2412.06410) and keeps decoder
    # directions in the raw activation space (so ρ_novel's before-PCA projection is in-coordinate).
    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "steps": steps, "activation_dim": d, "dict_size": dict_size, "k": k,
        "layer": layer_idx, "lm_name": "diff__gemma-2-2b__paligemma2-3b-pt",
        "lr": DIFFSAE_LR, "warmup_steps": DIFFSAE_WARMUP, "seed": seed, "device": device,
        "wandb_name": f"diffsae_L{layer_idx}_d{dict_size}_s{seed}",
    }
    ae = trainSAE(data=data_iter(), trainer_config=trainer_cfg, steps=steps,
                  use_wandb=False, dtype=torch.float32).to("cpu")
    cfg = {"dict_size": dict_size, "activation_dim": d, "k": k, "steps": steps,
           "lr": DIFFSAE_LR, "expansion": dict_size / d, "n_tokens": int(N), "seed": seed,
           "layer": layer_idx}
    return ae, cfg


def diff_sae_quality(ae, delta, device, n_tokens=20000, seed=SEED):
    """Fit-quality gate (prereg §5): FVU on Δa + dead-latent fraction. FVU is a FIT gate, NOT a
    faithfulness metric (faithfulness = R_KL, diff_metrics). A degenerate fit (FVU≈1 or many dead
    latents) is re-trained, never silently reported."""
    import numpy as np
    import torch
    N = delta.shape[0]
    ntok = min(n_tokens, N)
    sel = np.random.default_rng(seed).choice(N, ntok, replace=False)
    x = delta[sel].to(device)
    ae = ae.to(device).eval()
    with torch.inference_mode():
        code = ae.encode(x, use_threshold=True)
        recon = ae.decode(code)
    num = ((x - recon) ** 2).sum().item()
    den = ((x - x.mean(0, keepdim=True)) ** 2).sum().item()
    fvu = (num / den) if den > 0 else float("nan")
    fired = (code > 0).any(0).float()                       # (dict_size,)
    dead_frac = float((fired == 0).float().mean().item())
    mean_l0 = float((code > 0).float().sum(1).mean().item())
    ae = ae.to("cpu")
    return {"fvu": float(fvu), "dead_latent_frac": dead_frac, "mean_l0": mean_l0,
            "n_tokens": int(ntok), "dict_size": int(code.shape[1])}


def decoder_directions(ae):
    """Return (dict_size, d) UNIT-norm Diff-SAE decoder directions (feature j = column j of the
    decoder Linear weight). These are the candidate 'directions of change' ρ_novel tests."""
    import torch
    W = ae.decoder.weight.detach().cpu().float()            # (d, dict_size)
    dirs = W.T                                              # (dict_size, d)
    return torch.nn.functional.normalize(dirs, dim=1)


def encode_delta(ae, delta, device, micro=8192):
    """Encode Δa → (N, dict_size) sparse codes (threshold mode, inference). CPU result."""
    import torch
    ae = ae.to(device).eval()
    out = []
    with torch.inference_mode():
        for s in range(0, delta.shape[0], micro):
            code = ae.encode(delta[s:s+micro].to(device), use_threshold=True)
            out.append(code.cpu())
    ae = ae.to("cpu")
    return torch.cat(out, 0)
