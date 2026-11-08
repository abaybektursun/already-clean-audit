"""TESSERA Arm X — the pg port of the W4 crosscoder harvest/train/stats pipeline (NEW-port).

Source: `infra/modal_pg_w4_crosscoder.py` `{_train_crosscoder, _code_stats, _fvu}` (§13 names it
the PORT SOURCE). That module imports `modal` at file scope, so it cannot be imported on pg; the
three functions are carried here unchanged in their arithmetic, with the Modal-only plumbing
dropped. Their SHARED estimator — `infra/pg/delta_crosscoder.py` — is imported, not copied, so the
classification a TESSERA pair gets is the classification a PG2 pair got.

Every hyperparameter is inherited from the port source (`modal_pg_w4_crosscoder.py:112–124`), which
is what amendment A9 freezes. TESSERA does not tune any of them: a width or sparsity chosen after
seeing a lineage pair would make the emergent fraction a function of this program's choices.

  DICT_EXPANSION 16   dict_size = 16 × 2048 = 32768 at PG1's d_model
  BATCHTOPK_K   100   active latents per token
  CC_TRAIN_STEPS 4000 · CC_BATCH 2048 · CC_LR 1e-4 · WARMUP_STEPS 500 · CC_SEEDS (0, 1)

MIN_PAIRED_TOKENS is the one NEW constant, and it is a floor rather than a tuning knob: a
4000-step × 2048-token schedule over fewer than 200k distinct paired tokens is resampling the same
activations dozens of times, and the resulting dictionary describes the sample, not the models.
A pair below the floor is VOID — never trained on thin data and reported as if it were not.

No try/except (CLAUDE.md): a library or shape mismatch RAISES.
"""
from __future__ import annotations

# --- inherited verbatim from the port source (A9) -------------------------------------------
DICT_EXPANSION = 16
BATCHTOPK_K = 100
CC_LR = 1e-4
CC_TRAIN_STEPS = 4000
CC_BATCH = 2048
CC_SEEDS = (0, 1)
WARMUP_STEPS = 500
SEED = 1337
# --- NEW (A9): the harvest floor below which a pair is VOID rather than thin -----------------
MIN_PAIRED_TOKENS = 200_000


def train_crosscoder(acts_before, acts_after, layer_idx, seed, device,
                     shuffle_pairing=False, dict_size_override=None,
                     steps_override=None, lr_override=None):
    """Train a BatchTopKCrossCoder on the paired (N, 2, d) stack. Port of `_train_crosscoder`.

    `acts_before`/`acts_after` are row-aligned: token i is the SAME token in both models.
    `shuffle_pairing=True` permutes the AFTER rows, so the pairing carries no genuine before→after
    structure and any "emergent" fraction the crosscoder reports is the manufactured null.
    """
    import torch
    # `dictionary_learning/training.py` imports wandb at MODULE TOP, so trainSAE cannot be
    # imported without it even at use_wandb=False. The env AMI does not carry it; the shim is
    # idempotent and is the documented fix (REUSED from diff_harvest).
    from infra.pg.diff_harvest import ensure_wandb
    ensure_wandb()
    from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer
    from dictionary_learning.training import trainSAE

    g = torch.Generator().manual_seed(seed)
    if shuffle_pairing:
        acts_after = acts_after[torch.randperm(acts_after.shape[0], generator=g)]

    N, d = acts_before.shape
    stack = torch.stack([acts_before, acts_after], dim=1)
    if stack.shape[1] != 2:
        raise RuntimeError(f"crosscoder stack must be (N,2,d); got {tuple(stack.shape)}")
    dict_size = int(dict_size_override) if dict_size_override is not None else DICT_EXPANSION * d

    act_mean = stack.mean(0)
    act_std = stack.std(0).clamp_min(1e-6)

    steps = int(steps_override) if steps_override else CC_TRAIN_STEPS
    lr = float(lr_override) if lr_override else CC_LR

    def data_iter():
        idx_g = torch.Generator().manual_seed(seed + 1)
        for _ in range(steps):
            yield stack[torch.randint(0, N, (CC_BATCH,), generator=idx_g)]

    ae = trainSAE(
        data=data_iter(),
        trainer_config={
            "trainer": BatchTopKCrossCoderTrainer, "steps": steps,
            "activation_dim": d, "dict_size": dict_size, "k": BATCHTOPK_K,
            "layer": layer_idx, "lm_name": "tessera-pair", "num_layers": 2,
            "lr": lr, "warmup_steps": WARMUP_STEPS, "seed": seed, "device": device,
            "wandb_name": f"tessera_L{layer_idx}_s{seed}{'_shuf' if shuffle_pairing else ''}",
            "activation_mean": act_mean.to(device), "activation_std": act_std.to(device),
        },
        steps=steps, use_wandb=False, dtype=torch.float32,
    ).to("cpu")
    return ae, {"dict_size": dict_size, "activation_dim": d, "k": BATCHTOPK_K,
                "steps": steps, "lr": lr, "expansion": dict_size / d,
                "n_paired_tokens": int(N), "shuffle_pairing": bool(shuffle_pairing)}


def code_stats(ae, acts_before, acts_after, device):
    """Per-latent firing frequency over the paired stack. Port of `_code_stats`.

    The shared BatchTopK code plus a per-model decoupled encode (one side zeroed), which is the
    activation-side emergent signal — independent of the decoder-norm classifier.
    """
    import torch
    from infra.pg.delta_crosscoder import new_code_accumulator, update_code_accumulator
    N = acts_before.shape[0]
    acc = new_code_accumulator(ae.dict_size if hasattr(ae, "dict_size")
                               else ae.decoder.weight.shape[1])
    MICRO = 4096
    ae = ae.to(device).eval()
    zeros = None
    with torch.inference_mode():
        for s in range(0, N, MICRO):
            b = acts_before[s:s + MICRO].to(device)
            a = acts_after[s:s + MICRO].to(device)
            if zeros is None or zeros.shape[0] != b.shape[0]:
                zeros = torch.zeros_like(b)
            update_code_accumulator(
                acc,
                ae.encode(torch.stack([b, a], dim=1)).cpu().numpy(),
                ae.encode(torch.stack([b, zeros], dim=1)).cpu().numpy(),
                ae.encode(torch.stack([zeros, a], dim=1)).cpu().numpy())
    ae.to("cpu")
    return acc


def fvu(ae, acts_before, acts_after, device, seed=SEED):
    """Reconstruction quality per stream. Port of `_fvu`.

    A degenerate crosscoder reads FVU ≈ 1, so an "emergent ≈ 0" verdict is only meaningful when
    FVU is well below 1. This is the leg that makes the P1 gate two-sided.
    """
    import numpy as np
    import torch
    N = acts_before.shape[0]
    ntok = min(20000, N)
    sel = np.random.default_rng(seed).choice(N, ntok, replace=False)
    stack = torch.stack([acts_before[sel].to(device), acts_after[sel].to(device)], dim=1)
    ae = ae.to(device).eval()
    with torch.inference_mode():
        recon = ae.decode(ae.encode(stack))

    def _f(x, xr):
        num = ((x - xr) ** 2).sum().item()
        den = ((x - x.mean(0, keepdim=True)) ** 2).sum().item()
        return (num / den) if den > 0 else float("nan")
    out = {"fvu_before": _f(stack[:, 0], recon[:, 0]),
           "fvu_after": _f(stack[:, 1], recon[:, 1]), "n_tokens": int(ntok)}
    ae.to("cpu")
    return out


def pair_fractions(ae, acts_before, acts_after, device):
    """{emergent, lost, survival, dead} fractions for one trained crosscoder.

    WARNING (critic finding 10) — `survival_fraction` here is `1 - emergent_fraction`, which is
    NOT survival. It is the complement of the AFTER-ONLY fraction, so it says nothing about whether
    the parent's features persisted: a pair that loses 90% of the parent's features while adding
    none still scores `survival_fraction = 1.0`. The quantity that answers "did the parent's
    features persist" is `1 - lost_fraction`, built from the BEFORE-only population. The field is
    kept for artifact compatibility; no TESSERA result reports it.
    """
    from infra.pg import delta_crosscoder as DC
    dec_w = ae.decoder.weight.detach().cpu().numpy()
    nb, na = DC.decoder_norms(dec_w)
    acc = code_stats(ae, acts_before, acts_after, device)
    stats = DC.CrosscoderFeatureStats(norm_before=nb, norm_after=na,
                                      fire_before=acc.freq_before, fire_after=acc.freq_after)
    cl = DC.classify_features(stats)
    n_live = cl["n_live"]
    return {"counts": cl["counts"], "n_live": n_live,
            "emergent_fraction": cl["emergent_fraction"],
            "lost_fraction": cl["lost_fraction"],
            "survival_fraction": (1.0 - cl["emergent_fraction"]) if n_live else 0.0,
            "rel_norm_hist": cl["rel_norm_hist"]}
