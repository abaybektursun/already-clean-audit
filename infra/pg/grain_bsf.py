"""GRAIN — the block-sparse TopK featurizer + the pure grain logic
(studies/lineage/paligemma1-pi0/research/2026-07-08_grain_spec.md §3).

The program's FIRST in-house featurizer: a dictionary pair (E, D) with W latents partitioned
into contiguous blocks of size m; the forward keeps the top-K_blk blocks by L2 norm of their
coefficient sub-vector (training: batch-level top-(K_blk·batch), the exact block generalization
of the house BatchTopK; inference: block-norm > threshold, EMA-learned). **m = 1 reduces exactly
to the house BatchTopK SAE** (jkminder/dictionary_learning BatchTopKSAE + BatchTopKTrainer,
whose semantics are mirrored line-for-line below) — grain is the single moved variable across
the T2 grid. Matched active rank: K_blk × m = R_ACTIVE, identical width/optimizer/data/steps.

This module holds everything unit-testable without a model (spec §11.1 weight-free dry-runs):
the featurizer class + trainer, block subspaces / QR unions, alignment + support-spread (G3),
the G and S2 adjudicators (synthetic-table testable), Jaccard + permutation null, the
planted-block generator (T3.3 positive control), and the splice/steer pre-hooks (thin glue over
inherit_core's kwarg-robust adapter).

Design rules (CLAUDE.md): NO try/except hiding — a wrong assumption RAISES with a trace.
torch is imported at module level (house precedent: gemma_scope.py — the class needs nn.Module;
the control-plane gate is py_compile, which does not execute imports).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from infra.pg.ctrliface_core import orthonormalize, principal_angle_overlap
from infra.pg.inherit_core import _split_hidden

# ---------------------------------------------------------------------------
# Frozen T2 defaults (spec §3/§4: the W9 config as frozen defaults; grid never extended).
# ---------------------------------------------------------------------------
W_DICT = 16384
R_ACTIVE = 96                       # K_blk × m, matched across grains (divisible by every m)
M_GRID = (1, 4, 8, 16)
LR = 1e-4
STEPS = 4000
BATCH = 2048
WARMUP = 500
# BatchTopKTrainer internals, mirrored verbatim (the m=1 exact-reduction contract):
AUXK_ALPHA = 1.0 / 32
THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 1000
DEAD_TOKEN_THRESHOLD = 10_000_000   # note: inert at the frozen budget (4000×2048 < 10M tokens)


# ===========================================================================
# The featurizer. m=1 ≡ house BatchTopKSAE (init order, selection, threshold — all mirrored).
# ===========================================================================
def _unit_norm_decoder_(weight):
    """In-place unit-L2-norm decoder columns (house set_decoder_norm_to_unit_norm, dim=0)."""
    eps = torch.finfo(weight.dtype).eps
    weight.data /= weight.data.norm(dim=0, keepdim=True) + eps
    return weight.data


class BlockTopKSAE(nn.Module):
    """Block-sparse TopK dictionary. Latent j belongs to block j // m (contiguous blocks).

    encode:  z = relu(E(x − b_dec)); n_b = ‖z_block_b‖₂;
             training (use_threshold=False): keep the top-(K_blk·batch) blocks by norm across
             the FLATTENED (batch × n_blocks) norm matrix — at m=1 the norms ARE the post-relu
             values, so this is bit-identical to BatchTopKSAE's flattened top-(k·batch);
             inference (use_threshold=True): keep blocks with n_b > threshold — at m=1
             identical to the house z·(z > threshold).
    decode:  x̂ = D·z_sparse + b_dec (unit-norm decoder columns).

    `signed=True` (REGRAIN §5, BSF App. D: "no more ReLU involved") drops the rectification:
    the code is z = E(x − b_dec) and block selection ranks ‖z_block‖₂ of the SIGNED coefficients.
    Default False → every v1 bank loads and behaves bit-for-bit as before.
    """

    def __init__(self, activation_dim, dict_size, m, k_blk, signed=False):
        super().__init__()
        assert dict_size % m == 0, f"dict_size {dict_size} not divisible by m={m}"
        n_blocks = dict_size // m
        assert 1 <= k_blk <= n_blocks, f"k_blk={k_blk} outside [1, {n_blocks}]"
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)
        self.m = int(m)
        self.n_blocks = int(n_blocks)
        self.k_blk = int(k_blk)
        self.signed = bool(signed)
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        # construction ORDER mirrors the house class exactly (same RNG consumption → identical
        # m=1 init under the same manual_seed): decoder first, unit-normed, encoder = decoderᵀ.
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        _unit_norm_decoder_(self.decoder.weight)
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def block_norms(self, z):
        """(n, dict_size) coefficients → (n, n_blocks) per-block L2 norms."""
        return z.reshape(z.shape[0], self.n_blocks, self.m).norm(dim=2)

    def encode(self, x, return_active=False, use_threshold=True):
        pre = self.encoder(x - self.b_dec)
        code = pre if self.signed else nn.functional.relu(pre)
        norms = self.block_norms(code)
        if use_threshold:
            keep = norms > self.threshold
        else:
            flat = norms.flatten()
            n_keep = self.k_blk * x.shape[0]
            assert n_keep <= flat.numel(), f"k_blk·batch {n_keep} > {flat.numel()} block slots"
            top = flat.topk(n_keep, sorted=False)
            keep = (torch.zeros_like(flat, dtype=torch.bool)
                    .scatter_(0, top.indices, True).reshape(norms.shape))
        f = code * keep.repeat_interleave(self.m, dim=1).to(code.dtype)
        if return_active:
            return f, f.abs().sum(0) > 0, code
        return f

    def decode(self, f):
        return self.decoder(f) + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# ===========================================================================
# Trainer — BatchTopKTrainer.update mirrored (Adam, warmup, auxk, threshold EMA,
# decoder-tangent gradient, unit-norm renorm, geometric-median b_dec init).
# ===========================================================================
def _geometric_median(points, max_iter=100, tol=1e-5):
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


@torch.no_grad()
def _remove_parallel_grad_(weight, grad):
    """Remove the gradient component parallel to each decoder column (house util, dim=0).
    The @no_grad is LOAD-BEARING (house-exact): without it the in-place op on .grad makes the
    gradient graph-connected and persistent optimizer state drags an ever-growing autograd
    graph across steps (~15MB/step → the 2026-07-09 43GiB OOM at d=2048/W=16384)."""
    normed = weight / (weight.norm(dim=0, keepdim=True) + 1e-6)
    parallel = (grad * normed).sum(dim=0)
    grad -= parallel.unsqueeze(0) * normed
    return grad


def _lr_schedule(warmup_steps):
    """House get_lr_schedule with decay_start=None: linear warmup then constant."""
    def fn(step):
        return step / warmup_steps if step < warmup_steps else 1.0
    return fn


def train_block_sae(X, m, seed, device, dict_size=W_DICT, r_active=R_ACTIVE, steps=STEPS,
                    lr=LR, batch=BATCH, warmup=WARMUP, signed=False, freeze_b_dec=False,
                    x_scale=1.0):
    """Train ONE block-TopK dictionary on activation rows X (N, d) CPU (any float dtype).

    Frozen recipe = the W9 train_diff_sae config with the house trainer semantics; only m moves
    across the T2 grid (REGRAIN §5 additionally moves `signed`, holding data/steps/batch/LR/init
    protocol identical — the attribution factorial's confound control). Returns (ae on CPU eval
    mode, cfg dict). Fail-loud on non-finite loss (divergence → the §3 intercurrent-event rule:
    one retry at next seed, then FAILED, in the job).
    """
    assert r_active % m == 0, f"r_active {r_active} not divisible by m={m}"
    N, d = X.shape
    k_blk = r_active // m
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ae = BlockTopKSAE(d, dict_size, m, k_blk, signed=signed).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_schedule(warmup))
    since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    top_k_aux = d // 2

    idx_g = torch.Generator().manual_seed(seed + 1)
    last_loss = None
    for step in range(steps):
        # x_scale = the paper's √d input normalization (E‖x‖₂ = √d), applied in fp32 AFTER the
        # fp16 gather so no precision is lost. x_scale=1.0 reproduces the v1 recipe bit-for-bit.
        x = X[torch.randint(0, N, (batch,), generator=idx_g)].to(device).float() * float(x_scale)
        if step == 0:
            ae.b_dec.data = _geometric_median(x).to(ae.b_dec.dtype)
            if freeze_b_dec:
                # Pin the code's apex at the data's geometric median. With a LEARNED b_dec a
                # rectified block's non-negative cone can be widened arbitrarily by moving its
                # apex away from the data, which defeats the BSF cone argument for a single
                # manifold (measured, dry-run suite). Freezing isolates the cone restriction.
                ae.b_dec.requires_grad_(False)          # Adam skips params whose .grad stays None
        f, active, code = ae.encode(x, return_active=True, use_threshold=False)
        if step > THRESHOLD_START_STEP:
            with torch.no_grad():
                sel_norms = ae.block_norms(f)
                live = sel_norms[sel_norms > 0]
                min_act = live.min().float() if live.numel() else torch.tensor(0.0, device=device)
                if ae.threshold < 0:
                    ae.threshold.fill_(float(min_act))
                else:
                    ae.threshold.mul_(THRESHOLD_BETA).add_((1 - THRESHOLD_BETA) * float(min_act))
        x_hat = ae.decode(f)
        e = x - x_hat
        did_fire = torch.zeros_like(since_fired, dtype=torch.bool)
        did_fire[active] = True
        since_fired += x.shape[0]
        since_fired[did_fire] = 0
        l2 = e.pow(2).sum(dim=-1).mean()
        # auxk dead-latent revival (house get_auxiliary_loss; inert below DEAD_TOKEN_THRESHOLD)
        dead = since_fired >= DEAD_TOKEN_THRESHOLD
        if bool(dead.any()):
            k_aux = min(top_k_aux, int(dead.sum()))
            # rank dead latents by magnitude (|code| for signed; code itself is ≥0 rectified),
            # then gather the SIGNED values — house-exact when signed=False.
            rank_by = torch.where(dead[None], code.abs(), -torch.inf)
            aux_idx = rank_by.topk(k_aux, sorted=False).indices
            aux_acts = code.gather(-1, aux_idx)
            aux_full = torch.zeros_like(code).scatter_(-1, aux_idx, aux_acts)
            recon_aux = ae.decoder(aux_full)
            res = e.detach()
            l2_aux = (res.float() - recon_aux.float()).pow(2).sum(dim=-1).mean()
            denom = (res.float() - res.mean(0, keepdim=True).float()).pow(2).sum(dim=-1).mean()
            auxk = (l2_aux / denom).nan_to_num(0.0)
        else:
            auxk = torch.tensor(0.0, device=device)
        loss = l2 + AUXK_ALPHA * auxk
        assert torch.isfinite(loss), f"non-finite loss at step {step} (m={m}, seed={seed})"
        loss.backward()
        _remove_parallel_grad_(ae.decoder.weight, ae.decoder.weight.grad)
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        sched.step()
        with torch.no_grad():
            _unit_norm_decoder_(ae.decoder.weight)
        last_loss = float(loss.item())
    ae = ae.to("cpu").eval()
    cfg = {"kind": "blocktopk", "m": int(m), "k_blk": int(k_blk), "dict_size": int(dict_size),
           "activation_dim": int(d), "r_active": int(r_active), "steps": int(steps),
           "lr": float(lr), "batch": int(batch), "warmup": int(warmup), "seed": int(seed),
           "signed": bool(signed), "freeze_b_dec": bool(freeze_b_dec), "n_tokens": int(N),
           "x_scale": float(x_scale), "final_loss": last_loss,
           "threshold": float(ae.threshold.item())}
    return ae, cfg


def block_quality(ae, X, device, n_tokens=20000, seed=1337, micro=8192):
    """Held-out fit gate (T2): FVU + dead-block/dead-latent fraction + mean block-L0, threshold
    mode (the deployment mode — the splice/ablation cells run thresholded)."""
    import numpy as np
    N = X.shape[0]
    ntok = min(n_tokens, N)
    sel = np.random.default_rng(seed).choice(N, ntok, replace=False)
    ae = ae.to(device).eval()
    num = den = 0.0
    fired_lat = torch.zeros(ae.dict_size, dtype=torch.bool)
    fired_blk = torch.zeros(ae.n_blocks, dtype=torch.bool)
    l0_sum = 0.0
    mu = X[sel].float().mean(0)
    with torch.inference_mode():
        for s in range(0, ntok, micro):
            x = X[sel[s:s + micro]].to(device).float()
            f = ae.encode(x, use_threshold=True)
            rec = ae.decode(f)
            num += float(((x - rec) ** 2).sum().item())
            den += float(((x - mu.to(device)) ** 2).sum().item())
            # `f != 0`, not `f > 0`: a SIGNED latent that only ever fires negative is ALIVE.
            # Identical to the v1 path for rectified codes, where f ≥ 0 everywhere.
            fired_lat |= (f != 0).any(0).cpu()
            sel_blk = ae.block_norms(f) > 0
            fired_blk |= sel_blk.any(0).cpu()
            l0_sum += float(sel_blk.float().sum().item())
    ae = ae.to("cpu")
    assert den > 0, "degenerate held-out sample (zero variance)"
    return {"fvu": num / den, "dead_latent_frac": float((~fired_lat).float().mean().item()),
            "dead_block_frac": float((~fired_blk).float().mean().item()),
            "mean_block_l0": l0_sum / ntok, "n_tokens": int(ntok)}


# ===========================================================================
# Block geometry: subspaces, QR unions, alignment, usage, stable rank.
# ===========================================================================
def block_cols(decoder_weight, m, b):
    """Decoder columns of block b: (d, m). decoder_weight is the (d, W) Linear weight."""
    W = decoder_weight.shape[1]
    n_blocks = W // m
    assert 0 <= b < n_blocks, f"block {b} outside [0, {n_blocks})"
    return decoder_weight[:, b * m:(b + 1) * m]


def block_subspace(decoder_weight, m, b):
    """Orthonormalized span of block b's decoder columns (QR), fp64 (d, m)."""
    return orthonormalize(block_cols(decoder_weight, m, b).double())


def union_Q(decoder_weight, m, blocks):
    """Concatenated-QR span of a block union (spec §3): fp64 (d, len(blocks)·m) orthonormal.
    Fail-loud on cross-block rank deficiency (a degenerate union is a finding, not a fallback)."""
    assert len(blocks) >= 1 and len(set(blocks)) == len(blocks), f"bad union {blocks}"
    cols = torch.cat([block_cols(decoder_weight, m, b).double() for b in blocks], dim=1)
    return orthonormalize(cols)


def block_alignment(decoder_weight, m, V):
    """G1a ranking statistic per block: ‖Q_bᵀ V‖²_F / m (mean squared principal cosine against
    the frozen read bank V (d, k) orthonormal). Returns fp64 (n_blocks,)."""
    Vd = torch.as_tensor(V, dtype=torch.float64)
    n_blocks = decoder_weight.shape[1] // m
    out = torch.empty(n_blocks, dtype=torch.float64)
    for b in range(n_blocks):
        Qb = block_subspace(decoder_weight, m, b)
        out[b] = float(((Qb.transpose(0, 1) @ Vd) ** 2).sum().item()) / m
    return out


def block_usage(ae, X, labels, device, n_types=3, micro=8192):
    """Per-block usage on labeled activation rows: selection counts per position type +
    overall frequency + mean latent |activation|. labels: (N,) int in [0, n_types)
    (0=image, 1=text, 2=BOS — the §1 pre-mortem position-type covariate).
    Returns {counts (n_blocks, n_types), freq (n_blocks,), latent_mean_act (W,)}."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    assert labels.shape[0] == X.shape[0], f"labels {labels.shape} vs X {X.shape}"
    assert int(labels.min()) >= 0 and int(labels.max()) < n_types
    ae = ae.to(device).eval()
    counts = torch.zeros(ae.n_blocks, n_types, dtype=torch.float64)
    lat_sum = torch.zeros(ae.dict_size, dtype=torch.float64)
    n = X.shape[0]
    with torch.inference_mode():
        for s in range(0, n, micro):
            x = X[s:s + micro].to(device).float()
            lab = labels[s:s + micro]
            f = ae.encode(x, use_threshold=True)
            selected = (ae.block_norms(f) > 0).cpu()            # (b, n_blocks)
            for t in range(n_types):
                rows = lab == t
                if bool(rows.any()):
                    counts[:, t] += selected[rows].double().sum(0)
            lat_sum += f.abs().sum(0).double().cpu()
    ae = ae.to("cpu")
    freq = counts.sum(1) / n
    return {"counts": counts, "freq": freq, "latent_mean_act": lat_sum / n}


def block_stable_rank(decoder_weight, m, latent_scale):
    """S1 atlas: stable rank ‖·‖²_F/‖·‖²₂ of each ACTIVATION-WEIGHTED decoder block (columns
    scaled by the latent's mean |activation| — the dimensionality the block USES). Dead blocks
    (zero scale) → 0.0 (filtered by usage in the atlas). Returns fp64 (n_blocks,)."""
    scale = torch.as_tensor(latent_scale, dtype=torch.float64)
    Wt = torch.as_tensor(decoder_weight, dtype=torch.float64)
    assert scale.shape[0] == Wt.shape[1]
    n_blocks = Wt.shape[1] // m
    out = torch.zeros(n_blocks, dtype=torch.float64)
    for b in range(n_blocks):
        M = Wt[:, b * m:(b + 1) * m] * scale[b * m:(b + 1) * m]
        fro2 = float((M ** 2).sum().item())
        if fro2 <= 0:
            continue
        spec2 = float(torch.linalg.svdvals(M)[0].item()) ** 2
        out[b] = fro2 / spec2
    return out


# ===========================================================================
# G3 support spread (the H_oblique statistic) — from banks, no GPU.
# ===========================================================================
def support_spread(align_energies, n_active):
    """Cumulative share of the read's block-alignment energy vs #blocks (sorted descending).
    Frozen H_oblique input: n_for_90 = smallest n capturing ≥90% of the total alignment energy;
    frac_active_for_90 = n_for_90 / n_active (H_oblique fires iff > 0.25, spec §5.2)."""
    e = torch.as_tensor(align_energies, dtype=torch.float64)
    assert e.ndim == 1 and e.numel() >= 1 and int(n_active) >= 1
    total = float(e.sum().item())
    assert total > 0, "zero alignment energy — read bank vs dictionary mismatch"
    cum = torch.sort(e, descending=True).values.cumsum(0) / total
    n90 = int((cum >= 0.9).nonzero()[0].item()) + 1
    return {"cum_share": [float(x) for x in cum], "n_for_90": n90,
            "frac_active_for_90": n90 / float(n_active), "n_active": int(n_active)}


# ===========================================================================
# Frozen adjudicators (§5.2, §6-S2) — pure tables, every label reachable (dry-run-enforced).
# ===========================================================================
G_LABELS = ("H_block-grain", "H_sparse-not-block", "H_capacity-artifact", "H_oblique",
            "H_distributed-all-grains")


def adjudicate_grain(*, effect_bstar, effect_full, effect_em, effect_grain1, effect_rand_mean,
                     gap_cilo, delta_G, rho, bh_pass, support_frac_90, oblique_frac=0.25):
    """The frozen Arm-G decision table for one read. Effects are RF drops (1 − RF_cell, dead-
    model-floor convention); gap_cilo = CI-lower of (RF(random-union band) − RF(B*)).

      compact  = effect(B*) ≥ ρ·effect(full)  AND  gap CI-lo > δ_G
                 AND the em-random span does NOT reproduce B* (within δ_G)  AND BH survives
      compact ∧ m=1-union matches B* (within δ_G)  → H_sparse-not-block
      compact ∧ otherwise                          → H_block-grain
      ¬compact ∧ random unions crater (≥ ρ·full)   → H_capacity-artifact
      ¬compact ∧ support_frac_90 > oblique_frac    → H_oblique
      else                                         → H_distributed-all-grains
    """
    assert effect_full > 0, "full-read effect ≤ 0 — no dynamic range (UNRUN upstream, not here)"
    em_reproduces = effect_em >= effect_bstar - delta_G
    grain1_matches = effect_grain1 >= effect_bstar - delta_G
    compact = ((effect_bstar >= rho * effect_full) and (gap_cilo > delta_G)
               and (not em_reproduces) and bool(bh_pass))
    if compact:
        return "H_sparse-not-block" if grain1_matches else "H_block-grain"
    if effect_rand_mean >= rho * effect_full:
        return "H_capacity-artifact"
    if support_frac_90 > oblique_frac:
        return "H_oblique"
    return "H_distributed-all-grains"


S2_LABELS = ("H_basis-ambiguity", "H_genuinely-family")


def adjudicate_s2(*, jaccard_cilo, null_p95, subspace_canonicity, low_canonicity_max=0.5):
    """Frozen S2 rule: H_basis-ambiguity iff the across-draw block-set Jaccard CI-lo beats the
    permutation-null p95 at ≥2× WHILE the recomputed subspace-overlap canonicity stays low
    (< low_canonicity_max — 'at its artifact level', the ENDOW-F family-valued regime)."""
    assert null_p95 >= 0
    if (jaccard_cilo >= 2.0 * null_p95) and (subspace_canonicity < low_canonicity_max):
        return "H_basis-ambiguity"
    return "H_genuinely-family"


# ===========================================================================
# Jaccard + permutation null (S2 machinery).
# ===========================================================================
def jaccard(a, b):
    A, B = set(a), set(b)
    assert A and B, "empty block set in Jaccard"
    return len(A & B) / len(A | B)


def pairwise_jaccard(sets):
    """Mean pairwise Jaccard over ≥2 block sets (the across-draw agreement statistic)."""
    assert len(sets) >= 2, f"need ≥2 draws, got {len(sets)}"
    vals = [jaccard(sets[i], sets[j])
            for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(sum(vals) / len(vals)), vals


def jaccard_null(n_blocks, size, n_draws, seed):
    """Permutation null: pairwise Jaccard of independent uniform size-`size` subsets of
    [0, n_blocks). Returns a fp64 tensor of n_draws values (job takes p95 as the band)."""
    g = torch.Generator().manual_seed(int(seed))
    out = torch.empty(n_draws, dtype=torch.float64)
    for i in range(n_draws):
        a = torch.randperm(n_blocks, generator=g)[:size].tolist()
        b = torch.randperm(n_blocks, generator=g)[:size].tolist()
        out[i] = jaccard(a, b)
    return out


# ===========================================================================
# Planted-structure positive control (T3.3, weight-free): block-sparse synthetic data.
# ===========================================================================
def gen_planted(d, n_planted, m_true, n_samples, k_active, seed, noise=0.05):
    """Synthetic block-sparse data: n_planted mutually-orthogonal m_true-dim subspaces; each
    sample activates k_active of them with N(0,1) coefficients + isotropic noise.
    Returns (X (n, d) fp32, [U_i (d, m_true) fp64 orthonormal])."""
    assert n_planted * m_true <= d, "planted subspaces do not fit in d"
    g = torch.Generator().manual_seed(int(seed))
    basis = orthonormalize(torch.randn(d, n_planted * m_true, generator=g, dtype=torch.float64))
    U = [basis[:, i * m_true:(i + 1) * m_true].contiguous() for i in range(n_planted)]
    X = torch.zeros(n_samples, d, dtype=torch.float64)
    for i in range(n_samples):
        active = torch.randperm(n_planted, generator=g)[:k_active]
        for a in active.tolist():
            X[i] += U[a] @ torch.randn(m_true, generator=g, dtype=torch.float64)
    X += noise * torch.randn(n_samples, d, generator=g, dtype=torch.float64)
    return X.float(), U


def best_block_overlap(decoder_weight, m, U):
    """max over dictionary blocks of principal_angle_overlap(block subspace, planted U) —
    the planted-recovery readout (must be ≫ the Haar band for the block grain, T3.3)."""
    n_blocks = decoder_weight.shape[1] // m
    best, best_b = -1.0, -1
    for b in range(n_blocks):
        ov, _ = principal_angle_overlap(block_subspace(decoder_weight, m, b),
                                        torch.as_tensor(U, dtype=torch.float64))
        if ov > best:
            best, best_b = ov, b
    return float(best), int(best_b)


def matched_span_capture(decoder_weight, U, top_r):
    """The m=1 fragmentation comparator: orthonormal span of the top_r single latents most
    aligned to U (by per-column ‖u_colᵀU‖) → principal-angle overlap with U at matched rank."""
    Wt = torch.as_tensor(decoder_weight, dtype=torch.float64)
    Ud = torch.as_tensor(U, dtype=torch.float64)
    score = ((Wt / Wt.norm(dim=0, keepdim=True)).transpose(0, 1) @ Ud).pow(2).sum(1)
    idx = torch.topk(score, top_r).indices
    ov, _ = principal_angle_overlap(orthonormalize(Wt[:, idx]), Ud)
    return float(ov)


# ===========================================================================
# PG1 hooks (thin glue over the inherit_core kwarg-robust adapter; spec §3 NEW-glue).
# ===========================================================================
def make_pg1_pre_hook_splice(ae):
    """Reconstruction splice: the residual ENTERING layers[L] ← x̂(x) = decode(encode(x)),
    threshold mode (per-row — no batch-size coupling), fp32 compute, cast back. The T3.1
    fidelity-gate intervention and the carrier of every dictionary-side cell."""
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        flat = hs.reshape(-1, hs.shape[-1]).float()
        rec = ae.decode(ae.encode(flat, use_threshold=True))
        return rebuild(rec.reshape(hs.shape).to(hs.dtype))
    return hook


def make_pg1_pre_hook_passthrough():
    """T3.2 inertness control: rebuild with the UNTOUCHED hidden states — must be a bitwise
    no-op on the logits (adapter-wiring tripwire)."""
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        return rebuild(hs)
    return hook


def make_pg1_pre_hook_steer(ae, block, direction, alpha_sigma):
    """S3 steering: splice with the block's code shifted by alpha_sigma·direction (an m-vector
    in block coordinates) BEFORE decode — the ±ασ traversal of a block principal coordinate."""
    dvec = torch.as_tensor(direction, dtype=torch.float32)
    assert dvec.shape == (ae.m,), f"direction {tuple(dvec.shape)} != (m={ae.m},)"
    lo, hi = block * ae.m, (block + 1) * ae.m

    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        flat = hs.reshape(-1, hs.shape[-1]).float()
        f = ae.encode(flat, use_threshold=True)
        f[:, lo:hi] += float(alpha_sigma) * dvec.to(f.device)
        rec = ae.decode(f)
        return rebuild(rec.reshape(hs.shape).to(hs.dtype))
    return hook


@contextlib.contextmanager
def splice_at(layers, L, ae):
    """Reconstruction-splice context on layers[L] (the ablate_at/replace_at sibling)."""
    h = layers[L].register_forward_pre_hook(make_pg1_pre_hook_splice(ae), with_kwargs=True)
    yield
    h.remove()


@contextlib.contextmanager
def steer_at(layers, L, ae, block, direction, alpha_sigma):
    h = layers[L].register_forward_pre_hook(
        make_pg1_pre_hook_steer(ae, block, direction, alpha_sigma), with_kwargs=True)
    yield
    h.remove()


# ===========================================================================
# Bank I/O (T4: load-never-retrain downstream, the B1 discipline).
# ===========================================================================
def save_dict_bank(ae, cfg, quality, path, data_manifest_sha):
    """Bank one trained dictionary: state_dict + cfg + fit quality + data manifest sha."""
    payload = {"state_dict": ae.state_dict(), "cfg": cfg, "quality": quality,
               "data_manifest_sha": str(data_manifest_sha)}
    torch.save(payload, path)
    return payload


def load_dict_bank(path):
    """Load + validate a banked BlockTopKSAE. Returns (ae eval CPU, cfg, quality, manifest_sha).
    `signed` defaults False for v1 banks (written before the flag existed) — bit-for-bit reuse.
    Grassmannian banks (cfg['kind']=='grassmann') load via regrain_core.load_bank_any."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    assert cfg.get("kind", "blocktopk") == "blocktopk", \
        f"{path} is a '{cfg['kind']}' bank — use regrain_core.load_bank_any"
    ae = BlockTopKSAE(cfg["activation_dim"], cfg["dict_size"], cfg["m"], cfg["k_blk"],
                      signed=cfg.get("signed", False))
    ae.load_state_dict(payload["state_dict"])
    ae.eval()
    assert float(ae.threshold.item()) > 0, f"banked dict has no learned threshold ({path})"
    return ae, cfg, payload["quality"], payload["data_manifest_sha"]
