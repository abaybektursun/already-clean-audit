# W5 — Bidirectional manifold-steering of LM SAE-feature clusters (spec §2.2, §7 W5).
"""Causal test of the W4 gain-control hypothesis.

CONFIRMATORY. We steer a set of Gemma-Scope SAE latents' DECODER directions in
PaliGemma2's LM decoder residual stream during a forward, BOTH ↓ (ablate toward a
resample/mean floor) and ↑ (amplify), ALONG the data manifold (the SAE decoder
columns are the learned local manifold axes), and measure the effect on RefCOCO
(acc@0.5) and VQAv2 (soft-acc) vs a random equal-size latent set, across a dose
ladder. H_gc survives iff the suppressed-image cluster beats random with a
monotone dose-response (pre-reg §3).

Discipline (CLAUDE.md / spec hard rules):
  * NEVER zero-ablate (off-distribution, inflates effects — §5.1). Floor =
    resample- or mean-ablation of the FEATURE COMPONENT only; never zero the residual.
  * NEVER try/except to hide errors. Shape/d_model mismatch RAISES (gemma_scope).
  * Steer the SAE feature component along W_dec directions ONLY; the orthogonal
    residual + the SAE recon-error stay untouched → forward stays on-manifold.
  * PURE module (torch/numpy only, no Modal) so the analysis + the GPU harness
    share ONE intervention estimator and it is unit-importable.

The intervention (pre-reg §2), per token residual x with SAE encode f = encode(x),
target latent set S, per-latent decoder UNIT direction ŵ_i = W_dec[i]/||W_dec[i]||:

  AMPLIFY(α≥1):  x' = x + (α-1) · Σ_{i∈S} f_i · ŵ_i
                 (α=1 = no-op; push the feature component the model already uses harder)
  ABLATE(β∈[0,1]): x' = x - β · Σ_{i∈S} (f_i - r_i) · ŵ_i
                 (β=0 = no-op; β=1 = feature replaced by its resample/mean floor r_i)

f_i is the SCALAR activation (per token); the steering delta along ŵ_i uses the
feature's own present magnitude → on-manifold. r_i is the resample/mean floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ===========================================================================
# Target/control latent-set selection from W4 per-latent stats (deterministic).
# ===========================================================================

@dataclass(frozen=True)
class LatentSets:
    """The steered set + the two pre-registered controls, as latent-index arrays.

    All equal size n = |suppressed_img| (random/amplified draws capped to match)."""
    suppressed_img: np.ndarray        # TARGET: suppressed AND image-concentrated
    suppressed_all: np.ndarray        # robustness arm: all suppressed latents
    random_freq_matched: np.ndarray   # CONTROL-1: equal-size, frequency-band matched
    amplified_txt: np.ndarray         # CONTROL-2: amplified AND text-concentrated
    n: int
    meta: dict


def select_latent_sets(
    per_latent: dict,
    image_mass_fraction: np.ndarray,
    fired_mm: np.ndarray,
    seed: int = 1337,
) -> LatentSets:
    """Build the steered set + controls from the FROZEN W4 estimator output.

    per_latent: output of lm_feature_drift.classify_latents (has 'class',
                'freq_after', 'live_before'/'live_after', ...).
    image_mass_fraction: (d_sae,) per-latent multimodal image-mass fraction
                (PositionMass.image_mass_fraction) — the image-concentration axis.
    fired_mm:   (d_sae,) bool, latent fired at all in the multimodal regime.

    TARGET suppressed_img = suppressed-class latents whose own image_mass_fraction
    exceeds the PRESERVED-class latentwise mean (the empirical within-LM
    position-agnostic reference). CONTROL-1 random = equal-size, frequency-band
    matched to the target (binned by after-firing frequency → same activation-energy
    profile, a fair control). CONTROL-2 amplified_txt = amplified-class latents BELOW
    the preserved mean (text-concentrated; the directional opposite of the target).
    """
    cls = per_latent["class"]
    freq_after = per_latent["freq_after"]
    d_sae = cls.size
    rng = np.random.default_rng(seed)

    suppressed = (cls == "suppressed") & fired_mm
    amplified = (cls == "amplified") & fired_mm
    preserved = (cls == "preserved") & fired_mm

    # Empirical within-LM reference: preserved-class latentwise mean image-mass frac.
    pres_ref = float(np.mean(image_mass_fraction[preserved])) if preserved.any() else float("nan")

    sup_img = suppressed & (image_mass_fraction > pres_ref)
    sup_img_idx = np.where(sup_img)[0]
    sup_all_idx = np.where(suppressed)[0]
    n = int(sup_img_idx.size)
    if n == 0:
        raise RuntimeError(
            "No suppressed image-concentrated latents found — cannot form the W5 target. "
            f"suppressed n={int(suppressed.sum())}, preserved-ref imgfrac={pres_ref}."
        )

    # CONTROL-1: frequency-band-matched random set of size n, drawn from LIVE latents
    # that are NOT in the target. Match the target's after-firing-frequency histogram
    # (log-spaced bins) so the random set has the same activation profile (fair energy).
    live = (per_latent["live_after"]) & fired_mm
    pool = np.where(live & ~sup_img)[0]
    tgt_freq = freq_after[sup_img_idx]
    # log-spaced freq bins over the live population's range.
    lo = max(freq_after[live].min(), 1e-6)
    hi = freq_after[live].max()
    edges = np.geomspace(lo, hi, num=9)
    tgt_hist, _ = np.histogram(np.clip(tgt_freq, lo, hi), bins=edges)
    pool_bin = np.clip(np.digitize(freq_after[pool], edges) - 1, 0, len(edges) - 2)
    rand_idx_parts = []
    for b, want in enumerate(tgt_hist):
        if want == 0:
            continue
        cand = pool[pool_bin == b]
        if cand.size == 0:
            continue
        take = min(int(want), cand.size)
        rand_idx_parts.append(rng.choice(cand, size=take, replace=False))
    rand_idx = np.concatenate(rand_idx_parts) if rand_idx_parts else np.array([], dtype=int)
    # top up to n from the remaining pool if frequency-matching under-filled (fail-loud
    # if we genuinely cannot reach n).
    if rand_idx.size < n:
        remaining = np.setdiff1d(pool, rand_idx)
        if remaining.size < (n - rand_idx.size):
            raise RuntimeError(
                f"Cannot draw a size-{n} frequency-matched random control: only "
                f"{rand_idx.size + remaining.size} live non-target latents available."
            )
        rand_idx = np.concatenate([rand_idx, rng.choice(remaining, size=n - rand_idx.size, replace=False)])
    rand_idx = rand_idx[:n]

    # CONTROL-2: amplified-AND-text-concentrated (below preserved ref), size up to n.
    amp_txt = amplified & (image_mass_fraction < pres_ref)
    amp_txt_idx = np.where(amp_txt)[0]
    if amp_txt_idx.size > n:
        amp_txt_idx = rng.choice(amp_txt_idx, size=n, replace=False)

    meta = {
        "preserved_ref_image_mass_fraction": pres_ref,
        "n_suppressed_total": int(suppressed.sum()),
        "n_suppressed_img": int(sup_img_idx.size),
        "n_amplified_total": int(amplified.sum()),
        "n_amplified_txt": int(amp_txt_idx.size),
        "target_mean_image_mass_fraction": float(np.mean(image_mass_fraction[sup_img_idx])),
        "random_mean_image_mass_fraction": float(np.mean(image_mass_fraction[rand_idx])) if rand_idx.size else None,
        "amplified_txt_mean_image_mass_fraction": float(np.mean(image_mass_fraction[amp_txt_idx])) if amp_txt_idx.size else None,
        "target_mean_freq_after": float(np.mean(freq_after[sup_img_idx])),
        "random_mean_freq_after": float(np.mean(freq_after[rand_idx])) if rand_idx.size else None,
        "seed": seed,
    }
    return LatentSets(
        suppressed_img=sup_img_idx,
        suppressed_all=sup_all_idx,
        random_freq_matched=rand_idx,
        amplified_txt=amp_txt_idx,
        n=n,
        meta=meta,
    )


# ===========================================================================
# The steering forward-hook (manifold-steering; never zero).
# ===========================================================================

class SteeringHook:
    """Forward hook on one LM decoder layer that steers a target latent set in the
    residual stream, along the SAE decoder directions, in one mode/dose.

    mode = 'amplify' (dose=α≥1) or 'ablate' (dose=β∈[0,1]); 'none'/α=1/β=0 = no-op.
    floor = 'resample' (per-token resample of f_i from other batch positions) or
            'mean' (batch-mean f_i). Used only for ablate. NEVER zero (§5.1).
    positions = 'all' | 'image' | 'text' — restrict the steered token positions.

    GENERATION-ROBUST: the harness sets `prefill_image_mask` (b, T_prefill) before
    generate. The PREFILL forward (T == T_prefill, image tokens present) uses that
    mask. Autoregressive DECODE steps (T == 1, generating answer tokens) are ALWAYS
    text positions → 'image' steers nothing on decode, 'text'/'all' steer the new
    token. So a token's class is decided correctly across the full generation.

    The hook recomputes f = SAE.encode(hidden) on the fly (the SAE site IS this
    layer's output residual — gemma_scope), steers ONLY the target-direction
    component, and writes the modified hidden back into the layer output tuple."""

    def __init__(
        self,
        layer_module: torch.nn.Module,
        sae,
        target_idx: np.ndarray,
        mode: str,
        dose: float,
        floor: str = "resample",
        positions: str = "all",
    ):
        assert mode in ("amplify", "ablate", "none"), mode
        assert floor in ("resample", "mean"), floor
        assert positions in ("all", "image", "text"), positions
        self.layer = layer_module
        self.sae = sae
        self.idx = torch.as_tensor(np.asarray(target_idx, dtype=np.int64))
        self.mode = mode
        self.dose = float(dose)
        self.floor = floor
        self.positions = positions
        # Prefill per-token image mask (b, T_prefill) bool, set by the harness before
        # generate. The hook derives the steered-position mask from it + the current T
        # (see _resolve_mask) so it stays correct across the autoregressive decode.
        self.prefill_image_mask: torch.Tensor | None = None
        self._handle = None
        # Precompute unit decoder directions for the target latents: (n, d_model).
        Wd = sae.W_dec[self.idx.to(sae.W_dec.device)]            # (n, d_model)
        norm = Wd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.W_unit = (Wd / norm)                               # (n, d_model)

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args, output):
        if self.mode == "none" or (self.mode == "amplify" and self.dose == 1.0) \
                or (self.mode == "ablate" and self.dose == 0.0):
            return output  # no-op dose
        hs = output[0] if isinstance(output, tuple) else output  # (b, T, d_model)
        dev = hs.device
        idx = self.idx.to(dev)
        W_unit = self.W_unit.to(dev)                            # (n, d_model)
        b, T, d = hs.shape
        x = hs.reshape(-1, d).float()                           # (bT, d)

        # SAE feature activations for the target latents only (scalar f_i per token).
        # f = relu_gate(x @ W_enc + b_enc)[:, idx]  — reuse the JumpReLU encode.
        f_all = self.sae.encode(x)                              # (bT, d_sae)
        f = f_all[:, idx]                                       # (bT, n)

        if self.mode == "amplify":
            coeff = (self.dose - 1.0) * f                       # (bT, n)
        else:  # ablate toward the resample/mean floor (NEVER zero)
            r = self._floor_value(f)                            # (bT, n)
            coeff = -self.dose * (f - r)                        # move f → r by fraction β

        delta = coeff @ W_unit                                  # (bT, d_model)

        # Restrict to the steered positions (image/text/all), generation-robust.
        pm = self._resolve_mask(b, T, dev)                      # (b, T) bool or None
        if pm is not None:
            delta = delta * pm.reshape(-1)[:, None].to(delta.dtype)

        x_new = (x + delta).reshape(b, T, d).to(hs.dtype)
        if isinstance(output, tuple):
            return (x_new,) + tuple(output[1:])
        return x_new

    def _resolve_mask(self, b: int, T: int, dev) -> "torch.Tensor | None":
        """Per-token steered-position mask (b, T) for the CURRENT forward, robust to
        autoregressive decode. positions='all' → None (steer everything). 'image'/'text'
        use the prefill image mask when T matches the prefill length; on decode steps
        (T != T_prefill) the new tokens are answer (text) tokens → image-mask all-False,
        text-mask all-True."""
        if self.positions == "all":
            return None
        pim = self.prefill_image_mask
        if pim is not None and pim.shape[0] == b and pim.shape[1] == T:
            is_img = pim.to(dev).bool()                         # prefill: real image positions
        else:
            # decode step (or no prefill mask): generated tokens are text positions.
            is_img = torch.zeros((b, T), dtype=torch.bool, device=dev)
        return is_img if self.positions == "image" else (~is_img)

    def _floor_value(self, f: torch.Tensor) -> torch.Tensor:
        """Resample/mean floor for ablation (NEVER zero — §5.1).

        f: (bT, n) target-latent activations over all tokens in the micro-batch.
        'mean'    → column mean broadcast to every token.
        'resample'→ a permutation of the column (each token gets another token's
                    value) = resample-ablation (Conmy 2023 / Wurgaft 2026)."""
        if self.floor == "mean":
            return f.mean(0, keepdim=True).expand_as(f)
        # resample: per-column random permutation of the rows (deterministic given
        # the global torch seed set by the harness).
        perm = torch.randperm(f.shape[0], device=f.device)
        return f[perm]


def set_position_mask(hook: SteeringHook, is_image: torch.Tensor, attn: torch.Tensor):
    """Record the PREFILL image-token mask on the hook before generate (the hook then
    derives the per-step steered-position mask, decode-robust — see _resolve_mask).

    is_image (b,T) bool = image-token positions at prefill; attn unused here (the
    hook treats non-prefill steps as generated text tokens)."""
    hook.prefill_image_mask = is_image.bool()


# ===========================================================================
# Monotonicity + effect-size helpers (the pre-reg success criterion, §3).
# ===========================================================================

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no SciPy dep needed for a single pair)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def bootstrap_mean_ci(scores: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Nonparametric bootstrap mean + 95% CI over per-item scores (spec §5)."""
    scores = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    n = scores.size
    if n == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = scores[rng.integers(0, n, n)].mean()
    return {
        "mean": float(scores.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n": int(n),
    }


def paired_delta_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI on the PAIRED per-item difference a - b (same eval items, so the
    steering deltas are paired). Used for set-vs-set + dose-vs-noop contrasts."""
    a = np.asarray(scores_a, float); b = np.asarray(scores_b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    rng = np.random.default_rng(seed)
    n = a.size
    diff = a - b
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[k] = diff[idx].mean()
    return {
        "delta_mean": float(diff.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n": int(n),
    }
