# W4 Delta-Crosscoder — ADJUDICATE re-weighting vs new features (spec §7 W4, §2.2).
"""The instrument the frozen text-SAE (W4 lm_feature_drift.py) structurally cannot be.

W4's Gemma-Scope lens runs a *frozen text* SAE on PG2-LM activations: any genuinely
NEW multimodal feature lives in that SAE's FVU≈0.39 residual — invisible as a latent.
A **crosscoder trained on BEFORE+AFTER activations** (Lindsey et al. 2024; BatchTopK +
delta loss, 2603.04426; sparsity-artefact caveat 2504.02922) CAN represent and detect
new features, because its dictionary is fit jointly to both models' activations.

This module is PURE (numpy/torch only, no Modal) so it is unit-importable and the
heavy harvest+train (modal_pg_w4_crosscoder.py) and the analysis share ONE estimator.

THE CONTRAST (spec "What to do" §1):
  AFTER  = PG2-3b-pt-224 `model.language_model` residual at layer L.
  BEFORE = the SAME PaliGemma forward but with the decoder reset to base gemma-2-2b
           (contrast_a1._lm_swap_state_dict; vision_tower+projector+embeddings KEPT),
           residual at the same layer L, SAME multimodal image-token inputs.
  → both process identical (b, T, d_model) activations at identical positions; only the
    decoder weights differ. Perfectly PAIRED per token. The crosscoder sees a 2-layer
    stack (before, after) and learns a shared dictionary.

CLASSIFICATION (canonical crosscoder relative-decoder-norm strength, 2024/2504.02922):
  Each latent j has a per-model decoder vector. Its decoder-norm strength
      r_j = ||W_dec[AFTER, j]|| / (||W_dec[BEFORE, j]|| + ||W_dec[AFTER, j]||)
  r≈1 → the feature reconstructs ONLY the after model → EMERGENT (after-only).
  r≈0 → reconstructs ONLY before → LOST (before-only).
  r≈0.5 → SHARED. Thresholds (epsilon) fixed on the N4-shuffle null, NOT the real
  contrast (blind-analysis §6).

ADJUDICATION GATES (spec §2.2 / §4):
  * N2 method-seed: ≥2 crosscoder seeds — does the emergent SET survive (Jaccard)?
  * N4 shuffle:     shuffle before/after pairing → emergent partition must collapse to
                    chance (this is the achievable null for the emergent fraction).
  * §2.2 direct-activation cross-check: project each after-only latent's AFTER decoder
                    direction onto BEFORE activations → if it is genuinely active in the
                    before stream too, it is crosscoder SHRINKAGE, not a real new feature.
                    Significance by permutation; FDR-controlled.

Discipline (CLAUDE.md / spec hard rules):
  * Fail loud — shape/lib mismatch RAISES, never a silent reshape or try/except.
  * Delta-Crosscoder / BatchTopK, never vanilla-L1 (L1 manufactures false unique feats).
  * NEVER spin — a real emergent population MUST survive seed + shuffle + cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Layer order in the 2-model crosscoder stack. Index 0 = BEFORE, index 1 = AFTER.
# Fixed here so every downstream norm/class read agrees with the harvest stacking.
BEFORE = 0
AFTER = 1
NUM_LAYERS = 2

# --- classification thresholds (set on the N4-shuffle null per §6; defaults are
#     the conventional crosscoder cuts, overridable from the run config) ----------
# A latent is "after-only/emergent" if its decoder-norm strength r >= this, AND it is
# a live latent (fires above the dead floor). "Before-only/lost" if r <= 1 - this.
# Anthropic's crosscoder writeup uses ~0.9/0.1; we recompute the realized null and
# report the emergent fraction MINUS the null at this same cut.
RELNORM_EMERGENT_MIN = 0.90
RELNORM_LOST_MAX = 0.10
# Dead-latent floor: a latent firing on fewer than this fraction of tokens (in EITHER
# model's encode) is excluded from the live population the taxonomy is computed over.
DEAD_FIRE_FLOOR = 1e-3


# ===========================================================================
# Per-latent decoder-norm decomposition (the shared/emergent/lost classifier).
# ===========================================================================

@dataclass
class CrosscoderFeatureStats:
    """Per-latent crosscoder statistics. All arrays are length dict_size.

    norm_before / norm_after : per-model decoder vector norm of each latent
        (= ||W_dec[layer, j, :]||_2). Drives the relative-norm classifier.
    fire_before / fire_after : fraction of tokens the latent's code is active in the
        BEFORE / AFTER half of the paired stream (the live-feature filter).
    """
    norm_before: np.ndarray
    norm_after: np.ndarray
    fire_before: np.ndarray
    fire_after: np.ndarray

    @property
    def rel_norm(self) -> np.ndarray:
        """Decoder-norm strength r = norm_after / (norm_before + norm_after).
        r→1 after-only, r→0 before-only, r→0.5 shared. 0.5 where both norms are 0
        (a dead latent; filtered out separately by the live mask)."""
        denom = self.norm_before + self.norm_after
        out = np.full_like(denom, 0.5)
        nz = denom > 0
        out[nz] = self.norm_after[nz] / denom[nz]
        return out

    @property
    def live(self) -> np.ndarray:
        """Latent is live iff it fires above the dead floor in EITHER model."""
        return (self.fire_before >= DEAD_FIRE_FLOOR) | (self.fire_after >= DEAD_FIRE_FLOOR)


def decoder_norms(decoder_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-model per-latent decoder L2 norms from a crosscoder decoder weight.

    decoder_weight: (num_layers=2, dict_size, activation_dim). Layer 0 = BEFORE,
    layer 1 = AFTER (the harvest stacking contract). Returns (norm_before, norm_after),
    each (dict_size,). Fails loud on a wrong-shaped weight (no silent reshape)."""
    w = np.asarray(decoder_weight)
    if w.ndim != 3 or w.shape[0] != NUM_LAYERS:
        raise ValueError(
            f"decoder weight must be (2, dict_size, activation_dim); got {w.shape}. "
            f"The crosscoder must be the 2-model (before, after) stack.")
    norms = np.linalg.norm(w, axis=2)            # (2, dict_size)
    return norms[BEFORE], norms[AFTER]


def classify_features(stats: CrosscoderFeatureStats,
                      emergent_min: float = RELNORM_EMERGENT_MIN,
                      lost_max: float = RELNORM_LOST_MAX) -> dict:
    """Classify each LIVE latent shared / emergent / lost by decoder-norm strength.

    Returns dict of arrays + counts. Latents not live are 'dead' (excluded from the
    fraction denominators). The emergent fraction = n_emergent / n_live."""
    r = stats.rel_norm
    live = stats.live
    cls = np.array(["dead"] * r.size, dtype=object)
    cls[live] = "shared"
    cls[live & (r >= emergent_min)] = "emergent"
    cls[live & (r <= lost_max)] = "lost"

    n_live = int(live.sum())
    counts = {c: int((cls == c).sum()) for c in ("shared", "emergent", "lost", "dead")}
    # rel-norm histogram over LIVE latents (closes the critic's threshold-robustness gap:
    # is emergent=0 robust, or is there an after-dominant bump at r=0.7-0.85 below the 0.9 cut?).
    if n_live:
        hc, he = np.histogram(r[live], bins=50, range=(0.0, 1.0))
        rel_norm_hist = {"counts": hc.tolist(), "edges": he.tolist()}
    else:
        rel_norm_hist = {"counts": [], "edges": []}
    return {
        "class": cls,
        "rel_norm": r,
        "live": live,
        "counts": counts,
        "n_live": n_live,
        "emergent_fraction": (counts["emergent"] / n_live) if n_live else 0.0,
        "lost_fraction": (counts["lost"] / n_live) if n_live else 0.0,
        "rel_norm_hist": rel_norm_hist,
    }


# ===========================================================================
# Streamed code-activation accumulator (firing frequency per latent, per model).
# Run the crosscoder's encode on a paired (b, 2, d) block; the BatchTopK code is a
# single (b, dict_size) vector shared across the 2 models, but we also want per-model
# *reconstruction-side* activity — for the live filter we use the shared code firing
# under the standard "use a model's own activation through the shared dictionary".
# We accumulate the shared-code fire rate AND a per-model fire rate from encoding each
# model's half alone (decoupled encode), so a latent that only the AFTER model drives
# is visible as fire_after>0, fire_before≈0 (a second, activation-side emergent signal).
# ===========================================================================

@dataclass
class CodeAccumulator:
    n_tokens: int
    fire_shared: np.ndarray          # (dict_size,) shared-code fire count
    fire_before: np.ndarray          # (dict_size,) fire count encoding BEFORE-half alone
    fire_after: np.ndarray           # (dict_size,) fire count encoding AFTER-half alone

    @property
    def freq_shared(self) -> np.ndarray:
        return self.fire_shared / max(self.n_tokens, 1)

    @property
    def freq_before(self) -> np.ndarray:
        return self.fire_before / max(self.n_tokens, 1)

    @property
    def freq_after(self) -> np.ndarray:
        return self.fire_after / max(self.n_tokens, 1)


def new_code_accumulator(dict_size: int) -> CodeAccumulator:
    z = lambda: np.zeros(dict_size, dtype=np.float64)
    return CodeAccumulator(0, z(), z(), z())


def update_code_accumulator(acc: CodeAccumulator,
                            code_shared: np.ndarray,
                            code_before: np.ndarray,
                            code_after: np.ndarray) -> None:
    """Fold (n, dict_size) code activations (>0 = active) into the accumulator."""
    for name, c in (("shared", code_shared), ("before", code_before), ("after", code_after)):
        if c.ndim != 2:
            raise ValueError(f"{name} code must be (n_tokens, dict_size); got {c.shape}")
    acc.n_tokens += int(code_shared.shape[0])
    acc.fire_shared += (code_shared > 0).sum(0)
    acc.fire_before += (code_before > 0).sum(0)
    acc.fire_after += (code_after > 0).sum(0)


# ===========================================================================
# N4 shuffle null + N2 method-seed survival (the adjudication gates).
# ===========================================================================

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def emergent_set(cls: np.ndarray) -> set:
    return set(np.where(cls == "emergent")[0].tolist())


def method_seed_survival(cls_by_seed: list[np.ndarray]) -> dict:
    """N2: across ≥2 crosscoder seeds, does the EMERGENT set survive?

    Reports pairwise Jaccard of the emergent latent sets and the size of the
    intersection (the seed-stable emergent core). A method-seed-robust emergent
    population has high Jaccard; a sparsity artefact reshuffles with the seed."""
    sets = [emergent_set(c) for c in cls_by_seed]
    n = len(sets)
    pair_jacc = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_jacc.append(jaccard(sets[i], sets[j]))
    inter = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    return {
        "n_seeds": n,
        "emergent_counts_per_seed": [len(s) for s in sets],
        "pairwise_jaccard": pair_jacc,
        "mean_pairwise_jaccard": float(np.mean(pair_jacc)) if pair_jacc else float("nan"),
        "intersection_size": len(inter),
        "union_size": len(union),
        "seed_stable_core": sorted(inter),
    }


def emergent_fraction_minus_null(real_fraction: float,
                                 shuffle_fractions: list[float],
                                 seed_fractions: list[float]) -> dict:
    """The headline number: emergent fraction MINUS the achievable null.

    achievable null = the emergent fraction the SAME pipeline reports when the
    before/after pairing is broken (N4 shuffle) — that is the fraction a crosscoder
    manufactures with NO genuine model difference. We also report the method-seed
    spread (N2) as a second noise band. The signed excess is real_fraction − null_mean."""
    shuf = np.asarray(shuffle_fractions, dtype=float)
    null_mean = float(shuf.mean()) if shuf.size else 0.0
    null_max = float(shuf.max()) if shuf.size else 0.0
    seeds = np.asarray(seed_fractions, dtype=float)
    return {
        "real_emergent_fraction": real_fraction,
        "real_emergent_fraction_seed_mean": float(seeds.mean()) if seeds.size else real_fraction,
        "real_emergent_fraction_seed_std": float(seeds.std()) if seeds.size else 0.0,
        "shuffle_null_mean": null_mean,
        "shuffle_null_max": null_max,
        "shuffle_null_per_seed": shuffle_fractions,
        "excess_over_null_mean": real_fraction - null_mean,
        "excess_over_null_max": real_fraction - null_max,
    }


# ===========================================================================
# §2.2 direct-activation cross-check (guard against crosscoder SHRINKAGE).
# ===========================================================================

def direct_activation_crosscheck(
    after_decoder_dirs: np.ndarray,    # (n_candidates, activation_dim) unit AFTER dec dirs
    before_acts: np.ndarray,           # (n_tokens, activation_dim) BEFORE residual acts
    after_acts: np.ndarray,            # (n_tokens, activation_dim) AFTER residual acts
    n_perm: int = 1000,
    seed: int = 0,
    fdr_q: float = 0.05,
) -> dict:
    """For each candidate after-only latent, project its AFTER decoder direction onto
    the BEFORE activations and onto the AFTER activations. A GENUINE emergent feature
    is (near-)absent in BEFORE: its before-projection energy is ~0 while after is high
    (absence ratio before/after ≪ 1). A SHRINKAGE false-positive is present in BOTH
    (the crosscoder merely allocated it to the after layer due to sparsity) → before and
    after projection energy are comparable (absence ratio ≈ 1).

    Test statistic per candidate = absence ratio = before_energy / after_energy. A
    genuine emergent feature has a ratio in the LOW tail. The null is the absence-ratio
    distribution of RANDOM unit directions through the SAME two activation streams: a
    random direction is, by construction, equally (un)present in before and after
    (both streams share global covariance + isotropic noise), so its absence ratio is
    centred near 1. A candidate SURVIVES iff its absence ratio is significantly BELOW the
    random-direction ratios (one-sided), FDR-controlled — i.e. it is *more selectively
    after* than chance, the signature the after layer's decoder genuinely needs but the
    before layer does not. This is the correct guard against shrinkage: it asks whether
    the direction is differentially present, not merely whether before-energy is small.

    Returns per-candidate before/after energy, absence ratio, the permutation p, and the
    count surviving FDR.
    """
    D = after_decoder_dirs
    if D.ndim != 2:
        raise ValueError(f"after_decoder_dirs must be (n_cand, d); got {D.shape}")
    if before_acts.shape[1] != D.shape[1] or after_acts.shape[1] != D.shape[1]:
        raise ValueError(
            f"activation_dim mismatch: dirs d={D.shape[1]} "
            f"before d={before_acts.shape[1]} after d={after_acts.shape[1]}")
    # Unit-normalise the decoder directions (project = cosine-scaled energy).
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = D / norms                                              # (n_cand, d)

    # Per-candidate mean-squared projection on each stream.
    eps = 1e-12
    before_energy = ((before_acts @ U.T) ** 2).mean(0)         # (n_cand,)
    after_energy = ((after_acts @ U.T) ** 2).mean(0)           # (n_cand,)
    absence_ratio = before_energy / (after_energy + eps)       # genuine ≪ 1, shrinkage ≈ 1

    # Permutation null: absence ratio of random unit directions through the SAME streams.
    rng = np.random.default_rng(seed)
    d = D.shape[1]
    R = rng.standard_normal((n_perm, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rand_before = ((before_acts @ R.T) ** 2).mean(0)           # (n_perm,)
    rand_after = ((after_acts @ R.T) ** 2).mean(0)             # (n_perm,)
    rand_ratio = rand_before / (rand_after + eps)              # (n_perm,) centred ~1
    # one-sided p: P(random ratio <= candidate ratio). Small p ⇒ the candidate's
    # before/after presence ratio is in the LOW tail ⇒ genuinely after-selective.
    p = np.array([(rand_ratio <= ar).mean() for ar in absence_ratio], dtype=float)
    survive = _bh_fdr(p, q=fdr_q)

    return {
        "n_candidates": int(D.shape[0]),
        "before_energy": before_energy,
        "after_energy": after_energy,
        "absence_ratio": absence_ratio,
        "perm_p_absent": p,
        "fdr_q": fdr_q,
        "survives_crosscheck": survive,
        "n_survive": int(survive.sum()),
        "rand_ratio_mean": float(rand_ratio.mean()),
        "rand_ratio_median": float(np.median(rand_ratio)),
    }


def _bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg reject mask at FDR=q (matches lm_feature_drift.bh_fdr)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * q
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = int(np.max(np.where(passed)[0]))
    crit = ranked[kmax]
    return p <= crit


# ===========================================================================
# Verdict assembly (per layer): combine the gates into the adjudication.
# ===========================================================================

def adjudicate(
    real_fraction: float,
    shuffle_fractions: list[float],
    seed_survival: dict,
    n_crosscheck_survive: int,
    n_crosscheck_candidates: int,
    seed_fractions: list[float] | None = None,
) -> dict:
    """Combine N4 shuffle + N2 method-seed + §2.2 cross-check into a per-layer verdict.

    A genuine emergent population requires ALL THREE:
      (1) emergent fraction materially EXCEEDS the shuffle null,
      (2) the emergent SET survives across seeds (mean Jaccard not near chance),
      (3) candidate after-only features SURVIVE the direct-activation cross-check
          (genuinely absent in before, not shrinkage).
    Otherwise → RE-WEIGHTING (emergent ≈ null / artefact).

    The numeric cut for (1) is the shuffle null itself (excess>0 with the real fraction
    outside the shuffle band); for (2) we flag mean Jaccard; (3) is a hard count. The
    final label is computed conservatively — any failed gate ⇒ not a new population."""
    enull = emergent_fraction_minus_null(
        real_fraction, shuffle_fractions, seed_fractions or [])
    excess = enull["excess_over_null_mean"]
    excess_vs_max = enull["excess_over_null_max"]
    mean_jacc = seed_survival.get("mean_pairwise_jaccard", float("nan"))
    crosscheck_rate = (n_crosscheck_survive / n_crosscheck_candidates
                       if n_crosscheck_candidates else 0.0)

    # Gate booleans. (1) excess over the *max* shuffle seed (strict, the achievable
    # null's worst case). (2) Jaccard meaningfully above chance-ish (sets share a core).
    # (3) at least some candidates survive the cross-check.
    gate_excess = bool(excess_vs_max > 0.0 and excess > 0.0)
    gate_seed = bool(not np.isnan(mean_jacc) and mean_jacc >= 0.5)
    gate_crosscheck = bool(n_crosscheck_survive > 0)

    new_population = gate_excess and gate_seed and gate_crosscheck
    verdict = ("INSTALLS_NEW_FEATURES" if new_population else "RE_WEIGHTING")

    return {
        "emergent_vs_null": enull,
        "method_seed_mean_jaccard": mean_jacc,
        "crosscheck_survive": n_crosscheck_survive,
        "crosscheck_candidates": n_crosscheck_candidates,
        "crosscheck_survival_rate": crosscheck_rate,
        "gate_excess_over_null": gate_excess,
        "gate_seed_survival": gate_seed,
        "gate_crosscheck": gate_crosscheck,
        "genuine_emergent_population": new_population,
        "verdict": verdict,
    }
