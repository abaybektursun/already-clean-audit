# W4 — LM feature lineage (spec §2.3 exploratory arm, §7 W4).
"""Gemma Scope as a frozen-base lens on the LM before→after change.

EXPLORATORY (hypothesis-generating), not confirmatory. We run the SAME frozen
Gemma Scope JumpReLU SAE (the C5-validated lens) on residual activations from
BOTH the LM "before" (google/gemma-2-2b) and the LM "after" (PG2's
model.language_model), harvested under forward-condition parity (C2), and ask:

  1. Per SAE latent: activation-FREQUENCY (fraction of tokens it fires) and
     mean-MAGNITUDE (mean activation over firing tokens) on base vs PG2-LM.
  2. Cluster co-firing latents (Bhalla 2026: a single manifold dilutes across
     many features → classify CLUSTERS, not single latents).
  3. Classify each cluster into a 5-way drift taxonomy:
       preserved / drifted / suppressed / amplified / emergent (after-only).
  4. THE KEY QUESTION: do emergent / strongly-amplified clusters concentrate
     their activation MASS on IMAGE-TOKEN positions (the LM learning to ingest
     vision), vs a within-text baseline?

Discipline (CLAUDE.md / spec):
  * Fail loud. Shape/layer mismatch → RAISE (delegated to gemma_scope.JumpReLUSAE).
  * NEVER over-claim: every cluster is a HYPOTHESIS for the W5 causal test.
  * No try/except hiding.
  * This module is PURE (numpy/torch only, no Modal) so it is unit-importable and
    the heavy harvest (modal_pg_w4.py) and the analysis share ONE estimator.

Statistics:
  * Drift significance per latent = two-proportion z-test on firing frequency
    (base vs after), BH-FDR controlled at q=0.05 across all latents tested (§5).
  * Effect size per latent = log2 firing-frequency ratio + magnitude-ratio.
  * Emergent direct-activation cross-check (§2.2): a latent is only "emergent" if
    its base firing-frequency is ~0 (below a floor) AND its after frequency is
    materially above it — guarding against SAE "shrinkage" false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- taxonomy thresholds (set on control data per §6; defaults are the frozen
#     pilot values, overridable from the SAP). All are on the SAE-latent stats. ---

# A latent "fires" on a token iff its JumpReLU activation > 0 (the gate already
# applied a per-latent threshold inside the SAE encode — gemma_scope.JumpReLUSAE).
# "Active" latent = fires on at least this fraction of tokens (else it is dead /
# near-dead and excluded from the live-feature population we test).
ACTIVE_FREQ_FLOOR = 1e-3            # ~>= 1 in 1000 tokens to count as a live latent
# Emergent gate: base must be at/below this freq (≈ never) and after above the
# live floor. This is the §2.2 direct-activation cross-check operationalised on the
# lens latents (an after-only latent whose decoder direction is absent in `before`).
EMERGENT_BASE_FREQ_MAX = 1e-4      # base fires < 1 in 10k tokens → "≈ never in base"
# Drift magnitude bands (log2 ratio of after/before firing-frequency), applied to
# latents live in BOTH models.
AMPLIFY_LOG2 = 1.0                  # after freq >= 2x before  → amplified
SUPPRESS_LOG2 = -1.0               # after freq <= 0.5x before → suppressed
# Within "preserved vs drifted" (both ratios mild), a latent is "drifted" if its
# mean-magnitude (over firing tokens) ratio is large even when frequency is stable.
MAG_DRIFT_LOG2 = 1.0               # |log2(mag_after/mag_before)| >= 1 → drifted


# ===========================================================================
# Per-latent activation statistics over a harvested activation matrix.
# ===========================================================================

@dataclass
class LatentStats:
    """Per-latent stats from running an SAE encoder over a token×d_model matrix.

    All arrays are length d_sae. Computed STREAMED (the latent matrix is never
    fully materialized; see accumulate_latent_stats)."""
    n_tokens: int
    fire_count: np.ndarray          # (d_sae,) tokens where latent fired (>0)
    act_sum: np.ndarray             # (d_sae,) sum of activation over firing tokens
    act_sqsum: np.ndarray           # (d_sae,) sum of activation^2 (for magnitude var)

    @property
    def freq(self) -> np.ndarray:
        """Activation frequency: fraction of tokens the latent fires on."""
        return self.fire_count / max(self.n_tokens, 1)

    @property
    def mean_mag(self) -> np.ndarray:
        """Mean activation magnitude OVER FIRING TOKENS (0 where never fires)."""
        out = np.zeros_like(self.act_sum)
        nz = self.fire_count > 0
        out[nz] = self.act_sum[nz] / self.fire_count[nz]
        return out


def new_accumulator(d_sae: int) -> LatentStats:
    return LatentStats(
        n_tokens=0,
        fire_count=np.zeros(d_sae, dtype=np.float64),
        act_sum=np.zeros(d_sae, dtype=np.float64),
        act_sqsum=np.zeros(d_sae, dtype=np.float64),
    )


def update_accumulator(acc: LatentStats, latents: np.ndarray) -> None:
    """Fold a (n_tokens, d_sae) latent block (SAE encoder output) into acc.

    latents are the JumpReLU activations (already gated: exactly the non-fired
    entries are 0). We accumulate fire counts + activation moments per latent."""
    if latents.ndim != 2:
        raise ValueError(f"latents must be (n_tokens, d_sae); got {latents.shape}")
    fired = latents > 0
    acc.n_tokens += int(latents.shape[0])
    acc.fire_count += fired.sum(0)
    acc.act_sum += latents.sum(0)
    acc.act_sqsum += (latents * latents).sum(0)


# ===========================================================================
# Position-resolved activation MASS (the image-token concentration instrument).
# ===========================================================================

@dataclass
class PositionMass:
    """Per-latent activation MASS split by token-position class (image vs text).

    "Mass" = sum of (positive) activation, NOT just a fire count — a latent that
    fires hard on a few image tokens has high image mass. Used to answer: do
    emergent/amplified latents put their activation mass on image-token positions?
    """
    n_image_tokens: int
    n_text_tokens: int
    image_mass: np.ndarray          # (d_sae,) sum activation on image-token positions
    text_mass: np.ndarray           # (d_sae,) sum activation on text-token positions
    image_fire: np.ndarray          # (d_sae,) fire count on image positions
    text_fire: np.ndarray           # (d_sae,) fire count on text positions

    @property
    def image_mass_fraction(self) -> np.ndarray:
        """Fraction of a latent's total activation mass that lands on image tokens.
        0 where the latent never fires. This is the headline per-latent quantity."""
        tot = self.image_mass + self.text_mass
        out = np.zeros_like(tot)
        nz = tot > 0
        out[nz] = self.image_mass[nz] / tot[nz]
        return out


def new_position_mass(d_sae: int) -> PositionMass:
    return PositionMass(
        n_image_tokens=0, n_text_tokens=0,
        image_mass=np.zeros(d_sae, dtype=np.float64),
        text_mass=np.zeros(d_sae, dtype=np.float64),
        image_fire=np.zeros(d_sae, dtype=np.float64),
        text_fire=np.zeros(d_sae, dtype=np.float64),
    )


def update_position_mass(pm: PositionMass, latents: np.ndarray, is_image: np.ndarray) -> None:
    """Fold a (n_tokens, d_sae) latent block split by an (n_tokens,) image mask.

    is_image[i] True ⇔ token i is an image-token position (input_ids == image_token
    _index). Text = the complement (kept, non-special text positions)."""
    if latents.shape[0] != is_image.shape[0]:
        raise ValueError(f"latents {latents.shape} vs is_image {is_image.shape} row mismatch")
    img = is_image.astype(bool)
    txt = ~img
    li, lt = latents[img], latents[txt]
    pm.n_image_tokens += int(img.sum())
    pm.n_text_tokens += int(txt.sum())
    pm.image_mass += li.sum(0)
    pm.text_mass += lt.sum(0)
    pm.image_fire += (li > 0).sum(0)
    pm.text_fire += (lt > 0).sum(0)


# ===========================================================================
# Co-firing clusters (Bhalla 2026 — classify clusters, not single latents).
# ===========================================================================

def cofiring_clusters(
    fire_matrix: np.ndarray,
    live_idx: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    max_latents: int = 6000,
    priority_idx: np.ndarray | None = None,
) -> dict[int, list[int]]:
    """Cluster LIVE latents by co-firing (correlation of their per-token fire
    pattern) in the AFTER model — so emergent latents (after-only) can cluster.

    fire_matrix: (n_tokens, d_sae) boolean/float fire indicator from the after
                 model (the regime where emergent latents are visible).
    live_idx:    indices of latents to cluster (live in after; includes emergent).
    max_latents: cap on latents fed to the O(n^3) agglomerative step; at large SAE
                 widths n_live can be many thousands. When exceeded we keep ALL
                 priority_idx latents (the non-preserved / interesting ones) plus the
                 most-ACTIVE of the rest (by fire count in fire_matrix) up to the cap.
                 The per-class latentwise concentration (the load-bearing image-token
                 instrument) uses ALL latents regardless of this cap.
    Returns {cluster_id: [latent indices]}. Agglomerative on co-firing distance
    (1 - corr). Deterministic given inputs.

    Rationale: co-firing groups are the unit Bhalla 2026 recommends because one
    underlying concept dilutes across many SAE latents; a per-latent verdict is
    noisier than a per-cluster one."""
    from sklearn.cluster import AgglomerativeClustering

    if live_idx.size == 0:
        return {}

    # Cap the clustered set if needed (keep priority latents + most active).
    if live_idx.size > max_latents:
        activity = fire_matrix[:, live_idx].astype(np.float64).sum(0)   # per-latent fire count
        keep = np.zeros(live_idx.size, dtype=bool)
        if priority_idx is not None and priority_idx.size:
            pri_set = set(int(x) for x in priority_idx)
            keep |= np.array([int(j) in pri_set for j in live_idx], dtype=bool)
        n_remaining = max_latents - int(keep.sum())
        if n_remaining > 0:
            rest = np.where(~keep)[0]
            top = rest[np.argsort(-activity[rest])[:n_remaining]]
            keep[top] = True
        live_idx = live_idx[keep]

    if live_idx.size <= n_clusters:
        # Fewer live latents than requested clusters → each is its own cluster.
        return {i: [int(j)] for i, j in enumerate(live_idx)}

    F = fire_matrix[:, live_idx].astype(np.float64)   # (n_tokens, n_live)
    # Pearson corr across tokens between latents; guard zero-variance columns.
    F = F - F.mean(0, keepdims=True)
    norm = np.linalg.norm(F, axis=0, keepdims=True)
    norm[norm == 0] = 1.0
    Fn = F / norm
    corr = Fn.T @ Fn                                   # (n_live, n_live) in [-1,1]
    corr = np.clip(corr, -1.0, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)

    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average",
    ).fit_predict(dist)
    clusters: dict[int, list[int]] = {}
    for latent, lab in zip(live_idx.tolist(), labels.tolist()):
        clusters.setdefault(int(lab), []).append(int(latent))
    return clusters


# ===========================================================================
# Drift classification (5-way taxonomy) + FDR.
# ===========================================================================

def _log2_ratio(after: np.ndarray, before: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log2((after + eps) / (before + eps))


def two_proportion_p(k1, n1, k2, n2) -> np.ndarray:
    """Two-sided two-proportion z-test p-value, vectorised over latents.
    k1/n1 = after fires/tokens, k2/n2 = before fires/tokens. Used for the
    firing-frequency drift test; BH-FDR corrected across latents downstream."""
    from scipy.stats import norm as _norm
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))
    se = np.where(se == 0, np.nan, se)
    z = (p1 - p2) / se
    pval = 2.0 * (1.0 - _norm.cdf(np.abs(z)))
    # Latents dead in BOTH (p1==p2==0) → no test; p-value 1.
    return np.where(np.isnan(pval), 1.0, pval)


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean reject mask at FDR=q (§5). Vectorised."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * q
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    crit = ranked[kmax]
    return p <= crit


def classify_latents(
    before: LatentStats,
    after: LatentStats,
    fdr_q: float = 0.05,
) -> dict:
    """Per-latent 5-way drift class + stats + FDR. Returns dict of arrays.

    Classes (priority order):
      emergent  : base freq <= EMERGENT_BASE_FREQ_MAX AND after freq >= ACTIVE_FREQ
      suppressed: live in both, log2(freq ratio) <= SUPPRESS_LOG2
      amplified : live in both, log2(freq ratio) >= AMPLIFY_LOG2
      drifted   : live in both, freq stable but |log2(mag ratio)| >= MAG_DRIFT_LOG2
      preserved : live in both, freq + mag both stable
      (dead     : not live in either → excluded from the taxonomy population)
    Only latents whose frequency-drift is FDR-significant get a non-preserved
    *frequency* class; magnitude-drift latents are flagged separately."""
    fb, fa = before.freq, after.freq
    mb, ma = before.mean_mag, after.mean_mag

    live_before = fb >= ACTIVE_FREQ_FLOOR
    live_after = fa >= ACTIVE_FREQ_FLOOR
    live_both = live_before & live_after
    dead = ~(live_before | live_after)

    freq_log2 = _log2_ratio(fa, fb)
    mag_log2 = _log2_ratio(ma, mb)

    pvals = two_proportion_p(
        after.fire_count, after.n_tokens, before.fire_count, before.n_tokens
    )
    sig = bh_fdr(pvals, q=fdr_q)

    cls = np.array(["dead"] * fb.size, dtype=object)
    cls[live_both] = "preserved"

    # emergent: ≈ never in base, live in after (the §2.2 cross-check gate).
    emergent = (fb <= EMERGENT_BASE_FREQ_MAX) & live_after
    cls[emergent] = "emergent"

    # frequency drift (only among live-in-both AND FDR-significant).
    amp = live_both & sig & (freq_log2 >= AMPLIFY_LOG2)
    sup = live_both & sig & (freq_log2 <= SUPPRESS_LOG2)
    cls[amp] = "amplified"
    cls[sup] = "suppressed"

    # magnitude-only drift: live both, freq class stayed "preserved", but big mag move.
    still_pres = (cls == "preserved")
    drift = still_pres & (np.abs(mag_log2) >= MAG_DRIFT_LOG2)
    cls[drift] = "drifted"

    return {
        "class": cls,                       # (d_sae,) object array of labels
        "freq_before": fb, "freq_after": fa,
        "mag_before": mb, "mag_after": ma,
        "freq_log2_ratio": freq_log2,
        "mag_log2_ratio": mag_log2,
        "pval_freq": pvals,
        "fdr_reject": sig,
        "live_before": live_before, "live_after": live_after,
        "live_both": live_both, "dead": dead,
    }


def class_counts(cls: np.ndarray) -> dict:
    labels, counts = np.unique(cls, return_counts=True)
    return {str(l): int(c) for l, c in zip(labels, counts)}


# ===========================================================================
# Cluster-level summary + the image-token concentration finding.
# ===========================================================================

@dataclass
class ClusterDrift:
    cluster_id: int
    latents: list[int]
    dominant_class: str                 # plurality latent class in the cluster
    class_breakdown: dict               # {class: count}
    mean_freq_before: float
    mean_freq_after: float
    mean_freq_log2_ratio: float
    mean_mag_log2_ratio: float
    min_fdr_p: float                    # strongest single-latent FDR p in cluster
    n_fdr_sig: int                      # latents in cluster passing FDR
    # image-token concentration (filled when a multimodal PositionMass is given):
    image_mass_fraction: float | None = None   # cluster mass on image vs text tokens
    image_mass_fraction_latentwise: float | None = None  # mean of per-latent fractions
    n_image_tokens: int | None = None
    n_text_tokens: int | None = None


def summarize_clusters(
    clusters: dict[int, list[int]],
    per_latent: dict,
    pos_mass: PositionMass | None = None,
) -> list[ClusterDrift]:
    """Aggregate per-latent drift + (optional) image/text mass into cluster rows.

    A cluster's image_mass_fraction = (sum image_mass over its latents) /
    (sum total mass over its latents) — the fraction of the cluster's multimodal
    activation mass that lands on image-token positions. Compared downstream
    against the preserved-cluster baseline (a within-text reference)."""
    cls = per_latent["class"]
    flog = per_latent["freq_log2_ratio"]
    mlog = per_latent["mag_log2_ratio"]
    fb, fa = per_latent["freq_before"], per_latent["freq_after"]
    pvals = per_latent["pval_freq"]
    sig = per_latent["fdr_reject"]

    rows: list[ClusterDrift] = []
    for cid, lats in clusters.items():
        lats_arr = np.array(lats, dtype=int)
        sub_cls = cls[lats_arr]
        breakdown = class_counts(sub_cls)
        dominant = max(breakdown, key=breakdown.get)
        row = ClusterDrift(
            cluster_id=int(cid),
            latents=lats,
            dominant_class=dominant,
            class_breakdown=breakdown,
            mean_freq_before=float(fb[lats_arr].mean()),
            mean_freq_after=float(fa[lats_arr].mean()),
            mean_freq_log2_ratio=float(flog[lats_arr].mean()),
            mean_mag_log2_ratio=float(mlog[lats_arr].mean()),
            min_fdr_p=float(pvals[lats_arr].min()),
            n_fdr_sig=int(sig[lats_arr].sum()),
        )
        if pos_mass is not None:
            img = float(pos_mass.image_mass[lats_arr].sum())
            txt = float(pos_mass.text_mass[lats_arr].sum())
            tot = img + txt
            row.image_mass_fraction = (img / tot) if tot > 0 else None
            frac_lat = pos_mass.image_mass_fraction[lats_arr]
            fired_lat = (pos_mass.image_fire + pos_mass.text_fire)[lats_arr] > 0
            row.image_mass_fraction_latentwise = (
                float(frac_lat[fired_lat].mean()) if fired_lat.any() else None
            )
            row.n_image_tokens = pos_mass.n_image_tokens
            row.n_text_tokens = pos_mass.n_text_tokens
        rows.append(row)
    return rows


def image_concentration_summary(
    cluster_rows: list[ClusterDrift],
    pos_mass: PositionMass,
) -> dict:
    """THE KEY FINDING: do emergent/amplified clusters concentrate activation mass
    on image-token positions vs the within-text baseline?

    Baseline = the chance image-mass fraction if a latent were position-agnostic =
    n_image_tokens / (n_image_tokens + n_text_tokens). A cluster well above this
    baseline is putting disproportionate mass on image tokens. We also report the
    PRESERVED-cluster mean as an empirical within-LM reference (pure-LM features
    should sit near baseline; emergent vision-ingest features should sit above)."""
    n_img = pos_mass.n_image_tokens
    n_txt = pos_mass.n_text_tokens
    chance = n_img / (n_img + n_txt) if (n_img + n_txt) > 0 else float("nan")

    def _grp(name):
        fr = [r.image_mass_fraction for r in cluster_rows
              if r.dominant_class == name and r.image_mass_fraction is not None]
        return fr

    groups = {
        c: _grp(c)
        for c in ("emergent", "amplified", "drifted", "suppressed", "preserved")
    }
    out = {
        "chance_image_fraction": chance,
        "n_image_tokens": n_img,
        "n_text_tokens": n_txt,
        "per_class": {},
    }
    for c, fr in groups.items():
        if fr:
            arr = np.array(fr, dtype=float)
            out["per_class"][c] = {
                "n_clusters": int(arr.size),
                "mean_image_mass_fraction": float(arr.mean()),
                "median_image_mass_fraction": float(np.median(arr)),
                "max_image_mass_fraction": float(arr.max()),
                "frac_clusters_above_chance": float((arr > chance).mean()),
            }
        else:
            out["per_class"][c] = {"n_clusters": 0}
    # The headline contrast: emergent/amplified vs preserved (within-text reference).
    pres = np.array(groups["preserved"], dtype=float) if groups["preserved"] else np.array([])
    for c in ("emergent", "amplified"):
        fr = np.array(groups[c], dtype=float) if groups[c] else np.array([])
        if fr.size and pres.size:
            out["per_class"][c]["excess_over_preserved"] = float(fr.mean() - pres.mean())
            out["per_class"][c]["excess_over_chance"] = float(fr.mean() - chance)
    return out


def latent_class_concentration(
    per_latent: dict,
    pos_mass: PositionMass,
) -> dict:
    """Per-DRIFT-CLASS image-token concentration aggregated over LATENTS directly
    (clustering-independent). This is the robust form of the key finding: emergent
    latents may scatter across co-firing clusters (they fire sparsely / together
    only weakly), so we also pool the image-mass fraction over ALL latents of each
    class. Two aggregations per class:
      * pooled: sum(image_mass)/sum(total_mass) over the class's latents — mass-weighted.
      * latentwise: mean over latents of each latent's own image_mass_fraction
        (restricted to latents that fire at all under the multimodal regime).
    Compared to chance = n_image/(n_image+n_text) and to the PRESERVED class
    (the empirical within-LM reference for a position-agnostic feature)."""
    cls = per_latent["class"]
    n_img, n_txt = pos_mass.n_image_tokens, pos_mass.n_text_tokens
    chance = n_img / (n_img + n_txt) if (n_img + n_txt) > 0 else float("nan")
    frac_lat = pos_mass.image_mass_fraction
    fired = (pos_mass.image_fire + pos_mass.text_fire) > 0

    out = {"chance_image_fraction": chance, "n_image_tokens": n_img,
           "n_text_tokens": n_txt, "per_class": {}}
    classes = ("emergent", "amplified", "drifted", "suppressed", "preserved")
    pooled_means = {}
    for c in classes:
        sel = (cls == c)
        n_lat = int(sel.sum())
        rec = {"n_latents": n_lat}
        if n_lat:
            img = float(pos_mass.image_mass[sel].sum())
            txt = float(pos_mass.text_mass[sel].sum())
            tot = img + txt
            rec["n_latents_firing_mm"] = int((sel & fired).sum())
            rec["pooled_image_mass_fraction"] = (img / tot) if tot > 0 else None
            lw = frac_lat[sel & fired]
            rec["latentwise_image_mass_fraction"] = float(lw.mean()) if lw.size else None
            rec["frac_latents_above_chance"] = float((lw > chance).mean()) if lw.size else None
            pooled_means[c] = rec["pooled_image_mass_fraction"]
        out["per_class"][c] = rec
    # Headline excesses vs the preserved within-LM reference + chance.
    pres = pooled_means.get("preserved")
    for c in ("emergent", "amplified"):
        pc = pooled_means.get(c)
        if pc is not None:
            out["per_class"][c]["excess_over_chance"] = float(pc - chance)
            if pres is not None:
                out["per_class"][c]["excess_over_preserved"] = float(pc - pres)
    return out
