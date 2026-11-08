"""flens — pure-logic core for the FUNCTIONAL-lens study: is the Gemma-2 mid-stack MMLU-MCQ deletion
sparse in the network's OWN (SAE-feature) basis, where it was structureless in the LINEAR basis?

Companion to `studies/lineage/paligemma2/research/2026-06-29_functional_lens_prereg.md` (slug `flens`).
PURE numpy — no torch, no model, no GPU — so it py_compiles and unit-tests on the control plane. The
torch FeaturePatchHook + the Gemma Scope SAE encode/decode live in `infra/aws_pg/jobs/flens_cell.py`;
ALL of the feature-attribution / delta-construction / null / RF arithmetic lives HERE, so the analysis
logic is provable weight-free (prereg §13 weight-free dry-run).

The question (prereg §1). LATABLATE showed the L16 deletion is NOT a low-rank removable LINEAR
direction (RF≈0 at every k≤32; sealed RF(k=2)=−0.002). Every program instrument is linear (rank, span,
energy, projection, R_prin, f_perp). A transformer computes in FEATURES, not subspaces. This study
swaps ONLY the lens — linear subspaces → the Gemma Scope SAE feature basis the network actually uses —
and asks whether the rewrite that is structureless under the first is structured under the second.

Estimand (prereg §2). Primary readout
    RF(K) = (acc_patch(K) − floor)/(intact − floor)   on blank-image MMLU-MCQ accuracy,
where acc_patch(K) is the DELETED model run with a live L16 LAST-TOKEN patch that sets the top-K
(intact−deleted)-differing SAE features to their intact per-item values:
    h[:,-1] ← h[:,-1] + (f_int[S] − f_del[S]) @ W_dec[S],   |S| = K,   S frozen on discovery.
Floor = deleted MCQ (K=0); intact = intact MCQ; both on the SAME items, paired by item.

The unifying simplicity: EVERY patch type — real top-K, matched-random-K, item-shuffle, and the
SVD-component BASIS CONTROL — is the SAME operation, a per-item (n, d_model) last-token delta; they
differ ONLY in how the delta is built (all pure functions here). The lens claim is
RF_feature(K) > RF_svd(K) at matched K — a difference attributable to the BASIS, not the operation.

Discipline: no try/except hiding; bad shapes / degenerate denominators RAISE. No model-name or baseline
literals are baked in. Reuses latablate.rf / rf_from_items / bootstrap_rf / minimal_k /
real_beats_random / is_monotone_nondecreasing / estimate_disabling_subspace (the verified linear core).
"""
from __future__ import annotations

import numpy as np

from infra.pg.latablate import (  # noqa: F401  — reuse the verified RF + SVD core, no re-implementation
    rf, rf_from_items, bootstrap_rf, minimal_k, real_beats_random,
    is_monotone_nondecreasing, estimate_disabling_subspace, is_orthonormal,
)

# ===========================================================================
# Frozen thresholds (prereg §6) — SINGLE SOURCE OF TRUTH, imported by the driver and the dry-run.
# Changing any of these is a re-freeze and must be journalled. (latablate's K_SMALL=8 is a DIFFERENT
# study; flens fixes its own K_SMALL=64 — a handful of features, ≪ d_sae=16384.)
# ===========================================================================
TAU_RESTORE = 0.50           # "substantial restoration": recover ≥ half the deleted→intact gap (= latablate)
K_SMALL = 64                 # "sparse in the feature basis" requires the minimal restoring K* ≤ 64 ≪ d_sae
DELTA_RAND = 0.20            # real RF(K*) must exceed the matched-random-K feature null by ≥ this (the
#                             "any-K-features" null — gating ONLY in the distributed regime; in the sparse
#                             regime it is near-vacuous (random active features barely differ → ~0 delta),
#                             so the LENS claim rests on DELTA_LENS below, not on this. (Audit B1/S3.)
DELTA_LENS = 0.20           # THE LENS GATE: the paired item-bootstrap CI-lower of RF_feature(K*) − RF_svd(K*)
#                             must exceed this, for BOTH the raw AND the massive-activation-robust (whitened)
#                             SVD-component controls. This is the only conjunct that isolates "feature basis
#                             beats the linear basis," so it carries a margin + CI, not a bare point compare.
K_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)   # the frozen feature-count sweep ("full" is separate)
K_MAX = 256                 # size of the banked top-differing feature set S (max numeric grid point)
RANDOM_NULL_SEEDS = (301, 302, 303, 304, 305)    # matched-random-K "any-K-features" null (see DELTA_RAND)
SHUFFLE_NULL_SEEDS = (401, 402, 403, 404, 405)   # item-shuffle (generic-vs-item-specific characterization)
BOOTSTRAP_SEED = 7
N_BOOT = 10000
# SAE-faithfulness gate (RE-SPECIFIED 2026-06-30 after the EV-gate diagnostic — see journal). The text-
# trained Gemma Scope SAE is partially OOD on the PG2 BLANK-IMAGE MULTIMODAL forward: on its in-distribution
# text gemma-2-2b L16 it reconstructs at EV≈+0.55 with normal L0≈80, but on the multimodal forward the
# variance-normalized EV goes negative (residual is mean-dominated / OOD in magnitude) WHILE the recon stays
# directionally faithful (cosine≈0.85-0.90) and PRESERVES the MCQ readout. So EV is the WRONG gate here; the
# study-aligned gate is BEHAVIORAL (round-trip MCQ) + a robust DIRECTIONAL floor (cosine). EV is reported.
ROUNDTRIP_RETAIN_DELTA = 0.05   # GATE: intact SAE round-trip MCQ ≥ intact − this (the readout is preserved)
COSINE_MIN = 0.50               # GATE: mean per-item cosine(intact_last, SAE-recon) ≥ this (recon directionally
#                                 correct — a robust not-degenerate floor, a priori; observed ≈0.90)
EV_MIN = 0.70               # REPORTED diagnostic threshold only (ev_pass flag) — NOT a gate (OOD-sensitive)

# The frozen locus (prereg §0): LATLOC's confirmed mid-stack onset = the Gemma Scope residual-SAE site
# (post-block output residual). Imported, not re-derived — recompute the latloc onset from its banked
# JSON at analysis time if ever in doubt (never quote it from prose).
L_LOCUS = 16


# ===========================================================================
# Feature-difference attribution (prereg §2/§6) — the feature analog of LATABLATE's top-k SVD of the
# (deleted − intact) residual difference. Rank features by how much their ACTIVATION changed.
# ===========================================================================

def feature_difference_attribution(f_int, f_del, k_max: int = K_MAX):
    """Top-k_max SAE features by mean |f_int − f_del| over items. f_int/f_del: (n, d_sae) per-item SAE
    activations (intact, deleted), row-paired by item. Returns S: (k_max,) int feature indices ordered
    by DESCENDING mean |Δf| (the dominant differing features first)."""
    A = np.asarray(f_int, float)
    B = np.asarray(f_del, float)
    if A.shape != B.shape:
        raise ValueError(f"attribution: f_int/f_del shape mismatch {A.shape} vs {B.shape}")
    n, d_sae = A.shape
    if not (0 < k_max <= d_sae):
        raise ValueError(f"attribution: k_max={k_max} out of (0, d_sae={d_sae}]")
    score = np.abs(A - B).mean(axis=0)                      # (d_sae,) mean activation-difference per feature
    S = np.argsort(score)[::-1][:k_max]                     # top-k_max features, descending
    return np.ascontiguousarray(S.astype(int))


def direction_split(f_int, f_del, S) -> dict:
    """Suppressed (intact>deleted: the deletion SILENCED a present feature) vs corrupting
    (deleted>intact: it ACTIVATED a disabling feature) over S — NAMES the mechanism (prereg §6/F3).
    Returns the signed mean Δf on S, the suppressed/corrupting fractions, and mean f_int/f_del on S."""
    A = np.asarray(f_int, float)
    B = np.asarray(f_del, float)
    S = np.asarray(S, int)
    dmean = (A - B).mean(axis=0)[S]                         # signed mean Δf on S
    suppressed = dmean > 0
    return {"signed_mean_df": [float(x) for x in dmean],
            "frac_suppressed": float(suppressed.mean()) if S.size else float("nan"),
            "frac_corrupting": float((~suppressed).mean()) if S.size else float("nan"),
            "mean_f_int_on_S": [float(x) for x in A.mean(0)[S]],
            "mean_f_del_on_S": [float(x) for x in B.mean(0)[S]]}


# ===========================================================================
# Delta builders (prereg §2). Each returns a per-item (n, d_model) last-token delta — the SAME
# operation fed to the one torch FeaturePatchHook; only the construction differs.
# ===========================================================================

def feature_patch_delta(f_int, f_del, S_k, W_dec):
    """Set features S_k to their intact per-item values at the last token:
    (f_int[:, S_k] − f_del[:, S_k]) @ W_dec[S_k] → (n, d_model). The shared 'patch-to-intact' operation
    in the SAE FEATURE basis (the basis = the decoder rows W_dec[S_k]). Empty S_k (K=0) → zeros (floor).
    Carries the deleted model's own residual error term (changes ONLY the |S_k| representable features)."""
    A = np.asarray(f_int, float)
    B = np.asarray(f_del, float)
    Wd = np.asarray(W_dec, float)
    S_k = np.asarray(S_k, int)
    n, d_sae = A.shape
    if Wd.shape[0] != d_sae:
        raise ValueError(f"feature_patch_delta: W_dec rows {Wd.shape[0]} != d_sae {d_sae}")
    if S_k.size == 0:
        return np.zeros((n, Wd.shape[1]))
    return (A[:, S_k] - B[:, S_k]) @ Wd[S_k]                # (n, d_model)


def full_patch_delta(f_int, f_del, W_dec):
    """The recoverable-ceiling delta (prereg §2 'K=full ≈ the LATLOC all-feature patch'): ALL features
    → intact, (f_int − f_del) @ W_dec → (n, d_model). RF here is the most restoration the
    SAE-representable change can buy (< 1 by the SAE reconstruction-error term)."""
    A = np.asarray(f_int, float)
    B = np.asarray(f_del, float)
    Wd = np.asarray(W_dec, float)
    if Wd.shape[0] != A.shape[1]:
        raise ValueError(f"full_patch_delta: W_dec rows {Wd.shape[0]} != d_sae {A.shape[1]}")
    return (A - B) @ Wd


def active_feature_pool(f_int, exclude_S, tol: float = 0.0):
    """Features ACTIVE (nonzero on ≥1 item) in the intact SAE encoding, MINUS the frozen S — the
    matched-random null draws K 'other active features' from here (prereg §5). Patching an inactive
    feature is a no-op (an unfair null), so the pool is restricted to active features. Returns sorted
    int indices."""
    A = np.asarray(f_int, float)
    active = np.where((np.abs(A) > tol).any(axis=0))[0]
    pool = np.setdiff1d(active, np.asarray(exclude_S, int))
    return pool.astype(int)


def random_feature_set(pool, k: int, seed: int):
    """K distinct random features from the active pool (the LOAD-BEARING matched-random-K null). Raises
    if the pool is smaller than k (fail-loud, no silent truncation). k=0 → empty."""
    pool = np.asarray(pool, int)
    if k == 0:
        return np.zeros(0, int)
    if k > pool.size:
        raise ValueError(f"random_feature_set: k={k} > active pool size {pool.size}")
    return np.sort(np.random.default_rng(seed).choice(pool, size=k, replace=False)).astype(int)


def shuffle_deltas(delta_real, seed: int):
    """Item-shuffle null (prereg §5): apply a DIFFERENT item's restoration delta to each item, by a
    CYCLIC SHIFT (a guaranteed derangement — no fixed point, a true permutation, unlike a where-patched
    permutation that can collide). Magnitude-matched to the real patch; restores IFF the patch is GENERIC
    (item-independent), fails IFF item-specific — informative either way, NOT a pass/fail gate. (n,d)."""
    D = np.asarray(delta_real, float)
    n = D.shape[0]
    if n < 2:
        raise ValueError("shuffle_deltas needs ≥2 items for a derangement")
    offset = int(np.random.default_rng(seed).integers(1, n))      # 1..n-1 → donor != self for every item
    return D[(np.arange(n) + offset) % n]


def svd_patch_delta(deleted_last, intact_last, k: int, scale=None):
    """The BASIS-CONTROL delta (prereg §2/§5): set the top-k SVD components of (deleted − intact) to
    intact — the SAME 'patch-to-intact' operation in a LINEAR top-variance basis. V = top-k right
    singular vectors of the (optionally per-dim-standardized) difference; delta_i in the ORIGINAL
    residual space → (n, d_model). k=0 → zeros.

    scale=None  → RAW SVD of (deleted − intact): the top-VARIANCE linear directions.
    scale=(d,)  → WHITENED SVD: standardize each residual dimension by `scale` (per-dim std) BEFORE the
                  SVD, then map the patch back. This is the massive-activation-robust linear control (audit
                  B1): Gemma-2 residuals carry axis-aligned massive-activation dims (~1e3-1e4) whose raw
                  variance would otherwise dominate V and let raw-SVD waste its budget on a readout-
                  irrelevant high-variance dim — inflating a spurious feature-vs-linear gap. Whitening
                  removes that confound. The LENS claim (prereg §6) is RF_feature(k) > BOTH RF_svd_raw(k)
                  AND RF_svd_white(k) by DELTA_LENS with a paired CI — so it cannot be the SVD ordering."""
    Xd = np.asarray(deleted_last, float)
    Xi = np.asarray(intact_last, float)
    if Xd.shape != Xi.shape:
        raise ValueError(f"svd_patch_delta: deleted/intact shape mismatch {Xd.shape} vs {Xi.shape}")
    if k == 0:
        return np.zeros_like(Xd)
    s = np.ones(Xd.shape[1]) if scale is None else np.asarray(scale, float)
    if s.shape != (Xd.shape[1],):
        raise ValueError(f"svd_patch_delta: scale shape {s.shape} != (d_model={Xd.shape[1]},)")
    V = estimate_disabling_subspace((Xd - Xi) / s, k)     # top-k of the standardized difference (d, k)
    proj = (((Xi - Xd) / s) @ V) @ V.T                    # (n, d) patch-to-intact in standardized coords
    return proj * s                                        # back to the original residual space


def bootstrap_rf_difference(a_items, b_items, floor_items, intact_items,
                            seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT):
    """Paired item-bootstrap of RF(a) − RF(b) (the LENS MARGIN: a = feature patch, b = an SVD-component
    control), using the SAME resample indices each draw so the two RFs share items and the difference CI
    is paired (tighter, honest). All four per-item 0/1 arrays are item-paired. Returns (point, ci_lo,
    ci_hi). The lens claim (prereg §6) requires ci_lo > DELTA_LENS for BOTH the raw and whitened controls."""
    a = np.asarray(a_items, float)
    b = np.asarray(b_items, float)
    f = np.asarray(floor_items, float)
    it = np.asarray(intact_items, float)
    n = len(a)
    if not (len(b) == len(f) == len(it) == n):
        raise ValueError("bootstrap_rf_difference: the four per-item arrays must be item-paired (equal length).")
    point = rf_from_items(a, f, it) - rf_from_items(b, f, it)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for j in range(n_boot):
        idx = rng.integers(0, n, n)
        denom = float(np.mean(it[idx]) - np.mean(f[idx]))
        if abs(denom) < 1e-9:
            draws[j] = np.nan
            continue
        fl = float(np.mean(f[idx]))
        draws[j] = (np.mean(a[idx]) - fl) / denom - (np.mean(b[idx]) - fl) / denom
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


# ===========================================================================
# A lossless mock JumpReLU SAE + a planted synthetic — weight-free validation (prereg §13). Mirrors
# gemma_scope.JumpReLUSAE's algebra (pre = x@W_enc+b_enc; f = pre*(pre>θ); decode = f@W_dec+b_dec).
# AXIS-ALIGNED decoder (feature j ⇒ residual coordinate j) + W_enc = W_decᵀ + zero bias/threshold so
# encode∘decode is EXACT on nonneg features. Axis-aligned is BOTH simpler and faithful to the threat the
# whitened control addresses: Gemma-2 massive activations live on specific residual COORDINATES, so the
# synthetic's high-variance readout-irrelevant "junk" sits on real coordinates [d_sae:] that per-dim
# whitening can target. (The REAL SAE is overcomplete + lossy; that lossiness is gated at preflight via
# EV_MIN, not here.)
# ===========================================================================

class JumpReLUMock:
    def __init__(self, d_model: int, d_sae: int, seed: int = 0):
        if d_sae > d_model:
            raise ValueError("mock: need d_sae ≤ d_model (features map to the first d_sae coordinates)")
        eye = np.eye(d_model)
        self.W_dec = np.ascontiguousarray(eye[:d_sae])     # (d_sae, d_model): row j = e_j (coordinate j)
        self.W_enc = np.ascontiguousarray(self.W_dec.T)    # (d_model, d_sae) = W_decᵀ (selects first d_sae)
        self.b_enc = np.zeros(d_sae)
        self.b_dec = np.zeros(d_model)
        self.threshold = np.zeros(d_sae)
        self.d_model, self.d_sae = d_model, d_sae

    def null_basis(self):
        """The decoder NULL coordinates [d_sae:] (axis-aligned): residual dims carrying NO feature — the
        SAE reconstruction-error / massive-activation span. (d_model, d_model − d_sae)."""
        return np.eye(self.d_model)[:, self.d_sae:]

    def encode(self, x):
        pre = np.asarray(x, float) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, f):
        return np.asarray(f, float) @ self.W_dec + self.b_dec


class SyntheticFeatureStack:
    """Tiny synthetic for the dry-run (prereg §13). A lossless mock SAE (orthonormal decoder); the first
    n_choices feature directions ARE the readout class axes (feature c ⇒ class c); class 0 is a RESERVED
    disabling class (never an answer). intact items activate their answer feature; the deletion changes
    a small set of features, disabling the readout. Modes:

      'sparse'        : the deletion activates the reserved disabling feature 0 (Δf concentrated in
                        r_plant features that ARE the high-energy linear directions) → real top-r_plant
                        restores; matched-random fails; the SVD-patch ALSO restores at r_plant (the
                        planted Δ is SVD-ALIGNED). The generic-disabling, latloc-consistent case.
      'lens'          : the SANDBAG case (audit B1). Same disabling feature 0, PLUS a high-variance
                        junk_rank perturbation on the null COORDINATES [d_sae:] (readout-irrelevant,
                        INVISIBLE to encode). The feature attribution finds feature 0 → real restores at
                        small K; the RAW SVD-patch spends its top components on the high-variance junk →
                        FAILS. But the WHITENED SVD-patch (per-dim std) downweights the junk coordinates →
                        ALSO restores. So real ≈ whitened-SVD ≫ raw-SVD: the apparent "lens isolation" is
                        a massive-activation artifact that a robust linear control catches — the lens
                        verdict (real beats BOTH controls by δ_lens) correctly does NOT fire here.
      'item_specific' : the deletion SUPPRESSES each item's own answer feature → the restoration delta
                        is item-specific. Real top-K restores; the item-SHUFFLE fails (a different
                        item's delta re-activates the wrong answer). Validates the shuffle null.
      'distributed'   : the deletion boosts MANY wrong classes per item (a per-item random spread over
                        n_choices−1 features) → no small-K feature patch restores (real ≈ random). The
                        H_irreducible/H_overwrite signature in the feature basis.
    """

    def __init__(self, d_model: int = 512, d_sae: int = 384, n_items: int = 600, n_choices: int = 12,
                 mode: str = "sparse", r_plant: int = 1, signal: float = 3.0, corrupt_scale: float = 8.0,
                 junk_rank: int = 8, junk_scale: float = 20.0, noise: float = 0.1, seed: int = 0):
        # d_sae (384) > K_MAX (256) by design: S = top-K_MAX features must leave an active-feature pool
        # for the matched-random-K null (active ∖ S), mirroring real data (|S|=256 ≪ d_sae=16384).
        if not (1 <= n_choices <= d_sae):
            raise ValueError(f"need 1 ≤ n_choices ≤ d_sae, got {n_choices}/{d_sae}")
        if d_model - d_sae < junk_rank:
            raise ValueError(f"null space ({d_model - d_sae}) < junk_rank ({junk_rank})")
        rng = np.random.default_rng(seed)
        self.sae = JumpReLUMock(d_model, d_sae, seed=seed)
        self.W_dec = self.sae.W_dec
        self.W = self.sae.W_dec[:n_choices]                # readout rows = class axes (orthonormal)
        self.d_model, self.d_sae, self.n_choices, self.mode = d_model, d_sae, n_choices, mode
        used = np.arange(1, n_choices)                      # class 0 reserved → answers ∈ {1 .. nc-1}
        self.answer = used[np.arange(n_items) % len(used)]

        f_int = np.zeros((n_items, d_sae))
        f_int[np.arange(n_items), self.answer] = signal     # the item's answer feature
        f_int[:, n_choices:] += noise * np.abs(rng.standard_normal((n_items, d_sae - n_choices)))
        self.intact_last = self.sae.decode(f_int)
        f_del = f_int.copy()
        junk = np.zeros((n_items, d_model))

        if mode == "sparse":
            f_del[:, :r_plant] += corrupt_scale            # activate reserved disabling features 0..r-1
        elif mode == "lens":
            f_del[:, 0] += corrupt_scale                   # reserved disabling feature 0 (rank-1, energy 8)
            Nb = self.sae.null_basis()[:, :junk_rank]      # (d_model, junk_rank), ⊥ every decoder row
            junk = junk_scale * (rng.standard_normal((n_items, junk_rank)) @ Nb.T)
        elif mode == "item_specific":
            f_del[np.arange(n_items), self.answer] = 0.0   # SUPPRESS this item's own answer feature
        elif mode == "distributed":
            f_del[:, 1:n_choices] += corrupt_scale * np.abs(rng.standard_normal((n_items, n_choices - 1)))
        else:
            raise ValueError(f"unknown mode {mode!r}")

        self.deleted_last = self.sae.decode(f_del) + junk
        self.f_int = self.sae.encode(self.intact_last)     # == f_int (lossless on nonneg features)
        self.f_del = self.sae.encode(self.deleted_last)    # == f_del (junk ⊥ encoder)

    def _acc(self, H):
        return (np.argmax(np.asarray(H, float) @ self.W.T, axis=1) == self.answer).astype(int)

    def floor_items(self):
        return self._acc(self.deleted_last)

    def intact_items(self):
        return self._acc(self.intact_last)

    def patched_items(self, delta):
        return self._acc(self.deleted_last + np.asarray(delta, float))

    def rf_at(self, delta):
        return rf_from_items(self.patched_items(delta), self.floor_items(), self.intact_items())


# ===========================================================================
# Weight-free dry-run (prereg §13) — RAISES on any failed invariant (fail-loud).
# ===========================================================================

def dry_run(verbose: bool = False) -> dict:
    out: dict = {"checks": {}}

    # ---- Check 4: RF math (reused latablate.rf) returns 0 at floor, 1 at intact ----
    assert abs(rf(0.25, 0.25, 0.80) - 0.0) < 1e-9, "RF at floor must be 0"
    assert abs(rf(0.80, 0.25, 0.80) - 1.0) < 1e-9, "RF at intact must be 1"
    assert abs(rf(0.525, 0.25, 0.80) - 0.5) < 1e-9, "RF midpoint must be 0.5"
    assert np.isnan(rf(0.5, 0.5, 0.5)), "RF with no dynamic range must be nan"
    out["checks"]["rf_math"] = "pass"

    # ---- Check 1: SAE-patch ALGEBRA (no readout) on the mock — the feature patch sets EXACTLY the K
    #      target features (decode(f_del with S→f_int) − decode(f_del) == (f_int[S]−f_del[S]) @ W_dec[S]);
    #      K=0 is identity; K=full reproduces the intact reconstruction difference; shape-mismatch RAISES.
    rng = np.random.default_rng(0)
    sae = JumpReLUMock(d_model=64, d_sae=48, seed=0)
    nA = 50
    f_del = np.abs(rng.standard_normal((nA, 48)))
    f_int = np.abs(rng.standard_normal((nA, 48)))
    S = np.array([3, 7, 11, 19, 27])
    patched_feats = f_del.copy()
    patched_feats[:, S] = f_int[:, S]
    decode_diff = sae.decode(patched_feats) - sae.decode(f_del)
    delta_S = feature_patch_delta(f_int, f_del, S, sae.W_dec)
    assert np.allclose(decode_diff, delta_S, atol=1e-9), "feature_patch_delta != exact decode difference on S"
    # the patch round-trips to intact features EXACTLY on S (lossless mock): encode(decode(f_del)+delta) on S == f_int[S]
    re_enc = sae.encode(sae.decode(f_del) + delta_S)
    assert np.allclose(re_enc[:, S], f_int[:, S], atol=1e-6), "patch did not set the K target features to intact"
    untouched = np.setdiff1d(np.arange(48), S)
    assert np.allclose(re_enc[:, untouched], f_del[:, untouched], atol=1e-6), "patch changed features OUTSIDE S"
    assert np.allclose(feature_patch_delta(f_int, f_del, np.zeros(0, int), sae.W_dec), 0.0), "K=0 not identity"
    full = full_patch_delta(f_int, f_del, sae.W_dec)
    assert np.allclose(full, sae.decode(f_int) - sae.decode(f_del), atol=1e-9), "full_patch != intact recon diff"
    _raised = False
    try:
        feature_patch_delta(f_int, f_del, S, sae.W_dec[:, :10].T)   # wrong d_sae rows
    except ValueError:
        _raised = True
    assert _raised, "shape-mismatch W_dec must RAISE"
    out["checks"]["sae_patch_algebra"] = {"exact_decode_diff": True, "sets_only_S": True,
                                          "k0_identity": True, "full_eq_recon_diff": True, "shape_raises": True}

    # ---- Check 2a: SPARSE separation — real top-r restores, matched-random does NOT (artifact gate);
    #      the planted Δ is SVD-ALIGNED so the SVD-patch ALSO restores (check 3 'restores if aligned'). ----
    sp = SyntheticFeatureStack(mode="sparse", r_plant=1, seed=1)
    floor_a, intact_a = float(sp.floor_items().mean()), float(sp.intact_items().mean())
    assert intact_a - floor_a > 0.5, f"sparse needs dynamic range (intact {intact_a}, floor {floor_a})"
    S_sp = feature_difference_attribution(sp.f_int, sp.f_del, k_max=K_MAX)
    delta_real1 = feature_patch_delta(sp.f_int, sp.f_del, S_sp[:1], sp.W_dec)
    rf_real1 = sp.rf_at(delta_real1)
    pool = active_feature_pool(sp.f_int, S_sp)
    rand_rf1 = [sp.rf_at(feature_patch_delta(sp.f_int, sp.f_del, random_feature_set(pool, 1, s), sp.W_dec))
                for s in RANDOM_NULL_SEEDS]
    rf_rand1 = float(np.mean(rand_rf1))
    sp_scale = sp.deleted_last.std(axis=0) + 1e-6                  # per-dim std (the whitening scale)
    rf_svd1 = sp.rf_at(svd_patch_delta(sp.deleted_last, sp.intact_last, 1))
    rf_svdw1 = sp.rf_at(svd_patch_delta(sp.deleted_last, sp.intact_last, 1, scale=sp_scale))
    assert rf_real1 > TAU_RESTORE, f"sparse real top-1 RF {rf_real1:.3f} did not restore"
    assert real_beats_random(rf_real1, rf_rand1, DELTA_RAND), f"sparse real {rf_real1:.3f} ≯ random {rf_rand1:.3f}"
    assert rf_svd1 > TAU_RESTORE, f"sparse raw-SVD should restore (aligned), got {rf_svd1:.3f}"
    assert rf_svdw1 > TAU_RESTORE, f"sparse whitened-SVD should restore (aligned), got {rf_svdw1:.3f}"
    rf_full = sp.rf_at(full_patch_delta(sp.f_int, sp.f_del, sp.W_dec))
    assert rf_full > 0.99, f"sparse full-patch should reach the recoverable ceiling (lossless mock), got {rf_full:.3f}"
    out["checks"]["sparse_separation"] = {"rf_real_k1": round(rf_real1, 3), "rf_random_k1": round(rf_rand1, 3),
                                          "rf_svd_raw_k1": round(rf_svd1, 3), "rf_svd_white_k1": round(rf_svdw1, 3),
                                          "rf_full": round(rf_full, 3)}

    # ---- Check 2b (SANDBAG / audit B1): the RAW SVD-patch FAILS (it spends its budget on a high-variance
    #      readout-irrelevant massive-activation coordinate) while the real feature patch restores — which
    #      LOOKS like a lens win. The WHITENED SVD-patch (per-dim std) downweights that coordinate and
    #      ALSO restores. So real ≈ whitened-SVD ≫ raw-SVD: a robust linear control catches the artifact,
    #      and the lens verdict (real beats BOTH controls by δ_lens) correctly does NOT fire. ----
    ln = SyntheticFeatureStack(mode="lens", junk_rank=8, junk_scale=20.0, seed=1)
    ln_scale = ln.deleted_last.std(axis=0) + 1e-6
    S_ln = feature_difference_attribution(ln.f_int, ln.f_del, k_max=K_MAX)
    rf_feat = {k: ln.rf_at(feature_patch_delta(ln.f_int, ln.f_del, S_ln[:k], ln.W_dec)) for k in (1, 2, 4, 8)}
    rf_svd_raw = {k: ln.rf_at(svd_patch_delta(ln.deleted_last, ln.intact_last, k)) for k in (1, 2, 4, 8)}
    rf_svd_wht = {k: ln.rf_at(svd_patch_delta(ln.deleted_last, ln.intact_last, k, scale=ln_scale)) for k in (1, 2, 4, 8)}
    assert rf_feat[1] > TAU_RESTORE, f"sandbag real top-1 should restore, got {rf_feat[1]:.3f}"
    assert all(rf_svd_raw[k] < 0.2 for k in (1, 2, 4, 8)), f"raw-SVD should be FOOLED (fail), got {rf_svd_raw}"
    assert rf_svd_wht[1] > TAU_RESTORE, f"whitened-SVD should CATCH the sandbag (restore), got {rf_svd_wht[1]:.3f}"
    assert (rf_feat[1] - rf_svd_wht[1]) <= DELTA_LENS, "sandbag: lens must NOT beat the ROBUST control by δ_lens"
    out["checks"]["sandbag_caught_by_whitening"] = {"rf_feature": {k: round(v, 3) for k, v in rf_feat.items()},
                                                    "rf_svd_raw": {k: round(v, 3) for k, v in rf_svd_raw.items()},
                                                    "rf_svd_white": {k: round(v, 3) for k, v in rf_svd_wht.items()}}

    # ---- Check 2b': the LENS-MARGIN machinery — bootstrap_rf_difference brackets a KNOWN difference and
    #      its sign gates δ_lens correctly. Hand-built per-item arrays: feature restores 80%, control 30%. ----
    n_d = 400
    floor_h = np.zeros(n_d); intact_h = np.ones(n_d)
    feat_h = (np.arange(n_d) < int(0.80 * n_d)).astype(float)      # RF_feature ≈ 0.80
    ctrl_h = (np.arange(n_d) < int(0.30 * n_d)).astype(float)      # RF_control ≈ 0.30
    pt, lo, hi = bootstrap_rf_difference(feat_h, ctrl_h, floor_h, intact_h, n_boot=2000)
    assert lo <= pt <= hi and abs(pt - 0.50) < 0.05, f"diff-bootstrap mis-bracketed: pt={pt:.3f} CI[{lo:.3f},{hi:.3f}]"
    assert lo > DELTA_LENS, f"a genuine 0.50 lens margin should clear δ_lens={DELTA_LENS}: CI-lo {lo:.3f}"
    pt0, lo0, hi0 = bootstrap_rf_difference(ctrl_h, ctrl_h, floor_h, intact_h, n_boot=2000)
    assert abs(pt0) < 1e-9 and lo0 <= 0 <= hi0, f"equal-arms diff CI should include 0: pt={pt0} CI[{lo0:.3f},{hi0:.3f}]"
    out["checks"]["lens_margin_bootstrap"] = {"genuine_pt": round(pt, 3), "genuine_ci_lo": round(lo, 3),
                                              "equal_pt": round(pt0, 3)}

    # ---- Check 3b: item-SHUFFLE discriminates — on an item-specific deletion, real restores but a
    #      different item's delta does NOT (shuffle ≈ chance ≪ real). ----
    it = SyntheticFeatureStack(mode="item_specific", seed=1)
    S_it = feature_difference_attribution(it.f_int, it.f_del, k_max=K_MAX)
    delta_real_it = feature_patch_delta(it.f_int, it.f_del, S_it[:it.n_choices], it.W_dec)
    rf_real_it = it.rf_at(delta_real_it)
    shuf_rf = [it.rf_at(shuffle_deltas(delta_real_it, s)) for s in SHUFFLE_NULL_SEEDS]
    rf_shuf_it = float(np.mean(shuf_rf))
    assert rf_real_it > TAU_RESTORE, f"item_specific real patch should restore, got {rf_real_it:.3f}"
    assert (rf_real_it - rf_shuf_it) > DELTA_RAND, f"shuffle should FAIL on item-specific Δ: real {rf_real_it:.3f} shuf {rf_shuf_it:.3f}"
    out["checks"]["shuffle_discriminates_item_specific"] = {"rf_real": round(rf_real_it, 3),
                                                            "rf_shuffle": round(rf_shuf_it, 3)}

    # ---- Check 2c: DISTRIBUTED — no small-K feature patch restores (real ≈ random ≈ low). H_irreducible. ----
    di = SyntheticFeatureStack(mode="distributed", seed=1)
    S_di = feature_difference_attribution(di.f_int, di.f_del, k_max=K_MAX)
    pool_di = active_feature_pool(di.f_int, S_di)
    rf_real_d = {k: di.rf_at(feature_patch_delta(di.f_int, di.f_del, S_di[:k], di.W_dec)) for k in (1, 2, 4, 8)}
    rf_rand_d = {k: float(np.mean([di.rf_at(feature_patch_delta(
        di.f_int, di.f_del, random_feature_set(pool_di, k, s), di.W_dec)) for s in RANDOM_NULL_SEEDS]))
        for k in (1, 2, 4, 8)}
    no_small_k = all(not (rf_real_d[k] > TAU_RESTORE and real_beats_random(rf_real_d[k], rf_rand_d[k], DELTA_RAND))
                     for k in (1, 2, 4, 8))
    assert no_small_k, f"distributed spuriously sparse-recovered: real={rf_real_d} rand={rf_rand_d}"
    out["checks"]["distributed_no_small_k"] = {"rf_real": {k: round(v, 3) for k, v in rf_real_d.items()},
                                               "rf_random": {k: round(v, 3) for k, v in rf_rand_d.items()}}

    # ---- Check (nulls structural): random feature sets are distinct + in-pool; attribution is sorted. ----
    for k in (1, 8, 64):
        rs = random_feature_set(pool, k, RANDOM_NULL_SEEDS[0])
        assert rs.size == k and len(set(rs.tolist())) == k and set(rs.tolist()) <= set(pool.tolist()), f"bad random set k={k}"
    sc = np.abs(sp.f_int - sp.f_del).mean(0)
    assert all(sc[S_sp[i]] >= sc[S_sp[i + 1]] - 1e-12 for i in range(20)), "attribution not descending by |Δf|"
    out["checks"]["nulls_and_attribution_structural"] = "pass"

    # ---- Check (minimal_k + RF bootstrap on a planted sparse RF curve): k* small; CI brackets. ----
    rf_curve = {k: sp.rf_at(feature_patch_delta(sp.f_int, sp.f_del, S_sp[:k], sp.W_dec)) for k in K_GRID}
    assert abs(rf_curve[0]) < 1e-9, "RF(K=0) must be 0 (floor)"
    kstar = minimal_k(rf_curve, TAU_RESTORE)
    assert kstar is not None and kstar <= K_SMALL, f"sparse minimal K* {kstar} not ≤ K_small={K_SMALL}"
    pt, lo, hi = bootstrap_rf(sp.patched_items(delta_real1), sp.floor_items(), sp.intact_items(), n_boot=2000)
    assert lo <= pt <= hi, f"bootstrap CI [{lo:.3f},{hi:.3f}] does not bracket point {pt:.3f}"
    assert lo > TAU_RESTORE, f"sparse RF CI-lower {lo:.3f} should clear τ={TAU_RESTORE}"
    out["checks"]["minimal_k_and_bootstrap"] = {"minimal_k_star": kstar,
                                                "rf_curve": {k: round(v, 3) for k, v in rf_curve.items()},
                                                "ci": [round(lo, 3), round(pt, 3), round(hi, 3)]}

    if verbose:
        import json
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(dry_run(), indent=2))
