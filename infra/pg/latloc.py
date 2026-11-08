"""latloc — pure-logic core for the Latent-Locus deletion-localization study.

Companion to `studies/lineage/paligemma2/research/2026-06-28_latent_locus_prereg.md`
(workstream slug `latloc`). PURE numpy — no torch, no model, no GPU — so it py_compiles
and unit-tests on the control plane. The GPU orchestration (residual capture, PatchHook
sweep, length-bucketed forwards, eval) lives in `infra/aws_pg/jobs/latloc_cell.py` and
calls into this module for ALL of the RF / ΔRF / step / greedy / null arithmetic, so the
analysis logic is provable weight-free (prereg §13 weight-free dry-run).

Estimand (prereg §2). Primary readout
    ΔRF(L) = RF(correct-donor, L) − RF(wrong-donor, L),
    RF = (acc_patched − floor)/(intact − floor)   on blank-image MMLU MCQ accuracy,
with floor and intact recomputed on the SAME sealed item set as acc_patched (paired).
Set-level means; the per-item 0/1 arrays are the resampling unit for the bootstrap.

What each instrument actually does (prereg §1.4–§1.6, corrected here — see the
journal 2026-06-28 entry "ΔRF removes content-free-ness, not depth"):
  * the WRONG-DONOR subtraction (ΔRF) removes CONTENT-FREE-NESS — it asks "is the
    restoration item-specific?". It does NOT cancel the shared-lm_head depth artifact:
    at the final layer the correct donor gives THIS item's answer (RF≈1) while the wrong
    donor gives a DIFFERENT item's answer (RF≈0), so ΔRF≈1 there too, trivially.
  * the localized-vs-smooth (depth) question is adjudicated by the STEP STATISTIC on the
    ΔRF-vs-depth curve, plus the §6 NOVELTY GATE (a final 1–2 layer locus is the EXPECTED
    trivial outcome, not a discovery). The dry-run (`dry_run`) demonstrates that the step
    is the real discriminator and that minimal-|S| is weak in this inject-and-survive
    paradigm (a single late patch restores under BOTH hypotheses).

Discipline: no try/except hiding; bad shapes/degenerate denominators RAISE or return nan
with an explicit guard. No model-name or baseline literals are baked in.
"""
from __future__ import annotations

import numpy as np

# ===========================================================================
# Frozen thresholds (prereg §6) — SINGLE SOURCE OF TRUTH, imported by the driver
# and the dry-run. Changing any of these is a re-freeze and must be journalled.
# ===========================================================================
DELTA = 0.20            # item-specificity margin: ΔRF(L*) bootstrap CI-lower must exceed this
TAU = 0.80             # raw-recovery criterion, DESCRIPTIVE only (raw RF is depth-confounded)
K_STEP = 3.0           # step-statistic threshold (calibrated on synthetics by dry_run)
FDR_Q = 0.05
N_SEEDS = 5
RANDOM_NULL_SEEDS = (101, 102, 103, 104, 105)
WRONG_DONOR_SEEDS = (201, 202, 203, 204, 205)
BOOTSTRAP_SEED = 7
SPLIT_SEED = 20260627
N_BOOT = 10000

# generation firewall (prereg §0) — the latloc job asserts the live decoder count equals
# the value for its generation. NOTE: this is NOT enforced by relaw.n_decoder_layers (that
# only asserts the index set is non-empty — verified against the tree 2026-06-28); the
# numeric assertion lives in the latloc job, see assert_generation_layers().
GEMMA2_N_LAYERS = 26
GEMMA1_N_LAYERS = 18


# ===========================================================================
# Recovery fraction (prereg §2, §6).
# ===========================================================================

def rf(acc_patched: float, floor: float, intact: float) -> float:
    """RF on set-level accuracies: 0 at floor, 1 at intact. nan if intact≈floor (no
    dynamic range — the preflight gate §9 forbids running in that regime)."""
    denom = intact - floor
    if abs(denom) < 1e-9:
        return float("nan")
    return (acc_patched - floor) / denom


def rf_from_items(patched_items, floor_items, intact_items) -> float:
    """RF from per-item 0/1 arrays (same items, paired by position)."""
    return rf(float(np.mean(patched_items)), float(np.mean(floor_items)), float(np.mean(intact_items)))


def delta_rf_point(correct_items, wrong_items, floor_items, intact_items) -> float:
    """ΔRF = RF(correct-donor) − RF(wrong-donor). The floor cancels in the numerator
    (ΔRF = (mean(correct) − mean(wrong)) / (intact − floor)); kept explicit for clarity."""
    return (rf_from_items(correct_items, floor_items, intact_items)
            - rf_from_items(wrong_items, floor_items, intact_items))


def bootstrap_delta_rf(correct_items, wrong_items, floor_items, intact_items,
                       seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT):
    """Item-level paired bootstrap of ΔRF. All four arrays are item-paired (same resample
    indices each draw). Returns (point, ci_lo, ci_hi) at 95%. The confirmation ΔRF(L*)
    CI-lower is the headline; H_localized requires ci_lo > DELTA (prereg §6)."""
    correct_items = np.asarray(correct_items, float)
    wrong_items = np.asarray(wrong_items, float)
    floor_items = np.asarray(floor_items, float)
    intact_items = np.asarray(intact_items, float)
    n = len(correct_items)
    if not (len(wrong_items) == len(floor_items) == len(intact_items) == n):
        raise ValueError("bootstrap_delta_rf: the four per-item arrays must be item-paired (equal length).")
    point = delta_rf_point(correct_items, wrong_items, floor_items, intact_items)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        f = float(np.mean(floor_items[idx]))
        it = float(np.mean(intact_items[idx]))
        d = it - f
        draws[b] = ((np.mean(correct_items[idx]) - np.mean(wrong_items[idx])) / d
                    if abs(d) >= 1e-9 else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_delta_rf_profile(correct_2d, wrong_2d, floor_items, intact_items,
                               seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT):
    """Vectorized per-layer paired bootstrap (the analysis driver). correct_2d/wrong_2d are
    (n_layers, n_items) 0/1; floor_items/intact_items are (n_items,). SHARED resample indices
    across layers (consistent item resampling; the figure's per-layer CIs are then comparable).
    Returns (points, ci_lo, ci_hi), each (n_layers,). Same estimator as `bootstrap_delta_rf`,
    just batched over layers without the Python resample loop."""
    correct_2d = np.asarray(correct_2d, float)
    wrong_2d = np.asarray(wrong_2d, float)
    floor = np.asarray(floor_items, float)
    intact = np.asarray(intact_items, float)
    nL, n = correct_2d.shape
    if not (wrong_2d.shape == correct_2d.shape and floor.shape == intact.shape == (n,)):
        raise ValueError("bootstrap_delta_rf_profile: shape mismatch (need (nL,n),(nL,n),(n,),(n,)).")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))                 # shared across layers
    denom = intact[idx].mean(1) - floor[idx].mean(1)      # (n_boot,)
    good = np.abs(denom) >= 1e-9
    points = np.array([delta_rf_point(correct_2d[L], wrong_2d[L], floor, intact) for L in range(nL)])
    lo = np.full(nL, np.nan)
    hi = np.full(nL, np.nan)
    for L in range(nL):
        drf = np.where(good, (correct_2d[L][idx].mean(1) - wrong_2d[L][idx].mean(1)) / denom, np.nan)
        lo[L], hi[L] = np.nanpercentile(drf, [2.5, 97.5])
    return points, lo, hi


def one_sided_p_delta_rf(correct_items, wrong_items, floor_items, intact_items,
                         seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT) -> float:
    """Bootstrap one-sided p for H0: ΔRF ≤ 0 (fraction of resamples with ΔRF ≤ 0).
    Used per single layer to BH-FDR-nominate L* on the discovery half (prereg §6)."""
    correct_items = np.asarray(correct_items, float)
    wrong_items = np.asarray(wrong_items, float)
    floor_items = np.asarray(floor_items, float)
    intact_items = np.asarray(intact_items, float)
    n = len(correct_items)
    rng = np.random.default_rng(seed)
    le0 = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = float(np.mean(intact_items[idx]) - np.mean(floor_items[idx]))
        if abs(d) < 1e-9:
            continue
        drf = (np.mean(correct_items[idx]) - np.mean(wrong_items[idx])) / d
        le0 += int(drf <= 0.0)
    return le0 / n_boot


# ===========================================================================
# Step statistic (prereg §6) — the concentrated-vs-smooth adjudicator.
# ===========================================================================

def step_statistic(delta_rf_curve):
    """max normalized second difference of the ΔRF-vs-depth curve (prereg §6):
        step = max_L |ΔRF[L+1] − 2ΔRF[L] + ΔRF[L−1]| / scale,  scale = typical slope.
    Returns (step_value, argmax_layer). A STEP is declared iff step > K_STEP AND the argmax
    layer coincides (±1) with L* (the caller checks the ±1).

    Denominator robustness (refined from the §6 draft's bare median): the draft normalized by
    median(|first diff|), which is EXACTLY 0 for a clean step (a [0,0,1,1] curve has mostly-zero
    first differences) → a nan on the very case the statistic is meant to flag. We use the median
    when it carries the noise scale, fall back to the mean if the median is ~0, and return +inf
    (a maximal step) for a perfectly flat-then-jump curve with no slope scale at all. Real ΔRF
    always carries finite-sample noise so this fallback only triggers on the clean synthetic."""
    c = np.asarray(delta_rf_curve, float)
    if c.size < 3:
        raise ValueError("step_statistic needs at least 3 layers.")
    second = np.abs(c[2:] - 2.0 * c[1:-1] + c[:-2])     # len n-2, position i ↔ layer i+1
    first = np.abs(np.diff(c))                           # len n-1
    max_second = float(np.max(second))
    arg = int(np.argmax(second)) + 1                     # +1 → real layer index
    denom = float(np.median(first))
    if denom < 1e-6:
        denom = float(np.mean(first))
    if denom < 1e-6:
        return (float("inf") if max_second > 1e-9 else 0.0), (arg if max_second > 1e-9 else None)
    return max_second / denom, arg


def stable_onset_depth(delta_rf_curve, delta: float = DELTA):
    """Earliest layer L from which ΔRF(L') ≥ delta for ALL deeper L' — the depth from which
    restoration is STABLY item-specific. Returns L, or None if even the last layer is < delta.

    This is the concentrated-vs-smooth ADJUDICATOR (replacing the brittle second-difference step
    statistic; dry-run finding 2026-06-28): a LOCALIZED deletion crosses δ once at the locus and
    STAYS above it (a clean mid-stack onset); a DISTRIBUTED deletion yields a JAGGED ΔRF that
    oscillates in and out of δ and only stabilizes at the trivial final layer(s) (shared head).
    Because a jagged distributed curve has a HIGH max-second-difference, `step_statistic` does not
    separate the two — it is retained as a DESCRIPTIVE secondary only. The onset reads directly into
    the §6 novelty gate: a final-1–2-layer onset is the trivial/expected outcome; a mid-stack onset
    is the novel localized result."""
    c = np.asarray(delta_rf_curve, float)
    for L in range(c.size):
        if np.all(c[L:] >= delta):
            return L
    return None


# ===========================================================================
# Greedy minimal restoring-set + selection-matched null (prereg §6, §7).
# The set-ΔRF is NON-additive (patching a set must be MEASURED, not summed), so these
# take a `set_delta_rf_fn(layer_tuple) -> float` callback — the GPU driver supplies the
# patch+eval one; the synthetic supplies an analytic one (dry-run).
# ===========================================================================

def greedy_minimal_set(single_layer_delta_rf, set_delta_rf_fn, delta: float = DELTA,
                       max_size: int | None = None):
    """Order layers by DESCENDING single-layer ΔRF; grow the set one at a time; return the
    smallest prefix S whose MEASURED set-ΔRF ≥ delta (else the full ordered set).
    Returns (S_sorted, sizes, set_delta_rf_curve). NOTE (dry-run finding): in the
    inject-and-survive patching paradigm a single late layer often restores under BOTH
    hypotheses, so |S| is a WEAK discriminator — the step statistic is the adjudicator."""
    order = list(np.argsort(-np.asarray(single_layer_delta_rf, float)))
    if max_size is not None:
        order = order[:max_size]
    S: list[int] = []
    sizes, curve = [], []
    for L in order:
        S.append(int(L))
        d = float(set_delta_rf_fn(tuple(sorted(S))))
        sizes.append(len(S))
        curve.append(d)
        if d >= delta:
            return sorted(S), sizes, curve
    return sorted(S), sizes, curve


def selected_vs_selected_null(set_delta_rf_fn, n_layers: int, target_delta: float = DELTA,
                              seeds=WRONG_DONOR_SEEDS, max_size: int | None = None):
    """Greedy on SHUFFLED layer order (label-blind selection) → the selection-matched null:
    how small a set a greedy that ignores the ΔRF signal reaches `target_delta`. The real
    greedy |S| must beat this (prereg §7). Returns [|S| per seed] (np.inf if never reached)."""
    out = []
    cap = n_layers if max_size is None else min(max_size, n_layers)
    for s in seeds:
        order = list(np.random.default_rng(s).permutation(n_layers))
        S: list[int] = []
        reached = np.inf
        for L in order[:cap]:
            S.append(int(L))
            if float(set_delta_rf_fn(tuple(sorted(S)))) >= target_delta:
                reached = len(S)
                break
        out.append(reached)
    return out


def random_layer_sets(n_layers: int, size: int, seeds=RANDOM_NULL_SEEDS):
    """Matched-SIZE random layer sets (size control ONLY — NOT depth-matched; prereg §5/§8
    row 3). One set per seed."""
    if size > n_layers:
        raise ValueError(f"random_layer_sets: size {size} > n_layers {n_layers}")
    return [tuple(sorted(np.random.default_rng(s).choice(n_layers, size, replace=False).tolist()))
            for s in seeds]


def donor_permutation(n: int, seed: int):
    """Item-shuffled donor index, donor != self (the make_pairs permutation logic,
    lm_patching.py:213-215). Applied WITHIN a same-length bucket so the patched residuals
    are shape- AND position-aligned (prereg §4; Catch-5 same-length-bucket rule)."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.where(perm == np.arange(n), (perm + 1) % n, perm)


def bh_fdr(pvals, q: float = FDR_Q):
    """Benjamini-Hochberg step-up; returns a boolean reject mask (prereg §6 nomination)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(n, bool)
    kmax = int(np.max(np.where(passed)[0]))
    cut = p[order][kmax]
    return p <= cut


# ===========================================================================
# Length-bucketed wrong-donor alignment (Catch-5) — pure index logic, no tensors.
# ===========================================================================

def length_buckets(token_lengths, min_bucket: int = 2):
    """Group item indices by EXACT token length so a within-bucket donor swap is
    position-for-position aligned (prereg §4). Returns {length: [item_idx,...]} keeping
    only buckets with ≥ min_bucket items (a singleton has no valid donor)."""
    lengths = np.asarray(token_lengths)
    out: dict[int, list[int]] = {}
    for i, L in enumerate(lengths.tolist()):
        out.setdefault(int(L), []).append(i)
    return {L: idx for L, idx in out.items() if len(idx) >= min_bucket}


# ===========================================================================
# Synthetic linear stack with a SHARED readout — weight-free validation (prereg §13).
# A tiny residual-stream model: intact and deleted differ ONLY in their per-layer maps;
# the readout W is SHARED (the shared-lm_head analog). Orthogonal maps preserve norm so
# the stack neither vanishes nor explodes.
# ===========================================================================

def _rand_orth(d, rng):
    """A random orthogonal d×d (QR of a Gaussian)."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))                        # fix QR sign ambiguity → uniform Haar


def _near_identity_orth(d, rng, eps):
    """A small rotation: orthonormalize I + eps·skew (eps→0 ⇒ identity)."""
    g = rng.standard_normal((d, d))
    skew = (g - g.T) / 2.0
    q, _ = np.linalg.qr(np.eye(d) + eps * skew)
    return q


class SyntheticStack:
    """Tiny linear residual stack for the weight-free dry-run (prereg §13).

    mode='planted'     : deleted == intact except ONE layer `locus` carries a full
                         corruption rotation → the deletion is item-specific & localized.
    mode='distributed' : every layer carries a SMALL corruption → smooth, no locus.
    mode='contentfree' : the deleted final layer projects onto a fixed vector → the
                         readout is input-independent, so correct- and wrong-donor patches
                         restore EQUALLY → ΔRF≈0 (tests that the subtraction is honest).
    """

    def __init__(self, n_layers=12, d=16, n_items=400, n_choices=4, mode="planted",
                 locus=6, eps=0.18, seed=0):
        if mode == "planted" and not (0 <= locus < n_layers):
            raise ValueError(f"planted locus {locus} out of range [0,{n_layers})")
        rng = np.random.default_rng(seed)
        self.n_layers, self.d, self.n_items, self.n_choices = n_layers, d, n_items, n_choices
        self.mode, self.locus = mode, locus
        self.A = [_rand_orth(d, rng) for _ in range(n_layers)]    # intact per-layer maps
        corrupt = _rand_orth(d, rng)                              # a full scramble for the locus
        self.B = []
        for L in range(n_layers):
            if mode == "planted":
                self.B.append(self.A[L] @ corrupt if L == locus else self.A[L].copy())
            elif mode == "distributed":
                self.B.append(self.A[L] @ _near_identity_orth(d, rng, eps))
            elif mode == "contentfree":
                self.B.append(self.A[L].copy())
            else:
                raise ValueError(f"unknown mode {mode!r}")
        # SHARED readout with NEAR-ORTHOGONAL rows (d >> n_choices) so each class direction is
        # distinct. Build the synthetic BACKWARDS from BALANCED answers (a[m]=m%n_choices): place
        # each intact final on its class's readout row so a WRONG-donor patch genuinely hands over a
        # DIFFERENT answer ~ (n_choices-1)/n_choices of the time. (Defining answers as argmax of a
        # random readout instead — the first dry-run version — skews the answer distribution, and a
        # skewed wrong-donor matches by chance far above 1/n_choices, falsely collapsing ΔRF.)
        self.W = rng.standard_normal((n_choices, d))
        self.answer = np.arange(n_items) % n_choices
        h_intact = 3.0 * self.W[self.answer] + 0.1 * rng.standard_normal((n_items, d))
        # intact_final = embed @ M.T with M = A[n-1] @ … @ A[0]; choose embed = h_intact @ M so the
        # intact final equals h_intact (⇒ argmax reads the planted balanced answer cleanly).
        M = np.eye(d)
        for L in range(n_layers):
            M = self.A[L] @ M
        self.embed = h_intact @ M
        if mode == "contentfree":
            self._fixed = rng.standard_normal(d)                 # deleted final → fixed point

    def _intact_residuals(self):
        """List L=0..n of intact residuals (L=0 is the embedding, L=k is post-layer-k)."""
        res = [self.embed.copy()]
        h = self.embed.copy()
        for L in range(self.n_layers):
            h = h @ self.A[L].T
            res.append(h.copy())
        return res

    def acc_patched(self, donor_idx, S):
        """Run the DELETED forward; at each layer in S replace the post-layer residual with
        the intact donor's residual at that layer; read the SHARED head. Returns per-item 0/1.
        donor_idx: (n_items,) — arange ⇒ correct-donor; a permutation ⇒ wrong-donor."""
        S = set(int(x) for x in S)
        intact = self._intact_residuals()
        h = self.embed.copy()
        for L in range(self.n_layers):
            h = h @ self.B[L].T                              # deleted layer L → post-layer residual
            if self.mode == "contentfree" and L == self.n_layers - 1:
                h = np.broadcast_to(self._fixed, h.shape).copy()   # input-independent collapse
            if L in S:
                h = intact[L + 1][donor_idx].copy()          # patch post-layer-L residual
        pred = np.argmax(h @ self.W.T, axis=1)
        return (pred == self.answer).astype(int)

    def floor_items(self):
        return self.acc_patched(np.arange(self.n_items), ())        # pure deleted

    def intact_items(self):
        return np.ones(self.n_items, dtype=int)                     # intact acc ≡ 1 by construction

    def set_delta_rf(self, S, donor_seed=WRONG_DONOR_SEEDS[0]):
        """Measured set-ΔRF: RF(correct-donor, S) − RF(wrong-donor, S), within-pool donor."""
        floor = self.floor_items()
        intact = self.intact_items()
        correct = self.acc_patched(np.arange(self.n_items), S)
        wrong = self.acc_patched(donor_permutation(self.n_items, donor_seed), S)
        return delta_rf_point(correct, wrong, floor, intact)

    def single_layer_delta_rf(self, donor_seeds=WRONG_DONOR_SEEDS):
        """ΔRF(L) for every single layer (correct − wrong-donor), AVERAGED over the wrong-donor
        seeds — matching the real protocol (prereg §6 n_seeds=5), which denoises the discrete-argmax
        sampling noise in the wrong-donor term."""
        seeds = (donor_seeds,) if isinstance(donor_seeds, int) else tuple(donor_seeds)
        curves = [[self.set_delta_rf((L,), s) for L in range(self.n_layers)] for s in seeds]
        return np.mean(curves, axis=0)


# ===========================================================================
# Weight-free dry-run (prereg §13 checks 2,3,4,5-logic) — calibrates k.
# Returns a JSON-able dict; RAISES on any failed invariant (fail-loud).
# ===========================================================================

def dry_run(verbose: bool = False) -> dict:
    out: dict = {"checks": {}, "k_calibration": {}}

    # ---- Check 4: RF math ----------------------------------------------------
    assert abs(rf(0.25, 0.25, 0.80) - 0.0) < 1e-9, "RF at floor must be 0"
    assert abs(rf(0.80, 0.25, 0.80) - 1.0) < 1e-9, "RF at intact must be 1"
    assert abs(rf(0.525, 0.25, 0.80) - 0.5) < 1e-9, "RF midpoint must be 0.5"
    assert np.isnan(rf(0.5, 0.5, 0.5)), "RF with no dynamic range must be nan"
    out["checks"]["rf_math"] = "pass"

    # ---- Check 3: item-specific locus recovered + k calibration -------------
    planted = SyntheticStack(mode="planted", locus=6, seed=1)
    dist = SyntheticStack(mode="distributed", eps=0.18, seed=1)

    p_floor = float(planted.floor_items().mean())
    d_floor = float(dist.floor_items().mean())
    # the deletion must actually bite (deleted ≪ intact=1) or there is no dynamic range.
    assert p_floor < 0.6, f"planted deleted-floor not depressed ({p_floor})"
    assert d_floor < 0.85, f"distributed deleted-floor not depressed ({d_floor})"
    out["checks"]["floor_depressed"] = {"planted": p_floor, "distributed": d_floor}

    p_curve = planted.single_layer_delta_rf()
    d_curve = dist.single_layer_delta_rf()
    out["curves"] = {"planted_delta_rf": [round(float(x), 3) for x in p_curve],
                     "distributed_delta_rf": [round(float(x), 3) for x in d_curve]}

    # planted: ΔRF must SPIKE from ~0 (L<locus) to ~large (L>=locus); confirm the locus.
    pre = float(p_curve[:planted.locus].max()) if planted.locus > 0 else 0.0
    at = float(p_curve[planted.locus])
    assert at - pre > DELTA, f"planted ΔRF did not spike at the locus (pre={pre}, at={at})"
    # earliest crossing L* must equal the planted locus.
    l_star = int(np.argmax(p_curve >= DELTA))
    assert l_star == planted.locus, f"planted L* {l_star} != planted locus {planted.locus}"
    out["checks"]["planted_locus_recovered"] = {"L_star": l_star, "delta_rf_pre": pre, "delta_rf_at": at}

    # item-specificity at the locus = the prereg δ margin: the CORRECT donor must restore at least
    # δ MORE than the depth-matched WRONG donor. The wrong donor restoring SOME is EXPECTED (it still
    # hands over a coherent, if wrong, answer at chance — the baseline ΔRF subtracts); item-specificity
    # is correct−wrong, NOT wrong≈0 (the over-strict first version asserted wrong≈0 and tripped on the
    # coherent-answer baseline).
    correct_at_locus = rf_from_items(
        planted.acc_patched(np.arange(planted.n_items), (planted.locus,)),
        planted.floor_items(), planted.intact_items())
    w_at_locus = rf_from_items(
        planted.acc_patched(donor_permutation(planted.n_items, WRONG_DONOR_SEEDS[0]), (planted.locus,)),
        planted.floor_items(), planted.intact_items())
    assert correct_at_locus - w_at_locus >= DELTA, (
        f"item-specific margin {correct_at_locus - w_at_locus:.3f} < δ={DELTA} "
        f"(correct_rf={correct_at_locus:.3f} wrong_rf={w_at_locus:.3f})")
    out["checks"]["item_specific_margin"] = {
        "correct_rf": round(correct_at_locus, 3), "wrong_rf": round(w_at_locus, 3),
        "margin": round(correct_at_locus - w_at_locus, 3)}

    # step statistic. HARD requirement: planted and distributed are SEPARABLE (p_step > d_step)
    # and the planted step lands at the locus. SOFT: whether k=3 sits between them is a calibration
    # outcome — if k mis-separates, the prereg §6 permits re-freezing k on these synthetics (never
    # on real data); the dry-run records the suggested value rather than hard-failing.
    p_step, p_arg = step_statistic(p_curve)
    d_step, d_arg = step_statistic(d_curve)
    assert p_step > d_step, f"planted step {p_step:.2f} not > distributed {d_step:.2f} — design not separable"
    assert abs(p_arg - planted.locus) <= 1, f"planted step argmax {p_arg} not within ±1 of locus {planted.locus}"
    k_separates = bool(d_step <= K_STEP < p_step)
    # a finite midpoint suggestion when either step is inf (clean synthetic) uses the finite side.
    if np.isfinite(p_step) and np.isfinite(d_step):
        suggested_k = round((p_step + d_step) / 2.0, 3)
    else:
        suggested_k = round(d_step * 2.0, 3) if np.isfinite(d_step) else K_STEP
    out["k_calibration"] = {"planted_step": p_step, "planted_argmax": p_arg,
                            "distributed_step": d_step, "distributed_argmax": d_arg,
                            "k_frozen": K_STEP, "k_separates": k_separates,
                            "suggested_k_if_refreeze": suggested_k}
    out["checks"]["step_discriminates"] = (
        "descriptive-only; demoted (jagged distributed → high step)" if not k_separates
        else "pass")

    # stable-onset depth — the ROBUST concentration adjudicator (the step statistic is fragile to a
    # jagged distributed ΔRF; the onset is not). Planted ⇒ mid-stack onset at the locus; distributed
    # ⇒ onset only at the trivial final layer(s) (or None).
    p_onset = stable_onset_depth(p_curve)
    d_onset = stable_onset_depth(d_curve)
    assert p_onset == planted.locus, f"planted stable-onset {p_onset} != locus {planted.locus}"
    assert d_onset is None or d_onset >= dist.n_layers - 2, (
        f"distributed stable-onset {d_onset} should be final-1-2-layer/None (trivial), not mid-stack")
    out["checks"]["stable_onset_separates"] = {
        "planted_onset": p_onset, "distributed_onset": d_onset, "n_layers": planted.n_layers,
        "reading": "planted=mid-stack locus (novel); distributed=final-layer-only (trivial per §6 gate)"}

    # greedy minimal set (secondary): record |S| under both; the dry-run DOCUMENTS that |S|
    # is weak (a single late layer restores under both) — the step is the real adjudicator.
    pS, _, _ = greedy_minimal_set(p_curve, planted.set_delta_rf)
    dS, _, _ = greedy_minimal_set(d_curve, dist.set_delta_rf)
    out["checks"]["minimal_set_is_weak"] = {"planted_S_size": len(pS), "distributed_S_size": len(dS)}

    # ---- Check 2: depth-injection baseline / honest subtraction (demonstrates Catch-6) ----
    # contentfree: the deleted final layer collapses to an input-independent fixed point, so a
    # patch at any NON-FINAL layer is washed out ⇒ ΔRF≈0 there (the subtraction invents nothing).
    # But the FINAL layer feeds the SHARED head directly, so a final-layer patch is trivially
    # item-specific ⇒ ΔRF≈1 there even with NO planted locus. This is exactly why ΔRF does NOT
    # cancel the shared-head depth artifact (Catch-6) and why the §6 NOVELTY GATE must exclude a
    # final-layer locus. The dry-run pins both halves of that statement.
    cf = SyntheticStack(mode="contentfree", seed=1)
    assert cf.intact_items().mean() - cf.floor_items().mean() > 0.3, "content-free needs dynamic range"
    cf_drf = cf.single_layer_delta_rf()
    nonfinal_max = float(np.nanmax(np.abs(cf_drf[:-1])))
    final_drf = float(cf_drf[-1])
    assert nonfinal_max < DELTA, f"content-free non-final ΔRF should be ~0, got max|{nonfinal_max}|"
    assert final_drf > DELTA, (
        f"content-free FINAL-layer ΔRF should be trivially high (shared-head injection), got {final_drf}")
    out["checks"]["contentfree_nonfinal_zero_final_trivial"] = {
        "nonfinal_max_abs_delta_rf": nonfinal_max, "final_layer_delta_rf": final_drf}

    # ---- Check 7: vectorized profile bootstrap (the analysis driver) -------
    # CI-lower must clear δ exactly at/after the planted locus, and NOT before it.
    p_correct = np.array([planted.acc_patched(np.arange(planted.n_items), (L,))
                          for L in range(planted.n_layers)])
    p_wrong = np.array([planted.acc_patched(donor_permutation(planted.n_items, WRONG_DONOR_SEEDS[0]), (L,))
                        for L in range(planted.n_layers)])
    _pts, _lo, _hi = bootstrap_delta_rf_profile(
        p_correct, p_wrong, planted.floor_items(), planted.intact_items(), n_boot=2000)
    assert _lo[planted.locus] > DELTA, f"profile-bootstrap CI-lower at locus {_lo[planted.locus]:.3f} !> δ"
    assert _lo[0] <= DELTA, f"profile-bootstrap CI-lower at L0 {_lo[0]:.3f} should be ≤ δ (no early locus)"
    out["checks"]["profile_bootstrap_ci"] = {
        "locus_ci_lo": round(float(_lo[planted.locus]), 3), "L0_ci_lo": round(float(_lo[0]), 3)}

    # ---- Check 5 (logic): answer-leak guard --------------------------------
    # MCQ forward has NO answer suffix (Catch-4); cloze does. The guard: when labels mark
    # answer positions, pos_mask='all' must EXCLUDE them (patching them would give RF≡1).
    attn = np.array([1, 1, 1, 1, 1], bool)
    labels = np.array([-100, -100, -100, 7, 9])         # last two are answer-suffix tokens
    is_prompt = attn & (labels == -100)
    assert not is_prompt[3] and not is_prompt[4], "answer-suffix positions must be excluded"
    assert is_prompt[:3].all(), "prompt positions must be patchable"
    # MCQ case: labels all -100 (no suffix) ⇒ the whole attended sequence is patchable.
    mcq_labels = np.array([-100, -100, -100, -100, -100])
    mcq_prompt = attn & (mcq_labels == -100)
    assert mcq_prompt.all(), "MCQ (no suffix): the full attended prompt is patchable"
    out["checks"]["answer_leak_guard"] = "pass"

    if verbose:
        import json
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(dry_run(), indent=2))
