"""latablate — pure-logic core for the LATLOC-ABLATE deletion-removability study.

Companion to `studies/lineage/paligemma2/research/2026-06-29_latloc_ablate_prereg.md`
(workstream slug `latablate`). PURE numpy — no torch, no model, no GPU — so it py_compiles
and unit-tests on the control plane. The GPU orchestration (residual harvest, the AblateHook
projector, the k-sweep eval) lives in `infra/aws_pg/jobs/latablate_cell.py` and calls into this
module for ALL of the subspace-estimation / projection / RF / null arithmetic, so the analysis
logic is provable weight-free (prereg §13 weight-free dry-run).

The question (prereg §1). LATLOC localized the Gemma-2 blank-image MMLU-MCQ deletion to a
mid-stack site (L16/26): patching the intact residual at L16 into the deleted forward restores
MCQ item-specifically, the corruption there is GENERIC + SATURATING (any deleted residual at L16
collapses intact MCQ to chance), and a localized WEIGHT-restore is null. This study asks whether
that generic corruption is a LOW-RANK "MCQ-disabling" DIRECTION at L16 that can be PROJECTED OUT
of the deleted forward — input-independent, no intact model, no retraining.

Estimand (prereg §2). Primary readout
    RF(k) = (acc_ablate(k) − floor)/(intact − floor)   on blank-image MMLU-MCQ accuracy,
where acc_ablate(k) is the DELETED model run with a fixed rank-k projector P_k = V_k V_kᵀ ablated
at L16 (h ← h − V_k V_kᵀ h, all positions). V_k = top-k right singular vectors of the (deleted −
intact) L16 LAST-TOKEN residual-difference matrix, estimated ONCE on discovery items (frozen).
Floor = deleted MCQ (k=0); intact = intact MCQ; both on the SAME items, paired by item.

Two nulls (prereg §5), with their roles made precise by the dry-run (see dry_run() check 2):
  * RANDOM-DIRECTION null (LOAD-BEARING): ablate k random orthonormal directions. real RF(k)
    must beat this by δ_rand. This is the artifact-closer — "the disabling subspace is SPECIFIC,
    not generic k-dim removal".
  * SHUFFLE-subspace null (CHARACTERIZATION, not a pass/fail gate): V_k from item-shuffled
    (deleted − intact) pairs. If the disabling direction is a GENERIC item-independent shift (the
    latloc finding), shuffling PRESERVES it → shuffle recovers too; if the direction is
    item-SPECIFIC, mismatched pairing destroys it → shuffle fails. So shuffle≈real ⟺ generic
    direction (consistent with latloc), shuffle≪real ⟺ item-specific — informative either way,
    but it does NOT gate H_artifact (the random-direction null does).

Discipline: no try/except hiding; bad shapes/degenerate denominators RAISE or return nan with an
explicit guard. No model-name or baseline literals are baked in. Reuses latloc.rf / rf_from_items.
"""
from __future__ import annotations

import numpy as np

from infra.pg.latloc import GEMMA2_N_LAYERS, rf, rf_from_items  # noqa: F401  (rf reused; firewall const)

# ===========================================================================
# Frozen thresholds (prereg §6) — SINGLE SOURCE OF TRUTH, imported by the driver
# and the dry-run. Changing any of these is a re-freeze and must be journalled.
# ===========================================================================
TAU_RESTORE = 0.50          # "substantial restoration": recover ≥ half the deleted→intact gap
K_SMALL = 8                 # the practical (low-rank) claim requires the minimal restoring k* ≤ 8
DELTA_RAND = 0.20           # real RF(k*) must exceed the equal-k random-direction null by ≥ this
K_GRID = (0, 1, 2, 4, 8, 16, 32)                 # the frozen ablation-rank sweep
RANDOM_NULL_SEEDS = (301, 302, 303, 304, 305)    # equal-k random-orthonormal-direction null
SHUFFLE_NULL_SEEDS = (401, 402, 403, 404, 405)   # item-shuffled (deleted−intact) subspace null
BOOTSTRAP_SEED = 7
N_BOOT = 10000
CLOZE_RETAIN_DELTA = 0.05   # frozen (prereg §6, audit S2): cloze content "retained" iff
#                             cloze_acc_ablated ≥ cloze_acc_floor − CLOZE_RETAIN_DELTA (no collapse)

# The frozen locus (prereg §0): LATLOC's confirmed mid-stack onset. The ablation acts at THIS
# decoder layer only. Imported, not re-derived — recompute the latloc onset from its banked JSON
# at analysis time if it is ever in doubt (never quote it from prose).
L_LOCUS = 16


# ===========================================================================
# Disabling-subspace estimator (prereg §2, §6) — top-k SVD of (deleted − intact).
# UNcentered: the disabling corruption may be a near-CONSTANT shift across items (the latloc
# "generic" finding); centering D would remove exactly that shift, so we do NOT center. The
# rank-1 mean-difference direction is reported alongside as the "common shift" candidate.
# ===========================================================================

def estimate_disabling_subspace(diff_matrix, k: int):
    """Top-k right singular vectors of the (deleted − intact) residual-difference matrix.

    diff_matrix: (n_items, d) — per-item last-token residual differences at the locus.
    Returns V: (d, k) with ORTHONORMAL columns (the disabling subspace), or (d, 0) for k=0.
    The columns are ordered by descending singular value (dominant disabling directions first)."""
    D = np.asarray(diff_matrix, float)
    if D.ndim != 2:
        raise ValueError(f"estimate_disabling_subspace: diff_matrix must be 2-D (n,d), got {D.shape}")
    n, d = D.shape
    if k < 0 or k > min(n, d):
        raise ValueError(f"estimate_disabling_subspace: k={k} out of range [0, min(n,d)={min(n, d)}]")
    if k == 0:
        return np.zeros((d, 0))
    # economy SVD: D = U S Vh, rows of Vh are the right singular vectors (directions in R^d).
    _U, _S, Vh = np.linalg.svd(D, full_matrices=False)
    return np.ascontiguousarray(Vh[:k].T)            # (d, k), orthonormal columns


def mean_difference_direction(diff_matrix):
    """The rank-1 'common shift' candidate: the unit mean of (deleted − intact) over items,
    (d, 1). For a near-constant disabling shift this coincides with the top SVD direction;
    reported alongside the SVD subspace (prereg §2, threats-row 2)."""
    D = np.asarray(diff_matrix, float)
    m = D.mean(axis=0)
    nrm = np.linalg.norm(m)
    if nrm < 1e-12:
        return np.zeros((D.shape[1], 1))
    return (m / nrm).reshape(-1, 1)


def energy_subspace(deleted_last, k: int):
    """The energy-matched null (prereg §5, audit S3): top-k right singular vectors of the DELETED
    last-token residual itself (uncentered) — the highest-ENERGY directions of the deleted stream,
    NOT difference-specific. Projecting these out tests whether "restoration" is merely "remove any
    high-energy direction" (e.g. a Gemma-2 massive-activation dim) vs removing the disabling
    (deleted − intact) direction specifically. Deterministic; reuses the SVD core."""
    return estimate_disabling_subspace(deleted_last, k)


def is_monotone_nondecreasing(rf_by_k: dict, up_to_k) -> bool:
    """RF(k) non-decreasing across the grid up to and including `up_to_k` (dose-response descriptor,
    prereg §7/§11; audit S1 — a discovery descriptor, NOT a hard verdict conjunct). up_to_k None →
    check the whole grid. A single dip is a flag, not a result."""
    ks = sorted(k for k in rf_by_k if up_to_k is None or k <= up_to_k)
    vals = [rf_by_k[k] for k in ks]
    return all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1))


def random_subspace(d: int, k: int, seed: int):
    """A matched-size random orthonormal subspace (d, k) — the LOAD-BEARING null (prereg §5).
    QR of a Gaussian; sign-fixed so it is reproducible. (d, 0) for k=0."""
    if k < 0 or k > d:
        raise ValueError(f"random_subspace: k={k} out of range [0, d={d}]")
    if k == 0:
        return np.zeros((d, 0))
    g = np.random.default_rng(seed).standard_normal((d, k))
    q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))                    # (d, k), orthonormal columns


def shuffle_subspace(deleted_last, intact_last, k: int, seed: int):
    """Item-shuffled (deleted − intact) subspace — the CHARACTERIZATION null (prereg §5). Pairs
    each deleted item with a DIFFERENT intact item (a derangement) before differencing, then takes
    the top-k SVD. A generic constant shift survives the shuffle (top dir unchanged); an
    item-specific direction does not. Returns (d, k)."""
    A = np.asarray(deleted_last, float)
    B = np.asarray(intact_last, float)
    if A.shape != B.shape:
        raise ValueError(f"shuffle_subspace: deleted/intact shape mismatch {A.shape} vs {B.shape}")
    n = A.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)   # derangement: donor != self
    return estimate_disabling_subspace(A - B[perm], k)


# ===========================================================================
# The projector (prereg §2). h ← h − (h V) Vᵀ. Input-independent; numpy mirror of AblateHook.
# ===========================================================================

def project_out(H, V):
    """Remove span(V) from each row of H: H − (H V) Vᵀ. V is (d, k) with orthonormal columns;
    H is (..., d). k=0 (V is (d,0)) is the identity (the floor). Pure numpy — the AblateHook torch
    forward computes the SAME thing on output[0]."""
    H = np.asarray(H, float)
    V = np.asarray(V, float)
    if V.shape[0] != H.shape[-1]:
        raise ValueError(f"project_out: V rows {V.shape[0]} != H last dim {H.shape[-1]}")
    return H - (H @ V) @ V.T


def is_orthonormal(V, tol: float = 1e-6) -> bool:
    """VᵀV ≈ I_k (empty V trivially passes)."""
    V = np.asarray(V, float)
    if V.shape[1] == 0:
        return True
    return bool(np.allclose(V.T @ V, np.eye(V.shape[1]), atol=tol))


def singular_spectrum(diff_matrix, k_top: int | None = None) -> dict:
    """Singular values of (deleted − intact) + cumulative-energy fraction (prereg F2). A steep
    spectrum (top-k energy ≈ 1 for small k) ⇒ the disabling corruption is low-rank."""
    D = np.asarray(diff_matrix, float)
    s = np.linalg.svd(D, compute_uv=False)
    energy = s ** 2
    total = float(energy.sum())
    cum = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)
    kt = len(s) if k_top is None else min(k_top, len(s))
    return {"singular_values": [float(x) for x in s[:kt]],
            "cum_energy_fraction": [float(x) for x in cum[:kt]],
            "n_singular_values": int(len(s))}


# ===========================================================================
# Recovery fraction + bootstrap (prereg §6, §7). RF is reused from latloc (rf / rf_from_items);
# the single-RF item bootstrap is new here (latloc's bootstrap is for the ΔRF contrast).
# ===========================================================================

def bootstrap_rf(ablate_items, floor_items, intact_items,
                 seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT):
    """Item-level paired bootstrap of RF = (mean(ablate) − mean(floor))/(mean(intact) − mean(floor)).
    All three per-item 0/1 arrays are paired (same items, same resample indices each draw).
    Returns (point, ci_lo, ci_hi) at 95%. The confirmation RF(k*) CI-lower is the headline; H_lowrank
    requires ci_lo > TAU_RESTORE (prereg §6)."""
    a = np.asarray(ablate_items, float)
    f = np.asarray(floor_items, float)
    it = np.asarray(intact_items, float)
    n = len(a)
    if not (len(f) == len(it) == n):
        raise ValueError("bootstrap_rf: the three per-item arrays must be item-paired (equal length).")
    point = rf_from_items(a, f, it)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        denom = float(np.mean(it[idx]) - np.mean(f[idx]))
        draws[b] = (np.mean(a[idx]) - np.mean(f[idx])) / denom if abs(denom) >= 1e-9 else np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def minimal_k(rf_by_k: dict, tau: float = TAU_RESTORE):
    """Smallest k in the grid whose RF(k) ≥ tau (the minimal restoring k*). rf_by_k maps k→RF.
    Returns the int k*, or None if no k reaches tau. (k=0 is the floor: RF=0, never selected.)"""
    for k in sorted(rf_by_k):
        v = rf_by_k[k]
        if v is not None and not np.isnan(v) and v >= tau:
            return int(k)
    return None


def real_beats_random(rf_real: float, rf_rand_mean: float, delta: float = DELTA_RAND) -> bool:
    """The artifact gate (prereg §6): real RF(k) exceeds the equal-k random-direction null mean by δ."""
    return bool((rf_real - rf_rand_mean) > delta)


# ===========================================================================
# Synthetic last-token residuals with a SHARED linear readout — weight-free validation (§13).
# intact residuals encode a balanced answer along class-axis directions; deleted = intact + a
# planted corruption that DISABLES the readout. Orthonormal class axes so the readout is clean.
# ===========================================================================

def _rand_orth(d, rng):
    """A random orthogonal d×d (QR of a Gaussian), sign-fixed → Haar."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


class SyntheticAblateStack:
    """Tiny synthetic for the weight-free dry-run (prereg §13).

    mode='lowrank'      : the corruption is a near-CONSTANT shift along `r_plant` class axes that
                          the intact answer NEVER uses (answers ∈ {1..n_choices-1}; the corruption
                          pushes the reserved class 0). Adding it makes every item argmax the
                          reserved wrong class ⇒ readout disabled; because the disabling axis is
                          ORTHOGONAL to the signal axes, projecting it out restores cleanly (RF≈1).
                          This is the H_lowrank-positive, latloc-consistent (generic shift) case.
    mode='distributed'  : the corruption is a per-item random combination of ALL signal axes ⇒
                          high-rank and entangled with the signal; projecting any small-k subset
                          removes signal too ⇒ no clean small-k recovery (real ≲ τ ≈ random).
    """

    def __init__(self, d=64, n_items=600, n_choices=5, mode="lowrank", r_plant=1,
                 corrupt_scale=8.0, corrupt_noise=0.05, seed=0):
        if n_choices < 3:
            raise ValueError("need ≥3 choices (1 reserved disabling class + ≥2 answer classes)")
        if d <= max(K_GRID):
            raise ValueError(f"synthetic d={d} must exceed max(K_GRID)={max(K_GRID)}")
        rng = np.random.default_rng(seed)
        Q = _rand_orth(d, rng)
        E = Q[:, :n_choices]                       # (d, n_choices) orthonormal class axes (columns)
        self.W = E.T                               # readout rows = class axes; argmax = dominant class
        self.d, self.n_items, self.n_choices, self.mode = d, n_items, n_choices, mode
        used = np.arange(1, n_choices)             # class 0 is RESERVED (never an answer)
        self.answer = used[np.arange(n_items) % len(used)]
        self.intact = 3.0 * E[:, self.answer].T + 0.1 * rng.standard_normal((n_items, d))
        if mode == "lowrank":
            U = E[:, :r_plant]                     # disabling axes = the reserved class 0 (..r_plant)
            coeff = corrupt_scale * np.ones((n_items, r_plant))        # CONSTANT shift (generic)
            # + tiny isotropic noise so (deleted−intact) is FULL-RANK with a graded spectrum (the
            # dominant top-r is the disabling shift; the tail is noise) — mirrors real data, where
            # the top-k are always meaningful, not an arbitrary completion of an exactly rank-r D.
            self.deleted = self.intact + coeff @ U.T + corrupt_noise * rng.standard_normal((n_items, d))
            self.U_true = U
        elif mode == "distributed":
            G = rng.standard_normal((n_items, n_choices))             # per-item random class mix
            self.deleted = self.intact + corrupt_scale * (G @ E.T) / np.sqrt(n_choices)
            self.U_true = None
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def _acc(self, H):
        return (np.argmax(np.asarray(H, float) @ self.W.T, axis=1) == self.answer).astype(int)

    def floor_items(self):
        return self._acc(self.deleted)              # deleted == k=0 projector

    def intact_items(self):
        return self._acc(self.intact)

    def ablate_items(self, V):
        return self._acc(project_out(self.deleted, V))

    def diff_matrix(self):
        return self.deleted - self.intact           # (n,d) — what the estimator sees

    def rf_at(self, V):
        return rf_from_items(self.ablate_items(V), self.floor_items(), self.intact_items())


# ===========================================================================
# Weight-free dry-run (prereg §13) — RAISES on any failed invariant (fail-loud).
# ===========================================================================

def dry_run(verbose: bool = False) -> dict:
    out: dict = {"checks": {}}

    # ---- Check 4: RF math (reused latloc.rf) returns 0 at floor, 1 at intact ----
    assert abs(rf(0.25, 0.25, 0.80) - 0.0) < 1e-9, "RF at floor must be 0"
    assert abs(rf(0.80, 0.25, 0.80) - 1.0) < 1e-9, "RF at intact must be 1"
    assert abs(rf(0.525, 0.25, 0.80) - 0.5) < 1e-9, "RF midpoint must be 0.5"
    assert np.isnan(rf(0.5, 0.5, 0.5)), "RF with no dynamic range must be nan"
    out["checks"]["rf_math"] = "pass"

    # ---- Check 1: projector ALGEBRA (no readout) — SVD recovers a planted subspace; projecting
    #      it out of deleted equals projecting it out of intact (Δ is removed); idempotent;
    #      orthonormal; never zeros the residual. Pure linear algebra, mode-independent. ----
    rng = np.random.default_rng(0)
    d, n, r = 24, 300, 3
    U = _rand_orth(d, rng)[:, :r]                                    # planted rank-r subspace (d,r)
    H_i = rng.standard_normal((n, d))
    Delta = rng.standard_normal((n, r)) @ U.T                        # Δ ∈ span(U), per-item coeffs
    H_d = H_i + Delta
    V_r = estimate_disabling_subspace(H_d - H_i, r)
    assert is_orthonormal(V_r), "estimated V not orthonormal"
    # span(V_r) == span(U): the two rank-r projectors must coincide.
    P_U = U @ U.T
    P_V = V_r @ V_r.T
    assert np.allclose(P_U, P_V, atol=1e-6), f"recovered span != planted span (‖ΔP‖={np.abs(P_U - P_V).max():.2e})"
    # projecting span(U) out of deleted == out of intact (the Δ is gone): ‖Δ − PΔ‖ ≈ 0.
    resid = project_out(H_d, V_r) - project_out(H_i, V_r)
    assert np.linalg.norm(resid) < 1e-8, f"projection did not remove Δ (‖resid‖={np.linalg.norm(resid):.2e})"
    # idempotent: P² = P.
    P2 = P_V @ P_V
    assert np.allclose(P2, P_V, atol=1e-9), "projector not idempotent"
    # never zeros the residual: project_out keeps the complement (d−r dims) of a generic H.
    assert np.linalg.norm(project_out(H_d, V_r)) > 1e-3, "projector zeroed the whole residual"
    # k > r recovers a SUPERSET that still removes Δ; k < r leaves part of Δ.
    V_under = estimate_disabling_subspace(H_d - H_i, r - 1)
    assert np.linalg.norm(project_out(H_d, V_under) - project_out(H_i, V_under)) > 1e-6, \
        "under-rank projector should NOT fully remove Δ"
    out["checks"]["projector_algebra"] = {
        "span_recovered": True, "delta_removed_at_k>=r": True, "idempotent": True,
        "never_zero": True, "under_rank_leaves_delta": True}

    # ---- Check 2: RF separation — real direction restores, RANDOM does not (the load-bearing
    #      artifact gate); SHUFFLE recovers too on a GENERIC shift (documented, not a gate). ----
    low = SyntheticAblateStack(mode="lowrank", r_plant=1, seed=1)
    floor_acc = float(low.floor_items().mean())
    intact_acc = float(low.intact_items().mean())
    assert intact_acc - floor_acc > 0.4, f"lowrank needs dynamic range (intact {intact_acc}, floor {floor_acc})"
    Dlow = low.diff_matrix()
    V1_real = estimate_disabling_subspace(Dlow, 1)
    rf_real = low.rf_at(V1_real)
    rand_rfs = [low.rf_at(random_subspace(low.d, 1, s)) for s in RANDOM_NULL_SEEDS]
    rf_rand_mean = float(np.mean(rand_rfs))
    shuf_rfs = [low.rf_at(shuffle_subspace(low.deleted, low.intact, 1, s)) for s in SHUFFLE_NULL_SEEDS]
    rf_shuf_mean = float(np.mean(shuf_rfs))
    assert rf_real > TAU_RESTORE, f"real k=1 RF {rf_real:.3f} did not restore (≤ τ={TAU_RESTORE})"
    assert real_beats_random(rf_real, rf_rand_mean), \
        f"real {rf_real:.3f} does not beat random {rf_rand_mean:.3f} by δ={DELTA_RAND}"
    # mean-difference direction (the rank-1 'common shift') coincides with the top SVD dir here.
    mdir = mean_difference_direction(Dlow)
    assert abs(abs(float((mdir[:, 0] @ V1_real[:, 0]))) - 1.0) < 1e-2, \
        "mean-difference dir should ≈ the top SVD dir for a (near-)constant shift"
    out["checks"]["rf_separation_lowrank"] = {
        "floor_acc": round(floor_acc, 3), "intact_acc": round(intact_acc, 3),
        "rf_real_k1": round(rf_real, 3), "rf_random_k1_mean": round(rf_rand_mean, 3),
        "rf_shuffle_k1_mean": round(rf_shuf_mean, 3),
        "real_beats_random": True,
        "shuffle_reading": ("shuffle≈real ⇒ GENERIC shift (latloc-consistent), not an artifact; "
                            "the random null is the artifact gate")}

    # ---- Check 3: distributed mode — no clean small-k recovery (real ≲ τ ≈ random). H_distributed. ----
    dist = SyntheticAblateStack(mode="distributed", seed=1)
    Ddist = dist.diff_matrix()
    rf_real_small = {k: dist.rf_at(estimate_disabling_subspace(Ddist, k)) for k in (1, 2, 4, 8)}
    rf_rand_small = {k: float(np.mean([dist.rf_at(random_subspace(dist.d, k, s))
                                       for s in RANDOM_NULL_SEEDS])) for k in (1, 2, 4, 8)}
    # H_distributed signature: no small k (≤ K_small) reaches τ with a real-beats-random margin.
    distributed_low_rank_fails = all(
        not (rf_real_small[k] > TAU_RESTORE and real_beats_random(rf_real_small[k], rf_rand_small[k]))
        for k in (1, 2, 4, 8))
    assert distributed_low_rank_fails, \
        f"distributed mode spuriously low-rank-recovered: real={rf_real_small} rand={rf_rand_small}"
    out["checks"]["distributed_no_small_k_recovery"] = {
        "rf_real_small_k": {k: round(v, 3) for k, v in rf_real_small.items()},
        "rf_random_small_k": {k: round(v, 3) for k, v in rf_rand_small.items()},
        "low_rank_correctly_rejected": True}

    # ---- Check (nulls structural): random + shuffle return ORTHONORMAL subspaces of shape (d,k). ----
    for k in (1, 4, 8):
        Vr = random_subspace(low.d, k, RANDOM_NULL_SEEDS[0])
        Vs = shuffle_subspace(low.deleted, low.intact, k, SHUFFLE_NULL_SEEDS[0])
        assert Vr.shape == (low.d, k) and is_orthonormal(Vr), f"random subspace bad at k={k}"
        assert Vs.shape == (low.d, k) and is_orthonormal(Vs), f"shuffle subspace bad at k={k}"
    out["checks"]["nulls_orthonormal_shaped"] = "pass"

    # ---- Check (k=0 floor + grid monotonicity sanity): k=0 projector is identity (RF=0); the real
    #      k-sweep on lowrank is non-decreasing into a small-k saturation. ----
    assert np.allclose(project_out(low.deleted, estimate_disabling_subspace(Dlow, 0)), low.deleted), \
        "k=0 projector must be the identity (floor)"
    rf_curve = {k: low.rf_at(estimate_disabling_subspace(Dlow, k)) for k in K_GRID}
    assert abs(rf_curve[0]) < 1e-9, "RF(k=0) must be 0 (floor)"
    kstar = minimal_k(rf_curve)
    assert kstar is not None and kstar <= K_SMALL, f"lowrank minimal k* {kstar} not ≤ K_small={K_SMALL}"
    out["checks"]["k_sweep_lowrank"] = {"rf_by_k": {k: round(v, 3) for k, v in rf_curve.items()},
                                        "minimal_k_star": kstar}

    # ---- Check (RF bootstrap): item-level CI on a planted lowrank RF brackets the point. ----
    abl = low.ablate_items(V1_real)
    pt, lo, hi = bootstrap_rf(abl, low.floor_items(), low.intact_items(), n_boot=2000)
    assert lo <= pt <= hi, f"bootstrap CI [{lo:.3f},{hi:.3f}] does not bracket point {pt:.3f}"
    assert lo > TAU_RESTORE, f"lowrank RF CI-lower {lo:.3f} should clear τ={TAU_RESTORE}"
    out["checks"]["rf_bootstrap"] = {"point": round(pt, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}

    # ---- Check (audit S3/S1 helpers): energy_subspace orthonormal/shaped; monotonicity flag ----
    Ven = energy_subspace(low.deleted, 4)
    assert Ven.shape == (low.d, 4) and is_orthonormal(Ven), "energy_subspace not orthonormal/shaped"
    assert is_monotone_nondecreasing({0: 0.0, 1: 0.3, 2: 0.6, 4: 0.6}, 4) is True, "monotone false-neg"
    assert is_monotone_nondecreasing({0: 0.0, 1: 0.3, 2: 0.1, 4: 0.6}, 4) is False, "monotone false-pos"
    out["checks"]["energy_and_monotone_helpers"] = "pass"

    if verbose:
        import json
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(dry_run(), indent=2))
