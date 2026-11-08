"""LATMECH — the latent-mechanism science core (pure, importable, weight-free).

No model, no datasets, no GPU: small-tensor numpy/torch math, laptop-unit-tested
(infra/pg/tests/test_latmech.py) BEFORE any GPU job spends a minute. The latmech jobs
(infra/aws_pg/jobs/latmech_*.py) are thin orchestration around these functions plus the
inherited relearn / mmlu-cloze / decoder-gradient spine (the §13 reuse ledger).

Prereg: studies/lineage/paligemma2/research/2026-06-26_latent_mechanism_prereg.md.

The genuinely-NEW math the MDE needs (everything else is REUSED):

  WEIGHT CONDITIONS (per decoder matrix, fp64):
    * carrier_direction_idx     — top-r directions of W by content-sensitivity score
                                  |u_dᵀ G v_d| (G = ∂cloze-CE/∂W). Reuses
                                  w15_carrier.{svd_fp64,direction_scores,rank_select}.
    * project_out_subspace      — W − Σ_{i∈idx} σ_i u_i v_iᵀ (null those singular
                                  components). Removes exactly Σ σ_i² Frobenius energy.
    * removed_energy            — Σ_{i∈idx} σ_i² (the matched-energy budget).
    * energy_matched_random_idx — a RANDOM set of W's singular directions whose Σ σ²
                                  matches a target energy (the MDE denominator: generic
                                  same-energy damage, not the content carrier).
    * shrink_perturb            — λ·W + σ·N(0, std(W)) (Ash & Adams 2020, the basin arm).
    * interpolate_weights       — (1−t)·W_deleted + t·W_floor (Arm-2 deletion spectrum).

  ESTIMATORS (recovery-fraction MDE; reuse reprice_stats for the paired bootstrap):
    * mde_contrast              — paired item bootstrap CI of RF(emrand) − RF(content)
                                  (= Δ_content − Δ_emrand; the `deleted` reference cancels).
    * select_carrier_k          — the §6 frozen rule: smallest k whose content-ablation
                                  drops cloze ≥ thresh while emrand at matched energy drops
                                  it < thresh (the energy-controlled validity gate).
    * mde_verdict               — apply the §6 decision rule to frozen CIs (the §9.6 unblind
                                  predicate; pure → testable on synthetic banked cells).
    * position_align_tripwire   — assert two per-item hash sequences are identical in order
                                  (REPRICE position-alignment reorder tripwire, T5).

Design rules (CLAUDE.md): no try/except hiding; a wrong assumption RAISES with a trace.
Heavy imports (torch/numpy) live inside functions so the module py_compiles on the
control plane (matches infra/pg/w15_carrier.py + w14_localize.py house style).
"""
from __future__ import annotations


# ===========================================================================
# Weight conditions — per-matrix fp64 transforms of one decoder weight W.
# ===========================================================================

def carrier_direction_idx(G, U, V, r):
    """The top-r directions of W (in its own SVD basis U,V) by content-sensitivity
    |u_dᵀ G v_d|, where G = ∂(cloze-CE)/∂W. = rank_select(direction_scores(G,U,V), r).
    These are the directions of W the content loss cares about most — the content carrier
    to project out. Returns a 1-D long index tensor (ascending), length min(r, n_dirs)."""
    from infra.pg import w15_carrier as C
    scores = C.direction_scores(G, U, V)
    return C.rank_select(scores, r)


def project_out_subspace(W, U, S, V, idx):
    """Null the selected singular components of W: W_ablated = W − Σ_{i∈idx} σ_i u_i v_iᵀ.
    (U,S,V) = w15_carrier.svd_fp64(W) (U:(m,r) S:(r,) V:(n,r)); `idx` a 1-D index set into
    those directions. Removes exactly Σ_{i∈idx} σ_i² Frobenius energy (the u_iv_iᵀ are
    orthonormal in the trace inner product). idx empty → W unchanged. fp64; job casts to
    the deploy dtype."""
    import torch
    W = torch.as_tensor(W, dtype=torch.float64)
    idx = torch.as_tensor(idx, dtype=torch.long)
    if idx.numel() == 0:
        return W.clone()
    U = torch.as_tensor(U, dtype=torch.float64)
    S = torch.as_tensor(S, dtype=torch.float64)
    V = torch.as_tensor(V, dtype=torch.float64)
    dWk = (U[:, idx] * S[idx]) @ V[:, idx].transpose(0, 1)
    return W - dWk


def removed_energy(S, idx):
    """Σ_{i∈idx} σ_i² — the Frobenius² energy project_out_subspace removes (= ‖W − W_abl‖²_F).
    The matched-energy budget the emrand null must hit. Returns a float."""
    import torch
    S = torch.as_tensor(S, dtype=torch.float64)
    idx = torch.as_tensor(idx, dtype=torch.long)
    if idx.numel() == 0:
        return 0.0
    return float((S[idx] ** 2).sum().item())


def energy_matched_random_idx(S, target_energy, seed, exclude=None):
    """A RANDOM subset of W's singular directions whose Σ σ² is as close as possible to
    `target_energy` — the MDE denominator (same removed activation-energy as the content
    ablation, but NOT the content directions, so the contrast isolates capability-specific
    content from generic competence damage). Greedy over a seeded permutation of the eligible
    directions; adds a direction iff it does not worsen |Σσ² − target|, stops once Σσ² ≥ target.
    `exclude` (idx tensor/list) keeps emrand disjoint from the content carrier. Returns
    (idx: ascending long tensor, achieved_energy: float). achieved is logged, never hidden."""
    import torch
    S = torch.as_tensor(S, dtype=torch.float64)
    n = S.numel()
    elig = torch.ones(n, dtype=torch.bool)
    if exclude is not None:
        ex = torch.as_tensor(exclude, dtype=torch.long)
        if ex.numel() > 0:
            elig[ex] = False
    eligible = torch.nonzero(elig, as_tuple=False).reshape(-1)
    g = torch.Generator().manual_seed(int(seed))
    order = eligible[torch.randperm(eligible.numel(), generator=g)]
    e2 = S ** 2
    chosen, acc = [], 0.0
    for i in order.tolist():
        ek = float(e2[i].item())
        if acc + ek <= target_energy or abs(acc + ek - target_energy) < abs(acc - target_energy):
            chosen.append(i)
            acc += ek
        if acc >= target_energy:
            break
    idx = torch.tensor(sorted(chosen), dtype=torch.long)
    return idx, acc


def shrink_perturb(W, lam, sigma, seed):
    """Ash & Adams (2020) shrink-and-perturb: W' = λ·W + σ·N(0, std(W)) — the optimization-basin
    manipulation (H-basin arm). λ<1 shrinks toward 0; σ>0 injects std-matched noise. The job tunes
    (λ, σ) so ‖W' − W‖_F matches the content-ablation removed magnitude (§5 matched-magnitude null).
    λ=1, σ=0 → W unchanged. Deterministic per seed. fp64."""
    import torch
    W = torch.as_tensor(W, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed))
    std = W.std()
    noise = torch.randn(W.shape, generator=g, dtype=torch.float64) * std
    return float(lam) * W + float(sigma) * noise


def interpolate_weights(W_deleted, W_floor, t):
    """Arm-2 deletion-spectrum point: W = (1−t)·W_deleted + t·W_floor (no training). t=0 → the
    deleted (content-preserved) decoder; t=1 → the random-floor decoder. The cheap dynamic-range
    sweep that holds task/modality/magnitude/recipe fixed (the SAV confound dissolver). fp64."""
    import torch
    Wd = torch.as_tensor(W_deleted, dtype=torch.float64)
    Wf = torch.as_tensor(W_floor, dtype=torch.float64)
    return (1.0 - float(t)) * Wd + float(t) * Wf


def frob(A):
    """Frobenius norm of A (fp64)."""
    import torch
    return float(torch.as_tensor(A, dtype=torch.float64).norm().item())


def delta_frob(A, B):
    """‖A − B‖_F (fp64) — the magnitude-match readout for the shrink-perturb null."""
    import torch
    A = torch.as_tensor(A, dtype=torch.float64)
    B = torch.as_tensor(B, dtype=torch.float64)
    return float((A - B).norm().item())


# ===========================================================================
# Estimators — the recovery-fraction MDE (reuse reprice_stats for the bootstrap).
# ===========================================================================

def rf_diff_contrast(plus_S, minus_S, floor, ceiling, n_boot: int = 10000, seed: int = 1337):
    """Paired item cluster bootstrap of RF(plus) − RF(minus), where RF_x = (mean(S_x) − mean(floor))
    / (mean(ceiling) − mean(floor)) (reprice_stats.recovery_fraction). All four are per-item arrays
    paired BY POSITION; one resample indexes every array (items = clusters). A resample with
    ceiling≈floor (denom ≤ 0) is dropped as nan. The reusable Δ-RF engine behind the MDE and the
    shrink-perturb / competence-matched CIs. Returns the bootstrap dict {point, lo, hi, n_valid, …}."""
    import numpy as np
    from infra.pg import reprice_stats as RS
    pS = np.asarray(plus_S, float)
    mS = np.asarray(minus_S, float)
    F = np.asarray(floor, float)
    C = np.asarray(ceiling, float)
    assert pS.shape == mS.shape == F.shape == C.shape and pS.ndim == 1, \
        f"RF-diff arrays must be paired per item (1-D, equal length): {pS.shape}/{mS.shape}/{F.shape}/{C.shape}"

    def stat(arrs):
        ps, ms, f, c = arrs
        denom = c.mean() - f.mean()
        if denom <= 0:
            return np.nan
        return (ps.mean() - f.mean()) / denom - (ms.mean() - f.mean()) / denom

    return RS.paired_cluster_bootstrap([pS, mS, F, C], stat, n=n_boot, seed=seed)


def mde_contrast(content_S, emrand_S, floor, ceiling, n_boot: int = 10000, seed: int = 1337):
    """The MDE primary statistic: Δ_content − Δ_emrand on the recovery fraction, where
    Δ_x = RF(deleted) − RF(x-ablated). The `deleted` reference cancels algebraically, so the
    statistic is RF(emrand) − RF(content). H-store (content privileged ⇒ content-ablation hurts
    re-entry MORE) ⇒ CI entirely > 0. Thin wrapper over rf_diff_contrast(emrand, content, …)."""
    return rf_diff_contrast(emrand_S, content_S, floor, ceiling, n_boot=n_boot, seed=seed)


def select_carrier_k(k_grid, cloze_drop_content, cloze_drop_emrand, thresh):
    """The §6 carrier-k rule (frozen on discovery): the SMALLEST k in k_grid whose
    content-ablation drops cloze by ≥ `thresh` AND whose energy-matched-random ablation drops
    it by < `thresh` — the energy-controlled validity gate (a content carrier must beat
    same-energy generic damage on the cloze axis it is defined by). The three arrays are aligned
    over k_grid. Returns the chosen k (int), or None if no k satisfies the gate ⇒ the caller
    HALTs with INSTRUMENT-NULL (do not interpret the MDE)."""
    for k, dc, de in zip(k_grid, cloze_drop_content, cloze_drop_emrand):
        if dc >= thresh and de < thresh:
            return int(k)
    return None


def mde_verdict(mde_ci, sp_ci, comp_matched_ci, seed_signs, mde_fdr_ok: bool = True):
    """Apply the §6 MDE decision rule to FROZEN CIs (all computed inside the frozen instrument;
    BH-FDR already applied upstream → `mde_fdr_ok`). The §9.6 unblind predicate, kept pure so
    the end-to-end verdict is auditable + testable on synthetic banked cells.

      mde_ci          : {lo, hi} of Δ_content − Δ_emrand  (= RF(emrand) − RF(content))
      sp_ci           : {lo, hi} of C_sp − C_deleted      (shrink-perturb basin move)
      comp_matched_ci : {lo, hi} of (competence-matched-absent − deleted) re-entry RF
      seed_signs      : list of per-seed signs of the MDE point (sign-stability gate)

    Returns one of 'H_store' / 'H_basin' / 'H_data' / 'H_mixed'. No rule swap post-unblind;
    a mixed/over-determined outcome returns 'H_mixed' (the dominant term is named by the caller)."""
    seed_stable = bool(seed_signs) and len({s > 0 for s in seed_signs}) == 1
    mde_pos = mde_ci["lo"] > 0 and seed_stable and mde_fdr_ok
    mde_brackets0 = mde_ci["lo"] <= 0 <= mde_ci["hi"]
    sp_moves = not (sp_ci["lo"] <= 0 <= sp_ci["hi"])
    comp_within = comp_matched_ci["lo"] <= 0 <= comp_matched_ci["hi"]
    if mde_pos:
        return "H_store"
    if sp_moves and mde_brackets0:
        return "H_basin"
    if comp_within and mde_brackets0:
        return "H_data"
    return "H_mixed"


# ===========================================================================
# Pairing tripwire (REPRICE position-alignment; qa_item_hash is NOT a join key).
# ===========================================================================

def position_align_tripwire(hash_seq_a, hash_seq_b):
    """REPRICE position-alignment tripwire (§8 T5): items are paired by CANONICAL POSITION
    (sorted-subject row order); the per-item hash sequences must be identical IN ORDER. The hash
    is a reorder tripwire ONLY — qa_item_hash has a verified 895-way collision so it is never a
    join key. RAISES on any length/order mismatch (fail-loud); returns True on match."""
    a = list(hash_seq_a)
    b = list(hash_seq_b)
    if a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        raise AssertionError(
            f"position-alignment tripwire: hash sequences differ "
            f"(len {len(a)} vs {len(b)}; first diff at index {first})")
    return True
