"""W15 carrier-gate — the pure, importable science core (no model, no datasets, no
GPU). Everything here is small-tensor numpy/torch math so it is unit-testable on a
laptop CPU (infra/pg/tests/test_w15_carrier.py) BEFORE any GPU job spends a single
minute. The W15 jobs (infra/aws_pg/jobs/w15_*.py) are thin orchestration around these
functions + the inherited graft / SVD / MMLU spine.

The prereg: studies/lineage/paligemma2/research/2026-06-04_w15_carrier_gate_prereg.md.
Two arms over ONE shared instrument (a function-weighted importance of weight-change
directions), answered training-free on cached weights:

  INSTALL arm (§3, §6) — operates in ΔW's OWN fp64 SVD basis:
    * direction_scores       — the shared instrument: project a task weight-gradient G
                               into the SVD basis and read each direction's first-order
                               sensitivity, diag(Uᵀ G V). leverage = scores at W_before;
                               oracle  = σ · scores at the FULL graft (realized marginal
                               contribution). energy = the σ ordering itself.
    * rank_select / energy_indices — pick the top-r directions by a score (energy = [0:r]).
    * graft_from_indices     — W_before + Σ_{i∈idx} σ_i u_i v_iᵀ. The selection graft;
                               generalises e4_capability._rankk (which is idx=[0:r]).
    * random_rotation_graft  — the matched-norm random-rotation null (W14-Arm-1 inspan):
                               a rank-r graft whose kept subspace is a random rotation
                               inside ΔW's top-R_eff span, scaled to the energy-r ‖·‖_F.

  RETENTION arm (§2, §6) — operates in W_BEFORE's fp64 SVD basis:
    * to_basis               — Uᵀ G V (the coupling-edge matrix). M = to_basis(ΔW).
    * edge_saliency          — |M ⊙ (Uᵀ G_task V)|, per-edge movement saliency I_task.
    * energy_topk_count      — k = #largest-|M| edges capturing a frac of ‖M‖²_F (frozen
                               headline frac=0.50; robustness 0.25/0.75).
    * topk_mask / overlap    — top-k membership and O = |A∩B|/k.
    * permuted_overlap       — the function-permuted null (value-shuffle one map) → the
                               analytic k/N expectation; the dry-run gate.
    * pooled_overlap         — O pooled across the 182 matrices (Σ|A∩B| / Σk), the unit
                               the cluster-bootstrap resamples.
    * grad_out_of_span_frac  — f_perp: install-gradient energy outside W_before's top-eff
                               span (W12 in-span recon 0/300: large f_perp ⇒ the in-span
                               edge map cannot see the install carrier → a finding, §6).
    * plant_disjoint / plant_identical — the §5 positive controls (O≈0 / O≈1).

Design rules (CLAUDE.md): no try/except hiding; a wrong assumption RAISES with a trace.
Heavy imports (torch/numpy) live inside functions so the module py_compiles on the
control plane (matches infra/pg/text_eval.py + w14_localize.py house style).

decoder_keys()/N_LAYERS are re-exported from w14_localize so W15 can never drift from
the 182-matrix key list (Gemma-2, 26L — the §0 generation firewall).
"""
from __future__ import annotations

from infra.pg.w14_localize import (          # the 182-key list + R_eff — single source of truth
    N_LAYERS,
    decoder_keys,
    module_keys,
    effective_rank_energy,
)

# Frozen headline edge fraction (§6): k = #largest-|M| edges capturing 50% of ‖M‖²_F per
# matrix. 25%/75% are the pre-registered robustness rungs (F5).
ENERGY_FRAC = 0.50
ENERGY_FRAC_ROBUST = (0.25, 0.50, 0.75)
RECON_TOL = 1e-10                            # M = Uᵀ ΔW V round-trip residual (fp64 mandate)


# ===========================================================================
# fp64 SVD helpers (the spectral-roundoff trap is closed only in float64).
# ===========================================================================

def svd_fp64(W):
    """Thin fp64 SVD of a 2-D matrix. Returns (U, s, V) with U:(m,r), s:(r,), V:(n,r),
    r=min(m,n); W ≈ U diag(s) Vᵀ. Mirrors gain_routing._svd_fp64 (same fp64 mandate);
    re-implemented here so the science core has no jobs-layer dependency."""
    import torch
    W = torch.as_tensor(W, dtype=torch.float64)
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    return U, s, Vh.transpose(0, 1)


def dw_svd(dW):
    """fp64 SVD of a weight-change matrix ΔW = W_after − W_before, the INSTALL basis.
    Returns (U, S, V) (full thin SVD). Jobs may instead pass the cached svd_lowrank
    factors from e4_capability._build_edit_family; the selection/graft fns below accept
    (U, S, V) directly so the basis is computed once."""
    return svd_fp64(dW)


# ===========================================================================
# INSTALL arm — the shared instrument + selection grafts (ΔW's own SVD basis).
# ===========================================================================

def direction_scores(G, U, V):
    """The SHARED INSTRUMENT (prereg §3, build once). Project a task weight-gradient
    `G` (∂loss/∂W, shape (out,in)) into ΔW's SVD basis and read each direction's
    first-order sensitivity: score_d = u_dᵀ G v_d = diag(Uᵀ G V). This is the realised
    change in `loss` per unit coefficient on SVD-direction d, i.e. how much moving along
    (u_d, v_d) matters to the decision. Returns a 1-D fp64 tensor of length r=U.shape[1].

    leverage  = direction_scores(G_refcoco_at_base,  U, V)            (job, §9.4)
    oracle    = S * direction_scores(G_refcoco_at_full_graft, U, V)   (realized marginal
                contribution under the full graft — the privileged-info ceiling, §2/§8-T5)
    The job ranks |score| descending; energy needs no gradient (it IS the σ ordering)."""
    import torch
    G = torch.as_tensor(G, dtype=torch.float64)
    U = torch.as_tensor(U, dtype=torch.float64)
    V = torch.as_tensor(V, dtype=torch.float64)
    # diag(Uᵀ G V) without forming the full r×r product: (Uᵀ G) is (r,in); times V col-wise.
    UtG = U.transpose(0, 1) @ G                       # (r, in)
    return (UtG * V.transpose(0, 1)).sum(dim=1)       # (r,) = Σ_n (Uᵀ G)[d,n] V[n,d]


def energy_indices(r, n_dirs):
    """The energy ordering: top-r directions by singular value. Since SVD returns σ
    descending, this is simply [0, 1, …, r-1] (clamped to n_dirs). Returned as a long
    tensor so it composes with rank_select/graft_from_indices."""
    import torch
    r = int(min(r, n_dirs))
    return torch.arange(r, dtype=torch.long)


def rank_select(scores, r):
    """Top-r direction indices by |score| descending (ties broken by lower index for
    determinism). `scores` is the 1-D output of direction_scores (or σ·that for oracle).
    Returns a long tensor of length min(r, len(scores))."""
    import torch
    scores = torch.as_tensor(scores, dtype=torch.float64)
    r = int(min(r, scores.numel()))
    # stable: sort by (−|score|, index). argsort is stable in torch for stable=True.
    order = torch.argsort(scores.abs(), descending=True, stable=True)
    return order[:r].sort().values                     # ascending idx for reproducible grafts


def graft_from_indices(W_before, U, S, V, idx):
    """The SELECTION GRAFT: W = W_before + Σ_{i∈idx} σ_i u_i v_iᵀ (no retrain). Generalises
    e4_capability._rankk — for idx = energy_indices(r) this is byte-equivalent to the
    energy rank-r graft. fp64 throughout; the job casts the result to the deploy dtype.
    `idx` is a 1-D index tensor/list into ΔW's SVD directions."""
    import torch
    W_before = torch.as_tensor(W_before, dtype=torch.float64)
    dev = W_before.device                              # unify every factor onto W_before's device
    U = torch.as_tensor(U, dtype=torch.float64).to(dev)
    S = torch.as_tensor(S, dtype=torch.float64).to(dev)
    V = torch.as_tensor(V, dtype=torch.float64).to(dev)
    idx = torch.as_tensor(idx, dtype=torch.long).to(dev)
    dWk = (U[:, idx] * S[idx]) @ V[:, idx].transpose(0, 1)
    return W_before + dWk


def selection_norm(S, idx):
    """Frobenius norm of the selection graft Σ_{i∈idx} σ_i u_i v_iᵀ = sqrt(Σ_{i∈idx} σ_i²)
    (u_i v_iᵀ are orthonormal in the trace inner product). Used to assert the leverage/
    oracle grafts are NOT trivially larger-norm than energy (a magnitude confound)."""
    import torch
    S = torch.as_tensor(S, dtype=torch.float64)
    idx = torch.as_tensor(idx, dtype=torch.long)
    return float((S[idx] ** 2).sum().sqrt().item())


def random_rotation_graft(W_before, dW, r, gen, energy: float = 0.99):
    """The matched-norm random-rotation null (W14-Arm-1 'inspan'): a rank-r graft whose
    kept subspace is a random orthonormal rotation INSIDE ΔW's top-R_eff span, scaled so
    ‖graft‖_F equals the energy rank-r graft's ‖·‖_F. A real ordering must beat THIS, not
    zero (§5/§6). Delegates the construction + matched-norm to w14_localize.subspace_AB so
    the null is bit-identical to W14's. Returns W_before + A@B (fp64)."""
    import torch
    from infra.pg.w14_localize import subspace_AB
    W_before = torch.as_tensor(W_before, dtype=torch.float64)
    A, B = subspace_AB(dW, rank=r, kind="inspan", gen=gen, energy=energy)
    return W_before + A @ B


def base_after_rank_corr(scores_base, scores_after):
    """Spearman rank-correlation between the base-evaluated and full-graft-evaluated
    direction scores (§5 shuffle/instrument-validity control for INSTALL). The FLOOR emits
    no <loc>, so the base leverage map may be degenerate; if this correlation is low the
    base map is non-resolving and the oracle (after-based) ceiling is the verdict (§8-T2).
    Returns a float in [−1, 1]."""
    import torch
    a = torch.as_tensor(scores_base, dtype=torch.float64).abs()
    b = torch.as_tensor(scores_after, dtype=torch.float64).abs()
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-30)
    return float((ra @ rb / denom).item())


# ===========================================================================
# RETENTION arm — coupling-edge saliency + overlap (W_before's SVD basis).
# ===========================================================================

def to_basis(G, U, V):
    """Project a matrix G into W_before's SVD basis: Uᵀ G V (the full r×r coupling-edge
    matrix). M = to_basis(ΔW, U, V) is the gain/routing matrix; I_task uses to_basis on a
    task gradient. fp64 mandate — any fp32 intermediate leaks ~1e-7 and breaks the
    reconstruction gate."""
    import torch
    G = torch.as_tensor(G, dtype=torch.float64)
    U = torch.as_tensor(U, dtype=torch.float64)
    V = torch.as_tensor(V, dtype=torch.float64)
    return U.transpose(0, 1) @ G @ V


def recon_residual(dW, U, V):
    """fp64 round-trip gate (§9.1): ‖U (Uᵀ ΔW V) Vᵀ − P_U ΔW P_V‖_F / ‖ΔW‖_F, the in-span
    reconstruction identity. Must be < RECON_TOL on the real basis. O(n³) projector — run
    in preflight, NOT per-matrix in the unblind loop (mirrors gain_routing check_recon)."""
    import torch
    dW = torch.as_tensor(dW, dtype=torch.float64)
    U = torch.as_tensor(U, dtype=torch.float64)
    V = torch.as_tensor(V, dtype=torch.float64)
    M = U.transpose(0, 1) @ dW @ V
    recon = U @ M @ V.transpose(0, 1)
    par = (U @ U.transpose(0, 1)) @ dW @ (V @ V.transpose(0, 1))
    denom = dW.norm().clamp_min(1e-30)
    return float(((recon - par).norm() / denom).item())


def edge_saliency(M, Gb):
    """Per-edge movement saliency I(edge) = |M_edge · Gb_edge|, where M = Uᵀ ΔW V (how far
    direction-pair (i,j) actually MOVED) and Gb = Uᵀ G_task V (how much the task LOSS cares
    about (i,j)). The product is the function-WEIGHTED movement — energy ≠ function (§1).
    Both args are r×r in the same basis. Returns an r×r fp64 saliency map."""
    import torch
    M = torch.as_tensor(M, dtype=torch.float64)
    Gb = torch.as_tensor(Gb, dtype=torch.float64)
    return (M * Gb).abs()


def energy_topk_count(M, frac: float = ENERGY_FRAC):
    """k = the number of largest-|M| edges whose squared sum first reaches `frac` of the
    total ‖M‖²_F (the pre-committed top-edge fraction, §6). Per matrix — wide MLP and tall
    attention matrices get different k. Returns an int ≥ 1."""
    import torch
    M = torch.as_tensor(M, dtype=torch.float64)
    e = (M.reshape(-1) ** 2)
    e_sorted, _ = torch.sort(e, descending=True)
    total = e_sorted.sum().clamp_min(1e-300)
    cum = torch.cumsum(e_sorted, 0) / total
    k = int(torch.searchsorted(cum, torch.tensor(float(frac), dtype=torch.float64)).item()) + 1
    return max(1, min(k, e.numel()))


def topk_mask(I, k):
    """Boolean mask of the top-k edges of saliency map I (flattened argtop-k, deterministic
    tie-break by index). Returns a flat bool tensor of length I.numel() with exactly k Trues."""
    import torch
    I = torch.as_tensor(I, dtype=torch.float64).reshape(-1)
    k = int(min(k, I.numel()))
    idx = torch.topk(I, k, largest=True, sorted=False).indices
    mask = torch.zeros(I.numel(), dtype=torch.bool)
    mask[idx] = True
    return mask


def overlap(mask_a, mask_b):
    """O = |top-k(a) ∩ top-k(b)| / k. Both masks must have the SAME number of Trues (paired
    top-k). Returns a float in [0, 1]."""
    import torch
    a = torch.as_tensor(mask_a, dtype=torch.bool).reshape(-1)
    b = torch.as_tensor(mask_b, dtype=torch.bool).reshape(-1)
    ka, kb = int(a.sum().item()), int(b.sum().item())
    assert ka == kb and ka > 0, f"paired top-k requires equal non-empty k; got {ka} vs {kb}"
    return float((a & b).sum().item()) / ka


def overlap_maps(I_a, I_b, k):
    """Convenience: overlap of the top-k of two saliency maps at a shared k."""
    return overlap(topk_mask(I_a, k), topk_mask(I_b, k))


def permuted_overlap(I_a, I_b, k, n_draws: int, seed: int):
    """The function-permuted null (§5/§6): hold top-k(a) fixed, randomly PERMUTE the edge
    values of map b, and recompute O. Destroys b's alignment with a while preserving its
    top-k count, so the null centres on the analytic chance overlap k/N (N = #edges). This
    is the dry-run-gateable value-shuffle null + analytic anchor; the retention JOB also
    builds the stronger item→gradient-shuffle null (B≥1000) from cached per-item gradients.
    Returns (samples: 1-D fp64 tensor of length n_draws, analytic_kN: float)."""
    import torch
    I_a = torch.as_tensor(I_a, dtype=torch.float64).reshape(-1)
    I_b = torch.as_tensor(I_b, dtype=torch.float64).reshape(-1)
    n = I_a.numel()
    k = int(min(k, n))
    mask_a = topk_mask(I_a, k)
    g = torch.Generator().manual_seed(int(seed))
    samples = torch.empty(n_draws, dtype=torch.float64)
    for d in range(n_draws):
        perm = torch.randperm(n, generator=g)
        mb = topk_mask(I_b[perm], k)
        samples[d] = float((mask_a & mb).sum().item()) / k
    return samples, float(k) / float(n)


def pooled_overlap(per_matrix):
    """Pool O across matrices: O = Σ_m |A_m ∩ B_m| / Σ_m k_m (per-matrix top-k membership,
    cluster = matrix). `per_matrix` is a list of (mask_a, mask_b) pairs; degenerate
    matrices are excluded by the caller (logged count). Returns (O_pooled, total_k, n_mat)."""
    import torch
    inter, ksum = 0, 0
    for ma, mb in per_matrix:
        a = torch.as_tensor(ma, dtype=torch.bool).reshape(-1)
        b = torch.as_tensor(mb, dtype=torch.bool).reshape(-1)
        ka, kb = int(a.sum().item()), int(b.sum().item())
        assert ka == kb and ka > 0, f"paired top-k mismatch in pool: {ka} vs {kb}"
        inter += int((a & b).sum().item())
        ksum += ka
    assert ksum > 0, "no edges pooled (all matrices excluded?)"
    return inter / ksum, ksum, len(per_matrix)


def pooled_analytic_null(km_list, Nm_list):
    """The analytic pooled chance overlap (prereg §5 'the analytic k/N_edges expected overlap'),
    pooled across matrices: E[O] = Σ_m (k_m²/N_m) / Σ_m k_m. Under independent top-k membership the
    per-matrix intersection has mean k_m·(k_m/N_m). Returns a float."""
    km = [float(k) for k in km_list]
    Nm = [float(n) for n in Nm_list]
    num = sum(k * k / n for k, n in zip(km, Nm) if n > 0)
    den = sum(km)
    assert den > 0, "no edges pooled"
    return num / den


def pooled_permuted_null(km_list, Nm_list, n_draws: int, seed: int):
    """The pooled function-permuted null, sampled EXACTLY (prereg §5/§9.5, B≥1000). Value-permuting
    one saliency map and re-taking top-k makes the per-matrix intersection Hypergeometric(N_m, k_m,
    k_m); the pooled null O = Σ_m inter_m / Σ_m k_m. Sampling hypergeometrics is O(B·M) — it scales to
    the ~10⁹ edges of 182 full matrices where an explicit torch.randperm (permuted_overlap) would not.
    Returns (samples: np.ndarray length n_draws). Equivalent to permuted_overlap; the latter is the
    small-tensor dry-run gate, this is the at-scale job null."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    km = np.asarray(km_list, dtype=np.int64)
    Nm = np.asarray(Nm_list, dtype=np.int64)
    ksum = int(km.sum())
    assert ksum > 0, "no edges pooled"
    acc = np.zeros(n_draws, dtype=np.float64)
    for k, n in zip(km, Nm):
        # hypergeometric(ngood=k, nbad=N-k, nsample=k): |top-k ∩ permuted-top-k|
        acc += rng.hypergeometric(int(k), int(n - k), int(k), size=n_draws)
    return acc / ksum


def grad_out_of_span_frac(G, U, V, R_eff):
    """f_perp of a task gradient (§6 / §8-T3): the fraction of G's energy OUTSIDE
    W_before's top-R_eff left/right span = 1 − ‖(Uᵀ G V)[:R_eff, :R_eff]‖²_F / ‖G‖²_F.
    Large f_perp (install gradient lives out-of-span) means the in-span edge map cannot see
    the install carrier — report it, do not silently pass. Returns a float in [0, 1]."""
    import torch
    G = torch.as_tensor(G, dtype=torch.float64)
    U = torch.as_tensor(U, dtype=torch.float64)
    V = torch.as_tensor(V, dtype=torch.float64)
    R = int(min(R_eff, U.shape[1], V.shape[1]))
    Gb = U.transpose(0, 1) @ G @ V
    in_span = (Gb[:R, :R] ** 2).sum()
    total = (G ** 2).sum().clamp_min(1e-300)
    return float((1.0 - in_span / total).item())


# ===========================================================================
# §5 positive controls — planted edge-saliency maps with known overlap.
# ===========================================================================

def plant_identical(shape, k, gen):
    """Two saliency maps whose top-k sets are IDENTICAL by construction (→ O = 1, tol 1e-6).
    Validates the estimator's upper rail. Returns (I_a, I_b, k) with both maps r×r ≥ 0."""
    import torch
    r0, r1 = shape
    n = r0 * r1
    k = int(min(k, n))
    base = torch.rand(n, generator=gen, dtype=torch.float64)
    # force a clean top-k by lifting k random positions above the rest in BOTH maps.
    pos = torch.randperm(n, generator=gen)[:k]
    a = base.clone(); a[pos] += 10.0
    b = (torch.rand(n, generator=gen, dtype=torch.float64)); b[pos] += 10.0
    return a.reshape(r0, r1), b.reshape(r0, r1), k


def plant_disjoint(shape, k, gen):
    """Two saliency maps whose top-k sets are DISJOINT by construction (→ O = 0, tol 1e-6).
    Validates the estimator's lower rail. Requires 2k ≤ N. Returns (I_a, I_b, k)."""
    import torch
    r0, r1 = shape
    n = r0 * r1
    k = int(min(k, n // 2))
    perm = torch.randperm(n, generator=gen)
    pos_a, pos_b = perm[:k], perm[k:2 * k]
    a = torch.rand(n, generator=gen, dtype=torch.float64); a[pos_a] += 10.0
    b = torch.rand(n, generator=gen, dtype=torch.float64); b[pos_b] += 10.0
    return a.reshape(r0, r1), b.reshape(r0, r1), k


# ===========================================================================
# Decision rules (frozen at §9.3; thresholds filled from recomputed anchors at §9.0).
# These are PURE predicate fns so the unblind step (§9.6) is auditable + testable.
# ===========================================================================

def install_verdict(r_oracle, r_leverage, r_energy, r_wholeloss, null_hi,
                    delta_install, seeds_ok: bool, fdr_ok: bool):
    """Apply the §6 INSTALL rule. `null_hi` = the random-rotation null upper band (e.g. p95)
    on R_cap; `delta_install` = δ frozen on the recomputed energy@r* + null at §9.0.
    Returns one of 'I-ORDERABLE' / 'I-RANK-TAIL' / 'I-MURKY' (KILL = I-RANK-TAIL)."""
    if r_oracle <= null_hi + delta_install:
        return "I-RANK-TAIL"                                       # even the ceiling is dead → KILL
    orderable = ((r_leverage - r_energy) > delta_install
                 and r_leverage > null_hi
                 and r_leverage > r_wholeloss
                 and seeds_ok and fdr_ok)
    return "I-ORDERABLE" if orderable else "I-MURKY"


def retention_verdict(O, null_p5, null_p95, o_forget_mag_p5, o_install_mag_p5,
                     seeds_ok: bool, fdr_ok: bool):
    """Apply the §6 RETENTION rule. Returns 'R-INSTRUMENT-NULL' (maps are magnitude →
    redesign), else 'R-SEPARABLE' / 'R-COINCIDENT' / 'R-ANTI'."""
    if not (O < o_forget_mag_p5 and O < o_install_mag_p5):
        return "R-INSTRUMENT-NULL"                                 # O not below |M| controls
    if O > null_p95:
        return "R-ANTI"
    if O < null_p5 and seeds_ok and fdr_ok:
        return "R-SEPARABLE"
    return "R-COINCIDENT"
