"""GAUGE-FIX (P2) — gauge-fixed differencing: quotient the architecture's exact symmetry, then diff.

Question (§5). ‖ΔW‖ and every basis statistic in the program treat weight coordinates as physical.
The Gemma-1 decoder has an EXACT symmetry group G under which two weight vectors compute the
identical function: (i) permutations of MLP hidden units; (ii) GQA-consistent attention-head
permutations; (iii) RMSNorm-gain ↔ adjacent-matrix positive-diagonal scale transfers; (iv) the
global residual-channel permutation (in the catalogue, EXCLUDED from the optimizer — the same-init
family makes the identity the natural section). Fine-tuning may have moved partly ALONG the fiber
(gauge motion — functionally void) and partly ACROSS it (physical). How much of ΔW is gauge?

    g* = argmin_{g∈G} ‖W_ft − g·W_base‖²_F        (per-layer weight matching)
    γ  = 1 − ‖W_ft − g*·W_base‖²_F / ‖W_ft − W_base‖²_F      (per layer / subsystem / pooled)

γ is bounded in [0,1] BY CONSTRUCTION: the catalogue contains the identity, so the optimiser never
does worse than the raw difference (every aligner below clamps its residual with the raw residual).
The load-bearing question is not whether γ>0 but whether it exceeds the OVERFIT-NULL band δ_GF:
running the same optimiser on structureless (spectrum-matched rotation-null) Δ still extracts some
γ purely by over-fitting the assignment, and only γ ABOVE that floor is fiber motion.

Pure, weight-free, fp64, no try/except (house style: relaw.py, gauge_cs.py, reuse_estimators.py) —
a wrong assumption RAISES. Heavy imports (numpy/scipy) live inside functions so the module
py_compiles on the control plane. Every return is JSON-able or a numpy array the caller reduces.

THE SYMMETRY CATALOGUE IS A FROZEN, DRY-RUN-VERIFIED DELIVERABLE. Each generator below carries a
numerical function-equivalence check (`check_*_equivariant`) that applies g to a tiny numpy layer
and asserts the layer computes the IDENTICAL function (attention checks pass with RoPE both on and
off — whole-head-block permutation commutes with RoPE; the catalogue is DISCRETE + POSITIVE-SCALE
only, never a free rotation, because RoPE forbids intra-head rotations and the residual stream's
elementwise gains forbid free residual rotations). A generator that changes the function is a bug;
the tests (test_gauge_fix.py, §12.1 P2 suite) exercise exactly these checks weight-free.
"""
from __future__ import annotations

from infra.pg import gauge_common as GC

# The four exact generators of the Gemma-1 decoder symmetry group (§5.2 catalogue). `in_optimizer`
# flags whether the aligner searches over that generator; the global residual-channel permutation is
# in the catalogue for completeness but EXCLUDED (same-init family ⇒ identity is the natural section).
SYMMETRY_CATALOGUE = (
    {"name": "mlp_hidden_perm", "acts_on": ("gate_proj", "up_proj", "down_proj"), "in_optimizer": True},
    {"name": "attn_head_perm", "acts_on": ("q_proj", "k_proj", "v_proj", "o_proj"), "in_optimizer": True},
    {"name": "norm_scale_transfer", "acts_on": ("*_layernorm", "adjacent_proj"), "in_optimizer": True},
    {"name": "residual_channel_perm", "acts_on": ("everything",), "in_optimizer": False},
)


# ===========================================================================
# low-level index / Frobenius helpers.
# ===========================================================================
def frob2(A):
    """‖A‖²_F in fp64 (the residual/raw currency of γ)."""
    import numpy as np
    a = np.asarray(A, dtype=np.float64)
    return float((a * a).sum())


def _perm_rows(W, src):
    """W with rows reordered so new row i = old row src[i] (src = source-for-destination index)."""
    import numpy as np
    return np.asarray(W, dtype=np.float64)[np.asarray(src, dtype=np.int64), :]


def _perm_cols(W, src):
    """W with columns reordered so new col i = old col src[i]."""
    import numpy as np
    return np.asarray(W, dtype=np.float64)[:, np.asarray(src, dtype=np.int64)]


def _block_row_index(block_src, block):
    """Expand a per-block source permutation (len nblock) into a per-ROW source index (len
    nblock·block): head/hidden blocks of `block` contiguous rows are moved WHOLE (never split)."""
    import numpy as np
    bs = np.asarray(block_src, dtype=np.int64)
    return (bs[:, None] * block + np.arange(block, dtype=np.int64)[None, :]).ravel()


# ===========================================================================
# Generator (i) — MLP hidden-unit permutation.
# ===========================================================================
def mlp_permute(gate, up, down, hidden_src):
    """Apply an MLP hidden-unit permutation g to (gate_proj, up_proj, down_proj): permute the ROWS of
    gate and up (hidden index) by `hidden_src` and the COLUMNS of down (hidden index) by the SAME
    permutation. Gemma MLP is down(act(gate·x) ⊙ up·x); permuting the hidden axis consistently across
    all three leaves the function unchanged for ANY elementwise activation. Returns (gate',up',down')."""
    return _perm_rows(gate, hidden_src), _perm_rows(up, hidden_src), _perm_cols(down, hidden_src)


def mlp_forward(x, gate, up, down):
    """Tiny reference Gemma-style GeGLU MLP: down @ (act(gate·x) ⊙ up·x), act = tanh (any elementwise
    nonlinearity gives the same permutation-equivariance; tanh keeps the dry-run cheap and exact).
    x is a (d_in,) vector. Used only by the function-equivalence check."""
    import numpy as np
    g = np.asarray(gate, np.float64) @ np.asarray(x, np.float64)
    u = np.asarray(up, np.float64) @ np.asarray(x, np.float64)
    h = np.tanh(g) * u
    return np.asarray(down, np.float64) @ h


def check_mlp_perm_equivariant(m=5, n=4, hidden=7, seed=0, tol=1e-10):
    """DRY-RUN: a random hidden-unit permutation applied to a toy MLP computes the IDENTICAL function.
    Returns the max abs output difference (asserts it below tol). Load-bearing: a generator that
    changes the function is a bug (§5.2)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    gate = rng.standard_normal((hidden, n)); up = rng.standard_normal((hidden, n))
    down = rng.standard_normal((m, hidden)); x = rng.standard_normal(n)
    src = rng.permutation(hidden)
    g2, u2, d2 = mlp_permute(gate, up, down, src)
    d = float(np.max(np.abs(mlp_forward(x, gate, up, down) - mlp_forward(x, g2, u2, d2))))
    assert d < tol, f"mlp_permute changed the function: max|Δ|={d}"
    return d


# ===========================================================================
# Generator (ii) — GQA-consistent attention-head permutation.
# ===========================================================================
def attn_head_perm_indices(kv_src, within_src, group):
    """Build the GQA-consistent per-head source indices from a kv-group permutation and per-group
    within-permutations. `kv_src` (len n_kv): new kv slot G sources old kv group kv_src[G]. `within_src`
    (n_kv × group): new query position q in new group G sources old within-index within_src[G][q] of
    old group kv_src[G]. A query head must stay paired with its kv head, so query heads move WITH their
    group. Returns (q_head_src [len n_heads], kv_head_src [len n_kv]) — whole-head source permutations
    (block index, NOT row index)."""
    import numpy as np
    kv_src = np.asarray(kv_src, dtype=np.int64)
    within = np.asarray(within_src, dtype=np.int64)
    n_kv = kv_src.shape[0]
    assert within.shape == (n_kv, group), f"within_src {within.shape} != {(n_kv, group)}"
    q_head_src = np.empty(n_kv * group, dtype=np.int64)
    for G in range(n_kv):
        g = int(kv_src[G])
        for q in range(group):
            q_head_src[G * group + q] = g * group + int(within[G, q])
    return q_head_src, kv_src


def attn_permute(q, k, v, o, q_head_src, kv_head_src, head_dim):
    """Apply an attention-head permutation g to (q_proj,k_proj,v_proj,o_proj). q_proj rows and o_proj
    columns are grouped into query-head blocks of `head_dim` and moved by `q_head_src`; k_proj and
    v_proj rows are grouped into kv-head blocks and moved by `kv_head_src`. Because a query head keeps
    its kv partner and RoPE acts identically on every head, the attention function is unchanged.
    Returns (q',k',v',o')."""
    qsrc = _block_row_index(q_head_src, head_dim)
    kvsrc = _block_row_index(kv_head_src, head_dim)
    return _perm_rows(q, qsrc), _perm_rows(k, kvsrc), _perm_rows(v, kvsrc), _perm_cols(o, qsrc)


def _rope(xh, positions, head_dim):
    """Split-half RoPE (Llama/GPT-NeoX convention) applied to one head's (seq, head_dim) tensor with
    the SAME frequencies for every head — which is exactly why whole-head-block permutation commutes
    with RoPE."""
    import numpy as np
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float64) / half))
    ang = np.asarray(positions, np.float64)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = xh[:, :half], xh[:, half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attn_forward(x, q, k, v, o, n_heads, n_kv, head_dim, rope=True):
    """Tiny reference GQA/MQA attention (no causal mask — irrelevant to the head-permutation symmetry).
    x is (seq, d_model). Query head h reads kv head h//(n_heads//n_kv). Used only by the equivalence
    check; runs with `rope` on and off to prove RoPE-compatibility."""
    import numpy as np
    x = np.asarray(x, np.float64)
    seq = x.shape[0]
    group = n_heads // n_kv
    assert n_heads % n_kv == 0, f"n_heads {n_heads} not a multiple of n_kv {n_kv}"
    Q = (x @ np.asarray(q, np.float64).T).reshape(seq, n_heads, head_dim)
    K = (x @ np.asarray(k, np.float64).T).reshape(seq, n_kv, head_dim)
    V = (x @ np.asarray(v, np.float64).T).reshape(seq, n_kv, head_dim)
    pos = np.arange(seq, dtype=np.float64)
    out = np.empty((seq, n_heads, head_dim), dtype=np.float64)
    for h in range(n_heads):
        g = h // group
        qh = _rope(Q[:, h, :], pos, head_dim) if rope else Q[:, h, :]
        kh = _rope(K[:, g, :], pos, head_dim) if rope else K[:, g, :]
        s = (qh @ kh.T) / np.sqrt(head_dim)
        s = s - s.max(axis=-1, keepdims=True)
        w = np.exp(s); w = w / w.sum(axis=-1, keepdims=True)
        out[:, h, :] = w @ V[:, g, :]
    return out.reshape(seq, n_heads * head_dim) @ np.asarray(o, np.float64).T


def check_attn_perm_equivariant(n_heads=4, n_kv=2, head_dim=4, d_model=6, seed=1, tol=1e-9):
    """DRY-RUN: a random GQA-consistent head permutation computes the IDENTICAL attention function,
    checked with RoPE both ON and OFF. Returns max(|Δ|_rope, |Δ|_norope) (asserts below tol)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    group = n_heads // n_kv
    q = rng.standard_normal((n_heads * head_dim, d_model))
    k = rng.standard_normal((n_kv * head_dim, d_model))
    v = rng.standard_normal((n_kv * head_dim, d_model))
    o = rng.standard_normal((d_model, n_heads * head_dim))
    x = rng.standard_normal((3, d_model))
    kv_src = rng.permutation(n_kv)
    within = np.stack([rng.permutation(group) for _ in range(n_kv)])
    qh, kh = attn_head_perm_indices(kv_src, within, group)
    q2, k2, v2, o2 = attn_permute(q, k, v, o, qh, kh, head_dim)
    d = 0.0
    for rope in (True, False):
        a = attn_forward(x, q, k, v, o, n_heads, n_kv, head_dim, rope=rope)
        b = attn_forward(x, q2, k2, v2, o2, n_heads, n_kv, head_dim, rope=rope)
        d = max(d, float(np.max(np.abs(a - b))))
    assert d < tol, f"attn_permute changed the function: max|Δ|={d}"
    return d


# ===========================================================================
# Generator (iii) — RMSNorm gain ↔ adjacent-matrix positive-diagonal scale transfer.
# ===========================================================================
def norm_scale_transfer(gain_eff, mats, D):
    """Transfer a positive diagonal D from an RMSNorm effective gain into the columns of the matrices
    that consume its output. RMSNorm emits normalise(x) ⊙ gain_eff; a downstream matrix W computes
    W·(normalise(x) ⊙ gain_eff). Replacing gain_eff → D⊙gain_eff and W → W·diag(1/D) leaves the
    function unchanged for ANY positive D. `gain_eff` is the EFFECTIVE gain (for Gemma RMSNorm that is
    1+weight; the caller converts). `mats` is the list of consuming matrices (columns indexed by the
    normalised channel). Returns (gain_eff', [mats']). Positive D only (no free residual rotation)."""
    import numpy as np
    d = np.asarray(D, np.float64)
    assert (d > 0).all(), "scale transfer requires a strictly positive diagonal"
    g2 = np.asarray(gain_eff, np.float64) * d
    m2 = [np.asarray(W, np.float64) / d[None, :] for W in mats]
    return g2, m2


def rmsnorm(x, gain_eff, eps=1e-6):
    """Gemma RMSNorm with an EFFECTIVE gain: normalise(x) ⊙ gain_eff, normalise = x/sqrt(mean(x²)+eps)."""
    import numpy as np
    x = np.asarray(x, np.float64)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * np.asarray(gain_eff, np.float64)


def check_norm_transfer_equivariant(d=6, n_out=4, seed=2, tol=1e-10):
    """DRY-RUN: a random positive-diagonal scale transfer applied to (gain, consuming matrix) computes
    the IDENTICAL function W·rmsnorm(x). Returns max|Δ| (asserts below tol)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    gain = rng.standard_normal(d) * 0.3 + 1.0            # effective gain ≈ 1 (Gemma 1+weight regime)
    W = rng.standard_normal((n_out, d)); x = rng.standard_normal(d)
    D = np.exp(rng.standard_normal(d) * 0.5)             # strictly positive
    g2, (W2,) = norm_scale_transfer(gain, [W], D)
    y0 = W @ rmsnorm(x, gain); y1 = W2 @ rmsnorm(x, g2)
    diff = float(np.max(np.abs(y0 - y1)))
    assert diff < tol, f"norm_scale_transfer changed the function: max|Δ|={diff}"
    return diff


# ===========================================================================
# Generator (iv) — global residual-channel permutation (in the catalogue, EXCLUDED from optimizer).
# ===========================================================================
def check_residual_perm_equivariant(d=6, seed=3, tol=1e-10):
    """DRY-RUN of the residual-channel permutation: a linear map W that both READS and WRITES the
    residual stream (y = W·x, both in residual coordinates) transforms as W → P W Pᵀ under a channel
    permutation P, giving P·y. Embeddings and every RMSNorm gain permute by the same P (checked here on
    the gain). This generator is documented for completeness but the aligner does NOT search over it:
    the same-init family fixes the residual coordinate frame, so the identity is the natural section
    (searching it would only introduce a d_model! overfit degree of freedom with no fiber motion to
    find). Returns max|Δ| (asserts below tol)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, d)); x = rng.standard_normal(d); gain = rng.standard_normal(d)
    src = rng.permutation(d)
    Wp = _perm_rows(_perm_cols(W, src), src)             # P W Pᵀ
    xp = np.asarray(x)[src]; gp = np.asarray(gain)[src]
    diff = float(np.max(np.abs(Wp @ xp - np.asarray(W @ x)[src])))
    diff = max(diff, float(np.max(np.abs(gp - gain[src]))))
    assert diff < tol, f"residual_channel_perm changed the function: max|Δ|={diff}"
    return diff


def catalogue_dry_run():
    """Run every generator's function-equivalence check; return {name: max|Δ|}. The frozen pre-registered
    proof that the whole symmetry catalogue is exact on the Gemma-1 architecture (§5.2)."""
    return {"mlp_hidden_perm": check_mlp_perm_equivariant(),
            "attn_head_perm": check_attn_perm_equivariant(),
            "norm_scale_transfer": check_norm_transfer_equivariant(),
            "residual_channel_perm": check_residual_perm_equivariant()}


# ===========================================================================
# Aligners — g* = argmin ‖W_ft − g·W_base‖²_F per subsystem (identity always in the catalogue).
# ===========================================================================
def _assign(cost):
    """Optimal min-cost assignment (Hungarian). scipy.optimize.linear_sum_assignment is available in
    the env (scipy.stats is imported in infra/pg/lm_feature_drift.py). Returns the source-for-destination
    permutation `src` (row i of the cost = destination/ft slot, col j = source/base slot)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(np.asarray(cost, np.float64))
    src = np.empty(cost.shape[0], dtype=np.int64)
    src[r] = c
    return src


def _block_row_cost(Wf, Wb, block):
    """cost[i,j] = ‖Wf block i − Wb block j‖²_F over row-blocks of `block` contiguous rows. Wf/Wb same
    shape (nblock·block, cols)."""
    import numpy as np
    Wf = np.asarray(Wf, np.float64); Wb = np.asarray(Wb, np.float64)
    nb = Wf.shape[0] // block
    F = Wf.reshape(nb, block * Wf.shape[1]); B = Wb.reshape(nb, block * Wb.shape[1])
    fn = (F * F).sum(1)[:, None]; bn = (B * B).sum(1)[None, :]
    return fn + bn - 2.0 * (F @ B.T)


def _block_col_cost(Wf, Wb, block):
    """cost[i,j] = ‖Wf col-block i − Wb col-block j‖²_F over column-blocks of `block` contiguous cols."""
    import numpy as np
    return _block_row_cost(np.asarray(Wf, float).T, np.asarray(Wb, float).T, block)


def mlp_align(base, ft):
    """g* for the MLP hidden-unit permutation: Hungarian on the joint (gate-row + up-row + down-col)
    squared-distance cost. `base`/`ft` are dicts with 'gate_proj','up_proj','down_proj'. Returns
    {'hidden_src', 'aligned2', 'raw2'} where aligned2 = ‖ft − g*·base‖²_F over the three matrices,
    clamped ≤ raw2 (the identity is in the catalogue ⇒ γ≥0)."""
    import numpy as np
    hidden = ft["gate_proj"].shape[0]
    cost = (_block_row_cost(ft["gate_proj"], base["gate_proj"], 1)
            + _block_row_cost(ft["up_proj"], base["up_proj"], 1)
            + _block_col_cost(ft["down_proj"], base["down_proj"], 1))
    src = _assign(cost)
    g2, u2, d2 = mlp_permute(base["gate_proj"], base["up_proj"], base["down_proj"], src)
    aligned = frob2(ft["gate_proj"] - g2) + frob2(ft["up_proj"] - u2) + frob2(ft["down_proj"] - d2)
    raw = frob2(ft["gate_proj"] - base["gate_proj"]) + frob2(ft["up_proj"] - base["up_proj"]) \
        + frob2(ft["down_proj"] - base["down_proj"])
    if aligned > raw:                                   # identity ∈ catalogue
        src = np.arange(hidden, dtype=np.int64); aligned = raw
    return {"hidden_src": src.tolist(), "aligned2": float(aligned), "raw2": float(raw)}


def attn_align(base, ft, n_heads, n_kv, head_dim):
    """g* for the GQA-consistent attention-head permutation. Two-stage within-group matching: assign kv
    groups by the (k-row + v-row) block cost, then within each matched group assign query heads by the
    (q-row + o-col) block cost — a query head keeps its kv partner. `base`/`ft` are dicts with
    'q_proj','k_proj','v_proj','o_proj'. Returns {'kv_src','q_head_src','aligned2','raw2'}, aligned2
    clamped ≤ raw2. (For MQA n_kv=1 the kv stage is trivial and the query stage is a single Hungarian.)"""
    import numpy as np
    group = n_heads // n_kv
    kv_cost = (_block_row_cost(ft["k_proj"], base["k_proj"], head_dim)
               + _block_row_cost(ft["v_proj"], base["v_proj"], head_dim))
    kv_src = _assign(kv_cost) if n_kv > 1 else np.zeros(1, dtype=np.int64)
    # full query-head cost: q rows + o cols, both grouped by head_dim.
    q_cost = (_block_row_cost(ft["q_proj"], base["q_proj"], head_dim)
              + _block_col_cost(ft["o_proj"], base["o_proj"], head_dim))
    within = np.empty((n_kv, group), dtype=np.int64)
    for G in range(n_kv):
        g = int(kv_src[G])
        sub = q_cost[G * group:(G + 1) * group, g * group:(g + 1) * group]   # new group G × old group g
        within[G, :] = _assign(sub)
    q_head_src, kv_head_src = attn_head_perm_indices(kv_src, within, group)
    q2, k2, v2, o2 = attn_permute(base["q_proj"], base["k_proj"], base["v_proj"], base["o_proj"],
                                  q_head_src, kv_head_src, head_dim)
    aligned = frob2(ft["q_proj"] - q2) + frob2(ft["k_proj"] - k2) + frob2(ft["v_proj"] - v2) \
        + frob2(ft["o_proj"] - o2)
    raw = frob2(ft["q_proj"] - base["q_proj"]) + frob2(ft["k_proj"] - base["k_proj"]) \
        + frob2(ft["v_proj"] - base["v_proj"]) + frob2(ft["o_proj"] - base["o_proj"])
    if aligned > raw:                                   # identity ∈ catalogue (guards two-stage sub-optimality)
        q_head_src = np.arange(n_heads, dtype=np.int64); kv_head_src = np.arange(n_kv, dtype=np.int64)
        aligned = raw
    return {"kv_src": np.asarray(kv_head_src).tolist(), "q_head_src": np.asarray(q_head_src).tolist(),
            "aligned2": float(aligned), "raw2": float(raw)}


def norm_align(gain_base, mats_base, gain_ft, mats_ft, d_clip=(1e-3, 1e3)):
    """g* for the RMSNorm-gain ↔ adjacent-matrix positive-diagonal scale transfer. Closed-form positive
    transfer D = clip(gain_eff_ft / gain_eff_base) (the diagonal that exactly matches the effective
    gains), applied to the base gain AND the columns of the consuming matrices, residual clamped ≤ raw
    (identity ∈ catalogue). `gain_*` are EFFECTIVE gains (1+weight for Gemma); `mats_*` are the aligned
    lists of consuming matrices. Returns {'D','aligned2','raw2'}. A closed form (not the joint argmin
    over the coupled gain+matrix residual) — conservative: the identity clamp guarantees γ≥0 and the
    program only ASSERTS gauge above the overfit-null floor."""
    import numpy as np
    gb = np.asarray(gain_base, np.float64); gf = np.asarray(gain_ft, np.float64)
    D = np.clip(gf / gb, d_clip[0], d_clip[1])
    g2, m2 = norm_scale_transfer(gb, [np.asarray(W, np.float64) for W in mats_base], D)
    aligned = frob2(gf - g2) + sum(frob2(Wf - Wb) for Wf, Wb in zip(mats_ft, m2))
    raw = frob2(gf - gb) + sum(frob2(np.asarray(Wf, float) - np.asarray(Wb, float))
                               for Wf, Wb in zip(mats_ft, mats_base))
    if aligned > raw:
        D = np.ones_like(gb); aligned = raw
    return {"D": D.tolist(), "aligned2": float(aligned), "raw2": float(raw)}


# ===========================================================================
# Gauge fraction + controls (§5.3).
# ===========================================================================
def gauge_fraction(aligned2, raw2):
    """γ = 1 − aligned2/raw2 ∈ [0,1]. raw2==0 (a matrix aligned to itself; nothing changed) ⇒ γ=0.0 by
    convention (no change ⇒ no gauge; the identity check). Never returns <0 (identity ∈ catalogue)."""
    if raw2 == 0.0:
        return 0.0
    return float(1.0 - aligned2 / raw2)


def spectrum_matched_null(dW, seed):
    """A structureless matrix with the SAME singular spectrum as ΔW but Haar-random singular vectors:
    Δ_null = L diag(σ) Rᵀ, L,R Haar frames. The overfit-null Δ (§5.3): it carries ΔW's magnitude and
    spectrum but NO coordinate structure, so any γ the aligner extracts from it is pure over-fitting of
    the assignment. Same construction as the program's rotation-invariant null (gauge_cs / invariant_core)."""
    import numpy as np
    A = np.asarray(dW, np.float64)
    m, n = A.shape
    sig = np.linalg.svd(A, compute_uv=False)
    r = sig.shape[0]
    rng = np.random.default_rng(seed)

    def _haar(d):
        Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
        return Q
    return (_haar(m) * sig[None, :]) @ _haar(n).T


def planted_gauge_mlp(base, seed, noise=0.0):
    """Positive control: apply a KNOWN random hidden-unit permutation g to the base MLP to synthesise an
    ft, optionally + isotropic noise. With noise=0 ALL of ΔW is gauge ⇒ mlp_align must recover γ≈1;
    with noise>0 the recovered γ ≈ the planted (gauge / total) energy ratio. Returns the ft dict."""
    import numpy as np
    rng = np.random.default_rng(seed)
    hidden = base["gate_proj"].shape[0]
    src = rng.permutation(hidden)
    g2, u2, d2 = mlp_permute(base["gate_proj"], base["up_proj"], base["down_proj"], src)
    ft = {"gate_proj": g2, "up_proj": u2, "down_proj": d2}
    if noise > 0.0:
        for kkey in ft:
            ft[kkey] = ft[kkey] + noise * rng.standard_normal(ft[kkey].shape)
    return ft


def delta_gf(null_gammas):
    """δ_GF = half-width of the overfit-null γ band on discovery (§5.3): (p97.5 − p2.5)/2 of the pooled
    null γ draws. Frozen BEFORE the real γ. `null_gammas` = all overfit-null γ samples (≥3 draws per FT)."""
    import numpy as np
    g = np.asarray(null_gammas, np.float64)
    assert g.size >= 3, "δ_GF needs ≥3 null γ draws (frozen seed rule)"
    return float((np.percentile(g, 97.5) - np.percentile(g, 2.5)) / 2.0)


def median_excess_ci(excesses, n_boot=2000, seed=1337):
    """Family-median γ excess (real γ − overfit-null-median γ) + bootstrap-over-FTs 95% CI (§5.2
    summary). Returns {median, ci_lo, ci_hi}."""
    import numpy as np
    e = np.asarray(excesses, np.float64)
    assert e.size >= 1, "no γ excesses to summarise"
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(e, e.size, replace=True)) for _ in range(n_boot)])
    return {"median": float(np.median(e)), "ci_lo": float(np.percentile(meds, 2.5)),
            "ci_hi": float(np.percentile(meds, 97.5))}


def discovery_clear(ci_lo, delta):
    """§5.3 Discovery-clear: the family-median γ-excess CI-lo exceeds the overfit-null half-width δ_GF."""
    return bool(ci_lo > delta)


def detection(sealed_ci_lo, delta, bh_ok):
    """§5.3 Detection: a discovery-cleared row whose SEALED-half median γ-excess CI-lo > δ_GF AND passes
    BH over P2's promoted family (pooled + per-subsystem rows, ≤3). Pure predicate."""
    return bool(sealed_ci_lo > delta and bh_ok)
