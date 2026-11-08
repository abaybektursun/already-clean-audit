"""COMWRITE — the common-write object U_wr (comwrite spec §0/§3/§4).

U_wr(L,t;k) = the top-k eigenvectors of the per-FT-unit-Frobenius consensus Gram on the
RESIDUAL-FACING (d=2048) side of the ΔW family at matrix (L,t): the INPUT side for
q/k/v/gate/up (Gram = Σ ΔWᵀΔW/‖ΔW‖²_F — exactly invariant_core._consensus_gram's matrix),
the OUTPUT side for o/down (Gram = Σ ΔWΔWᵀ/‖ΔW‖²_F — the NEW o/down variant, §3). U_wr is
the top-k mirror of invariant_core's bottom-k U_unt.

Pure, weight-free, fp64, NO try/except (house style: gauge_cs.py, gauge_tree.py). Heavy imports
inside functions so the module py_compiles on the control plane. Gram terms are exposed
separately because the consensus Gram is ADDITIVE over FTs — the LOO discipline (§2: donor i's
statistics never use a basis its own ΔW entered) is sum-minus-one, never a recompute.
"""
from __future__ import annotations

from infra.pg.gauge_common import N_LAYERS, PROJ  # noqa: F401  (the 18×7 unit of analysis)

# The residual-facing (d=2048) side per matrix type (§0): the side whose basis lives in the
# decoder residual stream. q/k/v/gate/up READ the residual (input side); o/down WRITE it
# (output side).
RESIDUAL_SIDE = {"q_proj": "input", "k_proj": "input", "v_proj": "input",
                 "gate_proj": "input", "up_proj": "input",
                 "o_proj": "output", "down_proj": "output"}
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128)          # the A3 rank ladder (frozen, §4)


def residual_side(mtype):
    """'input' or 'output' — which side of a (out,in) matrix faces the residual stream."""
    assert mtype in RESIDUAL_SIDE, f"unknown matrix type {mtype!r} (∈ {tuple(RESIDUAL_SIDE)})"
    return RESIDUAL_SIDE[mtype]


def residual_dim(shape, mtype):
    """The residual-facing dimension of a matrix with `shape` (out,in) — in for input-side
    types, out for output-side types. The d the U_wr basis lives in."""
    out_d, in_d = shape
    return int(in_d) if residual_side(mtype) == "input" else int(out_d)


# ---------------------------------------------------------------------------
# consensus Gram on the residual-facing side (additive per-FT terms → LOO for free).
# ---------------------------------------------------------------------------
def gram_term(dW, mtype):
    """One FT's unit-trace Gram term on the residual-facing side, d×d fp64.
    input side:  ΔWᵀΔW/‖ΔW‖²_F   (invariant_core._consensus_gram's summand, verbatim)
    output side: ΔWΔWᵀ/‖ΔW‖²_F   (the NEW o/down variant)."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    assert A.ndim == 2, f"ΔW must be 2-D, got {A.ndim}-D"
    fro2 = float((A ** 2).sum())
    assert fro2 > 0, "zero ΔW — a frozen/degenerate ft slipped the FT-scale gate"
    if residual_side(mtype) == "input":
        return (A.T @ A) / fro2
    return (A @ A.T) / fro2


def consensus_gram(dW_list, mtype):
    """G = Σ_i gram_term(ΔW_i) — trace(G) = len(dW_list) by construction."""
    import numpy as np
    assert dW_list, "empty ΔW list"
    G = gram_term(dW_list[0], mtype)
    for dW in dW_list[1:]:
        T = gram_term(dW, mtype)
        assert T.shape == G.shape, f"Gram shape drift {T.shape} vs {G.shape}"
        G = G + T
    tr = float(np.trace(G))
    assert abs(tr - len(dW_list)) < 1e-6 * len(dW_list), f"trace {tr} != n={len(dW_list)}"
    return G


def gram_minus(G, term):
    """The LOO Gram: the full consensus Gram minus one FT's term (additivity). The ONLY
    sanctioned way to build donor i's basis (§2 LOO discipline)."""
    import numpy as np
    G = np.asarray(G, dtype=np.float64)
    T = np.asarray(term, dtype=np.float64)
    assert G.shape == T.shape, f"shape mismatch {G.shape} vs {T.shape}"
    return G - T


def u_wr(G, k):
    """The top-k eigenvectors of a consensus Gram, DESCENDING eigenvalue order — the U_wr
    basis (d,k), fp64 orthonormal columns. The mirror of invariant_core's bottom-k U_unt.
    scipy subset solver (top-k only, ~4× cheaper than a full eigh at k≪d)."""
    import numpy as np
    from scipy.linalg import eigh
    G = np.asarray(G, dtype=np.float64)
    d = G.shape[0]
    assert G.shape == (d, d), f"Gram must be square, got {G.shape}"
    assert 1 <= k <= d, f"k={k} outside [1, {d}]"
    _w, V = eigh(G, subset_by_index=[d - k, d - 1])     # ascending within the subset
    return V[:, ::-1].copy()


def eig_profile(G):
    """Descending eigenvalues of the consensus Gram (clipped at 0 — PSD up to fp roundoff)."""
    import numpy as np
    w = np.linalg.eigvalsh(np.asarray(G, dtype=np.float64))
    return np.clip(w, 0.0, None)[::-1].copy()


def topk_trace_share(G, k):
    """Top-k eigenvalue mass / trace ∈ (0,1] — the A1 concentration statistic (verdicted
    against the invariant_core Haar band computed by the job layer)."""
    import numpy as np
    w = eig_profile(G)
    tr = float(w.sum())
    assert tr > 0, "zero-trace Gram"
    return float(w[: int(k)].sum()) / tr


# ---------------------------------------------------------------------------
# energy capture (the A3 ladder statistic + the A5 write-fraction, reuse_estimators'
# ENERGY convention: rho = captured Frobenius energy / total).
# ---------------------------------------------------------------------------
def energy_fraction(dW, U, mtype):
    """‖P(ΔW)‖²_F / ‖ΔW‖²_F with P = UUᵀ applied on the residual-facing side ∈ [0,1].
    Chance floor for a Haar-random k-basis is k/d (energy convention — the SQUARE of
    endow_writefrac.write_fraction's amplitude ratio)."""
    import numpy as np
    A = np.asarray(dW, dtype=np.float64)
    Ub = np.asarray(U, dtype=np.float64)
    d = residual_dim(A.shape, mtype)
    assert Ub.ndim == 2 and Ub.shape[0] == d, f"U{Ub.shape} does not face d={d} of {mtype}"
    fro2 = float((A ** 2).sum())
    assert fro2 > 0, "zero ΔW — undefined energy fraction"
    if residual_side(mtype) == "input":
        cap = float(((A @ Ub) ** 2).sum())
    else:
        cap = float(((Ub.T @ A) ** 2).sum())
    return cap / fro2


def chance_floor(k, d):
    """E[energy_fraction] under a Haar-random k-subspace of dimension d (rotation invariance)."""
    assert 1 <= k <= d
    return float(k) / float(d)


def concentration_ladder(dW_list, U_full, k_grid=K_GRID):
    """{k: mean over ΔWs of energy_fraction into U_full[:, :k]} — the A3 ladder read. U_full
    is a single descending basis (d, k_max); its k-prefixes are the nested U_wr(k). The job
    feeds SEALED-half ΔWs against a DISCOVERY-half basis (out-of-sample concentration).
    `dW_list` entries are (dW, mtype) pairs so mixed types are impossible to feed silently."""
    import numpy as np
    assert dW_list, "empty ΔW list for the ladder"
    kmax = max(k_grid)
    assert U_full.shape[1] >= kmax, f"basis has {U_full.shape[1]} cols < k_max={kmax}"
    out = {}
    for k in k_grid:
        vals = [energy_fraction(dW, U_full[:, :k], mtype) for dW, mtype in dW_list]
        out[int(k)] = float(np.mean(vals))
    return out


def k_star(ladder, frac=0.9):
    """The frozen §5.1 selector: the SMALLEST k in the ladder whose sealed concentration
    reaches ≥ frac (0.9, frozen) of the k_max (=128) value. Weight-space only — never sees
    behavioral data (§5.4 garden-path threat)."""
    ks = sorted(int(k) for k in ladder)
    top = float(ladder[ks[-1]])
    assert top > 0, "degenerate ladder (zero concentration at k_max)"
    for k in ks:
        if float(ladder[k]) >= frac * top:
            return int(k)
    raise AssertionError("no k reaches the k_star fraction — ladder not monotone?")


# ---------------------------------------------------------------------------
# structure statistics (A1): mean-direction cosine + subspace overlaps.
# ---------------------------------------------------------------------------
def mean_direction(dW_list, mtype):
    """The residual-facing top singular direction of the family MEAN of unit-Frobenius ΔWs —
    the H_star-rank1 comparator (d,). If the family is a rank-1 star, U_wr's top eigvec
    aligns with this. Computed as the top eigenvector of the mean matrix's residual-facing
    Gram (identical to the SVD singular vector, at d² eigh cost instead of a full m×n SVD)."""
    import numpy as np
    assert dW_list, "empty ΔW list"
    M = None
    for dW in dW_list:
        A = np.asarray(dW, dtype=np.float64)
        fro = float(np.sqrt((A ** 2).sum()))
        assert fro > 0, "zero ΔW in mean direction"
        M = A / fro if M is None else M + A / fro
    M /= len(dW_list)
    G = M.T @ M if residual_side(mtype) == "input" else M @ M.T
    return u_wr(G, 1)[:, 0].copy()


def abs_cos(u, v):
    """|cos| between two vectors (sign-gauge-free — eigvec sign is arbitrary)."""
    import numpy as np
    a = np.asarray(u, dtype=np.float64).ravel()
    b = np.asarray(v, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    assert na > 0 and nb > 0, "zero vector in cosine"
    return float(abs(a @ b) / (na * nb))


def subspace_overlap(U1, U2):
    """‖U1ᵀU2‖²_F / k ∈ [0,1] for two orthonormal (d,k) bases — 1 = identical span, Haar
    floor = k/d. Delegates to gauge_rmt.subspace_overlap (the ONE in-repo /k-normalised
    implementation) after a same-shape assert (the A1 statistics compare equal-k bases)."""
    import numpy as np
    from infra.pg.gauge_rmt import subspace_overlap as _rmt_overlap
    A = np.asarray(U1)
    B = np.asarray(U2)
    assert A.shape == B.shape, f"basis shape mismatch {A.shape} vs {B.shape}"
    return _rmt_overlap(A, B)


# ---------------------------------------------------------------------------
# Arm C — one-sided gradient projection in the consensus basis (§6). Unlike
# relaw.project_grad (two-sided top-σ BLOCK), the COMWRITE channel is one-sided: only the
# residual-facing side is constrained; the other side is free.
# ---------------------------------------------------------------------------
def project_grad_residual(g, U, mtype, mode):
    """Project a weight-gradient g (out,in) onto/off span(U) on the residual-facing side:
      input side:  into = g·(UUᵀ),  output side: into = (UUᵀ)·g,  off = g − into.
    into and off are Frobenius-orthogonal ⇒ ‖into‖² + ‖off‖² = ‖g‖² (the §10.1 dry-run
    partition identity). The projection is computed in fp32 REGARDLESS of g's dtype and
    cast back: a bf16 projection leaks ~0.3% RMS in-span residue per step, which over
    10³ momentum-SGD steps rebuilds exactly the channel the 'off' cell must avoid (a
    per-step accumulating bias toward H_channel-not-necessary). Returned in g's
    dtype/device."""
    import torch
    g = torch.as_tensor(g)
    g32 = g.to(torch.float32)
    Ub = torch.as_tensor(U).to(dtype=torch.float32, device=g.device)
    d = residual_dim(tuple(g.shape), mtype)
    assert Ub.shape[0] == d, f"U{tuple(Ub.shape)} does not face d={d} of {mtype}"
    if residual_side(mtype) == "input":
        into = (g32 @ Ub) @ Ub.transpose(0, 1)
    else:
        into = Ub @ (Ub.transpose(0, 1) @ g32)
    if mode == "into":
        return into.to(g.dtype)
    if mode == "off":
        return (g32 - into).to(g.dtype)
    raise ValueError(f"project_grad_residual mode must be 'into'/'off', got {mode!r}")
