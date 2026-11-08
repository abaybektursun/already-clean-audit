"""TESSERA — pure instrument core (numpy/torch only, no model, no S3).

Everything here is unit-testable weight-free on tiny tensors, which is what the §9 step-2 offline
gate runs before any compute is spent. Orchestration lives in `infra/aws_pg/jobs/tessera_*.py`.

Objects (spec §0): every ΔW is **raw-ΔW**. Never compared to an install-gradient cotangent object
(the f_perp two-objects rule).

The subspace rank is r = 64 everywhere (amendment A6), inherited from
`specter_core.LANCZOS_TOPK` rather than minted here, so no TESSERA number depends on a rank this
program chose after seeing data.

No try/except (CLAUDE.md): a shape or rank violation must fail loud.
"""
from __future__ import annotations

import numpy as np

# Amendment A6 — the one subspace rank, inherited from specter_core.LANCZOS_TOPK.
R_SUBSPACE = 64
# The census ladder the footprint/overlap is ALSO reported over. Never searched for a verdict:
# the registered test reads R_SUBSPACE only.
R_LADDER = (8, 16, 32, 64)


# ===========================================================================
# Subspaces of a raw-ΔW
# ===========================================================================

def top_right_subspace(dW, r=R_SUBSPACE):
    """The r leading RIGHT-singular vectors of ΔW (n×r, orthonormal columns), fp64.

    Right-singular vectors live in the INPUT space — the same space `invariant_core`'s consensus
    Gram is built in, so an overlap computed here and a core computed there are statements about
    the same ℝⁿ. Using left vectors instead would silently compare two different spaces.

    Computed through the SMALLER Gram (the `lacuna_a2_bytes._top_left` trick, applied to the right
    side): the LM's `down_proj` is 2048×16384, so forming the 16384² Gram would cost 500× the
    2048² one. When the input dim is the larger side we eigendecompose A Aᵀ for U and recover
    V = Aᵀ U / σ, which is exact, not an approximation.
    """
    A = np.asarray(dW, dtype=np.float64)
    assert A.ndim == 2, f"top_right_subspace expects a matrix, got shape {A.shape}"
    out_d, in_d = A.shape
    rr = int(min(r, min(A.shape)))
    assert rr > 0, f"rank {r} is empty for shape {A.shape}"
    if in_d <= out_d:
        w, V = np.linalg.eigh(A.T @ A)              # (in, in)
        idx = np.argsort(w)[::-1][:rr]
        return np.ascontiguousarray(V[:, idx])
    w, U = np.linalg.eigh(A @ A.T)                  # (out, out)
    idx = np.argsort(w)[::-1][:rr]
    s = np.sqrt(np.clip(w[idx], 1e-300, None))
    V = (A.T @ U[:, idx]) / s[None, :]              # (in, rr), orthonormal by construction
    return np.ascontiguousarray(V)


def spectrum_normsq(dW):
    """p_i = σ_i²/Σσ_i² of one ΔW (sums to 1), fp64 — the matched-rotation null's only input.

    Also via the smaller Gram: eigenvalues of the Gram ARE the squared singular values, so the
    normalised squared spectrum needs no SVD at all.
    """
    A = np.asarray(dW, dtype=np.float64)
    G = (A.T @ A) if A.shape[1] <= A.shape[0] else (A @ A.T)
    s2 = np.clip(np.linalg.eigvalsh(G), 0.0, None)[::-1]
    tot = s2.sum()
    assert tot > 0, "zero ΔW — a frozen/degenerate unit reached the spectral census"
    return s2 / tot


# ===========================================================================
# Principal angles between two subspaces + the Haar null they must beat (N2)
# ===========================================================================

def principal_cosines(U, V):
    """cos of the principal angles between span(U) and span(V), descending.

    Both must have orthonormal columns. Length = min(#cols) — the number of angles that exist.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    assert U.shape[0] == V.shape[0], f"ambient dim mismatch {U.shape[0]} vs {V.shape[0]}"
    for name, M in (("U", U), ("V", V)):
        g = M.T @ M
        assert np.allclose(g, np.eye(M.shape[1]), atol=1e-8), f"{name} columns are not orthonormal"
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def subspace_overlap(U, V):
    """Mean squared principal cosine — the scalar the overlap test reads.

    = (1/r)‖UᵀV‖²_F, which is the expected squared length of a random unit vector of span(U)
    projected into span(V). Bounded in [0, 1]: 0 = orthogonal, 1 = identical subspace. Scale-free
    in r, so overlaps at different ranks are comparable.
    """
    c = principal_cosines(U, V)
    return float((c ** 2).mean())


def haar_overlap_null(n, r_u, r_v, n_draws=2000, seed=1337):
    """Null band for `subspace_overlap` between two INDEPENDENT Haar-random subspaces of ℝⁿ.

    This is the achievable null for "these two hops write in the same directions": two random
    r-dim subspaces of a 2048-dim space already overlap by ≈ r_v/n on average, so an overlap must
    be read against THIS, never against zero.

    Analytic mean r_v/n is asserted against the draws — a null whose own mean is wrong would make
    every overlap verdict wrong in the same direction, and that is exactly the failure this
    program has shipped twice.
    """
    rng = np.random.default_rng(seed)
    vals = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        Qu, _ = np.linalg.qr(rng.standard_normal((n, r_u)))
        Qv, _ = np.linalg.qr(rng.standard_normal((n, r_v)))
        vals[i] = float(((Qu.T @ Qv) ** 2).sum() / r_u)
    analytic = r_v / n
    assert abs(vals.mean() - analytic) < 10 * vals.std() / np.sqrt(n_draws) + 1e-12, (
        f"Haar overlap null mean {vals.mean():.6g} disagrees with the analytic {analytic:.6g} — "
        f"the null itself is miscomputed")
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "analytic_mean": float(analytic),
            "p2.5": float(np.percentile(vals, 2.5)), "p50": float(np.percentile(vals, 50.0)),
            "p97.5": float(np.percentile(vals, 97.5)),
            "p99.5": float(np.percentile(vals, 99.5)), "n_draws": int(n_draws)}


def overlap_verdict(observed, null, z_margin=3.0):
    """H_hop-disjoint is killed by an overlap ABOVE the Haar band; the band is two-sided.

    Both tails are named so the statistic cannot only-succeed: an overlap BELOW the band is
    reported as `anti-aligned`, which is just as much a structure claim as `shared-channel` and
    would be missed by a one-sided rule.
    """
    z = (observed - null["mean"]) / max(null["std"], 1e-300)
    if observed > null["p97.5"] and z >= z_margin:
        return {"verdict": "SHARED-CHANNEL", "z": float(z),
                "why": "overlap above the matched Haar band — the two writes occupy a common "
                       "subspace beyond what independent random subspaces of this size produce"}
    if observed < null["p2.5"] and z <= -z_margin:
        return {"verdict": "ANTI-ALIGNED", "z": float(z),
                "why": "overlap BELOW the Haar band — the writes avoid each other more than "
                       "independent random subspaces do"}
    return {"verdict": "HAAR-DISJOINT", "z": float(z),
            "why": "overlap inside the matched Haar band — no shared-channel structure beyond "
                   "what two independent random subspaces of this rank already produce"}


# ===========================================================================
# K1/K2 — the curvature footprint
# ===========================================================================

def footprint(dW, U_curv):
    """Fraction of a unit's write ENERGY that lands inside a faculty's curvature subspace.

    footprint(u, f) = ‖ΔW P_f‖²_F / ‖ΔW‖²_F, where P_f = U_f U_fᵀ projects onto the top-r
    curvature directions of faculty f, measured at PG1.

    The projector acts on the INPUT side (right-multiplication), matching `top_right_subspace`
    and the consensus Gram — all three are statements about the same ℝⁿ.

    Range [0, 1]. The matched-random reference is r/n (a write with no curvature preference), so
    a footprint is only interpretable against that, never against zero.
    """
    A = np.asarray(dW, dtype=np.float64)
    U = np.asarray(U_curv, dtype=np.float64)
    assert A.shape[1] == U.shape[0], (
        f"footprint: ΔW input dim {A.shape[1]} ≠ curvature subspace ambient dim {U.shape[0]}")
    tot = float((A ** 2).sum())
    assert tot > 0, "zero ΔW reached the footprint — a degenerate unit slipped the scale gate"
    proj = A @ U                      # (m, r): the write expressed in the curvature basis
    return float((proj ** 2).sum() / tot)


def footprint_null(n, r):
    """The energy fraction a curvature-indifferent write puts in an r-dim subspace of ℝⁿ."""
    return float(r) / float(n)


# ===========================================================================
# The consensus-core roster extension (Arm S, H_core-survives)
# ===========================================================================

def consensus_gram(dW_list):
    """G = Σ_i ΔW_iᵀΔW_i/‖ΔW_i‖²_F — `invariant_core._consensus_gram`'s matrix, in numpy.

    Kept here so the offline gate can exercise the roster logic weight-free; the JOB imports
    invariant_core's own function so the shipped number comes from the REUSED instrument, never
    from this copy. `test_consensus_gram_matches_invariant_core` pins them together.
    """
    n = np.asarray(dW_list[0]).shape[1]
    G = np.zeros((n, n), dtype=np.float64)
    for dW in dW_list:
        A = np.asarray(dW, dtype=np.float64)
        fro2 = (A ** 2).sum()
        assert fro2 > 0, "zero ΔW — a frozen/degenerate unit slipped the scale gate"
        G += (A.T @ A) / fro2
    return G


def bottom_k_mean(G, k):
    """Mean of the k smallest eigenvalues of a PSD Gram — the invariant-core score."""
    w = np.clip(np.linalg.eigvalsh(np.asarray(G, dtype=np.float64)), 0.0, None)
    kk = int(min(k, w.size))
    return float(w[:kk].mean())


def core_verdict(real, null_mean, null_std, null_p2_5, z_margin=3.0):
    """`invariant_core`'s rule verbatim: BEYOND_NULL needs the percentile cut AND |z| ≥ 3.

    The upper tail is named too. A consensus core that sits ABOVE the null band means the units
    write MORE uniformly into every direction than matched-random rotations do — the opposite
    structure, and a rule that could only report "core" would silently discard it.
    """
    z = (real - null_mean) / max(null_std, 1e-300)
    if real < null_p2_5 and z <= -z_margin:
        return {"verdict": "CORE-BEYOND-NULL", "z": float(z),
                "why": "the roster's jointly-untouched subspace is more coherent than matched "
                       "rotations produce"}
    if z >= z_margin:
        return {"verdict": "ABOVE-NULL", "z": float(z),
                "why": "the roster writes MORE uniformly than matched rotations — no untouched "
                       "subspace, and less structure than chance rather than more"}
    return {"verdict": "INSIDE-NULL", "z": float(z),
            "why": "no untouched subspace beyond the matched-rotation null"}


# ===========================================================================
# Arm X — crosscoder fractions against the identity-pair band (P2)
# ===========================================================================

def fraction_verdict(observed, identity_band, floor_key="p97.5"):
    """Label a crosscoder fraction against the (PG1, PG1) identity-pair null band.

    The identity pair is the only calibration that can fail: it is the SAME model on both sides,
    so any "emergent" fraction it reports is manufactured by the instrument. A real emergent
    fraction must clear that band; a survival fraction must not fall below its lower edge.
    """
    lo, hi = float(identity_band["p2.5"]), float(identity_band[floor_key])
    if observed > hi:
        return {"verdict": "ABOVE-IDENTITY-BAND", "why": "beyond what the identity pair "
                                                         "manufactures"}
    if observed < lo:
        return {"verdict": "BELOW-IDENTITY-BAND", "why": "below the identity pair's own floor"}
    return {"verdict": "INSIDE-IDENTITY-BAND", "why": "indistinguishable from what the "
                                                      "instrument manufactures on two copies of "
                                                      "the same model"}


# ===========================================================================
# Bootstrap CI used by every summary (house convention)
# ===========================================================================

def boot_ci(values, n_boot=10000, seed=1337, stat=np.median):
    """Percentile bootstrap CI of `stat` over `values`."""
    v = np.asarray(values, dtype=np.float64)
    assert v.size > 0, "bootstrap over an empty sample"
    rng = np.random.default_rng(seed)
    draws = np.array([stat(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    return {"point": float(stat(v)), "lo95": float(np.percentile(draws, 2.5)),
            "hi95": float(np.percentile(draws, 97.5)), "n": int(v.size), "n_boot": int(n_boot)}
