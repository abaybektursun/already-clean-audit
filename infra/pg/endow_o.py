"""ENDOW-O core — rotation-aware transport + Site-R KV-branch contract (endow spec §3).

Pure small-tensor torch/numpy math with NO model / datasets / GPU dependency, weight-free
unit-tested (infra/pg/tests/test_endow_o.py) before any GPU minute — the §9.1 dry-run gate.
Reuses the FROZEN INHERIT/CTRLIFACE instrument (apply_mean_ablation, orthonormalize,
principal_angle_overlap, the bank loader) and adds ONLY what ENDOW-O needs beyond INHERIT:

  * ORTHOGONAL PROCRUSTES transport R(L) — the rotation-aware map that upgrades INHERIT's
    flat-at-L14 identity read into a three-way genesis verdict (aligned / rotated /
    constructed). Same Schoenemann (1966) SVD core as decoder_alignment.procrustes_aligned_cosine
    — but that returns an aligned-COSINE metric; ENDOW-O needs the MAP R itself, to rotate the
    frozen π0.5 V bank into PG1 coordinates (V' = Rᵀ·V) and ablate span(V') in PG1.
  * PROCRUSTES GUARD datum — ‖R−I‖_F (how much fine-tuning rotated the geometry, by depth) and
    the transport of an em-random span (which must stay null; the behavioral guard runs at eval,
    §3.4 — a rotation that manufactures bite on a random span voids the rotated branch).
  * SITE-R KV-branch projector (Arm O-C) — the SAME tested projector applied only to the k/v the
    action expert reads, leaving the backbone's own residual intact. The pure projector is here +
    unit-tested; wiring it into lerobot's shared-KV branch is a Gate-0 FEASIBILITY item (frozen
    skip rule §3.2) — the monkeypatch must expose that branch, verified remotely, never faked here.

Design rules (CLAUDE.md): NO try/except hiding — a wrong assumption RAISES with a trace.
Heavy imports (torch/numpy) live inside functions so the module py_compiles on the control plane.
"""
from __future__ import annotations

from infra.pg.ctrliface_core import apply_mean_ablation, orthonormalize, principal_angle_overlap


# ===========================================================================
# Orthogonal Procrustes transport (Schoenemann 1966; the same SVD core as
# decoder_alignment.procrustes_aligned_cosine — here returning the MAP, not a metric).
# ===========================================================================
def orthogonal_procrustes(src, dst, center=True):
    """Fit the orthogonal R minimising ‖src·R − dst‖_F over paired rows (row convention:
    src (n,d), dst (n,d) are the SAME n items — alignment key enforced by the caller). With
    center=True both streams are mean-centred first (the transport is of directions, not the
    mean offset). Returns R (d,d) fp64 orthogonal. R = U Vᵀ where U S Vᵀ = svd(srcᵀ·dst)."""
    import torch
    A = torch.as_tensor(src, dtype=torch.float64)
    B = torch.as_tensor(dst, dtype=torch.float64)
    assert A.shape == B.shape and A.ndim == 2, f"paired (n,d) rows required: {A.shape} vs {B.shape}"
    # AᵀB is d×d for ANY n, so R=U Vᵀ is orthogonal even when n<d (underdetermined only on the
    # data's null space — an arbitrary-but-orthogonal completion; the O-D guard catches spurious bite).
    if center:
        A = A - A.mean(0)
        B = B - B.mean(0)
    U, _S, Vt = torch.linalg.svd(A.transpose(0, 1) @ B, full_matrices=False)
    return (U @ Vt).contiguous()


def transport_basis(R, V):
    """Map a src-coordinate column basis V (d,kA orthonormal) into dst coordinates: V' = Rᵀ·V,
    re-orthonormalized (R orthogonal already preserves it; QR is the numerical guard). This is
    the frame in which span(V') ablated from a dst-residual removes the SAME variance that
    span(V) removes from the paired src-residual (dst = src·R ⇒ dst·V' = src·V ⇔ V' = Rᵀ·V)."""
    import torch
    R = torch.as_tensor(R, dtype=torch.float64)
    V = torch.as_tensor(V, dtype=torch.float64)
    assert R.shape[0] == R.shape[1] == V.shape[0], f"shape mismatch R{tuple(R.shape)} V{tuple(V.shape)}"
    return orthonormalize(R.transpose(0, 1) @ V)


def frob_rel_identity(R):
    """‖R − I‖_F / √d ∈ [0, 2/√d·…] — the descriptive genesis datum: how far the transport
    rotates the geometry (0 = identity/aligned; grows as fine-tuning re-coordinatizes)."""
    import torch
    R = torch.as_tensor(R, dtype=torch.float64)
    d = R.shape[0]
    return float((R - torch.eye(d, dtype=torch.float64)).norm() / (d ** 0.5))


def is_orthogonal(R, atol=1e-8):
    """RᵀR ≈ I within atol (the fit's own sanity gate)."""
    import torch
    R = torch.as_tensor(R, dtype=torch.float64)
    return float((R.transpose(0, 1) @ R - torch.eye(R.shape[0], dtype=torch.float64)).abs().max()) < atol


# ===========================================================================
# Site-R (Arm O-C) — the SAME projector, applied only to the reader-visible k/v branch.
# The pure projector is unit-tested here; the lerobot wiring is a Gate-0 feasibility item.
# ===========================================================================
SITE_R_FEASIBILITY = (
    "Arm O-C requires the pi05 forward to expose the SHARED k/v the action expert attends to as a "
    "hookable branch inside compute_layer_complete (distinct from the backbone residual out[0]). "
    "This is asserted at Gate-0 on the live lerobot module; if not exposable → SKIP + disclose "
    "(frozen §3.2 rule). The origin verdict does not depend on O-C."
)


def kv_branch_ablate(kv, Q, hbar):
    """Mean-ablate span(Q)→h̄ on the reader-visible k/v tensor ONLY (same projector contract as
    apply_mean_ablation: out' = kv − QQᵀ(kv − h̄)). `kv` is the shared key/value the expert reads;
    the backbone's own residual stream is left untouched (that is what makes O-C a READ-path, not a
    stream-site, ablation — the ablation-site memo folded in). fp32-stable, every position."""
    return apply_mean_ablation(kv, Q, hbar)


def transported_overlap(R, V):
    """Diagnostic: overlap(span(V), span(Rᵀ·V)) — how much the transport MOVES the span (1 = R
    leaves span(V) invariant, i.e. identity transport is already the right frame at this layer)."""
    ov, _ = principal_angle_overlap(orthonormalize(V), transport_basis(R, V))
    return ov
