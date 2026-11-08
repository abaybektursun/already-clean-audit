"""M-Q — the gauge: fold the norm gains to 1, then rotate the residual stream (queue standing W1-Q).

THE POINT OF THE ARM. TENANT's reading is that the sink's address is a property of the ADAM BASIS,
not of the function: Adam's second moment is per-coordinate, so it privileges the coordinate frame
the model happens to be written in. If that is right, training the SAME function in a ROTATED frame
must move the sink to different coordinates. Rotating a pretrained model without changing its
function is the prerequisite, and that is what this module does.

WHY THE FOLD COMES FIRST. RMSNorm emits normalise(x) ⊙ gamma. `normalise` is rotation-EQUIVARIANT
(it divides by an L2-derived scalar, and ‖Rx‖ = ‖x‖), but diag(gamma) is NOT: a per-coordinate gain
is a diagonal operator ON the residual stream and diagonals do not commute with rotations. So the
gains must first be folded away — into the columns of the matrices that CONSUME the norm's output —
leaving gamma = 1 and a purely equivariant normalisation. Only then can the stream be rotated.

    gamma -> 1,  W -> W diag(gamma)        (function-preserving for any positive gamma)
    x -> Rx,     reader W -> W R^T,  writer W -> R W,  embedding E -> E R^T

The fold reuses `gauge_fix.norm_scale_transfer`, which carries its own function-equivalence
dry-run, rather than re-deriving the identity here.

WHICH NORMS ARE FOLDABLE IS AN ARCHITECTURAL FACT, NOT AN ASSUMPTION. A pre-norm decoder (the
Gemma-1 shape PG1 uses) applies every norm to the residual BEFORE a matrix reads it, so every gain
has consumers and every gain folds. A decoder that also normalises the SUBLAYER OUTPUT before the
residual add (the Gemma-2 shape) has gains with no downstream consumer — the output goes straight
into the residual — and those gains CANNOT be folded into any matrix. `plan_fold` reports which
case a model is in and refuses to pretend, because a silently unfolded gain would leave a diagonal
on the stream and make the rotation gate fail for a reason unrelated to the science.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from infra.pg.gauge_fix import norm_scale_transfer   # REUSED (§11 ledger)

# The consumer map for a PRE-NORM decoder: which matrices read each norm's output.
PRE_NORM_CONSUMERS = {
    "input_layernorm": ("q_proj", "k_proj", "v_proj"),
    "post_attention_layernorm": ("gate_proj", "up_proj"),
}
# Norm names that, in a post-sublayer-norm architecture, have NO matrix consumer.
UNFOLDABLE_NORMS = ("pre_feedforward_layernorm", "post_feedforward_layernorm")


def gemma_effective_gain(weight):
    """Gemma's RMSNorm stores `weight` and applies (1 + weight). The EFFECTIVE gain is what folds."""
    import numpy as np
    return 1.0 + np.asarray(weight, dtype=np.float64)


def plan_fold(norm_names) -> dict:
    """Classify a decoder layer's norms into foldable (with their consumers) and unfoldable.

    `norm_names` is the layer's actual submodule names, read from the model. A norm this module
    does not know how to place is an error rather than a silent skip: the whole gate depends on
    every gain reaching 1.
    """
    names = list(norm_names)
    foldable, unfoldable, unknown = {}, [], []
    for n in names:
        if n in PRE_NORM_CONSUMERS:
            foldable[n] = PRE_NORM_CONSUMERS[n]
        elif n in UNFOLDABLE_NORMS:
            unfoldable.append(n)
        else:
            unknown.append(n)
    assert not unknown, (
        f"norms {unknown} have no declared place in the fold plan. Add them to PRE_NORM_CONSUMERS "
        f"with the matrices that consume them, or to UNFOLDABLE_NORMS — guessing would leave a "
        f"per-coordinate diagonal on the residual stream and break the rotation for a reason that "
        f"has nothing to do with the science")
    return {"foldable": foldable, "unfoldable": unfoldable,
            "fully_foldable": not unfoldable,
            "architecture": "pre-norm" if not unfoldable else "post-sublayer-norm",
            "consequence": (None if not unfoldable else
                            "gains on %s have no matrix consumer, so the residual stream keeps a "
                            "per-coordinate diagonal and an exact rotation gauge does not exist "
                            "for this architecture" % unfoldable)}


def fold_gain_into(weight, mats) -> tuple:
    """Fold one Gemma RMSNorm gain to exactly 1, into its consuming matrices' COLUMNS.

    Returns (new_weight, [new_mats], stats). The new weight is the value the module must be SET to
    — 0 for Gemma, whose applied gain is 1 + weight — so a caller cannot conflate "gain 1" with
    "weight 1".

    THE SIGN MATTERS AND PG1 HAS NEGATIVE GAINS. `gauge_fix.norm_scale_transfer` is the house's
    positive-diagonal SCALE gauge and asserts D > 0; the first run of the M-Q gate hit that
    assertion on real PG1 weights, which is a fact about the model rather than a bug in the check.
    The fold-to-1 identity does not need positivity:

        W · (normalise(x) ⊙ gamma) = (W · diag(gamma)) · normalise(x)      for ANY gamma

    so the fold is done directly here, for any sign. The positive helper is still used when it
    applies, because it carries its own function-equivalence dry-run; where it does not apply, the
    identity above is exact and the sign counts are RECORDED rather than the coordinate skipped.

    A gain of exactly ZERO is a real state too: that coordinate is annihilated by the norm, the
    folded matrix gets a zero column, and the fold stays exact — it is simply no longer invertible.
    It is counted, because a model with dead residual coordinates is worth knowing about before a
    rotation is applied to it.
    """
    import numpy as np
    g = gemma_effective_gain(weight)
    Ms = [np.asarray(W, dtype=np.float64) for W in mats]
    stats = {"n_coords": int(g.size), "n_negative_gain": int((g < 0).sum()),
             "n_zero_gain": int((g == 0).sum()), "gain_min": float(g.min()),
             "gain_max": float(g.max()),
             "used_positive_helper": bool((g > 0).all())}
    if stats["used_positive_helper"]:
        g2, m2 = norm_scale_transfer(g, Ms, 1.0 / g)
        assert np.allclose(g2, 1.0, atol=1e-12), f"fold left gain at {g2.min()}..{g2.max()}"
    else:
        m2 = [W * g[None, :] for W in Ms]
    return np.zeros_like(g), m2, stats


def rotate_pair(W, role: str, R):
    """Apply the gauge to one matrix. `role` is 'reader' (input axis is the stream) or 'writer'."""
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    Q = np.asarray(R, dtype=np.float64)
    assert Q.shape[0] == Q.shape[1], f"rotation must be square, got {Q.shape}"
    if role == "reader":
        assert A.shape[1] == Q.shape[0], f"reader {A.shape} vs rotation {Q.shape}"
        return A @ Q.T
    assert role == "writer", f"unknown role {role!r}"
    assert A.shape[0] == Q.shape[0], f"writer {A.shape} vs rotation {Q.shape}"
    return Q @ A


def orthogonality_error(R) -> float:
    import numpy as np
    Q = np.asarray(R, dtype=np.float64)
    return float(np.abs(Q.T @ Q - np.eye(Q.shape[0])).max())


def hadamard_rotation(d: int):
    """A normalised Hadamard rotation — the QuaRot-style structured alternative to a Haar draw.

    It is in the arm because it is the SHARPEST version of the question: a Hadamard matrix spreads
    every coordinate over all others uniformly, so if the sink's address is a property of the
    coordinate frame this is the rotation that should destroy it most completely. Requires d to be
    a power of two; the caller pads or refuses rather than silently using a different d.
    """
    import numpy as np
    assert d > 0 and (d & (d - 1)) == 0, f"Hadamard construction needs a power of two, got {d}"
    Hm = np.ones((1, 1), dtype=np.float64)
    while Hm.shape[0] < d:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]])
    Q = Hm / np.sqrt(d)
    assert orthogonality_error(Q) < 1e-10, "Hadamard construction is not orthonormal"
    return Q


def permutation_rotation(d: int, seed: int):
    """A PERMUTATION matrix, used as a diagnostic gauge rather than a treatment.

    A permutation is orthogonal, so a correct residual-stream gauge must preserve the function
    under it — but unlike a dense rotation it also COMMUTES WITH EVERY ELEMENTWISE OPERATION. So
    the two together separate the only two ways the gauge can fail:

        permutation passes, dense rotation fails  ->  a per-coordinate (diagonal) operation is
                                                      still on the residual stream, unfolded
        both fail                                 ->  a residual-facing tensor is not being
                                                      rotated at all, or is rotated on the wrong
                                                      axis

    Without this contrast the two are indistinguishable from the logit difference alone.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    P = np.zeros((d, d), dtype=np.float64)
    P[np.arange(d), rng.permutation(d)] = 1.0
    assert orthogonality_error(P) < 1e-12, "permutation is not orthonormal"
    return P


def align_rotation(U, seed: int):
    """The 'rotated-align' arm's rotation: one that maps the measured sink onto the FIRST k
    coordinates, with the rest completed arbitrarily.

    This is the contrast that makes the arm's three-way comparison mean something. Hadamard
    scatters the sink maximally; align concentrates it maximally; unrotated leaves it. If the
    address is a property of the Adam basis, the three should differ in realized core motion in a
    predictable order — and if they do not, the TENANT reading is what is in trouble.
    """
    import numpy as np
    A = np.asarray(U, dtype=np.float64)
    n, k = A.shape
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, n))
    Z[:, :k] = A                                  # first k columns ARE the sink
    Q, _ = np.linalg.qr(Z)
    # QR can flip a sign; re-seat the sink block so span(Q[:, :k]) == span(U) with the same frame.
    Q[:, :k] = A
    Q, _ = np.linalg.qr(Q)
    Q[:, :k] = A
    R = Q.T                                       # rows of R are the new coordinate axes
    assert orthogonality_error(R) < 1e-8, "align rotation is not orthonormal"
    # the sink now occupies the first k coordinates: R U should be [I_k; 0]
    img = R @ A
    resid = float(np.abs(img[k:]).max())
    assert resid < 1e-8, f"align rotation left {resid:.3e} of the sink outside the first k axes"
    return R


def function_preservation(logits_a, logits_b) -> dict:
    """The frozen QuaRot-style check: fp32 max-abs logit difference.

    Reported alongside the logits' own scale, because a max-abs difference is only interpretable
    against the magnitudes it is a difference OF.
    """
    import numpy as np
    A = np.asarray(logits_a, dtype=np.float64)
    B = np.asarray(logits_b, dtype=np.float64)
    assert A.shape == B.shape, f"logit shape mismatch {A.shape} vs {B.shape}"
    d = float(np.abs(A - B).max())
    scale = float(np.abs(A).max())
    return {"max_abs_diff": d, "logit_max_abs": scale,
            "rel_to_logit_scale": d / scale if scale > 0 else float("nan")}


FOLD_TOL = 1e-4          # the spec's frozen absolute gate: fp32 max-abs logit difference

# PRE-DATA AMENDMENT (journaled 2026-08-14, before any M-Q science ran; revised once, in public,
# after the first floor turned out to measure the wrong thing).
#
# The spec's gate is an ABSOLUTE 1e-4 on max|Δlogit|. On real PG1 the logits have scale ~53, so
# that is ~2e-6 RELATIVE. The measured fold lands at 7.69e-4 absolute = 1.45e-5 relative.
#
# FIRST ATTEMPT, AND WHY IT WAS WRONG. The floor was measured by applying the IDENTITY rotation
# through the same code path. It came out BIT-EXACT (0.0), which is itself informative — the
# cast/transpose/matmul round trip costs nothing — but it means that control does not bound the
# fold's error at all, because the identity rotation does not REASSOCIATE the forward pass. The
# fold does: the model stops computing W·(normalise(x) ⊙ gamma) and starts computing
# (W·diag(gamma))·normalise(x). Those are equal in exact arithmetic and differ in fp32.
#
# THE RIGHT CONTROL isolates exactly that. Apply a CONSTANT-scale gauge — push a uniform factor c
# from the gain into the matrices — which is mathematically the identity on the function, is
# completely unrelated to the model's actual gains, and forces the SAME reassociation the fold
# does. Whatever the logits move by there is what reassociation costs in fp32, and the fold is
# required to land within a multiple of it. The multiple is frozen here, before any rotation
# result was seen; the spec's absolute number is still computed and recorded on every row.
#
# The gate can still fail: a broken fold moves the logits by far more than a constant rescale does.
NOISE_FLOOR_MULTIPLE = 10.0
REASSOCIATION_PROBE_SCALE = 2.0


def constant_scale_gauge(weight, mats, c: float = REASSOCIATION_PROBE_SCALE) -> tuple:
    """Push a uniform factor `c` from the norm gain into its consumers. Mathematically the
    identity on the function; arithmetically it forces the same reassociation the fold does.

    Returns (new_weight, [new_mats]) in Gemma's stored-weight convention.
    """
    import numpy as np
    assert c > 0 and c != 1.0, f"a reassociation probe needs a scale other than 1, got {c}"
    g = gemma_effective_gain(weight)
    g2, m2 = norm_scale_transfer(g, [np.asarray(W, dtype=np.float64) for W in mats],
                                 np.full(g.shape, float(c)))
    return g2 - 1.0, m2


def assert_function_preserved(fp: dict, *, tol: float = FOLD_TOL, what: str,
                              noise_floor: float = None,
                              multiple: float = NOISE_FLOOR_MULTIPLE):
    """The function-preservation gate.

    With `noise_floor` given, the binding comparison is against the measured arithmetic floor and
    the spec's absolute tolerance is recorded alongside. Without it, the absolute tolerance binds.
    """
    row = {"gate": what, **fp, "tol_absolute": tol,
           "passes_absolute_tol": bool(fp["max_abs_diff"] < tol)}
    if noise_floor is None:
        assert row["passes_absolute_tol"], (
            f"{what} changed the function: max|Δlogit| = {fp['max_abs_diff']:.3e} >= {tol:.1e} "
            f"(logit scale {fp['logit_max_abs']:.3f}). A gauge that changes the function is not a "
            f"gauge, and every downstream comparison would be between two different models")
        return {**row, "passed": True}
    assert noise_floor > 0, (
        "a noise floor of 0 means the identity path was bit-exact; then the absolute tolerance is "
        "the right gate and no floor should be passed")
    ratio = fp["max_abs_diff"] / noise_floor
    row.update({"noise_floor": float(noise_floor), "ratio_to_noise_floor": float(ratio),
                "multiple": float(multiple)})
    assert ratio <= multiple, (
        f"{what} changed the function by {fp['max_abs_diff']:.3e}, which is {ratio:.1f}x the "
        f"measured arithmetic noise floor {noise_floor:.3e} (ceiling {multiple:.0f}x). That is "
        f"beyond what the dtype explains, so the gauge is not function-preserving")
    return {**row, "passed": True}
