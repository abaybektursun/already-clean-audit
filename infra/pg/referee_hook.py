"""REFEREE pure core — the update-replacement hook algebra, calibration arithmetic,
unit-checked power, and the divergence protocol (spec §2/§5). Numpy-only so every piece is
weight-free-testable on the laptop mirror; the training loop applies the same algebra to
torch tensors via `apply_replacement` semantics (identical formula, one code path per §11).

The §2 formula, literal:  u_total = u_AdamW + P_core(−η_c·λ·g_t − u_AdamW)
  - P is the IN-axis projector of the arm's subspace Q (d_in, k): P(M) = (M @ Q) @ Q.T
  - λ=0 denotes the UNHOOKED baseline (pre-freeze resolution: the formula at λ=0 would
    zero in-core motion, which contradicts P1's "λ=0 ≡ bit-identical AdamW"; the baseline
    arm therefore runs plain AdamW with no hook, and the formula applies for λ>0).

Fail-loud; no try/except."""
from __future__ import annotations

import numpy as np


def project_in(M, Q):
    """IN-axis projection (M @ Q) @ Q.T for M (out, d_in), Q (d_in, k) orthonormal."""
    return (M @ Q) @ Q.T


def replaced_update(u_adam, g, Q, eta_c, lam):
    """The §2 formula for one matrix. Returns u_total with the in-core component of the
    AdamW update REPLACED by the calibrated raw-gradient step; out-of-core untouched."""
    assert lam > 0, "lambda=0 is the unhooked baseline — do not call the hook"
    return u_adam + project_in(-eta_c * lam * np.asarray(g, np.float64)
                               - np.asarray(u_adam, np.float64), Q)


def calibrate_eta(disp25_tot, disp25_core, grad_incore_norms, n_steps=25):
    """eta_c from banked artifacts (frozen rule, provenance = tenant_x4/out/adamw_s0):
      target  = median over matrices of sqrt(disp25_tot - disp25_core) / n_steps
                (per-step OUT-of-core AdamW displacement magnitude; linear-accumulation
                convention, disclosed — P3 audits the realized motion and recalibrates
                pre-freeze on mismatch)
      source  = median over (matrix, step) of ||g_in-core||
      eta_c   = target / source
    All inputs are magnitudes from the SAME banked run; units cancel to a pure step-size."""
    disp25_tot = np.asarray(disp25_tot, np.float64)
    disp25_core = np.asarray(disp25_core, np.float64)
    oo = disp25_tot - disp25_core
    assert (oo > 0).all(), "out-of-core displacement must be positive"
    target = float(np.median(np.sqrt(oo)) / n_steps)
    source = float(np.median(np.asarray(grad_incore_norms, np.float64)))
    assert source > 0
    return target / source, {"target_perstep_oocore_disp": target,
                             "median_incore_grad_norm": source}


# ===========================================================================
# Unit-checked power (P6; the TENANT units-error replay is the offline gate's known event)
# ===========================================================================
def power_one_sided(effect, se, alpha, effect_units, se_units):
    """Max power of a one-sided z-test. REFUSES mismatched units — the TENANT F2 defect
    (S-scale effect against a raw-accuracy SE) is exactly what this assert exists to stop."""
    from scipy.stats import norm
    assert effect_units == se_units, \
        f"UNITS ERROR: effect in '{effect_units}' vs SE in '{se_units}' (TENANT F2 replay)"
    z = norm.ppf(1 - alpha)
    return float(1 - norm.cdf(z - effect / se))


# ===========================================================================
# Divergence protocol (P5; symmetric across arms)
# ===========================================================================
def divergence_verdict(final_loss, ref_loss, retried):
    """Mechanical P5: diverged iff final loss > DIVERGE_FACTOR x ref. First offence →
    'retry_halved'; second → 'VOID'. Converged → 'ok'."""
    DIVERGE_FACTOR = 3.0
    if final_loss <= DIVERGE_FACTOR * ref_loss:
        return "ok"
    return "VOID" if retried else "retry_halved"
