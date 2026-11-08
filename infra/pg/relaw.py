"""RELAW — the re-learnability-law science core (pure, importable, weight-free).

No model, no datasets, no GPU: small-tensor numpy/torch math, laptop-unit-tested
(infra/pg/tests/test_relaw.py) BEFORE any GPU job spends a minute. The relaw jobs
(infra/aws_pg/jobs/relaw_*.py) are thin orchestration around these functions plus the
REUSED savb law-fit engine + latmech weight conditions + relearn/mmlu spine (§13 ledger).

Prereg: studies/lineage/paligemma2/research/2026-06-27_relearnability_law_prereg.md.

The genuinely-NEW math RELAW needs (everything else is REUSED):

  PREDICTOR (the candidate law read, per decoder matrix, fp64):
    * principal_survival        — top-r principal-subspace SURVIVAL through a fine-tune:
                                  mean of (top-r LEFT, top-r RIGHT) singular-subspace overlap
                                  between W_base and W_FT. Built on w15_carrier.svd_fp64 +
                                  savb.subspace_overlap (rotation-invariant cos² principal
                                  angles). NOT savb.topk_right_basis, which mean-CENTRES the
                                  matrix (correct for activation rows, WRONG for a weight
                                  matrix — would subtract the column-mean and corrupt the
                                  singular geometry). Returns a scalar ∈ [0,1] per matrix.
    * pooled_principal_survival — pool principal_survival across the shared 2-D decoder
                                  matrices (unweighted mean) → the per-(base,FT)-pair read R_prin.

  INSULATION (Arm-3 causal lever, the projector reused every optimizer step):
    * topk_subspace             — cache the top-r LEFT/RIGHT singular bases of W_base once.
    * project_grad              — project a weight-gradient g onto (into) / off the top-σ
                                  principal BLOCK span{u_i v_jᵀ : i,j<r} = U_r U_rᵀ g V_r V_rᵀ.
                                  into ⊥ off in the Frobenius inner product, so
                                  ‖into‖²+‖off‖²=‖g‖² (the dry-run identity). off=insulated,
                                  into=anti-insulated (magnitude-matched adversarial null).
                                  This is EXACTLY the subspace principal_survival reads, so an
                                  insulation effect is the causal complement of the predictor.

  LAW-FIT predicates (pure; the §9.6 unblind predicate, testable on synthetic banked cells):
    * select_predictor_r        — freeze r on discovery: argmax_r LOO-R²(R_prin,G) (ties→smaller r).
    * loo_sign_stable           — drop each fine-tune, refit ΔLOO-R²; a sign flip ⇒ non-law.
    * bh_fdr                     — Benjamini-Hochberg reject set over the read family.
    * law_verdict               — the §6 MDE rule (perm-p<.05 ∧ FDR ∧ held-out lift>0 ∧ LOO-stable).
    * planted_g                 — the §5 positive control: synthetic G = slope·R + noise.

  FIREWALL (§0, load-bearing — RELAW spans Gemma-1/18L and Gemma-2/26L):
    * n_decoder_layers          — count distinct decoder layer indices in a key list; the job
                                  asserts ==18 (Gemma-1) vs ==26 (Gemma-2). No weight crosses.

Design rules (CLAUDE.md): no try/except hiding; a wrong assumption RAISES with a trace.
Heavy imports (torch/numpy) live inside functions so the module py_compiles on the control
plane (matches infra/pg/latmech.py + w15_carrier.py house style).
"""
from __future__ import annotations

import re


# ===========================================================================
# Predictor — principal-subspace survival (the candidate law read R_prin).
# ===========================================================================

def overlap_at_rank(Ub, Vb, Uf, Vf, r):
    """½·[left top-r subspace overlap + right top-r subspace overlap] from PRECOMPUTED fp64 SVD
    factors (U:(m,k), V:(n,k) per w15_carrier.svd_fp64). The hot-loop primitive: the reads job
    SVDs the shared base decoder ONCE and calls this at every r in the sweep, avoiding a re-SVD
    per rank. savb.subspace_overlap wants rows=basis vectors → take top-r columns, transpose."""
    from infra.pg import savb as SB
    rr = int(min(int(r), Ub.shape[1], Uf.shape[1], Vb.shape[1], Vf.shape[1]))
    assert rr >= 1, f"overlap_at_rank rank collapsed to {rr}"
    L = SB.subspace_overlap(Ub[:, :rr].transpose(0, 1).contiguous().numpy(),
                            Uf[:, :rr].transpose(0, 1).contiguous().numpy())
    R = SB.subspace_overlap(Vb[:, :rr].transpose(0, 1).contiguous().numpy(),
                            Vf[:, :rr].transpose(0, 1).contiguous().numpy())
    return 0.5 * (L + R)


def principal_survival(W_base, W_FT, r):
    """One 2-D decoder matrix pair → top-r principal-subspace survival ∈ [0,1].

    = ½·[ overlap(top-r LEFT singular subspace of W_base, of W_FT)
        + overlap(top-r RIGHT singular subspace of W_base, of W_FT) ],
    overlap = savb.subspace_overlap = mean cos² principal angles (1=identical span, →0
    orthogonal; rotation-invariant inside each top-r block). The fraction of the base
    model's dominant representational geometry that SURVIVES the fine-tune.

    Uses w15_carrier.svd_fp64 (raw fp64 SVD) — deliberately NOT savb.topk_right_basis,
    which mean-centres its input (right for activation matrices, wrong for weights). r is
    clamped to the available rank; returns a float."""
    from infra.pg import w15_carrier as C
    Ub, _Sb, Vb = C.svd_fp64(W_base)          # U:(m,k) V:(n,k), k=min(m,n)
    Uf, _Sf, Vf = C.svd_fp64(W_FT)
    return overlap_at_rank(Ub, Vb, Uf, Vf, r)


def pooled_principal_survival(dec_base, dec_ft, keys, r):
    """Pool principal_survival across the shared 2-D decoder matrices (unweighted mean) →
    the per-(base,FT)-pair read R_prin. `keys` must be present in BOTH decoders, ndim==2,
    matching shapes. Returns {pooled, per_matrix:{k:o}, n_mat, r_used}. Fail-loud."""
    import numpy as np
    assert keys, "no decoder keys to pool"
    per = {}
    for k in keys:
        Wb, Wf = dec_base[k], dec_ft[k]
        assert Wb.ndim == 2 and Wf.ndim == 2, f"{k}: not 2-D ({Wb.ndim}/{Wf.ndim})"
        assert tuple(Wb.shape) == tuple(Wf.shape), f"{k}: shape mismatch {tuple(Wb.shape)} vs {tuple(Wf.shape)}"
        per[k] = float(principal_survival(Wb, Wf, r))
    return {"pooled": float(np.mean(list(per.values()))), "per_matrix": per,
            "n_mat": len(per), "r_used": int(r)}


# ===========================================================================
# Insulation — gradient projection onto / off the top-σ principal block (Arm-3).
# ===========================================================================

def topk_subspace(W, r):
    """Cache the top-r LEFT/RIGHT singular bases of W_base once (fp64). Returns
    (U_r:(m,r), V_r:(n,r)) contiguous; project_grad reuses them every optimizer step."""
    from infra.pg import w15_carrier as C
    U, _S, V = C.svd_fp64(W)
    rr = int(min(int(r), U.shape[1], V.shape[1]))
    assert rr >= 1, f"topk_subspace rank collapsed to {rr}"
    return U[:, :rr].contiguous(), V[:, :rr].contiguous()


def project_grad(g, U_r, V_r, mode):
    """Project a weight-gradient g (out,in) onto/off the top-σ principal BLOCK of W_base.

      into = U_r (U_rᵀ g V_r) V_rᵀ   (component in span{u_i v_jᵀ : i,j<r})
      off  = g − into

    into and off are Frobenius-orthogonal ⇒ ‖into‖² + ‖off‖² = ‖g‖² (the §9.1 dry-run
    identity). mode='off' → INSULATED (the FT update avoids the principal subspace);
    mode='into' → ANTI-insulated (the magnitude-matched adversarial null, Arm-3 denominator).
    Runs in g's dtype/device (the trainer's bf16/fp32); cast U_r/V_r to match."""
    import torch
    g = torch.as_tensor(g)
    Ur = torch.as_tensor(U_r).to(dtype=g.dtype, device=g.device)
    Vr = torch.as_tensor(V_r).to(dtype=g.dtype, device=g.device)
    into = Ur @ (Ur.transpose(0, 1) @ g @ Vr) @ Vr.transpose(0, 1)
    if mode == "into":
        return into
    if mode == "off":
        return g - into
    raise ValueError(f"project_grad mode must be 'off'/'into', got {mode!r}")


def insulated_update(g, U_r, V_r, mode, eps=1e-12):
    """The per-step Arm-3 gradient surgery (one decoder matrix). mode:
      'standard' → g (the raw gradient; the natural FT).
      'off'      → INSULATED: the gradient orthogonalised off the top-σ block (spares the
                   principal subspace R_prin reads).
      'into'     → ANTI-insulated: the in-subspace component, MAGNITUDE-MATCHED to the
                   insulated 'off' step (‖anti‖ = ‖off‖) but pointed INTO the principal
                   subspace — the adversarial null that maximally damages the scaffold at the
                   SAME step size, so an 'insulated > anti' effect cannot be a magnitude artifact
                   (§5/§8-T6). Returns a tensor the optimizer steps on in place of g."""
    import torch
    g = torch.as_tensor(g)
    if mode == "standard":
        return g
    off = project_grad(g, U_r, V_r, "off")
    if mode == "off":
        return off
    if mode == "into":
        into = project_grad(g, U_r, V_r, "into")
        scale = off.norm() / into.norm().clamp_min(eps)
        return into * scale
    raise ValueError(f"insulated_update mode must be 'standard'/'off'/'into', got {mode!r}")


# ===========================================================================
# Law-fit predicates (pure; reuse savb.{loo_r2,delta_loo_perm} for the heavy lifting).
# ===========================================================================

def _loo_delta(X_base, x_add, y):
    """ΔLOO-R²(x_add | X_base) directly (loo_r2(aug) − loo_r2(base)), no permutation."""
    import numpy as np
    from infra.pg import savb as SB
    X_base = np.asarray(X_base, float)
    x_add = np.asarray(x_add, float).reshape(-1, 1)
    y = np.asarray(y, float)
    return SB.loo_r2(np.column_stack([X_base, x_add]), y) - SB.loo_r2(X_base, y)


def select_predictor_r(r_grid, loo_r2_by_r):
    """Freeze the predictor rank on discovery: argmax_r LOO-R²(R_prin, G) over r_grid (ties →
    smaller r). `loo_r2_by_r` maps r → LOO-R². Returns the chosen int r."""
    best_r, best = None, None
    for r in r_grid:
        v = float(loo_r2_by_r[r])
        if best is None or v > best + 1e-12:
            best, best_r = v, int(r)
    assert best_r is not None, "empty r_grid"
    return best_r


def loo_sign_stable(X_base, x_add, y):
    """Drop each fine-tune, recompute ΔLOO-R²(x_add | X_base); STABLE iff the sign never flips
    (the §6 'does not depend on any single fine-tune flipping the sign'). Returns
    {stable, base_delta, drop_deltas}."""
    import numpy as np
    X_base = np.asarray(X_base, float)
    x_add = np.asarray(x_add, float)
    y = np.asarray(y, float)
    n = y.size
    base = _loo_delta(X_base, x_add, y)
    drops = []
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        drops.append(_loo_delta(X_base[keep], x_add[keep], y[keep]))
    stable = all((d > 0) == (base > 0) for d in drops)
    return {"stable": bool(stable), "base_delta": float(base),
            "drop_deltas": [float(d) for d in drops]}


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg reject set at level q. Returns a boolean array aligned to pvals."""
    import numpy as np
    p = np.asarray(pvals, float)
    n = p.size
    assert n >= 1, "no p-values"
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    reject = np.zeros(n, bool)
    if passed.any():
        kmax = int(np.nonzero(passed)[0].max())
        reject[order[:kmax + 1]] = True
    return reject


def law_verdict(perm_p, fdr_ok, heldout_lift, loo_stable):
    """§6 MDE rule — a read PREDICTS iff its ΔLOO-R²(read|R_dist) permutation-p < 0.05
    (BH-FDR-surviving) AND the held-out fine-tune OOS R²(read) > R²(R_dist) (lift>0) AND it is
    LOO-stable. Pure boolean predicate."""
    return bool(perm_p < 0.05 and fdr_ok and heldout_lift > 0 and loo_stable)


def planted_g(R, slope, noise_sd, seed):
    """The §5 positive control / dry-run synthetic family: G = slope·R + N(0, noise_sd).
    Used to prove the law-fit machinery RECOVERS a real predictor (not impotent)."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    R = np.asarray(R, float)
    return slope * R + rng.normal(0.0, float(noise_sd), R.shape)


# ===========================================================================
# Generation firewall (§0) — never diff weights across model generations.
# ===========================================================================

def n_decoder_layers(keys):
    """Count distinct decoder layer indices ('...layers.<N>...') in a state_dict key list.
    The job asserts ==18 (Gemma-1 PG1) vs ==26 (Gemma-2 PG2) so a cross-generation weight
    diff RAISES instead of silently producing an invalid number (§0 firewall, §8-T4)."""
    idx = set()
    for k in keys:
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            idx.add(int(m.group(1)))
    assert idx, f"no decoder layer indices found in {len(list(keys))} keys"
    return len(idx)
