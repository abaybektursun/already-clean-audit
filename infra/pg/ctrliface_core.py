"""CTRLIFACE core — the pure, importable science of the control-interface study
(studies/lineage/paligemma1-pi0/research/2026-07-01_control_interface_spec.md).

Everything here is small-tensor torch math with NO model / datasets / GPU / lerobot
dependency, so the whole instrument's LOGIC is unit-testable weight-free on a laptop CPU
(infra/pg/tests/test_ctrliface_core.py) BEFORE any GPU minute is spent — the §9.1 dry-run
gate, distinct from the §9.0 runtime preflight. The CTRLIFACE jobs
(infra/aws_pg/jobs/ctrliface_*.py) are thin orchestration around these functions + the
reused rollout stack (lerobot [libero,pi]) + the pi0_port compartment gate.

The objects, per the frozen spec §2:
  * ACTIVATION-space gradient Gram  G_L = Σ ∇_{h_L} ℓ · ∇_{h_L} ℓ ᵀ  (a d×d matrix,
    d = backbone hidden). Its top-k eigenvectors are the sensitivity subspace S(k,L):
    S_action (ℓ = the deployed π0.5 policy's OWN flow-matching action loss) and
    S_text (ℓ = next-token CE on text). This is NEW code — w15_carrier's Gram is
    weight-space (diag(Uᵀ G V) on ΔW), a different object (spec §2 correction).
  * MEAN-ABLATION projector  out' = out − Q Qᵀ (out − h̄)  on the backbone residual
    stream (prefix positions), reusing projector_ablation.make_ablation_hook's math but
    re-registered on a decoder-layer output. Never zero-ablate (W5 lesson): replace the
    span(Q) component with h̄'s.
  * PRINCIPAL-ANGLE overlap  o(A,B) = ‖AᵀB‖_F² / min(kA,kB) = mean cos²θ_i  — the
    geometric relation between S_action and S_text (Arm B), vs a matched-random null.
  * ENERGY-MATCHED random subspace — the decisive null for H_interface: a random subspace
    whose EXPECTED removed residual energy equals S_action's, so the contrast controls for
    "how much variance was removed", not just rank.

Design rules (CLAUDE.md): NO try/except hiding — a wrong assumption RAISES with a trace.
Heavy imports (torch) live inside functions so the module py_compiles on the control plane.
"""
from __future__ import annotations

# The QR-of-Gaussian random orthonormal basis is reused verbatim from the W11 Leg-B
# instrument (pure torch, no heavy deps at import) — the artifact-gate / matched-random null.
from infra.pg.projector_ablation import random_basis  # (out_dim, k, seed, device, dtype) -> Q


# ===========================================================================
# Activation-space gradient Gram → sensitivity subspace S(k,L).
# ===========================================================================
def gram_update(G, grads):
    """Accumulate the d×d gradient Gram in place: G += gradsᵀ grads, where `grads` is an
    (n, d) block of per-position activation gradients ∇_{h_L} ℓ (n = positions×samples in the
    block). G is a d×d fp64 accumulator. Returns G (mutated). Pure — no model."""
    import torch
    grads = torch.as_tensor(grads, dtype=torch.float64)
    if grads.device != G.device:                          # accumulate on G's device (GPU-Gram ok)
        grads = grads.to(G.device)
    assert grads.ndim == 2, f"grads must be (n,d); got {tuple(grads.shape)}"
    assert G.shape == (grads.shape[1], grads.shape[1]), \
        f"G {tuple(G.shape)} vs grads d={grads.shape[1]}"
    G.add_(grads.transpose(0, 1) @ grads)
    return G


def topk_eigvecs(G, k):
    """Top-k eigenvectors (descending eigenvalue) of the SYMMETRIC PSD Gram G (d×d).
    Returns (Q: d×k orthonormal, eigvals: k,) fp64. Uses eigh (G is symmetric by
    construction). A negative eigenvalue beyond fp round-off RAISES (Gram must be PSD)."""
    import torch
    G = torch.as_tensor(G, dtype=torch.float64)
    assert G.shape[0] == G.shape[1], f"G must be square; got {tuple(G.shape)}"
    k = int(min(k, G.shape[0]))
    evals, evecs = torch.linalg.eigh((G + G.transpose(0, 1)) / 2.0)   # symmetrize for stability
    assert float(evals.min()) > -1e-6 * float(evals.abs().max().clamp_min(1e-30)), \
        f"Gram not PSD: min eigval {float(evals.min())}"
    order = torch.argsort(evals, descending=True)[:k]
    return evecs[:, order].contiguous(), evals[order].contiguous()


# ===========================================================================
# Mean-ablation on the residual stream (prefix positions), reused math.
# ===========================================================================
def apply_mean_ablation(x, Q, hbar):
    """out' = x − Q Qᵀ (x − h̄): mean-ablate the span(Q) component of x to h̄'s component.
    `x` is (..., d), `Q` is (d, k) orthonormal, `h̄` is (d,). Never zero (replace with h̄).
    Returns x' same shape/dtype as x. Pure — the projector_ablation.make_ablation_hook math,
    kept here so the dry-run can test it without a hook. Computed in a dtype promoted to ≥fp32
    (fp32 for a bf16/fp16 real hook — stability; fp64 for the fp64 dry-run — exactness), cast back."""
    import torch
    dt = x.dtype
    cdt = torch.promote_types(dt, torch.float32)   # bf16/fp16/fp32 → fp32 ; fp64 → fp64
    xf = x.to(cdt)
    Qf = Q.to(cdt)
    hf = hbar.to(cdt)
    coeff = (xf - hf) @ Qf              # (..., k)
    return (xf - coeff @ Qf.transpose(-1, -2)).to(dt)


def removed_energy(X, Q, hbar):
    """Σ_rows ‖Qᵀ(x − h̄)‖² — the residual energy the mean-ablation with basis Q removes
    over the rows of X (n, d). Used to compute S_action's removed energy E_a and to
    energy-match the random null. Returns a python float (fp64)."""
    import torch
    X = torch.as_tensor(X, dtype=torch.float64)
    Q = torch.as_tensor(Q, dtype=torch.float64)
    hbar = torch.as_tensor(hbar, dtype=torch.float64)
    C = (X - hbar) @ Q                  # (n, k)
    return float((C ** 2).sum().item())


def make_layer_ablation_hook(Q, hbar, n_prefix=None):
    """A forward hook for a Gemma decoder layer that mean-ablates span(Q) → h̄ on the
    residual-stream output. HF Gemma layers return a tuple (hidden_states, ...) OR a bare
    tensor — both handled. `n_prefix`: if given, ablate only the FIRST n_prefix positions
    (the image+language prefix the action expert reads from); if None, ablate all positions
    (the backbone processes only prefix tokens — Gate 0 confirms which). Fail-loud on a
    hidden-state shape whose last dim != Q.shape[0]."""
    import torch
    Qd = Q.shape[0]

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hs = out[0] if is_tuple else out
        assert hs.shape[-1] == Qd, f"hidden dim {hs.shape[-1]} != Q dim {Qd}"
        Qh = Q.to(hs.device)
        hh = hbar.to(hs.device)
        if n_prefix is None:
            hs2 = apply_mean_ablation(hs, Qh, hh)
        else:
            hs2 = hs.clone()
            hs2[:, :n_prefix, :] = apply_mean_ablation(hs[:, :n_prefix, :], Qh, hh)
        return (hs2, *out[1:]) if is_tuple else hs2

    return hook


def make_mean_replace_hook(hbar, n_prefix=None):
    """Positive-control hook: REPLACE the residual with h̄ on the (prefix) positions — the full-read
    ablation (every direction removed → the backbone stream carries only its mean → control must
    floor if the harness has teeth, spec §5). Handles tuple/tensor layer output."""
    import torch

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hs = out[0] if is_tuple else out
        hh = hbar.to(hs.device).to(hs.dtype)
        hs2 = hs.clone()
        if n_prefix is None:
            hs2[:] = hh
        else:
            hs2[:, :n_prefix, :] = hh
        return (hs2, *out[1:]) if is_tuple else hs2

    return hook


# ===========================================================================
# Principal-angle subspace overlap (Arm B) + matched-random overlap null.
# ===========================================================================
def orthonormalize(M):
    """QR-orthonormalize the columns of M (d, k) → Q (d, k) with orthonormal columns.
    Fail-loud on rank deficiency (a zero singular direction => the subspace is degenerate)."""
    import torch
    M = torch.as_tensor(M, dtype=torch.float64)
    Q, R = torch.linalg.qr(M)
    diag = R.diagonal().abs()
    assert float(diag.min()) > 1e-9, f"rank-deficient basis: min |R_ii|={float(diag.min())}"
    return Q[:, :M.shape[1]].contiguous()


def principal_angle_overlap(A, B):
    """o(A,B) = ‖AᵀB‖_F² / min(kA,kB) = mean over principal angles of cos²θ_i, in [0,1].
    A (d,kA), B (d,kB) must have ORTHONORMAL columns (caller orthonormalizes). 1 = nested,
    0 = orthogonal. Returns (overlap: float, cos_spectrum: 1-D fp64 = svdvals(AᵀB))."""
    import torch
    A = torch.as_tensor(A, dtype=torch.float64)
    B = torch.as_tensor(B, dtype=torch.float64)
    assert A.shape[0] == B.shape[0], f"A,B live in different R^d: {A.shape[0]} vs {B.shape[0]}"
    M = A.transpose(0, 1) @ B                       # (kA, kB)
    cos = torch.linalg.svdvals(M)                   # min(kA,kB) principal cosines
    kmin = int(min(A.shape[1], B.shape[1]))
    overlap = float((M ** 2).sum().item()) / kmin   # = Σ cos²θ / kmin
    return overlap, cos.contiguous()


def matched_random_overlap(d, kA, kB, n_draws, seed, ref=None):
    """Null distribution of principal_angle_overlap between two RANDOM orthonormal subspaces
    of ranks kA, kB in R^d (chance overlap under no geometric relation). If `ref` (d,kA
    orthonormal) is given, overlap is measured between `ref` and a random kB-subspace
    (holds one side fixed — the tighter null for 'is S_text specially aligned to S_action').
    Returns a 1-D fp64 tensor of length n_draws. Analytic mean ≈ max(kA,kB)/d for kA·kB≪d."""
    import torch
    out = torch.empty(n_draws, dtype=torch.float64)
    for i in range(n_draws):
        A = ref if ref is not None else random_basis(d, kA, seed * 100003 + 2 * i, "cpu", torch.float64)
        B = random_basis(d, kB, seed * 100003 + 2 * i + 1, "cpu", torch.float64)
        out[i], _ = principal_angle_overlap(A, B)
    return out


# ===========================================================================
# Energy-matched random subspace (the decisive null for H_interface).
# ===========================================================================
def energy_matched_rank(E_target, trace_C, d):
    """Rank r_e of a uniformly-random subspace whose EXPECTED removed residual energy equals
    E_target. For a uniformly random unit vector, E[qᵀ C q] = tr(C)/d; r orthonormal random
    directions remove ≈ r·tr(C)/d. So r_e = round(E_target · d / tr(C)), clamped to [1, d].
    (tr(C) = Σ_rows ‖x − h̄‖² over the SAME activation set E_target was measured on.)"""
    r = round(float(E_target) * float(d) / float(trace_C))
    return max(1, min(int(r), int(d)))


def energy_matched_random(d, E_target, trace_C, seed, device="cpu", dtype=None):
    """A random orthonormal basis Q (d, r_e) whose expected removed energy matches E_target
    (rank r_e = energy_matched_rank). Returns (Q, r_e). The §5 'equal-energy random' null;
    the 'equal-k random' artifact-gate null is just random_basis(d, k, ...)."""
    import torch
    r_e = energy_matched_rank(E_target, trace_C, d)
    Q = random_basis(d, r_e, seed, device, dtype if dtype is not None else torch.float64)
    return Q, r_e


# ===========================================================================
# RF (recoverable-fraction) readout + frozen decision rules (spec §6).
# ===========================================================================
def rf_control(success_ablate, success_intact, floor):
    """RF = (success_ablate − floor) / (success_intact − floor). 1 = ablation harmless,
    0 = ablation floors control. `floor` = measured random-action success (§2). Fail-loud
    if the intact policy is itself at floor (no dynamic range — a Gate-0 / P7 violation)."""
    denom = float(success_intact) - float(floor)
    assert denom > 1e-9, f"no dynamic range: intact {success_intact} ≈ floor {floor} (P7 halt)"
    return (float(success_ablate) - float(floor)) / denom


def _rf_from_pooled(pool_ablate, pool_intact, floor):
    import numpy as np
    mi = float(np.mean(pool_intact))
    denom = mi - floor
    if denom <= 1e-9:
        return float("nan")
    return (float(np.mean(pool_ablate)) - floor) / denom


def rf_cluster_bootstrap(ablate_by_task, intact_by_task, floor, n_boot=2000, seed=1337):
    """Task-CLUSTER bootstrap 95% CI on RF_control (spec §2/§7: resample TASKS, not episodes).
    `*_by_task` = {task_id: 1-D array of per-episode successes}. Point RF is on the full data; each
    bootstrap resamples the task set with replacement, pools those tasks' episodes, recomputes RF.
    Returns {rf, ci:[lo,hi], success_ablate, success_intact, n_tasks}."""
    import numpy as np
    tasks = sorted(set(ablate_by_task) & set(intact_by_task))
    assert tasks, "no shared tasks between ablate and intact"
    a_all = np.concatenate([np.asarray(ablate_by_task[t], float) for t in tasks])
    i_all = np.concatenate([np.asarray(intact_by_task[t], float) for t in tasks])
    point = _rf_from_pooled(a_all, i_all, floor)
    rng = np.random.default_rng(seed)
    boots = []
    T = len(tasks)
    for _ in range(n_boot):
        samp = rng.integers(0, T, size=T)
        pa = np.concatenate([np.asarray(ablate_by_task[tasks[j]], float) for j in samp])
        pi = np.concatenate([np.asarray(intact_by_task[tasks[j]], float) for j in samp])
        boots.append(_rf_from_pooled(pa, pi, floor))
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"rf": point, "ci": [lo, hi], "success_ablate": float(a_all.mean()),
            "success_intact": float(i_all.mean()), "n_tasks": T}


def gap_cluster_bootstrap(saction_by_task, random_by_task, intact_by_task, floor,
                          n_boot=2000, seed=1337):
    """The DECISIVE H_interface statistic: task-cluster bootstrap of the gap
    (RF_random_energy − RF_Saction). SAME resampled task set drives both RFs each draw (paired).
    Returns {gap, ci:[lo,hi], ci_lo}. ci_lo > δ ⇒ S_action beats the energy-matched null (§6)."""
    import numpy as np
    tasks = sorted(set(saction_by_task) & set(random_by_task) & set(intact_by_task))
    assert tasks, "no shared tasks across S_action/random/intact"

    def _pool(by_task, js):
        return np.concatenate([np.asarray(by_task[tasks[j]], float) for j in js])

    idx = np.arange(len(tasks))
    point = (_rf_from_pooled(_pool(random_by_task, idx), _pool(intact_by_task, idx), floor)
             - _rf_from_pooled(_pool(saction_by_task, idx), _pool(intact_by_task, idx), floor))
    rng = np.random.default_rng(seed)
    T = len(tasks)
    boots = []
    for _ in range(n_boot):
        js = rng.integers(0, T, size=T)
        pi = _pool(intact_by_task, js)
        g = (_rf_from_pooled(_pool(random_by_task, js), pi, floor)
             - _rf_from_pooled(_pool(saction_by_task, js), pi, floor))
        boots.append(g)
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"gap": point, "ci": [lo, hi], "ci_lo": lo}


def interface_verdict(rf_saction, rf_random_energy_lo_gap, tau_low, delta, monotone_ok):
    """Frozen §6 rule for Axis 1. `rf_saction` = RF under S_action ablation on the SEALED
    split; `rf_random_energy_lo_gap` = cluster-bootstrap CI-LOWER of (RF_random_energy −
    RF_Saction); `monotone_ok` = RF non-increasing in k up to k*. Returns 'H_interface' or
    'H_distributed'. H_interface iff the crater is real (RF≤τ_low), it BEATS the energy-
    matched null (gap CI-lo > δ), and it is monotone in k."""
    if (rf_saction <= tau_low) and (rf_random_energy_lo_gap > delta) and monotone_ok:
        return "H_interface"
    return "H_distributed"


def disjoint_verdict(overlap, overlap_null_hi, rf_control_stext, tau_high):
    """Frozen §6 rule for Axis 2. `overlap` = o(S_action,S_text) on the sealed split;
    `overlap_null_hi` = upper band of the matched-random overlap null; `rf_control_stext` =
    RF of control under S_text ablation. Returns 'H_disjoint' or 'H_shared'. H_disjoint iff
    S_action ⟂̸-below-chance to S_text (overlap < null) AND ablating text dirs preserves
    control (RF_control(S_text) ≥ τ_high)."""
    if (overlap < overlap_null_hi) and (rf_control_stext >= tau_high):
        return "H_disjoint"
    return "H_shared"
