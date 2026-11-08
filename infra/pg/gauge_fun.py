"""GAUGE-FUN (P5) — a correctly-typed function-space reachability metric (pure, weight-free).

Spec: studies/lineage/paligemma1-pi0/research/2026-07-04_gauge_program_spec.md §8.

Question (§8.1). RELAW's deflation — no cheap WEIGHT read predicts re-learnability beyond weight
distance — compared weight-space rulers to a FUNCTION-space quantity. The correctly-typed attack:
does a function-space distance (output-divergence / representation / curvature-aware) predict the
banked savings G where a ‖ΔW‖-equivalent read saturates? Estimand (§8.2): predictive power for G
of each candidate metric M GIVEN R_dist — ΔLOO-R²(M | R_dist) with permutation p — the FROZEN
RELAW instrument, REUSED VERBATIM (savb.delta_loo_perm; relaw.{bh_fdr,loo_sign_stable,law_verdict}).
BH (q=0.10) over the ≤4-metric menu. A RANGE GATE is evaluated FIRST (banked-G spread vs the G
measurement-noise band, both from the banked artifacts); if it fires → H_range-bound and NO metric
verdict is issued.

This module holds ONLY the pure, model-free science: the 4 metric MATH KERNELS (each a pure
function on activation/logit/gradient tensors, weight-free-testable), the range gate, the frozen
rank freeze, and the menu→verdict wiring. All model forwards + artifact reads live in the job
(infra/aws_pg/jobs/gauge_fun.py). Heavy imports (numpy) live inside functions so the module
py_compiles on the control plane (house style: relaw.py, gauge_cs.py). fp64 throughout. No
try/except — a wrong assumption RAISES with a trace.

The frozen metric menu (§8.2, ≤4, CLOSED at freeze — any new idea → §13 ledger, never here):
  1. D_KL     — symmetrized output KL on the fixed frozen probe stream (base vs unit logits).
  2. D_rep    — 1 − linear-CKA of mid-depth residual activations on the same stream (depth = the
                banked SAV overlap layer, read from the artifact by the job).
  3. D_fisher — diagonal-Fisher-weighted parameter distance (per-coordinate Fisher from per-token
                gradients on the probe stream, SKETCHED — the cheapest curvature-aware ruler).
  4. D_ntk    — principal-angle distance between the top-r per-example-gradient subspaces of base
                vs unit (r FROZEN at the RELAW rank-grid midpoint; a tangent-kernel proxy that
                never materialises the NTK).
"""
from __future__ import annotations

# The frozen, closed 4-metric menu (§8.2). Order is the BH/reporting order. No post-hoc additions.
MENU = ("D_KL", "D_rep", "D_fisher", "D_ntk")


# ===========================================================================
# Metric kernel 1 — D_KL: symmetrized output KL on the fixed probe stream.
# ===========================================================================

def sym_kl(logits_p, logits_q):
    """Symmetrized output KL between two per-token logit streams on the FIXED probe stream.

    logits_p, logits_q: (N_tokens, vocab) fp64 logits (base vs unit) over the SAME probe tokens.
    Returns the mean over tokens of ½·[KL(P‖Q) + KL(Q‖P)] in nats, where P=softmax(logits_p),
    Q=softmax(logits_q). Softmax is computed stably (subtract the row max). Symmetric in its
    arguments, ≥0, and exactly 0 iff the two logit rows induce the same distribution at every
    token. fp64."""
    import numpy as np
    P = np.asarray(logits_p, np.float64)
    Q = np.asarray(logits_q, np.float64)
    assert P.shape == Q.shape and P.ndim == 2, f"sym_kl wants matching 2-D logits, got {P.shape}/{Q.shape}"
    logp = P - _logsumexp(P)
    logq = Q - _logsumexp(Q)
    p = np.exp(logp)
    q = np.exp(logq)
    kpq = (p * (logp - logq)).sum(axis=1)          # KL(P‖Q) per token
    kqp = (q * (logq - logp)).sum(axis=1)          # KL(Q‖P) per token
    return float((0.5 * (kpq + kqp)).mean())


def _logsumexp(A):
    """Row-wise log-sum-exp of a 2-D array (stable normaliser for softmax), keepdims. fp64."""
    import numpy as np
    A = np.asarray(A, np.float64)
    mx = A.max(axis=1, keepdims=True)
    return mx + np.log(np.exp(A - mx).sum(axis=1, keepdims=True))


# ===========================================================================
# Metric kernel 2 — D_rep: 1 − linear-CKA of mid-depth residual activations.
# ===========================================================================

def linear_cka(X, Y):
    """Linear CKA between two activation matrices X (N×d1), Y (N×d2) captured on the SAME N probe
    tokens (base vs unit, at the banked overlap layer). Both column-centred, then
        CKA = ‖YᵀX‖_F² / (‖XᵀX‖_F · ‖YᵀY‖_F)  ∈ [0, 1],
    invariant to orthogonal transforms and to isotropic scaling of either side (a representation-
    similarity index, Kornblith 2019). fp64. Returns the scalar CKA."""
    import numpy as np
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0], \
        f"linear_cka wants same-N 2-D activations, got {X.shape}/{Y.shape}"
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = float(np.linalg.norm(Y.T @ X, "fro") ** 2)
    nrm = float(np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))
    assert nrm > 0, "degenerate (zero-variance) activations in linear_cka"
    return hsic_xy / nrm


def d_rep(X, Y):
    """The P5 representation-divergence metric = 1 − linear-CKA(X, Y) ∈ [0, 1]. 0 iff the base/unit
    mid-depth representations align up to an orthogonal transform + isotropic scale."""
    return float(1.0 - linear_cka(X, Y))


# ===========================================================================
# Metric kernel 3 — D_fisher: diagonal-Fisher-weighted parameter distance.
# ===========================================================================

def fisher_diag(per_token_grads):
    """Sketched diagonal Fisher = mean over probe tokens of the squared (sketched) per-token loss
    gradient. per_token_grads: (N_tokens, p_sketch) — the job feeds RANDOM-PROJECTED / coordinate-
    sub-sampled per-token gradients of the next-token NLL (never the full NTK). Returns the
    length-p_sketch non-negative diagonal. fp64."""
    import numpy as np
    G = np.asarray(per_token_grads, np.float64)
    assert G.ndim == 2, f"fisher_diag wants (N_tokens, p_sketch), got {G.shape}"
    return (G * G).mean(axis=0)


def fisher_weighted_distance(fisher_diag_vec, delta):
    """Diagonal-Fisher-weighted parameter distance D_fisher = sqrt( Σ_i F_i · Δ_i² ).

    fisher_diag_vec: per-coordinate Fisher (≥0; the sketched diagonal from `fisher_diag`).
    delta:           the matched per-coordinate parameter change (W_unit − W_base) in the SAME
                     sketch/coordinate frame as the Fisher.
    With F ≡ 1 this reduces to the plain ℓ2 distance ‖Δ‖ — the curvature-agnostic ruler RELAW
    already deflated; the Fisher weighting is what makes D_fisher a *function-space* read. fp64."""
    import numpy as np
    F = np.asarray(fisher_diag_vec, np.float64).ravel()
    d = np.asarray(delta, np.float64).ravel()
    assert F.shape == d.shape, f"fisher/delta shape mismatch {F.shape} vs {d.shape}"
    assert (F >= 0).all(), "Fisher diagonal must be non-negative"
    return float(np.sqrt(float((F * d * d).sum())))


# ===========================================================================
# Metric kernel 4 — D_ntk: principal-angle distance between top-r gradient subspaces.
# ===========================================================================

def _top_r_grad_subspace(G, r):
    """Top-r right singular subspace (rows = r orthonormal basis vectors) of a per-example gradient
    matrix G (N_examples × p_sketch), via savb.topk_right_basis (mean-centred over examples,
    fp64). REUSED so D_ntk shares the exact subspace primitive RELAW/SAV-B use."""
    from infra.pg import savb as SB
    G_np = _asarray64(G)
    assert G_np.ndim == 2, f"grad matrix must be 2-D, got {G_np.shape}"
    r = int(r)
    assert 1 <= r <= min(G_np.shape), f"D_ntk rank {r} out of range for grad matrix {G_np.shape}"
    return SB.topk_right_basis(G_np, k=r)


def principal_angle_distance(Gb, Gu, r):
    """D_ntk — principal-angle distance between the top-r per-example-gradient subspaces of base vs
    unit on the probe stream:
        = 1 − mean cos²(principal angles) = 1 − savb.subspace_overlap(top-r(Gb), top-r(Gu)),
    ∈ [0, 1]; 0 iff the two tangent-kernel top-r subspaces coincide, →1 as they become orthogonal.
    Gb, Gu: (N_examples × p_sketch) per-example loss-gradient matrices (base, unit). r is FROZEN at
    the RELAW rank-grid midpoint (`frozen_ntk_rank`) — NEVER selected post-hoc. A tangent-kernel
    proxy that never materialises the NTK (the subspaces are the only NTK-derived object touched)."""
    from infra.pg import savb as SB
    Vb = _top_r_grad_subspace(Gb, r)
    Vu = _top_r_grad_subspace(Gu, r)
    return float(1.0 - SB.subspace_overlap(Vb, Vu))


def frozen_ntk_rank(r_grid):
    """The D_ntk subspace rank, FROZEN at the RELAW rank-grid midpoint = r_grid[len//2] (§8.2). For
    the canonical RELAW 7-grid [8,16,32,64,128,256,512] this is 64. Deliberately NOT chosen by
    relaw.select_predictor_r: P5 freezes r pre-real (single-variable discipline — no post-hoc rank
    fishing over the metric menu). The job clamps the returned r to the sketch/example dimensions if
    the frozen midpoint would exceed them, and discloses the clamp."""
    g = [int(x) for x in r_grid]
    assert g, "empty rank grid"
    return int(g[len(g) // 2])


def _asarray64(A):
    import numpy as np
    return np.asarray(A, np.float64)


# ===========================================================================
# Range gate (§8.2) — evaluated FIRST, before any metric verdict.
# ===========================================================================

def range_gate(g_values, g_sigmas):
    """The §8.2 range gate — the pre-specified H_range-bound rule, evaluated BEFORE any metric
    verdict. H_range-bound FIRES iff the between-unit SPREAD of the banked savings G does not exceed
    the within-unit G MEASUREMENT-NOISE band (both from the banked artifacts):

        SNR = std(G) / rms(σ_G)   →   range_bound iff SNR ≤ 1.

    g_values: per-unit banked G (= savc S_16k). g_sigmas: per-unit G measurement σ, read by the job
    from the banked SAV bootstrap band (unit_*.json outcomes.16000.S_boot; the savc pool SUMMARY
    drops it — a GATE-0 read the job performs, fail-loud). If the gate fires the FAMILY, not the
    metric, is the limit and NO metric verdict is issued. Returns
    {range_bound, snr, g_spread, g_noise, n}."""
    import numpy as np
    g = np.asarray(g_values, np.float64).ravel()
    s = np.asarray(g_sigmas, np.float64).ravel()
    assert g.shape == s.shape, f"G/σ length mismatch {g.shape} vs {s.shape}"
    assert g.size >= 2, "range gate needs ≥2 units"
    assert (s >= 0).all(), "negative G measurement σ"
    spread = float(g.std(ddof=1))                  # between-unit signal
    noise = float(np.sqrt((s * s).mean()))         # within-unit measurement noise (RMS)
    assert noise > 0, "zero G measurement-noise band — cannot form the range-gate SNR"
    snr = spread / noise
    return {"range_bound": bool(snr <= 1.0), "snr": float(snr),
            "g_spread": spread, "g_noise": noise, "n": int(g.size)}


# ===========================================================================
# Menu → verdict wiring (REUSES the frozen RELAW instrument verbatim).
# ===========================================================================

def run_menu(metrics, R_dist, G, g_sigmas, q=0.10, n_perm=2000, seed=1337):
    """The P5 estimand end-to-end, per §8.2 — the RELAW instrument REUSED VERBATIM, no re-derivation:

      1. RANGE GATE first (`range_gate`). If it fires → verdict H_range-bound, metric verdicts NOT
         issued.
      2. else per frozen-menu metric M: ΔLOO-R²(M | R_dist) with permutation p
         (savb.delta_loo_perm), LOO sign-stability (relaw.loo_sign_stable).
      3. BH (q, default 0.10) over the ≤4 menu p-values (relaw.bh_fdr).
      4. per metric law_verdict (relaw.law_verdict): perm-p<0.05 ∧ BH-survives ∧ ΔLOO-R²>0
         (out-of-sample lift) ∧ LOO-stable.
      5. overall: H_function-metric if ANY metric clears, else H_deflation-deepens.

    metrics: {name: per-unit array} over a subset of MENU (unknown-metric names RAISE). R_dist, G,
    g_sigmas: per-unit arrays sharing one unit order. Returns a JSON-able verdict record. fp64."""
    import numpy as np
    from infra.pg import savb as SB
    from infra.pg import relaw as RL

    names = [m for m in MENU if m in metrics]
    unknown = [m for m in metrics if m not in MENU]
    assert not unknown, f"metric(s) {unknown} are OUTSIDE the frozen menu {MENU} — §13 ledger, not here"
    assert names, "no menu metrics supplied"

    R = np.asarray(R_dist, np.float64).reshape(-1, 1)
    y = np.asarray(G, np.float64).ravel()
    n = y.size
    assert R.shape[0] == n, f"R_dist length {R.shape[0]} != n_units {n}"
    for m in names:
        assert np.asarray(metrics[m]).size == n, f"{m}: length {np.asarray(metrics[m]).size} != n_units {n}"

    gate = range_gate(y, g_sigmas)
    if gate["range_bound"]:
        return {"object": "gauge_fun_verdict", "n_units": int(n), "range_gate": gate,
                "menu": list(names), "metrics_issued": False, "verdict": "H_range-bound",
                "note": "banked-G SNR ≤ 1 — the family, not the metric, is the limit; metric "
                        "verdicts NOT issued (§8.1 H_range-bound, pre-specified before any metric)."}

    per = {}
    pvals = []
    for m in names:
        x = np.asarray(metrics[m], np.float64).ravel()
        dperm = SB.delta_loo_perm(R, x, y, n_perm=n_perm, seed=seed)
        stab = RL.loo_sign_stable(R, x, y)
        per[m] = {"delta_loo_r2": dperm["delta"], "p_perm": dperm["p_perm"],
                  "r2_base": dperm["r2_base"], "r2_aug": dperm["r2_aug"],
                  "loo_stable": bool(stab["stable"]), "base_delta": stab["base_delta"]}
        pvals.append(dperm["p_perm"])

    reject = RL.bh_fdr(pvals, q=q)
    verdicts = {}
    for i, m in enumerate(names):
        v = RL.law_verdict(per[m]["p_perm"], bool(reject[i]),
                           per[m]["delta_loo_r2"], per[m]["loo_stable"])
        per[m]["fdr_ok"] = bool(reject[i])
        per[m]["law_verdict"] = bool(v)
        verdicts[m] = bool(v)

    any_clears = any(verdicts.values())
    return {"object": "gauge_fun_verdict", "n_units": int(n), "range_gate": gate,
            "menu": list(names), "bh_q": float(q), "n_perm": int(n_perm),
            "per_metric": per, "verdicts": verdicts, "metrics_issued": True,
            "verdict": "H_function-metric" if any_clears else "H_deflation-deepens",
            "note": ("≥1 function-space metric adds ΔLOO-R² power beyond R_dist under the RELAW "
                     "rule (§8.1 H_function-metric) — cheap reuse-triage reopens") if any_clears else
                    ("no menu metric clears the RELAW rule with dynamic range intact "
                     "(§8.1 H_deflation-deepens) — the closure widens to function space")}
