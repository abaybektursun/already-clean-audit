"""W14 localization/restoration — the pure, importable science core (no model, no
datasets, no GPU). Everything here is small-tensor numpy/torch math so it is unit-
testable on a laptop CPU (infra/pg/tests/test_w14_localize.py) BEFORE any GPU job
spends a single minute. The W14 jobs (infra/aws_pg/jobs/w14_*.py) are thin
orchestration around these functions + the inherited bridge spine.

What lives here (the genuinely NEW W14 science; see the prereg
studies/lineage/paligemma2/research/2026-06-02_w14_mlp_localize_restore_prereg.md):

  Arm 2 (localize the MMLU loss — MLP vs attention):
    * module_keys / norm_keys      — the q/k/v/o (104) vs gate/up/down (78) matrix
                                      partition of the 182 decoder matrices, + the norms.
    * reset_overrides              — build the {key: base_weight} override dict for a
                                      SUBSET reset (canvas = AFTER, reset only the subset
                                      to base). Consumed by e4_compress_curve._set_decoder.
                                      Asserts every targeted key is present + shape-matched
                                      and the swapped count == expected (fail-loud).
    * matched_numel_random_keys    — the GQA-asymmetric matched-Σ-numel random null
                                      (match real tensor numel, NOT key-count — q_proj is
                                      bigger than k/v_proj under GQA).

  Arm 1 (subspace privilege — does install need the LEARNED ΔW subspace or just rank?):
    * subspace_AB                  — the three rank-confined adapter inits at matched rank
                                      r: topk (= the no-retrain SVD edit), inspan (random
                                      rotation WITHIN ΔW's top-R_eff span), oospan (in the
                                      ORTHOGONAL COMPLEMENT). inspan/oospan are NEW (EQ1 only
                                      had unconstrained PEFT-random). All matched-norm to topk.
    * construction_audit           — the pre-train HALT gate: residual<1e-5 onto the right
                                      subspace, Gram−I<1e-5 orthonormality, matched-norm ±1e-3.

  Arm 3 (cheapest restoring edit):
    * restore_svd_graft            — rank-k SVD truncation of the RESTORE direction
                                      (base − after); grafted onto AFTER (W_after + dW_restore_k).
                                      rank→full ⇒ → base decoder (the positive control).

Design rules (CLAUDE.md): no try/except hiding; a wrong assumption RAISES with a trace.
Heavy imports (torch/numpy) live inside functions so the module py_compiles on the
control plane (matches infra/pg/text_eval.py house style).

The decoder-key constants below MIRROR infra/aws_pg/jobs/e4_compress_curve.py (which
lives in the jobs layer); test_w14_localize asserts decoder_keys() == that module's
_decoder_keys() so the two can never silently drift. N_LAYERS==26 is the §0 generation
firewall (Gemma-2, NOT the Gemma-1/18L sibling) — jobs assert it at load.
"""
from __future__ import annotations

# Gemma-2 LM architecture (PaliGemma 2). The §0 generation firewall: 26 layers.
N_LAYERS = 26
ATTN_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TYPES = ("gate_proj", "up_proj", "down_proj")
LM_MATRIX_TYPES = ATTN_TYPES + MLP_TYPES            # fixed order, mirrors e4_compress_curve

CONSTRUCTION_TOL = 1e-5     # residual / Gram−I tolerance for the in/out-span gate (prereg §A1.4)
MATCHED_NORM_TOL = 1e-3     # ‖A@B‖_F match tolerance vs top-k (prereg §A1.4)


# ===========================================================================
# Decoder-key partition (Arm 2 module reset; mirrors e4_compress_curve._decoder_keys).
# ===========================================================================

def _decoder_key(layer: int, mtype: str) -> str:
    sub = "self_attn" if mtype in ATTN_TYPES else "mlp"
    return f"layers.{layer}.{sub}.{mtype}.weight"


def decoder_keys() -> list[str]:
    """The 182 decoder MATRIX keys (no norms, no embed/lm_head), in the canonical
    e4_compress_curve order: layer 0→25, then mtype in LM_MATRIX_TYPES."""
    return [_decoder_key(L, mt) for L in range(N_LAYERS) for mt in LM_MATRIX_TYPES]


def module_keys(module: str) -> list[str]:
    """Matrix keys for one module family. 'attn' → 104 (q/k/v/o × 26),
    'mlp' → 78 (gate/up/down × 26), 'all' → 182."""
    if module == "all":
        return decoder_keys()
    if module == "attn":
        return [k for k in decoder_keys() if ".self_attn." in k]
    if module == "mlp":
        return [k for k in decoder_keys() if ".mlp." in k]
    raise ValueError(f"module must be one of attn/mlp/all, got {module!r}")


def norm_keys(sd_keys) -> list[str]:
    """The decoder RMSNorm keys present in a language_model state_dict (per-layer
    input/post-attention/pre-feedforward/post-feedforward norms + the final `norm`),
    EXCLUDING embed_tokens/lm_head. Used by the H_norm sensitivity arm (§A2.5).
    Derived from the live state_dict keys because norms are 1-D and not in decoder_keys()."""
    out = [k for k in sd_keys if "norm" in k and "embed" not in k and "lm_head" not in k]
    return sorted(out)


# ===========================================================================
# Subset reset overrides + matched-Σ-numel random null (Arm 2).
# ===========================================================================

def reset_overrides(after_sd: dict, base_sd: dict, reset_keys) -> dict:
    """Override dict for a SUBSET decoder reset: {k: base_sd[k]} for k in reset_keys,
    leaving every other key at AFTER. Consumed by e4_compress_curve._set_decoder(lm,
    after_sd, overrides, device). Fail-loud: every reset key must be present in BOTH
    state_dicts and shape-matched (a mistyped filter or missing key HALTS, never silently
    leaks PG/base weights)."""
    reset_keys = list(reset_keys)
    overrides = {}
    for k in reset_keys:
        if k not in after_sd:
            raise KeyError(f"reset key {k!r} absent from AFTER state_dict")
        if k not in base_sd:
            raise KeyError(f"reset key {k!r} absent from BASE state_dict")
        if tuple(after_sd[k].shape) != tuple(base_sd[k].shape):
            raise RuntimeError(
                f"shape mismatch on {k!r}: after={tuple(after_sd[k].shape)} "
                f"base={tuple(base_sd[k].shape)}")
        overrides[k] = base_sd[k]
    assert len(overrides) == len(set(reset_keys)) == len(reset_keys), (
        f"duplicate reset keys: {len(reset_keys)} given, {len(set(reset_keys))} unique")
    return overrides


def shuffled_reset_overrides(after_sd: dict, base_sd: dict, reset_keys, seed: int) -> dict:
    """Structure-destroying shuffle control for the Arm-2 reset (prereg §A2.5). The true
    reset removes the delta (after − base); this removes a delta of the SAME Frobenius norm
    but with its entries PERMUTED: W = after − perm(after − base). If recovery depends on the
    specific weight structure (not just the magnitude removed), the shuffled reset recovers
    far less than the true reset. Returns an override dict (same form as reset_overrides)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    ov = {}
    for k in reset_keys:
        delta = (after_sd[k] - base_sd[k]).reshape(-1)
        perm = torch.randperm(delta.numel(), generator=g)
        shuffled = delta[perm].reshape(after_sd[k].shape)
        ov[k] = after_sd[k] - shuffled
    return ov


def numel_of(sd: dict, keys) -> int:
    """Total real tensor numel over keys (GQA-asymmetric: q_proj > k/v_proj)."""
    return int(sum(int(sd[k].numel()) for k in keys))


def matched_numel_random_keys(candidate_keys, key_numel: dict, target_numel: int,
                              seed: int) -> tuple[list[str], int]:
    """Pick a random whole-matrix subset of candidate_keys whose Σ numel is as close as
    possible to target_numel — the matched-Σ-numel random null (prereg §A2.5/T12). Matches
    REAL numel, not key-count (GQA: q_proj has more params than k/v_proj). Whole-matrix
    granularity (no per-head selectivity — GQA grouping + RoPE phase make per-head claims
    artifactual, §8 T3). Greedy on a seeded shuffle; stops when the next key would worsen
    the |Σ − target| gap. Returns (chosen_keys, achieved_numel); the job logs achieved/target
    so any residual is disclosed, never hidden."""
    import numpy as np
    rng = np.random.default_rng(seed)
    order = list(candidate_keys)
    rng.shuffle(order)
    chosen, acc = [], 0
    for k in order:
        nk = int(key_numel[k])
        if acc + nk <= target_numel or abs(acc + nk - target_numel) < abs(acc - target_numel):
            chosen.append(k)
            acc += nk
        if acc >= target_numel:
            break
    return chosen, acc


# ===========================================================================
# Arm 1 — the three rank-confined adapter inits (topk / inspan / oospan).
# ===========================================================================

def effective_rank_energy(s, energy: float = 0.99) -> int:
    """Smallest rank whose cumulative σ² energy fraction ≥ energy (mirrors
    w13_decomp.lowrank_n). `s` is a 1-D singular-value tensor/array (descending)."""
    import torch
    s = torch.as_tensor(s, dtype=torch.float64)
    e2 = (s * s)
    frac = torch.cumsum(e2, 0) / e2.sum()
    idx = int(torch.searchsorted(frac, torch.tensor(energy, dtype=torch.float64, device=frac.device)).item()) + 1
    return max(1, min(idx, s.numel()))


def _orthonormal(n: int, k: int, gen) -> "object":
    """A random (n × k) matrix with orthonormal columns (k ≤ n) via QR of Gaussian."""
    import torch
    M = torch.randn(n, k, generator=gen, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    return Q[:, :k]


def subspace_feasible(dW, rank: int, kind: str, energy: float = 0.99) -> dict:
    """CPU pre-flight HALT gate (no GPU, no model): can a rank-`rank` `kind` adapter even be
    CONSTRUCTED in ΔW's energy span / complement? The geometry, honestly (prereg §A1.4):
    in-span draws k vectors from the R_eff-dim span ⇒ needs k ≤ R_eff; out-of-span draws k from
    the complement ⇒ needs k ≤ out_d − R_eff (left) AND k ≤ in_d − R_eff (right). R_eff is ΔW's
    TRUE effective energy rank from an fp64 full SVD (recompute per matrix, NO clamp). The Arm-1
    job calls this on the REAL ΔW for every targeted matrix BEFORE any model/GPU load and HALTs
    on infeasible — so an impossible (rank, kind) never burns a GPU minute."""
    import torch
    dW = torch.as_tensor(dW, dtype=torch.float64)
    out_d, in_d = dW.shape
    k = min(rank, out_d, in_d)
    _, S, _ = torch.linalg.svd(dW, full_matrices=False)
    R_eff = effective_rank_energy(S, energy)
    max_rank_inspan = R_eff
    max_rank_oospan = min(out_d - R_eff, in_d - R_eff)
    if kind == "topk":
        feasible = k <= R_eff                              # topk = the rank-k SVD edit, lives in the span
        reason = "" if feasible else f"k={k} > R_eff={R_eff} (topk lives in ΔW's energy span)"
    elif kind == "inspan":
        feasible = k <= max_rank_inspan
        reason = "" if feasible else f"k={k} > R_eff={R_eff} (in-span draws k from the R_eff-dim span)"
    elif kind == "oospan":
        feasible = k <= max_rank_oospan
        reason = "" if feasible else (
            f"k={k} > out_d−R_eff={out_d - R_eff} or in_d−R_eff={in_d - R_eff} "
            f"(out-of-span draws k from the complement)")
    else:
        raise ValueError(f"kind must be topk/inspan/oospan, got {kind!r}")
    return {"feasible": bool(feasible), "R_eff": int(R_eff), "out_d": int(out_d), "in_d": int(in_d),
            "k": int(k), "max_rank_inspan": int(max_rank_inspan),
            "max_rank_oospan": int(max_rank_oospan), "reason": reason}


def subspace_AB(dW, rank: int, kind: str, gen, energy: float = 0.99):
    """Build a rank-`rank` adapter init (A:(out,k), B:(k,in)) for one decoder matrix,
    confined to one of three subspaces of ΔW = W_after − W_before. The FROZEN recipe then
    trains A,B identically for all three — only the initial subspace differs, so the
    contrast isolates subspace identity (prereg §A1.3).

      kind='topk'   : A,B from the rank-k SVD of ΔW (= the no-retrain rank-k edit at init).
      kind='inspan' : a random orthonormal rotation of k directions DRAWN FROM ΔW's top-R_eff
                      left/right singular span (A non-zero, in-span throughout — unlike EQ1's
                      PEFT-random A=0 unconstrained search).
      kind='oospan' : the same, but directions drawn from the ORTHOGONAL COMPLEMENT of ΔW's
                      top-R_eff span (the correct matched-rank null EQ1 lacked).

    inspan/oospan are matched-norm to topk (‖A@B‖_F equal) so the contrast is subspace
    identity, not magnitude. Returns (A, B) in fp64 (job casts to deploy dtype)."""
    import torch
    dW = torch.as_tensor(dW, dtype=torch.float64)
    out_d, in_d = dW.shape
    k = min(rank, out_d, in_d)
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)   # U:(out,r) S:(r) Vh:(r,in)
    V = Vh.transpose(0, 1)                                  # (in, r)
    sq = S[:k].clamp_min(0).sqrt()

    if kind == "topk":
        A = U[:, :k] * sq
        B = (sq[:, None] * Vh[:k, :])
        return A, B

    R_eff = effective_rank_energy(S, energy)               # ΔW's TRUE energy span (no clamp — §A1.4)
    # fail-loud feasibility brace: subspace_feasible() is the CPU pre-flight gate the job runs first;
    # this catches a direct call at infeasible k (which would otherwise QR-truncate to a rank-deficient
    # draw and return silently). No try/except — HALT (§A1.4).
    if kind == "inspan":
        assert k <= R_eff, (f"in-span infeasible: k={k} > R_eff={R_eff} — gate with subspace_feasible().")
    elif kind == "oospan":
        cdim = min(out_d - R_eff, in_d - R_eff)
        assert k <= cdim, (f"out-of-span infeasible: k={k} > complement dim {cdim} "
                           f"(out_d−R_eff={out_d - R_eff}, in_d−R_eff={in_d - R_eff}) — gate with subspace_feasible().")
    target_fro = (S[:k] ** 2).sum().sqrt()                 # = ‖topk A@B‖_F (the matched norm)

    if kind == "inspan":
        U_R, V_R = U[:, :R_eff], V[:, :R_eff]              # (out,R_eff), (in,R_eff)
        Qu = U_R @ _orthonormal(R_eff, k, gen)             # k cols inside span(U_R)
        Qv = V_R @ _orthonormal(R_eff, k, gen)             # k cols inside span(V_R)
    elif kind == "oospan":
        Pu = U[:, :R_eff] @ U[:, :R_eff].transpose(0, 1)   # projector onto top-R_eff left span
        Pv = V[:, :R_eff] @ V[:, :R_eff].transpose(0, 1)
        Mu = torch.randn(out_d, k, generator=gen, dtype=torch.float64)
        Mv = torch.randn(in_d, k, generator=gen, dtype=torch.float64)
        Qu, _ = torch.linalg.qr(Mu - Pu @ Mu)              # orthonormalise in the complement
        Qv, _ = torch.linalg.qr(Mv - Pv @ Mv)
        Qu, Qv = Qu[:, :k], Qv[:, :k]
    else:
        raise ValueError(f"kind must be topk/inspan/oospan, got {kind!r}")

    # equal per-direction scale, then match ‖A@B‖_F to top-k exactly.
    sigma = target_fro / (k ** 0.5)
    A = Qu * sigma
    B = sigma * Qv.transpose(0, 1)
    cur = (A @ B).norm()
    if cur > 0:
        A = A * (target_fro / cur).sqrt()
        B = B * (target_fro / cur).sqrt()
    return A, B


def construction_audit(A, B, dW, rank: int, kind: str, energy: float = 0.99) -> dict:
    """The pre-train HALT gate (prereg §A1.4): verify (A,B) really sit in the claimed
    subspace and are matched-norm to top-k. Returns a dict of the measured residuals +
    a boolean `ok`; the job asserts ok and HALTs on failure (no try/except)."""
    import torch
    A = torch.as_tensor(A, dtype=torch.float64)
    B = torch.as_tensor(B, dtype=torch.float64)
    dW = torch.as_tensor(dW, dtype=torch.float64)
    out_d, in_d = dW.shape
    k = min(rank, out_d, in_d)
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    V = Vh.transpose(0, 1)
    R_eff = effective_rank_energy(S, energy)               # ΔW's TRUE energy span (no clamp — §A1.4)
    Pu = U[:, :R_eff] @ U[:, :R_eff].transpose(0, 1)       # projector onto top-R_eff left span
    Pv = V[:, :R_eff] @ V[:, :R_eff].transpose(0, 1)

    def _rel(x, P, project_in):
        comp = (P @ x) if project_in else (x - P @ x)      # the part that SHOULD vanish
        d = x.norm()
        return float((comp.norm() / d).item()) if d > 0 else 0.0

    target_fro = float((S[:k] ** 2).sum().sqrt().item())
    cur_fro = float((A @ B).norm().item())
    matched_norm_err = abs(cur_fro - target_fro)

    if kind == "topk":
        # top-k is trivially in-span and norm-exact by construction.
        a_resid = _rel(A, Pu, project_in=False)
        b_resid = _rel(B.transpose(0, 1), Pv, project_in=False)
    elif kind == "inspan":
        a_resid = _rel(A, Pu, project_in=False)            # off-span part of A must vanish
        b_resid = _rel(B.transpose(0, 1), Pv, project_in=False)
    elif kind == "oospan":
        a_resid = _rel(A, Pu, project_in=True)             # in-span part of A must vanish
        b_resid = _rel(B.transpose(0, 1), Pv, project_in=True)
    else:
        raise ValueError(f"kind must be topk/inspan/oospan, got {kind!r}")

    # orthonormality of the k init directions (prereg §A1.4: "Gram−I Frobenius < 1e-5"). Column-
    # normalise then ‖Q̂ᵀQ̂ − I‖_F; a rank-DEFICIENT draw (dependent columns) also fails this, so it
    # is the post-build belt to subspace_feasible's CPU pre-flight braces.
    def _gram_dev(M):
        n = M.norm(dim=0, keepdim=True).clamp_min(1e-30)
        Q = M / n
        return float((Q.transpose(0, 1) @ Q - torch.eye(Q.shape[1], dtype=Q.dtype)).norm().item())
    gram_A = _gram_dev(A)
    gram_B = _gram_dev(B.transpose(0, 1))

    ok = bool(a_resid < CONSTRUCTION_TOL and b_resid < CONSTRUCTION_TOL
              and gram_A < CONSTRUCTION_TOL and gram_B < CONSTRUCTION_TOL
              and matched_norm_err < MATCHED_NORM_TOL)
    return {"kind": kind, "rank": k, "R_eff": int(R_eff),
            "A_subspace_residual": a_resid, "B_subspace_residual": b_resid,
            "A_gram_dev": gram_A, "B_gram_dev": gram_B,
            "ab_frobenius": cur_fro, "topk_frobenius": target_fro,
            "matched_norm_abs_err": matched_norm_err,
            "construction_tol": CONSTRUCTION_TOL, "matched_norm_tol": MATCHED_NORM_TOL,
            "ok": ok}


# ===========================================================================
# Arm 3 — restore-direction rank-k graft.
# ===========================================================================

def restore_svd_graft(after_W, base_W, rank: int):
    """Targeted/untargeted low-rank RESTORE edit for one matrix (prereg §A3.3): graft the
    rank-k SVD truncation of the restore direction dW_restore = (W_base − W_after) onto the
    adapted weight: W = W_after + dW_restore_k. rank ≥ effective rank ⇒ W → W_base (the
    full-decoder restore positive control). fp64 SVD for a faithful eff-rank tail; the job
    casts the result to the deploy dtype and asserts fp64↔deploy agreement on the positive
    control. Returns the restored weight (fp64)."""
    import torch
    after_W = torch.as_tensor(after_W, dtype=torch.float64)
    base_W = torch.as_tensor(base_W, dtype=torch.float64)
    dW = base_W - after_W                                  # the RESTORE direction
    k = min(rank, *dW.shape)
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    dW_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
    return after_W + dW_k
