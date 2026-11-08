"""HUMBOLDT core — banks, the twin battery, the statistic catalog, and the atlas writer.

Spec: studies/lineage/paligemma1-pi0/research/2026-08-09_humboldt_program_spec.md
This is a MEASUREMENT library. It records; it does not decide. Nothing here computes a verdict.

FAIL LOUD (CLAUDE.md): no try/except, no synthetic fallback. Heavy imports live inside functions so
this py_compiles + unit-tests on the control plane with numpy alone.

Objects (§2.1). Per (type, layer) matrix of the PG1 LM decoder, over the N=24 official PG1
fine-tunes, each ΔW normalised to unit Frobenius:

    G = Σ_i ΔW_iᵀ ΔW_i / ‖ΔW_i‖²_F        (n×n, fp64, trace = N)
    G = V Λ Vᵀ,  Λ ascending
    core32 = V[:, :32]     (consensus-AVOIDED input subspace)
    anti32 = V[:, -32:]    (consensus-maximally-WRITTEN input subspace)

Precision (journal C3). G is accumulated in fp64 straight from the fp32 checkpoints. The cached
invariant_core scratch is fp16, whose quantisation noise is the same order as the bottom-k
eigenvalues; it is used ONLY for the G2a bit-determinism gate, never to define a bank.

The twin battery (§5) is mandatory: every statistic on every real bank is recorded, in the same job
and the same code path, for haar32 (×3 seeds), bottomsigma32, anti32, and complement32.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

# ---------------------------------------------------------------------------
# Frozen catalog constants (§2.1, §2.3). Changing any of these post-freeze is an ADDENDUM.
# ---------------------------------------------------------------------------
PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"
N_LAYERS = 18
BANK_K = 32

# All seven decoder matrix types. `_IN_DIM` is the INPUT (row-space) dimension the bank lives in.
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TYPES_RESID = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj")   # n = 2048
IN_DIM = {"q_proj": 2048, "k_proj": 2048, "v_proj": 2048, "o_proj": 2048,
          "gate_proj": 2048, "up_proj": 2048, "down_proj": 16384}

# Which activation tensor each type's bank lives in (journal C2). The bank of a type is measured at
# the module whose INPUT spans that space — that is the only site where e_U is the right object.
SITE = {"q_proj": "q_proj_in", "k_proj": "q_proj_in", "v_proj": "q_proj_in",
        "o_proj": "o_proj_in", "gate_proj": "gate_proj_in", "up_proj": "gate_proj_in",
        "down_proj": "down_proj_in"}
# Phase B sites actually forwarded (§2.1 excludes down for cost).
SITES_PHASE_B = ("q_proj_in", "o_proj_in", "gate_proj_in")

# Bank ids in the battery. The real objects are core32/anti32; the rest are twins (§5).
HAAR_SEEDS = (0, 1, 2)
BANK_IDS = (("core32", "anti32", "bottomsigma32", "complement32")
            + tuple(f"haar32_s{s}" for s in HAAR_SEEDS))
TWIN_IDS = ("haar32_s0", "haar32_s1", "haar32_s2", "bottomsigma32", "anti32", "complement32")

# Rank-deficient types: the base weight has an exact kernel in its input space, so a literal
# "bottom-32 right-singular subspace" would be an arbitrary kernel basis (journal C5).
RANK_DEFICIENT = {"k_proj", "v_proj", "down_proj"}
MQA_KERNEL_TYPES = {"k_proj", "v_proj"}     # 256×2048 MQA shape (ERLANGEN critic)

DISCOVERY = "DISCOVERY"
ADDENDUM = "ADDENDUM"


def type_flags(t: str) -> dict:
    """The per-type caveat flags baked into EVERY row of that type (§2.1, §8, journal C5)."""
    n = IN_DIM[t]
    m = {"q_proj": 2048, "k_proj": 256, "v_proj": 256, "o_proj": 2048,
         "gate_proj": 16384, "up_proj": 16384, "down_proj": 2048}[t]
    return {"mqa_kernel_caveat": t in MQA_KERNEL_TYPES,
            "rank_deficient": t in RANK_DEFICIENT,
            "bottomsigma_economy": t in RANK_DEFICIENT,
            "n_sigma": int(min(m, n)), "n_kernel_dims": int(max(0, n - min(m, n))),
            "site": SITE[t]}


# ---------------------------------------------------------------------------
# Summary measure (§2.3). Ledger correction 1: battery_tasks.bootstrap_S_ci is a three-arm
# floor/ceiling S-estimator and does NOT fit "mean + bootstrap 95% CI". This is that estimator.
# ---------------------------------------------------------------------------
RESAMPLE_MAX_N = 5000        # above this, the resampling bootstrap is replaced by its exact SE
Z95 = 1.959963984540054


def mean_ci(values, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Mean + bootstrap 95% CI + n. The house summary measure for sampled quantities.

    Two regimes, and the one used is RECORDED in `ci_method`:
      * n ≤ RESAMPLE_MAX_N — the percentile bootstrap, resampled in memory-bounded chunks.
      * n > RESAMPLE_MAX_N — the mean's bootstrap SE has a closed form, sqrt(Σ(xᵢ−x̄)²/n²), and at
        n in the tens of thousands the percentile interval is indistinguishable from x̄ ± 1.96·SE.
        A per-token B1 series has n ≈ 75 000 and there are ~750 of them per (model, corpus);
        resampling those would allocate a 1.2 GB index array per call and turn a minutes-long arm
        into an hours-long one, to move the interval by less than its last printed digit.
    """
    import numpy as np
    a = np.asarray(values, dtype=np.float64).ravel()
    assert a.size > 0, "mean_ci on an empty sample"
    assert np.isfinite(a).all(), "mean_ci saw non-finite values"
    m = float(a.mean())
    if a.size == 1:
        return {"mean": m, "ci_lo": m, "ci_hi": m, "n": 1, "n_boot": 0, "seed": seed,
                "ci_method": "degenerate"}
    if a.size > RESAMPLE_MAX_N:
        se = float(np.sqrt(((a - m) ** 2).sum() / a.size ** 2))
        return {"mean": m, "ci_lo": m - Z95 * se, "ci_hi": m + Z95 * se, "n": int(a.size),
                "n_boot": 0, "seed": int(seed), "ci_method": "exact_bootstrap_se"}
    rng = np.random.default_rng(seed)
    per = max(1, int(2_000_000 // a.size))          # bound the index array to ~16 MB
    chunks = []
    drawn = 0
    while drawn < n_boot:
        k = min(per, n_boot - drawn)
        chunks.append(a[rng.integers(0, a.size, (k, a.size))].mean(axis=1))
        drawn += k
    boots = np.concatenate(chunks)
    return {"mean": m, "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "n": int(a.size),
            "n_boot": int(n_boot), "seed": int(seed), "ci_method": "percentile_bootstrap"}


# ---------------------------------------------------------------------------
# Subspace primitives.
# ---------------------------------------------------------------------------
def orthonormal_check(U, tol: float = 1e-8) -> float:
    """max |UᵀU − I|. Fail loud above tol — a non-orthonormal bank silently corrupts every angle,
    every leverage and every energy fraction downstream."""
    import numpy as np
    k = U.shape[1]
    d = float(np.abs(U.T @ U - np.eye(k)).max())
    assert d < tol, f"bank not orthonormal: max|UᵀU−I| = {d:.3e} (tol {tol})"
    return d


def principal_angles(U, V):
    """Principal angles (radians, ascending) between two orthonormal bases of the SAME ambient
    space. Length min(k_U, k_V). cos θ = singular values of UᵀV, clipped to [0,1] for fp safety."""
    import numpy as np
    assert U.shape[0] == V.shape[0], f"ambient mismatch {U.shape[0]} vs {V.shape[0]}"
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.arccos(np.clip(s, 0.0, 1.0)).astype(np.float64)


def haar_angle_band(n: int, k: int, n_draw: int = 200, seed: int = 1337) -> dict:
    """Analytic-by-simulation reference band for principal angles between two INDEPENDENT Haar
    k-subspaces of ℝⁿ (A3 smoke test; recomputed, never hardcoded). Returns per-angle-index
    mean/p2.5/p97.5 over n_draw independent pairs."""
    import numpy as np
    rng = np.random.default_rng(seed)
    acc = np.empty((n_draw, k), dtype=np.float64)
    for d in range(n_draw):
        A = np.linalg.qr(rng.standard_normal((n, k)))[0][:, :k]
        B = np.linalg.qr(rng.standard_normal((n, k)))[0][:, :k]
        acc[d] = principal_angles(A, B)
    return {"mean": acc.mean(axis=0).tolist(),
            "p2.5": np.percentile(acc, 2.5, axis=0).tolist(),
            "p97.5": np.percentile(acc, 97.5, axis=0).tolist(),
            "n_draw": int(n_draw), "n": int(n), "k": int(k), "seed": int(seed)}


def leverage(U):
    """Per-coordinate leverage of a bank: ℓ_i = ‖U[i, :]‖² ∈ [0, 1], Σ_i ℓ_i = k. For a Haar
    k-subspace of ℝⁿ, E[ℓ_i] = k/n (the '1/64 class' at k=32, n=2048). Full vector, never summarised
    away (§3/A1)."""
    import numpy as np
    return (np.asarray(U, dtype=np.float64) ** 2).sum(axis=1)


def participation_ratio(v) -> float:
    """PR = (Σ v)² / Σ v² of a non-negative vector. n for a flat profile, →1 for a single spike."""
    import numpy as np
    a = np.asarray(v, dtype=np.float64)
    assert (a >= 0).all(), "participation_ratio on a signed vector"
    s2 = float((a ** 2).sum())
    assert s2 > 0, "participation_ratio on an all-zero vector"
    return float(a.sum() ** 2 / s2)


def haar_subspace(n: int, k: int, seed: int):
    """A Haar-random orthonormal k-frame in ℝⁿ (QR of a seeded Gaussian)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return np.ascontiguousarray(Q[:, :k].astype(np.float64))


def complement_subspace(U_core, seed: int):
    """A random k-subspace drawn INSIDE the orthogonal complement of U_core ('generic non-core
    direction', §5.4). k matches U_core. Gaussian → project out U_core → QR → project again.

    PROJECT, ORTHONORMALISE, PROJECT AGAIN. The original order was project-then-QR, and QR is the
    step that undoes it: the projected Z still carries core components at rounding level, and the
    triangular solve inside QR divides by the small norms of the later columns and amplifies them.
    One re-projection after the QR costs nothing and is the textbook fix ("twice is enough"), and
    the second orthonormalisation perturbs the first only to second order.

    The inputs are checked because this function is handed the output of an eigensolver and then
    uses it as an EXACT projector: `U (Uᵀ Z)` removes the span of U only if UᵀU is actually I, so a
    basis that is merely nearly-orthonormal silently leaves a residue that shows up here, at the
    assert, rather than where it came from. Checking the input names the real culprit instead.
    """
    import numpy as np
    U = np.asarray(U_core, dtype=np.float64)
    n, k = U.shape
    assert np.isfinite(U).all(), (
        f"U_core has non-finite entries ({int((~np.isfinite(U)).sum())} of {U.size}) — the "
        f"complement cannot be drawn against a basis that is not a basis")
    assert n - k >= k, (
        f"the orthogonal complement of a {k}-subspace of R^{n} has dimension {n - k}, which "
        f"cannot hold {k} orthonormal directions")
    dev = float(np.abs(U.T @ U - np.eye(k)).max())
    assert dev < 1e-6, (
        f"U_core is not orthonormal (max |UᵀU - I| = {dev:.3e}); it is used here as an exact "
        f"projector, so fix it where it is produced rather than loosening this bound")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    Z -= U @ (U.T @ Z)
    Q, _ = np.linalg.qr(Z)
    Q = np.ascontiguousarray(Q[:, :k], dtype=np.float64)
    Q -= U @ (U.T @ Q)
    Q, _ = np.linalg.qr(Q)
    Q = np.ascontiguousarray(Q[:, :k], dtype=np.float64)
    leak = float(np.abs(U.T @ Q).max())
    assert leak < 1e-8, (
        f"complement sample leaked into the core (max |UᵀQ| = {leak:.3e}, n={n}, k={k}, "
        f"input orthonormality deviation {dev:.3e})")
    return Q


def sigma_subspaces(W0, k: int):
    """Economy right-singular subspaces of the base weight W0 (m×n), fp64.
    Returns (bottom_k, top_k, sigma) where bottom_k = the k SMALLEST-σ economy right-singular
    vectors and top_k the k largest. For rank-deficient types (m < n) the economy basis spans only
    min(m,n) of ℝⁿ — that is deliberate (journal C5): the bottom-σ twin must be the smallest USED
    directions, not an arbitrary basis of the exact kernel."""
    import numpy as np
    A = np.asarray(W0, dtype=np.float64)
    _, s, Vh = np.linalg.svd(A, full_matrices=False)      # Vh: r×n, σ descending
    V = np.ascontiguousarray(Vh.T)                        # n×r
    r = V.shape[1]
    assert k <= r, f"k={k} exceeds economy rank r={r}"
    return np.ascontiguousarray(V[:, r - k:]), np.ascontiguousarray(V[:, :k]), s.astype(np.float64)


def build_banks(V_gram, W0, k: int = BANK_K, seed_base: int = 1337) -> dict:
    """The full battery for one (type, layer) matrix (§2.1 + §5), as an ordered dict
    bank_id → (n×k) fp64 orthonormal basis. V_gram = eigenvectors of G, ASCENDING eigenvalue order.

    seed_base must be derived from the run seed AND the cell so twins differ across cells but are
    reproducible; callers pass cell_seed(...)."""
    import numpy as np
    n = V_gram.shape[0]
    core = np.ascontiguousarray(V_gram[:, :k].astype(np.float64))
    anti = np.ascontiguousarray(V_gram[:, -k:].astype(np.float64))
    bots, _tops, _s = sigma_subspaces(W0, k)
    banks = {"core32": core, "anti32": anti, "bottomsigma32": bots,
             "complement32": complement_subspace(core, seed_base + 900)}
    for s in HAAR_SEEDS:
        banks[f"haar32_s{s}"] = haar_subspace(n, k, seed_base + s)
    for bid, U in banks.items():
        assert U.shape == (n, k), f"{bid} shape {U.shape} != {(n, k)}"
        orthonormal_check(U)
    assert set(banks) == set(BANK_IDS), f"battery mismatch: {sorted(banks)} vs {sorted(BANK_IDS)}"
    return banks


def cell_seed(run_seed: int, *parts) -> int:
    """Deterministic per-cell seed derived from the run seed + the cell key (logged in every row)."""
    h = hashlib.sha1(("|".join(str(p) for p in parts)).encode()).hexdigest()[:8]
    return (run_seed * 1_000_003 + int(h, 16)) % (2 ** 31 - 1)


# ---------------------------------------------------------------------------
# G1 — bank loading / key alignment (§1 kill-table row 1).
# ---------------------------------------------------------------------------
def reconstruction_residual(W, V_full, k: int = BANK_K) -> float:
    """‖W − (WU)Uᵀ − (WU⊥)U⊥ᵀ‖_F / ‖W‖_F where U = V_full[:, :k] and U⊥ = V_full[:, k:] are the two
    halves of the SAME eigenbasis (§1 kill-table row 1).

    U⊥ must be materialised from the eigenbasis, NOT defined as W(I − UUᵀ): with that definition
    the expression collapses to W − WUUᵀ − (W − WUUᵀ) ≡ 0 identically, and the gate can only ever
    report 0 — a check that cannot fail is not a check. Splitting the real eigenbasis instead tests
    what the gate is for: that V_full is a COMPLETE orthonormal basis of the space W's rows live in,
    i.e. that the bank was keyed to the right matrix, at the right layer, in the right orientation.
    A wrong key, a transposed matrix, or a truncated eigenbasis all show up here as a large residual.
    """
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    Vf = np.asarray(V_full, dtype=np.float64)
    assert A.shape[1] == Vf.shape[0], \
        f"key misalignment: W input dim {A.shape[1]} vs eigenbasis ambient {Vf.shape[0]}"
    U, Up = Vf[:, :k], Vf[:, k:]
    resid = A - (A @ U) @ U.T - (A @ Up) @ Up.T
    return float(np.linalg.norm(resid) / np.linalg.norm(A))


def bank_energy_fraction(W, U) -> float:
    """‖W U‖²_F / ‖W‖²_F — how much of the base matrix's action is carried on the bank. A recorded
    statistic (not a gate); the Haar reference is k/n."""
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    return float(np.linalg.norm(A @ U) ** 2 / np.linalg.norm(A) ** 2)


# ---------------------------------------------------------------------------
# A2 — RoPE band coupling (§3/A2).
# ---------------------------------------------------------------------------
def rope_bands(cfg) -> dict:
    """Read the rotary layout off the model config — do NOT assume (§3/A2). HF Gemma applies
    `rotate_half`: the head-dim vector is split in half and dim j is paired with dim j + head_dim/2
    by the SAME inverse frequency θ_b, b = 0 .. head_dim/2 − 1. Returns the layout + a per-band pair
    index array, and fails loud on any config that is not this convention."""
    import numpy as np
    head_dim = int(getattr(cfg, "head_dim"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv = int(getattr(cfg, "num_key_value_heads"))
    prf = float(getattr(cfg, "partial_rotary_factor", 1.0))
    assert prf == 1.0, f"partial rotary factor {prf} != 1 — rotate_half layout assumption invalid"
    # transformers normalises rope config: a model on plain RoPE may carry rope_scaling=None, {},
    # or {"rope_type": "default"} depending on the version. Only "default" preserves the band
    # layout this decomposition assumes; anything else (linear, dynamic, yarn, llama3) rescales the
    # frequencies and must fail loud rather than be silently mis-binned.
    rs = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
    rs_type = None if not rs else (rs.get("rope_type") or rs.get("type") if isinstance(rs, dict)
                                   else str(rs))
    assert rs_type in (None, "default"), \
        f"rope_scaling={rs} (type {rs_type!r}) — band layout is only verified for plain RoPE"
    assert head_dim % 2 == 0, f"odd head_dim {head_dim}"
    n_band = head_dim // 2
    # transformers has moved the rope base around between versions (top-level `rope_theta`, then a
    # `rope_parameters` dict). Read it from wherever it is and RECORD which name supplied it; fail
    # loud with the available keys rather than assuming a default — §3/A2 says read the config.
    base, base_src = None, None
    if getattr(cfg, "rope_theta", None) is not None:
        base, base_src = float(cfg.rope_theta), "cfg.rope_theta"
    else:
        for attr in ("rope_parameters", "rope_scaling"):
            d = getattr(cfg, attr, None)
            if isinstance(d, dict) and d.get("rope_theta") is not None:
                base, base_src = float(d["rope_theta"]), f"cfg.{attr}['rope_theta']"
                break
    assert base is not None, (
        "no rope base on the config under rope_theta / rope_parameters / rope_scaling; "
        f"available keys: {sorted(cfg.to_dict())[:40]}")
    b = np.arange(n_band, dtype=np.float64)
    return {"head_dim": head_dim, "num_attention_heads": n_heads, "num_key_value_heads": n_kv,
            "n_bands": n_band, "rope_theta": base, "rope_theta_source": base_src,
            "inv_freq": (1.0 / (base ** (2.0 * b / head_dim))).tolist(),
            "pairs": np.stack([b, b + n_band], axis=1).astype(np.int64).tolist(),
            "convention": "rotate_half(dim j ↔ j+head_dim/2)"}


def rope_coupling(W, U, head_dim: int, n_out_heads: int):
    """coupling(b, U) = ‖P_b W U‖²_F / ‖W U‖²_F over rotary bands b, summed over heads (§3/A2).
    W is (n_out_heads·head_dim, n); U is (n, k). Returns a length-head_dim/2 vector summing to 1."""
    import numpy as np
    A = np.asarray(W, dtype=np.float64)
    assert A.shape[0] == n_out_heads * head_dim, \
        f"W rows {A.shape[0]} != heads {n_out_heads} × head_dim {head_dim}"
    P = (A @ U).reshape(n_out_heads, head_dim, U.shape[1])          # (H, D, k)
    e = (P ** 2).sum(axis=2)                                        # (H, D) energy per out-dim
    nb = head_dim // 2
    band = e[:, :nb].sum(axis=0) + e[:, nb:].sum(axis=0)            # (nb,) pair (j, j+nb)
    tot = float(e.sum())
    assert tot > 0, "W U has zero energy — the bank is in the null space of W"
    return (band / tot).astype(np.float64)


# ---------------------------------------------------------------------------
# Activation-side statistic (§3/Phase B).
# ---------------------------------------------------------------------------
def energy_fraction(H, U):
    """e_U = ‖U Uᵀ h‖² / ‖h‖² per token. H is (T, n) fp32/fp64, U is (n, k) orthonormal.
    Returns a length-T fp64 vector in [0, 1]. fp64 accumulation (§8 bit-nondeterminism row)."""
    import numpy as np
    X = np.asarray(H, dtype=np.float64)
    C = X @ np.asarray(U, dtype=np.float64)          # (T, k)
    num = (C ** 2).sum(axis=1)
    den = (X ** 2).sum(axis=1)
    assert (den > 0).all(), "zero-norm residual token — e_U undefined"
    return (num / den).astype(np.float64)


def autocorr(x, max_lag: int):
    """Biased autocorrelation of a 1-D series at lags 1..max_lag (mean-removed, r(0)-normalised).
    B2's series statistic. Returns a length-max_lag fp64 vector; lags beyond the series give nan."""
    import numpy as np
    a = np.asarray(x, dtype=np.float64)
    a = a - a.mean()
    denom = float((a ** 2).sum())
    out = np.full(max_lag, np.nan, dtype=np.float64)
    if denom == 0:
        return out                       # a constant series has no autocorrelation structure
    for L in range(1, min(max_lag, a.size - 1) + 1):
        out[L - 1] = float((a[:-L] * a[L:]).sum() / denom)
    return out


def spectral_peaks(x, top: int = 5):
    """Top-`top` peaks of the periodogram of a 1-D series (period in tokens + normalised power).
    B2's 'spectral density peaks'. DC bin excluded."""
    import numpy as np
    a = np.asarray(x, dtype=np.float64)
    a = a - a.mean()
    if float((a ** 2).sum()) == 0:
        return {"period": [], "power": []}
    P = np.abs(np.fft.rfft(a)) ** 2
    P[0] = 0.0
    tot = float(P.sum())
    assert tot > 0, "zero-power spectrum on a non-constant series"
    idx = np.argsort(P)[::-1][:top]
    freqs = np.fft.rfftfreq(a.size)
    return {"period": [float(1.0 / f) if f > 0 else float("inf") for f in freqs[idx]],
            "power": [float(P[i] / tot) for i in idx]}


# ---------------------------------------------------------------------------
# The atlas (§10). ONE uniform table, written incrementally (§0.2.4).
# ---------------------------------------------------------------------------
ATLAS_COLUMNS = ("arm", "model_id", "model_sha", "layer", "type", "bank_id", "statistic",
                 "value", "vector_ref", "n", "ci_lo", "ci_hi", "seed", "corpus_id", "item_id",
                 "token_class", "flags", "run_id", "commit_sha", "tag")


def hf_setup():
    """§0.2.7 env pins, in the one place every HUMBOLDT job calls FIRST, before importing
    transformers/datasets. HF_HUB_DISABLE_XET=1 is the house fix for the Xet download wedge
    (reference_hf_xet_wedge); hf_online_setup authenticates the gated Gemma/PaliGemma repos."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    # A DOWNLOAD THAT HANGS IS WORSE THAN ONE THAT FAILS. Without a timeout `snapshot_download`
    # can sit on a stalled connection indefinitely: the re-census wedged mid-shard on a 7B model
    # and held a live instance with a frozen log for five hours until the reaper took it, and a
    # retry cannot help because a hang raises nothing (M-D97). Set HERE rather than in the one job
    # that hung, so every downloading job in the program gets it.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()


PROVENANCE_FILE = pathlib.Path(__file__).resolve().parent / "humboldt_provenance.json"


def provenance() -> dict:
    """Run provenance for every atlas row (§0.12: every number resolves to run id + path + SHA).

    `pg run` ships ONLY the `infra/` tree — there is no .git on the instance — so the sha is
    stamped into infra/pg/humboldt_provenance.json by scripts/humboldt_stamp.py immediately before
    each launch (scripts/humboldt_launch.sh does both in one step, so it cannot go stale). On the
    control plane the file may be absent; then git answers directly."""
    if PROVENANCE_FILE.exists():
        return json.loads(PROVENANCE_FILE.read_text())
    root = pathlib.Path(__file__).resolve().parents[2]
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       check=True, capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                       check=True, capture_output=True, text=True)
    return {"commit_sha": r.stdout.strip(), "dirty": bool(d.stdout.strip()),
            "source": "git (control plane)"}


def commit_sha() -> str:
    return provenance()["commit_sha"]


def _jsonable(v):
    """Flags land in a JSON column. numpy scalars/arrays are not JSON-serialisable, and finding that
    out three hours into an arm — after the rows that mattered — is the wrong time. Convert them
    here, and raise on anything genuinely unserialisable rather than coercing it silently."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    raise TypeError(f"flag value of type {type(v).__name__} is not JSON-serialisable: {v!r}")


class Atlas:
    """Append-only atlas writer. Rows land in out/atlas.jsonl as they are produced (crash-proof);
    `to_parquet()` renders out/atlas.parquet at the end of a job. Vector-valued statistics go to
    out/vectors/<ref>.npy and the row carries the ref (§10)."""

    def __init__(self, out_dir, arm: str, run_id: str, tag: str = DISCOVERY):
        # atlas.parquet is a named deliverable (§10). Prove the renderer exists in the FIRST second
        # of the job, not after four hours of Grams.
        import pandas            # noqa: F401
        import pyarrow           # noqa: F401
        self.out = pathlib.Path(out_dir)
        self.vec = self.out / "vectors"
        self.vec.mkdir(parents=True, exist_ok=True)
        self.arm = arm
        self.run_id = run_id
        self.tag = tag
        self.sha = commit_sha()
        self.path = self.out / "atlas.jsonl"
        self.n_rows = 0
        self._fh = open(self.path, "a", buffering=1)          # line-buffered: durable per row

    def vector(self, name: str, arr, dtype="float64") -> str:
        """Bank a vector-valued statistic. `name` must already be cell-unique; collisions fail loud
        (a silently overwritten vector is an unrecoverable data loss).

        2-D arrays are allowed: a per-token × per-bank block is stored ONCE and each bank's row
        references its column as `vectors/<name>.npy#<col>`. Storing one file per bank would mean
        ~23k tiny files for B1 alone. `dtype="float32"` halves the per-token blocks."""
        import numpy as np
        safe = name.replace("/", "_")
        p = self.vec / f"{safe}.npy"
        assert not p.exists(), f"vector ref collision: {p.name}"
        np.save(p, np.asarray(arr, dtype=dtype))
        return f"vectors/{p.name}"

    def add(self, statistic: str, *, value=None, vector_ref=None, layer=None, type=None,
            bank_id=None, model_id=None, model_sha=None, n=None, ci_lo=None, ci_hi=None,
            seed=None, corpus_id=None, item_id=None, token_class=None, flags=None, tag=None):
        assert (value is None) != (vector_ref is None), \
            f"{statistic}: exactly one of value / vector_ref (got {value!r} / {vector_ref!r})"
        if value is not None:
            value = float(value)
            assert value == value, f"{statistic}: NaN value (record it as a flag, not a silent NaN)"
        flags = _jsonable(flags or {})
        row = {"arm": self.arm, "model_id": model_id, "model_sha": model_sha,
               "layer": None if layer is None else int(layer), "type": type, "bank_id": bank_id,
               "statistic": statistic, "value": value, "vector_ref": vector_ref,
               "n": None if n is None else int(n),
               "ci_lo": None if ci_lo is None else float(ci_lo),
               "ci_hi": None if ci_hi is None else float(ci_hi),
               "seed": None if seed is None else int(seed), "corpus_id": corpus_id,
               "item_id": item_id, "token_class": token_class,
               "flags": json.dumps(flags or {}, sort_keys=True),
               "run_id": self.run_id, "commit_sha": self.sha, "tag": tag or self.tag}
        assert set(row) == set(ATLAS_COLUMNS), \
            f"schema drift: {sorted(set(row) ^ set(ATLAS_COLUMNS))}"
        self._fh.write(json.dumps(row) + "\n")
        self.n_rows += 1

    def add_ci(self, statistic: str, values, **kw):
        """Sampled statistic: mean + bootstrap CI + n, per §2.3. The CI regime rides in the flags —
        which interval a row carries is part of its provenance, not an implementation detail."""
        seed = kw.pop("seed", 1337)
        s = mean_ci(values, seed=seed)
        flags = {**(kw.pop("flags", None) or {}), "ci_method": s["ci_method"],
                 "n_boot": s["n_boot"]}
        self.add(statistic, value=s["mean"], n=s["n"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                 seed=seed, flags=flags, **kw)
        return s

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def to_parquet(self):
        """Render out/atlas.parquet from the jsonl. Fails loud if pyarrow is absent — the parquet is
        a named deliverable (§10), not an optional nicety.

        Every arm calls close() then to_parquet(); flushing an already-closed handle here would
        raise at the LAST step of a multi-hour job, after all the science was already computed."""
        import pandas as pd
        self.close()
        df = pd.read_json(self.path, lines=True)
        # AN EMPTY ATLAS IS A LEGITIMATE OUTCOME, NOT DRIFT (M-D92). An arm that reaches a
        # recorded outcome — VOID-INSTRUMENT, COVERAGE-IMPOSSIBLE — before adding a row leaves an
        # empty file, and pandas gives an empty frame no columns at all, so the drift assert fires
        # with `[]`. The column-order check is about a schema that CHANGED; an empty file has no
        # schema to change. Guarded in ALL THREE copies of this method: fixing only darwin_core's
        # left mercator_core's to abort the same pipeline one run later.
        if df.empty:
            return {"n_rows": 0, "path": str(self.path), "parquet": None,
                    "note": "atlas empty: the arm recorded an outcome before adding any row"}
        assert list(df.columns) == list(ATLAS_COLUMNS), \
            f"parquet column order drift: {list(df.columns)}"
        # object-dtype the free-text columns so an all-null column does not become float64 and break
        # a later concat across arms.
        for c in ("model_id", "model_sha", "type", "bank_id", "vector_ref", "corpus_id",
                  "item_id", "token_class", "flags", "run_id", "commit_sha", "tag", "arm",
                  "statistic"):
            df[c] = df[c].astype("object").where(df[c].notna(), None)
        p = self.out / "atlas.parquet"
        df.to_parquet(p, index=False)
        return {"path": str(p), "n_rows": int(len(df)), "n_vectors": len(list(self.vec.glob("*.npy")))}


class Gates:
    """out/gates.json — every G-gate with pass/fail + the evidence it was judged on (§10)."""

    def __init__(self, out_dir):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "gates.json"
        self.gates = json.loads(self.path.read_text()) if self.path.exists() else {}

    def record(self, name: str, ok: bool, **evidence):
        """Record a gate WITHOUT blocking on it. `_required=False` says so in the artifact: a
        recorded-only gate reading false is a flagged observation (§3/B3's `contrast_weak` is the
        designed example), not a validity failure, and an auditor must be able to tell the two apart
        from the artifact alone.

        Bookkeeping keys are UNDERSCORE-PREFIXED and evidence may not use that prefix. Three
        separate defects in this run came from bookkeeping colliding with evidence in one flat dict
        (`ok` passed twice by the smoke runner, `n_candidates` twice by A5, and `required` — which
        several gates already use for their THRESHOLD, e.g. B3's 1.25 and G5's 10×). Reserving the
        prefix makes that class of collision impossible rather than merely fixed."""
        clash = [k for k in evidence if k.startswith("_")]
        assert not clash, f"evidence keys may not start with '_' (reserved for bookkeeping): {clash}"
        self.gates[name] = {"ok": bool(ok), "_required": False, **evidence}
        self.path.write_text(json.dumps(self.gates, indent=2, sort_keys=True))
        print(f"[gate] {name}: {'PASS' if ok else 'FAIL'}  {json.dumps(evidence, sort_keys=True)[:400]}",
              flush=True)
        return ok

    def require(self, name: str, ok: bool, **evidence):
        """G-gates are the ONLY place 'stop and debug' overrides 'pivot and continue' (§1)."""
        self.record(name, ok, **evidence)
        self.gates[name]["_required"] = True
        self.path.write_text(json.dumps(self.gates, indent=2, sort_keys=True))
        assert ok, f"G-GATE FAIL {name}: {json.dumps(evidence, sort_keys=True)[:800]}"


CACHE_ROOT = "/work"       # run_job.sh's working volume; out/ is /work/out beneath it


def cache_dir(out_dir, name: str):
    """A scratch directory for FETCHED inputs, OUTSIDE the synced out/ tree.

    run_job.sh syncs the whole of out/ to S3 every 30 s, and every arm also syncs it per unit of
    work. Anything downloaded into out/ is therefore re-uploaded: the ~1 GB bank library in every
    arm, 3.6 GB of cached fp16 scratch in A1, several GB of B1 series in B2. That is pure waste and
    it would dominate the wall-clock of the short arms.

    Anchored at CACHE_ROOT rather than `out/..` because the pipeline runner gives each step
    `--out /work/out/<step>`, so a sibling-of-out rule would land the cache back inside /work/out.
    A fixed root also means the bank library is fetched ONCE per instance and reused by every step.
    `out_dir` is unused except to document the call site."""
    import os
    import tempfile
    root = pathlib.Path(CACHE_ROOT)
    if not (root.is_dir() and os.access(root, os.W_OK)):
        root = pathlib.Path(tempfile.gettempdir())          # control plane / tests
    d = root / f"hum_cache_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def s3_sync(out_dir, s3_prefix: str):
    """Incremental per-arm sync to S3 (§10). run_job.sh also syncs, but arms that run long must not
    wait for teardown to become durable."""
    bucket = os.environ.get("PG_BUCKET", "pg-lineage-769063704696")
    subprocess.run(["aws", "s3", "sync", str(out_dir), f"s3://{bucket}/{s3_prefix}",
                    "--region", "us-east-1"], check=True)
