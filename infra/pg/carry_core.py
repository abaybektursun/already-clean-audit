"""CARRY core — the stiff/flat split of fine-tuning's write, its nulls, and every §6
adjudicator (carry spec §2/§3/§6; journal E2 amendments A1–A6).

The object: Q = the SPECTER gated spectral completion of F̄ at (L0, dc, ppl) — the ONLY
validated stiffness metric (fidelity ρ = 0.927; the K-FAC Σ_y⊗F̄ pullback FAILED at ρ = 0.103
and is not used anywhere here). Q is materialized densely (2048², fp64) from the banked
ingredients exactly as `specter_k.completion_env` builds them, then eigendecomposed:
stiff = top eigvecs holding STIFF_TRACE_FRAC of tr(Q); flat = complement. ΔW splits via
`comwrite_surgery.project_delta` on the residual-facing side (comwrite_core.RESIDUAL_SIDE).

Everything here is weight-free math — D-suite-tested on synthetic operators before any GPU
minute. Heavy imports inside functions so the module py_compiles on the control plane.
No try/except hiding — a wrong assumption RAISES (CLAUDE.md).

Frozen conventions (spec §2/§6/§7 + amendment A5):
  * gain_frac_X = (score_X − score_base) / (score_full − score_base), per FT, same items.
  * Δcarry = median over scored FTs of (gain_frac_flat − gain_frac_stiff).
  * range guard: full-write gain ≥ RANGE_MULT × its paired-item bootstrap SE (the
    construction-matched noise of THIS gain instrument: same items, same readout — A5).
  * random-split null: matched-rank random subsets of Q's OWN eigenvectors, 3 seeds;
    per seed BOTH parts (rank k_stiff and its complement) are installed and read.
  * all-tied rank statistics := 0 (the SPECTER E16 successor rule, §0); a bootstrap over
    < 3 valid FTs returns VOID, not a CI.
"""
from __future__ import annotations

import math

# ---- frozen constants (spec §2/§6/§7; A5) --------------------------------------------------
STIFF_TRACE_FRAC = 0.5          # stiff subspace = top eigvecs holding 50% of tr(Q)
RANGE_MULT = 3.0                # full-write gain ≥ 3× its construction-matched noise
N_RAND_SEEDS = 3                # random-split null seeds per FT
N_BOOT = 10_000                 # FT-level pair bootstrap draws
DCARRY_THRESH = 0.2             # |Δcarry| threshold for the directional branches
GAINFRAC_HI = 0.6               # single-component "reproduces the gain" bar
GAINFRAC_BAND = (0.3, 0.7)      # H_distributed middle band
VOID_MAX_SEALED = 9             # > 9 of 18 sealed FTs VOID ⇒ VOID-range verdict
N_DISCOVERY_FTS = 6
RP_SINK_LO, RP_SINK_HI = 0.3, 0.7        # R-P sink_frac rule lines
RP_CONTENT_MIN, RP_SINK_MIN = 6, 4       # ≥6/8 content-clean vs ≥4 sink-loaded
RE_RHO_HI, RE_RHO_LO = 0.8, 0.5          # R-E unification thresholds
RE_SHUFFLE_MAX = 0.3
RE_SPAN_MIN = 10.0                       # damage span < 10× ⇒ VOID-no-range
RN_QK_VERIFIED, RN_QK_LEAK = 0.1, 0.5    # R-N(c) sink-head QK-curvature thresholds
RN_SINK_ATTN = 0.9                       # a "sink head" attends to sink ≥ 0.9
RN_ISO_TOL = 2.0                         # both diagnostics within 2× of reference
RW2_MEANSUB_MULT, RW2_ZERO_MULT = 2.0, 10.0
RD2_EPS_GRID = (0.6, 0.7, 0.8, 0.9)
RD2_TOPK = 10                            # top-10 post-FT singular vectors censused
KAPPA_GRID_UNITS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)   # ×√D; ∞ (pure witness) appended


# ===========================================================================
# Q — materialize the completion densely, split it, and draw matched-rank nulls.
# ===========================================================================
def materialize_q(spikes_U, spikes_lam, band_bulk, basis, sb_slices):
    """The dense (d,d) fp64 matrix of `specter_core.spectral_completion_quadform`:
    Q = UΛUᵀ + (I−UUᵀ)·B·diag(bulk)·Bᵀ·(I−UUᵀ). The D-suite asserts vᵀQv equals the
    closed-form functional on random probes to fp tolerance."""
    import torch
    B = torch.as_tensor(basis, dtype=torch.float64)
    d = B.shape[0]
    assert B.shape == (d, d), f"basis must be square, got {tuple(B.shape)}"
    bulk = torch.zeros(d, dtype=torch.float64)
    for b, (lo, hi) in enumerate(sb_slices):
        bulk[lo:hi] = float(band_bulk[b])
    M = B @ torch.diag(bulk) @ B.transpose(0, 1)
    if spikes_U is not None and torch.as_tensor(spikes_U).numel():
        U = torch.as_tensor(spikes_U, dtype=torch.float64)
        lam = torch.as_tensor(spikes_lam, dtype=torch.float64)
        assert U.shape[0] == d and U.shape[1] == lam.numel()
        P = torch.eye(d, dtype=torch.float64) - U @ U.transpose(0, 1)
        M = P @ M @ P + U @ torch.diag(lam) @ U.transpose(0, 1)
    return 0.5 * (M + M.transpose(0, 1))


def stiff_split(Q, frac=STIFF_TRACE_FRAC):
    """Eigendecompose Q (desc) and return (evals, evecs, k_stiff): k_stiff = the SMALLEST k
    whose top-k eigenvalue mass reaches ≥ frac of tr(Q). U_stiff = evecs[:, :k_stiff]."""
    import torch
    Q = torch.as_tensor(Q, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(Q)
    evals, evecs = evals.flip(0).clamp_min(0.0), evecs.flip(1)
    tr = float(evals.sum())
    assert tr > 0, "degenerate Q: zero trace"
    cum = torch.cumsum(evals, 0)
    k = int(torch.searchsorted(cum, frac * tr).item()) + 1
    assert 1 <= k < evals.numel(), f"degenerate split k={k} of d={evals.numel()}"
    return evals, evecs, k


def random_eigsplit(evecs, k, seed):
    """Matched-rank random eigenspace split of the SAME Q: a uniformly random k-subset of
    Q's OWN eigenvectors (partner = complement). Destroys the curvature ordering, preserves
    'spanned by Q eigvecs' and the rank. Returns (U_a (d,k), idx_a sorted)."""
    import numpy as np
    import torch
    evecs = torch.as_tensor(evecs)
    d = evecs.shape[1]
    assert 1 <= k < d
    idx = np.sort(np.random.default_rng(int(seed)).choice(d, size=k, replace=False))
    return evecs[:, idx.tolist()].contiguous(), idx.tolist()


# ===========================================================================
# Gains, gain fractions, the range guard, Δcarry, and the FT-level bootstrap.
# ===========================================================================
def paired_gain(per_item_cell, per_item_base, n_boot=2000, seed=1337):
    """Per-item-task gain = mean(cell − base) paired by item, with bootstrap SE.
    Returns {gain, se}. The SE is the construction-matched noise of the gain instrument
    (same items, same readout — A5); the range guard reads it."""
    import numpy as np
    a = np.asarray(per_item_cell, float)
    b = np.asarray(per_item_base, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) >= 10, "unpaired or tiny item sets"
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return {"gain": float(diff.mean()), "se": float(boots.std(ddof=1))}


def cider_paired_gain(cand_cell, cand_base, refs, n_boot=300, seed=1337):
    """Corpus (CIDEr-D) gain with a PAIRED item bootstrap: one index resample scores BOTH
    candidate sets, then differences — the frozen corpus-task treatment (A5). Returns
    {gain, se, score_cell, score_base}."""
    import numpy as np
    from pycocoevalcap.cider.cider import Cider
    ids = sorted(cand_cell)
    assert set(ids) == set(cand_base) == set(refs), "candidate/reference id mismatch"

    def _score(cand, keys):
        c = {j: cand[k] for j, k in enumerate(keys)}
        r = {j: refs[k] for j, k in enumerate(keys)}
        s, _ = Cider().compute_score(r, c)
        return float(s)
    s_cell, s_base = _score(cand_cell, ids), _score(cand_base, ids)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        keys = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        boots.append(_score(cand_cell, keys) - _score(cand_base, keys))
    boots = np.asarray(boots)
    return {"gain": s_cell - s_base, "se": float(boots.std(ddof=1)),
            "score_cell": s_cell, "score_base": s_base}


def range_guard_ok(gain_full, se_full, mult=RANGE_MULT):
    """True iff the full-write gain clears mult× its own construction-matched noise.
    Two-sided ceiling per the one-sided-thresholds law: a NEGATIVE full gain of any size
    fails (a write that hurts its assigned task has no carrying to decompose)."""
    assert se_full >= 0
    return float(gain_full) >= mult * float(se_full) and float(gain_full) > 0


def gain_frac(gain_cell, gain_full):
    """(gain of the component) / (gain of the full write). Caller guarantees the range
    guard passed, so gain_full > 0."""
    assert gain_full > 0, "gain_frac on a guard-failing FT — caller bug"
    return float(gain_cell) / float(gain_full)


def dcarry_bootstrap(gf_flat, gf_stiff, n_boot=N_BOOT, seed=1337):
    """Δcarry = median over FTs of (gf_flat − gf_stiff) with an FT-level pair bootstrap.
    < 3 valid FTs ⇒ VOID (the frozen degenerate-input completion, §2)."""
    import numpy as np
    a = np.asarray(gf_flat, float)
    b = np.asarray(gf_stiff, float)
    assert a.shape == b.shape and a.ndim == 1
    if len(a) < 3:
        return {"void": True, "n": int(len(a)),
                "reading": "bootstrap over < 3 valid FTs returns VOID, not a CI (§2)"}
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = np.median(diff[idx], axis=1)
    return {"void": False, "dcarry": float(np.median(diff)), "n": int(len(diff)),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "boots_sd": float(boots.std(ddof=1))}


# ===========================================================================
# Rank statistics with the frozen degenerate-input completions (E16 rule, §0).
# ===========================================================================
def _all_tied(v):
    import numpy as np
    v = np.asarray(v, float)
    return bool(np.all(v == v[0]))


def spearman0(a, b):
    """Spearman ρ with the frozen completion ρ(any all-tied input) := 0."""
    if _all_tied(a) or _all_tied(b):
        return 0.0
    from infra.pg.atlas_fit import spearman
    return spearman(a, b)


def kendall0(a, b):
    """Kendall τ-b with the frozen completion τ(any all-tied input) := 0."""
    if _all_tied(a) or _all_tied(b):
        return 0.0
    from infra.pg.atlas_fit import kendall_tau_b
    return kendall_tau_b(a, b)


def shuffle_null_rho(pred, meas, n_shuffle=10_000, seed=1337):
    """Label-shuffle null for ρ_S(pred, meas): permute the pairing, keep marginals.
    Returns {p95, p_value (one-sided ≥ observed), rho_obs}."""
    import numpy as np
    rho_obs = spearman0(pred, meas)
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, float)
    draws = np.empty(n_shuffle)
    for i in range(n_shuffle):
        draws[i] = spearman0(pred[rng.permutation(len(pred))], meas)
    p = float(((draws >= rho_obs).sum() + 1) / (n_shuffle + 1))     # (c+1)/(n+1), never 0
    return {"rho_obs": rho_obs, "p95": float(np.quantile(draws, 0.95)), "p_value": p}


# ===========================================================================
# R-C — the primary adjudicator (guards FIRST; every branch constructible).
# ===========================================================================
def adjudicate_rc(rows, *, n_sealed_total, boot_seed=1337):
    """`rows` = per-SCORED-FT dicts {ft, gf_stiff, gf_flat, band_stiff_max, band_flat_max}
    (band_*_max = the max gain_frac over that side's 3 random-split parts). FTs that failed
    the range guard are NOT in `rows`; n_void = n_sealed_total − len(rows).

    Branch order (frozen): VOID-range → H_flat-carries → H_stiff-carries → H_distributed.
    Ceilings + VOID per the signals-that-cannot-fail law: the VOID branch is reachable
    (guard starvation), both directional branches are killable (CI side + band), and the
    middle branch is the honest remainder — no branch is a floor without a ceiling."""
    import numpy as np
    n_void = int(n_sealed_total) - len(rows)
    assert n_void >= 0, "more scored rows than sealed FTs"
    if n_void > VOID_MAX_SEALED:
        return {"verdict": "VOID-range", "n_void": n_void, "n_scored": len(rows),
                "reading": "the swap-shell's range on this corpus is the finding (a "
                           "COMWRITE-lesson extension), reported with the per-FT table"}
    gf_s = [r["gf_stiff"] for r in rows]
    gf_f = [r["gf_flat"] for r in rows]
    boot = dcarry_bootstrap(gf_f, gf_s, seed=boot_seed)
    if boot["void"]:
        return {"verdict": "VOID-range", "n_void": n_void, "n_scored": len(rows),
                "boot": boot, "reading": "fewer than 3 guard-passing FTs — swap-shell range "
                                         "is the finding"}
    med_f, med_s = float(np.median(gf_f)), float(np.median(gf_s))
    beats_f = float(np.median([r["gf_flat"] - r["band_flat_max"] for r in rows])) > 0
    beats_s = float(np.median([r["gf_stiff"] - r["band_stiff_max"] for r in rows])) > 0
    out = {"n_void": n_void, "n_scored": len(rows), "dcarry": boot["dcarry"],
           "ci95": boot["ci95"], "median_gf_flat": med_f, "median_gf_stiff": med_s,
           "flat_beats_band": beats_f, "stiff_beats_band": beats_s}
    if boot["dcarry"] >= DCARRY_THRESH and boot["ci95"][0] > 0 and \
            med_f >= GAINFRAC_HI and beats_f:
        return {**out, "verdict": "H_flat-carries",
                "reading": "the write's stiff mass is functionally light — S10's lesson "
                           "holds for accumulated writes; protect-the-loud is licensed as "
                           "a transfer rule to test"}
    if boot["dcarry"] <= -DCARRY_THRESH and boot["ci95"][1] < 0 and \
            med_s >= GAINFRAC_HI and beats_s:
        return {**out, "verdict": "H_stiff-carries",
                "reading": "accumulated fine-tuning geometry departs from per-step SGD "
                           "geometry — a citable divergence from S10's regime; the "
                           "lazy-dial successor is promoted"}
    both_inside = (not beats_f) and (not beats_s)
    both_mid = (GAINFRAC_BAND[0] <= med_s <= GAINFRAC_BAND[1]
                and GAINFRAC_BAND[0] <= med_f <= GAINFRAC_BAND[1]
                and boot["ci95"][0] <= 0 <= boot["ci95"][1])
    if both_inside or both_mid:
        return {**out, "verdict": "H_distributed",
                "reading": "the stiff/flat axis is not the functional axis — the split "
                           "instrument is scoped; the per-FT residual decomposition is "
                           "banked and the functional axis is the successor's object"}
    return {**out, "verdict": "H_partial",
            "reading": "a mixed landing outside every frozen branch's full condition — "
                       "reported with the per-FT table; the closest branch's missing "
                       "clause is named in the result doc"}


# ===========================================================================
# R-P — per-spike sink-position stratification (A3).
# ===========================================================================
def sink_frac(q_sink, q_total):
    assert q_total > 0, "degenerate spike: zero total Rayleigh mass"
    return float(q_sink) / float(q_total)


def adjudicate_rp(fracs):
    """`fracs` = per-promoted-spike sink_frac list (len 8 expected; shorter is disclosed).
    All three branches are productive; none is a loss (§6 R-P)."""
    n = len(fracs)
    assert n >= 1, "no spikes to stratify"
    n_content = sum(1 for f in fracs if f <= RP_SINK_LO)
    n_sink = sum(1 for f in fracs if f >= RP_SINK_HI)
    out = {"n_spikes": n, "n_content": n_content, "n_sink_loaded": n_sink,
           "sink_fracs": [float(f) for f in fracs]}
    if n_content >= RP_CONTENT_MIN:
        return {**out, "verdict": "H_spikes-content",
                "reading": "SPECTER's spike claims stand at current scope"}
    if n_sink >= RP_SINK_MIN:
        return {**out, "verdict": "H_spikes-sink-loaded",
                "reading": "the claims are RE-SCOPED to sink-position curvature — a scoping "
                           "correction disclosed in the registry and the artifact page; the "
                           "gradient-sink literature (S21) gains a live confirmation"}
    return {**out, "verdict": "H_mixed", "reading": "the per-spike map is the finding"}


# ===========================================================================
# R-N — theorem anchors: Ford bound, isoperimetry diagnostics, sink-head QK, GGN gap.
# ===========================================================================
def ford_bound(sigma, mu):
    """ε*_q(E) ≤ −σ·Φ⁻¹(μ) (S8): median distance to the error set implied by a noise
    error-rate μ at scale σ. μ ∈ (0,1) strictly — a 0 or 1 rate has no finite bound."""
    from scipy.stats import norm
    assert 0.0 < mu < 1.0, f"Ford bound undefined at mu={mu}"
    return float(-sigma * norm.ppf(mu))


def thin_shell_cv(norms):
    """CV of the norm distribution vs the Gaussian thin-shell reference CV ≈ 1/√(2d) is
    computed by the CALLER (needs d); this returns the raw CV."""
    import numpy as np
    v = np.asarray(norms, float)
    assert len(v) >= 1000, "thin-shell diagnostic needs ≥ 1000 rows"
    assert v.mean() > 0
    return float(v.std(ddof=1) / v.mean())


def lipschitz_tail_chat(X, n_func=64, seed=1337, min_tail=50):
    """Isoperimetry diagnostic (b): the empirical concentration constant ĉ such that the
    worst observed tail of 64 random 1-Lipschitz linear functionals matches
    P[|f−Ef| ≥ t] = 2exp(−d t²/(2ĉ)) (S13's c-isoperimetry with L=1). The definition's
    d·t² exponent presumes UNIT-scale covariates — rows are rescaled so E‖x‖² = 1
    (Bubeck–Sellke normalization; a √d-scale normalization here would be wrong by a
    factor of d). Gaussian data ⇒ ĉ ≲ 1; heavy-tailed (scale-mixture) data ⇒ ĉ ≫ 1.
    Tail read at the largest t with ≥ min_tail exceedances (finite-sample floor)."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    assert n >= 10_000, "tail diagnostic needs ≥ 10k rows"
    scale = math.sqrt(1.0 / float((X ** 2).sum(axis=1).mean()))
    Xs = X * scale
    rng = np.random.default_rng(seed)
    chats = []
    for _ in range(n_func):
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        f = Xs @ u
        dev = np.abs(f - f.mean())
        t = np.quantile(dev, 1.0 - min_tail / n)
        p = float((dev >= t).mean())
        assert 0 < p < 1
        chats.append(d * t * t / (2.0 * abs(math.log(max(p / 2.0, 1e-300)))))
    return {"c_hat": float(np.max(chats)), "c_hat_median": float(np.median(chats)),
            "scale": scale, "n_func": n_func}


def adjudicate_rn_b(*, cv, cv_ref, c_hat, tol=RN_ISO_TOL):
    """R-N(b): H_isoperimetric-enough iff BOTH diagnostics within tol× of the Gaussian
    reference (cv_ref = 1/√(2d); c reference = 1)."""
    assert cv_ref > 0
    cv_ratio = float(cv) / float(cv_ref)
    out = {"cv": float(cv), "cv_ratio": cv_ratio, "c_hat": float(c_hat)}
    if cv_ratio <= tol and c_hat <= tol:
        return {**out, "verdict": "H_isoperimetric-enough",
                "reading": "the √d line binds; the witnesses' excess is theorem-anchored"}
    return {**out, "verdict": "H_non-isoperimetric",
            "reading": "the √d line is reported not-binding; PG1's activation geometry "
                       "violating the premise is itself a named finding that bounds S13's "
                       "applicability to transformer activations"}


def adjudicate_rn_c(head_rows):
    """R-N(c): `head_rows` = per-head {attn_sink_frac, qk_curv}. Sink heads (attn ≥ 0.9)
    against the median head's QK curvature. Fail-loud instrument event vs bounded-theory
    finding adjudicated by magnitude (§5 'theorem checks designed to fail')."""
    import numpy as np
    curvs = np.asarray([r["qk_curv"] for r in head_rows], float)
    med = float(np.median(curvs))
    assert med > 0, "degenerate head census: zero median QK curvature"
    sink = [r for r in head_rows if r["attn_sink_frac"] >= RN_SINK_ATTN]
    out = {"n_heads": len(head_rows), "n_sink_heads": len(sink), "median_qk_curv": med}
    if not sink:
        return {**out, "verdict": "VOID-no-sink-heads",
                "reading": "no head meets the attention-telemetry sink criterion — the "
                           "S23 prediction has no test bed on this batch; the census ships"}
    ratios = [r["qk_curv"] / med for r in sink]
    worst = max(ratios)
    out["sink_head_ratios"] = [float(x) for x in ratios]
    if worst <= RN_QK_VERIFIED:
        return {**out, "verdict": "H_theorem-verified"}
    if worst < RN_QK_LEAK:
        return {**out, "verdict": "H_measured-leak",
                "reading": "quantified leak between S23's collapse prediction and this "
                           "architecture's sink heads"}
    return {**out, "verdict": "H_s23-bounded",
            "reading": "S23's one-hot collapse does not govern this architecture — a bound "
                       "on the theory, reported with the head census"}


def adjudicate_rn_d(*, top_ggn, top_hess):
    """R-N(d): descriptive — top-eigenpair GGN/Hessian ratio beside the banked trace gap."""
    assert top_hess != 0
    return {"verdict": "descriptive", "top_ggn": float(top_ggn), "top_hess": float(top_hess),
            "ratio": float(top_ggn) / float(top_hess)}


# ===========================================================================
# R-E — error-family unification.
# ===========================================================================
def lp_growth_exponent(amps, damages):
    """The Lᵖ growth exponent (S3): OLS slope of log(damage) on log(amplitude). Degenerate
    inputs (any non-positive damage, < 3 points) return None — a family whose damage does
    not register at every rung has no exponent, and that is reported, not imputed."""
    import numpy as np
    a = np.asarray(amps, float)
    d = np.asarray(damages, float)
    if len(a) < 3 or (d <= 0).any() or (a <= 0).any():
        return None
    x, y = np.log(a), np.log(d)
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def adjudicate_re(*, pred, meas, labels, n_shuffle=10_000, seed=1337):
    """R-E: guard FIRST (span < 10× ⇒ VOID-no-range), then ρ_S(pred, meas) vs the
    label-shuffle null."""
    import numpy as np
    meas_arr = np.asarray(meas, float)
    pos = meas_arr[meas_arr > 0]
    span = float(pos.max() / pos.min()) if len(pos) >= 2 and pos.min() > 0 else 0.0
    out = {"labels": list(labels), "span_ratio": span}
    if span < RE_SPAN_MIN:
        return {**out, "verdict": "VOID-no-range",
                "reading": "measured damages span < 10× — the battery cannot rank; the "
                           "amplitude choice is the finding"}
    sh = shuffle_null_rho(pred, meas, n_shuffle=n_shuffle, seed=seed)
    out.update({"rho": sh["rho_obs"], "shuffle_p95": sh["p95"], "p_value": sh["p_value"]})
    # "shuffle ≤ 0.3" (§6 R-E) read as the label-shuffle p-value: at n≈8 families the
    # null's p95 is ~0.6, so a p95 reading could never fire — the p-value reading is the
    # only one that can both pass and fail (journal E3 interpretation note, pre-freeze).
    if sh["rho_obs"] >= RE_RHO_HI and sh["p_value"] <= RE_SHUFFLE_MAX:
        return {**out, "verdict": "H_operator-unifies",
                "reading": "four external results and two of ours land in one mechanism "
                           "figure"}
    if sh["rho_obs"] >= RE_RHO_LO:
        esc = [l for l, p, m in zip(labels, pred, meas) if _escapes(p, m, pred, meas)]
        return {**out, "verdict": "H_partial", "escaping_families": esc,
                "reading": "the escaping families are named — each is a mechanism lead"}
    return {**out, "verdict": "H_families-escape",
            "reading": "the operator does not govern structured-error benignity — the "
                       "strongest possible scoping of the SPECTER atlas; the residuals say "
                       "where to look"}


def _escapes(p, m, pred, meas):
    """A family 'escapes' iff its rank residual |rank(pred) − rank(meas)| is the largest
    tercile of the family set (pure convention for NAMING the leads; not a verdict)."""
    import numpy as np
    from infra.pg.atlas_fit import _ranks
    rp, rm = _ranks(np.asarray(pred, float)), _ranks(np.asarray(meas, float))
    resid = np.abs(rp - rm)
    i = list(pred).index(p) if p in list(pred) else None
    return resid[i] >= np.quantile(resid, 2.0 / 3.0) if i is not None else False


# ===========================================================================
# R-W2 — sink/gauge controls.
# ===========================================================================
def adjudicate_rw2_meansub(*, d_meansub, d_zero, seed_band):
    """Mean-substitution vs zeroing of the sink coordinates (S17's exact pattern):
    H_constant-bias iff |mean-sub damage| ≤ 2× band while zeroing ≥ 10× band."""
    assert seed_band > 0, "degenerate seed band"
    r_ms, r_z = abs(float(d_meansub)) / seed_band, abs(float(d_zero)) / seed_band
    out = {"meansub_over_band": r_ms, "zero_over_band": r_z}
    if r_ms <= RW2_MEANSUB_MULT and r_z >= RW2_ZERO_MULT:
        return {**out, "verdict": "H_constant-bias",
                "reading": "S17's exact pattern reproduced on PG1 — predicted-by credit "
                           "recorded"}
    return {**out, "verdict": "H_deviates",
            "reading": "PG1's sink differs from the S17 pattern — a cross-architecture "
                       "datum, reported with both magnitudes"}


def adjudicate_rw2_rotation(*, max_rel_err, tol=5e-2):
    """Rotation covariance: the recomputed banked statistic must match within fp tolerance
    under the Hadamard gauge (bf16 forward ⇒ tol is generous but FROZEN). A miss is
    fail-loud wiring (the §0.5-4 D-suite proof decides whether the (1+γ) fold or the
    harness is the suspect)."""
    out = {"max_rel_err": float(max_rel_err), "tol": tol}
    if max_rel_err <= tol:
        return {**out, "verdict": "H_instrument-certified"}
    return {**out, "verdict": "HALT-instrument",
            "reading": "the computational-invariance theorem is exact; a miss is wiring — "
                       "fail loud, fix, re-run (§0.5-4 decides (1+γ) fold vs harness)"}


# ===========================================================================
# R-D2 — intruder-dimension census (S6).
# ===========================================================================
def intruder_counts(U_ft_top, U_base, eps_grid=RD2_EPS_GRID):
    """Per ε: how many of the top post-FT singular vectors have max |cos| to EVERY base
    singular vector < ε (an 'intruder' at that threshold). U_ft_top (d,k_top);
    U_base (d,m) — the full base singular-vector set on the same side."""
    import numpy as np
    A = np.asarray(U_ft_top, dtype=np.float64)
    B = np.asarray(U_base, dtype=np.float64)
    assert A.shape[0] == B.shape[0], "side mismatch in the intruder census"
    maxcos = np.abs(B.T @ A).max(axis=0)          # (k_top,)
    assert maxcos.shape[0] == A.shape[1]
    return {f"{eps:.1f}": int((maxcos < eps).sum()) for eps in eps_grid}, \
        [float(x) for x in maxcos]


def entry_shuffle_null(U_base, k_top, seed, eps_grid=RD2_EPS_GRID):
    """The census null: per-column ENTRY permutation of the base singular vectors
    (destroys structure, keeps marginals), then the same max-cos census against k_top
    Haar probes drawn as shuffled ft-vectors stand-ins is NOT what we want — the null is
    computed by the CALLER shuffling the base and re-running `intruder_counts` on the real
    U_ft_top. This helper returns the shuffled base."""
    import numpy as np
    B = np.asarray(U_base, dtype=np.float64).copy()
    rng = np.random.default_rng(int(seed))
    for j in range(B.shape[1]):
        B[:, j] = B[rng.permutation(B.shape[0]), j]
    return B


def adjudicate_rd2(per_ft_counts, eps_grid=RD2_EPS_GRID):
    """`per_ft_counts` = list over FTs of {eps_str: count}. H_full-FT-clean iff the median
    intruder count is 0 at EVERY ε in the grid."""
    import numpy as np
    medians = {f"{e:.1f}": float(np.median([c[f"{e:.1f}"] for c in per_ft_counts]))
               for e in eps_grid}
    out = {"median_counts": medians, "n_fts": len(per_ft_counts)}
    if all(v == 0 for v in medians.values()):
        return {**out, "verdict": "H_full-FT-clean",
                "reading": "S6's full-FT claim replicates on our corpus — the "
                           "conserved-core account gains an independent spectral witness"}
    return {**out, "verdict": "H_intruders-present",
            "reading": "our 24-FT corpus departs from S6's regime — a corpus-level "
                       "finding; the intruder set is cross-referenced against U_unt's "
                       "complement and the forgetting locus"}


# ===========================================================================
# Arm B2 — the κ-bridge construction (S2's family, activation-space).
# ===========================================================================
def kappa_draw(witness, kappa, d, seed):
    """One κ-family draw: δ ∝ (z + κ·ŵ), z ~ N(0, I_d), unit-normalized (the caller sets
    the energy). E[cos θ(δ, ŵ)] = κ/√(d + κ²) (S2). kappa=None ⇒ the pure witness."""
    import torch
    w = torch.as_tensor(witness, dtype=torch.float32).reshape(-1)
    assert w.shape[0] == d
    what = w / w.norm()
    if kappa is None:
        return what
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn(d, generator=gen)
    v = z + float(kappa) * what
    return v / v.norm()


def expected_cos(kappa, d):
    return float(kappa) / math.sqrt(d + float(kappa) ** 2)


# ===========================================================================
# The (1+γ) Hadamard rotation gauge (S18 computational invariance on Gemma norms) —
# pure tensor math, D-suite-proven on a tiny block before any GPU use (A4).
# ===========================================================================
def hadamard_matrix(d):
    """Normalized Sylvester Hadamard H (d power of 2): H Hᵀ = I."""
    import torch
    assert d & (d - 1) == 0, f"Hadamard needs a power-of-2 dim, got {d}"
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def fold_gemma_norm(gamma, W_in):
    """Fold Gemma's (1+γ) RMSNorm scale into the downstream residual-READING matrix:
    y = W_in·((1+γ)⊙x̂) = (W_in·diag(1+γ))·x̂. Returns the folded W. After folding, the
    norm is scale-free (pure x/rms) and commutes with any orthogonal Q on the residual:
    rms(Qx) = rms(x) ⇒ the S18 invariance applies. The D-suite proves function
    preservation end-to-end on a tiny PG1-like block."""
    import torch
    g = torch.as_tensor(gamma, dtype=torch.float64).reshape(-1)
    W = torch.as_tensor(W_in, dtype=torch.float64)
    assert W.shape[1] == g.shape[0], "fold side mismatch: W_in must READ the normed stream"
    return W * (1.0 + g).unsqueeze(0)
