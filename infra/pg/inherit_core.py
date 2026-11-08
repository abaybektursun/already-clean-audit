"""INHERIT core — the pure, importable science of the origin/type/identity/content study of
S_action (studies/lineage/paligemma1-pi0/research/2026-07-02_inherit_spec.md).

Small-tensor torch/numpy math with NO model / datasets / GPU dependency, weight-free unit-tested
(infra/pg/tests/test_inherit_core.py) before any GPU minute — the §9.1 dry-run gate. The projector
/ Gram / overlap / energy-match math is REUSED from ctrliface_core (the banked object's own
instrument); this module adds only what INHERIT needs:

  * BANK LOADER  — validate + slice the frozen CTRLIFACE V-bank (bases.pt). Loading, never
    re-estimating, is what makes "the same directions" literal (the B1 lesson).
  * PG1 PRE-HOOK ADAPTER — kwarg-robust forward_pre_hooks for HF decoder layers (the residual
    ENTERING layer L, the exact point S_action[L] characterizes). transformers passes
    hidden_states positionally or as a kwarg depending on version → handle both, fail-loud.
  * MASS-MATCHED TAIL / MID-BAND builders (Arm O) — trailing / centered Gram-direction blocks
    whose captured mass matches top-k's.
  * LINEAR CKA + mismatched-layer null (transport gate) and the overlap NULL BAND with rejection
    room (Arm P-geo; the CTRLIFACE Axis-2 lesson: the margin is stated in null-band spread units).
  * sha1 CONTENT-HASH SPLIT (spec §4 — projector_ablation.split_items is rng-based, not sha1).
  * PAIRED per-item RF/gap bootstrap + one-sided bootstrap p + BH (the §7 plan).
  * U_unt DERIVATION (consensus Gram → bottom-k) + RMSNorm pullback into residual coordinates
    (S_action lives in the pre-layernorm residual; U_unt in the post-RMSNorm q/k input — the
    pullback is the frame-correct comparison; raw is the robustness leg).
  * RIDGE PROBE R² (Arm C kit) and the frozen layer-promotion rule (spec §3).

Design rules (CLAUDE.md): NO try/except hiding — a wrong assumption RAISES with a trace.
Heavy imports (torch/numpy) live inside functions so the module py_compiles on the control plane.
"""
from __future__ import annotations

import hashlib

from infra.pg.ctrliface_core import apply_mean_ablation, orthonormalize, principal_angle_overlap
from infra.pg.projector_ablation import random_basis  # noqa: F401  (re-exported for jobs)


# ===========================================================================
# Bank loader (transported-basis discipline: load, never re-estimate).
# ===========================================================================
def validate_bank(bundle, d=2048, n_layers=18, k_min=128):
    """Validate the CTRLIFACE bases.pt bundle: keys, layer coverage, shapes, orthonormality.
    Returns ({L:int → Q (d,kmax) fp64}, {L → hbar (d,)}, {L → trace float}). The bundle's hbar/trace
    are π0.5-NATIVE — callers must NEVER use them as the PG1 ablation target (spec §2: the span is
    transported; the replacement value belongs to the host)."""
    import torch
    for key in ("bases", "hbar", "trace", "d"):
        assert key in bundle, f"bank missing key {key}"
    assert int(bundle["d"]) == d, f"bank d={bundle['d']} != {d}"
    bases, hbars, traces = {}, {}, {}
    for L in range(n_layers):
        assert L in bundle["bases"], f"bank missing layer {L}"
        Q = torch.as_tensor(bundle["bases"][L], dtype=torch.float64)
        assert Q.shape[0] == d and Q.shape[1] >= k_min, f"L{L} bank shape {tuple(Q.shape)}"
        err = float((Q.transpose(0, 1) @ Q - torch.eye(Q.shape[1], dtype=torch.float64)).abs().max())
        assert err < 1e-6, f"L{L} bank not orthonormal (max |QᵀQ−I| = {err})"
        bases[L] = Q
        hbars[L] = torch.as_tensor(bundle["hbar"][L], dtype=torch.float64)
        traces[L] = float(bundle["trace"][L])
    return bases, hbars, traces


def slice_k(Q, k):
    """The first k bank columns (the frozen top-k subspace V_k). Fail-loud past coverage."""
    assert 1 <= k <= Q.shape[1], f"k={k} outside bank coverage {Q.shape[1]}"
    return Q[:, :k].contiguous()


# ===========================================================================
# PG1 pre-hook adapter (NEW; same projector contract as the core).
# ===========================================================================
def _split_hidden(args, kwargs):
    """(hidden_states, rebuild) for a decoder-layer pre-hook call — positional args[0] or the
    hidden_states kwarg. rebuild(new_hs) returns the (args, kwargs) pair with hs replaced."""
    if len(args) >= 1:
        hs = args[0]
        return hs, lambda h: ((h,) + tuple(args[1:]), kwargs)
    assert "hidden_states" in kwargs, f"no hidden_states in layer call: args={len(args)} kwargs={list(kwargs)}"
    hs = kwargs["hidden_states"]
    return hs, lambda h: (args, {**kwargs, "hidden_states": h})


def make_pg1_pre_hook_ablate(Q, hbar):
    """forward_pre_hook(with_kwargs=True) mean-ablating span(Q)→h̄ on the residual ENTERING the
    layer — out' = x − QQᵀ(x − h̄), fp32-stable (apply_mean_ablation), every position (prefill and
    each decode step; the KV-cache-consistent transport of the CTRLIFACE n_prefix=None contract)."""
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        Qd = Q.to(hs.device)
        hd = hbar.to(hs.device)
        return rebuild(apply_mean_ablation(hs, Qd, hd))
    return hook


def make_pg1_pre_hook_replace(hbar):
    """forward_pre_hook(with_kwargs=True) replacing the entering residual with h̄ (the full-read
    positive-control / floor construction at one layer)."""
    def hook(_module, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        hd = hbar.to(hs.device).to(hs.dtype)
        return rebuild(hd.expand_as(hs).contiguous())
    return hook


def make_pg1_pre_hook_capture(acc, d, x_cap=0):
    """forward_pre_hook(with_kwargs=True), capture-only: accumulate sum/sumsq/count of the entering
    residual into `acc` (for h̄ + tr(C)); bank up to x_cap raw rows into acc['X'] (for E_target).
    Chunked/capped — never the full activation set (OOM lesson)."""
    def hook(_module, args, kwargs):
        hs, _ = _split_hidden(args, kwargs)
        flat = hs.detach().to("cpu").double().reshape(-1, d)
        acc["sum"] = acc.get("sum", 0) + flat.sum(0)
        acc["sq"] = acc.get("sq", 0.0) + float((flat ** 2).sum().item())
        acc["n"] = acc.get("n", 0) + flat.shape[0]
        if x_cap and sum(x.shape[0] for x in acc.setdefault("X", [])) < x_cap:
            acc["X"].append(flat[: x_cap - sum(x.shape[0] for x in acc["X"])])
    return hook


def finalize_capture(acc):
    """(hbar fp64, trace_C float, X (n,d) fp64-or-None, n) from a capture accumulator."""
    import torch
    assert acc.get("n", 0) > 0, "capture saw no positions"
    hbar = acc["sum"] / acc["n"]
    trace_C = acc["sq"] - acc["n"] * float((hbar ** 2).sum().item())
    X = torch.cat(acc["X"], 0) if acc.get("X") else None
    return hbar, trace_C, X, acc["n"]


# ===========================================================================
# Arm O — mass-matched tail / mid-band builders (spectrum in DESCENDING order).
# ===========================================================================
def _check_spectrum(eigvals):
    import torch
    ev = torch.as_tensor(eigvals, dtype=torch.float64)
    assert ev.ndim == 1 and ev.shape[0] >= 2, f"bad spectrum shape {tuple(ev.shape)}"
    assert float(ev.min()) >= -1e-9 * max(float(ev.abs().max()), 1e-30), "negative eigval (not a PSD Gram)"
    assert bool((ev[:-1] >= ev[1:] - 1e-12 * float(ev.abs().max())).all()), "spectrum not descending"
    return ev.clamp_min(0.0)


def mass_matched_tail(eigvals, k):
    """Smallest TRAILING block [j, d) of the descending Gram spectrum whose mass ≥ the top-k mass.
    Returns (j, tail_mass, target_mass). Fail-loud if even the full complement [k, d) cannot match
    (then a mass-matched tail does not exist and the job must halt-and-document, spec §9)."""
    ev = _check_spectrum(eigvals)
    d = ev.shape[0]
    assert 1 <= k < d, f"k={k} out of range for d={d}"
    target = float(ev[:k].sum())
    csum_from = ev.flip(0).cumsum(0).flip(0)              # csum_from[j] = Σ ev[j:], decreasing in j
    assert float(csum_from[k]) >= target, (
        f"complement mass {float(csum_from[k]):.4g} < top-{k} mass {target:.4g} — no mass-matched tail")
    j = max(jj for jj in range(k, d) if float(csum_from[jj]) >= target)   # smallest matching tail
    return int(j), float(csum_from[j]), target


def mass_matched_mid(eigvals, k):
    """Contiguous mid-band [lo, hi) grown symmetrically around index d//2 until its mass ≥ the
    top-k mass. Returns (lo, hi, band_mass, target_mass). Fail-loud on infeasible (band would
    swallow the top-k block)."""
    ev = _check_spectrum(eigvals)
    d = ev.shape[0]
    assert 1 <= k < d // 2, f"k={k} too large for a mid-band in d={d}"
    target = float(ev[:k].sum())
    lo = hi = d // 2
    while float(ev[lo:hi].sum()) < target:
        if hi < d:
            hi += 1
        if float(ev[lo:hi].sum()) >= target:
            break
        assert lo - 1 > k, f"mid-band reached the top-{k} block before matching mass {target:.4g}"
        lo -= 1
    return int(lo), int(hi), float(ev[lo:hi].sum()), target


# ===========================================================================
# Transport gate — linear CKA + mismatched-layer null calibration.
# ===========================================================================
def linear_cka(X, Y):
    """Linear CKA between paired feature matrices X (n,dx), Y (n,dy) — rows are the SAME n items
    (alignment key enforced by the caller). Column-centered; 1 = same geometry up to rotation/scale."""
    import torch
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    assert X.shape[0] == Y.shape[0] and X.shape[0] >= 3, f"CKA needs paired rows: {X.shape} vs {Y.shape}"
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    num = float((Xc.transpose(0, 1) @ Yc).pow(2).sum())
    den = float((Xc.transpose(0, 1) @ Xc).norm() * (Yc.transpose(0, 1) @ Yc).norm())
    assert den > 0, "degenerate (constant) features in CKA"
    return num / den


def alignment_bound(null_values, q=0.95):
    """The frozen transport-alignment bound: the q-quantile of the MISMATCHED-LAYER CKA null
    (CKA(PG1 L, π0.5 L′≠L) over all off-diagonal pairs). Matched-layer CKA must exceed it."""
    import numpy as np
    v = np.asarray(list(null_values), dtype=float)
    assert v.size >= 10, f"null too small to calibrate a bound (n={v.size})"
    return float(np.quantile(v, q))


# ===========================================================================
# Arm P-geo — overlap null band WITH rejection room (margin in spread units).
# ===========================================================================
def overlap_null_band(ref, d, kB, n_draws, seed):
    """Null band of principal-angle overlap between the FIXED orthonormal ref (d,kA) and random
    kB-subspaces. Returns {mean, p95, spread(std), draws}. The frozen §6 P-margin: separation iff
    overlap ≥ p95 + spread (one spread-width past the p95 — rejection room by construction)."""
    import numpy as np
    from infra.pg.ctrliface_core import matched_random_overlap
    draws = matched_random_overlap(d, ref.shape[1], kB, n_draws, seed, ref=ref).numpy()
    return {"mean": float(np.mean(draws)), "p95": float(np.quantile(draws, 0.95)),
            "spread": float(np.std(draws)), "draws": [float(x) for x in draws]}


def beyond_band(overlap, band):
    """The frozen Arm-P geometric predicate: overlap beyond the calibrated null band by the margin."""
    return bool(float(overlap) >= band["p95"] + band["spread"])


# ===========================================================================
# sha1 content-hash split (spec §4; item-content-stable, unlike split_items' rng).
# ===========================================================================
def sha1_split(content_hashes, name="inherit"):
    """held[i] = True (SEALED confirmation) iff sha1(content_hash_i) is odd. Deterministic in item
    CONTENT (stable under re-ordering/re-streaming). Routed through enforce_split (fail-loud on an
    empty half). Returns the enforce_split dict."""
    import numpy as np
    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(h.encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


# ===========================================================================
# §7 statistics — paired per-item RF/gap bootstrap, bootstrap p, BH.
# ===========================================================================
def rf_point(mean_ablate, mean_intact, floor):
    from infra.pg.ctrliface_core import rf_control
    return rf_control(mean_ablate, mean_intact, floor)


def rf_item_bootstrap(per_item_ablate, per_item_intact, floor, n_boot=2000, seed=1337):
    """Per-item PAIRED bootstrap 95% CI on RF — the same resampled item indices drive the ablate
    and intact means each draw (items are the SAME eval set across arms; alignment = content hash).
    Returns {rf, ci, mean_ablate, mean_intact, n}."""
    import numpy as np
    a = np.asarray(per_item_ablate, float)
    i = np.asarray(per_item_intact, float)
    assert a.shape == i.shape and a.ndim == 1 and a.size > 1, f"paired arrays required: {a.shape} vs {i.shape}"
    point = rf_point(float(a.mean()), float(i.mean()), floor)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    denom = i[idx].mean(1) - floor
    boots = (a[idx].mean(1) - floor) / denom
    boots = boots[np.isfinite(boots) & (denom > 1e-9)]
    return {"rf": point, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "mean_ablate": float(a.mean()), "mean_intact": float(i.mean()), "n": int(a.size)}


def gap_item_bootstrap(per_item_saction, per_item_null, per_item_intact, floor,
                       n_boot=2000, seed=1337):
    """The Arm-G decisive statistic: paired per-item bootstrap of gap = RF(null) − RF(S_action).
    SAME resampled items drive all three means each draw. Returns {gap, ci, ci_lo, boots}."""
    import numpy as np
    s = np.asarray(per_item_saction, float)
    r = np.asarray(per_item_null, float)
    i = np.asarray(per_item_intact, float)
    assert s.shape == r.shape == i.shape and s.ndim == 1, "paired per-item arrays required"
    point = (rf_point(float(r.mean()), float(i.mean()), floor)
             - rf_point(float(s.mean()), float(i.mean()), floor))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(n_boot, s.size))
    denom = i[idx].mean(1) - floor
    ok = denom > 1e-9
    boots = ((r[idx].mean(1) - floor) / denom - (s[idx].mean(1) - floor) / denom)[ok]
    return {"gap": point, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "ci_lo": float(np.percentile(boots, 2.5)), "boots": boots.tolist()}


def boot_p_gt(boots, thresh):
    """One-sided bootstrap p for 'statistic > thresh' (BH input): the fraction of draws ≤ thresh,
    add-one smoothed so p is never exactly 0."""
    import numpy as np
    b = np.asarray(boots, float)
    assert b.size > 0
    return float((np.sum(b <= float(thresh)) + 1) / (b.size + 1))


def bh_adjust(pvals):
    """Benjamini–Hochberg adjusted p-values (monotone step-up), preserving input order."""
    import numpy as np
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 1.0
    for rank_from_end, oi in enumerate(order[::-1]):
        rank = m - rank_from_end
        running = min(running, p[oi] * m / rank)
        adj[oi] = running
    return adj.tolist()


def choose_n_items(sd_intact, dynamic_range, frac=0.05, n_min=2048, n_max=8192):
    """The frozen §4 power rule, made mechanical: the smallest n with a 95% CI half-width of the
    intact mean (1.96·sd/√n) below frac·(intact − floor), clamped to [n_min, n_max]. Inputs come
    from the Gate-0 feasibility probe on the discovery half. Fail-loud on no dynamic range."""
    import math
    assert dynamic_range > 1e-6, f"no dynamic range ({dynamic_range}) — P7-analog halt"
    n = math.ceil((1.96 * float(sd_intact) / (frac * float(dynamic_range))) ** 2)
    return int(min(max(n, n_min), n_max))


# ===========================================================================
# Arm P — U_unt derivation (consensus Gram → bottom-k) + RMSNorm pullback.
# ===========================================================================
def consensus_gram(deltas):
    """G = Σ_i ΔᵢᵀΔᵢ / ‖Δᵢ‖²_F (fp64) over per-fine-tune ΔW matrices — the invariant_core rule,
    verbatim math, applied to the banked scratch ΔWs. Fail-loud on a zero ΔW."""
    import torch
    G = None
    for Dl in deltas:
        D = torch.as_tensor(Dl, dtype=torch.float64)
        assert D.ndim == 2, f"ΔW must be 2-D, got {tuple(D.shape)}"
        fro2 = float((D ** 2).sum())
        assert fro2 > 0, "zero ΔW in consensus Gram"
        term = D.transpose(0, 1) @ D / fro2
        G = term if G is None else G + term
    assert G is not None, "no ΔWs"
    return G


def bottom_eigvecs(G, k):
    """The k SMALLEST-eigenvalue eigvecs of symmetric G (the consensus-untouched band U_unt).
    Returns (U (d,k) fp64 orthonormal, eigvals ascending (k,))."""
    import torch
    G = torch.as_tensor(G, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh((G + G.transpose(0, 1)) / 2.0)
    return evecs[:, :k].contiguous(), evals[:k].contiguous()


def rmsnorm_pullback(U, gamma):
    """Pull a q/k-INPUT-space (post-RMSNorm) basis U (d,k) back into residual coordinates:
    Gemma RMSNorm is y = (x/rms(x)) ⊙ (1+γ), so the y-direction u corresponds (up to the per-sample
    rms scalar, which preserves spans) to the x-direction diag(1/(1+γ))·u. Orthonormalized. This is
    the frame-correct side for overlap(S_action, U_unt); the raw overlap is the robustness leg."""
    import torch
    U = torch.as_tensor(U, dtype=torch.float64)
    g = torch.as_tensor(gamma, dtype=torch.float64)
    assert g.shape == (U.shape[0],), f"gamma shape {tuple(g.shape)} != d {U.shape[0]}"
    scale = 1.0 + g
    assert float(scale.abs().min()) > 1e-6, "degenerate RMSNorm weight (1+γ ≈ 0)"
    return orthonormalize(U / scale.unsqueeze(1))


# ===========================================================================
# Arm C — probe kit (closed-form ridge, held-out R²).
# ===========================================================================
def ridge_r2(Z_train, y_train, Z_test, y_test, alpha=1.0):
    """Held-out R² of a ridge probe y ≈ Z w (+ intercept via centering). Multi-target: returns
    (mean R² over targets, per-target list). Pure fp64; the caller owns the episode-level split."""
    import torch
    Ztr = torch.as_tensor(Z_train, dtype=torch.float64)
    Zte = torch.as_tensor(Z_test, dtype=torch.float64)
    ytr = torch.as_tensor(y_train, dtype=torch.float64)
    yte = torch.as_tensor(y_test, dtype=torch.float64)
    if ytr.ndim == 1:
        ytr, yte = ytr.unsqueeze(1), yte.unsqueeze(1)
    assert Ztr.shape[0] == ytr.shape[0] and Zte.shape[0] == yte.shape[0]
    zm, ym = Ztr.mean(0), ytr.mean(0)
    A = Ztr - zm
    W = torch.linalg.solve(A.transpose(0, 1) @ A + alpha * torch.eye(A.shape[1], dtype=torch.float64),
                           A.transpose(0, 1) @ (ytr - ym))
    pred = (Zte - zm) @ W + ym
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
    assert bool((ss_tot > 1e-12).all()), "constant probe target on the held-out split"
    r2 = (1.0 - ss_res / ss_tot)
    return float(r2.mean()), [float(x) for x in r2]


# ===========================================================================
# Frozen layer-promotion rule (spec §3, values read from the CTRLIFACE artifact at freeze).
# ===========================================================================
def promote_layers(rf_by_L):
    """{L: point RF at the profile's k} → {primary, secondary, anchors:[a1,a2]}: primary/secondary
    = the two LOWEST-RF (strongest-bite) layers; anchors = the two smallest |RF−1| among the rest.
    Ties broken toward lower L (deterministic)."""
    prof = sorted(((int(L), float(rf)) for L, rf in rf_by_L.items()), key=lambda t: (t[1], t[0]))
    assert len(prof) >= 4, f"need ≥4 profiled layers, got {len(prof)}"
    primary, secondary = prof[0][0], prof[1][0]
    rest = [(L, rf) for L, rf in prof if L not in (primary, secondary)]
    anchors = sorted(rest, key=lambda t: (abs(t[1] - 1.0), t[0]))[:2]
    return {"primary": primary, "secondary": secondary, "anchors": sorted(a[0] for a in anchors)}


def principal_overlap(A, B):
    """Convenience re-export: orthonormality-checked principal-angle overlap (Arm P-geo)."""
    return principal_angle_overlap(A, B)
