"""RATION core — the rate-distortion bit allocation, the in-basis uniform quantiser, the 7 policy
(R,w) builders, Σ_x finalisation, the oracle leave-one-out expansion, the paired-item excess
bootstrap, and `adjudicate_ration` (spec §6 frozen verdict order). Plus `make_residual_substitutor`
— the injection surface that SUBSTITUTES x̂ = R·Q_b(Rᵀx) for the entering residual (not add a delta).

The single genuinely NEW component of RATION. Everything measured (Q, Σ_x, behavior, MSE) reuses
verified surfaces; only the water-filling + per-coordinate uniform quantiser is new. Weight-free
math — D-suite-tested on synthetic tensors before any GPU minute. Heavy imports inside functions so
the module py_compiles on the control plane (no torch on the laptop). No try/except hiding — a wrong
assumption RAISES (CLAUDE.md).

Frozen conventions (spec §1.2/§3/§6; journal amendments A-RATION-1..4):
  * quantiser: per-coordinate uniform mid-rise over [center ± K_SIGMA·std] IN basis R, 2^b levels;
    b=0 collapses to the single centre level (proved in the D-suite). K_SIGMA=4.
  * water-fill: b_i = clip(round(b̄ + ½·log2(w_i/geomean(w))), 0, b_max), integer-renormalised to
    hold the average b̄ EXACTLY (largest-remainder ±1 on free coords). Tied w → uniform (E16 rule).
  * susceptibility (R,w)=(evecs,evals) read directly from carry_q_bank.pt — a full-rank ROTATION,
    not a projection, so the CONFLUX R9 circularity does not apply (A-RATION-4).
  * every null/guard two-sided with a ceiling + a VOID branch (feedback_one_sided_thresholds).
"""
from __future__ import annotations

import math

# ---- frozen constants (spec §1.2/§3/§6) ---------------------------------------------------------
K_SIGMA = 4.0                       # quantiser clip range = centre ± K_SIGMA·std, per coordinate
B_MAX = 16                          # max bits/coord (also the identity/lossless setting)
BUDGET_GRID = (2, 3, 4, 6, 8)       # average-bit budgets b̄ swept (spec §3)
MASSIVE_COORDS = (738, 674, 604, 2015)   # CONFLUX sink coords (re-detected on discovery)
SINK_EIGEN_FRAC = 0.5               # eigvec is "sink-carrying" if its sink-coord energy ≥ this
N_ORACLE_BANDS = 64                 # coarse grouping for the oracle leave-one-out (§O)
SEED = 1337
N_BOOT = 10_000
DELTA_RANGE_MULT = 3.0              # range guard: uniform loss ≥ MULT × its paired SE (carry A5 rule)

POLICIES = ("susceptibility", "variance", "sink_aware_variance", "magnitude",
            "uniform", "random", "oracle")


# ===========================================================================
# Water-filling bit allocation (the one non-trivial new piece).
# ===========================================================================
def _adjust_to_target(b, target, order_up, order_dn, b_max):
    """In-place ±1 on FREE coords (given as preference orders) until sum(b)==target. Cycles the
    order so a gap larger than d is handled; RAISES if a full cycle makes no progress (infeasible)."""
    import itertools
    total = int(b.sum())
    if total == target:
        return
    it = itertools.cycle(order_up if total < target else order_dn)
    step = 1 if total < target else -1
    progressed, seen = False, 0
    n = len(order_up)
    while total != target:
        j = int(next(it))
        seen += 1
        nv = b[j] + step
        if 0 <= nv <= b_max:
            b[j] = nv
            total += step
            progressed = True
        if seen >= n:                       # completed a full cycle
            assert progressed, "water-fill renorm infeasible (all free coords saturated)"
            progressed, seen = False, 0


def water_fill(w, b_bar, b_max=B_MAX, pinned=None):
    """Per-coordinate integer bit budget (spec §1.2). b_i = clip(round(b̄ + ½·log2(w_i/geomean w)),
    0, b_max), then integer-renormalised (largest remainder) so the mean over ALL d equals b̄ exactly.
    `pinned`: {coord: bits} forced (e.g. sink→b_max) before renorm; the free coords absorb the
    deficit. Tied importances → uniform bits (blind-safe; SPECTER E16 successor)."""
    import numpy as np
    w = np.asarray(w, dtype=np.float64)
    d = w.shape[0]
    pinned = {int(k): int(np.clip(v, 0, b_max)) for k, v in (pinned or {}).items()}
    if np.ptp(w) == 0.0 and not pinned:                 # all importances tied
        cont = np.full(d, float(b_bar))
    else:
        wp = np.clip(w, 1e-30, None)
        gm = math.exp(float(np.log(wp).mean()))          # geometric mean
        cont = float(b_bar) + 0.5 * np.log2(wp / gm)
    cont = np.clip(cont, 0.0, float(b_max))
    b = np.rint(cont).astype(np.int64)
    for c, v in pinned.items():
        b[c] = v
    free = np.setdiff1d(np.arange(d), np.array(sorted(pinned), dtype=np.int64))
    assert free.size > 0, "every coordinate pinned — nothing to renormalise"
    resid = cont[free] - b[free]
    order_up = free[np.argsort(-resid)]                  # biggest positive residual wants +1
    order_dn = free[np.argsort(resid)]                   # biggest negative residual wants −1
    _adjust_to_target(b, int(round(float(b_bar) * d)), order_up, order_dn, b_max)
    assert b.min() >= 0 and b.max() <= b_max
    return b


# ===========================================================================
# In-basis per-coordinate uniform quantiser: x̂ = R · Q_b(Rᵀx).
# ===========================================================================
def quantize_in_basis(X, R, bits, centers, stds, k_sigma=K_SIGMA):
    """X (n,d) → X̂ (n,d) = (R · Q_b(Rᵀ Xᵀ))ᵀ. Per-coordinate uniform mid-rise quantiser with
    2^bits[i] levels over [centre_i − k·std_i, centre_i + k·std_i]; bits[i]=0 collapses to centre_i.
    R (d,d) orthonormal; centers/stds (d,) are the per-coordinate stats IN basis R. dtype/device
    follow R — the D-suite passes fp64 CPU; the GPU hook passes fp32 CUDA."""
    import torch
    R = torch.as_tensor(R)
    dt, dev = R.dtype, R.device
    X = torch.as_tensor(X, dtype=dt, device=dev)
    bits = torch.as_tensor(bits, device=dev)
    centers = torch.as_tensor(centers, dtype=dt, device=dev)
    stds = torch.as_tensor(stds, dtype=dt, device=dev)
    C = X @ R                                            # (n,d) coordinates in basis R
    levels = torch.pow(torch.as_tensor(2.0, dtype=dt, device=dev), bits.to(dt)).clamp(min=1.0)
    lo = centers - k_sigma * stds
    hi = centers + k_sigma * stds
    span = (hi - lo).clamp(min=1e-20)
    step = span / levels
    Cc = torch.minimum(torch.maximum(C, lo), hi)
    q = torch.floor((Cc - lo) / step)
    q = torch.minimum(q, levels - 1.0)                  # top edge → last bin
    rec = lo + (q + 0.5) * step                         # mid-rise reconstruction (b=0 → centre)
    return rec @ R.transpose(0, 1)


def basis_stats(R, mean_x, C_x):
    """(centers, stds) per coordinate IN basis R from the banked (mean_x, C_x). centres = Rᵀ·mean_x;
    stds_i = sqrt(diag(Rᵀ C_x R)_ii). Follows R's dtype/device."""
    import torch
    R = torch.as_tensor(R)
    mean_x = torch.as_tensor(mean_x, dtype=R.dtype, device=R.device)
    C_x = torch.as_tensor(C_x, dtype=R.dtype, device=R.device)
    centers = R.transpose(0, 1) @ mean_x
    var = (R * (C_x @ R)).sum(0)                         # diag(Rᵀ C_x R) without forming the full matrix
    return centers, var.clamp(min=0).sqrt()


# ===========================================================================
# Σ_x finalisation and the eigenbasis helpers.
# ===========================================================================
def finalize_sigma(gram, sum_x, absmean_sum, n):
    """gram = Σ xxᵀ (d,d), sum_x = Σ x (d,), absmean_sum = Σ|x| (d,), n = row count. Returns
    (mean_x, C_x centred covariance, absmean_x). C_x = gram/n − mean·meanᵀ, symmetrised."""
    import torch
    gram = torch.as_tensor(gram, dtype=torch.float64)
    sum_x = torch.as_tensor(sum_x, dtype=torch.float64)
    mean_x = sum_x / n
    C_x = gram / n - torch.outer(mean_x, mean_x)
    C_x = 0.5 * (C_x + C_x.transpose(0, 1))
    return mean_x, C_x, torch.as_tensor(absmean_sum, dtype=torch.float64) / n


def _sigma_eig(C_x):
    """Descending eigenbasis of a symmetric PSD covariance. Returns (evecs (d,d), evals (d,) ≥0)."""
    import torch
    C = torch.as_tensor(C_x, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(0.5 * (C + C.transpose(0, 1)))
    return evecs.flip(1), evals.flip(0).clamp_min(0.0)


def sink_eigendirections(R, massive_coords=MASSIVE_COORDS, frac=SINK_EIGEN_FRAC):
    """Indices j of basis vectors R[:,j] whose squared energy on the sink coords ≥ frac — the
    'sink-carrying' directions the sink-aware rival pins to b_max. Detected + frozen on discovery."""
    import numpy as np
    import torch
    R = torch.as_tensor(R, dtype=torch.float64)
    mc = list(massive_coords)
    sink_energy = (R[mc, :] ** 2).sum(0)                # (d,) per-column energy on sink coords
    return np.where(sink_energy.numpy() >= float(frac))[0].tolist()


# ===========================================================================
# The 7 policy (R, w, pinned) builders (spec §1.2 table).
# ===========================================================================
def build_policy(name, *, Q_evecs, Q_evals, sigma_evecs, sigma_evals, mean_x, absmean_x,
                 oracle_w=None, sink_dirs=None, seed=SEED):
    """Return (R (d,d), w (d,), pinned {coord:bits}) for one policy. Bases: Q eigvecs
    (susceptibility/random/oracle), Σ_x eigvecs (variance/sink-aware), or coordinate axes
    (magnitude/uniform). Only the pair (R,w) [+ the sink pin] differs — the quantiser/budget/eval
    are identical across policies, which isolates the map's contribution exactly (spec §1.2)."""
    import numpy as np
    import torch
    Q_evecs = torch.as_tensor(Q_evecs, dtype=torch.float64)
    d = Q_evecs.shape[0]
    if name == "susceptibility":
        return Q_evecs, np.asarray(Q_evals, np.float64), {}
    if name == "variance":
        return torch.as_tensor(sigma_evecs, dtype=torch.float64), np.asarray(sigma_evals, np.float64), {}
    if name == "sink_aware_variance":
        assert sink_dirs is not None, "sink-aware needs the discovery-frozen sink directions"
        return (torch.as_tensor(sigma_evecs, dtype=torch.float64), np.asarray(sigma_evals, np.float64),
                {int(j): B_MAX for j in sink_dirs})
    if name == "magnitude":
        return torch.eye(d, dtype=torch.float64), np.asarray(absmean_x, np.float64), {}
    if name == "uniform":
        return torch.eye(d, dtype=torch.float64), np.ones(d, np.float64), {}
    if name == "random":
        perm = np.random.default_rng(seed).permutation(d)
        return Q_evecs, np.asarray(Q_evals, np.float64)[perm], {}          # shuffled λ: ranking null
    if name == "oracle":
        assert oracle_w is not None, "oracle needs the discovery leave-one-out damage"
        return Q_evecs, np.asarray(oracle_w, np.float64), {}
    raise ValueError(f"unknown policy {name!r}")


def policy_basis(name, banks):
    """(R, centers, stds, w, pinned) — everything for a policy that does NOT depend on the budget b̄.
    The job computes this ONCE per policy (the eigen-bases are precomputed in `banks`; only basis_stats
    is per-policy), then sweeps budgets cheaply via water_fill(w, b̄, pinned)."""
    R, w, pinned = build_policy(
        name, Q_evecs=banks["Q_evecs"], Q_evals=banks["Q_evals"],
        sigma_evecs=banks["sigma_evecs"], sigma_evals=banks["sigma_evals"],
        mean_x=banks["mean_x"], absmean_x=banks["absmean_x"],
        oracle_w=banks.get("oracle_w"), sink_dirs=banks.get("sink_dirs"), seed=banks.get("seed", SEED))
    centers, stds = basis_stats(R, banks["mean_x"], banks["C_x"])
    return R, centers, stds, w, pinned


def make_spec(name, b_bar, banks, k_sigma=K_SIGMA, b_max=B_MAX):
    """Full quantiser spec {R, bits, centers, stds, pinned} for (policy, budget). The R/centers/stds
    are the SAME across budgets; only `bits` moves (water_fill)."""
    R, centers, stds, w, pinned = policy_basis(name, banks)
    bits = water_fill(w, b_bar, b_max=b_max, pinned=pinned)
    return {"policy": name, "b_bar": float(b_bar), "R": R, "bits": bits,
            "centers": centers, "stds": stds, "pinned": pinned}


# ===========================================================================
# Oracle leave-one-out expansion (band damage → per-Q-direction importance).
# ===========================================================================
def oracle_band_edges(d, n_bands=N_ORACLE_BANDS):
    """Contiguous top-first bands over the Q eigendirections (index 0 = stiffest)."""
    import numpy as np
    return np.linspace(0, d, n_bands + 1).astype(int)


def band_bits(d, band_lo, band_hi, b_max=B_MAX):
    """A bit vector that quantises ONLY [band_lo:band_hi] to 0 (centre) and keeps the rest at b_max —
    the leave-one-band-out ablation the oracle measures on discovery."""
    import numpy as np
    b = np.full(d, b_max, dtype=np.int64)
    b[band_lo:band_hi] = 0
    return b


def oracle_importance(band_damage, d, n_bands=N_ORACLE_BANDS):
    """Expand per-band measured Δbehavior (len n_bands, clamped ≥0) to per-Q-direction importance w.
    All-zero (no measurable per-band damage) → uniform ones (degenerate-safe)."""
    import numpy as np
    band_damage = np.clip(np.asarray(band_damage, np.float64), 0, None)
    assert band_damage.shape[0] == n_bands, (band_damage.shape, n_bands)
    edges = oracle_band_edges(d, n_bands)
    w = np.zeros(d, np.float64)
    for j in range(n_bands):
        w[edges[j]:edges[j + 1]] = band_damage[j]
    if w.max() > 0:
        return np.clip(w, 1e-6 * w.max(), None)
    return np.ones(d, np.float64)


# ===========================================================================
# Paired-item excess bootstrap (A-RATION-3): two-sided, seed-fixed.
# ===========================================================================
def paired_excess(good_a, good_b, seed=SEED, n_boot=N_BOOT):
    """E = mean over items of (goodness_a − goodness_b), paired, with a two-sided 95% item-bootstrap
    CI. `good_*` are per-item GOODNESS aligned by item (VQA correctness 0/1, or −NLL per window for
    ppl). Positive E ⇒ policy A preserves more behavior. The 'clean' reference cancels in the paired
    difference, so E is directly the A−B behavior gap."""
    import numpy as np
    a = np.asarray(good_a, np.float64)
    b = np.asarray(good_b, np.float64)
    assert a.shape == b.shape and a.ndim == 1 and a.size >= 10, "unpaired or tiny item sets"
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"E": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(boots.std(ddof=1)), "n": int(diff.size)}


def loss_ci(good_clean, good_policy, seed=SEED, n_boot=N_BOOT):
    """Behavior LOSS = mean(clean − policy) with a two-sided item-bootstrap CI + SE. Used for the
    range guard (uniform loss) and each policy's Δbehavior vs clean."""
    return paired_excess(good_clean, good_policy, seed=seed, n_boot=n_boot)   # mean(clean − policy)


def ppl_goodness(per_window):
    """Per-window goodness = −mean-NLL (higher better) for the paired bootstrap. per_window is a list
    of [Σnll, n_tok] per window."""
    return [-(s / max(n, 1)) for s, n in per_window]


# ===========================================================================
# adjudicate_ration — the spec §6 frozen verdict order (applied once on confirmation).
# ===========================================================================
def adjudicate_ration(*, E_vs_sinkaware, E_vs_naive, uniform_loss, oracle_headroom,
                      delta_range, random_saturated, mirror_mse_excess, name="ration"):
    """Frozen branch order. Every guard two-sided (feedback_one_sided_thresholds).
      E_vs_sinkaware : paired-excess dict (susceptibility − sink-aware-variance goodness); floor ci_lo>0.
      E_vs_naive     : paired-excess dict (susceptibility − naive-variance goodness).
      uniform_loss   : {E(=mean loss), se, ci_lo, ci_hi} of uniform-vs-clean behavior loss.
      oracle_headroom: {E} uniform_loss − oracle_loss (POSITIVE when the informed ceiling beats
                       uniform; the range-guard ceiling — must exceed delta_range or → VOID-no-range).
      delta_range    : the frozen sensitive-regime threshold (= MULT·SE(uniform loss) on discovery).
      random_saturated: bool — random-allocation loss ≈ uniform ≈ 0 at b̄* (saturation → VOID).
      mirror_mse_excess: {val} variance_MSE − susceptibility_MSE (must be >0 else the quantiser is
                         mis-wired — variance water-filling is MSE-optimal by construction).
    Returns {verdict, reason, ...inputs}."""
    range_ok = (uniform_loss["E"] >= delta_range)        # measurable behavior range at b̄*
    headroom_ok = (oracle_headroom["E"] > delta_range)   # something to allocate (oracle ≫ uniform)
    r = {"name": name, "range_ok": bool(range_ok), "headroom_ok": bool(headroom_ok),
         "random_saturated": bool(random_saturated),
         "E_vs_sinkaware": E_vs_sinkaware, "E_vs_naive": E_vs_naive,
         "uniform_loss": uniform_loss, "oracle_headroom": oracle_headroom,
         "delta_range": float(delta_range), "mirror_mse_excess": mirror_mse_excess}
    # 1. VOID-no-range — the benign-error boundary (a reported truth, not a halt).
    if (not range_ok) or (not headroom_ok) or random_saturated:
        r["verdict"] = "VOID-no-range"
        r["reason"] = ("no measurable behavior range / no oracle headroom / all-lossless at b̄*: "
                       "at deployable budgets on PG1 activations the allocation axis cannot matter "
                       "because the error is benign")
        return r
    # 2. halt-miswired — the ONLY legitimate halt (a wiring bug, documented + fixed + re-run).
    if mirror_mse_excess["val"] <= 0:
        r["verdict"] = "halt-miswired"
        r["reason"] = ("variance allocation does NOT beat susceptibility on reconstruction MSE — "
                       "variance water-filling is MSE-optimal by construction; if it loses on MSE "
                       "the quantiser wiring is broken")
        return r
    # 3–5. the hypothesis branches (guards passed; mirror holds).
    if E_vs_sinkaware["ci_lo"] > 0:
        r["verdict"] = "H_prescriptive"
        r["reason"] = ("susceptibility allocates behavior better than the strongest energy baseline "
                       "at matched bits, and it is not the sink → the two-regime law is prescriptive")
    elif E_vs_naive["ci_lo"] > 0:
        r["verdict"] = "H_sink-only"
        r["reason"] = ("susceptibility beats NAIVE variance but not sink-aware variance → the "
                       "practical content is the known sink ('don't waste bits on the sink')")
    else:
        r["verdict"] = "H_energy-suffices"
        r["reason"] = ("susceptibility ties/loses to sink-aware AND naive variance on sealed behavior "
                       "→ the map is diagnostic-only; the boundary of the two-regime law's reach is drawn")
    return r


# ===========================================================================
# Injection surface — SUBSTITUTE x̂ for the entering residual (not add a delta).
# ===========================================================================
def make_residual_substitutor(layers, L):
    """(install, holder): forward_pre_hook on layers[L] that SUBSTITUTES the entering residual with
    its quantised reconstruction under holder['spec'] (a make_spec() dict with R/bits/centers/stds on
    the model device+dtype). holder['spec']=None → passthrough (identity — the sanctioned clean
    forward; proves the hook is transparent). If holder['collect_mse'] is set, accumulates the
    per-forward reconstruction MSE Σ||x−x̂||² and token count (the mirror M readout). Reuses
    inherit_core._split_hidden — the frozen shared injector is not modified."""
    import contextlib

    from infra.pg.inherit_core import _split_hidden
    holder = {"spec": None}

    def hook(_m, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        spec = holder.get("spec")
        if spec is None:
            return None                                 # passthrough: leave inputs unchanged
        import torch
        flat = hs.reshape(-1, hs.shape[-1])
        xhat = quantize_in_basis(flat, spec["R"], spec["bits"], spec["centers"], spec["stds"])
        if holder.get("collect_mse"):
            holder["mse_sum"] = holder.get("mse_sum", 0.0) + float(
                ((flat.to(xhat.dtype) - xhat) ** 2).sum())
            holder["mse_cnt"] = holder.get("mse_cnt", 0) + flat.shape[0]
        return rebuild(xhat.reshape(hs.shape).to(hs.dtype))

    @contextlib.contextmanager
    def install():
        h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
        yield
        h.remove()
    return install, holder


def make_proj_substitutor(modules):
    """(install, holder): forward hooks on a LIST of projection submodules (e.g. every layer's
    self_attn.k_proj) that SUBSTITUTE each output with its quantised reconstruction under
    holder['spec'][i] (per-module make_spec dict). holder['spec']=None → passthrough. K's k_proj
    output is PRE-RoPE (rotary is applied inside attention), so quantising it here is RoPE-aware by
    construction (Arm K). Used to compress the whole KV cache in one forward."""
    import contextlib
    holder = {"spec": None}

    def mk(i):
        def hook(_m, _a, out):
            spec = holder.get("spec")
            if spec is None:
                return None                             # passthrough
            s = spec[i]
            flat = out.reshape(-1, out.shape[-1])
            xhat = quantize_in_basis(flat, s["R"], s["bits"], s["centers"], s["stds"])
            return xhat.reshape(out.shape).to(out.dtype)
        return hook

    @contextlib.contextmanager
    def install():
        hs = [m.register_forward_hook(mk(i)) for i, m in enumerate(modules)]
        try:
            yield
        finally:
            for h in hs:
                h.remove()
    return install, holder
