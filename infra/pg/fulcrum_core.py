"""FULCRUM core — the causal curvature-removal instrument (spec §2/§6/§13).

FULCRUM makes QUORUM's correlation (curvature footprint ρ=0.654 predicts n*_energy) causal: it strips a
capability's curvature content on the ONE reproducible curvature subspace (r=679, the 8 R1-certified
layers), reads the necessary-rank on the R1-exempt energy basis, and asks whether removability rises MORE
than an equal-magnitude random strip AT MATCHED GAIN.

The genuinely NEW logic here is small and weight-free-testable (spec §13 dry-run gate):

  1. `theta_family(dW, U_P, mtype, thetas)` — ΔW(θ,P) = ΔW − θ·project_delta(ΔW, U_P, mtype), the
     per-matrix intervention family. θ=1 is exactly the `residual` cell (P fully removed); θ=0 identity.
     (The whole-decoder version is `_cleave_shared.combine_alpha(ft_cell, shared_cell, base, θ)` — the
     SAME arithmetic, verified equal in the dry-run.)
  2. `g_star_jointly_reachable` / `e_curv_at_matched_gain` — freeze the matched gain-retention level g*
     (the deepest removal both paths can match, `cleave_core.rho_star_jointly_reachable` convention) and
     interpolate n*_energy vs gain-retention to g* per projection. E_curv = n*(curv@g*) − n*(random@g*).
  3. `inject_energy_capability` / `inject_curvature_capability` — the two synthetic controls: an
     energy-only ΔW (E_curv MUST be ≈0 — the floor, curvature-removal has nothing to remove) and a
     curvature-only ΔW with a small energy seed (E_curv MUST be strongly negative — the ceiling; strip
     the curvature and only the energy-aligned seed remains, so n*_energy collapses).
  4. `curvature_penalty` — the Leg-B training term ‖P_curv·(W−W_base)‖²_{r=679,CERT8} (CONDITIONAL).
  5. `adjudicate_fulcrum` — the frozen §6 branch order (VOID → H_lever → H_bystander), applied ONCE.

Everything measured reuses verified surfaces (`comwrite_surgery.project_delta`,
`cleave_core.mechanism_census`, `quorum_core.necessary_rank`). Heavy imports inside functions so the
module py_compiles on the control plane (no torch on the laptop). No try/except hiding — a wrong
assumption RAISES (CLAUDE.md).
"""
from __future__ import annotations

from infra.pg.gauge_common import N_LAYERS, PROJ  # noqa: F401
from infra.pg.quorum_core import MIN_GAIN_FULL  # noqa: F401  (the retention-denominator screen, 0.08)

# The intervention support (spec §2/§13): r=679 on the 8 R1-certified layers (overlap 0.974, R1-safe).
# Re-verified against the banks at job time (`certified_layers(banks, CURV_RANK) == CERT8`), never
# trusted from prose (baseline discipline).
CURV_RANK = 679
CERT8 = (0, 3, 5, 6, 11, 13, 14, 15)

# The θ-ladder (spec §2). Frozen on discovery; θ=0 is the full natural update, θ=1 the `residual` cell.
THETA_LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)

# The curvature-projection policies. `susceptibility` is the map (curv); the rest are its nulls/arms.
CURV = "susceptibility"                  # the certified Q subspace — the treatment
RANDOM = "random"                        # the matched-magnitude, matched-support Haar null
SHUFFLE = "susceptibility_shuffle"       # the structure control (same membership, ordering destroyed)
ENERGY_PROJ = "energy"                   # exploratory-only (confounded by readout-basis overlap)

SEED = 1337
D_MODEL = 2048


# ===========================================================================
# 1. the per-matrix θ-family (dry-run unit; the job uses combine_alpha, verified equal).
# ===========================================================================
def theta_family(dW, U_P, mtype, thetas=THETA_LADDER):
    """{θ: ΔW − θ·project_delta(ΔW, U_P, mtype)} for one (layer, mtype) matrix.

    θ=0 → ΔW verbatim; θ=1 → ΔW − P(ΔW), which is ⊥ span(U_P) on the residual-facing side to fp
    tolerance (the §13a gate). `project_delta` is the verified COMWRITE surface — this is only a thin
    scalar-combine wrapper around it."""
    import torch

    from infra.pg import comwrite_surgery as SURG
    D = torch.as_tensor(dW, dtype=torch.float32)
    out = {}
    P = SURG.project_delta(D, U_P, mtype)
    for th in thetas:
        out[float(th)] = D - float(th) * P
    return out


def residual_orthogonality(dW_theta1, U_P, mtype, tol=1e-4):
    """The §13a gate value: the residual energy of ΔW(θ=1) still inside span(U_P), relative to
    ‖ΔW(θ=1)‖². MUST be ≤ tol (the projection removed the in-span component). Fail-loud caller."""
    import torch

    from infra.pg import comwrite_surgery as SURG
    M = torch.as_tensor(dW_theta1, dtype=torch.float32)
    resid = SURG.project_delta(M, U_P, mtype)
    m2 = float((M ** 2).sum())
    return float((resid ** 2).sum()) / m2 if m2 > 0 else 0.0


# ===========================================================================
# 2. matched-gain interpolation — g* and E_curv.
# ===========================================================================
def _gain_retention(gain_by_theta):
    """|gain(θ)| / |gain(0)|, keyed by θ. The intervention's behavioral-effect MAGNITUDE that survives
    the θ-projection: 1 at θ=0, → 0 as the projection is fully removed. Absolute value so the SYNTHETIC
    PERTURBATION controls (whose effect can be a task DEGRADATION, gain(0) < 0 — energy amplification
    hurts) are handled identically to a real capability (gain(0) > 0); for a real cap
    |gθ|/|g0| ≡ gθ/g0. gain(0) must be non-zero (else the intervention has no measurable effect)."""
    g0 = gain_by_theta[0.0] if 0.0 in gain_by_theta else gain_by_theta[min(gain_by_theta)]
    assert abs(float(g0)) > 1e-9, f"gain(θ=0)={g0} ≈ 0 — no measurable effect, retention undefined"
    return {float(th): abs(float(g)) / abs(float(g0)) for th, g in gain_by_theta.items()}


def path_floor(gain_by_theta):
    """The deepest gain-retention this projection's θ-path reaches (min over θ). The reachability
    floor for the joint-g* selection."""
    ret = _gain_retention(gain_by_theta)
    return float(min(ret.values()))


# A-FULCRUM-1 (pre-freeze amendment, discovery data): the R1-safe intervention (r=679 on the 8 CERT8
# layers) has WEAK purchase on most capabilities — a full curvature (or random) strip on 8-of-18 layers
# removes only a few % of gain for a diffuse capability (measured: textvqa curv-floor 0.958, coco35l
# 0.984, infovqa 1.0). The original `max over ALL (cap,proj) floors` rule then freezes g*→1.0
# (degenerate: E_curv≡0 at θ=0). §2's intercurrent-events rule already says a cap that cannot reach g* is
# EXCLUDED and reported — so g* must be selected only from the caps the intervention actually MOVES. A
# path is "movable" iff its floor ≤ MOVABLE_MAX (the strip removes ≥ 1−MOVABLE_MAX of gain). g* = the
# deepest jointly-reachable level over the movable paths; unmovable caps are reported as no-range.
MOVABLE_MAX = 0.90


def g_star_jointly_reachable(gain_by_theta_by_proj_by_cap, projections=(CURV, RANDOM), margin=0.0,
                             movable_max=MOVABLE_MAX):
    """g* := the deepest gain-retention reachable by BOTH curv and random paths over the MOVABLE
    discovery capabilities (both path floors ≤ `movable_max`) — the max over movable (cap,proj) floors.
    Capabilities the r=679/CERT8 intervention cannot move (floor > movable_max on either path) carry no
    matched-gain contrast and are reported as no-range (A-FULCRUM-1). If NO cap is movable, g* falls back
    to `movable_max` and `no_movable=True` is flagged (the intervention has no purchase — a reported
    feasibility result, not a worked-around one).

    `gain_by_theta_by_proj_by_cap` = {cap: {proj: {θ: gain}}}. Returns {g_star, per_cap_floor,
    movable_floors, movable_caps, no_movable, binding}."""
    all_floors, movable_floors, movable_caps = {}, {}, []
    for cap, byproj in gain_by_theta_by_proj_by_cap.items():
        cap_floors = {}
        for proj in projections:
            assert proj in byproj, f"cap {cap} missing projection {proj}"
            f = path_floor(byproj[proj])
            all_floors[f"{cap}|{proj}"] = f
            cap_floors[proj] = f
        if all(v <= movable_max for v in cap_floors.values()):
            movable_caps.append(cap)
            for proj, f in cap_floors.items():
                movable_floors[f"{cap}|{proj}"] = f
    assert all_floors, "no discovery paths to freeze g* from"
    if not movable_floors:
        return {"g_star": float(movable_max), "per_cap_floor": all_floors, "movable_floors": {},
                "movable_caps": [], "no_movable": True, "binding": None}
    g = float(max(movable_floors.values())) + float(margin)
    return {"g_star": g, "per_cap_floor": all_floors, "movable_floors": movable_floors,
            "movable_caps": sorted(set(movable_caps)), "no_movable": False,
            "binding": max(movable_floors, key=lambda k: movable_floors[k])}


def interp_nstar_at_gstar(nstar_by_theta, gain_by_theta, g_star):
    """n*_energy interpolated to gain-retention g* along ONE projection's θ-path.

    The path is (gain_retention(θ), n*_energy(θ)). We read n* at x=g* by linear interpolation on the
    monotone (gain-retention increasing with decreasing θ) curve. Returns {n_star, reachable,
    n_star_theta0} — reachable=False (g* outside the path's retention span) ⇒ this cap is EXCLUDED from
    the paired primary and REPORTED (spec §2 intercurrent events), never silently zeroed."""
    import numpy as np
    ret = _gain_retention(gain_by_theta)
    thetas = sorted(ret, key=lambda t: ret[t])            # ascending gain-retention
    xs = np.array([ret[t] for t in thetas], dtype=np.float64)
    ys = np.array([float(nstar_by_theta[t]) for t in thetas], dtype=np.float64)
    lo, hi = float(xs.min()), float(xs.max())
    if not (lo - 1e-9 <= g_star <= hi + 1e-9):
        return {"n_star": None, "reachable": False, "span": [lo, hi],
                "n_star_theta0": float(nstar_by_theta[0.0]) if 0.0 in nstar_by_theta else None}
    # np.interp needs ascending x; xs is ascending in gain-retention. Ties → average (rare).
    n = float(np.interp(g_star, xs, ys))
    return {"n_star": n, "reachable": True, "span": [lo, hi],
            "n_star_theta0": float(nstar_by_theta[0.0]) if 0.0 in nstar_by_theta else None}


def e_curv_at_matched_gain(nstar_by_theta_curv, gain_by_theta_curv,
                           nstar_by_theta_rand, gain_by_theta_rand, g_star):
    """E_curv = n*_energy(curv @ g*) − n*_energy(random @ g*) for ONE capability.

    < 0 ⇒ removing curvature content lowers the necessary-rank MORE than a matched random removal at
    matched gain ⇒ the curvature-stored content was what forced n*_energy high (H_lever). Returns
    {E_curv, curv, random, reachable}; reachable=False (either path cannot interpolate to g*) ⇒
    excluded from the paired primary, reported."""
    c = interp_nstar_at_gstar(nstar_by_theta_curv, gain_by_theta_curv, g_star)
    r = interp_nstar_at_gstar(nstar_by_theta_rand, gain_by_theta_rand, g_star)
    if not (c["reachable"] and r["reachable"]):
        return {"E_curv": None, "curv": c, "random": r, "reachable": False}
    return {"E_curv": float(c["n_star"] - r["n_star"]), "curv": c, "random": r, "reachable": True}


def paired_e_curv_ci(e_curvs, seed=SEED, n_boot=10000, min_n=3):
    """Paired bootstrap median of E_curv across confirmation capabilities + 95% CI
    (`quorum_core.paired_delta_ci` mirror). H_lever needs ci_hi < 0. E=None + insufficient when
    < min_n reachable capabilities survive — the caller must not read that as a null."""
    import numpy as np
    d = np.asarray([float(e) for e in e_curvs if e is not None], dtype=np.float64)
    if d.size < min_n:
        return {"median": None, "ci_lo": None, "ci_hi": None, "n": int(d.size), "insufficient": True}
    rng = np.random.default_rng(int(seed))
    boots = np.median(d[rng.integers(0, d.size, size=(n_boot, d.size))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"median": float(np.median(d)), "ci_lo": float(lo), "ci_hi": float(hi),
            "mean": float(d.mean()), "n": int(d.size), "small_n": bool(d.size < 10),
            "insufficient": False}


def monotone_dose_response(nstar_by_theta, thetas=THETA_LADDER):
    """Spearman(θ, n*_energy) for one capability's curv path. H_lever wants < 0 (n* falls as more
    curvature is removed). Returns {rho, p_value, n}."""
    from scipy.stats import spearmanr
    th = [float(t) for t in thetas if t in nstar_by_theta or float(t) in nstar_by_theta]
    ns = [float(nstar_by_theta[float(t)]) for t in th]
    if len(set(ns)) <= 1 or len(ns) < 3:
        return {"rho": 0.0 if len(set(ns)) <= 1 else None, "p_value": 1.0, "n": len(ns)}
    rho, p = spearmanr(th, ns)
    return {"rho": float(rho), "p_value": float(p), "n": len(ns)}


# ===========================================================================
# 3. synthetic capabilities — the floor (energy-only) and ceiling (curvature-only) controls.
# ===========================================================================
def _side_block(U, k):
    """A rank-k symmetric block U_k U_kᵀ (d×d) living entirely in span(U[:, :k]) on BOTH sides — the
    residual-facing projector (used by the weight-free geometry check)."""
    import numpy as np
    Uk = np.asarray(U, dtype=np.float64)[:, :int(k)]
    return Uk @ Uk.T


def _inject_block(U, k, shape, mtype, seed):
    """A SHAPE-CORRECT rank-k ΔW of `shape`=(out_d, in_d) living entirely in span(U[:, :k]) on the
    residual-facing side of `mtype` (output side ⇒ rows in span(U); input side ⇒ columns in span(U)).
    Unlike `_side_block` this matches a possibly-NON-SQUARE matrix (o_proj/down_proj's in_d need not
    equal d). The off-residual factor is a fixed seeded Gaussian (arbitrary — only the residual side is
    constrained). Returns a numpy (out_d, in_d)."""
    import numpy as np

    from infra.pg import comwrite_core as CW
    out_d, in_d = int(shape[0]), int(shape[1])
    Uk = np.asarray(U, dtype=np.float64)[:, :int(k)]
    g = np.random.default_rng(int(seed))
    if CW.residual_side(mtype) == "output":
        assert Uk.shape[0] == out_d, f"U faces {Uk.shape[0]} != out_d {out_d}"
        return Uk @ g.standard_normal((int(k), in_d))            # rows ⊂ span(U)
    assert Uk.shape[0] == in_d, f"U faces {Uk.shape[0]} != in_d {in_d}"
    return g.standard_normal((out_d, int(k))) @ Uk.T            # columns ⊂ span(U)


def inject_energy_capability(base_lm_sd, banks, *, layers=CERT8, rank=8, scale=1.0,
                             curv_seed_frac=0.02, ttype="o_proj"):
    """A synthetic capability whose ΔW lives almost entirely in the TOP-`rank` ENERGY subspace, with a
    tiny curvature seed so curv-removal can still reach a matched gain. Injected at one output-side
    matrix (`ttype`) per `layers`.

    Physics (the §5 FLOOR / positive control): ΔW ⊂ span(U_energy[:, :rank]) ⇒ (a) small n*_energy
    (energy-removal at r≥rank annihilates it) and (b) E_curv ≈ 0 — the curvature projection has almost
    nothing to remove (Q⊥energy), so stripping it leaves n*_energy where it was. If curvature-removal
    DOES move this cap's n*_energy, the intervention is leaking into energy content ⇒ VOID (spec §6).

    Returns a synthetic ft decoder state-dict (base + the injection) — build_decoder-ready."""
    import numpy as np
    import torch
    ft = {k: v.detach().clone() for k, v in base_lm_sd.items()}
    from infra.pg.comwrite_surgery import grid_of
    grid = grid_of(base_lm_sd)
    for L in layers:
        key = grid[(int(L), ttype)]
        shape = tuple(base_lm_sd[key].shape)
        U_E = banks[int(L)]["sigma_evecs"]
        U_Q = banks[int(L)]["Q_evecs"]
        bulk = _inject_block(U_E, rank, shape, ttype, seed=1337 + int(L))          # energy bulk
        seed = _inject_block(U_Q, 1, shape, ttype, seed=7331 + int(L))             # tiny curvature seed
        bulk = bulk / (np.sqrt((bulk ** 2).sum()) + 1e-30)
        seed = seed / (np.sqrt((seed ** 2).sum()) + 1e-30)
        block = bulk + curv_seed_frac * seed                       # energy-dominated, tiny curv seed
        dW = torch.as_tensor(float(scale) * block, dtype=torch.float32).to(base_lm_sd[key].device)
        ft[key] = (base_lm_sd[key].float() + dW).to(base_lm_sd[key].dtype)
    return ft


def inject_curvature_capability(base_lm_sd, banks, *, layers=CERT8, rank=64, scale=1.0,
                                energy_seed_frac=0.05, ttype="o_proj"):
    """A synthetic capability whose ΔW lives in the TOP-`rank` CURVATURE (Q) subspace (⊥ energy), with
    a small ENERGY seed. Injected at one output-side matrix (`ttype`) per `layers`. `rank` is set well
    above the energy-control's so the curvature bulk is genuinely SPREAD across the energy spectrum
    (a high θ=0 n*_energy — the offline simulator confirms the reveal magnitude scales with it).

    Physics (the §5 CEILING / floor of the effect): the curvature bulk is ⊥ the energy readout basis
    (Q⊥energy) ⇒ n*_energy is HIGH at θ=0 (energy-removal barely touches it). Strip the curvature
    (θ→1) and ONLY the energy seed remains ⇒ n*_energy collapses to ≈`seed rank` — exactly the
    "what remains is energy-aligned" mechanism. A matched random strip leaves the bulk mostly intact
    ⇒ n*_energy stays high ⇒ E_curv STRONGLY NEGATIVE. This is the effect the instrument MUST see, or
    it is blind and the primary VOIDs. (The matched-gain g* is floor-limited by the random arm's
    rank-r capture, so E_curv is a CONSERVATIVE partial-reveal lower bound — the sign survives; see the
    offline-gate reachability note.)

    Returns a synthetic ft decoder state-dict (base + the injection)."""
    import numpy as np
    import torch
    ft = {k: v.detach().clone() for k, v in base_lm_sd.items()}
    from infra.pg.comwrite_surgery import grid_of
    grid = grid_of(base_lm_sd)
    for L in layers:
        key = grid[(int(L), ttype)]
        shape = tuple(base_lm_sd[key].shape)
        U_E = banks[int(L)]["sigma_evecs"]
        U_Q = banks[int(L)]["Q_evecs"]
        bulk = _inject_block(U_Q, rank, shape, ttype, seed=2337 + int(L))          # curvature bulk (⊥ energy)
        seed = _inject_block(U_E, 1, shape, ttype, seed=8331 + int(L))             # small energy seed
        bulk = bulk / (np.sqrt((bulk ** 2).sum()) + 1e-30)
        seed = seed / (np.sqrt((seed ** 2).sum()) + 1e-30)
        block = bulk + energy_seed_frac * seed                     # curvature-dominated + energy remainder
        dW = torch.as_tensor(float(scale) * block, dtype=torch.float32).to(base_lm_sd[key].device)
        ft[key] = (base_lm_sd[key].float() + dW).to(base_lm_sd[key].dtype)
    return ft


# ===========================================================================
# 4. curvature footprint (verify the intervention moved it) — energy-weighted census over CERT8.
# ===========================================================================
def footprint_census(ft_lm_sd, base_lm_sd, banks, rank=CURV_RANK, layers=CERT8):
    """Energy-weighted mean curvature / energy fraction of ΔW = ft − base over the `layers`×7 grid at
    `rank` (`cleave_core.mechanism_census` per matrix). The §5 footprint-moved gate reads: does
    ΔW(θ,curv)'s curvature-fraction FALL with θ while its energy-fraction stays ~stable?

    Returns {curvature, energy, random, chance_floor} — energy-weighted over the certified support."""
    import numpy as np

    from infra.pg import cleave_core as CL
    from infra.pg.comwrite_surgery import grid_of
    grid = grid_of(base_lm_sd)
    acc = {"susceptibility": 0.0, "energy": 0.0, "random": 0.0}
    wsum, floor = 0.0, None
    for L in layers:
        for t in PROJ:
            key = grid[(int(L), t)]
            dW = (ft_lm_sd[key].float() - base_lm_sd[key].float()).detach().cpu().numpy()
            fro2 = float((dW ** 2).sum())
            if fro2 <= 0:
                continue
            cen = CL.mechanism_census(dW, banks[int(L)], int(rank), t)
            floor = cen["chance_floor"]
            for p in acc:
                acc[p] += cen[p] * fro2
            wsum += fro2
    assert wsum > 0, "all-zero ΔW over the certified support — footprint undefined"
    return {"curvature": acc["susceptibility"] / wsum, "energy": acc["energy"] / wsum,
            "random": acc["random"] / wsum, "chance_floor": floor}


# ===========================================================================
# 5. curvature_penalty — the Leg-B training term (CONDITIONAL, only if learnability preflight GO).
# ===========================================================================
def curvature_penalty(model, base_weights, bases_by_layer, layers=CERT8, grid=None):
    """‖P_curv·(W−W_base)‖²_{r=679,CERT8} summed over the certified grid — the Leg-B penalty added to
    the task loss to STEER the curvature footprint down during training (H_steerable).

    `bases_by_layer` = {layer: U_curv[:, :679]}; `base_weights` = {param_name: W_base}. Reuses
    `comwrite_surgery.project_delta` so the penalized quantity is EXACTLY the footprint the swap-shell
    later removes. Returns a scalar torch loss (grad flows into W). NOT differentiated through U (a
    fixed base-model subspace)."""
    import torch

    from infra.pg import comwrite_surgery as CS
    from infra.pg.cleave_core import _target_grid
    grid = grid or _target_grid(model)
    params = dict(model.named_parameters())
    keep = {int(L) for L in layers}
    loss = None
    for n, (L, t) in grid.items():
        if L not in keep:
            continue
        p = params[n]
        Wb = base_weights[n].to(torch.float32)
        dW = p.to(torch.float32) - Wb
        into = CS.project_delta(dW, bases_by_layer[L].to(p.device), t)
        term = (into ** 2).sum()
        loss = term if loss is None else loss + term
    assert loss is not None, "curvature_penalty: no certified-layer grid params found"
    return loss


# ===========================================================================
# 6. adjudicate_fulcrum — the frozen §6 branch order (applied ONCE on confirmation).
# ===========================================================================
def adjudicate_fulcrum(*, e_curv_ci, null_band_hi, pos_floor_e_curv, ceiling_e_curv,
                       pos_floor_tol, ceiling_thresh, monotone_frac_curv,
                       footprint_moved, any_reach_gstar, shuffle_collapses,
                       name="fulcrum"):
    """Frozen §6 branch order. Every guard two-sided (`feedback_one_sided_thresholds`); no post-unblind
    rule swap (`feedback_no_post_unblind_rule_swap`).

      e_curv_ci         : paired E_curv dict (median + CI). H_lever needs ci_hi < 0 AND ci_hi < −null_band_hi.
      null_band_hi      : the 95th-pct of |E_curv| under random-vs-random (the symmetric zero-band edge).
      pos_floor_e_curv  : the ENERGY-ONLY synthetic's E_curv — MUST be ≈0 (|·| ≤ pos_floor_tol); if
                          curvature-removal moved its n*, the intervention LEAKS ⇒ VOID (spec §5/§6).
      ceiling_e_curv    : the CURVATURE-ONLY synthetic's E_curv — MUST be strongly negative
                          (≤ ceiling_thresh); if the instrument cannot see THIS, it is blind ⇒ VOID.
      monotone_frac_curv: fraction of confirmation caps with Spearman(θ, n*_curv) < 0 at BH-FDR.
      footprint_moved   : ΔW(θ,curv) curvature-fraction fell with θ while energy-fraction stayed stable.
      any_reach_gstar   : ≥1 confirmation capability interpolable to g* on BOTH curv and random paths.
      shuffle_collapses : the curvature-shuffle control's E collapsed to the random-vs-random band.
    Returns {verdict, reason, ...inputs}.
    """
    r = {"name": name, "e_curv_ci": e_curv_ci, "null_band_hi": null_band_hi,
         "pos_floor_e_curv": pos_floor_e_curv, "ceiling_e_curv": ceiling_e_curv,
         "pos_floor_tol": pos_floor_tol, "ceiling_thresh": ceiling_thresh,
         "monotone_frac_curv": monotone_frac_curv, "footprint_moved": bool(footprint_moved),
         "any_reach_gstar": bool(any_reach_gstar), "shuffle_collapses": bool(shuffle_collapses)}

    pos_ok = (pos_floor_e_curv is not None) and (abs(float(pos_floor_e_curv)) <= float(pos_floor_tol))
    ceil_ok = (ceiling_e_curv is not None) and (float(ceiling_e_curv) <= float(ceiling_thresh))
    r["pos_control_fired"] = bool(pos_ok and ceil_ok)

    # 1. VOID — the instrument is blind or leaking, or no cap reached g*. A "bystander" verdict from a
    #    blind/leaky instrument is the trap the two synthetic controls exist to spring (spec §5/§6).
    if (not pos_ok) or (not ceil_ok) or (not any_reach_gstar):
        r["verdict"] = "VOID"
        reasons = []
        if not pos_ok:
            reasons.append(f"the ENERGY-ONLY positive control leaked (|E_curv|={pos_floor_e_curv} > "
                           f"{pos_floor_tol}: curvature-removal moved a capability with no curvature "
                           "content — the projection is not clean)")
        if not ceil_ok:
            reasons.append(f"the CURVATURE-ONLY ceiling did not fire (E_curv={ceiling_e_curv} > "
                           f"{ceiling_thresh}: the instrument cannot see the effect it is built to "
                           "detect — it is blind)")
        if not any_reach_gstar:
            reasons.append("no confirmation capability interpolated to g* on both curv and random "
                           "paths (the matched-gain contrast is undefined)")
        r["reason"] = ("VOID — " + "; ".join(reasons) + ". Reported, not worked around (spec §6). "
                       "A null here would be an artifact of a blind/leaky instrument, not a bystander.")
        return r

    ci_hi = (e_curv_ci or {}).get("ci_hi")
    insuff = (e_curv_ci or {}).get("insufficient")
    # 2. INSUFFICIENT — the paired E_curv was not estimable (too few reachable caps). Withhold; do not
    #    read "did not clear" as "bystander" (the CLEAVE E17 / QUORUM lesson).
    if insuff or ci_hi is None:
        r["verdict"] = "INSUFFICIENT-primary"
        r["reason"] = (f"E_curv was not estimable (n={(e_curv_ci or {}).get('n')} reachable paired "
                       "capabilities; the estimator needs ≥3) — the primary is UNDECIDED. Missing "
                       "evidence, not evidence of bystander; report the point estimates and the axes "
                       "that landed (footprint-moved, monotonicity, the controls).")
        return r

    # 3. H_lever — curvature-removal lowers n*_energy MORE than random at matched gain, past the null
    #    band, with a monotone θ-dose-response and the footprint verified moved.
    below_band = ci_hi < -(float(null_band_hi) if null_band_hi is not None else 0.0)
    monotone_ok = (monotone_frac_curv is not None) and (float(monotone_frac_curv) >= 0.5)
    if ci_hi < 0 and below_band and monotone_ok and footprint_moved:
        r["verdict"] = "H_lever"
        r["reason"] = ("median E_curv < 0 with CI-hi below the −(random-vs-random) null band, a "
                       "monotone θ-dose-response in the majority of capabilities, and the curvature "
                       "footprint verified to fall with θ → curvature content CAUSALLY sets "
                       "energy-removability. QUORUM's ρ=0.65 is a LEVER: a targeted-unlearning / "
                       "protection knob, not a bystander correlation (spec §6).")
        return r

    # 4. H_bystander — E_curv is not distinguishable from random: its CI brackets 0, OR the median
    #    effect is SMALLER than the random-vs-random noise band (an effect below the noise floor is
    #    indistinguishable from random). WITH the controls firing, so a null means bystander, not a
    #    blind instrument.
    ci_lo = (e_curv_ci or {}).get("ci_lo")
    median = (e_curv_ci or {}).get("median")
    nb = float(null_band_hi) if null_band_hi is not None else 0.0
    brackets_zero = (ci_lo is not None and ci_lo <= 0 <= ci_hi)
    effect_within_band = (median is not None and abs(float(median)) <= nb)
    if brackets_zero or effect_within_band:
        r["verdict"] = "H_bystander"
        r["reason"] = ("E_curv brackets 0 and/or its median sits inside the random-vs-random null band "
                       "(curvature-removal is no more surgical than a matched random removal at "
                       "matched gain), with BOTH synthetic controls firing → QUORUM's ρ=0.65 was "
                       "carried by generic weakening / a confound, not a causal handle. The footprint "
                       "is a bystander predictor; we correct QUORUM's correlation (spec §6).")
        return r

    # 5. a SIGNIFICANT effect beyond the band (CI excludes 0, |median| > band) but not the clean lever —
    #    sign-aware (`feedback_one_sided_thresholds`: the positive tail is a real, reportable finding).
    if ci_hi < 0:
        r["verdict"] = "H_lever-weak"
        r["reason"] = ("median E_curv is negative and beyond the null band, but did not clear the full "
                       f"H_lever bar (CI-hi below −band={below_band}, monotone={monotone_ok}, "
                       f"footprint_moved={footprint_moved}) — curvature-removal is more surgical than "
                       "random but the causal claim is not fully resolved at this power. Borderline; "
                       "more capabilities / fuller layer coverage would sharpen it.")
        return r
    r["verdict"] = "H_anti-lever"
    r["reason"] = ("E_curv is significantly POSITIVE beyond the null band (ci_lo > 0): removing curvature "
                   "content makes the capability HARDER to energy-remove, not easier — the opposite of "
                   "H_lever. Reported as the other tail of the two-sided test; curvature-removal disturbs "
                   "the energy-aligned remainder rather than exposing it.")
    return r
