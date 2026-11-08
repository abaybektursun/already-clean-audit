"""CLEAVE core — the four STENCIL defects, each promoted to first-class engineering, plus the
protective-frontier statistic and `adjudicate_cleave` (spec §6 frozen verdict order).

CLEAVE's central object is the **functional decomposition of a natural update by Q**:
`ΔW = ΔW_in + ΔW_off` (in/off `span(U_sus)`), asked three ways — PROTECT (does removing `ΔW_in` keep
the task and shed the forgetting?), TARGET (which subspace carries the task?), MECHANISM (where does
the task gradient actually live?).

The genuinely NEW logic here is small and weight-free-testable:
  1. `certify_resolution`  — STENCIL's T1 confound closed as a *measured* gate (per-rank stability of
     `U_sus` across independent Q measurements), two-sided, with an independence guard so the gate
     cannot silently certify vacuously-identical repeats.
  2. `off_bases` / `retention` / `interp_forget_at_retention` / `knee_rank` / `rho_star_from_null` /
     `rho_reachable` — the protective frontier + the ρ*-reachability check (STENCIL's T2 fix).
  3. `mechanism_census` — task-gradient / natural-ΔW mass per policy subspace (STENCIL's T4 fix).
  4. `attach_grad_projection` / `project_iterate_off` — the PROTECT-trained glue (T5), including the
     AdamW-does-not-preserve-subspaces fix (A-CLEAVE-6).
  5. `adjudicate_cleave` — the §6 branch order, applied ONCE on confirmation.

Everything measured reuses verified surfaces (`comwrite_surgery.build_decoder/project_delta`,
`comwrite_core.energy_fraction/project_grad_residual/subspace_overlap`, `stencil_core`,
`_stencil_shared`, `ration_core.paired_excess`). Heavy imports inside functions so the module
py_compiles on the control plane (no torch on the laptop). No try/except hiding — a wrong assumption
RAISES (CLAUDE.md).
"""
from __future__ import annotations

from infra.pg.gauge_common import N_LAYERS, PROJ

# ---- the policies CLEAVE removes / writes into. 'susceptibility' is the map; the rest are its nulls.
POLICIES = ("susceptibility", "energy", "random", "susceptibility_shuffle")
MAP_POLICY = "susceptibility"
NULL_POLICY = "random"                  # the matched-rank protective null — H_protective lives or dies here
SPEC_POLICY = "energy"                  # the specificity control (is the win *susceptibility*-specific?)

# ---- A-CLEAVE-5: the PROTECT ladder MUST reach d, else retention does not span [1,0] and the spec's
# "reachable by construction" (T2) is false. HANDLE's ladder {5..679} + the completion to d=2048.
D_MODEL = 2048
PROTECT_LADDER = (5, 10, 20, 50, 150, 350, 679, 1024, 1536, 2048)

# ---- resolution certification (A-CLEAVE-3). Frozen before any map arm runs.
CERT_RANKS = (5, 10, 20, 50, 150, 350, 679)
CERT_TOL = 0.90                 # mean pairwise subspace_overlap required to certify a rank
CERT_MIN_REPEATS = 3            # ≥3 independent Q measurements (spec §2 exact instrument)
CERT_INDEP_TOL = 1e-6           # repeats must actually DIFFER (else the gate is vacuous)

SEED = 1337


# ===========================================================================
# 1. Operator-resolution certification — STENCIL's T1, closed as a measured gate.
# ===========================================================================
def certify_resolution(U_reps, ranks=CERT_RANKS, tol=CERT_TOL, evals_reps=None, d=D_MODEL):
    """Per-rank stability of the susceptibility basis across INDEPENDENT Q measurements.

    `U_reps` = [U^(1), U^(2), ...], each a (d, ≥max(ranks)) descending-eigenvalue Q eigenbasis
    measured with an independent (lanczos_seed, Hutchinson trace_seed, operator-item bootstrap).
    For each rank r: the mean/min pairwise `subspace_overlap(U^(i)[:, :r], U^(j)[:, :r])` ∈ [0,1]
    (1 = identical span; Haar floor = r/d). Rank r is **certified** iff mean overlap ≥ `tol`.

    Two-sided by construction (`feedback_one_sided_thresholds`): the floor is `tol`, the ceiling is
    1.0, the chance level r/d is reported at every rank, and an INDEPENDENCE guard fails loud if the
    repeats are (near-)identical — a gate that certifies vacuously-identical measurements is a
    statistic that cannot fail. `evals_reps` (optional, the matching Q spectra) powers that guard.

    Returns {per_rank:{r:{mean_overlap,min_overlap,haar_floor,certified}}, certified_ranks,
             certified_max, n_repeats, independence:{...}, tol}.
    """
    import itertools

    import numpy as np

    from infra.pg import comwrite_core as CW

    assert len(U_reps) >= CERT_MIN_REPEATS, \
        f"resolution certification needs ≥{CERT_MIN_REPEATS} independent Q measurements, got {len(U_reps)}"
    Us = [np.asarray(U, dtype=np.float64) for U in U_reps]
    for i, U in enumerate(Us):
        assert U.ndim == 2 and U.shape[0] == d, f"repeat {i}: U{U.shape} is not (d={d}, ·)"
        assert U.shape[1] >= max(ranks), f"repeat {i}: only {U.shape[1]} cols < max rank {max(ranks)}"

    # --- independence guard: the repeats must actually differ, or this gate cannot fail. ----------
    indep = {"checked": evals_reps is not None}
    if evals_reps is not None:
        E = [np.asarray(e, dtype=np.float64).ravel() for e in evals_reps]
        n = min(e.size for e in E)
        spread = 0.0
        for a, b in itertools.combinations(E, 2):
            denom = np.maximum(np.abs(a[:n]), 1e-300)
            spread = max(spread, float(np.max(np.abs(a[:n] - b[:n]) / denom)))
        indep["max_rel_eval_spread"] = spread
        assert spread > CERT_INDEP_TOL, (
            f"Q repeats are identical to {spread:.2e} — the resolution gate would certify vacuously. "
            "The repeats must vary (lanczos_seed, trace_seed, operator-item bootstrap).")

    per_rank, certified = {}, []
    for r in ranks:
        ovs = [float(CW.subspace_overlap(A[:, :r], B[:, :r])) for A, B in itertools.combinations(Us, 2)]
        assert ovs, "need ≥2 repeats to form a pairwise overlap"
        mean_ov, min_ov = float(np.mean(ovs)), float(np.min(ovs))
        haar = float(r) / float(d)
        # VACUITY guard (`feedback_one_sided_thresholds`): as r → d the overlap of two full bases is
        # 1.0 BY CONSTRUCTION — and so is the Haar floor. A rank whose chance level already clears
        # `tol` cannot fail the test, so it certifies nothing and must not be counted as resolved.
        vacuous = haar >= tol
        ok = (mean_ov >= tol) and not vacuous
        per_rank[int(r)] = {"mean_overlap": mean_ov, "min_overlap": min_ov,
                            "haar_floor": haar, "certified": bool(ok), "vacuous": bool(vacuous),
                            "pairwise": ovs}
        if ok:
            certified.append(int(r))
    return {"per_rank": per_rank, "certified_ranks": certified,
            "certified_max": max(certified) if certified else 0,
            "n_repeats": len(Us), "independence": indep, "tol": float(tol), "d": int(d)}


def resolved_ladder(ladder, cert):
    """The subset of `ladder` that is certified-resolved (map arms run ONLY here, spec §5).

    **Certification is a SET, not a ceiling.** An earlier version of this function treated
    `certified_max` as an upper bound (`r <= cap`), on the intuition that a spectral method resolves
    its top directions best and degrades into the tail. The L0 measurement says the opposite:
    stability RISES with rank (r=20 → 0.19, r=350 → 0.91, r=679 → 0.97). The reason is structural —
    `carry_core.materialize_q` builds Q by spectral completion as *a few Lanczos spikes + one bulk
    scalar per Σ sub-band*, so eigenvalues WITHIN a sub-band are degenerate and their eigenvectors
    are exchangeable. A small r cuts inside a band and selects an arbitrary slice of it; a large r
    contains whole bands and so spans a well-defined subspace. (At r=5 the measured overlap is 0.395
    ≈ **2/5** — precisely the 2 reproducible spikes out of 5 requested directions.)

    So the certified set is whatever ranks actually passed, in any order. Dropped ranks are returned
    so the job can `log()` them — a silently-truncated ladder reads as "covered everything" when it
    did not (`feedback_signals_that_cannot_fail`)."""
    ok = {int(r) for r in cert["certified_ranks"]}
    kept = [int(r) for r in ladder if int(r) in ok]
    dropped = [int(r) for r in ladder if int(r) not in ok]
    return {"kept": kept, "dropped": dropped,
            "certified_ranks": sorted(ok), "certified_max": int(cert.get("certified_max", 0))}


# ===========================================================================
# 2. The protective frontier — PROTECT's bases, retention, iso-ρ* read, and the frozen selectors.
# ===========================================================================
def off_bases(banks, policy, rank, seed=SEED, n_layers=N_LAYERS, device=None, dtype=None):
    """The `bases` grid for `comwrite_surgery.build_decoder("residual", …)` = {(L, mtype): U_P[L][:, :r]}.

    Every one of the 7 matrix types at layer L takes the SAME per-layer residual-space basis: for
    PG1 all 7 have residual-facing dim d=2048 (q/gate/up/k/v read the residual on their INPUT side,
    o/down write it on their OUTPUT side — `comwrite_core.RESIDUAL_SIDE`). The "residual" cell then
    installs `ΔW_off = ΔW − proj_{U_P}ΔW` at every matrix simultaneously (A-CLEAVE-2).

    The per-LAYER sub-seed (sha1, `attach_fixed_adapters`' recipe) matters for the nulls: a single
    seed would draw the SAME random subspace at all 18 layers, which is a correlated — and therefore
    weaker — null than the independent draw the design intends.

    `device`/`dtype` place the bases beside the state-dict being projected. This is a throughput knob,
    not a semantic one: `project_delta` already casts to fp32 internally and follows its inputs'
    device, so a CUDA basis + CUDA state-dict runs the identical arithmetic ~100× faster (at rank
    2048 the 16384-wide MLP projections are ~15 TFLOP per install — minutes on 8 vCPU, sub-second on
    an L40S)."""
    import hashlib

    from infra.pg import stencil_core as SC
    out = {}
    for L in range(n_layers):
        assert L in banks, f"no bank for layer {L}"
        sub = int(hashlib.sha1(f"{seed}:{policy}:L{L}".encode()).hexdigest()[:8], 16)
        U = SC.build_policy_basis(policy, banks[L], rank, seed=sub)
        if dtype is not None:
            U = U.to(dtype)
        if device is not None:
            U = U.to(device)
        for t in PROJ:
            out[(L, t)] = U
    return out


def certified_layers(banks, rank, n_layers=N_LAYERS):
    """The layers at which `rank` is certified-resolved — the support of a well-defined map
    intervention at that rank (A-CLEAVE-11)."""
    return sorted(L for L in range(n_layers)
                  if int(rank) in {int(r) for r in banks[L].get("certified_ranks", [])})


def off_bases_restricted(banks, policy, rank, layers, seed=SEED, n_layers=N_LAYERS, device=None):
    """`(bases, excluded)` for `comwrite_surgery.build_decoder` — project ONLY at `layers`.

    **A-CLEAVE-11.** The certification is per (layer, rank): `U_sus[L]` reproduces at some layers and
    at none at others (L1 certifies nothing). A whole-stack projection at a single rank therefore has
    no reproducible referent — the strict intersection over 18 layers is empty, and one layer would
    veto the entire study. The coherent question that remains is *"does removing the map's directions,
    **where the map is defined**, protect?"* — so the intervention's support becomes the certified
    layer set, and the excluded layers pass the update through **untouched** (`build_decoder`'s
    `excluded` contract: shared→0, so the `residual` cell injects the full ΔW there).

    This is blind-safe, not a garden path: the layer set is fixed by the CERTIFICATION — a base-model
    object measured before any behavioural read — and the **identical** set is applied to the
    treatment and to every null, so the contrast stays matched. The support is reported per rank.
    """
    import hashlib

    from infra.pg import stencil_core as SC
    keep = {int(L) for L in layers}
    assert keep, f"no certified layers at rank {rank} — the intervention has no support"
    bases, excluded = {}, []
    for L in range(n_layers):
        if L in keep:
            assert L in banks, f"no bank for layer {L}"
            sub = int(hashlib.sha1(f"{seed}:{policy}:L{L}".encode()).hexdigest()[:8], 16)
            U = SC.build_policy_basis(policy, banks[L], rank, seed=sub)
            if device is not None:
                U = U.to(device)
            for t in PROJ:
                bases[(L, t)] = U
        else:
            for t in PROJ:
                excluded.append((L, t))
    return bases, excluded


# A-CLEAVE-12: the removal-STRENGTH ladder, swept at a fixed certified rank.
ALPHA_LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)

# A-CLEAVE-13: the retention denominator must clear the measurement noise, or `retention` is a
# ratio estimator dividing by ~0. Held-out subject accuracy on the eval half (n≈767 for the large
# MMLU subjects) has SE ≈ √(0.25·0.75/767) ≈ 0.0156, so a GAIN (two reads) carries SE ≲ 0.022.
# Measured on PG1: professional_law's natural update gains **0.0244** — about **1.1σ**, giving
# retention ≈ **91% relative noise**, which is exactly what produced non-monotone α-curves
# (energy: 0.63 → 0.74 → 0.37 → 0.79 — not signal). A cell is admitted only if its natural update
# clears ~3.6σ, i.e. retention noise ≲ 28%. Cells below it are EXCLUDED and REPORTED, never
# silently averaged in (`feedback_signals_that_cannot_fail`).
MIN_GAIN_FULL = 0.08


def pick_r_star_by_support(support):
    """r* := the certified rank with the LARGEST layer support; ties → the smaller rank.

    **A-CLEAVE-12.** §6 specifies r* as "the knee of the off-random frontier" across ranks. The
    certification makes that quantity ill-defined here: the support is itself rank-dependent (measured
    on PG1: r=20 → 1 layer, r=50 → 1, r=350 → 3, r=679 → 8), so a sweep across ranks varies the
    intervention's COVERAGE at the same time as its rank — the resulting "knee" would be a knee in two
    variables at once, not the quantity §6 intended. Worse, a rank sweep at *fixed* support is
    impossible: those layers certify only at 679.

    So r* is chosen to give the map its **most complete well-defined intervention** — the rank whose
    certified support covers the most layers. This is blind-safe and cannot garden-path: it reads ONLY
    the certification, a base-model object measured before any behavioural read, and it selects the
    setting most FAVOURABLE to the hypothesis under test. The retention curve is then swept at that
    fixed rank via `ALPHA_LADDER` (removal strength), not by moving the rank.

    `support` = {rank: [layers]}. Returns (r_star, n_layers)."""
    usable = {int(r): list(ls) for r, ls in support.items() if ls}
    assert usable, ("no rank has ANY certified layer support — U_sus has no reproducible referent "
                    "anywhere in the stack; report that, do not pick an r*")
    best = max(usable, key=lambda r: (len(usable[r]), -r))
    return best, len(usable[best])


def combine_alpha(ft_cell, shared_cell, base_sd, alpha):
    """`W_off(α) = W_ft − α·(W_shared − W_base)` = `W_base + (ΔW − α·proj_U ΔW)`.

    Built from two `build_decoder` outputs — the `ft` cell and the `shared` cell — so the projection
    itself stays inside the verified COMWRITE surface and this is only a weighted combine. α=0 → the
    full natural update (retention 1); α=1 → the ordinary `residual` cell (the map's component fully
    removed). Sweeping α therefore traces a genuine (retention, forgetting) curve **at a fixed,
    certified rank**, which a rank sweep cannot do here.

    Non-target keys are `W_base` in every cell already, so they pass through unchanged for any α."""
    import torch
    a = float(alpha)
    assert 0.0 <= a <= 1.0, f"alpha {a} outside [0,1]"
    out = {}
    for k, w in base_sd.items():
        f = ft_cell[k].to(torch.float32)
        s = shared_cell[k].to(torch.float32)
        b = w.to(torch.float32)
        out[k] = (f - a * (s - b)).to(w.dtype)
    return out


def retention(gain_off, gain_full, tol=1e-9):
    """Task-retention = gain(ΔW_off) / gain(ΔW_full) (spec §2 'Variable').

    The denominator must be POSITIVE, not merely non-zero: a natural update that made the model
    WORSE (measured on PG1: high_school_macroeconomics s2024 gains **−0.108**, i.e. the fine-tune cost
    10.8 points on its own subject) has no "gain" to retain, and dividing by it yields a number that
    looks like a retention and means nothing. An `abs()` guard admits exactly the case it exists to
    exclude."""
    assert float(gain_full) > tol, \
        f"natural update did not gain ({gain_full}) — retention undefined (VOID-no-range)"
    return float(gain_off) / float(gain_full)


def interp_forget_at_retention(retentions, forgets, rho):
    """Forgetting at task-retention ρ along ONE policy's own (retention, forgetting) curve.

    Delegates to the STENCIL interpolator verbatim — it is the same operation (read y at a frozen x
    along an arm's own trajectory, ties → worst-case max-forget, out-of-span → None). The x-axis is
    the CLEAVE fix: retention spans [1,0] for every policy because `PROTECT_LADDER` reaches d
    (A-CLEAVE-5), so `None` here means a genuine data defect, not the design failure that broke
    STENCIL's τ*."""
    from infra.pg import stencil_core as SC
    return SC.interp_forget_at_gain(retentions, forgets, rho)


def knee_rank(ranks, reductions):
    """r* := the knee of the NULL's own protective curve (spec §6 selection) — the rank past which
    forgetting-reduction diminishes. Classic elbow: on (log rank, reduction) normalized to the unit
    box, the point of maximum perpendicular distance from the chord joining the endpoints.

    Selecting r* from `off-random`'s curve (never the treatment's) is what keeps the selector off the
    garden path. Deterministic, hyperparameter-free. Fail-loud on a degenerate curve."""
    import numpy as np
    x = np.log(np.asarray(ranks, dtype=np.float64))
    y = np.asarray(reductions, dtype=np.float64)
    assert x.shape == y.shape and x.size >= 3, "need ≥3 (rank, reduction) points for a knee"
    order = np.argsort(x)
    x, y = x[order], y[order]
    xr, yr = float(x[-1] - x[0]), float(y[-1] - y[0])
    assert xr > 0, "degenerate rank axis"
    if abs(yr) < 1e-12:                       # a flat null curve has no knee → the widest useful rank
        return int(round(float(np.exp(x[-1]))))
    xn = (x - x[0]) / xr
    yn = (y - y[0]) / yr
    dist = np.abs(yn - xn) / np.sqrt(2.0)     # ⟂ distance to the unit-box diagonal (the chord)
    return int(round(float(np.exp(x[int(np.argmax(dist))]))))


def rho_star_from_null(null_curve, r_star):
    """ρ* := the task-retention the NULL (`off-random`) achieves at the frozen r* (spec §6). Derived
    from the null's own curve — never from the treatment — so the iso-point cannot be tuned toward a
    map win. `null_curve` = [{rank, retention}, …]. Fail-loud if r* is not on the null's curve."""
    for p in null_curve:
        if int(p["rank"]) == int(r_star):
            return float(p["retention"])
    raise AssertionError(f"r*={r_star} not present on the null curve "
                         f"(ranks {sorted(int(p['rank']) for p in null_curve)})")


def rho_reachable(policy_curves, rho, tol=1e-9):
    """Is ρ* inside EVERY policy's achieved retention span? (spec §4 — the explicit STENCIL fix: τ*
    was frozen without a held-out reachability check, and the susceptibility arm then reached it in
    0/9 cells, leaving the primary undefined.)

    `policy_curves` = {policy: [{rank, retention}, …]}. Returns {reachable, per_policy:{p:{lo,hi,ok}}}.
    With `PROTECT_LADDER` reaching d every span should be ≈[0,1]; a False here is a real defect and
    must block the freeze, not be worked around."""
    out, ok_all = {}, True
    assert policy_curves, "no policy curves for the reachability check"
    for p, curve in policy_curves.items():
        rs = [float(c["retention"]) for c in curve]
        assert rs, f"policy {p!r} has an empty curve"
        lo, hi = min(rs), max(rs)
        ok = (lo - tol) <= float(rho) <= (hi + tol)
        out[p] = {"lo": lo, "hi": hi, "ok": bool(ok)}
        ok_all = ok_all and ok
    return {"reachable": bool(ok_all), "rho": float(rho), "per_policy": out}


def paired_boot(a, b, seed=SEED, n_boot=10000, min_n=3):
    """Paired-cell bootstrap of mean(a − b) with a two-sided 95% CI — the CLEAVE primary estimator.

    Small-n-safe (n ≥ `min_n`): `ration_core.paired_excess` asserts n ≥ 10 items, but CLEAVE pairs by
    (task, seed) CELLS — the sealed grid is 9–25 cells, not 10+ items — so the item-level estimator
    would refuse to run on a perfectly valid design. Same statistic, cell-level. `small_n` is carried
    on the result so no downstream reader can mistake a 9-cell CI for an item-level one.

    Returns E=None + insufficient=True when fewer than `min_n` pairs survive — the caller must not
    silently treat that as a null result."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1, f"unpaired inputs {a.shape} vs {b.shape}"
    if a.size < min_n:
        return {"E": None, "ci_lo": None, "ci_hi": None, "n": int(a.size), "insufficient": True}
    diff = a - b
    rng = np.random.default_rng(int(seed))
    boots = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"E": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(boots.std(ddof=1)), "n": int(diff.size),
            "small_n": bool(diff.size < 10), "insufficient": False}


def clears_floor(excess):
    """True iff a paired-excess dict has a numeric CI strictly above 0. An insufficient/absent
    contrast (ci_lo None) does NOT clear — it is missing evidence, never evidence of an effect."""
    if not excess:
        return False
    lo = excess.get("ci_lo")
    return lo is not None and float(lo) > 0


def protective_frontier(cells, policy, key="alpha"):
    """One policy's (retention, forgetting) frontier pooled over cells → sorted [{`key`, retention,
    forgetting}]. The headline figure's curve; `up-and-left` (high retention, low forgetting) is better.

    `key` is the swept axis: "alpha" (removal STRENGTH at the fixed certified r*, A-CLEAVE-12) or
    "rank". Grouping by rank would collapse the whole α-sweep into a single point, since every α cell
    sits at the same r*."""
    import numpy as np
    pts = [c for c in cells if c["policy"] == policy]
    assert pts, f"no cells for policy {policy!r}"
    by = {}
    for c in pts:
        by.setdefault(float(c[key]), []).append(c)
    out = []
    for x in sorted(by):
        grp = by[x]
        out.append({key: x,
                    "retention": float(np.mean([g["retention"] for g in grp])),
                    "forgetting": float(np.mean([g["forgetting"] for g in grp])),
                    "n": len(grp)})
    return out


def rho_star_jointly_reachable(policy_curves, margin=0.0):
    """ρ* := the LOWEST retention every policy can reach — i.e. `max_i min_j retention_ij`.

    Each policy's α-curve runs from retention ≈1 (α=0, nothing removed) down to its own α=1 value, so
    the jointly-reachable band is [max_i ret_i(α=1), ~1]. Taking its floor makes ρ* the most removal
    ALL policies can match — the most discriminating iso-point, and reachable by every policy by
    construction (the policy that defines it sits exactly on its endpoint; the others interpolate,
    never extrapolate).

    This is the α-axis mirror of STENCIL's `common_tau` (min over arms of each arm's max achieved),
    the lineage's established reachability convention — not a new rule invented for this landing."""
    import numpy as np
    assert policy_curves, "no curves for ρ* selection"
    floors = {}
    for p, curve in policy_curves.items():
        rs = [float(c["retention"]) for c in curve]
        assert rs, f"policy {p!r} has an empty curve"
        floors[p] = min(rs)
    rho = float(max(floors.values())) + float(margin)
    return {"rho_star": rho, "per_policy_floor": floors,
            "binding_policy": max(floors, key=lambda p: floors[p])}


# ===========================================================================
# 3. MECHANISM — where does the task gradient / the natural ΔW actually live? (STENCIL's T4)
# ===========================================================================
def mechanism_census(M, layer_bank, rank, mtype, seed=SEED, policies=POLICIES):
    """Fraction of a matrix `M`'s Frobenius energy captured on its RESIDUAL-FACING side by each
    policy subspace at `rank` (`comwrite_core.energy_fraction`), plus the Haar chance floor r/d.

    `M` is either a task GRADIENT (the mechanism read: does the task gradient live in `U_energy`
    rather than `U_sus`?) or a natural ΔW (the decomposition read). This converts STENCIL's inferred
    "sensitivity ≠ learnability" into a measurement — the whole point of the axis. Every fraction is
    compared against `chance_floor`, never against 0."""
    from infra.pg import comwrite_core as CW
    from infra.pg import stencil_core as SC
    d = int(layer_bank["d"])
    out = {}
    for name in policies:
        U = SC.build_policy_basis(name, layer_bank, rank, seed=seed)
        Un = U.numpy() if hasattr(U, "numpy") else U
        out[name] = float(CW.energy_fraction(M, Un, mtype))
    out["chance_floor"] = CW.chance_floor(rank, d)
    return out


def census_contrast(census):
    """The mechanism's one number: energy(U_energy) − energy(U_sus), both against the same floor.
    > 0 ⇒ the task gradient mass is in the ENERGY directions, not the susceptibility directions —
    the unifying prediction's load-bearing leg (spec §1.3, a hypothesis-to-kill)."""
    return float(census[SPEC_POLICY]) - float(census[MAP_POLICY])


# ===========================================================================
# 4. PROTECT-trained (T5) — learn in the complement of U_P, exactly.
# ===========================================================================
VISION_MARKERS = ("vision", "image", "siglip")


def _target_grid(model):
    """{param_name: (L, mtype)} for the 126 TEXT-decoder matrices — the params the natural update
    trains and a trained-off run constrains. Mirrors `comwrite_surgery.grid_of` on named_parameters
    (weights, not state-dict).

    **The vision exclusion is load-bearing, not hygiene.** PaliGemma's SigLIP tower is also built from
    `...vision_model.encoder.layers.N.self_attn.{q,k,v}_proj`, so a name-only match returns
    **207 = 126 + 27 SigLIP layers × 3 colliding types** (SigLIP's other matrices are `out_proj` /
    `fc1` / `fc2`, which do not collide). Without this filter `mark_grid_trainable` would train the
    vision tower, and `layer_index` would index a TEXT layer's Q bank by a VISION layer number —
    projecting a 1152-wide SigLIP matrix against a 2048-wide residual basis. `carry_c.py` solved the
    same collision for state-dicts by scoping to `language_model`; this is that fix for parameters."""
    from infra.pg.gauge_common import layer_index, matrix_type
    out = {}
    for n, p in model.named_parameters():
        if p.ndim != 2 or "layers." not in n:
            continue
        if any(v in n.lower() for v in VISION_MARKERS):
            continue
        if any(f".{t}." in n for t in PROJ):
            out[n] = (layer_index(n), matrix_type(n))
    assert out, "no 18×7 grid params found on the model"
    return out


def mark_grid_trainable(model):
    """Unfreeze EXACTLY the 126 residual-facing projection matrices — the natural update's carrier.

    **A-CLEAVE-7.** `stencil_core.mark_full_ft_trainable` unfreezes every param matching `layers.N.`,
    which includes the per-layer LAYERNORM gains. But `comwrite_surgery.build_decoder` rebuilds only
    the 7 PROJ types and `assert_preserved` guarantees every non-target key stays **bitwise base** —
    so a layernorm ΔW would be silently DROPPED by the install. The `ft` cell could then never
    reproduce the trained model, and PROTECT's wiring gate (its ground truth) would fail for a reason
    that has nothing to do with the map.

    The fix is not to smuggle the layernorms through the install but to define the natural update
    where the projection can act: Q is a residual-space operator and the decomposition
    `ΔW = ΔW_in + ΔW_off` is defined on these matrices' residual-facing sides. A layernorm gain is a
    per-channel scalar — not in the decomposable space at all. Training exactly the grid makes ΔW
    fully decomposable, the `ft` cell bit-exact, and the wiring gate meaningful.

    Returns {trainable_params, n_param_tensors, arm}."""
    grid = _target_grid(model)
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for n in grid:
        params[n].requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert len(grid) == N_LAYERS * 7, f"grid has {len(grid)} matrices, expected {N_LAYERS * 7}"
    return {"trainable_params": int(trainable), "n_param_tensors": len(grid), "arm": "grid_ft"}


def policy_bases_by_layer(banks, policy, rank, seed=SEED, n_layers=N_LAYERS, device=None):
    """{layer: U_P[:, :rank]} — the per-layer residual-space basis, built ONCE. Same per-layer
    sub-seed recipe as `off_bases` (an independent draw per layer for the nulls)."""
    import hashlib

    from infra.pg import stencil_core as SC
    out = {}
    for L in range(n_layers):
        assert L in banks, f"no bank for layer {L}"
        sub = int(hashlib.sha1(f"{seed}:{policy}:L{L}".encode()).hexdigest()[:8], 16)
        U = SC.build_policy_basis(policy, banks[L], rank, seed=sub)
        out[L] = U.to(device).float() if device is not None else U
    return out


def attach_grad_projection(model, bases_by_layer):
    """Register a tensor hook on every grid weight that projects its gradient OFF `span(U_P)` on the
    residual-facing side (`comwrite_core.project_grad_residual(..., mode='off')` — the surface the
    spec names). Returns (handles, info); call `.remove()` on every handle to detach.

    NOTE (A-CLEAVE-6): gradient projection ALONE does not keep an AdamW iterate in the off-subspace —
    AdamW's elementwise m/√v preconditioning is not a linear map, so an update built from off-span
    gradients can re-acquire in-span components. `project_iterate_off` closes that hole after each
    step; the two are used together."""
    from infra.pg import comwrite_core as CW
    grid = _target_grid(model)
    params = dict(model.named_parameters())
    handles = []
    for n, (L, t) in grid.items():
        assert L in bases_by_layer, f"no basis for layer {L}"
        U = bases_by_layer[L]
        handles.append(params[n].register_hook(
            lambda g, U=U, t=t: CW.project_grad_residual(g, U, t, "off")))
    return handles, {"n_params": len(grid), "layers": sorted(bases_by_layer)}


def project_iterate_off(model, base_weights, bases_by_layer, grid=None):
    """Project the ACCUMULATED update off `span(U_P)` exactly: W ← W_base + (ΔW − proj_{U_P}ΔW) for
    every grid weight, with ΔW = W − W_base. Called after each optimizer step (projected AdamW).

    This is the constraint-set projection that makes "trained in the complement" *true* rather than
    approximate (A-CLEAVE-6). `base_weights` = {param_name: W_base} captured before training (kept on
    the param's device). Returns the mean in-span energy it REMOVED — reported rather than assumed to
    be zero, because a number that is always 0 is a signal that cannot fail: it is precisely the size
    of the hole AdamW's elementwise preconditioning opens in the spec's stated method."""
    import torch

    from infra.pg import comwrite_surgery as CS
    grid = grid or _target_grid(model)
    removed = []
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, (L, t) in grid.items():
            p = params[n]
            Wb = base_weights[n].to(torch.float32)      # stored in the model's dtype; fp32 to project
            dW = p.detach().to(torch.float32) - Wb
            into = CS.project_delta(dW, bases_by_layer[L], t).to(dW.device)
            fro2 = float((dW ** 2).sum())
            removed.append(float((into ** 2).sum()) / fro2 if fro2 > 0 else 0.0)
            p.copy_((Wb + (dW - into)).to(p.dtype))
    return {"mean_inspan_energy_removed": sum(removed) / len(removed), "n_params": len(removed)}


# ===========================================================================
# 5. adjudicate_cleave — the spec §6 frozen verdict order (applied ONCE on confirmation).
# ===========================================================================
def adjudicate_cleave(*, E_protect, specificity, mechanism_confirms,
                      wiring_ok, resolution_ok,
                      forget_range_ok, removable_range_ok,
                      null_saturates, targeting_wins,
                      map_hurts_task=None, name="cleave"):
    """Frozen branch order (spec §6). Every guard two-sided (`feedback_one_sided_thresholds`); no
    post-unblind rule swap (`feedback_no_post_unblind_rule_swap`).

      E_protect        : paired-excess dict, Forgetting(off-random) − Forgetting(off-sus) @ (r*, ρ*);
                         floor ci_lo > 0 ⇒ removing the map's directions cuts more forgetting than
                         removing a random subspace AT EQUAL TASK COST.
      specificity      : paired-excess dict, Forgetting(off-energy) − Forgetting(off-sus) @ (r*, ρ*);
                         floor ci_lo > 0 ⇒ the win is specific to SUSCEPTIBILITY, not to "any
                         structured subspace".
      mechanism_confirms : the task-gradient mass sits in U_energy, not U_sus (census_contrast > 0
                         beyond the chance floor) — the unifying prediction's measured leg.
      wiring_ok        : the full-ΔW swap-shell install reproduces the natural fine-tune.
      resolution_ok    : U_sus certified stable at the ranks the map arms used.
      forget_range_ok  : the natural fine-tune forgot > δ_forget (there IS collateral to remove).
      removable_range_ok : full − off-full forgetting > δ_range (the projection CAN move forgetting).
      null_saturates   : off-random already removes all forgetting at no task cost (ANY subspace
                         protects) — then this is not a map win.
      targeting_wins   : into-U_sus beats into-U_energy on iso-forgetting task-gain (STENCIL overturned).
      map_hurts_task   : optional descriptor — off-sus costs as much task as off-random (H_map-hurts,
                         §1.2). Reported, never a verdict of its own (§6 licenses 5 branches only).
    Returns {verdict, reason, ...inputs}.
    """
    r = {"name": name, "E_protect": E_protect, "specificity": specificity,
         "mechanism_confirms": bool(mechanism_confirms), "wiring_ok": bool(wiring_ok),
         "resolution_ok": bool(resolution_ok), "forget_range_ok": bool(forget_range_ok),
         "removable_range_ok": bool(removable_range_ok), "null_saturates": bool(null_saturates),
         "targeting_wins": bool(targeting_wins), "map_hurts_task": map_hurts_task}

    # 1. halt-miswired — the instrument is not measuring what it claims. Fix; do NOT interpret.
    if (not wiring_ok) or (not resolution_ok):
        r["verdict"] = "halt-miswired"
        r["reason"] = ("the full-ΔW swap-shell install does not reproduce the natural fine-tune, "
                       "and/or U_sus failed the operator-resolution certification at the ranks used "
                       "→ the subspace under test is noise, not the map. Fix and re-run; the only "
                       "legitimate halt (spec §1.4/§6.1).")
        return r
    # 2. VOID-no-range — the benign boundary (a reported truth, not a halt).
    if (not forget_range_ok) or (not removable_range_ok):
        r["verdict"] = "VOID-no-range"
        r["reason"] = ("the natural fine-tune does not disturb pretraining enough to have removable "
                       "collateral (no forgetting range and/or no removable range) → the protective "
                       "intervention has no room to move; re-pick tasks (spec §1.4).")
        return r
    # 3. saturation guard — if ANY subspace protects, a map win is not what we are looking at.
    if null_saturates:
        r["verdict"] = "H_subspace-agnostic-protection"
        r["reason"] = ("off-random already removes the forgetting at no task cost → protection is "
                       "subspace-indifferent; this is a property of removing ANY structured "
                       "subspace, not of the susceptibility map (spec §6 saturation guard).")
        return r
    # 3b. INSUFFICIENT-primary — the frozen §6 branches below all read "E_protect CI-lo ≤ 0", which
    # PRESUPPOSES a CI exists. When the primary was never estimated (too few surviving paired cells),
    # that precondition is unmet and no branch applies. Without this, `clears_floor(insufficient)` =
    # False silently satisfies branch 4 and the study reports **H_diagnostic-only … airtight** on a
    # measurement that was never made — a verdict of ABSENCE manufactured from MISSING EVIDENCE.
    # This is not a post-unblind rule swap: it moves no threshold and favours no hypothesis; it
    # WITHHOLDS a verdict the frozen rule never licensed (`feedback_signals_that_cannot_fail`).
    if (E_protect or {}).get("insufficient"):
        r["verdict"] = "INSUFFICIENT-primary"
        r["reason"] = (
            f"E_protect was not estimable (n={(E_protect or {}).get('n')} paired cells survived the "
            "range control; the estimator needs ≥3) — so §6's branches, which all read 'E_protect "
            "CI-lo ≤ 0', have no CI to read. The primary is UNDECIDED: this is missing evidence, not "
            "evidence of absence, and it must not be reported as H_diagnostic-only. Report the point "
            "estimates and the axes that did land (mechanism, targeting, range), and say plainly that "
            "the sealed primary is underpowered.")
        return r
    # 4. H_diagnostic-only — neither prescriptive reading survives its null. `clears_floor` treats an
    # insufficient contrast (ci_lo None) as NOT clearing: missing evidence is never evidence.
    e_clears = clears_floor(E_protect)
    protective = e_clears and clears_floor(specificity) and bool(mechanism_confirms)
    if (not e_clears) and (not targeting_wins):
        r["verdict"] = "H_diagnostic-only"
        r["reason"] = ("off-sus is no better than off-random at equal task cost (E_protect ci_lo ≤ 0) "
                       "AND writing into U_sus does not beat U_energy → the map predicts damage but "
                       "prescribes nothing, protectively or by targeting. The mission's practical "
                       "leg is closed, airtight (spec §6.3).")
        return r
    # 5. H_targeting — would overturn STENCIL; reconcile explicitly.
    if targeting_wins:
        r["verdict"] = "H_targeting"
        r["reason"] = ("writing INTO U_sus beats U_energy on iso-forgetting task-gain → the map "
                       "prescribes where to write. This CONTRADICTS STENCIL R2 (into-Q learned "
                       "worst) and must be reconciled against it before it is asserted (spec §6.4).")
        return r
    # 6. H_protective — the practical gate.
    if protective:
        r["verdict"] = "H_protective"
        r["reason"] = ("removing the map's directions cuts more forgetting than removing a random "
                       "subspace at equal task cost (E_protect ci_lo > 0), the win is specific to "
                       "susceptibility rather than to any structured subspace (specificity ci_lo > "
                       "0), and the task-gradient mass is measured in U_energy not U_sus → the map "
                       "CLEAVES adaptation: project off U_sus to keep the task and shed the "
                       "forgetting. The practical gate is earned (spec §6.5).")
        return r
    # E_protect cleared its floor but a corroborating leg did not — the map does not cleave cleanly.
    r["verdict"] = "H_diagnostic-only"
    r["reason"] = ("E_protect cleared its floor but the corroborating legs did not (specificity "
                   f"ci_lo={specificity.get('ci_lo')} and/or the mechanism census does not put the "
                   "task gradient in U_energy) → the forgetting-reduction is not specific to the "
                   "susceptibility map; the map remains diagnostic (spec §6.3, §1.4 falsification).")
    return r
