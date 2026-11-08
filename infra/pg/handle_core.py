"""HANDLE core — the compact-core ladder statistics and every §6 adjudicator (handle spec
§2/§6; §0.5 pre-freeze amendments H-A*).

The object of HANDLE: does a SMALL core inside CARRY's 679-dim stiff space carry most of the
direction-specific gain, and is the Q-eigenvalue ORDERING load-bearing? The primary statistic is
`excess_k = gain_frac(core_k) − band_k` — null-referenced by construction (the CARRY R-C
rule-shape lesson, now law): every decision statistic in this module is measured against a
matched-random band, never as a raw fraction.

Two matched-random bands (spec Glossary):
  * band_k        — max over 3 seeds of gain_frac for a rank-k random subset of Q's OWN d
                    eigvectors (the CARRY matched-rank null, per rung).
  * band_k^stiff  — same, but the k directions are drawn from inside the top-k_stiff (679-dim)
                    stiff space — the control that SEPARATES "top-of-ordering matters"
                    (H_compact) from "any stiff directions suffice" (H_set-property).

Everything here is weight-free math — D-suite-tested on synthetic ladders before any GPU minute.
Heavy imports inside functions so the module py_compiles on the control plane. No try/except
hiding — a wrong assumption RAISES (CLAUDE.md). Rank/correlation degenerate completions are the
standing SPECTER-E16 law (all-tied := 0; n < 3 → VOID); the ladder-statistic completions
(k_half on a non-carrying stiff install; single-rung saturation) are named as H-A amendments and
logged in the journal.

Reused verbatim from carry_core (NOT re-derived): random_eigsplit (the band draw),
paired_gain / cider_paired_gain (the gain instrument + its construction-matched SE),
range_guard_ok, gain_frac, spearman0 / kendall0 / shuffle_null_rho (the degenerate-safe rank
stats). Reused from comwrite_surgery: project_delta / build_decoder (the swap-shell). Reused
from bispec_core: principal_angles / haar_frames / grassmann_diameter. Reused from
comwrite_core: subspace_overlap. Q eigvectors are LOADED from carry_scope, never re-derived
(spec §0.5-1).
"""
from __future__ import annotations

import math

# ---- frozen constants (spec §2/§6; §0.5) --------------------------------------------------
K_LADDER_FIXED = (5, 10, 20, 50, 150, 350)   # the top rung (= k_stiff) is appended at runtime
PRIMARY_RUNG = 50               # excess_50 — the pre-named null-referenced primary statistic
KHALF_COMPACT = 50              # median k_half ≤ 50 ⇒ H_compact side
KHALF_BROAD = 150               # median k_half > 150 ⇒ H_ordered-broad side
N_RAND_SEEDS = 3                # band_k / band_k^stiff seeds per rung (max over seeds = the band)
N_BOOT = 10_000                 # FT-level pair-bootstrap draws
BH_Q = 0.10
N_DISCOVERY_FTS = 6
N_SEALED_FTS = 12               # the CARRY-scored sealed roster (§2 population)
VOID_MAX_SEALED = 6             # > 6 of 12 sealed FTs range-VOID ⇒ VOID-range verdict (§2)
ORDERED_BROAD_MIN_RUNGS = 3     # excess vs band^stiff CI-lo > 0 at ≥ 3 rungs (R-H)

# R-T (the demonstration)
KSTAR_SETPROP = 50              # K* under H_set-property (pre-specified, §6 R-T)
RT_N_PAIRS = 30                 # 6×5 ordered discovery pairs
RT_MODULE_WIN = 27             # ≥ 27/30 own-beats-other for H_modules
RT_MODULE_ALPHA = 0.01
RT_DIAG_SHARED_MAX = 18         # diag dominance ≤ 18/30 ⇒ shared/neither side
RT_DELETION_MULT = 3.0          # own-task drop ≥ 3× the ppl/non-target drop
RT_VOID_MAX_DISC = 2            # > 2 of 6 discovery FTs range-VOID on cross-cells ⇒ VOID

# R-Q (the benignity mechanism race)
RQ_RHO_WIN = 0.7                # dependence fraction wins the race at ρ ≥ 0.7
RQ_SHUFFLE_MAX = 0.3            # label-shuffle p-value ≤ 0.3
RQ_MULTI_MARGIN = 0.15          # H_multi-factor: ≥ 2 clear with no winner by ≥ 0.15 ρ margin
RQ_DELTA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # the δ dependence-fraction ladder

# R-N-a (carrier round 2) / R-N-b (L0 spikes)
RNA_TAU_WIN = 0.6              # candidate wins at τ ≥ 0.6
RNA_P_MAX = 0.05
RNB_IDENT_MULT = 3.0           # census identification at ≥ 3× the matched-rank Haar p95
RNB_RAYLEIGH_DROP = 0.5        # the causal cell: spike Rayleigh must drop ≥ 50 %

# R-B (the witness set as an object)
RB_DEFF_MAX = 10               # d_eff ≤ 10 ⇒ low-dim surface side
RB_ANGLE_TOL = 0.5             # "angles ≪ Haar": mean-min principal angle ≤ tol × the Haar band


# ===========================================================================
# Ladder construction — nested top-k slices + the two matched-random bands.
# ===========================================================================
def ladder_ks(k_stiff):
    """The 7 rungs: the fixed low rungs strictly below k_stiff, then k_stiff itself (the full
    stiff space). Nested + strictly ascending by construction."""
    ks = [k for k in K_LADDER_FIXED if k < int(k_stiff)] + [int(k_stiff)]
    assert ks == sorted(set(ks)) and ks[-1] == int(k_stiff), f"degenerate ladder {ks}"
    assert len(ks) >= 3, f"ladder too short (k_stiff={k_stiff})"
    return tuple(ks)


def core_basis(evecs, k):
    """core_k basis = the top-k Q-eigvector slice (nested: k ⊂ k' for k < k')."""
    import torch
    E = torch.as_tensor(evecs)
    assert 1 <= int(k) <= E.shape[1]
    return E[:, :int(k)].contiguous()


def band_basis(evecs, k, seed, within_stiff=None):
    """A matched-rank random band basis of the SAME Q eigenspace.
      within_stiff=None  → band_k      : k random columns from ALL d eigvectors.
      within_stiff=k_st  → band_k^stiff: k random columns from the first k_st (stiff) eigvectors.
    Delegates to carry_core.random_eigsplit (the frozen CARRY null draw)."""
    from infra.pg import carry_core as CC
    import torch
    E = torch.as_tensor(evecs)
    pool = E if within_stiff is None else E[:, :int(within_stiff)].contiguous()
    assert int(k) <= pool.shape[1], f"band rank {k} exceeds pool width {pool.shape[1]}"
    if int(k) == pool.shape[1]:
        # random k-of-k = the full pool (the only possible subset) — the degenerate top rung
        # of band^stiff; return it directly (random_eigsplit requires k < d strictly).
        return pool, list(range(pool.shape[1]))
    U, idx = CC.random_eigsplit(pool, int(k), int(seed))
    return U, idx


def assert_nesting(evecs, ks, k_stiff, stiff_projector=None, tol=1e-5):
    """Projector-algebra guard (spec §0.5-2): P_k ⊂ P_{k'} for k < k' (nested), and
    P_{k_stiff} ≡ CARRY's stiff projector (bit-for-bit up to fp tolerance if supplied).
    Returns the per-pair max containment error. tol=1e-5 accommodates the fp32-banked Q
    eigvectors (orthonormality good to ~fp32 precision); a genuine non-nesting is O(1)."""
    import torch
    E = torch.as_tensor(evecs, dtype=torch.float64)
    errs = []
    for a, b in zip(ks[:-1], ks[1:]):
        assert a < b, f"ladder not ascending at ({a},{b})"
        Pa = E[:, :a] @ E[:, :a].transpose(0, 1)
        Pb = E[:, :b] @ E[:, :b].transpose(0, 1)
        # nested ⇔ Pb Pa = Pa (the smaller projector is absorbed by the larger)
        errs.append(float((Pb @ Pa - Pa).abs().max()))
    max_nest = max(errs) if errs else 0.0
    assert max_nest <= tol, f"nesting violated: max |P_b P_a − P_a| = {max_nest:.2e}"
    out = {"max_nest_err": max_nest, "ks": list(ks)}
    if stiff_projector is not None:
        Pks = E[:, :int(k_stiff)] @ E[:, :int(k_stiff)].transpose(0, 1)
        S = torch.as_tensor(stiff_projector, dtype=torch.float64)
        out["stiff_proj_err"] = float((Pks - S).abs().max())
        assert out["stiff_proj_err"] <= 1e-5, \
            f"P_k_stiff ≠ CARRY stiff projector: {out['stiff_proj_err']:.2e}"
    return out


# ===========================================================================
# Ladder statistics — k_half, excess, and the FT-level median CIs.
# ===========================================================================
def k_half(gf_by_k, ks):
    """smallest k on the ladder with gain_frac(core_k) ≥ 0.5·gain_frac(core_{k_top}) (spec
    Glossary). k_top = ks[-1] = k_stiff.

    H-A1 (pre-freeze degenerate completion, dated 2026-07-13): if gain_frac(core_{k_top}) ≤ 0 the
    stiff install does not carry on this FT, so 'half of the final gain' has no positive target
    and k_half is UNDEFINED for that FT → returns None. The R-H adjudicator medians over the
    FTs with a defined k_half and counts the None's separately (never imputes a value that would
    bias the compact/broad call)."""
    top = float(gf_by_k[ks[-1]])
    if top <= 0.0:
        return None
    target = 0.5 * top
    for k in ks:                                # ascending
        if float(gf_by_k[k]) >= target:
            return int(k)
    return int(ks[-1])                          # reaches half only at the top rung


def excess_at(core_by_k, band_by_k, k):
    """excess_k = gain_frac(core_k) − band_k at one rung (the null-referenced primary)."""
    return float(core_by_k[k]) - float(band_by_k[k])


def median_ci(vals, n_boot=N_BOOT, seed=1337):
    """FT-level bootstrap of a median. < 3 valid values ⇒ VOID (the frozen §2 completion)."""
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], float)
    if v.size < 3:
        return {"void": True, "n": int(v.size),
                "reading": "median over < 3 valid FTs ⇒ VOID, not a CI (§2)"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.median(v[idx], axis=1)
    return {"void": False, "median": float(np.median(v)), "n": int(v.size),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def ladder_frame(rows, ks):
    """Per-rung median-excess CIs vs BOTH bands, plus per-FT k_half, from the sealed rows.

    `rows` = per-scored-FT dicts:
        {ft, core:{k:gf}, band:{k:gf_max_over_seeds}, band_stiff:{k:gf_max_over_seeds}}
    (band/band_stiff already reduced to the max over the 3 seeds = the band value per rung).

    Returns {per_rung:{k:{excess_band:med_ci, excess_stiff:med_ci}}, k_half:{ft:kh},
             median_k_half:med_ci_over_fts}."""
    per_rung = {}
    for k in ks:
        eb = [excess_at(r["core"], r["band"], k) for r in rows]
        es = [excess_at(r["core"], r["band_stiff"], k) for r in rows]
        per_rung[k] = {"excess_band": median_ci(eb), "excess_stiff": median_ci(es),
                       "n": len(rows)}
    khs = {r["ft"]: k_half(r["core"], ks) for r in rows}
    return {"per_rung": per_rung, "k_half": khs,
            "median_k_half": median_ci([v for v in khs.values()]),
            "n_khalf_void": sum(1 for v in khs.values() if v is None)}


# ===========================================================================
# R-H — the primary adjudicator (guards FIRST; every branch constructible).
# ===========================================================================
def _cleared(mci):
    """excess CI-lo > 0 (a null-referenced clearance), False on VOID."""
    return (not mci.get("void", False)) and mci["ci95"][0] > 0.0


def _straddles(mci):
    """excess CI straddles 0 (CI-lo ≤ 0 ≤ CI-hi); VOID straddles by convention (no evidence)."""
    if mci.get("void", False):
        return True
    return mci["ci95"][0] <= 0.0 <= mci["ci95"][1]


def adjudicate_rh(rows, *, ks, k_stiff, n_sealed_total, primary_rung=PRIMARY_RUNG):
    """The MDE verdict. `rows` = per-range-passing sealed FT (see ladder_frame). n_void =
    n_sealed_total − len(rows).

    Branch order (frozen, guards first): VOID-range → H_carry-fails-to-redetect → H_compact →
    H_ordered-broad → H_set-property → H_partial. Every branch carries a ceiling: H_compact needs
    BOTH bands cleared AND a small median k_half; the parent-replication failure (H_carry-fails)
    is reachable; the middle branches are killable on the band^stiff CI; the fallthrough is the
    honest remainder — no branch is a floor without a ceiling (the one-sided-thresholds law)."""
    n_void = int(n_sealed_total) - len(rows)
    assert n_void >= 0, "more scored rows than sealed FTs"
    if n_void > VOID_MAX_SEALED:
        return {"verdict": "VOID-range", "n_void": n_void, "n_scored": len(rows),
                "reading": "> 6 of 12 sealed FTs range-VOID under the handle| salt — the "
                           "swap-shell's range on this corpus IS the finding (§2)"}
    assert primary_rung in ks, f"primary rung {primary_rung} not on the ladder {ks}"
    fr = ladder_frame(rows, ks)
    top = ks[-1]
    ex50_band = fr["per_rung"][primary_rung]["excess_band"]
    ex50_stiff = fr["per_rung"][primary_rung]["excess_stiff"]
    extop_band = fr["per_rung"][top]["excess_band"]
    mkh = fr["median_k_half"]
    n_stiff_cleared = sum(1 for k in ks if _cleared(fr["per_rung"][k]["excess_stiff"]))
    all_stiff_straddle = all(_straddles(fr["per_rung"][k]["excess_stiff"]) for k in ks)
    out = {"n_void": n_void, "n_scored": len(rows), "n_khalf_void": fr["n_khalf_void"],
           "median_k_half": mkh, "excess_50_vs_band": ex50_band,
           "excess_50_vs_band_stiff": ex50_stiff, "excess_top_vs_band": extop_band,
           "n_rungs_stiff_cleared": n_stiff_cleared, "per_rung": fr["per_rung"],
           "k_half": fr["k_half"]}

    # guard: does CARRY's substructure even reproduce under the fresh salt? (senior event)
    if _straddles(extop_band):
        return {**out, "verdict": "H_carry-fails-to-redetect",
                "reading": "excess_{k_stiff} vs band straddles 0 — the CARRY stiff substructure "
                           "does not reproduce under the handle| salt; a replication event "
                           "senior to every ladder question, reported first, ladder descriptive"}

    kh_ok = (not mkh.get("void", False)) and mkh["median"] <= KHALF_COMPACT
    if kh_ok and _cleared(ex50_band) and _cleared(ex50_stiff):
        return {**out, "verdict": "H_compact",
                "reading": "the map is a handle: a ≤ 50-direction core carries the majority of "
                           "direction-specific skill and the Q-ordering is load-bearing "
                           "(excess_50 clears BOTH bands); Arm T adjudicates the practical leg"}

    kh_broad = (not mkh.get("void", False)) and mkh["median"] > KHALF_BROAD
    if n_stiff_cleared >= ORDERED_BROAD_MIN_RUNGS and kh_broad:
        return {**out, "verdict": "H_ordered-broad",
                "reading": "the Q-ordering is functionally real (band^stiff cleared at ≥ 3 "
                           "rungs) but storage is graded (median k_half > 150); the gain-vs-k "
                           "profile is the named object"}

    if all_stiff_straddle and _cleared(extop_band):
        return {**out, "verdict": "H_set-property",
                "reading": "stiffness is a SET property: top-k never beats band^stiff while the "
                           "stiff space still beats random-rank (band cleared at the top) — a "
                           "second level-set-not-coordinates law; the functional ordering, if "
                           "any, lives off the Q-spectrum"}

    return {**out, "verdict": "H_partial",
            "reading": "a mixed landing outside every frozen branch's full condition — reported "
                       "with the per-FT gain-vs-k table; the closest branch's missing clause is "
                       "named in the result doc"}


def select_kstar(rh, ks):
    """K* = smallest rung clearing excess CI-lo > 0 vs BOTH bands (frozen R-T selector). Under
    H_set-property, K* := 50 (pre-specified). None if no rung clears both (⇒ R-T runs at 50 with
    the set-property module definition)."""
    if rh["verdict"] == "H_set-property":
        return KSTAR_SETPROP
    pr_all = rh["per_rung"]
    for k in ks:
        # robust to int keys (fresh in-memory rh) AND string keys (rh re-loaded from JSON)
        pr = pr_all.get(k, pr_all.get(str(k)))
        if _cleared(pr["excess_band"]) and _cleared(pr["excess_stiff"]):
            return int(k)
    return KSTAR_SETPROP


# ===========================================================================
# R-T — the demonstration: module matrix, deletion selectivity, core census.
# ===========================================================================
def binom_p_ge(k, n, p=0.5):
    """One-sided exact binomial P[X ≥ k] under Binomial(n, p) — the module-dominance test."""
    from math import comb
    k, n = int(k), int(n)
    assert 0 <= k <= n
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def diag_dominance_count(M):
    """M[i][j] = gain of core_i installed on task_j (6×6). Count of ordered pairs (i,j), i≠j,
    where the OWN core beats the OTHER core on the own task i: M[i][i] > M[j][i]. Range 0..30."""
    import numpy as np
    A = np.asarray(M, float)
    n = A.shape[0]
    assert A.shape == (n, n), "module matrix must be square"
    c = 0
    for i in range(n):
        for j in range(n):
            if i != j and A[i, i] > A[j, i]:
                c += 1
    return int(c), int(n * (n - 1))


def adjudicate_rt(*, module_M, deletion_rows, census, n_disc_void):
    """R-T verdict (family = {diag-dominance, deletion-selectivity}, spec §6).
      module_M      : 6×6 gains M[i][j] = core_i on task_j at K*.
      deletion_rows : per-FT {own_drop, ppl_drop, nontarget_drop} for FT_i − core_i on task_i.
      census        : {cross_overlap_med, haar_p95, core_uwr_overlap, core_uwr_haar_p95, ...}.
      n_disc_void   : discovery FTs that range-VOID on cross-cells.
    Guards first (VOID), then H_modules / H_shared-substrate / H_neither."""
    if n_disc_void > RT_VOID_MAX_DISC:
        return {"verdict": "VOID", "n_disc_void": int(n_disc_void),
                "reading": "> 2 of 6 discovery FTs range-VOID on the cross-install shell — the "
                           "shell's cross range is the finding"}
    diag, n_pairs = diag_dominance_count(module_M)
    assert n_pairs == RT_N_PAIRS, f"expected {RT_N_PAIRS} ordered pairs, got {n_pairs}"
    p_diag = binom_p_ge(diag, n_pairs, 0.5)
    # deletion selectivity: own-task drop ≥ 3× the max of {ppl, non-target} drop, per FT; the
    # demonstration needs the MEDIAN FT selective (a ceiling — a single lucky FT cannot carry it).
    import numpy as np
    sel = [row["own_drop"] >= RT_DELETION_MULT * max(row["ppl_drop"], row["nontarget_drop"])
           for row in deletion_rows]
    n_sel = int(sum(sel))
    median_selective = n_sel > len(deletion_rows) / 2.0
    cross_over_haar = census["cross_overlap_med"] > census["haar_p95"]
    out = {"diag_dominance": diag, "n_pairs": n_pairs, "p_diag": p_diag,
           "n_deletion_selective": n_sel, "n_deletion_fts": len(deletion_rows),
           "cross_overlap_med": census["cross_overlap_med"], "haar_p95": census["haar_p95"]}
    if diag >= RT_MODULE_WIN and p_diag <= RT_MODULE_ALPHA and median_selective:
        return {**out, "verdict": "H_modules",
                "reading": "skills are compact portable modules — own-core beats other-core "
                           f"{diag}/{n_pairs} (p={p_diag:.1e}) and deletion is surgical; with "
                           "H_compact this is the bar-clearing demonstration"}
    if diag <= RT_DIAG_SHARED_MAX and cross_over_haar:
        return {**out, "verdict": "H_shared-substrate",
                "reading": "cross-installs ≈ own-installs and cores overlap ≫ Haar — the cores "
                           "are one shared object (the consensus write with a function); "
                           "COMWRITE's U_wr unifies with CARRY's carrier"}
    if diag <= RT_DIAG_SHARED_MAX:
        return {**out, "verdict": "H_neither",
                "reading": "portability fails and cores are FT-idiosyncratic (census at Haar) — "
                           "skill storage is contextual, the strongest form of CARRY R3's "
                           "non-additivity"}
    return {**out, "verdict": "H_partial-modules",
            "reading": "diagonal dominance clears the count but not both family legs — reported "
                       "with the module heatmap and deletion dumbbells"}


# ===========================================================================
# R-Q — the benignity mechanism race + the δ-dependence dose-response.
# ===========================================================================
def property_race(scores_by_prop, damages, labels, *, n_shuffle=10_000, seed=1337, q=BH_Q):
    """ρ_S(property_score, damage) over the ≥ 12 families for each of the 4 candidate properties,
    each vs its own label-shuffle null; BH over the family. `scores_by_prop` = {prop: [score/family]}.
    Returns per-property {rho, shuffle_p, bh_pass} + the winner analysis."""
    from infra.pg import carry_core as CC
    props = list(scores_by_prop)
    rows, pvals = {}, []
    for pname in props:
        sh = CC.shuffle_null_rho(scores_by_prop[pname], damages, n_shuffle=n_shuffle, seed=seed)
        rows[pname] = {"rho": sh["rho_obs"], "shuffle_p95": sh["p95"], "shuffle_p": sh["p_value"]}
        pvals.append(sh["p_value"])
    from infra.pg.atlas_fit import bh_pass
    passed = bh_pass(pvals, q=q)
    for pname, ok in zip(props, passed):
        rows[pname]["bh_pass"] = bool(ok)
    return {"labels": list(labels), "per_property": rows,
            "p_values": [float(p) for p in pvals]}


def delta_mix(R_own, R_off, delta, energy_match=True):
    """The Arm-Q δ-dependence mixture (spec Glossary): per token, blend δ·(own quantization
    residual) + (1−δ)·(matched off-token construction), then rescale each token row to the OWN
    residual's per-token norm so the per-token ENERGY is matched across rungs (the marginal that
    the D-suite asserts — damage differences must reflect δ, not energy). δ ∈ [0,1]. Returns
    (T,d) fp64."""
    import torch
    A = torch.as_tensor(R_own, dtype=torch.float64)
    B = torch.as_tensor(R_off, dtype=torch.float64)
    assert A.shape == B.shape and A.ndim == 2, "own/off residual shape mismatch"
    E = float(delta) * A + (1.0 - float(delta)) * B
    if energy_match:
        own = A.norm(dim=1, keepdim=True)
        E = E * (own / E.norm(dim=1, keepdim=True).clamp_min(1e-30))
    return E


def rotate_offtoken(R_off, seed):
    """The δ-ladder rotated twin: a fixed Haar rotation of the off-token component — destroys its
    DIRECTION, preserves each token's norm (so the dose-response must survive it only if the trend
    rides direction, not energy)."""
    import torch
    B = torch.as_tensor(R_off, dtype=torch.float64)
    d = B.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    return B @ Q.transpose(0, 1).to(B.device)   # Q is CPU-generated (RNG); move to B's device


def monotone_negative(deltas, damages, tol=0.0):
    """The δ-ladder shape test: strictly non-increasing in δ beyond `tol` (no interior reversal).
    Returns {tau, monotone_negative} with Kendall τ(δ, damage) (degenerate-safe)."""
    from infra.pg import carry_core as CC
    import numpy as np
    d = np.asarray(deltas, float)
    y = np.asarray(damages, float)
    order = np.argsort(d)
    ys = y[order]
    reversals = [float(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
    mono = all(r <= tol for r in reversals)          # non-increasing (more dependence ⇒ less damage)
    tau = CC.kendall0(list(d), list(y))
    return {"tau": float(tau), "monotone_nonincreasing": bool(mono),
            "max_interior_rise": float(max(reversals)) if reversals else 0.0}


def adjudicate_rq(*, race, delta_ladder, delta_ladder_twin, damage_span,
                  dep_prop="dependence_fraction"):
    """R-Q verdict. `race` from property_race; `delta_ladder`/`delta_ladder_twin` from
    monotone_negative on the real and rotated-twin mixtures; damage_span = max/min positive
    family damage (the range guard). Guards first (VOID-no-range), then the four branches."""
    if damage_span < 10.0:
        return {"verdict": "VOID-no-range", "damage_span": float(damage_span),
                "reading": "family damages span < 10× — the battery cannot rank; the amplitude "
                           "choice is the finding"}
    rows = race["per_property"]
    dep = rows[dep_prop]
    cleared = {p: (r["rho"] >= RQ_RHO_WIN and r["shuffle_p"] <= RQ_SHUFFLE_MAX and r["bh_pass"])
               for p, r in rows.items()}
    n_clear = sum(cleared.values())
    twin_broken = (delta_ladder["monotone_nonincreasing"] and delta_ladder["tau"] < 0
                   and not (delta_ladder_twin["monotone_nonincreasing"]
                            and delta_ladder_twin["tau"] < 0))
    ladder_causal = delta_ladder["monotone_nonincreasing"] and delta_ladder["tau"] < 0 and twin_broken
    out = {"cleared": cleared, "n_clear": n_clear, "delta_tau": delta_ladder["tau"],
           "delta_monotone": delta_ladder["monotone_nonincreasing"], "twin_broken": twin_broken,
           "ladder_causal": ladder_causal}
    if cleared.get(dep_prop) and ladder_causal:
        return {**out, "verdict": "H_dependence-mechanism",
                "reading": "the A9 error–token dependence axis IS the mechanism: benignity is "
                           "fine-grained dependence, dose-responsive and direction-specific "
                           "(rotated twin breaks it) — a causal, usable compression-scheme screen"}
    winners = sorted((r["rho"], p) for p, r in rows.items() if cleared[p])
    if n_clear == 1:
        p = winners[-1][1]
        return {**out, "verdict": f"H_wins:{p}",
                "reading": f"the mechanism is the {p} property with its own practical screen"}
    if n_clear >= 2:
        top2 = [rho for rho, _ in winners[-2:]]
        if abs(top2[-1] - top2[0]) < RQ_MULTI_MARGIN:
            return {**out, "verdict": "H_multi-factor",
                    "reading": "benignity is a composite (≥ 2 clear, no ρ-winner by ≥ 0.15); the "
                               "fitted shares + residuals are the object; the ladder says which "
                               "factor is causal"}
        p = winners[-1][1]
        return {**out, "verdict": f"H_wins:{p}",
                "reading": f"the {p} property wins the race by margin ≥ 0.15"}
    return {**out, "verdict": "H_none-ranks",
            "reading": "the benignity carrier escapes all four named candidates — quadruply "
                       "instrumented now; the per-family residual table is the successor's map"}


# ===========================================================================
# R-N-a (carrier round 2) / R-N-b (L0 spikes).
# ===========================================================================
def adjudicate_rna(candidate_stats, *, q=BH_Q):
    """R-N-a: `candidate_stats` = {name:{tau, p}} over the 5 candidates (exact-permutation p).
    BH over 5; a winner (τ ≥ 0.6, p ≤ 0.05, BH-pass) NAMES the coherent-spread carrier."""
    from infra.pg.atlas_fit import bh_pass
    names = list(candidate_stats)
    pvals = [candidate_stats[n]["p"] for n in names]
    passed = bh_pass(pvals, q=q)
    winners = [n for n, ok in zip(names, passed)
               if ok and candidate_stats[n]["tau"] >= RNA_TAU_WIN
               and candidate_stats[n]["p"] <= RNA_P_MAX]
    out = {"candidates": {n: {**candidate_stats[n], "bh_pass": bool(ok)}
                          for n, ok in zip(names, passed)}, "winners": winners}
    if winners:
        best = max(winners, key=lambda n: candidate_stats[n]["tau"])
        return {**out, "verdict": "H_carrier-named", "carrier": best,
                "reading": f"the 110× spread carrier is NAMED after four programs: {best}"}
    return {**out, "verdict": "H_carrier-escapes",
            "reading": "all five candidates miss — the carrier is formally outside every "
                       "first-order structure the lineage has measured; the per-item (field) "
                       "hypothesis is promoted and five more exclusions bank"}


def adjudicate_rnb(library_overlaps, haar_p95_by_elem, *, rayleigh_drop=None):
    """R-N-b: `library_overlaps` = {elem:max_overlap}; identified iff any elem ≥ 3× its Haar p95.
    If identified AND the causal cell dropped the spike Rayleigh ≥ 50 %, the identification
    becomes a mechanism; else the spikes ship with the 'genuinely new object' certificate."""
    hits = {e: (v, haar_p95_by_elem[e], v / haar_p95_by_elem[e] if haar_p95_by_elem[e] > 0 else 0.0)
            for e, v in library_overlaps.items()}
    identified = [e for e, (_v, _p, r) in hits.items() if r >= RNB_IDENT_MULT]
    out = {"overlaps": {e: {"max_overlap": v, "haar_p95": haar_p95_by_elem[e], "ratio": r}
                        for e, (v, _p, r) in hits.items()}, "identified": identified}
    if not identified:
        return {**out, "verdict": "H_spikes-new-object",
                "reading": "the L0 spikes match nothing in the weight-side OR vision-side "
                           "library at ≥ 3× Haar — the eigvec bank ships with the strongest "
                           "'genuinely new object' certificate"}
    if rayleigh_drop is not None and rayleigh_drop >= RNB_RAYLEIGH_DROP:
        best = max(identified, key=lambda e: hits[e][2])
        return {**out, "verdict": "H_spikes-identified-causal", "structure": best,
                "rayleigh_drop": float(rayleigh_drop),
                "reading": f"the spikes ARE {best} — ablating it drops the spike Rayleigh "
                           f"{rayleigh_drop:.0%} (≥ 50 %), converting identification to mechanism"}
    return {**out, "verdict": "H_spikes-identified-descriptive",
            "rayleigh_drop": None if rayleigh_drop is None else float(rayleigh_drop),
            "reading": "the spikes overlap a library element at ≥ 3× Haar but the causal cell "
                       "did not clear the 50 % Rayleigh-drop bar — identification reported "
                       "descriptively"}


# ===========================================================================
# R-B — the witness set as an object.
# ===========================================================================
def adjudicate_rb(*, d_eff, mean_min_angle, haar_mean_min_angle, overlap_census,
                  install_rho, install_shuffle_p):
    """R-B verdict. d_eff of the witness set (2/CV² on its damage family); the principal-angle
    dispersion vs the matched-rank Haar band; the overlap census (BH within a family); and the
    install-tracking ρ_S over 8 probes (reported EITHER way — its positive branch links the
    install phenomenon to the same structure, its null localizes installability elsewhere)."""
    angles_much_less = mean_min_angle <= RB_ANGLE_TOL * haar_mean_min_angle
    out = {"d_eff": float(d_eff), "mean_min_angle": float(mean_min_angle),
           "haar_mean_min_angle": float(haar_mean_min_angle),
           "angles_much_less_than_haar": bool(angles_much_less),
           "overlap_census": overlap_census,
           "install_tracking": {"rho": float(install_rho), "shuffle_p": float(install_shuffle_p),
                                "tracks": bool(install_shuffle_p <= 0.05 and install_rho > 0)}}
    if d_eff <= RB_DEFF_MAX and angles_much_less:
        return {**out, "verdict": "H_low-dim-surface",
                "reading": "the witness set is a compact named object (d_eff ≤ 10, angles ≪ "
                           "Haar) — the control-surface reading; install-tracking reported"}
    if not angles_much_less:
        return {**out, "verdict": "H_scattered",
                "reading": "the worst cases are direction-idiosyncratic (angles ≈ Haar) — bounds "
                           "every compact-boundary story; install-tracking localizes elsewhere"}
    return {**out, "verdict": "H_mid",
            "reading": "a mixed landing (one of d_eff / angle criteria clears) — reported with "
                       "the angle spectrum and the overlap census"}


# ===========================================================================
# R-X — verdict-level replication on model 2 (PG2). Verdicts only; no aggregate score.
# ===========================================================================
def adjudicate_rx(law_verdicts):
    """`law_verdicts` = {law:{pg1, pg2, void}} with pg1/pg2 the branch labels each rule landed on
    PG1/PG2 (both PG2-calibrated on PG2 control data — the firewall). Per law:
    replicates / lands-elsewhere / VOID. The row vector IS the result (no aggregate)."""
    out = {}
    for law, v in law_verdicts.items():
        if v.get("void"):
            out[law] = "VOID"
        elif v["pg1"] == v["pg2"]:
            out[law] = "replicates"
        else:
            out[law] = "lands-elsewhere"
    lands = [l for l, r in out.items() if r == "lands-elsewhere"]
    return {"per_law": out, "n_lands_elsewhere": len(lands), "lands_elsewhere": lands,
            "reading": "each row is informative; a lands-elsewhere row names the first measured "
                       "scope boundary of that law across models — the external-validity datum "
                       "the lineage lacks. No aggregate score; the row vector is the result"}
