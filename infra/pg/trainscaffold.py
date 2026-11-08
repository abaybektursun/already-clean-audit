"""TRAINSCAFFOLD — the trainability-scaffold science core (pure, importable, weight-free).

No model, no datasets, no GPU: small-tensor numpy/torch math, laptop-unit-tested
(infra/pg/tests/test_trainscaffold.py) BEFORE any GPU minute. The jobs
(infra/aws_pg/jobs/trainscaffold_*.py) are thin orchestration around these functions plus the
REUSED LATMECH ablation + RELAW LM/full-FT forward + relearn/eval spine (§13 ledger).

Prereg: studies/lineage/paligemma2/research/2026-06-28_trainability_scaffold_prereg.md.

The genuinely-NEW math (everything else is REUSED):

  DIAGNOSTIC (Arm-0, DISCOVERY-only):
    * grad_in_removed_dyad_fraction — fraction of a full-weight gradient's energy in the REMOVED
      rank-k DIAGONAL dyads {u_i v_iᵀ : i∈idx} of a matrix (matched to latmech.project_out_subspace,
      which removes Σ σ_i u_i v_iᵀ). = Σ_{i∈idx}(u_iᵀ g v_i)² / ‖g‖². The job reports the
      emrand-NORMALIZED contrast (raw fraction is consistent with both capacity and severance).

  PRIMARY estimator + frozen verdict predicates (§6, PINNED margins):
    * rf_novel_joint_boot — RF_novel = (acc_cell − acc_R)/(acc_B − acc_R), items resampled JOINTLY
      across the {cell, R floor, B ceiling} arrays so B/R correlated noise propagates. Returns
      point + 95% + 90% CIs.
    * generalizes        — RF_novel 95%-CI lower > 0.5 AND beats its shuffle twin by ≥ 0.2.
    * at_floor_tost      — TOST equivalence to the R floor: 90% CI ⊂ [−0.2, +0.2].
    * graded             — neither generalizes nor at-floor.
    * reachability_verdict — H-severance / H-capacity / graded on the refill-blocked full-FT RF_MCQ.
    * trainscaffold_verdict — the §6 hierarchical-gatekeeping 4-way verdict on a banked cell dict.

Design rules (CLAUDE.md): no try/except hiding; a wrong assumption RAISES. Heavy imports inside
functions so the module py_compiles on the control plane (latmech/relaw/w15_carrier house style).
"""
from __future__ import annotations

# ---- PINNED §6 margins (the contract; anchored RF_novel scale: R floor = 0, B ceiling = 1) ----
M_LEARN = 0.5          # RF_novel 95%-CI lower bound must exceed this to GENERALIZE
TOST_BAND = 0.2        # AT-FLOOR iff the 90% CI ⊂ [−TOST_BAND, +TOST_BAND]
SHUFFLE_MARGIN = 0.2   # real RF_novel must beat its shuffle twin by ≥ this
S_FMT = 0.3            # min |RF_novel(A_free) − RF_novel(A_letter)| to call H-readout vs H-capability
GO_NOGO_B = 0.55       # acc_B (held-out, absolute) GO/NO-GO threshold
FLOOR_BAND_R = 0.10    # acc_R ≤ chance + this (the dead backbone near chance)
MIN_DYNRANGE = 0.25    # (acc_B − acc_R) ≥ this, else RF_novel UNDERPOWERED
REACH_SEVER_HI = 0.15  # refill-blocked full-FT RF_MCQ 95%-CI upper ≤ this ⇒ severance band
REACH_CTRL_CEIL = 0.6  # non-ablated full-FT control must reach RF_MCQ ≥ this for a valid severance call
REACH_CAPACITY = 0.2   # route-around/refill RF_MCQ 95%-CI lower > this above floor ⇒ capacity


# ===========================================================================
# Arm-0 diagnostic — gradient energy in the REMOVED diagonal dyads.
# ===========================================================================

def grad_in_removed_dyad_fraction(g, U, V, idx):
    """Fraction of g's (out,in) energy in the removed diagonal dyads {u_i v_iᵀ : i∈idx}, where the
    coefficient along u_i v_iᵀ is u_iᵀ g v_i (the dyads are orthonormal in the trace inner product,
    so Σ over ALL i = ‖g‖²; the removed subset gives a value in [0,1]). U:(m,k) V:(n,k) from the
    BANKED carrier (svd_lowrank factors), idx the removed indices. fp64."""
    import torch
    g = torch.as_tensor(g, dtype=torch.float64)
    U = torch.as_tensor(U, dtype=torch.float64).to(g.device)     # carrier loads on CPU; unify to g's device
    V = torch.as_tensor(V, dtype=torch.float64).to(g.device)
    idx = torch.as_tensor(idx, dtype=torch.long).to(g.device)
    if idx.numel() == 0:
        return 0.0
    Ui = U[:, idx]                       # (m, k)
    Vi = V[:, idx]                       # (n, k)
    coeffs = (Ui * (g @ Vi)).sum(dim=0)  # (k,): coeffs[j] = u_jᵀ g v_j
    total = float((g ** 2).sum().clamp_min(1e-300))
    return float((coeffs ** 2).sum() / total)


# ===========================================================================
# PRIMARY estimator — RF_novel with JOINT item bootstrap (floor R, ceiling B).
# ===========================================================================

def rf_novel_joint_boot(acc_cell, acc_R, acc_B, n_boot=10000, seed=1337):
    """RF_novel = (mean(cell) − mean(R)) / (mean(B) − mean(R)); the three per-item 0/1 arrays are
    paired by item and resampled JOINTLY (one item-index draw indexes all three), so the floor/ceiling
    correlated noise propagates into the CI (#10). A draw with mean(B)≈mean(R) (denom ≤ a small ε) is
    dropped + counted. Returns {point, ci95:[lo,hi], ci90:[lo,hi], n_dropped, dyn_range}."""
    import numpy as np
    c = np.asarray(acc_cell, float)
    r = np.asarray(acc_R, float)
    b = np.asarray(acc_B, float)
    assert c.shape == r.shape == b.shape and c.ndim == 1, "RF_novel arrays must be paired per item (1-D, equal length)"
    n = c.size
    denom0 = b.mean() - r.mean()
    point = float((c.mean() - r.mean()) / denom0) if denom0 > 1e-9 else float("nan")
    rng = np.random.default_rng(seed)
    draws, dropped = [], 0
    while len(draws) < n_boot:
        ix = rng.integers(0, n, n)
        denom = b[ix].mean() - r[ix].mean()
        if denom <= 1e-9:
            dropped += 1
            assert dropped < 50 * n_boot, "degenerate RF_novel bootstrap (B≈R — no dynamic range)"
            continue
        draws.append((c[ix].mean() - r[ix].mean()) / denom)
    d = np.asarray(draws)
    return {"point": point,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci90": [float(np.percentile(d, 5.0)), float(np.percentile(d, 95.0))],
            "n_dropped": dropped, "dyn_range": float(denom0)}


# ===========================================================================
# Frozen verdict predicates (§6, PINNED margins).
# ===========================================================================

def generalizes(rf, shuffle_rf_point):
    """GENERALIZES iff RF_novel 95%-CI lower > M_LEARN AND it beats its shuffle twin by ≥ SHUFFLE_MARGIN.
    `rf` = rf_novel_joint_boot output for the real cell; `shuffle_rf_point` = the shuffle twin's RF point."""
    return bool(rf["ci95"][0] > M_LEARN and (rf["point"] - shuffle_rf_point) >= SHUFFLE_MARGIN)


def at_floor_tost(rf):
    """AT-FLOOR (TOST equivalence to the R floor, RF_novel=0) iff the 90% CI ⊂ [−TOST_BAND, +TOST_BAND]
    (the two one-sided tests at 5% ≡ the 90% CI inside the equivalence band). #5 fix: never inferred
    from 'CI brackets 0'."""
    lo, hi = rf["ci90"]
    return bool(lo > -TOST_BAND and hi < TOST_BAND)


def cell_state(rf, shuffle_rf_point):
    """One of 'GENERALIZES' / 'AT_FLOOR' / 'GRADED' (the powered-equivalence escape from null-acceptance)."""
    if generalizes(rf, shuffle_rf_point):
        return "GENERALIZES"
    if at_floor_tost(rf):
        return "AT_FLOOR"
    return "GRADED"


def go_nogo(acc_B, acc_R, chance=0.25):
    """The absolute GO/NO-GO + dynamic-range gate (§6, PINNED). Returns {go, reasons}."""
    import numpy as np
    accB, accR = float(np.mean(acc_B)), float(np.mean(acc_R))
    go = bool(accB >= GO_NOGO_B and accR <= chance + FLOOR_BAND_R and (accB - accR) >= MIN_DYNRANGE)
    return {"go": go, "acc_B": accB, "acc_R": accR,
            "B_ok": accB >= GO_NOGO_B, "R_floor_ok": accR <= chance + FLOOR_BAND_R,
            "dynrange_ok": (accB - accR) >= MIN_DYNRANGE}


def reachability_verdict(refill_blocked_rf_ci, ctrl_fullft_rf_point, route_lift_rf_ci):
    """§6 REACHABILITY rule (PINNED). refill_blocked_rf_ci = [lo,hi] 95% CI of refill-BLOCKED full-FT
    RF_MCQ; ctrl_fullft_rf_point = non-ablated full-FT control RF_MCQ; route_lift_rf_ci = [lo,hi] of the
    best unconstrained route-around/refill lift above floor. Returns the verdict string."""
    severance = (refill_blocked_rf_ci[1] <= REACH_SEVER_HI and ctrl_fullft_rf_point >= REACH_CTRL_CEIL)
    capacity = route_lift_rf_ci[0] > REACH_CAPACITY
    if severance and not capacity:
        return "H-severance"
    if capacity:
        return "H-capacity"
    return "REACH-graded"


def trainscaffold_verdict(cells, collateral_intact, emrand_reproduces_failure,
                          reachability, n_informative):
    """The §6 hierarchical-gatekeeping 4-way verdict. `cells` maps cell-name → state string
    (from cell_state) for {A_free, A_letter, B_free, B_letter, R_free, R_letter}; `C_floored` is a bool
    folded via `reachability` context. Pure → testable on synthetic banked cells.

      gate 1 (R-validity): R_free AND R_letter AT_FLOOR else INSTRUMENT-NULL.
      gate 2 (B-validity): B_free AND B_letter GENERALIZES else ABORT.
      then the A-fork:
        A_free AT_FLOOR ∧ A_letter AT_FLOOR  → H-lobotomy-total / H-plasticity-loss (collateral selects),
                                               requires emrand NOT reproducing the failure.
        A_free GENERALIZES ∧ A_letter AT_FLOOR (sep>S_FMT handled upstream) → H-readout.
        A_free GENERALIZES ∧ A_letter GENERALIZES → H-capability (GATED on reachability==H-severance),
                                               else 'capacity/difficulty-gradient'.
        else → 'format-axis-UNRESOLVED' / 'H-mixed'.
    Returns {named, basis}."""
    def st(k):
        return cells[k]
    if not (st("R_free") == "AT_FLOOR" and st("R_letter") == "AT_FLOOR"):
        return {"named": "INSTRUMENT-NULL", "basis": "dead-backbone R did not floor — task lacks dynamic range"}
    if not (st("B_free") == "GENERALIZES" and st("B_letter") == "GENERALIZES"):
        return {"named": "ABORT", "basis": "matched positive control B did not pass GO/NO-GO"}
    if int(n_informative) < 2:
        return {"named": "NON-RESULT", "basis": f"only {n_informative} informative seed(s)"}
    af, al = st("A_free"), st("A_letter")
    if af == "AT_FLOOR" and al == "AT_FLOOR":
        if emrand_reproduces_failure:
            return {"named": "MATCHED-NULL", "basis": "emrand reproduces the A failure — effect is generic energy/σ-band removal, not the top directions"}
        named = "H-lobotomy-total" if not collateral_intact else "H-plasticity-loss"
        return {"named": named, "basis": "A cannot acquire novel knowledge in either format; collateral selects total vs plasticity-loss"}
    if af == "GENERALIZES" and al == "AT_FLOOR":
        return {"named": "H-readout", "basis": "novel knowledge acquirable free-gen but not via the letter readout — top-σ carries the head-deleted residual letter-routing"}
    if af == "GENERALIZES" and al == "GENERALIZES":
        if reachability == "H-severance":
            return {"named": "H-capability", "basis": "A learns+generalizes both formats; only OLD re-entry blocked AND refill-blocked full-FT confirms severance — a capability-specific re-entry operator"}
        return {"named": "capacity/difficulty-gradient", "basis": "A learns both formats but REACHABILITY is not severance — a capacity/difficulty gradient, NOT the diamond"}
    return {"named": "format-axis-UNRESOLVED", "basis": f"A_free={af} A_letter={al} — no forced 3-way label"}
