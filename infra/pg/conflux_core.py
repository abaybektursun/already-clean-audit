"""CONFLUX core — is the shared fine-tune backbone U_bb the base model's sensitive subspace U_sus?

Eval-only, weight-free math over banked tensors (spec `2026-07-14_conflux_spec.md`; the M0.3
amendments in `2026-07-14_conflux_journal.md`). Composes REUSED surfaces:
  * `gauge_rmt.outlier_subspaces` / `subspace_overlap` / `haar_overlap_floor`   (the overlap wiring),
  * `handle_core.core_basis` / `band_basis` / `median_ci`                        (U_sus slice, nulls, CI),
  * `carry_core.stiff_split` / `random_eigsplit`                                 (Q split; via band_basis),
  * `comwrite_core.subspace_overlap`                                            (same-shape /r overlap).

The single NEW content (spec §12):
  * `ubb_basis`      — the RAW mean-projector backbone estimator  U_bb = topEig((1/N)Σ U_iU_iᵀ),
                       U_i = top-r LEFT singular vectors of the RAW o_proj ΔW (NO Q-projection; the
                       non-circular object, journal A1);
  * `energy_null`    — the secondary activation-energy-matched random subspace (U_sus per-coordinate
                       energy footprint);  the DECISIVE null is the REUSED within-stiff `band_basis`;
  * `ladder_overlap` — per-rung o(U_bb,U_sus) vs {Haar-2048, within-stiff, energy} + reference
                       partners, FT-level pair-bootstrap excess CIs, and the c_sat saturation VOID
                       (journal A2/A4; `feedback_one_sided_thresholds`);
  * `adjudicate_conflux` — the frozen verdict branch order (VOID → H_independent → H_shell → H_identity).

fp64 throughout; heavy imports inside functions so the module py_compiles on the control plane; no
try/except hiding (CLAUDE.md). Orthonormal frames are (d, r) with columns = the subspace.
"""
from __future__ import annotations

# ---- frozen constants (spec §3/§6; journal A2/A4) -----------------------------------------------
K_LADDER_FIXED = (5, 10, 20, 50, 150, 350)     # + k_stiff appended at runtime = the 7 rungs
C_SAT = 0.90                                    # within-stiff null saturation ceiling (journal A2)
N_NULL = 200                                    # matched-rank null draws per rung
N_BOOT = 10_000                                 # FT-level pair-bootstrap draws (house default)
SEED = 1337
R_DEFAULT = 20                                  # pre-declared r* fallback (HANDLE's core rank, spec §6)
# CARRY massive-activation coords (critic #2, corrected) — the sink-quotient control (T4).
MASSIVE_COORDS = (738, 674, 604, 2015)


# ===========================================================================
# Backbone estimator U_bb — the RAW, non-circular object (journal A1).
# ===========================================================================
def ft_left_subspace(dW, r):
    """Top-r LEFT singular vectors of a raw ΔW (residual-facing/output side for o_proj), (d, r)
    orthonormal. Thin call to the REUSED gauge_rmt.outlier_subspaces (no Q-projection)."""
    from infra.pg.gauge_rmt import outlier_subspaces
    import numpy as np
    U_left, _V_right = outlier_subspaces(np.asarray(dW, dtype=np.float64), int(r))
    return U_left


def mean_projector(U_list):
    """M = (1/N) Σ_i U_i U_iᵀ for a list of (d, r) orthonormal frames. (d, d) fp64 symmetric.
    Kept for the D-suite/reference; `ubb_basis` uses the equivalent cheaper stacked-SVD route."""
    import numpy as np
    d = U_list[0].shape[0]
    M = np.zeros((d, d), dtype=np.float64)
    for U in U_list:
        A = np.asarray(U, dtype=np.float64)
        M += A @ A.T
    return M / float(len(U_list))


def ubb_basis(U_list, r):
    """U_bb(r) = top-r eigenvectors of the mean core projector M=(1/N)Σ U_iU_iᵀ over the per-FT LEFT
    subspaces `U_list`. Computed as the top-r LEFT singular vectors of the stacked W=[U_1|…|U_N]
    (d, N·r) — identical eigvectors (eig(M)=svd(W)²), far cheaper for small r. (d, r) orthonormal.
    The single-subspace estimator of the shared span (journal A1)."""
    import numpy as np
    W = np.concatenate([np.asarray(U, dtype=np.float64) for U in U_list], axis=1)
    Uw, _s, _Vt = np.linalg.svd(W, full_matrices=False)
    return np.ascontiguousarray(Uw[:, :int(r)])


def build_ubb(dW_list, r):
    """Convenience: raw per-FT top-r LEFT subspaces → U_bb(r). Returns (U_bb, [U_i])."""
    U_list = [ft_left_subspace(dW, r) for dW in dW_list]
    return ubb_basis(U_list, r), U_list


# ===========================================================================
# Overlap + nulls.  U_sus := top-r Q eigvecs (handle_core.core_basis).  Decisive null := within-stiff.
# ===========================================================================
def overlap(A, B):
    """‖AᵀB‖²_F / r for two (d, r) orthonormal frames — REUSED comwrite_core.subspace_overlap
    (same-shape asserted, delegates to the one /r-normalised gauge_rmt implementation)."""
    from infra.pg.comwrite_core import subspace_overlap
    return subspace_overlap(A, B)


def usus_basis(evecs, r):
    """U_sus(r) = top-r eigvecs of the banked Q (descending) — REUSED handle_core.core_basis."""
    from infra.pg.handle_core import core_basis
    import numpy as np
    return np.asarray(core_basis(evecs, int(r)), dtype=np.float64)


def within_stiff_null(evecs, r, k_stiff, seed):
    """DECISIVE null (journal A2): a matched-rank random frame inside span(top-k_stiff Q eigvecs) —
    REUSED handle_core.band_basis(..., within_stiff=k_stiff). (d, r) orthonormal."""
    from infra.pg.handle_core import band_basis
    import numpy as np
    U, _idx = band_basis(evecs, int(r), int(seed), within_stiff=int(k_stiff))
    return np.asarray(U, dtype=np.float64)


def energy_null(U_sus, r, seed):
    """SECONDARY activation-energy null (spec §5, journal A2 secondary): a random (d, r) frame whose
    per-coordinate energy footprint matches U_sus's own (row-energy p_j = Σ_i U_sus[j,i]²). Sample
    G[j,:] ~ N(0, p_j), QR-orthonormalise. Controls the massive-activation/high-energy-shell threat
    without the curvature information."""
    import numpy as np
    S = np.asarray(U_sus, dtype=np.float64)
    d = S.shape[0]
    p = (S ** 2).sum(axis=1)                     # per-coordinate energy of U_sus, sums to r
    rng = np.random.default_rng(int(seed))
    G = rng.standard_normal((d, int(r))) * np.sqrt(np.maximum(p, 1e-30))[:, None]
    Q, _R = np.linalg.qr(G)
    return np.ascontiguousarray(Q[:, :int(r)])


def haar_floor(r, d):
    """(mean r/d, p95) of the full-2048 Haar overlap floor — REUSED gauge_rmt.haar_overlap_floor."""
    from infra.pg.gauge_rmt import haar_overlap_floor
    h = haar_overlap_floor(int(r), int(d))
    return {"mean": float(h["mean"]), "p95": float(h["pcts"]["p95.0"])}


# ===========================================================================
# The primary readout: per-rung overlap ladder with FT-level bootstrap excess CIs + c_sat VOID.
# ===========================================================================
N_BOOT_NULL = 25                                # inner null set for the bootstrap excess (point uses N_NULL)


def _boot_excess_cis(U_list, r, U_sus, within_bases, haar_bases, d, n_boot=N_BOOT, seed=SEED):
    """One FT-level pair-bootstrap loop → excess CIs vs BOTH the within-stiff (decisive) and Haar-2048
    nulls. Each draw resamples the FTs, rebuilds U_bb*, and records o(U_bb*,U_sus) − median_j
    o(U_bb*, null_j). `within_bases`/`haar_bases` are small fixed null sets (they do not depend on the
    FTs). Returns {within:{median,ci95}, haar:{median,ci95}, o_sus:{median,ci95}}."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    n = len(U_list)
    ex_w, ex_h, o_s = [], [], []
    for _b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        Ub = ubb_basis([U_list[i] for i in idx], r)
        o_t = overlap(Ub, U_sus)
        o_w = float(np.median([overlap(Ub, Nb) for Nb in within_bases]))
        o_h = float(np.median([overlap(Ub, Nb) for Nb in haar_bases]))
        o_s.append(o_t); ex_w.append(o_t - o_w); ex_h.append(o_t - o_h)

    def _ci(a):
        a = np.asarray(a, dtype=np.float64)
        return {"median": float(np.median(a)),
                "ci95": [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]}
    return {"within": _ci(ex_w), "haar": _ci(ex_h), "o_sus": _ci(o_s)}


def rung_overlap(dW_list, evecs, k_stiff, r, partners, n_null=N_NULL, n_boot=N_BOOT, seed=SEED,
                 boot=True):
    """One rung r. Point overlaps o(U_bb, {U_sus, within-stiff nulls, energy nulls, Haar, partners}),
    the nulls' own o(null, U_sus) (the c_sat quantity), FT-level bootstrap excess CIs vs the within-
    stiff (decisive) and Haar nulls, and the saturation VOID flag. All (d, r) frames.

    partners = {name: (d, k) banked frame} (e.g. U_wr, U_unt); sliced to r.
    Returns a JSON-able dict.
    """
    import numpy as np
    d = evecs.shape[0]
    U_list = [ft_left_subspace(dW, r) for dW in dW_list]
    U_bb = ubb_basis(U_list, r)
    U_sus = usus_basis(evecs, r)

    ws = [within_stiff_null(evecs, r, k_stiff, seed + 11 + j) for j in range(n_null)]
    en = [energy_null(U_sus, r, seed + 7000 + j) for j in range(n_null)]
    o_ws = [overlap(U_bb, N) for N in ws]                       # U_bb vs within-stiff null
    o_en = [overlap(U_bb, N) for N in en]                       # U_bb vs energy null
    ws_vs_sus = [overlap(N, U_sus) for N in ws]                 # the c_sat quantity (null looks like sus?)
    hf = haar_floor(r, d)

    o_real = overlap(U_bb, U_sus)
    # partner frames may be banked at fewer than r columns → only compare where ≥ r available.
    part = {name: overlap(U_bb, np.asarray(P, dtype=np.float64)[:, :r])
            for name, P in partners.items() if np.asarray(P).shape[1] >= r}

    ws_median = float(np.median(ws_vs_sus))
    void = bool(ws_median >= C_SAT) or bool(r >= k_stiff)       # whole-shell rung saturates by construction

    out = {"r": int(r), "o_real": float(o_real),
           "haar": hf,
           "within_stiff_null": {"o_ubb_median": float(np.median(o_ws)),
                                 "o_ubb_p95": float(np.percentile(o_ws, 95)),
                                 "null_vs_sus_median": ws_median,
                                 "null_vs_sus_p95": float(np.percentile(ws_vs_sus, 95))},
           "energy_null": {"o_ubb_median": float(np.median(o_en)),
                           "o_ubb_p95": float(np.percentile(o_en, 95)),
                           "null_vs_sus_median": float(np.median([overlap(N, U_sus) for N in en]))},
           "partners": {k: float(v) for k, v in part.items()},
           "excess_haar_point": float(o_real - hf["mean"]),
           "excess_within_point": float(o_real - float(np.median(o_ws))),
           "excess_energy_point": float(o_real - float(np.median(o_en))),
           "excess_partner_point": float(o_real - max(part.values())) if part else None,
           "c_sat": C_SAT, "void": void}
    # Bootstrap CIs only on non-VOID rungs (a VOID rung is uninformative → the point stats suffice).
    if boot and not void:
        haar_bases = [_haar_basis(d, r, seed + 20000 + j) for j in range(N_BOOT_NULL)]
        ci = _boot_excess_cis(U_list, r, U_sus, ws[:N_BOOT_NULL], haar_bases, d, n_boot, seed)
        out["excess_within_ci"] = ci["within"]
        out["excess_haar_ci"] = ci["haar"]
        out["o_real_ci"] = ci["o_sus"]
    return out


def _haar_basis(d, r, seed):
    """A full-2048 Haar random (d, r) orthonormal frame (for the Haar-null bootstrap term)."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    Q, _R = np.linalg.qr(rng.standard_normal((int(d), int(r))))
    return np.ascontiguousarray(Q[:, :int(r)])


def ladder_ks(k_stiff):
    """The 7 rungs (REUSED handle_core.ladder_ks semantics): fixed rungs < k_stiff, then k_stiff."""
    from infra.pg.handle_core import ladder_ks as _lk
    return _lk(k_stiff)


BOOT_MAX_R = 150                                # never bootstrap rungs above this (r=350 SVD ≈ 40min/rung)


def overlap_ladder(dW_list, evecs, k_stiff, partners, ks=None, n_null=N_NULL, n_boot=N_BOOT,
                   seed=SEED, boot=True, boot_max_r=BOOT_MAX_R):
    """The full rung ladder → {rung: rung_overlap(...)} + the ks used. Bootstrap runs ASCENDING and
    stops after the FIRST clearing rung (r* is the smallest clearing rung, so higher-rung CIs are
    never needed); it is also capped at r ≤ boot_max_r (the r=350 SVD-per-draw is ~40 min/rung and
    r* is expected ≤ 50). Rungs above the first clear / above the cap are point-only (descriptive)."""
    ks = tuple(ks) if ks is not None else ladder_ks(k_stiff)
    rungs, cleared = {}, False
    for r in ks:                                                # ascending
        do_boot = bool(boot and not cleared and int(r) <= int(boot_max_r))
        rg = rung_overlap(dW_list, evecs, k_stiff, r, partners, n_null, n_boot, seed, do_boot)
        rungs[int(r)] = rg
        ci = rg.get("excess_within_ci", {}).get("ci95")
        if (not rg["void"]) and ci and ci[0] > 0.0:
            cleared = True                                      # first clearing rung → stop bootstrapping
    return {"ks": list(ks), "rungs": rungs}


# ===========================================================================
# r* selection (discovery) and the frozen verdict (confirmation).
# ===========================================================================
def select_rstar(disc_ladder, r_default=R_DEFAULT):
    """Smallest non-VOID rung with within-stiff excess CI-lo > 0 (journal A2); else r_default (clamped
    to a rung actually present on the ladder)."""
    ks = [int(r) for r in disc_ladder["ks"]]
    for r in ks:
        rg = disc_ladder["rungs"][int(r)]
        if (not rg["void"]) and rg.get("excess_within_ci", {}).get("ci95", [0])[0] > 0.0:
            return int(r)
    return int(r_default) if int(r_default) in ks else int(ks[0])


def adjudicate_conflux(rung):
    """Frozen branch order (spec §6, journal A2), applied ONCE on the confirmation rung at r*:
      1. VOID       — the rung saturated (c_sat) → uninformative.
      2. H_independent — Haar excess CI-lo ≤ 0: U_bb does not reach the random floor with U_sus.
      3. H_shell    — Haar cleared but within-stiff excess CI-lo ≤ 0: in the stiff shell, not its top.
      4. H_identity — within-stiff excess CI-lo > 0 AND partner excess > 0: the backbone IS the
                      sensitive top, beyond a random stiff direction and beyond the known writes.
    Returns {verdict, reading, ...}. No rule reinterpreted post-unblind (feedback_no_post_unblind...)."""
    if rung["void"]:
        return {"verdict": "VOID", "reading": "rung saturated (within-stiff null ≥ c_sat); uninformative"}
    haar_lo = rung["excess_haar_ci"]["ci95"][0]
    within_lo = rung["excess_within_ci"]["ci95"][0]
    partner = rung["excess_partner_point"]
    if haar_lo <= 0.0:
        return {"verdict": "H_independent", "haar_lo": haar_lo,
                "reading": "U_bb does not reach the random floor with U_sus — a distinct third structure"}
    if within_lo <= 0.0:
        return {"verdict": "H_shell", "haar_lo": haar_lo, "within_lo": within_lo,
                "reading": "U_bb overlaps U_sus only as much as a random stiff-shell frame — shell, not top"}
    if partner is None:                                        # partners not banked at this rank (r>128)
        return {"verdict": "H_top_partner_untested", "haar_lo": haar_lo, "within_lo": within_lo,
                "excess_partner": None,
                "reading": "U_bb beats the stiff shell (top-aligned), but U_wr/U_unt are unavailable at "
                           "this rank so the distinct-from-writes test could not run"}
    partner_ok = partner > 0.0
    return {"verdict": "H_identity" if partner_ok else "H_shell_top_ambiguous",
            "haar_lo": haar_lo, "within_lo": within_lo, "excess_partner": partner,
            "reading": ("U_bb aligns with the sensitive TOP beyond a random stiff frame AND beyond "
                        "U_wr/U_unt — the backbone is the base model's sensitive subspace"
                        if partner_ok else
                        "U_bb beats the stiff shell but not the known write partners — top-aligned yet "
                        "not distinct from the write consensus")}


# ===========================================================================
# Corroboration arms (each with its own null / disclosure).
# ===========================================================================
def per_layer_overlap(dW_by_layer, evecs, k_stiff, r, n_null=N_NULL, seed=SEED):
    """Arm D. For each layer L, U_bb^{(L)}(r) from that layer's per-FT o_proj ΔW vs the SINGLE L0
    U_sus(r) (only L0 Q banked — disclosed). Returns {L: {o_real, within_null_median, haar_mean}}."""
    import numpy as np
    U_sus = usus_basis(evecs, r)
    d = evecs.shape[0]
    out = {}
    for L, dW_list in dW_by_layer.items():
        U_bb = ubb_basis([ft_left_subspace(dW, r) for dW in dW_list], r)
        ws = [overlap(U_bb, within_stiff_null(evecs, r, k_stiff, seed + L * 1000 + j)) for j in range(n_null)]
        out[int(L)] = {"o_real": float(overlap(U_bb, U_sus)),
                       "within_null_median": float(np.median(ws)),
                       "haar_mean": float(r) / float(d)}
    return out


def three_way(U_bb, U_sus, carrier):
    """Arm C. Pairwise overlaps + the triple projector overlap tr(P_bb P_sus P_carrier)/r at the
    common rank r = min column count (carrier is rank-8). All frames sliced to r_common."""
    import numpy as np
    r = min(U_bb.shape[1], U_sus.shape[1], carrier.shape[1])
    A, B, C = (np.asarray(X, dtype=np.float64)[:, :r] for X in (U_bb, U_sus, carrier))
    Pb, Ps, Pc = A @ A.T, B @ B.T, C @ C.T
    return {"r": int(r), "bb_sus": overlap(A, B), "bb_carrier": overlap(A, C),
            "sus_carrier": overlap(B, C),
            "triple": float(np.trace(Pb @ Ps @ Pc)) / float(r)}


def residual_characterisation(U_bb, U_sus, massive_coords=MASSIVE_COORDS):
    """Arm N. The part of U_bb outside span(U_sus): R = (I − P_sus) U_bb, re-orthonormalised. Its
    captured-energy fraction off U_sus, its mass on the massive-activation coords, and its mean
    per-coordinate energy. Names what is left over when U_bb does not reduce to U_sus."""
    import numpy as np
    A = np.asarray(U_bb, dtype=np.float64)
    S = np.asarray(U_sus, dtype=np.float64)
    r = A.shape[1]
    R = A - S @ (S.T @ A)                                       # off-U_sus component of each column
    off_frac = float((R ** 2).sum() / A.shape[1])              # mean energy fraction of U_bb off U_sus
    Qr, _ = np.linalg.qr(R) if np.linalg.norm(R) > 1e-12 else (np.zeros_like(A), None)
    Qr = Qr[:, :r]
    coord_energy = (Qr ** 2).sum(axis=1)                       # per-coordinate energy of the residual
    mc = [int(c) for c in massive_coords if int(c) < coord_energy.shape[0]]   # in-range only (synthetic d<2048)
    sink_mass = float(coord_energy[mc].sum() / max(coord_energy.sum(), 1e-30)) if mc else 0.0
    return {"off_sus_energy_frac": off_frac, "sink_mass_frac": sink_mass,
            "residual_rank_energy": float(coord_energy.sum())}


def predictive_arm(dW_by_ft, evecs, k_stiff, r, U_bb, gains):
    """Arm B (curiosity). Per held-out FT: off-U_bb vs off-U_sus energy fraction of its o_proj ΔW,
    correlated (Spearman, no-training) with a banked behavioural quantity `gains[ft]`. Returns the
    two rho's + the per-FT table. INSTRUMENT-NULL if < 3 FTs have a finite gain."""
    import numpy as np
    from infra.pg.carry_core import spearman0
    U_sus = usus_basis(evecs, r)
    rows, off_bb, off_sus, g = [], [], [], []
    for ft, dW in dW_by_ft.items():
        if ft not in gains or gains[ft] is None:
            continue
        A = np.asarray(dW, dtype=np.float64)
        e_tot = float((A ** 2).sum())
        f_bb = 1.0 - float(((U_bb.T @ A) ** 2).sum() / e_tot)   # fraction of ΔW energy OFF U_bb
        f_sus = 1.0 - float(((U_sus.T @ A) ** 2).sum() / e_tot)
        rows.append({"ft": ft, "off_bb": f_bb, "off_sus": f_sus, "gain": float(gains[ft])})
        off_bb.append(f_bb); off_sus.append(f_sus); g.append(float(gains[ft]))
    if len(g) < 3:
        return {"instrument_null": True, "n": len(g), "reading": "< 3 FTs with a finite gain"}
    return {"instrument_null": False, "n": len(g),
            "rho_off_bb_vs_gain": spearman0(np.asarray(off_bb), np.asarray(g)),
            "rho_off_sus_vs_gain": spearman0(np.asarray(off_sus), np.asarray(g)),
            "rows": rows}


# ===========================================================================
# Sink-quotient control (T4) — project out the massive-activation coords, re-run one rung.
# ===========================================================================
def sink_quotient_overlap(dW_list, evecs, k_stiff, r, massive_coords=MASSIVE_COORDS, seed=SEED,
                          n_null=N_NULL):
    """Re-run o(U_bb, U_sus) and the within-stiff null AFTER zeroing the massive-activation coords in
    every frame and re-orthonormalising — if the coincidence survives, it is not an artifact of the
    one dominant direction (spec §5, threat T4)."""
    import numpy as np
    mc = [int(c) for c in massive_coords if int(c) < int(np.asarray(evecs).shape[0])]

    def _quotient(U):
        V = np.asarray(U, dtype=np.float64).copy()
        V[mc, :] = 0.0
        Q, _ = np.linalg.qr(V)
        return Q[:, :V.shape[1]]
    U_bb = _quotient(ubb_basis([ft_left_subspace(dW, r) for dW in dW_list], r))
    U_sus = _quotient(usus_basis(evecs, r))
    ws = [overlap(U_bb, _quotient(within_stiff_null(evecs, r, k_stiff, seed + 313 + j)))
          for j in range(n_null)]
    o_real = overlap(U_bb, U_sus)
    return {"r": int(r), "o_real_quotient": float(o_real),
            "within_null_median_quotient": float(np.median(ws)),
            "excess_within_quotient": float(o_real - float(np.median(ws)))}
