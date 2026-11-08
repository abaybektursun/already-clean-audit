"""N5c re-aggregation + the FROZEN §5 dual-cut decision (pi-trajectory N5 prereg, 2026-06-01).

PURE re-aggregation over the per-tensor tables the cloud job already extracted — NO ΔW re-extraction,
no model, no GPU, no hub, no estimator edit. The frozen E2b instrument carries unchanged. Geometry-only;
fail-loud; no try/except.

Two entry points:
  * ``cloud_member(per_tensor, repo_id)`` — collapse ONE checkpoint's per-tensor table to the member
    record N5c scores (reuses ``e2_aggregate.reaggregate`` for A / by_type_a / mlp_a_per_layer, then adds
    the pooled ρ_L-enrichment, ‖ΔW‖_F, participation-ratio, MLP-bundle a, MLP-positive count, null-FT flag,
    task-distance label).
  * ``decide_n5c(pi0, cloud)`` — the §5 INSIDE / OUTSIDE / INCONCLUSIVE-N waterfall (frozen at tag
    pi-n5c-prereg BEFORE π0's rank is seen), plus the registered secondaries, the 3 aggregation units,
    the task-distance gradient, the ‖ΔW‖ residualization, and the PR-range ρ_L downgrade.
"""
from __future__ import annotations

MLP = ("down_proj", "gate_proj", "up_proj")

# §9 frozen task-distance rule: VQA / caption / instruction = "near" FT#1; segmentation / grounding /
# localization = "distant". Of the 24 transformers -224 repos, only refcoco-seg is distant — a thin tail,
# reported as a registered limitation (control 4). Assignment fixed pre-unblind.
DISTANT_SLUGS = ("refcoco-seg",)


def repo_slug(repo_id):
    """`google/paligemma-3b-ft-vqav2-224` -> `vqav2`."""
    name = repo_id.split("/")[-1]
    assert name.startswith("paligemma-3b-ft-") and name.endswith("-224"), f"unexpected ft repo {repo_id}"
    return name[len("paligemma-3b-ft-"):-len("-224")]


def task_distance(repo_id):
    name = repo_id.split("/")[-1]
    if not (name.startswith("paligemma-3b-ft-") and name.endswith("-224")):
        return "subject"                                  # π0 (lerobot/pi0_base) — the subject, not an ft
    return "distant" if repo_slug(repo_id) in DISTANT_SLUGS else "near"


def _ew(vals, w):
    import numpy as np
    vals = np.asarray(vals, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.average(vals, weights=w)) if w.sum() > 0 else float("nan")


def cloud_member(per_tensor, repo_id, r_grid=(32, 64), seed=1337):
    """One checkpoint's member record. ``per_tensor`` is the frozen-schema table (E2b-identical fields:
    key/type/layer/frob_dW2/pr_dW2 + reuse[str(r)] with a / a_band / rho_L / haar_rho_L)."""
    import numpy as np
    from infra.pg import e2_aggregate

    agg = e2_aggregate.reaggregate(per_tensor, r_grid=r_grid, seed=seed)   # A / by_type_a / mlp_a_per_layer

    w = [rec["frob_dW2"] ** 2 for rec in per_tensor]
    rec = {
        "repo_id": repo_id,
        "task_distance": task_distance(repo_id),
        "n_tensors": len(per_tensor),
        "frob_dW_total": float(np.sqrt(np.sum(w))),           # ‖ΔW_other‖_F over the 18×7 spine
        "pr_pooled": _ew([t["pr_dW2"] for t in per_tensor], w),   # participation ratio (eff-rank proxy)
        "A": agg["A"],
        "by_type_a": agg["by_type_a"],
        "mlp_a_per_layer": agg["mlp_a_per_layer"],
    }

    # pooled ρ_L-enrichment = ρ_L / Haar-p95 (spectrum-conditioned; NEVER raw ρ_L — §4) per r.
    # MLP-bundle pooled signed-a (the one primary per-type contrast) per r.
    # null-FT flag: pooled ρ_L ≤ pooled Haar-p95 (enrichment ≤ 1) at this r.
    enr, mlp_a, null_flag = {}, {}, {}
    for r in r_grid:
        rr = str(r)
        rho = [t["reuse"][rr]["rho_L"] for t in per_tensor]
        p95 = [t["reuse"][rr]["haar_rho_L"]["p95.0"] for t in per_tensor]
        enr[rr] = _ew(rho, w) / _ew(p95, w) if _ew(p95, w) > 0 else float("nan")
        null_flag[rr] = bool(_ew(rho, w) <= _ew(p95, w))
        mlp = [t for t in per_tensor if t["type"] in MLP]
        mlp_a[rr] = _ew([t["reuse"][rr]["a"] for t in mlp], [t["frob_dW2"] ** 2 for t in mlp])
    rec["rhoL_enrichment"] = enr
    rec["mlp_bundle_a"] = mlp_a
    rec["null_ft_on_rhoL"] = null_flag                        # True at all r ⇒ a degenerate "null-FT" member

    # MLP-positive layer count (distribution-free, exchangeable) per r: # of layers 0..17 with a>0.
    mlp_count = {}
    for r in r_grid:
        prof = agg["mlp_a_per_layer"][str(r)]
        mlp_count[str(r)] = int(sum(1 for L in range(18) if prof[str(L)] is not None and prof[str(L)] > 0))
    rec["mlp_pos_count"] = mlp_count

    # count-democratic (UNWEIGHTED) pooled signed-a, L17-MLP excluded — a distinct aggregation SCHEME of
    # the SAME estimand (vs the energy-weighted A headline), for the §4 robustness triad.
    is_l17_mlp = lambda t: (t["layer"] == 17 and t["type"] in MLP)
    unw = {}
    for r in r_grid:
        keep = [t["reuse"][str(r)]["a"] for t in per_tensor if not is_l17_mlp(t)]
        unw[str(r)] = float(np.mean(keep))
    rec["a_unweighted_exL17mlp"] = unw
    return rec


# ---------------------------------------------------------------------------
# §5 frozen decision (waterfall) — INSIDE / OUTSIDE / INCONCLUSIVE-N.
# ---------------------------------------------------------------------------

def _pctl_membership(value, pop, lo=10.0, hi=90.0):
    """central-80% TOST-style equivalence band membership: value ∈ [p10, p90] of the ft population."""
    import numpy as np
    p = np.asarray(pop, dtype=float)
    return bool(np.percentile(p, lo) <= value <= np.percentile(p, hi)), \
        [float(np.percentile(p, lo)), float(np.percentile(p, hi))]


def _rank_fraction(value, pop):
    """(#ft with value ≥ π0 + 1)/(N+1) — small ⇒ π0 high. Reported as a FRACTION, not a p-value (§4)."""
    n = len(pop)
    return float((sum(1 for x in pop if x >= value) + 1) / (n + 1))


def _strict_max_with_gap(value, pop):
    """π0 strictly maximal AND the π0→max_ft gap exceeds the largest adjacent gap among the sorted ft
    order statistics (parameter-free clear-separation, robust to one extreme ft)."""
    import numpy as np
    s = np.sort(np.asarray(pop, dtype=float))
    if value <= s[-1]:
        return False, {"strict_max": False, "pi0_gap": float(value - s[-1])}
    pi0_gap = float(value - s[-1])
    adj = float(np.max(np.diff(s))) if s.size >= 2 else 0.0
    return bool(pi0_gap > adj), {"strict_max": True, "pi0_gap": pi0_gap, "max_adjacent_ft_gap": adj}


def _residualize(pi0_y, pi0_x, ys, xs):
    """π0's residual rank after OLS-conditioning a_pooled on ‖ΔW‖ across the cloud (§9 residualize-all)."""
    import numpy as np
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (b, c), *_ = np.linalg.lstsq(A, ys, rcond=None)
    ft_resid = ys - (b * xs + c)
    pi0_resid = pi0_y - (b * pi0_x + c)
    return {"slope": float(b), "intercept": float(c), "pi0_residual": float(pi0_resid),
            "pi0_residual_rank": _rank_fraction(pi0_resid, ft_resid.tolist())}


def decide_n5c(pi0, cloud, r_grid=(32, 64)):
    """The FROZEN §5 dual-cut. ``pi0`` and each ``cloud`` element are ``cloud_member``-shaped records
    (π0's A comes from the frozen e2_reagg.json). N = len(cloud); π0 is the (N+1)-th member."""
    import numpy as np
    N = len(cloud)
    out = {"N": N, "rank_floor": 1.0 / (N + 1), "GEOMETRY_ONLY": True, "r_grid": list(r_grid)}

    # PR-range ρ_L downgrade (control 2): if π0's pooled PR is outside the ft PR range, the ρ_L axis is
    # eff-rank-confounded → signed-a-only verdict (drop the ρ_L-enrichment INSIDE requirement).
    ft_pr = [m["pr_pooled"] for m in cloud]
    rho_downgraded = bool(pi0["pr_pooled"] < min(ft_pr) or pi0["pr_pooled"] > max(ft_pr))
    out["rhoL_axis_downgraded"] = rho_downgraded
    out["pr"] = {"pi0": pi0["pr_pooled"], "ft_range": [min(ft_pr), max(ft_pr)]}

    # ---- per-r PRIMARY (a_pooled_exL17mlp) + secondary (ρ_L-enrichment) cut ----
    per_r = {}
    for r in r_grid:
        rr = str(r)
        pi0_a = pi0["A"][rr]["a_pooled_exL17mlp"]
        ft_a = [m["A"][rr]["a_pooled_exL17mlp"] for m in cloud]
        inside_a, band_a = _pctl_membership(pi0_a, ft_a)
        outside_a, gap_a = _strict_max_with_gap(pi0_a, ft_a)

        pi0_e = pi0["rhoL_enrichment"][rr]
        ft_e = [m["rhoL_enrichment"][rr] for m in cloud]
        inside_e, band_e = _pctl_membership(pi0_e, ft_e)

        # INSIDE at this r := primary inside AND (ρ_L-enrichment inside OR axis downgraded).
        inside_r = inside_a and (inside_e or rho_downgraded)
        per_r[rr] = {
            "pi0_a_pooled_exL17mlp": pi0_a, "ft_a_median": float(np.median(ft_a)),
            "a_rank_fraction": _rank_fraction(pi0_a, ft_a), "a_central80_band": band_a, "a_inside": inside_a,
            "a_outside_strict_gap": outside_a, "a_gap": gap_a,
            "pi0_rhoL_enrichment": pi0_e, "enrichment_central80_band": band_e, "enrichment_inside": inside_e,
            "inside_r": bool(inside_r), "outside_r": bool(outside_a),
        }
    out["per_r"] = per_r

    inside_all = all(per_r[str(r)]["inside_r"] for r in r_grid)
    outside_all = all(per_r[str(r)]["outside_r"] for r in r_grid)

    # OUTSIDE OR-branch: π0 survives ex-L17-MLP where the ENTIRE cloud does not.
    cloud_survives = sum(1 for m in cloud if m["A"][str(r_grid[0])]["survives_exL17mlp"]
                         and m["A"][str(r_grid[-1])]["survives_exL17mlp"])
    pi0_survives = all(pi0["A"][str(r)]["survives_exL17mlp"] for r in r_grid)
    survives_outlier = bool(pi0_survives and cloud_survives == 0)
    out["survives"] = {"pi0_survives_both_r": pi0_survives, "n_cloud_survives_both_r": cloud_survives,
                       "survives_outlier": survives_outlier}

    # ---- the waterfall. OUTSIDE is evaluated FIRST: a robust outlier (strict-max+gap at both r, OR
    #      survives-where-the-cloud-does-not) is an INDEPENDENT escalation trigger (§5), and a false STOP
    #      ("architecture-intrinsic") is the worse error than a needless GPU check — so OUTSIDE wins ties.
    #      INSIDE/OUTSIDE must AGREE at both r; disagreement ⇒ inconclusive-N. ----
    if outside_all or survives_outlier:
        verdict = "OUTSIDE_candidate_special"
        note = ("π0 is a robust outlier among same-init siblings (strictly maximal + clear-separation gap at "
                "both r, or survives ex-L17-MLP where the cloud does not) → it differs along some of "
                "{task, modality, mechanism} — FULLY ALIASED, no standalone task-specific claim. Escalate to "
                "N5a (the only GPU spend).")
    elif inside_all:
        verdict = "INSIDE_architecture_intrinsic"
        note = ("π0's signed reuse sits inside the same-init full-FT cloud at BOTH r (central-80% equivalence "
                "band; ρ_L-enrichment concordant or axis-downgraded) → the FT#1→FT#2 extend-alignment is what "
                "any same-init full-FT of pt-224 does. Re-scope E2b 'extend-aligned' to a backbone-edit-cone "
                "property. STOP — CPU-final, no GPU.")
    else:
        verdict = "INCONCLUSIVE_N"
        note = ("neither a clean INSIDE nor a robust OUTSIDE at both r (or r=32/64 disagree). Report the rank "
                "as a fraction; note the 1/(N+1) small-N floor. Do NOT escalate.")
    out["verdict"] = verdict
    out["verdict_note"] = note

    # ---- registered secondaries (reported regardless of the verdict) ----
    # MLP-bundle a rank; MLP-positive-count vs the cloud distribution of that count.
    sec = {}
    for r in r_grid:
        rr = str(r)
        sec[rr] = {
            "mlp_bundle_a_rank": _rank_fraction(pi0["mlp_bundle_a"][rr], [m["mlp_bundle_a"][rr] for m in cloud]),
            "pi0_mlp_pos_count": pi0["mlp_pos_count"][rr],
            "ft_mlp_pos_count": sorted(m["mlp_pos_count"][rr] for m in cloud),
            "mlp_count_rank": _rank_fraction(pi0["mlp_pos_count"][rr],
                                             [m["mlp_pos_count"][rr] for m in cloud]),
        }
    out["secondaries"] = sec

    # ---- 3 aggregation units (energy-weighted headline / count-democratic / per-type-averaged); disagreement
    #      is itself a registered finding. r-keyed; the per-type-averaged unit unweights the 7 types. ----
    units = {}
    for r in r_grid:
        rr = str(r)
        ft_ew = [m["A"][rr]["a_pooled_exL17mlp"] for m in cloud]
        # per-type-averaged: unweighted mean of the by_type_a point estimates (excludes nothing here; the
        # readout-artifact concern is L17-MLP-specific and the headline already excludes it).
        def _typeavg(mem):
            pts = [mem["by_type_a"][rr][t]["point"] for t in mem["by_type_a"][rr]]
            return float(np.mean(pts))
        units[rr] = {
            "energy_weighted_rank": _rank_fraction(pi0["A"][rr]["a_pooled_exL17mlp"], ft_ew),
            "count_democratic_rank": _rank_fraction(pi0["a_unweighted_exL17mlp"][rr],
                                                    [m["a_unweighted_exL17mlp"][rr] for m in cloud]),
            "per_type_averaged_rank": _rank_fraction(_typeavg(pi0), [_typeavg(m) for m in cloud]),
        }
    ranks_at = lambda rr: [units[rr]["energy_weighted_rank"], units[rr]["count_democratic_rank"],
                           units[rr]["per_type_averaged_rank"]]
    # disagreement flag: a unit lands inside-bulk (>floor*?) while another lands at the outlier floor.
    out["aggregation_units"] = units
    out["aggregation_units_disagree"] = bool(
        any(max(ranks_at(str(r))) - min(ranks_at(str(r))) > 0.5 for r in r_grid))

    # ---- task-distance gradient (control 4): near vs distant a_pooled_exL17mlp. Flat-high ⇒ genuinely
    #      architecture-intrinsic; decaying-with-distance ⇒ FT#1-family-inflated null, re-read INSIDE. ----
    grad = {}
    for r in r_grid:
        rr = str(r)
        near = [m["A"][rr]["a_pooled_exL17mlp"] for m in cloud if m["task_distance"] == "near"]
        dist = [m["A"][rr]["a_pooled_exL17mlp"] for m in cloud if m["task_distance"] == "distant"]
        grad[rr] = {"near_mean": float(np.mean(near)) if near else None, "n_near": len(near),
                    "distant_mean": float(np.mean(dist)) if dist else None, "n_distant": len(dist),
                    "distant_values": dist}
    out["task_distance_gradient"] = grad
    out["task_distance_caveat"] = ("only refcoco-seg is 'distant' by the §9 rule (n_distant=1) — the gradient "
                                   "is a thin-tail diagnostic, not a powered test (registered limitation).")

    # ---- ‖ΔW‖ residualization (§9 residualize-all) on a_pooled_exL17mlp, per r. ----
    resid = {}
    for r in r_grid:
        rr = str(r)
        resid[rr] = _residualize(pi0["A"][rr]["a_pooled_exL17mlp"], pi0["frob_dW_total"],
                                 [m["A"][rr]["a_pooled_exL17mlp"] for m in cloud],
                                 [m["frob_dW_total"] for m in cloud])
    out["norm_residualized"] = resid
    out["null_ft_members"] = [m["repo_id"] for m in cloud if all(m["null_ft_on_rhoL"].values())]
    return out
