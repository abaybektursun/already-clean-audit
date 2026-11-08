"""E2b re-aggregation (pi-trajectory follow-up). PURE re-aggregation over the FROZEN per-tensor table —
NO ΔW re-extraction, no model, no GPU, no hub. Adds Deliverable-A (the decisive L17 check) + Deliverable-B
(`by_type_a` bootstrap CIs, the same n=18-layer unit as `by_type_rho_L`). The §A decision rule is frozen in
`decide()` at tag `pi-e2-reagg-prereg` BEFORE the number is seen. Geometry-only; fail-loud; no try/except.
"""
from __future__ import annotations

MLP = ("down_proj", "gate_proj", "up_proj")
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _ew(vals, w):
    """Energy-weighted mean (weights = frob_dW2²)."""
    import numpy as np
    vals = np.asarray(vals, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.average(vals, weights=w)) if w.sum() > 0 else float("nan")


def reaggregate(per_tensor, r_grid=(32, 64), seed=1337):
    import numpy as np
    from infra.pg.diff_metrics import bootstrap_ci

    def w2(rec):
        return rec["frob_dW2"] ** 2

    is_l17_mlp = lambda r: (r["layer"] == 17 and r["type"] in MLP)
    is_l17_any = lambda r: (r["layer"] == 17)

    out = {"r_grid": list(r_grid)}

    # ---- Deliverable B: by_type_a bootstrap CIs (n=18 per-layer a values, same unit as by_type_rho_L) ----
    by_type_a = {}
    for r in r_grid:
        bt = {}
        for t in PROJ:
            a_vals = [rec["reuse"][str(r)]["a"] for rec in per_tensor if rec["type"] == t]
            bt[t] = bootstrap_ci(a_vals, seed=seed, stat="mean")  # {point, lo, hi, n}
        by_type_a[str(r)] = bt
    out["by_type_a"] = by_type_a

    # ---- Deliverable A: pooled signed-a with L17-MLP excluded, vs the SAME frozen N1 a-band ----
    A = {}
    for r in r_grid:
        rr = str(r)
        a_all = [rec["reuse"][rr]["a"] for rec in per_tensor]
        w_all = [w2(rec) for rec in per_tensor]
        ub_all = [rec["reuse"][rr]["a_band"][1] for rec in per_tensor]
        # exclude the 3 L17 MLP tensors
        keep_mlp = [rec for rec in per_tensor if not is_l17_mlp(rec)]
        a_ex = [rec["reuse"][rr]["a"] for rec in keep_mlp]
        w_ex = [w2(rec) for rec in keep_mlp]
        ub_ex = [rec["reuse"][rr]["a_band"][1] for rec in keep_mlp]
        # exclude ALL of L17 (q/k/v/o too) — sensitivity bound
        keep_l17 = [rec for rec in per_tensor if not is_l17_any(rec)]
        a_exall = [rec["reuse"][rr]["a"] for rec in keep_l17]
        w_exall = [w2(rec) for rec in keep_l17]
        ub_exall = [rec["reuse"][rr]["a_band"][1] for rec in keep_l17]
        A[rr] = {
            "a_pooled_full": _ew(a_all, w_all),
            "a_ub_full": _ew(ub_all, w_all),
            "a_pooled_exL17mlp": _ew(a_ex, w_ex),
            "a_ub_exL17mlp": _ew(ub_ex, w_ex),
            "a_pooled_exL17all": _ew(a_exall, w_exall),
            "a_ub_exL17all": _ew(ub_exall, w_exall),
            "n_excluded_mlp": len(per_tensor) - len(keep_mlp),
            "survives_exL17mlp": bool(_ew(a_ex, w_ex) > _ew(ub_ex, w_ex)),
        }
        # graded exclusion: drop the top-k highest-energy tensors
        order = sorted(range(len(per_tensor)), key=lambda i: w2(per_tensor[i]), reverse=True)
        graded = {}
        for k in (0, 1, 2, 3, 5, 10):
            keep = [per_tensor[i] for i in order[k:]]
            graded[str(k)] = {"a_pooled": _ew([x["reuse"][rr]["a"] for x in keep], [w2(x) for x in keep]),
                              "dropped_keys": [per_tensor[i]["key"] for i in order[:k]]}
        A[rr]["graded_exclusion"] = graded
    out["A"] = A

    # MLP a-per-layer profile (energy-weighted a of {down,gate,up} at each layer), r=32 (+ r=64).
    mlp_profile = {}
    for r in r_grid:
        rr = str(r)
        prof = {}
        for L in range(18):
            recs = [rec for rec in per_tensor if rec["layer"] == L and rec["type"] in MLP]
            prof[str(L)] = _ew([x["reuse"][rr]["a"] for x in recs], [w2(x) for x in recs]) if recs else None
        mlp_profile[rr] = prof
    out["mlp_a_per_layer"] = mlp_profile

    out["verdict"] = decide(A, mlp_profile)
    return out


def decide(A, mlp_profile):
    """The FROZEN §A decision rule (tag pi-e2-reagg-prereg). Waterfall, exhaustive + exclusive."""
    surv32 = A["32"]["survives_exL17mlp"]
    surv64 = A["64"]["survives_exL17mlp"]
    # early_pos: MLP-a > 0 in ≥13 of layers 0..16 (broad positive sign, NOT confined to near L17), at r=32.
    prof32 = mlp_profile["32"]
    early = [prof32[str(L)] for L in range(17) if prof32[str(L)] is not None]
    n_early_pos = sum(1 for v in early if v > 0)
    early_pos = n_early_pos >= 13

    if not (surv32 and surv64):
        bin_, note = "L17_readout_artifact", ("signed-extend collapses INTO the N1 band once L17-MLP is "
            "removed at r=32 and/or r=64 → headline downgrades to 'non-rotation-random, mechanism "
            "unresolved'; 'MLP extends' retracted to 'L17-MLP only, plausibly readout adaptation'")
    elif early_pos:
        bin_, note = "backbone_wide_MLP_extend", ("a_pooled_exL17mlp clears the N1 band at BOTH r=32 and "
            "r=64 AND MLP-a>0 in ≥13/17 of layers 0..16 → MLP-extend is backbone-wide as a GEOMETRIC "
            "statement (N5-pending for task-specificity); §1/§3 framing stands")
    else:
        bin_, note = "intermediate", ("survives the band at both r but MLP-a-per-layer not broadly positive "
            "(≥13/17 of L0..16); report the curve, no strong backbone-wide claim")
    return {"bin": bin_, "surv32": bool(surv32), "surv64": bool(surv64),
            "n_early_layers_pos": int(n_early_pos), "early_pos_rule_13of17": bool(early_pos), "note": note}
