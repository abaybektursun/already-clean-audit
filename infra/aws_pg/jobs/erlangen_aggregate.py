"""ERLANGEN aggregate — assemble every arm's banked artifacts and apply the FROZEN rules.

Runs on CPU. Reads the arm JSONs from S3, then:
  T  clusters the banked fingerprint tensors (inventory, stability, cross-unit transfer)
  A  carries the arm's own P2 verdict through
  X  carries the compatibility matrix + mirror-v2 through
  M  assembles the 5×5 directed cost matrix from the per-cell jobs, builds the seed-based noise
     model, and applies the amendment-A3 axiom tests in BOTH gauges
  B  pools the per-run gradient/displacement ratios into the §1.2 rule
  D  summarises the witness family vs the matched-random control
  G  the compression table (no confirmatory claim)
  F-theory BH over the frozen m = 4 family; everything else is labelled census.

    python -m infra.aws_pg.jobs.erlangen_aggregate --out DIR [--n-boot 10000]

No try/except (CLAUDE.md) except at the S3 fetch boundary, where a MISSING artifact is a
legitimate state (an arm that has not landed yet) and is recorded as `pending`, never inferred.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

BUCKET = "pg-lineage-769063704696"
M_TASKS = ("cococap", "refcoco", "textvqa", "vqav2", "okvqa")


def _sh(args):
    import subprocess
    return subprocess.run(args, capture_output=True)


def _get(key):
    """Fetch a JSON artifact via the AWS CLI. boto3 is NOT installed in the job venv — every S3
    touch on this AMI goes through the CLI (the same path run_job.sh uses). A MISSING artifact is a
    legitimate state (an arm that has not landed) and reads as None; any other failure RAISES."""
    r = _sh(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--region", "us-east-1"])
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace")
        if any(m in err for m in ("Not Found", "does not exist", "NoSuchKey", "404")):
            return None
        raise RuntimeError(f"s3 cp {key} failed: {err[:300]}")
    return json.loads(r.stdout)


def _ls(prefix):
    r = _sh(["aws", "s3", "ls", f"s3://{BUCKET}/{prefix}", "--recursive", "--region", "us-east-1"])
    if r.returncode != 0 and not r.stdout:
        return []
    return [ln.split()[-1] for ln in r.stdout.decode().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--freeze-key", default="results/erlangen_gate0b/out/erlangen_freeze.json")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.lacuna_core import bh_reject     # REUSED: the house BH step-up, verbatim

    t0 = time.time()
    rep = {"object": "erlangen_aggregate_v1"}

    def bank():
        (out / "erlangen_aggregate.json").write_text(json.dumps(rep, indent=2))

    bank()
    rep["freeze"] = _get(a.freeze_key)
    assert rep["freeze"] and rep["freeze"].get("G0_GREEN"), "Gate-0 is not green"

    # ================= Arm T ===================================================================
    units = {}
    for k in [x for x in _ls("results/erlangen_t_") if x.endswith(".json") and "/out/" in x]:
        d = _get(k)
        if d and d.get("DONE"):
            units[d["unit"]] = d
    rep["T"] = {"units_measured": sorted(units), "n_units": len(units)}
    if units:
        ref = "pg1" if "pg1" in units else sorted(units)[0]
        comps = [c for c in units[ref]["components"]
                 if all(f"{c}|{kk}" in units[ref]["cells"] for kk in P.INTERVENTIONS)]

        def fp(u):
            cells = units[u]["cells"]
            base = cells["__base__"]
            F, A_, B_ = [], [], []
            for c in comps:
                row, ra, rb = [], [], []
                for kk in P.INTERVENTIONS:
                    cell = cells.get(f"{c}|{kk}")
                    nanrow = [float("nan")] * len(P.READOUT_NAMES)
                    if cell is None:
                        row += nanrow; ra += nanrow; rb += nanrow
                        continue
                    row += [x - y for x, y in zip(cell["vec"], base["vec"])]
                    ra += [x - y for x, y in zip(cell["half_a"], base["half_a"])]
                    rb += [x - y for x, y in zip(cell["half_b"], base["half_b"])]
                F.append(row); A_.append(ra); B_.append(rb)
            return (np.nan_to_num(np.asarray(F)), np.nan_to_num(np.asarray(A_)),
                    np.nan_to_num(np.asarray(B_)))

        Fref = fp(ref)[0]
        scale = EC.column_scale(Fref)["scale"]
        rep["T"]["n_components"] = len(comps)
        rep["T"]["column_scale_from"] = ref
        rep["T"]["k_selection"] = EC.select_k(Fref, scale, range(2, 9), n_split=40)
        kstar = rep["T"]["k_selection"]["k_star"]
        rep["T"]["inventory"], labs = {}, {}
        for u in units:
            F, A_, B_ = fp(u)
            labs[u] = EC.cluster(EC.preprocess(F, scale), kstar)
            floor = EC.fingerprint_noise_floor(A_, B_, scale)
            rep["T"]["inventory"][u] = {
                "k": kstar, "n_unmeasurable": floor["n_unmeasurable"],
                "stability": EC.split_half_stability(F, scale, kstar, n_split=60),
                "cluster_sizes": [int((np.asarray(labs[u]) == i).sum()) for i in range(kstar)],
                "labels": [int(x) for x in labs[u]],
                "forward_validity": units[u].get("forward_validity")}
        rep["T"]["components"] = comps
        rep["T"]["transfer"] = {f"{u}|{v}": EC.adjusted_rand(labs[u], labs[v])
                                for u in labs for v in labs if u < v}
        rep["T"]["transfer_mean"] = (float(np.mean(list(rep["T"]["transfer"].values())))
                                     if rep["T"]["transfer"] else float("nan"))
    bank()

    # ================= Arms A / X / D / G (verdicts computed in-arm) ============================
    # N3: a unit whose forward is invalid can never be interpreted, so its Arm-X row/column is
    # marked VOID here rather than being averaged into the compatibility structure.
    for nm, key in (("A", "results/erlangen_a/out/erlangen_a.json"),
                    ("X", "results/erlangen_x2/out/erlangen_x.json"),
                    ("D", "results/erlangen_d3/out/erlangen_d.json"),
                    ("G", "results/erlangen_g/out/erlangen_g.json")):
        d = _get(key)
        rep[nm] = ({k: v for k, v in d.items()
                    if k not in ("cells", "pairs", "singles", "oracle_calls", "param_shift",
                                 "commutators", "absorption")} if d else {"status": "pending"})
    if rep["X"].get("forward_validity"):
        bad = [u for u, v in rep["X"]["forward_validity"].items() if not v.get("valid")]
        rep["X"]["void_units_n3"] = bad
        rep["X"]["n3_note"] = ("rows/columns for these units FAILED the forward-validity gate "
                               "(no perception channel) and are not interpreted")
        # CRITIC FINDING 1. The unit-level compat_matrix correctly drops N3-VOID units, but the
        # per-GROUP statistic averaged them back in — and an N3-VOID unit has a near-zero or
        # negative deletion normaliser, so its cells carry enormous magnitudes. `embedding` read
        # −2.64 for that reason alone. Recomputed here over N3-valid cells only, excluding the bad
        # unit as BOTH host and donor; the raw (contaminated) value is kept beside it so the
        # correction is auditable rather than silent.
        ux = rep["X"].get("compat_matrix_units") or []
        keep = [i for i, u in enumerate(ux) if u not in bad]
        clean = {}
        for g, M in (rep["X"].get("compat_by_group") or {}).items():
            A = np.asarray(M, dtype=float)
            vals = [A[i, j] for i in keep for j in keep if i != j]
            vals = [v for v in vals if v == v]
            raw = [A[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])
                   if i != j and A[i, j] == A[i, j]]
            clean[g] = {"mean_n3_valid": float(np.mean(vals)) if vals else float("nan"),
                        "n_cells_n3_valid": len(vals),
                        "mean_contaminated_superseded": float(np.mean(raw)) if raw else float("nan")}
        rep["X"]["compat_by_group_n3_valid"] = clean
        rep["X"]["compat_group_note"] = ("headline group means are `mean_n3_valid`; "
                                         "`mean_contaminated_superseded` is the pre-correction "
                                         "value that included the N3-VOID unit")
    if rep["G"].get("hop_results"):
        # Complete the grammar's DENSE fallback and sweep ε.
        # The search loop enumerated {rank, sparse, gain}; `dense` (exact ΔW, ε = 0) is a v0 term
        # too, and without it a hop where NO structured program meets ε reports an UNDEFINED ratio
        # instead of the correct 1.0 ("the typed grammar buys nothing here"). dense's cost and
        # fidelity are analytic, so this is completed from numbers already banked — not a rule
        # change and not a re-run. The frozen ε is the headline; the sweep is reported beside it.
        eps_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        for h, r in rep["G"]["hop_results"].items():
            dense = r.get("dense_dl")
            sweep = {}
            for e in eps_grid:
                ok = [q for q in r["programs"] if q["fidelity"] <= e]
                ty = [q["dl"] for q in ok if q["term"] in ("rank", "gain")] + ([dense] if dense else [])
                fr = [q["dl"] for q in ok if q["term"] == "sparse"] + ([dense] if dense else [])
                sweep[str(e)] = {"typed_dl": min(ty) if ty else None,
                                 "free_dl": min(fr) if fr else None,
                                 "ratio": (min(ty) / min(fr)) if (ty and fr) else None,
                                 "n_typed_meeting_eps": len([q for q in ok
                                                             if q["term"] in ("rank", "gain")]),
                                 "n_sparse_meeting_eps": len([q for q in ok
                                                              if q["term"] == "sparse"])}
            r["eps_sweep"] = sweep
            r["ratio_with_dense_fallback"] = sweep[str(rep["G"]["eps"])]["ratio"] \
                if str(rep["G"]["eps"]) in sweep else None
            r["best_fidelity_any_program"] = min((q["fidelity"] for q in r["programs"]),
                                                 default=None)
        rep["G"]["hop_results"] = {
            h: {k: v for k, v in r.items() if k not in ("programs", "child_vec", "parent_vec")}
            for h, r in rep["G"]["hop_results"].items()}
    bank()

    # ================= Arm M + B ================================================================
    cells = {}
    for k in [x for x in _ls("results/erlangen_m_") if x.endswith(".json") and "/out/" in x]:
        d = _get(k)
        if not d:
            continue
        # a fleet JSON banks incrementally, so PARTIAL fleets contribute their finished cells —
        # waiting for rep["DONE"] would discard real measurements on a crashed instance
        for _key, c in (d.get("cells") or {}).items():
            if c.get("DONE"):
                cells[(c["src"], c["dst"], c["seed"])] = c
    # DEGENERATE DIAGONAL. d(A→A) starts AT the criterion anchor (identical weights), so its
    # headroom is exactly 0 and s = 0/0. The semantically correct cost of "adapt ft-A to task A" is
    # ZERO — you are already there — but the cell job's generic path scores an undefined s as no
    # progress and returns the full budget, which would invert P4. Completed here from the cell's
    # own `degenerate_headroom` flag; the flag was written before any matrix existed.
    n_degen = 0
    for c in cells.values():
        if c.get("degenerate_headroom"):
            c["cost_auc"] = 0.0
            c["censored"] = False
            c["examples_to_criterion"] = 0
            c["cost_by_budget"] = {k: 0.0 for k in c.get("cost_by_budget", {})}
            n_degen += 1
    rep["M_degenerate_cells"] = n_degen
    # Anchor-validity, recomputed here over OFF-DIAGONAL sources only (the in-job gate compared
    # against all sources including the diagonal, whose nll_0 equals the anchor by construction and
    # so could never satisfy a strict inequality). Diagnostic: it flags a destination column whose
    # published fine-tune is not the best model on its own task under this binding.
    rep["M_anchor_gate"] = {}
    for dst in {c["dst"] for c in cells.values()}:
        ref = next(c["nll_ref"] for c in cells.values() if c["dst"] == dst)
        off = [c["nll_0"] for c in cells.values() if c["dst"] == dst and c["src"] != dst]
        rep["M_anchor_gate"][dst] = {
            "nll_ref": ref, "min_offdiag_nll_0": min(off) if off else None,
            "valid": bool(off and ref < min(off)),
            "margin": (min(off) - ref) if off else None}
    rep["M"] = {"n_cells": len(cells),
                "cells": {f"{s}|{t}|{z}": {"cost": c["cost_auc"], "censored": c["censored"],
                                           "nll_0": c["nll_0"], "nll_ref": c["nll_ref"],
                                           "examples_to_criterion": c["examples_to_criterion"],
                                           "cost_by_budget": c["cost_by_budget"]}
                          for (s, t, z), c in cells.items()}}
    tasks = [t for t in M_TASKS if any(s == t or d == t for (s, d, _z) in cells)]
    if len(tasks) >= 3:
        n = len(tasks)
        D = np.full((n, n), np.nan)
        CEN, reps = {}, {}
        for i, s in enumerate(tasks):
            for j, t in enumerate(tasks):
                vals = [c["cost_auc"] for (ss, tt, _z), c in cells.items() if ss == s and tt == t]
                cens = [c["censored"] for (ss, tt, _z), c in cells.items() if ss == s and tt == t]
                if not vals:
                    continue
                D[i, j] = float(np.mean(vals))
                CEN[(s, t)] = bool(any(cens))
                if len(vals) >= 2:
                    reps[f"{s}|{t}"] = vals
        seed_sd = EC.pooled_seed_sd(reps)
        np.fill_diagonal(D, np.nan_to_num(np.diag(D), nan=0.0))
        rep["M"].update({"tasks": tasks, "cost_matrix": D.tolist(), "seed_sd": seed_sd,
                         "censored": {f"{k[0]}|{k[1]}": v for k, v in CEN.items()}})
        # CRITIC FINDING 9: report EVERY censored pair, not the subset that happened to be tabulated
        rep["M"]["censored_pairs"] = sorted(f"{k[0]}->{k[1]}" for k, v in CEN.items() if v)
        rep["M"]["n_censored_pairs"] = len(rep["M"]["censored_pairs"])
        rep["M"]["censoring_note"] = ("every censored cell is a LOWER BOUND on cost; they enter "
                                      "only through triples whose verdict cannot flip (A3c)")
        diag = [D[i, i] for i in range(n) if D[i, i] == D[i, i]]
        offd = [D[i, j] for i in range(n) for j in range(n) if i != j and D[i, j] == D[i, j]]
        rep["M"]["p4"] = {"diag_mean": float(np.mean(diag)) if diag else float("nan"),
                          "offdiag_mean": float(np.mean(offd)) if offd else float("nan"),
                          "passes": bool(diag and offd and np.mean(diag) < np.mean(offd)),
                          "failure_means": "the cost instrument cannot rank a task against itself "
                                           "→ Arm M VOID"}
        if seed_sd["sd"] is not None and np.isfinite(D).all():
            # The per-cell σ IS the pooled seed term: each cell is one trajectory, so there is no
            # separate within-cell item term to add. Stated rather than silently assumed (A3b).
            SIG = np.full((n, n), float(seed_sd["sd"]))
            Dl, SIGl = np.log1p(D), SIG / (1.0 + D)
            tri_d = EC.triangle_test(D.tolist(), SIG.tolist(), tasks, censored=CEN)
            sym_d = EC.symmetry_test(D.tolist(), SIG.tolist(), tasks)
            mask = np.asarray(tri_d["evaluable_mask"])
            scale_off = float(np.median(D[~np.eye(n, dtype=bool)]))
            dmin = 0.05 * scale_off
            # CRITIC FINDING 6. The log-gauge effect-size floor was `log1p(dmin)` = 4.73, but the
            # largest attainable log-gauge asymmetry is 1.72 — so the secondary gauge reported
            # p ≡ 1.0 for ANY data and `gauge_stable=False` was a structural constant, not a
            # finding. An effect size transforms by the DELTA METHOD, not by the transform itself:
            # g(d)=log1p(d) ⇒ g'(d)=1/(1+d), so a floor of `dmin` at scale `s` maps to dmin/(1+s).
            dmin_log = dmin / (1.0 + scale_off)
            rep["M"]["delta_min"] = dmin
            rep["M"]["delta_min_loggauge"] = dmin_log
            rep["M"]["delta_min_loggauge_note"] = (
                "delta-method transform of the effect-size floor; the previous log1p(dmin) made "
                "the secondary gauge unfailable (critic finding 6)")
            rep["M"]["verdict"] = EC.metric_verdict(
                EC.axiom_bootstrap(D, SIG, "symmetry", n_boot=a.n_boot, seed=0, delta_min=dmin),
                EC.axiom_bootstrap(D, SIG, "triangle", censored_mask=mask, n_boot=a.n_boot,
                                   seed=0, delta_min=dmin),
                EC.axiom_bootstrap(Dl, SIGl, "symmetry", n_boot=a.n_boot, seed=0,
                                   delta_min=dmin_log),
                EC.axiom_bootstrap(Dl, SIGl, "triangle", censored_mask=mask, n_boot=a.n_boot,
                                   seed=0, delta_min=dmin_log),
                "ok", sym_desc=sym_d, tri_desc=tri_d)
            rep["M"]["triangle_detail"] = {k: v for k, v in tri_d.items() if k != "evaluable_mask"}
            rep["M"]["symmetry_detail"] = sym_d
            # CRITIC FINDING 6, second pass. Fixing the effect-size floor by the delta method
            # revived the log-gauge SYMMETRY test (p 1e-4, agreeing with the primary), but the
            # log-gauge TRIANGLE test remains structurally unable to disconfirm: log1p is CONCAVE,
            # so log1p(a+b) ≤ log1p(a)+log1p(b) and subadditivity holds almost by identity. And a
            # monotone map cannot change the SIGN of a symmetry violation, so the log gauge adds
            # nothing there either. The A3a gauge-sensitivity check is therefore NOT INFORMATIVE as
            # implemented, and the "gauge-dependent" caveat it produced is withdrawn rather than
            # reported as a finding.
            rep["M"]["gauge_check_status"] = {
                "informative": False,
                "symmetry": "monotone maps preserve the sign of |d(A→B) − d(B→A)|, so the log "
                            "gauge cannot disagree — it agrees here (p 1e-4) by construction",
                "triangle": "log1p is concave ⇒ subadditivity is near-automatic, so the log gauge "
                            "cannot disconfirm a triangle violation (p ≡ 1.0)",
                "consequence": "the A3a secondary-gauge caveat is WITHDRAWN; the triangle verdict "
                               "is scoped by the anchor-validity argument alone, which is the "
                               "argument that actually carries it"}
            # ANCHOR-SCOPED RE-RUN. A destination whose published fine-tune is not the best model on
            # its own task under this binding has an uninterpretable column (refcoco's only roster
            # checkpoint is a SEGMENTATION fine-tune; okvqa's margin is a tie). The full-matrix
            # verdict is reported as computed AND repeated on the sub-matrix of anchor-valid
            # destinations, with whatever power that leaves — the scoped run is a disclosure, not a
            # replacement, and the frozen rule is applied identically to both.
            good = [t for t in tasks if rep["M_anchor_gate"].get(t, {}).get("valid")]
            rep["M"]["anchor_valid_tasks"] = good
            if len(good) >= 3:
                idx = [tasks.index(t) for t in good]
                Dg = D[np.ix_(idx, idx)]
                if np.isfinite(Dg).all():
                    SIGg = np.full(Dg.shape, float(seed_sd["sd"]))
                    Dgl, SIGgl = np.log1p(Dg), SIGg / (1.0 + Dg)
                    CENg = {(x, y): CEN.get((x, y), False) for x in good for y in good}
                    trig = EC.triangle_test(Dg.tolist(), SIGg.tolist(), good, censored=CENg)
                    symg = EC.symmetry_test(Dg.tolist(), SIGg.tolist(), good)
                    mg = np.asarray(trig["evaluable_mask"])
                    scale_g = float(np.median(Dg[~np.eye(len(good), dtype=bool)]))
                    dming = 0.05 * scale_g
                    dming_log = dming / (1.0 + scale_g)
                    rep["M"]["verdict_anchor_scoped"] = EC.metric_verdict(
                        EC.axiom_bootstrap(Dg, SIGg, "symmetry", n_boot=a.n_boot, seed=0,
                                           delta_min=dming),
                        EC.axiom_bootstrap(Dg, SIGg, "triangle", censored_mask=mg,
                                           n_boot=a.n_boot, seed=0, delta_min=dming),
                        EC.axiom_bootstrap(Dgl, SIGgl, "symmetry", n_boot=a.n_boot, seed=0,
                                           delta_min=dming_log),
                        EC.axiom_bootstrap(Dgl, SIGgl, "triangle", censored_mask=mg,
                                           n_boot=a.n_boot, seed=0, delta_min=dming_log),
                        "ok", sym_desc=symg, tri_desc=trig)
                    rep["M"]["cost_matrix_anchor_scoped"] = Dg.tolist()
                else:
                    rep["M"]["verdict_anchor_scoped"] = {
                        "verdict": "VOID", "reason": "anchor-valid sub-matrix incomplete"}
        else:
            rep["M"]["verdict"] = {"verdict": "VOID", "reason": seed_sd["note"],
                                   "complete_matrix": bool(np.isfinite(D).all())}
    bank()

    gr = [c["arm_b"]["gradient_ratio_mean"] for c in cells.values()
          if c.get("arm_b", {}).get("gradient_ratio_mean") is not None]
    dr = [c["arm_b"]["displacement_ratio"] for c in cells.values()
          if c.get("arm_b", {}).get("displacement_ratio") is not None]
    if gr and dr:
        rep["B"] = EC.boundary_verdict(np.asarray(gr), np.ones(len(gr)),
                                       np.asarray(dr), np.ones(len(dr)))
        rep["B"]["n_runs"] = len(gr)
        rep["B"]["note"] = ("each run's value is ALREADY core/matched-random, so the denominator "
                            "array is 1 by construction and the CI is over RUNS")
        rep["B"]["curvature"] = [c["curvature"] for c in cells.values() if c.get("curvature")][:5]
    else:
        rep["B"] = {"verdict": "pending", "n_runs": 0}
    bank()

    # ================= F-theory family (BH, m disclosed) ========================================
    mv = (rep.get("M", {}).get("verdict") or {})
    # CRITIC FINDING 5. `H_wall_gradient_ratio` previously occupied a BH slot with a LITERAL
    # (0.001 / 0.999), not a statistic. A hardcoded constant cannot be corrected for multiplicity.
    # It is now declared for what it always was — a CI-based GATE ("does the ratio's lower bound
    # clear the matched-random floor?") — and removed from the family, which drops to m = 3.
    fam = {"H_metric_symmetry": mv.get("symmetry_p"),
           "H_metric_triangle": mv.get("triangle_p"),
           "H_mirror_v2": ((rep.get("X") or {}).get("mirror_v2", {}).get("A", {}) or {})
           .get("p_two_sided")}
    # CRITIC FINDING 14: state the power floor beside the mirror-v2 register. With n labelled
    # units the permutation group has n! members, so no p below 1/n! is attainable at any n_perm.
    mvv = ((rep.get("X") or {}).get("mirror_v2") or {})
    if mvv:
        from math import factorial
        n_units_x = len((rep["X"].get("compat_matrix_units") or []))
        for tag, v in mvv.items():
            if isinstance(v, dict) and n_units_x:
                v.setdefault("n_units", n_units_x)
                v.setdefault("min_attainable_p", 1.0 / factorial(n_units_x))
        rep["X"]["mirror_v2_power_note"] = (
            f"{n_units_x} units ⇒ {factorial(n_units_x)} distinct permutations ⇒ the smallest "
            f"attainable p is {1.0/factorial(n_units_x):.4f}; 'not supported' is registered "
            f"against that ceiling, and this null is UNVERIFIABLE as a strong negative")
    rep["F_theory"] = {"pvals": fam, "bh": bh_reject(fam, q=0.05),
                       "m_declared": 3,
                       "H_wall_is_a_gate": {
                           "decided_by": "gradient-ratio CI vs the matched-random floor, then "
                                         "displacement-ratio, in the frozen §6 order",
                           "verdict": rep["B"].get("verdict"),
                           "gradient_ci": rep["B"].get("gradient_ci"),
                           "why_not_in_bh": "it yields a CI-based gate decision, not a p-value; "
                                            "the previous slot held a hardcoded constant "
                                            "(critic finding 5)"},
                       "note": "m counts only the tests that ran; VOID ids are listed, never "
                               "silently dropped"}
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[agg] DONE T_units={rep['T'].get('n_units')} M_cells={rep['M'].get('n_cells')} "
          f"M={(mv or {}).get('verdict')} B={rep['B'].get('verdict')} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
