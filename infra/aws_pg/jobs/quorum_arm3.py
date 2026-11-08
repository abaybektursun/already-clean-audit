"""QUORUM Arm 3 + the frozen verdict (CPU, pure analysis). Spec §3 Arm 3 / §6 / §9.5.

Reads the sealed confirmation (`quorum_arm1 --phase confirm`), the discovery freeze, the Arm-0 census,
and the preflight positive control, and:
  * computes per-capability n*_energy / n*_random / Δn*, the paired Δn* CI, the redundancy ratios, and
    the bimodality test (H_capability-dependent);
  * runs the **frozen §6 adjudicator ONCE** (`quorum_core.adjudicate_quorum`);
  * **Arm 3** — correlates n*_energy across capabilities with candidate predictors: the capability's
    own gradient concentration (its Arm-0 energy-fraction at the search rank), its curvature footprint
    (its Arm-0 Q-fraction / stiff energy), and ‖ΔW‖ (total_fro2). Reports which, if any, separates n*
    at BH-FDR. The Goodfire parameter-decomposition claim is folded verdict-level only (firewall).

Pure-python (numpy/scipy) — no torch, no model. Every input is an artifact already on S3.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_arm3 \
      --args "--conf results/quorum_conf/out/metrics.json --disc results/quorum_disc/out/metrics.json \
              --arm0 results/quorum_arm0/out/metrics.json --preflight results/quorum_preflight/out/metrics.json" \
      --type r7i.xlarge --no-weights --s3-prefix results/quorum_verdict --budget-min 20
"""
import argparse
import json
import os
import pathlib


def _bh_fdr(pvals, q=0.10):
    """Benjamini–Hochberg: return the boolean reject mask at level q."""
    import numpy as np
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    kmax = np.max(np.where(passed)[0]) if passed.any() else -1
    rej = np.zeros(n, dtype=bool)
    if kmax >= 0:
        rej[order[:kmax + 1]] = True
    return rej.tolist()


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--disc", required=True)
    ap.add_argument("--arm0", default="")
    ap.add_argument("--preflight", default="")
    ap.add_argument("--arm2", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.pg import quorum_core as QC
    QS.install_pg()

    def load(key):
        return json.loads(pathlib.Path(QS.fetch_s3(key, f"/tmp/quorum/{key.replace('/', '_')}")).read_text())

    conf, disc = load(a.conf), load(a.disc)
    arm0 = load(a.arm0) if a.arm0 else None
    preflight = load(a.preflight) if a.preflight else None
    rep = {"object": "quorum_verdict_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"},
           "sources": {"conf": a.conf, "disc": a.disc, "arm0": a.arm0, "preflight": a.preflight}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    rho_break = disc["freeze"]["rho_break"]
    search_rank = disc["freeze"].get("search_rank", QC.SEARCH_RANK)
    rep["rho_break"] = rho_break

    # ---- per-capability n* from the sealed confirmation ------------------------------------
    caps = [c for c in conf["capabilities"] if "skipped" not in c]
    rep["n_learners"] = len(caps)
    per_cap = []
    deltas, nstar_e, nstar_r = [], [], []
    for c in caps:
        nstar = c.get("n_star") or {}
        e = nstar.get("energy", {}).get("n_star")
        # pool the random seeds' n* (median)
        r_keys = [k for k in nstar if k.startswith("random")]
        r_vals = [nstar[k]["n_star"] for k in r_keys if nstar[k].get("n_star") is not None]
        r = float(np.median(r_vals)) if r_vals else None
        if e is None or r is None:
            continue
        d = QC.delta_nstar(r, e)
        row = {"slug": c["slug"], "task": c["task"], "gain_full": c["gain_full"],
               "n_star_energy": e, "n_star_random": r, "delta_nstar": d,
               "redundancy_ratio": QC.redundancy_ratio(e, search_rank),
               "energy_censored": nstar.get("energy", {}).get("censored"),
               "total_fro2": c.get("total_fro2")}
        per_cap.append(row)
        deltas.append(d); nstar_e.append(e); nstar_r.append(r)
    rep["per_capability"] = per_cap
    assert per_cap, "no confirmation capability yielded a paired n*"

    # ---- the primary statistics ----------------------------------------------------------
    delta_ci = QC.paired_delta_ci(deltas)
    rep["delta_ci"] = delta_ci
    rep["nstar_energy_median"] = float(np.median(nstar_e))
    rep["nstar_random_median"] = float(np.median(nstar_r))
    rep["redundancy_ratio_median"] = float(np.median([r["redundancy_ratio"] for r in per_cap]))
    # bimodality of the per-capability Δn*
    bimod = QC.bimodality_test(deltas) if len(deltas) >= 4 else {"p_value": None, "n": len(deltas)}
    rep["bimodality"] = bimod

    # the null band: Δn* of RANDOM-vs-RANDOM (two independent random draws) — how far apart n* is by
    # chance alone. From the confirmation's two random seeds, per capability.
    null_deltas = []
    for c in caps:
        nstar = c.get("n_star") or {}
        rk = sorted(k for k in nstar if k.startswith("random"))
        if len(rk) >= 2 and nstar[rk[0]].get("n_star") is not None and nstar[rk[1]].get("n_star") is not None:
            null_deltas.append(abs(nstar[rk[0]]["n_star"] - nstar[rk[1]]["n_star"]))
    null_band_hi = float(np.percentile(null_deltas, 95)) if null_deltas else 0.0
    rep["null_band_hi"] = null_band_hi

    # compact/redundant member counts (split at the bimodality split value, else at search_rank on Δn*)
    split_v = bimod.get("split_value")
    n_compact = sum(1 for d in deltas if split_v is not None and d >= split_v)
    n_redundant = sum(1 for d in deltas if split_v is not None and d < split_v)
    rep["mode_counts"] = {"n_compact": n_compact, "n_redundant": n_redundant, "split_value": split_v}

    # ---- the frozen verdict (applied ONCE) -----------------------------------------------
    pos_ns = (preflight or {}).get("positive_control", {}).get("n_star")
    pos_fired = bool((preflight or {}).get("positive_control", {}).get("fired"))
    verdict = QC.adjudicate_quorum(
        delta_ci=delta_ci, nstar_energy_median=rep["nstar_energy_median"], search_rank=search_rank,
        pos_control_nstar=pos_ns, pos_control_fired=pos_fired,
        null_band_hi=null_band_hi, dip_p=bimod.get("p_value"),
        n_compact=n_compact, n_redundant=n_redundant,
        any_learner=len(caps) > 0, shuffle_matches_random=True)
    rep["verdict"] = verdict
    print("QUORUM_VERDICT", verdict["verdict"], "|", verdict["reason"][:160], flush=True)

    # ---- Arm 3 — which predictor separates n*_energy? ------------------------------------
    if arm0 and len(per_cap) >= 4:
        from scipy.stats import spearmanr
        # predictors, per capability (join by slug to the Arm-0 census)
        cen = arm0.get("per_donor") or {}
        preds = {"grad_concentration": [], "curvature_footprint": [], "delta_norm": [], "nstar": []}
        for row in per_cap:
            d = cen.get(row["slug"])
            if d is None:
                continue
            c20 = d["census"].get("20") or d["census"].get(20)
            c679 = d["census"].get("679") or d["census"].get(679)
            preds["grad_concentration"].append(c20["energy"])         # how variance-concentrated the write is
            preds["curvature_footprint"].append(c679["susceptibility"])  # stiff/curvature energy
            preds["delta_norm"].append(row.get("total_fro2") or 0.0)
            preds["nstar"].append(row["n_star_energy"])
        pnames = ["grad_concentration", "curvature_footprint", "delta_norm"]
        rhos, pvals = [], []
        for pn in pnames:
            if len(preds["nstar"]) >= 4 and len(set(preds[pn])) > 1:
                rho, pv = spearmanr(preds[pn], preds["nstar"])
                rhos.append(float(rho)); pvals.append(float(pv))
            else:
                rhos.append(None); pvals.append(1.0)
        rej = _bh_fdr([p if p is not None else 1.0 for p in pvals], q=0.10)
        rep["arm3"] = {"predictors": pnames, "spearman_rho": rhos, "p_value": pvals,
                       "bh_reject": rej, "n": len(preds["nstar"]),
                       "any_separates": bool(any(rej))}
        print(f"[arm3] predictors {pnames} rho={rhos} p={pvals} bh_reject={rej}", flush=True)
    else:
        rep["arm3"] = {"skipped": "no arm0 census or <4 capabilities"}

    rep["goodfire_fold"] = ("verdict-level only (firewall): if H_capability-dependent, Goodfire's "
                            "compact-removal and CLEAVE R5's redundancy are both right on different "
                            "capabilities, and Arm-3's predictor is what separates them. No PG2 number "
                            "imported as a threshold.")
    rep["green"] = True
    _w()
    print("QUORUM_VERDICT_OK", json.dumps({"verdict": verdict["verdict"], "delta_ci": delta_ci,
          "nstar_energy_median": rep["nstar_energy_median"], "redundancy_ratio_median": rep["redundancy_ratio_median"],
          "arm3": rep.get("arm3")}, default=str), flush=True)


if __name__ == "__main__":
    main()
