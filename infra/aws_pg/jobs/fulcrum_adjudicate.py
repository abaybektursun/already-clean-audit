"""FULCRUM adjudication + the frozen §6 verdict (CPU, pure analysis). Spec §6 / §7 / §9.4.

Reads the sealed confirmation (`fulcrum_lega --phase confirm`), the discovery freeze (g*), the preflight
synthetic controls (the floor/ceiling), and:
  * per capability, interpolates n*_energy to the frozen g* on the curv and the two random paths →
    E_curv(c) = n*(curv@g*) − median_seed n*(random@g*), paired by capability;
  * the paired bootstrap median E_curv + 95% CI; the random-vs-random null band; the monotone
    θ-dose-response (Spearman(θ, n*_curv) < 0 at BH-FDR); the footprint-moved gate;
  * runs the **frozen §6 adjudicator ONCE** (`fulcrum_core.adjudicate_fulcrum`): VOID → H_lever →
    H_bystander, with the two synthetic controls as the ceiling/floor;
  * the F4 basis-match map (n*_energy vs curvature footprint, θ=0 vs θ=1 — where the intervention moves
    points along QUORUM's ρ=0.65 line).

Pure numpy/scipy — every input is an artifact already on S3.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.fulcrum_adjudicate \
      --args "--conf results/fulcrum_conf/out/metrics.json --disc results/fulcrum_disc/out/metrics.json \
              --preflight results/fulcrum_preflight/out/metrics.json" \
      --type r7i.xlarge --no-weights --s3-prefix results/fulcrum_verdict --budget-min 20
"""
import argparse
import json
import os
import pathlib


def _bh_reject(pvals, q=0.10):
    import numpy as np
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, n + 1)) / n
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
    ap.add_argument("--preflight", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.aws_pg.jobs import _fulcrum_shared as FS  # noqa: F401 (surface parity)
    from infra.pg import fulcrum_core as FC
    QS.install_pg()

    def load(key):
        return json.loads(pathlib.Path(QS.fetch_s3(key, f"/tmp/fulcrum/{key.replace('/', '_')}")).read_text())

    conf, disc, pre = load(a.conf), load(a.disc), load(a.preflight)
    rep = {"object": "fulcrum_verdict_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"},
           "sources": {"conf": a.conf, "disc": a.disc, "preflight": a.preflight}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    g_star = float(disc["freeze"]["g_star"])
    rep["g_star"] = g_star
    rep["rho_break"] = disc["freeze"]["rho_break"]

    def curves(paths, key):
        """(gain{θ}, nstar{θ}) over the θ where n* is defined, for one path key."""
        rows = paths.get(key)
        if not rows:
            return None, None
        gain = {r["theta"]: r["gain"] for r in rows if r.get("n_star") is not None}
        nstar = {r["theta"]: r["n_star"] for r in rows if r.get("n_star") is not None}
        return (gain, nstar) if gain else (None, None)

    caps = [c for c in conf["capabilities"] if "skipped" not in c and "paths" in c]
    rep["n_learners"] = len(caps)
    per_cap, e_curvs, null_es, monos, mono_p = [], [], [], [], []
    footprint_moved_flags = []
    for c in caps:
        paths = c["paths"]
        gc, nc = curves(paths, "susceptibility")
        rk = sorted(k for k in paths if k.startswith("random"))
        if gc is None or not rk:
            continue
        n_curv = FC.interp_nstar_at_gstar(nc, gc, g_star)
        rand_ns, rand_reach = [], []
        for k in rk:
            gr, nr = curves(paths, k)
            if gr is None:
                continue
            ni = FC.interp_nstar_at_gstar(nr, gr, g_star)
            rand_reach.append(ni)
        rand_ok = [x["n_star"] for x in rand_reach if x["reachable"]]
        row = {"slug": c["slug"], "task": c["task"], "gain_full": c["gain_full"],
               "nstar_curv_gstar": n_curv["n_star"], "curv_reachable": n_curv["reachable"],
               "nstar_random_gstar": (float(np.median(rand_ok)) if rand_ok else None),
               "nstar_energy_theta0": c.get("nstar_energy_theta0")}
        if n_curv["reachable"] and rand_ok:
            e = float(n_curv["n_star"] - np.median(rand_ok))
            row["e_curv"] = e
            e_curvs.append(e)
            if len(rand_ok) >= 2:
                null_es.append(abs(rand_ok[0] - rand_ok[1]))
        # monotone θ-dose-response on the curv path
        md = FC.monotone_dose_response(nc)
        row["mono_rho"] = md["rho"]
        row["mono_p"] = md["p_value"]
        monos.append(md["rho"] if md["rho"] is not None else 0.0)
        mono_p.append(md["p_value"] if md["p_value"] is not None else 1.0)
        # footprint moved: curvature fraction falls with θ; energy fraction ~stable
        cr = paths["susceptibility"]
        q0 = next((r["curvature_fraction"] for r in cr if r["theta"] == 0.0), None)
        q1 = next((r["curvature_fraction"] for r in cr if r["theta"] == 1.0), None)
        e0 = next((r["energy_fraction"] for r in cr if r["theta"] == 0.0), None)
        e1 = next((r["energy_fraction"] for r in cr if r["theta"] == 1.0), None)
        # footprint moved = the CURVATURE fraction fell substantially (≥2×) with θ. The earlier
        # "energy-fraction stable ≤0.10" secondary was miscalibrated: removing curvature (spread over the
        # spectrum) also drops the top-679 ENERGY census (Q and Σ's top-679 overlap), so energy-stable is
        # not the right condition — the curvature-fraction collapse is the signal (measured ~10× for all).
        moved = (q0 is not None and q1 is not None and q0 > 0 and q1 < 0.5 * q0)
        row["footprint_curv_theta0"], row["footprint_curv_theta1"] = q0, q1
        row["footprint_energy_theta0"], row["footprint_energy_theta1"] = e0, e1
        row["footprint_moved"] = bool(moved)
        footprint_moved_flags.append(moved)
        # EXPLORATORY full-strip (θ=1) effect: n*_curv(θ=1) − n*_random(θ=1), GAIN-CONFOUNDED (at θ=1
        # the two arms remove different amounts of gain — the matched-gain g* estimand exists precisely
        # to control this). Reported as the maximal curvature-removal effect the matched-gain contrast
        # cannot see (the reachability confound); NOT the primary.
        nc1 = nc.get(1.0)
        nr1 = [curves(paths, k)[1].get(1.0) for k in rk if curves(paths, k)[1]]
        nr1 = [x for x in nr1 if x is not None]
        if nc1 is not None and nr1:
            row["fullstrip_curv_theta1"] = nc1
            row["fullstrip_random_theta1"] = float(np.median(nr1))
            row["fullstrip_effect"] = float(nc1 - np.median(nr1))
            row["gain_ret_curv_theta1"] = next((r["gain_retention"] for r in cr if r["theta"] == 1.0), None)
        per_cap.append(row)
    rep["per_capability"] = per_cap
    assert per_cap, "no confirmation capability yielded a curv + random path"
    fullstrip = [r["fullstrip_effect"] for r in per_cap if r.get("fullstrip_effect") is not None]
    rep["fullstrip_median"] = float(np.median(fullstrip)) if fullstrip else None
    rep["fullstrip_note"] = ("EXPLORATORY, gain-confounded: n*_curv(θ=1) − n*_random(θ=1) over caps — the "
                             "maximal curvature-removal effect the matched-gain primary compresses away "
                             "(reachability confound). NOT a causal claim (§8).")

    # ---- the primary statistics ----------------------------------------------------------
    e_ci = FC.paired_e_curv_ci(e_curvs)
    rep["e_curv_ci"] = e_ci
    rep["e_curv_values"] = e_curvs
    null_band_hi = float(np.percentile(null_es, 95)) if null_es else 0.0
    rep["null_band_hi"] = null_band_hi
    rep["n_reach_gstar"] = len(e_curvs)

    # monotonicity fraction (BH-FDR across caps; count rho<0 that also pass)
    rej = _bh_reject(mono_p, q=0.10)
    mono_pass = [bool(rej[i] and monos[i] < 0) for i in range(len(monos))]
    monotone_frac = float(np.mean(mono_pass)) if mono_pass else 0.0
    rep["monotone"] = {"rhos": monos, "p_values": mono_p, "bh_reject": rej,
                       "pass_and_negative": mono_pass, "fraction": monotone_frac}
    footprint_moved = bool(np.mean(footprint_moved_flags) >= 0.5) if footprint_moved_flags else False
    rep["footprint_moved_fraction"] = float(np.mean(footprint_moved_flags)) if footprint_moved_flags else 0.0

    # ---- the two synthetic controls (from the preflight) + the frozen separation gate ------
    ctrls = pre.get("controls", {})
    floor_e = ctrls.get("floor_energy_only", {}).get("e_curv")
    ceil_e = ctrls.get("ceiling_curvature_only", {}).get("e_curv")
    n0_ceil = ctrls.get("ceiling_curvature_only", {}).get("n0")
    c_nb = ctrls.get("ceiling_null_band")
    f_nb = ctrls.get("floor_null_band")
    assert c_nb is not None and f_nb is not None, "preflight did not emit the control null bands"
    # frozen gates, RELATIVE to each control's random-vs-random null band (the matched-gain confound
    # compresses E_curv below n0): ceiling clears −ceiling_null_band; floor within ±max(null bands).
    pos_tol = max(float(f_nb), float(c_nb))
    ceil_thresh = -float(c_nb)
    rep["controls"] = {"floor_e_curv": floor_e, "ceiling_e_curv": ceil_e, "n0_ceiling": n0_ceil,
                       "floor_null_band": f_nb, "ceiling_null_band": c_nb,
                       "pos_floor_tol": pos_tol, "ceiling_thresh": ceil_thresh}

    # shuffle control (if the susceptibility_shuffle arm was run) — its E must collapse to the null band
    shuffle_es = []
    for c in caps:
        paths = c["paths"]
        gs_, ns_ = curves(paths, "susceptibility_shuffle")
        rk = sorted(k for k in paths if k.startswith("random"))
        if gs_ is None or not rk:
            continue
        n_sh = FC.interp_nstar_at_gstar(ns_, gs_, g_star)
        gr, nr = curves(paths, rk[0])
        if gr is None or not n_sh["reachable"]:
            continue
        ni = FC.interp_nstar_at_gstar(nr, gr, g_star)
        if ni["reachable"]:
            shuffle_es.append(n_sh["n_star"] - ni["n_star"])
    shuffle_median = float(np.median(shuffle_es)) if shuffle_es else None
    shuffle_collapses = bool(shuffle_median is None or abs(shuffle_median) <= null_band_hi + 1e-9)
    rep["shuffle"] = {"n": len(shuffle_es), "median_e": shuffle_median, "collapses": shuffle_collapses}

    # ---- the frozen verdict (applied ONCE) -----------------------------------------------
    verdict = FC.adjudicate_fulcrum(
        e_curv_ci=e_ci, null_band_hi=null_band_hi, pos_floor_e_curv=floor_e, ceiling_e_curv=ceil_e,
        pos_floor_tol=pos_tol, ceiling_thresh=ceil_thresh, monotone_frac_curv=monotone_frac,
        footprint_moved=footprint_moved, any_reach_gstar=len(e_curvs) > 0,
        shuffle_collapses=shuffle_collapses)
    rep["verdict"] = verdict
    print("FULCRUM_VERDICT", verdict["verdict"], "|", verdict["reason"][:180], flush=True)

    # ---- F4 basis-match map: n*_energy vs curvature footprint, θ=0 vs θ=1 -----------------
    f4 = []
    for c in caps:
        cr = c["paths"].get("susceptibility", [])
        r0 = next((r for r in cr if r["theta"] == 0.0), None)
        r1 = next((r for r in cr if r["theta"] == 1.0), None)
        if r0 and r1:
            f4.append({"slug": c["slug"], "footprint_theta0": r0["curvature_fraction"],
                       "nstar_theta0": r0.get("n_star"), "footprint_theta1": r1["curvature_fraction"],
                       "nstar_theta1": r1.get("n_star")})
    rep["f4_basis_match"] = f4
    rep["goodfire_fold"] = ("verdict-level only (firewall): H_lever ⇒ curvature footprint is a "
                            "targeted-unlearning lever consistent with parameter-decomposition removal; "
                            "H_bystander ⇒ QUORUM's ρ=0.65 corrected to a confound. No PG2 number "
                            "imported as a threshold.")
    rep["green"] = True
    _w()
    print("FULCRUM_VERDICT_OK", json.dumps(
        {"verdict": verdict["verdict"], "e_curv_ci": e_ci, "null_band_hi": null_band_hi,
         "monotone_frac": monotone_frac, "footprint_moved": footprint_moved,
         "floor_e_curv": floor_e, "ceiling_e_curv": ceil_e, "n_reach": len(e_curvs)}, default=str),
        flush=True)


if __name__ == "__main__":
    main()
