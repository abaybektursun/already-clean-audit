"""REFEREE offline gate (spec §9.2) — every NEW instrument replays a planted KNOWN EVENT,
including BOTH of TENANT's FATAL classes (the units error; the diverged-control-as-result).
Pure numpy/scipy; runs on the laptop mirror or a CPU box.

    python -m infra.aws_pg.jobs.referee_offline --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import referee_hook as RH

    rep = {"checks": {}}
    rng = np.random.default_rng(0)
    d, k, o = 512, 16, 128

    # ---- G1: hook algebra (P1) --------------------------------------------------------------
    Q = np.linalg.qr(rng.standard_normal((d, k)))[0]
    u = rng.standard_normal((o, d)) * 1e-4
    g = rng.standard_normal((o, d))
    eta, lam = 3e-6, 4.0
    ut = RH.replaced_update(u, g, Q, eta, lam)
    oo_pre = u - RH.project_in(u, Q)
    oo_post = ut - RH.project_in(ut, Q)
    in_post = RH.project_in(ut, Q)
    in_target = RH.project_in(-eta * lam * g, Q)
    oo_err = float(np.abs(oo_post - oo_pre).max())
    in_err = float(np.abs(in_post - in_target).max())
    # planted leakage event: a NON-orthonormal "projector" must corrupt out-of-core — the
    # gate detects the corruption (this is what P1 fires on in a broken hook)
    Qbad = Q + 0.05 * rng.standard_normal(Q.shape)
    ut_bad = u + (-eta * lam * g - u) @ Qbad @ Qbad.T
    oo_bad = float(np.abs((ut_bad - RH.project_in(ut_bad, Q)) - oo_pre).max())
    ok = oo_err < 1e-12 and in_err < 1e-12 and oo_bad > 1e-6
    rep["checks"]["g1_hook_algebra"] = {
        "oo_untouched_err": oo_err, "in_replaced_err": in_err,
        "planted_leak_detected": oo_bad, "ok": ok,
        "FAIL_IF": "oo/in errs > 1e-12 or the planted non-orthonormal hook NOT flagged"}
    assert ok, rep["checks"]["g1_hook_algebra"]

    # ---- G2: calibration arithmetic on a synthetic fixture ----------------------------------
    n_mat, n_step = 18, 40
    true_oo_perstep = 2.5e-4
    disp_core = rng.uniform(1e-9, 2e-9, n_mat)
    disp_tot = disp_core + (true_oo_perstep * 25) ** 2      # sqrt(oo)=25*true → /25 recovers
    g_norms = np.full(n_mat * n_step, 0.05)
    eta_c, meta = RH.calibrate_eta(disp_tot, disp_core, g_norms, n_steps=25)
    expect = true_oo_perstep / 0.05
    ok = abs(eta_c - expect) / expect < 1e-9
    rep["checks"]["g2_calibration"] = {"eta_c": eta_c, "expected": expect, "ok": ok,
                                       "FAIL_IF": "recovered eta_c != planted ratio"}
    assert ok, rep["checks"]["g2_calibration"]

    # ---- G3: THE TENANT UNITS-ERROR REPLAY (P6) ---------------------------------------------
    # TENANT F2: +0.11 S-scale effect tested against a raw-accuracy SE. The unit-checked
    # power function must REFUSE the mismatched call and accept the converted one.
    S_SLOPE = 0.1580
    refused = False
    try:
        RH.power_one_sided(0.11, 0.0087, 0.05 / 7, "S", "raw_acc")
    except AssertionError:
        refused = True
    p_correct = RH.power_one_sided(0.11 * S_SLOPE, 0.0087, 0.05 / 7, "raw_acc", "raw_acc")
    ok = refused and 0.20 < p_correct < 0.45           # TENANT's corrected ~0.296 regime
    rep["checks"]["g3_units_replay"] = {
        "mismatch_refused": refused, "corrected_power": p_correct, "ok": ok,
        "FAIL_IF": "the mismatched call is NOT refused, or the corrected power leaves the "
                   "known 0.20-0.45 band (TENANT F2 recomputed 0.296)"}
    assert ok, rep["checks"]["g3_units_replay"]

    # ---- G4: divergence protocol (P5; the TN-E12 replay) ------------------------------------
    seq = [RH.divergence_verdict(9.7, 1.02, retried=False),   # TENANT's diverged SGD arm
           RH.divergence_verdict(9.7, 1.02, retried=True),
           RH.divergence_verdict(1.08, 1.02, retried=False)]
    ok = seq == ["retry_halved", "VOID", "ok"]
    rep["checks"]["g4_divergence_protocol"] = {
        "sequence": seq, "ok": ok,
        "FAIL_IF": "the TENANT diverged-arm fixture does not produce retry->VOID, or a "
                   "converged run is not 'ok'"}
    assert ok, rep["checks"]["g4_divergence_protocol"]

    rep["ALL_GREEN"] = True
    (out / "metrics.json").write_text(json.dumps(rep, indent=1))
    print("REFEREE_OFFLINE_OK", json.dumps({k2: True for k2 in rep["checks"]}), flush=True)


if __name__ == "__main__":
    main()
