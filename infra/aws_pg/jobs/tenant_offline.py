"""TENANT offline gate (spec §9.2) — every NEW instrument replays a PLANTED known event,
and every gate names the observation that would make it report FAILURE (the
signals-that-cannot-fail rule). Pure numpy; runs on the laptop mirror or a CPU box.

    python -m infra.aws_pg.jobs.tenant_offline --out <dir>
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

    from infra.pg import tenant_core as TC

    rep = {"checks": {}}
    d, k = 2048, 32
    rng = np.random.default_rng(0)

    # ---- G1: leverage sampler — profile match + Haar-conditional + cheat-detection --------
    # A spiky target profile like the core's (row-leverage ~57x Haar per ERLANGEN).
    spike = np.zeros(d)
    spike[:8] = 0.85                                   # 8 heavy coordinates
    rest = (k - spike.sum()) / (d - 8)
    l_target = spike + np.where(np.arange(d) < 8, 0.0, rest)
    Qs, errs = [], []
    for s in range(6):
        Q, err = TC.leverage_matched_basis(l_target, k, seed=s)
        Qs.append(Q)
        errs.append(err)
    prof_ok = max(errs) <= 0.02
    # Haar-conditional: draw-vs-draw overlap distribution must match draw-vs-"core" overlap
    # where "core" is itself an independent draw playing the core's role.
    core_stand_in = Qs[0]
    ov_core = [TC.subspace_overlap(Q, core_stand_in) for Q in Qs[1:]]
    ov_pair = [TC.subspace_overlap(Qs[i], Qs[j]) for i in range(1, 6) for j in range(i + 1, 6)]
    m_core, m_pair = float(np.mean(ov_core)), float(np.mean(ov_pair))
    haar_ok = abs(m_core - m_pair) <= 0.05
    # PLANTED CHEAT: a sampler that leaks direction content (core + small noise, reprojected)
    cheat = np.linalg.qr(core_stand_in + 0.15 * rng.standard_normal((d, k)))[0]
    ov_cheat = TC.subspace_overlap(cheat, core_stand_in)
    cheat_detected = ov_cheat > m_pair + 0.10          # the test MUST flag it
    rep["checks"]["g1_leverage_sampler"] = {
        "max_profile_err": max(errs), "profile_ok": prof_ok,
        "overlap_vs_core_standin": m_core, "overlap_pairwise": m_pair, "haar_ok": haar_ok,
        "cheat_overlap": ov_cheat, "cheat_detected": cheat_detected,
        "FAIL_IF": "profile err > 0.02, |ov_core - ov_pair| > 0.05, or cheat NOT flagged"}
    assert prof_ok and haar_ok and cheat_detected, f"G1 FAIL: {rep['checks']['g1_leverage_sampler']}"

    # ---- G2: confinement checker — confined passes, leaked fails ---------------------------
    Qc = np.linalg.qr(rng.standard_normal((d, k)))[0]
    B_in = rng.standard_normal((64, k)) @ Qc.T                 # rows inside span
    B_leak = B_in + 0.3 * rng.standard_normal((64, d))
    f_in = TC.confinement_frac(B_in, Qc)
    f_leak = TC.confinement_frac(B_leak, Qc)
    ok = f_in < 1e-10 and f_leak > 0.05
    rep["checks"]["g2_confinement"] = {"frac_confined": f_in, "frac_leaked": f_leak, "ok": ok,
                                       "FAIL_IF": "confined > 1e-10 or planted leak < 0.05"}
    assert ok, f"G2 FAIL: {rep['checks']['g2_confinement']}"

    # ---- G3: q/k product-space — planted cancellation ~0, random ~1 ------------------------
    # head-sliced convention: Wq_head and Wk_head both (head_dim, d_in); the read product is
    # Wq^T Wk (d_in, d_in). Planted EXACT rank-1 cancellation:
    #   dWq = u p^T with p = Wq0^T v;  dWk = v (-q)^T with q = Wk0^T u
    #   => (dWq)^T Wk0 = p q^T  and  Wq0^T dWk = -p q^T  — combined exactly zero.
    o, din = 256, 512
    Wq0 = rng.standard_normal((o, din)) / np.sqrt(din)
    Wk0 = rng.standard_normal((o, din)) / np.sqrt(din)
    Qq = np.linalg.qr(rng.standard_normal((din, 16)))[0]
    Qk = np.linalg.qr(rng.standard_normal((din, 16)))[0]
    u, v = rng.standard_normal(o), rng.standard_normal(o)
    p, q = Wq0.T @ v, Wk0.T @ u
    dWq = np.outer(u, p) * 1e-3
    dWk = np.outer(v, -q) * 1e-3
    r_cancel = TC.qk_product_cancellation(dWq, dWk, Wq0, Wk0, Qq, Qk)
    dWk_rand = rng.standard_normal((o, din)) * 1e-3
    r_rand = TC.qk_product_cancellation(dWq, dWk_rand, Wq0, Wk0, Qq, Qk)
    ok = r_cancel["index"] < 0.05 and r_rand["index"] > 0.5
    rep["checks"]["g3_qk_product"] = {"planted_cancel_index": r_cancel["index"],
                                      "random_index": r_rand["index"], "ok": ok,
                                      "FAIL_IF": "planted >= 0.05 or random <= 0.5"}
    assert ok, f"G3 FAIL: {rep['checks']['g3_qk_product']}"

    # ---- G4: lag-autocorrelation — planted take-revert ~ -1 at lag 1, iid ~ 0 --------------
    T = 400
    v = rng.standard_normal(16)
    V_rev = np.array([v if t % 2 == 0 else -v for t in range(T)]) \
        + 0.05 * rng.standard_normal((T, 16))
    rho_rev = TC.lag_autocorr(V_rev, 3)
    V_iid = rng.standard_normal((T, 16))
    rho_iid = TC.lag_autocorr(V_iid, 3)
    ok = rho_rev[0] < -0.9 and abs(rho_iid[0]) < 0.1
    rep["checks"]["g4_lag_autocorr"] = {"planted_lag1": rho_rev[0], "iid_lag1": rho_iid[0],
                                        "planted_lags": rho_rev, "ok": ok,
                                        "FAIL_IF": "planted lag-1 >= -0.9 or |iid lag-1| >= 0.1"}
    assert ok, f"G4 FAIL: {rep['checks']['g4_lag_autocorr']}"

    rep["ALL_GREEN"] = True
    (out / "metrics.json").write_text(json.dumps(rep, indent=1))
    print("TENANT_OFFLINE_OK", json.dumps({k: v.get("ok", True) is not False
                                           for k, v in rep["checks"].items()}), flush=True)


if __name__ == "__main__":
    main()
