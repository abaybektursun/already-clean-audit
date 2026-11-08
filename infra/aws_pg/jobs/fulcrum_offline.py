"""FULCRUM offline gate (CPU) — spec §13, BEFORE the freeze / any GPU minute.

`feedback_offline_gate_before_gpu`: the whole Leg-A pipeline MINUS the model forward, on real banks and
real dimensions, so the GPU only validates what only the GPU can.

  A. the weight-free dry-run logic gate (`test_fulcrum_core.main`) — theta_family ⊥/identity, the two
     synthetic controls (E_curv≈0 floor, strong-negative ceiling), interp exactness, footprint-moves,
     every §6 adjudicator branch.
  B. at the REAL residual width d=2048 over a real-shaped 18×7 grid (synthetic ΔW): the whole-decoder
     `combine_alpha(ft,shared,base,θ)` equals ΔW − θ·project_delta(ΔW,U,·) at every CERT8 matrix and is
     the FULL ΔW at every excluded (non-CERT8) matrix; `footprint_census` reads a curvature-planted ΔW
     (curvature fraction ≫ floor) and its curvature fraction FALLS monotone with θ under the curv
     projection; and the matched-gain reachability confound is quantified (random floor bounds g*).
  C. the real bank loads; the frozen CERT8 support is RE-VERIFIED against `cleave_banks2`
     (`certified_layers(banks, 679) == {0,3,5,6,11,13,14,15}`); the hash split matches QUORUM's.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.fulcrum_offline \
      --args "--banks-s3 results/cleave_banks2/out/cleave_banks.pt" \
      --type r7i.4xlarge --no-weights --s3-prefix results/fulcrum_offline --budget-min 45
"""
import argparse
import json
import os
import pathlib

INNER = 64


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", default="results/cleave_banks2/out/cleave_banks.pt")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _fulcrum_shared as FS
    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import fulcrum_core as FC
    from infra.pg.gauge_common import PROJ

    QS.install_pg()
    rep = {"object": "fulcrum_offline_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- A. weight-free dry-run logic gate -----------------------------------------------
    from infra.pg.tests import test_fulcrum_core as T
    n_pass = T.main()
    rep["dryrun"] = {"n_pass": int(n_pass), "n_total": len(T.ALL)}
    assert n_pass == len(T.ALL), "dry-run logic gate incomplete"
    _w()

    # ---- B. combine_alpha ≡ theta_family + footprint-moves at real d over a real grid -----
    d, n_layers = QS.D, QS.N_LAYERS
    g = torch.Generator().manual_seed(0)
    banks = {}
    for L in range(n_layers):
        S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        banks[L] = {"sigma_evecs": S, "Q_evecs": Q, "d": d}

    from infra.pg import comwrite_core as CW
    base, ft = {}, {}
    for L in range(n_layers):
        for t in PROJ:
            scope = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            k = f"model.layers.{L}.{scope}.{t}.weight"
            sh = (d, INNER) if CW.residual_side(t) == "output" else (INNER, d)
            Wb = torch.randn(*sh, generator=g, dtype=torch.float32) * 0.02
            base[k] = Wb
            # plant ΔW in the top-64 CURVATURE (Q) subspace on the residual side (so footprint reads it)
            r0 = 64
            if CW.residual_side(t) == "output":
                UQ = banks[L]["Q_evecs"][:, :r0].float()
                ft[k] = Wb + (UQ @ torch.randn(r0, INNER, generator=g, dtype=torch.float32)) * 0.01
            else:
                UQ = banks[L]["Q_evecs"][:, :r0].float()
                ft[k] = Wb + (torch.randn(INNER, r0, generator=g, dtype=torch.float32) @ UQ.T) * 0.01
    QS.assert_grid(base)

    cert = list(FC.CERT8)
    bases, excluded = CL.off_bases_restricted(banks, "susceptibility", FC.CURV_RANK, cert,
                                              n_layers=n_layers)
    ft_cell, _ = SURG.build_decoder("ft", base, ft, bases, excluded=excluded, n_layers=n_layers)
    shared_cell, _ = SURG.build_decoder("shared", base, ft, bases, excluded=excluded, n_layers=n_layers)

    # combine_alpha ≡ ΔW − θ·project_delta at CERT8; ≡ full ΔW at excluded layers
    from infra.pg.gauge_common import layer_index, matrix_type
    grid = {(layer_index(k), matrix_type(k)): k for k in base if SURG.is_target(k, base[k])}
    max_err_cert, max_err_excl = 0.0, 0.0
    for th in FC.THETA_LADDER:
        ftθ = CL.combine_alpha(ft_cell, shared_cell, base, float(th))
        for (L, t), key in grid.items():
            dW = (ft[key].float() - base[key].float())
            got = (ftθ[key].float() - base[key].float())
            if L in FC.CERT8:
                U = bases[(L, t)]
                want = dW - float(th) * SURG.project_delta(dW, U, t)
                max_err_cert = max(max_err_cert, float((got - want).abs().max()))
            else:
                max_err_excl = max(max_err_excl, float((got - dW).abs().max()))
    assert max_err_cert < 1e-3, f"combine_alpha ≠ theta_family on CERT8: max err {max_err_cert}"
    assert max_err_excl < 1e-3, f"excluded layers not full ΔW: max err {max_err_excl}"
    rep["combine_alpha_identity"] = {"max_err_cert8": max_err_cert, "max_err_excluded": max_err_excl}
    _w()

    # footprint census reads the curvature plant, and curvature fraction FALLS with θ under curv
    fp_by_theta = []
    for th in FC.THETA_LADDER:
        ftθ = CL.combine_alpha(ft_cell, shared_cell, base, float(th))
        fp = FC.footprint_census(ftθ, base, banks, rank=FC.CURV_RANK, layers=cert)
        fp_by_theta.append({"theta": float(th), **fp})
    qfr = [x["curvature"] for x in fp_by_theta]
    assert qfr[0] > fp_by_theta[0]["chance_floor"], f"planted curvature not above floor: {qfr[0]}"
    assert all(qfr[i] >= qfr[i + 1] - 1e-9 for i in range(len(qfr) - 1)), f"Q-frac not falling: {qfr}"
    rep["footprint_moves"] = {"curvature_by_theta": qfr,
                              "energy_by_theta": [x["energy"] for x in fp_by_theta],
                              "chance_floor": fp_by_theta[0]["chance_floor"]}
    _w()

    # matched-gain reachability confound: a rank-679 Haar strip captures ≈679/2048 of a random ΔW, so
    # the random arm's gain-retention floor ≈ 1 − 679/2048 — g* cannot go below this (the partial-reveal
    # bound; a documented threat, not a blocker — the sign test survives, offline-sim confirmed).
    UR = FC._side_block  # noqa: F841 (documentation marker)
    rand_capture = FC.CURV_RANK / d
    rep["reachability_note"] = {
        "random_capture_frac_per_layer": rand_capture,
        "expected_random_gain_floor": 1.0 - rand_capture,
        "note": ("matched-gain g* is bounded below by the random arm's rank-679 capture (~0.33), so "
                 "E_curv measures a PARTIAL curvature reveal — a conservative lower-bound on the full "
                 "curvature-removal effect. Sign test survives (offline sim: curv-dominant ⇒ strong "
                 "negative, energy-dominant ⇒ ≈0). Reported as a §8 threat.")}
    _w()

    # ---- C. real bank + CERT8 re-verification + hash split -------------------------------
    real_banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/fulcrum/cleave_banks.pt"),
                            map_location="cpu", weights_only=False)
    assert set(real_banks) == set(range(n_layers)), f"banks cover {sorted(real_banks)}, need 18"
    measured_cert = FS.verify_cert_support(real_banks, rank=FC.CURV_RANK, expect=FC.CERT8)
    rep["cert_support"] = {"rank": FC.CURV_RANK, "measured": list(measured_cert),
                           "frozen_expect": list(FC.CERT8)}
    # discrimination sanity: at r=679 the certified layers' Q vs energy vs floor must differ (planted
    # real ΔW would be needed for a full check; here we assert the banks carry both bases at d=2048)
    for L in FC.CERT8:
        assert int(real_banks[L]["d"]) == d, f"layer {L} d != {d}"
        assert real_banks[L]["Q_evecs"].shape[1] >= FC.CURV_RANK, f"layer {L} Q too small"
        assert real_banks[L]["sigma_evecs"].shape[1] >= FC.CURV_RANK, f"layer {L} Σ too small"

    bank, base_sha = QS.load_capability_bank("/tmp/fulcrum")
    split = QS.hash_split(bank)
    rep["bank"] = {"n_scoring": len(bank), "base_sha": base_sha,
                   "capabilities": [(c["slug"], c["task"], round(c["carry_gain"], 4)) for c in bank],
                   "discovery": [c["slug"] for c in split["discovery"]],
                   "confirm": [c["slug"] for c in split["confirm"]]}
    assert len(bank) >= 8, f"only {len(bank)} scoring capabilities"
    assert not (set(rep["bank"]["discovery"]) & set(rep["bank"]["confirm"])), "disc/confirm overlap"
    tmap = QS.build_task_map()
    assert all(c["task"] in tmap for c in bank), "an assigned task is not in the task map"
    _w()

    rep["green"] = True
    _w()
    print("FULCRUM_OFFLINE_OK", json.dumps(
        {"dryrun": rep["dryrun"], "combine_alpha_identity": rep["combine_alpha_identity"],
         "cert_support": measured_cert, "footprint_q_by_theta": qfr,
         "n_scoring": len(bank), "discovery": rep["bank"]["discovery"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
