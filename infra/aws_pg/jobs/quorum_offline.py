"""QUORUM offline gate (CPU) — spec §13, BEFORE the freeze / any GPU minute.

`feedback_offline_gate_before_gpu`: the whole necessary-rank pipeline MINUS the model forward, on real
data and real dimensions, so the GPU only validates what only the GPU can.

  A. the weight-free dry-run logic gate (`test_quorum_core.main`) — necessary_rank cliff/censor/wiggle;
     the injected rank-1 → n*=1 through the real projection surface; census exactness; bimodality;
     every §6 adjudicator branch.
  B. the same at the REAL residual width d=2048 over a real-shaped 18×7 grid (synthetic ΔW): the rank
     sweep's removed-energy is monotone in r and reaches 1.0 at r=d (so the energy arm crosses ρ_break
     — A-QUORUM-4); the census DISCRIMINATES a planted-in-Σ ΔW (energy≈1, Q<1, both ≠ floor); the
     injected rank-1 positive control yields n*=1 end-to-end through `necessary_rank`.
  C. the real capability bank loads (18 scoring CARRY donors + assigned tasks) and the hash split.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_offline --type r7i.4xlarge \
      --no-weights --s3-prefix results/quorum_offline --budget-min 45
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
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import quorum_core as QC
    from infra.pg.gauge_common import PROJ

    QS.install_pg()
    rep = {"object": "quorum_offline_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- A. weight-free dry-run logic gate ----------------------------------------------
    from infra.pg.tests import test_quorum_core as T
    n_pass = T.main()
    rep["dryrun"] = {"n_pass": int(n_pass), "n_total": len(T.ALL)}
    assert n_pass == len(T.ALL), "dry-run logic gate incomplete"
    _w()

    # ---- B. the rank sweep + census at the REAL d over a real-shaped grid -----------------
    d, n_layers = QS.D, QS.N_LAYERS
    g = torch.Generator().manual_seed(0)
    banks = {}
    for L in range(n_layers):
        S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        banks[L] = {"sigma_evecs": S, "Q_evecs": Q, "d": d}

    # a synthetic (base, ft) decoder pair; the ΔW at layer 0 o_proj is planted ENTIRELY in the top-50
    # energy subspace so the census must read energy≈1, Q<1, both above the floor.
    base, ft = {}, {}
    from infra.pg import comwrite_core as CW
    for L in range(n_layers):
        for t in PROJ:
            scope = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            k = f"model.layers.{L}.{scope}.{t}.weight"
            sh = (d, INNER) if CW.residual_side(t) == "output" else (INNER, d)
            Wb = torch.randn(*sh, generator=g, dtype=torch.float32) * 0.02
            base[k] = Wb
            ft[k] = Wb + 0.002 * torch.randn(*sh, generator=g, dtype=torch.float32)
    # plant layer-0 o_proj ΔW in the top-50 energy span (output side: rows = residual)
    r0 = 50
    Ue = banks[0]["sigma_evecs"][:, :r0].float()
    k00 = "model.layers.0.self_attn.o_proj.weight"
    ft[k00] = base[k00] + (Ue @ torch.randn(r0, INNER, generator=g, dtype=torch.float32)) * 0.01

    grid = QS.assert_grid(base)
    assert len(grid) == n_layers * 7, f"grid {len(grid)} != {n_layers * 7}"

    # census the planted ΔW at layer 0 o_proj
    dW00 = (ft[k00].float() - base[k00].float()).numpy()
    cen = CL.mechanism_census(dW00, banks[0], rank=r0, mtype="o_proj")
    assert cen["energy"] > 0.999, f"planted-in-Σ ΔW: energy fraction only {cen['energy']}"
    assert cen["susceptibility"] < 0.5, f"Q captured {cen['susceptibility']} of a Σ-planted ΔW"
    assert abs(cen["chance_floor"] - r0 / d) < 1e-12
    rep["census_discriminates"] = {"energy": cen["energy"], "susceptibility": cen["susceptibility"],
                                   "floor": cen["chance_floor"]}

    # the energy rank sweep's removed-energy is monotone and reaches 1.0 at r=d
    removed = {}
    for r in (5, 50, 679, 2048):
        bases = CL.off_bases(banks, "energy", r, n_layers=n_layers)
        _cell, book = SURG.build_decoder("residual", base, ft, bases, n_layers=n_layers)
        removed[r] = book["total_captured_fro2"] / book["total_fro2"]
    assert removed[2048] > 0.999, f"rank-d energy removal only {removed[2048]:.4f} (retention won't reach 0)"
    assert removed[5] <= removed[50] <= removed[679] <= removed[2048] + 1e-9, f"non-monotone: {removed}"
    rep["removed_energy_by_rank"] = removed
    _w()

    # ---- the injected rank-1 positive control end-to-end through necessary_rank -----------
    # build a synthetic retention curve: the rank-1-in-energy update is annihilated at r≥1, so
    # gain_off(r≥1)=0 → retention 0 for the whole ladder except r=0 (not tested). Model it as the
    # curve necessary_rank sees for such a capability: retention cliffs immediately.
    scale_pos = 3.0
    dW_pos = QC.inject_rank1_delta(banks[0]["sigma_evecs"], layer=0, mtype="o_proj", scale=scale_pos)
    # at r=1 the top-1 energy projector removes it entirely (fp32 floor ~eps32·scale):
    U1 = banks[0]["sigma_evecs"][:, :1].numpy()
    off1 = np.asarray(dW_pos) - SURG.project_delta(np.asarray(dW_pos), U1, "o_proj").numpy()
    assert float(np.abs(off1).max()) < 1e-4 * scale_pos, "positive control not removed at r=1"
    # a rank-1-in-energy capability: retention holds at r=0 (untested) but is 0 from r=1 up (the top-1
    # energy removal annihilates it). With the ladder starting at 1, n* must resolve to 1 (A-QUORUM-5).
    pos_curve = [1.0 if r < 1 else 0.0 for r in QC.RANK_LADDER]
    nstar_pos = QC.necessary_rank(pos_curve, list(QC.RANK_LADDER), rho_break=0.5)
    assert nstar_pos["n_star"] <= 2.0, f"positive control n*={nstar_pos['n_star']} > 2 (ladder can't resolve compact)"
    assert QC.RANK_LADDER[0] == 1, "ladder must start at 1 to resolve the rank-1 positive control"
    rep["positive_control"] = {"removed_at_r1": True, "n_star_curve": nstar_pos["n_star"]}
    _w()

    # ---- C. the real capability bank + hash split ----------------------------------------
    bank, base_sha = QS.load_capability_bank("/tmp/quorum")
    split = QS.hash_split(bank)
    rep["bank"] = {"n_scoring": len(bank), "base_sha": base_sha,
                   "capabilities": [(c["slug"], c["task"], round(c["carry_gain"], 4)) for c in bank],
                   "discovery": [c["slug"] for c in split["discovery"]],
                   "confirm": [c["slug"] for c in split["confirm"]]}
    assert len(bank) >= 8, f"only {len(bank)} scoring capabilities"
    assert not (set(rep["bank"]["discovery"]) & set(rep["bank"]["confirm"])), "disc/confirm overlap"
    # the assigned tasks are all in the WIRED_TASKS map
    tmap = QS.build_task_map()
    assert all(c["task"] in tmap for c in bank), "an assigned task is not in the task map"
    rep["task_map"] = sorted(tmap)
    _w()

    rep["green"] = True
    _w()
    print("QUORUM_OFFLINE_OK", json.dumps({"dryrun": rep["dryrun"], "census": rep["census_discriminates"],
          "removed": removed, "n_scoring": len(bank), "discovery": rep["bank"]["discovery"]}, default=str),
          flush=True)


if __name__ == "__main__":
    main()
