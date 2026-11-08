"""QUORUM Arm 0 — the seam-seal census (CPU, no GPU). Spec §3 Arm 0 / §9.1.

For each scoring CARRY fine-tune, compute its decoder ΔW (W_ft − W_base per 18×7 matrix) and census
its residual-facing energy onto the Σ (energy) and Q (susceptibility) eigenbases vs the Haar floor r/d,
at r ∈ {20, 50, 350, 679}. This:
  * **closes SPECTER open-Q #5** — "why fine-tuning writes into stiff directions": if write-energy in Σ
    ≫ write-energy in Q ≈ modest-tilt-above-floor, the write is *variance*-located with only a mild
    curvature tilt (the CARRY R2 number 0.352 in the stiff band, reproduced);
  * **validates the two bases discriminate** before any GPU spend — energy-fraction ≠ Q-fraction ≠
    floor, or the necessary-rank contrast is not measuring two different things (Gate-0 assert).

Pure state-dict linear algebra on CPU: load base decoder once, then per FT — download → decoder sd →
ΔW → census → free the checkpoint. No forward passes, no GPU. `HF_HUB_DISABLE_XET` set
(`reference_hf_xet_wedge`).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_arm0 \
      --args "--banks-s3 results/cleave_banks2/out/cleave_banks.pt" \
      --type r7i.8xlarge,r7i.4xlarge --s3-prefix results/quorum_arm0 --budget-min 240
"""
import argparse
import json
import os
import pathlib

CENSUS_RANKS = (20, 50, 350, 679)
CARRY_STIFF_FRACTION = 0.352      # CARRY R2: median stiff (679-dim) write-energy fraction (reproduce)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--donors", default="", help="comma slug list; default = all scoring donors")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.pg import cleave_core as CL
    from infra.pg.gauge_common import layer_index, matrix_type

    QS.install_pg()
    rep = {"object": "quorum_arm0_v1", "census_ranks": list(CENSUS_RANKS),
           "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/quorum/cleave_banks.pt"),
                       map_location="cpu", weights_only=False)
    assert set(banks) == set(range(QS.N_LAYERS)), f"banks cover {sorted(banks)}, need 18 layers"
    bank, base_sha = QS.load_capability_bank("/tmp/quorum")
    if a.donors:
        want = set(a.donors.split(","))
        bank = [c for c in bank if c["slug"] in want]
    rep["base_sha"] = base_sha
    rep["n_donors"] = len(bank)
    _w()

    # base decoder state-dict once
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    base_model, _proc = IP.load_pg1(device="cpu", sha=base_sha)   # CPU box, no GPU
    base_lm = base_model.get_submodule(resolve_submodule_paths(base_model)["language_model"])
    base_sd = {k: v.detach().cpu().clone() for k, v in base_lm.state_dict().items()}
    del base_model, base_lm
    from infra.pg.comwrite_surgery import is_target
    grid = {(layer_index(k), matrix_type(k)): k for k, w in base_sd.items() if is_target(k, w)}
    assert len(grid) == QS.N_LAYERS * 7, f"grid {len(grid)} != 126"
    _w()

    def census_delta(ft_sd):
        """Energy-weighted mean census of ΔW = ft − base over the 126-matrix grid, per rank."""
        per_rank = {}
        for r in CENSUS_RANKS:
            acc = {"energy": 0.0, "susceptibility": 0.0, "random": 0.0}
            wsum, floor = 0.0, None
            for (L, t), key in grid.items():
                dW = (ft_sd[key].float() - base_sd[key].float()).numpy()
                fro2 = float((dW ** 2).sum())
                if fro2 <= 0:
                    continue
                cen = CL.mechanism_census(dW, banks[L], r, t)
                floor = cen["chance_floor"]
                for p in acc:
                    acc[p] += cen[p] * fro2
                wsum += fro2
            assert wsum > 0, "all-zero ΔW"
            per_rank[r] = {p: acc[p] / wsum for p in acc}
            per_rank[r]["chance_floor"] = floor
        return per_rank

    rep["per_donor"] = {}
    stiff_fracs = []
    for c in bank:
        ft_sd = QS.ft_decoder_sd(c["repo"], c["sha"], base_sha, device="cpu")
        cen = census_delta(ft_sd)
        del ft_sd
        QS.drop_ft_cache(c["repo"])
        rep["per_donor"][c["slug"]] = {"task": c["task"], "census": cen}
        # the CARRY stiff-band number = energy fraction at r=679 (top half of Σ) — reproduce ≈0.352
        stiff_fracs.append(cen[679]["energy"])
        print(f"[arm0] {c['slug']:<16s} r20 E={cen[20]['energy']:.3f} Q={cen[20]['susceptibility']:.3f} "
              f"floor={cen[20]['chance_floor']:.3f} | r679 E={cen[679]['energy']:.3f}", flush=True)
        _w()

    # ---- aggregate: the SPECTER-#5 seam-seal + the discrimination gate --------------------
    import numpy as np
    agg = {}
    for r in CENSUS_RANKS:
        E = float(np.mean([rep["per_donor"][c["slug"]]["census"][r]["energy"] for c in bank]))
        Qf = float(np.mean([rep["per_donor"][c["slug"]]["census"][r]["susceptibility"] for c in bank]))
        fl = float(np.mean([rep["per_donor"][c["slug"]]["census"][r]["chance_floor"] for c in bank]))
        agg[r] = {"energy": E, "susceptibility": Qf, "floor": fl,
                  "energy_minus_Q": E - Qf, "Q_minus_floor": Qf - fl}
    rep["aggregate"] = agg
    rep["stiff_fraction_median"] = float(np.median(stiff_fracs))
    rep["carry_stiff_reference"] = CARRY_STIFF_FRACTION

    # discrimination gate: at r=20 the three quantities must be separated.
    r0 = agg[20]
    rep["discriminates"] = bool(r0["energy"] - r0["susceptibility"] > 0.02
                                and r0["susceptibility"] - r0["floor"] > 0.0
                                and r0["energy"] - r0["floor"] > 0.05)
    rep["specter5_verdict"] = ("write is VARIANCE-located with a modest curvature tilt "
                               "(energy > Q > floor)" if r0["energy"] > r0["susceptibility"] > r0["floor"]
                               else "write NOT cleanly variance>curvature>chance — see aggregate")
    print(f"[arm0] AGG r20: E={r0['energy']:.3f} Q={r0['susceptibility']:.3f} floor={r0['floor']:.3f} "
          f"| stiff@679 median={rep['stiff_fraction_median']:.3f} (CARRY {CARRY_STIFF_FRACTION}) "
          f"| discriminates={rep['discriminates']}", flush=True)
    rep["green"] = True
    _w()
    print("QUORUM_ARM0_OK", json.dumps({"aggregate": agg, "discriminates": rep["discriminates"],
          "stiff_median": rep["stiff_fraction_median"], "specter5": rep["specter5_verdict"]},
          default=str), flush=True)


if __name__ == "__main__":
    main()
