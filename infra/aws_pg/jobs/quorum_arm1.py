"""QUORUM Arm 1 — the necessary-rank spectrum (the MDE, GPU). Spec §3 Arm 1 / §9.4–9.5.

Per capability c (a CARRY fine-tune): install the full ΔW → gain_full; then rank-sweep removal of the
top-r subspace on each basis {energy, random, susceptibility} → retention R(r) → n*_basis(c); the
primary gap Δn*(c) = n*_random − n*_energy.

  --phase discovery : sweep the discovery capabilities on all bases; freeze ρ_break (the knee of the
                      RANDOM-basis curve, pooled), the ladder, the isotonic knots.
  --phase confirm   : the sealed capabilities at the frozen ρ_break → Δn* → the paired CI.

`rep["capabilities"]` is bound BEFORE the loop so each completed capability is checkpointed
(`feedback_fleet_incremental_save`). `--donors` shards the (expensive: ~6GB download each) bank.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_arm1 \
      --args "--phase discovery --banks-s3 results/cleave_banks2/out/cleave_banks.pt" \
      --type p5.4xlarge,g6e.8xlarge,g6e.4xlarge --s3-prefix results/quorum_disc --budget-min 900
"""
import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("discovery", "confirm"), required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--frozen-s3", default="", help="confirm: the discovery metrics.json with ρ_break")
    ap.add_argument("--donors", default="", help="slug list override (sharding)")
    ap.add_argument("--policies", default="energy,random,susceptibility")
    ap.add_argument("--n-items", type=int, default=128)
    ap.add_argument("--seeds", default="1337,2024", help="random-basis draws (energy/Q deterministic)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import cleave_core as CL
    from infra.pg import quorum_core as QC

    QS.install_pg()
    device = "cuda"
    policies = [p for p in a.policies.split(",") if p.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    rep = {"object": f"quorum_arm1_{a.phase}_v1", "phase": a.phase, "policies": policies,
           "seeds": seeds, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("banks"):
        banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/quorum/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    assert set(banks) == set(range(QS.N_LAYERS)), "banks must cover all 18 layers"
    # Q certified support (for the secondary susceptibility arm; energy/random span all 18)
    q_support = {int(r): CL.certified_layers(banks, int(r), n_layers=QS.N_LAYERS) for r in QC.RANK_LADDER}
    rep["q_support_by_rank"] = {r: v for r, v in q_support.items()}

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = QS.load_model("/tmp/quorum")
    rep["pinned_sha"] = sha
    bank, base_sha = QS.load_capability_bank("/tmp/quorum")
    split = QS.hash_split(bank)
    caps = split["discovery"] if a.phase == "discovery" else split["confirm"]
    if a.donors:
        want = set(a.donors.split(","))
        caps = [c for c in bank if c["slug"] in want]
    rep["capability_slugs"] = [c["slug"] for c in caps]
    rep["split"] = {"discovery": [c["slug"] for c in split["discovery"]],
                    "confirm": [c["slug"] for c in split["confirm"]]}

    lm, paths, vt_orig = QS.lm_bundle(model)
    base_lm_sd = QS.decoder_sd(lm, device=device)
    QS.assert_grid(base_lm_sd)
    install = QS.make_installer(model, lm, paths, vt_orig)
    snap = QS.snapshot_base(model)

    def reset():
        QS.reset_model(model, snap)

    tmap = QS.build_task_map()

    # ---- frozen ρ_break (confirm) --------------------------------------------------------
    rho_break = None
    if a.phase == "confirm":
        assert a.frozen_s3, "confirm requires --frozen-s3"
        disc = json.loads(pathlib.Path(QS.fetch_s3(a.frozen_s3, "/tmp/quorum/disc.json")).read_text())
        rho_break = float(disc["freeze"]["rho_break"])
        rep["frozen"] = {"rho_break": rho_break, "source": a.frozen_s3}
        print(f"[arm1] FROZEN ρ_break={rho_break:.4f}", flush=True)
    _w()

    caps_out = []
    rep["capabilities"] = caps_out        # bound BEFORE the loop (incremental save)
    for c in caps:
        task = tmap[c["task"]]
        rows = QS.task_rows(task, a.n_items)
        reset()
        with tm.stage(f"{c['slug']}/base"):
            base_res = QS.score(model, proc, task, rows, device)

        def read(tag, _base=base_res, _task=task, _rows=rows):
            return QS.gain_of(QS.score(model, proc, _task, _rows, device), _base)

        with tm.stage(f"{c['slug']}/download"):
            ft_lm_sd = QS.ft_decoder_sd(c["repo"], c["sha"], base_sha, device=device)
        with tm.stage(f"{c['slug']}/full"):
            full_res, total_fro2 = QS.install_full_delta(base_lm_sd, ft_lm_sd, banks, install=install,
                                                         read=read, reset=reset, seed=QS.SEED, device=device)
        gain_full = full_res["gain"]
        entry = {"slug": c["slug"], "task": c["task"], "carry_gain": c["carry_gain"],
                 "gain_full": gain_full, "total_fro2": total_fro2, "base_score": base_res["score"]}
        # A-QUORUM-1: re-screen by QUORUM's own MIN_GAIN_FULL (retention needs a real denominator)
        if abs(gain_full) < QC.MIN_GAIN_FULL:
            entry["skipped"] = f"gain_full {gain_full:+.4f} < MIN_GAIN_FULL {QC.MIN_GAIN_FULL}"
            caps_out.append(entry)
            print(f"[arm1] SKIP {c['slug']}: gain_full {gain_full:+.4f} < {QC.MIN_GAIN_FULL}", flush=True)
            del ft_lm_sd; torch.cuda.empty_cache(); _w()
            continue

        entry["sweeps"] = {}
        entry["n_star"] = {}
        for policy in policies:
            support = q_support if policy in ("susceptibility", "susceptibility_shuffle") else None
            # energy/Q are deterministic (one draw); random uses the seeds
            pol_seeds = seeds if policy in ("random",) else [QS.SEED]
            for sd_seed in pol_seeds:
                with tm.stage(f"{c['slug']}/{policy}/s{sd_seed}"):
                    sweep = QS.rank_sweep(base_lm_sd, ft_lm_sd, banks, policy=policy,
                                          ladder=QC.RANK_LADDER, install=install, read=read,
                                          reset=reset, gain_full=gain_full, seed=sd_seed,
                                          device=device, support=(support if support else None))
                key = policy if policy != "random" else f"random_s{sd_seed}"
                entry["sweeps"][key] = sweep
                if rho_break is not None and sweep:
                    ns = QC.necessary_rank([s["retention"] for s in sweep],
                                           [s["rank"] for s in sweep], rho_break=rho_break)
                    entry["n_star"][key] = ns
                _w()
        caps_out.append(entry)
        del ft_lm_sd
        QS.drop_ft_cache(c["repo"])
        torch.cuda.empty_cache()
        _w()

    learners = [c for c in caps_out if "skipped" not in c]
    rep["n_learners"] = len(learners)

    # =====================================================================================
    if a.phase == "discovery":
        # ρ_break = knee of the RANDOM-basis retention curve, pooled over discovery learners+seeds.
        pooled_r, pooled_y = [], []
        for c in learners:
            for key, sweep in c.get("sweeps", {}).items():
                if key.startswith("random"):
                    for s in sweep:
                        pooled_r.append(s["rank"]); pooled_y.append(s["retention"])
        assert pooled_r, "no random-basis sweeps to derive ρ_break"
        order = np.argsort(pooled_r)
        pr = np.asarray(pooled_r)[order]; py = np.asarray(pooled_y)[order]
        # average retention per rank (the pooled null curve)
        uniq = sorted(set(pr.tolist()))
        mean_y = [float(py[pr == u].mean()) for u in uniq]
        knee = QC.rho_break_from_null(mean_y, uniq)
        rep["freeze"] = {"rho_break": knee["rho_break"], "knee_rank": knee["knee_rank"],
                         "ladder": list(QC.RANK_LADDER), "search_rank": QC.SEARCH_RANK,
                         "null_curve": [{"rank": u, "retention": y} for u, y in zip(uniq, mean_y)]}
        print(f"[arm1] FREEZE ρ_break={knee['rho_break']:.4f} knee_rank={knee['knee_rank']:.1f}", flush=True)
        _w()

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print(f"QUORUM_ARM1_{a.phase.upper()}_OK",
          json.dumps({"n_caps": len(caps_out), "n_learners": len(learners),
                      "rho_break": rep.get("freeze", {}).get("rho_break")}, default=str), flush=True)


if __name__ == "__main__":
    main()
