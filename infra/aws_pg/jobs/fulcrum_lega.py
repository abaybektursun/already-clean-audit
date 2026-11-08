"""FULCRUM Leg A — the causal curvature-projection dose-response (PRIMARY, GPU). Spec §3 Leg A / §9.3-9.4.

Per capability c (a CARRY fine-tune): install the full ΔW → gain_full; then for each projection P walk
θ ∈ {0,.25,.5,.75,1} over ΔW(θ,P) = ΔW − θ·P(ΔW) at r=679 / CERT8, reading gain(θ), n*_energy(θ)
(an energy rank-sweep, all 18 layers, R1-exempt), and the curvature footprint(θ). curv = the certified
Q subspace (susceptibility); random = a matched Haar subspace on the same layers (the null band uses two
seeds); energy = the exploratory confounded arm; susceptibility_shuffle = the ordering control.

  --phase discovery : sweep the discovery capabilities; freeze g* (the jointly-reachable matched-gain
                      level) and the θ-ladder + monotonicity rule.
  --phase confirm   : the sealed capabilities at the frozen g* / ρ_break → E_curv (adjudicated in
                      fulcrum_adjudicate).

`rep["capabilities"]` is bound BEFORE the loop so each completed capability is checkpointed
(`feedback_fleet_incremental_save`). `--donors` shards the (~6GB download each) bank. ρ_break is FROZEN
from QUORUM's discovery (the identical necessary-rank instrument) — never re-derived (baseline discipline).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.fulcrum_lega \
      --args "--phase discovery --banks-s3 results/cleave_banks2/out/cleave_banks.pt \
              --quorum-disc-s3 results/quorum_disc/out/metrics.json" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/fulcrum_disc --budget-min 1200
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
    ap.add_argument("--quorum-disc-s3", required=True, help="QUORUM discovery metrics.json (frozen ρ_break)")
    ap.add_argument("--frozen-s3", default="", help="confirm: the FULCRUM discovery metrics.json with g*")
    ap.add_argument("--donors", default="", help="slug list override (sharding)")
    ap.add_argument("--policies", default="susceptibility,random",
                    help="curv+null primary; add energy,susceptibility_shuffle for the controls")
    ap.add_argument("--seeds", default="1337,2024", help="random-basis draws (curv/energy/shuffle deterministic)")
    ap.add_argument("--n-items", type=int, default=128)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _fulcrum_shared as FS
    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import fulcrum_core as FC

    QS.install_pg()
    device = "cuda"
    policies = [p for p in a.policies.split(",") if p.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    rep = {"object": f"fulcrum_lega_{a.phase}_v1", "phase": a.phase, "policies": policies,
           "seeds": seeds, "curv_rank": FC.CURV_RANK, "cert8": list(FC.CERT8),
           "theta_ladder": list(FC.THETA_LADDER), "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- FROZEN ρ_break from QUORUM (identical instrument; cited, never re-derived) --------
    qd = json.loads(pathlib.Path(QS.fetch_s3(a.quorum_disc_s3, "/tmp/fulcrum/qdisc.json")).read_text())
    rho_break = float(qd["freeze"]["rho_break"])
    rep["rho_break"] = {"value": rho_break, "source": a.quorum_disc_s3, "note": "QUORUM discovery knee"}
    print(f"[lega] FROZEN ρ_break={rho_break:.4f} (QUORUM)", flush=True)

    with tm.stage("banks"):
        banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/fulcrum/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    assert set(banks) == set(range(QS.N_LAYERS)), "banks must cover all 18 layers"
    cert = FS.verify_cert_support(banks, rank=FC.CURV_RANK, expect=FC.CERT8)
    rep["cert_support"] = list(cert)

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = QS.load_model("/tmp/fulcrum")
    rep["pinned_sha"] = sha
    bank, base_sha = QS.load_capability_bank("/tmp/fulcrum")
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

    # ---- frozen g* (confirm) --------------------------------------------------------------
    g_star = None
    if a.phase == "confirm":
        assert a.frozen_s3, "confirm requires --frozen-s3 (the FULCRUM discovery g*)"
        disc = json.loads(pathlib.Path(QS.fetch_s3(a.frozen_s3, "/tmp/fulcrum/disc.json")).read_text())
        g_star = float(disc["freeze"]["g_star"])
        rep["frozen"] = {"g_star": g_star, "source": a.frozen_s3}
        print(f"[lega] FROZEN g*={g_star:.4f}", flush=True)
    _w()

    caps_out = []
    rep["capabilities"] = caps_out          # bound BEFORE the loop (incremental save)
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
        if abs(gain_full) < FC.MIN_GAIN_FULL:
            entry["skipped"] = f"gain_full {gain_full:+.4f} < MIN_GAIN_FULL {FC.MIN_GAIN_FULL}"
            caps_out.append(entry)
            print(f"[lega] SKIP {c['slug']}: gain_full {gain_full:+.4f}", flush=True)
            del ft_lm_sd; torch.cuda.empty_cache(); _w()
            continue

        entry["paths"] = {}
        for policy in policies:
            pol_seeds = seeds if policy == "random" else [QS.SEED]
            for sd_seed in pol_seeds:
                key = policy if policy != "random" else f"random_s{sd_seed}"
                with tm.stage(f"{c['slug']}/{key}"):
                    trows = FS.theta_sweep(base_lm_sd, ft_lm_sd, banks, policy=policy,
                                           cert_layers=cert, install=install, read=read, reset=reset,
                                           gain_full=gain_full, seed=sd_seed, device=device,
                                           rho_break=rho_break)
                entry["paths"][key] = trows
                _w()
        # the θ=0 n*_energy of the FULL ΔW (cross-check vs QUORUM's per-cap n*_energy)
        cur = entry["paths"].get("susceptibility") or next(iter(entry["paths"].values()))
        entry["nstar_energy_theta0"] = next((r["n_star"] for r in cur if r["theta"] == 0.0), None)
        caps_out.append(entry)
        del ft_lm_sd
        QS.drop_ft_cache(c["repo"])
        torch.cuda.empty_cache()
        _w()

    learners = [c for c in caps_out if "skipped" not in c]
    rep["n_learners"] = len(learners)

    # ---- discovery: FREEZE g* from the jointly-reachable floor of curv+random paths -------
    if a.phase == "discovery":
        gbt = {}
        for c in learners:
            pr = c.get("paths", {})
            curv = FS.paths_to_curves({"susceptibility": pr["susceptibility"]})["susceptibility"]["gain"] \
                if "susceptibility" in pr else {}
            # use the FIRST random seed as the paired null path for g* selection
            rk = next((k for k in pr if k.startswith("random")), None)
            rnd = FS.paths_to_curves({rk: pr[rk]})[rk]["gain"] if rk else {}
            if curv and rnd:
                gbt[c["slug"]] = {"susceptibility": curv, "random": rnd}
        assert gbt, "no discovery learner produced both a curv and a random gain path"
        gs = FC.g_star_jointly_reachable(gbt)
        rep["freeze"] = {"g_star": gs["g_star"], "per_cap_floor": gs["per_cap_floor"],
                         "binding": gs["binding"], "theta_ladder": list(FC.THETA_LADDER),
                         "rho_break": rho_break, "curv_rank": FC.CURV_RANK, "cert8": list(FC.CERT8),
                         "monotonicity_rule": "Spearman(θ, n*_curv) < 0, BH-FDR q=0.10 across caps"}
        print(f"[lega] FREEZE g*={gs['g_star']:.4f} (binding {gs['binding']})", flush=True)
        _w()

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print(f"FULCRUM_LEGA_{a.phase.upper()}_OK",
          json.dumps({"n_caps": len(caps_out), "n_learners": len(learners),
                      "g_star": rep.get("freeze", {}).get("g_star")}, default=str), flush=True)


if __name__ == "__main__":
    main()
