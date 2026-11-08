"""FULCRUM Leg B — the curvature-penalty training knob (CONDITIONAL, GPU). Spec §3 Leg B / §9.5.

Gated by a **learnability preflight**: unconstrained in-house full-FT of the 126-matrix grid on one MMLU
subject (the CLEAVE substrate) must reach a real swap-shell subject-gain (≥ `--go-floor`). If GO, train
at curvature-penalty strengths λ (the frozen §3 term ‖P_curv·(W−W_base)‖²_{r=679,CERT8} added to the CE
loss via `stencil_core.train_with_trajectory(aux_loss_fn=…)`), and per λ measure footprint(λ), gain(λ),
n*_energy(λ). H_steerable: footprint↓ and n*_energy↓ monotone in λ at preserved gain. The STENCIL
asymmetry sub-prediction is tested by one NEGATIVE-λ cell (rewarding curvature should cost gain).

If NO-GO: disclosed not-run (Leg A carries the causal claim).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.fulcrum_legb \
      --args "--banks-s3 results/cleave_banks2/out/cleave_banks.pt \
              --quorum-disc-s3 results/quorum_disc/out/metrics.json --subject high_school_biology \
              --lambdas 0,1.0,10.0,-1.0" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/fulcrum_legb --budget-min 900
"""
import argparse
import json
import os
import pathlib

GO_FLOOR = 0.08          # a real learnable capability (= MIN_GAIN_FULL) → the penalty has range


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--quorum-disc-s3", required=True)
    ap.add_argument("--subject", default="high_school_geography")  # CLEAVE: PG1's strongest MMLU learner (+0.186)
    ap.add_argument("--lambdas", default="0,0.001,0.01,0.1,-0.01")  # log sweep; the response reveals λ scale
    ap.add_argument("--penalty-policy", default="susceptibility",
                    help="the penalized subspace: susceptibility (curvature) | random (specificity control) | energy")
    ap.add_argument("--go-floor", type=float, default=GO_FLOOR)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.aws_pg.jobs import _fulcrum_shared as FS
    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import cleave_core as CL
    from infra.pg import fulcrum_core as FC
    from infra.pg import quorum_core as QC
    from infra.pg import stencil_core as SC

    QS.install_pg()
    device = "cuda"
    lambdas = [float(x) for x in a.lambdas.split(",")]
    rep = {"object": "fulcrum_legb_v1", "subject": a.subject, "lambdas": lambdas,
           "curv_rank": FC.CURV_RANK, "cert8": list(FC.CERT8), "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    qd = json.loads(pathlib.Path(QS.fetch_s3(a.quorum_disc_s3, "/tmp/fulcrum/qdisc.json")).read_text())
    rho_break = float(qd["freeze"]["rho_break"])
    rep["rho_break"] = rho_break

    with tm.stage("banks"):
        banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/fulcrum/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    cert = FS.verify_cert_support(banks, rank=FC.CURV_RANK, expect=FC.CERT8)
    rep["cert_support"] = list(cert)

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = CLS.load_model("/tmp/fulcrum")
    rep["pinned_sha"] = sha
    mmlu = CLS.mmlu_by_subject()
    assert a.subject in mmlu, f"subject {a.subject} not in the MMLU roster ({sorted(mmlu)[:5]}...)"
    train_rows, _evl, ev_hashes = CLS.subject_train_eval(mmlu[a.subject]["test"])
    examples, _dedup = CLS.train_examples(train_rows, ev_hashes)
    rep["n_train"], rep["n_eval"] = len(examples), len(ev_hashes)

    lm, paths, vt_orig = QS.lm_bundle(model)
    base_lm_sd = QS.decoder_sd(lm, device=device)
    QS.assert_grid(base_lm_sd)
    install = QS.make_installer(model, lm, paths, vt_orig)
    snap = CLS.snapshot_base(model)
    grid = CL._target_grid(model)

    def reset():
        CLS.reset_model(model, snap)

    reset()
    with tm.stage("base_subject"):
        base_sa = CLS.subject_accuracy(model, proc, device, ev_hashes)["acc"]
    rep["base_subject_acc"] = base_sa

    # curvature penalty bases (r=679; the penalty itself restricts to CERT8)
    with tm.stage("curv_bases"):
        curv_bases = CL.policy_bases_by_layer(banks, a.penalty_policy, FC.CURV_RANK,
                                              n_layers=QS.N_LAYERS, device=device)
    rep["penalty_policy"] = a.penalty_policy
    R = CLS.FT_RECIPE

    def subject_read(tag):
        return {"gain": CLS.subject_accuracy(model, proc, device, ev_hashes)["acc"] - base_sa}

    cells = []
    rep["penalty_sweep"] = cells       # bound BEFORE the loop (feedback_fleet_incremental_save)
    for lam in lambdas:
        reset()
        CL.mark_grid_trainable(model)
        params = dict(model.named_parameters())
        base_w = {n: params[n].detach().clone() for n in grid}
        aux = None
        if lam != 0.0:
            def aux(m, _lam=lam, _bw=base_w):
                return _lam * FC.curvature_penalty(m, _bw, curv_bases, layers=FC.CERT8, grid=grid)
        n_batches = max(1, (len(examples) + R["batch_size"] - 1) // R["batch_size"])
        total = R["epochs"] * n_batches
        with tm.stage(f"lam{lam}/train"):
            stats = SC.train_with_trajectory(
                model, proc, device, examples, lr=R["lr"], epochs=R["epochs"],
                batch_size=R["batch_size"], warmup_ratio=R["warmup_ratio"], seed=a.seed,
                max_steps=total, micro_batch=R["micro_batch"], checkpoint_steps=(),
                eval_fn=None, aux_loss_fn=aux)
        # snapshot the trained decoder ΔW (the 126-grid full-FT → build_decoder('ft') reproduces it)
        ft_lm_sd = QS.decoder_sd(lm, device=device)
        # measure via the swap-shell: gain_full, footprint, n*_energy
        reset()
        with tm.stage(f"lam{lam}/measure"):
            full_res, total_fro2 = QS.install_full_delta(base_lm_sd, ft_lm_sd, banks, install=install,
                                                         read=subject_read, reset=reset, seed=QS.SEED,
                                                         device=device)
            gain_full = full_res["gain"]
            fp = FC.footprint_census(ft_lm_sd, base_lm_sd, banks, rank=FC.CURV_RANK, layers=cert)
            nstar = None
            if abs(gain_full) >= FC.MIN_GAIN_FULL:
                sweep = FS.rank_sweep(base_lm_sd, ft_lm_sd, banks, policy="energy",
                                      ladder=QC.RANK_LADDER, install=install, read=subject_read,
                                      reset=reset, gain_full=gain_full, seed=QS.SEED, device=device,
                                      log=lambda *x, **k: None)
                ns = QC.necessary_rank([s["retention"] for s in sweep], [s["rank"] for s in sweep],
                                       rho_break=rho_break)
                nstar = ns["n_star"]
        cell = {"lambda": lam, "gain": gain_full, "footprint": fp["curvature"],
                "energy_fraction": fp["energy"], "n_star_energy": nstar, "total_fro2": total_fro2,
                "final_loss": stats["final_loss"], "aux_loss_mean": stats.get("aux_loss_mean")}
        cells.append(cell)
        print(f"[legb] λ={lam:+.2f} gain={gain_full:+.4f} footprint={fp['curvature']:.3f} "
              f"n*={nstar} aux={stats.get('aux_loss_mean')}", flush=True)
        del ft_lm_sd, base_w
        torch.cuda.empty_cache()
        _w()

    # learnability gate (λ=0 arm) + the H_steerable read
    base_cell = next((c for c in cells if c["lambda"] == 0.0), None)
    go = bool(base_cell and abs(base_cell["gain"]) >= a.go_floor)
    rep["learnability"] = {"go": go, "lambda0_gain": (base_cell or {}).get("gain"),
                           "go_floor": a.go_floor,
                           "note": ("GO — in-house full-FT reached a measurable capability; the penalty "
                                    "sweep is valid" if go else
                                    "NO-GO — in-house full-FT did not reach the learnability floor; Leg B "
                                    "disclosed not-run, Leg A carries the causal claim (spec §3)")}
    # H_steerable descriptor: footprint↓ and n*↓ as λ rises (positive λ), gain preserved
    pos = sorted([c for c in cells if c["lambda"] >= 0], key=lambda c: c["lambda"])
    if len(pos) >= 2:
        fp0, fpN = pos[0]["footprint"], pos[-1]["footprint"]
        rep["steer"] = {"footprint_fell": bool(fpN < fp0),
                        "n_star_by_lambda": [(c["lambda"], c["n_star_energy"]) for c in pos],
                        "gain_by_lambda": [(c["lambda"], c["gain"]) for c in pos]}
    print(f"[legb] learnability GO={go} (λ0 gain={(base_cell or {}).get('gain')})", flush=True)

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("FULCRUM_LEGB_OK", json.dumps({"go": go, "penalty_sweep": cells}, default=str), flush=True)


if __name__ == "__main__":
    main()
