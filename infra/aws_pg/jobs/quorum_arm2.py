"""QUORUM Arm 2 — create-vs-inherent discriminator (GPU). Spec §3 Arm 2 / §9.6 (corroboration).

**A-QUORUM-6.** The spec asks n* for a "deliberately trained rank-r adapter" to separate H_trajectory
(redundancy MADE by a long optimizer run) from H_representation (redundancy inherent to the code). A
trained multimodal adapter needs task-specific gold-answer extraction — fragile for a corroboration
arm that does not gate the primary. Instead, for the SAME capability, compare n*_energy of:
  (a) **full-FT**  — the donor ΔW (the full trajectory's update);
  (b) **rank-r truncation** — the donor ΔW SVD-truncated to rank r per matrix (a rank-r update that
      carries the task content, built from the trajectory but NOT low-rank by training).
If a rank-r update's task content still needs n* ≫ r energy directions to store, that is redundancy of
the REPRESENTATION (the content does not live in the r directions it was compressed to), not of the
trajectory. Training-free, robust, and it tests the load-bearing distinction. The trained-adapter
variant is named as a successor.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_arm2 \
      --args "--banks-s3 results/cleave_banks2/out/cleave_banks.pt --rho-break-s3 results/quorum_disc/out/metrics.json \
              --donor textvqa --trunc-ranks 1,8,64" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/quorum_arm2 --budget-min 400
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
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--rho-break-s3", required=True)
    ap.add_argument("--donor", default="")
    ap.add_argument("--trunc-ranks", default="1,8,64")
    ap.add_argument("--n-items", type=int, default=128)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import quorum_core as QC
    from infra.pg.gauge_common import layer_index, matrix_type

    QS.install_pg()
    device = "cuda"
    trunc_ranks = [int(x) for x in a.trunc_ranks.split(",") if x.strip()]
    rep = {"object": "quorum_arm2_v1", "trunc_ranks": trunc_ranks, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("banks"):
        banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/quorum/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    rho_break = float(json.loads(pathlib.Path(QS.fetch_s3(a.rho_break_s3, "/tmp/quorum/rb.json")).read_text())
                      ["freeze"]["rho_break"])
    rep["rho_break"] = rho_break

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = QS.load_model("/tmp/quorum")
    rep["pinned_sha"] = sha
    bank, base_sha = QS.load_capability_bank("/tmp/quorum")
    cap = next((c for c in bank if c["slug"] == a.donor), bank[0])
    rep["donor"] = {"slug": cap["slug"], "task": cap["task"]}

    lm, paths, vt_orig = QS.lm_bundle(model)
    base_lm_sd = QS.decoder_sd(lm, device=device)
    grid = {k: (layer_index(k), matrix_type(k)) for k, w in base_lm_sd.items()
            if SURG.is_target(k, w)}
    QS.assert_grid(base_lm_sd)
    install = QS.make_installer(model, lm, paths, vt_orig)
    snap = QS.snapshot_base(model)

    def reset():
        QS.reset_model(model, snap)

    tmap = QS.build_task_map()
    task = tmap[cap["task"]]
    rows = QS.task_rows(task, a.n_items)
    reset()
    with tm.stage("base_score"):
        base_res = QS.score(model, proc, task, rows, device)

    def read(tag):
        return QS.gain_of(QS.score(model, proc, task, rows, device), base_res)

    with tm.stage("download_ft"):
        ft_lm_sd = QS.ft_decoder_sd(cap["repo"], cap["sha"], base_sha, device=device)

    def truncate_ft(rank):
        """A decoder sd whose every grid ΔW is SVD-truncated to `rank` (residual-facing side)."""
        out_sd = {k: v.clone() for k, v in base_lm_sd.items()}
        for k, (L, t) in grid.items():
            dW = (ft_lm_sd[k].float() - base_lm_sd[k].float())
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            r = min(rank, S.numel())
            dW_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
            out_sd[k] = (base_lm_sd[k].float() + dW_r).to(base_lm_sd[k].dtype)
        return out_sd

    def nstar_of(ft_sd, cond, rank=None):
        full, _fro = QS.install_full_delta(base_lm_sd, ft_sd, banks, install=install, read=read,
                                           reset=reset, seed=QS.SEED, device=device)
        g = full["gain"]
        if abs(g) < QC.MIN_GAIN_FULL:
            return {"condition": cond, "gain_full": g, "rank": rank, "skipped": "gain < MIN_GAIN_FULL"}
        sweep = QS.rank_sweep(base_lm_sd, ft_sd, banks, policy="energy", ladder=QC.RANK_LADDER,
                              install=install, read=read, reset=reset, gain_full=g, seed=QS.SEED,
                              device=device)
        ns = QC.necessary_rank([s["retention"] for s in sweep], [s["rank"] for s in sweep],
                               rho_break=rho_break)
        return {"condition": cond, "gain_full": g, "rank": rank, "n_star": ns["n_star"],
                "censored": ns["censored"], "sweep": sweep}

    conditions = {}
    with tm.stage("full_FT"):
        conditions["full_FT"] = nstar_of(ft_lm_sd, "full_FT")
    _w()
    for r in trunc_ranks:
        with tm.stage(f"trunc_r{r}"):
            conditions[f"trunc_r{r}"] = nstar_of(truncate_ft(r), f"trunc_r{r}", rank=r)
        print(f"[arm2] trunc r{r}: gain={conditions[f'trunc_r{r}'].get('gain_full')} "
              f"n*={conditions[f'trunc_r{r}'].get('n_star')}", flush=True)
        _w()

    rep["conditions"] = conditions
    del ft_lm_sd
    QS.drop_ft_cache(cap["repo"])

    # the discriminator: does a rank-r truncation still need n* >> r?
    verdicts = {}
    for r in trunc_ranks:
        c = conditions.get(f"trunc_r{r}", {})
        if c.get("n_star") is not None:
            verdicts[f"r{r}"] = {"n_star": c["n_star"], "rank": r,
                                 "nstar_over_rank": c["n_star"] / r,
                                 "verdict": ("H_representation (n* >> r)" if c["n_star"] > 3 * r
                                             else "H_trajectory-consistent (n* ~ r)")}
    rep["trajectory"] = verdicts
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("QUORUM_ARM2_OK", json.dumps({"conditions": {k: v.get("n_star") for k, v in conditions.items()},
          "trajectory": verdicts}, default=str), flush=True)


if __name__ == "__main__":
    main()
