"""QUORUM preflight (GPU) — spec §9.2/§9.3, the gates only a GPU settles, before the fleet spend.

  1. **ppl anchor** — eager PG1 reproduces base ppl (rel < 5%; sdpa's mishandled prefix-LM mask reads
     36.8 vs 22.43, `reference_paligemma_sdpa_wrong_forward`).
  2. **swap-shell wiring** — install a real CARRY donor's FULL ΔW via `build_decoder("ft", …)`; its
     task-gain must reproduce the donor's known CARRY gain (the install path is wired) and the vision
     tower must be untouched (leak gate).
  3. **the POSITIVE CONTROL fires** — inject the rank-1 energy update, install it, measure |gain_full|
     > MIN_GAIN_FULL, then rank-sweep-remove energy: n* MUST be ≈1. If not, the necessary-rank readout
     cannot resolve compact storage → the whole primary VOIDs (spec §5). This is the ceiling that
     makes H_redundant falsifiable, run end-to-end on the real forward.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.quorum_preflight \
      --args "--banks-s3 results/cleave_banks2/out/cleave_banks.pt --donor textvqa --inject-scale 6.0" \
      --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/quorum_preflight --budget-min 120
"""
import argparse
import json
import os
import pathlib

PPL_REL_TOL = 0.05
WIRE_GAIN_REL_TOL = 0.25      # |install_gain − carry_gain| / |carry_gain| (readouts/item-sets differ)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--donor", default="", help="slug; default = the largest-gain scoring donor")
    ap.add_argument("--inject-scale", type=float, default=6.0)
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--n-items", type=int, default=128)
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
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import quorum_core as QC

    QS.install_pg()
    device = "cuda"
    rep = {"object": "quorum_preflight_v1", "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = QS.load_model("/tmp/quorum")
    rep["pinned_sha"] = sha
    with tm.stage("ppl"):
        pp = SPS.safe_ppl_gate(model, proc, device, n_seq=QS.N_PPL_BATTERY)
    rel = abs(pp["ppl"] - base_ppl) / base_ppl
    rep["ppl"] = {"measured": pp["ppl"], "anchor": base_ppl, "rel": rel}
    assert rel < PPL_REL_TOL, f"ppl {pp['ppl']:.3f} != anchor {base_ppl:.3f} — wrong forward"
    _w()

    banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/quorum/cleave_banks.pt"),
                       map_location="cpu", weights_only=False)
    bank, base_sha = QS.load_capability_bank("/tmp/quorum")
    cap = next((c for c in bank if c["slug"] == a.donor), bank[0])
    rep["donor"] = {"slug": cap["slug"], "task": cap["task"], "carry_gain": cap["carry_gain"]}

    lm, paths, vt_orig = QS.lm_bundle(model)
    base_lm_sd = QS.decoder_sd(lm, device=device)
    QS.assert_grid(base_lm_sd)
    install = QS.make_installer(model, lm, paths, vt_orig)
    snap = QS.snapshot_base(model)

    def reset():
        QS.reset_model(model, snap)

    tmap = QS.build_task_map()
    task = tmap[cap["task"]]
    rows = QS.task_rows(task, a.n_items)
    with tm.stage("base_score"):
        base_res = QS.score(model, proc, task, rows, device)
    rep["base_score"] = base_res["score"]

    def read(tag):
        res = QS.score(model, proc, task, rows, device)
        return QS.gain_of(res, base_res)

    # ---- 2. swap-shell wiring on the real donor ------------------------------------------
    with tm.stage("download_ft"):
        ft_lm_sd = QS.ft_decoder_sd(cap["repo"], cap["sha"], base_sha, device=device)
    with tm.stage("install_full"):
        full_res, total_fro2 = QS.install_full_delta(base_lm_sd, ft_lm_sd, banks,
                                                     install=install, read=read, reset=reset,
                                                     seed=QS.SEED, device=device)
    gain_full = full_res["gain"]
    wire_rel = abs(gain_full - cap["carry_gain"]) / max(abs(cap["carry_gain"]), 1e-9)
    rep["wiring"] = {"install_gain": gain_full, "carry_gain": cap["carry_gain"], "rel": wire_rel,
                     "total_fro2": total_fro2, "ok": bool(wire_rel < WIRE_GAIN_REL_TOL and total_fro2 > 0)}
    print(f"[preflight] WIRING {cap['slug']}: install_gain={gain_full:+.4f} carry={cap['carry_gain']:+.4f} "
          f"rel={wire_rel:.3f}", flush=True)
    assert total_fro2 > 0, "the donor ΔW is zero — wiring gate would pass vacuously"
    assert rep["wiring"]["ok"], (
        f"HALT: install gain {gain_full:+.4f} does not reproduce the donor's CARRY gain "
        f"{cap['carry_gain']:+.4f} (rel {wire_rel:.3f}) — the swap-shell is miswired")
    del ft_lm_sd
    torch.cuda.empty_cache()
    _w()

    # ---- 3. the POSITIVE CONTROL — inject rank-1 energy, install, sweep, n* must be ≈1 ----
    # Read the positive control on a SENSITIVE per-item task (textvqa accuracy), not the donor's task
    # (cococap's CIDEr is insensitive to a single-direction perturbation — the scale-6 gain was 0.004).
    pos_task = tmap.get("textvqa", task)
    pos_rows = QS.task_rows(pos_task, a.n_items)
    with tm.stage("pos_base"):
        reset()                                       # CRITICAL: the wiring test left the DONOR
        pos_base_res = QS.score(model, proc, pos_task, pos_rows, device)   # installed — score BASE

    def pos_read(tag):
        return QS.gain_of(QS.score(model, proc, pos_task, pos_rows, device), pos_base_res)

    with tm.stage("positive_control"):
        Ltgt, ttgt = a.inject_layer, "o_proj"
        key = next(k for k in base_lm_sd if f"layers.{Ltgt}." in k and f".{ttgt}." in k)
        U_energy = torch.as_tensor(banks[Ltgt]["sigma_evecs"])
        # A single rank-1 amplification of one variance direction is behaviourally weak until it is
        # large; SEARCH a geometric scale ladder for the first that clears MIN_GAIN_FULL (a bigger
        # effect only makes n*=1 easier — it does not bias the crossing rank). One job, no iteration.
        chosen = None
        for s in (a.inject_scale, a.inject_scale * 4, a.inject_scale * 16, a.inject_scale * 64):
            dW_pos = torch.as_tensor(QC.inject_rank1_delta(U_energy, Ltgt, ttgt, s),
                                     dtype=base_lm_sd[key].dtype, device=device)
            ft_pos_sd = {k: v.clone() for k, v in base_lm_sd.items()}
            ft_pos_sd[key] = (base_lm_sd[key].float() + dW_pos.float()).to(base_lm_sd[key].dtype)
            pos_full, _ = QS.install_full_delta(base_lm_sd, ft_pos_sd, banks, install=install,
                                                read=pos_read, reset=reset, seed=QS.SEED, device=device)
            print(f"[preflight] pos-control scale {s}: gain_full={pos_full['gain']:+.4f}", flush=True)
            if abs(pos_full["gain"]) > QC.MIN_GAIN_FULL:
                chosen = (s, pos_full["gain"], ft_pos_sd)
                break
        assert chosen is not None, (
            f"no scale up to {a.inject_scale * 64} made the rank-1 injection measurable (>MIN_GAIN_FULL) "
            "— a single variance direction is too weak on this readout; widen the search")
        s_used, pos_gain_full, ft_pos_sd = chosen
        rep["positive_control"] = {"inject_layer": Ltgt, "inject_scale_used": s_used,
                                   "gain_full": pos_gain_full, "task": pos_task.name}
        sweep = QS.rank_sweep(base_lm_sd, ft_pos_sd, banks, policy="energy",
                              ladder=QC.RANK_LADDER, install=install, read=pos_read, reset=reset,
                              gain_full=pos_gain_full, seed=QS.SEED, device=device)
        ns = QC.necessary_rank([s_["retention"] for s_ in sweep], [s_["rank"] for s_ in sweep],
                               rho_break=0.5)
    rep["positive_control"]["sweep"] = sweep
    rep["positive_control"]["n_star"] = ns["n_star"]
    rep["positive_control"]["fired"] = bool(ns["n_star"] <= 2.0 and not ns["censored"])
    print(f"[preflight] POSITIVE CONTROL: gain_full={pos_gain_full:+.4f} n*={ns['n_star']:.2f} "
          f"fired={rep['positive_control']['fired']}", flush=True)
    assert rep["positive_control"]["fired"], (
        f"positive control did not fire: n*={ns['n_star']:.2f} (need ≤2). The necessary-rank "
        "instrument cannot resolve compact storage — the primary would VOID (spec §5).")

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("QUORUM_PREFLIGHT_OK", json.dumps({"ppl": rep["ppl"], "wiring": rep["wiring"],
          "positive_control_nstar": ns["n_star"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
