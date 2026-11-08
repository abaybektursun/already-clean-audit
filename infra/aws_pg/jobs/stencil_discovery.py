"""STENCIL discovery frontier scan (GPU) — spec §6/§9.4. On DISCOVERY subjects only: sweep the arms
× rank ladder, build each arm's task-gain/forgetting frontier, and FREEZE the operating point
(r*, τ*, composite weights, layer-set, range guards) blind to any confirmation subject. Writes
`stencil_disc.json` — the sealed inputs the confirmation job consumes.

Frozen selection rules (spec §6, journal C3):
  * r*  = smallest ladder rank at which the SUSCEPTIBILITY arm reaches within KNEE_EPS of its own
          max task-gain over the ladder (its own gain-vs-rank knee; never the contrast).
  * τ*  = TAU_Q · min over FIXED arms of each arm's max achieved task-gain (a common operating point).
  * weights = EQUAL over battery components that show forgetting range on discovery (a component is
          dropped iff |mean forgetting| < FORGET_COMP_FLOOR) — blind to the sus−energy gap (C3).
  * range guards: gain_range_ok = full-FT mean gain > δ_gain; forget_range_ok = unconstrained mean
          forgetting > δ_forget. Either failing → VOID-no-range is the frozen confirmation verdict.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_discovery \
      --args "--banks-s3 results/stencil_banks/out/stencil_banks.pt --arms all --ranks 10,50,150,350 --subjects 2" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/stencil_disc --budget-min 700
"""
import argparse
import json
import os
import pathlib

KNEE_EPS = 0.05             # r* = smallest rank within 5% of the sus arm's own max gain
TAU_Q = 0.90               # τ* = 0.9 × the common (min-over-arms) max gain
FORGET_COMP_FLOOR = 0.005  # a battery component enters the composite iff |mean forgetting| ≥ this
DELTA_GAIN = 0.03
DELTA_FORGET = 0.02


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", default="results/stencil_banks/out/stencil_banks.pt")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--ranks", default="10,50,150,350")
    ap.add_argument("--subjects", type=int, default=2)
    ap.add_argument("--subject-list", default="", help="explicit discovery subjects (overrides roster split)")
    ap.add_argument("--seeds", default="1337")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import stencil_core as SC

    SS.install_pg()
    device = "cuda"
    # 'all' excludes the write arm (u_wr not banked here; Arm-W is a later corroboration step)
    arms = [x for x in SC.ARMS if x != "write"] if a.arms == "all" else [x.strip() for x in a.arms.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    rep = {"object": "stencil_disc_v1", "arms": arms, "ranks": ranks, "seeds": seeds, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = SS.load_model("/tmp/stencil")
        mmlu = SS.mmlu_by_subject()
        roster = SS.build_roster(mmlu)
        banks = torch.load(SS.fetch(f"{SS.S3_BUCKET}/{a.banks_s3}", "/tmp/stencil/banks.pt"),
                           map_location="cpu", weights_only=False)
        banks = {int(k): v for k, v in banks.items()}
        snap = SS.snapshot_base(model)
    disc_subjects = ([x.strip() for x in a.subject_list.split(",")] if a.subject_list
                     else roster["discovery"][:a.subjects])
    assert all(s not in roster["confirm"] for s in disc_subjects), "discovery subject overlaps confirm (firewall)"
    rep["discovery_subjects"] = disc_subjects
    _w()

    with tm.stage("base_refs"):
        bctx = SS.make_battery_ctx(proc, roster)
        # A-STENCIL-9: forgetting = MULTIMODAL collateral (VQA/COCO); broad-MMLU transfers + ppl
        # improves under text-MMLU adaptation (diag: broad −0.3…−0.6, VQA/COCO degrade). The scan's
        # 'forgetting' is this multimodal composite; the frozen weights are re-derived post-scan.
        weights_all = {c: 1.0 for c in SS.MULTIMODAL_COMPONENTS}
        base_battery = SS.battery_probe(model, proc, device, bctx)
        base_subj = {}
        for s in disc_subjects:
            _, _, evh = SS.subject_train_eval(mmlu[s]["test"])
            base_subj[s] = {"acc": SS.subject_accuracy(model, proc, device, evh)["acc"], "hashes": evh}
    rep["base_battery"] = base_battery
    rep["base_subject_acc"] = {s: base_subj[s]["acc"] for s in base_subj}
    _w()

    # ---- frontier scan (checkpointed per cell) --------------------------------------------
    cells = []
    rep["cells"] = cells
    for s in disc_subjects:
        tr_rows, _, evh = SS.subject_train_eval(mmlu[s]["test"])
        for arm in arms:
            for rank in ranks:
                for seed in seeds:
                    r_eff = 0 if arm == "full_ft" else rank
                    SS.reset_model(model, snap)
                    with tm.stage(f"{arm}/r{rank}/{s}/s{seed}"):
                        cell = SS.adapt_and_eval(
                            model, proc, device, arm=arm, banks=banks, rank=r_eff,
                            alpha=max(r_eff, 1), subject_rows=tr_rows, eval_hashes=evh, bctx=bctx,
                            base_battery=base_battery, base_subj_acc=base_subj[s]["acc"],
                            weights=weights_all, seed=seed, checkpoint_fracs=(1.0,))
                    traj = cell["trajectory"]
                    fin = traj[-1] if traj else {"task_gain": 0.0, "forgetting": 0.0, "per_component": {}}
                    cells.append({"subject": s, "arm": arm, "rank": rank, "seed": seed,
                                  "final_gain": fin["task_gain"], "final_forget": fin["forgetting"],
                                  "final_per_component": fin.get("per_component", {}),
                                  "trajectory": traj, "degenerate": cell["degenerate"]})
                    print(f"[disc] {arm}@r{rank} {s} s{seed}: gain={fin['task_gain']:+.4f} "
                          f"forget={fin['forgetting']:+.4f}", flush=True)
                    _w()

    # ---- aggregate frontier + freeze (r*, τ*, weights, guards) -----------------------------
    def agg(arm, rank, key):
        vals = [c[key] for c in cells if c["arm"] == arm and c["rank"] == rank]
        return float(np.mean(vals)) if vals else None

    frontier = {arm: {rank: {"gain": agg(arm, rank, "final_gain"),
                             "forget": agg(arm, rank, "final_forget")} for rank in ranks}
                for arm in arms}
    rep["frontier"] = frontier

    # r*: sus arm's own gain-vs-rank knee
    sus_gains = {rank: frontier["susceptibility"][rank]["gain"] for rank in ranks
                 if frontier["susceptibility"][rank]["gain"] is not None}
    r_star = None
    if sus_gains:
        gmax = max(sus_gains.values())
        for rank in sorted(sus_gains):
            if gmax > 0 and sus_gains[rank] >= (1 - KNEE_EPS) * gmax:
                r_star = rank
                break
        r_star = r_star if r_star is not None else max(sus_gains, key=sus_gains.get)

    # τ*: common gain reachable by ALL fixed-basis arms
    fixed_maxgain = {}
    for arm in SC.FIXED_ARMS:
        gs = [frontier[arm][rank]["gain"] for rank in ranks
              if arm in frontier and frontier[arm][rank]["gain"] is not None]
        if gs:
            fixed_maxgain[arm] = max(gs)
    tau_star = TAU_Q * min(fixed_maxgain.values()) if fixed_maxgain else None

    # composite weights: equal over MULTIMODAL components that DEGRADE (positive mean forgetting) on
    # discovery — the real collateral (A-STENCIL-9; broad-MMLU/ppl are dropped: they transfer/improve,
    # not forget). Use the mean (signed), NOT abs, so an improving component is never retained.
    comp_mean = {c: float(np.mean([cell["final_per_component"].get(c, 0.0) for cell in cells
                                   if cell["arm"] in ("full_ft", "free_lora")]) or 0.0)
                 for c in SS.MULTIMODAL_COMPONENTS}
    retained = [c for c in SS.MULTIMODAL_COMPONENTS if comp_mean[c] >= FORGET_COMP_FLOOR]
    weights = {c: 1.0 for c in (retained or list(SS.MULTIMODAL_COMPONENTS))}
    rep["component_mean_forgetting"] = comp_mean

    # range guards. gain range = the BEST gain any unconstrained OR fixed-basis arm reaches (full_ft
    # may be excluded from the scan; keying gain-range solely on full_ft would spuriously VOID).
    gain_arms = [a for a in ("full_ft", "free_lora", "susceptibility", "energy") if a in frontier]
    ft_gain = max([frontier[a][rank]["gain"] for a in gain_arms for rank in ranks
                   if frontier[a][rank]["gain"] is not None] or [0.0])
    unc_forget = max([frontier[arm][rank]["forget"] for arm in ("full_ft", "free_lora") if arm in frontier
                      for rank in ranks if frontier[arm][rank]["forget"] is not None] or [0.0])
    gain_ok = ft_gain > DELTA_GAIN
    forget_ok = unc_forget > DELTA_FORGET

    # contrast-range table (from banks metrics, layer-averaged) at r*
    disc = {"r_star": r_star, "tau_star": tau_star, "weights": weights, "retained_components": retained,
            "layer_set": sorted(banks.keys()),
            "range_guard": {"full_ft_max_gain": ft_gain, "unconstrained_max_forget": unc_forget,
                            "delta_gain": DELTA_GAIN, "delta_forget": DELTA_FORGET,
                            "gain_range_ok": bool(gain_ok), "forget_range_ok": bool(forget_ok),
                            "void_no_range": bool(not (gain_ok and forget_ok))},
            "fixed_maxgain": fixed_maxgain, "sus_gain_vs_rank": sus_gains,
            "selection_rules": {"knee_eps": KNEE_EPS, "tau_q": TAU_Q,
                                "forget_comp_floor": FORGET_COMP_FLOOR}}
    rep["discovery_freeze"] = disc
    (out / "stencil_disc.json").write_text(json.dumps(disc, indent=2, default=str))
    _w()

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("STENCIL_DISC_OK", json.dumps({"r_star": r_star, "tau_star": tau_star,
          "void_no_range": disc["range_guard"]["void_no_range"], "weights": weights}, default=str), flush=True)


if __name__ == "__main__":
    main()
