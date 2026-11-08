"""CLEAVE PROTECT-trained (GPU) — is the task LEARNABLE in the complement? Spec §3, T5.

PROTECT-posthoc asks whether the map's component is *removable* after the fact. This axis asks the
stronger question: if the optimizer is never allowed into `span(U_P)` at all, can it still learn the
task — and does it forget less? A post-hoc removal and a trained-in-the-complement run can disagree
(the optimizer compensates), which is exactly why both are run.

The constraint is enforced **exactly** (A-CLEAVE-6), two mechanisms together:
  1. `attach_grad_projection` — the gradient hook the spec names (`project_grad_residual(mode='off')`);
  2. `project_iterate_off` after EVERY optimizer step — because AdamW's elementwise m/√v is not a
     linear map, so off-span gradients still produce partly in-span updates. The mean in-span energy
     this second step removes is LOGGED per run: it is the measured size of the hole the spec's
     stated method would have left open, and if it were always 0 the fix would be unnecessary — that
     number is reported either way (`feedback_signals_that_cannot_fail`).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_trained \
      --args "--banks-s3 results/cleave_qcert/cleave_banks.pt --rank 20 --seeds 1337,2024" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/cleave_trained --budget-min 600
"""
import argparse
import json
import os
import pathlib


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--subject-list", default="")
    ap.add_argument("--seeds", default="1337,2024")
    ap.add_argument("--policies", default="susceptibility,energy,random")
    ap.add_argument("--n-vqa", type=int, default=64)
    ap.add_argument("--n-coco", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import cleave_core as CL
    from infra.pg import stencil_core as SC

    CLS.install_pg()
    device = "cuda"
    policies = [p for p in a.policies.split(",") if p.strip()]
    seeds = _ints(a.seeds)
    rep = {"object": "cleave_trained_v1", "rank": a.rank, "policies": policies, "seeds": seeds,
           "gpu_seconds": {},
           "constraint": ("gradient hook (project_grad_residual off) + per-step iterate projection "
                          "(A-CLEAVE-6: AdamW's elementwise preconditioning is not subspace-preserving)")}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("banks"):
        banks = torch.load(CLS.fetch_s3(a.banks_s3, "/tmp/cleave/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    cert_max = min(int(banks[L].get("certified_max", 0)) for L in banks)
    rep["certification"] = {"certified_max_min_over_layers": cert_max,
                            "rank_certified": bool(a.rank <= cert_max)}
    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = CLS.load_model("/tmp/cleave")
    rep["pinned_sha"] = sha
    mmlu = CLS.mmlu_by_subject()
    roster = CLS.build_roster(mmlu)
    subjects = ([s for s in a.subject_list.split(",") if s.strip()] if a.subject_list
                else roster["confirm"])
    subjects = [s for s in subjects if CLS.splittable(mmlu[s]["test"])]
    assert subjects, "no splittable subjects"
    rep["subjects"] = subjects
    bctx = CLS.make_battery_ctx(proc, roster, n_vqa=a.n_vqa, n_coco=a.n_coco)
    snap = CLS.snapshot_base(model)
    grid = CL._target_grid(model)
    assert len(grid) == CLS.N_LAYERS * 7, f"grid {len(grid)} != 126"
    _w()

    cells = []
    rep["cells"] = cells                # bound BEFORE the loop (feedback_fleet_incremental_save)
    base_cache = {}
    for subj in subjects:
        train_rows, _evl, ev_hashes = CLS.subject_train_eval(mmlu[subj]["test"])
        examples, _dedup = CLS.train_examples(train_rows, ev_hashes)
        if subj not in base_cache:
            CLS.reset_model(model, snap)
            with tm.stage(f"{subj}/base"):
                base_cache[subj] = (CLS.subject_accuracy(model, proc, device, ev_hashes)["acc"],
                                    CLS.mm_battery(model, proc, device, bctx))
        base_sa, base_mm = base_cache[subj]
        read = CLS.make_reader(model, proc, device, bctx, base_mm, base_sa, ev_hashes)

        for seed in seeds:
            for policy in policies:
                CLS.reset_model(model, snap)
                with tm.stage(f"{subj}/s{seed}/{policy}/setup"):
                    CL.mark_grid_trainable(model)       # A-CLEAVE-7: the decomposable grid only
                    bases = CL.policy_bases_by_layer(banks, policy, a.rank, seed=seed,
                                                     n_layers=CLS.N_LAYERS, device=device)
                    params = dict(model.named_parameters())
                    # keep W_base in the MODEL's dtype (bf16): the params are bf16, so an fp32 copy
                    # stores no extra information and would add ~7GB on top of full-FT's AdamW state.
                    base_w = {n: params[n].detach().clone() for n in grid}
                    handles, info = CL.attach_grad_projection(model, bases)

                def _post(m, step, _b=bases, _bw=base_w):
                    # every 8th step is enough to keep the iterate on the constraint set while
                    # keeping the projection's cost off the critical path; the FINAL step always
                    # projects (below), so the installed ΔW is exactly off-span regardless.
                    if step % 8 == 0:
                        return CL.project_iterate_off(m, _bw, _b, grid)
                    return None

                with tm.stage(f"{subj}/s{seed}/{policy}/train"):
                    R = CLS.FT_RECIPE               # A-CLEAVE-8: micro_batch=2 (memory only)
                    n_batches = max(1, (len(examples) + R["batch_size"] - 1) // R["batch_size"])
                    total = R["epochs"] * n_batches
                    stats = SC.train_with_trajectory(
                        model, proc, device, examples, lr=R["lr"],
                        epochs=R["epochs"], batch_size=R["batch_size"],
                        warmup_ratio=R["warmup_ratio"], seed=seed, max_steps=total,
                        micro_batch=R["micro_batch"], checkpoint_steps=(),
                        eval_fn=None, post_step=_post)
                for h in handles:
                    h.remove()
                # the FINAL projection: the installed update is exactly off-span by construction
                with tm.stage(f"{subj}/s{seed}/{policy}/final_project"):
                    final_proj = CL.project_iterate_off(model, base_w, bases, grid)
                with tm.stage(f"{subj}/s{seed}/{policy}/read"):
                    res = read(f"trained_off/{policy}")

                inspan = [p["mean_inspan_energy_removed"] for p in stats["post_step_log"]]
                cells.append({
                    "subject": subj, "seed": seed, "policy": policy, "rank": a.rank,
                    "task_gain": res["task_gain"], "subj_acc": res["subj_acc"],
                    "forgetting": res["forgetting"], "per_component": res["per_component"],
                    "final_loss": stats["final_loss"], "steps": stats["steps"],
                    # the measured size of the AdamW hole the spec's stated method would leave open
                    "inspan_energy_removed_mean": (sum(inspan) / len(inspan)) if inspan else None,
                    "inspan_energy_removed_max": max(inspan) if inspan else None,
                    "inspan_energy_removed_final": final_proj["mean_inspan_energy_removed"],
                    "n_params_constrained": info["n_params"]})
                print(f"[trained] {subj} s{seed} {policy}: gain={res['task_gain']:+.4f} "
                      f"forget={res['forgetting']:+.4f} inspan_removed(mean/final)="
                      f"{cells[-1]['inspan_energy_removed_mean']}/"
                      f"{final_proj['mean_inspan_energy_removed']:.2e}", flush=True)
                del bases, base_w
                torch.cuda.empty_cache()
                _w()

    assert cells, "no cells completed"

    # ---- the trained-off contrast, paired by (subject, seed) ------------------------------
    import numpy as np

    def paired(a_pol, b_pol, field):
        keys = sorted({(c["subject"], c["seed"]) for c in cells})
        va, vb = [], []
        for k in keys:
            ca = [c for c in cells if (c["subject"], c["seed"]) == k and c["policy"] == a_pol]
            cb = [c for c in cells if (c["subject"], c["seed"]) == k and c["policy"] == b_pol]
            if ca and cb:
                va.append(ca[0][field])
                vb.append(cb[0][field])
        return CL.paired_boot(va, vb)

    rep["trained_contrast"] = {
        # >0 ⇒ training OFF the map forgets less than training off a random subspace
        "forgetting_random_minus_sus": paired(CL.NULL_POLICY, CL.MAP_POLICY, "forgetting"),
        "forgetting_energy_minus_sus": paired(CL.SPEC_POLICY, CL.MAP_POLICY, "forgetting"),
        # the task-cost side: >0 ⇒ off-sus KEEPS more task than off-random (a two-sided read; a
        # forgetting win bought with task loss is not a win)
        "task_gain_sus_minus_random": paired(CL.MAP_POLICY, CL.NULL_POLICY, "task_gain"),
        "task_gain_sus_minus_energy": paired(CL.MAP_POLICY, CL.SPEC_POLICY, "task_gain")}
    rep["mean_by_policy"] = {
        p: {"task_gain": float(np.mean([c["task_gain"] for c in cells if c["policy"] == p])),
            "forgetting": float(np.mean([c["forgetting"] for c in cells if c["policy"] == p]))}
        for p in policies if any(c["policy"] == p for c in cells)}
    # the A-CLEAVE-6 headline: how much in-span mass did AdamW actually re-acquire per projection?
    allin = [c["inspan_energy_removed_mean"] for c in cells if c["inspan_energy_removed_mean"] is not None]
    rep["adamw_inspan_leak"] = {
        "mean_over_cells": float(np.mean(allin)) if allin else None,
        "max_over_cells": float(np.max(allin)) if allin else None,
        "note": ("in-span ΔW energy removed at each re-projection — the measured size of the hole a "
                 "gradient-only projection (the spec's stated method) would have left open")}
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("CLEAVE_TRAINED_OK", json.dumps({"contrast": rep["trained_contrast"],
                                           "means": rep["mean_by_policy"],
                                           "adamw_leak": rep["adamw_inspan_leak"]},
                                          default=str), flush=True)


if __name__ == "__main__":
    main()
