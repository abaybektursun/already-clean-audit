"""CLEAVE TARGET (GPU) — STENCIL's question, done right. Spec §1.1/§3, the H_targeting branch.

STENCIL asked "does writing INTO U_sus learn the task better than into U_energy?" and answered no —
but its primary was read at iso-task-**gain**, and the susceptibility arm never reached the frozen τ*
(0/9 cells), so the comparison collapsed to an ordering argument. CLEAVE fixes the axis:

  * iso-**forgetting**, not iso-gain. Forgetting IS jointly reachable (every arm forgets *something*
    and forgetting rises with training), whereas gain was not — that asymmetry is exactly what broke
    STENCIL's τ*. Read each arm's task-gain at a common forgetting level along its own trajectory.
  * only certified-resolved ranks carry a map claim (the T1 fix).
  * the fixed-basis arms are scale-matched (unit-norm pinned A) and differ in ONE thing: U_policy.

A landing of "into-sus still loses" REPLICATES STENCIL R2 at full strength; a landing of "into-sus
wins" fires §6.4 H_targeting and must be reconciled against STENCIL before it is asserted. Both are
complete answers.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_target \
      --args "--banks-s3 results/cleave_qcert/cleave_banks.pt --ranks 20,50 --seeds 1337,2024" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/cleave_target --budget-min 600
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
    ap.add_argument("--subject-list", default="")
    ap.add_argument("--seeds", default="1337,2024")
    ap.add_argument("--ranks", default="20,50")
    ap.add_argument("--arms", default="susceptibility,energy,random")
    ap.add_argument("--n-vqa", type=int, default=64)
    ap.add_argument("--n-coco", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import cleave_core as CL
    from infra.pg import stencil_core as SC

    CLS.install_pg()
    device = "cuda"
    arms = [x for x in a.arms.split(",") if x.strip()]
    ranks, seeds = _ints(a.ranks), _ints(a.seeds)
    rep = {"object": "cleave_target_v1", "arms": arms, "ranks": ranks, "seeds": seeds,
           "gpu_seconds": {},
           "fix": "iso-FORGETTING (jointly reachable), not iso-gain (which STENCIL could not reach)"}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("banks"):
        banks = torch.load(CLS.fetch_s3(a.banks_s3, "/tmp/cleave/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    cert_max = min(int(banks[L].get("certified_max", 0)) for L in banks)
    uncertified = [r for r in ranks if r > cert_max]
    rep["certification"] = {"certified_max_min_over_layers": cert_max,
                            "ranks_certified": [r for r in ranks if r <= cert_max],
                            "ranks_uncertified": uncertified}
    print(f"[target] certified_max={cert_max} uncertified_ranks={uncertified}", flush=True)
    _w()

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
    _w()

    cells = []
    rep["cells"] = cells                    # bound BEFORE the loop (feedback_fleet_incremental_save)
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
            for rank in ranks:
                for arm in arms:
                    CLS.reset_model(model, snap)
                    with tm.stage(f"{subj}/s{seed}/r{rank}/{arm}"):
                        SC.attach_arm(model, arm, banks, rank, alpha=rank, seed=seed)
                        n_batches = max(1, (len(examples) + CLS.RECIPE["batch_size"] - 1)
                                        // CLS.RECIPE["batch_size"])
                        total = CLS.RECIPE["epochs"] * n_batches
                        cps = sorted({max(1, int(round(f * total))) for f in (0.25, 0.5, 0.75, 1.0)})
                        traj = []

                        def _eval(m, step, _t=traj):
                            r = read(f"{arm}/r{rank}/step{step}")
                            _t.append({"step": step, "task_gain": r["task_gain"],
                                       "forgetting": r["forgetting"], "subj_acc": r["subj_acc"]})
                            return {}
                        SC.train_with_trajectory(
                            model, proc, device, examples, lr=CLS.RECIPE["lr"],
                            epochs=CLS.RECIPE["epochs"], batch_size=CLS.RECIPE["batch_size"],
                            warmup_ratio=CLS.RECIPE["warmup_ratio"], seed=seed, max_steps=total,
                            micro_batch=CLS.RECIPE.get("micro_batch"), checkpoint_steps=cps,
                            eval_fn=_eval)
                        SC.detach_adapters(model)
                    cells.append({"subject": subj, "seed": seed, "rank": rank, "arm": arm,
                                  "trajectory": traj,
                                  "max_gain": max((t["task_gain"] for t in traj), default=None),
                                  "max_forget": max((t["forgetting"] for t in traj), default=None)})
                    print(f"[target] {subj} s{seed} r{rank} {arm}: max_gain="
                          f"{cells[-1]['max_gain']} max_forget={cells[-1]['max_forget']}", flush=True)
                    _w()

    assert cells, "no cells completed"

    # ---- iso-FORGETTING task-gain: read each arm's gain at a jointly reachable forgetting φ* ----
    # φ* = the min over arms of each arm's MAX achieved forgetting (the common ceiling) — the mirror
    # of STENCIL's common_tau, but on the axis that IS jointly reachable.
    def gain_at_forget(traj, phi):
        """Task-gain at forgetting φ along one arm's own trajectory (worst-case on ties: the MIN
        gain, never optimistic for the arm)."""
        f = np.asarray([t["forgetting"] for t in traj], dtype=np.float64)
        g = np.asarray([t["task_gain"] for t in traj], dtype=np.float64)
        if f.size < 2 or phi < f.min() - 1e-12 or phi > f.max() + 1e-12:
            return None
        o = np.argsort(f)
        fs, gs = f[o], g[o]
        uf, ug = [], []
        i = 0
        while i < fs.size:
            j = i
            while j + 1 < fs.size and abs(fs[j + 1] - fs[i]) < 1e-12:
                j += 1
            uf.append(fs[i]); ug.append(float(np.min(gs[i:j + 1])))
            i = j + 1
        return float(np.interp(phi, np.asarray(uf), np.asarray(ug)))

    rep["iso_forgetting"] = {}
    for rank in ranks:
        maxf = {}
        for arm in arms:
            vals = [c["max_forget"] for c in cells if c["arm"] == arm and c["rank"] == rank
                    and c["max_forget"] is not None]
            if vals:
                maxf[arm] = float(np.min(vals))
        if len(maxf) < 2:
            rep["iso_forgetting"][rank] = {"insufficient": True}
            continue
        phi = float(min(maxf.values()))
        at = {arm: {} for arm in arms}
        for c in cells:
            if c["rank"] != rank:
                continue
            at[c["arm"]][(c["subject"], c["seed"])] = gain_at_forget(c["trajectory"], phi)

        def paired(x, y):
            keys = [k for k in at[x] if at[x][k] is not None and at[y].get(k) is not None]
            return CL.paired_boot([at[x][k] for k in keys], [at[y][k] for k in keys])

        # >0 ⇒ into-sus reaches MORE task-gain than into-energy at equal forgetting ⇒ H_targeting
        sus_vs_energy = paired("susceptibility", "energy") if {"susceptibility", "energy"} <= set(at) else None
        sus_vs_random = paired("susceptibility", "random") if {"susceptibility", "random"} <= set(at) else None
        rep["iso_forgetting"][rank] = {
            "phi_star": phi, "reached": {arm: int(sum(1 for v in at[arm].values() if v is not None))
                                         for arm in at},
            "gain_at_phi": {arm: {f"{s}|{sd}": v for (s, sd), v in at[arm].items()} for arm in at},
            "sus_vs_energy": sus_vs_energy, "sus_vs_random": sus_vs_random,
            "certified": bool(rank <= cert_max)}
        print(f"[target] r={rank} φ*={phi:.4f} sus_vs_energy={sus_vs_energy}", flush=True)
        _w()

    # the single boolean adjudicate_cleave consumes: targeting wins only at a CERTIFIED rank.
    wins = [v.get("sus_vs_energy") for r, v in rep["iso_forgetting"].items()
            if v.get("certified") and v.get("sus_vs_energy")]
    rep["targeting_wins"] = bool(any(CL.clears_floor(w) for w in wins))
    rep["targeting_note"] = ("STENCIL R2 replicated (into-sus does NOT beat into-energy)"
                             if not rep["targeting_wins"] else
                             "into-sus BEATS into-energy — §6.4 H_targeting; reconcile vs STENCIL R2")
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("CLEAVE_TARGET_OK", json.dumps({"targeting_wins": rep["targeting_wins"],
                                          "note": rep["targeting_note"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
