"""STENCIL confirmation (GPU) — spec §9.5, the SEALED primary. Reads the discovery freeze
(`stencil_disc.json`: r*, τ*, weights, range guards) and, on the sealed CONFIRMATION subjects ×
seeds, trains each arm at r* and reads the paired iso-competence forgetting excess

    E_adapt = Forgetting(energy @ τ*) − Forgetting(susceptibility @ τ*)   (paired by (subject, seed))

plus the two mechanism nulls (susceptibility − random, susceptibility − shuffle). Applies
`adjudicate_stencil` ONCE (frozen §6 branch order). No rule reinterpreted post-unblind.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_confirm \
      --args "--banks-s3 results/stencil_banks/out/stencil_banks.pt --disc-s3 results/stencil_disc/out/stencil_disc.json --subjects 4 --seeds 1337,2024,7" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/stencil_conf --budget-min 900
"""
import argparse
import json
import os
import pathlib

CONF_ARMS = ("susceptibility", "energy", "random", "susceptibility_shuffle", "free_lora", "full_ft")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", default="results/stencil_banks/out/stencil_banks.pt")
    ap.add_argument("--disc-s3", default="results/stencil_disc/out/stencil_disc.json")
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--subject-list", default="", help="explicit confirm subjects (overrides roster)")
    ap.add_argument("--seeds", default="1337,2024,7")
    ap.add_argument("--arms", default=",".join(CONF_ARMS))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import ration_core as RC
    from infra.pg import stencil_core as SC

    SS.install_pg()
    device = "cuda"
    seeds = [int(x) for x in a.seeds.split(",")]
    arms = [x.strip() for x in a.arms.split(",")]
    rep = {"object": "stencil_conf_v1", "seeds": seeds, "arms": arms, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    disc = json.loads(SS.fetch(f"{SS.S3_BUCKET}/{a.disc_s3}", "/tmp/stencil/disc.json").read_text())
    rep["discovery_freeze"] = disc
    r_star, tau_star = disc["r_star"], disc["tau_star"]
    weights = disc["weights"]
    guard = disc["range_guard"]
    _w()

    # frozen VOID short-circuit — if discovery drew no range, the primary is VOID (no unblind spend)
    if guard["void_no_range"] or r_star is None or tau_star is None:
        v = SC.adjudicate_stencil(
            E_adapt={"E": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}, sus_vs_random={"E": 0, "ci_lo": 0, "ci_hi": 0},
            sus_vs_shuffle={"E": 0, "ci_lo": 0, "ci_hi": 0}, gain_range_ok=guard["gain_range_ok"],
            forget_range_ok=guard["forget_range_ok"], random_saturates_frontier=False,
            free_strictly_dominates=False, structure_beats_random=False)
        rep["verdict"] = v
        rep["green"] = True
        _w()
        print("STENCIL_CONF_OK", json.dumps({"verdict": v["verdict"], "note": "VOID from discovery"}), flush=True)
        return

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = SS.load_model("/tmp/stencil")
        mmlu = SS.mmlu_by_subject()
        roster = SS.build_roster(mmlu)
        banks = {int(k): v for k, v in torch.load(
            SS.fetch(f"{SS.S3_BUCKET}/{a.banks_s3}", "/tmp/stencil/banks.pt"),
            map_location="cpu", weights_only=False).items()}
        snap = SS.snapshot_base(model)
    cand = ([x.strip() for x in a.subject_list.split(",")] if a.subject_list
            else roster["confirm"])
    # skip subjects whose [:64]-hash collides (degenerate train/eval split, e.g. moral_scenarios)
    conf_subjects = [s for s in cand if SS.splittable(mmlu[s]["test"])][:a.subjects]
    skipped = [s for s in cand if not SS.splittable(mmlu[s]["test"])]
    assert conf_subjects, f"no splittable confirm subjects among {cand} (all skipped: {skipped})"
    rep["skipped_unsplittable"] = skipped
    rep["confirmation_subjects"] = conf_subjects
    _w()

    with tm.stage("base_refs"):
        bctx = SS.make_battery_ctx(proc, roster, n_vqa=64, n_coco=48)   # multimodal composite = vqa+coco
        base_battery = SS.battery_probe(model, proc, device, bctx)
        base_subj = {}
        for s in conf_subjects:
            _, _, evh = SS.subject_train_eval(mmlu[s]["test"])
            base_subj[s] = {"acc": SS.subject_accuracy(model, proc, device, evh)["acc"], "hashes": evh}
    rep["base_subject_acc"] = {s: base_subj[s]["acc"] for s in base_subj}
    _w()

    # ---- train each arm at r* on every (subject, seed); iso-gain forgetting @ τ* -----------
    # forget_at_tau[arm][(subject,seed)] = interp forgetting to τ* along the arm's own trajectory
    forget_at_tau = {arm: {} for arm in arms}
    cells = []
    rep["cells"] = cells                      # incremental — a crash keeps the trained cell results
    for s in conf_subjects:
        tr_rows, _, evh = SS.subject_train_eval(mmlu[s]["test"])
        for seed in seeds:
            for arm in arms:
                r_eff = 0 if arm == "full_ft" else r_star
                SS.reset_model(model, snap)
                with tm.stage(f"{arm}/{s}/s{seed}"):
                    cell = SS.adapt_and_eval(
                        model, proc, device, arm=arm, banks=banks, rank=r_eff, alpha=max(r_eff, 1),
                        subject_rows=tr_rows, eval_hashes=evh, bctx=bctx, base_battery=base_battery,
                        base_subj_acc=base_subj[s]["acc"], weights=weights, seed=seed,
                        checkpoint_fracs=(0.33, 0.66, 1.0))
                traj = cell["trajectory"]
                gains = [t["task_gain"] for t in traj]
                forgets = [t["forgetting"] for t in traj]
                fT = SC.interp_forget_at_gain(gains, forgets, tau_star) if len(gains) >= 2 else None
                forget_at_tau[arm][(s, seed)] = fT
                cells.append({"subject": s, "seed": seed, "arm": arm, "forget_at_tau": fT,
                              "reached_tau": fT is not None, "gains": gains, "forgets": forgets,
                              "trajectory": traj, "degenerate": cell["degenerate"]})
                print(f"[conf] {arm} {s} s{seed}: forget@τ*={fT}", flush=True)
                _w()

    # ---- paired excesses over the (subject, seed) cells reaching τ* ------------------------
    import numpy as np

    def _paired_boot(a, b, seed=1337, n_boot=10000):
        """Paired-cell bootstrap of mean(a−b) with a two-sided 95% CI — small-n-safe (n≥3) local
        version (ration_core.paired_excess asserts n≥10; marginal subjects may not all reach τ*)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        diff = a - b
        rng = np.random.default_rng(seed)
        boots = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"E": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                "se": float(boots.std(ddof=1)), "n": int(diff.size),
                "small_n": bool(diff.size < 10)}

    def paired(a_arm, b_arm):
        keys = [k for k in forget_at_tau[a_arm]
                if forget_at_tau[a_arm][k] is not None and forget_at_tau[b_arm].get(k) is not None]
        if len(keys) < 3:
            return {"E": None, "ci_lo": None, "ci_hi": None, "n": len(keys), "insufficient": True}
        ga = [forget_at_tau[a_arm][k] for k in keys]      # forgetting of arm A
        gb = [forget_at_tau[b_arm][k] for k in keys]      # forgetting of arm B
        # mean(A − B); we want mean(other − sus) → A=other, B=sus (>0 ⇒ sus forgets less ⇒ map wins)
        return _paired_boot(ga, gb)

    _null = {"E": None, "ci_lo": None, "ci_hi": None, "n": 0, "insufficient": True, "not_run": True}
    E_adapt = paired("energy", "susceptibility") if "energy" in arms else _null   # >0 ⇒ map wins
    sus_vs_random = paired("random", "susceptibility") if "random" in arms else _null
    sus_vs_shuffle = paired("susceptibility_shuffle", "susceptibility") if "susceptibility_shuffle" in arms else _null
    rep["E_adapt"], rep["sus_vs_random"], rep["sus_vs_shuffle"] = E_adapt, sus_vs_random, sus_vs_shuffle

    # per-subject E_adapt (cross-task replication, Fig 8)
    per_subject = {}
    for s in conf_subjects:
        ga = [forget_at_tau["energy"][(s, sd)] for sd in seeds
              if forget_at_tau["energy"].get((s, sd)) is not None
              and forget_at_tau["susceptibility"].get((s, sd)) is not None]
        gb = [forget_at_tau["susceptibility"][(s, sd)] for sd in seeds
              if forget_at_tau["energy"].get((s, sd)) is not None
              and forget_at_tau["susceptibility"].get((s, sd)) is not None]
        per_subject[s] = {"E": float(sum(x - y for x, y in zip(ga, gb)) / len(ga)) if ga else None,
                          "n": len(ga)}
    rep["per_subject_E_adapt"] = per_subject

    # structure_beats_random: sus OR energy beats random at τ*
    energy_vs_random = paired("random", "energy") if ("random" in arms and "energy" in arms) else _null
    structure_beats_random = ((sus_vs_random.get("ci_lo") or -1) > 0
                              or (energy_vs_random.get("ci_lo") or -1) > 0)

    # ---- adjudicate ONCE (frozen §6) ------------------------------------------------------
    def _pe(d):
        return d if d.get("ci_lo") is not None else {"E": 0.0, "ci_lo": -1.0, "ci_hi": 0.0}
    verdict = SC.adjudicate_stencil(
        E_adapt=_pe(E_adapt), sus_vs_random=_pe(sus_vs_random), sus_vs_shuffle=_pe(sus_vs_shuffle),
        gain_range_ok=guard["gain_range_ok"], forget_range_ok=guard["forget_range_ok"],
        random_saturates_frontier=False, free_strictly_dominates=False,
        structure_beats_random=structure_beats_random)
    rep["verdict"] = verdict
    (out / "stencil_conf.json").write_text(json.dumps(
        {"verdict": verdict, "E_adapt": E_adapt, "sus_vs_random": sus_vs_random,
         "sus_vs_shuffle": sus_vs_shuffle, "per_subject_E_adapt": per_subject,
         "r_star": r_star, "tau_star": tau_star}, indent=2, default=str))
    _w()

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("STENCIL_CONF_OK", json.dumps({"verdict": verdict["verdict"], "E_adapt": E_adapt}, default=str), flush=True)


if __name__ == "__main__":
    main()
