"""CLEAVE PROTECT (GPU) — the headline axis. Spec §3 Minimum Decisive Experiment, §6, §9.5–9.6.

Per (subject, seed) cell: train a NATURAL full-FT, then reinstall that same update projected OFF each
policy subspace across the rank ladder and read (task-retention, forgetting). Removing a subspace
sweeps retention monotonically 1→0 (the ladder reaches d, A-CLEAVE-5), so the iso-point that broke
STENCIL cannot recur.

Two phases, ONE instrument (the sweep code is identical — that is the point):
  --phase discovery : sweep the full ladder on DISCOVERY subjects; freeze r* (the knee of the NULL's
                      own forgetting-reduction curve), ρ* (the retention the NULL reaches at r*), and
                      validate ρ* reachable on a HELD-OUT discovery fold — the explicit STENCIL fix
                      (τ* was frozen without a reachability check and then reached in 0/9 cells).
  --phase confirm   : sealed subjects at the frozen (r*, ρ*) → the single paired E_protect
                      (off-random − off-sus) + the off-energy specificity contrast → adjudicate.

`rep["cells"] = cells` is bound BEFORE the loop so every completed cell is checkpointed to S3 — a
crash in the post-loop analysis must never cost the trained cells (`feedback_fleet_incremental_save`).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_protect \
      --args "--phase discovery --banks-s3 results/cleave_qcert/cleave_banks.pt --seeds 1337,2024" \
      --type p5.4xlarge,g6e.8xlarge,g6e.4xlarge --s3-prefix results/cleave_disc --budget-min 900
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
    ap.add_argument("--phase", choices=("discovery", "confirm"), required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--frozen-s3", default="", help="confirm: the discovery metrics.json with r*/ρ*")
    ap.add_argument("--subject-list", default="", help="override the roster split (comma list)")
    ap.add_argument("--seeds", default="1337,2024")
    ap.add_argument("--policies", default="susceptibility,energy,random,susceptibility_shuffle")
    ap.add_argument("--ranks", default="", help="override the ladder (comma list)")
    ap.add_argument("--screen-only", action="store_true",
                    help="run the naturals only (spec §5 range control) and stop — no alpha sweep")
    ap.add_argument("--cpu-surgery", action="store_true",
                    help="run build_decoder on CPU (slower; a fallback if GPU RAM is tight)")
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

    CLS.install_pg()
    device = "cuda"
    policies = [p for p in a.policies.split(",") if p.strip()]
    seeds = _ints(a.seeds)
    rep = {"object": f"cleave_protect_{a.phase}_v1", "phase": a.phase, "policies": policies,
           "seeds": seeds, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- banks + the certified ladder (map arms run ONLY inside the certified range, §5) ----
    with tm.stage("banks"):
        banks = torch.load(CLS.fetch_s3(a.banks_s3, "/tmp/cleave/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    assert set(banks) == set(range(CLS.N_LAYERS)), f"banks cover {sorted(banks)}, need 18 layers"
    # certification is a SET, not a ceiling: stability RISES with rank (the sub-band
    # degeneracy — see cleave_core.resolved_ladder). Usable map ranks = the intersection
    # across all 18 layers, since the intervention projects at every layer at once.
    cert_sets = [set(banks[L].get("certified_ranks", [])) for L in banks]
    cert_ranks = sorted(set.intersection(*cert_sets)) if cert_sets else []
    cert_max = min(int(banks[L].get("certified_max", 0)) for L in banks)
    ladder = _ints(a.ranks) if a.ranks else list(CL.PROTECT_LADDER)
    # The PROTECT ladder must still reach d so retention spans [1,0]; the CERTIFICATION bounds where
    # a *map* claim may be read, not where the null's frontier may be traced. Both are reported.
    rl = CL.resolved_ladder(ladder, {"certified_ranks": cert_ranks, "certified_max": cert_max})
    # A-CLEAVE-11: the per-rank SUPPORT = the layers where that rank is certified. The strict
    # intersection over all 18 layers is the headline fact (it may be empty — L1 certifies nothing);
    # the support is what makes a well-defined intervention still askable, and it is fixed by the
    # certification (a base-model object) not by any outcome, so it stays blind-safe.
    support = {int(r): CL.certified_layers(banks, int(r), n_layers=CLS.N_LAYERS)
               for r in sorted(set(list(ladder) + list(CL.CERT_RANKS)))}
    rep["support_by_rank"] = {r: {"n_layers": len(v), "layers": v} for r, v in support.items()}
    # A-CLEAVE-12: r* = the certified rank with the LARGEST support (blind-safe; the map's most
    # complete well-defined intervention). The retention curve is then swept by removal STRENGTH
    # alpha at that fixed rank — a rank sweep cannot be used because the support is itself
    # rank-dependent, so moving the rank moves the coverage too.
    r_support, n_sup = CL.pick_r_star_by_support(support)
    rep["r_star_by_support"] = {"r_star": r_support, "n_support_layers": n_sup,
                                "support_layers": support[r_support]}
    print(f"[protect] A-CLEAVE-12: r*={r_support} support={n_sup} layers {support[r_support]}", flush=True)
    rep["certification"] = {"certified_ranks_all_layers": cert_ranks,
                            "certified_max_min_over_layers": cert_max,
                            "map_ranks_certified": rl["kept"], "map_ranks_dropped": rl["dropped"],
                            "ladder_used": ladder,
                            "note": ("the full ladder is swept so retention reaches 0 for every "
                                     "policy (A-CLEAVE-5); map-specific claims are read only at "
                                     "certified ranks, and dropped ranks are listed here, never "
                                     "silently omitted")}
    print(f"[protect] certified_max={cert_max} map_ranks={rl['kept']} dropped={rl['dropped']}", flush=True)
    _w()

    # ---- model + roster ------------------------------------------------------------------
    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = CLS.load_model("/tmp/cleave")
    rep["pinned_sha"] = sha
    mmlu = CLS.mmlu_by_subject()
    roster = CLS.build_roster(mmlu)
    if a.subject_list:
        subjects = [s for s in a.subject_list.split(",") if s.strip()]
    else:
        subjects = roster["discovery"] if a.phase == "discovery" else roster["confirm"]
    subjects = [s for s in subjects if CLS.splittable(mmlu[s]["test"])]
    assert subjects, "no splittable subjects for this phase"
    rep["subjects"] = subjects
    rep["roster"] = {"discovery": roster["discovery"], "confirm": roster["confirm"]}

    lm, paths, vt_orig = CLS.lm_bundle(model)
    # A-CLEAVE-9: hold the decoder state-dicts on the GPU so the surgery's projections run there.
    # Budget (44GB L40S): model 6 + base_sd 5 + ft_sd 5 + build_decoder output 5 ≈ 21GB for the
    # sweep; during training the ft_sd does not exist yet and grads+AdamW (~12GB) + base_sd 5 +
    # model 6 ≈ 23GB at micro_batch=2 (A-CLEAVE-8). Both fit.
    surgery_dev = device if not a.cpu_surgery else None
    base_lm_sd = CLS.decoder_sd(lm, device=surgery_dev)
    CLS.assert_grid(base_lm_sd)
    rep["surgery_device"] = str(surgery_dev)
    rep["identity_gate"] = CLS.assert_identity_inert(base_lm_sd, banks, device=surgery_dev)
    bctx = CLS.make_battery_ctx(proc, roster, n_vqa=a.n_vqa, n_coco=a.n_coco)
    snap = CLS.snapshot_base(model)
    install = CLS.make_installer(model, lm, paths, vt_orig)

    def reset():
        CLS.reset_model(model, snap)
    _w()

    # ---- the frozen (r*, ρ*) for confirm --------------------------------------------------
    r_star = rho_star = None
    if a.phase == "confirm":
        assert a.frozen_s3, "confirm requires --frozen-s3 (the discovery freeze)"
        disc = json.loads(pathlib.Path(CLS.fetch_s3(a.frozen_s3, "/tmp/cleave/disc.json")).read_text())
        r_star, rho_star = int(disc["freeze"]["r_star"]), float(disc["freeze"]["rho_star"])
        rep["frozen"] = {"r_star": r_star, "rho_star": rho_star, "source": a.frozen_s3,
                         "discovery_object": disc.get("object")}
        print(f"[protect] FROZEN r*={r_star} ρ*={rho_star:.4f} (from {a.frozen_s3})", flush=True)
        _w()

    # ---- the cell loop -------------------------------------------------------------------
    cells = []
    rep["cells"] = cells                    # bound BEFORE the loop — incremental checkpointing
    rep["naturals"] = []
    base_cache = {}
    for subj in subjects:
        train_rows, _evl, ev_hashes = CLS.subject_train_eval(mmlu[subj]["test"])
        examples, dedup = CLS.train_examples(train_rows, ev_hashes)
        if subj not in base_cache:
            reset()
            with tm.stage(f"{subj}/base"):
                base_sa = CLS.subject_accuracy(model, proc, device, ev_hashes)["acc"]
                base_mm = CLS.mm_battery(model, proc, device, bctx)
            base_cache[subj] = (base_sa, base_mm)
            print(f"[protect] base {subj}: acc={base_sa:.4f} mm={base_mm}", flush=True)
        base_sa, base_mm = base_cache[subj]
        read = CLS.make_reader(model, proc, device, bctx, base_mm, base_sa, ev_hashes)

        for seed in seeds:
            # 1. the NATURAL update
            reset()
            with tm.stage(f"{subj}/s{seed}/train"):
                info, stats = CLS.train_full_ft(model, proc, device, examples, seed=seed)
            with tm.stage(f"{subj}/s{seed}/trained_read"):
                trained = read("trained_in_place")
            ft_lm_sd = CLS.decoder_sd(lm, device=surgery_dev)

            # 2. the retention denominator + the per-cell wiring check (install the FULL ΔW)
            with tm.stage(f"{subj}/s{seed}/natural"):
                nat, nbook = CLS.install_natural(base_lm_sd, ft_lm_sd, banks, install=install,
                                                 read=read, reset=reset, device=surgery_dev)
            gain_full = nat["task_gain"]
            wire_dg = abs(nat["task_gain"] - trained["task_gain"])
            wire_df = abs(nat["forgetting"] - trained["forgetting"])
            natural = {"subject": subj, "seed": seed, "trained": trained, "installed": nat,
                       "wiring_delta_gain": wire_dg, "wiring_delta_forget": wire_df,
                       "gain_full": gain_full, "total_fro2": nbook["total_fro2"],
                       "n_examples": len(examples), "dedup_dropped": dedup.get("n_dropped"),
                       "train_stats": stats, "trainable_params": info["trainable_params"]}
            rep["naturals"].append(natural)
            print(f"[protect] NATURAL {subj} s{seed}: gain={gain_full:+.4f} "
                  f"forget={nat['forgetting']:+.4f} (wiring Δgain={wire_dg:.4f} Δforget={wire_df:.4f})",
                  flush=True)
            _w()

            # A-CLEAVE-13: retention = gain_off/gain_full is a RATIO estimator. Its denominator must
            # clear the measurement noise (SE(gain) ≲ 0.022 on the eval half) or the curve is noise:
            # professional_law's 0.0244 gain is ~1.1σ and yields ~91% retention noise. Admit a cell
            # only above MIN_GAIN_FULL (~3.6σ); exclude and REPORT the rest — this doubles as the
            # learnability screen, and it runs ONLY on discovery subjects (the confirm set stays sealed).
            if gain_full < CL.MIN_GAIN_FULL:
                natural["skipped"] = (f"natural gain {gain_full:+.4f} < MIN_GAIN_FULL "
                                      f"{CL.MIN_GAIN_FULL} — retention is a ratio with a "
                                      f"noise-dominated or NEGATIVE denominator (~{100*0.022/max(abs(gain_full),1e-9):.0f}% "
                                      f"relative noise); cell excluded, not averaged in")
                print(f"[protect] SKIP {subj} s{seed}: natural gain {gain_full:+.4f} < "
                      f"{CL.MIN_GAIN_FULL} (retention not estimable)", flush=True)
                _w()
                continue

            if a.screen_only:
                print(f"[protect] SCREEN-ONLY: {subj} s{seed} admitted (gain {gain_full:+.4f})", flush=True)
                del ft_lm_sd
                torch.cuda.empty_cache()
                _w()
                continue

            # 3. the protective sweep
            with tm.stage(f"{subj}/s{seed}/alpha_sweep"):
                # confirm uses the FROZEN r*; discovery uses the certification-derived one. They
                # coincide unless the banks changed — assert rather than silently diverge.
                r_use = r_star if a.phase == "confirm" else r_support
                if a.phase == "confirm":
                    assert r_use == r_support, (
                        f"frozen r*={r_use} != certification-derived r*={r_support} — the banks "
                        "changed between discovery and confirmation; do not silently proceed")
                got = CLS.alpha_sweep(base_lm_sd, ft_lm_sd, banks, install=install, read=read,
                                      reset=reset, policies=policies, r_star=r_use,
                                      support=support[r_use], gain_full=gain_full,
                                      seed=seed, device=surgery_dev)
            for c in got:
                c.update({"subject": subj, "seed": seed, "gain_full": gain_full,
                          "forgetting_full": nat["forgetting"]})
            cells.extend(got)
            del ft_lm_sd
            torch.cuda.empty_cache()
            _w()

    rep["learnability_screen"] = {
        "min_gain_full": CL.MIN_GAIN_FULL,
        "admitted": sorted({(n["subject"], n["seed"]) for n in rep["naturals"] if "skipped" not in n}),
        "excluded": [{"subject": n["subject"], "seed": n["seed"], "gain_full": n["gain_full"]}
                     for n in rep["naturals"] if "skipped" in n]}
    _w()
    assert cells or a.screen_only, (
        "no discovery cell cleared MIN_GAIN_FULL — on this substrate no DISCOVERY subject learns "
        "enough for retention to be estimable. That is a reportable boundary (the confirm set is "
        "sealed and must not be screened), not something to work around by lowering the bar.")

    # ---- frontiers (per policy, pooled over cells) ----------------------------------------
    rep["frontiers"] = {p: CL.protective_frontier(cells, p, key="alpha") for p in policies
                        if any(c["policy"] == p for c in cells)}
    rep["natural_forgetting_mean"] = float(np.mean([n["installed"]["forgetting"]
                                                    for n in rep["naturals"] if "skipped" not in n]))
    _w()

    # =====================================================================================
    if a.phase == "discovery":
        # r* is already fixed by the CERTIFICATION (A-CLEAVE-12, max-support certified rank).
        # ρ* is the lowest retention EVERY policy reaches on its own α-curve — the α-axis mirror of
        # STENCIL's common_tau, so it is reachable by construction rather than by luck (the STENCIL
        # defect this study exists to fix).
        curves = {p: [{"alpha": q["alpha"], "retention": q["retention"]} for q in fr]
                  for p, fr in rep["frontiers"].items()}
        rj = CL.rho_star_jointly_reachable(curves)
        rho_star = rj["rho_star"]
        reach = CL.rho_reachable(curves, rho_star)
        fold_seed = seeds[-1]
        fold_cells = [c for c in cells if c["seed"] == fold_seed]
        fold_curves = {p: [{"alpha": q["alpha"], "retention": q["retention"]}
                           for q in CL.protective_frontier(fold_cells, p, key="alpha")]
                       for p in policies if any(c["policy"] == p for c in fold_cells)}
        fold_reach = CL.rho_reachable(fold_curves, rho_star) if fold_curves else {"reachable": False}

        F_full = rep["natural_forgetting_mean"]
        null_fr = rep["frontiers"][CL.NULL_POLICY]
        off_full = min(q["forgetting"] for q in null_fr)
        rep["range"] = {"natural_forgetting": F_full, "min_off_forgetting": off_full,
                        "removable": F_full - off_full,
                        "forget_range_ok": bool(F_full > 0.01),
                        "removable_range_ok": bool((F_full - off_full) > 0.005)}
        rep["freeze"] = {"r_star": int(r_support), "rho_star": float(rho_star),
                         "selector_rule": "A-CLEAVE-12 max-support certified rank + "
                                          "jointly-reachable ρ* on the α-curves",
                         "support_layers": support[r_support],
                         "n_support_layers": len(support[r_support]),
                         "rho_star_detail": rj,
                         "reachable_pooled": reach, "reachable_heldout_fold": fold_reach,
                         "fold_seed": fold_seed, "alpha_ladder": list(CL.ALPHA_LADDER),
                         "mm_weights": CLS.MM_WEIGHTS, "policies": policies,
                         "certified_ranks_all_layers": cert_ranks}
        print(f"[protect] FREEZE r*={r_support} (support {len(support[r_support])} layers) "
              f"ρ*={rho_star:.4f} reachable={reach['reachable']} heldout={fold_reach['reachable']} "
              f"range={rep['range']}", flush=True)
        assert reach["reachable"], (
            f"ρ*={rho_star:.4f} is NOT reachable by every policy {reach['per_policy']} — it was "
            "derived as the joint floor, so this can only mean a degenerate curve. Do not freeze.")
        _w()

    # =====================================================================================
    else:
        # forgetting at the frozen ρ*, read along each policy's OWN (retention, forgetting) curve.
        at_rho = {p: {} for p in policies}
        for p in policies:
            for subj in subjects:
                for seed in seeds:
                    pts = [c for c in cells if c["policy"] == p and c["subject"] == subj
                           and c["seed"] == seed]
                    if len(pts) < 2:
                        continue
                    pts = sorted(pts, key=lambda c: c["alpha"])   # the swept axis (A-CLEAVE-12)
                    at_rho[p][(subj, seed)] = CL.interp_forget_at_retention(
                        [c["retention"] for c in pts], [c["forgetting"] for c in pts], rho_star)
        rep["forget_at_rho"] = {p: {f"{s}|{sd}": v for (s, sd), v in at_rho[p].items()} for p in at_rho}
        rep["reached_rho"] = {p: int(sum(1 for v in at_rho[p].values() if v is not None))
                              for p in at_rho}

        def paired(a_pol, b_pol):
            if a_pol not in at_rho or b_pol not in at_rho:
                return {"E": None, "ci_lo": None, "n": 0, "insufficient": True, "not_run": True}
            keys = [k for k in at_rho[a_pol]
                    if at_rho[a_pol][k] is not None and at_rho[b_pol].get(k) is not None]
            return CL.paired_boot([at_rho[a_pol][k] for k in keys],
                                  [at_rho[b_pol][k] for k in keys])

        E_protect = paired(CL.NULL_POLICY, CL.MAP_POLICY)        # >0 ⇒ off-sus forgets less than off-random
        specificity = paired(CL.SPEC_POLICY, CL.MAP_POLICY)      # >0 ⇒ the win is susceptibility-specific
        ordering = paired("susceptibility_shuffle", CL.MAP_POLICY)
        rep["E_protect"], rep["specificity"], rep["ordering_null"] = E_protect, specificity, ordering

        # per-subject replication (cross-task, Fig 8)
        rep["E_protect_by_subject"] = {}
        for subj in subjects:
            ks = [k for k in at_rho[CL.NULL_POLICY] if k[0] == subj
                  and at_rho[CL.NULL_POLICY][k] is not None
                  and at_rho[CL.MAP_POLICY].get(k) is not None]
            if len(ks) >= 2:
                rep["E_protect_by_subject"][subj] = CL.paired_boot(
                    [at_rho[CL.NULL_POLICY][k] for k in ks], [at_rho[CL.MAP_POLICY][k] for k in ks],
                    min_n=2)
        print(f"[protect] E_protect={E_protect} specificity={specificity}", flush=True)
        _w()

    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print(f"CLEAVE_PROTECT_{a.phase.upper()}_OK",
          json.dumps({"n_cells": len(cells),
                      "freeze": rep.get("freeze", {}).get("r_star"),
                      "E_protect": rep.get("E_protect")}, default=str), flush=True)


if __name__ == "__main__":
    main()
