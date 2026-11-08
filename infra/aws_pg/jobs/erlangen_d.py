"""ERLANGEN Arm D — the minimal restoration set for the PG1→π0 hop (spec §2-D).

Search, not summary: what is the SMALLEST part of the VLA hop's weight change that, put back to
PG1's values inside π0, restores a behavioural oracle? Hierarchical delta-debugging over
component → layer → matrix, with a forced-choice oracle whose chance floor is known exactly.

  oracle    4-way forced-choice VQA scored by teacher-forced NLL over the 4 candidates
            (chance = 0.25 exactly), plus the wikitext NLL anchor as the sanity leg
  restores  acc ≥ chance + 0.5·(acc_PG1 − chance)   — half-way from chance to PG1, frozen
  family    ≥3 ddmin restarts with different orderings; non-uniqueness is DATA (§1.3)
  control   matched-random restoration sets of the SAME size at every accepted level
  audit     monotonicity is delta-debugging's ASSUMPTION and is falsifiable — every observed
            violation (a passing subset inside a failing superset) is recorded

    python -m infra.aws_pg.jobs.erlangen_d --out DIR [--max-evals 600] [--n-oracle 64]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

CHANCE = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-evals", type=int, default=600)
    ap.add_argument("--n-oracle", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-restart", type=int, default=3)
    ap.add_argument("--policy", default="pi0")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.lacuna_shell import load_pg1, pg1_baseline_sd

    assert torch.cuda.is_available(), "Arm D needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "erlangen_d_v1", "policy": a.policy, "chance": CHANCE,
           "max_evals": a.max_evals, "n_oracle": a.n_oracle}

    def bank():
        (out / "erlangen_d.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    packs = P.build_items(n=a.n_oracle)

    # ---- the forced-choice oracle -------------------------------------------------------------
    # Four candidates per item: the true target plus three answers taken from OTHER items, so the
    # distractor distribution is the task's own answer distribution and chance is exactly 1/4.
    rng = np.random.default_rng(20260803)
    vqa = packs["vqa_real"]
    answers = [it["target"] for it in vqa]
    fc = []
    for i, it in enumerate(vqa):
        pool = [answers[j] for j in range(len(answers)) if answers[j].strip() != it["target"].strip()]
        distract = list(rng.choice(pool, size=3, replace=False))
        cands = [it["target"]] + distract
        order = rng.permutation(4)
        fc.append({"image": it["image"], "prompt": it["prompt"],
                   "cands": [cands[k] for k in order],
                   "correct": int(np.where(order == 0)[0][0])})
    rep["n_forced_choice"] = len(fc)

    def oracle_acc():
        """Accuracy of argmin-NLL over the 4 candidates."""
        flat = [{"image": f["image"], "prompt": f["prompt"], "target": c}
                for f in fc for c in f["cands"]]
        nll = P.score_pack(model, proc, device, flat, batch_size=a.batch)
        arr = np.asarray(nll).reshape(len(fc), 4)
        pred = arr.argmin(axis=1)
        return float(np.mean([int(p == f["correct"]) for p, f in zip(pred, fc)]))

    # ---- endpoints: PG1 (all restored) and the policy (nothing restored) ----------------------
    acc_pg1 = oracle_acc()
    policy_sd = P.unit_sd(a.policy, pg1_sd)
    P.install_unit(model, a.policy, pg1_sd, sd=policy_sd)
    fv = P.forward_validity(model, proc, device, packs)
    acc_pol = oracle_acc()
    rep["acc_pg1"] = acc_pg1
    rep["acc_policy"] = acc_pol
    rep["forward_validity_policy"] = fv
    rep["threshold"] = CHANCE + 0.5 * (acc_pg1 - CHANCE)
    print(f"[D] oracle: PG1={acc_pg1:.3f} {a.policy}={acc_pol:.3f} chance={CHANCE} "
          f"threshold={rep['threshold']:.3f}", flush=True)
    bank()
    # RANGE GUARD in both directions: no gap between the endpoints ⇒ nothing to search for, and a
    # policy already ABOVE threshold ⇒ the witness is empty. Either way the arm reports VOID-range
    # instead of running a search whose answer is fixed in advance.
    rep["range_ok"] = bool(acc_pg1 > rep["threshold"] >= acc_pol)
    if not rep["range_ok"]:
        rep["VERDICT"] = "VOID-range"
        rep["note"] = ("the oracle does not separate PG1 from the policy — no restoration search "
                       "is defined")
        rep["DONE"] = True
        bank()
        print(f"[D] VOID-range: acc_pg1={acc_pg1:.3f} acc_policy={acc_pol:.3f}", flush=True)
        return

    groups = P.component_groups(model)
    donors = {"pg1": pg1_sd}
    evals = {"n": 0}

    def mk_restores(table):
        """Oracle factory bound to ONE element table. Explicitly parameterised rather than closing
        over a rebindable name — the level-2 refinement swaps the table, and a closure over a
        mutable outer name is exactly the kind of quiet coupling that produces a wrong witness."""

        def restores(subset):
            parts = [p for g in subset for p in table[g]]
            if not parts:
                return False
            sv = P.apply_intervention(model, parts, "swap_pg1", donors, seed=0)
            acc = oracle_acc()
            P.restore(model, sv)
            evals["n"] += 1
            rep.setdefault("oracle_calls", []).append(
                {"n": len(subset), "acc": acc, "restores": bool(acc >= rep["threshold"]),
                 "subset": sorted(subset)})
            if evals["n"] % 10 == 0:
                print(f"[D] oracle call {evals['n']} |S|={len(subset)} acc={acc:.3f}", flush=True)
                bank()
            return bool(acc >= rep["threshold"])

        return restores

    restores = mk_restores(groups)

    # ---- level 1: which COMPONENT GROUPS ------------------------------------------------------
    rep["level1"] = EC.witness_family(list(groups), restores, n_restart=a.n_restart,
                                      max_evals=int(a.max_evals * 0.55), seed=0)
    rep["level1"]["runs"] = [{k: v for k, v in r.items() if k != "monotonicity_violations"}
                             for r in rep["level1"]["runs"]]
    print(f"[D] level-1 witness sizes={rep['level1']['sizes']} "
          f"core={rep['level1']['core_intersection']}", flush=True)
    bank()

    # ---- matched-random control at the accepted size (N1) --------------------------------------
    k = int(np.median(rep["level1"]["sizes"]))
    rc = []
    rep["random_control"] = rc
    names = list(groups)
    for s in range(6):
        if evals["n"] >= a.max_evals - 4:
            break
        pick = list(np.random.default_rng(500 + s).choice(names, size=k, replace=False))
        parts = [p for g in pick for p in groups[g]]
        sv = P.apply_intervention(model, parts, "swap_pg1", donors, seed=0)
        acc = oracle_acc()
        P.restore(model, sv)
        evals["n"] += 1
        rc.append({"subset": sorted(pick), "acc": acc,
                   "restores": bool(acc >= rep["threshold"])})
        bank()
    rep["random_control_restore_rate"] = (float(np.mean([r["restores"] for r in rc]))
                                          if rc else float("nan"))
    print(f"[D] matched-random sets of size {k}: restore rate "
          f"{rep['random_control_restore_rate']}", flush=True)

    # ---- level 2: refine an ACTUAL level-1 witness to LAYER granularity ------------------------
    # NOT the core intersection: the intersection of several witnesses is generally not itself a
    # witness (the first run of this arm crashed ddmin's precondition on exactly that — the
    # 4-group intersection scored 0.469 against a 0.492 threshold). The refinement target is the
    # SMALLEST run that actually restored, and if even that fails to reproduce on re-test the arm
    # records the fact instead of raising.
    smallest = min(rep["level1"]["runs"], key=lambda r: r["size"])
    core = smallest["witness"]
    rep["level2_target"] = {"witness": sorted(core), "size": len(core),
                            "from_run_size": smallest["size"]}
    fine = {}
    for g in core:
        for key, sl in groups[g]:
            if "layers." in key:
                lab = f"{g}::L{key.split('layers.')[1].split('.')[0]}"
            else:
                lab = f"{g}::all"
            fine.setdefault(lab, []).append((key, sl))
    rep["level2_elements"] = list(fine)
    if len(fine) > 1 and evals["n"] < a.max_evals - 12:
        fine_oracle = mk_restores(fine)
        # explicit precondition check: a level-1 witness that does not reproduce on re-test is a
        # RESULT (the oracle is noisy at this size), recorded — never an exception.
        if fine_oracle(list(fine)):
            rep["level2"] = EC.ddmin(list(fine), fine_oracle,
                                     max_evals=a.max_evals - evals["n"] - 2, seed=0)
            print(f"[D] level-2 witness size={rep['level2']['size']} "
                  f"truncated={rep['level2']['budget_exhausted']}", flush=True)
        else:
            rep["level2"] = {"status": "not-attempted",
                             "reason": "the level-1 witness did not reproduce on re-test at "
                                       "layer granularity — the oracle does not separate at this "
                                       "set size; reported, not raised"}
            print("[D] level-2 SKIPPED — level-1 witness did not reproduce", flush=True)
    # ---- CRITIC FINDING 13: nulls at the ACCEPTED sizes ----------------------------------------
    # The banked random control used size 7 while the headline witness is size 5 and the level-2
    # result is 16-of-30 blocks — neither had a null. Both are added here (~12 oracle calls).
    rep["random_control_accepted_sizes"] = {}
    k5 = len(core)
    rc5 = []
    rep["random_control_accepted_sizes"][f"groups_size_{k5}"] = rc5
    for sdraw in range(6):
        if evals["n"] >= a.max_evals - 2:
            break
        pick = list(np.random.default_rng(900 + sdraw).choice(list(groups), size=k5, replace=False))
        rc5.append({"subset": sorted(pick), "restores": bool(restores(pick))})
        bank()
    if rep.get("level2", {}).get("size"):
        k16 = rep["level2"]["size"]
        rc16 = []
        rep["random_control_accepted_sizes"][f"blocks_size_{k16}"] = rc16
        fine_oracle2 = mk_restores(fine)
        for sdraw in range(6):
            if evals["n"] >= a.max_evals - 2:
                break
            pick = list(np.random.default_rng(950 + sdraw).choice(list(fine), size=min(k16, len(fine)),
                                                                  replace=False))
            rc16.append({"n": len(pick), "restores": bool(fine_oracle2(pick))})
            bank()
    for tag, arr in rep["random_control_accepted_sizes"].items():
        if arr:
            print(f"[D] null @ {tag}: {sum(x['restores'] for x in arr)}/{len(arr)} restore",
                  flush=True)
    rep["n_oracle_evals"] = evals["n"]
    rep["monotonicity_violations"] = sum(
        r.get("n_monotonicity_violations", 0) for r in rep["level1"]["runs"])
    rep["VERDICT"] = "witness-family reported"
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[D] DONE {evals['n']} oracle calls in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
