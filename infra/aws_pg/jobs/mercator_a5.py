"""M-A5 — the regime probe: does the LOSS SHAPE change where the weights move? (queue A5). Wave 2.

Everything this program has measured about the sink was measured under supervised next-token
training. That is one regime, and the practical claim it feeds — that fine-tuning damages a base in
a predictable place — is a claim about training in general.

    REFUTER (pre-logged): non-supervised regimes show the same sink motion as supervised at
    matched realized motion.

THREE REGIMES ON ONE DECODER FAMILY. All three move gemma-1-family decoder cells, so the SAME
banks apply and a difference between them is a difference of loss rather than of coordinate space:

    supervised     next-token cross-entropy on the narrow-grammar code task — the house reference
    flow_matching  pi0's own flow-matching imitation loss on one LIBERO task, through
                   `ctrliface_lerobot.loss_fn_factory`, i.e. the real objective on real data
                   rather than a regression standing in for one
    reinforce      a scalar-reward policy gradient on a text task, with a moving-average baseline

CALLED REINFORCE, NOT PPO. The spec says "small PPO text run"; what is implemented is REINFORCE
with a baseline, which is the same loss SHAPE (a scalar reward multiplying a log-probability)
without the clipped-ratio machinery. Naming it accurately matters more than matching the spec's
word, and the difference is stated on every row.

MATCHED REALIZED MOTION IS THE WHOLE DESIGN. Regimes differ enormously in how far they move the
weights per step, so comparing "sink motion" at a matched STEP COUNT would compare three different
distances. Each regime instead trains until its total ||dW||_F crosses a shared target, and the
readout is the SHARE of that motion inside each bank. A regime that cannot reach the target inside
its step cap is recorded as unmatched rather than compared.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-A5"
REGIMES = ("supervised", "flow_matching", "reinforce")
BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
         "haar32_s0", "haar32_s1", "haar32_s2")


def motion_shares(dW_by_key, banks_by_key):
    """Share of realized motion inside each bank, plus the total, over the trained cells."""
    tot = 0.0
    inb = {b: 0.0 for b in BANKS}
    for k, dW in dW_by_key.items():
        tot += float((dW ** 2).sum())
        for b in BANKS:
            inb[b] += float(((dW @ banks_by_key[k][b]) ** 2).sum())
    assert tot > 0, "no cell moved"
    return {"total_motion": float(np.sqrt(tot)),
            "shares": {b: inb[b] / tot for b in BANKS}}


def main():
    ap = argparse.ArgumentParser(description="M-A5 — the regime probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="12,13,14,15,16,17")
    ap.add_argument("--target-motion", type=float, default=0.0,
                    help="0 = take the supervised reference's realized motion as the target")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--match-steps", type=int, default=0,
                    help="step budget for the regimes CHASING the reference's motion; 0 means "
                         "4 x --max-steps. They need more than the reference, not the same: the "
                         "reference sets the target BY running max_steps, so an equal cap makes a "
                         "slower-moving regime structurally unable to match (M-D78)")
    ap.add_argument("--ref-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--libero-batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    # NO torch.compile ON THIS ARM. The pi0.5 policy compiles part of its forward, and inductor
    # shells out to gcc to build cuda_utils.c — which needs Python.h, and the AMI carries no
    # python3.12-dev, so the run died with 'fatal error: Python.h: No such file or directory'
    # (M-D74). Installing headers is the wrong fix twice over: this arm MEASURES REALIZED MOTION
    # across training regimes, and a compiled kernel is free to reassociate the arithmetic that
    # produces it. Eager is the configuration whose numerics the rest of the program uses.
    import os as _os
    _os.environ["TORCHDYNAMO_DISABLE"] = "1"
    _os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    H.hf_setup()
    # BEFORE ANY MODEL IS LOADED. The flow-matching regime needs lerobot, which the AMI lacks, and
    # `lerobot[libero,pi]` pins its own transformers — installing it once the PG1 regimes have
    # already built a model would swap the library underneath one. The first run discovered the
    # missing module only after the supervised reference had trained 600 steps, because that is
    # where the import sat.
    if "flow_matching" in [r.strip() for r in a.regimes.split(",")]:
        M.ensure_lerobot()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w2") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_a5", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers, _load_full_pg1
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, text_task_rows

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    keys = [(t, L) for t in types for L in layers]
    banks = {k: {b: WT.load_bank(bank_dir, k[0], k[1], b) for b in BANKS} for k in keys}

    def run_pg1(regime, target):
        """The two regimes that run on PG1's own decoder: supervised and reinforce."""
        model = _load_full_pg1(base_sha, "cuda")
        for p in model.parameters():
            p.requires_grad_(False)
        mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
        for m in mods.values():
            m.weight.requires_grad_(True)
        pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        rows, _ = text_task_rows(a.n_train, 16, a.seed)
        batches = build_batches(processor, rows, "cuda")
        # PaliGemma's mask builder REFUSES a training forward without token_type_ids: 0 marks the
        # bidirectional prefix, 1 the causally-masked suffix. That split is exactly the labels
        # mask this batch already carries, so it is DERIVED rather than guessed — a constant 0 or 1
        # would silently change which tokens attend to which and train a different model (M-D55).
        def ttids(b):
            return (b["labels"] != -100).long()

        rng = np.random.default_rng(H.cell_seed(a.seed, ARM, regime, "train"))
        baseline, steps, reached = None, 0, False
        model.train()
        budget = a.max_steps if target is None else match_steps
        for step in range(budget):
            b = batches[int(rng.integers(0, len(batches)))]
            opt.zero_grad()
            if regime == "supervised":
                model(input_ids=b["input_ids"], labels=b["labels"],
                      token_type_ids=ttids(b)).loss.backward()
            else:
                # REINFORCE with a moving-average baseline. The reward is the model's own
                # sequence log-probability under a SHIFTED target — a scalar signal that depends
                # on the whole completion rather than on per-token supervision, which is the
                # property that distinguishes this regime's loss shape from the supervised one.
                o = model(input_ids=b["input_ids"], labels=b["labels"],
                          token_type_ids=ttids(b))
                logp = -o.loss                                # mean log-prob per supervised token
                r = float(logp.detach())
                baseline = r if baseline is None else 0.9 * baseline + 0.1 * r
                ((r - baseline) * -logp).backward()
            opt.step()
            steps = step + 1
            if step % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    tot = float(np.sqrt(sum(
                        float(((m.weight.detach().to(torch.float32) - pristine[k]) ** 2).sum())
                        for k, m in mods.items())))
                if target is not None and tot >= target:
                    reached = True
                    break
        dW = {k: (m.weight.detach().to(torch.float32) - pristine[k]).cpu().numpy()
              .astype(np.float64) for k, m in mods.items()}
        res = motion_shares(dW, banks)
        del model, mods, pristine, opt
        gc.collect()
        torch.cuda.empty_cache()
        return {"regime": regime, "steps": steps, "reached_target": reached, **res}

    def run_flow_matching(target):
        """pi0's own flow-matching imitation loss on one LIBERO task, through the LeRobot loader."""
        from infra.pg import ctrliface_lerobot as CL
        policy = CL.load_policy(device="cuda")
        # `get_backbone` returns (module, layers, hidden_dim) and asserts the 18-layer Gemma-1
        # spine is the LANGUAGE MODEL rather than the action expert — the compartment check this
        # arm needs, since it trains backbone cells and reads them with PG1-keyed banks.
        _bb, bb_layers, d_model = CL.get_backbone(policy)
        for p in policy.parameters():
            p.requires_grad_(False)
        mods = T.trainable_modules(bb_layers, types=types, layers=layers)
        for m in mods.values():
            m.weight.requires_grad_(True)
        pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        meta_ds = CL.dataset_meta()
        dt = CL.build_delta_timestamps(policy, meta_ds)
        # The DISCOVERY half of the frozen LIBERO split; the confirmation suites are not touched
        # by an exploratory arm.
        # NO SUITE FILTER (M-D63). CL.DISCOVERY_SUITES matches the substrings 'spatial'/'object'
        # against each episode's TASK STRING, and LIBERO task strings are natural-language
        # instructions ("pick up the black bowl…") that never contain the suite name — the suite
        # lives in the environment, not in the episode. So the filter matched nothing and the job
        # failed after loading the policy. M-A5 does not need the split anyway: the
        # discovery/confirmation firewall is CTRLIFACE's, and this arm compares realized motion
        # across training regimes on whatever episodes it is given.
        # build_loader returns (loader, n_episodes, filtered) — binding the whole TUPLE to `loader`
        # makes iter() yield the DataLoader itself as its first element, and the training step then
        # calls .items() on it (M-D68).
        loader, n_episodes, filtered = CL.build_loader(CL.DATASET_REPO, dt, suites=None,
                                                       batch_size=a.libero_batch, shuffle=True)
        print(f"[M-A5] LIBERO loader: {n_episodes} episodes, suite-filtered={filtered}", flush=True)
        # The policy consumes a PRE-PROCESSED batch: the raw LeRobot fields carry the task string,
        # not `observation.language.tokens`, and the pre-processor is what produces it (M-D70).
        pre, _post = CL.build_preprocessor(meta=meta_ds)
        loss_fn = CL.loss_fn_factory(policy)
        steps, reached = 0, False
        policy.train()
        it = iter(loader)
        budget = a.max_steps if target is None else match_steps
        for step in range(budget):
            try_batch = next(it, None)
            if try_batch is None:
                it = iter(loader)
                try_batch = next(it)
            # PRE-PROCESS, THEN MOVE. The pre-processor tokenizes the task string and builds new
            # CPU tensors, so running it after to_device would mix CPU and CUDA tensors in one
            # batch. Gate-0c's recipe is pre(raw) first, and this follows it.
            batch = CL.to_device(pre(try_batch), "cuda")
            opt.zero_grad()
            loss_fn(batch).backward()
            opt.step()
            steps = step + 1
            if step % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    tot = float(np.sqrt(sum(
                        float(((m.weight.detach().to(torch.float32) - pristine[k]) ** 2).sum())
                        for k, m in mods.items())))
                if target is not None and tot >= target:
                    reached = True
                    break
        dW = {k: (m.weight.detach().to(torch.float32) - pristine[k]).cpu().numpy()
              .astype(np.float64) for k, m in mods.items()}
        res = motion_shares(dW, banks)
        assert d_model == banks[keys[0]]["core32"].shape[0], (
            f"pi0 backbone width {d_model} does not match the bank ambient dimension "
            f"{banks[keys[0]]['core32'].shape[0]}; the PG1-keyed banks do not apply here")
        del policy, mods, pristine, opt
        gc.collect()
        torch.cuda.empty_cache()
        return {"regime": "flow_matching", "steps": steps, "reached_target": reached,
                "policy_repo": CL.POLICY_REPO, "dataset_repo": CL.DATASET_REPO,
                "suites": None, "suite_filter": "none (see M-D63)",
                "n_episodes": n_episodes, **res}

    # ---- the reference sets the target -------------------------------------------------------
    assert "supervised" in regimes, "the supervised reference defines the matched-motion target"
    # THE TARGET IS SET BY THE REFERENCE'S OWN BUDGET, so giving the chasers the same budget is a
    # design that cannot work for any regime slower than supervised. Two runs showed it: at 600
    # steps flow-matching reached 1.96 of 3.03, at 2000 it reached 3.64 of 5.47 — the same 65-67%
    # both times, because raising --max-steps moved the target too. The chasers now get their own,
    # larger budget; reinforce reaches the target in about 60% of the reference's steps and is
    # unaffected.
    match_steps = a.match_steps if a.match_steps > 0 else 4 * a.max_steps
    ref = run_pg1("supervised", None if a.target_motion <= 0 else a.target_motion)
    target = a.target_motion if a.target_motion > 0 else ref["total_motion"]
    print(f"[M-A5] supervised reference: {ref['steps']} steps, motion {ref['total_motion']:.5g}; "
          f"target = {target:.5g}", flush=True)

    cells = {"supervised": {**ref, "reached_target": True, "is_reference": True}}
    for b in BANKS:
        at.add("regime_motion_share", value=ref["shares"][b], bank_id=b, model_id=WT.PG1_REPO,
               model_sha=base_sha,
               flags={"regime": "supervised", "steps": ref["steps"],
                      "total_motion": ref["total_motion"], "target_motion": target,
                      "is_reference": True, "optimizer": optinfo})

    for regime in regimes:
        if regime == "supervised":
            continue
        c = run_flow_matching(target) if regime == "flow_matching" else run_pg1(regime, target)
        cells[regime] = c
        for b in BANKS:
            at.add("regime_motion_share", value=c["shares"][b], bank_id=b,
                   flags={"regime": regime, "steps": c["steps"],
                          "total_motion": c["total_motion"], "target_motion": target,
                          "reached_target": c["reached_target"],
                          "loss_shape": ("scalar reward x log-prob (REINFORCE with a "
                                         "moving-average baseline, NOT clipped-ratio PPO)"
                                         if regime == "reinforce" else
                                         "pi0 flow-matching imitation on one LIBERO task"),
                          "optimizer": optinfo})
        print(f"[M-A5] {regime:14s} {c['steps']:4d} steps motion {c['total_motion']:.5g} "
              f"(target {target:.5g}, reached {c['reached_target']}) "
              f"core={c['shares']['core32']:.5f} anti={c['shares']['anti32']:.5f}", flush=True)
        M.s3_sync_incremental(out)

    at.check_controls_per_statistic(["regime_motion_share"])
    matched = [r for r in regimes if r == "supervised" or cells[r]["reached_target"]]
    gates.record("a5_regimes_reached_matched_motion",
                 bool(set(matched) == set(regimes)), matched=matched,
                 total_motion={r: cells[r]["total_motion"] for r in regimes},
                 target=target, max_steps=a.max_steps, match_steps=match_steps,
                 outcome=None if set(matched) == set(regimes) else M.VOID_INSTRUMENT,
                 note=("a regime that never reached the target moved a different distance, and a "
                       "share comparison across different distances is not matched"))
    core = {r: cells[r]["shares"]["core32"] for r in regimes}
    twins = {r: [cells[r]["shares"][b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
             for r in regimes}
    differs = [r for r in regimes if r != "supervised"
               and (core[r] < min(twins["supervised"]) or core[r] > max(twins["supervised"]))]
    # THE COMPARISON INHERITS THE MATCHING GATE'S VERDICT. `differs` is a share comparison across
    # regimes, and the gate above just recorded whether those regimes moved the SAME DISTANCE. When
    # it did not, this one is comparing shares at different motions and cannot mean what it says —
    # the first run reported PASS here while the matching gate beside it read VOID-INSTRUMENT
    # (M-D75). The dependency is stamped rather than inferred by whoever reads the two rows.
    gates.record("a5_a_nonsupervised_regime_differs_from_supervised", bool(differs),
                 motion_matched=(set(matched) == set(regimes)),
                 unmatched_regimes=sorted(set(regimes) - set(matched)),
                 outcome=None if set(matched) == set(regimes) else M.VOID_INSTRUMENT,
                 differing=differs, core_share_by_regime=core,
                 supervised_twin_band=[min(twins["supervised"]), max(twins["supervised"])],
                 note=("the refuter fires when this is False: the same sink motion in every "
                       "regime at matched realized motion would make the sink a property of the "
                       "optimizer alone and not of what is being learned"))
    for r in regimes:
        if r == "supervised":
            continue
        # THE PRE-LOGGED GATE ABOVE IS KEPT EXACTLY AS WRITTEN, and this is recorded beside it rather
    # than replacing it — rereading a frozen rule after seeing which way it fell is the move this
    # program does not make. But it cannot be banked on its own, because it tests whether a
    # regime's core share falls outside the SUPERVISED TWIN BAND, and supervised's own core share
    # (0.00707) is already outside that band (0.01521–0.01601). The band sits where the twins are,
    # not where supervised is, so any regime landing anywhere near supervised "differs" from it.
    # It cannot fail for the case it was written to detect (M-D81).
    #
    # The comparison the refuter actually describes — "the same sink motion in every regime" — is
    # between each regime's core share and SUPERVISED's, judged against the twin spread as a
    # WIDTH rather than as a location.
    spread = max(twins["supervised"]) - min(twins["supervised"])
    by_spread = {r: abs(core[r] - core["supervised"]) for r in regimes if r != "supervised"}
    differs_by_spread = sorted([r for r, d in by_spread.items() if d > spread])
    gates.record("a5_regime_core_share_differs_by_more_than_the_twin_spread",
                 bool(differs_by_spread), differing=differs_by_spread,
                 core_share_by_regime=core, supervised_core=core["supervised"],
                 twin_spread=spread, distance_from_supervised=by_spread,
                 motion_matched=(set(matched) == set(regimes)),
                 outcome=None if set(matched) == set(regimes) else M.VOID_INSTRUMENT,
                 note=("added after the pre-logged gate was found unable to fail: it compared "
                       "each regime to the twins' LOCATION when the question is distance from "
                       "SUPERVISED, and every regime including supervised sits below the twin "
                       "band. Reported alongside the original, which is unchanged"))

    sur.add(arm=ARM, expected_ref=expected_ref, cell=r,
                statistic="regime_motion_share/core32",
                s=M.surprise(core[r], core["supervised"], twins["supervised"]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "types": list(types), "layers": layers, "lr": a.lr,
                  "max_steps": a.max_steps, "match_steps": match_steps,
                  "target_motion": target, "regimes": regimes,
                  "optimizer": optinfo, "cells": cells, "matched": matched,
                  "core_share_by_regime": core, "differing_regimes": differs,
                  "naming": ("the reinforce arm is REINFORCE with a baseline, not clipped-ratio "
                             "PPO; the loss shape is the same and the machinery is not")})
    print("A5_OK " + json.dumps({"rows": info["n_rows"], "core_share": core,
                                 "matched": matched, "differing": differs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
