"""M-4 — the erosion-curve program (queue item 4; A1 rides inside). Wave 1.

    --stage confound   THE MARK-VS-FUNCTION ARM, and it gates the other one. W1-L measured that
                       stamps placed in the sink erode far less than stamps placed in the highway.
                       Two readings fit that: the SUBSPACE protects the stamp, or sink stamps were
                       simply SMALLER — a mark rather than a function — and small perturbations
                       survive everywhere. This arm places stamps at matched, EDIT-SCALE amplitudes
                       in every subspace and asks whether the difference survives.
                       REFUTER: large stamps still die everywhere.

    --stage curve      THE EROSION CURVE, only worth running if the confound arm leaves something
                       to explain: decay rate vs subspace write-traffic across 8 subspaces × 4
                       durations × 2 recipes, then one fit.
                       REFUTER: no monotone traffic–erosion relation.

THE EDIT SCALE IS MEASURED, NOT CHOSEN. "Edit-scale amplitude" only means something relative to
what a real fine-tune of these cells actually moves, so the job first runs the recipe UNSTAMPED and
takes ‖ΔW‖_F per cell as the scale. A hardcoded amplitude would make the confound arm's answer a
property of the number someone picked.

DURATIONS ARE NESTED, NOT SEPARATE RUNS. One training run is checkpointed at every duration on the
ladder, so the 4 durations of a cell are 4 readings of ONE trajectory rather than 4 trajectories
that differ by their random draws as well as by their length.

A1 RIDES INSIDE. Every treatment is paired with a control at matched REALIZED motion via
`matched_motion_partner`, because REFEREE's successor question is whether an effect is about the
subspace or about how far the weights moved.

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
from infra.pg import mercator_erosion as E
from infra.pg import mercator_program as P

ARM = "M-4"
BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
           "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
RECIPES = ("code", "wikitext")          # the damaging narrow-grammar recipe, and base-distribution
# Below this spread across the duration ladder there is no rate to fit; the cell is recorded as
# flat and the curve is built from whatever cells did vary.
FLATNESS_FLOOR = 0.01
# THE DURATIONS HAVE TO STRADDLE THE EROSION, and the first curve run at an eroding amplitude
# showed they did not: retention was 0.824, 0.824, 0.824, 0.824 across 30-240 steps, i.e. the
# stamp lost what it was going to lose inside the first 30 steps and then held. An exponential fit
# on a flat series is a fit to nothing, and `fit_decay_vs_traffic` correctly refused it (some rates
# came out negative). The ladder now starts inside the erosion (defect M-D28).
DURATIONS = (2, 5, 15, 45, 135)


def main():
    ap = argparse.ArgumentParser(description="M-4 — mark-vs-function, then the erosion curve")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=("confound", "curve"))
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="12,13,14,15,16,17")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--n-amplitudes", type=int, default=4)
    ap.add_argument("--curve-amplitude", type=float, default=0.0,
                    help=("curve stage: the stamp amplitude, ABSOLUTE. 0 means 'the rung of the "
                          "confound ladder with the most dynamic range', which is the smallest."))
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id=f"m_4_{a.stage}", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import (build_batches, text_task_rows, wikitext_rows)

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    for p in model.parameters():
        p.requires_grad_(False)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    for mod in mods.values():
        mod.weight.requires_grad_(True)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    train_rows, _ = text_task_rows(a.n_train, 16, a.seed)
    batches = {"code": build_batches(processor, train_rows, "cuda"),
               "wikitext": build_batches(processor, wikitext_rows(a.n_train, a.seed), "cuda")}
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    banks = {k: {b: WT.load_bank(bank_dir, k[0], k[1], b) for b in BATTERY} for k in mods}
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])

    def train(recipe, steps, seed, stamped=None, checkpoints=()):
        """Train `steps` steps, optionally starting from a stamped model.

        Returns {step: {(t,L): realized delta from the UNSTAMPED pristine weights}} at each
        requested checkpoint plus the final step. The delta is measured against `pristine`, not
        against the stamped start, so it contains the stamp plus whatever training did to it —
        which is exactly what `retained_fraction` is defined on.
        """
        for k, m in mods.items():
            w = pristine[k].clone()
            if stamped is not None:
                w += torch.as_tensor(stamped[k], device=w.device, dtype=w.dtype)
            m.weight.data.copy_(w.to(m.weight.dtype))
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        rng = np.random.default_rng(H.cell_seed(seed, ARM, recipe, str(steps)))
        want = sorted(set(list(checkpoints) + [steps]))
        snaps = {}
        model.train()
        for step in range(1, steps + 1):
            b = batches[recipe][int(rng.integers(0, len(batches[recipe])))]
            opt.zero_grad()
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.backward()
            opt.step()
            if step in want:
                snaps[step] = {k: (m.weight.detach().to(torch.float32)
                                   - pristine[k]).cpu().numpy().astype(np.float64)
                               for k, m in mods.items()}
        return snaps

    # ---- the measured edit scale --------------------------------------------------------------
    ref = train("code", max(DURATIONS), a.seed)[max(DURATIONS)]
    edit_scale = {f"{k[0]}.{k[1]}": float(np.linalg.norm(v)) for k, v in ref.items()}
    scale_mean = float(np.mean(list(edit_scale.values())))
    at.add_ci("measured_edit_scale", np.array(list(edit_scale.values())),
              model_id=WT.PG1_REPO, model_sha=base_sha,
              flags={"definition": "||dW||_F per cell after the unstamped damaging recipe",
                     "steps": max(DURATIONS), "recipe": "code", "lr": a.lr})
    print(f"[M-4] measured edit scale (mean ||dW||_F) = {scale_mean:.5g} over {len(mods)} cells; "
          f"base ppl {base_ppl:.4f}", flush=True)

    results = {}
    if a.stage == "confound":
        amps = E.amplitude_ladder(scale_mean, a.n_amplitudes)
        M.check_dose_grid(amps, arm=ARM)
        print(f"[M-4] amplitude ladder {['%.4g' % x for x in amps]}", flush=True)
        for bid in BATTERY:
            for amp in amps:
                stamps, dirs = {}, {}
                for k in mods:
                    D, d = E.stamp(banks[k][bid], amp, H.cell_seed(a.seed, ARM, bid, f"{amp:g}"),
                                   out_dim=int(mods[k].weight.shape[0]))
                    stamps[k], dirs[k] = D, d
                snap = train("code", max(DURATIONS), a.seed, stamped=stamps)[max(DURATIONS)]
                ret = float(np.mean([E.retained_fraction(snap[k], dirs[k]) / amp for k in mods]))
                traffic = float(np.mean([E.write_traffic(ref[k], banks[k][bid]) for k in mods]))
                motion = float(np.mean([np.linalg.norm(snap[k]) for k in mods]))
                cell = {"bank": bid, "amplitude": amp, "retained_fraction": ret,
                        "write_traffic": traffic, "realized_motion": motion,
                        "amplitude_over_edit_scale": amp / scale_mean}
                results[f"{bid}@{amp:g}"] = cell
                at.add("stamp_retained_fraction", value=ret, dose=amp, bank_id=bid,
                       model_id=WT.PG1_REPO, model_sha=base_sha,
                       flags={**cell, "steps": max(DURATIONS), "recipe": "code",
                              "edit_scale": scale_mean, "optimizer": optinfo,
                              "note": ("not clipped: >1 means training reinforced the stamp, "
                                       "<0 means it reversed it")})
                print(f"[M-4] confound {bid:14s} amp={amp:9.4g} "
                      f"({amp / scale_mean:5.2f}x edit) retained={ret:+.4f} "
                      f"traffic={traffic:.5f}", flush=True)
        at.check_controls_per_statistic(["stamp_retained_fraction"])
        top = max(amps)
        # THE SPEC'S REFUTER IS ABOUT LARGE STAMPS, so it is read at the largest amplitude, which
        # is where it belongs and where it stays.
        all_dead = bool(max(results[f"{b}@{top:g}"]["retained_fraction"] for b in BATTERY) < 0.1)
        gates.record("m4_large_stamps_still_die_everywhere", all_dead,
                     retained_at_max_amplitude={b: results[f"{b}@{top:g}"]["retained_fraction"]
                                                for b in BATTERY},
                     amplitude=top, amplitude_over_edit_scale=top / scale_mean,
                     note="this gate PASSING is the refuter firing")
        # THE SUBSPACE COMPARISON IS READ WHERE IT HAS DYNAMIC RANGE, and the rule is stated
        # rather than chosen: the rung whose mean retention across banks is furthest from 1.0.
        # A stamp ten times the size of anything training does survives in every subspace, so a
        # comparison taken there is a comparison of two ceilings — which is what the first run of
        # this arm did by reading the top rung (defect M-D25). The amplitude that carries the
        # comparison is recorded on the row, so the reading can never be quoted without it.
        def spread_at(x):
            return 1.0 - float(np.mean([results[f"{b}@{x:g}"]["retained_fraction"]
                                        for b in BATTERY]))

        work_amp = max(amps, key=spread_at)
        core_ret = results[f"core32@{work_amp:g}"]["retained_fraction"]
        twins = [results[f"{b}@{work_amp:g}"]["retained_fraction"]
                 for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
        at.add("working_stamp_amplitude", value=float(work_amp), bank_id=None,
               flags={"rule": "the ladder rung whose mean retention is furthest from 1.0",
                      "mean_erosion_by_amplitude": {f"{x:g}": spread_at(x) for x in amps},
                      "amplitude_over_edit_scale": work_amp / scale_mean})
        gates.record("m4_core_stamp_outlives_twins", core_ret > max(twins),
                     core=core_ret, twins=twins, amplitude=work_amp,
                     amplitude_over_edit_scale=work_amp / scale_mean,
                     retained_by_bank={b: results[f"{b}@{work_amp:g}"]["retained_fraction"]
                                       for b in BATTERY},
                     ladder={f"{x:g}": {b: results[f"{b}@{x:g}"]["retained_fraction"]
                                        for b in BATTERY} for x in amps},
                     note=("read at the rung with dynamic range; the full ladder is on the row so "
                           "a saturated rung cannot be mistaken for agreement"))
        # A1: the motion-matched partner for the core cell, chosen from the other banks.
        # `matched_motion_partner` FAILS LOUD when nothing matches within tolerance, which is the
        # right contract for a comparison but the wrong thing to crash a whole arm over — so the
        # job checks matchability first and records UNMATCHED as a result rather than as an abort.
        target = results[f"core32@{work_amp:g}"]["realized_motion"]
        cands = {b: results[f"{b}@{work_amp:g}"]["realized_motion"]
                 for b in BATTERY if b != "core32"}
        gaps = {b: abs(v - target) / target for b, v in cands.items()}
        best = min(gaps, key=gaps.get)
        matched = gaps[best] <= E.MOTION_MATCH_TOL
        partner = (E.matched_motion_partner(cands, target) if matched else
                   {"matched": False, "nearest": best, "relative_gap": gaps[best],
                    "reason": "no control cell within tolerance of the treatment's realized motion"})
        gap = (core_ret - results[f"{best}@{work_amp:g}"]["retained_fraction"]) if matched else 0.0
        at.add("motion_matched_partner_gap", value=float(gap), bank_id="core32",
               flags={**partner, "core_retained": core_ret, "matched": matched,
                      "relative_gap": gaps[best], "nearest": best,
                      "outcome": None if matched else "UNMATCHED",
                      "definition": "core retention minus its motion-matched partner's"})
        sur.add(arm=ARM, expected_ref=expected_ref, cell=f"amp={work_amp:g}",
                statistic="stamp_retained_fraction/core32",
                s=M.surprise(core_ret, float(np.mean(twins)), twins))
    else:
        shape = E.check_sweep(BATTERY, DURATIONS, RECIPES)
        print(f"[M-4] sweep {shape}", flush=True)
        # THE CURVE NEEDS AN AMPLITUDE THAT ERODES. The top of the ladder is ten times the edit
        # scale and survives everything, so a decay constant fitted there is a fit to 1.000 four
        # times over (defect M-D25). Default to the SMALLEST rung, which the confound stage
        # measured as the one with dynamic range, and let the caller override with a number.
        amp = (a.curve_amplitude if a.curve_amplitude > 0
               else E.amplitude_ladder(scale_mean, a.n_amplitudes)[0])
        print(f"[M-4] curve stamp amplitude {amp:.5g} ({amp / scale_mean:.2f}x edit scale)",
              flush=True)
        traffic_by, lam_by = {}, {}
        for recipe in RECIPES:
            tref = train(recipe, max(DURATIONS), a.seed)[max(DURATIONS)]
            for bid in BATTERY:
                stamps, dirs = {}, {}
                for k in mods:
                    D, d = E.stamp(banks[k][bid], amp, H.cell_seed(a.seed, ARM, bid, recipe),
                                   out_dim=int(mods[k].weight.shape[0]))
                    stamps[k], dirs[k] = D, d
                snaps = train(recipe, max(DURATIONS), a.seed, stamped=stamps,
                              checkpoints=DURATIONS)
                retained = [float(np.mean([E.retained_fraction(snaps[t][k], dirs[k]) / amp
                                           for k in mods])) for t in DURATIONS]
                traffic = float(np.mean([E.write_traffic(tref[k], banks[k][bid]) for k in mods]))
                for t, r in zip(DURATIONS, retained):
                    at.add("stamp_retained_fraction", value=r, dose=float(t), bank_id=bid,
                           model_id=WT.PG1_REPO, model_sha=base_sha,
                           flags={"recipe": recipe, "duration": t, "amplitude": amp,
                                  "write_traffic": traffic, "optimizer": optinfo})
                fully_eroded = bool(min(retained) <= 0)
                # FLAT IS NOT A DECAY. If retention does not vary across the ladder there is no
                # rate to fit, and forcing one produces a number whose sign is noise. Recorded as
                # a reading rather than crashed on.
                spread = float(max(retained) - min(retained))
                flat = bool(spread < FLATNESS_FLOOR)
                dec = (None if (fully_eroded or flat) else
                       E.decay_constant(list(DURATIONS), retained))
                results[f"{recipe}/{bid}"] = {"retained": retained, "write_traffic": traffic,
                                              "decay": dec, "fully_eroded": fully_eroded,
                                              "flat": flat, "retention_spread": spread,
                                              "final_retained": retained[-1]}
                if dec is not None:
                    at.add("stamp_decay_constant", value=dec["lambda"], bank_id=bid,
                           model_id=WT.PG1_REPO, model_sha=base_sha,
                           flags={**dec, "recipe": recipe, "write_traffic": traffic})
                    traffic_by.setdefault(recipe, []).append(traffic)
                    lam_by.setdefault(recipe, []).append(dec["lambda"])
                print(f"[M-4] curve {recipe:9s} {bid:14s} traffic={traffic:.5f} "
                      f"retained={['%.3f' % r for r in retained]} "
                      f"lambda={'eroded' if dec is None else '%.5g' % dec['lambda']}", flush=True)
        at.check_controls_per_statistic(["stamp_retained_fraction"])
        # THE CURVE IS CIRCULAR AT AN AMPLITUDE THAT DOES NOT ERODE, and the first run of this
        # stage proved it. With a stamp ten times the training motion, retained stays at 1.0000 and
        #
        #     retained - 1  =  <dW_train, d> / amp
        #
        # where d is a random direction inside U. The expected size of that inner product is
        # proportional to ||P_U dW_train|| / sqrt(k) — which is the square root of the WRITE
        # TRAFFIC, by construction. So "decay rate rises with write traffic" comes out monotone
        # (Spearman 1.000 on one recipe) whether or not anything eroded, and the fit is of the
        # construction rather than of erosion (defect M-D25). The guard is that the stamp must
        # actually lose something somewhere before a curve is fitted at all.
        EROSION_FLOOR = 0.02
        worst = min(min(v["retained"]) for v in results.values())
        eroded = bool(1.0 - worst >= EROSION_FLOOR)
        gates.record("m4_stamps_actually_erode_at_this_amplitude", eroded,
                     min_retained=worst, floor=EROSION_FLOOR, amplitude=amp,
                     amplitude_over_edit_scale=amp / scale_mean,
                     outcome=None if eroded else M.VOID_INSTRUMENT,
                     note=("without erosion, retained-1 is the training update's projection onto "
                           "a random direction in the subspace, whose size IS sqrt(write "
                           "traffic) — so the curve would fit its own construction"))
        assert eroded, (
            f"no cell lost more than {1.0 - worst:.5f} of its stamp at amplitude {amp:.5g} "
            f"({amp / scale_mean:.2f}x edit scale); a decay-vs-traffic fit here would recover the "
            f"construction, not erosion. Re-run at an amplitude with dynamic range.")
        fits = {}
        for recipe in RECIPES:
            tv, lv = traffic_by.get(recipe, []), lam_by.get(recipe, [])
            n_flat = sum(1 for b in BATTERY if results[f"{recipe}/{b}"]["flat"])
            if len(tv) < E.MIN_SUBSPACES or min(lv, default=0.0) <= 0:
                fits[recipe] = {"fitted": False, "n_points": len(tv), "n_flat": n_flat,
                                "reason": (f"{len(tv)} of {len(BATTERY)} subspaces left a "
                                           f"fittable positive decay rate ({n_flat} were flat "
                                           f"across the duration ladder, the rest fully eroded "
                                           f"or non-positive); a power law needs positive rates"),
                                "outcome": M.VOID_INSTRUMENT}
                continue
            f = E.fit_decay_vs_traffic(tv, lv, seed=a.seed)
            fits[recipe] = {"fitted": True, "n_flat": n_flat, **f}
            at.add("erosion_curve_slope", value=float(f["slope"]), bank_id=None,
                   ci_lo=f["slope_ci_lo"], ci_hi=f["slope_ci_hi"], n=len(tv),
                   flags={**f, "recipe": recipe})
        # THE READING THAT SURVIVES A FLAT LADDER: retention at the END of the window against write
        # traffic. It is a level rather than a rate, so it answers a weaker question than the
        # decay curve — but it is measured, and a monotone relation here is the traffic-erosion
        # relation the refuter is written against, one derivative down.
        for recipe in RECIPES:
            tv = [results[f"{recipe}/{b}"]["write_traffic"] for b in BATTERY]
            rv = [results[f"{recipe}/{b}"]["final_retained"] for b in BATTERY]
            rho = E._spearman(tv, rv)
            # LEAVE-CORE-OUT, because one point can carry a rank correlation. core32 is expected
            # to sit at BOTH extremes — least written and best retained — so a Spearman computed
            # with it included can read -1 while the other seven subspaces show no relation at
            # all. If the traffic story is real it has to survive dropping its own best example.
            others = [b for b in BATTERY if b != "core32"]
            tv_o = [results[f"{recipe}/{b}"]["write_traffic"] for b in others]
            rv_o = [results[f"{recipe}/{b}"]["final_retained"] for b in others]
            rho_o = E._spearman(tv_o, rv_o)
            at.add("erosion_level_spearman_without_core", value=float(rho_o), bank_id=None,
                   flags={"recipe": recipe, "banks": others, "traffic": tv_o,
                          "retention": rv_o,
                          "definition": ("the same rank correlation with core32 dropped; a "
                                         "relation that only exists with it is one point")})
            fits[recipe] = {**fits.get(recipe, {}),
                            "level_spearman_traffic_vs_retention": float(rho),
                            "level_spearman_without_core": float(rho_o),
                            "traffic_range_without_core": [float(min(tv_o)), float(max(tv_o))],
                            "retention_range_without_core": [float(min(rv_o)), float(max(rv_o))],
                            "level_note": ("Spearman of write traffic against retention at the "
                                           "LAST duration; negative means more-written subspaces "
                                           "retain less, which is the relation the refuter is "
                                           "about, measured as a level rather than a rate")}
            at.add("erosion_level_spearman", value=float(rho), bank_id=None,
                   flags={"recipe": recipe, "traffic": tv, "retention": rv,
                          "banks": list(BATTERY)})
        monotone = {r: bool((v.get("fitted") and abs(v.get("spearman", 0.0)) >= 0.6)
                            or abs(v.get("level_spearman_traffic_vs_retention", 0.0)) >= 0.6)
                    for r, v in fits.items()}
        gates.record("m4_monotone_traffic_erosion_relation", bool(all(monotone.values())),
                     by_recipe=monotone, fits=fits,
                     note=("the REFUTER is written against MONOTONICITY, not against a functional "
                           "form, so Spearman gates and the power-law slope is reported"))
        # THE SIGN IS THE CLAIM, not the magnitude. "More write traffic erodes a stamp faster"
        # predicts a NEGATIVE rank correlation between traffic and retention; a strong POSITIVE
        # one is the opposite result and must not pass. The first version tested |rho| >= 0.6 and
        # would have passed the wikitext recipe's +0.714 — a gate that cannot fail in the
        # direction that matters (defect M-D34).
        survives = {r: bool(fits[r].get("level_spearman_without_core", 0.0) <= -0.6)
                    for r in RECIPES}
        signs = {r: float(fits[r].get("level_spearman_without_core", 0.0)) for r in RECIPES}
        gates.record("m4_relation_survives_dropping_core32", bool(all(survives.values())),
                     by_recipe=survives, spearman_signed=signs,
                     required="rho <= -0.6 (more traffic, less retention)",
                     recipes_with_the_wrong_sign=[r for r, v in signs.items() if v > 0],
                     spearman_with_core={r: fits[r].get(
                         "level_spearman_traffic_vs_retention") for r in RECIPES},
                     spearman_without_core={r: fits[r].get(
                         "level_spearman_without_core") for r in RECIPES},
                     note=("core32 is expected at both extremes, so a relation that vanishes "
                           "without it is one point rather than a curve — and the engineering "
                           "deliverable needs a curve"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_{a.stage}_result.json",
                 {"arm": ARM, "stage": a.stage, "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "base_repo": WT.PG1_REPO, "base_sha": base_sha, "types": list(types),
                  "layers": layers, "lr": a.lr, "durations": list(DURATIONS),
                  "recipes": list(RECIPES), "optimizer": optinfo, "base_ppl": base_ppl,
                  "measured_edit_scale": edit_scale, "edit_scale_mean": scale_mean,
                  "cells": results,
                  **({"fits": fits} if a.stage == "curve" else {})})
    print(f"M4_{a.stage.upper()}_OK " + json.dumps({"rows": info["n_rows"],
                                                    "edit_scale": scale_mean}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
