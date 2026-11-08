"""M-A2 — the breathe ceiling (queue A2). Wave 1.

W1-S opened the valve to 21x realized core motion with no damage; REFEREE's full SGD dose destroyed
the model. **The band between is unmapped.** This arm extends W1-S's gain ladder until damage
begins, adds the retention readout W1-S lacked, replicates W1-S's g4-better-than-stock anomaly over
seeds, and asks whether the damage that appears follows REFEREE's motion power law.

    REFUTER (pre-logged): no damage onset within the swept ladder, OR damage onset not tracked by
    the REFEREE power law.

FIVE THINGS THAT MAKE IT A MEASUREMENT RATHER THAN A DEMONSTRATION:

  * THE RECIPE IS W1-S's, ARGUMENT FOR ARGUMENT (q_proj+o_proj, all 18 layers, 120 steps, lr 1e-5,
    the same code-completion task, the same wikitext ppl readout). The claim "the two ladders join
    into one curve" is only true if the runs differ in the gain and nothing else, so the defaults
    here are copied from `darwin_w1_optdata` rather than chosen.
  * THE LADDER CONTINUES W1-S's GEOMETRICALLY and re-measures its top rung: W1-S ran 1/2/4/8, this
    runs 8/16/32/64/128. The shared rung at 8 is what lets a disagreement between the two arms be
    seen instead of assumed away.
  * THE VALVE IS OPENED ON THE FULL EIGHT-BANK BATTERY, at every gain. W1-S valved core32 ONLY and
    used the battery to *measure* motion, so it never had the control that matters: amplifying ANY
    32-dimensional subspace of the update costs something, and the question is only ever whether
    amplifying the sink costs MORE. Damage onset is read against the twins' own onset at the same
    dose, never against zero.
  * REALIZED MOTION IS MEASURED IN REFEREE'S OWN UNITS. A gain of 64 does not mean 64x the motion —
    the phenomenon under study is that the optimizer does not deliver requested motion — and
    REFEREE's `motion_units` is a RATIO of realized in-subspace motion to a matched reference run
    (`p4_ratio_horizon_matched * lambda`), not an energy fraction. So this arm measures
    ||dW U||_F in the valved run over the same quantity in the valve-off run at the same step
    budget. Feeding a fraction into a law fitted on a ratio would be the same class of error the
    scale mismatch below guards against.
  * THE POWER LAW IS READ FROM THE ANCHOR ARTIFACT, WITH ITS SCALE. MERCATOR's own anchor recompute
    found REFEREE's exponent is 1.4175 on ppl/ppl_base and 1.9922 on the excess scale over the same
    nine rows, so the fit is FETCHED by name from `anchors.json` rather than transcribed, and the
    damage series is built on whichever scale the fetched fit was made on. There is no default
    scale and no hand-copied coefficient.

§0.2.7 plus amendment A1 bind here: M-A2 is an optimizer-mechanics arm, so approximate Adam is
prohibited and `offload_exact_fp32` is stated explicitly.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P
from infra.pg import mercator_train as TR

ARM = "M-A2"
# The banks the valve is opened on. core32 is the treatment; the rest are the control battery, and
# every one of them is valved at every gain, because §3's battery is a property of the CONTRAST and
# a partial battery on the damage statistic is critic finding F7 no matter how many banks the
# arm's other statistics carry.
VALVE_BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
               "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"   # darwin_w1_u's recorded recipe substitution
ANCHORS_S3 = f"s3://{M.BUCKET}/results/mercator_preflight3/out/anchors/anchors.json"
# W1-S's anomaly cell: at gain 4 the valved run scored BETTER than stock. One reading is a real
# regularisation effect, the other is one seed. This is the gain that gets replicated.
ANOMALY_GAIN = 4.0
# ALL THREE damage scales the anchor recompute fitted, each built from the ppl ratio and the
# reference ppl. Naming them here is what makes "which scale" an argument rather than an
# assumption, and carrying all three is what makes a disagreement between them visible: the same
# nine REFEREE rows give exponent 1.42 on one and 1.99 on another, so a law "confirmed" on an
# unnamed scale is not confirmed at all.
SCALES = {"ratio": lambda r, p0: r,
          "excess_ratio": lambda r, p0: r - 1.0,
          "delta_ppl": lambda r, p0: (r - 1.0) * p0}


def referee_law(anchors_path: pathlib.Path, *, task: str, scale: str) -> dict:
    """The fitted power law for one task on one damage scale, read from the anchor artifact.

    The anchor job fits log(damage) = exponent * log(motion_units) + intercept, so the coefficient
    a caller needs is exp(intercept). Returning it computed here — rather than letting each arm
    exponentiate an intercept it transcribed — is the point.
    """
    d = json.loads(anchors_path.read_text())["anchors"]["REFEREE"]["damage_power_law"]
    assert task in d, f"no REFEREE power-law fit for task {task}; have {sorted(d)}"
    fit = d[task]["by_scale"]
    assert scale in fit, f"no fit on scale {scale}; have {sorted(fit)}"
    f = fit[scale]
    assert f["fitted"], f"REFEREE's {scale} fit did not converge: {f.get('reason')}"
    return {"exponent": float(f["exponent"]), "coefficient": float(np.exp(f["intercept"])),
            "intercept": float(f["intercept"]), "r2_log_in_referee": float(f["r2_log"]),
            "scale": scale, "task": task, "n_points_in_referee": int(d[task]["n_points"]),
            "source": str(anchors_path)}


def main():
    ap = argparse.ArgumentParser(description="M-A2 — the breathe ceiling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gains", default="")
    # W1-S's recipe, argument for argument (darwin_w1_optdata defaults).
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--anomaly-seeds", type=int, default=3)
    ap.add_argument("--anchors-s3", default=ANCHORS_S3)
    ap.add_argument("--referee-task", default="textvqa")
    ap.add_argument("--damage-scale", required=True, choices=sorted(SCALES),
                    help="the scale REFEREE's exponent was fitted on; there is no default")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gains = [float(x) for x in a.gains.split(",") if x.strip()] or TR.gain_ladder(8.0, 5)
    M.check_dose_grid(gains, arm=ARM)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_a2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

    anchors = H.cache_dir(out, "anchors") / "anchors.json"
    anchors.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", a.anchors_s3, str(anchors), "--region", "us-east-1"],
                   check=True)
    law_params = referee_law(anchors, task=a.referee_task, scale=a.damage_scale)
    print(f"[M-A2] REFEREE law ({a.damage_scale}): damage = {law_params['coefficient']:.4g} * "
          f"motion^{law_params['exponent']:.4f}", flush=True)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    for mod in mods.values():
        mod.weight.requires_grad_(True)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}

    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, a.seed)
    train_b = build_batches(processor, train_rows, "cuda")
    eval_b = build_batches(processor, eval_rows, "cuda")
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    banks = {k: {b: WT.load_bank(bank_dir, k[0], k[1], b) for b in VALVE_BANKS} for k in mods}
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    print(f"[M-A2] base ppl={base_ppl:.5f} over {len(mods)} cells", flush=True)

    def run_cell(bank, gain, seed):
        """One fine-tune at one gain on one bank. The model is restored to pristine first, so the
        only thing that differs between cells is the valve and the batch order."""
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        valve = TR.BreatheValve({k: banks[k][bank] for k in mods}, "cuda", gain) if bank else None
        rng = np.random.default_rng(H.cell_seed(seed, ARM, bank or "valve_off", str(gain)))
        n_applied = 0
        model.train()
        for _step in range(a.steps):
            b = train_b[int(rng.integers(0, len(train_b)))]
            opt.zero_grad()
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.backward()
            if valve is not None:
                n_applied = valve.apply(mods)
            opt.step()
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        task = eval_task_loss(model, eval_b)
        # Realized motion, from the weights: the ABSOLUTE in-core norm (which becomes REFEREE's
        # ratio once divided by the reference run) and the energy FRACTION (a different object,
        # kept as a secondary reading and never fed to the law).
        core_norm, frac, tot = [], [], []
        for k, m in mods.items():
            dW = (m.weight.detach().to(torch.float32) - pristine[k]).cpu().numpy()
            core_norm.append(float(np.linalg.norm(dW @ banks[k]["core32"])))
            frac.append(T.motion_fraction(dW, banks[k]["core32"]))
            tot.append(float(np.linalg.norm(dW)))
        return {"bank": bank or "valve_off", "gain": gain, "seed": seed, "ppl": ppl,
                "task_loss": task, "core_motion_norm": float(np.mean(core_norm)),
                "core_motion_fraction": float(np.mean(frac)),
                "total_motion": float(np.mean(tot)), "n_modules_valved": n_applied}

    def record(cell, *, statistic_suffix=""):
        """Damage and motion are BOTH relative to the valve-off reference at the same step budget,
        so 'damage' means what the valve added and 'motion' is REFEREE's ratio object."""
        cell["damage_ratio"] = cell["ppl"] / ref["ppl"]
        cell["damage_by_scale"] = {sc: f(cell["damage_ratio"], ref["ppl"])
                                   for sc, f in SCALES.items()}
        cell["damage"] = cell["damage_by_scale"][a.damage_scale]
        cell["motion_units"] = cell["core_motion_norm"] / ref["core_motion_norm"]
        cell["retention_delta"] = cell["task_loss"] - ref["task_loss"]
        bid = None if cell["bank"] == "valve_off" else cell["bank"]
        at.add(f"breathe_ppl{statistic_suffix}", value=cell["ppl"], dose=cell["gain"],
               bank_id=bid, corpus_id=PPL_CORPUS, model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={**cell, "damage_scale": a.damage_scale, "base_ppl": base_ppl,
                      "steps": a.steps, "lr": a.lr, "optimizer": optinfo})
        at.add(f"breathe_retention{statistic_suffix}", value=cell["task_loss"], dose=cell["gain"],
               bank_id=bid, corpus_id=TASK_CORPUS, model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={"retention_delta": cell["retention_delta"],
                      "recipe_substitution": ("code completion stands in for refcoco-seg; "
                                              "darwin_w1_u.text_task_rows records why")})
        at.add(f"breathe_motion_units{statistic_suffix}", value=cell["motion_units"],
               dose=cell["gain"], bank_id=bid, model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={"definition": ("||dW U_core||_F relative to the valve-off run at the same "
                                     "step budget; REFEREE's motion_units object"),
                      "core_motion_norm": cell["core_motion_norm"],
                      "core_motion_fraction": cell["core_motion_fraction"],
                      "total_motion": cell["total_motion"]})
        print(f"[M-A2] {cell['bank']:14s} gain={cell['gain']:6.1f} s{cell['seed']} "
              f"ppl={cell['ppl']:8.3f} damage={cell['damage']:+.4f} "
              f"motion={cell['motion_units']:7.3f}x task={cell['task_loss']:.4f}", flush=True)
        return cell

    # ---- the reference, then the ladder ------------------------------------------------------
    ref = run_cell(None, 1.0, a.seed)
    assert ref["core_motion_norm"] > 0, "the valve-off reference did not move; nothing to divide by"
    record(ref)
    results = {"valve_off@1.0": ref}
    for bank in VALVE_BANKS:
        for gain in gains:
            results[f"{bank}@{gain}"] = record(run_cell(bank, gain, a.seed))

    # ---- the g4 anomaly replication cell -----------------------------------------------------
    # W1-S saw gain 4 score BETTER than stock at one seed. Replication needs BOTH arms re-run at
    # fresh seeds: stock's own seed-to-seed spread is the only thing that says whether "better than
    # stock" is an effect or a draw. Recorded under its own statistic so the ladder's damage series
    # is not contaminated by cells at a gain the ladder does not contain.
    anomaly = {"gain": ANOMALY_GAIN, "seeds": [], "core32_ppl": [], "valve_off_ppl": []}
    for i in range(a.anomaly_seeds):
        sd = a.seed + 1 + i
        off = record(run_cell(None, 1.0, sd), statistic_suffix="_anomaly")
        on = record(run_cell("core32", ANOMALY_GAIN, sd), statistic_suffix="_anomaly")
        anomaly["seeds"].append(sd)
        anomaly["valve_off_ppl"].append(off["ppl"])
        anomaly["core32_ppl"].append(on["ppl"])
        results[f"anomaly_valve_off@s{sd}"] = off
        results[f"anomaly_core32@s{sd}"] = on
    d_anom = np.asarray(anomaly["core32_ppl"]) - np.asarray(anomaly["valve_off_ppl"])
    anomaly["paired_delta_ppl"] = d_anom.tolist()
    anomaly["mean_delta_ppl"] = float(d_anom.mean())
    anomaly["replicates"] = bool((d_anom < 0).all())
    anomaly["n_seeds_better_than_stock"] = int((d_anom < 0).sum())
    at.add("g4_anomaly_paired_delta_ppl", value=anomaly["mean_delta_ppl"], dose=ANOMALY_GAIN,
           bank_id="core32", corpus_id=PPL_CORPUS, model_id=WT.PG1_REPO, model_sha=base_sha,
           n=len(anomaly["seeds"]),
           flags={**anomaly, "definition": ("ppl(core32 valved at g4) - ppl(valve off), paired by "
                                            "seed; negative is the W1-S anomaly's direction")})
    gates.record("a2_g4_anomaly_replicates", anomaly["replicates"], **anomaly,
                 note=("W1-S's single-seed reading; replication means every fresh seed pair moves "
                       "in the same direction, not that the mean does"))

    # ---- onset, and the power law ------------------------------------------------------------
    at.check_controls_per_statistic(["breathe_ppl", "breathe_retention", "breathe_motion_units"])
    at.check_readout_corpus(["breathe_ppl", "breathe_retention", "breathe_ppl_anomaly",
                             "breathe_retention_anomaly", "g4_anomaly_paired_delta_ppl"])

    def series(bank, key):
        return [results[f"{bank}@{g}"][key] for g in gains]

    twin_dmg = np.array([series(b, "damage") for b in ("haar32_s0", "haar32_s1", "haar32_s2")])
    band_hi = float(twin_dmg.max())
    core_dmg, core_motion = series("core32", "damage"), series("core32", "motion_units")
    # Onset is invariant to which scale is used (all three are monotone in the ppl ratio, and the
    # twin band is transformed with it), so it is reported once.
    onset = TR.damage_onset(gains, core_dmg, band_hi=band_hi)
    at.add("breathe_damage_onset", value=float(onset["onset"] or 0.0), bank_id="core32",
           flags={**onset, "twin_band_hi": band_hi, "gains": gains,
                  "damage_scale": a.damage_scale, "onset_found": onset["onset"] is not None})
    # The power law IS scale-dependent, so every scale REFEREE fitted is evaluated and recorded.
    # One of them is the pre-registered reading (--damage-scale) and gates; the others are here so
    # that "the law picks up" can never be a statement about a scale nobody named.
    law_by_scale = {}
    for sc in SCALES:
        lp = referee_law(anchors, task=a.referee_task, scale=sc)
        d_sc = [results[f"core32@{g}"]["damage_by_scale"][sc] for g in gains]
        lw = TR.power_law_pickup(core_motion, d_sc, exponent=lp["exponent"],
                                 coefficient=lp["coefficient"])
        law_by_scale[sc] = {**lw, "referee_fit": lp}
        at.add("breathe_power_law_r2", value=float(lw["r2_against_referee_law"]),
               bank_id="core32", flags={**lw, "referee_fit": lp, "damage_scale": sc,
                                        "preregistered": sc == a.damage_scale})
    law = law_by_scale[a.damage_scale]
    s = M.surprise(core_dmg[-1], float(twin_dmg[:, -1].mean()), list(twin_dmg[:, -1]))
    sur.add(arm=ARM, expected_ref=expected_ref, cell=f"gain={gains[-1]}",
            statistic="breathe_damage/core32", s=s)
    gates.record("a2_damage_onset_found", onset["onset"] is not None, **onset)
    gates.record("a2_power_law_picks_up", float(law["r2_against_referee_law"]) > 0.0,
                 r2=float(law["r2_against_referee_law"]),
                 mean_ratio=law["mean_ratio_observed_over_predicted"],
                 damage_scale=a.damage_scale, referee_fit=law_params,
                 note=("R2 <= 0 means REFEREE's law predicts this arm's damage no better than "
                       "this arm's own mean; that is the second half of the refuter"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "gains": gains, "layers": layers, "types": list(types), "steps": a.steps,
                  "lr": a.lr, "n_train": a.n_train, "n_eval": a.n_eval, "n_seq": a.n_seq,
                  "recipe_source": ("darwin_w1_optdata defaults, copied so the two ladders join"),
                  "optimizer": optinfo, "base_ppl": base_ppl, "valve_off_reference": ref,
                  "cells": results, "damage_onset": onset, "twin_band_hi": band_hi,
                  "power_law": law, "power_law_by_scale": law_by_scale,
                  "damage_scale": a.damage_scale, "referee_fit": law_params,
                  "g4_anomaly": anomaly})
    print("A2_OK " + json.dumps({"rows": info["n_rows"], "onset": onset["onset"],
                                 "twin_band_hi": band_hi,
                                 "power_law_r2": law["r2_against_referee_law"],
                                 "g4_replicates": anomaly["replicates"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
