"""MENDEL MN-16b — do REALIZED v-seeding horizons match the predicted ones? (queue item 22)

THIS CANNOT ADJUDICATE MN-16 AND DOES NOT TRY. The arm's three frozen verdicts are BEATS-BOTH,
BEATS-EWC-ONLY and NO-BETTER-THAN-REPLAY — every one of them names EWC-lite or replay, and neither
method nor a protection benchmark exists in this repo. So the arm stays BLOCKED-WITH-UNLOCK and
this job writes no verdict token.

What it CAN settle is the arm's SECOND kill clause, which is separable and cheap:

    KILL: no better than replay, OR **realized horizons miss predicted ones**

The offline half derived T_p = ln(D-1) / (-ln beta2) and checked it against a SIMULATION of Adam
(worst relative error 0.6%). A simulation of Adam is not Adam on a real model: there g^2 is
non-stationary and correlated across steps, the beta2 equilibrium drifts as the loss falls, and
every other parameter is moving at the same time. Whether the law survives that is exactly the
question the second clause asks, and it is answerable with ONE run and no replay baseline. If it
fails, MN-16 is dead without anyone building EWC-lite.

THE INTERVENTION. After a warmup that brings v_hat to equilibrium, a random half of each tracked
matrix's INPUT COLUMNS has its Adam second-moment state multiplied by the dose D. The other half
is the control, and it is the SAME matrix at the SAME step under the SAME gradients — the tightest
matched control available, because the only thing that differs between the two column sets is the
seed itself.

THE PREDICTION, written out so the comparison is not a curve-fit. Seeding leaves an excess second
moment that decays geometrically,  v_t = v_eq * (1 + (D-1) beta2^t),  so the per-step update, which
goes as 1/sqrt(v_hat), is suppressed by

    rho(t) = 1 / sqrt(1 + (D-1) * beta2^t)

and T_p is exactly where (D-1) beta2^t = 1, i.e. where rho = 1/sqrt(2). The realized horizon is
read off the measured rho(t) at that same crossing, so predicted and realized are the same
quantity measured two ways rather than two quantities compared by eye.

GATE: |T_realized - T_p| / T_p <= 0.25 at every dose. Recorded whatever it says.

EPSILON IS CHECKED, NOT ASSUMED. Adam's update is m/(sqrt(v) + eps); if sqrt(v_eq) is not large
against eps the suppression law does not hold at all and the measurement would be of eps, not of
seeding. The job asserts the margin rather than discovering it in the numbers.

GPU. FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP

ARM = "MN-16"
BETA2 = 0.99
BETA1 = 0.9
DOSES = (4, 10, 100)
TOL = 0.25
EPS_MARGIN = 100.0
TRACKED = (("q_proj", 0), ("q_proj", 8), ("q_proj", 17),
           ("v_proj", 0), ("v_proj", 8), ("v_proj", 17))


def predicted_horizon(dose: int, beta2: float = BETA2) -> float:
    """T_p = ln(D - 1) / (-ln beta2) — the offline half's law, restated here as one line so this
    job does not depend on a number transcribed from its metrics."""
    assert dose > 1, "a dose of 1 is no seed at all and has no horizon"
    return float(np.log(dose - 1.0) / (-np.log(beta2)))


def predicted_rho(t, dose: int, beta2: float = BETA2):
    """Suppression of the per-step update at step t after a dose-D seed."""
    return 1.0 / np.sqrt(1.0 + (dose - 1.0) * beta2 ** np.asarray(t, dtype=np.float64))


def realized_horizon(rho, target: float = 1.0 / np.sqrt(2.0)):
    """First step at which the measured suppression recovers to rho = 1/sqrt(2).

    Linearly interpolated between the bracketing steps rather than rounded to the step index: at
    dose 4 the predicted horizon is ~109 steps and a whole-step answer would carry a 1% grid error
    into a 25% tolerance, which is fine, but at short horizons it would not be.
    """
    r = np.asarray(rho, dtype=np.float64)
    hit = np.nonzero(r >= target)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return 0.0
    lo, hi = r[i - 1], r[i]
    return float(i - 1 + (target - lo) / (hi - lo)) if hi > lo else float(i)


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-16b — realized vs predicted horizons")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()

    # FIRST, before any transformers/datasets import. run_job.sh exports HF_HUB_OFFLINE=1, under
    # which HfApi().model_info raises OfflineModeIsEnabled and load_dataset cannot reach the hub.
    # I reasoned past this because G4b makes the same calls and ran — but G4b ran against a
    # different huggingface_hub build, and "a sibling job did it" is not evidence about this one.
    # It cost MN-7 a launch.
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    from infra.aws_pg.jobs.mendel_mn7_span import _train_windows, trainable_sites

    assert torch.cuda.is_available(), "no CUDA — MN-16b trains"
    device = "cuda"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn16b", wave=meta["wave"],
                  queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn16b"), out)
    man.write()

    base_sha = HfApi().model_info(WT.PG1_REPO).sha
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        WT.PG1_REPO, revision=base_sha, torch_dtype=torch.float32,
        device_map=device, attn_implementation="eager")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)
    sites = trainable_sites(model)
    tracked = {k: sites[k] for k in TRACKED}
    texts = _train_windows(processor, a.warmup + a.steps)
    mods = dict(model.named_modules())
    img_id = model.config.image_token_index
    rng = np.random.default_rng(a.seed)

    def batch(i):
        inp = processor(text=[texts[i]], images=[blank], return_tensors="pt",
                        padding=False).to(device)
        labels = inp["input_ids"].clone()
        labels[labels == img_id] = -100
        return inp, labels

    def step(opt, i):
        inp, labels = batch(i)
        o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(model.dtype), labels=labels)
        opt.zero_grad(set_to_none=True)
        o.loss.backward()
        opt.step()
        return float(o.loss)

    for p in model.parameters():
        p.requires_grad_(False)
    params = [mods[n].weight for n in sites.values()]
    for p in params:
        p.requires_grad_(True)

    # ---- warmup: bring v_hat to its equilibrium before any seed is applied ---------------------
    model.train()
    opt = torch.optim.Adam(params, lr=a.lr, betas=(BETA1, BETA2))
    for i in range(a.warmup):
        loss = step(opt, i)
        if (i + 1) % 40 == 0:
            print(f"[mn16b] warmup {i + 1}/{a.warmup} loss {loss:.4f}", flush=True)

    warm_w = {k: mods[n].weight.data.clone() for k, n in sites.items()}
    warm_state = {k: {kk: v.clone() for kk, v in opt.state[mods[n].weight].items()
                      if torch.is_tensor(v)} for k, n in sites.items()}
    warm_steps = {k: opt.state[mods[n].weight]["step"] for k, n in sites.items()}

    v_eq = {k: float(warm_state[k]["exp_avg_sq"].mean()) for k in tracked}
    eps = 1e-8
    margin = min(float(np.sqrt(v)) / eps for v in v_eq.values())
    gates.record("mn16b_epsilon_is_negligible_against_the_equilibrium_second_moment",
                 margin >= EPS_MARGIN, sqrt_v_eq_over_eps=margin, required=EPS_MARGIN,
                 v_eq={f"{t}_L{L}": v for (t, L), v in v_eq.items()},
                 note=("Adam's update is m/(sqrt(v)+eps); if this margin is small the measurement "
                       "is of eps rather than of seeding and the suppression law does not apply"))
    assert margin >= EPS_MARGIN, (
        f"sqrt(v_eq)/eps = {margin:.1f} < {EPS_MARGIN} — the suppression law does not hold here "
        f"and a horizon measured under it would be meaningless")
    print(f"[mn16b] warmup done, sqrt(v_eq)/eps margin {margin:.0f}", flush=True)

    # the seeded column set: a random half of each tracked matrix's INPUT columns
    seeded = {k: rng.permutation(warm_w[k].shape[1])[: warm_w[k].shape[1] // 2] for k in tracked}

    cells, curves = [], {}
    for dose in DOSES:
        for k, n in sites.items():                      # restore weights AND optimizer state
            mods[n].weight.data.copy_(warm_w[k])
        opt = torch.optim.Adam(params, lr=a.lr, betas=(BETA1, BETA2))
        for k, n in sites.items():
            st = opt.state[mods[n].weight]
            for kk, v in warm_state[k].items():
                st[kk] = v.clone()
            st["step"] = warm_steps[k].clone() if torch.is_tensor(warm_steps[k]) else warm_steps[k]
        for k in tracked:                               # THE INTERVENTION
            S = torch.as_tensor(seeded[k], device=device)
            opt.state[mods[sites[k]].weight]["exp_avg_sq"][:, S] *= float(dose)

        rho = {k: [] for k in tracked}
        for i in range(a.steps):
            before = {k: mods[sites[k]].weight.data.clone() for k in tracked}
            loss = step(opt, a.warmup + i)
            for k in tracked:
                d = (mods[sites[k]].weight.data - before[k]).abs()
                S = torch.as_tensor(seeded[k], device=device)
                mask = torch.ones(d.shape[1], dtype=torch.bool, device=device)
                mask[S] = False
                num = float(d[:, S].pow(2).mean().sqrt())
                den = float(d[:, mask].pow(2).mean().sqrt())
                rho[k].append(num / den if den > 0 else float("nan"))
            if (i + 1) % 100 == 0:
                print(f"[mn16b] D={dose} {i + 1}/{a.steps} loss {loss:.4f} "
                      f"rho {np.mean([rho[k][-1] for k in tracked]):.4f}", flush=True)

        tp = predicted_horizon(dose)
        for k in tracked:
            # smoothed over 11 steps: rho is a ratio of two per-step RMS values and is noisy at the
            # single-step level, while the quantity the law describes is its trend
            sm = np.convolve(np.asarray(rho[k]), np.ones(11) / 11.0, mode="valid")
            tr = realized_horizon(sm)
            cells.append({"type": k[0], "layer": k[1], "dose": dose,
                          "predicted_horizon": tp, "realized_horizon": tr,
                          "rel_error": (abs(tr - tp) / tp if tr is not None else None),
                          "rho_at_1": float(sm[0]), "rho_at_end": float(sm[-1]),
                          "predicted_rho_at_1": float(predicted_rho(1, dose)),
                          "n_seeded_columns": int(seeded[k].size)})
            at.add("seeding_horizon", value=(tr if tr is not None else float("nan")),
                   layer=k[1], type=k[0], bank_id=f"D{dose}", control_id=MC.TREATMENT,
                   flags={"dose": dose, "predicted": tp, "beta2": BETA2})
            at.add("seeding_horizon", value=tp, layer=k[1], type=k[0], bank_id=f"D{dose}",
                   control_id=MC.STRUCTURED,
                   flags={"what": "ln(D-1)/(-ln beta2), the offline half's law"})
        curves[str(dose)] = {f"{k[0]}_L{k[1]}": [round(x, 5) for x in rho[k][::10]]
                             for k in tracked}
        got = [c["realized_horizon"] for c in cells if c["dose"] == dose]
        print(f"[mn16b] D={dose}: predicted {tp:.1f}, realized "
              f"{[None if g is None else round(g, 1) for g in got]}", flush=True)

    ok = [c for c in cells if c["rel_error"] is not None and c["rel_error"] <= TOL]
    censored = [c for c in cells if c["realized_horizon"] is None]
    matches = len(ok) == len(cells)
    gates.record("mn16b_realized_horizons_match_predicted", matches,
                 n_within_tol=len(ok), n_cells=len(cells), n_censored=len(censored), tol=TOL,
                 by_dose={str(d): float(np.median([c["rel_error"] for c in cells
                                                   if c["dose"] == d
                                                   and c["rel_error"] is not None]))
                          for d in DOSES})
    at.check_control_trinity([])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": "BLOCKED-WITH-UNLOCK", "queue_item": meta["item"],
        "half_completed": ("the SECOND kill clause — realized vs predicted seeding horizons on a "
                           "real fine-tune, at a dose ladder spanning a decade"),
        "why_no_verdict_token": (
            "MN-16's three frozen verdicts (BEATS-BOTH, BEATS-EWC-ONLY, NO-BETTER-THAN-REPLAY) "
            "every one name EWC-lite or replay. Neither exists in this repo, so no token is "
            "claimable and none is claimed. The FIRST kill clause is untouched."),
        "beta2": BETA2, "doses": list(DOSES), "tol": TOL, "lr": a.lr,
        "warmup_steps": a.warmup, "measured_steps": a.steps,
        "sqrt_v_eq_over_eps": margin, "v_eq": {f"{t}_L{L}": v for (t, L), v in v_eq.items()},
        "horizons_match": bool(matches), "n_within_tol": len(ok), "n_cells": len(cells),
        "n_censored": len(censored),
        "cells": cells, "rho_curves_every_10_steps": curves, "atlas": info})
    print("MN16B_OK " + json.dumps({
        "horizons_match": bool(matches), "n_within_tol": len(ok), "n_cells": len(cells),
        "by_dose": {str(d): [None if c["realized_horizon"] is None
                             else round(c["realized_horizon"], 1)
                             for c in cells if c["dose"] == d] for d in DOSES}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
