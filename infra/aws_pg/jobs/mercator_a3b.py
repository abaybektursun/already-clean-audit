"""M-A3 stage 2 — the denominator interventions (queue A3). Wave 1.

Stage 1 measured the mechanism and predicted from it. Stage 2 INTERVENES on the one term the
prediction rests on — Adam's per-coordinate denominator — and checks whether the predicted change
in realized core motion actually happens.

    THE CLAIM UNDER TEST. The sink is made by sqrt(v_hat) being LARGER in core coordinates than
    elsewhere, so the same gradient buys less motion there. If that is the mechanism, then two
    edits should each release the core, and for the same reason:

        scalar denominator   replace sqrt(v_hat) by its mean over the matrix — the per-coordinate
                             structure is gone, so no coordinate can be singled out by it
        large epsilon        as eps grows past sqrt(v_hat), the denominator becomes eps everywhere,
                             which is the same thing reached from the other direction

    and the eps ladder is what turns that into a DOSE rather than a demonstration: suppression
    should fall monotonically as eps climbs through the range where it overtakes sqrt(v_hat), and
    stage 1 measured where that range is.

    REFUTER (pre-logged): core suppression is unchanged by the scalar denominator AND flat across
    the eps ladder — i.e. the diagnostic separates the sink but intervening on it changes nothing,
    which would make the settling ratio a correlate rather than the cause.

THE STATISTIC IS THE SAME OBJECT M-Q STAGE 2 USES — gradient share in, motion share out, and the
delivery ratio between them, per bank, under the full eight-bank battery. Deliberately identical so
the two arms' answers can be read on one axis: M-Q changes the BASIS the denominator is computed
in, M-A3 changes the denominator's SHAPE. Both are edits to the same term.

TWO PREFLIGHT GATES, BOTH ON THE INSTANCE:

  * the scalar-denominator rule must reproduce ExactAdam EXACTLY on synthetic gradients whose
    second moment is uniform. That identity is the whole reason the swap is controlled, and an
    implementation that failed it would attribute its own bug to the mechanism.
  * eps must actually bracket sqrt(v_hat). An eps ladder that sits entirely below the second
    moment cannot move anything, and an arm that reported "flat across eps" from such a ladder
    would be reporting its own grid.

§0.2.7 plus amendment A1: optimizer-mechanics arm, so both rules are exact fp32 implementations
that can be read, not library calls with a flag.

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
from infra.pg import mercator_train as TR

ARM = "M-A3"
BANK = "core32"
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"
# The eps ladder. 1e-8 is the incumbent; the rest climb through and past where stage 1 measured
# sqrt(v_hat) to sit, so the ladder brackets the crossover instead of sampling one side of it.
EPS_LADDER = (1e-8, 1e-6, 1e-4, 1e-2)


def identity_check(device, *, n: int = 64, steps: int = 5, lr: float = 1e-3) -> dict:
    """ExactAdam and ScalarDenomAdam must agree EXACTLY when the second moment is uniform.

    Constructed by giving every coordinate the same gradient magnitude with independent signs: v is
    then identical in every coordinate at every step, so mean(sqrt(v_hat)) IS sqrt(v_hat) and the
    two rules are the same rule. Any disagreement is an implementation error, and finding it here
    rather than in the science is the point.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(20260815)
    signs = [torch.where(torch.rand(n, n, generator=g) < 0.5, -1.0, 1.0).to(device)
             for _ in range(steps)]
    out = []
    for cls in (T.ExactAdam, TR.ScalarDenomAdam):
        w = torch.zeros(n, n, device=device, dtype=torch.float32)
        p = torch.nn.Parameter(w.clone())
        opt = cls([p], lr=lr)
        for s in signs:
            p.grad = s.clone()
            opt.step()
        out.append(p.detach().to(torch.float32).cpu().numpy())
    d = float(np.abs(out[0] - out[1]).max())
    return {"max_abs_diff": d, "n": n, "steps": steps,
            "scale": float(np.abs(out[0]).max()),
            "relative": d / max(float(np.abs(out[0]).max()), 1e-30)}


def main():
    ap = argparse.ArgumentParser(description="M-A3 stage 2 — the denominator interventions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps-ladder", default=",".join(f"{e:g}" for e in EPS_LADDER))
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps_ladder = [float(x) for x in a.eps_ladder.split(",") if x.strip()]
    M.check_dose_grid(eps_ladder, arm=ARM)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_a3_stage2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam + mercator_train.ScalarDenomAdam",
                                offload_exact_fp32=False)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows

    assert torch.cuda.is_available(), "no CUDA"
    ident = identity_check("cuda")
    at.add("scalar_denom_identity_maxabs", value=ident["max_abs_diff"], flags=ident)
    assert ident["relative"] < 1e-6, (
        f"ScalarDenomAdam does not reproduce ExactAdam on a uniform second moment: "
        f"max|Δw| = {ident['max_abs_diff']:.3e} (relative {ident['relative']:.3e}). The swap is "
        f"only controlled if the two rules coincide there, so this is an implementation error, "
        f"not a finding")
    print(f"[M-A3b] scalar-denominator identity holds to {ident['relative']:.2e} relative",
          flush=True)

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
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, a.seed)
    train_b = build_batches(processor, train_rows, "cuda")
    eval_b = build_batches(processor, eval_rows, "cuda")
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    U = {k: {b: torch.as_tensor(WT.load_bank(bank_dir, k[0], k[1], b), device="cuda",
                                dtype=torch.float32) for b in FULL_BATTERY} for k in mods}
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    print(f"[M-A3b] base ppl={base_ppl:.5f} over {len(mods)} cells", flush=True)

    def run(rule, eps):
        """One fine-tune under one update rule, returning the per-bank in/out shares."""
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        cls = T.ExactAdam if rule == "adam" else TR.ScalarDenomAdam
        opt = cls([m.weight for m in mods.values()], lr=a.lr, eps=eps)
        rng = np.random.default_rng(H.cell_seed(a.seed, ARM, rule, f"{eps:g}"))
        grad_in = {b: 0.0 for b in FULL_BATTERY}
        grad_tot, sqrt_v = 0.0, []
        model.train()
        for _step in range(a.steps):
            batch = train_b[int(rng.integers(0, len(train_b)))]
            opt.zero_grad()
            model(input_ids=batch["input_ids"], labels=batch["labels"]).loss.backward()
            with torch.no_grad():
                for k, m in mods.items():
                    g = m.weight.grad.to(torch.float32)
                    grad_tot += float((g * g).sum())
                    for b in FULL_BATTERY:
                        grad_in[b] += float((g @ U[k][b]).pow(2).sum())
            opt.step()
        # WHERE THE DENOMINATOR ACTUALLY SITS, measured rather than assumed: the eps ladder only
        # means something relative to this.
        with torch.no_grad():
            bc2 = 1.0 - opt.b2 ** opt.t
            for v in opt.v:
                sqrt_v.append(float((v / bc2).sqrt().mean()))
        model.eval()
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        task = eval_task_loss(model, eval_b)
        move_in, move_tot = {b: 0.0 for b in FULL_BATTERY}, 0.0
        with torch.no_grad():
            for k, m in mods.items():
                dW = m.weight.detach().to(torch.float32) - pristine[k]
                move_tot += float((dW * dW).sum())
                for b in FULL_BATTERY:
                    move_in[b] += float((dW @ U[k][b]).pow(2).sum())
        assert move_tot > 0 and grad_tot > 0, f"{rule}/eps={eps}: nothing moved"
        return {"rule": rule, "eps": eps, "ppl": ppl, "task_loss": task,
                "total_motion": float(np.sqrt(move_tot)),
                "mean_sqrt_v": float(np.mean(sqrt_v)),
                "eps_over_sqrt_v": eps / float(np.mean(sqrt_v)),
                "shares": {b: {"grad": grad_in[b] / grad_tot, "motion": move_in[b] / move_tot}
                           for b in FULL_BATTERY}}

    cells, rows = {}, []
    for rule in ("adam", "scalar"):
        for eps in eps_ladder:
            # The scalar rule's own eps ladder would confound two edits at once; it runs at the
            # incumbent eps only, and the eps ladder is Adam's.
            if rule == "scalar" and eps != eps_ladder[0]:
                continue
            c = run(rule, eps)
            cells[f"{rule}@{eps:g}"] = c
            for b in FULL_BATTERY:
                sh = c["shares"][b]
                row = {"rule": rule, "eps": eps, "bank": b, "grad_share": sh["grad"],
                       "motion_share": sh["motion"],
                       "delivery_ratio": sh["motion"] / sh["grad"], "ppl": c["ppl"],
                       "task_loss": c["task_loss"], "mean_sqrt_v": c["mean_sqrt_v"]}
                rows.append(row)
                at.add("denom_delivery_ratio", value=row["delivery_ratio"], dose=eps, bank_id=b,
                       model_id=WT.PG1_REPO, model_sha=base_sha,
                       flags={**row, "steps": a.steps, "lr": a.lr, "optimizer": optinfo,
                              "definition": ("(||dW U||^2/||dW||^2) / (sum_t ||g_t U||^2 / "
                                             "sum_t ||g_t||^2); < 1 is suppression")})
                at.add("denom_motion_share", value=sh["motion"], dose=eps, bank_id=b,
                       model_id=WT.PG1_REPO, model_sha=base_sha, flags=row)
                at.add("denom_grad_share", value=sh["grad"], dose=eps, bank_id=b,
                       model_id=WT.PG1_REPO, model_sha=base_sha, flags=row)
            at.add("denom_ppl", value=c["ppl"], dose=eps, corpus_id=PPL_CORPUS,
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"rule": rule, "eps": eps, "base_ppl": base_ppl})
            at.add("denom_task_loss", value=c["task_loss"], dose=eps, corpus_id=TASK_CORPUS,
                   model_id=WT.PG1_REPO, model_sha=base_sha, flags={"rule": rule, "eps": eps})
            print(f"[M-A3b] {rule:7s} eps={eps:g}: core delivery="
                  f"{c['shares'][BANK]['motion'] / c['shares'][BANK]['grad']:.4f} "
                  f"eps/sqrt(v)={c['eps_over_sqrt_v']:.3g} ppl={c['ppl']:.3f}", flush=True)

    at.check_controls_per_statistic(["denom_delivery_ratio", "denom_motion_share",
                                     "denom_grad_share"])
    at.check_readout_corpus(["denom_ppl", "denom_task_loss"])

    # ---- the two gates -----------------------------------------------------------------------
    delivery = {k: v["shares"][BANK]["motion"] / v["shares"][BANK]["grad"]
                for k, v in cells.items()}
    incumbent = f"adam@{eps_ladder[0]:g}"
    scalar_key = f"scalar@{eps_ladder[0]:g}"
    # THE LADDER'S OWN VALIDITY. eps only competes with the denominator once it is comparable to
    # sqrt(v_hat); a ladder entirely below it could only ever read flat, and that flatness would be
    # a fact about the grid.
    ratios = [cells[f"adam@{e:g}"]["eps_over_sqrt_v"] for e in eps_ladder]
    ladder_brackets = bool(min(ratios) < 1.0 < max(ratios))
    gates.record("a3b_eps_ladder_brackets_sqrt_v", ladder_brackets,
                 eps_over_sqrt_v=dict(zip([f"{e:g}" for e in eps_ladder], ratios)),
                 outcome=None if ladder_brackets else M.VOID_INSTRUMENT,
                 note=("without a crossover the ladder cannot move the denominator and a flat "
                       "reading would be about the grid, not about Adam"))
    twin_delivery = {k: {b: v["shares"][b]["motion"] / v["shares"][b]["grad"]
                         for b in FULL_BATTERY} for k, v in cells.items()}
    haar = [twin_delivery[incumbent][b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    gates.record("a3b_core_suppressed_under_incumbent",
                 delivery[incumbent] < min(haar), core=delivery[incumbent], haar_band=haar,
                 note=("the premise: at the incumbent eps the core must actually be suppressed "
                       "relative to matched-random subspaces, or there is nothing to release"))
    released_scalar = bool(delivery[scalar_key] > delivery[incumbent])
    gates.record("a3b_scalar_denominator_releases_core", released_scalar,
                 scalar=delivery[scalar_key], adam=delivery[incumbent],
                 ratio=delivery[scalar_key] / delivery[incumbent],
                 twin_change={b: twin_delivery[scalar_key][b] / twin_delivery[incumbent][b]
                              for b in FULL_BATTERY},
                 note=("twin_change is the same ratio for every bank: a rule that raises delivery "
                       "everywhere has not released the CORE, it has changed the step size"))
    eps_series = [delivery[f"adam@{e:g}"] for e in eps_ladder]
    monotone = bool(all(y >= x for x, y in zip(eps_series, eps_series[1:])))
    gates.record("a3b_delivery_rises_with_eps", monotone,
                 eps=[f"{e:g}" for e in eps_ladder], delivery=eps_series,
                 note=("the dose leg: if the denominator is what suppresses the core, drowning it "
                       "in eps should release the core monotonically"))
    refuted = bool(not released_scalar and not monotone)
    # A ROLLUP MUST CARRY ITS INPUTS. `refuted` is a conjunction of two other gates' booleans, and
    # recording only the conjunction means the artifact cannot say WHICH leg drove the verdict —
    # a reader has to re-derive it from neighbouring gates and trust that they are the right two
    # (critic check C9).
    gates.record("a3b_refuter_fires", refuted,
                 released_scalar=bool(released_scalar), delivery_rises_with_eps=monotone,
                 note=("the settling ratio separates the sink but intervening on it changes "
                       "nothing, which would make it a correlate rather than the cause; "
                       "refuted = NOT released_scalar AND NOT delivery_rises_with_eps"))
    s = M.surprise(delivery[scalar_key], delivery[incumbent],
                   [twin_delivery[incumbent][b]
                    for b in ("haar32_s0", "haar32_s1", "haar32_s2")])
    sur.add(arm=ARM, expected_ref=expected_ref, cell="scalar_vs_adam",
            statistic="denom_delivery_ratio/core32", s=s)

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_stage2_result.json",
                 {"arm": ARM, "stage": "2 (denominator interventions)",
                  "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "eps_ladder": eps_ladder, "types": list(types), "layers": layers,
                  "steps": a.steps, "lr": a.lr, "optimizer": optinfo, "base_ppl": base_ppl,
                  "identity_check": ident, "cells": cells, "rows": rows,
                  "core_delivery": delivery, "twin_delivery": twin_delivery,
                  "eps_over_sqrt_v": dict(zip([f"{e:g}" for e in eps_ladder], ratios)),
                  "scalar_releases_core": released_scalar, "delivery_monotone_in_eps": monotone,
                  "refuter_fires": refuted})
    print("A3B_OK " + json.dumps({"rows": info["n_rows"], "core_delivery": delivery,
                                  "scalar_releases": released_scalar, "monotone": monotone}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
