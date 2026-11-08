"""M-E4 — data selection by gradient-sink alignment (queue E4). Wave 1.

W1-V asked whether the examples that damage the base model are identifiable IN ADVANCE from where
their gradients point. It ran on a recipe that barely damaged anything, so there was nothing to
select against. This is the retry on W1-U's damaging 400-step arm.

    THE PROCEDURE. Before any training, score every candidate example by ONE backward pass:
    what fraction of its gradient energy lands in the sink,

        s(x) = sum_cells ||g(x) U_core||^2 / sum_cells ||g(x)||^2

    then drop the top decile by that score and train on the rest.

    REFUTER (pre-logged): dropping the top decile by sink alignment retains no better than
    dropping a random decile of the same size.

FOUR DROP ARMS, ALL AT 90% OF THE DATA AND THE SAME STEP COUNT. This is the whole design, because
W1-V's own confound was that removing 10% of the data changes the data size:

    drop_core_top      the treatment
    drop_random        THE control. Same count, no selection signal at all.
    drop_core_bottom   the structured control: if the ORDER matters, dropping from the other end
                       must not help as much. A treatment that beats random but is matched by its
                       own reversal has found a size effect wearing a selection's name.
    drop_twin_top      the matched-random SUBSPACE control: same procedure, same top-decile rule,
                       scored against a haar32 bank instead of the sink. Separates "gradients that
                       point somewhere structured" from "gradients that point at the CORE".

`full` (100% of the data) is also run, as the reference for what the recipe does — not as the
control, since it differs in size from all four.

READ ON BOTH AXES, ALWAYS. Retention that arrives with a worse task loss is not a win, it is
training less. The arm reports task loss beside every retention number and gates on the pair.

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

ARM = "M-E4"
BANK = "core32"
TWIN = "haar32_s0"
SCORE_BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
               "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"


def main():
    ap = argparse.ArgumentParser(description="M-E4 — data selection by gradient-sink alignment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--drop-frac", type=float, default=0.10)
    ap.add_argument("--seeds", default="1337,1338,1339")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_e4", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

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
    for p in model.parameters():
        p.requires_grad_(False)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    for mod in mods.values():
        mod.weight.requires_grad_(True)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seeds[0])
    train_b = build_batches(processor, train_rows, "cuda")
    eval_b = build_batches(processor, eval_rows, "cuda")
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    U = {k: {b: torch.as_tensor(WT.load_bank(bank_dir, k[0], k[1], b), device="cuda",
                                dtype=torch.float32) for b in SCORE_BANKS} for k in mods}
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    print(f"[M-E4] base ppl={base_ppl:.5f}  {len(train_b)} candidate examples", flush=True)

    # ---- the scores, one backward per example, at the PRISTINE weights ------------------------
    # Scored at the untrained model on purpose: the selection has to be usable BEFORE training,
    # which is the only version of this that is a method rather than a post-hoc description.
    scores = {b: [] for b in SCORE_BANKS}
    for i, batch in enumerate(train_b):
        for m in mods.values():
            m.weight.grad = None
        model.train()
        model(input_ids=batch["input_ids"], labels=batch["labels"]).loss.backward()
        tot, inb = 0.0, {b: 0.0 for b in SCORE_BANKS}
        with torch.no_grad():
            for k, m in mods.items():
                g = m.weight.grad.to(torch.float32)
                tot += float((g * g).sum())
                for b in SCORE_BANKS:
                    inb[b] += float((g @ U[k][b]).pow(2).sum())
        assert tot > 0, f"example {i} has a zero gradient"
        for b in SCORE_BANKS:
            scores[b].append(inb[b] / tot)
        if (i + 1) % 50 == 0:
            print(f"    scored {i+1}/{len(train_b)}", flush=True)
    for m in mods.values():
        m.weight.grad = None
    scores = {b: np.asarray(v, dtype=np.float64) for b, v in scores.items()}
    for b in SCORE_BANKS:
        at.add_ci("sink_alignment_score", scores[b], bank_id=b, corpus_id=TASK_CORPUS,
                  model_id=WT.PG1_REPO, model_sha=base_sha,
                  flags={"definition": "||g U||^2 / ||g||^2 per example at pristine weights",
                         "n_examples": len(train_b)})
    print("[M-E4] score means: " + "  ".join(f"{b}={scores[b].mean():.4f}"
                                             for b in SCORE_BANKS), flush=True)

    n_drop = max(1, int(round(a.drop_frac * len(train_b))))
    order_core = np.argsort(scores[BANK])
    order_twin = np.argsort(scores[TWIN])
    keeps = {
        "full": list(range(len(train_b))),
        "drop_core_top": sorted(order_core[:-n_drop].tolist()),
        "drop_core_bottom": sorted(order_core[n_drop:].tolist()),
        "drop_twin_top": sorted(order_twin[:-n_drop].tolist()),
    }
    # The random control gets its own draw per seed, so "a random decile" is not one draw's luck.
    print(f"[M-E4] dropping {n_drop} of {len(train_b)} per drop arm", flush=True)

    def run(keep_idx, arm_name, seed):
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        rng = np.random.default_rng(H.cell_seed(seed, ARM, arm_name, "train"))
        model.train()
        for _step in range(a.steps):
            b = train_b[keep_idx[int(rng.integers(0, len(keep_idx)))]]
            opt.zero_grad()
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.backward()
            opt.step()
        model.eval()
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        task = eval_task_loss(model, eval_b)
        move_core, move_tot = 0.0, 0.0
        with torch.no_grad():
            for k, m in mods.items():
                dW = m.weight.detach().to(torch.float32) - pristine[k]
                move_tot += float((dW * dW).sum())
                move_core += float((dW @ U[k][BANK]).pow(2).sum())
        return {"arm": arm_name, "seed": seed, "n_kept": len(keep_idx), "ppl": ppl,
                "task_loss": task, "retention_loss": ppl - base_ppl,
                "core_motion_share": move_core / move_tot,
                "total_motion": float(np.sqrt(move_tot))}

    cells = {}
    for arm_name in list(keeps) + ["drop_random"]:
        for seed in seeds:
            if arm_name == "drop_random":
                r = np.random.default_rng(H.cell_seed(seed, ARM, "drop_random", "draw"))
                keep = sorted(r.permutation(len(train_b))[n_drop:].tolist())
            else:
                keep = keeps[arm_name]
            c = run(keep, arm_name, seed)
            cells[f"{arm_name}@s{seed}"] = c
            at.add("selection_ppl", value=c["ppl"], corpus_id=PPL_CORPUS, model_id=WT.PG1_REPO,
                   model_sha=base_sha,
                   flags={**c, "base_ppl": base_ppl, "steps": a.steps, "lr": a.lr,
                          "drop_frac": a.drop_frac, "optimizer": optinfo})
            at.add("selection_task_loss", value=c["task_loss"], corpus_id=TASK_CORPUS,
                   model_id=WT.PG1_REPO, model_sha=base_sha, flags=c)
            print(f"[M-E4] {arm_name:18s} s{seed} kept={c['n_kept']:3d} ppl={c['ppl']:8.4f} "
                  f"retention_loss={c['retention_loss']:+.4f} task={c['task_loss']:.4f} "
                  f"core_motion={c['core_motion_share']:.5f}", flush=True)

    at.check_readout_corpus(["selection_ppl", "selection_task_loss", "sink_alignment_score"])

    def mean(arm_name, key):
        return float(np.mean([cells[f"{arm_name}@s{s}"][key] for s in seeds]))

    def vals(arm_name, key):
        return [cells[f"{arm_name}@s{s}"][key] for s in seeds]

    arms = list(keeps) + ["drop_random"]
    summary = {n: {k: mean(n, k) for k in ("ppl", "task_loss", "retention_loss",
                                           "core_motion_share")} for n in arms}
    for n in arms:
        summary[n]["ppl_seeds"] = vals(n, "ppl")
        summary[n]["task_seeds"] = vals(n, "task_loss")

    # ---- the refuter and its two companions ---------------------------------------------------
    treat, ctrl = summary["drop_core_top"], summary["drop_random"]
    retains_better = bool(treat["ppl"] < min(vals("drop_random", "ppl")))
    task_not_worse = bool(treat["task_loss"] <= max(vals("drop_random", "task_loss")))
    gates.record("e4_core_selection_retains_better_than_random", retains_better,
                 core_top_ppl=treat["ppl"], random_ppl=ctrl["ppl"],
                 random_seeds=vals("drop_random", "ppl"), base_ppl=base_ppl,
                 note=("compared against the WHOLE random band, not its mean: three draws of a "
                       "random decile is the resolution this arm has"))
    gates.record("e4_task_loss_not_paid_for_it", task_not_worse,
                 core_top_task=treat["task_loss"], random_task=vals("drop_random", "task_loss"),
                 note=("retention bought with task performance is training less, not selecting "
                       "better; both axes gate"))
    gates.record("e4_order_matters", bool(treat["ppl"] < summary["drop_core_bottom"]["ppl"]),
                 top=treat["ppl"], bottom=summary["drop_core_bottom"]["ppl"],
                 note=("dropping from the other end of the same score; a treatment matched by its "
                       "own reversal has found a size effect"))
    gates.record("e4_subspace_matters", bool(treat["ppl"] < summary["drop_twin_top"]["ppl"]),
                 core=treat["ppl"], twin=summary["drop_twin_top"]["ppl"], twin_bank=TWIN,
                 note=("same procedure scored against a matched-random subspace: separates "
                       "'structured gradients' from 'the CORE'"))
    usable = bool(retains_better and task_not_worse)
    # The arm's headline gate is a conjunction; recording only the conjunction leaves the artifact
    # unable to say which half failed (critic check C9).
    gates.record("e4_selection_is_usable", usable,
                 retains_better=bool(retains_better), task_not_worse=bool(task_not_worse),
                 note=("the practical claim needs both: better retention AND no worse task "
                       "performance than the random-drop control"))
    s = M.surprise(treat["ppl"], ctrl["ppl"], vals("drop_random", "ppl"))
    sur.add(arm=ARM, expected_ref=expected_ref, cell="drop_core_top_vs_random",
            statistic="selection_ppl/core32", s=s)

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "types": list(types), "layers": layers, "steps": a.steps, "lr": a.lr,
                  "n_candidates": len(train_b), "n_dropped": n_drop, "drop_frac": a.drop_frac,
                  "seeds": seeds, "optimizer": optinfo, "base_ppl": base_ppl,
                  "score_summary": {b: {"mean": float(scores[b].mean()),
                                        "std": float(scores[b].std()),
                                        "max": float(scores[b].max())} for b in SCORE_BANKS},
                  "cells": cells, "summary": summary,
                  "retains_better_than_random": retains_better,
                  "task_loss_not_paid": task_not_worse, "selection_is_usable": usable})
    print("E4_OK " + json.dumps({"rows": info["n_rows"],
                                 "ppl": {n: summary[n]["ppl"] for n in arms},
                                 "usable": usable}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
