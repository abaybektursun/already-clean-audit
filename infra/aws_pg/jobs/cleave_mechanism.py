"""CLEAVE MECHANISM (GPU) — where does the task gradient actually live? Spec §3, T4.

STENCIL *inferred* "high behavioral sensitivity ≠ task-learning leverage" from a behavioral ordering.
CLEAVE measures it directly: take the task GRADIENT at the frozen base model — the direction
fine-tuning first moves — and ask what fraction of its energy each policy subspace captures on the
residual-facing side (`comwrite_core.energy_fraction`), against the Haar chance floor r/d.

Two objects, both censused per (layer, matrix-type):
  * **the task gradient at base** — ∂(task CE)/∂W accumulated over the subject's train examples
    through the PG1-correct blank-image forward. This is the mechanism read.
  * **the natural ΔW** — the accumulated full-FT update. This is the decomposition read: does the
    update's mass sit where the map says damage lives?

The unifying prediction (spec §1.3, a **hypothesis-to-kill**) is `energy > susceptibility` — learning
lives in the high-variance directions, forgetting in the susceptible ones. It is elegant and it would
unify SPECTER+CARRY+STENCIL, which is exactly why it is registered to be killed: every fraction is
reported against the chance floor, and `census_contrast` is signed so a landing either way is legible.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_mechanism \
      --args "--banks-s3 results/cleave_qcert/cleave_banks.pt --ranks 20,50,150" \
      --type g6e.4xlarge,g6e.8xlarge,p5.4xlarge --s3-prefix results/cleave_mech --budget-min 240
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
    ap.add_argument("--seeds", default="1337")
    ap.add_argument("--ranks", default="20,50,150")
    ap.add_argument("--grad-batches", type=int, default=8)
    ap.add_argument("--with-delta", action="store_true",
                    help="also census the natural ΔW (trains a full-FT per cell)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from PIL import Image

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import cleave_core as CL
    from infra.pg import relearn as RL

    CLS.install_pg()
    device = "cuda"
    ranks = _ints(a.ranks)
    seeds = _ints(a.seeds)
    rep = {"object": "cleave_mechanism_v1", "ranks": ranks, "seeds": seeds, "gpu_seconds": {},
           "prediction_under_test": ("unifying hypothesis-to-kill: energy > susceptibility "
                                     "(task gradient lives in the high-variance directions)")}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("banks"):
        banks = torch.load(CLS.fetch_s3(a.banks_s3, "/tmp/cleave/cleave_banks.pt"),
                           map_location="cpu", weights_only=False)
    assert set(banks) == set(range(CLS.N_LAYERS)), "banks must cover all 18 layers"
    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = CLS.load_model("/tmp/cleave")
    rep["pinned_sha"] = sha
    mmlu = CLS.mmlu_by_subject()
    roster = CLS.build_roster(mmlu)
    subjects = ([s for s in a.subject_list.split(",") if s.strip()] if a.subject_list
                else roster["discovery"])
    subjects = [s for s in subjects if CLS.splittable(mmlu[s]["test"])]
    assert subjects, "no splittable subjects"
    rep["subjects"] = subjects
    snap = CLS.snapshot_base(model)
    lm, paths, vt_orig = CLS.lm_bundle(model)
    _w()

    def task_gradient(examples, seed):
        """∂(task CE)/∂W at the CURRENT weights, accumulated over `grad_batches` batches through the
        PG1-correct blank-image forward (identical to `train_with_trajectory`'s, A-STENCIL-8) —
        the same loss the adaptation optimises, so this IS the task's search direction."""
        tok = proc.tokenizer
        letters = RL.letter_token_ids(tok)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        orig_side = tok.padding_side
        tok.padding_side = "left"
        gray = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        img_dtype = next(model.parameters()).dtype
        prompts = [e["prompt"] for e in examples]
        targets = [letters[e["answer_idx"]] for e in examples]

        CL.mark_grid_trainable(model)          # A-CLEAVE-7: the decomposable grid only
        model.zero_grad(set_to_none=True)
        g = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(examples), generator=g).tolist()
        mb, nb = CLS.GRAD_MICRO_BATCH, 0     # A-CLEAVE-8: memory only, same accumulated gradient
        for s in range(0, min(len(order), a.grad_batches * mb), mb):
            idx = order[s:s + mb]
            if not idx:
                break
            inp = proc(text=[prompts[i] for i in idx], images=[gray] * len(idx),
                       return_tensors="pt", padding="longest").to(device)
            inp["pixel_values"] = inp["pixel_values"].to(img_dtype)
            tgt = torch.tensor([targets[i] for i in idx], device=device)
            last = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"], logits_to_keep=1).logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(last.float(), tgt)
            assert torch.isfinite(loss), "non-finite task loss — HALT"
            loss.backward()
            nb += 1
        tok.padding_side = orig_side
        grads = {n: p.grad.detach().float().cpu().clone()
                 for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        assert grads, "no gradients — the trainable set is empty"
        return grads, nb

    def census_matrices(named_mats, tag):
        """Census a {param_name: matrix} map over the 18×7 grid at every rank → per-(layer,type)
        fractions + the pooled, Frobenius-energy-WEIGHTED mean per policy. Weighting by ‖M‖² is the
        honest pooling: an unweighted mean over 126 matrices lets a tiny-norm matrix count as much
        as the one carrying the update."""
        from infra.pg.gauge_common import layer_index, matrix_type
        rows, pooled = [], {}
        for r in ranks:
            acc = {p: 0.0 for p in CL.POLICIES}
            wsum, floor = 0.0, None
            for n, M in named_mats.items():
                L, t = layer_index(n), matrix_type(n)
                Mn = M.numpy() if hasattr(M, "numpy") else M
                fro2 = float((Mn ** 2).sum())
                if fro2 <= 0:
                    continue
                cen = CL.mechanism_census(Mn, banks[L], r, t)
                floor = cen["chance_floor"]
                rows.append({"object": tag, "rank": r, "layer": L, "type": t, "fro2": fro2,
                             **{p: cen[p] for p in CL.POLICIES}})
                for p in CL.POLICIES:
                    acc[p] += cen[p] * fro2
                wsum += fro2
            assert wsum > 0, f"{tag}: all matrices have zero energy"
            pooled[r] = {p: acc[p] / wsum for p in CL.POLICIES}
            pooled[r]["chance_floor"] = floor
            pooled[r]["contrast_energy_minus_sus"] = pooled[r]["energy"] - pooled[r]["susceptibility"]
            print(f"[mech] {tag} r={r}: sus={pooled[r]['susceptibility']:.4f} "
                  f"energy={pooled[r]['energy']:.4f} random={pooled[r]['random']:.4f} "
                  f"floor={floor:.4f} contrast={pooled[r]['contrast_energy_minus_sus']:+.4f}", flush=True)
        return rows, pooled

    grid = CL._target_grid(model)               # {param_name: (L, mtype)} over the 126 matrices
    assert len(grid) == CLS.N_LAYERS * 7, f"grid {len(grid)} != 126"
    assert all(n in snap for n in grid), "a grid param is absent from the base snapshot"
    rep["cells"] = []
    cells = rep["cells"]
    for subj in subjects:
        train_rows, _evl, ev_hashes = CLS.subject_train_eval(mmlu[subj]["test"])
        examples, _dedup = CLS.train_examples(train_rows, ev_hashes)
        for seed in seeds:
            CLS.reset_model(model, snap)
            # ---- 1. the task GRADIENT at base (the mechanism read) ------------------------
            with tm.stage(f"{subj}/s{seed}/grad"):
                grads, nb = task_gradient(examples, seed)
            grid_grads = {n: g for n, g in grads.items() if n in grid}
            assert len(grid_grads) == CLS.N_LAYERS * 7, \
                f"grid gradients {len(grid_grads)} != 126 — the trainable set does not cover the grid"
            with tm.stage(f"{subj}/s{seed}/grad_census"):
                g_rows, g_pooled = census_matrices(grid_grads, "task_gradient")
            cell = {"subject": subj, "seed": seed, "n_grad_batches": nb,
                    "gradient_pooled": g_pooled}
            del grads, grid_grads
            CLS.reset_model(model, snap)

            # ---- 2. the natural ΔW (the decomposition read) -------------------------------
            if a.with_delta:
                with tm.stage(f"{subj}/s{seed}/train"):
                    CLS.train_full_ft(model, proc, device, examples, seed=seed)
                dW = {n: (p.detach().float().cpu() - snap[n].float()).clone()
                      for n, p in model.named_parameters() if n in grid}
                assert len(dW) == len(grid), "the natural ΔW does not cover the grid"
                with tm.stage(f"{subj}/s{seed}/delta_census"):
                    d_rows, d_pooled = census_matrices(dW, "natural_delta")
                cell["delta_pooled"] = d_pooled
                cell["delta_rows"] = d_rows
                del dW
                CLS.reset_model(model, snap)
            cell["gradient_rows"] = g_rows
            cells.append(cell)
            _w()

    # ---- the mechanism headline: pooled across cells, per rank ----------------------------
    rep["mechanism"] = {}
    for r in ranks:
        sus = float(np.mean([c["gradient_pooled"][r]["susceptibility"] for c in cells]))
        en = float(np.mean([c["gradient_pooled"][r]["energy"] for c in cells]))
        rnd = float(np.mean([c["gradient_pooled"][r]["random"] for c in cells]))
        fl = float(np.mean([c["gradient_pooled"][r]["chance_floor"] for c in cells]))
        rep["mechanism"][r] = {
            "task_gradient_in_susceptibility": sus, "task_gradient_in_energy": en,
            "task_gradient_in_random": rnd, "chance_floor": fl,
            "contrast_energy_minus_sus": en - sus,
            # the two-sided read: BOTH legs are reported against the floor, so "energy wins" cannot
            # be claimed from energy alone, and a null (both at floor) is equally legible.
            "energy_above_floor": en - fl, "sus_above_floor": sus - fl,
            "mechanism_confirms_unifying": bool((en - sus) > 0 and (en - fl) > 0)}
    # the single boolean adjudicate_cleave consumes, at the smallest rank in the ladder (most
    # discriminating: a floor-level contrast at r=20 cannot hide behind a large subspace).
    r0 = min(ranks)
    rep["mechanism_confirms"] = bool(rep["mechanism"][r0]["mechanism_confirms_unifying"])
    rep["mechanism_rank_used"] = r0
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("CLEAVE_MECHANISM_OK", json.dumps({"mechanism": rep["mechanism"],
                                             "confirms": rep["mechanism_confirms"]}, default=str),
          flush=True)


if __name__ == "__main__":
    main()
